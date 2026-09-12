//! When a scheduled job runs.
//!
//! Two audiences have to agree on the same rule and they want to write it very
//! differently. A person says "every weekday at 9:20" — to ozgent in a chat,
//! or into a box on a page. Anyone who has run a server before says
//! `20 9 * * 1-5` and expects it to work. Both are accepted here and both
//! become the same [`Recur`], so there is one definition of "when" and one
//! place that can be wrong about it.
//!
//! Storage is separate from either. [`Recur::to_string`] writes a canonical
//! form that [`Recur::parse`] reads back exactly, so what is in the database
//! does not depend on how it was typed, and re-saving a job never quietly
//! reinterprets it. [`Recur::describe`] is the third form: what a person is
//! shown, which is never parsed.
//!
//! Times are wall-clock times in a [`Zone`], not offsets from UTC. That is the
//! whole reason this is not a few lines of modular arithmetic: 9:20 means 9:20
//! on the clock in the room, on the days either side of a daylight-saving
//! change as much as any other day.

use crate::datetime::DateTime;
use crate::zone::Zone;

/// How far ahead to look for the next fire before giving up.
///
/// A rule like "29 February" is legal and fires roughly every fourth year, so
/// the horizon has to clear one. Beyond that a rule that matches nothing —
/// `0 0 31 2 *`, the 31st of February — is a mistake to report rather than a
/// loop to run.
const HORIZON_DAYS: i64 = 5 * 366;

/// When a job runs.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Recur {
    /// One time, at a fixed instant. Does not repeat.
    ///
    /// For a time that was *already* an instant when it was written — "in two
    /// hours" is two hours from now wherever you are.
    Once { at: i64 },
    /// One time, at a wall-clock time in the job's zone. Does not repeat.
    ///
    /// Separate from [`Recur::Once`] because "at 2026-09-20 09:00" is nine in
    /// the morning *here*, and the zone is not known when the rule is parsed —
    /// it belongs to the job. Stored as the naive civil time and placed in the
    /// zone at [`Recur::next_after`]. Collapsing the two was the bug this
    /// exists to prevent: a one-off booked for 09:00 fired at 03:30.
    At { wall: i64 },
    /// Every so many seconds, counted from when the job was created.
    ///
    /// For intervals that do not divide an hour. `every 30 minutes` is a
    /// calendar rule so it lands on :00 and :30 like a person expects;
    /// `every 7 minutes` cannot be, and is this.
    Every { seconds: i64 },
    /// On a calendar: minutes, hours, days of the month, months, weekdays.
    Calendar(Calendar),
}

/// The shortest interval a repeating job may have.
///
/// A job runs a model and possibly a web search. Once a minute is already
/// aggressive; anything under it is a mistake that would keep a GPU busy
/// forever, and is refused when the rule is parsed rather than discovered
/// later.
pub const MIN_INTERVAL: i64 = 60;

impl Recur {
    /// Read a rule, however it was written.
    ///
    /// Tries the canonical storage form, then cron, then the phrasings a
    /// person actually types. The error is written to be shown to whoever
    /// typed it, including a model that got it wrong and can try again.
    pub fn parse(input: &str) -> Result<Self, String> {
        let text = input.trim();
        if text.is_empty() {
            return Err("no schedule given".into());
        }
        let lower = text.to_ascii_lowercase();

        // Canonical forms first: they are what comes back out of the database,
        // and they must never be reinterpreted by the friendlier parsers.
        if let Some(rest) = lower.strip_prefix("once ") {
            let at: i64 = rest
                .trim()
                .parse()
                .map_err(|_| format!("{:?} is not an instant", rest.trim()))?;
            return Ok(Self::Once { at });
        }
        if let Some(rest) = lower.strip_prefix("at ") {
            if let Ok(wall) = rest.trim().parse::<i64>() {
                return Ok(Self::At { wall });
            }
        }
        if let Some(rest) = lower.strip_prefix("cron ") {
            return Ok(Self::Calendar(Calendar::parse(rest)?));
        }
        if let Some(rest) = lower.strip_prefix("every ").and_then(|r| r.strip_suffix('s')) {
            if let Ok(seconds) = rest.trim().parse::<i64>() {
                return Self::every(seconds);
            }
        }

        // A bare cron expression: five fields, none of which is a word we use.
        if lower.split_whitespace().count() == 5 && !lower.starts_with("every") {
            if let Ok(cal) = Calendar::parse(&lower) {
                return Ok(Self::Calendar(cal));
            }
        }

        natural(&lower)
    }

    /// A fixed interval, checked against [`MIN_INTERVAL`].
    pub fn every(seconds: i64) -> Result<Self, String> {
        if seconds < MIN_INTERVAL {
            return Err(format!(
                "the shortest repeat is {} seconds; {seconds} would never stop running",
                MIN_INTERVAL
            ));
        }
        Ok(Self::Every { seconds })
    }

    /// Whether this rule can fire more than once.
    pub fn repeats(&self) -> bool {
        !matches!(self, Self::Once { .. } | Self::At { .. })
    }

    /// The first fire strictly after `after`, or `None` if there is not one.
    ///
    /// `anchor` is when the job was created, and is what a fixed interval
    /// counts from — so "every 90 minutes" keeps the same offsets across a
    /// restart instead of resetting to whenever the process came up.
    pub fn next_after(&self, after: i64, anchor: i64, zone: &Zone) -> Option<i64> {
        match self {
            Self::Once { at } => (*at > after).then_some(*at),
            Self::At { wall } => {
                let at = zone.instant_of(&DateTime::from_unix(*wall));
                (at > after).then_some(at)
            }
            Self::Every { seconds } => {
                let seconds = (*seconds).max(MIN_INTERVAL);
                let elapsed = after - anchor;
                if elapsed < 0 {
                    return Some(anchor);
                }
                // Strictly after, so a fire that lands exactly on `after` is
                // the one just delivered rather than the next one.
                Some(anchor + (elapsed / seconds + 1) * seconds)
            }
            Self::Calendar(cal) => cal.next_after(after, zone),
        }
    }

    /// What a person is shown. Never parsed back.
    pub fn describe(&self, zone: &Zone) -> String {
        match self {
            Self::Once { at } => {
                let t = zone.local_at(*at);
                format!(
                    "once, on {} {} {} at {:02}:{:02} {}",
                    t.day,
                    t.month_name(),
                    t.year,
                    t.hour,
                    t.minute,
                    zone.label_at(*at)
                )
            }
            Self::At { wall } => {
                let t = DateTime::from_unix(*wall);
                format!(
                    "once, on {} {} {} at {:02}:{:02} {}",
                    t.day,
                    t.month_name(),
                    t.year,
                    t.hour,
                    t.minute,
                    zone.label_at(zone.instant_of(&t))
                )
            }
            Self::Every { seconds } => format!("every {}", duration(*seconds)),
            Self::Calendar(cal) => cal.describe(zone),
        }
    }
}

impl std::fmt::Display for Recur {
    /// The canonical storage form, which [`Recur::parse`] reads back exactly.
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Once { at } => write!(f, "once {at}"),
            Self::At { wall } => write!(f, "at {wall}"),
            Self::Every { seconds } => write!(f, "every {seconds}s"),
            Self::Calendar(cal) => write!(f, "cron {cal}"),
        }
    }
}

impl std::str::FromStr for Recur {
    type Err = String;
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        Self::parse(s)
    }
}

// ------------------------------------------------------------- calendar

/// One cron field: which values in its range match.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct Field {
    /// Bit `n` set means `low + n` matches.
    bits: u64,
    /// Whether the field is unrestricted — written as a bare `*`.
    ///
    /// Needed beyond the bits because cron's day rule turns on which fields
    /// were *restricted*, not on which values they ended up matching. A step
    /// like `*/15` is restricted despite containing a star, and a field that
    /// happens to list every value is restricted too; only a bare `*` is not.
    /// Deriving it from the text this way is also what lets [`show`] write a
    /// field back and read the same rule again.
    star: bool,
}

impl Field {
    fn matches(&self, value: u32) -> bool {
        self.bits & (1 << value) != 0
    }

    /// Every matching value, ascending.
    fn values(&self, low: u32, high: u32) -> Vec<u32> {
        (low..=high).filter(|v| self.matches(*v)).collect()
    }

    /// Parse one field: `*`, `5`, `1-5`, `*/15`, `1-5/2`, `mon,fri`, or any
    /// comma-separated mixture of those.
    fn parse(text: &str, low: u32, high: u32, names: &[&str]) -> Result<Self, String> {
        let text = text.trim();
        if text.is_empty() {
            return Err("an empty field".into());
        }
        let mut bits = 0u64;
        let mut star = false;

        for part in text.split(',') {
            let part = part.trim();
            let (range, step) = match part.split_once('/') {
                Some((r, s)) => {
                    let step: u32 = s
                        .trim()
                        .parse()
                        .map_err(|_| format!("{s:?} is not a step"))?;
                    if step == 0 {
                        return Err("a step of zero".into());
                    }
                    (r.trim(), step)
                }
                None => (part, 1),
            };

            let (from, to) = if range == "*" {
                star |= step == 1;
                (low, high)
            } else if let Some((a, b)) = range.split_once('-') {
                (value(a, low, high, names)?, value(b, low, high, names)?)
            } else {
                let v = value(range, low, high, names)?;
                // `5/2` means "from 5 to the end, every 2" — cron's own rule.
                if step > 1 { (v, high) } else { (v, v) }
            };
            if from > to {
                return Err(format!("{from} to {to} runs backwards"));
            }
            let mut v = from;
            while v <= to {
                bits |= 1 << v;
                v += step;
            }
        }
        if bits == 0 {
            return Err(format!("{text:?} matches nothing"));
        }
        Ok(Self { bits, star })
    }
}

/// One value in a field: a number, or a three-letter name.
fn value(text: &str, low: u32, high: u32, names: &[&str]) -> Result<u32, String> {
    let text = text.trim();
    if !names.is_empty() {
        let short: String = text.chars().take(3).collect();
        if let Some(i) = names.iter().position(|n| *n == short) {
            return Ok(low + i as u32);
        }
    }
    let n: u32 = text
        .parse()
        .map_err(|_| format!("{text:?} is not a number this field accepts"))?;
    // Cron writes Sunday as either 0 or 7.
    let n = if high == 6 && n == 7 { 0 } else { n };
    if n < low || n > high {
        return Err(format!("{n} is outside {low}-{high}"));
    }
    Ok(n)
}

const DAY_NAMES: [&str; 7] = ["sun", "mon", "tue", "wed", "thu", "fri", "sat"];
const MONTH_NAMES: [&str; 12] = [
    "jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec",
];

/// A cron-style calendar rule, evaluated in a zone's wall-clock time.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Calendar {
    minute: Field,
    hour: Field,
    day: Field,
    month: Field,
    weekday: Field,
}

impl Calendar {
    /// Parse five whitespace-separated fields.
    pub fn parse(text: &str) -> Result<Self, String> {
        let fields: Vec<&str> = text.split_whitespace().collect();
        if fields.len() != 5 {
            return Err(format!(
                "a cron rule has five fields — minute hour day month weekday — but this has {}",
                fields.len()
            ));
        }
        Ok(Self {
            minute: Field::parse(fields[0], 0, 59, &[]).map_err(|e| format!("minute: {e}"))?,
            hour: Field::parse(fields[1], 0, 23, &[]).map_err(|e| format!("hour: {e}"))?,
            day: Field::parse(fields[2], 1, 31, &[]).map_err(|e| format!("day of month: {e}"))?,
            month: Field::parse(fields[3], 1, 12, &MONTH_NAMES).map_err(|e| format!("month: {e}"))?,
            weekday: Field::parse(fields[4], 0, 6, &DAY_NAMES).map_err(|e| format!("weekday: {e}"))?,
        })
    }

    /// Whether a date matches.
    ///
    /// The day rule is cron's, which is a wart worth keeping rather than
    /// fixing: when *both* day-of-month and weekday are restricted, a date
    /// matching *either* runs. So `0 0 1 * mon` is the 1st and every Monday,
    /// not Mondays that fall on the 1st. Every other cron behaves this way and
    /// a rule copied from one of them has to mean the same thing here.
    fn matches(&self, t: &DateTime) -> bool {
        if !self.month.matches(t.month) {
            return false;
        }
        let day = self.day.matches(t.day);
        let weekday = self.weekday.matches(t.weekday);
        let day_ok = match (self.day.star, self.weekday.star) {
            (true, true) => true,
            (false, true) => day,
            (true, false) => weekday,
            (false, false) => day || weekday,
        };
        day_ok
    }

    fn next_after(&self, after: i64, zone: &Zone) -> Option<i64> {
        let hours = self.hour.values(0, 23);
        let minutes = self.minute.values(0, 59);
        if hours.is_empty() || minutes.is_empty() {
            return None;
        }

        // Walk forward a day at a time in wall-clock terms. Starting a day
        // early costs one extra iteration and covers the zones where "now" in
        // local time is already tomorrow or still yesterday.
        let start = zone.local_at(after);
        let mut day = crate::datetime::days_from_civil(start.year, start.month, start.day) - 1;

        for _ in 0..HORIZON_DAYS {
            let midnight = DateTime::from_unix(day * 86_400);
            day += 1;
            if !self.matches(&midnight) {
                continue;
            }
            for hour in &hours {
                for minute in &minutes {
                    let wall = DateTime::civil(
                        midnight.year,
                        midnight.month,
                        midnight.day,
                        *hour,
                        *minute,
                        0,
                    );
                    let instant = zone.instant_of(&wall);
                    if instant > after {
                        return Some(instant);
                    }
                }
            }
        }
        None
    }

    fn describe(&self, zone: &Zone) -> String {
        let hours = self.hour.values(0, 23);
        let minutes = self.minute.values(0, 59);
        let label = zone.label_at(DateTime::now().to_unix());

        // "every minute" and "every N minutes" read badly as a list of times.
        if self.hour.star && minutes.len() > 1 {
            let step = minutes.windows(2).map(|w| w[1] - w[0]).min().unwrap_or(1);
            let even = minutes.windows(2).all(|w| w[1] - w[0] == step);
            if even && minutes[0] == 0 {
                // "every 30 minutes every day" is worse than "every 30
                // minutes": an interval that runs on every day does not need
                // to say so, and only a restricted day rule is worth naming.
                let days = self.on_days(true);
                let days = if days.trim() == "every day" { "" } else { &days };
                return format!("every {step} minutes{days}");
            }
        }
        if self.hour.star && self.minute.star {
            let days = self.on_days(true);
            let days = if days.trim() == "every day" { "" } else { &days };
            return format!("every minute{days}");
        }
        // "every 2 hours" listed as six clock times reads as six separate
        // jobs. Same shape as the minutes case above, one field along.
        if minutes.len() == 1 && minutes[0] == 0 && hours.len() > 1 {
            // `*` is the same as a step of one; both mean "on every hour".
            let step = if self.hour.star {
                1
            } else {
                hours.windows(2).map(|w| w[1] - w[0]).min().unwrap_or(1)
            };
            let even = hours.windows(2).all(|w| w[1] - w[0] == step);
            if even && hours[0] == 0 && 24 % step == 0 && hours.len() as u32 == 24 / step {
                let days = self.on_days(true);
                let days = if days.trim() == "every day" { "" } else { &days };
                return match step {
                    1 => format!("every hour{days}"),
                    n => format!("every {n} hours{days}"),
                };
            }
        }

        let times: Vec<String> = hours
            .iter()
            .flat_map(|h| minutes.iter().map(move |m| format!("{h:02}:{m:02}")))
            .take(6)
            .collect();
        let when = if times.is_empty() {
            String::new()
        } else {
            format!(" at {} {label}", times.join(", "))
        };
        format!("{}{when}", self.on_days(false).trim_start_matches(' '))
    }

    /// The day part of a description: "every weekday", "on the 1st", …
    fn on_days(&self, prefixed: bool) -> String {
        let lead = if prefixed { " " } else { "" };
        let weekdays = self.weekday.values(0, 6);
        let days = self.day.values(1, 31);
        let months = self.month.values(1, 12);

        let day_part = match (self.day.star, self.weekday.star) {
            (true, true) => "every day".to_string(),
            (true, false) => match weekdays.as_slice() {
                [1, 2, 3, 4, 5] => "every weekday".to_string(),
                [0, 6] => "every weekend".to_string(),
                _ => format!(
                    "every {}",
                    weekdays
                        .iter()
                        .map(|d| full_day(*d))
                        .collect::<Vec<_>>()
                        .join(", ")
                ),
            },
            _ => format!(
                "on the {}",
                days.iter().map(|d| ordinal(*d)).collect::<Vec<_>>().join(", ")
            ),
        };
        let month_part = if self.month.star {
            String::new()
        } else {
            format!(
                " in {}",
                months
                    .iter()
                    .map(|m| full_month(*m))
                    .collect::<Vec<_>>()
                    .join(", ")
            )
        };
        format!("{lead}{day_part}{month_part}")
    }
}

impl std::fmt::Display for Calendar {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "{} {} {} {} {}",
            show(&self.minute, 0, 59),
            show(&self.hour, 0, 23),
            show(&self.day, 1, 31),
            show(&self.month, 1, 12),
            show(&self.weekday, 0, 6)
        )
    }
}

/// A field back as cron text. Written as a plain list rather than trying to
/// recover ranges: it round-trips exactly, which is what storage needs.
fn show(field: &Field, low: u32, high: u32) -> String {
    if field.star {
        return "*".into();
    }
    field
        .values(low, high)
        .iter()
        .map(u32::to_string)
        .collect::<Vec<_>>()
        .join(",")
}

// ------------------------------------------------------------- phrasing

/// Split a trailing time zone off a phrase, if it names one.
///
/// "every day at 4pm UTC" has to mean UTC, and "every day at 4pm" has to mean
/// the clock in the room. Without this the first is a parse error, and a user
/// who writes it gets told their time is not a time.
///
/// Only unambiguous names are accepted: `UTC` and `GMT`, and full IANA names
/// like `Asia/Kolkata`. Three-letter abbreviations are deliberately refused —
/// `IST` is India, Ireland *and* Israel, and silently picking one would put a
/// job hours out with nothing to show for it. The error says so.
pub fn split_zone(text: &str) -> Result<(String, Option<String>), String> {
    let trimmed = text.trim();
    let Some((head, last)) = trimmed.rsplit_once(char::is_whitespace) else {
        return Ok((trimmed.to_string(), None));
    };
    let last = last.trim();
    if last.is_empty() {
        return Ok((trimmed.to_string(), None));
    }

    if last.eq_ignore_ascii_case("utc") || last.eq_ignore_ascii_case("gmt") {
        return Ok((head.trim().to_string(), Some("UTC".into())));
    }
    // An IANA name always has a slash, which nothing else in a schedule does.
    if last.contains('/') {
        return match Zone::named(last) {
            Ok(_) => Ok((head.trim().to_string(), Some(last.to_string()))),
            Err(e) => Err(e),
        };
    }
    // A bare three- or four-letter word after a time is almost certainly meant
    // as a zone. Saying why it is refused beats "that is not a time".
    let looks_like_a_zone = (3..=4).contains(&last.len())
        && last.chars().all(|c| c.is_ascii_alphabetic())
        && head.chars().any(|c| c.is_ascii_digit());
    if looks_like_a_zone && !last.eq_ignore_ascii_case("and") {
        return Err(format!(
            "{last:?} could be several different zones — {} is India, Ireland and Israel, \
             and there are three different ESTs. Write it in full, like \"Asia/Kolkata\", \
             or leave it out for this machine's own time.",
            last.to_uppercase()
        ));
    }
    Ok((trimmed.to_string(), None))
}

/// The forms a person types.
fn natural(text: &str) -> Result<Recur, String> {
    let text = text.replace(" on the ", " ").replace("  ", " ");
    let text = text.trim();

    // "in 20 minutes", "in 2 hours" — a one-off, relative to now.
    if let Some(rest) = text.strip_prefix("in ") {
        let seconds = interval(rest)?;
        return Ok(Recur::Once { at: DateTime::now().to_unix() + seconds });
    }

    // "at 2026-09-20 09:00" / "on 2026-09-20 at 09:00" — a one-off on the
    // calendar. Kept as a wall-clock time, not an instant: nine in the morning
    // means nine in the morning *here*, and the zone belongs to the job.
    if let Some(wall) = absolute(text) {
        return Ok(Recur::At { wall });
    }

    let (head, time) = match text.split_once(" at ") {
        Some((h, t)) => (h.trim(), Some(t.trim())),
        None => (text, None),
    };
    let times = match time {
        Some(t) => clock_times(t)?,
        None => Vec::new(),
    };

    let head = head.strip_prefix("every ").unwrap_or(head).trim();
    let head = match head {
        "" => "day",
        "hourly" => "hour",
        "daily" => "day",
        "weekly" => "week",
        "monthly" => "month",
        other => other,
    };

    // "every 30 minutes" and friends: no time of day, an interval instead.
    if head.chars().next().is_some_and(|c| c.is_ascii_digit()) && times.is_empty() {
        let seconds = interval(head)?;
        // Anything dividing an hour becomes a calendar rule so it lands on
        // round numbers — :00, :15, :30, :45 — which is what a person means.
        if seconds % 60 == 0 && seconds < 3600 && 3600 % seconds == 0 {
            let step = seconds / 60;
            return Ok(Recur::Calendar(Calendar::parse(&format!("*/{step} * * * *"))?));
        }
        if seconds % 3600 == 0 && seconds <= 12 * 3600 && 24 % (seconds / 3600) == 0 {
            let step = seconds / 3600;
            return Ok(Recur::Calendar(Calendar::parse(&format!("0 */{step} * * *"))?));
        }
        return Recur::every(seconds);
    }

    let (minute_field, hour_field) = if times.is_empty() {
        // "every day" with no time means midnight; "every hour" means on the
        // hour. Neither is a useful default to guess differently.
        match head {
            "hour" => ("0".to_string(), "*".to_string()),
            "minute" => ("*".to_string(), "*".to_string()),
            _ => ("0".to_string(), "0".to_string()),
        }
    } else {
        let minutes: Vec<String> = times.iter().map(|(_, m)| m.to_string()).collect();
        let hours: Vec<String> = times.iter().map(|(h, _)| h.to_string()).collect();
        // Several times only compose as a cron product, so 9:20 and 18:45
        // would also match 9:45 and 18:20. Accept the product only when the
        // minutes agree; otherwise say so rather than schedule the wrong thing.
        let same_minute = times.iter().all(|(_, m)| *m == times[0].1);
        if times.len() > 1 && !same_minute {
            return Err(
                "several times a day have to share the same minute — 9:20 and 18:20 work, \
                 9:20 and 18:45 do not. Use two jobs for those."
                    .into(),
            );
        }
        (minutes[0].clone(), dedup(hours).join(","))
    };

    let (day, month, weekday) = day_fields(head)?;
    Ok(Recur::Calendar(Calendar::parse(&format!(
        "{minute_field} {hour_field} {day} {month} {weekday}"
    ))?))
}

fn dedup(mut v: Vec<String>) -> Vec<String> {
    v.dedup();
    v
}

/// The day part of a phrase, as cron's three day fields.
fn day_fields(head: &str) -> Result<(String, String, String), String> {
    if matches!(head, "day" | "hour" | "minute" | "" | "week") {
        // "every week" with no day named is every Monday, which is the only
        // reading that does not silently depend on when the job was made.
        return Ok(match head {
            "week" => ("*".into(), "*".into(), "1".into()),
            _ => ("*".into(), "*".into(), "*".into()),
        });
    }
    if head == "weekday" || head == "weekdays" {
        return Ok(("*".into(), "*".into(), "1-5".into()));
    }
    if head == "weekend" || head == "weekends" {
        return Ok(("*".into(), "*".into(), "0,6".into()));
    }
    if head == "month" {
        return Ok(("1".into(), "*".into(), "*".into()));
    }
    // "month 15th", "15th", "1st"
    if let Some(day) = head.strip_prefix("month ").or(Some(head)) {
        let digits: String = day.chars().take_while(|c| c.is_ascii_digit()).collect();
        if !digits.is_empty() && digits.len() == day.trim_end_matches(['s', 't', 'n', 'd', 'r', 'h']).len()
        {
            let n: u32 = digits.parse().map_err(|_| format!("{day:?} is not a day"))?;
            if !(1..=31).contains(&n) {
                return Err(format!("there is no {n}th of the month"));
            }
            return Ok((n.to_string(), "*".into(), "*".into()));
        }
    }

    // A list of weekday names: "monday", "mon,fri", "monday and friday".
    let names: Vec<&str> = head
        .split(|c| c == ',' || c == '/')
        .flat_map(|p| p.split(" and "))
        .map(str::trim)
        .filter(|p| !p.is_empty())
        .collect();
    if names.is_empty() {
        return Err(format!("{head:?} is not a day I recognise"));
    }
    let mut days = Vec::new();
    for name in names {
        let short: String = name.trim_end_matches('s').chars().take(3).collect();
        match DAY_NAMES.iter().position(|d| *d == short) {
            Some(i) => days.push(i.to_string()),
            None => {
                return Err(format!(
                    "{name:?} is not a day. Try a weekday name, \"every day\", \
                     \"every weekday\", or a cron rule like \"20 9 * * 1-5\"."
                ));
            }
        }
    }
    Ok(("*".into(), "*".into(), days.join(",")))
}

/// `2026-09-20 09:00` or `2026-09-20T09:00`, as a naive wall-clock instant.
fn absolute(text: &str) -> Option<i64> {
    let text = text.strip_prefix("at ").or(text.strip_prefix("on ")).unwrap_or(text);
    let text = text.replace('T', " ");
    let (date, rest) = text.trim().split_once(' ').unwrap_or((text.trim(), "00:00"));
    let rest = rest.trim().strip_prefix("at ").unwrap_or(rest.trim());
    let mut parts = date.split('-');
    let year: i64 = parts.next()?.parse().ok()?;
    let month: u32 = parts.next()?.parse().ok()?;
    let day: u32 = parts.next()?.parse().ok()?;
    if parts.next().is_some() || !(1..=12).contains(&month) || !(1..=31).contains(&day) {
        return None;
    }
    let (hour, minute) = clock_times(rest).ok()?.first().copied()?;
    Some(DateTime::civil(year, month, day, hour, minute, 0).to_unix())
}

/// `9:20`, `09:20`, `9am`, `6:30 pm`, or several separated by commas.
fn clock_times(text: &str) -> Result<Vec<(u32, u32)>, String> {
    let mut out = Vec::new();
    for part in text.split(',').flat_map(|p| p.split(" and ")) {
        let part = part.trim();
        if part.is_empty() {
            continue;
        }
        let (body, shift) = if let Some(b) = part.strip_suffix("pm") {
            (b.trim(), 12)
        } else if let Some(b) = part.strip_suffix("am") {
            (b.trim(), 0)
        } else {
            (part, -1)
        };
        let (h, m) = match body.split_once(':') {
            Some((h, m)) => (h.trim(), m.trim()),
            None => (body, "0"),
        };
        let mut hour: u32 = h
            .parse()
            .map_err(|_| format!("{part:?} is not a time — try 9:20 or 09:20 or 6pm"))?;
        let minute: u32 = m
            .parse()
            .map_err(|_| format!("{part:?} is not a time — try 9:20 or 09:20 or 6pm"))?;
        match shift {
            12 if hour < 12 => hour += 12,
            0 if hour == 12 => hour = 0,
            _ => {}
        }
        if hour > 23 || minute > 59 {
            return Err(format!("there is no such time as {part:?}"));
        }
        out.push((hour, minute));
    }
    if out.is_empty() {
        return Err("no time of day given".into());
    }
    out.sort_unstable();
    out.dedup();
    Ok(out)
}

/// `30 minutes`, `2 hours`, `90m`, `1d`.
fn interval(text: &str) -> Result<i64, String> {
    let text = text.trim();
    let digits: String = text.chars().take_while(|c| c.is_ascii_digit()).collect();
    let count: i64 = if digits.is_empty() { 1 } else { digits.parse().unwrap_or(1) };
    let unit = text[digits.len()..].trim();
    let unit = unit.strip_suffix('s').unwrap_or(unit);
    let seconds = match unit {
        "second" | "sec" | "s" => 1,
        "minute" | "min" | "m" | "" => 60,
        "hour" | "hr" | "h" => 3600,
        "day" | "d" => 86_400,
        "week" | "w" => 7 * 86_400,
        other => {
            return Err(format!(
                "{other:?} is not a unit of time — try minutes, hours or days"
            ));
        }
    };
    Ok(count * seconds)
}

fn duration(seconds: i64) -> String {
    let plural = |n: i64, unit: &str| {
        if n == 1 { format!("{unit}") } else { format!("{n} {unit}s") }
    };
    match seconds {
        s if s % 86_400 == 0 => plural(s / 86_400, "day"),
        s if s % 3600 == 0 => plural(s / 3600, "hour"),
        s if s % 60 == 0 => plural(s / 60, "minute"),
        s => plural(s, "second"),
    }
}

fn full_day(d: u32) -> &'static str {
    [
        "Sunday", "Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday",
    ][(d as usize).min(6)]
}

fn full_month(m: u32) -> &'static str {
    [
        "January", "February", "March", "April", "May", "June", "July", "August",
        "September", "October", "November", "December",
    ][(m as usize).saturating_sub(1).min(11)]
}

fn ordinal(n: u32) -> String {
    let suffix = match (n % 10, n % 100) {
        (_, 11..=13) => "th",
        (1, _) => "st",
        (2, _) => "nd",
        (3, _) => "rd",
        _ => "th",
    };
    format!("{n}{suffix}")
}

#[cfg(test)]
mod tests {
    use super::*;

    /// +5:30, so every test crosses a non-zero and non-whole-hour offset —
    /// the case where a UTC-shaped bug is visible.
    fn ist() -> Zone {
        Zone::fixed("IST", 5 * 3600 + 1800)
    }

    fn at(zone: &Zone, y: i64, mo: u32, d: u32, h: u32, mi: u32) -> i64 {
        zone.instant_of(&DateTime::civil(y, mo, d, h, mi, 0))
    }

    fn local(zone: &Zone, instant: i64) -> (i64, u32, u32, u32, u32) {
        let t = zone.local_at(instant);
        (t.year, t.month, t.day, t.hour, t.minute)
    }

    // ------------------------------------------------------- parsing

    #[test]
    fn the_phrase_from_the_feature_request_works() {
        // "every weekday at 9:20" is the example this was built for.
        let r = Recur::parse("every weekday at 9:20").unwrap();
        assert_eq!(r.to_string(), "cron 20 9 * * 1,2,3,4,5");
        let zone = ist();
        // 2026-09-12 is a Saturday, so the next one is Monday the 14th.
        let next = r.next_after(at(&zone, 2026, 9, 12, 12, 0), 0, &zone).unwrap();
        assert_eq!(local(&zone, next), (2026, 9, 14, 9, 20));
    }

    #[test]
    fn cron_is_accepted_as_typed() {
        let r = Recur::parse("20 9 * * 1-5").unwrap();
        assert_eq!(r, Recur::parse("every weekday at 9:20").unwrap());
    }

    #[test]
    fn the_stored_form_reads_back_as_the_same_rule() {
        // What is in the database must not depend on how it was typed, and
        // re-saving a job must never reinterpret it.
        for text in [
            "every weekday at 9:20",
            "every day at 06:00",
            "every monday, friday at 18:00",
            "every 15 minutes",
            "every 2 hours",
            "every 7 minutes",
            "0 0 1 * *",
            "30 8 * * mon",
            "*/5 * * * *",
            "every weekend at 10:30",
            "every 1st at 9:00",
        ] {
            let once = Recur::parse(text).unwrap_or_else(|e| panic!("{text:?}: {e}"));
            let twice = Recur::parse(&once.to_string())
                .unwrap_or_else(|e| panic!("{text:?} -> {once}: {e}"));
            assert_eq!(once, twice, "{text:?} did not survive a round trip");
            assert_eq!(once.to_string(), twice.to_string(), "for {text:?}");
        }
    }

    #[test]
    fn the_phrasings_a_person_actually_types_are_understood() {
        let cases: [(&str, &str); 12] = [
            ("every day at 9:20", "cron 20 9 * * *"),
            ("daily at 9am", "cron 0 9 * * *"),
            ("every weekday at 09:20", "cron 20 9 * * 1,2,3,4,5"),
            ("every weekend at 10:00", "cron 0 10 * * 0,6"),
            ("every monday at 6pm", "cron 0 18 * * 1"),
            ("every mon,fri at 18:00", "cron 0 18 * * 1,5"),
            ("every monday and friday at 18:00", "cron 0 18 * * 1,5"),
            ("every hour", "cron 0 * * * *"),
            ("hourly", "cron 0 * * * *"),
            ("every 30 minutes", "cron 0,30 * * * *"),
            ("every 15m", "cron 0,15,30,45 * * * *"),
            ("every month at 9:00", "cron 0 9 1 * *"),
        ];
        for (text, expected) in cases {
            let r = Recur::parse(text).unwrap_or_else(|e| panic!("{text:?}: {e}"));
            assert_eq!(r.to_string(), expected, "for {text:?}");
        }
    }

    #[test]
    fn twelve_hour_times_land_on_the_right_half_of_the_day() {
        assert_eq!(clock_times("12am").unwrap(), [(0, 0)]);
        assert_eq!(clock_times("12pm").unwrap(), [(12, 0)]);
        assert_eq!(clock_times("9am").unwrap(), [(9, 0)]);
        assert_eq!(clock_times("9pm").unwrap(), [(21, 0)]);
        assert_eq!(clock_times("12:30am").unwrap(), [(0, 30)]);
    }

    #[test]
    fn an_interval_that_does_not_divide_an_hour_stays_an_interval() {
        // Every 7 minutes cannot be a cron rule without drifting, so it is
        // counted from the job's own anchor instead.
        assert_eq!(Recur::parse("every 7 minutes").unwrap(), Recur::Every { seconds: 420 });
        assert_eq!(Recur::parse("every 90 minutes").unwrap(), Recur::Every { seconds: 5400 });
    }

    #[test]
    fn an_interval_that_does_divide_an_hour_lands_on_round_numbers() {
        // "every 30 minutes" means :00 and :30, not 30 minutes after whenever
        // the job happened to be created.
        let zone = ist();
        let r = Recur::parse("every 30 minutes").unwrap();
        let next = r.next_after(at(&zone, 2026, 9, 12, 9, 7), 0, &zone).unwrap();
        assert_eq!(local(&zone, next), (2026, 9, 12, 9, 30));
    }

    #[test]
    fn a_schedule_that_would_never_stop_running_is_refused() {
        assert!(Recur::every(1).is_err());
        assert!(Recur::every(59).is_err());
        assert!(Recur::every(60).is_ok());
        assert!(Recur::parse("every 1 second").is_err());
    }

    #[test]
    fn nonsense_is_an_error_that_says_what_to_type_instead() {
        for text in ["", "   ", "every fortnight at 9:00", "every day at 25:00", "0 0 1 *"] {
            let err = Recur::parse(text).unwrap_err();
            assert!(!err.is_empty(), "for {text:?}");
        }
        let err = Recur::parse("every blursday at 9:00").unwrap_err();
        assert!(err.contains("cron") || err.contains("weekday"), "unhelpful: {err}");
    }

    #[test]
    fn two_times_with_different_minutes_are_refused_rather_than_multiplied() {
        // A cron rule is a product: 9:20 and 18:45 would also fire at 9:45 and
        // 18:20. Firing four times when two were asked for is worse than an
        // error that says to use two jobs.
        let err = Recur::parse("every day at 9:20, 18:45").unwrap_err();
        assert!(err.contains("two jobs"), "{err}");
        // The same minute is fine, and is the common case.
        let r = Recur::parse("every day at 9:20, 18:20").unwrap();
        assert_eq!(r.to_string(), "cron 20 9,18 * * *");
    }

    // ------------------------------------------------------- firing

    #[test]
    fn a_daily_job_fires_once_a_day_at_the_wall_clock_time() {
        let zone = ist();
        let r = Recur::parse("every day at 9:20").unwrap();
        let mut t = at(&zone, 2026, 9, 12, 0, 0);
        for day in 12..=15 {
            t = r.next_after(t, 0, &zone).unwrap();
            assert_eq!(local(&zone, t), (2026, 9, day, 9, 20));
        }
    }

    #[test]
    fn a_weekday_job_skips_the_weekend() {
        let zone = ist();
        let r = Recur::parse("every weekday at 9:20").unwrap();
        // Friday the 11th, after it has fired.
        let mut t = at(&zone, 2026, 9, 11, 10, 0);
        t = r.next_after(t, 0, &zone).unwrap();
        assert_eq!(local(&zone, t), (2026, 9, 14, 9, 20), "Monday, not Saturday");
    }

    #[test]
    fn the_next_fire_is_strictly_after_the_last_one() {
        // Otherwise a job that has just run schedules itself for the instant
        // it already ran at, and runs forever in a tight loop.
        let zone = ist();
        for text in ["every day at 9:20", "every 30 minutes", "every 7 minutes"] {
            let r = Recur::parse(text).unwrap();
            let fired = r.next_after(at(&zone, 2026, 9, 12, 0, 0), 0, &zone).unwrap();
            let again = r.next_after(fired, 0, &zone).unwrap();
            assert!(again > fired, "{text} repeated {fired}");
        }
    }

    #[test]
    fn a_one_off_fires_once_and_then_never_again() {
        let zone = ist();
        let when = at(&zone, 2026, 9, 20, 9, 0);
        let r = Recur::Once { at: when };
        assert!(!r.repeats());
        assert_eq!(r.next_after(when - 1, 0, &zone), Some(when));
        assert_eq!(r.next_after(when, 0, &zone), None, "already fired");
        assert_eq!(r.next_after(when + 86_400, 0, &zone), None);
    }

    #[test]
    fn an_interval_counts_from_the_job_rather_than_from_now() {
        // A restart must not shift every interval job to the restart time.
        let zone = Zone::utc();
        let anchor = 1_000_000;
        let r = Recur::Every { seconds: 3600 };
        assert_eq!(r.next_after(anchor + 10, anchor, &zone), Some(anchor + 3600));
        assert_eq!(r.next_after(anchor + 3600, anchor, &zone), Some(anchor + 7200));
        // Far in the future, it is still on the original phase.
        let far = r.next_after(anchor + 500_000, anchor, &zone).unwrap();
        assert_eq!((far - anchor) % 3600, 0);
    }

    #[test]
    fn a_rule_that_can_never_match_says_so_instead_of_looping() {
        // The 31st of February.
        let r = Recur::parse("0 0 31 2 *").unwrap();
        assert_eq!(r.next_after(0, 0, &Zone::utc()), None);
    }

    #[test]
    fn a_rule_that_matches_rarely_is_still_found() {
        // 29 February exists roughly every fourth year, inside the horizon.
        let r = Recur::parse("0 9 29 2 *").unwrap();
        let zone = Zone::utc();
        let next = r.next_after(at(&zone, 2026, 9, 12, 0, 0), 0, &zone).unwrap();
        assert_eq!(local(&zone, next), (2028, 2, 29, 9, 0));
    }

    #[test]
    fn cron_matches_either_day_rule_when_both_are_set() {
        // Vixie cron's behaviour, which every rule copied from elsewhere
        // assumes: the 1st *or* any Monday, not Mondays falling on the 1st.
        let zone = Zone::utc();
        let r = Recur::parse("0 0 1 * mon").unwrap();
        // 2026-09-12 is a Saturday; Monday the 14th comes before October 1st.
        let next = r.next_after(at(&zone, 2026, 9, 12, 0, 0), 0, &zone).unwrap();
        assert_eq!(local(&zone, next), (2026, 9, 14, 0, 0));
    }

    #[test]
    fn a_job_keeps_its_wall_clock_time_across_a_daylight_saving_change() {
        // The thing a UTC-only scheduler gets wrong: after the clocks change,
        // a 9:20 brief arrives at 8:20 or 10:20 forever.
        if !std::path::Path::new("/usr/share/zoneinfo/Europe/London").is_file() {
            return;
        }
        let zone = Zone::named("Europe/London").unwrap();
        let r = Recur::parse("every day at 9:20").unwrap();
        // Across the spring change on 29 March 2026.
        let mut t = at(&zone, 2026, 3, 27, 0, 0);
        for (day, month) in [(27, 3), (28, 3), (29, 3), (30, 3), (31, 3)] {
            t = r.next_after(t, 0, &zone).unwrap();
            let wall = zone.local_at(t);
            assert_eq!((wall.month, wall.day), (month, day));
            assert_eq!((wall.hour, wall.minute), (9, 20), "on {day}/{month}");
        }
    }

    #[test]
    fn a_days_worth_of_fires_is_complete_and_in_order() {
        // Walks a whole day of a fifteen-minute rule: catches an off-by-one
        // that skips or repeats one slot per hour.
        let zone = ist();
        let r = Recur::parse("every 15 minutes").unwrap();
        let start = at(&zone, 2026, 9, 12, 0, 0);
        let mut t = start;
        let mut seen = Vec::new();
        while seen.len() < 96 {
            t = r.next_after(t, 0, &zone).unwrap();
            let w = zone.local_at(t);
            seen.push((w.hour, w.minute));
        }
        assert_eq!(seen[0], (0, 15));
        assert_eq!(seen[95], (0, 0), "wrapped into the next day");
        assert!(seen.windows(2).all(|w| w[0] != w[1]), "a slot repeated");
    }

    // ------------------------------------------------------- describing

    #[test]
    fn a_rule_describes_itself_in_words_a_person_would_use() {
        let zone = ist();
        for (text, expected) in [
            ("every weekday at 9:20", "every weekday at 09:20 UTC+5:30"),
            ("every day at 6:00", "every day at 06:00 UTC+5:30"),
            ("every monday at 18:00", "every Monday at 18:00 UTC+5:30"),
            ("every weekend at 10:00", "every weekend at 10:00 UTC+5:30"),
            ("every 30 minutes", "every 30 minutes"),
            ("0,30 * * * 1-5", "every 30 minutes every weekday"),
            ("every 2 hours", "every 2 hours"),
            ("every 4 hours", "every 4 hours"),
            ("0 */6 * * 1-5", "every 6 hours every weekday"),
            ("every hour", "every hour"),
            ("every 7 minutes", "every 7 minutes"),
        ] {
            let got = Recur::parse(text).unwrap().describe(&zone);
            assert_eq!(got, expected, "for {text:?}");
        }
    }

    #[test]
    fn a_calendar_one_off_fires_at_that_time_on_the_local_clock() {
        // The bug this guards: "at 2026-09-20 09:00" stored as an instant
        // rather than a wall-clock time fires at 03:30 in a +5:30 zone.
        let zone = ist();
        let r = Recur::parse("at 2026-09-20 09:00").unwrap();
        assert!(matches!(r, Recur::At { .. }), "{r:?}");
        let fires = r.next_after(at(&zone, 2026, 9, 1, 0, 0), 0, &zone).unwrap();
        assert_eq!(local(&zone, fires), (2026, 9, 20, 9, 0));

        // And it is the same time in a different zone, not the same instant.
        let utc = Zone::utc();
        let there = r.next_after(0, 0, &utc).unwrap();
        assert_eq!(local(&utc, there), (2026, 9, 20, 9, 0));
        assert_ne!(there, fires, "a wall-clock time is not one instant");
    }

    #[test]
    fn a_relative_one_off_is_an_instant_and_does_not_move_with_the_zone() {
        // "in two hours" is two hours from now wherever you are, which is the
        // opposite of the case above — hence two variants.
        let r = Recur::parse("in 2 hours").unwrap();
        let Recur::Once { at } = r else { panic!("{r:?}") };
        assert!(at > DateTime::now().to_unix() + 7000);
        assert_eq!(
            r.next_after(0, 0, &ist()),
            r.next_after(0, 0, &Zone::utc()),
            "the same instant in every zone"
        );
    }

    #[test]
    fn both_kinds_of_one_off_fire_once_and_survive_storage() {
        let zone = ist();
        for text in ["at 2026-09-20 09:00", "in 2 hours"] {
            let r = Recur::parse(text).unwrap();
            assert!(!r.repeats(), "{text} must not repeat");
            assert_eq!(Recur::parse(&r.to_string()).unwrap(), r, "{text} did not round-trip");
            let fired = r.next_after(0, 0, &zone).expect("it has a future");
            assert_eq!(r.next_after(fired, 0, &zone), None, "{text} fired twice");
        }
    }

    #[test]
    fn a_one_off_describes_the_day_it_lands_on() {
        let zone = ist();
        let r = Recur::Once { at: at(&zone, 2026, 9, 20, 9, 0) };
        let text = r.describe(&zone);
        assert!(text.contains("20 September 2026"), "{text}");
        assert!(text.contains("09:00"), "{text}");
    }

    #[test]
    fn every_parsed_rule_describes_itself_without_panicking() {
        // describe() indexes name tables; a field out of range would panic in
        // a page render rather than fail a parse.
        let zone = ist();
        for text in [
            "every day at 9:20",
            "0 0 29 2 *",
            "*/5 * * * *",
            "* * * * *",
            "0 0 1 1 *",
            "0 9 1,15 * *",
            "30 8 * * 0",
            "every 90 minutes",
        ] {
            let r = Recur::parse(text).unwrap_or_else(|e| panic!("{text}: {e}"));
            assert!(!r.describe(&zone).is_empty(), "for {text:?}");
        }
    }

    #[test]
    fn ordinals_read_correctly() {
        let cases = [(1, "1st"), (2, "2nd"), (3, "3rd"), (4, "4th"), (11, "11th"),
                     (12, "12th"), (13, "13th"), (21, "21st"), (22, "22nd"), (31, "31st")];
        for (n, expected) in cases {
            assert_eq!(ordinal(n), expected);
        }
    }
}

#[cfg(test)]
mod zones_in_phrases {
    use super::*;

    /// The user is in India. An unqualified time means their clock, always.
    #[test]
    fn a_time_with_no_zone_names_no_zone() {
        for text in [
            "every day at 4pm",
            "every weekday at 9:20",
            "every 30 minutes",
            "20 9 * * 1-5",
            "in 2 hours",
            "every monday and friday at 18:00",
        ] {
            let (rest, zone) = split_zone(text).unwrap_or_else(|e| panic!("{text:?}: {e}"));
            assert_eq!(zone, None, "{text:?} must not pick up a zone");
            assert_eq!(rest, text, "{text:?} must come back unchanged");
        }
    }

    #[test]
    fn utc_and_gmt_are_taken_at_their_word() {
        for text in ["every day at 4pm UTC", "every day at 4pm utc", "every day at 16:00 GMT"] {
            let (rest, zone) = split_zone(text).unwrap();
            assert_eq!(zone.as_deref(), Some("UTC"), "{text:?}");
            assert!(!rest.to_lowercase().contains("utc"), "{rest:?} still has the zone in it");
            assert!(Recur::parse(&rest).is_ok(), "{rest:?} must still parse as a time");
        }
    }

    #[test]
    fn a_full_iana_name_is_accepted() {
        if !std::path::Path::new("/usr/share/zoneinfo/Asia/Kolkata").is_file() {
            return;
        }
        let (rest, zone) = split_zone("every day at 4pm Asia/Kolkata").unwrap();
        assert_eq!(zone.as_deref(), Some("Asia/Kolkata"));
        assert_eq!(rest, "every day at 4pm");
    }

    #[test]
    fn an_ambiguous_abbreviation_is_refused_rather_than_guessed() {
        // IST is India, Ireland and Israel. Picking one silently would put a
        // job hours out with nothing on screen to explain it.
        let err = split_zone("every day at 4pm IST").unwrap_err();
        assert!(err.contains("Asia/Kolkata"), "must say what to write instead: {err}");
        for bad in ["every day at 4pm EST", "every day at 9am PST", "every day at 4pm CET"] {
            assert!(split_zone(bad).is_err(), "{bad:?} was accepted");
        }
    }

    #[test]
    fn a_zone_that_does_not_exist_is_an_error_not_a_silent_local() {
        assert!(split_zone("every day at 4pm Mars/Olympus").is_err());
    }

    #[test]
    fn ordinary_words_at_the_end_are_not_mistaken_for_zones() {
        // The check only fires after something with a digit in it, and these
        // have to keep working.
        for text in ["every monday and friday at 18:00", "daily", "every hour"] {
            assert!(split_zone(text).is_ok(), "{text:?} was refused");
            assert_eq!(split_zone(text).unwrap().1, None, "{text:?}");
        }
    }

    #[test]
    fn a_phrase_that_names_a_zone_resolves_to_that_zone_not_this_machine() {
        // The whole point: 4pm UTC is 21:30 on an Indian clock, and a job
        // written that way must fire then.
        let (rest, zone) = split_zone("every day at 4pm UTC").unwrap();
        let recur = Recur::parse(&rest).unwrap();
        let utc = Zone::named(&zone.unwrap()).unwrap();
        let fires = recur.next_after(0, 0, &utc).unwrap();
        assert_eq!(DateTime::from_unix(fires).hour, 16, "16:00 UTC");

        let ist = Zone::fixed("IST", 5 * 3600 + 1800);
        let there = ist.local_at(fires);
        assert_eq!((there.hour, there.minute), (21, 30), "which is 21:30 in India");
    }
}
