//! When a scheduled trigger should next fire.
//!
//! Not cron. Cron is a small language with a parser, five fields, three of them
//! with special cases, and it answers a question most people do not have — the
//! two schedules a workflow actually wants are "every so often" and "once a
//! day". Those are stated here in a form that cannot be misread, and the daily
//! one says out loud that it is in UTC rather than quietly meaning it.
//!
//! Everything is a pure function of the schedule and a timestamp, so the
//! awkward parts — a daily time that has already passed today, a restart, a
//! clock that jumped — are tested rather than reasoned about.

use crate::model::Node;

pub const DAY: i64 = 86_400;

/// The shortest interval a workflow may run on.
///
/// A flow with a model step and a one-second interval is a machine kept
/// permanently busy by a typo. Half a minute is short enough for anything that
/// is genuinely a schedule.
pub const FLOOR: i64 = 30;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Schedule {
    /// Every `n` seconds.
    Every(i64),
    /// Once a day, at this UTC time.
    DailyAt { hour: u32, minute: u32 },
}

/// Read a schedule from a trigger step's settings.
pub fn read(node: &Node) -> Result<Schedule, String> {
    let at = node.text("at").trim().to_string();
    if !at.is_empty() {
        return daily(&at);
    }
    let every = node.text("every").trim().to_string();
    if every.is_empty() {
        return Err("this schedule does not say how often to run".into());
    }
    interval(&every).map(Schedule::Every)
}

/// Parse `30s`, `5m`, `2h`, `1d`.
pub fn interval(text: &str) -> Result<i64, String> {
    let text = text.trim().to_ascii_lowercase();
    let (number, unit) = text.split_at(text.find(|c: char| c.is_alphabetic()).unwrap_or(text.len()));
    let number: i64 = number
        .trim()
        .parse()
        .map_err(|_| format!("`{text}` is not a length of time — try 30s, 5m, 2h or 1d"))?;

    let seconds = match unit.trim() {
        "s" | "sec" | "secs" | "second" | "seconds" => number,
        "m" | "min" | "mins" | "minute" | "minutes" => number * 60,
        "h" | "hr" | "hrs" | "hour" | "hours" => number * 3600,
        "d" | "day" | "days" => number * DAY,
        // A bare number has no obvious unit, and guessing wrong is either 60
        // times too often or 60 times too rarely.
        "" => return Err(format!("`{text}` needs a unit — try {text}m for minutes")),
        other => return Err(format!("`{other}` is not a unit of time — try s, m, h or d")),
    };

    if seconds < FLOOR {
        return Err(format!("the shortest a workflow may run is every {FLOOR}s"));
    }
    Ok(seconds)
}

/// Parse `09:00`.
fn daily(text: &str) -> Result<Schedule, String> {
    let (h, m) = text
        .split_once(':')
        .ok_or_else(|| format!("`{text}` is not a time of day — try 09:00"))?;
    let hour: u32 = h.trim().parse().map_err(|_| format!("`{h}` is not an hour"))?;
    let minute: u32 = m.trim().parse().map_err(|_| format!("`{m}` is not a minute"))?;
    if hour > 23 || minute > 59 {
        return Err(format!("`{text}` is not a time of day"));
    }
    Ok(Schedule::DailyAt { hour, minute })
}

/// The first moment at or after `since` that this schedule fires.
///
/// `since` is when the schedule was last dealt with — the last run, or when the
/// flow was switched on. Never earlier than that, so switching a flow on does
/// not immediately fire it for every interval that passed while it was off.
pub fn next_after(schedule: Schedule, since: i64) -> i64 {
    match schedule {
        Schedule::Every(seconds) => since + seconds.max(FLOOR),
        Schedule::DailyAt { hour, minute } => {
            let target = |day_start: i64| day_start + (hour as i64) * 3600 + (minute as i64) * 60;
            let today = since - since.rem_euclid(DAY);
            let at = target(today);
            // Strictly after, so a run that finishes within the same minute
            // does not immediately qualify again.
            if at > since { at } else { target(today + DAY) }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::Kind;
    use serde_json::json;
    use std::collections::BTreeMap;

    fn node(params: &[(&str, &str)]) -> Node {
        Node {
            id: "s".into(),
            kind: Kind::Schedule,
            name: String::new(),
            x: 0.0,
            y: 0.0,
            params: params
                .iter()
                .map(|(k, v)| (k.to_string(), json!(v)))
                .collect::<BTreeMap<_, _>>(),
        }
    }

    #[test]
    fn intervals_are_read_in_the_units_people_write() {
        assert_eq!(interval("30s"), Ok(30));
        assert_eq!(interval("5m"), Ok(300));
        assert_eq!(interval("2h"), Ok(7200));
        assert_eq!(interval("1d"), Ok(86_400));
        assert_eq!(interval(" 15 MINUTES "), Ok(900));
    }

    #[test]
    fn a_bare_number_is_refused_rather_than_guessed_at() {
        // Guessing wrong is either sixty times too often or sixty times too
        // rarely, and both look like the workflow is broken.
        let e = interval("5").unwrap_err();
        assert!(e.contains("needs a unit"), "{e}");
    }

    #[test]
    fn nothing_may_run_more_often_than_the_floor() {
        // A model step on a one-second timer is a machine kept permanently
        // busy by a typo.
        let e = interval("1s").unwrap_err();
        assert!(e.contains("30s"), "{e}");
        assert!(interval("30s").is_ok());
    }

    #[test]
    fn an_unreadable_interval_says_what_would_have_worked() {
        for text in ["soon", "5 fortnights", ""] {
            let e = interval(text).unwrap_err();
            assert!(e.contains("try"), "{text:?} gave {e:?}");
        }
    }

    #[test]
    fn an_interval_fires_that_far_after_the_last_time() {
        let s = Schedule::Every(300);
        assert_eq!(next_after(s, 1_000), 1_300);
        assert_eq!(next_after(s, 1_300), 1_600);
    }

    #[test]
    fn a_daily_time_later_today_fires_today() {
        // Midnight UTC plus eight hours; a 09:00 schedule is an hour away.
        let eight_am = 8 * 3600;
        assert_eq!(
            next_after(Schedule::DailyAt { hour: 9, minute: 0 }, eight_am),
            9 * 3600
        );
    }

    #[test]
    fn a_daily_time_already_past_fires_tomorrow() {
        let ten_am = 10 * 3600;
        assert_eq!(
            next_after(Schedule::DailyAt { hour: 9, minute: 0 }, ten_am),
            DAY + 9 * 3600
        );
    }

    #[test]
    fn a_daily_time_exactly_now_waits_for_tomorrow() {
        // Otherwise a run finishing inside the same minute qualifies again and
        // the flow runs in a tight loop for sixty seconds.
        let nine = 9 * 3600;
        assert_eq!(next_after(Schedule::DailyAt { hour: 9, minute: 0 }, nine), DAY + nine);
    }

    #[test]
    fn switching_a_flow_on_does_not_fire_it_for_every_missed_interval() {
        // The next time is always measured from `since`, so a flow switched
        // back on after a week fires once, not a thousand times.
        let a_week = 7 * DAY;
        let next = next_after(Schedule::Every(300), a_week);
        assert_eq!(next, a_week + 300);
    }

    #[test]
    fn a_step_can_be_configured_either_way() {
        assert_eq!(read(&node(&[("every", "10m")])), Ok(Schedule::Every(600)));
        assert_eq!(
            read(&node(&[("at", "09:30")])),
            Ok(Schedule::DailyAt { hour: 9, minute: 30 })
        );
        // A time of day wins, so a step that has both does the more specific
        // thing rather than silently doing both.
        assert_eq!(
            read(&node(&[("every", "10m"), ("at", "09:30")])),
            Ok(Schedule::DailyAt { hour: 9, minute: 30 })
        );
    }

    #[test]
    fn a_schedule_that_says_nothing_is_an_error_not_a_default() {
        // A default would be a flow running on a timer nobody chose.
        assert!(read(&node(&[])).is_err());
    }

    #[test]
    fn a_time_of_day_outside_a_day_is_refused() {
        for text in ["25:00", "09:70", "9", "nine"] {
            assert!(daily(text).is_err(), "{text} was accepted");
        }
    }
}
