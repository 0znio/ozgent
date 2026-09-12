//! Telling the model what day it is.
//!
//! A model has no clock and its training data ends in the past, so "latest
//! news" or "tomorrow" resolve against whatever year it happened to memorise —
//! in practice it guesses, and guesses wrong. Stating the date in the system
//! prompt fixes that.
//!
//! The conversion is done here rather than with a date crate: it is thirty
//! lines of well-known arithmetic, and a new dependency would rebuild
//! llama.cpp.

use std::time::{SystemTime, UNIX_EPOCH};

/// A civil date and time, in UTC.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DateTime {
    pub year: i64,
    /// 1-12.
    pub month: u32,
    /// 1-31.
    pub day: u32,
    pub hour: u32,
    pub minute: u32,
    pub second: u32,
    /// 0 = Sunday.
    pub weekday: u32,
}

const WEEKDAYS: [&str; 7] = [
    "Sunday", "Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday",
];
const MONTHS: [&str; 12] = [
    "January", "February", "March", "April", "May", "June", "July", "August",
    "September", "October", "November", "December",
];

impl DateTime {
    /// The current UTC time, or the epoch if the clock is unreadable.
    pub fn now() -> Self {
        let secs = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|d| d.as_secs() as i64)
            .unwrap_or(0);
        Self::from_unix(secs)
    }

    pub fn from_unix(secs: i64) -> Self {
        // Floor division, so times before 1970 do not round toward zero.
        let days = secs.div_euclid(86_400);
        let rem = secs.rem_euclid(86_400);
        let (year, month, day) = civil_from_days(days);

        Self {
            year,
            month,
            day,
            hour: (rem / 3600) as u32,
            minute: ((rem % 3600) / 60) as u32,
            second: (rem % 60) as u32,
            // 1970-01-01 was a Thursday, which is index 4.
            weekday: (days + 4).rem_euclid(7) as u32,
        }
    }

    pub fn weekday_name(&self) -> &'static str {
        WEEKDAYS[(self.weekday as usize).min(6)]
    }

    pub fn month_name(&self) -> &'static str {
        MONTHS[(self.month as usize).saturating_sub(1).min(11)]
    }

    /// `2026-08-20`.
    pub fn iso_date(&self) -> String {
        format!("{:04}-{:02}-{:02}", self.year, self.month, self.day)
    }

    /// The line given to the model.
    ///
    /// The weekday is included deliberately: questions like "tomorrow's
    /// market" depend on it, and a model cannot derive it from the date.
    /// UTC is stated rather than implied, so the model does not assume local
    /// time it has no way to know.
    /// A wall-clock time, with no zone attached.
    ///
    /// `weekday` is derived, so callers building a time never have to know it.
    pub fn civil(year: i64, month: u32, day: u32, hour: u32, minute: u32, second: u32) -> Self {
        let days = days_from_civil(year, month, day);
        Self {
            year,
            month,
            day,
            hour,
            minute,
            second,
            weekday: (days + 4).rem_euclid(7) as u32,
        }
    }

    /// Seconds since the epoch, reading these fields as UTC.
    ///
    /// The exact inverse of [`DateTime::from_unix`]. A scheduler needs both
    /// directions: "09:20 on the next weekday" is decided in civil fields and
    /// has to come back as an instant.
    pub fn to_unix(&self) -> i64 {
        days_from_civil(self.year, self.month, self.day) * 86_400
            + self.hour as i64 * 3600
            + self.minute as i64 * 60
            + self.second as i64
    }

    pub fn prompt_line(&self) -> String {
        format!(
            "Today is {}, {} {} {}. The current time is {:02}:{:02} UTC.",
            self.weekday_name(),
            self.day,
            self.month_name(),
            self.year,
            self.hour,
            self.minute
        )
    }
}

/// Civil date from days since the Unix epoch.
///
/// Howard Hinnant's `civil_from_days`, which is exact for the proleptic
/// Gregorian calendar over any range that fits in `i64`.
pub(crate) fn civil_from_days(days: i64) -> (i64, u32, u32) {
    let z = days + 719_468;
    let era = if z >= 0 { z } else { z - 146_096 } / 146_097;
    let doe = z - era * 146_097;
    let yoe = (doe - doe / 1460 + doe / 36_524 - doe / 146_096) / 365;
    let y = yoe + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = (doy - (153 * mp + 2) / 5 + 1) as u32;
    let m = (if mp < 10 { mp + 3 } else { mp - 9 }) as u32;
    (if m <= 2 { y + 1 } else { y }, m, d)
}

/// Days since the Unix epoch from a civil date.
///
/// Hinnant's `days_from_civil`, the exact inverse of [`civil_from_days`] for
/// any date the proleptic Gregorian calendar defines.
pub(crate) fn days_from_civil(year: i64, month: u32, day: u32) -> i64 {
    let m = month as i64;
    let d = day as i64;
    let y = if m <= 2 { year - 1 } else { year };
    let era = if y >= 0 { y } else { y - 399 } / 400;
    let yoe = y - era * 400;
    let mp = if m > 2 { m - 3 } else { m + 9 };
    let doy = (153 * mp + 2) / 5 + d - 1;
    let doe = yoe * 365 + yoe / 4 - yoe / 100 + doy;
    era * 146_097 + doe - 719_468
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_epoch_is_a_thursday() {
        let t = DateTime::from_unix(0);
        assert_eq!((t.year, t.month, t.day), (1970, 1, 1));
        assert_eq!(t.weekday_name(), "Thursday");
        assert_eq!(t.hour, 0);
    }

    #[test]
    fn known_dates_convert_exactly() {
        for (secs, y, m, d, weekday) in [
            (1_000_000_000, 2001, 9, 9, "Sunday"),
            (1_600_000_000, 2020, 9, 13, "Sunday"),
            (1_767_225_600, 2026, 1, 1, "Thursday"),
            (1_755_648_000, 2025, 8, 20, "Wednesday"),
        ] {
            let t = DateTime::from_unix(secs);
            assert_eq!((t.year, t.month, t.day), (y, m, d), "for {secs}");
            assert_eq!(t.weekday_name(), weekday, "for {secs}");
        }
    }

    #[test]
    fn leap_days_are_handled() {
        // 2024 is divisible by 4, so 29 February exists.
        let leap = DateTime::from_unix(1_709_164_800);
        assert_eq!((leap.year, leap.month, leap.day), (2024, 2, 29));
    }

    #[test]
    fn the_century_rule_is_applied() {
        // 2100 is divisible by 100 but not 400, so it is *not* a leap year:
        // the day after 28 February is 1 March, not 29 February.
        let feb28 = DateTime::from_unix(4_107_456_000);
        assert_eq!((feb28.year, feb28.month, feb28.day), (2100, 2, 28));

        let next = DateTime::from_unix(4_107_456_000 + 86_400);
        assert_eq!(
            (next.year, next.month, next.day),
            (2100, 3, 1),
            "2100 is not a leap year"
        );

        // 2000 *is* divisible by 400, so it is a leap year.
        let feb29_2000 = DateTime::from_unix(951_782_400);
        assert_eq!((feb29_2000.year, feb29_2000.month, feb29_2000.day), (2000, 2, 29));
    }

    #[test]
    fn time_of_day_is_extracted() {
        // 1970-01-01 13:45:30
        let t = DateTime::from_unix(13 * 3600 + 45 * 60 + 30);
        assert_eq!((t.hour, t.minute, t.second), (13, 45, 30));
    }

    #[test]
    fn times_before_the_epoch_do_not_round_toward_zero() {
        // One second before the epoch is 1969-12-31 23:59:59, not 1970.
        let t = DateTime::from_unix(-1);
        assert_eq!((t.year, t.month, t.day), (1969, 12, 31));
        assert_eq!((t.hour, t.minute, t.second), (23, 59, 59));
    }

    #[test]
    fn weekdays_advance_one_per_day() {
        let base = 1_755_648_000; // a Wednesday
        let names: Vec<&str> = (0..8)
            .map(|i| DateTime::from_unix(base + i * 86_400).weekday_name())
            .collect();
        assert_eq!(
            names,
            [
                "Wednesday", "Thursday", "Friday", "Saturday", "Sunday", "Monday",
                "Tuesday", "Wednesday"
            ]
        );
    }

    #[test]
    fn the_prompt_line_names_the_weekday_and_says_utc() {
        let t = DateTime::from_unix(1_755_648_000);
        let line = t.prompt_line();
        assert!(line.contains("Wednesday"), "{line}");
        assert!(line.contains("20 August 2025"), "{line}");
        assert!(line.contains("UTC"), "the zone must be explicit: {line}");
    }

    #[test]
    fn iso_dates_are_zero_padded() {
        assert_eq!(DateTime::from_unix(1_767_225_600).iso_date(), "2026-01-01");
    }

    #[test]
    fn now_is_plausible() {
        // Guards against a sign or scaling error in the conversion.
        let t = DateTime::now();
        assert!(t.year >= 2024 && t.year < 2100, "implausible year {}", t.year);
        assert!((1..=12).contains(&t.month));
        assert!((1..=31).contains(&t.day));
        assert!(t.hour < 24 && t.minute < 60);
    }

    #[test]
    fn the_two_conversions_are_exact_inverses() {
        // Every scheduled fire is computed in civil fields and stored as an
        // instant, so a one-day drift here is a brief delivered on the wrong
        // morning.
        for secs in [
            0, -1, 1, 1_000_000_000, 1_600_000_000, 1_767_225_600, 1_709_164_800,
            4_107_456_000, 951_782_400, -2_208_988_800,
        ] {
            assert_eq!(DateTime::from_unix(secs).to_unix(), secs, "for {secs}");
        }
    }

    #[test]
    fn a_civil_time_knows_its_own_weekday() {
        // Callers build "09:20 on the 20th" without knowing the weekday; the
        // recurrence rules then match on it.
        let t = DateTime::civil(2025, 8, 20, 9, 20, 0);
        assert_eq!(t.weekday_name(), "Wednesday");
        assert_eq!(t.to_unix(), 1_755_648_000 + 9 * 3600 + 20 * 60);
    }

    #[test]
    fn every_day_of_a_leap_year_round_trips() {
        let mut day = days_from_civil(2024, 1, 1);
        let end = days_from_civil(2025, 1, 1);
        let mut seen = 0;
        while day < end {
            let t = DateTime::from_unix(day * 86_400);
            assert_eq!(days_from_civil(t.year, t.month, t.day), day);
            day += 1;
            seen += 1;
        }
        assert_eq!(seen, 366, "2024 is a leap year");
    }
}
