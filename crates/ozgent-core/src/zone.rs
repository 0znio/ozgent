//! Local time, read from the system's own zone database.
//!
//! Everything else in ozgent works in UTC and is right to. A scheduler cannot:
//! "every weekday at 9:20" is a wall-clock time, and a person who writes it
//! means their clock, not Greenwich's. Between March and November half the
//! world's wall clocks move relative to UTC, so the offset has to be looked up
//! per instant rather than measured once at startup.
//!
//! [`crate::datetime`] deliberately avoids a date crate, and this follows suit
//! for the same reason — a new dependency in this crate rebuilds llama.cpp.
//! What it costs is a TZif reader, which is a hundred lines of a format that
//! has not changed since 1994 and is already on every machine ozgent runs on.
//!
//! If the zone files are missing or unreadable ozgent runs in UTC rather than
//! failing: a scheduler that fires at the wrong hour is a bug worth reporting,
//! and one that refuses to start because `/etc/localtime` is a broken symlink
//! is worse.

use crate::datetime::DateTime;
use std::path::{Path, PathBuf};

/// Where zone files live on a Unix system.
const ZONEINFO: &str = "/usr/share/zoneinfo";

/// A time zone: the offsets from UTC it has used, and when each began.
///
/// Cheap to clone and to keep — a zone with a century of transitions is a few
/// kilobytes, and the lookup is a binary search.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Zone {
    name: String,
    /// `(starts_at_utc, offset_seconds)`, sorted by the instant it starts.
    transitions: Vec<(i64, i32)>,
    /// The offset before the first transition, and the answer when there are
    /// none at all.
    base: i32,
}

impl Zone {
    /// A zone that is always UTC. The fallback, and useful in tests.
    pub fn utc() -> Self {
        Self { name: "UTC".into(), transitions: Vec::new(), base: 0 }
    }

    /// A zone at a fixed offset, for a machine with no zone database.
    pub fn fixed(name: &str, offset_seconds: i32) -> Self {
        Self { name: name.into(), transitions: Vec::new(), base: offset_seconds }
    }

    /// This machine's zone: `$TZ` if it names one, else `/etc/localtime`.
    ///
    /// Never fails. A machine whose zone cannot be read runs in UTC, and says
    /// so through [`Zone::name`] rather than by refusing to start.
    pub fn local() -> Self {
        if let Some(tz) = std::env::var_os("TZ") {
            let tz = tz.to_string_lossy();
            let tz = tz.trim_start_matches(':');
            if !tz.is_empty() {
                if let Ok(zone) = Self::named(tz) {
                    return zone;
                }
            }
        }
        let path = Path::new("/etc/localtime");
        match std::fs::read(path) {
            Ok(bytes) => {
                let name = link_name(path).unwrap_or_else(|| "local".to_string());
                parse(&name, &bytes).unwrap_or_else(|_| Self::utc())
            }
            Err(_) => Self::utc(),
        }
    }

    /// A named zone, as the IANA database spells it: `Asia/Kolkata`.
    pub fn named(name: &str) -> Result<Self, String> {
        if name.eq_ignore_ascii_case("utc") || name.eq_ignore_ascii_case("gmt") {
            return Ok(Self::utc());
        }
        if name.eq_ignore_ascii_case("local") {
            return Ok(Self::local());
        }
        let path = zone_path(name).ok_or_else(|| format!("{name:?} is not a time zone name"))?;
        let bytes = std::fs::read(&path)
            .map_err(|e| format!("reading the zone file for {name}: {e}"))?;
        parse(name, &bytes)
    }

    pub fn name(&self) -> &str {
        &self.name
    }

    /// Seconds east of UTC at this instant.
    pub fn offset_at(&self, utc: i64) -> i32 {
        match self.transitions.partition_point(|(start, _)| *start <= utc) {
            0 => self.base,
            i => self.transitions[i - 1].1,
        }
    }

    /// The wall-clock time here at this instant.
    pub fn local_at(&self, utc: i64) -> DateTime {
        DateTime::from_unix(utc + self.offset_at(utc) as i64)
    }

    /// The instant at which the clock here reads this wall-clock time.
    ///
    /// Two hours a year have no single answer and both are handled rather than
    /// left to produce a wrong one:
    ///
    /// * The hour that repeats when clocks go back happens twice. The *first*
    ///   is returned — the earlier of the two is what "01:30" means to someone
    ///   who set an alarm for it.
    /// * The hour skipped when clocks go forward never happens. The instant
    ///   the clock jumps to is returned, so a job set for 02:30 on that one
    ///   morning runs at 03:00 instead of silently not running for a year.
    pub fn instant_of(&self, wall: &DateTime) -> i64 {
        let naive = wall.to_unix();
        // Solving utc = naive - offset(utc) by substitution finds *an* answer
        // but not reliably the right one: in the repeated hour both readings
        // satisfy it, and which one substitution lands on is an accident of
        // where it started. So try every offset in play around this instant
        // and keep the answers that actually read back as the time asked for.
        let mut best: Option<i64> = None;
        for probe in [naive - 86_400, naive, naive + 86_400] {
            let candidate = naive - self.offset_at(probe) as i64;
            if self.local_at(candidate).to_unix() != naive {
                continue;
            }
            best = Some(best.map_or(candidate, |b: i64| b.min(candidate)));
        }
        if let Some(utc) = best {
            return utc;
        }
        // No offset reads back: the clock jumped over this time. Answer with
        // the instant it jumped to, so a job set for the skipped hour runs
        // that morning instead of silently not running for a year.
        self.transitions
            .iter()
            .find(|(start, off)| *start + *off as i64 > naive)
            .map(|(start, _)| *start)
            .unwrap_or(naive - self.base as i64)
    }

    /// The abbreviation a person expects to see: `IST`, `+0530`, `UTC`.
    ///
    /// Derived from the offset rather than the zone file's own strings, which
    /// are ambiguous across zones — `IST` is India, Ireland and Israel.
    pub fn label_at(&self, utc: i64) -> String {
        let offset = self.offset_at(utc);
        if offset == 0 {
            return "UTC".into();
        }
        let sign = if offset < 0 { '-' } else { '+' };
        let offset = offset.abs();
        let (hours, minutes) = (offset / 3600, (offset % 3600) / 60);
        format!("UTC{sign}{hours}:{minutes:02}")
    }
}

impl Default for Zone {
    fn default() -> Self {
        Self::utc()
    }
}

/// Resolve a zone name to a file, refusing anything that escapes the database.
///
/// The name reaches here from a saved job, and a job can be written by anyone
/// allowed to use the scheduler. `../../etc/shadow` must not become a readable
/// path, and a name with a NUL or a leading slash must not either.
fn zone_path(name: &str) -> Option<PathBuf> {
    if name.is_empty() || name.len() > 64 {
        return None;
    }
    let ok = name.split('/').all(|part| {
        !part.is_empty()
            && part != "."
            && part != ".."
            && part
                .chars()
                .all(|c| c.is_ascii_alphanumeric() || c == '_' || c == '-' || c == '+')
    });
    if !ok {
        return None;
    }
    let path = Path::new(ZONEINFO).join(name);
    path.is_file().then_some(path)
}

/// The zone name `/etc/localtime` points at, when it is a symlink into the
/// database — which is how every mainstream distribution sets it.
fn link_name(path: &Path) -> Option<String> {
    let target = std::fs::read_link(path).ok()?;
    let target = target.to_string_lossy();
    let rest = target.split("zoneinfo/").nth(1)?;
    (!rest.is_empty()).then(|| rest.to_string())
}

/// Read a TZif file.
///
/// A v2+ file holds the whole thing twice: a 32-bit block for readers that
/// predate 1995, then a 64-bit block with the same data and more of it. The
/// second is the one to use, so the first is parsed only far enough to know
/// how many bytes to step over.
fn parse(name: &str, bytes: &[u8]) -> Result<Zone, String> {
    let head = Header::read(bytes, 0)?;
    let (header, at) = if head.version >= 2 {
        let after = 44 + head.block_size(4);
        (Header::read(bytes, after)?, after + 44)
    } else {
        (head, 44)
    };
    let width = if header.version >= 2 { 8 } else { 4 };

    let mut at = at;
    let mut starts = Vec::with_capacity(header.timecnt);
    for _ in 0..header.timecnt {
        starts.push(read_int(bytes, at, width)?);
        at += width;
    }
    let mut indices = Vec::with_capacity(header.timecnt);
    for _ in 0..header.timecnt {
        indices.push(*bytes.get(at).ok_or("truncated zone file")? as usize);
        at += 1;
    }
    let mut offsets = Vec::with_capacity(header.typecnt);
    let mut is_dst = Vec::with_capacity(header.typecnt);
    for _ in 0..header.typecnt {
        offsets.push(read_int(bytes, at, 4)? as i32);
        is_dst.push(*bytes.get(at + 4).ok_or("truncated zone file")? != 0);
        at += 6;
    }
    if offsets.is_empty() {
        return Err("a zone file with no offsets in it".into());
    }

    let transitions: Vec<(i64, i32)> = starts
        .into_iter()
        .zip(indices)
        .filter_map(|(start, i)| offsets.get(i).map(|off| (start, *off)))
        .collect();

    // Before the first transition, the convention is the first type that is
    // not daylight saving; a zone with only DST types falls back to the first.
    let base = is_dst
        .iter()
        .position(|dst| !dst)
        .map_or(offsets[0], |i| offsets[i]);

    Ok(Zone { name: name.to_string(), transitions, base })
}

/// The counts at the top of each data block.
struct Header {
    version: u8,
    timecnt: usize,
    typecnt: usize,
    charcnt: usize,
    leapcnt: usize,
    isstdcnt: usize,
    isutcnt: usize,
}

impl Header {
    fn read(bytes: &[u8], at: usize) -> Result<Self, String> {
        let head = bytes.get(at..at + 44).ok_or("not a zone file: too short")?;
        if &head[..4] != b"TZif" {
            return Err("not a zone file: wrong magic".into());
        }
        let version = match head[4] {
            0 => 1,
            v if v.is_ascii_digit() => v - b'0',
            _ => return Err("a zone file in a version this build does not read".into()),
        };
        let n = |i: usize| -> usize {
            u32::from_be_bytes([head[20 + i * 4], head[21 + i * 4], head[22 + i * 4], head[23 + i * 4]])
                as usize
        };
        Ok(Self {
            version,
            isutcnt: n(0),
            isstdcnt: n(1),
            leapcnt: n(2),
            timecnt: n(3),
            typecnt: n(4),
            charcnt: n(5),
        })
    }

    /// Bytes of data after this header, for transition times `width` wide.
    fn block_size(&self, width: usize) -> usize {
        self.timecnt * (width + 1)
            + self.typecnt * 6
            + self.charcnt
            + self.leapcnt * (width + 4)
            + self.isstdcnt
            + self.isutcnt
    }
}

fn read_int(bytes: &[u8], at: usize, width: usize) -> Result<i64, String> {
    let slice = bytes.get(at..at + width).ok_or("truncated zone file")?;
    Ok(match width {
        4 => i32::from_be_bytes(slice.try_into().unwrap()) as i64,
        _ => i64::from_be_bytes(slice.try_into().unwrap()),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Whether this machine has a zone database at all. The parsing tests are
    /// meaningless without one, and a minimal container may not have it.
    fn has_database() -> bool {
        Path::new(ZONEINFO).join("Asia/Kolkata").is_file()
    }

    #[test]
    fn utc_is_always_zero() {
        let z = Zone::utc();
        assert_eq!(z.offset_at(0), 0);
        assert_eq!(z.offset_at(1_755_648_000), 0);
        assert_eq!(z.label_at(0), "UTC");
    }

    #[test]
    fn a_fixed_zone_shifts_the_wall_clock() {
        let z = Zone::fixed("test", 5 * 3600 + 1800);
        // 2025-08-20 00:00 UTC is 05:30 in a +5:30 zone.
        let wall = z.local_at(1_755_648_000);
        assert_eq!((wall.hour, wall.minute), (5, 30));
        assert_eq!(z.label_at(0), "UTC+5:30");
    }

    #[test]
    fn a_wall_clock_time_converts_back_to_the_instant_it_names() {
        let z = Zone::fixed("test", 5 * 3600 + 1800);
        let wall = DateTime::civil(2025, 8, 20, 9, 20, 0);
        let utc = z.instant_of(&wall);
        // 09:20 at +5:30 is 03:50 UTC.
        assert_eq!(DateTime::from_unix(utc).hour, 3);
        assert_eq!(DateTime::from_unix(utc).minute, 50);
        // And it is a fixed point: converting back gives the same wall clock.
        assert_eq!(z.local_at(utc).to_unix(), wall.to_unix());
    }

    #[test]
    fn a_negative_offset_works_too() {
        let z = Zone::fixed("test", -5 * 3600);
        let wall = DateTime::civil(2025, 8, 20, 9, 20, 0);
        assert_eq!(z.local_at(z.instant_of(&wall)).to_unix(), wall.to_unix());
        assert_eq!(z.label_at(0), "UTC-5:00");
    }

    #[test]
    fn india_has_one_offset_all_year() {
        if !has_database() {
            return;
        }
        let z = Zone::named("Asia/Kolkata").expect("Asia/Kolkata must parse");
        // January and July, to catch a reader that finds no transitions and
        // silently answers UTC.
        for instant in [1_767_225_600, 1_782_950_400] {
            assert_eq!(z.offset_at(instant), 5 * 3600 + 1800, "at {instant}");
        }
        assert_eq!(z.label_at(0), "UTC+5:30");
    }

    #[test]
    fn a_zone_with_daylight_saving_changes_offset_across_the_year() {
        if !Path::new(ZONEINFO).join("Europe/London").is_file() {
            return;
        }
        let z = Zone::named("Europe/London").unwrap();
        // 2026-01-15 is GMT; 2026-07-15 is BST, an hour ahead.
        assert_eq!(z.offset_at(1_768_435_200), 0, "January is GMT");
        assert_eq!(z.offset_at(1_784_246_400), 3600, "July is BST");
    }

    #[test]
    fn the_hour_that_happens_twice_resolves_to_the_first_one() {
        if !Path::new(ZONEINFO).join("Europe/London").is_file() {
            return;
        }
        let z = Zone::named("Europe/London").unwrap();
        // Clocks go back at 02:00 BST on 25 October 2026, so 01:30 happens
        // once at 00:30 UTC and again at 01:30 UTC.
        let utc = z.instant_of(&DateTime::civil(2026, 10, 25, 1, 30, 0));
        assert_eq!(DateTime::from_unix(utc).hour, 0, "the earlier of the two");
    }

    #[test]
    fn the_hour_that_never_happens_still_produces_an_instant() {
        if !Path::new(ZONEINFO).join("Europe/London").is_file() {
            return;
        }
        // Clocks jump 01:00 -> 02:00 on 29 March 2026, so 01:30 does not
        // exist. A job set for it must still run that morning rather than
        // being skipped for a year.
        let z = Zone::named("Europe/London").unwrap();
        let utc = z.instant_of(&DateTime::civil(2026, 3, 29, 1, 30, 0));
        let wall = z.local_at(utc);
        assert_eq!((wall.year, wall.month, wall.day), (2026, 3, 29));
        assert_eq!(wall.hour, 2, "snapped to the moment the clock jumped");
    }

    #[test]
    fn every_instant_round_trips_through_a_real_zone() {
        if !has_database() {
            return;
        }
        let z = Zone::named("Asia/Kolkata").unwrap();
        // Four times a day for a year, which crosses every transition a zone
        // can have without taking measurable time.
        let mut utc = 1_767_225_600;
        while utc < 1_767_225_600 + 365 * 86_400 {
            assert_eq!(z.instant_of(&z.local_at(utc)), utc, "at {utc}");
            utc += 6 * 3600;
        }
    }

    #[test]
    fn utc_and_gmt_are_accepted_by_name_without_a_database() {
        assert_eq!(Zone::named("UTC").unwrap().offset_at(0), 0);
        assert_eq!(Zone::named("utc").unwrap().name(), "UTC");
        assert_eq!(Zone::named("GMT").unwrap().offset_at(0), 0);
    }

    #[test]
    fn a_name_that_is_not_a_zone_is_an_error_rather_than_utc() {
        // Silently falling back would put a job an unknown number of hours out
        // and say nothing about it.
        assert!(Zone::named("Mars/Olympus").is_err());
        assert!(Zone::named("").is_err());
    }

    #[test]
    fn a_zone_name_cannot_escape_the_database() {
        // The name comes from a saved job, which anyone allowed to use the
        // scheduler can write.
        for attempt in [
            "../../../etc/passwd",
            "/etc/passwd",
            "Asia/../../etc/passwd",
            "Asia/Kolkata\0",
            "./Asia/Kolkata",
        ] {
            assert!(zone_path(attempt).is_none(), "{attempt} must not resolve");
        }
    }

    #[test]
    fn a_truncated_file_is_an_error_rather_than_a_panic() {
        assert!(parse("x", b"").is_err());
        assert!(parse("x", b"TZif2").is_err());
        assert!(parse("x", &[0u8; 44]).is_err(), "wrong magic");
        // A valid header promising data that is not there.
        let mut bytes = vec![0u8; 44];
        bytes[..4].copy_from_slice(b"TZif");
        bytes[4] = b'2';
        bytes[32..36].copy_from_slice(&99u32.to_be_bytes()); // timecnt
        assert!(parse("x", &bytes).is_err());
    }

    #[test]
    fn the_local_zone_is_always_usable() {
        // Whatever this machine has, reading it must not fail.
        let z = Zone::local();
        let now = DateTime::now().to_unix();
        assert!(z.offset_at(now).abs() <= 16 * 3600, "implausible offset");
        assert_eq!(z.local_at(z.instant_of(&z.local_at(now))).to_unix(), z.local_at(now).to_unix());
    }
}
