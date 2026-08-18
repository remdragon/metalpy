# lib/time.py — high-resolution timing, Python-compatible names and semantics.
#
#   time.monotonic() -> f64   monotonic high-res counter, in SECONDS. The right
#                             tool for benchmarking: arbitrary zero point (only
#                             differences are meaningful), never runs backwards,
#                             immune to wall-clock/NTP adjustments.
#   time.time()      -> f64   wall-clock SECONDS since the Unix epoch, matching
#                             CPython's time.time() meaning (and its ~sub-µs
#                             float-precision limit at present-day magnitudes).
#
# Each function is defined once per OS via @compiler.target; discovery discards
# the non-matching body outright, so its platform-specific imports are never
# resolved on the other OS (the same pattern lib/sys.py and lib/fs.py use).
# Windows uses kernel32 (QueryPerformanceCounter/Frequency, GetSystemTimePrecise-
# AsFileTime); POSIX uses libc clock_gettime. See lib/windows/kernel32.py and
# lib/posix/time.py for the extern declarations and the rationale.

import compiler


# ---------------------------------------------------------------------------
# monotonic() -> f64 seconds
# ---------------------------------------------------------------------------

@compiler.target( os = 'windows' )
def monotonic() -> f64:
	from windows.kernel32 import QueryPerformanceCounter, QueryPerformanceFrequency
	freq: i64 = 0
	counts: i64 = 0
	QueryPerformanceFrequency( compiler.addrof( freq ) )
	QueryPerformanceCounter( compiler.addrof( counts ) )
	# counts / ticks-per-second = seconds. Done in f64, so no integer overflow;
	# at ~10 MHz, `counts` stays under 2^52 for ~14 years of uptime before f64's
	# mantissa would start to shed sub-µs precision - irrelevant in practice.
	with compiler.wrap_arithmetic:
		return f64( counts ) / f64( freq )

@compiler.target( os = not 'windows' )
def monotonic() -> f64:
	from posix.time import clock_gettime, timespec, CLOCK_MONOTONIC
	ts = timespec()
	clock_gettime( CLOCK_MONOTONIC, compiler.addrof( ts ) )
	with compiler.wrap_arithmetic:
		return f64( ts.tv_sec ) + f64( ts.tv_nsec ) / 1.0e9


# ---------------------------------------------------------------------------
# time() -> f64 seconds since the Unix epoch
# ---------------------------------------------------------------------------

# FILETIME counts 100-ns ticks from 1601-01-01; this is the tick delta to the
# 1970-01-01 Unix epoch.
_FILETIME_UNIX_EPOCH_DIFF: u64 = 116444736000000000

@compiler.target( os = 'windows' )
def time() -> f64:
	from windows.kernel32 import GetSystemTimePreciseAsFileTime
	ticks: u64 = 0
	GetSystemTimePreciseAsFileTime( compiler.addrof( ticks ) )
	# Subtract the epoch in u64 first (never underflows for any real system
	# clock), then convert - a u64 subtraction, so wrap_arithmetic to keep it a
	# plain value rather than a Result.
	with compiler.wrap_arithmetic:
		unix_100ns: u64 = ticks - _FILETIME_UNIX_EPOCH_DIFF
		return f64( unix_100ns ) / 1.0e7  # 100-ns ticks -> seconds

@compiler.target( os = not 'windows' )
def time() -> f64:
	from posix.time import clock_gettime, timespec, CLOCK_REALTIME
	ts = timespec()
	clock_gettime( CLOCK_REALTIME, compiler.addrof( ts ) )
	with compiler.wrap_arithmetic:
		return f64( ts.tv_sec ) + f64( ts.tv_nsec ) / 1.0e9


# ---------------------------------------------------------------------------
# get_local_timezone_name() -> str  the OS's configured local zone name -
# 'America/New_York'-style on POSIX, a native Windows zone name (e.g.
# 'Eastern Standard Time') unless windows_zones.install() has been called
# (see lib/windows_zones.py) on Windows. Both platform bodies already
# existed (windows/time.py, posix/time.py) but were never re-exported from
# this top-level module the way monotonic()/time() already are - this is
# that dispatcher, added for lib/datetime.py's factories (date.today(tz),
# datetime.now(tz), ...), which need an easy way for a caller to get their
# local zone name to pass in explicitly (tz stays a required argument on
# those factories regardless - see lib/datetime.py's own docstring for why).
# ---------------------------------------------------------------------------

@compiler.target( os = 'windows' )
def get_local_timezone_name() -> str:
	from windows.time import get_local_timezone_name as _get_local_timezone_name
	return _get_local_timezone_name()

@compiler.target( os = not 'windows' )
def get_local_timezone_name() -> str:
	from posix.time import get_local_timezone_name as _get_local_timezone_name
	return _get_local_timezone_name()
