# Real-compile-and-run behavioral tests for lib/datetime.py (timedelta,
# date, time, datetime) and lib/_civil_calendar.py.
#
# Each program returns 0 when every check passes and a distinct nonzero i32
# exit code per failed check; RealCompileMixin/assert_programs_run decodes
# that back to the failing case+check. Expected values throughout are
# cross-checked against real CPython datetime/zoneinfo output (computed
# independently, not derived from this implementation), matching the
# zoneinfo work's own verification approach.
#
# Tests that call windows_zones.install() or otherwise touch process-global
# state are NOT merged with others via assert_programs_run (see its own
# docstring in test_support.py) - each such case gets its own
# assert_programs_run([...]) call so it builds its own executable.

import unittest

import test_support
from test_support import RealCompileMixin


_TIMEDELTA_BEHAVIOR = '''
import compiler
from datetime import timedelta

def main() -> i32:
	# negative normalization - seconds/microseconds always non-negative,
	# only .days goes negative (Python's own documented invariant; the
	# concrete case C-style truncating // would get wrong)
	td1 = timedelta( seconds = -1 )
	if td1.days != -1:
		return 1
	if td1.seconds != 86399:
		return 2
	if td1.microseconds != 0:
		return 3
	if td1.__str__() != '-1 day, 23:59:59':
		return 4

	td2 = timedelta( days = 1, hours = 2, minutes = 3, seconds = 4 )
	if td2.days != 1 or td2.seconds != 7384 or td2.microseconds != 0:
		return 5
	if td2.__str__() != '1 day, 2:03:04':
		return 6

	td3 = timedelta( hours = 2, minutes = 3, seconds = 4, microseconds = 500000 )
	if td3.__str__() != '2:03:04.500000':
		return 7

	# arithmetic
	summed = td2 + td3
	if summed.total_us != 93784000000 + 7384500000:
		return 8
	diffed = td2 - td3
	if diffed.total_us != 93784000000 - 7384500000:
		return 9
	negated = -td1
	if negated.total_us != 1000000:
		return 10
	multiplied = timedelta( microseconds = 1 ) * i32( 5 )
	if multiplied.total_us != 5:
		return 11

	# comparisons
	if not ( td1 == timedelta( seconds = -1 ) ):
		return 12
	if td1 == td2:
		return 13
	if not ( td1 < td2 ):
		return 14
	if not ( td2 > td1 ):
		return 15

	with compiler.wrap_arithmetic:
		diff: f64 = td2.total_seconds() - 93784.0
		if diff < 0.0:
			diff = -diff
	if diff > 0.0001:
		return 16

	return 0
'''

_DATE_BEHAVIOR = '''
from datetime import date, timedelta, localtz

def main() -> i32:
	r1 = date( 2024, 3, 10 )
	d1: date = r1.unwrap( 'valid' )
	r2 = date( 2024, 3, 15 )
	d2: date = r2.unwrap( 'valid' )

	if d1.year != 2024 or d1.month != 3 or d1.day != 10:
		return 1
	if d1.toordinal() != 738955:
		return 2
	if d1.weekday() != 6:            # 2024-03-10 is a Sunday (Mon=0..Sun=6)
		return 3
	if d1.isoweekday() != 7:
		return 4
	if d1.isoformat() != '2024-03-10':
		return 5

	diff = d2 - d1
	if diff.days != 5:
		return 6
	plus5 = d1 + timedelta( days = 5 )
	if plus5.isoformat() != '2024-03-15':
		return 7

	# year 1 / year 9999 boundaries, cross-checked against real CPython
	r_min = date( 1, 1, 1 )
	if r_min.unwrap( 'valid' ).toordinal() != 1:
		return 8
	r_max = date( 9999, 12, 31 )
	if r_max.unwrap( 'valid' ).toordinal() != 3652059:
		return 9
	if date.fromordinal( 1 ).isoformat() != '0001-01-01':
		return 10

	# leap-year edges
	r_leap = date( 2024, 2, 29 )
	if not r_leap.is_ok():
		return 11
	r_not_leap = date( 2023, 2, 29 )
	if r_not_leap.is_ok():
		return 12
	r_century_leap = date( 2000, 2, 29 )
	if not r_century_leap.is_ok():
		return 13
	r_century_not_leap = date( 1900, 2, 29 )
	if r_century_not_leap.is_ok():
		return 14

	# invalid construction rejected via Result.Err, not a panic/crash
	if date( 2024, 2, 30 ).is_ok():
		return 15
	if date( 2024, 13, 1 ).is_ok():
		return 16
	if date( 0, 1, 1 ).is_ok():
		return 17
	if date( 10000, 1, 1 ).is_ok():
		return 18

	if not ( d1 < d2 ):
		return 19
	if not ( d1 == date( 2024, 3, 10 ).unwrap( 'valid' ) ):
		return 20

	# date - timedelta -> date (the __sub__ overload, mirroring __add__)
	minus5 = d2 - timedelta( days = 5 )
	if minus5.isoformat() != '2024-03-10':
		return 21
	if not ( minus5 == d1 ):
		return 22

	# tz-defaulting: date.today()/fromtimestamp() with tz omitted uses the
	# system's own local zone (localtz()), same result as passing it
	# explicitly
	explicit_tz = localtz()
	if date.today() != date.today( explicit_tz ):
		return 23
	if date.fromtimestamp( 1_700_000_000.0 ) != date.fromtimestamp( 1_700_000_000.0, explicit_tz ):
		return 24

	return 0
'''

_TIME_BEHAVIOR = '''
from datetime import time as dtime

def main() -> i32:
	r1 = dtime( 14, 30, 45, 123456 )
	t1: dtime = r1.unwrap( 'valid' )
	if t1.hour != 14 or t1.minute != 30 or t1.second != 45 or t1.microsecond != 123456:
		return 1
	if t1.isoformat() != '14:30:45.123456':
		return 2

	r2 = dtime( 0, 0, 0 )
	if r2.unwrap( 'valid' ).isoformat() != '00:00:00':
		return 3

	r3 = dtime( 23, 59, 59 )
	t3: dtime = r3.unwrap( 'valid' )
	if not ( t1 < t3 ):
		return 4

	if dtime( 24, 0, 0 ).is_ok():
		return 5
	if dtime( 0, 60, 0 ).is_ok():
		return 6
	if dtime( 0, 0, 60 ).is_ok():
		return 7
	if dtime( 0, 0, 0, 1000000 ).is_ok():
		return 8

	if not ( t1 == dtime( 14, 30, 45, 123456 ).unwrap( 'valid' ) ):
		return 9

	return 0
'''

# fixed-offset ZoneInfo (no OS zone lookup, no windows_zones.install() needed)
# for the parts of datetime that don't specifically need a real DST-observing
# zone - safe to merge with other cases via assert_programs_run.
_DATETIME_FIXED_OFFSET_BEHAVIOR = '''
import compiler
import zoneinfo
from datetime import datetime, timedelta, localtz

def main() -> i32:
	tz = zoneinfo.ZoneInfo.fixed_offset( 19800, '+05:30' )

	r1 = datetime( 2024, 6, 15, 10, 30, 0, tzinfo = tz )
	dt1: datetime = r1.unwrap( 'valid' )
	if dt1.isoformat() != '2024-06-15T10:30:00+05:30':
		return 1

	dt2 = dt1 + timedelta( hours = 1 )
	dt2u: datetime = dt2.unwrap( 'valid' )
	if dt2u.isoformat() != '2024-06-15T11:30:00+05:30':
		return 2

	elapsed = dt2u - dt1
	with compiler.wrap_arithmetic:
		diff: f64 = elapsed.total_seconds() - 3600.0
		if diff < 0.0:
			diff = -diff
	if diff > 0.001:
		return 3

	if not ( dt1 < dt2u ):
		return 4
	if not ( dt1 == datetime( 2024, 6, 15, 10, 30, 0, tzinfo = tz ).unwrap( 'valid' ) ):
		return 5

	d = dt1.date()
	if d.isoformat() != '2024-06-15':
		return 6
	t = dt1.time()
	if t.isoformat() != '10:30:00':
		return 7

	# invalid construction rejected via Result.Err
	if datetime( 2024, 2, 30, tzinfo = tz ).is_ok():
		return 8
	if datetime( 2024, 1, 1, 25, 0, 0, tzinfo = tz ).is_ok():
		return 9

	# datetime - timedelta -> datetime (the __sub__ overload, mirroring
	# __add__ - also Result-returning, for the same overflow reason)
	dt3 = dt1 - timedelta( hours = 1 )
	dt3u: datetime = dt3.unwrap( 'valid' )
	if dt3u.isoformat() != '2024-06-15T09:30:00+05:30':
		return 10
	dt3b: datetime = ( dt1 - timedelta( minutes = 60 ) ).unwrap( 'valid' )
	if not ( dt3u == dt3b ):
		return 11

	# datetime - timedelta overflowing past year 9999 -> Err, matching
	# __add__'s own overflow handling (dt - timedelta(days=-1) is dt +
	# timedelta(days=1))
	r_edge = datetime( 9999, 12, 31, 23, 59, 59, tzinfo = tz )
	edge: datetime = r_edge.unwrap( 'valid' )
	if ( edge - timedelta( days = -1 ) ).is_ok():
		return 12

	# tz-defaulting: now()/fromtimestamp()/astimezone() with tz omitted use
	# the system's own local zone (localtz()), same result as passing it
	# explicitly
	explicit_tz = localtz()
	if datetime.fromtimestamp( 1_700_000_000.0 ) != datetime.fromtimestamp( 1_700_000_000.0, explicit_tz ):
		return 13
	if dt1.astimezone() != dt1.astimezone( explicit_tz ):
		return 14

	return 0
'''

# real IANA zone + windows_zones.install() (process-global state) - a real
# DST transition is needed to exercise the two-iteration offset-refinement
# fix (a single-shot "treat wall-clock as UTC" lookup gets the post-
# transition hour wrong for a zone several hours off UTC - found directly
# this session). Kept as its own executable, not merged with the cases above.
_DATETIME_DST_CROSSING_BEHAVIOR = '''
import compiler
import zoneinfo
import windows_zones
from datetime import datetime, timedelta

def main() -> i32:
	windows_zones.install()
	tz = zoneinfo.ZoneInfo( 'America/New_York' )
	utc = zoneinfo.ZoneInfo( 'UTC' )

	# 2024-03-10 spring-forward: 01:59:59 EST -> 03:00:00 EDT (02:00-03:00 skipped)
	r1 = datetime( 2024, 3, 10, 1, 59, 59, tzinfo = tz )
	dt1: datetime = r1.unwrap( 'valid' )
	r2 = datetime( 2024, 3, 10, 3, 0, 0, tzinfo = tz )
	dt2: datetime = r2.unwrap( 'valid' )
	if dt1.isoformat() != '2024-03-10T01:59:59-05:00':
		return 1
	if dt2.isoformat() != '2024-03-10T03:00:00-04:00':          # the case a single-shot lookup gets wrong
		return 2

	with compiler.wrap_arithmetic:
		diff1: f64 = dt1.timestamp() - 1710053999.0
		if diff1 < 0.0:
			diff1 = -diff1
		diff2: f64 = dt2.timestamp() - 1710054000.0
		if diff2 < 0.0:
			diff2 = -diff2
	if diff1 > 0.001:
		return 3
	if diff2 > 0.001:
		return 4

	# datetime + timedelta across the transition: wall-clock noon-to-noon is
	# 24h of WALL time but only 23h of REAL elapsed time (spring-forward) -
	# __sub__ always gives the real elapsed time (see its own docstring for
	# the deliberate divergence from CPython's same-tzinfo __sub__ quirk,
	# which would give 24h here).
	r3 = datetime( 2024, 3, 9, 12, 0, 0, tzinfo = tz )
	dt3: datetime = r3.unwrap( 'valid' )
	r3p = dt3 + timedelta( days = 1 )
	dt3_plus1: datetime = r3p.unwrap( 'valid' )
	if dt3_plus1.isoformat() != '2024-03-10T12:00:00-04:00':
		return 5
	elapsed = dt3_plus1 - dt3
	with compiler.wrap_arithmetic:
		ediff: f64 = elapsed.total_seconds() - 82800.0
		if ediff < 0.0:
			ediff = -ediff
	if ediff > 0.001:
		return 6

	# datetime - datetime across different tzinfo
	r_ny = datetime( 2024, 3, 10, 12, 0, 0, tzinfo = tz )
	dt_ny: datetime = r_ny.unwrap( 'valid' )
	r_u = datetime( 2024, 3, 10, 17, 0, 0, tzinfo = utc )
	dt_u: datetime = r_u.unwrap( 'valid' )
	cross = dt_u - dt_ny
	with compiler.wrap_arithmetic:
		cdiff: f64 = cross.total_seconds() - 3600.0
		if cdiff < 0.0:
			cdiff = -cdiff
	if cdiff > 0.001:
		return 7

	# astimezone round-trip
	dt_ny_as_utc = dt_ny.astimezone( utc )
	if dt_ny_as_utc.isoformat() != '2024-03-10T16:00:00+00:00':
		return 8

	dt_now = datetime.now( utc )
	if dt_now.year < 2024:
		return 9

	return 0
'''

# RC-lifetime/no-leak stress test: construct/destroy many datetime objects
# via the fallible __init__ path (both Ok and Err outcomes) in a loop - the
# fallible-construction RC-cleanup bugs found and fixed this session (task_
# 76bb84de, and the separate uninitialized-RC-field heap corruption,
# task_1ee1f43a) were both found via exactly this kind of repeated-
# construction pattern, not a single-shot check.
_DATETIME_RC_STRESS = '''
import compiler
import zoneinfo
from datetime import datetime

def main() -> i32:
	tz = zoneinfo.ZoneInfo.fixed_offset( 0, 'UTC' )
	i: i32 = 0
	while i < 2000:
		with compiler.panic_arithmetic( 'unreachable: divisor is a non-zero literal' ):
			day: i32 = ( i % 28 ) + 1
		r = datetime( 2024, 1, day, 12, 0, 0, tzinfo = tz )
		dt: datetime = r.unwrap( 'valid' )
		if dt.day != day:
			return 1

		r_bad = datetime( 2024, 2, 30, tzinfo = tz )
		if r_bad.is_ok():
			return 2

		with compiler.wrap_arithmetic:
			i += 1
	return 0
'''

# regression for a real crash: lib/datetime.py's localtz() used to cache the
# system zone via a bare `if __localtz is None: __localtz = ZoneInfo()`, no
# lock - safe for a single-threaded caller, but a genuine unsynchronized
# read-check-then-write data race once called concurrently from more than
# one OS thread on a fresh process (every thread sees None at once, each
# constructs its own ZoneInfo, all race to store into the same global).
# Found via a real thread-per-connection HTTP demo (lib/tcpserver.py's
# ThreadPerConnectionDispatcher) crashing under concurrent load - SIGILL,
# zero output - specifically on routes calling datetime.now(), only on a
# fresh process, only with enough concurrent first-touch callers; confirmed
# by a from-scratch trivial-handler variant of the same server never
# crashing across the same repro. Many threads racing into localtz()'s
# first call, all before it's cached, is exactly that window - fixed by
# guarding the check-then-set with threading.FastLock (see localtz()'s own
# docstring). Every worker must observe the SAME cached zone name - a
# corrupted/premature-free race would show up as a crash or a mismatched/
# garbage name long before this check.
_LOCALTZ_CONCURRENT_INIT_STRESS = '''
import compiler
import threading
import zoneinfo
import atomic
from datetime import localtz

_N: i32 = 150
_arrived: atomic.Atomic[i32] = atomic.Atomic[i32]( 0 )

class Worker:
	name: str
	def __init__( self ) -> None:
		self.name = ''
	def run( self ) -> None:
		# a real start barrier, not just "spawned close together" - every
		# thread spins here until all _N have arrived, so they all hit
		# localtz()'s first-touch check as close to simultaneously as
		# possible (a tight back-to-back spawn loop alone wasn't enough to
		# reliably hit the race window in practice)
		_arrived.fetch_add( 1 )
		while _arrived.load() < _N:
			pass
		tz: zoneinfo.ZoneInfo = localtz()
		self.name = tz.name

def main() -> i32:
	n: i32 = _N
	workers: list[Worker] = list[Worker]()
	i: i32 = 0
	while i < n:
		w: Worker = Worker()
		workers.append( w )
		with compiler.wrap_arithmetic:
			i += 1

	# spawned only after every Worker already exists, so CreateThread/
	# pthread_create calls run back-to-back with no other work between them -
	# maximizes how many threads race into localtz()'s first-touch window
	# together
	threads: list[threading.Thread] = list[threading.Thread]()
	j: usize = 0
	nw: usize = workers.__len__()
	while j < nw:
		w2: Worker = workers.__getitem__( j ).unwrap( 'index in bounds by construction' )
		threads.append( threading.Thread( w2.run ) )
		with compiler.wrap_arithmetic:
			j += 1

	k: usize = 0
	nt: usize = threads.__len__()
	while k < nt:
		th: threading.Thread = threads.__getitem__( k ).unwrap( 'index in bounds by construction' )
		th.join()
		with compiler.wrap_arithmetic:
			k += 1

	expected: str = localtz().name
	if expected.byte_len() == 0:
		return 1
	m: usize = 0
	while m < nw:
		w3: Worker = workers.__getitem__( m ).unwrap( 'index in bounds by construction' )
		if w3.name != expected:
			return 2
		with compiler.wrap_arithmetic:
			m += 1
	return 0
'''


@unittest.skipUnless( test_support.HAS_CC, 'no C compiler (clang/gcc/msvc) found - skipping real-compile datetime tests' )
class TimedeltaTests( RealCompileMixin, unittest.TestCase ):
	def test_timedelta_behavior( self ) -> None:
		self.assert_programs_run([ ( 'timedelta_behavior', _TIMEDELTA_BEHAVIOR ) ])


@unittest.skipUnless( test_support.HAS_CC, 'no C compiler (clang/gcc/msvc) found - skipping real-compile datetime tests' )
class DateTests( RealCompileMixin, unittest.TestCase ):
	def test_date_behavior( self ) -> None:
		self.assert_programs_run([ ( 'date_behavior', _DATE_BEHAVIOR ) ])


@unittest.skipUnless( test_support.HAS_CC, 'no C compiler (clang/gcc/msvc) found - skipping real-compile datetime tests' )
class TimeTests( RealCompileMixin, unittest.TestCase ):
	def test_time_behavior( self ) -> None:
		self.assert_programs_run([ ( 'time_behavior', _TIME_BEHAVIOR ) ])


@unittest.skipUnless( test_support.HAS_CC, 'no C compiler (clang/gcc/msvc) found - skipping real-compile datetime tests' )
class DatetimeTests( RealCompileMixin, unittest.TestCase ):
	def test_fixed_offset_behavior( self ) -> None:
		self.assert_programs_run([ ( 'datetime_fixed_offset', _DATETIME_FIXED_OFFSET_BEHAVIOR ) ])

	def test_dst_crossing_behavior( self ) -> None:
		# own executable: windows_zones.install() is process-global state,
		# must not be merged with other cases (see test_support.py's own
		# assert_programs_run docstring)
		self.assert_programs_run([ ( 'datetime_dst_crossing', _DATETIME_DST_CROSSING_BEHAVIOR ) ])

	def test_rc_lifetime_stress( self ) -> None:
		self.assert_programs_run([ ( 'datetime_rc_stress', _DATETIME_RC_STRESS ) ], timeout = 30.0 )

	def test_localtz_concurrent_init_stress( self ) -> None:
		# own executable: relies on __localtz being freshly-unset at process
		# start (see this case's own docstring above) - must not be merged
		# with other cases via assert_programs_run
		self.assert_programs_run([ ( 'localtz_concurrent_init_stress', _LOCALTZ_CONCURRENT_INIT_STRESS ) ], timeout = 30.0 )


if __name__ == '__main__':
	unittest.main()
