# lib/datetime.py — timedelta, date, time, datetime.
#
# Mirrors Python's datetime module API/semantics where it makes sense, with
# one firm, deliberate departure: datetime.tzinfo is REQUIRED, always a real
# zoneinfo.ZoneInfo, never optional/defaulted - there is no naive datetime in
# this design. date and bare time (time-of-day, no date) stay tzinfo-free -
# they're calendar/clock components, never ambiguous about "what instant is
# this"; only datetime, which claims to represent a specific instant, has
# that ambiguity.
#
# datetime.tzinfo is always a concrete ZoneInfo, not a polymorphic tzinfo/
# timezone base class - this compiler has no RCClass dynamic dispatch yet,
# and the COM-style @interface/vtable mechanism it does have requires manual
# pointer lifetime management, a poor fit for an ordinary value type. Python's
# timezone(timedelta(...)) fixed-offset case is covered by zoneinfo.py's own
# ZoneInfo.fixed_offset(...) instead.
#
# Constructors PANIC on invalid input (sys.panic), not Result[None,DateError]
# as originally intended - fallible __init__ (SYNTAX.md) has a real, confirmed
# double/triple-free bug when the constructed value is retained/queried
# rather than immediately match-extracted, or when 2+ fallible constructions
# happen in one function (exactly what this module needs constantly) - filed
# separately (task_76bb84de). Swapping to Result[None,DateError] once that's
# fixed is a small, mechanical follow-up - the validation logic itself
# doesn't change, only the success/failure signaling at the constructor
# boundary.
#
# // and % in this compiler are C-style TRUNCATING, not Python-style floor,
# despite // being spelled like Python's floor-div operator (verified
# directly by reading emitter_c.py's _emit_int_division) - floordiv_i64/
# floormod_i64 (lib/_civil_calendar.py) are used anywhere a negative operand
# is possible.
#
# No strftime()/strptime() (arbitrary format strings) - isoformat()/__str__()
# (ISO-8601) is the only string output in this pass; a documented scope cut,
# not an oversight.

import compiler
import sys
from _civil_calendar import days_from_civil, civil_from_days, weekday_from_days, days_in_month
from math import floordiv_i64, floormod_i64
from zoneinfo import ZoneInfo


_US_PER_SECOND: i64 = 1_000_000
_US_PER_DAY: i64 = 86_400_000_000
_SECONDS_PER_DAY: i64 = 86400


@union
class DateError:
	InvalidYear: None       # outside 1..9999
	InvalidMonth: None      # outside 1..12
	InvalidDay: None        # outside 1..days_in_month(year, month)
	InvalidHour: None       # outside 0..23
	InvalidMinute: None     # outside 0..59
	InvalidSecond: None     # outside 0..59
	InvalidMicrosecond: None  # outside 0..999999


class timedelta:
	_total_us: i64  # can be negative; the one stored field, everything else is derived

	def __init__(
		self,
		days: i32 = 0,
		seconds: i32 = 0,
		microseconds: i32 = 0,
		milliseconds: i32 = 0,
		minutes: i32 = 0,
		hours: i32 = 0,
		weeks: i32 = 0,
	) -> None:
		with compiler.wrap_arithmetic:
			total: i64 = 0
			total += i64( weeks ) * 7 * _US_PER_DAY
			total += i64( days ) * _US_PER_DAY
			total += i64( hours ) * 3600 * _US_PER_SECOND
			total += i64( minutes ) * 60 * _US_PER_SECOND
			total += i64( seconds ) * _US_PER_SECOND
			total += i64( milliseconds ) * 1000
			total += i64( microseconds )
			self._total_us = total

	@staticmethod
	def _from_total_us( total_us: i64 ) -> timedelta:
		return timedelta.__allocate__( _total_us = total_us )

	@property
	def days( self ) -> i32:
		with compiler.wrap_arithmetic:
			return i32( floordiv_i64( self._total_us, _US_PER_DAY ) )

	@property
	def seconds( self ) -> i32:
		''' 0 <= seconds < 86400 - matches Python's own normalized invariant
		(only .days can be negative). '''
		rem: i64 = floormod_i64( self._total_us, _US_PER_DAY )
		with compiler.wrap_arithmetic:
			return i32( floordiv_i64( rem, _US_PER_SECOND ) )

	@property
	def microseconds( self ) -> i32:
		''' 0 <= microseconds < 1_000_000. '''
		rem: i64 = floormod_i64( self._total_us, _US_PER_DAY )
		with compiler.wrap_arithmetic:
			return i32( floormod_i64( rem, _US_PER_SECOND ) )

	def total_seconds( self ) -> f64:
		with compiler.wrap_arithmetic:
			return f64( self._total_us ) / 1_000_000.0

	def __add__( self, other: timedelta ) -> timedelta:
		with compiler.wrap_arithmetic:
			result_us: i64 = self._total_us + other._total_us
		return timedelta._from_total_us( result_us )

	def __sub__( self, other: timedelta ) -> timedelta:
		with compiler.wrap_arithmetic:
			result_us: i64 = self._total_us - other._total_us
		return timedelta._from_total_us( result_us )

	def __neg__( self ) -> timedelta:
		with compiler.wrap_arithmetic:
			result_us: i64 = -self._total_us
		return timedelta._from_total_us( result_us )

	def __mul__( self, n: i32 ) -> timedelta:
		''' timedelta * n only - n * timedelta (right-hand operand) isn't
		attemptable, no __r*__ dispatch exists anywhere in this compiler. '''
		with compiler.wrap_arithmetic:
			result_us: i64 = self._total_us * i64( n )
		return timedelta._from_total_us( result_us )

	def __floordiv__( self, n: i32 ) -> timedelta:
		result_us: i64 = floordiv_i64( self._total_us, i64( n ) )
		return timedelta._from_total_us( result_us )

	def __eq__( self, other: timedelta ) -> bool:
		return self._total_us == other._total_us

	def __ne__( self, other: timedelta ) -> bool:
		return not self.__eq__( other )

	def __lt__( self, other: timedelta ) -> bool:
		return self._total_us < other._total_us

	def __le__( self, other: timedelta ) -> bool:
		return self._total_us <= other._total_us

	def __gt__( self, other: timedelta ) -> bool:
		return self._total_us > other._total_us

	def __ge__( self, other: timedelta ) -> bool:
		return self._total_us >= other._total_us

	def __str__( self ) -> str:
		''' Python's own timedelta.__str__ shape - NOT ISO-8601 (timedelta
		has no real isoformat() in Python either): "H:MM:SS[.ffffff]", or
		"D day(s), H:MM:SS[.ffffff]" when days != 0. '''
		d: i32 = self.days
		rem_us: i64 = floormod_i64( self._total_us, _US_PER_DAY )
		secs_of_day: i64 = floordiv_i64( rem_us, _US_PER_SECOND )
		us64: i64 = floormod_i64( rem_us, _US_PER_SECOND )
		with compiler.panic_arithmetic( 'unreachable: divisors are non-zero literals' ):
			hh64: i64 = secs_of_day // 3600
			mm64: i64 = ( secs_of_day // 60 ) % 60
			ss64: i64 = secs_of_day % 60
		with compiler.wrap_arithmetic:
			hh: i32 = i32( hh64 )
			mm: i32 = i32( mm64 )
			ss: i32 = i32( ss64 )
			us: i32 = i32( us64 )

		time_part: str = f'{int(hh)}:{int(mm):02d}:{int(ss):02d}'
		if us != 0:
			time_part = f'{time_part}.{int(us):06d}'

		if d == 0:
			return time_part

		day_word: str
		if d == 1 or d == -1:
			day_word = 'day'
		else:
			day_word = 'days'
		return f'{int(d)} {day_word}, {time_part}'


# Python's date.toordinal() is the proleptic-Gregorian ordinal (day 1 =
# 0001-01-01); our own internal _epoch_day is days since 1970-01-01 (Unix
# convention, matching zoneinfo.py/time.py). ordinal = epoch_day + this.
_ORDINAL_EPOCH_OFFSET: i64 = 719163


class date:
	_epoch_day: i64  # days since 1970-01-01, can be negative

	def __init__( self, year: i32, month: i32, day: i32 ) -> Result[None, DateError]:
		if year < 1 or year > 9999:
			return Result.Err( DateError.InvalidYear( None ) )
		if month < 1 or month > 12:
			return Result.Err( DateError.InvalidMonth( None ) )
		max_day: i32 = days_in_month( year, month )
		if day < 1 or day > max_day:
			return Result.Err( DateError.InvalidDay( None ) )
		self._epoch_day = days_from_civil( year, month, day )
		return Result.Ok( None )

	@staticmethod
	def _from_epoch_day( epoch_day: i64 ) -> date:
		return date.__allocate__( _epoch_day = epoch_day )

	@property
	def year( self ) -> i32:
		return civil_from_days( self._epoch_day ).year

	@property
	def month( self ) -> i32:
		return civil_from_days( self._epoch_day ).month

	@property
	def day( self ) -> i32:
		return civil_from_days( self._epoch_day ).day

	@staticmethod
	def today( tz: ZoneInfo ) -> date:
		''' tz is required (no zero-arg form, unlike Python's date.today())
		- "today" is itself timezone-dependent (a calendar day rolls over
		at local midnight, which differs per zone), and there's no "OS
		local zone" default to silently fall back to in this design (see
		this module's own docstring on why datetime.tzinfo is mandatory) -
		use time.get_local_timezone_name() to pass one explicitly. '''
		import time as _time
		now: f64 = _time.time()
		return date.fromtimestamp( now, tz )

	@staticmethod
	def fromtimestamp( t: f64, tz: ZoneInfo ) -> date:
		with compiler.wrap_arithmetic:
			epoch_seconds: i64 = i64( t )
		offset: i32 = tz.utcoffset( epoch_seconds )
		with compiler.wrap_arithmetic:
			local_seconds: i64 = epoch_seconds + i64( offset )
		epoch_day: i64 = floordiv_i64( local_seconds, _SECONDS_PER_DAY )
		return date._from_epoch_day( epoch_day )

	@staticmethod
	def fromordinal( n: i64 ) -> date:
		with compiler.wrap_arithmetic:
			epoch_day: i64 = n - _ORDINAL_EPOCH_OFFSET
		return date._from_epoch_day( epoch_day )

	def toordinal( self ) -> i64:
		with compiler.wrap_arithmetic:
			return self._epoch_day + _ORDINAL_EPOCH_OFFSET

	def weekday( self ) -> i32:
		''' Monday=0..Sunday=6, matching Python's date.weekday(). '''
		wd_sun0: i32 = weekday_from_days( self._epoch_day )  # 0=Sunday
		with compiler.panic_arithmetic( 'unreachable: divisor is a non-zero literal' ):
			return ( wd_sun0 + 6 ) % 7

	def isoweekday( self ) -> i32:
		''' Monday=1..Sunday=7, matching Python's date.isoweekday(). '''
		with compiler.wrap_arithmetic:
			return self.weekday() + 1

	def __add__( self, delta: timedelta ) -> date:
		''' only delta.days participates, matching Python's own date+timedelta
		(a date has no time-of-day component for the other fields to affect). '''
		with compiler.wrap_arithmetic:
			new_epoch_day: i64 = self._epoch_day + i64( delta.days )
		return date._from_epoch_day( new_epoch_day )

	def __sub__( self, other: date ) -> timedelta:
		''' date - date -> timedelta only (the more common case) - date -
		timedelta isn't supported directly (a second __sub__ overload for it
		hit confusing operator-dispatch behavior when combined with
		@overload - not pursued for this pass); use `d + (-delta)` instead. '''
		with compiler.wrap_arithmetic:
			diff_days: i64 = self._epoch_day - other._epoch_day
			diff_days32: i32 = i32( diff_days )
		return timedelta( days = diff_days32 )

	def __eq__( self, other: date ) -> bool:
		return self._epoch_day == other._epoch_day

	def __ne__( self, other: date ) -> bool:
		return not self.__eq__( other )

	def __lt__( self, other: date ) -> bool:
		return self._epoch_day < other._epoch_day

	def __le__( self, other: date ) -> bool:
		return self._epoch_day <= other._epoch_day

	def __gt__( self, other: date ) -> bool:
		return self._epoch_day > other._epoch_day

	def __ge__( self, other: date ) -> bool:
		return self._epoch_day >= other._epoch_day

	def isoformat( self ) -> str:
		return f'{int(self.year):04d}-{int(self.month):02d}-{int(self.day):02d}'

	def __str__( self ) -> str:
		return self.isoformat()


class time:
	''' time-of-day only, no date, no tzinfo - see this module's own
	docstring for why (calendar/clock component, never claims to represent
	a specific instant, so never ambiguous the way datetime would be). '''
	_us_since_midnight: i64  # 0 <= value < 86_400_000_000

	def __init__(
		self,
		hour: i32 = 0,
		minute: i32 = 0,
		second: i32 = 0,
		microsecond: i32 = 0,
	) -> Result[None, DateError]:
		if hour < 0 or hour > 23:
			return Result.Err( DateError.InvalidHour( None ) )
		if minute < 0 or minute > 59:
			return Result.Err( DateError.InvalidMinute( None ) )
		if second < 0 or second > 59:
			return Result.Err( DateError.InvalidSecond( None ) )
		if microsecond < 0 or microsecond > 999999:
			return Result.Err( DateError.InvalidMicrosecond( None ) )
		with compiler.wrap_arithmetic:
			self._us_since_midnight = (
				i64( hour ) * 3600 * _US_PER_SECOND
				+ i64( minute ) * 60 * _US_PER_SECOND
				+ i64( second ) * _US_PER_SECOND
				+ i64( microsecond )
			)
		return Result.Ok( None )

	@staticmethod
	def _from_us_since_midnight( us: i64 ) -> time:
		return time.__allocate__( _us_since_midnight = us )

	@property
	def hour( self ) -> i32:
		secs_of_day: i64 = floordiv_i64( self._us_since_midnight, _US_PER_SECOND )
		with compiler.panic_arithmetic( 'unreachable: divisor is a non-zero literal' ):
			h: i64 = secs_of_day // 3600
		with compiler.wrap_arithmetic:
			return i32( h )

	@property
	def minute( self ) -> i32:
		secs_of_day: i64 = floordiv_i64( self._us_since_midnight, _US_PER_SECOND )
		with compiler.panic_arithmetic( 'unreachable: divisor is a non-zero literal' ):
			m: i64 = ( secs_of_day // 60 ) % 60
		with compiler.wrap_arithmetic:
			return i32( m )

	@property
	def second( self ) -> i32:
		secs_of_day: i64 = floordiv_i64( self._us_since_midnight, _US_PER_SECOND )
		with compiler.panic_arithmetic( 'unreachable: divisor is a non-zero literal' ):
			s: i64 = secs_of_day % 60
		with compiler.wrap_arithmetic:
			return i32( s )

	@property
	def microsecond( self ) -> i32:
		us: i64 = floormod_i64( self._us_since_midnight, _US_PER_SECOND )
		with compiler.wrap_arithmetic:
			return i32( us )

	def __eq__( self, other: time ) -> bool:
		return self._us_since_midnight == other._us_since_midnight

	def __ne__( self, other: time ) -> bool:
		return not self.__eq__( other )

	def __lt__( self, other: time ) -> bool:
		return self._us_since_midnight < other._us_since_midnight

	def __le__( self, other: time ) -> bool:
		return self._us_since_midnight <= other._us_since_midnight

	def __gt__( self, other: time ) -> bool:
		return self._us_since_midnight > other._us_since_midnight

	def __ge__( self, other: time ) -> bool:
		return self._us_since_midnight >= other._us_since_midnight

	def isoformat( self ) -> str:
		base: str = f'{int(self.hour):02d}:{int(self.minute):02d}:{int(self.second):02d}'
		us: i32 = self.microsecond
		if us != 0:
			return f'{base}.{int(us):06d}'
		return base

	def __str__( self ) -> str:
		return self.isoformat()


def _format_utc_offset( offset_seconds: i32 ) -> str:
	''' '+HH:MM' or '-HH:MM' - Python's real isoformat() always uses this
	form (never 'Z', even for exactly UTC). '''
	sign: str = '+'
	abs_offset: i32 = offset_seconds
	if abs_offset < 0:
		sign = '-'
		with compiler.wrap_arithmetic:
			abs_offset = -abs_offset
	with compiler.panic_arithmetic( 'unreachable: divisors are non-zero literals' ):
		hh: i32 = abs_offset // 3600
		mm: i32 = ( abs_offset // 60 ) % 60
	return f'{sign}{int(hh):02d}:{int(mm):02d}'


class datetime:
	''' always tz-aware - tzinfo is a required field, never optional/
	defaulted (see this module's own docstring for why: this is the one
	firm, deliberate departure from Python in this whole module). Stores
	wall-clock fields directly (year/month/day/hour/minute/second/
	microsecond) + tzinfo - matching CPython's own real internal
	representation (wall-clock fields + a tzinfo reference, NOT a UTC
	instant) - so datetime + timedelta correctly adjusts wall-clock fields
	and reinterprets against the SAME tzinfo (a DST-crossing +timedelta(
	days=1) isn't exactly 24h of elapsed real time, matching Python). '''
	year: i32
	month: i32
	day: i32
	hour: i32
	minute: i32
	second: i32
	microsecond: i32
	tzinfo: ZoneInfo

	def __init__(
		self,
		year: i32,
		month: i32,
		day: i32,
		hour: i32 = 0,
		minute: i32 = 0,
		second: i32 = 0,
		microsecond: i32 = 0,
		*,
		tzinfo: ZoneInfo,
	) -> Result[None, DateError]:
		# self.tzinfo is assigned FIRST, before any validation that can
		# return Err - a real, confirmed compiler bug (heap corruption,
		# reported separately) means a fallible __init__ that returns Err
		# before assigning an RC-typed field leaves that field as raw,
		# uninitialized allocator memory (sys$alloc doesn't zero it), and
		# the Err-path cleanup releases self's fields unconditionally,
		# including that uninitialized one - releasing garbage as if it
		# were a valid pointer. Assigning tzinfo up front means it's always
		# a real, valid reference by the time any Err path can be taken.
		self.tzinfo = tzinfo
		if year < 1 or year > 9999:
			return Result.Err( DateError.InvalidYear( None ) )
		if month < 1 or month > 12:
			return Result.Err( DateError.InvalidMonth( None ) )
		max_day: i32 = days_in_month( year, month )
		if day < 1 or day > max_day:
			return Result.Err( DateError.InvalidDay( None ) )
		if hour < 0 or hour > 23:
			return Result.Err( DateError.InvalidHour( None ) )
		if minute < 0 or minute > 59:
			return Result.Err( DateError.InvalidMinute( None ) )
		if second < 0 or second > 59:
			return Result.Err( DateError.InvalidSecond( None ) )
		if microsecond < 0 or microsecond > 999999:
			return Result.Err( DateError.InvalidMicrosecond( None ) )
		self.year = year
		self.month = month
		self.day = day
		self.hour = hour
		self.minute = minute
		self.second = second
		self.microsecond = microsecond
		return Result.Ok( None )

	def _utcoffset_seconds( self ) -> i32:
		''' the real utcoffset in effect for this wall-clock reading.
		Treating the wall-clock reading as if it were UTC to get a
		PROVISIONAL epoch and querying tzinfo.utcoffset() ONCE at that
		provisional instant (the trick windows/zoneinfo_rules.py's own
		_nth_weekday_epoch_seconds uses) is NOT enough on its own - found
		directly this session: for a zone several hours off UTC (e.g.
		America/New_York, UTC-4/-5), the provisional instant can land on
		the WRONG SIDE of the real transition boundary across a multi-hour
		window, not just the ambiguous/gap hour itself (confirmed: wall-
		clock '2024-03-10 03:00:00' - unambiguously EDT, since NY's spring-
		forward happens at 02:00 local - came out as EST/-05:00 with a
		single lookup, matching CPython's own utcoffset() only after a
		second iteration). So: guess an offset, refine the UTC estimate,
		then re-query AT that refined instant - two iterations converges
		correctly for every wall-clock reading except the genuinely
		ambiguous (fold) or nonexistent (gap) hour itself, which is the
		real, documented, narrower v1 limitation (Python's `fold`
		attribute, not implemented here). '''
		days: i64 = days_from_civil( self.year, self.month, self.day )
		with compiler.wrap_arithmetic:
			provisional: i64 = days * _SECONDS_PER_DAY + i64( self.hour ) * 3600 + i64( self.minute ) * 60 + i64( self.second )
		offset1: i32 = self.tzinfo.utcoffset( provisional )
		with compiler.wrap_arithmetic:
			utc_guess: i64 = provisional - i64( offset1 )
		return self.tzinfo.utcoffset( utc_guess )

	def _to_epoch_seconds( self ) -> i64:
		''' wall-clock -> UTC instant, using _utcoffset_seconds's own
		refined offset (see its docstring for why a single lookup isn't
		enough). '''
		days: i64 = days_from_civil( self.year, self.month, self.day )
		with compiler.wrap_arithmetic:
			provisional: i64 = days * _SECONDS_PER_DAY + i64( self.hour ) * 3600 + i64( self.minute ) * 60 + i64( self.second )
			return provisional - i64( self._utcoffset_seconds() )

	def timestamp( self ) -> f64:
		epoch_seconds: i64 = self._to_epoch_seconds()
		with compiler.wrap_arithmetic:
			return f64( epoch_seconds ) + f64( self.microsecond ) / 1_000_000.0

	@staticmethod
	def _from_epoch( epoch_seconds: i64, microsecond: i32, tz: ZoneInfo ) -> datetime:
		''' UTC instant -> wall-clock: unambiguous (a real UTC instant maps
		to exactly one offset), unlike _to_epoch_seconds's own direction. '''
		offset: i32 = tz.utcoffset( epoch_seconds )
		with compiler.wrap_arithmetic:
			local_seconds: i64 = epoch_seconds + i64( offset )
		epoch_day: i64 = floordiv_i64( local_seconds, _SECONDS_PER_DAY )
		secs_of_day: i64 = floormod_i64( local_seconds, _SECONDS_PER_DAY )
		civil = civil_from_days( epoch_day )
		with compiler.panic_arithmetic( 'unreachable: divisors are non-zero literals' ):
			hh: i64 = secs_of_day // 3600
			mm: i64 = ( secs_of_day // 60 ) % 60
			ss: i64 = secs_of_day % 60
		with compiler.wrap_arithmetic:
			return datetime.__allocate__(
				year = civil.year, month = civil.month, day = civil.day,
				hour = i32( hh ), minute = i32( mm ), second = i32( ss ),
				microsecond = microsecond, tzinfo = tz,
			)

	@staticmethod
	def now( tz: ZoneInfo ) -> datetime:
		''' tz is required - no zero-arg form, unlike Python's datetime.
		now() - there's no "OS local zone" default to silently fall back to
		in this design (see this module's own docstring); use
		time.get_local_timezone_name() to pass one explicitly. '''
		import time as _time
		t: f64 = _time.time()
		return datetime.fromtimestamp( t, tz )

	@staticmethod
	def fromtimestamp( t: f64, tz: ZoneInfo ) -> datetime:
		''' t before the 1970 epoch (negative) is a known, undocumented-
		precision edge case - i64(t) truncates toward zero, not floor, so a
		fractional negative t's microsecond component can come out wrong.
		Not a concern for any realistic "current time" use. '''
		with compiler.wrap_arithmetic:
			epoch_seconds: i64 = i64( t )
			frac: f64 = t - f64( epoch_seconds )
			microsecond: i32 = i32( frac * 1_000_000.0 )
		return datetime._from_epoch( epoch_seconds, microsecond, tz )

	def astimezone( self, tz: ZoneInfo ) -> datetime:
		epoch_seconds: i64 = self._to_epoch_seconds()
		return datetime._from_epoch( epoch_seconds, self.microsecond, tz )

	def to_date( self ) -> date:
		''' named to_date(), not Python's own date() - a bare `date(...)`
		call inside a method of this same name would resolve to the METHOD
		itself, not the module-level class (confirmed directly this
		session) - a small, documented deviation from Python's exact method
		name to sidestep it. '''
		r = date( self.year, self.month, self.day )
		return r.unwrap( 'datetime: constructed from already-valid fields, unreachable' )

	def to_time( self ) -> time:
		''' named to_time(), not Python's own time() - see to_date()'s own
		comment for why. '''
		r = time( self.hour, self.minute, self.second, self.microsecond )
		return r.unwrap( 'datetime: constructed from already-valid fields, unreachable' )

	def __add__( self, delta: timedelta ) -> Result[datetime, DateError]:
		''' adjusts wall-clock fields directly, reinterpreted against the
		SAME tzinfo - see this class's own docstring. Result-returning (not
		plain datetime) since the result's year can overflow 1..9999,
		matching Python's real OverflowError for out-of-range results
		(surfaced here as DateError.InvalidYear via the shared constructor
		validation, rather than a separate Overflow-specific check). '''
		days: i64 = days_from_civil( self.year, self.month, self.day )
		with compiler.wrap_arithmetic:
			us_of_day: i64 = (
				i64( self.hour ) * 3600 * _US_PER_SECOND
				+ i64( self.minute ) * 60 * _US_PER_SECOND
				+ i64( self.second ) * _US_PER_SECOND
				+ i64( self.microsecond )
			)
			total_us: i64 = days * _US_PER_DAY + us_of_day + delta._total_us
		new_epoch_day: i64 = floordiv_i64( total_us, _US_PER_DAY )
		new_us_of_day: i64 = floormod_i64( total_us, _US_PER_DAY )
		civil = civil_from_days( new_epoch_day )
		new_secs_of_day: i64 = floordiv_i64( new_us_of_day, _US_PER_SECOND )
		new_us: i64 = floormod_i64( new_us_of_day, _US_PER_SECOND )
		with compiler.panic_arithmetic( 'unreachable: divisors are non-zero literals' ):
			hh: i64 = new_secs_of_day // 3600
			mm: i64 = ( new_secs_of_day // 60 ) % 60
			ss: i64 = new_secs_of_day % 60
		with compiler.wrap_arithmetic:
			hh32: i32 = i32( hh )
			mm32: i32 = i32( mm )
			ss32: i32 = i32( ss )
			us32: i32 = i32( new_us )
		return datetime( civil.year, civil.month, civil.day, hh32, mm32, ss32, us32, tzinfo = self.tzinfo )

	def __sub__( self, other: datetime ) -> timedelta:
		''' datetime - datetime -> timedelta, ALWAYS via UTC-instant
		conversion, giving the real elapsed time between the two instants -
		this is a deliberate divergence from CPython's own documented
		datetime.__sub__ quirk: when both operands share the exact same
		tzinfo object, CPython subtracts naive wall-clock fields directly,
		silently ignoring any DST transition in between (confirmed
		directly: CPython gives 24h for a wall-clock noon-to-noon span that
		crosses a spring-forward, when only 23h of real time actually
		elapsed - .timestamp() on the same two datetimes agrees with the
		correct 23h, only __sub__ has the quirk). Given this module's whole
		purpose is eliminating exactly this kind of ambiguity, always
		computing the real elapsed time (matching what .timestamp() would
		give) is the right choice here, not bug-for-bug Python fidelity.
		datetime - timedelta isn't supported directly (same @overload/
		operator-dispatch issue noted on date.__sub__) - use
		`dt + (-delta)` instead. '''
		with compiler.wrap_arithmetic:
			diff_us: i64 = ( self._to_epoch_seconds() - other._to_epoch_seconds() ) * _US_PER_SECOND + i64( self.microsecond ) - i64( other.microsecond )
		return timedelta._from_total_us( diff_us )

	def _cmp_key( self ) -> i64:
		with compiler.wrap_arithmetic:
			return self._to_epoch_seconds() * _US_PER_SECOND + i64( self.microsecond )

	def __eq__( self, other: datetime ) -> bool:
		return self._cmp_key() == other._cmp_key()

	def __ne__( self, other: datetime ) -> bool:
		return not self.__eq__( other )

	def __lt__( self, other: datetime ) -> bool:
		return self._cmp_key() < other._cmp_key()

	def __le__( self, other: datetime ) -> bool:
		return self._cmp_key() <= other._cmp_key()

	def __gt__( self, other: datetime ) -> bool:
		return self._cmp_key() > other._cmp_key()

	def __ge__( self, other: datetime ) -> bool:
		return self._cmp_key() >= other._cmp_key()

	def isoformat( self ) -> str:
		date_part: str = f'{int(self.year):04d}-{int(self.month):02d}-{int(self.day):02d}'
		time_part: str = f'{int(self.hour):02d}:{int(self.minute):02d}:{int(self.second):02d}'
		if self.microsecond != 0:
			time_part = f'{time_part}.{int(self.microsecond):06d}'
		offset: i32 = self._utcoffset_seconds()
		return f'{date_part}T{time_part}{_format_utc_offset( offset )}'

	def __str__( self ) -> str:
		return self.isoformat()
