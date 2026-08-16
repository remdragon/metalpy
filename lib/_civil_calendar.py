# lib/_civil_calendar.py — shared Gregorian calendar math.
#
# Lifted and generalized out of windows/zoneinfo_rules.py's own private
# _is_leap_year/_days_in_month/_days_from_civil (that file's copy only ever
# needs a small +/-2-year window around "now", by its own comment - fine
# there, not for a general date/datetime class needing Python's real
# MINYEAR=1..MAXYEAR=9999 range). Adds the missing inverse (civil_from_days)
# and the floor-division/floor-modulo helpers real negative-operand callers
# need - // and % in this compiler are C-style TRUNCATING, not Python-style
# floor, despite // being spelled like Python's floor-div operator (verified
# directly by reading emitter_c.py's _emit_int_division, which emits raw C
# `/`/`%`).
#
# Both days_from_civil/civil_from_days are Howard Hinnant's well-known,
# widely-used public-domain algorithms (howardhinnant.github.io/date_algorithms.html),
# days counted from 1970-01-01 (day 0) - matching this codebase's own
# Unix-epoch-centric convention (time.py, zoneinfo.py) rather than Python's
# own proleptic-Gregorian ordinal (day 1 = 0001-01-01).

import compiler

def floordiv_i64( a: i64, b: i64 ) -> i64:
	''' Python-style floor division (rounds toward -infinity), unlike this
	compiler's own // (rounds toward zero, C-style). Only ever called here
	with a non-zero, non-(-1) literal-shaped divisor - panic_arithmetic
	turns the impossible ZeroDivisionError/OverflowError cases this compiler
	would otherwise force a Result for into what they actually are: an
	unreachable condition, not a real failure mode any real caller hits. '''
	with compiler.panic_arithmetic( 'unreachable: floordiv_i64 divisor is never zero in this codebase' ):
		q: i64 = a // b
		r: i64 = a % b
		if r != 0 and ( r < 0 ) != ( b < 0 ):
			q -= 1
		return q

def floormod_i64( a: i64, b: i64 ) -> i64:
	''' Python-style floor modulo (result always has the same sign as b),
	unlike this compiler's own % (sign follows a, C-style). '''
	with compiler.panic_arithmetic( 'unreachable: floormod_i64 divisor is never zero in this codebase' ):
		r: i64 = a % b
		if r != 0 and ( r < 0 ) != ( b < 0 ):
			r += b
		return r


def is_leap_year( year: i32 ) -> bool:
	with compiler.panic_arithmetic( 'unreachable: divisors are non-zero literals' ):
		if year % 4 != 0:
			return False
		if year % 100 != 0:
			return True
		return year % 400 == 0


def days_in_month( year: i32, month: i32 ) -> i32:
	if month == 1 or month == 3 or month == 5 or month == 7 or month == 8 or month == 10 or month == 12:
		return 31
	if month == 4 or month == 6 or month == 9 or month == 11:
		return 30
	if is_leap_year( year ):
		return 29
	return 28


def days_from_civil( year: i32, month: i32, day: i32 ) -> i64:
	''' Howard Hinnant's days_from_civil - days since 1970-01-01. Valid for
	any year (not just a narrow current-era window): the one division that
	genuinely needs floor semantics for a negative operand (era, when y-1
	goes negative for an early year) already goes through the compiler's
	own truncating // safely here because y is only ever negative when the
	whole expression is still handled correctly by the era/yoe split below -
	same shape Hinnant's own reference implementation uses, which relies on
    ordinary (truncating, in C) division being valid ONLY because floor and
	truncating division agree for the specific non-negative-after-offset
	values this algorithm's own internal terms are guaranteed to produce. '''
	with compiler.panic_arithmetic( 'unreachable: divisors are non-zero literals' ):
		y: i64 = i64( year )
		if month <= 2:
			y -= 1
		era: i64 = floordiv_i64( y, 400 )
		yoe: i64 = y - era * 400
		mp: i64 = ( i64( month ) + 9 ) % 12
		doy: i64 = ( 153 * mp + 2 ) // 5 + i64( day ) - 1
		doe: i64 = yoe * 365 + yoe // 4 - yoe // 100 + doy
		return era * 146097 + doe - 719468


class CivilDate:
	year: i32
	month: i32
	day: i32
	def __init__( self, year: i32, month: i32, day: i32 ) -> None:
		self.year = year
		self.month = month
		self.day = day


def civil_from_days( z: i64 ) -> CivilDate:
	''' Howard Hinnant's civil_from_days - the inverse of days_from_civil.
	zz = z + 719468 is non-negative for every year in date's supported range
	(1..9999) and quite a bit beyond it, so only the outer era division uses
	floordiv_i64 defensively (protects a future caller passing a genuinely
	out-of-range z) - every division after that operates on values already
	proven non-negative by construction, where plain truncating // already
	agrees with floor division, matching Hinnant's own reference form. '''
	with compiler.panic_arithmetic( 'unreachable: divisors are non-zero literals' ):
		zz: i64 = z + 719468
		era: i64 = floordiv_i64( zz, 146097 )
		doe: i64 = zz - era * 146097                                       # [0, 146096]
		yoe: i64 = ( doe - doe // 1460 + doe // 36524 - doe // 146096 ) // 365  # [0, 399]
		y: i64 = yoe + era * 400
		doy: i64 = doe - ( 365 * yoe + yoe // 4 - yoe // 100 )              # [0, 365]
		mp: i64 = ( 5 * doy + 2 ) // 153                                    # [0, 11]
		d: i64 = doy - ( 153 * mp + 2 ) // 5 + 1                            # [1, 31]
		m: i64
		if mp < 10:
			m = mp + 3
		else:
			m = mp - 9
		year: i64 = y
		if m <= 2:
			year += 1
		return CivilDate( year = i32( year ), month = i32( m ), day = i32( d ) )


def weekday_from_days( days: i64 ) -> i32:
	''' 0=Sunday..6=Saturday (Python's date.weekday()/Windows' own
	SYSTEMTIME.wDayOfWeek convention) - 1970-01-01 (day 0) was a Thursday. '''
	with compiler.wrap_arithmetic:
		shifted: i64 = days + 4
	wd: i64 = floormod_i64( shifted, 7 )
	with compiler.wrap_arithmetic:
		return i32( wd )
