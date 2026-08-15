# src/windows/zoneinfo_rules.py
#
# Windows has no system tzdata files (unlike POSIX's /usr/share/zoneinfo -
# see posix/zoneinfo_rules.py), so this asks the OS directly instead: find
# the named zone's DYNAMIC_TIME_ZONE_INFORMATION via EnumDynamicTimeZone-
# Information, then call GetTimeZoneInformationForYear for a small window of
# years around "now" (no calendar/datetime module exists yet to do anything
# date-range-aware, and Windows' own recurring-rule data seldom reflects
# distant-past/future rule changes anyway - a bounded, current-era window is
# the honest scope here, not an oversight). No embedding, no network - both
# APIs are the real, officially documented way to ask Windows for this
# (verified against Microsoft's own docs this session, not from memory).
#
# key resolution: if lib/windows_zones.py's opt-in IANA<->Windows table was
# installed AND resolves `key`, that Windows name is used; otherwise `key`
# is tried directly as a native Windows zone name (e.g. 'Eastern Standard
# Time') - matches the user-facing contract documented in lib/zoneinfo.py.

import compiler
import sys
from zoneinfo import ZoneInfo, TTInfo
from windows.time import _decode_ascii_utf16z, _field_ptr_u16

_YEAR_WINDOW: i32 = 2  # build transitions for current_year +/- this many years


def load_zone( zone: ZoneInfo, key: str ) -> None:
	from windows.kernel32 import DynamicTimeZoneInformation

	win_name: str = _resolve_windows_name( key )
	dtzi_raw: Ptr[u8] = _find_zone_by_key_name( win_name )
	dtzi_ptr = compiler.cast( Ptr[DynamicTimeZoneInformation], dtzi_raw )

	current_year: i32 = _get_current_year()
	with compiler.wrap_arithmetic:
		start_year: i32 = current_year - _YEAR_WINDOW
		end_year: i32 = current_year + _YEAR_WINDOW

	transition_times: list[i64] = []
	transition_rules: list[TTInfo] = []

	# start_year <= end_year always (_YEAR_WINDOW >= 0), so this always runs
	# at least once - a real TTInfo from the start, no TTInfo|None tracking
	# needed (assigning a narrowed `if x is not None: field = x` value into
	# an object field didn't propagate the narrowing - found directly this
	# session - simplest fix is just not needing that pattern at all).
	first_year_tzi: Ptr[u8] = _get_tzi_for_year( dtzi_ptr, start_year )
	first_year_rules: _YearRules = _rules_for_year( first_year_tzi, start_year )
	sys.free( first_year_tzi )
	zone.default_rule = first_year_rules.std_rule

	year: i32 = start_year
	while year <= end_year:
		rules: _YearRules
		if year == start_year:
			rules = first_year_rules
		else:
			tzi_raw: Ptr[u8] = _get_tzi_for_year( dtzi_ptr, year )
			rules = _rules_for_year( tzi_raw, year )
			sys.free( tzi_raw )

		if rules.has_dst:
			if rules.to_dst_epoch < rules.to_std_epoch:
				transition_times.append( rules.to_dst_epoch ).unwrap( 'zoneinfo: append failed' )
				transition_rules.append( rules.dst_rule ).unwrap( 'zoneinfo: append failed' )
				transition_times.append( rules.to_std_epoch ).unwrap( 'zoneinfo: append failed' )
				transition_rules.append( rules.std_rule ).unwrap( 'zoneinfo: append failed' )
			else:
				transition_times.append( rules.to_std_epoch ).unwrap( 'zoneinfo: append failed' )
				transition_rules.append( rules.std_rule ).unwrap( 'zoneinfo: append failed' )
				transition_times.append( rules.to_dst_epoch ).unwrap( 'zoneinfo: append failed' )
				transition_rules.append( rules.dst_rule ).unwrap( 'zoneinfo: append failed' )

		with compiler.wrap_arithmetic:
			year += 1

	zone.transition_times = transition_times
	zone.transition_rules = transition_rules

	sys.free( dtzi_raw )


def _resolve_windows_name( key: str ) -> str:
	import windows_zones
	win_name: str|None = windows_zones.to_windows( key )
	if win_name is not None:
		return win_name
	return key


def _get_current_year() -> i32:
	from windows.kernel32 import SYSTEMTIME, GetSystemTime
	st = SYSTEMTIME()
	GetSystemTime( compiler.addrof( st ))
	with compiler.wrap_arithmetic:
		return i32( st.wYear )


def _find_zone_by_key_name( win_name: str ) -> Ptr[u8]:
	''' scans EnumDynamicTimeZoneInformation(0), (1), ... until TimeZoneKeyName
	matches win_name, or panics if none does. Returns the raw heap buffer
	(caller frees it) holding the MATCHING entry's DynamicTimeZoneInformation -
	reusing the same scratch buffer across iterations until a match is
	found, so only one allocation is made regardless of how many entries are
	scanned. '''
	from windows.kernel32 import DynamicTimeZoneInformation, _TZKEYNAME_SIZE, _TZKEYNAME_OFFSET
	from windows.advapi32 import EnumDynamicTimeZoneInformation, ERROR_SUCCESS, ERROR_NO_MORE_ITEMS

	struct_size: usize = compiler.sizeof( DynamicTimeZoneInformation )
	raw: Ptr[u8] = sys.alloc[u8]( struct_size )

	index: u32 = 0
	# ~150 real Windows zones exist at this writing - 4096 is a generous
	# bound against an ever-growing-but-never-terminating loop, not a real
	# expected count.
	while index < 4096:
		sys.memzero( raw, struct_size )
		dtzi_ptr = compiler.cast( Ptr[DynamicTimeZoneInformation], raw )
		status: u32 = EnumDynamicTimeZoneInformation( index, dtzi_ptr )
		if status == ERROR_NO_MORE_ITEMS:
			sys.free( raw )
			sys.panic( 'zoneinfo: unknown Windows time zone name: ' + win_name )
		if status != ERROR_SUCCESS:
			sys.free( raw )
			sys.panic( 'zoneinfo: EnumDynamicTimeZoneInformation failed' )

		opaque_ptr: Ptr[None] = compiler.cast( Ptr[None], dtzi_ptr )
		key_ptr: Ptr[u16] = _field_ptr_u16( opaque_ptr, _TZKEYNAME_OFFSET )
		candidate: str = _decode_ascii_utf16z( key_ptr, _TZKEYNAME_SIZE )
		if candidate == win_name:
			return raw

		with compiler.wrap_arithmetic:
			index += 1

	sys.free( raw )
	sys.panic( 'zoneinfo: unknown Windows time zone name (scanned 4096 entries): ' + win_name )


def _get_tzi_for_year( dtzi_ptr: Ptr[None], year: i32 ) -> Ptr[u8]:
	from windows.kernel32 import DynamicTimeZoneInformation, TIME_ZONE_INFORMATION, GetTimeZoneInformationForYear
	struct_size: usize = compiler.sizeof( TIME_ZONE_INFORMATION )
	raw: Ptr[u8] = sys.alloc[u8]( struct_size )
	sys.memzero( raw, struct_size )
	tzi_ptr = compiler.cast( Ptr[TIME_ZONE_INFORMATION], raw )
	with compiler.wrap_arithmetic:
		wyear: u16 = u16( year )
	dtzi_typed = compiler.cast( Ptr[DynamicTimeZoneInformation], dtzi_ptr )
	ok: bool = GetTimeZoneInformationForYear( wyear, dtzi_typed, tzi_ptr )
	if not ok:
		sys.free( raw )
		sys.panic( 'zoneinfo: GetTimeZoneInformationForYear failed' )
	return raw


# --- recurring SYSTEMTIME rule -> concrete UTC transition instant ----------

def _bias_to_utcoffset_seconds( bias_minutes: i32, extra_bias_minutes: i32 ) -> i32:
	''' Win32: UTC = local + (Bias + <period bias>), in minutes - so the
	utcoffset this codebase stores (local = utc + utcoffset, seconds,
	matching TZif's own utoff convention) is the negation, in seconds. '''
	with compiler.wrap_arithmetic:
		total_minutes: i32 = bias_minutes + extra_bias_minutes
		return -( total_minutes * 60 )


def _is_leap_year( year: i32 ) -> bool:
	with compiler.panic_arithmetic( 'unreachable: divisors are non-zero literals' ):
		if year % 4 != 0:
			return False
		if year % 100 != 0:
			return True
		return year % 400 == 0


def _days_in_month( year: i32, month: i32 ) -> i32:
	if month == 1 or month == 3 or month == 5 or month == 7 or month == 8 or month == 10 or month == 12:
		return 31
	if month == 4 or month == 6 or month == 9 or month == 11:
		return 30
	if _is_leap_year( year ):
		return 29
	return 28


def _days_from_civil( year: i32, month: i32, day: i32 ) -> i64:
	''' Howard Hinnant's days_from_civil, days since 1970-01-01 - well-known,
	widely-used public-domain algorithm (see howardhinnant.github.io/
	date_algorithms.html). Only ever called here with year in a small
	current-era window (never negative/BCE-range), so the floor-vs-
	truncating integer division distinction the general algorithm has to
	care about never actually matters for any input this file produces. '''
	with compiler.panic_arithmetic( 'unreachable: divisors are non-zero literals, year is in a small current-era window' ):
		y: i64 = i64( year )
		if month <= 2:
			y -= 1
		era: i64 = y // 400
		yoe: i64 = y - era * 400
		mp: i64 = ( i64( month ) + 9 ) % 12
		doy: i64 = ( 153 * mp + 2 ) // 5 + i64( day ) - 1
		doe: i64 = yoe * 365 + yoe // 4 - yoe // 100 + doy
		return era * 146097 + doe - 719468


def _nth_weekday_epoch_seconds(
	year: i32, month: i32, day_of_week: i32, nth: i32,
	hour: i32, minute: i32, second: i32,
) -> i64:
	''' the local wall-clock reading (year, month, Nth day_of_week, time-of-
	day) reinterpreted AS IF it were UTC - just the epoch-seconds value of
	that calendar/time-of-day tuple, no bias applied yet (see this file's
	own callers for how the real UTC instant is derived from it). nth is
	1-4 for the 1st-4th occurrence in the month, 5 for the LAST occurrence -
	Windows' own SYSTEMTIME.wDay convention for a recurring *Date field
	(see kernel32.py's SYSTEMTIME comment). day_of_week is 0=Sunday. '''
	first_of_month_days: i64 = _days_from_civil( year, month, 1 )
	with compiler.panic_arithmetic( 'unreachable: divisor is a non-zero literal' ):
		weekday_of_1st: i64 = ( first_of_month_days + 4 ) % 7  # 1970-01-01 was a Thursday
		delta: i64 = ( i64( day_of_week ) - weekday_of_1st + 7 ) % 7
		first_occurrence_day: i32 = i32( 1 + delta )

	day_of_month: i32
	if nth <= 4:
		with compiler.wrap_arithmetic:
			day_of_month = first_occurrence_day + ( nth - 1 ) * 7
	else:
		days_in_month: i32 = _days_in_month( year, month )
		with compiler.panic_arithmetic( 'unreachable: divisor is a non-zero literal' ):
			day_of_month = first_occurrence_day + 7 * (( days_in_month - first_occurrence_day ) // 7 )

	days: i64 = _days_from_civil( year, month, day_of_month )
	with compiler.wrap_arithmetic:
		return days * 86400 + i64( hour ) * 3600 + i64( minute ) * 60 + i64( second )


class _YearRules:
	has_dst: bool
	std_rule: TTInfo
	dst_rule: TTInfo
	to_dst_epoch: i64  # instant of the std->dst transition (Windows' DaylightDate)
	to_std_epoch: i64  # instant of the dst->std transition (Windows' StandardDate)

	def __init__(
		self,
		has_dst: bool,
		std_rule: TTInfo,
		dst_rule: TTInfo,
		to_dst_epoch: i64,
		to_std_epoch: i64,
	) -> None:
		self.has_dst = has_dst
		self.std_rule = std_rule
		self.dst_rule = dst_rule
		self.to_dst_epoch = to_dst_epoch
		self.to_std_epoch = to_std_epoch


def _rules_for_year( tzi_raw: Ptr[u8], year: i32 ) -> _YearRules:
	from windows.kernel32 import TIME_ZONE_INFORMATION
	tzi_ptr = compiler.cast( Ptr[TIME_ZONE_INFORMATION], tzi_raw )

	std_offset: i32 = _bias_to_utcoffset_seconds( tzi_ptr.Bias, tzi_ptr.StandardBias )
	dst_offset: i32 = _bias_to_utcoffset_seconds( tzi_ptr.Bias, tzi_ptr.DaylightBias )
	# Windows never gives a stable, locale-independent short abbreviation
	# the way TZif's designation strings do (StandardName/DaylightName are
	# LOCALIZED display strings, not stable IDs - see kernel32.py's own
	# DynamicTimeZoneInformation comment on why they're never even decoded)
	# - 'STD'/'DST' are honest placeholders, not a real IANA-style
	# abbreviation.
	std_rule: TTInfo = TTInfo( utcoffset = std_offset, is_dst = False, abbr = 'STD' )
	dst_rule: TTInfo = TTInfo( utcoffset = dst_offset, is_dst = True, abbr = 'DST' )

	std_month: u16 = tzi_ptr.StandardDate_wMonth
	if std_month == 0:
		return _YearRules( has_dst = False, std_rule = std_rule, dst_rule = dst_rule, to_dst_epoch = 0, to_std_epoch = 0 )

	with compiler.wrap_arithmetic:
		dst_month: i32 = i32( tzi_ptr.DaylightDate_wMonth )
		dst_dow: i32 = i32( tzi_ptr.DaylightDate_wDayOfWeek )
		dst_nth: i32 = i32( tzi_ptr.DaylightDate_wDay )
		dst_hour: i32 = i32( tzi_ptr.DaylightDate_wHour )
		dst_minute: i32 = i32( tzi_ptr.DaylightDate_wMinute )
		dst_second: i32 = i32( tzi_ptr.DaylightDate_wSecond )
	# DaylightDate (std->dst transition): the local wall-clock reading just
	# before this instant is still in STANDARD time, so subtract the
	# STANDARD utcoffset to recover true UTC.
	dst_local: i64 = _nth_weekday_epoch_seconds( year, dst_month, dst_dow, dst_nth, dst_hour, dst_minute, dst_second )
	with compiler.wrap_arithmetic:
		to_dst_epoch: i64 = dst_local - i64( std_offset )

	with compiler.wrap_arithmetic:
		std_month_i32: i32 = i32( tzi_ptr.StandardDate_wMonth )
		std_dow: i32 = i32( tzi_ptr.StandardDate_wDayOfWeek )
		std_nth: i32 = i32( tzi_ptr.StandardDate_wDay )
		std_hour: i32 = i32( tzi_ptr.StandardDate_wHour )
		std_minute: i32 = i32( tzi_ptr.StandardDate_wMinute )
		std_second: i32 = i32( tzi_ptr.StandardDate_wSecond )
	# StandardDate (dst->std transition): local wall-clock just before this
	# one is still in DAYLIGHT time, so subtract the DAYLIGHT utcoffset.
	std_local: i64 = _nth_weekday_epoch_seconds( year, std_month_i32, std_dow, std_nth, std_hour, std_minute, std_second )
	with compiler.wrap_arithmetic:
		to_std_epoch: i64 = std_local - i64( dst_offset )

	return _YearRules( has_dst = True, std_rule = std_rule, dst_rule = dst_rule, to_dst_epoch = to_dst_epoch, to_std_epoch = to_std_epoch )
