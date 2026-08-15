# src/zoneinfo.py
#
# ZoneInfo(key) loads REAL DST/offset rules from the OS - no embedded tzdata
# blob, no hardcoded sample transitions. On POSIX this parses the real
# on-disk TZif file at /usr/share/zoneinfo/<key> (posix/zoneinfo_rules.py,
# same mechanism CPython's own zoneinfo module uses by default). On Windows
# there is no equivalent system file, so it calls the real Win32 timezone
# API instead (windows/zoneinfo_rules.py: EnumDynamicTimeZoneInformation +
# GetTimeZoneInformationForYear) - key must be a native Windows zone name
# (e.g. 'Eastern Standard Time') unless windows_zones.install() has been
# called to opt in to IANA-name translation (lib/windows_zones.py).
#
# _load_zone is defined twice via @compiler.target, exactly mirroring
# lib/time.py's own monotonic()/time() split - discovery discards the
# non-matching body outright, so its platform-specific import is never
# resolved on the other OS.

import compiler

class TTInfo:
	utcoffset: i32  # offset from UTC, in SECONDS (matches TZif's own on-disk
	                # unit directly - see posix/zoneinfo_rules.py - and avoids
	                # depending on datetime.timedelta, which doesn't exist yet)
	is_dst: bool    # Daylight Saving Time flag
	abbr: str       # e.g. "EST", "EDT"

	def __init__( self, utcoffset: i32, is_dst: bool, abbr: str ) -> None:
		self.utcoffset = utcoffset
		self.is_dst = is_dst
		self.abbr = abbr

class ZoneInfo:
	name: str
	default_rule: TTInfo         # in effect before the first transition
	transition_times: list[i64]  # sorted ascending, parallel to transition_rules
	transition_rules: list[TTInfo]

	def __init__( self, key: str ) -> None:
		self.name = key
		self.default_rule = TTInfo( utcoffset = 0, is_dst = False, abbr = '' )
		self.transition_times = []
		self.transition_rules = []
		_load_zone( self, key )

	def get_ttinfo( self, timestamp: i64 ) -> TTInfo:
		# hand-rolled binary search over transition_times (list[i64]), NOT
		# lib/bisect.py's bisect_right - bisect_right takes arr: slice[T],
		# and slice[T] has no real construction path from ordinary metalpy
		# source anywhere in this codebase yet (see str.join's own comment
		# in lib/builtins/__init__.py, which made this exact same call for
		# this exact same reason). Separately, a generic key=lambda call
		# here would also hit PLAN_LAMBDA.md's documented "not attempted
		# end to end" gap - this sidesteps both at once, at the cost of a
		# few duplicated lines instead of a shared helper.
		# arr[i] bracket sugar desugars to .__getitem__(i).or_return() (only
		# type-checks inside a function that itself returns a compatible
		# Result - see lowering.py's own _expr_Subscript comment), but
		# get_ttinfo returns a plain TTInfo, so every index here goes
		# through .__getitem__(...).unwrap(...) explicitly instead - same
		# "in bounds by construction" idiom str.join already uses.
		lo: usize = 0
		hi: usize = len( self.transition_times )
		while lo < hi:
			with compiler.panic_arithmetic( 'unreachable: hi > lo in binary search' ):
				mid: usize = lo + ( hi - lo ) // 2
			mid_time: i64 = self.transition_times.__getitem__( mid ).unwrap( 'zoneinfo: index in bounds by construction' )
			if timestamp < mid_time:
				hi = mid
			else:
				with compiler.wrap_arithmetic:
					lo = mid + 1

		if lo == 0:
			return self.default_rule

		with compiler.wrap_arithmetic:
			prev_idx: usize = lo - 1
		return self.transition_rules.__getitem__( prev_idx ).unwrap( 'zoneinfo: index in bounds by construction' )

	def utcoffset( self, timestamp: i64 ) -> i32:
		return self.get_ttinfo( timestamp ).utcoffset

	def abbr( self, timestamp: i64 ) -> str:
		return self.get_ttinfo( timestamp ).abbr


@compiler.target( os = 'windows' )
def _load_zone( zone: ZoneInfo, key: str ) -> None:
	from windows.zoneinfo_rules import load_zone
	load_zone( zone, key )

@compiler.target( os = not 'windows' )
def _load_zone( zone: ZoneInfo, key: str ) -> None:
	from posix.zoneinfo_rules import load_zone
	load_zone( zone, key )
