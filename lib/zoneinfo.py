# src/zoneinfo.py

from bisect import bisect_right
from datetime import timedelta

class TTInfo:
	utcoffset: timedelta # offset from UTC
	is_dst: bool    # Daylight Savings Flag
	abbr: str       # e.g., "EST", "EDT"
	
	def __init__( self,
		utcoffset: i32,
		is_dst: bool,
		abbr: str,
	) -> None:
		self.utcoffset = utcoffset
		self.is_dst = is_dst
		self.abbr = abbr

class Transition:
	timestamp: i64
	rule: TTInfo

class ZoneInfo:
	name: str
	rules: list[TTInfo]
	transitions: list[Transition]
	
	def __init__(
		self,
		name: str,
		default_rule: TTInfo,
		transitions: list[Transition] = [],
	) -> None:
		self.name = name
		self.default_rule = default_rule
		self.transitions = transitions
	
	def get_ttinfo( self, timestamp: i64 ) -> TTInfo:
		# If no transitions exist, return the default rule
		idx = bisect_right( self.transitions, timestamp, key = lambda tran: tran.timestamp )
		
		if idx == 0:
			return self.default_rule
		
		return self.transitions[idx-1].rule
	
	def utcoffset( self, timestamp: i64 ) -> timedelta:
		return self.get_ttinfo( timestamp ).utcoffset
	
	def abbr( self, timestamp: i64 ) -> str:
		return self.get_ttinfo( timestamp ).abbr

def create_utc_zone() -> ZoneInfo:
	utc_rule = TTInfo( utcoffset = timedelta( seconds = 0 ), is_dst = False, abbr = 'UTC' )
	return ZoneInfo( name = 'UTC', default_rule = utc_rule )

def create_america_new_york_zone() -> ZoneInfo:
	est = TTInfo( utcoffset = timedelta( hours = -5 ), is_dst = False, abbr = 'EST' )
	edt = TTInfo( utcoffset = timedelta( hours = -4 ), is_dst = True,  abbr = 'EDT' )
	
	# Sample transition boundaries (Unix timestamps)
	# Rule index 0 = EST, Rule index 1 = EDT
	transitions = [
		Transition( 1710054000, edt ),  # March 10, 2024 (Spring forward -> EDT)
		Transition( 1730617200, est ),  # Nov 3, 2024    (Fall back -> EST)
		Transition( 1741503600, edt ),  # March 9, 2025 (Spring forward -> EDT)
		Transition( 1762066800, est ),  # Nov 2, 2025    (Fall back -> EST)
	]
	
	return ZoneInfo(
		name = 'America/New_York',
		default_rule = est,
		transitions = transitions,
	)
