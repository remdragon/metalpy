# src/windows/time.py

from codecs import utf16

def get_local_timezone_name() -> str:
	from windows.kernel32 import (
		DynamicTimeZoneInformation,
		GetDynamicTimeZoneInformation,
	)
	
	tz_info = DynamicTimeZoneInformation()
	
	status: u32 = GetDynamicTimeZoneInformation( tz_info )
	# status values:
	# TIME_ZONE_ID_INVALID: u32   = 0xFFFFFFFF
	# TIME_ZONE_ID_UNKNOWN: u32   = 0
	# TIME_ZONE_ID_STANDARD: u32  = 1
	# TIME_ZONE_ID_DAYLIGHT: u32  = 2
	
	if status == 0xFFFFFFFF:
		return 'UTC'
	
	if tz_info.TimeZoneKeyName[0] != 0:
		win_name = utf16.decode( tz_info.TimeZoneKeyName )
		return _get_win_iana_map().get( win_name, win_name )
	
	return utf16.decode( tz_info.StandardName )
