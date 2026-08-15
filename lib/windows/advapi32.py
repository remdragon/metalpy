# src/windows/advapi32.py — advapi32.dll externs. Separate file from
# kernel32.py because @extern('advapi32', ...) auto-links advapi32.lib/
# -ladvapi32 (see linker_c.py's own lib-name-becomes-linker-flag logic),
# mirroring the existing one-file-per-DLL convention (kernel32.py, ntdll.py).

from windows.kernel32 import DynamicTimeZoneInformation

# DWORD EnumDynamicTimeZoneInformation(const DWORD dwIndex,
# PDYNAMIC_TIME_ZONE_INFORMATION lpTimeZoneInformation) - enumerates every
# DYNAMIC_TIME_ZONE_INFORMATION entry stored in the registry, one per
# dwIndex starting at 0. Returns ERROR_SUCCESS (0) while there's a zone at
# that index, ERROR_NO_MORE_ITEMS once past the last one - see
# windows/zoneinfo_rules.py for the index-scanning loop that resolves a
# Windows zone key name (e.g. 'Eastern Standard Time') to the
# DYNAMIC_TIME_ZONE_INFORMATION GetTimeZoneInformationForYear needs.
ERROR_SUCCESS: u32 = 0
ERROR_NO_MORE_ITEMS: u32 = 259

@extern( 'advapi32', 'EnumDynamicTimeZoneInformation' )
def EnumDynamicTimeZoneInformation(
	dwIndex: u32,
	lpTimeZoneInformation: Ptr[DynamicTimeZoneInformation],
) -> u32:
	...
