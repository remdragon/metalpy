HANDLE: TypeAlias = Ptr[None]

# INVALID_HANDLE_VALUE from winbase.h: (HANDLE)(LONG_PTR)-1
INVALID_HANDLE_VALUE: HANDLE = -1

STD_INPUT_HANDLE: u32 = u32( -10 )
STD_OUTPUT_HANDLE: u32 = u32( -11 )
STD_ERROR_HANDLE: u32 = u32( -12 )

_TZNAME_SIZE: usize = 32
_TZKEYNAME_SIZE: usize = 128

@cstruct
class SYSTEMTIME:
	# wYear == 0 in a *Date field below means "yearly recurring rule" (wDay
	# is then 1-5, the Nth occurrence of wDayOfWeek in wMonth, 5 = last) -
	# see windows/zoneinfo_rules.py's own rule-interpretation comment. Used
	# standalone (GetSystemTime-style callers construct one directly) AND
	# nested by value as StandardDate/DaylightDate below.
	wYear: u16 = 0
	wMonth: u16 = 0
	wDayOfWeek: u16 = 0
	wDay: u16 = 0
	wHour: u16 = 0
	wMinute: u16 = 0
	wSecond: u16 = 0
	wMilliseconds: u16 = 0


# DYNAMIC_TIME_ZONE_INFORMATION (timezoneapi.h) - verified field-for-field
# against Microsoft's own docs this session (learn.microsoft.com/.../
# ns-timezoneapi-dynamic_time_zone_information): Bias:LONG, StandardName:
# WCHAR[32], StandardDate:SYSTEMTIME, StandardBias:LONG, DaylightName:
# WCHAR[32], DaylightDate:SYSTEMTIME, DaylightBias:LONG, TimeZoneKeyName:
# WCHAR[128] (NOT 32 - the previous version of this struct wrongly reused
# _TZNAME_SIZE here, a 3x undersized buffer that GetDynamicTimeZoneInformation/
# EnumDynamicTimeZoneInformation would have written past),
# DynamicDaylightTimeDisabled:BOOLEAN.
#
# u16[N]-style fixed-size inline array fields are NOT implemented by this
# compiler (confirmed independently in lib/guid.py's own comment - no
# visit_Subscript branch handles an integer-literal array size), so every
# WCHAR array here is decomposed into individually-numbered u16 fields, same
# precedent lib/guid.py's own GUID.Data4 already set for a smaller case.
# Every field gets a "= 0" default (matching posix/time.py's own timespec
# precedent) so DynamicTimeZoneInformation() can be constructed with no
# args - a real @cstruct requires every field to be given explicitly
# otherwise.
#
# StandardDate/DaylightDate nest SYSTEMTIME by value directly - a small
# cstruct nested by value inside a large cross-module cstruct used to hit a
# real emitter bug here (a by-value-embedded field type reachable only
# through the containing struct never got scheduled for compilation at all -
# confirmed directly via a real clang "incomplete type" error, worked around
# at the time by inlining SYSTEMTIME's 8 fields by hand; fixed upstream
# since - task_421ed8be).
#
# StandardName/DaylightName's own content is never read anywhere in this
# codebase (only TimeZoneKeyName is) - they still need real, individually-
# declared u16 fields (not a collapsed/reinterpreted filler region) purely
# to hold TimeZoneKeyName at the correct byte offset.
#
# compiler.addrof(...) only accepts a bare local-variable name, not a field
# expression (verified directly this session - `compiler.addrof(x.field)`
# is a hard compile error), so there is no way to get Ptr[u16] directly at
# TimeZoneKeyName_0. windows/zoneinfo_rules.py's helpers instead take
# compiler.addrof() of the WHOLE struct (a bare local IS allowed), cast to
# Ptr[u8], and advance by _TZKEYNAME_OFFSET bytes - the sum of every
# field's size up to (not including) TimeZoneKeyName_0: Bias(4) +
# StandardName(32*2=64) + StandardDate(8*2=16) + StandardBias(4) +
# DaylightName(64) + DaylightDate(16) + DaylightBias(4) = 172. No struct
# field is ever reordered/renamed above without updating this constant to
# match.
@cstruct
class DynamicTimeZoneInformation:
	Bias: i32 = 0
	StandardName_0: u16 = 0
	StandardName_1: u16 = 0
	StandardName_2: u16 = 0
	StandardName_3: u16 = 0
	StandardName_4: u16 = 0
	StandardName_5: u16 = 0
	StandardName_6: u16 = 0
	StandardName_7: u16 = 0
	StandardName_8: u16 = 0
	StandardName_9: u16 = 0
	StandardName_10: u16 = 0
	StandardName_11: u16 = 0
	StandardName_12: u16 = 0
	StandardName_13: u16 = 0
	StandardName_14: u16 = 0
	StandardName_15: u16 = 0
	StandardName_16: u16 = 0
	StandardName_17: u16 = 0
	StandardName_18: u16 = 0
	StandardName_19: u16 = 0
	StandardName_20: u16 = 0
	StandardName_21: u16 = 0
	StandardName_22: u16 = 0
	StandardName_23: u16 = 0
	StandardName_24: u16 = 0
	StandardName_25: u16 = 0
	StandardName_26: u16 = 0
	StandardName_27: u16 = 0
	StandardName_28: u16 = 0
	StandardName_29: u16 = 0
	StandardName_30: u16 = 0
	StandardName_31: u16 = 0
	StandardDate: SYSTEMTIME = SYSTEMTIME()
	StandardBias: i32 = 0
	DaylightName_0: u16 = 0
	DaylightName_1: u16 = 0
	DaylightName_2: u16 = 0
	DaylightName_3: u16 = 0
	DaylightName_4: u16 = 0
	DaylightName_5: u16 = 0
	DaylightName_6: u16 = 0
	DaylightName_7: u16 = 0
	DaylightName_8: u16 = 0
	DaylightName_9: u16 = 0
	DaylightName_10: u16 = 0
	DaylightName_11: u16 = 0
	DaylightName_12: u16 = 0
	DaylightName_13: u16 = 0
	DaylightName_14: u16 = 0
	DaylightName_15: u16 = 0
	DaylightName_16: u16 = 0
	DaylightName_17: u16 = 0
	DaylightName_18: u16 = 0
	DaylightName_19: u16 = 0
	DaylightName_20: u16 = 0
	DaylightName_21: u16 = 0
	DaylightName_22: u16 = 0
	DaylightName_23: u16 = 0
	DaylightName_24: u16 = 0
	DaylightName_25: u16 = 0
	DaylightName_26: u16 = 0
	DaylightName_27: u16 = 0
	DaylightName_28: u16 = 0
	DaylightName_29: u16 = 0
	DaylightName_30: u16 = 0
	DaylightName_31: u16 = 0
	DaylightDate: SYSTEMTIME = SYSTEMTIME()
	DaylightBias: i32 = 0
	TimeZoneKeyName_0: u16 = 0
	TimeZoneKeyName_1: u16 = 0
	TimeZoneKeyName_2: u16 = 0
	TimeZoneKeyName_3: u16 = 0
	TimeZoneKeyName_4: u16 = 0
	TimeZoneKeyName_5: u16 = 0
	TimeZoneKeyName_6: u16 = 0
	TimeZoneKeyName_7: u16 = 0
	TimeZoneKeyName_8: u16 = 0
	TimeZoneKeyName_9: u16 = 0
	TimeZoneKeyName_10: u16 = 0
	TimeZoneKeyName_11: u16 = 0
	TimeZoneKeyName_12: u16 = 0
	TimeZoneKeyName_13: u16 = 0
	TimeZoneKeyName_14: u16 = 0
	TimeZoneKeyName_15: u16 = 0
	TimeZoneKeyName_16: u16 = 0
	TimeZoneKeyName_17: u16 = 0
	TimeZoneKeyName_18: u16 = 0
	TimeZoneKeyName_19: u16 = 0
	TimeZoneKeyName_20: u16 = 0
	TimeZoneKeyName_21: u16 = 0
	TimeZoneKeyName_22: u16 = 0
	TimeZoneKeyName_23: u16 = 0
	TimeZoneKeyName_24: u16 = 0
	TimeZoneKeyName_25: u16 = 0
	TimeZoneKeyName_26: u16 = 0
	TimeZoneKeyName_27: u16 = 0
	TimeZoneKeyName_28: u16 = 0
	TimeZoneKeyName_29: u16 = 0
	TimeZoneKeyName_30: u16 = 0
	TimeZoneKeyName_31: u16 = 0
	TimeZoneKeyName_32: u16 = 0
	TimeZoneKeyName_33: u16 = 0
	TimeZoneKeyName_34: u16 = 0
	TimeZoneKeyName_35: u16 = 0
	TimeZoneKeyName_36: u16 = 0
	TimeZoneKeyName_37: u16 = 0
	TimeZoneKeyName_38: u16 = 0
	TimeZoneKeyName_39: u16 = 0
	TimeZoneKeyName_40: u16 = 0
	TimeZoneKeyName_41: u16 = 0
	TimeZoneKeyName_42: u16 = 0
	TimeZoneKeyName_43: u16 = 0
	TimeZoneKeyName_44: u16 = 0
	TimeZoneKeyName_45: u16 = 0
	TimeZoneKeyName_46: u16 = 0
	TimeZoneKeyName_47: u16 = 0
	TimeZoneKeyName_48: u16 = 0
	TimeZoneKeyName_49: u16 = 0
	TimeZoneKeyName_50: u16 = 0
	TimeZoneKeyName_51: u16 = 0
	TimeZoneKeyName_52: u16 = 0
	TimeZoneKeyName_53: u16 = 0
	TimeZoneKeyName_54: u16 = 0
	TimeZoneKeyName_55: u16 = 0
	TimeZoneKeyName_56: u16 = 0
	TimeZoneKeyName_57: u16 = 0
	TimeZoneKeyName_58: u16 = 0
	TimeZoneKeyName_59: u16 = 0
	TimeZoneKeyName_60: u16 = 0
	TimeZoneKeyName_61: u16 = 0
	TimeZoneKeyName_62: u16 = 0
	TimeZoneKeyName_63: u16 = 0
	TimeZoneKeyName_64: u16 = 0
	TimeZoneKeyName_65: u16 = 0
	TimeZoneKeyName_66: u16 = 0
	TimeZoneKeyName_67: u16 = 0
	TimeZoneKeyName_68: u16 = 0
	TimeZoneKeyName_69: u16 = 0
	TimeZoneKeyName_70: u16 = 0
	TimeZoneKeyName_71: u16 = 0
	TimeZoneKeyName_72: u16 = 0
	TimeZoneKeyName_73: u16 = 0
	TimeZoneKeyName_74: u16 = 0
	TimeZoneKeyName_75: u16 = 0
	TimeZoneKeyName_76: u16 = 0
	TimeZoneKeyName_77: u16 = 0
	TimeZoneKeyName_78: u16 = 0
	TimeZoneKeyName_79: u16 = 0
	TimeZoneKeyName_80: u16 = 0
	TimeZoneKeyName_81: u16 = 0
	TimeZoneKeyName_82: u16 = 0
	TimeZoneKeyName_83: u16 = 0
	TimeZoneKeyName_84: u16 = 0
	TimeZoneKeyName_85: u16 = 0
	TimeZoneKeyName_86: u16 = 0
	TimeZoneKeyName_87: u16 = 0
	TimeZoneKeyName_88: u16 = 0
	TimeZoneKeyName_89: u16 = 0
	TimeZoneKeyName_90: u16 = 0
	TimeZoneKeyName_91: u16 = 0
	TimeZoneKeyName_92: u16 = 0
	TimeZoneKeyName_93: u16 = 0
	TimeZoneKeyName_94: u16 = 0
	TimeZoneKeyName_95: u16 = 0
	TimeZoneKeyName_96: u16 = 0
	TimeZoneKeyName_97: u16 = 0
	TimeZoneKeyName_98: u16 = 0
	TimeZoneKeyName_99: u16 = 0
	TimeZoneKeyName_100: u16 = 0
	TimeZoneKeyName_101: u16 = 0
	TimeZoneKeyName_102: u16 = 0
	TimeZoneKeyName_103: u16 = 0
	TimeZoneKeyName_104: u16 = 0
	TimeZoneKeyName_105: u16 = 0
	TimeZoneKeyName_106: u16 = 0
	TimeZoneKeyName_107: u16 = 0
	TimeZoneKeyName_108: u16 = 0
	TimeZoneKeyName_109: u16 = 0
	TimeZoneKeyName_110: u16 = 0
	TimeZoneKeyName_111: u16 = 0
	TimeZoneKeyName_112: u16 = 0
	TimeZoneKeyName_113: u16 = 0
	TimeZoneKeyName_114: u16 = 0
	TimeZoneKeyName_115: u16 = 0
	TimeZoneKeyName_116: u16 = 0
	TimeZoneKeyName_117: u16 = 0
	TimeZoneKeyName_118: u16 = 0
	TimeZoneKeyName_119: u16 = 0
	TimeZoneKeyName_120: u16 = 0
	TimeZoneKeyName_121: u16 = 0
	TimeZoneKeyName_122: u16 = 0
	TimeZoneKeyName_123: u16 = 0
	TimeZoneKeyName_124: u16 = 0
	TimeZoneKeyName_125: u16 = 0
	TimeZoneKeyName_126: u16 = 0
	TimeZoneKeyName_127: u16 = 0
	DynamicDaylightTimeDisabled: u8 = 0
	_pad0: u8 = 0
	_pad1: u8 = 0
	_pad2: u8 = 0

_TZNAME_OFFSET: usize = 4       # StandardName_0 starts right after Bias's 4 bytes
_TZKEYNAME_OFFSET: usize = 172  # see DynamicTimeZoneInformation's own comment


# TIME_ZONE_INFORMATION (timezoneapi.h) - same leading layout as
# DynamicTimeZoneInformation above, just without TimeZoneKeyName/
# DynamicDaylightTimeDisabled. This is GetTimeZoneInformationForYear's own
# [out] parameter type - StandardDate/DaylightDate are read directly as
# ordinary nested-struct field access (tzi.StandardDate.wMonth, etc.) - no
# addrof/offset trick needed there, unlike TimeZoneKeyName above, since
# these are real scalar reads, not "get a contiguous buffer" reads. See
# windows/zoneinfo_rules.py for how the recurring-rule encoding in these two
# fields becomes a concrete UTC transition instant for a given year.
@cstruct
class TIME_ZONE_INFORMATION:
	Bias: i32 = 0
	StandardName_0: u16 = 0
	StandardName_1: u16 = 0
	StandardName_2: u16 = 0
	StandardName_3: u16 = 0
	StandardName_4: u16 = 0
	StandardName_5: u16 = 0
	StandardName_6: u16 = 0
	StandardName_7: u16 = 0
	StandardName_8: u16 = 0
	StandardName_9: u16 = 0
	StandardName_10: u16 = 0
	StandardName_11: u16 = 0
	StandardName_12: u16 = 0
	StandardName_13: u16 = 0
	StandardName_14: u16 = 0
	StandardName_15: u16 = 0
	StandardName_16: u16 = 0
	StandardName_17: u16 = 0
	StandardName_18: u16 = 0
	StandardName_19: u16 = 0
	StandardName_20: u16 = 0
	StandardName_21: u16 = 0
	StandardName_22: u16 = 0
	StandardName_23: u16 = 0
	StandardName_24: u16 = 0
	StandardName_25: u16 = 0
	StandardName_26: u16 = 0
	StandardName_27: u16 = 0
	StandardName_28: u16 = 0
	StandardName_29: u16 = 0
	StandardName_30: u16 = 0
	StandardName_31: u16 = 0
	StandardDate: SYSTEMTIME = SYSTEMTIME()
	StandardBias: i32 = 0
	DaylightName_0: u16 = 0
	DaylightName_1: u16 = 0
	DaylightName_2: u16 = 0
	DaylightName_3: u16 = 0
	DaylightName_4: u16 = 0
	DaylightName_5: u16 = 0
	DaylightName_6: u16 = 0
	DaylightName_7: u16 = 0
	DaylightName_8: u16 = 0
	DaylightName_9: u16 = 0
	DaylightName_10: u16 = 0
	DaylightName_11: u16 = 0
	DaylightName_12: u16 = 0
	DaylightName_13: u16 = 0
	DaylightName_14: u16 = 0
	DaylightName_15: u16 = 0
	DaylightName_16: u16 = 0
	DaylightName_17: u16 = 0
	DaylightName_18: u16 = 0
	DaylightName_19: u16 = 0
	DaylightName_20: u16 = 0
	DaylightName_21: u16 = 0
	DaylightName_22: u16 = 0
	DaylightName_23: u16 = 0
	DaylightName_24: u16 = 0
	DaylightName_25: u16 = 0
	DaylightName_26: u16 = 0
	DaylightName_27: u16 = 0
	DaylightName_28: u16 = 0
	DaylightName_29: u16 = 0
	DaylightName_30: u16 = 0
	DaylightName_31: u16 = 0
	DaylightDate: SYSTEMTIME = SYSTEMTIME()
	DaylightBias: i32 = 0


@extern('kernel32', 'HeapAlloc')
def HeapAlloc(
	hHeap: HANDLE,
	dwFlags: u32,
	dwBytes: usize,
) -> Ptr[u8]:
	...

@extern('kernel32', 'HeapFree')
def HeapFree(
	hHeap: HANDLE,
	dwFlags: u32,
	lpMem: Ptr[u8],
) -> bool:
	...

@extern( 'kernel32', 'GetDynamicTimeZoneInformation' )
def GetDynamicTimeZoneInformation(
	pTimeZoneInformation: Ptr[DynamicTimeZoneInformation],
) -> u32:
	...

# BOOL GetTimeZoneInformationForYear(USHORT wYear, PDYNAMIC_TIME_ZONE_
# INFORMATION pdtzi, LPTIME_ZONE_INFORMATION ptzi) - the officially
# documented way to get a specific (non-current) zone's rule for a specific
# year (Microsoft's own docs, verified this session: to populate pdtzi for a
# zone that ISN'T the current one, call EnumDynamicTimeZoneInformation with
# the index of the zone you want - see windows/advapi32.py - rather than
# hand-constructing a DYNAMIC_TIME_ZONE_INFORMATION with just TimeZoneKeyName
# set, which isn't the documented/supported path even though it's known to
# work in practice). Returns nonzero on success; zero (+ GetLastError()) on
# failure, e.g. an unrecognized TimeZoneKeyName.
@extern( 'kernel32', 'GetTimeZoneInformationForYear' )
def GetTimeZoneInformationForYear(
	wYear: u16,
	pdtzi: Ptr[DynamicTimeZoneInformation],
	ptzi: Ptr[TIME_ZONE_INFORMATION],
) -> bool:
	...

@extern( 'kernel32', 'ExitProcess' )
def ExitProcess(
	uExitCode: u32,
) -> None:
	...

@extern( 'kernel32', 'GetLastError' )
def GetLastError() -> u32:
	...

# void GetSystemTime(LPSYSTEMTIME lpSystemTime) - current UTC time; used by
# windows/zoneinfo_rules.py purely to read off the current YEAR (a rough
# "now" for choosing which years to build DST transitions for) - no
# calendar/datetime module exists yet to derive that any other way.
@extern( 'kernel32', 'GetSystemTime' )
def GetSystemTime(
	lpSystemTime: Ptr[SYSTEMTIME],
) -> None:
	...

@extern( 'kernel32', 'GetProcessHeap' )
def GetProcessHeap() -> HANDLE:
	...

@extern('kernel32', 'GetStdHandle')
def GetStdHandle(
	nStdHandle: u32,
) -> HANDLE:
	...

@extern('kernel32', 'WriteFile')
def WriteFile(
	hFile: HANDLE,
	lpBuffer: ConstPtr[u8],
	nNumberOfBytesToWrite: u32,
	lpNumberOfBytesWritten: Ptr[u32],
	lpOverlapped: Ptr[None],
) -> bool:
	...

@extern('kernel32', 'CreateFileA')
def CreateFileA(
	lpFileName: ConstPtr[u8],
	dwDesiredAccess: u32,
	dwShareMode: u32,
	lpSecurityAttributes: Ptr[None],
	dwCreationDisposition: u32,
	dwFlagsAndAttributes: u32,
	hTemplateFile: HANDLE,
) -> HANDLE:
	...

@extern('kernel32', 'ReadFile')
def ReadFile(
	hFile: HANDLE,
	lpBuffer: Ptr[u8],
	nNumberOfBytesToRead: u32,
	lpNumberOfBytesRead: Ptr[u32],
	lpOverlapped: Ptr[None],
) -> bool:
	...

@extern('kernel32', 'CloseHandle')
def CloseHandle(
	hObject: HANDLE,
) -> bool:
	...

# DWORD WINAPI ThreadProc(LPVOID lpParameter) - the one shape every thread
# entry point takes; lib/threading.py's own Thread class always hands this
# the same fixed trampoline (never a per-closure one - see its own comment)
@extern('kernel32', 'CreateThread')
def CreateThread(
	lpThreadAttributes: Ptr[None],
	dwStackSize: usize,
	lpStartAddress: Ptr[Callable[[Ptr[None]], u32]],
	lpParameter: Ptr[None],
	dwCreationFlags: u32,
	lpThreadId: Ptr[u32],
) -> HANDLE:
	...

INFINITE: u32 = u32( -1 )

@extern('kernel32', 'WaitForSingleObject')
def WaitForSingleObject(
	hHandle: HANDLE,
	dwMilliseconds: u32,
) -> u32:
	...

@extern('kernel32', 'SetFilePointerEx')
def SetFilePointerEx(
	hFile: HANDLE,
	liDistanceToMove: i64,
	lpNewFilePointer: Ptr[i64],
	dwMoveMethod: u32,
) -> bool:
	...

@extern('kernel32', 'SetEndOfFile')
def SetEndOfFile(
	hFile: HANDLE,
) -> bool:
	...


# ---------------------------------------------------------------------------
# High-resolution timing — lib/time.py's monotonic() and time().
#
# QueryPerformanceCounter/QueryPerformanceFrequency are Microsoft's documented,
# recommended monotonic high-resolution timestamp (a fixed ~10 MHz counter on
# modern Windows, read largely in user mode). GetSystemTimePreciseAsFileTime is
# the sub-microsecond wall clock (Win8 / Server 2012+). All three live in
# kernel32 (always loaded, not C-runtime functions) so using them keeps Windows
# builds CRT-free, same as everything else here - no need to reach into ntdll's
# undocumented Rtl* equivalents. LARGE_INTEGER*/FILETIME* are each a single
# 8-byte little-endian value, so Ptr[i64]/Ptr[u64] are ABI-identical to the real
# out-parameter types (no @cstruct needed, same as WriteFile's Ptr[u32] count).
# ---------------------------------------------------------------------------

@extern('kernel32', 'QueryPerformanceCounter')
def QueryPerformanceCounter(
	lpPerformanceCount: Ptr[i64],  # LARGE_INTEGER*
) -> bool:
	...

@extern('kernel32', 'QueryPerformanceFrequency')
def QueryPerformanceFrequency(
	lpFrequency: Ptr[i64],  # LARGE_INTEGER*
) -> bool:
	...

@extern('kernel32', 'GetSystemTimePreciseAsFileTime')
def GetSystemTimePreciseAsFileTime(
	lpSystemTimeAsFileTime: Ptr[u64],  # LPFILETIME
) -> None:
	...


# ---------------------------------------------------------------------------
# SRWLOCK — slim reader/writer lock (exclusive-only for FastLock)
# ---------------------------------------------------------------------------

@cstruct
class _SRWLOCK:
	_opaque: Ptr[None]  # SRWLOCK is a single pointer-sized opaque struct


@extern('kernel32', 'AcquireSRWLockExclusive')
def AcquireSRWLockExclusive(
	SRWLock: Ptr[_SRWLOCK],
) -> None:
	...

@extern('kernel32', 'TryAcquireSRWLockExclusive')
def TryAcquireSRWLockExclusive(
	SRWLock: Ptr[_SRWLOCK],
) -> bool:
	...

@extern('kernel32', 'ReleaseSRWLockExclusive')
def ReleaseSRWLockExclusive(
	SRWLock: Ptr[_SRWLOCK],
) -> None:
	...


# ---------------------------------------------------------------------------
# Unicode-correct case mapping - str.upper()/str.lower() (see PLAN_STR_UPPER_
# LOWER.md) - MultiByteToWideChar/WideCharToMultiByte for UTF-8<->UTF-16, and
# LCMapStringEx (with LCMAP_LINGUISTIC_CASING) for the actual casing, which
# handles one-to-many expansions (ss -> SS) and context-sensitive rules
# (Greek final sigma) that a plain per-codepoint mapping can't.
# ---------------------------------------------------------------------------

CP_UTF8: u32 = 65001

@extern('kernel32', 'SetConsoleOutputCP')
def SetConsoleOutputCP(
	wCodePageID: u32,
) -> bool:
	...

LCMAP_LOWERCASE:         u32 = 0x00000100
LCMAP_UPPERCASE:         u32 = 0x00000200
LCMAP_LINGUISTIC_CASING: u32 = 0x01000000

@extern('kernel32', 'MultiByteToWideChar')
def MultiByteToWideChar(
	CodePage: u32,
	dwFlags: u32,
	lpMultiByteStr: ConstPtr[u8],
	cbMultiByte: i32,
	lpWideCharStr: Ptr[u16],
	cchWideChar: i32,
) -> i32:
	...

@extern('kernel32', 'WideCharToMultiByte')
def WideCharToMultiByte(
	CodePage: u32,
	dwFlags: u32,
	lpWideCharStr: ConstPtr[u16],
	cchWideChar: i32,
	lpMultiByteStr: Ptr[u8],
	cbMultiByte: i32,
	lpDefaultChar: ConstPtr[u8],
	lpUsedDefaultChar: Ptr[u32], # really LPBOOL (Win32 BOOL is a 4-byte int, not this language's 1-byte bool) - always passed None here, so the exact pointee width is moot in practice
) -> i32:
	...

@extern('kernel32', 'LCMapStringEx')
def LCMapStringEx(
	lpLocaleName: ConstPtr[u16],
	dwMapFlags: u32,
	lpSrcStr: ConstPtr[u16],
	cchSrc: i32,
	lpDestStr: Ptr[u16],
	cchDest: i32,
	lpVersionInformation: Ptr[None],
	lpReserved: Ptr[None],
	sortHandle: usize,
) -> i32:
	...

# ---------------------------------------------------------------------------
# Unicode codepoint classification - str.isalpha()/isdigit()/isspace()/
# isupper()/islower()/isalnum()/isprintable() (see __str.py's is_*_cp
# primitives, TODO.txt's str-methods plan). GetStringTypeW (not the
# locale-aware GetStringTypeExW - character TYPE classification is
# inherently locale-independent per Win32 docs, so the simpler, non-
# deprecated, no-locale-parameter API is the right one here, unlike
# LCMapStringEx above which genuinely needs LOCALE_NAME_INVARIANT for
# case mapping). CT_CTYPE1 selects the C1_* "character type 1" flag set.
# ---------------------------------------------------------------------------

CT_CTYPE1: u32 = 1

C1_UPPER:  u16 = 0x0001
C1_LOWER:  u16 = 0x0002
C1_DIGIT:  u16 = 0x0004
C1_SPACE:  u16 = 0x0008
C1_PUNCT:  u16 = 0x0010
C1_CNTRL:  u16 = 0x0020
C1_BLANK:  u16 = 0x0040
C1_XDIGIT: u16 = 0x0080
C1_ALPHA:  u16 = 0x0100

@extern('kernel32', 'GetStringTypeW')
def GetStringTypeW(
	dwInfoType: u32,
	lpSrcStr: ConstPtr[u16],
	cchSrc: i32,
	lpCharType: Ptr[u16],
) -> bool:
	...
