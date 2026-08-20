HANDLE: TypeAlias = Ptr[None]

# INVALID_HANDLE_VALUE from winbase.h: (HANDLE)(LONG_PTR)-1
INVALID_HANDLE_VALUE: HANDLE = -1

STD_INPUT_HANDLE: u32 = u32( -10 )
STD_OUTPUT_HANDLE: u32 = u32( -11 )
STD_ERROR_HANDLE: u32 = u32( -12 )

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


# DYNAMIC_TIME_ZONE_INFORMATION (timezoneapi.h): Bias:LONG, StandardName:
# WCHAR[32], StandardDate:SYSTEMTIME, StandardBias:LONG, DaylightName:
# WCHAR[32], DaylightDate:SYSTEMTIME, DaylightBias:LONG, TimeZoneKeyName:
# WCHAR[128], DynamicDaylightTimeDisabled:BOOLEAN. Every field gets a "= 0"
# default so DynamicTimeZoneInformation() can be constructed with no args.
#
# StandardName/DaylightName's own content is never read anywhere in this
# codebase (only TimeZoneKeyName is) - kept as real fields purely to hold
# TimeZoneKeyName at the correct byte offset.
#
# TimeZoneKeyName/StandardName are read as a whole null-terminated string
# via compiler.addrof(x.field) (Ptr[u16] at the array's own start, C array-
# to-pointer decay), not one element at a time.
@cstruct
class DynamicTimeZoneInformation:
	Bias: i32 = 0
	StandardName: u16[32] = 0
	StandardDate: SYSTEMTIME = SYSTEMTIME()
	StandardBias: i32 = 0
	DaylightName: u16[32] = 0
	DaylightDate: SYSTEMTIME = SYSTEMTIME()
	DaylightBias: i32 = 0
	TimeZoneKeyName: u16[128] = 0
	DynamicDaylightTimeDisabled: u8 = 0
	_pad: u8[3] = 0


# TIME_ZONE_INFORMATION (timezoneapi.h) - same leading layout as
# DynamicTimeZoneInformation above, just without TimeZoneKeyName/
# DynamicDaylightTimeDisabled. This is GetTimeZoneInformationForYear's own
# [out] parameter type - StandardDate/DaylightDate are read directly as
# ordinary nested-struct field access (tzi.StandardDate.wMonth, etc.), real
# scalar reads rather than TimeZoneKeyName's own "get a pointer to a whole
# array" case above. See windows/zoneinfo_rules.py for how the recurring-
# rule encoding in these two fields becomes a concrete UTC transition
# instant for a given year.
@cstruct
class TIME_ZONE_INFORMATION:
	Bias: i32 = 0
	StandardName: u16[32] = 0
	StandardDate: SYSTEMTIME = SYSTEMTIME()
	StandardBias: i32 = 0
	DaylightName: u16[32] = 0
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
) -> NoReturn:
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


# ---------------------------------------------------------------------------
# Fibers — cooperative, OS-scheduled stack-switching on one real OS thread.
# lpFiber return/params are LPVOID (a direct pointer to the fiber's own
# bookkeeping, not a kernel HANDLE), so these are typed as bare Ptr[None]
# rather than reusing HANDLE above. dwStackSize behaves like a thread's
# stack size (reserve dwStackSize, commit an initial slice, guard-page the
# rest) - CreateFiber gives no way to hand it caller-supplied stack memory.
# ---------------------------------------------------------------------------

# LPVOID WINAPI CreateFiber(SIZE_T dwStackSize, LPFIBER_START_ROUTINE
# lpStartAddress, LPVOID lpParameter) - VOID CALLBACK FiberProc(LPVOID) is
# the fixed fiber-entry shape, same Ptr[Callable[...]] pattern as
# CreateThread's lpStartAddress above (just void-returning, not u32).
@extern('kernel32', 'CreateFiber')
def CreateFiber(
	dwStackSize: usize,
	lpStartAddress: Ptr[Callable[[Ptr[None]], None]],
	lpParameter: Ptr[None],
) -> Ptr[None]:
	...

@extern('kernel32', 'ConvertThreadToFiber')
def ConvertThreadToFiber(
	lpParameter: Ptr[None],
) -> Ptr[None]:
	...


# ---------------------------------------------------------------------------
# TLS (thread-local storage) - lib/threading.py's own ThreadLocal[T].
# ---------------------------------------------------------------------------

# TlsAlloc's own documented failure sentinel: (DWORD)0xFFFFFFFF, not 0 -
# a freshly-allocated index of 0 is a completely ordinary, valid result
TLS_OUT_OF_INDEXES: u32 = u32( -1 )

@extern('kernel32', 'TlsAlloc')
def TlsAlloc() -> u32:
	...

@extern('kernel32', 'TlsGetValue')
def TlsGetValue(
	dwTlsIndex: u32,
) -> Ptr[None]:
	...

@extern('kernel32', 'TlsSetValue')
def TlsSetValue(
	dwTlsIndex: u32,
	lpTlsValue: Ptr[None],
) -> bool:
	...

@extern('kernel32', 'TlsFree')
def TlsFree(
	dwTlsIndex: u32,
) -> bool:
	...

@extern('kernel32', 'SwitchToFiber')
def SwitchToFiber(
	lpFiber: Ptr[None],
) -> None:
	...

@extern('kernel32', 'DeleteFiber')
def DeleteFiber(
	lpFiber: Ptr[None],
) -> None:
	...

# UINT WINAPI SetErrorMode(UINT uMode) - called before deliberately driving
# a process into an unhandled SEH exception (e.g. a stack-overflow-into-
# guard-page test), so the crash exits promptly with its NTSTATUS as the
# process exit code instead of popping a blocking Windows Error Reporting
# dialog that would otherwise hang the caller until it times out.
SEM_FAILCRITICALERRORS: u32 = 0x0001
SEM_NOGPFAULTERRORBOX:  u32 = 0x0002

@extern('kernel32', 'SetErrorMode')
def SetErrorMode(
	uMode: u32,
) -> u32:
	...
