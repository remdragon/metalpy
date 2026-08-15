HANDLE: TypeAlias = Ptr[None]

# INVALID_HANDLE_VALUE from winbase.h: (HANDLE)(LONG_PTR)-1
INVALID_HANDLE_VALUE: HANDLE = -1

STD_INPUT_HANDLE: u32 = u32( -10 )
STD_OUTPUT_HANDLE: u32 = u32( -11 )
STD_ERROR_HANDLE: u32 = u32( -12 )

_TZNAME_SIZE: u32 = 32

@cstruct
class DynamicTimeZoneInformation:
	Bias: i32
	StandardName: u16[_TZNAME_SIZE]
	StandardDate: u16[8]
	StandardBias: i32
	DaylightName: u16[_TZNAME_SIZE]
	DaylightDate: u16[8]
	DaylightBias: i32
	TimeZoneKeyName: u16[_TZNAME_SIZE]
	DynamicDaylightTimeDisabled: u8
	_pad: u8[3]


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

@extern( 'kernel32', 'ExitProcess' )
def ExitProcess(
	uExitCode: u32,
) -> None:
	...

@extern( 'kernel32', 'GetLastError' )
def GetLastError() -> u32:
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
