HANDLE: TypeAlias = Ptr[None]

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

@extern('kernel32', 'GetProcessHeap')
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
