HANDLE: TypeAlias = Ptr[None]

@extern('ntdll', 'RtlCopyMemory')
def RtlCopyMemory(
	Destination: Ptr[u8],
	Source: ConstPtr[u8],
	Length: usize,
) -> None:
	...

# ntdll.dll internally links a small set of C-runtime comparison functions
# as well (confirmed via dumpbin /exports ntdll.dll) — using these keeps
# Windows builds CRT-free, same as memcpy/memmove/memzero above.
@extern( 'ntdll', 'RtlCompareMemory' )
def RtlCompareMemory(
	Source1: ConstPtr[u8],
	Source2: ConstPtr[u8],
	Length: usize,
) -> usize:
	...

@extern( 'ntdll', 'RtlMoveMemory' )
def RtlMoveMemory(
	Destination: Ptr[u8],
	Source: ConstPtr[u8],
	Length: usize,
) -> None:
	...

@extern( 'ntdll', 'RtlZeroMemory' )
def RtlZeroMemory(
	ptr: Ptr[u8],
	length: usize,
) -> None:
	...

@extern( 'ntdll', 'RtlFillMemory' )
def RtlFillMemory(
	ptr: Ptr[u8],
	length: usize,
	fill: u8,
) -> None:
	...

# ntdll.dll internally links a small set of plain C-runtime string functions
# and exports them (confirmed via `dumpbin /exports ntdll.dll`: strlen,
# strnlen, strcpy, strcmp, ... are all present). Calling these instead of
# linking msvcrt/ucrt keeps Windows builds free of the C runtime entirely -
# ntdll is always loaded in every Windows process regardless.
@extern( 'ntdll', 'strnlen' )
def strnlen(
	str: ConstPtr[u8],
	maxlen: usize,
) -> usize:
	...
