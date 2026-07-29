HANDLE: TypeAlias = Ptr[None]

@extern('ntdll', 'RtlCopyMemory')
def RtlCopyMemory(
	Destination: Ptr[u8],
	Source: ConstPtr[u8],
	Length: usize,
) -> None:
	...

@extern('ntdll', 'RtlExitUserProcess')
def RtlExitUserProcess(
	ExitCode: u32,
) -> None:
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
