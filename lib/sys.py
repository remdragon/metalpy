import compiler

@union
class OwnershipError[T]:
	SharedReference: T
	# NOTE: the following aren't used and are left-overs from OwnershipError being an @enum
	#AlreadyBorrowed = 1
	#UseAfterFree = 2
	#DanglingReference = 3
	#Other = _

def alloc[T]( count: usize ) -> Ptr[T]:
	with compiler.panic_arithmetic( 'allocation size overflow' ):
		byte_count: usize = count * compiler.sizeof( T )
	ptr = _alloc( byte_count )
	if ptr is None:
		panic( 'out of memory' )
	if compiler.target.debug:
		memzero( ptr, count )
	return ptr

# ---------------------------------------------------------------------------
# stdout: minimal stream object (see TODO.txt - full IO interfaces, including
# a real stderr, buffering, and reading, are still future work; this is just
# enough for print()).
# ---------------------------------------------------------------------------

class _Stdout:
	@compiler.target( os = 'windows' )
	def write( self, s: str ) -> Result[u32,OSError]:
		from windows.kernel32 import GetLastError, GetStdHandle, WriteFile, STD_OUTPUT_HANDLE
		written: u32 = 0
		# u32(s.byte_len()) is a real narrowing cast (usize -> u32) - this
		# function returns None, so it can't propagate Check mode's default
		# Result[u32,OverflowError]; a write() call writing >4GB in one
		# syscall isn't a real scenario, so wrap (silent truncation) is the
		# pragmatic choice here, same as any C caller of WriteFile would make
		with compiler.wrap_arithmetic:
			success: bool = WriteFile(
				GetStdHandle( STD_OUTPUT_HANDLE ),
				s.get_cstr(),
				u32( s.byte_len() ),
				compiler.addrof( written ),
				None, # lpOverlapped
			)
			if success:
				return Result.Ok( written )
			else:
				dw: u32 = GetLastError()
				return Result.Err( OSError( dw ))

	@compiler.target( os = not 'windows' )
	def write( self, s: str ) -> Result[isize,OSError]:
		from crt import get_errno, write as _crt_write
		written: isize = _crt_write( 1, s.get_cstr(), s.byte_len() ) # STDOUT_FILENO is 1
		if written < isize( 0 ):
			err = get_errno()
			return Result.Err( OSError( err ))
		else:
			return Result.Ok( written )

stdout: _Stdout = _Stdout()

@compiler.target( os = 'windows' )
def cstrlen( ptr: ConstPtr[u8], max_length: usize ) -> usize:
	# ntdll exports a plain strnlen (confirmed via dumpbin /exports) - use that
	# rather than the C runtime, so Windows builds stay CRT-free.
	from windows.ntdll import strnlen
	return strnlen( ptr, max_length )

@compiler.target( os = not 'windows' )
def cstrlen( ptr: ConstPtr[u8], max_length: usize ) -> usize:
	from crt import strnlen
	return strnlen( ptr, max_length )

@compiler.target( os = 'windows' )
def free( ptr: Ptr[u8] ) -> None:
	from windows.kernel32 import GetProcessHeap, HeapFree
	HeapFree( GetProcessHeap(), 0, ptr )

@compiler.target( os = not 'windows' )
def free( ptr: Ptr[None] ) -> None:
	from crt import free as _crt_free
	_crt_free(ptr)

@compiler.target( os = 'windows' )
def memcpy( dest: Ptr[u8], src: ConstPtr[u8], count: usize ) -> Ptr[u8]:
	from windows.ntdll import RtlCopyMemory
	RtlCopyMemory( dest, src, count )
	return dest

@compiler.target( os = not 'windows' )
def memcpy( dest: Ptr[u8], src: ConstPtr[u8], count: usize ) -> Ptr[u8]:
	from crt import memcpy as _crt_memcpy
	return _crt_memcpy( dest, src, count )

@compiler.target( os = 'windows' )
def memmove( dest: Ptr[u8], src: ConstPtr[u8], count: usize ) -> Ptr[u8]:
	from windows.ntdll import RtlMoveMemory
	RtlMoveMemory( dest, src, count )
	return dest

@compiler.target( os = not 'windows' )
def memmove( dest: Ptr[u8], src: ConstPtr[u8], count: usize ) -> Ptr[u8]:
	from crt import memmove as _crt_memmove
	return _crt_memmove( dest, src, count )

@compiler.target( os = 'windows' )
def memzero( ptr: Ptr[u8], count: usize ) -> Ptr[u8]:
	from windows.ntdll import RtlZeroMemory
	RtlZeroMemory( ptr, count )
	return ptr

@compiler.target( os = not 'windows' )
def memzero( ptr: Ptr[u8], count: usize ) -> Ptr[u8]:
	from crt import memset
	memset( ptr, 0, count )
	return ptr

def panic( message: str ) -> NoReturn:
	# TODO: route through a real `stderr` stream once IO interfaces exist (see
	# TODO.txt); for now always use the OS low-level unbuffered write (no
	# allocations - important, since panic must still work when the reason
	# we're here is an allocation failure - get_cstr()/byte_len() are both
	# plain field reads, same as _Stdout.write's identical pattern). Every
	# real call site (sys.alloc's 'out of memory', Result.unwrap's errmsg,
	# ...) already passes a real `str`, never a raw ConstPtr[u8] - this used
	# to be typed ConstPtr[u8] anyway, which happened to go unnoticed under
	# MSVC's laxer pointer-type checking but is a hard error under GCC
	# (-Wincompatible-pointer-types, promoted to an error by default on
	# recent GCC).
	_write_stderr_cstr( message.get_cstr(), message.byte_len() )
	_exit_process( 1 )

# private helper functions:

@compiler.target( os = 'windows' )
def _alloc( size: usize ) -> Ptr[u8]|None:
	from windows.kernel32 import HeapAlloc, GetProcessHeap
	ptr = HeapAlloc( GetProcessHeap(), 0, size )
	return ptr

@compiler.target( os = not 'windows' )
def _alloc( size: usize ) -> Ptr[u8]|None:
	from crt import malloc
	ptr = malloc( size )
	return ptr

@compiler.target( os = 'windows' )
def _exit_process( code: u32 ) -> None:
	from windows.ntdll import RtlExitUserProcess
	RtlExitUserProcess( code )

@compiler.target( os = not 'windows' )
def _exit_process( code: u32 ) -> None:
	from crt import _exit
	_exit( code )

@compiler.target( os = 'windows' )
def _write_stderr_cstr( msg: ConstPtr[u8], length: usize ) -> None:
	from windows.kernel32 import GetStdHandle, WriteFile, STD_ERROR_HANDLE
	handle = GetStdHandle( STD_ERROR_HANDLE )
	if handle != 0 and handle != -1:
		written: u32 = 0
		# see _Stdout.write's identical comment - u32(length) is a real
		# narrowing cast (usize -> u32); this function returns None, so it
		# can't propagate Check mode's Result[u32,OverflowError]
		with compiler.wrap_arithmetic:
			WriteFile( handle, msg, u32( length ), compiler.addrof( written ), None )

@compiler.target( os = not 'windows' )
def _write_stderr_cstr( msg: ConstPtr[u8], length: usize ) -> None:
	from crt import write
	write( 2, msg, length ) # STDERR_FILENO is 2
