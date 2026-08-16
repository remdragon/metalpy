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
		# 0xCD ("uninitialized" - MSVC debug heap's own convention), not a
		# zero-fill: correctness must never depend on freshly allocated
		# memory happening to read as zero/null - release builds get real,
		# unfilled garbage here (this whole block is debug-only), so any
		# code relying on implicit zero-init would only ever "work" in
		# debug and corrupt memory in release. Filling with a nonzero
		# poison byte instead makes that class of bug reproduce in BOTH
		# configurations - a null-pointer read/deref would misleadingly
		# "just work" here otherwise. Also fixes a real, separate bug this
		# call used to have: `count` is the number of T-sized ELEMENTS
		# (e.g. 1 for a single object), not the allocation's own byte size
		# - passing it here left everything past the first `count` bytes
		# of a multi-byte T completely unfilled
		mempoison( ptr, byte_count )
	return ptr

# ---------------------------------------------------------------------------
# stdout/stderr: minimal stream objects (see TODO.txt - full IO interfaces,
# including buffering and reading, are still future work; this is just
# enough for print() and lib/logging.py's StreamHandler).
# ---------------------------------------------------------------------------

class _Stdout:
	@compiler.target( os = 'windows' )
	def write( self, s: str ) -> Result[None,OSError]:
		from fs import write_all
		from windows.kernel32 import GetStdHandle, STD_OUTPUT_HANDLE
		return write_all( GetStdHandle( STD_OUTPUT_HANDLE ), s.get_cstr(), s.byte_len() )

	@compiler.target( os = not 'windows' )
	def write( self, s: str ) -> Result[None,OSError]:
		from fs import write_all
		return write_all( 1, s.get_cstr(), s.byte_len() ) # STDOUT_FILENO is 1

stdout: _Stdout = _Stdout()

class _Stderr:
	@compiler.target( os = 'windows' )
	def write( self, s: str ) -> Result[None,OSError]:
		from fs import write_all
		from windows.kernel32 import GetStdHandle, STD_ERROR_HANDLE
		return write_all( GetStdHandle( STD_ERROR_HANDLE ), s.get_cstr(), s.byte_len() )

	@compiler.target( os = not 'windows' )
	def write( self, s: str ) -> Result[None,OSError]:
		from fs import write_all
		return write_all( 2, s.get_cstr(), s.byte_len() ) # STDERR_FILENO is 2

stderr: _Stderr = _Stderr()

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

# debug-only "uninitialized" poison fill - see alloc[T]'s own comment on why
# this is deliberately NOT zero
@compiler.target( os = 'windows' )
def mempoison( ptr: Ptr[u8], count: usize ) -> Ptr[u8]:
	from windows.ntdll import RtlFillMemory
	RtlFillMemory( ptr, count, 0xCD )
	return ptr

@compiler.target( os = not 'windows' )
def mempoison( ptr: Ptr[u8], count: usize ) -> Ptr[u8]:
	from crt import memset
	memset( ptr, 0xCD, count )
	return ptr

@compiler.target( os = 'windows' )
def memcmp( a: ConstPtr[u8], b: ConstPtr[u8], count: usize ) -> i32:
	from windows.ntdll import RtlCompareMemory
	matched: usize = RtlCompareMemory( a, b, count )
	if matched == count:
		return 0
	if a[matched] < b[matched]:
		return -1
	return 1

@compiler.target( os = not 'windows' )
def memcmp( a: ConstPtr[u8], b: ConstPtr[u8], count: usize ) -> i32:
	from crt import memcmp as _crt_memcmp
	return _crt_memcmp( a, b, count )

@compiler.target( os = 'windows' )
def exit( code: u32 ) -> NoReturn:
	from windows.kernel32 import ExitProcess
	ExitProcess( code )

@compiler.target( os = not 'windows' )
def exit( code: u32 ) -> NoReturn:
	from crt import _exit
	# POSIX _exit(int status) takes a signed int - code is u32 (matches
	# Windows' own u32 exit-code convention, see the sibling branch above),
	# an explicit narrowing/sign-changing cast either way
	with compiler.wrap_arithmetic:
		_exit( i32( code ) )

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
	exit( 1 )

def _assert( cond: bool, msg: str ) -> None:
	# TODO FIXME: change type_discovery.py to emit this logic directly
	if not cond:
		panic( msg )

# private helper functions:

@compiler.target( os = 'windows' )
def _alloc( size: usize ) -> Ptr[u8]:
	from windows.kernel32 import HeapAlloc, GetProcessHeap
	ptr = HeapAlloc( GetProcessHeap(), 0, size )
	return ptr

@compiler.target( os = not 'windows' )
def _alloc( size: usize ) -> Ptr[u8]:
	from crt import malloc
	ptr = malloc( size )
	return ptr

@compiler.target( os = 'windows' )
def _write_stderr_cstr( msg: ConstPtr[u8], length: usize ) -> None:
	from windows.kernel32 import GetStdHandle, WriteFile, STD_ERROR_HANDLE, INVALID_HANDLE_VALUE
	handle = GetStdHandle( STD_ERROR_HANDLE )
	if handle != INVALID_HANDLE_VALUE:
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
