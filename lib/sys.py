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
		# 0xCD ("uninitialized" - MSVC debug heap's own convention)
		# this helps catch bugs like use-after-free
		# we don't zero-fill because that can also hide bugs
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
def cpu_count() -> u32:
	''' the number of active logical processors - e.g. for sizing a
	reactor.Reactor's own worker count. Clamped to at least 1 (paranoid
	safety net for a platform reporting 0/unknown, matching Python's own
	`os.cpu_count() or 1` idiom), never fails/panics. '''
	from windows.kernel32 import GetActiveProcessorCount, ALL_PROCESSOR_GROUPS
	n: u32 = GetActiveProcessorCount( ALL_PROCESSOR_GROUPS )
	if n == 0:
		return 1
	return n

@compiler.target( os = not 'windows' )
def cpu_count() -> u32:
	from posix.unistd import sysconf, _SC_NPROCESSORS_ONLN
	n: i64 = sysconf( _SC_NPROCESSORS_ONLN )
	if n <= 0:
		return 1
	with compiler.wrap_arithmetic:
		return u32( n )

@compiler.target( os = 'windows' )
def free( ptr: Ptr[u8] ) -> None:
	from windows.kernel32 import GetProcessHeap, HeapFree, HeapSize, HEAP_SIZE_FAILED
	heap = GetProcessHeap()
	if compiler.target.debug:
		# same "0xCD before the real free" idea as alloc[T]'s own mempoison
		# call above, just on the other end of the block's lifetime - this
		# is what actually catches a use-after-free (reading/writing
		# through a stale pointer after this point now reliably sees
		# poison instead of whatever the allocator happened to leave
		# behind, on every compiler, not just the ones whose own debug
		# heap already does this)
		size: usize = HeapSize( heap, 0, ptr )
		if size != HEAP_SIZE_FAILED:
			mempoison( ptr, size )
	HeapFree( heap, 0, ptr )

@compiler.target( os = not 'windows' )
def free( ptr: Ptr[None] ) -> None:
	from crt import free as _crt_free, malloc_usable_size as _crt_malloc_usable_size
	if compiler.target.debug:
		# see the Windows branch's own comment above - same mempoison-
		# before-free, via glibc/macOS's own "how big was this block"
		# query (crt.malloc_usable_size, os-split there)
		size: usize = _crt_malloc_usable_size( ptr )
		mempoison( ptr, size )
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
def memset( ptr: Ptr[u8], fill: u8, count: usize ) -> Ptr[u8]:
	from windows.ntdll import RtlFillMemory
	RtlFillMemory( ptr, count, fill )
	return ptr

@compiler.target( os = not 'windows' )
def memset( ptr: Ptr[u8], fill: u8, count: usize ) -> Ptr[u8]:
	from crt import memset as _memset
	_memset( ptr, i32( fill ), count )
	return ptr

@compiler.target( os = 'windows' )
def memzero( ptr: Ptr[u8], count: usize ) -> Ptr[u8]:
	from windows.ntdll import RtlZeroMemory
	RtlZeroMemory( ptr, count )
	return ptr

@compiler.target( os = not 'windows' )
def memzero( ptr: Ptr[u8], count: usize ) -> Ptr[u8]:
	memset( ptr, 0, count )
	return ptr

def mempoison( ptr: Ptr[u8], count: usize ) -> Ptr[u8]:
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
def exit( code: i32 ) -> NoReturn:
	from windows.kernel32 import ExitProcess
	ExitProcess( u32( code ))

@compiler.target( os = not 'windows' )
def exit( code: i32 ) -> NoReturn:
	from crt import _exit
	_exit( code )

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

# raw, untyped allocation - the byte-counted primitive sys.alloc[T] itself
# builds on (below). Package-private (not module-private): lib/threading.py's
# FastLock genuinely needs it directly - a raw Ptr[u8] of an exact byte size,
# with none of sys.alloc[T]'s own generic-sizing/panic-on-OOM/debug-poisoning
# behavior (the memory becomes a real OS mutex, which pthread_mutex_init/
# SRWLOCK's own all-zero-is-unlocked contract must initialize on its own
# terms). sys.py and threading.py are both bare top-level lib/ modules (no
# enclosing package of their own) - module/package privacy enforcement
# (SYNTAX.md) treats every such module as one implicit shared package, so a
# single leading underscore already covers this cross-file reach correctly;
# no need to go fully public.
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

# ---------------------------------------------------------------------------
# argv: command-line arguments (argv[0] included, matching real Python)
#
# _raw_argc/_raw_argv are written DIRECTLY (raw C assignment, not through
# any metalpy-level Assign) by emit_c()'s own entry-point prelude, as the
# very first statements of main() - before __metalpy_init() (which is what
# actually calls _build_argv() below, via this file's own `argv: list[str]
# = _build_argv()` global) ever runs. Real argc/argv are only available at
# a normal CRT-linked entry; a no_crt/freestanding build has no OS-provided
# values to capture (mainCRTStartup calls main(0, NULL)), so argv is just
# empty there - an accepted limitation, not a bug (GetCommandLineA()+manual
# parsing would be the way to add it later if that's ever needed).
# ---------------------------------------------------------------------------

_raw_argc: i32 = 0
_raw_argv: Ptr[Ptr[u8]] = None

def _build_argv() -> list[str]:
	result: list[str] = list[str]()
	if _raw_argc <= 0:
		return result
	with compiler.panic_arithmetic( 'argc is never negative once positive-checked above' ):
		count: usize = usize( _raw_argc )
	i: usize = 0
	with compiler.panic_arithmetic( 'bounded by count/cstrlen, cannot overflow' ):
		while i < count:
			raw: ConstPtr[u8] = compiler.cast( ConstPtr[u8], _raw_argv[i] )
			n: usize = cstrlen( raw, 1_000_000 )
			size: usize = n + 1
			s: str = str.from_cstr( raw, size ).unwrap( 'sys.argv: invalid UTF-8 in argument' )
			result.append( s ).unwrap( 'sys.argv: too many arguments' )
			i += 1
	return result

argv: list[str] = _build_argv()
