import compiler
import fs

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
		# debug-mode alloc-site tracking (dump_live_objects) - records this
		# buffer in a SIDE-TABLE tracking node (compiler.__debug_raw_track__,
		# emitter_c.py's __metalpy_debug_raw_track), not a header prefixed
		# onto the real block: `ptr` itself is left completely unchanged
		# (some existing code queries the OS allocator directly on a
		# sys.alloc'd pointer - see free()'s own HeapSize/malloc_usable_size
		# calls just below - so it has to stay the EXACT pointer the OS
		# allocator returned). No per-call-site location is captured here
		# (unlike RC objects - see ir.Allocate.loc/emitter_c.py's
		# dump_live_objects support): this is a LIBRARY-INTERNAL call site
		# (every list/dict/... backing buffer in the whole program funnels
		# through this one alloc[T]), not the user's own call site, and
		# there's no caller-location intrinsic in this codebase to reach
		# past it - dump_live_objects reports these aggregated as one
		# generic "raw sys.alloc buffers" group instead (count + total live
		# bytes), not broken down further. Good enough for the leak-hunting
		# this feature exists for; a real per-site breakdown is future work
		# if it's ever needed.
		compiler.__debug_raw_track__( ptr, byte_count )
		# 0xCD ("uninitialized" - MSVC debug heap's own convention)
		# this helps catch bugs like use-after-free
		# we don't zero-fill because that can also hide bugs
		mempoison( ptr, byte_count )
	return ptr

def dump_live_objects() -> None:
	''' debug builds only: prints every still-live RC object and sys.alloc[T]
	raw buffer, grouped by (class, allocation site) for RC objects (a single
	aggregated group for raw buffers - see alloc[T]'s own comment on why),
	with a count and total byte size per group. Call this near the end of
	main() to check for leaks - there is no atexit/CRT-teardown hook (works
	the same under --no-crt), so nothing calls it automatically. A no-op in
	a release build: no allocation-site tracking exists there to dump (the
	`if` below folds away entirely - compiler.dump_live_objects() itself is a
	hard compile error if reached outside a debug build). '''
	if compiler.target.debug:
		compiler.dump_live_objects()

# ---------------------------------------------------------------------------
# stdout/stderr: minimal buffered stream objects (see TODO.txt - full IO
# interfaces, including reading, are still future work; this is just enough
# for print() and lib/logging.py's StreamHandler). Line-buffered when the
# destination is a real terminal (flush on '\n', so interactive output stays
# live), block-buffered otherwise (flush once _STDIO_BUF_CAP fills) - same
# convention as Python/C stdio, far fewer syscalls than a raw write-per-call
# for redirected/piped output. Flushed on every real exit path via
# _flush_stdio() below - see exit()'s own call to it, compiler.py's
# Compiler.run() force_reachable('sys', '_flush_stdio'), and emitter_c.py's
# call from __metalpy_main right after the user's main() returns. NOT flushed
# on a crash (SIGSEGV/unhandled exception): _PROLOGUE_CRASH_HANDLER
# deliberately avoids calling into any metalpy-level code once the process
# might be in a corrupted state (see its own comment in emitter_c.py) - any
# buffered-but-unflushed output from immediately before a crash is a known,
# accepted loss.
# ---------------------------------------------------------------------------

_STDIO_BUF_CAP: usize = 8192

class _BufferedStream:
	''' buffered wrapper around a raw fs.FD - shared by stdout/stderr today,
	general enough to become the base of real buffered file I/O later. '''
	_buf: Ptr[u8] = None
	_len: usize = 0
	_is_tty: bool = False
	_tty_checked: bool = False
	_fd: fs.FD = fs.INVALID_FD

	def __init__( self, fd: fs.FD ) -> None:
		self._fd = fd

	@compiler.target( os = 'windows' )
	def _check_tty( self ) -> None:
		from windows.kernel32 import GetConsoleMode
		mode: u32 = 0
		self._is_tty = GetConsoleMode( self._fd, compiler.addrof( mode ))
		self._tty_checked = True

	@compiler.target( os = not 'windows' )
	def _check_tty( self ) -> None:
		from crt import isatty
		self._is_tty = isatty( self._fd ) != 0
		self._tty_checked = True

	def flush( self ) -> Result[None,OSError]:
		if self._len == 0:
			return Result.Ok( None )
		fs.write_all( self._fd, self._buf, self._len ).or_return()
		self._len = 0
		return Result.Ok( None )

	def write( self, s: str ) -> Result[None,OSError]:
		if not self._tty_checked:
			self._check_tty()
		if self._buf is None:
			self._buf = alloc[u8]( _STDIO_BUF_CAP )
		n: usize = s.byte_len()
		src: ConstPtr[u8] = s.get_cstr()
		offset: usize = 0
		# all bounded by _STDIO_BUF_CAP/n above - never actually overflows,
		# same wrap_arithmetic-for-a-provably-safe-loop posture as fs.py's
		# own write_all
		with compiler.wrap_arithmetic:
			while offset < n:
				space: usize = _STDIO_BUF_CAP - self._len
				if space == 0:
					self.flush().or_return()
					space = _STDIO_BUF_CAP
				chunk: usize = n - offset
				if chunk > space:
					chunk = space
				memcpy( self._buf + self._len, src + offset, chunk )
				self._len += chunk
				offset += chunk
			if self._is_tty and n > 0 and src[ n - 1 ] == u8( 10 ): # '\n' - keep interactive output live
				self.flush().or_return()
		return Result.Ok( None )

	def __del__( self ) -> None:
		''' flush then free the backing buffer - a destructor can't propagate
		flush() failure (see lib/builtins/__File.py's own __del__ comment),
		deliberately ignored, not silently unchecked. Guarded so a second call
		is a no-op: _flush_stdio calls this directly on the stdout/stderr
		singletons (which never reach refcount 0 in a release build), and a
		future real File wrapping this class would ALSO get here for free via
		the ordinary RC destructor when its last reference drops - no manual
		cleanup call needed there. '''
		self.flush().is_ok()
		if self._buf is not None:
			free( self._buf )
			self._buf = None

@compiler.target( os = 'windows' )
def _stdout_fd() -> fs.FD:
	from windows.kernel32 import GetStdHandle, STD_OUTPUT_HANDLE
	return GetStdHandle( STD_OUTPUT_HANDLE )

@compiler.target( os = not 'windows' )
def _stdout_fd() -> fs.FD:
	return 1 # STDOUT_FILENO

@compiler.target( os = 'windows' )
def _stderr_fd() -> fs.FD:
	from windows.kernel32 import GetStdHandle, STD_ERROR_HANDLE
	return GetStdHandle( STD_ERROR_HANDLE )

@compiler.target( os = not 'windows' )
def _stderr_fd() -> fs.FD:
	return 2 # STDERR_FILENO

stdout: _BufferedStream = _BufferedStream( _stdout_fd() )
stderr: _BufferedStream = _BufferedStream( _stderr_fd() )

def _flush_stdio() -> None:
	''' force-called from every real exit path - see this module's own
	header comment above for why and where. Calls __del__ directly rather
	than waiting on refcounting (stdout/stderr are release-build-eternal
	globals - see _BufferedStream.__del__'s own comment) - this is always
	the last real use of them. '''
	stdout.__del__()
	stderr.__del__()

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
		# drop alloc[T]'s own debug-mode side-table tracking entry (see its
		# comment) - `ptr` itself is untouched, still the exact block
		# HeapAlloc returned
		compiler.__debug_raw_untrack__( ptr )
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
		# don't actually hand `ptr` back to the allocator yet - the pit
		# (compiler.__debug_quarantine__) holds it, poisoned, unreused, for
		# ~1000 more frees, so a stale double-free/UAF touch lands on
		# reliably-poisoned memory instead of memory the allocator has
		# already reused for something unrelated (confirmed necessary by a
		# real repro: without this, the SAME poisoned block silently went
		# back to exactly-zero content between two frees). Only what the pit
		# evicts to make room actually gets freed for real here.
		evicted: Ptr[u8] = compiler.__debug_quarantine__( ptr )
		if evicted is not None:
			HeapFree( heap, 0, evicted )
		return
	HeapFree( heap, 0, ptr )

@compiler.target( os = not 'windows' )
def free( ptr: Ptr[None] ) -> None:
	from crt import free as _crt_free, malloc_usable_size as _crt_malloc_usable_size
	if compiler.target.debug:
		# see the Windows branch's own comment above - same untrack-then-
		# mempoison-then-quarantine sequence
		compiler.__debug_raw_untrack__( ptr )
		size: usize = _crt_malloc_usable_size( ptr )
		mempoison( ptr, size )
		evicted: Ptr[None] = compiler.__debug_quarantine__( ptr )
		if evicted is not None:
			_crt_free( evicted )
		return
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

# _raw_exit: bare process termination, no flush. Package-private - the only
# other caller is emitter_c.py's own no_crt Windows mainCRTStartup synthesis,
# which calls __metalpy_main() (already runs flush_call + __metalpy_deinit -
# see emitter_c.py's own comment) and then needs ONLY to terminate the
# process, not flush again: by that point __metalpy_deinit has already
# released the RC-global stdout/stderr objects themselves, so a second
# _flush_stdio() call here would read self._buf off already-freed memory - a
# real use-after-free, confirmed via a genuine STATUS_HEAP_CORRUPTION
# repro (free() handed the 0xCD mempoison pattern as a pointer) before this
# split existed.
@compiler.target( os = 'windows' )
def _raw_exit( code: i32 ) -> NoReturn:
	from windows.kernel32 import ExitProcess
	ExitProcess( u32( code ))

@compiler.target( os = not 'windows' )
def _raw_exit( code: i32 ) -> NoReturn:
	from crt import _exit
	_exit( code )

def exit( code: i32 ) -> NoReturn:
	_flush_stdio() # buffered stdout/stderr - see their own header comment
	_raw_exit( code )

def panic( message: str ) -> NoReturn:
	# TODO: route through a real `stderr` stream once IO interfaces exist (see
	# TODO.txt); for now always use the OS low-level unbuffered write (no
	# allocations - important, since panic must still work when the reason
	# we're here is an allocation failure - get_cstr()/byte_len() are both
	# plain field reads, same as _BufferedStream.write's identical pattern). Every
	# real call site (sys.alloc's 'out of memory', Result.unwrap's errmsg,
	# ...) already passes a real `str`, never a raw ConstPtr[u8] - this used
	# to be typed ConstPtr[u8] anyway, which happened to go unnoticed under
	# MSVC's laxer pointer-type checking but is a hard error under GCC
	# (-Wincompatible-pointer-types, promoted to an error by default on
	# recent GCC).
	# flush any already-buffered stdout FIRST - allocation-free (flush()
	# never allocates, only the first write() to an empty stream does), so
	# this doesn't compromise the OOM-safety this function needs; keeps
	# ordinary program output from appearing AFTER this crash message on a
	# shared terminal.
	stdout.flush().is_ok()
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
		# see _BufferedStream.write's identical comment - u32(length) is a real
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
# Windows: GetCommandLineW()+CommandLineToArgvW() (kernel32/shell32) read the
# process' own command line directly, independent of main(argc,argv) - a
# no_crt/freestanding build's mainCRTStartup calls main(0, NULL) (see
# emitter_c.py's own comment there), so relying on the C-level argc/argv
# there would leave sys.argv silently empty regardless of the real command
# line. This works identically whether or not the CRT is linked.
#
# Everywhere else: _raw_argc/_raw_argv are written DIRECTLY (raw C
# assignment, not through any metalpy-level Assign) by emit_c()'s own
# entry-point prelude, as the very first statements of main() - before
# __metalpy_init() (which is what actually calls _build_argv() below, via
# this file's own `argv: list[str] = _build_argv()` global) ever runs. POSIX
# targets always link a real, CRT-provided main() in this codebase (no
# freestanding entry point exists there), so this is safe unconditionally.
# ---------------------------------------------------------------------------

_raw_argc: i32 = 0
_raw_argv: Ptr[Ptr[u8]] = None

@compiler.target( os = 'windows' )
def _build_argv() -> list[str]:
	from windows.kernel32 import GetCommandLineW, LocalFree
	from windows.shell32 import CommandLineToArgvW

	result: list[str] = list[str]()
	argc: i32 = 0
	argv_w: Ptr[Ptr[u16]] = CommandLineToArgvW( GetCommandLineW(), compiler.addrof( argc ))
	if argv_w is None:
		return result
	defer( LocalFree( compiler.cast( Ptr[None], argv_w )))
	if argc > 0:
		with compiler.panic_arithmetic( 'argc is never negative once positive-checked above' ):
			count: usize = usize( argc )
		i: usize = 0
		with compiler.panic_arithmetic( 'bounded by count, cannot overflow' ):
			while i < count:
				w: ConstPtr[u16] = argv_w[i]
				s: str = str.from_utf16( w, 1_000_000 ).unwrap( 'sys.argv: invalid UTF-16 in argument' )
				result.append( s )
				i += 1
	return result

@compiler.target( os = not 'windows' )
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
			result.append( s )
			i += 1
	return result

argv: list[str] = _build_argv()
