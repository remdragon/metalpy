import compiler
import fs
import threading
import queue
import atomic

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

# mirrors emitter_c.py's own #define METALPY_IMMORTAL_REFCOUNT EXACTLY - a
# compile-time-baked immortal object (a string literal, a static vtable
# instance, ...) has its header ref_count set to this sentinel and
# retain/release skip it entirely (see emitter_c.py's own
# _field_lock_prologue comment). Duplicated here rather than derived from a
# shared source: the #define lives in generated C text, nowhere a
# metalpy-level import could reach.
IMMORTAL_REFCOUNT: usize = 2147483647

def debug_register_immortal_cache( slot: Ptr[Ptr[u8]] ) -> None:
	''' debug-only, no-op in release: call right after publishing a
	lazily-computed cache into a field on a compile-time-immortal object
	(compiler.refcount(self) == IMMORTAL_REFCOUNT). Such a field's owner
	is never destructed (retain/release skip IMMORTAL_REFCOUNT objects
	entirely), so its cache would otherwise report as a permanent
	false-positive "leak" to dump_live_objects() even though it's an
	intentional, one-time, program-lifetime cache, not a bug.

	`slot` is the field's own address (e.g.
	compiler.addrof(self.__utf16), cast to Ptr[Ptr[u8]]) - NOT the cached
	pointer's value - compiler.__debug_track_immortal_cache__'s own
	side-table (emitter_c.py's __metalpy_dump_live_objects) frees AND
	resets it right before every dump_live_objects() report, keeping
	repeated calls in one program correct: a later cache-miss just
	re-populates and re-registers the same slot. This has to live at the
	compiler/emitter level, not as a plain metalpy-level list drained from
	here - the automatic end-of-program leak-check epilogue
	(__metalpy_deinit) calls compiler.dump_live_objects() directly, not
	this wrapper, so a list owned only here would never get drained on
	that path (confirmed via a real repro: a program that never calls
	sys.dump_live_objects() itself still reported these as leaked).

	NEVER call this for a non-immortal object's field: nothing guarantees
	that slot address stays valid until the drain runs. '''
	if compiler.target.debug:
		compiler.__debug_track_immortal_cache__( slot )

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
	general enough to become the base of real buffered file I/O later.

	write() is @virtual specifically so _ThreadedStream (below) can
	override it - see that class's own docstring for why it's a SEPARATE
	subclass rather than fields/methods added directly here. A program
	that never calls enable_threaded_stdout()/enable_threaded_stderr()
	never references _ThreadedStream at all, so nothing under it
	(queue.Queue/threading.Thread) ever gets reached/lowered - true zero cost
	when unused, unlike an earlier version of this that put the queue/
	thread fields directly on THIS class (that made threading.Thread
	reachable from _BufferedStream's own field layout unconditionally,
	which - confirmed via a real test failure, thread_detection_test.py -
	flipped compiler.spawns_threads-driven field-locking on for every
	program merely constructing sys.stdout, not just ones actually using
	threaded output). Subclassing avoids this entirely: a base-typed
	pointer/handle carries no knowledge of a subclass's own fields at all. '''
	_buf: Ptr[u8] = None
	_len: usize = 0
	_is_tty: bool = False
	_tty_checked: bool = False
	_fd: fs.FD = fs.INVALID_FD

	def __init__( self, fd: fs.FD ) -> None:
		self._fd = fd

	def fd( self ) -> fs.FD:
		''' public accessor for _fd - needed by enable_threaded_stdout()/
		enable_threaded_stderr() (plain module-level functions, not methods
		of this class or a subclass) to hand the same fd to a freshly
		constructed _ThreadedStream; a single-underscore field is only
		reachable from this class or a subclass, not arbitrary module-level
		code, even within this same file. '''
		return self._fd

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

	@virtual
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

	@virtual
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


# a plain, no-reactor blocking sleep - deliberately NOT lib/time.py's own
# sleep() (reactor-aware, and time.py/reactor.py/datetime.py already form a
# real import cycle among themselves - see time.py's own sleep() comment).
# sys.py is foundational (imported by nearly everything), so it stays off
# that graph entirely rather than risk it. Only ever called with a fixed
# 10ms interval (_ThreadedStream._shutdown()'s own poll loop below) -
# hardcoded rather than a general ms parameter to sidestep u32 division
# entirely (not a plain infix op on intrinsics here - would need its own
# fallible-Result handling for no real benefit at this one fixed call site).
@compiler.target( os = 'windows' )
def _blocking_sleep_10ms() -> None:
	from windows.kernel32 import Sleep
	Sleep( u32( 10 ))

@compiler.target( os = not 'windows' )
def _blocking_sleep_10ms() -> None:
	from posix.time import nanosleep, timespec
	req: timespec = timespec( tv_sec = 0, tv_nsec = 10000000 )
	nanosleep( compiler.addrof( req ), None )


class _ThreadedStream( _BufferedStream ):
	''' a _BufferedStream that hands write() off to a dedicated background
	writer thread via queue.Queue[str|None], instead of buffering and
	writing synchronously on the caller's own thread - see
	enable_threaded_stdout()/enable_threaded_stderr() below for how a
	stream gets upgraded to this. A completely separate subclass (not
	fields bolted onto _BufferedStream itself) so that a program which
	never enables this never references queue.Queue/threading.Thread at all -
	see _BufferedStream's own docstring for why that matters (a real,
	confirmed compiler-level interaction, not just tidiness).

	_writer_loop() drains the WHOLE queue at once (queue.Queue.drain()'s
	own contract) and joins every pending string into ONE buffer for ONE
	write_all() call - coalescing many small print()-driven writes into far
	fewer syscalls is the actual point of this feature, not just moving the
	write off the caller's thread. Write errors are NOT propagated back to
	whichever write() call originally enqueued the string - that caller
	already returned Result.Ok() the moment it enqueued, before any real
	syscall ran - a known, accepted tradeoff of async I/O (same posture as
	most queue-backed logging/output libraries). '''
	__queue:       queue.Queue[str|None]
	__writer:      threading.Thread|None = None
	__writer_done: atomic.Atomic[bool]
	__shut_down:   bool = False

	def __init__( self, fd: fs.FD ) -> None:
		super().__init__( fd )
		self.__queue = queue.Queue[str|None]()
		self.__writer_done = atomic.Atomic[bool]( False )

	def start( self ) -> None:
		''' spawns the background writer thread - called once, right after
		construction, by enable_threaded_stdout()/enable_threaded_stderr()
		(NOT from __init__ itself: capturing self in a closure before
		every field of a still-under-construction object is assigned is
		rejected by the compiler - a real safety rule, not a formality,
		since the spawned thread could otherwise start running before
		construction finishes). '''
		def entry() -> None:
			self._writer_loop()
		self.__writer = threading.Thread( entry )

	def _writer_loop( self ) -> None:
		while True:
			batch: UnsafeList[str|None] = self.__queue.drain()
			parts: list[str] = list[str]()
			stop: bool = False
			i: usize = 0
			while i < batch.__len__():
				item: str|None = batch.__getitem__( i ).unwrap( '_writer_loop: batch index in bounds by construction' )
				if item is None:
					stop = True
					break
				parts.append( item )
				with compiler.wrap_arithmetic:
					i += 1
			if parts.__len__() > 0:
				combined: str = ''.join( parts )
				fs.write_all( self._fd, combined.get_cstr(), combined.byte_len() ).is_ok()
			if stop:
				self.__writer_done.store( True )
				return

	@virtual
	def write( self, s: str ) -> Result[None,OSError]:
		# always Ok(None): this queue is unbounded, so put() can never
		# actually fail - a real write() error, if any, surfaces later on
		# the writer thread, not here - see this class's own docstring
		self.__queue.put( s ).unwrap( '_ThreadedStream.write: unbounded queue put always succeeds' )
		return Result.Ok( None )

	def _shutdown( self ) -> None:
		''' pushes the None sentinel, then waits with a BOUNDED timeout
		(~2s) for the writer thread to actually finish - Thread.join()
		itself has no timeout (WaitForSingleObject(...,INFINITE)/
		pthread_join, neither bounded), so this polls __writer_done
		instead of joining directly. Gives up WITHOUT joining if the
		writer hasn't finished in time - the process is about to exit
		either way (see __del__ below), so an unclaimed OS thread handle
		is harmless (Windows reclaims it) / the thread is torn down with
		the whole process regardless (POSIX). The alternative, an
		unbounded wait, risks hanging process exit forever on a stuck
		writer (a blocked console, a broken pipe) - exactly the scenario
		this exists to avoid, at the cost of a possible incomplete flush
		in that (hopefully rare) case. '''
		maybe_writer: threading.Thread|None = self.__writer
		if maybe_writer is None:
			panic( '_shutdown: called before start()' )
		writer: threading.Thread = maybe_writer
		self.__queue.put( None ).unwrap( '_shutdown: unbounded queue put always succeeds' )
		attempts: usize = 0
		while attempts < 200: # 200 * 10ms = up to ~2s
			if self.__writer_done.load():
				writer.join()
				return
			_blocking_sleep_10ms()
			with compiler.wrap_arithmetic:
				attempts += 1
		# gave up - see this method's own docstring

	@virtual
	def __del__( self ) -> None:
		''' shuts the writer thread down (see _shutdown()'s own docstring).
		Deliberately does NOT chain to the base class's own __del__ (no
		super().__del__() - that call shape isn't supported for __del__
		specifically, which gets special compiler-synthesized dispatch
		rather than ordinary virtual-method resolution) - harmless to skip
		here regardless, since write() is fully overridden above, so
		_buf/_len (all the base __del__ actually touches) never get
		touched on a _ThreadedStream at all; its cleanup would be a pure
		no-op even if it did run. Guarded so a second call is a no-op,
		same reasoning as _BufferedStream.__del__'s own comment - a second
		Thread.join() on an already-joined thread is undefined behavior on
		both platforms (a double CloseHandle on Windows, a reused/invalid
		thread id on POSIX), not just redundant work. '''
		if not self.__shut_down:
			self._shutdown()
			self.__shut_down = True

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

def enable_threaded_stdout() -> None:
	''' opt-in: replaces the global `stdout` with a _ThreadedStream, so
	every future print()/sys.stdout.write() call enqueues onto a
	background writer thread instead of writing synchronously - see
	_ThreadedStream's own docstring for the real motivation (coalescing
	many small writes into far fewer syscalls) and _BufferedStream's own
	docstring for why this is a whole-object swap (a module-level
	function reassigning the global) rather than a method that flips a
	flag on the existing object in place. Flushes whatever's already
	buffered synchronously in the OLD stream first, so nothing already
	written gets reordered after what's about to start flowing through
	the new queue. Not idempotency-guarded - calling this twice replaces
	an already-threaded stream with a second one (the first's own
	__del__ runs normally via the ordinary RC drop, shutting its writer
	thread down correctly) - wasteful if done by mistake, not unsafe. '''
	global stdout
	stdout.flush().is_ok()
	new_stream: _ThreadedStream = _ThreadedStream( stdout.fd() )
	new_stream.start()
	stdout = new_stream

def enable_threaded_stderr() -> None:
	''' see enable_threaded_stdout() - identical, for stderr. '''
	global stderr
	stderr.flush().is_ok()
	new_stream: _ThreadedStream = _ThreadedStream( stderr.fd() )
	new_stream.start()
	stderr = new_stream

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
