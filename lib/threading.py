# lib/threading.py — FastLock: a fast, non-timeout mutual-exclusion lock
#
# FastLock trades timeout/timed-wait functionality for the fastest available
# OS primitive on each platform:
#   Windows → SRWLOCK (AcquireSRWLockExclusive / TryAcquireSRWLockExclusive)
#   Linux   → pthread_mutex_t (pthread_mutex_lock / pthread_mutex_trylock)
#
# Per-platform opaque lock type — the field annotation Ptr[LockOpaque] gives
# the correct C pointer type at every call site.  On Linux the @extern
# prototypes are suppressed (header='pthread.h') so the C compiler sees the
# real signatures from <pthread.h> directly; if our types disagree, the C
# compiler catches it.

import compiler
import sys
import socket
import atomic

if compiler.target.os == 'windows':
	from windows.kernel32 import _SRWLOCK
	LockOpaque: TypeAlias = _SRWLOCK
	# a Win32 thread HANDLE is a void* (Ptr[None])
	ThreadHandle: TypeAlias = Ptr[None]
else:
	LockOpaque = compiler.c_type('pthread_mutex_t', header='pthread.h')
	# pthread_t is opaque and NOT a pointer on glibc (it's an unsigned long),
	# so it can't be modelled as Ptr[None]: pthread_join takes it BY VALUE and
	# pthread_create fills a pthread_t*, both of which a void* mis-types under
	# GCC/clang. Use the real C type (shared with lib/posix/pthread.py's own
	# extern signatures) so both the by-value pass and the out-pointer match.
	from posix.pthread import pthread_t
	ThreadHandle: TypeAlias = pthread_t


class FastLock:
	__lock: Ptr[LockOpaque]  # Ptr[_SRWLOCK] on Windows, Ptr[pthread_mutex_t] on Linux
	__locked: bool

	# ------------------------------------------------------------------
	# __init__ — allocate and initialise the inner OS lock
	# ------------------------------------------------------------------

	@compiler.target( os = 'windows' )
	def __init__( self ) -> None:
		self.__lock = sys.alloc[LockOpaque]( 1 )
		sys.memzero( self.__lock, compiler.sizeof( LockOpaque ))
		self.__locked = False

	@compiler.target( os = not 'windows' )
	def __init__( self ) -> None:
		from posix.pthread import pthread_mutex_init
		# sys.alloc_raw returns Ptr[u8] — pthread_mutex_init expects
		# pthread_mutex_t*, but void*/u8* implicitly converts there;
		# we zero the raw bytes before init for defense-in-depth
		mutex_size: usize = compiler.sizeof( LockOpaque )
		raw: Ptr[u8] = sys.alloc_raw( mutex_size )
		sys.memzero( raw, mutex_size )
		self.__lock = raw
		result: i32 = pthread_mutex_init( self.__lock, None )
		if result != 0:
			sys.panic( 'FastLock.__init__: pthread_mutex_init failed' )
		self.__locked = False

	# ------------------------------------------------------------------
	# __del__ — tear down the inner OS lock and free the buffer
	# ------------------------------------------------------------------

	@compiler.target( os = 'windows' )
	def __del__( self ) -> None:
		sys.free( self.__lock )

	@compiler.target( os = not 'windows' )
	def __del__( self ) -> None:
		from posix.pthread import pthread_mutex_destroy
		result: i32 = pthread_mutex_destroy( self.__lock )
		if result != 0:
			sys.panic( 'FastLock.__del__: pthread_mutex_destroy failed (lock still held?)' )
		sys.free( self.__lock )

	# ------------------------------------------------------------------
	# __enter__/__exit__ ( with lock )
	# ------------------------------------------------------------------
	
	def __enter__( self ) -> None:
		self.acquire()

	def __exit__( self ) -> None:
		self.release()

	# ------------------------------------------------------------------
	# acquire() -> None
	# ------------------------------------------------------------------

	@compiler.target( os = 'windows' )
	def acquire( self ) -> None:
		from windows.kernel32 import AcquireSRWLockExclusive
		AcquireSRWLockExclusive( self.__lock )
		self.__locked = True
	
	@compiler.target( os = not 'windows' )
	def acquire( self ) -> None:
		from posix.pthread import pthread_mutex_lock
		result: i32 = pthread_mutex_lock( self.__lock )
		if result != 0:
			sys.panic( f'pthread_mutex_lock() failed with error code {result!r}' )
		self.__locked = True

	# ------------------------------------------------------------------
	# try_acquire() -> bool
	# ------------------------------------------------------------------

	@compiler.target( os = 'windows' )
	def try_acquire( self ) -> bool:
		from windows.kernel32 import TryAcquireSRWLockExclusive
		if not TryAcquireSRWLockExclusive( self.__lock ):
			return False
		self.__locked = True
		return self.__locked
	
	@compiler.target( os = not 'windows' )
	def try_acquire( self ) -> bool:
		from posix.pthread import pthread_mutex_trylock
		if 0 != pthread_mutex_trylock( self.__lock ):
			return False
		self.__locked = True
		return self.__locked

	# ------------------------------------------------------------------
	# release() -> None
	# ------------------------------------------------------------------

	@compiler.target( os = 'windows' )
	def release( self ) -> None:
		from windows.kernel32 import ReleaseSRWLockExclusive
		self.__locked = False
		ReleaseSRWLockExclusive( self.__lock )

	@compiler.target( os = not 'windows' )
	def release( self ) -> None:
		from posix.pthread import pthread_mutex_unlock
		self.__locked = False
		pthread_mutex_unlock( self.__lock )

	# ------------------------------------------------------------------
	# locked -> bool
	# ------------------------------------------------------------------

	@property
	def locked( self ) -> bool:
		return self.__locked


# ---------------------------------------------------------------------------
# Thread — spawns a closure (a bound-method value, see PLAN_CALLABLE.md/the
# approved atomics-closures-threading plan) on its own OS thread.
#
# entry is an ORDINARY parameter, not move[T] - a closure is just another RC
# object, and sharing one closure across several Threads (spawn N threads off
# the same closure) is a real, legitimate use case move[T] would wrongly
# forbid. __init__ takes its own independent, increffed reference (the OS
# thread's own copy, released by _thread_entry once it's done calling
# through it) - ordinary borrowed-parameter-passing already means entry's
# OWN caller-side reference is untouched.
#
# Parameters, not just a bare `def worker() -> None`: entry's own captured
# receiver (worker.run's `worker`) IS the parameter-passing mechanism - see
# the plan's own reasoning for why a separate variadic thread-argument
# mechanism isn't needed. A result comes back the same way: write it into a
# field on that same receiver object, signal completion with a FastLock/
# AtomicBool, and read it after join() - Thread.run() always returns None,
# no generic JoinHandle[T] (ergonomics to revisit later, see the plan).
# ---------------------------------------------------------------------------

class Thread:
	__handle: ThreadHandle  # HANDLE (void*) on Windows, pthread_t on Linux

	@compiler.target( os = 'windows' )
	def __init__( self, entry: Closure[[], None] ) -> None:
		from windows.kernel32 import CreateThread
		compiler.incref( entry ) # a new, independent owner - the OS thread's own copy, released by _thread_entry
		arg: Ptr[None] = compiler.cast( Ptr[None], entry )
		self.__handle = CreateThread( None, 0, _thread_entry, arg, 0, None )

	@compiler.target( os = not 'windows' )
	def __init__( self, entry: Closure[[], None] ) -> None:
		from posix.pthread import pthread_create
		compiler.incref( entry )
		arg: Ptr[None] = compiler.cast( Ptr[None], entry )
		# pthread_create's own thread* out-param needs a real, standalone
		# pointer (compiler.addrof only accepts a bare local variable, not
		# self.field - see lib/atomic.py's identical Atomic[T] workaround) -
		# a one-slot pthread_t heap allocation, read back into self.__handle
		# (the pthread_t VALUE) and freed immediately. The slot is Ptr[pthread_t]
		# so the emitted arg is a real pthread_t*, matching pthread_create's
		# actual <pthread.h> prototype (a void** did not).
		slot: Ptr[ThreadHandle] = sys.alloc[ThreadHandle]( 1 )
		result: i32 = pthread_create( slot, None, _thread_entry, arg )
		if result != 0:
			sys.panic( 'Thread.__init__: pthread_create failed' )
		self.__handle = slot[0]
		sys.free( compiler.cast( Ptr[None], slot ) )

	@compiler.target( os = 'windows' )
	def join( self ) -> None:
		from windows.kernel32 import WaitForSingleObject, CloseHandle, INFINITE
		WaitForSingleObject( self.__handle, INFINITE )
		CloseHandle( self.__handle )

	@compiler.target( os = not 'windows' )
	def join( self ) -> None:
		from posix.pthread import pthread_join
		pthread_join( self.__handle, None )


@compiler.target( os = 'windows' )
def _thread_entry( arg: Ptr[None] ) -> u32:
	# takes ownership of the closure reference __init__'s own
	# compiler.incref gave it - the ordinary scope-exit decref for a fresh,
	# owned local (closure's own type is real again after this cast, no
	# type erasure left) is exactly the release this reference needs, no
	# explicit compiler.decref call required
	closure: Closure[[], None] = compiler.cast( Closure[[], None], arg )
	closure()
	return 0

@compiler.target( os = not 'windows' )
def _thread_entry( arg: Ptr[None] ) -> Ptr[None]:
	closure: Closure[[], None] = compiler.cast( Closure[[], None], arg )
	closure()
	return None


# ---------------------------------------------------------------------------
# ThreadLocal[T] — one T|None slot per OS thread, built on the real per-
# platform TLS primitive (Windows TlsAlloc/TlsGetValue/TlsSetValue/TlsFree,
# POSIX pthread_key_create/pthread_getspecific/pthread_setspecific/
# pthread_key_delete). Built for fiber.py's own "which fiber is running on
# THIS thread right now" ambient lookup (its module docstring has flagged
# this as needed since it was first written - a plain global is only
# correct for a single OS thread) and the reactor plan's own current_
# worker() - any "ambient lookup keyed by the calling thread" need.
#
# T is constrained to compiler.is_rc(T) (a real class or Closure, checked
# in __init__ - not a union/Optional; see get()'s own -> T|None return
# for how "no value set yet" is represented instead) - the underlying slot
# is always a raw, pointer-sized OS TLS value, exactly the same handle-
# only representation list[T]/dict[K,V] already use for an RC element (see
# lib/builtins/__list.py's own _read_element/_write_element comment) -
# reusing that established compiler.cast(T, raw)/compiler.cast(Ptr[None],
# value) idiom rather than inventing a new one.
#
# get()/set() do NOT own the slot's own stored reference - set(value) does
# NOT incref value before storing (the caller's own existing reference is
# what's stashed; the slot is a bookmark, not a second owner), matching
# fiber.py's pre-ThreadLocal plain-global convention exactly (`_current =
# self` there has never increfed either - the actual Fiber objects are
# owned by whichever Worker queue/pool holds them, `_current` is only ever
# a bookmark pointing at one of them). get() is DIFFERENT: it DOES incref
# before returning - not because the SLOT owns anything, but because EVERY
# call's result is unconditionally treated as a fresh, owned value by
# whoever's local it gets bound into (this compiler's own universal
# convention, not specific to RC-container peeks - list.__getitem__'s own
# "peek returns a genuinely new owned reference, the original stays valid"
# contract is the same shape). Confirmed via a real ASAN-caught heap-use-
# after-free that this ISN'T optional: `b = Box(...); tl.set(b); got =
# tl.get()` - without get()'s own incref, `b` and `got` both alias the
# same object, and BOTH get their own independent release_object() call in
# the caller's own epilogue - two decrefs for one incref. A ThreadLocal[T]
# that also owned a reference on SET (a second incref there, matching
# set()'s own field-assignment convention elsewhere in this codebase)
# would need real destructor-driven per-thread cleanup (pthread_key_
# create's own destructor callback / a Windows DllMain-style TLS callback)
# to avoid leaking whatever's left in the slot when a thread exits without
# clearing it first - a real, separate problem this class does not attempt
# to solve; callers that need cleanup on thread-exit must arrange it
# themselves (e.g. explicit clear() at the end of the thread's own entry
# closure).
# ---------------------------------------------------------------------------

if compiler.target.os == 'windows':
	ThreadLocalKey: TypeAlias = u32   # a TLS index, per TlsAlloc()
else:
	from posix.pthread import pthread_key_t
	ThreadLocalKey: TypeAlias = pthread_key_t

class ThreadLocal[T]:
	__key: ThreadLocalKey

	@compiler.target( os = 'windows' )
	def __init__( self ) -> None:
		if not compiler.is_rc( T ):
			sys.panic( 'ThreadLocal[T]: T must be an RC type (a class or Closure) - see this class\'s own module comment' )
		from windows.kernel32 import TlsAlloc, TLS_OUT_OF_INDEXES
		self.__key = TlsAlloc()
		if self.__key == TLS_OUT_OF_INDEXES:
			sys.panic( 'ThreadLocal.__init__: TlsAlloc failed' )

	@compiler.target( os = not 'windows' )
	def __init__( self ) -> None:
		if not compiler.is_rc( T ):
			sys.panic( 'ThreadLocal[T]: T must be an RC type (a class or Closure) - see this class\'s own module comment' )
		from posix.pthread import pthread_key_create
		# pthread_key_create's own key* out-param needs a real, standalone
		# pointer (compiler.addrof only accepts a bare local variable, not
		# self.field - same one-slot-heap-alloc workaround Thread.__init__
		# already uses for pthread_create's own thread* out-param above).
		# destructor=None: no automatic per-thread cleanup - see this
		# class's own module comment for why
		key_slot: Ptr[ThreadLocalKey] = sys.alloc[ThreadLocalKey]( 1 )
		result: i32 = pthread_key_create( key_slot, None )
		if result != 0:
			sys.panic( 'ThreadLocal.__init__: pthread_key_create failed' )
		self.__key = key_slot[0]
		sys.free( compiler.cast( Ptr[None], key_slot ))

	@compiler.target( os = 'windows' )
	def __del__( self ) -> None:
		from windows.kernel32 import TlsFree
		TlsFree( self.__key )

	@compiler.target( os = not 'windows' )
	def __del__( self ) -> None:
		from posix.pthread import pthread_key_delete
		pthread_key_delete( self.__key )

	@compiler.target( os = 'windows' )
	def get( self ) -> T|None:
		from windows.kernel32 import TlsGetValue
		raw: Ptr[None] = TlsGetValue( self.__key )
		if raw is None:
			return None
		# an intermediate T-typed local, not a bare `return compiler.cast(T,
		# raw)` - the return statement's own expected type (T|None) otherwise
		# leaks into how compiler.cast's first argument gets resolved,
		# casting to the WIDER union instead of the bare T actually written
		# (confirmed via a real compile failure: emitted C tried casting a
		# raw pointer directly to the union STRUCT type, not a pointer)
		result: T = compiler.cast( T, raw )
		# a NEW owned reference, not just an alias of whatever set() stored -
		# ANY local bound from a Call's result (here, this whole get() call,
		# from the CALLER's own point of view) is unconditionally treated as
		# fresh/owned and gets its own phantom epilogue Decref - without this
		# incref, that decref releases the SAME object the TLS slot's own
		# stored value still points at, with nothing having incremented it to
		# balance that release. Confirmed via a real ASAN-caught heap-use-
		# after-free: `b = Box(42); tl.set(b); got = tl.get()` - both `b` and
		# `got` alias the same Box, and both get their own independent
		# release_object() call in the caller's own epilogue. Same idiom
		# list.__getitem__ already uses for exactly this "peek returns a
		# genuinely new owned reference, the original stays valid" contract.
		compiler.incref( result )
		return result

	@compiler.target( os = not 'windows' )
	def get( self ) -> T|None:
		from posix.pthread import pthread_getspecific
		raw: Ptr[None] = pthread_getspecific( self.__key )
		if raw is None:
			return None
		result: T = compiler.cast( T, raw )
		compiler.incref( result )
		return result

	@compiler.target( os = 'windows' )
	def set( self, value: T ) -> None:
		from windows.kernel32 import TlsSetValue
		raw: Ptr[None] = compiler.cast( Ptr[None], value )
		if not TlsSetValue( self.__key, raw ):
			sys.panic( 'ThreadLocal.set: TlsSetValue failed' )

	@compiler.target( os = not 'windows' )
	def set( self, value: T ) -> None:
		from posix.pthread import pthread_setspecific
		raw: Ptr[None] = compiler.cast( Ptr[None], value )
		result: i32 = pthread_setspecific( self.__key, raw )
		if result != 0:
			sys.panic( 'ThreadLocal.set: pthread_setspecific failed' )

	@compiler.target( os = 'windows' )
	def clear( self ) -> None:
		from windows.kernel32 import TlsSetValue
		null: Ptr[None] = None
		if not TlsSetValue( self.__key, null ):
			sys.panic( 'ThreadLocal.clear: TlsSetValue failed' )

	@compiler.target( os = not 'windows' )
	def clear( self ) -> None:
		from posix.pthread import pthread_setspecific
		null: Ptr[None] = None
		result: i32 = pthread_setspecific( self.__key, null )
		if result != 0:
			sys.panic( 'ThreadLocal.clear: pthread_setspecific failed' )


# ---------------------------------------------------------------------------
# ThreadPool — a fixed-size pool of daemon worker threads + a bounded,
# round-robin submit() queue, for CPU-bound/blocking work a caller wants
# bounded concurrency for without spawning an unbounded thread per job.
#
# Generalizes the private pool lib/asyncfile.py has used internally since
# its own thread-pool-backed file I/O offload (_Pool/_PoolWorker/_Job
# there) — same fixed-worker-array + round-robin submit() shape, but NOT
# reactor-integrated: a job here is a plain Closure[[], None] side effect,
# not a Result[usize,OSError]-wrapped payload a reactor.Worker waits on via
# a CompletionHandle. asyncfile.py's own pool stays exactly as it is
# (deliberately reactor-coupled) — this is a new, independent, public
# primitive for any caller that just wants bounded thread concurrency. A
# caller that needs a job's result back writes it into a field on its own
# captured receiver and reads it after its own synchronization, the same
# convention Thread itself already documents.
# ---------------------------------------------------------------------------

class QueueFullError: pass


class _PoolJob:
	work: Closure[[], None]

	def __init__( self, work: Closure[[], None] ) -> None:
		self.work = work


class _PoolWorker:
	__jobs:          list[_PoolJob]
	__max_depth:     usize|None
	__wake_read:     socket.Socket
	__wake_write:    socket.Socket
	__shutting_down: atomic.Atomic[bool]

	def __init__( self, max_depth: usize|None ) -> None:
		self.__jobs = list[_PoolJob]()
		self.__max_depth = max_depth
		( read_side, write_side ) = socket.make_loopback_pair()
		self.__wake_read = read_side
		self.__wake_write = write_side
		self.__shutting_down = atomic.Atomic[bool]( False )

	def submit( self, job: _PoolJob ) -> Result[None, QueueFullError]:
		if self.__max_depth is not None:
			limit: usize = self.__max_depth
			if self.__jobs.__len__() >= limit:
				return Result.Err( QueueFullError() )
		self.__jobs.append( job ).unwrap( '_PoolWorker.submit: queue overflow' )
		poke: bytes = b'x'
		self.__wake_write.send( poke.get_const_ptr(), usize( 1 )).unwrap( '_PoolWorker.submit: wake failed' )
		return Result.Ok( None )

	def request_shutdown( self ) -> None:
		self.__shutting_down.store( True )
		poke: bytes = b'x'
		self.__wake_write.send( poke.get_const_ptr(), usize( 1 )).unwrap( '_PoolWorker.request_shutdown: wake failed' )

	def run_forever( self ) -> None:
		''' blocks (a genuine, thread-blocking recv - no Poller, this thread
		has nothing else to do while idle) until a poke arrives, then drains
		and runs every job currently queued - same "at least one byte means
		check the queue" discipline asyncfile._PoolWorker/reactor.Worker's
		own wake-drain already use. Exits once shutdown has been requested
		AND the queue is fully drained - a job submitted right before
		shutdown still runs to completion, matching reactor.Reactor.
		shutdown()'s own "does not reject work already queued" contract. '''
		buf: bytearray = bytearray( usize( 64 ))
		while True:
			self.__wake_read.recv( buf.get_ptr(), usize( 64 )).unwrap( '_PoolWorker.run_forever: wake recv failed' )
			while True:
				match self.__jobs.pop():
					case Result.Ok( job ):
						work: Closure[[], None] = job.work
						work()
					case Result.Err( _ ):
						break
			if self.__shutting_down.load() and self.__jobs.__len__() == 0:
				return


def default_pool_size() -> usize:
	''' Python's own concurrent.futures.ThreadPoolExecutor default heuristic:
	min(32, cpu_count() + 4) - a floor for low-core machines, a hard cap
	against runaway thread counts on many-core ones. NOT used as ThreadPool.
	__init__'s own default - that stays required, matching reactor.Reactor.
	__init__(num_workers)'s own convention. A caller that wants a
	reasonable size without picking one itself can call this explicitly:
	threading.ThreadPool( threading.default_pool_size() ). '''
	with compiler.wrap_arithmetic:
		n: usize = usize( sys.cpu_count() ) + usize( 4 )
	if n > usize( 32 ):
		return usize( 32 )
	return n


class ThreadPool:
	__workers: list[_PoolWorker]
	__threads: list[Thread]
	__next:    atomic.Atomic[usize]

	def __init__( self, size: usize, max_queue_depth: usize|None = None ) -> None:
		''' spawns `size` daemon worker threads immediately (Thread.
		__init__ starts them - there is no separate .run() to call, unlike
		reactor.Reactor). Required, no default - matches reactor.Reactor.
		__init__(num_workers)'s own convention exactly (see threading.
		default_pool_size() for a caller that wants a reasonable size
		without picking one itself). max_queue_depth is enforced PER WORKER
		(each of the `size` workers independently caps its own queue at
		this depth), not pool-wide - fits the existing round-robin
		architecture with no new cross-worker synchronization; a real
		tradeoff is that submit() can reject under skewed load even while a
		different worker sits idle. None (default) preserves the original
		unbounded behavior exactly. '''
		self.__workers = list[_PoolWorker]()
		self.__threads = list[Thread]()
		self.__next = atomic.Atomic[usize]( 0 )
		i: usize = 0
		while i < size:
			w: _PoolWorker = _PoolWorker( max_queue_depth )
			self.__workers.append( w ).unwrap( 'ThreadPool.__init__: worker list overflow' )
			t: Thread = Thread( w.run_forever )
			self.__threads.append( t ).unwrap( 'ThreadPool.__init__: thread list overflow' )
			with compiler.wrap_arithmetic:
				i = i + 1

	def submit( self, work: Closure[[], None] ) -> Result[None, QueueFullError]:
		''' round-robin across workers, via an atomic fetch_add rather than
		a plain load-then-store on a bare usize field: two threads calling
		submit() concurrently used to be able to both read the SAME index
		before either wrote back, silently skipping a worker (a lost
		round-robin step - not memory-unsafe, but a real fairness bug, and
		the one genuine data race left over in this class). fetch_add is a
		single atomic RMW, so every caller gets a distinct, monotonically
		increasing ticket with no lost updates - same effective (idx+1) %
		len idiom as reactor.Reactor.spawn(), just computed from an
		ever-growing ticket instead of a stored-and-wrapped index. '''
		ticket: usize = self.__next.fetch_add( usize( 1 ))
		with compiler.panic_arithmetic( 'ThreadPool.submit: pool size is zero' ):
			idx: usize = ticket % self.__workers.__len__()
		w: _PoolWorker = self.__workers.__getitem__( idx ).unwrap( 'ThreadPool.submit: index in bounds by construction' )
		return w.submit( _PoolJob( work ))

	def shutdown( self, wait: bool = True ) -> None:
		''' requests every worker to stop once its queue drains - callable
		from any thread, mirrors reactor.Reactor.shutdown()'s own "request,
		don't reject already-queued work" contract. wait=True (default)
		additionally joins every worker thread before returning; wait=False
		returns immediately and workers finish asynchronously. '''
		n: usize = self.__workers.__len__()
		i: usize = 0
		while i < n:
			w: _PoolWorker = self.__workers.__getitem__( i ).unwrap( 'ThreadPool.shutdown: worker index in bounds by construction' )
			w.request_shutdown()
			with compiler.wrap_arithmetic:
				i = i + 1
		if not wait:
			return
		n_threads: usize = self.__threads.__len__()
		i = 0
		while i < n_threads:
			joining: Thread = self.__threads.__getitem__( i ).unwrap( 'ThreadPool.shutdown: thread index in bounds by construction' )
			joining.join()
			with compiler.wrap_arithmetic:
				i = i + 1
