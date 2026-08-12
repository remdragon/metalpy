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

if compiler.target.os == 'windows':
	from windows.kernel32 import _SRWLOCK
	LockOpaque: TypeAlias = _SRWLOCK
else:
	LockOpaque = compiler.c_type('pthread_mutex_t', header='pthread.h')


class LockError:
	pass

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
		# sys._alloc returns Ptr[u8] — pthread_mutex_init expects
		# pthread_mutex_t*, but void*/u8* implicitly converts there;
		# we zero the raw bytes before init for defense-in-depth
		mutex_size: usize = compiler.sizeof( LockOpaque )
		raw: Ptr[u8] = sys._alloc( mutex_size )
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
	# acquire( blocking: bool = True ) -> Result[None, LockError]
	# ------------------------------------------------------------------

	@compiler.target( os = 'windows' )
	def acquire( self, blocking: bool = True ) -> Result[None, LockError]:
		from windows.kernel32 import AcquireSRWLockExclusive, TryAcquireSRWLockExclusive
		if blocking:
			AcquireSRWLockExclusive( self.__lock )
			self.__locked = True
			return Result.Ok( None )
		else:
			if TryAcquireSRWLockExclusive( self.__lock ):
				self.__locked = True
				return Result.Ok( None )
			return Result.Err( LockError() )

	@compiler.target( os = not 'windows' )
	def acquire( self, blocking: bool = True ) -> Result[None, LockError]:
		from posix.pthread import pthread_mutex_lock, pthread_mutex_trylock
		if blocking:
			result: i32 = pthread_mutex_lock( self.__lock )
			if result != 0:
				return Result.Err( LockError() )
			self.__locked = True
			return Result.Ok( None )
		else:
			result: i32 = pthread_mutex_trylock( self.__lock )
			if result != 0:
				return Result.Err( LockError() )
			self.__locked = True
			return Result.Ok( None )

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
	# locked() -> bool
	# ------------------------------------------------------------------

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
	__handle: Ptr[None]  # HANDLE on Windows, pthread_t on Linux

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
		# a one-slot heap allocation, read back into self.__handle and
		# freed immediately, keeps self.__handle itself uniformly Ptr[None]
		# (the handle VALUE, not a pointer to it) on both platforms
		slot: Ptr[Ptr[None]] = sys.alloc[Ptr[None]]( 1 )
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
