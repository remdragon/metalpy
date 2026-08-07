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
