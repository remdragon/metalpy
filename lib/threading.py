# lib/threading.py — FastLock: a fast, non-timeout mutual-exclusion lock
#
# FastLock trades timeout/timed-wait functionality for the fastest available
# OS primitive on each platform:
#   Windows → SRWLOCK (AcquireSRWLockExclusive / TryAcquireSRWLockExclusive)
#   Linux   → pthread_mutex_t (pthread_mutex_lock / pthread_mutex_trylock)

import compiler
import sys

class LockError:
	pass

class FastLock:
	__lock: Ptr[u8]    # heap-allocated platform-specific lock bytes
	__locked: bool      # tracks current lock state for locked() queries

	# ------------------------------------------------------------------
	# __init__ — allocate and initialise the inner OS lock
	# ------------------------------------------------------------------

	@compiler.target( os = 'windows' )
	def __init__( self ) -> None:
		# SRWLOCK is sizeof(PVOID) = 8 bytes on 64-bit; zero-init
		self.__lock = sys.alloc[u8]( 8 )
		sys.memzero( self.__lock, 8 )
		self.__locked = False

	@compiler.target( os = not 'windows' )
	def __init__( self ) -> None:
		from posix.pthread import pthread_mutex_init
		# sizeof(pthread_mutex_t) varies by platform (40 bytes glibc,
		# 48 bytes musl); 64 bytes is generous and safe for all common
		# 64-bit targets.
		self.__lock = sys.alloc[u8]( 64 )
		sys.memzero( self.__lock, 64 )
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
		pthread_mutex_destroy( self.__lock )
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
