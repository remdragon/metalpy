# lib/atomic.py — Atomic[T]: lock-free access to a single scalar-or-raw-
# pointer value shared across threads, built on compiler.atomic_*(Ptr[T], ...).
#
# T must be a plain scalar (bool, i8-i64, u8-u64, usize) or a raw Ptr[U]/
# ConstPtr[U] - the underlying compiler.atomic_* intrinsics enforce this
# themselves (see lowering.py's _atomic_pointee_type), so instantiating
# Atomic[SomeRCClass] fails to compile with a clear error rather than
# silently doing the wrong thing. Atomically swapping an RC pointer without
# incref/decref bookkeeping is a real, separate problem (hazard pointers /
# epoch reclamation) - out of scope here, deliberately. A raw Ptr[U]/
# ConstPtr[U] carries no refcount, so that concern doesn't apply to it.
#
# Stores __ptr typed as a real Ptr[T] (not type-erased to Ptr[None] like
# list[T]'s own buffer) since every method here needs T back immediately
# anyway - only __del__ casts to Ptr[None] to satisfy sys.free's own
# signature (Ptr[u8]/Ptr[None] only, see lib/sys.py), mirroring how
# list[T]/dict[K,V] already cast at their own free() call sites.

import compiler
import sys

class Atomic[T]:
	__ptr: Ptr[T]

	def __init__( self, initial: T ) -> None:
		self.__ptr = sys.alloc[T]( 1 )
		self.__ptr[0] = initial

	def __del__( self ) -> None:
		sys.free( compiler.cast( Ptr[None], self.__ptr ))

	def load( self ) -> T:
		return compiler.atomic_load( self.__ptr )

	def store( self, value: T ) -> None:
		compiler.atomic_store( self.__ptr, value )

	def fetch_add( self, value: T ) -> T:
		return compiler.atomic_add( self.__ptr, value )

	def fetch_sub( self, value: T ) -> T:
		return compiler.atomic_sub( self.__ptr, value )

	def exchange( self, value: T ) -> T:
		return compiler.atomic_exchange( self.__ptr, value )

	def compare_exchange( self, expected: Ptr[T], desired: T ) -> bool:
		# matches compiler.atomic_compare_exchange's own C11-shaped
		# signature directly (expected is an out-param the caller already
		# owns, updated with the actual current value on failure) rather
		# than inventing a Result-based API on top of it - keep this
		# wrapper thin, revisit ergonomics if a real caller wants more
		return compiler.atomic_compare_exchange( self.__ptr, expected, desired )
