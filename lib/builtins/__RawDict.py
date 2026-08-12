# RawDict: the non-generic implementation core behind dict[K,V] (see
# lib/builtins/__init__.py's own dict class, PLAN_CALLABLE.md). Compiled
# exactly once regardless of how many dict[K,V] instantiations exist -
# genuinely type-erased, not just "doesn't branch on RC-ness": RawDict
# never decodes a key_ptr/value_ptr back to a real K/V at all, never
# allocates/frees/increfs/decrefs one, never computes a hash. The ONE
# place it needs type-specific behavior (comparing two keys for equality,
# deep inside its own hash-collision scan) is a real function pointer
# (key_eq_fn: Ptr[Callable[...]]) supplied by the caller - a monomorphized
# @staticmethod on dict[K,V] itself, which does know K. Every other
# operation (hashing, ownership, storage, release) happens entirely in
# dict[K,V]'s own methods, where K/V are already concrete - no erasure
# boundary to cross there at all.
#
# __entries/__indices are a sorted-by-hash binary-searchable index
# (__indices, RawIndex{hash, entry_idx}) over an append-only entry log
# (__entries, RawEntry{hash, key_ptr, value_ptr}) - the same split the
# original port of this file used, just without the fictional K.__eq_fn__/
# reinterpret_cast it reached for before Callable[...] existed.

import compiler

KeyEqFn: TypeAlias = Ptr[Callable[[Ptr[None],Ptr[None]],bool]]

@cstruct
class RawEntry:
	hash: u64
	key_ptr: Ptr[None]   # OWNED by dict[K,V] - a heap copy (value K) or an increfed handle (RC K)
	value_ptr: Ptr[None] # OWNED by dict[K,V] - same convention, for V

@cstruct
class RawIndex:
	hash: u64
	entry_idx: usize

class RawDict:
	# UnsafeList[T], not list[T]: RawDict's own storage is private and never
	# escapes - it has nothing to do with cross-thread sharing, and
	# shouldn't silently pay list[T]'s own lock-acquire cost on every dict
	# operation just because it happens to be built on "a list" (see
	# __list.py's own header comment on this split)
	__entries: UnsafeList[RawEntry]
	__indices: UnsafeList[RawIndex]

	def __init__( self ) -> None:
		self.__entries = UnsafeList[RawEntry]()
		self.__indices = UnsafeList[RawIndex]()

	def __len__( self ) -> usize:
		return len( self.__entries )

	# first position in __indices whose hash is >= target (lower_bound) -
	# plain manual binary search, not bisect.bisect_left: bisect.py's own
	# key= parameter needs Callable/lambda support this compiler doesn't
	# have yet (see PLAN_CALLABLE.md's own "deferred" list), and RawIndex's
	# .hash field is known statically here anyway - no need for a key
	# extractor at all
	def _lower_bound( self, target_hash: u64 ) -> usize:
		lo: usize = 0
		hi: usize = len( self.__indices )
		with compiler.panic_arithmetic( 'RawDict _lower_bound: overflow' ):
			while lo < hi:
				mid: usize = ( lo + hi ) // 2
				mid_hash: u64 = self.__indices.__getitem__( mid ).unwrap( 'RawDict: index out of bounds' ).hash
				if mid_hash < target_hash:
					lo = mid + 1
				else:
					hi = mid
		return lo

	# the __entries index of the live entry matching (hash, key_ptr) via
	# key_eq_fn, scanning every __indices position with the same hash
	# (collisions) starting from the lower bound - Err(KeyError()) if none
	# match
	def _find_entry_idx( self, target_hash: u64, key_ptr: Ptr[None], key_eq_fn: KeyEqFn ) -> Result[usize, KeyError]:
		pos: usize = self._lower_bound( target_hash )
		n: usize = len( self.__indices )
		with compiler.panic_arithmetic( 'RawDict _find_entry_idx: overflow' ):
			while pos < n:
				idx_node: RawIndex = self.__indices.__getitem__( pos ).unwrap( 'RawDict: index out of bounds' )
				if idx_node.hash != target_hash:
					break
				entry: RawEntry = self.__entries.__getitem__( idx_node.entry_idx ).unwrap( 'RawDict: entry out of bounds' )
				if key_eq_fn( entry.key_ptr, key_ptr ):
					return Result.Ok( idx_node.entry_idx )
				pos += 1
		return Result.Err( KeyError() )

	def key_ptr_at( self, entry_idx: usize ) -> Ptr[None]:
		return self.__entries.__getitem__( entry_idx ).unwrap( 'RawDict: entry out of bounds' ).key_ptr

	def value_ptr_at( self, entry_idx: usize ) -> Ptr[None]:
		return self.__entries.__getitem__( entry_idx ).unwrap( 'RawDict: entry out of bounds' ).value_ptr

	# overwrites an EXISTING entry's value_ptr, returning the OLD one -
	# caller (dict[K,V], which knows whether V is RC) is responsible for
	# releasing it
	def overwrite_value_at( self, entry_idx: usize, value_ptr: Ptr[None] ) -> Ptr[None]:
		old_entry: RawEntry = self.__entries.__getitem__( entry_idx ).unwrap( 'RawDict: entry out of bounds' )
		new_entry: RawEntry = RawEntry( hash = old_entry.hash, key_ptr = old_entry.key_ptr, value_ptr = value_ptr )
		self.__entries.__setitem__( entry_idx, new_entry ).unwrap( 'RawDict: entry out of bounds' )
		return old_entry.value_ptr

	# appends a brand-new entry and threads it into the sorted __indices
	# array - key_ptr/value_ptr must already be OWNED (see this file's own
	# module docstring); RawDict never allocates/increfs anything itself
	def insert_new( self, hash: u64, key_ptr: Ptr[None], value_ptr: Ptr[None] ) -> None:
		pos: usize = self._lower_bound( hash )
		entry_idx: usize = len( self.__entries )
		self.__entries.append( RawEntry( hash = hash, key_ptr = key_ptr, value_ptr = value_ptr )).unwrap( 'RawDict: append overflow' )
		self.__indices.insert( pos, RawIndex( hash = hash, entry_idx = entry_idx )).unwrap( 'RawDict: insert overflow' )
