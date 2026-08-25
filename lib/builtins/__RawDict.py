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
# (__indices, RawIndex{hash, entry_idx}) over a compacted entry log
# (__entries, RawEntry{hash, key_ptr, value_ptr}) - the same split the
# original port of this file used, just without the fictional K.__eq_fn__/
# reinterpret_cast it reached for before Callable[...] existed. __entries
# is no longer append-only (see remove_entry) - removal keeps it fully
# compacted (memmove-based shift, never a swap-and-pop hole), so every
# index in [0, len) still names a live entry; only the entry_idx values
# stored in __indices need fixing up after a removal (see
# _fixup_indices_after_removal).

import bisect
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

def _raw_index_hash( node: RawIndex ) -> u64:
	return node.hash

class RawDict( Sized ):
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
	# bisect_left_by_key operates directly on __indices (an UnsafeList
	# [RawIndex], no view/copy needed) keyed on .hash. Used to be a plain
	# manual binary search (see git history) written before Callable[...]
	# existed to make bisect.py's own key= usable at all - now the first
	# real caller of bisect.py anywhere in this codebase.
	def _lower_bound( self, target_hash: u64 ) -> usize:
		key: Ptr[Callable[[RawIndex],u64]] = _raw_index_hash
		return bisect.bisect_left_by_key( self.__indices, target_hash, key )

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

	# same scan shape as _find_entry_idx (binary-search to the lower bound,
	# then linear-scan same-hash collisions), but returns the match's own
	# POSITION WITHIN __indices instead of its entry_idx - remove_entry
	# (below) needs that position too, to erase_at() the right __indices
	# slot. _find_entry_idx itself is left untouched (its own callers,
	# __getitem__/__setitem__, never need this) rather than widening its
	# signature for a caller that didn't exist when it was written.
	def _find_indices_pos( self, target_hash: u64, key_ptr: Ptr[None], key_eq_fn: KeyEqFn ) -> Result[usize, KeyError]:
		pos: usize = self._lower_bound( target_hash )
		n: usize = len( self.__indices )
		with compiler.panic_arithmetic( 'RawDict _find_indices_pos: overflow' ):
			while pos < n:
				idx_node: RawIndex = self.__indices.__getitem__( pos ).unwrap( 'RawDict: index out of bounds' )
				if idx_node.hash != target_hash:
					break
				entry: RawEntry = self.__entries.__getitem__( idx_node.entry_idx ).unwrap( 'RawDict: entry out of bounds' )
				if key_eq_fn( entry.key_ptr, key_ptr ):
					return Result.Ok( pos )
				pos += 1
		return Result.Err( KeyError() )

	# decrements entry_idx on every __indices node that pointed PAST the
	# just-removed __entries slot - erasing that slot (a memmove-based
	# shift-left) moved every LATER entry down by one, so every OTHER
	# RawIndex whose entry_idx was greater than the removed one is now
	# stale by exactly one. O(n): removal position within the sorted-by-
	# hash __indices array has nothing to do with entry_idx order, so the
	# stale nodes are scattered arbitrarily through it - a full walk is
	# the only way to find them all. Same complexity class insert_new
	# already pays (its own O(n) shift inside UnsafeList.insert).
	def _fixup_indices_after_removal( self, removed_entry_idx: usize ) -> None:
		i: usize = 0
		n: usize = len( self.__indices )
		with compiler.panic_arithmetic( 'RawDict _fixup_indices_after_removal: overflow' ):
			while i < n:
				node: RawIndex = self.__indices.__getitem__( i ).unwrap( 'RawDict: index out of bounds' )
				if node.entry_idx > removed_entry_idx:
					fixed: RawIndex = RawIndex( hash = node.hash, entry_idx = node.entry_idx - 1 )
					self.__indices.__setitem__( i, fixed ).unwrap( 'RawDict: index out of bounds' )
				i += 1

	# removes the entry matching (hash, key_ptr) via key_eq_fn and returns a
	# COPY of the removed RawEntry - Err(KeyError()) if none match. RawDict
	# owns nothing K/V-shaped (see this file's own module docstring), so the
	# returned entry's key_ptr/value_ptr are handed back OWNED to the
	# caller (dict[K,V], which knows whether K/V are RC) to release; nobody
	# else can, since erase_at on a non-RC RawEntry element just drops the
	# slot without decreffing/freeing anything - there is no other path to
	# these two pointers once this call returns.
	def remove_entry( self, target_hash: u64, key_ptr: Ptr[None], key_eq_fn: KeyEqFn ) -> Result[RawEntry, KeyError]:
		indices_pos: usize = self._find_indices_pos( target_hash, key_ptr, key_eq_fn ).or_return()
		removed_index: RawIndex = self.__indices.__getitem__( indices_pos ).unwrap( 'RawDict: index out of bounds' )
		removed_entry: RawEntry = self.__entries.__getitem__( removed_index.entry_idx ).unwrap( 'RawDict: entry out of bounds' )
		self.__indices.erase_at( indices_pos ).unwrap( 'RawDict remove_entry: indices erase index out of bounds' )
		self.__entries.erase_at( removed_index.entry_idx ).unwrap( 'RawDict remove_entry: entries erase index out of bounds' )
		self._fixup_indices_after_removal( removed_index.entry_idx )
		return Result.Ok( removed_entry )

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
		self.__entries.append( RawEntry( hash = hash, key_ptr = key_ptr, value_ptr = value_ptr ))
		self.__indices.insert( pos, RawIndex( hash = hash, entry_idx = entry_idx ))
