# this code is an attempt at converting the following project to metalpy
# https://github.com/johnBuffer/StableIndexVector/blob/main/index_vector.hpp
#
# NOTE: this is the ORIGINAL swap-and-pop algorithm, kept around under its
# own name because it's genuinely useful (O(1) erase, cache-friendly
# contiguous storage, stable IDs that survive other inserts/deletes) - but
# it does NOT preserve positional/insertion order on erase (confirmed by
# emitter_c_test.py's own test_erase_does_not_preserve_positional_order,
# and by the upstream README: "On deletion, the last element is swapped
# into the gap"). list[T] (lib/builtins/__list.py) is a separate, much
# simpler implementation for callers that need real array/Python-list
# semantics instead.

import compiler
import sys

# ---------------------------------------------------------------------------
# Metadata: per-slot reverse ID and validity tracking.
# Stored in a parallel buffer alongside the data buffer.
# ---------------------------------------------------------------------------

@cstruct
class _FastListMetadata:
	rid:         usize	# Reverse ID: maps from data-buffer index back to stable ID
	validity_id: usize	# Incremented on every erase; invalidates stale handles

FASTLIST_INVALID_ID: usize = usize.max

# ---------------------------------------------------------------------------
# RawFastList: the non-generic implementation core.
#
# Operates entirely on opaque Ptr[None] elements + element_size bytes.
# Compiled exactly once regardless of how many FastList[T] instantiations
# exist. All RC operations are the responsibility of the typed FastList[T]
# wrapper.
# ---------------------------------------------------------------------------

class RawFastList:
	__data:         Ptr[None]            # Contiguous element buffer (opaque bytes)
	__metadata:     Ptr[_FastListMetadata]   # Per-slot metadata (rid + validity_id)
	__indexes:      Ptr[usize]           # Stable ID -> data-buffer index
	__free_ids:     Ptr[usize]           # Stack of erased, reusable stable IDs
	__free_count:   usize                # Number of valid entries in __free_ids
	__next_id:      usize                # Next never-before-used stable ID
	__element_size: usize                # Byte width of a single element
	__len:          usize                # Active element count
	__cap:          usize                # Allocated capacity (in elements)

	def __init__(
		self,
		element_size:     usize,
		initial_capacity: usize,
	) -> None:
		self.__element_size = element_size
		self.__len          = 0
		self.__cap          = initial_capacity
		self.__free_count   = 0
		self.__next_id      = 0
		with compiler.panic_arithmetic( 'RawFastList init: capacity overflow' ):
			data_bytes: usize     = initial_capacity * element_size
			meta_bytes: usize     = initial_capacity * compiler.sizeof( _FastListMetadata )
			index_bytes: usize    = initial_capacity * compiler.sizeof( usize )
		self.__data     = compiler.cast( Ptr[None], sys.alloc[u8]( data_bytes ))
		self.__metadata = sys.alloc[_FastListMetadata]( meta_bytes )
		self.__indexes  = sys.alloc[usize]( index_bytes )
		self.__free_ids = sys.alloc[usize]( index_bytes )

	def __del__( self ) -> None:
		sys.free( self.__data )
		sys.free( self.__metadata )
		sys.free( self.__indexes )
		sys.free( self.__free_ids )

	def len( self ) -> usize:
		return self.__len

	def capacity( self ) -> usize:
		return self.__cap

	def is_valid_handle( self, id: usize, validity_id: usize ) -> bool:
		if id >= self.__len:
			return False
		data_idx: usize = self.__indexes[id]
		if data_idx >= self.__len:
			return False
		return self.__metadata[data_idx].validity_id == validity_id

	# Returns a raw (Ptr[None]) pointer to the element at data-buffer index idx.
	# Bounds-checked against __len (not __cap).
	def _ptr_at( self, idx: usize ) -> Result[Ptr[None], IndexError]:
		if idx >= self.__len:
			return Result.Err( IndexError() )
		with compiler.panic_arithmetic( 'RawFastList _ptr_at: offset overflow' ):
			slot_ptr: Ptr[None] = self.__data + idx * self.__element_size
		return Result.Ok( slot_ptr )

	# Returns a raw pointer to the element referenced by stable id.
	def _get( self, id: usize ) -> Result[Ptr[None], IndexError]:
		if id >= self.__cap:
			return Result.Err( IndexError() )
		data_idx: usize = self.__indexes[id]
		return self._ptr_at( data_idx )

	# Returns a raw pointer to the element at data-buffer index idx.
	def _get_at( self, idx: usize ) -> Result[Ptr[None], IndexError]:
		return self._ptr_at( idx )

	# Appends a pre-RC-increffed element (caller is responsible for RC).
	# Copies element_size bytes from val_ptr into the data buffer.
	# Returns the stable ID assigned to the new slot.
	def _append( self, val_ptr: Ptr[None] ) -> Result[usize, OverflowError]:
		if self.__len >= self.__cap:
			self._grow().or_return()

		# Assign a stable ID for the new slot
		id: usize = self._get_free_id()

		# Write element bytes into data buffer - _slot_ptr, not _ptr_at:
		# the new element goes into the NEXT free slot (data-buffer index
		# __len, one past the currently-counted bound but still within
		# __cap) - _ptr_at's own bounds check (idx < __len, valid EXISTING
		# elements only) would always reject exactly this slot
		slot_ptr: Ptr[None] = self._slot_ptr( self.__len )
		sys.memcpy( slot_ptr, val_ptr, self.__element_size )

		# Write metadata: this slot's reverse ID is id
		self.__metadata[self.__len] = _FastListMetadata(
			rid         = id,
			validity_id = self.__metadata[self.__len].validity_id,
		)

		# Update the stable ID -> data index mapping
		self.__indexes[id] = self.__len
		self.__len = self.__len + 1
		return Result.Ok( id )

	# O(1) swap-and-pop removal by stable ID.
	# Does NOT perform RC decref — caller is responsible.
	def _erase( self, id: usize ) -> Result[None, IndexError]:
		if id >= self.__cap:
			return Result.Err( IndexError() )
		data_idx: usize     = self.__indexes[id]
		if data_idx >= self.__len:
			return Result.Err( IndexError() )
		with compiler.panic_arithmetic( 'RawFastList _erase: underflow computing last_idx' ):
			last_idx: usize = self.__len - 1
		last_id: usize      = self.__metadata[last_idx].rid

		# Increment validity_id of the removed slot to invalidate existing handles
		with compiler.panic_arithmetic( 'RawFastList _erase: validity_id overflow' ):
			self.__metadata[data_idx].validity_id = self.__metadata[data_idx].validity_id + 1

		if data_idx != last_idx:
			# Swap element bytes
			dst_ptr: Ptr[None] = self._ptr_at( data_idx ).or_return()
			src_ptr: Ptr[None] = self._ptr_at( last_idx ).or_return()
			sys.memcpy( dst_ptr, src_ptr, self.__element_size )

			# Swap metadata
			tmp_meta: _FastListMetadata     = self.__metadata[data_idx]
			self.__metadata[data_idx]        = self.__metadata[last_idx]
			self.__metadata[last_idx]        = tmp_meta

			# Update the moved element's stable ID -> new data index
			self.__indexes[last_id] = data_idx

		with compiler.panic_arithmetic( 'RawFastList _erase: __len underflow' ):
			self.__len = self.__len - 1

		# recycle id - see _get_free_id's own comment: without this, the
		# next _append would reissue this exact id (since __len just
		# shrank) while it's still live at data_idx via the swap above,
		# silently aliasing two different elements onto the same id
		self.__free_ids[self.__free_count] = id
		with compiler.panic_arithmetic( 'RawFastList _erase: free_count overflow' ):
			self.__free_count = self.__free_count + 1
		return Result.Ok( None )

	# O(1) swap-and-pop removal by data-buffer index.
	# Does NOT perform RC decref — caller is responsible.
	def _erase_at( self, idx: usize ) -> Result[None, IndexError]:
		if idx >= self.__len:
			return Result.Err( IndexError() )
		id: usize = self.__metadata[idx].rid
		return self._erase( id )

	# Invalidates all existing handles without freeing or re-allocating buffers.
	def _clear( self ) -> None:
		i: usize = 0
		while i < self.__len:
			with compiler.panic_arithmetic( 'RawFastList _clear: overflow' ):
				self.__metadata[i].validity_id = self.__metadata[i].validity_id + 1
				id: usize = self.__metadata[i].rid
				self.__free_ids[self.__free_count] = id
				self.__free_count = self.__free_count + 1
				i += 1
		self.__len = 0

	# Returns a pointer to the raw bytes of the slot at data_idx (for caller to read before erase).
	def _slot_ptr( self, data_idx: usize ) -> Ptr[None]:
		with compiler.panic_arithmetic( 'RawFastList _slot_ptr: offset overflow' ):
			slot_ptr: Ptr[None] = self.__data + data_idx * self.__element_size
		return slot_ptr

	# Doubles buffer capacity.
	def _grow( self ) -> Result[None, OverflowError]:
		# any of the following 4 calculations can produce an OverflowError
		new_cap: usize = self.__cap * 2
		new_data_bytes:  usize = new_cap * self.__element_size
		new_meta_bytes:  usize = new_cap * compiler.sizeof( _FastListMetadata )
		new_index_bytes: usize = new_cap * compiler.sizeof( usize )

		new_data:     Ptr[None]              = compiler.cast( Ptr[None], sys.alloc[u8]( new_data_bytes ))
		errdefer( sys.free( new_data ))
		new_metadata: Ptr[_FastListMetadata] = sys.alloc[_FastListMetadata]( new_meta_bytes )
		errdefer( sys.free( new_metadata ))
		new_indexes:  Ptr[usize]             = sys.alloc[usize]( new_index_bytes )
		errdefer( sys.free( new_indexes ))
		new_free_ids: Ptr[usize]             = sys.alloc[usize]( new_index_bytes )
		errdefer( sys.free( new_free_ids ))

		# Copy existing data
		sys.memcpy( new_data,     self.__data,     self.__len * self.__element_size )
		sys.memcpy( new_metadata, self.__metadata, self.__len * compiler.sizeof( _FastListMetadata ) )
		sys.memcpy( new_indexes,  self.__indexes,  self.__cap * compiler.sizeof( usize ) )
		sys.memcpy( new_free_ids, self.__free_ids, self.__cap * compiler.sizeof( usize ) )

		sys.free( self.__data )
		sys.free( self.__metadata )
		sys.free( self.__indexes )
		sys.free( self.__free_ids )

		self.__data     = new_data
		self.__metadata = new_metadata
		self.__indexes  = new_indexes
		self.__free_ids = new_free_ids
		self.__cap      = new_cap
		return Result.Ok( None )

	# Returns the stable ID to use for the next slot: a previously-erased
	# id off the free-list if one exists, otherwise a genuinely never-
	# before-used id. NOTE: this used to just be `return self.__len`
	# ("simplicity" per the original port's own comment) - but __len
	# SHRINKS on erase, so that reissued ids that were still live via
	# _erase's own swap-and-keep-alive of the last element (see
	# emitter_c_test.py's test_append_after_erase_reissues_a_live_id,
	# written to pin this down before the fix)
	def _get_free_id( self ) -> usize:
		if self.__free_count > 0:
			with compiler.panic_arithmetic( 'RawFastList _get_free_id: underflow' ):
				self.__free_count = self.__free_count - 1
			return self.__free_ids[self.__free_count]
		id: usize = self.__next_id
		with compiler.panic_arithmetic( 'RawFastList _get_free_id: overflow' ):
			self.__next_id = self.__next_id + 1
		return id


# ---------------------------------------------------------------------------
# FastListHandle[T]: a stable, validatable reference to an element in a
# FastList[T].
#
# Does NOT own the element or the list. Validity is checked via
# validity_id comparison against the list's internal metadata.
# ---------------------------------------------------------------------------

@cstruct
class FastListHandle[T]:
	__id:          usize
	__validity_id: usize
	__raw:         Ptr[RawFastList]    # Non-owning raw pointer; list must outlive handle

	def is_valid( self ) -> bool:
		if self.__raw == None:
			return False
		return self.__raw[0].is_valid_handle( self.__id, self.__validity_id )

	# Returns a borrowed pointer to the element. Check is_valid() first.
	def get( self ) -> Result[Ptr[T], IndexError]:
		ptr: Ptr[None] = self.__raw[0]._get( self.__id ).or_return()
		return Result.Ok( compiler.cast( Ptr[T], ptr ))

	def id( self ) -> usize:
		return self.__id


# ---------------------------------------------------------------------------
# FastList[T]: thin type-safe wrapper over RawFastList.
#
# Monomorphized per T, but each method is a one-liner cast + RawFastList
# call. RC operations (incref/decref) are performed here so RawFastList
# stays generic.
#
# O(1) append/erase via swap-and-pop, stable IDs that survive other
# inserts/deletes, but does NOT preserve positional order on erase - see
# this file's own module docstring. Use list[T] instead if you need real
# array/Python-list ordering semantics.
# ---------------------------------------------------------------------------

class FastList[T]:
	__raw: RawFastList

	def __init__( self, initial_capacity: usize = 8 ) -> None:
		# an RC element's own SLOT in the buffer holds its handle (a
		# pointer, sizeof(usize)-wide - always pointer-width regardless of
		# T, same width usize itself is on every target this compiler
		# supports), never its struct body - compiler.sizeof(T) deliberately
		# stays the OBJECT's own real, layout-dependent size (needed by
		# sys.alloc[T]'s construction use, and would otherwise wildly
		# overallocate/misalign every element slot here). compiler.is_rc(T)
		# folds away at compile time (see type_resolver.py's own rewrite 4),
		# so only the branch that actually applies to THIS T ever compiles
		element_size: usize = 0
		if compiler.is_rc( T ):
			element_size = compiler.sizeof( usize )
		else:
			element_size = compiler.sizeof( T )
		self.__raw = RawFastList(
			element_size     = element_size,
			initial_capacity = initial_capacity,
		)

	# Read the T value stored at a raw slot address. An RC element's own
	# slot holds its HANDLE directly (see __init__'s own comment on why
	# element_size is pointer-width there) - reinterpreting the slot as
	# Ptr[T] and dereferencing it (the value-typed path below) would read
	# struct-BODY-sized memory out of a pointer-sized slot, since Ptr[T]
	# itself always stays single-indirection even for an RCClass T (needed
	# elsewhere for sys.alloc[T]'s own construction use - see PLAN_LIST_T.md).
	# compiler.is_rc(T) folds away entirely at compile time (only the
	# branch that actually applies to THIS T ever gets compiled - see
	# type_resolver.py's own rewrite 4), so this stays one shared,
	# readable method instead of scattering the distinction through every
	# accessor below.
	def _read_element( self, slot: Ptr[None] ) -> T:
		if compiler.is_rc( T ):
			handle_slot: Ptr[Ptr[None]] = compiler.cast( Ptr[Ptr[None]], slot )
			return compiler.cast( T, handle_slot[0] )
		else:
			ptr: Ptr[T] = compiler.cast( Ptr[T], slot )
			return ptr[0]

	def __del__( self ) -> None:
		# Decref all RC elements before RawFastList frees the buffer
		i: usize = 0
		while i < self.__raw.len():
			val: T = self._read_element( self.__raw._slot_ptr( i ))
			compiler.decref( val )
			with compiler.panic_arithmetic( 'FastList.__del__: overflow' ):
				i += 1
		# RawFastList.__del__ will free the raw buffers

	def __len__( self ) -> usize:
		return self.__raw.len()

	def capacity( self ) -> usize:
		return self.__raw.capacity()

	# Append a value. Increfs val if T is an RC type.
	# Returns the stable ID assigned to the element.
	def append( self, val: T ) -> Result[usize, OverflowError]:
		compiler.incref( val )
		id: usize = self.__raw._append( compiler.cast( Ptr[None], compiler.addrof( val ))).or_return()
		return Result.Ok( id )

	# Access element by stable ID. Returns a copy (with incref if RC).
	def __getitem__( self, id: usize ) -> Result[T, IndexError]:
		val: T = self._read_element( self.__raw._get( id ).or_return())
		compiler.incref( val )
		return Result.Ok( val )

	# Access element by contiguous data-buffer index (for fast iteration).
	def get_at( self, idx: usize ) -> Result[T, IndexError]:
		val: T = self._read_element( self.__raw._get_at( idx ).or_return())
		compiler.incref( val )
		return Result.Ok( val )

	# Get a borrowed pointer directly into the buffer (no copy, no incref).
	# Caller must NOT store this pointer beyond the next mutation of the list.
	# NOTE: for an RC element type, Ptr[T] itself stays single-indirection
	# (see _read_element's own comment) - a slot only ever holds a T
	# HANDLE, not a T value, so there is no correctly-typed Ptr[T] this
	# method could return today. Value-typed T only, for now.
	def get_ptr( self, id: usize ) -> Result[Ptr[T], IndexError]:
		ptr: Ptr[T] = compiler.cast( Ptr[T], self.__raw._get( id ).or_return())
		return Result.Ok( ptr )

	# Get a borrowed data-index pointer directly (for hot iteration loops).
	# NOTE: same RC-element limitation as get_ptr above.
	def get_ptr_at( self, idx: usize ) -> Result[Ptr[T], IndexError]:
		ptr: Ptr[T] = compiler.cast( Ptr[T], self.__raw._get_at( idx ).or_return())
		return Result.Ok( ptr )

	# Remove by stable ID. Decrefs the removed element if T is RC.
	def erase( self, id: usize ) -> Result[None, IndexError]:
		val: T = self._read_element( self.__raw._get( id ).or_return())
		compiler.decref( val )
		return self.__raw._erase( id )

	# Remove by data-buffer index. Decrefs the removed element if T is RC.
	def erase_at( self, idx: usize ) -> Result[None, IndexError]:
		val: T = self._read_element( self.__raw._get_at( idx ).or_return())
		compiler.decref( val )
		return self.__raw._erase_at( idx )

	# Erase all elements, decrefing each RC element first.
	def clear( self ) -> None:
		i: usize = 0
		while i < self.__raw.len():
			val: T = self._read_element( self.__raw._slot_ptr( i ))
			compiler.decref( val )
			with compiler.panic_arithmetic( 'FastList.clear: overflow' ):
				i += 1
		self.__raw._clear()

	# Create a stable handle to the element at stable ID.
	def create_handle( self, id: usize ) -> Result[FastListHandle[T], IndexError]:
		data_idx: usize = self.__raw._get( id ).or_return()  # validates bounds
		validity_id: usize = self.__raw.__metadata[self.__raw.__indexes[id]].validity_id
		h: FastListHandle[T] = FastListHandle.__allocate__(
			__id          = id,
			__validity_id = validity_id,
			__raw         = compiler.addrof( self.__raw ),
		)
		return Result.Ok( h )
