# this code is an attempt at converting the following project to metalpy
# https://github.com/johnBuffer/StableIndexVector/blob/main/index_vector.hpp

import compiler
import sys

# ---------------------------------------------------------------------------
# Metadata: per-slot reverse ID and validity tracking.
# Stored in a parallel buffer alongside the data buffer.
# ---------------------------------------------------------------------------

@cstruct
class _ListMetadata:
	rid:         usize	# Reverse ID: maps from data-buffer index back to stable ID
	validity_id: usize	# Incremented on every erase; invalidates stale handles

INVALID_ID: usize = usize.max

# ---------------------------------------------------------------------------
# RawList: the non-generic implementation core.
#
# Operates entirely on opaque Ptr[None] elements + element_size bytes.
# Compiled exactly once regardless of how many list[T] instantiations exist.
# All RC operations are the responsibility of the typed list[T] wrapper.
# ---------------------------------------------------------------------------

class RawList:
	__data:         Ptr[None]            # Contiguous element buffer (opaque bytes)
	__metadata:     Ptr[_ListMetadata]   # Per-slot metadata (rid + validity_id)
	__indexes:      Ptr[usize]           # Stable ID -> data-buffer index
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
		with compiler.panic_arithmetic( 'RawList init: capacity overflow' ):
			data_bytes: usize     = initial_capacity * element_size
			meta_bytes: usize     = initial_capacity * compiler.sizeof( _ListMetadata )
			index_bytes: usize    = initial_capacity * compiler.sizeof( usize )
		self.__data     = sys.alloc[None]( data_bytes )
		self.__metadata = sys.alloc[_ListMetadata]( meta_bytes )
		self.__indexes  = sys.alloc[usize]( index_bytes )
	
	def __del__( self ) -> None:
		sys.free( self.__data )
		sys.free( self.__metadata )
		sys.free( self.__indexes )
	
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
			return Result.Err( IndexError )
		with compiler.panic_arithmetic( 'RawList _ptr_at: offset overflow' ):
			slot_ptr: Ptr[None] = self.__data + idx * self.__element_size
		return Result.Ok( slot_ptr )
	
	# Returns a raw pointer to the element referenced by stable id.
	def _get( self, id: usize ) -> Result[Ptr[None], IndexError]:
		if id >= self.__cap:
			return Result.Err( IndexError )
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
		
		# Write element bytes into data buffer
		slot_ptr: Ptr[None] = self._ptr_at( self.__len )
		sys.memcpy( slot_ptr, val_ptr, self.__element_size )
		
		# Write metadata: this slot's reverse ID is id
		self.__metadata[self.__len] = _ListMetadata(
			rid         = id,
			validity_id = self.__metadata[self.__len].validity_id,
		)
		
		# Update the stable ID -> data index mapping
		self.__indexes[id] = self.__len
		self.__len += 1
		return Result.Ok( id )
	
	# O(1) swap-and-pop removal by stable ID.
	# Does NOT perform RC decref — caller is responsible.
	def _erase( self, id: usize ) -> Result[None, IndexError]:
		if id >= self.__cap:
			return Result.Err( IndexError )
		data_idx: usize     = self.__indexes[id]
		if data_idx >= self.__len:
			return Result.Err( IndexError )
		with compiler.panic_arithmetic:
			last_idx: usize = self.__len - 1
		last_id: usize      = self.__metadata[last_idx].rid
		
		# Increment validity_id of the removed slot to invalidate existing handles
		self.__metadata[data_idx].validity_id += 1
		
		if data_idx != last_idx:
			# Swap element bytes
			dst_ptr: Ptr[None] = self._ptr_at( data_idx ).or_return()
			src_ptr: Ptr[None] = self._ptr_at( last_idx ).or_return()
			sys.memcpy( dst_ptr, src_ptr, self.__element_size )
			
			# Swap metadata
			tmp_meta: _ListMetadata         = self.__metadata[data_idx]
			self.__metadata[data_idx]        = self.__metadata[last_idx]
			self.__metadata[last_idx]        = tmp_meta
			
			# Update the moved element's stable ID -> new data index
			self.__indexes[last_id] = data_idx
	
		self.__len -= 1
		return Result.Ok( None )
	
	# O(1) swap-and-pop removal by data-buffer index.
	# Does NOT perform RC decref — caller is responsible.
	def _erase_at( self, idx: usize ) -> Result[None, IndexError]:
		if idx >= self.__len:
			return Result.Err( IndexError )
		id: usize = self.__metadata[idx].rid
		return self._erase( id )
	
	# Invalidates all existing handles without freeing or re-allocating buffers.
	def _clear( self ) -> None:
		i: usize = 0
		while i < self.__len:
			self.__metadata[i].validity_id += 1
			i += 1
		self.__len = 0
	
	# Returns a pointer to the raw bytes of the slot at data_idx (for caller to read before erase).
	def _slot_ptr( self, data_idx: usize ) -> Ptr[None]:
		return self.__data.add( data_idx * self.__element_size )
	
	# Doubles buffer capacity.
	def _grow( self ) -> Result[None, OverflowError]:
		# any of the following 4 calculations can produce an OverflowError
		new_cap: usize = self.__cap * 2
		new_data_bytes:  usize = new_cap * self.__element_size
		new_meta_bytes:  usize = new_cap * compiler.sizeof( _ListMetadata )
		new_index_bytes: usize = new_cap * compiler.sizeof( usize )
		
		new_data:     Ptr[None]          = sys.alloc[None]( new_data_bytes )
		errdefer( sys.free( new_data ))
		new_metadata: Ptr[_ListMetadata] = sys.alloc[_ListMetadata]( new_meta_bytes )
		errdefer( sys.free( new_metadata ))
		new_indexes:  Ptr[usize]         = sys.alloc[usize]( new_index_bytes )
		errdefer( sys.free( new_indexes ))
		
		# Copy existing data
		sys.memcpy( new_data,     self.__data,     self.__len * self.__element_size )
		sys.memcpy( new_metadata, self.__metadata, self.__len * compiler.sizeof( _ListMetadata ) )
		sys.memcpy( new_indexes,  self.__indexes,  self.__cap * compiler.sizeof( usize ) )
		
		sys.free( self.__data )
		sys.free( self.__metadata )
		sys.free( self.__indexes )
		
		self.__data     = new_data
		self.__metadata = new_metadata
		self.__indexes  = new_indexes
		self.__cap      = new_cap
		return Result.Ok( None )
	
	# Returns the stable ID to use for the next slot.
	# If metadata exists beyond __len (a previously erased slot), reuse it.
	# Otherwise create a fresh ID == __len.
	def _get_free_id( self ) -> usize:
		# The metadata array may be longer than __len if slots were previously erased
		# (metadata is retained to support validity_id tracking on stale handles).
		# In that case, reuse the pre-existing metadata slot's rid.
		#
		# NOTE: m_metadata.size() > m_data.size() is the C++ equivalent check.
		# Here we track this implicitly: __indexes[__len] was set during erase to the
		# recycled ID if one exists. We scan for a free slot by checking __indexes.
		# For simplicity in this initial implementation, ID == __len always.
		# A free-list can be added later for recycling erased slot IDs without
		# growing the __indexes buffer.
		return self.__len


# ---------------------------------------------------------------------------
# Handle[T]: a stable, validatable reference to an element in a list[T].
#
# Does NOT own the element or the list. Validity is checked via
# validity_id comparison against the list's internal metadata.
# ---------------------------------------------------------------------------

@cstruct
class Handle[T]:
	__id:          usize
	__validity_id: usize
	__raw:         Ptr[RawList]    # Non-owning raw pointer; list must outlive handle
	
	def is_valid( self ) -> bool:
		if self.__raw == None:
			return False
		return self.__raw[0].is_valid_handle( self.__id, self.__validity_id )
	
	# Returns a borrowed pointer to the element. Check is_valid() first.
	def get( self ) -> Result[Ptr[T], IndexError]:
		ptr: Ptr[None] = self.__raw[0]._get( self.__id ).or_return()
		return Result.Ok( ptr.cast[T]() )
	
	def id( self ) -> usize:
		return self.__id


# ---------------------------------------------------------------------------
# list[T]: thin type-safe wrapper over RawList.
#
# Monomorphized per T, but each method is a one-liner cast + RawList call.
# RC operations (incref/decref) are performed here so RawList stays generic.
# ---------------------------------------------------------------------------

class list[T]:
	__raw: RawList
	
	def __init__( self, initial_capacity: usize = 8 ) -> None:
		self.__raw = RawList(
			element_size     = compiler.sizeof( T ),
			initial_capacity = initial_capacity,
		)
	
	def __del__( self ) -> None:
		# Decref all RC elements before RawList frees the buffer
		i: usize = 0
		while i < self.__raw.len():
			ptr: Ptr[T] = self.__raw._slot_ptr( i ).cast[T]()
			compiler.decref( ptr[0] )
			i += 1
		# RawList.__del__ will free the raw buffers
	
	def __len__( self ) -> usize:
		return self.__raw.len()
	
	def capacity( self ) -> usize:
		return self.__raw.capacity()
	
	# Append a value. Increfs val if T is an RC type.
	# Returns the stable ID assigned to the element.
	def append( self, val: T ) -> Result[usize, OverflowError]:
		compiler.incref( val )
		id: usize = self.__raw._append( compiler.addrof( val ).cast[None]() ).or_return()
		return Result.Ok( id )
	
	# Access element by stable ID. Returns a copy (with incref if RC).
	def __getitem__( self, id: usize ) -> Result[T, IndexError]:
		ptr: Ptr[T] = self.__raw._get( id ).or_return().cast[T]()
		val: T = ptr[0]
		compiler.incref( val )
		return Result.Ok( val )
	
	# Access element by contiguous data-buffer index (for fast iteration).
	def get_at( self, idx: usize ) -> Result[T, IndexError]:
		ptr: Ptr[T] = self.__raw._get_at( idx ).or_return().cast[T]()
		val: T = ptr[0]
		compiler.incref( val )
		return Result.Ok( val )
	
	# Get a borrowed pointer directly into the buffer (no copy, no incref).
	# Caller must NOT store this pointer beyond the next mutation of the list.
	def get_ptr( self, id: usize ) -> Result[Ptr[T], IndexError]:
		ptr: Ptr[T] = self.__raw._get( id ).or_return().cast[T]()
		return Result.Ok( ptr )
	
	# Get a borrowed data-index pointer directly (for hot iteration loops).
	def get_ptr_at( self, idx: usize ) -> Result[Ptr[T], IndexError]:
		ptr: Ptr[T] = self.__raw._get_at( idx ).or_return().cast[T]()
		return Result.Ok( ptr )
	
	# Remove by stable ID. Decrefs the removed element if T is RC.
	def erase( self, id: usize ) -> Result[None, IndexError]:
		ptr: Ptr[T] = self.__raw._get( id ).or_return().cast[T]()
		compiler.decref( ptr[0] )
		return self.__raw._erase( id )
	
	# Remove by data-buffer index. Decrefs the removed element if T is RC.
	def erase_at( self, idx: usize ) -> Result[None, IndexError]:
		ptr: Ptr[T] = self.__raw._get_at( idx ).or_return().cast[T]()
		compiler.decref( ptr[0] )
		return self.__raw._erase_at( idx )
	
	# Erase all elements, decrefing each RC element first.
	def clear( self ) -> None:
		i: usize = 0
		while i < self.__raw.len():
			ptr: Ptr[T] = self.__raw._slot_ptr( i ).cast[T]()
			compiler.decref( ptr[0] )
			i += 1
		self.__raw._clear()
	
	# Create a stable handle to the element at stable ID.
	def create_handle( self, id: usize ) -> Result[Handle[T], IndexError]:
		data_idx: usize = self.__raw._get( id ).or_return()  # validates bounds
		validity_id: usize = self.__raw.__metadata[self.__raw.__indexes[id]].validity_id
		h: Handle[T] = Handle.__allocate__(
			__id          = id,
			__validity_id = validity_id,
			__raw         = compiler.addrof( self.__raw ),
		)
		return Result.Ok( h )
