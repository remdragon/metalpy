# list[T]: a plain, contiguous, order-preserving growable array - real
# Python-list semantics (positional indices, insert/remove shift the
# elements around them via memmove). See lib/builtins/__fastlist.py for a
# different container (FastList[T]) that trades order preservation for O(1)
# erase and stable IDs that survive other inserts/deletes - that one is a
# straight port of https://github.com/johnBuffer/StableIndexVector, worth
# keeping for callers that actually want its trade-offs, but it is NOT a
# drop-in replacement for a real array (confirmed by emitter_c_test.py's
# test_erase_does_not_preserve_positional_order).

import compiler
import sys

# ---------------------------------------------------------------------------
# RawList: the non-generic implementation core.
#
# Operates entirely on opaque Ptr[None] elements + element_size bytes.
# Compiled exactly once regardless of how many list[T] instantiations exist.
# All RC operations are the responsibility of the typed list[T] wrapper.
#
# A single contiguous buffer, no separate stable-ID indirection layer -
# an index IS a position. insert/remove shift the elements after the
# target position via sys.memmove (overlap-safe, unlike memcpy).
# ---------------------------------------------------------------------------

class RawList:
	__data:         Ptr[None]  # Contiguous element buffer (opaque bytes)
	__element_size: usize      # Byte width of a single element
	__len:          usize      # Active element count
	__cap:          usize      # Allocated capacity (in elements)

	def __init__(
		self,
		element_size:     usize,
		initial_capacity: usize,
	) -> None:
		self.__element_size = element_size
		self.__len          = 0
		self.__cap          = initial_capacity
		with compiler.panic_arithmetic( 'RawList init: capacity overflow' ):
			data_bytes: usize = initial_capacity * element_size
		self.__data = compiler.cast( Ptr[None], sys.alloc[u8]( data_bytes ))

	def __del__( self ) -> None:
		sys.free( self.__data )

	def len( self ) -> usize:
		return self.__len

	def capacity( self ) -> usize:
		return self.__cap

	# Returns a pointer to the raw bytes of the slot at idx. NOT bounds-
	# checked against __len - used internally for the one-past-the-end
	# append slot and for iteration loops that already track their own
	# bound.
	def _slot_ptr( self, idx: usize ) -> Ptr[None]:
		with compiler.panic_arithmetic( 'RawList _slot_ptr: offset overflow' ):
			slot_ptr: Ptr[None] = self.__data + idx * self.__element_size
		return slot_ptr

	# Returns a raw pointer to the element at idx. Bounds-checked against __len.
	def _ptr_at( self, idx: usize ) -> Result[Ptr[None], IndexError]:
		if idx >= self.__len:
			return Result.Err( IndexError() )
		return Result.Ok( self._slot_ptr( idx ))

	# Doubles buffer capacity.
	def _grow( self ) -> Result[None, OverflowError]:
		new_cap: usize = self.__cap * 2
		new_data_bytes: usize = new_cap * self.__element_size

		new_data: Ptr[None] = compiler.cast( Ptr[None], sys.alloc[u8]( new_data_bytes ))
		errdefer( sys.free( new_data ))
		sys.memcpy( new_data, self.__data, self.__len * self.__element_size )
		sys.free( self.__data )

		self.__data = new_data
		self.__cap  = new_cap
		return Result.Ok( None )

	# Appends a pre-RC-increffed element (caller is responsible for RC).
	def _append( self, val_ptr: Ptr[None] ) -> Result[None, OverflowError]:
		if self.__len >= self.__cap:
			self._grow().or_return()
		sys.memcpy( self._slot_ptr( self.__len ), val_ptr, self.__element_size )
		with compiler.panic_arithmetic( 'RawList _append: overflow' ):
			self.__len = self.__len + 1
		return Result.Ok( None )

	# Inserts a pre-RC-increffed element at idx, shifting [idx, len) right
	# by one slot via memmove (memcpy is NOT safe here - the source and
	# destination ranges overlap). idx > len clamps to len (append),
	# matching Python's own list.insert - there's no failure mode here
	# distinct from OverflowError worth a separate error type for.
	def _insert_at( self, idx: usize, val_ptr: Ptr[None] ) -> Result[None, OverflowError]:
		clamped_idx: usize = idx
		if idx > self.__len:
			clamped_idx = self.__len
		if self.__len >= self.__cap:
			self._grow().or_return()
		with compiler.panic_arithmetic( 'RawList _insert_at: shift overflow' ):
			shift_count: usize = self.__len - clamped_idx
		if shift_count > 0:
			with compiler.panic_arithmetic( 'RawList _insert_at: shift overflow' ):
				shift_bytes: usize = shift_count * self.__element_size
				dest_idx:    usize = clamped_idx + 1
			sys.memmove( self._slot_ptr( dest_idx ), self._slot_ptr( clamped_idx ), shift_bytes )
		sys.memcpy( self._slot_ptr( clamped_idx ), val_ptr, self.__element_size )
		with compiler.panic_arithmetic( 'RawList _insert_at: overflow' ):
			self.__len = self.__len + 1
		return Result.Ok( None )

	# Removes the element at idx, shifting [idx+1, len) left by one slot
	# via memmove. Does NOT perform RC decref — caller is responsible.
	def _remove_at( self, idx: usize ) -> Result[None, IndexError]:
		if idx >= self.__len:
			return Result.Err( IndexError() )
		with compiler.panic_arithmetic( 'RawList _remove_at: shift overflow' ):
			tail_start:  usize = idx + 1
			shift_count: usize = self.__len - tail_start
		if shift_count > 0:
			with compiler.panic_arithmetic( 'RawList _remove_at: shift overflow' ):
				shift_bytes: usize = shift_count * self.__element_size
			sys.memmove( self._slot_ptr( idx ), self._slot_ptr( tail_start ), shift_bytes )
		with compiler.panic_arithmetic( 'RawList _remove_at: underflow' ):
			self.__len = self.__len - 1
		return Result.Ok( None )

	# Invalidates all elements without freeing or re-allocating the buffer.
	def _clear( self ) -> None:
		self.__len = 0


# ---------------------------------------------------------------------------
# list[T]: thin type-safe wrapper over RawList.
#
# Monomorphized per T, but each method is a one-liner cast + RawList call.
# RC operations (incref/decref) are performed here so RawList stays generic.
# ---------------------------------------------------------------------------

class list[T]:
	__raw: RawList

	def __init__( self, initial_capacity: usize = 8 ) -> None:
		# an RC element's own SLOT in the buffer holds its handle (a
		# pointer, sizeof(usize)-wide - always pointer-width regardless of
		# T), never its struct body - compiler.sizeof(T) deliberately stays
		# the OBJECT's own real, layout-dependent size (needed by
		# sys.alloc[T]'s construction use, and would otherwise wildly
		# overallocate/misalign every element slot here). compiler.is_rc(T)
		# folds away at compile time (see type_resolver.py's own rewrite 4),
		# so only the branch that actually applies to THIS T ever compiles
		element_size: usize = 0
		if compiler.is_rc( T ):
			element_size = compiler.sizeof( usize )
		else:
			element_size = compiler.sizeof( T )
		self.__raw = RawList(
			element_size     = element_size,
			initial_capacity = initial_capacity,
		)

	# Read the T value stored at a raw slot address. An RC element's own
	# slot holds its HANDLE directly (see __init__'s own comment on why
	# element_size is pointer-width there) - reinterpreting the slot as
	# Ptr[T] and dereferencing it (the value-typed path below) would read
	# struct-BODY-sized memory out of a pointer-sized slot, since Ptr[T]
	# itself always stays single-indirection even for an RCClass T.
	# compiler.is_rc(T) folds away entirely at compile time, so this stays
	# one shared, readable method instead of scattering the distinction
	# through every accessor below.
	def _read_element( self, slot: Ptr[None] ) -> T:
		if compiler.is_rc( T ):
			handle_slot: Ptr[Ptr[None]] = compiler.cast( Ptr[Ptr[None]], slot )
			return compiler.cast( T, handle_slot[0] )
		else:
			ptr: Ptr[T] = compiler.cast( Ptr[T], slot )
			return ptr[0]

	# The write-side mirror of _read_element - RC objects are only ever a
	# pointer wide, so writing one through is just overwriting the handle,
	# not the object it points at.
	def _write_element( self, slot: Ptr[None], val: T ) -> None:
		if compiler.is_rc( T ):
			handle_slot: Ptr[Ptr[None]] = compiler.cast( Ptr[Ptr[None]], slot )
			handle_slot[0] = compiler.cast( Ptr[None], val )
		else:
			ptr: Ptr[T] = compiler.cast( Ptr[T], slot )
			ptr[0] = val

	def __del__( self ) -> None:
		# Decref all RC elements before RawList frees the buffer
		i: usize = 0
		while i < self.__raw.len():
			val: T = self._read_element( self.__raw._slot_ptr( i ))
			compiler.decref( val )
			with compiler.panic_arithmetic( 'list.__del__: overflow' ):
				i += 1
		# RawList.__del__ will free the raw buffer

	def __len__( self ) -> usize:
		return self.__raw.len()

	def capacity( self ) -> usize:
		return self.__raw.capacity()

	# Append a value at the end. Increfs val if T is an RC type.
	def append( self, val: T ) -> Result[None, OverflowError]:
		compiler.incref( val )
		self.__raw._append( compiler.cast( Ptr[None], compiler.addrof( val ))).or_return()
		return Result.Ok( None )

	# Insert a value at idx, shifting everything at/after idx one slot to
	# the right. idx > len clamps to len (append), matching Python's own
	# list.insert. Increfs val if T is an RC type.
	def insert( self, idx: usize, val: T ) -> Result[None, OverflowError]:
		compiler.incref( val )
		self.__raw._insert_at( idx, compiler.cast( Ptr[None], compiler.addrof( val ))).or_return()
		return Result.Ok( None )

	# Access element by position. Returns a copy (with incref if RC).
	def __getitem__( self, idx: usize ) -> Result[T, IndexError]:
		val: T = self._read_element( self.__raw._ptr_at( idx ).or_return())
		compiler.incref( val )
		return Result.Ok( val )

	# Overwrite the element at idx. Increfs val and decrefs the value it replaces.
	def __setitem__( self, idx: usize, val: T ) -> Result[None, IndexError]:
		slot: Ptr[None] = self.__raw._ptr_at( idx ).or_return()
		old: T = self._read_element( slot )
		compiler.incref( val )
		self._write_element( slot, val )
		compiler.decref( old )
		return Result.Ok( None )

	# Get a borrowed pointer directly into the buffer (no copy, no incref).
	# Caller must NOT store this pointer beyond the next mutation of the
	# list (insert/remove/append may reallocate or shift the buffer).
	# NOTE: for an RC element type, Ptr[T] itself stays single-indirection
	# (see _read_element's own comment) - a slot only ever holds a T
	# HANDLE, not a T value, so there is no correctly-typed Ptr[T] this
	# method could return today. Value-typed T only, for now.
	def get_ptr( self, idx: usize ) -> Result[Ptr[T], IndexError]:
		ptr: Ptr[T] = compiler.cast( Ptr[T], self.__raw._ptr_at( idx ).or_return())
		return Result.Ok( ptr )

	# Remove the element at idx, shifting everything after it one slot to
	# the left. Decrefs the removed element if T is RC.
	def erase_at( self, idx: usize ) -> Result[None, IndexError]:
		val: T = self._read_element( self.__raw._ptr_at( idx ).or_return())
		compiler.decref( val )
		return self.__raw._remove_at( idx )

	# Erase all elements, decrefing each RC element first.
	def clear( self ) -> None:
		i: usize = 0
		while i < self.__raw.len():
			val: T = self._read_element( self.__raw._slot_ptr( i ))
			compiler.decref( val )
			with compiler.panic_arithmetic( 'list.clear: overflow' ):
				i += 1
		self.__raw._clear()
