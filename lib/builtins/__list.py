# list[T]: a plain, contiguous, order-preserving growable array - real
# Python-list semantics (positional indices, insert/remove shift the
# elements around them via memmove). See lib/builtins/__fastlist.py for a
# different container (FastList[T]) that trades order preservation for O(1)
# erase and stable IDs that survive other inserts/deletes - that one is a
# straight port of https://github.com/johnBuffer/StableIndexVector, worth
# keeping for callers that actually want its trade-offs, but it is NOT a
# drop-in replacement for a real array (confirmed by emitter_c_test.py's
# test_erase_does_not_preserve_positional_order).
#
# Three related types now, each a genuinely different tradeoff, not three
# names for the same thing:
#   list[T]        - THIS is what the name means by default: every method
#                     locked (a real FastLock, acquired/released around
#                     each call). list[T] is the container every program
#                     reaches for out of habit, so it defaults to safe
#                     rather than fast - an unsafe-by-default container
#                     that looks ordinary is a worse failure mode than a
#                     slightly slower safe one. RawList itself has no
#                     synchronization of its own (see its own comment
#                     below) - two threads racing _grow() is a real
#                     double-free, not a hypothetical.
#   UnsafeList[T]   - the SAME positional/order-preserving semantics as
#                     list[T], with no locking at all - the escape hatch
#                     for code that's confined to one thread and wants to
#                     skip the lock-acquire cost. Also what list[T] itself
#                     is built on, and what any OTHER internal, never-
#                     escaping, hot-path buffer in this stdlib should use
#                     directly (RawDict's own storage, int.divmod()'s own
#                     scratch multiples cache) - those have nothing to do
#                     with cross-thread sharing and shouldn't silently pay
#                     for it just because they're built on "a list".
#   FastList[T]     - unrelated to the safe/unsafe axis above: genuinely
#                     different erase/ordering semantics (swap-and-pop,
#                     stable IDs). Untouched by this split.
#
# A raw borrowed view (slice[T]) into the buffer is UnsafeList[T].as_slice()
# only, deliberately not the same shape on list[T] - the view's own validity
# window ("don't use it past the next mutation") is meaningless once the
# lock that made "the next mutation" observable has already been released.
# list[T] instead offers borrow_slice()/release_borrow(): an atomic borrow
# COUNT (see borrow_slice()'s own comment) that append/insert/erase_at/pop
# check and refuse to proceed against while nonzero - the same "an
# outstanding export blocks a resize" contract Python's own memoryview/
# buffer protocol enforces over bytearray, just via a plain counter instead
# of PEP 3118's own export-count machinery.

import compiler
import sys
import threading

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
	def _grow( self ) -> None:
		with compiler.panic_arithmetic( f'impossibly large RawList size requested' ):
			new_cap: usize = self.__cap * 2
			new_data_bytes: usize = new_cap * self.__element_size

			new_data: Ptr[None] = compiler.cast( Ptr[None], sys.alloc[u8]( new_data_bytes ))
			sys.memcpy( new_data, self.__data, self.__len * self.__element_size )
		sys.free( self.__data )

		self.__data = new_data
		self.__cap  = new_cap

	# Appends a pre-RC-increffed element (caller is responsible for RC).
	def _append( self, val_ptr: Ptr[None] ) -> None:
		if self.__len >= self.__cap:
			self._grow()
		sys.memcpy( self._slot_ptr( self.__len ), val_ptr, self.__element_size )
		with compiler.panic_arithmetic( 'RawList _append: overflow' ):
			self.__len = self.__len + 1

	# Inserts a pre-RC-increffed element at idx, shifting [idx, len) right
	# by one slot via memmove (memcpy is NOT safe here - the source and
	# destination ranges overlap). idx > len clamps to len (append),
	# matching Python's own list.insert - there's no failure mode here
	# distinct from OverflowError worth a separate error type for.
	def _insert_at( self, idx: usize, val_ptr: Ptr[None] ) -> None:
		clamped_idx: usize = idx
		if idx > self.__len:
			clamped_idx = self.__len
		if self.__len >= self.__cap:
			self._grow()
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
# UnsafeList[T]: thin type-safe wrapper over RawList - no locking of its
# own (see this file's own header comment for the three-way split with
# list[T]/FastList[T]).
#
# Monomorphized per T, but each method is a one-liner cast + RawList call.
# RC operations (incref/decref) are performed here so RawList stays generic.
# ---------------------------------------------------------------------------

class UnsafeList[T]:
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
	def append( self, val: T ) -> None:
		compiler.incref( val )
		self.__raw._append( compiler.cast( Ptr[None], compiler.addrof( val )))

	# Insert a value at idx, shifting everything at/after idx one slot to
	# the right. idx > len clamps to len (append), matching Python's own
	# list.insert. Increfs val if T is an RC type.
	def insert( self, idx: usize, val: T ) -> None:
		compiler.incref( val )
		self.__raw._insert_at( idx, compiler.cast( Ptr[None], compiler.addrof( val )))

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

	# A borrowed slice[T] view over the WHOLE buffer - "don't outlive the
	# next mutation" borrow contract (insert/remove/append may reallocate or
	# shift the buffer): slice[T]'s own _ptr is untyped (ConstPtr[None]),
	# and slice.get_unchecked/_element_size already do the same
	# compiler.is_rc(T) handle-vs-value branch UnsafeList's own
	# _read_element does. _slot_ptr(0), not _ptr_at(0) - the latter is
	# bounds-checked against __len and would fail on an empty list; a
	# zero-length slice is still well-formed (nothing can dereference
	# through it, since every real read goes through an index < len()
	# check first).
	def as_slice( self ) -> slice[T]:
		return slice[T]( _ptr = compiler.cast( ConstPtr[None], self.__raw._slot_ptr( 0 )), __len = self.__raw.len() )

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


# ---------------------------------------------------------------------------
# list[T]: the safe default - every method locked around a plain
# UnsafeList[T] (see this file's own header comment). __inner/__lock are
# both ordinary RC fields, so the compiler's own synthesized destructor
# already cascades into both of theirs correctly (decref every element,
# free the buffer, free the OS lock) - no __del__ needed here at all.
#
# Every wrapper method follows the same shape: acquire, defer the release
# (so it still runs on every exit path, not just the ordinary one - a
# panic or an early return inside the delegated call must never leave the
# lock held), delegate to __inner, return whatever it returned.
# ---------------------------------------------------------------------------

class list[T]:
	__inner:   UnsafeList[T]
	__lock:    threading.FastLock
	__borrows: usize  # see borrow_slice()'s own comment

	def __init__( self, initial_capacity: usize = 8 ) -> None:
		self.__inner   = UnsafeList[T]( initial_capacity )
		self.__lock    = threading.FastLock()
		self.__borrows = 0

	def __len__( self ) -> usize:
		with self.__lock:
			return self.__inner.__len__()

	def capacity( self ) -> usize:
		with self.__lock:
			return self.__inner.capacity()

	# Append a value at the end. Increfs val if T is an RC type.
	def append( self, val: T ) -> Result[None, BorrowError]:
		with self.__lock:
			if compiler.atomic_load( compiler.addrof( self.__borrows )) != 0:
				return Result.Err( BorrowError() )
			return self.__inner.append( val )

	# Insert a value at idx, shifting everything at/after idx one slot to
	# the right. idx > len clamps to len (append), matching Python's own
	# list.insert. Increfs val if T is an RC type.
	def insert( self, idx: usize, val: T ) -> Result[None, BorrowError]:
		with self.__lock:
			if compiler.atomic_load( compiler.addrof( self.__borrows )) != 0:
				return Result.Err( BorrowError() )
			return self.__inner.insert( idx, val )

	# Access element by position. Returns a copy (with incref if RC).
	def __getitem__( self, idx: usize ) -> Result[T, IndexError]:
		with self.__lock:
			return self.__inner.__getitem__( idx )

	# Overwrite the element at idx. Increfs val and decrefs the value it
	# replaces. NOT gated on __borrows: unlike append/insert/erase_at/pop,
	# this never reallocates or shifts anything - it's a fixed-offset write,
	# which can't invalidate a borrowed slice[T]'s own pointer or length
	# (whether the WRITE itself races logically with a concurrent reader is
	# a separate, pre-existing category of hazard borrow_slice() was never
	# meant to solve either - see its own comment).
	def __setitem__( self, idx: usize, val: T ) -> Result[None, IndexError]:
		with self.__lock:
			return self.__inner.__setitem__( idx, val )

	# Remove and return the LAST element (O(1), no shift needed) - list[T]
	# has no equivalent of this today; natural for a producer/consumer
	# queue/stack shape, which is exactly what sharing a list[T] across
	# threads is usually FOR. Composes __getitem__ (increfs) + erase_at
	# (reads the same slot again, decrefs) rather than a dedicated "take"
	# primitive on UnsafeList[T] - one extra, balanced incref/decref pair,
	# negligible next to the lock acquire/release this already pays for;
	# revisit only if profiling ever says otherwise.
	def pop( self ) -> Result[T, IndexError|BorrowError]:
		with self.__lock:
			if compiler.atomic_load( compiler.addrof( self.__borrows )) != 0:
				return Result.Err( BorrowError() )
			n: usize = self.__inner.__len__()
			if n == 0:
				return Result.Err( IndexError() )
			with compiler.wrap_arithmetic: # n > 0, just checked
				last: usize = n - 1
			val: T = self.__inner.__getitem__( last ).unwrap( 'list.pop: index in bounds by construction' )
			self.__inner.erase_at( last ).unwrap( 'list.pop: index in bounds by construction' )
			return Result.Ok( val )

	# Remove the element at idx, shifting everything after it one slot to
	# the left. Decrefs the removed element if T is RC.
	def erase_at( self, idx: usize ) -> Result[None, IndexError|BorrowError]:
		with self.__lock:
			if compiler.atomic_load( compiler.addrof( self.__borrows )) != 0:
				return Result.Err( BorrowError() )
			return self.__inner.erase_at( idx )

	# Borrow a read-only slice[T] view over the WHOLE buffer, valid until the
	# matching release_borrow() call. Unlike UnsafeList[T].as_slice() (safe
	# there because nothing else can touch an UnsafeList concurrently by
	# construction), a raw view into a LOCKED list[T]'s buffer would
	# otherwise dangle the instant this method returns and the lock
	# releases: another thread's append/insert/erase_at/pop could reallocate
	# or shift the buffer with no way for the borrower to ever know. Tracked
	# via an atomic borrow COUNT instead of holding the lock for the view's
	# whole lifetime (which would serialize every reader against every
	# other reader for no reason - multiple concurrent borrows are perfectly
	# safe, only a MUTATION racing a live borrow isn't): every mutator that
	# could invalidate a view checks this count FIRST (while holding the
	# lock, so the check itself is race-free) and refuses with BorrowError
	# rather than proceeding, the same "an outstanding export blocks a
	# resize" contract Python's own memoryview/buffer protocol enforces
	# over bytearray. Caller MUST pair this with release_borrow() (typically
	# via defer(), the same idiom every method here already uses for
	# __lock) - there is no automatic release: slice[T] is a plain @cstruct,
	# not an RCClass, so it has no __del__ to hook one into. Does NOT keep
	# this list[T] object itself alive - a slice[T] holds no reference back
	# to its origin, so destructing (not just mutating) the list while a
	# borrow is outstanding is still the caller's own responsibility to
	# avoid, exactly as it already is for UnsafeList[T].as_slice().
	def borrow_slice( self ) -> slice[T]:
		with self.__lock:
			view: slice[T] = self.__inner.as_slice()
			compiler.atomic_add( compiler.addrof( self.__borrows ), 1 )
			return view

	# Ends a borrow started by borrow_slice() - see its own comment. Safe to
	# call without holding __lock: this only needs to be atomic with respect
	# to the CHECK append/insert/erase_at/pop make against the same counter,
	# not with the rest of the container's own state.
	def release_borrow( self ) -> None:
		compiler.atomic_sub( compiler.addrof( self.__borrows ), 1 )

	# Erase all elements, decrefing each RC element first.
	def clear( self ) -> Result[None, BorrowError]:
		with self.__lock:
			if compiler.atomic_load( compiler.addrof( self.__borrows )) != 0:
				return Result.Err( BorrowError() )
			self.__inner.clear()
			return Result.Ok( None )

	# Hold the lock across more than one call - for compound, "check-then-
	# act" sequences that need to happen atomically (e.g. "append only if
	# not already full"), which no single method here can express safely
	# on its own. body is a bound-method closure (see the approved
	# atomics-closures-threading plan) - typically a method on the SAME
	# object that owns whatever else needs to be touched alongside this
	# list under the same critical section.
	def with_lock( self, body: Closure[[], None] ) -> None:
		with self.__lock:
			body()
