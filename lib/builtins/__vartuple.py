# VariadicTuple[T]: the REAL backing type for `tuple[T, ...]` (Python's own
# spelling for "a variable-length, homogeneous tuple" - distinct from the
# fixed-arity, heterogeneous tuple[T0,T1,...] TupleStorage synthesizes on
# demand, see tuple_storage.py). Immutable after construction - no append/
# insert/__setitem__ - so, like the fixed-arity tuple, it needs no
# FastLock-style thread-safety wrapper (PLAN_TUPLE.md's own reasoning:
# atomic refcounting alone already makes a construct-once value safely
# shareable across threads).
#
# Built directly on UnsafeList[T] (the same unlocked storage list[T] itself
# wraps) rather than a bespoke layout - the real API surface (__getitem__,
# __len__, __iter__, slicing) is exactly what UnsafeList[T] already
# provides, just with the mutating methods never exposed here.
#
# `tuple(some_list)` (the public, callable spelling - real Python's own
# `tuple(iterable)` constructor) is a separate top-level generic function,
# not this class's own __init__ - keeps the TYPE (VariadicTuple[T], what
# `tuple[T, ...]` resolves to - see discovery.py's visit_Subscript) and the
# CALLABLE (tuple(...), ordinary generic-function type inference from its
# argument - confirmed already supported for a bare, non-subscripted
# generic construction call) as two distinct, unconfusable names, rather
# than one class named literally `tuple` sitting alongside the unrelated
# fixed-arity `tuple[...]` spelling.

import compiler

class VariadicTuple[T]( Sequence[T], Iterable[T] ):
	__inner: UnsafeList[T]

	def __init__( self, src: list[T] ) -> None:
		raw: UnsafeList[T] = UnsafeList[T]( len( src ))
		i: usize = 0
		while i < len( src ):
			raw.append( src.__getitem__( i ).unwrap( 'VariadicTuple.__init__: index in bounds by construction' ))
			with compiler.wrap_arithmetic:
				i += 1
		self.__inner = raw

	def __len__( self ) -> usize:
		return self.__inner.__len__()

	def __bool__( self ) -> bool:
		# Python-style truthiness: an empty tuple is falsy - see
		# builtins.str.__bool__'s own docstring for why lowering.py's bare
		# (non-union) truthiness testing needs this dunder explicitly
		return self.__len__() != 0

	# Access element by position. Returns a copy (with incref if RC).
	@overload
	def __getitem__( self, idx: usize ) -> Result[T, IndexError]:
		return self.__inner.__getitem__( idx )

	# t[a:b] slice syntax (lowering.py's _lower_slice_subscript) - a NEW
	# VariadicTuple[T], every element copied (incref'd if RC), matching
	# every other slice target's own copy semantics (_resolve_slice_bounds).
	@overload
	def __getitem__( self, s: slice ) -> VariadicTuple[T]:
		( start, stop ) = _resolve_slice_bounds( s, self.__inner.__len__() )
		raw: UnsafeList[T] = UnsafeList[T]()
		i: usize = start
		while i < stop:
			raw.append( self.__inner.__getitem__( i ).unwrap( 'VariadicTuple.__getitem__(slice): index in bounds by construction' ))
			with compiler.wrap_arithmetic:
				i += 1
		return VariadicTuple[T].__allocate__( __inner = raw )

	def __iter__( self ) -> Generator[T, StopIteration]:
		return _sequence_iter( self )

def tuple[T]( src: list[T] ) -> VariadicTuple[T]:
	''' tuple(iterable) - real Python's own conversion constructor. list[T]
	only for now (not a general Iterable[T]) - a real caller needing more
	can widen this later; VariadicTuple.__init__'s own __len__()-driven
	exact-capacity allocation already needs random access anyway. '''
	return VariadicTuple[T]( src )
