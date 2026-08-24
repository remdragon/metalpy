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

	# takes ownership of an already-built UnsafeList[T] directly - the
	# public, iterable-driven construction path lives in the top-level
	# tuple(...) function below (built this way, not via __allocate__,
	# specifically so a free function can call it - __allocate__ itself is
	# restricted to a method of the class it constructs)
	def __init__( self, raw: UnsafeList[T] ) -> None:
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
		return VariadicTuple[T]( raw )

	def __iter__( self ) -> Generator[T, StopIteration]:
		return _sequence_iter( self )

	# positional comparison, real Python's own tuple equality contract
	# (unlike set[T].__eq__ above - unordered, same length + containment)
	def __eq__( self, other: VariadicTuple[T] ) -> bool:
		if self.__len__() != other.__len__():
			return False
		i: usize = 0
		with compiler.wrap_arithmetic:
			while i < self.__len__():
				a: T = self.__inner.__getitem__( i ).unwrap( 'VariadicTuple.__eq__: index in bounds by construction' )
				b: T = other.__inner.__getitem__( i ).unwrap( 'VariadicTuple.__eq__: index in bounds by construction' )
				if a != b:
					return False
				i += 1
		return True

	# _COMP_DUNDER (lowering.py) dispatches != to __ne__ directly - it never
	# auto-derives one from __eq__ - see set[T].__ne__'s own identical note
	def __ne__( self, other: VariadicTuple[T] ) -> bool:
		return not ( self == other )

	def __repr__( self ) -> str:
		# real Python's own trailing-comma convention for a 1-element tuple
		# ('(1,)') - disambiguates from a plain parenthesized expression in
		# SOURCE syntax; not strictly needed for a printed VALUE the way it
		# is for source, but kept for the familiar, recognizable shape
		if self.__len__() == 0:
			return '()'
		parts: list[str] = list[str]()
		i: usize = 0
		with compiler.wrap_arithmetic:
			while i < self.__len__():
				val: T = self.__inner.__getitem__( i ).unwrap( 'VariadicTuple.__repr__: index in bounds by construction' )
				parts.append( str( val ))
				i += 1
		if self.__len__() == 1:
			return '(' + parts.__getitem__( 0 ).unwrap( 'VariadicTuple.__repr__: just appended' ) + ',)'
		return '(' + ', '.join( parts ) + ')'

	def __str__( self ) -> str:
		return self.__repr__()

def tuple[T, S: Iterable[T]]( src: S ) -> VariadicTuple[T]:
	''' tuple(iterable) - real Python's own conversion constructor, over any
	Iterable[T] (not just list[T] - a for-loop over src drives its own
	__iter__()/Generator[T,StopIteration] the same way any other for-loop
	over a real iterable does, no manual __next__() driving needed here). '''
	raw: UnsafeList[T] = UnsafeList[T]()
	for item in src:
		raw.append( item )
	return VariadicTuple[T]( raw )
