# set[T]: mutable collection of unique values (SYNTAX.md). Wraps a single
# dict[T, bool] rather than adding a genuinely value-less RawDict variant -
# T IS the dict's key, and the bool is a dummy always-True value existing
# purely so dict[K,V]'s existing machinery (which needs SOME V) can be
# reused unmodified.
#
# bool, not None/NoneType, for that dummy value - deliberately, not a style
# choice: NoneType as a generic value-type argument hits a real, separate
# compiler bug today. lowering.py's compiler.sizeof(...) resolution (around
# line 2834) does `if size := getattr(target_type, 'sizeof', None)`, and
# NoneType.sizeof == 0 is falsy, so it wrongly falls through to a path that
# only accepts RCClass/CStruct/CUnion/TaggedUnion, raising
# "compiler.sizeof(NoneType) is not supported yet". Fixing that is out of
# scope here - bool (sizeof 1, always the truthy branch of that same
# getattr check) sidesteps it entirely, at the cost of one real byte per
# stored element, same overhead dict[K,V] already pays for any other
# value-typed V.
#
# {1, 2, 3} literal syntax works (lowering.py's _expr_Set) - {x for x in y}
# comprehensions don't (no comprehension-lowering infrastructure exists
# anywhere in this compiler yet, list comprehensions aren't implemented
# either).
#
# x in my_set / x not in my_set work (lowering.py's _lower_in_comparison,
# dispatching to __contains__ below). del my_set[x] does NOT - _stmt_Delete
# only accepts a bare name - discard()/remove() must be called directly.
#
# __len__ + __getitem__(idx: usize) -> Result[T, IndexError] together
# already match Sequence[T]'s own shape - `for x in my_set:` needs a real,
# declared Iterable[T] conformance (a for-loop's own subject must conform to
# Iterator[T] or Iterable[T], no structural duck-typing - lowering.py's
# _stmt_For), so __iter__ (below) delegates to _sequence_iter like list[T]'s
# own does.
class set[T]( Sequence[T], Iterable[T], Sized ):
	__inner: dict[T, bool]

	def __init__( self ) -> None:
		self.__inner = dict[T, bool]()

	def __len__( self ) -> usize:
		return self.__inner.__len__()

	def __bool__( self ) -> bool:
		# Python-style truthiness: an empty set is falsy - see
		# builtins.str.__bool__'s own docstring for why lowering.py's bare
		# (non-union) truthiness testing needs this dunder explicitly
		# rather than defaulting to always-true.
		return self.__len__() != 0

	# Add value to the set. Idempotent: dict[K,V].__setitem__'s own
	# overwrite-existing-key path makes inserting a value that's already
	# present a harmless dummy-value overwrite (True -> True), matching
	# Python set.add's own no-op-if-present semantics.
	def add( self, value: T ) -> None:
		self.__inner.__setitem__( value, True )

	def __contains__( self, value: T ) -> bool:
		return self.__inner.__contains__( value )

	# Python set.discard semantics: a no-op if value isn't present. The
	# compiler rejects a bare, unconsumed Result-returning call as a
	# statement (a discarded Result must be explicitly handled), so both
	# arms are matched and dropped here on purpose - neither one owns
	# anything (Ok is None, Err is an empty KeyError), so there's nothing
	# to leak or double-release either way.
	def discard( self, value: T ) -> None:
		match self.__inner.__delitem__( value ):
			case Result.Ok( _ ):
				pass
			case Result.Err( _ ):
				pass

	# Python set.remove semantics: unlike discard, the caller DOES want to
	# know if value wasn't present - forwarded as-is, never swallowed.
	def remove( self, value: T ) -> Result[None, KeyError]:
		return self.__inner.__delitem__( value )

	# Positional access - set[T]'s own "value" IS the underlying dict's
	# key, so this forwards straight to key_at (current live-entry order,
	# see UnsafeDict.key_at's own comment - order is unspecified/can shift
	# across a discard/remove, same as Python's own unordered set).
	@overload
	def __getitem__( self, index: usize ) -> Result[T, IndexError]:
		return self.__inner.key_at( index )

	# real Python's own negative-index convention - see _resolve_index's
	# own comment for why this is a sibling overload, not a widened
	# usize->isize parameter above. Positional order here is the same
	# unspecified/can-shift live-entry order __getitem__(usize) above
	# already documents, same caveat applies.
	@overload
	def __getitem__( self, index: isize ) -> Result[T, IndexError]:
		match _resolve_index( index, self.__len__() ):
			case Result.Ok( resolved ):
				return self.__getitem__( resolved )
			case Result.Err( e ):
				return Result.Err( e )

	def __iter__( self ) -> Generator[T, StopIteration]:
		return _sequence_iter( self )

	# --- set algebra ---------------------------------------------------
	# union/intersection/difference/symmetric_difference each build and
	# return a FRESH set[T], never mutating self/other - matching Python's
	# own set algebra semantics. All four walk self/other via a manual
	# index loop (self.__getitem__(i).unwrap(...)) rather than
	# `for x in self:` - the for-loop-over-indexable sugar's per-element
	# bind desugars to self[i].or_return(), which requires the ENCLOSING
	# function to itself return a Result[_,IndexError]-shaped type (see
	# emitter_c_test.py's SetTests own checksum_set helper, forced into
	# exactly this shape for the same reason) - these methods return
	# set[T]/bool instead, so or_return() would have nowhere to propagate
	# to. unwrap() here never actually panics: every index walked is
	# always < __len__() at the moment it's read, the same "no gaps"
	# invariant key_at/value_at's own comments already rely on.

	def union( self, other: set[T] ) -> set[T]:
		result: set[T] = set[T]()
		i: usize = 0
		with compiler.wrap_arithmetic:
			while i < self.__len__():
				result.add( self.__getitem__( i ).unwrap( 'set.union: index in bounds by construction' ))
				i += 1
		j: usize = 0
		with compiler.wrap_arithmetic:
			while j < other.__len__():
				result.add( other.__getitem__( j ).unwrap( 'set.union: index in bounds by construction' ))
				j += 1
		return result

	def __or__( self, other: set[T] ) -> set[T]:
		return self.union( other )

	def intersection( self, other: set[T] ) -> set[T]:
		result: set[T] = set[T]()
		i: usize = 0
		with compiler.wrap_arithmetic:
			while i < self.__len__():
				value: T = self.__getitem__( i ).unwrap( 'set.intersection: index in bounds by construction' )
				if other.__contains__( value ):
					result.add( value )
				i += 1
		return result

	def __and__( self, other: set[T] ) -> set[T]:
		return self.intersection( other )

	def difference( self, other: set[T] ) -> set[T]:
		result: set[T] = set[T]()
		i: usize = 0
		with compiler.wrap_arithmetic:
			while i < self.__len__():
				value: T = self.__getitem__( i ).unwrap( 'set.difference: index in bounds by construction' )
				if not other.__contains__( value ):
					result.add( value )
				i += 1
		return result

	def __sub__( self, other: set[T] ) -> set[T]:
		return self.difference( other )

	# self.difference(other) already covers "in self, not in other" - only
	# the reverse direction ("in other, not in self") needs its own walk
	def symmetric_difference( self, other: set[T] ) -> set[T]:
		result: set[T] = self.difference( other )
		j: usize = 0
		with compiler.wrap_arithmetic:
			while j < other.__len__():
				value: T = other.__getitem__( j ).unwrap( 'set.symmetric_difference: index in bounds by construction' )
				if not self.__contains__( value ):
					result.add( value )
				j += 1
		return result

	def __xor__( self, other: set[T] ) -> set[T]:
		return self.symmetric_difference( other )

	# same length + one-directional containment check - a Python set's own
	# equality contract (unordered, so no positional comparison makes sense)
	def __eq__( self, other: set[T] ) -> bool:
		if self.__len__() != other.__len__():
			return False
		i: usize = 0
		with compiler.wrap_arithmetic:
			while i < self.__len__():
				value: T = self.__getitem__( i ).unwrap( 'set.__eq__: index in bounds by construction' )
				if not other.__contains__( value ):
					return False
				i += 1
		return True

	# _COMP_DUNDER (lowering.py) dispatches != to __ne__ directly - it
	# never auto-derives one from __eq__ - so this is required, not
	# optional, same as str's own __eq__/__ne__ pair
	def __ne__( self, other: set[T] ) -> bool:
		return not ( self == other )
