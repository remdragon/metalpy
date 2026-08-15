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
# No {1, 2, 3} literal syntax exists for this (lowering.py has no ast.Set/
# ast.SetComp handling at all) - construct via set[T]() + .add(...).
#
# x in my_set / del my_set[x] syntax sugar don't exist either - same
# limitation dict[K,V].__contains__/__delitem__ already have (lowering.py
# doesn't lower ast.In/ast.NotIn, and _stmt_Delete only accepts a bare
# name) - __contains__/discard/remove must be called directly.
#
# __len__ + __getitem__(idx: usize) -> Result[T, IndexError] together are
# exactly the "indexable" shape lowering.py's for-loop lowering
# (lowering.py:3667-3731) looks for - `for x in my_set:` therefore works
# with no compiler changes, same as list[T]/dict[K,V].key_at.
class set[T]:
	__inner: dict[T, bool]

	def __init__( self ) -> None:
		self.__inner = dict[T, bool]()

	def __len__( self ) -> usize:
		return self.__inner.__len__()

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
	def __getitem__( self, index: usize ) -> Result[T, IndexError]:
		return self.__inner.key_at( index )
