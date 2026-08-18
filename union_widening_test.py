# Real-compile-and-run regression tests for a compiler gap found while
# implementing lib/http/client.py's HTTPConnection/Response layer (see
# PLAN_HTTP_CLIENT.md's "A real compiler gap found while landing Phase 3a"):
# a value whose type is itself a NOMINAL `@union class Foo:` (e.g. HTTPError,
# with None-payload variants exactly like lib/builtins/__int.py's IntError)
# could not be propagated into a function whose declared return type is a
# WIDER union that includes it (e.g. Result[_, OSError|HTTPError]), through
# any of three equivalent forms: `.or_return()`, a direct `return
# Result.Err(e)`, or staging `e` through an explicitly wider-typed local
# first.
#
# Root causes (two distinct bugs, both in the "does E_op's OWN type stay
# opaque, or get decomposed into its own variants' payload types" question):
#
#   (1) type_resolver._require_result_return / lowering._maybe_widen_return_
#   result both computed "is E_op covered by E_fn" via TaggedUnion.leaves(),
#   which decomposes ANY TaggedUnion - including a real, user-declared
#   `@union class Foo:` - into its own variants' PAYLOAD types (all `None`
#   for HTTPError's shape). That's the right decomposition for RC-leaf
#   purposes (see TaggedUnion.is_rc()), but wrong here: a nominal union
#   nested as a whole member of some OTHER union needs to be compared as
#   ONE opaque leaf (itself), not decomposed. This is what produced the
#   reported ".or_return() ... requires Result[_,NoneType|NoneType|...]"
#   symptom. Fixed by type_resolver._atomic_leaves, which only decomposes a
#   SYNTHESIZED anonymous union (t.file is None - the same distinguishing
#   test discovery.py._get_or_create_union already uses when flattening a
#   wider union's own operands), leaving a nominal union as `[t]`.
#
#   (2) lowering._coerce_or_check_operand's union-coercion step explicitly
#   REFUSED to call _coerce_into_union whenever operand.type was itself ANY
#   TaggedUnion, on the theory that a union-shaped operand could never
#   legitimately be a member of a wider expected union - true for a
#   genuinely unrelated Result[T,OtherE], but false for exactly this shape
#   (operand.type IS one of expected_union's own members verbatim).
#   _coerce_into_union's own leaf-matching (by _same_type against each of
#   the union's attribute types) already handles this correctly once
#   reached - the guard just never let it try. This is what produced the
#   reported "Result.Err(e)" ambiguous-inference error (E inferred as both
#   the wide union from context and the narrow union from the argument) and
#   the staged-local "expected OSError|HTTPError, got HTTPError" error -
#   both go through the same _lower_expr -> _coerce_or_check_operand path.

import unittest

import test_support
from test_support import RealCompileMixin

# ErrA is a nominal @union (None-payload variants, matching lib/builtins/
# __int.py's IntError shape exactly) with its own real method so the test
# can check WHICH variant survived propagation without relying on match
# syntax nested two levels deep. ErrB exists purely to make ErrA|ErrB a
# genuinely WIDER union than ErrA alone - the shape that triggered both bugs.
_COMMON_PREFIX = '''
@union
class ErrA:
	X: None
	Y: None
	def variant_id( self ) -> i32:
		match self:
			case ErrA.X( _ ):
				return 1
			case ErrA.Y( _ ):
				return 2
		return -1

@union
class ErrB:
	Z: None
'''

# bug (1): .or_return() widening a nominal @union error into a bigger union.
_OR_RETURN_WIDENING = _COMMON_PREFIX + '''
def inner() -> Result[i32, ErrA]:
	return Result.Err( ErrA.Y( None ))

def widen() -> Result[i32, ErrA|ErrB]:
	v: i32 = inner().or_return()
	return Result.Ok( v )

def main() -> i32:
	match widen():
		case Result.Err( e ):
			match e:
				case ErrA( a ):
					if a.variant_id() != 2:
						return 1
				case ErrB( _ ):
					return 2
			return 0
		case Result.Ok( _ ):
			return 3
'''

# bug (2), form A: a plain `return Result.Err(e)` widening a nominal @union
# error into a bigger union - directly, no staging local.
_DIRECT_RESULT_ERR_WIDENING = _COMMON_PREFIX + '''
def widen( e: ErrA ) -> Result[i32, ErrA|ErrB]:
	return Result.Err( e )

def main() -> i32:
	match widen( ErrA.X( None )):
		case Result.Err( e ):
			match e:
				case ErrA( a ):
					if a.variant_id() != 1:
						return 1
				case ErrB( _ ):
					return 2
			return 0
		case Result.Ok( _ ):
			return 3
'''

# bug (2), form B: staging the value through an explicitly wider-typed local
# first (the exact workaround shape lib/socket.py's own _err_invalid() uses
# for an unrelated ambiguity) - confirmed to ALSO fail before the fix.
_STAGED_LOCAL_WIDENING = _COMMON_PREFIX + '''
def widen( e: ErrA ) -> Result[i32, ErrA|ErrB]:
	widened: ErrA|ErrB = e
	return Result.Err( widened )

def main() -> i32:
	match widen( ErrA.Y( None )):
		case Result.Err( e ):
			match e:
				case ErrA( a ):
					if a.variant_id() != 2:
						return 1
				case ErrB( _ ):
					return 2
			return 0
		case Result.Ok( _ ):
			return 3
'''

# RC dimension of bug (1): does .or_return()'s widening (emitter_c.py's
# _emit_widen_error, the SINGLE-class struct-copy branch this fix now also
# routes a nominal union through) correctly preserve ownership when the
# nominal union's own live variant carries a real RC leaf (not just the
# None-payload shape every other case above uses)? The widen path is a bare
# struct copy of e_op's WHOLE value (tag+payload together, see
# _emit_widen_error's own docstring on why this is safe regardless of what's
# inside) - the only incref that should ever fire is ErrC.Boxed's own
# constructor storing its BORROWED `b` argument, exactly once, the same
# invariant union_coercion_rc_test.py's _RETURN_BORROWED_THROUGH_UNION checks
# for the unrelated (non-widening) union-coercion path.
_OR_RETURN_WIDENING_RC_LEAF = '''
class Box:
	v: i32
	def __init__( self, v: i32 ) -> None:
		self.v = v

@union
class ErrC:
	Nothing: None
	Boxed: Box

@union
class ErrD:
	Z: None

def inner( b: Box ) -> Result[i32, ErrC]:
	return Result.Err( ErrC.Boxed( b ))

def widen( b: Box ) -> Result[i32, ErrC|ErrD]:
	v: i32 = inner( b ).or_return()
	return Result.Ok( v )

def main() -> i32:
	with compiler.wrap_arithmetic:
		b: Box = Box( v = 99 )
		before: usize = compiler.refcount( b )
		r = widen( b )
		after: usize = compiler.refcount( b )
		rc_ok: bool = after == before + 1
		match r:
			case Result.Err( e ):
				match e:
					case ErrC( c ):
						match c:
							case ErrC.Boxed( boxed ):
								if not rc_ok:
									return 1
								if boxed.v != 99:
									return 2
							case ErrC.Nothing( _ ):
								return 3
					case ErrD( _ ):
						return 4
			case Result.Ok( _ ):
				return 5
	return 0
'''


@unittest.skipUnless( test_support.HAS_CC, 'no C compiler (clang/gcc/msvc) found - skipping real-compile tests' )
class UnionWideningTests( RealCompileMixin, unittest.TestCase ):
	def test_nominal_union_widens_into_a_bigger_union( self ) -> None:
		self.assert_programs_run([
			( 'or_return_widening', _OR_RETURN_WIDENING ),
			( 'direct_result_err_widening', _DIRECT_RESULT_ERR_WIDENING ),
			( 'staged_local_widening', _STAGED_LOCAL_WIDENING ),
			( 'or_return_widening_rc_leaf', _OR_RETURN_WIDENING_RC_LEAF ),
		])


if __name__ == '__main__':
	unittest.main()
