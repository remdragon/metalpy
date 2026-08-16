# Real-compile-and-run regression tests for return-type narrowing on
# Result[T,E].unwrap_or() (lib/builtins/__init__.py) when T is an ORDINARY
# (non-Optional) type - the general case left open by an earlier fix to the
# T-already-Optional shape (Result[i32|None,E].unwrap_or(x), see
# unwrap_or_overload_default_test.py in worktree
# fix-unwrap-or-overload-collision).
#
# unwrap_or is an @overload group:
#   @overload
#   def unwrap_or( self, default: T ) -> T:
#       ...
#   def unwrap_or( self, default: T|None = None ) -> T|None:
#       ...
# Calling it with an explicit, non-None default on an ORDINARY (non-Optional)
# T (e.g. Result[i32,str].unwrap_or(5)) should statically narrow the result
# to T (never None) - the whole point of the stub existing at all. Making
# that actually work (rather than either miscompiling or safely rejecting
# the call) needed THREE cooperating fixes in lowering.py:
#
# 1. _lower_call's Overload branch no longer builds a `replace()`d copy of
#    the shared implementation Function with an overridden return_type
#    (the old mechanism) - that copy was independently scheduled/lowered as
#    if it were its own real compile unit, but the shared body itself is
#    still typed against the WIDE `default: T|None` parameter (`return
#    default` genuinely IS T|None-typed there), so re-lowering it under a
#    narrower declared return type failed its own internal type-check
#    ("function returns T, not T|None") the moment T wasn't already
#    Optional (when T WAS already Optional, T and T|None happened to
#    collapse to the exact same interned type, masking this - see
#    unwrap_or_overload_default_test.py). Fixed by never re-lowering a
#    narrower-typed copy of the shared body at all: the call always targets
#    the ORIGINAL, shared Function, and the narrowing is instead realized
#    by extracting the matching leaf out of the call's real (wide-typed)
#    result via _maybe_unwrap_union_arg - sound specifically because
#    overload_resolution.stub_covers_call already proved this call's own
#    arguments can never actually produce the other (narrowed-away) member,
#    so no runtime tag check is needed.
#
# 2. _lower_overload_arg (which types a bare literal argument to an
#    overloaded call before the winning candidate is even chosen) redirects
#    a stub to its bound_to implementation's own parameter type first - a
#    stub is "signature-only, never actually called" (mpy_types.Overload's
#    own docstring), so its own (often narrower) declared type isn't what
#    the literal actually needs to satisfy at the real, emitted call.
#    Without this, a stub's narrower scalar type and its own bound
#    implementation's wider union type were treated as two independent
#    candidates, and this method's own magnitude-based int-literal
#    disambiguation (correctly designed for choosing between genuinely
#    different overload arms, e.g. i8 vs i32) picked the narrower SCALAR
#    one - the literal was then lowered against that type while the actual
#    call still targets the wider, union-typed implementation.
#
# 3. Once redirected, a union-typed candidate is matched via exactly one of
#    its own leaves (not just a direct top-level scalar match) - but the
#    literal itself is then lowered using that matched LEAF's own narrow
#    type, not the whole union: making the operand's own static type the
#    whole union (even though the literal's actual value is obviously,
#    unambiguously never the None variant) confused overload_resolution.
#    resolve_call's own dispatch computation (a pure function of types) into
#    building an unnecessary RUNTIME conditional dispatch - and that
#    dispatch path (_emit_dispatch_call) has no notion of a bound method
#    RECEIVER at all, so the call's own `self` argument went missing from
#    the emitted C entirely. A new post-resolution coercion step (mirroring
#    _coerce_into_union, once the winning target is actually known) wraps
#    the narrow-typed operand into the union JUST for the real call's own
#    argument list.
#
# Each of these three was independently confirmed via a real compile during
# development (removing any one in isolation reproduces a distinct failure -
# a "function returns ..." internal type mismatch, a "passing 'int' to
# parameter of incompatible type" C error, or a "too few arguments"/missing-
# receiver C error from a spurious conditional dispatch).

import unittest

import test_support
from test_support import RealCompileMixin

_UNWRAP_OR_EXPLICIT_DEFAULT_ON_ORDINARY_T = '''
def maybe( flag: bool ) -> Result[i32, str]:
	if flag:
		return Result.Ok( 7 )
	return Result.Err( 'nope' )

def main() -> i32:
	# explicit-default form on the Ok path - narrowed result (i32, not
	# i32|None) never needs an is-None check to use directly
	r1: Result[i32, str] = maybe( True )
	v1: i32 = r1.unwrap_or( 5 )
	if v1 != 7:
		return 1

	# explicit-default form on the Err path - falls back to the given
	# default, still narrowed to i32
	r2: Result[i32, str] = maybe( False )
	v2: i32 = r2.unwrap_or( 5 )
	if v2 != 5:
		return 2

	# zero-arg form on the Err path - still the impl's own WIDE i32|None
	# return type (the stub never covers a zero-argument call)
	r3: Result[i32, str] = maybe( False )
	v3: i32|None = r3.unwrap_or()
	if v3 is not None:
		return 3

	# zero-arg form on the Ok path
	r4: Result[i32, str] = maybe( True )
	v4: i32|None = r4.unwrap_or()
	if v4 is None:
		return 4
	if v4 != 7:
		return 5
	return 0
'''

_UNWRAP_OR_BOX_SAME_TYPE_STUB_AND_IMPL = '''
@union
class Box[T]:
	Some: T

	@overload
	def get_or( self, default: T ) -> T:
		...
	def get_or( self, default: T ) -> T:
		return self.data.v_Some

def main() -> i32:
	b: Box[i32] = Box.Some( 5 )
	fallback: i32 = -1
	w: i32 = b.get_or( fallback )
	if w != 5:
		return 1
	return 0
'''


@unittest.skipUnless( test_support.HAS_CC, 'no C compiler (clang/gcc/msvc) found - skipping real-compile RC tests' )
class UnwrapOrGeneralNarrowingTests( RealCompileMixin, unittest.TestCase ):
	def test_unwrap_or_explicit_default_on_ordinary_t( self ) -> None:
		self.assert_programs_run([ ( 'unwrap_or_explicit_default_on_ordinary_t', _UNWRAP_OR_EXPLICIT_DEFAULT_ON_ORDINARY_T ) ])

	def test_unwrap_or_box_same_type_stub_and_impl( self ) -> None:
		self.assert_programs_run([ ( 'unwrap_or_box_same_type_stub_and_impl', _UNWRAP_OR_BOX_SAME_TYPE_STUB_AND_IMPL ) ])


if __name__ == '__main__':
	unittest.main()
