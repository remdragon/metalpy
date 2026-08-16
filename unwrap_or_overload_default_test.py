# Real-compile-and-run regression tests for Result[T,E].unwrap_or() called
# WITH an explicit, non-None default argument, on a Result whose Ok-payload T
# is itself Optional (e.g. Result[i32|None,str]) - the T-already-Optional
# counterpart to unwrap_or_general_narrowing_test.py's ordinary-T tests. Both
# shapes are fixed by the SAME lowering.py changes (see that file's own,
# more detailed header comment for the full three-part root-cause story);
# this file specifically pins down the T-already-Optional edge case, where
# the stub's own `T` and the impl's own `T|None` substitute to the exact
# SAME interned union object (once discovery.py's own _get_or_create_union
# correctly flattens/dedupes an already-Optional T - a hard prerequisite,
# ported into this worktree's discovery.py alongside these fixes).

import unittest

import test_support
from test_support import RealCompileMixin

_UNWRAP_OR_EXPLICIT_DEFAULT_ON_OPTIONAL_OK_PAYLOAD = '''
def maybe_value( ok: bool, present: bool ) -> Result[i32|None, str]:
	if not ok:
		return Result.Err( 'nope' )
	if present:
		return Result.Ok( 7 )
	return Result.Ok( None )

def main() -> i32:
	# explicit-default form on the Ok(present) path - default is never used
	r1: Result[i32|None, str] = maybe_value( True, True )
	v1: i32|None = r1.unwrap_or( 42 )
	if v1 is None:
		return 1
	if v1 != 7:
		return 2

	# explicit-default form on the Ok(None) path - the Ok payload IS None,
	# so unwrap_or must still return None (not fall back to the default -
	# only the Err path falls back)
	r2: Result[i32|None, str] = maybe_value( True, False )
	v2: i32|None = r2.unwrap_or( 42 )
	if v2 is not None:
		return 3

	# explicit-default form on the Err path - falls back to the given default
	r3: Result[i32|None, str] = maybe_value( False, True )
	v3: i32|None = r3.unwrap_or( 42 )
	if v3 is None:
		return 4
	if v3 != 42:
		return 5

	# zero-arg AND explicit-default calls to the SAME group in the SAME
	# program - the exact shape that surfaced the redefinition half of the
	# original T-already-Optional bug (two independently-lowered Functions
	# sharing one mangled C symbol)
	r4: Result[i32|None, str] = maybe_value( True, True )
	v4: i32|None = r4.unwrap_or()
	if v4 is None:
		return 6
	if v4 != 7:
		return 7
	return 0
'''


@unittest.skipUnless( test_support.HAS_CC, 'no C compiler (clang/gcc/msvc) found - skipping real-compile RC tests' )
class UnwrapOrOverloadDefaultTests( RealCompileMixin, unittest.TestCase ):
	def test_unwrap_or_explicit_default_on_optional_ok_payload( self ) -> None:
		self.assert_programs_run([ ( 'unwrap_or_explicit_default_on_optional_ok_payload', _UNWRAP_OR_EXPLICIT_DEFAULT_ON_OPTIONAL_OK_PAYLOAD ) ])


if __name__ == '__main__':
	unittest.main()
