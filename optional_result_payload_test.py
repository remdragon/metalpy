# Real-compile-and-run regression tests for two related pre-existing bugs
# affecting Result[T,E] whose own Ok-payload T is itself Optional (X|None) -
# found incidentally while adding .send() support to generator functions
# (PLAN_GENERATORS.md Phase C: a fallible Generator[T,E]'s own __next__()
# returns exactly Result[T|None,E] by construction, so ANY fallible generator's
# `g.__next__().unwrap_or()` hit bug 1 below), but neither is generator-
# specific - both reproduce against a plain hand-written Result[i32|None,str].
#
# Bug 1 (discovery.py's _get_or_create_union): Result[T,E].unwrap_or()
# (lib/builtins/__init__.py) is declared `-> T|None`. Monomorphizing that
# declared return type for T=i32|None substitutes T with the concrete
# i32|None union, producing operands [i32|None, NoneType] fed to
# _get_or_create_union - which, before this fix, treated the already-a-union
# i32|None operand as a single OPAQUE member (its own qualname already
# containing '|') rather than flattening it, so the synthesized "T|None"
# return type came out as a doubled "intrinsics.NoneType|intrinsics.NoneType|
# intrinsics.i32" instead of collapsing to the correct, flat
# "intrinsics.NoneType|intrinsics.i32" - which then mismatched the Ok-payload
# local's own (correctly flat) type, rejecting `return ok` inside unwrap_or's
# own body with a bogus "function returns ...NoneType|...NoneType|...i32, not
# ...NoneType|...i32" compile error on every instantiation, since Result[T,E]
# is a real (non-generic-body) class whose methods share one compiled body.
#
# Bug 2 (type_resolver.py's _ReferenceResolver, the `case Owner.Member(x):`
# capture-bind branch of _match_pattern): a match-case-bound name never got
# an entry in self.locals, the best-effort forward type tracker every other
# None-narrowing rewrite (`if x is not None:`) consults via _type_of_expr -
# an ORDINARY assigned local gets one for free (visit_Assign), but a bind
# synthesized directly by _match_pattern and merely prepended to the case
# body (never itself run through self.visit()) did not. So `if v is not
# None:` on a match-bound v: T|None could never be recognized as a
# narrowable union-typed name at all, and the narrowing an ordinary local
# would receive silently never applied - confirmed via a real compile:
# `match r: case Result.Ok(v): if v is not None: result = v` (v: i32|None,
# result: i32) was rejected as an i32|None-into-i32 mismatch, even though
# copying v into a fresh ordinary local first and narrowing THAT compiled
# and ran correctly (the workaround used to route around this bug while it
# was still open).

import unittest

import test_support
from test_support import RealCompileMixin

_UNWRAP_OR_ON_OPTIONAL_OK_PAYLOAD = '''
def maybe_value( ok: bool, present: bool ) -> Result[i32|None, str]:
	if not ok:
		return Result.Err( 'nope' )
	if present:
		return Result.Ok( 7 )
	return Result.Ok( None )

def main() -> i32:
	# zero-arg form - the exact shape a fallible generator's own
	# g.__next__().unwrap_or() uses, and the one the compile error was
	# originally found through
	r1: Result[i32|None, str] = maybe_value( True, True )
	v1: i32|None = r1.unwrap_or()
	if v1 is None:
		return 1
	if v1 != 7:
		return 2

	r2: Result[i32|None, str] = maybe_value( True, False )
	v2: i32|None = r2.unwrap_or()
	if v2 is not None:
		return 3

	# zero-arg form on the Err path - falls back to the implicit None default
	r3: Result[i32|None, str] = maybe_value( False, True )
	v3: i32|None = r3.unwrap_or()
	if v3 is not None:
		return 4
	return 0
'''

_MATCH_BOUND_NAME_NARROWS_ON_IS_NOT_NONE = '''
def maybe_value( ok: bool, present: bool ) -> Result[i32|None, str]:
	if not ok:
		return Result.Err( 'nope' )
	if present:
		return Result.Ok( 7 )
	return Result.Ok( None )

def classify( ok: bool, present: bool ) -> i32:
	r: Result[i32|None, str] = maybe_value( ok, present )
	result: i32 = 0
	match r:
		case Result.Ok( v ):
			if v is not None:
				result = v
			else:
				result = -1
		case Result.Err( e ):
			result = -2
	return result

def main() -> i32:
	if classify( True, True ) != 7:
		return 1
	if classify( True, False ) != -1:
		return 2
	if classify( False, True ) != -2:
		return 3
	return 0
'''


@unittest.skipUnless( test_support.HAS_CC, 'no C compiler (clang/gcc/msvc) found - skipping real-compile RC tests' )
class OptionalResultPayloadTests( RealCompileMixin, unittest.TestCase ):
	def test_unwrap_or_on_optional_ok_payload( self ) -> None:
		self.assert_programs_run([ ( 'unwrap_or_on_optional_ok_payload', _UNWRAP_OR_ON_OPTIONAL_OK_PAYLOAD ) ])

	def test_match_bound_name_narrows_on_is_not_none( self ) -> None:
		self.assert_programs_run([ ( 'match_bound_name_narrows_on_is_not_none', _MATCH_BOUND_NAME_NARROWS_ON_IS_NOT_NONE ) ])


if __name__ == '__main__':
	unittest.main()
