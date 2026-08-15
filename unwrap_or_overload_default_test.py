# Real-compile-and-run regression tests for a pre-existing bug in
# Result[T,E].unwrap_or() (lib/builtins/__init__.py), reached by calling it
# WITH an explicit, non-None default argument (`.unwrap_or(42)`, as opposed
# to the zero-argument `.unwrap_or()` form) on a Result whose Ok-payload T is
# itself Optional (e.g. Result[i32|None,str]) - found as a side-discovery
# while fixing two OTHER, related Optional/Result bugs (discovery.py's
# _get_or_create_union flattening fix and type_resolver.py's match-bound
# narrowing fix; the union-flattening half of that fix is a hard prerequisite
# for this test file even compiling Result[i32|None,str] at all, and is
# ported into this worktree's discovery.py alongside it).
#
# unwrap_or is declared as an @overload group:
#   @overload
#   def unwrap_or( self, default: T ) -> T:
#       ...
#   def unwrap_or( self, default: T|None = None ) -> T|None:
#       ...
# For an ordinary (non-Optional) T, the stub's `-> T` and the plain impl's
# `-> T|None` are genuinely different return types, so lowering.py's
# _lower_call/_resolve_original correctly builds a distinct, `replace()`d
# view of the impl Function with the narrower return type for a call whose
# argument matches the stub's own domain. But when T is ALREADY Optional
# (T = i32|None), the stub's substituted `T` and the impl's substituted
# `T|None` collapse to the exact SAME interned union object (once
# discovery.py's _get_or_create_union correctly flattens/dedupes) - so
# _resolve_original was STILL unconditionally building a distinct `replace()`
# copy even though there was nothing left to narrow. That copy then got
# independently scheduled/lowered as its own real compile unit
# (self.lowering._ensure_resolved), while emitter_c.py's own
# mangle_function_qualname mangles purely off (qualname, overload_group) with
# no notion of "this Function object is a distinct return-type view of
# another" - so it produced the SAME mangled C symbol name as the original,
# already-scheduled impl. Two independently-lowered function bodies sharing
# one C symbol is a straight "redefinition of ..." C compile error. Fixed by
# only building the replace()'d copy when the narrowed return type is
# genuinely a different object from the original.
#
# A second, independent bug compounded this at the exact same call site: an
# explicit literal argument to an OVERLOADED call (lowering.py's
# _lower_overload_arg) only ever recognized a candidate parameter as
# "compatible with this literal" when the parameter's type was DIRECTLY a
# matching scalar (`param.type.stem in compatible_stems`) - never when the
# parameter's type was a UNION that merely CONTAINS one such scalar leaf
# (exactly unwrap_or's own `default: T|None` shape). So a literal default
# argument was never given the union type as its expected_type, silently
# skipping the ordinary _coerce_into_union wrapping every plain (non-
# overloaded) call site already gets for free - the literal reached
# emitter_c.py as a bare scalar handed to a C parameter whose real type is
# the whole tagged-union struct: "passing 'int' to parameter of incompatible
# type 'struct $__u$$...'". Fixed by also accepting a union-typed candidate
# parameter whenever exactly one of its own leaves is compatible with the
# literal's kind.
#
# Both bugs needed fixing together for `.unwrap_or(42)` on
# Result[i32|None,str] to compile and run correctly - each is exercised (and
# individually confirmed, via real compiles during development) by removing
# the other's fix in isolation.

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
	# program - the exact shape that surfaced the redefinition half of this
	# bug (two independently-lowered Functions sharing one mangled C symbol)
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
