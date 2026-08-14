# Real-compile-and-run regression tests for a use-after-free bug in
# `.or_return()` called on a NAMED Result-typed local variable (as opposed to a
# bare call-expression receiver, e.g. `foo().or_return()`).
#
# Root cause (see lowering.py's _consume_checked_result and cfg.py's
# current_epilogue_label/return_): the Err branch copies the receiver's error
# payload into the returned struct (unretained - ownership is meant to
# transfer), then jumped into the receiver's OWN shared epilogue label, which
# unconditionally decref'd that SAME payload a second time - a genuine
# double-free. Fixed by passing the receiver into current_epilogue_label() the
# same way _stmt_Return already does for a bare `return x`, so the identity-
# skip guard excludes the receiver's own entry from replay while still
# replaying every OTHER live binding correctly.
#
# These tests use a REAL, payload-carrying error class (not a zero-payload
# marker) so a double-free is observable: a corrupted/freed object's own field
# would no longer read back the value it was constructed with, and heap
# corruption from a genuine double-free is likely to crash outright under a
# real allocator (confirmed during development: the pre-fix code crashed with
# STATUS_ACCESS_VIOLATION on an equivalent repro).

import unittest

import test_support
from test_support import RealCompileMixin

_NAMED_VAR_OR_RETURN_ERR_PATH = '''
class ParseError:
	code: i32

def inner( bad: bool ) -> Result[i32, ParseError]:
	if bad:
		return Result.Err( ParseError( code = 42 ) )
	return Result.Ok( 7 )

def outer( bad: bool ) -> Result[i32, ParseError]:
	x: Result[i32, ParseError] = inner( bad )
	v: i32 = x.or_return()
	return Result.Ok( v )

def main() -> i32:
	r: Result[i32, ParseError] = outer( True )
	match r:
		case Result.Ok( v ):
			return 1
		case Result.Err( e ):
			if e.code != 42:
				return 2
	r2: Result[i32, ParseError] = outer( False )
	match r2:
		case Result.Ok( v ):
			if v != 7:
				return 3
		case Result.Err( e ):
			return 4
	return 0
'''

# a second, unrelated named local (y) is ALSO live across the .or_return()
# call - confirms the fix's exclusion of x's own entry doesn't also drop y's
# unrelated cleanup obligation (y must still be decref'd exactly once, on
# both the early-exit and the normal-fallthrough path)
_NAMED_VAR_OR_RETURN_WITH_OTHER_LIVE_BINDING = '''
class ParseError:
	code: i32

class OtherError:
	code: i32

def inner( bad: bool ) -> Result[i32, ParseError]:
	if bad:
		return Result.Err( ParseError( code = 42 ) )
	return Result.Ok( 7 )

def other_call() -> Result[i32, OtherError]:
	return Result.Ok( 99 )

def outer( bad: bool ) -> Result[i32, ParseError]:
	# y is inspected BEFORE x.or_return() runs, so it's satisfied on every
	# exit path (including x's own early-return) - but stays LIVE (is_err()
	# doesn't consume it), so it still needs its own cleanup wherever THIS
	# function actually exits, exactly like any other still-live binding
	y: Result[i32, OtherError] = other_call()
	if y.is_err():
		return Result.Err( ParseError( code = 0 ) )
	x: Result[i32, ParseError] = inner( bad )
	v: i32 = x.or_return()
	return Result.Ok( v )

def main() -> i32:
	r: Result[i32, ParseError] = outer( True )
	match r:
		case Result.Ok( v ):
			return 1
		case Result.Err( e ):
			if e.code != 42:
				return 2
	r2: Result[i32, ParseError] = outer( False )
	match r2:
		case Result.Ok( v ):
			if v != 7:
				return 3
		case Result.Err( e ):
			return 4
	return 0
'''

# repeat the propagation many times in a loop - a leaked/over-released
# refcount would compound across iterations, making it more likely to surface
# as an observable failure (wrong field value or crash) even without a
# sanitizer-enabled build
_NAMED_VAR_OR_RETURN_REPEATED = '''
class ParseError:
	code: i32

def inner( n: i32 ) -> Result[i32, ParseError]:
	if n == 7:
		return Result.Err( ParseError( code = n ) )
	return Result.Ok( n )

def outer( n: i32 ) -> Result[i32, ParseError]:
	x: Result[i32, ParseError] = inner( n )
	v: i32 = x.or_return()
	with compiler.panic_arithmetic( 'unreachable: v is always < 10' ):
		return Result.Ok( v * 2 )

def main() -> i32:
	with compiler.panic_arithmetic( 'unreachable: bounded loop counter' ):
		i: i32 = 0
		while i < 1000:
			n: i32 = i % 10
			r: Result[i32, ParseError] = outer( n )
			match r:
				case Result.Ok( v ):
					if n == 7:
						return 1
					if v != n * 2:
						return 2
				case Result.Err( e ):
					if n != 7:
						return 3
					if e.code != 7:
						return 4
			i += 1
	return 0
'''


@unittest.skipUnless( test_support.HAS_CC, 'no C compiler (clang/gcc/msvc) found - skipping real-compile RC tests' )
class OrReturnNamedVariableRCTests( RealCompileMixin, unittest.TestCase ):
	def test_named_variable_or_return_err_and_ok_paths( self ) -> None:
		self.assert_programs_run([ ( 'named_var_or_return_err_path', _NAMED_VAR_OR_RETURN_ERR_PATH ) ])

	def test_named_variable_or_return_with_other_live_binding( self ) -> None:
		self.assert_programs_run([ ( 'named_var_or_return_other_live', _NAMED_VAR_OR_RETURN_WITH_OTHER_LIVE_BINDING ) ])

	def test_named_variable_or_return_repeated( self ) -> None:
		self.assert_programs_run([ ( 'named_var_or_return_repeated', _NAMED_VAR_OR_RETURN_REPEATED ) ])


# --- bare `return x` widening (ir.WidenResult, lowering.py's
# _maybe_widen_return_result / _stmt_Return) ---------------------------------
#
# A bare `return x` where x: Result[T,NarrowE] inside a function declared
# -> Result[T,WideE] (WideE covering NarrowE) previously either silently
# compiled to INVALID C (a real C compiler rejects the mismatched struct
# return type outright - confirmed during development) or, after the general
# return-type check landed, was rejected as a compile error. This widens it
# instead, exactly like .or_return() already does, reusing the SAME
# _emit_widen_error payload-copy logic and the SAME current_epilogue_label()-
# based RC-exclusion mechanism the .or_return() fix above established (a
# named x's own scope-exit decref must still be excluded here, for the
# identical reason - confirmed via direct C inspection during development: a
# direct, un-gotoed `return` of the widened value, x's own epilogue label
# only reachable via the function's own separate fall-off-the-end path).

_WIDEN_RETURN_ERR_AND_OK_PATHS = '''
class ParseError:
	code: i32

def inner( fail: bool ) -> Result[bool, ParseError]:
	if fail:
		return Result.Err( ParseError( code = 77 ) )
	return Result.Ok( True )

def outer( fail: bool ) -> Result[bool, ParseError | ZeroDivisionError]:
	if fail:
		x: Result[bool, ParseError] = inner( True )
		return x
	return Result.Ok( False )

def main() -> i32:
	r_ok: Result[bool, ParseError | ZeroDivisionError] = outer( False )
	if r_ok.is_err():
		return 1
	r_err: Result[bool, ParseError | ZeroDivisionError] = outer( True )
	match r_err:
		case Result.Ok( v ):
			return 2
		case Result.Err( e ):
			err: ParseError | ZeroDivisionError = e
			match err:
				case ParseError( pe ):
					if pe.code != 77:
						return 3
				case ZeroDivisionError( ze ):
					return 4
	return 0
'''

# a second, unrelated named local (y) is ALSO live across x's widened
# `return x` - confirms the fix's exclusion of x's own entry doesn't also
# drop y's unrelated cleanup obligation, mirroring the equivalent
# .or_return()-side test above. y itself is ALSO widened-and-returned (a
# bare `return y`) on its own branch, exercising Stage 3 on a second,
# different narrow error type in the same function. (y's own branch
# deliberately returns y directly rather than constructing a fresh
# Result.Err(...) here - a leaf value needing implicit coercion into a
# union error type hits an unrelated, pre-existing emitter bug confirmed
# present even on unmodified, pre-this-session code - out of scope here)
_WIDEN_RETURN_WITH_OTHER_LIVE_BINDING = '''
class ParseError:
	code: i32

class OtherError:
	code: i32

def inner() -> Result[bool, ParseError]:
	return Result.Err( ParseError( code = 77 ) )

def other_call( fail: bool ) -> Result[bool, OtherError]:
	if fail:
		return Result.Err( OtherError( code = 88 ) )
	return Result.Ok( True )

def outer( other_fails: bool ) -> Result[bool, ParseError | OtherError]:
	y: Result[bool, OtherError] = other_call( other_fails )
	if y.is_err():
		return y
	x: Result[bool, ParseError] = inner()
	return x

def main() -> i32:
	r1: Result[bool, ParseError | OtherError] = outer( True )
	match r1:
		case Result.Ok( v ):
			return 1
		case Result.Err( e1 ):
			err1: ParseError | OtherError = e1
			match err1:
				case ParseError( pe1 ):
					return 2
				case OtherError( oe1 ):
					if oe1.code != 88:
						return 3

	r2: Result[bool, ParseError | OtherError] = outer( False )
	match r2:
		case Result.Ok( v ):
			return 4
		case Result.Err( e2 ):
			err2: ParseError | OtherError = e2
			match err2:
				case ParseError( pe2 ):
					if pe2.code != 77:
						return 5
				case OtherError( oe2 ):
					return 6
	return 0
'''


@unittest.skipUnless( test_support.HAS_CC, 'no C compiler (clang/gcc/msvc) found - skipping real-compile RC tests' )
class WidenReturnRCTests( RealCompileMixin, unittest.TestCase ):
	def test_widen_return_err_and_ok_paths( self ) -> None:
		self.assert_programs_run([ ( 'widen_return_err_and_ok', _WIDEN_RETURN_ERR_AND_OK_PATHS ) ])

	def test_widen_return_with_other_live_binding( self ) -> None:
		self.assert_programs_run([ ( 'widen_return_other_live', _WIDEN_RETURN_WITH_OTHER_LIVE_BINDING ) ])


# --- rc_leaves()/is_rc() union-leaf recursion (cfg.py) -----------------------
#
# rc_leaves()/is_rc() never recursed into a TaggedUnion that appears as a LEAF
# of an outer type - so Result[T, A|B] (any union-typed error, including the
# ZeroDivisionError|OverflowError this session's own division work
# introduced) was treated as having ZERO RC leaves, even when A/B carry real
# RC payloads. Consequence: no incref/decref ever fired for such a value's
# error side at all - aliasing one (`y = x`) never gave the alias its own
# independent reference. Fixed by teaching is_rc() to recurse into a
# TaggedUnion leaf's own members (cfg.py's only caller of is_rc() is
# rc_leaves() itself, so this has no separate blast radius), plus fixing
# _refcount_instructions' "every member shares one pointer layout, read any
# one's accessor" optimization to require every leaf be a genuine bare
# pointer (never a nested union, whose own runtime shape is a value struct,
# not a pointer) before taking that shortcut.
#
# Verified via a before/after compiler.refcount() delta around an aliasing
# assignment - independent of how many intermediate temps exist elsewhere in
# the propagation chain. Confirmed via a real A/B revert during development:
# the pre-fix code reports NO increment at all (before == after) across the
# alias; the fix reports exactly +1.

_UNION_LEAF_RC_ALIAS_DELTA = '''
class ParseError:
	code: i32

def inner() -> Result[bool, ParseError]:
	return Result.Err( ParseError( code = 42 ) )

def outer() -> Result[bool, ParseError | ZeroDivisionError]:
	x: Result[bool, ParseError] = inner()
	return x

def main() -> i32:
	with compiler.wrap_arithmetic:
		r: Result[bool, ParseError | ZeroDivisionError] = outer()
		match r:
			case Result.Ok( v ):
				return 1
			case Result.Err( e ):
				err: ParseError | ZeroDivisionError = e
				match err:
					case ParseError( pe ):
						before: usize = compiler.refcount( pe )
						# s = r is a plain aliasing assignment of a Result
						# whose error side is a union with a real RC leaf -
						# EXACTLY the shape rc_leaves() previously reported as
						# having no RC leaves at all, silently skipping the
						# incref this assignment needs
						s: Result[bool, ParseError | ZeroDivisionError] = r
						if s.is_err():
							pass
						after: usize = compiler.refcount( pe )
						if after != before + 1:
							return 2
					case ZeroDivisionError( ze ):
						return 3
	return 0
'''


@unittest.skipUnless( test_support.HAS_CC, 'no C compiler (clang/gcc/msvc) found - skipping real-compile RC tests' )
class UnionLeafRCTests( RealCompileMixin, unittest.TestCase ):
	def test_union_leaf_rc_alias_increfs( self ) -> None:
		self.assert_programs_run([ ( 'union_leaf_rc_alias_delta', _UNION_LEAF_RC_ALIAS_DELTA ) ])


# --- fresh-construction-into-inferred-union crash (lowering.py's
# _lower_allocate_fields) ----------------------------------------------------
#
# Result.Err( ParseError( code = 42 ) ) - a FRESH construction of a plain leaf
# class, passed directly as the argument to a call whose own type parameter
# (E) is inferred from the surrounding declared return type to a multi-leaf
# anonymous union (ParseError|OtherError) - used to crash the emitter with a
# bare AssertionError. Root cause: _lower_allocate_fields blindly trusted
# expected_type (hinted from the substituted, unified E, i.e. the UNION) as
# the constructed temp's own type, producing a self-inconsistent ir.Allocate
# (cls=ParseError, dest.type=the union) that also silently skipped
# _lower_expr's own union-coercion check (operand.type was, by accident,
# ALREADY identical to expected_type). Confirmed present on a fully unmodified
# checkout via git stash - pre-existing, not introduced by any of this
# session's own work. Fixed by only trusting expected_type as dest's type when
# it's actually rooted at target_cls (itself, or a Specialization of it);
# otherwise dest falls back to target_cls, so the real mismatch survives back
# in _lower_expr and correctly triggers _coerce_into_union instead.
#
# Both directions matter: pre-binding the leaf to a local (`e = ParseError(...);
# return Result.Err(e)`) already worked (never hits _lower_allocate_fields with
# a mismatched expected_type), so this test specifically keeps the FRESH
# construction expression inline as the call argument.

_FRESH_CONSTRUCTION_INTO_INFERRED_UNION = '''
class ParseError:
	code: i32

class OtherError:
	code: i32

def outer( fail: bool ) -> Result[bool, ParseError | OtherError]:
	if fail:
		return Result.Err( ParseError( code = 42 ) )
	return Result.Ok( True )

def main() -> i32:
	r_ok: Result[bool, ParseError | OtherError] = outer( False )
	if r_ok.is_err():
		return 1
	r_err: Result[bool, ParseError | OtherError] = outer( True )
	match r_err:
		case Result.Ok( v ):
			return 2
		case Result.Err( e ):
			err: ParseError | OtherError = e
			match err:
				case ParseError( pe ):
					if pe.code != 42:
						return 3
				case OtherError( oe ):
					return 4
	return 0
'''


@unittest.skipUnless( test_support.HAS_CC, 'no C compiler (clang/gcc/msvc) found - skipping real-compile RC tests' )
class FreshConstructionIntoInferredUnionTests( RealCompileMixin, unittest.TestCase ):
	def test_fresh_construction_into_inferred_union( self ) -> None:
		self.assert_programs_run([ ( 'fresh_construction_into_inferred_union', _FRESH_CONSTRUCTION_INTO_INFERRED_UNION ) ])


if __name__ == '__main__':
	unittest.main()
