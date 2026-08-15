# Real-compile-and-run behavioral test for multi-statement @inline bodies
# (the generalization of PLAN_INLINE.md - see its own STATUS section).
# lowering_test.py's InlineMultiStatementTests already cover the IR shape
# in isolation; this confirms the emitted C is actually correct end to
# end: a multi-statement inline body (locals, a nested if, a reassignment)
# produces the right runtime value, and no separate C function is ever
# emitted for the inlined target itself. Uses the shared test_support.
# RealCompileMixin harness (no copy-pasted compile+link+run).

import unittest

import emitter_c
import test_support
from test_support import RealCompileMixin

_MULTISTATEMENT_INLINE_BEHAVIOR = '''
@cstruct
class Counter:
	value: i32

	@inline
	def bumped( self, by: i32 ) -> i32:
		result: i32 = self.value
		with compiler.wrap_arithmetic:
			result = result + by
		if result < 0:
			result = 0
		return result

def main() -> i32:
	c: Counter = Counter( value = 10 )
	result: i32 = 100                 # deliberately shares a name with bumped()'s own local
	bumped_value: i32 = c.bumped( 5 )
	if bumped_value != 15:
		return 1
	if result != 100:                 # main's own `result` must be untouched by the splice
		return 2

	negative: Counter = Counter( value = -20 )
	clamped: i32 = negative.bumped( 5 )
	if clamped != 0:                  # exercises the nested if/reassignment path
		return 3

	return 0
'''


@unittest.skipUnless( test_support.HAS_CC, 'no C compiler (clang/gcc/msvc) found - skipping real-compile multi-statement inline tests' )
class InlineMultiStatementBehaviorTests( RealCompileMixin, unittest.TestCase ):
	def test_multistatement_inline_runs_correctly_and_compiles_no_separate_function( self ) -> None:
		compiler = self._compile_source( _MULTISTATEMENT_INLINE_BEHAVIOR )
		c_source = emitter_c.emit_c( compiler )
		self._assert_compiles_and_runs( c_source, expected_exit = 0, compiler = compiler )
		qualnames = { lf.function.qualname for lf in compiler.functions }
		self.assertFalse(
			any( 'bumped' in q for q in qualnames ),
			f'a real bumped() function was compiled, @inline should have spliced it instead: {sorted(qualnames)}',
		)


# Early-return/.or_return() generalization: the "notably important" runtime
# proof, not just an IR-shape assertion (lowering_test.py's InlineMultiStatement
# Tests.test_or_return_in_pre_return_statement_now_works already covers the
# IR shape) - an @inline method whose pre-return statement early-exits via
# .or_return() on an Err receiver must NOT trigger a return from the CALLING
# function: the caller's own code after the call site must still run, and
# the caller must correctly observe the propagated Err.
_OR_RETURN_INLINE_EARLY_EXIT_BEHAVIOR = '''
class MyError:
	pass

@union
class Result[T,E]:
	Ok: T
	Err: E

	def is_err( self ) -> bool:
		return self.tag == 1

@cstruct
class Widget:
	fail: bool

	def risky( self ) -> Result[i32,MyError]:
		if self.fail:
			return Result.Err( MyError() )
		return Result.Ok( 42 )

	@inline
	def doubled( self ) -> Result[i32,MyError]:
		x: i32 = self.risky().or_return()
		with compiler.wrap_arithmetic:
			x = x * 2
		return Result.Ok( x )

def main() -> i32:
	ok_widget: Widget = Widget( fail = False )
	r1: Result[i32,MyError] = ok_widget.doubled()
	if r1.is_err():
		return 1

	bad_widget: Widget = Widget( fail = True )
	after_call_ran: bool = False
	r2: Result[i32,MyError] = bad_widget.doubled()
	# reached only if doubled()'s own internal or_return() did NOT return
	# from main() itself - the exact regression this test guards against
	after_call_ran = True
	r2_is_err: bool = r2.is_err() # inspected before any return, so the compiler's own unchecked-Result discipline is satisfied on every exit path
	if not after_call_ran:
		return 2
	if not r2_is_err:
		return 3

	return 0
'''


@unittest.skipUnless( test_support.HAS_CC, 'no C compiler (clang/gcc/msvc) found - skipping real-compile multi-statement inline tests' )
class InlineOrReturnEarlyExitBehaviorTests( RealCompileMixin, unittest.TestCase ):
	def test_or_return_in_inline_body_does_not_return_from_caller( self ) -> None:
		compiler = self._compile_source( _OR_RETURN_INLINE_EARLY_EXIT_BEHAVIOR )
		c_source = emitter_c.emit_c( compiler )
		self._assert_compiles_and_runs( c_source, expected_exit = 0, compiler = compiler )
		qualnames = { lf.function.qualname for lf in compiler.functions }
		self.assertFalse(
			any( 'doubled' in q for q in qualnames ),
			f'a real doubled() function was compiled, @inline should have spliced it instead: {sorted(qualnames)}',
		)


if __name__ == '__main__':
	unittest.main()
