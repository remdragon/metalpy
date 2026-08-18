# Real-compile-and-run behavioral test for the @fallible_arithmetic decorator +
# mode-qualified dunder resolution in binop dispatch (lowering.py's
# _mode_qualified_dunder_names/_emit_binop_dunder_call). Proves the
# mechanism end to end on its one proof-of-concept slice: i32's own
# __add__/__wrapped_add__/__saturated_add__ (lib/builtins/__int32.py) -
# each is @inline, so binop dispatch resolving one of these must produce
# byte-identical behavior to the pre-existing direct-opcode path, with
# genuinely zero call overhead (no real i32__*_add__i32 function ever
# compiled - splicing only). Uses the shared test_support.RealCompileMixin
# harness (no copy-pasted compile+link+run).

from pathlib import Path
import unittest

from compiler import Compiler
from discovery import Discovery
import emitter_c
import test_support
from test_support import RealCompileMixin

# checked (default)/wrap/saturate all exercised in ONE program - each
# mode's own `with` block scopes independently, mirroring how a bare
# scalar `+` already behaved before this dunder mechanism existed.
_I32_ADD_MODES_BEHAVIOR = '''
def add_it( a: i32, b: i32 ) -> Result[i32,OverflowError]:
	# default (checked) mode: no `with` block - auto-propagates via the
	# enclosing function's own Result[_,OverflowError] return type, exactly
	# like a bare checked `+` always required
	c = a + b
	return Result.Ok( c )

def main() -> i32:
	maxv: i32 = 2147483647 # i32::MAX

	# checked mode, no overflow: ordinary Result.Ok
	ok = add_it( 5, 7 )
	if ok.is_err() or ok.unwrap( 'x' ) != 12:
		return 1

	# checked mode, real overflow: auto-propagated Err, not a panic/crash
	overflowed = add_it( maxv, 1 )
	if overflowed.is_ok():
		return 2

	with compiler.wrap_arithmetic:
		w = maxv + 1
	if w != -2147483648:
		return 3

	with compiler.saturate_arithmetic:
		s = maxv + 1
	if s != 2147483647:
		return 4

	# no overflow at all - every mode must agree
	with compiler.wrap_arithmetic:
		wn = 5 + 7
	with compiler.saturate_arithmetic:
		sn = 5 + 7
	if wn != 12 or sn != 12:
		return 5

	return 0
'''

# separate program: panic_arithmetic must actually panic (nonzero exit) on
# real overflow, not silently succeed or propagate a Result
_I32_ADD_PANIC_BEHAVIOR = '''
def main() -> i32:
	maxv: i32 = 2147483647
	with compiler.panic_arithmetic( 'overflow!' ):
		x = maxv + 1
	return 0
'''


@unittest.skipUnless( test_support.HAS_CC, 'no C compiler (clang/gcc/msvc) found - skipping real-compile checked-dunder tests' )
class CheckedDunderBehaviorTests( RealCompileMixin, unittest.TestCase ):
	def test_i32_add_modes_produce_correct_values_and_compile_no_real_dunder_function( self ) -> None:
		compiler = self._compile_source( _I32_ADD_MODES_BEHAVIOR )
		c_source = emitter_c.emit_c( compiler )
		self._assert_compiles_and_runs( c_source, expected_exit = 0, compiler = compiler )
		qualnames = { lf.function.qualname for lf in compiler.functions }
		self.assertFalse(
			any( 'i32__add__i32' in q or 'i32__wrapped_add__i32' in q or 'i32__saturated_add__i32' in q for q in qualnames ),
			f'a real i32 dunder-add function was compiled, @inline should have spliced it instead: {sorted(qualnames)}',
		)

	def test_i32_add_panics_under_panic_arithmetic_on_real_overflow( self ) -> None:
		compiler = self._compile_source( _I32_ADD_PANIC_BEHAVIOR )
		c_source = emitter_c.emit_c( compiler )
		self._assert_compiles_and_runs( c_source, expected_exit = 1, compiler = compiler )


# _emit_binop_dunder_call/_emit_fallible_method_call (the @fallible_arithmetic
# dispatch path for a real class's own dunder - e.g. int.__floordiv__/__mod__
# via `//`/`%`, or a Scalar-registered dunder like i32.__add__) used to skip
# the _require_result_return validation _lower_arithmetic_op's own plain
# scalar Check-mode opcodes already perform. That let a program whose checked
# arithmetic error can't propagate anywhere (the enclosing function doesn't
# return a covering Result[_,_]) reach emitter_c.py with an invalid OrReturn/
# OrJump - a Python AssertionError (_result_error_type: "not a Result[T,E]")
# instead of a clean compile error. Confirmed via a real repro: `r: int = a
# // b` inside a function declared `-> i32` crashed mpy.py entirely.
class FallibleArithmeticBinopValidationTests( unittest.TestCase ):
	def test_checked_floordiv_without_result_return_fails_cleanly_not_a_crash( self ) -> None:
		discovery = Discovery( import_builtins = True )
		compiler = Compiler( discovery )
		compiler.import_code( '\n'.join([
			'def main() -> i32:',
			'	a: int = int( 10 )',
			'	b: int = int( 3 )',
			'	r: int = a // b',
			'	return 0',
		]), Path( '__main__.py' ), scope = None )
		compiler.run() # must not raise (the crash this guards against was a real, uncaught AssertionError)
		self.assertTrue(
			any( 'requires the enclosing function to return Result' in e for e in discovery.errors.errors ),
			f'expected a clean compile error, got: {discovery.errors.errors}',
		)

	def test_checked_floordiv_with_result_return_still_compiles( self ) -> None:
		# the valid counterpart - guards against an overzealous fix rejecting
		# the exact shape it's meant to keep accepting
		discovery = Discovery( import_builtins = True )
		compiler = Compiler( discovery )
		compiler.import_code( '\n'.join([
			'def divide( a: int, b: int ) -> Result[i32, IntError | ZeroDivisionError]:',
			'	q: int = a // b',
			'	return Result.Ok( 0 )',
			'',
			'def main() -> i32:',
			'	r = divide( int( 10 ), int( 3 ))',
			'	return 1 if r.is_err() else 0',
		]), Path( '__main__.py' ), scope = None )
		compiler.run()
		self.assertEqual( discovery.errors.errors, [], f'unexpected compile errors: {discovery.errors.errors}' )


if __name__ == '__main__':
	unittest.main()
