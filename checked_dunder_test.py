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

import unittest

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
		# i32.__add__/__wrapped_add__/__saturated_add__ are now specializations
		# of shared generic bodies (i_add_checked[T]/i_add_wrapped[T]/
		# i_add_saturated[T] - lib/builtins/__scalar_arith.py), not per-type
		# functions named i32__add__i32 anymore - check for the GENERIC
		# stem instead, so this assertion still means something (checking
		# for the old, now-nonexistent name would trivially always pass)
		qualnames = { lf.function.qualname for lf in compiler.functions }
		self.assertFalse(
			any( 'i_add_checked' in q or 'i_add_wrapped' in q or 'i_add_saturated' in q for q in qualnames ),
			f'a real i32 dunder-add function was compiled, @inline should have spliced it instead: {sorted(qualnames)}',
		)

	def test_i32_add_panics_under_panic_arithmetic_on_real_overflow( self ) -> None:
		compiler = self._compile_source( _I32_ADD_PANIC_BEHAVIOR )
		c_source = emitter_c.emit_c( compiler )
		self._assert_compiles_and_runs( c_source, expected_exit = 1, compiler = compiler )


if __name__ == '__main__':
	unittest.main()
