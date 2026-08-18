# Real-compile-and-run behavioral test for x.to_T() scalar conversions
# (lib/builtins/__scalar_arith.py's i_to_i[S,T]/i_to_i_identity[T], backed
# by compiler.checked_convert/ir.ConvertCheck - see emitter_c.py's
# _emit_convert_check). Proves .to_T() is a genuinely different operation
# from T(x) construct-cast syntax: a value-range check against the target's
# [MIN,MAX], independent of bit width - it can fail even for a same-width
# conversion (i8(-1).to_u8()) that T(x) (u8(i8(-1))) never does. Uses the
# shared test_support.RealCompileMixin harness (no copy-pasted compile+
# link+run).

import unittest

import emitter_c
import test_support
from test_support import RealCompileMixin

_TO_T_MATRIX = '''
def main() -> i32:
	# identity: always succeeds
	a: i32 = 42
	r1: Result[i32,OverflowError] = a.to_i32()
	if r1.is_err() or r1.unwrap( 'r1' ) != 42:
		return 1

	# same-width cross-sign, in range
	b: i8 = 100
	r2: Result[u8,OverflowError] = b.to_u8()
	if r2.is_err() or r2.unwrap( 'r2' ) != 100:
		return 2

	# same-width cross-sign, out of range (negative -> unsigned) - the
	# case that most directly demonstrates the T(x)/.to_T() split: T(x)
	# (u8(i8(-1))) never fails here, .to_T() always does
	c: i8 = -1
	r3: Result[u8,OverflowError] = c.to_u8()
	if not r3.is_err():
		return 3

	# narrowing, in range
	d: i32 = 200
	r4: Result[u8,OverflowError] = d.to_u8()
	if r4.is_err() or r4.unwrap( 'r4' ) != 200:
		return 4

	# narrowing, out of range
	e: i32 = 1000
	r5: Result[u8,OverflowError] = e.to_u8()
	if not r5.is_err():
		return 5

	# widening, unsigned source: always in range
	f: u8 = 200
	r6: Result[i32,OverflowError] = f.to_i32()
	if r6.is_err() or r6.unwrap( 'r6' ) != 200:
		return 6

	# widening WITH a signedness change that can still overflow - T(x)
	# (u16(i8(-1))) always succeeds here (pure widening), .to_T() doesn't
	g: i8 = -1
	r7: Result[u16,OverflowError] = g.to_u16()
	if not r7.is_err():
		return 7

	# widening, same signedness: always succeeds, negative value preserved
	h: i8 = -5
	r8: Result[i32,OverflowError] = h.to_i32()
	if r8.is_err() or r8.unwrap( 'r8' ) != -5:
		return 8

	return 0
'''

# ambient arithmetic mode only affects HOW the Result is consumed at a
# binop-style dispatch site, not .to_T()'s own check - but .to_T() reached
# via ordinary method-call syntax always hands back a raw Result regardless
# of mode (see _emit_fallible_method_call's is_inline/non-inline split),
# so this just confirms the check itself is identical under every mode
_TO_T_MODE_INDEPENDENCE = '''
def check_it( x: i8 ) -> Result[u8,OverflowError]:
	return x.to_u8()

def main() -> i32:
	neg: i8 = -1
	with compiler.wrap_arithmetic:
		if not check_it( neg ).is_err():
			return 1
	with compiler.saturate_arithmetic:
		if not check_it( neg ).is_err():
			return 2
	with compiler.panic_arithmetic( 'unexpected' ):
		if not check_it( neg ).is_err():
			return 3
	return 0
'''


@unittest.skipUnless( test_support.HAS_CC, 'no C compiler (clang/gcc/msvc) found - skipping real-compile scalar-convert tests' )
class ScalarConvertBehaviorTests( RealCompileMixin, unittest.TestCase ):
	def test_to_t_correctness_matrix( self ) -> None:
		compiler = self._compile_source( _TO_T_MATRIX )
		c_source = emitter_c.emit_c( compiler )
		self._assert_compiles_and_runs( c_source, expected_exit = 0, compiler = compiler )
		# zero-overhead, matching every other @inline dunder in this file -
		# no real i_to_i/i_to_i_identity function should ever be compiled
		qualnames = { lf.function.qualname for lf in compiler.functions }
		self.assertFalse(
			any( 'i_to_i' in q for q in qualnames ),
			f'a real .to_T() conversion function was compiled, @inline should have spliced it instead: {sorted(qualnames)}',
		)

	def test_to_t_check_is_mode_independent( self ) -> None:
		compiler = self._compile_source( _TO_T_MODE_INDEPENDENCE )
		c_source = emitter_c.emit_c( compiler )
		self._assert_compiles_and_runs( c_source, expected_exit = 0, compiler = compiler )


if __name__ == '__main__':
	unittest.main()
