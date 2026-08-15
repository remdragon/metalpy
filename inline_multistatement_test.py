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


if __name__ == '__main__':
	unittest.main()
