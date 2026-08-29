# Real-compile-and-run tests for lib/math.py.

# stdlib imports:
import unittest

# local imports:
import test_support


class MathTests( test_support.RealCompileMixin, unittest.TestCase ):
	def setUp( self ) -> None:
		from discovery import Discovery
		from compiler import Compiler
		self.discovery = Discovery( import_builtins = True )
		self.compiler = Compiler( self.discovery )

	@unittest.skipUnless( test_support.HAS_CC, 'no C compiler (clang/gcc/msvc) found - skipping' )
	def test_sqrt( self ) -> None:
		import emitter_c
		self.assert_programs_run([
			( 'f64_sqrt', '''
import compiler
import math

def main() -> i32:
	if math.sqrt( 4.0 ) != 2.0:
		return 1
	if math.sqrt( 2.25 ) != 1.5:
		return 2
	if math.sqrt( 0.0 ) != 0.0:
		return 3
	if not compiler.is_nan( math.sqrt( -1.0 ) ):
		return 4
	return 0
''' ),
			( 'f32_sqrtf', '''
import compiler
import math

def main() -> i32:
	if math.sqrtf( f32( 4.0 ) ) != f32( 2.0 ):
		return 1
	if math.sqrtf( f32( 2.25 ) ) != f32( 1.5 ):
		return 2
	if not compiler.is_nan( math.sqrtf( f32( -1.0 ) ) ):
		return 3
	return 0
''' ),
		])


if __name__ == '__main__':
	unittest.main()
