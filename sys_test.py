# Real-compile-and-run test for lib/sys.py's cpu_count() - the one piece
# of lib/sys.py without any other coverage today (alloc/free/cstrlen are
# all exercised indirectly by everything else that compiles). Can't assert
# an exact value (machine-dependent), so this only checks the "never fails,
# always >= 1" contract cpu_count()'s own docstring promises.

import unittest

import test_support


class SysTests( test_support.RealCompileMixin, unittest.TestCase ):
	def setUp( self ) -> None:
		from discovery import Discovery
		from compiler import Compiler
		self.discovery = Discovery( import_builtins = True )
		self.compiler = Compiler( self.discovery )

	@unittest.skipUnless( test_support.HAS_CC, 'no C compiler (clang/gcc/msvc) found - skipping real-compile sys tests' )
	def test_programs_compile_and_run( self ) -> None:
		self.assert_programs_run([
			( 'cpu_count_is_at_least_one', '''
import sys

def main() -> i32:
	n: u32 = sys.cpu_count()
	if n < 1:
		return 1
	return 0
''' ),
		])


if __name__ == '__main__':
	unittest.main()
