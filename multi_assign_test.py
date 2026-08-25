import unittest

import test_support
from compiler import Compiler
from discovery import Discovery
from test_support import RealCompileMixin


class MultiAssignTests( RealCompileMixin, unittest.TestCase ):
	''' `a = b = c = value` (Python's chained-assignment syntax) - real
	compile-and-run coverage. discovery.py's own visit_Assign handles the
	module/class-attribute case (splits into independent single-target
	declarations, no runtime sequencing needed); type_resolver.py's own
	_ReferenceResolver.visit_Assign handles the function-body case (a real
	synthetic-temp desugar, since the RHS must be evaluated exactly once). '''

	def setUp( self ) -> None:
		self.discovery = Discovery( import_builtins = True )
		self.compiler = Compiler( self.discovery )

	@unittest.skipUnless( test_support.HAS_CC, 'no C compiler (clang/gcc/msvc) found - skipping' )
	def test_programs_compile_and_run( self ) -> None:
		self.assert_programs_run([
			( 'local_chained_assign', '''
def main() -> i32:
	eol1: i32 = 0
	eol2: i32 = 0
	mv_len: i32 = 42
	eol1 = eol2 = mv_len
	if eol1 != 42:
		return 1
	if eol2 != 42:
		return 2
	return 0
''' ),
			( 'local_chained_assign_first_declaration', '''
def main() -> i32:
	a = b = c = 7
	if a != 7 or b != 7 or c != 7:
		return 1
	return 0
''' ),
			( 'class_attribute_chained_assign', '''
class Grap:
	option1 = option2 = option3 = False

def main() -> i32:
	g = Grap()
	if g.option1 != False:
		return 1
	if g.option2 != False:
		return 2
	if g.option3 != False:
		return 3
	return 0
''' ),
			( 'chained_assign_rhs_evaluated_once', '''
class Counter:
	calls: i32

	def __init__( self ) -> None:
		self.calls = 0

	def bump( self ) -> i32:
		with compiler.wrap_arithmetic:
			self.calls = self.calls + 1
		return self.calls

def main() -> i32:
	c = Counter()
	x: i32 = 0
	y: i32 = 0
	x = y = c.bump()
	if x != 1 or y != 1:
		return 1
	if c.calls != 1: # a naive duplicate-the-RHS desugar would call bump() twice
		return 2
	return 0
''' ),
			( 'chained_assign_reassignment_of_existing_locals', '''
def main() -> i32:
	x: i32 = 1
	y: i32 = 2
	x = y = 99
	if x != 99 or y != 99:
		return 1
	return 0
''' ),
		] )


if __name__ == '__main__':
	unittest.main()
