import unittest

import test_support
from compiler import Compiler
from discovery import Discovery


class CsvEndToEndTests( test_support.RealCompileMixin, unittest.TestCase ):
	''' PLAN csv.py verification section: write a multi-row CSV (quoted
	field with embedded delimiter, embedded quote, and embedded newline)
	via writer(), read it back via reader(), assert the round-tripped rows
	exactly match the original data. '''

	def setUp( self ) -> None:
		self.discovery = Discovery( import_builtins = True )
		self.compiler = Compiler( self.discovery )

	@unittest.skipUnless( test_support.HAS_CC, 'no C compiler (clang/gcc/msvc) found - skipping' )
	def test_programs_compile_and_run( self ) -> None:
		self.assert_programs_run([
			( 'full_round_trip_all_features_together', '''
import csv

def row_of( a: str, b: str, c: str ) -> list[str]:
	r: list[str] = list[str]()
	r.append( a ).unwrap( 'x' )
	r.append( b ).unwrap( 'x' )
	r.append( c ).unwrap( 'x' )
	return r

def rows_equal( a: list[str], b: list[str] ) -> bool:
	if a.__len__() != b.__len__():
		return False
	n: usize = a.__len__()
	i: usize = 0
	while i < n:
		if a.__getitem__( i ).unwrap( 'x' ) != b.__getitem__( i ).unwrap( 'x' ):
			return False
		with compiler.wrap_arithmetic:
			i += 1
	return True

def main() -> i32:
	path: str = 'csv_e2e_case1.tmp'

	original: list[list[str]] = list[list[str]]()
	original.append( row_of( 'plain', 'fields', 'here' )).unwrap( 'x' )
	original.append( row_of( 'has,comma', 'plain', 'plain' )).unwrap( 'x' )
	original.append( row_of( 'has"quote', 'plain', 'plain' )).unwrap( 'x' )
	original.append( row_of( 'multi\\nline\\nfield', 'plain', 'plain' )).unwrap( 'x' )
	original.append( row_of( '', '', 'trailing empty above' )).unwrap( 'x' )
	original.append( row_of( 'combo: a,b"c\\nd', 'plain', 'plain' )).unwrap( 'x' )

	w = csv.writer( path ).unwrap( 'open for write' )
	w.writerows( original ).unwrap( 'writerows' )
	w.close()

	r = csv.reader( path ).unwrap( 'open for read' )

	row_count: usize = 0
	n: usize = original.__len__()
	while row_count < n:
		expected: list[str] = original.__getitem__( row_count ).unwrap( 'x' )
		match r.__next__():
			case Result.Ok( m ):
				if not m.has_row:
					return 1
				if not rows_equal( m.row, expected ):
					with compiler.wrap_arithmetic:
						return 100 + i32( row_count )
			case Result.Err( e ):
				return 2
		with compiler.wrap_arithmetic:
			row_count += 1

	final = r.__next__().unwrap( 'final' )
	if final.has_row:
		return 3
	return 0
''' ),
		])


if __name__ == '__main__':
	unittest.main()
