import unittest

import test_support
from compiler import Compiler
from discovery import Discovery


class CsvLineReaderTests( test_support.RealCompileMixin, unittest.TestCase ):
	''' PLAN csv.py, Step 3 - LineReader is the first chunked-buffered
	text-line-splitting layer in lib/ (only raw binary I/O existed before).
	Writes a real temp file via File.binary_writer, reads it back through
	LineReader. Each case's compiled exe runs with the worktree root as its
	cwd (inherited from the test-runner process), so relative filenames land
	there - each case uses its own filename to avoid collisions when the
	merged executable runs all cases in one process. '''

	def setUp( self ) -> None:
		self.discovery = Discovery( import_builtins = True )
		self.compiler = Compiler( self.discovery )

	@unittest.skipUnless( test_support.HAS_CC, 'no C compiler (clang/gcc/msvc) found - skipping' )
	def test_programs_compile_and_run( self ) -> None:
		self.assert_programs_run([
			( 'lf_only_lines', '''
import csv

def write_file( path: str, content: str ) -> None:
	match File.binary_writer( path ):
		case Result.Ok( w ):
			w.write( content.get_const_ptr(), content.byte_len() ).unwrap( 'write' )
			w.close()
		case Result.Err( e ):
			sys.panic( 'could not open for write' )

def main() -> i32:
	path: str = 'csv_lr_case1.tmp'
	write_file( path, 'a\\nbb\\nccc\\n' )
	r = File.binary_reader( path ).unwrap( 'open' )
	lr = csv.LineReader( r )

	m1 = lr.next_line().unwrap( 'l1' )
	if not m1.has_line:
		return 1
	if m1.line != 'a':
		return 2

	m2 = lr.next_line().unwrap( 'l2' )
	if not m2.has_line:
		return 3
	if m2.line != 'bb':
		return 4

	m3 = lr.next_line().unwrap( 'l3' )
	if not m3.has_line:
		return 5
	if m3.line != 'ccc':
		return 6

	m4 = lr.next_line().unwrap( 'l4' )
	if m4.has_line:
		return 7
	return 0
''' ),
			( 'crlf_lines', '''
import csv

def write_file( path: str, content: str ) -> None:
	match File.binary_writer( path ):
		case Result.Ok( w ):
			w.write( content.get_const_ptr(), content.byte_len() ).unwrap( 'write' )
			w.close()
		case Result.Err( e ):
			sys.panic( 'could not open for write' )

def main() -> i32:
	path: str = 'csv_lr_case2.tmp'
	write_file( path, 'x\\r\\nyy\\r\\n' )
	r = File.binary_reader( path ).unwrap( 'open' )
	lr = csv.LineReader( r )

	m1 = lr.next_line().unwrap( 'l1' )
	if not m1.has_line:
		return 1
	if m1.line != 'x':
		return 2

	m2 = lr.next_line().unwrap( 'l2' )
	if not m2.has_line:
		return 3
	if m2.line != 'yy':
		return 4

	m3 = lr.next_line().unwrap( 'l3' )
	if m3.has_line:
		return 5
	return 0
''' ),
			( 'no_trailing_newline', '''
import csv

def write_file( path: str, content: str ) -> None:
	match File.binary_writer( path ):
		case Result.Ok( w ):
			w.write( content.get_const_ptr(), content.byte_len() ).unwrap( 'write' )
			w.close()
		case Result.Err( e ):
			sys.panic( 'could not open for write' )

def main() -> i32:
	path: str = 'csv_lr_case3.tmp'
	write_file( path, 'onlyline' )
	r = File.binary_reader( path ).unwrap( 'open' )
	lr = csv.LineReader( r )

	m1 = lr.next_line().unwrap( 'l1' )
	if not m1.has_line:
		return 1
	if m1.line != 'onlyline':
		return 2

	m2 = lr.next_line().unwrap( 'l2' )
	if m2.has_line:
		return 3
	return 0
''' ),
			( 'empty_file', '''
import csv

def write_file( path: str, content: str ) -> None:
	match File.binary_writer( path ):
		case Result.Ok( w ):
			w.write( content.get_const_ptr(), content.byte_len() ).unwrap( 'write' )
			w.close()
		case Result.Err( e ):
			sys.panic( 'could not open for write' )

def main() -> i32:
	path: str = 'csv_lr_case4.tmp'
	write_file( path, '' )
	r = File.binary_reader( path ).unwrap( 'open' )
	lr = csv.LineReader( r )

	m1 = lr.next_line().unwrap( 'l1' )
	if m1.has_line:
		return 1
	return 0
''' ),
			( 'line_longer_than_initial_buffer_forces_growth_and_refill', '''
import csv

def write_file( path: str, content: str ) -> None:
	match File.binary_writer( path ):
		case Result.Ok( w ):
			w.write( content.get_const_ptr(), content.byte_len() ).unwrap( 'write' )
			w.close()
		case Result.Err( e ):
			sys.panic( 'could not open for write' )

def main() -> i32:
	long_line: str = ''
	i: usize = 0
	with compiler.wrap_arithmetic:
		while i < 5000:
			long_line = long_line + 'x'
			i += 1
	path: str = 'csv_lr_case5.tmp'
	write_file( path, long_line + '\\ntail\\n' )
	r = File.binary_reader( path ).unwrap( 'open' )
	lr = csv.LineReader( r )

	m1 = lr.next_line().unwrap( 'l1' )
	if not m1.has_line:
		return 1
	if m1.line != long_line:
		return 2
	if m1.line.byte_len() != 5000:
		return 3

	m2 = lr.next_line().unwrap( 'l2' )
	if not m2.has_line:
		return 4
	if m2.line != 'tail':
		return 5

	m3 = lr.next_line().unwrap( 'l3' )
	if m3.has_line:
		return 6
	return 0
''' ),
		])


if __name__ == '__main__':
	unittest.main()
