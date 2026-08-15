import unittest

import test_support
from compiler import Compiler
from discovery import Discovery


class CsvReaderTests( test_support.RealCompileMixin, unittest.TestCase ):
	''' PLAN csv.py, Step 4 - Reader/reader() over a real file, combining
	_LineReader + RowParser. Each case writes its own scratch file first
	(compiled exe's cwd is the worktree root, inherited from the
	test-runner process) then reads it back via csv.reader(). '''

	def setUp( self ) -> None:
		self.discovery = Discovery( import_builtins = True )
		self.compiler = Compiler( self.discovery )

	@unittest.skipUnless( test_support.HAS_CC, 'no C compiler (clang/gcc/msvc) found - skipping' )
	def test_programs_compile_and_run( self ) -> None:
		self.assert_programs_run([
			( 'multi_row_file_round_trip', '''
import csv

def write_file( path: str, content: str ) -> None:
	match File.binary_writer( path ):
		case Result.Ok( w ):
			w.write( content.get_const_ptr(), content.byte_len() ).unwrap( 'write' )
			w.close()
		case Result.Err( e ):
			sys.panic( 'could not open for write' )

def main() -> i32:
	path: str = 'csv_reader_case1.tmp'
	write_file( path, 'a,b,c\\n1,2,3\\n' )

	r: csv.Reader
	match csv.reader( path ):
		case Result.Ok( rr ):
			r = rr
		case Result.Err( e ):
			return 20

	row_count: usize = 0
	last_ok: bool = False
	while True:
		got: bool = False
		row: list[str] = list[str]()
		is_err: bool = False
		match r.__next__():
			case Result.Ok( m ):
				if m.has_row:
					got = True
					row = m.row
			case Result.Err( e2 ):
				is_err = True
		if is_err:
			return 21
		if not got:
			break
		if row_count == 0:
			if row.__len__() != 3:
				return 1
			if row.__getitem__( 0 ).unwrap( 'x' ) != 'a':
				return 2
		if row_count == 1:
			if row.__len__() != 3:
				return 3
			if row.__getitem__( 0 ).unwrap( 'x' ) != '1':
				return 4
			if row.__getitem__( 2 ).unwrap( 'x' ) != '3':
				return 5
			last_ok = True
		with compiler.wrap_arithmetic:
			row_count += 1

	if row_count != 2:
		return 6
	if not last_ok:
		return 7
	return 0
''' ),
			( 'quoted_field_spans_lines_via_reader', '''
import csv

def write_file( path: str, content: str ) -> None:
	match File.binary_writer( path ):
		case Result.Ok( w ):
			w.write( content.get_const_ptr(), content.byte_len() ).unwrap( 'write' )
			w.close()
		case Result.Err( e ):
			sys.panic( 'could not open for write' )

def main() -> i32:
	path: str = 'csv_reader_case2.tmp'
	write_file( path, 'a,"line1\\nline2",c\\n' )

	r: csv.Reader
	match csv.reader( path ):
		case Result.Ok( rr ):
			r = rr
		case Result.Err( e ):
			return 10

	row: list[str] = list[str]()
	got: bool = False
	match r.__next__():
		case Result.Ok( m ):
			if m.has_row:
				got = True
				row = m.row
		case Result.Err( e2 ):
			return 11
	if not got:
		return 1
	if row.__len__() != 3:
		return 2
	if row.__getitem__( 1 ).unwrap( 'x' ) != 'line1\\nline2':
		return 3

	got2: bool = False
	match r.__next__():
		case Result.Ok( m2 ):
			if m2.has_row:
				got2 = True
		case Result.Err( e3 ):
			return 12
	if got2:
		return 4
	return 0
''' ),
			( 'reader_on_missing_file_returns_err', '''
import csv

def main() -> i32:
	match csv.reader( 'csv_reader_this_file_does_not_exist.tmp' ):
		case Result.Ok( rr ):
			return 1
		case Result.Err( e ):
			return 0
''' ),
			( 'malformed_row_surfaces_as_err_from_next', '''
import csv

def write_file( path: str, content: str ) -> None:
	match File.binary_writer( path ):
		case Result.Ok( w ):
			w.write( content.get_const_ptr(), content.byte_len() ).unwrap( 'write' )
			w.close()
		case Result.Err( e ):
			sys.panic( 'could not open for write' )

def main() -> i32:
	path: str = 'csv_reader_case4.tmp'
	write_file( path, '"a"b,c\\n' )

	r: csv.Reader
	match csv.reader( path ):
		case Result.Ok( rr ):
			r = rr
		case Result.Err( e ):
			return 10

	is_err: bool = False
	match r.__next__():
		case Result.Ok( m ):
			return 1
		case Result.Err( e2 ):
			is_err = True
	if not is_err:
		return 2
	return 0
''' ),
		])


if __name__ == '__main__':
	unittest.main()
