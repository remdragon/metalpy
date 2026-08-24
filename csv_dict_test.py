import unittest

import test_support
from compiler import Compiler
from discovery import Discovery


class CsvDictTests( test_support.RealCompileMixin, unittest.TestCase ):
	''' PLAN csv.py, Step 6 - DictReader/DictWriter on top of Reader/Writer. '''

	def setUp( self ) -> None:
		self.discovery = Discovery( import_builtins = True )
		self.compiler = Compiler( self.discovery )

	@unittest.skipUnless( test_support.HAS_CC, 'no C compiler (clang/gcc/msvc) found - skipping' )
	def test_programs_compile_and_run( self ) -> None:
		self.assert_programs_run([
			( 'dict_reader_header_detection_and_basic_rows', '''
import csv

def write_file( path: str, content: str ) -> None:
	match File.binary_writer( path ):
		case Result.Ok( w ):
			w.write( content.get_const_ptr(), content.byte_len() ).unwrap( 'write' )
			w.close()
		case Result.Err( e ):
			sys.panic( 'could not open for write' )

def main() -> i32:
	path: str = 'csv_dict_case1.tmp'
	write_file( path, 'name,age\\nalice,30\\nbob,25\\n' )

	dr = csv.dict_reader( path ).unwrap( 'open' )

	m1 = dr.__next__().unwrap( 'n1' )
	if not m1.has_row:
		return 1
	if m1.row.__getitem__( 'name' ).unwrap( 'x' ) != 'alice':
		return 2
	if m1.row.__getitem__( 'age' ).unwrap( 'x' ) != '30':
		return 3

	m2 = dr.__next__().unwrap( 'n2' )
	if not m2.has_row:
		return 4
	if m2.row.__getitem__( 'name' ).unwrap( 'x' ) != 'bob':
		return 5

	m3 = dr.__next__().unwrap( 'n3' )
	if m3.has_row:
		return 6
	return 0
''' ),
			( 'dict_reader_short_row_gets_restval', '''
import csv

def write_file( path: str, content: str ) -> None:
	match File.binary_writer( path ):
		case Result.Ok( w ):
			w.write( content.get_const_ptr(), content.byte_len() ).unwrap( 'write' )
			w.close()
		case Result.Err( e ):
			sys.panic( 'could not open for write' )

def main() -> i32:
	path: str = 'csv_dict_case2.tmp'
	write_file( path, 'a,b,c\\nx\\n' )

	dr = csv.dict_reader( path ).unwrap( 'open' )
	m1 = dr.__next__().unwrap( 'n1' )
	if not m1.has_row:
		return 1
	if m1.row.__getitem__( 'a' ).unwrap( 'x' ) != 'x':
		return 2
	if m1.row.__getitem__( 'b' ).unwrap( 'x' ) != '':
		return 3
	if m1.row.__getitem__( 'c' ).unwrap( 'x' ) != '':
		return 4
	return 0
''' ),
			( 'dict_writer_header_and_rows', '''
import csv

def read_whole_file( path: str ) -> str:
	r = File.binary_reader( path ).unwrap( 'open' )
	buf: Ptr[u8] = sys.alloc[u8]( 65536 )
	defer( sys.free( buf ))
	n: usize = r.read( buf, 65536 ).unwrap( 'read' )
	buf[n] = 0
	const_buf: ConstPtr[u8] = compiler.cast( ConstPtr[u8], buf )
	with compiler.wrap_arithmetic:
		total: usize = n + 1
	return str.from_cstr( const_buf, total ).unwrap( 'decode' )

def main() -> i32:
	path: str = 'csv_dict_case3.tmp'
	fieldnames: list[str] = list[str]()
	fieldnames.append( 'name' )
	fieldnames.append( 'age' )

	dw = csv.dict_writer( path, fieldnames ).unwrap( 'open for write' )
	dw.writeheader().unwrap( 'header' )

	row1: dict[str,str] = dict[str,str]()
	row1.__setitem__( 'name', 'alice' )
	row1.__setitem__( 'age', '30' )
	dw.writerow( row1 ).unwrap( 'row1' )
	dw.close()

	content: str = read_whole_file( path )
	if content != 'name,age\\r\\nalice,30\\r\\n':
		return 1
	return 0
''' ),
			( 'dict_writer_missing_key_gets_restval', '''
import csv

def read_whole_file( path: str ) -> str:
	r = File.binary_reader( path ).unwrap( 'open' )
	buf: Ptr[u8] = sys.alloc[u8]( 65536 )
	defer( sys.free( buf ))
	n: usize = r.read( buf, 65536 ).unwrap( 'read' )
	buf[n] = 0
	const_buf: ConstPtr[u8] = compiler.cast( ConstPtr[u8], buf )
	with compiler.wrap_arithmetic:
		total: usize = n + 1
	return str.from_cstr( const_buf, total ).unwrap( 'decode' )

def main() -> i32:
	path: str = 'csv_dict_case4.tmp'
	fieldnames: list[str] = list[str]()
	fieldnames.append( 'a' )
	fieldnames.append( 'b' )

	dw = csv.dict_writer( path, fieldnames ).unwrap( 'open for write' )
	row1: dict[str,str] = dict[str,str]()
	row1.__setitem__( 'a', 'only_a' )
	dw.writerow( row1 ).unwrap( 'row1' )
	dw.close()

	content: str = read_whole_file( path )
	if content != 'only_a,\\r\\n':
		return 1
	return 0
''' ),
			( 'dict_writer_then_dict_reader_round_trip', '''
import csv

def main() -> i32:
	path: str = 'csv_dict_case5.tmp'
	fieldnames: list[str] = list[str]()
	fieldnames.append( 'id' )
	fieldnames.append( 'note' )

	dw = csv.dict_writer( path, fieldnames ).unwrap( 'open for write' )
	dw.writeheader().unwrap( 'header' )
	row1: dict[str,str] = dict[str,str]()
	row1.__setitem__( 'id', '1' )
	row1.__setitem__( 'note', 'has, comma and "quote"' )
	dw.writerow( row1 ).unwrap( 'row1' )
	dw.close()

	dr = csv.dict_reader( path ).unwrap( 'open for read' )
	m1 = dr.__next__().unwrap( 'n1' )
	if not m1.has_row:
		return 1
	if m1.row.__getitem__( 'id' ).unwrap( 'x' ) != '1':
		return 2
	if m1.row.__getitem__( 'note' ).unwrap( 'x' ) != 'has, comma and "quote"':
		return 3

	m2 = dr.__next__().unwrap( 'n2' )
	if m2.has_row:
		return 4
	return 0
''' ),
		])


if __name__ == '__main__':
	unittest.main()
