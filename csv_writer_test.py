import unittest

import test_support
from compiler import Compiler
from discovery import Discovery


class CsvWriterTests( test_support.RealCompileMixin, unittest.TestCase ):
	''' PLAN csv.py, Step 5 - Writer/writer() over a real BinaryWriter. No
	iteration involved on the write side. Verifies exact CRLF byte output
	(Writer always writes "\\r\\n" per the module's QUOTE_MINIMAL-equivalent
	v1 scope) and a write-then-read-back round trip through csv.reader(). '''

	def setUp( self ) -> None:
		self.discovery = Discovery( import_builtins = True )
		self.compiler = Compiler( self.discovery )

	@unittest.skipUnless( test_support.HAS_CC, 'no C compiler (clang/gcc/msvc) found - skipping' )
	def test_programs_compile_and_run( self ) -> None:
		self.assert_programs_run([
			( 'writerow_produces_exact_crlf_bytes', '''
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
	path: str = 'csv_writer_case1.tmp'
	w = csv.writer( path ).unwrap( 'open for write' )

	row1: list[str] = list[str]()
	row1.append( 'a' )
	row1.append( 'b,c' )
	w.writerow( row1 ).unwrap( 'writerow1' )

	row2: list[str] = list[str]()
	row2.append( '1' )
	row2.append( '2' )
	w.writerow( row2 ).unwrap( 'writerow2' )
	w.close()

	content: str = read_whole_file( path )
	if content != 'a,"b,c"\\r\\n1,2\\r\\n':
		return 1
	return 0
''' ),
			( 'writerows_convenience', '''
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
	path: str = 'csv_writer_case2.tmp'
	w = csv.writer( path ).unwrap( 'open for write' )

	rows: list[list[str]] = list[list[str]]()
	row1: list[str] = list[str]()
	row1.append( 'x' )
	row1.append( 'y' )
	rows.append( row1 )
	row2: list[str] = list[str]()
	row2.append( 'z' )
	rows.append( row2 )

	w.writerows( rows ).unwrap( 'writerows' )
	w.close()

	content: str = read_whole_file( path )
	if content != 'x,y\\r\\nz\\r\\n':
		return 1
	return 0
''' ),
			( 'write_then_read_back_round_trip', '''
import csv

def main() -> i32:
	path: str = 'csv_writer_case3.tmp'
	w = csv.writer( path ).unwrap( 'open for write' )

	row1: list[str] = list[str]()
	row1.append( 'name' )
	row1.append( 'note' )
	w.writerow( row1 ).unwrap( 'x' )

	row2: list[str] = list[str]()
	row2.append( 'alice' )
	row2.append( 'has "quotes" and, a comma' )
	w.writerow( row2 ).unwrap( 'x' )

	row3: list[str] = list[str]()
	row3.append( 'bob' )
	row3.append( 'multi\\nline\\nnote' )
	w.writerow( row3 ).unwrap( 'x' )
	w.close()

	r = csv.reader( path ).unwrap( 'open for read' )

	m1 = r.__next__().unwrap( 'n1' )
	if not m1.has_row:
		return 1
	if m1.row.__getitem__( 0 ).unwrap( 'x' ) != 'name':
		return 2

	m2 = r.__next__().unwrap( 'n2' )
	if not m2.has_row:
		return 3
	if m2.row.__getitem__( 1 ).unwrap( 'x' ) != 'has "quotes" and, a comma':
		return 4

	m3 = r.__next__().unwrap( 'n3' )
	if not m3.has_row:
		return 5
	if m3.row.__getitem__( 1 ).unwrap( 'x' ) != 'multi\\nline\\nnote':
		return 6

	m4 = r.__next__().unwrap( 'n4' )
	if m4.has_row:
		return 7
	return 0
''' ),
			( 'writer_on_invalid_path_returns_err', '''
import csv

def main() -> i32:
	match csv.writer( 'csv_writer_nonexistent_dir/nope.tmp' ):
		case Result.Ok( w ):
			return 1
		case Result.Err( e ):
			return 0
''' ),
		])


if __name__ == '__main__':
	unittest.main()
