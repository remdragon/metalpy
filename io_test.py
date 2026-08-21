# io_test.py — real compile+link+run coverage for lib/io.py's Reader/Writer/
# Seekable protocols and the BinaryReader/BinaryWriter/BinaryReadWriter
# conformance added to lib/builtins/__File.py.

import unittest
from pathlib import Path

import test_support
from compiler import Compiler
from discovery import Discovery

@unittest.skipUnless( test_support.HAS_CC, 'no C compiler (clang/gcc/msvc) found - skipping' )
class IoTests( test_support.RealCompileMixin, unittest.TestCase ):
	def setUp( self ) -> None:
		self.discovery = Discovery( import_builtins = True )
		self.compiler = Compiler( self.discovery )

	def _run( self, code: str ) -> None:
		self.compiler.import_code( code, Path( '__main__.py' ), scope = None )
		self.compiler.run()

	def test_binary_read_writer_conforms_to_reader_writer_seekable( self ) -> None:
		self._run( '''
import compiler
import io
import fs

def read_all_lines[T: io.Reader]( r: T ) -> Result[list[bytearray], OSError]:
	lines: list[bytearray] = list[bytearray]()
	while True:
		line: bytearray = r.readline().or_return()
		if len( line ) == 0:
			return Result.Ok( lines )
		lines.append( line ).unwrap( 'read_all_lines: append' )

def run() -> Result[i32, OSError]:
	path: str = 'io_test_tmp.bin'
	msg: bytes = b'hello\\nworld\\n'
	w: BinaryWriter = File.binary_writer( path ).or_return()
	io.write_all( w, msg.get_const_ptr(), len( msg )).or_return()
	w.close()

	f: BinaryReadWriter = File.binary_read_writer( path, truncate = False, exists = True ).or_return()
	if f.tell().or_return() != i64( 0 ):
		return Result.Ok( 1 )
	lines: list[bytearray] = read_all_lines( f ).or_return()
	if len( lines ) != 2:
		return Result.Ok( 2 )
	first: bytearray = lines.__getitem__( 0 ).unwrap( 'lines[0]' )
	if first.decode().unwrap( 'decode' ) != 'hello\\n':
		return Result.Ok( 3 )
	end_pos: i64 = f.tell().or_return()
	if end_pos != i64( len( msg )):
		return Result.Ok( 4 )
	f.seek( i64( 0 ), fs.SEEK_SET ).or_return()
	if f.tell().or_return() != i64( 0 ):
		return Result.Ok( 5 )
	f.close()
	return Result.Ok( 0 )

def main() -> i32:
	match run():
		case Result.Ok( code ):
			return code
		case Result.Err( _ ):
			return 90
''' )
		self.assertEqual( self.discovery.errors.errors, [] )
		self._assert_compiles_and_runs( _emit( self.compiler ), expected_exit = 0 )

	def test_readuntil_returns_partial_bytes_at_eof_without_delimiter( self ) -> None:
		self._run( '''
import compiler
import io

def run() -> Result[i32, OSError]:
	path: str = 'io_test_partial_tmp.bin'
	msg: bytes = b'no newline here'
	w: BinaryWriter = File.binary_writer( path ).or_return()
	io.write_all( w, msg.get_const_ptr(), len( msg )).or_return()
	w.close()

	r: BinaryReader = File.binary_reader( path ).or_return()
	line: bytearray = r.readline().or_return()
	if line.decode().unwrap( 'decode' ) != 'no newline here':
		return Result.Ok( 1 )
	# a second readline() past EOF must return Ok(empty), not an error
	again: bytearray = r.readline().or_return()
	if len( again ) != usize( 0 ):
		return Result.Ok( 2 )
	r.close()
	return Result.Ok( 0 )

def main() -> i32:
	match run():
		case Result.Ok( code ):
			return code
		case Result.Err( _ ):
			return 90
''' )
		self.assertEqual( self.discovery.errors.errors, [] )
		self._assert_compiles_and_runs( _emit( self.compiler ), expected_exit = 0 )

def _emit( compiler: Compiler ) -> str:
	import emitter_c
	return emitter_c.emit_c( compiler )

if __name__ == '__main__':
	unittest.main()
