import unittest

import test_support
from compiler import Compiler
from discovery import Discovery


class Crc32Tests( test_support.RealCompileMixin, unittest.TestCase ):
	''' Real compile-and-run coverage for lib/crc32.py. '''

	def setUp( self ) -> None:
		self.discovery = Discovery( import_builtins = True )
		self.compiler = Compiler( self.discovery )

	@unittest.skipUnless( test_support.HAS_CC, 'no C compiler (clang/gcc/msvc) found - skipping' )
	def test_programs_compile_and_run( self ) -> None:
		self.assert_programs_run([
			( 'standard_check_vector', '''
import crc32

def main() -> i32:
	# the standard CRC-32 check value for the ASCII string "123456789"
	data: bytes = '123456789'.encode().unwrap( 'x' )
	result: u32 = crc32.crc32( data )
	if result != 0xCBF43926:
		return 1
	return 0
''' ),
			( 'empty_input_is_zero', '''
import crc32

def main() -> i32:
	empty: bytes = bytes.from_bytearray( move( bytearray( 0 ) ) )
	if crc32.crc32( empty ) != 0:
		return 1
	return 0
''' ),
			( 'incremental_matches_whole', '''
import crc32

def main() -> i32:
	whole: bytes = 'hello world'.encode().unwrap( 'x' )
	whole_crc: u32 = crc32.crc32( whole )

	first: bytes = 'hello '.encode().unwrap( 'x' )
	second: bytes = 'world'.encode().unwrap( 'x' )
	partial: u32 = crc32.crc32( first )
	incremental: u32 = crc32.crc32( second, initial = partial )

	if incremental != whole_crc:
		return 1
	return 0
''' ),
			( 'different_inputs_differ', '''
import crc32

def main() -> i32:
	a: bytes = 'aaaa'.encode().unwrap( 'x' )
	b: bytes = 'bbbb'.encode().unwrap( 'x' )
	if crc32.crc32( a ) == crc32.crc32( b ):
		return 1
	return 0
''' ),
		] )
