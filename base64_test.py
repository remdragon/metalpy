import unittest

import test_support
from compiler import Compiler
from discovery import Discovery


class Base64Tests( test_support.RealCompileMixin, unittest.TestCase ):
	''' Real compile-and-run coverage for lib/base64.py. '''

	def setUp( self ) -> None:
		self.discovery = Discovery( import_builtins = True )
		self.compiler = Compiler( self.discovery )

	@unittest.skipUnless( test_support.HAS_CC, 'no C compiler (clang/gcc/msvc) found - skipping' )
	def test_programs_compile_and_run( self ) -> None:
		self.assert_programs_run([
			( 'b64_known_vectors_round_trip', '''
import base64

def main() -> i32:
	raw: bytes = 'hello world'.encode().unwrap( 'x' )
	enc: bytes = base64.b64encode( raw )
	if enc.decode().unwrap( 'x' ) != 'aGVsbG8gd29ybGQ=':
		return 1
	back: bytes = base64.b64decode( enc ).unwrap( 'b64decode' )
	if back.decode().unwrap( 'x' ) != 'hello world':
		return 2
	return 0
''' ),
			( 'b16_known_vectors_round_trip', '''
import base64

def main() -> i32:
	raw: bytes = 'ab'.encode().unwrap( 'x' )
	enc: bytes = base64.b16encode( raw )
	if enc.decode().unwrap( 'x' ) != '6162':
		return 1
	back: bytes = base64.b16decode( enc ).unwrap( 'b16decode' )
	if back.decode().unwrap( 'x' ) != 'ab':
		return 2
	return 0
''' ),
			( 'base64_error_str_and_repr', '''
import base64

def main() -> i32:
	odd: bytes = 'abc'.encode().unwrap( 'x' ) # odd length - not valid base16
	match base64.b16decode( odd ):
		case Result.Err( e ):
			if str( e ) != 'invalid base16-encoded string: odd length':
				return 1
			if f'{e}' != 'invalid base16-encoded string: odd length':
				return 2
			if e.__repr__() != "Base64Error('invalid base16-encoded string: odd length')":
				return 3
			return 0
		case Result.Ok( _ ):
			return 4
''' ),
		] )


if __name__ == '__main__':
	unittest.main()
