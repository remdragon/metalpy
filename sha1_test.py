import unittest

import test_support
from compiler import Compiler
from discovery import Discovery


class Sha1Tests( test_support.RealCompileMixin, unittest.TestCase ):
	''' Real compile-and-run coverage for lib/sha1.py, against FIPS 180-1 /
	NIST test vectors. '''

	def setUp( self ) -> None:
		self.discovery = Discovery( import_builtins = True )
		self.compiler = Compiler( self.discovery )

	@unittest.skipUnless( test_support.HAS_CC, 'no C compiler (clang/gcc/msvc) found - skipping' )
	def test_programs_compile_and_run( self ) -> None:
		self.assert_programs_run([
			( 'empty_string', '''
import base64
import sha1

def main() -> i32:
	empty: bytes = bytes.from_bytearray( move( bytearray( 0 ) ) )
	digest: bytes = sha1.sha1( empty )
	hex: bytes = base64.b16encode( digest )
	if hex.decode().unwrap( 'x' ) != 'DA39A3EE5E6B4B0D3255BFEF95601890AFD80709':
		return 1
	return 0
''' ),
			( 'abc', '''
import base64
import sha1

def main() -> i32:
	data: bytes = 'abc'.encode().unwrap( 'x' )
	digest: bytes = sha1.sha1( data )
	hex: bytes = base64.b16encode( digest )
	if hex.decode().unwrap( 'x' ) != 'A9993E364706816ABA3E25717850C26C9CD0D89D':
		return 1
	return 0
''' ),
			# NIST's classic two-block vector, exercising the message-
			# expansion/multi-block loop past a single 64-byte block.
			( 'nist_two_block', '''
import base64
import sha1

def main() -> i32:
	data: bytes = 'abcdbcdecdefdefgefghfghighijhijkijkljklmklmnlmnomnopnopq'.encode().unwrap( 'x' )
	digest: bytes = sha1.sha1( data )
	hex: bytes = base64.b16encode( digest )
	if hex.decode().unwrap( 'x' ) != '84983E441C3BD26EBAAE4AA1F95129E5E54670F1':
		return 1
	return 0
''' ),
			( 'quick_brown_fox', '''
import base64
import sha1

def main() -> i32:
	data: bytes = 'The quick brown fox jumps over the lazy dog'.encode().unwrap( 'x' )
	digest: bytes = sha1.sha1( data )
	hex: bytes = base64.b16encode( digest )
	if hex.decode().unwrap( 'x' ) != '2FD4E1C67A2D28FCED849EE1BB76E7391B93EB12':
		return 1
	return 0
''' ),
			( 'digest_length_is_20', '''
import sha1

def main() -> i32:
	data: bytes = 'abc'.encode().unwrap( 'x' )
	digest: bytes = sha1.sha1( data )
	if len( digest ) != 20:
		return 1
	return 0
''' ),
			( 'different_inputs_differ', '''
import base64
import sha1

def main() -> i32:
	a: bytes = 'aaaa'.encode().unwrap( 'x' )
	b: bytes = 'bbbb'.encode().unwrap( 'x' )
	hex_a: str = base64.b16encode( sha1.sha1( a ) ).decode().unwrap( 'x' )
	hex_b: str = base64.b16encode( sha1.sha1( b ) ).decode().unwrap( 'x' )
	if hex_a == hex_b:
		return 1
	return 0
''' ),
		] )


if __name__ == '__main__':
	unittest.main()
