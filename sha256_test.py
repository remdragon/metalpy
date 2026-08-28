import unittest

import test_support
from compiler import Compiler
from discovery import Discovery


class Sha256Tests( test_support.RealCompileMixin, unittest.TestCase ):
	''' Real compile-and-run coverage for lib/sha256.py, against the FIPS
	180-4 / NIST test vectors. '''

	def setUp( self ) -> None:
		self.discovery = Discovery( import_builtins = True )
		self.compiler = Compiler( self.discovery )

	@unittest.skipUnless( test_support.HAS_CC, 'no C compiler (clang/gcc/msvc) found - skipping' )
	def test_programs_compile_and_run( self ) -> None:
		self.assert_programs_run([
			( 'empty_string', '''
import base64
import sha256

def main() -> i32:
	empty: bytes = bytes.from_bytearray( move( bytearray( 0 ) ) )
	digest: bytes = sha256.sha256( empty )
	hex: bytes = base64.b16encode( digest )
	if hex.decode().unwrap( 'x' ) != 'E3B0C44298FC1C149AFBF4C8996FB92427AE41E4649B934CA495991B7852B855':
		return 1
	return 0
''' ),
			( 'abc', '''
import base64
import sha256

def main() -> i32:
	data: bytes = 'abc'.encode().unwrap( 'x' )
	digest: bytes = sha256.sha256( data )
	hex: bytes = base64.b16encode( digest )
	if hex.decode().unwrap( 'x' ) != 'BA7816BF8F01CFEA414140DE5DAE2223B00361A396177A9CB410FF61F20015AD':
		return 1
	return 0
''' ),
			# NIST's classic two-block vector (56 bytes: 0x80 + zero pad
			# pushes it into a second 64-byte block), exercising the
			# multi-block loop and the padding/length-encoding path a
			# single-block input never reaches.
			( 'nist_two_block', '''
import base64
import sha256

def main() -> i32:
	data: bytes = 'abcdbcdecdefdefgefghfghighijhijkijkljklmklmnlmnomnopnopq'.encode().unwrap( 'x' )
	digest: bytes = sha256.sha256( data )
	hex: bytes = base64.b16encode( digest )
	if hex.decode().unwrap( 'x' ) != '248D6A61D20638B8E5C026930C3E6039A33CE45964FF2167F6ECEDD419DB06C1':
		return 1
	return 0
''' ),
			( 'quick_brown_fox', '''
import base64
import sha256

def main() -> i32:
	data: bytes = 'The quick brown fox jumps over the lazy dog'.encode().unwrap( 'x' )
	digest: bytes = sha256.sha256( data )
	hex: bytes = base64.b16encode( digest )
	if hex.decode().unwrap( 'x' ) != 'D7A8FBB307D7809469CA9ABCB0082E4F8D5651E46D3CDB762D02D0BF37C9E592':
		return 1
	return 0
''' ),
			( 'digest_length_is_32', '''
import sha256

def main() -> i32:
	data: bytes = 'abc'.encode().unwrap( 'x' )
	digest: bytes = sha256.sha256( data )
	if len( digest ) != 32:
		return 1
	return 0
''' ),
			( 'different_inputs_differ', '''
import base64
import sha256

def main() -> i32:
	a: bytes = 'aaaa'.encode().unwrap( 'x' )
	b: bytes = 'bbbb'.encode().unwrap( 'x' )
	hex_a: str = base64.b16encode( sha256.sha256( a ) ).decode().unwrap( 'x' )
	hex_b: str = base64.b16encode( sha256.sha256( b ) ).decode().unwrap( 'x' )
	if hex_a == hex_b:
		return 1
	return 0
''' ),
		] )


if __name__ == '__main__':
	unittest.main()
