import unittest

import test_support
from compiler import Compiler
from discovery import Discovery


class Sha512Tests( test_support.RealCompileMixin, unittest.TestCase ):
	''' Real compile-and-run coverage for lib/sha512.py, against the FIPS
	180-4 / NIST test vectors. '''

	def setUp( self ) -> None:
		self.discovery = Discovery( import_builtins = True )
		self.compiler = Compiler( self.discovery )

	@unittest.skipUnless( test_support.HAS_CC, 'no C compiler (clang/gcc/msvc) found - skipping' )
	def test_programs_compile_and_run( self ) -> None:
		self.assert_programs_run([
			( 'empty_string', '''
import base64
import sha512

def main() -> i32:
	empty: bytes = bytes.from_bytearray( move( bytearray( 0 ) ) )
	digest: bytes = sha512.sha512( empty )
	hex: bytes = base64.b16encode( digest )
	if hex.decode().unwrap( 'x' ) != 'CF83E1357EEFB8BDF1542850D66D8007D620E4050B5715DC83F4A921D36CE9CE47D0D13C5D85F2B0FF8318D2877EEC2F63B931BD47417A81A538327AF927DA3E':
		return 1
	return 0
''' ),
			( 'abc', '''
import base64
import sha512

def main() -> i32:
	data: bytes = 'abc'.encode().unwrap( 'x' )
	digest: bytes = sha512.sha512( data )
	hex: bytes = base64.b16encode( digest )
	if hex.decode().unwrap( 'x' ) != 'DDAF35A193617ABACC417349AE20413112E6FA4E89A97EA20A9EEEE64B55D39A2192992A274FC1A836BA3C23A3FEEBBD454D4423643CE80E2A9AC94FA54CA49F':
		return 1
	return 0
''' ),
			( 'quick_brown_fox', '''
import base64
import sha512

def main() -> i32:
	data: bytes = 'The quick brown fox jumps over the lazy dog'.encode().unwrap( 'x' )
	digest: bytes = sha512.sha512( data )
	hex: bytes = base64.b16encode( digest )
	if hex.decode().unwrap( 'x' ) != '07E547D9586F6A73F73FBAC0435ED76951218FB7D0C8D788A309D785436BBB642E93A252A954F23912547D1E8A3B5ED6E1BFD7097821233FA0538F3DB854FEE6':
		return 1
	return 0
''' ),
			# 112-byte input spans two 128-byte blocks, exercising the
			# multi-block loop and length-encoding path a single-block
			# input never reaches.
			( 'two_block', '''
import base64
import sha512

def main() -> i32:
	data: bytes = 'abcdefghbcdefghicdefghijdefghijkefghijklfghijklmghijklmnhijklmnoijklmnopjklmnopqklmnopqrlmnopqrsmnopqrstnopqrstu'.encode().unwrap( 'x' )
	digest: bytes = sha512.sha512( data )
	hex: bytes = base64.b16encode( digest )
	if hex.decode().unwrap( 'x' ) != '8E959B75DAE313DA8CF4F72814FC143F8F7779C6EB9F7FA17299AEADB6889018501D289E4900F7E4331B99DEC4B5433AC7D329EEB6DD26545E96E55B874BE909':
		return 1
	return 0
''' ),
			( 'digest_length_is_64', '''
import sha512

def main() -> i32:
	data: bytes = 'abc'.encode().unwrap( 'x' )
	digest: bytes = sha512.sha512( data )
	if len( digest ) != 64:
		return 1
	return 0
''' ),
			( 'different_inputs_differ', '''
import base64
import sha512

def main() -> i32:
	a: bytes = 'aaaa'.encode().unwrap( 'x' )
	b: bytes = 'bbbb'.encode().unwrap( 'x' )
	hex_a: str = base64.b16encode( sha512.sha512( a ) ).decode().unwrap( 'x' )
	hex_b: str = base64.b16encode( sha512.sha512( b ) ).decode().unwrap( 'x' )
	if hex_a == hex_b:
		return 1
	return 0
''' ),
		] )


if __name__ == '__main__':
	unittest.main()
