import unittest

import test_support
from compiler import Compiler
from discovery import Discovery


class BinasciiTests( test_support.RealCompileMixin, unittest.TestCase ):
	''' Real compile-and-run coverage for lib/binascii.py - hexlify/unhexlify. '''

	def setUp( self ) -> None:
		self.discovery = Discovery( import_builtins = True )
		self.compiler = Compiler( self.discovery )

	@unittest.skipUnless( test_support.HAS_CC, 'no C compiler (clang/gcc/msvc) found - skipping' )
	def test_programs_compile_and_run( self ) -> None:
		self.assert_programs_run([
			( 'hexlify_known_vectors_round_trip', '''
import binascii

def main() -> i32:
	raw = bytearray( 4 )
	rp: Ptr[u8] = raw.get_ptr()
	rp[0] = 0x00
	rp[1] = 0x01
	rp[2] = 0xFE
	rp[3] = 0xFF
	rb: bytes = bytes.from_bytearray( move( raw ) )

	hx: bytes = binascii.hexlify( rb )
	hx_s: str = hx.decode().unwrap( 'x' )
	if hx_s != '0001feff':
		return 1

	back: bytes = binascii.unhexlify( hx ).unwrap( 'unhexlify failed' )
	if len( back ) != 4:
		return 2
	bp: ConstPtr[u8] = back.get_const_ptr()
	if bp[0] != 0x00 or bp[1] != 0x01 or bp[2] != 0xFE or bp[3] != 0xFF:
		return 3
	return 0
''' ),
			( 'empty_input_round_trip', '''
import binascii

def main() -> i32:
	empty: bytes = bytes.from_bytearray( move( bytearray( 0 ) ) )
	hx: bytes = binascii.hexlify( empty )
	if len( hx ) != 0:
		return 1

	back: bytes = binascii.unhexlify( empty ).unwrap( 'unhexlify of empty input failed' )
	if len( back ) != 0:
		return 2
	return 0
''' ),
			( 'unhexlify_is_case_insensitive', '''
import binascii

def main() -> i32:
	upper: bytes = 'AaBbCcDdEeFf'.encode().unwrap( 'x' )
	lower: bytes = 'aabbccddeeff'.encode().unwrap( 'x' )

	from_upper: bytes = binascii.unhexlify( upper ).unwrap( 'mixed-case decode failed' )
	from_lower: bytes = binascii.unhexlify( lower ).unwrap( 'lowercase decode failed' )

	if len( from_upper ) != len( from_lower ):
		return 1
	up: ConstPtr[u8] = from_upper.get_const_ptr()
	lo: ConstPtr[u8] = from_lower.get_const_ptr()
	i: usize = 0
	with compiler.panic_arithmetic( 'bounded by shared length, cannot overflow' ):
		while i < len( from_upper ):
			if up[i] != lo[i]:
				return 2
			i += 1
	return 0
''' ),
			( 'unhexlify_error_cases', '''
import binascii

def main() -> i32:
	odd: bytes = 'abc'.encode().unwrap( 'x' ) # odd length
	if binascii.unhexlify( odd ).is_ok():
		return 1

	bad_char: bytes = 'zz'.encode().unwrap( 'x' ) # not hex digits
	if binascii.unhexlify( bad_char ).is_ok():
		return 2
	return 0
''' ),
		] )
