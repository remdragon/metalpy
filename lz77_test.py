import unittest

import test_support
from compiler import Compiler
from discovery import Discovery


class Lz77Tests( test_support.RealCompileMixin, unittest.TestCase ):
	''' Real compile-and-run coverage for lib/lz77.py. '''

	def setUp( self ) -> None:
		self.discovery = Discovery( import_builtins = True )
		self.compiler = Compiler( self.discovery )

	@unittest.skipUnless( test_support.HAS_CC, 'no C compiler (clang/gcc/msvc) found - skipping' )
	def test_programs_compile_and_run( self ) -> None:
		self.assert_programs_run([
			( 'repetitive_text_round_trips_via_replay', '''
import lz77

def replay( tokens: UnsafeList[lz77.LZ77Token] ) -> Result[UnsafeList[u8], lz77.LZ77Error]:
	out: UnsafeList[u8] = UnsafeList[u8]()
	i: usize = 0
	with compiler.panic_arithmetic( 'bounded by len(tokens), cannot overflow' ):
		while i < len( tokens ):
			tok: lz77.LZ77Token = tokens.__getitem__( i ).unwrap( 'i < len(tokens)' )
			match tok:
				case lz77.LZ77Token.Literal( b ):
					out.append( b )
				case lz77.LZ77Token.Match( m ):
					lz77.copy_match( out, usize( m.distance ), usize( m.length )).or_return()
			i += 1
	return Result.Ok( out )

def main() -> i32:
	original: bytes = ( 'abcabcabcabcabcabcabcabcabcabcabcabcabcabcabcabc' ).encode().unwrap( 'x' )
	tokens: UnsafeList[lz77.LZ77Token] = lz77.find_matches( original )

	# highly repetitive input should produce far fewer tokens than bytes
	if len( tokens ) >= len( original ):
		return 1

	replayed: UnsafeList[u8] = replay( tokens ).unwrap( 'replay' )
	if len( replayed ) != len( original ):
		return 2

	orig_ptr: ConstPtr[u8] = original.get_const_ptr()
	i: usize = 0
	with compiler.panic_arithmetic( 'bounded by len(original), cannot overflow' ):
		while i < len( original ):
			got: u8 = replayed.__getitem__( i ).unwrap( 'i < len(replayed)' )
			if got != orig_ptr[i]:
				return 3
			i += 1
	return 0
''' ),
			( 'random_bytes_degrade_to_literals_and_round_trip', '''
import lz77

def replay( tokens: UnsafeList[lz77.LZ77Token] ) -> Result[UnsafeList[u8], lz77.LZ77Error]:
	out: UnsafeList[u8] = UnsafeList[u8]()
	i: usize = 0
	with compiler.panic_arithmetic( 'bounded by len(tokens), cannot overflow' ):
		while i < len( tokens ):
			tok: lz77.LZ77Token = tokens.__getitem__( i ).unwrap( 'i < len(tokens)' )
			match tok:
				case lz77.LZ77Token.Literal( b ):
					out.append( b )
				case lz77.LZ77Token.Match( m ):
					lz77.copy_match( out, usize( m.distance ), usize( m.length )).or_return()
			i += 1
	return Result.Ok( out )

def main() -> i32:
	# a short, non-repeating byte sequence - no 3-byte substring repeats
	raw: bytearray = bytearray( 12 )
	rp: Ptr[u8] = raw.get_ptr()
	rp[0] = 5
	rp[1] = 200
	rp[2] = 17
	rp[3] = 99
	rp[4] = 3
	rp[5] = 250
	rp[6] = 42
	rp[7] = 1
	rp[8] = 222
	rp[9] = 77
	rp[10] = 8
	rp[11] = 190
	original: bytes = bytes.from_bytearray( move( raw ))

	tokens: UnsafeList[lz77.LZ77Token] = lz77.find_matches( original )
	if len( tokens ) != len( original ):  # every byte should be a literal
		return 1

	replayed: UnsafeList[u8] = replay( tokens ).unwrap( 'replay' )
	if len( replayed ) != len( original ):
		return 2
	orig_ptr: ConstPtr[u8] = original.get_const_ptr()
	i: usize = 0
	with compiler.panic_arithmetic( 'bounded by len(original), cannot overflow' ):
		while i < len( original ):
			got: u8 = replayed.__getitem__( i ).unwrap( 'i < len(replayed)' )
			if got != orig_ptr[i]:
				return 3
			i += 1
	return 0
''' ),
			( 'copy_match_handles_distance_less_than_length_overlap', '''
import lz77

def main() -> i32:
	out: UnsafeList[u8] = UnsafeList[u8]()
	out.append( u8( 65 ))  # seed byte 'A'

	# distance=1, length=10: must produce 10 more copies of 'A' - a
	# straight memcpy of the current 1-byte buffer would NOT do this
	lz77.copy_match( out, usize( 1 ), usize( 10 )).unwrap( 'copy_match' )

	if len( out ) != 11:
		return 1
	i: usize = 0
	with compiler.panic_arithmetic( 'bounded by 11, cannot overflow' ):
		while i < usize( 11 ):
			b: u8 = out.__getitem__( i ).unwrap( 'i < 11' )
			if b != 65:
				return 2
			i += 1
	return 0
''' ),
			( 'copy_match_rejects_invalid_distance', '''
import lz77

def main() -> i32:
	out: UnsafeList[u8] = UnsafeList[u8]()
	out.append( u8( 1 ))
	out.append( u8( 2 ))

	if lz77.copy_match( out, usize( 0 ), usize( 1 )).is_ok():
		return 1
	if lz77.copy_match( out, usize( 3 ), usize( 1 )).is_ok():  # distance > out_len
		return 2
	return 0
''' ),
			( 'empty_input_round_trips', '''
import lz77

def main() -> i32:
	empty: bytes = bytes.from_bytearray( move( bytearray( 0 )))
	tokens: UnsafeList[lz77.LZ77Token] = lz77.find_matches( empty )
	if len( tokens ) != usize( 0 ):
		return 1
	return 0
''' ),
		] )
