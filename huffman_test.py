import unittest

import test_support
from compiler import Compiler
from discovery import Discovery


class HuffmanTests( test_support.RealCompileMixin, unittest.TestCase ):
	''' Real compile-and-run coverage for lib/huffman.py. '''

	def setUp( self ) -> None:
		self.discovery = Discovery( import_builtins = True )
		self.compiler = Compiler( self.discovery )

	@unittest.skipUnless( test_support.HAS_CC, 'no C compiler (clang/gcc/msvc) found - skipping' )
	def test_programs_compile_and_run( self ) -> None:
		self.assert_programs_run([
			( 'rfc1951_worked_example_codes', '''
import huffman

def main() -> i32:
	# RFC 1951 SS3.2.2's own worked example: alphabet {A..H} (indices 0..7),
	# lengths {3,3,3,3,3,2,4,4} -> canonical codes F=00 A=010 B=011 C=100
	# D=101 E=110 G=1110 H=1111
	lengths: UnsafeList[u8] = UnsafeList[u8]( usize( 8 ))
	lengths.append( u8( 3 )).unwrap( 'A' )
	lengths.append( u8( 3 )).unwrap( 'B' )
	lengths.append( u8( 3 )).unwrap( 'C' )
	lengths.append( u8( 3 )).unwrap( 'D' )
	lengths.append( u8( 3 )).unwrap( 'E' )
	lengths.append( u8( 2 )).unwrap( 'F' )
	lengths.append( u8( 4 )).unwrap( 'G' )
	lengths.append( u8( 4 )).unwrap( 'H' )

	enc: huffman.HuffmanEncoder = huffman.HuffmanEncoder( lengths ).unwrap( 'valid table' )

	expected_codes: UnsafeList[u16] = UnsafeList[u16]( usize( 8 ))
	expected_codes.append( u16( 2 )).unwrap( 'x' )   # A
	expected_codes.append( u16( 3 )).unwrap( 'x' )   # B
	expected_codes.append( u16( 4 )).unwrap( 'x' )   # C
	expected_codes.append( u16( 5 )).unwrap( 'x' )   # D
	expected_codes.append( u16( 6 )).unwrap( 'x' )   # E
	expected_codes.append( u16( 0 )).unwrap( 'x' )   # F
	expected_codes.append( u16( 14 )).unwrap( 'x' )  # G
	expected_codes.append( u16( 15 )).unwrap( 'x' )  # H

	i: usize = 0
	with compiler.panic_arithmetic( 'bounded by 8, cannot overflow' ):
		while i < usize( 8 ):
			got: tuple[u16,u8] = enc.code_for( u16( i )).unwrap( 'symbol in alphabet' )
			want_code: u16 = expected_codes.__getitem__( i ).unwrap( 'i < 8' )
			want_len: u8 = lengths.__getitem__( i ).unwrap( 'i < 8' )
			if got[0] != want_code:
				return 1
			if got[1] != want_len:
				return 2
			i += 1
	return 0
''' ),
			( 'encode_decode_round_trip_rfc_example', '''
import huffman
import bitstream

def main() -> i32:
	lengths: UnsafeList[u8] = UnsafeList[u8]( usize( 8 ))
	lengths.append( u8( 3 )).unwrap( 'x' )
	lengths.append( u8( 3 )).unwrap( 'x' )
	lengths.append( u8( 3 )).unwrap( 'x' )
	lengths.append( u8( 3 )).unwrap( 'x' )
	lengths.append( u8( 3 )).unwrap( 'x' )
	lengths.append( u8( 2 )).unwrap( 'x' )
	lengths.append( u8( 4 )).unwrap( 'x' )
	lengths.append( u8( 4 )).unwrap( 'x' )

	enc: huffman.HuffmanEncoder = huffman.HuffmanEncoder( lengths ).unwrap( 'x' )
	dec: huffman.HuffmanDecoder = huffman.HuffmanDecoder( lengths ).unwrap( 'x' )

	w = bitstream.BitWriter()
	# encode the sequence F A H F B G E (mixes every code length present)
	seq: UnsafeList[u16] = UnsafeList[u16]( usize( 7 ))
	seq.append( u16( 5 )).unwrap( 'x' )  # F
	seq.append( u16( 0 )).unwrap( 'x' )  # A
	seq.append( u16( 7 )).unwrap( 'x' )  # H
	seq.append( u16( 5 )).unwrap( 'x' )  # F
	seq.append( u16( 1 )).unwrap( 'x' )  # B
	seq.append( u16( 6 )).unwrap( 'x' )  # G
	seq.append( u16( 4 )).unwrap( 'x' )  # E

	i: usize = 0
	with compiler.panic_arithmetic( 'bounded by 7, cannot overflow' ):
		while i < usize( 7 ):
			sym: u16 = seq.__getitem__( i ).unwrap( 'i < 7' )
			enc.write_symbol( w, sym ).unwrap( 'write_symbol' )
			i += 1
	out: bytes = w.finish()

	r = bitstream.BitReader( out )
	j: usize = 0
	with compiler.panic_arithmetic( 'bounded by 7, cannot overflow' ):
		while j < usize( 7 ):
			expected: u16 = seq.__getitem__( j ).unwrap( 'j < 7' )
			decoded: u16 = dec.decode( r ).unwrap( 'decode' )
			if decoded != expected:
				return 1
			j += 1
	return 0
''' ),
			( 'oversubscribed_code_is_rejected', '''
import huffman

def main() -> i32:
	# 4 symbols all claiming length 1 - only 2 length-1 codes can exist
	lengths: UnsafeList[u8] = UnsafeList[u8]( usize( 4 ))
	lengths.append( u8( 1 )).unwrap( 'x' )
	lengths.append( u8( 1 )).unwrap( 'x' )
	lengths.append( u8( 1 )).unwrap( 'x' )
	lengths.append( u8( 1 )).unwrap( 'x' )
	if huffman.HuffmanEncoder( lengths ).is_ok():
		return 1
	if huffman.HuffmanDecoder( lengths ).is_ok():
		return 2
	return 0
''' ),
			( 'build_lengths_from_frequencies_round_trips', '''
import huffman
import bitstream

def main() -> i32:
	# skewed but not pathological frequencies over 5 symbols
	freqs: UnsafeList[u32] = UnsafeList[u32]( usize( 5 ))
	freqs.append( u32( 100 )).unwrap( 'x' )
	freqs.append( u32( 50 )).unwrap( 'x' )
	freqs.append( u32( 20 )).unwrap( 'x' )
	freqs.append( u32( 5 )).unwrap( 'x' )
	freqs.append( u32( 1 )).unwrap( 'x' )

	lengths: UnsafeList[u8] = huffman.build_code_lengths_from_frequencies( freqs, u8( 15 )).unwrap( 'x' )

	enc: huffman.HuffmanEncoder = huffman.HuffmanEncoder( lengths ).unwrap( 'valid complete code' )
	dec: huffman.HuffmanDecoder = huffman.HuffmanDecoder( lengths ).unwrap( 'valid complete code' )

	w = bitstream.BitWriter()
	i: usize = 0
	with compiler.panic_arithmetic( 'bounded by 5, cannot overflow' ):
		while i < usize( 5 ):
			enc.write_symbol( w, u16( i )).unwrap( 'write_symbol' )
			i += 1
	out: bytes = w.finish()

	r = bitstream.BitReader( out )
	j: usize = 0
	with compiler.panic_arithmetic( 'bounded by 5, cannot overflow' ):
		while j < usize( 5 ):
			decoded: u16 = dec.decode( r ).unwrap( 'decode' )
			if decoded != u16( j ):
				return 1
			j += 1
	return 0
''' ),
			( 'build_lengths_single_symbol_degenerate_case', '''
import huffman
import bitstream

def main() -> i32:
	freqs: UnsafeList[u32] = UnsafeList[u32]( usize( 3 ))
	freqs.append( u32( 0 )).unwrap( 'x' )
	freqs.append( u32( 42 )).unwrap( 'x' )
	freqs.append( u32( 0 )).unwrap( 'x' )

	lengths: UnsafeList[u8] = huffman.build_code_lengths_from_frequencies( freqs, u8( 15 )).unwrap( 'x' )
	only: u8 = lengths.__getitem__( usize( 1 )).unwrap( 'x' )
	if only != 1:
		return 1

	enc: huffman.HuffmanEncoder = huffman.HuffmanEncoder( lengths ).unwrap( 'x' )
	dec: huffman.HuffmanDecoder = huffman.HuffmanDecoder( lengths ).unwrap( 'x' )
	w = bitstream.BitWriter()
	enc.write_symbol( w, u16( 1 )).unwrap( 'x' )
	enc.write_symbol( w, u16( 1 )).unwrap( 'x' )
	out: bytes = w.finish()
	r = bitstream.BitReader( out )
	a: u16 = dec.decode( r ).unwrap( 'x' )
	b: u16 = dec.decode( r ).unwrap( 'x' )
	if a != 1 or b != 1:
		return 2
	return 0
''' ),
		] )
