import unittest

import test_support
from compiler import Compiler
from discovery import Discovery


class BitstreamTests( test_support.RealCompileMixin, unittest.TestCase ):
	''' Real compile-and-run coverage for lib/bitstream.py. '''

	def setUp( self ) -> None:
		self.discovery = Discovery( import_builtins = True )
		self.compiler = Compiler( self.discovery )

	@unittest.skipUnless( test_support.HAS_CC, 'no C compiler (clang/gcc/msvc) found - skipping' )
	def test_programs_compile_and_run( self ) -> None:
		self.assert_programs_run([
			( 'hand_computed_lsb_first_byte', '''
import bitstream

def main() -> i32:
	w = bitstream.BitWriter()
	w.write_bits( u32( 5 ), u32( 3 ) )  # 0b101, bits 1,0,1 at positions 0,1,2
	w.write_bits( u32( 3 ), u32( 2 ) )  # 0b11, bits 1,1 at positions 3,4
	out: bytes = w.finish()
	if len( out ) != 1:
		return 1
	p: ConstPtr[u8] = out.get_const_ptr()
	# expected byte, LSB-first: pos0=1 pos1=0 pos2=1 pos3=1 pos4=1 pos5..7=0
	# = 0b00011101 = 0x1D
	if p[0] != 0x1D:
		return 2
	return 0
''' ),
			( 'write_then_read_round_trip', '''
import bitstream

def main() -> i32:
	w = bitstream.BitWriter()
	w.write_bits( u32( 5 ), u32( 3 ) )
	w.write_bits( u32( 3 ), u32( 2 ) )
	w.write_bits( u32( 0x2A ), u32( 7 ) )  # crosses a byte boundary
	out: bytes = w.finish()

	r = bitstream.BitReader( out )
	a: u32 = r.read_bits( u32( 3 ) ).unwrap( 'a' )
	b: u32 = r.read_bits( u32( 2 ) ).unwrap( 'b' )
	c: u32 = r.read_bits( u32( 7 ) ).unwrap( 'c' )
	if a != 5:
		return 1
	if b != 3:
		return 2
	if c != 0x2A:
		return 3
	return 0
''' ),
			( 'align_to_byte_and_raw_bytes_round_trip', '''
import bitstream

def main() -> i32:
	w = bitstream.BitWriter()
	w.write_bits( u32( 1 ), u32( 3 ) )  # leaves a partial byte
	w.align_to_byte()
	raw: bytearray = bytearray( 3 )
	rp: Ptr[u8] = raw.get_ptr()
	rp[0] = 0xAA
	rp[1] = 0xBB
	rp[2] = 0xCC
	w.write_aligned_bytes( raw.get_const_ptr(), usize( 3 ) )
	out: bytes = w.finish()
	if len( out ) != 4:  # 1 partial-flushed byte + 3 raw bytes
		return 1

	r = bitstream.BitReader( out )
	first: u32 = r.read_bits( u32( 3 ) ).unwrap( 'first' )
	if first != 1:
		return 2
	r.align_to_byte()
	back: bytearray = bytearray( 3 )
	bp: Ptr[u8] = back.get_ptr()
	r.read_aligned_bytes( bp, usize( 3 ) ).unwrap( 'read_aligned_bytes' )
	if bp[0] != 0xAA or bp[1] != 0xBB or bp[2] != 0xCC:
		return 3
	return 0
''' ),
			( 'read_past_end_is_error', '''
import bitstream

def main() -> i32:
	w = bitstream.BitWriter()
	w.write_bits( u32( 1 ), u32( 1 ) )
	out: bytes = w.finish()  # exactly 1 byte

	r = bitstream.BitReader( out )
	r.read_bits( u32( 8 ) ).unwrap( 'first 8 bits, exactly the whole stream' )
	match r.read_bits( u32( 1 ) ):
		case Result.Err( e ):
			if str( e ) != 'read_bits: unexpected end of stream':
				return 1
			if f'{e}' != 'read_bits: unexpected end of stream':
				return 2
			if e.__repr__() != "BitstreamError('read_bits: unexpected end of stream')":
				return 3
			return 0
		case Result.Ok( _ ):
			return 4
''' ),
			( 'bits_remaining_tracks_consumption', '''
import bitstream

def main() -> i32:
	w = bitstream.BitWriter()
	w.write_bits( u32( 0 ), u32( 16 ) )
	out: bytes = w.finish()

	r = bitstream.BitReader( out )
	if r.bits_remaining() != usize( 16 ):
		return 1
	r.read_bits( u32( 5 ) ).unwrap( 'x' )
	if r.bits_remaining() != usize( 11 ):
		return 2
	return 0
''' ),
		] )
