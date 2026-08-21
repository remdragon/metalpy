import unittest
import zlib

import emitter_c
import test_support
from compiler import Compiler
from discovery import Discovery


def _mpy_bytes_literal( data: bytes ) -> str:
	''' renders `data` as a MetalPy b'...' byte literal (hex-escaped, safe for
	any byte value) for embedding real zlib-produced fixtures directly in
	compiled test source. '''
	body = ''.join( f'\\x{b:02x}' for b in data )
	return f"b'{body}'"


class DeflateTests( test_support.RealCompileMixin, unittest.TestCase ):
	''' Real compile-and-run coverage for lib/deflate.py, cross-checked
	against Python's own zlib (a genuinely independent implementation) in
	both directions - self-round-trip alone can't catch a self-consistent-
	but-spec-wrong codec. '''

	def setUp( self ) -> None:
		self.discovery = Discovery( import_builtins = True )
		self.compiler = Compiler( self.discovery )

	@unittest.skipUnless( test_support.HAS_CC, 'no C compiler (clang/gcc/msvc) found - skipping' )
	def test_programs_compile_and_run( self ) -> None:
		repetitive = b'the quick brown fox jumps over the lazy dog. ' * 20
		short_text = b'hello world'
		binary_ish = bytes( range( 256 )) * 4  # exercises every literal byte value

		# real zlib, default level -> whichever block type(s) zlib itself
		# chooses (typically dynamic Huffman for anything nontrivial)
		zlib_default = zlib.compressobj( 6, zlib.DEFLATED, -15 )
		repetitive_deflate = zlib_default.compress( repetitive ) + zlib_default.flush()

		zlib_default2 = zlib.compressobj( 6, zlib.DEFLATED, -15 )
		binary_deflate = zlib_default2.compress( binary_ish ) + zlib_default2.flush()

		# level 0 -> zlib emits STORED blocks, exercising that decode path
		zlib_stored = zlib.compressobj( 0, zlib.DEFLATED, -15 )
		short_stored_deflate = zlib_stored.compress( short_text ) + zlib_stored.flush()

		self.assert_programs_run([
			( 'self_round_trip_repetitive', f'''
import deflate

def main() -> i32:
	original: bytes = {_mpy_bytes_literal( repetitive )}
	compressed: bytes = deflate.compress( original ).unwrap( 'compress' )
	back: bytes = deflate.decompress_exact( compressed, len( original )).unwrap( 'decompress' )
	if len( back ) != len( original ):
		return 1
	op: ConstPtr[u8] = original.get_const_ptr()
	bp: ConstPtr[u8] = back.get_const_ptr()
	i: usize = 0
	with compiler.panic_arithmetic( 'bounded by len(original), cannot overflow' ):
		while i < len( original ):
			if op[i] != bp[i]:
				return 2
			i += 1
	return 0
''' ),
			( 'decode_real_zlib_dynamic_block_repetitive', f'''
import deflate

def main() -> i32:
	compressed: bytes = {_mpy_bytes_literal( repetitive_deflate )}
	expected: bytes = {_mpy_bytes_literal( repetitive )}
	back: bytes = deflate.decompress_exact( compressed, len( expected )).unwrap( 'decompress' )
	if len( back ) != len( expected ):
		return 1
	ep: ConstPtr[u8] = expected.get_const_ptr()
	bp: ConstPtr[u8] = back.get_const_ptr()
	i: usize = 0
	with compiler.panic_arithmetic( 'bounded by len(expected), cannot overflow' ):
		while i < len( expected ):
			if ep[i] != bp[i]:
				return 2
			i += 1
	return 0
''' ),
			( 'decode_real_zlib_all_byte_values', f'''
import deflate

def main() -> i32:
	compressed: bytes = {_mpy_bytes_literal( binary_deflate )}
	expected: bytes = {_mpy_bytes_literal( binary_ish )}
	back: bytes = deflate.decompress_exact( compressed, len( expected )).unwrap( 'decompress' )
	if len( back ) != len( expected ):
		return 1
	ep: ConstPtr[u8] = expected.get_const_ptr()
	bp: ConstPtr[u8] = back.get_const_ptr()
	i: usize = 0
	with compiler.panic_arithmetic( 'bounded by len(expected), cannot overflow' ):
		while i < len( expected ):
			if ep[i] != bp[i]:
				return 2
			i += 1
	return 0
''' ),
			( 'decode_real_zlib_stored_block', f'''
import deflate

def main() -> i32:
	compressed: bytes = {_mpy_bytes_literal( short_stored_deflate )}
	expected: bytes = {_mpy_bytes_literal( short_text )}
	back: bytes = deflate.decompress_exact( compressed, len( expected )).unwrap( 'decompress' )
	if len( back ) != len( expected ):
		return 1
	ep: ConstPtr[u8] = expected.get_const_ptr()
	bp: ConstPtr[u8] = back.get_const_ptr()
	i: usize = 0
	with compiler.panic_arithmetic( 'bounded by len(expected), cannot overflow' ):
		while i < len( expected ):
			if ep[i] != bp[i]:
				return 2
			i += 1
	return 0
''' ),
			( 'decompress_unbounded_matches_exact', f'''
import deflate

def main() -> i32:
	compressed: bytes = {_mpy_bytes_literal( repetitive_deflate )}
	expected: bytes = {_mpy_bytes_literal( repetitive )}
	back: bytes = deflate.decompress_unbounded( compressed ).unwrap( 'decompress_unbounded' )
	if len( back ) != len( expected ):
		return 1
	return 0
''' ),
			( 'truncated_stream_is_error_not_crash', f'''
import deflate

def main() -> i32:
	full: bytes = {_mpy_bytes_literal( repetitive_deflate )}
	raw: bytearray = bytearray( len( full ))
	fp: ConstPtr[u8] = full.get_const_ptr()
	rp: Ptr[u8] = raw.get_ptr()
	i: usize = 0
	with compiler.panic_arithmetic( 'bounded by len(full), cannot overflow' ):
		while i < len( full ):
			rp[i] = fp[i]
			i += 1
	rp[0] = rp[0] ^ 0xFF  # corrupt the BFINAL/BTYPE/first Huffman bits
	corrupted: bytes = bytes.from_bytearray( move( raw ))
	if deflate.decompress_unbounded( corrupted ).is_ok():
		return 1  # not a guaranteed failure for every possible corruption, but this one is
	return 0
''' ),
			( 'empty_input_round_trips', '''
import deflate

def main() -> i32:
	empty: bytes = bytes.from_bytearray( move( bytearray( 0 )))
	compressed: bytes = deflate.compress( empty ).unwrap( 'compress' )
	back: bytes = deflate.decompress_exact( compressed, usize( 0 )).unwrap( 'decompress' )
	if len( back ) != usize( 0 ):
		return 1
	return 0
''' ),
		] )

	@unittest.skipUnless( test_support.HAS_CC, 'no C compiler (clang/gcc/msvc) found - skipping' )
	def test_encoder_output_decodes_with_real_zlib( self ) -> None:
		''' validates the ENCODER against a real independent decoder - pure
		self-round-trip can never catch a spec-wrong-but-self-consistent
		bitstream. The compiled program hex-encodes its own compressed
		output to stdout; this test decodes that hex with Python's own
		zlib.decompress(wbits=-15) and compares to the original. '''
		original = b'the quick brown fox jumps over the lazy dog. ' * 20
		source = f'''
import deflate
import binascii

def main() -> i32:
	original: bytes = {_mpy_bytes_literal( original )}
	compressed: bytes = deflate.compress( original ).unwrap( 'compress' )
	hex_bytes: bytes = binascii.hexlify( compressed )
	hex_str: str = hex_bytes.decode().unwrap( 'decode' )
	print( hex_str )
	return 0
'''
		compiler = self._compile_source( source )
		result = self._build_and_run( compiler, emitter_c.emit_c( compiler ), timeout = None )
		self.assertEqual( result.returncode, 0, f'program failed: {result.stdout!r} {result.stderr!r}' )
		hex_str = result.stdout.decode( 'utf-8' ).strip()
		compressed = bytes.fromhex( hex_str )
		decompressed = zlib.decompress( compressed, -15 )
		self.assertEqual( decompressed, original )
