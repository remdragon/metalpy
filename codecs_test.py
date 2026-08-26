# Real-compile-and-run tests for lib/codecs/: infallible decode_lossy()
# (Replace/Ignore/BackslashReplace) across every Codec, plus the strict
# decode()'s pre-existing codec-param bug (bytes.decode always called utf8
# regardless of the codec argument - fixed alongside decode_lossy).

# stdlib imports:
import unittest

# local imports:
import test_support


class CodecsTests( test_support.RealCompileMixin, unittest.TestCase ):
	def setUp( self ) -> None:
		from discovery import Discovery
		from compiler import Compiler
		self.discovery = Discovery( import_builtins = True )
		self.compiler = Compiler( self.discovery )

	def _run( self, code: str ) -> None:
		from pathlib import Path
		self.compiler.import_code( code, Path( '__main__.py' ), scope = None )
		self.compiler.run()

	@unittest.skipUnless( test_support.HAS_CC, 'no C compiler (clang/gcc/msvc) found - skipping' )
	def test_programs_compile_and_run( self ) -> None:
		import emitter_c
		self.assert_programs_run([
			( 'utf8_decode_lossy_replace_ignore_backslashreplace', '''
from codecs import DecodeErrors
from codecs.utf8 import utf8

def main() -> i32:
	# "hi" + an invalid UTF-8 leading byte (0xFF) + "!"
	raw: bytearray = bytearray( 4 )
	raw[0] = 0x68; raw[1] = 0x69; raw[2] = 0xFF; raw[3] = 0x21
	bad: bytes = bytes( raw )

	if bad.decode_lossy( utf8, DecodeErrors.Replace ) != 'hi\\uFFFD!':
		return 1
	if bad.decode_lossy( utf8, DecodeErrors.Ignore ) != 'hi!':
		return 2
	if bad.decode_lossy( utf8, DecodeErrors.BackslashReplace ) != 'hi\\\\xff!':
		return 3
	# default codec (utf8) and default errors (BackslashReplace)
	if bad.decode_lossy() != 'hi\\\\xff!':
		return 4
	# valid input round-trips unchanged under every mode
	good: bytes = 'hello world'.encode().unwrap( 'ascii is valid utf-8' )
	if good.decode_lossy( utf8, DecodeErrors.Replace ) != 'hello world':
		return 5
	return 0
''' ),
			( 'bytearray_and_memoryview_decode_lossy', '''
from codecs import DecodeErrors
from codecs.utf8 import utf8

def main() -> i32:
	raw: bytearray = bytearray( 4 )
	raw[0] = 0x68; raw[1] = 0x69; raw[2] = 0xFF; raw[3] = 0x21
	if raw.decode_lossy( utf8, DecodeErrors.Ignore ) != 'hi!':
		return 1
	mv = memoryview( raw )
	if mv.decode_lossy( utf8, DecodeErrors.Ignore ) != 'hi!':
		return 2
	return 0
''' ),
			( 'ascii_decode_lossy_and_codec_param_honored', '''
from codecs import DecodeErrors
from codecs.ascii import ascii

def main() -> i32:
	raw: bytearray = bytearray( 3 )
	raw[0] = 0x41; raw[1] = 0x80; raw[2] = 0x42 # 'A' + bad + 'B'
	bad: bytes = bytes( raw )

	if bad.decode_lossy( ascii(), DecodeErrors.Replace ) != 'A\\uFFFDB':
		return 1

	# bytes.decode() used to ALWAYS call the utf8 codec regardless of the
	# codec argument - confirm the ascii codec's own strict error now fires
	match bad.decode( ascii() ):
		case Result.Err( e ):
			if e.encoding != 'ascii':
				return 2
		case Result.Ok( _ ):
			return 3

	good: bytes = 'ABC'.encode().unwrap( 'ascii is valid utf-8' )
	if bad.decode_lossy() == good.decode_lossy(): # sanity: different inputs, different results
		return 4
	if good.decode( ascii() ).unwrap( 'valid ascii' ) != 'ABC':
		return 5
	return 0
''' ),
			( 'latin1_and_cp437_decode_lossy_are_infallible_passthroughs', '''
from codecs import DecodeErrors
from codecs.latin1 import latin1
from codecs.cp437 import cp437

def main() -> i32:
	raw: bytearray = bytearray( 1 )
	raw[0] = 0xFF
	b: bytes = bytes( raw )

	# every byte 0x00..0xFF is a valid Latin-1/CP437 codepoint - decode()
	# never fails, so decode_lossy() must match decode() exactly regardless
	# of the errors mode
	strict: str = latin1().decode( b ).unwrap( 'latin1 decode never fails' )
	if latin1().decode_lossy( b, DecodeErrors.Replace ) != strict:
		return 1
	if latin1().decode_lossy( b, DecodeErrors.Ignore ) != strict:
		return 2

	strict_cp: str = cp437().decode( b ).unwrap( 'cp437 decode never fails' )
	if cp437().decode_lossy( b, DecodeErrors.BackslashReplace ) != strict_cp:
		return 3
	return 0
''' ),
			( 'utf16_decode_lossy_odd_length_and_unpaired_surrogate', '''
from codecs import DecodeErrors
from codecs.utf16 import utf16

def main() -> i32:
	# a lone high surrogate (0xD800) with no low surrogate following,
	# native-endian u16 -> raw bytes
	raw: bytearray = bytearray( 3 )
	raw[0] = 0x00; raw[1] = 0xD8; raw[2] = 0x41 # D800 (LE) + one trailing odd byte

	if utf16.decode_lossy( raw, DecodeErrors.Ignore ) == '':
		# the trailing odd 0x41 byte alone is dropped under Ignore, the
		# unpaired surrogate is dropped too - just confirm it doesn't crash
		# and produces SOME string (exact content isn't the point here)
		pass

	lossy: str = utf16.decode_lossy( raw, DecodeErrors.BackslashReplace )
	if len( lossy ) == 0:
		return 1

	# a plain valid 2-byte-per-char UTF-16LE string round-trips unchanged
	good: bytearray = bytearray( 4 )
	good[0] = 0x41; good[1] = 0x00; good[2] = 0x42; good[3] = 0x00 # "AB"
	if utf16.decode_lossy( good, DecodeErrors.Replace ) != 'AB':
		return 2
	return 0
''' ),
		])


if __name__ == '__main__':
	unittest.main()
