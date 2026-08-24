# Real-compile-and-run tests for str.from_utf16() (lib/builtins/__init__.py)
# - builds a null-terminated u16 buffer by hand (no Windows API dependency)
# so this exercises the decode path on any platform.

import unittest

import test_support


class StrFromUtf16Tests( test_support.RealCompileMixin, unittest.TestCase ):

	def setUp( self ) -> None:
		from discovery import Discovery
		from compiler import Compiler
		self.discovery = Discovery( import_builtins = True )
		self.compiler = Compiler( self.discovery )

	@unittest.skipUnless( test_support.HAS_CC, 'no C compiler (clang/gcc/msvc) found - skipping' )
	def test_programs_compile_and_run( self ) -> None:
		self.assert_programs_run([
			( 'ascii_roundtrip', '''
def main() -> i32:
	buf = bytearray( 6 ) # "AB" + null terminator, 3 u16 units
	u16buf: Ptr[u16] = compiler.cast( Ptr[u16], buf.get_ptr() )
	u16buf[0] = 0x41
	u16buf[1] = 0x42
	u16buf[2] = 0

	s: str = str.from_utf16( compiler.cast( ConstPtr[u16], buf.get_const_ptr() ), 10 ).unwrap( 'decode failed' )
	if s != 'AB':
		return 1
	return 0
''' ),
			( 'surrogate_pair_roundtrip', '''
def main() -> i32:
	buf = bytearray( 6 ) # U+1F600 (surrogate pair) + null terminator, 3 u16 units
	u16buf: Ptr[u16] = compiler.cast( Ptr[u16], buf.get_ptr() )
	u16buf[0] = 0xD83D
	u16buf[1] = 0xDE00
	u16buf[2] = 0

	s: str = str.from_utf16( compiler.cast( ConstPtr[u16], buf.get_const_ptr() ), 10 ).unwrap( 'decode failed' )
	if s != '\U0001F600':
		return 1
	return 0
''' ),
			( 'missing_null_terminator_errors', '''
def main() -> i32:
	buf = bytearray( 4 ) # "AB", no null terminator - max_len exactly matches
	u16buf: Ptr[u16] = compiler.cast( Ptr[u16], buf.get_ptr() )
	u16buf[0] = 0x41
	u16buf[1] = 0x42

	match str.from_utf16( compiler.cast( ConstPtr[u16], buf.get_const_ptr() ), 2 ):
		case Result.Ok( _ ):
			return 1
		case Result.Err( _ ):
			pass
	return 0
''' ),
		])
