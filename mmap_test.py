# Real-compile-and-run tests for lib/mmap.py: memory-mapped file access
# (Windows CreateFileMapping/MapViewOfFile, POSIX mmap(2)), plus its
# integration with the memoryview builtin (lib/builtins/__memoryview.py)
# - the exact `with mmap.mmap(...) as mm: with memoryview(mm) as mv: ...`
# shape grap.mpy (the motivating real-world port) uses.
#
# mmap.mmap(...) is a FALLIBLE constructor (`Result[mmap.mmap, OSError]`
# at the call site - see SYNTAX.md's "Fallible __init__() Construction"),
# not a raising one like real Python's own mmap.mmap().

# stdlib imports:
import os
from pathlib import Path
import unittest

# local imports:
import test_support


class MmapTests( test_support.RealCompileMixin, unittest.TestCase ):
	def setUp( self ) -> None:
		from discovery import Discovery
		from compiler import Compiler
		self.discovery = Discovery( import_builtins = True )
		self.compiler = Compiler( self.discovery )

		import tempfile
		self._tmpdir_ctx = tempfile.TemporaryDirectory()
		self.tmpdir = self._tmpdir_ctx.name
		self.fixture_path = os.path.join( self.tmpdir, 'mmap_fixture.txt' )
		with open( self.fixture_path, 'wb' ) as f:
			f.write( b'Hello, mmap world!' )  # 18 bytes

	def tearDown( self ) -> None:
		self._tmpdir_ctx.cleanup()

	def _run( self, code: str ) -> None:
		self.compiler.import_code( code, Path( '__main__.py' ), scope = None )
		self.compiler.run()

	@unittest.skipUnless( test_support.HAS_CC, 'no C compiler (clang/gcc/msvc) found - skipping' )
	def test_read_whole_file_length_zero_maps_current_size( self ) -> None:
		import emitter_c
		path_literal = repr( self.fixture_path )
		self._run( f'''
import mmap

def main() -> i32:
	r = File.binary_reader( {path_literal} )
	f = r.unwrap( 'open failed' )
	defer( f.close() )
	result: Result[mmap.mmap, OSError] = mmap.mmap( f.fileno(), 0, access = mmap.ACCESS_READ )
	mm: mmap.mmap = result.unwrap( 'mmap failed' )
	if len( mm ) != 18:
		return 1
	p: ConstPtr[u8] = mm.get_const_ptr()
	if p[0] != 72:  # 'H'
		return 2
	if p[17] != 33:  # '!'
		return 3
	mm.close()
	mm.close()  # idempotent - must not double-unmap/crash
	return 0
''' )
		self.assertEqual( self.discovery.errors.errors, [] )
		self._assert_compiles_and_runs( emitter_c.emit_c( self.compiler ), expected_exit = 0 )

	@unittest.skipUnless( test_support.HAS_CC, 'no C compiler (clang/gcc/msvc) found - skipping' )
	def test_for_loop_iterates_bytes( self ) -> None:
		''' mmap conforms to Sequence[u8]/Iterable[u8] - a plain for loop
		over it (not just index access) must work. '''
		import emitter_c
		path_literal = repr( self.fixture_path )
		self._run( f'''
import compiler
import mmap

def main() -> i32:
	r = File.binary_reader( {path_literal} )
	f = r.unwrap( 'open failed' )
	defer( f.close() )
	mm: mmap.mmap = mmap.mmap( f.fileno(), 0, access = mmap.ACCESS_READ ).unwrap( 'mmap failed' )
	total: i32 = 0
	with compiler.wrap_arithmetic:
		for b in mm:
			total += i32( b )
	if total != 1620: # sum of 'Hello, mmap world!'s byte values
		return 1
	return 0
''' )
		self.assertEqual( self.discovery.errors.errors, [] )
		self._assert_compiles_and_runs( emitter_c.emit_c( self.compiler ), expected_exit = 0 )

	@unittest.skipUnless( test_support.HAS_CC, 'no C compiler (clang/gcc/msvc) found - skipping' )
	def test_explicit_length_maps_a_prefix( self ) -> None:
		import emitter_c
		path_literal = repr( self.fixture_path )
		self._run( f'''
import mmap

def main() -> i32:
	r = File.binary_reader( {path_literal} )
	f = r.unwrap( 'open failed' )
	defer( f.close() )
	result: Result[mmap.mmap, OSError] = mmap.mmap( f.fileno(), 5, access = mmap.ACCESS_READ )
	mm: mmap.mmap = result.unwrap( 'mmap failed' )
	if len( mm ) != 5:
		return 1
	p: ConstPtr[u8] = mm.get_const_ptr()
	if p[4] != 111:  # 'o' (end of "Hello")
		return 2
	return 0
''' )
		self.assertEqual( self.discovery.errors.errors, [] )
		self._assert_compiles_and_runs( emitter_c.emit_c( self.compiler ), expected_exit = 0 )

	@unittest.skipUnless( test_support.HAS_CC, 'no C compiler (clang/gcc/msvc) found - skipping' )
	def test_with_memoryview_grap_style( self ) -> None:
		''' the exact shape grap.mpy's own port uses: with mmap.mmap(...) as
		mm: with memoryview(mm) as mv: ... including mv[a:b] slicing. '''
		import emitter_c
		path_literal = repr( self.fixture_path )
		self._run( f'''
import mmap

def main() -> i32:
	r = File.binary_reader( {path_literal} )
	f = r.unwrap( 'open failed' )
	defer( f.close() )
	mm: mmap.mmap = mmap.mmap( f.fileno(), 0, access = mmap.ACCESS_READ ).unwrap( 'mmap failed' )
	with memoryview( mm ) as mv:
		whole: memoryview = mv[:len( mm )]
		if len( whole ) != 18:
			return 1
		sub: memoryview = whole[7:11]
		if len( sub ) != 4:
			return 2
		s0: u8 = sub.__getitem__( 0 ).unwrap( 'idx' )
		if s0 != 109:  # 'm' (start of "mmap")
			return 3
	return 0
''' )
		self.assertEqual( self.discovery.errors.errors, [] )
		self._assert_compiles_and_runs( emitter_c.emit_c( self.compiler ), expected_exit = 0 )

	@unittest.skipUnless( test_support.HAS_CC, 'no C compiler (clang/gcc/msvc) found - skipping' )
	def test_access_write_persists_to_the_file( self ) -> None:
		import emitter_c
		path_literal = repr( self.fixture_path )
		self._run( f'''
import mmap

def main() -> i32:
	r = File.binary_read_writer( {path_literal}, truncate = False )
	f = r.unwrap( 'open failed' )
	defer( f.close() )
	mm: mmap.mmap = mmap.mmap( f.fileno(), 0, access = mmap.ACCESS_WRITE ).unwrap( 'mmap failed' )
	p: Ptr[u8] = mm.get_ptr()
	p[0] = 74  # 'J' - was 'H'
	mm.close()
	return 0
''' )
		self.assertEqual( self.discovery.errors.errors, [] )
		self._assert_compiles_and_runs( emitter_c.emit_c( self.compiler ), expected_exit = 0 )
		with open( self.fixture_path, 'rb' ) as f:
			content = f.read()
		self.assertEqual( content[:1], b'J', 'ACCESS_WRITE mmap write did not reach the underlying file' )

	@unittest.skipUnless( test_support.HAS_CC, 'no C compiler (clang/gcc/msvc) found - skipping' )
	def test_access_copy_does_not_persist_to_the_file( self ) -> None:
		import emitter_c
		path_literal = repr( self.fixture_path )
		self._run( f'''
import mmap

def main() -> i32:
	r = File.binary_read_writer( {path_literal}, truncate = False )
	f = r.unwrap( 'open failed' )
	defer( f.close() )
	mm: mmap.mmap = mmap.mmap( f.fileno(), 0, access = mmap.ACCESS_COPY ).unwrap( 'mmap failed' )
	p: Ptr[u8] = mm.get_ptr()
	p[0] = 90  # 'Z' - copy-on-write, must never reach the file
	mm.close()
	return 0
''' )
		self.assertEqual( self.discovery.errors.errors, [] )
		self._assert_compiles_and_runs( emitter_c.emit_c( self.compiler ), expected_exit = 0 )
		with open( self.fixture_path, 'rb' ) as f:
			content = f.read()
		self.assertEqual( content[:1], b'H', 'ACCESS_COPY mmap write leaked through to the underlying file' )


if __name__ == '__main__':
	unittest.main()
