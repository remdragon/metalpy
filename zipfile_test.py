import io as _io
import tempfile
import unittest
import zipfile as _pyzipfile
from pathlib import Path

import emitter_c
import test_support
from compiler import Compiler
from discovery import Discovery


def _mpy_str_literal( s: str ) -> str:
	escaped = s.replace( '\\', '\\\\' ).replace( "'", "\\'" )
	return f"'{escaped}'"


class ZipfileTests( test_support.RealCompileMixin, unittest.TestCase ):
	''' Real compile-and-run coverage for lib/zipfile.py. '''

	def setUp( self ) -> None:
		self.discovery = Discovery( import_builtins = True )
		self.compiler = Compiler( self.discovery )

	@unittest.skipUnless( test_support.HAS_CC, 'no C compiler (clang/gcc/msvc) found - skipping' )
	def test_programs_compile_and_run( self ) -> None:
		self.assert_programs_run([
			( 'write_then_read_round_trip', '''
import zipfile

def main() -> i32:
	w: zipfile.ZipWriter = zipfile.ZipFile.create( 'test.zip' ).unwrap( 'create' )
	w.writestr( 'hello.txt', 'hello world'.encode().unwrap( 'x' )).unwrap( 'writestr' )
	w.close().unwrap( 'close' )

	r: zipfile.ZipReader = zipfile.ZipFile.open( 'test.zip' ).unwrap( 'open' )
	names: list[str] = r.namelist()
	if len( names ) != usize( 1 ):
		return 1
	data: bytes = r.read( 'hello.txt' ).unwrap( 'read' )
	s: str = data.decode().unwrap( 'decode' )
	if s != 'hello world':
		return 2
	return 0
''' ),
			( 'multi_entry_mixed_compression', '''
import zipfile

def main() -> i32:
	w: zipfile.ZipWriter = zipfile.ZipFile.create( 'multi.zip' ).unwrap( 'create' )
	w.writestr( 'stored.txt', 'aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa'.encode().unwrap( 'x' ), compress_type = zipfile.ZIP_STORED ).unwrap( 'writestr stored' )
	w.writestr( 'deflated.txt', 'bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb'.encode().unwrap( 'x' ), compress_type = zipfile.ZIP_DEFLATED ).unwrap( 'writestr deflated' )
	w.writestr( 'empty.txt', bytes.from_bytearray( move( bytearray( 0 )))).unwrap( 'writestr empty' )
	w.close().unwrap( 'close' )

	r: zipfile.ZipReader = zipfile.ZipFile.open( 'multi.zip' ).unwrap( 'open' )
	names: list[str] = r.namelist()
	if len( names ) != usize( 3 ):
		return 1

	stored: bytes = r.read( 'stored.txt' ).unwrap( 'read stored' )
	if stored.decode().unwrap( 'x' ) != 'aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa':
		return 2
	deflated: bytes = r.read( 'deflated.txt' ).unwrap( 'read deflated' )
	if deflated.decode().unwrap( 'x' ) != 'bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb':
		return 3
	empty: bytes = r.read( 'empty.txt' ).unwrap( 'read empty' )
	if len( empty ) != usize( 0 ):
		return 4

	info: zipfile.ZipInfo = r.getinfo( 'stored.txt' ).unwrap( 'getinfo' )
	if info.compress_type != zipfile.ZIP_STORED:
		return 5
	info2: zipfile.ZipInfo = r.getinfo( 'deflated.txt' ).unwrap( 'getinfo2' )
	if info2.compress_type != zipfile.ZIP_DEFLATED:
		return 6
	return 0
''' ),
			( 'read_nonexistent_entry_is_error', '''
import zipfile

def main() -> i32:
	w: zipfile.ZipWriter = zipfile.ZipFile.create( 'empty_archive.zip' ).unwrap( 'create' )
	w.writestr( 'real.txt', 'x'.encode().unwrap( 'x' )).unwrap( 'writestr' )
	w.close().unwrap( 'close' )

	r: zipfile.ZipReader = zipfile.ZipFile.open( 'empty_archive.zip' ).unwrap( 'open' )
	if r.read( 'does_not_exist.txt' ).is_ok():
		return 1
	if r.getinfo( 'does_not_exist.txt' ).is_ok():
		return 2
	return 0
''' ),
			( 'compressible_data_actually_shrinks', '''
import zipfile

def main() -> i32:
	original: str = 'the quick brown fox jumps over the lazy dog. ' * 20
	original_bytes: bytes = original.encode().unwrap( 'x' )
	w: zipfile.ZipWriter = zipfile.ZipFile.create( 'compressible.zip' ).unwrap( 'create' )
	w.writestr( 'big.txt', original_bytes, compress_type = zipfile.ZIP_DEFLATED ).unwrap( 'writestr' )
	w.close().unwrap( 'close' )

	r: zipfile.ZipReader = zipfile.ZipFile.open( 'compressible.zip' ).unwrap( 'open' )
	info: zipfile.ZipInfo = r.getinfo( 'big.txt' ).unwrap( 'getinfo' )
	with compiler.panic_arithmetic( 'len fits comfortably in u32 for this test input' ):
		orig_len: u32 = u32( len( original_bytes ))
	if info.compress_size >= orig_len:
		return 1
	back: bytes = r.read( 'big.txt' ).unwrap( 'read' )
	if back.decode().unwrap( 'x' ) != original:
		return 2
	return 0
''' ),
		] )

	@unittest.skipUnless( test_support.HAS_CC, 'no C compiler (clang/gcc/msvc) found - skipping' )
	def test_reads_real_zip_from_python_zipfile( self ) -> None:
		''' cross-check the READ direction: a real .zip built by Python's own
		zipfile module (mixing STORED and DEFLATED entries) must open and
		extract correctly through lib/zipfile.py's ZipReader. '''
		with tempfile.TemporaryDirectory() as tmp:
			zip_path = Path( tmp ) / 'fixture.zip'
			with _pyzipfile.ZipFile( zip_path, 'w' ) as zf:
				zf.writestr( 'stored.txt', 'hello from python stored', _pyzipfile.ZIP_STORED )
				zf.writestr( 'deflated.txt', 'hello from python deflated ' * 50, _pyzipfile.ZIP_DEFLATED )

			zip_path_posix = zip_path.as_posix()
			source = f'''
import zipfile

def main() -> i32:
	r: zipfile.ZipReader = zipfile.ZipFile.open( {_mpy_str_literal( zip_path_posix )} ).unwrap( 'open' )
	names: list[str] = r.namelist()
	if len( names ) != usize( 2 ):
		return 1
	a: bytes = r.read( 'stored.txt' ).unwrap( 'read stored' )
	if a.decode().unwrap( 'x' ) != 'hello from python stored':
		return 2
	b: bytes = r.read( 'deflated.txt' ).unwrap( 'read deflated' )
	if b.decode().unwrap( 'x' ) != 'hello from python deflated ' * 50:
		return 3
	return 0
'''
			self._assert_compiles_and_runs_isolated( source )

	@unittest.skipUnless( test_support.HAS_CC, 'no C compiler (clang/gcc/msvc) found - skipping' )
	def test_writes_zip_readable_by_python_zipfile( self ) -> None:
		''' cross-check the WRITE direction: an archive built by
		lib/zipfile.py's ZipWriter must open correctly with Python's own
		independent zipfile implementation. '''
		with tempfile.TemporaryDirectory() as tmp:
			zip_path = Path( tmp ) / 'written.zip'
			zip_path_posix = zip_path.as_posix()
			source = f'''
import zipfile

def main() -> i32:
	w: zipfile.ZipWriter = zipfile.ZipFile.create( {_mpy_str_literal( zip_path_posix )} ).unwrap( 'create' )
	w.writestr( 'a.txt', 'stored content here'.encode().unwrap( 'x' ), compress_type = zipfile.ZIP_STORED ).unwrap( 'writestr a' )
	w.writestr( 'b.txt', ( 'deflated content ' * 50 ).encode().unwrap( 'x' ), compress_type = zipfile.ZIP_DEFLATED ).unwrap( 'writestr b' )
	w.close().unwrap( 'close' )
	return 0
'''
			self._assert_compiles_and_runs_isolated( source )

			self.assertTrue( zip_path.exists(), 'ZipWriter did not create the archive file' )
			with _pyzipfile.ZipFile( zip_path, 'r' ) as zf:
				bad_entry = zf.testzip()
				self.assertIsNone( bad_entry, f'zipfile module reports a bad CRC in {bad_entry!r}' )
				names = set( zf.namelist() )
				self.assertEqual( names, { 'a.txt', 'b.txt' } )
				self.assertEqual( zf.read( 'a.txt' ).decode(), 'stored content here' )
				self.assertEqual( zf.read( 'b.txt' ).decode(), 'deflated content ' * 50 )
				info_a = zf.getinfo( 'a.txt' )
				self.assertEqual( info_a.compress_type, _pyzipfile.ZIP_STORED )
				info_b = zf.getinfo( 'b.txt' )
				self.assertEqual( info_b.compress_type, _pyzipfile.ZIP_DEFLATED )

	def _assert_compiles_and_runs_isolated( self, source: str ) -> None:
		''' compiles+runs ONE standalone program (not merged with others via
		assert_programs_run) so the test file paths embedded in `source`
		point at a temp dir THIS test method controls, rather than the
		auto-cleaned-up one assert_programs_run's own _build_and_run uses
		internally. '''
		compiler = self._compile_source( source )
		result = self._build_and_run( compiler, emitter_c.emit_c( compiler ), timeout = None )
		self.assertEqual( result.returncode, 0, f'program failed: exit={result.returncode} stdout={result.stdout!r} stderr={result.stderr!r}' )
