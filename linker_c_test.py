# stdlib imports:
import contextlib
import hashlib
import io
import os
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch

# local imports:
import emitter_c
import linker_c
import test_support
from compiler import Compiler
from discovery import Discovery
from test_support import KNOWN_LIB, KNOWN_SYMBOLS

_CC = linker_c.detect_cc()


def _cache_file( lib: str, symbol: str, cc_name: str ) -> Path:
	key = hashlib.sha256( f'{lib}\0{symbol}\0{cc_name}'.encode() ).hexdigest()[:16]
	return Path( tempfile.gettempdir() ) / 'metalpy' / 'has_symbol' / key


@unittest.skipUnless( _CC is not None, 'no C compiler (clang/gcc/msvc) found' )
class HasSymbolTests( unittest.TestCase ):
	''' linker_c.has_symbol() - a compile+link-only probe (no run, unlike
	compiler.cexpr()'s own _eval_cexpr - see its own docstring), so this
	works even cross-compiling. '''

	def setUp( self ) -> None:
		# a unique-per-test symbol name avoids collisions with the disk
		# cache other test files/processes may have already populated for
		# the SAME (lib, symbol, cc) triple
		self._cache_files: list[Path] = []

	def tearDown( self ) -> None:
		for f in self._cache_files:
			f.unlink( missing_ok = True )

	def _fresh_cache_file( self, lib: str, symbol: str ) -> Path:
		f = _cache_file( lib, symbol, _CC.name )
		f.unlink( missing_ok = True )
		self._cache_files.append( f )
		return f

	def test_real_symbol_is_available( self ) -> None:
		self._fresh_cache_file( KNOWN_LIB, KNOWN_SYMBOLS[0] )
		self.assertTrue( linker_c.has_symbol( _CC, KNOWN_LIB, KNOWN_SYMBOLS[0] ))

	def test_bogus_symbol_in_real_library_is_unavailable( self ) -> None:
		self._fresh_cache_file( KNOWN_LIB, 'ThisSymbolDoesNotExist987' )
		self.assertFalse( linker_c.has_symbol( _CC, KNOWN_LIB, 'ThisSymbolDoesNotExist987' ))

	def test_bogus_library_is_unavailable( self ) -> None:
		self._fresh_cache_file( 'ThisLibraryDoesNotExist987', KNOWN_SYMBOLS[0] )
		self.assertFalse( linker_c.has_symbol( _CC, 'ThisLibraryDoesNotExist987', KNOWN_SYMBOLS[0] ))

	def test_result_is_cached_to_disk( self ) -> None:
		cache_file = self._fresh_cache_file( KNOWN_LIB, KNOWN_SYMBOLS[1] )
		self.assertFalse( cache_file.is_file() )
		linker_c.has_symbol( _CC, KNOWN_LIB, KNOWN_SYMBOLS[1] )
		self.assertTrue( cache_file.is_file() )

	def test_second_call_hits_the_cache_not_the_compiler( self ) -> None:
		self._fresh_cache_file( KNOWN_LIB, KNOWN_SYMBOLS[2] )
		first = linker_c.has_symbol( _CC, KNOWN_LIB, KNOWN_SYMBOLS[2] )
		# if the second call actually re-invoked the compiler instead of
		# reading the cache, this patch would make it explode
		with patch.object( linker_c.CcTool, 'compile', side_effect = AssertionError( 'compiler invoked - cache was not hit' ) ):
			second = linker_c.has_symbol( _CC, KNOWN_LIB, KNOWN_SYMBOLS[2] )
		self.assertEqual( first, second )
		self.assertTrue( first )


class ResolveNoCrtTests( unittest.TestCase ):
	''' linker_c.resolve_no_crt() - --asan requires the C runtime, so a
	no-CRT auto-detected program requesting asan must have no_crt forced
	False (with a warning) rather than dying with a wall of unresolved-
	external link errors against the ASan runtime itself. '''

	def test_asan_overrides_no_crt( self ) -> None:
		stderr = io.StringIO()
		with contextlib.redirect_stderr( stderr ):
			result = linker_c.resolve_no_crt( True, True )
		self.assertFalse( result )
		self.assertIn( 'WARNING', stderr.getvalue() )
		self.assertIn( '--asan', stderr.getvalue() )

	def test_no_asan_leaves_no_crt_true_unchanged( self ) -> None:
		stderr = io.StringIO()
		with contextlib.redirect_stderr( stderr ):
			result = linker_c.resolve_no_crt( True, False )
		self.assertTrue( result )
		self.assertEqual( stderr.getvalue(), '' )

	def test_asan_with_crt_already_requested_is_silent( self ) -> None:
		# no_crt was already False (program imports c / caller forced CRT) -
		# nothing is being overridden, so no warning should fire
		stderr = io.StringIO()
		with contextlib.redirect_stderr( stderr ):
			result = linker_c.resolve_no_crt( False, True )
		self.assertFalse( result )
		self.assertEqual( stderr.getvalue(), '' )

	def test_no_asan_no_crt_false_unchanged( self ) -> None:
		self.assertFalse( linker_c.resolve_no_crt( False, False ) )


@unittest.skipUnless( _CC is not None, 'no C compiler (clang/gcc/msvc) found' )
class AsanNoCrtRealCompileTests( unittest.TestCase ):
	''' end-to-end: a program that would normally build freestanding/no-CRT
	(no `import c`) must still build, link, AND RUN successfully under
	--asan - the motivating bug was --asan silently keeping the caller's
	no_crt=True and dying with dozens of LNK2019s against the ASan
	runtime's own CRT dependencies (getenv, memcpy, ...). '''

	def test_no_crt_program_builds_and_runs_under_asan( self ) -> None:
		code = 'def main() -> i32:\n\treturn 0\n'
		discovery = Discovery( import_builtins = True )
		compiler = Compiler( discovery )
		compiler.import_code( code, Path( '__main__.py' ), scope = None )
		compiler.run()
		self.assertEqual( discovery.errors.errors, [] )

		no_crt = 'c' not in compiler.extern_libs
		self.assertTrue( no_crt, 'test program is expected to auto-detect as no-CRT' )

		stderr = io.StringIO()
		with contextlib.redirect_stderr( stderr ):
			no_crt = linker_c.resolve_no_crt( no_crt, True )
		self.assertFalse( no_crt )
		self.assertIn( 'WARNING', stderr.getvalue() )

		c_source = emitter_c.emit_c( compiler, no_crt = no_crt )

		with tempfile.TemporaryDirectory() as tmp:
			src_path = Path( tmp ) / 'generated.c'
			obj_path = Path( tmp ) / 'generated.o'
			exe_path = Path( tmp ) / 'test_exe.exe'
			src_path.write_text( c_source, encoding = 'utf-8' )

			cc_result = _CC.compile( src_path, obj_path, no_crt = no_crt, asan = True )
			self.assertEqual( cc_result.returncode, 0, f'{_CC.name} compile failed:\n{cc_result.stdout}{test_support.c_source_on_failure( c_source )}' )

			link_result = _CC.link( exe_path, [ obj_path ], no_crt = no_crt, asan = True )
			self.assertEqual( link_result.returncode, 0, f'{_CC.name} link failed:\n{link_result.stdout}' )

			run_result = subprocess.run( [ str( exe_path ) ], capture_output = True )
			self.assertEqual( run_result.returncode, 0, f'stdout:\n{run_result.stdout}\nstderr:\n{run_result.stderr}' )


def _ntdll_cache_file( cc_name: str, symbols: set[str] ) -> Path:
	key = hashlib.sha256( f'{cc_name}\0{",".join( sorted( symbols ))}'.encode() ).hexdigest()[:16]
	return Path( tempfile.gettempdir() ) / 'metalpy' / 'ntdll_import_lib' / f'{key}.lib'


@unittest.skipUnless( _CC is not None and os.name == 'nt', 'ntdll import-lib generation is Windows-only and needs a C compiler' )
class NtdllImportLibTests( unittest.TestCase ):
	''' linker_c.build_ntdll_import_lib()/resolve_lib_ldflag() - see
	build_ntdll_import_lib's own docstring for the motivating bug: the
	Windows SDK's ntdll.lib import library is a curated subset of the real
	ntdll.dll's export table (confirmed missing: strnlen, despite `dumpbin
	/exports` showing it's a genuine export) - a real, no-CRT Windows build
	needing such a symbol previously hit LNK2019 at link time despite the
	DLL actually providing it. '''

	def setUp( self ) -> None:
		self._cache_files: list[Path] = []

	def tearDown( self ) -> None:
		for f in self._cache_files:
			f.unlink( missing_ok = True )

	def _fresh_cache_file( self, symbols: set[str] ) -> Path:
		f = _ntdll_cache_file( _CC.name, symbols )
		f.unlink( missing_ok = True )
		self._cache_files.append( f )
		return f

	def test_real_export_missing_from_sdk_stub_builds_a_working_lib( self ) -> None:
		# strnlen: genuinely exported by the real ntdll.dll but absent from
		# the SDK's own ntdll.lib stub - the motivating case for this whole
		# mechanism (see lib/windows/ntdll.py's own note on its binding)
		self._fresh_cache_file( { 'strnlen' } )
		lib_path = linker_c.build_ntdll_import_lib( _CC, { 'strnlen' } )
		self.assertTrue( lib_path.is_file() )
		self.assertGreater( lib_path.stat().st_size, 0 )

	def test_bogus_symbol_raises( self ) -> None:
		with self.assertRaises( RuntimeError ):
			linker_c.build_ntdll_import_lib( _CC, { 'ThisSymbolDoesNotExist987' } )

	def test_result_is_cached_to_disk( self ) -> None:
		cache_file = self._fresh_cache_file( { 'RtlCopyMemory' } )
		self.assertFalse( cache_file.is_file() )
		linker_c.build_ntdll_import_lib( _CC, { 'RtlCopyMemory' } )
		self.assertTrue( cache_file.is_file() )

	def test_second_call_hits_the_cache_not_the_toolchain( self ) -> None:
		self._fresh_cache_file( { 'RtlZeroMemory' } )
		first = linker_c.build_ntdll_import_lib( _CC, { 'RtlZeroMemory' } )
		# if the second call actually re-probed ntdll.dll's export table
		# instead of reading the cache, this patch would make it explode
		with patch( 'linker_c._real_ntdll_exports', side_effect = AssertionError( 'toolchain invoked - cache was not hit' ) ):
			second = linker_c.build_ntdll_import_lib( _CC, { 'RtlZeroMemory' } )
		self.assertEqual( first, second )

	def test_resolve_lib_ldflag_ntdll_returns_generated_lib_path( self ) -> None:
		self._fresh_cache_file( { 'strnlen' } )
		flag = linker_c.resolve_lib_ldflag( _CC, 'ntdll', { 'strnlen' } )
		self.assertTrue( Path( flag ).is_file() )

	def test_resolve_lib_ldflag_other_lib_is_unaffected( self ) -> None:
		# only 'ntdll' is special-cased - every other library still resolves
		# via the plain -l/.lib flag, unaffected by this whole mechanism
		flag = linker_c.resolve_lib_ldflag( _CC, 'kernel32', { 'GetLastError' } )
		expected = 'kernel32.lib' if _CC.name == 'cl' else '-lkernel32'
		self.assertEqual( flag, expected )


class FindDllTests( unittest.TestCase ):
	''' linker_c.find_dll() - PATH-order lookup by bare filename, feeding
	mpy.py's post-link bundling step for @extern(..., dll=...)
	dependencies (see compiler.extern_dlls). Uses a scratch PATH (mocked
	os.environ, not the real one) so this doesn't depend on what happens
	to be installed on the machine running the tests. '''

	def test_finds_a_real_file_on_path( self ) -> None:
		with tempfile.TemporaryDirectory() as tmp:
			dll_path = Path( tmp ) / 'fake_dep.dll'
			dll_path.write_bytes( b'not a real PE, just needs to exist' )
			with patch.dict( os.environ, { 'PATH': tmp } ):
				found = linker_c.find_dll( 'fake_dep.dll' )
			self.assertEqual( found, dll_path )

	def test_searches_path_entries_in_order( self ) -> None:
		with tempfile.TemporaryDirectory() as tmp:
			first_dir = Path( tmp ) / 'first'
			second_dir = Path( tmp ) / 'second'
			first_dir.mkdir()
			second_dir.mkdir()
			( second_dir / 'fake_dep.dll' ).write_bytes( b'second' )
			( first_dir / 'fake_dep.dll' ).write_bytes( b'first' )
			with patch.dict( os.environ, { 'PATH': os.pathsep.join([ str( first_dir ), str( second_dir ) ]) } ):
				found = linker_c.find_dll( 'fake_dep.dll' )
			self.assertEqual( found, first_dir / 'fake_dep.dll' )

	def test_returns_none_when_not_found_anywhere_on_path( self ) -> None:
		with tempfile.TemporaryDirectory() as tmp:
			with patch.dict( os.environ, { 'PATH': tmp } ):
				found = linker_c.find_dll( 'does_not_exist_987.dll' )
			self.assertIsNone( found )

	def test_ignores_empty_path_entries( self ) -> None:
		# a PATH like "C:\foo;;C:\bar" (empty segment from a trailing/
		# doubled separator) must not be treated as "search cwd" via a bare
		# Path('') / name - real Windows PATH values sometimes have these
		with patch.dict( os.environ, { 'PATH': os.pathsep.join([ '', '' ]) } ):
			found = linker_c.find_dll( 'kernel32.dll' )
		self.assertIsNone( found )
