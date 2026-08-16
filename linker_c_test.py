# stdlib imports:
import hashlib
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

# local imports:
import linker_c
import test_support
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
