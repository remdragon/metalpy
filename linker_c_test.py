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
	DLL actually providing it.

	These tests all pass no_crt=True explicitly - the freestanding scenario
	that motivated this mechanism in the first place - since strnlen is ALSO
	a symbol a CRT-linked build's own default libraries provide; see
	NtdllUcrtStrnlenCollisionTests below for that (no_crt=False) side, the
	one covering the actual duplicate-symbol regression this file's git
	history is really about. '''

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
		lib_path = linker_c.build_ntdll_import_lib( _CC, { 'strnlen' }, no_crt = True )
		self.assertIsNotNone( lib_path )
		self.assertTrue( lib_path.is_file() )
		self.assertGreater( lib_path.stat().st_size, 0 )

	def test_bogus_symbol_raises( self ) -> None:
		with self.assertRaises( RuntimeError ):
			linker_c.build_ntdll_import_lib( _CC, { 'ThisSymbolDoesNotExist987' }, no_crt = True )

	def test_result_is_cached_to_disk( self ) -> None:
		cache_file = self._fresh_cache_file( { 'RtlCopyMemory' } )
		self.assertFalse( cache_file.is_file() )
		linker_c.build_ntdll_import_lib( _CC, { 'RtlCopyMemory' }, no_crt = True )
		self.assertTrue( cache_file.is_file() )

	def test_second_call_hits_the_cache_not_the_toolchain( self ) -> None:
		self._fresh_cache_file( { 'RtlZeroMemory' } )
		first = linker_c.build_ntdll_import_lib( _CC, { 'RtlZeroMemory' }, no_crt = True )
		# if the second call actually re-probed ntdll.dll's export table
		# instead of reading the cache, this patch would make it explode
		with patch( 'linker_c._real_ntdll_exports', side_effect = AssertionError( 'toolchain invoked - cache was not hit' ) ):
			second = linker_c.build_ntdll_import_lib( _CC, { 'RtlZeroMemory' }, no_crt = True )
		self.assertEqual( first, second )

	def test_resolve_lib_ldflag_ntdll_returns_generated_lib_path( self ) -> None:
		self._fresh_cache_file( { 'strnlen' } )
		flag = linker_c.resolve_lib_ldflag( _CC, 'ntdll', { 'strnlen' }, no_crt = True )
		self.assertTrue( Path( flag ).is_file() )

	def test_resolve_lib_ldflag_other_lib_is_unaffected( self ) -> None:
		# only 'ntdll' is special-cased - every other library still resolves
		# via the plain -l/.lib flag, unaffected by this whole mechanism
		flag = linker_c.resolve_lib_ldflag( _CC, 'kernel32', { 'GetLastError' } )
		expected = 'kernel32.lib' if _CC.name == 'cl' else '-lkernel32'
		self.assertEqual( flag, expected )


@unittest.skipUnless( _CC is not None and os.name == 'nt', 'ntdll import-lib generation is Windows-only and needs a C compiler' )
class NtdllUcrtStrnlenCollisionTests( unittest.TestCase ):
	''' linker_c.build_ntdll_import_lib()/resolve_lib_ldflag() under
	no_crt=False (a build that DOES link its default C runtime for real) -
	the actual regression this class covers: ntdll.dll and a CRT-linked
	build's own default libraries (ucrt.lib under MSVC/clang) both export a
	real `strnlen` - two unrelated functions sharing a name - so
	synthesizing an ntdll import entry for it on top of a build that ALSO
	links ucrt produced a real LNK2005 "already defined" the moment
	anything actually called sys.cstrlen() on Windows (nothing did, until
	lib/os.py's own work surfaced it - see CstrlenNtdllUcrtCollisionRealCompileTests
	below for the true end-to-end repro). '''

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

	def test_strnlen_alone_needs_no_synthetic_import_lib( self ) -> None:
		# ucrt.lib already provides it under a real CRT-linked build - no
		# import library needed at all, not even an empty one
		self._fresh_cache_file( { 'strnlen' } )
		lib_path = linker_c.build_ntdll_import_lib( _CC, { 'strnlen' }, no_crt = False )
		self.assertIsNone( lib_path )

	def test_resolve_lib_ldflag_ntdll_is_empty_when_crt_already_provides_everything( self ) -> None:
		self._fresh_cache_file( { 'strnlen' } )
		flag = linker_c.resolve_lib_ldflag( _CC, 'ntdll', { 'strnlen' }, no_crt = False )
		self.assertEqual( flag, '' )

	def test_mixed_symbols_only_the_crt_provided_one_is_dropped( self ) -> None:
		# RtlNtStatusToDosError has no CRT-linked equivalent (confirmed: it's
		# a real ntdll export, but _default_link_provides reports False for
		# it, unlike strnlen/the Rtl*Memory family below) - a synthetic
		# import library must still be built for it, scoped to just that one
		# symbol, even though strnlen (requested alongside it) needs none
		self._fresh_cache_file( { 'strnlen', 'RtlNtStatusToDosError' } )
		lib_path = linker_c.build_ntdll_import_lib( _CC, { 'strnlen', 'RtlNtStatusToDosError' }, no_crt = False )
		self.assertIsNotNone( lib_path )
		self.assertTrue( lib_path.is_file() )

	def test_all_of_ntdll_pys_rtl_memory_family_are_also_dropped_when_crt_linked( self ) -> None:
		# not just strnlen: a real CRT-linked build's default libraries
		# (via the CRT startup chain's own /DEFAULTLIB directives, confirmed
		# empirically to ultimately resolve through KERNEL32.dll) ALSO
		# already provide RtlCopyMemory/RtlMoveMemory/RtlFillMemory/
		# RtlZeroMemory/RtlCompareMemory - lib/windows/ntdll.py's entire
		# Rtl*Memory family, not just strnlen. See
		# CstrlenNtdllUcrtCollisionRealCompileTests for end-to-end proof
		# this is actually safe (real programs using all of these still
		# link AND run correctly under a CRT-linked build).
		symbols = { 'strnlen', 'RtlCopyMemory', 'RtlMoveMemory', 'RtlFillMemory', 'RtlZeroMemory', 'RtlCompareMemory' }
		self._fresh_cache_file( symbols )
		lib_path = linker_c.build_ntdll_import_lib( _CC, symbols, no_crt = False )
		self.assertIsNone( lib_path )

	def test_no_crt_true_never_drops_strnlen( self ) -> None:
		# the filtering only applies to a REAL CRT-linked build - a
		# genuinely freestanding one never links ucrt at all (MSVC's
		# explicit /NODEFAULTLIB; clang/gcc's own default CRT libraries
		# never entering the link because nothing here competes with their
		# CRT startup object's own mainCRTStartup), so strnlen still needs
		# its synthetic ntdll entry there, same as any other requested
		# symbol
		self._fresh_cache_file( { 'strnlen' } )
		lib_path = linker_c.build_ntdll_import_lib( _CC, { 'strnlen' }, no_crt = True )
		self.assertIsNotNone( lib_path )
		self.assertTrue( lib_path.is_file() )


@unittest.skipUnless( _CC is not None and os.name == 'nt', 'ntdll import-lib generation is Windows-only and needs a C compiler' )
class CstrlenNtdllUcrtCollisionRealCompileTests( unittest.TestCase ):
	''' end-to-end regression for the real bug: a MetalPy program that calls
	sys.cstrlen() (lib/sys.py, Windows branch -> windows.ntdll.strnlen)
	previously failed to LINK - LNK2019 "unresolved external symbol strnlen"
	under a freestanding build (build_ntdll_import_lib was wrongly probing a
	plain `int main(void)` shape that doesn't match the real freestanding
	program's own mainCRTStartup, so it wrongly concluded ucrt already
	provided strnlen and dropped the synthetic ntdll entry a freestanding
	build genuinely still needs), and LNK2005 "already defined" under a
	CRT-linked build (ucrt.lib's own strnlen colliding with a synthetically
	generated ntdll one for the exact same name) before that. Nothing
	previously exercised a real call to sys.cstrlen() at all - grep for
	`cstrlen(` under lib/builtins confirmed only a comment referenced it. '''

	def _compile_and_run( self, no_crt_forced: bool, expected_exit: int ) -> None:
		code = '\n'.join([
			'import sys',
			'',
			'def main() -> i32:',
			'	buf: ConstPtr[u8] = "hello".get_cstr()',
			'	n: usize = sys.cstrlen( buf, 10 )',
			'	if n != 5:',
			'		return 1',
			'	return 0',
		])
		discovery = Discovery( import_builtins = True )
		compiler = Compiler( discovery )
		compiler.import_code( code, Path( '__main__.py' ), scope = None )
		compiler.run()
		self.assertEqual( discovery.errors.errors, [] )

		# mirrors mpy.py's own no_crt derivation (see mpy.py's --crt flag) -
		# forcing CRT linking here is what actually exercises the ucrt/ntdll
		# strnlen collision; the natural (unforced) no_crt is the OTHER real
		# bug this class covers (see class docstring)
		no_crt = ( 'c' not in compiler.extern_libs and not compiler.requires_crt ) and not no_crt_forced
		c_source = emitter_c.emit_c( compiler, no_crt = no_crt )

		with tempfile.TemporaryDirectory() as tmp:
			src_path = Path( tmp ) / 'generated.c'
			obj_path = Path( tmp ) / 'generated.o'
			exe_path = Path( tmp ) / 'test_exe.exe'
			src_path.write_text( c_source, encoding = 'utf-8' )

			cc_result = _CC.compile( src_path, obj_path, no_crt = no_crt )
			self.assertEqual( cc_result.returncode, 0, f'{_CC.name} compile failed:\n{cc_result.stdout}{test_support.c_source_on_failure( c_source )}' )

			ldflags = ''
			for lib in sorted( compiler.extern_libs ):
				if lib == 'c':
					continue
				flag = linker_c.resolve_lib_ldflag( _CC, lib, compiler.extern_libs[lib], no_crt = no_crt )
				ldflags = ldflags + f' {flag}' if ldflags else flag

			link_result = _CC.link( exe_path, [ obj_path ], ldflags = ldflags, no_crt = no_crt )
			self.assertEqual( link_result.returncode, 0, f'{_CC.name} link failed:\n{link_result.stdout}' )

			result = subprocess.run( [ str( exe_path ) ], capture_output = True )
			self.assertEqual( result.returncode, expected_exit, f'exe exited {result.returncode}, expected {expected_exit} (stderr: {result.stderr})' )

	def test_cstrlen_freestanding_build_links_and_runs( self ) -> None:
		self._compile_and_run( no_crt_forced = False, expected_exit = 0 )

	def test_cstrlen_crt_linked_build_links_and_runs( self ) -> None:
		self._compile_and_run( no_crt_forced = True, expected_exit = 0 )


@unittest.skipUnless( _CC is not None, 'no C compiler (clang/gcc/msvc) found' )
class StripLinkFlagRealCompileTests( unittest.TestCase ):
	''' CcTool.link(strip=True) - plain -s is a compile-stage flag to clang's
	own driver; this link() call only ever runs a link (objs are already-
	compiled .o files, no cc1 invocation happens here), so on native Windows
	(clang driving lld-link) the driver never got a chance to consume it and
	warned "argument unused during compilation" instead of forwarding it
	anywhere - confirmed via a real repro (`mpy --release --strip`). -Wl,-s
	fixes GNU ld (POSIX) but lld-link doesn't understand -s either (LNK4044
	"unrecognized option"), so native Windows needs -Wl,/OPT:REF /OPT:ICF
	instead (the same "closest real analog" the 'cl' branch already used).
	Real end-to-end regression: build+link+run with strip=True must produce
	no linker warnings at all, on whichever compiler this test env has. '''

	def test_strip_produces_no_linker_warnings_and_still_runs( self ) -> None:
		code = '\n'.join([
			'def main() -> i32:',
			'	return 0',
		])
		discovery = Discovery( import_builtins = True )
		compiler = Compiler( discovery )
		compiler.import_code( code, Path( '__main__.py' ), scope = None )
		compiler.run()
		self.assertEqual( discovery.errors.errors, [] )

		no_crt = 'c' not in compiler.extern_libs and not compiler.requires_crt
		c_source = emitter_c.emit_c( compiler, no_crt = no_crt )

		with tempfile.TemporaryDirectory() as tmp:
			src_path = Path( tmp ) / 'generated.c'
			obj_path = Path( tmp ) / 'generated.o'
			exe_path = Path( tmp ) / 'test_exe.exe'
			src_path.write_text( c_source, encoding = 'utf-8' )

			cc_result = _CC.compile( src_path, obj_path, no_crt = no_crt )
			self.assertEqual( cc_result.returncode, 0, f'{_CC.name} compile failed:\n{cc_result.stdout}{test_support.c_source_on_failure( c_source )}' )

			ldflags = ''
			for lib in sorted( compiler.extern_libs ):
				if lib == 'c':
					continue
				flag = linker_c.resolve_lib_ldflag( _CC, lib, compiler.extern_libs[lib], no_crt = no_crt )
				ldflags = ldflags + f' {flag}' if ldflags else flag

			link_result = _CC.link( exe_path, [ obj_path ], ldflags = ldflags, no_crt = no_crt, strip = True )
			self.assertEqual( link_result.returncode, 0, f'{_CC.name} link failed:\n{link_result.stdout}' )
			self.assertEqual( link_result.stdout.strip(), '', f'{_CC.name} strip link produced unexpected warnings:\n{link_result.stdout}' )

			result = subprocess.run( [ str( exe_path ) ], capture_output = True )
			self.assertEqual( result.returncode, 0, f'exe exited {result.returncode} (stderr: {result.stderr})' )


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
