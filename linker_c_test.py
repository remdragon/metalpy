# stdlib imports:
import hashlib
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
