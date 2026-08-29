# Real-compile-and-run tests for lib/functools.py's Memoize[K,V].

# stdlib imports:
import unittest

# local imports:
import test_support


class FunctoolsTests( test_support.RealCompileMixin, unittest.TestCase ):
	def setUp( self ) -> None:
		from discovery import Discovery
		from compiler import Compiler
		self.discovery = Discovery( import_builtins = True )
		self.compiler = Compiler( self.discovery )

	@unittest.skipUnless( test_support.HAS_CC, 'no C compiler (clang/gcc/msvc) found - skipping' )
	def test_programs_compile_and_run( self ) -> None:
		import emitter_c
		self.assert_programs_run([
			( 'memoize_caches_repeat_calls', '''
import compiler
import functools

_calls: i32 = 0

def expensive( x: i64 ) -> i64:
	global _calls
	with compiler.wrap_arithmetic:
		_calls += 1
		return x * x

def main() -> i32:
	cache: functools.Memoize[i64, i64] = functools.Memoize[i64,i64]( expensive )
	a: i64 = cache.call( 5 )
	b: i64 = cache.call( 5 )
	c: i64 = cache.call( 5 )
	if a != 25 or b != 25 or c != 25:
		return 1
	if _calls != 1:  # only the FIRST call() should have reached expensive()
		return 2
	return 0
''' ),
			( 'memoize_tracks_distinct_keys_separately', '''
import compiler
import functools

def square( x: i64 ) -> i64:
	with compiler.wrap_arithmetic:
		return x * x

def main() -> i32:
	cache: functools.Memoize[i64, i64] = functools.Memoize[i64,i64]( square )
	if cache.cached_count() != usize( 0 ):
		return 1
	if cache.call( 3 ) != 9:
		return 2
	if cache.cached_count() != usize( 1 ):
		return 3
	if cache.call( 4 ) != 16:
		return 4
	if cache.cached_count() != usize( 2 ):
		return 5
	if cache.call( 3 ) != 9:  # repeat of an already-cached key - count stays 2
		return 6
	if cache.cached_count() != usize( 2 ):
		return 7
	return 0
''' ),
			( 'memoize_over_string_keys', '''
import functools

def greeting( name: str ) -> str:
	return 'hello, ' + name

def main() -> i32:
	cache: functools.Memoize[str, str] = functools.Memoize[str,str]( greeting )
	if cache.call( 'world' ) != 'hello, world':
		return 1
	if cache.call( 'world' ) != 'hello, world':
		return 2
	if cache.cached_count() != usize( 1 ):
		return 3
	return 0
''' ),
		])


if __name__ == '__main__':
	unittest.main()
