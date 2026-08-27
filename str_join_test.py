# Real-compile-and-run tests for str.join(): the fast list[str] path plus
# the IteratorProtocol[T]/Iterable[T] overload pair that lets it accept a
# generator (e.g. map(...)'s own return value) or any other Iterable[str]
# conformer directly, matching real Python's own str.join() flexibility.

# stdlib imports:
import unittest

# local imports:
import test_support


class StrJoinTests( test_support.RealCompileMixin, unittest.TestCase ):
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
			( 'join_list_fast_path_unchanged', '''
def main() -> i32:
	parts: list[str] = [ 'a', 'b', 'c' ]
	if '-'.join( parts ) != 'a-b-c':
		return 1
	if ''.join( parts ) != 'abc':
		return 2
	empty: list[str] = []
	if ','.join( empty ) != '':
		return 3
	return 0
''' ),
			# the exact shape a real repro (grap.mpy) hit: joining a bare
			# generator (map(...)'s own return value), never materialized
			# into a list[str] by the caller - str.join used to only accept
			# list[str] directly, a real gap vs. real Python's own
			# str.join(any_iterable)
			( 'join_generator_from_map', '''
def to_upper( s: str ) -> str:
	return s.upper()

def main() -> i32:
	parts: list[str] = [ 'a', 'b', 'c' ]
	if '|'.join( map( to_upper, parts )) != 'A|B|C':
		return 1
	empty: list[str] = []
	if ','.join( map( to_upper, empty )) != '':
		return 2
	return 0
''' ),
			# NOTE: the sibling Iterable[str] overload (anything with __iter__
			# that isn't already a list[str] or an in-progress iterator, e.g.
			# set[str]) is NOT exercised here as its own real-compile case -
			# combining it with the map(...) case above in one compiled
			# program (assert_programs_run merges every case into ONE
			# executable) hits a real, separate, pre-existing compiler bug:
			# two structurally different anonymous Generator specializations
			# collide on the same mangled __new__ symbol name (confirmed via
			# a real repro; unrelated to str.join itself) - flagged as its
			# own follow-up, not fixed here.
		])


if __name__ == '__main__':
	unittest.main()
