# Real-compile-and-run tests for lib/itertools.py.

# stdlib imports:
import unittest

# local imports:
import test_support


class ItertoolsTests( test_support.RealCompileMixin, unittest.TestCase ):
	def setUp( self ) -> None:
		from discovery import Discovery
		from compiler import Compiler
		self.discovery = Discovery( import_builtins = True )
		self.compiler = Compiler( self.discovery )

	@unittest.skipUnless( test_support.HAS_CC, 'no C compiler (clang/gcc/msvc) found - skipping' )
	def test_programs_compile_and_run( self ) -> None:
		import emitter_c
		self.assert_programs_run([
			( 'count_yields_start_then_step_increments', '''
import compiler
import itertools

def main() -> i32:
	seen: list[i32] = list[i32]()
	n: usize = 0
	with compiler.wrap_arithmetic:
		for x in itertools.count[i32]( 5, 2 ):
			seen.append( x )
			n += usize( 1 )
			if n >= usize( 4 ):
				break
	if seen.__len__() != usize( 4 ):
		return 1
	if seen.__getitem__( 0 ).unwrap( '' ) != 5:
		return 2
	if seen.__getitem__( 3 ).unwrap( '' ) != 11:
		return 3
	return 0
''' ),
			( 'count_defaults_to_zero_and_one', '''
import compiler
import itertools

def main() -> i32:
	seen: list[i32] = list[i32]()
	n: usize = 0
	with compiler.wrap_arithmetic:
		for x in itertools.count[i32]():
			seen.append( x )
			n += usize( 1 )
			if n >= usize( 3 ):
				break
	if seen.__getitem__( 0 ).unwrap( '' ) != 0:
		return 1
	if seen.__getitem__( 2 ).unwrap( '' ) != 2:
		return 2
	return 0
''' ),
			( 'chain_concatenates_two_lists_lazily', '''
import itertools

def main() -> i32:
	a: list[i32] = list[i32]()
	a.append( 1 )
	a.append( 2 )
	b: list[i32] = list[i32]()
	b.append( 3 )
	b.append( 4 )
	b.append( 5 )
	out: list[i32] = list[i32]()
	for x in itertools.chain( a, b ):
		out.append( x )
	if out.__len__() != usize( 5 ):
		return 1
	if out.__getitem__( 0 ).unwrap( '' ) != 1:
		return 2
	if out.__getitem__( 4 ).unwrap( '' ) != 5:
		return 3
	return 0
''' ),
			( 'chain_with_an_empty_side_is_a_passthrough', '''
import itertools

def main() -> i32:
	a: list[i32] = list[i32]()
	empty: list[i32] = list[i32]()
	a.append( 1 )
	a.append( 2 )
	out: list[i32] = list[i32]()
	for x in itertools.chain( a, empty ):
		out.append( x )
	if out.__len__() != usize( 2 ):
		return 1
	out2: list[i32] = list[i32]()
	for x in itertools.chain( empty, a ):
		out2.append( x )
	if out2.__len__() != usize( 2 ):
		return 2
	return 0
''' ),
			( 'islice_stop_only_takes_a_prefix', '''
import itertools

def main() -> i32:
	src: list[i32] = list[i32]()
	src.append( 10 )
	src.append( 20 )
	src.append( 30 )
	src.append( 40 )
	src.append( 50 )
	out: list[i32] = list[i32]()
	for x in itertools.islice( src, usize( 3 )):
		out.append( x )
	if out.__len__() != usize( 3 ):
		return 1
	if out.__getitem__( 2 ).unwrap( '' ) != 30:
		return 2
	return 0
''' ),
			( 'islice_start_stop_takes_a_window', '''
import itertools

def main() -> i32:
	src: list[i32] = list[i32]()
	src.append( 10 )
	src.append( 20 )
	src.append( 30 )
	src.append( 40 )
	src.append( 50 )
	out: list[i32] = list[i32]()
	for x in itertools.islice( src, usize( 1 ), usize( 4 )):
		out.append( x )
	if out.__len__() != usize( 3 ):
		return 1
	if out.__getitem__( 0 ).unwrap( '' ) != 20:
		return 2
	if out.__getitem__( 2 ).unwrap( '' ) != 40:
		return 3
	return 0
''' ),
			( 'islice_stop_beyond_length_is_clamped', '''
import itertools

def main() -> i32:
	src: list[i32] = list[i32]()
	src.append( 1 )
	src.append( 2 )
	out: list[i32] = list[i32]()
	for x in itertools.islice( src, usize( 100 )):
		out.append( x )
	if out.__len__() != usize( 2 ):
		return 1
	return 0
''' ),
		])


if __name__ == '__main__':
	unittest.main()
