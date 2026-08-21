# Real-compile-and-run tests for the memoryview builtin (lib/builtins/
# __memoryview.py): a no-copy view over a bytearray's own buffer, slice
# syntax (mv[a:b]/mv[:b]/mv[a:]), and the `with` context-manager protocol
# (a deliberate extension beyond real Python - see __memoryview.py's own
# module docstring for why).

# stdlib imports:
import unittest

# local imports:
import test_support


class MemoryviewTests( test_support.RealCompileMixin, unittest.TestCase ):
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
			( 'construct_len_getitem', '''
def main() -> i32:
	b: bytearray = bytearray( 3 )
	p: Ptr[u8] = b.get_ptr()
	p[0] = 10
	p[1] = 20
	p[2] = 30
	mv: memoryview = memoryview( b )
	if len( mv ) != 3:
		return 1
	v0: u8 = mv.__getitem__( 0 ).unwrap( 'idx' )
	if v0 != 10:
		return 2
	v2: u8 = mv.__getitem__( 2 ).unwrap( 'idx' )
	if v2 != 30:
		return 3
	if mv.__getitem__( 3 ).is_ok():
		return 4
	return 0
''' ),
			( 'with_statement_context_manager', '''
def main() -> i32:
	b: bytearray = bytearray( 2 )
	p: Ptr[u8] = b.get_ptr()
	p[0] = 1
	p[1] = 2
	with memoryview( b ) as mv:
		if len( mv ) != 2:
			return 1
	return 0
''' ),
			( 'slice_is_a_view_not_a_copy', '''
def main() -> i32:
	b: bytearray = bytearray( 5 )
	p: Ptr[u8] = b.get_ptr()
	p[0] = 1
	p[1] = 2
	p[2] = 3
	p[3] = 4
	p[4] = 5
	mv: memoryview = memoryview( b )

	mid: memoryview = mv[1:4]
	if len( mid ) != 3:
		return 1
	m0: u8 = mid.__getitem__( 0 ).unwrap( 'idx' )
	if m0 != 2:
		return 2
	m2: u8 = mid.__getitem__( 2 ).unwrap( 'idx' )
	if m2 != 4:
		return 3

	tail: memoryview = mv[3:]
	if len( tail ) != 2:
		return 4
	t0: u8 = tail.__getitem__( 0 ).unwrap( 'idx' )
	if t0 != 4:
		return 5

	head: memoryview = mv[:2]
	if len( head ) != 2:
		return 6
	h1: u8 = head.__getitem__( 1 ).unwrap( 'idx' )
	if h1 != 2:
		return 7

	# a slice sees mutations made through the ORIGINAL bytearray after
	# the slice was taken - confirms it's a real view, not a copy
	p[2] = 99
	mid_after: u8 = mid.__getitem__( 1 ).unwrap( 'idx' )
	if mid_after != 99:
		return 8
	return 0
''' ),
			( 'slice_of_a_slice', '''
def main() -> i32:
	b: bytearray = bytearray( 6 )
	p: Ptr[u8] = b.get_ptr()
	i: usize = 0
	with compiler.wrap_arithmetic:
		while i < 6:
			p[i] = u8( i )
			i += 1
	mv: memoryview = memoryview( b )
	outer: memoryview = mv[1:5]      # [1,2,3,4]
	inner: memoryview = outer[1:3]   # [2,3]
	if len( inner ) != 2:
		return 1
	v0: u8 = inner.__getitem__( 0 ).unwrap( 'idx' )
	if v0 != 2:
		return 2
	v1: u8 = inner.__getitem__( 1 ).unwrap( 'idx' )
	if v1 != 3:
		return 3
	return 0
''' ),
			( 'source_bytearray_kept_alive_by_the_view', '''
def make_view() -> memoryview:
	# the bytearray argument is a LOCAL here - once make_view() returns,
	# nothing outside still names it directly. The returned memoryview
	# must still keep it alive (an owned __source field, not a bare
	# pointer) for this to be safe rather than a real use-after-free.
	b: bytearray = bytearray( 3 )
	p: Ptr[u8] = b.get_ptr()
	p[0] = 7
	p[1] = 8
	p[2] = 9
	return memoryview( b )

def main() -> i32:
	mv: memoryview = make_view()
	if len( mv ) != 3:
		return 1
	v1: u8 = mv.__getitem__( 1 ).unwrap( 'idx' )
	if v1 != 8:
		return 2
	return 0
''' ),
		])


if __name__ == '__main__':
	unittest.main()
