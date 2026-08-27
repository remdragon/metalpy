# Real-compile-and-run tests for negative-index subscript support (Python's
# own s[-1]-is-the-last-element convention), added as a sibling __getitem__(
# idx: isize)/__setitem__(idx: isize, ...) overload alongside every existing
# __getitem__/__setitem__(idx: usize, ...) - not a widened parameter type,
# which would have broken every pre-existing usize-indexed call site (see
# _resolve_index's own comment in lib/builtins/__init__.py). Covers every
# real container this landed on: list[T] (read+write), str, bytes,
# bytearray (read+write), memoryview, VariadicTuple (tuple(...)), and the
# fixed-arity tuple literal's own compile-time direct-field-access rewrite
# (a separate code path in lowering.py, not __getitem__ at all - read-only,
# tuples have no __setitem__).

# stdlib imports:
import unittest

# local imports:
import test_support


class NegativeIndexTests( test_support.RealCompileMixin, unittest.TestCase ):
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
			( 'list_negative_index', '''
def run() -> Result[i32, IndexError]:
	ar: list[str] = [ 'a', 'b', 'c' ]
	last: str = ar[-1]
	if last != 'c':
		return Result.Ok( 1 )
	first: str = ar[-3]
	if first != 'a':
		return Result.Ok( 2 )
	if ar.__getitem__( -4 ).is_ok(): # out of range, still a real IndexError
		return Result.Ok( 3 )
	# a usize-typed variable index still works completely unchanged (the
	# historically common `for i in range(len(x)): x[i]` shape)
	i: usize = 1
	mid: str = ar[i]
	if mid != 'b':
		return Result.Ok( 4 )
	# negative-index ASSIGNMENT (a separate __setitem__ overload, same
	# reasoning) - x[-1] = val overwrites the LAST element
	ar[-1] = 'z'
	changed: str = ar[-1]
	if changed != 'z':
		return Result.Ok( 5 )
	unchanged_first: str = ar[0]
	if unchanged_first != 'a':
		return Result.Ok( 6 )
	return Result.Ok( 0 )

def main() -> i32:
	return run().unwrap( 'list_negative_index' )
''' ),
			( 'str_negative_index', '''
def run() -> Result[i32, IndexError]:
	s: str = 'hello'
	last: str = s[-1]
	if last != 'o':
		return Result.Ok( 1 )
	first: str = s[-5]
	if first != 'h':
		return Result.Ok( 2 )
	if s.__getitem__( -6 ).is_ok(): # out of range
		return Result.Ok( 3 )
	return Result.Ok( 0 )

def main() -> i32:
	return run().unwrap( 'str_negative_index' )
''' ),
			( 'bytes_bytearray_memoryview_negative_index', '''
def run() -> Result[i32, IndexError]:
	ba: bytearray = bytearray( 3 )
	ba[0] = 10
	ba[1] = 20
	ba[2] = 30
	last_ba: u8 = ba[-1]
	if last_ba != 30:
		return Result.Ok( 1 )

	bts: bytes = bytes( ba )
	last_b: u8 = bts[-1]
	if last_b != 30:
		return Result.Ok( 2 )

	with memoryview( ba ) as mv:
		last_mv: u8 = mv[-1]
		if last_mv != 30:
			return Result.Ok( 3 )

	# negative-index ASSIGNMENT (bytearray's own __setitem__ overload -
	# infallible/debug-assert-only, unlike list[T]'s Result-returning one -
	# see its own comment)
	ba[-1] = 99
	changed: u8 = ba[-1]
	if changed != 99:
		return Result.Ok( 4 )
	unchanged_first: u8 = ba[0]
	if unchanged_first != 10:
		return Result.Ok( 5 )

	return Result.Ok( 0 )

def main() -> i32:
	return run().unwrap( 'bytes_bytearray_memoryview_negative_index' )
''' ),
			( 'variadic_tuple_and_fixed_arity_tuple_negative_index', '''
def run() -> Result[i32, IndexError]:
	src: list[i32] = [ i32( 1 ), i32( 2 ), i32( 3 ) ]
	vt = tuple( src )
	last_vt: i32 = vt[-1]
	if last_vt != 3:
		return Result.Ok( 1 )

	# fixed-arity tuple literal - a SEPARATE compile-time direct-field-
	# access rewrite in lowering.py, not __getitem__ at all
	t = ( i32( 10 ), i32( 20 ), i32( 30 ), i32( 40 ) )
	last_t: i32 = t[-1]
	if last_t != 40:
		return Result.Ok( 2 )
	second_t: i32 = t[-2]
	if second_t != 30:
		return Result.Ok( 3 )

	return Result.Ok( 0 )

def main() -> i32:
	return run().unwrap( 'variadic_tuple_and_fixed_arity_tuple_negative_index' )
''' ),
			# a positive-valued literal index must still pick the ORIGINAL
			# usize overload unambiguously, not the new isize sibling - a
			# real regression this landed and fixed along the way (every
			# non-negative literal argument to an OVERLOADED __getitem__,
			# not just subscript syntax, briefly became "ambiguous literal
			# argument 0 - matches more than one overload candidate type
			# (usize, isize)" once the isize sibling existed, confirmed via
			# lib/re.py's own out.__getitem__(0) call sites)
			( 'positive_literal_index_stays_unambiguous', '''
def run() -> Result[i32, IndexError]:
	ar: list[str] = [ 'x', 'y' ]
	first: str = ar.__getitem__( 0 ).unwrap( 'positive literal must resolve to usize unambiguously' )
	if first != 'x':
		return Result.Ok( 1 )
	return Result.Ok( 0 )

def main() -> i32:
	return run().unwrap( 'positive_literal_index_stays_unambiguous' )
''' ),
		])


if __name__ == '__main__':
	unittest.main()
