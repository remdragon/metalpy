# Sequence[T]/Iterable[T] protocol-bound builtins: min(seq)/max(seq)/iter/
# next/any/all/enumerate/map/reduce/sum. Real compile+link+run coverage,
# following re_test.py's own template (RealCompileMixin + assert_programs_run,
# print()-free, exit code 0 = every check passed, distinct nonzero i32 per
# failing check so a regression decodes back to exactly which assertion broke).

import unittest

import test_support
from test_support import RealCompileMixin

_LIST_REDUCTIONS = '''
def main() -> i32:
	lst: list[i32] = list[i32]()
	lst.append( 3 )
	lst.append( 1 )
	lst.append( 4 )
	lst.append( 1 )
	lst.append( 5 )
	if min( lst ) != 1:
		return 1
	if max( lst ) != 5:
		return 2
	if sum( lst ) != 14:
		return 3
	if not any( lst ):
		return 4
	empty: list[i32] = list[i32]()
	if any( empty ):
		return 5
	if not all( lst ):
		return 6
	return 0
'''

_ITER_NEXT_AND_ENUMERATE = '''
def main() -> i32:
	lst: list[i32] = list[i32]()
	lst.append( 10 )
	lst.append( 20 )
	lst.append( 30 )

	it = iter( lst )
	first: i32 = next( it ).unwrap( 'x' )
	if first != 10:
		return 1
	rest_sum: i32 = 0
	with compiler.wrap_arithmetic:
		for x in it:
			rest_sum += x
	if rest_sum != 50:
		return 2

	count: isize = 0
	idx_sum: isize = 0
	with compiler.wrap_arithmetic:
		for pair in enumerate( lst ):
			i, v = pair
			idx_sum += i
			count += 1
	if count != 3:
		return 3
	if idx_sum != 3: # 0+1+2
		return 4
	return 0
'''

_MAP_AND_REDUCE = '''
def double( x: i32 ) -> i32:
	with compiler.wrap_arithmetic:
		return x * 2

def add( a: i32, b: i32 ) -> i32:
	with compiler.wrap_arithmetic:
		return a + b

def main() -> i32:
	lst: list[i32] = list[i32]()
	lst.append( 1 )
	lst.append( 2 )
	lst.append( 3 )

	doubled: list[i32] = list[i32]()
	for v in map( double, lst ):
		doubled.append( v )
	if sum( doubled ) != 12:
		return 1

	total: i32 = reduce( add, lst )
	if total != 6:
		return 2
	return 0
'''

_ZIP_STOPS_AT_SHORTER = '''
def main() -> i32:
	a: list[i32] = list[i32]()
	a.append( 1 )
	a.append( 2 )
	a.append( 3 )
	b: list[i32] = list[i32]()
	b.append( 10 )
	b.append( 20 )

	pairs: list[tuple[i32,i32]] = list[tuple[i32,i32]]()
	for p in zip( a, b ):
		pairs.append( p )
	if pairs.__len__() != usize( 2 ):
		return 1
	first: tuple[i32,i32] = pairs.__getitem__( 0 ).unwrap( 'pairs has 2 entries' )
	if first[0] != 1 or first[1] != 10:
		return 2
	second: tuple[i32,i32] = pairs.__getitem__( 1 ).unwrap( 'pairs has 2 entries' )
	if second[0] != 2 or second[1] != 20:
		return 3
	return 0
'''

_FILTER_KEEPS_ONLY_MATCHING = '''
def is_even( x: i32 ) -> bool:
	with compiler.panic_arithmetic( 'divisor is a nonzero literal' ):
		return x % 2 == 0

def main() -> i32:
	src: list[i32] = list[i32]()
	src.append( 1 )
	src.append( 2 )
	src.append( 3 )
	src.append( 4 )
	src.append( 5 )
	src.append( 6 )
	evens: list[i32] = list[i32]()
	for x in filter( is_even, src ):
		evens.append( x )
	if evens.__len__() != usize( 3 ):
		return 1
	if evens.__getitem__( 0 ).unwrap( '' ) != 2:
		return 2
	if evens.__getitem__( 2 ).unwrap( '' ) != 6:
		return 3
	return 0
'''

_SORTED_ASCENDING_AND_REVERSE = '''
def main() -> i32:
	src: list[i32] = list[i32]()
	src.append( 5 )
	src.append( 3 )
	src.append( 1 )
	src.append( 4 )
	src.append( 2 )

	asc: list[i32] = sorted( src )
	i: usize = 0
	with compiler.wrap_arithmetic:
		while i < usize( 5 ):
			if asc.__getitem__( i ).unwrap( 'i < 5' ) != i32( i ) + 1:
				return 1
			i += usize( 1 )

	desc: list[i32] = sorted( src, reverse = True )
	if desc.__getitem__( 0 ).unwrap( '' ) != 5:
		return 2
	if desc.__getitem__( 4 ).unwrap( '' ) != 1:
		return 3

	# original untouched - sorted() returns a NEW list, doesn't sort in place
	if src.__getitem__( 0 ).unwrap( '' ) != 5:
		return 4

	empty: list[i32] = list[i32]()
	if sorted( empty ).__len__() != usize( 0 ):
		return 5
	single: list[i32] = list[i32]()
	single.append( 42 )
	if sorted( single ).__getitem__( 0 ).unwrap( '' ) != 42:
		return 6
	return 0
'''

_HOMOGENEOUS_TUPLE_CONFORMANCE = '''
def main() -> i32:
	t: tuple[i32,i32,i32] = ( 7, 2, 9 )
	if min( t ) != 2:
		return 1
	if max( t ) != 9:
		return 2
	if sum( t ) != 18:
		return 3
	count: isize = 0
	with compiler.wrap_arithmetic:
		for x in iter( t ):
			count += 1
	if count != 3:
		return 4
	return 0
'''

_TWO_ARG_MIN_MAX_STILL_WORK = '''
def main() -> i32:
	if min( 3, 5 ) != 3:
		return 1
	if max( 3, 5 ) != 5:
		return 2
	return 0
'''

_DICT_ITERABLE_CONFORMANCE = '''
def main() -> i32:
	d: dict[i32, i32] = dict[i32, i32]()
	d[10] = 100
	d[20] = 200
	d[30] = 300

	keysum: i32 = 0
	with compiler.wrap_arithmetic:
		for k in d.keys(): # __iter__ delegates to keys(), which delegates to the actual generator - exercises that 3-deep pass-through chain
			keysum += k
	if keysum != 60:
		return 8
	if sum( d ) != 60: # walks KEYS, matching real python's dict.__iter__
		return 1
	if min( d ) != 10:
		return 2
	if max( d ) != 30:
		return 3
	count: isize = 0
	with compiler.wrap_arithmetic:
		for _k in iter( d ):
			count += 1
	if count != 3:
		return 4
	if not any( d ):
		return 5
	empty: dict[i32, i32] = dict[i32, i32]()
	if any( empty ):
		return 6
	if not all( d ):
		return 7
	return 0
'''

_STR_ITERABLE_CONFORMANCE = '''
def main() -> i32:
	s: str = 'abc'
	count: i32 = 0
	with compiler.wrap_arithmetic:
		for ch in s:
			if ch == 'a':
				count += 1
			elif ch == 'b':
				count += 2
			elif ch == 'c':
				count += 4
	if count != 7:
		return 1
	if min( 'bca' ) != 'a':
		return 2
	if max( 'bca' ) != 'c':
		return 3
	return 0
'''

_BYTES_BYTEARRAY_MEMORYVIEW_ITERABLE_CONFORMANCE = '''
def main() -> i32:
	ba: bytearray = bytearray( 3 )
	ba[0] = 1
	ba[1] = 2
	ba[2] = 3

	total: i32 = 0
	with compiler.wrap_arithmetic:
		for x in ba:
			total += i32( x )
	if total != 6:
		return 1

	mv: memoryview = memoryview( ba )
	total = 0
	with compiler.wrap_arithmetic:
		for x in mv:
			total += i32( x )
	if total != 6:
		return 2

	b: bytes = bytes.from_bytearray( move( bytearray( 3 )))
	count: isize = 0
	with compiler.wrap_arithmetic:
		for x in b:
			count += 1
	if count != 3:
		return 3

	if b[0].unwrap( 'in bounds' ) != 0:
		return 4

	return 0
'''

_GENERIC_TUPLE_TYPED_PARAMETER_INFERENCE = '''
def first_of[T]( t: tuple[T, T] ) -> T:
	a, b = t
	with compiler.wrap_arithmetic:
		return a + b - b # use b so it is not "unused"

def main() -> i32:
	pair: tuple[i32, i32] = ( 3, 4 )
	if first_of( pair ) != 3:
		return 1
	return 0
'''


@unittest.skipUnless( test_support.HAS_CC, 'no C compiler (clang/gcc/msvc) found - skipping real-compile builtins-iteration tests' )
class SequenceIterableBuiltinsTests( RealCompileMixin, unittest.TestCase ):
	def test_list_reductions( self ) -> None:
		self.assert_programs_run([
			( 'list_reductions', _LIST_REDUCTIONS ),
		])

	def test_iter_next_and_enumerate( self ) -> None:
		self.assert_programs_run([
			( 'iter_next_and_enumerate', _ITER_NEXT_AND_ENUMERATE ),
		])

	def test_map_and_reduce( self ) -> None:
		self.assert_programs_run([
			( 'map_and_reduce', _MAP_AND_REDUCE ),
		])

	def test_zip_stops_at_shorter( self ) -> None:
		self.assert_programs_run([
			( 'zip_stops_at_shorter', _ZIP_STOPS_AT_SHORTER ),
		])

	def test_filter_keeps_only_matching( self ) -> None:
		self.assert_programs_run([
			( 'filter_keeps_only_matching', _FILTER_KEEPS_ONLY_MATCHING ),
		])

	def test_sorted_ascending_and_reverse( self ) -> None:
		self.assert_programs_run([
			( 'sorted_ascending_and_reverse', _SORTED_ASCENDING_AND_REVERSE ),
		])

	def test_homogeneous_tuple_conformance( self ) -> None:
		self.assert_programs_run([
			( 'homogeneous_tuple_conformance', _HOMOGENEOUS_TUPLE_CONFORMANCE ),
		])

	def test_two_arg_min_max_still_work( self ) -> None:
		''' the pre-existing min[T](a,b)/max[T](a,b) overload, unaffected
		by the new seq-taking overload sharing its name. '''
		self.assert_programs_run([
			( 'two_arg_min_max', _TWO_ARG_MIN_MAX_STILL_WORK ),
		])

	def test_dict_iterable_conformance( self ) -> None:
		''' dict[K,V] conforms to Iterable[K] via its own key-walking
		__iter__ (matches real python: dict.__iter__ walks keys) - NOT
		Sequence[K], since dict's own __getitem__ takes a K key, not a
		usize index. '''
		self.assert_programs_run([
			( 'dict_iterable_conformance', _DICT_ITERABLE_CONFORMANCE ),
		])

	def test_str_iterable_conformance( self ) -> None:
		''' str conforms to Sequence[str]/Iterable[str] via its existing
		codepoint-indexed __getitem__(usize) - a plain for loop (not just
		s[i] access) and min/max(str) now work. '''
		self.assert_programs_run([
			( 'str_iterable_conformance', _STR_ITERABLE_CONFORMANCE ),
		])

	def test_bytes_bytearray_memoryview_iterable_conformance( self ) -> None:
		''' bytes/bytearray/memoryview all conform to Sequence[u8]/
		Iterable[u8] - bytes needed a new scalar __getitem__(usize)
		overload added alongside its pre-existing slice-only one. '''
		self.assert_programs_run([
			( 'bytes_bytearray_memoryview_iterable_conformance', _BYTES_BYTEARRAY_MEMORYVIEW_ITERABLE_CONFORMANCE ),
		])

	def test_generic_tuple_typed_parameter_inference( self ) -> None:
		''' T occurring only inside a tuple[T,T]-shaped parameter type
		(not behind a Sequence[T]/Iterable[T] bound) must still be
		inferable from a real tuple[i32,i32] argument - _unify_type_param
		previously had no TupleType branch at all. '''
		self.assert_programs_run([
			( 'generic_tuple_typed_parameter_inference', _GENERIC_TUPLE_TYPED_PARAMETER_INFERENCE ),
		])


class SequenceIterableProtocolErrorTests( unittest.TestCase ):
	def setUp( self ) -> None:
		from discovery import Discovery
		from compiler import Compiler
		from pathlib import Path
		self._Path = Path
		self.discovery = Discovery( import_builtins = True )
		self.compiler = Compiler( self.discovery )

	def _run( self, code: str ) -> None:
		self.compiler.import_code( code, self._Path( '__main__.py' ), scope = None )
		self.compiler.run()

	def test_non_conforming_class_rejected_with_clear_protocol_error( self ) -> None:
		''' a class that structurally HAS a matching get()/__getitem__-
		shaped method but never DECLARED Sequence[T]/Iterable[T]
		conformance is rejected with a clear protocol-violation error at
		the min(...) call site - not a confusing, buried structural-
		dispatch failure deep inside min's own body. This is the whole
		reason the user chose a real @protocol design over duck-typed
		unconstrained-T generics for this feature. '''
		self._run( '''
def _not_a_sequence_iter[T]( seq: NotASequence[T] ) -> Generator[T, StopIteration]:
	yield seq.v

class NotASequence[T]: # structurally has __getitem__/__iter__, but never declares Sequence[T]/Iterable[T] conformance
	v: T
	def __init__( self, v: T ) -> None:
		self.v = v
	def __getitem__( self, i: usize ) -> Result[T,IndexError]:
		if i == 0:
			return Result.Ok( self.v )
		return Result.Err( IndexError() )
	def __iter__( self ) -> Generator[T, StopIteration]:
		return _not_a_sequence_iter( self )

def main() -> i32:
	n = NotASequence[i32]( 42 )
	v = min( n )
	return 0
''' )
		self.assertTrue( self.discovery.errors.errors )
		message = str( self.discovery.errors.errors[0] )
		self.assertIn( 'does not implement protocol', message )
		self.assertIn( 'Iterable', message )

	def test_heterogeneous_tuple_does_not_conform( self ) -> None:
		''' tuple[i32,str] has no single element type - never declared
		Sequence[T]/Iterable[T] conformance at all (see tuple_storage.py's
		_declare_sequence_conformance) - min() on it is a clear rejection,
		not a crash or silently-wrong dispatch. '''
		self._run( '''
def main() -> i32:
	t: tuple[i32,str] = ( 1, 'x' )
	v = min( t )
	return 0
''' )
		self.assertTrue( self.discovery.errors.errors )
