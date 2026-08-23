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
	lst.append( 3 ).unwrap( 'a' )
	lst.append( 1 ).unwrap( 'b' )
	lst.append( 4 ).unwrap( 'c' )
	lst.append( 1 ).unwrap( 'd' )
	lst.append( 5 ).unwrap( 'e' )
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
	lst.append( 10 ).unwrap( 'a' )
	lst.append( 20 ).unwrap( 'b' )
	lst.append( 30 ).unwrap( 'c' )

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
	lst.append( 1 ).unwrap( 'a' )
	lst.append( 2 ).unwrap( 'b' )
	lst.append( 3 ).unwrap( 'c' )

	doubled: list[i32] = list[i32]()
	for v in map( double, lst ):
		doubled.append( v ).unwrap( 'd' )
	if sum( doubled ) != 12:
		return 1

	total: i32 = reduce( add, lst )
	if total != 6:
		return 2
	return 0
'''

_SLICE_CONFORMANCE = '''
def main() -> i32:
	lst: list[i32] = list[i32]()
	lst.append( 3 ).unwrap( 'a' )
	lst.append( 1 ).unwrap( 'b' )
	lst.append( 4 ).unwrap( 'c' )
	lst.append( 1 ).unwrap( 'd' )
	sl: slice[i32] = lst[1:4]
	if sum( sl ) != 6: # 1+4+1
		return 1
	if min( sl ) != 1:
		return 2
	if max( sl ) != 4:
		return 3
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

	def test_slice_conformance( self ) -> None:
		self.assert_programs_run([
			( 'slice_conformance', _SLICE_CONFORMANCE ),
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
