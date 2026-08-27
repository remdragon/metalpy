# Real-compile-and-run regression tests for two related fixes:
#
# 1. overload_resolution.py's own wildcard-tie bug: two @overload candidates
#    of the same arity, both bare-TypeVar-typed at the same slot, differing
#    only in that TypeVar's own .bound, used to always resolve to whichever
#    was declared first - the bound was never consulted. Fixed via resolve_
#    call's new protocol_conforms disambiguation (see overload_resolution.py
#    and overload_resolution_test.py's own pure unit tests for the engine
#    itself). PickByProtocolTests below exercises it through an ordinary,
#    non-construction overloaded free function.
#
# 2. Construction dispatch through an overloaded __init__ (previously a hard
#    compile error - "__init__ is overloaded - not supported yet") plus the
#    combined class+own-type-param generic inference gap that blocked it
#    (a generic class's __init__ declaring its OWN extra type param, e.g.
#    list[T].__init__[S: IteratorProtocol[T]], left T unsubstituted - see
#    Lowering._lower_generic_construction_args). ListInitOverloadTests below
#    exercises list[T]'s own new iterator/iterable overloads, in the SAME
#    compile unit as the untouched plain capacity constructor AND a list
#    literal (a THIRD, zero-arg-only caller into the same overload group) -
#    guards against a C-symbol collision between the three, the same bug
#    class as commit 0e470b1 (Pattern.finditer's bytes/memoryview overloads).
#
# 3. $$__new__ symbol collision when the SAME __init__ overload leaf (one
#    single .line - e.g. the IteratorProtocol[T] leaf above) is monomorphized
#    for its OWN type param (S) more than once in one compile unit, with two
#    DIFFERENT concrete S (two distinct synthesized Generator backing types).
#    type_resolver.py's _synthesize_rcclass_constructor used to disambiguate
#    an overloaded __init__'s $$__new__ symbol by init.line alone - correct
#    across sibling overload leaves (always distinct lines) but not across
#    two instantiations of the SAME leaf, which share one line. Two
#    genuinely different C struct parameter types ended up declared/defined
#    under the identical mangled symbol (a real repro: clang "conflicting
#    types", one call site silently passed the wrong struct to the other
#    specialization). ListInitOverloadMultipleGeneratorSpecializationsTests
#    below guards against it: two distinct generator functions, both
#    consumed by list(...) for the same element type.

import unittest

import test_support
from test_support import RealCompileMixin

_PICK_BY_PROTOCOL_BOUND = '''
@protocol
class ProtoA:
	def tag( self ) -> i32: ...

@protocol
class ProtoB:
	def tag( self ) -> i32: ...

class ImplA( ProtoA ):
	def tag( self ) -> i32:
		return 1

class ImplB( ProtoB ):
	def tag( self ) -> i32:
		return 2

@overload
def pick[S: ProtoA]( x: S ) -> i32:
	with compiler.wrap_arithmetic:
		return 100 + x.tag()

@overload
def pick[S: ProtoB]( x: S ) -> i32:
	with compiler.wrap_arithmetic:
		return 200 + x.tag()

def main() -> i32:
	a: ImplA = ImplA()
	b: ImplB = ImplB()
	# before the fix, pick(b) also resolved to the FIRST-declared candidate
	# (pick[S:ProtoA]) regardless of b's own type, since neither wildcard
	# candidate's bound was ever consulted
	if pick( a ) != 101:
		return 1
	if pick( b ) != 202:
		return 2
	return 0
'''

_LIST_INIT_OVERLOADS = '''
class Elem:
	v: i32
	def __init__( self, v: i32 ) -> None:
		self.v = v

def gen_elems( n: i32 ) -> Iterator[Result[Elem,StopIteration]]:
	with compiler.wrap_arithmetic:
		i: i32 = 0
		while i < n:
			yield Elem( v = i )
			i += 1

def main() -> i32:
	with compiler.wrap_arithmetic:
		# plain capacity-based constructor(s) - untouched path, same compile
		# unit/same concrete class (list[i32]) as every other check below
		cap_a: list[i32] = list[i32]()
		cap_b: list[i32] = list[i32]( 16 )
		if cap_a.capacity() != 8:
			return 1
		if cap_b.capacity() != 16:
			return 2

		# construction from an IteratorProtocol[T] (a real generator) - RC element
		xs: list[Elem] = list( gen_elems( 3 ) )
		if len( xs ) != 3:
			return 3
		if xs.__getitem__( 0 ).unwrap( 'xs[0]' ).v != 0:
			return 4
		if xs.__getitem__( 2 ).unwrap( 'xs[2]' ).v != 2:
			return 5

		# construction from an Iterable[T] (another list instance) - RC
		# element, with a real refcount check: list(src) must take its own
		# independent reference per element, not alias src's
		e0: Elem = Elem( v = 100 )
		src: list[Elem] = list[Elem]()
		src.append( e0 )
		src.append( Elem( v = 101 ) )
		before: usize = compiler.refcount( e0 )
		ys: list[Elem] = list( src )
		after: usize = compiler.refcount( e0 )
		if len( ys ) != 2:
			return 6
		if ys.__getitem__( 0 ).unwrap( 'ys[0]' ).v != 100:
			return 7
		if after != before + 1:
			return 8

		# a list LITERAL - a third, zero-argument-only caller into the same
		# overloaded __init__ group (Lowering._construct_generic_instance,
		# a separate code path from ordinary ClassName(...) construction)
		zs: list[i32] = [ 1, 2, 3 ]
		if len( zs ) != 3:
			return 9
	return 0
'''

@unittest.skipUnless( test_support.HAS_CC, 'no C compiler (clang/gcc/msvc) found - skipping real-compile RC tests' )
class PickByProtocolBoundTests( RealCompileMixin, unittest.TestCase ):
	def test_wildcard_candidates_disambiguated_by_typevar_bound( self ) -> None:
		self.assert_programs_run([
			( 'pick_by_protocol_bound', _PICK_BY_PROTOCOL_BOUND ),
		])

_LIST_INIT_MULTIPLE_GENERATOR_SPECIALIZATIONS = '''
class Elem:
	v: i32
	def __init__( self, v: i32 ) -> None:
		self.v = v

# two DISTINCT generator functions - each gets its own synthesized Generator
# backing type, even though both conform to IteratorProtocol[Elem] and both
# feed the SAME list[Elem].__init__[S: IteratorProtocol[Elem]] overload leaf
def gen_elems_a( n: i32 ) -> Iterator[Result[Elem,StopIteration]]:
	with compiler.wrap_arithmetic:
		i: i32 = 0
		while i < n:
			yield Elem( v = i )
			i += 1

def gen_elems_b( n: i32 ) -> Iterator[Result[Elem,StopIteration]]:
	with compiler.wrap_arithmetic:
		i: i32 = 0
		while i < n:
			yield Elem( v = 100 + i )
			i += 1

def main() -> i32:
	with compiler.wrap_arithmetic:
		xs: list[Elem] = list( gen_elems_a( 3 ) )
		ys: list[Elem] = list( gen_elems_b( 2 ) )
		if len( xs ) != 3:
			return 1
		if xs.__getitem__( 0 ).unwrap( 'xs[0]' ).v != 0:
			return 2
		if xs.__getitem__( 2 ).unwrap( 'xs[2]' ).v != 2:
			return 3
		if len( ys ) != 2:
			return 4
		if ys.__getitem__( 0 ).unwrap( 'ys[0]' ).v != 100:
			return 5
		if ys.__getitem__( 1 ).unwrap( 'ys[1]' ).v != 101:
			return 6
	return 0
'''

@unittest.skipUnless( test_support.HAS_CC, 'no C compiler (clang/gcc/msvc) found - skipping real-compile RC tests' )
class ListInitOverloadTests( RealCompileMixin, unittest.TestCase ):
	def test_list_init_overloads_iterator_iterable_and_capacity( self ) -> None:
		self.assert_programs_run([
			( 'list_init_overloads', _LIST_INIT_OVERLOADS ),
		])

@unittest.skipUnless( test_support.HAS_CC, 'no C compiler (clang/gcc/msvc) found - skipping real-compile RC tests' )
class ListInitOverloadMultipleGeneratorSpecializationsTests( RealCompileMixin, unittest.TestCase ):
	def test_same_overload_leaf_two_distinct_generator_specializations( self ) -> None:
		self.assert_programs_run([
			( 'list_init_multiple_generator_specializations', _LIST_INIT_MULTIPLE_GENERATOR_SPECIALIZATIONS ),
		])

if __name__ == '__main__':
	unittest.main()
