# stdlib imports:
from pathlib import Path
import unittest

# local imports:
from discovery import Discovery
from errors import CompileError
from mpy_types import Module
import overload_resolution as OR

class OverloadCallResolutionTests( unittest.TestCase ):
	''' overload_resolution.resolve_call - pure function of types, no AST/call-site involved, so this is testable ahead of stage 2. Migrated from discovery_test.py's old OverloadCallResolutionTests (which tested Overload.resolve_call directly, before that method moved here). '''

	def setUp( self ) -> None:
		self.discovery = Discovery( import_builtins = False )

	def _import( self, code: str ) -> Module:
		return self.discovery.import_code( code, Path( '__main__.py' ), scope = None )

	def _resolve( self, group, args, kwargs ):
		return OR.resolve_call( group.stubs, group.implementations, args, kwargs, qualname = group.qualname )

	def _worked_example( self ):
		mod = self._import( '''
class bool: pass
class int: pass
class str: pass
class bytes: pass

@overload
def foo( x: int ) -> None:
	...

@overload
def foo( x: str = '' ) -> None:
	...

def foo( x: int|None = None ) -> None:
	pass

def foo( x: str|bytes ) -> None:
	pass
''' )
		group = mod.get_local( 'foo' )
		foo3, foo4 = group.implementations
		return (
			group, foo3, foo4,
			mod.get_local( 'bool' ), mod.get_local( 'int' ), mod.get_local( 'str' ), mod.get_local( 'bytes' ),
		)

	def test_int_resolves_via_first_stub( self ) -> None:
		group, foo3, foo4, bool_cls, int_cls, str_cls, bytes_cls = self._worked_example()
		branches, default = self._resolve( group, [ int_cls ], {} )
		self.assertEqual( branches, [] )
		self.assertIs( default, foo3 )

	def test_str_resolves_via_second_stub( self ) -> None:
		group, foo3, foo4, bool_cls, int_cls, str_cls, bytes_cls = self._worked_example()
		branches, default = self._resolve( group, [ str_cls ], {} )
		self.assertEqual( branches, [] )
		self.assertIs( default, foo4 )

	def test_none_falls_through_to_unique_implementation( self ) -> None:
		group, foo3, foo4, bool_cls, int_cls, str_cls, bytes_cls = self._worked_example()
		branches, default = self._resolve( group, [ self.discovery.get_none_type() ], {} )
		self.assertEqual( branches, [] )
		self.assertIs( default, foo3 )

	def test_bytes_falls_through_to_unique_implementation( self ) -> None:
		group, foo3, foo4, bool_cls, int_cls, str_cls, bytes_cls = self._worked_example()
		branches, default = self._resolve( group, [ bytes_cls ], {} )
		self.assertEqual( branches, [] )
		self.assertIs( default, foo4 )

	def test_union_argument_produces_conditional_dispatch( self ) -> None:
		group, foo3, foo4, bool_cls, int_cls, str_cls, bytes_cls = self._worked_example()
		union = self.discovery._get_or_create_union( [ int_cls, bytes_cls ] )
		branches, default = self._resolve( group, [ union ], {} )
		self.assertEqual( len( branches ), 1 )
		self.assertIs( default, foo4 )
		self.assertIs( branches[0].function, foo3 )
		self.assertEqual( len( branches[0].conditions ), 1 )
		param, expected = branches[0].conditions[0]
		self.assertIs( param, foo3.parameters[0] )
		self.assertIs( expected, int_cls )

	def test_uncovered_type_errors( self ) -> None:
		# resolve_call is a pure function of types with no Discovery reference
		# by design - it raises CompileError directly, unrecorded; a real call
		# site (lowering.py) attaches location and records it via Discovery.fail()
		group, foo3, foo4, bool_cls, int_cls, str_cls, bytes_cls = self._worked_example()
		with self.assertRaises( CompileError ):
			self._resolve( group, [ bool_cls ], {} )

	def test_kwargs_match_differently_named_parameters( self ) -> None:
		mod = self._import( '''
class int: pass
class str: pass

@overload
def f( a: int ) -> None:
	...

@overload
def f( b: str ) -> None:
	...

def f( a: int ) -> None:
	pass

def f( b: str ) -> None:
	pass
''' )
		group = mod.get_local( 'f' )
		int_impl, str_impl = group.implementations
		int_cls = mod.get_local( 'int' )
		str_cls = mod.get_local( 'str' )

		branches, default = self._resolve( group, [], { 'a': int_cls } )
		self.assertEqual( branches, [] )
		self.assertIs( default, int_impl )

		branches, default = self._resolve( group, [], { 'b': str_cls } )
		self.assertEqual( branches, [] )
		self.assertIs( default, str_impl )

	def test_omitted_parameter_with_default_still_matches( self ) -> None:
		mod = self._import( '''
class int: pass
class str: pass

@overload
def f( a: int, b: str = '' ) -> None:
	...

def f( a: int, b: str = '' ) -> None:
	pass
''' )
		group = mod.get_local( 'f' )
		impl = group.implementations[0]
		int_cls = mod.get_local( 'int' )

		branches, default = self._resolve( group, [ int_cls ], {} ) # 'b' entirely omitted
		self.assertEqual( branches, [] )
		self.assertIs( default, impl )

	def test_multi_parameter_cartesian_product( self ) -> None:
		mod = self._import( '''
class int: pass
class str: pass

@overload
def pair( a: int, b: int ) -> None:
	...

def pair( a: int, b: int ) -> None:
	pass
def pair( a: int, b: str ) -> None:
	pass
def pair( a: str, b: int ) -> None:
	pass
def pair( a: str, b: str ) -> None:
	pass
''' )
		group = mod.get_local( 'pair' )
		p_ii, p_is, p_si, p_ss = group.implementations
		int_cls = mod.get_local( 'int' )
		str_cls = mod.get_local( 'str' )
		union = self.discovery._get_or_create_union( [ int_cls, str_cls ] )

		branches, default = self._resolve( group, [ union, union ], {} )
		target_ids = { id( b.function ) for b in branches } | { id( default ) }
		self.assertEqual( target_ids, { id( p_ii ), id( p_is ), id( p_si ), id( p_ss ) })


class TodoWorkedExampleTests( unittest.TestCase ):
	''' TODO.txt's own worked-through examples (the design this module implements) - each @overload-decorated with a real body (metalpy's own flexibility beyond Python's @overload convention: multiple real-bodied @overload members are allowed, priority-ordered by declaration, no shared single implementation required) '''

	def setUp( self ) -> None:
		self.discovery = Discovery( import_builtins = False )

	def _import( self, code: str ) -> Module:
		return self.discovery.import_code( code, Path( '__main__.py' ), scope = None )

	def _resolve( self, group, args, kwargs ):
		return OR.resolve_call( group.stubs, group.implementations, args, kwargs, qualname = group.qualname )

	def test_two_overload_no_fallback_is_a_compile_error( self ) -> None:
		# foo(A,A) and foo(B,B) only - called with x,y: A|B. (x:B,y:A) and
		# (x:A,y:B) are never covered by either - must fail to compile, and
		# the error should name both unmatched states
		mod = self._import( '''
class A: pass
class B: pass

@overload
def foo( v: A, w: A ) -> None:
	pass

@overload
def foo( v: B, w: B ) -> None:
	pass
''' )
		group = mod.get_local( 'foo' )
		a_cls = mod.get_local( 'A' )
		b_cls = mod.get_local( 'B' )
		union = self.discovery._get_or_create_union( [ a_cls, b_cls ] )
		with self.assertRaises( CompileError ) as ctx:
			self._resolve( group, [ union, union ], {} )
		message = str( ctx.exception )
		self.assertIn( 'A', message )
		self.assertIn( 'B', message )

	def test_three_overload_with_broad_fallback_compiles( self ) -> None:
		# adding foo(A|B,A|B) as a third @overload absorbs everything the
		# first two leave unmatched - must compile, with the broad overload
		# as the (possibly one of several, but ultimately unconditional) default
		mod = self._import( '''
class A: pass
class B: pass

@overload
def foo( v: A, w: A ) -> None:
	pass

@overload
def foo( v: B, w: B ) -> None:
	pass

@overload
def foo( v: A|B, w: A|B ) -> None:
	pass
''' )
		group = mod.get_local( 'foo' )
		foo_aa, foo_bb, foo_abab = group.implementations
		a_cls = mod.get_local( 'A' )
		b_cls = mod.get_local( 'B' )
		union = self.discovery._get_or_create_union( [ a_cls, b_cls ] )
		branches, default = self._resolve( group, [ union, union ], {} )
		# every branch/default target must be one of the three real
		# candidates - nothing else could ever be scheduled
		all_targets = { id( b.function ) for b in branches } | { id( default ) }
		self.assertTrue( all_targets <= { id( foo_aa ), id( foo_bb ), id( foo_abab ) })
		# foo(A,A) is reached unconditionally when both args really are A -
		# it must appear with a real (non-empty) condition somewhere (it's
		# never the trailing default, since the broad overload always is)
		aa_branches = [ b for b in branches if b.function is foo_aa ]
		self.assertEqual( len( aa_branches ), 1 )
		self.assertTrue( aa_branches[0].conditions )

	def test_genuine_multi_condition_branch( self ) -> None:
		# a branch whose reachability depends on BOTH parameters at once -
		# today's lowering.py (before this refactor) could only ever express
		# a single condition per branch; this is the concrete case that
		# couldn't be built at all previously
		mod = self._import( '''
class A: pass
class B: pass

@overload
def foo( v: A, w: A ) -> None:
	pass

@overload
def foo( v: B, w: B ) -> None:
	pass

@overload
def foo( v: A|B, w: A|B ) -> None:
	pass
''' )
		group = mod.get_local( 'foo' )
		foo_aa, foo_bb, foo_abab = group.implementations
		a_cls = mod.get_local( 'A' )
		b_cls = mod.get_local( 'B' )
		union = self.discovery._get_or_create_union( [ a_cls, b_cls ] )
		branches, default = self._resolve( group, [ union, union ], {} )
		multi_condition_branches = [ b for b in branches if len( b.conditions ) > 1 ]
		self.assertTrue( multi_condition_branches, 'expected at least one branch with more than one condition' )
		# foo(A,A) is exactly this case: reachable only when v=A AND w=A
		aa = next( b for b in branches if b.function is foo_aa )
		self.assertEqual( len( aa.conditions ), 2 )
		params = { p.stem for p, _ in aa.conditions }
		self.assertEqual( params, { 'v', 'w' } )

	def test_leftover_after_overload_sweep_falls_to_exactly_one_plain( self ) -> None:
		# an @overload group whose sweep doesn't fully absorb everything -
		# whatever's left must match exactly one plain (non-@overload)
		# implementation, which becomes the unconditional else
		mod = self._import( '''
class A: pass
class B: pass

@overload
def foo( v: A ) -> None:
	pass

def foo( v: A|B ) -> None:
	pass
''' )
		group = mod.get_local( 'foo' )
		foo_a = group.implementations[0]     # the @overload member
		foo_fallback = group.implementations[1]  # the plain catch-all
		a_cls = mod.get_local( 'A' )
		b_cls = mod.get_local( 'B' )
		union = self.discovery._get_or_create_union( [ a_cls, b_cls ] )
		branches, default = self._resolve( group, [ union ], {} )
		self.assertEqual( len( branches ), 1 )
		self.assertIs( branches[0].function, foo_a )
		self.assertIs( default, foo_fallback )

	def test_leftover_ambiguous_between_plains_is_a_compile_error( self ) -> None:
		# whatever's left after the @overload sweep must match EXACTLY one
		# plain - if two plains both structurally accept it, that's still
		# an error, same as the historical plain-only ambiguity check
		mod = self._import( '''
class A: pass
class B: pass

@overload
def foo( v: A ) -> None:
	pass

def foo( v: A|B ) -> None:
	pass

def foo( v: B ) -> None:
	pass
''' )
		group = mod.get_local( 'foo' )
		a_cls = mod.get_local( 'A' )
		b_cls = mod.get_local( 'B' )
		union = self.discovery._get_or_create_union( [ a_cls, b_cls ] )
		with self.assertRaises( CompileError ):
			self._resolve( group, [ union ], {} )

	def test_overload_sweep_fully_absorbing_never_consults_plains( self ) -> None:
		# if the @overload members fully absorb the whole argument space,
		# no plain implementation is even consulted - not just "resolves
		# unambiguously despite plains existing", but genuinely never asked.
		# proven by making the "plain" a decoy that would raise if binds()
		# were ever even attempted against it (a parameter type that was
		# never declared/resolvable would blow up _translate_indices's own
		# lookup if reached) - simplest proof is checking every branch/
		# default target is one of the TWO @overload members, never the plain
		mod = self._import( '''
class A: pass
class B: pass

@overload
def foo( v: A ) -> None:
	pass

@overload
def foo( v: B ) -> None:
	pass

def foo( v: A|B ) -> None:
	pass
''' )
		group = mod.get_local( 'foo' )
		foo_a, foo_b, foo_plain = group.implementations
		a_cls = mod.get_local( 'A' )
		b_cls = mod.get_local( 'B' )
		union = self.discovery._get_or_create_union( [ a_cls, b_cls ] )
		branches, default = self._resolve( group, [ union ], {} )
		all_targets = { id( b.function ) for b in branches } | { id( default ) }
		self.assertEqual( all_targets, { id( foo_a ), id( foo_b ) })
		self.assertNotIn( id( foo_plain ), all_targets )


class DifferentlyOrderedParametersTests( unittest.TestCase ):
	''' each candidate's kwarg->slot translation is independent - different
	candidates in the same group can have different parameter names AND a
	different parameter order from each other '''

	def setUp( self ) -> None:
		self.discovery = Discovery( import_builtins = False )

	def _import( self, code: str ) -> Module:
		return self.discovery.import_code( code, Path( '__main__.py' ), scope = None )

	def test_both_candidates_applicable_with_differently_ordered_kwargs( self ) -> None:
		mod = self._import( '''
class int: pass
class str: pass
class Bar: pass

def foo( a: int, b: str ) -> None:
	pass

def foo( b: int, a: Bar ) -> None:
	pass
''' )
		group = mod.get_local( 'foo' )
		foo1, foo2 = group.implementations
		int_cls = mod.get_local( 'int' )
		str_cls = mod.get_local( 'str' )
		bar_cls = mod.get_local( 'Bar' )

		# a=int, b=str -> only foo1 (a:int,b:str) accepts this combination
		branches, default = OR.resolve_call( group.stubs, group.implementations, [], { 'a': int_cls, 'b': str_cls }, qualname = group.qualname )
		self.assertEqual( branches, [] )
		self.assertIs( default, foo1 )

		# a=Bar, b=int -> only foo2 (b:int,a:Bar) accepts this combination
		branches, default = OR.resolve_call( group.stubs, group.implementations, [], { 'a': bar_cls, 'b': int_cls }, qualname = group.qualname )
		self.assertEqual( branches, [] )
		self.assertIs( default, foo2 )

	def test_candidate_excluded_when_it_cannot_be_translated( self ) -> None:
		# a candidate missing a parameter the call supplies by name, or
		# missing a value for one of its own required parameters, is simply
		# excluded from consideration - not an error by itself
		mod = self._import( '''
class int: pass
class str: pass

def foo( a: int, b: str ) -> None:
	pass

def foo( c: int ) -> None:
	pass
''' )
		group = mod.get_local( 'foo' )
		foo_ab, foo_c = group.implementations
		int_cls = mod.get_local( 'int' )
		str_cls = mod.get_local( 'str' )
		# foo_c has no parameter named 'b' - excluded; only foo_ab can apply
		branches, default = OR.resolve_call( group.stubs, group.implementations, [], { 'a': int_cls, 'b': str_cls }, qualname = group.qualname )
		self.assertEqual( branches, [] )
		self.assertIs( default, foo_ab )


class UnrelatedCandidatesNeverConsideredTests( unittest.TestCase ):
	''' a candidate whose declared type isn't even a leaf of the call's own
	argument type must never appear in the dispatch plan at all - the
	cartesian product is built purely from the call's OWN leaves, so an
	unrelated candidate is never even checked against, not just "checked
	and correctly excluded" '''

	def setUp( self ) -> None:
		self.discovery = Discovery( import_builtins = False )

	def _import( self, code: str ) -> Module:
		return self.discovery.import_code( code, Path( '__main__.py' ), scope = None )

	def test_len_style_group_with_narrower_union_never_considers_unrelated_candidate( self ) -> None:
		# mirrors lib/builtins/__init__.py's real len() shape: 3 plain
		# candidates (str/bytes/bytearray), called with a NARROWER union
		# (bytes|bytearray, not the full 3-way) - str must never appear
		mod = self._import( '''
class str: pass
class bytes: pass
class bytearray: pass

def len( x: str ) -> usize:
	pass
def len( x: bytes ) -> usize:
	pass
def len( x: bytearray ) -> usize:
	pass
''' )
		group = mod.get_local( 'len' )
		len_str, len_bytes, len_bytearray = group.implementations
		bytes_cls = mod.get_local( 'bytes' )
		bytearray_cls = mod.get_local( 'bytearray' )
		union = self.discovery._get_or_create_union( [ bytes_cls, bytearray_cls ] )

		branches, default = OR.resolve_call( group.stubs, group.implementations, [ union ], {}, qualname = group.qualname )
		all_targets = { id( b.function ) for b in branches } | { id( default ) }
		self.assertEqual( all_targets, { id( len_bytes ), id( len_bytearray ) })
		self.assertNotIn( id( len_str ), all_targets )


class ComplexityCapTests( unittest.TestCase ):
	def setUp( self ) -> None:
		self.discovery = Discovery( import_builtins = False )

	def _import( self, code: str ) -> Module:
		return self.discovery.import_code( code, Path( '__main__.py' ), scope = None )

	def test_comfortably_under_cap_still_succeeds( self ) -> None:
		mod = self._import( '''
class A: pass
class B: pass

def foo( v: A, w: A ) -> None:
	pass
def foo( v: B, w: B ) -> None:
	pass
def foo( v: A, w: B ) -> None:
	pass
def foo( v: B, w: A ) -> None:
	pass
''' )
		group = mod.get_local( 'foo' )
		a_cls = mod.get_local( 'A' )
		b_cls = mod.get_local( 'B' )
		union = self.discovery._get_or_create_union( [ a_cls, b_cls ] )
		branches, default = OR.resolve_call( group.stubs, group.implementations, [ union, union ], {}, qualname = group.qualname )
		self.assertEqual( len( branches ) + 1, 4 )

	def test_trips_cap_with_a_pathologically_wide_group( self ) -> None:
		# a single argument whose union has more leaves than the cap - the
		# cartesian-product size check alone exceeds the budget
		original_cap = OR._MAX_TRACKED_STATES
		OR._MAX_TRACKED_STATES = 4
		try:
			classes = '\n'.join( f'class T{i}: pass' for i in range( 10 ) )
			fns = '\n'.join( f'def foo( v: T{i} ) -> None:\n\tpass' for i in range( 10 ) )
			mod = self._import( f'{classes}\n{fns}\n' )
			group = mod.get_local( 'foo' )
			leaves = [ mod.get_local( f'T{i}' ) for i in range( 10 ) ]
			union = self.discovery._get_or_create_union( leaves )
			with self.assertRaises( CompileError ) as ctx:
				OR.resolve_call( group.stubs, group.implementations, [ union ], {}, qualname = group.qualname )
			self.assertIn( 'too complex', str( ctx.exception ) )
		finally:
			OR._MAX_TRACKED_STATES = original_cap


if __name__ == '__main__':
	unittest.main()
