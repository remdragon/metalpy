# Real-compile-and-run + compile-error regression tests for a general
# "narrower union flows into a wider superset union" coercion - lowering.py's
# _coerce_union_subset, called from _coerce_or_check_operand whenever
# operand.type is itself an anonymous union (e.g. BoxA|BoxB) and every one of
# its own leaves is also a leaf of the expected (wider) union (e.g.
# BoxA|BoxB|BoxC).
#
# Before this existed, only two shapes coerced into a union: a bare leaf
# value (_coerce_into_union's own main case), and a NOMINAL @union class
# nested as ONE opaque member of a wider union (union_widening_test.py's own
# coverage). A structurally-narrower ANONYMOUS union (operand's own leaves
# individually present in, but not identical to, the target union) was
# silently REJECTED - confirmed as a real regression while widening
# lib/builtins/__init__.py's bytes.__init__ to also accept memoryview
# (bytes|bytearray|memoryview): every existing bytes|bytearray-typed caller
# (lib/zipfile.py's own _compress_payload) broke, since a bytes|bytearray
# value no longer satisfied the wider parameter at all.
#
# _coerce_union_subset builds a runtime tag dispatch over operand's own
# members (the same shape _emit_eq_dispatch_tree already uses), extracting
# each member's payload as a bare borrow and re-wrapping it through the
# EXISTING _coerce_into_union machinery against the wider union - so RC
# ownership follows the exact same "ctor increfs, caller cancels the extra
# incref back out when operand was already a fresh temp" convention the
# plain-leaf coercion case already relies on (see _coerce_into_union's own
# is_union_coerce_result comment, and union_coercion_rc_test.py for that
# convention's own dedicated coverage).

# stdlib imports:
from pathlib import Path
import unittest

# local imports:
from compiler import Compiler
from discovery import Discovery
import test_support
from test_support import RealCompileMixin


# --- real compile+run: positive coverage -------------------------------

_FRESH_OPERAND_WIDENS_CORRECTLY = '''
class BoxA:
	n: i32

	def value( self ) -> i32:
		return self.n

class BoxB:
	n: i32

	def value( self ) -> i32:
		return self.n

class BoxC:
	n: i32

	def value( self ) -> i32:
		return self.n

def make( which: bool ) -> BoxA|BoxB:
	if which:
		return BoxA( n = 1 )
	return BoxB( n = 2 )

def widen( which: bool ) -> BoxA|BoxB|BoxC:
	# make(which)'s own Call result is a FRESH temp reaching the coercion -
	# exercises _coerce_union_subset's "was_fresh: decref+untrack pre_coerce"
	# cancellation path. BoxC is never actually produced - it's here only
	# to make BoxA|BoxB a genuine structural SUBSET of the wider union
	# (BoxA|BoxB itself, matching one of its OWN leaves verbatim, would hit
	# _coerce_into_union's existing same-type case instead of this one).
	return make( which )

def main() -> i32:
	r1: BoxA|BoxB|BoxC = widen( True )
	if r1.value() != 1:
		return 1
	r2: BoxA|BoxB|BoxC = widen( False )
	if r2.value() != 2:
		return 2
	return 0
'''

_NON_FRESH_OPERAND_KEEPS_ITS_OWN_REFERENCE = '''
class BoxA:
	n: i32

	def value( self ) -> i32:
		return self.n

class BoxB:
	n: i32

	def value( self ) -> i32:
		return self.n

class BoxC:
	n: i32

	def value( self ) -> i32:
		return self.n

def widen( v: BoxA|BoxB ) -> BoxA|BoxB|BoxC:
	# v is an ordinary, still-alive local (an ALIASING read, not a fresh
	# temp) - exercises _coerce_union_subset's OTHER path: the ctor's own
	# incref must be a real, additional reference (no caller-side
	# cancellation here), since v's own binding still needs its reference
	# after this call returns
	return v

def main() -> i32:
	i: i32 = 0
	is_even: bool = True
	while i < 200:
		v: BoxA|BoxB = BoxA( n = i ) if is_even else BoxB( n = i )
		w: BoxA|BoxB|BoxC = widen( v )
		if w.value() != i:
			return 1
		# v itself must still be valid here - a missing incref during
		# widen()'s own coercion would leave v's reference stolen/
		# double-released by now, an over-eager one would leak every
		# iteration - 200 iterations is enough for either to surface
		if v.value() != i:
			return 2
		is_even = not is_even
		with compiler.wrap_arithmetic:
			i += 1
	return 0
'''

_UPCAST_LEAF_INSIDE_NARROWER_UNION_WIDENS_CORRECTLY = '''
class Animal:
	pass

class Dog( Animal ):
	pass

class Cat( Animal ):
	pass

def make( which: bool ) -> Dog|Cat:
	if which:
		return Dog()
	return Cat()

def widen( which: bool ) -> Animal|None:
	# Dog|Cat -> Animal|None: neither Dog nor Cat is itself a LEAF of
	# Animal|None (Animal is), so each branch needs _coerce_into_union's own
	# RCClass-upcast leaf match, not a same-type one - exercises
	# _coerce_union_subset's own upcast probe
	return make( which )

def main() -> i32:
	if widen( True ) is None:
		return 1
	if widen( False ) is None:
		return 2
	return 0
'''


@unittest.skipUnless( test_support.HAS_CC, 'no C compiler (clang/gcc/msvc) found - skipping real-compile RC tests' )
class UnionSubsetCoercionRealCompileTests( RealCompileMixin, unittest.TestCase ):
	def test_fresh_and_non_fresh_operands_widen_with_correct_rc( self ) -> None:
		self.assert_programs_run([
			( 'fresh_operand_widens', _FRESH_OPERAND_WIDENS_CORRECTLY ),
			( 'non_fresh_operand_keeps_reference', _NON_FRESH_OPERAND_KEEPS_ITS_OWN_REFERENCE ),
		])

	def test_upcast_leaf_inside_narrower_union_widens( self ) -> None:
		self.assert_programs_run([
			( 'upcast_leaf_widens', _UPCAST_LEAF_INSIDE_NARROWER_UNION_WIDENS_CORRECTLY ),
		])


# --- compile-error coverage (no real C compiler needed) -----------------

class UnionSubsetCoercionRejectionTests( unittest.TestCase ):
	def setUp( self ) -> None:
		self.discovery = Discovery( import_builtins = True )
		self.compiler = Compiler( self.discovery )

	def test_leaf_not_covered_by_target_union_is_rejected( self ) -> None:
		# BoxA|BoxC has a leaf (BoxC) the target union (BoxA|BoxB) doesn't
		# cover - _coerce_union_subset must never fire here (its own
		# all(...) leaf-coverage guard), leaving this a genuine, real
		# compile error, same as any other unrelated-type mismatch
		code = '\n'.join([
			'class BoxA:',
			'	pass',
			'class BoxB:',
			'	pass',
			'class BoxC:',
			'	pass',
			'',
			'def make( which: bool ) -> BoxA|BoxC:',
			'	if which:',
			'		return BoxA()',
			'	return BoxC()',
			'',
			'def take( v: BoxA|BoxB ) -> None:',
			'	return',
			'',
			'def main() -> None:',
			'	take( make( True ))',
			'	return',
		])
		self.compiler.import_code( code, Path( '__test__.py' ), scope = None )
		self.compiler.run()
		self.assertNotEqual( self.discovery.errors.errors, [] )


if __name__ == '__main__':
	unittest.main()
