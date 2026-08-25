# Real-compile-and-run regression test: a method declaring its OWN type
# param on top of its enclosing generic class's (e.g. list[T].__init__[S:
# Iterable[T]]'s shape) used to leave the CLASS's own type params entirely
# unsubstituted - monomorphize.py's monomorphized_function treated "the
# function's own type params" and "its enclosing class's type params" as
# either/or, so `self` inside the monomorphized method stayed typed against
# the abstract, unspecialized class (a real RC-layout miscompile risk for any
# RC-typed field), and a sibling type param's own protocol bound
# (S: Iterable[T]) still referenced the abstract T, rejecting every real
# argument at the bound check. See monomorphize.py's
# _partial_class_substituted_method.

import unittest

import test_support
from test_support import RealCompileMixin

_GENERIC_METHOD_OWN_TYPEPARAM_ON_GENERIC_CLASS = '''
class Elem:
	v: i32
	def __init__( self, v: i32 ) -> None:
		self.v = v

class Box[T]:
	v: T
	def __init__( self, v: T ) -> None:
		self.v = v
	def load[S: Iterable[T]]( self, src: S ) -> None:
		for x in src:
			self.v = x

def main() -> i32:
	with compiler.wrap_arithmetic:
		e0: Elem = Elem( v = 1 )
		b: Box[Elem] = Box( v = e0 )
		xs: list[Elem] = list[Elem]()
		e1: Elem = Elem( v = 2 )
		xs.append( e1 )
		before: usize = compiler.refcount( e1 )
		b.load( xs )
		after: usize = compiler.refcount( e1 )
		# self ended up typed against the CONCRETE Box[Elem] specialization
		# (not the abstract, unresolved Box) - b.v really is the RC-typed
		# Elem field, and assigning into it through the generic method took
		# a genuine, correctly-counted extra reference
		if b.v.v != 2:
			return 1
		if after != before + 1:
			return 2
	return 0
'''

@unittest.skipUnless( test_support.HAS_CC, 'no C compiler (clang/gcc/msvc) found - skipping real-compile RC tests' )
class GenericMethodOwnTypeParamTests( RealCompileMixin, unittest.TestCase ):
	def test_generic_method_own_typeparam_bakes_in_class_typeparam( self ) -> None:
		self.assert_programs_run([
			( 'generic_method_own_typeparam_on_generic_class', _GENERIC_METHOD_OWN_TYPEPARAM_ON_GENERIC_CLASS ),
		])

if __name__ == '__main__':
	unittest.main()
