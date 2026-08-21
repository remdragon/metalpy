# stdlib imports:
import time
import unittest

# local imports:
from mpy_types import Scalar, RCClass, Function, Specialization

def _scalar( stem: str ) -> Scalar:
	return Scalar( stem = stem, qualname = f'intrinsics.{stem}', file = None, line = None, sizeof = 0 )

class TypeReprCycleTestCase( unittest.TestCase ):
	''' repr() on a Type-family value must never recurse into another Type's
	own fields - not just for a direct object-identity cycle (already guarded
	by dataclasses' own reprlib.recursive_repr), but for a DAG where the same
	object is reachable via multiple sibling fields, which recursive_repr does
	NOT catch (it only blocks re-entering an object already on the CURRENT
	repr call stack, not a second, sibling path to the same object) - see
	emitter_c.py:912's assertion, whose own f'{base!r}' hit exactly this and
	hung consuming unbounded memory instead of raising promptly. '''

	def test_self_referencing_cycle_reprs_promptly( self ) -> None:
		rc = RCClass( stem = 'Foo', qualname = 'mymod.Foo', file = None, line = None )
		fn = Function( stem = 'bar', qualname = 'mymod.Foo.bar', file = None, line = None, cls = rc, node = None )
		rc.methods.append( fn )

		start = time.perf_counter()
		text = repr( rc )
		self.assertLess( time.perf_counter() - start, 1.0 )
		self.assertEqual( text, "<RCClass 'mymod.Foo'>" )

	def test_diamond_dag_reprs_promptly_without_exponential_blowup( self ) -> None:
		# each level's Specialization references the PREVIOUS level twice
		# (base + two args) - a naive recursive repr re-expands the whole
		# subtree at every convergence, costing O(3^depth) with no true cycle
		# anywhere (confirmed: depth 10 produced a 9.7MB repr pre-fix)
		prev = _scalar( 'i32' )
		for i in range( 60 ):
			prev = Specialization( stem = 'S', qualname = f'S{i}', file = None, line = None, base = prev, args = [ prev, prev ] )

		start = time.perf_counter()
		text = repr( prev )
		self.assertLess( time.perf_counter() - start, 1.0 )
		self.assertEqual( text, "<Specialization 'S59'>" )

if __name__ == '__main__':
	unittest.main()
