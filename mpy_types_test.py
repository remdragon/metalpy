# stdlib imports:
import copy
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

class NameDeepcopyIdentityTestCase( unittest.TestCase ):
	''' Name/Type/Function/... instances are process-wide identity-based
	singletons (interned via Discovery._get_or_create_specialization and
	friends) - copy.deepcopy() reaching one of these (typically via an AST
	node's own cached resolved-reference tag, e.g. node.resolved_callee)
	must never clone it, or every identity-keyed cache downstream silently
	corrupts. Confirmed via a real repro: type_resolver.py's
	_apply_live_flag_guards deep-copies a generator's promoted-field
	assignment statement to build its "first assignment" branch - when
	that statement's own value expression carried a resolved_callee tag
	pointing at a real Function, deepcopy recursively cloned the Function's
	entire return-type graph (including supposedly-singleton scalars like
	i32), producing a second, non-identical object with the same qualname.
	Two DIFFERENT generic instantiations of one shared generator sharing a
	match-subject-crossing-a-yield promoted field then registered both the
	original and the clone as compile units with the same qualname,
	crashing with a RecursionError inside TaggedUnion's own dataclass
	__eq__ (infinite structural comparison, no true cycle needed - just an
	unbounded diamond of "equal but not the same object" nodes). '''

	def test_deepcopy_of_a_type_object_returns_the_same_instance( self ) -> None:
		rc = RCClass( stem = 'Foo', qualname = 'mymod.Foo', file = None, line = None )
		self.assertIs( copy.deepcopy( rc ), rc )

	def test_deepcopy_of_a_container_holding_a_type_object_preserves_identity( self ) -> None:
		# the realistic shape: an AST node (here, a plain dict standing in
		# for one) holds a reference to a Type/Function - deep-copying the
		# CONTAINER (as _apply_live_flag_guards does for a whole ast.stmt)
		# must still leave every Name-family value inside it untouched.
		i32 = Scalar( stem = 'i32', qualname = 'intrinsics.i32', file = None, line = None, sizeof = 4 )
		rc = RCClass( stem = 'Foo', qualname = 'mymod.Foo', file = None, line = None )
		fn = Function( stem = 'bar', qualname = 'mymod.Foo.bar', file = None, line = None, cls = rc, node = None )
		fn.return_type = i32
		holder = { 'resolved_callee': fn, 'nested': [ rc, i32 ] }

		cloned = copy.deepcopy( holder )

		self.assertIsNot( cloned, holder ) # the container itself DOES copy
		self.assertIs( cloned[ 'resolved_callee' ], fn )
		self.assertIs( cloned[ 'resolved_callee' ].return_type, i32 )
		self.assertIs( cloned[ 'nested' ][0], rc )
		self.assertIs( cloned[ 'nested' ][1], i32 )

if __name__ == '__main__':
	unittest.main()
