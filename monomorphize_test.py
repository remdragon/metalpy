# stdlib imports:
from pathlib import Path
import unittest

# local imports:
from discovery import Discovery
from monomorphize import Monomorphizer
from mpy_types import Overload, TaggedUnion, Variable
from tuple_storage import TupleStorage
from union_storage import UnionStorage

class MonomorphizeTests( unittest.TestCase ):
	''' Monomorphizer built directly against a real Discovery instance (same
	discipline as cfg_test.py) - never lowering.py itself, since
	Monomorphizer depends only on Discovery/schedule/UnionStorage/
	TupleStorage. '''

	def setUp( self ) -> None:
		self.discovery = Discovery( import_builtins = False )
		self.scheduled: list[object] = []
		self.union_storage = UnionStorage( self.discovery, self.scheduled.append )
		self.tuple_storage = TupleStorage( self.discovery, self.scheduled.append )
		self.monomorphizer = Monomorphizer( self.discovery, self.scheduled.append, self.union_storage, self.tuple_storage )

	def _import( self, code: str ):
		return self.discovery.import_code( code, filename = Path( '__test__.py' ))

	def _cls( self, mod, name: str ):
		cls = mod.get_local( name )
		if cls.resolve is not None:
			cls.resolve()
		return cls

	def test_monomorphize_class_substitutes_field_type_in_both_attributes_and_names( self ) -> None:
		# regression test: .names used to stay a stale, unsubstituted copy
		# of the abstract base's own names for every field except a
		# TaggedUnion's synthesized 'data' - disagreeing with .attributes,
		# which was already correctly substituted
		mod = self._import( '''
class Box[T]:
	v: T
''' )
		box_cls = self._cls( mod, 'Box' )
		i32_cls = self.discovery.get_intrinsics()['i32']
		spec = self.discovery._get_or_create_specialization( box_cls, [ i32_cls ] )
		monomorphized = self.monomorphizer.monomorphize_class( spec )
		self.assertIs( monomorphized.attributes[0].type, i32_cls )
		self.assertIs( monomorphized.names['v'].type, i32_cls )

	def test_monomorphize_class_substitutes_plain_method_signature( self ) -> None:
		mod = self._import( '''
class Box[T]:
	v: T
	def get( self ) -> T:
		return self.v
''' )
		box_cls = self._cls( mod, 'Box' )
		i32_cls = self.discovery.get_intrinsics()['i32']
		spec = self.discovery._get_or_create_specialization( box_cls, [ i32_cls ] )
		monomorphized = self.monomorphizer.monomorphize_class( spec )
		get_fn = monomorphized.names['get']
		self.assertIsNot( get_fn, box_cls.names['get'] ) # a real, distinct substituted copy
		self.assertIs( get_fn.return_type, i32_cls )
		self.assertIsNone( get_fn.type_params )

	def test_monomorphize_class_substitutes_overload_group( self ) -> None:
		# regression test: an @overload group used to be left exactly as
		# declared - every specialization sharing the SAME abstract Overload
		# object, own candidates still typed with the class's bare TypeVars.
		# lowering.py's _lower_call had to reconstruct a substituted copy by
		# hand, per call site, from the receiver's own type - now
		# monomorphize_class does it once, up front, the same way it
		# already does for a plain (non-overloaded) method
		mod = self._import( '''
class Box[T]:
	v: T
	@overload
	def make( self, x: T ) -> T:
		...
	def make( self, x: T ) -> T:
		return x
''' )
		box_cls = self._cls( mod, 'Box' )
		i32_cls = self.discovery.get_intrinsics()['i32']
		spec = self.discovery._get_or_create_specialization( box_cls, [ i32_cls ] )
		monomorphized = self.monomorphizer.monomorphize_class( spec )
		group = monomorphized.names['make']
		self.assertIsInstance( group, Overload )
		self.assertIsNot( group, box_cls.names['make'] ) # a real, distinct substituted copy
		self.assertEqual( len( group.stubs ), 1 )
		self.assertEqual( len( group.implementations ), 1 )
		for candidate in ( *group.stubs, *group.implementations ):
			self.assertIsNone( candidate.type_params )
			self.assertIs( candidate.return_type, i32_cls ) # every candidate's own T substituted
			self.assertIs( candidate.parameters[0].type, i32_cls ) # self is excluded from .parameters (see discovery.py's add_param)
		# the stub's own .bound_to must be re-pointed at the SUBSTITUTED
		# implementation, not the abstract one it was bound to before
		# monomorphization - _lower_call's own winning_stub lookup matches
		# by `s.bound_to is <the resolved implementation>`, which only
		# ever sees the substituted implementations
		self.assertIs( group.stubs[0].bound_to, group.implementations[0] )
		self.assertIsNot( group.stubs[0].bound_to, box_cls.names['make'].stubs[0].bound_to )

	def test_monomorphize_class_tagged_union_tag_and_data_correct( self ) -> None:
		mod = self._import( '''
@union
class Choice[T,U]:
	First: T
	Second: U
''' )
		choice_cls = self._cls( mod, 'Choice' )
		i32_cls = self.discovery.get_intrinsics()['i32']
		u8_intrinsic = self.discovery.get_intrinsics()['u8']
		spec = self.discovery._get_or_create_specialization( choice_cls, [ i32_cls, i32_cls ] )
		monomorphized = self.monomorphizer.monomorphize_class( spec )
		self.assertIsInstance( monomorphized, TaggedUnion )
		tag_attr = monomorphized.names['tag']
		data_attr = monomorphized.names['data']
		self.assertIsInstance( tag_attr, Variable )
		self.assertIsInstance( data_attr, Variable )
		self.assertIs( tag_attr.type, u8_intrinsic )
		self.assertNotEqual( data_attr.type.qualname, choice_cls.names['data'].type.qualname ) # substituted payload, not the shared abstract one

if __name__ == '__main__':
	unittest.main()
