# stdlib imports:
from pathlib import Path
import unittest

# local imports:
from discovery import Discovery
from monomorphize import Monomorphizer
from mpy_types import Overload, TaggedUnion, Variable
from union_storage import UnionStorage

class MonomorphizeTests( unittest.TestCase ):
	''' Monomorphizer built directly against a real Discovery instance (same
	discipline as cfg_test.py) - never lowering.py itself, since
	Monomorphizer depends only on Discovery/schedule/UnionStorage. '''

	def setUp( self ) -> None:
		self.discovery = Discovery( import_builtins = False )
		self.scheduled: list[object] = []
		self.union_storage = UnionStorage( self.discovery, self.scheduled.append )
		self.monomorphizer = Monomorphizer( self.discovery, self.scheduled.append, self.union_storage )

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

	def test_monomorphize_class_leaves_overload_group_untouched( self ) -> None:
		mod = self._import( '''
class Box[T]:
	v: T
	@overload
	@staticmethod
	def make( x: i32 ) -> i32:
		return x
	@overload
	@staticmethod
	def make( x: usize ) -> i32:
		return 0
''' )
		box_cls = self._cls( mod, 'Box' )
		i32_cls = self.discovery.get_intrinsics()['i32']
		spec = self.discovery._get_or_create_specialization( box_cls, [ i32_cls ] )
		monomorphized = self.monomorphizer.monomorphize_class( spec )
		self.assertIsInstance( monomorphized.names['make'], Overload )
		self.assertIs( monomorphized.names['make'], box_cls.names['make'] ) # same object - not substituted

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
