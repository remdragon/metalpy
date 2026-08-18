# Throwaway spike: does a @union with a list[Self]/dict[str,Self] variant
# actually compile? No test anywhere in this compiler's suite confirms this
# shape works - see the plan for lib/json.py (JSONValue needs exactly this
# shape for its Array/Object variants). This file is deleted once the real
# json_test.py's own first test case absorbs it.

import unittest

import test_support
from test_support import RealCompileMixin


_SELF_REF_LIST = '''
@union
class Node:
	Leaf: i32
	Children: list[Node]

def sum_node( n: Node ) -> i32:
	match n:
		case Node.Leaf( v ):
			return v
		case Node.Children( kids ):
			total: i32 = 0
			i: usize = 0
			with compiler.wrap_arithmetic:
				while i < len( kids ):
					total += sum_node( kids.__getitem__( i ).unwrap( 'spike: index in bounds' ) )
					i += 1
			return total

def main() -> i32:
	kids: list[Node] = list[Node]()
	kids.append( Node.Leaf( 1 ) ).unwrap( 'spike: append failed' )
	kids.append( Node.Leaf( 2 ) ).unwrap( 'spike: append failed' )
	root: Node = Node.Children( kids )
	if sum_node( root ) != 3:
		return 1
	return 0
'''

_SELF_REF_DICT = '''
@union
class Node:
	Leaf: i32
	Children: dict[str, Node]

def sum_node( n: Node ) -> i32:
	match n:
		case Node.Leaf( v ):
			return v
		case Node.Children( kids ):
			total: i32 = 0
			i: usize = 0
			with compiler.wrap_arithmetic:
				while i < len( kids ):
					child: Node = kids.value_at( i ).unwrap( 'spike: index in bounds' )
					total += sum_node( child )
					i += 1
			return total

def main() -> i32:
	kids: dict[str, Node] = dict[str, Node]()
	kids.__setitem__( 'a', Node.Leaf( 1 ) )
	kids.__setitem__( 'b', Node.Leaf( 2 ) )
	root: Node = Node.Children( kids )
	if sum_node( root ) != 3:
		return 1
	return 0
'''


@unittest.skipUnless( test_support.HAS_CC, 'no C compiler (clang/gcc/msvc) found - skipping real-compile spike' )
class SelfReferentialGenericSpikeTests( RealCompileMixin, unittest.TestCase ):
	def setUp( self ) -> None:
		from discovery import Discovery
		from compiler import Compiler
		self.discovery = Discovery( import_builtins = True )
		self.compiler = Compiler( self.discovery )

	def test_self_referential_list_and_dict_variants_compile( self ) -> None:
		self.assert_programs_run([
			( 'self_ref_list', _SELF_REF_LIST ),
			( 'self_ref_dict', _SELF_REF_DICT ),
		])


if __name__ == '__main__':
	unittest.main()
