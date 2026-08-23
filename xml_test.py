# Real-compile-and-run behavioral tests for lib/xml.py (well-formed XML
# parsing into a DOM-style tree, plus serialization). Mirrors json_test.py's
# own structure and RealCompileMixin usage.

import unittest

import test_support
from test_support import RealCompileMixin
from discovery import Discovery
from compiler import Compiler


# ---------------------------------------------------------------------------
# 0. Required first test: does a @union whose payload is an RC class that
# itself holds list[Self] (union -> class -> list[union], ONE level removed
# from json.py's own union -> list[union] shape) actually compile? xml.py's
# ElementData.children needs exactly this. Manually confirmed via a
# standalone mpy.py compile+run before any of xml.py was written - this is
# that same spike, kept as a permanent regression test.
# ---------------------------------------------------------------------------

_SELF_REF_UNION_VIA_CLASS = '''
class BranchData:
	tag: str
	children: list[Node]

	def __init__( self, tag: str, children: list[Node] ) -> None:
		self.tag = tag
		self.children = children

@union
class Node:
	Leaf: i32
	Branch: BranchData

def sum_node( n: Node ) -> i32:
	match n:
		case Node.Leaf( v ):
			return v
		case Node.Branch( b ):
			total: i32 = 0
			i: usize = 0
			with compiler.wrap_arithmetic:
				while i < len( b.children ):
					total += sum_node( b.children.__getitem__( i ).unwrap( 'spike: index in bounds' ) )
					i += 1
			return total

def main() -> i32:
	kids: list[Node] = list[Node]()
	kids.append( Node.Leaf( 1 ) ).unwrap( 'spike: append failed' )
	kids.append( Node.Leaf( 2 ) ).unwrap( 'spike: append failed' )
	root: Node = Node.Branch( BranchData( 'root', kids ) )
	if sum_node( root ) != 3:
		return 1
	return 0
'''


# ---------------------------------------------------------------------------
# 1. Basic parsing: single element, nested elements, self-closing, attrs.
# ---------------------------------------------------------------------------

_BASIC_PARSING = '''
import xml

def main() -> i32:
	n1: xml.XMLNode = xml.parse( '<a/>' ).unwrap( 'parse a' )
	e1: xml.ElementData = n1.as_element().unwrap( 'as_element' )
	if e1.tag != 'a':
		return 1
	if len( e1.children ) != 0:
		return 2

	n2: xml.XMLNode = xml.parse( '<a><b/><c/></a>' ).unwrap( 'parse nested' )
	kids: list[xml.ElementData] = n2.child_elements()
	if len( kids ) != 2:
		return 3
	first: xml.ElementData = kids.__getitem__( 0 ).unwrap( 'index' )
	if first.tag != 'b':
		return 4
	second: xml.ElementData = kids.__getitem__( 1 ).unwrap( 'index' )
	if second.tag != 'c':
		return 5

	n3: xml.XMLNode = xml.parse( '<a x="1" y="two"></a>' ).unwrap( 'parse attrs' )
	e3: xml.ElementData = n3.as_element().unwrap( 'as_element' )
	if e3.get_attr( 'x' ).unwrap( 'attr x' ) != '1':
		return 6
	if e3.get_attr( 'y' ).unwrap( 'attr y' ) != 'two':
		return 7
	match e3.get_attr( 'missing' ):
		case Result.Ok( _ ):
			return 8
		case Result.Err( _ ):
			pass

	n4: xml.XMLNode = xml.parse( '  <a>  </a>  ' ).unwrap( 'parse with surrounding ws' )
	e4: xml.ElementData = n4.as_element().unwrap( 'as_element' )
	if e4.tag != 'a':
		return 9

	n5: xml.XMLNode = xml.parse( '<?xml version="1.0"?><!DOCTYPE a><a/>' ).unwrap( 'parse with prolog' )
	e5: xml.ElementData = n5.as_element().unwrap( 'as_element' )
	if e5.tag != 'a':
		return 10
	return 0
'''


# ---------------------------------------------------------------------------
# 2. Entity decoding: predefined + numeric (decimal and hex).
# ---------------------------------------------------------------------------

_ENTITIES = '''
import xml

def text_of( doc: str ) -> str:
	n: xml.XMLNode = xml.parse( doc ).unwrap( 'parse failed for ' + doc )
	return n.text_content()

def main() -> i32:
	if text_of( '<a>&amp;</a>' ) != '&':
		return 1
	if text_of( '<a>&lt;</a>' ) != '<':
		return 2
	if text_of( '<a>&gt;</a>' ) != '>':
		return 3
	if text_of( '<a>&quot;</a>' ) != '"':
		return 4
	if text_of( '<a>&apos;</a>' ) != chr( 0x27 ):
		return 5
	if text_of( '<a>&#65;&#66;</a>' ) != 'AB':
		return 6
	if text_of( '<a>&#x41;&#x42;</a>' ) != 'AB':
		return 7
	if text_of( '<a>&#x1F600;</a>' ) != chr( 0x1F600 ):
		return 8
	n2: xml.XMLNode = xml.parse( '<a b="x&amp;y"/>' ).unwrap( 'parse attr entity' )
	e2: xml.ElementData = n2.as_element().unwrap( 'as_element' )
	if e2.get_attr( 'b' ).unwrap( 'attr b' ) != 'x&y':
		return 9
	return 0
'''


# ---------------------------------------------------------------------------
# 3. Comments and CDATA as tree nodes.
# ---------------------------------------------------------------------------

_COMMENTS_CDATA = '''
import xml

def main() -> i32:
	n: xml.XMLNode = xml.parse( '<a><!--hi--><b/><![CDATA[<raw>]]></a>' ).unwrap( 'parse' )
	e: xml.ElementData = n.as_element().unwrap( 'as_element' )
	if len( e.children ) != 3:
		return 1
	c0: xml.XMLNode = e.children.__getitem__( 0 ).unwrap( 'idx0' )
	if c0.as_comment().unwrap( 'comment' ) != 'hi':
		return 2
	c1: xml.XMLNode = e.children.__getitem__( 1 ).unwrap( 'idx1' )
	e1: xml.ElementData = c1.as_element().unwrap( 'element' )
	if e1.tag != 'b':
		return 3
	c2: xml.XMLNode = e.children.__getitem__( 2 ).unwrap( 'idx2' )
	if c2.as_cdata().unwrap( 'cdata' ) != '<raw>':
		return 4

	match xml.parse( '<a><!-- bad -- comment --></a>' ):
		case Result.Ok( _ ):
			return 5
		case Result.Err( _ ):
			pass
	return 0
'''


# ---------------------------------------------------------------------------
# 4. Namespaces: default xmlns inheritance, prefixed elements/attributes,
# xmlns:prefix declarations, unbound-prefix error, attribute non-inheritance
# of the default namespace.
# ---------------------------------------------------------------------------

_NAMESPACES = '''
import xml

def main() -> i32:
	n: xml.XMLNode = xml.parse( '<root xmlns="urn:default" xmlns:x="urn:x"><a/><x:b/></root>' ).unwrap( 'parse' )
	root: xml.ElementData = n.as_element().unwrap( 'root' )
	if root.namespace_uri != 'urn:default':
		return 1
	kids: list[xml.ElementData] = n.child_elements()
	a: xml.ElementData = kids.__getitem__( 0 ).unwrap( 'a' )
	if a.namespace_uri != 'urn:default':
		return 2
	b: xml.ElementData = kids.__getitem__( 1 ).unwrap( 'b' )
	if b.namespace_uri != 'urn:x':
		return 3
	if b.local_name() != 'b':
		return 4
	if b.prefix() != 'x':
		return 5

	n2: xml.XMLNode = xml.parse( '<root xmlns:x="urn:x" x:attr="v"><child/></root>' ).unwrap( 'parse2' )
	root2: xml.ElementData = n2.as_element().unwrap( 'root2' )
	if root2.get_attr( 'attr', 'urn:x' ).unwrap( 'ns attr' ) != 'v':
		return 6
	kids2: list[xml.ElementData] = n2.child_elements()
	child2: xml.ElementData = kids2.__getitem__( 0 ).unwrap( 'child2' )
	if child2.namespace_uri != '':
		return 7

	n3: xml.XMLNode = xml.parse( '<root xmlns="urn:default" attr="v"/>' ).unwrap( 'parse3' )
	root3: xml.ElementData = n3.as_element().unwrap( 'root3' )
	if root3.get_attr( 'attr', '' ).unwrap( 'unprefixed attr has no ns' ) != 'v':
		return 8

	match xml.parse( '<x:a/>' ):
		case Result.Err( xml.XMLError.UnboundPrefix( _ )):
			pass
		case _:
			return 9
	return 0
'''


# ---------------------------------------------------------------------------
# 5. Malformed input -> Err, and the specific XMLError variant it maps to.
# ---------------------------------------------------------------------------

_MALFORMED_INPUT = '''
import xml

def expect_err( text: str ) -> bool:
	match xml.parse( text ):
		case Result.Ok( _ ):
			return False
		case Result.Err( _ ):
			return True

def main() -> i32:
	if not expect_err( '<a>' ):
		return 1
	if not expect_err( '<a></b>' ):
		return 2
	if not expect_err( '<a a="1" a="2"/>' ):
		return 3
	if not expect_err( '' ):
		return 4
	if not expect_err( '<a/><b/>' ):
		return 5
	if not expect_err( '<a/> trailing' ):
		return 6
	if not expect_err( '<a>&bogus;</a>' ):
		return 7
	if not expect_err( '<a b=value></a>' ):
		return 8
	if not expect_err( '<1a/>' ):
		return 9
	if not expect_err( '<x:a/>' ):
		return 10
	return 0
'''

_MALFORMED_ERROR_VARIANTS = '''
import xml

def main() -> i32:
	match xml.parse( '<a>' ):
		case Result.Err( xml.XMLError.UnexpectedEnd( _ )):
			pass
		case _:
			return 1
	match xml.parse( '<a></b>' ):
		case Result.Err( xml.XMLError.MismatchedEndTag( _ )):
			pass
		case _:
			return 2
	match xml.parse( '<a a="1" a="2"/>' ):
		case Result.Err( xml.XMLError.DuplicateAttribute( _ )):
			pass
		case _:
			return 3
	match xml.parse( '' ):
		case Result.Err( xml.XMLError.NoRootElement( _ )):
			pass
		case _:
			return 4
	match xml.parse( '<a/><b/>' ):
		case Result.Err( xml.XMLError.MultipleRootElements( _ )):
			pass
		case _:
			return 5
	match xml.parse( '<a/> trailing text' ):
		case Result.Err( xml.XMLError.TrailingGarbage( _ )):
			pass
		case _:
			return 6
	match xml.parse( '<a>&bogus;</a>' ):
		case Result.Err( xml.XMLError.InvalidEntity( _ )):
			pass
		case _:
			return 7
	match xml.parse( '<1a/>' ):
		case Result.Err( xml.XMLError.InvalidName( _ )):
			pass
		case _:
			return 8
	match xml.parse( '<a b=value></a>' ):
		case Result.Err( xml.XMLError.UnexpectedChar( _ )):
			pass
		case _:
			return 9
	match xml.parse( '<x:a/>' ):
		case Result.Err( xml.XMLError.UnboundPrefix( _ )):
			pass
		case _:
			return 10
	return 0
'''

# Remaining XMLError variants not reachable via xml.parse() failures above:
# WrongType/AttributeNotFound (accessor errors) and InvalidComment.
_REMAINING_ERROR_VARIANTS = '''
import xml

def main() -> i32:
	n: xml.XMLNode = xml.parse( '<a/>' ).unwrap( 'parse' )
	match n.as_text():
		case Result.Err( xml.XMLError.WrongType( _ )):
			pass
		case _:
			return 1
	e: xml.ElementData = n.as_element().unwrap( 'as_element' )
	match e.get_attr( 'missing' ):
		case Result.Err( xml.XMLError.AttributeNotFound( _ )):
			pass
		case _:
			return 2
	match xml.parse( '<a><!-- bad -- comment --></a>' ):
		case Result.Err( xml.XMLError.InvalidComment( _ )):
			pass
		case _:
			return 3
	return 0
'''


# ---------------------------------------------------------------------------
# 6. Serialization round-trip + escaping.
# ---------------------------------------------------------------------------

_SERIALIZATION = '''
import xml

def roundtrips( doc: str ) -> bool:
	n: xml.XMLNode = xml.parse( doc ).unwrap( 'parse failed for ' + doc )
	dumped: str = xml.dumps( n )
	back: xml.XMLNode = xml.parse( dumped ).unwrap( 'reparse failed for ' + dumped )
	return xml.dumps( back ) == dumped

def main() -> i32:
	if not roundtrips( '<a/>' ):
		return 1
	if not roundtrips( '<a x="1"><b>text</b><c/></a>' ):
		return 2
	if not roundtrips( '<a><!--c--><![CDATA[raw]]></a>' ):
		return 3

	n: xml.XMLNode = xml.parse( '<a x="v"/>' ).unwrap( 'parse' )
	e: xml.ElementData = n.as_element().unwrap( 'as_element' )
	if e.get_attr( 'x' ).unwrap( 'x' ) != 'v':
		return 4

	esc: xml.XMLNode = xml.parse( '<a x="&amp;&lt;&quot;"/>' ).unwrap( 'parse esc' )
	dumped2: str = xml.dumps( esc )
	if dumped2.find( '&amp;' ) == isize( -1 ):
		return 5
	if dumped2.find( '&lt;' ) == isize( -1 ):
		return 6
	if dumped2.find( '&quot;' ) == isize( -1 ):
		return 7

	text_esc: xml.XMLNode = xml.parse( '<a>1 &lt; 2 &amp;&amp; 3 &gt; 0</a>' ).unwrap( 'parse text esc' )
	if text_esc.text_content() != '1 < 2 && 3 > 0':
		return 8
	dumped3: str = xml.dumps( text_esc )
	if dumped3.find( '&lt;' ) == isize( -1 ):
		return 9
	if dumped3.find( '&gt;' ) == isize( -1 ):
		return 10
	return 0
'''


@unittest.skipUnless( test_support.HAS_CC, 'no C compiler (clang/gcc/msvc) found - skipping real-compile xml tests' )
class XmlSelfRefSpikeTests( RealCompileMixin, unittest.TestCase ):
	def setUp( self ) -> None:
		self.discovery = Discovery( import_builtins = True )
		self.compiler = Compiler( self.discovery )

	def test_self_referential_union_via_rc_class_compiles( self ) -> None:
		self.assert_programs_run([
			( 'self_ref_union_via_class', _SELF_REF_UNION_VIA_CLASS ),
		])


@unittest.skipUnless( test_support.HAS_CC, 'no C compiler (clang/gcc/msvc) found - skipping real-compile xml tests' )
class XmlBasicParsingTests( RealCompileMixin, unittest.TestCase ):
	def setUp( self ) -> None:
		self.discovery = Discovery( import_builtins = True )
		self.compiler = Compiler( self.discovery )

	def test_basic_parsing( self ) -> None:
		self.assert_programs_run([
			( 'basic_parsing', _BASIC_PARSING ),
		])


@unittest.skipUnless( test_support.HAS_CC, 'no C compiler (clang/gcc/msvc) found - skipping real-compile xml tests' )
class XmlEntityTests( RealCompileMixin, unittest.TestCase ):
	def setUp( self ) -> None:
		self.discovery = Discovery( import_builtins = True )
		self.compiler = Compiler( self.discovery )

	def test_entities( self ) -> None:
		self.assert_programs_run([
			( 'entities', _ENTITIES ),
		])


@unittest.skipUnless( test_support.HAS_CC, 'no C compiler (clang/gcc/msvc) found - skipping real-compile xml tests' )
class XmlCommentsCDataTests( RealCompileMixin, unittest.TestCase ):
	def setUp( self ) -> None:
		self.discovery = Discovery( import_builtins = True )
		self.compiler = Compiler( self.discovery )

	def test_comments_and_cdata( self ) -> None:
		self.assert_programs_run([
			( 'comments_cdata', _COMMENTS_CDATA ),
		])


@unittest.skipUnless( test_support.HAS_CC, 'no C compiler (clang/gcc/msvc) found - skipping real-compile xml tests' )
class XmlNamespaceTests( RealCompileMixin, unittest.TestCase ):
	def setUp( self ) -> None:
		self.discovery = Discovery( import_builtins = True )
		self.compiler = Compiler( self.discovery )

	def test_namespaces( self ) -> None:
		self.assert_programs_run([
			( 'namespaces', _NAMESPACES ),
		])


@unittest.skipUnless( test_support.HAS_CC, 'no C compiler (clang/gcc/msvc) found - skipping real-compile xml tests' )
class XmlMalformedInputTests( RealCompileMixin, unittest.TestCase ):
	def setUp( self ) -> None:
		self.discovery = Discovery( import_builtins = True )
		self.compiler = Compiler( self.discovery )

	def test_malformed_input( self ) -> None:
		self.assert_programs_run([
			( 'malformed_input', _MALFORMED_INPUT ),
			( 'malformed_error_variants', _MALFORMED_ERROR_VARIANTS ),
			( 'remaining_error_variants', _REMAINING_ERROR_VARIANTS ),
		])


@unittest.skipUnless( test_support.HAS_CC, 'no C compiler (clang/gcc/msvc) found - skipping real-compile xml tests' )
class XmlSerializationTests( RealCompileMixin, unittest.TestCase ):
	def setUp( self ) -> None:
		self.discovery = Discovery( import_builtins = True )
		self.compiler = Compiler( self.discovery )

	def test_serialization( self ) -> None:
		self.assert_programs_run([
			( 'serialization', _SERIALIZATION ),
		])


if __name__ == '__main__':
	unittest.main()
