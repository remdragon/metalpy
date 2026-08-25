# Real-compile-and-run behavioral tests for lib/json.py (JSON parsing,
# construction/manipulation, path access, flatten/unflatten). Every case
# compiles + links + runs a real executable via the shared
# test_support.RealCompileMixin (no copy-pasted harness) - correctness here
# depends on real RC/refcount and arithmetic behavior, not just "does it
# compile", so a compile-only check would miss the exact class of bug this
# module's own development hit (see union_rc_leaf_plain_class_bug.md - a
# @union RC-leaf-retrieval heap corruption bug, fixed on master before this
# file was written).
#
# Each program returns 0 on success and a distinct nonzero i32 exit code per
# failed check; RealCompileMixin decodes a merged-executable failure back to
# the failing case name and its own sub-code.

import unittest

import test_support
from test_support import RealCompileMixin
from discovery import Discovery
from compiler import Compiler


# ---------------------------------------------------------------------------
# 0. Required first test: does a @union with a list[Self]/dict[str,Self]
# variant actually compile? JSONValue's Array/Object variants need exactly
# this shape - see the module's own PLAN. Nothing else in this file should
# be trusted until this passes.
# ---------------------------------------------------------------------------

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
	kids.append( Node.Leaf( 1 ) )
	kids.append( Node.Leaf( 2 ) )
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


# ---------------------------------------------------------------------------
# 1. Round-trip parse -> serialize for each JSON type.
# ---------------------------------------------------------------------------

_ROUNDTRIP_SCALARS = '''
import json

def check_one( text: str, expected: str ) -> bool:
	v: json.JSONValue = json.loads( text ).unwrap( 'loads failed for ' + text )
	got: str = json.dumps( v ).unwrap( 'dumps failed for ' + text )
	return got == expected

def main() -> i32:
	if not check_one( 'null', 'null' ):
		return 1
	if not check_one( 'true', 'true' ):
		return 2
	if not check_one( 'false', 'false' ):
		return 3
	if not check_one( '0', '0' ):
		return 4
	if not check_one( '42', '42' ):
		return 5
	if not check_one( '-17', '-17' ):
		return 6
	if not check_one( '123456789012345678901234567890', '123456789012345678901234567890' ):
		return 7
	if not check_one( '""', '""' ):
		return 8
	if not check_one( '"hello world"', '"hello world"' ):
		return 9
	return 0
'''

_ROUNDTRIP_FLOATS = '''
import json

def main() -> i32:
	v1: json.JSONValue = json.loads( '3.5' ).unwrap( 'loads 3.5' )
	f1: f64 = v1.as_float().unwrap( 'as_float 3.5' )
	if f1 != 3.5:
		return 1
	d1: str = json.dumps( v1 ).unwrap( 'dumps 3.5' )
	if d1.find( '.' ) == -1:
		return 2

	v2: json.JSONValue = json.loads( '1.0e2' ).unwrap( 'loads 1.0e2' )
	f2: f64 = v2.as_float().unwrap( 'as_float 1.0e2' )
	if f2 != 100.0:
		return 3
	return 0
'''

_ROUNDTRIP_INT_VS_FLOAT_TEXT = '''
import json

def main() -> i32:
	# 1e10 is mathematically integral but must stay a Float and dump back
	# out WITHOUT bare-digit integer text.
	v: json.JSONValue = json.loads( '1e10' ).unwrap( 'loads 1e10' )
	match v:
		case json.JSONValue.Float( _ ):
			pass
		case json.JSONValue.Int( _ ):
			return 1
	dumped: str = json.dumps( v ).unwrap( 'dumps 1e10' )
	if dumped == '10000000000':
		return 2

	zero: json.JSONValue = json.loads( '0' ).unwrap( 'loads 0' )
	match zero:
		case json.JSONValue.Int( n ):
			if n.__str__() != '0':
				return 3
		case json.JSONValue.Float( _ ):
			return 4

	negzero: json.JSONValue = json.loads( '-0' ).unwrap( 'loads -0' )
	match negzero:
		case json.JSONValue.Int( n2 ):
			if n2.__str__() != '0':
				return 5
		case json.JSONValue.Float( _ ):
			return 6

	zerofloat: json.JSONValue = json.loads( '0.0' ).unwrap( 'loads 0.0' )
	match zerofloat:
		case json.JSONValue.Float( _ ):
			pass
		case json.JSONValue.Int( _ ):
			return 7

	e5: json.JSONValue = json.loads( '1E5' ).unwrap( 'loads 1E5' )
	match e5:
		case json.JSONValue.Float( _ ):
			pass
		case json.JSONValue.Int( _ ):
			return 8

	eplus5: json.JSONValue = json.loads( '1e+5' ).unwrap( 'loads 1e+5' )
	match eplus5:
		case json.JSONValue.Float( _ ):
			pass
		case json.JSONValue.Int( _ ):
			return 9

	eminus5: json.JSONValue = json.loads( '1e-5' ).unwrap( 'loads 1e-5' )
	match eminus5:
		case json.JSONValue.Float( _ ):
			pass
		case json.JSONValue.Int( _ ):
			return 10

	leadingzero: Result[json.JSONValue, json.JSONError] = json.loads( '01' )
	match leadingzero:
		case Result.Ok( _ ):
			return 11
		case Result.Err( _ ):
			pass
	return 0
'''

_ROUNDTRIP_CONTAINERS = '''
import json

def main() -> i32:
	empty_arr: json.JSONValue = json.loads( '[]' ).unwrap( 'loads []' )
	if empty_arr.array_len().unwrap( 'len' ) != 0:
		return 1
	if json.dumps( empty_arr ).unwrap( 'dumps []' ) != '[]':
		return 2

	empty_obj: json.JSONValue = json.loads( '{}' ).unwrap( 'loads {}' )
	if empty_obj.object_len().unwrap( 'len' ) != 0:
		return 3
	if json.dumps( empty_obj ).unwrap( 'dumps {}' ) != '{}':
		return 4

	nested: str = '{"a":[1,{"b":2},[3,4]],"c":null}'
	v: json.JSONValue = json.loads( nested ).unwrap( 'loads nested' )
	dumped: str = json.dumps( v ).unwrap( 'dumps nested' )
	if dumped != nested:
		return 5

	a: json.JSONValue = v.object_get( 'a' ).unwrap( 'get a' )
	first: json.JSONValue = a.array_get( 0 ).unwrap( 'get a[0]' )
	if first.as_int().unwrap( 'as_int' ).__str__() != '1':
		return 6
	second: json.JSONValue = a.array_get( 1 ).unwrap( 'get a[1]' )
	b: json.JSONValue = second.object_get( 'b' ).unwrap( 'get a[1].b' )
	if b.as_int().unwrap( 'as_int' ).__str__() != '2':
		return 7
	return 0
'''


# ---------------------------------------------------------------------------
# 2. String escape decoding.
# ---------------------------------------------------------------------------

_STRING_ESCAPES = '''
import json

def check_escape( text: str, expected: str ) -> bool:
	v: json.JSONValue = json.loads( text ).unwrap( 'loads failed for ' + text )
	got: str = v.as_str().unwrap( 'as_str' )
	return got == expected

def main() -> i32:
	if not check_escape( '"\\\\""', '"' ):
		return 1
	if not check_escape( '"\\\\\\\\"', '\\\\' ):
		return 2
	if not check_escape( '"\\\\/"', '/' ):
		return 3
	if not check_escape( '"\\\\b"', chr( 0x08 )):
		return 4
	if not check_escape( '"\\\\f"', chr( 0x0C )):
		return 5
	if not check_escape( '"\\\\n"', chr( 0x0A )):
		return 6
	if not check_escape( '"\\\\r"', chr( 0x0D )):
		return 7
	if not check_escape( '"\\\\t"', chr( 0x09 )):
		return 8
	if not check_escape( '"\\\\u0041"', 'A' ):
		return 9
	# surrogate pair for U+1F600 (grinning face emoji)
	if not check_escape( '"\\\\uD83D\\\\uDE00"', chr( 0x1F600 )):
		return 10
	return 0
'''

_STRING_ESCAPE_ROUNDTRIP = '''
import json

def main() -> i32:
	v: json.JSONValue = json.JSONValue.from_str( 'line1' + chr( 0x0A ) + 'line2' + chr( 0x09 ) + '"quoted"' + chr( 0x5C ) )
	dumped: str = json.dumps( v ).unwrap( 'dumps' )
	back: json.JSONValue = json.loads( dumped ).unwrap( 'loads back' )
	if back.as_str().unwrap( 'as_str' ) != v.as_str().unwrap( 'as_str v' ):
		return 1
	return 0
'''


# ---------------------------------------------------------------------------
# 3. Malformed input -> correct JSONError variant.
# ---------------------------------------------------------------------------

_MALFORMED_INPUT = '''
import json

def expect_err( text: str ) -> bool:
	match json.loads( text ):
		case Result.Ok( _ ):
			return False
		case Result.Err( _ ):
			return True

def main() -> i32:
	if not expect_err( '{' ):
		return 1
	if not expect_err( '[1,2' ):
		return 2
	if not expect_err( '"unterminated' ):
		return 3
	if not expect_err( '42 garbage' ):
		return 4
	if not expect_err( '"bad \\\\q escape"' ):
		return 5
	if not expect_err( '01' ):
		return 6
	if not expect_err( '-' ):
		return 7
	if not expect_err( '.5' ):
		return 8
	if not expect_err( '1.' ):
		return 9
	if not expect_err( '1e' ):
		return 10
	if not expect_err( '{"a":}' ):
		return 11
	if not expect_err( '[1,]' ):
		return 12
	if not expect_err( '' ):
		return 13
	if not expect_err( 'nul' ):
		return 14
	return 0
'''

_MALFORMED_ERROR_VARIANTS = '''
import json

def main() -> i32:
	match json.loads( '{' ):
		case Result.Err( json.JSONError.UnexpectedEnd( _ )):
			pass
		case _:
			return 1
	match json.loads( '42x' ):
		case Result.Err( json.JSONError.TrailingGarbage( _ )):
			pass
		case _:
			return 2
	match json.loads( '"bad \\\\q"' ):
		case Result.Err( json.JSONError.InvalidEscape( _ )):
			pass
		case _:
			return 3
	match json.loads( '01' ):
		case Result.Err( json.JSONError.InvalidNumber( _ )):
			pass
		case _:
			return 4
	match json.loads( '{1:2}' ):
		case Result.Err( json.JSONError.UnexpectedChar( _ )):
			pass
		case _:
			return 5
	return 0
'''


# ---------------------------------------------------------------------------
# 4. flatten() / unflatten() round-trip.
# ---------------------------------------------------------------------------

_FLATTEN_UNFLATTEN = '''
import json

def roundtrips( text: str ) -> bool:
	v: json.JSONValue = json.loads( text ).unwrap( 'loads' )
	flat: dict[str, json.JSONValue] = json.flatten( v )
	back: json.JSONValue = json.unflatten( flat ).unwrap( 'unflatten' )
	original: str = json.dumps( v ).unwrap( 'dumps v' )
	rebuilt: str = json.dumps( back ).unwrap( 'dumps back' )
	return original == rebuilt

def main() -> i32:
	if not roundtrips( '{"a":{"b":1},"c":[1,2,3]}' ):
		return 1
	if not roundtrips( '{"a":[{"x":1},{"y":2}]}' ):
		return 2
	if not roundtrips( '{"a":{},"b":[]}' ):
		return 3
	if not roundtrips( '[1,2,[3,4]]' ):
		return 4
	if not roundtrips( '42' ):
		return 5

	v: json.JSONValue = json.loads( '{"a":{"b":1}}' ).unwrap( 'loads' )
	flat: dict[str, json.JSONValue] = json.flatten( v )
	got: json.JSONValue = flat.__getitem__( 'a.b' ).unwrap( 'flat key a.b missing' )
	if got.as_int().unwrap( 'as_int' ).__str__() != '1':
		return 6

	v2: json.JSONValue = json.loads( '{"a":[10,20]}' ).unwrap( 'loads' )
	flat2: dict[str, json.JSONValue] = json.flatten( v2 )
	got2: json.JSONValue = flat2.__getitem__( 'a[0]' ).unwrap( 'flat key a[0] missing' )
	if got2.as_int().unwrap( 'as_int' ).__str__() != '10':
		return 7

	empty_flat: dict[str, json.JSONValue] = json.flatten( json.loads( '{"a":{}}' ).unwrap( 'loads' ))
	empty_v: json.JSONValue = empty_flat.__getitem__( 'a' ).unwrap( 'flat key a missing' )
	if empty_v.object_len().unwrap( 'len' ) != 0:
		return 8

	return 0
'''


# ---------------------------------------------------------------------------
# 5. get_path() / set_path().
# ---------------------------------------------------------------------------

_GET_PATH_HAPPY = '''
import json

def main() -> i32:
	v: json.JSONValue = json.loads( '{"a":{"b":[1,2,{"c":3}]}}' ).unwrap( 'loads' )

	r1: json.JSONValue = v.get_path( 'a.b[0]' ).unwrap( 'get_path a.b[0]' )
	if r1.as_int().unwrap( 'as_int' ).__str__() != '1':
		return 1

	r2: json.JSONValue = v.get_path( 'a.b[2].c' ).unwrap( 'get_path a.b[2].c' )
	if r2.as_int().unwrap( 'as_int' ).__str__() != '3':
		return 2

	arr: json.JSONValue = json.loads( '[10,20,30]' ).unwrap( 'loads' )
	r3: json.JSONValue = arr.get_path( '[1]' ).unwrap( 'get_path [1]' )
	if r3.as_int().unwrap( 'as_int' ).__str__() != '20':
		return 3

	return 0
'''

_GET_PATH_ERRORS = '''
import json

def main() -> i32:
	v: json.JSONValue = json.loads( '{"a":1,"b":[1,2]}' ).unwrap( 'loads' )

	match v.get_path( 'missing' ):
		case Result.Err( json.JSONError.KeyNotFound( _ )):
			pass
		case _:
			return 1

	match v.get_path( 'b[10]' ):
		case Result.Err( json.JSONError.IndexOutOfBounds( _ )):
			pass
		case _:
			return 2

	match v.get_path( 'a.x' ):
		case Result.Err( json.JSONError.WrongType( _ )):
			pass
		case _:
			return 3

	match v.get_path( 'a..b' ):
		case Result.Err( json.JSONError.InvalidPath( _ )):
			pass
		case _:
			return 4

	match v.get_path( 'a.' ):
		case Result.Err( json.JSONError.InvalidPath( _ )):
			pass
		case _:
			return 5

	match v.get_path( '[abc]' ):
		case Result.Err( json.JSONError.InvalidPath( _ )):
			pass
		case _:
			return 6

	match v.get_path( '[1' ):
		case Result.Err( json.JSONError.InvalidPath( _ )):
			pass
		case _:
			return 7

	match v.get_path( 'b[]' ):
		case Result.Err( json.JSONError.InvalidPath( _ )):
			pass
		case _:
			return 8

	return 0
'''

_SET_PATH = '''
import json

def main() -> i32:
	v: json.JSONValue = json.JSONValue.object()
	v.set_path( 'a.b', json.JSONValue.from_int( int.from_str( '5' ).unwrap( 'int' ))).unwrap( 'set_path a.b' )
	got: json.JSONValue = v.get_path( 'a.b' ).unwrap( 'get_path a.b' )
	if got.as_int().unwrap( 'as_int' ).__str__() != '5':
		return 1

	arr: json.JSONValue = json.JSONValue.array()
	arr.set_path( '[0]', json.JSONValue.from_int( int.from_str( '1' ).unwrap( 'int' ))).unwrap( 'set_path [0]' )
	arr.set_path( '[1]', json.JSONValue.from_int( int.from_str( '2' ).unwrap( 'int' ))).unwrap( 'set_path [1]' )
	if arr.array_len().unwrap( 'len' ) != 2:
		return 2

	# strict: setting past end-by-more-than-one is an error, no Null padding
	match arr.set_path( '[5]', json.JSONValue.from_int( int.from_str( '9' ).unwrap( 'int' ))):
		case Result.Err( json.JSONError.IndexOutOfBounds( _ )):
			pass
		case _:
			return 3

	# overwrite an existing index
	arr.set_path( '[0]', json.JSONValue.from_int( int.from_str( '100' ).unwrap( 'int' ))).unwrap( 'overwrite [0]' )
	got2: json.JSONValue = arr.get_path( '[0]' ).unwrap( 'get_path [0]' )
	if got2.as_int().unwrap( 'as_int' ).__str__() != '100':
		return 4

	return 0
'''


# ---------------------------------------------------------------------------
# 6. Every JSONError variant must be constructed by at least one reachable
# test case, or a @union used only as a Result error type crashes emit_c()
# outright (int_test.py's own confirmed compiler gotcha).
# ---------------------------------------------------------------------------

_ALL_JSONERROR_VARIANTS_CONSTRUCTED = '''
import json

def main() -> i32:
	e1: json.JSONError = json.JSONError.WrongType( None )
	e2: json.JSONError = json.JSONError.KeyNotFound( None )
	e3: json.JSONError = json.JSONError.IndexOutOfBounds( None )
	e4: json.JSONError = json.JSONError.Overflow( None )
	e5: json.JSONError = json.JSONError.NonFiniteFloat( None )
	e6: json.JSONError = json.JSONError.InvalidPath( 0 )
	e7: json.JSONError = json.JSONError.PathTypeMismatch( 0 )
	e8: json.JSONError = json.JSONError.UnexpectedEnd( 0 )
	e9: json.JSONError = json.JSONError.UnexpectedChar( 0 )
	e10: json.JSONError = json.JSONError.InvalidEscape( 0 )
	e11: json.JSONError = json.JSONError.InvalidNumber( 0 )
	e12: json.JSONError = json.JSONError.TrailingGarbage( 0 )
	if (
		e1.tag == e2.tag or e2.tag == e3.tag or e3.tag == e4.tag or e4.tag == e5.tag or
		e5.tag == e6.tag or e6.tag == e7.tag or e7.tag == e8.tag or e8.tag == e9.tag or
		e9.tag == e10.tag or e10.tag == e11.tag or e11.tag == e12.tag
	):
		return 1
	return 0
'''

_NON_FINITE_FLOAT_DUMPS_ERR = '''
import json

def main() -> i32:
	with compiler.wrap_arithmetic:
		zero: f64 = 0.0
		nan: f64 = zero / zero
	v: json.JSONValue = json.JSONValue.from_float( nan )
	match json.dumps( v ):
		case Result.Err( json.JSONError.NonFiniteFloat( _ )):
			pass
		case _:
			return 1
	return 0
'''

_PATH_TYPE_MISMATCH_VIA_UNFLATTEN = '''
import json

def main() -> i32:
	flat: dict[str, json.JSONValue] = dict[str, json.JSONValue]()
	flat.__setitem__( 'a', json.JSONValue.from_int( int.from_str( '1' ).unwrap( 'int' )))
	flat.__setitem__( 'a.b', json.JSONValue.from_int( int.from_str( '2' ).unwrap( 'int' )))
	match json.unflatten( flat ):
		case Result.Ok( _ ):
			return 1
		case Result.Err( _ ):
			pass
	return 0
'''


@unittest.skipUnless( test_support.HAS_CC, 'no C compiler (clang/gcc/msvc) found - skipping real-compile json tests' )
class JsonSelfRefSpikeTests( RealCompileMixin, unittest.TestCase ):
	def setUp( self ) -> None:
		self.discovery = Discovery( import_builtins = True )
		self.compiler = Compiler( self.discovery )

	def test_self_referential_list_and_dict_variants_compile( self ) -> None:
		self.assert_programs_run([
			( 'self_ref_list', _SELF_REF_LIST ),
			( 'self_ref_dict', _SELF_REF_DICT ),
		])


@unittest.skipUnless( test_support.HAS_CC, 'no C compiler (clang/gcc/msvc) found - skipping real-compile json tests' )
class JsonRoundtripTests( RealCompileMixin, unittest.TestCase ):
	def setUp( self ) -> None:
		self.discovery = Discovery( import_builtins = True )
		self.compiler = Compiler( self.discovery )

	def test_roundtrip_and_classification( self ) -> None:
		self.assert_programs_run([
			( 'roundtrip_scalars', _ROUNDTRIP_SCALARS ),
			( 'roundtrip_floats', _ROUNDTRIP_FLOATS ),
			( 'roundtrip_int_vs_float_text', _ROUNDTRIP_INT_VS_FLOAT_TEXT ),
			( 'roundtrip_containers', _ROUNDTRIP_CONTAINERS ),
		])


@unittest.skipUnless( test_support.HAS_CC, 'no C compiler (clang/gcc/msvc) found - skipping real-compile json tests' )
class JsonStringEscapeTests( RealCompileMixin, unittest.TestCase ):
	def setUp( self ) -> None:
		self.discovery = Discovery( import_builtins = True )
		self.compiler = Compiler( self.discovery )

	def test_string_escapes( self ) -> None:
		self.assert_programs_run([
			( 'string_escapes', _STRING_ESCAPES ),
			( 'string_escape_roundtrip', _STRING_ESCAPE_ROUNDTRIP ),
		])


@unittest.skipUnless( test_support.HAS_CC, 'no C compiler (clang/gcc/msvc) found - skipping real-compile json tests' )
class JsonMalformedInputTests( RealCompileMixin, unittest.TestCase ):
	def setUp( self ) -> None:
		self.discovery = Discovery( import_builtins = True )
		self.compiler = Compiler( self.discovery )

	def test_malformed_input( self ) -> None:
		self.assert_programs_run([
			( 'malformed_input', _MALFORMED_INPUT ),
			( 'malformed_error_variants', _MALFORMED_ERROR_VARIANTS ),
		])


@unittest.skipUnless( test_support.HAS_CC, 'no C compiler (clang/gcc/msvc) found - skipping real-compile json tests' )
class JsonFlattenTests( RealCompileMixin, unittest.TestCase ):
	def setUp( self ) -> None:
		self.discovery = Discovery( import_builtins = True )
		self.compiler = Compiler( self.discovery )

	def test_flatten_unflatten( self ) -> None:
		self.assert_programs_run([
			( 'flatten_unflatten', _FLATTEN_UNFLATTEN ),
		])


@unittest.skipUnless( test_support.HAS_CC, 'no C compiler (clang/gcc/msvc) found - skipping real-compile json tests' )
class JsonPathTests( RealCompileMixin, unittest.TestCase ):
	def setUp( self ) -> None:
		self.discovery = Discovery( import_builtins = True )
		self.compiler = Compiler( self.discovery )

	def test_get_set_path( self ) -> None:
		self.assert_programs_run([
			( 'get_path_happy', _GET_PATH_HAPPY ),
			( 'get_path_errors', _GET_PATH_ERRORS ),
			( 'set_path', _SET_PATH ),
		])


@unittest.skipUnless( test_support.HAS_CC, 'no C compiler (clang/gcc/msvc) found - skipping real-compile json tests' )
class JsonErrorVariantCoverageTests( RealCompileMixin, unittest.TestCase ):
	def setUp( self ) -> None:
		self.discovery = Discovery( import_builtins = True )
		self.compiler = Compiler( self.discovery )

	def test_all_error_variants_constructed( self ) -> None:
		self.assert_programs_run([
			( 'all_jsonerror_variants_constructed', _ALL_JSONERROR_VARIANTS_CONSTRUCTED ),
			( 'non_finite_float_dumps_err', _NON_FINITE_FLOAT_DUMPS_ERR ),
			( 'path_type_mismatch_via_unflatten', _PATH_TYPE_MISMATCH_VIA_UNFLATTEN ),
		])


if __name__ == '__main__':
	unittest.main()
