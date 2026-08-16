# stdlib imports:
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest

# local imports:
import ir
import emitter_c
import linker_c
import test_support
from compiler import Compiler
from discovery import Discovery
from errors import CompileError
from mpy_types import (
	CEnum, CStruct, Function, Parameter, RCClass, Scalar, Specialization, TaggedUnion, Variable,
)

def _scalar( stem: str, qualname: str|None = None ) -> Scalar:
	return Scalar( stem = stem, qualname = qualname or f'intrinsics.{stem}', file = None, line = None, sizeof = 0 )

class MangleQualnameTests( unittest.TestCase ):
	def test_dots_become_dollars( self ) -> None:
		self.assertEqual( emitter_c.mangle_qualname( 'builtins.str' ), 'builtins$str' )

	def test_generic_specialization_brackets( self ) -> None:
		# matches the plan's own worked example
		self.assertEqual(
			emitter_c.mangle_qualname( 'sys.alloc[intrinsics.u8]' ),
			'sys$alloc$$g$intrinsics$u8',
		)

	def test_multi_arg_specialization_commas( self ) -> None:
		self.assertEqual(
			emitter_c.mangle_qualname( 'builtins.Result[intrinsics.i32,builtins.OverflowError]' ),
			'builtins$Result$$g$intrinsics$i32$$builtins$OverflowError',
		)

class MangleTypeTests( unittest.TestCase ):
	def test_ordinary_class_falls_back_to_mangle_qualname( self ) -> None:
		cls = RCClass( stem = 'Foo', qualname = '__main__.Foo', file = Path( '__main__.py' ), line = 1 )
		self.assertEqual( emitter_c.mangle_type( cls ), '__main__$Foo' )

	def test_synthesized_anonymous_union_uses_u_scheme( self ) -> None:
		# mirrors discovery.py's _get_or_create_union: members stored
		# ascii-sorted by qualname (int before str), file/line both None
		int_member = Variable( stem = 'int', qualname = 'builtins.int|builtins.str.int', file = None, line = None, type = _scalar( 'int', 'builtins.int' ))
		str_member = Variable( stem = 'str', qualname = 'builtins.int|builtins.str.str', file = None, line = None, type = _scalar( 'str', 'builtins.str' ))
		union = TaggedUnion(
			stem = 'builtins.int|builtins.str', qualname = 'builtins.int|builtins.str',
			file = None, line = None, attributes = [ int_member, str_member ],
		)
		self.assertEqual( emitter_c.mangle_type( union ), '$__u$$builtins$int$$builtins$str' )

	def test_real_union_class_is_not_treated_as_synthesized( self ) -> None:
		# a genuine `@union class Foo:` always has file/line set - must NOT
		# hit the $__u... scheme, ordinary qualname mangling applies
		union = TaggedUnion( stem = 'Foo', qualname = '__main__.Foo', file = Path( '__main__.py' ), line = 1, attributes = [] )
		self.assertEqual( emitter_c.mangle_type( union ), '__main__$Foo' )

class CTypeTests( unittest.TestCase ):
	def test_scalar_widths( self ) -> None:
		self.assertEqual( emitter_c.c_type( _scalar( 'i32' )), 'int32_t' )
		self.assertEqual( emitter_c.c_type( _scalar( 'u64' )), 'uint64_t' )
		self.assertEqual( emitter_c.c_type( _scalar( 'i128' )), '__metalpy_wideint' )
		self.assertEqual( emitter_c.c_type( _scalar( 'u128' )), '__metalpy_wideuint' )
		self.assertEqual( emitter_c.c_type( _scalar( 'isize' )), 'intptr_t' )
		self.assertEqual( emitter_c.c_type( _scalar( 'usize' )), 'uintptr_t' )
		self.assertEqual( emitter_c.c_type( _scalar( 'bool', 'builtins.bool' )), 'bool' )

	def test_nonetype_is_metalpynone_noreturn_is_void( self ) -> None:
		# NoneType as a value (parameter/field) is MetalpyNone, not void —
		# void is only for return types and Ptr[None] pointees
		self.assertEqual( emitter_c.c_type( _scalar( 'NoneType' )), 'MetalpyNone' )
		self.assertEqual( emitter_c.c_type( _scalar( 'NoReturn' )), 'void' )
		self.assertEqual( emitter_c.c_type( None ), 'void' )

	def test_ptr_of_nonetype_is_void_star( self ) -> None:
		ptr_cls = Scalar( stem = 'Ptr', qualname = 'intrinsics.Ptr', file = None, line = None, sizeof = 8 )
		none_type = _scalar( 'NoneType' )
		ptr_none = Specialization( stem = 'Ptr[NoneType]', qualname = 'intrinsics.Ptr[intrinsics.NoneType]', file = None, line = None, base = ptr_cls, args = [ none_type ] )
		self.assertEqual( emitter_c.c_type( ptr_none ), 'void*' )

	def test_ptr_and_constptr_specializations( self ) -> None:
		ptr_cls = Scalar( stem = 'Ptr', qualname = 'intrinsics.Ptr', file = None, line = None, sizeof = 8 )
		const_ptr_cls = Scalar( stem = 'ConstPtr', qualname = 'intrinsics.ConstPtr', file = None, line = None, sizeof = 8 )
		u8 = _scalar( 'u8' )
		ptr_u8 = Specialization( stem = 'Ptr[u8]', qualname = 'intrinsics.Ptr[intrinsics.u8]', file = None, line = None, base = ptr_cls, args = [ u8 ] )
		const_ptr_u8 = Specialization( stem = 'ConstPtr[u8]', qualname = 'intrinsics.ConstPtr[intrinsics.u8]', file = None, line = None, base = const_ptr_cls, args = [ u8 ] )
		self.assertEqual( emitter_c.c_type( ptr_u8 ), 'uint8_t*' )
		self.assertEqual( emitter_c.c_type( const_ptr_u8 ), 'const uint8_t*' )

	def test_rcclass_is_a_pointer( self ) -> None:
		cls = RCClass( stem = 'Foo', qualname = '__main__.Foo', file = None, line = None )
		self.assertEqual( emitter_c.c_type( cls ), 'struct __main__$Foo*' )

	def test_cstruct_is_a_value( self ) -> None:
		cls = CStruct( stem = 'Foo', qualname = '__main__.Foo', file = None, line = None )
		self.assertEqual( emitter_c.c_type( cls ), 'struct __main__$Foo' )

# CEnum member-VALUE expressions (Color.Red used as a real runtime value) are
# a pre-existing lowering.py gap, not something this phase chases - see the
# plan's own grounding facts ("CEnum never being referenced by real
# lowering.py output today"). emit_cenum itself is still real, testable work
# (decision 1's per-unit philosophy) - built and verified directly against a
# hand-built CEnum object, same as MangleTypeTests/CTypeTests above, rather
# than through a full compiler.run() that has no way to produce a real value
# of the enum's type yet.
class EmitCEnumTests( unittest.TestCase ):
	def _color( self ) -> CEnum:
		cls = CEnum( stem = 'Color', qualname = '__main__.Color', file = None, line = None, value_type = _scalar( 'u32' ))
		cls.members = { 'Red': 0, 'Green': 1 }
		cls.values = { 0: 'Red', 1: 'Green' }
		return cls

	def test_typedef_line( self ) -> None:
		src = emitter_c.emit_cenum( self._color() )
		self.assertIn( 'typedef uint32_t __main__$Color;', src )

	def test_one_static_const_per_member( self ) -> None:
		src = emitter_c.emit_cenum( self._color() )
		self.assertIn( 'static const __main__$Color __main__$Color$Red = 0;', src )
		self.assertIn( 'static const __main__$Color __main__$Color$Green = 1;', src )

class CompilerTestCase( unittest.TestCase ):
	def setUp( self ) -> None:
		self.discovery = Discovery( import_builtins = False )
		self.compiler = Compiler( self.discovery )

	def _run( self, code: str ) -> None:
		self.compiler.import_code( code, Path( '__main__.py' ), scope = None )
		self.compiler.run()

class EmitFunctionTests( CompilerTestCase ):
	def test_empty_function_prototype_and_body( self ) -> None:
		self._run( '''
def main() -> None:
	return
''' )
		self.assertEqual( self.discovery.errors.errors, [] )
		lf = self.compiler.functions[0]
		# 'main' is reserved by discovery.py for the entry point - bare,
		# never module-qualified (discovery.py:994-995) - and compiles to
		# C's own real `int main(void)`, not `void`
		self.assertEqual( lf.function.qualname, 'main' )
		src = emitter_c.emit_function( lf )
		self.assertIn( 'int main( void ) {', src )
		self.assertIn( 'return 0;', src )
		self.assertTrue( src.rstrip().endswith( '}' ))

	def test_prototype_only_has_no_body( self ) -> None:
		self._run( '''
def main() -> None:
	return
''' )
		lf = self.compiler.functions[0]
		src = emitter_c.emit_function( lf, prototype_only = True )
		self.assertEqual( src, 'int main( void );' )

	def test_non_entry_function_keeps_its_declared_return_type( self ) -> None:
		self._run( '''
def main() -> None:
	helper()

def helper() -> None:
	return
''' )
		helper = next( lf for lf in self.compiler.functions if lf.function.qualname == '__main__.helper' )
		src = emitter_c.emit_function( helper )
		self.assertIn( 'void __main__$helper( void ) {', src )
		self.assertIn( '\treturn;', src )

class EmitCTests( CompilerTestCase ):
	def test_prologue_and_empty_main_present( self ) -> None:
		self._run( '''
def main() -> None:
	return
''' )
		src = emitter_c.emit_c( self.compiler )
		self.assertIn( 'ObjectHeader', src )
		self.assertIn( 'int main( void );', src ) # forward-declared
		self.assertIn( 'int main( void ) {', src ) # then defined

# shared by every test needing Result[T,E] - matches lowering_test.py's own
# _RESULT_FIXTURE (self-contained snippet, not a real lib/ import -
# established convention for CompilerTestCase-style tests, see
# compiler_test.py), with one deliberate difference: OverflowError is a
# @cstruct here, not a plain (RCClass) class like lowering_test.py's own
# fixture - RCClass struct/header emission is Phase 3 work, not built yet,
# and OverflowError's own kind is incidental to what THESE tests are
# actually verifying (Check-mode arithmetic/Result specialization
# synthesis/OrReturn), so it's kept within what Phase 1 actually covers.
# Result itself is a real @union (TaggedUnion), matching lib/builtins's own
# current definition - Ok/Err construction and is_ok/is_err go through the
# same generic union machinery any other @union does.
_RESULT_FIXTURE = '\n'.join([
	'@cstruct',
	'class OverflowError: pass',
	'',
	'@union',
	'class Result[T,E]:',
	'\tOk: T',
	'\tErr: E',
	'',
	'\tdef is_ok( self ) -> bool:',
	'\t\treturn self.tag == 0',
	'',
	'\tdef is_err( self ) -> bool:',
	'\t\treturn self.tag == 1',
])

class SpecializationSynthesisTests( CompilerTestCase ):
	def test_result_specialization_is_a_real_compiler_tagged_unions_entry( self ) -> None:
		# a concrete generic class specialization (Result[i32,
		# OverflowError]) is a real compile unit by the time it reaches
		# this module - lowering.py's Lowering.monomorphize_class (wired
		# through compiler.py's own _enqueue/_lower, NOT emitter_c.py -
		# stage 3 does no discovery of its own) already substituted its
		# .attributes and gave it a concrete qualname, landing it directly
		# in compiler.tagged_unions alongside the (still-abstract, correctly
		# excluded from emission) bare Result.
		self._run( _RESULT_FIXTURE + '\n' + '\n'.join([
			'def main() -> Result[i32,OverflowError]:',
			'\twith compiler.wrap_arithmetic:',
			'\t\tx: i32 = 1',
			'\treturn Result.Ok( x )',
		]))
		self.assertEqual( self.discovery.errors.errors, [] )
		names = [ cls.qualname for cls in self.compiler.tagged_unions ]
		self.assertIn( '__main__.Result[intrinsics.i32,__main__.OverflowError]', names )
		spec_cls = next( cls for cls in self.compiler.tagged_unions if cls.qualname == '__main__.Result[intrinsics.i32,__main__.OverflowError]' )
		self.assertIsNone( spec_cls.type_params ) # concrete now, not generic
		src = emitter_c.emit_tagged_union( spec_cls )
		self.assertIn( 'struct', src )
		self.assertIn( 'tag;', src )
		self.assertIn( 'data;', src )

class GenericMethodDispatchTests( CompilerTestCase ):
	''' Stage 2 of plans/fluttering-growing-chipmunk.md: a method call
	through a receiver whose type already pins a concrete generic
	Specialization must resolve to an already-substituted Function - not
	the abstract one, and not by _lower_class_generic_method_call detecting
	and re-substituting it per call site (that whole branch was proven
	unreachable and removed). '''

	def test_is_ok_on_concrete_result_receiver_is_monomorphized_once( self ) -> None:
		self._run( _RESULT_FIXTURE + '\n' + '\n'.join([
			'def get() -> Result[i32,OverflowError]:',
			'\treturn Result.Ok( 1 )',
			'',
			'def main() -> None:',
			'\tr: Result[i32,OverflowError] = get()',
			'\tif r.is_ok():',
			'\t\tpass',
			'\treturn',
		]))
		self.assertEqual( self.discovery.errors.errors, [] )
		is_ok_fns = [ lf for lf in self.compiler.functions if lf.function.qualname.startswith( '__main__.Result.is_ok' ) ]
		self.assertEqual( len( is_ok_fns ), 1 )
		main_lf = next( lf for lf in self.compiler.functions if lf.function.qualname == 'main' )
		calls = [ i for i in main_lf.instructions if isinstance( i, ir.Call ) and i.target.stem == 'is_ok' ]
		self.assertEqual( len( calls ), 1 )
		self.assertEqual( calls[0].receiver.type.qualname, '__main__.Result[intrinsics.i32,__main__.OverflowError]' )

	def test_or_return_on_concrete_result_receiver_still_lowers_textually( self ) -> None:
		# or_return() must never become a real compiled function or a real
		# Call to one - it has no declared body at all (a user-written
		# `def or_return(...)` is a discovery-time compile error, see
		# discovery.py's _parse_function) and is recognized purely by AST
		# shape in Lowering._lower_call, before ordinary call resolution
		# ever runs (see that check's own comment) - this is the exact
		# regression the eager-substitution work risked: target.cls became
		# a Specialization for a concrete receiver, breaking the old
		# `target.cls is Result` identity check that used to route here
		self._run( _RESULT_FIXTURE + '\n' + '\n'.join([
			'def get() -> Result[i32,OverflowError]:',
			'\treturn Result.Ok( 1 )',
			'',
			'def main() -> Result[i32,OverflowError]:',
			'\tr: Result[i32,OverflowError] = get()',
			'\tv: i32 = r.or_return()',
			'\treturn Result.Ok( v )',
		]))
		self.assertEqual( self.discovery.errors.errors, [] )
		self.assertFalse( any( 'or_return' in lf.function.qualname for lf in self.compiler.functions ))
		main_lf = next( lf for lf in self.compiler.functions if lf.function.qualname == 'main' )
		self.assertTrue( any( isinstance( i, ir.OrReturn ) for i in main_lf.instructions ))
		self.assertFalse( any( isinstance( i, ir.Call ) and i.target.stem == 'or_return' for i in main_lf.instructions ))

	def test_generic_rcclass_method_call_through_concrete_receiver_is_substituted( self ) -> None:
		self._run( '\n'.join([
			'class Box[T]:',
			'\tv: T',
			'\tdef get( self ) -> T:',
			'\t\treturn self.v',
			'',
			'def main() -> None:',
			'\tb: Box[i32]',
			'\tx = b.get()',
			'\treturn',
		]))
		self.assertEqual( self.discovery.errors.errors, [] )
		get_fns = [ lf for lf in self.compiler.functions if lf.function.qualname.startswith( '__main__.Box.get' ) ]
		self.assertEqual( len( get_fns ), 1 )
		self.assertEqual( get_fns[0].function.return_type.qualname, 'intrinsics.i32' ) # substituted, not bare T
		main_lf = next( lf for lf in self.compiler.functions if lf.function.qualname == 'main' )
		calls = [ i for i in main_lf.instructions if isinstance( i, ir.Call ) and i.target.stem == 'get' ]
		self.assertEqual( len( calls ), 1 )

	def test_union_receiver_dispatch_rejects_incompatible_leaf_parameter_types( self ) -> None:
		# a union mixing two DIFFERENT concrete instantiations of the same
		# generic class (Box[i32]|Box[u32]) with a same-named method taking
		# a generic-typed argument - the per-leaf consistency check
		# (type_resolver.py's _resolve_union_receiver_members) only compares
		# return-type identity and parameter COUNT, never per-position
		# parameter TYPE, so lowering itself has to catch a genuinely
		# non-coercible leaf. _lower_union_receiver_call now runs
		# _coerce_or_check_operand once per leaf (not just once overall,
		# against the first leaf) - the Box[u32] leaf's own mismatch is
		# caught and located, naming both types, the leaf, and the
		# parameter. No re-lowering of the argument expression happens
		# (side-effect safety): the SAME originally-lowered operand is still
		# reused, unchanged, across leaves whenever no coercion applies -
		# only now it's also validated per leaf.
		self._run( '\n'.join([
			'class Box[T]:',
			'\tv: T',
			'\tdef set( self, x: T ) -> None:',
			'\t\tself.v = x',
			'',
			'def main() -> None:',
			'\tb: Box[i32]|Box[u32]',
			'\tx: i32 = 5',
			'\tb.set( x )',
			'\treturn',
		]))
		errors = self.discovery.errors.errors
		self.assertEqual( len( errors ), 1 )
		error = errors[0]
		self.assertIn( '__main__.Box.set[intrinsics.u32]', error )
		self.assertIn( "parameter 'x'", error )
		self.assertIn( 'expected intrinsics.u32', error )
		self.assertIn( 'got intrinsics.i32', error )
		main_lf = next( lf for lf in self.compiler.functions if lf.function.qualname == 'main' )
		calls = [ i for i in main_lf.instructions if isinstance( i, ir.Call ) and i.target.stem == 'set' ]
		# only the i32 leaf's own Call (checked first, and legal) got
		# emitted - the u32 leaf's own failing _check_assignable raises,
		# aborting the rest of this statement via the ordinary per-
		# statement recovery boundary, same as any other lowering error
		self.assertEqual( len( calls ), 1 )

	def test_union_receiver_dispatch_applies_per_leaf_scalar_widening( self ) -> None:
		# the real fix, on the happy path: Box[i32]|Box[i64], x: i32 - the
		# i32 leaf keeps the original operand unchanged (exact type match,
		# _coerce_or_check_operand's own same-type fast path), the i64 leaf
		# gets its OWN distinct operand, fed by a real ir.CastWrap widening
		# that SAME original x - never re-lowering/re-evaluating the
		# argument expression itself
		self._run( '\n'.join([
			'class Box[T]:',
			'\tv: T',
			'\tdef set( self, x: T ) -> None:',
			'\t\tself.v = x',
			'',
			'def main() -> None:',
			'\tb: Box[i32]|Box[i64]',
			'\tx: i32 = 5',
			'\tb.set( x )',
			'\treturn',
		]))
		self.assertEqual( self.discovery.errors.errors, [] )
		main_lf = next( lf for lf in self.compiler.functions if lf.function.qualname == 'main' )
		calls = [ i for i in main_lf.instructions if isinstance( i, ir.Call ) and i.target.stem == 'set' ]
		self.assertEqual( len( calls ), 2 )
		i32_call = next( c for c in calls if c.target.qualname.endswith( '[intrinsics.i32]' ) )
		i64_call = next( c for c in calls if c.target.qualname.endswith( '[intrinsics.i64]' ) )
		self.assertEqual( i32_call.args[0].type.qualname, 'intrinsics.i32' )
		self.assertIsNot( i64_call.args[0], i32_call.args[0] )
		casts = [ i for i in main_lf.instructions if isinstance( i, ir.CastWrap ) and i.dest is i64_call.args[0] ]
		self.assertEqual( len( casts ), 1 )
		self.assertIs( casts[0].operand, i32_call.args[0] )

class EmitArithmeticTests( CompilerTestCase ):
	def test_wrap_arithmetic_smoke_test( self ) -> None:
		self._run( '''
def main() -> i32:
	with compiler.wrap_arithmetic:
		x: i32 = 1
		return x + 1
''' )
		self.assertEqual( self.discovery.errors.errors, [] )
		main_lf = next( lf for lf in self.compiler.functions if lf.function.qualname == 'main' )
		src = emitter_c.emit_function( main_lf )
		self.assertNotIn( '__builtin', src ) # wrap mode must NOT use the overflow builtins

	def test_default_check_mode_uses_result_and_or_return( self ) -> None:
		self._run( _RESULT_FIXTURE + '\n' + '\n'.join([
			'def main() -> Result[i32,OverflowError]:',
			'	x: i32 = 1',
			'	y: i32 = x + 1',
			'	return Result.Ok( y )',
		]))
		self.assertEqual( self.discovery.errors.errors, [] )
		main_lf = next( lf for lf in self.compiler.functions if lf.function.qualname == 'main' )
		kinds = [ type( i ).__name__ for i in main_lf.instructions ]
		self.assertIn( 'AddCheck', kinds )
		self.assertIn( 'OrReturn', kinds )
		src = emitter_c.emit_function( main_lf )
		self.assertIn( '__metalpy_add_overflow', src )
		self.assertIn( 'tag == 1', src )

_POINT_FIXTURE = '\n'.join([
	'@cstruct',
	'class Point:',
	'	x: i32',
	'	y: i32',
	'',
	'	@staticmethod',
	'	def make( x: i32, y: i32 ) -> Point:',
	'		return Point.__allocate__( x = x, y = y )',
])

class EmitCStructConstructTests( CompilerTestCase ):
	def test_construct_and_read_back( self ) -> None:
		self._run( _POINT_FIXTURE + '\n' + '\n'.join([
			'def main() -> i32:',
			'	p: Point = Point.make( 1, 2 )',
			'	with compiler.wrap_arithmetic:', # sidesteps needing a Result[i32,OverflowError] fixture - default Check mode isn't what this test is about
			'		return p.x + p.y',
		]))
		self.assertEqual( self.discovery.errors.errors, [] )
		main_lf = next( lf for lf in self.compiler.functions if lf.function.qualname == 'main' )
		src = emitter_c.emit_function( main_lf )
		self.assertIn( '.x', src )
		self.assertIn( '.y', src )
		point_cls = next( cls for cls in self.compiler.cstructs if cls.qualname == '__main__.Point' )
		struct_src = emitter_c.emit_cstruct( point_cls )
		self.assertIn( 'struct __main__$Point {', struct_src )
		self.assertIn( 'int32_t x;', struct_src )
		self.assertIn( 'int32_t y;', struct_src )

class EmitPointerOpsTests( CompilerTestCase ):
	def test_addrof_getitem_setitem( self ) -> None:
		# SetItem's RHS is lowered with expected_type=None (see lowering.py's
		# _stmt_Assign Subscript-target branch) - a bare literal can't be
		# inferred there, same as lowering_test.py's own test_getitem_setitem,
		# so an already-typed local (`seven`) sidesteps that, matching the
		# established convention rather than working around it here
		self._run( '''
def main() -> None:
	x: u8 = 5
	i: usize = 0
	seven: u8 = 7
	p: Ptr[u8] = compiler.addrof( x )
	p[i] = seven
	y: u8 = p[i]
	return
''' )
		self.assertEqual( self.discovery.errors.errors, [] )
		main_lf = next( lf for lf in self.compiler.functions if lf.function.qualname == 'main' )
		src = emitter_c.emit_function( main_lf )
		self.assertIn( '= &x;', src ) # AddrOf
		self.assertIn( '[i] = seven;', src ) # SetItem
		self.assertIn( '(p)[i];', src ) # GetItem

_UNION_FIXTURE = '\n'.join([
	'@union',
	'class Foo:',
	'	Bar: i32',
	'	Baz: usize',
])

class EmitTaggedUnionTests( CompilerTestCase ):
	def test_construct_and_match_round_trip( self ) -> None:
		# mirrors lowering_test.py's own test_match_union_construction_and_
		# extraction_round_trip - construct with one member, match takes
		# that arm, extracts the value. No new control-flow ops needed here
		# (match/tag-check lowering already reduces to Phase 1's own
		# Cmp/Jump*/Label) - this test mainly proves emit_tagged_union()
		# itself (the outer tag+data struct) plus the synthesized payload
		# CUnion both actually emit and compile
		self._run( _UNION_FIXTURE + '\n' + '\n'.join([
			'def main() -> i32:',
			'	f: Foo = Foo.Bar( 5 )',
			'	match f:',
			'		case Foo.Bar( x ):',
			'			return x',
			'		case Foo.Baz( z ):',
			'			with compiler.wrap_arithmetic:',
			'				return z + 1',
			'	return 0',
		]))
		self.assertEqual( self.discovery.errors.errors, [] )
		foo_union = next( u for u in self.compiler.tagged_unions if u.qualname == '__main__.Foo' )
		union_src = emitter_c.emit_tagged_union( foo_union )
		self.assertIn( 'struct __main__$Foo {', union_src )
		self.assertIn( 'uint8_t tag;', union_src )
		self.assertIn( 'union __main__$Foo$data data;', union_src )
		payload_cls = next( c for c in self.compiler.cunions if c.qualname == '__main__.Foo$data' )
		payload_src = emitter_c.emit_cunion( payload_cls )
		self.assertIn( 'union __main__$Foo$data {', payload_src )
		self.assertIn( 'int32_t v_Bar;', payload_src )
		self.assertIn( 'uintptr_t v_Baz;', payload_src )
		# Foo.Bar(5) is now an ordinary call to a real, synthesized
		# @staticmethod constructor (see union_storage.py's
		# _build_member_constructor) - the tag/payload assignment lives in
		# THAT function's own emitted body, not inline in main
		ctor_lf = next( lf for lf in self.compiler.functions if lf.function.qualname == '__main__.Foo.Bar' )
		ctor_src = emitter_c.emit_function( ctor_lf )
		self.assertIn( '.tag = 0', ctor_src ) # Foo.Bar's ordinal
		self.assertIn( '.v_Bar = ', ctor_src )
		main_lf = next( lf for lf in self.compiler.functions if lf.function.qualname == 'main' )
		src = emitter_c.emit_function( main_lf )
		self.assertIn( ').tag;', src ) # the match's case Foo.Bar(...) tag read, compared against the ordinal separately
		self.assertIn( '== (0)', src )

_CC = linker_c.detect_cc()

class UnionAsUnconstructedResultErrorTypeTests( test_support.RealCompileMixin, CompilerTestCase ):
	''' regression coverage for a real, previously-crashing gap: a @union
	referenced ONLY as a Result[T,E]'s own error type, with no reachable
	code anywhere constructing one of its own variants, used to crash
	emit_c() outright - AssertionError, "_tagged_union_storage has not run
	yet - no real storage shape to emit" (union_storage.get(union) is
	normally triggered lazily, the first time some code path constructs or
	matches a variant; nothing here ever does either for MyErr). Fixed in
	type_resolver.py's schedule() - see its own new
	_schedule_uniontype_storage/_union_storage_scheduling. Needs real
	builtins (Result itself), unlike EmitTaggedUnionTests above. '''
	def setUp( self ) -> None:
		self.discovery = Discovery( import_builtins = True )
		self.compiler = Compiler( self.discovery )

	@unittest.skipUnless( test_support.HAS_CC, 'no C compiler (clang/gcc/msvc) found - skipping' )
	def test_programs_compile_and_run( self ) -> None:
		''' every real compile-and-run program in this class, merged into a
		single executable (one build for the whole class); a nonzero exit is
		decoded back to the failing sub-program and its own return code. '''
		self.assert_programs_run([
			( 'unconstructed_union_error_type_compiles_and_runs', '''
@union
class MyErr:
	A: None
	B: None

def helper() -> Result[i32, MyErr]:
	return Result.Ok( 5 )

def main() -> i32:
	v: i32 = helper().unwrap( 'x' )
	if v != 5:
		return 1
	return 0
''' ),
			# a real, confirmed gap distinct from the one above: a 3+-member
			# anonymous union used only as a PARAMETER type, where the program
			# only ever constructs SOME of its members (here: an i32 flows in,
			# str and bool never do) - used to fail C compilation outright,
			# "incomplete definition of type 'struct builtins$str'", because
			# the union's own generic tag-gated Incref/Decref cleanup code
			# (cfg.py's _tag_gated_refcount_instructions, emitted for ANY
			# function receiving the union, since its own static type always
			# admits every member regardless of what this particular program
			# happens to construct) references str's full struct layout
			# directly, but str's own class body was never scheduled - nothing
			# schedules a union member's own LEAF TYPE unless something
			# separately constructs/uses it. Root-caused to union_storage.py's
			# UnionStorage.get(): its own `for attr in union.attributes:
			# self._ensure_resolved(attr)` loop scheduled the member's
			# attribute VARIABLE (a class-attribute-shaped Variable, not a
			# global one) - which schedule()'s own guard silently ignores,
			# since it isn't a Function/ClassLike/global Variable - never the
			# member's own attr.type (the actual RCClass that needs emitting).
			# Fixed by also explicitly scheduling attr.type.
			( 'union_member_never_constructed_still_gets_full_struct', '''
def describe( x: i32|str|bool ) -> i32:
	return 1

def main() -> i32:
	with compiler.wrap_arithmetic:
		return describe( 5 ) - 1
''' ),
		] )

class CStructNestedByValueOnlyReachedViaSizeofTests( test_support.RealCompileMixin, CompilerTestCase ):
	''' regression coverage for task_421ed8be: a small @cstruct (Inner)
	nested BY VALUE inside another @cstruct (Outer), where Outer is only
	ever reached through compiler.sizeof(Outer)/Ptr[Outer] - never actually
	CONSTRUCTED (Outer(...)) anywhere reachable, and Inner is never
	independently constructed/sized/pointed-to either. Used to fail real C
	compilation outright - "field has incomplete type 'struct ...Inner'",
	"forward declaration of ..." - because Inner never got scheduled as a
	real compile unit at all: resolving Outer's own `nested: Inner` field
	(compiler.py's `for attr in unit.attributes: self.lowering.
	_ensure_resolved(attr)` loop) only resolves the ATTRIBUTE Variable
	itself, never attr.type - schedule()'s own guard silently ignores a
	class-attribute Variable (is_global=False), so Inner was never added to
	compiler.cstructs, and _emit_value_type_bodies had nothing to emit a
	definition for, even though Outer's own struct body still references
	it by name. The exact same root cause union_storage.py's UnionStorage.
	get() was already fixed for once (see UnionAsUnconstructedResultErrorTypeTests
	above, "union_member_never_constructed_still_gets_full_struct") - this
	is the general case, fixed in compiler.py's CStruct/CUnion/TaggedUnion
	branches and monomorphize.py's monomorphize_class (mpy_types.py's new
	by_value_dependency helper). Confirmed this exact minimal shape crashes
	on a clean checkout (reverting the fix reproduces the real clang error
	directly - a 3-field Outer, no ~70-field struct needed; the original
	report's large real-world struct just happened to be the shape that
	first surfaced it). '''
	def setUp( self ) -> None:
		self.discovery = Discovery( import_builtins = True )
		self.compiler = Compiler( self.discovery )

	@unittest.skipUnless( test_support.HAS_CC, 'no C compiler (clang/gcc/msvc) found - skipping' )
	def test_programs_compile_and_run( self ) -> None:
		self.assert_programs_run([
			( 'nested_by_value_cstruct_only_reached_via_sizeof_compiles', '''
@cstruct
class Inner:
	a: u16 = 0
	b: u16 = 0

@cstruct
class Outer:
	x: i32 = 0
	nested: Inner = Inner( a = 0, b = 0 )
	y: i32 = 0

def main() -> i32:
	struct_size: usize = compiler.sizeof( Outer )
	raw: Ptr[u8] = sys.alloc[u8]( struct_size )
	sys.memzero( raw, struct_size )
	sys.free( raw )
	return 0
''' ),
		] )

class RCClassSubclassingPhase1Tests( CompilerTestCase ):
	''' Phase 1 of the RCClass-subclassing plan (base-chain lookup +
	attribute-shadowing rejection, no constructor chaining/@virtual/
	@abstractmethod yet - see the plan's own phasing): RCClass gained
	chain_lookup/own_new_virtual_slots/vtbl_owner/virtual_slots
	(mpy_types.py, generalized from CStruct's own, since both classes share
	an identical .base/.methods/.names/.resolve shape), and lowering.py's
	_find_method/_attr_lookup + type_resolver.py's _attr_lookup_callable now
	walk it for RCClass too, not just CStruct. Needs real builtins, like
	UnionAsUnconstructedResultErrorTypeTests above (RCClass construction
	goes through the real sys.alloc[T] path). '''
	def setUp( self ) -> None:
		self.discovery = Discovery( import_builtins = True )
		self.compiler = Compiler( self.discovery )

	def _extern_ldflags( self ) -> str:
		flags: list[str] = []
		for lib in sorted( self.compiler.extern_libs ):
			if lib == 'c':
				continue
			flags.append( linker_c.resolve_lib_ldflag( _CC, lib, self.compiler.extern_libs[lib] ) )
		return ' '.join( flags )

	def _assert_compiles_and_runs( self, c_source: str, expected_exit: int = 0 ) -> None:
		with tempfile.TemporaryDirectory() as tmp:
			src_path = Path( tmp ) / 'generated.c'
			obj_path = Path( tmp ) / 'generated.o'
			exe_path = Path( tmp ) / 'test_exe'
			src_path.write_text( c_source, encoding = 'utf-8' )
			cc_result = _CC.compile( src_path, obj_path )
			self.assertEqual( cc_result.returncode, 0,
				f'{_CC.name} compile failed:\nstdout: {cc_result.stdout}\nstderr: {cc_result.stderr}{test_support.c_source_on_failure( c_source )}' )
			ldflags = self._extern_ldflags()
			link_result = _CC.link( exe_path, [ obj_path ], ldflags = ldflags )
			self.assertEqual( link_result.returncode, 0,
				f'{_CC.name} link failed:\nstdout: {link_result.stdout}\nstderr: {link_result.stderr}' )
			run_result = subprocess.run( [ str( exe_path ) ], capture_output = True )
			self.assertEqual( run_result.returncode, expected_exit,
				f'exited {run_result.returncode}, expected {expected_exit} (stderr: {run_result.stderr})' )

	@unittest.skipUnless( _CC is not None, 'no C compiler (clang/gcc/msvc) found - skipping' )
	def test_subclass_calls_inherited_method( self ) -> None:
		# a subclass instance can call a method it never redeclared, found
		# by walking to its base via the new RCClass.chain_lookup - real
		# compile+run, not just a discovery-level resolution check.
		# Deliberately doesn't touch any INHERITED field (Derived(y=10) only
		# needs its OWN field) - field=value construction sugar flattening
		# the base chain is a later phase's scope, not this one's.
		self._run( '''
class Base:
	def hello( self ) -> i32:
		return 42

class Derived( Base ):
	y: i32

def main() -> i32:
	d: Derived = Derived( y = 10 )
	if d.hello() != 42:
		return 1
	return 0
''' )
		self.assertEqual( self.discovery.errors.errors, [] )
		self._assert_compiles_and_runs( emitter_c.emit_c( self.compiler ))

	def test_subclass_shadowing_base_field_is_a_compile_error( self ) -> None:
		self._run( '''
class Base:
	x: i32

class Derived( Base ):
	x: i32

def main() -> i32:
	d: Derived = Derived( x = 5 )
	return 0
''' )
		self.assertTrue( any( 'shadows' in e for e in self.discovery.errors.errors ), self.discovery.errors.errors )

	def test_subclass_shadowing_base_method_is_a_compile_error( self ) -> None:
		self._run( '''
class Base:
	def get_x( self ) -> i32:
		return 1

class Derived( Base ):
	def get_x( self ) -> i32:
		return 2

def main() -> i32:
	d: Derived = Derived()
	return 0
''' )
		self.assertTrue( any( 'shadows' in e for e in self.discovery.errors.errors ), self.discovery.errors.errors )

class RCClassSubclassingPhase2Tests( test_support.RealCompileMixin, CompilerTestCase ):
	''' Phase 2 of the RCClass-subclassing plan: super().__init__(...)
	constructor chaining. A subclass's own __init__ must open with
	super().__init__(...) (or, when the base's own __init__ is fallible,
	super().__init__(...).or_return()) as literally its first statement -
	after it runs, cfg.py's new complete_base_construction marks every
	base-chain attribute initialized in one shot, so the subclass's own
	construction-safety checking only ever has to track its OWN new
	attributes. See lowering.py's _lower_super_init_if_required/
	cfg.py's complete_base_construction.

	Found and fixed along the way (not specific to subclassing): every
	fallible RCClass __init__ in the language - subclassed or not - had
	its Ok/Err branch inverted (_emit_fallible_construction emitted
	JumpIfFalse where it needed JumpIfTrue), discovered while prototyping
	this phase's own fallible super().__init__() chaining requirement,
	confirmed via a plain, non-subclassed repro on a clean checkout before
	this phase's own changes. Fixed; see test_fallible_root_construction_
	ok_path_actually_returns_ok below for the regression coverage. '''
	def setUp( self ) -> None:
		self.discovery = Discovery( import_builtins = True )
		self.compiler = Compiler( self.discovery )

	@unittest.skipUnless( test_support.HAS_CC, 'no C compiler (clang/gcc/msvc) found - skipping' )
	def test_programs_compile_and_run( self ) -> None:
		''' every real compile-and-run program in this class, merged into a
		single executable (one build for the whole class); a nonzero exit is
		decoded back to the failing sub-program and its own return code. '''
		self.assert_programs_run([
			( 'basic_constructor_chaining', '''
class Base:
	x: i32
	def __init__( self, x: i32 ) -> None:
		self.x = x
	def get_x( self ) -> i32:
		return self.x

class Derived( Base ):
	y: i32
	def __init__( self, x: i32, y: i32 ) -> None:
		super().__init__( x )
		self.y = y
	def get_y( self ) -> i32:
		return self.y

def main() -> i32:
	with compiler.wrap_arithmetic:
		d: Derived = Derived( x = 5, y = 10 )
		total: i32 = d.get_x() + d.get_y()
		if total != 15:
			return 1
		return 0
''' ),
			( 'three_level_chain', '''
class Root:
	a: i32
	def __init__( self, a: i32 ) -> None:
		self.a = a

class Mid( Root ):
	b: i32
	def __init__( self, a: i32, b: i32 ) -> None:
		super().__init__( a )
		self.b = b

class Leaf( Mid ):
	c: i32
	def __init__( self, a: i32, b: i32, c: i32 ) -> None:
		super().__init__( a, b )
		self.c = c
	def total( self ) -> i32:
		with compiler.wrap_arithmetic:
			return self.a + self.b + self.c

def main() -> i32:
	leaf: Leaf = Leaf( a = 1, b = 2, c = 3 )
	if leaf.total() != 6:
		return 1
	return 0
''' ),
			# regression test for the inverted-branch bug this phase's own
			# fallible-chaining prototyping found (see this class's own
			# docstring) - a plain, non-subclassed fallible __init__, no
			# subclassing involved at all
			( 'fallible_root_construction_ok_path_actually_returns_ok', '''
class MyError:
	pass

class Base:
	x: i32
	def __init__( self, x: i32 ) -> Result[None, MyError]:
		self.x = x
		return Result.Ok( None )

def main() -> i32:
	with compiler.wrap_arithmetic:
		r: Result[Base, MyError] = Base( x = 5 )
		if r.is_err():
			return 100
		b: Base = r.unwrap( 'construction failed' )
		if b.x != 5:
			return 1
		return 0
''' ),
			( 'fallible_base_chaining_with_or_return', '''
class MyError:
	pass

class Base:
	x: i32
	def __init__( self, x: i32 ) -> Result[None, MyError]:
		self.x = x
		return Result.Ok( None )

class Derived( Base ):
	y: i32
	def __init__( self, x: i32, y: i32 ) -> Result[None, MyError]:
		super().__init__( x ).or_return()
		self.y = y
		return Result.Ok( None )
	def total( self ) -> i32:
		with compiler.wrap_arithmetic:
			return self.x + self.y

def main() -> i32:
	r: Result[Derived, MyError] = Derived( x = 5, y = 10 )
	if r.is_err():
		return 100
	d: Derived = r.unwrap( 'construction failed' )
	if d.total() != 15:
		return 1
	return 0
''' ),
			# real RC-lifetime stress check under repetition, same rigor as
			# this codebase's own established convention (see e.g.
			# MatchArmSameNameNarrowingTests.test_rc_lifetime_repeated_calls_
			# no_leak) - Base's own str field is set via super().__init__(),
			# never touched directly by Derived's own body, so this exercises
			# complete_base_construction's own bookkeeping specifically: if it
			# mishandled ownership (a spurious extra incref, or none at all
			# where one was needed), repeated construction/teardown would
			# either leak or double-free under repetition even if a single
			# iteration looked fine. 'hello'.upper() (not a literal) forces a
			# real heap allocation - a literal binds to immortal static
			# storage and can't distinguish a leak/double-release from doing
			# nothing.
			( 'rc_lifetime_repeated_chained_construction_no_leak', '''
class Base:
	s: str
	def __init__( self, s: str ) -> None:
		self.s = s

class Derived( Base ):
	y: i32
	def __init__( self, s: str, y: i32 ) -> None:
		super().__init__( s )
		self.y = y
	def byte_len( self ) -> usize:
		return self.s.byte_len()

def main() -> i32:
	with compiler.wrap_arithmetic:
		i: i32 = 0
		while i < 1000:
			d: Derived = Derived( s = 'hello'.upper(), y = 10 )
			if d.byte_len() != 5:
				return 1
			i += 1
		return 0
''' ),
		] )

class FallibleInitConstructionRCLifetimeTests( test_support.RealCompileMixin, CompilerTestCase ):
	''' Regression coverage for a real, confirmed double-free in fallible
	`__init__()` construction (SYNTAX.md's "Fallible __init__() Construction"
	- __init__ declared -> Result[None,E] makes `Foo(...)` itself yield
	Result[Foo,E]), found by reading the emitted C directly. Two independent
	over-release bugs lived in lowering.py's _emit_fallible_construction:

	1. dest_var (the hidden local threading the synthesized Result.Ok/Err
	   wrapping through) is a persistent Variable, not an ir.Temp - so unlike
	   every other fresh Call/Allocate result in this file, cfg.assign()'s
	   own "ownership transfers into dest, untrack the momentary Temp" branch
	   never fired for it when the caller consumed the returned value (e.g.
	   `r = Foo(...)`). dest_var stayed independently OWNED in cfg's own
	   bookkeeping on top of whatever the caller's own binding tracked for
	   the SAME object, so BOTH got decref'd at their own scope exit - a
	   real double-free, confirmed via direct compile-and-run (access
	   violation) with two fallible constructions of the same class in one
	   function, or a single one whose Result is retained/queried
	   (.is_ok()/.is_err()) rather than immediately match-extracted (a bare
	   match-and-extract happened to mask it: unwrap()'s own move-out of the
	   payload neutralizes the OUTER binding's own entry first, leaving only
	   dest_var's spurious decref to actually fire - net correct by
	   coincidence, not because the underlying tracking was ever right).
	   Fixed by cfg.move()'ing dest_var (the same primitive move[T] call
	   arguments use) right before returning it, cancelling its own pending
	   epilogue decref once its value is handed off.
	2. The Err branch's self_var cleanup used a bare `ir.Decref(value=
	   self_var)` instead of the compiler.decref()-style
	   cfg.decref()+cfg.manually_decreffed() pair - so self_var's own
	   binding stayed OWNED in cfg's bookkeeping and got decref'd AGAIN at
	   the function's own scope exit on top of this explicit one, double-
	   freeing the failed instance on every Err return. Fixed by routing
	   through cfg.decref()+manually_decreffed(), mirroring
	   _lower_compiler_decref's own compiler.decref(x) handling exactly.

	3. The Err branch's self_var cleanup (even after fix #2 above routed it
	   through cfg.decref()+manually_decreffed() instead of a bare
	   ir.Decref) still released self via the class's ordinary, SHARED
	   vtable destructor - the exact same one used to destroy any fully-
	   valid instance, which unconditionally decrefs EVERY RC-typed
	   attribute. self is only PARTIALLY constructed on the Err path -
	   reading an attribute this __init__ never reached an assignment for
	   reads whatever raw, unrelated bytes sys.alloc's allocator happened to
	   return, not a valid reference. A real, confirmed
	   STATUS_HEAP_CORRUPTION (0xC0000374 on Windows), not just a logical
	   bug - see fallible_construction_unassigned_rc_field_on_err_path_no_
	   crash below. A second, subtler half of the same bug: even an
	   attribute that WAS assigned before the failing Err path could
	   silently stop being released by __init__'s OWN unwind the moment a
	   LATER, textually-subsequent success-path return in the same __init__
	   completed construction (Epilogue.cancelled is one mutable flag
	   shared by every jump into that entry's label, not a per-jump-site
	   snapshot - build_epilogue_ladder() bakes in the FINAL state, not the
	   state as of each earlier jump's own time) - masked as a leak, not a
	   crash, only because the generic-destructor fallback this fix removes
	   happened to independently release the same attribute again on its
	   own way out; see fallible_construction_assigned_rc_field_before_
	   later_err_no_leak below. Fixed in two parts: _stmt_Return's own
	   Err-path-of-a-fallible-__init__ case now always does an inline
	   (never shared-label) unwind, so it correctly releases exactly
	   whichever RC attributes were actually assigned along the path taken,
	   using its own precise, path-sensitive CFG state; and
	   _emit_fallible_construction's Err branch no longer releases self via
	   the generic destructor at all - it frees self's raw allocation
	   directly (sys.free), skipping both the field cascade (now entirely
	   __init__'s own responsibility) and any user __del__ (forbidden on
	   this path regardless, per SYNTAX.md).

	Every RC-lifetime case below loops hundreds of times with a real heap
	allocation per iteration (matching this codebase's own
	rc_lifetime_repeated_*_no_leak convention above) rather than checking
	just one iteration: a single double-free doesn't reliably corrupt the
	heap badly enough to crash immediately, but repetition makes both
	directions (double-free AND any leak from an over-corrected fix) show up
	reliably. The fallible_construction_as_direct_match_subject case is
	unrelated to RC lifetime - it's a compiler-crash (AssertionError)
	regression in type_resolver.py's visit_Match, deterministic on the first
	attempt, so it doesn't need the loop convention. '''
	def setUp( self ) -> None:
		self.discovery = Discovery( import_builtins = True )
		self.compiler = Compiler( self.discovery )

	@unittest.skipUnless( test_support.HAS_CC, 'no C compiler (clang/gcc/msvc) found - skipping' )
	def test_programs_compile_and_run( self ) -> None:
		self.assert_programs_run([
			# the original repro: TWO fallible constructions of the same
			# class in one function, each only queried via is_ok() (never
			# match-extracted) - this is exactly the shape that segfaulted
			( 'two_fallible_constructions_one_function_ok_path', '''
class MyError:
	pass

class Box:
	v: i32
	def __init__( self, v: i32 ) -> Result[None, MyError]:
		self.v = v
		return Result.Ok( None )

def main() -> i32:
	r1: Result[Box, MyError] = Box( 5 )
	ok1: bool = r1.is_ok()
	r2: Result[Box, MyError] = Box( 6 )
	ok2: bool = r2.is_ok()
	if not ok1:
		return 1
	if not ok2:
		return 2
	return 0
''' ),
			# a SINGLE fallible construction per iteration, retained and only
			# queried (never match-extracted or unwrap()'d) - dest_var's own
			# leftover decref, unmasked by any move-out, double-frees the
			# constructed object almost immediately under repetition
			( 'fallible_construction_retained_and_queried_no_extraction_no_double_free', '''
class MyError:
	pass

class Widget:
	v: i32
	def __init__( self, v: i32 ) -> Result[None, MyError]:
		self.v = v
		return Result.Ok( None )

def main() -> i32:
	with compiler.wrap_arithmetic:
		i: i32 = 0
		while i < 500:
			r: Result[Widget, MyError] = Widget( v = i )
			if r.is_err():
				return 1
			i += 1
		return 0
''' ),
			# the Err path specifically, repeated - targets self_var's own
			# double-decref bug (bare ir.Decref never cancelled cfg's own
			# pending epilogue release for it)
			( 'fallible_construction_err_path_repeated_no_double_free', '''
class MyError:
	pass

class Validated:
	v: i32
	def __init__( self, v: i32 ) -> Result[None, MyError]:
		if v < 0:
			return Result.Err( MyError() )
		self.v = v
		return Result.Ok( None )

def main() -> i32:
	with compiler.wrap_arithmetic:
		i: i32 = 0
		while i < 500:
			r: Result[Validated, MyError] = Validated( v = -1 )
			if not r.is_err():
				return 1
			i += 1
		return 0
''' ),
			# precise refcount accounting (not just "doesn't crash"): after a
			# single fallible construction is unwrap()'d, EXACTLY TWO owning
			# references should be live - self_var's own original reference
			# (dropped from cfg's automatic, branch-unaware scope-exit
			# tracking and released explicitly instead - see
			# _emit_fallible_construction's own Ok-branch comment) plus
			# unwrap()'s own retained copy (Result.unwrap's accessor takes an
			# independent Incref'd copy rather than moving the payload out -
			# see lib/builtins/__init__.py's Result class docstring). NOT 1:
			# an earlier version of this fix assumed 1 and asserted it here,
			# which was itself wrong - catches either an over-release
			# (crashes before this check even reads a valid header) or a
			# leak (an extra, un-decref'd reference from an over-corrected
			# fix) in the same assertion
			( 'fallible_construction_unwrap_leaves_exactly_two_references', '''
class MyError:
	pass

class Gadget:
	v: i32
	def __init__( self, v: i32 ) -> Result[None, MyError]:
		self.v = v
		return Result.Ok( None )

def main() -> i32:
	with compiler.wrap_arithmetic:
		r: Result[Gadget, MyError] = Gadget( 7 )
		if r.is_err():
			return 1
		g: Gadget = r.unwrap( 'construction failed' )
		if g.v != 7:
			return 2
		rc: usize = compiler.refcount( g )
		if rc != 2:
			return compiler.cast( i32, 3 + rc )
		return 0
''' ),
			# a fallible construction that fails on its VERY FIRST attempt at
			# a given call site (no loop needed) - dest_var gets assigned in
			# BOTH the Ok and Err branches of _emit_fallible_construction,
			# but cfg sees them as one flat sequence (no enter_branch()/
			# restore() around the Ok/Err split), so it treated the Err
			# branch's assignment as a REASSIGNMENT of an already-live
			# dest_var, emitting a decref of dest_var's "previous value"
			# before overwriting it - except the Ok branch never actually
			# ran on this path, so dest_var's storage was uninitialized
			# garbage. A real, confirmed crash (illegal instruction),
			# distinct from the double-free bugs above and not caught by any
			# existing test since none of them construct a class whose
			# __init__ can actually fail
			( 'fallible_construction_fails_on_first_attempt_no_crash', '''
class MyError:
	pass

class Validated:
	v: i32
	def __init__( self, v: i32 ) -> Result[None, MyError]:
		if v < 0:
			return Result.Err( MyError() )
		self.v = v
		return Result.Ok( None )

def main() -> i32:
	r: Result[Validated, MyError] = Validated( -1 )
	if not r.is_err():
		return 1
	return 0
''' ),
			# a wholly separate bug from the RC/double-free ones above: a
			# fallible construction used DIRECTLY as a match statement's own
			# subject (`match Box(5):`), never bound to a variable first. type_
			# resolver.py's visit_Match built its synthesized `__match_subj_N =
			# <subject>` assignment via generic_visit_expr(node.subject), which
			# only visits the subject expression's own CHILDREN (ast.
			# NodeTransformer.generic_visit's own semantics when called
			# directly on an expr, rather than on its parent statement) - so a
			# Call subject's own visit_Call, which is what actually resolves
			# __init__ (sets resolved_construction / clears Function.resolve),
			# never ran. lowering.py's _try_lower_construct_call then hit its
			# own `assert init.resolve is None` sentinel meant to catch exactly
			# that unresolved state - a compiler crash (AssertionError), not a
			# runtime one. Fixed by dispatching through self.visit(node.
			# subject) instead, matching how every other statement
			# (visit_Assign, visit_AnnAssign, ...) already visits ITS OWN child
			# expressions via generic_visit(self) called on the PARENT node
			( 'fallible_construction_as_direct_match_subject', '''
class MyError:
	pass

class Box:
	v: i32
	def __init__( self, v: i32 ) -> Result[None, MyError]:
		self.v = v
		return Result.Ok( None )

def main() -> i32:
	with compiler.wrap_arithmetic:
		match Box( 5 ):
			case Result.Ok( b ):
				return b.v - 5
			case Result.Err( e ):
				return 99
''' ),
			# a THIRD, distinct bug from the two RC/double-free ones above and
			# the match-subject compiler crash just above: a class with an RC-
			# typed field that ISN'T assigned along the path that returns Result.
			# Err(...) - self is only PARTIALLY constructed on this path, but
			# _emit_fallible_construction's own Err-branch cleanup used to release
			# self via the class's ordinary, shared vtable destructor
			# (release_object -> $$__destructor__), the SAME one used to destroy
			# any fully-valid instance - which unconditionally decrefs EVERY RC-
			# typed field, including `held` here, which was never written on this
			# path. Reading self->held then reads whatever raw bytes sys.alloc's
			# allocator happened to return (release builds never zero fresh
			# allocations at all; even the debug-only fill lib/sys.py's alloc[T]
			# applies is deliberately a nonzero poison byte, not zero - see its
			# own comment), so release_object() dereferences/decrements a
			# refcount through a garbage pointer - a real, confirmed
			# STATUS_HEAP_CORRUPTION (0xC0000374) on Windows, not just a logical
			# bug. Fixed in two parts: (1) __init__'s own Err-path return now
			# always does an inline (never shared-label) unwind, so it correctly
			# releases only whichever RC attributes IT actually assigned, using
			# its own precise, path-sensitive CFG state (_stmt_Return's
			# construction_err_path special case); (2) the call site's Err-branch
			# release of self no longer goes through the generic destructor at
			# all - it frees self's raw allocation directly (sys.free), skipping
			# both the field cascade and any user __del__ (forbidden here by
			# SYNTAX.md regardless)
			( 'fallible_construction_unassigned_rc_field_on_err_path_no_crash', '''
class MyError:
	pass

class Holder:
	v: i32
	def __init__( self, v: i32 ) -> None:
		self.v = v

class Box:
	tag: i32
	held: Holder
	def __init__( self, tag: i32, held: Holder ) -> Result[None, MyError]:
		if tag < 0:
			return Result.Err( MyError() )
		self.tag = tag
		self.held = held
		return Result.Ok( None )

def main() -> i32:
	with compiler.wrap_arithmetic:
		h: Holder = Holder( 7 )
		i: i32 = 0
		while i < 500:
			r: Result[Box, MyError] = Box( -1, h )
			if not r.is_err():
				return 1
			i += 1
		return 0
''' ),
			# the companion shape to the one just above: the RC-typed field IS
			# assigned before a LATER validation fails and returns Result.
			# Err(...) - exercises the OTHER half of the same fix.
			# complete_construction()'s own success-path cancellation (run by a
			# LATER, textually-subsequent Result.Ok(...) return in the same
			# __init__) mutates a mutable Epilogue.cancelled flag shared by every
			# jump into that entry's label - before this fix, an EARLIER Err-path
			# return that had already committed to a shared epilogue label
			# (while `held`'s entry was still live) silently lost its own decref
			# of `held` the moment that LATER success path ran
			# complete_construction(), since build_epilogue_ladder() bakes each
			# entry's FINAL cancelled state into every jump site that shares it,
			# not the state as of each jump's own time - a real leak, masked
			# only by the call site's own generic-destructor fallback (removed
			# by this same fix, for the never-assigned case above)
			# coincidentally releasing `held` again on its way out. Checks exact
			# refcount, not just "doesn't crash" - a leak wouldn't crash within
			# 500 iterations either
			( 'fallible_construction_assigned_rc_field_before_later_err_no_leak', '''
class MyError:
	pass

class Holder:
	v: i32
	def __init__( self, v: i32 ) -> None:
		self.v = v

class Box:
	tag: i32
	held: Holder
	def __init__( self, tag: i32, held: Holder, extra: i32 ) -> Result[None, MyError]:
		self.held = held
		if extra < 0:
			return Result.Err( MyError() )
		self.tag = tag
		return Result.Ok( None )

def main() -> i32:
	with compiler.wrap_arithmetic:
		h: Holder = Holder( 7 )
		before: usize = compiler.refcount( h )
		i: i32 = 0
		while i < 500:
			r: Result[Box, MyError] = Box( 3, h, -1 )
			if not r.is_err():
				return 1
			i += 1
		after: usize = compiler.refcount( h )
		if before != after:
			return compiler.cast( i32, 2 + after )
		return 0
''' ),
		] )

class RCClassSubclassingPhase4Tests( test_support.RealCompileMixin, CompilerTestCase ):
	''' Phase 4 of the RCClass-subclassing plan: @virtual for RCClass,
	unifying destructor dispatch with @virtual dispatch. ObjectHeader's
	own destructor field became a real (if often minimal) vtable pointer
	- see the PROLOGUE's own __metalpy_ObjectVtbl comment - reusing
	CStruct's already-shipped vtable machinery (mpy_types.py's
	virtual_slots/vtbl_owner, compiler.py's _validate_interface_vtable,
	emitter_c.py's vtable-instance emission) generalized to RCClass|
	CStruct, plus RCClass-specific mechanics CStruct never needed: the
	shared minimal type for the (still common) non-virtual case, a
	destroy-prefixed synthesized type for virtual-bearing classes, and an
	UNCONDITIONAL per-class vtable instance (never opt-in like CStruct's
	own, since every RCClass instance needs one for destructor dispatch
	regardless of whether it has any @virtual methods). '''
	def setUp( self ) -> None:
		self.discovery = Discovery( import_builtins = True )
		self.compiler = Compiler( self.discovery )

	@unittest.skipUnless( test_support.HAS_CC, 'no C compiler (clang/gcc/msvc) found - skipping' )
	def test_programs_compile_and_run( self ) -> None:
		''' every real compile-and-run program in this class, merged into a
		single executable (one build for the whole class); a nonzero exit is
		decoded back to the failing sub-program and its own return code. '''
		self.assert_programs_run([
			# real vtable dispatch (not a direct call) reaching the override -
			# the core proof this works for a heap-allocated, refcounted
			# RCClass object, not just CStruct's own COM-style pointer
			( 'virtual_dispatch_through_base_typed_reference', '''
class Base:
	@virtual
	def hello( self ) -> i32:
		return 1

class Derived( Base ):
	@virtual
	def hello( self ) -> i32:
		return 2

def call_hello( b: Base ) -> i32:
	return b.hello()

def main() -> i32:
	b: Base = Base()
	d: Derived = Derived()
	if call_hello( b ) != 1:
		return 1
	if call_hello( d ) != 2:
		return 2
	return 0
''' ),
			( 'three_level_hierarchy_dispatches_to_the_leaf_override', '''
class Root:
	@virtual
	def hello( self ) -> i32:
		return 1

class Mid( Root ):
	pass

class Leaf( Mid ):
	@virtual
	def hello( self ) -> i32:
		return 3

def call_it( r: Root ) -> i32:
	return r.hello()

def main() -> i32:
	leaf: Leaf = Leaf()
	if call_it( leaf ) != 3:
		return 1
	root: Root = Root()
	if call_it( root ) != 1:
		return 2
	return 0
''' ),
			# real RC-lifetime stress check under repetition, same rigor as
			# this codebase's own established convention - a virtual-bearing
			# RCClass's own $header.vtable wiring (cast to its own synthesized
			# Vtbl type, not the shared minimal one) must not perturb
			# construction/teardown correctness under repeated allocation
			( 'rc_lifetime_repeated_virtual_dispatch_no_leak', '''
class Base:
	s: str
	def __init__( self, s: str ) -> None:
		self.s = s
	@virtual
	def describe( self ) -> usize:
		return self.s.byte_len()

class Derived( Base ):
	@virtual
	def describe( self ) -> usize:
		with compiler.wrap_arithmetic:
			return self.s.byte_len() + 1

def main() -> i32:
	with compiler.wrap_arithmetic:
		i: i32 = 0
		while i < 1000:
			d: Derived = Derived( s = 'hello'.upper() )
			if d.describe() != 6:
				return 1
			i += 1
		return 0
''' ),
			# a @virtual method on a GENERIC RCClass. Monomorphization sets the
			# method's own .cls to a Specialization wrapping the class (see
			# monomorphize.py's substituted_cls), never to a bare RCClass - so
			# the emitter's old isinstance( instr.target.cls, RCClass ) dispatch
			# check answered False here and fell through to CStruct's COM form,
			# emitting `(b)->$vtable->get( b )`. An RCClass has no $vtable member
			# at all (its vtable pointer lives inside $header - see the PROLOGUE),
			# so that was a reference to a field that doesn't exist. Now asked as
			# has_object_header(), which a Specialization answers by delegating.
			( 'virtual_dispatch_on_a_generic_rcclass', '''
class Holder[T]:
	v: T
	def __init__( self, v: T ) -> None:
		self.v = v
	@virtual
	def tag( self ) -> i32:
		return 7

def main() -> i32:
	h: Holder[i32] = Holder[i32]( v = 5 )
	if h.tag() != 7: # goes through the vtable, not a direct call
		return 1
	s: Holder[str] = Holder[str]( v = 'x' )
	if s.tag() != 7:
		return 2
	return 0
''' ),
		] )

	@unittest.skipUnless( _CC is not None, 'no C compiler (clang/gcc/msvc) found - skipping' )
	def test_non_virtual_rcclass_gets_no_synthesized_vtbl_type( self ) -> None:
		# the common case (no @virtual methods anywhere in the chain)
		# costs nothing extra - $$vtable uses the shared, built-in
		# __metalpy_ObjectVtbl directly, no per-class Vtbl struct type at
		# all, confirmed by inspecting the emitted C directly (not just
		# that it compiles and runs)
		self._run( '''
class Plain:
	x: i32
	def __init__( self, x: i32 ) -> None:
		self.x = x
	def get_x( self ) -> i32:
		return self.x

def main() -> i32:
	p: Plain = Plain( x = 42 )
	if p.get_x() != 42:
		return 1
	return 0
''' )
		self.assertEqual( self.discovery.errors.errors, [] )
		src = emitter_c.emit_c( self.compiler )
		self.assertNotIn( 'typedef struct __main__$PlainVtbl', src )
		self.assertIn( 'static const __metalpy_ObjectVtbl __main__$Plain$$vtable', src )
		self._assert_compiles_and_runs( src )

	@unittest.skipUnless( _CC is not None, 'no C compiler (clang/gcc/msvc) found - skipping' )
	def test_override_signature_mismatch_is_a_compile_error( self ) -> None:
		self._run( '''
class Base:
	@virtual
	def hello( self ) -> i32:
		return 1

class Derived( Base ):
	@virtual
	def hello( self, extra: i32 ) -> i32:
		return extra

def main() -> i32:
	d: Derived = Derived()
	return 0
''' )
		self.assertTrue( any( 'does not match' in e for e in self.discovery.errors.errors ), self.discovery.errors.errors )

class RCClassSubclassingPhase5Tests( test_support.RealCompileMixin, CompilerTestCase ):
	''' Phase 5 of the RCClass-subclassing plan: @abstractmethod for
	RCClass, using the explicit is_abstract marker (not CStruct's
	implicit stub-body-means-unimplemented convention) - construction-
	time enforcement (lowering.py's _check_rcclass_fully_implemented,
	shared by both _lower_allocate_fields and _try_lower_construct_call)
	and vtable-instance emission (emitter_c.py's emit_rcclass_vtable_
	instance, None/skipped for an abstract class, matching CStruct's own
	"None if unfulfilled" convention). The 3+-level chain tests below are
	exactly the scenario CStruct's own root-only-implicit-stub convention
	can't express (an abstract method left unfulfilled through an
	intermediate level, only fulfilled at the leaf), which is the whole
	reason RCClass gets a real, explicit decorator instead of reusing
	CStruct's own convention. @abstractmethod alone implies @virtual (no
	need to repeat it on the abstract declaration itself - see
	discovery.py's _parse_function) - a concrete OVERRIDE still has to
	write @virtual itself though, same as any other virtual override. '''
	def setUp( self ) -> None:
		self.discovery = Discovery( import_builtins = True )
		self.compiler = Compiler( self.discovery )

	def test_construct_abstract_class_directly_is_a_compile_error( self ) -> None:
		self._run( '''
class Base:
	@abstractmethod
	def hello( self ) -> i32: ...

def main() -> i32:
	b: Base = Base()
	return 0
''' )
		self.assertTrue( any( 'cannot be constructed - abstract method(s)' in e for e in self.discovery.errors.errors ), self.discovery.errors.errors )

	def test_construct_intermediate_still_abstract_class_is_a_compile_error( self ) -> None:
		# the abstract method is declared at the ROOT, left unfulfilled at
		# an INTERMEDIATE level (Mid adds nothing), and only implemented
		# at the LEAF - constructing Mid itself must still be rejected
		self._run( '''
class Root:
	@abstractmethod
	def hello( self ) -> i32: ...

class Mid( Root ):
	pass

class Leaf( Mid ):
	@virtual
	def hello( self ) -> i32:
		return 99

def main() -> i32:
	m: Mid = Mid()
	return 0
''' )
		self.assertTrue( any( 'cannot be constructed - abstract method(s)' in e for e in self.discovery.errors.errors ), self.discovery.errors.errors )

	@unittest.skipUnless( test_support.HAS_CC, 'no C compiler (clang/gcc/msvc) found - skipping' )
	def test_programs_compile_and_run( self ) -> None:
		''' every real compile-and-run program in this class, merged into a
		single executable (one build for the whole class); a nonzero exit is
		decoded back to the failing sub-program and its own return code. '''
		self.assert_programs_run([
			( 'subclass_implementing_abstract_method_constructs_and_dispatches', '''
class Base:
	@abstractmethod
	def hello( self ) -> i32: ...

class Derived( Base ):
	@virtual
	def hello( self ) -> i32:
		return 42

def call_hello( b: Base ) -> i32:
	return b.hello()

def main() -> i32:
	d: Derived = Derived()
	if call_hello( d ) != 42:
		return 1
	return 0
''' ),
			# the exact scenario CStruct's own implicit stub-body convention
			# can't express - an abstract method declared at the root, still
			# unfulfilled through an intermediate level, only implemented at
			# the leaf, dispatched through a ROOT-typed reference
			( 'three_level_chain_abstract_fulfilled_only_at_leaf', '''
class Root:
	@abstractmethod
	def hello( self ) -> i32: ...

class Mid( Root ):
	pass

class Leaf( Mid ):
	@virtual
	def hello( self ) -> i32:
		return 7

def call_hello( r: Root ) -> i32:
	return r.hello()

def main() -> i32:
	leaf: Leaf = Leaf()
	if call_hello( leaf ) != 7:
		return 1
	return 0
''' ),
		] )

	@unittest.skipUnless( _CC is not None, 'no C compiler (clang/gcc/msvc) found - skipping' )
	def test_abstract_base_gets_no_vtable_instance_emitted( self ) -> None:
		# an abstract class used only as a base (never constructed
		# directly) never needs its own vtable instance - confirmed by
		# inspecting the emitted C directly, not just that it compiles
		self._run( '''
class Base:
	@abstractmethod
	def hello( self ) -> i32: ...
	x: i32
	def __init__( self, x: i32 ) -> None:
		self.x = x

class Derived( Base ):
	@virtual
	def hello( self ) -> i32:
		return self.x

def main() -> i32:
	d: Derived = Derived( x = 7 )
	if d.hello() != 7:
		return 1
	return 0
''' )
		self.assertEqual( self.discovery.errors.errors, [] )
		src = emitter_c.emit_c( self.compiler )
		self.assertNotIn( '__main__$Base$$vtable', src )
		self.assertIn( '__main__$Derived$$vtable', src )
		self._assert_compiles_and_runs( src )

class TupleFieldTeardownTests( test_support.RealCompileMixin, CompilerTestCase ):
	''' an RCClass holding a tuple-typed FIELD must release it in its own
	destructor.

	type_resolver.py's _build_field_teardown_ast is a separate re-derivation
	of "which parts of this type are RC" from cfg.py's, and it used to be an
	isinstance ladder that had no TupleType branch at all - a tuple field
	matched nothing and fell through to `return []`, so the owner simply never
	decref'd it. Every instance leaked its tuple, silently: the emitted
	destructor freed the object itself and never touched the field.

	Verified by refcount rather than by exit code alone - a leak does not
	crash, so nothing short of observing the refcount can fail on it. '''
	def setUp( self ) -> None:
		self.discovery = Discovery( import_builtins = True )
		self.compiler = Compiler( self.discovery )

	@unittest.skipUnless( test_support.HAS_CC, 'no C compiler (clang/gcc/msvc) found - skipping' )
	def test_programs_compile_and_run( self ) -> None:
		self.assert_programs_run([
			# the shared str must come back to refcount 1 after the Holder is
			# gone. 'x'.upper() (not a literal) forces a real heap allocation -
			# a literal binds to immortal static storage and can't distinguish
			# a leak from doing nothing.
			( 'rcclass_with_a_tuple_field_releases_it', '''
class Holder:
	t: tuple[str, i32]
	def __init__( self, t: tuple[str, i32] ) -> None:
		self.t = t

def main() -> i32:
	s: str = 'x'.upper()
	if compiler.refcount( s ) != 1:
		return 1
	h: Holder = Holder( t = ( s, 3 ) )
	if compiler.refcount( s ) != 2: # the tuple now holds a reference too
		return 2
	compiler.decref( h )
	if compiler.refcount( s ) != 1: # ...released again with the Holder
		return 3
	return 0
''' ),
			# and under repetition, which is what turns a missed release into
			# unbounded growth rather than one stray allocation
			( 'rc_lifetime_repeated_tuple_field_no_leak', '''
class Holder:
	t: tuple[str, i32]
	def __init__( self, t: tuple[str, i32] ) -> None:
		self.t = t
	def byte_len( self ) -> usize:
		return self.t[0].byte_len()

def main() -> i32:
	with compiler.wrap_arithmetic:
		i: i32 = 0
		while i < 1000:
			h: Holder = Holder( t = ( 'hello'.upper(), 1 ) )
			if h.byte_len() != 5:
				return 1
			i += 1
		return 0
''' ),
		] )

class AugAssignRealCompileTests( test_support.RealCompileMixin, CompilerTestCase ):
	''' real compile+run coverage for _stmt_AugAssign's Attribute/Subscript-
	target support (lowering.py) - unlike the IR-shape assertions in
	lowering_test.py, these confirm the generated code actually computes
	the right value AND doesn't corrupt memory (an RC attribute replaced
	many times in a loop is exactly the shape a wrong/missing decref would
	show up in). Needs real builtins, like
	UnionAsUnconstructedResultErrorTypeTests above. '''
	def setUp( self ) -> None:
		self.discovery = Discovery( import_builtins = True )
		self.compiler = Compiler( self.discovery )

	@unittest.skipUnless( test_support.HAS_CC, 'no C compiler (clang/gcc/msvc) found - skipping' )
	def test_programs_compile_and_run( self ) -> None:
		''' every real compile-and-run program in this class, merged into a
		single executable (one build for the whole class); a nonzero exit is
		decoded back to the failing sub-program and its own return code. '''
		self.assert_programs_run([
			# f.s += 'a' 200 times - exercises the dunder (str.__add__) dispatch
			# path plus cfg.attr_replace's decref of the OLD str each iteration;
			# a missing/wrong decref here either leaks or double-frees, and 200
			# iterations is enough for ASan/heap-corruption-on-double-free to
			# reliably surface if it were broken (confirmed against a
			# deliberately-reintroduced bug before writing this test)
			( 'attribute_target_rc_value_replaced_in_a_loop', '''
class Foo:
	s: str

def main() -> i32:
	f: Foo = Foo( s = "a" )
	i: i32 = 0
	while i < 200:
		f.s += "a"
		with compiler.wrap_arithmetic:
			i += 1
	if len( f.s ) != 201:
		return 1
	return 0
''' ),
			# p[0] += 5 through the flat GetItem/SetItem fallback (no
			# __getitem__/__setitem__) - confirms the read-modify-write actually
			# lands in the right memory, not just that it compiles
			( 'subscript_target_raw_pointer_fallback', '''
import sys

def main() -> i32:
	p: Ptr[i32] = sys.alloc[i32]( 1 )
	p[0] = 10
	with compiler.wrap_arithmetic:
		p[0] += 5
	v: i32 = p[0]
	sys.free( p )
	if v != 15:
		return 1
	return 0
''' ),
			# d[1] += 5 through a real __getitem__/__setitem__ pair (dict[K,V]) -
			# both fallible, auto-consumed exactly like an ordinary d[1] read/
			# write already is, and both driven off the SAME index operand
			# (lowered once). main() itself can't return Result (the C entry
			# point's signature is fixed - see emitter_c._is_entry_point), so
			# the dict logic lives in a helper that does, mirroring how every
			# other real-run test needing a fallible operation at top level
			# already structures this (see
			# UnionAsUnconstructedResultErrorTypeTests above)
			( 'subscript_target_with_real_getitem_setitem_methods', '''
def helper() -> Result[i32, KeyError]:
	d: dict[i32,i32] = dict[i32,i32]()
	d[1] = 10
	with compiler.wrap_arithmetic:
		d[1] += 5
	v: i32 = d[1]
	return Result.Ok( v )

def main() -> i32:
	v: i32 = helper().unwrap( 'x' )
	if v != 15:
		return 1
	return 0
''' ),
		] )

class _ClangCompileMixin:
	def _assert_compiles( self, c_source: str ) -> None:
		with tempfile.TemporaryDirectory() as tmp:
			src_path = Path( tmp ) / 'generated.c'
			obj_path = Path( tmp ) / 'generated.o'
			src_path.write_text( c_source, encoding = 'utf-8' )
			result = _CC.compile( src_path, obj_path )
			self.assertEqual( result.returncode, 0, f'{_CC.name} failed:\nstdout: {result.stdout}\nstderr: {result.stderr}{test_support.c_source_on_failure( c_source )}' )

@unittest.skipUnless( _CC is not None, 'no C compiler (clang/gcc/msvc) found - skipping real-compile verification' )
class RealCompileTests( _ClangCompileMixin, CompilerTestCase ):
	def test_empty_main_compiles( self ) -> None:
		self._run( '''
def main() -> None:
	return
''' )
		self._assert_compiles( emitter_c.emit_c( self.compiler ))

	def test_generic_method_returning_t_monomorphized_to_none_compiles( self ) -> None:
		# regression test: a generic method whose body does `return self.
		# <T-typed-field>` (Result[None,E].unwrap()'s own `return self.data
		# .v_Ok` is the real-world case that found this) still had a real
		# ir.Return(value=<some operand>) once T monomorphizes to NoneType -
		# _emit_instruction's own ir.Return handling only checked whether
		# THAT operand was Python None (i.e. "no expression"), not whether
		# the function's own C return type was void, so it emitted
		# `return $t0;` from a function _function_prototype had separately
		# (correctly) declared `void` - "void function should not return a
		# value" from every C compiler. Fixed by sharing one
		# _returns_void_in_c() check between the prototype and the return
		# statement itself.
		self._run( '''
@cstruct
class Box[T]:
	v: T
	def get( self ) -> T:
		return self.v

def main() -> None:
	b: Box[None] = Box( v = None )
	b.get()
	return
''' )
		self.assertEqual( self.discovery.errors.errors, [] )
		self._assert_compiles( emitter_c.emit_c( self.compiler ))

	def test_wrap_arithmetic_smoke_test_compiles( self ) -> None:
		# Phase 1 milestone (a): the first real smoke test, sidesteps
		# Result plumbing entirely
		self._run( '''
def main() -> i32:
	with compiler.wrap_arithmetic:
		x: i32 = 1
		return x + 1
''' )
		self._assert_compiles( emitter_c.emit_c( self.compiler ))

	def test_checked_cast_i32_to_usize_does_not_false_positive_overflow( self ) -> None:
		# regression test: __metalpy_wideint (used to range-check every
		# Check-mode cast) used to be selected by a bare `#ifdef _MSC_VER`
		# in the emitted C prelude - but clang targeting Windows ALSO
		# defines _MSC_VER (for MSVC source compatibility) despite fully
		# supporting __int128, unlike real MSVC (cl.exe). That wrongly
		# routed clang-on-Windows through the int64_t/uint64_t fallback,
		# whose __metalpy_wideint (signed 64-bit) can't represent usize's
		# full unsigned range - a checked cast to usize/u64 (see _emit_cast
		# in emitter_c.py) could then produce a garbage max-value constant
		# once forced into that signed 64-bit type, causing even a tiny,
		# clearly in-range value like 11 to spuriously "overflow". Found by
		# str.upper()'s own real end-to-end test (see StrUpperLowerTests)
		# panicking on ordinary short ASCII input.
		self._run( '''
def main() -> i32:
	x: i32 = 11
	y: usize
	with compiler.saturate_arithmetic:
		y = usize( x )
	if y != 11:
		return 1
	return 0
''' )
		self.assertEqual( self.discovery.errors.errors, [] )
		c_source = emitter_c.emit_c( self.compiler )
		with tempfile.TemporaryDirectory() as tmp:
			src_path = Path( tmp ) / 'generated.c'
			obj_path = Path( tmp ) / 'generated.o'
			exe_path = Path( tmp ) / 'test_exe'
			src_path.write_text( c_source, encoding = 'utf-8' )
			cc_result = _CC.compile( src_path, obj_path )
			self.assertEqual( cc_result.returncode, 0, f'{_CC.name} compile failed:\n{cc_result.stdout}' )
			link_result = _CC.link( exe_path, [ obj_path ] )
			self.assertEqual( link_result.returncode, 0, f'{_CC.name} link failed:\n{link_result.stdout}' )
			run_result = subprocess.run( [ str( exe_path ) ], capture_output = True )
			self.assertEqual( run_result.returncode, 0, f'exited {run_result.returncode}, expected 0 (a nonzero panic-exit or wrong-value exit means the false-overflow bug regressed)' )

	def test_default_check_mode_arithmetic_compiles( self ) -> None:
		# Phase 1 milestone (b): default Check-mode arithmetic, proving
		# AddCheck + the synthesized Result[i32,OverflowError] struct +
		# OrReturn all compile clean together. Also exercises generic-
		# class-method monomorphization end to end (Result.Ok is a method
		# on a generic CStruct - see lowering.py's _lower_class_generic_
		# method_call/Lowering.monomorphize_class) - previously a real,
		# documented gap (Result.Ok's parameters/return_type stayed
		# literally TypeVar-typed, with no C representation at all), now
		# resolved at the lowering.py/compiler.py level, not worked around
		# in the emitter.
		# main() itself always compiles to C's own `int main(void)`
		# (see emit_function's entry-point special-case) - a realistic
		# program never declares main() -> Result[...] (that wouldn't even
		# make sense given main always returns int), so the Result-
		# returning function under test is a separate, ordinary helper
		self._run( _RESULT_FIXTURE + '\n' + '\n'.join([
			'def foo() -> Result[i32,OverflowError]:',
			'	x: i32 = 1',
			'	y: i32 = x + 1',
			'	return Result.Ok( y )',
			'',
			'def main() -> None:',
			'	foo()',
		]))
		self._assert_compiles( emitter_c.emit_c( self.compiler ))

	def test_cenum_typedef_and_member_reference_compiles( self ) -> None:
		# Phase 2 milestone: "WindowsError/ErrnoError declared + one member
		# referenced, compiles clean" - built directly from emit_cenum's own
		# per-unit output (decision 1) rather than through a full
		# compiler.run(), since CEnum member-VALUE expressions (Color.Red
		# used as a real value) are a pre-existing lowering.py gap, not
		# emitter_c.py scope - see EmitCEnumTests' own comment and the
		# plan's grounding facts ("CEnum never being referenced by real
		# lowering.py output today"). This still proves the emitted
		# typedef+consts are real, valid C - "one member referenced" here
		# means the generated static const symbol itself, used from a
		# plain C harness.
		color = CEnum( stem = 'Color', qualname = '__main__.Color', file = None, line = None, value_type = _scalar( 'u32' ))
		color.members = { 'Red': 0, 'Green': 1 }
		color.values = { 0: 'Red', 1: 'Green' }
		harness = '\n'.join([
			'#include <stdint.h>',
			emitter_c.emit_cenum( color ),
			'int main( void ) {',
			'\treturn (int)__main__$Color$Red;',
			'}',
		])
		self._assert_compiles( harness )

	def test_cstruct_construct_and_read_back_compiles( self ) -> None:
		self._run( _POINT_FIXTURE + '\n' + '\n'.join([
			'def main() -> i32:',
			'	p: Point = Point.make( 1, 2 )',
			'	with compiler.wrap_arithmetic:', # sidesteps needing a Result[i32,OverflowError] fixture - default Check mode isn't what this test is about
			'		return p.x + p.y',
		]))
		self._assert_compiles( emitter_c.emit_c( self.compiler ))

	def test_addrof_getitem_setitem_compiles( self ) -> None:
		self._run( '''
def main() -> None:
	x: u8 = 5
	i: usize = 0
	seven: u8 = 7
	p: Ptr[u8] = compiler.addrof( x )
	p[i] = seven
	y: u8 = p[i]
	return
''' )
		self._assert_compiles( emitter_c.emit_c( self.compiler ))

	def test_union_construct_and_match_compiles( self ) -> None:
		# Phase 5 milestone (a): synthetic @union construct/match round trip
		self._run( _UNION_FIXTURE + '\n' + '\n'.join([
			'def main() -> i32:',
			'	f: Foo = Foo.Bar( 5 )',
			'	match f:',
			'		case Foo.Bar( x ):',
			'			return x',
			'		case Foo.Baz( z ):',
			'			with compiler.wrap_arithmetic:',
			'				return z + 1',
			'	return 0',
		]))
		self._assert_compiles( emitter_c.emit_c( self.compiler ))

	def test_ptr_or_none_return_and_is_none_check_compiles( self ) -> None:
		# Phase 5 milestone (b): mirrors the real, load-bearing shape every
		# allocation in the language ultimately runs through - lib/sys.py's
		# own _alloc(size: usize) -> Ptr[u8]|None, consumed via `if ptr is
		# None:`. A synthetic extern stands in for the real HeapAlloc/malloc
		# call (same posture as every other fixture in this file - no real
		# lib/ dependency needed to exercise this shape)
		self._run( '\n'.join([
			"@extern( 'c', '_metalpy_test_maybe_alloc' )",
			'def _test_maybe_alloc( size: usize ) -> Ptr[u8]|None:',
			'	...',
			'',
			'def alloc_or_none( size: usize ) -> Ptr[u8]|None:',
			'	ptr = _test_maybe_alloc( size )',
			'	return ptr',
			'',
			'def main() -> i32:',
			'	p: Ptr[u8]|None = alloc_or_none( 4 )',
			'	if p is None:',
			'		return 0',
			'	return 1',
		]))
		self._assert_compiles( emitter_c.emit_c( self.compiler ))

	def test_scalar_casts_in_every_arithmetic_mode_compile( self ) -> None:
		# CastWrap/CastCheck/CastSaturate (compiler.cast(...)/T(x) sugar) -
		# CastWrap is already exercised end to end by the Phase 6 milestone
		# (RealCompileTests further down), this covers the other two modes
		# directly
		self._run( _RESULT_FIXTURE + '\n' + '\n'.join([
			'def foo() -> Result[u32,OverflowError]:',
			'	x: usize = 300',
			'	with compiler.saturate_arithmetic:',
			'		y: u8 = u8( x )', # narrowing, out of range - clamps to 255
			'	z: u32 = compiler.cast( u32, x )', # default Check mode
			'	return Result.Ok( z )',
			'',
			'def main() -> None:',
			'	foo()',
		]))
		self._assert_compiles( emitter_c.emit_c( self.compiler ))

	def test_defer_compiles( self ) -> None:
		# Phase 7 (confirm-only): defer/errdefer already fully desugar to
		# plain Label/Jump/JumpIfFalse by stage 2 (lowering.py replays the
		# deferred body inline at the epilogue) - Phase 1's own coverage of
		# those ops should already be sufficient, with zero new emitter
		# code needed. Confirmed here by an actual real compile, not just
		# an IR-shape assertion.
		self._run( '\n'.join([
			'def cleanup() -> None:',
			'	return',
			'',
			'def main() -> None:',
			'	defer( cleanup() )',
			'	return',
		]))
		self._assert_compiles( emitter_c.emit_c( self.compiler ))

	def test_errdefer_compiles( self ) -> None:
		self._run( _RESULT_FIXTURE + '\n' + '\n'.join([
			'def cleanup() -> None:',
			'	return',
			'',
			'def checked() -> Result[i32,OverflowError]:',
			'	errdefer( cleanup() )',
			'	return Result.Ok( 5 )',
			'',
			'def main() -> None:',
			'	checked()',
			'	return',
		]))
		self._assert_compiles( emitter_c.emit_c( self.compiler ))

	def test_generic_union_method_and_construction_arg_use_substituted_types( self ) -> None:
		# regression test for two real bugs found verifying Result-as-@union
		# (lib/builtins/__init__.py) against actual generic-union usage
		# beyond _RESULT_FIXTURE's own minimal is_ok/is_err:
		# (1) _attr_lookup didn't substitute a TaggedUnion's own
		#     synthesized `data` field for a concrete specialization - a
		#     method body reading `self.data.v_<member>` directly (e.g.
		#     unwrap_ok below, mirroring Result.unwrap's real body) got the
		#     ABSTRACT, shared, TypeVar-typed payload_cls, which then got
		#     scheduled as a real but ill-typed compile unit the moment
		#     anything read it.
		# (2) _try_lower_union_construct_call didn't substitute the
		#     expected_type used to lower `TaggedUnion.Member(<value>)`'s
		#     own value expression - a Check-mode arithmetic expression
		#     nested directly inside a union constructor call (e.g.
		#     `Result.Ok(y + 1)`) got typed against the abstract member's
		#     own bare TypeVar instead of the concrete specialization,
		#     corrupting the Result[T,OverflowError] the arithmetic itself
		#     needs to build into a bogus, unsubstituted one.
		self._run( _RESULT_FIXTURE + '\n' + '\n'.join([
			'@union',
			'class Box[T]:',
			'\tFull: T',
			'',
			'\tdef unwrap_ok( self ) -> T:',
			'\t\treturn self.data.v_Full',
			'',
			'def make_full( x: i32 ) -> Box[i32]:',
			'\treturn Box.Full( x )',
			'',
			'def add_one( x: i32 ) -> Result[i32,OverflowError]:',
			'\treturn Result.Ok( x + 1 )',
			'',
			'def main() -> None:',
			'\tb = make_full( 10 )',
			'\tv = b.unwrap_ok()',
			'\tr = add_one( v )',
		]))
		self._assert_compiles( emitter_c.emit_c( self.compiler ))

	def test_generic_specialization_naming_compiles( self ) -> None:
		# Phase 7 (confirm-only): generic monomorphization naming is
		# already covered by decision 3's uniform mangling, exercised
		# repeatedly by Phases 3/5 already (sys.alloc[T], Result[T,E],
		# ResultPayload[T,E]) - this just adds one direct, explicit-
		# generic-call-syntax confirmation (Name[T](...), not just the
		# inferred/bare-call path every other fixture already goes through)
		self._run( '\n'.join([
			'def identity[T]( x: T ) -> T:',
			'	return x',
			'',
			'def main() -> i32:',
			'	return identity[i32]( 5 )',
		]))
		self._assert_compiles( emitter_c.emit_c( self.compiler ))

	def test_overload_call_on_generic_class_specialization_compiles( self ) -> None:
		# regression test (see lowering_test.py's own identical-purpose
		# test) for a real, pre-existing overload_resolution.py bug found
		# while verifying Result-as-@union: candidate matching for an
		# @overload group declared inside a generic class compared the
		# call's real, concrete argument type directly against the
		# group's own abstract, unsubstituted class TypeVar, so it never
		# matched - affects real library code directly (builtins.Result
		# [T,E].unwrap_or's own `default: T` stub). Uses a @union (not an
		# RCClass) receiver deliberately - generic RCClass construction via
		# plain ClassName(...) has its own, separate, unrelated gap (its
		# __init__ never gets monomorphized when reached this way), not
		# what this test is checking.
		self._run( '\n'.join([
			'@union',
			'class Box[T]:',
			'	Some: T',
			'',
			'	@overload',
			'	def get_or( self, default: T ) -> T:',
			'		...',
			'	def get_or( self, default: T ) -> T:',
			'		return self.data.v_Some',
			'',
			'def main() -> i32:',
			'	b: Box[i32] = Box.Some( 5 )',
			'	fallback: i32 = -1',
			'	return b.get_or( fallback )',
		]))
		self._assert_compiles( emitter_c.emit_c( self.compiler ))

# routing RCClass construction through the REAL sys.alloc[T] means sys.alloc's
# own body actually gets lowered end to end (unlike every other fixture in
# this file, which never touches real lib/ code) - the real lib/sys.py's own
# alloc[T] pulls in Windows/CRT externs, panic, and a synthesized Ptr[u8]|None
# TaggedUnion (Phase 5 work, not built yet), none of which is what THIS phase
# is actually testing. A self-contained sys.py fixture (a temp-directory
# Discovery search path, not the real lib/) keeps this phase scoped to what
# it says: RCClass layout + the sys.alloc[T] calling convention, not "make
# the entire current (and still actively evolving - see TODO.txt) real
# stdlib compile."
_SYS_FIXTURE = '\n'.join([
	'def alloc[T]( count: usize ) -> Ptr[T]:',
	'	with compiler.wrap_arithmetic:', # sidesteps needing a Result[usize,OverflowError] fixture - default Check mode isn't what this phase is testing
	'		byte_count: usize = count * compiler.sizeof( T )',
	'	return _test_raw_alloc[T]( byte_count )',
	'',
	"@extern( 'c', '_metalpy_test_alloc' )", # a fictitious symbol name - avoids any collision with clang's own builtin knowledge of real allocator names like malloc
	'def _test_raw_alloc[T]( size: usize ) -> Ptr[T]:',
	'	...',
	'',
	'def free( ptr: Ptr[None] ) -> None:',
	'	_test_raw_free( ptr )',
	'',
	"@extern( 'c', '_metalpy_test_free' )",
	'def _test_raw_free( ptr: Ptr[None] ) -> None:',
	'	...',
])

class RCClassTestCase( CompilerTestCase ):
	def setUp( self ) -> None:
		#self._tmpdir = tempfile.TemporaryDirectory()
		#self.addCleanup( self._tmpdir.cleanup )
		#tmp_path = Path( self._tmpdir.name )
		#( tmp_path / 'sys.py' ).write_text( _SYS_FIXTURE, encoding = 'utf-8' )
		self.discovery = Discovery(
			#paths = [ tmp_path ],
			import_builtins = True,
		)
		self.compiler = Compiler( self.discovery )

_FOO_FIXTURE = '\n'.join([
	'class Foo:',
	'	x: i32',
	'',
	'	@staticmethod',
	'	def make( v: i32 ) -> Foo:',
	'		return Foo.__allocate__( x = v )',
])

class RCClassConstructTests( RCClassTestCase ):
	def test_construct_read_back_and_refcount( self ) -> None:
		self._run( _FOO_FIXTURE + '\n' + '\n'.join([
			'def main() -> None:',
			'	foo: Foo = Foo.make( 1 )',
			'	bar: Foo = foo', # aliasing - exercises Incref
			'	rc: usize = compiler.refcount( bar )',
			'	return',
		]))
		self.assertEqual( self.discovery.errors.errors, [] )
		main_lf = next( lf for lf in self.compiler.functions if lf.function.qualname == 'main' )
		src = emitter_c.emit_function( main_lf )
		self.assertIn( 'retain_object', src ) # bar = foo aliasing
		self.assertIn( 'release_object', src ) # epilogue decref(s) for foo/bar going out of scope
		self.assertIn( '->$header.ref_count', src ) # compiler.refcount(bar)
		foo_cls = next( cls for cls in self.compiler.rcclasses if cls.qualname == '__main__.Foo' )
		struct_src = emitter_c.emit_rcclass( foo_cls )
		self.assertIn( 'struct __main__$Foo {', struct_src )
		self.assertIn( 'ObjectHeader $header;', struct_src )
		self.assertIn( 'int32_t x;', struct_src )

	def test_construction_sets_header_destructor_field( self ) -> None:
		# every RCClass construction wires up $header.vtable to its own
		# static $$vtable instance (RCClass-subclassing plan Phase 4 -
		# unified destructor dispatch with @virtual dispatch, see
		# ObjectHeader's own comment); release_object (both the ordinary
		# Decref path and the type-erased DecrefDynamic path a closure's
		# own __del__ uses) reads the destructor back through
		# $header.vtable->destroy uniformly - see ObjectHeader's own
		# comment on why passing it again as an explicit argument at every
		# release site would just be redundant with what's already there
		self._run( _FOO_FIXTURE + '\n' + '\n'.join([
			'def main() -> None:',
			'	foo: Foo = Foo.make( 1 )',
			'	return',
		]))
		self.assertEqual( self.discovery.errors.errors, [] )
		make_lf = next( lf for lf in self.compiler.functions if lf.function.qualname == '__main__.Foo.make' )
		src = emitter_c.emit_function( make_lf )
		self.assertIn( '$header.vtable = &__main__$Foo$$vtable;', src )
		# Foo has no @virtual methods anywhere in its chain - its own
		# $$vtable instance uses the shared, built-in minimal type, not a
		# per-class synthesized one (no cast needed at the wiring site)
		self.assertNotIn( '(const __metalpy_ObjectVtbl*)&__main__$Foo$$vtable', src )
		# ordinary Decref (main's own epilogue for `foo`) calls the single,
		# merged release_object with just the header pointer - no
		# destructor argument, no _rcclass_destructor_name reference here
		main2_lf = next( lf for lf in self.compiler.functions if lf.function.qualname == 'main' )
		main_src = emitter_c.emit_function( main2_lf )
		self.assertIn( 'release_object( (ObjectHeader*)(foo) )', main_src )
		self.assertNotIn( '__main__$Foo$$__destructor__', main_src )

	def test_release_object_reads_destructor_from_header( self ) -> None:
		# a single, merged release_object - see ObjectHeader's own comment
		# on why a separate release_object_dynamic isn't needed: every
		# release already has the destructor one field-read away (through
		# $header.vtable->destroy - unified with @virtual dispatch, see
		# Phase 4 of the RCClass-subclassing plan), adjacent to ref_count
		# in the same cache line the atomic decrement below already touches
		c_source = emitter_c.PROLOGUE
		self.assertIn( 'void (*destroy)( void* );', c_source )
		self.assertIn( 'const __metalpy_ObjectVtbl* vtable;', c_source )
		self.assertIn( 'static inline void release_object( ObjectHeader* obj )', c_source )
		self.assertIn( 'obj->vtable->destroy( obj );', c_source )
		self.assertNotIn( 'release_object_dynamic', c_source )

	def test_init_construction_schedules_sys_alloc_for_the_constructed_class( self ) -> None:
		# regression test: _try_lower_construct_call's own ir.Allocate (the
		# real __init__ path, as opposed to _lower_allocate_fields's field=
		# value sugar above) never scheduled sys.alloc[TargetClass]/sys.free/
		# __del__ at all - confirmed via a full compiler.run(), sys.alloc[Bar]
		# was simply absent from compiler.functions, only sys.alloc[Foo] (the
		# unrelated field sub-expression, which DOES go through
		# _lower_allocate_fields) showed up
		self._run( '\n'.join([
			'class Foo: pass',
			'class Bar:',
			'	a: Foo',
			'	def __init__( self, x: Foo ) -> None:',
			'		self.a = x',
			'',
			'def main() -> None:',
			'	b = Bar( Foo() )',
			'	return',
		]))
		self.assertEqual( self.discovery.errors.errors, [] )
		alloc_specializations = [
			f.function.qualname for f in self.compiler.functions
			if f.function.qualname.startswith( 'sys.alloc[' )
		]
		self.assertIn( 'sys.alloc[__main__.Bar]', alloc_specializations )
		self.assertIn( 'sys.alloc[__main__.Foo]', alloc_specializations )

	def test_generic_init_construction_two_instantiations_are_independent( self ) -> None:
		# Stage 3b: two different concrete instantiations of the same
		# generic RCClass's __init__ get their own independent, correctly-
		# substituted compiled bodies - not a single shared, abstract one
		# (which could never emit correct C for more than one concrete type)
		self._run( '\n'.join([
			'class Box[T]:',
			'	v: T',
			'	def __init__( self, v: T ) -> None:',
			'		self.v = v',
			'',
			'def main() -> None:',
			'	b: Box[i32] = Box( 1 )',
			'	c: Box[u32]',
			'	c = Box( 2 )',
			'	return',
		]))
		self.assertEqual( self.discovery.errors.errors, [] )
		box_rcclasses = [ cls.qualname for cls in self.compiler.rcclasses if cls.qualname.startswith( '__main__.Box' ) ]
		self.assertEqual( sorted( box_rcclasses ), [ '__main__.Box[intrinsics.i32]', '__main__.Box[intrinsics.u32]' ]) # exactly one each, no abstract Box, no duplicates
		init_fns = { f.function.qualname: f for f in self.compiler.functions if f.function.qualname.startswith( '__main__.Box.__init__' ) }
		self.assertEqual( set( init_fns.keys() ), { '__main__.Box.__init__[intrinsics.i32]', '__main__.Box.__init__[intrinsics.u32]' })
		i32_setattr = next( i for i in init_fns['__main__.Box.__init__[intrinsics.i32]'].instructions if isinstance( i, ir.SetAttr ))
		u32_setattr = next( i for i in init_fns['__main__.Box.__init__[intrinsics.u32]'].instructions if isinstance( i, ir.SetAttr ))
		self.assertEqual( i32_setattr.value.type.qualname, 'intrinsics.i32' ) # substituted, not a shared bare T
		self.assertEqual( u32_setattr.value.type.qualname, 'intrinsics.u32' )

	def test_generic_init_construction_infers_type_args_from_arguments( self ) -> None:
		# Stage 3b: no expected_type annotation pinning the concrete args -
		# infer from __init__'s own arguments instead (mirrors
		# _lower_class_generic_method_call's identical inference for
		# Result.Ok(val)), not just "always require an annotation"
		self._run( '\n'.join([
			'class Box[T]:',
			'	v: T',
			'	def __init__( self, v: T ) -> None:',
			'		self.v = v',
			'',
			'def main() -> None:',
			'	x: i32 = 5',
			'	b = Box( x )',
			'	return',
		]))
		self.assertEqual( self.discovery.errors.errors, [] )
		self.assertIn( '__main__.Box[intrinsics.i32]', [ cls.qualname for cls in self.compiler.rcclasses ])

	def test_generic_init_construction_defaults_still_apply( self ) -> None:
		# exercises lower_function's own construction-self setup: fn.cls is
		# a Specialization for a monomorphized generic __init__ (see
		# Lowering.lower_function's own unwrap-for-isinstance-checks fix) -
		# a non-generic field with a class-level default must still get its
		# default applied BEFORE __init__'s own body runs, same as a plain
		# non-generic RCClass's own construction defaults
		self._run( '\n'.join([
			'class Box[T]:',
			'	v: T',
			'	count: i32 = 0',
			'	def __init__( self, v: T ) -> None:',
			'		self.v = v',
			'',
			'def main() -> None:',
			'	b: Box[i32] = Box( 1 )',
			'	return',
		]))
		self.assertEqual( self.discovery.errors.errors, [] )
		init_fn = next( f for f in self.compiler.functions if f.function.qualname == '__main__.Box.__init__[intrinsics.i32]' )
		setattrs = [ i for i in init_fn.instructions if isinstance( i, ir.SetAttr ) ]
		self.assertEqual( [ i.attr for i in setattrs ], [ 'count', 'v' ]) # default applied before __init__'s own body

	def test_generic_init_construction_pins_type_args_through_fallible_wrapping( self ) -> None:
		# regression test: for a FALLIBLE __init__, expected_type here is
		# Result[Box[i32],MyError] (SYNTAX.md's own fallible-construction
		# wrapping), not Box[i32] directly - the pinning check has to see
		# through that one level of Result[_,_] wrapping (find_name_or_none,
		# not find_name - a program that never defines Result at all must
		# not hard-fail here), or it silently falls back to argument-based
		# inference and - a separate, narrower pre-existing gap this
		# surfaced along the way (_expr_Constant accepts a still-abstract
		# TypeVar as an expected_type without complaint) - builds a
		# nonsensical Box[Box.T] instead of Box[i32]
		self._run( '\n'.join([
			'@cstruct',
			'class MyError: pass',
			'',
			'@union',
			'class Result[T,E]:',
			'	Ok: T',
			'	Err: E',
			'	def is_ok( self ) -> bool:',
			'		return self.tag == 0',
			'	def is_err( self ) -> bool:',
			'		return self.tag == 1',
			'',
			'class Box[T]:',
			'	v: T',
			'	def __init__( self, v: T ) -> Result[None,MyError]:',
			'		self.v = v',
			'		return Result.Ok( None )',
			'',
			'def main() -> None:',
			'	r: Result[Box[i32],MyError] = Box( 1 )',
			'	r.is_ok()',
			'	return',
		]))
		self.assertEqual( self.discovery.errors.errors, [] )
		# the annotation alone (r: Result[Box[i32],MyError]) schedules
		# Box[intrinsics.i32] regardless of whether the constructor call
		# itself resolved correctly, so check the CALL's own target instead -
		# this is what actually failed before the fix (target was
		# Box.__init__[Box.T], not Box.__init__[intrinsics.i32])
		init_fns = [ f.function.qualname for f in self.compiler.functions if f.function.qualname.startswith( '__main__.Box.__init__' ) ]
		self.assertEqual( init_fns, [ '__main__.Box.__init__[intrinsics.i32]' ])

	# regression tests: a fallible __init__ whose ERROR type is itself one
	# of the class's own type params (Result[None,T], not a fixed
	# unrelated error class like MyError above) - substituting T for a
	# concrete type actually changes Result[None,T]'s own args, which
	# Monomorphizer.substitute_type_params's eager-monomorphize step (see
	# PLAN_RESOLVE_CLASS_SPECIALIZATIONS.md) then immediately resolves
	# into a real, concrete TaggedUnion object instead of leaving it a
	# Specialization - _result_shape/_require_result_return/_unify_type_
	# param all used to assume a Result-shaped type was ALWAYS still a
	# bare Specialization, so this used to fail with a spurious "must
	# return None or Result[None,_]" (or, once that was fixed, "cannot
	# infer type parameter(s) E", or "inferred as both X and X" -
	# comparing a bare Specialization against its own already-
	# monomorphized form by identity) even though init's own return type
	# is perfectly well-formed
	_GENERIC_ERROR_TYPE_FIXTURE = '\n'.join([
		'@union',
		'class Result[T,E]:',
		'	Ok: T',
		'	Err: E',
		'	def is_ok( self ) -> bool:',
		'		return self.tag == 0',
		'	def is_err( self ) -> bool:',
		'		return self.tag == 1',
		'',
		'class Box[T]:',
		'	v: T',
		'	def __init__( self, v: T ) -> Result[None,T]:',
		'		self.v = v',
		'		return Result.Ok( None )',
		'',
	])

	def test_generic_init_construction_error_type_reusing_the_class_own_type_param_compiles( self ) -> None:
		# non-literal argument - resolved by type_resolver.py's own tagging
		self._run( self._GENERIC_ERROR_TYPE_FIXTURE + '\n'.join([
			'def main() -> None:',
			'	x: i32 = 5',
			'	r: Result[Box[i32],i32] = Box( x )',
			'	r.is_ok()',
			'	return',
		]))
		self.assertEqual( self.discovery.errors.errors, [] )

	def test_generic_init_construction_error_type_reusing_the_class_own_type_param_compiles_via_untagged_fallback( self ) -> None:
		# literal argument - falls back to lowering.py's own untagged path
		# (Lowering._lower_generic_construction_args), never tagged by
		# type_resolver.py at all (trust_literals=False)
		self._run( self._GENERIC_ERROR_TYPE_FIXTURE + '\n'.join([
			'def main() -> None:',
			'	r: Result[Box[i32],i32] = Box( 5 )',
			'	r.is_ok()',
			'	return',
		]))
		self.assertEqual( self.discovery.errors.errors, [] )

	def test_generic_init_construction_uninferable_type_args_is_a_clear_error( self ) -> None:
		# T genuinely never appears in __init__'s own parameter list here, so
		# nothing could ever bind it - a clean "cannot infer" error, not a
		# crash or a silently-wrong result. (A bare literal AT an inferred
		# position, e.g. Box(5) where __init__ takes v: T, is a narrower,
		# pre-existing gap this construction path inherits unchanged from
		# _lower_class_generic_method_call's identical inference branch:
		# _expr_Constant only rejects expected_type is None, not "still an
		# unresolved TypeVar" - it silently builds a nonsensical Box[T]
		# rather than failing. Not introduced by this work and not fixed
		# here - same posture as the other documented-not-fixed gaps this
		# session found.)
		self._run( '\n'.join([
			'class Box[T]:',
			'	v: T',
			'	def __init__( self, other: i32 ) -> None:',
			'		pass',
			'',
			'def main() -> None:',
			'	b = Box( 5 )',
			'	return',
		]))
		self.assertTrue( any( 'cannot infer' in e for e in self.discovery.errors.errors ))

	def test_user_field_named_header_does_not_collide( self ) -> None:
		# the automatic ObjectHeader member is named $header, not header -
		# '$' can never appear in a real metalpy identifier, so a user class
		# declaring its own field named `header` can't collide with it
		self._run( '\n'.join([
			'class Foo:',
			'	header: i32',
			'',
			'	@staticmethod',
			'	def make( v: i32 ) -> Foo:',
			'		return Foo.__allocate__( header = v )',
			'',
			'def main() -> None:',
			'	f: Foo = Foo.make( 1 )',
			'	return',
		]))
		self.assertEqual( self.discovery.errors.errors, [] )
		foo_cls = next( cls for cls in self.compiler.rcclasses if cls.qualname == '__main__.Foo' )
		struct_src = emitter_c.emit_rcclass( foo_cls )
		self.assertIn( 'ObjectHeader $header;', struct_src )
		self.assertIn( 'int32_t header;', struct_src )

_OWNER_FIXTURE = '\n'.join([
	'import sys',
	'',
	'class Owner:',
	'	ptr: Ptr[None]',
	'',
	'	@staticmethod',
	'	def make( p: Ptr[None] ) -> Owner:',
	'		return Owner.__allocate__( ptr = p )',
	'',
	'	def __del__( self ) -> None:',
	'		sys.free( self.ptr )',
])

_BOX_FIXTURE = _FOO_FIXTURE + '\n' + '\n'.join([
	'',
	'class Box:',
	'	inner: Foo',
	'',
	'	@staticmethod',
	'	def make( f: Foo ) -> Box:',
	'		return Box.__allocate__( inner = f )',
])

class RCClassDestructorTests( RCClassTestCase ):
	def _emit_and_find_destructor( self, qualname: str ) -> str:
		''' emit full C and extract the destructor body for the given
		class qualname. '''
		c_src = emitter_c.emit_c( self.compiler )
		mangled = emitter_c.mangle_qualname( qualname )
		dtor_name = f'{mangled}$$__destructor__'
		lines = c_src.split( chr( 10 ))
		start = None
		for i, line in enumerate( lines ):
			if dtor_name in line and 'void* __obj ) {' in line:
				start = i
				break
		self.assertIsNotNone( start, f'destructor {dtor_name} not found in emitted C' )
		# collect lines until the closing brace
		body_lines = []
		for j in range( start, len( lines )):
			body_lines.append( lines[j] )
			if lines[j].strip() == '}':
				break
		return '\n'.join( body_lines )

	def test_del_method_is_called_from_synthesized_destructor( self ) -> None:
		self._run( _OWNER_FIXTURE + '\n' + '\n'.join([
			'def main() -> None:',
			'	x: i32 = 0',
			'	p: Ptr[None] = compiler.addrof( x )',
			'	o: Owner = Owner.make( p )',
			'	return',
		]))
		self.assertEqual( self.discovery.errors.errors, [] )
		destructor_src = self._emit_and_find_destructor( '__main__.Owner' )
		self.assertIn( '__main__$Owner$__del__( self )', destructor_src )
		# self (a bare RCClass) now goes through an explicit compiler.cast(...)
		# before reaching sys.free (see type_resolver.py's _synthesize_
		# rcclass_destructor - lowering.py's new general assignability check,
		# _check_assignable, made the implicit self->Ptr[u8] reinterpret this
		# destructor used to rely on into a real compile error, same as any
		# other mismatched call site now gets), so the argument is a real
		# temp holding the cast result, not `self` passed bare
		self.assertIn( 'sys$free( (void*)($t0) )', destructor_src )
		# __del__ runs BEFORE sys.free - fields must still be valid when it runs
		self.assertLess( destructor_src.index( '__del__' ), destructor_src.index( 'sys$free' ))

	def test_rcclass_field_cascades_decref_with_no_user_del( self ) -> None:
		self._run( _BOX_FIXTURE + '\n' + '\n'.join([
			'def main() -> None:',
			'	f: Foo = Foo.make( 1 )',
			'	b: Box = Box.make( f )',
			'	return',
		]))
		self.assertEqual( self.discovery.errors.errors, [] )
		destructor_src = self._emit_and_find_destructor( '__main__.Box' )
		self.assertNotIn( '__del__', destructor_src ) # Box declares none
		# release_object now reads the field's own destructor back off its
		# own header at runtime (see ObjectHeader's own comment) rather
		# than this call site naming it as a literal argument
		self.assertIn( 'release_object( (ObjectHeader*)($t0) )', destructor_src )
		# see test_del_method_is_called_from_synthesized_destructor's own
		# comment on why this is a real temp ($t1, following the field
		# decref's own $t0) rather than `self` passed bare
		self.assertIn( 'sys$free( (void*)($t1) )', destructor_src )

	def test_taggedunion_field_cascades_a_tag_gated_decref( self ) -> None:
		# a TaggedUnion-typed field with an RC-leaf member (MaybeFoo.Some)
		# goes through normal lowering now — the tag check becomes an
		# ordinary if statement in the lowered IR, no longer raw C text
		self._run( _FOO_FIXTURE + '\n' + '\n'.join([
			'@union',
			'class MaybeFoo:',
			'	Some: Foo',
			'	Nothing: i32',
			'',
			'class Box:',
			'	maybe: MaybeFoo',
			'',
			'	@staticmethod',
			'	def make( f: Foo ) -> Box:',
			'		return Box.__allocate__( maybe = MaybeFoo.Some( f ) )',
			'',
			'def main() -> None:',
			'	f: Foo = Foo.make( 1 )',
			'	b: Box = Box.make( f )',
			'	return',
		]))
		self.assertEqual( self.discovery.errors.errors, [] )
		destructor_src = self._emit_and_find_destructor( '__main__.Box' )
		# should have a tag comparison (lowered from the AST if statement)
		self.assertIn( '.tag', destructor_src )
		self.assertIn( 'goto L_', destructor_src )  # the if/else branching
		# should decref the RC member only
		self.assertIn( 'release_object', destructor_src )
		self.assertIn( 'v_Some', destructor_src )
		self.assertNotIn( 'v_Nothing', destructor_src ) # the non-RC member needs no branch at all

	def test_nested_cstruct_field_cascades_decref_into_its_own_fields( self ) -> None:
		# a by-value CStruct field is always fully live (unlike a union, no
		# discriminant needed) - the cascade walks through the struct to
		# reach the nested RC leaf
		self._run( _FOO_FIXTURE + '\n' + '\n'.join([
			'@cstruct',
			'class Wrapper:',
			'	inner: Foo',
			'',
			'class Box:',
			'	w: Wrapper',
			'',
			'	@staticmethod',
			'	def make( f: Foo ) -> Box:',
			'		return Box.__allocate__( w = Wrapper( inner = f ) )',
			'',
			'def main() -> None:',
			'	f: Foo = Foo.make( 1 )',
			'	b: Box = Box.make( f )',
			'	return',
		]))
		self.assertEqual( self.discovery.errors.errors, [] )
		destructor_src = self._emit_and_find_destructor( '__main__.Box' )
		# release_object now reads the field's own destructor back off its
		# own header at runtime rather than this call site naming it
		self.assertIn( 'release_object( (ObjectHeader*)($t1) )', destructor_src )

@unittest.skipUnless( _CC is not None, 'no C compiler (clang or gcc) found - skipping real-compile verification' )
class RCClassDestructorRealCompileTests( _ClangCompileMixin, RCClassTestCase ):
	def test_del_method_compiles( self ) -> None:
		self._run( _OWNER_FIXTURE + '\n' + '\n'.join([
			'def main() -> None:',
			'	x: i32 = 0',
			'	p: Ptr[None] = compiler.addrof( x )',
			'	o: Owner = Owner.make( p )',
			'	return',
		]))
		self._assert_compiles( emitter_c.emit_c( self.compiler ))

	def test_rcclass_field_cascading_decref_compiles( self ) -> None:
		self._run( _BOX_FIXTURE + '\n' + '\n'.join([
			'def main() -> None:',
			'	f: Foo = Foo.make( 1 )',
			'	b: Box = Box.make( f )',
			'	return',
		]))
		self._assert_compiles( emitter_c.emit_c( self.compiler ))

	def test_taggedunion_and_nested_cstruct_field_cascading_decref_compiles( self ) -> None:
		# combines both: a TaggedUnion-typed field with an RC-leaf member,
		# and a by-value CStruct field with its own nested RC field, on
		# the SAME class, torn down together
		self._run( _FOO_FIXTURE + '\n' + '\n'.join([
			'@union',
			'class MaybeFoo:',
			'	Some: Foo',
			'	Nothing: i32',
			'',
			'@cstruct',
			'class Wrapper:',
			'	inner: Foo',
			'',
			'class Box:',
			'	maybe: MaybeFoo',
			'	w: Wrapper',
			'',
			'	@staticmethod',
			'	def make( f: Foo, g: Foo ) -> Box:',
			'		return Box.__allocate__( maybe = MaybeFoo.Some( f ), w = Wrapper( inner = g ) )',
			'',
			'def main() -> None:',
			'	f: Foo = Foo.make( 1 )',
			'	g: Foo = Foo.make( 2 )',
			'	b: Box = Box.make( f, g )',
			'	return',
		]))
		self._assert_compiles( emitter_c.emit_c( self.compiler ))

@unittest.skipUnless( _CC is not None, 'no C compiler (clang or gcc) found - skipping real-compile verification' )
class RCClassRealCompileTests( _ClangCompileMixin, RCClassTestCase ):
	def test_construct_read_back_and_refcount_compiles( self ) -> None:
		# Phase 3 milestone: synthetic class Foo: x: i32 constructed, field
		# read back, compiler.refcount(x) called - compiles clean. Not
		# leak-free by design yet (release_object is called with a NULL
		# destructor - Phase 4 fills it in), only compile-clean, per the plan
		self._run( _FOO_FIXTURE + '\n' + '\n'.join([
			'def main() -> i32:',
			'	foo: Foo = Foo.make( 1 )',
			'	bar: Foo = foo',
			'	rc: usize = compiler.refcount( bar )',
			'	with compiler.wrap_arithmetic:',
			'		return bar.x',
		]))
		self._assert_compiles( emitter_c.emit_c( self.compiler ))

	def test_init_construction_compiles( self ) -> None:
		# real-compile confirmation for the sys.alloc/free/__del__ scheduling
		# fix in _try_lower_construct_call (see RCClassConstructTests'
		# identical-purpose, IR-level test) - a real __init__ (not the
		# __allocate__ field=value sugar every other fixture in this class
		# uses) must actually emit clang-clean C, not just schedule the
		# right compile units
		self._run( '\n'.join([
			'class Foo: pass',
			'class Bar:',
			'	a: Foo',
			'	def __init__( self, x: Foo ) -> None:',
			'		self.a = x',
			'',
			'def main() -> None:',
			'	b = Bar( Foo() )',
			'	return',
		]))
		self._assert_compiles( emitter_c.emit_c( self.compiler ))

	def test_generic_init_construction_compiles( self ) -> None:
		# real-compile confirmation for Stage 3b (generic RCClass __init__
		# construction) - caught a real, separate emitter_c.py bug on the
		# way here: _member_access_operator checked isinstance(obj_type,
		# RCClass) without unwrapping Specialization, so a monomorphized
		# generic method's own `self` (typed as a Specialization) emitted
		# `.` instead of `->` for every field access - self.v = v produced
		# `(self).v = v` instead of `(self)->v = v`, a real clang error
		# ("member reference type ... is a pointer; did you mean '->'?"),
		# not just a scheduling gap
		self._run( '\n'.join([
			'class Box[T]:',
			'	v: T',
			'	def __init__( self, v: T ) -> None:',
			'		self.v = v',
			'',
			'def main() -> None:',
			'	b: Box[i32] = Box( 1 )',
			'	return',
		]))
		self._assert_compiles( emitter_c.emit_c( self.compiler ))

class SizeofValueArgumentRealCompileTests( RCClassTestCase ):
	''' real compile+run coverage for compiler.sizeof(x)'s value-argument
	path (lowering.py's _static_type_of_value_expr) - unlike the IR-shape
	assertions in lowering_test.py, these confirm the generated C
	`sizeof(...)` expression actually agrees with the type-argument
	spelling at runtime, for both self and an ordinary local. '''

	def _extern_ldflags( self ) -> str:
		flags: list[str] = []
		for lib in sorted( self.compiler.extern_libs ):
			if lib == 'c':
				continue
			flags.append( linker_c.resolve_lib_ldflag( _CC, lib, self.compiler.extern_libs[lib] ) )
		return ' '.join( flags )

	def _assert_compiles_and_runs( self, c_source: str, expected_exit: int = 0 ) -> None:
		with tempfile.TemporaryDirectory() as tmp:
			src_path = Path( tmp ) / 'generated.c'
			obj_path = Path( tmp ) / 'generated.o'
			exe_path = Path( tmp ) / 'test_exe'
			src_path.write_text( c_source, encoding = 'utf-8' )
			cc_result = _CC.compile( src_path, obj_path )
			self.assertEqual( cc_result.returncode, 0,
				f'{_CC.name} compile failed:\nstdout: {cc_result.stdout}\nstderr: {cc_result.stderr}{test_support.c_source_on_failure( c_source )}' )
			ldflags = self._extern_ldflags()
			link_result = _CC.link( exe_path, [ obj_path ], ldflags = ldflags )
			self.assertEqual( link_result.returncode, 0,
				f'{_CC.name} link failed:\nstdout: {link_result.stdout}\nstderr: {link_result.stderr}' )
			run_result = subprocess.run( [ str( exe_path ) ], capture_output = True )
			self.assertEqual( run_result.returncode, expected_exit,
				f'exe exited {run_result.returncode}, expected {expected_exit}' )

	@unittest.skipUnless( _CC is not None, 'no C compiler (clang/gcc/msvc) found - skipping' )
	def test_sizeof_self_matches_sizeof_type_argument( self ) -> None:
		self._run( '\n'.join([
			'class Foo:',
			'	a: i32',
			'	b: i64',
			'',
			'	def check( self ) -> i32:',
			'		if compiler.sizeof( self ) != compiler.sizeof( Foo ):',
			'			return 1',
			'		return 0',
			'',
			'def main() -> i32:',
			'	f: Foo = Foo( a = 1, b = 2 )',
			'	return f.check()',
		]))
		self.assertEqual( self.discovery.errors.errors, [] )
		self._assert_compiles_and_runs( emitter_c.emit_c( self.compiler ))

	@unittest.skipUnless( _CC is not None, 'no C compiler (clang/gcc/msvc) found - skipping' )
	def test_sizeof_local_variable_matches_sizeof_scalar_type( self ) -> None:
		self._run( '\n'.join([
			'def main() -> i32:',
			'	v: u32 = 0',
			'	if compiler.sizeof( v ) != compiler.sizeof( u32 ):',
			'		return 1',
			'	if compiler.sizeof( v ) != 4:',
			'		return 2',
			'	return 0',
		]))
		self.assertEqual( self.discovery.errors.errors, [] )
		self._assert_compiles_and_runs( emitter_c.emit_c( self.compiler ))

# str/bytes literal static-baking (Phase 6) is detected via the REAL
# qualname 'builtins.str'/'builtins.bytes' (see the plan's grounding facts) -
# unlike every other fixture in this file, this means the fixture class
# actually needs to live under a module literally named 'builtins', not
# just __main__ (Discovery.import_name('builtins') searches self.paths the
# same way any other import does). A minimal, self-contained builtins.py
# (not the real, actively-evolving lib/builtins/__init__.py) keeps this
# scoped to what Phase 6 is actually testing, same posture as every other
# fixture here.
_BUILTINS_STR_FIXTURE = '\n'.join([
	'import sys',
	'',
	'class str:',
	'	__data: ConstPtr[u8]',
	'	__byte_size: usize',
	'',
	'	def get_data( self ) -> ConstPtr[u8]:',
	'		return self.__data',
	'',
	'	def get_len( self ) -> usize:',
	'		return self.__byte_size',
])

_SYS_FREE_ONLY_FIXTURE = '\n'.join([
	'def free( ptr: Ptr[None] ) -> None:',
	'	pass',
])

class BuiltinsStrTestCase( CompilerTestCase ):
	def setUp( self ) -> None:
		self._tmpdir = tempfile.TemporaryDirectory()
		self.addCleanup( self._tmpdir.cleanup )
		tmp_path = Path( self._tmpdir.name )
		( tmp_path / 'builtins.py' ).write_text( _BUILTINS_STR_FIXTURE, encoding = 'utf-8' )
		# the emitter always synthesizes a destructor for str (RCClass)
		# which calls sys$free — provide a minimal sys.py so it compiles
		( tmp_path / 'sys.py' ).write_text( _SYS_FREE_ONLY_FIXTURE, encoding = 'utf-8' )
		self.discovery = Discovery( paths = [ tmp_path ], import_builtins = True )
		self.compiler = Compiler( self.discovery )

class StringLiteralTests( BuiltinsStrTestCase ):
	def test_literal_baked_as_static_immortal_object( self ) -> None:
		self._run( '\n'.join([
			'def take_str( s: str ) -> None:',
			'	return',
			'',
			'def main() -> None:',
			'	take_str( \'hello\' )',
			'	return',
		]))
		self.assertEqual( self.discovery.errors.errors, [] )
		src = emitter_c.emit_c( self.compiler )
		self.assertIn( 'static const uint8_t __literal_', src )
		self.assertIn( 'static struct builtins$str __literal_', src )
		self.assertIn( '.ref_count = METALPY_IMMORTAL_REFCOUNT', src )
		self.assertIn( '"hello\\000";', src ) # 'hello' + NUL as C string literal (octal escape - see _c_string_literal's own comment on why not \x)
		self.assertIn( '.__byte_size = 6', src )
		main_lf = next( lf for lf in self.compiler.functions if lf.function.qualname == 'main' )
		main_src = emitter_c.emit_function( main_lf )
		self.assertRegex( main_src, r'__main__\$take_str\( &__literal_[0-9a-f]+ \);' )
		# a destructor is always emitted for every non-generic RCClass —
		# the emitter synthesizes one unconditionally (see emit_c.py)
		self.assertIn( '$__destructor__', src )

	def test_identical_literal_used_twice_shares_one_static_definition( self ) -> None:
		self._run( '\n'.join([
			'def take_str( s: str ) -> None:',
			'	return',
			'',
			'def main() -> None:',
			'	take_str( \'same\' )',
			'	take_str( \'same\' )',
			'	return',
		]))
		self.assertEqual( self.discovery.errors.errors, [] )
		src = emitter_c.emit_c( self.compiler )
		self.assertEqual( src.count( 'static struct builtins$str __literal_' ), 1 )

@unittest.skipUnless( _CC is not None, 'no C compiler (clang or gcc) found - skipping real-compile verification' )
class StringLiteralRealCompileTests( _ClangCompileMixin, BuiltinsStrTestCase ):
	def test_literal_passed_to_a_function_compiles( self ) -> None:
		self._run( '\n'.join([
			'def take_str( s: str ) -> None:',
			'	return',
			'',
			'def main() -> None:',
			'	take_str( \'hello\' )',
			'	return',
		]))
		self._assert_compiles( emitter_c.emit_c( self.compiler ))

	def test_non_ascii_literal_with_ambiguous_hex_escape_compiles( self ) -> None:
		# regression test: _c_string_literal used to escape non-printable
		# bytes as \xXX - but C's \x escape has NO length limit and keeps
		# consuming hex-digit CHARACTERS for as long as they appear next,
		# so a byte like 0x9F (from 'ß', UTF-8 C3 9F) immediately followed
		# by the literal, RAW-emitted printable byte 'e' (itself just a
		# hex digit character) became the single escape \x9fe (0xf9e, out
		# of uint8_t's range) instead of two separate bytes - "hex escape
		# sequence out of range" from clang/gcc on ANY non-ASCII string
		# followed by a hex-digit-looking character. Fixed by switching to
		# fixed-3-digit octal escapes (\ooo), which the C standard caps at
		# exactly 3 digits regardless of what follows - see
		# _c_string_literal's own comment.
		self._run( '\n'.join([
			'def take_str( s: str ) -> None:',
			'	return',
			'',
			'def main() -> None:',
			"	take_str( 'straße' )", # 'ß' (UTF-8 C3 9F) directly followed by the raw printable byte 'e'
			'	return',
		]))
		self._assert_compiles( emitter_c.emit_c( self.compiler ))

	def test_extern_addrof_method_call_and_literal_together_compiles( self ) -> None:
		# Phase 6 milestone: mirrors lib/sys.py's own real _Stdout.write
		# shape (a WinAPI-style extern call, an AddrOf out-param, and a
		# string literal argument) without depending on its actively-
		# evolving Result/OSError/module-global machinery (Phase 7 work) -
		# same posture as every other real-lib-shaped fixture in this file.
		# Also exercises two gaps this test surfaced and fixed along the
		# way: method calls with a real receiver (s.get_data()) and
		# compiler.cast(...)/T(x) scalar casts (u32(s.get_len())) had never
		# been implemented in the emitter at all before this.
		self._run( '\n'.join([
			"@extern( 'kernel32', 'WriteFile' )",
			'def _test_write_file(',
			'	handle: Ptr[None],',
			'	buffer: ConstPtr[u8],',
			'	count: u32,',
			'	written: Ptr[u32],',
			'	overlapped: Ptr[None],',
			') -> bool:',
			'	...',
			'',
			'def write_message( s: str ) -> bool:',
			'	handle_target: u32 = 0',
			'	written: u32 = 0',
			'	overlapped_target: u32 = 0',
			'	handle: Ptr[None] = compiler.addrof( handle_target )',
			'	overlapped: Ptr[None] = compiler.addrof( overlapped_target )',
			'	with compiler.wrap_arithmetic:',
			'		return _test_write_file( handle, s.get_data(), u32( s.get_len() ), compiler.addrof( written ), overlapped )',
			'',
			'def main() -> bool:',
			"	return write_message( 'hello' )",
		]))
		self._assert_compiles( emitter_c.emit_c( self.compiler ))

class EmitGlobalTests( CompilerTestCase ):
	def test_trivial_global_is_a_real_static_initializer( self ) -> None:
		# mirrors lib/windows/kernel32.py's own real STD_OUTPUT_HANDLE:
		# u32 = u32(-11) - a literal argument to a scalar cast always folds
		# to a bare Const at lowering time (_lower_scalar_cast never even
		# emits a CastWrap instruction for it), so the global's own
		# instruction sequence collapses to a single Assign(Const)
		self._run( '\n'.join([
			'STD_OUTPUT_HANDLE: u32 = u32( -11 )',
			'',
			'def main() -> None:',
			'	x: u32 = STD_OUTPUT_HANDLE',
			'	return',
		]))
		self.assertEqual( self.discovery.errors.errors, [] )
		g = self.compiler.globals[0]
		src = emitter_c.emit_global( g )
		self.assertEqual( src, 'uint32_t __main__$STD_OUTPUT_HANDLE = -11;' )
		self.assertNotIn( '__metalpy_init', src ) # trivial - no init function needed

class EmitGlobalRCClassTests( RCClassTestCase ):
	def test_non_trivial_global_flattens_into_an_init_function( self ) -> None:
		# mirrors lib/sys.py's own real stdout: _Stdout = _Stdout() - a
		# real RCClass construction, needing a real init function (the
		# global itself gets a {0} zero initializer in the meantime -
		# wiring the init function into a real process entry point is out
		# of scope, C_EMITTER.md excludes linking-adjacent work; it just
		# needs to exist and compile, per the plan's own milestone wording)
		self._run( _FOO_FIXTURE + '\n' + '\n'.join([
			'g_foo: Foo = Foo.make( 1 )',
			'',
			'def main() -> None:',
			'	x: Foo = g_foo',
			'	return',
		]))
		self.assertEqual( self.discovery.errors.errors, [] )
		g = self.compiler.globals[0]
		src = emitter_c.emit_global( g )
		self.assertIn( 'struct __main__$Foo* __main__$g_foo = {0};', src )
		self.assertIn( 'static void __metalpy_init___main__$g_foo( void ) {', src )
		self.assertIn( '__main__$g_foo = $t0;', src )

@unittest.skipUnless( _CC is not None, 'no C compiler (clang or gcc) found - skipping real-compile verification' )
class EmitGlobalRealCompileTests( _ClangCompileMixin, CompilerTestCase ):
	def test_trivial_global_compiles( self ) -> None:
		self._run( '\n'.join([
			'STD_OUTPUT_HANDLE: u32 = u32( -11 )',
			'',
			'def main() -> None:',
			'	x: u32 = STD_OUTPUT_HANDLE',
			'	return',
		]))
		self._assert_compiles( emitter_c.emit_c( self.compiler ))

@unittest.skipUnless( _CC is not None, 'no C compiler (clang or gcc) found - skipping real-compile verification' )
class EmitGlobalRCClassRealCompileTests( test_support.RealCompileMixin, RCClassTestCase ):
	def test_non_trivial_global_compiles_and_constructor_actually_ran( self ) -> None:
		# Phase 7 milestone: both global-initializer shapes compile clean -
		# this is the RCClass-construction shape (mirrors lib/sys.py's own
		# real stdout: _Stdout = _Stdout()), the trivial-constant shape is
		# covered by EmitGlobalRealCompileTests above.
		#
		# Upgraded from compile-only to compile-AND-RUN (PLAN_GLOBAL_INIT.md's
		# own verification section): reads g_foo.x back in main() and fails
		# unless it's exactly what Foo.make(1)'s own constructor set - real
		# proof that __metalpy_init() (which this plan wires up to call every
		# non-trivial global's own init function - see emitter_c.py's
		# emit_c()) actually ran the constructor before main()'s own body
		# executed, not just that the generated C happens to compile. Before
		# PLAN_GLOBAL_INIT.md, g_foo would have stayed a null pointer forever
		# (the C-level {0} zero-initializer, with nothing ever calling its
		# own __metalpy_init_g_foo()) - reading g_foo.x here would have
		# dereferenced a null pointer, not just returned the wrong value.
		self._run( _FOO_FIXTURE + '\n' + '\n'.join([
			'g_foo: Foo = Foo.make( 1 )',
			'',
			'def main() -> i32:',
			'	if g_foo.x != 1:',
			'		return 1',
			'	return 0',
		]))
		self.assertEqual( self.discovery.errors.errors, [] )
		self._assert_compiles_and_runs( emitter_c.emit_c( self.compiler ), expected_exit = 0 )

	def test_fieldless_subclass_global_calling_inherited_virtual_actually_constructed( self ) -> None:
		# a global whose static type is a SUBCLASS (participates in a
		# vtable via an inherited/overridden @virtual method) but has no
		# fields of its own - g's own ir.Allocate has fields={} - used to
		# be misclassified by _global_init_is_all_zero_value_type as an
		# all-zero VALUE type (which vacuously matches "every field is
		# zero" on an EMPTY fields dict) and its real sys.alloc(...)
		# construction call got skipped entirely, leaving g permanently
		# NULL - reading g.get() then dereferenced a null $header.vtable
		# and crashed (real access violation, not a plain wrong-value
		# failure). Confirmed the bug needs BOTH a subclass (a plain,
		# non-inherited RCClass global already worked) and zero fields
		# (a global with any real field already worked, since a non-zero
		# field value fails the all-zero check) - this fixture is the
		# minimal shape hitting both.
		self._run( '\n'.join([
			'class Base:',
			'	@abstractmethod',
			'	def get( self ) -> i32:',
			'		...',
			'',
			'class Derived( Base ):',
			'	@virtual',
			'	def get( self ) -> i32:',
			'		return 42',
			'',
			'g: Derived = Derived()',
			'',
			'def main() -> i32:',
			'	return g.get()',
		]))
		self.assertEqual( self.discovery.errors.errors, [] )
		self._assert_compiles_and_runs( emitter_c.emit_c( self.compiler ), expected_exit = 42 )

@unittest.skipUnless( _CC is not None, 'no C compiler (clang or gcc) found - skipping real-compile verification' )
class NoCrtExitCodeRealCompileTests( unittest.TestCase ):
	# test_support.RealCompileMixin's _build_and_run doesn't thread no_crt
	# through compile()/link() at all (it always compiles/links as if the
	# CRT were linked) - float_test.py/wide_int_test.py/return_inference_
	# test.py all hit the same gap for their own no-CRT real-compile needs
	# and duplicate this same compile+link+run shape locally rather than
	# use the mixin; mirrored here rather than inventing a third variant
	def test_no_crt_windows_exit_code_round_trips_through_sys_exit( self ) -> None:
		# proves the __result plumbing survived the ExitProcess -> sys.exit()
		# rewrite: a distinctive, non-{0,1} exit code, so this can't pass by
		# accident the way a bare 0/1 check might (0 = success fallback, 1 =
		# an uncaught panic/error - 42 is neither)
		discovery = Discovery( import_builtins = True )
		compiler = Compiler( discovery )
		compiler.import_code( '''
def main() -> i32:
	return 42
''', Path( '__main__.py' ), scope = None )
		compiler.run()
		self.assertEqual( discovery.errors.errors, [] )

		no_crt = 'c' not in compiler.extern_libs
		self.assertTrue( no_crt, 'fixture unexpectedly pulled in the CRT' )
		c_source = emitter_c.emit_c( compiler, no_crt = no_crt )

		with tempfile.TemporaryDirectory() as tmp:
			src_path = Path( tmp ) / 'generated.c'
			obj_path = Path( tmp ) / 'generated.o'
			exe_path = Path( tmp ) / 'test_exe.exe'
			src_path.write_text( c_source, encoding = 'utf-8' )

			cc_result = _CC.compile( src_path, obj_path, no_crt = no_crt )
			self.assertEqual( cc_result.returncode, 0, f'{_CC.name} compile failed:\n{cc_result.stdout}{test_support.c_source_on_failure( c_source )}' )

			ldflags = ''
			for lib in sorted( compiler.extern_libs ):
				if lib == 'c':
					continue
				flag = linker_c.resolve_lib_ldflag( _CC, lib, compiler.extern_libs[lib] )
				ldflags = ldflags + f' {flag}' if ldflags else flag

			link_result = _CC.link( exe_path, [ obj_path ], ldflags = ldflags, no_crt = no_crt )
			self.assertEqual( link_result.returncode, 0, f'{_CC.name} link failed:\n{link_result.stdout}' )

			result = subprocess.run( [ str( exe_path ) ], capture_output = True )
			self.assertEqual( result.returncode, 42, f'exe exited {result.returncode}, expected 42 (stderr: {result.stderr})' )

@unittest.skipUnless( _CC is not None, 'no C compiler (clang or gcc) found - skipping real-compile verification' )
class GlobalInitOrderingRealCompileTests( test_support.RealCompileMixin, RCClassTestCase ):
	def test_global_constructor_referencing_a_forward_declared_sibling_class( self ) -> None:
		# PLAN_GLOBAL_INIT.md flags TRUE cross-global dependency ordering
		# (global B's own initializer reading global A) as deferred/
		# unverified. This test asks a narrower, related question instead:
		# does a SINGLE global's own constructor, which itself constructs
		# two other RCClasses (Bar builds Foo1 and Foo2 inside its own
		# __init__), need any source-order help at all - specifically when
		# one of those classes (Foo2) is declared textually AFTER both the
		# class that uses it (Bar) and the global that transitively
		# constructs it (bar)?
		#
		# Foo1/Foo2 are plain (non-@interface) RCClasses built entirely
		# inside Bar.__init__, so this does NOT exercise the vtable-forward-
		# reference bug PLAN_GLOBAL_INIT.md's own landing commit fixed (that
		# one needed an @interface CStruct's own $$vtable instance). This is
		# a different, more basic question: does discovery/lowering/emission
		# care about SOURCE order among sibling classes at all. Expected to
		# already be a non-issue - discovery.py registers every top-level
		# class's NAME before resolving any class body (see Discovery's own
		# docstring: "every top-level class/function/global it directly
		# contains is registered right away ... so cross-references anywhere
		# in the program can always find each other regardless of source
		# order") - this test is the real-compile-and-run proof of that
		# claim for the specific "global variable's constructor builds
		# forward-referenced sibling classes" shape, not just an assertion
		# taken on faith.
		self._run( '\n'.join([
			'class Foo1:',
			'	x: i32',
			'	def __init__( self ) -> None:',
			'		self.x = 111',
			'',
			'class Bar:',
			'	foo1: Foo1',
			'	foo2: Foo2',
			'	def __init__( self ) -> None:',
			'		self.foo1 = Foo1()',
			'		self.foo2 = Foo2()',
			'',
			'class Foo2:',
			'	y: i32',
			'	def __init__( self ) -> None:',
			'		self.y = 222',
			'',
			'bar: Bar = Bar()',
			'',
			'def main() -> i32:',
			'	if bar.foo1.x != 111:',
			'		return 1',
			'	if bar.foo2.y != 222:',
			'		return 2',
			'	return 0',
		]))
		self.assertEqual( self.discovery.errors.errors, [] )
		self._assert_compiles_and_runs( emitter_c.emit_c( self.compiler ), expected_exit = 0 )

	def test_global_initializer_reading_another_globals_value_runs_in_dependency_order( self ) -> None:
		# a REAL, previously-reachable bug found while stress-testing
		# PLAN_GLOBAL_INIT.md's own "Deferred: true dependency-ordering
		# between globals" limitation (originally flagged as unverified,
		# not a known-good non-issue): compiler.globals is TypeResolver's
		# own FIFO scheduling order (first-referenced-while-lowering-
		# reachable-code), which has NO relationship to which global's own
		# initializer reads which OTHER global's value. Here, main() only
		# ever references `b` directly - `a` is only discovered as b's own
		# dependency, mid-way through lowering b's initializer - so a gets
		# scheduled (and therefore lowered, and therefore appended to
		# compiler.globals) strictly AFTER b, even though b's own
		# initializer reads a.x. Before _topologically_sort_globals
		# (emitter_c.py), this produced generated C that didn't even
		# COMPILE (b's own init function referenced the not-yet-declared
		# __main__$a - confirmed via a real clang -fsyntax-only run), a
		# louder failure than the null-pointer-dereference-at-runtime this
		# test's own comment originally predicted. Now fixed: every
		# global's DECLARATION is emitted before any global's own init
		# FUNCTION BODY (needs no dependency order at all - see _emit_
		# global_declaration/_emit_global_init_fn), and __metalpy_init()
		# calls each non-trivial global's own init function in REAL
		# dependency order (_topologically_sort_globals), not compiler.
		# globals' own scheduling order.
		self._run( _FOO_FIXTURE + '\n' + '\n'.join([
			'a: Foo = Foo.make( 1 )',
			'b: Foo = Foo.make( a.x )', # reads global a's value - main() never touches a directly
			'',
			'def main() -> i32:',
			'	if b.x != 1:',
			'		return 1',
			'	return 0',
		]))
		self.assertEqual( self.discovery.errors.errors, [] )
		scheduled = [ g.variable.qualname for g in self.compiler.globals if g.variable.qualname in ( '__main__.a', '__main__.b' ) ]
		self.assertEqual( scheduled, [ '__main__.b', '__main__.a' ],
			'fixture assumption broken: b should schedule before a (main only references b) - '
			'if this now fails, the scheduling order itself changed and this test no longer '
			'exercises the dependency-ordering fix at all' )
		self._assert_compiles_and_runs( emitter_c.emit_c( self.compiler ), expected_exit = 0 )

class GlobalInitCycleDetectionTests( RCClassTestCase ):
	def test_circular_global_value_dependency_is_a_clean_compile_error( self ) -> None:
		# the one shape _topologically_sort_globals can never satisfy: two
		# globals whose own initializers EACH read the other's value -
		# fundamentally unorderable, unlike a merely circular IMPORT or a
		# circular CLASS reference (both confirmed fine elsewhere - see
		# GlobalInitOrderingRealCompileTests/PLAN_GLOBAL_INIT.md). Must
		# fail with a clear, located CompileError - not hang, not crash
		# with an unrelated traceback, not silently emit some arbitrary
		# order that compiles but runs one side against an uninitialized
		# value.
		self._run( _FOO_FIXTURE + '\n' + '\n'.join([
			'a: Foo = Foo.make( b.x )',
			'b: Foo = Foo.make( a.x )',
			'',
			'def main() -> i32:', # references a so both a and (transitively, via a's own initializer) b actually get scheduled/lowered as real compile units at all
			'	if a.x != 0:',
			'		return 1',
			'	return 0',
		]))
		self.assertEqual( self.discovery.errors.errors, [] ) # the cycle itself isn't detected until emit_c - discovery/lowering never needed a full order
		with self.assertRaises( CompileError ):
			emitter_c.emit_c( self.compiler )
		self.assertTrue(
			any( 'circular global-initializer dependency' in e for e in self.discovery.errors.errors ),
			self.discovery.errors.errors,
		)

class MetalpyInitSynthesisTests( unittest.TestCase ):
	''' fast, no-C-compiler-needed checks against emit_c()'s own generated
	text - PLAN_GLOBAL_INIT.md's own original Verification section called
	for these three specific white-box assertions (single __metalpy_init
	definition, main()'s own prepend being active_target-independent,
	multiple non-trivial globals all getting called), but the landed
	implementation substituted one real compile+link+run test instead
	(EmitGlobalRCClassRealCompileTests/GlobalInitOrderingRealCompileTests) -
	stronger end-to-end evidence, but not literally what the checklist
	asked for. These fill that gap directly. Deliberately plain
	unittest.TestCase (not RCClassTestCase) - each test needs its own
	Discovery with an explicit active_target override, which the shared
	setUp doesn't support. '''

	_FIXTURE = '\n'.join([
		'class Foo:',
		'	x: i32',
		'	@staticmethod',
		'	def make( v: i32 ) -> Foo:',
		'		return Foo.__allocate__( x = v )',
		'',
		'g1: Foo = Foo.make( 1 )',
		'g2: Foo = Foo.make( 2 )',
		'',
		'def main() -> i32:',
		'	if g1.x != 1:',
		'		return 1',
		'	if g2.x != 2:',
		'		return 2',
		'	return 0',
	])

	# minimal but complete active_target dicts (see discovery.py's own
	# _detect_active_target for the real shape) - only 'os' actually
	# matters to anything this test class checks
	_WINDOWS_TARGET = { 'os': 'windows', 'arch': 'x86_64', 'family': 'windows', 'bits': 64, 'debug': True, 'posix': False }
	_LINUX_TARGET = { 'os': 'linux', 'arch': 'x86_64', 'family': 'unix', 'bits': 64, 'debug': True, 'posix': True }

	def _compiled_source( self, active_target: dict[str,object] ) -> str:
		discovery = Discovery( import_builtins = True, active_target = active_target )
		compiler = Compiler( discovery )
		compiler.import_code( self._FIXTURE, Path( '__main__.py' ), scope = None )
		compiler.run()
		self.assertEqual( discovery.errors.errors, [] )
		return emitter_c.emit_c( compiler )

	def _metalpy_init_body( self, src: str ) -> str:
		start = src.index( 'static void __metalpy_init( void ) {' )
		end = src.index( '\n}', start )
		return src[ start : end ]

	def _compiled_source_no_crt( self, active_target: dict[str,object] ) -> str:
		# separate from _compiled_source above (which always passes emit_c()'s
		# own no_crt=False default) - mainCRTStartup is only emitted when
		# no_crt=True is passed to emit_c(), so this threads the compiler's
		# own real no_crt determination through, mirroring mpy.py's own
		# 'c' not in compiler.extern_libs computation
		discovery = Discovery( import_builtins = True, active_target = active_target )
		compiler = Compiler( discovery )
		compiler.import_code( self._FIXTURE, Path( '__main__.py' ), scope = None )
		compiler.run()
		self.assertEqual( discovery.errors.errors, [] )
		no_crt = 'c' not in compiler.extern_libs
		self.assertTrue( no_crt, 'fixture unexpectedly pulled in the CRT' )
		return emitter_c.emit_c( compiler, no_crt = True )

	def test_exactly_one_metalpy_init_definition( self ) -> None:
		# the two competing #ifdef'd definitions this plan replaced
		# (emitter_c.py's old PROLOGUE) are gone - never more than one
		# `static void __metalpy_init( void ) {` anywhere in the output,
		# on either target
		for target in ( self._WINDOWS_TARGET, self._LINUX_TARGET ):
			with self.subTest( target = target[ 'os' ] ):
				src = self._compiled_source( target )
				self.assertEqual( src.count( 'static void __metalpy_init( void ) {' ), 1 )

	def test_windows_console_codepage_call_is_an_ordinary_global_init_call( self ) -> None:
		# SetConsoleOutputCP is no longer hardcoded/gated inside __metalpy_init
		# itself - it's windows/_console.py's _console_init global (forced
		# reachable on every Windows target by Compiler.run()), called from
		# here exactly like any other global's own init function
		src = self._compiled_source( self._WINDOWS_TARGET )
		body = self._metalpy_init_body( src )
		self.assertIn( '__metalpy_init_windows$_console$_console_init();', body )
		self.assertIn( 'SetConsoleOutputCP(', src )
		self.assertNotIn( '#ifdef _WIN32', body )

	def test_windows_no_crt_exit_is_an_ordinary_sys_exit_call( self ) -> None:
		# ExitProcess is no longer hand-declared/hardcoded raw C text inside
		# mainCRTStartup itself - it's sys.py's own public exit() (forced
		# reachable whenever no_crt by Compiler.force_reachable), called here
		# by its own mangled C symbol name, same shape as any other call
		src = self._compiled_source_no_crt( self._WINDOWS_TARGET )
		start = src.index( 'void mainCRTStartup( void ) {' )
		end = src.index( '\n}', start )
		body = src[ start : end ]
		self.assertIn( 'sys$exit( (uint32_t)__result );', body )
		self.assertIn( 'ExitProcess(', src ) # real @extern prototype/call, somewhere
		self.assertNotIn( 'void __stdcall ExitProcess( unsigned int );', src )

	def test_main_prepends_metalpy_init_call_on_every_target( self ) -> None:
		# not just Windows - global initializers must run everywhere now,
		# not only the Windows-specific console-codepage setup (this is
		# exactly the condition PLAN_GLOBAL_INIT.md's own implementation
		# changed from `_is_entry_point(...) and active_target['os'] ==
		# 'windows'` to a plain `_is_entry_point(...)`)
		for target in ( self._WINDOWS_TARGET, self._LINUX_TARGET ):
			with self.subTest( target = target[ 'os' ] ):
				src = self._compiled_source( target )
				main_start = src.index( 'int main( void ) {' )
				second_line = src[ main_start: ].split( '\n', 2 )[1]
				self.assertIn( '__metalpy_init();', second_line )

	def test_metalpy_init_calls_every_non_trivial_globals_init_function( self ) -> None:
		src = self._compiled_source( self._LINUX_TARGET )
		body = self._metalpy_init_body( src )
		self.assertIn( '__metalpy_init___main__$g1();', body )
		self.assertIn( '__metalpy_init___main__$g2();', body )

class WindowsTargetCTypeTests( unittest.TestCase ):
	def test_invalid_handle_value_emits_with_pointer_cast( self ) -> None:
		import ir
		from mpy_types import Scalar, Specialization
		ptr_none = Specialization(
			stem = 'Ptr[NoneType]',
			qualname = 'intrinsics.Ptr[intrinsics.NoneType]',
			file = None, line = None,
			base = Scalar( stem = 'Ptr', qualname = 'intrinsics.Ptr', file = None, line = None, sizeof = 8 ),
			args = [ Scalar( stem = 'NoneType', qualname = 'intrinsics.NoneType', file = None, line = None, sizeof = 0 ) ],
		)
		c = ir.Const( type = ptr_none, value = -1 )
		self.assertEqual( emitter_c._emit_const( c ), '(void*)-1' )

	def test_null_literal_emits_zero_not_null( self ) -> None:
		import ir
		none_type = Scalar( stem = 'NoneType', qualname = 'intrinsics.NoneType', file = None, line = None, sizeof = 0 )
		c = ir.Const( type = none_type, value = None )
		self.assertEqual( emitter_c._emit_const( c ), '0' )


@unittest.skipUnless( _CC is not None, 'no C compiler (clang or gcc) found - skipping real-compile+run verification' )
class FastLockCompileRunTests( CompilerTestCase ):
	def setUp( self ) -> None:
		self.discovery = Discovery( import_builtins = True )
		self.compiler = Compiler( self.discovery )

	def _extern_ldflags( self ) -> str:
		''' derive linker flags from self.compiler.extern_libs, matching
			mpy.py's own link step '''
		flags: list[str] = []
		for lib in sorted( self.compiler.extern_libs ):
			if lib == 'c':
				continue
			flags.append( linker_c.resolve_lib_ldflag( _CC, lib, self.compiler.extern_libs[lib] ) )
		return ' '.join( flags )

	def _assert_compiles_and_runs( self, c_source: str, expected_exit: int = 0 ) -> None:
		with tempfile.TemporaryDirectory() as tmp:
			src_path = Path( tmp ) / 'generated.c'
			obj_path = Path( tmp ) / 'generated.o'
			exe_path = Path( tmp ) / 'test_exe'
			src_path.write_text( c_source, encoding = 'utf-8' )
			cc_result = _CC.compile( src_path, obj_path )
			self.assertEqual( cc_result.returncode, 0,
				f'{_CC.name} compile failed:\nstdout: {cc_result.stdout}\nstderr: {cc_result.stderr}{test_support.c_source_on_failure( c_source )}' )
			ldflags = self._extern_ldflags()
			link_result = _CC.link( exe_path, [ obj_path ], ldflags = ldflags )
			self.assertEqual( link_result.returncode, 0,
				f'{_CC.name} link failed:\nstdout: {link_result.stdout}\nstderr: {link_result.stderr}' )
			run_result = subprocess.run( [ str( exe_path ) ], capture_output = True )
			self.assertEqual( run_result.returncode, expected_exit,
				f'exe exited {run_result.returncode}, expected {expected_exit}' )

	def test_acquire_release_and_nonblocking_reacquire( self ) -> None:
		self._run( '''
import threading

def main() -> i32:
	lock: threading.FastLock = threading.FastLock()
	# blocking acquire should succeed
	if lock.acquire().is_err():
		return 1
	# non-blocking re-acquire should fail (already locked)
	if not lock.acquire( False ).is_err():
		return 2
	# release it
	lock.release()
	# after release, non-blocking acquire should succeed again
	if lock.acquire( False ).is_err():
		return 3
	lock.release()
	return 0
''' )
		self.assertEqual( self.discovery.errors.errors, [] )
		c_source = emitter_c.emit_c( self.compiler )
		self._assert_compiles_and_runs( c_source )


class StrUpperLowerTests( CompilerTestCase ):
	def setUp( self ) -> None:
		self.discovery = Discovery( import_builtins = True )
		self.compiler = Compiler( self.discovery )

	def _extern_ldflags( self ) -> str:
		flags: list[str] = []
		for lib in sorted( self.compiler.extern_libs ):
			if lib == 'c':
				continue
			flags.append( linker_c.resolve_lib_ldflag( _CC, lib, self.compiler.extern_libs[lib] ) )
		return ' '.join( flags )

	@unittest.skipUnless( _CC is not None, 'no C compiler (clang/gcc/msvc) found - skipping' )
	def test_str_comparison_ops( self ) -> None:
		self._run( '''
def test( a: str, op: str, b: str ) -> bool:
	if op == '==':
		return a == b
	if op == '!=':
		return a != b
	if op == '<':
		return a < b
	if op == '>':
		return a > b
	if op == '<=':
		return a <= b
	if op == '>=':
		return a >= b
	return False

def main() -> i32:
	if test( 'a', '!=', 'a' ): return 1
	if test( 'a', '==', 'b' ): return 2
	if test( 'a', '>=', 'b' ): return 3
	if test( 'a', '>', 'b' ): return 4
	if test( 'b', '<=', 'a' ): return 5
	if test( 'b', '<', 'a' ): return 6
	# specifically compare unequal length strings with shared prefix:
	if not test( 'a', '!=', 'aa' ): return 7
	if test( 'a', '==', 'aa' ): return 8
	if test( 'aa', '<', 'a' ): return 9
	if test( 'aa', '<=', 'a' ): return 10
	if test( 'a', '>', 'aa' ): return 11
	if test( 'a', '>=', 'aa' ): return 12
	return 0
''' )
		c_source = emitter_c.emit_c( self.compiler )
		with tempfile.TemporaryDirectory() as tmp:
			src_path = Path( tmp ) / 'generated.c'
			obj_path = Path( tmp ) / 'generated.o'
			exe_path = Path( tmp ) / 'test_exe'
			src_path.write_text( c_source, encoding = 'utf-8' )
			cc_result = _CC.compile( src_path, obj_path )
			self.assertEqual( cc_result.returncode, 0,
				f'{_CC.name} compile failed:\nstdout: {cc_result.stdout}\nstderr: {cc_result.stderr}' )
			ldflags = self._extern_ldflags()
			link_result = _CC.link( exe_path, [ obj_path ], ldflags = ldflags )
			self.assertEqual( link_result.returncode, 0,
				f'{_CC.name} link failed:\nstdout: {link_result.stdout}\nstderr: {link_result.stderr}' )
			run_result = subprocess.run( [ str( exe_path ) ], capture_output = True )
			self.assertEqual( run_result.returncode, 0,
				f'str comparison ops failed, exit {run_result.returncode}' )

	def _assert_compiles_and_runs( self, c_source: str, expected_exit: int = 0 ) -> None:
		with tempfile.TemporaryDirectory() as tmp:
			src_path = Path( tmp ) / 'generated.c'
			obj_path = Path( tmp ) / 'generated.o'
			exe_path = Path( tmp ) / 'test_exe'
			src_path.write_text( c_source, encoding = 'utf-8' )
			cc_result = _CC.compile( src_path, obj_path )
			self.assertEqual( cc_result.returncode, 0,
				f'{_CC.name} compile failed:\nstdout: {cc_result.stdout}\nstderr: {cc_result.stderr}' )
			ldflags = self._extern_ldflags()
			link_result = _CC.link( exe_path, [ obj_path ], ldflags = ldflags )
			self.assertEqual( link_result.returncode, 0,
				f'{_CC.name} link failed:\nstdout: {link_result.stdout}\nstderr: {link_result.stderr}' )
			run_result = subprocess.run( [ str( exe_path ) ], capture_output = True )
			self.assertEqual( run_result.returncode, expected_exit,
				f'exited {run_result.returncode}, expected {expected_exit} (stderr: {run_result.stderr})' )

	@unittest.skipUnless( _CC is not None, 'no C compiler (clang/gcc/msvc) found - skipping' )
	def test_upper_lower_ascii( self ) -> None:
		self._run( '''
def main() -> i32:
	if 'hello world'.upper() != 'HELLO WORLD':
		return 1
	if 'HELLO WORLD'.lower() != 'hello world':
		return 2
	return 0
''' )
		self.assertEqual( self.discovery.errors.errors, [] )
		self._assert_compiles_and_runs( emitter_c.emit_c( self.compiler ))

	@unittest.skipUnless( _CC is not None, 'no C compiler (clang/gcc/msvc) found - skipping' )
	def test_upper_lower_simple_non_ascii_mapping( self ) -> None:
		# single-codepoint Unicode case mapping, covering accented Latin
		# and Greek - see _case_map_windows'/_case_map_posix's own comments
		# on why this is the ceiling (no ICU, no SpecialCasing.txt one-to-
		# many/context-sensitive rules - "straße".upper() staying "STRAßE"
		# rather than "STRASSE" is expected here, not a bug)
		self._run( '''
def main() -> i32:
	if 'café'.upper() != 'CAFÉ':
		return 1
	if 'CAFÉ'.lower() != 'café':
		return 2
	if 'Σίσυφος'.upper() != 'ΣΊΣΥΦΟΣ':
		return 3
	return 0
''' )
		self.assertEqual( self.discovery.errors.errors, [] )
		self._assert_compiles_and_runs( emitter_c.emit_c( self.compiler ))

	@unittest.skipUnless( _CC is not None, 'no C compiler (clang/gcc/msvc) found - skipping' )
	def test_upper_lower_empty_and_roundtrip( self ) -> None:
		self._run( '''
def main() -> i32:
	if ''.upper() != '':
		return 1
	if ''.lower() != '':
		return 2
	if 'MiXeD CaSe 123!'.upper() != 'MIXED CASE 123!':
		return 3
	if 'MiXeD CaSe 123!'.lower() != 'mixed case 123!':
		return 4
	return 0
''' )
		self.assertEqual( self.discovery.errors.errors, [] )
		self._assert_compiles_and_runs( emitter_c.emit_c( self.compiler ))


	@unittest.skipUnless( _CC is not None, 'no C compiler (clang/gcc/msvc) found - skipping' )
	def test_case_folder_dispatch_gate_before_install( self ) -> None:
		# proves the dispatch GATE itself (str.upper()/lower() checking
		# case_folder.upper_count/lower_count) independent of install() or
		# real Unicode data: before any mutation, both counts are 0 (a
		# cstruct global's own C {0} static zero-init - see CaseFolding's
		# own comment on why it's a cstruct, not a class), so str.upper()
		# takes the OS-native path. Pointing upper_table/upper_count at a
		# tiny hand-built one-entry table (no case_folding.py, no network)
		# makes str.upper() dispatch through CaseFolding.upper()'s real
		# binary-search lookup instead, with no change at the .upper() call
		# site itself - and confirms an unmapped codepoint passes through
		# unchanged (the "not every codepoint has an entry" path)
		self._run( '''
import builtins
import compiler
import sys

def main() -> i32:
	before: str = 'ab'.upper()
	if before != 'AB':
		return 1

	# one entry: 'a' (0x61) -> 'Z' (0x5A), sorted, 8 bytes (u32 LE codepoint + u32 LE mapped)
	table: Ptr[u8] = sys.alloc[u8]( 8 )
	table[0] = 0x61
	table[1] = 0
	table[2] = 0
	table[3] = 0
	table[4] = 0x5A
	table[5] = 0
	table[6] = 0
	table[7] = 0
	builtins.case_folder.upper_table = table
	builtins.case_folder.upper_count = 1

	after: str = 'ab'.upper()
	if after != 'Zb': # 'a' hits the table entry, 'b' has no entry and passes through unchanged
		return 2
	return 0
''' )
		self.assertEqual( self.discovery.errors.errors, [] )
		self._assert_compiles_and_runs( emitter_c.emit_c( self.compiler ))

	@unittest.skipUnless( _CC is not None, 'no C compiler (clang/gcc/msvc) found - skipping' )
	def test_case_folding_install_real_unicode_table( self ) -> None:
		# end-to-end with the REAL case_folding.py module: install() fetches
		# (or reuses the disk cache for) the actual UnicodeData.txt and
		# populates case_folder's tables for real - see PLAN_CASE_FOLDING.md's
		# explicit "build-time download, no version pinning" decision, and
		# lib/case_folding.py's own install(). Assertions stick to long-
		# stable simple mappings only (plain ASCII, and one well-known
		# accented Latin letter), never a hardcoded exhaustive table, so a
		# newer UCD release can't break this test - see
		# FetchUnicodeTableTests in lowering_test.py for the structural
		# (sortedness/no-duplicates/valid-range) invariants that DO get
		# checked against the real fetched data.
		self._run( '''
import case_folding

def main() -> i32:
	if 'hello'.upper() != 'HELLO':
		return 1
	if 'HELLO'.lower() != 'hello':
		return 2

	case_folding.install()

	if 'hello'.upper() != 'HELLO':
		return 3
	if 'HELLO'.lower() != 'hello':
		return 4
	if 'café'.upper() != 'CAFÉ':
		return 5
	if 'CAFÉ'.lower() != 'café':
		return 6
	return 0
''' )
		self.assertEqual( self.discovery.errors.errors, [] )
		self._assert_compiles_and_runs( emitter_c.emit_c( self.compiler ))

class InterfaceCStructLayoutTests( CompilerTestCase ):
	''' @interface CStruct layout - see PLAN_SUBCLASSING_VTABLES_COM.md's
	Phase 1 (type model + layout, no dispatch yet). Needs import_builtins
	(string literals in error messages, sys.alloc's own body) the same way
	StrUpperLowerTests above does. '''

	def setUp( self ) -> None:
		self.discovery = Discovery( import_builtins = True )
		self.compiler = Compiler( self.discovery )

	def _extern_ldflags( self ) -> str:
		flags: list[str] = []
		for lib in sorted( self.compiler.extern_libs ):
			if lib == 'c':
				continue
			flags.append( linker_c.resolve_lib_ldflag( _CC, lib, self.compiler.extern_libs[lib] ) )
		return ' '.join( flags )

	def _assert_compiles_and_runs( self, c_source: str, expected_exit: int = 0 ) -> None:
		with tempfile.TemporaryDirectory() as tmp:
			src_path = Path( tmp ) / 'generated.c'
			obj_path = Path( tmp ) / 'generated.o'
			exe_path = Path( tmp ) / 'test_exe'
			src_path.write_text( c_source, encoding = 'utf-8' )
			cc_result = _CC.compile( src_path, obj_path )
			self.assertEqual( cc_result.returncode, 0,
				f'{_CC.name} compile failed:\nstdout: {cc_result.stdout}\nstderr: {cc_result.stderr}{test_support.c_source_on_failure( c_source )}' )
			ldflags = self._extern_ldflags()
			link_result = _CC.link( exe_path, [ obj_path ], ldflags = ldflags )
			self.assertEqual( link_result.returncode, 0,
				f'{_CC.name} link failed:\nstdout: {link_result.stdout}\nstderr: {link_result.stderr}' )
			run_result = subprocess.run( [ str( exe_path ) ], capture_output = True )
			self.assertEqual( run_result.returncode, expected_exit,
				f'exited {run_result.returncode}, expected {expected_exit} (stderr: {run_result.stderr})' )

	def test_vtable_typedef_and_base_chain_flattening( self ) -> None:
		# matches the plan doc's own Layout worked example: a root
		# interface's $vtable is its only member; a subclass inherits the
		# SAME $vtable field type (never its own FooImplVtbl) at the same
		# first-member position, with its own fields appended after.
		# self is Ptr[T] for every @interface method (consistency, per the
		# plan doc's revised decision), so the vtable slot's own self type
		# is a plain (non-const) pointer too.
		self._run( '''
@interface
class IFoo:
	@virtual
	def helper( self, n: i32 ) -> i32: ...

@interface
class FooImpl( IFoo ):
	x: i32
	y: i32

	@virtual
	def helper( self, n: i32 ) -> i32:
		return n

def main() -> None:
	f: Ptr[FooImpl] = FooImpl( x = 1, y = 2 )
	return
''' )
		self.assertEqual( self.discovery.errors.errors, [] )
		src = emitter_c.emit_c( self.compiler )
		self.assertIn(
			'typedef struct __main__$IFooVtbl {\n'
			'\tint32_t (*helper)( struct __main__$IFoo* self, int32_t n );\n'
			'} __main__$IFooVtbl;',
			src,
		)
		self.assertIn(
			'struct __main__$IFoo {\n\tconst __main__$IFooVtbl* $vtable;\n};',
			src,
		)
		self.assertIn(
			'struct __main__$FooImpl {\n\tconst __main__$IFooVtbl* $vtable;\n\tint32_t x;\n\tint32_t y;\n};',
			src,
		)

	@unittest.skipUnless( _CC is not None, 'no C compiler (clang/gcc/msvc) found - skipping' )
	def test_inherited_method_callable_through_subclass_instance( self ) -> None:
		# base-chain lookup (CStruct.chain_lookup) - a subclass instance can
		# call a method it never redeclared, found by walking to its base.
		# self is Ptr[T] uniformly now, so an inherited call is just a
		# plain pointer-to-pointer cast (Ptr[FooImpl] -> Ptr[IFoo]) at the
		# call site - see _emit_self_operand in emitter_c.py.
		self._run( '''
def main() -> i32:
	f: Ptr[FooImpl] = FooImpl( x = 1 )
	with compiler.wrap_arithmetic:
		result: i32 = f.helper() - 42
	return result

@interface
class IFoo:
	def helper( self ) -> i32:
		return 42

@interface
class FooImpl( IFoo ):
	x: i32
''' )
		self.assertEqual( self.discovery.errors.errors, [] )
		call = next( lf for lf in self.compiler.functions if lf.function.qualname == 'main' )
		target_call = next( instr for instr in call.instructions if isinstance( instr, ir.Call ))
		self.assertEqual( target_call.target.qualname, '__main__.IFoo.helper' )
		self._assert_compiles_and_runs( emitter_c.emit_c( self.compiler ))

	@unittest.skipUnless( _CC is not None, 'no C compiler (clang/gcc/msvc) found - skipping' )
	def test_ptr_interface_cstruct_field_and_method_access( self ) -> None:
		# the Ptr[T]/ConstPtr[T] dot-operator (p.attr / p.method() means
		# arrow, redirecting name lookup to the pointee - see
		# lowering.py's _attr_lookup/_attr_lookup_callable) makes field
		# assignment and method calls through a Ptr[FooImpl] read like
		# ordinary attribute access despite f being a genuine pointer
		self._run( '''
@interface
class IFoo:
	def do_thing( self, n: i32 ) -> i32: ...

@interface
class FooImpl( IFoo ):
	x: i32

	def do_thing( self, n: i32 ) -> i32:
		with compiler.wrap_arithmetic:
			result: i32 = self.x + n
		return result

def main() -> i32:
	f: Ptr[FooImpl] = FooImpl( x = 5 )
	f.x = 7
	result: i32 = f.do_thing( 1 )
	with compiler.wrap_arithmetic:
		diff: i32 = result - 8
	return diff
''' )
		self.assertEqual( self.discovery.errors.errors, [] )
		src = emitter_c.emit_c( self.compiler )
		self.assertIn( '(f)->x = 7;', src ) # real arrow write-through, not a copy-mutate-discard
		self._assert_compiles_and_runs( emitter_c.emit_c( self.compiler ))

	@unittest.skipUnless( _CC is not None, 'no C compiler (clang/gcc/msvc) found - skipping' )
	def test_ptr_dot_operator_write_through_on_plain_cstruct( self ) -> None:
		# the Ptr[T]/ConstPtr[T] dot-operator is general - not specific to
		# @interface CStructs - so this ALSO fixes the pre-existing
		# Ptr[T][idx].field = value write-through bug (found while building
		# Phase 1/2) for the p.field spelling: `p.x = 5` now emits a real
		# `(p)->x = 5;` and writes through. The `p[0].x = 5` spelling is a
		# separate case (see test_ptr_index_field_write_through_nonzero_index
		# below, fixed via a read-modify-write GetItem/SetAttr/SetItem)
		self._run( '''
import sys

@cstruct
class Point:
	x: i32
	y: i32

def main() -> i32:
	p: Ptr[Point] = sys.alloc[Point]( 1 )
	p.x = 5
	p.y = 6
	with compiler.wrap_arithmetic:
		diff: i32 = ( p.x - 5 ) + ( p.y - 6 )
	sys.free( p )
	return diff
''' )
		self.assertEqual( self.discovery.errors.errors, [] )
		src = emitter_c.emit_c( self.compiler )
		self.assertIn( '(p)->x = 5;', src )
		self.assertIn( '(p)->y = 6;', src )
		self._assert_compiles_and_runs( emitter_c.emit_c( self.compiler ))

	@unittest.skipUnless( _CC is not None, 'no C compiler (clang/gcc/msvc) found - skipping' )
	def test_ptr_index_field_write_through_nonzero_index( self ) -> None:
		# Ptr[T][idx].field = value used to silently drop the write for any
		# index (including 0): _expr_Subscript's raw-pointer fallback loads
		# *(p + idx) into a VALUE COPY temp, and the old code then mutated
		# and discarded that copy. Fixed via _lower_attr_target_obj, which
		# does a read (GetItem), mutate (the ordinary SetAttr path,
		# unchanged), write-back (SetItem) - evaluating the pointer/index
		# expressions exactly once, same double-evaluation concern as the
		# AugAssign restriction elsewhere in this file.
		self._run( '''
import sys

@cstruct
class Point:
	x: i32
	y: i32

def main() -> i32:
	p: Ptr[Point] = sys.alloc[Point]( 3 )
	p[1].x = 5
	p[1].y = 6
	result: i32 = p[1].x
	with compiler.wrap_arithmetic:
		diff: i32 = ( result - 5 ) + ( p[1].y - 6 )
	sys.free( p )
	return diff
''' )
		self.assertEqual( self.discovery.errors.errors, [] )
		src = emitter_c.emit_c( self.compiler )
		self.assertIn( '(p)[((uintptr_t)1ULL)] = $t1;', src ) # real write-back, not a copy-mutate-discard
		self._assert_compiles_and_runs( emitter_c.emit_c( self.compiler ))

	@unittest.skipUnless( _CC is not None, 'no C compiler (clang/gcc/msvc) found - skipping' )
	def test_virtual_dispatch_calls_the_override_not_the_stub( self ) -> None:
		# real vtable dispatch, not a direct call - IFoo's own get_value is
		# an unfulfilled stub (never emitted as a real C function - see
		# lowering.py's construction-time check, which would reject
		# constructing a bare IFoo directly); FooImpl's own override is
		# what the vtable slot actually points at, via a direct function-
		# pointer cast (no trampoline needed now that self is uniformly
		# Ptr[T] - see emit_interface_vtable_instance). Calling through
		# `f.get_value()` goes through `(f)->$vtable->get_value(...)`, not
		# a direct call to either function by name - if dispatch were
		# silently reverting to a direct static call, this would still
		# pass (the static target IS the right override already), so the
		# real assertion is the generated C shape below, not just the
		# exit code.
		self._run( '''
@interface
class IFoo:
	@virtual
	def get_value( self ) -> i32: ...

@interface
class FooImpl( IFoo ):
	x: i32

	@virtual
	def get_value( self ) -> i32:
		return self.x

def main() -> i32:
	f: Ptr[FooImpl] = FooImpl( x = 99 )
	direct: i32 = f.get_value()
	with compiler.wrap_arithmetic:
		diff: i32 = direct - 99
	return diff
''' )
		self.assertEqual( self.discovery.errors.errors, [] )
		src = emitter_c.emit_c( self.compiler )
		self.assertIn( '(f)->$vtable->get_value( (struct __main__$IFoo*)(f) )', src )
		self.assertIn(
			'static const __main__$IFooVtbl __main__$FooImpl$$vtable = '
			'{ .get_value = (int32_t (*)( struct __main__$IFoo* self ))__main__$FooImpl$get_value };',
			src,
		)
		self.assertIn( '$vtable = &__main__$FooImpl$$vtable;', src )
		self._assert_compiles_and_runs( emitter_c.emit_c( self.compiler ))

	def test_construct_unfulfilled_interface_is_a_compile_error( self ) -> None:
		# a "pure interface" (any @interface class with an unfulfilled
		# @virtual slot anywhere in its own chain) is never meant to be
		# constructed directly - see lowering.py's _lower_allocate_fields
		self._run( '''
@interface
class IFoo:
	@virtual
	def get_value( self ) -> i32: ...

def main() -> None:
	f: Ptr[IFoo] = IFoo()
	return
''' )
		self.assertIn( 'cannot be constructed', self.discovery.errors.errors[0] )

	def test_new_virtual_slot_below_root_gets_its_own_vtbl_type( self ) -> None:
		# REVISION: any level can introduce new @virtual slots now, not
		# just the root - real COM interface hierarchies routinely add
		# methods at every level (IUnknown -> ICustom (adds methods) ->
		# ConcreteImpl), which the original root-only-introduces-slots
		# rule could never express. FooImpl adds a NEW slot (helper) on
		# top of what it inherits from IFoo (get_value) - FooImpl becomes
		# its own vtbl_owner(), with its own FooImplVtbl type (a superset
		# of IFooVtbl: get_value first, then FooImpl's own new helper).
		self._run( '''
@interface
class IFoo:
	@virtual
	def get_value( self ) -> i32: ...

@interface
class FooImpl( IFoo ):
	@virtual
	def get_value( self ) -> i32:
		return 1

	@virtual
	def helper( self ) -> i32:
		return 2

def main() -> None:
	f: Ptr[FooImpl] = FooImpl()
	return
''' )
		self.assertEqual( self.discovery.errors.errors, [] )
		src = emitter_c.emit_c( self.compiler )
		self.assertIn(
			'typedef struct __main__$FooImplVtbl {\n'
			'\tint32_t (*get_value)( struct __main__$FooImpl* self );\n'
			'\tint32_t (*helper)( struct __main__$FooImpl* self );\n'
			'} __main__$FooImplVtbl;',
			src,
		)
		self.assertIn( 'const __main__$FooImplVtbl* $vtable;', src )

	@unittest.skipUnless( _CC is not None, 'no C compiler (clang/gcc/msvc) found - skipping' )
	def test_three_level_hierarchy_dispatches_through_the_right_vtbl_type( self ) -> None:
		# the real COM pattern this whole revision exists for: IUnknown-
		# shaped root -> ICustom (adds a method, becomes its own
		# vtbl_owner) -> ConcreteImpl (adds nothing new, reuses ICustom's
		# own Vtbl type unchanged). Dispatch through a ConcreteImpl-typed
		# receiver for a method ICustom declared (not ConcreteImpl) has to
		# use ConcreteImpl's own vtbl_owner() (ICustom) for the self cast,
		# not IRoot and not ConcreteImpl itself - see emitter_c.py's
		# _emit_self_operand.
		self._run( '''
@interface
class IRoot:
	@virtual
	def base_method( self ) -> i32: ...

@interface
class ICustom( IRoot ):
	@virtual
	def custom_method( self ) -> i32: ...

@interface
class ConcreteImpl( ICustom ):
	@virtual
	def base_method( self ) -> i32:
		return 10

	@virtual
	def custom_method( self ) -> i32:
		return 20

def main() -> i32:
	c: Ptr[ConcreteImpl] = ConcreteImpl()
	a: i32 = c.base_method()
	b: i32 = c.custom_method()
	with compiler.wrap_arithmetic:
		diff: i32 = ( a - 10 ) + ( b - 20 )
	return diff
''' )
		self.assertEqual( self.discovery.errors.errors, [] )
		self._assert_compiles_and_runs( emitter_c.emit_c( self.compiler ))

	def test_override_signature_mismatch_is_a_compile_error( self ) -> None:
		self._run( '''
@interface
class IFoo:
	@virtual
	def get_value( self, n: i32 ) -> i32: ...

@interface
class FooImpl( IFoo ):
	@virtual
	def get_value( self, n: i64 ) -> i32:
		return 1

def main() -> None:
	f: Ptr[FooImpl] = FooImpl()
	return
''' )
		self.assertIn( 'does not match', self.discovery.errors.errors[0] )


class ListGenericTests( test_support.RealCompileMixin, CompilerTestCase ):
	''' list[T] (lib/builtins/__list.py) end-to-end, for both a value type
	(i32) and an RC type (str). A plain contiguous order-preserving array -
	real Python-list semantics, positions ARE the index, insert/erase shift
	via memmove. See FastListGenericTests below for the OTHER container
	(FastList[T]) that trades order for O(1) erase + stable IDs. Mirrors
	InterfaceCStructLayoutTests/StrUpperLowerTests' own
	import_builtins=True + real compile-and-run convention (list[T] needs
	str/Result/the rest of builtins for real). '''

	def setUp( self ) -> None:
		self.discovery = Discovery( import_builtins = True )
		self.compiler = Compiler( self.discovery )

	@unittest.skipUnless( test_support.HAS_CC, 'no C compiler (clang/gcc/msvc) found - skipping' )
	def test_programs_compile_and_run( self ) -> None:
		''' every real compile-and-run program in this class, merged into a
		single executable (one build for the whole class); a nonzero exit is
		decoded back to the failing sub-program and its own return code. '''
		self.assert_programs_run([
			# a non-RC element type: list[i32]() construction/destruction alone
			# (x never used past declaration) already exercises RawList's own
			# alloc/free and list[T].__del__'s decref-skip loop; append/
			# __getitem__ round-trip three values through the buffer, by
			# POSITION (an index IS a position now - no separate stable ID)
			( 'list_i32_construct_append_getitem_del', '''
def main() -> i32:
	x: list[i32] = list[i32]()
	r0: Result[None,OverflowError] = x.append( 10 )
	r1: Result[None,OverflowError] = x.append( 20 )
	r2: Result[None,OverflowError] = x.append( 30 )
	if r0.is_err() or r1.is_err() or r2.is_err():
		return 9
	if x.__len__() != 3:
		return 1
	g0: Result[i32,IndexError] = x.__getitem__( 0 )
	g1: Result[i32,IndexError] = x.__getitem__( 1 )
	g2: Result[i32,IndexError] = x.__getitem__( 2 )
	if g0.is_err() or g1.is_err() or g2.is_err():
		return 8
	if g0.unwrap( 'getitem failed' ) != 10:
		return 2
	if g1.unwrap( 'getitem failed' ) != 20:
		return 3
	if g2.unwrap( 'getitem failed' ) != 30:
		return 4
	return 0
''' ),
			# regression test for a real double-Decref, found while reverting
			# an int.py workaround (see __int.py's own divmod() comment): a
			# real (RC-typed, not i32) element read back via
			# `x.__getitem__(i).unwrap(msg)` chained directly - the receiver
			# Result[T,IndexError] never bound to a name - used to free the
			# element while x itself still referenced it. unwrap()'s own
			# declared body (`return self.data.v_Ok`) never increfs; the
			# receiver's own pending cleanup (registered the moment its owning
			# Call was emitted - see cfg.py's fresh_temp) was never cancelled
			# to reflect that its one real reference now backs the .unwrap()
			# return value instead, so BOTH the receiver's own cleanup and the
			# destination binding's own future decref tried to release it -
			# confirmed with AddressSanitizer, not merely by this passing.
			# int(5), not i32, matters: element_size shortcuts for a non-RC T
			# never exercised the buggy path at all - see list.__init__'s own
			# is_rc(T) branch. Fixed in lowering.py's _lower_call (the new
			# Temp-receiver branch for unwrap()/unwrap_or(), alongside the
			# existing Variable-receiver one).
			( 'list_rc_element_getitem_unwrap_chained_on_bare_receiver', '''
def main() -> i32:
	x: list[int] = list[int]()
	x.append( int( 5 )).unwrap( 'append failed' )
	got: int = x.__getitem__( 0 ).unwrap( 'getitem failed' )
	if got != int( 5 ):
		return 1
	return 0
''' ),
			# initial_capacity defaults to 8 - 20 appends forces RawList._grow()
			# at least once, and every value must still read back correctly
			# afterward (the copy during growth must preserve contents)
			( 'list_i32_grows_past_initial_capacity', '''
def main() -> i32:
	x: list[i32] = list[i32]()
	i: usize = 0
	with compiler.panic_arithmetic( 'overflow' ):
		while i < 20:
			ar: Result[None,OverflowError] = x.append( compiler.cast( i32, i ))
			if ar.is_err():
				return 9
			i += 1
	if x.__len__() != 20:
		return 1
	j: usize = 0
	with compiler.panic_arithmetic( 'overflow' ):
		while j < 20:
			gr: Result[i32,IndexError] = x.__getitem__( j )
			if gr.is_err():
				return 8
			v: i32 = gr.unwrap( 'getitem failed' )
			if v != compiler.cast( i32, j ):
				return 2
			j += 1
	return 0
''' ),
			# the whole point of this container vs. FastList[T]: erase_at
			# shifts everything after the removed slot left by one (memmove),
			# it does not swap the last element into the gap. Append
			# 10,20,30,40,50, erase_at(2) (the value 30) - must read back
			# 10,20,40,50, never 10,20,50,40 (that shape would mean this
			# regressed to FastList's swap-and-pop behavior)
			( 'erase_at_preserves_positional_order', '''
def main() -> i32:
	x: list[i32] = list[i32]()
	r0: Result[None,OverflowError] = x.append( 10 )
	r1: Result[None,OverflowError] = x.append( 20 )
	r2: Result[None,OverflowError] = x.append( 30 )
	r3: Result[None,OverflowError] = x.append( 40 )
	r4: Result[None,OverflowError] = x.append( 50 )
	if r0.is_err() or r1.is_err() or r2.is_err() or r3.is_err() or r4.is_err():
		return 9
	er: Result[None,IndexError] = x.erase_at( 2 )
	if er.is_err():
		return 8
	if x.__len__() != 4:
		return 1
	g0: Result[i32,IndexError] = x.__getitem__( 0 )
	g1: Result[i32,IndexError] = x.__getitem__( 1 )
	g2: Result[i32,IndexError] = x.__getitem__( 2 )
	g3: Result[i32,IndexError] = x.__getitem__( 3 )
	if g0.is_err() or g1.is_err() or g2.is_err() or g3.is_err():
		return 7
	v0: i32 = g0.unwrap( 'x' )
	v1: i32 = g1.unwrap( 'x' )
	v2: i32 = g2.unwrap( 'x' )
	v3: i32 = g3.unwrap( 'x' )
	if v0 == 10 and v1 == 20 and v2 == 40 and v3 == 50:
		return 0 # order preserved - everything after the gap shifted left
	return 99
''' ),
			# the mirror image of erase_at above: insert(1, 99) into
			# [10,20,30] must produce [10,99,20,30], not overwrite or corrupt
			# anything - everything at/after the insertion point shifts right
			( 'insert_shifts_tail_right_and_preserves_order', '''
def main() -> i32:
	x: list[i32] = list[i32]()
	r0: Result[None,OverflowError] = x.append( 10 )
	r1: Result[None,OverflowError] = x.append( 20 )
	r2: Result[None,OverflowError] = x.append( 30 )
	if r0.is_err() or r1.is_err() or r2.is_err():
		return 9
	ir: Result[None,OverflowError] = x.insert( 1, 99 )
	if ir.is_err():
		return 8
	if x.__len__() != 4:
		return 1
	g0: Result[i32,IndexError] = x.__getitem__( 0 )
	g1: Result[i32,IndexError] = x.__getitem__( 1 )
	g2: Result[i32,IndexError] = x.__getitem__( 2 )
	g3: Result[i32,IndexError] = x.__getitem__( 3 )
	if g0.is_err() or g1.is_err() or g2.is_err() or g3.is_err():
		return 7
	v0: i32 = g0.unwrap( 'x' )
	v1: i32 = g1.unwrap( 'x' )
	v2: i32 = g2.unwrap( 'x' )
	v3: i32 = g3.unwrap( 'x' )
	if v0 == 10 and v1 == 99 and v2 == 20 and v3 == 30:
		return 0
	return 99
''' ),
			# matches Python's own list.insert - an out-of-range index doesn't
			# error, it just appends
			( 'insert_past_end_clamps_to_append', '''
def main() -> i32:
	x: list[i32] = list[i32]()
	r0: Result[None,OverflowError] = x.append( 10 )
	r1: Result[None,OverflowError] = x.append( 20 )
	if r0.is_err() or r1.is_err():
		return 9
	ir: Result[None,OverflowError] = x.insert( 100, 30 )
	if ir.is_err():
		return 8
	if x.__len__() != 3:
		return 1
	g2: Result[i32,IndexError] = x.__getitem__( 2 )
	if g2.is_err():
		return 7
	if g2.unwrap( 'x' ) != 30:
		return 2
	return 0
''' ),
			# exercises x[i] = v as real assignment syntax (not
			# .__setitem__(...) called directly) - this only actually reaches
			# list[T].__setitem__ because of lowering.py's own dispatch fix
			# (obj[i] = v used to always emit a raw SetItem, ignoring any real
			# __setitem__ the type declared)
			( 'setitem_via_assignment_syntax_overwrites_in_place', '''
def set_it( x: list[i32] ) -> Result[None,IndexError]:
	x[1] = 99
	return Result.Ok( None )

def main() -> i32:
	x: list[i32] = list[i32]()
	r0: Result[None,OverflowError] = x.append( 10 )
	r1: Result[None,OverflowError] = x.append( 20 )
	r2: Result[None,OverflowError] = x.append( 30 )
	if r0.is_err() or r1.is_err() or r2.is_err():
		return 9
	sr: Result[None,IndexError] = set_it( x )
	if sr.is_err():
		return 7
	g0: Result[i32,IndexError] = x.__getitem__( 0 )
	g1: Result[i32,IndexError] = x.__getitem__( 1 )
	g2: Result[i32,IndexError] = x.__getitem__( 2 )
	if g0.is_err() or g1.is_err() or g2.is_err():
		return 8
	if g0.unwrap( 'x' ) == 10 and g1.unwrap( 'x' ) == 99 and g2.unwrap( 'x' ) == 30:
		return 0
	return 99
''' ),
			# an RC element type - a list[T] slot holds str's own HANDLE
			# (pointer-width), not its struct body (see list.__init__'s own
			# comment); __del__ must decref every stored element without
			# reading struct-body-sized memory out of a pointer-sized slot
			( 'list_str_construct_append_getitem_del', '''
def main() -> i32:
	x: list[str] = list[str]()
	r0: Result[None,OverflowError] = x.append( 'hello' )
	r1: Result[None,OverflowError] = x.append( 'world' )
	if r0.is_err() or r1.is_err():
		return 9
	if x.__len__() != 2:
		return 1
	g0: Result[str,IndexError] = x.__getitem__( 0 )
	g1: Result[str,IndexError] = x.__getitem__( 1 )
	if g0.is_err() or g1.is_err():
		return 8
	if g0.unwrap( 'getitem failed' ) != 'hello':
		return 2
	if g1.unwrap( 'getitem failed' ) != 'world':
		return 3
	return 0
''' ),
			( 'list_str_grows_past_initial_capacity', '''
def main() -> i32:
	x: list[str] = list[str]()
	i: usize = 0
	with compiler.panic_arithmetic( 'overflow' ):
		while i < 20:
			ar: Result[None,OverflowError] = x.append( 'item' )
			if ar.is_err():
				return 9
			i += 1
	if x.__len__() != 20:
		return 1
	all_ok: bool = True
	j: usize = 0
	with compiler.panic_arithmetic( 'overflow' ):
		while j < 20:
			gr: Result[str,IndexError] = x.__getitem__( j )
			if gr.is_err():
				all_ok = False
			else:
				v: str = gr.unwrap( 'getitem failed' )
				if v != 'item':
					all_ok = False
			j += 1
	if not all_ok:
		return 2
	return 0
''' ),
			# regression test for a real compiler bug found while building
			# dict[K,V] (see type_resolver.py's own _schedule_rcclass_
			# destructor_deps fix): list[T] used as a FIELD of another class
			# (as opposed to a local variable) used to crash - schedule()'s own
			# Specialization branch called _schedule_rcclass_destructor_deps
			# with the BARE, unspecialized `list` class, which then scheduled
			# list's own __del__ directly for compilation with T still an
			# unbound TypeVar ("compiler.is_rc(T) requires a concrete type").
			# Whether this actually crashed depended on resolution ordering -
			# a plain local `x: list[i32] = list[i32]()` never triggered it,
			# only a FIELD assignment (self.items = list[i32]()) reliably did
			( 'list_as_class_field_constructs_and_destructs', '''
class Holder:
	items: list[i32]

	def __init__( self ) -> None:
		self.items = list[i32]()

	def add( self, v: i32 ) -> None:
		r: Result[None,OverflowError] = self.items.append( v )
		if r.is_err():
			sys.panic( 'append failed' )

def main() -> i32:
	h: Holder = Holder()
	h.add( 10 )
	h.add( 20 )
	if h.items.__len__() != 2:
		return 1
	g0: Result[i32,IndexError] = h.items.__getitem__( 0 )
	g1: Result[i32,IndexError] = h.items.__getitem__( 1 )
	if g0.is_err() or g1.is_err():
		return 2
	if g0.unwrap( 'x' ) == 10 and g1.unwrap( 'x' ) == 20:
		return 0
	return 99
''' ),
			# regression test for a real double-Decref, found while building
			# int.divmod() (see __int.py's own divmod() comment): list.__del__
			# reads each element into a named local and manually
			# compiler.decref()s it, but cfg.py never learned that decref
			# already released it, so the local's own scope-exit epilogue
			# decref'd it a SECOND time - heap-use-after-free, confirmed with
			# AddressSanitizer (Windows raw HeapAlloc/HeapFree corruption isn't
			# ASan-visible directly; diagnosed by shimming HeapAlloc/HeapFree
			# onto malloc/free for one instrumented build). NOT a
			# type_resolver.py/_schedule_rcclass_destructor_deps issue (an
			# earlier, incorrect theory this comment used to describe) - that
			# function was already correct. Fixed in cfg.py's
			# manually_decreffed(), called from lowering.py's
			# _lower_compiler_decref. This particular repro's 3-field/list[T]
			# shape doesn't bear on the bug itself (the double-decref happens
			# for ANY compiler.decref'd named local, any field count) - it's
			# just the shape that happened to be large enough to make the OS
			# heap allocator's own corruption detection fire reliably; smaller
			# objects can silently corrupt the heap without an immediate crash,
			# so passing here is necessary but not sufficient - see this
			# session's own investigation notes for the direct-PowerShell-
			# execution proof this file's own _assert_compiles_and_runs can't
			# fully replace (subprocess.run from a process tree rooted in a
			# git-bash/MSYS shell was observed to silently swallow this exact
			# STATUS_HEAP_CORRUPTION rather than propagate it as a nonzero
			# exit code, in this project's actual dev environment)
			( 'list_local_var_of_multi_field_rcclass', '''
class Triple:
	a: usize
	b: usize
	c: usize

	def __init__( self, v: usize = 0 ) -> None:
		self.a = v
		self.b = v
		self.c = v

def main() -> i32:
	xs: list[Triple] = list[Triple]()
	r1: Result[None,OverflowError] = xs.append( Triple( 5 ))
	if r1.is_err():
		return 1
	g1: Result[Triple,IndexError] = xs.__getitem__( 0 )
	if g1.is_err():
		return 2
	got: Triple = g1.unwrap( 'x' )
	if got.a != 5 or got.b != 5 or got.c != 5:
		return 3
	return 0
''' ),
			# same as test_list_local_var_of_multi_field_rcclass above, but for
			# FastList[T] - a separate implementation (lib/builtins/__fastlist.py)
			# with the identical `val: T = <read>; compiler.decref(val)` shape in
			# its own __del__, so it reproduced the identical double-Decref bug
			# for the identical reason (see the other test's own updated comment
			# - cfg.py's manually_decreffed(), not anything list/FastList-specific)
			( 'fastlist_local_var_of_multi_field_rcclass', '''
class Triple:
	a: usize
	b: usize
	c: usize

	def __init__( self, v: usize = 0 ) -> None:
		self.a = v
		self.b = v
		self.c = v

def main() -> i32:
	xs: FastList[Triple] = FastList[Triple]()
	r1: Result[usize,OverflowError] = xs.append( Triple( 7 ))
	if r1.is_err():
		return 1
	g1: Result[Triple,IndexError] = xs.__getitem__( 0 )
	if g1.is_err():
		return 2
	got: Triple = g1.unwrap( 'x' )
	if got.a != 7 or got.b != 7 or got.c != 7:
		return 3
	return 0
''' ),
			# RC-element coverage for erase_at's ordering guarantee - 'b' is
			# decreffed on removal, 'a' and 'c' must survive (and read back
			# correctly) in the shifted positions. If decref/incref bookkeeping
			# were wrong here, this would double-free or leak at __del__ time
			# (list[T].__del__ decrefs every remaining slot on the way out) -
			# not something this test can observe directly without ASAN, but a
			# wrong refcount is exactly the kind of thing that turns into a
			# crash on a real run
			( 'list_str_erase_at_preserves_order_and_refcounts', '''
def main() -> i32:
	x: list[str] = list[str]()
	r0: Result[None,OverflowError] = x.append( 'a' )
	r1: Result[None,OverflowError] = x.append( 'b' )
	r2: Result[None,OverflowError] = x.append( 'c' )
	if r0.is_err() or r1.is_err() or r2.is_err():
		return 9
	er: Result[None,IndexError] = x.erase_at( 1 )
	if er.is_err():
		return 8
	if x.__len__() != 2:
		return 1
	g0: Result[str,IndexError] = x.__getitem__( 0 )
	g1: Result[str,IndexError] = x.__getitem__( 1 )
	if g0.is_err() or g1.is_err():
		return 7
	if g0.unwrap( 'x' ) == 'a' and g1.unwrap( 'x' ) == 'c':
		return 0
	return 99
''' ),
		] )


class UnsafeListGenericTests( CompilerTestCase ):
	''' UnsafeList[T] (lib/builtins/__list.py) - the same positional/order-
	preserving semantics list[T] itself has, minus the lock. Confirms the
	list[T] -> UnsafeList[T] rename (list[T] becoming the locked default)
	didn't change UnsafeList[T]'s own behavior at all - it's the exact
	code list[T] used to be. '''
	def setUp( self ) -> None:
		self.discovery = Discovery( import_builtins = True )
		self.compiler = Compiler( self.discovery )

	def _extern_ldflags( self ) -> str:
		flags: list[str] = []
		for lib in sorted( self.compiler.extern_libs ):
			if lib == 'c':
				continue
			flags.append( linker_c.resolve_lib_ldflag( _CC, lib, self.compiler.extern_libs[lib] ) )
		return ' '.join( flags )

	def _assert_compiles_and_runs( self, c_source: str, expected_exit: int = 0 ) -> None:
		with tempfile.TemporaryDirectory() as tmp:
			src_path = Path( tmp ) / 'generated.c'
			obj_path = Path( tmp ) / 'generated.o'
			exe_path = Path( tmp ) / 'test_exe'
			src_path.write_text( c_source, encoding = 'utf-8' )
			cc_result = _CC.compile( src_path, obj_path )
			self.assertEqual( cc_result.returncode, 0,
				f'{_CC.name} compile failed:\nstdout: {cc_result.stdout}\nstderr: {cc_result.stderr}{test_support.c_source_on_failure( c_source )}' )
			ldflags = self._extern_ldflags()
			link_result = _CC.link( exe_path, [ obj_path ], ldflags = ldflags )
			self.assertEqual( link_result.returncode, 0,
				f'{_CC.name} link failed:\nstdout: {link_result.stdout}\nstderr: {link_result.stderr}' )
			run_result = subprocess.run( [ str( exe_path ) ], capture_output = True )
			self.assertEqual( run_result.returncode, expected_exit,
				f'exited {run_result.returncode}, expected {expected_exit} (stderr: {run_result.stderr})' )

	@unittest.skipUnless( _CC is not None, 'no C compiler (clang/gcc/msvc) found - skipping' )
	def test_construct_append_getitem_get_ptr_erase( self ) -> None:
		# get_ptr() specifically - list[T] itself no longer exposes it (a
		# borrowed pointer is incompatible with a type whose whole point is
		# "safe to hand to another thread"), but UnsafeList[T] still does
		self._run( '''
def main() -> i32:
	x: UnsafeList[i32] = UnsafeList[i32]()
	x.append( 1 ).unwrap( 'append failed' )
	x.append( 2 ).unwrap( 'append failed' )
	x.append( 3 ).unwrap( 'append failed' )
	if x.__len__() != 3:
		return 1
	v: i32 = x.__getitem__( 1 ).unwrap( 'getitem failed' )
	if v != 2:
		return 2
	p: Ptr[i32] = x.get_ptr( 0 ).unwrap( 'get_ptr failed' )
	if p[0] != 1:
		return 3
	x.erase_at( 0 ).unwrap( 'erase_at failed' )
	if x.__len__() != 2:
		return 4
	v = x.__getitem__( 0 ).unwrap( 'getitem failed' )
	if v != 2:
		return 5
	return 0
''' )
		self.assertEqual( self.discovery.errors.errors, [] )
		self._assert_compiles_and_runs( emitter_c.emit_c( self.compiler ))


class ListThreadSafetyTests( test_support.RealCompileMixin, CompilerTestCase ):
	''' list[T] (lib/builtins/__list.py) is now locked by default (a real
	FastLock, acquired/released around every method) - these are the real
	compile+run stress tests that actually exercise concurrent access, not
	just single-threaded behavior (ListGenericTests above already covers
	that, and still passes unchanged against the new locked wrapper).

	Each test uses a KNOWN, closed-form expected total (count and/or sum),
	computed from the exact values each thread pushes - a wrong final
	number reliably indicates a lost/duplicated/corrupted element, not
	just "probably fine". subprocess timeout matches ThreadRealCompileTests
	(a real hang - e.g. a lost pop causing the consumer to spin forever -
	should fail loudly, not wedge the suite). '''
	def setUp( self ) -> None:
		self.discovery = Discovery( import_builtins = True )
		self.compiler = Compiler( self.discovery )

	@unittest.skipUnless( test_support.HAS_CC, 'no C compiler (clang/gcc/msvc) found - skipping' )
	def test_programs_compile_and_run( self ) -> None:
		''' every real compile-and-run program in this class, merged into a
		single executable (one build for the whole class); a nonzero exit is
		decoded back to the failing sub-program and its own return code. '''
		self.assert_programs_run([
			( 'concurrent_append_from_8_threads_known_sum', '''
import threading

class Pusher:
	target: list[i32]
	base: i32

	@staticmethod
	def make( target: list[i32], base: i32 ) -> Pusher:
		return Pusher.__allocate__( target = target, base = base )

	def run( self ) -> None:
		i: i32 = 0
		while i < 1000:
			with compiler.wrap_arithmetic:
				v: i32 = self.base + i
			self.target.append( v ).unwrap( 'append failed' )
			with compiler.wrap_arithmetic:
				i += 1

def main() -> i32:
	l: list[i32] = list[i32]()
	threads: list[threading.Thread] = list[threading.Thread]()
	t: i32 = 0
	while t < 8:
		with compiler.wrap_arithmetic:
			base: i32 = t * 1000
		p: Pusher = Pusher.make( l, base )
		threads.append( threading.Thread( p.run ) ).unwrap( 'append failed' )
		with compiler.wrap_arithmetic:
			t += 1
	i: usize = 0
	while i < 8:
		th: threading.Thread = threads.__getitem__( i ).unwrap( 'getitem failed' )
		th.join()
		with compiler.wrap_arithmetic:
			i += 1
	if l.__len__() != 8000:
		return 1
	total: i64 = 0
	i = 0
	while i < 8000:
		v: i32 = l.__getitem__( i ).unwrap( 'getitem failed' )
		with compiler.wrap_arithmetic:
			total += i64( v )
			i += 1
	if total != 31996000:
		return 2
	return 0
''' ),
			( '8_producers_1_consumer_concurrent_push_pop_known_sum', '''
import threading

class Pusher:
	target: list[i32]
	base: i32

	@staticmethod
	def make( target: list[i32], base: i32 ) -> Pusher:
		return Pusher.__allocate__( target = target, base = base )

	def run( self ) -> None:
		i: i32 = 0
		while i < 1000:
			with compiler.wrap_arithmetic:
				v: i32 = self.base + i
			self.target.append( v ).unwrap( 'append failed' )
			with compiler.wrap_arithmetic:
				i += 1

class Consumer:
	source: list[i32]
	total: i64
	popped: usize

	@staticmethod
	def make( source: list[i32] ) -> Consumer:
		return Consumer.__allocate__( source = source, total = 0, popped = 0 )

	def run( self ) -> None:
		while self.popped < 8000:
			r: Result[i32, IndexError] = self.source.pop()
			if r.is_ok():
				v: i32 = r.unwrap( 'checked is_ok' )
				with compiler.wrap_arithmetic:
					self.total += i64( v )
					self.popped += 1

def main() -> i32:
	l: list[i32] = list[i32]()
	c: Consumer = Consumer.make( l )
	consumer_thread: threading.Thread = threading.Thread( c.run )
	threads: list[threading.Thread] = list[threading.Thread]()
	t: i32 = 0
	while t < 8:
		with compiler.wrap_arithmetic:
			base: i32 = t * 1000
		p: Pusher = Pusher.make( l, base )
		threads.append( threading.Thread( p.run ) ).unwrap( 'append failed' )
		with compiler.wrap_arithmetic:
			t += 1
	i: usize = 0
	while i < 8:
		th: threading.Thread = threads.__getitem__( i ).unwrap( 'getitem failed' )
		th.join()
		with compiler.wrap_arithmetic:
			i += 1
	consumer_thread.join()
	if c.popped != 8000:
		return 1
	if c.total != 31996000:
		return 2
	if l.__len__() != 0:
		return 3
	return 0
''' ),
			( 'concurrent_append_of_rc_elements_no_leak_or_double_free', '''
import threading

class Item:
	value: i32

	@staticmethod
	def make( v: i32 ) -> Item:
		return Item.__allocate__( value = v )

class ItemPusher:
	target: list[Item]
	base: i32

	@staticmethod
	def make( target: list[Item], base: i32 ) -> ItemPusher:
		return ItemPusher.__allocate__( target = target, base = base )

	def run( self ) -> None:
		i: i32 = 0
		while i < 250:
			with compiler.wrap_arithmetic:
				v: i32 = self.base + i
			it: Item = Item.make( v )
			self.target.append( it ).unwrap( 'append failed' )
			compiler.decref( it )
			with compiler.wrap_arithmetic:
				i += 1

def main() -> i32:
	l: list[Item] = list[Item]()
	threads: list[threading.Thread] = list[threading.Thread]()
	t: i32 = 0
	while t < 4:
		with compiler.wrap_arithmetic:
			base: i32 = t * 250
		p: ItemPusher = ItemPusher.make( l, base )
		threads.append( threading.Thread( p.run ) ).unwrap( 'append failed' )
		with compiler.wrap_arithmetic:
			t += 1
	i: usize = 0
	while i < 4:
		th: threading.Thread = threads.__getitem__( i ).unwrap( 'getitem failed' )
		th.join()
		with compiler.wrap_arithmetic:
			i += 1
	if l.__len__() != 1000:
		return 1
	i = 0
	while i < 1000:
		it: Item = l.pop().unwrap( 'pop failed' )
		rc: usize = compiler.refcount( it )
		if rc != 1:
			return 2
		compiler.decref( it )
		with compiler.wrap_arithmetic:
			i += 1
	if l.__len__() != 0:
		return 3
	return 0
''' ),
		], timeout = 30 )


class FastListGenericTests( test_support.RealCompileMixin, CompilerTestCase ):
	''' FastList[T] (lib/builtins/__fastlist.py) end-to-end - the ORIGINAL
	StableIndexVector port: O(1) swap-and-pop erase, stable IDs that
	survive other inserts/deletes, but positional order is NOT preserved
	across an erase. Split out from list[T] (which now has real
	Python-list/array semantics instead - see ListGenericTests above) once
	that distinction became load-bearing enough to need two containers. '''

	def setUp( self ) -> None:
		self.discovery = Discovery( import_builtins = True )
		self.compiler = Compiler( self.discovery )

	@unittest.skipUnless( test_support.HAS_CC, 'no C compiler (clang/gcc/msvc) found - skipping' )
	def test_programs_compile_and_run( self ) -> None:
		''' every real compile-and-run program in this class, merged into a
		single executable (one build for the whole class); a nonzero exit is
		decoded back to the failing sub-program and its own return code. '''
		self.assert_programs_run([
			( 'fastlist_i32_construct_append_getitem_del', '''
def main() -> i32:
	x: FastList[i32] = FastList[i32]()
	r0: Result[usize,OverflowError] = x.append( 10 )
	r1: Result[usize,OverflowError] = x.append( 20 )
	r2: Result[usize,OverflowError] = x.append( 30 )
	if r0.is_err() or r1.is_err() or r2.is_err():
		return 9
	id0: usize = r0.unwrap( 'append failed' )
	id1: usize = r1.unwrap( 'append failed' )
	id2: usize = r2.unwrap( 'append failed' )
	if x.__len__() != 3:
		return 1
	g0: Result[i32,IndexError] = x.__getitem__( id0 )
	g1: Result[i32,IndexError] = x.__getitem__( id1 )
	g2: Result[i32,IndexError] = x.__getitem__( id2 )
	if g0.is_err() or g1.is_err() or g2.is_err():
		return 8
	if g0.unwrap( 'getitem failed' ) != 10:
		return 2
	if g1.unwrap( 'getitem failed' ) != 20:
		return 3
	if g2.unwrap( 'getitem failed' ) != 30:
		return 4
	return 0
''' ),
			# FastList[T]/RawFastList ports StableIndexVector (see the file's
			# own module docstring) - a swap-and-pop design, not an
			# insertion-order-preserving one. Its own README says so plainly:
			# "On deletion, the last element is swapped into the gap." This is
			# NOT a bug - it's what buys the O(1) erase and the "stable ID
			# survives other inserts/deletes" guarantee FastListHandle depends
			# on. This test pins that behavior down with a real compile-and-run
			# so it can't be "fixed" by accident later: append 10,20,30,40,50
			# (data positions 0..4 in insertion order), erase the middle one
			# (30, at position 2) - if order were preserved, positions 0..3
			# would read back 10,20,40,50; instead the LAST element (50) gets
			# swapped into the vacated slot, giving 10,20,50,40.
			( 'erase_does_not_preserve_positional_order', '''
def main() -> i32:
	x: FastList[i32] = FastList[i32]()
	r0: Result[usize,OverflowError] = x.append( 10 )
	r1: Result[usize,OverflowError] = x.append( 20 )
	r2: Result[usize,OverflowError] = x.append( 30 )
	r3: Result[usize,OverflowError] = x.append( 40 )
	r4: Result[usize,OverflowError] = x.append( 50 )
	if r0.is_err() or r1.is_err() or r2.is_err() or r3.is_err() or r4.is_err():
		return 9
	id2: usize = r2.unwrap( 'append failed' )
	er: Result[None,IndexError] = x.erase( id2 )
	if er.is_err():
		return 8
	if x.__len__() != 4:
		return 1
	g0: Result[i32,IndexError] = x.get_at( 0 )
	g1: Result[i32,IndexError] = x.get_at( 1 )
	g2: Result[i32,IndexError] = x.get_at( 2 )
	g3: Result[i32,IndexError] = x.get_at( 3 )
	if g0.is_err() or g1.is_err() or g2.is_err() or g3.is_err():
		return 7
	v0: i32 = g0.unwrap( 'x' )
	v1: i32 = g1.unwrap( 'x' )
	v2: i32 = g2.unwrap( 'x' )
	v3: i32 = g3.unwrap( 'x' )
	if v0 == 10 and v1 == 20 and v2 == 40 and v3 == 50:
		return 42 # order WAS preserved - contradicts the documented algorithm
	if v0 == 10 and v1 == 20 and v2 == 50 and v3 == 40:
		return 0 # swap-and-pop confirmed: last element (50) filled the gap
	return 99 # neither shape - something else entirely is wrong
''' ),
			# complements test_erase_does_not_preserve_positional_order above:
			# get_at (position-based) breaks order, but __getitem__ (stable-ID-
			# based) is a DIFFERENT accessor - _erase repoints the swapped
			# element's __indexes entry at its new position (see
			# RawFastList._erase), so every surviving id still resolves to the
			# same VALUE it always did, regardless of where the swap physically
			# moved it. Same 10,20,30,40,50 / erase id for 30 setup as the
			# position test, but reading back via the original ids (0,1,3,4)
			# instead of positions (0,1,2,3) - this is expected to read back
			# 10,20,40,50 (identity preserved), even though the POSITIONAL read
			# of the same list does not (10,20,50,40, per the other test)
			( 'erase_preserves_stable_id_identity_despite_position_swap', '''
def main() -> i32:
	x: FastList[i32] = FastList[i32]()
	r0: Result[usize,OverflowError] = x.append( 10 )
	r1: Result[usize,OverflowError] = x.append( 20 )
	r2: Result[usize,OverflowError] = x.append( 30 )
	r3: Result[usize,OverflowError] = x.append( 40 )
	r4: Result[usize,OverflowError] = x.append( 50 )
	if r0.is_err() or r1.is_err() or r2.is_err() or r3.is_err() or r4.is_err():
		return 9
	id0: usize = r0.unwrap( 'append failed' )
	id1: usize = r1.unwrap( 'append failed' )
	id2: usize = r2.unwrap( 'append failed' )
	id3: usize = r3.unwrap( 'append failed' )
	id4: usize = r4.unwrap( 'append failed' )
	er: Result[None,IndexError] = x.erase( id2 )
	if er.is_err():
		return 8
	g0: Result[i32,IndexError] = x.__getitem__( id0 )
	g1: Result[i32,IndexError] = x.__getitem__( id1 )
	g3: Result[i32,IndexError] = x.__getitem__( id3 )
	g4: Result[i32,IndexError] = x.__getitem__( id4 )
	if g0.is_err() or g1.is_err() or g3.is_err() or g4.is_err():
		return 7
	v0: i32 = g0.unwrap( 'x' )
	v1: i32 = g1.unwrap( 'x' )
	v3: i32 = g3.unwrap( 'x' )
	v4: i32 = g4.unwrap( 'x' )
	if v0 == 10 and v1 == 20 and v3 == 40 and v4 == 50:
		return 0 # stable-ID identity survived the positional swap
	return 99
''' ),
			# regression test for the id-collision bug found while
			# investigating order preservation (see emitter_c_test.py history -
			# RawFastList._get_free_id used to always return __len, which
			# SHRINKS on erase, so the next append could hand out an id that
			# was still held by a live element, silently aliasing two elements
			# onto the same id). Fixed via a real free-list (__free_ids/
			# __free_count) plus a monotonic __next_id counter that never goes
			# backwards. Same 10,20,30,40,50 / erase id for 30 / append 60
			# setup that used to demonstrate the collision: id4 (50) must
			# survive untouched, and the recycled id (from erasing 30) must be
			# handed to 60 rather than colliding with id4
			( 'append_after_erase_reuses_freed_id_without_aliasing', '''
def main() -> i32:
	x: FastList[i32] = FastList[i32]()
	r0: Result[usize,OverflowError] = x.append( 10 )
	r1: Result[usize,OverflowError] = x.append( 20 )
	r2: Result[usize,OverflowError] = x.append( 30 )
	r3: Result[usize,OverflowError] = x.append( 40 )
	r4: Result[usize,OverflowError] = x.append( 50 )
	if r0.is_err() or r1.is_err() or r2.is_err() or r3.is_err() or r4.is_err():
		return 9
	id2: usize = r2.unwrap( 'append failed' )
	id4: usize = r4.unwrap( 'append failed' )
	er: Result[None,IndexError] = x.erase( id2 )
	if er.is_err():
		return 8
	r5: Result[usize,OverflowError] = x.append( 60 )
	if r5.is_err():
		return 7
	id5: usize = r5.unwrap( 'append failed' )
	if id5 == id4:
		return 50 # still colliding - the fix did not take
	g4: Result[i32,IndexError] = x.__getitem__( id4 )
	g5: Result[i32,IndexError] = x.__getitem__( id5 )
	if g4.is_err() or g5.is_err():
		return 6
	v4: i32 = g4.unwrap( 'x' )
	v5: i32 = g5.unwrap( 'x' )
	if v4 == 50 and v5 == 60:
		return 0 # no aliasing - both ids resolve to their own, correct values
	return 99
''' ),
		] )


class ChrOrdTests( test_support.RealCompileMixin, CompilerTestCase ):
	''' chr()/ord() (lib/builtins/__init__.py) - built on str's own private
	UTF-8 encode/decode helpers (the same ones upper()/lower()/case-
	folding already use), not separate logic. '''
	def setUp( self ) -> None:
		self.discovery = Discovery( import_builtins = True )
		self.compiler = Compiler( self.discovery )

	@unittest.skipUnless( test_support.HAS_CC, 'no C compiler (clang/gcc/msvc) found - skipping' )
	def test_programs_compile_and_run( self ) -> None:
		''' every real compile-and-run program in this class, merged into a
		single executable (one build for the whole class); a nonzero exit is
		decoded back to the failing sub-program and its own return code. '''
		self.assert_programs_run([
			( 'ascii_round_trip', '''
def main() -> i32:
	if chr( 65 ) != "A":
		return 1
	if ord( "A" ) != 65:
		return 2
	if ord( "0" ) != 48:
		return 3
	return 0
''' ),
			# U+00E9 (e-acute, 2 UTF-8 bytes) and U+1F600 (grinning face emoji,
			# 4 UTF-8 bytes) - exercises _utf8_encoded_len/_encode_utf8_at/
			# _decode_utf8_at's own 2-byte and 4-byte branches, not just ASCII
			( 'multibyte_round_trip', '''
def main() -> i32:
	c2: str = chr( 0xE9 )
	if c2.byte_len() != 2:
		return 1
	if ord( c2 ) != 0xE9:
		return 2
	c4: str = chr( 0x1F600 )
	if c4.byte_len() != 4:
		return 3
	if ord( c4 ) != 0x1F600:
		return 4
	return 0
''' ),
		] )

	@unittest.skipUnless( _CC is not None, 'no C compiler (clang/gcc/msvc) found - skipping' )
	def test_ord_on_empty_string_panics( self ) -> None:
		self._run( '''
def main() -> i32:
	ord( "" )
	return 0
''' )
		self.assertEqual( self.discovery.errors.errors, [] )
		self._assert_compiles_and_runs( emitter_c.emit_c( self.compiler ), expected_exit = 1 )

	@unittest.skipUnless( _CC is not None, 'no C compiler (clang/gcc/msvc) found - skipping' )
	def test_ord_on_multi_codepoint_string_panics( self ) -> None:
		self._run( '''
def main() -> i32:
	ord( "ab" )
	return 0
''' )
		self.assertEqual( self.discovery.errors.errors, [] )
		self._assert_compiles_and_runs( emitter_c.emit_c( self.compiler ), expected_exit = 1 )

	@unittest.skipUnless( _CC is not None, 'no C compiler (clang/gcc/msvc) found - skipping' )
	def test_chr_on_surrogate_half_panics( self ) -> None:
		# U+D800 is a UTF-16 surrogate half - not a valid Unicode code point
		# on its own, same as Python's own chr(0xD800) raising ValueError
		self._run( '''
def main() -> i32:
	chr( 0xD800 )
	return 0
''' )
		self.assertEqual( self.discovery.errors.errors, [] )
		self._assert_compiles_and_runs( emitter_c.emit_c( self.compiler ), expected_exit = 1 )


class MatchValuePatternRealCompileTests( test_support.RealCompileMixin, CompilerTestCase ):
	''' real compile+run coverage for type_resolver.py's _match_pattern
	ast.MatchValue handling - `case Color.Red:`/`case 5:` desugaring to a
	plain == Compare. Unlike type_resolver_test.py's own MatchValue tests
	(which only check the desugared AST shape), these confirm the
	generated code actually branches correctly. '''
	def setUp( self ) -> None:
		self.discovery = Discovery( import_builtins = True )
		self.compiler = Compiler( self.discovery )

	@unittest.skipUnless( test_support.HAS_CC, 'no C compiler (clang/gcc/msvc) found - skipping' )
	def test_programs_compile_and_run( self ) -> None:
		''' every real compile-and-run program in this class, merged into a
		single executable (one build for the whole class); a nonzero exit is
		decoded back to the failing sub-program and its own return code. '''
		self.assert_programs_run([
			( 'match_cenum_member_value_patterns', '''
@enum( i32 )
class Color:
	Red = 1
	Green = 2
	Blue = _

def classify( c: Color ) -> i32:
	match c:
		case Color.Red:
			return 1
		case Color.Green:
			return 2
		case _:
			return 99

def main() -> i32:
	if classify( Color.Green ) != 2:
		return 1
	if classify( Color.Blue ) != 99:
		return 2
	if classify( Color.Red ) != 1:
		return 3
	return 0
''' ),
			( 'match_plain_literal_value_patterns', '''
def classify( x: i32 ) -> i32:
	match x:
		case 1:
			return 100
		case 2:
			return 200
		case _:
			return 999

def main() -> i32:
	if classify( 1 ) != 100:
		return 1
	if classify( 2 ) != 200:
		return 2
	if classify( 3 ) != 999:
		return 3
	return 0
''' ),
		] )


class CEnumReturnCoercionTests( test_support.RealCompileMixin, CompilerTestCase ):
	''' real compile+run coverage for _stmt_Return's own CEnum<->underlying-
	scalar coercion - "a CEnum has exactly the same runtime representation as
	its underlying type" (see _try_lower_construct_call's own CEnum-
	construction comment), so returning one where the other is declared is a
	value-preserving reinterpretation in EITHER direction, mirroring
	_check_assignable's own bidirectional exemption elsewhere. _stmt_Return
	can't just delegate to _check_assignable (strict=False is deliberate, to
	avoid it firing before the covered-Result-error-widening case gets a
	chance - see that method's own comment), so it re-derives every exemption
	_check_assignable would apply - PLAN_COMPILER_BUG_SWEEP.md's own audit
	found this had only ever re-derived ONE of the two directions: returning
	a CEnum value where the function declares its own underlying scalar type
	worked, but the REVERSE (returning a raw scalar where the function
	declares the CEnum) was wrongly rejected - confirmed via a real repro
	before the fix ("function returns X, not Y" for a case that should be a
	legitimate reinterpretation). '''

	def setUp( self ) -> None:
		self.discovery = Discovery( import_builtins = True )
		self.compiler = Compiler( self.discovery )

	@unittest.skipUnless( test_support.HAS_CC, 'no C compiler (clang/gcc/msvc) found - skipping' )
	def test_programs_compile_and_run( self ) -> None:
		self.assert_programs_run([
			( 'both_cenum_underlying_return_directions_work', '''
@enum( i32 )
class Color:
	Red = 0
	Blue = 1

def get_underlying() -> i32:
	return Color.Blue

def get_red() -> Color:
	x: i32 = 0
	return x

def main() -> i32:
	if get_underlying() != 1:
		return 1
	c: Color = get_red()
	if c != Color.Red:
		return 2
	return 0
''' ),
		] )

	@unittest.skipUnless( _CC is not None, 'no C compiler (clang/gcc/msvc) found - skipping' )
	def test_genuinely_mismatched_return_type_still_rejected( self ) -> None:
		# negative companion - a completely unrelated type (str) returned
		# where a CEnum is declared must still be rejected, not silently
		# accepted by an over-broadened coercion
		self._run( '''
@enum( i32 )
class Color:
	Red = 0
	Blue = 1

def get_wrong() -> Color:
	s: str = 'not a color'
	return s

def main() -> i32:
	c: Color = get_wrong()
	return 0
''' )
		self.assertNotEqual( self.discovery.errors.errors, [] )

	@unittest.skipUnless( test_support.HAS_CC, 'no C compiler (clang/gcc/msvc) found - skipping' )
	def test_bare_literal_via_return_and_construction( self ) -> None:
		# _expr_Constant's own CEnum handling (distinct from _stmt_Return's
		# coercion tested above) - a BARE integer literal, not a named local,
		# validated/typed against the CEnum's own underlying scalar, both via
		# a plain `return 1` and via an explicit Color(1) construction call
		self._run( '''
@enum( i32 )
class Color:
	Red = 0
	Blue = 1

def get_via_return() -> Color:
	return 1

def get_via_construct() -> Color:
	return Color( 1 )

def main() -> i32:
	a: Color = get_via_return()
	if a != Color.Blue:
		return 1
	b: Color = get_via_construct()
	if b != Color.Blue:
		return 2
	return 0
''' )
		self.assertEqual( self.discovery.errors.errors, [] )

	@unittest.skipUnless( _CC is not None, 'no C compiler (clang/gcc/msvc) found - skipping' )
	def test_kind_mismatched_literal_rejected_cleanly_not_crashed( self ) -> None:
		# _expr_Constant's own CEnum branch used to exempt EVERY CEnum
		# expected_type from kind-validation outright (meant only for a raw
		# INT literal's own magnitude, per the construction-call comment) -
		# a kind-mismatched literal (a bare string, here) sailed through
		# untyped-checked, tagging the resulting Const with the CEnum type
		# while its own .value stayed the Python string - confirmed to crash
		# emitter_c.py's _emit_const with an uncaught Python
		# NotImplementedError (a raw traceback, not a compile error) rather
		# than being cleanly rejected. Exercises the bare-literal-via-return
		# shape directly (distinct from test_genuinely_mismatched_return_
		# type_still_rejected above, which uses a named local of the wrong
		# type, not a mismatched literal)
		self._run( '''
@enum( i32 )
class Color:
	Red = 0
	Blue = 1

def get_wrong() -> Color:
	return 'not a color'

def main() -> i32:
	c: Color = get_wrong()
	return 0
''' )
		self.assertNotEqual( self.discovery.errors.errors, [] )

	@unittest.skipUnless( _CC is not None, 'no C compiler (clang/gcc/msvc) found - skipping' )
	def test_kind_mismatched_construction_literal_rejected_cleanly_not_crashed( self ) -> None:
		# same bug, the OTHER call site that reaches _expr_Constant's CEnum
		# branch: an explicit Color(...) construction call whose own argument
		# is a kind-mismatched literal. _try_lower_construct_call's own CEnum
		# branch only validates an INT literal's own magnitude directly -
		# anything else is deferred entirely to _expr_Constant, so this
		# crashed the identical way before the fix
		self._run( '''
@enum( i32 )
class Color:
	Red = 0
	Blue = 1

def main() -> i32:
	c: Color = Color( 'bad' )
	return 0
''' )
		self.assertNotEqual( self.discovery.errors.errors, [] )

	@unittest.skipUnless( _CC is not None, 'no C compiler (clang/gcc/msvc) found - skipping' )
	def test_out_of_range_literal_rejected( self ) -> None:
		# the magnitude check (previously only reachable via the
		# construction-call path's own separate, duplicate check) now also
		# applies via the bare-literal-return path, using the CEnum's own
		# underlying scalar's real range rather than skipping validation
		self._run( '''
@enum( u8 )
class Small:
	A = 0
	B = 1

def get_bad() -> Small:
	return 999

def main() -> i32:
	s: Small = get_bad()
	return 0
''' )
		self.assertNotEqual( self.discovery.errors.errors, [] )


class AtomicRealCompileTests( test_support.RealCompileMixin, CompilerTestCase ):
	''' real compile+run coverage for compiler.atomic_*(Ptr[T], ...)
	(lowering.py's _lower_compiler_atomic_*, ir.py's Atomic* instructions,
	emitter_c.py's stdatomic.h-based codegen) and lib/atomic.py's Atomic[T]
	wrapper built on top of them. Single-threaded correctness only here -
	a real multi-threaded stress test lands once Phase 3 (Thread) exists. '''
	def setUp( self ) -> None:
		self.discovery = Discovery( import_builtins = True )
		self.compiler = Compiler( self.discovery )

	@unittest.skipUnless( test_support.HAS_CC, 'no C compiler (clang/gcc/msvc) found - skipping' )
	def test_programs_compile_and_run( self ) -> None:
		''' every real compile-and-run program in this class, merged into a
		single executable (one build for the whole class); a nonzero exit is
		decoded back to the failing sub-program and its own return code. '''
		self.assert_programs_run([
			( 'load_store_add_sub_exchange_round_trip', '''
import sys

def main() -> i32:
	p: Ptr[i32] = sys.alloc[i32]( 1 )
	compiler.atomic_store( p, 10 )
	if compiler.atomic_load( p ) != 10:
		return 1
	old: i32 = compiler.atomic_add( p, 5 )
	if old != 10 or compiler.atomic_load( p ) != 15:
		return 2
	old = compiler.atomic_sub( p, 3 )
	if old != 15 or compiler.atomic_load( p ) != 12:
		return 3
	old = compiler.atomic_exchange( p, 100 )
	if old != 12 or compiler.atomic_load( p ) != 100:
		return 4
	sys.free( p )
	return 0
''' ),
			( 'compare_exchange_success_and_failure', '''
import sys

def main() -> i32:
	p: Ptr[i32] = sys.alloc[i32]( 1 )
	compiler.atomic_store( p, 15 )
	expected: Ptr[i32] = sys.alloc[i32]( 1 )
	expected[0] = 15
	if not compiler.atomic_compare_exchange( p, expected, 100 ):
		return 1
	if compiler.atomic_load( p ) != 100:
		return 2
	# stale expected - must fail and be updated to the real current value
	expected[0] = 15
	if compiler.atomic_compare_exchange( p, expected, 999 ):
		return 3
	if expected[0] != 100:
		return 4
	if compiler.atomic_load( p ) != 100:
		return 5
	sys.free( p )
	sys.free( expected )
	return 0
''' ),
			( 'atomic_wrapper_i32_and_bool', '''
import atomic

def main() -> i32:
	a: atomic.Atomic[i32] = atomic.Atomic[i32]( 5 )
	if a.load() != 5:
		return 1
	old: i32 = a.fetch_add( 10 )
	if old != 5 or a.load() != 15:
		return 2
	old = a.fetch_sub( 5 )
	if old != 15 or a.load() != 10:
		return 3
	old = a.exchange( 42 )
	if old != 10 or a.load() != 42:
		return 4

	b: atomic.Atomic[bool] = atomic.Atomic[bool]( False )
	if b.load():
		return 5
	b.store( True )
	if not b.load():
		return 6
	return 0
''' ),
			( 'atomic_wrapper_compare_exchange', '''
import atomic
import sys

def main() -> i32:
	a: atomic.Atomic[usize] = atomic.Atomic[usize]( 7 )
	expected: Ptr[usize] = sys.alloc[usize]( 1 )
	expected[0] = 7
	if not a.compare_exchange( expected, 200 ):
		return 1
	if a.load() != 200:
		return 2
	sys.free( expected )
	return 0
''' ),
		] )


class ClosureRealCompileTests( test_support.RealCompileMixin, CompilerTestCase ):
	''' real compile+run coverage for bound-method closures
	(Closure[[...],...] - mpy_types.ClosureType, lowering.py's
	_lower_bound_method_closure/_get_or_create_closure_trampoline/
	_try_lower_closure_call). Unlike the IR-shape tests in
	lowering_test.py, these confirm the generated code actually computes
	the right values AND manages the captured receiver's refcount
	correctly - a wrong/missing incref or decref here either leaks or
	double-frees, and calling a closure must never touch the receiver's
	refcount at all (confirmed by real, repeated calls in a loop, not
	just a single call). '''
	def setUp( self ) -> None:
		self.discovery = Discovery( import_builtins = True )
		self.compiler = Compiler( self.discovery )

	@unittest.skipUnless( test_support.HAS_CC, 'no C compiler (clang/gcc/msvc) found - skipping' )
	def test_programs_compile_and_run( self ) -> None:
		''' every real compile-and-run program in this class, merged into a
		single executable (one build for the whole class); a nonzero exit is
		decoded back to the failing sub-program and its own return code. '''
		self.assert_programs_run([
			( 'construct_and_call_no_args', '''
class Worker:
	x: i32

	@staticmethod
	def make( v: i32 ) -> Worker:
		return Worker.__allocate__( x = v )

	def get( self ) -> i32:
		return self.x

def main() -> i32:
	w: Worker = Worker.make( 42 )
	c: Closure[[], i32] = w.get
	if c() != 42:
		return 1
	return 0
''' ),
			( 'construct_and_call_with_args', '''
class Worker:
	x: i32

	@staticmethod
	def make( v: i32 ) -> Worker:
		return Worker.__allocate__( x = v )

	def add( self, n: i32 ) -> i32:
		with compiler.wrap_arithmetic:
			return self.x + n

def main() -> i32:
	w: Worker = Worker.make( 42 )
	c: Closure[[i32], i32] = w.add
	if c( 8 ) != 50:
		return 1
	if c( 100 ) != 142:
		return 2
	return 0
''' ),
			# the exact same class/method also called the ordinary way
			# (w.get(), no Closure[...] anywhere) must keep working unchanged
			( 'ordinary_method_call_unaffected', '''
class Worker:
	x: i32

	@staticmethod
	def make( v: i32 ) -> Worker:
		return Worker.__allocate__( x = v )

	def get( self ) -> i32:
		return self.x

def main() -> i32:
	w: Worker = Worker.make( 7 )
	if w.get() != 7:
		return 1
	return 0
''' ),
			# construct once, incref by exactly 1; call it 200 times in a loop
			# (calling a closure must never touch the receiver's refcount -
			# the cast back to the real receiver type inside the trampoline is
			# a borrowed reinterpretation, not a new owned reference); decref
			# once, back to the pre-construction refcount
			( 'refcount_incremented_once_on_construction_and_calls_are_neutral', '''
class Worker:
	x: i32

	@staticmethod
	def make( v: i32 ) -> Worker:
		return Worker.__allocate__( x = v )

	def get( self ) -> i32:
		return self.x

def main() -> i32:
	w: Worker = Worker.make( 42 )
	rc0: usize = compiler.refcount( w )
	c: Closure[[], i32] = w.get
	rc1: usize = compiler.refcount( w )
	with compiler.wrap_arithmetic:
		if rc1 != rc0 + 1:
			return 1
	i: i32 = 0
	while i < 200:
		if c() != 42:
			return 2
		with compiler.wrap_arithmetic:
			i += 1
	rc_after_calls: usize = compiler.refcount( w )
	if rc_after_calls != rc1:
		return 3
	compiler.decref( c )
	rc2: usize = compiler.refcount( w )
	if rc2 != rc0:
		return 4
	return 0
''' ),
			# `d = c` (an ordinary aliasing read) makes d a SECOND owner of the
			# SAME closure object - an ordinary RC incref on the closure
			# itself, not a second incref of w (w's own refcount only moves at
			# closure construction/destruction, never at aliasing) - this is
			# exactly the "same closure handed to N places" shape Thread will
			# need (spawn N threads off one closure), just via a local alias
			# here rather than N constructor calls
			( 'shared_closure_across_multiple_owners', '''
class Worker:
	x: i32

	@staticmethod
	def make( v: i32 ) -> Worker:
		return Worker.__allocate__( x = v )

	def get( self ) -> i32:
		return self.x

def main() -> i32:
	w: Worker = Worker.make( 9 )
	rc0: usize = compiler.refcount( w )
	c: Closure[[], i32] = w.get
	rc1: usize = compiler.refcount( w )
	with compiler.wrap_arithmetic:
		if rc1 != rc0 + 1:
			return 1
	d: Closure[[], i32] = c
	closure_rc: usize = compiler.refcount( c )
	if closure_rc != 2:
		return 2
	if c() != 9 or d() != 9:
		return 3
	compiler.decref( c )
	rc_mid: usize = compiler.refcount( w )
	if rc_mid != rc1:
		return 4
	compiler.decref( d )
	rc2: usize = compiler.refcount( w )
	if rc2 != rc0:
		return 5
	return 0
''' ),
		] )


class PropertyRealCompileTests( test_support.RealCompileMixin, CompilerTestCase ):
	''' real compile+run coverage for @property (lowering.py's _expr_
	Attribute is_property branch) - unlike the IR-shape assertions in
	lowering_test.py, these confirm the generated code actually calls the
	getter and produces the right value, and (for an RC-typed property)
	doesn't leak or double-free the returned value across repeated reads. '''
	def setUp( self ) -> None:
		self.discovery = Discovery( import_builtins = True )
		self.compiler = Compiler( self.discovery )

	@unittest.skipUnless( test_support.HAS_CC, 'no C compiler (clang/gcc/msvc) found - skipping' )
	def test_programs_compile_and_run( self ) -> None:
		''' every real compile-and-run program in this class, merged into a
		single executable (one build for the whole class); a nonzero exit is
		decoded back to the failing sub-program and its own return code. '''
		self.assert_programs_run([
			# obj.attr (no call parens) actually calls the getter - a plain
			# GetAttr would find no real field named 'doubled' and fail to
			# compile at all, so a successful compile+correct value together
			# confirm the Call-based dispatch is really happening
			( 'scalar_property_computed_from_field', '''
class Box:
	x: i32

	@property
	def doubled( self ) -> i32:
		with compiler.wrap_arithmetic:
			return self.x * 2

def main() -> i32:
	b: Box = Box( x = 21 )
	if b.doubled != 42:
		return 1
	return 0
''' ),
			# the property's result participates in an ordinary expression
			# exactly like a real field would - not just a bare read
			( 'property_used_in_expression', '''
class Box:
	x: i32

	@property
	def doubled( self ) -> i32:
		with compiler.wrap_arithmetic:
			return self.x * 2

def main() -> i32:
	b: Box = Box( x = 5 )
	with compiler.wrap_arithmetic:
		y: i32 = b.doubled + 1
	if y != 11:
		return 1
	return 0
''' ),
			# an RC-typed property (returns a fresh str each read) read
			# repeatedly in a loop - a missing/wrong incref or decref on the
			# Call's own result would leak or double-free, and 200
			# iterations is enough for that to reliably surface
			( 'rc_typed_property_read_in_a_loop', '''
class Greeter:
	name: str

	@property
	def greeting( self ) -> str:
		return "hello " + self.name

def main() -> i32:
	g: Greeter = Greeter( name = "world" )
	i: i32 = 0
	while i < 200:
		if len( g.greeting ) != 11:
			return 1
		with compiler.wrap_arithmetic:
			i += 1
	return 0
''' ),
		] )


class ThreadRealCompileTests( test_support.RealCompileMixin, CompilerTestCase ):
	''' real compile+run coverage for threading.Thread (lib/threading.py),
	built on Phase 1 (compiler.atomic_*/lib/atomic.py) and Phase 2b
	(Closure[[...],...]) - see the approved atomics-closures-threading
	plan. A real OS thread is the only way to confirm the closure-as-
	thread-payload design actually works end to end: the receiver's own
	RC lifecycle across a genuine thread boundary, and N threads safely
	sharing ONE closure via an ordinary aliasing incref (not move[T] - see
	the plan's own reasoning for why that would have been wrong).
	subprocess timeout is set explicitly (unlike every other real-run test
	class here) since a real hang (a deadlocked join()) should fail loudly
	rather than wedge the whole suite indefinitely. '''
	def setUp( self ) -> None:
		self.discovery = Discovery( import_builtins = True )
		self.compiler = Compiler( self.discovery )

	@unittest.skipUnless( test_support.HAS_CC, 'no C compiler (clang/gcc/msvc) found - skipping' )
	def test_programs_compile_and_run( self ) -> None:
		''' every real compile-and-run program in this class, merged into a
		single executable (one build for the whole class); a nonzero exit is
		decoded back to the failing sub-program and its own return code. '''
		self.assert_programs_run([
			( 'spawn_one_thread_mutates_shared_state', '''
import threading

class Worker:
	x: i32

	@staticmethod
	def make( v: i32 ) -> Worker:
		return Worker.__allocate__( x = v )

	def run( self ) -> None:
		self.x = 99

def main() -> i32:
	w: Worker = Worker.make( 0 )
	t: threading.Thread = threading.Thread( w.run )
	t.join()
	if w.x != 99:
		return 1
	return 0
''' ),
			# the real end-to-end proof: one closure, shared (ordinary
			# aliasing incref, NOT move[T]) across 8 separate Thread spawns,
			# each running 1000 lock-free fetch_add's on the SAME Atomic[usize]
			# - the exact shape the plan's own design discussion converged on
			( 'n_threads_share_one_closure_via_atomic_counter', '''
import threading
import atomic

class Counter:
	n: atomic.Atomic[usize]

	@staticmethod
	def make() -> Counter:
		return Counter.__allocate__( n = atomic.Atomic[usize]( 0 ) )

	def bump( self ) -> None:
		i: usize = 0
		while i < 1000:
			self.n.fetch_add( 1 )
			with compiler.wrap_arithmetic:
				i += 1

def main() -> i32:
	c: Counter = Counter.make()
	closure: Closure[[], None] = c.bump
	threads: list[threading.Thread] = list[threading.Thread]()
	i: usize = 0
	while i < 8:
		threads.append( threading.Thread( closure ) ).unwrap( 'append failed' )
		with compiler.wrap_arithmetic:
			i += 1
	i = 0
	while i < 8:
		t: threading.Thread = threads.__getitem__( i ).unwrap( 'getitem failed' )
		t.join()
		with compiler.wrap_arithmetic:
			i += 1
	if c.n.load() != 8000:
		return 1
	return 0
''' ),
		], timeout = 30 )


class StrFindIndexSplitTests( test_support.RealCompileMixin, CompilerTestCase ):
	''' str.find()/str.index()/str.split() (lib/builtins/__init__.py) -
	both listed missing in TODO.txt, implemented as real general-purpose
	methods (byte-level substring search built directly off str's own
	__data/__byte_size fields) rather than one-off logic embedded in
	split() alone - split() itself is built on find(), not its own
	separate scanning. Same import_builtins=True + real compile-and-run
	convention as ListGenericTests above (split() needs list[T] for real). '''

	def setUp( self ) -> None:
		self.discovery = Discovery( import_builtins = True )
		self.compiler = Compiler( self.discovery )

	@unittest.skipUnless( test_support.HAS_CC, 'no C compiler (clang/gcc/msvc) found - skipping' )
	def test_programs_compile_and_run( self ) -> None:
		''' every real compile-and-run program in this class, merged into a
		single executable (one build for the whole class); a nonzero exit is
		decoded back to the failing sub-program and its own return code. '''
		self.assert_programs_run([
			( 'find_and_index', '''
def main() -> i32:
	s: str = 'deadbeef-dead-beef-dead-beefdeadbeef'
	r0: Result[usize,IndexError] = s.find( '-' )
	if r0.is_err() or r0.unwrap( 'x' ) != 8:
		return 1
	r1: Result[usize,IndexError] = s.find( 'zzz' )
	if r1.is_ok():
		return 2
	r2: Result[usize,IndexError] = s.find( '' )
	if r2.is_err() or r2.unwrap( 'x' ) != 0:
		return 3
	if s.index( 'beef' ) != 4:
		return 4
	if s.find( s ).unwrap( 'x' ) != 0:
		return 5
	r3: Result[usize,IndexError] = s.find( 'toolongtoolongtoolongtoolongtoolongtoolong' )
	if r3.is_ok():
		return 6
	# find() with an explicit start offset - resumes past the first match
	r4: Result[usize,IndexError] = s.find( '-', 9 )
	if r4.is_err() or r4.unwrap( 'x' ) != 13:
		return 7
	return 0
''' ),
			# the exact motivating case from PLAN_SUBCLASSING_VTABLES_COM.md's
			# own blocked-on note: GUID's constructor parsing a hyphenated hex
			# string via str.split('-')
			( 'split_guid_like_string', '''
def main() -> i32:
	s: str = 'deadbeef-dead-beef-dead-beefdeadbeef'
	parts: list[str] = s.split( '-' )
	if parts.__len__() != 5:
		return 1
	g0: Result[str,IndexError] = parts.__getitem__( 0 )
	g1: Result[str,IndexError] = parts.__getitem__( 1 )
	g2: Result[str,IndexError] = parts.__getitem__( 2 )
	g3: Result[str,IndexError] = parts.__getitem__( 3 )
	g4: Result[str,IndexError] = parts.__getitem__( 4 )
	if g0.is_err() or g1.is_err() or g2.is_err() or g3.is_err() or g4.is_err():
		return 9
	if g0.unwrap( 'x' ) != 'deadbeef':
		return 2
	if g1.unwrap( 'x' ) != 'dead':
		return 3
	if g2.unwrap( 'x' ) != 'beef':
		return 4
	if g3.unwrap( 'x' ) != 'dead':
		return 5
	if g4.unwrap( 'x' ) != 'beefdeadbeef':
		return 6
	return 0
''' ),
			( 'split_edge_cases', '''
def main() -> i32:
	empty: list[str] = ''.split( ',' )
	if empty.__len__() != 1:
		return 1
	e0: Result[str,IndexError] = empty.__getitem__( 0 )
	if e0.unwrap( 'x' ) != '':
		return 2

	leading: list[str] = ',a,b'.split( ',' )
	if leading.__len__() != 3:
		return 3
	l0: Result[str,IndexError] = leading.__getitem__( 0 )
	if l0.unwrap( 'x' ) != '':
		return 4

	no_sep: list[str] = 'abc'.split( ',' )
	if no_sep.__len__() != 1:
		return 5
	n0: Result[str,IndexError] = no_sep.__getitem__( 0 )
	if n0.unwrap( 'x' ) != 'abc':
		return 6

	consecutive: list[str] = 'a,,b'.split( ',' )
	if consecutive.__len__() != 3:
		return 7
	c1: Result[str,IndexError] = consecutive.__getitem__( 1 )
	if c1.unwrap( 'x' ) != '':
		return 8
	return 0
''' ),
			# str.__contains__ (find().is_ok()) plus the `in`/`not in`
			# operator dispatch to it (lowering.py's _lower_in_comparison)
			( 'contains_and_in_operator', '''
def main() -> i32:
	s: str = 'deadbeef-dead-beef-dead-beefdeadbeef'
	if not s.__contains__( 'beef' ):
		return 1
	if s.__contains__( 'zzz' ):
		return 2
	if not ( 'beef' in s ):
		return 3
	if 'zzz' in s:
		return 4
	if not ( 'zzz' not in s ):
		return 5
	if 'beef' not in s:
		return 6
	return 0
''' ),
		] )


class StrPhase1MethodsTests( test_support.RealCompileMixin, CompilerTestCase ):
	''' Phase 1 of TODO.txt's str-methods plan: startswith/endswith/
	removeprefix/removesuffix/rfind/rindex/replace/rsplit/join/partition/
	rpartition/isascii (lib/builtins/__init__.py) - all built directly on
	find()/_byte_slice()/byte_len(), no new OS primitives. partition()/
	rpartition() return a real tuple[str,str,str] (PLAN_TUPLE.md). '''

	def setUp( self ) -> None:
		self.discovery = Discovery( import_builtins = True )
		self.compiler = Compiler( self.discovery )

	@unittest.skipUnless( test_support.HAS_CC, 'no C compiler (clang/gcc/msvc) found - skipping' )
	def test_programs_compile_and_run( self ) -> None:
		''' every real compile-and-run program in this class, merged into a
		single executable (one build for the whole class); a nonzero exit is
		decoded back to the failing sub-program and its own return code. '''
		self.assert_programs_run([
			( 'startswith_endswith', '''
def main() -> i32:
	if not 'hello world'.startswith( 'hello' ):
		return 1
	if 'hello world'.startswith( 'world' ):
		return 2
	if not 'hello world'.startswith( 'world', 6 ):
		return 3
	if not 'hello world'.startswith( '' ):
		return 4
	if not 'hello world'.endswith( 'world' ):
		return 5
	if 'hello world'.endswith( 'hello' ):
		return 6
	if not 'hello world'.endswith( '' ):
		return 7
	if 'short'.startswith( 'much too long' ):
		return 8
	if 'short'.endswith( 'much too long' ):
		return 9
	return 0
''' ),
			( 'removeprefix_removesuffix', '''
def main() -> i32:
	if 'hello world'.removeprefix( 'hello ' ) != 'world':
		return 1
	if 'hello world'.removeprefix( 'nope' ) != 'hello world':
		return 2
	if 'hello world'.removesuffix( ' world' ) != 'hello':
		return 3
	if 'hello world'.removesuffix( 'nope' ) != 'hello world':
		return 4
	return 0
''' ),
			( 'rfind_rindex', '''
def main() -> i32:
	r1: Result[usize,IndexError] = 'abcabc'.rfind( 'abc' )
	if r1.unwrap( 'x' ) != 3:
		return 1
	r2: Result[usize,IndexError] = 'abcabc'.rfind( 'nope' )
	if r2.is_ok():
		return 2
	if 'abcabc'.rindex( 'bc' ) != 4:
		return 3
	if 'abcabc'.rfind( '' ).unwrap( 'x' ) != 6:
		return 4
	return 0
''' ),
			( 'replace', '''
def main() -> i32:
	if 'banana'.replace( 'a', 'o' ) != 'bonono':
		return 1
	if 'banana'.replace( 'nope', 'x' ) != 'banana':
		return 2
	if 'aaa'.replace( 'a', 'bb' ) != 'bbbbbb':
		return 3
	if 'aaa'.replace( 'aa', 'b' ) != 'ba':
		return 4
	return 0
''' ),
			( 'rsplit_matches_split', '''
def main() -> i32:
	a: list[str] = 'a,b,c'.rsplit( ',' )
	if a.__len__() != 3:
		return 1
	if a.__getitem__( 0 ).unwrap( 'x' ) != 'a':
		return 2
	if a.__getitem__( 2 ).unwrap( 'x' ) != 'c':
		return 3
	return 0
''' ),
			( 'join', '''
def main() -> i32:
	parts: list[str] = 'a,b,c'.split( ',' )
	joined: str = '-'.join( parts )
	if joined != 'a-b-c':
		return 1
	empty: list[str] = list[str]()
	if '-'.join( empty ) != '':
		return 2
	single: list[str] = list[str]()
	single.append( 'solo' ).unwrap( 'append failed' )
	if '-'.join( single ) != 'solo':
		return 3
	return 0
''' ),
			( 'partition_rpartition', '''
def main() -> i32:
	p1: tuple[str,str,str] = 'key=value'.partition( '=' )
	if p1[0] != 'key' or p1[1] != '=' or p1[2] != 'value':
		return 1
	p2: tuple[str,str,str] = 'noequals'.partition( '=' )
	if p2[0] != 'noequals' or p2[1] != '' or p2[2] != '':
		return 2
	p3: tuple[str,str,str] = 'a.b.c'.rpartition( '.' )
	if p3[0] != 'a.b' or p3[1] != '.' or p3[2] != 'c':
		return 3
	p4: tuple[str,str,str] = 'noequals'.rpartition( '=' )
	if p4[0] != '' or p4[1] != '' or p4[2] != 'noequals':
		return 4
	return 0
''' ),
			( 'isascii', '''
def main() -> i32:
	if not 'hello'.isascii():
		return 1
	if not ''.isascii():
		return 2
	if 'héllo'.isascii():
		return 3
	return 0
''' ),
		] )

	@unittest.skipUnless( _CC is not None, 'no C compiler (clang/gcc/msvc) found - skipping' )
	def test_rindex_panics_when_not_found( self ) -> None:
		self._run( '''
def main() -> i32:
	'abc'.rindex( 'nope' )
	return 0
''' )
		self.assertEqual( self.discovery.errors.errors, [] )
		self._assert_compiles_and_runs( emitter_c.emit_c( self.compiler ), expected_exit = 1 )


class StrPhase2PaddingTests( test_support.RealCompileMixin, CompilerTestCase ):
	''' Phase 2 of TODO.txt's str-methods plan: ljust/rjust/zfill
	(lib/builtins/__init__.py). '''

	def setUp( self ) -> None:
		self.discovery = Discovery( import_builtins = True )
		self.compiler = Compiler( self.discovery )

	@unittest.skipUnless( test_support.HAS_CC, 'no C compiler (clang/gcc/msvc) found - skipping' )
	def test_programs_compile_and_run( self ) -> None:
		''' every real compile-and-run program in this class, merged into a
		single executable (one build for the whole class); a nonzero exit is
		decoded back to the failing sub-program and its own return code. '''
		self.assert_programs_run([
			( 'ljust_rjust', '''
def main() -> i32:
	if 'hi'.ljust( 5 ) != 'hi   ':
		return 1
	if 'hi'.rjust( 5 ) != '   hi':
		return 2
	if 'hi'.ljust( 5, '*' ) != 'hi***':
		return 3
	if 'hi'.rjust( 5, '*' ) != '***hi':
		return 4
	if 'toolong'.ljust( 3 ) != 'toolong':
		return 5
	if 'toolong'.rjust( 3 ) != 'toolong':
		return 6
	if 'exact'.ljust( 5 ) != 'exact':
		return 7
	return 0
''' ),
			( 'zfill', '''
def main() -> i32:
	if '42'.zfill( 5 ) != '00042':
		return 1
	if '-42'.zfill( 5 ) != '-0042':
		return 2
	if '+42'.zfill( 5 ) != '+0042':
		return 3
	if '42'.zfill( 1 ) != '42':
		return 4
	if '42'.zfill( 2 ) != '42':
		return 5
	return 0
''' ),
		] )

	@unittest.skipUnless( _CC is not None, 'no C compiler (clang/gcc/msvc) found - skipping' )
	def test_ljust_rjust_bad_fillchar_panics( self ) -> None:
		self._run( '''
def main() -> i32:
	'hi'.ljust( 5, 'ab' )
	return 0
''' )
		self.assertEqual( self.discovery.errors.errors, [] )
		self._assert_compiles_and_runs( emitter_c.emit_c( self.compiler ), expected_exit = 1 )


class StrPhase3ClassificationTests( test_support.RealCompileMixin, CompilerTestCase ):
	''' Phase 3 of TODO.txt's str-methods plan: OS-native Unicode
	classification (__str.py's is_alpha_cp/is_digit_cp/is_space_cp/
	is_upper_cp/is_lower_cp/is_alnum_cp/is_printable_cp, each a real
	GetStringTypeW/iswalpha_l-family call, not an ASCII fallback) and the
	str methods built on them, plus strip()/lstrip()/rstrip() (whitespace-
	only - see this file's own comment on the str|None compiler gap that
	blocks a chars= form for now). Includes at least one non-ASCII case
	per classification method, to actually exercise the OS-native path
	rather than just ASCII-range-looking input a broken implementation
	could also pass by accident. '''

	def setUp( self ) -> None:
		self.discovery = Discovery( import_builtins = True )
		self.compiler = Compiler( self.discovery )

	@unittest.skipUnless( test_support.HAS_CC, 'no C compiler (clang/gcc/msvc) found - skipping' )
	def test_programs_compile_and_run( self ) -> None:
		''' every real compile-and-run program in this class, merged into a
		single executable (one build for the whole class); a nonzero exit is
		decoded back to the failing sub-program and its own return code. '''
		self.assert_programs_run([
			( 'isalpha', '''
def main() -> i32:
	if not 'hello'.isalpha():
		return 1
	if 'hello world'.isalpha():
		return 2
	if 'hello1'.isalpha():
		return 3
	if ''.isalpha():
		return 4
	if not 'café'.isalpha():
		return 5
	return 0
''' ),
			( 'isdigit_isdecimal_isnumeric', '''
def main() -> i32:
	if not '12345'.isdigit():
		return 1
	if '123a5'.isdigit():
		return 2
	if ''.isdigit():
		return 3
	if not '12345'.isdecimal():
		return 4
	if not '12345'.isnumeric():
		return 5
	return 0
''' ),
			( 'isspace', '''
def main() -> i32:
	if not '   \\t\\n'.isspace():
		return 1
	if ' a '.isspace():
		return 2
	if ''.isspace():
		return 3
	return 0
''' ),
			( 'isupper_islower', '''
def main() -> i32:
	if not 'HELLO'.isupper():
		return 1
	if 'Hello'.isupper():
		return 2
	if not 'HELLO123'.isupper():
		return 3
	if '123'.isupper():
		return 4
	if not 'hello'.islower():
		return 5
	if 'Hello'.islower():
		return 6
	if not 'ÉCOLE'.isupper():
		return 7
	if not 'école'.islower():
		return 8
	return 0
''' ),
			( 'isalnum', '''
def main() -> i32:
	if not 'abc123'.isalnum():
		return 1
	if 'abc 123'.isalnum():
		return 2
	if ''.isalnum():
		return 3
	return 0
''' ),
			( 'isprintable', '''
def main() -> i32:
	if not 'hello world'.isprintable():
		return 1
	if not ''.isprintable():
		return 2
	if not 'café'.isprintable():
		return 3
	if '\\n'.isprintable():
		return 4
	return 0
''' ),
			( 'isidentifier', '''
def main() -> i32:
	if not 'valid_name'.isidentifier():
		return 1
	if not '_leading'.isidentifier():
		return 2
	if '1starts_with_digit'.isidentifier():
		return 3
	if 'has space'.isidentifier():
		return 4
	if ''.isidentifier():
		return 5
	if not 'name123'.isidentifier():
		return 6
	return 0
''' ),
			( 'strip_lstrip_rstrip', '''
def main() -> i32:
	if '  hi  '.strip() != 'hi':
		return 1
	if '  hi  '.lstrip() != 'hi  ':
		return 2
	if '  hi  '.rstrip() != '  hi':
		return 3
	if '\\t\\n hi \\t\\n'.strip() != 'hi':
		return 4
	if 'noleadingortrailing'.strip() != 'noleadingortrailing':
		return 5
	if '     '.strip() != '':
		return 6
	if ''.strip() != '':
		return 7
	return 0
''' ),
			# the real chars: str|None = None parameter - this is TODO.txt's
			# own former "union disambiguation" blocker note (strip's chars=
			# form couldn't get a value back OUT of a str|None parameter),
			# resolved now that real union narrowing/extraction exists (see
			# PLAN_MATCH_NARROWING's own capstone) - _should_strip_cp
			# (lib/builtins/__init__.py) uses `match chars:` internally
			( 'strip_lstrip_rstrip_chars_argument', '''
def main() -> i32:
	if 'xxhixx'.strip( 'x' ) != 'hi':
		return 1
	if 'xxhixx'.lstrip( 'x' ) != 'hixx':
		return 2
	if 'xxhixx'.rstrip( 'x' ) != 'xxhi':
		return 3
	if 'ab-hi-ba'.strip( 'ab-' ) != 'hi':
		return 4
	if 'hi'.strip( 'x' ) != 'hi':
		return 5
	if 'xxx'.strip( 'x' ) != '':
		return 6
	# explicit chars=None must match the no-argument (whitespace) form
	if '  hi  '.strip( None ) != 'hi':
		return 7
	return 0
''' ),
		] )


class StrPhase4CaseCompositeTests( test_support.RealCompileMixin, CompilerTestCase ):
	''' Phase 4 of TODO.txt's str-methods plan: swapcase()/title()/
	istitle() (lib/builtins/__init__.py), built on __str.py's new
	case_map_one primitive (a real, new small-buffer LCMapStringEx call on
	Windows - swapcase()'s own Windows path is the one genuinely new piece
	of Win32 plumbing in this whole plan, so it gets its own explicit
	test, not just incidental coverage through title()). '''

	def setUp( self ) -> None:
		self.discovery = Discovery( import_builtins = True )
		self.compiler = Compiler( self.discovery )

	@unittest.skipUnless( test_support.HAS_CC, 'no C compiler (clang/gcc/msvc) found - skipping' )
	def test_programs_compile_and_run( self ) -> None:
		''' every real compile-and-run program in this class, merged into a
		single executable (one build for the whole class); a nonzero exit is
		decoded back to the failing sub-program and its own return code. '''
		self.assert_programs_run([
			( 'swapcase_ascii', '''
def main() -> i32:
	if 'Hello World'.swapcase() != 'hELLO wORLD':
		return 1
	if '12345'.swapcase() != '12345':
		return 2
	if ''.swapcase() != '':
		return 3
	return 0
''' ),
			# exercises the new per-codepoint Windows LCMapStringEx path
			# (case_map_one) directly - the one genuinely new piece of Win32
			# plumbing this whole phase adds
			( 'swapcase_non_ascii', '''
def main() -> i32:
	if 'café'.swapcase() != 'CAFÉ':
		return 1
	if 'ÉCOLE'.swapcase() != 'école':
		return 2
	return 0
''' ),
			( 'title', '''
def main() -> i32:
	if 'hello world'.title() != 'Hello World':
		return 1
	if "they're bill's".title() != "They'Re Bill'S":
		return 2
	if ''.title() != '':
		return 3
	if 'ALREADY UPPER'.title() != 'Already Upper':
		return 4
	return 0
''' ),
			( 'istitle', '''
def main() -> i32:
	if not 'Hello World'.istitle():
		return 1
	if 'Hello world'.istitle():
		return 2
	if 'HELLO WORLD'.istitle():
		return 3
	if ''.istitle():
		return 4
	if '123'.istitle():
		return 5
	if not 'Abc123'.istitle():
		return 6
	return 0
''' ),
		] )


class EarlyReturnFromLoopWithLiveRCLocalTests( test_support.RealCompileMixin, CompilerTestCase ):
	''' cfg.py's current_epilogue_label() shared-ladder optimization used to
	assume every entry on the epilogue stack survives to the function's own
	real end, where build_epilogue_ladder() walks it and emits each entry's
	own label. Not true for a plain RC local declared INSIDE a loop body -
	restore() (called once per loop, after the body's fully lowered) drops
	it silently once the body's own lowering is done, well before the
	function's real end - so an early `return` reached DURING that body's
	lowering, while the entry was still the topmost active one, could
	capture a label that never actually gets emitted ("undeclared
	identifier" in the generated C). Found while verifying list[str]/str.
	split() end-to-end (both exercise a while loop with a live str local
	inside a with-block) - confirmed independent of list[T]/str via the
	minimal repro below (no generics, no str methods beyond ==). '''

	def setUp( self ) -> None:
		self.discovery = Discovery( import_builtins = True )
		self.compiler = Compiler( self.discovery )

	@unittest.skipUnless( test_support.HAS_CC, 'no C compiler (clang/gcc/msvc) found - skipping' )
	def test_programs_compile_and_run( self ) -> None:
		''' every real compile-and-run program in this class, merged into a
		single executable (one build for the whole class); a nonzero exit is
		decoded back to the failing sub-program and its own return code. '''
		self.assert_programs_run([
			# the early return is never actually reached at runtime (v is
			# always 'item') - this is purely a "does it even compile, and
			# does the untaken branch not corrupt the normal exit" check
			( 'early_return_past_a_loop_confined_rc_local_never_taken', '''
def main() -> i32:
	j: usize = 0
	with compiler.panic_arithmetic( 'overflow' ):
		while j < 20:
			v: str = 'item'
			if v != 'item':
				return 2
			j += 1
	return 0
''' ),
			# break (not return) reaching the SAME loop-confined entry - a
			# different code path (unwind_to(), not current_epilogue_label())
			# that this fix must not have disturbed
			( 'break_past_a_loop_confined_rc_local_still_works', '''
def main() -> i32:
	j: usize = 0
	with compiler.panic_arithmetic( 'overflow' ):
		while j < 20:
			v: str = 'item'
			if j == 5:
				break
			j += 1
	if j != 5:
		return 1
	return 0
''' ),
		] )

	@unittest.skipUnless( _CC is not None, 'no C compiler (clang/gcc/msvc) found - skipping' )
	def test_early_return_past_a_loop_confined_rc_local_actually_taken( self ) -> None:
		# this time the early return DOES fire (at j == 5) - proves the
		# inline unwind path is correct, not just non-crashing: the real
		# exit code has to come back through it
		self._run( '''
def main() -> i32:
	j: usize = 0
	with compiler.panic_arithmetic( 'overflow' ):
		while j < 20:
			v: str = 'item'
			if j == 5:
				return 2
			j += 1
	return 0
''' )
		self.assertEqual( self.discovery.errors.errors, [] )
		self._assert_compiles_and_runs( emitter_c.emit_c( self.compiler ), expected_exit = 2 )

	@unittest.skipUnless( _CC is not None, 'no C compiler (clang/gcc/msvc) found - skipping' )
	def test_early_return_past_both_an_outer_and_a_loop_confined_rc_local( self ) -> None:
		# a function-scoped RC local (outer, pushed before the loop) AND a
		# loop-confined one (inner, pushed inside it) both still live at
		# the same early return - both must actually get decref'd, not
		# just whichever one this fix was specifically chasing
		self._run( '''
def main() -> i32:
	outer: str = 'outer'
	j: usize = 0
	with compiler.panic_arithmetic( 'overflow' ):
		while j < 20:
			inner: str = 'inner'
			if j == 5:
				if outer != 'outer' or inner != 'inner':
					return 9
				return 2
			j += 1
	return 0
''' )
		self.assertEqual( self.discovery.errors.errors, [] )
		self._assert_compiles_and_runs( emitter_c.emit_c( self.compiler ), expected_exit = 2 )


class CapturedThenCancelledEpilogueEntryTests( test_support.RealCompileMixin, CompilerTestCase ):
	''' regression tests for a real bug in cfg.py's shared epilogue-ladder
	mechanism, independent of generators/defer/errdefer - pure ordinary
	function lowering. Shape: a function-scoped RC local, TWO early `return`
	statements (an ordinary call sandwiched between them) that both jump
	into the SAME shared epilogue label while the local is still live, then
	an explicit compiler.decref() of that same local, then the function's
	own final `return`.

	current_epilogue_label() correctly hands both early returns the shared
	label (used_shared_epilogue_label() records that a real jump was
	committed). But compiler.decref() then called cfg.py's
	manually_decreffed(), which used to just set the entry's own
	`cancelled` flag unconditionally - and build_epilogue_ladder() (called
	once, at the function's real end) bakes each entry's FINAL cancelled
	state into every jump site sharing it, not the state each jump site
	actually saw when it was emitted. The shared label still got built
	(used_shared_epilogue_label() already covers the "undeclared label"
	failure mode), but empty - silently dropping the decref BOTH early
	returns were relying on it for, a permanent refcount leak on every path
	that takes an early return before the manual decref ever runs.

	Fixed in cfg.py by tracking whether an entry's label was ever handed
	out while still live (Epilogue.captured), and converting a cancellation
	of a captured entry into a genuine runtime flag-guard (_neutralize())
	instead of a compile-time-only cancel - mirroring how errdefer's own
	flag-guarded replay already works, just armed by default instead of
	disarmed. '''

	def setUp( self ) -> None:
		self.discovery = Discovery( import_builtins = True )
		self.compiler = Compiler( self.discovery )

	@unittest.skipUnless( _CC is not None, 'no C compiler (clang/gcc/msvc) found - skipping' )
	def test_two_early_returns_then_manual_decref_compiles_and_runs( self ) -> None:
		# the exact reported repro: two early-return refcount checks around
		# an ordinary call, then an explicit compiler.decref() balancing the
		# construction, then a final return - neither early-return check
		# actually fires here (refcount stays 1 throughout), so this is
		# first and foremost a "does it even compile" check (a naive fix
		# attempt can leave the shared label's own goto undeclared)
		self._run( '''
class Box:
	v: i32
	def __init__( self, v: i32 ) -> None:
		self.v = v

def helper( b: Box ) -> None:
	return

def main() -> i32:
	with compiler.wrap_arithmetic:
		b = Box( v = 1 )
		if compiler.refcount( b ) != 1:
			return 1
		helper( b )
		if compiler.refcount( b ) != 1:
			return 2
		compiler.decref( b )
		return 0
''' )
		self.assertEqual( self.discovery.errors.errors, [] )
		self._assert_compiles_and_runs( emitter_c.emit_c( self.compiler ), expected_exit = 0 )

	@unittest.skipUnless( _CC is not None, 'no C compiler (clang/gcc/msvc) found - skipping' )
	def test_one_early_return_then_manual_decref_compiles_and_runs( self ) -> None:
		# the simpler, single-early-return variant of the same shape - used
		# to compile fine (only the two-return shape risked an undeclared
		# label) but silently emitted an EMPTY shared epilogue label,
		# dropping the early return's own decref of `b` - a real refcount
		# leak with no compile-time symptom at all
		self._run( '''
class Box:
	v: i32
	def __init__( self, v: i32 ) -> None:
		self.v = v

def helper( b: Box ) -> None:
	return

def main() -> i32:
	with compiler.wrap_arithmetic:
		b = Box( v = 1 )
		if compiler.refcount( b ) != 1:
			return 1
		helper( b )
		compiler.decref( b )
		return 0
''' )
		self.assertEqual( self.discovery.errors.errors, [] )
		self._assert_compiles_and_runs( emitter_c.emit_c( self.compiler ), expected_exit = 0 )

	@unittest.skipUnless( _CC is not None, 'no C compiler (clang/gcc/msvc) found - skipping' )
	def test_captured_early_returns_still_release_after_later_manual_decref( self ) -> None:
		# proves the leak is actually fixed, not just "still compiles" -
		# `leaky` reproduces the exact reported shape (two early-return
		# checks around an ordinary call, then compiler.decref(), then a
		# final return) with a copy[Box] PARAMETER instead of a freshly
		# constructed local, specifically so main() keeps its own,
		# independently-observable reference to the same underlying object
		# and can check its EXACT refcount right after each call - a leak
		# on the early-return paths (the entry captured by both `if`
		# statements never actually getting released once compiler.decref()
		# cancelled it) shows up directly as compiler.refcount(b) != 1
		# after a call that was supposed to have released its own copy.
		# Exercises `trigger` == 1 (first early return), == 2 (second early
		# return), and == 0 (falls through to the manual decref itself) in
		# turn, so all three paths sharing the one epilogue label are each
		# checked independently
		self._run( '''
class Box:
	v: i32
	def __init__( self, v: i32 ) -> None:
		self.v = v

def helper( b: Box ) -> None:
	return

def leaky( b: copy[Box], trigger: i32 ) -> i32:
	if trigger == 1:
		return 1
	helper( b )
	if trigger == 2:
		return 2
	compiler.decref( b )
	return 0

def main() -> i32:
	b: Box = Box( v = 1 )
	if compiler.refcount( b ) != 1:
		return 1
	r1: i32 = leaky( b, 1 )
	if r1 != 1:
		return 2
	if compiler.refcount( b ) != 1:
		return 3
	r2: i32 = leaky( b, 2 )
	if r2 != 2:
		return 4
	if compiler.refcount( b ) != 1:
		return 5
	r3: i32 = leaky( b, 0 )
	if r3 != 0:
		return 6
	if compiler.refcount( b ) != 1:
		return 7
	compiler.decref( b )
	return 0
''' )
		self.assertEqual( self.discovery.errors.errors, [] )
		self._assert_compiles_and_runs( emitter_c.emit_c( self.compiler ), expected_exit = 0 )


class DeferAsLastStatementOfNonTerminatingIfBranchTests( test_support.RealCompileMixin, CompilerTestCase ):
	''' regression tests for a real compile error in lowering.py's
	_stmt_diverges: `defer(...)`/`errdefer(...)` used as the LAST statement
	of an if-branch that doesn't itself terminate (no return/break/continue,
	just falls through to whatever comes after the if) failed to compile
	with "name 'defer' is not defined".

	_stmt_If calls _stmt_diverges on a branch's last statement to decide
	whether that branch's own ending state can reach the if's join point at
	all (used for narrowing/bindings merge - see _stmt_diverges' own
	docstring). _stmt_diverges recognizes a bare call-expression statement
	and tries to resolve its callee to check for a -> NoReturn return type
	(catching sys.panic()-shaped branches) - but defer/errdefer are never
	real, resolvable callables; they're recognized purely by AST shape
	(_stmt_Expr, before ordinary call resolution ever runs - see this
	file's own module docstring on defer/errdefer). _stmt_diverges didn't
	share that recognition, so `defer(expr)`/`errdefer(expr)` as a branch's
	own last statement fell through to _resolve_callee_target, which tried
	to look up the bare name 'defer' as an ordinary function and failed
	outright.

	Found via lib/builtins/__str.py's case_map (POSIX target): `if loc is
	not None: defer(freelocale(loc))` is the only defer call in the whole
	codebase shaped exactly this way - every other call site either sits at
	a function's top level or right after an early-return guard clause
	(`if loc is None: return False` then an UNCONDITIONAL defer below it),
	never as an if-branch's own last statement - which is why this went
	unnoticed until real POSIX-target compilation (WSL/gcc, not just
	Windows/clang/MSVC) actually exercised that code path for the first
	time. Fixed by having _stmt_diverges recognize the defer/errdefer AST
	shape (the same check _stmt_Expr already uses) and treat it as non-
	diverging - defer/errdefer always falls through (arms a flag, never
	itself returns/panics), never NoReturn-shaped. '''
	def setUp( self ) -> None:
		self.discovery = Discovery( import_builtins = True )
		self.compiler = Compiler( self.discovery )

	@unittest.skipUnless( _CC is not None, 'no C compiler (clang/gcc/msvc) found - skipping' )
	def test_defer_as_only_statement_of_non_terminating_if_branch( self ) -> None:
		# proves the fix is real, not just "compiles": bump() must actually
		# run when the defer was armed (should_arm=True), and must NOT run
		# a second time when it wasn't (should_arm=False) - a Ptr[i32]
		# out-param makes the deferred call's own side effect directly
		# observable from main()
		self._run( '''
def bump( counter: Ptr[i32] ) -> None:
	with compiler.wrap_arithmetic:
		counter[0] = counter[0] + 1

def maybe_defer( should_arm: bool, counter: Ptr[i32] ) -> None:
	if should_arm:
		defer( bump( counter ))
	return

def main() -> i32:
	n: i32 = 0
	maybe_defer( True, compiler.addrof( n ))
	if n != 1:
		return 1
	maybe_defer( False, compiler.addrof( n ))
	if n != 1:
		return 2
	return 0
''' )
		self.assertEqual( self.discovery.errors.errors, [] )
		self._assert_compiles_and_runs( emitter_c.emit_c( self.compiler ), expected_exit = 0 )

	@unittest.skipUnless( _CC is not None, 'no C compiler (clang/gcc/msvc) found - skipping' )
	def test_errdefer_as_only_statement_of_non_terminating_if_branch( self ) -> None:
		self._run( _RESULT_FIXTURE + '\n' + '''
def bump( counter: Ptr[i32] ) -> None:
	with compiler.wrap_arithmetic:
		counter[0] = counter[0] + 1

def maybe_errdefer( should_arm: bool, fail: bool, counter: Ptr[i32] ) -> Result[i32,OverflowError]:
	if should_arm:
		errdefer( bump( counter ))
	if fail:
		return Result.Err( OverflowError() )
	return Result.Ok( 5 )

def main() -> i32:
	n: i32 = 0
	r1: Result[i32,OverflowError] = maybe_errdefer( True, True, compiler.addrof( n ))
	if not r1.is_err():
		return 1
	if n != 1:
		return 2
	r2: Result[i32,OverflowError] = maybe_errdefer( True, False, compiler.addrof( n ))
	if not r2.is_ok():
		return 3
	if n != 1:
		return 4
	return 0
''' )
		self.assertEqual( self.discovery.errors.errors, [] )
		self._assert_compiles_and_runs( emitter_c.emit_c( self.compiler ), expected_exit = 0 )

	@unittest.skipUnless( _CC is not None, 'no C compiler (clang/gcc/msvc) found - skipping' )
	def test_str_upper_matches_the_exact_reported_shape( self ) -> None:
		# the ACTUAL originally-failing code (lib/builtins/__str.py's
		# case_map, minus the POSIX-only towupper_l plumbing this test
		# doesn't need): `if loc is not None: defer(freelocale(loc))` -
		# same "if <cond>: defer(...)" shape with nothing after the if,
		# confirming the fix covers the real repro, not just a synthetic
		# stand-in
		self._run( '''
def maybe_release( loc: Ptr[None], counter: Ptr[i32] ) -> None:
	if loc is not None:
		defer( bump( counter ))
	return

def bump( counter: Ptr[i32] ) -> None:
	with compiler.wrap_arithmetic:
		counter[0] = counter[0] + 1

def main() -> i32:
	n: i32 = 0
	sentinel: i32 = 0
	maybe_release( compiler.addrof( sentinel ), compiler.addrof( n ))
	if n != 1:
		return 1
	maybe_release( None, compiler.addrof( n ))
	if n != 1:
		return 2
	return 0
''' )
		self.assertEqual( self.discovery.errors.errors, [] )
		self._assert_compiles_and_runs( emitter_c.emit_c( self.compiler ), expected_exit = 0 )


class GUIDTests( test_support.RealCompileMixin, CompilerTestCase ):
	''' lib/guid.py's GUID type - PLAN_SUBCLASSING_VTABLES_COM.md's Phase 3
	(COM specifics). Needs import_builtins=True (str.split, list[str]). '''

	def setUp( self ) -> None:
		self.discovery = Discovery( import_builtins = True )
		self.compiler = Compiler( self.discovery )

	@unittest.skipUnless( test_support.HAS_CC, 'no C compiler (clang/gcc/msvc) found - skipping' )
	def test_programs_compile_and_run( self ) -> None:
		''' every real compile-and-run program in this class, merged into a
		single executable (one build for the whole class); a nonzero exit is
		decoded back to the failing sub-program and its own return code. '''
		self.assert_programs_run([
			( 'from_str_parses_each_field_correctly', '''
import guid

def main() -> i32:
	g: guid.GUID = guid.GUID.from_str( 'deadbeef-dead-beef-dead-beefdeadbeef' )
	if g.data1 != 0xdeadbeef:
		return 1
	if g.data2 != 0xdead:
		return 2
	if g.data3 != 0xbeef:
		return 3
	if g.data4_0 != 0xde or g.data4_1 != 0xad:
		return 4
	if g.data4_2 != 0xbe or g.data4_3 != 0xef or g.data4_4 != 0xde or g.data4_5 != 0xad or g.data4_6 != 0xbe or g.data4_7 != 0xef:
		return 5
	return 0
''' ),
			( 'eq_ne_compare_by_value', '''
import guid

def main() -> i32:
	a: guid.GUID = guid.GUID.from_str( 'deadbeef-dead-beef-dead-beefdeadbeef' )
	b: guid.GUID = guid.GUID.from_str( 'deadbeef-dead-beef-dead-beefdeadbeef' )
	c: guid.GUID = guid.GUID.from_str( '00000000-0000-0000-0000-000000000000' )
	if a != b:
		return 1
	if not ( a != c ):
		return 2
	if a == c:
		return 3
	return 0
''' ),
		] )


class DictTests( test_support.RealCompileMixin, CompilerTestCase ):
	''' dict[K,V] (lib/builtins/__init__.py's own dict class + lib/builtins/
	__RawDict.py's RawDict/RawEntry/RawIndex) end-to-end - see
	PLAN_CALLABLE.md: the whole point of building Callable[...]/indirect
	calls was to let RawDict stay genuinely type-erased (never branches on
	RC-ness, never decodes a key_ptr/value_ptr) while still comparing keys
	via a real function pointer supplied by the monomorphized dict[K,V].
	Never compiled anywhere before this. Mirrors ListGenericTests' own
	import_builtins=True + real compile-and-run convention. '''

	def setUp( self ) -> None:
		self.discovery = Discovery( import_builtins = True )
		self.compiler = Compiler( self.discovery )

	@unittest.skipUnless( test_support.HAS_CC, 'no C compiler (clang/gcc/msvc) found - skipping' )
	def test_programs_compile_and_run( self ) -> None:
		''' every real compile-and-run program in this class, merged into a
		single executable (one build for the whole class); a nonzero exit is
		decoded back to the failing sub-program and its own return code. '''
		self.assert_programs_run([
			( 'str_key_insert_and_lookup', '''
def main() -> i32:
	d: dict[str, i32] = dict[str, i32]()
	d[ 'a' ] = 1
	d[ 'b' ] = 2
	ra: Result[i32,KeyError] = d.__getitem__( 'a' )
	rb: Result[i32,KeyError] = d.__getitem__( 'b' )
	if ra.is_err() or rb.is_err():
		return 8
	va: i32 = ra.unwrap( 'missing a' )
	vb: i32 = rb.unwrap( 'missing b' )
	with compiler.wrap_arithmetic:
		diff: i32 = ( va - 1 ) + ( vb - 2 )
	return diff
''' ),
			( 'overwrite_existing_key_replaces_value', '''
def main() -> i32:
	d: dict[str, i32] = dict[str, i32]()
	d[ 'a' ] = 1
	d[ 'a' ] = 100
	if d.__len__() != 1:
		return 1
	r: Result[i32,KeyError] = d.__getitem__( 'a' )
	if r.is_err():
		return 8
	if r.unwrap( 'missing a' ) != 100:
		return 2
	return 0
''' ),
			( 'missing_key_returns_key_error', '''
def main() -> i32:
	d: dict[str, i32] = dict[str, i32]()
	d[ 'a' ] = 1
	r: Result[i32,KeyError] = d.__getitem__( 'nope' )
	if r.is_ok():
		return 1
	return 0
''' ),
			# the other K/V combination - a plain value-typed key (byte-hashed
			# via _fnv1a_hash, no __hash__ method needed) paired with an RC
			# value, the mirror image of str-keyed dict[str,i32] above
			( 'non_rc_key_i32_with_rc_value_str', '''
def main() -> i32:
	d: dict[i32, str] = dict[i32, str]()
	d[ 7 ] = 'seven'
	d[ 9 ] = 'nine'
	if d.__len__() != 2:
		return 1
	r7: Result[str,KeyError] = d.__getitem__( 7 )
	r9: Result[str,KeyError] = d.__getitem__( 9 )
	if r7.is_err() or r9.is_err():
		return 8
	if r7.unwrap( 'missing 7' ) != 'seven':
		return 2
	if r9.unwrap( 'missing 9' ) != 'nine':
		return 3
	return 0
''' ),
			# 50 distinct keys forces list[T]'s own growth (both __entries and
			# __indices) and exercises RawDict's binary search over a real
			# range, not just a handful of entries
			( 'many_entries_forces_growth_and_stays_correct', '''
def main() -> i32:
	d: dict[i32, i32] = dict[i32, i32]()
	i: usize = 0
	with compiler.wrap_arithmetic:
		while i < 50:
			key: i32 = compiler.cast( i32, i )
			val: i32 = key * 2
			d[ key ] = val
			i += 1
	if d.__len__() != 50:
		return 1
	j: usize = 0
	with compiler.wrap_arithmetic:
		while j < 50:
			key: i32 = compiler.cast( i32, j )
			r: Result[i32,KeyError] = d.__getitem__( key )
			if r.is_err():
				return 2
			if r.unwrap( 'x' ) != key * 2:
				return 3
			j += 1
	return 0
''' ),
			# str keys AND str values together, insert/overwrite/destroy - a
			# proxy for correct incref/decref bookkeeping: wrong refcounting
			# here would double-free or leak, and a double-free would crash
			# the process (nonzero/abnormal exit), not just misbehave quietly
			( 'rc_key_and_rc_value_destruction_does_not_crash', '''
def main() -> i32:
	d: dict[str, str] = dict[str, str]()
	d[ 'a' ] = 'apple'
	d[ 'b' ] = 'banana'
	d[ 'a' ] = 'avocado'
	r: Result[str,KeyError] = d.__getitem__( 'a' )
	if r.is_err():
		return 8
	if r.unwrap( 'missing a' ) != 'avocado':
		return 1
	return 0
''' ),
			# key_at/value_at - positional access into insertion order
			# (__entries is append-only with no removal, so index i always
			# names the i-th inserted entry)
			( 'key_at_value_at_positional_access_in_insertion_order', '''
def main() -> i32:
	d: dict[str, i32] = dict[str, i32]()
	d[ 'a' ] = 1
	d[ 'b' ] = 2
	d[ 'c' ] = 3
	k0: Result[str,IndexError] = d.key_at( 0 )
	v0: Result[i32,IndexError] = d.value_at( 0 )
	k2: Result[str,IndexError] = d.key_at( 2 )
	v2: Result[i32,IndexError] = d.value_at( 2 )
	if k0.is_err() or v0.is_err() or k2.is_err() or v2.is_err():
		return 8
	if k0.unwrap( 'x' ) != 'a' or v0.unwrap( 'x' ) != 1:
		return 1
	if k2.unwrap( 'x' ) != 'c' or v2.unwrap( 'x' ) != 3:
		return 2
	return 0
''' ),
			( 'key_at_value_at_out_of_bounds_returns_index_error', '''
def main() -> i32:
	d: dict[str, i32] = dict[str, i32]()
	d[ 'a' ] = 1
	if d.key_at( 1 ).is_ok():
		return 1
	if d.value_at( 1 ).is_ok():
		return 2
	if d.key_at( 0 ).is_err():
		return 3
	return 0
''' ),
			# overwriting an existing key must not shift/duplicate its
			# position - key_at(0) stays 'a', len stays 1
			( 'key_at_value_at_after_overwrite_keeps_same_position', '''
def main() -> i32:
	d: dict[str, i32] = dict[str, i32]()
	d[ 'a' ] = 1
	d[ 'a' ] = 100
	if d.__len__() != 1:
		return 1
	k0: Result[str,IndexError] = d.key_at( 0 )
	v0: Result[i32,IndexError] = d.value_at( 0 )
	if k0.is_err() or v0.is_err():
		return 8
	if k0.unwrap( 'x' ) != 'a' or v0.unwrap( 'x' ) != 100:
		return 2
	return 0
''' ),
			# RC key AND RC value, repeatedly borrowed out via key_at/value_at
			# then dropped - a proxy for correct incref/decref bookkeeping,
			# same posture as rc_key_and_rc_value_destruction_does_not_crash
			# above (wrong refcounting here would double-free or leak)
			( 'key_at_value_at_rc_key_and_value_does_not_crash', '''
def main() -> i32:
	d: dict[str, str] = dict[str, str]()
	d[ 'a' ] = 'apple'
	d[ 'b' ] = 'banana'
	i: usize = 0
	with compiler.wrap_arithmetic:
		while i < 2:
			k: Result[str,IndexError] = d.key_at( i )
			v: Result[str,IndexError] = d.value_at( i )
			if k.is_err() or v.is_err():
				return 8
			ks: str = k.unwrap( 'x' )
			vs: str = v.unwrap( 'x' )
			i += 1
	return 0
''' ),
			# __contains__ - RC key (str), found and missing
			( 'contains_found_and_missing_rc_key', '''
def main() -> i32:
	d: dict[str, i32] = dict[str, i32]()
	d[ 'a' ] = 1
	if not d.__contains__( 'a' ):
		return 1
	if d.__contains__( 'nope' ):
		return 2
	return 0
''' ),
			# __contains__ - non-RC key (i32), found and missing
			( 'contains_found_and_missing_non_rc_key', '''
def main() -> i32:
	d: dict[i32, str] = dict[i32, str]()
	d[ 7 ] = 'seven'
	if not d.__contains__( 7 ):
		return 1
	if d.__contains__( 8 ):
		return 2
	return 0
''' ),
			# __delitem__ on a missing key returns Err, and leaves the dict untouched
			( 'delitem_missing_key_returns_key_error', '''
def main() -> i32:
	d: dict[str, i32] = dict[str, i32]()
	d[ 'a' ] = 1
	r: Result[None,KeyError] = d.__delitem__( 'nope' )
	if r.is_ok():
		return 1
	if d.__len__() != 1:
		return 2
	return 0
''' ),
			# __delitem__ removing the MIDDLE entry of a real hash-collision
			# bucket - CollidingKey.__hash__ always returns the same value,
			# forcing every insert into one bucket, so this directly
			# exercises RawDict.remove_entry/_fixup_indices_after_removal's
			# collision-scan + entry_idx renumbering, not just the common
			# no-collision case
			( 'delitem_middle_of_hash_collision_bucket', '''
class CollidingKey:
	value: i32
	def __init__( self, value: i32 ) -> None:
		self.value = value
	def __hash__( self ) -> u64:
		return 42
	def __eq__( self, other: CollidingKey ) -> bool:
		return self.value == other.value

def main() -> i32:
	d: dict[CollidingKey, i32] = dict[CollidingKey, i32]()
	d[ CollidingKey( 0 ) ] = 100
	d[ CollidingKey( 1 ) ] = 200
	d[ CollidingKey( 2 ) ] = 300
	if d.__len__() != 3:
		return 1
	r: Result[None,KeyError] = d.__delitem__( CollidingKey( 1 ) )
	if r.is_err():
		return 2
	if d.__len__() != 2:
		return 3
	if d.__contains__( CollidingKey( 1 ) ):
		return 4
	r0: Result[i32,KeyError] = d.__getitem__( CollidingKey( 0 ) )
	r2: Result[i32,KeyError] = d.__getitem__( CollidingKey( 2 ) )
	if r0.is_err() or r2.is_err():
		return 5
	if r0.unwrap( 'x' ) != 100 or r2.unwrap( 'x' ) != 300:
		return 6
	return 0
''' ),
			# RC key AND RC value, repeatedly inserted then deleted - a
			# double-free/leak proxy for __delitem__'s own release path
			# (wrong refcounting here would crash the process, not just
			# misbehave quietly)
			( 'delitem_rc_key_and_rc_value_repeated_does_not_crash', '''
def main() -> i32:
	d: dict[str, str] = dict[str, str]()
	i: usize = 0
	with compiler.wrap_arithmetic:
		while i < 5:
			d[ 'a' ] = 'apple'
			d[ 'b' ] = 'banana'
			ra: Result[None,KeyError] = d.__delitem__( 'a' )
			rb: Result[None,KeyError] = d.__delitem__( 'b' )
			if ra.is_err() or rb.is_err():
				return 1
			if d.__len__() != 0:
				return 2
			i += 1
	return 0
''' ),
			# `in`/`not in` dispatch to __contains__ (lowering.py's
			# _lower_in_comparison) - the reversed receiver/arg order
			( 'in_and_not_in_operator', '''
def main() -> i32:
	d: dict[str, i32] = dict[str, i32]()
	d[ 'a' ] = 1
	if not ( 'a' in d ):
		return 1
	if 'nope' in d:
		return 2
	if not ( 'nope' not in d ):
		return 3
	if 'a' not in d:
		return 4
	return 0
''' ),
		] )


class SetTests( test_support.RealCompileMixin, CompilerTestCase ):
	''' set[T] (lib/builtins/__set.py) - a thin wrapper around dict[T, bool],
	built on top of the __contains__/__delitem__ added to dict[K,V] above.
	Mirrors DictTests' own real compile-and-run convention. '''

	def setUp( self ) -> None:
		self.discovery = Discovery( import_builtins = True )
		self.compiler = Compiler( self.discovery )

	@unittest.skipUnless( test_support.HAS_CC, 'no C compiler (clang/gcc/msvc) found - skipping' )
	def test_programs_compile_and_run( self ) -> None:
		''' every real compile-and-run program in this class, merged into a
		single executable (one build for the whole class); a nonzero exit is
		decoded back to the failing sub-program and its own return code. '''
		self.assert_programs_run([
			( 'add_and_contains_non_rc_element', '''
def main() -> i32:
	s: set[i32] = set[i32]()
	s.add( 7 )
	s.add( 9 )
	if not s.__contains__( 7 ):
		return 1
	if not s.__contains__( 9 ):
		return 2
	if s.__contains__( 8 ):
		return 3
	return 0
''' ),
			( 'add_and_contains_rc_element', '''
def main() -> i32:
	s: set[str] = set[str]()
	s.add( 'apple' )
	s.add( 'banana' )
	if not s.__contains__( 'apple' ):
		return 1
	if not s.__contains__( 'banana' ):
		return 2
	if s.__contains__( 'cherry' ):
		return 3
	return 0
''' ),
			# duplicate add is a no-op, matching Python set.add semantics
			( 'duplicate_add_is_noop', '''
def main() -> i32:
	s: set[i32] = set[i32]()
	s.add( 5 )
	s.add( 5 )
	s.add( 5 )
	if s.__len__() != 1:
		return 1
	if not s.__contains__( 5 ):
		return 2
	return 0
''' ),
			( 'contains_returns_false_for_never_added_value', '''
def main() -> i32:
	s: set[i32] = set[i32]()
	s.add( 1 )
	if s.__contains__( 42 ):
		return 1
	return 0
''' ),
			# discard: no-op on a missing value, actually removes a present one
			( 'discard_present_and_absent_value', '''
def main() -> i32:
	s: set[i32] = set[i32]()
	s.add( 1 )
	s.add( 2 )
	s.discard( 1 )
	if s.__contains__( 1 ):
		return 1
	if not s.__contains__( 2 ):
		return 2
	if s.__len__() != 1:
		return 3
	s.discard( 999 )  # absent - must be a silent no-op, not an error
	if s.__len__() != 1:
		return 4
	return 0
''' ),
			# remove: succeeds on a present value, reports Err on an absent one
			( 'remove_present_and_absent_value', '''
def main() -> i32:
	s: set[i32] = set[i32]()
	s.add( 1 )
	r: Result[None,KeyError] = s.remove( 1 )
	if r.is_err():
		return 1
	if s.__contains__( 1 ):
		return 2
	r2: Result[None,KeyError] = s.remove( 999 )
	if r2.is_ok():
		return 3
	return 0
''' ),
			# 50 distinct elements forces RawDict's own growth path (both
			# __entries and __indices), same rationale as DictTests'
			# many_entries_forces_growth_and_stays_correct
			( 'many_elements_forces_growth_and_stays_correct', '''
def main() -> i32:
	s: set[i32] = set[i32]()
	i: usize = 0
	with compiler.wrap_arithmetic:
		while i < 50:
			s.add( compiler.cast( i32, i ))
			i += 1
	if s.__len__() != 50:
		return 1
	j: usize = 0
	with compiler.wrap_arithmetic:
		while j < 50:
			if not s.__contains__( compiler.cast( i32, j )):
				return 2
			j += 1
	return 0
''' ),
			# RC element (str) add/discard/re-add repeated several times - a
			# double-free/leak proxy, same posture as DictTests'
			# delitem_rc_key_and_rc_value_repeated_does_not_crash
			( 'rc_element_add_discard_repeated_does_not_crash', '''
def main() -> i32:
	s: set[str] = set[str]()
	i: usize = 0
	with compiler.wrap_arithmetic:
		while i < 5:
			s.add( 'apple' )
			s.add( 'banana' )
			if s.__len__() != 2:
				return 1
			s.discard( 'apple' )
			s.discard( 'banana' )
			if s.__len__() != 0:
				return 2
			i += 1
	return 0
''' ),
			# for x in my_set: - proves the __len__ + __getitem__(usize)
			# "indexable" for-loop protocol wiring (lowering.py's
			# _lower_for_over_indexable) actually works for set[T], with no
			# compiler changes of its own. The per-iteration bind desugars
			# to obj[i].or_return() (since __getitem__ returns
			# Result[T,IndexError]), which requires the ENCLOSING function
			# to itself return a Result[_,IndexError]-shaped type - main()
			# returns plain i32 (needed for this test harness's own exit-
			# code dispatch), so the loop lives in a small helper instead,
			# unwrapped by main(). xor-checksum the visited elements
			# against the expected total (order-independent, since sets
			# are unordered) as proof every element was visited exactly once.
			( 'for_loop_over_set_visits_every_element_once', '''
def checksum_set( s: set[i32] ) -> Result[i32, IndexError]:
	checksum: i32 = 0
	with compiler.wrap_arithmetic:
		for x in s:
			checksum = checksum ^ x
	return Result.Ok( checksum )

def main() -> i32:
	s: set[i32] = set[i32]()
	s.add( 1 )
	s.add( 2 )
	s.add( 4 )
	s.add( 8 )
	if s.__len__() != 4:
		return 1
	r: Result[i32,IndexError] = checksum_set( s )
	if r.is_err():
		return 2
	if r.unwrap( 'x' ) != 15:  # 1 ^ 2 ^ 4 ^ 8 == 15
		return 3
	return 0
''' ),
			# `in`/`not in` dispatch to __contains__ (lowering.py's
			# _lower_in_comparison) - the reversed receiver/arg order
			( 'in_and_not_in_operator', '''
def main() -> i32:
	s: set[i32] = set[i32]()
	s.add( 7 )
	if not ( 7 in s ):
		return 1
	if 8 in s:
		return 2
	if not ( 8 not in s ):
		return 3
	if 7 not in s:
		return 4
	return 0
''' ),
			# union/intersection/difference/symmetric_difference, via both
			# the named methods and the operators (|/&/-/^, wired through
			# _BINOP_DUNDER) - a: {1,2,3}, b: {2,3,4}
			( 'set_algebra_named_methods_and_operators', '''
def main() -> i32:
	a: set[i32] = { 1, 2, 3 }
	b: set[i32] = { 2, 3, 4 }

	u: set[i32] = a.union( b )
	if u.__len__() != 4:
		return 1
	if not ( u.__contains__( 1 ) and u.__contains__( 2 ) and u.__contains__( 3 ) and u.__contains__( 4 )):
		return 2

	x: set[i32] = a.intersection( b )
	if x.__len__() != 2:
		return 3
	if not ( x.__contains__( 2 ) and x.__contains__( 3 )):
		return 4

	d: set[i32] = a.difference( b )
	if d.__len__() != 1 or not d.__contains__( 1 ):
		return 5

	sd: set[i32] = a.symmetric_difference( b )
	if sd.__len__() != 2:
		return 6
	if not ( sd.__contains__( 1 ) and sd.__contains__( 4 )):
		return 7

	# same results via the operator forms
	u2: set[i32] = a | b
	if u2.__len__() != 4:
		return 8
	x2: set[i32] = a & b
	if x2.__len__() != 2:
		return 9
	d2: set[i32] = a - b
	if d2.__len__() != 1 or not d2.__contains__( 1 ):
		return 10
	sd2: set[i32] = a ^ b
	if sd2.__len__() != 2:
		return 11

	# neither a nor b was mutated by any of the above
	if a.__len__() != 3 or b.__len__() != 3:
		return 12
	return 0
''' ),
			# __eq__/__ne__ - unordered-set equality (same length + one-
			# directional containment), and _COMP_DUNDER's need for an
			# EXPLICIT __ne__ (never auto-derived from __eq__)
			( 'set_equality_and_inequality', '''
def main() -> i32:
	a: set[i32] = { 1, 2, 3 }
	b: set[i32] = { 3, 2, 1 }  # same members, different insertion order
	c: set[i32] = { 1, 2, 4 }

	if not ( a == b ):
		return 1
	if a != b:
		return 2
	if a == c:
		return 3
	if not ( a != c ):
		return 4

	empty1: set[i32] = set[i32]()
	empty2: set[i32] = set[i32]()
	if not ( empty1 == empty2 ):
		return 5

	# different length alone must be enough to reject equality, even
	# with no element mismatch scanned yet
	small: set[i32] = { 1, 2 }
	if small == a:
		return 6
	return 0
''' ),
			# RC element type (str) through the algebra methods - a
			# double-free/leak proxy, same posture as the earlier RC
			# add/discard repeated-cycle test
			( 'set_algebra_rc_element_does_not_crash', '''
def main() -> i32:
	a: set[str] = { 'a', 'b', 'c' }
	b: set[str] = { 'b', 'c', 'd' }
	u: set[str] = a | b
	x: set[str] = a & b
	d: set[str] = a - b
	sd: set[str] = a ^ b
	if u.__len__() != 4 or x.__len__() != 2 or d.__len__() != 1 or sd.__len__() != 2:
		return 1
	# a itself must be untouched by any of the algebra calls above
	original: set[str] = { 'a', 'b', 'c' }
	if not ( a == original ):
		return 2
	return 0
''' ),
		] )


class DictThreadSafetyTests( test_support.RealCompileMixin, CompilerTestCase ):
	''' dict[K,V] (lib/builtins/__init__.py) is now locked by default (a
	real FastLock, acquired/released around every method) - same split as
	list[T]/UnsafeList[T] (see ListThreadSafetyTests above). These are the
	real compile+run stress tests that actually exercise concurrent
	access; DictTests above already covers single-threaded behavior and
	still passes unchanged against the new locked wrapper. subprocess
	timeout matches ListThreadSafetyTests, for the same reason (a real
	hang should fail loudly, not wedge the suite). '''
	def setUp( self ) -> None:
		self.discovery = Discovery( import_builtins = True )
		self.compiler = Compiler( self.discovery )

	@unittest.skipUnless( test_support.HAS_CC, 'no C compiler (clang/gcc/msvc) found - skipping' )
	def test_programs_compile_and_run( self ) -> None:
		''' every real compile-and-run program in this class, merged into a
		single executable (one build for the whole class); a nonzero exit is
		decoded back to the failing sub-program and its own return code. '''
		self.assert_programs_run([
			( 'concurrent_insert_disjoint_keys_from_8_threads', '''
import threading

class DictWriter:
	target: dict[i32,i32]
	base: i32

	@staticmethod
	def make( target: dict[i32,i32], base: i32 ) -> DictWriter:
		return DictWriter.__allocate__( target = target, base = base )

	def run( self ) -> None:
		i: i32 = 0
		while i < 100:
			with compiler.wrap_arithmetic:
				key: i32 = self.base + i
				val: i32 = key * 2
			self.target[ key ] = val
			with compiler.wrap_arithmetic:
				i += 1

def main() -> i32:
	d: dict[i32,i32] = dict[i32,i32]()
	threads: list[threading.Thread] = list[threading.Thread]()
	t: i32 = 0
	while t < 8:
		with compiler.wrap_arithmetic:
			base: i32 = t * 100
		w: DictWriter = DictWriter.make( d, base )
		threads.append( threading.Thread( w.run ) ).unwrap( 'append failed' )
		with compiler.wrap_arithmetic:
			t += 1
	i: usize = 0
	while i < 8:
		th: threading.Thread = threads.__getitem__( i ).unwrap( 'getitem failed' )
		th.join()
		with compiler.wrap_arithmetic:
			i += 1
	if d.__len__() != 800:
		return 1
	key: i32 = 0
	with compiler.wrap_arithmetic:
		while key < 800:
			r: Result[i32,KeyError] = d.__getitem__( key )
			if r.is_err():
				return 2
			with compiler.wrap_arithmetic:
				expected: i32 = key * 2
			if r.unwrap( 'x' ) != expected:
				return 3
			key += 1
	return 0
''' ),
			( 'with_lock_serializes_compound_increment_from_8_threads', '''
import threading

class Incrementer:
	target: dict[str,i32]

	@staticmethod
	def make( target: dict[str,i32] ) -> Incrementer:
		return Incrementer.__allocate__( target = target )

	def bump_once( self, raw: UnsafeDict[str,i32] ) -> None:
		r: Result[i32,KeyError] = raw.__getitem__( 'counter' )
		v: i32 = r.unwrap( 'counter missing' )
		with compiler.wrap_arithmetic:
			v += 1
		raw.__setitem__( 'counter', v )

	def run( self ) -> None:
		i: i32 = 0
		while i < 1000:
			self.target.with_lock( self.bump_once )
			with compiler.wrap_arithmetic:
				i += 1

def main() -> i32:
	d: dict[str,i32] = dict[str,i32]()
	d[ 'counter' ] = 0
	threads: list[threading.Thread] = list[threading.Thread]()
	inc: Incrementer = Incrementer.make( d )
	t: i32 = 0
	while t < 8:
		threads.append( threading.Thread( inc.run ) ).unwrap( 'append failed' )
		with compiler.wrap_arithmetic:
			t += 1
	i: usize = 0
	while i < 8:
		th: threading.Thread = threads.__getitem__( i ).unwrap( 'getitem failed' )
		th.join()
		with compiler.wrap_arithmetic:
			i += 1
	r: Result[i32,KeyError] = d.__getitem__( 'counter' )
	if r.is_err():
		return 1
	if r.unwrap( 'x' ) != 8000:
		return 2
	return 0
''' ),
			( 'concurrent_insert_of_rc_keys_and_values_no_leak_or_double_free', '''
import threading

class StrDictWriter:
	target: dict[str,str]
	base: i32

	@staticmethod
	def make( target: dict[str,str], base: i32 ) -> StrDictWriter:
		return StrDictWriter.__allocate__( target = target, base = base )

	def run( self ) -> None:
		i: i32 = 0
		while i < 50:
			with compiler.wrap_arithmetic:
				n: i32 = self.base + i
			key: str = _key_for( n )
			val: str = _key_for( n ) + '!'
			self.target[ key ] = val
			with compiler.wrap_arithmetic:
				i += 1

# str has no int-to-string conversion (only a copy constructor) - build a
# distinct 2-character key from n (0..199) as two base-26 letters via
# chr(), the same way ord()/chr() are used elsewhere in this stdlib
def _key_for( n: i32 ) -> str:
	with compiler.panic_arithmetic( '_key_for: overflow' ):
		d1: i32 = n // 26
		d2: i32 = n % 26
	with compiler.wrap_arithmetic:
		c1: u32 = compiler.cast( u32, 97 + d1 )
		c2: u32 = compiler.cast( u32, 97 + d2 )
	return chr( c1 ) + chr( c2 )

def main() -> i32:
	d: dict[str,str] = dict[str,str]()
	threads: list[threading.Thread] = list[threading.Thread]()
	t: i32 = 0
	while t < 4:
		with compiler.wrap_arithmetic:
			base: i32 = t * 50
		w: StrDictWriter = StrDictWriter.make( d, base )
		threads.append( threading.Thread( w.run ) ).unwrap( 'append failed' )
		with compiler.wrap_arithmetic:
			t += 1
	i: usize = 0
	while i < 4:
		th: threading.Thread = threads.__getitem__( i ).unwrap( 'getitem failed' )
		th.join()
		with compiler.wrap_arithmetic:
			i += 1
	if d.__len__() != 200:
		return 1
	n: i32 = 0
	with compiler.wrap_arithmetic:
		while n < 200:
			key: str = _key_for( n )
			r: Result[str,KeyError] = d.__getitem__( key )
			if r.is_err():
				return 2
			expected: str = _key_for( n ) + '!'
			if r.unwrap( 'x' ) != expected:
				return 3
			n += 1
	return 0
''' ),
		], timeout = 30 )


class TupleTests( test_support.RealCompileMixin, CompilerTestCase ):
	''' tuple[T0, T1, ...] (tuple_storage.py's TupleStorage, discovery.py's
	tuple[...] recognition, lowering.py's _expr_Tuple/_expr_Subscript) end-
	to-end - see PLAN_TUPLE.md. Mirrors ListGenericTests'/DictTests' own
	import_builtins=True + real compile-and-run convention (an RC element
	like str needs the rest of builtins for real). Unlike list[T]/dict[K,V]
	there's no FastLock wrapper to test - a tuple's fields are only ever
	written once, by construction, so no ThreadSafetyTests sibling class is
	needed here (see PLAN_TUPLE.md's own "Deferred" list on why). '''

	def setUp( self ) -> None:
		self.discovery = Discovery( import_builtins = True )
		self.compiler = Compiler( self.discovery )

	@unittest.skipUnless( test_support.HAS_CC, 'no C compiler (clang/gcc/msvc) found - skipping' )
	def test_programs_compile_and_run( self ) -> None:
		''' every real compile-and-run program in this class, merged into a
		single executable (one build for the whole class); a nonzero exit is
		decoded back to the failing sub-program and its own return code. '''
		self.assert_programs_run([
			# a value element (i32), an RC element (str), and a value element
			# again (bool) - constant-index reads each one back correctly, and
			# a deliberately WRONG expected value on each check (return 1/2/3)
			# rules out a vacuous pass (every check actually has to fire)
			( 'construct_and_read_back_heterogeneous_elements', '''
def main() -> i32:
	t: tuple[i32, str, bool] = ( 10, "hi", True )
	if t[0] != 10:
		return 1
	if t[1] != "hi":
		return 2
	if t[2] != True:
		return 3
	return 0
''' ),
			# no annotation at all - each element's own type is inferred
			# independently (bare int literals default to i32), same as any
			# other unannotated local declaration
			( 'unannotated_local_infers_element_types', '''
def main() -> i32:
	t = ( 1, 2, 3 )
	if t[0] != 1 or t[1] != 2 or t[2] != 3:
		return 1
	return 0
''' ),
			# an RC element (str) constructed into a tuple that's never read
			# again after construction - exercises the synthesized backing
			# RCClass's own destructor cascade (type_resolver.py's
			# _synthesize_rcclass_destructor, triggered here via the ordinary
			# scope-exit path, same as any other RC-holding local) tearing down
			# the str field correctly. No leak/double-free assertion is made
			# directly (this file has no ASan integration) - a clean exit code
			# 0 from a real compiled-and-linked binary is the same correctness
			# signal ListGenericTests'/DictTests' own RC-element tests rely on.
			( 'rc_element_constructed_and_dropped_without_crashing', '''
def main() -> i32:
	t: tuple[str, i32] = ( "owned", 5 )
	return 0
''' ),
			# tuple[i32,str] and tuple[str,i32] are two different backing
			# classes (see tuple_storage.py) - constructing/reading both in the
			# same function proves they don't collide with each other
			( 'two_distinct_tuple_shapes_construct_and_read_back_independently', '''
def main() -> i32:
	a: tuple[i32, str] = ( 1, "x" )
	b: tuple[str, i32] = ( "y", 2 )
	if a[0] != 1 or a[1] != "x":
		return 1
	if b[0] != "y" or b[1] != 2:
		return 2
	return 0
''' ),
			# regression test for PLAN_TUPLE.md's own "UPDATE (int.divmod()
			# migration)" section: every OTHER test in this class constructs a
			# tuple literal straight into an annotated local in the SAME
			# statement - the one shape lowering.py's _expr_Tuple defensively
			# resolves expected_type for. A tuple bound to a generic type
			# parameter T (here, by Result.Ok((a,b)) inferring T from the
			# tuple literal's own type) and then read back through a SEPARATE
			# statement's .unwrap() call is structurally different - exactly
			# the shape that broke int.divmod()'s own real migration (a bare,
			# unresolved TupleType reached emitter_c.py three different ways,
			# each fixed in monomorphize.py/lowering.py/emitter_c.py/cfg.py -
			# see the plan doc for the full account). This guards those fixes
			# directly, independent of int itself.
			( 'tuple_bound_through_generic_inference_not_a_bare_literal', '''
class PairError:
	pass

def make_pair( a: i32, b: i32 ) -> Result[tuple[i32,i32], PairError]:
	return Result.Ok( ( a, b ) )

def main() -> i32:
	pair: tuple[i32,i32] = make_pair( 3, 4 ).unwrap( 'x' )
	if pair[0] != 3 or pair[1] != 4:
		return 1
	return 0
''' ),
			# regression test: tuple[...] as a NESTED, EXPLICIT type argument
			# to another generic class's own constructor CALL -
			# list[tuple[str,str]]() - used to fail with "name 'tuple' is not
			# defined" even though the exact same list[tuple[str,str]]
			# ANNOTATION resolved fine one line above it (see type_resolver.
			# py's _try_resolve_namespace - the constructor-call counterpart
			# to discovery.py's own visit_Subscript, which already recognized
			# tuple[...] textually for annotations). Mirrors the real
			# lib/http/client.py HTTPHeaders shape this bug was found in:
			# construct a list of pairs, append, read each field back
			( 'list_of_tuple_as_explicit_constructor_type_argument', '''
def main() -> i32:
	entries: list[tuple[str,str]] = list[tuple[str,str]]()
	entries.append( ( "Content-Type", "text/plain" ) ).unwrap( 'append' )
	entries.append( ( "X-Test", "1" ) ).unwrap( 'append' )
	if len( entries ) != 2:
		return 1
	first: tuple[str,str] = entries.__getitem__( 0 ).unwrap( 'idx' )
	if first[0] != "Content-Type" or first[1] != "text/plain":
		return 2
	second: tuple[str,str] = entries.__getitem__( 1 ).unwrap( 'idx' )
	if second[0] != "X-Test" or second[1] != "1":
		return 3
	return 0
''' ),
			# regression test: a tuple whose ELEMENT type is itself a union
			# (tuple[str|None, i32], not the tuple as a whole being optional) -
			# lib/http/client.py's _encode_body() hit this exact shape
			# (tuple[bytes|None, str|None]) and had to work around it with a
			# dedicated class instead (see that file's own comment). Before the
			# fix, _expr_Tuple lowered each element with expected_type=None,
			# so a leaf value never got coerced into the declared union - the
			# tuple type inferred from the elements' own NATURAL types then
			# differed from the annotation's tuple[str|None,i32], leaving the
			# real (annotated) backing class's own allocator function
			# unscheduled ("call to undeclared function ...") and each field
			# assignment storing a raw leaf into what the emitter declared as
			# a tagged-union-typed field ("assigning to ... from incompatible
			# type"). Both the None and non-None leaf both need checking, on
			# both an unannotated inline construction and a separately
			# annotated local.
			( 'tuple_element_union_none_and_value_cases', '''
def main() -> i32:
	none_first: tuple[str|None, i32] = ( None, 5 )
	# bind the constant-index read to a named local, then extract the leaf
	# via match - same pattern union_coercion_rc_test.py's own passing
	# tests already use. `x is None` narrowing (type_resolver.py's
	# _type_of_expr) has no ast.Subscript case, and comparing a still-
	# union-typed value directly against a literal (`x != "hi"`) has no
	# dunder/flat-Cmp support either - both pre-existing, unrelated gaps
	# (neither specific to tuples: they'd reproduce on any bare str|None
	# local too), sidestepped here rather than fixed, to keep this
	# regression test scoped to the tuple-construction bug alone
	first_none: str|None = none_first[0]
	match first_none:
		case str( unexpected ):
			return 1
		case None:
			pass
	if none_first[1] != 5:
		return 2
	value_first: tuple[str|None, i32] = ( "hi", 6 )
	first_value: str|None = value_first[0]
	match first_value:
		case str( got ):
			if got != "hi":
				return 3
		case None:
			return 4
	if value_first[1] != 6:
		return 5
	return 0
''' ),
			# TWO independent union elements in the same tuple - the exact
			# arity/shape of the real _encode_body() bug (tuple[bytes|None,
			# str|None]), using str for both since bytes literals aren't
			# needed to exercise the same coercion code path
			( 'tuple_two_independent_union_elements', '''
def main() -> i32:
	t: tuple[str|None, str|None] = ( "body", None )
	first: str|None = t[0]
	match first:
		case str( got ):
			if got != "body":
				return 1
		case None:
			return 2
	second: str|None = t[1]
	match second:
		case str( unexpected ):
			return 3
		case None:
			pass
	return 0
''' ),
		] )


@unittest.skipUnless( test_support.HAS_CC, 'no C compiler (clang/gcc/msvc) found - skipping real-compile RC tests' )
class TupleUnionElementRCTests( test_support.RealCompileMixin, CompilerTestCase ):
	''' RC-correctness regression test for the tuple[T|None,...] fix above -
	construction must incref an aliasing leaf EXACTLY once (going through
	_coerce_into_union's own constructor Incref, same as any other union-
	typed field/local - see union_coercion_rc_test.py) and the tuple's own
	scope-exit teardown must release it back down again, not leak or double-
	free it. 'held'.upper() (not a literal) forces a real heap allocation -
	an immortal string literal can't tell a leak/over-release apart from
	doing nothing. '''
	def setUp( self ) -> None:
		self.discovery = Discovery( import_builtins = True )
		self.compiler = Compiler( self.discovery )

	def test_programs_compile_and_run( self ) -> None:
		self.assert_programs_run([
			( 'union_element_construction_increfs_once_and_releases_on_scope_exit', '''
def during_refcount( s: str ) -> usize:
	t: tuple[str|None, i32] = ( s, 1 )
	return compiler.refcount( s )

def main() -> i32:
	with compiler.wrap_arithmetic:
		s: str = 'held'.upper()
		before: usize = compiler.refcount( s )
		during: usize = during_refcount( s )
		if during != before + 1:
			return 1
		after: usize = compiler.refcount( s )
		if after != before:
			return 2
	return 0
''' ),
		] )


class UnionLeafCoercionTests( test_support.RealCompileMixin, CompilerTestCase ):
	''' a plain leaf value (a literal, a variable, bare None) flowing into
	a T|None (TaggedUnion)-typed slot - a call argument, a default value,
	or a variable declaration/assignment. Never worked before (confirmed:
	every T|None usage anywhere in this codebase - bisect.py's key=,
	__File.py's exists=, unwrap_or's own default - only ever received its
	None default at every real call site; passing an actual override
	value either crashed at Python-level emission (a literal) or silently
	produced invalid C that only a real clang invocation would catch (a
	plain variable) - see lowering.py's _lower_expr/_coerce_into_union and
	their own comments for the fix, and TODO.txt's "opportunistic union
	emission" section this closes. Reuses UnionStorage's own per-member
	constructor Functions (union_storage.py) - the exact mechanism a real,
	explicit Result.Ok(x) call already goes through - so this is
	implicit/automatic construction through that same path, not new
	construction machinery. '''

	def setUp( self ) -> None:
		self.discovery = Discovery( import_builtins = True )
		self.compiler = Compiler( self.discovery )

	@unittest.skipUnless( test_support.HAS_CC, 'no C compiler (clang/gcc/msvc) found - skipping' )
	def test_programs_compile_and_run( self ) -> None:
		''' every real compile-and-run program in this class, merged into a
		single executable (one build for the whole class); a nonzero exit is
		decoded back to the failing sub-program and its own return code. '''
		self.assert_programs_run([
			( 'literal_variable_and_none_as_call_arguments', '''
def helper( x: str|None ) -> bool:
	return x is None

def main() -> i32:
	s: str = "hi"
	if helper( s ):
		return 1
	if helper( "literal" ):
		return 2
	if not helper( None ):
		return 3
	return 0
''' ),
			( 'literal_and_none_in_variable_declarations', '''
def main() -> i32:
	x: str|None = "hi"
	if x is None:
		return 1
	y: str|None = None
	if y is not None:
		return 2
	return 0
''' ),
			# spot-checks a real pre-existing site, not just a synthetic
			# example: lib/builtins/__File.py's binary_writer()/etc all take
			# exists: bool|None = None and branch `if exists is None: ...
			# elif exists: ... else: ...` - this mirrors that exact shape
			# (own `is None`/truthiness checks directly on the union value,
			# no leaf EXTRACTION needed, so no further gap blocks it) with a
			# real True/False override at the call site, never exercised
			# anywhere in the codebase before this fix
			( 'bool_leaf_matches_lib_File_py_own_exists_parameter_shape', '''
def resolve( exists: bool|None = None ) -> i32:
	if exists is None:
		return 0
	elif exists:
		return 1
	else:
		return 2

def main() -> i32:
	if resolve() != 0:
		return 1
	if resolve( True ) != 1:
		return 2
	if resolve( False ) != 2:
		return 3
	e: bool = True
	if resolve( e ) != 1:
		return 4
	return 0
''' ),
			# a real RC-lifetime check, not just "doesn't crash once" - calls a
			# str|None-taking function in a loop, each iteration constructing a
			# fresh str and letting it flow through the union wrap/unwrap
			# round trip; a leaked or double-released reference here would
			# drift the refcount or crash under repetition, not just once.
			# 'hello'.upper() (not the bare literal 'hello') - a bare string
			# literal binds straight to its own static, immortal storage (no
			# allocation, refcount reads as a sentinel, never 1), so it can't
			# tell a leak/double-release apart from doing nothing; .upper()
			# always allocates a genuine, freshly refcounted buffer (__str.py's
			# case_map), independent of this fix
			( 'rc_leaf_refcount_correct_after_repeated_calls', '''
def identity_len( x: str|None ) -> usize:
	if x is None:
		return 0
	return 1

def main() -> i32:
	i: i32 = 0
	while i < 1000:
		s: str = 'hello'.upper()
		if compiler.refcount( s ) != 1:
			return 1
		identity_len( s )
		if compiler.refcount( s ) != 1:
			return 2
		with compiler.wrap_arithmetic:
			i += 1
	return 0
''' ),
			# a CALL's return value (not a literal/bare-name/None) flowing into
			# a T|None slot - the one leaf-value kind d712226 missed. Covers an
			# AnnAssign RHS, a call argument, and a function's own return
			# statement. Before the fix, lowering.py's _lower_call shared tail
			# typed the call's dest as expected_type (the UNION) up front
			# instead of target.return_type (str, the call's REAL C return
			# type), so _lower_expr's post-hoc _coerce_into_union never even
			# ran - dest ended up declared as the union struct while the
			# emitted call actually assigned a raw str* into it, a real clang
			# type error ("assigning to 'struct ...NoneType' from incompatible
			# type 'struct builtins$str *'")
			( 'call_result_as_leaf_coerces_into_union', '''
def make_or_none( n: i32 ) -> str|None:
	if n < 0:
		return None
	return 'ok'.upper()

def helper( x: str|None ) -> bool:
	return x is None

def main() -> i32:
	x: str|None = 'hello'.upper()
	if x is None:
		return 1
	if helper( 'world'.upper() ):
		return 2
	y: str|None = make_or_none( -1 )
	if y is not None:
		return 3
	z: str|None = make_or_none( 1 )
	if z is None:
		return 4
	return 0
''' ),
			# regression test: a TUPLE leaf (not a plain RCClass/str leaf like
			# every case above) flowing into a T|None-typed slot - found via
			# real lib/http/client.py Session work (auth: tuple[str,str]|None).
			# lowering.py's _expr_Tuple used to trust `expected_type` blindly
			# for its own dest's type, even when expected_type was this outer
			# union (a coercion HINT meant for _coerce_or_check_operand to act
			# on afterward, not a description of the tuple itself) - so dest
			# ended up typed as the union's own TaggedUnion (a value struct,
			# never an RCClass) instead of the tuple's real backing RCClass,
			# crashing emitter_c.py's Allocate emission outright with `assert
			# isinstance(concrete_cls, RCClass)` before ever reaching
			# _coerce_into_union. Covers both a real tuple value and None
			# flowing through the same parameter, plus a repeated-call
			# refcount check (mirrors rc_leaf_refcount_correct_after_repeated_
			# calls above) since the tuple's own str elements are RC and a
			# double-incref/masked-decref in the coercion path wouldn't show
			# up as a crash, just a drifting refcount.
			( 'tuple_leaf_coerces_into_union', '''
def creds( x: tuple[str,str]|None = None ) -> i32:
	if x is None:
		return 0
	t: tuple[str,str] = x
	if t[0] != 'user':
		return 1
	if t[1] != 'pass':
		return 2
	return 3

def main() -> i32:
	with compiler.wrap_arithmetic:
		if creds() != 0:
			return 1
		if creds( ( 'user', 'pass' ) ) != 3:
			return 2
		i: i32 = 0
		while i < 1000:
			u: str = 'user'.upper().lower()
			if compiler.refcount( u ) != 1:
				return 3
			if creds( ( u, 'pass' ) ) != 3:
				return 4
			if compiler.refcount( u ) != 1:
				return 5
			i += 1
	return 0
''' ),
			# `t[0] is None`/`is not None` narrowing directly on a tuple
			# constant-index Subscript, with no intermediate named local -
			# type_resolver.py's _type_of_expr had no ast.Subscript case at
			# all, so this used to fall through to _lower_is_comparison's own
			# flat-Cmp fallback and emit invalid C comparing a union STRUCT
			# against a bare int/another struct
			( 'tuple_subscript_is_none_narrowing_without_a_local', '''
def main() -> i32:
	t: tuple[str|None, i32] = ( None, 5 )
	if t[0] is not None:
		return 1
	if t[1] != 5:
		return 2
	u: tuple[str|None, i32] = ( "hi", 6 )
	if u[0] is None:
		return 3
	if u[1] != 6:
		return 4
	return 0
''' ),
			# `union_val == leaf` / `!=` - comparing a still-union-typed value
			# directly against a leaf, with no match-based extraction needed
			# first. Used to fall through to the plain dunder-or-flat-Cmp path,
			# which either called a leaf's dunder with a UNION-typed argument
			# (a real type mismatch in the generated C) or compared two union
			# STRUCTS directly (C rejects this outright) - see
			# _lower_union_eq_against_leaf. Covers an RC leaf (str), a scalar
			# leaf (i32), the None member via == / != (not is/is not, which
			# already worked), and negation of each
			( 'union_leaf_equality_str_and_scalar_and_none', '''
def main() -> i32:
	x: str|None = "hi"
	if x != "hi":
		return 1
	if x == "bye":
		return 2
	y: str|None = None
	if y == "hi":
		return 3
	if not ( y != "hi" ):
		return 4
	if not ( y == None ):
		return 5
	if y != None:
		return 6
	n: i32|None = 42
	if n != 42:
		return 7
	if n == 7:
		return 8
	m: i32|None = None
	if m == 42:
		return 9
	return 0
''' ),
			# a real RC-lifetime check for the new payload-comparison path -
			# _lower_union_eq_against_leaf's own narrowed/payload_dest/tag_dest
			# temps are all bare GetAttr borrows (never fresh_temp()-registered,
			# same as _lower_union_receiver_call's identical extraction), so
			# none of them should need - or get - any incref/decref of their
			# own; this repeats the comparison 1000x and checks the RC leaf's
			# own refcount never drifts
			( 'union_leaf_equality_rc_no_leak_under_repetition', '''
def main() -> i32:
	with compiler.wrap_arithmetic:
		i: i32 = 0
		while i < 1000:
			s: str = 'hello'.upper()
			if compiler.refcount( s ) != 1:
				return 1
			x: str|None = s
			if compiler.refcount( s ) != 2: # the union now holds a reference too
				return 2
			if x != "HELLO":
				return 3
			if compiler.refcount( s ) != 2: # comparing must not have changed it
				return 4
			i += 1
	return 0
''' ),
		] )

	@unittest.skipUnless( _CC is not None, 'no C compiler (clang/gcc/msvc) found - skipping' )
	def test_non_leaf_type_is_a_compile_error( self ) -> None:
		# a genuine type mismatch (not one of the union's own members) must
		# stay a real compile error, not get silently passed through
		self._run( '''
def helper( x: str|None ) -> bool:
	return x is None

def main() -> i32:
	helper( 5 )
	return 0
''' )
		self.assertNotEqual( self.discovery.errors.errors, [] )


class UnionReceiverDispatchCoercionTests( test_support.RealCompileMixin, CompilerTestCase ):
	''' real compile-and-run companion to GenericMethodDispatchTests'
	test_union_receiver_dispatch_applies_per_leaf_scalar_widening - proves
	the per-leaf ir.CastWrap actually widens the runtime VALUE correctly
	through both leaves of a union receiver, not just that the IR has the
	right shape.

	Uses two plain, unrelated classes (not two Specializations of one
	generic class, unlike the lowering-level test) deliberately - this was
	written before _coerce_into_union's own identity-based leaf lookup
	(`attr.type is operand.type`) was fixed to use _same_type instead (see
	PLAN_COMPILER_BUG_SWEEP.md), at a time when assigning a freshly-
	constructed generic RCClass value into a union of that same generic
	class's own instantiations (a Specialization built by a constructor
	call never reconciled with the one the union's own member list held)
	was a real, separate, confirmed bug, flagged but deliberately not
	fixed as out of scope for the plan active at the time. That fix has
	since landed - see test_generic_union_member_construction_and_
	assignment below, added once this class was revisited and the bug
	confirmed already resolved. '''

	def setUp( self ) -> None:
		self.discovery = Discovery( import_builtins = True )
		self.compiler = Compiler( self.discovery )

	@unittest.skipUnless( test_support.HAS_CC, 'no C compiler (clang/gcc/msvc) found - skipping' )
	def test_programs_compile_and_run( self ) -> None:
		# matches() returns bool (identical across leaves) rather than each
		# leaf's own field type deliberately: union-receiver dispatch
		# requires every leaf's own method to share one return type, so
		# reading the per-leaf-widened value back has to go through a
		# same-return-type-everywhere method instead
		self.assert_programs_run([
			( 'per_leaf_scalar_widening_produces_correct_runtime_value', '''
class BoxI32:
	v: i32
	def set( self, x: i32 ) -> None:
		self.v = x
	def matches( self, expected: i64 ) -> bool:
		with compiler.wrap_arithmetic:
			return i64( self.v ) == expected

class BoxI64:
	v: i64
	def set( self, x: i64 ) -> None:
		self.v = x
	def matches( self, expected: i64 ) -> bool:
		return self.v == expected

def main() -> i32:
	x: i32 = 5

	u64: BoxI32|BoxI64 = BoxI64( v = 0 )
	u64.set( x )
	if not u64.matches( 5 ):
		return 1

	u32: BoxI32|BoxI64 = BoxI32( v = 0 )
	u32.set( x )
	if not u32.matches( 5 ):
		return 2
	return 0
''' ),
		] )

	@unittest.skipUnless( test_support.HAS_CC, 'no C compiler (clang/gcc/msvc) found - skipping' )
	def test_generic_union_member_construction_and_assignment( self ) -> None:
		# the case this class's own docstring used to flag as a separate,
		# unfixed bug: a freshly-constructed generic RCClass value (Box[i32](5),
		# built via a real constructor call) assigned into a union of that
		# same generic class's own DIFFERENT instantiations (Box[i32]|Box[i64]).
		# The constructed value's own Specialization and the union's own member
		# list's Specialization for Box[i32] used to be two different objects
		# for the identical instantiation - _coerce_into_union's identity-based
		# leaf lookup rejected this outright before its _same_type fix
		self._run( '''
class Box[T]:
	v: T
	def __init__( self, v: T ) -> None:
		self.v = v

def main() -> i32:
	u: Box[i32]|Box[i64] = Box[i32]( 5 )
	return 0
''' )
		self.assertEqual( self.discovery.errors.errors, [] )

	@unittest.skipUnless( test_support.HAS_CC, 'no C compiler (clang/gcc/msvc) found - skipping' )
	def test_generic_leaves_with_equal_return_types_do_not_false_positive( self ) -> None:
		# _resolve_union_receiver_members' own leaf-agreement check
		# (type_resolver.py, "leaf implementations disagree on return type")
		# used to compare each leaf's own resolved return type via raw `is`.
		# Box[i32].get_list's own -> list[T] gets EAGERLY monomorphized to
		# list[i32] as part of specializing Box[i32] itself, while Other.
		# get_list's own -> list[i32] is resolved fresh, straight from its
		# own annotation - two different Specialization objects for the
		# textually-identical list[i32], wrongly reported as "disagreeing"
		# before the _same_type fix (see PLAN_COMPILER_BUG_SWEEP.md)
		self._run( '''
class Box[T]:
	def get_list( self ) -> list[T]:
		return list[T]()

class Other:
	def get_list( self ) -> list[i32]:
		return list[i32]()

def pick( flag: bool ) -> Box[i32]|Other:
	if flag:
		return Box[i32]()
	return Other()

def main() -> i32:
	u: Box[i32]|Other = pick( True )
	l: list[i32] = u.get_list()
	return 0
''' )
		self.assertEqual( self.discovery.errors.errors, [] )

	@unittest.skipUnless( test_support.HAS_CC, 'no C compiler (clang/gcc/msvc) found - skipping' )
	def test_genuinely_disagreeing_leaf_return_types_still_rejected( self ) -> None:
		# negative companion to the test above - the _same_type fix must not
		# make this check too permissive: two leaves with GENUINELY different
		# return types must still be rejected
		self._run( '''
class LeafA:
	def make( self ) -> i32:
		return 1

class LeafB:
	def make( self ) -> str:
		return 'x'

def pick( flag: bool ) -> LeafA|LeafB:
	if flag:
		return LeafA()
	return LeafB()

def main() -> i32:
	u: LeafA|LeafB = pick( True )
	x = u.make()
	return 0
''' )
		self.assertNotEqual( self.discovery.errors.errors, [] )


class WalrusOperatorRealCompileTests( test_support.RealCompileMixin, CompilerTestCase ):
	''' _expr_NamedExpr (ast.NamedExpr, `x := expr`) - real compile-and-run
	companion to lowering_test.py's WalrusOperatorTests. Deliberately
	avoids `if (x := opt()) is not None: use(x)`-shaped fixtures: `is not
	None` narrowing for a plain if-statement is a real, separate,
	pre-existing gap in this compiler (confirmed independent of walrus -
	the identical failure reproduces with an ordinary, non-walrus `x: T|
	None; if x is not None: use(x)`; only while/match/`type(x) is T`
	narrow today) - out of scope here, not something walrus needs to
	solve. These fixtures instead use plain scalar/bool conditions, which
	already work end to end. '''

	def setUp( self ) -> None:
		self.discovery = Discovery( import_builtins = True )
		self.compiler = Compiler( self.discovery )

	@unittest.skipUnless( test_support.HAS_CC, 'no C compiler (clang/gcc/msvc) found - skipping' )
	def test_programs_compile_and_run( self ) -> None:
		self.assert_programs_run([
			# the walrus target starts undeclared (first-declaration branch
			# of _expr_NamedExpr), then the SAME while condition re-evaluates
			# it every subsequent iteration (the reassignment branch) -
			# exercises both branches in one natural fixture, and confirms
			# the binding survives (and is reused) past the loop
			( 'walrus_in_while_condition_first_decl_then_rebind', '''
def main() -> i32:
	i: i32 = 0
	total: i32 = 0
	with compiler.wrap_arithmetic:
		while ( x := i ) < 5:
			total += x
			i += 1
	if total != 10:
		return 1
	if i != 5:
		return 2
	return 0
''' ),
			# the walrus expression's own return value used directly as an
			# if-condition, then the same binding read again afterward
			( 'walrus_return_value_used_directly_as_condition', '''
def f( n: i32 ) -> i32:
	with compiler.wrap_arithmetic:
		return n + 1

def main() -> i32:
	if ( y := f( 4 ) ) != 5:
		return 1
	if y != 5:
		return 2
	return 0
''' ),
		] )


class SliceSyntaxTests( test_support.RealCompileMixin, CompilerTestCase ):
	''' x[a:b] / x[:b] / x[a:] (ast.Slice) - PLAN_POSIX_FEATURE.md's scope,
	str/bytearray only (list[T] slicing deferred - no real caller). Byte-
	offset semantics, not Python's real Unicode-codepoint offsets - see
	_lower_slice_subscript's own docstring on why. '''

	def setUp( self ) -> None:
		self.discovery = Discovery( import_builtins = True )
		self.compiler = Compiler( self.discovery )

	@unittest.skipUnless( test_support.HAS_CC, 'no C compiler (clang/gcc/msvc) found - skipping' )
	def test_programs_compile_and_run( self ) -> None:
		self.assert_programs_run([
			( 'str_slice_shapes', '''
def main() -> i32:
	s: str = "hello world"
	if s[:5] != "hello":
		return 1
	if s[6:] != "world":
		return 2
	if s[2:5] != "llo":
		return 3
	return 0
''' ),
			( 'bytearray_slice_shapes', '''
def main() -> i32:
	b: bytearray = bytearray( 5 )
	p: Ptr[u8] = b.get_ptr()
	p[0] = 1
	p[1] = 2
	p[2] = 3
	p[3] = 4
	p[4] = 5
	c: bytearray = b[1:4]
	if len( c ) != 3:
		return 1
	cp: ConstPtr[u8] = c.get_const_ptr()
	if cp[0] != 2 or cp[1] != 3 or cp[2] != 4:
		return 2
	if len( b[:2] ) != 2:
		return 3
	if len( b[3:] ) != 2:
		return 4
	return 0
''' ),
			# mirrors lib/posix/fs.py:24's buf[:nbytes] shape - slicing a
			# bytearray to a runtime-computed length, not a constant
			( 'bytearray_slice_to_computed_length', '''
def fill( buf: bytearray ) -> usize:
	p: Ptr[u8] = buf.get_ptr()
	p[0] = 65
	p[1] = 66
	p[2] = 67
	return 3

def main() -> i32:
	buf: bytearray = bytearray( 128 )
	nbytes: usize = fill( buf )
	result: bytearray = buf[:nbytes]
	if len( result ) != 3:
		return 1
	return 0
''' ),
			# mirrors lib/posix/time.py:52's target_path[idx+9:] shape -
			# slicing a str from a runtime-computed (str.find()'s own byte
			# offset) start, no upper bound
			( 'str_slice_from_computed_find_offset', '''
def main() -> i32:
	target_path: str = "/usr/share/zoneinfo/America/New_York"
	idx: usize = target_path.find( "zoneinfo/" ).unwrap( "expected match" )
	with compiler.wrap_arithmetic:
		tz: str = target_path[idx+9:]
	if tz != "America/New_York":
		return 1
	return 0
''' ),
		] )

	def test_slice_step_is_rejected( self ) -> None:
		self._run( '\n'.join([
			'def main() -> None:',
			'	s: str = "hello"',
			'	a: str = s[::2]',
			'	return',
		]))
		errors = self.discovery.errors.errors
		self.assertEqual( len( errors ), 1 )
		self.assertIn( 'slice step is not supported', errors[0] )

	def test_unsupported_receiver_type_is_rejected( self ) -> None:
		self._run( '\n'.join([
			'def main() -> None:',
			'	x: i32 = 5',
			'	y: i32 = x[0:2]',
			'	return',
		]))
		errors = self.discovery.errors.errors
		self.assertEqual( len( errors ), 1 )
		self.assertIn( 'slicing is not supported for intrinsics.i32', errors[0] )


class ListLiteralRealCompileTests( test_support.RealCompileMixin, CompilerTestCase ):
	''' _expr_List (ast.List, `[a, b, c]`) - real compile-and-run companion
	to lowering_test.py's ListLiteralTests. '''

	def setUp( self ) -> None:
		self.discovery = Discovery( import_builtins = True )
		self.compiler = Compiler( self.discovery )

	@unittest.skipUnless( test_support.HAS_CC, 'no C compiler (clang/gcc/msvc) found - skipping' )
	def test_programs_compile_and_run( self ) -> None:
		self.assert_programs_run([
			( 'str_list_literal', '''
def main() -> i32:
	x: list[str] = [ 'a', 'b', 'c' ]
	if len( x ) != 3:
		return 1
	if x.__getitem__( 0 ).unwrap( 'idx failed' ) != 'a':
		return 2
	if x.__getitem__( 2 ).unwrap( 'idx failed' ) != 'c':
		return 3
	return 0
''' ),
			( 'i32_list_literal', '''
def main() -> i32:
	x: list[i32] = [ 10, 20, 30 ]
	if len( x ) != 3:
		return 1
	if x.__getitem__( 1 ).unwrap( 'idx failed' ) != 20:
		return 2
	return 0
''' ),
			( 'empty_list_literal', '''
def main() -> i32:
	x: list[i32] = []
	if len( x ) != 0:
		return 1
	return 0
''' ),
			# mirrors the real forcing case: lib/codecs/*.py's own
			# names(self) -> list[str]: return [...] shape
			( 'list_literal_returned_from_function', '''
def names() -> list[str]:
	return [ 'utf8', 'utf-8', 'UTF8', 'UTF-8' ]

def main() -> i32:
	n = names()
	if len( n ) != 4:
		return 1
	if n.__getitem__( 0 ).unwrap( 'idx failed' ) != 'utf8':
		return 2
	if n.__getitem__( 3 ).unwrap( 'idx failed' ) != 'UTF-8':
		return 3
	return 0
''' ),
		] )


class SetLiteralRealCompileTests( test_support.RealCompileMixin, CompilerTestCase ):
	''' _expr_Set (ast.Set, `{a, b, c}`) - real compile-and-run companion to
	lowering_test.py's SetLiteralTests. '''

	def setUp( self ) -> None:
		self.discovery = Discovery( import_builtins = True )
		self.compiler = Compiler( self.discovery )

	@unittest.skipUnless( test_support.HAS_CC, 'no C compiler (clang/gcc/msvc) found - skipping' )
	def test_programs_compile_and_run( self ) -> None:
		self.assert_programs_run([
			( 'str_set_literal', '''
def main() -> i32:
	x: set[str] = { 'a', 'b', 'c' }
	if x.__len__() != 3:
		return 1
	if not ( x.__contains__( 'a' ) and x.__contains__( 'b' ) and x.__contains__( 'c' )):
		return 2
	if x.__contains__( 'z' ):
		return 3
	return 0
''' ),
			( 'i32_set_literal', '''
def main() -> i32:
	x: set[i32] = { 10, 20, 30 }
	if x.__len__() != 3:
		return 1
	if not ( x.__contains__( 10 ) and x.__contains__( 20 ) and x.__contains__( 30 )):
		return 2
	if x.__contains__( 40 ):
		return 3
	return 0
''' ),
			# a repeated literal element is exactly one dedup add(), not a
			# real duplicate - the same overwrite-existing-key semantics
			# set[T].add already relies on, just reached through literal
			# syntax instead of explicit .add() calls
			( 'duplicate_elements_in_literal_are_deduped', '''
def main() -> i32:
	x: set[i32] = { 1, 1, 2 }
	if x.__len__() != 2:
		return 1
	if not ( x.__contains__( 1 ) and x.__contains__( 2 )):
		return 2
	return 0
''' ),
			( 'set_literal_returned_from_function', '''
def codes() -> set[i32]:
	return { 200, 201, 204 }

def main() -> i32:
	c = codes()
	if c.__len__() != 3:
		return 1
	if not ( c.__contains__( 200 ) and c.__contains__( 201 ) and c.__contains__( 204 )):
		return 2
	if c.__contains__( 404 ):
		return 3
	return 0
''' ),
		] )


class MoveParameterRealCompileTests( test_support.RealCompileMixin, CompilerTestCase ):
	''' move[T] is an ownership status on a binding, not a distinct type
	from T (Parameter.is_move, not a Move-wrapped .type) - real compile-
	and-run companion to lowering_test.py's MoveParameterTests. Exercises
	lib/builtins/__init__.py's own real bytes.from_bytearray, previously-
	untested dead code (nothing in lib/ ever called it before this fix)
	that reads len(src) before consuming src via .release() - also depends
	on bytearray.release()'s own return-type fix (bare sys.OwnershipError
	-> sys.OwnershipError[bytearray], a separate, real, pre-existing
	authoring bug this same investigation found: the unspecialized
	annotation left T unbound, so Err(SharedReference(x))'s own x never
	resolved to a real bytearray anywhere that pattern was matched).

	str.from_cstr's identical move[bytearray] overload is exercised by the
	'move_through_overload_resolution' case below - move(...)'s own sugar
	not being recognized during OVERLOAD resolution ("name 'move' is not
	defined") is fixed (peeled before candidate type-matching in
	_lower_overload_arg's own caller, then validated+applied via the real
	ownership-transfer hook once resolve_call picks a single concrete
	winner - see lowering.py's _lower_call, the Overload branch;
	lowering_test.py's own OverloadMoveResolutionTests verifies this
	directly via IR inspection). Getting a REAL compile-and-run test
	against str.from_cstr specifically also required fixing a separate,
	general, pre-existing bug this investigation found: emitter_c.py used
	to mangle every candidate in an @overload group to the SAME C symbol
	name, so a program needing real C bodies for more than one candidate
	(str.from_cstr's own move[bytearray] overload unconditionally falls
	back to calling its 2-arg sibling in one branch, so both always need
	real bodies together) failed to compile at the C level - see
	OverloadRealCompileTests below for a minimal, move-unrelated repro of
	that bug; fixed via mangle_function_qualname consulting each
	Function's own overload_group/position within it. '''

	def setUp( self ) -> None:
		self.discovery = Discovery( import_builtins = True )
		self.compiler = Compiler( self.discovery )

	@unittest.skipUnless( test_support.HAS_CC, 'no C compiler (clang/gcc/msvc) found - skipping' )
	def test_programs_compile_and_run( self ) -> None:
		self.assert_programs_run([
			( 'move_parameter_read_before_consume', '''
def consume( src: move[bytearray] ) -> usize:
	n: usize = len( src )
	return n

def main() -> i32:
	b: bytearray = bytearray( 5 )
	if consume( move( b )) != 5:
		return 1
	return 0
''' ),
			( 'bytes_from_bytearray_real_usage', '''
def main() -> i32:
	b: bytearray = bytearray( 5 )
	p: Ptr[u8] = b.get_ptr()
	p[0] = 104
	p[1] = 101
	p[2] = 108
	p[3] = 108
	p[4] = 111
	bs: bytes = bytes.from_bytearray( move( b ))
	if len( bs ) != 5:
		return 1
	cp: ConstPtr[u8] = bs.get_const_ptr()
	if cp[0] != 104:
		return 2
	return 0
''' ),
			( 'move_through_overload_resolution', '''
def main() -> i32:
	b: bytearray = bytearray( 6 )
	p: Ptr[u8] = b.get_ptr()
	p[0] = 104
	p[1] = 101
	p[2] = 108
	p[3] = 108
	p[4] = 111
	p[5] = 0
	s: str = str.from_cstr( move( b )).unwrap( 'from_cstr failed' )
	if s != "hello":
		return 1
	if s.byte_len() != 5:
		return 2
	return 0
''' ),
		] )


class OverloadRealCompileTests( test_support.RealCompileMixin, CompilerTestCase ):
	''' regression test for the general emitter_c.py bug found alongside the
	move(...)-through-overload-resolution fix above (see
	MoveParameterRealCompileTests' own docstring): every candidate in an
	@overload group used to mangle to the SAME C symbol name (mpy_types.
	Overload's members all share one .qualname - "the same named function",
	just different signatures), so a program that actually needs real C
	bodies for more than one candidate in the same group failed to compile
	at the C level ("conflicting types"/"too many arguments", depending on
	whether the two happened to share an arity). This is a minimal,
	move-unrelated repro: two @overload-decorated candidates distinguished
	purely by arity, both with real bodies, both actually called. Fixed via
	emitter_c.py's mangle_function_qualname consulting each Function's own
	overload_group/position within it (mpy_types.Function.overload_group) -
	a group with only one real implementation (the common case: signature-
	only stubs routed to one real body) still mangles unsuffixed. '''

	def setUp( self ) -> None:
		self.discovery = Discovery( import_builtins = True )
		self.compiler = Compiler( self.discovery )

	@unittest.skipUnless( test_support.HAS_CC, 'no C compiler (clang/gcc/msvc) found - skipping' )
	def test_programs_compile_and_run( self ) -> None:
		self.assert_programs_run([
			( 'two_overloads_both_called', '''
@overload
def combine( a: i32, b: i32 ) -> i32:
	with compiler.wrap_arithmetic:
		return a + b

@overload
def combine( x: i32 ) -> i32:
	with compiler.wrap_arithmetic:
		return x * 10

def main() -> i32:
	if combine( 2, 3 ) != 5:
		return 1
	if combine( 7 ) != 70:
		return 2
	return 0
''' ),
		] )


class OverloadGenericSubstitutionMatchingRealCompileTests( test_support.RealCompileMixin, CompilerTestCase ):
	''' overload_resolution.py's own _contains/_intersect/_subtract used a raw
	`is` identity check to decide whether a call-site argument's type matches
	an @overload candidate's own declared parameter type - _leaf_is_accepted's
	own claimed invariant ("the existing dedup caches already guarantee 'same
	type' is the same object") turned out not to hold in general: a generic
	function's own list[T], specialized to list[i32], is a DIFFERENT
	Specialization object than an @overload candidate's own freshly-annotated
	list[i32] parameter - the exact same duality TypeResolver._same_type
	exists to handle elsewhere in this compiler. Before the fix, calling
	through a generic function into an @overload group with a generic-
	substituted argument type raised "no matching overload" for a call that
	should resolve cleanly - confirmed via a real repro
	(PLAN_COMPILER_BUG_SWEEP.md). Fixed by threading a caller-supplied
	`same_type` predicate (TypeResolver._same_type) through resolve_call/
	stub_covers_call, defaulting to plain `is` so overload_resolution_test.
	py's own isolated unit tests (which never exercise this duality) stay
	unchanged. This also unblocks two OTHER, previously-unconfirmed fixes
	(lowering.py's _lower_dispatch_tests/_maybe_unwrap_union_arg, fixed in an
	earlier pass but gated behind this same upstream bug) - the runtime
	conditional-dispatch shape below genuinely exercises both. '''

	def setUp( self ) -> None:
		self.discovery = Discovery( import_builtins = True )
		self.compiler = Compiler( self.discovery )

	@unittest.skipUnless( test_support.HAS_CC, 'no C compiler (clang/gcc/msvc) found - skipping' )
	def test_programs_compile_and_run( self ) -> None:
		self.assert_programs_run([
			( 'generic_substituted_argument_matches_overload_candidate', '''
@overload
def handle( x: list[i32] ) -> i32:
	return 1

@overload
def handle( x: str ) -> i32:
	return 2

def dispatch[T]( v: list[T]|str ) -> i32:
	return handle( v )

def main() -> i32:
	if dispatch[i32]( list[i32]() ) != 1:
		return 1
	if dispatch[i32]( 'hi' ) != 2:
		return 2
	return 0
''' ),
		] )


class Utf8CodecRealCompileTests( test_support.RealCompileMixin, CompilerTestCase ):
	''' Codec.decode widened to bytes|bytearray, against the REAL Utf8
	class (not a synthetic stand-in) - constructing a real Utf8() instance
	forces its whole vtable (names/encode/decode) to compile, so this also
	depends on: Utf8.names()'s list literal (_expr_List), Utf8.encode()'s
	get_ptr()/get_const_ptr() fix, and the move[T] fix above (Utf8.encode()
	-> bytes.from_bytearray() -> len(src)/src.release()). Also covers the
	module-level `utf8 = Utf8()` singleton every real decode()/encode()
	default value actually uses. '''

	def setUp( self ) -> None:
		self.discovery = Discovery( import_builtins = True )
		self.compiler = Compiler( self.discovery )

	@unittest.skipUnless( test_support.HAS_CC, 'no C compiler (clang/gcc/msvc) found - skipping' )
	def test_programs_compile_and_run( self ) -> None:
		self.assert_programs_run([
			# explicit construction via the real class (not the singleton) -
			# keeps the vtable-forcing/construction path covered too
			( 'decode_bytes', '''
from codecs.utf8 import Utf8

def main() -> i32:
	b: bytearray = bytearray( 5 )
	p: Ptr[u8] = b.get_ptr()
	p[0] = 104
	p[1] = 101
	p[2] = 108
	p[3] = 108
	p[4] = 111
	bs: bytes = bytes( b )
	codec = Utf8()
	s: str = codec.decode( bs ).unwrap( 'decode failed' )
	if s != "hello":
		return 1
	return 0
''' ),
			# the rest use the shared `utf8` singleton directly
			( 'decode_bytearray', '''
from codecs.utf8 import utf8

def main() -> i32:
	b: bytearray = bytearray( 5 )
	p: Ptr[u8] = b.get_ptr()
	p[0] = 104
	p[1] = 105
	p[2] = 33
	p[3] = 33
	p[4] = 33
	c: bytearray = b[:3]
	s: str = utf8.decode( c ).unwrap( 'decode failed' )
	if s != "hi!":
		return 1
	return 0
''' ),
			# multi-byte UTF-8 round trip via the real utf8.encode() ->
			# utf8.decode() path - guards the alloc/memcpy/terminate
			# arithmetic in both directions
			( 'decode_multibyte_utf8_round_trip', '''
from codecs.utf8 import utf8

def main() -> i32:
	src: str = "héllo"
	eb: bytes = utf8.encode( src ).unwrap( 'encode failed' )
	s: str = utf8.decode( eb ).unwrap( 'decode failed' )
	if s != src:
		return 1
	if s.byte_len() != src.byte_len():
		return 2
	return 0
''' ),
			# a genuinely bytes|bytearray-typed local (not two separately-
			# typed locals) - exercises union-receiver dispatch for real
			( 'decode_through_union_typed_local', '''
from codecs.utf8 import utf8

def decode_it( x: bytes|bytearray ) -> str:
	return utf8.decode( x ).unwrap( 'decode failed' )

def main() -> i32:
	b: bytearray = bytearray( 3 )
	p: Ptr[u8] = b.get_ptr()
	p[0] = 97
	p[1] = 98
	p[2] = 99
	if decode_it( b ) != "abc":
		return 1
	bs: bytes = bytes( b )
	if decode_it( bs ) != "abc":
		return 2
	return 0
''' ),
			# mirrors the real forcing case: fs.py:24's
			# codec.decode(buf[:nbytes]) shape - and, unlike the other
			# cases here, relies entirely on decode()'s own now-fixed
			# `codec: Codec = utf8` DEFAULT (no codec argument passed at
			# all), proving the default itself works, not just the
			# singleton used explicitly
			( 'decode_bytearray_slice_result_via_default_codec', '''
def main() -> i32:
	buf: bytearray = bytearray( 128 )
	p: Ptr[u8] = buf.get_ptr()
	p[0] = 104
	p[1] = 105
	nbytes: usize = 2
	s: str = buf[:nbytes].decode().unwrap( 'decode failed' )
	if s != "hi":
		return 1
	return 0
''' ),
			( 'names_list_literal', '''
from codecs.utf8 import utf8

def main() -> i32:
	n = utf8.names()
	if len( n ) != 4:
		return 1
	if n.__getitem__( 0 ).unwrap( 'idx failed' ) != 'utf8':
		return 2
	if n.__getitem__( 3 ).unwrap( 'idx failed' ) != 'UTF-8':
		return 3
	return 0
''' ),
		] )


class AsciiCp437Latin1CodecRealCompileTests( test_support.RealCompileMixin, CompilerTestCase ):
	''' Real compile-and-run coverage for lib/codecs/ascii.py, cp437.py and
	latin1.py - previously constructed only via _build_registry()'s own
	.register() (itself only calling .names()), so their encode()/decode()
	bodies were never actually reached by any compiled program, real test,
	or the 1057-test suite passing. Getting these three to real-compile,
	link, and run surfaced (and this fix resolves):

	  - s.get_ptr() on str (only bytes|bytearray has get_ptr; str only
	    exposes get_const_ptr) in all three encode()s.
	  - missing checked-arithmetic wrappers around every +/- op in bodies
	    whose own Result[...] error type doesn't cover OverflowError.
	  - cp437.py's own `with compiler.panic_arithmetic:` (no call/message -
	    unsupported with statement; panic_arithmetic always takes one).
	  - bytes.from_bytearray( bytes, move( out )) in cp437.py/latin1.py
	    (bytes passed as a stray extra positional argument - too many
	    positional arguments; should just be from_bytearray( move( out ))).
	  - str.from_cstr( ptr, len ) in cp437.py/latin1.py/ascii.py's decode()
	    - that overload's second argument means size INCLUDING the zero
	    terminator (checked: from_cstr errors if buf[size-1] isn't 0), not
	    a plain byte count, and none of these buffers were ever actually
	    null-terminated - fixed by allocating an exact len+1 buffer,
	    memcpy'ing, explicitly terminating, and going through
	    str._from_owned_cstr directly (matching utf8.py's own decode()
	    shape) instead.
	  - DECODE_TABLE[i]/[usize(byte-0x80)] (plain __getitem__ sugar) in
	    cp437.py requiring encode()/decode() to return Result[_,IndexError]
	    (they return Result[_,CodecError]) - fixed via the real
	    .__getitem__(...).unwrap(...) call other list-indexing lib code
	    already uses.

	Also found and fixed two bugs invisible to discovery/emit_c alone (only
	surfaced by an actual C compile+link+run):

	  - A real, general, pre-existing compiler bug: a `return` reachable
	    while an RC-tracked local (e.g. a bytearray) is still live, in a
	    function that later consumes that SAME local via move() on its
	    fall-through success path, leaves the early return's own epilogue-
	    cleanup label un-emitted ("use of undeclared label" at the C
	    level) - current_epilogue_label() hands the return a label whose
	    backing _epilogue_stack entry the later move() consumption then
	    silently drops, instead of leaving a decref-less "cancelled" entry
	    the way every other consumption path does. Confirmed via minimal,
	    codec-independent repros. Not fixed here (out of this scope - a
	    cfg.py/lowering.py issue, not a lib/codecs one); ascii.py/cp437.py/
	    latin1.py's own encode()s just avoid the trigger shape (bytes(out)
	    copy instead of bytes.from_bytearray(move(out)) directly on a
	    local live across an earlier return).
	  - cp437.py/latin1.py's own encode() allocated their output bytearray
	    to the worst-case size (one output byte per INPUT byte) but multi-
	    byte UTF-8 input sequences collapse to a single output byte, so the
	    actually-written length (out_idx) can be less than that allocation
	    - wrapping the oversized, unfilled-tail buffer directly into the
	    returned bytes silently included trailing garbage. Fixed by
	    copying down to a final buffer sized to out_idx before returning. '''

	def setUp( self ) -> None:
		self.discovery = Discovery( import_builtins = True )
		self.compiler = Compiler( self.discovery )

	@unittest.skipUnless( test_support.HAS_CC, 'no C compiler (clang/gcc/msvc) found - skipping' )
	def test_programs_compile_and_run( self ) -> None:
		self.assert_programs_run([
			( 'ascii_round_trip', '''
from codecs.ascii import ascii

def main() -> i32:
	a = ascii()
	eb: bytes = a.encode( "Hello, World!" ).unwrap( 'encode failed' )
	s: str = a.decode( eb ).unwrap( 'decode failed' )
	if s != "Hello, World!":
		return 1
	return 0
''' ),
			( 'ascii_encode_out_of_range_errors', '''
from codecs.ascii import ascii

def main() -> i32:
	a = ascii()
	match a.encode( "héllo" ):
		case Result.Ok( b ):
			return 1
		case Result.Err( e ):
			pass
	return 0
''' ),
			( 'ascii_decode_out_of_range_errors', '''
from codecs.ascii import ascii

def main() -> i32:
	a = ascii()
	b = bytearray( 1 )
	p: Ptr[u8] = b.get_ptr()
	p[0] = 0xFF
	match a.decode( bytes( b )):
		case Result.Ok( s ):
			return 1
		case Result.Err( e ):
			pass
	return 0
''' ),
			( 'cp437_ascii_passthrough_round_trip', '''
from codecs.cp437 import cp437

def main() -> i32:
	c = cp437()
	eb: bytes = c.encode( "Hello, World!" ).unwrap( 'encode failed' )
	s: str = c.decode( eb ).unwrap( 'decode failed' )
	if s != "Hello, World!":
		return 1
	return 0
''' ),
			( 'cp437_extended_char_round_trip', '''
from codecs.cp437 import cp437

def main() -> i32:
	c = cp437()
	# accented/box-drawing chars only, no ASCII passthrough at all - also
	# exercises the output-buffer-trim fix (3 codepoints, 6 UTF-8 input
	# bytes, but only 3 CP437 output bytes)
	eb: bytes = c.encode( "éàü" ).unwrap( 'encode failed' )
	if len( eb ) != 3:
		return 1
	s: str = c.decode( eb ).unwrap( 'decode failed' )
	if s != "éàü":
		return 2
	return 0
''' ),
			( 'cp437_decode_raw_byte', '''
from codecs.cp437 import cp437

def main() -> i32:
	c = cp437()
	b = bytearray( 1 )
	p: Ptr[u8] = b.get_ptr()
	p[0] = 0x82 # cp437 0x82 -> DECODE_TABLE[2] -> U+00E9 (e-acute)
	s: str = c.decode( bytes( b )).unwrap( 'decode failed' )
	if s != "é":
		return 1
	return 0
''' ),
			( 'cp437_encode_unmappable_char_errors', '''
from codecs.cp437 import cp437

def main() -> i32:
	c = cp437()
	# U+1F600 (grinning face) is a 4-byte UTF-8 sequence - outside every
	# branch cp437's encode() handles (2-byte/3-byte only)
	match c.encode( "\U0001F600" ):
		case Result.Ok( b ):
			return 1
		case Result.Err( e ):
			pass
	return 0
''' ),
			( 'latin1_ascii_passthrough_round_trip', '''
from codecs.latin1 import latin1

def main() -> i32:
	l = latin1()
	eb: bytes = l.encode( "Hello, World!" ).unwrap( 'encode failed' )
	s: str = l.decode( eb ).unwrap( 'decode failed' )
	if s != "Hello, World!":
		return 1
	return 0
''' ),
			( 'latin1_extended_char_round_trip', '''
from codecs.latin1 import latin1

def main() -> i32:
	l = latin1()
	# U+00E9/U+00E0/U+00FC are all within Latin-1 range (<=0xFF) - also
	# exercises the output-buffer-trim fix (3 codepoints, 6 UTF-8 input
	# bytes, but only 3 Latin-1 output bytes)
	eb: bytes = l.encode( "éàü" ).unwrap( 'encode failed' )
	if len( eb ) != 3:
		return 1
	s: str = l.decode( eb ).unwrap( 'decode failed' )
	if s != "éàü":
		return 2
	return 0
''' ),
			( 'latin1_decode_raw_byte', '''
from codecs.latin1 import latin1

def main() -> i32:
	l = latin1()
	b = bytearray( 1 )
	p: Ptr[u8] = b.get_ptr()
	p[0] = 0xE9 # Latin-1 0xE9 IS U+00E9 (e-acute) directly
	s: str = l.decode( bytes( b )).unwrap( 'decode failed' )
	if s != "é":
		return 1
	return 0
''' ),
			( 'latin1_encode_out_of_range_errors', '''
from codecs.latin1 import latin1

def main() -> i32:
	l = latin1()
	# U+3042 (hiragana A) is a 3-byte UTF-8 sequence, codepoint > 0xFF -
	# outside Latin-1 range
	match l.encode( "あ" ):
		case Result.Ok( b ):
			return 1
		case Result.Err( e ):
			pass
	return 0
''' ),
		] )


class Base64Tests( test_support.RealCompileMixin, CompilerTestCase ):
	''' Real compile-and-run coverage for lib/base64.py - b64encode/b64decode,
	urlsafe_b64encode/urlsafe_b64decode, and b16encode/b16decode. See
	PLAN_HTTP_CLIENT.md, which names base64 as a zero-prerequisite piece
	needed for auth= (HTTP Basic -> base64 Authorization header). '''

	def setUp( self ) -> None:
		self.discovery = Discovery( import_builtins = True )
		self.compiler = Compiler( self.discovery )

	@unittest.skipUnless( test_support.HAS_CC, 'no C compiler (clang/gcc/msvc) found - skipping' )
	def test_programs_compile_and_run( self ) -> None:
		self.assert_programs_run([
			# RFC 4648 known-answer vectors - the standard "f"/"fo"/"foo"/
			# "foob"/"fooba"/"foobar" test vectors, each checked round-trip
			# (encode matches the known string, decode recovers the original)
			( 'b64_rfc4648_vectors_round_trip', '''
import base64

def check( plain: str, encoded: str ) -> bool:
	pb: bytes = plain.encode().unwrap( 'encode failed' )
	eb: bytes = base64.b64encode( pb )
	es: str = eb.decode().unwrap( 'decode of encoded output failed' )
	if es != encoded:
		return False
	db: bytes = base64.b64decode( eb ).unwrap( 'decode failed' )
	ds: str = db.decode().unwrap( 'decode of decoded output failed' )
	return ds == plain

def main() -> i32:
	if not check( '', '' ): return 1
	if not check( 'f', 'Zg==' ): return 2
	if not check( 'fo', 'Zm8=' ): return 3
	if not check( 'foo', 'Zm9v' ): return 4
	if not check( 'foob', 'Zm9vYg==' ): return 5
	if not check( 'fooba', 'Zm9vYmE=' ): return 6
	if not check( 'foobar', 'Zm9vYmFy' ): return 7
	return 0
''' ),
			# standard vs urlsafe alphabets diverge exactly on '+'/'/' vs
			# '-'/'_' - bytes 0xFB,0xFF,0xBF hit both symbols in both
			# alphabets, and urlsafe_b64decode must recover the original bytes
			( 'urlsafe_vs_standard_alphabet_divergence', '''
import base64

def main() -> i32:
	raw = bytearray( 3 )
	rp: Ptr[u8] = raw.get_ptr()
	rp[0] = 0xFB
	rp[1] = 0xFF
	rp[2] = 0xBF
	rb: bytes = bytes.from_bytearray( move( raw ) )

	std: bytes = base64.b64encode( rb )
	safe: bytes = base64.urlsafe_b64encode( rb )
	std_s: str = std.decode().unwrap( 'x' )
	safe_s: str = safe.decode().unwrap( 'x' )
	if std_s != '+/+/':
		return 1
	if safe_s != '-_-_':
		return 2

	back: bytes = base64.urlsafe_b64decode( safe ).unwrap( 'urlsafe decode failed' )
	if len( back ) != 3:
		return 3
	bp: ConstPtr[u8] = back.get_const_ptr()
	if bp[0] != 0xFB or bp[1] != 0xFF or bp[2] != 0xBF:
		return 4
	return 0
''' ),
			( 'b16_round_trip_and_casefold', '''
import base64

def main() -> i32:
	fb: bytes = 'foobar'.encode().unwrap( 'x' )
	hx: bytes = base64.b16encode( fb )
	hx_s: str = hx.decode().unwrap( 'x' )
	if hx_s != '666F6F626172':
		return 1

	unhex: bytes = base64.b16decode( hx ).unwrap( 'b16decode failed' )
	unhex_s: str = unhex.decode().unwrap( 'x' )
	if unhex_s != 'foobar':
		return 2

	# lowercase hex rejected by default (casefold=False)...
	lh: bytes = '666f6f626172'.encode().unwrap( 'x' )
	if base64.b16decode( lh ).is_ok():
		return 3
	# ...but accepted with casefold=True
	lh_ok: bytes = base64.b16decode( lh, casefold = True ).unwrap( 'casefold decode failed' )
	lh_ok_s: str = lh_ok.decode().unwrap( 'x' )
	if lh_ok_s != 'foobar':
		return 4
	return 0
''' ),
			# validate=True is the default (this codebase's own convention -
			# see guid.py/ascii.py - overriding Python's own lenient default),
			# so malformed input must be Result.Err in every case below
			( 'b64_and_b16_decode_error_cases', '''
import base64

def main() -> i32:
	bad_len: bytes = 'Zg'.encode().unwrap( 'x' ) # length 2, not a multiple of 4
	if base64.b64decode( bad_len ).is_ok():
		return 1

	bad_char: bytes = 'Z!=='.encode().unwrap( 'x' ) # '!' not in the alphabet
	if base64.b64decode( bad_char ).is_ok():
		return 2

	bad_pad: bytes = 'Zg=g'.encode().unwrap( 'x' ) # '=' not at the very end
	if base64.b64decode( bad_pad ).is_ok():
		return 3

	odd_hex: bytes = 'ABC'.encode().unwrap( 'x' ) # odd length
	if base64.b16decode( odd_hex ).is_ok():
		return 4
	return 0
''' ),
			# validate=False (opt-in) matches Python's own lenient default:
			# non-alphabet bytes (e.g. embedded whitespace) are discarded
			# before decoding, rather than rejected
			( 'b64_decode_lenient_mode', '''
import base64

def main() -> i32:
	with_ws: bytes = 'Zm9v\\nYmFy'.encode().unwrap( 'x' )
	if base64.b64decode( with_ws ).is_ok(): # strict default must reject the embedded newline
		return 1
	lenient: bytes = base64.b64decode( with_ws, validate = False ).unwrap( 'lenient decode failed' )
	lenient_s: str = lenient.decode().unwrap( 'x' )
	if lenient_s != 'foobar':
		return 2
	return 0
''' ),
		] )


class MatchArmSameNameNarrowingTests( test_support.RealCompileMixin, CompilerTestCase ):
	''' `match x: case T(x): ...` - the arm rebinds the SAME name as its
	own subject - used to crash outright (monomorphize.py silently
	clobbering a monomorphized generic union's own per-member constructor
	Functions back into plain, non-callable Variables - see
	monomorphize_class's own comment) and, once that crash was fixed,
	silently resolved the rebound name to its OLD (whole-union) type
	instead of the narrowed leaf (lowering.py's _stmt_Assign has no
	concept of a block-scoped shadow - reusing an existing name just
	re-lowers the RHS against the existing Variable's own fixed type).

	Fixed via a pure compile-time read-rewrite, not a new Variable: the
	rebound name's own Variable/storage is never touched. cfg.py tracks a
	CFG-scoped name->member map (narrow()/unnarrow()/narrowed_member(),
	pushed/popped via the same _Snapshot machinery _unchecked_results
	already uses), and lowering.py's _expr_Name rewrites a narrowed read
	into a borrowed GetAttr(data).GetAttr(v_member) chain in place of the
	raw (still union-typed) operand - no new incref/decref, x is still x. '''

	def setUp( self ) -> None:
		self.discovery = Discovery( import_builtins = True )
		self.compiler = Compiler( self.discovery )

	@unittest.skipUnless( test_support.HAS_CC, 'no C compiler (clang/gcc/msvc) found - skipping' )
	def test_programs_compile_and_run( self ) -> None:
		''' every real compile-and-run program in this class, merged into a
		single executable (one build for the whole class); a nonzero exit is
		decoded back to the failing sub-program and its own return code. '''
		self.assert_programs_run([
			# both arms actually exercised (a real Ok and a real Err value, not
			# just one) - a same-named Ok(r) that silently kept r's OLD
			# (whole-union) type would fail to resolve r.byte_len() at all
			# (Result has no byte_len()), so a clean compile here is already
			# meaningful; the exit-code checks confirm the NARROWED value
			# (the payload, not the union) is what's actually read
			( 'same_name_rebind_narrows_type_in_each_arm', '''
class MyError:
	pass

def make( n: i32 ) -> Result[str,MyError]:
	if n > 0:
		return Result.Ok( "hello" )
	return Result.Err( MyError() )

def helper( n: i32 ) -> usize:
	r: Result[str,MyError] = make( n )
	result: usize = 0
	match r:
		case Result.Ok( r ):
			result = r.byte_len()
		case Result.Err( e ):
			result = 0
	return result

def main() -> i32:
	if helper( 1 ) != 5:
		return 1
	if helper( -1 ) != 0:
		return 2
	return 0
''' ),
			# narrowing must NOT survive past the match statement's own block -
			# a second, independent match reusing the same name right after the
			# first must see the WHOLE union again (cfg.py's restore() pops
			# _narrowed back to whatever it was before the branch, unconditionally)
			( 'narrowing_confined_to_match_arm_reverts_after', '''
class MyError:
	pass

def make( n: i32 ) -> Result[str,MyError]:
	if n > 0:
		return Result.Ok( "hello" )
	return Result.Err( MyError() )

def helper( n: i32 ) -> usize:
	r: Result[str,MyError] = make( n )
	result: usize = 0
	match r:
		case Result.Ok( r ):
			result = r.byte_len()
		case Result.Err( e ):
			result = 0
	with compiler.wrap_arithmetic:
		match r:
			case Result.Ok( r ):
				result = result + r.byte_len()
			case Result.Err( e ):
				result = result + 100
	return result

def main() -> i32:
	if helper( 1 ) != 10:
		return 1
	if helper( -1 ) != 100:
		return 2
	return 0
''' ),
			# an UNRELATED match nested inside a match arm's own body - the
			# inner match's own snapshot()/restore() cycle (around its own
			# if/elif branches) must not clobber the OUTER arm's still-active
			# narrowing of r: restore() reverts _narrowed back to whatever it
			# was AT THAT BRANCH's OWN ENTRY (which already includes the
			# outer r->Ok narrowing), not wipe the whole dict. r.byte_len()
			# after the inner match, still inside the outer Ok arm, only
			# resolves at all if the outer narrowing survived the inner
			# match's own push/pop cycle
			# NB: outcome is ASSIGNED to an outer variable rather than returned
			# directly from inside the match arms - kept that way to isolate
			# THIS test's own narrowing concern from a `return` statement's own
			# epilogue-label handling (see MatchArmSameNameNarrowingTests's own
			# test_return_directly_inside_match_arm_compiles_and_runs, a
			# formerly-dangling-label bug now fixed by cfg.py's
			# enter_branch()/exit_branch()).
			( 'nested_match_preserves_outer_narrowing', '''
class MyError:
	pass

def helper( r: Result[str,MyError], r2: Result[str,MyError] ) -> i32:
	outcome: i32 = 0
	match r:
		case Result.Ok( r ):
			junk: usize = 0
			match r2:
				case Result.Ok( x ):
					junk = x.byte_len()
				case Result.Err( e ):
					junk = 0
			with compiler.wrap_arithmetic:
				outcome = i32( r.byte_len() )
		case Result.Err( e ):
			outcome = -2
	return outcome

def main() -> i32:
	if helper( Result.Ok( "hi" ), Result.Ok( "x" )) != 2:
		return 1
	if helper( Result.Ok( "hi" ), Result.Err( MyError() )) != 2:
		return 2
	if helper( Result.Err( MyError() ), Result.Ok( "x" )) != -2:
		return 3
	return 0
''' ),
			# a real RC-lifetime stress check, not just "doesn't crash once" -
			# if the narrowed read (lowering.py's _expr_Name rewrite, meant to
			# be a BORROWED GetAttr chain with no incref) instead double-
			# released the payload (the abandoned shadow-Variable design this
			# replaced would have), repeated construction/match/decref cycles
			# would corrupt the heap under repetition even if a single
			# iteration looked fine - checking the extracted value's actual
			# CONTENT (not just that it exists) every iteration catches a
			# use-after-free that a bare "did it crash" check could miss.
			# 'hello'.upper() (not the bare literal) forces a real heap
			# allocation - a literal binds to immortal static storage and
			# can't distinguish a leak/double-release from doing nothing.
			#
			# NB: still does NOT assert compiler.refcount(r) == 1 inside the
			# arm, even though the bug this comment used to describe (match-
			# statement lowering taking out a spurious extra retain of the
			# first case's own payload, computed from the raw subject before
			# __match_subj_N's own assignment even ran - see cfg.py's assign()
			# borrow= parameter) is now fixed - see
			# test_rc_lifetime_repeated_match_exact_refcount below for the
			# exact-count regression test that fix enabled. This test still
			# can't assert an exact count because of a SEPARATE, still-open
			# bug: make()'s own `return Result.Ok('hello'.upper())` - a single-
			# statement body - leaves the intermediate str temp's own release
			# instruction emitted AFTER the ir.Return in the generated C
			# (unreachable dead code), permanently inflating every Result
			# make() returns by one extra retain. Confirmed via direct
			# inspection of emit_c's output for make() alone; unrelated to
			# match/narrowing, flagged separately.
			( 'rc_lifetime_repeated_calls_no_leak', '''
class MyError:
	pass

def make() -> Result[str,MyError]:
	return Result.Ok( 'hello'.upper() )

def main() -> i32:
	i: i32 = 0
	while i < 1000:
		r: Result[str,MyError] = make()
		match r:
			case Result.Ok( r ):
				if r.byte_len() != 5:
					return 1
			case Result.Err( e ):
				return 2
		with compiler.wrap_arithmetic:
			i += 1
	return 0
''' ),
			# regression test for the match-subject-alias bug fixed via cfg.py's
			# assign() borrow= parameter: `match r:` used to lower
			# `__match_subj_N = r` as an owning COPY (its own Incref, paired
			# with its own epilogue Decref) even though r itself already owns a
			# live reference for the whole rest of its (function-scoped)
			# lifetime - the synthesized subject temp never needed an
			# independent one. That extra, always-superfluous retain showed up
			# in the generated C as a real retain_object() call computed
			# straight from the raw subject BEFORE __match_subj_N's own
			# assignment even ran (no tag check, always the union's first
			# member) - harmless in the sense that it was eventually balanced
			# by r's own release at scope exit, but it inflated every
			# compiler.refcount() read taken inside a match arm by exactly one,
			# and paid for a wholly unneeded retain/release pair on every match
			# execution.
			#
			# Unlike test_rc_lifetime_repeated_calls_no_leak above, this
			# constructs the Result INLINE in the same function as the match
			# (no separate make()-style helper returning it) specifically to
			# avoid that other, still-open, unrelated return-statement temp-
			# cleanup bug documented on that test - keeping this assertion an
			# exact, uncontaminated check of match-subject aliasing alone: r's
			# own ownership (1) plus x's own separately-tracked extraction-bind
			# (1) is exactly 2, every single one of 1000 iterations, on a real
			# heap allocation ('hello'.upper(), not a literal - see that test's
			# own comment for why).
			( 'rc_lifetime_repeated_match_exact_refcount', '''
class MyError:
	pass

def main() -> i32:
	with compiler.wrap_arithmetic:
		i: i32 = 0
		while i < 1000:
			r: Result[str,MyError] = Result.Ok( 'hello'.upper() )
			match r:
				case Result.Ok( x ):
					if compiler.refcount( x ) != 2:
						return 1
				case Result.Err( e ):
					return 2
			i += 1
		return 0
''' ),
			# a `return` as a match arm's own body (not assigned to an outer
			# variable first) - regression test for a real "undeclared label"
			# compile failure: the arm's own payload binding (s/e below) pushes
			# a fresh RC epilogue entry, and current_epilogue_label() used to
			# hand the `return` that entry's own label as its shared jump
			# target without knowing _stmt_If's own restore() was about to
			# silently discard that entry (branch-local, never merged past the
			# arm) - leaving a `goto` into a label build_epilogue_ladder()
			# never emitted. cfg.py's enter_branch()/exit_branch() now confine
			# it the same way enter_loop()/exit_loop() already did for a
			# loop-local RC entry, forcing an inline unwind instead of a
			# dangling shared label.
			( 'return_directly_inside_match_arm_compiles_and_runs', '''
class MyError:
	pass

def make( n: i32 ) -> Result[str,MyError]:
	if n > 0:
		return Result.Ok( "hello" )
	return Result.Err( MyError() )

def helper( n: i32 ) -> usize:
	r: Result[str,MyError] = make( n )
	match r:
		case Result.Ok( s ):
			return s.byte_len()
		case Result.Err( e ):
			return 0

def main() -> i32:
	if helper( 1 ) != 5:
		return 1
	if helper( -1 ) != 0:
		return 2
	return 0
''' ),
			# regression test for a real, previously-confirmed gap: a SECOND
			# union-shaped check (here, `x is None`) on a name the SAME arm
			# already narrowed via same-name reuse (`case str(x):`) used to
			# crash with "builtins.str has no attribute 'tag'" -
			# type_resolver.py's own is-None rewrite (visit_Compare) runs in an
			# earlier, purely AST-level pass that had no knowledge of cfg.py's
			# _narrowed state at all, so it kept reading x's OUTER declared
			# type (str|None, a TaggedUnion) and built a `.tag` access even
			# though x is already narrowed to plain `str` by the time
			# lowering.py actually processes this arm's body. Fixed by giving
			# _ReferenceResolver its own parallel self._narrowed dict (pushed
			# on entering a narrowing arm, popped on leaving it - see
			# visit_Match), consulted by _type_of_expr before falling back to
			# self.locals - flagged in the codebase's own comments (see
			# TypeIsInstanceofTests.test_rc_lifetime_repeated_calls_no_leak)
			# as "out of scope" until this test closed it.
			( 'is_none_recheck_on_already_narrowed_name_in_same_arm', '''
def describe( x: str|None ) -> i32:
	match x:
		case str( x ):
			if x is None:
				return 1
			return 0
		case None:
			return 2
	return 3

def main() -> i32:
	if describe( "hello" ) != 0:
		return 1
	if describe( None ) != 2:
		return 2
	return 0
''' ),
			# combines the previous test's in-arm recheck with
			# test_narrowing_confined_to_match_arm_reverts_after's own concern:
			# an already-narrowed x rechecked with `x is None` INSIDE its own
			# arm must not leave self._narrowed poisoned for code AFTER the
			# match - a second, independent `x is None` check right after the
			# match must still see the WHOLE str|None union again, exactly as
			# if the in-arm recheck had never happened
			( 'narrowed_name_reverts_after_arm_even_when_rechecked_inside', '''
def describe( x: str|None ) -> i32:
	with compiler.wrap_arithmetic:
		result: i32 = 0
		match x:
			case str( x ):
				if x is None:
					result = -1
				else:
					result = i32( x.byte_len() )
			case None:
				result = 999
		if x is None:
			result = result + 1000
		return result

def main() -> i32:
	if describe( "hello" ) != 5:
		return 1
	if describe( None ) != 1999:
		return 2
	return 0
''' ),
		] )


class MatchAnonymousUnionTests( test_support.RealCompileMixin, CompilerTestCase ):
	''' Phase 2 of PLAN_MATCH_NARROWING (see steady-dancing-haven.md): match
	support for anonymous unions - `case None:` (ast.MatchSingleton) and
	`case str(c):` (ast.MatchClass whose .cls is a bare ast.Name, not the
	pre-existing ast.Attribute-only `case Result.Ok(x):` path) against a
	T|None-shaped subject. Neither was parseable at all before - `case
	None:` fell through to "unsupported match pattern", `case str(c):`
	fell through to "unsupported match pattern class" (pattern.cls wasn't
	an ast.Attribute). This is a hard prerequisite for Phase 4's own
	if-to-match desugaring, since the desugar target IS a match statement.

	`case None:` reuses visit_Compare's own "does this union have a None
	member" tag-comparison logic (_resolved_union_members, factored out of
	both). `case str(c):` resolves the union off the SUBJECT's own static
	type (self._type_of_expr, newly wired up for __match_subj_N via
	visit_Match's self.locals[subj_name] assignment - the pre-existing
	`case Result.Ok(x):` path never needed this, since it reads the union
	off the PATTERN's own text instead) and finds the member by TYPE
	IDENTITY rather than by name, then shares the exact same tag-Cmp +
	payload-GetAttr codegen (_match_union_member) the named-member path
	already uses - including Phase 1's same-name narrowing. '''

	def setUp( self ) -> None:
		self.discovery = Discovery( import_builtins = True )
		self.compiler = Compiler( self.discovery )

	@unittest.skipUnless( test_support.HAS_CC, 'no C compiler (clang/gcc/msvc) found - skipping' )
	def test_programs_compile_and_run( self ) -> None:
		''' every real compile-and-run program in this class, merged into a
		single executable (one build for the whole class); a nonzero exit is
		decoded back to the failing sub-program and its own return code. '''
		self.assert_programs_run([
			# both `case None:` and `case str(s):` exercised for real, against
			# both a None and a non-None value passed through
			( 'none_and_leaf_type_patterns_both_arms', '''
def describe( x: str|None ) -> i32:
	match x:
		case None:
			return 0
		case str( s ):
			return 1
	return 2

def main() -> i32:
	if describe( None ) != 0:
		return 1
	if describe( "hi" ) != 1:
		return 2
	return 0
''' ),
			# case _: as the fallback arm, no case None: at all - confirms the
			# leaf-type-identity path doesn't require an exhaustive None arm
			( 'wildcard_fallback_arm', '''
def describe( x: str|None ) -> i32:
	match x:
		case str( s ):
			return 1
		case _:
			return 2

def main() -> i32:
	if describe( "hi" ) != 1:
		return 1
	if describe( None ) != 2:
		return 2
	return 0
''' ),
			# Phase 1's same-name narrowing, now reachable through the NEW
			# leaf-type-identity path too (match x: case str(x): ...) - x.
			# byte_len() only resolves at all if x was actually narrowed to
			# str, not left at its original str|None type
			( 'same_name_reuse_narrows_leaf_type_pattern', '''
def describe( x: str|None ) -> usize:
	result: usize = 0
	match x:
		case None:
			result = 999
		case str( x ):
			result = x.byte_len()
	return result

def main() -> i32:
	if describe( "hello" ) != 5:
		return 1
	if describe( None ) != 999:
		return 2
	return 0
''' ),
			# real RC-lifetime stress check under repetition, same rigor as
			# MatchArmSameNameNarrowingTests' own. 'hello'.upper() is assigned
			# to a plain `str`-typed local FIRST, then that local assigned into
			# the str|None-typed slot - NOT `x: str|None = 'hello'.upper()`
			# directly, which hits a separate, unrelated, pre-existing bug
			# (lowering.py's _lower_call shared tail using expected_type
			# directly for a call's own destination type instead of the
			# call's real return type, skipping union-coercion entirely -
			# confirmed via git-independent tracing, flagged separately, out
			# of scope here)
			( 'rc_lifetime_repeated_calls_no_leak', '''
def main() -> i32:
	with compiler.wrap_arithmetic:
		i: i32 = 0
		while i < 1000:
			s: str = 'hello'.upper()
			x: str|None = s
			match x:
				case None:
					return 1
				case str( c ):
					if c.byte_len() != 5:
						return 2
			i += 1
		return 0
''' ),
		] )

	def test_leaf_type_not_a_union_member_is_a_compile_error( self ) -> None:
		# a genuine mismatch (int is not a member of str|None) must stay a
		# real compile error, not get silently passed through - needs a
		# real call site to force describe()'s own body to actually be
		# resolved (this compiler resolves function bodies lazily, only
		# once reachable from main())
		self._run( '''
def describe( x: str|None ) -> i32:
	match x:
		case int( n ):
			return 1
		case _:
			return 2

def main() -> i32:
	return describe( "hi" )
''' )
		self.assertNotEqual( self.discovery.errors.errors, [] )


class TypeIsInstanceofTests( test_support.RealCompileMixin, CompilerTestCase ):
	''' Phase 3 of PLAN_MATCH_NARROWING (see steady-dancing-haven.md):
	`type(x) is T` / `type(x) is not T` / `instanceof(x, T)` syntax
	recognition. This compiler has no runtime reflection/RTTI (no
	vtables), so `type(x)` isn't a genuine "evaluates to a first-class
	type value" feature the way real Python's is - `type` and
	`instanceof` are never real registered names anywhere in lib/,
	textually recognized special syntax instead (same posture as
	move[T]/copy[T]/compiler.sizeof(...) elsewhere).

	type_resolver.py's visit_Compare recognizes `type(x) is T`/`is not T`
	(either side may be the type() call - `T is type(x)` works too) when
	x's own static type is a TaggedUnion (or Specialization of one) and T
	names one of its members, rewriting to the same tag-Cmp shape the
	pre-existing is/is-not-None rewrite and Phase 2's _match_union_member
	both already produce (factored to share _resolved_union_members).
	visit_Call recognizes instanceof(x, T) as sugar for the same thing,
	rewriting to the equivalent Compare and re-dispatching through
	self.visit() so it shares the identical codegen.

	Deliberately does NOT narrow x's type inside the if-branch - this
	phase only ever produces a plain bool, usable anywhere a bool
	expression is. Narrowing is Phase 4's job (if-statement desugaring to
	an equivalent match statement, which already narrows via Phase 1). '''

	def setUp( self ) -> None:
		self.discovery = Discovery( import_builtins = True )
		self.compiler = Compiler( self.discovery )

	@unittest.skipUnless( test_support.HAS_CC, 'no C compiler (clang/gcc/msvc) found - skipping' )
	def test_programs_compile_and_run( self ) -> None:
		''' every real compile-and-run program in this class, merged into a
		single executable (one build for the whole class); a nonzero exit is
		decoded back to the failing sub-program and its own return code. '''
		self.assert_programs_run([
			( 'type_is_and_is_not', '''
def describe( x: str|None ) -> i32:
	if type( x ) is str:
		return 1
	if type( x ) is not str:
		return 2
	return 3

def main() -> i32:
	if describe( "hi" ) != 1:
		return 1
	if describe( None ) != 2:
		return 2
	return 0
''' ),
			# T is type(x) - the type() call may be on either side of `is`
			( 'type_call_on_either_side', '''
def describe( x: str|None ) -> i32:
	if str is type( x ):
		return 1
	return 0

def main() -> i32:
	with compiler.wrap_arithmetic:
		return describe( "hi" ) - 1
''' ),
			( 'instanceof_sugar', '''
def describe( x: str|None ) -> i32:
	if instanceof( x, str ):
		return 1
	return 0

def main() -> i32:
	if describe( "hi" ) != 1:
		return 1
	if describe( None ) != 0:
		return 2
	return 0
''' ),
			# not just an if-condition - assignable to a bool local, same as
			# any other boolean expression
			( 'usable_as_plain_bool_value', '''
def describe( x: str|None ) -> bool:
	b: bool = type( x ) is str
	return b

def main() -> i32:
	if not describe( "hi" ):
		return 1
	if describe( None ):
		return 2
	return 0
''' ),
			# real RC-lifetime stress check under repetition, same rigor as
			# MatchAnonymousUnionTests' own (see that test's own comment for
			# why the call result is assigned to a plain `str` local first,
			# not directly into the str|None-typed slot).
			#
			# NB: does NOT check `x is None` (or any other union-shaped
			# recheck of x) inside the type(x) is str branch - this test just
			# confirms x is correctly usable AS its narrowed str type. A
			# second union-shaped check on an already-narrowed name in the
			# same arm used to crash outright (type_resolver.py's own is-
			# None/type-is rewrites ran before lowering with no awareness of
			# cfg.py's narrowing state) - now fixed via _ReferenceResolver's
			# own self._narrowed tracking; see
			# TypeIsIfDesugaringTests.test_is_none_recheck_on_already_narrowed_name_in_type_branch
			# and MatchArmSameNameNarrowingTests.test_is_none_recheck_on_already_narrowed_name_in_same_arm
			# for the dedicated regression coverage.
			( 'rc_lifetime_repeated_calls_no_leak', '''
def main() -> i32:
	with compiler.wrap_arithmetic:
		i: i32 = 0
		while i < 1000:
			s: str = 'hello'.upper()
			x: str|None = s
			if type( x ) is str:
				if x.byte_len() != 5:
					return 1
			else:
				return 2
			i += 1
		return 0
''' ),
		] )

	def test_leaf_type_not_a_union_member_is_a_compile_error( self ) -> None:
		# a genuine mismatch (int is not a member of str|None) must stay a
		# real compile error, not get silently passed through - needs a
		# real call site to force describe()'s own body to actually be
		# resolved (this compiler resolves function bodies lazily, only
		# once reachable from main())
		self._run( '''
def describe( x: str|None ) -> i32:
	if type( x ) is int:
		return 1
	return 0

def main() -> i32:
	return describe( "hi" )
''' )
		self.assertNotEqual( self.discovery.errors.errors, [] )

	def test_both_sides_type_call_is_a_compile_error( self ) -> None:
		# type(x) is type(y) - neither side names a real type, a genuine
		# misuse of the syntax, must stay a real compile error
		self._run( '''
def describe( x: str|None, y: str|None ) -> i32:
	if type( x ) is type( y ):
		return 1
	return 0

def main() -> i32:
	return describe( "hi", "bye" )
''' )
		self.assertNotEqual( self.discovery.errors.errors, [] )


class TypeIsIfDesugaringTests( test_support.RealCompileMixin, CompilerTestCase ):
	''' Phase 4 of PLAN_MATCH_NARROWING (see steady-dancing-haven.md):
	desugar `if type(x) is T: A else: B` into the equivalent `match x:
	case T(x): A \n case _: B` BEFORE lowering ever sees it -
	type_resolver.py's new _try_desugar_type_is_if, hooked into visit_If
	right where _try_fold_is_rc_if already runs its own compile-time-only
	if-rewrite. `if instanceof(x, T):` is the identical rewrite, recognized
	directly (never routed through visit_Call's own Compare-detour, since
	the shape has to survive intact for this check to see it at all).

	Unlike Phase 3 alone (`type(x) is T` as a bare boolean, anywhere), this
	actually NARROWS x inside the T-branch - reusing Phase 1's own
	same-name match-arm narrowing entirely for free, since the synthesized
	case pattern binds the SAME name as the subject whenever x is a bare
	Name. An elif chain desugars into a NESTED match purely as a side
	effect of the wildcard arm's own body (the original orelse) being
	visited normally - no special elif-chain code needed at all. Real
	compile-and-run tests, full suite green (bash + PowerShell).

	Deliberately does NOT implement TODO.txt's "narrowing survives past a
	non-fallthrough branch" case (see the plan's own "Explicitly deferred"
	section) - narrowing here is scoped to the match statement's own
	block, exactly like it already is for a literal `match` statement. '''

	def setUp( self ) -> None:
		self.discovery = Discovery( import_builtins = True )
		self.compiler = Compiler( self.discovery )

	@unittest.skipUnless( test_support.HAS_CC, 'no C compiler (clang/gcc/msvc) found - skipping' )
	def test_programs_compile_and_run( self ) -> None:
		''' every real compile-and-run program in this class, merged into a
		single executable (one build for the whole class); a nonzero exit is
		decoded back to the failing sub-program and its own return code. '''
		self.assert_programs_run([
			# x.byte_len() only resolves at all if x was actually narrowed to
			# str inside the branch, not left at its original str|None type
			( 'narrows_inside_the_type_branch', '''
def describe( x: str|None ) -> usize:
	if type( x ) is str:
		return x.byte_len()
	else:
		return 999

def main() -> i32:
	if describe( "hello" ) != 5:
		return 1
	if describe( None ) != 999:
		return 2
	return 0
''' ),
			# ordinary code AFTER the if must still see x at its original,
			# unnarrowed type - a second, independent `x is None` check right
			# after the if must still work as an ordinary union check
			( 'narrowing_does_not_survive_past_the_if', '''
def describe( x: str|None ) -> usize:
	with compiler.wrap_arithmetic:
		result: usize = 0
		if type( x ) is str:
			result = x.byte_len()
		else:
			result = 999
		if x is None:
			result = result + 1000
		return result

def main() -> i32:
	if describe( "hello" ) != 5:
		return 1
	if describe( None ) != 1999:
		return 2
	return 0
''' ),
			( 'no_else_clause', '''
def describe( x: str|None ) -> usize:
	result: usize = 999
	if type( x ) is str:
		result = x.byte_len()
	return result

def main() -> i32:
	if describe( "hello" ) != 5:
		return 1
	if describe( None ) != 999:
		return 2
	return 0
''' ),
			# type(x) is not T swaps which body lands in the T-arm vs the
			# wildcard arm - the else branch (T-arm) still narrows x
			( 'negated_type_is_not', '''
def describe( x: str|None ) -> usize:
	if type( x ) is not str:
		return 999
	else:
		return x.byte_len()

def main() -> i32:
	if describe( "hello" ) != 5:
		return 1
	if describe( None ) != 999:
		return 2
	return 0
''' ),
			# instanceof(x, T) directly as an if's own condition - recognized
			# without ever detouring through visit_Call's own Compare rewrite
			( 'instanceof_as_if_condition', '''
def describe( x: str|None ) -> usize:
	if instanceof( x, str ):
		return x.byte_len()
	else:
		return 999

def main() -> i32:
	if describe( "hello" ) != 5:
		return 1
	if describe( None ) != 999:
		return 2
	return 0
''' ),
			# a three-member, non-None-paired union (i32|str|None) with a real
			# elif chain - each arm narrows independently, the final else
			# still reachable
			( 'elif_chain_desugars_to_nested_match', '''
def describe( x: i32|str|None ) -> i32:
	with compiler.wrap_arithmetic:
		if type( x ) is i32:
			return x + 100
		elif type( x ) is str:
			return i32( x.byte_len() )
		else:
			return -1

def main() -> i32:
	if describe( 5 ) != 105:
		return 1
	if describe( "hello" ) != 5:
		return 2
	if describe( None ) != -1:
		return 3
	return 0
''' ),
			# real RC-lifetime stress check under repetition, same rigor as
			# every other RC test this session established
			( 'rc_lifetime_repeated_calls_no_leak', '''
def main() -> i32:
	with compiler.wrap_arithmetic:
		i: i32 = 0
		while i < 1000:
			s: str = 'hello'.upper()
			x: str|None = s
			if type( x ) is str:
				if x.byte_len() != 5:
					return 1
			else:
				return 2
			i += 1
		return 0
''' ),
			# regression test for the same gap MatchArmSameNameNarrowingTests.
			# test_is_none_recheck_on_already_narrowed_name_in_same_arm covers
			# for a literal `match` statement, reached here through if-
			# desugaring instead: `if type(x) is str:` desugars to `match x:
			# case str(x): ...` (this class's own docstring), which narrows x
			# the same way - a SECOND union-shaped check (`x is None`) on that
			# already-narrowed x, inside the SAME branch, used to crash with
			# "builtins.str has no attribute 'tag'" since type_resolver.py's
			# own rewrites ran with no knowledge of the narrowing their own
			# desugaring had just introduced. Previously flagged as out of
			# scope on TypeIsInstanceofTests.test_rc_lifetime_repeated_calls_
			# no_leak's own comment; fixed by _ReferenceResolver's new
			# self._narrowed tracking (see type_resolver.py's visit_Match).
			( 'is_none_recheck_on_already_narrowed_name_in_type_branch', '''
def describe( x: str|None ) -> i32:
	if type( x ) is str:
		if x is None:
			return 1
		return 0
	return 2

def main() -> i32:
	if describe( "hello" ) != 0:
		return 1
	if describe( None ) != 2:
		return 2
	return 0
''' ),
		] )

	def test_leaf_type_not_a_union_member_is_a_compile_error( self ) -> None:
		# a genuine mismatch must stay a real compile error even when the
		# condition is an if-statement's own (desugar-eligible) test
		self._run( '''
def describe( x: str|None ) -> i32:
	if type( x ) is int:
		return 1
	return 0

def main() -> i32:
	return describe( "hi" )
''' )
		self.assertNotEqual( self.discovery.errors.errors, [] )


class NarrowingSurvivalTests( test_support.RealCompileMixin, CompilerTestCase ):
	''' Phases 5-6 of PLAN_MATCH_NARROWING (see steady-dancing-haven.md):
	narrowing surviving PAST its own match/if statement, not just confined
	to one arm - the piece explicitly deferred at the end of Phase 4.

	cfg.py's merge_if now reconciles _narrowed the same way it already
	reconciles bindings/unchecked-Results: when exactly one branch
	terminates (return/break/continue), the survivor's own narrowed state
	carries forward; when neither terminates, two branches that BOTH
	narrowed the same name - even to DIFFERENT members - UNION together
	(x proven int on one path, str on the other, is real information: it
	rules out every OTHER member) rather than being discarded just because
	they disagree about which one specifically.

	type_resolver.py's visit_Match separately recognizes when a match is
	PROVABLY exhaustive (a literal wildcard, or explicit cases that
	collectively cover every one of the union's own members) and splices
	the final arm's body in unconditionally instead of chaining it behind
	a now-provably-redundant tag check - without this, merge_if's own
	correct-per-its-own-rules soft merge would see an ambiguous, empty
	"else" as a competing (unnarrowed) path and drop the narrowing before
	it ever reached the real join point. For a 2-member union specifically,
	an UNNAMED wildcard (case _:, not a named capture) also gets narrowed
	to the union's own remaining OTHER member, deduced from its sibling
	case - this is what makes `if isinstance(x, int): return` (no else at
	all) narrow x to str afterward. Real compile-and-run tests, full suite
	green (bash + PowerShell). '''

	def setUp( self ) -> None:
		self.discovery = Discovery( import_builtins = True )
		self.compiler = Compiler( self.discovery )

	@unittest.skipUnless( test_support.HAS_CC, 'no C compiler (clang/gcc/msvc) found - skipping' )
	def test_programs_compile_and_run( self ) -> None:
		''' every real compile-and-run program in this class, merged into a
		single executable (one build for the whole class); a nonzero exit is
		decoded back to the failing sub-program and its own return code. '''
		self.assert_programs_run([
			# the user's own motivating example: no else at all - x is
			# narrowed to str for the rest of the function purely because the
			# int branch terminates. instanceof(x, T) (not Python's real
			# isinstance, which this compiler doesn't recognize) is this
			# compiler's own sugar for type(x) is T - see Phase 3
			( 'headline_example_isinstance_return_no_else', '''
def describe( x: i32|str ) -> usize:
	with compiler.wrap_arithmetic:
		if instanceof( x, i32 ):
			return 999
		return x.byte_len()

def main() -> i32:
	if describe( "hello" ) != 5:
		return 1
	if describe( 42 ) != 999:
		return 2
	return 0
''' ),
			# a real match, both members named explicitly (no wildcard at
			# all) - the surviving (non-terminating) arm's own narrowing still
			# carries forward, exercising merge_if's own survivor-wins path
			# directly rather than the wildcard/negation trick
			( 'both_arms_explicit_one_terminates', '''
def describe( x: i32|str ) -> usize:
	with compiler.wrap_arithmetic:
		match x:
			case i32( x ):
				return 999
			case str( x ):
				pass
		return x.byte_len()

def main() -> i32:
	if describe( "hello" ) != 5:
		return 1
	if describe( 42 ) != 999:
		return 2
	return 0
''' ),
			# neither arm terminates and they narrow to DIFFERENT members
			# (i32 vs str) - the merged post-match state is neither "i32
			# only" nor "unnarrowed", it's "one of i32|str" (None ruled out) -
			# a subsequent type(x) is i32 check must still resolve correctly
			# (i32 IS one of the remaining possibilities, ambiguous, so this
			# falls through to an ordinary tag check against x's own full
			# declared type rather than being folded outright - the important
			# thing is it doesn't error as "not a union type", which is
			# exactly what happened before the union-instead-of-discard fix)
			( 'disagreeing_arms_union_instead_of_discard', '''
def describe( x: i32|str|None ) -> i32:
	with compiler.wrap_arithmetic:
		match x:
			case i32( x ):
				pass
			case str( x ):
				pass
			case _:
				return -1
		if type( x ) is i32:
			return 1
		return 2

def main() -> i32:
	if describe( 5 ) != 1:
		return 1
	if describe( "hi" ) != 2:
		return 2
	if describe( None ) != -1:
		return 3
	return 0
''' ),
			# no else clause at all - the implicit "fell through" path IS the
			# match's own second arm once the union is fully covered
			( 'no_explicit_else_still_narrows', '''
def describe( x: i32|str ) -> usize:
	with compiler.wrap_arithmetic:
		result: usize = 0
		if type( x ) is i32:
			return 999
		return x.byte_len()

def main() -> i32:
	if describe( "hello" ) != 5:
		return 1
	if describe( 1 ) != 999:
		return 2
	return 0
''' ),
			# a 3+-member union can't express "one of the two remaining
			# possibilities" with the single-member narrowing design (that's
			# the abandoned UnionView scope) - the wildcard arm correctly
			# stays UNNARROWED rather than guessing, which is still a real,
			# valid compile (not an error). Both a str AND an i32 argument are
			# passed at real call sites (not just i32) - a union member that's
			# never actually constructed anywhere reachable hits a separate,
			# pre-existing, unrelated scheduling gap (confirmed independent of
			# this work: reproduces for an i32|str|bool parameter with no
			# narrowing/matching involved at all) where its own RC cleanup
			# code fails to compile against an incomplete forward declaration
			( 'three_member_union_wildcard_declines_gracefully', '''
def describe( x: i32|str|bool ) -> i32:
	if type( x ) is i32:
		return 1
	return 0

def main() -> i32:
	if describe( 5 ) != 1:
		return 1
	return describe( "hi" )
''' ),
			# TODO.txt's own original worked example (now with real syntax) -
			# an elif chain over a 3-member union desugars into nested
			# matches, each arm narrowing independently within its own scope;
			# a 3-member union can't narrow the FINAL else (see the 3-member
			# decline test above) but the two explicit arms still work
			# correctly on their own
			( 'todo_elif_chain_worked_example', '''
def describe( x: i32|str|bool ) -> i32:
	with compiler.wrap_arithmetic:
		if type( x ) is i32:
			return x + 100
		elif type( x ) is str:
			return i32( x.byte_len() )
		else:
			return -1

def main() -> i32:
	if describe( 5 ) != 105:
		return 1
	if describe( "hello" ) != 5:
		return 2
	if describe( True ) != -1:
		return 3
	return 0
''' ),
			# real RC-lifetime stress check under repetition, same rigor as
			# every other RC test this session established - narrowing
			# surviving past the if (via type(x) is str, not a plain `is
			# None` check), then reading the narrowed str repeatedly
			( 'rc_lifetime_repeated_calls_no_leak', '''
def main() -> i32:
	with compiler.wrap_arithmetic:
		i: i32 = 0
		while i < 1000:
			s: str = 'hello'.upper()
			x: str|None = s
			if type( x ) is str:
				pass
			else:
				return 1
			if x.byte_len() != 5:
				return 2
			i += 1
		return 0
''' ),
			# the plain `is not None`/`is None` rewrite ALSO narrows now
			# (this plan's own item 4 - previously ONLY type(x) is T/
			# instanceof/match narrowed; a bare is-not-None check on a
			# real T|None union did not, at all). Both the in-body
			# narrowing AND post-if survival (the None branch returns) are
			# exercised together, under the same repeated-call RC-lifetime
			# rigor as the case just above
			( 'is_not_none_narrows_body_and_survives_past_the_if', '''
def main() -> i32:
	with compiler.wrap_arithmetic:
		i: i32 = 0
		while i < 1000:
			s: str = 'hello'.upper()
			x: str|None = s
			if x is None:
				return 1
			# narrowing survived the whole if - x is str here, not str|None
			if x.byte_len() != 5:
				return 2
			i += 1
		return 0
''' ),
			# a match's own case arm, not just an if-branch, ending in a call
			# to a -> NoReturn function (sys.panic) must ALSO count as
			# "never reaches the match's own join point" - type_resolver.py's
			# visit_Match had the identical syntactic-only terminates check
			# _stmt_If had before its own NoReturn fix (see PLAN_COMPILER_
			# BUG_SWEEP.md). Ordinary expressions (arithmetic, attribute
			# access, ...) after the match are NOT a useful test here - those
			# are type-checked by lowering.py's OWN, separate, already-correct
			# cfg-based narrowing over the if-chain visit_Match desugars into,
			# regardless of whether THIS bug is fixed. The observable effect is
			# narrower: type_resolver.py's own _narrowed dict backs its
			# _rewrite_type_is_comparison fold-to-constant optimization for a
			# LATER type(x) is T check - without the fix, that rewrite assumes
			# s is STILL union-typed at this point (since its own bookkeeping
			# never recorded the narrowing) and emits a tag-comparison against
			# it, while lowering.py's OWN independent narrowing has ALREADY
			# narrowed s's real, lowered type to plain usize by here - the
			# mismatch produces invalid C ('usize' has no '.tag' member),
			# confirmed via a real repro before this fix existed
			( 'match_arm_sys_panic_narrows_past_the_match', '''
def classify( s: usize|None ) -> bool:
	match s:
		case None:
			sys.panic( 'unreachable' )
		case _:
			pass
	return type( s ) is usize

def main() -> i32:
	if not classify( usize( 5 ) ):
		return 1
	return 0
''' ),
			# a case arm ending in an ORDINARY receiver method call
			# (self.touch(), not sys.panic) must NOT be mistaken for a
			# diverging arm, and - the actual regression caught while
			# building the above fix - must not crash the compiler either.
			# _stmt_diverges resolves a bare call's callee via
			# _resolve_callee_target, which walks discovery's scope-stack-
			# based find_name - that raises (not returns None) for a
			# receiver rooted in a local like self, since locals live in
			# this resolver's own self.locals dict, never in discovery's
			# scope stack. The raise ALSO permanently records a bogus "name
			# 'self' is not defined" error (discovery.fail's own contract),
			# so even catching the exception wasn't enough to fix the first
			# attempt at this. Guarded via discovery.find_name_or_none directly
			# (not a self.locals membership check, which turned out to be an
			# incomplete record of "this name is local" - see the next test)
			( 'match_arm_receiver_call_does_not_crash_the_compiler', '''
class Widget:
	touched: i32

	def __init__( self ) -> None:
		self.touched = 0

	def touch( self ) -> None:
		self.touched = 1

	# the last statement of a case arm being a BARE receiver call
	# (self.touch(), no return/assignment wrapping it) is the exact shape
	# that crashed - _stmt_diverges only even LOOKS at a stmt shaped like
	# ast.Expr(ast.Call(...)); a return/assign short-circuits before ever
	# reaching the receiver-call resolution this test guards
	def maybe_touch( self, r: Result[i32,i32] ) -> None:
		match r:
			case Result.Ok( v ):
				self.touch()
			case Result.Err( e ):
				pass

def main() -> i32:
	w: Widget = Widget()
	w.maybe_touch( Result.Ok( 5 ) )
	if w.touched != 1:
		return 1
	return 0
''' ),
			# a case arm's own PATTERN-BOUND name (case Result.Ok(w):, w bound
			# fresh by the match itself, not a pre-existing local/parameter)
			# used as a receiver in the arm's last statement - the real,
			# already-merged regression this whole test class caught: w's own
			# binding is a plain ast.Assign spliced directly into the case
			# body by _match_pattern/_match_union_member, never routed through
			# self.visit()/visit_Assign, so it never updated self.locals -
			# the ORIGINAL fix's self.locals membership check let w.close()
			# through uncaught, same crash as the self.touch() case above,
			# just for a different reason. Matches the exact real-world shape
			# that surfaced this (lib/builtins/__File.py's File.binary_writer
			# used via `match ...: case Result.Ok(w): w.write(...); w.close()`)
			( 'match_bound_name_receiver_call_does_not_crash_the_compiler', '''
class Widget:
	def touch( self ) -> None:
		pass

	# w.close(), with NOTHING after it in this arm, is the exact shape
	# that crashed - a return/assign wrapping it would short-circuit
	# _stmt_diverges before ever reaching the receiver-call resolution
	# this test guards, same reasoning as the self.touch() case above
	def close( self ) -> None:
		pass

def get( ok: bool ) -> Result[Widget,i32]:
	if ok:
		return Result.Ok( Widget() )
	return Result.Err( -1 )

def use( ok: bool ) -> None:
	match get( ok ):
		case Result.Ok( w ):
			w.touch()
			w.close()
		case Result.Err( e ):
			sys.panic( 'unreachable in this test' )

def main() -> i32:
	use( True )
	return 0
''' ),
		] )

	@unittest.skipUnless( _CC is not None, 'no C compiler (clang/gcc/msvc) found - skipping' )
	def test_named_wildcard_does_not_narrow( self ) -> None:
		# case y: (a NAMED capture) means "rebind the whole union", not
		# "prove the other member" - unlike case _: (bare, unnamed), this
		# must NOT narrow: y.byte_len() must still fail to resolve since y
		# stays union-typed (i32 has no byte_len())
		self._run( '''
def describe( x: i32|str ) -> usize:
	match x:
		case i32( x ):
			return 999
		case y:
			return y.byte_len()

def main() -> i32:
	return describe( "hi" )
''' )
		self.assertNotEqual( self.discovery.errors.errors, [] )


class WhileNarrowingTests( test_support.RealCompileMixin, CompilerTestCase ):
	''' Phase 7 of PLAN_MATCH_NARROWING (see steady-dancing-haven.md):
	`while type(x) is T:`/`while type(x) is not T:`/`while instanceof(x,
	T):` against a bare-Name, union-typed x - narrows x for the loop
	BODY's own duration, and separately narrows x for code AFTER the loop
	once it exits (the condition is checked at least once even for a
	zero-iteration loop, so this holds regardless of how many times the
	body actually ran - no "runs at least once" reasoning needed, unlike a
	body-always-does-X kind of claim would). `is`'s own exit narrowing
	needs a 2-member union (same "not T uniquely determines the other
	member" reasoning the if/match wildcard case has); `is not`'s own
	exit narrowing works for ANY union size - `not(x is not T)` means `x
	is T` directly, no disambiguation needed.

	Confirms the user's own worked example: `while isinstance(x, int):
	return` (return inside the body) narrows x to str AFTER the loop
	purely via the natural exit path - the return contributes NOTHING to
	that fact (it exits the FUNCTION, never reaches "after the loop" at
	all - see steady-dancing-haven.md's own "Context" section). Real
	compile-and-run tests, full suite green (bash + PowerShell). '''

	def setUp( self ) -> None:
		self.discovery = Discovery( import_builtins = True )
		self.compiler = Compiler( self.discovery )

	@unittest.skipUnless( test_support.HAS_CC, 'no C compiler (clang/gcc/msvc) found - skipping' )
	def test_programs_compile_and_run( self ) -> None:
		''' every real compile-and-run program in this class, merged into a
		single executable (one build for the whole class); a nonzero exit is
		decoded back to the failing sub-program and its own return code. '''
		self.assert_programs_run([
			( 'narrows_the_loop_body', '''
def main() -> i32:
	with compiler.wrap_arithmetic:
		x: i32|str = 5
		total: i32 = 0
		while type( x ) is i32:
			total = total + x
			x = "done"
		if total != 5:
			return 1
		return 0
''' ),
			# a 2-member union, `is not` form - the body is only entered while
			# x is NOT i32, i.e. while it's str
			( 'is_not_narrows_the_loop_body_to_the_other_member', '''
def main() -> i32:
	with compiler.wrap_arithmetic:
		x: i32|str = "hi"
		count: i32 = 0
		while type( x ) is not i32:
			if x.byte_len() != 2:
				return 1
			count = count + 1
			x = 5
		if count != 1:
			return 2
		return 0
''' ),
			# post-loop exit narrowing - x.byte_len() only resolves at all if
			# x was actually narrowed to str once the loop's own condition
			# went false
			( 'narrows_after_the_loop_exits', '''
def main() -> i32:
	with compiler.wrap_arithmetic:
		x: i32|str = 5
		while type( x ) is i32:
			x = "done"
		if x.byte_len() != 4:
			return 1
		return 0
''' ),
			# the user's own example: a `return` inside the loop body
			# contributes NOTHING to post-loop narrowing (it exits the
			# function, never reaches "after the loop") - x is narrowed to
			# str after the loop purely via the natural exit path
			( 'user_worked_example_return_inside_while', '''
def describe( x: i32|str ) -> i32:
	with compiler.wrap_arithmetic:
		while type( x ) is i32:
			return 999
		return i32( x.byte_len() )

def main() -> i32:
	if describe( "hello" ) != 5:
		return 1
	if describe( 1 ) != 999:
		return 2
	return 0
''' ),
			# a 3+-member union: the `is` form still narrows the BODY
			# correctly (single-member narrowing to the matched member is
			# always well-defined), even though exit-narrowing can't apply
			# (declines gracefully, no error, x just stays unnarrowed after)
			( 'three_member_union_body_narrows_but_exit_declines', '''
def main() -> i32:
	with compiler.wrap_arithmetic:
		x: i32|str|bool = 5
		total: i32 = 0
		while type( x ) is i32:
			total = total + x
			x = "done"
		if total != 5:
			return 1
		return 0
''' ),
			# real RC-lifetime stress check under repetition, same rigor as
			# every other RC test this session established - the outer loop
			# runs the whole while-narrowing construct 1000 times
			( 'rc_lifetime_repeated_loop_executions_no_leak', '''
def main() -> i32:
	with compiler.wrap_arithmetic:
		i: i32 = 0
		while i < 1000:
			x: str|None = 'hello'.upper()
			while type( x ) is not str:
				return 1
			if x.byte_len() != 5:
				return 2
			i += 1
		return 0
''' ),
		] )


class LoopBreakNarrowingTests( test_support.RealCompileMixin, CompilerTestCase ):
	''' Phase 8 of PLAN_MATCH_NARROWING (see steady-dancing-haven.md):
	`break`-based narrowing survival past a `while`/`for` loop, generalizing
	Phase 7's own condition-based exit narrowing to loops whose exit isn't
	tied to a `type(x) is T` condition at all.

	The key semantic, confirmed by working through several test shapes
	that initially seemed like they should narrow but correctly don't: a
	name survives PAST a loop only if EVERY way of reaching that point
	agrees. For an ordinary (non-`while True`) loop, the loop's own
	NATURAL (condition-false, or range-exhausted) exit is always a real,
	competing candidate - if the condition itself doesn't prove anything
	about the name (an unrelated `while i < 3:`, or a for-loop's own range
	check), that candidate contributes NOTHING, and a `break`'s own
	narrowing - even if it's the only break in the whole loop - gets
	dropped by the merge, same as any other disagreement. This is correct,
	not a bug: nothing guarantees the break is ever actually taken. The
	loop's own natural exit only becomes a non-issue when it's PROVABLY
	unreachable (`while True:` with no other exit condition at all) or
	when it already agrees (the name was ALREADY narrowed before the loop
	even started, and nothing inside changes that).

	Real compile-and-run tests, full suite green (bash + PowerShell). '''

	def setUp( self ) -> None:
		self.discovery = Discovery( import_builtins = True )
		self.compiler = Compiler( self.discovery )

	@unittest.skipUnless( test_support.HAS_CC, 'no C compiler (clang/gcc/msvc) found - skipping' )
	def test_programs_compile_and_run( self ) -> None:
		''' every real compile-and-run program in this class, merged into a
		single executable (one build for the whole class); a nonzero exit is
		decoded back to the failing sub-program and its own return code. '''
		self.assert_programs_run([
			# while True: has no natural exit at all (provably unreachable) -
			# the single break is the ONLY way out, so its own narrowing
			# survives unconditionally. s: str = x only compiles if x was
			# actually narrowed to str (its raw type is the whole union)
			( 'while_true_break_narrows', '''
def main() -> i32:
	with compiler.wrap_arithmetic:
		x: i32|str = "hello"
		while True:
			if type( x ) is str:
				break
		s: str = x
		if s.byte_len() != 5:
			return 1
		return 0
''' ),
			# x is narrowed to str BEFORE the for-loop even starts (Phase 5's
			# own if-survival) - the for-loop's own natural exit (range
			# exhausted, or zero iterations) inherits that same fact from
			# loop_snapshot, and the break inside doesn't disturb it, so both
			# the natural-exit and break candidates agree
			( 'for_loop_break_survives_when_already_narrowed', '''
def main() -> i32:
	with compiler.wrap_arithmetic:
		x: i32|str = "hello"
		if type( x ) is str:
			for i in range( 3 ):
				if i == 1:
					break
			s: str = x
			if s.byte_len() != 5:
				return 1
		return 0
''' ),
			# an ordinary while loop (condition unrelated to x) - the natural,
			# condition-false exit is a real, reachable path that proves
			# NOTHING about x, so even a single break's own narrowing is
			# correctly dropped by the merge (not every way of reaching this
			# point agrees) - the code still compiles and runs correctly via
			# the ordinary, un-narrowed path, it's just not narrowed
			( 'ordinary_loop_natural_exit_drops_break_narrowing', '''
def describe( x: i32|str ) -> i32:
	with compiler.wrap_arithmetic:
		i: usize = 0
		while i < 3:
			if type( x ) is str:
				break
			i += 1
		if type( x ) is str:
			return 1
		return 2

def main() -> i32:
	if describe( "hi" ) != 1:
		return 1
	if describe( 5 ) != 2:
		return 2
	return 0
''' ),
			# two DIFFERENT breaks (different source locations, only one ever
			# actually reachable for a given call) narrowing to DIFFERENT
			# members - the compiler conservatively treats both as real,
			# competing candidates and the soft-merge drops the disagreement,
			# same as merge_if's own disagreeing-branches behavior - genuine
			# disagreement silently stays unnarrowed, never a compile error
			( 'disagreeing_breaks_drop_narrowing_not_error', '''
def describe( x: i32|str ) -> i32:
	with compiler.wrap_arithmetic:
		i: usize = 0
		while i < 3:
			if type( x ) is i32:
				break
			if type( x ) is str:
				break
			i += 1
		if type( x ) is i32:
			return 1
		return 2

def main() -> i32:
	if describe( 5 ) != 1:
		return 1
	if describe( "hi" ) != 2:
		return 2
	return 0
''' ),
			# real RC-lifetime stress check under repetition, same rigor as
			# every other RC test this session established - the break-
			# narrowed str is read (and its own real content checked) every
			# iteration of the outer stress loop
			( 'rc_lifetime_repeated_break_narrowing_no_leak', '''
def main() -> i32:
	with compiler.wrap_arithmetic:
		i: i32 = 0
		while i < 1000:
			x: str|None = 'hello'.upper()
			while True:
				if type( x ) is str:
					break
			s: str = x
			if s.byte_len() != 5:
				return 1
			i += 1
		return 0
''' ),
		] )


class ReturnStatementTempLifetimeTests( CompilerTestCase ):
	''' regression tests for a real leak in lowering.py's _stmt_Return: a
	function whose entire body is a single `return SomeConstructor(
	helper_expr())`-shaped statement (no earlier statement for the ordinary
	per-statement pending-temp flush to land harmlessly before) used to emit
	the intermediate temp's own cleanup (the argument's Decref/DeleteTemp,
	previously appended by _lower_stmt's own post-method-call loop) AFTER
	the ir.Return/ir.Jump that statement's OWN handler already emitted -
	dead, unreachable C code, permanently leaking one retained reference on
	the argument every call. Fixed by having _stmt_Return flush its own
	still-pending temps itself (lowering.py's _flush_pending_temps), before
	its own terminator, in both branches (the plain inline ir.Return, and
	the shared-epilogue-label ir.Jump - cfg.py's untrack_temp() keeps
	either branch from also decref'ing the value being handed to the
	caller). '''

	def setUp( self ) -> None:
		self.discovery = Discovery( import_builtins = True )
		self.compiler = Compiler( self.discovery )

	def _extern_ldflags( self ) -> str:
		flags: list[str] = []
		for lib in sorted( self.compiler.extern_libs ):
			if lib == 'c':
				continue
			flags.append( linker_c.resolve_lib_ldflag( _CC, lib, self.compiler.extern_libs[lib] ) )
		return ' '.join( flags )

	def _assert_compiles_and_runs( self, c_source: str, expected_exit: int = 0 ) -> None:
		with tempfile.TemporaryDirectory() as tmp:
			src_path = Path( tmp ) / 'generated.c'
			obj_path = Path( tmp ) / 'generated.o'
			exe_path = Path( tmp ) / 'test_exe'
			src_path.write_text( c_source, encoding = 'utf-8' )
			cc_result = _CC.compile( src_path, obj_path )
			self.assertEqual( cc_result.returncode, 0,
				f'{_CC.name} compile failed:\nstdout: {cc_result.stdout}\nstderr: {cc_result.stderr}{test_support.c_source_on_failure( c_source )}' )
			ldflags = self._extern_ldflags()
			link_result = _CC.link( exe_path, [ obj_path ], ldflags = ldflags )
			self.assertEqual( link_result.returncode, 0,
				f'{_CC.name} link failed:\nstdout: {link_result.stdout}\nstderr: {link_result.stderr}' )
			run_result = subprocess.run( [ str( exe_path ) ], capture_output = True )
			self.assertEqual( run_result.returncode, expected_exit,
				f'exited {run_result.returncode}, expected {expected_exit} (stderr: {run_result.stderr})' )

	@unittest.skipUnless( _CC is not None, 'no C compiler (clang/gcc/msvc) found - skipping' )
	def test_single_statement_return_body_does_not_leak_argument_temp( self ) -> None:
		# make()'s ENTIRE body is one `return Result.Ok('hello'.upper())` -
		# no earlier statement exists for the intermediate str temp's own
		# cleanup to land after harmlessly, so this is the minimal shape
		# that exposed the bug: before the fix, the generated C had
		# `release_object(&$t0->header)` positioned AFTER `return $t1;`
		# inside make() (confirmed via direct source inspection), an
		# unreachable statement that left every string make() ever returned
		# permanently over-retained by one. Checked here end-to-end by
		# asserting the EXACT refcount of the returned string once it
		# reaches main() (through an ordinary match bind, itself already
		# covered/fixed separately - see
		# MatchArmSameNameNarrowingTests.test_rc_lifetime_repeated_match_exact_refcount):
		# 2 (make()'s own Result payload's ownership, plus x's own
		# separately-tracked extraction-bind) - 3 before this fix, from the
		# permanently-leaked extra retain make() itself introduced.
		self._run( '''
class MyError:
	pass

def make() -> Result[str,MyError]:
	return Result.Ok( 'hello'.upper() )

def main() -> i32:
	with compiler.wrap_arithmetic:
		r: Result[str,MyError] = make()
		rc: usize = 0
		match r:
			case Result.Ok( x ):
				rc = compiler.refcount( x )
			case Result.Err( e ):
				rc = 999
		return i32( rc )
''' )
		self.assertEqual( self.discovery.errors.errors, [] )
		src = emitter_c.emit_c( self.compiler )
		# the generated C itself should never have a release/decref call
		# positioned after make()'s own return - a direct, source-level
		# check that the dead-code-after-return shape is really gone, not
		# just that its net effect (the refcount below) happens to work out
		make_start = src.index( '__main__$make( void ) {' )
		make_body = src[ make_start : src.index( '\n}', make_start ) ]
		return_pos = make_body.index( 'return $t1;' )
		self.assertNotIn( 'release_object', make_body[ return_pos: ] )
		self._assert_compiles_and_runs( src, expected_exit = 2 )

	@unittest.skipUnless( _CC is not None, 'no C compiler (clang/gcc/msvc) found - skipping' )
	def test_single_statement_return_body_repeated_calls_no_leak( self ) -> None:
		# the same single-statement-body shape as above, but stressed over
		# 1000 repeated calls+matches on FRESH heap allocations each time -
		# a real RC-lifetime check, not just "doesn't crash once" (mirrors
		# MatchArmSameNameNarrowingTests.test_rc_lifetime_repeated_calls_no_leak's
		# own reasoning for why a bare content check every iteration matters)
		self._run( '''
class MyError:
	pass

def make() -> Result[str,MyError]:
	return Result.Ok( 'hello'.upper() )

def main() -> i32:
	with compiler.wrap_arithmetic:
		i: i32 = 0
		while i < 1000:
			r: Result[str,MyError] = make()
			match r:
				case Result.Ok( x ):
					if compiler.refcount( x ) != 2:
						return 1
				case Result.Err( e ):
					return 2
			i += 1
		return 0
''' )
		self.assertEqual( self.discovery.errors.errors, [] )
		self._assert_compiles_and_runs( emitter_c.emit_c( self.compiler ))


class IfExpTempLifetimeTests( test_support.RealCompileMixin, CompilerTestCase ):
	''' regression tests for a real UAF/double-free in lowering.py's
	_expr_IfExp: a ternary `A if cond else B` whose branches produce a
	fresh RC value (e.g. `str('-') if cond else str('+')`) merges both
	branches into one dest temp via a plain ir.Assign, but never untracked
	the branch's own temp - so _flush_pending_temps' later decref of the
	branch temp ran AGAINST THE SAME OBJECT dest (and whatever dest is
	later assigned into) still holds, freeing it out from under the merged
	result. Confirmed as a real, reproducible bug (found while building
	float64's shortest-round-trip repr - PLAN_STR_FORMAT.md item 4 -
	whose scientific-notation exponent-sign construction is exactly this
	shape): every scientific-notation float repr crashed or printed
	garbage before this fix. Worse, the UNTAKEN branch's own temp
	(declared but never assigned, since only one branch runs at runtime)
	was ALSO unconditionally decref'd at flush time - freeing
	uninitialized memory. Fixed by untrack_temp()-ing a fresh branch value
	before the merge Assign (mirroring _stmt_Return's own identical
	pattern), Incref-ing an ALIASING branch value instead (mirroring
	cfg.assign()'s own is_alias split - an existing binding read via the
	ternary becomes an independent, longer-lived reference), and
	registering the merge temp itself as the fresh owner afterward. '''

	def setUp( self ) -> None:
		self.discovery = Discovery( import_builtins = True )
		self.compiler = Compiler( self.discovery )

	@unittest.skipUnless( test_support.HAS_CC, 'no C compiler (clang/gcc/msvc) found - skipping' )
	def test_fresh_branch_values_no_double_free( self ) -> None:
		# both branches are fresh str(...) constructions (never assigned to
		# a name first) - the exact shape that crashed/corrupted before the
		# fix. Checked over 1000 iterations against FRESH heap allocations
		# each time, matching ReturnStatementTempLifetimeTests' own
		# reasoning for why a bare single-shot check isn't enough to catch
		# a leak (as opposed to the double-free, which a single shot alone
		# already reliably reproduced).
		self._run( '''
def main() -> i32:
	with compiler.wrap_arithmetic:
		i: i32 = 0
		cond: bool = True
		while i < 1000:
			x: str = str( '-' ) if cond else str( '+' )
			expected: str = str( '-' ) if cond else str( '+' )
			if x != expected:
				return 1
			if compiler.refcount( x ) != 1:
				return 2
			cond = not cond
			i += 1
		return 0
''' )
		self.assertEqual( self.discovery.errors.errors, [] )
		self._assert_compiles_and_runs( emitter_c.emit_c( self.compiler ))

	@unittest.skipUnless( test_support.HAS_CC, 'no C compiler (clang/gcc/msvc) found - skipping' )
	def test_aliasing_branch_value_gets_its_own_incref( self ) -> None:
		# both branches read EXISTING bindings (a, b) rather than
		# constructing fresh values - the merged result must be an
		# independently-owned reference (refcount bumped), not a bare
		# pointer copy: mutating/dropping a or b afterward must not affect
		# the merged result, and vice versa
		self._run( '''
def main() -> i32:
	with compiler.wrap_arithmetic:
		a: str = str( 'A' )
		b: str = str( 'B' )
		cond: bool = True
		z: str = a if cond else b
		if z != str( 'A' ):
			return 1
		if compiler.refcount( a ) != 2:
			return 2
		if compiler.refcount( z ) != 2:
			return 3
		if a != str( 'A' ) or b != str( 'B' ):
			return 4
		return 0
''' )
		self.assertEqual( self.discovery.errors.errors, [] )
		self._assert_compiles_and_runs( emitter_c.emit_c( self.compiler ))

	@unittest.skipUnless( test_support.HAS_CC, 'no C compiler (clang/gcc/msvc) found - skipping' )
	def test_mixed_fresh_and_aliasing_branches( self ) -> None:
		# one branch fresh (str.upper()'s own new allocation), the other
		# aliasing (a plain Name read) - each branch needs its OWN correct
		# treatment independently of what the other branch does
		self._run( '''
def main() -> i32:
	with compiler.wrap_arithmetic:
		existing: str = str( 'lower' )
		cond: bool = False
		result: str = existing.upper() if cond else existing
		if result != str( 'lower' ):
			return 1
		if compiler.refcount( existing ) != 2:
			return 2
		cond2: bool = True
		result2: str = existing.upper() if cond2 else existing
		if result2 != str( 'LOWER' ):
			return 3
		if compiler.refcount( result2 ) != 1:
			return 4
		return 0
''' )
		self.assertEqual( self.discovery.errors.errors, [] )
		self._assert_compiles_and_runs( emitter_c.emit_c( self.compiler ))


class CallableTests( test_support.RealCompileMixin, CompilerTestCase ):
	''' Callable[[Args],Ret]/Ptr[Callable[...]] end-to-end - see
	PLAN_CALLABLE.md: a bare function reference used as a value (never
	compiled anywhere before this), stored/passed as a real C function
	pointer, and called indirectly through it. Mirrors ListGenericTests'
	own import_builtins=True + real compile-and-run convention. '''

	def setUp( self ) -> None:
		self.discovery = Discovery( import_builtins = True )
		self.compiler = Compiler( self.discovery )

	@unittest.skipUnless( _CC is not None, 'no C compiler (clang/gcc/msvc) found - skipping' )
	def test_function_reference_stored_and_called_indirectly( self ) -> None:
		# add_one's address is taken (a real C function-pointer cast, not
		# a call), stored in a Ptr[Callable[[i32],i32]] local, passed to
		# another function, and called THROUGH it - no direct call to
		# add_one appears anywhere in this source
		self._run( '''
def add_one( x: i32 ) -> i32:
	with compiler.wrap_arithmetic:
		return x + 1

def call_it( f: Ptr[Callable[[i32],i32]], v: i32 ) -> i32:
	return f( v )

def main() -> i32:
	f: Ptr[Callable[[i32],i32]] = add_one
	result: i32 = call_it( f, 5 )
	with compiler.wrap_arithmetic:
		diff: i32 = result - 6
	return diff
''' )
		self.assertEqual( self.discovery.errors.errors, [] )
		src = emitter_c.emit_c( self.compiler )
		self.assertIn( '(int32_t (*)( int32_t ))__main__$add_one', src ) # a real function-pointer cast, not a call
		self.assertIn( '(f)( v )', src ) # a call THROUGH the pointer, no explicit deref needed
		self._assert_compiles_and_runs( src )

	@unittest.skipUnless( test_support.HAS_CC, 'no C compiler (clang/gcc/msvc) found - skipping' )
	def test_programs_compile_and_run( self ) -> None:
		''' every real compile-and-run program in this class, merged into a
		single executable (one build for the whole class); a nonzero exit is
		decoded back to the failing sub-program and its own return code. '''
		self.assert_programs_run([
			# the exact shape dict[K,V]'s own comparator/hasher helpers will
			# use: a @staticmethod referenced bare from a sibling method of the
			# same class, stored in a local, called indirectly from there.
			# NOTE: deliberately does NOT return the Ptr[Callable[...]] value
			# from a function - that's a real, separate gap (a function
			# RETURNING a function pointer is C's gnarliest declarator shape,
			# `RetType (*name(Params))(InnerParams)` - _declarator only covers
			# parameter/local declarations, per PLAN_CALLABLE.md's own scope).
			# Not needed here: dict[K,V] only ever passes a callback as a
			# parameter, never returns one.
			( 'staticmethod_reference_called_indirectly', '''
class Ops:
	@staticmethod
	def double( x: i32 ) -> i32:
		with compiler.wrap_arithmetic:
			return x * 2

	def run( self, v: i32 ) -> i32:
		f: Ptr[Callable[[i32],i32]] = double
		return f( v )

def main() -> i32:
	o: Ops = Ops()
	result: i32 = o.run( 21 )
	with compiler.wrap_arithmetic:
		diff: i32 = result - 42
	return diff
''' ),
			# regression test for a real compiler bug found while exploring
			# PLAN_LAMBDA.md's own zoneinfo.py blocker: _unify_type_param/
			# substitute_type_params used to stop recursing at a CallableType
			# (not a Specialization, so the Ptr[T]-vs-Ptr[i32] recursion never
			# looked inside a Ptr[Callable[[T],K]] parameter's own arg_types/
			# return_type) - even a plain function reference argument (no
			# lambda at all) failed to infer K this way before the fix
			( 'generic_call_infers_type_params_through_callable_parameter', '''
def identity_i32( v: i32 ) -> i32:
	return v

def apply[T,K]( x: T, key: Ptr[Callable[[T],K]] ) -> K:
	return key( x )

def main() -> i32:
	result: i32 = apply( 5, key = identity_i32 )
	with compiler.wrap_arithmetic:
		diff: i32 = result - 5
	return diff
''' ),
			# real compile-and-run version of PLAN_LAMBDA.md's "eager lambda
			# lowering" piece: unlike the function-reference case just above, K
			# is only knowable from the LAMBDA's own inferred return type -
			# _expr_Lambda has to lower the lambda's body right now, at this
			# call site, instead of only ever deferring it onto the work queue
			# (see FunctionLowering/Compiler._lower's own _compile_now
			# backreference). Checking the actual returned value (not just that
			# it compiles) matters here specifically: if the eager path got the
			# wrong return type, or clobbered the enclosing function's own
			# in-progress lowering state, this is the kind of bug that would
			# still compile and link, just produce a silently wrong answer
			( 'lambda_eagerly_lowered_infers_generic_return_type', '''
def apply[T,K]( x: T, key: Ptr[Callable[[T],K]] ) -> K:
	return key( x )

def main() -> i32:
	result: i32 = apply( 5, key = lambda v: v )
	with compiler.wrap_arithmetic:
		diff: i32 = result - 5
	return diff
''' ),
		] )

	@unittest.skipUnless( _CC is not None, 'no C compiler (clang/gcc/msvc) found - skipping' )
	def test_self_dot_staticmethod_call_passes_no_receiver( self ) -> None:
		# regression test for a real compiler bug found while building
		# dict[K,V]: self.static_method(...) (as opposed to
		# ClassName.static_method(...)) used to always attach self as a
		# receiver argument at the call site, even though a @staticmethod
		# takes none - ir.Call.receiver's own docstring already promised
		# "None for a free function, staticmethod, or classmethod call",
		# but _resolve_callee's Attribute fallback (used for ANY dotted
		# callee, since it can't know staticness before resolving the
		# attribute) always computed one regardless. Confirmed via real
		# emitted C: the call site passed 2 arguments to a 1-parameter
		# prototype ("too many arguments to function call"). ClassName.
		# method(...) never hit this (a different, namespace-lookup
		# resolution path that never computes a receiver at all) - only
		# self.-qualified calls did, which is exactly the shape dict[K,V]
		# needs throughout (its own RC-branching helpers are all
		# @staticmethods, called via self. from __getitem__/__setitem__/
		# __del__)
		self._run( '''
class Box:
	v: i32

	@staticmethod
	def double( x: i32 ) -> i32:
		with compiler.wrap_arithmetic:
			return x * 2

	def use_it( self, x: i32 ) -> i32:
		return self.double( x )

def main() -> i32:
	b: Box = Box( v = 0 )
	result: i32 = b.use_it( 21 )
	with compiler.wrap_arithmetic:
		diff: i32 = result - 42
	return diff
''' )
		self.assertEqual( self.discovery.errors.errors, [] )
		src = emitter_c.emit_c( self.compiler )
		self.assertIn( 'int32_t __main__$Box$double( int32_t x )', src ) # no self parameter
		self.assertIn( '__main__$Box$double( x )', src ) # no self argument at the call site either
		self._assert_compiles_and_runs( src )

	@unittest.skipUnless( _CC is not None, 'no C compiler (clang/gcc/msvc) found - skipping' )
	def test_setitem_returning_plain_none_does_not_assign_void( self ) -> None:
		# regression test for a real compiler bug found while building
		# dict[K,V]: obj[i] = v always created a destination Temp for
		# __setitem__'s call, even when __setitem__ returns plain None
		# (C void) - the ordinary/conventional case, matching Python's own
		# __setitem__ protocol. Confirmed via real emitted C: `t1 = some_
		# void_returning_call();` doesn't compile ("assigning to
		# 'MetalpyNone' from incompatible type 'void'"). Only __setitem__
		# implementations returning a real Result[None,E] (the OTHER,
		# fallible case - see test_subscript_assign_with_setitem_resolves_
		# and_consumes_result in lowering_test.py) ever exercised the
		# call-with-a-real-dest path before, so this never surfaced
		self._run( '''
class Box:
	y: i32

	def __setitem__( self, i: usize, v: i32 ) -> None:
		self.y = v

def main() -> i32:
	b: Box = Box( y = 0 )
	b[0] = 42
	return b.y
''' )
		self.assertEqual( self.discovery.errors.errors, [] )
		self._assert_compiles_and_runs( emitter_c.emit_c( self.compiler ), expected_exit = 42 )


class NestedFunctionTests( test_support.RealCompileMixin, CompilerTestCase ):
	''' non-capturing nested function defs - see PLAN_LAMBDA.md. Never
	compiled anywhere before this (previously hit lowering.py's generic
	"unsupported statement" fallback). Mirrors CallableTests' own
	import_builtins=True + real compile-and-run convention. '''

	def setUp( self ) -> None:
		self.discovery = Discovery( import_builtins = True )
		self.compiler = Compiler( self.discovery )

	@unittest.skipUnless( test_support.HAS_CC, 'no C compiler (clang/gcc/msvc) found - skipping' )
	def test_programs_compile_and_run( self ) -> None:
		''' every real compile-and-run program in this class, merged into a
		single executable (one build for the whole class); a nonzero exit is
		decoded back to the failing sub-program and its own return code. '''
		self.assert_programs_run([
			( 'nested_def_called_directly', '''
def outer() -> i32:
	def inner( x: i32 ) -> i32:
		with compiler.wrap_arithmetic:
			return x + 1
	return inner( 5 )

def main() -> i32:
	with compiler.wrap_arithmetic:
		diff: i32 = outer() - 6
	return diff
''' ),
			( 'nested_def_bare_reference_called_indirectly', '''
def call_it( f: Ptr[Callable[[i32],i32]], v: i32 ) -> i32:
	return f( v )

def outer() -> i32:
	def inner( x: i32 ) -> i32:
		with compiler.wrap_arithmetic:
			return x + 1
	f: Ptr[Callable[[i32],i32]] = inner
	return call_it( f, 5 )

def main() -> i32:
	with compiler.wrap_arithmetic:
		diff: i32 = outer() - 6
	return diff
''' ),
			( 'lambda_param_types_inferred_and_called_indirectly', '''
def call_it( f: Ptr[Callable[[i32],i32]], v: i32 ) -> i32:
	return f( v )

def main() -> i32:
	result: i32 = call_it( lambda x: x, 5 )
	with compiler.wrap_arithmetic:
		diff: i32 = result - 5
	return diff
''' ),
		] )

@unittest.skipUnless( os.name == 'nt', 'COM interop is Windows-only (lib/windows/com) - skipping off Windows' )
class ComTests( test_support.RealCompileMixin, CompilerTestCase ):
	''' lib/windows/com.py's HRESULT/IUnknown pattern -
	PLAN_SUBCLASSING_VTABLES_COM.md's Phase 3 worked example: a
	metalpy-implemented COM interface (IUnknown -> IFoo, adding a method -
	the real COM pattern the per-level-Vtbl-types revision exists for),
	hand-written QueryInterface/AddRef/Release, constructed and dispatched
	through end-to-end. Needs import_builtins=True (str.split, list[str],
	GUID's own dependencies). '''

	def setUp( self ) -> None:
		self.discovery = Discovery( import_builtins = True )
		self.compiler = Compiler( self.discovery )

	@unittest.skipUnless( test_support.HAS_CC, 'no C compiler (clang/gcc/msvc) found - skipping' )
	def test_programs_compile_and_run( self ) -> None:
		''' every real compile-and-run program in this class, merged into a
		single executable (one build for the whole class); a nonzero exit is
		decoded back to the failing sub-program and its own return code. '''
		self.assert_programs_run([
			( 'succeeded_failed_helpers', '''
from windows.com import S_OK, E_FAIL, SUCCEEDED, FAILED

def main() -> i32:
	if not SUCCEEDED( S_OK ):
		return 1
	if SUCCEEDED( E_FAIL ):
		return 2
	if FAILED( S_OK ):
		return 3
	if not FAILED( E_FAIL ):
		return 4
	return 0
''' ),
			# IFoo(IUnknown) adds get_value - the exact COM pattern that
			# motivated the per-level-Vtbl-types revision (IUnknown's own 3
			# slots first, then IFoo's own new one, all in IFoo's own
			# FooImpl-shared Vtbl type). Hand-written QueryInterface/AddRef/
			# Release (no compiler synthesis, per the plan's own "hand-rolling
			# first" decision) - QueryInterface writes a real pointer through
			# its Ptr[Ptr[None]] out-param and calls AddRef itself, matching
			# real COM QueryInterface semantics.
			( 'iunknown_subclass_construct_dispatch_queryinterface_addref_release', '''
from windows.com import IUnknown, HRESULT, S_OK, E_NOINTERFACE, SUCCEEDED, FAILED
import guid

@interface
class IFoo( IUnknown ):
	@virtual
	def get_value( self ) -> i32: ...

@interface
class FooImpl( IFoo ):
	x: i32
	ref_count: u32

	@virtual
	def QueryInterface( self, riid: ConstPtr[guid.GUID], ppvObject: Ptr[Ptr[None]] ) -> HRESULT:
		ppvObject[0] = compiler.cast( Ptr[None], self )
		self.AddRef()
		return S_OK

	@virtual
	def AddRef( self ) -> u32:
		with compiler.wrap_arithmetic:
			self.ref_count = self.ref_count + 1
		return self.ref_count

	@virtual
	def Release( self ) -> u32:
		with compiler.wrap_arithmetic:
			self.ref_count = self.ref_count - 1
		return self.ref_count

	@virtual
	def get_value( self ) -> i32:
		return self.x

def main() -> i32:
	f: Ptr[FooImpl] = FooImpl( x = 99, ref_count = 1 )

	direct: i32 = f.get_value()
	with compiler.wrap_arithmetic:
		diff: i32 = direct - 99
	if diff != 0:
		return 1

	out: Ptr[None] = None
	requested_iid: guid.GUID = guid.GUID.from_str( '00000000-0000-0000-0000-000000000000' )
	hr: HRESULT = f.QueryInterface( compiler.addrof( requested_iid ), compiler.addrof( out ))
	if not SUCCEEDED( hr ):
		return 2
	if f.ref_count != 2:
		return 3
	if out == None:
		return 4

	f.Release()
	if f.ref_count != 1:
		return 5

	if not FAILED( E_NOINTERFACE ):
		return 6

	return 0
''' ),
			# consumes a REAL foreign Windows COM object - CoCreateInstance's
			# own CLSID_ShellLink (shell32's IShellLinkW implementation,
			# always present on any real Windows install), requesting its
			# IPersist view. IPersist is about as minimal as a real, standard
			# COM interface gets: IUnknown's 3 slots plus exactly one method,
			# GetClassID(CLSID*) -> HRESULT (verified directly against
			# Microsoft Learn's own IPersist::GetClassID docs, not just
			# memory, given a wrong vtable slot COUNT/ORDER here would be
			# calling into whatever real function actually sits at that
			# offset - not a graceful failure). Querying an object for its
			# OWN class id and checking it matches the well-known,
			# published CLSID_ShellLink is the "queries for a piece of
			# data" the whole point of this test is to prove: the metalpy-
			# declared vtable shape genuinely lines up with a real, foreign,
			# already-compiled COM object's actual in-memory layout - not
			# just with other metalpy code.
			( 'real_windows_com_service_shelllink_getclassid', '''
from windows.com import HRESULT, SUCCEEDED, CoInitializeEx, CoUninitialize, CoCreateInstance, COINIT_APARTMENTTHREADED, CLSCTX_INPROC_SERVER
from windows.com.ipersist import IPersist
import guid

def main() -> i32:
	init_hr: HRESULT = CoInitializeEx( None, COINIT_APARTMENTTHREADED )
	if not SUCCEEDED( init_hr ):
		return 1
	defer( CoUninitialize() )

	# well-known, published GUIDs (verified against Microsoft Learn, not
	# just memory) - CLSID_ShellLink and IID_IPersist
	clsid_shelllink: guid.GUID = guid.GUID.from_str( '00021401-0000-0000-c000-000000000046' )
	iid_ipersist: guid.GUID = guid.GUID.from_str( '0000010c-0000-0000-c000-000000000046' )

	out: Ptr[None] = None
	create_hr: HRESULT = CoCreateInstance(
		compiler.addrof( clsid_shelllink ),
		None,
		CLSCTX_INPROC_SERVER,
		compiler.addrof( iid_ipersist ),
		compiler.addrof( out ),
	)
	if not SUCCEEDED( create_hr ):
		return 2

	persist: Ptr[IPersist] = compiler.cast( Ptr[IPersist], out )

	returned_clsid: guid.GUID = guid.GUID.from_str( '00000000-0000-0000-0000-000000000000' )
	getclassid_hr: HRESULT = persist.GetClassID( compiler.addrof( returned_clsid ))
	if not SUCCEEDED( getclassid_hr ):
		return 3

	if returned_clsid != clsid_shelllink:
		return 4

	return 0
''' ),
		] )


class CEnumConstructionArgumentShapeTests( test_support.RealCompileMixin, CompilerTestCase ):
	''' Regression test for a real bug: `EnumName(value)` (CEnum construction,
	e.g. `OSError(rc)`) was rejected with a confusing "expected EnumName, got
	<value's own type>" whenever `value` was a bare `ast.Name` (a local
	variable or parameter reference), while the IDENTICAL underlying value
	via a Call or BinOp argument (`OSError(get_rc())`, `OSError(rc + 0)`)
	compiled fine - an inconsistency across argument AST SHAPE, not a real
	difference in what was being constructed.

	Root cause: `_try_lower_construct_call`'s CEnum branch used to pass
	target_cls (the enum type ITSELF) as `_lower_expr`'s `expected_type` for
	the argument, for every argument shape. A bare `ast.Name` operand
	(`_expr_Name`) ignores `expected_type` and keeps its own declared type,
	so the later `_check_assignable` correctly (if confusingly-worded)
	rejected a genuine underlying-type mismatch - `OSError`'s value_type is
	u32, and a plain `i32` local doesn't automatically become one. But an
	ordinary Call/BinOp argument's own result-typing tail (`_lower_call`'s
	final dest allocation, `expected_type or target_return_type`) SILENTLY
	relabeled the destination temp's type to target_cls directly, with no
	check that the callee's real return type was even compatible - so those
	shapes "worked" by accident, not because they were validated.

	Fixed: every non-literal argument shape now lowers against value_type
	(the underlying scalar, the argument's real natural type space) and is
	explicitly relabeled onto the enum type via CastWrap - the same zero-
	cost, well-defined C reinterpret cast an explicit T(x) scalar cast uses,
	consistent across every argument shape, and permissive of genuine
	cross-signedness reinterpretation (OSError's own construction is
	documented as "a plain cast to the enum's underlying type", the same
	promise an explicit u32(-11)-style WinAPI cast makes). A literal integer
	argument keeps its own pre-existing fast path (a plain ir.Const, no
	runtime cast) unchanged. '''

	def setUp( self ) -> None:
		self.discovery = Discovery( import_builtins = True )
		self.compiler = Compiler( self.discovery )

	@unittest.skipUnless( test_support.HAS_CC, 'no C compiler (clang/gcc/msvc) found - skipping' )
	def test_programs_compile_and_run( self ) -> None:
		self.assert_programs_run([
			# the exact repro shape: a bare local variable argument to a
			# CEnum constructor, feeding straight into Result.Err(...) -
			# used to be rejected outright
			( 'bare_name_argument_to_enum_constructor_compiles_and_runs', '''
def f() -> Result[None, OSError]:
	rc: i32 = -5
	return Result.Err( OSError( rc ))

def main() -> i32:
	r = f()
	if r.is_err():
		return 0
	return 1
''' ),
			# consistency check: bare Name / Call / BinOp / literal argument
			# shapes must all produce the SAME bit-exact underlying value for
			# the identical logical error code, and a plain assignment back
			# to the enum's own value_type (already-established, unrelated
			# CEnum<->value_type duality) must read that same value back out.
			# OSError's own value_type is platform-dependent (lib/builtins/
			# __errors.py: u32 on Windows, i32 on Linux) - a fixed `v_name:
			# u32` readback only type-checks on the Windows build, so this is
			# gated the same way OSError's OWN declaration is (two
			# @compiler.target(os=...)-filtered variants of one function,
			# only one of which is ever actually compiled for a given
			# target), rather than hardcoding one platform's width and
			# silently only ever exercising this consistency check there.
			( 'enum_constructor_argument_shapes_agree_bit_exactly', '''
import compiler

def get_rc() -> i32:
	return -5

@compiler.target( os = 'windows' )
def check_argument_shapes() -> i32:
	rc: i32 = -5
	e_name: OSError = OSError( rc )
	e_call: OSError = OSError( get_rc() )
	with compiler.wrap_arithmetic:
		e_binop: OSError = OSError( rc + 0 )
	e_member: OSError = OSError.FileNotFoundError

	v_name: u32 = e_name
	v_call: u32 = e_call
	v_binop: u32 = e_binop
	expected: u32 = u32( -5 )

	if v_name != expected:
		return 1
	if v_call != expected:
		return 2
	if v_binop != expected:
		return 3
	if e_member != OSError.FileNotFoundError:
		return 4
	return 0

@compiler.target( os = not 'windows' )
def check_argument_shapes() -> i32:
	rc: i32 = -5
	e_name: OSError = OSError( rc )
	e_call: OSError = OSError( get_rc() )
	with compiler.wrap_arithmetic:
		e_binop: OSError = OSError( rc + 0 )
	e_member: OSError = OSError.FileNotFoundError

	v_name: i32 = e_name
	v_call: i32 = e_call
	v_binop: i32 = e_binop
	expected: i32 = -5

	if v_name != expected:
		return 1
	if v_call != expected:
		return 2
	if v_binop != expected:
		return 3
	if e_member != OSError.FileNotFoundError:
		return 4
	return 0

def main() -> i32:
	return check_argument_shapes()
''' ),
			# a routed-through-a-parameter shape (not just a local) - the
			# task's own report specifically called out that this ALSO
			# failed identically (not a locals-vs-parameters distinction).
			# Same platform-dependent value_type split as the case above -
			# OSError's value_type is u32 on Windows, i32 on Linux.
			( 'bare_name_parameter_argument_to_enum_constructor_compiles_and_runs', '''
import compiler

def mk( code: i32 ) -> OSError:
	return OSError( code )

@compiler.target( os = 'windows' )
def check_parameter_shape() -> i32:
	e: OSError = mk( -5 )
	v: u32 = e
	if v != u32( -5 ):
		return 1
	return 0

@compiler.target( os = not 'windows' )
def check_parameter_shape() -> i32:
	e: OSError = mk( -5 )
	v: i32 = e
	if v != -5:
		return 1
	return 0

def main() -> i32:
	return check_parameter_shape()
''' ),
			# the literal fast path (unaffected by this fix) - still folds to
			# a plain constant and still range-checks correctly
			( 'literal_argument_to_enum_constructor_still_works', '''
def main() -> i32:
	e: OSError = OSError( 2 )
	if e != OSError.FileNotFoundError:
		return 1
	return 0
''' ),
		] )

	def test_out_of_range_literal_still_rejected( self ) -> None:
		# negative check: the pre-existing literal magnitude/range validation
		# (unrelated to and unchanged by this fix) must still reject a
		# genuinely out-of-range literal argument, not just silently accept
		# everything now that non-literal shapes are more permissive
		self._run( '\n'.join([
			'def main() -> None:',
			'	e: OSError = OSError( 99999999999 )',
			'	return',
		]))
		self.assertTrue( self.discovery.errors.errors )
		self.assertIn( 'out of range', self.discovery.errors.errors[0] )


class FixedSizeArrayFieldTests( test_support.RealCompileMixin, CompilerTestCase ):
	''' Regression test for SYNTAX.md's documented-but-unimplemented
	"Fixed-Size Inline Array (inside @struct): u16[32], u8[8]" - a bare
	`ElemType[N]` field annotation used to fail outright at annotation-
	resolution time ("intrinsics.u8 is not generic, cannot subscript it" -
	visit_Subscript's generic-subscript path unconditionally rejected any
	non-generic base). lib/guid.py's own GUID class already documented
	hitting this exact gap for its `Data4[8]` field and worked around it by
	unrolling into 8 separate `data4_0..data4_7: u8` fields instead.

	Fixed via a new mpy_types.FixedArrayType, recognized in discovery.py's
	visit_Subscript (a non-generic base subscripted by a bare positive int
	constant, as opposed to a real generic type argument - which always
	uses a TYPE expression as its slice, never a bare int, so this can
	never misfire against a genuine generic instantiation) and given a real
	C array declarator in struct/union body emission (`TYPE NAME[N];`,
	special-cased in _struct_or_union_body the same way _declarator already
	special-cases a function-pointer field's own discontinuous C syntax).

	Deliberately scoped, not a general-purpose value type: a bare C array
	is not assignable via `=` at all (only a whole containing struct/union
	is), so this fix only supports (1) declaring the field, inside a plain
	@cstruct/@cunion only - rejected everywhere else (parameters, return
	types, module/class-level variables, RCClass/@interface fields) - and
	(2) a `= 0` field default / explicit `ClassName(field=0)` construction
	argument, meaning "zero-fill the whole array" (the one shape a C
	designated initializer can express, `.field = {0}`). Reading a
	FixedArrayType field back out as a whole value, or assigning one after
	construction, is explicitly rejected with a clean error rather than
	reaching emission and producing invalid C - element-level indexed
	access is a separate, real, currently-unimplemented follow-up (the same
	kind of gap this repo's own bytearray has today), not attempted here. '''

	def setUp( self ) -> None:
		self.discovery = Discovery( import_builtins = True )
		self.compiler = Compiler( self.discovery )

	@unittest.skipUnless( test_support.HAS_CC, 'no C compiler (clang/gcc/msvc) found - skipping' )
	def test_programs_compile_and_run( self ) -> None:
		self.assert_programs_run([
			# the exact SYNTAX.md-documented shape - a real @cstruct with a
			# fixed-size inline array field, zero-filled by default,
			# constructed bare, real sizeof() confirms correct C layout (no
			# silent size-0/opaque-type fallback)
			( 'fixed_array_field_declares_and_zero_fill_constructs', '''
import compiler

@cstruct
class Foo:
	a: u16 = 0
	b: u8[8] = 0

def main() -> i32:
	f = Foo()
	sz: usize = compiler.sizeof( Foo )
	if sz != usize( 10 ):
		return 1
	return 0
''' ),
			# explicit ClassName(field=0) construction argument (not just the
			# class-body default) - same zero-fill path, different call site
			( 'fixed_array_field_explicit_zero_construction_argument', '''
@cstruct
class Foo:
	a: u16 = 0
	b: u8[8] = 0

def main() -> i32:
	f = Foo( a = 5, b = 0 )
	if f.a != 5:
		return 1
	return 0
''' ),
			# multiple array fields of different element types/counts in one
			# struct, interleaved with scalar fields - mirrors SYNTAX.md's own
			# DynamicTimeZoneInformation worked example almost verbatim
			( 'multiple_fixed_array_fields_interleaved_with_scalars', '''
@cstruct
class Multi:
	bias: i32 = 0
	name: u16[32] = 0
	date: u16[8] = 0
	flag: u8 = 0
	pad: u8[3] = 0

def main() -> i32:
	m = Multi( bias = 7 )
	if m.bias != 7:
		return 1
	return 0
''' ),
		] )

	def test_out_of_range_field_annotation_type_still_rejects_generic_subscript_errors( self ) -> None:
		# negative check: an actually-invalid subscript (a real, non-generic,
		# non-array-shaped misuse) must still be rejected the same way it
		# always was - this fix only ever WIDENS what's accepted (a non-
		# generic base + a bare positive int constant slice), never narrows
		# the existing "not generic, cannot subscript it" rejection for
		# every other shape
		self._run( '\n'.join([
			'def main() -> None:',
			'	x: bool[i32] = None', # bool is non-generic, i32 is a TYPE not an int constant - still invalid
			'	return',
		]))
		self.assertTrue( self.discovery.errors.errors )
		self.assertIn( 'not generic', self.discovery.errors.errors[0] )

	def test_fixed_array_field_rejected_as_parameter_type( self ) -> None:
		self._run( '\n'.join([
			'def f( x: u8[8] ) -> i32:',
			'	return 0',
			'',
			'def main() -> None:',
			'	f( 0 )',
			'	return',
		]))
		self.assertTrue( self.discovery.errors.errors )
		self.assertIn( 'only allowed as a plain @cstruct/@cunion field', self.discovery.errors.errors[0] )

	def test_fixed_array_field_rejected_as_module_global( self ) -> None:
		self._run( '\n'.join([
			'g: u8[8] = 0',
			'',
			'def main() -> None:',
			'	x = g',
			'	return',
		]))
		self.assertTrue( self.discovery.errors.errors )
		self.assertIn( 'only allowed as a plain @cstruct/@cunion field', self.discovery.errors.errors[0] )

	def test_reading_fixed_array_field_as_a_whole_value_is_rejected( self ) -> None:
		self._run( '\n'.join([
			'@cstruct',
			'class Foo:',
			'	b: u8[8] = 0',
			'',
			'def main() -> None:',
			'	f = Foo()',
			'	x = f.b',
			'	return',
		]))
		self.assertTrue( self.discovery.errors.errors )
		self.assertIn( 'cannot be read as a whole value', self.discovery.errors.errors[0] )


class AddrofFieldAccessRealCompileTests( test_support.RealCompileMixin, CompilerTestCase ):
	''' Regression test for a real, confirmed gap: `compiler.addrof(x)`
	rejected any argument that wasn't a bare local-variable Name outright
	("compiler.addrof(...) argument must be a bare local variable, not
	compiler.addrof(f.a)"), even a single level of plain field access on an
	already-stable local (`compiler.addrof(f.a)`) - a very common C idiom
	needed by any FFI code that fills one field of a stack struct via an
	out-parameter (e.g. inet_pton(af, str, &addr.sin_addr)), forcing
	callers to stage the value through an extra local first.

	Investigated and confirmed safe to widen for exactly this shape: a bare
	Name's own operand is always a genuine, stable-lifetime Variable (never
	a Temp), and requiring the field-access chain's ROOT to be one too
	preserves that same guarantee one level deeper (a field of an already-
	stable object is itself just as addressable - C's own well-defined
	`&x.field`/`&x->field`). Deliberately NOT widened further: a Call-
	rooted argument (`compiler.addrof(make().field)`) is a real, not just
	theoretical, dangling-pointer risk (the call's own result is a
	TEMPORARY with no guaranteed lifetime past the current statement under
	this compiler's RC discipline) and a multi-level chain
	(`compiler.addrof(a.b.c)`) is unneeded, unanalyzed extra scope - both
	still rejected with a clear message.

	Fixed via a new ir.AddrOfField instruction (distinct from
	AddrOf(GetAttr(...)) - GetAttr loads a COPY of the field's value into a
	fresh temp, whose address would be the copy's, not the real field's,
	defeating the entire point of the FFI out-parameter idiom this exists
	for) - emits one flat `&(obj)OP field` C expression, obj always the
	chain's root object. '''

	def setUp( self ) -> None:
		self.discovery = Discovery( import_builtins = True )
		self.compiler = Compiler( self.discovery )

	@unittest.skipUnless( test_support.HAS_CC, 'no C compiler (clang/gcc/msvc) found - skipping' )
	def test_programs_compile_and_run( self ) -> None:
		self.assert_programs_run([
			# write-through proof, not just "compiles": a callee writes
			# through the pointer into ONE field of a caller's stack
			# struct, and the OTHER field must stay untouched - confirms
			# the address genuinely targets the real field's own storage
			# inside the original struct, not some throwaway copy
			( 'addrof_field_writes_through_to_the_real_struct_field', '''
@cstruct
class Foo:
	a: u32 = 0
	b: u32 = 0

def write_via_ptr( p: Ptr[u32] ) -> None:
	p[0] = u32( 42 )
	return

def main() -> i32:
	f = Foo()
	write_via_ptr( compiler.addrof( f.b ))
	if f.b != u32( 42 ):
		return 1
	if f.a != u32( 0 ):
		return 2
	return 0
''' ),
		] )

	@unittest.skipUnless( test_support.HAS_CC, 'no C compiler (clang/gcc/msvc) found - skipping' )
	@unittest.skipUnless( os.name == 'nt', 'needs a real Winsock DLL to call (ws2_32.dll)' )
	def test_real_ffi_out_parameter_shape_compiles_and_runs( self ) -> None:
		# the exact motivating shape from the task report: a real Win32 FFI
		# call filling ONE field of a stack struct via compiler.addrof on
		# that field directly (inet_pton's own real, documented contract),
		# not staged through an extra local first
		self.assert_programs_run([
			( 'inet_pton_writes_into_a_nested_struct_field_via_addrof', '''
import compiler
import sys

@cstruct
class InAddr:
	s_addr: u32 = 0

@cstruct
class SockAddrIn:
	sin_family: i16 = 0
	sin_port: u16 = 0
	sin_addr: InAddr
	sin_zero0: u32 = 0
	sin_zero1: u32 = 0

@extern( 'ws2_32', 'WSAStartup' )
def WSAStartup( wVersionRequested: u16, lpWSAData: Ptr[None] ) -> i32: ...

@extern( 'ws2_32', 'inet_pton' )
def inet_pton( family: i32, pszAddrString: ConstPtr[u8], pAddrBuf: Ptr[None] ) -> i32: ...

def main() -> i32:
	wsadata: Ptr[u8] = sys.alloc[u8]( 512 )
	WSAStartup( 0x0202, compiler.cast( Ptr[None], wsadata ))
	sys.free( compiler.cast( Ptr[None], wsadata ))

	addr: SockAddrIn = SockAddrIn( sin_addr = InAddr() )
	rc: i32 = inet_pton( 2, '127.0.0.1'.get_cstr(), compiler.cast( Ptr[None], compiler.addrof( addr.sin_addr )))
	if rc != 1:
		return 1
	if addr.sin_addr.s_addr != u32( 0x0100007F ):
		return 2
	return 0
''' ),
		] )

	def test_addrof_rejects_multi_level_field_chain( self ) -> None:
		self._run( '\n'.join([
			'@cstruct',
			'class Inner:',
			'	x: u32 = 0',
			'',
			'@cstruct',
			'class Outer:',
			'	inner: Inner',
			'	y: u32 = 0',
			'',
			'def main() -> None:',
			'	o = Outer( inner = Inner() )',
			'	compiler.addrof( o.inner.x )',
			'	return',
		]))
		self.assertTrue( self.discovery.errors.errors )
		self.assertIn( 'only one level of field access is supported', self.discovery.errors.errors[0] )

	def test_addrof_rejects_call_rooted_field_access( self ) -> None:
		self._run( '\n'.join([
			'@cstruct',
			'class Foo:',
			'	a: u32 = 0',
			'',
			'def make() -> Foo:',
			'	return Foo()',
			'',
			'def main() -> None:',
			'	compiler.addrof( make().a )',
			'	return',
		]))
		self.assertTrue( self.discovery.errors.errors )
		self.assertIn( 'must be rooted at a bare local variable', self.discovery.errors.errors[0] )


class ByteArrayScalarIndexingRealCompileTests( test_support.RealCompileMixin, CompilerTestCase ):
	''' Regression test for a real gap: bytearray had no scalar
	__getitem__/__setitem__ (only slice syntax via get_ptr()/get_const_ptr()/
	_byte_slice) - `buf[0] = 65`/`x: u8 = buf[0]` failed at C-compile time
	with e.g. "assigning to 'struct builtins$bytearray' from incompatible
	type 'uint8_t'" (the generated C treated buf[0] as indexing the WHOLE
	struct rather than dispatching through __getitem__/__setitem__, since
	neither existed to dispatch through). bytes/bytearray fully support
	buf[i]/buf[i] = x in real Python. lib/socket.py's own real code worked
	around this throughout by always going through .get_ptr()[i]/
	.get_const_ptr()[i] (raw pointer indexing, which already worked) instead.

	Fixed by adding real __getitem__(self, index: usize) ->
	Result[u8,IndexError] / __setitem__(self, index: usize, value: u8) ->
	None methods to bytearray in lib/builtins/__init__.py, following the
	existing UnsafeList/dict __getitem__/__setitem__ patterns already in
	that file, and matching bytearray's own debug-mode BYTEARRAY_INVALID-
	after-release assertion style already used by its other methods (plus a
	debug-mode index-in-range assertion for __setitem__, which has no
	Result to report an out-of-range write through, matching its own bare
	`-> None` signature). '''

	def setUp( self ) -> None:
		self.discovery = Discovery( import_builtins = True )
		self.compiler = Compiler( self.discovery )

	@unittest.skipUnless( test_support.HAS_CC, 'no C compiler (clang/gcc/msvc) found - skipping' )
	def test_programs_compile_and_run( self ) -> None:
		self.assert_programs_run([
			# the exact repro shape from the task report - buf[0] = ...
			# (bare subscript assignment) and x: u8 = buf[0] (bare
			# subscript read) both used to fail at C-compile time
			( 'bytearray_scalar_getitem_setitem_bare_subscript_syntax', '''
def helper() -> Result[i32, IndexError]:
	buf: bytearray = bytearray( 4 )
	buf[0] = 65
	x: u8 = buf[0]
	if x != 65:
		return Result.Ok( 1 )
	return Result.Ok( 0 )

def main() -> i32:
	match helper():
		case Result.Ok( code ):
			return code
		case Result.Err( _ ):
			return 2
''' ),
			# explicit .__getitem__()/.__setitem__() calls (not just the
			# bare subscript sugar), and an out-of-range __getitem__
			# correctly reports Err rather than reading out of bounds
			( 'bytearray_explicit_getitem_setitem_calls_and_out_of_range_getitem', '''
def main() -> i32:
	buf: bytearray = bytearray( 4 )
	buf.__setitem__( 0, 65 )
	buf.__setitem__( 3, 99 )
	v0: u8 = buf.__getitem__( 0 ).unwrap( 'index 0 in range' )
	v3: u8 = buf.__getitem__( 3 ).unwrap( 'index 3 in range' )
	if v0 != 65:
		return 1
	if v3 != 99:
		return 2
	match buf.__getitem__( 4 ): # one past the end - out of range
		case Result.Ok( _ ):
			return 3
		case Result.Err( _ ):
			pass
	return 0
''' ),
		] )


class ExternNullablePointerReturnRealCompileTests( test_support.RealCompileMixin, CompilerTestCase ):
	''' Regression coverage for a real, confirmed silent-data-corruption bug:
	an `@extern` function declared with a `T|None` return type where T is a
	pointer (Ptr[T]/ConstPtr[T]) got its C prototype declared as returning
	the FULL tagged-union struct BY VALUE (plain c_type(function.return_type))
	- but the real foreign symbol's actual ABI just returns a bare, possibly-
	null pointer in a single register. The mismatched calling convention
	silently corrupted the returned pointer VALUE (not a crash, not a
	null-vs-non-null confusion - the wrong bit pattern, non-null but
	incorrect). Confirmed via ws2_32's real inet_ntop, whose documented
	contract is "returns pStringBuf on success": before the original fix,
	the returned pointer compared unequal to pStringBuf even on success.

	That original fix bridged the union at the extern call site (auto-
	wrapping a raw pointer result back into a tagged union, picking the tag
	at runtime from the pointer's own null-ness) - a workaround, not a real
	fix, and one that only covered RETURN types (the identical shape on an
	extern PARAMETER was never fixed at all). Superseded: `T|None` (and any
	other TaggedUnion, and RCClass) is now a compile-time error on any
	`@extern` parameter or return type (see
	Discovery._reject_non_c_type_on_extern_signature) - there is no longer
	any bridging to test. What's left worth testing for real, against the
	real DLL: the CORRECT idiom this compiler already supports natively -
	a bare, inherently-nullable `Ptr[T]`/`ConstPtr[T]` return, checked with
	`is None` (exactly what lib/windows/ws2_32.py's own inet_ntop, and
	lib/socket.py's real caller of it, already do) - still needs to
	round-trip the real pointer VALUE correctly, not just null-vs-non-null,
	against the actual foreign ABI. RejectionTests below covers the new
	compile-time ban itself. '''

	def setUp( self ) -> None:
		self.discovery = Discovery( import_builtins = True )
		self.compiler = Compiler( self.discovery )

	@unittest.skipUnless( test_support.HAS_CC, 'no C compiler (clang/gcc/msvc) found - skipping' )
	@unittest.skipUnless( os.name == 'nt', 'needs a real Winsock DLL to call (ws2_32.dll)' )
	def test_programs_compile_and_run( self ) -> None:
		self.assert_programs_run([
			# the exact shape that demonstrated the bug: inet_ntop's real,
			# documented contract is "returns pStringBuf on success" - a
			# pointer EQUALITY check against a buffer this program itself
			# passed in, not just a null/non-null check, so a corrupted (but
			# still non-null) return value is caught, not just a crash
			( 'nullable_pointer_extern_return_roundtrips_correctly', '''
import compiler
import sys

@extern( 'ws2_32', 'WSAStartup' )
def WSAStartup( wVersionRequested: u16, lpWSAData: Ptr[None] ) -> i32: ...

@extern( 'ws2_32', 'inet_ntop' )
def inet_ntop_nullable( family: i32, pAddr: Ptr[None], pStringBuf: Ptr[u8], StringBufSize: usize ) -> ConstPtr[u8]: ...

def main() -> i32:
	wsadata: Ptr[u8] = sys.alloc[u8]( 512 )
	WSAStartup( 0x0202, compiler.cast( Ptr[None], wsadata ))
	sys.free( compiler.cast( Ptr[None], wsadata ))

	addr_val: u32 = u32( 0x0100007F ) # 127.0.0.1
	strbuf: Ptr[u8] = sys.alloc[u8]( 16 )
	sys.memzero( strbuf, usize( 16 ))
	res = inet_ntop_nullable( 2, compiler.cast( Ptr[None], compiler.addrof( addr_val )), strbuf, usize( 16 ))
	if res is None:
		sys.free( strbuf )
		return 1 # real failure - AF_INET should never actually fail here
	if res != strbuf: # pre-fix (on the return-type bridging hack): fired - the returned pointer VALUE was wrong
		sys.free( strbuf )
		return 2
	sys.free( strbuf )
	return 0
''' ),
			# the null branch: an invalid address family makes inet_ntop
			# return NULL for real - confirms a bare nullable pointer's own
			# `is None` still correctly recognizes it, with no tag/union
			# machinery involved at all
			( 'nullable_pointer_extern_return_null_case_still_recognized_as_none', '''
import compiler
import sys

@extern( 'ws2_32', 'WSAStartup' )
def WSAStartup( wVersionRequested: u16, lpWSAData: Ptr[None] ) -> i32: ...

@extern( 'ws2_32', 'inet_ntop' )
def inet_ntop_nullable( family: i32, pAddr: Ptr[None], pStringBuf: Ptr[u8], StringBufSize: usize ) -> ConstPtr[u8]: ...

def main() -> i32:
	wsadata: Ptr[u8] = sys.alloc[u8]( 512 )
	WSAStartup( 0x0202, compiler.cast( Ptr[None], wsadata ))
	sys.free( compiler.cast( Ptr[None], wsadata ))

	addr_val: u32 = u32( 0x0100007F )
	strbuf: Ptr[u8] = sys.alloc[u8]( 16 )
	sys.memzero( strbuf, usize( 16 ))
	res = inet_ntop_nullable( 999, compiler.cast( Ptr[None], compiler.addrof( addr_val )), strbuf, usize( 16 )) # invalid family -> real NULL
	if res is None:
		sys.free( strbuf )
		return 0
	sys.free( strbuf )
	return 1
''' ),
		] )


class ExternNonCTypeSignatureRejectionTests( CompilerTestCase ):
	''' Discovery._reject_non_c_type_on_extern_signature: an `@extern`
	function's foreign C symbol has no notion of this compiler's own
	RC-managed objects or synthesized tagged unions - only a genuine plain C
	value (a Scalar, a @cstruct/@cunion, a raw CType, a CEnum, a function
	pointer, or Ptr[T]/ConstPtr[T] to one of those) crosses that boundary
	correctly. Covers both halves of what the old auto-bridging hack
	(3e0e325f, removed) left exposed: a TaggedUnion on a PARAMETER (never
	fixed at all, since the hack only ever special-cased RETURN types) and a
	TaggedUnion on a RETURN (previously silently bridged instead of
	rejected) - plus RCClass, which the old hack never considered either
	way. Each fixture calls the offending @extern function from main() -
	an @extern function's own parameter/return types are only resolved (and
	only then can this check ever fire) once something actually schedules
	it, mirroring how every other Function's signature resolution is lazy
	in this compiler (see discovery.py's _make_function_resolver). '''

	def _assert_rejected( self, code: str, expected_substring: str ) -> None:
		self._run( code )
		self.assertTrue(
			any( expected_substring in e for e in self.discovery.errors.errors ),
			self.discovery.errors.errors,
		)

	def test_extern_parameter_tagged_union_rejected( self ) -> None:
		# the previously-unexercised, never-fixed twin of the return-side bug:
		# _lower_call_args -> _coerce_into_union would have built a full
		# tag+payload struct for this argument and handed it to a foreign
		# symbol expecting a bare pointer register - now a compile error
		# instead of a silent ABI mismatch
		self._assert_rejected( '''
@extern( 'c', 'some_extern_fn' )
def some_extern_fn( p: ConstPtr[u8]|None ) -> i32: ...

def main() -> i32:
	some_extern_fn( None )
	return 0
''', 'cannot cross an @extern boundary' )

	def test_extern_return_tagged_union_rejected( self ) -> None:
		# the exact shape 3e0e325f's now-removed auto-bridging hack used to
		# paper over instead of rejecting outright
		self._assert_rejected( '''
@extern( 'c', 'some_extern_fn' )
def some_extern_fn() -> ConstPtr[u8]|None: ...

def main() -> i32:
	some_extern_fn()
	return 0
''', 'cannot cross an @extern boundary' )

	def test_extern_parameter_rcclass_rejected( self ) -> None:
		# an RCClass reference already happens to be pointer-shaped in the
		# generated C, but it points at THIS compiler's own ObjectHeader-
		# prefixed layout, not a plain C value any foreign library was
		# compiled to understand or refcount correctly
		self._assert_rejected( '''
class Foo:
	x: i32

@extern( 'c', 'some_extern_fn' )
def some_extern_fn( f: Foo ) -> i32: ...

def main() -> i32:
	some_extern_fn( Foo( x = 1 ) )
	return 0
''', 'cannot cross an @extern boundary' )

	def test_extern_return_rcclass_rejected( self ) -> None:
		self._assert_rejected( '''
class Foo:
	x: i32

@extern( 'c', 'some_extern_fn' )
def some_extern_fn() -> Foo: ...

def main() -> i32:
	some_extern_fn()
	return 0
''', 'cannot cross an @extern boundary' )


class LocalImportAnnotationResolutionTests( test_support.RealCompileMixin, CompilerTestCase ):
	''' A function-body-local `from X import Y` immediately followed by a
	same-function annotation using Y (`h: Y = ...`) previously failed to
	compile with a spurious "name 'Y' is not defined", even though the
	identical import resolves fine written at module level, and even though
	an UNANNOTATED use of the same locally-imported name (`x = Y`) was
	unaffected. Root cause: type_resolver.py's _ReferenceResolver (the AST-
	rewrite pass resolve_function_body runs over a function body) never
	registered a local ImportFrom's own names into fn.names - only lowering.
	py's own _stmt_ImportFrom did that, which runs in a separate, LATER pass
	(real codegen), too late for THIS pass's own visit_AnnAssign, which
	resolves its annotation via self.discovery.visit() - walking self.
	discovery.scope_stack, which fn IS already pushed onto for the whole of
	resolve_function_body's walk (see its own scope_context(fn)) - just
	nothing had populated fn.names from the import yet by the time the very
	next statement's annotation was resolved. Hit for real in lib/fs.py's
	own open_raw (Windows branch): `from windows.kernel32 import ...,
	HANDLE, ...` immediately followed by `handle: HANDLE = CreateFileA(...)`
	- worked around there by dropping the redundant `: HANDLE` annotation
	(CreateFileA's own declared return type already IS HANDLE), which is
	fine to leave as-is, but left this general resolver gap itself
	unfixed until now. '''

	def setUp( self ) -> None:
		self.discovery = Discovery( import_builtins = True )
		self.compiler = Compiler( self.discovery )

	@unittest.skipUnless( test_support.HAS_CC, 'no C compiler (clang/gcc/msvc) found - skipping' )
	def test_programs_compile_and_run( self ) -> None:
		''' every real compile-and-run program in this class, merged into a
		single executable (one build for the whole class); a nonzero exit is
		decoded back to the failing sub-program and its own return code. '''
		self.assert_programs_run([
			( 'local_import_immediately_used_in_annotation', '''
def main() -> i32:
	from windows.kernel32 import HANDLE, INVALID_HANDLE_VALUE
	h: HANDLE = INVALID_HANDLE_VALUE
	if h != INVALID_HANDLE_VALUE:
		return 1
	return 0
''' ),
			# an aliased import (`as`), with unrelated statements between the
			# import and the annotation that uses it - guards against a fix
			# that only special-cases "the very next statement" or the
			# original (non-aliased) name.
			( 'aliased_local_import_used_in_annotation_after_a_gap', '''
def main() -> i32:
	from windows.kernel32 import HANDLE as H, INVALID_HANDLE_VALUE
	x: i32 = 1
	y: i32 = 2
	h: H = INVALID_HANDLE_VALUE
	if x != 1 or y != 2:
		return 1
	if h != INVALID_HANDLE_VALUE:
		return 2
	return 0
''' ),
			# the identical name, used as a plain VALUE rather than a type
			# annotation, must keep compiling too (this path never broke -
			# see this class's own docstring - but a fix that regresses it
			# would be just as real a bug).
			( 'local_import_used_as_a_value_not_just_a_type', '''
def main() -> i32:
	from windows.kernel32 import INVALID_HANDLE_VALUE
	x = INVALID_HANDLE_VALUE
	if x != INVALID_HANDLE_VALUE:
		return 1
	return 0
''' ),
		] )


class FStringTests( test_support.RealCompileMixin, CompilerTestCase ):
	''' f-string (PEP 498) real end-to-end compile-and-run tests
	(PLAN_FSTRINGS.md). Mirrors StrUpperLowerTests/ListGenericTests' own
	import_builtins=True + real compile-and-run convention - the runtime
	N-part path needs str/Result/UnsafeList[T] for real. '''

	def setUp( self ) -> None:
		self.discovery = Discovery( import_builtins = True )
		self.compiler = Compiler( self.discovery )

	@unittest.skipUnless( test_support.HAS_CC, 'no C compiler (clang/gcc/msvc) found - skipping' )
	def test_programs_compile_and_run( self ) -> None:
		''' every real compile-and-run program in this class, merged into a
		single executable (one build for the whole class); a nonzero exit is
		decoded back to the failing sub-program and its own return code. '''
		self.assert_programs_run([
			# exercises the real N-part runtime path (UnsafeList[str]/slice[str]/
			# str.concat) - a and b are real runtime parameters (not folded away
			# by compile_time_transformer.py), so this is the test that actually
			# proves the whole pass end to end, not just compile-time folding
			( 'multipart_runtime_fstring', '''
def build( a: str, b: str ) -> str:
	return f"{a} {b}!"

def main() -> i32:
	if build( 'hello', 'world' ) != 'hello world!':
		return 1
	return 0
''' ),
			# len(node.values) == 1 - no UnsafeList/slice/concat machinery at
			# all, just the FormattedValue's own str-typed operand directly
			( 'single_interpolation_short_circuit', '''
def build( a: str ) -> str:
	return f"{a}"

def main() -> i32:
	if build( 'solo' ) != 'solo':
		return 1
	return 0
''' ),
			# an int value has no natural str type - resolved via int.__str__()
			# (PLAN_FSTRINGS.md's own value-to-str resolution rules)
			( 'non_str_value_uses_str_dunder', '''
def build( n: int ) -> str:
	return f"n={n}"

def main() -> i32:
	if build( int( 42 )) != 'n=42':
		return 1
	if build( int( -7 )) != 'n=-7':
		return 2
	return 0
''' ),
			( 'repr_conversion_uses_repr_dunder', '''
def build( n: int ) -> str:
	return f"{n!r}"

def main() -> i32:
	if build( int( 5 )) != '5':
		return 1
	return 0
''' ),
			# fully compile-time-known - compile_time_transformer.py's own
			# visit_JoinedStr already collapsed this to a plain str Constant
			# before lowering.py ever sees a JoinedStr node at all; this is an
			# end-to-end proof the fold produces correct, runnable output, not
			# just the right ast.unparse() text (compile_time_transformer_
			# test.py already covers that in isolation)
			( 'compile_time_constant_fstring_folds_away', '''
def main() -> i32:
	if f"answer={1+41}" != 'answer=42':
		return 1
	return 0
''' ),
			# runs the N-part runtime path many times over - a real stress
			# check for the UnsafeList[str] scratch buffer's own lifecycle
			# (construction, N appends, get_ptr, destruction) - matches this
			# codebase's own "repeat-run stress test, not just reasoning"
			# verification convention (see ListThreadSafetyTests)
			( 'repeated_fstring_construction_does_not_leak_or_double_free', '''
def build( n: int ) -> str:
	return f"iteration: value={n}"

def main() -> i32:
	for i in range( 1000 ):
		s: str = build( int( 7 ))
		if s != 'iteration: value=7':
			return 1
	return 0
''' ),
			# !a (PLAN_FSTRINGS.md follow-up) - a dedicated real compile-and-run
			# test against non-ASCII input, per the plan's own verification
			# section. Expected text is real Python's own ascii()-equivalent
			# escaping (repr() here has no surrounding quotes to strip since
			# !a's own metalpy semantics never add quotes - see lowering.py's
			# _lower_ascii_escape comment): a 2-byte-UTF8 codepoint (café,
			# U+00E9) escapes as \xE9-style... actually str._ascii_escape's
			# own lowercase-hex convention is checked directly against real
			# Python's escaping of the bare codepoints, not against repr()'s
			# own quoting.
			( 'bang_a_conversion_escapes_non_ascii', '''
def build( s: str ) -> str:
	return f"{s!a}"

def main() -> i32:
	if build( 'caf\\u00e9' ) != 'caf\\\\xe9':
		return 1
	if build( '\\u00e9\\u0100\\U0001F600' ) != '\\\\xe9\\\\u0100\\\\U0001f600':
		return 2
	if build( 'plain ascii' ) != 'plain ascii':
		return 3
	return 0
''' ),
		] )

	@unittest.skipUnless( _CC is not None, 'no C compiler (clang/gcc/msvc) found - skipping' )
	def test_str_format_spec_width_align_fill( self ) -> None:
		# real Python's own f-string output is the oracle throughout this
		# class's format-spec tests, per PLAN_FSTRINGS.md's own
		# verification section
		self._run( f'''
def build( s: str ) -> str:
	return f"{{s:*^11}}"

def main() -> i32:
	if build( 'hi' ) != {f"{'hi':*^11}"!r}:
		return 1
	if f"{{'left':<8}}" != {f"{'left':<8}"!r}:
		return 2
	if f"{{'right':>8}}" != {f"{'right':>8}"!r}:
		return 3
	return 0
''' )
		self.assertEqual( self.discovery.errors.errors, [] )
		self._assert_compiles_and_runs( emitter_c.emit_c( self.compiler ))

	@unittest.skipUnless( _CC is not None, 'no C compiler (clang/gcc/msvc) found - skipping' )
	def test_str_format_spec_precision_truncates( self ) -> None:
		self._run( f'''
def main() -> i32:
	if f"{{'hello world':.5}}" != {f"{'hello world':.5}"!r}:
		return 1
	return 0
''' )
		self.assertEqual( self.discovery.errors.errors, [] )
		self._assert_compiles_and_runs( emitter_c.emit_c( self.compiler ))

	@unittest.skipUnless( _CC is not None, 'no C compiler (clang/gcc/msvc) found - skipping' )
	def test_int_format_spec_decimal_sign_and_width( self ) -> None:
		self._run( f'''
def build( n: int ) -> str:
	return f"{{n:+06d}}"

def main() -> i32:
	if build( int( 42 )) != {f"{42:+06d}"!r}:
		return 1
	if build( int( -42 )) != {f"{-42:+06d}"!r}:
		return 2
	if f"{{int(0):5d}}" != {f"{0:5d}"!r}:
		return 3
	return 0
''' )
		self.assertEqual( self.discovery.errors.errors, [] )
		self._assert_compiles_and_runs( emitter_c.emit_c( self.compiler ))

	@unittest.skipUnless( _CC is not None, 'no C compiler (clang/gcc/msvc) found - skipping' )
	def test_int_format_spec_decimal_grouping( self ) -> None:
		self._run( f'''
def main() -> i32:
	if f"{{int(1234567):,d}}" != {f"{1234567:,d}"!r}:
		return 1
	if f"{{int(-1234567):_d}}" != {f"{-1234567:_d}"!r}:
		return 2
	return 0
''' )
		self.assertEqual( self.discovery.errors.errors, [] )
		self._assert_compiles_and_runs( emitter_c.emit_c( self.compiler ))

	@unittest.skipUnless( _CC is not None, 'no C compiler (clang/gcc/msvc) found - skipping' )
	def test_int_format_spec_radix_conversions( self ) -> None:
		self._run( f'''
def main() -> i32:
	if f"{{int(255):#x}}" != {f"{255:#x}"!r}:
		return 1
	if f"{{int(255):#X}}" != {f"{255:#X}"!r}:
		return 2
	if f"{{int(8):#o}}" != {f"{8:#o}"!r}:
		return 3
	if f"{{int(5):#b}}" != {f"{5:#b}"!r}:
		return 4
	if f"{{int(0):x}}" != {f"{0:x}"!r}:
		return 5
	return 0
''' )
		self.assertEqual( self.discovery.errors.errors, [] )
		self._assert_compiles_and_runs( emitter_c.emit_c( self.compiler ))

	@unittest.skipUnless( _CC is not None, 'no C compiler (clang/gcc/msvc) found - skipping' )
	def test_int_format_spec_zero_pad_is_sign_aware( self ) -> None:
		# the '0' shorthand's own sign-aware zero-fill: f"{-255:#010x}" ->
		# the '-' and '0x' prefix stay in front, zeros fill AFTER them,
		# not before ('-0000000ff', not '000000-0xff') - this is the one
		# real '=' alignment behavior PLAN_FSTRINGS.md's own scope covers
		self._run( f'''
def main() -> i32:
	if f"{{int(-255):#010x}}" != {f"{-255:#010x}"!r}:
		return 1
	if f"{{int(255):#010x}}" != {f"{255:#010x}"!r}:
		return 2
	if f"{{int(-5):05d}}" != {f"{-5:05d}"!r}:
		return 3
	return 0
''' )
		self.assertEqual( self.discovery.errors.errors, [] )
		self._assert_compiles_and_runs( emitter_c.emit_c( self.compiler ))

	@unittest.skipUnless( _CC is not None, 'no C compiler (clang/gcc/msvc) found - skipping' )
	def test_int_format_spec_zero_pad_is_grouping_aware( self ) -> None:
		# the '0' shorthand COMBINED with grouping (,/_) - a real, confirmed
		# bug (PLAN_STR_FORMAT.md item 4's own writeup): the padding zeros
		# themselves need their own separators too, matching real Python's
		# f"{1234567:015,d}" == '000,001,234,567', NOT '0000001,234,567'
		# (raw zeros in front of an already-grouped string, what a naive
		# "group first, then rjust-pad" two-step gives instead)
		self._run( f'''
def main() -> i32:
	if f"{{int(1234567):015,d}}" != {f"{1234567:015,d}"!r}:
		return 1
	if f"{{int(1234567):013,d}}" != {f"{1234567:013,d}"!r}:
		return 2
	if f"{{int(-1234567):016,d}}" != {f"{-1234567:016,d}"!r}:
		return 3
	if f"{{int(0):06,d}}" != {f"{0:06,d}"!r}:
		return 4
	if f"{{int(1234567):015_d}}" != {f"{1234567:015_d}"!r}:
		return 5
	return 0
''' )
		self.assertEqual( self.discovery.errors.errors, [] )
		self._assert_compiles_and_runs( emitter_c.emit_c( self.compiler ))

	@unittest.skipUnless( _CC is not None, 'no C compiler (clang/gcc/msvc) found - skipping' )
	def test_float_format_spec_precision( self ) -> None:
		# 'f'/'F' fixed-point only (PLAN_STR_FORMAT.md item 4) - real
		# Python's own f-string output is the oracle, same convention as
		# every str/int format-spec test above. f"{1.0:.1f}" is the exact
		# motivating case this pass exists for.
		self._run( f'''
def build( x: f64 ) -> str:
	return f"{{x:.1f}}"

def main() -> i32:
	if build( 1.0 ) != {f"{1.0:.1f}"!r}:
		return 1
	if f"{{3.14159:.3f}}" != {f"{3.14159:.3f}"!r}:
		return 2
	if f"{{7.0:.0f}}" != {f"{7.0:.0f}"!r}:
		return 3
	if f"{{0.0:.2f}}" != {f"{0.0:.2f}"!r}:
		return 4
	return 0
''' )
		self.assertEqual( self.discovery.errors.errors, [] )
		self._assert_compiles_and_runs( emitter_c.emit_c( self.compiler ))

	@unittest.skipUnless( _CC is not None, 'no C compiler (clang/gcc/msvc) found - skipping' )
	def test_float_format_spec_sign_and_width( self ) -> None:
		self._run( f'''
def main() -> i32:
	if f"{{-2.5:.1f}}" != {f"{-2.5:.1f}"!r}:
		return 1
	if f"{{2.5:+.1f}}" != {f"{2.5:+.1f}"!r}:
		return 2
	if f"{{2.5: .1f}}" != {f"{2.5: .1f}"!r}:
		return 3
	if f"{{1.5:>10.1f}}" != {f"{1.5:>10.1f}"!r}:
		return 4
	if f"{{1.5:<10.1f}}" != {f"{1.5:<10.1f}"!r}:
		return 5
	if f"{{1.5:*^10.1f}}" != {f"{1.5:*^10.1f}"!r}:
		return 6
	return 0
''' )
		self.assertEqual( self.discovery.errors.errors, [] )
		self._assert_compiles_and_runs( emitter_c.emit_c( self.compiler ))

	@unittest.skipUnless( _CC is not None, 'no C compiler (clang/gcc/msvc) found - skipping' )
	def test_float_format_spec_zero_pad_is_sign_aware( self ) -> None:
		# the '0' shorthand's own sign-aware zero-fill, same shape int's own
		# equivalent test already covers - the '-' stays in front, zeros
		# fill AFTER it, not before ('-00001.5', not '0000-1.5')
		self._run( f'''
def main() -> i32:
	if f"{{1.5:08.1f}}" != {f"{1.5:08.1f}"!r}:
		return 1
	if f"{{-1.5:08.1f}}" != {f"{-1.5:08.1f}"!r}:
		return 2
	return 0
''' )
		self.assertEqual( self.discovery.errors.errors, [] )
		self._assert_compiles_and_runs( emitter_c.emit_c( self.compiler ))

	def test_str_type_char_on_float_is_a_compile_error( self ) -> None:
		# 'x' is a valid type char for int/radix, but not for float - a
		# clear, named error, not a crash or silently wrong output
		self._run( '''
def main() -> i32:
	return len( f"{1.0:x}" )
''' )
		self.assertTrue( any( "'x' is not valid for float" in e for e in self.discovery.errors.errors ), self.discovery.errors.errors )

	@unittest.skipUnless( _CC is not None, 'no C compiler (clang/gcc/msvc) found - skipping' )
	def test_float_format_spec_exponential( self ) -> None:
		# 'e'/'E' (PLAN_STR_FORMAT.md item 4) - real Python's own f-string
		# output is the oracle, same convention as every other format-spec
		# test in this class. Backed by real snprintf/msvcrt _snprintf -
		# msvcrt's own exponent is always 3 digits ("e+003"), unlike Python/
		# C99's 2-digit floor ("e+03") - emitter_c.py's PROLOGUE fixes this
		# up on Windows (__metalpy_fixup_msvcrt_exponent); this test is the
		# real end-to-end proof that fixup actually produces Python-matching
		# output, not just that it compiles.
		self._run( f'''
def build( x: f64 ) -> str:
	return f"{{x:.2e}}"

def main() -> i32:
	if build( 1234.5 ) != {f"{1234.5:.2e}"!r}:
		return 1
	if f"{{1234.5:.2E}}" != {f"{1234.5:.2E}"!r}:
		return 2
	if f"{{1234.5:e}}" != {f"{1234.5:e}"!r}:
		return 3
	if f"{{0.0001234:e}}" != {f"{0.0001234:e}"!r}:
		return 4
	if f"{{-1234.5:.2e}}" != {f"{-1234.5:.2e}"!r}:
		return 5
	if f"{{5.0:.0e}}" != {f"{5.0:.0e}"!r}:
		return 6
	return 0
''' )
		self.assertEqual( self.discovery.errors.errors, [] )
		self._assert_compiles_and_runs( emitter_c.emit_c( self.compiler ))

	@unittest.skipUnless( _CC is not None, 'no C compiler (clang/gcc/msvc) found - skipping' )
	def test_float_format_spec_exponential_sign_and_width( self ) -> None:
		self._run( f'''
def main() -> i32:
	if f"{{1234.5:+.2e}}" != {f"{1234.5:+.2e}"!r}:
		return 1
	if f"{{1234.5:012.2e}}" != {f"{1234.5:012.2e}"!r}:
		return 2
	if f"{{-1234.5:012.2e}}" != {f"{-1234.5:012.2e}"!r}:
		return 3
	if f"{{1234.5:>15.2e}}" != {f"{1234.5:>15.2e}"!r}:
		return 4
	return 0
''' )
		self.assertEqual( self.discovery.errors.errors, [] )
		self._assert_compiles_and_runs( emitter_c.emit_c( self.compiler ))

	@unittest.skipUnless( _CC is not None, 'no C compiler (clang/gcc/msvc) found - skipping' )
	def test_float_format_spec_general( self ) -> None:
		# 'g'/'G' - precision means SIGNIFICANT digits here, not fractional
		# digits like 'f'/'e' (real snprintf handles this distinction
		# itself), and switches between fixed/exponential notation based on
		# magnitude, stripping trailing zeros - all exercised against real
		# Python's own output
		self._run( f'''
def main() -> i32:
	if f"{{1234.5:.3g}}" != {f"{1234.5:.3g}"!r}:
		return 1
	if f"{{0.0001234:.3g}}" != {f"{0.0001234:.3g}"!r}:
		return 2
	if f"{{1234.5:g}}" != {f"{1234.5:g}"!r}:
		return 3
	if f"{{100000.0:g}}" != {f"{100000.0:g}"!r}:
		return 4
	if f"{{1000000.0:g}}" != {f"{1000000.0:g}"!r}:
		return 5
	if f"{{0.0:.3g}}" != {f"{0.0:.3g}"!r}:
		return 6
	if f"{{123.456:.3G}}" != {f"{123.456:.3G}"!r}:
		return 7
	if f"{{5.0:.0g}}" != {f"{5.0:.0g}"!r}:
		return 8
	return 0
''' )
		self.assertEqual( self.discovery.errors.errors, [] )
		self._assert_compiles_and_runs( emitter_c.emit_c( self.compiler ))

	@unittest.skipUnless( _CC is not None, 'no C compiler (clang/gcc/msvc) found - skipping' )
	def test_float_format_spec_percent( self ) -> None:
		# '%' has no printf equivalent - lib/builtins/__float.py's own
		# _percent_digits scales by 100 and formats as 'f' in metalpy
		# source, then appends the literal '%' - this is the real end-to-
		# end proof that scaling + suffix + sign/width assembly all compose
		# correctly, matching real Python's own f"{x:%}" output
		self._run( f'''
def main() -> i32:
	if f"{{0.1234:.2%}}" != {f"{0.1234:.2%}"!r}:
		return 1
	if f"{{0.1234:%}}" != {f"{0.1234:%}"!r}:
		return 2
	if f"{{-0.1234:8.2%}}" != {f"{-0.1234:8.2%}"!r}:
		return 3
	if f"{{0.1234:8.2%}}" != {f"{0.1234:8.2%}"!r}:
		return 4
	if f"{{1.0:%}}" != {f"{1.0:%}"!r}:
		return 5
	return 0
''' )
		self.assertEqual( self.discovery.errors.errors, [] )
		self._assert_compiles_and_runs( emitter_c.emit_c( self.compiler ))

	@unittest.skipUnless( _CC is not None, 'no C compiler (clang/gcc/msvc) found - skipping' )
	def test_float_format_spec_alt_flag( self ) -> None:
		# '#' (always show the decimal point for 'f'/'F'/'e'/'E', keep
		# trailing zeros for 'g'/'G') - passed straight through to real
		# snprintf/msvcrt _snprintf, which already matches Python's own
		# semantics exactly for every type char, confirmed against real
		# Python's own output
		self._run( f'''
def main() -> i32:
	if f"{{5.0:#.0f}}" != {f"{5.0:#.0f}"!r}:
		return 1
	if f"{{5.0:#f}}" != {f"{5.0:#f}"!r}:
		return 2
	if f"{{5.0:#.0e}}" != {f"{5.0:#.0e}"!r}:
		return 3
	if f"{{5.0:#g}}" != {f"{5.0:#g}"!r}:
		return 4
	if f"{{100000.0:#g}}" != {f"{100000.0:#g}"!r}:
		return 5
	if f"{{5.0:#.0%}}" != {f"{5.0:#.0%}"!r}:
		return 6
	if f"{{5.0:#.0F}}" != {f"{5.0:#.0F}"!r}:
		return 7
	return 0
''' )
		self.assertEqual( self.discovery.errors.errors, [] )
		self._assert_compiles_and_runs( emitter_c.emit_c( self.compiler ))

	@unittest.skipUnless( _CC is not None, 'no C compiler (clang/gcc/msvc) found - skipping' )
	def test_float_format_spec_grouping( self ) -> None:
		# ','/'_' grouping - has no printf equivalent at all (unlike '#'),
		# so it's a separate post-processing pass (lib/builtins/__float.py's
		# _group_integer_part) applied to whatever snprintf already
		# returned, touching only the digits before the first '.' - a
		# correct no-op for 'e'/'E' and for 'g'/'G' in exponential form
		# (only ever one digit there), confirmed against real Python
		self._run( f'''
def main() -> i32:
	if f"{{1234567.891:,.2f}}" != {f"{1234567.891:,.2f}"!r}:
		return 1
	if f"{{1234567.891:_.2f}}" != {f"{1234567.891:_.2f}"!r}:
		return 2
	if f"{{-1234567.891:,.2f}}" != {f"{-1234567.891:,.2f}"!r}:
		return 3
	if f"{{1234567.891:,.0f}}" != {f"{1234567.891:,.0f}"!r}:
		return 4
	if f"{{1234.5:,e}}" != {f"{1234.5:,e}"!r}:
		return 5
	if f"{{1234567.891:,g}}" != {f"{1234567.891:,g}"!r}:
		return 6
	if f"{{1234.56:,g}}" != {f"{1234.56:,g}"!r}:
		return 7
	if f"{{1234567.891:,.2%}}" != {f"{1234567.891:,.2%}"!r}:
		return 8
	if f"{{1234567.891:20,.2f}}" != {f"{1234567.891:20,.2f}"!r}:
		return 9
	return 0
''' )
		self.assertEqual( self.discovery.errors.errors, [] )
		self._assert_compiles_and_runs( emitter_c.emit_c( self.compiler ))

	@unittest.skipUnless( _CC is not None, 'no C compiler (clang/gcc/msvc) found - skipping' )
	def test_float_format_spec_zero_pad_is_grouping_aware( self ) -> None:
		# the '0' shorthand COMBINED with grouping (,/_) - same real,
		# confirmed bug int's own equivalent test documents
		# (PLAN_STR_FORMAT.md item 4), just for float: only the digits
		# BEFORE the first '.' (or, for '%', before the trailing '%') are
		# the groupable "integer part" that gets padded+grouped together -
		# the fractional digits/exponent/'%' suffix are left untouched and
		# reappended, confirmed against real Python's own output, which
		# groups the zero-fill itself just like the plain digits
		# (f"{1234567.89:018,.2f}" == '000,001,234,567.89')
		self._run( f'''
def main() -> i32:
	if f"{{1234567.89:018,.2f}}" != {f"{1234567.89:018,.2f}"!r}:
		return 1
	if f"{{1234567.89:017,.2f}}" != {f"{1234567.89:017,.2f}"!r}:
		return 2
	if f"{{-1234567.89:018,.2f}}" != {f"{-1234567.89:018,.2f}"!r}:
		return 3
	if f"{{1234567.891:020,e}}" != {f"{1234567.891:020,e}"!r}:
		return 4
	if f"{{1234567.891:020,g}}" != {f"{1234567.891:020,g}"!r}:
		return 5
	if f"{{1234567.891:015,.2%}}" != {f"{1234567.891:015,.2%}"!r}:
		return 6
	if f"{{-1234.5:020,.2%}}" != {f"{-1234.5:020,.2%}"!r}:
		return 7
	return 0
''' )
		self.assertEqual( self.discovery.errors.errors, [] )
		self._assert_compiles_and_runs( emitter_c.emit_c( self.compiler ))

	@unittest.skipUnless( _CC is not None, 'no C compiler (clang/gcc/msvc) found - skipping' )
	def test_float_format_spec_uppercase_F( self ) -> None:
		# uppercase 'F' specifically - legacy msvcrt.dll's own _snprintf
		# silently produces empty output for "%F" (confirmed by a real
		# test against this system's own msvcrt.dll: unlike 'E'/'G', which
		# it supports fine, 'F' was only added to printf in C99, after
		# legacy msvcrt), a real, already-shipped bug this test would have
		# caught immediately - emitter_c.py's PROLOGUE now substitutes
		# lowercase 'f' internally on Windows for this one conversion
		# character, correct for every finite value. inf/nan display
		# (see test_float_format_spec_inf_nan below) is handled entirely
		# separately, in metalpy source, before compiler.format_f64 (and
		# so this 'f'-vs-'F' substitution) is ever reached - Python shows
		# "inf"/"nan" identically regardless of 'f' vs 'F', so there's no
		# capitalization difference left to worry about here either
		self._run( f'''
def main() -> i32:
	if f"{{1.0:.1F}}" != {f"{1.0:.1F}"!r}:
		return 1
	if f"{{-2.5:8.1F}}" != {f"{-2.5:8.1F}"!r}:
		return 2
	if f"{{5.0:#.0F}}" != {f"{5.0:#.0F}"!r}:
		return 3
	return 0
''' )
		self.assertEqual( self.discovery.errors.errors, [] )
		self._assert_compiles_and_runs( emitter_c.emit_c( self.compiler ))

	@unittest.skipUnless( _CC is not None, 'no C compiler (clang/gcc/msvc) found - skipping' )
	def test_float_format_spec_inf_nan( self ) -> None:
		# a real, confirmed bug (PLAN_STR_FORMAT.md item 4): legacy
		# msvcrt.dll's own _snprintf produces outright GARBAGE for
		# infinity ("1.$" for "%.1f" of +inf, confirmed against this
		# system's own msvcrt.dll - not merely untested, actually wrong).
		# lib/builtins/__float.py now special-cases NaN/infinity (via the
		# new compiler.is_nan/is_inf intrinsics) before ever calling
		# compiler.format_f64 at all, matching real Python: precision/
		# type_char/alt are all ignored ("inf" regardless of 'f'/'e'/'g'),
		# but sign/width/zero-pad still apply, and grouping is a no-op
		# even when requested (no comma ever appears inside "inf")
		self._run( f'''
def get_pos_inf() -> f64:
	with compiler.saturate_arithmetic:
		big: f64 = 1.0e300
		return big * big

def get_neg_inf() -> f64:
	return -get_pos_inf()

def get_nan() -> f64:
	with compiler.wrap_arithmetic:
		return get_pos_inf() + get_neg_inf()

def main() -> i32:
	if f"{{get_pos_inf():.1f}}" != {f"{float('inf'):.1f}"!r}:
		return 1
	if f"{{get_neg_inf():.1f}}" != {f"{float('-inf'):.1f}"!r}:
		return 2
	if f"{{get_nan():.1f}}" != {f"{float('nan'):.1f}"!r}:
		return 3
	if f"{{get_pos_inf():.1e}}" != {f"{float('inf'):.1e}"!r}:
		return 4
	if f"{{get_nan():.1g}}" != {f"{float('nan'):.1g}"!r}:
		return 5
	if f"{{get_pos_inf():+.1f}}" != {f"{float('inf'):+.1f}"!r}:
		return 6
	if f"{{get_pos_inf():08.1f}}" != {f"{float('inf'):08.1f}"!r}:
		return 7
	if f"{{get_pos_inf():.1%}}" != {f"{float('inf'):.1%}"!r}:
		return 8
	if f"{{get_pos_inf():015,.1f}}" != {f"{float('inf'):015,.1f}"!r}:
		return 9
	if f"{{get_neg_inf():015,.1f}}" != {f"{float('-inf'):015,.1f}"!r}:
		return 10
	if f"{{get_neg_inf():08.1%}}" != {f"{float('-inf'):08.1%}"!r}:
		return 11
	if f"{{get_nan():+.1f}}" != {f"{float('nan'):+.1f}"!r}:
		return 12
	return 0
''' )
		self.assertEqual( self.discovery.errors.errors, [] )
		self._assert_compiles_and_runs( emitter_c.emit_c( self.compiler ))

	@unittest.skipUnless( _CC is not None, 'no C compiler (clang/gcc/msvc) found - skipping' )
	def test_float_format_spec_none_type_with_precision( self ) -> None:
		# f"{x:.2}" (a literal spec with a precision but no type char) -
		# real Python's own "None" presentation type, closer to 'g' than
		# to plain 'f' (PLAN_STR_FORMAT.md item 4's own note), except
		# fixed-point results always keep at least one fractional digit
		# (f"{5.0:.2}" == '5.0', not 'g''s own '5') - confirmed against
		# real Python. No-precision-no-type (f"{x:10}"/bare f"{x}") still
		# falls back to plain 'f' - that needs Python's real shortest-
		# round-trip repr algorithm instead, not implemented yet
		self._run( f'''
def main() -> i32:
	if f"{{5.0:.2}}" != {f"{5.0:.2}"!r}:
		return 1
	if f"{{1234.5:.2}}" != {f"{1234.5:.2}"!r}:
		return 2
	if f"{{0.0001234:.2}}" != {f"{0.0001234:.2}"!r}:
		return 3
	if f"{{1234.5:10.2}}" != {f"{1234.5:10.2}"!r}:
		return 4
	if f"{{1234.5:.6}}" != {f"{1234.5:.6}"!r}:
		return 5
	if f"{{1234.5:#.2}}" != {f"{1234.5:#.2}"!r}:
		return 6
	if f"{{-5.0:.2}}" != {f"{-5.0:.2}"!r}:
		return 7
	if f"{{5.0:015,.2}}" != {f"{5.0:015,.2}"!r}:
		return 8
	return 0
''' )
		self.assertEqual( self.discovery.errors.errors, [] )
		self._assert_compiles_and_runs( emitter_c.emit_c( self.compiler ))

	@unittest.skipUnless( _CC is not None, 'no C compiler (clang/gcc/msvc) found - skipping' )
	def test_float_repr_shortest_roundtrip( self ) -> None:
		# bare f"{x}" (no format spec at all) and the no-type/no-precision
		# spec both fall through to f64._repr_digits/_repr_digits_raw - the
		# shortest decimal text that round-trips back to the exact same
		# double (via the new compiler.parse_f64 intrinsic, an iterative
		# search over compiler.format_f64's 'e'-conversion precision), then
		# re-rendered into Python's own fixed-vs-scientific presentation
		# (fixed for -4 <= exponent < 16, scientific otherwise - see
		# _f64_repr_from_scientific's own comment for how that threshold was
		# confirmed against real Python). Covers both sides of that exact
		# threshold (1e15 fixed / 1e16 scientific, 1e-4 fixed / 1e-5
		# scientific) plus the smallest/largest finite doubles, since those
		# scientific-notation cases are exactly where a real bug lived
		# before this test existed: an IfExp (ternary) lowering bug -
		# `str('-') if exponent < 0 else str('+')`, used to build the
		# exponent's sign character - double-freed/UAF'd the branch value
		# (see IfExpTempLifetimeTests for the general fix), so every
		# scientific-notation repr crashed or produced garbage.
		self._run( f'''
def build( x: f64 ) -> str:
	return f"{{x}}"

def main() -> i32:
	if build( 1.0 ) != {str(1.0)!r}:
		return 1
	if f"{{0.1}}" != {str(0.1)!r}:
		return 2
	if f"{{100.0}}" != {str(100.0)!r}:
		return 3
	if f"{{1000000.0}}" != {str(1000000.0)!r}:
		return 4
	if f"{{1e15}}" != {str(1e15)!r}:
		return 5
	if f"{{1e16}}" != {str(1e16)!r}:
		return 6
	if f"{{1e17}}" != {str(1e17)!r}:
		return 7
	if f"{{0.0001}}" != {str(0.0001)!r}:
		return 8
	if f"{{1e-05}}" != {str(1e-05)!r}:
		return 9
	if f"{{123456789012345.0}}" != {str(123456789012345.0)!r}:
		return 10
	if f"{{3.14159265358979}}" != {str(3.14159265358979)!r}:
		return 11
	if f"{{-5.0}}" != {str(-5.0)!r}:
		return 12
	if f"{{-0.1}}" != {str(-0.1)!r}:
		return 13
	if f"{{5e-324}}" != {str(5e-324)!r}:
		return 14
	if f"{{1.7976931348623157e+308}}" != {str(1.7976931348623157e+308)!r}:
		return 15
	if f"{{1234567.0}}" != {str(1234567.0)!r}:
		return 16
	if f"{{1234567890123.0}}" != {str(1234567890123.0)!r}:
		return 17
	if f"{{10.0}}" != {str(10.0)!r}:
		return 18
	return 0
''' )
		self.assertEqual( self.discovery.errors.errors, [] )
		self._assert_compiles_and_runs( emitter_c.emit_c( self.compiler ))

	@unittest.skipUnless( _CC is not None, 'no C compiler (clang/gcc/msvc) found - skipping' )
	def test_float_repr_width_no_type_no_precision( self ) -> None:
		# f"{x:10}" - a width/align/fill spec with no type char and no
		# precision - takes the SAME _repr_digits path as bare f"{x}"
		# (is_none_type_no_precision in _lower_float_format_spec), just
		# padded afterward
		self._run( f'''
def main() -> i32:
	if f"{{1e16:>12}}" != {f"{1e16:>12}"!r}:
		return 1
	if f"{{1e-05:<12}}" != {f"{1e-05:<12}"!r}:
		return 2
	if f"{{1.5:010}}" != {f"{1.5:010}"!r}:
		return 3
	return 0
''' )
		self.assertEqual( self.discovery.errors.errors, [] )
		self._assert_compiles_and_runs( emitter_c.emit_c( self.compiler ))

	@unittest.skipUnless( _CC is not None, 'no C compiler (clang/gcc/msvc) found - skipping' )
	def test_explicit_conversion_plus_format_spec_runtime( self ) -> None:
		# f"{n!r:>8}" against a real runtime int - the spec formats the
		# ALREADY-converted str (padding), not n's own int-typed value, so
		# this exercises the two-stage conversion-then-spec path end to
		# end (not just the IR-shape assertion lowering_test.py already
		# makes for the same shape)
		self._run( f'''
def build( n: int ) -> str:
	return f"{{n!r:>8}}"

def main() -> i32:
	if build( int( 5 )) != {f"{5!r:>8}"!r}:
		return 1
	return 0
''' )
		self.assertEqual( self.discovery.errors.errors, [] )
		self._assert_compiles_and_runs( emitter_c.emit_c( self.compiler ))


class GeneratorFunctionTests( test_support.RealCompileMixin, CompilerTestCase ):
	''' PLAN_GENERATORS.md - a plain function containing `yield`, where
	every yield is a direct top-level statement (v1), the single yield
	inside a direct top-level while loop (Phase 2 - PLAN_GENERATORS.md's
	own motivating range()-style example), or the single yield inside a
	direct top-level for loop - over range() (Phase 4), a list-like
	__len__/__getitem__ indexable, or another generator's own __next__()
	(both Phase 5, matching the user-facing "remaining phases roadmap"'s
	own Phase 1 - one generator consuming another this way is the
	realistic way generators actually get exercised/tested). Real
	compile-and-run - not just "does it lower", the whole point is the
	generated C state machine actually behaves like Python's own generator
	semantics, including RC correctness on early abandonment (the
	"function epilogue moves into __del__" idea this plan doc is built
	around) for every unit shape. '''
	def setUp( self ) -> None:
		self.discovery = Discovery( import_builtins = True )
		self.compiler = Compiler( self.discovery )

	@unittest.skipUnless( test_support.HAS_CC, 'no C compiler (clang/gcc/msvc) found - skipping' )
	def test_programs_compile_and_run( self ) -> None:
		self.assert_programs_run([
			( 'multi_yield_state_transitions_and_exhaustion', '''
def counter() -> Iterator[i32]:
	x: i32 = 1
	yield x
	x = 2
	yield x
	x = 3
	yield x

def main() -> i32:
	with compiler.wrap_arithmetic:
		g = counter()
		a = g.__next__()
		if a is None:
			return 1
		b = g.__next__()
		if b is None:
			return 2
		c = g.__next__()
		if c is None:
			return 3
		d = g.__next__()
		if d is not None:
			return 4
		return 0
''' ),
			( 'calling_the_generator_function_runs_no_body_code', '''
def only_yields_if_called() -> Iterator[i32]:
	yield 1

def main() -> i32:
	with compiler.wrap_arithmetic:
		g = only_yields_if_called() # constructing it must not run any body code
		return 0
''' ),
			( 'dropped_mid_iteration_decrefs_captured_parameter', '''
class Box:
	v: i32
	def __init__( self, v: i32 ) -> None:
		self.v = v

def gen( b: Box ) -> Iterator[i32]:
	yield b.v
	yield b.v

def make_and_partially_consume( b: Box ) -> None:
	g = gen( b )
	first = g.__next__() # only one of the two yields is ever consumed
	# g goes out of scope here, still mid-iteration - PLAN_GENERATORS.md's
	# own point: dropping it must still decref its captured Box parameter,
	# via the ordinary, unmodified $$__destructor__ cascade every other
	# RCClass already gets (see type_resolver.py's ensure_generator_
	# synthesized - ordinary destructor synthesis is untouched, a captured
	# parameter is valid from construction onward unconditionally)

def main() -> i32:
	with compiler.wrap_arithmetic:
		b = Box( v = 42 )
		if compiler.refcount( b ) != 1:
			return 1
		make_and_partially_consume( b )
		if compiler.refcount( b ) != 1:
			return 2
		return 0
''' ),
			# --- Phase 2: a while loop containing exactly one yield, as a
			# direct statement of the loop's own body - PLAN_GENERATORS.md's
			# own motivating example (a real range()) finally compiles and
			# runs, not just textual `range()` sugar (lowering.py's
			# _is_range_call). See _build_while_unit_guard's own docstring
			# for the resumable-loop restructuring this compiles down to.
			( 'while_loop_resumable_across_next_calls', '''
def counter( count: usize ) -> Iterator[usize]:
	i: usize = 0
	while i < count:
		yield i
		with compiler.wrap_arithmetic:
			i += 1

def main() -> i32:
	with compiler.wrap_arithmetic:
		g = counter( 3 )
		a = g.__next__()
		if a is None:
			return 1
		b = g.__next__()
		if b is None:
			return 2
		c = g.__next__()
		if c is None:
			return 3
		d = g.__next__()
		if d is not None:
			return 4
		return 0
''' ),
			( 'while_loop_dropped_mid_iteration_decrefs_captured_parameter', '''
class Box:
	v: usize
	def __init__( self, v: usize ) -> None:
		self.v = v

def gen( b: Box ) -> Iterator[usize]:
	i: usize = 0
	while i < b.v:
		yield i
		with compiler.wrap_arithmetic:
			i += 1

def make_and_partially_consume( b: Box ) -> None:
	g = gen( b )
	first = g.__next__()
	second = g.__next__() # b.v is 10 - only 2 of 10 iterations consumed

def main() -> i32:
	with compiler.wrap_arithmetic:
		b = Box( v = 10 )
		if compiler.refcount( b ) != 1:
			return 1
		make_and_partially_consume( b )
		if compiler.refcount( b ) != 1:
			return 2
		return 0
''' ),
			( 'bare_yield_and_while_unit_mixed_in_one_generator', '''
def mixed( count: usize ) -> Iterator[usize]:
	hundred: usize = 100
	yield hundred
	i: usize = 0
	while i < count:
		yield i
		with compiler.wrap_arithmetic:
			i += 1
	two_hundred: usize = 200
	yield two_hundred

def main() -> i32:
	with compiler.wrap_arithmetic:
		g = mixed( 2 )
		a = g.__next__()
		if a is None:
			return 1
		b = g.__next__()
		if b is None:
			return 2
		c = g.__next__()
		if c is None:
			return 3
		d = g.__next__()
		if d is None:
			return 4
		e = g.__next__()
		if e is not None:
			return 5
		return 0
''' ),
			# --- Phase 3: `for x in <generator call>:` consumption
			# (lowering.py's _lower_for_over_iterator) - drives __next__()
			# directly, no manual is-None/narrowing boilerplate at the call
			# site. Also the first test to check EXACT yielded VALUES (not
			# just is-None/is-not-None) - it does so by summing/counting
			# through ordinary arithmetic on the for-loop's own bound target,
			# which only works correctly if _lower_for_over_iterator's own
			# narrow()-based payload extraction is genuinely correct, not
			# just "is None" correct (see PLAN_GENERATORS.md's own STATUS
			# section on why this couldn't be checked directly until now).
			( 'for_loop_consumes_a_while_unit_generator', '''
def counter( count: usize ) -> Iterator[usize]:
	i: usize = 0
	while i < count:
		yield i
		with compiler.wrap_arithmetic:
			i += 1

def main() -> i32:
	with compiler.wrap_arithmetic:
		total: usize = 0
		n: usize = 0
		for x in counter( 5 ):
			total += x
			n += 1
		if n != 5:
			return 1
		if total != 10: # 0+1+2+3+4
			return 2
		return 0
''' ),
			( 'for_loop_over_generator_releases_it_and_its_captured_parameter', '''
class Box:
	v: usize
	def __init__( self, v: usize ) -> None:
		self.v = v

def gen( b: Box ) -> Iterator[usize]:
	i: usize = 0
	while i < b.v:
		yield i
		with compiler.wrap_arithmetic:
			i += 1

def consume_fully( b: Box ) -> None:
	with compiler.wrap_arithmetic:
		total: usize = 0
		for x in gen( b ):
			total += x
	# gen(b)'s own generator object went fully out of scope here (the loop
	# ran to exhaustion) - the hidden __for_obj local must still be
	# released at this function's own end, decref-ing b in turn

def main() -> i32:
	with compiler.wrap_arithmetic:
		b = Box( v = 5 )
		if compiler.refcount( b ) != 1:
			return 1
		consume_fully( b )
		if compiler.refcount( b ) != 1:
			return 2
		return 0
''' ),
			( 'for_loop_breaking_early_still_releases_the_generator', '''
class Box:
	v: usize
	def __init__( self, v: usize ) -> None:
		self.v = v

def gen( b: Box ) -> Iterator[usize]:
	i: usize = 0
	while i < b.v:
		yield i
		with compiler.wrap_arithmetic:
			i += 1

def consume_partially( b: Box ) -> None:
	for x in gen( b ):
		if x == 2:
			break

def main() -> i32:
	with compiler.wrap_arithmetic:
		b = Box( v = 5 )
		if compiler.refcount( b ) != 1:
			return 1
		consume_partially( b )
		if compiler.refcount( b ) != 1:
			return 2
		return 0
''' ),
			# --- Phase 4: `for x in range(...):` containing yield, as a
			# direct top-level statement of the generator body -
			# type_resolver.py's _desugar_generator_for_loops rewrites this,
			# in place, into the exact while-loop shape Phase 2 already
			# handles, BEFORE unit collection ever runs - zero changes to
			# the while-unit machinery itself. Confirmed the desugared C is
			# structurally IDENTICAL to the hand-written while-loop version
			# (same $t-numbered instructions, same __gen_resuming_0 local)
			# by inspecting the emitted C directly during development.
			( 'for_loop_over_range_containing_yield', '''
def counter( count: usize ) -> Iterator[usize]:
	for i in range( count ):
		yield i

def main() -> i32:
	with compiler.wrap_arithmetic:
		g = counter( 5 )
		a = g.__next__()
		if a is None:
			return 1
		b = g.__next__()
		if b is None:
			return 2
		c = g.__next__()
		if c is None:
			return 3
		d = g.__next__()
		if d is None:
			return 4
		e = g.__next__()
		if e is None:
			return 5
		f = g.__next__()
		if f is not None:
			return 6
		return 0
''' ),
			( 'for_loop_over_range_composes_with_for_loop_consumption', '''
class Box:
	v: usize
	def __init__( self, v: usize ) -> None:
		self.v = v

def gen( b: Box ) -> Iterator[usize]:
	for i in range( b.v ):
		yield i

def consume_via_for( b: Box ) -> None:
	with compiler.wrap_arithmetic:
		total: usize = 0
		for x in gen( b ):
			total += x

def main() -> i32:
	with compiler.wrap_arithmetic:
		b = Box( v = 5 )
		if compiler.refcount( b ) != 1:
			return 1
		consume_via_for( b )
		if compiler.refcount( b ) != 1:
			return 2
		return 0
''' ),
			# --- Phase 1: `for x in <expr>:` inside a generator body, over
			# a non-range() iterable - both the indexable shape (__len__/
			# __getitem__, e.g. list[T]) and the iterator shape (__next__()
			# -> T|None, i.e. one generator consuming another). Unblocked by
			# type_resolver.py's _resolve_expr_type_for_desugar, reusing
			# _ReferenceResolver._type_of_expr (already proven for match-
			# statement subjects) to resolve the iterated expression's type
			# entirely from AST, before any real lowering exists.
			( 'for_loop_over_list_inside_generator', '''
def double_all( xs: list[i32] ) -> Iterator[i32]:
	for x in xs:
		doubled: i32 = 0
		with compiler.wrap_arithmetic:
			doubled = x * 2
		yield doubled

def main() -> i32:
	with compiler.wrap_arithmetic:
		xs: list[i32] = list[i32]()
		xs.append( 1 ).unwrap( 'append failed' )
		xs.append( 2 ).unwrap( 'append failed' )
		xs.append( 3 ).unwrap( 'append failed' )
		g = double_all( xs )
		a = g.__next__()
		if a is None:
			return 1
		b = g.__next__()
		if b is None:
			return 2
		c = g.__next__()
		if c is None:
			return 3
		d = g.__next__()
		if d is not None:
			return 4
		return 0
''' ),
			( 'for_loop_over_list_releases_it_and_its_captured_parameter', '''
def double_all( xs: list[i32] ) -> Iterator[i32]:
	for x in xs:
		doubled: i32 = 0
		with compiler.wrap_arithmetic:
			doubled = x * 2
		yield doubled

def make_and_partially_consume( xs: list[i32] ) -> None:
	g = double_all( xs )
	first = g.__next__()
	# g goes out of scope here, mid-iteration - g's own __for_obj_N field
	# holds a SEPARATE reference to xs, must also be released

def main() -> i32:
	with compiler.wrap_arithmetic:
		xs: list[i32] = list[i32]()
		xs.append( 1 ).unwrap( 'append failed' )
		xs.append( 2 ).unwrap( 'append failed' )
		xs.append( 3 ).unwrap( 'append failed' )
		if compiler.refcount( xs ) != 1:
			return 1
		make_and_partially_consume( xs )
		if compiler.refcount( xs ) != 1:
			return 2
		return 0
''' ),
			( 'for_loop_consumes_another_generator_inside_a_generator', '''
def counter( count: usize ) -> Iterator[usize]:
	i: usize = 0
	while i < count:
		yield i
		with compiler.wrap_arithmetic:
			i += 1

def doubled( count: usize ) -> Iterator[usize]:
	for x in counter( count ):
		y: usize = 0
		with compiler.wrap_arithmetic:
			y = x * 2
		yield y

def main() -> i32:
	with compiler.wrap_arithmetic:
		g = doubled( 3 )
		a = g.__next__()
		if a is None:
			return 1
		b = g.__next__()
		if b is None:
			return 2
		c = g.__next__()
		if c is None:
			return 3
		d = g.__next__()
		if d is not None:
			return 4
		return 0
''' ),
			( 'for_loop_over_nested_generator_releases_both_levels', '''
class Box:
	v: usize
	def __init__( self, v: usize ) -> None:
		self.v = v

def counter( b: Box ) -> Iterator[usize]:
	i: usize = 0
	while i < b.v:
		yield i
		with compiler.wrap_arithmetic:
			i += 1

def doubled( b: Box ) -> Iterator[usize]:
	for x in counter( b ):
		y: usize = 0
		with compiler.wrap_arithmetic:
			y = x * 2
		yield y

def make_and_partially_consume( b: Box ) -> None:
	g = doubled( b )
	first = g.__next__()
	# g's own __for_obj_N field holds the inner counter(b) generator,
	# which ITSELF holds b as its own captured parameter - both levels
	# must release correctly when g is dropped mid-iteration

def main() -> i32:
	with compiler.wrap_arithmetic:
		b = Box( v = 5 )
		if compiler.refcount( b ) != 1:
			return 1
		make_and_partially_consume( b )
		if compiler.refcount( b ) != 1:
			return 2
		return 0
''' ),
		# --- Phase 2a: `if`/`else` containing yield, at most one yield
		# per branch, as a direct statement of its own branch. Uses the
		# branch-stable-condition resume pattern (see
		# _build_if_unit_guard's own docstring): re-evaluating `cond`
		# on resume is safe because nothing but the generator's own
		# code touches its fields between __next__() calls, so it
		# always lands back in the branch that yielded. Exercises both
		# branches (separate generator instances) and post-yield code
		# in the yielding branch (the `resuming` flag's own body),
		# falling through to a shared tail unit after the if/else.
		( 'if_else_yield_resumes_correct_branch_and_falls_through', '''
def alternator( flag: bool ) -> Iterator[i32]:
	if flag:
		one: i32 = 1
		yield one
		side: i32 = 0
		with compiler.wrap_arithmetic:
			side = 100
	else:
		two: i32 = 2
		yield two
	three: i32 = 3
	yield three

def main() -> i32:
	with compiler.wrap_arithmetic:
		g = alternator( True )
		a = g.__next__() # if-branch
		if a is None:
			return 1
		b = g.__next__() # resumes if-branch post-yield code, falls through to tail
		if b is None:
			return 2
		c = g.__next__() # exhausted
		if c is not None:
			return 3

		g2 = alternator( False )
		d = g2.__next__() # else-branch
		if d is None:
			return 4
		e = g2.__next__() # falls through to tail
		if e is None:
			return 5
		f = g2.__next__() # exhausted
		if f is not None:
			return 6
		return 0
''' ),
		# --- Phase 2b: `with compiler.wrap_arithmetic/saturate_arithmetic/
		# panic_arithmetic(...):` wrapping a bare yield as its entire
		# body - transparent unwrap/rewrap during unit collection
		# (_yield_with_wrapper), no new unit kind. Confirms the mode
		# is still correctly scoped when the segment is lowered.
		( 'with_wrapped_yield_units', '''
def counter( start: i32 ) -> Iterator[i32]:
	x: i32 = start
	with compiler.wrap_arithmetic:
		yield x
	with compiler.wrap_arithmetic:
		x = x + 1
	with compiler.wrap_arithmetic:
		yield x

def main() -> i32:
	with compiler.wrap_arithmetic:
		g = counter( 10 )
		a = g.__next__()
		if a is None:
			return 1
		b = g.__next__()
		if b is None:
			return 2
		c = g.__next__()
		if c is not None:
			return 3
		return 0
''' ),
		# --- Phase 3: generic generator functions (`def gen[T](x: T) ->
		# Iterator[T]:`) - both an explicit instantiation (`gen[i32](...)`)
		# and an inferred one (`gen(seven)`, T inferred from the
		# argument's own static type - a bare int LITERAL argument hits a
		# pre-existing, generator-unrelated inference gap in this
		# compiler, confirmed via a standalone repro against an ordinary
		# non-generator generic function too, so this uses a typed local
		# instead, same as any other generic-inference call site would
		# need to). monomorphize.py's substitute_type_params gained a
		# GeneratorType case (substitutes elem_type, never interned - see
		# GeneratorType's own docstring); type_resolver.py's ensure_
		# resolved and _ReferenceResolver.visit_Call/_type_of_expr all
		# needed their own fix to actually reach a generic call's
		# monomorphized copy with ensure_generator_synthesized (the plain-
		# Function fast path only ever sees a Specialization-wrapped
		# generic function's ABSTRACT base, never the concrete copy - see
		# each fix's own comment for the specific gap it closes).
		( 'generic_generator_explicit_and_inferred_instantiation', '''
def gen[T]( x: T ) -> Iterator[T]:
	yield x

def main() -> i32:
	with compiler.wrap_arithmetic:
		g1 = gen[i32]( 5 ) # explicit instantiation
		a = g1.__next__()
		if a is None:
			return 1
		b = g1.__next__()
		if b is not None:
			return 2

		seven: i32 = 7
		g2 = gen( seven ) # inferred instantiation
		c = g2.__next__()
		if c is None:
			return 3
		d = g2.__next__()
		if d is not None:
			return 4
		return 0
''' ),
		( 'generic_generator_two_instantiations_coexist_independently', '''
def gen[T]( x: T ) -> Iterator[T]:
	yield x

def main() -> i32:
	with compiler.wrap_arithmetic:
		g1 = gen[i32]( 5 )
		a = g1.__next__()
		if a is None:
			return 1

		g2 = gen[usize]( 9 ) # a DIFFERENT instantiation - independent backing class
		b = g2.__next__()
		if b is None:
			return 2
		return 0
''' ),
		( 'generic_generator_consumed_via_for_loop', '''
def gen[T]( x: T, count: usize ) -> Iterator[T]:
	i: usize = 0
	while i < count:
		yield x
		with compiler.wrap_arithmetic:
			i += 1

def main() -> i32:
	with compiler.wrap_arithmetic:
		total: i32 = 0
		n: usize = 0
		for v in gen[i32]( 3, 4 ):
			total += v
			n += 1
		if n != 4:
			return 1
		if total != 12: # 3+3+3+3
			return 2
		return 0
''' ),
		# --- Phase 4: fallible generators (`Generator[T,E]`, TODO.txt's
		# original open question). __next__ returns Result[elem_type|None,
		# E] instead of the bare union - or_return() inside the body
		# engages the EXISTING checked-arithmetic/_require_result_return
		# machinery for free (no special generator-side flag - purely a
		# consequence of __next__'s own declared return type, same as any
		# other fallible function). The one genuinely new piece
		# (type_resolver.py's _pessimistic_done_prefix): self.__state gets
		# set to the "permanently done" sentinel BEFORE, not after, every
		# block of user code that might contain a fallible early return -
		# so an or_return() failure partway through a unit leaves the
		# generator correctly, permanently exhausted (Ok(None) forever
		# after) rather than re-entering and re-running the same
		# (possibly already-mutated-state) code again on the next call.
		( 'fallible_generator_or_return_propagates_and_then_stays_done', '''
@union
class BoomError:
	Boom: None

def maybe_bad( i: usize, boom_at: usize ) -> Result[usize, BoomError]:
	if i == boom_at:
		return Result.Err( BoomError.Boom( None ))
	return Result.Ok( i )

def counter( limit: usize, boom_at: usize ) -> Generator[usize, BoomError]:
	i: usize = 0
	while i < limit:
		v: usize = maybe_bad( i, boom_at ).or_return()
		yield i
		with compiler.wrap_arithmetic:
			i += 1

def main() -> i32:
	with compiler.wrap_arithmetic:
		g = counter( 5, 2 )
		r0 = g.__next__() # i=0 - succeeds
		match r0:
			case Result.Err( e ):
				return 1
			case Result.Ok( a ):
				pass
		r1 = g.__next__() # i=1 - succeeds
		match r1:
			case Result.Err( e ):
				return 2
			case Result.Ok( b ):
				pass
		r2 = g.__next__() # i=2 == boom_at - or_return() fires
		errored = False
		match r2:
			case Result.Err( e ):
				errored = True
			case Result.Ok( c ):
				return 3
		if not errored:
			return 4
		r3 = g.__next__() # permanently done - Ok(None), not a re-run of the failing code
		match r3:
			case Result.Err( e ):
				return 5
			case Result.Ok( d ):
				pass
		r4 = g.__next__() # still permanently done, still no crash
		match r4:
			case Result.Err( e ):
				return 6
			case Result.Ok( f ):
				pass
		return 0
''' ),
		( 'fallible_generator_or_return_error_releases_captured_parameter', '''
@union
class BoomError:
	Boom: None

class Box:
	v: i32
	def __init__( self, v: i32 ) -> None:
		self.v = v

def maybe_bad( i: usize, boom_at: usize ) -> Result[usize, BoomError]:
	if i == boom_at:
		return Result.Err( BoomError.Boom( None ))
	return Result.Ok( i )

def gen( b: Box, limit: usize, boom_at: usize ) -> Generator[usize, BoomError]:
	i: usize = 0
	while i < limit:
		v: usize = maybe_bad( i, boom_at ).or_return()
		yield i
		with compiler.wrap_arithmetic:
			i += 1

def make_and_partially_consume( b: Box ) -> None:
	g = gen( b, 5, 1 )
	first = g.__next__() # i=0, succeeds
	first.unwrap( 'unexpected error' )
	second = g.__next__() # i=1 == boom_at - or_return() fires, Err returned
	match second:
		case Result.Ok( x ):
			pass
		case Result.Err( e ):
			pass
	# g goes out of scope here, permanently done via or_return()'s own
	# error exit (not normal exhaustion) - its own captured Box parameter
	# must still be released via the ordinary, unmodified $$__destructor__
	# cascade, exactly like every other generator abandoned mid-iteration

def main() -> i32:
	with compiler.wrap_arithmetic:
		b = Box( v = 42 )
		if compiler.refcount( b ) != 1:
			return 1
		make_and_partially_consume( b )
		if compiler.refcount( b ) != 1:
			return 2
		return 0
''' ),
		# --- Phase 5 (roadmap Phase 5, the last roadmap item): RC-typed
		# locals crossing a yield. Lifts v1's own scalar-only restriction
		# on promoted locals (a not-yet-initialized RC field would make
		# the ordinary unconditional $$__destructor__ cascade decref
		# garbage) via a state/flag-gated destructor - each RC-typed
		# promoted local gets its own `__<stem>_live` companion field,
		# set once actually assigned, checked by the generator's own
		# custom-built destructor before decref-ing it. A yielded RC value
		# now returns straight through (no intermediate temp routing -
		# see 048af0f, "Fix double-incref/masked-decref when coercing a
		# value into a union type": the union-coercion path this compiles
		# down to, `return <yielded>` against __next__'s own elem_type|
		# None return type, already increfs correctly on its own, whether
		# `yielded` is a field read or a tracked local/parameter). Values
		# are read back correctly across MULTIPLE reassignment cycles of
		# the SAME promoted local (not just constructed once) - the
		# generator's own field is reassigned fresh each loop iteration,
		# each old value correctly released before the new one lands.
		( 'rc_typed_promoted_local_reassigned_each_iteration', '''
class Box:
	v: i32
	def __init__( self, v: i32 ) -> None:
		self.v = v

def make_boxes( count: usize ) -> Iterator[Box]:
	i: usize = 0
	while i < count:
		b: Box = Box( v = 100 )
		yield b
		with compiler.wrap_arithmetic:
			i += 1

def consume_fully( count: usize ) -> None:
	with compiler.wrap_arithmetic:
		n: usize = 0
		while True:
			match make_boxes( count ).__next__():
				case Box( x ):
					if x.v != 100:
						return
					n += 1
				case None:
					break

def gen_from_box( b: Box, count: usize ) -> Iterator[Box]:
	i: usize = 0
	while i < count:
		yield b
		with compiler.wrap_arithmetic:
			i += 1

def make_and_partially_consume( b: Box ) -> None:
	g = gen_from_box( b, 5 )
	match g.__next__():
		case Box( first ):
			pass
		case None:
			pass
	# g goes out of scope here, still mid-iteration (only 1 of 5 yields
	# consumed) - dropping it must decref the captured parameter b,
	# exactly like the existing (scalar-yielding) drop tests already
	# verify, now with an RC value flowing through the yield itself too

def main() -> i32:
	with compiler.wrap_arithmetic:
		b0 = Box( v = 1 )
		if compiler.refcount( b0 ) != 1:
			return 1
		b1 = Box( v = 2 )
		if compiler.refcount( b1 ) != 1:
			return 2
		make_and_partially_consume( b1 )
		if compiler.refcount( b0 ) != 1: # unaffected by an unrelated generator
			return 3
		if compiler.refcount( b1 ) != 1: # g's own captured parameter released
			return 4
		return 0
''' ),
		# 048af0f regression guard: a yielded RC value now returns straight
		# through with no intermediate temp routing (see the comment above
		# this section's own list) - `yield b` compiles to a bare `return
		# self.b` against __next__'s own elem_type|None return type,
		# exactly the shape union_coercion_rc_test.py's own
		# _UNION_COERCE_FIELD_READ proves increfs exactly once for a plain
		# (non-generator) field read. Mirrors that test's own before/after-
		# on-the-SOURCE methodology (never compiler.refcount() on the
		# union-typed result itself - see that test's own comment on why)
		# plus its del-then-still-valid check: if the return had silently
		# stopped increfing (the pre-048af0f bug), dropping g1 here would
		# free the captured parameter out from under got, and reading
		# got.v right after would be a real use-after-free, not just a
		# wrong number.
		( 'yielded_rc_value_increfs_exactly_once_caller_side', '''
class Box:
	v: i32
	def __init__( self, v: i32 ) -> None:
		self.v = v

def yield_param_directly( b: Box ) -> Iterator[Box]:
	yield b

def main() -> i32:
	with compiler.wrap_arithmetic:
		p = Box( v = 1 )
		g1 = yield_param_directly( p )
		before: usize = compiler.refcount( p ) # p itself + g1's own captured field
		r = g1.__next__()
		after: usize = compiler.refcount( p )
		if after != before + 1:
			return 1
		del g1
		match r:
			case Box( got ):
				if got.v != 1:
					return 2
			case None:
				return 3
		return 0
''' ),
		( 'generator_for_loop_over_rc_typed_list_element', '''
class Box:
	v: i32
	def __init__( self, v: i32 ) -> None:
		self.v = v

def double_all( xs: list[Box] ) -> Iterator[Box]:
	for x in xs:
		yield x

def consume_fully( xs: list[Box] ) -> None:
	with compiler.wrap_arithmetic:
		total: i32 = 0
		n: usize = 0
		for y in double_all( xs ):
			total += y.v
			n += 1
		if n != 3:
			return
		if total != 60: # 10+20+30
			return

def main() -> i32:
	with compiler.wrap_arithmetic:
		xs: list[Box] = list[Box]()
		xs.append( Box( v = 10 ) ).unwrap( 'append failed' )
		xs.append( Box( v = 20 ) ).unwrap( 'append failed' )
		xs.append( Box( v = 30 ) ).unwrap( 'append failed' )
		consume_fully( xs )
		if compiler.refcount( xs ) != 1:
			return 1
		return 0
''' ),
			# --- prerequisite for defer/errdefer support (PLAN_GENERATORS.md) -
			# a bare `return` inside a generator body must permanently end
			# iteration (self.__state pinned at the DONE sentinel), not just
			# return None once and leave __state wherever it was - a real,
			# previously-untested gap (_reject_generator_value_return only ever
			# rejected a VALUE return, nothing rewrote a bare one)
			( 'bare_return_nested_in_while_unit_loop_body_ends_iteration_permanently', '''
def gen( limit: i32 ) -> Iterator[i32]:
	x: i32 = 0
	while x < limit:
		if x == 2:
			return
		yield x
		with compiler.wrap_arithmetic:
			x += 1

def main() -> i32:
	with compiler.wrap_arithmetic:
		g = gen( 5 )
		a = g.__next__()
		if a is None:
			return 1
		b = g.__next__()
		if b is None:
			return 2
		c = g.__next__() # x becomes 2 here, hits the bare `return` before yielding again
		if c is not None:
			return 3
		d = g.__next__() # must stay permanently None, not resume mid-loop
		if d is not None:
			return 4
		e = g.__next__()
		if e is not None:
			return 5
		return 0
''' ),
			( 'bare_return_in_tail_after_yield_ends_iteration_permanently', '''
def gen( flag: bool ) -> Iterator[i32]:
	yield 1
	if flag:
		return
	yield 2

def main() -> i32:
	with compiler.wrap_arithmetic:
		g = gen( True )
		a = g.__next__()
		if a is None:
			return 1
		b = g.__next__() # the tail's `if flag: return` fires here
		if b is not None:
			return 2
		c = g.__next__() # must stay None - not fall through to the second yield
		if c is not None:
			return 3
		return 0
''' ),
			# --- defer/errdefer in generators (PLAN_GENERATORS.md) - Mechanism
			# 1 (normal exits): a promoted __defer_armed_N flag field is set
			# True in place of the `with defer:` statement, and every armed,
			# PLAIN `defer` site (never `errdefer` - that only ever fires via
			# mechanism 2's OrReturn.epilogue hook, not landed yet) replays
			# LIFO at the tail's own natural exhaustion, a bare-return exit, or
			# the destructor (abandonment) - real refcount-observable side
			# effects, same "before/after a helper call" style every other RC
			# test in this class already uses
			( 'defer_replays_exactly_once_at_natural_exhaustion', '''
class Box:
	v: i32
	def __init__( self, v: i32 ) -> None:
		self.v = v

def gen( b: Box, count: usize ) -> Iterator[usize]:
	i: usize = 0
	with defer:
		compiler.incref( b )
	while i < count:
		yield i
		with compiler.wrap_arithmetic:
			i += 1

def main() -> i32:
	with compiler.wrap_arithmetic:
		b = Box( v = 1 )
		g = gen( b, 2 )
		if compiler.refcount( b ) != 2: # caller + generator's own captured param
			return 1
		a = g.__next__()
		if a is None:
			return 2
		if compiler.refcount( b ) != 2: # defer must not have fired yet
			return 3
		c = g.__next__()
		if c is None:
			return 4
		d = g.__next__() # exhausts here - tail's own exit replays the armed defer
		if d is not None:
			return 5
		if compiler.refcount( b ) != 3: # defer fired exactly once
			return 6
		e = g.__next__() # already done - must not re-fire
		if e is not None:
			return 7
		if compiler.refcount( b ) != 3:
			return 8
		return 0
''' ),
			( 'defer_replays_exactly_once_at_bare_return_exit', '''
class Box:
	v: i32
	def __init__( self, v: i32 ) -> None:
		self.v = v

def gen( b: Box, limit: usize ) -> Iterator[usize]:
	i: usize = 0
	with defer:
		compiler.incref( b )
	while i < limit:
		if i == 1:
			return
		yield i
		with compiler.wrap_arithmetic:
			i += 1

def main() -> i32:
	with compiler.wrap_arithmetic:
		b = Box( v = 1 )
		g = gen( b, 5 )
		if compiler.refcount( b ) != 2:
			return 1
		a = g.__next__() # i=0, yields 0
		if a is None:
			return 2
		if compiler.refcount( b ) != 2: # defer must not have fired yet
			return 3
		c = g.__next__() # i becomes 1, hits the bare `return` inside the loop - defer fires
		if c is not None:
			return 4
		if compiler.refcount( b ) != 3:
			return 5
		return 0
''' ),
			( 'defer_replays_exactly_once_on_abandonment_via_destructor', '''
class Box:
	v: i32
	def __init__( self, v: i32 ) -> None:
		self.v = v

def gen( b: Box, count: usize ) -> Iterator[usize]:
	i: usize = 0
	with defer:
		compiler.incref( b )
	while i < count:
		yield i
		with compiler.wrap_arithmetic:
			i += 1

def make_and_partially_consume( b: Box ) -> None:
	g = gen( b, 5 )
	first = g.__next__() # only 1 of 5 iterations consumed
	# g goes out of scope here, still mid-iteration - abandonment must still
	# replay the armed defer, via the destructor, before its own ordinary
	# captured-parameter teardown

def main() -> i32:
	with compiler.wrap_arithmetic:
		b = Box( v = 1 )
		if compiler.refcount( b ) != 1:
			return 1
		make_and_partially_consume( b )
		if compiler.refcount( b ) != 2: # captured param released, defer's own incref remains
			return 2
		return 0
''' ),
			( 'two_defer_sites_replay_in_lifo_order', '''
class Box:
	v: i32
	def __init__( self, v: i32 ) -> None:
		self.v = v

def gen( b: Box, c: Box, count: usize ) -> Iterator[usize]:
	i: usize = 0
	with defer:
		compiler.incref( b )
	with defer:
		compiler.incref( c )
	while i < count:
		yield i
		with compiler.wrap_arithmetic:
			i += 1

def main() -> i32:
	with compiler.wrap_arithmetic:
		b = Box( v = 1 )
		c = Box( v = 2 )
		g = gen( b, c, 1 )
		a = g.__next__()
		if a is None:
			return 1
		d = g.__next__() # exhausts - both defers replay LIFO: c's own first, then b's
		if d is not None:
			return 2
		if compiler.refcount( b ) != 3: # captured param + b's own armed defer
			return 3
		if compiler.refcount( c ) != 3: # captured param + c's own armed defer
			return 4
		return 0
''' ),
			# --- Mechanism 2 (PLAN_GENERATORS.md's defer/errdefer phase) -
			# error exits (or_return()/checked-arithmetic failure): a real,
			# targeted lowering.py change (_build_generator_error_defer_replay,
			# hooked into _consume_checked_result's own OrReturn.epilogue
			# construction) - errdefer only EVER fires here, never at a normal
			# exit (Mechanism 1 explicitly skips errdefer sites)
			( 'errdefer_fires_on_error_exit_only', '''
class Box:
	v: i32
	def __init__( self, v: i32 ) -> None:
		self.v = v

@union
class BoomError:
	Boom: None

def maybe_bad( i: usize, boom_at: usize ) -> Result[usize, BoomError]:
	if i == boom_at:
		return Result.Err( BoomError.Boom( None ))
	return Result.Ok( i )

def gen( b: Box, limit: usize, boom_at: usize ) -> Generator[usize, BoomError]:
	with errdefer:
		compiler.incref( b )
	i: usize = 0
	while i < limit:
		v: usize = maybe_bad( i, boom_at ).or_return()
		yield i
		with compiler.wrap_arithmetic:
			i += 1

def main() -> i32:
	with compiler.wrap_arithmetic:
		b = Box( v = 1 )
		g = gen( b, 5, 2 )
		if compiler.refcount( b ) != 2:
			return 1
		r0 = g.__next__() # i=0 succeeds
		match r0:
			case Result.Err( e ):
				return 2
			case Result.Ok( a ):
				pass
		if compiler.refcount( b ) != 2: # errdefer must not have fired yet
			return 3
		r1 = g.__next__() # i=1 succeeds
		match r1:
			case Result.Err( e ):
				return 4
			case Result.Ok( a ):
				pass
		r2 = g.__next__() # i becomes 2 == boom_at - or_return() fires, errdefer replays
		match r2:
			case Result.Err( e ):
				pass
			case Result.Ok( a ):
				return 5
		if compiler.refcount( b ) != 3: # errdefer fired exactly once
			return 6
		r3 = g.__next__() # permanently done - Ok(None), errdefer must not re-fire
		match r3:
			case Result.Err( e ):
				return 7
			case Result.Ok( a ):
				pass
		if compiler.refcount( b ) != 3:
			return 9
		return 0
''' ),
			( 'defer_and_errdefer_both_fire_on_same_error_exit', '''
class Box:
	v: i32
	def __init__( self, v: i32 ) -> None:
		self.v = v

@union
class BoomError:
	Boom: None

def maybe_bad( i: usize, boom_at: usize ) -> Result[usize, BoomError]:
	if i == boom_at:
		return Result.Err( BoomError.Boom( None ))
	return Result.Ok( i )

def gen( b: Box, c: Box, limit: usize, boom_at: usize ) -> Generator[usize, BoomError]:
	with defer:
		compiler.incref( b )
	with errdefer:
		compiler.incref( c )
	i: usize = 0
	while i < limit:
		v: usize = maybe_bad( i, boom_at ).or_return()
		yield i
		with compiler.wrap_arithmetic:
			i += 1

def main() -> i32:
	with compiler.wrap_arithmetic:
		b = Box( v = 1 )
		c = Box( v = 2 )
		g = gen( b, c, 5, 1 )
		r0 = g.__next__() # i=0 succeeds
		match r0:
			case Result.Err( e ):
				return 1
			case Result.Ok( a ):
				pass
		r1 = g.__next__() # i becomes 1 == boom_at - or_return() fires, BOTH replay
		match r1:
			case Result.Err( e ):
				pass
			case Result.Ok( a ):
				return 2
		if compiler.refcount( b ) != 3: # plain defer also fired on this SAME error exit
			return 3
		if compiler.refcount( c ) != 3: # errdefer fired here too
			return 4
		return 0
''' ),
			( 'defer_does_not_replay_again_when_generator_later_dropped', '''
class Box:
	v: i32
	def __init__( self, v: i32 ) -> None:
		self.v = v

def gen( b: Box, count: usize ) -> Iterator[usize]:
	i: usize = 0
	with defer:
		compiler.incref( b )
	while i < count:
		yield i
		with compiler.wrap_arithmetic:
			i += 1

def drain_fully( b: Box, count: usize ) -> None:
	g = gen( b, count )
	i: usize = 0
	while i < count:
		v = g.__next__()
		with compiler.wrap_arithmetic:
			i += 1
	last = g.__next__() # natural exhaustion - tail replay fires the defer, unsets its own flag
	# g goes out of scope HERE - $$__destructor__ must see the flag already
	# unset and must NOT replay the same defer body a second time

def main() -> i32:
	with compiler.wrap_arithmetic:
		b = Box( v = 1 )
		if compiler.refcount( b ) != 1:
			return 1
		drain_fully( b, 2 )
		if compiler.refcount( b ) != 2: # captured param released, defer fired EXACTLY once (not twice)
			return 2
		return 0
''' ),
		])

	def test_for_loop_over_neither_shape_is_rejected( self ) -> None:
		# PLAN_GENERATORS.md Phase 1 - a for loop over something with
		# neither __len__/__getitem__ NOR __next__ must be a clear
		# compile error, not a silently wrong state machine
		self._run( '''
class NotIterable:
	pass

def gen( x: NotIterable ) -> Iterator[i32]:
	for y in x:
		yield 1

def main() -> None:
	g = gen( NotIterable() )
''' )
		self.assertTrue( self.discovery.errors.errors )
		self.assertIn( '__len__', str( self.discovery.errors.errors[0] ))

	def test_for_loop_over_bad_next_shape_is_rejected( self ) -> None:
		# a __next__() that returns something other than T|None - real, not
		# a generator's own (compiler-synthesized __next__ always has the
		# right shape) - must be a clear compile error, not a miscompile.
		# Not generator-specific: any user class implementing __next__ by
		# hand hits the same check.
		self._run( '''
class NotReallyAnIterator:
	def __next__( self ) -> i32:
		return 1

def main() -> None:
	it = NotReallyAnIterator()
	for x in it:
		pass
''' )
		self.assertTrue( self.discovery.errors.errors )
		self.assertIn( 'T|None', str( self.discovery.errors.errors[0] ))

	def test_yield_nested_in_if_is_rejected( self ) -> None:
		# Phase 2 only recognizes a bare top-level yield or a single-yield
		# top-level while loop as valid units - a yield nested one level
		# deeper (inside an if inside the loop) must be a clear compile
		# error, not a silently wrong state machine
		self._run( '''
def gen( flag: bool ) -> Iterator[i32]:
	while True:
		if flag:
			yield 1

def main() -> None:
	g = gen( True )
''' )
		self.assertTrue( self.discovery.errors.errors )
		self.assertIn( 'PLAN_GENERATORS.md', str( self.discovery.errors.errors[0] ))

	def test_while_loop_with_two_yields_is_rejected( self ) -> None:
		self._run( '''
def gen() -> Iterator[i32]:
	while True:
		yield 1
		yield 2

def main() -> None:
	g = gen()
''' )
		self.assertTrue( self.discovery.errors.errors )
		self.assertIn( 'exactly one yield', str( self.discovery.errors.errors[0] ))

	def test_break_inside_yielding_while_loop_is_rejected( self ) -> None:
		self._run( '''
def gen() -> Iterator[i32]:
	while True:
		yield 1
		break

def main() -> None:
	g = gen()
''' )
		self.assertTrue( self.discovery.errors.errors )
		self.assertIn( 'break/continue', str( self.discovery.errors.errors[0] ))

	def test_if_elif_chain_with_yield_is_rejected( self ) -> None:
		# Phase 2a only recognizes a single if/else - an elif chain
		# generalizes the same branch-stable-resume idea but multiplies the
		# state/testing surface, deliberately deferred (PLAN_GENERATORS.md)
		self._run( '''
def gen( flag: i32 ) -> Iterator[i32]:
	if flag == 0:
		yield 1
	elif flag == 1:
		yield 2
	else:
		yield 3

def main() -> None:
	g = gen( 0 )
''' )
		self.assertTrue( self.discovery.errors.errors )
		self.assertIn( 'elif', str( self.discovery.errors.errors[0] ))

	def test_if_else_with_two_yields_in_one_branch_is_rejected( self ) -> None:
		self._run( '''
def gen( flag: bool ) -> Iterator[i32]:
	if flag:
		yield 1
		yield 2
	else:
		yield 3

def main() -> None:
	g = gen( True )
''' )
		self.assertTrue( self.discovery.errors.errors )
		self.assertIn( 'at most one yield per branch', str( self.discovery.errors.errors[0] ))

	def test_defer_inside_while_unit_loop_body_is_rejected( self ) -> None:
		# PLAN_GENERATORS.md's defer/errdefer phase - only a direct top-
		# level statement (preamble/tail) is supported for now, same start-
		# narrow posture as break/continue inside a yield-containing loop
		self._run( '''
def gen( count: usize ) -> Iterator[usize]:
	i: usize = 0
	while i < count:
		with defer:
			i = i
		yield i
		with compiler.wrap_arithmetic:
			i += 1

def main() -> None:
	g = gen( 3 )
''' )
		self.assertTrue( self.discovery.errors.errors )
		self.assertIn( 'direct top-level statement', str( self.discovery.errors.errors[0] ))

	def test_errdefer_inside_if_unit_branch_is_rejected( self ) -> None:
		self._run( '''
def gen( flag: bool ) -> Generator[i32,str]:
	if flag:
		with errdefer:
			pass
		yield 1
	else:
		yield 2

def main() -> None:
	g = gen( True )
''' )
		self.assertTrue( self.discovery.errors.errors )
		self.assertIn( 'direct top-level statement', str( self.discovery.errors.errors[0] ))

	def test_return_inside_generator_defer_body_is_rejected( self ) -> None:
		# a return inside a generator's defer/errdefer body would jump out
		# of $$__next__ early, skipping any later-armed site's own replay
		# and the flag-unset every replay guard relies on to avoid firing
		# twice - mirrors the identical, pre-existing rejection ordinary
		# (non-generator) defer/errdefer bodies already get
		# (lowering.py's _stmt_Return, gated on _in_deferred_body), which
		# never fires here since a generator's own defer body never routes
		# through _register_defer_block at all
		self._run( '''
def gen( count: usize ) -> Iterator[usize]:
	i: usize = 0
	with defer:
		return
	while i < count:
		yield i
		with compiler.wrap_arithmetic:
			i += 1

def main() -> None:
	g = gen( 3 )
''' )
		self.assertTrue( self.discovery.errors.errors )
		self.assertIn( 'return is not allowed inside a defer/errdefer body', str( self.discovery.errors.errors[0] ))

	def test_return_inside_generator_errdefer_body_nested_in_if_is_rejected( self ) -> None:
		# same check, errdefer instead of defer, and the return nested one
		# level deeper (inside an if) - confirms the whole-body walk
		# catches a nested return too, not just a direct top-level one
		self._run( '''
def gen( flag: bool ) -> Generator[i32,str]:
	with errdefer:
		if flag:
			return
	yield 1

def main() -> None:
	g = gen( True )
''' )
		self.assertTrue( self.discovery.errors.errors )
		self.assertIn( 'return is not allowed inside a defer/errdefer body', str( self.discovery.errors.errors[0] ))

	def test_generic_generator_referencing_own_type_param_in_body_is_rejected( self ) -> None:
		# Phase 3's recommended interim scope (PLAN_GENERATORS.md) - a
		# generic generator body that references its own type param
		# outside a parameter/return annotation (here, a nested generic
		# call using it) is rejected for now - _build_generator_next_
		# function's synthesized __next__ doesn't inherit the type-param
		# substitution monomorphized_function recorded only on the
		# generator function itself, confirmed by a real repro that
		# otherwise fails downstream with a confusing "name 'T' is not
		# defined" instead of this clear, upfront rejection
		self._run( '''
def identity[T]( v: T ) -> T:
	return v

def gen[T]( x: T ) -> Iterator[T]:
	y: T = identity( x )
	yield y

def main() -> None:
	g = gen[i32]( 5 )
''' )
		self.assertTrue( self.discovery.errors.errors )
		self.assertIn( 'PLAN_GENERATORS.md', str( self.discovery.errors.errors[0] ))

	def test_or_return_inside_infallible_iterator_is_rejected( self ) -> None:
		# Phase 4 (roadmap Phase 4) - or_return() stays rejected inside a
		# plain Iterator[T] (infallible) generator, exactly as before this
		# phase - unchanged, since it's purely a consequence of __next__'s
		# own declared return type (elem_type|None, not Result-shaped),
		# same _require_result_return check every other non-Result-
		# returning function already hits. Generator[T,E] is what lifts
		# this - see test_programs_compile_and_run's own fallible_generator_
		# ... test cases.
		self._run( '''
@union
class BoomError:
	Boom: None

def maybe_bad( flag: bool ) -> Result[i32, BoomError]:
	if flag:
		return Result.Err( BoomError.Boom( None ))
	return Result.Ok( 1 )

def gen( flag: bool ) -> Iterator[i32]:
	v: i32 = maybe_bad( flag ).or_return()
	yield v

def main() -> None:
	g = gen( True )
''' )
		self.assertTrue( self.discovery.errors.errors )
		self.assertIn( 'or_return()', str( self.discovery.errors.errors[0] ))


class OverloadWithDefaultParameterRealCompileTests( test_support.RealCompileMixin, CompilerTestCase ):
	''' regression test for a real, confirmed bug: Result[T,E].unwrap_or()
	called with NO argument (relying on its own `default: T|None = None`
	fallback) resolved to the WRONG return type - the enclosing (generic,
	monomorphized) unwrap_or() body itself failed to compile with
	"function returns builtins.str, not builtins.str|intrinsics.NoneType"
	(seen for real via lib/posix/time.py's `readlink(...).unwrap_or()` on
	a branch with posix syscalls; reproduced here without any OS
	dependency). Root cause: lowering.py's _lower_call unconditionally
	narrowed an Overload group's resolved return type to whatever STUB
	happened to be bound_to the winning plain implementation, regardless
	of whether THIS call's own arguments actually matched the stub's
	narrower signature - unwrap_or()'s `default: T` stub is bound_to the
	plain `default: T|None = None` impl, but a zero-argument call only
	ever matches the impl's own broader signature, never the stub's.
	Fixed via overload_resolution.stub_covers_call, which re-checks the
	call's real argument types against the stub before narrowing.

	Two further gaps surfaced once the return type itself was fixed, both
	fixed alongside it: (1) type_resolver.py's _type_of_expr didn't handle
	a Call resolving to an Overload group at all (only a plain Function),
	so a local assigned from such a call never got its type tracked,
	silently disabling _rewrite_tagged_union_truthiness's `if x:` rewrite
	for it further down the same function body; (2) the Overload branch of
	_lower_call never filled in defaults for parameters the call site
	omitted (unlike the plain-Function call path), so a zero-argument
	unwrap_or() reached real C emission with no 'default' entry in its own
	Call instruction's kwargs at all - a bare KeyError in emitter_c.py's
	_emit_call_args.

	T=str (not e.g. i32) deliberately: matches the real-world repro
	exactly, and forces the non-bool leaf of _rewrite_tagged_union_
	truthiness's rewrite (str.__bool__(), newly added alongside this fix -
	str had no truthiness dunder at all before, so this path was never
	reachable for any RC leaf type, only the bool-leaf shortcut). '''

	def setUp( self ) -> None:
		self.discovery = Discovery( import_builtins = True )
		self.compiler = Compiler( self.discovery )

	@unittest.skipUnless( test_support.HAS_CC, 'no C compiler (clang/gcc/msvc) found - skipping' )
	def test_programs_compile_and_run( self ) -> None:
		self.assert_programs_run([
			( 'result_unwrap_or_no_argument_return_type_and_truthiness', '''
def make( ok: bool, s: str ) -> Result[str,OverflowError]:
	if ok:
		return Result.Ok( s )
	return Result.Err( OverflowError() )

def main() -> i32:
	# Ok("hello") -> unwrap_or() with no fallback -> "hello", truthy
	a: str|None = make( True, "hello" ).unwrap_or()
	if a:
		pass
	else:
		return 1
	# Ok("") -> unwrap_or() -> "" (not None, but empty) -> falsy
	b: str|None = make( True, "" ).unwrap_or()
	if b:
		return 2
	# Err(...) -> unwrap_or() -> None (the impl's own default) -> falsy
	c: str|None = make( False, "hello" ).unwrap_or()
	if c:
		return 3
	return 0
''' ),
		] )

if __name__ == '__main__':
	unittest.main()
