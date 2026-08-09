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
from compiler import Compiler
from discovery import Discovery
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

_RESULT_FIXTURE_WITH_OR_RETURN = '\n'.join([
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
	'',
	'\tdef or_return( self ) -> T:',
	'\t\tif self.is_err():',
	'\t\t\tcompiler.early_return( self.data.v_Err )',
	'\t\treturn self.data.v_Ok',
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
		# Call to one - Result.or_return's own declared body is a spec of
		# the intended behavior, not literally compilable (see Lowering.
		# _lower_or_return's own comment) - this is the exact regression
		# the eager-substitution work risked: target.cls became a
		# Specialization for a concrete receiver, breaking the `target.cls
		# is Result` identity check _lower_call used to route here
		self._run( _RESULT_FIXTURE_WITH_OR_RETURN + '\n' + '\n'.join([
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

	def test_known_gap_union_receiver_dispatch_does_not_check_per_leaf_parameter_types( self ) -> None:
		# documents a pre-existing gap, NOT fixed as part of this plan: a
		# union mixing two DIFFERENT concrete instantiations of the same
		# generic class (Box[i32]|Box[u32]) with a same-named method taking
		# a generic-typed argument - the per-leaf consistency check only
		# compares return-type identity and parameter COUNT, never
		# per-position parameter TYPE, so this compiles with no error, and
		# the SAME lowered argument operand (typed i32 here) is silently
		# reused for BOTH leaves' Call, including the Box[u32] one that
		# actually expects a u32. Before Stage 1/2 this couldn't happen at
		# all - every leaf's method stayed abstract/bare-T, so there was
		# nothing to disagree about
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
		self.assertEqual( self.discovery.errors.errors, [] ) # no error today - this is the gap
		main_lf = next( lf for lf in self.compiler.functions if lf.function.qualname == 'main' )
		calls = [ i for i in main_lf.instructions if isinstance( i, ir.Call ) and i.target.stem == 'set' ]
		self.assertEqual( len( calls ), 2 )
		# same operand passed to both, including the Box[u32] leaf that
		# actually declares x: u32 - the mismatch nothing catches
		self.assertIs( calls[0].args[0], calls[1].args[0] )

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

class _ClangCompileMixin:
	def _assert_compiles( self, c_source: str ) -> None:
		with tempfile.TemporaryDirectory() as tmp:
			src_path = Path( tmp ) / 'generated.c'
			obj_path = Path( tmp ) / 'generated.o'
			src_path.write_text( c_source, encoding = 'utf-8' )
			result = _CC.compile( src_path, obj_path )
			self.assertEqual( result.returncode, 0, f'{_CC.name} failed:\nstdout: {result.stdout}\nstderr: {result.stderr}\n\n--- generated.c ---\n{c_source}' )

@unittest.skipUnless( _CC is not None, 'no C compiler (clang/gcc/msvc) found - skipping real-compile verification' )
class RealCompileTests( _ClangCompileMixin, CompilerTestCase ):
	def test_empty_main_compiles( self ) -> None:
		self._run( '''
def main() -> None:
	return
''' )
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
		self.assertIn( 'sys$free( (void*)(self) )', destructor_src )
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
		self.assertIn( 'release_object', destructor_src )
		self.assertIn( '__main__$Foo$$__destructor__', destructor_src )
		self.assertIn( 'sys$free( (void*)(self) )', destructor_src )

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
		self.assertIn( 'release_object', destructor_src )
		self.assertIn( '__main__$Foo$$__destructor__', destructor_src )

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
		self.assertIn( '__main__$g_foo = t0;', src )

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
class EmitGlobalRCClassRealCompileTests( _ClangCompileMixin, RCClassTestCase ):
	def test_non_trivial_global_compiles( self ) -> None:
		# Phase 7 milestone: both global-initializer shapes compile clean -
		# this is the RCClass-construction shape (mirrors lib/sys.py's own
		# real stdout: _Stdout = _Stdout()), the trivial-constant shape is
		# covered by EmitGlobalRealCompileTests above. This is the FINAL
		# milestone of the whole C-emitter plan.
		self._run( _FOO_FIXTURE + '\n' + '\n'.join([
			'g_foo: Foo = Foo.make( 1 )',
			'',
			'def main() -> None:',
			'	x: Foo = g_foo',
			'	return',
		]))
		self._assert_compiles( emitter_c.emit_c( self.compiler ))

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
			if _CC is not None and _CC.name == 'cl':
				flags.append( f'{lib}.lib' )
			else:
				flags.append( f'-l{lib}' )
		return ' '.join( flags )

	def _assert_compiles_and_runs( self, c_source: str, expected_exit: int = 0 ) -> None:
		with tempfile.TemporaryDirectory() as tmp:
			src_path = Path( tmp ) / 'generated.c'
			obj_path = Path( tmp ) / 'generated.o'
			exe_path = Path( tmp ) / 'test_exe'
			src_path.write_text( c_source, encoding = 'utf-8' )
			cc_result = _CC.compile( src_path, obj_path )
			self.assertEqual( cc_result.returncode, 0,
				f'{_CC.name} compile failed:\nstdout: {cc_result.stdout}\nstderr: {cc_result.stderr}\n\n--- generated.c ---\n{c_source}' )
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
			if _CC is not None and _CC.name == 'cl':
				flags.append( f'{lib}.lib' )
			else:
				flags.append( f'-l{lib}' )
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


if __name__ == '__main__':
	unittest.main()
