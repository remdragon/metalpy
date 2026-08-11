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

class UnionAsUnconstructedResultErrorTypeTests( CompilerTestCase ):
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

	@unittest.skipUnless( _CC is not None, 'no C compiler (clang/gcc/msvc) found - skipping' )
	def test_unconstructed_union_error_type_compiles_and_runs( self ) -> None:
		self._run( '''
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
''' )
		self.assertEqual( self.discovery.errors.errors, [] )
		self._assert_compiles_and_runs( emitter_c.emit_c( self.compiler ))

class AugAssignRealCompileTests( CompilerTestCase ):
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

	@unittest.skipUnless( _CC is not None, 'no C compiler (clang/gcc/msvc) found - skipping' )
	def test_attribute_target_rc_value_replaced_in_a_loop( self ) -> None:
		# f.s += 'a' 200 times - exercises the dunder (str.__add__) dispatch
		# path plus cfg.attr_replace's decref of the OLD str each iteration;
		# a missing/wrong decref here either leaks or double-frees, and 200
		# iterations is enough for ASan/heap-corruption-on-double-free to
		# reliably surface if it were broken (confirmed against a
		# deliberately-reintroduced bug before writing this test)
		self._run( '''
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
''' )
		self.assertEqual( self.discovery.errors.errors, [] )
		self._assert_compiles_and_runs( emitter_c.emit_c( self.compiler ))

	@unittest.skipUnless( _CC is not None, 'no C compiler (clang/gcc/msvc) found - skipping' )
	def test_subscript_target_raw_pointer_fallback( self ) -> None:
		# p[0] += 5 through the flat GetItem/SetItem fallback (no
		# __getitem__/__setitem__) - confirms the read-modify-write actually
		# lands in the right memory, not just that it compiles
		self._run( '''
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
''' )
		self.assertEqual( self.discovery.errors.errors, [] )
		self._assert_compiles_and_runs( emitter_c.emit_c( self.compiler ))

	@unittest.skipUnless( _CC is not None, 'no C compiler (clang/gcc/msvc) found - skipping' )
	def test_subscript_target_with_real_getitem_setitem_methods( self ) -> None:
		# d[1] += 5 through a real __getitem__/__setitem__ pair (dict[K,V]) -
		# both fallible, auto-consumed exactly like an ordinary d[1] read/
		# write already is, and both driven off the SAME index operand
		# (lowered once). main() itself can't return Result (the C entry
		# point's signature is fixed - see emitter_c._is_entry_point), so
		# the dict logic lives in a helper that does, mirroring how every
		# other real-run test needing a fallible operation at top level
		# already structures this (see
		# UnionAsUnconstructedResultErrorTypeTests above)
		self._run( '''
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
''' )
		self.assertEqual( self.discovery.errors.errors, [] )
		self._assert_compiles_and_runs( emitter_c.emit_c( self.compiler ))

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

	def test_generic_method_returning_t_monomorphized_to_none_compiles( self ) -> None:
		# regression test: a generic method whose body does `return self.
		# <T-typed-field>` (Result[None,E].unwrap()'s own `return self.data
		# .v_Ok` is the real-world case that found this) still had a real
		# ir.Return(value=<some operand>) once T monomorphizes to NoneType -
		# _emit_instruction's own ir.Return handling only checked whether
		# THAT operand was Python None (i.e. "no expression"), not whether
		# the function's own C return type was void, so it emitted
		# `return t0;` from a function _function_prototype had separately
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
		self.assertIn( '(p)[1] = t1;', src ) # real write-back, not a copy-mutate-discard
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


class ListGenericTests( CompilerTestCase ):
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
				f'exited {run_result.returncode}, expected {expected_exit} (stderr: {run_result.stderr})' )

	@unittest.skipUnless( _CC is not None, 'no C compiler (clang/gcc/msvc) found - skipping' )
	def test_list_i32_construct_append_getitem_del( self ) -> None:
		# a non-RC element type: list[i32]() construction/destruction alone
		# (x never used past declaration) already exercises RawList's own
		# alloc/free and list[T].__del__'s decref-skip loop; append/
		# __getitem__ round-trip three values through the buffer, by
		# POSITION (an index IS a position now - no separate stable ID)
		self._run( '''
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
''' )
		self.assertEqual( self.discovery.errors.errors, [] )
		self._assert_compiles_and_runs( emitter_c.emit_c( self.compiler ))

	@unittest.skipUnless( _CC is not None, 'no C compiler (clang/gcc/msvc) found - skipping' )
	def test_list_rc_element_getitem_unwrap_chained_on_bare_receiver( self ) -> None:
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
		self._run( '''
def main() -> i32:
	x: list[int] = list[int]()
	x.append( int( 5 )).unwrap( 'append failed' )
	got: int = x.__getitem__( 0 ).unwrap( 'getitem failed' )
	if got != int( 5 ):
		return 1
	return 0
''' )
		self.assertEqual( self.discovery.errors.errors, [] )
		self._assert_compiles_and_runs( emitter_c.emit_c( self.compiler ))

	@unittest.skipUnless( _CC is not None, 'no C compiler (clang/gcc/msvc) found - skipping' )
	def test_list_i32_grows_past_initial_capacity( self ) -> None:
		# initial_capacity defaults to 8 - 20 appends forces RawList._grow()
		# at least once, and every value must still read back correctly
		# afterward (the copy during growth must preserve contents)
		self._run( '''
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
''' )
		self.assertEqual( self.discovery.errors.errors, [] )
		self._assert_compiles_and_runs( emitter_c.emit_c( self.compiler ))

	@unittest.skipUnless( _CC is not None, 'no C compiler (clang/gcc/msvc) found - skipping' )
	def test_erase_at_preserves_positional_order( self ) -> None:
		# the whole point of this container vs. FastList[T]: erase_at
		# shifts everything after the removed slot left by one (memmove),
		# it does not swap the last element into the gap. Append
		# 10,20,30,40,50, erase_at(2) (the value 30) - must read back
		# 10,20,40,50, never 10,20,50,40 (that shape would mean this
		# regressed to FastList's swap-and-pop behavior)
		self._run( '''
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
''' )
		self.assertEqual( self.discovery.errors.errors, [] )
		self._assert_compiles_and_runs( emitter_c.emit_c( self.compiler ))

	@unittest.skipUnless( _CC is not None, 'no C compiler (clang/gcc/msvc) found - skipping' )
	def test_insert_shifts_tail_right_and_preserves_order( self ) -> None:
		# the mirror image of erase_at above: insert(1, 99) into
		# [10,20,30] must produce [10,99,20,30], not overwrite or corrupt
		# anything - everything at/after the insertion point shifts right
		self._run( '''
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
''' )
		self.assertEqual( self.discovery.errors.errors, [] )
		self._assert_compiles_and_runs( emitter_c.emit_c( self.compiler ))

	@unittest.skipUnless( _CC is not None, 'no C compiler (clang/gcc/msvc) found - skipping' )
	def test_insert_past_end_clamps_to_append( self ) -> None:
		# matches Python's own list.insert - an out-of-range index doesn't
		# error, it just appends
		self._run( '''
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
''' )
		self.assertEqual( self.discovery.errors.errors, [] )
		self._assert_compiles_and_runs( emitter_c.emit_c( self.compiler ))

	@unittest.skipUnless( _CC is not None, 'no C compiler (clang/gcc/msvc) found - skipping' )
	def test_setitem_via_assignment_syntax_overwrites_in_place( self ) -> None:
		# exercises x[i] = v as real assignment syntax (not
		# .__setitem__(...) called directly) - this only actually reaches
		# list[T].__setitem__ because of lowering.py's own dispatch fix
		# (obj[i] = v used to always emit a raw SetItem, ignoring any real
		# __setitem__ the type declared)
		self._run( '''
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
''' )
		self.assertEqual( self.discovery.errors.errors, [] )
		self._assert_compiles_and_runs( emitter_c.emit_c( self.compiler ))

	@unittest.skipUnless( _CC is not None, 'no C compiler (clang/gcc/msvc) found - skipping' )
	def test_list_str_construct_append_getitem_del( self ) -> None:
		# an RC element type - a list[T] slot holds str's own HANDLE
		# (pointer-width), not its struct body (see list.__init__'s own
		# comment); __del__ must decref every stored element without
		# reading struct-body-sized memory out of a pointer-sized slot
		self._run( '''
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
''' )
		self.assertEqual( self.discovery.errors.errors, [] )
		self._assert_compiles_and_runs( emitter_c.emit_c( self.compiler ))

	@unittest.skipUnless( _CC is not None, 'no C compiler (clang/gcc/msvc) found - skipping' )
	def test_list_str_grows_past_initial_capacity( self ) -> None:
		self._run( '''
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
''' )
		self.assertEqual( self.discovery.errors.errors, [] )
		self._assert_compiles_and_runs( emitter_c.emit_c( self.compiler ))

	@unittest.skipUnless( _CC is not None, 'no C compiler (clang/gcc/msvc) found - skipping' )
	def test_list_as_class_field_constructs_and_destructs( self ) -> None:
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
		self._run( '''
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
''' )
		self.assertEqual( self.discovery.errors.errors, [] )
		self._assert_compiles_and_runs( emitter_c.emit_c( self.compiler ))

	@unittest.skipUnless( _CC is not None, 'no C compiler (clang/gcc/msvc) found - skipping' )
	def test_list_local_var_of_multi_field_rcclass( self ) -> None:
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
		self._run( '''
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
''' )
		self.assertEqual( self.discovery.errors.errors, [] )
		self._assert_compiles_and_runs( emitter_c.emit_c( self.compiler ))

	@unittest.skipUnless( _CC is not None, 'no C compiler (clang/gcc/msvc) found - skipping' )
	def test_fastlist_local_var_of_multi_field_rcclass( self ) -> None:
		# same as test_list_local_var_of_multi_field_rcclass above, but for
		# FastList[T] - a separate implementation (lib/builtins/__fastlist.py)
		# with the identical `val: T = <read>; compiler.decref(val)` shape in
		# its own __del__, so it reproduced the identical double-Decref bug
		# for the identical reason (see the other test's own updated comment
		# - cfg.py's manually_decreffed(), not anything list/FastList-specific)
		self._run( '''
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
''' )
		self.assertEqual( self.discovery.errors.errors, [] )
		self._assert_compiles_and_runs( emitter_c.emit_c( self.compiler ))

	@unittest.skipUnless( _CC is not None, 'no C compiler (clang/gcc/msvc) found - skipping' )
	def test_list_str_erase_at_preserves_order_and_refcounts( self ) -> None:
		# RC-element coverage for erase_at's ordering guarantee - 'b' is
		# decreffed on removal, 'a' and 'c' must survive (and read back
		# correctly) in the shifted positions. If decref/incref bookkeeping
		# were wrong here, this would double-free or leak at __del__ time
		# (list[T].__del__ decrefs every remaining slot on the way out) -
		# not something this test can observe directly without ASAN, but a
		# wrong refcount is exactly the kind of thing that turns into a
		# crash on a real run
		self._run( '''
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
''' )
		self.assertEqual( self.discovery.errors.errors, [] )
		self._assert_compiles_and_runs( emitter_c.emit_c( self.compiler ))


class FastListGenericTests( CompilerTestCase ):
	''' FastList[T] (lib/builtins/__fastlist.py) end-to-end - the ORIGINAL
	StableIndexVector port: O(1) swap-and-pop erase, stable IDs that
	survive other inserts/deletes, but positional order is NOT preserved
	across an erase. Split out from list[T] (which now has real
	Python-list/array semantics instead - see ListGenericTests above) once
	that distinction became load-bearing enough to need two containers. '''

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
				f'exited {run_result.returncode}, expected {expected_exit} (stderr: {run_result.stderr})' )

	@unittest.skipUnless( _CC is not None, 'no C compiler (clang/gcc/msvc) found - skipping' )
	def test_fastlist_i32_construct_append_getitem_del( self ) -> None:
		self._run( '''
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
''' )
		self.assertEqual( self.discovery.errors.errors, [] )
		self._assert_compiles_and_runs( emitter_c.emit_c( self.compiler ))

	@unittest.skipUnless( _CC is not None, 'no C compiler (clang/gcc/msvc) found - skipping' )
	def test_erase_does_not_preserve_positional_order( self ) -> None:
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
		self._run( '''
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
''' )
		self.assertEqual( self.discovery.errors.errors, [] )
		self._assert_compiles_and_runs( emitter_c.emit_c( self.compiler ))

	@unittest.skipUnless( _CC is not None, 'no C compiler (clang/gcc/msvc) found - skipping' )
	def test_erase_preserves_stable_id_identity_despite_position_swap( self ) -> None:
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
		self._run( '''
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
''' )
		self.assertEqual( self.discovery.errors.errors, [] )
		self._assert_compiles_and_runs( emitter_c.emit_c( self.compiler ))

	@unittest.skipUnless( _CC is not None, 'no C compiler (clang/gcc/msvc) found - skipping' )
	def test_append_after_erase_reuses_freed_id_without_aliasing( self ) -> None:
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
		self._run( '''
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
''' )
		self.assertEqual( self.discovery.errors.errors, [] )
		self._assert_compiles_and_runs( emitter_c.emit_c( self.compiler ))


class ChrOrdTests( CompilerTestCase ):
	''' chr()/ord() (lib/builtins/__init__.py) - built on str's own private
	UTF-8 encode/decode helpers (the same ones upper()/lower()/case-
	folding already use), not separate logic. '''
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

	@unittest.skipUnless( _CC is not None, 'no C compiler (clang/gcc/msvc) found - skipping' )
	def test_ascii_round_trip( self ) -> None:
		self._run( '''
def main() -> i32:
	if chr( 65 ) != "A":
		return 1
	if ord( "A" ) != 65:
		return 2
	if ord( "0" ) != 48:
		return 3
	return 0
''' )
		self.assertEqual( self.discovery.errors.errors, [] )
		self._assert_compiles_and_runs( emitter_c.emit_c( self.compiler ))

	@unittest.skipUnless( _CC is not None, 'no C compiler (clang/gcc/msvc) found - skipping' )
	def test_multibyte_round_trip( self ) -> None:
		# U+00E9 (e-acute, 2 UTF-8 bytes) and U+1F600 (grinning face emoji,
		# 4 UTF-8 bytes) - exercises _utf8_encoded_len/_encode_utf8_at/
		# _decode_utf8_at's own 2-byte and 4-byte branches, not just ASCII
		self._run( '''
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
''' )
		self.assertEqual( self.discovery.errors.errors, [] )
		self._assert_compiles_and_runs( emitter_c.emit_c( self.compiler ))

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


class MatchValuePatternRealCompileTests( CompilerTestCase ):
	''' real compile+run coverage for type_resolver.py's _match_pattern
	ast.MatchValue handling - `case Color.Red:`/`case 5:` desugaring to a
	plain == Compare. Unlike type_resolver_test.py's own MatchValue tests
	(which only check the desugared AST shape), these confirm the
	generated code actually branches correctly. '''
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

	@unittest.skipUnless( _CC is not None, 'no C compiler (clang/gcc/msvc) found - skipping' )
	def test_match_cenum_member_value_patterns( self ) -> None:
		self._run( '''
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
''' )
		self.assertEqual( self.discovery.errors.errors, [] )
		self._assert_compiles_and_runs( emitter_c.emit_c( self.compiler ))

	@unittest.skipUnless( _CC is not None, 'no C compiler (clang/gcc/msvc) found - skipping' )
	def test_match_plain_literal_value_patterns( self ) -> None:
		self._run( '''
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
''' )
		self.assertEqual( self.discovery.errors.errors, [] )
		self._assert_compiles_and_runs( emitter_c.emit_c( self.compiler ))


class StrFindIndexSplitTests( CompilerTestCase ):
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
				f'exited {run_result.returncode}, expected {expected_exit} (stderr: {run_result.stderr})' )

	@unittest.skipUnless( _CC is not None, 'no C compiler (clang/gcc/msvc) found - skipping' )
	def test_find_and_index( self ) -> None:
		self._run( '''
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
''' )
		self.assertEqual( self.discovery.errors.errors, [] )
		self._assert_compiles_and_runs( emitter_c.emit_c( self.compiler ))

	@unittest.skipUnless( _CC is not None, 'no C compiler (clang/gcc/msvc) found - skipping' )
	def test_split_guid_like_string( self ) -> None:
		# the exact motivating case from PLAN_SUBCLASSING_VTABLES_COM.md's
		# own blocked-on note: GUID's constructor parsing a hyphenated hex
		# string via str.split('-')
		self._run( '''
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
''' )
		self.assertEqual( self.discovery.errors.errors, [] )
		self._assert_compiles_and_runs( emitter_c.emit_c( self.compiler ))

	@unittest.skipUnless( _CC is not None, 'no C compiler (clang/gcc/msvc) found - skipping' )
	def test_split_edge_cases( self ) -> None:
		self._run( '''
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
''' )
		self.assertEqual( self.discovery.errors.errors, [] )
		self._assert_compiles_and_runs( emitter_c.emit_c( self.compiler ))


class EarlyReturnFromLoopWithLiveRCLocalTests( CompilerTestCase ):
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
				f'exited {run_result.returncode}, expected {expected_exit} (stderr: {run_result.stderr})' )

	@unittest.skipUnless( _CC is not None, 'no C compiler (clang/gcc/msvc) found - skipping' )
	def test_early_return_past_a_loop_confined_rc_local_never_taken( self ) -> None:
		# the early return is never actually reached at runtime (v is
		# always 'item') - this is purely a "does it even compile, and
		# does the untaken branch not corrupt the normal exit" check
		self._run( '''
def main() -> i32:
	j: usize = 0
	with compiler.panic_arithmetic( 'overflow' ):
		while j < 20:
			v: str = 'item'
			if v != 'item':
				return 2
			j += 1
	return 0
''' )
		self.assertEqual( self.discovery.errors.errors, [] )
		self._assert_compiles_and_runs( emitter_c.emit_c( self.compiler ))

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

	@unittest.skipUnless( _CC is not None, 'no C compiler (clang/gcc/msvc) found - skipping' )
	def test_break_past_a_loop_confined_rc_local_still_works( self ) -> None:
		# break (not return) reaching the SAME loop-confined entry - a
		# different code path (unwind_to(), not current_epilogue_label())
		# that this fix must not have disturbed
		self._run( '''
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
''' )
		self.assertEqual( self.discovery.errors.errors, [] )
		self._assert_compiles_and_runs( emitter_c.emit_c( self.compiler ))


class GUIDTests( CompilerTestCase ):
	''' lib/guid.py's GUID type - PLAN_SUBCLASSING_VTABLES_COM.md's Phase 3
	(COM specifics). Needs import_builtins=True (str.split, list[str]). '''

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
				f'exited {run_result.returncode}, expected {expected_exit} (stderr: {run_result.stderr})' )

	@unittest.skipUnless( _CC is not None, 'no C compiler (clang/gcc/msvc) found - skipping' )
	def test_from_str_parses_each_field_correctly( self ) -> None:
		self._run( '''
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
''' )
		self.assertEqual( self.discovery.errors.errors, [] )
		self._assert_compiles_and_runs( emitter_c.emit_c( self.compiler ))

	@unittest.skipUnless( _CC is not None, 'no C compiler (clang/gcc/msvc) found - skipping' )
	def test_eq_ne_compare_by_value( self ) -> None:
		self._run( '''
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
''' )
		self.assertEqual( self.discovery.errors.errors, [] )
		self._assert_compiles_and_runs( emitter_c.emit_c( self.compiler ))


class DictTests( CompilerTestCase ):
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
				f'exited {run_result.returncode}, expected {expected_exit} (stderr: {run_result.stderr})' )

	@unittest.skipUnless( _CC is not None, 'no C compiler (clang/gcc/msvc) found - skipping' )
	def test_str_key_insert_and_lookup( self ) -> None:
		self._run( '''
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
''' )
		self.assertEqual( self.discovery.errors.errors, [] )
		self._assert_compiles_and_runs( emitter_c.emit_c( self.compiler ))

	@unittest.skipUnless( _CC is not None, 'no C compiler (clang/gcc/msvc) found - skipping' )
	def test_overwrite_existing_key_replaces_value( self ) -> None:
		self._run( '''
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
''' )
		self.assertEqual( self.discovery.errors.errors, [] )
		self._assert_compiles_and_runs( emitter_c.emit_c( self.compiler ))

	@unittest.skipUnless( _CC is not None, 'no C compiler (clang/gcc/msvc) found - skipping' )
	def test_missing_key_returns_key_error( self ) -> None:
		self._run( '''
def main() -> i32:
	d: dict[str, i32] = dict[str, i32]()
	d[ 'a' ] = 1
	r: Result[i32,KeyError] = d.__getitem__( 'nope' )
	if r.is_ok():
		return 1
	return 0
''' )
		self.assertEqual( self.discovery.errors.errors, [] )
		self._assert_compiles_and_runs( emitter_c.emit_c( self.compiler ))

	@unittest.skipUnless( _CC is not None, 'no C compiler (clang/gcc/msvc) found - skipping' )
	def test_non_rc_key_i32_with_rc_value_str( self ) -> None:
		# the other K/V combination - a plain value-typed key (byte-hashed
		# via _fnv1a_hash, no __hash__ method needed) paired with an RC
		# value, the mirror image of str-keyed dict[str,i32] above
		self._run( '''
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
''' )
		self.assertEqual( self.discovery.errors.errors, [] )
		self._assert_compiles_and_runs( emitter_c.emit_c( self.compiler ))

	@unittest.skipUnless( _CC is not None, 'no C compiler (clang/gcc/msvc) found - skipping' )
	def test_many_entries_forces_growth_and_stays_correct( self ) -> None:
		# 50 distinct keys forces list[T]'s own growth (both __entries and
		# __indices) and exercises RawDict's binary search over a real
		# range, not just a handful of entries
		self._run( '''
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
''' )
		self.assertEqual( self.discovery.errors.errors, [] )
		self._assert_compiles_and_runs( emitter_c.emit_c( self.compiler ))

	@unittest.skipUnless( _CC is not None, 'no C compiler (clang/gcc/msvc) found - skipping' )
	def test_rc_key_and_rc_value_destruction_does_not_crash( self ) -> None:
		# str keys AND str values together, insert/overwrite/destroy - a
		# proxy for correct incref/decref bookkeeping: wrong refcounting
		# here would double-free or leak, and a double-free would crash
		# the process (nonzero/abnormal exit), not just misbehave quietly
		self._run( '''
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
''' )
		self.assertEqual( self.discovery.errors.errors, [] )
		self._assert_compiles_and_runs( emitter_c.emit_c( self.compiler ))


class CallableTests( CompilerTestCase ):
	''' Callable[[Args],Ret]/Ptr[Callable[...]] end-to-end - see
	PLAN_CALLABLE.md: a bare function reference used as a value (never
	compiled anywhere before this), stored/passed as a real C function
	pointer, and called indirectly through it. Mirrors ListGenericTests'
	own import_builtins=True + real compile-and-run convention. '''

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
				f'exited {run_result.returncode}, expected {expected_exit} (stderr: {run_result.stderr})' )

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

	@unittest.skipUnless( _CC is not None, 'no C compiler (clang/gcc/msvc) found - skipping' )
	def test_staticmethod_reference_called_indirectly( self ) -> None:
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
		self._run( '''
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
''' )
		self.assertEqual( self.discovery.errors.errors, [] )
		self._assert_compiles_and_runs( emitter_c.emit_c( self.compiler ))

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


class ComTests( CompilerTestCase ):
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
				f'exited {run_result.returncode}, expected {expected_exit} (stderr: {run_result.stderr})' )

	def test_succeeded_failed_helpers( self ) -> None:
		self._run( '''
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
''' )
		self.assertEqual( self.discovery.errors.errors, [] )
		self._assert_compiles_and_runs( emitter_c.emit_c( self.compiler ))

	@unittest.skipUnless( _CC is not None, 'no C compiler (clang/gcc/msvc) found - skipping' )
	def test_iunknown_subclass_construct_dispatch_queryinterface_addref_release( self ) -> None:
		# IFoo(IUnknown) adds get_value - the exact COM pattern that
		# motivated the per-level-Vtbl-types revision (IUnknown's own 3
		# slots first, then IFoo's own new one, all in IFoo's own
		# FooImpl-shared Vtbl type). Hand-written QueryInterface/AddRef/
		# Release (no compiler synthesis, per the plan's own "hand-rolling
		# first" decision) - QueryInterface writes a real pointer through
		# its Ptr[Ptr[None]] out-param and calls AddRef itself, matching
		# real COM QueryInterface semantics.
		self._run( '''
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
''' )
		self.assertEqual( self.discovery.errors.errors, [] )
		self._assert_compiles_and_runs( emitter_c.emit_c( self.compiler ))

	@unittest.skipUnless( sys.platform == 'win32', 'real Windows COM interop - windows-only' )
	@unittest.skipUnless( _CC is not None, 'no C compiler (clang/gcc/msvc) found - skipping' )
	def test_real_windows_com_service_shelllink_getclassid( self ) -> None:
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
		self._run( '''
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
''' )
		self.assertEqual( self.discovery.errors.errors, [] )
		self._assert_compiles_and_runs( emitter_c.emit_c( self.compiler ))


if __name__ == '__main__':
	unittest.main()
