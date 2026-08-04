# stdlib imports:
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest

# local imports:
import ir
import emitter_c
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
		self.assertEqual( emitter_c.c_type( _scalar( 'i128' )), '__int128' )
		self.assertEqual( emitter_c.c_type( _scalar( 'u128' )), 'unsigned __int128' )
		self.assertEqual( emitter_c.c_type( _scalar( 'isize' )), 'intptr_t' )
		self.assertEqual( emitter_c.c_type( _scalar( 'usize' )), 'uintptr_t' )
		self.assertEqual( emitter_c.c_type( _scalar( 'bool', 'builtins.bool' )), 'bool' )

	def test_nonetype_and_noreturn_are_void( self ) -> None:
		self.assertEqual( emitter_c.c_type( _scalar( 'NoneType' )), 'void' )
		self.assertEqual( emitter_c.c_type( _scalar( 'NoReturn' )), 'void' )
		self.assertEqual( emitter_c.c_type( None ), 'void' )

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
# synthesis/OrReturn), so it's kept within what Phase 1 actually covers
_RESULT_FIXTURE = '\n'.join([
	'@cstruct',
	'class OverflowError: pass',
	'',
	'@cunion',
	'class ResultPayload[T,E]:',
	'	ok: T',
	'	err: E',
	'',
	'@cstruct',
	'class Result[T,E]:',
	'	_payload: ResultPayload[T,E]',
	'	_tag: u8',
	'',
	'	@staticmethod',
	'	def Ok( val: T ) -> Result[T,E]:',
	'		return Result.__allocate__( _payload = ResultPayload( ok = val ), _tag = 0 )',
	'',
	'	@staticmethod',
	'	def Err( err: E ) -> Result[T,E]:',
	'		return Result.__allocate__( _payload = ResultPayload( err = err ), _tag = 1 )',
	'',
	'	def is_ok( self ) -> bool:',
	'		return self._tag == 0',
	'',
	'	def is_err( self ) -> bool:',
	'		return self._tag == 1',
])

class SpecializationSynthesisTests( CompilerTestCase ):
	def test_result_specialization_is_a_real_compiler_cstructs_entry( self ) -> None:
		# a concrete generic class specialization (Result[i32,
		# OverflowError]) is a real compile unit by the time it reaches
		# this module - lowering.py's Lowering.monomorphize_class (wired
		# through compiler.py's own _enqueue/_lower, NOT emitter_c.py -
		# stage 3 does no discovery of its own) already substituted its
		# .attributes and gave it a concrete qualname, landing it directly
		# in compiler.cstructs alongside the (still-abstract, correctly
		# excluded from emission) bare Result.
		self._run( _RESULT_FIXTURE + '\n' + '\n'.join([
			'def main() -> Result[i32,OverflowError]:',
			'	with compiler.wrap_arithmetic:',
			'		x: i32 = 1',
			'	return Result.Ok( x )',
		]))
		self.assertEqual( self.discovery.errors.errors, [] )
		names = [ cls.qualname for cls in self.compiler.cstructs ]
		self.assertIn( '__main__.Result[intrinsics.i32,__main__.OverflowError]', names )
		spec_cls = next( cls for cls in self.compiler.cstructs if cls.qualname == '__main__.Result[intrinsics.i32,__main__.OverflowError]' )
		self.assertIsNone( spec_cls.type_params ) # concrete now, not generic
		src = emitter_c.emit_cstruct( spec_cls )
		self.assertIn( 'struct', src )
		self.assertIn( '_tag;', src )
		self.assertIn( '_payload;', src )

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
		self.assertIn( '__builtin_add_overflow', src )
		self.assertIn( '_tag == 1', src )

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
		self.assertIn( '= &main$x;', src ) # AddrOf
		self.assertIn( '[main$i] = main$seven;', src ) # SetItem
		self.assertIn( '(main$p)[main$i];', src ) # GetItem

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
		main_lf = next( lf for lf in self.compiler.functions if lf.function.qualname == 'main' )
		src = emitter_c.emit_function( main_lf )
		self.assertIn( '.tag = 0', src ) # Foo.Bar's ordinal
		self.assertIn( '.v_Bar = ', src )
		self.assertIn( ').tag;', src ) # the match's case Foo.Bar(...) tag read, compared against the ordinal separately
		self.assertIn( '== (0)', src )

CLANG = shutil.which( 'clang' ) or r'C:\Program Files\LLVM\bin\clang.exe'

class _ClangCompileMixin:
	def _assert_compiles( self, c_source: str ) -> None:
		with tempfile.TemporaryDirectory() as tmp:
			src_path = Path( tmp ) / 'generated.c'
			obj_path = Path( tmp ) / 'generated.obj'
			src_path.write_text( c_source, encoding = 'utf-8' )
			result = subprocess.run(
				[ CLANG, '-std=c11', '-Wall', '-Wextra', '-c', str( src_path ), '-o', str( obj_path ) ],
				capture_output = True, text = True,
			)
			self.assertEqual( result.returncode, 0, f'clang failed:\nstdout: {result.stdout}\nstderr: {result.stderr}\n\n--- generated.c ---\n{c_source}' )

@unittest.skipUnless( Path( CLANG ).exists(), 'clang.exe not found - skipping real-compile verification' )
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
		self._tmpdir = tempfile.TemporaryDirectory()
		self.addCleanup( self._tmpdir.cleanup )
		tmp_path = Path( self._tmpdir.name )
		( tmp_path / 'sys.py' ).write_text( _SYS_FIXTURE, encoding = 'utf-8' )
		self.discovery = Discovery( paths = [ tmp_path ], import_builtins = False )
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
	def test_del_method_is_called_from_synthesized_destructor( self ) -> None:
		self._run( _OWNER_FIXTURE + '\n' + '\n'.join([
			'def main() -> None:',
			'	x: i32 = 0',
			'	p: Ptr[None] = compiler.addrof( x )',
			'	o: Owner = Owner.make( p )',
			'	return',
		]))
		self.assertEqual( self.discovery.errors.errors, [] )
		owner_cls = next( cls for cls in self.compiler.rcclasses if cls.qualname == '__main__.Owner' )
		destructor_src = emitter_c.emit_rcclass_destructor( owner_cls )
		self.assertIn( '__main__$Owner$__del__( self );', destructor_src )
		self.assertIn( 'sys$free( ( void* )self );', destructor_src )
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
		box_cls = next( cls for cls in self.compiler.rcclasses if cls.qualname == '__main__.Box' )
		destructor_src = emitter_c.emit_rcclass_destructor( box_cls )
		self.assertNotIn( '__del__', destructor_src ) # Box declares none
		self.assertIn( 'release_object( &(self->inner)->$header, __main__$Foo$$__destructor__ );', destructor_src )
		self.assertIn( 'sys$free( ( void* )self );', destructor_src )

	def test_taggedunion_field_cascades_a_tag_gated_decref( self ) -> None:
		# a TaggedUnion-typed field with an RC-leaf member (MaybeFoo.Some)
		# needs the same tag-gated shape cfg.py builds at the IR level for
		# LOCAL bindings, reimplemented in raw C here since no IR backs a
		# synthesized destructor body - the non-RC member (Nothing) needs
		# no branch at all
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
		box_cls = next( cls for cls in self.compiler.rcclasses if cls.qualname == '__main__.Box' )
		destructor_src = emitter_c.emit_rcclass_destructor( box_cls )
		self.assertIn( 'uint8_t __tag = (self->maybe).tag;', destructor_src )
		self.assertIn( 'if ( __tag == 0 ) {', destructor_src )
		self.assertIn( 'release_object( &((self->maybe).data.v_Some)->$header, __main__$Foo$$__destructor__ );', destructor_src )
		self.assertNotIn( 'v_Nothing', destructor_src ) # the non-RC member needs no branch at all

	def test_nested_cstruct_field_cascades_decref_into_its_own_fields( self ) -> None:
		# a by-value CStruct field is always fully live (unlike a union, no
		# discriminant needed) - safe to walk its own fields unconditionally,
		# recursively, looking for further RC leaves
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
		box_cls = next( cls for cls in self.compiler.rcclasses if cls.qualname == '__main__.Box' )
		destructor_src = emitter_c.emit_rcclass_destructor( box_cls )
		self.assertIn( 'release_object( &((self->w).inner)->$header, __main__$Foo$$__destructor__ );', destructor_src )

@unittest.skipUnless( Path( CLANG ).exists(), 'clang.exe not found - skipping real-compile verification' )
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

@unittest.skipUnless( Path( CLANG ).exists(), 'clang.exe not found - skipping real-compile verification' )
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

class BuiltinsStrTestCase( CompilerTestCase ):
	def setUp( self ) -> None:
		self._tmpdir = tempfile.TemporaryDirectory()
		self.addCleanup( self._tmpdir.cleanup )
		tmp_path = Path( self._tmpdir.name )
		( tmp_path / 'builtins.py' ).write_text( _BUILTINS_STR_FIXTURE, encoding = 'utf-8' )
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
		self.assertIn( '104, 101, 108, 108, 111, 0', src ) # 'hello' + NUL, per str's own __byte_size convention
		self.assertIn( '.__byte_size = 6', src )
		main_lf = next( lf for lf in self.compiler.functions if lf.function.qualname == 'main' )
		main_src = emitter_c.emit_function( main_lf )
		self.assertRegex( main_src, r'__main__\$take_str\( &__literal_[0-9a-f]+ \);' )
		# str is never dynamically constructed here (only baked as an
		# immortal literal) - no destructor should be emitted for it at all
		# (release_object always skips an immortal object's destructor call,
		# so the function would just be unreachable dead code that still
		# has to compile - simplest to not emit it, see
		# _rcclass_was_constructed)
		self.assertNotIn( '$__destructor__', src )

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

@unittest.skipUnless( Path( CLANG ).exists(), 'clang.exe not found - skipping real-compile verification' )
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

@unittest.skipUnless( Path( CLANG ).exists(), 'clang.exe not found - skipping real-compile verification' )
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

@unittest.skipUnless( Path( CLANG ).exists(), 'clang.exe not found - skipping real-compile verification' )
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

if __name__ == '__main__':
	unittest.main()
