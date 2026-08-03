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
	CStruct, Function, Parameter, RCClass, Scalar, Specialization, TaggedUnion, Variable,
)

def _scalar( stem: str, qualname: str|None = None ) -> Scalar:
	return Scalar( stem = stem, qualname = qualname or f'intrinsics.{stem}', file = None, line = None )

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
		ptr_cls = Scalar( stem = 'Ptr', qualname = 'intrinsics.Ptr', file = None, line = None )
		const_ptr_cls = Scalar( stem = 'ConstPtr', qualname = 'intrinsics.ConstPtr', file = None, line = None )
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
# _RESULT_FIXTURE exactly (self-contained snippet, not a real lib/ import -
# established convention for CompilerTestCase-style tests, see
# compiler_test.py)
_RESULT_FIXTURE = '\n'.join([
	'class bool: pass',
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
	def test_result_specialization_gets_a_real_struct_body( self ) -> None:
		# compiler.py's own _enqueue() never schedules a ClassLike
		# Specialization as its own compile unit (see emitter_c.py's
		# _collect_specializations docstring) - Result[i32,OverflowError]
		# never appears in compiler.cstructs, only bare Result does. This
		# confirms the emitter finds and synthesizes it anyway.
		self._run( _RESULT_FIXTURE + '\n' + '\n'.join([
			'def main() -> Result[i32,OverflowError]:',
			'	with compiler.wrap_arithmetic:',
			'		x: i32 = 1',
			'	return Result.Ok( x )',
		]))
		self.assertEqual( self.discovery.errors.errors, [] )
		specs = emitter_c._collect_specializations( self.compiler )
		names = [ s.qualname for s in specs ]
		self.assertIn( '__main__.Result[intrinsics.i32,__main__.OverflowError]', names )
		spec = next( s for s in specs if s.qualname == '__main__.Result[intrinsics.i32,__main__.OverflowError]' )
		src = emitter_c.emit_specialization( spec, self.discovery )
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

CLANG = shutil.which( 'clang' ) or r'C:\Program Files\LLVM\bin\clang.exe'

@unittest.skipUnless( Path( CLANG ).exists(), 'clang.exe not found - skipping real-compile verification' )
class RealCompileTests( CompilerTestCase ):
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

	@unittest.expectedFailure
	def test_default_check_mode_arithmetic_compiles( self ) -> None:
		# Phase 1 milestone (b): default Check-mode arithmetic, proving
		# AddCheck + the synthesized Result[i32,OverflowError] struct +
		# OrReturn all compile clean together.
		#
		# KNOWN GAP (found via this exact test, real, pre-existing,
		# discovered by this emitter work - not something to patch around
		# here): Result.Ok(...)/.Err(...) are METHODS on a generic CStruct
		# (Result[T,E]). lowering.py monomorphizes generic FREE functions
		# on call (_monomorphized_function/lower_function_specialization -
		# sys.alloc[u8] etc.) but never does the equivalent for a generic
		# CLASS's own methods - Result.Ok's Function object is scheduled
		# and lowered with its parameters/return_type still literally
		# holding Result's own unbound TypeVars (T, E), which have no C
		# representation at all. Existing tests never caught this because
		# they only assert on IR *shape* (duck-typed, doesn't care whether
		# a type is concrete) - this is the first thing to require REAL
		# concrete types out of a generic method's own signature. Needs a
		# real lowering.py/discovery.py fix (generic-method monomorphization,
		# mirroring the existing generic-function path) before this can
		# pass - out of scope for the emitter itself to work around.
		self._run( _RESULT_FIXTURE + '\n' + '\n'.join([
			'def main() -> Result[i32,OverflowError]:',
			'	x: i32 = 1',
			'	y: i32 = x + 1',
			'	return Result.Ok( y )',
		]))
		self._assert_compiles( emitter_c.emit_c( self.compiler ))

if __name__ == '__main__':
	unittest.main()
