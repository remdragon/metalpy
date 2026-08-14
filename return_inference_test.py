# Real-compile-and-run behavioral tests for return-only generic type-
# parameter inference (PLAN_RETURN_INFERENCE.md) - lowering_test.py's
# ReturnOnlyTypeParamInferenceTests already cover the IR/inference shape in
# isolation; this file confirms the emitted C is actually correct end to
# end: the discovered R becomes a real, concrete return type/prototype (not
# a leaked bare TypeVar, which would simply fail to compile as C), the
# program runs and returns the right values, and calling with the same
# concrete T from multiple sites never produces more than one real C
# function for that instantiation. Mirrors inline_test.py's own
# compile+link+run harness (itself mirroring int_test.py's).

# stdlib imports:
from pathlib import Path
import subprocess
import tempfile
import unittest

# local imports:
import emitter_c
import linker_c
import test_support
from compiler import Compiler
from discovery import Discovery

_CC = linker_c.detect_cc()


@unittest.skipUnless( _CC is not None, 'no C compiler (clang/gcc/msvc) found - skipping real-compile return-inference tests' )
class ReturnOnlyInferenceBehaviorTests( unittest.TestCase ):
	def _compile( self, code: str ) -> tuple[Compiler, str]:
		discovery = Discovery( import_builtins = True )
		compiler = Compiler( discovery )
		compiler.import_code( code, Path( '__main__.py' ), scope = None )
		compiler.run()
		self.assertEqual( discovery.errors.errors, [], f'compile errors:\n' + '\n'.join( str(e) for e in discovery.errors.errors ))
		# always the real C runtime (no_crt=False), not the 'c' not in
		# compiler.extern_libs heuristic int_test.py/inline_test.py use -
		# their own real-builtins-heavy programs (list/str construction)
		# always end up needing sys.alloc, which naturally pulls in a real
		# extern_libs entry; this file's own minimal @cstruct-only programs
		# (no allocation at all) don't, so that heuristic would pick the
		# no_crt/raw-syscall startup path here, which needs kernel32 linked
		# explicitly for ExitProcess/SetConsoleOutputCP - unrelated to what
		# this file is actually testing, so sidestepped entirely
		c_source = emitter_c.emit_c( compiler, no_crt = False )
		return compiler, c_source

	def _run_program( self, code: str ) -> tuple[subprocess.CompletedProcess, Compiler, str]:
		compiler, c_source = self._compile( code )
		no_crt = False
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
				flag = f'{lib}.lib' if _CC.name == 'cl' else f'-l{lib}'
				ldflags = ldflags + f' {flag}' if ldflags else flag

			link_result = _CC.link( exe_path, [ obj_path ], ldflags = ldflags, no_crt = no_crt )
			self.assertEqual( link_result.returncode, 0, f'{_CC.name} link failed:\n{link_result.stdout}' )

			result = subprocess.run( [ str( exe_path ) ], capture_output = True )
			return result, compiler, c_source

	def test_make_infers_correct_concrete_return_type_and_runs( self ) -> None:
		# two DIFFERENT concrete T's, each with its own differently-typed
		# factory method - R must be independently discovered per T
		code = '\n'.join([
			'@cstruct',
			'class IntProducer:',
			'	def produce( self ) -> i32:',
			'		return 42',
			'',
			'@cstruct',
			'class BoolProducer:',
			'	def produce( self ) -> bool:',
			'		return True',
			'',
			'def make[T,R]( t: T ) -> R:',
			'	return t.produce()',
			'',
			'def main() -> i32:',
			'	ip: IntProducer = IntProducer()',
			'	bp: BoolProducer = BoolProducer()',
			'	n: i32 = make( ip )',
			'	b: bool = make( bp )',
			'	if n != 42:',
			'		return 1',
			'	if not b:',
			'		return 2',
			'	return 0',
		])
		result, compiler, c_source = self._run_program( code )
		self.assertEqual( result.returncode, 0, f'stdout: {result.stdout}\nstderr: {result.stderr}{test_support.c_source_on_failure( c_source )}' )
		# both instantiations must be real, distinctly-typed compiled units
		qualnames = { lf.function.qualname for lf in compiler.functions }
		self.assertIn( '__main__.make[__main__.IntProducer,intrinsics.i32]', qualnames )
		self.assertIn( '__main__.make[__main__.BoolProducer,intrinsics.bool]', qualnames )

	def test_two_call_sites_same_binding_compile_to_one_c_function( self ) -> None:
		# calling make() twice with the SAME concrete T must not produce two
		# separate C function definitions for the same instantiation - a
		# real duplicate-symbol link error if it did
		code = '\n'.join([
			'@cstruct',
			'class IntProducer:',
			'	def produce( self ) -> i32:',
			'		return 7',
			'',
			'def make[T,R]( t: T ) -> R:',
			'	return t.produce()',
			'',
			'def main() -> i32:',
			'	p: IntProducer = IntProducer()',
			'	a: i32 = make( p )',
			'	b: i32 = make( p )',
			'	if a != 7 or b != 7:',
			'		return 1',
			'	return 0',
		])
		result, compiler, c_source = self._run_program( code )
		self.assertEqual( result.returncode, 0, f'stdout: {result.stdout}\nstderr: {result.stderr}{test_support.c_source_on_failure( c_source )}' )
		make_functions = [ lf for lf in compiler.functions if lf.function.qualname.startswith( '__main__.make[' ) ]
		self.assertEqual( len( make_functions ), 1, f'expected exactly one compiled make[...] instantiation, got: {[lf.function.qualname for lf in make_functions]}' )

if __name__ == '__main__':
	unittest.main()
