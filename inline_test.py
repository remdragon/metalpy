# Real-compile-and-run behavioral tests for @inline (PLAN_INLINE.md).
# Mirrors int_test.py's own _ClangCompileMixin-less pattern (a bare
# skipUnless-gated real compile+link+run) - lowering_test.py's InlineTests
# already cover the IR shape in isolation with fake classes; this file
# confirms the SAME thing end to end against the real lib/builtins/__init__.py
# len[T] (now @inline - see lib/builtins/__init__.py's own comment), for a
# couple of genuinely different concrete T, and confirms no separate len
# function is ever compiled at all - not just that main's own instructions
# look right in isolation.

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


@unittest.skipUnless( _CC is not None, 'no C compiler (clang/gcc/msvc) found - skipping real-compile inline tests' )
class InlineLenBehaviorTests( unittest.TestCase ):
	def _compile( self, code: str ) -> tuple[Compiler, str]:
		''' compiles `code` (a full MetalPy source, needs its own def main()
		-> i32) against the real builtins - returns the Compiler (post-run,
		so .functions/.errors are final) and the generated C source, without
		yet invoking a real C compiler (some tests only need the former). '''
		discovery = Discovery( import_builtins = True )
		compiler = Compiler( discovery )
		compiler.import_code( code, Path( '__main__.py' ), scope = None )
		compiler.run()
		self.assertEqual( discovery.errors.errors, [], f'compile errors:\n' + '\n'.join( str(e) for e in discovery.errors.errors ))
		no_crt = 'c' not in compiler.extern_libs
		c_source = emitter_c.emit_c( compiler, no_crt = no_crt )
		return compiler, c_source

	def _run_program( self, code: str ) -> tuple[subprocess.CompletedProcess, Compiler, str]:
		compiler, c_source = self._compile( code )
		no_crt = 'c' not in compiler.extern_libs
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
				flag = linker_c.resolve_lib_ldflag( _CC, lib, compiler.extern_libs[lib], no_crt = no_crt )
				ldflags = ldflags + f' {flag}' if ldflags else flag

			link_result = _CC.link( exe_path, [ obj_path ], ldflags = ldflags, no_crt = no_crt )
			self.assertEqual( link_result.returncode, 0, f'{_CC.name} link failed:\n{link_result.stdout}' )

			result = subprocess.run( [ str( exe_path ) ], capture_output = True )
			return result, compiler, c_source

	def _assert_no_len_function_was_compiled( self, compiler: Compiler ) -> None:
		# len[T] is @inline - a real, separate compiled function for it
		# (any specialization of builtins.len) must never appear among
		# compiler.functions, regardless of how many different T's it was
		# called with in the program under test. __len__ methods (list[i32].
		# __len__, str.__len__, ...) are a DIFFERENT, expected thing - only
		# len[T] itself (the forwarder) must be absent
		qualnames = { lf.function.qualname for lf in compiler.functions }
		self.assertFalse(
			any( q.startswith( 'builtins.len[' ) or q == 'builtins.len' for q in qualnames ),
			f'a real len[T] function was compiled, @inline should have spliced it instead: {sorted(qualnames)}',
		)

	def test_len_on_list_and_str_returns_correct_values( self ) -> None:
		code = '\n'.join([
			'def main() -> i32:',
			'	x: list[i32] = list[i32]()',
			"	x.append( 1 ).unwrap( 'append failed' )",
			"	x.append( 2 ).unwrap( 'append failed' )",
			"	x.append( 3 ).unwrap( 'append failed' )",
			'	if len( x ) != 3:',
			'		return 1',
			'',
			"	s: str = 'hello'",
			'	if len( s ) != 5:',
			'		return 2',
			'',
			'	empty: list[i32] = list[i32]()',
			'	if len( empty ) != 0:',
			'		return 3',
			'',
			'	return 0',
		])
		result, compiler, _c_source = self._run_program( code )
		self.assertEqual( result.returncode, 0, f'stdout: {result.stdout}\nstderr: {result.stderr}' )
		self._assert_no_len_function_was_compiled( compiler )

	def test_len_generic_instantiation_never_compiled_as_a_real_function( self ) -> None:
		# same assertion as above, isolated from the runtime-behavior check -
		# calling len() against TWO different concrete T (list[i32] AND str)
		# in the same program must not produce two (or any) real len[T]
		# functions either
		code = '\n'.join([
			'def main() -> i32:',
			'	x: list[i32] = list[i32]()',
			"	x.append( 1 ).unwrap( 'append failed' )",
			"	n: usize = len( x )",
			"	s: str = 'hi'",
			'	m: usize = len( s )',
			'	if n != 1 or m != 2:',
			'		return 1',
			'	return 0',
		])
		compiler, _c_source = self._compile( code )
		self._assert_no_len_function_was_compiled( compiler )

if __name__ == '__main__':
	unittest.main()
