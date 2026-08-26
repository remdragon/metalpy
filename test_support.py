# Shared support for the real-compile-and-run tests (emitter_c_test.py etc.).
#
# Historically every test class that compiles MetalPy to a native executable
# carried its own copy-pasted _extern_ldflags/_assert_compiles_and_runs and
# built ONE executable per test METHOD. Building + linking a real binary is the
# slow part of the suite, and almost all of each emitted program is the shared
# builtins base (a trivial program is already ~27 KB of C), so N tiny programs
# re-pay that whole base N times.
#
# RealCompileMixin provides both:
#   * _assert_compiles_and_runs() - the original one-program-per-call path,
#     deduplicated here so the ~40 classes that used it stop copy-pasting it.
#   * assert_programs_run() - combines several standalone MetalPy programs into
#     ONE compiled executable (each sub-program is given its own symbol
#     namespace so their top-level names can't collide, then a synthesized
#     main() dispatches to each and encodes "which sub-test failed" in the exit
#     code). This lets a whole test class build and run a single executable.
#
# The namespacing is a plain AST transform: every top-level def/class/global in
# a sub-program (including its own main) is renamed with a per-case prefix, and
# every reference to those names is rewritten to match. This was validated to
# be behavior-preserving across the entire existing snippet corpus (each
# transformed-in-isolation program still compiles and exits with its original
# code). Programs that depend on PROCESS-GLOBAL one-time initialization (e.g. a
# case-folding table installed once per process) must NOT be merged - run those
# through _assert_compiles_and_runs individually instead.

# stdlib imports:
import ast
import os
from pathlib import Path
import re
import subprocess
import tempfile

# local imports:
import emitter_c
import linker_c
from compiler import Compiler
from discovery import Discovery

_CC = linker_c.detect_cc()


def c_source_on_failure( c_source: str ) -> str:
	''' The generated C, to be appended to a compile/link failure message ONLY
	when METALPY_TEST_DUMP_C is set - otherwise ''. Dumping the whole translation
	unit was useful while bringing the emitter up, but in normal runs it just
	buries the actual compiler error(s) in thousands of lines of noise. Set
	METALPY_TEST_DUMP_C=1 to bring it back when you need to inspect the C. '''
	return f'\n\n--- generated.c ---\n{c_source}' if os.environ.get( 'METALPY_TEST_DUMP_C' ) else ''


# A real (library, symbol) triple that resolves on the HOST toolchain, for the
# has_symbol / has_library tests that need a KNOWN-AVAILABLE symbol. Windows
# links a kernel32 export; every other platform links a libc export. Three
# distinct symbols are provided so tests that must not share a has_symbol disk
# cache entry can each take their own. (Before this, these tests hard-coded
# kernel32/GetLastError and so failed on Linux, where kernel32 does not exist -
# the feature was correct, the fixture was Windows-only.)
if os.name == 'nt':
	KNOWN_LIB = 'kernel32'
	KNOWN_SYMBOLS = ( 'GetLastError', 'CloseHandle', 'HeapAlloc' )
else:
	KNOWN_LIB = 'c'
	KNOWN_SYMBOLS = ( 'printf', 'malloc', 'free' )
KNOWN_SYMBOL = KNOWN_SYMBOLS[0]

# exit-code stride: a failing sub-test returns  case_index * _STRIDE + subcode .
# case_index is small (< number of methods in a class) and subcode is the tiny
# 1..N the original program returned, so this stays well inside the 32-bit
# process exit code Windows preserves. Sub-codes must be < _STRIDE.
_STRIDE = 1000


# --- AST namespacing --------------------------------------------------------

def _names_in_target( target: ast.expr ) -> set[str]:
	out: set[str] = set()
	if isinstance( target, ast.Name ):
		out.add( target.id )
	elif isinstance( target, ( ast.Tuple, ast.List ) ):
		for elt in target.elts:
			out |= _names_in_target( elt )
	return out


def _top_level_names( module: ast.Module ) -> set[str]:
	''' the names a MetalPy module binds at top level: functions, classes, and
	module-level (annotated or plain) assignments. These are exactly the names
	that could collide between two independent programs merged into one. '''
	names: set[str] = set()
	for node in module.body:
		if isinstance( node, ( ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef ) ):
			names.add( node.name )
		elif isinstance( node, ast.Assign ):
			for tgt in node.targets:
				names |= _names_in_target( tgt )
		elif isinstance( node, ast.AnnAssign ) and isinstance( node.target, ast.Name ):
			names.add( node.target.id )
	return names


class _Renamer( ast.NodeTransformer ):
	''' renames every top-level name (and every reference to one) per a mapping.
	Uniform prefixing keeps within-program shadowing consistent and can only
	ever produce a fresh, unique identifier, so a mistake surfaces as a loud
	duplicate/undefined-symbol compile error - never a silently passing test. '''
	def __init__( self, mapping: dict[str, str] ) -> None:
		self.mapping = mapping

	def visit_Name( self, node: ast.Name ) -> ast.Name:
		if node.id in self.mapping:
			node.id = self.mapping[ node.id ]
		return node

	def _rename_def( self, node ):
		if node.name in self.mapping:
			node.name = self.mapping[ node.name ]
		self.generic_visit( node )
		return node

	visit_FunctionDef = _rename_def
	visit_AsyncFunctionDef = _rename_def
	visit_ClassDef = _rename_def


def _transform_case( source: str, index: int ) -> dict:
	''' namespaces one sub-program. Returns its hoisted import lines, its
	remaining (renamed) body source, the renamed entry-function name, and
	whether that entry returns i32 (vs None). '''
	module = ast.parse( source )
	mapping = { name: f'_case{index}_{name}' for name in _top_level_names( module ) }
	_Renamer( mapping ).visit( module )
	ast.fix_missing_locations( module )

	imports: list[str] = []
	rest: list[ast.stmt] = []
	for node in module.body:
		( imports if isinstance( node, ( ast.Import, ast.ImportFrom ) ) else rest ).append( node )

	entry = mapping.get( 'main' )
	returns_i32 = False
	for node in rest:
		if isinstance( node, ast.FunctionDef ) and node.name == entry:
			returns_i32 = isinstance( node.returns, ast.Name ) and node.returns.id == 'i32'

	body_module = ast.Module( body = rest, type_ignores = [] )
	ast.fix_missing_locations( body_module )
	return {
		'imports': [ ast.unparse( node ) for node in imports ],
		'body': ast.unparse( body_module ),
		'entry': entry,
		'returns_i32': returns_i32,
	}


def _merge_programs( cases: list[tuple[str, str]] ) -> str:
	''' combines (name, source) programs into one MetalPy source: deduped
	imports, each program's namespaced body, and a dispatch main() that calls
	each entry and returns case_index*_STRIDE + subcode on the first failure. '''
	seen_imports: set[str] = set()
	import_lines: list[str] = []
	bodies: list[str] = []
	dispatch: list[str] = []
	for index, ( name, source ) in enumerate( cases ):
		case = _transform_case( source, index )
		for line in case[ 'imports' ]:
			if line not in seen_imports:
				seen_imports.add( line )
				import_lines.append( line )
		bodies.append( f'# --- case {index}: {name} ---\n' + case[ 'body' ] )
		entry = case[ 'entry' ]
		if entry is None:
			continue  # program with no main(): its top-level/compile is the test
		if case[ 'returns_i32' ]:
			dispatch.append( f'\t\t_r{index}: i32 = {entry}()' )
			dispatch.append( f'\t\tif _r{index} != 0:' )
			dispatch.append( f'\t\t\treturn {index * _STRIDE} + _r{index}' )
		else:
			dispatch.append( f'\t\t{entry}()' )

	# the base+subcode additions are ordinary bounded arithmetic; metalpy's
	# default checked mode would reject a bare `+`, so run the dispatch under
	# wrap_arithmetic (the calls themselves keep their own arithmetic modes)
	main_src = 'def main() -> i32:\n\twith compiler.wrap_arithmetic:\n' + '\n'.join( dispatch ) + '\n\treturn 0\n'
	return '\n'.join( import_lines ) + '\n\n' + '\n\n'.join( bodies ) + '\n\n' + main_src


# --- the mixin --------------------------------------------------------------

class RealCompileMixin:
	''' mix into a unittest.TestCase to compile+link+run real MetalPy programs.
	Requires a C compiler; classes using it should be decorated with
	@unittest.skipUnless( test_support.HAS_CC, ... ). '''

	def _extern_ldflags( self, compiler: Compiler ) -> str:
		''' derive linker flags from compiler.extern_libs, matching mpy.py's
		own link step (including linker_c.resolve_lib_ldflag's ntdll special
		case). 'c' is the CRT, handled by the compiler/link defaults. '''
		flags: list[str] = []
		for lib in sorted( compiler.extern_libs ):
			if lib == 'c':
				continue
			flags.append( linker_c.resolve_lib_ldflag( _CC, lib, compiler.extern_libs[lib] ) )
		return ' '.join( flags )

	def _build_and_run( self, compiler: Compiler, c_source: str, timeout: float | None, *, extra_args: list[str] | None = None ) -> subprocess.CompletedProcess:
		''' shared tail: write C, compile, link (with the program's extern libs),
		run, and return the finished process. Asserts compile/link succeed.
		extra_args are appended to the exe's own argv (argv[0] is always the
		exe path itself, same as any real process) - for tests of sys.argv
		specifically; every other caller leaves this at its default (none). '''
		with tempfile.TemporaryDirectory() as tmp:
			src_path = Path( tmp ) / 'generated.c'
			obj_path = Path( tmp ) / 'generated.o'
			exe_path = Path( tmp ) / 'test_exe'
			src_path.write_text( c_source, encoding = 'utf-8' )
			cc_result = _CC.compile( src_path, obj_path )
			self.assertEqual( cc_result.returncode, 0,
				f'{_CC.name} compile failed:\nstdout: {cc_result.stdout}\nstderr: {cc_result.stderr}{c_source_on_failure( c_source )}' )
			link_result = _CC.link( exe_path, [ obj_path ], ldflags = self._extern_ldflags( compiler ) )
			self.assertEqual( link_result.returncode, 0,
				f'{_CC.name} link failed:\nstdout: {link_result.stdout}\nstderr: {link_result.stderr}' )
			try:
				# cwd=tmp: a compiled program that writes/reads a relative
				# path (csv_*_test.py's File.binary_writer('some.tmp'), etc)
				# otherwise inherits the TEST RUNNER's own cwd, littering the
				# repo/worktree root with files that never get cleaned up.
				# tmp already gets deleted when this `with` block exits, so
				# this is free cleanup too.
				argv = [ str( exe_path ) ] + ( extra_args or [] )
				return subprocess.run( argv, capture_output = True, timeout = timeout, cwd = tmp )
			except subprocess.TimeoutExpired:
				self.fail( f'exe did not finish within {timeout}s' )

	_LIVE_RC_OBJECTS_RE = re.compile( rb'-- live RC objects \((\d+)\) --\n' )

	def _split_off_leak_report( self, stdout: bytes ) -> bytes:
		''' a debug build's own __metalpy_deinit() appends a
		"-- live RC objects (N) --" leak-check report (emitter_c.py's
		emit_c(..., leak_check=True), the default) after main()'s real
		output - asserts N is 0 (a real, checkable "no RC bugs introduced"
		result, not just tolerating whatever it prints) and returns
		everything BEFORE that marker line, i.e. the program's own real
		output. Only meaningful when the leak-check epilogue is actually
		compiled in (emit_c.py's own deinit_enabled: debug build, default
		leak_check, and a parameterless main - see its own comment) - a
		program compiled without one has no such line to find at all. '''
		m = self._LIVE_RC_OBJECTS_RE.search( stdout )
		self.assertIsNotNone( m, f'expected a "-- live RC objects (N) --" leak-check line in stdout:\n{stdout!r}' )
		self.assertEqual( m.group( 1 ), b'0', f'RC leak check reported live objects:\n{stdout.decode("utf-8", errors="replace")}' )
		return stdout[ :m.start() ]

	def _compile_source( self, source: str ) -> Compiler:
		''' discover builtins, import + compile one MetalPy program, and assert
		it produced no compile errors. Returns the driven Compiler. '''
		discovery = Discovery( import_builtins = True )
		compiler = Compiler( discovery )
		compiler.import_code( source, Path( '__main__.py' ), scope = None )
		compiler.run()
		self.assertEqual( discovery.errors.errors, [],
			'compile errors:\n' + '\n'.join( str( e ) for e in discovery.errors.errors ) )
		return compiler

	def _assert_compiles_and_runs( self, c_source: str, expected_exit: int = 0, *, compiler: Compiler | None = None, timeout: float | None = None, extra_args: list[str] | None = None ) -> None:
		''' original one-program path: compile+link+run already-emitted C and
		assert the exit code. `compiler` supplies the program's extern libs for
		linking (defaults to self.compiler, matching the legacy callers). '''
		result = self._build_and_run( compiler or self.compiler, c_source, timeout, extra_args = extra_args )
		self.assertEqual( result.returncode, expected_exit,
			f'exe exited {result.returncode}, expected {expected_exit} (stderr: {result.stderr})' )

	def assert_programs_run( self, cases: list[tuple[str, str]], *, timeout: float | None = None ) -> None:
		''' combine several standalone MetalPy programs into ONE executable and
		run it once. Each case is (name, source); each source is a full program
		whose entry `def main() -> i32` returns 0 on success (or `-> None` for a
		compile/run-only smoke test). A nonzero exit is decoded back to the
		failing case name and its own sub-code. Do NOT use for programs that
		depend on process-global one-time init - run those individually. '''
		compiler = self._compile_source( _merge_programs( cases ) )
		result = self._build_and_run( compiler, emitter_c.emit_c( compiler ), timeout )
		if result.returncode == 0:
			self._split_off_leak_report( result.stdout )
			return
		code = result.returncode
		if 0 < code < len( cases ) * _STRIDE:
			name = cases[ code // _STRIDE ][ 0 ]
			self.fail( f'merged program exited {code}: case {code // _STRIDE} ({name}) '
				f'sub-check {code % _STRIDE} failed\nstdout: {result.stdout}\nstderr: {result.stderr}' )
		self.fail( f'merged program exited {code} (unmapped - likely a crash/corruption, not a plain '
			f'check failure)\nstdout: {result.stdout}\nstderr: {result.stderr}' )


HAS_CC = _CC is not None
