# Real-compile-and-run tests for sys.argv (lib/sys.py).
#
# emit_c()'s entry-point prelude captures the C-level main(argc, argv) into
# two raw globals (sys$_raw_argc/sys$_raw_argv) BEFORE __metalpy_init() runs
# - see emitter_c.py's own comment on that injection - so this needs a real
# compiled exe invoked with real extra command-line arguments, not just a
# compile-time check. test_support.RealCompileMixin's _build_and_run/
# _assert_compiles_and_runs grew an `extra_args` parameter for exactly this.

# stdlib imports:
from pathlib import Path
import subprocess
import tempfile
import unittest

# local imports:
import linker_c
import test_support


class SysArgvTests( test_support.RealCompileMixin, unittest.TestCase ):
	def setUp( self ) -> None:
		from discovery import Discovery
		from compiler import Compiler
		self.discovery = Discovery( import_builtins = True )
		self.compiler = Compiler( self.discovery )

	def _run( self, code: str ) -> None:
		from pathlib import Path
		self.compiler.import_code( code, Path( '__main__.py' ), scope = None )
		self.compiler.run()

	@unittest.skipUnless( test_support.HAS_CC, 'no C compiler (clang/gcc/msvc) found - skipping' )
	def test_argv0_present_with_no_extra_args( self ) -> None:
		import emitter_c
		self._run( '''
import sys

def main() -> i32:
	if len( sys.argv ) < 1:
		return 1
	return 0
''' )
		self.assertEqual( self.discovery.errors.errors, [] )
		self._assert_compiles_and_runs( emitter_c.emit_c( self.compiler ), expected_exit = 0 )

	@unittest.skipUnless( test_support.HAS_CC, 'no C compiler (clang/gcc/msvc) found - skipping' )
	def test_extra_args_captured_in_order( self ) -> None:
		import emitter_c
		self._run( '''
import sys

def main() -> i32:
	if len( sys.argv ) != 4:
		return 1
	if sys.argv.__getitem__( 1 ).unwrap( 'idx' ) != 'hello':
		return 2
	if sys.argv.__getitem__( 2 ).unwrap( 'idx' ) != 'world 2':
		return 3
	if sys.argv.__getitem__( 3 ).unwrap( 'idx' ) != '-x':
		return 4
	return 0
''' )
		self.assertEqual( self.discovery.errors.errors, [] )
		self._assert_compiles_and_runs(
			emitter_c.emit_c( self.compiler ), expected_exit = 0,
			extra_args = [ 'hello', 'world 2', '-x' ],
		)

	@unittest.skipUnless( test_support.HAS_CC, 'no C compiler (clang/gcc/msvc) found - skipping' )
	def test_argv_correct_under_freestanding_no_crt_build( self ) -> None:
		# regression test: every other test in this file calls emit_c()/
		# _assert_compiles_and_runs() with no_crt defaulted to False, always
		# linking the real CRT regardless of what the program actually needs
		# - so they never exercised the no_crt decision mpy.py's own CLI
		# makes (`'c' not in extern_libs and not compiler.requires_crt`). A
		# plain `for arg in sys.argv:` program has no @extern('c', ...) call
		# of its own, so on Windows it picks the freestanding no_crt entry
		# point (mainCRTStartup calls main(0, NULL) - see emitter_c.py's own
		# comment there); the old _raw_argc/_raw_argv-from-main()
		# implementation silently left sys.argv permanently empty there
		# regardless of the real command line. sys.py's Windows _build_argv
		# now reads GetCommandLineW()/CommandLineToArgvW() directly instead,
		# which works identically whether or not the CRT is linked.
		#
		# Unlike every other test here, this builds+links manually (mirrors
		# emitter_c_test.py's RequiresCrtDecoratorRealCompileTests) rather
		# than going through _assert_compiles_and_runs/_build_and_run - that
		# shared helper's own compile()/link() calls don't take a no_crt
		# argument at all, so it always compiles+links in CRT mode
		# regardless of what emit_c() generated. Passing no_crt=True only to
		# emit_c() while compiling/linking as if no_crt=False mismatches the
		# generated C (its own freestanding mainCRTStartup) against CRT-mode
		# compile flags (e.g. MSVC's /RTC1, which needs CRT-provided
		# support symbols /RTC1 itself gates off under real no_crt - see
		# linker_c.py's own comment) - confirmed via a real repro to fail
		# only on MSVC (clang/gcc tolerate the mismatch). mpy.py's real CLI
		# always threads the same no_crt value through emit_c()/compile()/
		# link() together (see mpy.py's own no_crt computation) - this test
		# does the same.
		import emitter_c
		self._run( '''
import sys

def main() -> i32:
	if len( sys.argv ) != 3:
		return 1
	if sys.argv.__getitem__( 1 ).unwrap( 'idx' ) != 'hello':
		return 2
	if sys.argv.__getitem__( 2 ).unwrap( 'idx' ) != 'world':
		return 3
	return 0
''' )
		self.assertEqual( self.discovery.errors.errors, [] )
		self.assertFalse( self.compiler.requires_crt )
		no_crt = 'c' not in self.compiler.extern_libs and not self.compiler.requires_crt
		if self.discovery.active_target['os'] == 'windows':
			# no @extern('c', ...) call anywhere in this program - confirms
			# the freestanding path is genuinely being exercised below, not
			# silently upgraded to a CRT-linked build for some other reason
			self.assertNotIn( 'c', self.compiler.extern_libs )
			self.assertTrue( no_crt )
		c_source = emitter_c.emit_c( self.compiler, no_crt = no_crt )

		with tempfile.TemporaryDirectory() as tmp:
			src_path = Path( tmp ) / 'generated.c'
			obj_path = Path( tmp ) / 'generated.o'
			exe_path = Path( tmp ) / 'test_exe.exe'
			src_path.write_text( c_source, encoding = 'utf-8' )

			cc = test_support._CC
			cc_result = cc.compile( src_path, obj_path, no_crt = no_crt )
			self.assertEqual( cc_result.returncode, 0,
				f'{cc.name} compile failed:\n{cc_result.stdout}{test_support.c_source_on_failure( c_source )}' )

			ldflags = ''
			for lib in sorted( self.compiler.extern_libs ):
				if lib == 'c':
					continue
				flag = linker_c.resolve_lib_ldflag( cc, lib, self.compiler.extern_libs[lib], no_crt = no_crt )
				ldflags = ldflags + f' {flag}' if ldflags else flag

			link_result = cc.link( exe_path, [ obj_path ], ldflags = ldflags, no_crt = no_crt )
			self.assertEqual( link_result.returncode, 0, f'{cc.name} link failed:\n{link_result.stdout}' )

			result = subprocess.run(
				[ str( exe_path ), 'hello', 'world' ], capture_output = True, cwd = tmp,
			)
			self.assertEqual( result.returncode, 0, f'exe exited {result.returncode}, expected 0 (stderr: {result.stderr})' )

	@unittest.skipUnless( test_support.HAS_CC, 'no C compiler (clang/gcc/msvc) found - skipping' )
	def test_empty_argument_preserved( self ) -> None:
		# an empty string IS a valid, distinct argv element (e.g. `grap ""`)
		# - not the same as "no argument was passed there at all"
		import emitter_c
		self._run( '''
import sys

def main() -> i32:
	if len( sys.argv ) != 3:
		return 1
	if sys.argv.__getitem__( 1 ).unwrap( 'idx' ) != '':
		return 2
	if sys.argv.__getitem__( 2 ).unwrap( 'idx' ) != 'after':
		return 3
	return 0
''' )
		self.assertEqual( self.discovery.errors.errors, [] )
		self._assert_compiles_and_runs(
			emitter_c.emit_c( self.compiler ), expected_exit = 0,
			extra_args = [ '', 'after' ],
		)


if __name__ == '__main__':
	unittest.main()
