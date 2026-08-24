# Real-compile-and-run tests for sys.argv (lib/sys.py).
#
# emit_c()'s entry-point prelude captures the C-level main(argc, argv) into
# two raw globals (sys$_raw_argc/sys$_raw_argv) BEFORE __metalpy_init() runs
# - see emitter_c.py's own comment on that injection - so this needs a real
# compiled exe invoked with real extra command-line arguments, not just a
# compile-time check. test_support.RealCompileMixin's _build_and_run/
# _assert_compiles_and_runs grew an `extra_args` parameter for exactly this.

# stdlib imports:
import unittest

# local imports:
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
	def test_using_argv_forces_crt_and_captures_real_args( self ) -> None:
		# regression test: every other test in this file calls emit_c() with
		# no_crt defaulted to False, always linking the real CRT regardless
		# of what the program actually needs - so they never exercised the
		# no_crt decision mpy.py's own CLI makes (`'c' not in extern_libs and
		# not compiler.requires_crt`). A plain `for arg in sys.argv:` program
		# has no @extern('c', ...) call of its own, so on Windows it used to
		# pick the freestanding no_crt entry point (mainCRTStartup calls
		# main(0, NULL) - see emitter_c.py's own comment there), silently
		# leaving sys.argv permanently empty regardless of the real command
		# line. _build_argv is now @requires_crt (lib/sys.py) specifically
		# so this can't happen - confirm both the flag and the real captured
		# values, computing no_crt the same way mpy.py does instead of
		# relying on emit_c()'s CRT-by-default parameter.
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
		self.assertTrue( self.compiler.requires_crt )
		if self.discovery.active_target['os'] == 'windows':
			# 'c' is a real, Windows-only concept (see emitter_c.py's no_crt
			# handling) - on POSIX targets sys._alloc/etc already extern('c',
			# ...) directly regardless of @requires_crt, so this assertion
			# would be meaningless there (confirmed: 'c' is always present)
			self.assertNotIn( 'c', self.compiler.extern_libs )
		no_crt = 'c' not in self.compiler.extern_libs and not self.compiler.requires_crt
		self.assertFalse( no_crt )
		self._assert_compiles_and_runs(
			emitter_c.emit_c( self.compiler, no_crt = no_crt ), expected_exit = 0,
			extra_args = [ 'hello', 'world' ],
		)

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
