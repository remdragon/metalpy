# Real-compile-and-run tests for assert's compiler.target.debug gating
# (type_resolver.py's visit_Assert): the message argument stays mandatory
# regardless of target, but whether the check actually RUNS is target-
# dependent - stripped entirely (dead code, not even a runtime `if`) in a
# release build, same mechanism sys.alloc's own poison-fill already uses.

import unittest

import test_support
from test_support import RealCompileMixin
from discovery import ActiveTarget, Discovery
from compiler import Compiler
from pathlib import Path
import emitter_c


def _release_target() -> ActiveTarget:
	# start from the real default (matches the host this suite actually
	# runs on) and flip only 'debug' - active_target replaces the whole
	# dict, so every other key (os/arch/family/bits/posix) has to survive
	debug_target = Discovery( import_builtins = True ).active_target
	release_target = dict( debug_target )
	release_target['debug'] = False
	return release_target


class AssertRequiresMessageTests( unittest.TestCase ):
	def test_bare_assert_without_message_is_a_compile_error( self ) -> None:
		discovery = Discovery( import_builtins = True )
		compiler = Compiler( discovery )
		compiler.import_code( '''
def main() -> i32:
	assert True
	return 0
''', Path( '__main__.py' ), scope = None )
		compiler.run()
		self.assertTrue( any( 'assert requires a message' in e for e in discovery.errors.errors ) )


@unittest.skipUnless( test_support.HAS_CC, 'no C compiler (clang/gcc/msvc) found - skipping real-compile assert tests' )
class AssertDebugGatingTests( RealCompileMixin, unittest.TestCase ):
	def test_debug_build_panics_on_a_false_condition( self ) -> None:
		discovery = Discovery( import_builtins = True ) # default target: debug=True
		compiler = Compiler( discovery )
		compiler.import_code( '''
def main() -> i32:
	assert 1 == 2, "one is not two"
	return 0
''', Path( '__main__.py' ), scope = None )
		compiler.run()
		self.assertEqual( discovery.errors.errors, [] )
		self._assert_compiles_and_runs( emitter_c.emit_c( compiler ), expected_exit = 1, compiler = compiler )

	def test_debug_build_passes_on_a_true_condition( self ) -> None:
		discovery = Discovery( import_builtins = True )
		compiler = Compiler( discovery )
		compiler.import_code( '''
def main() -> i32:
	assert 1 == 1, "unreachable"
	return 0
''', Path( '__main__.py' ), scope = None )
		compiler.run()
		self.assertEqual( discovery.errors.errors, [] )
		self._assert_compiles_and_runs( emitter_c.emit_c( compiler ), expected_exit = 0, compiler = compiler )

	def test_release_build_strips_the_check_entirely( self ) -> None:
		# a condition that would panic in a debug build must be silently
		# skipped here - the whole point of the gating
		discovery = Discovery( import_builtins = True, active_target = _release_target() )
		compiler = Compiler( discovery )
		compiler.import_code( '''
def main() -> i32:
	assert 1 == 2, "should never run in release"
	return 0
''', Path( '__main__.py' ), scope = None )
		compiler.run()
		self.assertEqual( discovery.errors.errors, [] )
		self._assert_compiles_and_runs( emitter_c.emit_c( compiler ), expected_exit = 0, compiler = compiler )


if __name__ == '__main__':
	unittest.main()
