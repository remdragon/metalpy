# Real-compile-and-run tests for lib/signal.py: signal.context()'s scoped
# SIGINT redirection. raise(SIGINT) (plain ANSI C <signal.h>, portable
# across POSIX/Windows CRT the same way signal() itself is) is used to
# actually DELIVER the signal to this same process/thread synchronously,
# rather than merely asserting the code compiles - per "verify, don't
# trust IR-level success" (a handler that's installed but never actually
# invoked proves nothing).

import unittest

import test_support
from test_support import RealCompileMixin


class SignalContextTests( RealCompileMixin, unittest.TestCase ):
	@unittest.skipUnless( test_support.HAS_CC, 'no C compiler (clang/gcc/msvc) found - skipping' )
	def test_programs_compile_and_run( self ) -> None:
		import emitter_c
		self.assert_programs_run([
			( 'flag_starts_clear', '''
import signal

def main() -> i32:
	flag: signal.Flag = signal.Flag()
	if flag.get():
		return 1
	return 0
''' ),
			( 'raised_sigint_sets_the_flag_inside_the_context', '''
import compiler
import signal

@extern( 'c', 'raise', header = 'signal.h' )
def _raise( sig: i32 ) -> i32:
	...

def main() -> i32:
	flag: signal.Flag = signal.Flag()
	with signal.context( signal.SIGINT, flag ):
		if flag.get():
			return 1
		_raise( signal.SIGINT )
		if not flag.get():
			return 2
	return 0
''' ),
			( 'flag_from_a_finished_context_is_inert_afterward', '''
import compiler
import signal

@extern( 'c', 'raise', header = 'signal.h' )
def _raise( sig: i32 ) -> i32:
	...

def main() -> i32:
	flag: signal.Flag = signal.Flag()
	with signal.context( signal.SIGINT, flag ):
		_raise( signal.SIGINT )
	if not flag.get():
		return 1 # the flag itself keeps whatever value it had - only the OS-level handler is torn down
	return 0
''' ),
		])

	@unittest.skipUnless( test_support.HAS_CC, 'no C compiler (clang/gcc/msvc) found - skipping real-compile signal tests' )
	def test_nested_context_panics( self ) -> None:
		''' a real process-exit check, not merged via assert_programs_run:
		a panicking sub-program's abrupt exit(1) would break the merged
		dispatch's own "each case's main() returns normally" assumption. '''
		from discovery import Discovery
		from compiler import Compiler
		from pathlib import Path
		import emitter_c
		discovery = Discovery( import_builtins = True )
		compiler = Compiler( discovery )
		compiler.import_code( '''
import signal

def main() -> i32:
	a: signal.Flag = signal.Flag()
	b: signal.Flag = signal.Flag()
	with signal.context( signal.SIGINT, a ):
		with signal.context( signal.SIGINT, b ):
			pass
	return 0
''', Path( '__main__.py' ), scope = None )
		compiler.run()
		self.assertEqual( discovery.errors.errors, [] )
		self._assert_compiles_and_runs( emitter_c.emit_c( compiler ), expected_exit = 1, compiler = compiler )


if __name__ == '__main__':
	unittest.main()
