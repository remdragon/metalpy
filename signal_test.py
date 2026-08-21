# Real-compile-and-run tests for lib/signal.py: signal.signal()'s permanent,
# process-wide registration and signal.context()'s scoped, reentrant
# override. raise(SIGINT) (plain ANSI C <signal.h>, portable across
# POSIX/Windows CRT the same way signal() itself is) is used to actually
# DELIVER the signal to this same process/thread synchronously, rather than
# merely asserting the code compiles - per "verify, don't trust IR-level
# success" (a handler that's installed but never actually invoked proves
# nothing).

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
		return 1 # the flag itself keeps whatever value it had - only the registration pointing at it is gone
	return 0
''' ),
			( 'nested_context_same_signal_shadows_then_restores', '''
import compiler
import signal

@extern( 'c', 'raise', header = 'signal.h' )
def _raise( sig: i32 ) -> i32:
	...

def main() -> i32:
	outer: signal.Flag = signal.Flag()
	inner: signal.Flag = signal.Flag()
	with signal.context( signal.SIGINT, outer ):
		with signal.context( signal.SIGINT, inner ):
			_raise( signal.SIGINT )
			if not inner.get():
				return 1
			if outer.get():
				return 2
		_raise( signal.SIGINT )
		if not outer.get():
			return 3
	return 0
''' ),
		])

	@unittest.skipUnless( test_support.HAS_CC, 'no C compiler (clang/gcc/msvc) found - skipping real-compile signal tests' )
	def test_signal_signal_registers_a_permanent_handler( self ) -> None:
		''' not merged via assert_programs_run: signal.signal()'s own
		registration is process-wide and outlives the sub-program that made
		it - exactly the "process-global one-time init" shape
		test_support.assert_programs_run's own docstring says must run
		standalone, since a later merged case sharing the same process
		would otherwise see this leftover registration too. '''
		self._assert_compiles_and_runs( self._emit( '''
import compiler
import signal

@extern( 'c', 'raise', header = 'signal.h' )
def _raise( sig: i32 ) -> i32:
	...

_fired: bool = False

def _on_sigint( sig: i32 ) -> None:
	global _fired
	_fired = True

def main() -> i32:
	signal.signal( signal.SIGINT, _on_sigint )
	_raise( signal.SIGINT )
	if not _fired:
		return 1
	return 0
''' ) )

	@unittest.skipUnless( test_support.HAS_CC, 'no C compiler (clang/gcc/msvc) found - skipping real-compile signal tests' )
	def test_context_shadows_signal_signal_then_reverts( self ) -> None:
		''' not merged - same process-global-registration reason as
		test_signal_signal_registers_a_permanent_handler above. Checks the
		full priority order: this thread's own context() registration wins
		over signal.signal()'s while the context is active, and
		signal.signal()'s own registration resumes once the context exits. '''
		self._assert_compiles_and_runs( self._emit( '''
import compiler
import signal

@extern( 'c', 'raise', header = 'signal.h' )
def _raise( sig: i32 ) -> i32:
	...

_global_fired: bool = False

def _on_sigint( sig: i32 ) -> None:
	global _global_fired
	_global_fired = True

def main() -> i32:
	signal.signal( signal.SIGINT, _on_sigint )
	_raise( signal.SIGINT )
	if not _global_fired:
		return 1
	flag: signal.Flag = signal.Flag()
	with signal.context( signal.SIGINT, flag ):
		_global_fired = False
		_raise( signal.SIGINT )
		if not flag.get():
			return 2
		if _global_fired:
			return 3
	_raise( signal.SIGINT )
	if not _global_fired:
		return 4
	return 0
''' ) )

	def _emit( self, source: str ) -> str:
		''' compile+emit one standalone MetalPy program's C, storing its
		Compiler on self so _assert_compiles_and_runs can find its extern
		libs (matches _assert_compiles_and_runs's own `compiler` default). '''
		import emitter_c
		self.compiler = self._compile_source( source )
		return emitter_c.emit_c( self.compiler )


if __name__ == '__main__':
	unittest.main()
