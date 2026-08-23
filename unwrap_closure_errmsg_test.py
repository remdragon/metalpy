# Real-compile-and-run tests for Result[T,E].unwrap()'s second overload
# (lib/builtins/__init__.py): errmsg can now be either a plain str (the
# original form) or a one-argument Ptr[Callable[[E],str]] formatter called
# with the Err payload to build the panic message lazily. See
# unwrap_or_general_narrowing_test.py for the sibling unwrap_or coverage
# this mirrors.
#
# Design note (why Ptr[Callable[[E],str]], not Closure[[E],str]): a
# NON-capturing lambda (e.g. `lambda e: f'...{e}'`, the common case and the
# one demonstrated in demo/http_server_demo_*.mpy) always compiles to the
# zero-cost Ptr[Callable[...]] shape, never a ClosureType, regardless of
# what expected_type asks for (real_closures_shipped's own "empty captures =
# unchanged zero-cost Ptr[Callable[...]] path" design decision) - so a
# Closure[[E],str]-only overload would reject the exact lambda form users
# actually write. A capturing lambda (closing over an outer local) is left
# unsupported here - the two implementations share Result.unwrap's own
# qualname, and monomorphize.py's _get_or_create_specialization cache is
# keyed by qualname alone, so a third bare-implementation candidate would
# collide with this one the same way a first attempt at two independent
# plain `unwrap` implementations did (confirmed via a real compile: the
# second implementation silently got back the first one's cached
# specialization) - hence the @overload-stub + single-shared-impl shape
# below, matching unwrap_or's own precedent.
#
# Found and fixed two real, PRE-EXISTING compiler bugs while building this:
# 1. lowering.py's _lower_overload_arg only ever special-cased a bare
#    ast.Constant literal argument to an overloaded call - any other
#    expression (including a lambda) fell through to expected_type=None,
#    which _expr_Lambda rejects outright ("no expected Callable[...]
#    context"). New _lower_overload_lambda_arg mirrors the literal path.
# 2. lowering.py's _construct_capturing_closure blindly trusted its own
#    expected_type for a freshly-allocated closure's dest type, even when
#    expected_type was something else entirely (e.g. a Ptr[Callable[...]]
#    slot a capturing lambda doesn't actually satisfy) - silently produced
#    an ir.Allocate whose dest.type wasn't the RCClass being constructed,
#    crashing emitter_c.py's own internal `assert isinstance(concrete_cls,
#    RCClass)` instead of failing cleanly. Confirmed via a real compile: a
#    capturing lambda passed to unwrap() crashed the emitter until fixed;
#    now correctly rejected with an ordinary "expected X, got Y" error.

import unittest

import test_support
from test_support import RealCompileMixin

_UNWRAP_CALLABLE_ERRMSG_OK_PATHS = '''
def maybe( flag: bool ) -> Result[i32, str]:
	if flag:
		return Result.Ok( 7 )
	return Result.Err( 'nope' )

def main() -> i32:
	# str form, unchanged from before this feature - still works alongside
	# the new overload
	v1: i32 = maybe( True ).unwrap( 'unexpected' )
	if v1 != 7:
		return 1

	# new callable form, Ok path - the closure is never called
	v2: i32 = maybe( True ).unwrap( lambda e: f'unexpected: {e}' )
	if v2 != 7:
		return 2
	return 0
'''

_UNWRAP_STR_ERRMSG_ERR_PANICS = '''
def main() -> i32:
	bad: Result[i32, str] = Result.Err( 'boom' )
	v: i32 = bad.unwrap( 'plain message' )
	return v
'''

_UNWRAP_CALLABLE_ERRMSG_ERR_PANICS = '''
def main() -> i32:
	bad: Result[i32, str] = Result.Err( 'boom' )
	v: i32 = bad.unwrap( lambda e: f'formatted: {e}' )
	return v
'''


@unittest.skipUnless( test_support.HAS_CC, 'no C compiler (clang/gcc/msvc) found - skipping real-compile RC tests' )
class UnwrapClosureErrmsgTests( RealCompileMixin, unittest.TestCase ):
	def test_unwrap_callable_errmsg_ok_paths( self ) -> None:
		self.assert_programs_run([ ( 'unwrap_callable_errmsg_ok_paths', _UNWRAP_CALLABLE_ERRMSG_OK_PATHS ) ])

	def test_unwrap_str_errmsg_err_panics_with_message( self ) -> None:
		compiler = self._compile_source( _UNWRAP_STR_ERRMSG_ERR_PANICS )
		import emitter_c
		result = self._build_and_run( compiler, emitter_c.emit_c( compiler ), timeout = None )
		self.assertEqual( result.returncode, 1, f'stderr: {result.stderr}' )
		self.assertIn( b'plain message', result.stderr )

	def test_unwrap_callable_errmsg_err_panics_with_formatted_message( self ) -> None:
		compiler = self._compile_source( _UNWRAP_CALLABLE_ERRMSG_ERR_PANICS )
		import emitter_c
		result = self._build_and_run( compiler, emitter_c.emit_c( compiler ), timeout = None )
		self.assertEqual( result.returncode, 1, f'stderr: {result.stderr}' )
		self.assertIn( b'formatted: boom', result.stderr )

	def test_unwrap_capturing_lambda_errmsg_is_a_clean_compile_error( self ) -> None:
		# a capturing lambda (closing over an outer local) isn't supported
		# for this overload - see this file's own module docstring for why.
		# Before the _construct_capturing_closure fix this crashed the
		# emitter with an internal AssertionError instead of failing here
		from discovery import Discovery
		from compiler import Compiler
		from pathlib import Path
		discovery = Discovery( import_builtins = True )
		compiler = Compiler( discovery )
		compiler.import_code( '''
def main() -> i32:
	host: str = 'example'
	bad: Result[i32, str] = Result.Err( 'boom' )
	v: i32 = bad.unwrap( lambda e: f'{host}: {e}' )
	return v
''', Path( '__main__.py' ), scope = None )
		compiler.run()
		self.assertTrue( any( 'expected' in e and 'got' in e for e in discovery.errors.errors ),
			f'expected a clean type-mismatch error, got: {discovery.errors.errors}' )
