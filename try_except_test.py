# Real-compile-and-run + compile-error regression tests for limited
# try/except/else/finally + Result.or_throw() (NOT real exceptions - no
# `raise`, no unwinding: the only way control reaches an except handler is
# `.or_throw()` called on a Result-typed expression, textually inside that
# try body, in the same function). See lowering.py's _stmt_Try/_lower_or_throw
# and ir.OrThrow for the implementation.
#
# The compile-error cases (rejection of user-defined or_throw, except*, bare
# except:, loop-escape, generator-body, insufficient function return-type
# coverage) don't need a real C compiler at all - those use Discovery/
# Compiler directly, same pattern lowering_test.py's own rejection tests use.
# The positive/behavioral cases need a real compile+run (RealCompileMixin),
# same pattern or_return_rc_test.py uses.

from pathlib import Path
import unittest

from compiler import Compiler
from discovery import Discovery
import test_support
from test_support import RealCompileMixin


# --- real compile+run: positive/behavioral coverage -------------------------

_MATCHED_LEAF_BINDS_PAYLOAD_AND_HANDLED_FULLY_NEEDS_NO_RESULT_RETURN = '''
class ParseError:
	code: i32

def risky( bad: bool ) -> Result[i32, ParseError]:
	if bad:
		return Result.Err( ParseError( code = 42 ) )
	return Result.Ok( 7 )

def run( bad: bool ) -> i32:
	# every leaf of ParseError is handled below - run() needs no
	# Result[_,_] return type at all (item 5)
	result: i32 = 0
	try:
		result = risky( bad ).or_throw()
	except ParseError as e:
		result = e.code
	return result

def main() -> i32:
	if run( True ) != 42:
		return 1
	if run( False ) != 7:
		return 2
	return 0
'''

_TUPLE_EXCEPT_CLAUSE_MULTIPLE_LEAVES_ONE_HANDLER = '''
class ErrorA:
	pass

class ErrorB:
	pass

def risky( which: i32 ) -> Result[i32, ErrorA | ErrorB]:
	if which == 1:
		return Result.Err( ErrorA() )
	if which == 2:
		return Result.Err( ErrorB() )
	return Result.Ok( 5 )

def run( which: i32 ) -> i32:
	result: i32 = -1
	try:
		result = risky( which ).or_throw()
	except (ErrorA, ErrorB) as e:
		compiler.decref( e )
		result = 99
	return result

def main() -> i32:
	if run( 1 ) != 99:
		return 1
	if run( 2 ) != 99:
		return 2
	if run( 0 ) != 5:
		return 3
	return 0
'''

_UNMATCHED_LEAF_PROPAGATES_TO_WIDER_FUNCTION_RETURN = '''
class ErrorA:
	pass

class ErrorB:
	pass

def risky( which: i32 ) -> Result[i32, ErrorA | ErrorB]:
	if which == 1:
		return Result.Err( ErrorA() )
	if which == 2:
		return Result.Err( ErrorB() )
	return Result.Ok( 5 )

def run( which: i32 ) -> Result[i32, ErrorB]:
	try:
		v: i32 = risky( which ).or_throw()
		return Result.Ok( v )
	except ErrorA:
		return Result.Ok( -1 )

def main() -> i32:
	match run( 1 ):
		case Result.Ok( v ):
			if v != -1:
				return 1
		case Result.Err( e ):
			return 2
	match run( 2 ):
		case Result.Ok( v ):
			return 3
		case Result.Err( e ):
			pass
	match run( 0 ):
		case Result.Ok( v ):
			if v != 5:
				return 4
		case Result.Err( e ):
			return 5
	return 0
'''

_FINALLY_RUNS_ON_EVERY_EXIT_PATH = '''
class Counter:
	n: i32

def bump( c: Counter ) -> None:
	with compiler.wrap_arithmetic:
		c.n = c.n + 1

class ErrorA:
	pass

def risky( which: i32 ) -> Result[i32, ErrorA]:
	if which == 1:
		return Result.Err( ErrorA() )
	return Result.Ok( 5 )

def run_fallthrough( c: Counter, which: i32 ) -> i32:
	result: i32 = -1
	try:
		result = risky( which ).or_throw()
	except ErrorA:
		result = -2
	finally:
		bump( c )
	return result

def run_early_return( c: Counter ) -> i32:
	try:
		return 7
	finally:
		bump( c )

def run_propagate( c: Counter, which: i32 ) -> Result[i32, ErrorA]:
	# no except clause at all here - risky()'s ErrorA leaf is uncovered and
	# propagates straight to run_propagate's own Result return, but
	# finally must still run on that path too
	try:
		v: i32 = risky( which ).or_throw()
		return Result.Ok( v )
	finally:
		bump( c )

def main() -> i32:
	c: Counter = Counter( n = 0 )
	if run_fallthrough( c, 0 ) != 5:
		return 1
	if c.n != 1:
		return 2
	if run_fallthrough( c, 1 ) != -2:
		return 3
	if c.n != 2:
		return 4
	if run_early_return( c ) != 7:
		return 5
	if c.n != 3:
		return 6
	match run_propagate( c, 1 ):
		case Result.Ok( v ):
			return 7
		case Result.Err( e ):
			pass
	if c.n != 4:
		return 8
	return 0
'''

_ELSE_RUNS_ONLY_ON_NON_DISPATCH_FALLTHROUGH = '''
class ErrorA:
	pass

def risky( which: i32 ) -> Result[i32, ErrorA]:
	if which == 1:
		return Result.Err( ErrorA() )
	return Result.Ok( 5 )

def run( which: i32 ) -> i32:
	result: i32 = 0
	try:
		v: i32 = risky( which ).or_throw()
	except ErrorA:
		result = -1
	else:
		with compiler.wrap_arithmetic:
			result = v + 100
	return result

def main() -> i32:
	if run( 0 ) != 105:
		return 1
	if run( 1 ) != -1:
		return 2
	return 0
'''

_NESTED_TRY_INNER_UNCOVERED_LEAF_NOT_CAUGHT_BY_OUTER = '''
class ErrorA:
	pass

def risky() -> Result[i32, ErrorA]:
	return Result.Err( ErrorA() )

def run() -> Result[i32, ErrorA]:
	# the INNER try has no handler for ErrorA at all - its or_throw() must
	# propagate straight to run()'s own Result return, WITHOUT ever being
	# caught by the OUTER try's own `except ErrorA:` even though it's
	# textually enclosing (a documented scope limitation, not a bug - only
	# the INNERMOST enclosing try's own handlers are ever consulted)
	try:
		try:
			v: i32 = risky().or_throw()
			return Result.Ok( v )
		finally:
			pass
	except ErrorA:
		return Result.Ok( -1 )

def main() -> i32:
	match run():
		case Result.Ok( v ):
			return 1 # would wrongly fire if the outer handler had caught it
		case Result.Err( e ):
			return 0 # expected: propagated all the way out uncaught
'''

_OR_THROW_WITH_NO_ENCLOSING_TRY_IS_LIKE_OR_RETURN = '''
class ErrorA:
	pass

def risky( bad: bool ) -> Result[i32, ErrorA]:
	if bad:
		return Result.Err( ErrorA() )
	return Result.Ok( 9 )

def run( bad: bool ) -> Result[i32, ErrorA]:
	v: i32 = risky( bad ).or_throw()
	return Result.Ok( v )

def main() -> i32:
	match run( True ):
		case Result.Ok( v ):
			return 1
		case Result.Err( e ):
			pass
	match run( False ):
		case Result.Ok( v ):
			if v != 9:
				return 2
		case Result.Err( e ):
			return 3
	return 0
'''


@unittest.skipUnless( test_support.HAS_CC, 'no C compiler (clang/gcc/msvc) found - skipping real-compile try/except tests' )
class TryExceptRealCompileTests( RealCompileMixin, unittest.TestCase ):
	def test_matched_leaf_binds_payload_and_fully_handled_needs_no_result_return( self ) -> None:
		self.assert_programs_run([ ( 'matched_leaf', _MATCHED_LEAF_BINDS_PAYLOAD_AND_HANDLED_FULLY_NEEDS_NO_RESULT_RETURN ) ])

	def test_tuple_except_clause_multiple_leaves_one_handler( self ) -> None:
		self.assert_programs_run([ ( 'tuple_except', _TUPLE_EXCEPT_CLAUSE_MULTIPLE_LEAVES_ONE_HANDLER ) ])

	def test_unmatched_leaf_propagates_to_wider_function_return( self ) -> None:
		self.assert_programs_run([ ( 'unmatched_propagates', _UNMATCHED_LEAF_PROPAGATES_TO_WIDER_FUNCTION_RETURN ) ])

	def test_finally_runs_on_every_exit_path( self ) -> None:
		self.assert_programs_run([ ( 'finally_every_exit', _FINALLY_RUNS_ON_EVERY_EXIT_PATH ) ])

	def test_else_runs_only_on_non_dispatch_fallthrough( self ) -> None:
		self.assert_programs_run([ ( 'else_fallthrough_only', _ELSE_RUNS_ONLY_ON_NON_DISPATCH_FALLTHROUGH ) ])

	def test_nested_try_inner_uncovered_leaf_not_caught_by_outer( self ) -> None:
		self.assert_programs_run([ ( 'nested_try_scope', _NESTED_TRY_INNER_UNCOVERED_LEAF_NOT_CAUGHT_BY_OUTER ) ])

	def test_or_throw_with_no_enclosing_try_is_like_or_return( self ) -> None:
		self.assert_programs_run([ ( 'or_throw_no_try', _OR_THROW_WITH_NO_ENCLOSING_TRY_IS_LIKE_OR_RETURN ) ])


# --- compile-error coverage (no real C compiler needed) ---------------------

class TryExceptCompileErrorTests( unittest.TestCase ):
	def setUp( self ) -> None:
		self.discovery = Discovery( import_builtins = True )
		self.compiler = Compiler( self.discovery )

	def _import( self, code: str ):
		return self.compiler.import_code( code, filename = Path( '__test__.py' ))

	def _lower_and_get_errors( self, code: str, fn_name: str ) -> list:
		mod = self._import( code )
		fn = mod.get_local( fn_name )
		if fn.resolve is not None:
			fn.resolve()
		self.compiler._lower( fn )
		return self.discovery.errors.errors

	def test_unmatched_leaf_with_insufficient_function_return_is_a_compile_error( self ) -> None:
		code = '\n'.join([
			'class ErrorA: pass',
			'class ErrorB: pass',
			'',
			'def risky( which: i32 ) -> Result[i32, ErrorA | ErrorB]:',
			'	if which == 1:',
			'		return Result.Err( ErrorA() )',
			'	return Result.Ok( 5 )',
			'',
			'def run( which: i32 ) -> i32:',
			'	try:',
			'		v: i32 = risky( which ).or_throw()',
			'		return v',
			'	except ErrorB:',
			'		return -1',
		])
		errors = self._lower_and_get_errors( code, 'run' )
		self.assertTrue( errors, 'expected a compile error for the uncovered ErrorA leaf' )
		self.assertTrue( any( 'ErrorA' in e for e in errors ), errors )

	def test_user_defined_or_throw_is_rejected( self ) -> None:
		code = '\n'.join([
			'def or_throw() -> None:',
			'	pass',
		])
		self._import( code )
		self.assertTrue( any( "'or_throw' is reserved" in e for e in self.discovery.errors.errors ), self.discovery.errors.errors )

	def test_bare_except_is_rejected( self ) -> None:
		code = '\n'.join([
			'class ErrorA: pass',
			'',
			'def risky() -> Result[i32, ErrorA]:',
			'	return Result.Ok( 1 )',
			'',
			'def run() -> i32:',
			'	try:',
			'		v: i32 = risky().or_throw()',
			'		return v',
			'	except:',
			'		return -1',
		])
		errors = self._lower_and_get_errors( code, 'run' )
		self.assertTrue( any( 'bare' in e or 'except' in e for e in errors ), errors )

	def test_except_star_is_rejected( self ) -> None:
		code = '\n'.join([
			'class ErrorA: pass',
			'',
			'def run() -> i32:',
			'	try:',
			'		pass',
			'	except* ErrorA:',
			'		pass',
			'	return 0',
		])
		errors = self._lower_and_get_errors( code, 'run' )
		self.assertTrue( any( 'except*' in e for e in errors ), errors )

	def test_loop_escaping_break_inside_except_handler_is_rejected( self ) -> None:
		code = '\n'.join([
			'class ErrorA: pass',
			'',
			'def run() -> i32:',
			'	i: i32 = 0',
			'	while i < 3:',
			'		try:',
			'			pass',
			'		except ErrorA:',
			'			break',
			'		with compiler.wrap_arithmetic:',
			'			i = i + 1',
			'	return i',
		])
		errors = self._lower_and_get_errors( code, 'run' )
		self.assertTrue( any( 'loop' in e for e in errors ), errors )

	def test_try_inside_generator_body_is_rejected( self ) -> None:
		# generator bodies are lowered into a synthesized $$__next__ method,
		# only reached once something actually consumes gen() - compiling
		# gen() alone (never called) never schedules/lowers its own body at
		# all, so this needs a real main() driving it through
		# self.compiler.run(), unlike the other rejection tests above
		code = '\n'.join([
			'class ErrorA: pass',
			'',
			'def gen() -> Iterator[Result[i32,StopIteration]]:',
			'	try:',
			'		yield 1',
			'	except ErrorA:',
			'		yield 2',
			'',
			'def main() -> i32:',
			'	for x in gen():',
			'		pass',
			'	return 0',
		])
		self._import( code )
		self.compiler.run()
		errors = self.discovery.errors.errors
		self.assertTrue( any( 'generator' in e for e in errors ), errors )

	def test_nested_result_in_ok_position_is_rejected( self ) -> None:
		# Result[T,E] may never itself contain a Result (directly, or as a
		# union leaf) - see discovery.py's visit_Subscript. This isn't just
		# hygiene: it removes a real ambiguity for the general auto-
		# or_throw() machinery, where an unbound generic parameter's
		# argument being Result-shaped is now always safe to auto-unwrap,
		# since a caller can never have "genuinely meant" a nested Result.
		code = '\n'.join([
			'class ErrorA: pass',
			'',
			'def bad() -> Result[Result[i32, ErrorA], ErrorA]:',
			'	pass',
		])
		mod = self._import( code )
		fn = mod.get_local( 'bad' )
		if fn.resolve is not None:
			fn.resolve()
		self.assertTrue( any( 'nested Result' in e for e in self.discovery.errors.errors ), self.discovery.errors.errors )

	def test_nested_result_in_err_position_is_rejected( self ) -> None:
		code = '\n'.join([
			'class ErrorA: pass',
			'',
			'def bad() -> Result[i32, Result[i32, ErrorA]]:',
			'	pass',
		])
		mod = self._import( code )
		fn = mod.get_local( 'bad' )
		if fn.resolve is not None:
			fn.resolve()
		self.assertTrue( any( 'nested Result' in e for e in self.discovery.errors.errors ), self.discovery.errors.errors )

	def test_nested_result_as_union_leaf_is_rejected( self ) -> None:
		code = '\n'.join([
			'class ErrorA: pass',
			'',
			'def bad() -> Result[i32, ErrorA | Result[i32, ErrorA]]:',
			'	pass',
		])
		mod = self._import( code )
		fn = mod.get_local( 'bad' )
		if fn.resolve is not None:
			fn.resolve()
		self.assertTrue( any( 'nested Result' in e for e in self.discovery.errors.errors ), self.discovery.errors.errors )


if __name__ == '__main__':
	unittest.main()
