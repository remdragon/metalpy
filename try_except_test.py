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

_HANDLER_RC_LOCAL_REASSIGNED_ACROSS_TWO_HANDLERS = '''
class ErrorA:
	pass

class ErrorB:
	pass

class Box:
	n: i32

def risky( which: i32 ) -> Result[i32, ErrorA | ErrorB]:
	if which == 1:
		return Result.Err( ErrorA() )
	if which == 2:
		return Result.Err( ErrorB() )
	return Result.Ok( which )

def run( which: i32 ) -> i32:
	# b is declared BEFORE the try, then reassigned (a fresh Box each time)
	# in the try body AND in both handlers - exercises the chained
	# merge_if's OWNED-reconciliation path (each branch pushes its own
	# fresh epilogue entry for the same name) generalized past a single
	# if/else to N handlers
	b: Box = Box( n = -100 )
	try:
		v: i32 = risky( which ).or_throw()
		b = Box( n = v )
	except ErrorA:
		b = Box( n = -1 )
	except ErrorB:
		b = Box( n = -2 )
	return b.n

def main() -> i32:
	if run( 5 ) != 5:
		return 1
	if run( 1 ) != -1:
		return 2
	if run( 2 ) != -2:
		return 3
	return 0
'''

# Three-handler chaining stress test: a Widget is fresh-constructed on BOTH
# the try-body's own fallthrough path AND handler1's own path (same name,
# same OwnState - the ordinary "ownership agrees" merge at the FIRST
# pairwise merge_if call), but ABSENT on handler2/handler3's own paths. By
# the time handler2 is merged in (the SECOND pairwise call), the "combined"
# side already spans TWO distinct physical instruction blocks (the try
# body's own captured code AND handler1's own captured code) - either one
# could be the actual runtime path, so the teardown merge_if hands back for
# this now-only-on-one-side name must be broadcast onto BOTH, not just the
# most recently folded one. A missing broadcast would leak Widget on
# whichever physical block didn't get its own copy of the decref (silent
# under an ordinary run, but compounds into an observable leak/corruption
# across many repetitions - see the RC stress test below for that check).
_THREE_HANDLER_CHAIN_BROADCASTS_TEARDOWN_TO_ALL_PRIOR_BLOCKS = '''
class ErrorA:
	pass

class ErrorB:
	pass

class ErrorC:
	pass

class Widget:
	n: i32

def risky( which: i32 ) -> Result[i32, ErrorA | ErrorB | ErrorC]:
	if which == 1:
		return Result.Err( ErrorA() )
	if which == 2:
		return Result.Err( ErrorB() )
	if which == 3:
		return Result.Err( ErrorC() )
	return Result.Ok( which )

def run( which: i32 ) -> i32:
	result: i32 = -1000
	try:
		v: i32 = risky( which ).or_throw()
		w: Widget = Widget( n = v ) # fresh here...
		result = w.n
	except ErrorA:
		w = Widget( n = -1 ) # ...and fresh here too (same shape - no
		# re-annotation, this language's flat namespace only allows one
		# `w: Widget = ...` ever; a plain reassign reuses the type already
		# on record from the try-body's own declaration, still a genuinely
		# fresh CFG-level binding here since entry_snapshot has no `w`)
		result = w.n
	except ErrorB:
		result = -2 # no Widget at all on this path
	except ErrorC:
		result = -3 # nor this one
	return result

def main() -> i32:
	if run( 9 ) != 9:
		return 1
	if run( 1 ) != -1:
		return 2
	if run( 2 ) != -2:
		return 3
	if run( 3 ) != -3:
		return 4
	return 0
'''

# Nested try/except inside a handler body of an OUTER try - the outer's own
# _try_stack entry must be popped before the outer's handlers are lowered
# (already true - _try_stack push/pop only ever brackets the outer BODY),
# and the outer's new per-handler enter_branch/restore/merge_if machinery
# must not interfere with an entirely independent inner try lowered inside
# that same handler.
_NESTED_TRY_INSIDE_HANDLER_BODY = '''
class OuterError:
	pass

class InnerError:
	pass

def outer_risky( fail: bool ) -> Result[i32, OuterError]:
	if fail:
		return Result.Err( OuterError() )
	return Result.Ok( 1 )

def inner_risky( fail: bool ) -> Result[i32, InnerError]:
	if fail:
		return Result.Err( InnerError() )
	return Result.Ok( 2 )

def run( outer_fail: bool, inner_fail: bool ) -> i32:
	result: i32 = 0
	try:
		result = outer_risky( outer_fail ).or_throw()
	except OuterError:
		try:
			result = inner_risky( inner_fail ).or_throw()
		except InnerError:
			result = -1
	return result

def main() -> i32:
	if run( False, False ) != 1:
		return 1
	if run( True, False ) != 2:
		return 2
	if run( True, True ) != -1:
		return 3
	return 0
'''

_ALL_PARTS_TOGETHER = '''
class Counter:
	n: i32

def bump( c: Counter ) -> None:
	with compiler.wrap_arithmetic:
		c.n = c.n + 1

class SomeError:
	pass

def risky( bad: bool ) -> Result[i32, SomeError]:
	if bad:
		return Result.Err( SomeError() )
	return Result.Ok( 3 )

def run( c: Counter, bad: bool ) -> i32:
	result: i32 = 0
	try:
		v: i32 = risky( bad ).or_throw() # always runs
	except SomeError as e: # only runs if there's an unhandled Result
		result = -1
	else: # only runs if no except triggered
		result = v
	finally: # always runs, even on the early-return path below
		bump( c )
	return result

def run_early_return( c: Counter ) -> i32:
	try:
		return 42
	except SomeError as e:
		return -1
	finally:
		bump( c )

def main() -> i32:
	c: Counter = Counter( n = 0 )
	if run( c, False ) != 3:
		return 1
	if c.n != 1:
		return 2
	if run( c, True ) != -1:
		return 3
	if c.n != 2:
		return 4
	if run_early_return( c ) != 42:
		return 5
	if c.n != 3:
		return 6
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

	def test_all_parts_together_matches_python_shape( self ) -> None:
		# try body always runs; except only runs on an unhandled Result;
		# else only runs when no except fired; finally always runs, even on
		# an early return out of the try body
		self.assert_programs_run([ ( 'all_parts_together', _ALL_PARTS_TOGETHER ) ])

	def test_rc_local_reassigned_across_two_handlers( self ) -> None:
		self.assert_programs_run([ ( 'handler_rc_reassigned', _HANDLER_RC_LOCAL_REASSIGNED_ACROSS_TWO_HANDLERS ) ])

	def test_three_handler_chain_broadcasts_teardown_to_all_prior_blocks( self ) -> None:
		self.assert_programs_run([ ( 'three_handler_chain_broadcast', _THREE_HANDLER_CHAIN_BROADCASTS_TEARDOWN_TO_ALL_PRIOR_BLOCKS ) ])

	def test_nested_try_inside_handler_body( self ) -> None:
		self.assert_programs_run([ ( 'nested_try_inside_handler', _NESTED_TRY_INSIDE_HANDLER_BODY ) ])


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

	def test_return_inside_finally_is_rejected( self ) -> None:
		# Python's own well-known footgun: a return in finally silently
		# discards whatever the try/except was actually about to return -
		# rejected outright here rather than allowed
		code = '\n'.join([
			'def run() -> i32:',
			'	try:',
			'		pass',
			'	finally:',
			'		return 1',
		])
		errors = self._lower_and_get_errors( code, 'run' )
		self.assertTrue( any( 'finally' in e and 'return' in e for e in errors ), errors )

	def test_return_inside_nested_if_within_finally_is_rejected( self ) -> None:
		# the rejection recurses through nested if/while/for/with/try, not
		# just a bare top-level return statement
		code = '\n'.join([
			'def run( x: bool ) -> i32:',
			'	try:',
			'		pass',
			'	finally:',
			'		if x:',
			'			return 1',
			'	return 0',
		])
		errors = self._lower_and_get_errors( code, 'run' )
		self.assertTrue( any( 'finally' in e and 'return' in e for e in errors ), errors )

	def test_try_body_local_assigned_after_dispatch_point_unreadable_in_handler( self ) -> None:
		# CFG/RC integration regression (bug 1): `w` is assigned in the try
		# body AFTER the .or_throw() dispatch point that jumps into the
		# handler - the real emitted `goto` skips that assignment on the
		# handler's own path, so reading `w` from inside the handler must
		# be rejected, not silently accepted with a stale/uninitialized
		# value. Every handler is conservatively modeled as diverging from
		# the try's own ENTRY snapshot (before `v`/`w` exist at all), so
		# `w` is correctly never live there.
		code = '\n'.join([
			'class ErrorA: pass',
			'',
			'def risky( bad: bool ) -> Result[i32, ErrorA]:',
			'	if bad:',
			'		return Result.Err( ErrorA() )',
			'	return Result.Ok( 7 )',
			'',
			'def run( bad: bool ) -> i32:',
			'	try:',
			'		v: i32 = risky( bad ).or_throw()',
			'		w: i32 = v',
			'	except ErrorA:',
			'		return w',
			'	return w',
		])
		errors = self._lower_and_get_errors( code, 'run' )
		self.assertTrue( any( "'w' is not initialized on all code branches" in e for e in errors ), errors )

	def test_fresh_rc_local_after_dispatch_point_unreadable_past_handler( self ) -> None:
		# CFG/RC integration regression (bug 2): `f` is a fresh RC-class
		# local constructed in the try body AFTER the .or_throw() dispatch
		# point - only actually constructed on the Ok path, never on the
		# handler's own path. Reading f.n after the construct (a spot BOTH
		# the try-body-fallthrough and the (non-terminating) handler path
		# can reach) must be rejected - not silently accepted with a
		# release_object() on a never-constructed pointer at the shared
		# epilogue (the real heap-corruption hazard this whole integration
		# closes). Surfaces via the definite-assignment _live mechanism at
		# the read site, same as bug 1 above - merge_if itself is fine with
		# `f` being fresh-on-only-one-branch (a genuinely confined local),
		# it's the LATER read past the merge that's unsound.
		code = '\n'.join([
			'class ErrorA: pass',
			'',
			'class Foo:',
			'	n: i32',
			'',
			'def risky( bad: bool ) -> Result[i32, ErrorA]:',
			'	if bad:',
			'		return Result.Err( ErrorA() )',
			'	return Result.Ok( 7 )',
			'',
			'def run( bad: bool ) -> i32:',
			'	try:',
			'		v: i32 = risky( bad ).or_throw()',
			'		f: Foo = Foo( n = 1 )',
			'	except ErrorA:',
			'		pass',
			'	return f.n',
		])
		errors = self._lower_and_get_errors( code, 'run' )
		self.assertTrue( any( "'f' is not initialized on all code branches" in e for e in errors ), errors )

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


# --- RC stress: sound cross-handler fresh-construction, repeated many times -
#
# Same pattern or_return_rc_test.py's own _NAMED_VAR_OR_RETURN_REPEATED uses:
# loop the whole thing many times INSIDE the compiled program (not spawning
# the binary externally) - a leaked/over-released refcount compounds across
# iterations, more likely to surface as an observable failure even without a
# sanitizer build. Each branch also checks compiler.refcount() immediately
# after its own fresh construction (must be exactly 1 - no double-incref from
# a broadcast merge_if extra landing on more physical blocks than it should,
# no premature free/UAF from missing one it should have).

_RC_STRESS_ACROSS_HANDLERS = '''
class ErrorA:
	pass

class ErrorB:
	pass

class ErrorC:
	pass

class Widget:
	n: i32

def risky( which: i32 ) -> Result[i32, ErrorA | ErrorB | ErrorC]:
	if which == 1:
		return Result.Err( ErrorA() )
	if which == 2:
		return Result.Err( ErrorB() )
	if which == 3:
		return Result.Err( ErrorC() )
	return Result.Ok( which )

def run( which: i32 ) -> i32:
	result: i32 = -1000
	try:
		v: i32 = risky( which ).or_throw()
		w: Widget = Widget( n = v )
		if compiler.refcount( w ) != usize( 1 ):
			result = -999
		else:
			result = w.n
	except ErrorA:
		w = Widget( n = -1 ) # plain reassign - see the other 3-handler test's comment on why (no re-annotation)
		if compiler.refcount( w ) != usize( 1 ):
			result = -999
		else:
			result = w.n
	except ErrorB:
		result = -2
	except ErrorC:
		result = -3
	return result

def main() -> i32:
	with compiler.panic_arithmetic( 'unreachable: bounded loop counter' ):
		i: i32 = 0
		while i < 1000:
			which: i32 = i % 4
			r: i32 = run( which )
			if which == 0:
				if r != 0:
					return 1
			if which == 1:
				if r != -1:
					return 2
			if which == 2:
				if r != -2:
					return 3
			if which == 3:
				if r != -3:
					return 4
			i += 1
	return 0
'''


@unittest.skipUnless( test_support.HAS_CC, 'no C compiler (clang/gcc/msvc) found - skipping real-compile try/except RC stress test' )
class TryExceptRCStressTests( RealCompileMixin, unittest.TestCase ):
	def test_fresh_construction_across_handlers_repeated_no_leak_or_double_free( self ) -> None:
		self.assert_programs_run([ ( 'try_except_rc_stress', _RC_STRESS_ACROSS_HANDLERS ) ])


if __name__ == '__main__':
	unittest.main()
