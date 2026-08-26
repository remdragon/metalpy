# Real-compile-and-run + compile-error regression tests for limited
# try/except/else/finally + Result.or_throw() + `raise EXPR` (still not real
# exceptions/unwinding: control reaches an except handler only via
# `.or_throw()` on a Result-typed expression, or a `raise EXPR` statement,
# textually inside that try body, in the same function - a leaf uncovered by
# the innermost try walks every OUTER enclosing try too, innermost first,
# before propagating to the function's own return). See lowering.py's
# _stmt_Try/_emit_or_throw/_stmt_Raise and ir.OrThrow/ir.Raise for the
# implementation.
#
# The compile-error cases (rejection of user-defined or_throw, except*, bare
# except:, loop-escape, generator-body, insufficient function return-type
# coverage, bare/chained raise, dead except clauses) don't need a real C
# compiler at all - those use Discovery/Compiler directly, same pattern
# lowering_test.py's own rejection tests use. The positive/behavioral cases
# need a real compile+run (RealCompileMixin), same pattern or_return_rc_test.py
# uses.

from pathlib import Path
import unittest

from compiler import Compiler
from discovery import Discovery
import emitter_c
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

_NESTED_TRY_INNER_UNCOVERED_LEAF_CAUGHT_BY_OUTER = '''
class ErrorA:
	pass

def risky() -> Result[i32, ErrorA]:
	return Result.Err( ErrorA() )

def run() -> Result[i32, ErrorA]:
	# the INNER try has no handler for ErrorA at all - its or_throw() now
	# walks the WHOLE enclosing try stack (innermost first), so this leaf
	# IS caught by the OUTER try's own `except ErrorA:`, even though it's
	# only textually enclosing, not the innermost try
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
			return 0 if v == -1 else 1 # expected: caught by the outer handler
		case Result.Err( e ):
			return 2 # would wrongly fire if the outer handler had NOT caught it
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

# Regression: an except-clause bind (named `as e`, or the hidden per-handler
# value when there's no `as NAME`) that the handler body never re-raises,
# returns, or otherwise consumes used to leak unconditionally - it was only
# ever mark_live()'d (definite-assignment bookkeeping), never registered as
# an OWNED RC binding with its own decref-at-scope-exit epilogue entry (see
# cfg.py's declare_exception_bind). Covers three shapes: a single-leaf bind
# left unused, a multi-leaf tuple-except union bind left unused, and the
# no-`as NAME` hidden bind (still owns the payload even with nothing to
# name it). assert_programs_run's own leak-check output ("-- live RC objects
# (0) --") is what actually catches a regression here.
_UNUSED_EXCEPT_BIND_DOES_NOT_LEAK = '''
class ErrorA:
	code: i32

class ErrorB:
	code: i32

def risky( which: i32 ) -> Result[i32, ErrorA | ErrorB]:
	if which == 1:
		return Result.Err( ErrorA( code = 1 ) )
	if which == 2:
		return Result.Err( ErrorB( code = 2 ) )
	return Result.Ok( which )

def run_single_leaf_unused( which: i32 ) -> i32:
	try:
		return risky( which ).or_throw()
	except ErrorA as e:
		return -1
	except ErrorB as e:
		return -1

def run_tuple_leaf_unused( which: i32 ) -> i32:
	try:
		return risky( which ).or_throw()
	except ( ErrorA, ErrorB ) as e:
		return -2

def run_no_as_name_unused( which: i32 ) -> i32:
	try:
		return risky( which ).or_throw()
	except ErrorA:
		return -3
	except ErrorB:
		return -3

def main() -> i32:
	if run_single_leaf_unused( 1 ) != -1:
		return 1
	if run_single_leaf_unused( 0 ) != 0:
		return 2
	if run_tuple_leaf_unused( 1 ) != -2:
		return 3
	if run_tuple_leaf_unused( 2 ) != -2:
		return 4
	if run_no_as_name_unused( 1 ) != -3:
		return 5
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

# Change 1: 3-level nesting - the INNERMOST try's own uncovered leaf must
# walk PAST two enclosing stack frames (the immediate parent, which also
# doesn't cover it) to reach the OUTERMOST try's own handler.
_THREE_LEVEL_NESTED_TRY_WALKS_PAST_TWO_FRAMES = '''
class ErrorA:
	pass

class ErrorB:
	pass

def risky() -> Result[i32, ErrorA]:
	return Result.Err( ErrorA() )

def risky_b( bad: bool ) -> Result[i32, ErrorB]:
	if bad:
		return Result.Err( ErrorB() )
	return Result.Ok( 0 )

def run() -> i32:
	result: i32 = 0
	try: # outermost - the only one that covers ErrorA
		try: # middle - covers ErrorB (genuinely thrown here, just never on
			# THIS leaf) - irrelevant to the ErrorA leaf below, which must
			# walk PAST this frame too, not stop here just because this
			# frame has SOME handler
			ignored: i32 = risky_b( False ).or_throw()
			try: # innermost - covers nothing
				v: i32 = risky().or_throw()
				result = v
			finally:
				pass
		except ErrorB:
			result = -2
	except ErrorA:
		result = -1
	return result

def main() -> i32:
	if run() != -1:
		return 1
	return 0
'''

# Change 1: innermost-first priority - a leaf covered by BOTH an inner and
# an outer try's handlers must dispatch to the INNER one, never the outer.
# The outer handler is ALSO genuinely reachable (via `direct`), so it's a
# real handler, not accidentally dead code (Change 3) - it just must never
# fire when the leaf goes through the inner try instead.
_INNERMOST_TRY_WINS_WHEN_BOTH_COVER_SAME_LEAF = '''
class ErrorA:
	pass

def risky() -> Result[i32, ErrorA]:
	return Result.Err( ErrorA() )

def run( direct: bool ) -> i32:
	result: i32 = 0
	try:
		if direct:
			v: i32 = risky().or_throw() # outer's own handler covers this one directly
			result = v
		else:
			try:
				v2: i32 = risky().or_throw()
				result = v2
			except ErrorA:
				result = -1 # must fire - the innermost handler
	except ErrorA:
		result = -2 # must fire only when direct, never via the inner try's own leaf
	return result

def main() -> i32:
	if run( False ) != -1:
		return 1
	if run( True ) != -2:
		return 2
	return 0
'''

# Change 1: CFG/RC interaction - an RC-typed local constructed BEFORE the
# inner try, whose own uncovered leaf now jumps to the OUTER handler, which
# constructs a FRESH RC local only on that one path. Real compile+run,
# checking refcount() to confirm no leak/double-free.
_OUTER_HANDLER_FRESH_RC_LOCAL_VIA_INNER_UNCOVERED_LEAF = '''
class ErrorA:
	pass

class Box:
	n: i32

def risky() -> Result[i32, ErrorA]:
	return Result.Err( ErrorA() )

def run() -> i32:
	b: Box = Box( n = -100 )
	try:
		try:
			v: i32 = risky().or_throw() # uncovered by the inner try...
			b = Box( n = v )
		finally:
			pass
	except ErrorA:
		b = Box( n = -1 ) # ...caught here instead, a fresh Box only on this path
		if compiler.refcount( b ) != usize( 1 ):
			return -999
	return b.n

def main() -> i32:
	with compiler.wrap_arithmetic:
		i: i32 = 0
		while i < 100:
			if run() != -1:
				return 1
			i = i + 1
	return 0
'''

# Change 2: raise EXPR caught by a same-function except clause, single leaf.
_RAISE_CAUGHT_BY_SAME_FUNCTION_EXCEPT_SINGLE_LEAF = '''
class ErrorA:
	pass

def run( bad: bool ) -> i32:
	result: i32 = 0
	try:
		if bad:
			raise ErrorA()
		result = 5
	except ErrorA:
		result = -1
	return result

def main() -> i32:
	if run( False ) != 5:
		return 1
	if run( True ) != -1:
		return 2
	return 0
'''

# regression: raise EXPR's own raised value used to be released (via the
# ordinary end-of-statement pending-temp flush) BEFORE the dispatch below
# ever assigned it into `e` - a real use-after-free, not just a leak. `e`
# previously read freed memory here instead of the field it was just
# constructed with.
_RAISE_CAUGHT_BY_SAME_FUNCTION_EXCEPT_SINGLE_LEAF_FIELD_READABLE = '''
class Boom:
	code: i32
	def __init__( self, code: i32 ) -> None:
		self.code = code

def run() -> i32:
	try:
		raise Boom( code = 5 )
	except Boom as e:
		result: i32 = e.code
		compiler.decref( e )
		return result
	return 1

def main() -> i32:
	if run() != 5:
		return 1
	return 0
'''

# regression: an RC local declared in the try body (ABOVE the try's own
# entry snapshot), never explicitly released before a later covered raise
# dispatches past it, used to leak silently - a goto straight into a
# handler never unwound anything on its own (merge_if() deliberately
# doesn't clean up a terminating branch's own bindings - that's the
# terminator's own job, same as return_()/unwind_to() already do for
# return/break/continue). Fixed via ir.ThrowLeaf.epilogue/cfg.py's
# unwind_confined().
_TRY_BODY_LOCAL_RELEASED_BEFORE_COVERED_RAISE_DISPATCH = '''
class Guard:
	tag: i32

class Boom:
	tag: i32

def run( bad: bool ) -> i32:
	try:
		g: Guard = Guard( tag = 111 )
		if bad:
			raise Boom( tag = 222 )
		del g
	except Boom as e:
		del e
		return -1
	return 0

def main() -> i32:
	if run( False ) != 0:
		return 1
	if run( True ) != -1:
		return 2
	return 0
'''

# regression: an RC local declared BEFORE the try (not confined to it at
# all - the ordinary function-scope case) that's `del`'d on the try body's
# own non-raising fall-through used to double-release on the RAISING path -
# not a restore()/.cancelled bug (restore() already resyncs a survivor's own
# object identity back to the pristine, uncancelled entry_snapshot object
# correctly), but a merge_if() gap: when one branch of a construct
# terminates (the handler's own `return`), merge_if()'s terminating-branch
# shortcut only re-establishes whatever the SURVIVING branch's own end
# state still has - it never even looks at a name that was live entering
# the construct but is ABSENT from the survivor (del'd there). The
# terminating branch's own restore()-reverted, uncancelled entry for that
# name was therefore never reconciled at all, and leaked straight through
# into the merged post-construct state - build_epilogue_ladder() then
# released it a SECOND time. Fixed in cfg.py's merge_if(): the
# terminating-branch path now also walks entry_bindings for names missing
# from the survivor, neutralizing (flag-guarding, since the handler's own
# `return` already captured this entry's shared label) whatever stale
# object the terminating branch's restore() left behind.
_LOCAL_BEFORE_TRY_SURVIVES_ACROSS_REPEATED_CALLS_REGARDLESS_OF_ORDER = '''
class Guard:
	tag: i32

class Boom:
	tag: i32

def run( bad: bool ) -> i32:
	g: Guard = Guard( tag = 111 )
	try:
		if bad:
			raise Boom( tag = 222 )
		del g
	except Boom as e:
		del e
		return -1
	return 0

def main() -> i32:
	if run( False ) != 0:
		return 1
	if run( True ) != -1:
		return 2
	return 0
'''

# Change 2: raise EXPR caught by a tuple except-clause, `as e:` binding.
_RAISE_CAUGHT_BY_TUPLE_EXCEPT_CLAUSE_BINDING = '''
class ErrorA:
	pass

class ErrorB:
	pass

def run( which: i32 ) -> i32:
	result: i32 = 0
	try:
		if which == 1:
			raise ErrorA()
		if which == 2:
			raise ErrorB()
		result = 5
	except (ErrorA, ErrorB) as e:
		compiler.decref( e )
		with compiler.wrap_arithmetic:
			result = which * 11
	return result

def main() -> i32:
	if run( 0 ) != 5:
		return 1
	if run( 1 ) != 11:
		return 2
	if run( 2 ) != 22:
		return 3
	return 0
'''

# Change 2: THE key test - a single `raise` of a union-typed value where SOME
# leaves are caught locally and others propagate via a real function return.
# Exercises the per-leaf split within one raise statement.
_RAISE_MIXED_COVERED_AND_UNCOVERED_LEAVES_SPLITS_PER_LEAF = '''
class ErrorA:
	pass

class ErrorB:
	pass

def pick( which: i32 ) -> ErrorA | ErrorB:
	if which == 1:
		return ErrorA()
	return ErrorB()

def run( which: i32 ) -> Result[i32, ErrorB]:
	try:
		raise pick( which )
	except ErrorA:
		return Result.Ok( -1 ) # covered locally - no Result ever built for this leaf

def main() -> i32:
	match run( 1 ):
		case Result.Ok( v ):
			if v != -1:
				return 1
		case Result.Err( e ):
			return 2 # would wrongly fire - ErrorA is covered
	match run( 2 ):
		case Result.Ok( v ):
			return 3 # would wrongly fire - ErrorB is uncovered, must propagate
		case Result.Err( e ):
			pass
	return 0
'''

# Change 2: raise with no enclosing try at all - degrades to plain
# function-return propagation, mirroring or_throw()'s own identical case.
_RAISE_WITH_NO_ENCLOSING_TRY_IS_LIKE_OR_RETURN = '''
class ErrorA:
	pass

def run( bad: bool ) -> Result[i32, ErrorA]:
	if bad:
		raise ErrorA()
	return Result.Ok( 9 )

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

# Change 2: raise AND or_throw() in the same try body, both dispatching to
# the SAME handler.
_RAISE_AND_OR_THROW_SAME_TRY_DISPATCH_TO_SAME_HANDLER = '''
class ErrorA:
	pass

def risky( bad: bool ) -> Result[i32, ErrorA]:
	if bad:
		return Result.Err( ErrorA() )
	return Result.Ok( 4 )

def run( which: i32 ) -> i32:
	result: i32 = 0
	try:
		if which == 1:
			raise ErrorA()
		result = risky( which == 2 ).or_throw()
	except ErrorA:
		result = -1
	return result

def main() -> i32:
	if run( 0 ) != 4:
		return 1
	if run( 1 ) != -1:
		return 2
	if run( 2 ) != -1:
		return 3
	return 0
'''

# Bare `raise` (re-raise): logs then re-raises the handler's own caught
# value, caught by an OUTER try's matching handler.
_BARE_RAISE_RERAISE_TO_OUTER_HANDLER = '''
class ErrorA:
	code: i32

def risky( bad: bool ) -> Result[i32, ErrorA]:
	if bad:
		return Result.Err( ErrorA( code = 7 ) )
	return Result.Ok( 1 )

def run( bad: bool ) -> i32:
	result: i32 = 0
	try:
		try:
			result = risky( bad ).or_throw()
		except ErrorA as e:
			result = e.code # "log" it
			raise
	except ErrorA as e2:
		with compiler.wrap_arithmetic:
			result = e2.code * 100
	return result

def main() -> i32:
	if run( False ) != 1:
		return 1
	if run( True ) != 700:
		return 2
	return 0
'''

# Bare `raise` with NO outer try at all - must propagate to the enclosing
# function's own return, same as a real `raise EXPR` with no enclosing try.
_BARE_RAISE_RERAISE_PROPAGATES_TO_FUNCTION_RETURN = '''
class ErrorA:
	code: i32

def risky( bad: bool ) -> Result[i32, ErrorA]:
	if bad:
		return Result.Err( ErrorA( code = 9 ) )
	return Result.Ok( 2 )

def run( bad: bool ) -> Result[i32, ErrorA]:
	try:
		v: i32 = risky( bad ).or_throw()
		return Result.Ok( v )
	except ErrorA as e:
		raise # nothing else covers ErrorA - propagates to run()'s own return

def main() -> i32:
	match run( True ):
		case Result.Ok( v ):
			return 1
		case Result.Err( e ):
			if e.code != 9:
				return 2
	match run( False ):
		case Result.Ok( v ):
			if v != 2:
				return 3
		case Result.Err( e ):
			return 4
	return 0
'''

# Bare `raise` escapes to the OUTER try's own matching handler one level at a
# time - proves _try_stack's own pop-before-handler-lowering (see _stmt_Try)
# correctly excludes the CURRENT try's own siblings from a re-raise's
# dispatch: the only handler a re-raised ErrorA can reach here is the
# OUTER try's, never a (nonsensical, impossible) loop back into its own try.
_BARE_RAISE_ESCAPES_TO_OUTER_MATCHING_HANDLER_ONE_LEVEL_AT_A_TIME = '''
class ErrorA:
	code: i32

def risky() -> Result[i32, ErrorA]:
	return Result.Err( ErrorA( code = 3 ) )

def run() -> i32:
	result: i32 = 0
	try:
		try:
			v: i32 = risky().or_throw()
			result = v
		except ErrorA as e:
			result = e.code
			raise
	except ErrorA as e2:
		with compiler.wrap_arithmetic:
			result = e2.code + 1000
	return result

def main() -> i32:
	if run() != 1003:
		return 1
	return 0
'''

# Bare `raise` inside `except SomeError:` with NO `as NAME` at all - exercises
# the hidden, compiler-synthesized bind variable (TryHandler.raise_value_var)
# that every handler now gets regardless of whether the user named one.
_BARE_RAISE_WITH_NO_AS_NAME_STILL_WORKS = '''
class ErrorA:
	pass

def risky( bad: bool ) -> Result[i32, ErrorA]:
	if bad:
		return Result.Err( ErrorA() )
	return Result.Ok( 4 )

def run( bad: bool ) -> i32:
	result: i32 = 0
	try:
		try:
			result = risky( bad ).or_throw()
		except ErrorA:
			raise
	except ErrorA:
		result = -1
	return result

def main() -> i32:
	if run( False ) != 4:
		return 1
	if run( True ) != -1:
		return 2
	return 0
'''

# Nested-handler shadowing: a handler body containing its OWN nested
# try/except, whose handler ALSO does a bare `raise` - must re-raise ITS OWN
# caught value (ErrorB), never the enclosing handler's (ErrorA) - proves
# _active_raise_values is a real stack, not a single slot.
_BARE_RAISE_NESTED_HANDLER_SHADOWS_OUTER_OWN_VALUE = '''
class ErrorA:
	code: i32

class ErrorB:
	code: i32

def risky_a() -> Result[i32, ErrorA]:
	return Result.Err( ErrorA( code = 1 ) )

def risky_b( bad: bool ) -> Result[i32, ErrorB]:
	if bad:
		return Result.Err( ErrorB( code = 2 ) )
	return Result.Ok( 0 )

def run( inner_bad: bool ) -> i32:
	result: i32 = 0
	try:
		try:
			v: i32 = risky_a().or_throw()
			result = v
		except ErrorA as e:
			try:
				result = risky_b( inner_bad ).or_throw()
			except ErrorB as e2:
				raise # must re-raise e2 (ErrorB), NOT e (ErrorA)
			raise # only reached if risky_b succeeded - re-raises e (ErrorA)
	except ErrorA as outer_e:
		with compiler.wrap_arithmetic:
			result = 100 + outer_e.code
	except ErrorB as outer_e2:
		with compiler.wrap_arithmetic:
			result = 200 + outer_e2.code
	return result

def main() -> i32:
	if run( True ) != 202:
		return 1
	if run( False ) != 101:
		return 2
	return 0
'''

# Change 3: a tuple except-clause where only ONE of two leaves is ever
# thrown - the whole clause must still NOT be flagged dead.
_TUPLE_EXCEPT_CLAUSE_ONE_LEAF_NEVER_THROWN_STILL_NOT_DEAD = '''
class ErrorA:
	pass

class ErrorB:
	pass

def run() -> i32:
	result: i32 = 0
	try:
		raise ErrorA() # ErrorB is never thrown anywhere in this body
	except (ErrorA, ErrorB) as e:
		compiler.decref( e )
		result = -1
	return result

def main() -> i32:
	if run() != -1:
		return 1
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
	# plain try/finally, no except clause at all - nothing in this body
	# ever throws, so an `except SomeError:` here would be flagged as
	# unreachable dead code (Change 3) - this fixture only cares about
	# finally running on the early-return path, not dispatch
	try:
		return 42
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


# --- @inline multi-statement splice prelude - or_throw()/raise generalization
# --- (PLAN_INLINE.md's own inline_exit shape, now also covering ir.OrThrow/
# --- ir.Raise, not just OrReturn/OrJump) -------------------------------------

_OR_THROW_INSIDE_INLINE_SPLICE_COVERED_BY_CALLER_TRY = '''
class ErrorA:
	pass

def risky( bad: bool ) -> Result[i32, ErrorA]:
	if bad:
		return Result.Err( ErrorA() )
	return Result.Ok( 7 )

@cstruct
class Adder:
	base: i32

	@inline
	def add_checked( self, bad: bool ) -> i32:
		v: i32 = risky( bad ).or_throw() # covered by the CALL SITE's own enclosing try below
		result: i32 = 0
		with compiler.wrap_arithmetic:
			result = self.base + v
		return result

def main() -> i32:
	a: Adder = Adder( base = 100 )
	try:
		if a.add_checked( False ) != 107:
			return 1
	except ErrorA:
		return 2
	try:
		x: i32 = a.add_checked( True )
		return 3
	except ErrorA:
		pass # dispatches straight to this handler - splicing must not corrupt main()'s own frame
	return 0
'''

_OR_THROW_INSIDE_INLINE_SPLICE_UNCOVERED_PROPAGATES_TO_CALLER_RETURN = '''
class ErrorA:
	pass

def risky( bad: bool ) -> Result[i32, ErrorA]:
	if bad:
		return Result.Err( ErrorA() )
	return Result.Ok( 7 )

@cstruct
class Adder:
	base: i32

	@inline
	def add_checked( self, bad: bool ) -> Result[i32, ErrorA]:
		# no enclosing try at either call site below - ErrorA is uncovered,
		# must propagate straight through the splice into add_checked's own
		# Result[i32,ErrorA] value at the call site - the pre-fix latent bug
		# (return_slot wired to the CALLER's own return_value_var) would
		# have corrupted main()'s own i32 return slot instead
		v: i32 = risky( bad ).or_throw()
		result: i32 = 0
		with compiler.wrap_arithmetic: # must be SKIPPED entirely on the error path
			result = self.base + v
		return Result.Ok( result )

def main() -> i32:
	a: Adder = Adder( base = 100 )
	match a.add_checked( False ):
		case Result.Ok( v ):
			if v != 107:
				return 1
		case Result.Err( e ):
			return 2
	match a.add_checked( True ):
		case Result.Ok( v ):
			return 3
		case Result.Err( e ):
			pass
	return 0
'''

_RAISE_INSIDE_INLINE_SPLICE_COVERED_BY_CALLER_TRY = '''
class ErrorA:
	pass

@cstruct
class Counter:
	value: i32

	@inline
	def bumped( self, by: i32 ) -> i32:
		if by < 0:
			raise ErrorA()
		result: i32 = 0
		with compiler.wrap_arithmetic:
			result = self.value + by
		return result

def main() -> i32:
	c: Counter = Counter( value = 10 )
	try:
		if c.bumped( 5 ) != 15:
			return 1
	except ErrorA:
		return 2
	try:
		x: i32 = c.bumped( -3 )
		return 3
	except ErrorA:
		pass
	return 0
'''

_RAISE_INSIDE_INLINE_SPLICE_UNCOVERED_PROPAGATES_TO_CALLER_RETURN = '''
class ErrorA:
	pass

@cstruct
class Counter:
	value: i32

	@inline
	def bumped_raise( self, by: i32 ) -> Result[i32, ErrorA]:
		if by < 0:
			raise ErrorA() # uncovered at either call site below
		result: i32 = 0
		with compiler.wrap_arithmetic:
			result = self.value + by
		return Result.Ok( result )

def main() -> i32:
	c: Counter = Counter( value = 10 )
	match c.bumped_raise( 5 ):
		case Result.Ok( v ):
			if v != 15:
				return 1
		case Result.Err( e ):
			return 2
	match c.bumped_raise( -3 ):
		case Result.Ok( v ):
			return 3
		case Result.Err( e ):
			pass
	return 0
'''

# mixed union error type - some leaves covered by the call site's own
# enclosing try, others not - proves the per-leaf split (already correct,
# untouched) still composes correctly with the new inline_exit wiring on
# just the uncovered side.
#
# NOTE: the covered-leaf call site below deliberately reads the Result via
# a receiver-position method call (.is_err()) rather than binding it to a
# name or `match`-ing it directly inside the try body - doing either of
# those surfaced a SEPARATE, pre-existing cfg.py bug (a named/match-bound
# Result-shaped local declared inside a try body's own top-level statements
# loses its epilogue entry to the try's own branch-confinement restore,
# leaving stray RC-cleanup temps referencing declarations that never make
# it into the emitted C - confirmed with a minimal repro using no @inline/
# or_throw/raise at all, so unrelated to this task's own change; flagged
# separately, not fixed here per this task's own "don't touch cfg.py"
# scope). A bare receiver expression never gets a NAMED epilogue entry (it's
# cleaned up via the ordinary Temp/DeleteTemp path instead), sidestepping it.
_MIXED_UNION_OR_THROW_INSIDE_INLINE_SPLICE_PARTIAL_COVERAGE = '''
class ErrorA:
	pass

class ErrorB:
	pass

def risky2( which: i32 ) -> Result[i32, ErrorA | ErrorB]:
	if which == 1:
		return Result.Err( ErrorA() )
	if which == 2:
		return Result.Err( ErrorB() )
	return Result.Ok( which )

@cstruct
class Picker:
	base: i32

	@inline
	def pick_checked( self, which: i32 ) -> Result[i32, ErrorA | ErrorB]:
		v: i32 = risky2( which ).or_throw() # ErrorA covered only at the try-wrapped call site below; both leaves uncovered at the other two, needing this declared union
		result: i32 = 0
		with compiler.wrap_arithmetic:
			result = self.base + v
		return Result.Ok( result )

def main() -> i32:
	p: Picker = Picker( base = 100 )
	code: i32 = 0
	try:
		# runtime: risky2(1) always hits ErrorA, dispatched straight to the
		# handler below from INSIDE the splice - code stays 0, the is_err()
		# check below never even runs
		if p.pick_checked( 1 ).is_err():
			code = 2
		else:
			code = 1
	except ErrorA:
		pass
	if code != 0:
		result: i32 = 0
		with compiler.wrap_arithmetic:
			result = 100 + code
		return result # would only fire if ErrorA weren't actually dispatched above

	# no enclosing try here - both leaves uncovered, must propagate through
	# the splice into pick_checked's own declared Result[i32,ErrorA|ErrorB]
	match p.pick_checked( 2 ): # runtime: hits ErrorB
		case Result.Ok( v ):
			return 1
		case Result.Err( e ):
			pass

	match p.pick_checked( 0 ): # runtime: Ok
		case Result.Ok( v ):
			if v != 100:
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

	def test_nested_try_inner_uncovered_leaf_caught_by_outer( self ) -> None:
		self.assert_programs_run([ ( 'nested_try_scope', _NESTED_TRY_INNER_UNCOVERED_LEAF_CAUGHT_BY_OUTER ) ])

	def test_or_throw_with_no_enclosing_try_is_like_or_return( self ) -> None:
		self.assert_programs_run([ ( 'or_throw_no_try', _OR_THROW_WITH_NO_ENCLOSING_TRY_IS_LIKE_OR_RETURN ) ])

	def test_all_parts_together_matches_python_shape( self ) -> None:
		# try body always runs; except only runs on an unhandled Result;
		# else only runs when no except fired; finally always runs, even on
		# an early return out of the try body
		self.assert_programs_run([ ( 'all_parts_together', _ALL_PARTS_TOGETHER ) ])

	def test_rc_local_reassigned_across_two_handlers( self ) -> None:
		self.assert_programs_run([ ( 'handler_rc_reassigned', _HANDLER_RC_LOCAL_REASSIGNED_ACROSS_TWO_HANDLERS ) ])

	def test_unused_except_bind_does_not_leak( self ) -> None:
		# relies on the debug build's own automatic leak-check report
		# (_split_off_leak_report), not assert_programs_run's plain exit-code
		# check (see ClosureCallResultFreshTempTests' own identical note in
		# or_return_rc_test.py) - the original bug (declare_exception_bind's
		# own docstring) never crashed or produced a wrong exit code, only a
		# permanently-live except-bind object at teardown.
		compiler = self._compile_source( _UNUSED_EXCEPT_BIND_DOES_NOT_LEAK )
		result = self._build_and_run( compiler, emitter_c.emit_c( compiler ), None )
		self.assertEqual( result.returncode, 0, f'exe exited {result.returncode} (stderr: {result.stderr})' )
		self._split_off_leak_report( result.stdout )

	def test_three_handler_chain_broadcasts_teardown_to_all_prior_blocks( self ) -> None:
		self.assert_programs_run([ ( 'three_handler_chain_broadcast', _THREE_HANDLER_CHAIN_BROADCASTS_TEARDOWN_TO_ALL_PRIOR_BLOCKS ) ])

	def test_nested_try_inside_handler_body( self ) -> None:
		self.assert_programs_run([ ( 'nested_try_inside_handler', _NESTED_TRY_INSIDE_HANDLER_BODY ) ])

	def test_three_level_nested_try_walks_past_two_frames( self ) -> None:
		self.assert_programs_run([ ( 'three_level_nested', _THREE_LEVEL_NESTED_TRY_WALKS_PAST_TWO_FRAMES ) ])

	def test_innermost_try_wins_when_both_cover_same_leaf( self ) -> None:
		self.assert_programs_run([ ( 'innermost_wins', _INNERMOST_TRY_WINS_WHEN_BOTH_COVER_SAME_LEAF ) ])

	def test_outer_handler_fresh_rc_local_via_inner_uncovered_leaf( self ) -> None:
		self.assert_programs_run([ ( 'outer_handler_fresh_rc', _OUTER_HANDLER_FRESH_RC_LOCAL_VIA_INNER_UNCOVERED_LEAF ) ])

	def test_raise_caught_by_same_function_except_single_leaf( self ) -> None:
		self.assert_programs_run([ ( 'raise_single_leaf', _RAISE_CAUGHT_BY_SAME_FUNCTION_EXCEPT_SINGLE_LEAF ) ])

	def test_raise_caught_by_same_function_except_single_leaf_field_readable( self ) -> None:
		self.assert_programs_run([ ( 'raise_single_leaf_field', _RAISE_CAUGHT_BY_SAME_FUNCTION_EXCEPT_SINGLE_LEAF_FIELD_READABLE ) ])

	def test_try_body_local_released_before_covered_raise_dispatch( self ) -> None:
		self.assert_programs_run([ ( 'raise_try_body_local', _TRY_BODY_LOCAL_RELEASED_BEFORE_COVERED_RAISE_DISPATCH ) ])

	def test_local_before_try_survives_across_repeated_calls_regardless_of_order( self ) -> None:
		self.assert_programs_run([ ( 'local_before_try_repeated', _LOCAL_BEFORE_TRY_SURVIVES_ACROSS_REPEATED_CALLS_REGARDLESS_OF_ORDER ) ])

	def test_raise_caught_by_tuple_except_clause_binding( self ) -> None:
		self.assert_programs_run([ ( 'raise_tuple_except', _RAISE_CAUGHT_BY_TUPLE_EXCEPT_CLAUSE_BINDING ) ])

	def test_raise_mixed_covered_and_uncovered_leaves_splits_per_leaf( self ) -> None:
		self.assert_programs_run([ ( 'raise_mixed_leaves', _RAISE_MIXED_COVERED_AND_UNCOVERED_LEAVES_SPLITS_PER_LEAF ) ])

	def test_raise_with_no_enclosing_try_is_like_or_return( self ) -> None:
		self.assert_programs_run([ ( 'raise_no_try', _RAISE_WITH_NO_ENCLOSING_TRY_IS_LIKE_OR_RETURN ) ])

	def test_raise_and_or_throw_same_try_dispatch_to_same_handler( self ) -> None:
		self.assert_programs_run([ ( 'raise_and_or_throw', _RAISE_AND_OR_THROW_SAME_TRY_DISPATCH_TO_SAME_HANDLER ) ])

	def test_bare_raise_reraise_to_outer_handler( self ) -> None:
		self.assert_programs_run([ ( 'bare_raise_outer', _BARE_RAISE_RERAISE_TO_OUTER_HANDLER ) ])

	def test_bare_raise_reraise_propagates_to_function_return( self ) -> None:
		self.assert_programs_run([ ( 'bare_raise_fn_return', _BARE_RAISE_RERAISE_PROPAGATES_TO_FUNCTION_RETURN ) ])

	def test_bare_raise_escapes_to_outer_matching_handler_one_level_at_a_time( self ) -> None:
		self.assert_programs_run([ ( 'bare_raise_one_level', _BARE_RAISE_ESCAPES_TO_OUTER_MATCHING_HANDLER_ONE_LEVEL_AT_A_TIME ) ])

	def test_bare_raise_with_no_as_name_still_works( self ) -> None:
		self.assert_programs_run([ ( 'bare_raise_no_as_name', _BARE_RAISE_WITH_NO_AS_NAME_STILL_WORKS ) ])

	def test_bare_raise_nested_handler_shadows_outer_own_value( self ) -> None:
		self.assert_programs_run([ ( 'bare_raise_nested_shadow', _BARE_RAISE_NESTED_HANDLER_SHADOWS_OUTER_OWN_VALUE ) ])

	def test_tuple_except_clause_one_leaf_never_thrown_still_not_dead( self ) -> None:
		self.assert_programs_run([ ( 'tuple_except_one_leaf', _TUPLE_EXCEPT_CLAUSE_ONE_LEAF_NEVER_THROWN_STILL_NOT_DEAD ) ])

	def test_or_throw_inside_inline_splice_covered_by_caller_try( self ) -> None:
		self.assert_programs_run([ ( 'or_throw_inline_covered', _OR_THROW_INSIDE_INLINE_SPLICE_COVERED_BY_CALLER_TRY ) ])

	def test_or_throw_inside_inline_splice_uncovered_propagates_to_caller_return( self ) -> None:
		self.assert_programs_run([ ( 'or_throw_inline_uncovered', _OR_THROW_INSIDE_INLINE_SPLICE_UNCOVERED_PROPAGATES_TO_CALLER_RETURN ) ])

	def test_raise_inside_inline_splice_covered_by_caller_try( self ) -> None:
		self.assert_programs_run([ ( 'raise_inline_covered', _RAISE_INSIDE_INLINE_SPLICE_COVERED_BY_CALLER_TRY ) ])

	def test_raise_inside_inline_splice_uncovered_propagates_to_caller_return( self ) -> None:
		self.assert_programs_run([ ( 'raise_inline_uncovered', _RAISE_INSIDE_INLINE_SPLICE_UNCOVERED_PROPAGATES_TO_CALLER_RETURN ) ])

	def test_mixed_union_or_throw_inside_inline_splice_partial_coverage( self ) -> None:
		self.assert_programs_run([ ( 'or_throw_inline_mixed', _MIXED_UNION_OR_THROW_INSIDE_INLINE_SPLICE_PARTIAL_COVERAGE ) ])


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

	def test_try_inside_generator_body_rejection_error_is_a_single_line( self ) -> None:
		# regression: this diagnostic (and several sibling _stmt_Try ones)
		# used to unparse the WHOLE ast.Try node (including its entire
		# body/handlers/finally) instead of just naming the problem - a real
		# repro (a multi-statement try body) produced a multi-line error
		# message that buried the actual problem
		code = '\n'.join([
			'class ErrorA: pass',
			'',
			'def gen() -> Iterator[Result[i32,StopIteration]]:',
			'	try:',
			'		x: i32 = 1',
			'		y: i32 = 2',
			'		z: i32 = 3',
			'		yield x + y + z',
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
		self.assertTrue( errors )
		for error in errors:
			self.assertNotIn( '\n', error )

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

	# --- Change 2: raise EXPR rejection coverage ----------------------------

	def test_bare_raise_inside_try_body_not_a_handler_is_rejected( self ) -> None:
		# the try BODY, not a handler's own body - _active_raise_values is
		# empty there even though a handler exists right below it
		code = '\n'.join([
			'class ErrorA: pass',
			'',
			'def run() -> i32:',
			'	try:',
			'		raise',
			'	except ErrorA:', # unused on purpose; the bare `raise` itself fails first
			'		pass',
			'	return 0',
		])
		errors = self._lower_and_get_errors( code, 'run' )
		self.assertTrue( any( 'bare raise' in e for e in errors ), errors )
		self.assertTrue( any( 'only valid inside an except handler' in e for e in errors ), errors )

	def test_bare_raise_at_function_top_level_is_rejected( self ) -> None:
		# not inside a try/except at all
		code = '\n'.join([
			'def run() -> i32:',
			'	raise',
			'	return 0',
		])
		errors = self._lower_and_get_errors( code, 'run' )
		self.assertTrue( any( 'only valid inside an except handler' in e for e in errors ), errors )

	def test_bare_raise_inside_else_block_is_rejected( self ) -> None:
		code = '\n'.join([
			'class ErrorA: pass',
			'',
			'def risky() -> Result[i32, ErrorA]:',
			'	return Result.Ok( 1 )',
			'',
			'def run() -> i32:',
			'	try:',
			'		v: i32 = risky().or_throw()',
			'	except ErrorA:',
			'		return -1',
			'	else:',
			'		raise',
			'	return 0',
		])
		errors = self._lower_and_get_errors( code, 'run' )
		self.assertTrue( any( 'only valid inside an except handler' in e for e in errors ), errors )

	def test_bare_raise_inside_finally_block_is_rejected( self ) -> None:
		code = '\n'.join([
			'class ErrorA: pass',
			'',
			'def risky() -> Result[i32, ErrorA]:',
			'	return Result.Ok( 1 )',
			'',
			'def run() -> i32:',
			'	try:',
			'		v: i32 = risky().or_throw()',
			'	except ErrorA:',
			'		return -1',
			'	finally:',
			'		raise',
			'	return 0',
		])
		errors = self._lower_and_get_errors( code, 'run' )
		self.assertTrue( any( 'only valid inside an except handler' in e for e in errors ), errors )

	def test_raise_from_is_rejected( self ) -> None:
		code = '\n'.join([
			'class ErrorA: pass',
			'class ErrorB: pass',
			'',
			'def run() -> i32:',
			'	raise ErrorA() from ErrorB()',
		])
		errors = self._lower_and_get_errors( code, 'run' )
		self.assertTrue( any( 'from' in e and 'chaining' in e for e in errors ), errors )

	def test_raise_inside_generator_body_is_rejected( self ) -> None:
		# same reasoning as test_try_inside_generator_body_is_rejected -
		# gen()'s body is only lowered once something actually drives it
		code = '\n'.join([
			'class ErrorA: pass',
			'',
			'def gen() -> Iterator[Result[i32,StopIteration]]:',
			'	yield 1',
			'	raise ErrorA()',
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

	def test_raise_inside_inline_splice_prelude_uncovered_still_requires_result_return( self ) -> None:
		# @inline splice prelude no longer rejects raise/or_throw outright
		# (see try_except_test.py's TryExceptRealCompileTests own real-
		# compile coverage for the positive path) - but the ordinary
		# or_throw()/raise coverage requirement still applies: an uncovered
		# leaf still needs the INLINE TARGET's own declared return type to
		# be a covering Result[_,_] (bumped here stays plain i32), same as
		# it would for a non-spliced function
		code = '\n'.join([
			'class ErrorA: pass',
			'',
			'@cstruct',
			'class Counter:',
			'	value: i32',
			'',
			'	@inline',
			'	def bumped( self, by: i32 ) -> i32:',
			'		if by < 0:',
			'			raise ErrorA()',
			'		result: i32 = self.value + by',
			'		return result',
			'',
			'def main() -> i32:',
			'	c: Counter = Counter( value = 10 )',
			'	return c.bumped( 5 )',
		])
		errors = self._lower_and_get_errors( code, 'main' )
		self.assertTrue( any( 'ErrorA' in e and 'Result' in e for e in errors ), errors )

	def test_raise_uncovered_leaf_with_insufficient_function_return_is_a_compile_error( self ) -> None:
		code = '\n'.join([
			'class ErrorA: pass',
			'class ErrorB: pass',
			'',
			'def pick( which: i32 ) -> ErrorA | ErrorB:',
			'	if which == 1:',
			'		return ErrorA()',
			'	return ErrorB()',
			'',
			'def run( which: i32 ) -> i32:',
			'	try:',
			'		raise pick( which )',
			'	except ErrorB:',
			'		return -1',
			'	return 0',
		])
		errors = self._lower_and_get_errors( code, 'run' )
		self.assertTrue( errors, 'expected a compile error for the uncovered ErrorA leaf' )
		self.assertTrue( any( 'ErrorA' in e for e in errors ), errors )

	# --- Change 3: dead (never-thrown) except clause ------------------------

	def test_never_thrown_except_clause_is_a_compile_error( self ) -> None:
		code = '\n'.join([
			'class ErrorA: pass',
			'class ErrorB: pass',
			'',
			'def risky() -> Result[i32, ErrorA]:',
			'	return Result.Ok( 1 )',
			'',
			'def run() -> i32:',
			'	try:',
			'		v: i32 = risky().or_throw()',
			'	except ErrorA:',
			'		v = -1',
			'	except ErrorB:', # never thrown anywhere in this try body
			'		v = -2',
			'	return v',
		])
		errors = self._lower_and_get_errors( code, 'run' )
		self.assertTrue( any( 'unreachable' in e and 'ErrorB' in e for e in errors ), errors )

	def test_except_clause_matched_only_via_or_throw_is_not_dead( self ) -> None:
		code = '\n'.join([
			'class ErrorA: pass',
			'',
			'def risky() -> Result[i32, ErrorA]:',
			'	return Result.Err( ErrorA() )',
			'',
			'def run() -> i32:',
			'	try:',
			'		v: i32 = risky().or_throw()',
			'	except ErrorA:',
			'		v = -1',
			'	return v',
		])
		errors = self._lower_and_get_errors( code, 'run' )
		self.assertFalse( errors, errors )

	def test_except_clause_matched_only_via_raise_is_not_dead( self ) -> None:
		code = '\n'.join([
			'class ErrorA: pass',
			'',
			'def run() -> i32:',
			'	try:',
			'		raise ErrorA()',
			'	except ErrorA:',
			'		return -1',
			'	return 0',
		])
		errors = self._lower_and_get_errors( code, 'run' )
		self.assertFalse( errors, errors )

	def test_except_clause_matched_only_via_inner_nested_try_bubbling_up_is_not_dead( self ) -> None:
		# the direct Change-1 interaction: the OUTER except clause is never
		# dispatched to by anything textually inside the outer try's own
		# body directly - only via the INNER try's own uncovered leaf
		# bubbling up (Change 1) - must still not be flagged dead (Change 3)
		code = '\n'.join([
			'class ErrorA: pass',
			'',
			'def risky() -> Result[i32, ErrorA]:',
			'	return Result.Err( ErrorA() )',
			'',
			'def run() -> i32:',
			'	result: i32 = 0',
			'	try:',
			'		try:',
			'			v: i32 = risky().or_throw()',
			'			result = v',
			'		finally:',
			'			pass',
			'	except ErrorA:',
			'		result = -1',
			'	return result',
		])
		errors = self._lower_and_get_errors( code, 'run' )
		self.assertFalse( errors, errors )


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

# regression for a real cfg.py bug: a NAMED Result[T,E]-shaped local (or a
# compiler-synthesized match-subject) declared as a top-level statement
# directly inside a try body lost its epilogue bookkeeping when the try's
# own per-handler restore() truncated the epilogue stack back to the try's
# entry snapshot - even after a `return`/match arm inside the try body had
# already committed a goto into that entry's own shared label ("use of
# undeclared label"). Fixed by having restore() keep a CAPTURED entry alive
# across the truncation, same as an already-armed defer/errdefer entry.
_NAMED_RESULT_LOCAL_MATCH_RETURN_INSIDE_TRY_BODY = '''
class ErrorA:
	pass

def indicator( which: i32 ) -> Result[i32, ErrorA]:
	if which == 1:
		return Result.Err( ErrorA() )
	return Result.Ok( which )

def other() -> Result[i32, ErrorA]:
	return Result.Ok( 9 )

def run( which: i32 ) -> i32:
	try:
		x: i32 = other().or_throw()
		match indicator( which ):
			case Result.Ok( v ):
				return 0
			case Result.Err( e ):
				return -1
	except ErrorA:
		return -2
	return -3

def main() -> i32:
	if run( 1 ) != -1:
		return 1
	if run( 0 ) != 0:
		return 2
	return 0
'''

# regression for a second, related cfg.py bug: even with no `return`/match at
# all, simply declaring a named Result[T,E] local inside a try body and
# is_err()-checking it produced references to fresh $tN temps whose
# DeclareTemp never made it into the emitted C. Root cause: _stmt_Try's own
# per-handler merge_if() call minted those temps while self._instructions
# still pointed at the just-lowered HANDLER's own instruction list (never
# reset back to the real outer list first, unlike every other merge_if()
# call site) - the DeclareTemp landed in the handler's own (unrelated, often
# dead) block while the matching compute/use instructions landed on the try
# body's own fallthrough path instead ("use of undeclared identifier").
_NAMED_RESULT_LOCAL_IS_ERR_CHECK_INSIDE_TRY_BODY = '''
class ErrorA:
	pass

def indicator( which: i32 ) -> Result[i32, ErrorA]:
	if which == 1:
		return Result.Err( ErrorA() )
	return Result.Ok( which )

def other() -> Result[i32, ErrorA]:
	return Result.Ok( 9 )

def run( which: i32 ) -> i32:
	result: i32 = -3
	try:
		r1: Result[i32, ErrorA] = indicator( which )
		if r1.is_err():
			result = -1
		else:
			result = 0
		x: i32 = other().or_throw()
	except ErrorA:
		result = -2
	return result

def main() -> i32:
	if run( 1 ) != -1:
		return 1
	if run( 0 ) != 0:
		return 2
	return 0
'''


@unittest.skipUnless( test_support.HAS_CC, 'no C compiler (clang/gcc/msvc) found - skipping real-compile try/except RC stress test' )
class TryExceptRCStressTests( RealCompileMixin, unittest.TestCase ):
	def test_fresh_construction_across_handlers_repeated_no_leak_or_double_free( self ) -> None:
		self.assert_programs_run([ ( 'try_except_rc_stress', _RC_STRESS_ACROSS_HANDLERS ) ])

	def test_named_result_local_match_return_inside_try_body( self ) -> None:
		self.assert_programs_run([ ( 'try_except_named_result_match_return', _NAMED_RESULT_LOCAL_MATCH_RETURN_INSIDE_TRY_BODY ) ])

	def test_named_result_local_is_err_check_inside_try_body( self ) -> None:
		self.assert_programs_run([ ( 'try_except_named_result_is_err', _NAMED_RESULT_LOCAL_IS_ERR_CHECK_INSIDE_TRY_BODY ) ])


if __name__ == '__main__':
	unittest.main()
