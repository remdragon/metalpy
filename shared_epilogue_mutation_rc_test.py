# Real-compile-and-run regression tests for a shared-mutation class of RC
# bug found across every construct that lowers more than one mutually-
# exclusive sibling pass over its own body, each independently restore()'d
# back to the SAME entry snapshot (cfg.py's own `entry_snapshot = snapshot();
# ...; restore(entry_snapshot)` shape) - _stmt_If's true/false branches,
# _lower_binary_branch's own true/false thunks (or_return()/or_throw()'s Err/
# Ok dispatch, the for-loop-over-iterator StopIteration dispatch), and
# _stmt_Try's try-body-fall-through/each-handler (see try_except_test.py's
# own _TRY_BODY_LOCAL_RELEASED_BEFORE_COVERED_RAISE_DISPATCH and
# _LOCAL_BEFORE_TRY_SURVIVES_ACROSS_REPEATED_CALLS_REGARDLESS_OF_ORDER for
# that construct's own two variants of this bug).
#
# Root cause: every entry declared BEFORE the branching construct has its own
# Epilogue object shared by reference across every sibling pass - restore()
# only ever reverts STACK MEMBERSHIP, never an already-mutated entry's own
# .cancelled flag. A manually_decreffed()/move()/deleted() call on one,
# already-lowered sibling (e.g. `compiler.decref(g)` on an if's own true
# branch) permanently mutates the SAME object an earlier-taken, mutually-
# exclusive sibling (the false branch, restored back to the SAME entry
# snapshot) still depends on - its own release of that entry, if any, walks
# right past it, thinking it's already handled. Fixed via cfg.py's
# enter_diverging_paths()/exit_diverging_paths() (CFGState._protected_
# entries), routing any manual cancellation of an entry declared before the
# construct through the SAME runtime-flag mechanism an already-captured
# entry's own cancellation already used, rather than a static, irreversible
# cancel that corrupts every other sibling sharing the same object.
#
# That fix itself surfaced a SECOND, deeper bug in cfg.py's merge_if(): once
# an entry is flag-guarded (armed True by default, disarmed only on the path
# that consumed it), EVERY later release of that SAME entry must go through
# the flag check - but merge_if()'s own "exactly one branch terminates, the
# survivor's bindings carry forward" reconciliation (reestablish(), the
# `already_live` check) only recognized an entry as already tracked by
# comparing against the construct's own ENTRY snapshot - never checking
# whether it was ALREADY sitting in self._epilogue_stack right now, kept
# alive there specifically BECAUSE it's flag-guarded (see restore()'s own
# "survivors" comment). For anything declared MID-BODY (never present in the
# entry snapshot at all - the true_/or_throw(mapper)_-receiver shape, or a
# nested try's own outer-scope local decref'd inside an inner handler before
# re-raising outward), this pushed a SECOND, BRAND NEW, non-flag-guarded
# entry for the SAME already-tracked object - its own unconditional release
# then stacked on top of the original's still-live flag-guarded one, a real
# double release/heap corruption (confirmed with ASan-adjacent glibc
# "malloc(): unaligned tcache chunk detected" under gcc, and a silent no-
# output SIGABRT under clang/MSVC) on whichever path never actually touched
# the entry at all. Fixed by ALSO checking entry-object identity against the
# live self._epilogue_stack, not just the entry snapshot.

import unittest

import test_support
from test_support import RealCompileMixin

# the plainest possible repro: no try/except, no loop, not even or_return -
# just an ordinary if/else, the single most common branching construct in
# the language.
_PLAIN_IF_ELSE_LOCAL_DECREFFED_ON_ONE_BRANCH_ONLY = '''
class Guard:
	tag: i32

def run( bad: bool ) -> i32:
	g: Guard = Guard( tag = 111 )
	if bad:
		compiler.decref( g )
		return -1
	return 0

def main() -> i32:
	if run( True ) != -1:
		return 1
	if run( False ) != 0:
		return 2
	return 0
'''

# both branches decref the SAME pre-if local (each on its own path) - a
# sanity check that the fix doesn't over-protect and cause a leak instead.
_PLAIN_IF_ELSE_LOCAL_DECREFFED_ON_BOTH_BRANCHES = '''
class Guard:
	tag: i32

def run( bad: bool ) -> i32:
	g: Guard = Guard( tag = 111 )
	if bad:
		compiler.decref( g )
		return -1
	compiler.decref( g )
	return 0

def main() -> i32:
	if run( True ) != -1:
		return 1
	if run( False ) != 0:
		return 2
	return 0
'''

# or_throw(mapper)'s own Err arm (_lower_binary_branch's true_thunk)
# explicitly compiler.decref()s the receiver (a goto-based ir.Raise never
# unwinds anything on its own - see or_throw_mapper_test.py) - the Ok arm
# (false_thunk), restored back to the SAME entry snapshot, must still see
# the receiver as needing its own normal release wherever it's eventually
# consumed.
_OR_THROW_MAPPER_RECEIVER_NOT_PREMATURELY_CANCELLED_ON_OK_PATH = '''
class Elem:
	tag: i32

def probe( i: usize, e: Elem ) -> Result[Elem, IndexError]:
	if i == 0:
		return Result.Ok( e )
	return Result.Err( IndexError() )

def to_stop( err: IndexError ) -> StopIteration:
	return StopIteration()

def inner( i: usize, e: Elem ) -> Result[Elem, StopIteration]:
	try:
		w: Elem = probe( i, e ).or_throw( to_stop )
	except StopIteration as ex:
		compiler.decref( ex )
		return Result.Ok( Elem( tag = -1 ) )
	return Result.Ok( w )

def main() -> i32:
	e: Elem = Elem( tag = 0 )
	before: usize = compiler.refcount( e )
	match inner( 0, e ):
		case Result.Ok( _ ):
			pass
		case Result.Err( _ ):
			return 1
	if compiler.refcount( e ) != before:
		return 2
	return 0
'''

# the merge_if() double-release bug's own minimal repro: an ordinary if
# (declared MID-BODY, never present in the try's own entry snapshot at all)
# nested inside a try, whose true branch decrefs the pre-if local before
# raising - g's own manual decref goes through the flag path (protected by
# the if's own enter_diverging_paths), and merge_if()'s "one branch
# terminates" survivor reconciliation used to push a SECOND, non-flag-
# guarded entry for the SAME object on the non-raising path.
_IF_NESTED_IN_TRY_LOCAL_DECREFFED_BEFORE_RAISE = '''
class Guard:
	tag: i32

class Boom:
	tag: i32

def run( bad: bool ) -> i32:
	try:
		g: Guard = Guard( tag = 111 )
		if bad:
			compiler.decref( g )
			raise Boom( tag = 222 )
	except Boom as e:
		compiler.decref( e )
		return -1
	return 0

def main() -> i32:
	if run( False ) != 0:
		return 1
	if run( True ) != -1:
		return 2
	return 0
'''

# the SAME bug, one level deeper: a try nested inside another try, whose
# inner handler decrefs an OUTER-scope local before re-raising to the outer
# handler - same "flag-guarded entry, merge_if pushes a second one" shape,
# just via _stmt_Try's own enter_diverging_paths instead of _stmt_If's.
_TRY_NESTED_IN_TRY_OUTER_LOCAL_DECREFFED_BEFORE_REROUTED_RAISE = '''
class Guard:
	tag: i32

class Boom:
	tag: i32

class Bang:
	tag: i32

def run( bad: bool ) -> i32:
	try:
		g: Guard = Guard( tag = 111 )
		try:
			if bad:
				raise Boom( tag = 222 )
		except Boom as e:
			compiler.decref( e )
			compiler.decref( g )
			raise Bang( tag = 333 )
	except Bang as e2:
		compiler.decref( e2 )
		return -1
	return 0

def main() -> i32:
	if run( False ) != 0:
		return 1
	if run( True ) != -1:
		return 2
	return 0
'''


@unittest.skipUnless( test_support.HAS_CC, 'no C compiler (clang/gcc/msvc) found - skipping' )
class SharedEpilogueMutationRCTests( RealCompileMixin, unittest.TestCase ):
	def test_plain_if_else_local_decreffed_on_one_branch_only( self ) -> None:
		self.assert_programs_run([
			( 'if_else_decref_one_branch', _PLAIN_IF_ELSE_LOCAL_DECREFFED_ON_ONE_BRANCH_ONLY ),
		])

	def test_plain_if_else_local_decreffed_on_both_branches( self ) -> None:
		self.assert_programs_run([
			( 'if_else_decref_both_branches', _PLAIN_IF_ELSE_LOCAL_DECREFFED_ON_BOTH_BRANCHES ),
		])

	def test_or_throw_mapper_receiver_not_prematurely_cancelled_on_ok_path( self ) -> None:
		self.assert_programs_run([
			( 'or_throw_mapper_receiver_ok_path', _OR_THROW_MAPPER_RECEIVER_NOT_PREMATURELY_CANCELLED_ON_OK_PATH ),
		])

	def test_if_nested_in_try_local_decreffed_before_raise( self ) -> None:
		self.assert_programs_run([
			( 'if_nested_in_try_decref_before_raise', _IF_NESTED_IN_TRY_LOCAL_DECREFFED_BEFORE_RAISE ),
		])

	def test_try_nested_in_try_outer_local_decreffed_before_rerouted_raise( self ) -> None:
		self.assert_programs_run([
			( 'try_nested_in_try_outer_decref', _TRY_NESTED_IN_TRY_OUTER_LOCAL_DECREFFED_BEFORE_REROUTED_RAISE ),
		])


if __name__ == '__main__':
	unittest.main()
