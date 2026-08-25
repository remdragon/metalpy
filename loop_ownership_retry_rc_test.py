# Real-compile-and-run regression test for cfg.py's hard_restore() - used by
# lowering.py's _lower_loop_body_with_ownership_retry (see
# lowering_test.py's own test_loop_promotes_borrowed_param_mixed_with_owned_
# reassignment for the retry mechanism itself) to discard a failed loop-body
# lowering attempt and re-lower it from scratch once a borrowed-entering,
# owned-by-the-back-edge name is pre-promoted.
#
# Two bugs in an abandoned attempt's own rollback, both in state OUTSIDE
# hard_restore()'s original "just truncate the stack" scope:
#
# 1. hard_restore() reset an already-mutated Epilogue's own .cancelled/.flag
#    FIELDS on whatever object currently sat in each surviving slot, instead
#    of swapping identity back to the snapshot's own original object (like
#    restore() does - see its own docstring). A pre-loop local `del`'d
#    during the abandoned attempt gets a REPLACEMENT object swapped into
#    self._epilogue_stack (_neutralize()'s static-cancel path never mutates
#    in place); reverting fields on that replacement, rather than restoring
#    the original object's own identity, left self.bindings (correctly
#    reverted) and self._epilogue_stack (still pointing at the abandoned
#    replacement) disagreeing about which object IS the entry - the retried
#    attempt's own _neutralize() then can't find its own binding's entry in
#    the stack, silently failing to cancel it there.
#
# 2. A pre-loop local captured by an early `return` inside the loop body and
#    THEN `del`'d mints a runtime cancel flag (already-captured entries are
#    flag-guarded, not statically cancelled - _neutralize()'s docstring) by
#    mutating entry.flag IN PLACE on the shared object itself - unlike the
#    replace-don't-mutate static path, there's no fresh object here to
#    protect the snapshot's own recorded identity. Reverting identity alone
#    (fix 1 above) isn't enough: the snapshot's entry_objects list holds a
#    live reference to that SAME object, so the abandoned attempt's flag
#    mutation "poisons" the snapshot retroactively. The retried attempt's
#    own del then reused that now-truncated-out-of-_cancel_flags flag
#    reference instead of minting its own, and the resulting C referenced a
#    variable never declared at the function's own top: "use of undeclared
#    identifier '__cancel_flag_0'" from clang, a hard compile failure.
#
# A third, related bug lived in lowering.py itself, not cfg.py: fn.names (the
# name->Variable table `del`/reassignment go through - see _stmt_Delete) was
# never part of hard_restore()'s own rollback at all. A `del`-then-reassign
# inside the loop body mints a fresh, uid-suffixed Variable and registers it
# in fn.names; on rollback that registration lingered, so the retried
# attempt's own `del` resolved through the ABANDONED attempt's own
# now-discarded Variable (a real "use of undeclared identifier" too, for the
# variable itself this time, not a flag).
#
# NOTE: `del g` is deliberately never followed by a reassignment of `g`
# inside this loop body (unlike an earlier version of this test) - that
# combination hits a SEPARATE, more fundamental bug, unrelated to retry: a
# `del`-then-reassign of a pre-loop-declared name mints a fresh, uid-suffixed
# C variable for the new value, but the loop's own back edge still jumps to
# code that reads the ORIGINAL (stale, already-released) C variable on every
# later iteration, since nothing re-points "the current g" at the new one for
# the next lap - a real double-free even with NO retry involved at all
# (confirmed with a plain `while` loop, no promotion, no second parameter).
# That gap needs its own dedicated fix in lowering.py's loop-carried name
# handling and is out of scope here - this test instead puts the manual
# `del` on a branch that exits the loop outright (never reassigns, never
# reaches the back edge), so it exercises hard_restore()'s own rollback
# without also tripping over the unrelated bug.

import unittest

import test_support
from test_support import RealCompileMixin

_CAPTURED_DECREF_BEFORE_LOOP_SURVIVES_UNRELATED_OWNERSHIP_RETRY = '''
class Elem:
	tag: i32

def run( p: Elem, bad1: bool, bad2: bool ) -> i32:
	g: Elem = Elem( tag = 111 )
	i: usize = 0
	while i < 2:
		if bad1:
			return -1
		if bad2:
			del g
			return -2
		p = Elem( tag = 222 ) # borrowed-entering p reassigned owned - triggers the retry
		with compiler.wrap_arithmetic:
			i = i + 1
	del g
	del p
	return 0

def main() -> i32:
	e1: Elem = Elem( tag = 0 )
	if run( e1, True, False ) != -1:
		return 1
	e2: Elem = Elem( tag = 0 )
	if run( e2, False, True ) != -2:
		return 2
	e3: Elem = Elem( tag = 0 )
	if run( e3, False, False ) != 0:
		return 3
	return 0
'''


@unittest.skipUnless( test_support.HAS_CC, 'no C compiler (clang/gcc/msvc) found - skipping' )
class LoopOwnershipRetryRCTests( RealCompileMixin, unittest.TestCase ):
	def test_captured_decref_before_loop_survives_unrelated_ownership_retry( self ) -> None:
		self.assert_programs_run([
			( 'loop_retry_captured_decref', _CAPTURED_DECREF_BEFORE_LOOP_SURVIVES_UNRELATED_OWNERSHIP_RETRY ),
		])


if __name__ == '__main__':
	unittest.main()
