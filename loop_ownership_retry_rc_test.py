# Real-compile-and-run regression test for cfg.py's hard_restore() - used by
# lowering.py's _lower_loop_body_with_ownership_retry (see
# lowering_test.py's own test_loop_promotes_borrowed_param_mixed_with_owned_
# reassignment for the retry mechanism itself) to discard a failed loop-body
# lowering attempt and re-lower it from scratch once a borrowed-entering,
# owned-by-the-back-edge name is pre-promoted.
#
# hard_restore() only ever reset STACK MEMBERSHIP (entries pushed since the
# retried loop's own entry snapshot) - it never reverted an already-mutated
# Epilogue object's own .cancelled/.captured/.flag for an entry declared
# BEFORE the loop (same shared-mutation class of bug cfg.py's enter_try()
# fixes for try/except - see its own docstring). A local declared before the
# loop, captured by an early `return` inside the loop body and THEN manually
# compiler.decref()'d (which mints a runtime cancel flag, since the entry is
# already captured), left that flag stored on the entry's own .flag - if the
# SAME loop body's retry mechanism then fires (for a completely unrelated
# name), the abandoned attempt's own flag survived on the entry even though
# cfg.py's own _cancel_flags list (and lowering.py's matching bookkeeping)
# had already rolled it back - the retried attempt REUSED that now-orphaned
# flag reference instead of minting its own, and the resulting C referenced
# a variable never declared at the function's own top: "use of undeclared
# identifier '__cancel_flag_0'" from clang, a hard compile failure, not just
# a silent RC bug.

import unittest

import test_support
from test_support import RealCompileMixin

_CAPTURED_DECREF_BEFORE_LOOP_SURVIVES_UNRELATED_OWNERSHIP_RETRY = '''
class Elem:
	tag: i32

def run( p: Elem, bad: bool ) -> i32:
	g: Elem = Elem( tag = 111 )
	i: usize = 0
	while i < 2:
		if bad:
			return -1
		compiler.decref( g )
		g = Elem( tag = 333 )
		p = Elem( tag = 222 ) # borrowed-entering p reassigned owned - triggers the retry
		with compiler.wrap_arithmetic:
			i = i + 1
	compiler.decref( g )
	compiler.decref( p )
	return 0

def main() -> i32:
	e1: Elem = Elem( tag = 0 )
	if run( e1, True ) != -1:
		return 1
	e2: Elem = Elem( tag = 0 )
	if run( e2, False ) != 0:
		return 2
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
