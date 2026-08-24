# Real-compile-and-run coverage for the general auto-or_throw() rule
# (PLAN_CHECKED_ARITHMETIC_GAP.md, now closed): every Result[T,E]-producing
# expression - checked arithmetic, obj[i]/obj[i]=v/obj[i] op= v, and an
# ordinary fallible call alike - now auto-inserts .or_throw() (not
# .or_return()) whenever (1) it's a discarded statement, or (2) it flows into
# a context wanting its own Ok-payload type T directly rather than the whole
# Result[T,E]. or_throw() degrades to exactly or_return()'s own unconditional-
# propagate semantics whenever there's no enclosing try (or no leaf is
# covered) - see lowering.py's _auto_or_throw/_emit_or_throw.
#
# lowering_test.py's own IR-shape assertions (OrThrow instead of OrReturn/
# OrJump at every formerly-special-cased site) and the compile-error rejection
# cases already cover the mechanism in isolation; this file proves the actual
# NEW capability end to end with a real compile+link+run: wrapping checked
# arithmetic/obj[i]/a fallible call in try/except now genuinely dispatches to
# a handler, and declaring an explicit Result[T,E]-typed target now captures
# the raw Result instead of auto-propagating - both were previously
# impossible (checked arithmetic auto-consumed unconditionally; obj[i] had
# ZERO sugar and was a hard type-mismatch).
#
# See checked_dunder_test.py's own test_i32_add_panics_under_panic_arithmetic_
# on_real_overflow for panic_arithmetic's own real compile+run regression
# coverage (unaffected by this whole file's change) - not duplicated here.

import unittest

import emitter_c
import test_support
from test_support import RealCompileMixin

_I32_MAX = '2147483647'

# --- item 2: checked arithmetic inside try/except actually dispatches ------

_CHECKED_ARITHMETIC_INSIDE_TRY_EXCEPT_DISPATCHES = f'''
def risky_add( a: i32, b: i32 ) -> i32:
	result: i32 = 0
	try:
		result = a + b
	except OverflowError:
		result = -1
	return result

def main() -> i32:
	if risky_add( 5, 7 ) != 12:
		return 1
	if risky_add( {_I32_MAX}, 1 ) != -1:
		return 2
	return 0
'''

# --- item 3: an explicit Result[T,E]-typed target captures the raw Result --

_EXPLICIT_RESULT_TARGET_CAPTURES_RAW_RESULT = f'''
def main() -> i32:
	a: i32 = {_I32_MAX}
	b: i32 = 1
	overflowed: Result[i32,OverflowError] = a + b
	if overflowed.is_ok():
		return 1
	if not overflowed.is_err():
		return 2

	c: i32 = 5
	d: i32 = 7
	ok: Result[i32,OverflowError] = c + d
	if ok.is_err():
		return 3
	if ok.unwrap( 'unreachable' ) != 12:
		return 4
	return 0
'''

# --- item 4: a bare discarded fallible call auto-propagates / dispatches ---

_BARE_DISCARDED_FALLIBLE_CALL_PROPAGATES_AND_DISPATCHES = '''
class MyError:
	pass

def risky( bad: bool ) -> Result[None,MyError]:
	if bad:
		return Result.Err( MyError() )
	return Result.Ok( None )

def propagate( bad: bool ) -> Result[None,MyError]:
	risky( bad ) # bare discarded statement - case 1 of the general rule
	return Result.Ok( None )

def dispatch( bad: bool ) -> i32:
	result: i32 = 0
	try:
		risky( bad ) # bare discarded statement, inside try - dispatches
	except MyError:
		result = 1
	return result

def main() -> i32:
	match propagate( True ):
		case Result.Ok( v ):
			return 1
		case Result.Err( e ):
			pass
	match propagate( False ):
		case Result.Ok( v ):
			pass
		case Result.Err( e ):
			return 2
	if dispatch( True ) != 1:
		return 3
	if dispatch( False ) != 0:
		return 4
	return 0
'''

# --- item 5: arr[0] as `x: i32 = arr[0]` now compiles/propagates/dispatches

_SUBSCRIPT_AS_DIRECT_T_TYPED_TARGET_COMPILES = '''
def get_first( lst: list[i32] ) -> Result[i32,IndexError]:
	x: i32 = lst[0] # previously a hard type-mismatch (zero sugar) - now case 2
	return Result.Ok( x )

def get_first_or_default( lst: list[i32] ) -> i32:
	result: i32 = -1
	try:
		x: i32 = lst[0]
		result = x
	except IndexError:
		result = -2
	return result

def main() -> i32:
	lst: list[i32] = list[i32]()
	lst.append( 42 )
	match get_first( lst ):
		case Result.Ok( v ):
			if v != 42:
				return 1
		case Result.Err( e ):
			return 2
	if get_first_or_default( lst ) != 42:
		return 3
	empty: list[i32] = list[i32]()
	if get_first_or_default( empty ) != -2:
		return 4
	return 0
'''

# --- item 6 (regression): obj[i]=v / obj[i] op= v, no try, propagate -------

_SUBSCRIPT_ASSIGN_AND_AUGASSIGN_STILL_PROPAGATE_WITH_NO_TRY = '''
def set_first( lst: list[i32], v: i32 ) -> Result[None,IndexError]:
	lst[0] = v
	return Result.Ok( None )

def bump_first( lst: list[i32] ) -> Result[None,IndexError | OverflowError]:
	lst[0] += 10 # AugAssign's own arithmetic is checked too - needs OverflowError coverage on top of IndexError
	return Result.Ok( None )

def main() -> i32:
	lst: list[i32] = list[i32]()
	lst.append( 1 )
	match set_first( lst, 99 ):
		case Result.Ok( v ):
			pass
		case Result.Err( e ):
			return 1
	if lst[0].unwrap( 'x' ) != 99:
		return 2
	match bump_first( lst ):
		case Result.Ok( v ):
			pass
		case Result.Err( e2 ):
			return 3
	if lst[0].unwrap( 'x' ) != 109:
		return 4
	return 0
'''

# --- item 8 (regression): checked arithmetic inside a generator body -------

_CHECKED_ARITHMETIC_INSIDE_GENERATOR_BODY_STILL_WORKS = f'''
def gen_sums( a: i32, b: i32 ) -> Generator[i32,OverflowError | StopIteration]:
	yield a + b

def main() -> i32:
	g = gen_sums( 3, 4 )
	match g.__next__():
		case Result.Ok( v ):
			if v != 7:
				return 1
		case Result.Err( e ):
			return 2

	g2 = gen_sums( {_I32_MAX}, 1 )
	match g2.__next__():
		case Result.Ok( v ):
			return 3
		case Result.Err( e ):
			pass
	return 0
'''

# --- item 9 (regression): checked arithmetic inside a multi-statement ------
# --- @inline splice's pre-return statements (the inline_exit carve-out) ----

_CHECKED_ARITHMETIC_INSIDE_MULTISTATEMENT_INLINE_SPLICE_PRELUDE = f'''
@cstruct
class Counter:
	value: i32

	@inline
	def bumped_checked( self, by: i32 ) -> Result[i32,OverflowError]:
		result: i32 = self.value + by
		if result < 0:
			result = 0
		return Result.Ok( result )

def main() -> i32:
	c: Counter = Counter( value = 10 )
	match c.bumped_checked( 5 ):
		case Result.Ok( v ):
			if v != 15:
				return 1
		case Result.Err( e ):
			return 2

	maxed: Counter = Counter( value = {_I32_MAX} )
	match maxed.bumped_checked( 1 ):
		case Result.Ok( v ):
			return 3
		case Result.Err( e ):
			pass
	return 0
'''

# --- item 10 (regression): (a + b) + c, no try (propagate) and with try ----
# --- (catchable) - exercises _reject_unconsumed_result_operand's own -------
# --- turned-auto-consume hook -----------------------------------------------

_CHAINED_CHECKED_ARITHMETIC_PROPAGATES_AND_IS_CATCHABLE = f'''
def chain_no_try( a: i32, b: i32, c: i32 ) -> Result[i32,OverflowError]:
	x: i32 = ( a + b ) + c
	return Result.Ok( x )

def chain_with_try( a: i32, b: i32, c: i32 ) -> i32:
	result: i32 = -1
	try:
		result = ( a + b ) + c
	except OverflowError:
		result = -2
	return result

def main() -> i32:
	match chain_no_try( 1, 2, 3 ):
		case Result.Ok( v ):
			if v != 6:
				return 1
		case Result.Err( e ):
			return 2

	match chain_no_try( {_I32_MAX}, 1, 0 ):
		case Result.Ok( v ):
			return 3
		case Result.Err( e ):
			pass

	if chain_with_try( 1, 2, 3 ) != 6:
		return 4
	if chain_with_try( {_I32_MAX}, 1, 0 ) != -2:
		return 5
	return 0
'''


@unittest.skipUnless( test_support.HAS_CC, 'no C compiler (clang/gcc/msvc) found - skipping real-compile auto-or_throw tests' )
class AutoOrThrowBehaviorTests( RealCompileMixin, unittest.TestCase ):
	def test_checked_arithmetic_inside_try_except_dispatches( self ) -> None:
		self.assert_programs_run([ ( 'checked_arith_try', _CHECKED_ARITHMETIC_INSIDE_TRY_EXCEPT_DISPATCHES ) ])

	def test_explicit_result_typed_target_captures_raw_result( self ) -> None:
		self.assert_programs_run([ ( 'explicit_result_target', _EXPLICIT_RESULT_TARGET_CAPTURES_RAW_RESULT ) ])

	def test_bare_discarded_fallible_call_propagates_and_dispatches( self ) -> None:
		self.assert_programs_run([ ( 'bare_discard_call', _BARE_DISCARDED_FALLIBLE_CALL_PROPAGATES_AND_DISPATCHES ) ])

	def test_subscript_as_direct_t_typed_target_compiles( self ) -> None:
		self.assert_programs_run([ ( 'subscript_direct_target', _SUBSCRIPT_AS_DIRECT_T_TYPED_TARGET_COMPILES ) ])

	def test_subscript_assign_and_augassign_still_propagate_with_no_try( self ) -> None:
		self.assert_programs_run([ ( 'subscript_assign_augassign', _SUBSCRIPT_ASSIGN_AND_AUGASSIGN_STILL_PROPAGATE_WITH_NO_TRY ) ])

	def test_checked_arithmetic_inside_generator_body_still_works( self ) -> None:
		self.assert_programs_run([ ( 'checked_arith_generator', _CHECKED_ARITHMETIC_INSIDE_GENERATOR_BODY_STILL_WORKS ) ])

	def test_checked_arithmetic_inside_multistatement_inline_splice_prelude( self ) -> None:
		self.assert_programs_run([ ( 'checked_arith_inline_splice', _CHECKED_ARITHMETIC_INSIDE_MULTISTATEMENT_INLINE_SPLICE_PRELUDE ) ])

	def test_chained_checked_arithmetic_propagates_and_is_catchable( self ) -> None:
		self.assert_programs_run([ ( 'chained_checked_arith', _CHAINED_CHECKED_ARITHMETIC_PROPAGATES_AND_IS_CATCHABLE ) ])
