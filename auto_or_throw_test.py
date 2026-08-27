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

from pathlib import Path
import unittest

from compiler import Compiler
from discovery import Discovery
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

# regression: risky( bad ) above passes only scalar args, so it never builds a
# fresh pending temp as part of the discarded call's own argument list. This
# variant's `try_insert( MyError(), bad )` does - a fresh RC arg constructed
# INLINE for a discarded call whose Err leaf is uncovered (propagates via
# auto-or_throw's early-return path, not a normal fall-through statement end).
# That argument temp leaked (never released on the early-return branch) until
# fixed - see _finish_call_result/_auto_or_throw/_emit_or_throw's own
# receiver_pending_start threading.
_BARE_DISCARDED_FALLIBLE_CALL_WITH_FRESH_RC_ARG_DOES_NOT_LEAK = '''
class MyError:
	pass

def try_insert( key: MyError, bad: bool ) -> Result[None,MyError]:
	if bad:
		return Result.Err( MyError() )
	return Result.Ok( None )

def propagate_generic( bad: bool ) -> Result[None,MyError]:
	try_insert( MyError(), bad ) # discarded call, fresh RC arg - must not leak
	return Result.Ok( None )

def main() -> i32:
	match propagate_generic( True ):
		case Result.Ok( v ):
			return 1
		case Result.Err( e ):
			pass
	match propagate_generic( False ):
		case Result.Ok( v ):
			pass
		case Result.Err( e ):
			return 2
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

# regression: a __setitem__/__getitem__ pair returning Result[_,E] whose Err
# leaf is uncovered (no try) - the AugAssign's own combined `result` (a FRESH
# RC value, here a new Box built via __add__) is passed as the discarded
# __setitem__ call's own argument. That argument temp leaked (never released
# on the early-return propagate path) until fixed - same bug family/fix as
# _BARE_DISCARDED_FALLIBLE_CALL_WITH_FRESH_RC_ARG_DOES_NOT_LEAK above, found
# by auditing every other _auto_or_throw( ..., want_result = False ) site for
# the same shape (see lowering.py's `subscript_pending_start`).
_SUBSCRIPT_AUGASSIGN_WITH_FRESH_RC_RESULT_DOES_NOT_LEAK = '''
import compiler

class MyError:
	pass

class Box:
	v: i32 = 0

	def __init__( self, v: i32 ) -> None:
		self.v = v

	def __add__( self, other: Box ) -> Box:
		with compiler.wrap_arithmetic:
			return Box( self.v + other.v )

class Container:
	slot: Box = Box( 0 )
	fail: bool = False

	def __getitem__( self, i: i32 ) -> Box:
		return self.slot

	def __setitem__( self, i: i32, v: Box ) -> Result[None,MyError]:
		if self.fail:
			return Result.Err( MyError() )
		self.slot = v
		return Result.Ok( None )

def bump( c: Container ) -> Result[None,MyError]:
	c[0] += Box( 5 ) # discarded __setitem__ call, fresh RC `result` arg - must not leak
	return Result.Ok( None )

def main() -> i32:
	c: Container = Container()
	c.fail = True
	match bump( c ):
		case Result.Ok( v ):
			return 1
		case Result.Err( e ):
			pass
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


# --- item 11: bare discarded call to a GENERIC Result-returning function ---
# --- (v1 gap #1 - didn't reach _lower_call's shared tail at all) -----------

_BARE_DISCARDED_GENERIC_CALL_PROPAGATES_AND_DISPATCHES = '''
class MyError:
	pass

def try_insert[T]( key: T, bad: bool ) -> Result[None,MyError]:
	if bad:
		return Result.Err( MyError() )
	return Result.Ok( None )

def propagate_generic( bad: bool ) -> Result[None,MyError]:
	try_insert( 5, bad ) # bare discarded GENERIC call - case 1, never reached _lower_call's shared tail
	return Result.Ok( None )

def dispatch_generic( bad: bool ) -> i32:
	result: i32 = 0
	try:
		try_insert( 5, bad ) # same, inside a covering try - dispatches
	except MyError:
		result = 1
	return result

def main() -> i32:
	match propagate_generic( True ):
		case Result.Ok( v ):
			return 1
		case Result.Err( e ):
			pass
	match propagate_generic( False ):
		case Result.Ok( v ):
			pass
		case Result.Err( e ):
			return 2
	if dispatch_generic( True ) != 1:
		return 3
	if dispatch_generic( False ) != 0:
		return 4
	return 0
'''

# --- item 12 (regression): an already-ASSIGNED generic call is unaffected --

_ASSIGNED_GENERIC_CALL_STILL_CAPTURES_RESULT = '''
class MyError:
	pass

def try_insert[T]( key: T, bad: bool ) -> Result[None,MyError]:
	if bad:
		return Result.Err( MyError() )
	return Result.Ok( None )

def main() -> i32:
	r: Result[None,MyError] = try_insert( 5, True )
	if r.is_ok():
		return 1
	if not r.is_err():
		return 2
	r2: Result[None,MyError] = try_insert( 5, False )
	if r2.is_err():
		return 3
	return 0
'''

# --- item 13: bare discarded call requiring runtime UNION-ARGUMENT dispatch
# --- (v1 gap #2 - _lower_conditional_dispatch never reached _finish_call_result)

_BARE_DISCARDED_UNION_ARG_DISPATCH_PROPAGATES_AND_DISPATCHES = '''
class FeedError:
	pass

class Cat:
	pass

class Dog:
	pass

@overload
def feed( pet: Cat, bad: bool ) -> Result[None,FeedError]: ...
@overload
def feed( pet: Dog, bad: bool ) -> Result[None,FeedError]: ...

def feed( pet: Cat, bad: bool ) -> Result[None,FeedError]:
	if bad:
		return Result.Err( FeedError() )
	return Result.Ok( None )

def feed( pet: Dog, bad: bool ) -> Result[None,FeedError]:
	if bad:
		return Result.Err( FeedError() )
	return Result.Ok( None )

def propagate_union_arg( pet: Cat|Dog, bad: bool ) -> Result[None,FeedError]:
	feed( pet, bad ) # bare discarded call, runtime union-ARGUMENT dispatch
	return Result.Ok( None )

def dispatch_union_arg( pet: Cat|Dog, bad: bool ) -> i32:
	result: i32 = 0
	try:
		feed( pet, bad ) # same, inside a covering try - dispatches
	except FeedError:
		result = 1
	return result

def main() -> i32:
	match propagate_union_arg( Cat(), True ):
		case Result.Ok( v ):
			return 1
		case Result.Err( e ):
			pass
	match propagate_union_arg( Dog(), False ):
		case Result.Ok( v ):
			pass
		case Result.Err( e ):
			return 2
	if dispatch_union_arg( Cat(), True ) != 1:
		return 3
	if dispatch_union_arg( Dog(), False ) != 0:
		return 4
	return 0
'''

# --- item 14 (regression): an already-ASSIGNED union-argument dispatch call
# --- is unaffected -----------------------------------------------------------

_ASSIGNED_UNION_ARG_DISPATCH_CALL_STILL_CAPTURES_RESULT = '''
class FeedError:
	pass

class Cat:
	pass

class Dog:
	pass

@overload
def feed( pet: Cat, bad: bool ) -> Result[None,FeedError]: ...
@overload
def feed( pet: Dog, bad: bool ) -> Result[None,FeedError]: ...

def feed( pet: Cat, bad: bool ) -> Result[None,FeedError]:
	if bad:
		return Result.Err( FeedError() )
	return Result.Ok( None )

def feed( pet: Dog, bad: bool ) -> Result[None,FeedError]:
	if bad:
		return Result.Err( FeedError() )
	return Result.Ok( None )

def main() -> i32:
	pet: Cat|Dog = Cat()
	r: Result[None,FeedError] = feed( pet, True )
	if r.is_ok():
		return 1
	if not r.is_err():
		return 2
	pet2: Cat|Dog = Dog()
	r2: Result[None,FeedError] = feed( pet2, False )
	if r2.is_err():
		return 3
	return 0
'''

# --- item 15: bare discarded call requiring runtime UNION-RECEIVER dispatch
# --- (v1 gap #2, other half - _lower_union_receiver_call never reached
# --- _finish_call_result either) --------------------------------------------

_BARE_DISCARDED_UNION_RECEIVER_DISPATCH_PROPAGATES_AND_DISPATCHES = '''
class FeedError:
	pass

class Cat:
	def feed( self, bad: bool ) -> Result[None,FeedError]:
		if bad:
			return Result.Err( FeedError() )
		return Result.Ok( None )

class Dog:
	def feed( self, bad: bool ) -> Result[None,FeedError]:
		if bad:
			return Result.Err( FeedError() )
		return Result.Ok( None )

def propagate_union_receiver( pet: Cat|Dog, bad: bool ) -> Result[None,FeedError]:
	pet.feed( bad ) # bare discarded call, runtime union-RECEIVER dispatch
	return Result.Ok( None )

def dispatch_union_receiver( pet: Cat|Dog, bad: bool ) -> i32:
	result: i32 = 0
	try:
		pet.feed( bad ) # same, inside a covering try - dispatches
	except FeedError:
		result = 1
	return result

def main() -> i32:
	match propagate_union_receiver( Cat(), True ):
		case Result.Ok( v ):
			return 1
		case Result.Err( e ):
			pass
	match propagate_union_receiver( Dog(), False ):
		case Result.Ok( v ):
			pass
		case Result.Err( e ):
			return 2
	if dispatch_union_receiver( Cat(), True ) != 1:
		return 3
	if dispatch_union_receiver( Dog(), False ) != 0:
		return 4
	return 0
'''

# --- item 16 (regression): an already-ASSIGNED union-receiver dispatch call
# --- is unaffected -----------------------------------------------------------

_ASSIGNED_UNION_RECEIVER_DISPATCH_CALL_STILL_CAPTURES_RESULT = '''
class FeedError:
	pass

class Cat:
	def feed( self, bad: bool ) -> Result[None,FeedError]:
		if bad:
			return Result.Err( FeedError() )
		return Result.Ok( None )

class Dog:
	def feed( self, bad: bool ) -> Result[None,FeedError]:
		if bad:
			return Result.Err( FeedError() )
		return Result.Ok( None )

def main() -> i32:
	pet: Cat|Dog = Cat()
	r: Result[None,FeedError] = pet.feed( True )
	if r.is_ok():
		return 1
	if not r.is_err():
		return 2
	pet2: Cat|Dog = Dog()
	r2: Result[None,FeedError] = pet2.feed( False )
	if r2.is_err():
		return 3
	return 0
'''

# regression: `s[0] != 'h'` auto-or_throw's the fallible __getitem__ result to
# get the bare str, then uses it only as a `!=` operand (never bound to a
# name) - _emit_or_throw's own unwrapped payload was never cfg.fresh_temp()-
# registered (unlike _consume_checked_result's identical or_return() path),
# so nothing ever decref'd it: a real leak of the extracted RC element.
_SUBSCRIPT_RESULT_AS_BARE_COMPARE_OPERAND_DOES_NOT_LEAK = '''
def check_str( s: str ) -> Result[bool, IndexError]:
	if s[0] != 'h':
		return Result.Ok( False )
	return Result.Ok( True )

def check_list( xs: list[str] ) -> Result[bool, IndexError]:
	if xs[0] != 'x':
		return Result.Ok( False )
	return Result.Ok( True )

def main() -> i32:
	if not check_str( 'hello' ).unwrap( 'str' ):
		return 1
	if check_str( 'world' ).unwrap( 'str' ):
		return 2
	xs: list[str] = list[str]()
	xs.append( 'x' )
	if not check_list( xs ).unwrap( 'list' ):
		return 3
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

	def test_bare_discarded_fallible_call_with_fresh_rc_arg_does_not_leak( self ) -> None:
		self.assert_programs_run([ ( 'bare_discard_fresh_rc_arg', _BARE_DISCARDED_FALLIBLE_CALL_WITH_FRESH_RC_ARG_DOES_NOT_LEAK ) ])

	def test_subscript_as_direct_t_typed_target_compiles( self ) -> None:
		self.assert_programs_run([ ( 'subscript_direct_target', _SUBSCRIPT_AS_DIRECT_T_TYPED_TARGET_COMPILES ) ])

	def test_subscript_result_as_bare_compare_operand_does_not_leak( self ) -> None:
		self.assert_programs_run([ ( 'subscript_bare_compare', _SUBSCRIPT_RESULT_AS_BARE_COMPARE_OPERAND_DOES_NOT_LEAK ) ])

	def test_subscript_assign_and_augassign_still_propagate_with_no_try( self ) -> None:
		self.assert_programs_run([ ( 'subscript_assign_augassign', _SUBSCRIPT_ASSIGN_AND_AUGASSIGN_STILL_PROPAGATE_WITH_NO_TRY ) ])

	def test_subscript_augassign_with_fresh_rc_result_does_not_leak( self ) -> None:
		self.assert_programs_run([ ( 'subscript_augassign_fresh_rc', _SUBSCRIPT_AUGASSIGN_WITH_FRESH_RC_RESULT_DOES_NOT_LEAK ) ])

	def test_checked_arithmetic_inside_generator_body_still_works( self ) -> None:
		self.assert_programs_run([ ( 'checked_arith_generator', _CHECKED_ARITHMETIC_INSIDE_GENERATOR_BODY_STILL_WORKS ) ])

	def test_checked_arithmetic_inside_multistatement_inline_splice_prelude( self ) -> None:
		self.assert_programs_run([ ( 'checked_arith_inline_splice', _CHECKED_ARITHMETIC_INSIDE_MULTISTATEMENT_INLINE_SPLICE_PRELUDE ) ])

	def test_chained_checked_arithmetic_propagates_and_is_catchable( self ) -> None:
		self.assert_programs_run([ ( 'chained_checked_arith', _CHAINED_CHECKED_ARITHMETIC_PROPAGATES_AND_IS_CATCHABLE ) ])

	def test_bare_discarded_generic_call_propagates_and_dispatches( self ) -> None:
		self.assert_programs_run([ ( 'bare_discard_generic', _BARE_DISCARDED_GENERIC_CALL_PROPAGATES_AND_DISPATCHES ) ])

	def test_assigned_generic_call_still_captures_result( self ) -> None:
		self.assert_programs_run([ ( 'assigned_generic', _ASSIGNED_GENERIC_CALL_STILL_CAPTURES_RESULT ) ])

	def test_bare_discarded_union_arg_dispatch_propagates_and_dispatches( self ) -> None:
		self.assert_programs_run([ ( 'bare_discard_union_arg', _BARE_DISCARDED_UNION_ARG_DISPATCH_PROPAGATES_AND_DISPATCHES ) ])

	def test_assigned_union_arg_dispatch_call_still_captures_result( self ) -> None:
		self.assert_programs_run([ ( 'assigned_union_arg', _ASSIGNED_UNION_ARG_DISPATCH_CALL_STILL_CAPTURES_RESULT ) ])

	def test_bare_discarded_union_receiver_dispatch_propagates_and_dispatches( self ) -> None:
		self.assert_programs_run([ ( 'bare_discard_union_recv', _BARE_DISCARDED_UNION_RECEIVER_DISPATCH_PROPAGATES_AND_DISPATCHES ) ])

	def test_assigned_union_receiver_dispatch_call_still_captures_result( self ) -> None:
		self.assert_programs_run([ ( 'assigned_union_recv', _ASSIGNED_UNION_RECEIVER_DISPATCH_CALL_STILL_CAPTURES_RESULT ) ])


class AutoOrThrowDiscardCheckCompileErrorTests( unittest.TestCase ):
	''' compile-error coverage for case 1 (discarded-statement) auto-or_throw
	at the two call-emission tails that used to bypass it entirely (generic
	calls, runtime union-argument dispatch) - mirrors try_except_test.py's
	own TryExceptCompileErrorTests pattern (Discovery/Compiler directly, no
	C compiler needed). '''

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

	def test_bare_discarded_generic_call_with_insufficient_return_is_a_compile_error( self ) -> None:
		code = '\n'.join([
			'class MyError: pass',
			'',
			'def try_insert[T]( key: T, bad: bool ) -> Result[None,MyError]:',
			'	if bad:',
			'		return Result.Err( MyError() )',
			'	return Result.Ok( None )',
			'',
			'def run( bad: bool ) -> i32:',
			'	try_insert( 5, bad )', # discarded, no try, i32 return can't cover MyError
			'	return 0',
		])
		errors = self._lower_and_get_errors( code, 'run' )
		self.assertTrue( errors, 'expected a compile error for the uncovered MyError leaf' )
		self.assertTrue( any( 'MyError' in e for e in errors ), errors )

	def test_bare_discarded_union_arg_dispatch_with_insufficient_return_is_a_compile_error( self ) -> None:
		code = '\n'.join([
			'class FeedError: pass',
			'class Cat: pass',
			'class Dog: pass',
			'',
			'@overload',
			'def feed( pet: Cat, bad: bool ) -> Result[None,FeedError]: ...',
			'@overload',
			'def feed( pet: Dog, bad: bool ) -> Result[None,FeedError]: ...',
			'',
			'def feed( pet: Cat, bad: bool ) -> Result[None,FeedError]:',
			'	if bad:',
			'		return Result.Err( FeedError() )',
			'	return Result.Ok( None )',
			'',
			'def feed( pet: Dog, bad: bool ) -> Result[None,FeedError]:',
			'	if bad:',
			'		return Result.Err( FeedError() )',
			'	return Result.Ok( None )',
			'',
			'def run( pet: Cat|Dog, bad: bool ) -> i32:',
			'	feed( pet, bad )', # discarded, no try, i32 return can't cover FeedError
			'	return 0',
		])
		errors = self._lower_and_get_errors( code, 'run' )
		self.assertTrue( errors, 'expected a compile error for the uncovered FeedError leaf' )
		self.assertTrue( any( 'FeedError' in e for e in errors ), errors )
