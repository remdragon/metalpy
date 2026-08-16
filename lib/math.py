# lib/math.py — numeric primitives this compiler's own operators don't quite
# give you (unlike Python, whose math module never needs a floor-division
# helper because // and % already do the right thing natively).
#
# // and % in this compiler are C-style TRUNCATING, not Python-style floor,
# despite // being spelled like Python's floor-div operator (verified
# directly by reading emitter_c.py's _emit_int_division, which emits raw C
# `/`/`%`). floordiv_i64/floormod_i64 are the standard fixup for any caller
# that needs real floor semantics (Python-matching negative-remainder
# normalization, Howard Hinnant-style calendar math, ...) - lifted out of
# lib/_civil_calendar.py once a second, non-calendar consumer made clear
# these don't belong under a calendar-specific module.

import compiler

def floordiv_i64( a: i64, b: i64 ) -> i64:
	''' Python-style floor division (rounds toward -infinity), unlike this
	compiler's own // (rounds toward zero, C-style). Only ever called with a
	non-zero, non-(-1) literal-shaped divisor in this codebase today -
	panic_arithmetic turns the impossible ZeroDivisionError/OverflowError
	cases this compiler would otherwise force a Result for into what they
	actually are: an unreachable condition, not a real failure mode any real
	caller hits. '''
	with compiler.panic_arithmetic( 'unreachable: floordiv_i64 divisor is never zero in this codebase' ):
		q: i64 = a // b
		r: i64 = a % b
		if r != 0 and ( r < 0 ) != ( b < 0 ):
			q -= 1
		return q

def floormod_i64( a: i64, b: i64 ) -> i64:
	''' Python-style floor modulo (result always has the same sign as b),
	unlike this compiler's own % (sign follows a, C-style). '''
	with compiler.panic_arithmetic( 'unreachable: floormod_i64 divisor is never zero in this codebase' ):
		r: i64 = a % b
		if r != 0 and ( r < 0 ) != ( b < 0 ):
			r += b
		return r
