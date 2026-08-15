# f-string format-spec support for float (f32/f64) - PLAN_STR_FORMAT.md
# item 4.
#
# float stays a bare Scalar intrinsic (no boxed RCClass, unlike int) - the
# methods below attach directly onto f64.names/f32.names via the
# `Scalar.method = fn` sigil (discovery.py's visit_Assign; tested by
# discovery_test.py's ScalarMethodRegistrationTests), the same mechanism
# that already attaches scalar-to-scalar cast methods like usize.__u32__ -
# not a new capability, just the first real user of a hook that was already
# built and tested for exactly this. lowering.py's _find_method/
# _lower_method_call already dispatch to it generically (any receiver whose
# type isn't a CStruct/RCClass falls back to its own `.names` dict), and
# emitter_c.py's _emit_self_operand already passes a Scalar receiver by
# plain value (no cast, no refcounting) - no new machinery needed on either
# side. The receiver parameter below is named `value`, not `self` -
# `self` is special-cased elsewhere in this compiler for REAL class-method
# bodies (@cstruct/@rcclass), and using it on a bare top-level function
# like these doesn't resolve the same way (confirmed by a real "name 'self'
# is not defined" compile error while writing this) - discovery_test.py's
# own ScalarMethodRegistrationTests example (`def my_func(x: usize)`) uses
# an ordinary name for exactly this reason.
#
# Scoped to exactly what f-string format specs need (lowering.py's
# _lower_float_format_spec) - 'f'/'F' (fixed-point, explicit precision)
# only. __str__/__repr__ (bare f"{x}", needing Python's own shortest-
# round-trip default formatting - a materially harder, separate problem)
# stay deferred, same as every OTHER scalar's bare f"{x}"
# (PLAN_STR_FORMAT.md item 6) - f'{1.0:.1f}' never reaches __str__/__repr__
# at all (_lower_fstring_part dispatches an explicit format spec straight
# against the operand's own type, only falling back to __str__/__repr__
# for a spec-less interpolation or an explicit !s/!r).

import sys

# a max-magnitude f64 (~1.8e308) needs at most 309 integer digits - this is
# a generous fixed upper bound for the format buffer, not a tightly computed
# one (matches int's own str conversion's "just alloc enough" style).
_MAX_INTEGER_DIGITS: usize = 320


@private
def _f64_sign_prefix( value: f64, mode: str ) -> str:
	''' the sign CHARACTER (a 0-or-1-codepoint str) an f-string format
	spec's own sign mode ('+', '-', or ' ') should show before this float's
	own magnitude digits - mirrors int._sign_prefix (lib/builtins/__int.py)
	exactly: '-' if value is negative regardless of mode, else mode itself
	if mode != '-' (explicit '+'/' ' show for non-negative values too),
	else '' (the default '-' mode shows nothing for a non-negative value,
	matching Python: f"{5.0:f}" == '5.000000', not '+5.000000'). Uses
	value < 0.0 as the negativity check, same as ordinary float comparison
	everywhere else in this language - a bit-exact -0.0 negative-zero sign
	(Python's f"{-0.0:.1f}" == '-0.0') is a known, deliberately out-of-
	scope edge case for now: no bit-reinterpret/sign-bit-read
	infrastructure exists in this compiler to distinguish -0.0 from 0.0
	otherwise (IEEE754 defines -0.0 == 0.0). '''
	if value < 0.0:
		return str( '-' )
	if mode == '+':
		return str( '+' )
	if mode == ' ':
		return str( ' ' )
	return str( '' )


@private
def _f64_fixed_digits( value: f64, precision: usize ) -> str:
	''' value's own MAGNITUDE (sign ignored - callers prepend it themselves
	via _f64_sign_prefix, the same split int's _to_radix_digits/
	_decimal_digits_with_grouping already keep) as fixed-point decimal text
	with exactly `precision` fractional digits - the f-string format-spec
	'f'/'F' dispatch's actual digit-conversion work (lowering.py's
	_lower_float_format_spec). Built on compiler.format_f64 (a hand-written
	C helper - see its own comment in emitter_c.py's PROLOGUE) rather than
	a hand-rolled metalpy-source conversion: getting float-to-decimal
	rounding exactly right by hand is genuinely hard (naive fractional-
	digit extraction accumulates floating-point error), so this reuses the
	platform's own proven conversion instead - matching int's own design
	choice to push real control flow into plain metalpy source methods
	rather than hand-built IR in lowering.py (see the RC use-after-free
	commit 52333fd int._to_radix_digits' own comment documents), just with
	the numeric conversion itself delegated to compiler.format_f64 instead
	of being hand-rolled here too. '''
	with compiler.wrap_arithmetic:
		magnitude: f64 = -value if value < 0.0 else value
		with compiler.panic_arithmetic( 'an integer-digit bound plus a decimal point plus precision fractional digits plus a zero terminator cannot overflow usize for any real f-string format spec' ):
			buf_size: usize = _MAX_INTEGER_DIGITS + 1 + precision + 1
		buf: Ptr[u8] = sys.alloc[u8]( buf_size )
		n: i32 = compiler.format_f64( buf, buf_size, i32( precision ), magnitude )
		if n < 0:
			sys.free( buf )
			sys.panic( 'f-string float formatting failed' )
		return str._from_owned_cstr( buf, usize( n ) + 1 ).unwrap(
			'compiler.format_f64 produced invalid utf-8 (unreachable - only ASCII digits and \'.\' are ever written)'
		)


f64._sign_prefix = _f64_sign_prefix
f64._fixed_digits = _f64_fixed_digits


@private
def _f32_sign_prefix( value: f32, mode: str ) -> str:
	''' f32 has no format-spec digit conversion of its own - widens to f64
	and delegates, same as _f32_fixed_digits below. Widening f32 -> f64 is
	always exact (every f32 value is exactly representable in f64), so this
	loses no precision beyond what value already had. '''
	return f64( value )._sign_prefix( mode )


@private
def _f32_fixed_digits( value: f32, precision: usize ) -> str:
	return f64( value )._fixed_digits( precision )


f32._sign_prefix = _f32_sign_prefix
f32._fixed_digits = _f32_fixed_digits
