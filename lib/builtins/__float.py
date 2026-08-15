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
# _lower_float_format_spec) - 'f'/'F'/'e'/'E'/'g'/'G'/'%' (fixed-point,
# exponential, general, percent - every float type char fstring_format_
# spec.FORMAT_SPEC_TYPE_CHARS recognizes). __str__/__repr__ (bare f"{x}",
# needing Python's own shortest-round-trip default formatting - a
# materially harder, separate problem) stay deferred, same as every OTHER
# scalar's bare f"{x}" (PLAN_STR_FORMAT.md item 6) - f'{1.0:.1f}' never
# reaches __str__/__repr__ at all (_lower_fstring_part dispatches an
# explicit format spec straight against the operand's own type, only
# falling back to __str__/__repr__ for a spec-less interpolation or an
# explicit !s/!r).

import sys

# a max-magnitude f64 (~1.8e308) needs at most 309 integer digits - this is
# a generous fixed upper bound for the format buffer, not a tightly computed
# one (matches int's own str conversion's "just alloc enough" style). Every
# type char below (f/F/e/E/g/G/%) fits comfortably within it - 'e'/'E'/'g'/
# 'G' never need anywhere near this many integer digits, but reusing one
# generous bound for all of them is simpler than computing a tighter one
# per type char.
_MAX_INTEGER_DIGITS: usize = 320

# printf conversion character ASCII codes - compiler.format_f64's own
# type_char argument (an i32, not a str - see its own comment) is always
# one of these, chosen at compile time by lowering.py's _lower_float_
# format_spec based on the f-string's own literal spec.type.
_TYPE_CHAR_F: i32 = 102 # ord('f')


@private
def _group_integer_part( digits: str, sep: str ) -> str:
	''' inserts `sep` (empty, ',', or '_' - an f-string format spec's own
	grouping option, or '' when none was given) every 3 digits from the
	right into the INTEGER part of `digits` only - everything up to its
	first '.' (if any); the fractional part and any 'e'/'E' exponent suffix
	that follows are left untouched. Matches real Python's own f-string
	grouping semantics for every float type char (f"{1234567.89:,.2f}" ==
	'1,234,567.89') - grouping is a correct no-op wherever there's only
	ever one digit before the decimal point, which this handles for free
	(count <= 3 below): always true for 'e'/'E', and true for 'g'/'G'
	whenever they pick their own exponential form. Same grouping algorithm
	as int's own _decimal_digits_with_grouping (lib/builtins/__int.py) -
	duplicated here in miniature rather than shared, since int's own
	version starts from self.__str__() (its own internal digit
	representation), not an already-in-hand plain string the way
	compiler.format_f64's output already is here; sep='' still reconstructs
	the original text unchanged, same convention int's own version
	documents ("splitting into groups of 3 and joining with nothing
	reconstructs the plain digit text unchanged"), so callers never need to
	special-case "no grouping requested". '''
	dot_index: usize = digits.byte_len()
	match digits.find( str( '.' )):
		case Result.Ok( idx ):
			dot_index = idx
		case Result.Err( _ ):
			pass
	int_part: str = digits._byte_slice( 0, dot_index )
	rest: str = digits._byte_slice( dot_index, digits.byte_len() )
	count: usize = int_part.byte_len() # ASCII-only digit text - byte length is codepoint count here
	if count <= 3:
		# digits is a BORROWED parameter (ordinary, non-move calling
		# convention) - returning it directly as this function's own result
		# needs an explicit incref first, giving the caller a real +1 of its
		# own, the same "explicit incref after a borrowing return" pattern
		# str.concat's own comment documents (lib/builtins/__init__.py) -
		# without it, this function's own local `digits` going out of scope
		# on return double-releases the very value the caller still holds a
		# reference to (confirmed by a real crash/garbage-read while writing
		# this, the exact failure shape that pattern's own comment warns
		# about).
		compiler.incref( digits )
		return digits
	groups: list[str] = list[str]() # least-significant GROUP first
	end: usize = count
	with compiler.wrap_arithmetic:
		while end > 3:
			with compiler.panic_arithmetic( 'bounded by count, cannot overflow' ):
				start: usize = end - 3
			groups.append( int_part._byte_slice( start, end )).unwrap( '_group_integer_part: append failed' )
			end = start
		groups.append( int_part._byte_slice( 0, end )).unwrap( '_group_integer_part: append failed' )
		group_count: usize = groups.__len__()
		ordered: list[str] = list[str]() # most-significant GROUP first
		i: usize = group_count
		with compiler.panic_arithmetic( 'bounded by group_count, cannot underflow' ):
			while i > 0:
				i -= 1
				ordered.append( groups.__getitem__( i ).unwrap( '_group_integer_part: index in bounds by construction' )).unwrap( '_group_integer_part: append failed' )
	return sep.join( ordered ) + rest


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
	otherwise (IEEE754 defines -0.0 == 0.0). Also used for '%' (lowering.py
	calls this on the ORIGINAL, un-scaled value - multiplying by the
	positive constant 100 never changes the sign). '''
	if value < 0.0:
		return str( '-' )
	if mode == '+':
		return str( '+' )
	if mode == ' ':
		return str( ' ' )
	return str( '' )


@private
def _f64_fixed_digits( value: f64, precision: usize, type_char: i32, alt: bool, sep: str ) -> str:
	''' value's own MAGNITUDE (sign ignored - callers prepend it themselves
	via _f64_sign_prefix, the same split int's _to_radix_digits/
	_decimal_digits_with_grouping already keep) as decimal text per a
	printf-style type_char ('f'/'F'/'e'/'E'/'g'/'G' - see fstring_format_
	spec.FORMAT_SPEC_TYPE_CHARS; '%' is NOT passed here, see
	_f64_percent_digits below), with `precision` meaning fractional digits
	for 'f'/'F'/'e'/'E' or significant digits for 'g'/'G' (matching both
	Python's own format-spec precision semantics and C's %g precision
	semantics exactly - no special-casing needed here for that split),
	`alt` the '#' flag (always show the decimal point for 'f'/'F'/'e'/'E',
	keep trailing zeros for 'g'/'G' - passed straight through to
	compiler.format_f64, real snprintf's own '#' already matches Python's
	semantics exactly), and `sep` an f-string format spec's own grouping
	option (',', '_', or '' for none) - the f-string format-spec dispatch's
	actual digit-conversion work (lowering.py's _lower_float_format_spec).
	Built on compiler.format_f64 (a hand-written C helper - see its own
	comment in emitter_c.py's PROLOGUE) rather than a hand-rolled metalpy-
	source conversion: getting float-to-decimal rounding exactly right by
	hand is genuinely hard (naive fractional-digit extraction accumulates
	floating-point error), so this reuses the platform's own proven
	conversion instead - matching int's own design choice to push real
	control flow into plain metalpy source methods rather than hand-built
	IR in lowering.py (see the RC use-after-free commit 52333fd
	int._to_radix_digits' own comment documents), just with the numeric
	conversion itself delegated to compiler.format_f64 instead of being
	hand-rolled here too. Grouping (unlike '#') has no printf equivalent at
	all - it's applied as a separate post-processing pass, _group_integer_
	part, since real snprintf simply doesn't support it for any type char. '''
	with compiler.wrap_arithmetic:
		magnitude: f64 = -value if value < 0.0 else value
		with compiler.panic_arithmetic( 'an integer-digit bound plus a decimal point plus precision fractional digits plus a zero terminator cannot overflow usize for any real f-string format spec' ):
			buf_size: usize = _MAX_INTEGER_DIGITS + 1 + precision + 1
		buf: Ptr[u8] = sys.alloc[u8]( buf_size )
		n: i32 = compiler.format_f64( buf, buf_size, i32( precision ), type_char, alt, magnitude )
		if n < 0:
			sys.free( buf )
			sys.panic( 'f-string float formatting failed' )
		digits: str = str._from_owned_cstr( buf, usize( n ) + 1 ).unwrap(
			'compiler.format_f64 produced invalid utf-8 (unreachable - only ASCII digits, \'.\', and \'e\'/\'E\'/\'+\'/\'-\' are ever written)'
		)
	return _group_integer_part( digits, sep )


@private
def _f64_percent_digits( value: f64, precision: usize, alt: bool, sep: str ) -> str:
	''' '%' (PLAN_STR_FORMAT.md item 4) has no printf equivalent - Python
	defines it as: multiply by 100, format as fixed-point ('f') with the
	given precision (and '#'/grouping, same as any other type char), append
	a literal '%'. Done here in metalpy source (not passed down to
	compiler.format_f64 as some 8th type_char) since it needs a real
	arithmetic step first, not just a different format string - reuses
	_f64_fixed_digits for the actual digit conversion (and its own '#'/
	grouping handling) once scaled, same as every other type char. Sign is
	unaffected by scaling by the positive constant 100, so lowering.py
	still calls _f64_sign_prefix on the ORIGINAL, un-scaled value for this
	case - no separate percent-specific sign handling needed. '''
	with compiler.wrap_arithmetic:
		scaled: f64 = value * 100.0
	return _f64_fixed_digits( scaled, precision, _TYPE_CHAR_F, alt, sep ) + str( '%' )


f64._sign_prefix = _f64_sign_prefix
f64._fixed_digits = _f64_fixed_digits
f64._percent_digits = _f64_percent_digits


@private
def _f32_sign_prefix( value: f32, mode: str ) -> str:
	''' f32 has no format-spec digit conversion of its own - widens to f64
	and delegates, same as _f32_fixed_digits/_f32_percent_digits below.
	Widening f32 -> f64 is always exact (every f32 value is exactly
	representable in f64), so this loses no precision beyond what value
	already had. '''
	return f64( value )._sign_prefix( mode )


@private
def _f32_fixed_digits( value: f32, precision: usize, type_char: i32, alt: bool, sep: str ) -> str:
	return f64( value )._fixed_digits( precision, type_char, alt, sep )


@private
def _f32_percent_digits( value: f32, precision: usize, alt: bool, sep: str ) -> str:
	return f64( value )._percent_digits( precision, alt, sep )


f32._sign_prefix = _f32_sign_prefix
f32._fixed_digits = _f32_fixed_digits
f32._percent_digits = _f32_percent_digits
