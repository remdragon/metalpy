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
# Scoped to what f-string format specs need (lowering.py's _lower_float_
# format_spec) - 'f'/'F'/'e'/'E'/'g'/'G'/'%' (fixed-point, exponential,
# general, percent - every float type char fstring_format_spec.
# FORMAT_SPEC_TYPE_CHARS recognizes), PLUS __str__/__repr__ (bare f"{x}")
# and the "no type char, no precision" format-spec shape (f"{x:10}"), both
# needing Python's own shortest-round-trip repr algorithm - see
# _f64_repr_digits_raw below (PLAN_STR_FORMAT.md item 4's own writeup on
# why this was deferred initially, then picked up as a real follow-up).

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
_TYPE_CHAR_G: i32 = 103 # ord('g') - the "no type char at all" default's own underlying conversion, see _f64_none_type_digits_raw
_TYPE_CHAR_E: i32 = 101 # ord('e') - _f64_repr_digits_raw's own shortest-round-trip search always uses this conversion (see its own comment on why 'e', not 'f'/'g')

_DECIMAL_DIGIT_CHARS: str = str( '0123456789' ) # _f64_exponent_text's own hand-built int-to-string table - see its own comment on why not a general-purpose one

# 17 significant digits is always enough to exactly round-trip any IEEE754
# double (the standard DBL_DECIMAL_DIG guarantee) - _f64_repr_digits_raw's
# own search never needs to go further.
_MAX_REPR_SIGNIFICANT_DIGITS: usize = 17
# "d.ddddddddddddddddde+308\0" - 1 leading digit + '.' + up to 16 more +
# 'e' + sign + up to 3 exponent digits + a zero terminator - generous, not
# tightly computed, matching this file's other buffer-sizing constants.
_REPR_SEARCH_BUF_SIZE: usize = 32


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
	found: isize = digits.find( str( '.' ))
	if found != isize( -1 ):
		with compiler.panic_arithmetic( 'bounded by self_len, cannot overflow' ):
			dot_index = usize( found )
	int_part: str = digits._byte_slice( 0, dot_index )
	rest: str = digits._byte_slice( dot_index, digits.byte_len() )
	count: usize = int_part.byte_len() # ASCII-only digit text - byte length is codepoint count here
	if count <= 3:
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
def _f64_fixed_digits_raw( value: f64, precision: usize, type_char: i32, alt: bool ) -> str:
	''' value's own MAGNITUDE (sign ignored - callers prepend it themselves
	via _f64_sign_prefix, the same split int's _to_radix_digits/
	_decimal_digits already keep) as UNGROUPED decimal text per a
	printf-style type_char ('f'/'F'/'e'/'E'/'g'/'G' - see fstring_format_
	spec.FORMAT_SPEC_TYPE_CHARS; '%' is NOT passed here, see
	_f64_percent_digits_raw below), with `precision` meaning fractional
	digits for 'f'/'F'/'e'/'E' or significant digits for 'g'/'G' (matching
	both Python's own format-spec precision semantics and C's %g precision
	semantics exactly - no special-casing needed here for that split), and
	`alt` the '#' flag (always show the decimal point for 'f'/'F'/'e'/'E',
	keep trailing zeros for 'g'/'G' - passed straight through to
	compiler.format_f64, real snprintf's own '#' already matches Python's
	semantics exactly) - the f-string format-spec dispatch's actual
	digit-conversion work (lowering.py's _lower_float_format_spec). Built
	on compiler.format_f64 (a hand-written C helper - see its own comment
	in emitter_c.py's PROLOGUE) rather than a hand-rolled metalpy-source
	conversion: getting float-to-decimal rounding exactly right by hand is
	genuinely hard (naive fractional-digit extraction accumulates
	floating-point error), so this reuses the platform's own proven
	conversion instead - matching int's own design choice to push real
	control flow into plain metalpy source methods rather than hand-built
	IR in lowering.py (see the RC use-after-free commit 52333fd
	int._to_radix_digits' own comment documents), just with the numeric
	conversion itself delegated to compiler.format_f64 instead of being
	hand-rolled here too. Deliberately UNGROUPED (unlike _f64_fixed_digits
	below, which is this function plus grouping) - the '0' zero-pad
	shorthand combined with grouping (,/_) needs to group the PADDING
	digits together with these (lowering.py's _lower_float_format_spec,
	via str._pad_and_group_before_dot), which only works starting from
	ungrouped text - see str._pad_and_group_after_prefix's own comment for
	the real, confirmed bug pre-grouping first would cause here too.

	NaN/infinity are special-cased FIRST, before ever calling compiler.
	format_f64 at all - real Python always shows plain "nan"/"inf" text
	for these, ignoring precision/type_char/alt entirely (f"{nan:.1e}" ==
	f"{nan:.1g}" == f"{nan:.1f}" == 'nan'), which real snprintf does NOT
	reliably give: legacy msvcrt.dll's own _snprintf was confirmed (by a
	real test against this system's own msvcrt.dll) to produce outright
	garbage for infinity ("1.$" for "%.1f" of +inf, not "inf") - a
	correctness bug this special-casing also fixes, not just a
	convenience. compiler.is_nan/is_inf (lowering.py's own new
	intrinsics, mirroring compiler.format_f64's shape) reuse the C-level
	__metalpy_isnan/__metalpy_isinf macros already used for checked
	arithmetic (emitter_c.py's PROLOGUE). '''
	if compiler.is_nan( value ):
		return str( 'nan' )
	if compiler.is_inf( value ):
		return str( 'inf' )
	with compiler.wrap_arithmetic:
		magnitude: f64 = -value if value < 0.0 else value
		with compiler.panic_arithmetic( 'an integer-digit bound plus a decimal point plus precision fractional digits plus a zero terminator cannot overflow usize for any real f-string format spec' ):
			buf_size: usize = _MAX_INTEGER_DIGITS + 1 + precision + 1
		buf: Ptr[u8] = sys.alloc[u8]( buf_size )
		n: i32 = compiler.format_f64( buf, buf_size, i32( precision ), type_char, alt, magnitude )
		if n < 0:
			sys.panic( 'f-string float formatting failed' )
		return str._from_owned_cstr( buf, usize( n ) + 1 ).unwrap(
			'compiler.format_f64 produced invalid utf-8 (unreachable - only ASCII digits, \'.\', and \'e\'/\'E\'/\'+\'/\'-\' are ever written)'
		)


@private
def _f64_fixed_digits( value: f64, precision: usize, type_char: i32, alt: bool, sep: str ) -> str:
	''' _f64_fixed_digits_raw's own output, grouped (_group_integer_part) -
	used by every format-spec path EXCEPT the '0' zero-pad shorthand,
	which needs the raw, ungrouped text instead (see _f64_fixed_digits_raw
	and str._pad_and_group_before_dot's own comments for why). Grouping
	(unlike '#', which compiler.format_f64/real snprintf already handles)
	has no printf equivalent at all - it's applied as a separate post-
	processing pass, since real snprintf simply doesn't support it for any
	type char. '''
	return _group_integer_part( _f64_fixed_digits_raw( value, precision, type_char, alt ), sep )


@private
def _f64_percent_digits_raw( value: f64, precision: usize, alt: bool ) -> str:
	''' like _f64_percent_digits below, minus grouping AND the trailing
	'%' - just value scaled by 100 and formatted as ungrouped fixed-point
	text (see _f64_fixed_digits_raw's own comment on why the '0' zero-pad
	shorthand combined with grouping needs this split). '''
	with compiler.wrap_arithmetic:
		scaled: f64 = value * 100.0
	return _f64_fixed_digits_raw( scaled, precision, _TYPE_CHAR_F, alt )


@private
def _f64_percent_digits( value: f64, precision: usize, alt: bool, sep: str ) -> str:
	''' '%' (PLAN_STR_FORMAT.md item 4) has no printf equivalent - Python
	defines it as: multiply by 100, format as fixed-point ('f') with the
	given precision (and '#'/grouping, same as any other type char), append
	a literal '%'. Done here in metalpy source (not passed down to
	compiler.format_f64 as some 8th type_char) since it needs a real
	arithmetic step first, not just a different format string. Sign is
	unaffected by scaling by the positive constant 100, so lowering.py
	still calls _f64_sign_prefix on the ORIGINAL, un-scaled value for this
	case - no separate percent-specific sign handling needed. '''
	return _group_integer_part( _f64_percent_digits_raw( value, precision, alt ), sep ) + str( '%' )


@private
def _f64_none_type_digits_raw( value: f64, precision: usize, alt: bool ) -> str:
	''' the "no type char at all" default's own UNGROUPED digit text -
	f"{x:.2}" (a literal format spec with no 'f'/'e'/'g'/etc after the
	precision). Real Python's own "None" presentation type is closer to
	'g' than 'f' (PLAN_STR_FORMAT.md item 4's own note) - built on 'g'
	here, via _f64_fixed_digits_raw, with ONE extra tweak 'g' itself
	doesn't have: fixed-point results always keep at least one digit past
	the decimal point (f"{5.0:.2}" == '5.0', not '5.0:.2g}' == '5' - 'g'
	strips a bare integer's own trailing '.0' entirely, "None" doesn't).
	NaN/infinity ("nan"/"inf", handled inside _f64_fixed_digits_raw
	already) are passed through completely unchanged - appending ".0" to
	them would be wrong (there's no Python behavior distinction between
	"None" and 'g' for these; both just show "nan"/"inf"). Exponential-
	form results ('g' switching to scientific notation, e.g. "1.2e+03")
	are also passed through unchanged - the "always show a fractional
	digit" tweak only applies to FIXED-point results. '''
	raw: str = _f64_fixed_digits_raw( value, precision, _TYPE_CHAR_G, alt )
	if raw == str( 'nan' ) or raw == str( 'inf' ):
		return raw
	has_dot: bool = raw.find( str( '.' )) != isize( -1 )
	if has_dot:
		return raw
	has_exp: bool = raw.find( str( 'e' )) != isize( -1 )
	if has_exp:
		return raw
	return raw + str( '.0' )


@private
def _f64_none_type_digits( value: f64, precision: usize, alt: bool, sep: str ) -> str:
	return _group_integer_part( _f64_none_type_digits_raw( value, precision, alt ), sep )


f64._sign_prefix = _f64_sign_prefix
f64._fixed_digits = _f64_fixed_digits
f64._fixed_digits_raw = _f64_fixed_digits_raw
f64._percent_digits = _f64_percent_digits
f64._percent_digits_raw = _f64_percent_digits_raw
f64._none_type_digits = _f64_none_type_digits
f64._none_type_digits_raw = _f64_none_type_digits_raw


@private
def _f64_exponent_text( magnitude: i32 ) -> str:
	''' magnitude is always 0..308 (f64's own decimal exponent range) -
	Python's own repr always shows AT LEAST 2 exponent digits (e.g.
	"e+06", never "e+6"), and never more than the value actually needs
	beyond that (e.g. "e+300", never "e+0300") - hand-built directly
	(3 digits is the most this can ever need) rather than a general-
	purpose int-to-string loop, the same "small bounded case, hand-build
	it" choice _pad_and_group_after_prefix's own width-search loop makes
	elsewhere in this codebase. '''
	with compiler.panic_arithmetic( 'dividing/moduloing by the literals 100/10 never zero-divides' ):
		hundreds: i32 = magnitude // 100
		remainder: i32 = magnitude % 100
		tens: i32 = remainder // 10
		ones: i32 = remainder % 10
	with compiler.panic_arithmetic( 'each of hundreds/tens/ones is always a single decimal digit (0-9), by construction above' ):
		hundreds_i: usize = usize( hundreds )
		tens_i: usize = usize( tens )
		ones_i: usize = usize( ones )
		ones_end: usize = ones_i + 1
		tens_end: usize = tens_i + 1
	one_digit: str = _DECIMAL_DIGIT_CHARS._byte_slice( ones_i, ones_end )
	ten_digit: str = _DECIMAL_DIGIT_CHARS._byte_slice( tens_i, tens_end )
	if hundreds > 0:
		with compiler.panic_arithmetic( 'bounded by hundreds_i, cannot overflow' ):
			hundreds_end: usize = hundreds_i + 1
		hundred_digit: str = _DECIMAL_DIGIT_CHARS._byte_slice( hundreds_i, hundreds_end )
		return hundred_digit + ten_digit + one_digit
	return ten_digit + one_digit


@private
def _f64_repr_from_scientific( sci_text: str ) -> str:
	''' converts compiler.format_f64's own 'e'-conversion output (e.g.
	"1.234568e+06" or "5e+00" - always ASCII digits plus at most one '.',
	one 'e', and one exponent sign, courtesy of compiler.format_f64/real
	snprintf) into Python's own repr text: FIXED notation when
	-4 <= exponent < 16 (confirmed against real Python as the exact
	threshold - 1e15 stays fixed, 1e16 switches to scientific; 1e-4 stays
	fixed, 1e-5 switches - a FIXED threshold, notably NOT tied to how many
	significant digits the value actually needed, unlike plain '%g' - see
	_f64_repr_digits_raw's own comment for why this function exists
	separately from just reusing 'g'), SCIENTIFIC otherwise (mantissa +
	'e' + sign + >=2-digit exponent, e.g. "1e+16"/"5e-324"). Fixed-point
	results always keep at least one fractional digit (Python's own
	"100.0", not "100"), matching _f64_none_type_digits_raw's own "g"
	tweak - unlike that function though, there are no trailing zeros left
	to strip here in the first place, since the digit text this receives
	is already the FEWEST significant digits that round-trip exactly (see
	this function's only caller). '''
	e_index: usize = sci_text.byte_len()
	e_found: isize = sci_text.find( str( 'e' ))
	if e_found != isize( -1 ):
		with compiler.panic_arithmetic( 'bounded by self_len, cannot overflow' ):
			e_index = usize( e_found )
	mantissa: str = sci_text._byte_slice( 0, e_index )
	dot_index: usize = mantissa.byte_len()
	dot_found: isize = mantissa.find( str( '.' ))
	if dot_found != isize( -1 ):
		with compiler.panic_arithmetic( 'bounded by self_len, cannot overflow' ):
			dot_index = usize( dot_found )
	digits: str
	if dot_index < mantissa.byte_len():
		with compiler.panic_arithmetic( 'bounded by mantissa length' ):
			after_dot: usize = dot_index + 1
		digits = mantissa._byte_slice( 0, dot_index ) + mantissa._byte_slice( after_dot, mantissa.byte_len() )
	else:
		digits = mantissa
	digit_count: usize = digits.byte_len()

	with compiler.panic_arithmetic( 'bounded by sci_text length' ):
		exp_start: usize = e_index + 1
	exp_text: str = sci_text._byte_slice( exp_start, sci_text.byte_len() )
	exp_cstr: ConstPtr[u8] = exp_text.get_cstr()
	exp_negative: bool = exp_cstr[0] == 45 # '-'
	exp_len: usize = exp_text.byte_len()
	with compiler.wrap_arithmetic:
		exponent: i32 = 0
		i: usize = 1 # skip the leading sign byte - compiler.format_f64's 'e' output always has one
		while i < exp_len:
			digit: i32 = i32( exp_cstr[i] ) - 48 # '0'
			exponent = exponent * 10 + digit
			i += 1
		if exp_negative:
			exponent = -exponent

	if exponent >= -4 and exponent < 16:
		if exponent >= 0:
			with compiler.wrap_arithmetic:
				int_digit_count: usize = usize( exponent ) + 1
			if int_digit_count >= digit_count:
				return digits.ljust( int_digit_count, str( '0' )) + str( '.0' )
			int_part: str = digits._byte_slice( 0, int_digit_count )
			frac_part: str = digits._byte_slice( int_digit_count, digit_count )
			return int_part + str( '.' ) + frac_part
		with compiler.wrap_arithmetic:
			zero_count: usize = usize( -exponent ) - 1
		leading_zeros: str = str( '' ).rjust( zero_count, str( '0' ))
		return str( '0.' ) + leading_zeros + digits

	mantissa_text: str
	if digit_count > 1:
		mantissa_text = digits._byte_slice( 0, 1 ) + str( '.' ) + digits._byte_slice( 1, digit_count )
	else:
		mantissa_text = digits
	exp_sign: str = str( '-' ) if exponent < 0 else str( '+' )
	with compiler.wrap_arithmetic:
		exp_magnitude: i32 = -exponent if exponent < 0 else exponent
	return mantissa_text + str( 'e' ) + exp_sign + _f64_exponent_text( exp_magnitude )


@private
def _f64_repr_digits_raw( value: f64 ) -> str:
	''' value's own MAGNITUDE (sign ignored - same split every other
	digit-producing function in this file keeps) as Python's own
	shortest-round-trip repr text - what bare f"{x}" (no format spec at
	all, dispatched via __str__/__repr__ below) and f"{x:10}"/f"{x:.2}"'s
	OWN "no type char AND no precision" combination (the one shape
	_f64_none_type_digits_raw's 'g'-based approach doesn't cover - see its
	own comment) both need. Tries increasing precision (1 to 17
	significant digits - always enough, see _MAX_REPR_SIGNIFICANT_DIGITS)
	via compiler.format_f64's own 'e' conversion, re-parsing each
	candidate with compiler.parse_f64 and stopping at the first EXACT
	round-trip - the same "verify by reparsing" technique real dtoa
	implementations are checked against, not a full from-scratch Grisu/
	Ryu-style shortest-digit-string algorithm (substantially more code to
	get right, for no benefit this compiler has any other use for). 'e'
	specifically (not 'f' or 'g'): a fixed conversion type keeps the
	SIGNIFICANT-DIGIT COUNT search independent of DISPLAY FORMAT (fixed
	vs scientific) - Python's own display threshold is fixed at exponent
	16 regardless of how many significant digits were actually needed
	(confirmed against real Python: repr(1e16) == '1e+16', a single
	significant digit, still switches to scientific - if the search used
	'%g' instead, its OWN scientific-notation threshold is tied to
	precision, so searching over 'g' precision would make DISPLAY FORMAT
	change together with digit count, not matching Python at all) -
	_f64_repr_from_scientific applies Python's own real threshold
	separately, afterward, once the true minimal digit count is known.
	NaN/infinity/zero are special-cased first, same as every other digit-
	producing function here - zero specifically because it has no
	meaningful "significant digits" to search for (compiler.format_f64's
	own 'e' conversion of 0.0 is always just "0e+00" regardless of
	precision, which _f64_repr_from_scientific would otherwise turn into
	"0.0" anyway, but skipping the search entirely for a known, constant
	answer is simpler and cheaper). '''
	if compiler.is_nan( value ):
		return str( 'nan' )
	if compiler.is_inf( value ):
		return str( 'inf' )
	if value == 0.0:
		return str( '0.0' )
	with compiler.wrap_arithmetic:
		magnitude: f64 = -value if value < 0.0 else value
		buf: Ptr[u8] = sys.alloc[u8]( _REPR_SEARCH_BUF_SIZE )
		precision: usize = 0
		n: i32 = 0
		while True:
			n = compiler.format_f64( buf, _REPR_SEARCH_BUF_SIZE, i32( precision ), _TYPE_CHAR_E, False, magnitude )
			if n < 0:
				sys.panic( 'f-string float repr formatting failed' )
			parsed: f64 = compiler.parse_f64( compiler.cast( ConstPtr[u8], buf ))
			if parsed == magnitude or precision >= _MAX_REPR_SIGNIFICANT_DIGITS - 1:
				break
			precision += 1
		sci_text: str = str._from_owned_cstr( buf, usize( n ) + 1 ).unwrap(
			'compiler.format_f64 produced invalid utf-8 (unreachable - only ASCII digits, \'.\', \'e\'/\'+\'/\'-\' are ever written)'
		)
	return _f64_repr_from_scientific( sci_text )


@private
def _f64_repr_digits( value: f64, sep: str ) -> str:
	return _group_integer_part( _f64_repr_digits_raw( value ), sep )


@private
def _f64_str( value: f64 ) -> str:
	''' bare f"{x}" (no format spec at all) / str(x) - real Python's own
	str(float)/repr(float) are identical, always (unlike int, where they
	merely happen to coincide) - see _f64_repr below. '''
	return _f64_sign_prefix( value, str( '-' )) + _f64_repr_digits_raw( value )


@private
def _f64_repr( value: f64 ) -> str:
	return _f64_str( value )


f64.__str__ = _f64_str
f64.__repr__ = _f64_repr
f64._repr_digits = _f64_repr_digits
f64._repr_digits_raw = _f64_repr_digits_raw


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
def _f32_fixed_digits_raw( value: f32, precision: usize, type_char: i32, alt: bool ) -> str:
	return f64( value )._fixed_digits_raw( precision, type_char, alt )


@private
def _f32_percent_digits( value: f32, precision: usize, alt: bool, sep: str ) -> str:
	return f64( value )._percent_digits( precision, alt, sep )


@private
def _f32_percent_digits_raw( value: f32, precision: usize, alt: bool ) -> str:
	return f64( value )._percent_digits_raw( precision, alt )


@private
def _f32_none_type_digits( value: f32, precision: usize, alt: bool, sep: str ) -> str:
	return f64( value )._none_type_digits( precision, alt, sep )


@private
def _f32_none_type_digits_raw( value: f32, precision: usize, alt: bool ) -> str:
	return f64( value )._none_type_digits_raw( precision, alt )


@private
def _f32_repr_digits( value: f32, sep: str ) -> str:
	return f64( value )._repr_digits( sep )


@private
def _f32_repr_digits_raw( value: f32 ) -> str:
	return f64( value )._repr_digits_raw()


@private
def _f32_str( value: f32 ) -> str:
	return f64( value ).__str__()


@private
def _f32_repr( value: f32 ) -> str:
	return f64( value ).__repr__()


f32._sign_prefix = _f32_sign_prefix
f32._fixed_digits = _f32_fixed_digits
f32._fixed_digits_raw = _f32_fixed_digits_raw
f32._percent_digits = _f32_percent_digits
f32._percent_digits_raw = _f32_percent_digits_raw
f32._none_type_digits = _f32_none_type_digits
f32._none_type_digits_raw = _f32_none_type_digits_raw
f32._repr_digits = _f32_repr_digits
f32._repr_digits_raw = _f32_repr_digits_raw
f32.__str__ = _f32_str
f32.__repr__ = _f32_repr
