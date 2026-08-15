# f-string format-spec parsing (PLAN_FSTRINGS.md follow-up: format specs +
# !a). LITERAL specs only - a format_spec containing any real
# ast.FormattedValue (a dynamic width/precision/etc.) is never handed to
# this parser at all (rejected by its own two callers first: lowering.py's
# FunctionLowering._lower_fstring_part for the runtime path, compile_time_
# transformer.py's _ConstFolder.visit_JoinedStr for the compile-time-fold
# path). Parsing happens entirely in PYTHON, against the joined literal
# text, mirroring CPython's own format-spec mini-language BNF:
#   [[fill]align][sign]["#"]["0"][width][grouping_option]["." precision][type]
#
# Lives in its own module (not lowering.py, where it was first written)
# specifically so compile_time_transformer.py can import it too without a
# circular import: discovery.py already imports compile_time_transformer.py,
# and lowering.py already imports discovery.py, so compile_time_transformer.py
# importing lowering.py directly would close that loop. This module has no
# dependency on either.
#
# Scoped to the subset PLAN_FSTRINGS.md's own follow-up proposal covers -
# see FORMAT_SPEC_TYPE_CHARS below for exactly which type chars parse;
# general '=' alignment (only its '0'-shorthand form is in scope) and any
# other character this doesn't recognize raise FormatSpecError, not a
# best-effort guess.

from dataclasses import dataclass


class FormatSpecError( ValueError ):
	''' raised by parse_format_spec() for anything it can't parse or that's
	outside this pass's own scope - carries a plain, human-readable message
	(no ast node/location baked in, since this module has no dependency on
	ast-node-to-source-location machinery); callers attach location
	themselves (lowering.py wraps it in discovery.fail(str(e), node); compile_
	time_transformer.py's own best-effort fold just catches it and leaves the
	JoinedStr node for lowering.py to fail on properly later, same as any
	other unfoldable element). '''


@dataclass( kw_only = True )
class FStringFormatSpec:
	fill: str = ' '
	align: str|None = None # '<', '>', '^', or '=' (only ever set here via the '0' shorthand - see this module's own header comment on scope)
	sign: str = '-' # '+', '-', or ' ' - '-' is the default (matches Python: only '-' is ever implicit)
	alt: bool = False # '#'
	width: int|None = None
	grouping: str|None = None # ',' or '_'
	precision: int|None = None
	type: str|None = None # one of FORMAT_SPEC_TYPE_CHARS, or None (no type char - a real, distinct case from e.g. 'd': governs default vs decimal-with-no-grouping-adjustment the same way Python's own format() distinguishes them)


# int: decimal (default/'d'), radix ('b'/'o'/'x'/'X'). float: fixed/
# exponential/general/percent ('f'/'F'/'e'/'E'/'g'/'G'/'%') - all recognized
# by the PARSER (so a clear, named error can be produced for whichever of
# these isn't implemented for the operand's own type, rather than a generic
# one), even though only 'f'/'F' (fixed-point) are actually dispatchable for
# float today - validate_float_spec names 'e'/'E'/'g'/'G'/'%' explicitly as
# "not implemented yet" (PLAN_STR_FORMAT.md item 4). str only ever uses 's'
# or no type char. 'c' (int-as-codepoint) and 'n' (locale-aware) are
# deliberately NOT included - this compiler has no locale-awareness anywhere
# else either, and 'c' is rare.
FORMAT_SPEC_TYPE_CHARS = frozenset( 's' 'bdoxX' 'fFeEgG%' )
FORMAT_SPEC_ALIGN_CHARS = frozenset( '<>=^' )
FORMAT_SPEC_SIGN_CHARS = frozenset( '+- ' )
FORMAT_SPEC_GROUPING_CHARS = frozenset( ',_' )


def parse_format_spec( spec: str ) -> FStringFormatSpec:
	''' raises FormatSpecError on anything unparseable or out of this
	pass's own scope. '''
	# left-to-right, matching the BNF in this module's own header comment -
	# each stage consumes a prefix of the remaining text (via a plain index
	# `i`, no regex - no regex/parsing infra exists anywhere else in this
	# compiler either, matching PLAN_FSTRINGS.md's own research finding)
	# and falls through to the next stage unconsumed; whatever's left after
	# every stage has run must be empty, or the whole spec is unparseable.
	result = FStringFormatSpec()
	i = 0
	n = len( spec )

	# [[fill]align]
	if i + 1 < n and spec[i + 1] in FORMAT_SPEC_ALIGN_CHARS:
		result.fill = spec[i]
		result.align = spec[i + 1]
		i += 2
	elif i < n and spec[i] in FORMAT_SPEC_ALIGN_CHARS:
		result.align = spec[i]
		i += 1
	if result.align == '=':
		raise FormatSpecError(
			f"f-string format spec: explicit '=' alignment is not supported yet (only the '0' zero-pad shorthand's own implicit '=' is) - {spec!r}"
		)

	# [sign]
	if i < n and spec[i] in FORMAT_SPEC_SIGN_CHARS:
		result.sign = spec[i]
		i += 1

	# ["#"]
	if i < n and spec[i] == '#':
		result.alt = True
		i += 1

	# ["0"] - shorthand for fill='0', align='=' (sign-aware zero-fill),
	# UNLESS an explicit [[fill]align] was already given above (matches
	# Python: f"{-5:=05d}" keeps its own explicit fill/align, '0' here is
	# just a width digit in that case - but that shape never reaches this
	# branch anyway, since '=' align already failed above; this only
	# matters for e.g. f"{-5:<05d}", where '0' is genuinely the START of
	# width, not a zero-pad shorthand, because align was already '<')
	if i < n and spec[i] == '0' and result.align is None:
		result.align = '='
		result.fill = '0'
		i += 1

	# [width]
	width_start = i
	while i < n and spec[i].isdigit():
		i += 1
	if i > width_start:
		result.width = int( spec[width_start:i] )

	# [grouping_option]
	if i < n and spec[i] in FORMAT_SPEC_GROUPING_CHARS:
		result.grouping = spec[i]
		i += 1

	# ["." precision]
	if i < n and spec[i] == '.':
		i += 1
		precision_start = i
		while i < n and spec[i].isdigit():
			i += 1
		if i == precision_start:
			raise FormatSpecError( f"f-string format spec: {spec!r} - a precision (after '.') needs at least one digit" )
		result.precision = int( spec[precision_start:i] )

	# [type]
	if i < n and spec[i] in FORMAT_SPEC_TYPE_CHARS:
		result.type = spec[i]
		i += 1

	if i != n:
		raise FormatSpecError( f'f-string format spec: {spec!r} - could not parse starting at {spec[i:]!r}' )
	return result


# Type-compatibility validation - shared by lowering.py's own runtime
# dispatch (FunctionLowering._lower_str_format_spec/_lower_int_format_spec,
# which catch FormatSpecError and attach the real ast node/location via
# discovery.fail) and compile_time_transformer.py's own compile-time fold
# (which catches FormatSpecError to mean "leave this JoinedStr for the
# runtime path to fold/fail on properly instead"). ONE set of rules, not
# two independently-maintained copies that could silently drift apart -
# see this module's own header comment on why that drift risk matters here
# specifically (a spec that folds successfully but the runtime path would
# have rejected, or vice-versa, is a real, user-visible inconsistency, not
# just a missed optimization).

def validate_str_spec( spec: FStringFormatSpec ) -> None:
	''' raises FormatSpecError if `spec` isn't valid for a str operand. '''
	if spec.type not in ( None, 's' ):
		raise FormatSpecError( f"f-string format spec: {spec.type!r} is not valid for str (only 's' or no type char is)" )
	if spec.sign != '-':
		raise FormatSpecError( 'f-string format spec: sign is not allowed for str' )
	if spec.alt:
		raise FormatSpecError( "f-string format spec: '#' is not allowed for str" )
	if spec.grouping is not None:
		raise FormatSpecError( f"f-string format spec: cannot specify {spec.grouping!r} grouping with str" )
	if spec.align == '=': # only reachable via the '0' shorthand - explicit '=' already raises inside parse_format_spec itself
		raise FormatSpecError( "f-string format spec: '0' zero-pad (implies '=' alignment) is not allowed for str" )


def validate_int_spec( spec: FStringFormatSpec ) -> None:
	''' raises FormatSpecError if `spec` isn't valid for an int operand. '''
	type_char = spec.type
	if type_char not in ( None, 'd', 'b', 'o', 'x', 'X' ):
		if type_char in ( 'f', 'F', 'e', 'E', 'g', 'G', '%' ):
			raise FormatSpecError( f"f-string format spec: {type_char!r} needs a real float type with formatting support, which doesn't exist yet" )
		raise FormatSpecError( f"f-string format spec: {type_char!r} is not valid for int" )
	if spec.precision is not None:
		raise FormatSpecError( 'f-string format spec: precision is not allowed for int' )
	if spec.grouping is not None and type_char not in ( None, 'd' ):
		raise FormatSpecError( f"f-string format spec: cannot specify {spec.grouping!r} grouping with {type_char!r}" )


def validate_float_spec( spec: FStringFormatSpec ) -> None:
	''' raises FormatSpecError if `spec` isn't valid for a float operand.
	Scoped to 'f'/'F' (fixed-point, explicit precision) only for now -
	PLAN_STR_FORMAT.md item 4's 'e'/'E'/'g'/'G'/'%' (exponential/general/
	percent) stay unimplemented, same "not there yet" shape validate_int_spec
	already gives a float type char reaching int, just the other direction:
	the float type itself now exists (unlike when validate_int_spec's own
	message was written), only most of its format-spec type chars don't yet.
	Precision here means DIGIT COUNT (fractional digits after the point),
	not truncation like str's own precision - sign/'#'/width/fill/align all
	reuse the existing FStringFormatSpec fields unchanged, validated the
	same way str/int's own specs already are. '''
	type_char = spec.type
	if type_char not in ( None, 'f', 'F' ):
		if type_char in ( 'e', 'E', 'g', 'G', '%' ):
			raise FormatSpecError( f"f-string format spec: {type_char!r} is not implemented for float yet (only 'f'/'F' fixed-point are)" )
		raise FormatSpecError( f"f-string format spec: {type_char!r} is not valid for float" )
	if spec.grouping is not None:
		raise FormatSpecError( f"f-string format spec: cannot specify {spec.grouping!r} grouping with float yet" )
	if spec.alt:
		raise FormatSpecError( "f-string format spec: '#' is not implemented for float yet" )
