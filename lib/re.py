# lib/re.py — Python-`re`-compatible regular expression engine.
#
# Phase 1 (this file, so far): a backtracking VM (opcodes + an explicit
# frame stack, no real C-stack recursion — see PLAN_RE.md's engine-choice
# writeup) covering literals, `.`, character classes (`[...]`, `\d \D \w \W
# \s \S`), greedy quantifiers (`* + ? {m,n}`), alternation `|`, the `^ $`
# anchors, and non-capturing grouping `(...)` (grouping is parser-required
# scaffolding for alternation/quantifier precedence even before capturing
# groups exist — Phase 2 turns the same `(...)` syntax into a real capture
# by allocating it a slot pair instead of being silently transparent).
# Pattern.match/search/fullmatch report only the whole match (group 0) —
# Match.group(n>0) is a Phase 2 addition.
#
# A runaway match (e.g. `(a+)+b` against a long non-matching string) is
# bounded by a step budget (`max_steps`, default 65536), not a length cap —
# see the plan's "Runaway-match protection" section for why a length cap
# alone can't actually stop catastrophic backtracking. Exceeding the budget
# is reported as MatchError.StepLimitExceeded, kept distinct from
# MatchError.NoMatch, since a caller using this for security-relevant
# validation must not silently treat "gave up" as "definitely no match".

import builtins
import compiler
import sys


# ---------------------------------------------------------------------------
# str-slicing workaround — str._byte_slice is @private (unreachable from an
# ordinary lib module, confirmed by reading lib/builtins/__init__.py), and
# there is no s[a:b] operator. This rebuilds the same result from public
# API only: a bytearray one byte larger than the piece (zero-initialized,
# so the extra trailing byte is already a valid null terminator), filled in
# via public byte-pointer indexing, then handed to str.from_cstr to
# validate + take ownership. If a real public str slice ever lands, this
# function goes away and callers switch to it directly.
# ---------------------------------------------------------------------------

def _substr( source: str, start: usize, end: usize ) -> str:
	''' source[start:end] (byte offsets, must land on UTF-8 codepoint
	boundaries) as a new, independently-owned str. A bad range here is
	always a matcher bug (an internal slot pair pointing outside the
	string), never runtime-supplied data, so it panics rather than
	returning a Result — same posture as lib/guid.py's hex-digit parsing. '''
	with compiler.panic_arithmetic( 're._substr: end < start' ):
		piece_len: usize = end - start
		buf_size: usize = piece_len + 1
	buf = bytearray( buf_size )
	src: ConstPtr[u8] = source.get_cstr()
	dest: Ptr[u8] = buf.get_ptr()
	i: usize = 0
	while i < piece_len:
		with compiler.panic_arithmetic( 're._substr: out of bounds' ):
			dest[i] = src[start + i]
		with compiler.wrap_arithmetic:
			i += 1
	return str.from_cstr( move( buf )).unwrap( 're._substr: invalid UTF-8 boundary' )


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------

class PatternError:
	''' re.compile()'s own error payload — a malformed pattern is routinely
	runtime-supplied data (user search terms, config-driven validation),
	not a programmer typo, so compile() returns Result[Pattern,PatternError]
	rather than panicking (see lib/guid.py's own panic-vs-Result split for
	the same reasoning applied the other way, on GUID literals). '''
	message: str

	def __init__( self, message: str ) -> None:
		self.message = message


@enum( i32 )
class MatchError:
	NoMatch = 0
	StepLimitExceeded = 1


# ---------------------------------------------------------------------------
# Flags — plain OR-able bitmask constants (matches lib/windows/kernel32.py's
# own LCMAP_* / lib/posix/errors.py's own constant style), not an @enum:
# @enum models a single tagged value, not a bitmask several values combine
# into. ASCII is accepted for forward-API-compatibility but is currently a
# no-op — \w/\d are ASCII-only in this engine regardless (see PLAN_RE.md's
# API-surface section); IGNORECASE/MULTILINE/DOTALL are wired in Phase 3.
# ---------------------------------------------------------------------------

IGNORECASE: u32 = 0x01
MULTILINE:  u32 = 0x02
DOTALL:     u32 = 0x04
ASCII:      u32 = 0x08

DEFAULT_MAX_STEPS: usize = 65536


# ---------------------------------------------------------------------------
# Compiled program — opcodes + character classes referenced by IN ops.
# ---------------------------------------------------------------------------

@enum( i32 )
class OpKind:
	CHAR = 0    # match a specific codepoint (op.ch)
	ANY = 1     # match `.`
	IN = 2      # match a character class (op.class_idx into Pattern.classes)
	BOL = 3     # `^`
	EOL = 4     # `$`
	SPLIT = 5   # try target_a first, target_b on backtrack (alternation/quantifiers)
	JUMP = 6    # unconditional jump to target_a
	SAVE = 7    # record the current string position into capture slot `slot`
	MATCH = 8   # whole pattern matched
	WORDB = 9   # `\b` - zero-width word boundary
	NWORDB = 10 # `\B` - zero-width NOT-a-word-boundary


class Op:
	kind: OpKind
	ch: u32
	class_idx: usize
	target_a: usize
	target_b: usize
	slot: usize

	def __init__( self, kind: OpKind ) -> None:
		self.kind = kind
		self.ch = 0
		self.class_idx = 0
		self.target_a = 0
		self.target_b = 0
		self.slot = 0


def _op_char( ch: u32 ) -> Op:
	op = Op( OpKind.CHAR )
	op.ch = ch
	return op

def _op_any() -> Op:
	return Op( OpKind.ANY )

def _op_in( class_idx: usize ) -> Op:
	op = Op( OpKind.IN )
	op.class_idx = class_idx
	return op

def _op_bol() -> Op:
	return Op( OpKind.BOL )

def _op_eol() -> Op:
	return Op( OpKind.EOL )

def _op_wordb() -> Op:
	return Op( OpKind.WORDB )

def _op_nwordb() -> Op:
	return Op( OpKind.NWORDB )

def _op_split( a: usize, b: usize ) -> Op:
	op = Op( OpKind.SPLIT )
	op.target_a = a
	op.target_b = b
	return op

def _op_jump( target: usize ) -> Op:
	op = Op( OpKind.JUMP )
	op.target_a = target
	return op

def _op_save( slot: usize ) -> Op:
	op = Op( OpKind.SAVE )
	op.slot = slot
	return op

def _op_match() -> Op:
	return Op( OpKind.MATCH )


class CharClass:
	''' [...] or a \\d/\\w/\\s shorthand: a set of inclusive codepoint
	ranges, optionally negated ([^...]/\\D/\\W/\\S). '''
	lo: list[u32]
	hi: list[u32]
	negate: bool

	def __init__( self, negate: bool ) -> None:
		self.lo = list[u32]()
		self.hi = list[u32]()
		self.negate = negate

	def add_range( self, lo: u32, hi: u32 ) -> None:
		self.lo.append( lo ).unwrap( 're: CharClass.add_range' )
		self.hi.append( hi ).unwrap( 're: CharClass.add_range' )

	def contains( self, cp: u32 ) -> bool:
		found: bool = False
		i: usize = 0
		n: usize = len( self.lo )
		while i < n:
			lo: u32 = self.lo.__getitem__( i ).unwrap( 're: CharClass.contains' )
			hi: u32 = self.hi.__getitem__( i ).unwrap( 're: CharClass.contains' )
			if cp >= lo and cp <= hi:
				found = True
			with compiler.wrap_arithmetic:
				i += 1
		if self.negate:
			return not found
		return found


# ---------------------------------------------------------------------------
# Fragment helpers — every grammar rule below builds and returns its OWN
# fresh `list[Op]` fragment, with SPLIT/JUMP targets absolute WITHIN that
# fragment's own index space (0..len-1). _append_fragment concatenates a
# fragment onto a growing program by CLONING each op with its targets
# shifted by the destination's current length — cloning (not mutating the
# source in place) is what lets the same fragment be reused several times
# ({m,n} unrolling, e+ desugaring to e followed by e*), since two placements
# of the same fragment need independently-offset targets.
# ---------------------------------------------------------------------------

def _clone_op_at_offset( op: Op, offset: usize ) -> Op:
	out = Op( op.kind )
	out.ch = op.ch
	out.class_idx = op.class_idx
	out.slot = op.slot
	if op.kind == OpKind.SPLIT:
		with compiler.wrap_arithmetic:
			out.target_a = op.target_a + offset
			out.target_b = op.target_b + offset
	elif op.kind == OpKind.JUMP:
		with compiler.wrap_arithmetic:
			out.target_a = op.target_a + offset
	return out

def _append_fragment( dest: list[Op], frag: list[Op] ) -> None:
	offset: usize = len( dest )
	i: usize = 0
	n: usize = len( frag )
	while i < n:
		src_op: Op = frag.__getitem__( i ).unwrap( 're: fragment index' )
		dest.append( _clone_op_at_offset( src_op, offset )).unwrap( 're: fragment append' )
		with compiler.wrap_arithmetic:
			i += 1

def _single_op_fragment( op: Op ) -> list[Op]:
	out: list[Op] = list[Op]()
	out.append( op ).unwrap( 're: single op fragment' )
	return out


# ---------------------------------------------------------------------------
# Quantifiers — each takes an already-compiled atom fragment and returns a
# new fragment; none of these mutate `atom` (see _append_fragment above).
# ---------------------------------------------------------------------------

def _quantify_star( atom: list[Op] ) -> list[Op]:
	# greedy e*:  L0: SPLIT L1,L2 / L1: <e> / JUMP L0 / L2:
	out: list[Op] = list[Op]()
	_append_fragment( out, _single_op_fragment( _op_split( 0, 0 )))
	_append_fragment( out, atom )
	_append_fragment( out, _single_op_fragment( _op_jump( 0 )))
	with compiler.wrap_arithmetic:
		jump_idx: usize = 1 + len( atom )
	split_op: Op = out.__getitem__( 0 ).unwrap( 're: quantify_star patch split' )
	split_op.target_a = 1
	split_op.target_b = len( out )
	jump_op: Op = out.__getitem__( jump_idx ).unwrap( 're: quantify_star patch jump' )
	jump_op.target_a = 0
	return out

def _quantify_plus( atom: list[Op] ) -> list[Op]:
	# greedy e+ == e followed by e*
	out: list[Op] = list[Op]()
	_append_fragment( out, atom )
	_append_fragment( out, _quantify_star( atom ))
	return out

def _quantify_optional( atom: list[Op] ) -> list[Op]:
	# greedy e?:  SPLIT L1,L2 / L1: <e> / L2:
	out: list[Op] = list[Op]()
	_append_fragment( out, _single_op_fragment( _op_split( 0, 0 )))
	_append_fragment( out, atom )
	split_op: Op = out.__getitem__( 0 ).unwrap( 're: quantify_optional patch' )
	split_op.target_a = 1
	split_op.target_b = len( out )
	return out

def _quantify_range( atom: list[Op], m: usize, n: usize, unbounded: bool ) -> list[Op]:
	# {m,n}: m mandatory copies, then either e* (unbounded, {m,}) or (n-m)
	# optional copies ({m,n}) — a simple unrolling compiler, not the most
	# code-size-efficient shape (a counted-repeat opcode would be), but
	# correct and simple; revisit if generated program size matters later.
	out: list[Op] = list[Op]()
	i: usize = 0
	while i < m:
		_append_fragment( out, atom )
		with compiler.wrap_arithmetic:
			i += 1
	if unbounded:
		_append_fragment( out, _quantify_star( atom ))
	else:
		i = m
		while i < n:
			_append_fragment( out, _quantify_optional( atom ))
			with compiler.wrap_arithmetic:
				i += 1
	return out


# ---------------------------------------------------------------------------
# Parser — a one-pass recursive-descent compiler straight to Op fragments
# (no separate AST layer). Each parse_* method returns Result[list[Op],
# PatternError], propagated up via .or_return() the same way SYNTAX.md's own
# `codec = Codec.get(encoding).or_return()` example does - a plain `T|None`
# return was tried first here and rejected by the compiler for a generic
# collection type (list[Op]|None), so Result is used throughout instead,
# consistent with the rest of this module and the wider stdlib.
# ---------------------------------------------------------------------------

_BYTE_DOT: u8 = 46          # '.'
_BYTE_STAR: u8 = 42         # '*'
_BYTE_PLUS: u8 = 43         # '+'
_BYTE_QUESTION: u8 = 63     # '?'
_BYTE_LPAREN: u8 = 40       # '('
_BYTE_RPAREN: u8 = 41       # ')'
_BYTE_LBRACKET: u8 = 91     # '['
_BYTE_RBRACKET: u8 = 93     # ']'
_BYTE_LBRACE: u8 = 123      # '{'
_BYTE_RBRACE: u8 = 125      # '}'
_BYTE_BACKSLASH: u8 = 92    # '\'
_BYTE_PIPE: u8 = 124        # '|'
_BYTE_CARET: u8 = 94        # '^'
_BYTE_DOLLAR: u8 = 36       # '$'
_BYTE_COMMA: u8 = 44        # ','
_BYTE_COLON: u8 = 58        # ':'
_BYTE_QUESTION_MARK: u8 = 63

class Parser:
	text: str
	text_len: usize
	pos: usize
	next_slot: usize
	classes: list[CharClass]

	def __init__( self, text: str ) -> None:
		self.text = text
		self.text_len = text.byte_len()
		self.pos = 0
		self.next_slot = 2  # 0/1 are reserved for the whole match's own span
		self.classes = list[CharClass]()

	def _at_end( self ) -> bool:
		return self.pos >= self.text_len

	def _peek_byte( self ) -> u8:
		return self.text.get_cstr()[ self.pos ]

	def _advance_byte( self ) -> None:
		with compiler.wrap_arithmetic:
			self.pos += 1

	def _decode_cp( self ) -> u32:
		width: usize = 0
		cp: u32 = builtins.decode_utf8_at( self.text.get_cstr(), self.pos, compiler.addrof( width ))
		with compiler.wrap_arithmetic:
			self.pos += width
		return cp

	# --- alternation: lowest precedence -------------------------------

	def parse_alt( self ) -> Result[list[Op], PatternError]:
		first: list[Op] = self.parse_concat().or_return()
		if self._at_end() or self._peek_byte() != _BYTE_PIPE:
			return Result.Ok( first )
		branches: list[list[Op]] = list[list[Op]]()
		branches.append( first ).unwrap( 're: parse_alt' )
		while not self._at_end() and self._peek_byte() == _BYTE_PIPE:
			self._advance_byte()
			nxt: list[Op] = self.parse_concat().or_return()
			branches.append( nxt ).unwrap( 're: parse_alt' )
		return Result.Ok( self._build_alternation( branches ))

	def _build_alternation( self, branches: list[list[Op]] ) -> list[Op]:
		# right-folded chain: SPLIT b0, (SPLIT b1, (SPLIT b2, b3)) ...
		count: usize = len( branches )
		with compiler.wrap_arithmetic:
			last_idx: usize = count - 1
			idx: usize = count - 1
		last: list[Op] = branches.__getitem__( last_idx ).unwrap( 're: alternation last branch' )
		acc: list[Op] = last
		while idx > 0:
			with compiler.wrap_arithmetic:
				idx -= 1
			branch: list[Op] = branches.__getitem__( idx ).unwrap( 're: alternation branch' )
			acc = self._alt2( branch, acc )
		return acc

	def _alt2( self, a: list[Op], b: list[Op] ) -> list[Op]:
		# SPLIT L1,L2 / L1: <a> / JUMP L3 / L2: <b> / L3:
		out: list[Op] = list[Op]()
		_append_fragment( out, _single_op_fragment( _op_split( 0, 0 )))
		_append_fragment( out, a )
		jump_idx: usize = len( out )
		_append_fragment( out, _single_op_fragment( _op_jump( 0 )))
		b_start: usize = len( out )
		_append_fragment( out, b )
		split_op: Op = out.__getitem__( 0 ).unwrap( 're: alt2 patch split' )
		split_op.target_a = 1
		split_op.target_b = b_start
		jump_op: Op = out.__getitem__( jump_idx ).unwrap( 're: alt2 patch jump' )
		jump_op.target_a = len( out )
		return out

	# --- concatenation ---------------------------------------------------

	def parse_concat( self ) -> Result[list[Op], PatternError]:
		out: list[Op] = list[Op]()
		while not self._at_end() and self._peek_byte() != _BYTE_PIPE and self._peek_byte() != _BYTE_RPAREN:
			piece: list[Op] = self.parse_repeat().or_return()
			_append_fragment( out, piece )
		return Result.Ok( out )

	# --- postfix quantifiers ---------------------------------------------

	def parse_repeat( self ) -> Result[list[Op], PatternError]:
		atom: list[Op] = self.parse_atom().or_return()
		if self._at_end():
			return Result.Ok( atom )
		b: u8 = self._peek_byte()
		if b == _BYTE_STAR:
			self._advance_byte()
			return Result.Ok( _quantify_star( atom ))
		if b == _BYTE_PLUS:
			self._advance_byte()
			return Result.Ok( _quantify_plus( atom ))
		if b == _BYTE_QUESTION:
			self._advance_byte()
			return Result.Ok( _quantify_optional( atom ))
		if b == _BYTE_LBRACE:
			return self._parse_brace_quantifier( atom )
		return Result.Ok( atom )

	def _parse_brace_quantifier( self, atom: list[Op] ) -> Result[list[Op], PatternError]:
		save_pos: usize = self.pos
		self._advance_byte()  # consume '{'
		m: usize = 0
		have_m: bool = False
		while not self._at_end() and self._is_digit_byte( self._peek_byte()):
			have_m = True
			with compiler.wrap_arithmetic:
				m = m * 10 + usize( self._peek_byte() - 48 )
			self._advance_byte()
		if not have_m:
			# not a real {m,n} quantifier (e.g. a literal '{') - back out
			# and treat '{' as an ordinary literal character instead.
			self.pos = save_pos
			return Result.Ok( atom )
		if not self._at_end() and self._peek_byte() == _BYTE_RBRACE:
			self._advance_byte()
			return Result.Ok( _quantify_range( atom, m, m, False ))
		if self._at_end() or self._peek_byte() != _BYTE_COMMA:
			return Result.Err( PatternError( 're: malformed {m,n} quantifier' ))
		self._advance_byte()  # consume ','
		if not self._at_end() and self._peek_byte() == _BYTE_RBRACE:
			self._advance_byte()
			return Result.Ok( _quantify_range( atom, m, 0, True ))
		n: usize = 0
		have_n: bool = False
		while not self._at_end() and self._is_digit_byte( self._peek_byte()):
			have_n = True
			with compiler.wrap_arithmetic:
				n = n * 10 + usize( self._peek_byte() - 48 )
			self._advance_byte()
		if not have_n or self._at_end() or self._peek_byte() != _BYTE_RBRACE:
			return Result.Err( PatternError( 're: malformed {m,n} quantifier' ))
		self._advance_byte()
		if n < m:
			return Result.Err( PatternError( 're: {m,n} quantifier with n < m' ))
		return Result.Ok( _quantify_range( atom, m, n, False ))

	def _is_digit_byte( self, b: u8 ) -> bool:
		return b >= 48 and b <= 57

	# --- atoms -------------------------------------------------------------

	def parse_atom( self ) -> Result[list[Op], PatternError]:
		if self._at_end():
			return Result.Err( PatternError( 're: unexpected end of pattern' ))
		b: u8 = self._peek_byte()
		if b == _BYTE_DOT:
			self._advance_byte()
			return Result.Ok( _single_op_fragment( _op_any()))
		if b == _BYTE_CARET:
			self._advance_byte()
			return Result.Ok( _single_op_fragment( _op_bol()))
		if b == _BYTE_DOLLAR:
			self._advance_byte()
			return Result.Ok( _single_op_fragment( _op_eol()))
		if b == _BYTE_LPAREN:
			return self._parse_group()
		if b == _BYTE_LBRACKET:
			return self._parse_class()
		if b == _BYTE_BACKSLASH:
			return self._parse_escape_atom()
		if b == _BYTE_RPAREN or b == _BYTE_PIPE:
			return Result.Err( PatternError( 're: unexpected metacharacter' ))
		if b == _BYTE_STAR or b == _BYTE_PLUS or b == _BYTE_QUESTION:
			return Result.Err( PatternError( 're: nothing to repeat' ))
		cp: u32 = self._decode_cp()
		return Result.Ok( _single_op_fragment( _op_char( cp )))

	def _parse_group( self ) -> Result[list[Op], PatternError]:
		self._advance_byte()  # consume '('
		capturing: bool = True
		if not self._at_end() and self._peek_byte() == _BYTE_QUESTION_MARK:
			self._advance_byte()  # consume '?'
			if self._at_end() or self._peek_byte() != _BYTE_COLON:
				return Result.Err( PatternError( 're: unsupported group syntax (only (?:...) is recognized so far)' ))
			self._advance_byte()  # consume ':'
			capturing = False
		start_slot: usize = 0
		end_slot: usize = 0
		if capturing:
			# allocate this group's own slot pair: group N (1-indexed, group
			# 0 is the whole match, reserved as slots 0/1 by compile()) gets
			# slots 2N/2N+1 - the VM's SAVE opcode and Frame's slot_values/
			# slot_set arrays are already sized generically off Pattern's
			# own n_slots, so no matcher changes are needed for this, only
			# allocating the slot numbers here and wrapping this group's
			# fragment in SAVE(start)/SAVE(end) below.
			start_slot = self.next_slot
			with compiler.wrap_arithmetic:
				end_slot = start_slot + 1
				self.next_slot = start_slot + 2
		inner: list[Op] = self.parse_alt().or_return()
		if self._at_end() or self._peek_byte() != _BYTE_RPAREN:
			return Result.Err( PatternError( 're: unbalanced parenthesis' ))
		self._advance_byte()  # consume ')'
		if not capturing:
			return Result.Ok( inner )
		wrapped: list[Op] = list[Op]()
		_append_fragment( wrapped, _single_op_fragment( _op_save( start_slot )))
		_append_fragment( wrapped, inner )
		_append_fragment( wrapped, _single_op_fragment( _op_save( end_slot )))
		return Result.Ok( wrapped )

	def _parse_escape_atom( self ) -> Result[list[Op], PatternError]:
		self._advance_byte()  # consume '\'
		if self._at_end():
			return Result.Err( PatternError( 're: dangling backslash at end of pattern' ))
		b: u8 = self._peek_byte()
		if b == 100 or b == 68 or b == 119 or b == 87 or b == 115 or b == 83:  # d D w W s S
			cc: CharClass = self._shorthand_class_for_byte( b )
			self._advance_byte()
			idx: usize = len( self.classes )
			self.classes.append( cc ).unwrap( 're: register shorthand class' )
			return Result.Ok( _single_op_fragment( _op_in( idx )))
		if b == 98:  # 'b' - zero-width word boundary (outside a class; \b
			self._advance_byte()  # inside a class means backspace instead - see _class_member_cp)
			return Result.Ok( _single_op_fragment( _op_wordb()))
		if b == 66:  # 'B' - zero-width NOT-a-word-boundary
			self._advance_byte()
			return Result.Ok( _single_op_fragment( _op_nwordb()))
		common: u32|None = _common_escape_cp( b )
		if common is not None:
			self._advance_byte()
			return Result.Ok( _single_op_fragment( _op_char( common )))
		# anything else escaped is a literal codepoint (covers `\. \\ \* \+`
		# and friends the same way real regex engines treat "no special
		# meaning for this escape" - as the literal character itself).
		cp: u32 = self._decode_cp()
		return Result.Ok( _single_op_fragment( _op_char( cp )))

	def _shorthand_class_for_byte( self, b: u8 ) -> CharClass:
		# only ever called after _parse_escape_atom's own d/D/w/W/s/S guard
		if b == 100:  # 'd'
			return _digit_class( False )
		if b == 68:   # 'D'
			return _digit_class( True )
		if b == 119:  # 'w'
			return _word_class( False )
		if b == 87:   # 'W'
			return _word_class( True )
		if b == 115:  # 's'
			return _space_class( False )
		return _space_class( True )  # b == 83 ('S'), the only remaining case

	def _parse_class( self ) -> Result[list[Op], PatternError]:
		self._advance_byte()  # consume '['
		negate: bool = False
		if not self._at_end() and self._peek_byte() == _BYTE_CARET:
			negate = True
			self._advance_byte()
		cc = CharClass( negate )
		first: bool = True
		while True:
			if self._at_end():
				return Result.Err( PatternError( 're: unterminated character class' ))
			if self._peek_byte() == _BYTE_RBRACKET and not first:
				break
			first = False
			lo: u32 = self._class_member_cp().or_return()
			if not self._at_end() and self._peek_byte() == 45 and self._class_has_range_after():  # '-'
				self._advance_byte()  # consume '-'
				hi: u32 = self._class_member_cp().or_return()
				if hi < lo:
					return Result.Err( PatternError( 're: character class range out of order' ))
				cc.add_range( lo, hi )
			else:
				cc.add_range( lo, lo )
		self._advance_byte()  # consume ']'
		idx: usize = len( self.classes )
		self.classes.append( cc ).unwrap( 're: register class' )
		return Result.Ok( _single_op_fragment( _op_in( idx )))

	def _class_has_range_after( self ) -> bool:
		# '-' only introduces a range if it's not immediately followed by
		# ']' (a trailing '-' is a literal, e.g. `[a-]`)
		save_pos: usize = self.pos
		self._advance_byte()
		is_range: bool = not self._at_end() and self._peek_byte() != _BYTE_RBRACKET
		self.pos = save_pos
		return is_range

	def _class_member_cp( self ) -> Result[u32, PatternError]:
		if self._peek_byte() == _BYTE_BACKSLASH:
			self._advance_byte()
			if self._at_end():
				return Result.Err( PatternError( 're: dangling backslash in character class' ))
			b: u8 = self._peek_byte()
			if b == 98:  # 'b' inside a class means backspace (0x08), unlike
				self._advance_byte()  # outside a class where it's a word boundary
				return Result.Ok( u32( 8 ))
			common: u32|None = _common_escape_cp( b )
			if common is not None:
				self._advance_byte()
				return Result.Ok( common )
		return Result.Ok( self._decode_cp())


def _common_escape_cp( b: u8 ) -> u32|None:
	''' the handful of backslash escapes whose meaning is the same inside
	and outside a character class (unlike \\b, which is a word-boundary
	assertion outside a class but a literal backspace inside one - see
	Parser._parse_escape_atom/_class_member_cp's own separate \\b handling). '''
	# each return goes through a locally-typed variable, not a bare integer
	# literal - a bare literal return into a T|None union position fails
	# to compile here the same way a bare @enum member did in Result.Err
	# (see the compiler-bug investigation task); an annotated local sidesteps
	# it the same way.
	if b == 110:  # 'n'
		cp_n: u32 = 10
		return cp_n
	if b == 116:  # 't'
		cp_t: u32 = 9
		return cp_t
	if b == 114:  # 'r'
		cp_r: u32 = 13
		return cp_r
	if b == 102:  # 'f'
		cp_f: u32 = 12
		return cp_f
	if b == 118:  # 'v'
		cp_v: u32 = 11
		return cp_v
	if b == 97:   # 'a' (bell)
		cp_a: u32 = 7
		return cp_a
	if b == 48:   # '0' (NUL)
		cp_nul: u32 = 0
		return cp_nul
	return None


def _digit_class( negate: bool ) -> CharClass:
	cc = CharClass( negate )
	cc.add_range( 48, 57 )  # '0'-'9'
	return cc

def _word_class( negate: bool ) -> CharClass:
	cc = CharClass( negate )
	cc.add_range( 48, 57 )   # '0'-'9'
	cc.add_range( 65, 90 )   # 'A'-'Z'
	cc.add_range( 97, 122 )  # 'a'-'z'
	cc.add_range( 95, 95 )   # '_'
	return cc

def _space_class( negate: bool ) -> CharClass:
	cc = CharClass( negate )
	cc.add_range( 32, 32 )  # ' '
	cc.add_range( 9, 13 )   # '\t'..'\r' (tab, newline, vtab, formfeed, cr)
	return cc

def _is_word_byte_cp( cp: u32 ) -> bool:
	# same ASCII-only membership as _word_class(False), as a plain
	# predicate - used by \b/\B, which run once per VM step and shouldn't
	# allocate a fresh CharClass every time.
	if cp >= 48 and cp <= 57:    # '0'-'9'
		return True
	if cp >= 65 and cp <= 90:    # 'A'-'Z'
		return True
	if cp >= 97 and cp <= 122:   # 'a'-'z'
		return True
	if cp == 95:                 # '_'
		return True
	return False


# ---------------------------------------------------------------------------
# Matcher — a backtracking VM over the compiled Op list, using an explicit
# `list[Frame]` stack for choice points instead of real recursion (see
# PLAN_RE.md's engine-choice writeup). `steps` is a running counter shared
# across every start-position attempt within one search() call, so the
# `max_steps` budget bounds the WHOLE call, not just a single attempt.
# ---------------------------------------------------------------------------

class Frame:
	pc: usize
	sp: usize
	slot_values: list[usize]
	slot_set: list[bool]

	def __init__( self, pc: usize, sp: usize, slot_values: list[usize], slot_set: list[bool] ) -> None:
		self.pc = pc
		self.sp = sp
		self.slot_values = slot_values
		self.slot_set = slot_set


def _clone_usize_list( src: list[usize] ) -> list[usize]:
	out: list[usize] = list[usize]()
	i: usize = 0
	n: usize = len( src )
	while i < n:
		out.append( src.__getitem__( i ).unwrap( 're: clone usize list' )).unwrap( 're: clone usize list append' )
		with compiler.wrap_arithmetic:
			i += 1
	return out

def _clone_bool_list( src: list[bool] ) -> list[bool]:
	out: list[bool] = list[bool]()
	i: usize = 0
	n: usize = len( src )
	while i < n:
		out.append( src.__getitem__( i ).unwrap( 're: clone bool list' )).unwrap( 're: clone bool list append' )
		with compiler.wrap_arithmetic:
			i += 1
	return out


class Matcher:
	ops: list[Op]
	classes: list[CharClass]
	text: str
	text_len: usize
	max_steps: usize
	steps: usize
	flags: u32

	def __init__( self, ops: list[Op], classes: list[CharClass], text: str, max_steps: usize, flags: u32 ) -> None:
		self.ops = ops
		self.classes = classes
		self.text = text
		self.text_len = text.byte_len()
		self.max_steps = max_steps
		self.steps = 0
		self.flags = flags

	def _codepoint_at( self, pos: usize ) -> u32:
		width: usize = 0
		return builtins.decode_utf8_at( self.text.get_cstr(), pos, compiler.addrof( width ))

	def _codepoint_width_at( self, pos: usize ) -> usize:
		width: usize = 0
		builtins.decode_utf8_at( self.text.get_cstr(), pos, compiler.addrof( width ))
		return width

	def _is_word_boundary( self, sp: usize ) -> bool:
		before: bool = False
		after: bool = False
		if sp > 0:
			# the raw byte immediately before sp is a word BYTE iff the
			# preceding codepoint was itself that single ASCII word char -
			# \w is ASCII-only in this engine (see _word_class), and any
			# multi-byte UTF-8 codepoint's own last byte is always >= 0x80,
			# which never collides with an ASCII word-char byte value, so
			# no separate "decode the previous codepoint" step is needed.
			with compiler.panic_arithmetic( 're: _is_word_boundary: sp > 0 checked above' ):
				prev_pos: usize = sp - 1
			prev_byte: u8 = self.text.get_cstr()[ prev_pos ]
			with compiler.wrap_arithmetic:
				before = _is_word_byte_cp( u32( prev_byte ))
		if sp < self.text_len:
			after = _is_word_byte_cp( self._codepoint_at( sp ))
		return before != after

	def run_at( self, start_pos: usize, n_slots: usize ) -> Result[Frame, MatchError]:
		''' attempts an anchored match beginning exactly at start_pos.
		Returns the final Frame (whose slot_values/slot_set carry the
		capture results) on success. '''
		pc: usize = 0
		sp: usize = start_pos
		slot_values: list[usize] = list[usize]()
		slot_set: list[bool] = list[bool]()
		i: usize = 0
		while i < n_slots:
			slot_values.append( 0 ).unwrap( 're: run_at init slots' )
			slot_set.append( False ).unwrap( 're: run_at init slots' )
			with compiler.wrap_arithmetic:
				i += 1
		stack: list[Frame] = list[Frame]()

		while True:
			with compiler.wrap_arithmetic:
				self.steps += 1
			if self.steps > self.max_steps:
				step_limit_err: MatchError = MatchError.StepLimitExceeded
				return Result.Err( step_limit_err )

			op: Op = self.ops.__getitem__( pc ).unwrap( 're: run_at pc out of range' )
			matched: bool = True

			if op.kind == OpKind.CHAR:
				if sp < self.text_len and self._codepoint_at( sp ) == op.ch:
					with compiler.wrap_arithmetic:
						sp += self._codepoint_width_at( sp )
				else:
					matched = False
			elif op.kind == OpKind.ANY:
				dotall: bool = ( self.flags & DOTALL ) != 0
				if sp < self.text_len and ( dotall or self._codepoint_at( sp ) != 10 ):  # '\n'
					with compiler.wrap_arithmetic:
						sp += self._codepoint_width_at( sp )
				else:
					matched = False
			elif op.kind == OpKind.IN:
				if sp < self.text_len:
					cc: CharClass = self.classes.__getitem__( op.class_idx ).unwrap( 're: run_at class_idx' )
					if cc.contains( self._codepoint_at( sp )):
						with compiler.wrap_arithmetic:
							sp += self._codepoint_width_at( sp )
					else:
						matched = False
				else:
					matched = False
			elif op.kind == OpKind.BOL:
				matched = sp == 0
				if not matched and ( self.flags & MULTILINE ) != 0 and sp > 0:
					with compiler.panic_arithmetic( 're: run_at BOL: sp > 0 checked above' ):
						prev_pos: usize = sp - 1
					matched = self.text.get_cstr()[ prev_pos ] == 10  # '\n'
			elif op.kind == OpKind.EOL:
				matched = sp == self.text_len
				if not matched and ( self.flags & MULTILINE ) != 0 and sp < self.text_len:
					matched = self.text.get_cstr()[ sp ] == 10  # '\n'
			elif op.kind == OpKind.WORDB:
				matched = self._is_word_boundary( sp )
			elif op.kind == OpKind.NWORDB:
				matched = not self._is_word_boundary( sp )
			elif op.kind == OpKind.SPLIT:
				frame = Frame( op.target_b, sp, _clone_usize_list( slot_values ), _clone_bool_list( slot_set ))
				stack.append( frame ).unwrap( 're: run_at push split frame' )
				pc = op.target_a
				continue
			elif op.kind == OpKind.JUMP:
				pc = op.target_a
				continue
			elif op.kind == OpKind.SAVE:
				slot_values.__setitem__( op.slot, sp ).unwrap( 're: run_at save slot' )
				slot_set.__setitem__( op.slot, True ).unwrap( 're: run_at save slot' )
				with compiler.wrap_arithmetic:
					pc += 1
				continue
			elif op.kind == OpKind.MATCH:
				return Result.Ok( Frame( pc, sp, slot_values, slot_set ))
			else:
				matched = False

			if matched:
				with compiler.wrap_arithmetic:
					pc += 1
				continue

			if len( stack ) == 0:
				no_match_err: MatchError = MatchError.NoMatch
				return Result.Err( no_match_err )
			back: Frame = stack.pop().unwrap( 're: run_at pop backtrack frame' )
			pc = back.pc
			sp = back.sp
			slot_values = back.slot_values
			slot_set = back.slot_set


# ---------------------------------------------------------------------------
# Public API — Pattern / Match / module-level convenience functions.
# ---------------------------------------------------------------------------

class GroupResult:
	''' one entry of Match.groups() - `list[str|None]` isn't available here
	(a union type isn't accepted as a generic type argument: `list[str|None]`
	itself fails to compile), so this stands in for "str, or nothing" in a
	list context; `matched` false means the group didn't participate (an
	optional/alternated-away capturing group), matching what a plain None
	would have meant in Match.group(n)'s own (non-list, so union-typed
	return is fine there) `str|None`. '''
	text: str
	matched: bool

	def __init__( self, text: str, matched: bool ) -> None:
		self.text = text
		self.matched = matched


class Match:
	# slot 2N/2N+1 holds group N's (start,end) byte offsets (N=0 is the
	# whole match, always set on any successful match); slot_set tracks
	# whether a group actually participated (an optional/alternated-away
	# capturing group leaves its pair unset, matching Python's own
	# group(n) -> None for a group that didn't participate).
	__source: str
	__slot_values: list[usize]
	__slot_set: list[bool]

	def __init__( self, source: str, slot_values: list[usize], slot_set: list[bool] ) -> None:
		self.__source = source
		self.__slot_values = slot_values
		self.__slot_set = slot_set

	def group( self, n: usize = 0 ) -> str|None:
		with compiler.panic_arithmetic( 're: Match.group: group index overflow' ):
			lo_slot: usize = n * 2
			hi_slot: usize = n * 2 + 1
		if not self.__slot_set.__getitem__( lo_slot ).unwrap( 're: Match.group: group index out of range' ):
			return None
		lo: usize = self.__slot_values.__getitem__( lo_slot ).unwrap( 're: Match.group slot' )
		hi: usize = self.__slot_values.__getitem__( hi_slot ).unwrap( 're: Match.group slot' )
		return _substr( self.__source, lo, hi )

	def groups( self ) -> list[GroupResult]:
		out: list[GroupResult] = list[GroupResult]()
		with compiler.panic_arithmetic( 're: Match.groups: unreachable (dividing by the constant 2)' ):
			count: usize = len( self.__slot_values ) // 2 - 1
		i: usize = 1
		while i <= count:
			g: str|None = self.group( i )
			if g is None:
				out.append( GroupResult( '', False )).unwrap( 're: Match.groups append' )
			else:
				out.append( GroupResult( g, True )).unwrap( 're: Match.groups append' )
			with compiler.wrap_arithmetic:
				i += 1
		return out

	def start( self, n: usize = 0 ) -> usize|None:
		with compiler.panic_arithmetic( 're: Match.start: group index overflow' ):
			slot: usize = n * 2
		if not self.__slot_set.__getitem__( slot ).unwrap( 're: Match.start: group index out of range' ):
			return None
		return self.__slot_values.__getitem__( slot ).unwrap( 're: Match.start slot' )

	def end( self, n: usize = 0 ) -> usize|None:
		with compiler.panic_arithmetic( 're: Match.end: group index overflow' ):
			slot: usize = n * 2 + 1
		if not self.__slot_set.__getitem__( slot ).unwrap( 're: Match.end: group index out of range' ):
			return None
		return self.__slot_values.__getitem__( slot ).unwrap( 're: Match.end slot' )

	def span( self ) -> tuple[usize,usize]:
		# group 0 (the whole match) is unconditionally SAVE'd by compile()'s
		# own tail, so it is always set on any Match that exists at all -
		# these None branches are an unreachable defensive backstop, not a
		# real code path. Returning a dummy (0,0) rather than sys.panic()-
		# ing here: an early `return` is what actually narrows `s`/`e` from
		# usize|None to usize on the fallthrough path in this compiler,
		# unlike a sys.panic() call, which isn't recognized as diverging -
		# comparing/returning the un-narrowed union afterward is a compile
		# error (confirmed directly; flagged for the compiler bug report).
		s: usize|None = self.start()
		if s is None:
			return ( 0, 0 )
		e: usize|None = self.end()
		if e is None:
			return ( 0, 0 )
		return ( s, e )


class Pattern:
	__ops: list[Op]
	__classes: list[CharClass]
	__n_slots: usize
	__flags: u32

	def __init__( self, ops: list[Op], classes: list[CharClass], n_slots: usize, flags: u32 ) -> None:
		self.__ops = ops
		self.__classes = classes
		self.__n_slots = n_slots
		self.__flags = flags

	@staticmethod
	def compile( pattern: str, flags: u32 = 0 ) -> Result[Pattern, PatternError]:
		parser = Parser( pattern )
		body: list[Op] = parser.parse_alt().or_return()
		if not parser._at_end():
			return Result.Err( PatternError( 're: unbalanced parenthesis' ))
		prog: list[Op] = list[Op]()
		_append_fragment( prog, _single_op_fragment( _op_save( 0 )))
		_append_fragment( prog, body )
		tail: list[Op] = list[Op]()
		tail.append( _op_save( 1 )).unwrap( 're: compile tail' )
		tail.append( _op_match()).unwrap( 're: compile tail' )
		_append_fragment( prog, tail )
		return Result.Ok( Pattern( prog, parser.classes, parser.next_slot, flags ))

	def search( self, s: str, max_steps: usize = DEFAULT_MAX_STEPS ) -> Result[Match, MatchError]:
		matcher = Matcher( self.__ops, self.__classes, s, max_steps, self.__flags )
		slen: usize = s.byte_len()
		pos: usize = 0
		while True:
			outcome: Result[Frame, MatchError] = matcher.run_at( pos, self.__n_slots )
			match outcome:
				case Result.Ok( frame ):
					return Result.Ok( Match( s, frame.slot_values, frame.slot_set ))
				case Result.Err( e ):
					if e == MatchError.StepLimitExceeded:
						return Result.Err( e )
			if pos >= slen:
				no_match_err: MatchError = MatchError.NoMatch
				return Result.Err( no_match_err )
			with compiler.wrap_arithmetic:
				pos += matcher._codepoint_width_at( pos )

	def match( self, s: str, max_steps: usize = DEFAULT_MAX_STEPS ) -> Result[Match, MatchError]:
		matcher = Matcher( self.__ops, self.__classes, s, max_steps, self.__flags )
		outcome: Result[Frame, MatchError] = matcher.run_at( 0, self.__n_slots )
		match outcome:
			case Result.Ok( frame ):
				return Result.Ok( Match( s, frame.slot_values, frame.slot_set ))
			case Result.Err( e ):
				return Result.Err( e )

	def fullmatch( self, s: str, max_steps: usize = DEFAULT_MAX_STEPS ) -> Result[Match, MatchError]:
		result: Result[Match, MatchError] = self.match( s, max_steps )
		match result:
			case Result.Ok( m ):
				end_pos: usize|None = m.end()
				# group 0 is always set on a successful match - see
				# Match.span()'s own comment on why this is an early
				# `return`, not a sys.panic(), even though unreachable.
				if end_pos is None:
					no_match_err0: MatchError = MatchError.NoMatch
					return Result.Err( no_match_err0 )
				if end_pos == s.byte_len():
					return Result.Ok( m )
				no_match_err: MatchError = MatchError.NoMatch
				return Result.Err( no_match_err )
			case Result.Err( e ):
				return Result.Err( e )


def compile( pattern: str, flags: u32 = 0 ) -> Result[Pattern, PatternError]:
	return Pattern.compile( pattern, flags )

def search( pattern: str, s: str, flags: u32 = 0 ) -> Result[Match, MatchError]:
	p: Pattern = Pattern.compile( pattern, flags ).unwrap( 're.search: invalid pattern' )
	return p.search( s )

def match( pattern: str, s: str, flags: u32 = 0 ) -> Result[Match, MatchError]:
	p: Pattern = Pattern.compile( pattern, flags ).unwrap( 're.match: invalid pattern' )
	return p.match( s )

def fullmatch( pattern: str, s: str, flags: u32 = 0 ) -> Result[Match, MatchError]:
	p: Pattern = Pattern.compile( pattern, flags ).unwrap( 're.fullmatch: invalid pattern' )
	return p.fullmatch( s )
