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

def _substr( source: ConstPtr[u8], start: usize, end: usize ) -> str:
	''' source[start:end] (byte offsets, must land on UTF-8 codepoint
	boundaries) as a new, independently-owned str. A bad range here is
	always a matcher bug (an internal slot pair pointing outside the
	string), never runtime-supplied data, so it panics rather than
	returning a Result — same posture as lib/guid.py's hex-digit parsing.
	Takes a raw ConstPtr[u8] rather than a str so it works uniformly for
	both a str subject/pattern (pass s.get_cstr()) and Parser's own
	pattern-text buffer (self.data, already a ConstPtr[u8] - see Parser's
	own byte_mode split). Never called on byte-mode SUBJECT data (that data
	isn't assumed to be valid UTF-8 at all - see Match.group()'s own
	byte-mode panic) - only on str data, or on a pattern's own ASCII-only
	syntax substrings (group names), both always valid UTF-8. '''
	with compiler.panic_arithmetic( 're._substr: end < start' ):
		piece_len: usize = end - start
		buf_size: usize = piece_len + 1
	buf = bytearray( buf_size )
	src: ConstPtr[u8] = source
	dest: Ptr[u8] = buf.get_ptr()
	i: usize = 0
	while i < piece_len:
		with compiler.panic_arithmetic( 're._substr: out of bounds' ):
			dest[i] = src[start + i]
		with compiler.wrap_arithmetic:
			i += 1
	return str.from_cstr( move( buf )).unwrap( 're._substr: invalid UTF-8 boundary' )


def _codepoint_width_at_str( s: str, pos: usize ) -> usize:
	''' byte width of the codepoint at byte offset pos in s - the Pattern-
	level (not Matcher-level) counterpart of Matcher._codepoint_width_at,
	used by finditer/sub/split to step forward by a whole codepoint when
	skipping a non-matching position. '''
	width: usize = 0
	builtins.decode_utf8_at( s.get_cstr(), pos, compiler.addrof( width ))
	return width


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

# short aliases - Python's own re.M/re.I are literally the same integer
# values as re.MULTILINE/re.IGNORECASE, not a separate bit
I: u32 = IGNORECASE
M: u32 = MULTILINE
S: u32 = DOTALL
A: u32 = ASCII

DEFAULT_MAX_STEPS: usize = 65536


def escape( pattern: str ) -> str:
	''' backslash-escapes every character that isn't alnum/underscore - the
	simpler, always-safe "escape everything special-or-not" rule Python's
	own re.escape() used before 3.7 (which narrowed it to just the real
	engine's actual metacharacters), kept broad here rather than hand-
	enumerating this engine's own special-character set: over-escaping a
	literal char is always harmless (`\\a` is just `a` - this parser's own
	"an unrecognized escape falls back to a literal codepoint" convention,
	see e.g. _parse_g_backref's identical fallback), but under-escaping one
	this engine DOES treat specially wouldn't be. '''
	out: str = ''
	i: usize = 0
	n: usize = pattern.byte_len()
	data: ConstPtr[u8] = pattern.get_cstr()
	while i < n:
		width: usize = 0
		cp: u32 = builtins.decode_utf8_at( data, i, compiler.addrof( width ))
		with compiler.panic_arithmetic( 're.escape: codepoint width bounded by n - i, cannot overflow' ):
			piece: str = _substr( data, i, i + width )
		if builtins.is_alnum_cp( cp ) or cp == 95:  # '_'
			out = out + piece
		else:
			out = out + '\\' + piece
		with compiler.wrap_arithmetic:
			i += width
	return out


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
	LOOKAHEAD_POS = 11  # `(?=...)` - zero-width, op.sub must match at sp
	LOOKAHEAD_NEG = 12  # `(?!...)` - zero-width, op.sub must NOT match at sp
	LOOKBEHIND_POS = 13 # `(?<=...)` - zero-width, op.sub must match ending exactly at sp, anchored at sp-op.width
	LOOKBEHIND_NEG = 14 # `(?<!...)` - zero-width negation of the above
	BACKREF = 15        # `\N` / `\g<N>` - op.slot holds the referenced GROUP number (not a slot index - see _op_backref)


class Op:
	kind: OpKind
	ch: u32
	class_idx: usize
	target_a: usize
	target_b: usize
	slot: usize
	width: usize       # LOOKBEHIND_*: fixed byte width to step back from sp
	sub: list[Op]      # LOOKAHEAD_*/LOOKBEHIND_*: the assertion's own self-contained compiled body (ends in MATCH)

	def __init__( self, kind: OpKind ) -> None:
		self.kind = kind
		self.ch = 0
		self.class_idx = 0
		self.target_a = 0
		self.target_b = 0
		self.slot = 0
		self.width = 0
		self.sub = list[Op]()


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

def _op_lookahead( sub: list[Op], negate: bool ) -> Op:
	op = Op( OpKind.LOOKAHEAD_NEG if negate else OpKind.LOOKAHEAD_POS )
	op.sub = sub
	return op

def _op_lookbehind( sub: list[Op], width: usize, negate: bool ) -> Op:
	op = Op( OpKind.LOOKBEHIND_NEG if negate else OpKind.LOOKBEHIND_POS )
	op.sub = sub
	op.width = width
	return op

def _op_backref( group_idx: usize ) -> Op:
	op = Op( OpKind.BACKREF )
	op.slot = group_idx
	return op


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
		self.lo.append( lo )
		self.hi.append( hi )

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

	def contains_ci( self, cp: u32 ) -> bool:
		''' IGNORECASE membership: cp itself, or its ASCII case-flip, either
		matching - handles [a-z]/[A-Z]/[a-zA-Z] alike without needing to
		fold the stored ranges themselves. ASCII-only, consistent with
		this engine's existing \\w/\\d/\\s ASCII-only stance. '''
		if self.contains( cp ):
			return True
		flipped: u32 = _case_flip_ascii( cp )
		if flipped != cp:
			return self.contains( flipped )
		return False


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
	out.width = op.width
	# op.sub (LOOKAHEAD_*/LOOKBEHIND_*) is its own self-contained, already-
	# finished fragment with local-to-itself indices - never mutated after
	# construction (only the OUTER op's target_a/target_b, unused by these
	# kinds, ever get offset-shifted), so sharing the reference across
	# clones is safe, not just an optimization.
	out.sub = op.sub
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
		dest.append( _clone_op_at_offset( src_op, offset ))
		with compiler.wrap_arithmetic:
			i += 1

def _single_op_fragment( op: Op ) -> list[Op]:
	out: list[Op] = list[Op]()
	out.append( op )
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


def _quantify_star_lazy( atom: list[Op] ) -> list[Op]:
	# lazy e*?: same shape as greedy e* but with the SPLIT's two targets
	# swapped - target_a (tried first) skips the loop, target_b (tried
	# only on backtrack) enters it, so the VM prefers "as few repeats as
	# possible" instead of "as many as possible".
	out: list[Op] = list[Op]()
	_append_fragment( out, _single_op_fragment( _op_split( 0, 0 )))
	_append_fragment( out, atom )
	_append_fragment( out, _single_op_fragment( _op_jump( 0 )))
	with compiler.wrap_arithmetic:
		jump_idx: usize = 1 + len( atom )
	split_op: Op = out.__getitem__( 0 ).unwrap( 're: quantify_star_lazy patch split' )
	split_op.target_a = len( out )
	split_op.target_b = 1
	jump_op: Op = out.__getitem__( jump_idx ).unwrap( 're: quantify_star_lazy patch jump' )
	jump_op.target_a = 0
	return out

def _quantify_plus_lazy( atom: list[Op] ) -> list[Op]:
	# lazy e+? == e (still mandatory) followed by e*?
	out: list[Op] = list[Op]()
	_append_fragment( out, atom )
	_append_fragment( out, _quantify_star_lazy( atom ))
	return out

def _quantify_optional_lazy( atom: list[Op] ) -> list[Op]:
	# lazy e??: SPLIT targets swapped vs greedy e? - skip tried first.
	out: list[Op] = list[Op]()
	_append_fragment( out, _single_op_fragment( _op_split( 0, 0 )))
	_append_fragment( out, atom )
	split_op: Op = out.__getitem__( 0 ).unwrap( 're: quantify_optional_lazy patch' )
	split_op.target_a = len( out )
	split_op.target_b = 1
	return out

def _quantify_range_lazy( atom: list[Op], m: usize, n: usize, unbounded: bool ) -> list[Op]:
	# {m,n}?: m mandatory copies (no choice involved, same as greedy),
	# then the lazy variant of the remaining repetition.
	out: list[Op] = list[Op]()
	i: usize = 0
	while i < m:
		_append_fragment( out, atom )
		with compiler.wrap_arithmetic:
			i += 1
	if unbounded:
		_append_fragment( out, _quantify_star_lazy( atom ))
	else:
		i = m
		while i < n:
			_append_fragment( out, _quantify_optional_lazy( atom ))
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
	# raw (pointer, length) instead of `text: str` - a bytes pattern (e.g.
	# re.compile(b'...')) parses through the exact same grammar as a str
	# one; only _decode_cp's own per-codepoint step differs (see byte_mode
	# below), same design as Matcher's identical data/byte_mode split.
	data: ConstPtr[u8]
	text_len: usize
	byte_mode: bool
	pos: usize
	next_slot: usize
	classes: list[CharClass]
	group_names: dict[str, usize]  # (?P<name>...) -> group number

	def __init__( self, data: ConstPtr[u8], text_len: usize, byte_mode: bool ) -> None:
		self.data = data
		self.text_len = text_len
		self.byte_mode = byte_mode
		self.pos = 0
		self.next_slot = 2  # 0/1 are reserved for the whole match's own span
		self.classes = list[CharClass]()
		self.group_names = dict[str, usize]()

	def _at_end( self ) -> bool:
		return self.pos >= self.text_len

	def _peek_byte( self ) -> u8:
		return self.data[ self.pos ]

	def _advance_byte( self ) -> None:
		with compiler.wrap_arithmetic:
			self.pos += 1

	def _decode_cp( self ) -> u32:
		# byte_mode: one raw byte, zero-extended, NOT decode_utf8_at - see
		# Matcher._codepoint_at's identical comment on why (unchecked
		# decoder, a byte-mode pattern's own bytes aren't assumed to be
		# valid UTF-8 either, e.g. `re.compile(b'\xff')` as a literal).
		if self.byte_mode:
			cp: u32 = u32( self.data[ self.pos ] )
			with compiler.wrap_arithmetic:
				self.pos += 1
			return cp
		width: usize = 0
		cp2: u32 = builtins.decode_utf8_at( self.data, self.pos, compiler.addrof( width ))
		with compiler.wrap_arithmetic:
			self.pos += width
		return cp2

	# --- alternation: lowest precedence -------------------------------

	def parse_alt( self ) -> Result[list[Op], PatternError]:
		first: list[Op] = self.parse_concat().or_return()
		if self._at_end() or self._peek_byte() != _BYTE_PIPE:
			return Result.Ok( first )
		branches: list[list[Op]] = list[list[Op]]()
		branches.append( first )
		while not self._at_end() and self._peek_byte() == _BYTE_PIPE:
			self._advance_byte()
			nxt: list[Op] = self.parse_concat().or_return()
			branches.append( nxt )
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
			if self._consume_lazy_marker():
				return Result.Ok( _quantify_star_lazy( atom ))
			return Result.Ok( _quantify_star( atom ))
		if b == _BYTE_PLUS:
			self._advance_byte()
			if self._consume_lazy_marker():
				return Result.Ok( _quantify_plus_lazy( atom ))
			return Result.Ok( _quantify_plus( atom ))
		if b == _BYTE_QUESTION:
			self._advance_byte()
			if self._consume_lazy_marker():
				return Result.Ok( _quantify_optional_lazy( atom ))
			return Result.Ok( _quantify_optional( atom ))
		if b == _BYTE_LBRACE:
			return self._parse_brace_quantifier( atom )
		return Result.Ok( atom )

	def _consume_lazy_marker( self ) -> bool:
		''' the trailing `?` that makes a quantifier lazy (`*?` `+?` `??`
		`{m,n}?`) rather than greedy - a plain, unconsumed `?` right after
		one of those is otherwise meaningless (Python rejects it as
		"multiple repeat" too), so treating it as this marker whenever it
		appears is unambiguous. '''
		if not self._at_end() and self._peek_byte() == _BYTE_QUESTION:
			self._advance_byte()
			return True
		return False

	def _finish_quantify_range( self, atom: list[Op], m: usize, n: usize, unbounded: bool ) -> list[Op]:
		if self._consume_lazy_marker():
			return _quantify_range_lazy( atom, m, n, unbounded )
		return _quantify_range( atom, m, n, unbounded )

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
			return Result.Ok( self._finish_quantify_range( atom, m, m, False ))
		if self._at_end() or self._peek_byte() != _BYTE_COMMA:
			return Result.Err( PatternError( 're: malformed {m,n} quantifier' ))
		self._advance_byte()  # consume ','
		if not self._at_end() and self._peek_byte() == _BYTE_RBRACE:
			self._advance_byte()
			return Result.Ok( self._finish_quantify_range( atom, m, 0, True ))
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
		return Result.Ok( self._finish_quantify_range( atom, m, n, False ))

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
		is_lookahead: bool = False
		is_lookbehind: bool = False
		negate_lookaround: bool = False
		group_name: str = ''  # '' means unnamed - empty names are rejected below, so this doubles as "no name"
		if not self._at_end() and self._peek_byte() == _BYTE_QUESTION_MARK:
			self._advance_byte()  # consume '?'
			if self._at_end():
				return Result.Err( PatternError( 're: unexpected end of pattern after (?' ))
			qb: u8 = self._peek_byte()
			if qb == _BYTE_COLON:
				self._advance_byte()
				capturing = False
			elif qb == 61:  # '=' -> (?=...) positive lookahead
				self._advance_byte()
				capturing = False
				is_lookahead = True
			elif qb == 33:  # '!' -> (?!...) negative lookahead
				self._advance_byte()
				capturing = False
				is_lookahead = True
				negate_lookaround = True
			elif qb == 80:  # 'P' -> (?P<name>...) named capturing group
				self._advance_byte()  # consume 'P'
				if self._at_end() or self._peek_byte() != 60:  # '<'
					return Result.Err( PatternError( 're: unsupported group syntax (only (?P<name>...) is recognized after (?P)' ))
				self._advance_byte()  # consume '<'
				name_start: usize = self.pos
				while not self._at_end() and self._peek_byte() != 62:  # '>'
					self._advance_byte()
				if self._at_end():
					return Result.Err( PatternError( 're: unterminated (?P<name>...) group name' ))
				group_name = _substr( self.data, name_start, self.pos )
				if group_name == '':
					return Result.Err( PatternError( 're: empty group name in (?P<...>...)' ))
				self._advance_byte()  # consume '>'
				# capturing stays True - falls through to ordinary slot
				# allocation below, same as a bare (...)
			elif qb == 60:  # '<' -> (?<=...)/(?<!...) lookbehind (Python has no bare (?<name>...))
				self._advance_byte()
				if self._at_end():
					return Result.Err( PatternError( 're: unexpected end of pattern after (?<' ))
				lb: u8 = self._peek_byte()
				if lb == 61:  # '='
					self._advance_byte()
					capturing = False
					is_lookbehind = True
				elif lb == 33:  # '!'
					self._advance_byte()
					capturing = False
					is_lookbehind = True
					negate_lookaround = True
				else:
					return Result.Err( PatternError( 're: unsupported group syntax (only (?<=...)/(?<!...) lookbehind recognized after (?<)' ))
			else:
				return Result.Err( PatternError( 're: unsupported group syntax (recognized: (?:...) (?=...) (?!...) (?<=...) (?<!...) (?P<name>...))' ))
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
			if group_name != '':
				if self.group_names.__getitem__( group_name ).is_ok():
					return Result.Err( PatternError( 're: redefinition of group name ' + group_name ))
				with compiler.panic_arithmetic( 're: _parse_group: unreachable (start_slot always even, >= 2)' ):
					group_index: usize = start_slot // 2
				self.group_names.__setitem__( group_name, group_index )
		inner: list[Op] = self.parse_alt().or_return()
		if self._at_end() or self._peek_byte() != _BYTE_RPAREN:
			return Result.Err( PatternError( 're: unbalanced parenthesis' ))
		self._advance_byte()  # consume ')'
		if is_lookahead:
			sub_prog: list[Op] = list[Op]()
			_append_fragment( sub_prog, inner )
			_append_fragment( sub_prog, _single_op_fragment( _op_match()))
			return Result.Ok( _single_op_fragment( _op_lookahead( sub_prog, negate_lookaround )))
		if is_lookbehind:
			width: usize|None = _fragment_fixed_byte_width( inner, self.classes )
			if width is None:
				return Result.Err( PatternError( 're: look-behind requires a fixed-width pattern' ))
			sub_prog2: list[Op] = list[Op]()
			_append_fragment( sub_prog2, inner )
			_append_fragment( sub_prog2, _single_op_fragment( _op_match()))
			return Result.Ok( _single_op_fragment( _op_lookbehind( sub_prog2, width, negate_lookaround )))
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
			self.classes.append( cc )
			return Result.Ok( _single_op_fragment( _op_in( idx )))
		if b == 98:  # 'b' - zero-width word boundary (outside a class; \b
			self._advance_byte()  # inside a class means backspace instead - see _class_member_cp)
			return Result.Ok( _single_op_fragment( _op_wordb()))
		if b == 66:  # 'B' - zero-width NOT-a-word-boundary
			self._advance_byte()
			return Result.Ok( _single_op_fragment( _op_nwordb()))
		if b >= 49 and b <= 57:  # '1'-'9' - numbered backreference (\0 is NUL, handled below)
			return self._parse_numbered_backref()
		if b == 103:  # 'g' - possible \g<N> explicit numbered backreference
			return self._parse_g_backref()
		common: u32|None = _common_escape_cp( b )
		if common is not None:
			self._advance_byte()
			return Result.Ok( _single_op_fragment( _op_char( common )))
		# anything else escaped is a literal codepoint (covers `\. \\ \* \+`
		# and friends the same way real regex engines treat "no special
		# meaning for this escape" - as the literal character itself).
		cp: u32 = self._decode_cp()
		return Result.Ok( _single_op_fragment( _op_char( cp )))

	def _parse_numbered_backref( self ) -> Result[list[Op], PatternError]:
		n: usize = 0
		while not self._at_end() and self._is_digit_byte( self._peek_byte()):
			with compiler.wrap_arithmetic:
				n = n * 10 + usize( self._peek_byte() - 48 )
			self._advance_byte()
		return self._backref_fragment( n )

	def _parse_g_backref( self ) -> Result[list[Op], PatternError]:
		# \g<N> (numeric) or \g<name> (looked up against group_names, the
		# same name->group-number map (?P<name>...) populates - a name
		# not yet registered there is rejected the same way a numeric
		# forward reference is: self.group_names only contains groups
		# OPENED so far during this left-to-right parse, so "not found"
		# and "forward reference" are naturally the same case, no extra
		# bookkeeping needed. If it doesn't look like \g<...> at all,
		# 'g' falls back to an ordinary literal codepoint, matching this
		# parser's "unrecognized escape is a literal" convention.
		save_pos: usize = self.pos
		self._advance_byte()  # consume 'g'
		if self._at_end() or self._peek_byte() != 60:  # '<'
			self.pos = save_pos
			cp: u32 = self._decode_cp()
			return Result.Ok( _single_op_fragment( _op_char( cp )))
		self._advance_byte()  # consume '<'
		content_start: usize = self.pos
		while not self._at_end() and self._peek_byte() != 62:  # '>'
			self._advance_byte()
		if self._at_end():
			return Result.Err( PatternError( 're: unterminated \\g<...>' ))
		content: str = _substr( self.data, content_start, self.pos )
		self._advance_byte()  # consume '>'
		if content == '':
			return Result.Err( PatternError( 're: \\g<...> group reference cannot be empty' ))
		if self._is_digit_byte( content.get_cstr()[0] ):
			return self._parse_g_backref_numeric( content )
		idx_result: Result[usize, KeyError] = self.group_names.__getitem__( content )
		if idx_result.is_err():
			return Result.Err( PatternError( 're: \\g<...> references an undefined group name ' + content ))
		idx: usize = idx_result.unwrap( 're: checked ok above' )
		return self._backref_fragment( idx )

	def _parse_g_backref_numeric( self, content: str ) -> Result[list[Op], PatternError]:
		n: usize = 0
		i: usize = 0
		clen: usize = content.byte_len()
		data: ConstPtr[u8] = content.get_cstr()
		while i < clen:
			b: u8 = data[i]
			if not self._is_digit_byte( b ):
				return Result.Err( PatternError( 're: \\g<...> numeric group reference must be all digits' ))
			with compiler.wrap_arithmetic:
				n = n * 10 + usize( b - 48 )
				i += 1
		return self._backref_fragment( n )

	def _backref_fragment( self, n: usize ) -> Result[list[Op], PatternError]:
		if n == 0:
			return Result.Err( PatternError( 're: \\0 is not a valid backreference (group numbering starts at 1)' ))
		# a backreference can only be validated against groups OPENED so
		# far (this parser has no separate lookahead pass over the whole
		# pattern) - a forward reference to a not-yet-parsed group is
		# rejected here rather than silently compiled into an op that can
		# never succeed at match time.
		with compiler.panic_arithmetic( 're: _backref_fragment: next_slot underflow (unreachable, starts at 2)' ):
			groups_so_far: usize = ( self.next_slot - 2 ) // 2
		if n > groups_so_far:
			return Result.Err( PatternError( 're: invalid backreference to a group that is not yet defined' ))
		return Result.Ok( _single_op_fragment( _op_backref( n )))

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
		self.classes.append( cc )
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

def _case_flip_ascii( cp: u32 ) -> u32:
	''' the opposite-case ASCII letter, or cp unchanged if cp isn't an
	ASCII letter at all - used for IGNORECASE (Matcher's CHAR comparison
	and CharClass.contains_ci). ASCII-only, consistent with this engine's
	existing \\w/\\d/\\s ASCII-only stance - not the full-Unicode case
	folding lib/case_folding.py/builtins.case_map_one can do, which would
	be inconsistent with that stance rather than more correct. '''
	if cp >= 97 and cp <= 122:   # 'a'-'z'
		with compiler.wrap_arithmetic:
			return cp - 32
	if cp >= 65 and cp <= 90:    # 'A'-'Z'
		with compiler.wrap_arithmetic:
			return cp + 32
	return cp


def _class_fixed_byte_width( cls: CharClass ) -> usize|None:
	''' the single UTF-8 byte width every codepoint this class can match
	is guaranteed to have, or None if that varies (a negated class, or
	one whose member ranges span more than one UTF-8 width band) - used
	by _fragment_fixed_byte_width for lookbehind validation. '''
	if cls.negate:
		return None
	n: usize = len( cls.lo )
	if n == 0:
		return None
	common_width: usize = 0
	first: bool = True
	i: usize = 0
	while i < n:
		lo: u32 = cls.lo.__getitem__( i ).unwrap( 're: class width lo' )
		hi: u32 = cls.hi.__getitem__( i ).unwrap( 're: class width hi' )
		w_lo: usize = builtins.utf8_encoded_len( lo )
		w_hi: usize = builtins.utf8_encoded_len( hi )
		if w_lo != w_hi:
			return None
		if first:
			common_width = w_lo
			first = False
		elif w_lo != common_width:
			return None
		with compiler.wrap_arithmetic:
			i += 1
	return common_width


def _fragment_fixed_byte_width( frag: list[Op], classes: list[CharClass] ) -> usize|None:
	''' the fixed byte count every possible execution path through frag
	consumes, or None if it isn't fixed (a SPLIT/JUMP - alternation or a
	variable-count quantifier - or ANY, or a variable-width character
	class, appears anywhere in it). Mirrors Python re's own "look-behind
	requires fixed-width pattern" restriction; an exact-count quantifier
	like {3} is fine (it unrolls to 3 plain copies with no SPLIT/JUMP -
	see _quantify_range), only a variable count (*, +, ?, {m,n} with
	n != m, or {m,}) is rejected. '''
	total: usize = 0
	i: usize = 0
	n: usize = len( frag )
	while i < n:
		op: Op = frag.__getitem__( i ).unwrap( 're: fixed width scan' )
		if op.kind == OpKind.CHAR:
			with compiler.wrap_arithmetic:
				total += builtins.utf8_encoded_len( op.ch )
		elif op.kind == OpKind.IN:
			cls: CharClass = classes.__getitem__( op.class_idx ).unwrap( 're: fixed width scan class' )
			w: usize|None = _class_fixed_byte_width( cls )
			if w is None:
				return None
			with compiler.wrap_arithmetic:
				total += w
		elif op.kind == OpKind.SAVE or op.kind == OpKind.BOL or op.kind == OpKind.EOL or op.kind == OpKind.WORDB or op.kind == OpKind.NWORDB:
			pass  # zero-width
		else:
			return None  # ANY, SPLIT, JUMP, MATCH, LOOKAHEAD_*, LOOKBEHIND_* - variable or not applicable
		with compiler.wrap_arithmetic:
			i += 1
	return total


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
	# Match.lastindex bookkeeping (Python's own "which numbered group most
	# recently closed on the WINNING path") - plain value fields, unlike
	# slot_values/slot_set, so no _clone_*_list needed: usize/bool already
	# copy by value on every assignment, including into a Frame pushed at a
	# SPLIT choice point and restored from one on backtrack-pop.
	# last_group_set stays False (last_group unused) until the first real
	# (numbered >= 1) capturing group's own END save actually executes.
	last_group: usize
	last_group_set: bool

	def __init__(
		self, pc: usize, sp: usize, slot_values: list[usize], slot_set: list[bool],
		last_group: usize, last_group_set: bool,
	) -> None:
		self.pc = pc
		self.sp = sp
		self.slot_values = slot_values
		self.slot_set = slot_set
		self.last_group = last_group
		self.last_group_set = last_group_set


def _clone_usize_list( src: list[usize] ) -> list[usize]:
	out: list[usize] = list[usize]()
	i: usize = 0
	n: usize = len( src )
	while i < n:
		out.append( src.__getitem__( i ).unwrap( 're: clone usize list' ))
		with compiler.wrap_arithmetic:
			i += 1
	return out

def _clone_bool_list( src: list[bool] ) -> list[bool]:
	out: list[bool] = list[bool]()
	i: usize = 0
	n: usize = len( src )
	while i < n:
		out.append( src.__getitem__( i ).unwrap( 're: clone bool list' ))
		with compiler.wrap_arithmetic:
			i += 1
	return out


class Matcher:
	ops: list[Op]
	classes: list[CharClass]
	# raw (pointer, length) instead of `text: str` - str is one possible
	# source (str.get_cstr()/byte_len()), bytes/bytearray/memoryview
	# (whenever that lands) are others; the whole VM below only ever needs
	# byte-level access plus the codepoint-decode entry points immediately
	# below, both gated on byte_mode. See _codepoint_at/_codepoint_width_at.
	data: ConstPtr[u8]
	text_len: usize
	byte_mode: bool
	max_steps: usize
	steps: usize
	flags: u32

	def __init__(
		self, ops: list[Op], classes: list[CharClass], data: ConstPtr[u8], text_len: usize, byte_mode: bool,
		max_steps: usize, flags: u32,
	) -> None:
		self.ops = ops
		self.classes = classes
		self.data = data
		self.text_len = text_len
		self.byte_mode = byte_mode
		self.max_steps = max_steps
		self.steps = 0
		self.flags = flags

	def _codepoint_at( self, pos: usize ) -> u32:
		# byte_mode: one raw byte, zero-extended - NOT builtins.decode_utf8_at,
		# which is UNCHECKED and would misinterpret arbitrary high-bit-set
		# binary bytes as (the start or continuation of) a multi-byte UTF-8
		# sequence rather than erroring - a byte-mode subject is never
		# assumed to be valid UTF-8 at all.
		if self.byte_mode:
			with compiler.wrap_arithmetic:
				return u32( self.data[ pos ] )
		width: usize = 0
		return builtins.decode_utf8_at( self.data, pos, compiler.addrof( width ))

	def _codepoint_width_at( self, pos: usize ) -> usize:
		if self.byte_mode:
			return 1
		width: usize = 0
		builtins.decode_utf8_at( self.data, pos, compiler.addrof( width ))
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
			# no separate "decode the previous codepoint" step is needed
			# (true in byte_mode too: an ASCII word byte is still just
			# itself there).
			with compiler.panic_arithmetic( 're: _is_word_boundary: sp > 0 checked above' ):
				prev_pos: usize = sp - 1
			prev_byte: u8 = self.data[ prev_pos ]
			with compiler.wrap_arithmetic:
				before = _is_word_byte_cp( u32( prev_byte ))
		if sp < self.text_len:
			after = _is_word_byte_cp( self._codepoint_at( sp ))
		return before != after

	def _backref_matches_at( self, sp: usize, g_start: usize, g_len: usize ) -> bool:
		''' does the g_len bytes of self.data starting at g_start
		(an already-closed capture group's own span) reappear literally
		at sp? compares raw bytes, not codepoints - a byte-exact
		reappearance of valid UTF-8 is itself valid UTF-8, so this needs
		no separate decode step (same reasoning str._byte_slice's own
		docstring gives for byte-exact needle matches always landing on
		codepoint boundaries) - and is exactly what byte_mode wants too. '''
		with compiler.wrap_arithmetic:
			end_pos: usize = sp + g_len
		if end_pos > self.text_len:
			return False
		data: ConstPtr[u8] = self.data
		i: usize = 0
		while i < g_len:
			with compiler.wrap_arithmetic:
				a: usize = g_start + i
				b: usize = sp + i
			if data[a] != data[b]:
				return False
			with compiler.wrap_arithmetic:
				i += 1
		return True

	def run_at( self, start_pos: usize, n_slots: usize ) -> Result[Frame, MatchError]:
		''' attempts an anchored match beginning exactly at start_pos, with
		every capture slot starting unset. Returns the final Frame (whose
		slot_values/slot_set carry the capture results) on success. '''
		slot_values: list[usize] = list[usize]()
		slot_set: list[bool] = list[bool]()
		i: usize = 0
		while i < n_slots:
			slot_values.append( 0 )
			slot_set.append( False )
			with compiler.wrap_arithmetic:
				i += 1
		return self._run_from( start_pos, slot_values, slot_set, 0, False )

	def _run_from(
		self, start_pos: usize, slot_values_in: list[usize], slot_set_in: list[bool],
		last_group_in: usize, last_group_set_in: bool,
	) -> Result[Frame, MatchError]:
		''' like run_at, but the caller supplies the starting capture-slot
		state instead of it being reset to unset - used by LOOKAHEAD_*/
		LOOKBEHIND_* to run a nested sub-match that shares (a snapshot of)
		the outer match's own slots, so a capturing group nested inside a
		lookaround still populates the outer Match's groups on success.
		self.ops is swapped to the assertion's own op.sub for the duration
		of the nested run (self.steps/self.max_steps stay shared, so the
		step budget bounds the combined effort of outer + nested matching). '''
		pc: usize = 0
		sp: usize = start_pos
		slot_values: list[usize] = slot_values_in
		slot_set: list[bool] = slot_set_in
		last_group: usize = last_group_in
		last_group_set: bool = last_group_set_in
		stack: list[Frame] = list[Frame]()

		while True:
			with compiler.wrap_arithmetic:
				self.steps += 1
			if self.steps > self.max_steps:
				return Result.Err( MatchError.StepLimitExceeded )

			op: Op = self.ops.__getitem__( pc ).unwrap( 're: run_at pc out of range' )
			matched: bool = True

			if op.kind == OpKind.CHAR:
				ignorecase: bool = ( self.flags & IGNORECASE ) != 0
				char_ok: bool = False
				if sp < self.text_len:
					cp_here: u32 = self._codepoint_at( sp )
					if cp_here == op.ch:
						char_ok = True
					elif ignorecase and _case_flip_ascii( cp_here ) == op.ch:
						char_ok = True
				if char_ok:
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
					cp_in: u32 = self._codepoint_at( sp )
					in_ok: bool = cc.contains( cp_in )
					if not in_ok and ( self.flags & IGNORECASE ) != 0:
						in_ok = cc.contains_ci( cp_in )
					if in_ok:
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
					matched = self.data[ prev_pos ] == 10  # '\n'
			elif op.kind == OpKind.EOL:
				matched = sp == self.text_len
				if not matched and ( self.flags & MULTILINE ) != 0 and sp < self.text_len:
					matched = self.data[ sp ] == 10  # '\n'
			elif op.kind == OpKind.WORDB:
				matched = self._is_word_boundary( sp )
			elif op.kind == OpKind.NWORDB:
				matched = not self._is_word_boundary( sp )
			elif op.kind == OpKind.BACKREF:
				group_idx: usize = op.slot
				with compiler.wrap_arithmetic:
					lo_slot: usize = group_idx * 2
					hi_slot: usize = group_idx * 2 + 1
				if not slot_set.__getitem__( lo_slot ).unwrap( 're: run_at backref slot_set' ):
					# the referenced group never participated in this
					# particular match attempt (e.g. an alternated-away or
					# unmatched optional group) - matches Python re's own
					# "an unmatched group's backreference never matches".
					matched = False
				else:
					g_start: usize = slot_values.__getitem__( lo_slot ).unwrap( 're: run_at backref slot' )
					g_end: usize = slot_values.__getitem__( hi_slot ).unwrap( 're: run_at backref slot' )
					with compiler.panic_arithmetic( 're: run_at backref: end < start (invariant)' ):
						g_len: usize = g_end - g_start
					if self._backref_matches_at( sp, g_start, g_len ):
						with compiler.wrap_arithmetic:
							sp += g_len
					else:
						matched = False
			elif op.kind == OpKind.LOOKAHEAD_POS or op.kind == OpKind.LOOKAHEAD_NEG:
				saved_ops: list[Op] = self.ops
				self.ops = op.sub
				sub_outcome: Result[Frame, MatchError] = self._run_from(
					sp, _clone_usize_list( slot_values ), _clone_bool_list( slot_set ), last_group, last_group_set )
				self.ops = saved_ops
				positive: bool = op.kind == OpKind.LOOKAHEAD_POS
				match sub_outcome:
					case Result.Ok( sub_frame ):
						if positive:
							slot_values = sub_frame.slot_values
							slot_set = sub_frame.slot_set
							last_group = sub_frame.last_group
							last_group_set = sub_frame.last_group_set
							matched = True
						else:
							matched = False
					case Result.Err( sub_err ):
						if sub_err == MatchError.StepLimitExceeded:
							return Result.Err( sub_err )
						matched = not positive
			elif op.kind == OpKind.LOOKBEHIND_POS or op.kind == OpKind.LOOKBEHIND_NEG:
				positive2: bool = op.kind == OpKind.LOOKBEHIND_POS
				if sp < op.width:
					matched = not positive2  # can't look behind far enough - POS fails, NEG succeeds
				else:
					with compiler.wrap_arithmetic:
						behind_start: usize = sp - op.width
					saved_ops2: list[Op] = self.ops
					self.ops = op.sub
					sub_outcome2: Result[Frame, MatchError] = self._run_from(
						behind_start, _clone_usize_list( slot_values ), _clone_bool_list( slot_set ), last_group, last_group_set )
					self.ops = saved_ops2
					match sub_outcome2:
						case Result.Ok( sub_frame2 ):
							if positive2:
								slot_values = sub_frame2.slot_values
								slot_set = sub_frame2.slot_set
								last_group = sub_frame2.last_group
								last_group_set = sub_frame2.last_group_set
								matched = True
							else:
								matched = False
						case Result.Err( sub_err2 ):
							if sub_err2 == MatchError.StepLimitExceeded:
								return Result.Err( sub_err2 )
							matched = not positive2
			elif op.kind == OpKind.SPLIT:
				frame = Frame(
					op.target_b, sp, _clone_usize_list( slot_values ), _clone_bool_list( slot_set ),
					last_group, last_group_set,
				)
				stack.append( frame )
				pc = op.target_a
				continue
			elif op.kind == OpKind.JUMP:
				pc = op.target_a
				continue
			elif op.kind == OpKind.SAVE:
				slot_values.__setitem__( op.slot, sp ).unwrap( 're: run_at save slot' )
				slot_set.__setitem__( op.slot, True ).unwrap( 're: run_at save slot' )
				# an END slot (odd, >= 3) closing a real (numbered >= 1)
				# capturing group - slots 0/1 are the whole match's own
				# span, not a real group, so never update lastindex
				with compiler.panic_arithmetic( 're: run_at: % 2 / // 2 by a nonzero literal, cannot divide by zero' ):
					is_end_slot: bool = ( op.slot % 2 ) == 1
					if is_end_slot and op.slot >= 3:
						last_group = op.slot // 2
						last_group_set = True
				with compiler.wrap_arithmetic:
					pc += 1
				continue
			elif op.kind == OpKind.MATCH:
				return Result.Ok( Frame( pc, sp, slot_values, slot_set, last_group, last_group_set ))
			else:
				matched = False

			if matched:
				with compiler.wrap_arithmetic:
					pc += 1
				continue

			if len( stack ) == 0:
				return Result.Err( MatchError.NoMatch )
			back: Frame = stack.pop().unwrap( 're: run_at pop backtrack frame' )
			pc = back.pc
			sp = back.sp
			slot_values = back.slot_values
			slot_set = back.slot_set
			last_group = back.last_group
			last_group_set = back.last_group_set


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
	# __source is None only for a byte-mode match (the pattern/subject was
	# bytes, not str) - group()/groupdict()/groups() all need a real str to
	# decode a captured span back out of (matching a byte-mode subject
	# isn't guaranteed to BE valid UTF-8, so there's no safe str to hand
	# back), and panic if called there. start()/end()/span()/regs/
	# lastindex never touch __source at all, so they work identically
	# either way - that's everything grap.mpy's own usage needs.
	__source: str|None
	__slot_values: list[usize]
	__slot_set: list[bool]
	__group_names: dict[str, usize]
	__last_group: usize
	__last_group_set: bool

	def __init__(
		self, source: str|None, slot_values: list[usize], slot_set: list[bool], group_names: dict[str, usize],
		last_group: usize, last_group_set: bool,
	) -> None:
		self.__source = source
		self.__slot_values = slot_values
		self.__slot_set = slot_set
		self.__group_names = group_names
		self.__last_group = last_group
		self.__last_group_set = last_group_set

	def group( self, n: usize = 0 ) -> str|None:
		# narrowed via a local, not `self.__source` directly - narrowing a
		# private field access after an is-None check isn't tracked the
		# same way narrowing a local is elsewhere in this codebase (e.g.
		# span()'s own s/e locals below)
		source: str|None = self.__source
		if source is None:
			sys.panic( 're: Match.group() is not supported on a byte-mode match (the pattern/subject was '
				'bytes, not str) - use .span()/.start()/.end()/.regs and slice the original bytes yourself' )
		with compiler.panic_arithmetic( 're: Match.group: group index overflow' ):
			lo_slot: usize = n * 2
			hi_slot: usize = n * 2 + 1
		if not self.__slot_set.__getitem__( lo_slot ).unwrap( 're: Match.group: group index out of range' ):
			return None
		lo: usize = self.__slot_values.__getitem__( lo_slot ).unwrap( 're: Match.group slot' )
		hi: usize = self.__slot_values.__getitem__( hi_slot ).unwrap( 're: Match.group slot' )
		return _substr( source.get_cstr(), lo, hi )

	def group( self, name: str ) -> str|None:
		''' overload resolved by argument type - group() with no args
		still picks the usize overload above (it's the only one with a
		default), group('name') picks this one. Panics on an unknown
		group name - a hardcoded name that doesn't exist in the compiled
		pattern is a programmer error, same posture as an out-of-range
		numeric group index above. '''
		idx_result: Result[usize, KeyError] = self.__group_names.__getitem__( name )
		idx: usize = idx_result.unwrap( 're: Match.group: unknown group name' )
		return self.group( idx )

	def groupdict( self ) -> dict[str, GroupResult]:
		out: dict[str, GroupResult] = dict[str, GroupResult]()
		count: usize = len( self.__group_names )
		i: usize = 0
		while i < count:
			name: str = self.__group_names.key_at( i ).unwrap( 're: Match.groupdict key' )
			idx: usize = self.__group_names.value_at( i ).unwrap( 're: Match.groupdict value' )
			g: str|None = self.group( idx )
			if g is None:
				out.__setitem__( name, GroupResult( '', False ))
			else:
				out.__setitem__( name, GroupResult( g, True ))
			with compiler.wrap_arithmetic:
				i += 1
		return out

	def groups( self ) -> list[GroupResult]:
		out: list[GroupResult] = list[GroupResult]()
		with compiler.panic_arithmetic( 're: Match.groups: unreachable (dividing by the constant 2)' ):
			count: usize = len( self.__slot_values ) // 2 - 1
		i: usize = 1
		while i <= count:
			g: str|None = self.group( i )
			if g is None:
				out.append( GroupResult( '', False ))
			else:
				out.append( GroupResult( g, True ))
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
		# real code path.
		s: usize|None = self.start()
		if s is None:
			sys.panic( 're: Match.span: whole match start unexpectedly unset' )
		e: usize|None = self.end()
		if e is None:
			sys.panic( 're: Match.span: whole match end unexpectedly unset' )
		return ( s, e )

	@property
	def regs( self ) -> list[tuple[i32,i32]]:
		''' one (start,end) pair per group, group 0 (the whole match) first
		- matching Python's own Match.regs. An unset (optional/alternated-
		away) group reports (-1,-1), the same sentinel Python uses (i32, not
		usize, specifically to be able to represent that sentinel). '''
		out: list[tuple[i32,i32]] = list[tuple[i32,i32]]()
		with compiler.panic_arithmetic( 're: Match.regs: unreachable (dividing by the constant 2)' ):
			count: usize = len( self.__slot_values ) // 2
		i: usize = 0
		while i < count:
			with compiler.wrap_arithmetic:
				lo_slot: usize = i * 2
				hi_slot: usize = i * 2 + 1
			if self.__slot_set.__getitem__( lo_slot ).unwrap( 're: Match.regs slot_set' ):
				lo: usize = self.__slot_values.__getitem__( lo_slot ).unwrap( 're: Match.regs slot' )
				hi: usize = self.__slot_values.__getitem__( hi_slot ).unwrap( 're: Match.regs slot' )
				with compiler.panic_arithmetic( 're: Match.regs: offset does not fit in i32' ):
					out.append(( i32( lo ), i32( hi )))
			else:
				out.append(( i32( -1 ), i32( -1 )))
			with compiler.wrap_arithmetic:
				i += 1
		return out

	@property
	def lastindex( self ) -> usize|None:
		''' the highest-numbered capturing group that actually participated
		in the match, or None if the pattern had no capturing groups, or
		none of them did (matches Python's own Match.lastindex). '''
		if not self.__last_group_set:
			return None
		return self.__last_group


class Pattern:
	__ops: list[Op]
	__classes: list[CharClass]
	__n_slots: usize
	__flags: u32
	__group_names: dict[str, usize]
	__byte_mode: bool

	def __init__(
		self, ops: list[Op], classes: list[CharClass], n_slots: usize, flags: u32, group_names: dict[str, usize],
		byte_mode: bool,
	) -> None:
		self.__ops = ops
		self.__classes = classes
		self.__n_slots = n_slots
		self.__flags = flags
		self.__group_names = group_names
		self.__byte_mode = byte_mode

	@staticmethod
	def compile( pattern: str, flags: u32 = 0 ) -> Result[Pattern, PatternError]:
		parser = Parser( pattern.get_cstr(), pattern.byte_len(), False )
		return Pattern._compile_from_parser( parser, flags, False )

	@staticmethod
	def compile( pattern: bytes, flags: u32 = 0 ) -> Result[Pattern, PatternError]:
		parser = Parser( pattern.get_const_ptr(), len( pattern ), True )
		return Pattern._compile_from_parser( parser, flags, True )

	@staticmethod
	def _compile_from_parser( parser: Parser, flags: u32, byte_mode: bool ) -> Result[Pattern, PatternError]:
		body: list[Op] = parser.parse_alt().or_return()
		if not parser._at_end():
			return Result.Err( PatternError( 're: unbalanced parenthesis' ))
		prog: list[Op] = list[Op]()
		_append_fragment( prog, _single_op_fragment( _op_save( 0 )))
		_append_fragment( prog, body )
		tail: list[Op] = list[Op]()
		tail.append( _op_save( 1 ))
		tail.append( _op_match())
		_append_fragment( prog, tail )
		return Result.Ok( Pattern( prog, parser.classes, parser.next_slot, flags, parser.group_names, byte_mode ))

	def _search_from_raw( self, data: ConstPtr[u8], data_len: usize, start_pos: usize, max_steps: usize ) -> Result[Frame, MatchError]:
		''' the shared core behind search()'s str/bytes overloads - scans
		forward from start_pos (by one codepoint/byte at a time, per
		self.__byte_mode) through data_len, returning the first position's
		successful Frame, or NoMatch/StepLimitExceeded if none. Callers
		wrap the Frame into a Match with whichever `source` (str or None)
		fits their own input type - see Match's own byte-mode note. '''
		matcher = Matcher( self.__ops, self.__classes, data, data_len, self.__byte_mode, max_steps, self.__flags )
		pos: usize = start_pos
		while True:
			outcome: Result[Frame, MatchError] = matcher.run_at( pos, self.__n_slots )
			match outcome:
				case Result.Ok( frame ):
					return Result.Ok( frame )
				case Result.Err( e ):
					if e == MatchError.StepLimitExceeded:
						return Result.Err( e )
			if pos >= data_len:
				return Result.Err( MatchError.NoMatch )
			with compiler.wrap_arithmetic:
				pos += matcher._codepoint_width_at( pos )

	def _match_at_zero_raw( self, data: ConstPtr[u8], data_len: usize, max_steps: usize ) -> Result[Frame, MatchError]:
		''' the shared core behind match()'s str/bytes overloads - a SINGLE
		anchored attempt at position 0 only, no scan-forward (unlike
		_search_from_raw/search()) - Python's own re.match() semantics. '''
		matcher = Matcher( self.__ops, self.__classes, data, data_len, self.__byte_mode, max_steps, self.__flags )
		return matcher.run_at( 0, self.__n_slots )

	# max_steps stays the literal 65536 here (not `= DEFAULT_MAX_STEPS`) on
	# every overloaded method in this class, plus module-level finditer()
	# below - task_85803192's landed fix does not reliably resolve "name
	# 'DEFAULT_MAX_STEPS' is not defined" on these real overload groups
	# (search/match/fullmatch/finditer), including 2-overload cases that an
	# isolated repro of the same shape compiles fine with. Root cause of
	# the discrepancy not yet identified; see task_85803192's own report
	# for the follow-up note.
	def search( self, s: str, pos: usize = 0, max_steps: usize = 65536 ) -> Result[Match, MatchError]:
		return self._search_from( s, pos, max_steps )

	def _search_from( self, s: str, start_pos: usize, max_steps: usize ) -> Result[Match, MatchError]:
		''' like search(), but start_pos is always explicit (never
		defaulted) - the shared core finditer() calls repeatedly to find
		each successive match without re-scanning from the beginning, while
		still searching the SAME full string (not a slice of it), so
		anchors like ^ / MULTILINE-BOL / lookbehind stay correct relative
		to absolute string position. '''
		outcome: Result[Frame, MatchError] = self._search_from_raw( s.get_cstr(), s.byte_len(), start_pos, max_steps )
		match outcome:
			case Result.Ok( frame ):
				return Result.Ok( Match( s, frame.slot_values, frame.slot_set, self.__group_names, frame.last_group, frame.last_group_set ))
			case Result.Err( e ):
				return Result.Err( e )

	def search( self, s: bytes, pos: usize = 0, max_steps: usize = 65536 ) -> Result[Match, MatchError]:
		return self._search_from_bytes( s, pos, max_steps )

	def _search_from_bytes( self, s: bytes, start_pos: usize, max_steps: usize ) -> Result[Match, MatchError]:
		outcome: Result[Frame, MatchError] = self._search_from_raw( s.get_const_ptr(), len( s ), start_pos, max_steps )
		match outcome:
			case Result.Ok( frame ):
				return Result.Ok( Match( None, frame.slot_values, frame.slot_set, self.__group_names, frame.last_group, frame.last_group_set ))
			case Result.Err( e ):
				return Result.Err( e )

	def search( self, s: memoryview, pos: usize = 0, max_steps: usize = 65536 ) -> Result[Match, MatchError]:
		return self._search_from_memoryview( s, pos, max_steps )

	def _search_from_memoryview( self, s: memoryview, start_pos: usize, max_steps: usize ) -> Result[Match, MatchError]:
		outcome: Result[Frame, MatchError] = self._search_from_raw( s.get_const_ptr(), len( s ), start_pos, max_steps )
		match outcome:
			case Result.Ok( frame ):
				return Result.Ok( Match( None, frame.slot_values, frame.slot_set, self.__group_names, frame.last_group, frame.last_group_set ))
			case Result.Err( e ):
				return Result.Err( e )

	def match( self, s: str, max_steps: usize = 65536 ) -> Result[Match, MatchError]:
		outcome: Result[Frame, MatchError] = self._match_at_zero_raw( s.get_cstr(), s.byte_len(), max_steps )
		match outcome:
			case Result.Ok( frame ):
				return Result.Ok( Match( s, frame.slot_values, frame.slot_set, self.__group_names, frame.last_group, frame.last_group_set ))
			case Result.Err( e ):
				return Result.Err( e )

	def match( self, s: bytes, max_steps: usize = 65536 ) -> Result[Match, MatchError]:
		outcome: Result[Frame, MatchError] = self._match_at_zero_raw( s.get_const_ptr(), len( s ), max_steps )
		match outcome:
			case Result.Ok( frame ):
				return Result.Ok( Match( None, frame.slot_values, frame.slot_set, self.__group_names, frame.last_group, frame.last_group_set ))
			case Result.Err( e ):
				return Result.Err( e )

	def fullmatch( self, s: str, max_steps: usize = 65536 ) -> Result[Match, MatchError]:
		result: Result[Match, MatchError] = self.match( s, max_steps )
		match result:
			case Result.Ok( m ):
				end_pos: usize|None = m.end()
				# group 0 is always set on a successful match - unreachable.
				if end_pos is None:
					sys.panic( 're: fullmatch: whole match end unexpectedly unset' )
				if end_pos == s.byte_len():
					return Result.Ok( m )
				return Result.Err( MatchError.NoMatch )
			case Result.Err( e ):
				return Result.Err( e )

	def fullmatch( self, s: bytes, max_steps: usize = 65536 ) -> Result[Match, MatchError]:
		result: Result[Match, MatchError] = self.match( s, max_steps )
		match result:
			case Result.Ok( m ):
				end_pos: usize|None = m.end()
				if end_pos is None:
					sys.panic( 're: fullmatch: whole match end unexpectedly unset' )
				if end_pos == usize( len( s )):
					return Result.Ok( m )
				return Result.Err( MatchError.NoMatch )
			case Result.Err( e ):
				return Result.Err( e )

	# No Pattern.finditer() METHOD - "a generator method is not supported
	# yet" (confirmed directly). See the module-level finditer() function
	# below for the free-function form and its own further limitation
	# (confirmed unusable from any module other than this one - a general
	# compiler bug, not specific to this API).

	def findall( self, s: str, max_steps: usize = 65536 ) -> list[str]:
		''' the whole (group 0) text of every non-overlapping match, in
		order. Python's own findall() returns per-group tuples when the
		pattern has groups - simplified here to always be the whole match;
		use finditer() + Match.group(n) for per-group access (same-module
		callers only - see finditer()'s own docstring). Uses
		_find_next_match/_advance_pos_after_match directly in a plain
		while loop rather than consuming finditer()'s own generator -
		calling the same generator function from several call sites
		within this module produced confusing, seemingly unrelated
		compile errors at OTHER call sites (confirmed directly; not
		investigated further, just avoided), so only the public
		module-level finditer() is a real generator; every internal user
		re-does the same scan-forward directly instead. '''
		slen: usize = s.byte_len()
		out: list[str] = list[str]()
		pos: usize = 0
		has_next: bool = _has_match_at_or_after( self, s, pos, slen, max_steps )
		while has_next:
			mm: Match = _require_next_match( self, s, pos, slen, max_steps )
			g: str|None = mm.group()
			if g is None:
				sys.panic( 're: findall: whole match text unexpectedly unset' )
			out.append( g )
			pos = _advance_pos_after_match( mm, s )
			has_next = _has_match_at_or_after( self, s, pos, slen, max_steps )
		return out

	def sub( self, repl: str, s: str, count: usize = 0, max_steps: usize = 65536 ) -> str:
		pair: tuple[str,usize] = self._sub_impl( repl, s, count, max_steps )
		return pair[0]

	def subn( self, repl: str, s: str, count: usize = 0, max_steps: usize = 65536 ) -> tuple[str,usize]:
		return self._sub_impl( repl, s, count, max_steps )

	def _sub_impl( self, repl: str, s: str, count: usize, max_steps: usize ) -> tuple[str,usize]:
		''' shared implementation for sub()/subn() - repl is inserted
		literally (no \\1-style backreference expansion in v1). count == 0
		means unlimited, matching Python's own re.sub/subn convention.
		See findall()'s own comment for why this is a plain while loop
		over _find_next_match rather than consuming finditer(). '''
		slen: usize = s.byte_len()
		out: str = ''
		last_end: usize = 0
		n: usize = 0
		pos: usize = 0
		has_next: bool = _has_match_at_or_after( self, s, pos, slen, max_steps )
		while has_next and ( count == 0 or n < count ):
			mm: Match = _require_next_match( self, s, pos, slen, max_steps )
			start: usize|None = mm.start()
			if start is None:
				sys.panic( 're: sub: whole match start unexpectedly unset' )
			end: usize|None = mm.end()
			if end is None:
				sys.panic( 're: sub: whole match end unexpectedly unset' )
			out = out + _substr( s.get_cstr(), last_end, start )
			out = out + repl
			last_end = end
			with compiler.wrap_arithmetic:
				n += 1
			pos = _advance_pos_after_match( mm, s )
			has_next = _has_match_at_or_after( self, s, pos, slen, max_steps )
		out = out + _substr( s.get_cstr(), last_end, s.byte_len())
		return ( out, n )

	def split( self, s: str, maxsplit: usize = 0, max_steps: usize = 65536 ) -> list[str]:
		''' maxsplit == 0 means unlimited, matching Python's own re.split
		convention. Python also interleaves captured groups into the
		result when the pattern has any - simplified here to just the
		substrings between whole-pattern matches. See findall()'s own
		comment for why this is a plain while loop over _find_next_match
		rather than consuming finditer(). '''
		slen: usize = s.byte_len()
		out: list[str] = list[str]()
		last_end: usize = 0
		n: usize = 0
		pos: usize = 0
		has_next: bool = _has_match_at_or_after( self, s, pos, slen, max_steps )
		while has_next and ( maxsplit == 0 or n < maxsplit ):
			mm: Match = _require_next_match( self, s, pos, slen, max_steps )
			start: usize|None = mm.start()
			if start is None:
				sys.panic( 're: split: whole match start unexpectedly unset' )
			end: usize|None = mm.end()
			if end is None:
				sys.panic( 're: split: whole match end unexpectedly unset' )
			out.append( _substr( s.get_cstr(), last_end, start ))
			last_end = end
			with compiler.wrap_arithmetic:
				n += 1
			pos = _advance_pos_after_match( mm, s )
			has_next = _has_match_at_or_after( self, s, pos, slen, max_steps )
		out.append( _substr( s.get_cstr(), last_end, s.byte_len()))
		return out


def _advance_pos_after_match( m: Match, s: str ) -> usize:
	''' the next scan position after m: m.end(), or one codepoint past
	m.start() for a zero-width match (end == start), to avoid finditer
	looping forever on the same position. Both m.start()/m.end() are
	group 0's, which is unconditionally set on any successful match -
	the None branches below are an unreachable defensive backstop. '''
	start_pos: usize|None = m.start()
	if start_pos is None:
		sys.panic( 're: _advance_pos_after_match: whole match start unexpectedly unset' )
	end_pos: usize|None = m.end()
	if end_pos is None:
		sys.panic( 're: _advance_pos_after_match: whole match end unexpectedly unset' )
	if end_pos > start_pos:
		return end_pos
	with compiler.wrap_arithmetic:
		return start_pos + _codepoint_width_at_str( s, start_pos )


def _find_next_match( pattern: Pattern, s: str, start_pos: usize, slen: usize, max_steps: usize ) -> Match|None:
	''' ordinary (non-generator) scan-forward helper: the next match at or
	after start_pos, or None if there isn't one - Pattern._search_from
	already scans every position from start_pos through slen internally
	(only returning Err(NoMatch) once none of them work), so this makes
	exactly one call, no retry loop of its own. A StepLimitExceeded
	attempt is folded into the same "no more matches" outcome as an
	ordinary NoMatch - Iterator[T] has no error channel to report it
	through separately (a fallible Generator[T,E] would, but the extra
	plumbing isn't worth it for v1), so a pathological pattern just
	yields fewer matches than a truly unbounded engine would, bounded by
	the same per-call max_steps budget every other call already
	respects, not by silently hanging. '''
	if start_pos > slen:
		return None
	attempt: Result[Match, MatchError] = pattern._search_from( s, start_pos, max_steps )
	if attempt.is_ok():
		return attempt.unwrap( 're: _find_next_match: is_ok checked above' )
	return None


def _has_match_at_or_after( pattern: Pattern, s: str, pos: usize, slen: usize, max_steps: usize ) -> bool:
	m: Match|None = _find_next_match( pattern, s, pos, slen, max_steps )
	return m is not None

def _require_next_match( pattern: Pattern, s: str, pos: usize, slen: usize, max_steps: usize ) -> Match:
	''' only ever called right after _has_match_at_or_after confirmed one
	exists at this same pos - the panic is an unreachable backstop, not a
	real code path (matching wasn't going to become non-deterministic
	between the two calls). '''
	m: Match|None = _find_next_match( pattern, s, pos, slen, max_steps )
	if m is None:
		sys.panic( 're: _require_next_match: unreachable (_has_match_at_or_after already confirmed true)' )
	return m


# --- byte-mode siblings of the four helpers above, for finditer(bytes) ----

def _advance_pos_after_match_bytes( m: Match, slen: usize ) -> usize:
	''' like _advance_pos_after_match, but byte_mode's own "one codepoint"
	step is always exactly 1 byte (see Matcher._codepoint_width_at) - no
	need for a decode call at all, unlike the str sibling. slen is only
	used as an (unreachable in practice) upper bound sanity backstop; kept
	for signature symmetry with the str sibling. '''
	start_pos: usize|None = m.start()
	if start_pos is None:
		sys.panic( 're: _advance_pos_after_match_bytes: whole match start unexpectedly unset' )
	end_pos: usize|None = m.end()
	if end_pos is None:
		sys.panic( 're: _advance_pos_after_match_bytes: whole match end unexpectedly unset' )
	if end_pos > start_pos:
		return end_pos
	with compiler.wrap_arithmetic:
		return start_pos + 1

def _find_next_match_bytes( pattern: Pattern, s: bytes, start_pos: usize, slen: usize, max_steps: usize ) -> Match|None:
	if start_pos > slen:
		return None
	attempt: Result[Match, MatchError] = pattern._search_from_bytes( s, start_pos, max_steps )
	if attempt.is_ok():
		return attempt.unwrap( 're: _find_next_match_bytes: is_ok checked above' )
	return None

def _has_match_at_or_after_bytes( pattern: Pattern, s: bytes, pos: usize, slen: usize, max_steps: usize ) -> bool:
	m: Match|None = _find_next_match_bytes( pattern, s, pos, slen, max_steps )
	return m is not None

def _require_next_match_bytes( pattern: Pattern, s: bytes, pos: usize, slen: usize, max_steps: usize ) -> Match:
	m: Match|None = _find_next_match_bytes( pattern, s, pos, slen, max_steps )
	if m is None:
		sys.panic( 're: _require_next_match_bytes: unreachable (_has_match_at_or_after_bytes already confirmed true)' )
	return m


# --- memoryview siblings, for finditer(memoryview) - _advance_pos_after_
# match_bytes above is reused as-is (byte_mode's own "one codepoint" step
# is always 1 byte regardless of which byte-mode input type it came from,
# so it never actually touches `s`) ---------------------------------------

def _find_next_match_memoryview( pattern: Pattern, s: memoryview, start_pos: usize, slen: usize, max_steps: usize ) -> Match|None:
	if start_pos > slen:
		return None
	attempt: Result[Match, MatchError] = pattern._search_from_memoryview( s, start_pos, max_steps )
	if attempt.is_ok():
		return attempt.unwrap( 're: _find_next_match_memoryview: is_ok checked above' )
	return None

def _has_match_at_or_after_memoryview( pattern: Pattern, s: memoryview, pos: usize, slen: usize, max_steps: usize ) -> bool:
	m: Match|None = _find_next_match_memoryview( pattern, s, pos, slen, max_steps )
	return m is not None

def _require_next_match_memoryview( pattern: Pattern, s: memoryview, pos: usize, slen: usize, max_steps: usize ) -> Match:
	m: Match|None = _find_next_match_memoryview( pattern, s, pos, slen, max_steps )
	if m is None:
		sys.panic( 're: _require_next_match_memoryview: unreachable (_has_match_at_or_after_memoryview already confirmed true)' )
	return m


# max_steps stays the literal 65536 across all 3 of finditer()'s own
# overloads (str/bytes/memoryview) - see Pattern.search()'s own comment
# above (task_85803192).
def finditer( pattern: Pattern, s: str, max_steps: usize = 65536 ) -> Iterator[Result[Match, StopIteration]]:
	''' yields each successive non-overlapping match, scanning forward
	from the end of the previous one (or by one codepoint, for a
	zero-width match). Externally consumable via a real for-loop as of
	the compiler fix in 2cb18c4 ("Fix cross-module generator synthesis
	resolving names in wrong module") - confirmed directly; previously
	this only worked for same-module callers, which is why
	findall/sub/subn/split below still don't call it internally (they
	predate the fix and re-do the same scan-forward directly against
	_find_next_match instead - no need to revisit now that it works,
	but also no need to change working code just to share it).

	A free function, not a Pattern method - confirmed directly that a
	generator METHOD isn't supported yet, and separately that an
	Iterator[T] value merely returned/passed through a non-generator
	function (even a trivial `return other_generator(...)`, same
	module) has no usable __next__ for the receiver - only a DIRECT
	call to the actual generator function works as a for-loop's
	iterable expression. There is also no module-level str-pattern
	convenience overload here (unlike search/match/fullmatch/findall/
	sub/subn/split below): a second `finditer(pattern: str, ...)`
	generator that re-yields from this one via `for m in finditer(p,
	...): yield m` was tried and produced nonsensical errors (undefined
	names inside THIS function's own already-correct body) once two
	same-named overloads were both generators - not investigated
	further (this was before the cross-module fix landed; may be worth
	retrying, but not revisited here since compile-then-call works
	fine). Compile the pattern with re.compile() first, then call
	finditer(pattern, s) with the result.

	The generator body itself must also keep yield as a direct, unnested
	statement of a single top-level while loop - nesting it inside an
	if/else within the loop (the natural first-cut shape) is a separate,
	unsupported combination from a bare top-level if/else containing
	yield (confirmed directly), so the "is there a match" branching has
	to live in the while loop's own CONDITION instead of its body. That
	in turn means the loop can't carry a Match|None as its own persisted
	state across the yield boundary either (confirmed directly -
	promoting an Optional RC-typed local across a yield produces a type
	mismatch in the synthesized state field, unlike a bare, non-Optional
	RC-typed local, which Phase 9 of PLAN_GENERATORS.md's own generator
	work does support) - so the loop state here is two plain scalars
	(pos: usize, has_next: bool) instead, and the actual Match value is
	recomputed fresh each iteration via _require_next_match rather than
	carried across the yield. This costs an extra redundant _search_from
	call per position (once to check has_next, once more to fetch the
	value) - an accepted v1 inefficiency, not a correctness issue, since
	matching is deterministic. '''
	slen: usize = s.byte_len()
	pos: usize = 0
	has_next: bool = _has_match_at_or_after( pattern, s, pos, slen, max_steps )
	while has_next:
		m: Match = _require_next_match( pattern, s, pos, slen, max_steps )
		pos = _advance_pos_after_match( m, s )
		yield m
		has_next = _has_match_at_or_after( pattern, s, pos, slen, max_steps )
	return


def finditer( pattern: Pattern, s: bytes, max_steps: usize = 65536 ) -> Iterator[Result[Match, StopIteration]]:
	''' byte-mode sibling of finditer() above - same generator-shape
	constraints apply (see that docstring), same accepted "recompute
	instead of carry across yield" v1 inefficiency. Returned Match objects
	are byte-mode (their own .group()/.groupdict()/.groups() panic if
	called - see Match's own note); .span()/.start()/.end()/.regs/
	.lastindex all work identically to the str case, which is everything
	grap.mpy's own port needs from this. '''
	slen: usize = len( s )
	pos: usize = 0
	has_next: bool = _has_match_at_or_after_bytes( pattern, s, pos, slen, max_steps )
	while has_next:
		m: Match = _require_next_match_bytes( pattern, s, pos, slen, max_steps )
		pos = _advance_pos_after_match_bytes( m, slen )
		yield m
		has_next = _has_match_at_or_after_bytes( pattern, s, pos, slen, max_steps )
	return


def finditer( pattern: Pattern, s: memoryview, max_steps: usize = 65536 ) -> Iterator[Result[Match, StopIteration]]:
	''' memoryview sibling of finditer() above - same generator-shape
	constraints, same byte-mode Match caveats (see the bytes sibling's own
	docstring). This is the exact shape grap.mpy's own port needs:
	`for m in re.finditer(pattern, mv[a:b]):` over a memoryview slice. '''
	slen: usize = len( s )
	pos: usize = 0
	has_next: bool = _has_match_at_or_after_memoryview( pattern, s, pos, slen, max_steps )
	while has_next:
		m: Match = _require_next_match_memoryview( pattern, s, pos, slen, max_steps )
		pos = _advance_pos_after_match_bytes( m, slen )
		yield m
		has_next = _has_match_at_or_after_memoryview( pattern, s, pos, slen, max_steps )
	return


def compile( pattern: str, flags: u32 = 0 ) -> Result[Pattern, PatternError]:
	return Pattern.compile( pattern, flags )

def compile( pattern: bytes, flags: u32 = 0 ) -> Result[Pattern, PatternError]:
	return Pattern.compile( pattern, flags )

def search( pattern: str, s: str, flags: u32 = 0 ) -> Result[Match, MatchError]:
	p: Pattern = Pattern.compile( pattern, flags ).unwrap( 're.search: invalid pattern' )
	return p.search( s )

def search( pattern: bytes, s: bytes, flags: u32 = 0 ) -> Result[Match, MatchError]:
	p: Pattern = Pattern.compile( pattern, flags ).unwrap( 're.search: invalid pattern' )
	return p.search( s )

def match( pattern: str, s: str, flags: u32 = 0 ) -> Result[Match, MatchError]:
	p: Pattern = Pattern.compile( pattern, flags ).unwrap( 're.match: invalid pattern' )
	return p.match( s )

def match( pattern: bytes, s: bytes, flags: u32 = 0 ) -> Result[Match, MatchError]:
	p: Pattern = Pattern.compile( pattern, flags ).unwrap( 're.match: invalid pattern' )
	return p.match( s )

def fullmatch( pattern: str, s: str, flags: u32 = 0 ) -> Result[Match, MatchError]:
	p: Pattern = Pattern.compile( pattern, flags ).unwrap( 're.fullmatch: invalid pattern' )
	return p.fullmatch( s )

def fullmatch( pattern: bytes, s: bytes, flags: u32 = 0 ) -> Result[Match, MatchError]:
	p: Pattern = Pattern.compile( pattern, flags ).unwrap( 're.fullmatch: invalid pattern' )
	return p.fullmatch( s )

def findall( pattern: str, s: str, flags: u32 = 0 ) -> list[str]:
	p: Pattern = Pattern.compile( pattern, flags ).unwrap( 're.findall: invalid pattern' )
	return p.findall( s )

def sub( pattern: str, repl: str, s: str, count: usize = 0, flags: u32 = 0 ) -> str:
	p: Pattern = Pattern.compile( pattern, flags ).unwrap( 're.sub: invalid pattern' )
	return p.sub( repl, s, count )

def subn( pattern: str, repl: str, s: str, count: usize = 0, flags: u32 = 0 ) -> tuple[str,usize]:
	p: Pattern = Pattern.compile( pattern, flags ).unwrap( 're.subn: invalid pattern' )
	return p.subn( repl, s, count )

def split( pattern: str, s: str, maxsplit: usize = 0, flags: u32 = 0 ) -> list[str]:
	p: Pattern = Pattern.compile( pattern, flags ).unwrap( 're.split: invalid pattern' )
	return p.split( s, maxsplit )
