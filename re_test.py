# Real-compile-and-run behavioral tests for lib/re.py.
#
# Phase 1: a backtracking regex VM (literals, `.`, character classes incl.
# \d \w \s shorthands, greedy quantifiers `* + ? {m,n}`, alternation `|`,
# `^ $` anchors, non-capturing `(...)` grouping) plus the step-budget
# safety valve (max_steps, MatchError.StepLimitExceeded vs NoMatch).
#
# Phase 2: numbered capturing groups - bare `(...)` now allocates a real
# slot pair (see Parser._parse_group), Match.group(n)/groups()/start(n)/
# end(n) beyond the whole match (group 0).
#
# Phase 3: `\b`/`\B` word-boundary assertions, a common backslash-escape
# table (\n \t \r \f \v \a \0, shared between atom and character-class
# parsing except `\b`, which is a word boundary outside a class but a
# literal backspace inside one), and the MULTILINE/DOTALL flags.
#
# Phase 4: lookahead `(?=...)`/`(?!...)` and lookbehind `(?<=...)`/`(?<!...)`
# - zero-width assertions that run a nested sub-match (sharing the outer
# match's own capture slots, so a group nested inside an assertion still
# populates the outer Match) without consuming input. Lookbehind requires
# a fixed-width body (Parser._parse_group computes it via
# _fragment_fixed_byte_width) since the VM needs to know exactly how far
# back to anchor the nested match.
#
# Phase 5: backreferences - `\1`..`\9` and the explicit `\g<N>` form
# (`\g<name>` landed later, in Phase 7 alongside named groups - see
# _RE_IGNORECASE_LAZY_NAMED below). A backreference can only refer to
# a group already OPENED earlier in the
# pattern (this parser has no separate lookahead pass), so a forward
# reference is a compile error rather than an op that could never
# succeed. An unmatched/non-participating group's backreference never
# matches, same as Python re.
#
# Phase 6: Pattern.findall/sub/subn/split, plus module-level convenience
# wrappers taking a pattern string directly. Module-level `finditer` was
# implemented here too but, at the time, confirmed unusable from any
# module other than the one that defines it (a general compiler bug in
# cross-module generator consumption) - findall/sub/subn/split were
# deliberately written to not depend on it internally for that reason,
# and it stayed untested here until that bug was fixed (see the Phase 7
# note below) - it's exercised for real now, in _RE_FINDITER.
#
# Phase 7: IGNORECASE (ASCII-only case-flip, both literal CHAR ops and
# character classes), lazy quantifiers `*? +? ?? {m,n}?` (same SPLIT-
# based compilation as their greedy counterparts, just with the two
# targets swapped so the VM prefers fewer repeats), named groups
# `(?P<name>...)` - real numbered capturing groups underneath, plus a
# name->group-number map threaded from Parser through Pattern to Match
# for Match.group(name)/groupdict() - and `\g<name>` (looked up
# against that same map; landed alongside named groups rather than in
# Phase 5 with the numeric backreferences, since it needs them to
# exist first). See PLAN_RE.md for all seven phases - this is the
# last one on the original roadmap.
#
# Also: as of the merge bringing in 2cb18c4 ("Fix cross-module
# generator synthesis resolving names in wrong module"), finditer() IS
# now externally consumable via a real for-loop - confirmed directly -
# so it's exercised for real below (_RE_FINDITER), no longer skipped.
#
# Follows time_test.py's own template: RealCompileMixin + assert_programs_run,
# print()-free, exit code 0 = every check passed, distinct nonzero i32 per
# failing check so a regression is decoded back to exactly which assertion
# broke rather than just "the program exited nonzero".

import unittest

import test_support
from test_support import RealCompileMixin

_RE_LITERAL_AND_SPAN = '''
import re

def main() -> i32:
	p: re.Pattern = re.compile( 'abc' ).unwrap( 'bad pattern' )
	m: Result[re.Match, re.MatchError] = p.search( 'xxabcxx' )
	if m.is_err():
		return 1
	mm: re.Match = m.unwrap( 'checked ok above' )
	mstart: usize|None = mm.start()
	if mstart is None:
		return 2
	if mstart != 2:
		return 2
	mend: usize|None = mm.end()
	if mend is None:
		return 3
	if mend != 5:
		return 3
	mg: str|None = mm.group()
	if mg is None:
		return 4
	if mg != 'abc':
		return 4
	if p.search( 'xxxyz' ).is_ok():
		return 5
	return 0
'''

_RE_CHAR_CLASSES_AND_QUANTIFIERS = '''
import re

def main() -> i32:
	d: re.Pattern = re.compile( r'\\d+' ).unwrap( 'bad pattern' )
	dm: Result[re.Match, re.MatchError] = d.search( 'ab123cd' )
	if dm.is_err():
		return 1
	dmg: str|None = dm.unwrap( 'checked ok above' ).group()
	if dmg is None:
		return 2
	if dmg != '123':
		return 2

	dot: re.Pattern = re.compile( 'a.c' ).unwrap( 'bad' )
	if dot.fullmatch( 'abc' ).is_err():
		return 3
	if dot.fullmatch( 'a\\nc' ).is_ok():  # DOTALL is off by default
		return 4

	star: re.Pattern = re.compile( 'ab*c' ).unwrap( 'bad' )
	if star.fullmatch( 'ac' ).is_err():
		return 5
	if star.fullmatch( 'abbbc' ).is_err():
		return 6

	plus: re.Pattern = re.compile( 'a+' ).unwrap( 'bad' )
	if plus.fullmatch( '' ).is_ok():
		return 7
	if plus.fullmatch( 'aaaa' ).is_err():
		return 8

	opt: re.Pattern = re.compile( 'colou?r' ).unwrap( 'bad' )
	if opt.fullmatch( 'color' ).is_err():
		return 9
	if opt.fullmatch( 'colour' ).is_err():
		return 10

	braces: re.Pattern = re.compile( 'a{2,3}' ).unwrap( 'bad' )
	if braces.fullmatch( 'a' ).is_ok():
		return 11
	if braces.fullmatch( 'aa' ).is_err():
		return 12
	if braces.fullmatch( 'aaaa' ).is_ok():
		return 13

	cls: re.Pattern = re.compile( '[a-c]+' ).unwrap( 'bad' )
	cm: Result[re.Match, re.MatchError] = cls.search( 'xxabccbaxx' )
	if cm.is_err():
		return 14
	cmg: str|None = cm.unwrap( 'checked ok above' ).group()
	if cmg is None:
		return 15
	if cmg != 'abccba':
		return 15

	neg: re.Pattern = re.compile( '[^0-9]+' ).unwrap( 'bad' )
	nm: Result[re.Match, re.MatchError] = neg.search( 'ab12cd' )
	if nm.is_err():
		return 16
	nmg: str|None = nm.unwrap( 'checked ok above' ).group()
	if nmg is None:
		return 17
	if nmg != 'ab':
		return 17

	word: re.Pattern = re.compile( r'\\w+' ).unwrap( 'bad' )
	wm: Result[re.Match, re.MatchError] = word.search( '  hello_world!  ' )
	if wm.is_err():
		return 18
	wmg: str|None = wm.unwrap( 'checked ok above' ).group()
	if wmg is None:
		return 19
	if wmg != 'hello_world':
		return 19

	space: re.Pattern = re.compile( r'a\\sb' ).unwrap( 'bad' )
	if space.fullmatch( 'a b' ).is_err():
		return 20
	if space.fullmatch( 'a\\tb' ).is_err():
		return 21
	return 0
'''

_RE_ALTERNATION_GROUPS_AND_ANCHORS = '''
import re

def main() -> i32:
	alt: re.Pattern = re.compile( 'cat|dog' ).unwrap( 'bad' )
	if alt.match( 'dog' ).is_err():
		return 1
	if alt.match( 'fish' ).is_ok():
		return 2

	multi: re.Pattern = re.compile( 'red|green|blue|yellow' ).unwrap( 'bad' )
	if multi.fullmatch( 'blue' ).is_err():
		return 3
	if multi.fullmatch( 'yellow' ).is_err():
		return 4
	if multi.fullmatch( 'purple' ).is_ok():
		return 5

	noncap: re.Pattern = re.compile( '(?:ab)+c' ).unwrap( 'bad' )
	if noncap.fullmatch( 'ababc' ).is_err():
		return 6
	if noncap.fullmatch( 'abac' ).is_ok():
		return 7

	anchored: re.Pattern = re.compile( '^abc$' ).unwrap( 'bad' )
	if anchored.fullmatch( 'abc' ).is_err():
		return 8
	if anchored.search( 'xabc' ).is_ok():
		return 9

	uni: re.Pattern = re.compile( 'café' ).unwrap( 'bad' )
	um: Result[re.Match, re.MatchError] = uni.search( 'xxcaféxx' )
	if um.is_err():
		return 10
	umg: str|None = um.unwrap( 'checked ok above' ).group()
	if umg is None:
		return 11
	if umg != 'café':
		return 11
	return 0
'''

_RE_STEP_BUDGET = '''
import re

def main() -> i32:
	evil: re.Pattern = re.compile( '(a+)+b' ).unwrap( 'bad' )

	tiny_budget: Result[re.Match, re.MatchError] = evil.search(
		'aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaac', max_steps = 64 )
	if tiny_budget.is_ok():
		return 1
	match tiny_budget:
		case Result.Err( e ):
			if e != re.MatchError.StepLimitExceeded:
				return 2
		case Result.Ok( m ):
			return 3

	# same pattern, generous default budget, a short genuinely-non-matching
	# string - must report plain NoMatch, not StepLimitExceeded
	plenty: Result[re.Match, re.MatchError] = evil.fullmatch( 'aaax' )
	match plenty:
		case Result.Err( e2 ):
			if e2 != re.MatchError.NoMatch:
				return 4
		case Result.Ok( m2 ):
			return 5
	return 0
'''

_RE_CAPTURING_GROUPS = '''
import re

def main() -> i32:
	p: re.Pattern = re.compile( r'(\\w+)@(\\w+)' ).unwrap( 'bad pattern' )
	m: Result[re.Match, re.MatchError] = p.search( 'user@host' )
	if m.is_err():
		return 1
	mm: re.Match = m.unwrap( 'checked ok above' )
	g0: str|None = mm.group()
	if g0 is None:
		return 2
	if g0 != 'user@host':
		return 2
	g1: str|None = mm.group( 1 )
	if g1 is None:
		return 3
	if g1 != 'user':
		return 4
	g2: str|None = mm.group( 2 )
	if g2 is None:
		return 5
	if g2 != 'host':
		return 6

	gs: list[re.GroupResult] = mm.groups()
	if len( gs ) != 2:
		return 7

	s1: usize|None = mm.start( 1 )
	if s1 is None:
		return 8
	if s1 != 0:
		return 8
	e1: usize|None = mm.end( 1 )
	if e1 is None:
		return 9
	if e1 != 4:
		return 9

	# an optional capturing group that doesn't participate reports None
	opt: re.Pattern = re.compile( r'a(b)?c' ).unwrap( 'bad' )
	om: Result[re.Match, re.MatchError] = opt.fullmatch( 'ac' )
	if om.is_err():
		return 10
	omm: re.Match = om.unwrap( 'checked ok above' )
	og1: str|None = omm.group( 1 )
	if og1 is not None:
		return 11

	om2: Result[re.Match, re.MatchError] = opt.fullmatch( 'abc' )
	if om2.is_err():
		return 12
	omm2: re.Match = om2.unwrap( 'checked ok above' )
	og2: str|None = omm2.group( 1 )
	if og2 is None:
		return 13
	if og2 != 'b':
		return 13

	# nested groups
	nested: re.Pattern = re.compile( r'((a)(b))c' ).unwrap( 'bad' )
	nm: Result[re.Match, re.MatchError] = nested.fullmatch( 'abc' )
	if nm.is_err():
		return 14
	nmm: re.Match = nm.unwrap( 'checked ok above' )
	n1: str|None = nmm.group( 1 )
	if n1 is None:
		return 15
	if n1 != 'ab':
		return 15
	n2: str|None = nmm.group( 2 )
	if n2 is None:
		return 16
	if n2 != 'a':
		return 16
	n3: str|None = nmm.group( 3 )
	if n3 is None:
		return 17
	if n3 != 'b':
		return 17

	# whole-match span/start/end unaffected by adding groups
	sp: tuple[usize,usize] = nmm.span()
	start_of_span: usize = sp[0]
	if start_of_span != 0:
		return 18
	return 0
'''

_RE_WORD_BOUNDARIES_ESCAPES_AND_FLAGS = '''
import re

def main() -> i32:
	wb: re.Pattern = re.compile( r'\\bcat\\b' ).unwrap( 'bad' )
	if wb.search( 'the cat sat' ).is_err():
		return 1
	if wb.search( 'concatenate' ).is_ok():
		return 2

	nwb: re.Pattern = re.compile( r'\\Bcat\\B' ).unwrap( 'bad' )
	if nwb.search( 'concatenate' ).is_err():
		return 3
	if nwb.search( 'the cat sat' ).is_ok():
		return 4

	ml: re.Pattern = re.compile( '^b', re.MULTILINE ).unwrap( 'bad' )
	m: Result[re.Match, re.MatchError] = ml.search( 'a\\nb\\nc' )
	if m.is_err():
		return 5
	mstart: usize|None = m.unwrap( 'ok' ).start()
	if mstart is None:
		return 6
	if mstart != 2:
		return 6

	no_ml: re.Pattern = re.compile( '^b' ).unwrap( 'bad' )
	if no_ml.search( 'a\\nb\\nc' ).is_ok():
		return 7

	dotall: re.Pattern = re.compile( 'a.c', re.DOTALL ).unwrap( 'bad' )
	if dotall.fullmatch( 'a\\nc' ).is_err():
		return 8

	no_dotall: re.Pattern = re.compile( 'a.c' ).unwrap( 'bad' )
	if no_dotall.fullmatch( 'a\\nc' ).is_ok():
		return 9

	esc: re.Pattern = re.compile( 'a\\\\tb\\\\nc' ).unwrap( 'bad' )
	if esc.fullmatch( 'a\\tb\\nc' ).is_err():
		return 10

	# multiline + dotall combined via bitwise OR
	both: re.Pattern = re.compile( '^a.c$', re.MULTILINE | re.DOTALL ).unwrap( 'bad' )
	if both.search( 'x\\na\\nc\\ny' ).is_err():
		return 11

	# common escapes inside a character class
	cls_esc: re.Pattern = re.compile( '[\\\\t\\\\n]+' ).unwrap( 'bad' )
	if cls_esc.fullmatch( '\\t\\n\\t' ).is_err():
		return 12
	return 0
'''

_RE_LOOKAROUND = '''
import re

def main() -> i32:
	la: re.Pattern = re.compile( 'foo(?=bar)' ).unwrap( 'bad' )
	if la.search( 'foobar' ).is_err():
		return 1
	if la.search( 'foobaz' ).is_ok():
		return 2

	nla: re.Pattern = re.compile( 'foo(?!bar)' ).unwrap( 'bad' )
	if nla.search( 'foobaz' ).is_err():
		return 3
	if nla.search( 'foobar' ).is_ok():
		return 4

	lb: re.Pattern = re.compile( r'(?<=\\$)\\d+' ).unwrap( 'bad' )
	m: Result[re.Match, re.MatchError] = lb.search( 'price: $42' )
	if m.is_err():
		return 5
	mg: str|None = m.unwrap( 'ok' ).group()
	if mg is None:
		return 6
	if mg != '42':
		return 6
	if lb.search( 'price: 42' ).is_ok():  # no '$' before the digits
		return 7

	nlb: re.Pattern = re.compile( r'(?<!\\$)\\d+' ).unwrap( 'bad' )
	if nlb.search( 'x42' ).is_err():
		return 8

	# lookbehind must reject a variable-width body
	bad_lb: Result[re.Pattern, re.PatternError] = re.compile( r'(?<=a+)b' )
	if bad_lb.is_ok():
		return 9

	# lookahead containing a capturing group still populates the outer match
	grp_la: re.Pattern = re.compile( r'\\w+(?=@(\\w+))' ).unwrap( 'bad' )
	gm: Result[re.Match, re.MatchError] = grp_la.search( 'user@host' )
	if gm.is_err():
		return 10
	gmm: re.Match = gm.unwrap( 'ok' )
	whole: str|None = gmm.group()
	if whole is None:
		return 11
	if whole != 'user':  # lookahead itself is zero-width, doesn't extend the match
		return 11
	g1: str|None = gmm.group( 1 )
	if g1 is None:
		return 12
	if g1 != 'host':
		return 12
	return 0
'''

_RE_BACKREFERENCES = '''
import re

def main() -> i32:
	dup: re.Pattern = re.compile( r'(\\w+) \\1' ).unwrap( 'bad' )
	if dup.search( 'hey hey' ).is_err():
		return 1
	if dup.search( 'hey bye' ).is_ok():
		return 2

	# lazy quantifiers aren't implemented yet (later phase) - no ".*?" here
	tag2: re.Pattern = re.compile( r'<(\\w+)></\\1>' ).unwrap( 'bad' )
	if tag2.fullmatch( '<div></div>' ).is_err():
		return 3
	if tag2.fullmatch( '<div></span>' ).is_ok():
		return 4

	# \\g<N> explicit numbered backreference
	gform: re.Pattern = re.compile( r'(\\w+)-\\g<1>' ).unwrap( 'bad' )
	if gform.fullmatch( 'abc-abc' ).is_err():
		return 5
	if gform.fullmatch( 'abc-xyz' ).is_ok():
		return 6

	# backreference to an optional group that didn't participate never matches
	opt: re.Pattern = re.compile( r'(a)?b\\1' ).unwrap( 'bad' )
	if opt.fullmatch( 'aba' ).is_err():
		return 7
	if opt.fullmatch( 'b' ).is_ok():  # group 1 unset -> \\1 never matches
		return 8

	# forward reference (referring to a group not yet opened) is a compile error
	fwd: Result[re.Pattern, re.PatternError] = re.compile( r'\\1(a)' )
	if fwd.is_ok():
		return 9

	# \\0 is not a valid backreference (group numbering starts at 1) -
	# should still be a valid NUL escape via the common-escape table
	nul: re.Pattern = re.compile( 'a\\\\0b' ).unwrap( 'bad' )
	if nul.fullmatch( 'a\\0b' ).is_err():
		return 10

	# 'g' with no following '<' is just a literal 'g'
	lit_g: re.Pattern = re.compile( 'a\\\\gb' ).unwrap( 'bad' )
	if lit_g.fullmatch( 'agb' ).is_err():
		return 11
	return 0
'''

_RE_FINDALL_SUB_SPLIT = '''
import re

def main() -> i32:
	p: re.Pattern = re.compile( r'\\d+' ).unwrap( 'bad' )

	fa: list[str] = p.findall( 'a1 b22 c333' )
	if len( fa ) != 3:
		return 1
	fa0: str = fa.__getitem__( 0 ).unwrap( 'ok' )
	if fa0 != '1':
		return 2
	fa1: str = fa.__getitem__( 1 ).unwrap( 'ok' )
	if fa1 != '22':
		return 3
	fa2: str = fa.__getitem__( 2 ).unwrap( 'ok' )
	if fa2 != '333':
		return 4

	subbed: str = p.sub( '#', 'a1b22c333' )
	if subbed != 'a#b#c#':
		return 5
	pair: tuple[str,usize] = p.subn( '#', 'a1b22c333' )
	if pair[0] != 'a#b#c#':
		return 6
	if pair[1] != 3:
		return 7

	limited: str = p.sub( '#', 'a1b22c333', 2 )
	if limited != 'a#b#c333':
		return 8

	sp: re.Pattern = re.compile( r'\\s*,\\s*' ).unwrap( 'bad' )
	parts: list[str] = sp.split( 'a, b,c ,  d' )
	if len( parts ) != 4:
		return 9
	p0: str = parts.__getitem__( 0 ).unwrap( 'ok' )
	if p0 != 'a':
		return 10
	p3: str = parts.__getitem__( 3 ).unwrap( 'ok' )
	if p3 != 'd':
		return 11

	# module-level convenience wrappers (compile-then-delegate)
	if len( re.findall( r'\\w+', 'foo bar' )) != 2:
		return 12
	if re.sub( r'\\d', 'X', 'a1b2' ) != 'aXbX':
		return 13
	sn: tuple[str,usize] = re.subn( r'\\d', 'X', 'a1b2' )
	if sn[1] != 2:
		return 14
	if len( re.split( r',', 'a,b,c' )) != 3:
		return 15
	return 0
'''

_RE_FINDITER = '''
import re

def main() -> i32:
	p: re.Pattern = re.compile( r'\\d+' ).unwrap( 'bad pattern' )
	count: usize = 0
	total_len: usize = 0
	for m in re.finditer( p, 'a1 b22 c333' ):
		with compiler.wrap_arithmetic:
			count += 1
		g: str|None = m.group()
		if g is None:
			return 1
		gg: str = g
		with compiler.wrap_arithmetic:
			total_len += gg.byte_len()
	if count != 3:
		return 2
	if total_len != 6:  # '1' + '22' + '333' = 1+2+3 chars
		return 3
	return 0
'''

_RE_IGNORECASE_LAZY_NAMED = '''
import re

def main() -> i32:
	ci: re.Pattern = re.compile( 'HELLO', re.IGNORECASE ).unwrap( 'bad' )
	if ci.fullmatch( 'hello' ).is_err():
		return 1
	if ci.fullmatch( 'HeLLo' ).is_err():
		return 2
	no_ci: re.Pattern = re.compile( 'HELLO' ).unwrap( 'bad' )
	if no_ci.fullmatch( 'hello' ).is_ok():
		return 3

	ci_cls: re.Pattern = re.compile( '[a-f]+', re.IGNORECASE ).unwrap( 'bad' )
	if ci_cls.fullmatch( 'ABCdef' ).is_err():
		return 4
	no_ci_cls: re.Pattern = re.compile( '[a-f]+' ).unwrap( 'bad' )
	if no_ci_cls.fullmatch( 'ABCdef' ).is_ok():
		return 5

	lazy_star: re.Pattern = re.compile( '<.*?>' ).unwrap( 'bad' )
	m: Result[re.Match, re.MatchError] = lazy_star.search( '<a><b>' )
	if m.is_err():
		return 6
	g: str|None = m.unwrap( 'ok' ).group()
	if g is None:
		return 7
	if g != '<a>':  # lazy: shortest match, not '<a><b>'
		return 7

	greedy_star: re.Pattern = re.compile( '<.*>' ).unwrap( 'bad' )
	gm: Result[re.Match, re.MatchError] = greedy_star.search( '<a><b>' )
	if gm.is_err():
		return 8
	gg: str|None = gm.unwrap( 'ok' ).group()
	if gg is None:
		return 9
	if gg != '<a><b>':  # greedy: longest match
		return 9

	lazy_plus: re.Pattern = re.compile( 'a+?' ).unwrap( 'bad' )
	lp: Result[re.Match, re.MatchError] = lazy_plus.search( 'aaaa' )
	if lp.is_err():
		return 10
	lpg: str|None = lp.unwrap( 'ok' ).group()
	if lpg is None:
		return 11
	if lpg != 'a':  # lazy +: minimal one repeat
		return 11

	lazy_opt: re.Pattern = re.compile( 'colou??r' ).unwrap( 'bad' )
	if lazy_opt.fullmatch( 'color' ).is_err():
		return 12
	if lazy_opt.fullmatch( 'colour' ).is_err():
		return 13

	lazy_range: re.Pattern = re.compile( 'a{2,4}?' ).unwrap( 'bad' )
	lr: Result[re.Match, re.MatchError] = lazy_range.search( 'aaaa' )
	if lr.is_err():
		return 14
	lrg: str|None = lr.unwrap( 'ok' ).group()
	if lrg is None:
		return 15
	if lrg != 'aa':  # lazy {2,4}: minimal 2 repeats
		return 15

	named: re.Pattern = re.compile( r'(?P<year>\\d{4})-(?P<month>\\d{2})' ).unwrap( 'bad' )
	nm: Result[re.Match, re.MatchError] = named.search( '2026-08' )
	if nm.is_err():
		return 16
	nmm: re.Match = nm.unwrap( 'ok' )
	year: str|None = nmm.group( 'year' )
	if year is None:
		return 17
	if year != '2026':
		return 17
	month: str|None = nmm.group( 'month' )
	if month is None:
		return 18
	if month != '08':
		return 18

	gd: dict[str, re.GroupResult] = nmm.groupdict()
	if len( gd ) != 2:
		return 19
	yr: re.GroupResult = gd.__getitem__( 'year' ).unwrap( 'ok' )
	if yr.text != '2026':
		return 20
	if not yr.matched:
		return 21

	# duplicate named group is a compile error
	dup_name: Result[re.Pattern, re.PatternError] = re.compile( r'(?P<x>a)(?P<x>b)' )
	if dup_name.is_ok():
		return 22

	# \\g<name> backreference
	tag: re.Pattern = re.compile( r'<(?P<tag>\\w+)></\\g<tag>>' ).unwrap( 'bad' )
	if tag.fullmatch( '<div></div>' ).is_err():
		return 23
	if tag.fullmatch( '<div></span>' ).is_ok():
		return 24

	# \\g<name> referencing a name not yet defined (forward reference) is
	# a compile error, same restriction as numeric backreferences
	fwd: Result[re.Pattern, re.PatternError] = re.compile( r'\\g<x>(?P<x>a)' )
	if fwd.is_ok():
		return 25

	# \\g<name> referencing an undefined name entirely is a compile error
	undef: Result[re.Pattern, re.PatternError] = re.compile( r'(?P<y>a)\\g<nope>' )
	if undef.is_ok():
		return 26
	return 0
'''

# Phase 8: re.compile(bytes)/Pattern.search|match|fullmatch(bytes) - a
# second, byte_mode Matcher/Parser code path (no UTF-8 decoding: one raw
# byte per step, so an arbitrary binary subject is never assumed to be
# valid UTF-8) sharing every op/opcode-level behavior with the str path
# (see lib/re.py's own Matcher/Parser comments) - plus Match.regs/
# .lastindex (mode-agnostic: both work identically for str and bytes,
# since neither touches the matched TEXT, only slot offsets) and
# re.escape()/re.M/re.I (short flag aliases). grap.mpy (the motivating
# real-world port) needs exactly this subset: bytes patterns/subjects,
# .search(subject, pos), finditer(pattern, bytes_subject), .regs,
# .lastindex - NOT Match.group()/groupdict()/groups() on a byte-mode
# match (panics - see Match.group()'s own note: a byte-mode subject
# isn't guaranteed valid UTF-8, so there's no safe str to hand back),
# and NOT findall/sub/subn/split for bytes (str-only still, deliberately
# out of scope - not needed by grap.mpy, and each would need its own
# byte-mode sibling of _find_next_match/_advance_pos_after_match/etc,
# same shape as finditer's own _bytes siblings, if ever added later).
_RE_BYTES_BASIC = '''
import re

def main() -> i32:
	p: re.Pattern = re.compile( b'a(b+)c' ).unwrap( 'bad pattern' )
	m: re.Match = p.search( b'xxabbbcxx' ).unwrap( 'search' )
	sp: tuple[usize,usize] = m.span()
	if sp[0] != 2 or sp[1] != 7:
		return 1

	li: usize|None = m.lastindex
	if li is None:
		return 2
	if li != 1:
		return 20

	regs: list[tuple[i32,i32]] = m.regs
	if len( regs ) != 2:
		return 3
	r0: tuple[i32,i32] = regs.__getitem__( 0 ).unwrap( 'idx' )
	if r0[0] != 2 or r0[1] != 7:
		return 4
	r1: tuple[i32,i32] = regs.__getitem__( 1 ).unwrap( 'idx' )
	if r1[0] != 3 or r1[1] != 6:
		return 5

	# search(subject, pos) - explicit start position, matching real
	# Python's Pattern.search(string, pos, endpos)
	m2: re.Match = p.search( b'abbbcXabbbc', 1 ).unwrap( 'search pos' )
	sp2: tuple[usize,usize] = m2.span()
	if sp2[0] != 6 or sp2[1] != 11:
		return 6
	# nothing left to find starting past the last occurrence
	if p.search( b'abbbcXabbbc', 7 ).is_ok():
		return 7

	# match()/fullmatch()
	if p.match( b'abbbc' ).is_err():
		return 8
	if p.fullmatch( b'abbbc' ).is_err():
		return 9
	if p.fullmatch( b'abbbcX' ).is_ok():
		return 10
	# match() is anchored at 0 only, no scan-forward (unlike search())
	if p.match( b'xabbbc' ).is_ok():
		return 11

	# an unset (didn't participate) optional group reports (-1,-1) in
	# .regs and doesn't move .lastindex
	p2: re.Pattern = re.compile( b'a(b)?c' ).unwrap( 'bad pattern 2' )
	m3: re.Match = p2.search( b'ac' ).unwrap( 'search3' )
	# via a local, not `m3.lastindex is not None` directly - a confirmed
	# compiler bug (task_c98beffa): "<property returning T|None> is None"
	# used directly (not through an intermediate local) fails to compile
	li3: usize|None = m3.lastindex
	if li3 is not None:
		return 12
	regs3: list[tuple[i32,i32]] = m3.regs
	r3: tuple[i32,i32] = regs3.__getitem__( 1 ).unwrap( 'idx' )
	if r3[0] != -1 or r3[1] != -1:
		return 13

	return 0
'''

_RE_BYTES_FINDITER = '''
import re

def main() -> i32:
	p: re.Pattern = re.compile( b'a+bc' ).unwrap( 'bad pattern' )
	count: usize = 0
	total_len: usize = 0
	with compiler.wrap_arithmetic:
		for m in re.finditer( p, b'abcXaabcXaaabc' ):
			count += 1
			sp: tuple[usize,usize] = m.span()
			total_len += sp[1] - sp[0]
	if count != 3:
		return 1
	if total_len != 3 + 4 + 5:
		return 2
	return 0
'''

_RE_M_I_ALIASES_AND_ESCAPE = '''
import re

def main() -> i32:
	# re.M/re.I are literally the same values as MULTILINE/IGNORECASE
	if re.M != re.MULTILINE:
		return 1
	if re.I != re.IGNORECASE:
		return 2
	if re.S != re.DOTALL:
		return 3
	if re.A != re.ASCII:
		return 4

	p: re.Pattern = re.compile( '^b', re.M | re.I ).unwrap( 'bad pattern' )
	if p.search( 'a\\nB' ).is_err():
		return 5
	if p.search( 'aB' ).is_ok():  # not at a line start
		return 6

	if re.escape( 'a.b*c' ) != 'a\\\\.b\\\\*c':
		return 7
	if re.escape( 'plain_text123' ) != 'plain_text123':
		return 8
	if re.escape( '' ) != '':
		return 9
	# an escaped pattern always matches its own original literal text
	esc: re.Pattern = re.compile( re.escape( '(a.b)' )).unwrap( 'bad escaped pattern' )
	if esc.fullmatch( '(a.b)' ).is_err():
		return 10
	if esc.search( 'Xa.bY' ).is_ok():  # '(' ')' are literal now, not grouping
		return 11
	return 0
'''


@unittest.skipUnless( test_support.HAS_CC, 'no C compiler (clang/gcc/msvc) found - skipping real-compile re tests' )
class RePhase1BehaviorTests( RealCompileMixin, unittest.TestCase ):
	def test_phase1_engine( self ) -> None:
		self.assert_programs_run([
			( 'literal_and_span', _RE_LITERAL_AND_SPAN ),
			( 'char_classes_and_quantifiers', _RE_CHAR_CLASSES_AND_QUANTIFIERS ),
			( 'alternation_groups_and_anchors', _RE_ALTERNATION_GROUPS_AND_ANCHORS ),
			( 'step_budget_safety_valve', _RE_STEP_BUDGET ),
		])


@unittest.skipUnless( test_support.HAS_CC, 'no C compiler (clang/gcc/msvc) found - skipping real-compile re tests' )
class RePhase2BehaviorTests( RealCompileMixin, unittest.TestCase ):
	def test_phase2_capturing_groups( self ) -> None:
		self.assert_programs_run([
			( 'capturing_groups', _RE_CAPTURING_GROUPS ),
		])


@unittest.skipUnless( test_support.HAS_CC, 'no C compiler (clang/gcc/msvc) found - skipping real-compile re tests' )
class RePhase3BehaviorTests( RealCompileMixin, unittest.TestCase ):
	def test_phase3_boundaries_escapes_and_flags( self ) -> None:
		self.assert_programs_run([
			( 'word_boundaries_escapes_and_flags', _RE_WORD_BOUNDARIES_ESCAPES_AND_FLAGS ),
		])


@unittest.skipUnless( test_support.HAS_CC, 'no C compiler (clang/gcc/msvc) found - skipping real-compile re tests' )
class RePhase4BehaviorTests( RealCompileMixin, unittest.TestCase ):
	def test_phase4_lookaround( self ) -> None:
		self.assert_programs_run([
			( 'lookaround', _RE_LOOKAROUND ),
		])


@unittest.skipUnless( test_support.HAS_CC, 'no C compiler (clang/gcc/msvc) found - skipping real-compile re tests' )
class RePhase5BehaviorTests( RealCompileMixin, unittest.TestCase ):
	def test_phase5_backreferences( self ) -> None:
		self.assert_programs_run([
			( 'backreferences', _RE_BACKREFERENCES ),
		])


@unittest.skipUnless( test_support.HAS_CC, 'no C compiler (clang/gcc/msvc) found - skipping real-compile re tests' )
class RePhase6BehaviorTests( RealCompileMixin, unittest.TestCase ):
	def test_phase6_findall_sub_split( self ) -> None:
		self.assert_programs_run([
			( 'findall_sub_split', _RE_FINDALL_SUB_SPLIT ),
		])

	def test_phase6_finditer( self ) -> None:
		''' separate assert_programs_run call, not merged into the case
		above: finditer's own cross-module generator consumption only
		works as of the compiler fix in 2cb18c4 (see finditer()'s own
		docstring in lib/re.py) - isolating it means a future regression
		in just this path fails only this test, not test_phase6_findall_
		sub_split too. '''
		self.assert_programs_run([
			( 'finditer', _RE_FINDITER ),
		])


@unittest.skipUnless( test_support.HAS_CC, 'no C compiler (clang/gcc/msvc) found - skipping real-compile re tests' )
class RePhase7BehaviorTests( RealCompileMixin, unittest.TestCase ):
	def test_phase7_ignorecase_lazy_named( self ) -> None:
		self.assert_programs_run([
			( 'ignorecase_lazy_named', _RE_IGNORECASE_LAZY_NAMED ),
		])


@unittest.skipUnless( test_support.HAS_CC, 'no C compiler (clang/gcc/msvc) found - skipping real-compile re tests' )
class RePhase8BehaviorTests( RealCompileMixin, unittest.TestCase ):
	''' bytes/memoryview support (re.compile(bytes), Pattern.search/match/
	fullmatch(bytes)), Match.regs/.lastindex, re.M/re.I/re.escape() - see
	_RE_BYTES_BASIC's own comment for exact scope. '''

	def test_phase8_bytes_and_flags_and_escape( self ) -> None:
		self.assert_programs_run([
			( 'bytes_basic', _RE_BYTES_BASIC ),
			( 'm_i_aliases_and_escape', _RE_M_I_ALIASES_AND_ESCAPE ),
		])

	def test_phase8_bytes_finditer( self ) -> None:
		''' separate call, not merged into the case above - same isolation
		reasoning as test_phase6_finditer's own comment. '''
		self.assert_programs_run([
			( 'bytes_finditer', _RE_BYTES_FINDITER ),
		])

	def test_phase8_group_panics_on_byte_mode_match( self ) -> None:
		''' Match.group()/groupdict()/groups() all route through group(),
		so this one panic-exit check covers all three - a byte-mode
		subject isn't guaranteed valid UTF-8, so there's no safe str for
		any of them to hand back (see Match.group()'s own note). A real
		process-exit check, not merged via assert_programs_run: a
		panicking sub-program's abrupt exit(1) would break the merged
		dispatch's own "each case's main() returns normally" assumption. '''
		from discovery import Discovery
		from compiler import Compiler
		from pathlib import Path
		import emitter_c
		discovery = Discovery( import_builtins = True )
		compiler = Compiler( discovery )
		compiler.import_code( '''
import re

def main() -> i32:
	p: re.Pattern = re.compile( b"abc" ).unwrap( "bad pattern" )
	m: re.Match = p.search( b"abc" ).unwrap( "search" )
	g: str|None = m.group()
	return 0
''', Path( '__main__.py' ), scope = None )
		compiler.run()
		self.assertEqual( discovery.errors.errors, [] )
		self._assert_compiles_and_runs( emitter_c.emit_c( compiler ), expected_exit = 1, compiler = compiler )


if __name__ == '__main__':
	unittest.main()
