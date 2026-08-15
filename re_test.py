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
# literal backspace inside one), and the MULTILINE/DOTALL flags. See
# PLAN_RE.md for all three phases.
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
		'aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaac', 64 )
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


if __name__ == '__main__':
	unittest.main()
