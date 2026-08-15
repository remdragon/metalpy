# Real-compile-and-run behavioral tests for lib/re.py, Phase 1: a
# backtracking regex VM (literals, `.`, character classes incl. \d \w \s
# shorthands, greedy quantifiers `* + ? {m,n}`, alternation `|`, `^ $`
# anchors, non-capturing `(...)` grouping) plus the step-budget safety
# valve (max_steps, MatchError.StepLimitExceeded vs MatchError.NoMatch).
# Match.group(0)/span()/start()/end() only - named/numbered capture groups
# beyond the whole match are a Phase 2 addition (see PLAN_RE.md).
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
	if mm.start() != 2:
		return 2
	if mm.end() != 5:
		return 3
	if mm.group() != 'abc':
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
	if dm.unwrap( 'checked ok above' ).group() != '123':
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
	if cm.unwrap( 'checked ok above' ).group() != 'abccba':
		return 15

	neg: re.Pattern = re.compile( '[^0-9]+' ).unwrap( 'bad' )
	nm: Result[re.Match, re.MatchError] = neg.search( 'ab12cd' )
	if nm.is_err():
		return 16
	if nm.unwrap( 'checked ok above' ).group() != 'ab':
		return 17

	word: re.Pattern = re.compile( r'\\w+' ).unwrap( 'bad' )
	wm: Result[re.Match, re.MatchError] = word.search( '  hello_world!  ' )
	if wm.is_err():
		return 18
	if wm.unwrap( 'checked ok above' ).group() != 'hello_world':
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
	if um.unwrap( 'checked ok above' ).group() != 'café':
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


@unittest.skipUnless( test_support.HAS_CC, 'no C compiler (clang/gcc/msvc) found - skipping real-compile re tests' )
class RePhase1BehaviorTests( RealCompileMixin, unittest.TestCase ):
	def test_phase1_engine( self ) -> None:
		self.assert_programs_run([
			( 'literal_and_span', _RE_LITERAL_AND_SPAN ),
			( 'char_classes_and_quantifiers', _RE_CHAR_CLASSES_AND_QUANTIFIERS ),
			( 'alternation_groups_and_anchors', _RE_ALTERNATION_GROUPS_AND_ANCHORS ),
			( 'step_budget_safety_valve', _RE_STEP_BUDGET ),
		])


if __name__ == '__main__':
	unittest.main()
