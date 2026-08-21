# stdlib imports
import ast
import unittest

# local imports
import compile_time_transformer as ctt
import linker_c
import test_support
from compiler import Compiler
from discovery import ActiveTarget, Discovery


def _fold( src: str, active_target: ActiveTarget, detect_cc = None ) -> str:
	body = ctt.transform_function_body( ast.parse( src ).body, active_target, detect_cc )
	return '\n'.join( ast.unparse( stmt ) for stmt in body )


_CC = linker_c.detect_cc()


class ConstantFoldingTests( unittest.TestCase ):
	def test_binop_folds( self ) -> None:
		self.assertEqual( _fold( 'x = 1 + 1', {} ), 'x = 2' )

	def test_binop_bitwise_folds( self ) -> None:
		self.assertEqual( _fold( 'x = 6 & 3', {} ), 'x = 2' )

	def test_binop_division_by_zero_left_unfolded( self ) -> None:
		self.assertEqual( _fold( 'x = 1 // 0', {} ), 'x = 1 // 0' )
		self.assertEqual( _fold( 'x = 1 % 0', {} ), 'x = 1 % 0' )

	def test_binop_floordiv_mod_are_c_truncating_not_python_floor( self ) -> None:
		# this compiler's runtime // and % are C-style truncating (quotient
		# toward zero, remainder sign matches the dividend), NOT Python's
		# own floor-based semantics - folding must match whatever the
		# runtime opcode would compute for the same operands (see
		# lib/math.py's floordiv_i64/floormod_i64 comments)
		self.assertEqual( _fold( 'x = -7 % 2', {} ), 'x = -1' )
		self.assertEqual( _fold( 'x = 7 % -2', {} ), 'x = 1' )
		self.assertEqual( _fold( 'x = -7 // 2', {} ), 'x = -3' )
		self.assertEqual( _fold( 'x = 7 // -2', {} ), 'x = -3' )
		self.assertEqual( _fold( 'x = -7 // -2', {} ), 'x = 3' )
		self.assertEqual( _fold( 'x = -7 % -2', {} ), 'x = -1' )
		# positive operands: C-truncating and Python-floor coincide, matches
		# the pre-fix behavior exactly
		self.assertEqual( _fold( 'x = 7 // 2', {} ), 'x = 3' )
		self.assertEqual( _fold( 'x = 7 % 2', {} ), 'x = 1' )

	def test_binop_non_constant_operand_left_unfolded( self ) -> None:
		self.assertEqual( _fold( 'x = y + 1', {} ), 'x = y + 1' )

	def test_unary_not_folds( self ) -> None:
		self.assertEqual( _fold( 'x = not True', {} ), 'x = False' )

	def test_unary_neg_folds( self ) -> None:
		self.assertEqual( _fold( 'x = -5', {} ), 'x = -5' )

	def test_boolop_and_folds_to_first_falsy( self ) -> None:
		self.assertEqual( _fold( 'x = True and False and True', {} ), 'x = False' )

	def test_boolop_or_folds_to_first_truthy( self ) -> None:
		self.assertEqual( _fold( 'x = False or 3 or 0', {} ), 'x = 3' )

	def test_compare_eq_folds( self ) -> None:
		self.assertEqual( _fold( "x = 'windows' == 'windows'", {} ), 'x = True' )

	def test_compare_lt_folds( self ) -> None:
		self.assertEqual( _fold( 'x = 1<2', {} ), 'x = True' )

	def test_compare_chained_cmp_folds( self ) -> None:
		self.assertEqual( _fold( 'y = 1 < 2 < 3', {} ), 'y = True' ) # both conditions are always True
		self.assertEqual( _fold( 'y = 1 < 2 < x', {} ), 'y = 2 < x' ) # first condition is always True
		self.assertEqual( _fold( 'y = 1 > 2 > x', {} ), 'y = False' ) # first condition is never True
		self.assertEqual( _fold( 'y = x < 2 < 3', {} ), 'y = x < 2' ) # second condition is always True
		self.assertEqual( _fold( 'y = x > 2 > 3', {} ), 'y = False' ) # second condition is never True

	def test_compare_is_left_unfolded( self ) -> None:
		# matches lowering.py's own deliberate exclusion of is/is not - see
		# _expr_Compare's docstring
		self.assertEqual( _fold( 'x = None is None', {} ), 'x = None is None' )

	def test_nested_folds_bottom_up( self ) -> None:
		self.assertEqual( _fold( 'x = (1 + 1) == 2', {} ), 'x = True' )

	def test_str_cmp( self ) -> None:
		self.assertEqual( _fold( 'x = "abc" == "ABC"', {} ), 'x = False' )
		self.assertEqual( _fold( 'x = "abc" == "abc"', {} ), 'x = True' )
		self.assertEqual( _fold( 'x = "abc" != "ABC"', {} ), 'x = True' )
		self.assertEqual( _fold( 'x = "abc" != "abc"', {} ), 'x = False' )
		self.assertEqual( _fold( 'x = "abc" < "ABC"', {} ), 'x = False' )
		self.assertEqual( _fold( 'x = "abc" < "abc"', {} ), 'x = False' )
		self.assertEqual( _fold( 'x = "ABC" < "abc"', {} ), 'x = True' )
		self.assertEqual( _fold( 'x = "abc" <= "ABC"', {} ), 'x = False' )
		self.assertEqual( _fold( 'x = "abc" <= "abc"', {} ), 'x = True' )
		self.assertEqual( _fold( 'x = "ABC" <= "abc"', {} ), 'x = True' )
		self.assertEqual( _fold( 'x = "abc" >= "ABC"', {} ), 'x = True' )
		self.assertEqual( _fold( 'x = "abc" >= "abc"', {} ), 'x = True' )
		self.assertEqual( _fold( 'x = "ABC" >= "abc"', {} ), 'x = False' )
		self.assertEqual( _fold( 'x = "abc" > "ABC"', {} ), 'x = True' )
		self.assertEqual( _fold( 'x = "abc" > "abc"', {} ), 'x = False' )
		self.assertEqual( _fold( 'x = "ABC" > "abc"', {} ), 'x = False' )


class JoinedStrFoldingTests( unittest.TestCase ):
	def test_all_literal_folds( self ) -> None:
		self.assertEqual( _fold( 'x = f"hello world"', {} ), "x = 'hello world'" )

	def test_literal_plus_constant_expr_folds( self ) -> None:
		self.assertEqual( _fold( 'x = f"answer={1+21}"', {} ), "x = 'answer=22'" )

	def test_negative_int_folds( self ) -> None:
		self.assertEqual( _fold( 'x = f"n={-5}"', {} ), "x = 'n=-5'" )

	def test_str_constant_value_folds( self ) -> None:
		self.assertEqual( _fold( 'x = f"{\"abc\"}"', {} ), "x = 'abc'" )

	def test_repr_conversion_on_int_folds( self ) -> None:
		self.assertEqual( _fold( 'x = f"{1+1!r}"', {} ), "x = '2'" )

	def test_empty_joinedstr_folds( self ) -> None:
		self.assertEqual( _fold( 'x = f""', {} ), "x = ''" )

	def test_runtime_name_left_unfolded( self ) -> None:
		self.assertEqual( _fold( 'x = f"{y}"', {} ), "x = f'{y}'" )

	def test_mixed_foldable_and_runtime_still_folds_the_foldable_part_in_place( self ) -> None:
		# the whole JoinedStr can't collapse (y is runtime), but the nested
		# 1+1 inside the OTHER FormattedValue still folds in place - proves
		# generic_visit's own bottom-up cascade runs regardless of whether
		# the outer JoinedStr itself ends up foldable
		self.assertEqual( _fold( 'x = f"n={1+1}{y}"', {} ), "x = f'n={2}{y}'" )

	def test_bool_value_left_unfolded( self ) -> None:
		# no metalpy bool.__str__() exists to fold against - see
		# visit_JoinedStr's own comment
		self.assertEqual( _fold( 'x = f"{True}"', {} ), "x = f'{True}'" )

	def test_float_value_left_unfolded( self ) -> None:
		self.assertEqual( _fold( 'x = f"{1.5}"', {} ), "x = f'{1.5}'" )

	def test_ascii_conversion_left_unfolded( self ) -> None:
		self.assertEqual( _fold( 'x = f"{y!a}"', {} ), "x = f'{y!a}'" )

	def test_float_type_char_on_int_left_unfolded_even_when_value_is_constant( self ) -> None:
		# '.2f' is a float type char - not valid for an int value (no boxed
		# float class exists yet to fold this the way it would need to,
		# see fstring_format_spec.validate_int_spec) - the nested (1+1)
		# still folds to 2 in place, same "fold what you can" cascade
		# every other case here relies on, but the outer JoinedStr itself
		# correctly stays unfolded rather than silently producing wrong
		# text
		self.assertEqual( _fold( 'x = f"{(1+1):.2f}"', {} ), "x = f'{2:.2f}'" )

	def test_literal_int_format_spec_folds( self ) -> None:
		self.assertEqual( _fold( 'x = f"{5:d}"', {} ), "x = '5'" )
		self.assertEqual( _fold( 'x = f"{-5:05d}"', {} ), "x = '-0005'" )
		self.assertEqual( _fold( 'x = f"{5:+d}"', {} ), "x = '+5'" )
		self.assertEqual( _fold( 'x = f"{255:#x}"', {} ), "x = '0xff'" )
		self.assertEqual( _fold( 'x = f"{255:#X}"', {} ), "x = '0XFF'" )
		self.assertEqual( _fold( 'x = f"{8:b}"', {} ), "x = '1000'" )
		self.assertEqual( _fold( 'x = f"{1234567:,}"', {} ), "x = '1,234,567'" )
		self.assertEqual( _fold( 'x = f"{-255:#010x}"', {} ), "x = '-0x00000ff'" )

	def test_literal_str_format_spec_folds( self ) -> None:
		self.assertEqual( _fold( 'x = f"{\'hi\':>10}"', {} ), "x = '        hi'" )
		self.assertEqual( _fold( 'x = f"{\'hi\':<5}"', {} ), "x = 'hi   '" )
		self.assertEqual( _fold( 'x = f"{\'hi\':^6}"', {} ), "x = '  hi  '" )
		self.assertEqual( _fold( 'x = f"{\'hello\':.3}"', {} ), "x = 'hel'" )

	def test_explicit_conversion_plus_format_spec_folds( self ) -> None:
		# !r/!s + a spec applies the spec to the CONVERTED text, not the
		# original value - matches lowering.py's own _lower_fstring_part
		# ordering (and Python's own real f-string semantics)
		self.assertEqual( _fold( 'x = f"{5!r:>5}"', {} ), "x = '    5'" )
		self.assertEqual( _fold( 'x = f"{5!s:>5}"', {} ), "x = '    5'" )

	def test_unsupported_type_char_left_unfolded( self ) -> None:
		self.assertEqual( _fold( 'x = f"{5:c}"', {} ), "x = f'{5:c}'" )

	def test_dynamic_format_spec_left_unfolded( self ) -> None:
		self.assertEqual( _fold( 'x = f"{5:{y}}"', {} ), "x = f'{5:{y}}'" )


class CompilerTargetSubstitutionTests( unittest.TestCase ):
	def test_os_substituted( self ) -> None:
		self.assertEqual( _fold( 'x = compiler.target.os', { 'os': 'windows' } ), "x = 'windows'" )

	def test_debug_substituted( self ) -> None:
		self.assertEqual( _fold( 'x = compiler.target.debug', { 'debug': True } ), 'x = True' )

	def test_unmodeled_key_left_unfolded( self ) -> None:
		self.assertEqual( _fold( 'x = compiler.target.vendor', { 'os': 'windows' } ), 'x = compiler.target.vendor' )

	def test_bare_compiler_target_left_unfolded( self ) -> None:
		# only the 3-level `compiler.target.<key>` chain folds, not the
		# TargetQuery object itself
		self.assertEqual( _fold( 'x = compiler.target', { 'os': 'windows' } ), 'x = compiler.target' )

	def test_substitution_feeds_downstream_folding( self ) -> None:
		self.assertEqual(
			_fold( "x = compiler.target.os == 'windows'", { 'os': 'windows' } ),
			'x = True',
		)


class CompilerHasLibraryFoldingTests( unittest.TestCase ):
	''' compiler.has_library(lib, symbol) - lazy expression-level folding,
	only probed (a real compile+link, via linker_c.has_symbol) when a
	detect_cc callback is actually supplied and the call is actually
	reached, unlike compiler.target.<key>'s always-known dict lookup. '''

	def test_no_detect_cc_left_unfolded( self ) -> None:
		# the default (detect_cc=None) - a caller that never needs this
		# folding shouldn't pay for it, or fail merely for omitting it
		self.assertEqual(
			_fold( "x = compiler.has_library( 'kernel32', 'GetLastError' )", {} ),
			"x = compiler.has_library('kernel32', 'GetLastError')",
		)

	def test_malformed_args_left_unfolded( self ) -> None:
		self.assertEqual(
			_fold( "x = compiler.has_library( 'kernel32' )", {}, lambda: _CC ),
			"x = compiler.has_library('kernel32')",
		)

	@unittest.skipUnless( _CC is not None, 'no C compiler (clang/gcc/msvc) found' )
	def test_available_symbol_folds_to_true( self ) -> None:
		self.assertEqual(
			_fold( f"x = compiler.has_library( '{test_support.KNOWN_LIB}', '{test_support.KNOWN_SYMBOL}' )", {}, lambda: _CC ),
			'x = True',
		)

	@unittest.skipUnless( _CC is not None, 'no C compiler (clang/gcc/msvc) found' )
	def test_unavailable_symbol_folds_to_false( self ) -> None:
		self.assertEqual(
			_fold( "x = compiler.has_library( 'kernel32', 'ThisIsNotARealSymbol123' )", {}, lambda: _CC ),
			'x = False',
		)

	@unittest.skipUnless( _CC is not None, 'no C compiler (clang/gcc/msvc) found' )
	def test_substitution_feeds_if_collapse( self ) -> None:
		# the whole point: the losing branch is gone from the AST entirely,
		# not just skipped at runtime - a real @extern reference inside it
		# would never even reach lowering.py, let alone the linker
		src = f"if compiler.has_library( '{test_support.KNOWN_LIB}', '{test_support.KNOWN_SYMBOL}' ):\n\ta = 1\nelse:\n\ta = 2"
		self.assertEqual( _fold( src, {}, lambda: _CC ), 'a = 1' )
		src2 = f"if compiler.has_library( '{test_support.KNOWN_LIB}', 'ThisIsNotARealSymbol123' ):\n\ta = 1\nelse:\n\ta = 2"
		self.assertEqual( _fold( src2, {}, lambda: _CC ), 'a = 2' )

	@unittest.skipUnless( _CC is not None, 'no C compiler (clang/gcc/msvc) found' )
	def test_negation_via_not( self ) -> None:
		src = "if not compiler.has_library( 'kernel32', 'ThisIsNotARealSymbol123' ):\n\ta = 1\nelse:\n\ta = 2"
		self.assertEqual( _fold( src, {}, lambda: _CC ), 'a = 1' )


class IfSimplificationTests( unittest.TestCase ):
	def test_true_test_keeps_body_drops_orelse( self ) -> None:
		self.assertEqual( _fold( 'if True:\n\ta = 1\nelse:\n\ta = 2', {} ), 'a = 1' )

	def test_false_test_keeps_orelse_drops_body( self ) -> None:
		self.assertEqual( _fold( 'if False:\n\ta = 1\nelse:\n\ta = 2', {} ), 'a = 2' )

	def test_false_test_with_no_orelse_eliminated_entirely( self ) -> None:
		self.assertEqual( _fold( 'if False:\n\ta = 1\nb = 2', {} ), 'b = 2' )

	def test_non_constant_test_left_alone( self ) -> None:
		self.assertEqual( _fold( 'if cond:\n\ta = 1\nelse:\n\ta = 2', {} ), 'if cond:\n    a = 1\nelse:\n    a = 2' )

	def test_compiler_target_condition_folds_end_to_end( self ) -> None:
		src = "if compiler.target.os == 'windows':\n\ta = 1\nelse:\n\ta = 2"
		self.assertEqual( _fold( src, { 'os': 'windows' } ), 'a = 1' )
		self.assertEqual( _fold( src, { 'os': 'linux' } ), 'a = 2' )


class WhileSimplificationTests( unittest.TestCase ):
	def test_false_test_eliminated_entirely( self ) -> None:
		self.assertEqual( _fold( 'while False:\n\tfoo()\nbar()', {} ), 'bar()' )

	def test_true_test_left_as_ordinary_infinite_loop( self ) -> None:
		self.assertEqual( _fold( 'while True:\n\tbreak', {} ), 'while True:\n    break' )

	def test_non_constant_test_left_alone( self ) -> None:
		self.assertEqual( _fold( 'while cond:\n\tfoo()', {} ), 'while cond:\n    foo()' )


class MatchSimplificationTests( unittest.TestCase ):
	_SRC = (
		'match compiler.target.bits:\n'
		'\tcase 32:\n'
		'\t\ta = 1\n'
		'\tcase 64:\n'
		'\t\ta = 2\n'
		'\tcase _:\n'
		'\t\ta = 3\n'
	)

	def test_matching_literal_case_wins( self ) -> None:
		self.assertEqual( _fold( self._SRC, { 'bits': 32 } ), 'a = 1' )
		self.assertEqual( _fold( self._SRC, { 'bits': 64 } ), 'a = 2' )

	def test_wildcard_case_is_the_fallback( self ) -> None:
		self.assertEqual( _fold( self._SRC, { 'bits': 16 } ), 'a = 3' )

	def test_no_matching_case_and_no_wildcard_drops_statement_entirely( self ) -> None:
		src = "match compiler.target.os:\n\tcase 'windows':\n\t\ta = 1\n"
		self.assertEqual( _fold( src, { 'os': 'linux' } ), '' )

	def test_or_pattern_folds( self ) -> None:
		src = 'match compiler.target.bits:\n\tcase 32 | 64:\n\t\ta = 1\n\tcase _:\n\t\ta = 2\n'
		self.assertEqual( _fold( src, { 'bits': 64 } ), 'a = 1' )
		self.assertEqual( _fold( src, { 'bits': 16 } ), 'a = 2' )

	def test_bare_name_capture_synthesizes_bind( self ) -> None:
		src = 'match compiler.target.bits:\n\tcase x:\n\t\ta = x\n'
		self.assertEqual( _fold( src, { 'bits': 64 } ), 'x = 64\na = x' ) # TODO FIXME: I think this needs to be: x = 64\na = x\ndel x

	def test_as_binding_on_a_literal_pattern_synthesizes_bind( self ) -> None:
		src = 'match compiler.target.bits:\n\tcase 64 as x:\n\t\ta = x\n'
		self.assertEqual( _fold( src, { 'bits': 64 } ), 'x = 64\na = x' ) # TODO FIXME: I think this needs to be: x = 64\na = x\ndel x

	def test_singleton_pattern_folds( self ) -> None:
		src = 'match compiler.target.debug:\n\tcase True:\n\t\ta = 1\n\tcase False:\n\t\ta = 2\n'
		self.assertEqual( _fold( src, { 'debug': False } ), 'a = 2' )

	def test_guard_true_keeps_case( self ) -> None:
		src = 'match compiler.target.bits:\n\tcase 64 if True:\n\t\ta = 1\n\tcase _:\n\t\ta = 2\n'
		self.assertEqual( _fold( src, { 'bits': 64 } ), 'a = 1' )

	def test_guard_false_skips_to_next_case( self ) -> None:
		src = 'match compiler.target.bits:\n\tcase 64 if False:\n\t\ta = 1\n\tcase _:\n\t\ta = 2\n'
		self.assertEqual( _fold( src, { 'bits': 64 } ), 'a = 2' )

	def test_non_constant_guard_bails_on_whole_statement( self ) -> None:
		# the subject itself still substitutes (compiler.target.bits -> 64,
		# same generic substitution as everywhere else) even though the
		# match structure can't collapse since `cond`'s truth isn't known
		src = 'match compiler.target.bits:\n\tcase 64 if cond:\n\t\ta = 1\n\tcase _:\n\t\ta = 2\n'
		expected = 'match 64:\n    case 64 if cond:\n        a = 1\n    case _:\n        a = 2'
		self.assertEqual( _fold( src, { 'bits': 64 } ), expected )

	def test_non_constant_subject_left_alone( self ) -> None:
		src = 'match some_runtime_value:\n\tcase 1:\n\t\ta = 1\n'
		self.assertEqual( _fold( src, {} ), src.strip( '\n' ).replace( '\t', '    ' ))

	def test_matchclass_pattern_bails_on_whole_statement( self ) -> None:
		# MatchClass/MatchSequence/MatchMapping/MatchStar aren't attempted -
		# bail rather than guess (subject still substitutes, same as above)
		src = 'match compiler.target.bits:\n\tcase Foo(x):\n\t\ta = 1\n\tcase _:\n\t\ta = 2\n'
		expected = 'match 64:\n    case Foo(x):\n        a = 1\n    case _:\n        a = 2'
		self.assertEqual( _fold( src, { 'bits': 64 } ), expected )


class TransformExprTests( unittest.TestCase ):
	''' the single-expression sibling of transform_function_body - used by
	discovery.py's visit_Assign/visit_AnnAssign to fold a global/attribute
	initializer, a context that never lowers a statement list at all '''

	def _fold_expr( self, src: str, active_target: ActiveTarget ) -> str:
		node = ast.parse( src, mode = 'eval' ).body
		return ast.unparse( ctt.transform_expr( node, active_target ))

	def test_binop_folds( self ) -> None:
		self.assertEqual( self._fold_expr( '1 + 1', {} ), '2' )

	def test_unary_minus_on_constant_folds( self ) -> None:
		self.assertEqual( self._fold_expr( '-12', {} ), '-12' )

	def test_compiler_target_substitution_folds( self ) -> None:
		self.assertEqual( self._fold_expr( "compiler.target.os == 'windows'", { 'os': 'windows' } ), 'True' )

	def test_non_foldable_expr_returned_unchanged( self ) -> None:
		self.assertEqual( self._fold_expr( 'some_call()', {} ), 'some_call()' )


class TopLevelFoldingTests( unittest.TestCase ):
	''' top-level if/else folding: module-level `if compiler.target.os == ...`
	blocks are folded by transform_stmt_list the same way function-body if
	statements already were — this is the feature that lets lib/fs.py write a
	single write_all body with a compile-time-conditional TypeAlias instead of
	two @compiler.target overloads. '''

	def _fold_top( self, src: str, active_target: ActiveTarget ) -> str:
		body = ctt.transform_stmt_list( ast.parse( src ).body, active_target )
		return '\n'.join( ast.unparse( stmt ) for stmt in body )

	def test_constant_true_if_keeps_body( self ) -> None:
		self.assertEqual( self._fold_top( 'if True:\n\ta = 1\nelse:\n\ta = 2', {} ), 'a = 1' )

	def test_constant_false_if_keeps_else( self ) -> None:
		self.assertEqual( self._fold_top( 'if False:\n\ta = 1\nelse:\n\ta = 2', {} ), 'a = 2' )

	def test_constant_false_if_without_else_dropped( self ) -> None:
		self.assertEqual( self._fold_top( 'if False:\n\ta = 1\nb = 2', {} ), 'b = 2' )

	def test_compiler_target_condition_folds_top_level( self ) -> None:
		src = "if compiler.target.os == 'windows':\n\ta = 1\nelse:\n\ta = 2"
		self.assertEqual( self._fold_top( src, { 'os': 'windows' } ), 'a = 1' )
		self.assertEqual( self._fold_top( src, { 'os': 'linux' } ), 'a = 2' )

	def test_typealias_pattern_folds( self ) -> None:
		''' the motivating case: compile-time conditional TypeAlias '''
		src = (
			"if compiler.target.os == 'windows':\n"
			'\tFD: TypeAlias = 42\n'
			'else:\n'
			'\tFD: TypeAlias = 99\n'
		)
		self.assertEqual( self._fold_top( src, { 'os': 'windows' } ), 'FD: TypeAlias = 42' )
		self.assertEqual( self._fold_top( src, { 'os': 'linux' } ), 'FD: TypeAlias = 99' )

	def test_non_constant_if_left_untouched( self ) -> None:
		src = 'if runtime_val:\n\ta = 1\nelse:\n\ta = 2'
		self.assertEqual( self._fold_top( src, {} ).strip(), src.replace( '\t', '    ' ))

	def test_folded_statement_lands_adjacent_to_unfolded_statement( self ) -> None:
		''' after folding, the surviving statement appears in the correct
		position relative to other top-level statements '''
		src = "if True:\n\ta = 1\nb = 2"
		self.assertEqual( self._fold_top( src, {} ), 'a = 1\nb = 2' )


@unittest.skipUnless( test_support.HAS_CC, 'no C compiler (clang/gcc/msvc) found - skipping' )
class RealCompileFoldConsistencyTests( test_support.RealCompileMixin, unittest.TestCase ):
	''' the property that actually matters for // and % constant folding:
	a folded (compile-time-constant) expression and the SAME expression
	computed at runtime (operands hidden behind named locals, so this
	pass can't fold it) must produce IDENTICAL results - fold-vs-no-fold
	must never change program behavior. '''

	def setUp( self ) -> None:
		self.discovery = Discovery( import_builtins = True )
		self.compiler = Compiler( self.discovery )

	def test_folded_and_runtime_mod_floordiv_agree_on_negative_operands( self ) -> None:
		self.assert_programs_run([
			( 'folded_negative_dividend_mod', '''
def main() -> i32:
	with compiler.panic_arithmetic('x'):
		m: i32 = -7 % 2
	if m != -1:
		return 1
	return 0
''' ),
			( 'runtime_negative_dividend_mod', '''
def main() -> i32:
	a: i32 = -7
	b: i32 = 2
	with compiler.panic_arithmetic('x'):
		m: i32 = a % b
	if m != -1:
		return 1
	return 0
''' ),
			( 'folded_negative_divisor_mod', '''
def main() -> i32:
	with compiler.panic_arithmetic('x'):
		m: i32 = 7 % -2
	if m != 1:
		return 1
	return 0
''' ),
			( 'runtime_negative_divisor_mod', '''
def main() -> i32:
	a: i32 = 7
	b: i32 = -2
	with compiler.panic_arithmetic('x'):
		m: i32 = a % b
	if m != 1:
		return 1
	return 0
''' ),
			( 'folded_negative_dividend_floordiv', '''
def main() -> i32:
	with compiler.panic_arithmetic('x'):
		q: i32 = -7 // 2
	if q != -3:
		return 1
	return 0
''' ),
			( 'runtime_negative_dividend_floordiv', '''
def main() -> i32:
	a: i32 = -7
	b: i32 = 2
	with compiler.panic_arithmetic('x'):
		q: i32 = a // b
	if q != -3:
		return 1
	return 0
''' ),
		])


if __name__ == '__main__':
	unittest.main()
