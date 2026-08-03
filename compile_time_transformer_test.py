# stdlib imports
import ast
import unittest

# local imports
import compile_time_transformer as ctt


def _fold( src: str, active_target: dict[str,object] ) -> str:
	body = ctt.transform_function_body( ast.parse( src ).body, active_target )
	return '\n'.join( ast.unparse( stmt ) for stmt in body )


class ConstantFoldingTests( unittest.TestCase ):
	def test_binop_folds( self ) -> None:
		self.assertEqual( _fold( 'x = 1 + 1', {} ), 'x = 2' )

	def test_binop_bitwise_folds( self ) -> None:
		self.assertEqual( _fold( 'x = 6 & 3', {} ), 'x = 2' )

	def test_binop_division_by_zero_left_unfolded( self ) -> None:
		self.assertEqual( _fold( 'x = 1 // 0', {} ), 'x = 1 // 0' )

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

	def test_compare_chained_left_unfolded( self ) -> None:
		self.assertEqual( _fold( 'x = 1 < 2 < 3', {} ), 'x = 1 < 2 < 3' )

	def test_compare_is_left_unfolded( self ) -> None:
		# matches lowering.py's own deliberate exclusion of is/is not - see
		# _expr_Compare's docstring
		self.assertEqual( _fold( 'x = None is None', {} ), 'x = None is None' )

	def test_nested_folds_bottom_up( self ) -> None:
		self.assertEqual( _fold( 'x = (1 + 1) == 2', {} ), 'x = True' )


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


if __name__ == '__main__':
	unittest.main()
