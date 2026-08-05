# stdlib imports:
import ast
from pathlib import Path
import unittest

# local imports:
from discovery import Discovery
from type_resolution import TypeResolver

class TypeResolutionTests( unittest.TestCase ):
	''' TypeResolver built directly against a real Discovery instance (same
	discipline as monomorphize_test.py/cfg_test.py) - never lowering.py
	itself, since TypeResolver depends only on Discovery/UnionStorage/
	Monomorphizer. Assertions read the rewritten AST directly via
	ast.unparse() - the whole point of moving union disambiguation here is
	that it's now visually checkable, the way compile_time_transformer_test.py
	already checks constant folding. '''

	def setUp( self ) -> None:
		self.discovery = Discovery( import_builtins = False )
		self.resolver = TypeResolver( self.discovery )

	def _import( self, code: str ):
		module = self.discovery.import_code( code, filename = Path( '__test__.py' ))
		# entry modules aren't registered in disco.modules on their own
		# (that's keyed by import package name) - resolve_function_body's
		# own _find_module_for needs to be able to find this one by file,
		# same registration Compiler.import_code already does
		self.discovery.modules[module.qualname] = module
		return module

	def _resolved_fn( self, mod, name: str ):
		fn = mod.get_local( name )
		if fn.resolve is not None:
			fn.resolve()
		self.resolver.resolve_function_body( fn )
		return fn

	# --- is None / is not None -------------------------------------------

	def test_is_none_on_real_union_rewrites_to_tag_compare( self ) -> None:
		mod = self._import( '\n'.join([
			'@union',
			'class Maybe:',
			'	Some: i32',
			'	Nothing: None',
			'',
			'def main( x: Maybe ) -> bool:',
			'	return x is None',
		]))
		fn = self._resolved_fn( mod, 'main' )
		src = ast.unparse( fn.node )
		self.assertNotIn( 'is None', src )
		self.assertIn( 'x.tag == 1', src ) # Nothing is declared second (ordinal 1)

	def test_is_not_none_uses_not_equal( self ) -> None:
		mod = self._import( '\n'.join([
			'@union',
			'class Maybe:',
			'	Nothing: None',
			'	Some: i32',
			'',
			'def main( x: Maybe ) -> bool:',
			'	return x is not None',
		]))
		fn = self._resolved_fn( mod, 'main' )
		src = ast.unparse( fn.node )
		self.assertNotIn( 'is not None', src )
		self.assertIn( 'x.tag != 0', src ) # Nothing is declared first (ordinal 0)

	def test_is_none_on_annotated_local_rewrites( self ) -> None:
		# the operand doesn't have to be a parameter - a plain annotated
		# local's declared type is tracked forward through the function body
		mod = self._import( '\n'.join([
			'@union',
			'class Maybe:',
			'	Some: i32',
			'	Nothing: None',
			'',
			'def main() -> bool:',
			'	x: Maybe = Maybe.Nothing( None )',
			'	return x is None',
		]))
		fn = self._resolved_fn( mod, 'main' )
		src = ast.unparse( fn.node )
		self.assertIn( 'x.tag == 1', src )

	def test_is_none_on_non_union_type_is_left_unchanged( self ) -> None:
		# ordinary is/is-not against something that isn't a TaggedUnion at
		# all - not this rewrite's shape, left for lowering.py's own
		# _lower_is_comparison to handle
		mod = self._import( '\n'.join([
			'class Foo: pass',
			'',
			'def main( x: Foo ) -> bool:',
			'	return x is None',
		]))
		fn = self._resolved_fn( mod, 'main' )
		src = ast.unparse( fn.node )
		self.assertIn( 'is None', src )
		self.assertNotIn( '.tag', src )

	def test_is_none_on_unresolvable_expression_is_left_unchanged( self ) -> None:
		# _type_of_expr is a best-effort, narrow tracker (Name/Attribute/
		# Call only) - anything else (a bare Constant, here) is left alone
		# rather than guessed at, falling back to lowering's own handling
		mod = self._import( '\n'.join([
			'def main() -> bool:',
			'	return True is None',
		]))
		fn = self._resolved_fn( mod, 'main' )
		src = ast.unparse( fn.node )
		self.assertIn( 'is None', src )

	def test_is_none_on_generic_union_specialization_rewrites( self ) -> None:
		# a Specialization-wrapped union (Box[i32], not the bare class) -
		# must unwrap to the abstract base for UnionStorage.get() (tag
		# ordinals don't vary by specialization), and must force the
		# abstract class's own body to resolve first (a bare annotation
		# never does that on its own - see visit_Compare's own comment)
		mod = self._import( '\n'.join([
			'@union',
			'class Box[T]:',
			'	Full: T',
			'	Empty: None',
			'',
			'def main() -> bool:',
			'	x: Box[i32]',
			'	return x is None',
		]))
		fn = self._resolved_fn( mod, 'main' )
		src = ast.unparse( fn.node )
		self.assertIn( 'x.tag == 1', src )

	# --- match ------------------------------------------------------------

	def test_match_union_rewrites_to_if_elif_chain( self ) -> None:
		mod = self._import( '\n'.join([
			'@union',
			'class Foo:',
			'	Bar: i32',
			'	Baz: i32',
			'',
			'def main( f: Foo ) -> i32:',
			'	match f:',
			'		case Foo.Bar( x ):',
			'			return x',
			'		case Foo.Baz( z ):',
			'			return z',
			'	return 0',
		]))
		fn = self._resolved_fn( mod, 'main' )
		src = ast.unparse( fn.node )
		self.assertNotIn( 'match ', src )
		self.assertNotIn( 'case ', src )
		self.assertIn( 'if __match_subj_0.tag == 0', src )
		self.assertIn( 'elif __match_subj_0.tag == 1', src )
		self.assertIn( 'x = __match_subj_0.data.v_Bar', src )
		self.assertIn( 'z = __match_subj_0.data.v_Baz', src )

	def test_match_bare_wildcard_binds_unconditionally( self ) -> None:
		mod = self._import( '\n'.join([
			'@union',
			'class Foo:',
			'	Bar: i32',
			'',
			'def main( f: Foo ) -> Foo:',
			'	match f:',
			'		case rest:',
			'			return rest',
			'	return f',
		]))
		fn = self._resolved_fn( mod, 'main' )
		src = ast.unparse( fn.node )
		self.assertNotIn( 'match ', src )
		self.assertIn( 'if True', src )
		self.assertIn( 'rest = __match_subj_0', src )

	def test_match_guard_is_rejected( self ) -> None:
		mod = self._import( '\n'.join([
			'@union',
			'class Foo:',
			'	Bar: i32',
			'',
			'def main( f: Foo ) -> i32:',
			'	match f:',
			'		case Foo.Bar( x ) if x > 0:',
			'			return x',
			'	return 0',
		]))
		fn = mod.get_local( 'main' )
		if fn.resolve is not None:
			fn.resolve()
		self.resolver.resolve_function_body( fn )
		self.assertTrue( self.discovery.errors.errors )
		self.assertIn( 'guards', self.discovery.errors.errors[0] )

	def test_match_unknown_member_is_rejected( self ) -> None:
		mod = self._import( '\n'.join([
			'@union',
			'class Foo:',
			'	Bar: i32',
			'',
			'def main( f: Foo ) -> None:',
			'	match f:',
			'		case Foo.NotAMember( x ):',
			'			pass',
		]))
		fn = mod.get_local( 'main' )
		if fn.resolve is not None:
			fn.resolve()
		self.resolver.resolve_function_body( fn )
		self.assertTrue( self.discovery.errors.errors )
		self.assertIn( 'has no member', self.discovery.errors.errors[0] )

	# --- idempotency --------------------------------------------------------

	def test_resolve_function_body_is_idempotent( self ) -> None:
		# a generic function's monomorphized copies all share the exact
		# same fn.node - calling this twice must not double-rewrite it
		mod = self._import( '\n'.join([
			'@union',
			'class Maybe:',
			'	Some: i32',
			'	Nothing: None',
			'',
			'def main( x: Maybe ) -> bool:',
			'	return x is None',
		]))
		fn = self._resolved_fn( mod, 'main' )
		first = ast.unparse( fn.node )
		self.resolver.resolve_function_body( fn ) # second call, same fn.node
		second = ast.unparse( fn.node )
		self.assertEqual( first, second )
