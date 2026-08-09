# stdlib imports:
import ast
import logging
from pathlib import Path
import queue as queue_module
import unittest

# local imports:
from discovery import Discovery
from mpy_types import CEnum, Function
from type_resolver import TypeResolver

class TypeResolutionTests( unittest.TestCase ):
	''' TypeResolver built directly against a real Discovery instance (same
	discipline as monomorphize_test.py/cfg_test.py) - never lowering.py
	itself, since TypeResolver depends only on Discovery/UnionStorage/
	Monomorphizer. Assertions read the rewritten AST directly via
	ast.unparse() - the whole point of moving union disambiguation here is
	that it's now visually checkable, the way compile_time_transformer_test.py
	already checks constant folding. '''

	def setUp( self ) -> None:
		self.discovery = Discovery()
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

	# --- if / while truthiness (rewrite 1b: T|None truthiness) ----------

	def test_if_bool_or_none_param_rewrites_to_tag_and_payload( self ) -> None:
		# `if exists:` where exists: bool|None — tag check AND the bool payload
		# itself (bool has no __bool__() to call; the value IS the boolean)
		mod = self._import( '\n'.join([
			'@union',
			'class MaybeBool:',
			'	Val: bool',
			'	Absent: None',
			'',
			'def main( x: MaybeBool ) -> i32:',
			'	if x:',
			'		return 1',
			'	return 0',
		]))
		fn = self._resolved_fn( mod, 'main' )
		src = ast.unparse( fn.node )
		self.assertNotIn( 'if x:', src )
		self.assertIn( 'if x.tag != 1 and x.data.v_Val:', src )

	def test_if_bool_or_none_param_with_else_rewrites( self ) -> None:
		mod = self._import( '\n'.join([
			'@union',
			'class MaybeBool:',
			'	Absent: None',
			'	Val: bool',
			'',
			'def main( x: MaybeBool ) -> i32:',
			'	if x:',
			'		return 1',
			'	else:',
			'		return 0',
		]))
		fn = self._resolved_fn( mod, 'main' )
		src = ast.unparse( fn.node )
		self.assertIn( 'if x.tag != 0 and x.data.v_Val:', src )
		self.assertIn( 'else', src )

	def test_while_bool_or_none_param_rewrites( self ) -> None:
		mod = self._import( '\n'.join([
			'@union',
			'class MaybeBool:',
			'	Val: bool',
			'	Absent: None',
			'',
			'def main( x: MaybeBool ) -> None:',
			'	while x:',
			'		pass',
		]))
		fn = self._resolved_fn( mod, 'main' )
		src = ast.unparse( fn.node )
		self.assertNotIn( 'while x:', src )
		self.assertIn( 'while x.tag != 1 and x.data.v_Val:', src )

	def test_if_with_non_bool_leaf_calls___bool__( self ) -> None:
		# x: MaybeString where the non-None variant is `str`, not `bool` -
		# needs x.data.v_Some.__bool__() since the payload isn't itself truthy
		mod = self._import( '\n'.join([
			'@union',
			'class MaybeString:',
			'	Some: str',
			'	Nothing: None',
			'',
			'def main( x: MaybeString ) -> i32:',
			'	if x:',
			'		return 1',
			'	return 0',
		]))
		fn = self._resolved_fn( mod, 'main' )
		src = ast.unparse( fn.node )
		self.assertIn( 'x.data.v_Some.__bool__()', src )

	def test_if_on_annotated_bool_or_none_local_rewrites( self ) -> None:
		mod = self._import( '\n'.join([
			'@union',
			'class MaybeBool:',
			'	Val: bool',
			'	Absent: None',
			'',
			'def main() -> i32:',
			'	x: MaybeBool = MaybeBool.Absent( None )',
			'	if x:',
			'		return 1',
			'	return 0',
		]))
		fn = self._resolved_fn( mod, 'main' )
		src = ast.unparse( fn.node )
		self.assertIn( 'x.tag != 1 and x.data.v_Val:', src )

	def test_if_on_non_union_type_is_left_unchanged( self ) -> None:
		mod = self._import( '\n'.join([
			'def main( flag: bool ) -> i32:',
			'	if flag:',
			'		return 1',
			'	return 0',
		]))
		fn = self._resolved_fn( mod, 'main' )
		src = ast.unparse( fn.node )
		self.assertIn( 'if flag:', src )
		self.assertNotIn( '.tag', src )

	def test_if_on_union_without_none_is_left_unchanged( self ) -> None:
		# T|U with no None member — auto-generated union __bool__ is future
		# work; leave the condition untouched for now
		mod = self._import( '\n'.join([
			'@union',
			'class IntOrStr:',
			'	V: i32',
			'	S: str',
			'',
			'def main( x: IntOrStr ) -> i32:',
			'	if x:',
			'		return 1',
			'	return 0',
		]))
		fn = self._resolved_fn( mod, 'main' )
		src = ast.unparse( fn.node )
		self.assertIn( 'if x:', src )
		self.assertNotIn( '.tag', src )


	# --- BoolOp (and / or) truthiness -------------------------------------

	def test_and_with_bool_or_none_operands_rewrites_both( self ) -> None:
		mod = self._import( '\n'.join([
			'@union',
			'class MaybeBool:',
			'	Val: bool',
			'	Absent: None',
			'',
			'def main( x: MaybeBool, y: MaybeBool ) -> i32:',
			'	if x and y:',
			'		return 1',
			'	return 0',
		]))
		fn = self._resolved_fn( mod, 'main' )
		src = ast.unparse( fn.node )
		# each operand of `and` gets its own truthiness rewrite
		self.assertIn( '(x.tag != 1 and x.data.v_Val) and (y.tag != 1 and y.data.v_Val)', src )

	def test_or_with_bool_or_none_operands_rewrites_both( self ) -> None:
		mod = self._import( '\n'.join([
			'@union',
			'class MaybeBool:',
			'	Val: bool',
			'	Absent: None',
			'',
			'def main( x: MaybeBool, y: MaybeBool ) -> i32:',
			'	if x or y:',
			'		return 1',
			'	return 0',
		]))
		fn = self._resolved_fn( mod, 'main' )
		src = ast.unparse( fn.node )
		self.assertIn( 'x.tag != 1 and x.data.v_Val or (y.tag != 1 and y.data.v_Val)', src )

	def test_and_with_mixed_bool_or_none_and_plain_bool_rewrites_only_union_operand( self ) -> None:
		# one operand is bool|None, the other is plain bool — only the union
		# operand should be rewritten; the plain bool stays as-is
		mod = self._import( '\n'.join([
			'@union',
			'class MaybeBool:',
			'	Val: bool',
			'	Absent: None',
			'',
			'def main( x: MaybeBool, flag: bool ) -> i32:',
			'	if x and flag:',
			'		return 1',
			'	return 0',
		]))
		fn = self._resolved_fn( mod, 'main' )
		src = ast.unparse( fn.node )
		# x gets rewritten, flag stays bare
		self.assertIn( '(x.tag != 1 and x.data.v_Val) and flag', src )

	def test_ternary_with_bool_or_none_condition_rewrites( self ) -> None:
		mod = self._import( '\n'.join([
			'@union',
			'class MaybeBool:',
			'	Val: bool',
			'	Absent: None',
			'',
			'def main( x: MaybeBool ) -> i32:',
			'	return 1 if x else 0',
		]))
		fn = self._resolved_fn( mod, 'main' )
		src = ast.unparse( fn.node )
		self.assertIn( '1 if x.tag != 1 and x.data.v_Val else 0', src )
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

	def test_cenum_member_reference_resolves_and_schedules_the_enum( self ) -> None:
		mod = self._import( '\n'.join([
			'import compiler',
			'@enum( u32 )',
			'class MyError:',
			'	FileNotFound = 2',
			'	Other = _',
			'',
			'def main() -> u32:',
			'	return MyError.FileNotFound',
		]))
		fn = self._resolved_fn( mod, 'main' )
		self.assertEqual( self.discovery.errors.errors, [] )
		# the CEnum must be fully resolved after resolve_function_body
		myerr = next( u for u in self.resolver.queue.queue if isinstance( u, CEnum ) )
		self.assertIsNone( myerr.resolve )
		self.assertEqual( myerr.members, { 'FileNotFound': 2, 'Other': 3 } )

	# --- idempotency --------------------------------------------------------

	def test_resolve_function_body_is_idempotent( self ) -> None:
		# memoized by id(fn.node) - calling this twice against the same
		# fn.node must not double-rewrite it
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

	# --- generic function call resolution ------------------------------------

	def _resolved_callees( self, fn ) -> list:
		# every top-level expression-statement's own Call node, in order -
		# every test below is a straight-line sequence of bare `foo(...)`
		# statements, so this is enough to pull out node.resolved_callee
		# (or None, for a call this pass declined to resolve) per statement
		return [ getattr( stmt.value, 'resolved_callee', None ) for stmt in fn.node.body if isinstance( stmt, ast.Expr ) ]

	def test_implicit_generic_call_tags_resolved_callee_per_argument_type( self ) -> None:
		# the exact shape from ARCHITECTURE's own generic-function example -
		# foo('hello') and foo(42) each get their own monomorphized foo,
		# distinguished by argument type alone (no explicit foo[T])
		mod = self._import( '\n'.join([
			'class str: pass',
			'class int: pass',
			'',
			'def foo[T]( t: T ) -> T:',
			'	return t',
			'',
			'def main() -> None:',
			"	foo( 'hello' )",
			'	foo( 42 )',
		]))
		fn = self._resolved_fn( mod, 'main' )
		self.assertEqual( self.discovery.errors.errors, [] )
		str_callee, int_callee = self._resolved_callees( fn )
		self.assertIsNotNone( str_callee )
		self.assertIsNotNone( int_callee )
		self.assertEqual( str_callee.qualname, '__test__.foo[__test__.str]' )
		self.assertEqual( int_callee.qualname, '__test__.foo[__test__.int]' )
		self.assertIsNot( str_callee, int_callee )
		self.assertIsNot( str_callee.node, int_callee.node ) # independent, deep-copied bodies - not the shared abstract one

	def test_explicit_generic_call_tags_resolved_callee( self ) -> None:
		mod = self._import( '\n'.join([
			'def identity[T]( x: T ) -> T:',
			'	return x',
			'',
			'def main() -> None:',
			'	identity[i32]( 5 )',
		]))
		fn = self._resolved_fn( mod, 'main' )
		self.assertEqual( self.discovery.errors.errors, [] )
		[ callee ] = self._resolved_callees( fn )
		self.assertIsNotNone( callee )
		self.assertEqual( callee.qualname, '__test__.identity[intrinsics.i32]' )

	def test_repeated_call_with_same_inferred_type_reuses_the_same_callee( self ) -> None:
		mod = self._import( '\n'.join([
			'class int: pass',
			'',
			'def identity[T]( x: T ) -> T:',
			'	return x',
			'',
			'def main() -> None:',
			'	identity( 1 )',
			'	identity( 2 )',
		]))
		fn = self._resolved_fn( mod, 'main' )
		self.assertEqual( self.discovery.errors.errors, [] )
		first, second = self._resolved_callees( fn )
		self.assertIsNotNone( first )
		self.assertIs( first, second )

	def test_generic_call_on_receiver_local_is_left_untagged( self ) -> None:
		# x.method() where x is a plain local - fn.names only gains local
		# entries incrementally as LOWERING itself walks the body, which
		# hasn't happened yet at this pre-lowering pass, so this pass can't
		# (and shouldn't) resolve a receiver-based call at all; must not
		# raise/report a spurious "not defined" error either - deferred
		# entirely to lowering.py's own, unaffected receiver-call handling
		mod = self._import( '\n'.join([
			'class Foo:',
			'	def method( self ) -> i32:',
			'		return 1',
			'',
			'def main() -> None:',
			'	x: Foo',
			'	x.method()',
		]))
		fn = self._resolved_fn( mod, 'main' )
		self.assertEqual( self.discovery.errors.errors, [] )
		[ callee ] = self._resolved_callees( fn )
		self.assertIsNone( callee )

	def test_generic_call_with_uninferrable_argument_is_left_untagged( self ) -> None:
		# T never appears in any parameter position - nothing for this
		# pass to infer it from either, same as lowering's own eventual
		# failure - left untagged (no error recorded HERE; lowering's own
		# fallback is what actually reports it, unchanged)
		mod = self._import( '\n'.join([
			'def make[T]() -> i32:',
			'	return 0',
			'',
			'def main() -> None:',
			'	make()',
		]))
		fn = self._resolved_fn( mod, 'main' )
		self.assertEqual( self.discovery.errors.errors, [] )
		[ callee ] = self._resolved_callees( fn )
		self.assertIsNone( callee )

	# --- generic construction resolution --------------------------------------

	def test_generic_construction_via_argument_type_tags_resolved_construction( self ) -> None:
		# the TODO.txt-confirmed bug this closes: a generic class's own
		# __init__ was never monomorphized when reached through plain
		# ClassName(...) construction inside a helper function - it only
		# ever worked when constructed directly against an annotation
		# lowering.py could pin an expected_type from
		mod = self._import( '\n'.join([
			'class Box[T]:',
			'	v: T',
			'	def __init__( self, v: T ) -> None:',
			'		self.v = v',
			'',
			'def make_box( x: i32 ) -> None:',
			'	b: Box[i32] = Box( x )',
		]))
		fn = self._resolved_fn( mod, 'make_box' )
		self.assertEqual( self.discovery.errors.errors, [] )
		ann_assign = fn.node.body[0]
		self.assertIsInstance( ann_assign, ast.AnnAssign )
		construction = getattr( ann_assign.value, 'resolved_construction', None )
		self.assertIsNotNone( construction )
		concrete_cls, concrete_init = construction
		self.assertEqual( concrete_cls.qualname, '__test__.Box[intrinsics.i32]' )
		self.assertEqual( concrete_init.qualname, '__test__.Box.__init__[intrinsics.i32]' )

	def test_generic_construction_with_literal_argument_is_left_untagged( self ) -> None:
		# a bare literal argument (Box(1)) has no safe, context-free
		# answer for this pass - its own default type (builtins.int) can
		# genuinely disagree with what an expected_type annotation would
		# have pinned (b: Box[i32] = Box(1) means T=i32, not int) -
		# lowering.py's own _lower_generic_construction_args sees that
		# annotation and gets it right; this pass has no expected-type
		# context at all, so it must never guess here (confirmed by a
		# real repro during development: guessing built and compiled an
		# extra, wrong Box[int] specialization alongside the real one)
		mod = self._import( '\n'.join([
			'class Box[T]:',
			'	v: T',
			'	def __init__( self, v: T ) -> None:',
			'		self.v = v',
			'',
			'def main() -> None:',
			'	b: Box[i32] = Box( 1 )',
		]))
		fn = self._resolved_fn( mod, 'main' )
		self.assertEqual( self.discovery.errors.errors, [] )
		ann_assign = fn.node.body[0]
		self.assertIsNone( getattr( ann_assign.value, 'resolved_construction', None ))

	def test_generic_construction_with_fallible_init_is_left_untagged( self ) -> None:
		# Foo(...) where __init__ returns Result[None,E] becomes
		# Result[Foo,E] (SYNTAX.md) - deliberately out of scope for this
		# pass (no Ok/Err-wrapping synthesis here), left for lowering's
		# existing _init_fallibility/_emit_fallible_construction machinery
		mod = self._import( '\n'.join([
			'class MyError: pass',
			'',
			'class Box[T]:',
			'	v: T',
			'	def __init__( self, v: T ) -> Result[None,MyError]:',
			'		self.v = v',
			'		return Result.Ok( None )',
			'',
			'def make_box( x: i32 ) -> None:',
			'	r = Box( x )',
		]))
		fn = self._resolved_fn( mod, 'make_box' )
		self.assertEqual( self.discovery.errors.errors, [] )
		assign = fn.node.body[0]
		self.assertIsInstance( assign, ast.Assign )
		self.assertIsNone( getattr( assign.value, 'resolved_construction', None ))

	# --- destructor synthesis -------------------------------------------------

	def _resolve_class( self, mod, name: str ):
		''' resolve a class body and its attributes, same as
		compiler._lower would do for a bare RCClass '''
		cls = mod.get_local( name )
		if cls.resolve is not None:
			cls.resolve()
		for attr in cls.attributes:
			if attr.resolve is not None:
				attr.resolve()
		return cls

	def _synthesize_and_dequeue( self, cls ):
		''' call _synthesize_rcclass_destructor and dequeue the resulting
		Function from the resolver's work queue '''
		self.resolver._synthesize_rcclass_destructor( cls )
		queued = []
		while True:
			try:
				queued.append( self.resolver.queue.get_nowait() )
			except queue_module.Empty:
				break
		dtors = [ u for u in queued if isinstance( u, Function ) and u.is_destructor and u.qualname.startswith( cls.qualname ) ]
		self.assertEqual( len( dtors ), 1, f'expected exactly one destructor in the queue for {cls.qualname}, got {[u.qualname for u in dtors]}' )
		return dtors[0]

	def test_destructor_synthesis_basic( self ) -> None:
		mod = self._import( '\n'.join([
			'class Foo: pass',
		]))
		cls = self._resolve_class( mod, 'Foo' )
		dtor = self._synthesize_and_dequeue( cls )
		src = ast.unparse( dtor.node )
		# must call sys.free(self)
		self.assertIn( 'sys.free', src )
		self.assertIn( 'self', src )
		# no __del__ call since Foo doesn't declare one
		self.assertNotIn( '__del__', src )
		# no compiler.decref since Foo has no fields
		self.assertNotIn( 'compiler.decref', src )
		self.assertTrue( dtor.is_destructor )
		self.assertTrue( dtor.is_static )
		self.assertIsNone( dtor.cls )

	def test_destructor_synthesis_calls_del_before_free( self ) -> None:
		mod = self._import( '\n'.join([
			'class Owner:',
			'	x: i32',
			'	def __del__( self ) -> None:',
			'		pass',
		]))
		cls = self._resolve_class( mod, 'Owner' )
		dtor = self._synthesize_and_dequeue( cls )
		src = ast.unparse( dtor.node )
		# __del__ must appear before sys.free
		del_pos = src.index( 'self.__del__()' )
		free_pos = src.index( 'sys.free' )
		self.assertLess( del_pos, free_pos )

	def test_destructor_synthesis_decrefs_rc_field( self ) -> None:
		mod = self._import( '\n'.join([
			'class Inner: pass',
			'class Outer:',
			'	inner: Inner',
		]))
		cls = self._resolve_class( mod, 'Outer' )
		dtor = self._synthesize_and_dequeue( cls )
		src = ast.unparse( dtor.node )
		self.assertIn( 'compiler.decref(self.inner)', src )
		# decref before sys.free
		decref_pos = src.index( 'compiler.decref' )
		free_pos = src.index( 'sys.free' )
		self.assertLess( decref_pos, free_pos )

	def test_destructor_synthesis_cascades_through_cstruct_field( self ) -> None:
		mod = self._import( '\n'.join([
			'class Inner: pass',
			'@cstruct',
			'class Wrapper:',
			'	inner: Inner',
			'class Outer:',
			'	w: Wrapper',
		]))
		cls = self._resolve_class( mod, 'Outer' )
		dtor = self._synthesize_and_dequeue( cls )
		src = ast.unparse( dtor.node )
		# should see compiler.decref(self.w.inner) — the cascade through the CStruct
		self.assertIn( 'compiler.decref(self.w.inner)', src )
		self.assertNotIn( 'compiler.decref(self.w)', src )  # Wrapper itself isn't RC

	def test_destructor_synthesis_tagged_union_decrefs_active_rc_member( self ) -> None:
		mod = self._import( '\n'.join([
			'class Foo: pass',
			'@union',
			'class MaybeFoo:',
			'	Some: Foo',
			'	Nothing: i32',
			'class Outer:',
			'	maybe: MaybeFoo',
		]))
		cls = self._resolve_class( mod, 'Outer' )
		dtor = self._synthesize_and_dequeue( cls )
		src = ast.unparse( dtor.node )
		# should have a conditional on the tag
		self.assertIn( '__dtor_tag_', src )
		self.assertIn( 'if __dtor_tag_0 == 0:', src )  # tag == 0 → Some
		self.assertIn( 'compiler.decref(self.maybe.data.v_Some)', src )
		# the i32 (Nothing) member should not trigger any decref
		self.assertNotIn( 'self.maybe.data.v_Nothing', src )

	def test_destructor_synthesis_is_idempotent( self ) -> None:
		mod = self._import( '\n'.join([
			'class Foo: pass',
		]))
		cls = self._resolve_class( mod, 'Foo' )
		self.resolver._synthesize_rcclass_destructor( cls )
		# drain the queue
		while True:
			try:
				self.resolver.queue.get_nowait()
			except queue_module.Empty:
				break
		# second call should be a no-op (idempotency guard)
		self.resolver._synthesize_rcclass_destructor( cls )
		self.assertTrue( self.resolver.queue.empty(), 'second synthesis should not enqueue anything' )

if __name__ == '__main__':
	logging.basicConfig( level = logging.DEBUG, force = True )
	unittest.main()
