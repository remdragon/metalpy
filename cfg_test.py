# stdlib imports:
from pathlib import Path
import unittest

# local imports:
import cfg
import ir
from discovery import Discovery
from errors import CompileError
from mpy_types import Variable, CUnion

class CFGTestBase( unittest.TestCase ):
	''' cfg.py itself never imports ast/Discovery - but building realistic
	Function/Parameter/RCClass/TaggedUnion inputs by hand is exactly what
	overload_resolution_test.py already avoids by using a real Discovery
	instance just to construct them (never to run resolve_call itself) -
	same idea here: Discovery builds the inputs, cfg.CFGState is what's
	under test. '''

	def setUp( self ) -> None:
		self.discovery = Discovery( import_builtins = False )
		self._temp_id = 0
		self._label_id = 0
		self._union_storage_cache = {}

	def _import( self, code: str ):
		self._module = self.discovery.import_code( code, Path( '__test__.py' ))
		return self._module

	def _fn( self, name: str ):
		fn = self._module.get_local( name )
		if fn.resolve is not None:
			fn.resolve()
		return fn

	def _new_temp( self, t ) -> ir.Temp:
		temp = ir.Temp( type = t, id = self._temp_id )
		self._temp_id += 1
		return temp

	def _new_label( self, prefix: str ) -> str:
		self._label_id += 1
		return f'__{prefix}_{self._label_id}__'

	def _union_storage( self, union ):
		key = id( union )
		cached = self._union_storage_cache.get( key )
		if cached is not None:
			return cached
		u8_cls = self.discovery.get_intrinsics()['u8']
		tag_attr = Variable( stem = 'tag', qualname = f'{union.qualname}.tag', file = None, line = None, type = u8_cls )
		payload_fields = [
			Variable( stem = f'v_{a.stem}', qualname = f'{union.qualname}.data.v_{a.stem}', file = None, line = None, type = a.type )
			for a in union.attributes
		]
		payload_cls = CUnion(
			stem = f'{union.stem}$data', qualname = f'{union.qualname}$data', file = None, line = None,
			attributes = payload_fields, names = { f.stem: f for f in payload_fields },
		)
		data_attr = Variable( stem = 'data', qualname = f'{union.qualname}.data', file = None, line = None, type = payload_cls )
		tags = { a.stem: i for i, a in enumerate( union.attributes ) }
		result = ( tag_attr, data_attr, payload_cls, tags )
		self._union_storage_cache[key] = result
		return result

	def _state( self, fn ) -> cfg.CFGState:
		bool_cls = self.discovery.get_intrinsics()['bool']
		return cfg.CFGState(
			fn,
			bool_type = bool_cls,
			new_temp = self._new_temp,
			new_label = self._new_label,
			union_storage = self._union_storage,
		)

	def _kinds( self, instructions ) -> list[str]:
		return [ type( i ).__name__ for i in instructions ]

# --- prologue ----------------------------------------------------------------

class PrologueTests( CFGTestBase ):

	def test_plain_parameter_is_borrowed_no_instructions( self ) -> None:
		self._import( '''
class Foo: pass

def foo( x: Foo ) -> None:
	pass
''' )
		state = self._state( self._fn( 'foo' ))
		self.assertEqual( state.prologue_instructions, [] )
		self.assertEqual( state.bindings['x'].state, cfg.OwnState.BORROWED )
		self.assertIsNone( state.bindings['x'].entry )

	def test_move_parameter_is_owned_no_prologue_instructions( self ) -> None:
		self._import( '''
class Foo: pass

def foo( x: move[Foo] ) -> None:
	pass
''' )
		state = self._state( self._fn( 'foo' ))
		self.assertEqual( state.prologue_instructions, [] )
		self.assertEqual( state.bindings['x'].state, cfg.OwnState.OWNED )
		self.assertIsNotNone( state.bindings['x'].entry )

	def test_copy_parameter_increfs_in_prologue( self ) -> None:
		self._import( '''
class Foo: pass

def foo( x: copy[Foo] ) -> None:
	pass
''' )
		state = self._state( self._fn( 'foo' ))
		self.assertEqual( self._kinds( state.prologue_instructions ), ['Incref'] )
		self.assertIs( state.prologue_instructions[0].value, self._fn( 'foo' ).parameters[0] )
		self.assertEqual( state.bindings['x'].state, cfg.OwnState.COPY )

	def test_non_rc_parameter_is_not_tracked( self ) -> None:
		self._import( '''
def foo( x: i32 ) -> None:
	pass
''' )
		state = self._state( self._fn( 'foo' ))
		self.assertEqual( state.prologue_instructions, [] )
		self.assertNotIn( 'x', state.bindings )

	def test_mixed_rc_union_copy_parameter_is_tag_gated( self ) -> None:
		self._import( '''
class Foo: pass

def foo( x: copy[Foo|i32] ) -> None:
	pass
''' )
		state = self._state( self._fn( 'foo' ))
		self.assertEqual( self._kinds( state.prologue_instructions ), ['GetAttr', 'Cmp', 'JumpIfFalse', 'GetAttr', 'GetAttr', 'Incref', 'Jump', 'Label', 'Label'] )

	def test_all_rc_union_copy_parameter_is_unconditional( self ) -> None:
		self._import( '''
class Foo: pass
class Bar: pass

def foo( x: copy[Foo|Bar] ) -> None:
	pass
''' )
		state = self._state( self._fn( 'foo' ))
		self.assertEqual( self._kinds( state.prologue_instructions ), ['GetAttr', 'GetAttr', 'Incref'] )

	def test_enter_self_on_generic_rcclass_specialization_is_tracked( self ) -> None:
		# regression test: self typed as a concrete generic RCClass
		# Specialization (Box[i32]) must still be recognized as RC-managed by
		# is_rc/rc_leaves - a Specialization isn't an RCClass INSTANCE itself,
		# so before the unwrap fix, enter_self's own `if not rc_leaves(
		# self_param.type): return` silently skipped tracking self's lifecycle
		# entirely for every generic-class instance method
		self._import( '''
class Box[T]:
	v: T
	def get( self ) -> T:
		return self.v
''' )
		box_cls = self._module.get_local( 'Box' )
		if box_cls.resolve is not None:
			box_cls.resolve()
		get_fn = box_cls.get_local( 'get' )
		if get_fn.resolve is not None:
			get_fn.resolve()
		i32_cls = self.discovery.get_intrinsics()['i32']
		spec = self.discovery._get_or_create_specialization( box_cls, [ i32_cls ] )
		self_param = Variable( stem = 'self', qualname = f'{get_fn.qualname}.self', file = None, line = None, type = spec )
		state = self._state( get_fn )
		state.enter_self( self_param, is_move = False )
		self.assertIn( 'self', state.bindings )
		self.assertEqual( state.bindings['self'].state, cfg.OwnState.BORROWED )

# --- assign: fresh / alias / replace ------------------------------------------

class AssignTests( CFGTestBase ):

	def setUp( self ) -> None:
		super().setUp()
		self._import( 'class Foo: pass\ndef foo() -> None:\n\tpass\n' )
		self.foo_cls = self._module.get_local( 'Foo' )
		self.state = self._state( self._fn( 'foo' ))

	def _local( self, stem: str ):
		return Variable( stem = stem, qualname = f'foo.{stem}', file = None, line = None, type = self.foo_cls )

	def test_fresh_assign_pushes_entry_no_incref( self ) -> None:
		x = self._local( 'x' )
		instrs = self.state.assign( x, self._new_temp( self.foo_cls ), is_alias = False )
		self.assertEqual( instrs, [] )
		self.assertEqual( self.state.bindings['x'].state, cfg.OwnState.OWNED )
		self.assertEqual( len( self.state._epilogue_stack ), 1 )

	def test_aliasing_assign_increfs( self ) -> None:
		s = self._local( 's' )
		self.state.assign( s, self._new_temp( self.foo_cls ), is_alias = False )
		s2 = self._local( 's2' )
		instrs = self.state.assign( s2, s, is_alias = True )
		self.assertEqual( self._kinds( instrs ), ['Incref'] )
		self.assertIs( instrs[0].value, s )
		self.assertEqual( len( self.state._epilogue_stack ), 2 )

	def test_replace_decrefs_old_value_reuses_entry( self ) -> None:
		x = self._local( 'x' )
		self.state.assign( x, self._new_temp( self.foo_cls ), is_alias = False )
		instrs = self.state.assign( x, self._new_temp( self.foo_cls ), is_alias = False )
		self.assertEqual( self._kinds( instrs ), ['Decref'] )
		self.assertIs( instrs[0].value, x )
		self.assertEqual( len( self.state._epilogue_stack ), 1 ) # reused, not a second entry

	def test_non_rc_dest_is_a_noop( self ) -> None:
		i32 = self.discovery.get_intrinsics()['i32']
		x = Variable( stem = 'x', qualname = 'foo.x', file = None, line = None, type = i32 )
		instrs = self.state.assign( x, ir.Const( type = i32, value = 1 ), is_alias = False )
		self.assertEqual( instrs, [] )
		self.assertNotIn( 'x', self.state.bindings )

	def test_delete_temp_decrefs_a_tracked_fresh_temp( self ) -> None:
		t = self._new_temp( self.foo_cls )
		self.state.fresh_temp( t, self.foo_cls )
		instrs = self.state.delete_temp( t )
		self.assertEqual( self._kinds( instrs ), ['Decref'] )
		self.assertIs( instrs[0].value, t )
		self.assertEqual( self.state.delete_temp( t ), [] ) # already consumed - no double decref

	def test_assign_from_fresh_temp_untracks_it( self ) -> None:
		t = self._new_temp( self.foo_cls )
		self.state.fresh_temp( t, self.foo_cls )
		x = self._local( 'x' )
		self.state.assign( x, t, is_alias = False )
		self.assertEqual( self.state.delete_temp( t ), [] ) # ownership transferred into x, not a second owner

# --- struct/union field construction (Allocate) -------------------------------

class FieldValueTests( CFGTestBase ):

	def setUp( self ) -> None:
		super().setUp()
		self._import( 'class Foo: pass\ndef foo() -> None:\n\tpass\n' )
		self.foo_cls = self._module.get_local( 'Foo' )
		self.state = self._state( self._fn( 'foo' ))

	def _local( self, stem: str ):
		return Variable( stem = stem, qualname = f'foo.{stem}', file = None, line = None, type = self.foo_cls )

	def test_aliasing_field_value_increfs( self ) -> None:
		x = self._local( 'x' )
		self.state.assign( x, self._new_temp( self.foo_cls ), is_alias = False )
		instrs = self.state.field_value( self.foo_cls, x, is_alias = True )
		self.assertEqual( self._kinds( instrs ), ['Incref'] )
		self.assertIs( instrs[0].value, x )
		# stateless - doesn't touch bindings/the epilogue stack
		self.assertEqual( len( self.state._epilogue_stack ), 1 )
		self.assertEqual( self.state.bindings['x'].state, cfg.OwnState.OWNED )

	def test_fresh_field_value_is_a_noop( self ) -> None:
		t = self._new_temp( self.foo_cls )
		instrs = self.state.field_value( self.foo_cls, t, is_alias = False )
		self.assertEqual( instrs, [] )

	def test_fresh_field_value_untracks_the_temp( self ) -> None:
		# ownership transfers into the field, not a second independent
		# owner - mirrors assign()'s own Temp-untracking branch. Without
		# this, a fresh_temp()-registered temp embedded into a field would
		# still look "pending" to its own statement's DeleteTemp, decref'ing
		# the very value just handed into the field
		t = self._new_temp( self.foo_cls )
		self.state.fresh_temp( t, self.foo_cls )
		self.state.field_value( self.foo_cls, t, is_alias = False )
		self.assertEqual( self.state.delete_temp( t ), [] )

	def test_non_rc_field_value_is_a_noop_even_if_aliasing( self ) -> None:
		i32 = self.discovery.get_intrinsics()['i32']
		x = Variable( stem = 'x', qualname = 'foo.x', file = None, line = None, type = i32 )
		instrs = self.state.field_value( i32, x, is_alias = True )
		self.assertEqual( instrs, [] )

	def test_borrowed_source_embedded_in_field_still_increfs( self ) -> None:
		# mirrors Result.Ok(val)'s own body embedding a plain (BORROWED)
		# parameter into ResultPayload(ok=val) - field_value() doesn't
		# require OWNED/COPY like move() does, since embedding is meant to
		# behave like an ordinary aliasing read (an independent Incref),
		# not an ownership transfer
		self._import( 'class Foo: pass\ndef foo( x: Foo ) -> None:\n\tpass\n' )
		state = self._state( self._fn( 'foo' ))
		param = self._fn( 'foo' ).parameters[0]
		self.assertEqual( state.bindings['x'].state, cfg.OwnState.BORROWED )
		instrs = state.field_value( param.type, param, is_alias = True )
		self.assertEqual( self._kinds( instrs ), ['Incref'] )

# --- move[T] call arguments ------------------------------------------------

class MoveTests( CFGTestBase ):

	def setUp( self ) -> None:
		super().setUp()
		self._import( '''
class Foo: pass

def takeown( x: Foo ) -> None:
	pass
''' )
		self.foo_cls = self._module.get_local( 'Foo' )
		self.state = self._state( self._fn( 'takeown' ))

	def _local( self, stem: str ):
		return Variable( stem = stem, qualname = f'foo.{stem}', file = None, line = None, type = self.foo_cls )

	def test_move_owned_binding_cancels_its_entry_no_instructions( self ) -> None:
		x = self._local( 'x' )
		self.state.assign( x, self._new_temp( self.foo_cls ), is_alias = False )
		instrs = self.state.move( x, target_qualname = 'takeown', param_stem = 'p' )
		self.assertEqual( instrs, [] )
		self.assertEqual( self.state.bindings['x'].state, cfg.OwnState.MOVED )
		self.assertTrue( self.state._epilogue_stack[0].cancelled )

	def test_move_borrowed_is_a_compile_error( self ) -> None:
		param = self._fn( 'takeown' ).parameters[0] # BORROWED
		with self.assertRaises( CompileError ) as ctx:
			self.state.move( param, target_qualname = 'takeown', param_stem = 'p' )
		self.assertIn( 'not owned', str( ctx.exception ))

	def test_move_already_moved_is_a_compile_error( self ) -> None:
		x = self._local( 'x' )
		self.state.assign( x, self._new_temp( self.foo_cls ), is_alias = False )
		self.state.move( x, target_qualname = 'takeown', param_stem = 'p' )
		with self.assertRaises( CompileError ) as ctx:
			self.state.move( x, target_qualname = 'takeown', param_stem = 'p' )
		self.assertIn( 'not owned', str( ctx.exception ))

	def test_move_non_rc_operand_is_a_noop( self ) -> None:
		i32 = self.discovery.get_intrinsics()['i32']
		x = Variable( stem = 'n', qualname = 'foo.n', file = None, line = None, type = i32 )
		instrs = self.state.move( x, target_qualname = 'takeown', param_stem = 'p' )
		self.assertEqual( instrs, [] )

# --- del x -------------------------------------------------------------------

class DeletedTests( CFGTestBase ):

	def setUp( self ) -> None:
		super().setUp()
		self._import( '''
class Foo: pass

def foo( x: Foo ) -> None:
	pass
''' )
		self.foo_cls = self._module.get_local( 'Foo' )
		self.state = self._state( self._fn( 'foo' ))

	def test_del_owned_decrefs_and_removes( self ) -> None:
		x = Variable( stem = 'y', qualname = 'foo.y', file = None, line = None, type = self.foo_cls )
		self.state.assign( x, self._new_temp( self.foo_cls ), is_alias = False )
		instrs = self.state.deleted( x )
		self.assertEqual( self._kinds( instrs ), ['Decref'] )
		self.assertIs( instrs[0].value, x )
		self.assertNotIn( 'y', self.state.bindings )
		self.assertTrue( self.state._epilogue_stack[0].cancelled )

	def test_del_borrowed_is_a_noop( self ) -> None:
		param = self._fn( 'foo' ).parameters[0]
		instrs = self.state.deleted( param )
		self.assertEqual( instrs, [] )
		self.assertNotIn( 'x', self.state.bindings )

# --- IF/ELSE/ENDIF -----------------------------------------------------------

class IfMergeTests( CFGTestBase ):

	def setUp( self ) -> None:
		super().setUp()
		self._import( '''
class Foo: pass

def takeown( x: move[Foo] ) -> None:
	pass

def foo() -> None:
	pass
''' )
		self.foo_cls = self._module.get_local( 'Foo' )
		self.state = self._state( self._fn( 'foo' ))

	def _local( self, stem: str ):
		return Variable( stem = stem, qualname = f'foo.{stem}', file = None, line = None, type = self.foo_cls )

	def test_foo1_moved_on_true_branch_only_is_indeterminate( self ) -> None:
		# s: copy[Foo] entering the if, takeown(move(s)) inside the true
		# branch only (no else) - the spec's own foo1 example
		s = self._local( 's' )
		self.state.assign( s, self._new_temp( self.foo_cls ), is_alias = False )
		entry = self.state.snapshot()
		self.state.move( s, target_qualname = 'takeown', param_stem = 'x' )
		true_end = dict( self.state.bindings )
		false_end = dict( entry.bindings ) # no else
		with self.assertRaises( CompileError ) as ctx:
			self.state.merge_if( entry.bindings, true_end, false_end, 'foo' )
		self.assertIn( 'indeterminate state', str( ctx.exception ))

	def test_both_branches_construct_same_name_survives_the_merge( self ) -> None:
		# your explicit request: if cond: x = Foo() else: x = Bar(); use(x)
		# afterward - both branches independently push their OWN entry for
		# x, only one should survive onto the real stack after ENDIF
		entry = self.state.snapshot()
		x_true = self._local( 'x' )
		self.state.assign( x_true, self._new_temp( self.foo_cls ), is_alias = False )
		true_end = dict( self.state.bindings )
		self.state.restore( entry )
		x_false = self._local( 'x' ) # a distinct object, same stem - mirrors two independently-lowered branches both declaring `x`
		self.state.assign( x_false, self._new_temp( self.foo_cls ), is_alias = False )
		false_end = dict( self.state.bindings )
		self.state.restore( entry )
		true_instrs, false_instrs, removed = self.state.merge_if( entry.bindings, true_end, false_end, 'foo' )
		self.assertEqual( true_instrs, [] )
		self.assertEqual( false_instrs, [] )
		self.assertEqual( removed, [] )
		self.assertIn( 'x', self.state.bindings )
		self.assertEqual( self.state.bindings['x'].state, cfg.OwnState.OWNED )
		self.assertEqual( len( self.state._epilogue_stack ), 1 ) # exactly one surviving entry, not two

	def test_local_confined_to_one_branch_is_torn_down_at_endif( self ) -> None:
		# if cond: z = Foo() (no else, z never used again) - fine per your
		# clarification, torn down right here with no error. Its decref
		# must land in the TRUE branch's own instructions specifically (it
		# only exists on that path), never the false branch's
		entry = self.state.snapshot()
		z = self._local( 'z' )
		self.state.assign( z, self._new_temp( self.foo_cls ), is_alias = False )
		true_end = dict( self.state.bindings )
		false_end = dict( entry.bindings )
		true_instrs, false_instrs, removed = self.state.merge_if( entry.bindings, true_end, false_end, 'foo' )
		self.assertEqual( self._kinds( true_instrs ), ['Decref'] )
		self.assertIs( true_instrs[0].value, z )
		self.assertEqual( false_instrs, [] )
		self.assertEqual( removed, ['z'] )

	def test_preexisting_binding_confined_to_one_branch_is_an_error( self ) -> None:
		# a binding that existed BEFORE the if can't just vanish on one
		# branch only (moved/deleted there) while surviving on the other -
		# same indeterminate-scope problem as a state mismatch
		s = self._local( 's' )
		self.state.assign( s, self._new_temp( self.foo_cls ), is_alias = False )
		entry = self.state.snapshot()
		self.state.deleted( s )
		true_end = dict( self.state.bindings )
		false_end = dict( entry.bindings ) # s still alive here
		with self.assertRaises( CompileError ) as ctx:
			self.state.merge_if( entry.bindings, true_end, false_end, 'foo' )
		self.assertIn( "only one branch", str( ctx.exception ))

	def test_preexisting_local_untouched_by_both_branches_is_not_duplicated( self ) -> None:
		# the actual bug: a local declared BEFORE the if, left completely
		# untouched by both branches - restore() already puts its one real
		# entry back on the stack before merge_if() ever runs; the old
		# unconditional self._push() here duplicated it, producing a real
		# double-decref at whatever exit ran next (the single most common
		# if/else shape there is)
		y = self._local( 'y' )
		self.state.assign( y, self._new_temp( self.foo_cls ), is_alias = False )
		entry = self.state.snapshot()
		true_end = dict( self.state.bindings ) # untouched
		false_end = dict( self.state.bindings ) # untouched
		true_instrs, false_instrs, removed = self.state.merge_if( entry.bindings, true_end, false_end, 'foo' )
		self.assertEqual( true_instrs, [] )
		self.assertEqual( false_instrs, [] )
		self.assertEqual( removed, [] )
		self.assertEqual( len( self.state._epilogue_stack ), 1 ) # still exactly one entry, not two
		self.assertIs( self.state.bindings['y'].entry, entry.bindings['y'].entry )

	def test_true_terminates_false_survives_no_comparison_needed( self ) -> None:
		# if cond: takeown(move(s)); return  (no else) - s is MOVED on the
		# terminating true branch but still OWNED on the surviving false
		# branch. Must NOT be treated as foo1's indeterminate-state
		# mismatch - the true branch never reaches the join at all, so
		# only the false branch's own ending state matters
		s = self._local( 's' )
		self.state.assign( s, self._new_temp( self.foo_cls ), is_alias = False )
		entry = self.state.snapshot()
		self.state.move( s, target_qualname = 'takeown', param_stem = 'x' )
		true_end = dict( self.state.bindings )
		self.state.restore( entry )
		false_end = dict( entry.bindings ) # no else - untouched
		true_instrs, false_instrs, removed = self.state.merge_if(
			entry.bindings, true_end, false_end, 'foo', true_terminates = True,
		)
		self.assertEqual( true_instrs, [] )
		self.assertEqual( false_instrs, [] )
		self.assertEqual( removed, [] )
		self.assertEqual( self.state.bindings['s'].state, cfg.OwnState.OWNED ) # false branch's own state wins
		self.assertEqual( len( self.state._epilogue_stack ), 1 ) # reused, not duplicated
		self.assertIs( self.state.bindings['s'].entry, entry.bindings['s'].entry )

	def test_both_branches_terminate_nothing_survives( self ) -> None:
		true_instrs, false_instrs, removed = self.state.merge_if(
			{}, {}, {}, 'foo', true_terminates = True, false_terminates = True,
		)
		self.assertEqual( true_instrs, [] )
		self.assertEqual( false_instrs, [] )
		self.assertEqual( removed, [] )

# --- loops ---------------------------------------------------------------

class LoopTests( CFGTestBase ):

	def setUp( self ) -> None:
		super().setUp()
		self._import( '''
class Foo: pass

def takeown( x: move[Foo] ) -> None:
	pass

def foo( s: copy[Foo] ) -> None:
	pass
''' )
		self.foo_cls = self._module.get_local( 'Foo' )
		self.state = self._state( self._fn( 'foo' ))

	def _local( self, stem: str ):
		return Variable( stem = stem, qualname = f'foo.{stem}', file = None, line = None, type = self.foo_cls )

	def test_foo3_persisting_local_moved_inside_loop_is_indeterminate( self ) -> None:
		entry = self.state.snapshot() # s: COPY
		s = self.state.bindings['s'].operand
		self.state.move( s, target_qualname = 'takeown', param_stem = 'x' ) # takeown(move(s)) inside the while body
		with self.assertRaises( CompileError ) as ctx:
			self.state.loop_back_edge( entry.bindings, 'foo' )
		self.assertIn( 'indeterminate state', str( ctx.exception ))

	def test_foo4_loop_scoped_local_reassigned_and_moved_each_iteration_is_fine( self ) -> None:
		entry = self.state.snapshot()
		s2 = self._local( 's2' )
		self.state.assign( s2, self._new_temp( self.foo_cls ), is_alias = False ) # for s2 in s.split(','):
		self.state.move( s2, target_qualname = 'takeown', param_stem = 'x' ) # takeown(move(s2))
		instrs = self.state.loop_back_edge( entry.bindings, 'foo' )
		self.assertEqual( instrs, [] ) # nothing to decref - moved away before the iteration ends

	def test_loop_scoped_local_never_moved_is_torn_down_every_iteration( self ) -> None:
		entry = self.state.snapshot()
		y = self._local( 'y' )
		self.state.assign( y, self._new_temp( self.foo_cls ), is_alias = False ) # y = Foo() inside the loop body, never referenced after
		instrs = self.state.loop_back_edge( entry.bindings, 'foo' )
		self.assertEqual( self._kinds( instrs ), ['Decref'] )
		self.assertIs( instrs[0].value, y )

	def test_stable_reassignment_across_iterations_is_fine( self ) -> None:
		# x declared before the loop, replaced (not fresh-created) every
		# iteration - state stays OWNED both entering and at the back edge
		x = self._local( 'x' )
		self.state.assign( x, self._new_temp( self.foo_cls ), is_alias = False )
		entry = self.state.snapshot()
		replace_instrs = self.state.assign( x, self._new_temp( self.foo_cls ), is_alias = False ) # x = Foo() again, inside the loop body
		self.assertEqual( self._kinds( replace_instrs ), ['Decref'] ) # the replace's own inline decref-old-value
		back_edge_instrs = self.state.loop_back_edge( entry.bindings, 'foo' )
		self.assertEqual( back_edge_instrs, [] ) # stable - no loop-scoped teardown, no mismatch
		self.assertEqual( self.state.bindings['x'].state, cfg.OwnState.OWNED )

# --- break/continue -----------------------------------------------------

class UnwindToTests( CFGTestBase ):

	def test_unwind_to_loop_entry_stops_at_loop_depth( self ) -> None:
		self._import( 'class Foo: pass\ndef foo() -> None:\n\tpass\n' )
		foo_cls = self._module.get_local( 'Foo' )
		state = self._state( self._fn( 'foo' ))
		y = Variable( stem = 'y', qualname = 'foo.y', file = None, line = None, type = foo_cls )
		state.assign( y, self._new_temp( foo_cls ), is_alias = False ) # pushed BEFORE the loop - must survive a break inside it
		loop_entry = state.snapshot()
		z = Variable( stem = 'z', qualname = 'foo.z', file = None, line = None, type = foo_cls )
		state.assign( z, self._new_temp( foo_cls ), is_alias = False ) # pushed INSIDE the loop body
		instrs = state.unwind_to( loop_entry )
		self.assertEqual( self._kinds( instrs ), ['Decref'] )
		self.assertIs( instrs[0].value, z ) # only z (loop-scoped) unwinds on break - y (outer) stays untouched

# --- return / fall-off-the-end -----------------------------------------------

class ReturnTests( CFGTestBase ):

	def setUp( self ) -> None:
		super().setUp()
		self._import( 'class Foo: pass\ndef foo() -> None:\n\tpass\n' )
		self.foo_cls = self._module.get_local( 'Foo' )
		self.state = self._state( self._fn( 'foo' ))

	def _local( self, stem: str ):
		return Variable( stem = stem, qualname = f'foo.{stem}', file = None, line = None, type = self.foo_cls )

	def test_return_excludes_the_returned_binding( self ) -> None:
		x = self._local( 'x' )
		self.state.assign( x, self._new_temp( self.foo_cls ), is_alias = False )
		y = self._local( 'y' )
		self.state.assign( y, self._new_temp( self.foo_cls ), is_alias = False )
		instrs = self.state.return_( x )
		self.assertEqual( self._kinds( instrs ), ['Decref'] )
		self.assertIs( instrs[0].value, y )

	def test_returning_a_bare_fresh_temp_untracks_it( self ) -> None:
		# return SomeClass() - never assigned to a name, so DeleteTemp would
		# otherwise (wrongly) decref the exact value we just handed to the
		# caller
		t = self._new_temp( self.foo_cls )
		self.state.fresh_temp( t, self.foo_cls )
		instrs = self.state.return_( t )
		self.assertEqual( instrs, [] )
		self.assertEqual( self.state.delete_temp( t ), [] ) # no double decref

	def test_multiple_returns_get_independent_decref_sets( self ) -> None:
		x = self._local( 'x' )
		self.state.assign( x, self._new_temp( self.foo_cls ), is_alias = False )
		early = self.state.return_( None )
		self.assertEqual( self._kinds( early ), ['Decref'] )
		y = self._local( 'y' )
		self.state.assign( y, self._new_temp( self.foo_cls ), is_alias = False )
		late = self.state.return_( None )
		self.assertEqual( self._kinds( late ), ['Decref', 'Decref'] )

	def test_moved_binding_is_skipped_by_return( self ) -> None:
		self._import( '''
class Foo: pass

def takeown( x: move[Foo] ) -> None:
	pass

def foo() -> None:
	pass
''' )
		foo_cls = self._module.get_local( 'Foo' )
		state = self._state( self._fn( 'foo' ))
		x = Variable( stem = 'x', qualname = 'foo.x', file = None, line = None, type = foo_cls )
		state.assign( x, self._new_temp( foo_cls ), is_alias = False )
		state.move( x, target_qualname = 'takeown', param_stem = 'p' )
		self.assertEqual( state.return_( None ), [] )

# --- self construction (__init__) -----------------------------------------

class ConstructionTests( CFGTestBase ):

	def setUp( self ) -> None:
		super().setUp()
		self._import( '''
class Foo: pass

class Bar:
	a: Foo
	b: Foo
	n: i32

def bar( self_obj: Bar ) -> None:
	pass
''' )
		self.bar_cls = self._module.get_local( 'Bar' )
		if self.bar_cls.resolve is not None:
			self.bar_cls.resolve()
		for attr in self.bar_cls.attributes:
			if attr.resolve is not None:
				attr.resolve()
		self.a_attr, self.b_attr, self.n_attr = self.bar_cls.attributes
		self.state = self._state( self._fn( 'bar' ))
		self.self_param = self._fn( 'bar' ).parameters[0]
		self.state.enter_construction( self.self_param, self.bar_cls.attributes )

	def test_attr_assign_fresh_rc_pushes_entry_no_incref( self ) -> None:
		instrs = self.state.attr_assign( self.a_attr, self._new_temp( self.a_attr.type ), is_alias = False )
		self.assertEqual( instrs, [] )
		self.assertIn( 'self.a', self.state.bindings )
		self.assertEqual( len( self.state._epilogue_stack ), 1 )

	def test_attr_assign_aliasing_increfs( self ) -> None:
		# a's own fresh value aliased into b (two DIFFERENT attributes, no
		# prior binding for b) - a fresh assign, just with is_alias=True
		src = self._new_temp( self.a_attr.type )
		self.state.attr_assign( self.a_attr, src, is_alias = False )
		instrs = self.state.attr_assign( self.b_attr, src, is_alias = True )
		self.assertEqual( self._kinds( instrs ), ['Incref'] )
		self.assertIs( instrs[0].value, src )

	def test_attr_assign_replace_decrefs_old_reuses_entry( self ) -> None:
		self.state.attr_assign( self.a_attr, self._new_temp( self.a_attr.type ), is_alias = False )
		instrs = self.state.attr_assign( self.a_attr, self._new_temp( self.a_attr.type ), is_alias = False )
		self.assertEqual( self._kinds( instrs ), ['Decref'] )
		self.assertEqual( len( self.state._epilogue_stack ), 1 ) # reused, not a second entry

	def test_attr_assign_non_rc_is_tracked_with_no_instructions( self ) -> None:
		i32 = self.discovery.get_intrinsics()['i32']
		instrs = self.state.attr_assign( self.n_attr, ir.Const( type = i32, value = 1 ), is_alias = False )
		self.assertEqual( instrs, [] )
		self.assertIn( 'self.n', self.state.bindings )
		self.assertEqual( len( self.state._epilogue_stack ), 0 ) # no entry - nothing to decref, ever

	def test_complete_construction_succeeds_and_cancels_entries_without_decref( self ) -> None:
		self.state.attr_assign( self.a_attr, self._new_temp( self.a_attr.type ), is_alias = False )
		self.state.attr_assign( self.b_attr, self._new_temp( self.b_attr.type ), is_alias = False )
		self.state.attr_assign( self.n_attr, ir.Const( type = self.discovery.get_intrinsics()['i32'], value = 0 ), is_alias = False )
		self.state.complete_construction( 'Bar.__init__' )
		for entry in self.state._epilogue_stack:
			self.assertTrue( entry.cancelled )
		# the epilogue's own unwind (ordinary return_()) emits nothing for
		# the now-cancelled attribute entries - ownership transferred into
		# the now-complete self, no decref
		self.assertEqual( self.state.return_( None ), [] )

	def test_complete_construction_raises_when_an_attribute_is_missing( self ) -> None:
		self.state.attr_assign( self.a_attr, self._new_temp( self.a_attr.type ), is_alias = False )
		with self.assertRaises( CompileError ) as ctx:
			self.state.complete_construction( 'Bar.__init__' )
		self.assertIn( 'b', str( ctx.exception ))
		self.assertIn( 'n', str( ctx.exception ))

	def test_check_self_escape_is_a_noop_outside_construction( self ) -> None:
		self._import( 'class Foo: pass\ndef foo() -> None:\n\tpass\n' )
		state = self._state( self._fn( 'foo' ))
		state.check_self_escape( self._new_temp( self.bar_cls ), 'foo' ) # no raise - nothing under construction

	def test_check_self_escape_raises_before_all_attributes_initialized( self ) -> None:
		self.state.attr_assign( self.a_attr, self._new_temp( self.a_attr.type ), is_alias = False )
		with self.assertRaises( CompileError ) as ctx:
			self.state.check_self_escape( self.self_param, 'Bar.__init__' )
		self.assertIn( 'b', str( ctx.exception ))

	def test_check_self_escape_is_fine_for_a_different_operand( self ) -> None:
		other = self._new_temp( self.a_attr.type )
		self.state.check_self_escape( other, 'Bar.__init__' ) # not self - no raise regardless of construction state

	def test_check_self_escape_allows_self_once_all_attributes_initialized( self ) -> None:
		self.state.attr_assign( self.a_attr, self._new_temp( self.a_attr.type ), is_alias = False )
		self.state.attr_assign( self.b_attr, self._new_temp( self.b_attr.type ), is_alias = False )
		self.state.attr_assign( self.n_attr, ir.Const( type = self.discovery.get_intrinsics()['i32'], value = 0 ), is_alias = False )
		self.state.check_self_escape( self.self_param, 'Bar.__init__' ) # no raise - already complete, even before complete_construction() itself runs

	def test_attr_replace_increfs_new_and_decrefs_old( self ) -> None:
		old = self._new_temp( self.a_attr.type )
		new = self._new_temp( self.a_attr.type )
		instrs = self.state.attr_replace( self.a_attr.type, old, new, is_alias = True )
		self.assertEqual( self._kinds( instrs ), ['Incref', 'Decref'] )
		self.assertIs( instrs[0].value, new )
		self.assertIs( instrs[1].value, old )

	def test_attr_replace_fresh_new_value_only_decrefs_old( self ) -> None:
		old = self._new_temp( self.a_attr.type )
		new = self._new_temp( self.a_attr.type )
		instrs = self.state.attr_replace( self.a_attr.type, old, new, is_alias = False )
		self.assertEqual( self._kinds( instrs ), ['Decref'] )
		self.assertIs( instrs[0].value, old )

if __name__ == '__main__':
	unittest.main()
