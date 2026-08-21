# stdlib imports:
import ast
import copy
from contextlib import nullcontext
import queue
import threading

# local imports:
import compile_time_transformer
from discovery import Discovery
from errors import CompileError
from monomorphize import Monomorphizer
from mpy_types import (
	CallableType, CEnum, ClassLike, CStruct, CUnion, Function, GeneratorType, Module, Name, Overload,
	Parameter, RCClass, Scalar, Specialization, TaggedUnion, Type,
	TupleType, TypeVar, Variable,
)
import overload_resolution
from tuple_storage import TupleStorage
from union_storage import UnionStorage



def _union_member_ast_path( union: TaggedUnion, member_stem: str ) -> ast.Attribute:
	''' build an ast.Attribute path for a union's member reference in a
	match-case pattern, e.g. builtins.MaybeFoo.Some — the union's own
	qualname dotted then the member name. '''
	parts = union.qualname.rsplit( '.', 1 )
	if len( parts ) == 2:
		return ast.Attribute(
			value = ast.Attribute(
				value = ast.Name( id = parts[0], ctx = ast.Load() ),
				attr = parts[1], ctx = ast.Load(),
			),
			attr = member_stem, ctx = ast.Load(),
		)
	return ast.Attribute(
		value = ast.Name( id = parts[0], ctx = ast.Load() ),
		attr = member_stem, ctx = ast.Load(),
	)


def _id( name: str ) -> ast.Name:
	return ast.Name( id = name, ctx = ast.Load() )


def _expr_stmt( value: ast.expr ) -> ast.Expr:
	return ast.Expr( value = value )


def _build_field_teardown_ast( field_expr: ast.Attribute, field_type: Type ) -> list[ast.stmt]:
	''' recursively build AST statements to decref every RC leaf
	reachable from field_expr, given its declared type. '''
	base = field_type.base if isinstance( field_type, Specialization ) else field_type
	if isinstance( base, RCClass ):
		return [ _expr_stmt( ast.Call(
			func = ast.Attribute( value = _id('compiler'), attr = 'decref', ctx = ast.Load() ),
			args = [ field_expr ], keywords = [],
		)) ]
	if isinstance( base, CStruct ):
		stmts: list[ast.stmt] = []
		for attr in base.attributes:
			sub_expr = ast.Attribute( value = field_expr, attr = attr.stem, ctx = ast.Load() )
			stmts.extend( _build_field_teardown_ast( sub_expr, attr.type ))
		return stmts
	if isinstance( base, TaggedUnion ):
		cases: list[ast.match_case] = []
		for member in base.attributes:
			member_base = member.type.base if isinstance( member.type, Specialization ) else member.type
			if isinstance( member_base, RCClass ):
				bind_name = f'__dtor_{member.stem}'
				cases.append( ast.match_case(
					pattern = ast.MatchClass(
						cls = _union_member_ast_path( base, member.stem ),
						patterns = [ ast.MatchAs( name = bind_name ) ],
						kwd_attrs = [], kwd_patterns = [],
					),
					guard = None,
					body = [ _expr_stmt( ast.Call(
						func = ast.Attribute( value = _id('compiler'), attr = 'decref', ctx = ast.Load() ),
						args = [ _id( bind_name ) ], keywords = [],
					)) ],
				))
			else:
				cases.append( ast.match_case(
					pattern = ast.MatchClass(
						cls = _union_member_ast_path( base, member.stem ),
						patterns = [ ast.MatchAs( name = None ) ],
						kwd_attrs = [], kwd_patterns = [],
					),
					guard = None,
					body = [ ast.Pass() ],
				))
		return [ ast.Match( subject = field_expr, cases = cases ) ]
	return []


class _GeneratorNameRenamer( ast.NodeTransformer ):
	''' PLAN_GENERATORS.md - rewrites every ast.Name(id=stem) reference
	(Load or Store) for a promoted stem (a generator's own parameter or
	local - see TypeResolver._collect_generator_locals) into self.<stem>,
	since each is now a field on the backing class rather than a stack
	local/parameter of $$__next__ itself. Does not recurse into a nested
	def/lambda (matches PLAN_LAMBDA.md's own scope boundary - a nested
	def/lambda referencing one of these names was never a supported
	capture to begin with; leaving it untouched means it fails name
	resolution or the existing no-capture check normally, not silently
	miscompiled). '''
	def __init__( self, targets: set ) -> None:
		self.targets = targets
	def visit_FunctionDef( self, node ): return node
	def visit_AsyncFunctionDef( self, node ): return node
	def visit_Lambda( self, node ): return node
	def visit_Name( self, node ):
		if node.id in self.targets:
			inner = ast.Name( id = 'self', ctx = ast.Load() )
			ast.copy_location( inner, node )
			attr = ast.Attribute( value = inner, attr = node.id, ctx = node.ctx )
			ast.copy_location( attr, node )
			return attr
		return node
	def visit_AnnAssign( self, node ):
		# a generator local's OWN declaring `x: T = expr` becomes `self.x =
		# expr` (an ordinary Assign, not AnnAssign - self.x's type already
		# lives on the backing class's own field declaration, and this
		# compiler's AnnAssign lowering doesn't support an Attribute target
		# at all, confirmed via a real repro: "unsupported AnnAssign
		# target: self.x: i32 = 1"). Every AnnAssign inside a generator
		# body targets a promoted local by construction (TypeResolver.
		# _collect_generator_locals requires exactly this shape), so the
		# target is always renamed to self.<x> here - no need to check
		# self.targets first. A bare `x: T` with no value (rare, and inert
		# in real Python too) becomes a no-op Pass rather than an invalid
		# valueless Assign.
		target = self.visit( node.target )
		if node.value is None:
			result = ast.Pass()
		else:
			result = ast.Assign( targets = [ target ], value = self.visit( node.value ))
		ast.copy_location( result, node )
		return result


class TypeResolver:
	'''
	stage 1.5: sits between discovery.py (lazy name-binding + skeleton type
	registry) and lowering.py (pure IR emission). Owns the reachable-from-
	main work queue - previously Compiler._enqueue/.queue/._seen - and the
	shared UnionStorage/Monomorphizer instances - previously constructed
	inside Lowering.__init__ (both already depended on nothing but Discovery
	and an injected `schedule` callback, so moving them here is a pure
	relocation, not a rewrite - see union_storage.py/monomorphize.py's own
	docstrings, and monomorphize_test.py's "never lowering.py itself" test
	discipline, which this class follows too).

	`schedule(...)` is still "the single place that decides what a
	dependency actually means" (see its own docstring below, moved verbatim
	from Compiler._enqueue) - lowering.py hands it anything it comes across
	unconditionally, same as always. Compiler still owns the per-kind
	lowering dispatch and the compiled-object collections (functions/
	rcclasses/...) - this class only decides what's reachable and makes
	sure it's resolved by the time something asks for it.
	'''
	def __init__( self, discovery: Discovery ) -> None:
		self.discovery = discovery
		self.queue: queue.Queue = queue.Queue()
		self._seen: set[int] = set()
		self._seen_lock = threading.Lock()
		self.union_storage = UnionStorage( discovery, self.schedule )
		# constructed before monomorphizer (below), which now depends on it -
		# substitute_type_params needs to resolve a bare TupleType bound to a
		# generic class's own type param the same way it already eagerly
		# monomorphizes a bare Specialization (see monomorphize.py's own
		# comment on why)
		self.tuple_storage = TupleStorage( discovery, self.schedule ) # PLAN_TUPLE.md - same "depends on nothing but Discovery + schedule" shape as union_storage above
		self.monomorphizer = Monomorphizer( discovery, self.schedule, self.union_storage, self.tuple_storage )
		# keyed by id(fn.node), not id(fn) - the SAME shared AST body object
		# is reused by every monomorphized copy of a generic function (see
		# resolve_function_body's own docstring)
		self._body_resolved: set[int] = set()
		# provenance tracking for --dep-report: id(unit) -> qualname of
		# the unit being lowered at the time this one was scheduled
		self._triggered_by: dict[int,str] = {}
		self._current_trigger: str|None = None
		# the emitter synthesizes a destructor body for every non-generic
		# RCClass — see _synthesize_rcclass_destructor. Schedule sys.free
		# once, lazily, the first time an RCClass actually needs one
		self._sys_free_scheduled: bool = False
		self._destructors_synthesized: set[int] = set()
		self._constructors_synthesized: set[int] = set() # id(RCClass) -> $$__new__ already synthesized - see _synthesize_rcclass_constructor
		self._dtor_label_id = 0
		self._sys_functions: dict[str,Function] = {}
		# re-entrancy guard for _schedule_uniontype_storage: union_storage.
		# get(union) itself calls schedule() on the union's own attributes
		# (UnionStorage._ensure_resolved), which for a bare TaggedUnion
		# attribute routes straight back into _schedule_uniontype_storage -
		# without this, that's unbounded recursion for any union with an
		# RC-typed member (including Result itself, Ok: T/Err: E), since
		# get()'s own memoization cache isn't populated until get() finishes
		self._union_storage_scheduling: set[int] = set()
		# PLAN_GENERATORS.md - id(fn) of every generator function already
		# rewritten by ensure_generator_synthesized (keyed by id(fn), not
		# id(fn.node): unlike a generic function's body, a generator is
		# never monomorphized, so fn.node is never copied for it - this
		# still mirrors _destructors_synthesized's own "idempotent, once
		# per real object" spirit)
		self._generators_synthesized: set[int] = set()
		# PLAN_GENERATORS.md Phase 1 - unique per-desugared-for-loop suffix
		# for __for_obj_N/__for_len_N/__for_index_N field names, global
		# across every generator function this TypeResolver ever processes
		# (never reset per-function) so two different generators never
		# collide even though each gets its own backing RCClass anyway -
		# simplest way to guarantee uniqueness without threading a fresh
		# counter through every desugaring call site
		self._for_desugar_counter = 0
		# PLAN_GENERATORS.md Phase C - unique __gen_send_capture_N suffix
		# for _hoist_yield_from_rc_reassignment's own synthesized capture
		# temp, same "global across every generator function, never reset
		# per-function" reasoning as _for_desugar_counter just above
		self._gen_send_capture_counter = 0

	def _ensure_sys_free_scheduled( self ) -> None:
		if self._sys_free_scheduled:
			return
		self._sys_free_scheduled = True
		# only meaningful when builtins (and therefore sys) are actually
		# loaded — without builtins there's no sys.free to schedule
		module = self.discovery.modules.get( 'sys' )
		if module is None:
			return
		free_fn = module.get_local( 'free' )
		assert isinstance( free_fn, Function ), f'sys.free is required but was not found: {free_fn!r}'
		self.schedule( free_fn )

	# --- generator functions (PLAN_GENERATORS.md) ---------------------------
	#
	# v1 scope: a plain (non-generic, non-method) function containing at
	# least one `yield`, where every `yield` is a direct top-level statement
	# of the function's own body (not nested inside if/while/for/with/try) -
	# this is what lets the whole state machine be expressed as a flat
	# sequence of `if self.__state <= i:` guards (below) instead of needing
	# real goto/switch machinery: each guard is just an ordinary, already-
	# correct nested CFG scope, so cfg.py's epilogue/RC tracking (lowering.py/
	# cfg.py) needs no changes at all. Every local assigned anywhere in the
	# body is unconditionally promoted to a field on a synthesized backing
	# RCClass (over-promotion is safe, avoids a separate liveness pass - see
	# _collect_generator_locals), restricted for now to scalar types only
	# (an RC-typed LOCAL surviving a yield would need a state-gated
	# destructor cascade this pass doesn't build). A captured PARAMETER has
	# no such restriction: it's valid from construction onward
	# unconditionally, so the ORDINARY, unmodified $$__destructor__
	# synthesis (_synthesize_rcclass_destructor below, wired in
	# automatically via compiler.py's own RCClass scheduling once the
	# backing class is scheduled) already decrefs it correctly on drop, mid-
	# iteration or not - this IS the "epilogue moves into __del__" mechanism
	# the plan doc describes, just scoped to parameters in this first pass
	# rather than every promoted local.
	#
	# Runs eagerly, from ensure_resolved (not lower_function/lowering.py) -
	# a call site needs a generator function's REAL return type (the
	# synthesized backing class) the moment it resolves that function as a
	# callee, which can happen long before - or never relative to - that
	# function's own turn on the work queue (see ensure_resolved's own
	# comment on this).

	def _walk_generator_body( self, nodes ):
		''' yields every node reachable from `nodes` in program (pre-)order -
		parent before children, siblings left to right - NOT recursing into
		a nested def/lambda/async def (PLAN_LAMBDA.md's own scope boundary -
		a nested def/lambda containing yield is a separate, unrelated
		generator; one referencing a name from here is already rejected by
		that machinery's own no-capture check once it's lowered). Program
		order matters here, not just reachability: _collect_generator_locals
		relies on seeing a local's own declaring AnnAssign before any later
		statement that merely reassigns it (a plain stack/LIFO walk visits
		siblings in REVERSE order, which - confirmed by a real repro, `x: i32
		= 1` followed by two later `x = ...` reassignments - wrongly reports
		the reassignments as coming before the declaration). Shared by every
		generator-body scan below. '''
		for node in nodes:
			if isinstance( node, ( ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda )):
				continue
			yield node
			yield from self._walk_generator_body( ast.iter_child_nodes( node ))

	def _find_all_yield_nodes( self, fn: Function ) -> list[ast.expr]:
		return [ n for n in self._walk_generator_body( fn.node.body ) if isinstance( n, ( ast.Yield, ast.YieldFrom )) ]

	def _function_contains_yield( self, fn: Function ) -> bool:
		return any( isinstance( n, ( ast.Yield, ast.YieldFrom )) for n in self._walk_generator_body( fn.node.body ))

	def _self_attr( self, name: str, node: ast.AST ) -> ast.Attribute:
		inner = ast.Name( id = 'self', ctx = ast.Load() )
		ast.copy_location( inner, node )
		attr = ast.Attribute( value = inner, attr = name, ctx = ast.Load() )
		ast.copy_location( attr, node )
		return attr

	def _generator_defer_site_kind( self, node: ast.stmt ) -> 'str|None':
		''' 'defer'/'errdefer' if `node` is ITSELF a defer/errdefer site, in
		either syntax lowering.py's own _defer_kind_of_with/_defer_kind_of_
		call recognize: the `with defer:`/`with errdefer:` block form, or
		the single-statement `defer(expr)`/`errdefer(expr)` call form
		(always wrapped in an ast.Expr statement - _stmt_Expr's own
		handling). None for anything else, including a nested Name/Call
		reachable INSIDE a defer body that happens to reference the name
		`defer`/`errdefer` some other way - only the statement shape itself
		counts. '''
		if isinstance( node, ast.With ) and len( node.items ) == 1:
			expr = node.items[0].context_expr
			if isinstance( expr, ast.Name ) and expr.id in ( 'defer', 'errdefer' ):
				return expr.id
		if isinstance( node, ast.Expr ) and isinstance( node.value, ast.Call ) and isinstance( node.value.func, ast.Name ) and node.value.func.id in ( 'defer', 'errdefer' ):
			return node.value.func.id
		return None

	def _validate_generator_defer_sites( self, fn: Function ) -> None:
		''' PLAN_GENERATORS.md's defer/errdefer-in-generators phase - a
		`with defer:`/`with errdefer:`/`defer(...)`/`errdefer(...)` site is
		only allowed as a DIRECT TOP-LEVEL statement of `fn.node.body`
		itself (this stays true post-Phase F even though ordinary yield
		nesting/multiplicity restrictions were lifted - a deliberate,
		independent scope decision, not related to yield dispatch at
		all). Nested one level further in - a while/if/for/with's own
		body/branch - is rejected: "start narrow, no obviously-correct
		place to run a resumable loop's own per-iteration arming/replay
		yet" posture. '''
		top_level_ids = { id( s ) for s in fn.node.body }
		for node in self._walk_generator_body( fn.node.body ):
			kind = self._generator_defer_site_kind( node )
			if kind is None:
				continue
			if id( node ) not in top_level_ids:
				self.discovery.fail(
					f'{fn.qualname}: {kind} is only supported as a direct top-level statement of a generator '
					f'body (not nested inside a while/if/for/with) - see PLAN_GENERATORS.md',
					node,
				)

	def _capture_defer_site_body( self, node: ast.stmt ) -> list[ast.stmt]:
		''' the ORIGINAL (un-renamed, un-copied - callers deep-copy per
		insertion site themselves) statement list a defer/errdefer site's
		own body is made of: the `with defer: BODY` form's own node.body
		directly, or the single-statement `defer(expr)` call form's own
		lone argument, wrapped in an ast.Expr the same way lowering.py's
		own _stmt_Expr already decomposes that call form (_register_defer_
		block, ordinary non-generator defer) - so both syntaxes end up
		with an identical downstream shape here too. '''
		if isinstance( node, ast.With ):
			return node.body
		assert isinstance( node, ast.Expr ) and isinstance( node.value, ast.Call )
		single_stmt = ast.Expr( value = node.value.args[0] )
		ast.copy_location( single_stmt, node )
		return [ single_stmt ]

	def _reject_return_inside_generator_defer_body( self, fn: Function, body_stmts: list[ast.stmt] ) -> None:
		''' mirrors lowering.py's own _stmt_Return check for ordinary (non-
		generator) defer/errdefer bodies ("return is not allowed inside a
		defer/errdefer body") - that check is gated on self._in_deferred_
		body, which never gets set here: a generator's own defer/errdefer
		body never routes through _register_defer_block at all (Mechanism
		1's _build_defer_replay_guards and Mechanism 2's lowering.py hook
		both lower it via their own, separate AST-If-wrap + _lower_stmt
		technique), so nothing catches this today without an explicit
		check. Same reasoning applies: a generator defer body's own
		statements run later, replayed inline at an exit point (or_
		return() error, tail exhaustion, abandonment) - a `return` inside
		one would jump out of $$__next__ early, skipping any later-armed
		site's own replay and the flag-unset every replay guard relies on
		to avoid firing twice (see _build_defer_replay_guards' own
		docstring). Walks the WHOLE body (_walk_generator_body, not just
		top-level statements), so a return nested inside an if/while
		inside the defer body is caught too - same "stays set for the
		whole capture, not just the top-level statement" posture the
		ordinary check already documents. '''
		for node in self._walk_generator_body( body_stmts ):
			if isinstance( node, ast.Return ):
				self.discovery.fail( f'{fn.qualname}: return is not allowed inside a defer/errdefer body: {ast.unparse(node)}', node )

	def _desugar_generator_defer_sites( self, fn: Function ) -> list[tuple[str,bool,list[ast.stmt]]]:
		''' Mechanism 1 (PLAN_GENERATORS.md's defer/errdefer phase) - every
		top-level defer/errdefer site (already validated by _validate_
		generator_defer_sites to be exactly that - a direct top-level
		statement of fn.node.body, i.e. living in some preamble or the
		tail) gets a promoted boolean "armed" flag field, self.
		__defer_armed_N (N = this site's own index, 0-based, in the order
		found), and is replaced IN PLACE with `self.__defer_armed_N =
		True` - an ordinary assignment, transparent to every later pass
		(unit-collection already ran; locals-collection, segment-
		splitting, bare-return rewriting all still run AFTER this and see
		nothing but an ordinary Assign here).

		The site's own BODY is captured (still bare-named, not yet
		renamed - see _capture_defer_site_body) and returned alongside its
		flag field name and whether it's an errdefer, in PROGRAM order;
		callers needing LIFO replay (every insertion site does - see
		_build_defer_replay_guards) reverse this list themselves. Because
		arming is a FIELD, not a call-stack entry, this only has to run
		ONCE here regardless of how many separate places/how much later
		each site's own replay ends up firing (_build_generator_next_
		function's tail/bare-return replay, _build_generator_destructor's
		abandonment replay, or - for errdefer - lowering.py's OrReturn.
		epilogue hook, task #40/#41) - the field stays armed correctly
		across however many further $$__next__() calls happen first. '''
		sites: list[tuple[str,bool,list[ast.stmt]]] = []
		new_body: list[ast.stmt] = []
		for stmt in fn.node.body:
			kind = self._generator_defer_site_kind( stmt )
			if kind is None:
				new_body.append( stmt )
				continue
			flag_stem = f'__defer_armed_{len( sites )}'
			body_stmts = self._capture_defer_site_body( stmt )
			self._reject_return_inside_generator_defer_body( fn, body_stmts )
			sites.append( ( flag_stem, kind == 'errdefer', body_stmts ))
			arm = ast.Assign( targets = [ self._self_attr( flag_stem, stmt ) ], value = ast.Constant( value = True ))
			ast.copy_location( arm, stmt )
			new_body.append( arm )
		fn.node.body = new_body
		return sites

	def _build_defer_replay_guards( self, defer_sites: list[tuple[str,bool,list[ast.stmt]]], anchor: ast.AST ) -> list[ast.stmt]:
		''' LIFO `if self.__defer_armed_N: <deep-copied BODY>` for every
		ARMED, PLAIN `defer` site - `errdefer` sites are skipped entirely
		here (kind[1] True): they only ever replay via mechanism 2's
		OrReturn.epilogue hook (task #40/#41), never at a normal exit.
		Each site's own captured body is deep-copied FRESH per call - see
		_rename_and_track_liveness's own docstring for why sharing one
		node object across insertion sites is unsafe (lowering attaches
		mutable per-occurrence attributes like resolved_* that would
		corrupt a shared node).

		Callers still owe the result a rename pass: either explicitly, via
		_rename_and_track_liveness (the tail/destructor call sites, which
		have a renamer in hand already and want RC-local-reassignment-
		inside-a-defer-body to get the same live-flag-split treatment
		everything else gets), or implicitly, by embedding this raw result
		inside a still-to-be-renamed segment/tail list a LATER renamer.
		visit(...)/_rename_and_track_liveness call already covers end to
		end (_rewrite_bare_return_stmts's own call site - see its own
		comment for why that's safe: ast.NodeTransformer.generic_visit
		recurses into a nested If's own body/orelse automatically, no
		special-casing needed).

		Each guard ALSO unsets its own flag (self.__defer_armed_N = False)
		right after replaying the body: a plain `defer` armed at, say, a
		bare-return exit fires HERE, but the object itself often isn't
		actually freed until later (whatever reference the caller still
		holds), at which point $$__destructor__'s OWN replay (this same
		method, called again from there) would see the SAME flag still
		True and fire the identical body a SECOND time - a real double-
		replay bug, confirmed by reasoning through exactly this sequence
		(natural exhaustion replays a defer, main() later lets the
		generator go out of scope too). Clearing it here is a no-op the
		one time this method is actually called FROM the destructor
		itself (nothing reads the field again before sys.free(self)). '''
		guards: list[ast.stmt] = []
		for flag_stem, is_errdefer, body_stmts in reversed( defer_sites ):
			if is_errdefer:
				continue
			body_copy = [ copy.deepcopy( s ) for s in body_stmts ]
			unset = ast.Assign( targets = [ self._self_attr( flag_stem, anchor ) ], value = ast.Constant( value = False ) )
			ast.copy_location( unset, anchor )
			guard = ast.If( test = self._self_attr( flag_stem, anchor ), body = ( body_copy or [ ast.Pass() ] ) + [ unset ], orelse = [] )
			ast.copy_location( guard, anchor )
			guards.append( guard )
		return guards

	def _armed_flag_stem_of( self, stmt: ast.stmt ) -> 'str|None':
		''' recognizes one of _desugar_generator_defer_sites' own arm-assign
		statements (`self.__defer_armed_N = True`, which replaced the
		original `with defer:`/`defer(...)` site in place) - used by
		_tag_armed_defer_sites below to track, while walking a generator
		body in program order, exactly which prefix of `defer_sites` is
		"currently armed" at any later point (arming only ever happens at
		a direct top-level statement - see _validate_generator_defer_
		sites - so a simple in-order scan is enough, no separate control-
		flow analysis needed). '''
		if isinstance( stmt, ast.Assign ) and len( stmt.targets ) == 1:
			target = stmt.targets[0]
			if ( isinstance( target, ast.Attribute ) and isinstance( target.value, ast.Name )
				and target.value.id == 'self' and target.attr.startswith( '__defer_armed_' )):
				return target.attr
		return None

	def _tag_armed_defer_sites( self, stmts: list[ast.stmt], defer_sites: list[tuple[str,bool,list[ast.stmt]]], armed_count: list[int] ) -> None:
		''' Mechanism 2 (PLAN_GENERATORS.md's defer/errdefer phase) - tags
		every statement in `stmts` (a preamble, or a while/if-unit's own
		pre-/post-yield slice - the exact same granularity _pessimistic_
		done_prefix already wraps, called from the same call sites) with
		`generator_armed_defer_sites`: the PREFIX of `defer_sites` armed by
		the time this statement runs. `armed_count` is a shared, mutable
		single-element list (a plain int can't be mutated through a
		function boundary) threaded through every call across one
		$$__next__ build, advanced in place whenever an arm-assign
		(_armed_flag_stem_of) is crossed - since arming only ever happens
		at a top-level statement, and this method is only ever called on
		top-level-or-unit-slice statement lists in PROGRAM order, a simple
        running count is sufficient; no separate control-flow walk needed.

		Tagging only the TOP-LEVEL statement in each slice (not descending
		into a nested if/while's own body) is sufficient: lowering.py's own
		hook (task #41) reads this tag once, in its per-statement dispatch,
		and keeps it active for that ENTIRE statement's own recursive
		lowering (mirroring how _arithmetic_mode is already pushed/popped
		around a whole statement, not re-read per sub-expression) - so a
		fallible operation nested inside an ordinary if/call within a
		tagged statement still sees the right armed set.

		A no-op when defer_sites is empty (nothing to ever tag) - every
		call site already guards on this to skip the work entirely. '''
		for stmt in stmts:
			flag_stem = self._armed_flag_stem_of( stmt )
			if flag_stem is not None:
				armed_count[0] += 1
				continue
			stmt.generator_armed_defer_sites = defer_sites[ : armed_count[0] ]

	def _reject_generator_value_return( self, fn: Function ) -> None:
		for node in self._walk_generator_body( fn.node.body ):
			if isinstance( node, ast.Return ) and node.value is not None and not ( isinstance( node.value, ast.Constant ) and node.value.value is None ):
				self.discovery.fail( f'{fn.qualname}: a generator function cannot `return` a value (a bare `return` ends iteration) - see PLAN_GENERATORS.md', node )

	def _rewrite_generator_bare_returns( self, fn: Function, defer_sites: list[tuple[str,bool,list[ast.stmt]]] ) -> list[ast.Assign]:
		''' a bare `return` inside a generator body (already confirmed, by
		_reject_generator_value_return running just before this, to carry no
		value) compiles today but doesn't end iteration the way real Python
		generator semantics require - it's just an ordinary early `return
		None` out of $$__next__, which leaves self.__state exactly where it
		was BEFORE this call. A later manual .__next__() call would then
		wrongly resume and re-run whatever this return was meant to skip,
		instead of staying permanently exhausted (see PLAN_GENERATORS.md's
		defer/errdefer phase writeup for how this gap was found).

		Fixed the same way Phase 4 (roadmap Phase 4) already fixed the
		analogous or_return()-early-exit gap for fallible generators (see
		_pessimistic_done_prefix's own docstring): rewrite every bare
		`return` - wherever it's reachable, including nested inside an
		ordinary if/while/with/match, a while-unit's own loop body, or an
		if-unit's own branch (today only break/continue get checked inside
		those - see _validate_while_yield_unit) - into `<armed-defer-replay
		guards>; self.__state = <placeholder>; return None`, mutating
		fn.node.body itself, IN PLACE, BEFORE _collect_generator_units/
		_split_generator_segments/the guard builders ever consume it. Each
		guard builder already renames/wraps whatever it's handed (_rename_
		and_track_liveness, _pessimistic_done_prefix) without caring how
		many statements are in a given segment, so no changes are needed
		there - this only has to run early enough that the extra statements
		are already sitting in the body by the time those methods slice/
		copy it. This is also why the defer-replay guards inserted here are
		left RAW (un-renamed) - see _build_defer_replay_guards's own
		docstring: whatever segment/tail this bare return ends up part of
		gets renamed as a WHOLE, later, by the ordinary per-segment
		renamer.visit(...)/_rename_and_track_liveness call every other
		statement in it already goes through.

		The real done_state value isn't known until AFTER every unit is
		built (same reason _pessimistic_done_prefix's own pending_done_
		assigns list is patched late, not while building this) - unlike
		that mechanism, THIS one has to run regardless of fallible-ness (an
		Iterator[T] generator needs to stay permanently exhausted after a
		bare return exactly as much as a Generator[T,E] one does), so it
		gets its OWN always-populated pending list, patched to done_state by
		_build_generator_next_function alongside pending_done_assigns. '''
		pending: list[ast.Assign] = []
		fn.node.body = self._rewrite_bare_return_stmts( fn.node.body, pending, defer_sites )
		return pending

	def _rewrite_bare_return_stmts( self, stmts: list[ast.stmt], pending: list[ast.Assign], defer_sites: list[tuple[str,bool,list[ast.stmt]]] ) -> list[ast.stmt]:
		''' helper for _rewrite_generator_bare_returns - see its own
		docstring. Recurses into every nested statement-list-bearing field
		this language's statements can have (If/While/With's own `body`/
		`orelse`, plus match_case's own `body` - there's no try/except here
		to worry about, confirmed elsewhere in this file), but NOT into a
		nested def/lambda (a separate, unrelated scope - same boundary
		_walk_generator_body's own docstring explains). '''
		result: list[ast.stmt] = []
		for stmt in stmts:
			if isinstance( stmt, ast.Return ) and ( stmt.value is None or ( isinstance( stmt.value, ast.Constant ) and stmt.value.value is None )):
				# PLAN_GENERATORS.md's StopIteration reversal - a user-
				# written `return`/`return None` is a real generator-ending
				# exit, same as the tail's own natural exhaustion and the
				# DONE short-circuit above - tag it the same way so
				# _wrap_generator_next_returns_in_ok wraps its value in
				# Result.Err(StopIteration()) instead of Result.Ok(None)
				stmt.generator_exhaustion_return = True
				assign = ast.Assign( targets = [ self._self_attr( '__state', stmt ) ], value = ast.Constant( value = 0 ) )
				ast.copy_location( assign, stmt )
				pending.append( assign )
				if stmt.value is None:
					# a truly bare `return` (no expression at all) lowers to
					# a void C `return;` - wrong, $$__next__'s own declared
					# return type is never void (always elem_type|None, or
					# Result[...] when fallible) - every OTHER synthesized
					# return in this file already uses an explicit
					# `ast.Constant(value=None)` (see the DONE short-
					# circuit/tail's own returns just above), so normalize
					# this one the same way, confirmed via a real repro that
					# failed to even COMPILE otherwise ("non-void function
					# ... should return a value")
					stmt.value = ast.Constant( value = None )
					ast.copy_location( stmt.value, stmt )
				# PLAN_GENERATORS.md's defer/errdefer phase - a bare return
				# is a real generator-ending exit, exactly like the tail's
				# own natural exhaustion, so every currently-armed plain
				# `defer` site (LIFO) replays here too, right before the
				# state gets pinned to done - see _build_defer_replay_
				# guards's own docstring for why these are left un-renamed
				result.extend( self._build_defer_replay_guards( defer_sites, stmt ))
				result.append( assign )
				result.append( stmt )
				continue
			if isinstance( stmt, ( ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda )):
				result.append( stmt )
				continue
			if isinstance( stmt, ast.Match ):
				for case in stmt.cases:
					case.body = self._rewrite_bare_return_stmts( case.body, pending, defer_sites )
			elif hasattr( stmt, 'body' ):
				stmt.body = self._rewrite_bare_return_stmts( stmt.body, pending, defer_sites )
			if hasattr( stmt, 'orelse' ):
				stmt.orelse = self._rewrite_bare_return_stmts( stmt.orelse, pending, defer_sites )
			result.append( stmt )
		return result

	def _is_generator_range_call( self, node: ast.expr ) -> bool:
		# textual recognition, same shape as lowering.py's own
		# _is_range_call (deliberately duplicated rather than reached
		# across the Lowering/TypeResolver boundary - TypeResolver is
		# constructed before Lowering and holds no back-reference to it;
		# this check is cheap, self-contained, and already purely textual,
		# so duplicating it is simpler and safer than threading a new
		# dependency through). range() itself stays a compiler intrinsic
		# always - see ARCHITECTURE.md's own "design decision" section -
		# this is only about RECOGNIZING a for-loop over it inside a
		# generator body, not about implementing range() as a generator
		if isinstance( node, ast.Call ) and isinstance( node.func, ast.Name ) and node.func.id == 'range':
			return True
		return False

	def _probe_method( self, owner_type: Type|None, name: str ) -> Function|None:
		''' PLAN_GENERATORS.md Phase 1 - non-failing probe (unlike
		_attr_lookup_callable, which raises on a miss - a real error for an
		ordinary method call, but "this type has no such method" is a
		perfectly normal outcome here, deciding which for-loop desugaring
		shape applies). Mirrors lowering.py's own _find_method exactly,
		deliberately duplicated rather than reached across the
		TypeResolver/Lowering boundary - same reasoning
		_is_generator_range_call's own comment already gives. '''
		if owner_type is None:
			return None
		owner_type = self.ensure_resolved( owner_type )
		if isinstance( owner_type, ( CStruct, RCClass )):
			found = owner_type.chain_lookup( name )
		else:
			names = getattr( owner_type, 'names', None )
			found = names.get( name ) if isinstance( names, dict ) else None
		return found if isinstance( found, Function ) else None

	def _resolve_expr_type_for_desugar( self, fn: Function, expr: ast.expr ) -> Type|None:
		''' PLAN_GENERATORS.md Phase 1 - best-effort "what type does this
		expression have", used to decide which for-loop desugaring shape
		applies (indexable vs. __next__-based) BEFORE any real lowering
		exists to ask lowering.py's own _lower_expr. Reuses
		_ReferenceResolver._type_of_expr - already proven for exactly this
		"type an arbitrary expression from AST alone" need (it's what
		visit_Match uses to type a match subject, which already accepts
		arbitrary expressions, not just bare names - real precedent, not
		a new capability). Constructed standalone, NEVER calling its own
		.visit() (which would rewrite is-None/match/generic-call shapes
		this desugaring pass has no business touching) - .locals is seeded
		from parameters (by _ReferenceResolver's own __init__) plus a
		permissive walk collecting every already-declared generator
		local's annotated type (mirrors _collect_generator_locals's own
		AnnAssign scan, but without its strict validation - a real
		validation error, if any, still surfaces correctly once
		_collect_generator_locals runs for real, after desugaring
		completes). '''
		ref_resolver = _ReferenceResolver( self, fn )
		for node in self._walk_generator_body( fn.node.body ):
			if isinstance( node, ast.AnnAssign ) and isinstance( node.target, ast.Name ):
				ref_resolver.locals[ node.target.id ] = self.discovery.visit( node.annotation )
		return ref_resolver._type_of_expr( expr )

	def _desugar_range_for( self, fn: Function, node: ast.For ) -> list[ast.stmt]:
		''' PLAN_GENERATORS.md Phase 4 - `for x in range(...): BODY`
		(containing yield) becomes the EXACT equivalent while-loop shape
		(`x: usize = start; while x < stop: BODY; with compiler.
		wrap_arithmetic: x += 1`), mirroring lowering.py's own
		_lower_for_range exactly: usize target, 1 or 2 positional args, no
		keywords/step, the increment structurally guaranteed safe (x < stop
		strictly before every increment) so it bypasses ordinary checked-
		arithmetic policy the same way _lower_for_range's own raw AddWrap
		does - here, from synthesized AST rather than raw IR, the only way
		to get the same bypass is an explicit `with compiler.
		wrap_arithmetic:` wrapper. Returns a real ast.While, so every
		existing while-unit mechanism (Phase 2 - _validate_while_yield_
		unit/_build_while_unit_guard) picks it up with zero changes - this
		IS the entire mechanism Phase 4 needed (see
		_desugar_generator_for_loops, this method's only caller). '''
		if not isinstance( node.target, ast.Name ):
			self.discovery.fail( f'{fn.qualname}: for loop target must be a plain name: {ast.unparse(node)}', node )
		if node.orelse:
			self.discovery.fail( f'{fn.qualname}: for/else is not supported', node )
		call = node.iter
		if not self._is_generator_range_call( call ):
			self.discovery.fail(
				f'{fn.qualname}: a for loop containing yield is only supported over range(...) yet - see PLAN_GENERATORS.md',
				node,
			)
		if call.keywords:
			self.discovery.fail( f'{fn.qualname}: range(...) does not support keyword arguments: {ast.unparse(call)}', call )
		if len( call.args ) == 1:
			start_expr = ast.Constant( value = 0 )
			ast.copy_location( start_expr, call )
			stop_expr = call.args[0]
		elif len( call.args ) == 2:
			start_expr, stop_expr = call.args
		else:
			self.discovery.fail( f'{fn.qualname}: range(...) supports 1 or 2 arguments only (no step yet): {ast.unparse(call)}', call )

		target_id = node.target.id
		usize_name = ast.Name( id = 'usize', ctx = ast.Load() )
		ast.copy_location( usize_name, node )
		init = ast.AnnAssign(
			target = ast.Name( id = target_id, ctx = ast.Store() ),
			annotation = usize_name, value = start_expr, simple = 1,
		)
		ast.copy_location( init, node )

		increment = ast.AugAssign( target = ast.Name( id = target_id, ctx = ast.Store() ), op = ast.Add(), value = ast.Constant( value = 1 ) )
		wrapped_increment = ast.With(
			items = [ ast.withitem(
				context_expr = ast.Attribute( value = ast.Name( id = 'compiler', ctx = ast.Load() ), attr = 'wrap_arithmetic', ctx = ast.Load() ),
				optional_vars = None,
			) ],
			body = [ increment ],
		)
		ast.copy_location( wrapped_increment, node )

		while_node = ast.While(
			test = ast.Compare( left = ast.Name( id = target_id, ctx = ast.Load() ), ops = [ ast.Lt() ], comparators = [ stop_expr ] ),
			body = list( node.body ) + [ wrapped_increment ],
			orelse = [],
		)
		ast.copy_location( while_node, node )
		ast.fix_missing_locations( while_node )
		ast.fix_missing_locations( init )
		return [ init, while_node ]

	def _desugar_generator_for_loops( self, fn: Function ) -> dict[str,Type]:
		''' PLAN_GENERATORS.md Phase 4 (range()) + Phase 1 (indexable/
		iterator) - a top-level `for x in <expr>: BODY` containing a yield
		is rewritten, in place, into its own exactly-equivalent while form
		BEFORE unit collection ever runs - the ONLY new mechanism `for`-
		loop generator support needed: every downstream step (locals
		collection, unit recognition/guard-building) already handles a
		while unit correctly (Phase 2), so ANY for-loop shape this desugars
		gets that support for free, with zero changes to any of it.
		range() is recognized textually (_is_generator_range_call, zero
		type resolution needed, unchanged from Phase 4); anything else
		goes through _desugar_general_for, which resolves <expr>'s type
		(_resolve_expr_type_for_desugar) to decide indexable vs. iterator
		shape.

		Returns extra_locals: name -> type for every promoted local this
		desugaring itself needs beyond what _collect_generator_locals's
		own AnnAssign scan can discover on its own (currently: just
		__for_obj_N, the iterated expression, for the indexable/iterator
		shapes - range()'s own desugaring needs none, its only new local
		is the scalar loop counter, already covered by the ordinary
		AnnAssign-scan mechanism). __for_obj_N is re-evaluated every time
		this desugared for-loop is actually reached (an ordinary Assign,
		not a real annotation - hence extra_locals rather than a real
		AnnAssign the scan would find on its own) - see _new_for_obj_
		field's own docstring for why that matters once this is reachable
		more than once per generator lifetime.

		A for-loop with no yield in it at all is left completely alone
		(ordinary preamble/body content, not this pass's concern).

		PLAN_GENERATORS.md A.4a follow-up - generalized from top-level-
		only to recursing into nested if/while/for/with bodies (same
		"_recurse_*_wrap" shape used elsewhere in this file - see
		_recurse_desugar_for_loops) - a for-loop-with-yield reachable
		through if/with now desugars correctly at any nesting depth, not
		just the top level; so does one reachable through a while/for that
		could re-enter it, now that __for_obj_N is safe to re-derive on
		every entry (see _new_for_obj_field) rather than only constructed
		once. '''
		extra_locals: dict[str,Type] = {}
		fn.node.body = self._recurse_desugar_for_loops( fn, fn.node.body, extra_locals )
		return extra_locals

	def _recurse_desugar_for_loops( self, fn: Function, stmts: list[ast.stmt], extra_locals: dict[str,Type] ) -> list[ast.stmt]:
		new_body: list[ast.stmt] = []
		for stmt in stmts:
			if isinstance( stmt, ast.For ) and any(
				isinstance( n, ( ast.Yield, ast.YieldFrom )) for n in self._walk_generator_body( stmt.body )
			):
				if self._is_generator_range_call( stmt.iter ):
					desugared = self._desugar_range_for( fn, stmt )
				else:
					desugared = self._desugar_general_for( fn, stmt, extra_locals )
				# a for-loop-with-yield DIRECTLY nested inside this one's
				# own original body (e.g. `for x in xs: for y in gen():
				# yield y`) is now sitting, unrecognized, inside the just-
				# built while_node's own spliced-in body - `desugared`
				# itself is no longer an ast.For (so the isinstance check
				# above would never fire on it again), but recursing here
				# reaches it via the ast.While branch below, exactly like
				# any other nested for-loop-with-yield would be discovered.
				# Without this, that inner one silently fell through to
				# lowering.py's ORDINARY (non-generator-aware) for-loop
				# lowering instead of ever getting its own while-unit
				# desugaring - unreachable before A.4a (a for-loop-with-
				# yield nested inside another while/for was always
				# rejected outright, regardless of which one was the
				# outer), confirmed via a real repro once that rejection
				# lifted: MSVC crashed (debug: heap-corruption breakpoint;
				# release: access violation) on exactly this shape - clang/
				# gcc's own codegen happened not to visibly corrupt anything
				# for the same wrong IR, masking it completely.
				new_body.extend( self._recurse_desugar_for_loops( fn, desugared, extra_locals ))
				continue
			if isinstance( stmt, ( ast.If, ast.While, ast.For )):
				stmt.body = self._recurse_desugar_for_loops( fn, stmt.body, extra_locals )
				stmt.orelse = self._recurse_desugar_for_loops( fn, stmt.orelse, extra_locals )
			elif isinstance( stmt, ast.With ):
				stmt.body = self._recurse_desugar_for_loops( fn, stmt.body, extra_locals )
			new_body.append( stmt )
		return new_body

	def _desugar_general_for( self, fn: Function, node: ast.For, extra_locals: dict[str,Type] ) -> list[ast.stmt]:
		''' PLAN_GENERATORS.md Phase 1 - `for x in <expr>: BODY` where
		<expr> isn't range() - resolves <expr>'s type (best-effort, AST-
		only - see _resolve_expr_type_for_desugar) and dispatches to
		whichever shape it has: __next__() -> T|None (another generator,
		or any hand-written iterator - checked FIRST, matching lowering.
		py's own _stmt_For priority for an ordinary for-loop) or
		__len__()+__getitem__() (an indexable like list[T]). Neither
		found, or the type can't be determined at all, is a clear compile
		error - not a silent fallback to some other behavior. '''
		if not isinstance( node.target, ast.Name ):
			self.discovery.fail( f'{fn.qualname}: for loop target must be a plain name: {ast.unparse(node)}', node )
		if node.orelse:
			self.discovery.fail( f'{fn.qualname}: for/else is not supported', node )
		obj_type = self._resolve_expr_type_for_desugar( fn, node.iter )
		if obj_type is None:
			self.discovery.fail(
				f'{fn.qualname}: cannot determine the type of {ast.unparse(node.iter)} to desugar this for '
				f'loop - a for loop containing yield needs its iterated expression\'s type to be resolvable '
				f'without lowering (a parameter, an already-declared local, or a simple attribute/call chain) '
				f'- see PLAN_GENERATORS.md',
				node,
			)
		next_fn = self._probe_method( obj_type, '__next__' )
		if next_fn is not None:
			return self._desugar_iterator_for( fn, node, obj_type, next_fn, extra_locals )
		len_fn = self._probe_method( obj_type, '__len__' )
		getitem_fn = self._probe_method( obj_type, '__getitem__' )
		if len_fn is not None and getitem_fn is not None:
			return self._desugar_indexable_for( fn, node, obj_type, getitem_fn, extra_locals )
		self.discovery.fail(
			f'{fn.qualname}: a for loop containing yield needs __len__ and __getitem__ (or __next__ '
			f'returning T|None) on {obj_type.qualname if obj_type else "?"}: {ast.unparse(node)}',
			node,
		)

	def _new_for_obj_field( self, node: ast.For, obj_type: Type, extra_locals: dict[str,Type] ) -> tuple[str,ast.stmt]:
		''' registers a fresh __for_obj_N promoted local (type obj_type,
		initial value node.iter) and returns (obj_name, obj_init) - shared
		by _desugar_indexable_for/_desugar_iterator_for. The caller MUST
		place obj_init as the FIRST statement of its own returned list, so
		it (re-)runs every time program execution reaches this desugared
		for-loop, not just once - see below for why that matters.

		Used to be EAGER instead (evaluated exactly once, in the
		generator's own CONSTRUCTOR, via extra_fields - a field
		unconditionally valid from construction onward, same posture as a
		captured parameter, needing no live-flag guard): at the time,
		v1's RC-safety model only supported fields with exactly that
		shape, so eager construction-time evaluation was the only way to
		get a real destructor cascade for free. That gap was real, not
		cosmetic - a for-loop-with-yield (or `yield from`, which desugars
		into one) reachable through a while/for loop that could re-enter
		it reused the SAME already-exhausted __for_obj_N on every re-entry
		instead of it being freshly reconstructed, which is why
		_reject_generator_for_or_yield_from_nested_inside_loop used to
		exist at all (confirmed via a real repro before this fix: `while
		j < count: yield from inner(); j += 1` only ever forwarded
		inner()'s own values during the outer loop's FIRST pass - every
		later pass silently forwarded nothing). Once Phase 5's live-flag-
		guarded promoted-local machinery existed there was no longer a
		reason to accept that gap: obj_name now gets the exact same
		treatment as any other RC-typed promoted local (__for_next_N, this
		file's own A.4a work) - re-evaluated, with its own stale value
		correctly decref'd first, every time program execution reaches
		it - which is exactly a real Python generator's own lazy,
		per-entry construction, not merely a safe approximation of it.
		Tagged compiler_synthesized_for_loop_temp (same exemption
		__for_next_N's own non-promoted form uses) since obj_name's type
		is supplied directly via extra_locals (merged into locals_decl by
		this method's ultimate caller, ensure_generator_synthesized)
		rather than a real annotation - _collect_generator_locals's own
		AnnAssign scan never needs to discover it independently. '''
		unique = self._for_desugar_counter
		self._for_desugar_counter += 1
		obj_name = f'__for_obj_{unique}'
		extra_locals[ obj_name ] = obj_type
		obj_init = ast.Assign( targets = [ ast.Name( id = obj_name, ctx = ast.Store() ) ], value = node.iter )
		ast.copy_location( obj_init, node )
		obj_init.compiler_synthesized_for_loop_temp = True
		return obj_name, obj_init

	def _maybe_unwrap_call( self, call_expr: ast.expr, return_type: Type|None, node: ast.AST, msg: str ) -> tuple[ast.expr,Type|None]:
		''' PLAN_GENERATORS.md Phase 1 - if return_type is Result[T,E]-
		shaped, wraps call_expr in an explicit `.unwrap(msg)` (confirmed
		via a real repro: list[T].__getitem__/__len__ are BOTH fallible,
		Result[T,IndexError] - and unlike lowering.py's own _lower_for_
		over_indexable, which auto-propagates via _maybe_consume_result
		because IT'S lowering an ordinary for-loop where the enclosing
		function might legitimately be Result-shaped, a generator's own
		$$__next__ never is in v1 - propagation isn't an option here,
		only a panic. Safe: every call site this is used for has a
		structurally-guaranteed-safe precondition (an index strictly less
		than a just-read length), matching the exact reasoning _lower_
		for_range's own raw-AddWrap bypass already relies on for its
		increment - this is that same guarantee, just for a fallible
		METHOD instead of arithmetic) and returns (wrapped_expr, T);
		otherwise returns (call_expr, return_type) unchanged - not every
		indexable's own __len__/__getitem__ need be fallible, only
		list[T]'s confirmed to be. '''
		shape = self._result_shape( return_type )
		if shape is None:
			return call_expr, return_type
		unwrap_call = ast.Call(
			func = ast.Attribute( value = call_expr, attr = 'unwrap', ctx = ast.Load() ),
			args = [ ast.Constant( value = msg ) ], keywords = [],
		)
		ast.copy_location( unwrap_call, node )
		return unwrap_call, shape[0]

	def _desugar_indexable_for( self, fn: Function, node: ast.For, obj_type: Type, getitem_fn: Function, extra_locals: dict[str,Type] ) -> list[ast.stmt]:
		''' `for x in <expr>: BODY` (has __len__/__getitem__) desugars into
		the exact while-loop equivalent lowering.py's own _lower_for_over_
		indexable already builds at IR level - here as source AST feeding
		the existing Phase 2 while-unit machinery unchanged. __for_obj
		itself is the ONLY new promoted local (_new_for_obj_field);
		__for_len/__for_index are ordinary scalar generator locals,
		already covered by _collect_generator_locals with zero changes.
		Both __len__() and __getitem__() are called explicitly (not via
		`[]` subscript syntax, which hard-codes propagation) and passed
		through _maybe_unwrap_call - see its own docstring for why panic,
		not propagation, is the only option available to a generator's
		own $$__next__. x's own element type is __getitem__'s UNWRAPPED
		return type, spelled as a bare ast.Name(id=elem_type.stem) for its
		own AnnAssign annotation - an ordinary promoted local like any
		other since PLAN_GENERATORS.md Phase 5 (roadmap Phase 5) lifted
		_collect_generator_locals' former scalar-only restriction, RC-typed
		elem_type included. '''
		self.ensure_resolved( getitem_fn )
		len_fn = self._probe_method( obj_type, '__len__' )
		assert len_fn is not None # caller (_desugar_general_for) already confirmed this
		self.ensure_resolved( len_fn )
		obj_name, obj_init = self._new_for_obj_field( node, obj_type, extra_locals )
		unique = self._for_desugar_counter
		self._for_desugar_counter += 1
		len_name = f'__for_len_{unique}'
		index_name = f'__for_index_{unique}'

		usize_name = ast.Name( id = 'usize', ctx = ast.Load() )
		ast.copy_location( usize_name, node )
		len_call = ast.Call(
			func = ast.Attribute( value = ast.Name( id = obj_name, ctx = ast.Load() ), attr = '__len__', ctx = ast.Load() ),
			args = [], keywords = [],
		)
		ast.copy_location( len_call, node )
		len_expr, _len_type = self._maybe_unwrap_call( len_call, len_fn.return_type, node, 'generator for-loop __len__() failed (unreachable)' )
		len_init = ast.AnnAssign(
			target = ast.Name( id = len_name, ctx = ast.Store() ), annotation = usize_name,
			value = len_expr, simple = 1,
		)
		index_init = ast.AnnAssign(
			target = ast.Name( id = index_name, ctx = ast.Store() ), annotation = usize_name,
			value = ast.Constant( value = 0 ), simple = 1,
		)
		ast.copy_location( len_init, node ); ast.copy_location( index_init, node )

		getitem_call = ast.Call(
			func = ast.Attribute( value = ast.Name( id = obj_name, ctx = ast.Load() ), attr = '__getitem__', ctx = ast.Load() ),
			args = [ ast.Name( id = index_name, ctx = ast.Load() ) ], keywords = [],
		)
		ast.copy_location( getitem_call, node )
		getitem_expr, elem_type = self._maybe_unwrap_call( getitem_call, getitem_fn.return_type, node, 'generator for-loop index is structurally guaranteed in bounds (unreachable)' )

		elem_type_name = ast.Name( id = elem_type.stem, ctx = ast.Load() ) if elem_type is not None else ast.Name( id = '?', ctx = ast.Load() )
		ast.copy_location( elem_type_name, node )
		target_bind = ast.AnnAssign(
			target = ast.Name( id = node.target.id, ctx = ast.Store() ), annotation = elem_type_name,
			value = getitem_expr, simple = 1,
		)
		ast.copy_location( target_bind, node )

		increment = ast.AugAssign( target = ast.Name( id = index_name, ctx = ast.Store() ), op = ast.Add(), value = ast.Constant( value = 1 ) )
		wrapped_increment = ast.With(
			items = [ ast.withitem(
				context_expr = ast.Attribute( value = ast.Name( id = 'compiler', ctx = ast.Load() ), attr = 'wrap_arithmetic', ctx = ast.Load() ),
				optional_vars = None,
			) ],
			body = [ increment ],
		)
		ast.copy_location( wrapped_increment, node )

		while_node = ast.While(
			test = ast.Compare( left = ast.Name( id = index_name, ctx = ast.Load() ), ops = [ ast.Lt() ], comparators = [ ast.Name( id = len_name, ctx = ast.Load() ) ] ),
			body = [ target_bind ] + list( node.body ) + [ wrapped_increment ],
			orelse = [],
		)
		ast.copy_location( while_node, node )
		ast.fix_missing_locations( while_node )
		ast.fix_missing_locations( len_init ); ast.fix_missing_locations( index_init )
		return [ obj_init, len_init, index_init, while_node ]

	def _type_annotation_ast( self, t: Type, node: ast.AST ) -> ast.expr:
		''' builds a fresh annotation-position AST expression resolving
		back to `t` via discovery.py's own visit(...) machinery - a bare
		ast.Name(id=t.stem) for a plain type (the pattern this file already
		uses throughout for elem_type_name etc.), or a chain of `A | B | C`
		BinOps rebuilt from t.leaves() for an anonymous union (t.stem
		itself, e.g. "A|B", is not a valid Python identifier and wouldn't
		resolve via ordinary name lookup - has to be spelled out textually,
		the same shape a user writing it by hand would, which discovery.
		py's own visit_BinOp/_flatten_union/_get_or_create_union already
		knows how to resolve back to the identical interned union). A
		NOMINAL @union (t.file is not None, e.g. a user's own MyError)
		stays a bare Name - its own stem IS a valid identifier, and it
		must NOT be decomposed into its own variants (matches _atomic_
		leaves' identical distinction elsewhere in this file). Needed by
		_desugar_iterator_for's own StopIteration-stripped remaining-error
		type, which can legitimately be a fresh multi-member union. '''
		if t is self.discovery.get_none_type():
			result: ast.expr = ast.Constant( value = None )
			ast.copy_location( result, node )
			return result
		if isinstance( t, TaggedUnion ) and t.file is None:
			leaves = t.leaves()
			expr = self._type_annotation_ast( leaves[0], node )
			for leaf in leaves[1:]:
				expr = ast.BinOp( left = expr, op = ast.BitOr(), right = self._type_annotation_ast( leaf, node ) )
				ast.copy_location( expr, node )
			return expr
		result = ast.Name( id = t.stem, ctx = ast.Load() )
		ast.copy_location( result, node )
		return result

	def _desugar_iterator_for( self, fn: Function, node: ast.For, obj_type: Type, next_fn: Function, extra_locals: dict[str,Type] ) -> list[ast.stmt]:
		''' `for x in <expr>: BODY` where <expr> has __next__() ->
		Result[T,E] (E always includes StopIteration - PLAN_GENERATORS.md's
		StopIteration reversal; most commonly: another generator).
		Extracting the payload needs real narrowing, and the only working
		mechanism is `match subject: case T(subject): ...` reusing the
		subject's own name (cfg.py's narrow()/narrowed_member(), same as
		lowering.py's own _lower_for_over_iterator uses at the IR level) -
		and that narrowing does NOT survive past the branch that
		established it, so the extraction has to happen INSIDE the match's
		own case arm, writing directly into x (an ordinary field by then,
		no narrowing concern once written). BODY itself (containing the
		yield) stays a SIBLING of the match statement, not nested inside
		it - keeping yield at the exact nesting depth _validate_while_
		yield_unit already requires, with zero changes to that validator.

		x's own binding shape (confirmed directly with the user): if E is
		JUST StopIteration (no other error), x binds to plain T - the loop
		itself handles StopIteration as ordinary termination, never
		surfaced to the loop body. If E has any OTHER error alongside
		StopIteration, x binds to Result[T,E'] with StopIteration already
		stripped out of E' - the caller handles the real error explicitly
		inside the loop body (match/.is_err()/.or_return()/.unwrap()), the
		loop does NOT auto-propagate it (deliberately different from
		_maybe_consume_result's own auto-propagate idiom for __len__/
		__getitem__ elsewhere in this file - don't conflate the two).
		Narrowing `e`'s own type down to E' inside the wildcard arm after
		one explicit `case StopIteration(_):` arm reuses visit_Match's own
		existing wildcard-narrows-to-the-union's-remaining-members
		mechanism (Phase 6) - confirmed via a real repro, no bespoke
		per-leaf rewrap machinery needed. `yield from` (_desugar_generator_
		yield_from, below) requires an EXACT match between the inner and
		outer generator's own Result[T,E] shapes instead of going through
		this general binding - see its own docstring. '''
		self.ensure_resolved( next_fn )
		shape = self._result_shape( next_fn.return_type )
		if shape is None:
			self.discovery.fail(
				f'{fn.qualname}: for loop needs __next__() to return Result[T,E] on '
				f'{obj_type.qualname if obj_type else "?"}: {ast.unparse(node)}',
				node,
			)
		elem_type, full_error_type = shape
		stop_iteration_cls = self.discovery.find_name_or_none( 'StopIteration' )
		if stop_iteration_cls is None or stop_iteration_cls not in self._atomic_leaves( full_error_type ):
			self.discovery.fail(
				f'{fn.qualname}: for loop needs __next__()\'s own error type to include StopIteration on '
				f'{obj_type.qualname if obj_type else "?"}: {ast.unparse(node)}',
				node,
			)
		remaining_leaves = [ leaf for leaf in self._atomic_leaves( full_error_type ) if leaf is not stop_iteration_cls ]
		if not remaining_leaves:
			remaining_error_type: Type|None = None
		elif len( remaining_leaves ) == 1:
			remaining_error_type = remaining_leaves[0]
		else:
			remaining_error_type = self.discovery._get_or_create_union( remaining_leaves )

		obj_name, obj_init = self._new_for_obj_field( node, obj_type, extra_locals )
		unique = self._for_desugar_counter
		self._for_desugar_counter += 1
		next_name = f'__for_next_{unique}'

		if remaining_error_type is None:
			x_type: Type|None = elem_type
			x_annotation = self._type_annotation_ast( elem_type, node )
		else:
			result_cls = self.discovery.find_name_or_none( 'Result' )
			assert isinstance( result_cls, ClassLike ), 'builtins.Result is required for a for-loop-with-yield but was not found'
			x_type = self.discovery._get_or_create_specialization( result_cls, [ elem_type, remaining_error_type ] )
			x_annotation = ast.Subscript(
				value = ast.Name( id = 'Result', ctx = ast.Load() ),
				slice = ast.Tuple( elts = [ self._type_annotation_ast( elem_type, node ), self._type_annotation_ast( remaining_error_type, node ) ], ctx = ast.Load() ),
				ctx = ast.Load(),
			)
			ast.copy_location( x_annotation, node )
			ast.copy_location( x_annotation.slice, node )
		if isinstance( x_type, Scalar ) and x_type.stem == 'bool':
			target_zero = ast.Constant( value = False )
		else:
			target_zero = ast.Constant( value = 0 )
			if x_type is not None and x_type.is_rc():
				# same exemption _rewrite_generator_constructor's own RC-
				# typed field zero-placeholders already use (_expr_
				# Constant's generator_zero_rc_field) - this loop target
				# is about to be immediately overwritten by the match arm
				# just below (never actually READ as this placeholder
				# value), but it's still a real, ordinary promoted local
				# needing SOME initial value satisfying its own RC-typed
				# declared annotation - confirmed via a real repro: a for-
				# loop consuming another generator whose elem_type is
				# RC-typed (e.g. `for x in some_gen_of_boxes():`) failed
				# to compile at all ("an int literal cannot be used where
				# Box is expected") - found via A.4a's own `yield from`
				# (a natural way to forward an RC-typed inner generator's
				# values), but reproduces identically with an ordinary
				# user-written for-loop, no yield-from involved. Checked
				# against x_type (bare elem_type, OR Result[elem_type,
				# remaining_error_type] once StopIteration is stripped
				# out) rather than elem_type alone - a Result wrapping is
				# RC whenever EITHER side is, so a scalar elem_type paired
				# with an RC-carrying remaining error still needs this.
				target_zero.generator_zero_rc_field = True
		target_init = ast.AnnAssign(
			target = ast.Name( id = node.target.id, ctx = ast.Store() ), annotation = x_annotation,
			value = target_zero, simple = 1,
		)
		ast.copy_location( target_init, node )

		next_call = ast.Call(
			func = ast.Attribute( value = ast.Name( id = obj_name, ctx = ast.Load() ), attr = '__next__', ctx = ast.Load() ),
			args = [], keywords = [],
		)
		# A.4a: when BODY itself contains a yield (the for-x-in-generator-
		# forwarding shape yield-from always desugars into), __for_next_N's
		# own raw __next__() result is alive ACROSS that yield's suspend -
		# the very next statement after it (the match extracting the
		# narrowed payload) only runs on the FOLLOWING resume, a genuinely
		# separate C function call with a fresh stack frame. The docstring
		# above ("recomputed fresh every resume, never crosses one") is only
		# true when BODY has no yield of its own - confirmed by a real
		# nested-generator yield-from repro that leaked one whole reference
		# per forwarded RC value: __for_next_N stayed a bare stack local, so
		# the value it captured from inner's own yield-wrap incref was
		# silently abandoned (never released) once the outer loop's own
		# match arm copied it onward into the promoted __yield_from_N field.
		# Fix: give it the SAME ordinary promoted-local treatment as any
		# user-written one (an AnnAssign, so _collect_generator_locals picks
		# it up and _apply_live_flag_guards gives it the standard live-flag-
		# guarded reassignment/destructor teardown) instead of opting it out.
		# Unlike the pre-StopIteration-reversal version of this method,
		# needs_promotion no longer needs its own RC check - __for_next_N's
		# own type, Result[elem_type,full_error_type], is now ALWAYS RC
		# (full_error_type always includes StopIteration, itself a real,
		# always-RC class like every other class in this language, even
		# with zero fields of its own), so body_has_yield alone already
		# implies it.
		body_has_yield = any( isinstance( n, ast.Yield ) for n in self._walk_generator_body( node.body ) )
		needs_promotion = body_has_yield
		if needs_promotion:
			next_annotation = ast.Subscript(
				value = ast.Name( id = 'Result', ctx = ast.Load() ),
				slice = ast.Tuple( elts = [ self._type_annotation_ast( elem_type, node ), self._type_annotation_ast( full_error_type, node ) ], ctx = ast.Load() ),
				ctx = ast.Load(),
			)
			ast.copy_location( next_annotation, node )
			ast.copy_location( next_annotation.slice, node )
			next_assign = ast.AnnAssign(
				target = ast.Name( id = next_name, ctx = ast.Store() ), annotation = next_annotation,
				value = next_call, simple = 1,
			)
		else:
			next_assign = ast.Assign( targets = [ ast.Name( id = next_name, ctx = ast.Store() ) ], value = next_call )
		ast.copy_location( next_assign, node )
		if not needs_promotion:
			# exempted from _collect_generator_locals's own "must be declared
			# with an explicit annotation" check - __for_next_N is deliberately
			# an ordinary $$__next__-scoped local (recomputed fresh every
			# resume, never crosses one - see this method's own docstring),
			# same posture as _build_while_unit_guard's own __gen_resuming_N,
			# just built one stage earlier (during desugaring, before locals
			# collection ever runs) so it needs an explicit opt-out here
			# instead of simply never being visible to that scan at all
			next_assign.compiler_synthesized_for_loop_temp = True

		exhausted_break = ast.Break()
		exhausted_break.compiler_synthesized_break = True # exempted from _validate_while_yield_unit's own break/continue rejection - see its own comment

		def bind_name( base: str ) -> str:
			return base if not needs_promotion else f'__for_{base}_{unique}'

		# same reasoning next_name's own "reuses next_name whenever it stays
		# a plain local" comment further down gives: a bind name only needs
		# to be distinct from any promoted field's own stem once needs_
		# promotion means everything here is renamed to self.<stem> -
		# _GeneratorNameRenamer only ever touches ast.Name nodes, never a
		# MatchAs pattern's own raw string .name, so reusing an ALREADY-
		# promoted stem here would silently bind a second, disjoint plain
		# local instead (see A.4a's own historical bug on this exact point,
		# _for_elem_N's original docstring, still accurate about WHY, just
		# renamed here since this method's own binding shape changed).
		ok_bind_name = bind_name( 'ok' )
		ok_case_body: list[ast.stmt] = []
		if remaining_error_type is None:
			ok_case_body.append( ast.Assign(
				targets = [ ast.Name( id = node.target.id, ctx = ast.Store() ) ], value = ast.Name( id = ok_bind_name, ctx = ast.Load() ),
			))
		else:
			rewrap = ast.Call(
				func = ast.Attribute( value = ast.Name( id = 'Result', ctx = ast.Load() ), attr = 'Ok', ctx = ast.Load() ),
				args = [ ast.Name( id = ok_bind_name, ctx = ast.Load() ) ], keywords = [],
			)
			ast.copy_location( rewrap, node ); ast.copy_location( rewrap.func, node ); ast.copy_location( rewrap.func.value, node )
			ok_case_body.append( ast.Assign( targets = [ ast.Name( id = node.target.id, ctx = ast.Store() ) ], value = rewrap ))
		if needs_promotion and elem_type is not None and elem_type.is_rc():
			# ok_bind_name stays a PLAIN, non-promoted local here on
			# purpose (tried promoting it via extra_locals first, for the
			# original single-level version of this shape - wrong:
			# _match_pattern always synthesizes its bind as a bare
			# ast.Assign into a bare ast.Name, built fresh at visit_Match
			# time - well AFTER _GeneratorNameRenamer has already run over
			# this whole body during _build_generator_next_function, so a
			# promoted bind name would just create a second, disjoint
			# storage location: the bind writes a genuinely fresh plain
			# local, while the re-store above - built here, so it DOES go
			# through the renamer - reads self.<bind_name> instead, the
			# STILL-UNINITIALIZED field. Confirmed via a real repro on the
			# original single-level shape: `x` came back None instead of
			# the real extracted value.
			#
			# Kept plain, this temp's own extraction incref (the ordinary
			# aliasing-read cost of pulling a payload out of the union,
			# doubled here since the re-wrap into Result.Ok(...)/Result.
			# Err(...) above ALSO increfs its own argument) still needs a
			# real decref, and cfg.py's own loop_back_edge() - which would
			# ordinarily supply that automatically, for any plain Name
			# binding confined to the loop - schedules it for the loop's
			# own back edge, past the yield a few statements below, in a
			# separate $$__resume__ call with a fresh stack frame
			# (confirmed via a real repro on the original single-level
			# shape: release_object() on a never-initialized local). So:
			# decref it explicitly, right here, immediately after the
			# re-store has taken its own independent reference - manually_
			# decreffed (cfg.py, reached via compiler.decref's own
			# lowering) marks it consumed, so the loop's own back-edge
			# reconciliation no longer tries a second time.
			ok_case_body.append( _expr_stmt( ast.Call(
				func = ast.Attribute( value = _id( 'compiler' ), attr = 'decref', ctx = ast.Load() ),
				args = [ ast.Name( id = ok_bind_name, ctx = ast.Load() ) ], keywords = [],
			)))
		elem_type_name = self._type_annotation_ast( elem_type, node )
		ok_case = ast.match_case(
			pattern = ast.MatchClass(
				cls = ast.Attribute( value = ast.Name( id = 'Result', ctx = ast.Load() ), attr = 'Ok', ctx = ast.Load() ),
				patterns = [ ast.MatchAs( name = ok_bind_name ) ], kwd_attrs = [], kwd_patterns = [],
			),
			guard = None,
			body = ok_case_body,
		)

		if remaining_error_type is None:
			# E is JUST StopIteration - Result[T,StopIteration].Err(_) can
			# only ever BE StopIteration, no inner match needed at all
			err_case = ast.match_case(
				pattern = ast.MatchClass(
					cls = ast.Attribute( value = ast.Name( id = 'Result', ctx = ast.Load() ), attr = 'Err', ctx = ast.Load() ),
					patterns = [ ast.MatchAs( name = None, pattern = None ) ], kwd_attrs = [], kwd_patterns = [],
				),
				guard = None,
				body = [ exhausted_break ],
			)
		else:
			err_bind_name = bind_name( 'err' )
			# ONE explicit case per remaining leaf - NOT a single trailing
			# wildcard covering all of them at once. A wildcard arm's own
			# subject read only narrows to a SINGLE concrete member when
			# EXACTLY one candidate remains after every sibling case
			# (cfg.py's narrowed_member() - see its own comment: a multi-
			# element narrowed set, the case with 2+ remaining leaves,
			# never collapses to one, so err_bind_name would stay typed
			# as the WHOLE full_error_type there, not remaining_error_
			# type) - confirmed via a real repro with 2 remaining leaves:
			# Result.Err(err_bind_name)'s own T/E inference disagreed
			# between the assignment target's declared Result[_,
			# remaining_error_type] and err_bind_name's own un-narrowed
			# full_error_type. Each leaf's own EXPLICIT case, by
			# contrast, always narrows to exactly that one leaf (same
			# mechanism the StopIteration(_) case below already uses),
			# giving err_bind_name a concrete single-class type that
			# widens cleanly into remaining_error_type.
			def build_leaf_case( leaf: Type, rebind: str|None, body: list[ast.stmt] ) -> ast.match_case:
				# a bare `case Leaf(_):` (rebind=None) tests the tag without
				# narrowing err_bind_name's own STATIC type for later reads -
				# only a trailing WILDCARD arm gets that treatment (visit_
				# Match's own Phase 6 "narrows to whatever remains"), confirmed
				# via a real repro (Result.Err(err_bind_name)'s own T/E
				# inference still saw the WIDE full_error_type inside a plain
				# `case Leaf(_):` arm). Binding a name directly into the
				# class's own single positional slot instead - same mechanism
				# `case Result.Err(err_bind_name):` already uses at the OUTER
				# level - narrows correctly even for a zero-field marker class
				# like StopIteration/MyError (the slot represents "the whole
				# matched value" then, not a real field) and even nested one
				# match deep - confirmed via a standalone repro
				# (zero_field_bind_check.py); a wrapping `as` pattern was tried
				# first and rejected ("unsupported match pattern") specifically
				# when nested inside another match's own case body - a real,
				# general pre-existing gap, sidestepped here rather than fixed.
				pattern = ast.MatchClass(
					cls = ast.Name( id = leaf.stem, ctx = ast.Load() ),
					patterns = [ ast.MatchAs( name = rebind, pattern = None ) ], kwd_attrs = [], kwd_patterns = [],
				)
				ast.copy_location( pattern, node )
				case = ast.match_case( pattern = pattern, guard = None, body = body )
				return case

			inner_stop_iteration_break = ast.Break()
			inner_stop_iteration_break.compiler_synthesized_break = True
			inner_cases = [ build_leaf_case( stop_iteration_cls, None, [ inner_stop_iteration_break ] ) ]
			for leaf in remaining_leaves:
				narrowed_name = bind_name( f'err_{leaf.stem}' )
				rewrap = ast.Call(
					func = ast.Attribute( value = ast.Name( id = 'Result', ctx = ast.Load() ), attr = 'Err', ctx = ast.Load() ),
					args = [ ast.Name( id = narrowed_name, ctx = ast.Load() ) ], keywords = [],
				)
				ast.copy_location( rewrap, node ); ast.copy_location( rewrap.func, node ); ast.copy_location( rewrap.func.value, node )
				leaf_body: list[ast.stmt] = [
					ast.Assign( targets = [ ast.Name( id = node.target.id, ctx = ast.Store() ) ], value = rewrap ),
				]
				if needs_promotion and leaf.is_rc():
					leaf_body.append( _expr_stmt( ast.Call(
						func = ast.Attribute( value = _id( 'compiler' ), attr = 'decref', ctx = ast.Load() ),
						args = [ ast.Name( id = narrowed_name, ctx = ast.Load() ) ], keywords = [],
					)))
				inner_cases.append( build_leaf_case( leaf, narrowed_name, leaf_body ))
			inner_match = ast.Match( subject = ast.Name( id = err_bind_name, ctx = ast.Load() ), cases = inner_cases )
			ast.copy_location( inner_match, node )
			err_case = ast.match_case(
				pattern = ast.MatchClass(
					cls = ast.Attribute( value = ast.Name( id = 'Result', ctx = ast.Load() ), attr = 'Err', ctx = ast.Load() ),
					patterns = [ ast.MatchAs( name = err_bind_name ) ], kwd_attrs = [], kwd_patterns = [],
				),
				guard = None,
				body = [ inner_match ],
			)
		match_stmt = ast.Match( subject = ast.Name( id = next_name, ctx = ast.Load() ), cases = [ err_case, ok_case ] )
		ast.copy_location( match_stmt, node )

		while_node = ast.While(
			test = ast.Constant( value = True ),
			body = [ next_assign, match_stmt ] + list( node.body ),
			orelse = [],
		)
		ast.copy_location( while_node, node )
		ast.fix_missing_locations( while_node )
		ast.fix_missing_locations( target_init )
		return [ obj_init, target_init, while_node ]

	def _desugar_generator_yield_from( self, fn: Function, elem_type: Type, error_type: Type ) -> dict[str,Type]:
		''' PLAN_GENERATORS.md's StopIteration reversal - `yield from
		<expr>` requires <expr>'s own __next__() to return EXACTLY
		Result[elem_type,error_type] (this generator's own declared
		shape, identity-compared - every Result[T,E] specialization is
		interned, same posture _require_result_return's own leaves-
		containment check already relies on) - confirmed directly with
		the user: no covering/widening check, no auto-propagation, every
		value (Ok AND Err alike) forwarded untouched except Err(
		StopIteration) specifically, which terminates yield-from's own
		loop (falls through to whatever follows the statement) rather
		than being forwarded as this generator's own exhaustion. A
		narrower/wider mismatch is a clear compile error directing the
		user to write an explicit `for` loop instead (which has its own,
		more permissive binding rule - see _desugar_iterator_for) - NOT
		silently downgraded to that shared path, which would double-wrap
		(the shared for-loop path always yields the UNWRAPPED bare-T/
		Result[T,E'] binding, auto-Ok-wrapped afterward by _wrap_
		generator_next_returns_in_ok same as any other yield - forwarding
		an ALREADY Result[elem_type,error_type]-shaped raw next() value
		through that same auto-wrap would produce Ok(Result[...]), not
		Result[...] itself).

		Run BEFORE _desugar_generator_for_loops - unlike A.4a's original
		version, no longer reuses that shared machinery at all for this
		exact-match case, so ordering no longer matters for THAT reason,
		but still runs first to keep both desugaring passes' own
		responsibilities cleanly separated (this handles yield-from
		expressions specifically; that handles for statements, which
		still includes any ordinary `for` loop the user wrote by hand to
		work around a yield-from mismatch this method rejects). Recurses
		into nested if/while/for/with bodies (same "_recurse_*_wrap"
		shape used elsewhere in this file) - a `yield from` reachable
		through a while/for that could re-enter it used to be rejected
		here (a nesting validator ran before this point); lifted once
		__for_obj_N-style locals stopped needing eager, construction-time-
		only evaluation to be RC-safe - see _new_for_obj_field's own
		docstring (this method's own __yield_from_obj_N field follows the
		identical lazy-per-entry-reconstruction posture).

		Returns extra_locals (name -> type) for every promoted local this
		desugaring itself needs beyond what _collect_generator_locals's
		own AnnAssign scan can discover - __yield_from_obj_N (the
		iterated expression, an ordinary Assign not a real annotation,
		same reason _new_for_obj_field's own __for_obj_N needs this) -
		merged by the caller into _desugar_generator_for_loops's own
		return value, same convention that method already establishes. '''
		extra_locals: dict[str,Type] = {}
		fn.node.body = self._recurse_desugar_yield_from( fn, fn.node.body, elem_type, error_type, extra_locals )
		return extra_locals

	def _recurse_desugar_yield_from( self, fn: Function, stmts: list[ast.stmt], elem_type: Type, error_type: Type, extra_locals: dict[str,Type] ) -> list[ast.stmt]:
		result: list[ast.stmt] = []
		for s in stmts:
			if isinstance( s, ast.Expr ) and isinstance( s.value, ast.YieldFrom ):
				result.extend( self._desugar_one_yield_from( fn, s, elem_type, error_type, extra_locals ))
				continue
			if isinstance( s, ( ast.If, ast.While, ast.For )):
				s.body = self._recurse_desugar_yield_from( fn, s.body, elem_type, error_type, extra_locals )
				s.orelse = self._recurse_desugar_yield_from( fn, s.orelse, elem_type, error_type, extra_locals )
			elif isinstance( s, ast.With ):
				s.body = self._recurse_desugar_yield_from( fn, s.body, elem_type, error_type, extra_locals )
			result.append( s )
		return result

	def _desugar_one_yield_from( self, fn: Function, s: ast.Expr, elem_type: Type, error_type: Type, extra_locals: dict[str,Type] ) -> list[ast.stmt]:
		iter_expr = s.value.value
		assert isinstance( s.value, ast.YieldFrom )
		obj_type = self._resolve_expr_type_for_desugar( fn, iter_expr )
		if obj_type is None:
			self.discovery.fail(
				f'{fn.qualname}: cannot determine the type of {ast.unparse(iter_expr)} to desugar this yield from '
				f'- its type needs to be resolvable without lowering (a parameter, an already-declared local, or a '
				f'simple attribute/call chain) - see PLAN_GENERATORS.md',
				s,
			)
		next_fn = self._probe_method( obj_type, '__next__' )
		if next_fn is None:
			self.discovery.fail( f'{fn.qualname}: yield from needs __next__() on {obj_type.qualname if obj_type else "?"}: {ast.unparse(s)}', s )
		self.ensure_resolved( next_fn )
		shape = self._result_shape( next_fn.return_type )
		stop_iteration_cls = self.discovery.find_name_or_none( 'StopIteration' )
		if shape is None or stop_iteration_cls is None or stop_iteration_cls not in self._atomic_leaves( shape[1] ):
			self.discovery.fail(
				f'{fn.qualname}: yield from needs __next__() to return Result[T,E] (E including StopIteration) on '
				f'{obj_type.qualname if obj_type else "?"}: {ast.unparse(s)}',
				s,
			)
		inner_elem_type, inner_error_type = shape
		if inner_elem_type is not elem_type or inner_error_type is not error_type:
			self.discovery.fail(
				f'{fn.qualname}: yield from requires an EXACT match between the consumed Result[T,E] '
				f'(Result[{inner_elem_type.qualname},{inner_error_type.qualname}]) and this generator\'s own '
				f'declared Result[T,E] (Result[{elem_type.qualname},{error_type.qualname}]) - write an explicit '
				f'for loop instead to handle the difference: {ast.unparse(s)}',
				s,
			)

		unique = self._for_desugar_counter
		self._for_desugar_counter += 1
		obj_name = f'__yield_from_obj_{unique}'
		extra_locals[obj_name] = obj_type
		obj_init = ast.Assign( targets = [ ast.Name( id = obj_name, ctx = ast.Store() ) ], value = iter_expr )
		ast.copy_location( obj_init, s )
		obj_init.compiler_synthesized_for_loop_temp = True

		next_name = f'__yield_from_next_{unique}'
		next_call = ast.Call(
			func = ast.Attribute( value = ast.Name( id = obj_name, ctx = ast.Load() ), attr = '__next__', ctx = ast.Load() ),
			args = [], keywords = [],
		)
		next_annotation = ast.Subscript(
			value = ast.Name( id = 'Result', ctx = ast.Load() ),
			slice = ast.Tuple( elts = [ self._type_annotation_ast( elem_type, s ), self._type_annotation_ast( error_type, s ) ], ctx = ast.Load() ),
			ctx = ast.Load(),
		)
		ast.copy_location( next_annotation, s )
		ast.copy_location( next_annotation.slice, s )
		# always promoted - a yield-from loop unconditionally yields on
		# every iteration (that's the whole point), so __yield_from_next_N
		# always crosses a yield, unlike a general for-loop's __for_next_N
		# which only needs promotion when its own BODY happens to yield
		next_assign = ast.AnnAssign(
			target = ast.Name( id = next_name, ctx = ast.Store() ), annotation = next_annotation,
			value = next_call, simple = 1,
		)
		ast.copy_location( next_assign, s )

		def forward_yield() -> ast.stmt:
			# forwards __yield_from_next_N UNCHANGED as this generator's own
			# $$__next__ return - already exactly Result[elem_type,error_
			# type]-shaped (the exact-match check above guarantees it), so
			# _wrap_generator_next_returns_in_ok must NOT auto-Ok-wrap it
			# like an ordinary yielded value - generator_already_result_
			# shaped tells it to leave this node's value untouched
			yield_expr = ast.Yield( value = ast.Name( id = next_name, ctx = ast.Load() ))
			ast.copy_location( yield_expr, s )
			yield_expr.generator_already_result_shaped = True
			stmt = ast.Expr( value = yield_expr )
			ast.copy_location( stmt, s )
			return stmt

		ok_case = ast.match_case(
			pattern = ast.MatchClass(
				cls = ast.Attribute( value = ast.Name( id = 'Result', ctx = ast.Load() ), attr = 'Ok', ctx = ast.Load() ),
				patterns = [ ast.MatchAs( name = None, pattern = None ) ], kwd_attrs = [], kwd_patterns = [],
			),
			guard = None,
			body = [ forward_yield() ],
		)

		exhausted_break = ast.Break()
		exhausted_break.compiler_synthesized_break = True
		if self._atomic_leaves( error_type ) == [ stop_iteration_cls ]:
			# error_type is BARE StopIteration - nothing else it could ever
			# be, so Result[elem_type,error_type].Err(_) is unconditionally
			# exhaustion - no inner match needed at all (mirrors _desugar_
			# iterator_for's own identical remaining_error_type-is-None
			# special case). Matching `case StopIteration(_): ... case _:
			# ...` against a subject whose OWN static type isn't a union at
			# all (nothing to distinguish) is rejected outright ("match
			# subject is not a union type") - confirmed via a real repro
			# (yield_from_rc.py, Iterator[Result[Box,StopIteration]])
			err_case = ast.match_case(
				pattern = ast.MatchClass(
					cls = ast.Attribute( value = ast.Name( id = 'Result', ctx = ast.Load() ), attr = 'Err', ctx = ast.Load() ),
					patterns = [ ast.MatchAs( name = None, pattern = None ) ], kwd_attrs = [], kwd_patterns = [],
				),
				guard = None,
				body = [ exhausted_break ],
			)
		else:
			err_bind_name = f'__yield_from_err_{unique}'
			forward_body: list[ast.stmt] = []
			if error_type.is_rc():
				# err_bind_name's own extraction (below) is an aliasing-read
				# incref - it's never consumed by anything (forward_yield
				# forwards __yield_from_next_N itself, not err_bind_name), so
				# unlike this file's other match-extracted temps it has no
				# re-store to hand its reference off to; same "explicit decref,
				# right where the value stops being needed" treatment _desugar_
				# iterator_for's own remaining_case_body uses, and for the same
				# reason - loop_back_edge() would otherwise schedule it for the
				# loop's own back edge, past forward_yield's own suspend, in a
				# separate $$__resume__ call where this plain local no longer
				# exists
				forward_body.append( _expr_stmt( ast.Call(
					func = ast.Attribute( value = _id( 'compiler' ), attr = 'decref', ctx = ast.Load() ),
					args = [ ast.Name( id = err_bind_name, ctx = ast.Load() ) ], keywords = [],
				)))
			forward_body.append( forward_yield() )
			inner_match = ast.Match(
				subject = ast.Name( id = err_bind_name, ctx = ast.Load() ),
				cases = [
					ast.match_case(
						pattern = ast.MatchClass(
							cls = ast.Name( id = 'StopIteration', ctx = ast.Load() ),
							patterns = [ ast.MatchAs( name = None, pattern = None ) ], kwd_attrs = [], kwd_patterns = [],
						),
						guard = None,
						body = [ exhausted_break ],
					),
					ast.match_case(
						pattern = ast.MatchAs( name = None, pattern = None ), guard = None,
						body = forward_body,
					),
				],
			)
			ast.copy_location( inner_match, s )
			err_case = ast.match_case(
				pattern = ast.MatchClass(
					cls = ast.Attribute( value = ast.Name( id = 'Result', ctx = ast.Load() ), attr = 'Err', ctx = ast.Load() ),
					patterns = [ ast.MatchAs( name = err_bind_name ) ], kwd_attrs = [], kwd_patterns = [],
				),
				guard = None,
				body = [ inner_match ],
			)
		match_stmt = ast.Match( subject = ast.Name( id = next_name, ctx = ast.Load() ), cases = [ err_case, ok_case ] )
		ast.copy_location( match_stmt, s )

		while_node = ast.While( test = ast.Constant( value = True ), body = [ next_assign, match_stmt ], orelse = [] )
		ast.copy_location( while_node, s )
		ast.fix_missing_locations( while_node )
		ast.fix_missing_locations( obj_init )
		return [ obj_init, while_node ]

	def _assign_generator_yield_dispatch( self, fn: Function ) -> list[tuple[int,str]]:
		''' PLAN_GENERATORS.md Phase F - replaces the old AST-synthesis
		unit-matcher (_collect_generator_units/_split_generator_segments/
		_build_yield_unit_guard/_build_while_unit_guard/_build_if_unit_
		guard, all deleted) with a real IR-level dispatch: every ast.Yield
		reachable anywhere in the body (arbitrary depth - if-in-while,
		while-in-if, elif chains, multiple yields per loop/branch, all of
		which the old unit model rejected outright) gets a unique
		sequential dispatch state (starting at 1 - state 0 means "not yet
		started") and a fresh resume label, tagged directly onto the node
		(`generator_yield_state`/`generator_resume_label`) so lowering.py's
		ordinary statement/expression pipeline can build `ir.Yield`
		in-place once it reaches each one, and so
		FunctionLowering._emit_generator_dispatch_prologue (built from the
		returned list, cached on the FunctionDef itself as `.
		generator_yield_states`) knows every resume target up front,
		before the body it jumps INTO has been lowered at all - safe
		because ir.Jump/ir.Label targets are plain string labels, resolved
		at emission time, not requiring the target to already exist.
		Walks fn.node.body in the SAME program-order _walk_generator_body
		every other generator pass already relies on, so two yields at
		the same nesting depth still get states in textual left-to-right/
		outer-to-inner order (not that dispatch correctness depends on
		ORDER - each state is independently unique - but a stable,
		predictable order keeps a dumped instruction stream readable). '''
		states: list[tuple[int,str]] = []
		state = 0
		for n in self._walk_generator_body( fn.node.body ):
			if isinstance( n, ast.Yield ):
				state += 1
				label = f'__gen_resume_{state}'
				n.generator_yield_state = state
				n.generator_resume_label = label
				states.append( ( state, label ) )
		return states

	def _collect_generator_locals( self, fn: Function ) -> dict[str,Type]:
		''' every local assigned anywhere in the body becomes a field - see
		this section's own docstring above. A local's TYPE comes from its
		own first `x: T = ...` annotated assignment (required - a generator
		local can't rely on plain-assignment type inference); a later plain
		`x = ...` reassignment is fine once `x` is already declared.

		PLAN_GENERATORS.md Phase 5 (roadmap Phase 5) lifted the v1 scalar-
		only restriction here (previously: only bool/integer types were
		accepted) - any type is now allowed, including RC-typed ones. An
		RC-typed promoted local gets its own "live" companion field (see
		_build_generator_backing_class) so the generator's own state/flag-
		gated destructor (_build_generator_destructor) only ever decrefs
		it once it's actually been assigned - the exact problem that made
		v1 restrict this in the first place (an unconditional decref of a
		not-yet-initialized field would touch garbage). '''
		param_stems = { p.stem for p in fn.parameters or [] }
		locals_decl: dict[str,Type] = {}
		for node in self._walk_generator_body( fn.node.body ):
			if isinstance( node, ast.AnnAssign ) and isinstance( node.target, ast.Name ):
				stem = node.target.id
				if stem in param_stems:
					self.discovery.fail( f'{fn.qualname}: generator local {stem!r} has the same name as a parameter', node )
				local_type = self.discovery.visit( node.annotation )
				if stem in locals_decl and locals_decl[stem] is not local_type:
					self.discovery.fail( f'{fn.qualname}: generator local {stem!r} redeclared with a different type', node )
				locals_decl[stem] = local_type
			elif isinstance( node, ast.Assign ) and not getattr( node, 'compiler_synthesized_for_loop_temp', False ):
				for target in node.targets:
					if isinstance( target, ast.Name ) and target.id not in param_stems and target.id not in locals_decl:
						self.discovery.fail(
							f'{fn.qualname}: generator local {target.id!r} must be declared with an explicit type '
							f'annotation (`{target.id}: T = ...`) before first use',
							node,
						)
		return locals_decl

	def _live_flag_stem( self, local_stem: str ) -> str:
		''' PLAN_GENERATORS.md Phase 5 (roadmap Phase 5) - the companion
		boolean field name for an RC-typed promoted local, tracking
		whether it's actually been assigned yet (see _build_generator_
		backing_class/_build_generator_destructor's own docstrings). Never
		used for a scalar/non-RC local (nothing to gate - see is_rc). '''
		return f'__{local_stem}_live'

	def _build_generator_backing_class( self, fn: Function, locals_decl: dict[str,Type], defer_sites: list[tuple[str,bool,list[ast.stmt]]], send_type: 'Type|None' = None ) -> RCClass:
		''' the per-function backing RCClass a generator's constructor
		allocates and its own $$__next__/$$__resume__ method operates on -
		fields: `__state` (resume discriminant) + one per parameter + one
		per promoted local (_collect_generator_locals, which includes
		Phase-1 for-loop-desugaring locals like __for_obj_N - the iterated
		expression a non-range() for-loop needs, re-evaluated every time
		the loop is reached, not just once - see _new_for_obj_field's own
		docstring) + one `__<stem>_live: bool` companion field
		per RC-typed promoted LOCAL (PLAN_GENERATORS.md Phase 5/roadmap
		Phase 5 - NOT for parameters, which stay always-valid
		from construction onward, unchanged) + one `__defer_armed_N: bool`
		field per defer/errdefer site (PLAN_GENERATORS.md's defer/errdefer
		phase - see _desugar_generator_defer_sites) + (PLAN_GENERATORS.md
		Phase C, send_type not None) `__send_slot: send_type` - treated
		EXACTLY like an RC-typed promoted local (its own `__send_slot_live`
		companion field when send_type.is_rc(), same live-flag-guarded
		reassignment/zero-placeholder-construction/destructor-teardown
		machinery, no new RC design needed - see _build_generator_send_
		wrappers) - plus `__send_ready: bool`, a SEPARATE protocol flag
		(armed by send(), consumed+cleared by the next captured-yield
		resume - see lowering.py's _expr_Yield) that has nothing to do with
        whether __send_slot has ever been assigned. resolve=None/every
		attribute's own resolve=None (mirrors tuple_storage.TupleStorage.
		get()'s identical "already fully known, nothing to defer" shape).
		Unlike every other RCClass, this one's own $$__destructor__ is
		NOT built by compiler.py's ordinary, unconditional RCClass
		handling - ensure_generator_synthesized pre-marks it as already
		synthesized and builds a state/flag-gated one itself
		(_build_generator_destructor) instead, since an RC-typed
		promoted local is only conditionally valid (see that method's own
		docstring for why the unconditional cascade would be wrong here). '''
		usize_cls = self.discovery.get_intrinsics()['usize']
		bool_cls = self.discovery.get_intrinsics()['bool']
		qualname = f'{fn.qualname}$$generator'
		state_attr = Variable( stem = '__state', qualname = f'{qualname}.__state', file = fn.file, line = fn.line, type = usize_cls )
		param_attrs = [
			Variable( stem = p.stem, qualname = f'{qualname}.{p.stem}', file = fn.file, line = fn.line, type = p.type )
			for p in fn.parameters or []
		]
		local_attrs = [
			Variable( stem = stem, qualname = f'{qualname}.{stem}', file = fn.file, line = fn.line, type = t )
			for stem, t in locals_decl.items()
		]
		live_flag_attrs = [
			Variable( stem = self._live_flag_stem( stem ), qualname = f'{qualname}.{self._live_flag_stem( stem )}', file = fn.file, line = fn.line, type = bool_cls )
			for stem, t in locals_decl.items() if t.is_rc()
		]
		defer_armed_attrs = [
			Variable( stem = flag_stem, qualname = f'{qualname}.{flag_stem}', file = fn.file, line = fn.line, type = bool_cls )
			for flag_stem, _is_errdefer, _body in defer_sites
		]
		send_attrs: list[Variable] = []
		if send_type is not None:
			send_attrs.append( Variable( stem = '__send_slot', qualname = f'{qualname}.__send_slot', file = fn.file, line = fn.line, type = send_type ) )
			if send_type.is_rc():
				send_attrs.append( Variable( stem = '__send_slot_live', qualname = f'{qualname}.__send_slot_live', file = fn.file, line = fn.line, type = bool_cls ) )
			send_attrs.append( Variable( stem = '__send_ready', qualname = f'{qualname}.__send_ready', file = fn.file, line = fn.line, type = bool_cls ) )
		attributes = [ state_attr ] + param_attrs + local_attrs + live_flag_attrs + defer_armed_attrs + send_attrs
		return RCClass(
			stem = qualname, qualname = qualname, file = fn.file, line = fn.line,
			base = None, type_params = None,
			attributes = attributes, methods = [], names = { a.stem: a for a in attributes },
			resolve = None,
		)

	def _assigned_self_attr_stem( self, stmt: ast.stmt ) -> 'str|None':
		''' PLAN_GENERATORS.md Phase 5 (roadmap Phase 5) - after
		_GeneratorNameRenamer has already run, an assignment TO a
		promoted field looks like `self.<stem> = ...` (Assign) or
		`self.<stem>: T = ...` (AnnAssign, a promoted local's own FIRST/
		declaring occurrence - already valid, working AST shape today for
		every scalar local this whole plan has synthesized so far, see
		_rename_and_track_liveness's own docstring). Returns the stem, or
		None if `stmt` isn't (post-rename) an assignment into a self
		attribute at all. '''
		if isinstance( stmt, ast.Assign ) and len( stmt.targets ) == 1:
			target = stmt.targets[0]
		elif isinstance( stmt, ast.AnnAssign ):
			target = stmt.target
		else:
			return None
		if isinstance( target, ast.Attribute ) and isinstance( target.value, ast.Name ) and target.value.id == 'self':
			return target.attr
		return None

	def _rename_and_track_liveness( self, stmts: list[ast.stmt], renamer: '_GeneratorNameRenamer', rc_local_stems: set ) -> list[ast.stmt]:
		''' PLAN_GENERATORS.md Phase 5 (roadmap Phase 5) - renames each
		statement (same as the old bare `[renamer.visit(s) for s in
		stmts]` every guard builder used before this phase), and splits
		any assignment INTO an RC-typed promoted local into:

			if self.__<stem>_live:
				self.<stem> = <value>      # unchanged - ordinary field
				                            # reassignment lowering reads
				                            # the CURRENT value and decrefs
				                            # it before storing, correct
				                            # here since it's a real prior
				                            # object
			else:
				self.<stem> = <value>      # tagged generator_first_rc_
				                            # assign (see lowering.py's
				                            # _stmt_Assign) - the field's
				                            # current value is still the
				                            # construction-time placeholder
				                            # (a bare `0`), so this skips
				                            # the read-old-value-and-decref
				                            # step entirely instead of
				                            # computing &(NULL)->$header,
				                            # a real, confirmed UBSan trap
				self.__<stem>_live = True

		A single occurrence in the SOURCE (this method only ever sees one
		AST node per assignment - _GeneratorNameRenamer already turned the
		promoted local's own declaring `x: T = expr` into a plain `self.x
		= expr`, indistinguishable from a later plain reassignment - see
		_GeneratorNameRenamer.visit_AnnAssign's own docstring) can run at
		RUNTIME any number of times if it's inside a loop - the live flag,
		not source position, is what actually determines whether a given
		DYNAMIC execution of this statement is the first one ever
		(confirmed via a real repro: `b: Box = Box(v=100)` inside a while
		loop, yielded each iteration - the textually-first-and-only
		occurrence in the source runs once per iteration at runtime, so a
		static "first occurrence = no decref" AST-position rule would be
		wrong starting with the second iteration). The value expression is
		deep-copied (not shared) between the two branches so each has its
		own independent AST node - safe at this pre-lowering stage (no
		resolved_*/generator_first_rc_assign-adjacent attributes attached
		to either copy yet), and correct at runtime since only one branch
		ever actually executes per statement instance, so the expression
		is still evaluated exactly once.

		rc_local_stems is empty for a generator with no RC-typed promoted
		locals at all - a no-op then, identical to the old bare rename.

		PLAN_GENERATORS.md Phase F - `stmts` is now the WHOLE (or a whole
		nested if/while/for/with block's) statement list, not just one
		unit's own flat pre/post-yield slice, so the live-flag-guard step
		is applied via a SEPARATE recursive pass (_apply_live_flag_
		guards, below) over the already-fully-renamed tree, rather than
		inline here - renamer.visit(s) on a compound statement already
		renames its ENTIRE subtree in one call (ast.NodeTransformer's
		default generic_visit recurses), so re-visiting a child with
		renamer again here would be redundant; the live-flag-guard
		transform, by contrast, has to run AFTER renaming (it matches on
		the RENAMED self.<stem>=... shape) and has to recurse independently
		to reach an RC-reassignment buried inside a nested block. '''
		renamed = [ renamer.visit( s ) for s in stmts ]
		if not rc_local_stems:
			return renamed
		hoisted = self._hoist_yield_from_rc_reassignment( renamed, rc_local_stems )
		return self._apply_live_flag_guards( hoisted, rc_local_stems )

	def _hoist_yield_from_rc_reassignment( self, stmts: list[ast.stmt], rc_local_stems: set ) -> list[ast.stmt]:
		''' PLAN_GENERATORS.md Phase C - _apply_live_flag_guards (below)
		deep-copies the WHOLE statement for its own "first assignment"
		branch - safe for an ordinary value expression (only one of the
		two branches ever actually RUNS per dynamic execution, so a
		duplicated Call/constructor still only executes once), but NOT
		for a yield: it's a real suspend point, so duplicating it creates
		TWO independent (state, resume_label) dispatch targets for what
		must be ONE textual yield site - confirmed via a real repro
		("redefinition of label" - a real C compile error, not just a
		latent correctness gap). Runs BEFORE _apply_live_flag_guards
		(after _assign_generator_yield_dispatch has already tagged every
		yield with its own state/resume_label - those tags travel with
		the node wherever it moves) and recurses the same way that does.

		Scoped to the DIRECT case only - `self.<stem> = (yield expr)`,
		the entire RHS is the captured yield, exactly what `.send()`
		naturally looks like (`held: Box = yield i`) - restructured into
		`__gen_send_capture_N = (yield expr); self.<stem> = __gen_send_
		capture_N`, so the yield now appears exactly once, textually and
		state-wise, and _apply_live_flag_guards only ever deep-copies the
		cheap re-store afterward. The capture assignment is tagged is_
		match_subject (same "no independent tracked ownership, just
		borrows the ALREADY-correctly-increfed source" treatment
		visit_Match's own __match_subj_N relay already gets - see
		lowering.py's _stmt_Assign) - _expr_Yield's own returned operand
		needs no incref of its OWN (a plain field read of self.__
		send_slot), relying entirely on the capture assignment's own
		is_alias=True (now that _is_aliasing_expr recognizes a captured
		ast.Yield) to do it; the capture temp then just relays that SAME
		single reference into the re-store below via an ordinary Name
		read (aliasing by the general rule, needing no special-casing at
		all there). A yield embedded deeper inside a LARGER expression
		assigned to an RC-typed promoted local (`held = (yield i) if
		flag else other`) is rejected instead of guessed at - same "start
		narrow" posture PLAN_GENERATORS.md applies elsewhere; no forcing
		use case for the general form yet. '''
		result: list[ast.stmt] = []
		for s in stmts:
			stem = self._assigned_self_attr_stem( s )
			value = s.value if isinstance( s, ast.Assign ) else None
			if stem is not None and stem in rc_local_stems and isinstance( value, ast.Yield ):
				capture_name = f'__gen_send_capture_{self._gen_send_capture_counter}'
				self._gen_send_capture_counter += 1
				capture_assign = ast.Assign( targets = [ ast.Name( id = capture_name, ctx = ast.Store() ) ], value = value )
				capture_assign.is_match_subject = True
				ast.copy_location( capture_assign, s )
				ast.fix_missing_locations( capture_assign )
				restore_assign = ast.Assign( targets = s.targets, value = ast.Name( id = capture_name, ctx = ast.Load() ) )
				ast.copy_location( restore_assign, s )
				ast.fix_missing_locations( restore_assign )
				result.append( capture_assign )
				result.append( restore_assign )
				continue
			if stem is not None and stem in rc_local_stems and any( isinstance( n, ast.Yield ) for n in ast.walk( value ) if value is not None ):
				self.discovery.fail(
					f'a yield embedded inside a larger expression assigned to an RC-typed generator local is not supported yet '
					f'(`{stem} = yield expr` directly is fine) - see PLAN_GENERATORS.md',
					s,
				)
				continue
			if isinstance( s, ( ast.If, ast.While, ast.For ) ):
				s.body = self._hoist_yield_from_rc_reassignment( s.body, rc_local_stems )
				s.orelse = self._hoist_yield_from_rc_reassignment( s.orelse, rc_local_stems )
			elif isinstance( s, ast.With ):
				s.body = self._hoist_yield_from_rc_reassignment( s.body, rc_local_stems )
			result.append( s )
		return result

	def _apply_live_flag_guards( self, stmts: list[ast.stmt], rc_local_stems: set ) -> list[ast.stmt]:
		''' PLAN_GENERATORS.md Phase F - the live-flag-guard half of
		_rename_and_track_liveness's own old docstring (see above),
		generalized to recurse into nested if/while/for/with bodies -
		same "_recurse_*_wrap" shape PLAN_GENERATORS.md's own Phase F/A.4a
		write-up describes reusing for defer-tagging/bare-return-rewriting/
		for-loop-desugaring elsewhere in this file. Operates on an
		ALREADY-RENAMED tree (never calls renamer itself) - see caller. '''
		result: list[ast.stmt] = []
		for s in stmts:
			stem = self._assigned_self_attr_stem( s )
			if stem is not None and stem in rc_local_stems:
				already_live = s
				first_time = copy.deepcopy( s )
				first_time.generator_first_rc_assign = True
				flag_assign = ast.Assign(
					targets = [ self._self_attr( self._live_flag_stem( stem ), s ) ],
					value = ast.Constant( value = True ),
				)
				ast.copy_location( flag_assign, s )
				guard = ast.If(
					test = self._self_attr( self._live_flag_stem( stem ), s ),
					body = [ already_live ],
					orelse = [ first_time, flag_assign ],
				)
				ast.copy_location( guard, s )
				result.append( guard )
				continue
			if isinstance( s, ( ast.If, ast.While, ast.For ) ):
				s.body = self._apply_live_flag_guards( s.body, rc_local_stems )
				s.orelse = self._apply_live_flag_guards( s.orelse, rc_local_stems )
			elif isinstance( s, ast.With ):
				s.body = self._apply_live_flag_guards( s.body, rc_local_stems )
			result.append( s )
		return result

	# PLAN_GENERATORS.md Phase 5 (roadmap Phase 5) used to route every
	# RC-typed yielded value through two chained intermediate locals here
	# (_maybe_route_yield_through_temp) to dodge three real, generator-
	# unrelated RC bugs in how a bare value coerces into a declared union
	# return type (elem_type|None, exactly __next__'s own shape):
	# `return self.<field>`, `return <bare tracked value>`, and an
	# AnnAssign coercing a field read into a union local all under- or
	# over-counted the incref the union-member constructor already does
	# internally. All three were root-caused and fixed by 048af0f ("Fix
	# double-incref/masked-decref when coercing a value into a union
	# type" - lowering.py's _is_aliasing_expr now checks the actual
	# coerced operand via _coerce_into_union's own is_union_coerce_result
	# tag, not the pre-coercion ast node) - a plain `return <yielded>`
	# through elem_type|None now increfs exactly once regardless of
	# whether `yielded` is a field read, a tracked local, or a bare
	# parameter, confirmed both by 048af0f's own union_coercion_rc_test.py
	# and directly against this exact self.<field>-return shape. The
	# routing (and the elem_type/elem_is_rc plumbing that only ever fed
	# it) was removed once that was confirmed - yield sites below just
	# return the renamed value straight through.

	def _build_generator_next_function( self, fn: Function, backing_cls: RCClass, locals_decl: dict[str,Type], next_return_type: Type, pending_bare_return_assigns: 'list[ast.Assign]', defer_sites: list[tuple[str,bool,list[ast.stmt]]], send_type: 'Type|None' = None ) -> Function:
		''' PLAN_GENERATORS.md Phase F - builds $$__next__: self.__state ==
		DONE short-circuits to `return None`, then the generator's own
		body, lowered essentially AS-IS (structurally intact - no more
		per-shape unit guards; the earlier AST-synthesis unit-matcher this
		replaced is gone, see this section's own top-of-file docstring),
		then a tail (natural-exhaustion exit: armed plain-`defer` replay +
		self.__state = DONE; return None). Every ast.Yield reachable
		anywhere in the body already carries its own dispatch state/resume
		label by the time this runs (_assign_generator_yield_dispatch,
		below) - lowering.py's ordinary statement pipeline turns each one
		into a real `ir.Yield` in place (arbitrary nesting depth composes
		for free, no CFG/merge_if changes needed - a yield doesn't unwind
		or fork anything, see ir.Yield's own docstring), and
		FunctionLowering._emit_generator_dispatch_prologue builds the
		real state-check-and-goto dispatch AT LOWERING TIME from node.
		generator_yield_states (tagged onto the FunctionDef below) - a
		real goto/switch has no AST spelling in Python, so that piece
		can't be synthesized here the way everything else in this method
		still is.

		next_return_type: PLAN_GENERATORS.md's StopIteration reversal -
		every generator is unconditionally fallible now (its own error_type
		always includes StopIteration, never None - see discovery.py's
		visit_Subscript), so next_return_type is always Result[elem_type,
		error_type], never a bare elem_type|None union.
		The old AST-level pessimistic-done pre-write this method used to
		orchestrate per-unit (_pessimistic_done_prefix) is gone too - Phase
		F re-derives it at the LOWERING level instead (lowering.py's
		_consume_checked_result appends a SetAttr(self.__state,done_state)
		into every OrReturn's own epilogue whenever self._current_fn.
		is_generator_next, reusing the exact hook Mechanism 2's own
		error-defer replay already established there), so this method
		doesn't need to know or care where a fallible operation might be
		reached from anymore. Every ast.Return AND ast.Yield in the
		assembled body then gets its value wrapped in Result.Ok(...) - EXCEPT
		a tagged synthesized exhaustion return, which wraps into
		Result.Err(StopIteration()) instead - see _wrap_generator_next_
		returns_in_ok. '''
		rename_targets = { p.stem for p in fn.parameters or [] } | set( locals_decl.keys() )
		renamer = _GeneratorNameRenamer( rename_targets )

		rc_local_stems = { stem for stem, t in locals_decl.items() if t.is_rc() }

		# PLAN_GENERATORS.md's defer/errdefer phase (Mechanism 2) - a
		# SEPARATE, already-renamed copy of defer_sites, used ONLY for
		# tagging (_tag_armed_defer_sites, below) - lowering.py's own
		# Mechanism 2 hook (_build_generator_error_defer_replay) has no
		# access to this file's _GeneratorNameRenamer, so unlike
		# Mechanism 1's own _build_defer_replay_guards (which embeds the
		# RAW body and relies on a LATER renamer.visit(...)/_rename_and_
		# track_liveness call to cover it - see that method's own
		# docstring), the tag itself has to already carry self.<field>-
		# qualified statements. Rendered ONCE here (not per insertion
		# site - a single canonical copy is enough for tagging;
		# lowering.py's own hook deep-copies its own fresh instance per
		# OrReturn site that actually consumes it).
		rendered_defer_sites: list[tuple[str,bool,list[ast.stmt]]] = [
			( flag_stem, is_errdefer, [ renamer.visit( copy.deepcopy( s )) for s in body_stmts ] )
			for flag_stem, is_errdefer, body_stmts in defer_sites
		]

		# tag the WHOLE original (un-renamed - the tag values themselves
		# are the pre-rendered copy above, but the NODES being tagged are
		# still the raw body, walked in its own natural program order)
		# body in ONE pass now: Phase F removed the unit/segment split
		# that used to make "top-level" mean "this particular preamble/
		# tail slice" - it now means exactly what _validate_generator_
		# defer_sites already always meant by it, fn.node.body's own
		# direct children (see _tag_armed_defer_sites' own docstring:
		# tagging only the top-level statement of a slice is sufficient,
		# lowering.py's own push/pop keeps it active for that whole
		# statement's recursive lowering - unaffected by this
		# generalization from many small slices to one whole-body slice)
		armed_count: list[int] = [ 0 ]
		if defer_sites:
			self._tag_armed_defer_sites( fn.node.body, rendered_defer_sites, armed_count )

		yield_states = self._assign_generator_yield_dispatch( fn )
		done_state = len( yield_states ) + 1
		for pending in pending_bare_return_assigns:
			pending.value = ast.Constant( value = done_state )

		body_stmts = self._rename_and_track_liveness( fn.node.body, renamer, rc_local_stems )

		anchor = fn.node.body[-1] if fn.node.body else fn.node
		# PLAN_GENERATORS.md's defer/errdefer phase - natural exhaustion is
		# a real generator-ending exit like any other, so every currently-
		# armed plain `defer` site replays here too (LIFO), right before
		# the state gets pinned to done
		defer_replay = self._rename_and_track_liveness( self._build_defer_replay_guards( defer_sites, anchor ), renamer, rc_local_stems )
		tail_exhaustion_return = ast.Return( value = ast.Constant( value = None ) )
		tail_exhaustion_return.generator_exhaustion_return = True
		tail_body: list[ast.stmt] = defer_replay + [
			ast.Assign( targets = [ self._self_attr( '__state', anchor ) ], value = ast.Constant( value = done_state ) ),
			tail_exhaustion_return,
		]

		done_short_circuit_return = ast.Return( value = ast.Constant( value = None ) )
		done_short_circuit_return.generator_exhaustion_return = True
		next_body: list[ast.stmt] = [
			ast.If(
				test = ast.Compare( left = self._self_attr( '__state', fn.node ), ops = [ ast.Eq() ], comparators = [ ast.Constant( value = done_state ) ] ),
				body = [ done_short_circuit_return ],
				orelse = [],
			),
		]
		next_body.extend( body_stmts )
		next_body.extend( tail_body )

		# PLAN_GENERATORS.md's StopIteration reversal - always run now (every
		# generator is unconditionally fallible, see this method's own
		# docstring)
		self._wrap_generator_next_returns_in_ok( next_body )

		# PLAN_GENERATORS.md Phase C - when SendType is declared
		# (Generator[T,SendType,E]), the real body-bearing method is
		# renamed $$__resume__ (double-dollar, same "never user-callable
		# through ordinary name resolution" convention $$__destructor__
		# already uses) - __next__() and send(v) become thin wrappers
		# over it instead (_build_generator_send_wrappers, called from
		# ensure_generator_synthesized once this returns). Iterator[T]/
		# the 2-arg Generator[T,E] form are completely unaffected -
		# __next__ stays the one real method, exactly as every phase
		# before this one built it.
		method_stem = '$$__resume__' if send_type is not None else '__next__'
		node = ast.FunctionDef(
			name = method_stem,
			args = ast.arguments( posonlyargs = [], args = [], vararg = None, kwonlyargs = [], kw_defaults = [], kwarg = None, defaults = [] ),
			body = next_body, decorator_list = [], returns = None, type_params = [],
			lineno = fn.line or 1, col_offset = 0, end_lineno = fn.line or 1, end_col_offset = 0,
		)
		ast.fix_missing_locations( node )
		node.generator_yield_states = yield_states
		# PLAN_GENERATORS.md Phase C - lowering.py's _expr_Yield reads this
		# back (via self._current_fn.node) to know whether a captured
		# `(yield expr)` is even legal here at all (only when SendType is
		# declared) and, if so, what type to deliver it as
		node.generator_send_type = send_type

		next_fn = Function(
			stem = method_stem, qualname = f'{backing_cls.qualname}.{method_stem}', file = fn.file, line = fn.line,
			cls = backing_cls, node = node,
			parameters = [], return_type = next_return_type,
			is_static = False, resolve = None,
			is_generator_next = True,
		)
		# PLAN_GENERATORS.md Phase F - unlike the old unit-matcher (which
		# rewrote every ast.Yield into a plain ast.Return before this
		# point, so $$__next__'s own assembled body never contained one),
		# next_fn.node.body now embeds the REAL ast.Yield nodes directly -
		# _function_contains_yield would otherwise see them and treat
		# $$__next__ itself as an unrelated generator needing its OWN
		# synthesis (confirmed via a real repro: "a generator method is
		# not supported yet", ensure_generator_synthesized reached with
		# fn.cls already set). Pre-marking id(next_fn) here, the same way
		# id(fn) itself gets marked at the top of ensure_generator_
		# synthesized, makes that check a no-op the moment anything
		# (lowering.py's own safety-net call, in particular) reaches it.
		self._generators_synthesized.add( id( next_fn ))
		backing_cls.methods.append( next_fn )
		backing_cls.names[ next_fn.stem ] = next_fn
		return next_fn

	def _build_generator_send_wrappers( self, fn: Function, backing_cls: RCClass, resume_fn: Function, send_type: Type, next_return_type: Type ) -> None:
		''' PLAN_GENERATORS.md Phase C - two thin public wrappers delegating
		into $$__resume__ (built separately, see _build_generator_next_
		function's own docstring): `__next__()` (leaves __send_ready
		untouched - a captured yield resumed this way sees __send_ready
		still False and panics, via _expr_Yield, pointing at .send()
		instead) and `send(v)` (panics via sys.panic() if self.__state ==
		0, mirroring Python's own TypeError for sending before the first
		yield; otherwise arms __send_slot/__send_ready, then resumes).
		Both are ordinary Attribute-call syntax (`self.$$__resume__()`),
		resolved through the SAME generic _attr_lookup_callable/
		_find_method machinery any other self.<method>() call already
		uses - backing_cls.names['$$__resume__'] already has a real entry
		(the caller already registered it), so this needs no resolved_
		callee escape hatch at all, unlike sys.free(self)'s own call in
		_build_generator_destructor (a receiver-less FREE function). '''
		self_read = lambda attr: ast.Attribute( value = ast.Name( id = 'self', ctx = ast.Load() ), attr = attr, ctx = ast.Load() )
		resume_call = lambda: ast.Call( func = self_read( resume_fn.stem ), args = [], keywords = [] ) # a fresh node per use - see _build_defer_replay_guards' own docstring for why sharing one node object across sites is unsafe (lowering attaches mutable per-occurrence attributes)

		next_node = ast.FunctionDef(
			name = '__next__',
			args = ast.arguments( posonlyargs = [], args = [], vararg = None, kwonlyargs = [], kw_defaults = [], kwarg = None, defaults = [] ),
			body = [ ast.Return( value = resume_call() ) ], decorator_list = [], returns = None, type_params = [],
			lineno = fn.line or 1, col_offset = 0, end_lineno = fn.line or 1, end_col_offset = 0,
		)
		ast.fix_missing_locations( next_node )
		next_fn = Function(
			stem = '__next__', qualname = f'{backing_cls.qualname}.__next__', file = fn.file, line = fn.line,
			cls = backing_cls, node = next_node,
			parameters = [], return_type = next_return_type,
			is_static = False, resolve = None,
		)
		self._generators_synthesized.add( id( next_fn )) # same "never a generator of its own" pre-mark as $$__resume__/$$__next__ - see that call site's own comment
		backing_cls.methods.append( next_fn )
		backing_cls.names[ next_fn.stem ] = next_fn

		v_param = Parameter( stem = 'v', qualname = f'{backing_cls.qualname}.send.v', file = fn.file, line = fn.line, type = send_type )
		panic_call = ast.Call(
			func = ast.Attribute( value = ast.Name( id = 'sys', ctx = ast.Load() ), attr = 'panic', ctx = ast.Load() ),
			args = [ ast.Constant( value = f'{fn.qualname}: cannot send a value before the first yield' ) ], keywords = [],
		)
		not_started_guard = ast.If(
			test = ast.Compare( left = self_read( '__state' ), ops = [ ast.Eq() ], comparators = [ ast.Constant( value = 0 ) ] ),
			body = [ ast.Expr( panic_call ) ], orelse = [],
		)
		v_read = lambda: ast.Name( id = 'v', ctx = ast.Load() ) # fresh node per use, same reasoning as resume_call above
		if send_type.is_rc():
			# __send_slot is treated exactly like an RC-typed promoted
			# local's own first-or-later reassignment - same live-flag-
			# guard shape _apply_live_flag_guards builds for one, hand-
			# built here directly since send()'s own body isn't part of
			# the user's original generator body that pass ever walks
			already_live = ast.Assign( targets = [ self_read( '__send_slot' ) ], value = v_read() )
			first_time = ast.Assign( targets = [ self_read( '__send_slot' ) ], value = v_read() )
			first_time.generator_first_rc_assign = True
			flag_assign = ast.Assign( targets = [ self_read( '__send_slot_live' ) ], value = ast.Constant( value = True ) )
			send_slot_assign = [ ast.If( test = self_read( '__send_slot_live' ), body = [ already_live ], orelse = [ first_time, flag_assign ] ) ]
		else:
			send_slot_assign = [ ast.Assign( targets = [ self_read( '__send_slot' ) ], value = v_read() ) ]
		ready_assign = ast.Assign( targets = [ self_read( '__send_ready' ) ], value = ast.Constant( value = True ) )
		send_body: list[ast.stmt] = [ not_started_guard ] + send_slot_assign + [ ready_assign, ast.Return( value = resume_call() ) ]
		send_node = ast.FunctionDef(
			name = 'send',
			args = ast.arguments( posonlyargs = [], args = [ ast.arg( arg = 'v' ) ], vararg = None, kwonlyargs = [], kw_defaults = [], kwarg = None, defaults = [] ),
			body = send_body, decorator_list = [], returns = None, type_params = [],
			lineno = fn.line or 1, col_offset = 0, end_lineno = fn.line or 1, end_col_offset = 0,
		)
		ast.fix_missing_locations( send_node )
		send_fn = Function(
			stem = 'send', qualname = f'{backing_cls.qualname}.send', file = fn.file, line = fn.line,
			cls = backing_cls, node = send_node,
			parameters = [ v_param ],
			return_type = next_return_type,
			is_static = False, resolve = None,
		)
		send_fn.add_name( 'v', v_param )
		self._generators_synthesized.add( id( send_fn ))
		backing_cls.methods.append( send_fn )
		backing_cls.names[ send_fn.stem ] = send_fn

	def _wrap_generator_next_returns_in_ok( self, next_body: list[ast.stmt] ) -> None:
		''' PLAN_GENERATORS.md's StopIteration reversal - $$__next__ always
		declares -> Result[elem_type,error_type] now, so every `return
		<value>`/`yield <value>` reachable anywhere in the body (PLAN_
		GENERATORS.md Phase F - lowering.py's own yield-lowering coerces
		ast.Yield.value against self._current_fn.return_type exactly like
		_stmt_Return already coerces its own value, so wrapping it here,
		the SAME uniform way, needs zero special-casing there) needs to
		become `Result.Ok(<value>)` - EXCEPT a node tagged generator_
		exhaustion_return (the DONE short-circuit, the tail's own natural
		exhaustion, and a user-written bare `return`/`return None` - see
		_build_generator_next_function/_rewrite_bare_return_stmts, the
		three sites that set this tag), which becomes `Result.Err(
		StopIteration())` instead: reaching the end of the generator is no
		longer a nullable None bundled into the success channel, it's a
		real Err in the existing error channel. Run once, after the WHOLE
		body is assembled, rather than threading Result-wrapping through
		individual construction sites - simpler, and correct because
		$$__next__ can never contain a nested def/lambda (generator
		bodies already reject those), so a plain ast.walk (no "don't
		recurse into a nested scope" concern, unlike _walk_generator_body
		elsewhere in this file) safely reaches every ast.Return/ast.Yield
		belonging to THIS function. Result.Ok(...)/Result.Err(...)'s own
		payload argument is coerced the ordinary way (same _lower_expr(arg,
		expected_type) machinery any other call argument gets) - so this
		never needs to know what shape a non-exhaustion `value` already
		is. '''
		for stmt in next_body:
			for n in ast.walk( stmt ):
				if isinstance( n, ( ast.Return, ast.Yield )):
					if getattr( n, 'generator_already_result_shaped', False ):
						# _desugar_one_yield_from's own forwarding yield -
						# already exactly Result[elem_type,error_type]-shaped
						# (the exact-match check there guarantees it), so
						# wrapping it in ANOTHER Ok(...) here would produce
						# Ok(Result[...]) instead of Result[...] itself
						continue
					if getattr( n, 'generator_exhaustion_return', False ):
						assert n.value is not None and isinstance( n.value, ast.Constant ) and n.value.value is None, (
							f'exhaustion-tagged node with an unexpected non-None value: {ast.dump(n)}'
						)
						err_call = ast.Call(
							func = ast.Attribute( value = ast.Name( id = 'Result', ctx = ast.Load() ), attr = 'Err', ctx = ast.Load() ),
							args = [ ast.Call( func = ast.Name( id = 'StopIteration', ctx = ast.Load() ), args = [], keywords = [] ) ],
							keywords = [],
						)
						ast.copy_location( err_call, n )
						ast.copy_location( err_call.func, n )
						ast.copy_location( err_call.func.value, n )
						ast.copy_location( err_call.args[0], n )
						n.value = err_call
						continue
					value = n.value if n.value is not None else ast.Constant( value = None )
					ok_call = ast.Call(
						func = ast.Attribute( value = ast.Name( id = 'Result', ctx = ast.Load() ), attr = 'Ok', ctx = ast.Load() ),
						args = [ value ], keywords = [],
					)
					ast.copy_location( ok_call, n )
					ast.copy_location( ok_call.func, n )
					ast.copy_location( ok_call.func.value, n )
					n.value = ok_call

	def _build_generator_destructor( self, fn: Function, backing_cls: RCClass, locals_decl: dict[str,Type], defer_sites: list[tuple[str,bool,list[ast.stmt]]], send_type: 'Type|None' = None ) -> None:
		''' PLAN_GENERATORS.md Phase 5 (roadmap Phase 5) - a generator's
		backing class does NOT get the ordinary, unconditional
		$$__destructor__ cascade _synthesize_rcclass_destructor builds
		for every other RCClass: a promoted LOCAL is only known-
		initialized once it's actually been assigned (an RC-typed one
		crossing a yield might never have been reached if the generator
		is dropped before its first assignment - e.g. mid-way through an
		EARLIER unit, or before an if-unit's non-taken branch ever runs),
		so an unconditional decref would touch garbage - the exact
		problem v1 sidestepped by restricting promoted locals to scalar
		types in the first place (see this section's own top docstring).

		Structurally this mirrors _synthesize_rcclass_destructor closely
		(same 3-part shape: no self.__del__() here though - a generator's
		backing class is entirely compiler-synthesized, never user-
		declared, so there's no user __del__ to call; field cascade;
		sys.free(self)) but gates each RC-typed promoted local's own
		teardown behind `if self.__<stem>_live:` (see _build_generator_
		backing_class/_rename_and_track_liveness for how that field gets
		declared and set). Parameters stay UNCONDITIONAL, unchanged from
		every earlier phase - valid from construction onward, same
		reasoning as always (Phase 1's __for_obj_N used to get this same
		unconditional treatment too, back when it was eagerly constructed
		once; now that it's an ordinary re-derived-per-loop-entry promoted
		local like any other - see _new_for_obj_field's own docstring - it
		goes through step 2's live-flag-gated teardown below like every
		other RC-typed promoted local, not this unconditional one).
		backing_cls has no base (never subclassed - PLAN_
		GENERATORS.md's synthesized classes are always leaves), so unlike
		_synthesize_rcclass_destructor this never needs to walk an
		inheritance chain.

		PLAN_GENERATORS.md's defer/errdefer phase - dropping the generator
		mid-iteration (abandonment, never reaching exhaustion or a bare
		return) is ALSO a real generator-ending exit, so every currently-
		armed plain `defer` site replays here too (LIFO), BEFORE step 1's
		own field teardown - not after, and not interleaved: under the
		preamble/tail-only restriction (_validate_generator_defer_sites),
		every defer site can only ever reference a local/parameter declared
		BEFORE it in program order, so running every armed defer body
		first, then the existing decref cascade, guarantees nothing a
		defer body touches has already been freed.

		Called directly from ensure_generator_synthesized, which also
		pre-marks id(backing_cls) in self._destructors_synthesized so
		compiler.py's own ordinary, unconditional RCClass handling (which
		would otherwise also try to build one) becomes a no-op for it -
		see that call site's own comment. '''
		self._destructors_synthesized.add( id( backing_cls ))
		sys_module = self.discovery.modules.get( 'sys' )
		if sys_module is None:
			return  # sys.free must be available
		none_type = self.discovery.get_none_type()
		qualname = f'{backing_cls.qualname}$$__destructor__'

		body: list[ast.stmt] = []

		# 0. defer replay (abandonment) - see this method's own docstring
		# for why it must run before ANY teardown below, not just the RC-
		# local one. Needs its own renamer (destructor otherwise never
		# renames arbitrary user-authored statements - every OTHER
		# statement here is built directly against self.<field>)
		if defer_sites:
			rename_targets = { p.stem for p in fn.parameters or [] } | set( locals_decl.keys() )
			dtor_renamer = _GeneratorNameRenamer( rename_targets )
			rc_local_stems = { stem for stem, t in locals_decl.items() if t.is_rc() }
			body.extend( self._rename_and_track_liveness( self._build_defer_replay_guards( defer_sites, fn.node ), dtor_renamer, rc_local_stems ))

		# 1. captured parameters - unconditional, always valid from
		# construction onward (unchanged from every earlier phase)
		for p in fn.parameters or []:
			attr = backing_cls.get_local_or_raise( p.stem )
			assert isinstance( attr, Variable )
			body.extend( self._build_field_teardown_ast(
				ast.Attribute( value = ast.Name( id = 'self', ctx = ast.Load() ), attr = p.stem, ctx = ast.Load() ),
				attr.type,
			))

		# 2. RC-typed promoted locals - gated behind their own live-flag;
		# non-RC (scalar/CEnum/...) locals need no teardown at all, same
		# as every earlier phase
		for stem, t in locals_decl.items():
			if not t.is_rc():
				continue
			teardown = self._build_field_teardown_ast(
				ast.Attribute( value = ast.Name( id = 'self', ctx = ast.Load() ), attr = stem, ctx = ast.Load() ),
				t,
			)
			if not teardown:
				continue
			guard = ast.If(
				test = self._self_attr( self._live_flag_stem( stem ), fn.node ),
				body = teardown, orelse = [],
			)
			body.append( guard )

		# 2b. PLAN_GENERATORS.md Phase C - __send_slot, treated exactly
		# like an RC-typed promoted local (see _build_generator_backing_
		# class's own docstring): gated behind __send_slot_live, NOT
		# __send_ready (a separate protocol flag that gets cleared as soon
		# as a captured yield consumes a pending send, while __send_slot
		# itself keeps its own independent reference regardless - see
		# lowering.py's _expr_Yield)
		if send_type is not None and send_type.is_rc():
			teardown = self._build_field_teardown_ast(
				ast.Attribute( value = ast.Name( id = 'self', ctx = ast.Load() ), attr = '__send_slot', ctx = ast.Load() ),
				send_type,
			)
			if teardown:
				body.append( ast.If(
					test = self._self_attr( '__send_slot_live', fn.node ),
					body = teardown, orelse = [],
				))

		# 3. sys.free(self) - identical to _synthesize_rcclass_destructor's
		# own ending, see its own comments for why the explicit cast is needed
		free_overload = sys_module.get_local( 'free' )
		from mpy_types import Overload
		if isinstance( free_overload, Overload ):
			free_fn = free_overload.implementations[0]
		else:
			free_fn = free_overload
		if free_fn.resolve is not None:
			free_fn.resolve()
		free_call = ast.Call(
			func = ast.Attribute( value = ast.Name( id = 'sys', ctx = ast.Load() ), attr = 'free', ctx = ast.Load() ),
			args = [ ast.Name( id = 'self', ctx = ast.Load() ) ], keywords = [],
		)
		free_call.resolved_callee = free_fn
		free_call.end_lineno = None; free_call.end_col_offset = None
		free_param_type = free_fn.parameters[0].type
		cast_type_ref = ast.Name( id = '<sys.free.ptr>', ctx = ast.Load() )
		cast_type_ref.resolved_type = free_param_type
		free_call.args = [ ast.Call(
			func = ast.Attribute( value = ast.Name( id = 'compiler', ctx = ast.Load() ), attr = 'cast', ctx = ast.Load() ),
			args = [ cast_type_ref, ast.Name( id = 'self', ctx = ast.Load() ) ], keywords = [],
		)]
		body.append( ast.Expr( free_call ))

		self_param = Parameter(
			stem = 'self', qualname = f'{qualname}.self',
			file = backing_cls.file, line = backing_cls.line, type = backing_cls,
		)
		node = ast.FunctionDef(
			name = '$$__destructor__',
			args = ast.arguments(
				posonlyargs = [], args = [], vararg = None,
				kwonlyargs = [], kw_defaults = [], kwarg = None, defaults = [],
			),
			body = body, decorator_list = [], returns = None, type_params = [],
			lineno = backing_cls.line or 1, col_offset = 0,
			end_lineno = backing_cls.line or 1, end_col_offset = 0,
		)
		ast.fix_missing_locations( node )

		dtor_fn = Function(
			stem = '$$__destructor__', qualname = qualname,
			file = backing_cls.file, line = backing_cls.line,
			cls = None, node = node,
			parameters = [ self_param ], return_type = none_type,
			is_static = True, is_destructor = True, resolve = None,
		)
		dtor_fn.add_name( 'self', self_param )
		self.schedule( dtor_fn )

	def _rewrite_generator_constructor( self, fn: Function, backing_cls: RCClass, locals_decl: dict[str,Type], defer_sites: list[tuple[str,bool,list[ast.stmt]]], send_type: 'Type|None' = None ) -> None:
		''' replaces the original generator def's own body with a single
		`return <allocate the backing class, state=0, fields=args/zeros>` -
		matches Python's own "calling a generator function doesn't run any
		of the body" semantics for free, since __next__ (built separately
		above) is where the real body now lives. Routed through
		Lowering._lower_allocate_fields (the same field-value-lowering/
		incref logic an ordinary no-__init__ `ClassName(field=value,...)`
		construction call already uses) via the `generator_backing_cls`
		escape-hatch tag - see lowering.py's own
		_try_lower_generator_allocate_call - rather than requiring this
		synthesized class to be resolvable by name through any real scope,
		mirroring the established resolved_callee/resolved_construction
		convention that file already uses for other compiler-synthesized
		call sites.

		Every promoted local (locals_decl - this now includes Phase 1's
		for-loop-desugaring locals like __for_obj_N, no longer a separate
		extra_fields mechanism) gets the SAME zero-placeholder/live-flag-
		false treatment here, unconditionally - none of them are actually
		evaluated at construction time anymore (see _new_for_obj_field's
		own docstring for __for_obj_N specifically: it's re-derived from
		its real expression every time program execution reaches the
		for-loop it belongs to, inside $$__next__/$$__resume__ itself, not
		here). '''
		none_type = self.discovery.get_none_type()
		keywords = [ ast.keyword( arg = '__state', value = ast.Constant( value = 0 ) ) ]
		for p in fn.parameters or []:
			name_node = ast.Name( id = p.stem, ctx = ast.Load() )
			ast.copy_location( name_node, fn.node )
			keywords.append( ast.keyword( arg = p.stem, value = name_node ) )
		for stem, t in locals_decl.items():
			if t.is_rc():
				t_base = t.base if isinstance( t, Specialization ) else t
				if isinstance( t_base, TaggedUnion ) and any( a.type is none_type for a in t_base.attributes ):
					# a T|None promoted local (e.g. A.4a's own __for_next_N,
					# holding a for-loop-desugared iterator's raw .__next__()
					# result across a yield) has an obvious, always-valid
					# "not assigned yet" placeholder already: None itself -
					# a real member of its own declared type, needing no
					# generator_zero_rc_field exemption (that exemption is
					# TaggedUnion-excluded below in lowering.py's
					# _check_assignable - a bare `0` was never a meaningful
					# stand-in for an arbitrary union's tag+data shape the
					# way it is for a plain RCClass pointer)
					zero = ast.Constant( value = None )
				else:
					# PLAN_GENERATORS.md Phase 5 (roadmap Phase 5) - never
					# read before its own first real assignment (gated by
					# the companion live-flag field below, checked by the
					# generator's own state/flag-gated destructor) - see
					# _expr_Constant's own generator_zero_rc_field exemption.
					# NOTE: doesn't cover a promoted local typed as an RC
					# union WITHOUT a None member (e.g. `held: A|B = ...`) -
					# no zero-cost placeholder exists for that shape either,
					# unexercised by anything built so far (every generator-
					# yield_from/for-loop-forwarding site produces T|None by
					# construction)
					zero = ast.Constant( value = 0 )
					zero.generator_zero_rc_field = True
			else:
				zero = ast.Constant( value = False if ( isinstance( t, Scalar ) and t.stem == 'bool' ) else 0 )
			keywords.append( ast.keyword( arg = stem, value = zero ) )
			if t.is_rc():
				keywords.append( ast.keyword( arg = self._live_flag_stem( stem ), value = ast.Constant( value = False ) ) )
		for flag_stem, _is_errdefer, _body in defer_sites:
			keywords.append( ast.keyword( arg = flag_stem, value = ast.Constant( value = False ) ) )
		if send_type is not None:
			# PLAN_GENERATORS.md Phase C - __send_slot starts exactly like
			# an RC-typed promoted local's own zero-placeholder (never read
			# before __send_slot_live gates it True - see _expr_Constant's
			# generator_zero_rc_field exemption), or an ordinary scalar
			# zero/False otherwise; __send_ready always starts False (no
			# pending send at construction time)
			if send_type.is_rc():
				send_zero = ast.Constant( value = 0 )
				send_zero.generator_zero_rc_field = True
			else:
				send_zero = ast.Constant( value = False if ( isinstance( send_type, Scalar ) and send_type.stem == 'bool' ) else 0 )
			keywords.append( ast.keyword( arg = '__send_slot', value = send_zero ) )
			if send_type.is_rc():
				keywords.append( ast.keyword( arg = '__send_slot_live', value = ast.Constant( value = False ) ) )
			keywords.append( ast.keyword( arg = '__send_ready', value = ast.Constant( value = False ) ) )
		call = ast.Call( func = ast.Name( id = backing_cls.stem, ctx = ast.Load() ), args = [], keywords = keywords )
		call.generator_backing_cls = backing_cls
		fn.node.body = [ ast.Return( value = call ) ]
		ast.fix_missing_locations( fn.node )

	def ensure_generator_synthesized( self, fn: Function, origin_type_param_stems: 'list[str]|None' = None ) -> None:
		''' idempotent (id(fn)-memoized) - a no-op unless fn's own body
		actually contains a `yield` (checked first, cheaply). See this
		section's own top docstring for the full design and why this runs
		from ensure_resolved rather than lowering.py.

		origin_type_param_stems: PLAN_GENERATORS.md Phase 3 (roadmap Phase
		3) - non-None only when `fn` is a monomorphized copy of a GENERIC
		generator template (passed by both call sites that build one -
		ensure_resolved's Specialization branch and visit_Call's own
		nested-generic-call resolution - each already has the abstract
		base Function's own .type_params in hand at the point it calls
		this). See the interim-scope rejection below for why this is
		needed at all. '''
		if id( fn ) in self._generators_synthesized:
			return
		if not self._function_contains_yield( fn ):
			return
		self._generators_synthesized.add( id( fn ))

		if origin_type_param_stems:
			# Recommended interim scope (PLAN_GENERATORS.md's own roadmap
			# Phase 3 write-up): a generic generator body that itself
			# calls another generic function referencing the enclosing
			# generator's own type param is rejected for now, sidestepping
			# a real ordering hazard confirmed by a minimal repro, not
			# just a hypothetical one - _build_generator_next_function
			# copies this function's OWN raw body statements into a FRESH
			# `__next__` method/backing-class scope that does NOT inherit
			# the T -> concrete-arg substitution monomorphized_function
			# recorded on `fn.names` (that substitution lives only on
			# THIS Function object, never propagated to the new one built
			# for it) - so a body statement that still needs it (e.g. `y:
			# T = identity(x)`, whether or not identity's own call
			# actually depends on T) fails with "name 'T' is not defined"
			# once __next__'s body is itself resolved later. A bare `x: T`
			# PARAMETER (the v1 baseline case) is unaffected - parameter
			# types flow through fn.parameters, already correctly
			# substituted independent of this - only a body-level
			# reference to the type param's own bare name is at risk,
			# which is exactly what this scans for. Lifting this needs
			# __next__/the backing class to inherit the substitution
			# (thread origin_type_param_stems's underlying (stem,
			# concrete-type) pairs through _build_generator_next_function/
			# _build_generator_backing_class's own names dicts) - not
			# attempted here, see PLAN_GENERATORS.md's own Phase 3 write-up
			for stmt in fn.node.body:
				for n in ast.walk( stmt ):
					if isinstance( n, ast.Name ) and n.id in origin_type_param_stems:
						self.discovery.fail(
							f'{fn.qualname}: a generic generator body that references its own type parameter '
							f'({n.id}) outside a parameter/return annotation is not supported yet - see PLAN_GENERATORS.md',
							fn.node,
						)

		if fn.type_params:
			# PLAN_GENERATORS.md Phase 3 (roadmap Phase 3) - the ABSTRACT,
			# still-generic template (`def gen[T](x: T) -> Iterator[T]:`
			# itself, T unbound) must never get a backing class of its own
			# - same "skip the unbound template, only the concrete
			# instantiation gets synthesized" posture as _schedule_rcclass_
			# destructor_deps's identical cls.type_params guard. Reached
			# harmlessly and often: ensure_resolved's own Specialization
			# branch resolves EACH concrete gen[i32]/gen[str]/... copy
			# separately (each with type_params cleared - see monomorphize.
			# py's _build_monomorphized_function - so THOSE go on to
			# synthesize normally, below), but plenty of other paths
			# (_type_of_expr's Call handling, resolving a narrowed local's
			# type) legitimately still reach the bare abstract Function
			# first, well before any concrete instantiation exists - a
			# hard failure here would reject the very first `gen(...)` or
			# `gen[i32](...)` call site in the program, not just a
			# genuinely unsupported shape
			return
		if fn.cls is not None:
			self.discovery.fail( f'{fn.qualname}: a generator method is not supported yet - only a plain function may contain yield - see PLAN_GENERATORS.md', fn.node )
		if not isinstance( fn.return_type, GeneratorType ):
			self.discovery.fail( f'{fn.qualname} contains yield but is not declared -> Iterator[T]', fn.node )

		# this runs eagerly from a CALL SITE (see this function's own
		# top docstring), which may live in a different module than fn
		# itself - every name lookup below (locals' own type
		# annotations, in particular) must resolve against fn's OWN
		# defining module, not whichever module happens to be active on
		# discovery.module_stack at the call site. Mirrors
		# resolve_function_body's identical push, just triggered earlier.
		module = self._find_module_for( fn )
		with self.discovery.module_context( module ):
			elem_type = fn.return_type.elem_type
			error_type = fn.return_type.error_type
			send_type = fn.return_type.send_type # PLAN_GENERATORS.md Phase C - None unless Generator[T,SendType,E] (3-arg form)
			self.schedule( elem_type )
			if send_type is not None:
				self.schedule( send_type )

			# PLAN_GENERATORS.md A.4a follow-up - `yield from` desugars into
			# an ordinary for-loop first, so the for-loop desugaring pass
			# right after picks up both user-written AND synthesized for-
			# loops uniformly, with zero special-casing. A for-loop-with-
			# yield/yield-from reachable through a while/for that could
			# re-enter it used to be rejected outright here (a nesting
			# validator ran before this point) - lifted once __for_obj_N
			# (_new_for_obj_field, below) stopped being eagerly constructed
			# once, at the generator's own construction, and started being
			# re-derived every time the loop is actually reached instead,
			# same as a real Python generator's own lazy per-entry
			# construction - see that method's own docstring for the real
			# bug this used to paper over.
			yield_from_extra_locals = self._desugar_generator_yield_from( fn, elem_type, error_type )
			extra_locals = self._desugar_generator_for_loops( fn )
			extra_locals.update( yield_from_extra_locals )
			self._validate_generator_defer_sites( fn )
			defer_sites = self._desugar_generator_defer_sites( fn )
			self._reject_generator_value_return( fn )
			pending_bare_return_assigns = self._rewrite_generator_bare_returns( fn, defer_sites )
			locals_decl = self._collect_generator_locals( fn )
			locals_decl.update( extra_locals )

			# PLAN_GENERATORS.md's StopIteration reversal - error_type is
			# never None for a legally-constructed GeneratorType (discovery.py's
			# visit_Subscript requires it to already include StopIteration
			# among its own leaves), so __next__ is unconditionally fallible:
			# it returns Result[elem_type, error_type], never a bare nullable
			# elem_type|None. Reaching the end of the generator produces
			# Err(StopIteration()) (see _wrap_generator_next_returns_in_ok's
			# own exhaustion-tag handling below), not Ok(None) - there is no
			# more "the success channel is itself nullable" case to build.
			# or_return()/checked-arithmetic inside the body engage the
			# existing _require_result_return machinery for free (no special
			# generator-side flag needed - it's purely a consequence of
			# __next__'s own declared return type, exactly like any other
			# fallible function).
			assert error_type is not None, 'GeneratorType.error_type is never None for a legally-constructed generator (see discovery.py)'
			self.schedule( error_type )
			result_cls = self.discovery.find_name_or_none( 'Result' )
			assert isinstance( result_cls, ClassLike ), 'builtins.Result is required for a generator\'s own __next__ but was not found'
			next_return_type = self.discovery._get_or_create_specialization( result_cls, [ elem_type, error_type ] )
			self.schedule( next_return_type )

			backing_cls = self._build_generator_backing_class( fn, locals_decl, defer_sites, send_type )
			resume_fn = self._build_generator_next_function( fn, backing_cls, locals_decl, next_return_type, pending_bare_return_assigns, defer_sites, send_type )
			# PLAN_GENERATORS.md Phase C - __next__()/send(v) thin wrappers
			# over $$__resume__ (built just above) - only when SendType is
			# declared; Iterator[T]/Generator[T,E] already got their own
			# real __next__ directly from _build_generator_next_function
			if send_type is not None:
				self._build_generator_send_wrappers( fn, backing_cls, resume_fn, send_type, next_return_type )
			# PLAN_GENERATORS.md Phase 5 (roadmap Phase 5) - built BEFORE
			# backing_cls is ever scheduled below, so its own pre-mark of
			# id(backing_cls) in self._destructors_synthesized (see its own
			# docstring) beats compiler.py's ordinary, unconditional RCClass
			# handling to the punch - that path checks the SAME memo set
			# before ever building its own (wrong, unconditional-decref)
			# destructor for this class
			self._build_generator_destructor( fn, backing_cls, locals_decl, defer_sites, send_type )

			self.schedule( backing_cls )
			self.schedule( backing_cls.get_local_or_raise( '__next__' ))
			if send_type is not None:
				self.schedule( backing_cls.get_local_or_raise( 'send' ))

			self._rewrite_generator_constructor( fn, backing_cls, locals_decl, defer_sites, send_type )
			fn.return_type = backing_cls

	def _schedule_rcclass_destructor_deps( self, cls: RCClass ) -> None:
		''' the emitter always synthesizes a destructor for every
		non-generic RCClass — ensure its transitive dependencies
		(sys.free and __del__ if declared) are scheduled.

		schedule()'s own Specialization branch calls this with the BARE
		base class even when that base is generic (e.g. list[RawEntry]'s
		own schedule() call passes bare `list`, not the specialization) -
		cls.type_params is the guard for that case: a generic class's own
		__del__ has an unbound T, never directly compilable (confirmed:
		scheduling it bare crashes downstream with "compiler.is_rc(T)
		requires a concrete type"). monomorphize_class's own per-
		Specialization method-substitution loop already builds and
		schedules the correctly-substituted __del__[ConcreteArgs] for
		whichever specialization is actually in use - only sys.free needs
		ensuring here for that case, not a second, wrong scheduling of the
		abstract __del__ template. '''
		self._ensure_sys_free_scheduled()
		if cls.type_params:
			return
		del_fn = cls.get_local( '__del__' )
		if isinstance( del_fn, Function ):
			self.schedule( del_fn )

	def _schedule_uniontype_storage( self, union: TaggedUnion ) -> None:
		''' every TaggedUnion that gets scheduled for real emission needs its
		runtime tag/data storage shape built (union_storage.get(union) —
		see UnionStorage.get's own docstring for what that computes).
		Normally that happens lazily, the first time some reachable code
		path actually constructs a variant (ClassName.Variant(value)) or
		matches on one. A union referenced ONLY as a Result[T,E]/other
		generic's type argument, with no reachable code anywhere
		constructing one of its own variants, never hits that lazy path -
		it still gets scheduled here (as a Specialization's own .base, or
		bare, both below), but its storage is never built, and emit_c()
		crashes outright trying to emit a union whose storage doesn't
		exist yet (confirmed directly: AssertionError, "_tagged_union_
		storage has not run yet - no real storage shape to emit", from a
		minimal repro with a @union used purely as a Result error type and
		never constructed). Calling union_storage.get() unconditionally
		here, mirroring _schedule_rcclass_destructor_deps's identical role
		for RCClass's own synthesized destructor just above, closes that
		gap - same type_params guard for the same reason (an abstract
		generic union's own attributes still carry unbound TypeVars, not
		yet substituted into anything union_storage.get() could build real
		C storage from; monomorphize_class handles that substitution
		separately for whichever concrete specialization is actually in
		use, the same way it does for RCClass methods - see
		_schedule_rcclass_destructor_deps's own comment on that split).

		Re-entrancy guard (self._union_storage_scheduling) is required, not
		optional: union_storage.get(union) itself starts with _ensure_
		resolved(union), which calls schedule(union) right back - routing
		straight into THIS method again for the same union, before get()'s
		own memoization cache has anything in it yet to short-circuit on
		(the cache is only populated once get() finishes). Confirmed
		directly: without this guard, scheduling any union at all (even
		plain Result[i32,OverflowError]) blew the stack immediately. '''
		if union.type_params:
			return
		key = id( union )
		if key in self._union_storage_scheduling:
			return
		self._union_storage_scheduling.add( key )
		try:
			self.union_storage.get( union )
		finally:
			self._union_storage_scheduling.discard( key )

	def _synthesize_rcclass_destructor( self, cls: RCClass ) -> None:
		''' build an AST Function for $$__destructor__ that the emitter
		can lower like any other function. Called from compiler._lower
		after the class body is resolved, never from within schedule(). '''
		if cls.type_params:
			return  # only concrete RCClasses get a destructor
		# an RCClass with any unfulfilled @abstractmethod slot anywhere in
		# its own chain can never be directly constructed (same is_abstract
		# walk lowering.py's _check_rcclass_fully_implemented/emitter_c.py's
		# _rcclass_fulfilled_slot_impls already use - all three have to
		# agree on exactly which classes are "complete"). Its OWN
		# destructor is then unreachable dead code, unconditionally, in
		# EVERY program that ever uses it purely as a base: nothing ever
		# calls it directly, and emit_rcclass_vtable_instance already skips
		# building a vtable instance for an abstract class (the only thing
		# that would ever reference $$__destructor__'s own address) - a
		# real, confirmed -Wunused-function suite-wide (codecs.Codec,
		# logging.Handler, ...), not a hypothetical. A concrete subclass's
		# own destructor tears down the FULL inherited field set directly
		# (base-first, via _build_field_teardown_ast below) - it never
		# calls into an ancestor's own separately-synthesized destructor -
		# so skipping synthesis entirely here removes nothing anything else
		# depends on.
		for slot in cls.virtual_slots():
			impl = cls.chain_lookup( slot.stem )
			if not isinstance( impl, Function ):
				return
			if impl.resolve is not None:
				impl.resolve()
			if impl.is_abstract:
				return
		sys_module = self.discovery.modules.get( 'sys' )
		if sys_module is None:
			return  # sys.free must be available
		# a bare RCClass may reach _lower twice (once as Specialization,
		# once directly via param type scheduling) — synthesize only once
		if id( cls ) in self._destructors_synthesized:
			return
		self._destructors_synthesized.add( id( cls ))

		none_type = self.discovery.get_none_type()
		qualname = f'{cls.qualname}$$__destructor__'
		del_fn = cls.get_local( '__del__' )
		if not isinstance( del_fn, Function ):
			del_fn = None

		body: list[ast.stmt] = []

		# 1. self.__del__() if declared (fields still intact)
		if del_fn is not None:
			body.append( ast.Expr( ast.Call(
				func = ast.Attribute(
					value = ast.Name( id = 'self', ctx = ast.Load() ),
					attr = '__del__', ctx = ast.Load(),
				),
				args = [], keywords = [],
			)))

		# 2. field cascade: decref every RC leaf, base-first
		if cls.resolve is not None:
			cls.resolve()
		chain: list[RCClass] = []
		node_ref: RCClass|None = cls
		while node_ref is not None:
			chain.append( node_ref )
			node_ref = node_ref.base
		# _build_field_teardown_ast may need to synthesize a union member
		# constructor for the FIRST time (a field typed as an anonymous
		# X|Y never otherwise touched, e.g. a tuple[X|None,...] element
		# reached here as a queued RCClass before anything else ever
		# constructs a real value of that tuple type) - UnionStorage.get() stamps
		# that constructor's own file from "whichever module is currently
		# active" (module_stack[-1]), which is otherwise NOT the case here:
		# compiler._lower's RCClass branch calls this method directly,
		# with no module_context of its own (unlike resolve_function_body,
		# which always pushes one first - see this class's own identical
		# _find_module_for pattern there). Without this, the synthesized
		# constructor's file stays None and a later _find_module_for on IT
		# fails outright ("no module found owning ...") the first time
		# anything actually needs to lower/call it.
		with self.discovery.module_context( self._find_module_for( cls )):
			for base_cls in reversed( chain ):
				for attr in base_cls.attributes:
					if attr.resolve is not None:
						attr.resolve()
					body.extend( self._build_field_teardown_ast(
						ast.Attribute(
							value = ast.Name( id = 'self', ctx = ast.Load() ),
							attr = attr.stem, ctx = ast.Load(),
						),
						attr.type,
					))

		# 3. sys.free(self) — resolve the callee and tag it so lowering
		# skips overload resolution (sys.free is an Overload group,
		# self: RCClass doesn't match any overload's declared type)
		free_overload = sys_module.get_local( 'free' )
		from mpy_types import Overload
		if isinstance( free_overload, Overload ):
			free_fn = free_overload.implementations[0]
		else:
			free_fn = free_overload
		if free_fn.resolve is not None:
			free_fn.resolve()
		free_call = ast.Call(
			func = ast.Attribute(
				value = ast.Name( id = 'sys', ctx = ast.Load() ),
				attr = 'free', ctx = ast.Load(),
			),
			args = [ ast.Name( id = 'self', ctx = ast.Load() ) ], keywords = [],
		)
		free_call.resolved_callee = free_fn
		free_call.end_lineno = None; free_call.end_col_offset = None
		# self (a bare RCClass) and sys.free's own declared parameter
		# (Ptr[u8]/Ptr[None]) are both pointer-representable (same bit
		# pattern - see TypeResolver._is_pointer_representable) but NOT the
		# same TYPE, which lowering.py's own general assignability check
		# (_lower_expr's _check_assignable) now correctly rejects for
		# ordinary code - this synthesized call needs the identical explicit
		# reinterpret a real user would have to write, via the same
		# resolved_type escape hatch _get_or_create_closure_trampoline
		# already uses (lowering.py's _lower_compiler_cast), rather than
		# relying on an implicit conversion nothing used to check
		free_param_type = free_fn.parameters[0].type
		cast_type_ref = ast.Name( id = '<sys.free.ptr>', ctx = ast.Load() )
		cast_type_ref.resolved_type = free_param_type
		free_call.args = [ ast.Call(
			func = ast.Attribute(
				value = ast.Name( id = 'compiler', ctx = ast.Load() ),
				attr = 'cast', ctx = ast.Load(),
			),
			args = [ cast_type_ref, ast.Name( id = 'self', ctx = ast.Load() ) ], keywords = [],
		)]
		body.append( ast.Expr( free_call ))

		self_param = Parameter(
			stem = 'self', qualname = f'{qualname}.self',
			file = cls.file, line = cls.line, type = cls,
		)
		node = ast.FunctionDef(
			name = '$$__destructor__',
			args = ast.arguments(
				posonlyargs = [], args = [], vararg = None,
				kwonlyargs = [], kw_defaults = [], kwarg = None, defaults = [],
			),
			body = body, decorator_list = [], returns = None, type_params = [],
			lineno = cls.line or 1, col_offset = 0,
			end_lineno = cls.line or 1, end_col_offset = 0,
		)
		ast.fix_missing_locations( node )

		fn = Function(
			stem = '$$__destructor__', qualname = qualname,
			file = cls.file, line = cls.line,
			cls = None, node = node,
			parameters = [ self_param ], return_type = none_type,
			is_static = True, is_destructor = True, resolve = None,
		)
		fn.add_name( 'self', self_param )
		self.schedule( fn )

	def _synthesize_rcclass_constructor( self, cls: RCClass, init: Function ) -> None:
		''' build an AST Function for $$__new__ - a per-class constructor
		mirroring _synthesize_rcclass_destructor: allocate a raw,
		uninitialized instance (compiler.__raw_alloc__) and call the
		class's own __init__ on it, so every Foo(...) call site
		(lowering.py's _try_lower_construct_call) can call this ONE
		function instead of inlining alloc+header-init+__init__-call
		machinery at every construction site. Unlike the destructor, this
		is called DIRECTLY BY NAME - construction always knows its concrete
		class statically, never dispatched through a vtable - so it needs
		no emitter special-casing at all, ordinary Function emission
		handles it.

		Called ONLY eagerly from _try_lower_construct_call itself, never
		from compiler._lower's own class-registration trigger the way the
		destructor is: unlike the destructor (needed for every RCClass,
		since any instance, however constructed, might need releasing),
		$$__new__ is only ever needed by an actual `Foo(...)` construction
		call site, which already synthesizes it eagerly itself, at the
		exact moment it needs the live Function object (schedule() is a
		deferred queue that can't guarantee that timing). Synthesizing it
		unconditionally for every REGISTERED class too, regardless of
		whether anything ever actually constructs it, was tried and
		reverted: it forced a vtable reference (the header.vtable
		assignment inside $$__new__'s own body) for classes never meant to
		be constructed at all - a real regression, confirmed by a test
		asserting an abstract base class, only ever used polymorphically
		through a subclass, never gets its own vtable INSTANCE emitted.

		`init` is used AS-IS instead of being re-derived via
		cls.get_local('__init__') - required for a monomorphized generic
		class: _try_lower_construct_call's own eager call already holds
		the correctly-monomorphized __init__ (T substituted to the real
		concrete type) as a local (_lower_generic_construction_args' own
		monomorphized_init) - re-deriving it here via a fresh
		cls.get_local('__init__') lookup instead is NOT reliably the same
		object (confirmed by a real repro: Box(7) with T inferred purely
		from the argument, no surrounding annotation - the fresh lookup
		here produced an __init__ whose own parameter type was still the
		bare, unsubstituted TypeVar T, crashing the emitter outright once
		it tried to mangle a TypeVar into a C type). '''
		if cls.type_params:
			return  # only concrete RCClasses get a constructor
		if id( cls ) in self._constructors_synthesized:
			return
		self._constructors_synthesized.add( id( cls ))

		if cls.resolve is not None:
			cls.resolve()
		if not isinstance( init, Function ):
			return  # an Overload - lowering.py's _try_lower_construct_call already rejects this case with its own error message
		if init.resolve is not None:
			init.resolve()
		if any( p.is_vararg or p.is_kwarg or p.is_move or p.is_copy for p in ( init.parameters or [] )):
			# no real __init__ in this codebase declares any of these -
			# forwarding them correctly (re-spelling *args/**kwargs
			# unpacking, or the explicit move(x)/copy(x) call-site marker
			# move[T]/copy[T] params require) through a synthesized AST
			# body is unsupported for now rather than silently miscompiled
			self.discovery.fail(
				f'{init.qualname}: *args/**kwargs/move[T]/copy[T] parameters are not supported yet for construction',
				init.node,
			)

		none_type = self.discovery.get_none_type()
		# fallibility check mirrors lowering.py's own Lowering._init_
		# fallibility exactly (duplicated, not shared - that one lives on
		# Lowering, not TypeResolver). _result_shape/find_name_or_none
		# resolve 'Result' relative to discovery.module_stack[-1] - safe
		# when this method runs eagerly (mid-lowering of some real
		# function, module_stack already correctly populated), but
		# module_stack can be genuinely EMPTY when reached from compiler.
		# py's own class-registration trigger instead (confirmed by a real
		# crash: IndexError in find_name_or_none, from a merged-executable
		# test where no eager construction call site ever ran first) -
		# push cls's own declaring module explicitly, same as
		# resolve_function_body's own module_context push, so this is
		# correct regardless of which trigger reached it first
		with self.discovery.module_context( self._find_module_for( cls )):
			if init.return_type is none_type:
				fallible = False
				error_cls = None
				result_cls = None
			else:
				shape = self._result_shape( init.return_type )
				if shape is None or shape[0] is not none_type:
					self.discovery.fail(
						f'{init.qualname} must return None or Result[None,_], got '
						f'{init.return_type.qualname if init.return_type else None}',
						init.node,
					)
				fallible = True
				error_cls = shape[1]
				result_cls = self.discovery.find_name_or_none( 'Result' )

		qualname = f'{cls.qualname}$$__new__'
		new_params: list[Parameter] = []
		for p in ( init.parameters or [] ):
			new_params.append( Parameter(
				stem = p.stem, qualname = f'{qualname}.{p.stem}',
				file = cls.file, line = cls.line, type = p.type,
				is_posonly = p.is_posonly, is_kwonly = p.is_kwonly,
			))

		# self = compiler.__raw_alloc__(<cls>) - <cls> handed over directly
		# via the resolved_type escape hatch (no natural source-level
		# spelling for a monomorphized generic class - same technique the
		# destructor's own <sys.free.ptr> node above uses)
		class_ref = ast.Name( id = '<$$__new__.cls>', ctx = ast.Load() )
		class_ref.resolved_type = cls
		self_assign = ast.Assign(
			targets = [ ast.Name( id = 'self', ctx = ast.Store() ) ],
			value = ast.Call(
				func = ast.Attribute( value = ast.Name( id = 'compiler', ctx = ast.Load() ), attr = '__raw_alloc__', ctx = ast.Load() ),
				args = [ class_ref ], keywords = [],
			),
		)

		# self.__init__(<forward every param>) - kwonly params must be
		# forwarded as keywords (Python calling convention), everything
		# else positionally; new_params' own stems/order are a direct 1:1
		# copy of init.parameters, so this is always a valid, complete call
		init_call = ast.Call(
			func = ast.Attribute( value = ast.Name( id = 'self', ctx = ast.Load() ), attr = '__init__', ctx = ast.Load() ),
			args = [ ast.Name( id = p.stem, ctx = ast.Load() ) for p in new_params if not p.is_kwonly ],
			keywords = [ ast.keyword( arg = p.stem, value = ast.Name( id = p.stem, ctx = ast.Load() ) ) for p in new_params if p.is_kwonly ],
		)

		body: list[ast.stmt] = [ self_assign ]
		if not fallible:
			body.append( ast.Expr( init_call ))
			body.append( ast.Return( value = ast.Name( id = 'self', ctx = ast.Load() )))
			return_type: Type = cls
		else:
			body.append( ast.Assign( targets = [ ast.Name( id = 'result', ctx = ast.Store() ) ], value = init_call ))
			is_err_call = ast.Call(
				func = ast.Attribute( value = ast.Name( id = 'result', ctx = ast.Load() ), attr = 'is_err', ctx = ast.Load() ),
				args = [], keywords = [],
			)
			# Result.Err(result.data.v_Err) - same union-payload shape
			# _emit_fallible_construction's own former Err branch used
			err_expr = ast.Call(
				func = ast.Attribute( value = ast.Name( id = 'Result', ctx = ast.Load() ), attr = 'Err', ctx = ast.Load() ),
				args = [ ast.Attribute(
					value = ast.Attribute( value = ast.Name( id = 'result', ctx = ast.Load() ), attr = 'data', ctx = ast.Load() ),
					attr = 'v_Err', ctx = ast.Load(),
				) ], keywords = [],
			)
			raw_free_stmt = ast.Expr( ast.Call(
				func = ast.Attribute( value = ast.Name( id = 'compiler', ctx = ast.Load() ), attr = '__raw_free__', ctx = ast.Load() ),
				args = [ ast.Name( id = 'self', ctx = ast.Load() ) ], keywords = [],
			))
			body.append( ast.If(
				test = is_err_call,
				body = [ raw_free_stmt, ast.Return( value = err_expr ) ],
				orelse = [],
			))
			# self is fully constructed here. Result.Ok(self)'s own
			# construction takes an INDEPENDENT incref'd copy of self for
			# the payload it builds (confirmed empirically: returning
			# Result.Ok(self) directly, relying on self's own scope-exit
			# epilogue to release its original reference, leaked one ref
			# per successful construction - the returned expression is
			# Result.Ok(self)'s OWN result, not self itself, so the "return
			# your own local directly, skip its release" fast path the
			# plain non-fallible branch above relies on never applies here)
			# - self's own original reference is a SEPARATE unit that still
			# needs its own explicit release, same as the old raw-IR
			# _emit_fallible_construction's own Ok branch had to do by hand.
			#
			# The intermediate `ok` local needs an explicit Result[cls,
			# error_cls] annotation - Result.Ok(value)'s own E type param
			# can never be inferred from `value: T` alone (SYNTAX.md/
			# _lower_generic_construction_args's own comment on this), so a
			# bare, un-annotated `ok = Result.Ok(self)` fails to infer E.
			# For a MONOMORPHIZED GENERIC cls specifically, a by-name
			# annotation (Result[Box, error_cls], built from cls.stem)
			# would be actively WRONG, not just unspellable: cls.stem is
			# still the ABSTRACT template's own bare name ('Box'), so
			# ordinary scope lookup resolves the annotation's own T slot to
			# the wrong (abstract) class - which then conflicts with T
			# ALSO being inferred, correctly, as the concrete Box[i32] from
			# self's own argument type, a genuine "T inferred as both X and
			# Y" compile error (confirmed by a real repro: RCClassConstruct
			# Tests' own generic-init-construction fallible-wrapping tests,
			# which construct exactly this shape). Uses discovery.py's own
			# node.resolved_type escape hatch instead (this session's own
			# addition to visit_Name, mirroring the identical, already-
			# established lowering.py-side convention _lower_compiler_cast/
			# _lower_compiler_raw_alloc's own arguments already use) -
			# tags a single Name node with the already-built, concrete
			# Result[cls,error_cls] Specialization object directly
			return_type = self.discovery._get_or_create_specialization( result_cls, [ cls, error_cls ])
			ok_expr = ast.Call(
				func = ast.Attribute( value = ast.Name( id = 'Result', ctx = ast.Load() ), attr = 'Ok', ctx = ast.Load() ),
				args = [ ast.Name( id = 'self', ctx = ast.Load() ) ], keywords = [],
			)
			ok_annotation = ast.Name( id = '<$$__new__.result_type>', ctx = ast.Load() )
			ok_annotation.resolved_type = return_type
			decref_self_stmt = ast.Expr( ast.Call(
				func = ast.Attribute( value = ast.Name( id = 'compiler', ctx = ast.Load() ), attr = 'decref', ctx = ast.Load() ),
				args = [ ast.Name( id = 'self', ctx = ast.Load() ) ], keywords = [],
			))
			body.append( ast.AnnAssign( target = ast.Name( id = 'ok', ctx = ast.Store() ), annotation = ok_annotation, value = ok_expr, simple = 1 ))
			body.append( decref_self_stmt )
			body.append( ast.Return( value = ast.Name( id = 'ok', ctx = ast.Load() )))

		node = ast.FunctionDef(
			name = '$$__new__',
			args = ast.arguments(
				posonlyargs = [], args = [], vararg = None,
				kwonlyargs = [], kw_defaults = [], kwarg = None, defaults = [],
			),
			body = body, decorator_list = [], returns = None, type_params = [],
			lineno = cls.line or 1, col_offset = 0,
			end_lineno = cls.line or 1, end_col_offset = 0,
		)
		ast.fix_missing_locations( node )

		fn = Function(
			stem = '$$__new__', qualname = qualname,
			file = cls.file, line = cls.line,
			cls = cls, node = node,
			parameters = new_params, return_type = return_type,
			is_static = True, resolve = None,
		)
		for p in new_params:
			fn.add_name( p.stem, p )
		self.schedule( fn )
		cls.add_name( '$$__new__', fn )

	def _build_field_teardown_ast( self, field_expr: ast.Attribute, field_type: Type ) -> list[ast.stmt]:
		''' recursively build AST statements to decref every RC leaf
		reachable from field_expr, given its declared type. '''
		base = field_type.base if isinstance( field_type, Specialization ) else field_type
		line = field_expr.lineno if hasattr( field_expr, 'lineno' ) and field_expr.lineno else 1

		# any bare RC pointer — compiler.decref(expr).
		#
		# is_rc_pointer(), not isinstance( base, RCClass ): a tuple-typed field
		# is just as much a single RC pointer, but used to match NOTHING in
		# this ladder and fell all the way through to the `return []` at the
		# bottom, so an RCClass holding a tuple field never released it. The
		# synthesized backing class is a real RCClass with a real destructor -
		# it simply never got decref'd from the owner, because this walk was
		# the one place that had to say so.
		if field_type.is_rc_pointer():
			# getattr: a TupleType has no .resolve at all (nothing to resolve -
			# its backing class is synthesized on demand by tuple_storage),
			# unlike an RCClass whose body may still be unparsed
			resolve = getattr( base, 'resolve', None )
			if resolve is not None:
				resolve()
			return [ ast.Expr( ast.Call(
				func = ast.Attribute(
					value = ast.Name( id = 'compiler', ctx = ast.Load(), lineno = line, col_offset = 0 ),
					attr = 'decref', ctx = ast.Load(), lineno = line, col_offset = 0,
				),
				args = [ field_expr ], keywords = [],
			), lineno = line, col_offset = 0 ) ]

		# CStruct — recurse into each field (all are always live)
		if isinstance( base, CStruct ):
			if base.resolve is not None:
				base.resolve()
			stmts: list[ast.stmt] = []
			for attr in base.attributes:
				if attr.resolve is not None:
					attr.resolve()
				sub_expr = ast.Attribute(
					value = field_expr, attr = attr.stem, ctx = ast.Load(),
					lineno = line, col_offset = 0,
				)
				stmts.extend( self._build_field_teardown_ast( sub_expr, attr.type ))
			return stmts

		# TaggedUnion — if-chain: read tag, decref active RC member
		if isinstance( base, TaggedUnion ):
			if base.resolve is not None:
				base.resolve()
			for attr in base.attributes:
				if attr.resolve is not None:
					attr.resolve()
			# a GENERIC union's own base.attributes are its bare, unsubstituted
			# declared field types (Result's own Ok: T / Err: E) - a bare TypeVar
			# is never RC (same "substitution has to happen BEFORE the is_rc()
			# filter" trap Specialization.rc_leaves's own docstring documents),
			# so checking member.type.is_rc_pointer() directly here, unsubstituted,
			# silently skipped every member of EVERY generic-union instantiation
			# regardless of what its own args actually were - confirmed via a
			# real reference leak: a generator's own promoted Result[Box,
			# StopIteration]-typed field read its own tag then released nothing,
			# for either member. Substitute field_type's own concrete args in
			# first, same identity-keyed substitution rc_leaves() already uses.
			substitution: dict[int,Type] = {}
			if isinstance( field_type, Specialization ) and base.type_params:
				substitution = { id( param ): arg for param, arg in zip( base.type_params, field_type.args ) }

			# no RC-pointer member at all (every leaf is_rc_pointer() below
			# is False, post-substitution) means the loop below would never
			# build a single If reading the tag - nothing here actually needs
			# tearing down, so skip the tag read entirely rather than emitting
			# `__dtor_tag_N = expr.tag;` with nothing left to compare it
			# against (a real, confirmed -Wunused-variable/C4189: the tag was
			# always computed unconditionally, up front, regardless of
			# whether the loop below ever turned out to need it)
			if not any( substitution.get( id( attr.type ), attr.type ).is_rc_pointer() for attr in base.attributes ):
				return []
			tag_attr, data_attr, _payload_cls, tags = self.union_storage.get( base )

			# __tag = expr.tag
			tag_name = f'__dtor_tag_{self._dtor_label_id}'
			self._dtor_label_id += 1
			stmts: list[ast.stmt] = [ ast.Assign(
				targets = [ ast.Name( id = tag_name, ctx = ast.Store(), lineno = line, col_offset = 0 ) ],
				value = ast.Attribute( value = field_expr, attr = tag_attr.stem, ctx = ast.Load(), lineno = line, col_offset = 0 ),
				lineno = line, col_offset = 0,
			) ]

			for i, member in enumerate( base.attributes ):
				member_type = substitution.get( id( member.type ), member.type )
				# same is_rc_pointer() widening as the top of this method - a
				# tuple-typed union MEMBER was skipped here for the same reason
				# a tuple-typed field was skipped there. Still pointer-only, not
				# is_rc(): the decref below reads the member's payload accessor
				# as a bare RC pointer, which a NESTED union member is not (that
				# case needs its own tag ladder and remains unhandled here -
				# cfg.py's _refcount_instructions is what covers it for values).
				if not member_type.is_rc_pointer():
					continue
				member_expr = ast.Attribute(
					value = ast.Attribute(
						value = field_expr,
						attr = data_attr.stem, ctx = ast.Load(),
						lineno = line, col_offset = 0,
					),
					attr = f'v_{member.stem}', ctx = ast.Load(),
					lineno = line, col_offset = 0,
				)
				stmts.append( ast.If(
					test = ast.Compare(
						left = ast.Name( id = tag_name, ctx = ast.Load(), lineno = line, col_offset = 0 ),
						ops = [ ast.Eq() ],
						comparators = [ ast.Constant( value = tags[member.stem], lineno = line, col_offset = 0 ) ],
						lineno = line, col_offset = 0,
					),
					body = [ ast.Expr( ast.Call(
						func = ast.Attribute(
							value = ast.Name( id = 'compiler', ctx = ast.Load(), lineno = line, col_offset = 0 ),
							attr = 'decref', ctx = ast.Load(), lineno = line, col_offset = 0,
						),
						args = [ member_expr ], keywords = [],
					), lineno = line, col_offset = 0 ) ],
					orelse = [],
					lineno = line, col_offset = 0,
				))
			return stmts

		# CUnion / CEnum / Scalar / Ptr — never RC, nothing to tear down.
		#
		# Asserted rather than assumed: this fallthrough is exactly how the
		# tuple gap above went unnoticed - a field kind that IS RC but matched
		# no branch silently produced "nothing to release" instead of an error.
		# A future RC-bearing Type kind now trips here instead of leaking.
		assert not field_type.is_rc(), (
			f'{field_type.qualname} is RC but reached the teardown fallthrough - '
			f'it needs its own branch in _build_field_teardown_ast'
		)
		return []
	

	# --- pure type queries (moved from lowering.py) ---------------------------

	def _is_ptr_specialization( self, t: Type|None ) -> bool:
		return ( isinstance( t, Specialization )
			and isinstance( t.base, Scalar )
			and t.base.stem in ( 'Ptr', 'ConstPtr' ))

	def _callable_type_of( self, t: Type|None ) -> CallableType|None:
		''' t's own CallableType if t is Ptr[Callable[...]] (a function-
		pointer value - see PLAN_CALLABLE.md), else None. A bare
		CallableType (unwrapped by Ptr) never appears as a real value's
		type - same "always a pointer" rule as every other "address of"
		result in this language (see ir.FunctionRef's own docstring). '''
		if not self._is_ptr_specialization( t ):
			return None
		inner = t.args[0]
		return inner if isinstance( inner, CallableType ) else None

	def _is_RC( self, t: Type|None ) -> bool:
		''' true if `t`'s own runtime representation IS a single bare RC
		pointer - a None-tolerant shim for mpy_types.Type.is_rc_pointer
		(call sites here and in lowering.py hold Type|None).

		Deliberately is_rc_POINTER, not the deeper is_rc(): every caller
		(compiler.incref/decref/addrof/cast, and the compiler.is_rc(T)
		intrinsic's own compile-time fold) goes on to emit a DIRECT pointer
		operation, which is only valid for a genuine pointer. A TaggedUnion
		carrying RC members answers False here and must keep doing so - its
		runtime shape is a tag+data value struct, and it needs cfg.py's
		tag-gated ladder instead.

		This used to be an RCClass-only isinstance check, which left tuples
		out: compiler.is_rc(tuple[str,str]) answered False even though a
		tuple is every bit as much a bare RC pointer as an RCClass, so
		lib/builtins' RawDict/list skipped their key/value increfs entirely
		for tuple element types. '''
		return t is not None and t.is_rc_pointer()

	def _is_pointer_representable( self, t: Type|None ) -> bool:
		''' true if `t`'s own runtime representation IS a single machine
		pointer - a real Ptr[T]/ConstPtr[T], OR an RC pointer (always a
		pointer to its heap object everywhere in this compiler - see
		_is_RC). Used by compiler.cast(...) to allow a plain reinterpret
		cast between ANY two of these (Ptr[None] <-> Ptr[T], Ptr[None] <->
		a bare RCClass, ...) - they're all the same bit pattern, just
		typed differently at the metalpy level '''
		return self._is_ptr_specialization( t ) or self._is_RC( t )

	def _as_specialization( self, t: Type|None ) -> Specialization|None:
		''' `t` itself if it's already a Specialization, else the
		Specialization it was eagerly monomorphized FROM (see
		Monomorphizer.origin_of's own docstring, and PLAN_RESOLVE_CLASS_
		SPECIALIZATIONS.md) - None if `t` is neither (a genuinely plain,
		non-generic type, or a Specialization whose base isn't a ClassLike
		at all - Ptr[T]/ConstPtr[T] never get monomorphized, so they never
		show up in Monomorphizer's own origin table, but they're already
		handled by the isinstance(t, Specialization) branch directly).
		Needed anywhere a caller wants a Specialization's own .base/.args
		from a Type that MIGHT have already been substitute_type_params's
		eager-monomorphize target instead (a rebuilt Specialization whose
		args are now all concrete gets resolved to the real object
		immediately, rather than staying a Specialization wrapper -
		correct and desired everywhere EXCEPT here, where the wrapper
		itself was the only thing carrying "which instantiation is this" '''
		if isinstance( t, Specialization ):
			return t
		return self.monomorphizer.origin_of( t )

	def _same_type( self, a: Type|None, b: Type|None ) -> bool:
		''' true if `a`/`b` are the same type, even when one is a bare
		Specialization and the other is ITS OWN monomorphized form (or
		vice versa) - two representations of the identical instantiation,
		not a genuine conflict. Needed specifically for _unify_type_param's
		own "already bound to something else" check: two different
		argument positions can easily reveal the SAME class specialization
		through different representations (e.g. one from an already-
		monomorphized receiver, another built fresh via _get_or_create_
		specialization from an annotation) - see Monomorphizer.origin_of's
		own docstring for why a Specialization and its monomorphized form
		aren't always identity-equal even though they mean the same thing.

		PLAN_TUPLE.md: the exact same duality exists for a bare TupleType
		vs its own resolved backing RCClass (TupleType.backing) - confirmed
		by a real failure, not anticipated up front: `Result.Ok((q, r))`
		unified against a declared `Result[tuple[int,int],E]` return-type
		annotation saw the ANNOTATION's own bare TupleType (never resolved,
		since an annotation's own type is only resolved on demand) as
		`existing`, and the tuple LITERAL's own already-resolved backing
		RCClass (ensure_resolved runs during _expr_Tuple itself) as
		`actual` - same qualname text, genuinely different objects, wrongly
		reported as "inferred as both X and X" without this. '''
		if a is b:
			return True
		a_spec = self._as_specialization( a )
		b_spec = self._as_specialization( b )
		if a_spec is not None and a_spec is b_spec:
			return True
		a_backing = a.backing if isinstance( a, TupleType ) else a
		b_backing = b.backing if isinstance( b, TupleType ) else b
		return a_backing is not None and a_backing is b_backing

	def _atomic_leaves( self, t: Type ) -> list[Type]:
		''' like t.leaves(), but treats a NOMINAL @union class (t.file is not
		None) as a single opaque leaf - itself - rather than decomposing into
		its own variants' payload types. Mirrors the exact anonymous-vs-
		nominal distinction discovery.py's _get_or_create_union already uses
		when flattening a wider union's own operands (only a synthesized
		anonymous union, t.file is None, is fair game to flatten there too).
		t.leaves() itself stays general-purpose - RC-leaf decomposition
		genuinely wants a union's real payload types even when it's nominal
		(see TaggedUnion.is_rc()) - this is the separate "is t covered by /
		a member of some other union" notion _require_result_return and
		_maybe_widen_return_result need instead. Without this, a nominal
		@union (e.g. HTTPError, all-None-payload variants) widening into a
		bigger union (OSError|HTTPError) decomposed into its own variants'
		payload types (five NoneTypes) instead of being compared as the one
		opaque HTTPError member it actually is. '''
		if isinstance( t, TaggedUnion ) and t.file is None:
			return t.leaves()
		return [ t ]

	def _result_shape( self, t: Type|None ) -> tuple[Type,Type]|None:
		''' (T, E) if `t` is Result[T,E], else None. '''
		result_cls = self.discovery.find_name_or_none( 'Result' )
		spec = self._as_specialization( t )
		if result_cls is None or not ( spec is not None and spec.base is result_cls and len( spec.args ) == 2 ):
			return None
		return spec.args[0], spec.args[1]

	def _tagged_union_shape( self, t: Type|None ) -> tuple[TaggedUnion,list[Variable]]|None:
		''' (abstract base, substituted member list) or None if t isn't
		a TaggedUnion (possibly Specialization-wrapped). '''
		base = t.base if isinstance( t, Specialization ) else t
		if not isinstance( base, TaggedUnion ):
			return None
		members = self.monomorphizer.monomorphize_class( t ).attributes if isinstance( t, Specialization ) else base.attributes
		return base, members

	def _require_result_return( self, node: ast.AST, result_cls: ClassLike, error_cls: ClassLike, alternatives: str, fn: Function|None = None ) -> None:
		# the enclosing function must return Result[_, E_fn] where E_fn COVERS
		# the op/receiver's error type E_op (error_cls) - every leaf of E_op is
		# also a leaf of E_fn. This admits both an exact match (E_fn is E_op,
		# the common case) AND WIDENING: a function may declare one wider error
		# union (e.g. Result[_, A | B | C]) that covers each fallible op's
		# narrower error (A, or B|C, ...), and OrReturn/OrJump remap the narrow
		# error into that union at the propagation site (see emitter_c.py's
		# _emit_widen_error). Leaves compared by identity - every union is
		# interned by _get_or_create_union, so the SAME A|B object backs both
		# an op's error and a matching annotation.
		return_type = fn.return_type if fn is not None else None
		spec = self._as_specialization( return_type )
		covered = False
		if fn is not None and spec is not None and spec.base is result_cls and len( spec.args ) == 2:
			fn_error_leaves = self._atomic_leaves( spec.args[1] )
			covered = all( leaf in fn_error_leaves for leaf in self._atomic_leaves( error_cls ))
		if not covered:
			want = ' | '.join( sorted( leaf.stem for leaf in self._atomic_leaves( error_cls )))
			where = f'{fn.qualname} returns {return_type.qualname if return_type else None}' if fn is not None else 'this is not inside a function'
			self.discovery.fail(
				f'{ast.unparse(node)} requires the enclosing function to return Result[_,{want}] '
				f'(or a wider union covering it) ({where}) - {alternatives}',
				node,
			)

	def _resolve_sys_function( self, name: str ) -> Function:
		''' resolve and cache a real stdlib function (sys.panic, sys.alloc,
		sys.free, ...) — reached via discovery.import_name rather than
		user-namespace lookup. '''
		cached = self._sys_functions.get( name )
		if cached is not None:
			return cached
		module = self.discovery.import_name( 'sys' )
		fn = module.get_local( name )
		assert isinstance( fn, Function ), f'sys.{name} is required but was not found: {fn!r}'
		if fn.resolve is not None:
			fn.resolve()
		self._sys_functions[name] = fn
		return fn


	# --- namespace resolution (moved from lowering.py) ----------------------

	def _try_resolve_namespace( self, node: ast.expr ) -> Name|None:
		''' a silent probe: is this expression a compile-time-resolvable
		namespace path (a free function, or Class.staticmethod reached
		by class name)? Returns None rather than failing for an ordinary
		value expression — that means the caller should do receiver-based
		resolution. '''
		if isinstance( node, ast.Constant ) and node.value is None:
			# a bare `None` used as a TYPE reference (sys.alloc[None](...),
			# compiler.sizeof(None), Ptr[None]'s own inner arg) - same
			# NoneType-literal special case discovery.py's own annotation
			# resolution already applies everywhere else
			return self.discovery.get_none_type()
		if isinstance( node, ast.Name ):
			result = self.discovery.find_name( node.id, node )
			if getattr( result, 'resolve', None ) is not None:
				result.resolve()
			return result
		if isinstance( node, ast.Attribute ):
			base = self._try_resolve_namespace( node.value )
			if base is None:
				return None
			self.ensure_resolved( base )
			if isinstance( base, TaggedUnion ):
				self.union_storage.get( base )
			names = getattr( base, 'names', None )
			if not isinstance( names, dict ):
				return None
			result = names.get( node.attr )
			if getattr( result, 'resolve', None ) is not None:
				result.resolve()
			return result
		if isinstance( node, ast.BinOp ) and isinstance( node.op, ast.BitOr ):
			# T|None (or any X|Y) used as a generic type ARGUMENT
			# (list[str|None]()) - discovery.py's own visit_BinOp already
			# builds a TaggedUnion for this exact shape in ordinary
			# ANNOTATION position (-> T|None, x: T|None); this function is
			# the parallel path for a type reference reached through a
			# subscript's [...] (Name[T], Callable[[...],T]) rather than an
			# annotation, and previously had no case for it at all - fell
			# through to `return None` below, which the Name[T]/Callable[...]
			# branches just below then reported as "argument is not a type"
			# even though X|Y is a perfectly real type. Delegates to the
			# SAME _flatten_union/_get_or_create_union discovery.py's own
			# visit_BinOp uses, so the two paths canonicalize identically.
			operand_nodes = self.discovery._flatten_union( node )
			operands: list[Type] = []
			for operand_node in operand_nodes:
				resolved = self._try_resolve_namespace( operand_node )
				if not isinstance( resolved, Type ):
					return None
				operands.append( resolved )
			return self.discovery._get_or_create_union( operands )
		if (
			isinstance( node, ast.Subscript ) and isinstance( node.value, ast.Name )
			and node.value.id in ( 'move', 'copy', 'Callable', 'Closure', 'tuple', 'Iterator', 'Generator' )
		):
			# move[T]/copy[T]/Callable[[Arg1,...],Ret]/Closure[[Arg1,...],Ret]/
			# tuple[T0,T1,...]/Iterator[T]/Generator[T,E] as a TYPE-REFERENCE-
			# context expression (compiler.cast(Closure[[],None], x),
			# compiler.sizeof(Callable[...]), an explicit generic construction
			# call's own type argument like list[tuple[str,str]](), ...) - all
			# of these are recognized TEXTUALLY in discovery.py's own
			# visit_Subscript, not through find_name (a bare
			# ast.Name(id='tuple') node.value would otherwise fail resolution
			# outright, same as any other undefined name - this was exactly
			# the "name 'tuple' is not defined" bug on list[tuple[str,str]]()).
			# Reuse that same implementation directly rather than duplicating
			# it here (this used to hand-roll just the Callable/Closure case)
			# so every textually-special subscript form resolves identically
			# whether it appears in an ANNOTATION or as an explicit type
			# argument to a generic constructor CALL
			return self.discovery.visit_Subscript( node )
		if isinstance( node, ast.Subscript ):
			base = self._try_resolve_namespace( node.value )
			# Name[T] - a generic FUNCTION (mylen[i32]), a generic CLASS
			# construction (list[i32]()), or an intrinsic generic pointer
			# scalar (Ptr[u8]/ConstPtr[u8], as a type reference - e.g.
			# compiler.sizeof(Ptr[u8])) all share this same shape. Scalar
			# has no .resolve of its own (intrinsics aren't parsed from a
			# real source file - see Scalar's own docstring), unlike the
			# other two - getattr rather than a bare access
			if not isinstance( base, ( Function, RCClass, CStruct, CUnion, TaggedUnion, CEnum, Scalar )) or not getattr( base, 'type_params', None ):
				return None
			resolve = getattr( base, 'resolve', None )
			if resolve is not None:
				resolve()
			arg_nodes = node.slice.elts if isinstance( node.slice, ast.Tuple ) else [ node.slice ]
			if len( arg_nodes ) != len( base.type_params ):
				self.discovery.fail(
					f'{base.qualname}[...] expects {len(base.type_params)} type argument(s), got {len(arg_nodes)}: {ast.unparse(node)}',
					node,
				)
			args: list[Type] = []
			for a in arg_nodes:
				resolved = self._try_resolve_namespace( a )
				if not isinstance( resolved, Type ):
					self.discovery.fail( f'{base.qualname}[...] argument is not a type: {ast.unparse(a)}', node )
				args.append( resolved )
			return self.discovery._get_or_create_specialization( base, args )
		return None

	def _attr_lookup_callable( self, owner_type: Type|None, attr: str, ctx: ast.AST ) -> Function|Overload:
		owner_type = self.ensure_resolved( owner_type )
		if isinstance( owner_type, Specialization ) and isinstance( owner_type.base, Scalar ) and owner_type.base.stem in ( 'Ptr', 'ConstPtr' ):
			# dot-operator on a raw pointer means arrow - see lowering.py's
			# _attr_lookup's identical redirect for the non-callable case -
			# EXCEPT for a dunder Ptr[T]/ConstPtr[T] registers on ITSELF
			# (__str__/__repr__/__eq__/etc - lib/builtins/__ptr_arith.py):
			# those are the pointer's OWN protocol methods, not something
			# meant to be reached by dereferencing first (same precedent
			# Python itself follows - len(x) always calls type(x).__len__(x),
			# never something found by chasing through x's own contents).
			# Confirmed via a real repro: str(some_ptr) (builtins.str.
			# __call__'s own generic `return x.__str__()` body, ordinary
			# dot-call syntax) used to redirect through the arrow rule and
			# find the POINTEE's own __str__ instead (e.g. u8's, for
			# Ptr[u8]), then pass the raw pointer where a plain scalar
			# value was expected - a real type-confusion crash at C emission
			# time, not just the wrong answer.
			is_dunder = attr.startswith( '__' ) and attr.endswith( '__' )
			own_dunder = owner_type.base.names.get( attr ) if is_dunder else None
			if own_dunder is None:
				owner_type = self.ensure_resolved( owner_type.args[0] )
		if isinstance( owner_type, ( CStruct, RCClass )):
			found = owner_type.chain_lookup( attr )
		else:
			names = getattr( owner_type, 'names', None )
			if not isinstance( names, dict ):
				self.discovery.fail( f'{owner_type!r} has no members, cannot look up {attr!r} ({ast.unparse(ctx)})', ctx )
			found = names.get( attr )
		if (
			isinstance( found, Function ) and found.type_params
			and isinstance( owner_type, Specialization ) and isinstance( owner_type.base, Scalar )
			and owner_type.base.stem in ( 'Ptr', 'ConstPtr' )
		):
			# the own-dunder case just above found a bare generic Function
			# (Ptr[T]'s own dunders are registered unspecialized - T is only
			# ever bound from the receiver's own concrete pointee type at
			# the call site, never at registration time - see lowering.py's
			# _resolve_receiver_generic_dunder, the identical fix for the
			# operator-dispatch path) - same binding needed here, otherwise
			# `found` still carries the unbound TypeVar T and crashes
			# emitter_c.py's c_type at prototype-emission time.
			spec = self.discovery._get_or_create_specialization( found, list( owner_type.args ))
			found = self.monomorphizer.monomorphized_function( spec )
		if isinstance( found, Specialization ):
			# a Scalar-registered generic method (`i32.to_u32 = i__to__i[i32,u32]`)
			# - discovery.py's visit_Assign stores the raw Specialization,
			# unmonomorphized (no Monomorphizer exists that early) - resolve
			# it to the real, concrete Function here, on first actual use,
			# same as lowering.py's own _resolve_scalar_name does for the
			# other two Scalar.names readers (_find_method/_find_dunder_for_arg)
			found = self.monomorphizer.monomorphized_function( found )
		if not isinstance( found, ( Function, Overload )):
			self.discovery.fail( f'{attr!r} is not callable on {owner_type.qualname if owner_type else "?"}', ctx )
		if isinstance( found, ( Function, Overload )):
			self._resolve_callable( found )
		return found

	def resolve_declared_types( self, fn: Function ) -> None:
		''' resolve()s `fn` (if not already) then eagerly monomorphizes any
		fully-concrete, ClassLike-based Specialization directly typing one
		of its own declared parameters or its return type - the SAME
		eager-monomorphize step Monomorphizer.substitute_type_params
		already applies to a SUBSTITUTED field/parameter (monomorphize.py,
		the Specialization branch), just for a PLAIN, never-substituted
		declaration (an ordinary function's own `def f(x: list[i32])`, an
		@overload candidate's own parameter, ...), which never goes
		through substitute_type_params at all - discovery.py's own
		annotation resolver (_get_or_create_specialization) is the only
		thing that ever builds its .type/.parameters[*].type, and stops
		there, at the bare Specialization wrapper. Without this, a
		generic-substituted argument type (already monomorphized to the
		real RCClass by substitute_type_params - see its own TaggedUnion/
		Specialization branches) and an @overload candidate's own plain
		`list[i32]` parameter end up as two DIFFERENT kinds of object for
		the identical instantiation - a real RCClass vs. a bare
		Specialization wrapper - which `is` can never bridge no matter how
		well the Specialization layer itself is interned (confirmed via a
		real repro: PLAN_COMPILER_BUG_SWEEP.md's overload_resolution.py
		fix, which papered over this with an injected `_same_type`
		predicate instead of closing the gap here, at its actual source).

		PLAN_RESOLVE_CLASS_SPECIALIZATIONS.md's own "Source 2" - proposed,
		attempted, and reverted (18 test failures) before origin-tracking
		(_as_specialization/_same_type) existed to keep the many
		`isinstance(t, Specialization)` shape-checks elsewhere working once
		the type they're checking is no longer wrapped. That mechanism is
		now in place (see `_result_shape`/`_require_result_return`, both
		already `_as_specialization`-based) - this only wires the two real
		production callers of `overload_resolution.resolve_call` through
		this method (both already the sole place a Function/Overload's own
		members get `.resolve()`d for a real call site), the narrowest
		slice of the original plan that closes the specific duality this
		was found through, not the full "every declared type everywhere"
		sweep the original plan scoped - that stays a separate, bigger
		piece of work if it's ever wanted. '''
		if fn.resolve is not None:
			fn.resolve()
		for param in fn.parameters or []:
			param.type = self._eagerly_monomorphize_declared_type( param.type )
		fn.return_type = self._eagerly_monomorphize_declared_type( fn.return_type )

	def _eagerly_monomorphize_declared_type( self, t: Type|None ) -> Type|None:
		if (
			isinstance( t, Specialization ) and isinstance( t.base, ( RCClass, CStruct, CUnion, TaggedUnion, CEnum ))
			and self.monomorphizer._is_concrete( t )
		):
			self.schedule( t )
			return self.monomorphizer.monomorphize_class( t )
		return t

	def _resolve_callable( self, callee: Function|Overload ) -> None:
		if isinstance( callee, Function ):
			self.resolve_declared_types( callee )
		else:
			for fn in ( *callee.stubs, *callee.implementations ):
				self.resolve_declared_types( fn )

	def _resolve_union_receiver_members( self, union: TaggedUnion, members: list[Variable], attr: str, ctx: ast.AST ):
		# imported here to avoid circular dependency
		from union_storage import ReceiverDispatch
		per_leaf: list[tuple[Variable,Function]] = []
		for member in members:
			if member.resolve is not None:
				member.resolve()
			found = self._attr_lookup_callable( member.type, attr, ctx )
			if not isinstance( found, Function ):
				self.discovery.fail(
					f'{member.type.qualname if member.type else "?"}.{attr} is an overload group - calling an overloaded '
					f'method through a union receiver is not supported yet: {ast.unparse(ctx)}',
					ctx,
				)
			self.ensure_resolved( found )
			per_leaf.append(( member, found ))

		for member, fn in per_leaf:
			if fn.is_move:
				# the receiver-move-hook (lowering.py's _lower_call, gated
				# on isinstance(target, Function)) never fires for a
				# ReceiverDispatch target at all - calling an @move method
				# through a union receiver would neither track ownership
				# correctly nor error, so it's rejected outright here
				# instead (matching this same union-receiver resolution's
				# own existing return-type/param-count checks below)
				self.discovery.fail(
					f'{union.qualname}.{attr}(...): calling an @move method through a union-typed receiver is not '
					f'supported - leaf {member.type.qualname if member.type else "?"}.{attr} is @move-decorated',
					ctx,
				)

		reference = per_leaf[0][1]
		for member, fn in per_leaf[1:]:
			# _same_type, not raw `is` - two leaves' own independently-
			# resolved return-type annotations can be genuinely equal
			# generic instantiations (e.g. both list[i32]) reached through
			# two different Specialization objects (one substituted during
			# a generic leaf class's own monomorphization, one built fresh
			# from a concrete leaf's own annotation) - the same duality
			# _check_assignable/_unify_type_param already guard against
			# elsewhere (see TypeResolver._same_type's own docstring).
			# Without this, PLAN_COMPILER_BUG_SWEEP.md's own audit found a
			# real false-positive "leaf implementations disagree" here for
			# two leaves whose return types were textually identical.
			if not self._same_type( fn.return_type, reference.return_type ):
				self.discovery.fail(
					f'{union.qualname}.{attr}(...): leaf implementations disagree on return type '
					f'({reference.cls.qualname if reference.cls else "?"}.{attr} -> '
					f'{reference.return_type.qualname if reference.return_type else "None"}, '
					f'{member.type.qualname if member.type else "?"}.{attr} -> '
					f'{fn.return_type.qualname if fn.return_type else "None"})',
					ctx,
				)
			if len( fn.parameters or [] ) != len( reference.parameters or [] ):
				self.discovery.fail(
					f'{union.qualname}.{attr}(...): leaf implementations have differing parameter counts, not supported yet',
					ctx,
				)
		return ReceiverDispatch( union = union, attr = attr, per_leaf = per_leaf )

	def _resolve_callee_target( self, func_node: ast.expr ) -> Function|Overload|Specialization|object|None:
		''' resolve the textual call target to a Function/Overload/
		Specialization, or a _ReceiverDispatch for union receiver calls.
		Returns None if this needs receiver-based resolution
		(some_local.method(...)) — the caller must lower the receiver
		and resolve through its type. '''
		namespace_result = self._try_resolve_namespace( func_node )
		if isinstance( namespace_result, ( Function, Overload )):
			self._resolve_callable( namespace_result )
			return namespace_result
		if isinstance( namespace_result, Specialization ) and isinstance( namespace_result.base, Function ):
			return namespace_result
		return None  # caller must resolve through receiver type
	
	def schedule( self, unit: object ) -> None:
		# moved verbatim from Compiler._enqueue - lowering.py hands this
		# anything it comes across (a Function, a class, a Variable, a
		# Specialization, even a Module walked mid-namespace-lookup)
		# without needing to know which of those are real compile units.
		# A Specialization decomposes into its base + each type arg
		# (recursively - Result[Result[i32,E1],E2] schedules i32/E1/E2 too);
		# anything that isn't a Function, a ClassLike, or a genuinely
		# module-level Variable (Variable.is_global - the same class also
		# represents class attributes and lowering.py's own local
		# variables, neither a standalone unit) is silently ignored rather
		# than enqueued

		def _record( self ) -> None:
			if self._current_trigger is not None:
				self._triggered_by[ id( unit ) ] = self._current_trigger

		if isinstance( unit, Specialization ):
			if isinstance( unit.base, Function ):
				with self._seen_lock:
					if id( unit ) in self._seen:
						return
					self._seen.add( id( unit ))
				self.queue.put( unit )
				_record( self )
				return
			if isinstance( unit.base, ( RCClass, CStruct, CUnion, TaggedUnion )):
				with self._seen_lock:
					if id( unit ) in self._seen:
						return
					self._seen.add( id( unit ))
				self.queue.put( unit )
				_record( self )
				for arg in unit.args:
					self.schedule( arg )
				if isinstance( unit.base, RCClass ):
					self._schedule_rcclass_destructor_deps( unit.base )
				if isinstance( unit.base, TaggedUnion ):
					self._schedule_uniontype_storage( unit.base )
				return
			self.schedule( unit.base )
			for arg in unit.args:
				self.schedule( arg )
			return
		if not isinstance( unit, ( Function, ClassLike )) and not ( isinstance( unit, Variable ) and unit.is_global ):
			return
		if isinstance( unit, RCClass ):
			self._schedule_rcclass_destructor_deps( unit )
		if isinstance( unit, TaggedUnion ):
			self._schedule_uniontype_storage( unit )
		with self._seen_lock:
			if id( unit ) in self._seen:
				return
			self._seen.add( id( unit ))
		self.queue.put( unit )
		_record( self )

	def triggered_by( self, unit: object ) -> str|None:
		return self._triggered_by.get( id( unit ))

	def next_unit( self ) -> object|None:
		try:
			return self.queue.get_nowait()
		except queue.Empty:
			return None

	def ensure_resolved( self, obj: object ) -> object:
		# moved verbatim from Lowering._ensure_resolved - resolving
		# (populating .names/.parameters/whatever) needs to happen
		# immediately, mid-statement, for whoever's asking - unlike
		# schedule(), which just queues obj for whenever the work queue
		# gets to it, this can't wait.
		#
		# unconditionally hands obj to schedule() too - schedule() is the
		# single place that judges what's actually a compile unit worth
		# queuing, what decomposes into more of those, and what to just
		# quietly ignore. Nothing here needs to know or duplicate that
		# judgment.
		#
		# the SINGLE place a Specialization gets swapped for the real,
		# substituted thing it stands in for: every caller MUST use the
		# returned value, not the object passed in, or they see the
		# abstract, unsubstituted base instead (Specialization.names/.resolve
		# are raw passthroughs to it - see mpy_types.py). schedule() still
		# gets the ORIGINAL Specialization (schedule() dispatches on
		# isinstance(unit, Specialization) to actually build/register the
		# real compile unit) - only the return value here is swapped
		resolve = getattr( obj, 'resolve', None )
		if resolve is not None:
			resolve()
		if isinstance( obj, Specialization ) and not self.monomorphizer._is_concrete( obj ):
			# a Specialization still mentioning a TypeVar (e.g. a still-
			# generic function's own declared return type, found via a bare-
			# name lookup that never bound its type params - see
			# _type_of_expr's Call-node handling) is not a real compile unit
			# and must never reach schedule()/monomorphize_class(): the
			# Specialization+ClassLike branch below has no concreteness
			# guard of its own (unlike _eagerly_monomorphize_declared_type's
			# identical check), so handing it a bare TypeVar-typed spec
			# silently built a bogus "concrete" class whose own fields were
			# still typed with that TypeVar - confirmed by a real repro (a
			# generic free function converting between two Result error
			# types, forwarding the same T success payload, crashed
			# emitter_c.py's c_type on the unresolved T). Same "any doubt,
			# bail" discipline as every other caller in this pass - hand
			# back the abstract Specialization unchanged rather than
			# corrupting it into a fake concrete one
			return obj
		if isinstance( obj, Function ):
			# PLAN_GENERATORS.md - must happen HERE, not deferred until obj's
			# own turn on the work queue: a call site needs obj's REAL return
			# type (the synthesized backing RCClass, once obj turns out to
			# contain a yield) immediately, to type its own call-result
			# binding - it can't wait for obj to actually be dequeued and
			# lowered, which might happen much later (or never, if nothing
			# else reaches it). A no-op for an ordinary, non-generator
			# function (checked first, cheaply, inside the method itself).
			self.ensure_generator_synthesized( obj )
		self.schedule( obj )
		if isinstance( obj, Specialization ):
			if isinstance( obj.base, Function ):
				monomorphized = self.monomorphizer.monomorphized_function( obj )
				# PLAN_GENERATORS.md Phase 3 (roadmap Phase 3) - a generic
				# generator function (`def gen[T](x: T) -> Iterator[T]:`)
				# only ever appears here as obj.base, never as `obj` itself
				# (obj is the Specialization wrapper) - the plain-Function
				# check above this branch never sees it. A caller resolving
				# gen[i32](...)'s return type needs THIS monomorphized
				# copy's real return type (the synthesized backing
				# RCClass), same eager-resolution requirement v1 already
				# needed for the non-generic case, just one level further
				# in through the Specialization indirection
				origin_stems = [ tv.stem for tv in obj.base.type_params ] if obj.base.type_params else None
				self.ensure_generator_synthesized( monomorphized, origin_stems )
				return monomorphized
			if isinstance( obj.base, ( RCClass, CStruct, CUnion, TaggedUnion, CEnum )):
				return self.monomorphizer.monomorphize_class( obj )
			# Scalar (Ptr[T]/ConstPtr[T], the intrinsic generic-pointer
			# scalars - see mpy_types.py's Scalar) has no monomorphization
			# support at all - .names/.resolve stay raw passthroughs to the
			# abstract base, same as always
		if isinstance( obj, TupleType ):
			# PLAN_TUPLE.md - the same "swap for the real, substituted
			# thing" spot the Specialization branch above uses, so every
			# EXISTING _ensure_resolved/ensure_resolved call site (there are
			# ~15+ across lowering.py/type_resolver.py) transparently
			# receives tuple_storage.TupleStorage's synthesized backing
			# RCClass instead of the bare, field-less TupleType, with zero
			# changes needed at any of those call sites - a TupleType is
			# never itself further specialized/monomorphized (it has no
			# type_params of its own to substitute), so this is a flat
			# swap, not a recursive one
			return self.tuple_storage.get( obj )
		return obj

	def _find_module_for( self, fn: Function ) -> Module:
		# same lookup as Lowering._find_module_for - Function/ClassLike/
		# Variable.file is always set to their owning module's .file, but
		# none of them keep a direct back-reference to the Module itself
		for module in self.discovery.modules.values():
			if module.file == fn.file:
				return module
		self.discovery.fail_loc( f'no module found owning {fn.qualname} (file={fn.file})', fn.file, fn.line )

	def resolve_function_body( self, fn: Function ) -> None:
		''' item 3's AST-rewrite pass: runs `_ReferenceResolver` once over a
		Function's body (`is None`/`is not None` against a TaggedUnion
		rewritten to a tag comparison, `match` statements rewritten to an
		equivalent if/elif/else chain, a call to a generic function tagged
		with its own resolved, monomorphized callee), so lowering.py never
		has to recognize any of the three itself. Memoized by id(fn.node) -
		Monomorphizer.monomorphized_function deep-copies .node for every
		specialization (each one gets its own independent id), so this
		naturally runs once per distinct specialization, not just once for
		a shared abstract body. That matters specifically for rewrite 3
		(generic-call resolution): it's substitution-DEPENDENT (which
		concrete function a nested `bar(t)` resolves to, inside `foo[T]`'s
		own body, depends on which T this copy of foo was bound to) -
		unlike rewrites 1/2, which stay substitution-independent (a union
		member's tag ordinal never changes between specializations, and a
		match pattern's own class reference is resolved by name, never by
		the subject's type - see _ReferenceResolver's own docstring), so
		those two would have been safe to run only once even back when
		.node was shared. Compiler._lower calls this once on a
		Specialization's own (abstract) unit.base first - so rewrites 1/2
		still only ever run against the abstract body - and again on the
		monomorphized copy itself once it's built, which is what actually
		resolves rewrite 3 against concrete, bound type arguments.
		Mirrors compile_time_transformer's own fold (discovery.py's
		_make_function_resolver) in spirit, not in this specific
		once-per-node-id memoization detail. '''
		if id( fn.node ) in self._body_resolved:
			return
		self._body_resolved.add( id( fn.node ))
		module = self._find_module_for( fn )
		with self.discovery.module_context( module ):
			with ( self.discovery.scope_context( fn.cls ) if fn.cls is not None else nullcontext() ):
				with self.discovery.scope_context( fn ):
					resolver = _ReferenceResolver( self, fn )
					# one bad top-level statement doesn't stop the rest of
					# the body from being processed (mirrors lower_function's
					# own per-statement try/except) - a statement that fails
					# is left completely unrewritten; whatever made it fail
					# gets reported again (same error, recorded first here)
					# when lowering.py's own _lower_stmt reaches it moments
					# later, so nothing is silently swallowed
					new_body: list[ast.stmt] = []
					for stmt in fn.node.body:
						try:
							result = resolver.visit( stmt )
						except CompileError:
							new_body.append( stmt )
							continue
						if isinstance( result, list ):
							new_body.extend( result )
						elif result is not None:
							new_body.append( result )
					fn.node.body = new_body

	def resolve_global_init( self, var: Variable ) -> None:
		''' resolve_function_body's sibling for a global variable's own
		initializer EXPRESSION (not a statement list) - without this,
		lower_global (lowering.py's run_global) never gets item 3's Call-
		visiting rewrite at all, specifically the "construction-call
		detection" branch (visit_Call, this class, below) that eagerly
		resolves a bare ClassName(...) construction target's own __init__
		signature. An ordinary function body always gets that treatment
		first (Compiler._lower's Function branch calls resolve_function_
		body before lower_function ever runs) - a global never did,
		because Compiler._lower's Variable branch went straight to lower_
		global. Confirmed as a real, reachable crash (not theoretical): a
		global initialized via a bare `ClassName()` construction call
		(going through a real __init__, unlike `ClassName.make(...)`'s own
		staticmethod path, which never hits this) trips lowering.py's own
		_try_lower_construct_call assert ("... was not resolved before
		construction") - PLAN_GLOBAL_INIT.md's own real-compile fixture
		(g_foo: Foo = Foo.make(1)) never exercised this path. Memoized by
		id(var.init), the same id()-keyed convention resolve_function_body
		uses (id(fn.node)) for the identical "idempotent even if reached
		twice" reason. '''
		if var.init is None:
			return
		if id( var.init ) in self._body_resolved:
			return
		self._body_resolved.add( id( var.init ))
		module = self._find_module_for( var )
		with self.discovery.module_context( module ):
			resolver = _ReferenceResolver( self, None )
			try:
				var.init = resolver.visit( var.init )
			except CompileError:
				# already recorded - lowering.py's own lower_global re-reaches
				# and re-reports the same failure moments later, same
				# recovery discipline as resolve_function_body's per-
				# statement try/except. var's own type-resolution failing is
				# no longer reachable here at all (Compiler._lower's Variable
				# branch raises RedundantCompilationError - silently - before
				# ever calling this method, once var.broken is set - see
				# mpy_types.Name.broken) - what CAN still land here is a
				# failure specific to THIS method's own construction-call
				# detection (e.g. constructing an instance of a class whose
				# OWN resolution is broken), unrelated to var itself
				pass

	def resolve_parameter_default( self, target: Function, param: Parameter ) -> None:
		''' resolve_global_init's sibling for a parameter's own DEFAULT VALUE
		expression - lowering.py's _lower_call_args lowers `param.default`
		directly at every CALL SITE that omits the argument, inside the
		CALLING function's own lowering, never as part of target's OWN body
		(resolve_function_body only ever walks fn.node.body - a parameter's
		default lives on fn.node.args instead) - and often before target
		itself has had its own turn on the compile-unit queue at all (a
		caller only needs target.resolve() to have populated .parameters,
		already guaranteed by the time _lower_call_args runs). Without this,
		a construction call embedded in a default (`def f(x: Foo = Foo()):
		...`) never gets item 3's eager __init__ pre-resolution, tripping
		lowering.py's own _try_lower_construct_call assert ("... was not
		resolved before construction") exactly the way an unresolved global
		initializer once did (see resolve_global_init) - confirmed as a
		real, reachable crash (not theoretical): a class constructed only
		ever as another function's own defaulted-parameter value, called
		from a THIRD function that omits that argument, reaches real
		lowering with its __init__ never pre-resolved. Memoized by
		id(param.default), the same idempotent-even-if-reached-twice
		convention every sibling here uses - a shared default can be
		lowered at more than one omitted-argument call site. '''
		if param.default is None:
			return
		if id( param.default ) in self._body_resolved:
			return
		self._body_resolved.add( id( param.default ))
		module = self._find_module_for( target )
		with self.discovery.module_context( module ):
			with self.discovery.scope_context( target ):
				resolver = _ReferenceResolver( self, None )
				try:
					param.default = resolver.visit( param.default )
				except CompileError:
					# same recovery discipline as resolve_global_init - already
					# recorded, and lowering.py's own _lower_call_args re-reaches
					# and re-reports the same failure moments later
					pass


class _ReferenceResolver( ast.NodeTransformer ):
	'''
	Walks one Function's body doing three rewrites (see resolve_function_
	body's own docstring for which of these are safe to run once against a
	shared, abstract body vs. which need their own run per specialization):

	1. `x is None` / `x is not None`, when x's type is a TaggedUnion with a
	   None member, becomes a plain `x.tag == N` / `x.tag != N` - lowering's
	   own ordinary Eq/NotEq Compare handling takes it from there with no
	   further union-specific code at all.
	2. `match subject: case Owner.Member(pattern): ...` becomes an ordinary
	   `if`/`elif`/`else` chain (subject assigned to a synthesized local
	   first, so it's only ever evaluated once) - ported from lowering.py's
	   own _stmt_Match/_match_pattern (same AST-synthesis technique, moved
	   here to run once upfront instead of lazily per lowering call).
	   Resolves each pattern's `Owner` by NAME (_try_resolve_namespace),
	   never by inferring the subject's own type - a pattern's class
	   reference is written directly in the source (`case Foo.Bar(x):`),
	   so no type tracking is needed for this rewrite at all, only for #1.
	3. A call to a generic function - `foo[str]('hi')` (explicit) or
	   `foo('hi')` (T inferred from the argument's own type) - gets tagged
	   `node.resolved_callee = <the monomorphized Function>` rather than
	   having node.func itself rewritten (see _try_resolve_generic_call).
	   lowering.py's _lower_call prefers this tag over its own
	   _resolve_callee whenever it's present, so a call resolved here never
	   goes through Specialization at all on the lowering side. Best-effort
	   and NEVER authoritative about failure: on any doubt (an argument
	   whose type this pass can't determine, a still-unbound TypeVar found
	   while walking an abstract generic body, a genuinely malformed call)
	   this leaves the node untagged and defers entirely to lowering.py's
	   own (unchanged, still fully correct) generic-call machinery - this
	   rewrite only ever gets MORE calls resolved earlier, never fewer.

	`self.locals` is a small, best-effort forward type tracker (seeded from
	parameters, updated through AnnAssign/plain Assign) - NOT a general
	expression-type system. It only has to answer "what's the current type
	of this dotted-name-or-call expression", for #1's own target search and
	#3's own implicit-call argument inference, which is why it's fine for
	`_type_of_expr` to just return None (skip rewriting) for any expression
	shape it doesn't recognize - real code
	overwhelmingly writes `if ptr is None:` against a bare local anyway.
	'''
	def __init__( self, resolver: TypeResolver, fn: Function|None ) -> None:
		self.resolver = resolver
		self.discovery = resolver.discovery
		self.fn = fn
		self.locals: dict[str,Type] = {}
		# fn is None for a global variable's own initializer expression
		# (resolve_global_init) - no parameters, no self, matching
		# lowering.py's own FunctionLowering( self, None ) convention for
		# the identical case (see run_global's docstring)
		if fn is not None:
			for param in fn.parameters or []:
				self.locals[param.stem] = param.type
			if fn.cls is not None and not fn.is_static and not fn.is_classmethod:
				self.locals['self'] = fn.cls
		self._label_id = 0
		# parallel to self.locals, but for a name CURRENTLY known to be one
		# of a non-empty SET of possible union members within the lexical
		# span of a match/if construct that narrowed it (see visit_Match's
		# own whole-dict snapshot/restore around each case, and its own
		# post-loop merge) - mirrors cfg.py's own _narrowed dict/narrow()/
		# unnarrow(), but at this AST-rewriting-pass level, so every
		# union-shaped rewrite below (is-None, type(x) is T, T|None
		# truthiness) that goes through _type_of_expr sees the NARROWED
		# leaf type instead of the name's outer declared type when the set
		# has collapsed to exactly one possibility - see the comment on
		# _match_union_member's own narrow_marker for the bug this closes
		# (a second union-shaped check on an already-narrowed name inside
		# the same arm). A single-arm narrow always starts as a one-
		# element list; merging two DISAGREEING but both-still-possible
		# arms (see visit_Match's own _merge_case_narrowing) unions them
		# into a longer list rather than discarding the fact entirely -
		# `_type_of_expr` only ever returns a narrowed TYPE for the
		# single-element case (a multi-element set has no one Type to
		# report, so a plain Name expression just falls back to its
		# declared type - a bounded, deliberate scope cut, see the plan's
		# own note on why this doesn't need a full UnionView).
		self._narrowed: dict[str,list[Type]] = {}

	# --- best-effort "type of this expression", Name/Attribute/Call only ---

	def _type_of_expr( self, node: ast.expr ) -> Type|None:
		if isinstance( node, ast.Constant ):
			# a bare literal's own, context-free type (42 -> builtins.int,
			# 'x' -> builtins.str, ...) - the SAME mapping Discovery.
			# visit_Constant uses for annotation-position constants, but
			# reimplemented with a SILENT lookup (find_name_or_none, not
			# visit_Constant's own find_name) - visit_Constant is right to
			# raise for an annotation (`x: int` with no `int` registered at
			# all really is an error), but this pass runs speculatively over
			# every implicit generic call's arguments, including ones with
			# no useful expected type to give a literal - `import_builtins
			# = False` test harnesses (or any dialect that never registers
			# 'int') would otherwise spuriously fail here rather than just
			# leaving this argument's type undetermined, same as any other
			# expression shape this pass can't confidently type. NOT the
			# same rule Lowering._expr_Constant uses for ordinary lowering
			# (which deliberately takes its type from the surrounding
			# expected_type, never defaults one) - that's a distinct,
			# context-driven rule for ordinary code; this one only matters
			# for inferring a still-unbound generic type parameter from a
			# literal argument, where no such context exists to draw from
			if node.value is None:
				return self.discovery.get_none_type()
			if isinstance( node.value, bool ):
				name = 'bool'
			elif isinstance( node.value, int ):
				name = 'int'
			elif isinstance( node.value, float ):
				name = 'float'
			elif isinstance( node.value, str ):
				name = 'str'
			else:
				return None
			return self.discovery.find_name_or_none( name )
		if isinstance( node, ast.Name ):
			narrowed = self._narrowed.get( node.id )
			if narrowed is not None and len( narrowed ) == 1:
				return narrowed[0]
			local_type = self.locals.get( node.id )
			if local_type is not None:
				return local_type
			# self.locals only ever gets populated from params/self/body-
			# locals (see __init__/visit_AnnAssign/visit_Assign above) -
			# never from a module-level global, so a global subject fell
			# through here as unresolvable, and everything downstream that
			# needs a real type (is-None narrowing chief among them -
			# _is_none_narrowing_shape bails outright when this returns
			# None) silently declined for a global the exact same way it
			# would for a genuinely undefined name. Ordinary scope-chain
			# name resolution already has a global's real declared type on
			# hand - fall back to it here, same as lowering.py's own name
			# resolution already does for a global read.
			found = self.discovery.find_name_or_none( node.id )
			if not isinstance( found, Variable ):
				return None
			# a global Variable's own .type is populated lazily (via its
			# .resolve callable, same as everywhere else in this pass that
			# hands a not-yet-resolved object onward - see this class's own
			# ensure_resolved) - a param/local's type is always already
			# resolved by the time self.locals records it, so this was
			# never needed above; a global reached here for the first time
			# in THIS function still has type=None until forced.
			self.resolver.ensure_resolved( found )
			return found.type
		if isinstance( node, ast.Attribute ):
			if isinstance( node.value, ast.Name ):
				# mirrors the ast.Name branch's own narrowed-lookup exactly,
				# for a single-level field subject (`self.field`/`x.field`)
				# narrowed via _narrow_subject_key's synthetic f'{base}::
				# {attr}' key. Without this, a field re-narrowed by e.g.
				# visit_While's own exit-narrowing (self.field provably
				# NoneType after the loop) still reported its plain
				# declared union type here - which made a SUBSEQUENT `self.
				# field is not None:` re-check still look like a genuine
				# union-vs-None comparison worth rewriting into a tag-Cmp
				# (`self.field.tag != ...`), instead of degenerating the
				# same way the already-narrowed Name case does. That
				# rewrite's own tag_expr then re-lowered `self.field` a
				# SECOND time at LOWERING time, where cfg.py's real
				# narrowing (unaware of type_resolver's own separate
				# tracker) DOES apply and extracts the narrowed NoneType
				# payload eagerly - so the outer `.tag` read landed on a
				# raw NoneType value with no such field at all. Confirmed
				# via a real repro (`while type(self.field) is T: self.
				# field = None` followed by `if self.field is not None:`).
				narrowed = self._narrowed.get( f'{node.value.id}::{node.attr}' )
				if narrowed is not None and len( narrowed ) == 1:
					return narrowed[0]
			owner_type = self._type_of_expr( node.value )
			if owner_type is None:
				return None
			owner_type = self.resolver.ensure_resolved( owner_type )
			if isinstance( owner_type, TaggedUnion ):
				self.resolver.union_storage.get( owner_type )
			names = getattr( owner_type, 'names', None )
			if not isinstance( names, dict ):
				return None
			found = names.get( node.attr )
			if isinstance( found, Function ) and found.is_property:
				# `obj.attr` reading a @property getter (no call parens) means
				# "call this zero-arg getter", same as lowering.py's own
				# _expr_Attribute is_property branch - this pass needs the
				# SAME reading so is-None narrowing (etc) fires for a property
				# read used DIRECTLY (`f.val is None`), not just through an
				# already-materialized local (`x = f.val; x is None`, which
				# worked fine already since x's tracked type comes from the
				# Assign branch below, not this one). Confirmed by a real
				# repro: `f.val is None` on a `usize|None`-returning property
				# fell through to `not isinstance(found, Variable)` below
				# (a property getter is a Function, never a Variable) and on
				# to lowering.py's flat Cmp, which can't compare a TaggedUnion
				# struct against None at all - mirrors _type_of_expr's own
				# Call-branch Function handling further down for the exact
				# same resolve-then-read-return_type reason
				self.resolver.resolve_declared_types( found )
				return found.return_type
			if not isinstance( found, Variable ):
				return None
			# a field Variable's .type is populated lazily too, exactly like a
			# global's (see the ast.Name branch's own comment above) - the
			# class's OWN resolve() (just forced via ensure_resolved above)
			# only runs its body_fn far enough to register each field's
			# Variable in .names, via _make_annotation_resolver's own separate
			# lazy .resolve; it does NOT force that resolver too. Confirmed by
			# a real repro: `resp.headers.get(...) is None` (a chained
			# field-access receiver, `resp.headers` a still-unresolved
			# HTTPHeaders-typed field) silently declined this whole rewrite -
			# found.type was still None - and fell through to lowering.py's
			# flat Cmp, which doesn't know how to compare a TaggedUnion
			# struct against None at all
			self.resolver.ensure_resolved( found )
			return found.type
		if isinstance( node, ast.Subscript ):
			# tuple[...]'s own constant-index element access ONLY (t[0]) -
			# mirrors lowering.py's _expr_Subscript tuple branch exactly
			# (same tuple_storage.tuple_type_for/valid-index logic), needed
			# so `t[0] is None` can narrow at all now that tuple[T|None,...]
			# construction actually works (a real repro: none_first[0] is
			# not None, on a tuple[str|None,i32] local, used to fall through
			# to _lower_is_comparison's own flat-Cmp path and emit invalid C
			# comparing a union STRUCT against a bare int). Every OTHER
			# subscript shape (list[T]/dict[K,V]/a user __getitem__, ...) is
			# deliberately left unresolved here - this class's own docstring
			# already documents that returning None for an unrecognized
			# shape is fine (narrowing just doesn't fire, same as any other
			# expression this best-effort pass doesn't understand), and
			# those shapes would need real generic-container type inference
			# this pass was never meant to duplicate from lowering.py
			owner_type = self._type_of_expr( node.value )
			if owner_type is None:
				return None
			owner_type = self.resolver.ensure_resolved( owner_type )
			tuple_type = self.resolver.tuple_storage.tuple_type_for( owner_type )
			if tuple_type is None:
				return None
			valid_index = (
				isinstance( node.slice, ast.Constant )
				and isinstance( node.slice.value, int )
				and not isinstance( node.slice.value, bool )
			)
			if not valid_index or not ( 0 <= node.slice.value < len( tuple_type.elem_types )):
				return None
			return tuple_type.elem_types[ node.slice.value ]
		if isinstance( node, ast.Call ):
			# PLAN_GENERATORS.md Phase 3 (roadmap Phase 3) - a call to a
			# GENERIC function (explicit gen[i32](...) or inferred
			# gen(...)) was already resolved by visit_Call, which tags
			# node.resolved_callee with the real, substituted,
			# already-generator-synthesized-if-applicable Function -
			# reuse it directly rather than re-deriving anything. Needed
			# specifically because the fallback below (_try_resolve_
			# callable_namespace) has no ast.Subscript case at all (gen
			# [i32](...)'s own node.func), and for a BARE inferred call
			# would resolve to the still-abstract, unbound generic
			# Function instead of this call's own concrete instantiation
			# - either way giving back the wrong (or no) type. _type_of_
			# expr always runs AFTER generic_visit has already visited
			# this same Call node (see e.g. visit_Assign's own ordering),
			# so this tag is always populated by the time we get here,
			# for every generic call - never just for generator ones
			resolved_callee = getattr( node, 'resolved_callee', None )
			if isinstance( resolved_callee, Function ):
				return resolved_callee.return_type
			target: object|None
			if isinstance( node.func, ast.Attribute ):
				if node.func.attr == 'or_return':
					# <result_expr>.or_return() - recognized by AST shape
					# alone, mirroring lowering.py's own _lower_call fix
					# (see that check's comment for the full story). Never
					# look for a real declared 'or_return' method here -
					# discovery.py now rejects defining one outright, on
					# ANY class, so names.get('or_return') below would
					# never find one anyway post-fix; and even when the
					# real library still had a hand-written one, scheduling
					# it via ensure_resolved() below was itself the bug
					# (confirmed directly: a self-referential/same-file T,
					# e.g. a method of Foo returning Result[Foo,E] and
					# or_return()-ing it from elsewhere in Foo, crashed with
					# "compiler.early_return(...) requires the enclosing
					# function to return Result[_,_]"). Result[T,E].
					# or_return() always returns T - read it straight off
					# the receiver's own Specialization args instead, no
					# scheduling, no real method lookup, needed at all.
					receiver_type = self._type_of_expr( node.func.value )
					if receiver_type is None:
						return None
					receiver_type = self.resolver.ensure_resolved( receiver_type )
					receiver_cls_base = (
						receiver_type.base if isinstance( receiver_type, Specialization ) else receiver_type
					)
					if (
						receiver_cls_base is self.discovery.find_name_or_none( 'Result' )
						and isinstance( receiver_type, Specialization ) and receiver_type.args
					):
						return receiver_type.args[0]
					return None
				receiver_type = self._type_of_expr( node.func.value )
				target = None
				if receiver_type is not None:
					receiver_type = self.resolver.ensure_resolved( receiver_type )
					names = getattr( receiver_type, 'names', None )
					if isinstance( names, dict ):
						target = names.get( node.func.attr )
				if target is None:
					target = self._try_resolve_callable_namespace( node.func )
			else:
				target = self._try_resolve_callable_namespace( node.func )
			if isinstance( target, Function ):
				# resolve_declared_types, NOT ensure_resolved - this is pure
				# type inference (what type would `x = ...` bind, not a real
				# call being lowered), so target itself never needs
				# scheduling as a compile unit here - ensure_resolved's
				# unconditional scheduling side effect (see its own
				# docstring) means an @inline target would otherwise get
				# compiled as real, dead, never-called code purely from
				# being assigned to a local (confirmed by a real repro: any
				# `x = receiver.some_inline_method()` reaches exactly this
				# line during type inference, before lowering.py's own,
				# already-inline-aware call-emission ever runs - same root
				# cause lowering.py's _resolve_call_target already carves
				# out for its own, later call site). resolve_declared_types
				# still does everything actually needed here: resolves
				# target's signature and (separately) schedules/monomorphizes
				# its OWN return type, just never target itself. Still need
				# ensure_generator_synthesized explicitly, though - unlike
				# scheduling, that one's genuinely still required here (a
				# generator's real return type only exists after synthesis -
				# ensure_resolved calls it for exactly this reason, see its
				# own PLAN_GENERATORS.md comment; dropping it broke real
				# `for x in a_generator_call():` type inference, confirmed
				# by a real repro, since it's a no-op for the overwhelming
				# majority of ordinary, non-generator functions anyway).
				# Order matters: ensure_resolved's own sequence is resolve()
				# THEN ensure_generator_synthesized (which itself checks
				# fn.return_type, so it needs the bare annotation populated
				# first) - resolve_declared_types' own eager monomorphize
				# step has to come LAST, after synthesis may have rewritten
				# return_type into a real GeneratorType, or it eagerly
				# monomorphizes the PRE-synthesis annotation instead
				# (confirmed by a real repro: reversing this order broke
				# even the most basic generator - "contains yield but is
				# not declared -> Iterator[T]" on a function that plainly
				# was).
				if target.resolve is not None:
					target.resolve()
				self.resolver.ensure_generator_synthesized( target )
				self.resolver.resolve_declared_types( target )
				return target.return_type if isinstance( target, Function ) else None
			if isinstance( target, Overload ):
				# an @overload-decorated method group (e.g. Result[T,E].
				# unwrap_or) - previously fell all the way through to the
				# `return None` below (neither a Function nor a ClassLike),
				# which meant a local assigned from one of these calls never
				# got its type tracked at all, silently disabling the
				# TaggedUnion truthiness rewrite (and any other rewrite in
				# this class) for it further down the same body
				return self._overload_call_return_type( target, node )
			if isinstance( target, ( RCClass, CStruct, CUnion, TaggedUnion, CEnum )):
				# a plain (non-generic) construction call, Foo(...) - its own
				# type is just the class itself. An IMPLICIT generic
				# construction (Result(...), inferring its own type params
				# from the call's arguments with no explicit subscript) is
				# deliberately not handled here - that's the classes half of
				# this work, not yet done
				return target
			if isinstance( target, Specialization ):
				# EXPLICIT generic construction, Holder[Box](...) -
				# _try_resolve_callable_namespace's own ast.Subscript case
				# (added alongside this) already built the concrete
				# Specialization; that IS this call's own result type
				# directly, no further inference needed (unlike the
				# implicit-construction case above, which this pass still
				# doesn't attempt). Confirmed via a real repro:
				# `h = Holder[Box](); if h.val is None: ...` never narrowed
				# at all without this - h's own tracked type fell through to
				# None here, same root cause _try_resolve_generic_
				# construction's own docstring already flagged.
				return target
			return None
		return None

	def _overload_call_return_type( self, group: Overload, node: ast.Call ) -> Type|None:
		''' best-effort return type of a call to an @overload group, for
		_type_of_expr's Call branch above. Mirrors lowering.py's own
		Overload dispatch (_lower_call's _resolve_original/stub_covers_call)
		closely enough that a local's TRACKED type here never disagrees with
		what lowering.py itself actually resolves it to - disagreeing would
		either wrongly trigger _rewrite_tagged_union_truthiness's rewrite for
		a name lowering later types as a plain scalar (synthesizing a bogus
		`.tag`/`.data` access on it) or wrongly skip the rewrite for one it
		types as a nullable union (see the unwrap_or()-with-no-arguments bug
		this whole call chain was added for: Result[T,E].unwrap_or's `default:
		T` stub is bound_to the plain `default: T|None = None` impl, but a
		zero-argument call only ever matches the impl's own broader
		signature, never the stub's - stub_covers_call is what tells the two
		cases apart). Any doubt at all - an argument this pass can't type,
		resolve_call itself raising - just returns None, same discipline as
		every other branch of _type_of_expr. '''
		for fn in ( *group.stubs, *group.implementations ):
			self.resolver.resolve_declared_types( fn )
		if any( kw.arg is None for kw in node.keywords ):
			return None
		arg_types = [ self._type_of_expr( a ) for a in node.args ]
		if any( t is None for t in arg_types ):
			return None
		kwarg_types: dict[str,Type] = {}
		for kw in node.keywords:
			kw_type = self._type_of_expr( kw.value )
			if kw_type is None:
				return None
			kwarg_types[kw.arg] = kw_type
		try:
			_, resolved = overload_resolution.resolve_call(
				group.stubs, group.implementations, arg_types, kwarg_types,
				qualname = group.qualname, same_type = self.resolver._same_type,
			)
		except CompileError:
			return None
		winning_stub = next( ( s for s in group.stubs if s.bound_to is resolved ), None )
		if winning_stub is None:
			return resolved.return_type
		call_slots: list[int|str] = [ *range( len( arg_types )), *kwarg_types.keys() ]
		arg_leaves: dict[int|str,tuple[Type,...]] = {
			**{ i: tuple( t.leaves() ) for i, t in enumerate( arg_types ) },
			**{ name: tuple( t.leaves() ) for name, t in kwarg_types.items() },
		}
		if overload_resolution.stub_covers_call( winning_stub, call_slots, arg_leaves, self.resolver._same_type ):
			return winning_stub.return_type
		return resolved.return_type

	# --- namespace resolution (Name/Attribute only - no Subscript here; ---
	# --- rewrite 3 below needs Subscript too, for Name[T](...)/Attribute ---
	# --- [T](...), but handles it itself rather than folding it in here, ---
	# --- since rewrites 1/2 never need it ---

	def _try_resolve_namespace( self, node: ast.expr ) -> Name|None:
		if isinstance( node, ast.Name ):
			return self.discovery.find_name( node.id, node )
		if isinstance( node, ast.Attribute ):
			base = self._try_resolve_namespace( node.value )
			if base is None:
				return None
			base = self.resolver.ensure_resolved( base )
			if isinstance( base, TaggedUnion ):
				self.resolver.union_storage.get( base )
			names = getattr( base, 'names', None )
			if not isinstance( names, dict ):
				return None
			return names.get( node.attr )
		return None

	# --- rewrite 3: generic function call resolution ---

	def _try_resolve_callable_namespace( self, node: ast.expr ) -> Name|None:
		''' a SILENT probe, unlike _try_resolve_namespace above (which is
		only ever used where the name MUST already be a real, defined
		namespace path - a match pattern's own Owner - so find_name's
		raise-and-record-an-error behavior is exactly right there).
		_try_resolve_generic_call runs against EVERY call site in a
		function body, including ones this pass has no business touching:
		sugar recognized entirely inside lowering.py itself before it ever
		does namespace resolution (move(x), a for-loop's own range(...),
		compiler.sizeof(...)/compiler.cast(...)/...) whose bare Name/
		Attribute head is never a real registered name at all, and
		ordinary receiver-based calls (x.method()) whose receiver is a
		plain LOCAL variable - fn.names only gains local entries
		incrementally as LOWERING itself walks the body (see lower_
		function's own fn.add_name calls), which hasn't happened yet at
		this pre-lowering pass, so a perfectly ordinary local is
		indistinguishable from a genuinely undefined name here. Either way
		this is simply "not a generic call this pass can resolve", never a
		real error to report - if a name really is undefined, it's reported
		exactly once, when something that already MUST call the real,
		raising find_name reaches it (a real namespace path elsewhere, or
		lowering.py's own call-site handling) '''
		if isinstance( node, ast.Name ):
			return self.discovery.find_name_or_none( node.id )
		if isinstance( node, ast.Attribute ):
			base = self._try_resolve_callable_namespace( node.value )
			if base is None:
				return None
			# resolve() alone (populating base's own member table) is cheap
			# and side-effect-free from this pass's point of view - it's
			# ensure_resolved's OTHER half, schedule() (queuing base, and
			# everything it transitively calls, for real compilation), that
			# must stay gated on node.attr actually being a real member.
			# Checking membership BEFORE scheduling is what makes this
			# method's own "silent probe, no side effects on a miss"
			# docstring true - a bare Name lookup above can easily land on
			# the wrong, unrelated base (a local variable shadowing a
			# same-named module-level function/class - fn.names isn't
			# populated yet at this pre-lowering pass, see this method's own
			# docstring), and unconditionally scheduling that wrong guess
			# used to pull in everything IT calls even on a confirmed miss
			resolve = getattr( base, 'resolve', None )
			if resolve is not None:
				resolve()
			names = getattr( base, 'names', None )
			if not isinstance( names, dict ) or node.attr not in names:
				return None
			base = self.resolver.ensure_resolved( base )
			if isinstance( base, TaggedUnion ):
				self.resolver.union_storage.get( base )
			names = getattr( base, 'names', None )
			if not isinstance( names, dict ):
				return None
			return names.get( node.attr )
		if isinstance( node, ast.Subscript ):
			# Name[T] - a generic FUNCTION, a generic CLASS construction
			# (explicit Holder[Box](...), matching _try_resolve_namespace's
			# own identical Subscript case above, just silent-on-any-doubt
			# instead of raising - this pass's own SILENT-probe discipline
			# throughout), or an intrinsic generic pointer scalar. Was
			# entirely unhandled before (this method had no ast.Subscript
			# case at all) - confirmed via a real repro: `h = Holder[Box]();
			# if h.val is None: ...` never narrowed at all, since
			# _type_of_expr(h) fell through to None here for h's own
			# initializing Call (whose func is exactly this Subscript
			# shape), same gap _try_resolve_generic_construction's own
			# docstring already flagged ("explicit-subscript construction
			# isn't even resolvable by name lookup today").
			base = self._try_resolve_callable_namespace( node.value )
			if not isinstance( base, ( Function, RCClass, CStruct, CUnion, TaggedUnion, CEnum, Scalar )) or not getattr( base, 'type_params', None ):
				return None
			resolve = getattr( base, 'resolve', None )
			if resolve is not None:
				resolve()
			arg_nodes = node.slice.elts if isinstance( node.slice, ast.Tuple ) else [ node.slice ]
			if len( arg_nodes ) != len( base.type_params ):
				return None
			args: list[Type] = []
			for a in arg_nodes:
				resolved = self._try_resolve_callable_namespace( a )
				if not isinstance( resolved, Type ):
					return None
				args.append( resolved )
			return self.discovery._get_or_create_specialization( base, args )
		return None

	def _try_resolve_generic_call( self, node: ast.Call ) -> tuple[Function,list[Type]]|None:
		''' is `node` a call to a generic function - explicit
		Name[T](...)/Attribute[T](...), or a bare call whose target just
		happens to be generic (T inferred from the arguments)? Returns the
		target Function and its concrete type args on success, None on
		ANY doubt at all (wrong argument count, a subscript element that
		isn't a type, an argument type this pass can't determine, a type
		param nothing binds) - a miss here is never a compile error, just
		a missed rewrite: visit_Call leaves node.func/node.resolved_callee
		untouched and lowering.py's own _resolve_callee/
		_lower_generic_function_call/_lower_inferred_generic_call (still
		fully intact) resolve/report it exactly as before, so this never
		needs to duplicate an error message or risk double-reporting one -
		same "fine to just return None" discipline as _type_of_expr '''
		func = node.func
		if isinstance( func, ast.Subscript ):
			# Name[T](...)/Attribute[T](...) - mirrors Lowering.
			# _try_resolve_namespace's identical Subscript branch
			base = self._try_resolve_callable_namespace( func.value )
			if not isinstance( base, Function ) or not base.type_params:
				return None
			if base.resolve is not None:
				base.resolve()
			arg_nodes = func.slice.elts if isinstance( func.slice, ast.Tuple ) else [ func.slice ]
			if len( arg_nodes ) != len( base.type_params ):
				return None
			args: list[Type] = []
			for a in arg_nodes:
				resolved = self._try_resolve_callable_namespace( a )
				if not isinstance( resolved, Type ) or isinstance( resolved, TypeVar ):
					return None # not a type at all, or a still-unbound TypeVar (walking an abstract generic body) - either way, not this pass's to resolve
				args.append( resolved )
			if not self._type_params_satisfy_bounds( base.type_params, args ):
				return None # a real TypeVar(bound=...) violation - bail so lowering.py's own _lower_generic_function_call reports it with full context
			return base, args

		target = self._try_resolve_callable_namespace( func )
		if not isinstance( target, Function ) or not target.type_params:
			return None
		if target.resolve is not None:
			target.resolve()
		args = self._infer_generic_args( node, target, target.type_params )
		if args is None:
			return None
		if not self._type_params_satisfy_bounds( target.type_params, args ):
			return None # bail so lowering.py's own _finish_generic_call reports the bound violation
		return target, args

	def _pair_call_args_for_inference( self, target: Function, node: ast.Call ) -> list[tuple[Parameter,ast.expr]]|None:
		''' a deliberately non-failing cousin of Lowering._match_call_args -
		used only to figure out which parameter each argument binds, for
		type inference purposes, never to validate the call site's shape
		(that's still entirely lowering.py's job, run unchanged whether or
		not this pass ends up resolving anything). Any shape this doesn't
		recognize (*args, **kwargs, too many positional, an unknown
		keyword) just returns None rather than failing - same discipline
		as _try_resolve_generic_call itself '''
		if target.parameters is None:
			return None
		if any( isinstance( a, ast.Starred ) for a in node.args ):
			return None
		if any( kw.arg is None for kw in node.keywords ):
			return None
		positional_params = [ p for p in target.parameters if not p.is_vararg and not p.is_kwarg and not p.is_kwonly ]
		if len( node.args ) > len( positional_params ):
			return None
		pairs = list( zip( positional_params, node.args ))
		for kw in node.keywords:
			param = next( ( p for p in target.parameters if p.stem == kw.arg and not p.is_vararg and not p.is_kwarg ), None )
			if param is None:
				return None
			pairs.append(( param, kw.value ))
		return pairs

	def _natural_literal_type( self, node: ast.Constant ) -> Type|None:
		''' a literal's own no-context default type, exactly mirroring
		Lowering._expr_Constant's expected_type-is-None branch - deliberately
		NOT the same mapping _type_of_expr's Constant branch uses (that one
		means what an ANNOTATION spelling would: `int` the arbitrary-precision
		class, `float` an alias for f32). Only for _infer_generic_args' own
		trust_literals path below, where the question is what type the
		argument literal will actually be lowered as. '''
		intrinsics = self.discovery.get_intrinsics()
		if node.value is None:
			return self.discovery.get_none_type()
		if isinstance( node.value, bool ):
			return intrinsics['bool']
		if isinstance( node.value, int ):
			return intrinsics['i32']
		if isinstance( node.value, float ):
			return intrinsics['f64']
		if isinstance( node.value, str ):
			return self.discovery.find_name_or_none( 'str' )
		if isinstance( node.value, bytes ):
			return self.discovery.find_name_or_none( 'bytes' )
		return None

	def _infer_generic_args(
		self, node: ast.Call, target: Function, type_params: list[TypeVar], *, trust_literals: bool = True,
	) -> list[Type]|None:
		# a BARE call to a generic function (mylen(a), no explicit [T]) - T
		# has to be inferred from the arguments' own (best-effort,
		# _type_of_expr-derived) types. Mirrors Lowering.
		# _lower_inferred_generic_call's unification, just against
		# source-level types instead of already-lowered IR operand types.
		# `type_params` is passed in explicitly, not read off target.
		# type_params, so this same inference is shared by a generic
		# FREE function's own type params (target IS the generic thing)
		# and a generic CLASS's own type params via its __init__ (target
		# is __init__, whose own .type_params is empty - the class's are
		# what's actually being solved for) - see _try_resolve_generic_
		# construction, mirroring Lowering._unify_type_param's identical
		# generalization
		#
		# trust_literals=False (construction's own call) refuses to let a
		# bare literal argument (Box(1)) contribute a binding via its own
		# context-free default type (_type_of_expr's Constant branch,
		# int/str/bool/float) - unlike a function call, a construction's
		# class type param can ALSO be pinned from an expected_type
		# Lowering._lower_generic_construction_args sees (b: Box[i32] =
		# Box(1) pins T=i32 before the literal is even lowered, giving it
		# an i32 hint directly) but this pass has no expected-type context
		# threaded through it at all - inferring T=builtins.int from the
		# literal's own default here instead would be an outright WRONG
		# answer, not just a missed one, silently building and compiling
		# an extra, incorrect Box[int] specialization alongside the real
		# Box[i32] (confirmed by a real repro, not just this reasoning) -
		# so this pass simply never trusts a literal for construction; a
		# class type param only ever inferable from one always falls
		# through to lowering's own, correct, expected_type-aware pass
		pairs = self._pair_call_args_for_inference( target, node )
		if pairs is None:
			return None
		bindings: dict[int,Type] = {} # id(TypeVar) -> the concrete Type it was inferred as
		for param, expr in pairs:
			if isinstance( expr, ast.Constant ):
				if not trust_literals:
					continue
				# a literal argument's inferred type must match what Lowering.
				# _expr_Constant will ACTUALLY tag it as once this pass's
				# binding turns the type param concrete (i32/f64/bool/str/
				# bytes/NoneType) - NOT _type_of_expr's Constant mapping, which
				# deliberately means the same thing an ANNOTATION would (42's
				# `int` is the arbitrary-precision class, 3.14's `float` is an
				# alias for f32). Using that mapping here bound T to the
				# annotation-int/float type instead, so the literal then failed
				# lowering's own compatible-stems check against its own
				# concrete (non-scalar, or narrower-float) parameter type
				actual = self._natural_literal_type( expr )
			else:
				actual = self._type_of_expr( expr )
			if actual is None or isinstance( actual, TypeVar ):
				continue # can't determine this one - not an error here, just doesn't contribute a binding (see the "missing" check below)
			if not self._unify_type_param( type_params, param.type, actual, bindings ):
				return None # conflicting inference - bail, let lowering's own _unify_type_param report it
		if any( id( tv ) not in bindings for tv in type_params ):
			return None # couldn't infer everything from what this pass could determine - lowering's own (stronger, IR-level) inference gets a full attempt
		return [ bindings[id(tv)] for tv in type_params ]

	def _try_resolve_generic_construction( self, node: ast.Call, target_cls: object, init: object ) -> tuple[RCClass,Function]|None:
		''' Foo(...) where Foo is a generic RCClass with a plain __init__
		(fallible - returns Result[None,_] - or not) and no base class -
		the construction analogue of _try_resolve_generic_call, mirroring
		Lowering._lower_generic_construction_args's own inference (unify
		the class's own type params from the constructor's arguments, via
		the same _infer_generic_args this pass already uses for generic
		function calls - see its own comment on being generalized over an
		explicit type_params list for exactly this reason). expected_type-
		based pinning is deliberately NOT replicated here - this pass has
		no expected-type context threaded through it at all (visit_Call
		never receives one), unlike lowering's own two-phase strategy - a
		construction inferable ONLY from expected_type, never from any
		argument, is simply left for lowering's existing, still fully
		correct fallback to resolve. Same "any doubt, bail" discipline as
		_try_resolve_generic_call throughout: explicit-subscript
		construction (Box[i32](...)) isn't even resolvable by name lookup
		today (_try_resolve_callable_namespace has no Subscript-over-a-
		class handling), and an overloaded __init__, a class with a base,
		a malformed __init__ return type (neither None nor Result[None,_] -
		left for lowering's own _init_fallibility to report), or no
		__init__ at all (bare field=value sugar) are all left untouched
		too - none of those are this pass's to resolve. Takes target_cls/
		init already resolved by visit_Call's own caller, rather than
		re-resolving them here, since that lookup has to happen
		unconditionally anyway (see visit_Call's own comment on why -
		lowering.py's own asserts rely on it) '''
		if not isinstance( target_cls, RCClass ) or not target_cls.type_params or target_cls.base is not None:
			return None
		if not isinstance( init, Function ) or init.type_params:
			return None
		none_type = self.discovery.get_none_type()
		if init.return_type is not none_type:
			# not the plain (non-fallible) shape - only a genuine
			# Result[None,_] (fallible __init__, SYNTAX.md) is otherwise
			# acceptable; the tagged callee/construction resolution itself
			# doesn't care WHICH shape init.return_type has (only lowering.
			# py's own _init_fallibility/_emit_fallible_construction, run
			# unchanged against the CONCRETE, substituted init built below,
			# ever branch on it) - this is purely "is it one of the two
			# legal shapes, or something malformed lowering should report"
			shape = self.resolver._result_shape( init.return_type )
			if shape is None or shape[0] is not none_type:
				return None
		args = self._infer_generic_args( node, init, target_cls.type_params, trust_literals = False )
		if args is None:
			return None
		if not self._type_params_satisfy_bounds( target_cls.type_params, args ):
			return None # bail so lowering.py's own _lower_generic_construction_args reports the bound violation
		spec = self.discovery._get_or_create_specialization( target_cls, args )
		concrete_cls = self.resolver.monomorphizer.monomorphize_class( spec )
		concrete_init = concrete_cls.get_local( '__init__' )
		if not isinstance( concrete_init, Function ) or concrete_init.broken:
			return None # shouldn't happen (monomorphize_class's own method loop always substitutes a plain __init__ too), but stay silent/consistent with this pass's own discipline rather than assert
		return concrete_cls, concrete_init

	def _type_params_satisfy_bounds( self, type_params: list[TypeVar], args: list[Type] ) -> bool:
		# a real TypeVar(bound=...) violation is CONFIRMED, not a doubt - but
		# this pass never calls discovery.fail itself (see _try_resolve_
		# generic_call's own docstring), so callers bail (return None) on a
		# False here, same as any other "not this pass's to resolve or
		# report" case, letting lowering.py's own Lowering._check_type_param_
		# bounds raise the real error with full node/context
		return all( tv.bound_satisfied_by( arg ) for tv, arg in zip( type_params, args ))

	def _unify_type_param( self, type_params: list[TypeVar], declared: Type|None, actual: Type|None, bindings: dict[int,Type] ) -> bool:
		# ported from Lowering._unify_type_param, minus the discovery.fail()
		# call on conflict - returns False instead, meaning "bail, this
		# isn't this pass's to resolve or report" (see
		# _try_resolve_generic_call's own docstring)
		if declared is None or actual is None:
			return True
		if any( declared is tv for tv in type_params ):
			existing = bindings.get( id( declared ))
			# _same_type, not a bare `is` - see Lowering._unify_type_param's
			# identical comment
			if existing is not None and existing is not actual and not self.resolver._same_type( existing, actual ):
				return False
			bindings[ id( declared )] = actual
			return True
		if isinstance( declared, Specialization ):
			# _as_specialization, not a bare isinstance(actual, Specialization)
			# check - actual may already be the real, monomorphized object
			# itself (not a Specialization wrapper) if substitute_type_params'
			# own eager-monomorphize step got to it first - see Monomorphizer.
			# origin_of's own docstring
			actual_spec = self.resolver._as_specialization( actual )
			if actual_spec is not None and declared.base is actual_spec.base:
				return all( self._unify_type_param( type_params, d, a, bindings ) for d, a in zip( declared.args, actual_spec.args ))
		return True # this parameter position doesn't mention any of type_params - nothing to infer here

	def _instanceof_args( self, expr: ast.expr ) -> tuple[ast.expr,ast.expr]|None:
		''' does `expr` have the shape `instanceof(x, T)`? Returns (x, T),
		or None. Shared by visit_Call (rewrites a bare instanceof(...) call
		to the equivalent `type(x) is T` Compare) and _try_desugar_type_is_if
		(rewrites a WHOLE if-statement whose condition is instanceof(x, T)
		directly to the equivalent match statement, without ever building an
		intermediate Compare at all) - both need this same (x, T) pair.
		`instanceof` is never a real registered name (this compiler has no
		runtime reflection/RTTI - same posture as
		move[T]/copy[T]/compiler.sizeof(...) elsewhere). '''
		if isinstance( expr, ast.Call ) and isinstance( expr.func, ast.Name ) and expr.func.id == 'instanceof' and len( expr.args ) == 2 and not expr.keywords:
			return expr.args[0], expr.args[1]
		return None

	def visit_Call( self, node: ast.Call ) -> ast.expr:
		instanceof_args = self._instanceof_args( node )
		if instanceof_args is not None:
			# sugar for `type(x) is T` (see visit_Compare's own
			# _rewrite_type_is_comparison) - rather than duplicate the
			# recognition logic, rewrite to the equivalent Compare here and
			# re-dispatch through self.visit() - same technique visit_Match's
			# own nested-match handling uses (a visit_X method returning an
			# entirely different node kind). This makes instanceof(x, T)
			# usable anywhere a bool expression is (an if condition, a
			# boolean AND/OR, assigned to a bool variable, ...), same as
			# `type(x) is T` itself - EXCEPT directly as an if-statement's
			# own condition, where _try_desugar_type_is_if intercepts it
			# before it ever reaches here (see visit_If)
			subject_expr, type_expr = instanceof_args
			compare = ast.Compare(
				left = ast.Call( func = ast.Name( id = 'type', ctx = ast.Load() ), args = [ subject_expr ], keywords = [] ),
				ops = [ ast.Is() ],
				comparators = [ type_expr ],
			)
			ast.copy_location( compare, node )
			ast.copy_location( compare.left, node )
			ast.copy_location( compare.left.func, node )
			return self.visit( compare )
		self.generic_visit( node )
		resolved = self._try_resolve_generic_call( node )
		if resolved is None:
			# construction-call detection: Box(...) / Namespace.Class(...)
			# — the class body and its __init__'s parameter list need to
			# be fully resolved before lowering ever reaches this call
			# site regardless (lowering.py's own asserts rely on it) -
			# resolve (populate .names/.attributes) unconditionally, then
			# separately try the same generic-construction inference
			# _try_resolve_generic_call already does for functions (see
			# _try_resolve_generic_construction's own docstring for why
			# most of this doesn't apply - fallible/overloaded/subclassed
			# __init__, a no-__init__ class, explicit Box[i32](...) - and
			# falls back to lowering's unchanged machinery whenever it
			# doesn't). Never schedules anything itself either way - a
			# generic class must never become a compile unit until a
			# concrete Specialization is built from it, and node.
			# resolved_construction's own concrete_cls/concrete_init get
			# scheduled the ordinary way, by lowering.py, exactly once,
			# whenever it actually reaches this call site - same
			# discipline as node.resolved_callee above
			target = self._try_resolve_callable_namespace( node.func )
			if isinstance( target, ( RCClass, CStruct, CUnion, TaggedUnion, CEnum )):
				if isinstance( target, ( RCClass, CStruct )):
					# a CHAIN lookup, not target.get_local('__init__') alone -
					# a subclass with no own __init__ construction-lowers
					# through its nearest ANCESTOR's __init__ instead (see
					# lowering.py's own _try_lower_construct_call, which
					# looks up the exact same way and depends on this pass
					# having already resolved+scheduled whichever __init__
					# it's about to find, own or inherited). chain_lookup's
					# own walk resolves target AND every ancestor as a side
					# effect, replacing the plain target.resolve() call the
					# other branch below still needs for itself
					init = target.chain_lookup( '__init__' )
				else:
					if target.resolve is not None:
						target.resolve()
					init = target.get_local( '__init__' )
				if isinstance( init, Function ) and not init.broken:
					self.resolver._resolve_callable( init )
				construction = self._try_resolve_generic_construction( node, target, init )
				if construction is not None:
					node.resolved_construction = construction
			return node
		target, args = resolved
		spec = self.discovery._get_or_create_specialization( target, args )
		# builds/caches the real, substituted Function directly (own deep-
		# copied body, own concrete parameters/return_type/names - see
		# Monomorphizer.monomorphized_function) WITHOUT scheduling `spec`
		# itself as a compile unit, unlike TypeResolver.ensure_resolved -
		# the tagged callee below gets scheduled the ordinary way, exactly
		# once, whenever lowering.py's own call-lowering (Lowering.
		# _ensure_resolved, in its plain-Function call path) actually
		# reaches this call site. Scheduling `spec` here TOO would queue
		# the exact same compiled function under two different identities
		# (the Specialization, AND the bare monomorphized Function it
		# caches) - compiler.py would then lower and emit it twice
		node.resolved_callee = self.resolver.monomorphizer.monomorphized_function( spec )
		# PLAN_GENERATORS.md Phase 3 (roadmap Phase 3) - this is the ONLY
		# path that resolves a NESTED generic call (foo's own body calling
		# bar[T](...)) - it deliberately never goes through ensure_
		# resolved (see the comment just above), so it's also the only
		# place that can catch a generic call whose target turns out to
		# be a generator here. Idempotent/id(fn)-memoized, so this is safe
		# to call even when ensure_resolved's own Specialization branch
		# ALSO reaches the exact same memoized monomorphized_function
		# object via a different route (e.g. a caller assigning the call
		# result to a local, resolved through _type_of_expr instead)
		origin_stems = [ tv.stem for tv in target.type_params ] if target.type_params else None
		self.resolver.ensure_generator_synthesized( node.resolved_callee, origin_stems )
		return node

	# --- local imports ---

	def visit_ImportFrom( self, node: ast.ImportFrom ) -> ast.ImportFrom:
		''' registers a function-body-local `from X import Y` into fn.names,
		mirroring lowering.py's own _stmt_ImportFrom (see its docstring: function
		bodies are deliberately never walked by discovery.py's own visitor, so
		an in-function import only ever gets registered for real at lowering
		time). Without this, a local import followed immediately by an
		annotation using the imported name (`from windows.kernel32 import
		HANDLE` then `h: HANDLE = ...`, both inside the same function) failed
		to compile with a spurious "name not defined" - visit_AnnAssign below
		resolves its annotation via self.discovery.visit(), which walks
		self.discovery.scope_stack (fn is pushed onto it for the whole of
		resolve_function_body's walk), but nothing had put the import's name
		into fn.names yet at that point; a plain (non-annotated) use of the
		same name was unaffected, since this pass never resolves ordinary
		value expressions by name the way it resolves annotations. Registering
		it here (rather than only at lowering time) closes that gap for this
		pass's own annotation resolution while leaving the ImportFrom node
		itself untouched in the body - lowering.py's _stmt_ImportFrom still
		runs against it normally afterward and re-registers the same cached
		Name object, which is harmless (add_name is a plain dict assignment). '''
		if self.fn is None:
			return node
		parts: list[str] = []
		if node.level:
			# same package-relative counting as discovery.py's own
			# visit_ImportFrom/lowering.py's own _stmt_ImportFrom
			package = self.discovery.module_stack[-1].package
			strip = node.level - 1
			parts.extend(( package.split( '.' )[:-strip] if strip else package.split( '.' )) if package else [] )
			if not parts:
				self.discovery.fail( f'unable to relative import from here: {ast.unparse(node)}', node )
		if node.module:
			parts.append( node.module )
		package = '.'.join( parts )
		try:
			mod = self.discovery.import_name( package )
		except FileNotFoundError as e:
			self.discovery.fail( str( e ), node )
		if not mod:
			self.discovery.fail( f'module {package!r} not found', node )
		for alias in node.names:
			item = mod.get_local_or_raise( alias.name )
			if item is None:
				self.discovery.fail( f'module {package} does not export {alias.name!r}', node )
			self.fn.add_name( alias.asname or alias.name, item )
		return node

	# --- local type tracking ---

	def _unnarrow_attr_target( self, target: ast.expr ) -> None:
		''' the Attribute-target counterpart of visit_AnnAssign/visit_Assign's
		own Name-target self._narrowed.pop() below - a real reassignment to
		a single-level field (`self.field = ...`) invalidates whatever THAT
		field was previously narrowed to, exactly like a rebound local does.
		Without this, a field narrowed by an outer construct (e.g. `if self.
		field is not None: self.field = other; if self.field is not None:
		...`) left this pass's OWN _type_of_expr reporting the STALE
		narrowed (non-union) type for the field even after it was
		reassigned - which made the INNER re-check's own shape-detection
		wrongly decline (a non-union "type" has no tag to check), silently
		leaving its body un-narrowed. Confirmed via a real repro: cfg.py's
		own (correctly-unnarrowed, via lowering.py's matching fix) real
		narrowing state disagreed with this stale advisory one, and the
		disagreement manifested as the inner body's own method call failing
		to resolve against the concrete leaf. Mirrors _narrow_subject_key's
		own single-level/bare-Name-base restriction - a chained or
		subscripted target was never narrowable via this key format to
		begin with, so there's nothing to invalidate for it. '''
		if isinstance( target, ast.Attribute ) and isinstance( target.value, ast.Name ):
			self._narrowed.pop( f'{target.value.id}::{target.attr}', None )

	def visit_AnnAssign( self, node: ast.AnnAssign ) -> ast.AnnAssign:
		self.generic_visit( node )
		if isinstance( node.target, ast.Name ):
			self.locals[node.target.id] = self.discovery.visit( node.annotation )
			# a fresh declaration invalidates whatever this name was
			# previously narrowed to (same reasoning as cfg.py's own
			# unnarrow() - a rebound name no longer denotes the union
			# member it was proven to hold) - safe to call on a name that
			# was never narrowed
			self._narrowed.pop( node.target.id, None )
		else:
			self._unnarrow_attr_target( node.target )
		return node

	def visit_Assign( self, node: ast.Assign ) -> ast.Assign:
		self.generic_visit( node )
		if len( node.targets ) == 1 and isinstance( node.targets[0], ast.Name ):
			self.locals[node.targets[0].id] = self._type_of_expr( node.value )
			# see visit_AnnAssign's own comment - an ordinary reassignment
			# invalidates prior narrowing
			self._narrowed.pop( node.targets[0].id, None )
		elif len( node.targets ) == 1:
			self._unnarrow_attr_target( node.targets[0] )
		return node

	def visit_NamedExpr( self, node: ast.NamedExpr ) -> ast.NamedExpr:
		# walrus (`x := expr`) - same local-type-tracking bookkeeping as
		# visit_Assign above, for parity: without this, an unhandled
		# NamedExpr still falls through to generic_visit fine (this class's
		# own self.locals/_type_of_expr are best-effort, tolerating unknown
		# shapes by returning None - see this class's own docstring), but a
		# walrus-bound name wouldn't get its type tracked here, so the
		# best-effort is/is-not-None truthiness rewrite and implicit-
		# generic-call-argument inference passes elsewhere in this class
		# wouldn't see it either. target is always a bare ast.Name per
		# Python's own grammar (walrus forbids attribute/subscript/tuple
		# targets at parse time).
		self.generic_visit( node )
		if isinstance( node.target, ast.Name ):
			self.locals[node.target.id] = self._type_of_expr( node.value )
			self._narrowed.pop( node.target.id, None )
		return node

	def visit_Attribute( self, node: ast.Attribute ) -> ast.expr:
		# CEnum member VALUE expressions (OSError.FileNotFoundError
		# used as a runtime value) - the base is a class, not a
		# runtime value. Lowering.py's _expr_Attribute owns the
		# actual int folding; this pass only guarantees the CEnum
		# itself is resolved and scheduled before lowering ever sees
		# this Attribute node - same division of labour as
		# visit_Compare (resolves union for tag ordinals) / visit_Match
		# (resolves union for member payloads), neither of which
		# rewrite the subject expression itself.
		self.generic_visit( node )
		# _try_resolve_callable_namespace (silent, via find_name_or_none)
		# rather than _try_resolve_namespace (raising, via find_name) -
		# compiler.wrap_arithmetic and similar intrinsics are recognized
		# textually by lowering.py and are never real registered names,
		# so a raising probe would spuriously record an error for them
		base = self._try_resolve_callable_namespace( node.value )
		if isinstance( base, CEnum ):
			# _try_resolve_callable_namespace only calls ensure_resolved
			# on recursive ast.Attribute traversal, not on a bare
			# ast.Name result - resolve now, which also schedules the
			# CEnum as a compile unit (TypeResolver.ensure_resolved
			# always calls schedule() on its argument)
			self.resolver.ensure_resolved( base )
		return node

	# --- rewrite 1: is/is not None ---

	def _type_call_subject( self, expr: ast.expr ) -> ast.expr|None:
		''' does `expr` have the shape `type(x)` - a bare, single-argument
		call to a Name literally spelled `type`? Returns x, or None if not.
		`type` is never a real registered name in this compiler (no
		runtime reflection/RTTI - see PLAN_SUBCLASSING_VTABLES_COM.md), so
		this is textually recognized special syntax, same posture as
		move[T]/copy[T]/compiler.sizeof(...) elsewhere - NOT an ordinary
		call needing resolution. '''
		if isinstance( expr, ast.Call ) and isinstance( expr.func, ast.Name ) and expr.func.id == 'type' and len( expr.args ) == 1 and not expr.keywords:
			return expr.args[0]
		return None

	# type(x) as a general, first-class type-reference-producing intrinsic
	# (usable anywhere a type is expected, not just inside an is/instanceof
	# comparison) was investigated - and deliberately deferred - alongside
	# adding compiler.sizeof(x)'s value-argument support (lowering.py's
	# _static_type_of_value_expr). Findings:
	#   - the ONLY existing meaning of type(x) is the textual recognition
	#     above, consumed exclusively by visit_Compare's is/is-not rewrite
	#     and visit_Call's instanceof(x, T) sugar - both compile-time-only,
	#     both require x's static type to be a TaggedUnion. There is no
	#     runtime type(x) callable, and no runtime reflection/type-object
	#     value anywhere in this compiler to build one on top of.
	#   - compiler.sizeof(x)'s value-argument fix already covers the
	#     motivating case (a self-escape exemption for "type-only" self
	#     use - self's static type, without evaluating self) directly:
	#     compiler.sizeof(self) reads self.type at compile time and never
	#     makes self an instruction operand, so it's self-escape-safe with
	#     zero changes to cfg.py. compiler.sizeof(type(self)) would just be
	#     a redundant second spelling of the same thing.
	#   - type(self) is T is ALSO already self-escape-safe today, for the
	#     one case where it's legal syntax (self must be TaggedUnion-typed,
	#     which an RCClass's self never is) - it rewrites to `self.tag ==
	#     N`, a GetAttr(obj=self, ...) read, and GetAttr.obj is already
	#     excluded from check_self_escape's operand list.
	# Conclusion: no new type(x) intrinsic added. If a real future
	# consumer needs a type-reference-position spelling of "x's own static
	# type" outside an is/instanceof comparison, the extension point is
	# here: generalize this method's callers beyond visit_Compare/
	# visit_Call to also let _try_resolve_namespace (lowering.py) or this
	# class's own _try_resolve_namespace recognize the type(x) call shape
	# and substitute _type_of_expr(x)/_static_type_of_value_expr(x).

	def _is_plain_field_attribute( self, node: ast.Attribute ) -> bool:
		''' True only when `node` ("x.attr") resolves to a genuine field
		(a Variable), never a @property getter (or anything else, e.g. a
		bound method) - visit_If's own single-level field-narrowing shape
		(`self.field is not None: ...`) must never fire for a property:
		re-reading it to build the narrow-marker's own extraction/re-checks
		would call a possibly side-effecting getter extra times, and
		_resolve_narrow_attr_member's own lowering.py-side lookup
		(_attr_lookup) only ever finds a plain field to begin with. Mirrors
		_type_of_expr's own ast.Attribute branch's owner/names lookup
		exactly, just checking isinstance(..., Variable) instead of the
		is_property Function case that branch handles. '''
		owner_type = self._type_of_expr( node.value )
		if owner_type is None:
			return False
		owner_type = self.resolver.ensure_resolved( owner_type )
		names = getattr( owner_type, 'names', None )
		if not isinstance( names, dict ):
			return False
		return isinstance( names.get( node.attr ), Variable )

	def _narrow_subject_key( self, subject_expr: ast.expr ) -> tuple[str,str|None,str|None] | None:
		''' resolves a narrowing SUBJECT expression (the thing a comparison
		proved something about) to the (key, attr_base, attr_name) triple
		every narrowing construct needs: `key` is what cfg.py's own
		narrow()/narrowed_member() dict is keyed by (fed to
		_build_narrow_marker as `name`), attr_base/attr_name are set only
		for a field subject and are what lowering.py's _resolve_narrow_
		attr_member needs to re-resolve the real field. Returns None for
		anything this can't narrow at all - a property getter, a chained
		attribute (`a.b.c`), or any expression shape other than a bare Name
		or a single-level field access.

		Shared by every construct that narrows an ORIGINAL subject
		expression directly: visit_If (`x is not None`/`x:`), visit_While's
		Phase 7 (`type(x) is T`), visit_Match's own wildcard-deduces-the-
		other-member narrowing. visit_Match's PATTERN-binding machinery
		(`case T(x):` reusing the match subject's own name) does NOT go
		through this - that's a distinct feature (binding a NEW name in a
		pattern, only meaningful for a bare Name subject to begin with, not
		a narrowing-key resolution) and stays exactly as restrictive as it
		already was.

		A field subject is deliberately restricted to a single level off a
		bare Name, and to a genuine field (never a @property getter) - see
		_is_plain_field_attribute's own docstring for why (re-evaluating a
		side-effecting getter an extra time to build the narrow-marker's
		own extraction/re-checks). '''
		if isinstance( subject_expr, ast.Name ):
			return subject_expr.id, None, None
		if (
			isinstance( subject_expr, ast.Attribute ) and isinstance( subject_expr.value, ast.Name )
			and self._is_plain_field_attribute( subject_expr )
		):
			base, attr = subject_expr.value.id, subject_expr.attr
			return f'{base}::{attr}', base, attr
		return None

	def _is_none_narrowing_shape( self, test: ast.expr ) -> tuple[ast.expr,TaggedUnion,list[Variable],Variable,bool]|None:
		''' recognizes `x is None` / `x is not None` against a union-typed
		x, resolving all the way through to the real (union, members,
		none_member) - shared shape-detection half of visit_Compare's own
		rewrite-1 below (this method IS that detection, factored out
		unchanged) and visit_If's own is-not-None narrowing. Returns
		(subject_expr, base, members, none_member, is_not), or None on ANY
		doubt - same "caller declines silently" philosophy _type_is_shape
		documents. Deliberately does NOT restrict how many non-None
		members the union has - visit_Compare's own boolean rewrite below
		needs no single narrowing target to build `x.tag != TAG_NONE`, only
		narrowing itself does (that restriction, matching
		_rewrite_tagged_union_truthiness's identical one, is applied by
		visit_If itself, same as visit_While applies its own extra
		restriction on top of the equally general _type_is_shape). '''
		if not ( isinstance( test, ast.Compare ) and len( test.ops ) == 1 and isinstance( test.ops[0], ( ast.Is, ast.IsNot ))):
			return None
		left_is_none = isinstance( test.left, ast.Constant ) and test.left.value is None
		right_is_none = isinstance( test.comparators[0], ast.Constant ) and test.comparators[0].value is None
		if left_is_none == right_is_none:
			return None # both-None/neither-None - not this rewrite's shape, leave for lowering's ordinary is/is-not handling
		subject_expr = test.comparators[0] if left_is_none else test.left
		subject_type = self._type_of_expr( subject_expr )
		if subject_type is None:
			return None # can't determine - leave as ordinary `is`/`is not`, lowering's own _lower_is_comparison handles the non-union fallback
		# _as_specialization, not a bare isinstance(subject_type, Specialization) -
		# subject_type may already be eagerly-monomorphized (resolve_declared_
		# types) to the concrete union itself; base must still resolve to the
		# ABSTRACT union so it agrees with whatever else compares against it
		# by identity (union_storage.get's own cache key, any caller that
		# resolves a pattern's Owner by NAME - always the abstract class).
		# Critically, must NOT call ensure_resolved(subject_type) first: that
		# would swap a Specialization for its MONOMORPHIZED copy, whose own
		# tag/data (already built by monomorphize_class) would collide with
		# UnionStorage.get() trying to synthesize them again as if for a
		# fresh union (same mistake, and fix, as lowering.py's
		# _lower_allocate_fields TaggedUnion branch had)
		spec = self.resolver._as_specialization( subject_type )
		base = spec.base if spec is not None else subject_type
		if not isinstance( base, TaggedUnion ):
			return None
		members = self._resolved_union_members( subject_type, base )
		none_type = self.discovery.get_none_type()
		none_member = next( ( attr for attr in members if attr.type is none_type ), None )
		if none_member is None:
			return None
		is_not = isinstance( test.ops[0], ast.IsNot )
		return subject_expr, base, members, none_member, is_not

	def _bare_truthiness_narrowing_shape( self, test: ast.expr ) -> tuple[ast.expr,TaggedUnion,list[Variable],Variable,bool]|None:
		''' `if x:` / `if not x:` against a union-typed, bare-Name x - a
		DIFFERENT shape from _is_none_narrowing_shape's own `is None`/`is
		not None` comparison, but narrows the same way. Only the TRUTHY
		case narrows: it always safely implies non-None (None is always
		falsy, so truthy entails not-None), regardless of whether the
		leaf's own __bool__ could ALSO be False for a real, non-None
		instance (e.g. an empty str) - a falsy leaf is still non-None. The
		FALSY case is deliberately left un-narrowed: it could be None OR a
		real-but-falsy leaf, so nothing new is provable there in general
		(unlike is-None narrowing's own else branch, which DOES prove
		non-None). Same single-non-None-member restriction as
		_is_none_narrowing_shape/_rewrite_tagged_union_truthiness. Returns
		the identical shape _is_none_narrowing_shape does so visit_If's
		existing narrowing machinery (built for that comparison case)
		drives this one too, unchanged - only is_not's OWN meaning differs
		here (True selects the TRUTHY branch, not the not-None one). '''
		is_not = True
		subject_expr = test
		if isinstance( test, ast.UnaryOp ) and isinstance( test.op, ast.Not ):
			subject_expr = test.operand
			is_not = False
		if not isinstance( subject_expr, ast.Name ):
			return None
		subject_type = self._type_of_expr( subject_expr )
		if subject_type is None:
			return None
		spec = self.resolver._as_specialization( subject_type )
		base = spec.base if spec is not None else subject_type
		if not isinstance( base, TaggedUnion ):
			return None
		members = self._resolved_union_members( subject_type, base )
		none_type = self.discovery.get_none_type()
		none_member = next( ( attr for attr in members if attr.type is none_type ), None )
		if none_member is None:
			return None
		return subject_expr, base, members, none_member, is_not

	def visit_Compare( self, node: ast.Compare ) -> ast.expr:
		self.generic_visit( node )
		if len( node.ops ) != 1 or not isinstance( node.ops[0], ( ast.Is, ast.IsNot )):
			return node
		# rewrite 2: type(x) is T / type(x) is not T - see
		# _type_call_subject's own comment for why this is checked before,
		# and instead of, the ordinary is/is-not-None rewrite below (a
		# type() call on either side is never itself a real None/union
		# value the other rewrite's shape-detection would otherwise try to
		# make sense of)
		left_subject = self._type_call_subject( node.left )
		right_subject = self._type_call_subject( node.comparators[0] )
		if left_subject is not None or right_subject is not None:
			if left_subject is not None and right_subject is not None:
				self.discovery.fail( f'type(...) is type(...): only one side may be a type() call: {ast.unparse(node)}', node )
			subject_expr = left_subject if left_subject is not None else right_subject
			type_expr = node.comparators[0] if left_subject is not None else node.left
			return self._rewrite_type_is_comparison( node, subject_expr, type_expr )
		# rewrite 1: x is None / x is not None
		shape = self._is_none_narrowing_shape( node )
		if shape is None:
			return node
		subject_expr, base, _members, none_member, is_not = shape
		tag_attr, _data_attr, _payload_cls, tags = self.resolver.union_storage.get( base )
		tag_expr = ast.Attribute( value = subject_expr, attr = tag_attr.stem, ctx = ast.Load() )
		ast.copy_location( tag_expr, node )
		op = ast.NotEq() if is_not else ast.Eq()
		result = ast.Compare( left = tag_expr, ops = [ op ], comparators = [ ast.Constant( value = tags[none_member.stem] ) ] )
		ast.copy_location( result, node )
		return result

	def _rewrite_type_is_comparison( self, node: ast.Compare, subject_expr: ast.expr, type_expr: ast.expr ) -> ast.expr:
		''' type(x) is T / type(x) is not T, and instanceof(x, T) (sugar for
		the same thing, see visit_Call). Only valid when x's own static
		type is a TaggedUnion (or Specialization of one) and T names one of
		its members - rewrites to the SAME tag-Cmp shape visit_Compare's
		own is/is-not-None rewrite and _match_union_member both already
		produce. This alone only ever yields a plain bool, usable anywhere
		a bool expression is (an if condition, a boolean AND/OR, assigned
		to a bool variable, ...) - NARROWING x's type inside an if-branch
		built from this is Phase 4's job (if-statement desugaring to
		match), not this rewrite's. '''
		leaf_type = self._try_resolve_namespace( type_expr )
		if leaf_type is None:
			self.discovery.fail( f'type(...) is ...: {ast.unparse(type_expr)} does not name a type: {ast.unparse(node)}', node )
		if isinstance( subject_expr, ast.Name ):
			narrowed = self._narrowed.get( subject_expr.id )
			if narrowed is not None:
				# x is already known to be one of a (possibly multi-element,
				# see visit_Match's own _merge_case_narrowing) SET of
				# possible members - a redundant type(x) is T check on an
				# already-narrowed name must fold to a constant here, not
				# fall into the ordinary union-type validation below
				# (_type_of_expr only returns a narrowed leaf TYPE for the
				# single-element case, so an already-fully-narrowed x would
				# otherwise hit "not a union type" - the exact bug this
				# closes). Foldable whenever the answer is DEFINITE: a
				# single remaining possibility (matches T or doesn't,
				# either way is certain), or T simply isn't among several
				# remaining possibilities at all (definitely not it).
				# T being ONE OF several remaining possibilities is
				# genuinely ambiguous - falls through to the ordinary path
				# below, which (since _type_of_expr won't return a narrowed
				# leaf for a multi-element set) resolves against x's own
				# full declared union, same as if it were never narrowed -
				# safe, just not maximally precise
				matches = any( t is leaf_type for t in narrowed )
				if len( narrowed ) == 1 or not matches:
					is_not = isinstance( node.ops[0], ast.IsNot )
					result = ast.Constant( value = matches != is_not )
					ast.copy_location( result, node )
					return result
		subj_type = self._type_of_expr( subject_expr )
		if subj_type is None:
			self.discovery.fail( f'type(...) is ...: cannot determine the type of {ast.unparse(subject_expr)}: {ast.unparse(node)}', node )
		spec = self.resolver._as_specialization( subj_type ) # not a bare isinstance check - subj_type may already be eagerly-monomorphized, see visit_Match's own comment
		base = spec.base if spec is not None else subj_type
		if not isinstance( base, TaggedUnion ):
			self.discovery.fail( f'type(...) is ...: {ast.unparse(subject_expr)} is not a union type: {ast.unparse(node)}', node )
		members = self._resolved_union_members( subj_type, base )
		member = next( ( attr for attr in members if attr.type is leaf_type ), None )
		if member is None:
			self.discovery.fail( f'{base.qualname} has no member of type {getattr( leaf_type, "qualname", leaf_type )}: {ast.unparse(node)}', node )
		tag_attr, _data_attr, _payload_cls, tags = self.resolver.union_storage.get( base )
		tag_expr = ast.Attribute( value = subject_expr, attr = tag_attr.stem, ctx = ast.Load() )
		ast.copy_location( tag_expr, node )
		op = ast.NotEq() if isinstance( node.ops[0], ast.IsNot ) else ast.Eq()
		result = ast.Compare( left = tag_expr, ops = [ op ], comparators = [ ast.Constant( value = tags[member.stem] ) ] )
		ast.copy_location( result, node )
		return result

	# --- rewrite 1b: T|None truthiness (if x: / while x:) ---

	def _rewrite_tagged_union_truthiness( self, expr_node: ast.expr, ctx_node: ast.AST ) -> ast.expr|None:
		''' if x: or while x: where x's type is a TaggedUnion that includes
		None — rewrite `x` to `x.tag != TAG_NONE and x.data.v_<T>.__bool__()`
		(or just `x.data.v_bool` when the leaf type IS bool).
		Returns None when the type isn't a TaggedUnion, has no None member,
		or has multiple non-None variants (auto-generated union __bool__ is
		future work).

		`not x` (a UnaryOp wrapping the same shape - e.g. `if not tz:`) is
		handled here too, by recursing on the unwrapped operand and negating
		the result - lowering.py's own _expr_UnaryOp assumes ANY `not`
		operand is already a plain scalar (`ir.Not`/emitter_c.py's bare
		`!operand`), which is invalid C for a TaggedUnion's struct
		representation; this rewrite runs first (visit_If/visit_While call
		it on their own node.test before any other visitation), replacing
		the whole `not x` with `not (tag_cmp and value_expr)` - both
		operands of that inner BoolOp are real bools, so the OUTER `not`
		lowers through the ordinary (correct) scalar path unchanged. '''
		if isinstance( expr_node, ast.UnaryOp ) and isinstance( expr_node.op, ast.Not ):
			inner = self._rewrite_tagged_union_truthiness( expr_node.operand, ctx_node )
			if inner is None:
				return None
			negated = ast.UnaryOp( op = ast.Not(), operand = inner )
			ast.copy_location( negated, ctx_node )
			return negated
		expr_type = self._type_of_expr( expr_node )
		if expr_type is None:
			return None
		# _as_specialization, not a bare isinstance check - expr_type may
		# already be eagerly-monomorphized (resolve_declared_types), see
		# visit_Match's own comment
		spec = self.resolver._as_specialization( expr_type )
		base = spec.base if spec is not None else expr_type
		if not isinstance( base, TaggedUnion ):
			return None
		if spec is not None:
			members = self.resolver.monomorphizer.monomorphize_class( spec ).attributes
		else:
			self.resolver.ensure_resolved( base )
			for attr in base.attributes:
				self.resolver.ensure_resolved( attr )
			members = base.attributes
		none_type = self.discovery.get_none_type()
		none_member = next( ( attr for attr in members if attr.type is none_type ), None )
		if none_member is None:
			return None # no None member — auto-generated union __bool__ is future work
		tag_attr, data_attr, _payload_cls, tags = self.resolver.union_storage.get( base )
		# synthesize: expr.tag != TAG_NONE
		tag_expr = ast.Attribute( value = expr_node, attr = tag_attr.stem, ctx = ast.Load() )
		ast.copy_location( tag_expr, ctx_node )
		tag_cmp = ast.Compare(
			left = tag_expr,
			ops = [ ast.NotEq() ],
			comparators = [ ast.Constant( value = tags[none_member.stem] ) ],
		)
		ast.copy_location( tag_cmp, ctx_node )
		# synthesize: expr.data.v_<T> (or .__bool__() on it for non-bool leaf)
		non_none = [ m for m in members if m.type is not none_type ]
		if len( non_none ) != 1:
			return None # multiple non-None variants — need auto-generated union __bool__
		member = non_none[0]
		data_expr = ast.Attribute( value = expr_node, attr = data_attr.stem, ctx = ast.Load() )
		ast.copy_location( data_expr, ctx_node )
		payload_expr = ast.Attribute( value = data_expr, attr = f'v_{member.stem}', ctx = ast.Load() )
		ast.copy_location( payload_expr, ctx_node )
		# if the leaf type IS bool, the value itself is the boolean — no __bool__() call needed
		leaf_type = member.type
		if isinstance( leaf_type, Scalar ) and leaf_type.stem == 'bool':
			value_expr: ast.expr = payload_expr
		else:
			# only synthesize the .__bool__() call when the leaf type
			# actually defines one - real Python's own default object
			# truthiness is always-True unless __bool__/__len__ is
			# overridden, but this compiler doesn't auto-synthesize a
			# default __bool__ method the way Python effectively does, so
			# a class with no override would otherwise hit a hard "not
			# callable" resolution failure here just for participating in
			# a T|None truthiness check - matching Python's real default
			# directly (a bare Constant(True), no call at all) instead of
			# requiring every such class to hand-write a trivial `return
			# True` override.
			chain_lookup = getattr( leaf_type, 'chain_lookup', None )
			has_bool_method = chain_lookup is not None and chain_lookup( '__bool__' ) is not None
			if has_bool_method:
				value_expr = ast.Call(
					func = ast.Attribute( value = payload_expr, attr = '__bool__', ctx = ast.Load() ),
					args = [],
					keywords = [],
				)
			else:
				value_expr = ast.Constant( value = True )
			ast.copy_location( value_expr, ctx_node )
		# synthesize: tag_cmp and value_expr
		result = ast.BoolOp( op = ast.And(), values = [ tag_cmp, value_expr ] )
		ast.copy_location( result, ctx_node )
		return result

	def _tagged_union_payload_expr( self, expr_node: ast.expr, ctx_node: ast.AST ) -> ast.expr|None:
		''' the raw `expr.data.v_<T>` extraction alone (no truthiness test,
		no __bool__() call) - used by visit_BoolOp's value-coalescing
		rewrite for `x or y`'s TRUTHY branch, where `x` is proven non-None
		by the very fact that branch is being taken, so the branch's own
		VALUE should be the unwrapped T, not the still-Optional x (matching
		Python: `x or y` narrows the "x" case exactly the same way an `if
		x:` block would). Same type/shape restrictions as
		_rewrite_tagged_union_truthiness (single non-None member) -
		deliberately not factored to share code with it, since that method
		has its own additional `not x` recursion this one never needs. '''
		expr_type = self._type_of_expr( expr_node )
		if expr_type is None:
			return None
		spec = self.resolver._as_specialization( expr_type )
		base = spec.base if spec is not None else expr_type
		if not isinstance( base, TaggedUnion ):
			return None
		if spec is not None:
			members = self.resolver.monomorphizer.monomorphize_class( spec ).attributes
		else:
			self.resolver.ensure_resolved( base )
			for attr in base.attributes:
				self.resolver.ensure_resolved( attr )
			members = base.attributes
		none_type = self.discovery.get_none_type()
		none_member = next( ( attr for attr in members if attr.type is none_type ), None )
		if none_member is None:
			return None
		non_none = [ m for m in members if m.type is not none_type ]
		if len( non_none ) != 1:
			return None
		member = non_none[0]
		_tag_attr, data_attr, _payload_cls, _tags = self.resolver.union_storage.get( base )
		data_expr = ast.Attribute( value = expr_node, attr = data_attr.stem, ctx = ast.Load() )
		ast.copy_location( data_expr, ctx_node )
		payload_expr = ast.Attribute( value = data_expr, attr = f'v_{member.stem}', ctx = ast.Load() )
		ast.copy_location( payload_expr, ctx_node )
		return payload_expr

	def _try_fold_is_rc_if( self, node: ast.If ) -> list[ast.stmt]|None:
		''' rewrite 4: `if compiler.is_rc(T): A else: B` (T a generic class's
		own type param) folds to just A's or B's statements, the OTHER
		branch dropped entirely before it's ever type-checked - same
		compile-time-branch-elimination shape compile_time_transformer.py's
		own `if compiler.target.os == ...` folding already has, just keyed
		on a monomorphization's own concrete type-parameter binding instead
		of the active build target. This is what lets generic library code
		(list[T]'s own per-slot storage/access, an RCClass value being a
		pointer everywhere else in this compiler, but Ptr[T] itself staying
		single-indirection always - see PLAN_LIST_T.md's own grounding
		notes) write ONE shared method body with a plain `if`, rather than
		needing two separately-selected whole function bodies (compiler.
		target's own granularity) or scattering is_rc calls through every
		accessor.
		Only ever fires once T is genuinely CONCRETE (this pass also runs
		against the shared, abstract body first, where T is still its own
		unbound TypeVar - _try_resolve_namespace returns the TypeVar itself
		there, correctly declining to fold; the SAME node.test, revisited
		against the monomorphized copy's own deep-copied body once T is
		bound, folds correctly then - same "run once per specialization"
		discipline rewrite 3 already relies on, see resolve_function_body's
		own docstring). Never authoritative about failure, matching every
		other rewrite in this class: any doubt at all (not this exact
		shape, T not concrete, T not even a real type) returns None and
		leaves the if statement untouched for lowering.py's own unchanged,
		ordinary if-handling to report whatever's actually wrong '''
		test = node.test
		if not (
			isinstance( test, ast.Call ) and not test.keywords and len( test.args ) == 1
			and isinstance( test.func, ast.Attribute ) and test.func.attr == 'is_rc'
			and isinstance( test.func.value, ast.Name ) and test.func.value.id == 'compiler'
		):
			return None
		target_type = self._try_resolve_namespace( test.args[0] )
		if target_type is None or isinstance( target_type, TypeVar ):
			return None
		winning_body = node.body if self.resolver._is_RC( target_type ) else node.orelse
		folded: list[ast.stmt] = []
		for stmt in winning_body:
			result = self.visit( stmt )
			if isinstance( result, list ):
				folded.extend( result )
			elif result is not None:
				folded.append( result )
		return folded

	def _type_is_shape( self, test: ast.expr ) -> tuple[ast.expr,ast.expr,TaggedUnion,Variable,bool]|None:
		''' recognizes type(x) is T / type(x) is not T / instanceof(x, T)
		against a union-typed x, resolving all the way through to the real
		(union, member) pair - shared by _try_desugar_type_is_if (an if's
		own condition, Phase 4) and visit_While (a while loop's own
		condition, Phase 7). Returns (subject_expr, type_expr, base,
		member, is_not), or None on ANY doubt - never authoritative about
		failure, matching _try_fold_is_rc_if's own philosophy: the caller
		declines silently and falls through to visit_Compare's/
		visit_Call's own existing recognition and error-reporting for
		ordinary (non-narrowing) use. '''
		instanceof_args = self._instanceof_args( test )
		if instanceof_args is not None:
			subject_expr, type_expr = instanceof_args
			is_not = False
		else:
			if not ( isinstance( test, ast.Compare ) and len( test.ops ) == 1 and isinstance( test.ops[0], ( ast.Is, ast.IsNot )) ):
				return None
			left_subject = self._type_call_subject( test.left )
			right_subject = self._type_call_subject( test.comparators[0] )
			if left_subject is None and right_subject is None:
				return None
			if left_subject is not None and right_subject is not None:
				return None # type(x) is type(y) - let visit_Compare's own rewrite report this
			subject_expr = left_subject if left_subject is not None else right_subject
			type_expr = test.comparators[0] if left_subject is not None else test.left
			is_not = isinstance( test.ops[0], ast.IsNot )
		leaf_type = self._try_resolve_namespace( type_expr )
		if leaf_type is None:
			return None
		subj_type = self._type_of_expr( subject_expr )
		if subj_type is None:
			return None
		spec = self.resolver._as_specialization( subj_type ) # not a bare isinstance check - subj_type may already be eagerly-monomorphized, see visit_Match's own comment
		base = spec.base if spec is not None else subj_type
		if not isinstance( base, TaggedUnion ):
			return None
		members = self._resolved_union_members( subj_type, base )
		member = next( ( attr for attr in members if attr.type is leaf_type ), None )
		if member is None:
			return None
		return subject_expr, type_expr, base, member, is_not

	def _try_desugar_type_is_if( self, node: ast.If ) -> list[ast.stmt]|None:
		''' rewrite 5 (Phase 4): `if type(x) is T: A else: B` -> `match x:
		case T(x): A \n case _: B` (`if instanceof(x, T):` is the identical
		rewrite - recognized directly here via _instanceof_args, never
		routed through visit_Call's own Compare-rewriting detour, since
		node.test needs to stay in ORIGINAL form for THIS check to even
		recognize it: by the time generic_visit would otherwise reach it,
		visit_Call/visit_Compare would already have collapsed it into a
		plain tag-Cmp bool with no shape left to desugar). `type(x) is not
		T` swaps which body lands in the T-arm vs the wildcard arm rather
		than negating anything else. An elif chain (node.orelse holding a
		single nested ast.If) naturally desugars into a NESTED match purely
		as a side effect of the wildcard arm's own body being visited
		normally - if that nested if ALSO matches this shape, visiting it
		recurses into this same method again; if it doesn't, it's left as
		an ordinary nested if inside the wildcard arm, exactly matching
		TODO.txt's own `if x is int: ... elif x is str: ... else: ...`
		worked example structurally.

		Reuses visit_Match's own same-name narrowing entirely for free:
		the synthesized case pattern binds the SAME name as the match
		subject whenever x is itself a bare Name (case T(x), not case
		T(_)) - visit_Match's own original_subject_name recognition then
		narrows it exactly as it already does for a literal `match x: case
		T(x):` written directly by the user, no separate narrowing logic
		needed here at all. A non-Name subject (`if type(make()) is T:`)
		has nothing meaningful to bind (the original body couldn't have
		referenced anything from it either), so its arm captures nothing
		(case T():), preserving only the branch-taking behavior.

		Never authoritative about failure, matching _try_fold_is_rc_if's
		own philosophy exactly: any doubt at all (not this exact shape, T
		not a real type, T not a member of x's union, x not union-typed at
		all) returns None and leaves the if statement untouched - the
		SAME shape then falls through to generic_visit below, which
		reaches visit_Compare's/visit_Call's own already-existing
		recognition and error-reporting for `type(x) is T`/`instanceof(x,
		T)` used as an ordinary (non-narrowing) boolean condition, so
		nothing is ever silently dropped - just narrowing declining to
		apply, never validation being skipped. '''
		shape = self._type_is_shape( node.test )
		if shape is None:
			return None
		subject_expr, type_expr, _base, _member, is_not = shape
		match_body = node.orelse if is_not else node.body
		fallback_body = node.body if is_not else node.orelse
		inner_pattern = ast.MatchAs( pattern = None, name = subject_expr.id if isinstance( subject_expr, ast.Name ) else None )
		ast.copy_location( inner_pattern, node )
		class_pattern = ast.MatchClass( cls = type_expr, patterns = [ inner_pattern ], kwd_patterns = [], kwd_attrs = [] )
		ast.copy_location( class_pattern, node )
		wildcard_pattern = ast.MatchAs( pattern = None, name = None )
		ast.copy_location( wildcard_pattern, node )
		match_case = ast.match_case( pattern = class_pattern, guard = None, body = list( match_body ) if match_body else [ ast.Pass() ] )
		fallback_case = ast.match_case( pattern = wildcard_pattern, guard = None, body = list( fallback_body ) if fallback_body else [ ast.Pass() ] )
		match_node = ast.Match( subject = subject_expr, cases = [ match_case, fallback_case ] )
		ast.copy_location( match_node, node )
		result = self.visit( match_node )
		assert isinstance( result, list )
		return result

	def visit_If( self, node: ast.If ) -> ast.If|list[ast.stmt]:
		''' `if x is not None:`/`if x is None: ... else:` against a
		union-typed, bare-Name x - narrows x for whichever branch is
		actually "live" given the comparison (body for `is not`, orelse
		for `is`), for that branch's own duration. Restricted to the same
		exactly-one-non-None-member shape _rewrite_tagged_union_truthiness
		already restricts itself to (a 3+-member union's own "is not None"
		doesn't uniquely determine a single narrowed type) - no multi-
		member narrowing-marker support exists yet, matching visit_While's
		identical restriction on top of the equally general _type_is_shape.
		Post-if survival (narrowing surviving past the WHOLE if-statement
		when the un-narrowed branch terminates) needs no changes for the
		REAL, lowering-time narrowing - merge_if/_merge_narrowed_soft
		(cfg.py) are already fully generic over any branch's own end-of-
		branch _narrowed snapshot, already exercised today via the
		type(x) is T -> match desugar path. It DOES need an explicit update
		to this pass's OWN, separate self._narrowed (below, once
		other_terminates is known) - self._narrowed only drives this same
		pass's eager, best-effort inference (_type_of_expr, in turn used by
		_infer_generic_args for a bare generic call like len(x)), and
		unlike cfg.py's narrowing it does NOT automatically survive past a
		terminating sibling branch merely because cfg.py's does; the two
		are entirely separate trackers over separate representations.

		Manually walks node.body/node.orelse itself (not left to
		generic_visit's own field-list traversal) once a narrowing target
		is found - same reasoning visit_While/visit_Match's own manual
		per-statement loops document: a synthesized narrow-marker Assign
		must never be re-visited through the ordinary visit_Assign path. '''
		folded = self._try_fold_is_rc_if( node )
		if folded is not None:
			return folded
		desugared = self._try_desugar_type_is_if( node )
		if desugared is not None:
			return desugared
		# computed from the ORIGINAL, not-yet-rewritten test - visit_Compare's
		# own is-not-None tag rewrite (triggered below, via self.visit on the
		# test) would otherwise already have destroyed this shape by the time
		# it's looked for
		none_shape = self._is_none_narrowing_shape( node.test )
		if none_shape is None:
			# not an `is None`/`is not None` comparison - try the bare
			# truthiness shape instead (`if x:`/`if not x:`), see its own
			# docstring for why only its TRUTHY branch narrows
			none_shape = self._bare_truthiness_narrowing_shape( node.test )
		subject_name: str|None = None
		narrow_member: Variable|None = None
		is_not = False
		narrow_attr_base: str|None = None
		narrow_attr_name: str|None = None
		if none_shape is not None and isinstance( none_shape[0], ( ast.Name, ast.Attribute )):
			subject_expr, _base, members, none_member, shape_is_not = none_shape
			non_none = [ m for m in members if m is not none_member ]
			if len( non_none ) == 1:
				key = self._narrow_subject_key( subject_expr )
				if key is not None:
					subject_name, narrow_attr_base, narrow_attr_name = key
					narrow_member = non_none[0]
				is_not = shape_is_not
		# rewrite test BEFORE recursing into it, so the new BoolOp children
		# (Name references, Compare, Call) are visited normally (unchanged
		# from before this method's own narrowing support)
		rewritten = self._rewrite_tagged_union_truthiness( node.test, node )
		if rewritten is not None:
			node.test = rewritten
		elif isinstance( node.test, ast.BoolOp ):
			# a bare `if x or y:`/`if x and y:` (the rewrite above only
			# fires for the WHOLE test being a single T|None subject, not
			# a BoolOp of several) still needs to reach visit_BoolOp in
			# its plain bool-forcing mode, not the value-coalescing one -
			# see visit_BoolOp's own is_condition_context comment
			node.test.is_condition_context = True
		node.test = self.visit( node.test )

		def _visit_stmts( stmts: list[ast.stmt] ) -> list[ast.stmt]:
			result: list[ast.stmt] = []
			for stmt in stmts:
				visited = self.visit( stmt )
				if isinstance( visited, list ):
					result.extend( visited )
				elif visited is not None:
					result.append( visited )
			return result

		if narrow_member is None or subject_name is None:
			node.body = _visit_stmts( node.body )
			node.orelse = _visit_stmts( node.orelse )
			return node

		narrowed_body = node.body if is_not else node.orelse
		other_body = node.orelse if is_not else node.body
		case_entry_narrowed = dict( self._narrowed )
		self._narrowed[subject_name] = [ narrow_member.type ]
		try:
			narrowed_visited = _visit_stmts( narrowed_body )
		finally:
			self._narrowed = case_entry_narrowed
		narrowed_visited = [
			self._build_narrow_marker( subject_name, narrow_member, node, attr_base = narrow_attr_base, attr_name = narrow_attr_name ),
			*narrowed_visited,
		]
		other_visited = _visit_stmts( other_body )
		# the OTHER branch has no comparison to narrow it from - but if ITS
		# OWN code reassigns subject_name to exactly the narrowed member's
		# type (the "if x is None: x = Owned(...)" idiom - self.locals
		# tracks this via visit_Assign's own bookkeeping above), it ends up
		# narrowed too, just via a fresh value instead of a proven
		# comparison. Without this, cfg.py's own _merge_narrowed_soft sees
		# the fact on only ONE branch (the comparison-proven one) and drops
		# it entirely, even though both branches provably agree by the join
		# point. Skipped when the branch terminates (return/break/continue/
		# raise as its own last statement) - nothing past it reaches the
		# join, so there's nothing for this marker to narrow, and appending
		# one after a terminator would corrupt cfg.py's own terminates
		# detection (which keys off the branch's LAST statement).
		other_terminates = bool( other_body ) and isinstance( other_body[-1], ( ast.Return, ast.Break, ast.Continue, ast.Raise ))
		if not other_terminates and self.locals.get( subject_name ) is narrow_member.type:
			other_visited = [ *other_visited, self._build_narrow_marker( subject_name, narrow_member, node ) ]
		if other_terminates:
			# the un-narrowed branch never reaches the join - every path that
			# DOES (whatever follows this if-statement in the same enclosing
			# body) provably has subject_name narrowed, same as cfg.py's own
			# real narrowing already concludes. Persist that into THIS pass's
			# self._narrowed too (deliberately not restored to case_entry_
			# narrowed here, unlike narrowed_body's own try/finally above) so
			# a sibling statement visited after this method returns - e.g. a
			# bare generic call's own eager inference (_infer_generic_args ->
			# _type_of_expr) - sees the narrowed type instead of the stale,
			# still-unioned declared type.
			self._narrowed[subject_name] = [ narrow_member.type ]
		if is_not:
			node.body, node.orelse = narrowed_visited, other_visited
		else:
			node.orelse, node.body = narrowed_visited, other_visited
		return node

	def visit_While( self, node: ast.While ) -> ast.While:
		''' Phase 7: `while type(x) is T:`/`while type(x) is not T:`/
		`while instanceof(x, T):` against a union-typed subject x - a bare
		Name, or a single-level plain field (`self.field`/`y.field`, via
		the shared _narrow_subject_key - see its own docstring for exactly
		what qualifies) - narrows x for the loop BODY's own duration (the
		matched member for
		`is`, or - 2-member union only - the union's OTHER member for `is
		not`), and, separately, narrows x for CODE AFTER the loop once it
		exits (the loop's own condition is checked at least once even for
		a zero-iteration loop, so this holds regardless of how many times
		the body actually ran - see steady-dancing-haven.md's own "Context"
		section on why this needs no special "runs at least once"
		reasoning, unlike a body-always-does-X kind of claim would).
		`is`'s own exit narrowing needs a 2-member union (same "not T
		uniquely determines the other member" reasoning as the if/match
		wildcard case); `is not`'s own exit narrowing works for ANY union
		size - `not(x is not T)` means `x is T` directly, no
		disambiguation needed at all.

		Manually walks node.body itself (not left to generic_visit's own
		field-list traversal) - a synthesized narrow-marker Assign
		(is_narrowing_bind=True) prepended to node.body must NEVER be
		re-visited through the ordinary visit_Assign path (which would
		immediately clobber it - see _build_narrow_marker's own callers
		elsewhere, none of which are ever visited either), the exact same
		reasoning visit_Match's own manual per-case body loop already has. '''
		shape = self._type_is_shape( node.test )
		subject_name: str|None = None
		body_member: Variable|None = None
		exit_member: Variable|None = None
		narrow_attr_base: str|None = None
		narrow_attr_name: str|None = None
		if shape is not None:
			subject_expr, _type_expr, base, member, is_not = shape
			key = self._narrow_subject_key( subject_expr )
			if key is not None:
				subject_name, narrow_attr_base, narrow_attr_name = key
		if subject_name is not None:
			subj_type = self._type_of_expr( subject_expr )
			members = self._resolved_union_members( subj_type, base )
			others = [ m for m in members if m is not member ]
			other = others[0] if len( others ) == 1 else None
			if is_not:
				body_member = other
				exit_member = member # not(x is not T) -> x is T, any union size
			else:
				body_member = member
				exit_member = other # not(x is T) -> x is the sole OTHER member, 2-member unions only
			tag_attr, _data_attr, _payload_cls, tags = self.resolver.union_storage.get( base )
			tag_expr = ast.Attribute( value = subject_expr, attr = tag_attr.stem, ctx = ast.Load() )
			ast.copy_location( tag_expr, node )
			op = ast.NotEq() if is_not else ast.Eq()
			new_test = ast.Compare( left = tag_expr, ops = [ op ], comparators = [ ast.Constant( value = tags[member.stem] ) ] )
			ast.copy_location( new_test, node )
			node.test = new_test
		else:
			rewritten = self._rewrite_tagged_union_truthiness( node.test, node )
			if rewritten is not None:
				node.test = rewritten
			else:
				node.test = self.generic_visit_expr( node.test )
		case_entry_narrowed = dict( self._narrowed )
		if body_member is not None and subject_name is not None:
			self._narrowed[subject_name] = [ body_member.type ]
		try:
			new_body: list[ast.stmt] = []
			for stmt in node.body:
				visited = self.visit( stmt )
				if isinstance( visited, list ):
					new_body.extend( visited )
				elif visited is not None:
					new_body.append( visited )
			node.body = new_body
		finally:
			self._narrowed = case_entry_narrowed
		if body_member is not None and subject_name is not None:
			node.body = [
				self._build_narrow_marker( subject_name, body_member, node, attr_base = narrow_attr_base, attr_name = narrow_attr_name ),
				*node.body,
			]
		if node.orelse:
			# while/else isn't supported (lowering.py's _stmt_While fails
			# it outright) - still flattened correctly here (mirroring
			# node.body's own loop above) so a program using it fails with
			# THAT clear error at lowering time, not a confusing crash here
			new_orelse: list[ast.stmt] = []
			for stmt in node.orelse:
				visited = self.visit( stmt )
				if isinstance( visited, list ):
					new_orelse.extend( visited )
				elif visited is not None:
					new_orelse.append( visited )
			node.orelse = new_orelse
		if exit_member is not None and subject_name is not None:
			node.exit_narrows_name = subject_name
			node.exit_narrows_member_stem = exit_member.stem
			if narrow_attr_base is not None:
				node.exit_narrows_attr_base = narrow_attr_base
				node.exit_narrows_attr_name = narrow_attr_name
			self._narrowed[subject_name] = [ exit_member.type ]
		return node

	def visit_For( self, node: ast.For ) -> ast.For:
		''' Phase 8: a for-loop's own body can narrow a name via a nested
		if/match that terminates with `break` (Phase 6's own post-if/match
		merge) - unlike visit_Match/visit_While (which already bracket
		their own body-visit with a save/restore of self._narrowed), a
		bare for-loop had no such bracket at all before this, so that
		narrowing would LEAK past the loop's own body at this AST-
		rewriting-pass level once Phases 5-6 made narrowing survival a
		real thing - a for-loop might never run its body at all, or might
		run to completion without ever taking that break, so nothing
		proven only inside is safe to assume once back outside.

		cfg.py's own merge_loop_exits (Phase 8) is the REAL, authoritative
		lowering-time reconciliation for what narrowing survives a for-
		loop's break(s) - this pass makes NO attempt to mirror that here
		(a deliberate, bounded scope cut, same posture as visit_While's own
		body-only, not break-based, narrowing at this layer): it only needs
		to stop leaking STALE state, not to also propagate the real
		post-loop fact forward (a subsequent check after the loop just
		sees the name's ordinary declared type, safe, just not maximally
		precise - identical tradeoff to _type_of_expr's own narrowed
		lookup falling back for a multi-element set). '''
		case_entry_narrowed = dict( self._narrowed )
		try:
			self.generic_visit( node )
		finally:
			self._narrowed = case_entry_narrowed
		return node

	def visit_BoolOp( self, node: ast.BoolOp ) -> ast.expr:
		# is_condition_context: set by visit_If/visit_While/visit_IfExp/
		# visit_Assert on their OWN node.test right before dispatching
		# into it (generic_visit or self.visit both eventually reach
		# THIS method for a top-level BoolOp test) - those callers need a
		# guaranteed bool result (Python's `if x or y:` only cares about
		# truthiness, never which operand "won"), so they opt out of the
		# value-coalescing rewrite below entirely, always getting the
		# plain bool-forcing behavior instead - confirmed as a real
		# regression via a pre-existing test (`if x or y:` against two
		# TaggedUnion operands) that this rewrite silently broke before
		# this flag existed: it turned the condition into a ternary
		# PRODUCING one of the two operands, instead of combining both
		# operands' own truthiness into a single bool.
		#
		# value-coalescing: real Python and/or semantics (the actual
		# OPERAND survives, not a bool) - restricted to exactly 2
		# operands, left operand a bare Name (safe to reference twice -
		# once for its own truthiness, once as the resulting value -
		# without re-evaluating a call/side-effecting expression a second
		# time), whose type is a TaggedUnion with a None member (the
		# "fill in a default when None/falsy" idiom, e.g. `tz or
		# localtz()`). Desugars into an ordinary ternary, reusing
		# visit_IfExp/_expr_IfExp's own already-correct rewrite/RC
		# handling entirely rather than reimplementing it here: `x or y`
		# is exactly `x if <truthy(x)> else y`; `x and y` is exactly `y
		# if <truthy(x)> else x`. Anything outside this shape (more than
		# 2 operands, a non-Name left operand, or a left operand that's
		# plain bool/not a TaggedUnion at all) falls through unchanged to
		# the existing bool-only path below (e.g. match's own nested-
		# pattern tests, already bool on both sides).
		if not getattr( node, 'is_condition_context', False ) and len( node.values ) == 2 and isinstance( node.values[0], ast.Name ):
			left, right = node.values
			truthy = self._rewrite_tagged_union_truthiness( left, node )
			if truthy is not None:
				is_and = isinstance( node.op, ast.And )
				if is_and:
					# x and y: truthy -> y (as-is); falsy -> x, UNCHANGED
					# (matches real Python - a falsy-but-non-None x is still
					# possible, so the falsy branch can't be unwrapped here;
					# the ternary's own two branches naturally end up typed
					# y's-type | x's-declared-type, same as Python's real
					# `and` would produce)
					body, orelse = right, left
				else:
					# x or y: truthy -> x, but UNWRAPPED to its non-None
					# payload (this branch proves x isn't None, exactly like
					# an `if x:` block would - matches _rewrite_tagged_
					# union_truthiness's own narrowing for that shape);
					# falsy -> y, as-is
					unwrapped = self._tagged_union_payload_expr( left, node )
					body, orelse = ( unwrapped if unwrapped is not None else left ), right
				if_exp = ast.IfExp( test = truthy, body = body, orelse = orelse )
				ast.copy_location( if_exp, node )
				return self.visit_IfExp( if_exp )
		# each operand of `and`/`or` is a boolean context — rewrite
		# T|None operands BEFORE generic_visit recurses into the old nodes
		for i, value in enumerate( node.values ):
			rewritten = self._rewrite_tagged_union_truthiness( value, node )
			if rewritten is not None:
				node.values[i] = rewritten
			elif isinstance( value, ast.BoolOp ):
				# a nested boolop operand (`(a or b) or c`) is ALSO
				# purely a boolean context here, once this outer BoolOp
				# has reached this plain bool-forcing path itself - see
				# visit_BoolOp's own is_condition_context comment
				value.is_condition_context = True
		self.generic_visit( node )
		return node

	def visit_IfExp( self, node: ast.IfExp ) -> ast.IfExp:
		# ternary `x if cond else y` — cond is a boolean context
		rewritten = self._rewrite_tagged_union_truthiness( node.test, node )
		if rewritten is not None:
			node.test = rewritten
		elif isinstance( node.test, ast.BoolOp ):
			# see visit_BoolOp's own is_condition_context comment - `z if
			# (x or y) else w`'s own `(x or y)` must stay plain-bool, not
			# get value-coalesced
			node.test.is_condition_context = True
		self.generic_visit( node )
		return node

	def visit_Assert( self, node: ast.Assert ) -> list[ast.stmt]:
		if isinstance( node.test, ast.BoolOp ):
			# see visit_BoolOp's own is_condition_context comment -
			# `assert x or y, msg` must stay plain-bool, not get value-
			# coalesced
			node.test.is_condition_context = True
		self.generic_visit( node )
		if node.msg is None:
			self.discovery.fail(
				'assert requires a message (e.g. assert cond, "reason"): '
				+ ast.unparse( node ),
				node,
			)
		# force-import sys and resolve its _assert function — the rewritten
		# AST uses a bare `sys._assert(...)` reference, but that name won't
		# resolve in the current scope; tag the Call node with resolved_callee
		# so lowering uses the Function object directly (same pattern as
		# visit_Call's generic-call resolution), never touching `sys` at all
		try:
			sys_module = self.discovery.import_name( 'sys' )
		except FileNotFoundError:
			self.discovery.fail(
				'assert requires the sys module, but it could not be found '
				'(try calling sys._assert() directly instead): '
				+ ast.unparse( node ),
				node,
			)
		_assert_fn = sys_module.get_local( '_assert' )
		if isinstance( _assert_fn, Function ):
			self.resolver.schedule( _assert_fn )
		call = ast.Call(
			func = ast.Attribute(
				value = ast.Name( id = 'sys', ctx = ast.Load() ),
				attr = '_assert',
				ctx = ast.Load(),
			),
			args = [ node.test, node.msg ],
			keywords = [],
		)
		ast.copy_location( call, node )
		call.resolved_callee = _assert_fn
		stmt = ast.Expr( value = call )
		ast.copy_location( stmt, node )
		# gated on compiler.target.debug, stripped entirely in a release
		# build - same mechanism sys.alloc's own poison-fill already uses.
		# The message argument stays mandatory regardless (checked above),
		# only whether the check RUNS is target-dependent. This node is
		# synthesized AFTER compile_time_transformer.transform_function_body
		# already ran over the rest of this function body (_make_function_
		# resolver's own body() calls it before ever walking statements, see
		# its own comment) - it will never be visited by that pass, so the
		# fold has to be applied here, by hand, right now instead.
		guard = ast.If(
			test = ast.Attribute(
				value = ast.Attribute(
					value = ast.Name( id = 'compiler', ctx = ast.Load() ),
					attr = 'target',
					ctx = ast.Load(),
				),
				attr = 'debug',
				ctx = ast.Load(),
			),
			body = [ stmt ],
			orelse = [],
		)
		ast.copy_location( guard, node )
		folded = compile_time_transformer.transform_stmt_list(
			[ guard ], self.discovery.active_target, self.discovery._detect_cc,
		)
		return folded

	# --- rewrite 2: match statements ---

	def _stmt_diverges( self, stmt: ast.stmt ) -> bool:
		''' true if `stmt` never falls through - either structurally (return/
		break/continue) or because it's a bare call to a function declared
		-> NoReturn (sys.panic, most commonly). The type_resolver.py-level
		analogue of lowering.py's own _stmt_diverges (used by _stmt_If's
		true_terminates/false_terminates) - this one backs visit_Match's own
		per-case `terminates` computation (_merge_case_narrowing), for the
		identical reason: a `case ...: sys.panic(...)` arm should be treated
		as never reaching the match's own join point, the same as an
		explicit return/break/continue arm, or narrowing established inside
		it is wrongly dropped from self._narrowed instead of surviving past
		the match. Resolved via _resolve_callee_target - a pure lookup, no
		scheduling side effects beyond _resolve_callable's ordinary
		signature resolution - since this only needs the callee's declared
		return type, not a real lowered call. A receiver call (x.method())
		or anything _resolve_callee_target can't resolve without a receiver
		just isn't recognized here, same scope cut as lowering.py's own
		version - EXCEPT unlike lowering.py's call site (at LOWERING time),
		this one runs during type_resolver.py's OWN pass, where a receiver
		rooted in a local (self, or any other parameter/local) genuinely
		isn't resolvable via discovery's scope-stack-based find_name at all
		(that lookup is module/class-level names only - locals live in
		this resolver's own separate self.locals dict, never registered
		into discovery's scope stack) - _try_resolve_namespace's own
		ast.Name branch calls the RAISING find_name, not find_name_or_none,
		so a receiver like self.foo() THROWS instead of returning None
		here. Catching the CompileError is NOT enough to make this safe:
		discovery.fail() (errors.py's ErrorCollector.fail) permanently
		records the message in discovery.errors.errors BEFORE raising, by
		design ("the failure is already recorded... callers that catch it
		need no data from it") - so even a caught-and-ignored exception
		here would still poison the overall compile into reporting failure,
		confirmed via a real repro (case Result.Ok(v): self.touch() left
		'name \'self\' is not defined' in the error list even after
		wrapping the call in try/except CompileError).

		Originally pre-checked the call's own ultimate base name against
		self.locals (a name tracked there is DEFINITELY a local, never a
		resolvable namespace path) - but self.locals is NOT a complete
		record of every local: a match-pattern binding (case Result.Ok(w):)
		produces a plain ast.Assign via _match_pattern/_match_union_member
		that's spliced directly into the case's own output body, never
		routed through self.visit()/visit_Assign, so it never updates
		self.locals at all - confirmed via a real regression (case
		Result.Ok(w): ... w.close() as the arm's last statement crashed the
		SAME way self.touch() originally did, self.locals notwithstanding).
		Checks discovery.find_name_or_none directly instead - the SAME
		safe, non-raising lookup _try_resolve_namespace's own ast.Name
		branch SHOULD be using itself (see that branch's own comment) -
		since that authoritatively answers "is this name resolvable as a
		namespace path at all" without needing this method to separately
		enumerate every way a name could turn out to be local. '''
		if isinstance( stmt, ( ast.Return, ast.Break, ast.Continue )):
			return True
		if isinstance( stmt, ast.If ):
			# the type_resolver.py-level analogue of lowering.py's own
			# _stmt_diverges fix - an if/else both of whose branches
			# diverge is itself terminating, even though it isn't literally
			# a Return/Break/Continue. This is the exact shape a nested
			# `match` statement desugars to (visit_Match below, chained
			# ast.If via tail.orelse) whenever every case of the NESTED
			# match returns - without this, a case whose own last statement
			# is such a nested match wrongly reports terminates=False,
			# feeding a live/non-terminating candidate into
			# _merge_case_narrowing that should have been excluded entirely.
			# Recursing through _stmt_diverges itself handles arbitrarily
			# long desugared case chains. No orelse means the false path
			# always falls through, so it can never qualify.
			return (
				bool( stmt.body ) and self._stmt_diverges( stmt.body[-1] )
				and bool( stmt.orelse ) and self._stmt_diverges( stmt.orelse[-1] )
			)
		if not ( isinstance( stmt, ast.Expr ) and isinstance( stmt.value, ast.Call )):
			return False
		root = stmt.value.func
		while isinstance( root, ( ast.Attribute, ast.Subscript )):
			root = root.value
		if not isinstance( root, ast.Name ) or self.discovery.find_name_or_none( root.id ) is None:
			return False
		target = self.resolver._resolve_callee_target( stmt.value.func )
		fn = target.base if isinstance( target, Specialization ) else target
		if not isinstance( fn, Function ):
			return False
		return isinstance( fn.return_type, Scalar ) and fn.return_type.stem == 'NoReturn'

	def _try_fold_match_type( self, node: ast.Match ) -> list[ast.stmt]|None:
		''' rewrite: `match type(<Name>): case ConcreteClass(binding): ...
		case _: ...` - compile-time ARM SELECTION for a bare-Name subject
		whose own static type is concrete (most usefully, a generic
		method's own type-parameter-typed parameter/local, once
		monomorphization has bound it to a concrete type) - same "declines
		on the still-abstract body, folds once T is concrete" discipline
		as _try_fold_is_rc_if (this pass runs once against the shared,
		abstract body, where a generic T is still its own unbound TypeVar
		and this correctly declines, and again against the monomorphized
		copy's own deep-copied body once T is bound - see
		resolve_function_body's own docstring), just for `match` instead
		of `if`. This is a DIFFERENT rewrite from _rewrite_type_is_
		comparison/visit_Match's own ordinary handling below: those require
		the subject's static type to already be a TaggedUnion (a real,
		tagged runtime value); this one is for the OPPOSITE case, a
		non-union concrete type, where there is nothing to check at
		runtime at all - the whole match collapses to exactly one arm's
		own statements at compile time, no `ast.If`/Cmp left behind.

		Two DIFFERENT kinds of "not yet" have to be told apart here, unlike
		_try_fold_is_rc_if (whose own decline just lets visit_If's ordinary
		machinery harmlessly re-visit compiler.is_rc(T) as a plain,
		unrecognized Call - a no-op, never an error, since nothing else in
		this class attaches any meaning to is_rc outside the fold):
		visit_Match's own ordinary (TaggedUnion-only) handling below is
		NOT that forgiving - the moment it can't determine the subject's
		type, or determines it isn't a union, it calls discovery.fail()
		OUTRIGHT (a real, PERMANENT error), because rewrite 2 (ordinary
		match desugaring) is documented as substitution-INDEPENDENT and
		was never meant to be retried on a second pass. So when the
		subject genuinely IS `type(<Name>)` and Name's type is still an
		unbound TypeVar - the one case that's certain to resolve cleanly
		once monomorphization binds it - this returns the node COMPLETELY
		UNTOUCHED (`[node]`, not None) rather than falling through, so
		none of visit_Match's ordinary machinery ever sees it on this
		pass at all. That's safe for the exact same reason _try_fold_is_
		rc_if's own second pass is (see resolve_function_body's own
		docstring): a monomorphized copy's body is independently deep-
		copied, so the held, unvisited node here is simply revisited fresh
		- and this time foldable - against THAT copy.
		Every OTHER kind of doubt (not a `type(Name)` subject at all,
		Name's type genuinely undeterminable for some unrelated reason,
		Name's type IS a TaggedUnion, or any single arm shaped other than
		a plain single-capture class pattern or a bare `case _:`) declines
		with a plain None instead - these reproduce exactly the SAME
		"cannot determine the match subject's type"/"is not a union type"
		errors visit_Match's ordinary handling already gives `match
		type(...)` today (never a supported shape before this rewrite
		either), not a new regression. Once the shape is confirmed to
		apply on a genuinely concrete, non-union type, though, this IS
		authoritative - a concrete type with no covering arm is a real,
		reported error (see the no-wildcard branch below), not a silent
		no-op. '''
		subject_expr = self._type_call_subject( node.subject )
		if subject_expr is None or not isinstance( subject_expr, ast.Name ):
			return None
		subj_type = self._type_of_expr( subject_expr )
		if isinstance( subj_type, TypeVar ):
			return [ node ] # still abstract - hold unvisited for the monomorphized copy's own second pass, see docstring
		if subj_type is None:
			return None # genuinely undeterminable for some other reason - not this rewrite's doubt to resolve
		spec = self.resolver._as_specialization( subj_type )
		base = spec.base if spec is not None else subj_type
		if isinstance( base, TaggedUnion ):
			return None # `match type(x):` for a real union isn't a shape anything supports, before or after this rewrite - decline to the same pre-existing error
		winning_stmts: list[ast.stmt]|None = None
		winning_bind: str|None = None
		for case in node.cases:
			pattern = case.pattern
			if isinstance( pattern, ast.MatchAs ) and pattern.pattern is None and pattern.name is None:
				# a true, UNNAMED wildcard (case _:) - always matches. A
				# NAMED bare pattern (case leftover:) is deliberately NOT
				# treated as a wildcard here: it would mean binding the
				# whole `type(other)` VALUE, and this compiler has no
				# runtime type-object value to bind it to (see
				# type_resolver.py's own "no runtime reflection/RTTI"
				# comment, ~line 4090) - decline the whole fold instead of
				# guessing what that should mean
				winning_stmts = case.body
				break
			if not (
				isinstance( pattern, ast.MatchClass ) and not pattern.kwd_patterns and not pattern.kwd_attrs
				and len( pattern.patterns ) == 1 and isinstance( pattern.patterns[0], ast.MatchAs ) and pattern.patterns[0].pattern is None
			):
				return None # not a plain single-capture class pattern (or a named wildcard, handled above) - decline entirely, don't partially fold
			leaf_type = self._try_resolve_callable_namespace( pattern.cls )
			if leaf_type is None:
				return None
			if self.resolver._same_type( leaf_type, subj_type ):
				winning_stmts = case.body
				winning_bind = pattern.patterns[0].name
				break
		if winning_stmts is None:
			self.discovery.fail( f'match type(...): no arm covers {getattr( subj_type, "qualname", subj_type )} for this instantiation: {ast.unparse(node)}', node )
			return []
		folded: list[ast.stmt] = []
		if winning_bind is not None and winning_bind != subject_expr.id:
			# subject already IS exactly the matched concrete type - no
			# payload to extract (unlike a real TaggedUnion match's own
			# .data.v_<member> unwrap), just a plain rebind. Skipped
			# entirely when the capture reuses the subject's OWN name
			# (`case str(other):` against `match type(other):`) - not just
			# an optimization: synthesizing `other = other` for an RC-
			# tracked type would self-alias-assign, and nothing else in
			# this rewrite needs that statement to exist at all when the
			# name already denotes the right value with the right type
			rebind = ast.Assign( targets = [ ast.Name( id = winning_bind, ctx = ast.Store() ) ], value = subject_expr )
			ast.copy_location( rebind, node )
			folded.append( rebind )
		if winning_bind is not None:
			self.locals[winning_bind] = subj_type
		for stmt in winning_stmts:
			result = self.visit( stmt )
			if isinstance( result, list ):
				folded.extend( result )
			elif result is not None:
				folded.append( result )
		return folded

	def visit_Match( self, node: ast.Match ) -> list[ast.stmt]:
		folded = self._try_fold_match_type( node )
		if folded is not None:
			return folded
		# resolve_function_body's own docstring classifies match desugaring
		# (rewrite 2) as substitution-INDEPENDENT - true for pattern
		# resolution itself (an Owner.Member reference is resolved by NAME,
		# never by the subject's type), but NOT for this method's own later
		# exhaustiveness/flattening pre-pass just below, which DOES need the
		# subject's real type (_type_of_expr(node.subject)) to know whether
		# every case together covers a union's own members. When the
		# CURRENT function is still generic (an unbound type param, e.g.
		# `def fill_from[T](self, transport: T)` matching on `transport.
		# recv(...)`'s own return type) that type genuinely can't be known
		# yet - _type_of_expr correctly comes back None - but rewrite 1/2
		# (compiler.py's _lower, Specialization+Function branch) still runs
		# this pass exactly once against the SHARED, abstract base Function,
		# unconditionally, well before any concrete specialization exists.
		# Proceeding anyway would permanently bake a wrongly-non-exhaustive
		# if/elif (no real trailing else, since last_guaranteed can only
		# ever be True) into that SHARED node - and since Monomorphizer.
		# monomorphized_function deep-copies THAT node per specialization,
		# whichever specialization's own copy happens to be taken AFTER this
		# pass runs (a genuine compile-order race - a DIFFERENT specialization
		# whose own copy was taken EARLIER, before this mutation, still gets
		# a pristine, correctly-desugared-later copy) inherits the wrong
		# structure permanently, with no case left for rewrite 3's own later,
		# per-specialization pass (T now concretely bound) to ever revisit -
		# a real, confirmed -Wreturn-type/C4715 (non-void function falls off
		# the end), not a hypothetical (lib/http/client.py's own _GrowableBuffer.
		# fill_from[socket.Socket], confirmed via a real before/after
		# generated-C diff and direct monomorphization tracing, not guessed).
		# Same "on any doubt, defer entirely" discipline visit_Call's own
		# generic-call resolution already uses for the identical reason (see
		# this class's own docstring, rewrite 3's paragraph) - leaving node
		# completely untouched here means the SHARED base's own body still
		# holds a genuine, un-mutated ast.Match, so EVERY specialization's
		# own deep copy (regardless of which race it's on) gets a fresh,
		# correct shot at this exact method, once via rewrite 3, with its
		# own T concretely bound.
		if (
			self.fn is not None and ( self.fn.type_params or getattr( self.fn.cls, 'type_params', None ))
			and self._type_of_expr( node.subject ) is None
		):
			return [ node ]
		unique = self._label_id
		self._label_id += 1
		subj_name = f'__match_subj_{unique}'
		subj_assign = ast.Assign( targets = [ ast.Name( id = subj_name, ctx = ast.Store() ) ], value = self.visit( node.subject ))
		ast.copy_location( subj_assign, node )
		# two attributes lowering.py's own _stmt_Assign reads (getattr(...,
		# default), same bridging technique visit_Call's own resolved_callee
		# already uses) - see cfg.py's unchecked-Result tracking. Always set
		# is_match_subject: __match_subj_N is compiler-internal scaffolding
		# that the if-chain below only ever reads via raw tag Compares
		# (never is_ok()/is_err()/etc), so if it were tracked like an
		# ordinary Result-typed local, nothing would ever clear it and every
		# match over a Result would falsely report an unchecked result.
		# match_clears_name is set only when the subject is a bare Name:
		# ordinary aliasing assignment deliberately does NOT propagate a
		# clear back to its source (see cfg.py's "Independent tracking"),
		# but `match r:` genuinely IS the inspection of r itself, so the
		# original name needs an explicit clear here that a plain alias
		# assign wouldn't give it for free
		subj_assign.is_match_subject = True
		if isinstance( node.subject, ast.Name ):
			subj_assign.match_clears_name = node.subject.id
		subj_ref = ast.Name( id = subj_name, ctx = ast.Load() )
		ast.copy_location( subj_ref, node )
		# __match_subj_N is built directly here, never dispatched through
		# self.visit()/visit_Assign - so unlike an ordinary Assign, nothing
		# populates self.locals[subj_name] for free. _match_pattern's own
		# new `case None:`/`case str(c):` handling (unlike the pre-existing
		# `case Result.Ok(x):` handling, which reads the union off the
		# PATTERN's own text, never the subject) needs the subject's own
		# static type to know which union this leaf/None belongs to - same
		# type inference visit_Assign already gives an ordinary local for
		# free, just done explicitly here since this Assign bypasses that path
		self.locals[subj_name] = self._type_of_expr( node.subject )
		# only meaningful at this top level (never threaded into
		# _match_pattern's own recursive calls against an EXTRACTED
		# payload - see _match_pattern's own comment on why): lets a
		# top-level `case T(x):` that reuses the ORIGINAL subject's own
		# name (`match x: case T(x): ...`) be recognized as narrowing x
		# itself, rather than binding a same-named-but-distinct value
		original_subject_name = node.subject.id if isinstance( node.subject, ast.Name ) else None
		# separate from original_subject_name above (which stays Name-only
		# by design - it feeds _match_pattern's own `case T(x):` PATTERN-
		# BINDING reuse, only ever meaningful for a bare Name subject to
		# begin with): this is for the WILDCARD-deduces-the-other-member
		# narrowing further down, which - like visit_If/visit_While's own
		# narrowing - can target a single-level plain field too. See
		# _narrow_subject_key's own docstring for exactly what qualifies.
		wildcard_narrow_key = self._narrow_subject_key( node.subject )
		entry_narrowed = dict( self._narrowed )

		# Phase 5/6 exhaustiveness pre-pass (narrowing surviving PAST the
		# whole match, not just confined to one arm - see steady-dancing-
		# haven.md's own "Trap 1"): a case's own empty `orelse=[]` (built
		# below) is only semantically "there's really nothing else this
		# could be" when either (a) it's a literal wildcard (test=True
		# already), or (b) it's the LAST case and every case together
		# (this one plus all before it) provably covers every one of the
		# union's own members - otherwise it's a genuine "nothing matched"
		# fallthrough that's a real, competing (unnarrowed) path. Only the
		# LAST case's own orelse is ever actually empty (every earlier
		# case's orelse is immediately overwritten by the next case being
		# chained in), so only it needs this check. subj_type/base/members
		# are computed here (not left to each case's own _match_pattern
		# call) so this pre-pass and the main loop below agree on the
		# exact same resolved member objects (identity matters - see
		# _resolve_case_member's own comment on "owner is not base").
		subj_type = self.locals.get( subj_name )
		# _as_specialization, not a bare isinstance(subj_type, Specialization) -
		# subj_type can now be an EAGERLY-MONOMORPHIZED concrete union (e.g.
		# csv.reader()'s return type, once resolve_declared_types has run for
		# it) rather than a bare Specialization wrapper. Treating that
		# concrete union as `base` directly is wrong: _resolve_case_member
		# below matches each case pattern's Owner (`Result.Ok`) against the
		# ABSTRACT class's own member objects (textual patterns are always
		# resolved through the abstract, generic `Result`, never through a
		# concrete specialization) - `members` must come from that SAME
		# abstract base or every case fails to match its own pattern by
		# identity (confirmed via a real repro: a second, independently-
		# compiled call site sharing the same already-monomorphized callee
		# silently dropped one match arm's whole body - see
		# resolve_declared_types's own docstring for why the return type is
		# no longer reliably a bare Specialization here)
		spec = self.resolver._as_specialization( subj_type )
		base = spec.base if spec is not None else subj_type
		members = self._resolved_union_members( subj_type, base ) if isinstance( base, TaggedUnion ) else []
		last_is_wildcard = bool( node.cases ) and isinstance( node.cases[-1].pattern, ast.MatchAs ) and node.cases[-1].pattern.pattern is None
		last_guaranteed = last_is_wildcard
		wildcard_narrow_member: Variable|None = None
		if isinstance( base, TaggedUnion ) and node.cases and not last_is_wildcard:
			resolved = [ self._resolve_case_member( base, members, case.pattern ) for case in node.cases ]
			if resolved[-1] is not None and all( m is not None for m in resolved ):
				distinct_ids = { id( m ) for m in resolved }
				if len( distinct_ids ) == len( resolved ) == len( members ):
					last_guaranteed = True
		elif last_is_wildcard and isinstance( base, TaggedUnion ) and len( members ) == 2 and len( node.cases ) >= 2:
			# a bare, UNNAMED wildcard specifically (case _:, not a named
			# capture case y: - a named capture means "rebind the whole
			# union", not "prove the other member") as the LAST of exactly
			# the union's own 2 members, one already matched by some prior
			# case - the union's OTHER member is then exactly what the
			# wildcard arm must hold. Declines (stays None) for anything
			# else: a 3+-member union (single-member narrowing can't
			# express "one of several remaining possibilities" - the
			# abandoned UnionView design was for exactly that, not
			# rebuilt here), or when the prior cases don't cleanly resolve
			# to exactly one distinct member.
			wildcard_pattern = node.cases[-1].pattern
			if isinstance( wildcard_pattern, ast.MatchAs ) and wildcard_pattern.name is None:
				prior_resolved = [ self._resolve_case_member( base, members, case.pattern ) for case in node.cases[:-1] ]
				non_none = [ m for m in prior_resolved if m is not None ]
				if len( non_none ) == 1 and len( { id( m ) for m in non_none } ) == 1:
					others = [ m for m in members if m is not non_none[0] ]
					if len( others ) == 1:
						wildcard_narrow_member = others[0]

		chain: ast.If|None = None
		tail: ast.If|None = None
		singleton_body: list[ast.stmt]|None = None
		case_infos: list[tuple[bool,dict[str,list[Type]]]] = []
		for case_index, case in enumerate( node.cases ):
			if case.guard is not None:
				self.discovery.fail( f'match guards (case ... if ...) are not yet supported: {ast.unparse(case.pattern)}', node )
			test, binds = self._match_pattern( subj_ref, case.pattern, node, original_subject_name )
			is_last = case_index == len( node.cases ) - 1
			flatten_this_case = is_last and last_guaranteed
			if (
				flatten_this_case and wildcard_narrow_member is not None and wildcard_narrow_key is not None
				and isinstance( case.pattern, ast.MatchAs ) and case.pattern.pattern is None and case.pattern.name is None
			):
				# the wildcard's own _match_pattern call above returned
				# binds=[] (a true, unnamed wildcard never binds anything on
				# its own) - override with a real narrow-marker targeting the
				# DEDUCED other member, computed in the pre-pass above
				wc_name, wc_attr_base, wc_attr_name = wildcard_narrow_key
				binds = [ self._build_narrow_marker( wc_name, wildcard_narrow_member, node, attr_base = wc_attr_base, attr_name = wc_attr_name ) ]
			# a narrowing arm (see _match_union_member's own narrow_marker)
			# pushes name -> [its narrowed leaf type] into self._narrowed for
			# exactly the span of THIS case's own body - every union-shaped
			# rewrite below (visit_Compare's is-None check,
			# _rewrite_type_is_comparison, _rewrite_tagged_union_truthiness)
			# goes through _type_of_expr, which now consults self._narrowed
			# first, so a SECOND union-shaped check on the same
			# already-narrowed name inside this same arm sees the narrowed
			# leaf type instead of stale outer union type (the gap this
			# closes). Whole-dict snapshot/restore (not just the one key
			# THIS case's own narrow-marker touches) - matters once a
			# NESTED construct inside this case's own body can survive PAST
			# its own boundary (this phase's whole point): a single-key
			# restore would leak whatever that nested construct narrowed
			# into the NEXT sibling case
			case_entry_narrowed = dict( self._narrowed )
			if len( binds ) == 1 and getattr( binds[0], 'is_narrowing_bind', False ):
				self._narrowed[ binds[0].targets[0].id ] = [ binds[0].narrowed_type ]
			try:
				# self.visit(stmt) returns a bare ast.stmt for most statement
				# kinds, but a NESTED ast.Match (this method's own visit_Match,
				# called recursively) returns a list[ast.stmt] instead (its own
				# subj_assign + if-chain) - unlike ast.NodeTransformer's own
				# generic_visit(), a manual comprehension doesn't auto-flatten
				# that, so a naive `[self.visit(stmt) for stmt in case.body]`
				# would embed the nested match's own 2-element list as ONE
				# malformed body entry instead of two real statements - only
				# surfaced once something actually reaches lowering.py's
				# _lower_stmt, which has no _stmt_list handler ("unsupported
				# statement", unparsed because ast.unparse() happens to accept a
				# bare list of stmts too, which is what made this so confusing
				# to trace)
				body: list[ast.stmt] = []
				for stmt in case.body:
					visited = self.visit( stmt )
					if isinstance( visited, list ):
						body.extend( visited )
					elif visited is not None:
						body.append( visited )
				terminates = bool( case.body ) and self._stmt_diverges( case.body[-1] )
				case_infos.append( ( terminates, dict( self._narrowed )))
			finally:
				self._narrowed = case_entry_narrowed
			flattened_body = [ *binds, *body ]
			if flatten_this_case:
				# this case's own test is PROVABLY true whenever it's
				# reached (a literal wildcard, or the last of an
				# exhaustive set of explicit member-cases) - wrapping it
				# in its own `ast.If(test=True, ..., orelse=[])` would give
				# it a vacuous, ALWAYS-empty "false branch" that a
				# reconciliation pass has no way to tell apart from a
				# genuine "nothing matched" fallthrough, silently
				# discarding this arm's own narrowing before it ever
				# reaches the real, enclosing join point - splice its body
				# directly instead, exactly as if it were unconditional
				# (which, per the reasoning above, it provably is)
				if chain is None:
					singleton_body = flattened_body
				else:
					tail.orelse = flattened_body
				break
			arm = ast.If( test = test, body = flattened_body, orelse = [] )
			ast.copy_location( arm, node )
			if chain is None:
				chain = arm
			else:
				tail.orelse = [ arm ]
			tail = arm
		if not last_guaranteed:
			# the match isn't provably exhaustive - a real "nothing
			# matched" fallthrough exists and must participate in the
			# merge below as its own, genuine (unnarrowed) candidate path,
			# mirroring lowering.py's _stmt_If's own false_end = entry_
			# bindings default for a missing orelse - otherwise a
			# non-exhaustive match would wipe out narrowing that had
			# nothing to do with it
			case_infos.append( ( False, entry_narrowed ))
		self._narrowed = self._merge_case_narrowing( case_infos )
		if singleton_body is not None:
			return [ subj_assign, *singleton_body ]
		return [ subj_assign, chain ] if chain is not None else [ subj_assign ]

	def _merge_case_narrowing( self, case_infos: list[tuple[bool,dict[str,list[Type]]]] ) -> dict[str,list[Type]]:
		''' the N-ary, type_resolver.py-level analogue of cfg.py's own
		merge_if narrowed-reconciliation - survivors = every case that
		DOESN'T terminate (return/break/continue as its own last
		statement); if every case terminates, nothing reaches whatever
		follows the match at all (dead code past there - empty is the
		safe/correct answer). A name survives into the merged, POST-match
		state only if narrowed on EVERY surviving case - its value becomes
		the UNION (dedup by identity) of what each survivor narrowed it
		to, not just an identical-only intersection: if one arm proves x
		is int and another (also surviving) proves x is str, code reaching
		past the match genuinely could be either - "one of int|str" is
		real, actionable information (rules out every OTHER member, e.g.
		None), not something to discard just because the arms disagree
		about WHICH ONE. '''
		survivors = [ d for terminates, d in case_infos if not terminates ]
		if not survivors:
			return {}
		merged: dict[str,list[Type]] = dict( survivors[0] )
		for other in survivors[1:]:
			next_merged: dict[str,list[Type]] = {}
			for name, types in merged.items():
				if name not in other:
					continue
				combined = list( types )
				for t in other[name]:
					if not any( t is existing for existing in combined ):
						combined.append( t )
				next_merged[name] = combined
			merged = next_merged
		return merged

	def generic_visit_expr( self, node: ast.expr ) -> ast.expr:
		# generic_visit() itself returns the node (mutated in place, for an
		# expr) - named wrapper only so visit_Match's own use above reads
		# clearly as "visit this expression", matching visit_Compare/
		# visit_AnnAssign/visit_Assign's own self.generic_visit(node) calls
		self.generic_visit( node )
		return node

	def _match_pattern( self, subj_expr: ast.expr, pattern: ast.pattern, node: ast.AST, original_subject_name: str|None = None ) -> tuple[ast.expr,list[ast.stmt]]:
		# ported from lowering.py's Lowering._match_pattern - same shape,
		# plus a third pattern kind lowering.py's own version never had: a
		# bare name/wildcard always matches; a TaggedUnion member becomes a
		# tag Cmp + recurse into the member's own sub-pattern against
		# data.v_<member>; a VALUE pattern (ast.MatchValue - `case
		# Color.Red:`, `case 5:`, anything spelled as a dotted name or a
		# literal with no call-parens) becomes a plain == Compare against
		# the value expression, unresolved here - whatever `case Color.Red:`
		# folds to at lowering time (lowering.py's _expr_Attribute's own
		# CEnum-member-to-Const handling, for the enum case) decides
		# correctness the same way an ordinary `subj == Color.Red`
		# comparison already would. Anything else fails outright, exactly
		# as it always has, just reported here instead of lazily during
		# lowering.
		#
		# original_subject_name is only ever non-None on the OUTERMOST call
		# (visit_Match's own, never threaded into a recursive call against
		# an EXTRACTED payload below - narrowing only makes sense against a
		# name that already denotes a real, existing binding, which an
		# extracted payload expression never is) - see the ast.MatchClass
		# branch below for where it's actually used.
		if isinstance( pattern, ast.MatchAs ) and pattern.pattern is None:
			test = ast.Constant( value = True )
			ast.copy_location( test, node )
			if pattern.name is None:
				return test, []
			bind = ast.Assign( targets = [ ast.Name( id = pattern.name, ctx = ast.Store() ) ], value = subj_expr )
			ast.copy_location( bind, node )
			# is_match_binding: lowering.py's _stmt_Assign reads this to mark
			# the payload as read (ir.MarkUsed) right after its own real
			# Assign, regardless of whether this specific arm's body ever
			# goes on to use it - see its own comment for why. Distinct from
			# is_match_subject above (that one's for the SUBJECT's own
			# __match_subj_N relay, this is for an actual `case T(name):`
			# extracted payload) and from is_narrowing_bind (_build_narrow_
			# marker's own, mutually-exclusive same-name-reuse shape, never
			# reaches this branch at all - see _match_union_member's own
			# check above it)
			bind.is_match_binding = True
			# same reasoning as visit_Match's own subj_assign comment above:
			# this Assign is built directly, never dispatched through
			# self.visit()/visit_Assign, so nothing populates
			# self.locals[pattern.name] for free. Without this, a case body
			# statement that calls a method on the bound name as its LAST
			# statement (`case Result.Ok(w): w.close()`) hits
			# _stmt_diverges's `root.id in self.locals` pre-check, finds it
			# absent, and falls through to _resolve_callee_target - which
			# RAISES via the scope-stack-based find_name (a match-bound
			# local was never registered there either) and permanently
			# records a bogus "name 'w' is not defined" (discovery.fail()
			# records before raising, same trap 876fdc0 already fixed for
			# `self.foo()` - this is the same gap, just for an ordinary
			# extracted payload binding instead of the `self` parameter).
			# Same fix independently also closes a second gap: without a
			# self.locals entry, a later `if v is not None:` inside the same
			# case body couldn't recognize v as a narrowable union-typed
			# name (_is_none_narrowing_shape's own _type_of_expr call
			# returned None for it), silently skipping the narrowing an
			# ordinary local would get - confirmed via a real compile:
			# `match r: case Result.Ok(v): if v is not None: x = v` (v:
			# i32|None) failed to narrow, rejecting `x = v` as
			# i32|None-into-i32, even though the identical pattern against a
			# plain `v: i32|None = ...` local already narrowed correctly.
			self.locals[ pattern.name ] = self._type_of_expr( subj_expr )
			return test, [ bind ]

		if isinstance( pattern, ast.MatchValue ):
			# the value expression comes straight from user source (case
			# Color.Red:) and, unlike subj_expr (already visited by
			# whichever caller built it - visit_Match's own subj_ref, or an
			# outer _match_pattern call's synthesized payload_expr), has
			# never been visited yet - same reasoning as visit_Match's own
			# subj_assign.value above
			value = self.generic_visit_expr( pattern.value )
			test = ast.Compare( left = subj_expr, ops = [ ast.Eq() ], comparators = [ value ] )
			ast.copy_location( test, node )
			return test, []

		if isinstance( pattern, ast.MatchSingleton ) and pattern.value is None:
			# case None: against a T|None-shaped subject - same "does this
			# union have a None member" logic as visit_Compare's own `x is
			# None` rewrite, just reached from a match pattern instead of a
			# Compare. No sub-pattern to recurse into (None never binds
			# anything), so this is just the tag check alone, unlike the
			# ast.MatchClass branch below.
			subj_type = self._type_of_expr( subj_expr )
			if subj_type is None:
				self.discovery.fail( f'cannot determine the match subject\'s type: {ast.unparse(pattern)}', node )
			spec = self.resolver._as_specialization( subj_type ) # not a bare isinstance check - subj_type may already be eagerly-monomorphized, see visit_Match's own comment
			base = spec.base if spec is not None else subj_type
			if not isinstance( base, TaggedUnion ):
				self.discovery.fail( f'case None: requires a union-typed subject, got {getattr( subj_type, "qualname", subj_type )}: {ast.unparse(pattern)}', node )
			members = self._resolved_union_members( subj_type, base )
			none_type = self.discovery.get_none_type()
			none_member = next( ( attr for attr in members if attr.type is none_type ), None )
			if none_member is None:
				self.discovery.fail( f'{base.qualname} has no None member: {ast.unparse(pattern)}', node )
			tag_attr, _data_attr, _payload_cls, tags = self.resolver.union_storage.get( base )
			tag_expr = ast.Attribute( value = subj_expr, attr = tag_attr.stem, ctx = ast.Load() )
			ast.copy_location( tag_expr, node )
			test = ast.Compare( left = tag_expr, ops = [ ast.Eq() ], comparators = [ ast.Constant( value = tags[none_member.stem] ) ] )
			ast.copy_location( test, node )
			return test, []

		if isinstance( pattern, ast.MatchSequence ):
			# `case (a, b):` / `case Result.Ok((a, b)):` - (a, b) inside a
			# pattern parses to ast.MatchSequence. Arity is a static,
			# compile-time fact about the subject's tuple type (checked
			# below), unlike ast.MatchClass's real runtime tag Cmp, so no
			# runtime test is needed for it - only each element's own
			# sub-pattern test, ANDed together (mirrors ast.MatchAs's own
			# "no test needed" ast.Constant(True) convention above).
			# subj_expr here is always side-effect-free by construction (a
			# bare match-subject Name, or an Attribute chain built by
			# _match_union_member below) - re-lowering it into N synthesized
			# ast.Subscript reads (one per element, each recursed into
			# _match_pattern) is safe for exactly that reason, unlike
			# lowering.py's own plain-assignment tuple-unpacking, which
			# lowers node.value exactly once since IT can be side-effecting.
			if any( isinstance( p, ast.MatchStar ) for p in pattern.patterns ):
				self.discovery.fail( f'starred sequence patterns are not supported: {ast.unparse(pattern)}', node )
			subj_type = self._type_of_expr( subj_expr )
			if subj_type is None:
				self.discovery.fail( f'cannot determine the match subject\'s type: {ast.unparse(pattern)}', node )
			resolved_subj_type = self.resolver.ensure_resolved( subj_type )
			tuple_type = self.resolver.tuple_storage.tuple_type_for( resolved_subj_type )
			if tuple_type is None:
				self.discovery.fail(
					f'sequence pattern requires a tuple-typed subject, got {getattr( subj_type, "qualname", subj_type )}: {ast.unparse(pattern)}',
					node,
				)
			if len( tuple_type.elem_types ) != len( pattern.patterns ):
				self.discovery.fail(
					f'sequence pattern has {len(pattern.patterns)} element(s), tuple has {len(tuple_type.elem_types)}: {ast.unparse(pattern)}',
					node,
				)
			test: ast.expr = ast.Constant( value = True )
			ast.copy_location( test, node )
			binds: list[ast.stmt] = []
			for i, subpattern in enumerate( pattern.patterns ):
				elem_expr = ast.Subscript( value = subj_expr, slice = ast.Constant( value = i ), ctx = ast.Load() )
				ast.copy_location( elem_expr, node )
				elem_test, elem_binds = self._match_pattern( elem_expr, subpattern, node )
				binds.extend( elem_binds )
				if not ( isinstance( elem_test, ast.Constant ) and elem_test.value is True ):
					combined = ast.BoolOp( op = ast.And(), values = [ test, elem_test ] )
					ast.copy_location( combined, node )
					test = combined
			return test, binds

		if not isinstance( pattern, ast.MatchClass ):
			self.discovery.fail( f'unsupported match pattern: {ast.unparse(pattern)}', node )
		if pattern.kwd_patterns or len( pattern.patterns ) != 1:
			self.discovery.fail( f'match patterns support exactly one positional sub-pattern: {ast.unparse(pattern)}', node )

		if isinstance( pattern.cls, ast.Attribute ):
			# case Result.Ok(x): - the class path directly NAMES the union
			# (Result) and the member (Ok) as text - the union comes from
			# the PATTERN, the subject's own static type is never consulted
			# for THIS resolution step (finding owner/member by name) - only
			# below, to detect the nested-opaque-member case a bare name
			# lookup can't see on its own.
			owner = self._try_resolve_namespace( pattern.cls.value )
			if not isinstance( owner, TaggedUnion ):
				self.discovery.fail( f'unsupported match pattern class: {ast.unparse(pattern)}', node )
			owner = self.resolver.ensure_resolved( owner )
			member = next( ( attr for attr in owner.attributes if attr.stem == pattern.cls.attr ), None )
			if member is None:
				self.discovery.fail( f'{owner.qualname} has no member {pattern.cls.attr!r}: {ast.unparse(pattern)}', node )
			# a real, confirmed bug (not hypothetical): `owner` above is
			# resolved PURELY from the pattern's own text, with zero regard
			# for what the SUBJECT's own actual type is. That's correct
			# when the subject genuinely IS owner's own type directly (the
			# overwhelmingly common case, `match r: case Result.Ok(x):`
			# where r: Result[...]) - but when `owner` (a nominal union,
			# e.g. MyError) is instead nested OPAQUELY as one member of a
			# WIDER union that's the subject's real type (e.g. `e: MyError
			# | StopIteration`, `case MyError.Bad(_):`), the code built
			# below tests MyError's OWN internal tag position (Bad's
			# position within MyError) directly against the SUBJECT - which
			# is really the OUTER union's own tag storage, an entirely
			# different tag space. Confirmed via a real repro: silently
			# WRONG generated code (not a crash, not a compile error) -
			# `case MyError.Bad(_):` matched whenever the outer union's own
			# tag happened to equal Bad's position within MyError, which is
			# only ever correct by coincidence (MyError sorting first in
			# the outer union's own canonicalized member order). Every
			# `Generator[T,E]`'s error type now includes StopIteration
			# (PLAN_GENERATORS.md's StopIteration reversal) - since
			# `StopIteration` lives in builtins, it sorts ahead of almost
			# any user error type, so this shape is now the COMMON case for
			# generator error handling, not a rare edge case.
			#
			# Fixed by detecting the nested-opaque case here and building
			# the outer union's own match_union_member step FIRST, handing
			# it this SAME pattern node as its own inner_pattern - the
			# resulting recursive _match_pattern call re-enters this exact
			# branch, but against payload_expr (self.<data>.v_<owner's own
			# stem>), whose type genuinely IS `owner` directly, so the
			# ordinary (already-correct) case handles it from there with
			# zero further special-casing.
			subj_type = self._type_of_expr( subj_expr )
			subj_spec = self.resolver._as_specialization( subj_type ) if subj_type is not None else None
			subj_base = subj_spec.base if subj_spec is not None else subj_type
			if isinstance( subj_base, TaggedUnion ) and subj_base is not owner:
				outer_members = self._resolved_union_members( subj_type, subj_base )
				outer_member = next( ( attr for attr in outer_members if attr.type is owner ), None )
				if outer_member is not None:
					return self._match_union_member( subj_expr, subj_base, outer_member, pattern, node, original_subject_name )
				self.discovery.fail(
					f'{owner.qualname} is not {subj_base.qualname} and is not one of its members - match pattern '
					f'names a union unrelated to the subject\'s own type: {ast.unparse(pattern)}',
					node,
				)
			return self._match_union_member( subj_expr, owner, member, pattern.patterns[0], node, original_subject_name )

		if isinstance( pattern.cls, ast.Name ):
			# case str(c): / case None-leaf-typed-class(c): - the class
			# names a LEAF type directly (str), not a union+member path -
			# resolve the subject's own static union type (unlike the
			# ast.Attribute branch above, this genuinely needs it) and find
			# whichever member's .type IS this leaf (same identity lookup
			# _coerce_into_union already uses), then fall into the SAME
			# tag-Cmp + payload-GetAttr codegen the ast.Attribute branch
			# above already builds - just keyed by type identity instead of
			# member name (see _match_union_member).
			leaf_type = self._try_resolve_namespace( pattern.cls )
			if leaf_type is None:
				self.discovery.fail( f'unsupported match pattern class: {ast.unparse(pattern)}', node )
			subj_type = self._type_of_expr( subj_expr )
			if subj_type is None:
				self.discovery.fail( f'cannot determine the match subject\'s type: {ast.unparse(pattern)}', node )
			spec = self.resolver._as_specialization( subj_type ) # not a bare isinstance check - subj_type may already be eagerly-monomorphized, see visit_Match's own comment
			base = spec.base if spec is not None else subj_type
			if not isinstance( base, TaggedUnion ):
				self.discovery.fail( f'{ast.unparse(pattern)}: match subject is not a union type', node )
			members = self._resolved_union_members( subj_type, base )
			member = next( ( attr for attr in members if attr.type is leaf_type ), None )
			if member is None:
				self.discovery.fail( f'{base.qualname} has no member of type {getattr( leaf_type, "qualname", leaf_type )}: {ast.unparse(pattern)}', node )
			return self._match_union_member( subj_expr, base, member, pattern.patterns[0], node, original_subject_name )

		self.discovery.fail( f'unsupported match pattern class: {ast.unparse(pattern)}', node )

	def _resolved_union_members( self, subj_type: Type, base: TaggedUnion ) -> list[Variable]:
		''' base.attributes with every field's own .type force-resolved
		(each field is lazily resolved separately from the class itself -
		same as UnionStorage.get's/_lower_allocate_fields's identical
		loop), substituted for a genuine Specialization (T/E still bare
		TypeVars on the abstract base) via monomorphize_class - mirrors
		visit_Compare's own `x is None` rewrite and
		_rewrite_tagged_union_truthiness exactly. '''
		# _as_specialization, not a bare isinstance(subj_type, Specialization) -
		# subj_type may already be eagerly-monomorphized (resolve_declared_
		# types) to the concrete union itself, not a Specialization wrapper -
		# still needs the substituted (not abstract/TypeVar-typed) attrs, same
		# as the genuine-Specialization case below
		spec = self.resolver._as_specialization( subj_type )
		if spec is not None:
			return self.resolver.monomorphizer.monomorphize_class( spec ).attributes
		self.resolver.ensure_resolved( base )
		for attr in base.attributes:
			self.resolver.ensure_resolved( attr )
		return base.attributes

	def _resolve_case_member( self, base: TaggedUnion, members: list[Variable], pattern: ast.pattern ) -> Variable|None:
		''' a SILENT (never calls discovery.fail) probe: does `pattern`
		resolve to one specific member of `base` (whose own resolved
		members are `members`)? Mirrors _match_pattern's own
		ast.MatchSingleton(None)/ast.MatchClass resolution logic, but as a
		pure lookup with no codegen and no error reporting - used by
		visit_Match's own pre-pass to determine exhaustiveness (does the
		LAST case's own coverage, combined with every PRIOR case, provably
		account for every one of the union's own members - Trap 1's own
		generalization beyond a literal wildcard). "can't tell" is always
		a safe answer here (None), never a guess - a case whose pattern
		names a DIFFERENT union entirely (owner is not base) also returns
		None, since counting it toward THIS union's own coverage would be
		a real bug (silently dropping a still-needed tag check). '''
		if isinstance( pattern, ast.MatchSingleton ) and pattern.value is None:
			none_type = self.discovery.get_none_type()
			return next( ( attr for attr in members if attr.type is none_type ), None )
		if not isinstance( pattern, ast.MatchClass ) or pattern.kwd_patterns or len( pattern.patterns ) != 1:
			return None
		if isinstance( pattern.cls, ast.Attribute ):
			owner = self._try_resolve_callable_namespace( pattern.cls.value )
			if owner is not base:
				return None
			return next( ( attr for attr in members if attr.stem == pattern.cls.attr ), None )
		if isinstance( pattern.cls, ast.Name ):
			leaf_type = self._try_resolve_callable_namespace( pattern.cls )
			if leaf_type is None:
				return None
			return next( ( attr for attr in members if attr.type is leaf_type ), None )
		return None

	def _build_narrow_marker( self, name: str, member: Variable, node: ast.AST, *, attr_base: str|None = None, attr_name: str|None = None ) -> ast.Assign:
		''' the narrow-marker Assign shape - factored out of
		_match_union_member (its own same-name-reuse branch) so
		visit_Match's own wildcard/negation narrowing (Phase 6 - a
		wildcard arm narrowed to the union's OTHER member, deduced from a
		sibling case, rather than resolved from the pattern's own text)
		can build the identical shape without duplicating it. `member.type`
		must already be the resolved, concrete leaf type (both callers
		force-resolve/monomorphize before reaching here) - lowering.py's
		_stmt_Assign recognizes is_narrowing_bind and calls cfg.narrow(name,
		member) instead of emitting an ordinary assignment; narrowed_type
		is what _ReferenceResolver.__init__'s own self._narrowed uses to
		keep THIS pass's own union-shaped rewrites in sync with cfg.py's
		lowering-time narrowing (see its own comment). narrows_member_stem
		carries only the STEM (a plain string), not the Variable object
		itself - lowering.py re-resolves the real, substituted member
		against the subject's own already-monomorphized type instead, same
		pattern _coerce_into_union uses (member.type here may still be
		reached via an ABSTRACT class with T/E still bare TypeVars).

		attr_base/attr_name: set only for a single-level field-narrowing
		subject (`self.field`/`x.field`, see visit_If's own Attribute-shape
		branch) - `name` is then a synthetic `f'{base}::{attr}'` key (never
		collides with a real identifier, which can't contain `::`), used
		purely as cfg.py's own narrow()/narrowed_member() dict key. Real
		local names never set these two - lowering.py's _resolve_narrow_
		member uses their presence to tell "look up a local by this name"
		apart from "look up FIELD attr_name on local attr_base". '''
		narrow_marker = ast.Assign(
			targets = [ ast.Name( id = name, ctx = ast.Store() ) ],
			value = ast.Constant( value = None ),
		)
		ast.copy_location( narrow_marker, node )
		narrow_marker.is_narrowing_bind = True
		narrow_marker.narrows_member_stem = member.stem
		narrow_marker.narrowed_type = member.type
		if attr_base is not None:
			narrow_marker.narrow_attr_base = attr_base
			narrow_marker.narrow_attr_name = attr_name
		return narrow_marker

	def _match_union_member( self, subj_expr: ast.expr, union: TaggedUnion, member: Variable, inner_pattern: ast.pattern, node: ast.AST, original_subject_name: str|None ) -> tuple[ast.expr,list[ast.stmt]]:
		''' shared by both ways of landing on a (union, member) pair to
		match against - `case Result.Ok(x):` (member resolved by NAME off
		the pattern's own text) and `case str(c):` (member resolved by
		TYPE IDENTITY off the subject's own static type). Builds the tag
		Cmp, then either narrows (same-name reuse against the ORIGINAL
		subject) or extracts-and-recurses into the sub-pattern against
		data.v_<member>. '''
		tag_attr, data_attr, _payload_cls, tags = self.resolver.union_storage.get( union )
		tag_expr = ast.Attribute( value = subj_expr, attr = tag_attr.stem, ctx = ast.Load() )
		ast.copy_location( tag_expr, node )
		test = ast.Compare( left = tag_expr, ops = [ ast.Eq() ], comparators = [ ast.Constant( value = tags[member.stem] ) ] )
		ast.copy_location( test, node )
		if (
			isinstance( inner_pattern, ast.MatchAs ) and inner_pattern.pattern is None
			and inner_pattern.name is not None and inner_pattern.name == original_subject_name
		):
			# `match x: case T(x):` - the inner pattern reuses the
			# OUTER SUBJECT's own name (not some unrelated binding that
			# just happens to share it - original_subject_name is only
			# ever set by visit_Match's own top-level call, see
			# _match_pattern's own comment). x's own real Variable/storage
			# never changes - narrow it instead of extracting-and-
			# binding a same-named-but-distinct value (a real bug this
			# fixes: the old extract-and-bind here left x's own type
			# permanently stuck at the union's type, since lowering.py's
			# _stmt_Assign reuses an EXISTING name's own Variable
			# unchanged rather than ever narrowing it - confirmed via a
			# real repro, `case Result.Ok(r):` on a Result[str,MyError]
			# named r never actually narrowing r to str).
			# lowering.py's _stmt_Assign recognizes is_narrowing_bind
			# and calls cfg.narrow(name, member) instead of emitting an
			# ordinary assignment - see its own comment. Carries only
			# `member`'s own STEM (a plain string), not the Variable
			# object itself: `union` here may still be the ABSTRACT class
			# (T/E still bare TypeVars) - lowering.py re-resolves the real,
			# SUBSTITUTED member (str, not T) against the subject
			# variable's own already-monomorphized type instead, the
			# same pattern _coerce_into_union already uses.
			return test, [ self._build_narrow_marker( inner_pattern.name, member, node ) ]
		payload_expr = ast.Attribute(
			value = ast.Attribute( value = subj_expr, attr = data_attr.stem, ctx = ast.Load() ),
			attr = f'v_{member.stem}',
			ctx = ast.Load(),
		)
		ast.copy_location( payload_expr, node )
		inner_test, inner_binds = self._match_pattern( payload_expr, inner_pattern, node )
		combined = ast.BoolOp( op = ast.And(), values = [ test, inner_test ] )
		ast.copy_location( combined, node )
		return combined, inner_binds
