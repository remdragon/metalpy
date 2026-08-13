# stdlib imports:
import ast
from contextlib import nullcontext
import queue
import threading

# local imports:
from discovery import Discovery
from errors import CompileError
from monomorphize import Monomorphizer
from mpy_types import (
	CallableType, CEnum, ClassLike, CStruct, CUnion, Function, Module, Name, Overload,
	Parameter, RCClass, Scalar, Specialization, TaggedUnion, Type,
	TupleType, TypeVar, Variable,
)
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
			if free_fn.resolve is not None:
				free_fn.resolve()
		else:
			free_fn = free_overload
		free_call = ast.Call(
			func = ast.Attribute(
				value = ast.Name( id = 'sys', ctx = ast.Load() ),
				attr = 'free', ctx = ast.Load(),
			),
			args = [ ast.Name( id = 'self', ctx = ast.Load() ) ], keywords = [],
		)
		free_call.resolved_callee = free_fn
		free_call.end_lineno = None; free_call.end_col_offset = None
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

	def _build_field_teardown_ast( self, field_expr: ast.Attribute, field_type: Type ) -> list[ast.stmt]:
		''' recursively build AST statements to decref every RC leaf
		reachable from field_expr, given its declared type. '''
		base = field_type.base if isinstance( field_type, Specialization ) else field_type
		line = field_expr.lineno if hasattr( field_expr, 'lineno' ) and field_expr.lineno else 1

		# RCClass — compiler.decref(expr)
		if isinstance( base, RCClass ):
			if base.resolve is not None:
				base.resolve()
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
				member_base = member.type.base if isinstance( member.type, Specialization ) else member.type
				if not isinstance( member_base, RCClass ):
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

		# CUnion / CEnum / Scalar / Ptr — never RC, nothing to tear down
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
		''' true if `t` is an RCClass, possibly wrapped in a Specialization. '''
		base = t.base if isinstance( t, Specialization ) else t
		return isinstance( base, RCClass )

	def _is_pointer_representable( self, t: Type|None ) -> bool:
		''' true if `t`'s own runtime representation IS a single machine
		pointer - a real Ptr[T]/ConstPtr[T], OR an RCClass value (always a
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

	def _lookup_result_and_error_types( self, node: ast.AST, error_name: str ) -> tuple[ClassLike,ClassLike]:
		result_cls = self.discovery.find_name( 'Result', node )
		error_cls = self.discovery.find_name( error_name, node )
		return result_cls, error_cls

	def _require_result_return( self, node: ast.AST, result_cls: ClassLike, error_cls: ClassLike, alternatives: str, fn: Function|None = None ) -> None:
		return_type = fn.return_type if fn is not None else None
		spec = self._as_specialization( return_type )
		ok = (
			fn is not None
			and spec is not None
			and spec.base is result_cls
			and len( spec.args ) == 2
			and spec.args[1] is error_cls
		)
		if not ok:
			where = f'{fn.qualname} returns {return_type.qualname if return_type else None}' if fn is not None else 'this is not inside a function'
			self.discovery.fail(
				f'{ast.unparse(node)} requires the enclosing function to return Result[_,{error_cls.stem}] '
				f'({where}) - {alternatives}',
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
		if (
			isinstance( node, ast.Subscript ) and isinstance( node.value, ast.Name )
			and node.value.id in ( 'Callable', 'Closure' )
		):
			# Callable[[Arg1,...],Ret]/Closure[[Arg1,...],Ret] as a TYPE-
			# REFERENCE-context expression (compiler.cast(Closure[[],None],
			# x), compiler.sizeof(Callable[...]), ...) - discovery.py's own
			# visit_Subscript already recognizes this shape for ANNOTATIONS,
			# but that path is never reached from here (Callable/Closure
			# are recognized textually, not through find_name - a bare
			# ast.Name(id='Callable') node.value would otherwise fail
			# resolution outright, same as any other undefined name).
			# Mirrors visit_Subscript's own shape validation exactly
			shape_ok = (
				isinstance( node.slice, ast.Tuple )
				and len( node.slice.elts ) == 2
				and isinstance( node.slice.elts[0], ast.List )
			)
			if not shape_ok:
				self.discovery.fail( f"{node.value.id}[...] must look like {node.value.id}[[ArgType, ...], RetType]: {ast.unparse(node)}", node )
			arg_nodes, ret_node = node.slice.elts
			arg_types: list[Type] = []
			for a in arg_nodes.elts:
				resolved = self._try_resolve_namespace( a )
				if not isinstance( resolved, Type ):
					self.discovery.fail( f'{node.value.id}[...] argument is not a type: {ast.unparse(a)}', node )
				arg_types.append( resolved )
			return_type = self._try_resolve_namespace( ret_node )
			if not isinstance( return_type, Type ):
				self.discovery.fail( f'{node.value.id}[...] return type is not a type: {ast.unparse(ret_node)}', node )
			if node.value.id == 'Callable':
				return self.discovery._get_or_create_callable_type( arg_types, return_type )
			return self.discovery._get_or_create_closure_type( arg_types, return_type )
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
			# _attr_lookup's identical redirect for the non-callable case
			owner_type = self.ensure_resolved( owner_type.args[0] )
		if isinstance( owner_type, CStruct ):
			found = owner_type.chain_lookup( attr )
		else:
			names = getattr( owner_type, 'names', None )
			if not isinstance( names, dict ):
				self.discovery.fail( f'{owner_type!r} has no members, cannot look up {attr!r} ({ast.unparse(ctx)})', ctx )
			found = names.get( attr )
		if not isinstance( found, ( Function, Overload )):
			self.discovery.fail( f'{attr!r} is not callable on {owner_type.qualname if owner_type else "?"}', ctx )
		if isinstance( found, ( Function, Overload )):
			self._resolve_callable( found )
		return found

	def _resolve_callable( self, callee: Function|Overload ) -> None:
		if isinstance( callee, Function ):
			if callee.resolve is not None:
				callee.resolve()
		else:
			for fn in ( *callee.stubs, *callee.implementations ):
				if fn.resolve is not None:
					fn.resolve()

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

		reference = per_leaf[0][1]
		for member, fn in per_leaf[1:]:
			if fn.return_type is not reference.return_type:
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
		self.schedule( obj )
		if isinstance( obj, Specialization ):
			if isinstance( obj.base, Function ):
				return self.monomorphizer.monomorphized_function( obj )
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
	def __init__( self, resolver: TypeResolver, fn: Function ) -> None:
		self.resolver = resolver
		self.discovery = resolver.discovery
		self.fn = fn
		self.locals: dict[str,Type] = {}
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
			return self.locals.get( node.id )
		if isinstance( node, ast.Attribute ):
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
			return found.type if isinstance( found, Variable ) else None
		if isinstance( node, ast.Call ):
			target: object|None
			if isinstance( node.func, ast.Attribute ):
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
				if target.stem == 'or_return':
					# or_return() is never a real, scheduled/compiled function
					# (its declared body is a spec for lowering.py's own
					# _lower_or_return to special-case at the call site, not
					# something literally compilable - see that method's own
					# comment) - but ensure_resolved() below unconditionally
					# schedules whatever it's handed, with no exemption for
					# this one method. Reached here specifically for a bare
					# `x = <result_expr>.or_return()` (no type annotation) -
					# this pass's own speculative bookkeeping for x's type
					# then resolves the RECEIVER (a Result[T,E] Specialization)
					# down to its monomorphized concrete class two lines up,
					# whose own already-monomorphized 'or_return' entry (found
					# via names.get above) is what target is here - scheduling
					# THAT literally compiles or_return[T,E]'s spec-only body,
					# which fails the moment it does (confirmed directly: a
					# self-referential/same-file T, e.g. a method of Foo
					# returning Result[Foo,E] and or_return()-ing it from
					# elsewhere in Foo, reaches exactly this path and crashes
					# with "compiler.early_return(...) requires the enclosing
					# function to return Result[_,_]"). Result[T,E].or_return()
					# always returns T - read it straight off the receiver's
					# own Specialization args instead, no scheduling needed.
					target_cls_base = target.cls.base if isinstance( target.cls, Specialization ) else target.cls
					if (
						target_cls_base is self.discovery.find_name_or_none( 'Result' )
						and isinstance( target.cls, Specialization ) and target.cls.args
					):
						return target.cls.args[0]
					return None
				target = self.resolver.ensure_resolved( target )
				return target.return_type if isinstance( target, Function ) else None
			if isinstance( target, ( RCClass, CStruct, CUnion, TaggedUnion, CEnum )):
				# a plain (non-generic) construction call, Foo(...) - its own
				# type is just the class itself. A GENERIC construction
				# (Result(...), inferring its own type params from the call's
				# arguments) is deliberately not handled here - that's the
				# classes half of this work, not yet done
				return target
			return None
		return None

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
			base = self.resolver.ensure_resolved( base )
			if isinstance( base, TaggedUnion ):
				self.resolver.union_storage.get( base )
			names = getattr( base, 'names', None )
			if not isinstance( names, dict ):
				return None
			return names.get( node.attr )
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
			return base, args

		target = self._try_resolve_callable_namespace( func )
		if not isinstance( target, Function ) or not target.type_params:
			return None
		if target.resolve is not None:
			target.resolve()
		args = self._infer_generic_args( node, target, target.type_params )
		if args is None:
			return None
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
			if not trust_literals and isinstance( expr, ast.Constant ):
				continue
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
		spec = self.discovery._get_or_create_specialization( target_cls, args )
		concrete_cls = self.resolver.monomorphizer.monomorphize_class( spec )
		concrete_init = concrete_cls.names.get( '__init__' )
		if not isinstance( concrete_init, Function ):
			return None # shouldn't happen (monomorphize_class's own method loop always substitutes a plain __init__ too), but stay silent/consistent with this pass's own discipline rather than assert
		return concrete_cls, concrete_init

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
				if target.resolve is not None:
					target.resolve()
				init = target.names.get( '__init__' )
				if isinstance( init, Function ):
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
		return node

	# --- local type tracking ---

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
		return node

	def visit_Assign( self, node: ast.Assign ) -> ast.Assign:
		self.generic_visit( node )
		if len( node.targets ) == 1 and isinstance( node.targets[0], ast.Name ):
			self.locals[node.targets[0].id] = self._type_of_expr( node.value )
			# see visit_AnnAssign's own comment - an ordinary reassignment
			# invalidates prior narrowing
			self._narrowed.pop( node.targets[0].id, None )
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
		left_is_none = isinstance( node.left, ast.Constant ) and node.left.value is None
		right_is_none = isinstance( node.comparators[0], ast.Constant ) and node.comparators[0].value is None
		if left_is_none == right_is_none:
			return node # both-None/neither-None - not this rewrite's shape, leave for lowering's ordinary is/is-not handling
		other = node.comparators[0] if left_is_none else node.left
		other_type = self._type_of_expr( other )
		if other_type is None:
			return node # can't determine - leave as ordinary `is`/`is not`, lowering's own _lower_is_comparison handles the non-union fallback
		# unwrap a Specialization to its ABSTRACT base, same as
		# Lowering._tagged_union_shape - "does this have a None member" is
		# substitution-independent (None doesn't vary by specialization), so
		# no monomorphize_class call is needed here. Critically, must NOT
		# call ensure_resolved(other_type) first: that would swap a
		# Specialization for its MONOMORPHIZED copy, whose own tag/data
		# (already built by monomorphize_class) would collide with
		# UnionStorage.get() trying to synthesize them again as if for a
		# fresh union (same mistake, and fix, as lowering.py's
		# _lower_allocate_fields TaggedUnion branch had)
		base = other_type.base if isinstance( other_type, Specialization ) else other_type
		if not isinstance( base, TaggedUnion ):
			return node
		members = self._resolved_union_members( other_type, base )
		none_type = self.discovery.get_none_type()
		none_member = next( ( attr for attr in members if attr.type is none_type ), None )
		if none_member is None:
			return node
		tag_attr, _data_attr, _payload_cls, tags = self.resolver.union_storage.get( base )
		tag_expr = ast.Attribute( value = other, attr = tag_attr.stem, ctx = ast.Load() )
		ast.copy_location( tag_expr, node )
		op = ast.NotEq() if isinstance( node.ops[0], ast.IsNot ) else ast.Eq()
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
		base = subj_type.base if isinstance( subj_type, Specialization ) else subj_type
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
		future work). '''
		expr_type = self._type_of_expr( expr_node )
		if expr_type is None:
			return None
		base = expr_type.base if isinstance( expr_type, Specialization ) else expr_type
		if not isinstance( base, TaggedUnion ):
			return None
		if isinstance( expr_type, Specialization ):
			members = self.resolver.monomorphizer.monomorphize_class( expr_type ).attributes
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
			value_expr = ast.Call(
				func = ast.Attribute( value = payload_expr, attr = '__bool__', ctx = ast.Load() ),
				args = [],
				keywords = [],
			)
			ast.copy_location( value_expr, ctx_node )
		# synthesize: tag_cmp and value_expr
		result = ast.BoolOp( op = ast.And(), values = [ tag_cmp, value_expr ] )
		ast.copy_location( result, ctx_node )
		return result

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
		base = subj_type.base if isinstance( subj_type, Specialization ) else subj_type
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
		folded = self._try_fold_is_rc_if( node )
		if folded is not None:
			return folded
		desugared = self._try_desugar_type_is_if( node )
		if desugared is not None:
			return desugared
		# rewrite test BEFORE generic_visit recurses into it, so the new BoolOp
		# children (Name references, Compare, Call) are visited normally
		rewritten = self._rewrite_tagged_union_truthiness( node.test, node )
		if rewritten is not None:
			node.test = rewritten
		self.generic_visit( node )
		return node

	def visit_While( self, node: ast.While ) -> ast.While:
		''' Phase 7: `while type(x) is T:`/`while type(x) is not T:`/
		`while instanceof(x, T):` against a union-typed, bare-Name x -
		narrows x for the loop BODY's own duration (the matched member for
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
		if shape is not None and isinstance( shape[0], ast.Name ):
			subject_expr, _type_expr, base, member, is_not = shape
			subject_name = subject_expr.id
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
			node.body = [ self._build_narrow_marker( subject_name, body_member, node ), *node.body ]
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

	def visit_BoolOp( self, node: ast.BoolOp ) -> ast.BoolOp:
		# each operand of `and`/`or` is a boolean context — rewrite
		# T|None operands BEFORE generic_visit recurses into the old nodes
		for i, value in enumerate( node.values ):
			rewritten = self._rewrite_tagged_union_truthiness( value, node )
			if rewritten is not None:
				node.values[i] = rewritten
		self.generic_visit( node )
		return node

	def visit_IfExp( self, node: ast.IfExp ) -> ast.IfExp:
		# ternary `x if cond else y` — cond is a boolean context
		rewritten = self._rewrite_tagged_union_truthiness( node.test, node )
		if rewritten is not None:
			node.test = rewritten
		self.generic_visit( node )
		return node

	def visit_Assert( self, node: ast.Assert ) -> list[ast.stmt]:
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
		return [ stmt ]

	# --- rewrite 2: match statements ---

	def visit_Match( self, node: ast.Match ) -> list[ast.stmt]:
		unique = self._label_id
		self._label_id += 1
		subj_name = f'__match_subj_{unique}'
		subj_assign = ast.Assign( targets = [ ast.Name( id = subj_name, ctx = ast.Store() ) ], value = self.generic_visit_expr( node.subject ))
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
		base = subj_type.base if isinstance( subj_type, Specialization ) else subj_type
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
				flatten_this_case and wildcard_narrow_member is not None and original_subject_name is not None
				and isinstance( case.pattern, ast.MatchAs ) and case.pattern.pattern is None and case.pattern.name is None
			):
				# the wildcard's own _match_pattern call above returned
				# binds=[] (a true, unnamed wildcard never binds anything on
				# its own) - override with a real narrow-marker targeting the
				# DEDUCED other member, computed in the pre-pass above
				binds = [ self._build_narrow_marker( original_subject_name, wildcard_narrow_member, node ) ]
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
				terminates = bool( case.body ) and isinstance( case.body[-1], ( ast.Return, ast.Break, ast.Continue ))
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
			base = subj_type.base if isinstance( subj_type, Specialization ) else subj_type
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

		if not isinstance( pattern, ast.MatchClass ):
			self.discovery.fail( f'unsupported match pattern: {ast.unparse(pattern)}', node )
		if pattern.kwd_patterns or len( pattern.patterns ) != 1:
			self.discovery.fail( f'match patterns support exactly one positional sub-pattern: {ast.unparse(pattern)}', node )

		if isinstance( pattern.cls, ast.Attribute ):
			# case Result.Ok(x): - the class path directly NAMES the union
			# (Result) and the member (Ok) as text - the union comes from
			# the PATTERN, the subject's own static type is never consulted
			owner = self._try_resolve_namespace( pattern.cls.value )
			if not isinstance( owner, TaggedUnion ):
				self.discovery.fail( f'unsupported match pattern class: {ast.unparse(pattern)}', node )
			owner = self.resolver.ensure_resolved( owner )
			member = next( ( attr for attr in owner.attributes if attr.stem == pattern.cls.attr ), None )
			if member is None:
				self.discovery.fail( f'{owner.qualname} has no member {pattern.cls.attr!r}: {ast.unparse(pattern)}', node )
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
			base = subj_type.base if isinstance( subj_type, Specialization ) else subj_type
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
		if isinstance( subj_type, Specialization ):
			return self.resolver.monomorphizer.monomorphize_class( subj_type ).attributes
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

	def _build_narrow_marker( self, name: str, member: Variable, node: ast.AST ) -> ast.Assign:
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
		reached via an ABSTRACT class with T/E still bare TypeVars). '''
		narrow_marker = ast.Assign(
			targets = [ ast.Name( id = name, ctx = ast.Store() ) ],
			value = ast.Constant( value = None ),
		)
		ast.copy_location( narrow_marker, node )
		narrow_marker.is_narrowing_bind = True
		narrow_marker.narrows_member_stem = member.stem
		narrow_marker.narrowed_type = member.type
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
