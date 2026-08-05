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
	CEnum, ClassLike, CStruct, CUnion, Function, Module, Name, Parameter, RCClass,
	Specialization, TaggedUnion, Type, TypeVar, Variable,
)
from union_storage import UnionStorage

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
		self.monomorphizer = Monomorphizer( discovery, self.schedule, self.union_storage )
		# keyed by id(fn.node), not id(fn) - the SAME shared AST body object
		# is reused by every monomorphized copy of a generic function (see
		# resolve_function_body's own docstring)
		self._body_resolved: set[int] = set()

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
		if isinstance( unit, Specialization ):
			if isinstance( unit.base, Function ):
				# an explicit generic function instantiation (sys.alloc[u8])
				# monomorphizes - a real, distinct compile unit in its own
				# right: queued directly rather than decomposed
				with self._seen_lock:
					if id( unit ) in self._seen:
						return
					self._seen.add( id( unit ))
				self.queue.put( unit )
				return
			if isinstance( unit.base, ( RCClass, CStruct, CUnion, TaggedUnion )):
				# a concrete generic CLASS specialization (Result[i32,
				# OverflowError]) - also queued directly (mirroring the
				# Function-based branch above), giving it its own real
				# struct/union layout via Lowering.monomorphize_class (see
				# Compiler._lower), so it lands in compiler.cstructs/.cunions/
				# .tagged_unions/.rcclasses just like any other compile
				# unit - a future emitter never has to independently
				# rediscover/resynthesize a concrete specialization itself.
				# Unlike the Function case, its own concrete type ARGS are
				# also independently scheduled here: a generic function's
				# args get scheduled incidentally via its own body/
				# signature (_emit_generic_call's explicit schedule()
				# calls), but a class specialization's substituted field
				# types (Result[i32,OverflowError]'s own OverflowError, for
				# instance) have no equivalent "body" to walk for that
				with self._seen_lock:
					if id( unit ) in self._seen:
						return
					self._seen.add( id( unit ))
				self.queue.put( unit )
				for arg in unit.args:
					self.schedule( arg )
				return
			self.schedule( unit.base )
			for arg in unit.args:
				self.schedule( arg )
			return
		if not isinstance( unit, ( Function, ClassLike )) and not ( isinstance( unit, Variable ) and unit.is_global ):
			# not a real compile unit: a Module (walked mid-namespace-lookup,
			# e.g. the `sys` in `sys.alloc(...)`), a class field/parameter/
			# local Variable, a bare Scalar/TypeVar, an Overload group itself
			# (only a resolved member is ever actually compiled) - all
			# harmless to just drop here rather than every caller having to
			# know not to pass them in the first place
			return
		with self._seen_lock:
			if id( unit ) in self._seen:
				return
			self._seen.add( id( unit ))
		self.queue.put( unit )

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
		args = self._infer_generic_args( node, target )
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

	def _infer_generic_args( self, node: ast.Call, target: Function ) -> list[Type]|None:
		# a BARE call to a generic function (mylen(a), no explicit [T]) - T
		# has to be inferred from the arguments' own (best-effort,
		# _type_of_expr-derived) types. Mirrors Lowering.
		# _lower_inferred_generic_call's unification, just against
		# source-level types instead of already-lowered IR operand types
		pairs = self._pair_call_args_for_inference( target, node )
		if pairs is None:
			return None
		type_params = target.type_params or []
		bindings: dict[int,Type] = {} # id(TypeVar) -> the concrete Type it was inferred as
		for param, expr in pairs:
			actual = self._type_of_expr( expr )
			if actual is None or isinstance( actual, TypeVar ):
				continue # can't determine this one - not an error here, just doesn't contribute a binding (see the "missing" check below)
			if not self._unify_type_param( type_params, param.type, actual, bindings ):
				return None # conflicting inference - bail, let lowering's own _unify_type_param report it
		if any( id( tv ) not in bindings for tv in type_params ):
			return None # couldn't infer everything from what this pass could determine - lowering's own (stronger, IR-level) inference gets a full attempt
		return [ bindings[id(tv)] for tv in type_params ]

	def _unify_type_param( self, type_params: list[TypeVar], declared: Type|None, actual: Type|None, bindings: dict[int,Type] ) -> bool:
		# ported from Lowering._unify_type_param, minus the discovery.fail()
		# call on conflict - returns False instead, meaning "bail, this
		# isn't this pass's to resolve or report" (see
		# _try_resolve_generic_call's own docstring)
		if declared is None or actual is None:
			return True
		if any( declared is tv for tv in type_params ):
			existing = bindings.get( id( declared ))
			if existing is not None and existing is not actual:
				return False
			bindings[ id( declared )] = actual
			return True
		if isinstance( declared, Specialization ) and isinstance( actual, Specialization ) and declared.base is actual.base:
			return all( self._unify_type_param( type_params, d, a, bindings ) for d, a in zip( declared.args, actual.args ))
		return True # this parameter position doesn't mention any of type_params - nothing to infer here

	def visit_Call( self, node: ast.Call ) -> ast.Call:
		self.generic_visit( node )
		resolved = self._try_resolve_generic_call( node )
		if resolved is None:
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
		return node

	def visit_Assign( self, node: ast.Assign ) -> ast.Assign:
		self.generic_visit( node )
		if len( node.targets ) == 1 and isinstance( node.targets[0], ast.Name ):
			self.locals[node.targets[0].id] = self._type_of_expr( node.value )
		return node

	# --- rewrite 1: is/is not None ---

	def visit_Compare( self, node: ast.Compare ) -> ast.expr:
		self.generic_visit( node )
		if len( node.ops ) != 1 or not isinstance( node.ops[0], ( ast.Is, ast.IsNot )):
			return node
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
		# base.attributes may still be empty/unresolved here (self.locals'
		# own values, e.g. from a bare annotation, never force a class body
		# to actually resolve) - mirror Lowering._tagged_union_shape exactly:
		# for a Specialization, monomorphize_class already resolves base as
		# a side effect of building the substituted copy; for a bare union,
		# force it directly. Either way, UnionStorage.get() below still
		# always takes the ABSTRACT base, never a monomorphized copy (see
		# comment above)
		if isinstance( other_type, Specialization ):
			members = self.resolver.monomorphizer.monomorphize_class( other_type ).attributes
		else:
			self.resolver.ensure_resolved( base )
			for attr in base.attributes: # each field's own .type is lazily resolved, separate from the class itself - same as UnionStorage.get's/_lower_allocate_fields's identical loop
				self.resolver.ensure_resolved( attr )
			members = base.attributes
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

	# --- rewrite 2: match statements ---

	def visit_Match( self, node: ast.Match ) -> list[ast.stmt]:
		unique = self._label_id
		self._label_id += 1
		subj_name = f'__match_subj_{unique}'
		subj_assign = ast.Assign( targets = [ ast.Name( id = subj_name, ctx = ast.Store() ) ], value = self.generic_visit_expr( node.subject ))
		ast.copy_location( subj_assign, node )
		subj_ref = ast.Name( id = subj_name, ctx = ast.Load() )
		ast.copy_location( subj_ref, node )

		chain: ast.If|None = None
		tail: ast.If|None = None
		for case in node.cases:
			if case.guard is not None:
				self.discovery.fail( f'match guards (case ... if ...) are not yet supported: {ast.unparse(case.pattern)}', node )
			test, binds = self._match_pattern( subj_ref, case.pattern, node )
			body = [ self.visit( stmt ) for stmt in case.body ]
			arm = ast.If( test = test, body = [ *binds, *body ], orelse = [] )
			ast.copy_location( arm, node )
			if chain is None:
				chain = arm
			else:
				tail.orelse = [ arm ]
			tail = arm
		return [ subj_assign, chain ] if chain is not None else [ subj_assign ]

	def generic_visit_expr( self, node: ast.expr ) -> ast.expr:
		# generic_visit() itself returns the node (mutated in place, for an
		# expr) - named wrapper only so visit_Match's own use above reads
		# clearly as "visit this expression", matching visit_Compare/
		# visit_AnnAssign/visit_Assign's own self.generic_visit(node) calls
		self.generic_visit( node )
		return node

	def _match_pattern( self, subj_expr: ast.expr, pattern: ast.pattern, node: ast.AST ) -> tuple[ast.expr,list[ast.stmt]]:
		# ported from lowering.py's Lowering._match_pattern - same shape,
		# same two supported pattern kinds (a bare name/wildcard always
		# matches; a TaggedUnion member becomes a tag Cmp + recurse into
		# the member's own sub-pattern against data.v_<member>) - anything
		# else fails outright, exactly as it always has, just reported
		# here instead of lazily during lowering
		if isinstance( pattern, ast.MatchAs ) and pattern.pattern is None:
			test = ast.Constant( value = True )
			ast.copy_location( test, node )
			if pattern.name is None:
				return test, []
			bind = ast.Assign( targets = [ ast.Name( id = pattern.name, ctx = ast.Store() ) ], value = subj_expr )
			ast.copy_location( bind, node )
			return test, [ bind ]

		if not isinstance( pattern, ast.MatchClass ):
			self.discovery.fail( f'unsupported match pattern: {ast.unparse(pattern)}', node )
		if pattern.kwd_patterns or len( pattern.patterns ) != 1:
			self.discovery.fail( f'match patterns support exactly one positional sub-pattern: {ast.unparse(pattern)}', node )
		if not isinstance( pattern.cls, ast.Attribute ):
			self.discovery.fail( f'unsupported match pattern class: {ast.unparse(pattern)}', node )

		owner = self._try_resolve_namespace( pattern.cls.value )
		if isinstance( owner, TaggedUnion ):
			owner = self.resolver.ensure_resolved( owner )
			member = next( ( attr for attr in owner.attributes if attr.stem == pattern.cls.attr ), None )
			if member is None:
				self.discovery.fail( f'{owner.qualname} has no member {pattern.cls.attr!r}: {ast.unparse(pattern)}', node )
			tag_attr, data_attr, _payload_cls, tags = self.resolver.union_storage.get( owner )
			tag_expr = ast.Attribute( value = subj_expr, attr = tag_attr.stem, ctx = ast.Load() )
			ast.copy_location( tag_expr, node )
			test = ast.Compare( left = tag_expr, ops = [ ast.Eq() ], comparators = [ ast.Constant( value = tags[member.stem] ) ] )
			ast.copy_location( test, node )
			payload_expr = ast.Attribute(
				value = ast.Attribute( value = subj_expr, attr = data_attr.stem, ctx = ast.Load() ),
				attr = f'v_{member.stem}',
				ctx = ast.Load(),
			)
			ast.copy_location( payload_expr, node )
			inner_test, inner_binds = self._match_pattern( payload_expr, pattern.patterns[0], node )
			combined = ast.BoolOp( op = ast.And(), values = [ test, inner_test ] )
			ast.copy_location( combined, node )
			return combined, inner_binds

		self.discovery.fail( f'unsupported match pattern class: {ast.unparse(pattern)}', node )
