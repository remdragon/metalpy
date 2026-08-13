# stdlib imports:
import copy
from dataclasses import replace
from typing import Callable

# local imports:
from discovery import Discovery
from mpy_types import Type, TypeVar, Specialization, TaggedUnion, CUnion, ClassLike, Function, Overload, Variable, CallableType, ClosureType, TupleType
from tuple_storage import TupleStorage
from union_storage import UnionStorage, build_member_constructor

'''
Monomorphization - giving a concrete generic instantiation (sys.alloc[u8],
Result[i32,OverflowError]) its own real, independent compiled body/layout,
memoized by id(Specialization) so every reference to the same instantiation
reuses the same object - split out of lowering.py since it depends only on
Discovery, schedule, and UnionStorage, never on statement/expression
lowering or CFG state. See Monomorphizer's own docstring.
'''

class Monomorphizer:
	'''
	Owns both halves of monomorphization:
	- monomorphized_function: a distinct compiled Function per explicit
	  generic instantiation (sys.alloc[u8] vs sys.alloc[u32] are two
	  separate functions, each with T bound to a concrete type throughout)
	- monomorphize_class: a concrete generic class specialization's own
	  real struct/union layout (Result[i32,OverflowError]'s own attributes,
	  substituted from Result's shared, abstract ones)
	both built via the same substitute_type_params helper, and both
	memoized here for the whole Lowering instance's lifetime.
	'''

	def __init__( self, discovery: Discovery, schedule: Callable[[object],None], union_storage: UnionStorage, tuple_storage: TupleStorage ) -> None:
		self.discovery = discovery
		self.schedule = schedule
		self._union_storage = union_storage
		self._tuple_storage = tuple_storage
		# no separate memo tables here - the monomorphized result (Function
		# or ClassLike, whichever matches spec.base's own kind) is cached
		# directly on spec.monomorphized (see mpy_types.py's Specialization)
		# since a Specialization is already the canonical, memoized-by-
		# qualname object for its own (base, args) pair - every reference to
		# the same instantiation, from anywhere, shares that one cache
		#
		# id(Specialization) currently mid-construction inside
		# monomorphize_class - guards substitute_type_params's own eager-
		# monomorphize step (see _is_concrete's use there) against infinite
		# recursion on a self-referential generic class: a method declared
		# inside Result[T,E] that itself returns Result[T,E] substitutes,
		# for Result[i32,MyError], to a return type of Result[i32,MyError]
		# again - the SAME spec this monomorphize_class call is still in
		# the middle of building, whose .monomorphized cache slot isn't
		# set yet. Recursing into monomorphize_class for it again would
		# just repeat the same unfinished work forever; this set lets
		# substitute_type_params notice and fall back to handing back the
		# bare (but now concrete) Specialization instead, exactly like
		# before this eager step existed - a real, correct Specialization
		# whose own .monomorphized DOES get filled in, by the very
		# monomorphize_class call already in progress for it
		self._building: set[int] = set()
		# id(monomorphized ClassLike) -> the Specialization it came from -
		# recovers "what generic instantiation is this" for a caller that
		# only has the concrete object in hand, no Specialization wrapper
		# to read .base/.args from. Needed specifically because of the
		# eager-monomorphize step above: substitute_type_params can hand
		# back an ALREADY-MONOMORPHIZED object instead of a Specialization
		# the moment a substitution actually changes something (e.g. a
		# method's own Result[None,E] return type, once E is bound to a
		# concrete error class) - a caller checking "is this Result-
		# shaped, and if so what are its args" (see type_resolver.py's
		# _result_shape/_require_result_return) needs this fallback or it
		# silently stops recognizing a perfectly well-formed Result[None,
		# _] the moment eager substitution beat it to unwrapping the
		# Specialization. Only monomorphize_class populates this -
		# monomorphized_function's own output is never affected by the
		# eager-monomorphize step this exists for (that step only ever
		# fires on a ClassLike-based Specialization, substitute_type_
		# params' own condition - see its own comment)
		self._origins: dict[int,Specialization] = {}

	def _ensure_resolved( self, obj: object ) -> None:
		# same discipline as Lowering._ensure_resolved - duplicated here
		# rather than depending back on Lowering, since this class is meant
		# to be usable standalone
		resolve = getattr( obj, 'resolve', None )
		if resolve is not None:
			resolve()
		self.schedule( obj )

	def _is_concrete( self, t: Type|None ) -> bool:
		''' true if `t` has no TypeVar anywhere in it, recursively. A
		Specialization built from a fully-concrete arg list is a REAL,
		instantiable type (Result[i32,MyError]); one that still mentions a
		TypeVar (Result[T,E], or Result[Box[T],E]) is still abstract -
		monomorphizing it would silently build a bogus "concrete" class
		whose own fields are still typed with those TypeVars, and cache it
		under spec.monomorphized as if it really were one. Same shape as
		_ReferenceResolver's identical TypeVar check for the function-call
		case (type_resolver.py's _try_resolve_generic_call) '''
		if isinstance( t, TypeVar ):
			return False
		if isinstance( t, Specialization ):
			return self._is_concrete( t.base ) and all( self._is_concrete( a ) for a in t.args )
		return True

	def substitute_type_params( self, t: Type|None, type_params: list[TypeVar], args: list[Type] ) -> Type|None:
		if isinstance( t, TypeVar ):
			for param, arg in zip( type_params, args ):
				if t is param:
					# PLAN_TUPLE.md, found by a real hang (not anticipated
					# up front): a bare, unresolved TupleType bound to T
					# (e.g. Result[T,E].Ok's own `value: T`, T bound to
					# tuple[int,int] by generic-call inference) is NEVER
					# itself emittable - unlike a bare Specialization
					# (which the branch just below already eagerly
					# monomorphizes for the exact same reason, "the single
					# highest-leverage fix point... every future reader
					# would otherwise need its OWN ensure_resolved call to
					# unwrap"), emitter_c.py has no ensure_resolved to call
					# on its own (no Discovery/TypeResolver instance around
					# - see _callable_ptr_type's own comment) and genuinely
					# has no c_type()/mangle_type() support for a bare
					# TupleType at all. Resolved here, once, regardless of
					# which "argument position" (a literal's own already-
					# resolved type, or a bare annotation never resolved at
					# all) `arg` happened to come from - _unify_type_param's
					# own last-writer-wins bindings dict makes the ORDER of
					# those two non-deterministic from here, so this can't
					# be fixed by reordering call sites instead.
					if isinstance( arg, TupleType ):
						return self._tuple_storage.get( arg )
					return arg
			return t
		if isinstance( t, Specialization ):
			substituted_args = [ self.substitute_type_params( a, type_params, args ) for a in t.args ]
			if all( sa is a for sa, a in zip( substituted_args, t.args )):
				return t
			substituted = self.discovery._get_or_create_specialization( t.base, substituted_args )
			if isinstance( t.base, ClassLike ) and self._is_concrete( substituted ) and id( substituted ) not in self._building:
				# the substitution just produced a fully-concrete class
				# Specialization - e.g. Result[T,E]'s own `data: Result$data
				# [T,E]` field, substituted for Result[i32,MyError], becomes
				# Result$data[i32,MyError]. Monomorphize it immediately
				# rather than handing back a bare Specialization that every
				# future reader (attribute access, self-typing, isinstance-
				# style queries) would need its OWN ensure_resolved call to
				# unwrap - this is the single highest-leverage fix point:
				# every concrete field/parameter/return type substituted
				# through monomorphize_class or monomorphized_function flows
				# through here (see PLAN_RESOLVE_CLASS_SPECIALIZATIONS.md)
				#
				# schedule() too, not just monomorphize_class() - this path
				# is reached from monomorphized_function's own substitution
				# of a METHOD's return/parameter types (see monomorphize_
				# class's method loop), which runs unconditionally for
				# every plain method a generic class has, whether or not
				# that method is ever actually CALLED anywhere. Without an
				# explicit schedule() here, a method that's never called
				# has nothing else to ever enqueue this nested generic type
				# as a real compile unit - monomorphize_class() alone only
				# builds and memoizes the concrete object (spec.
				# monomorphized), it never appends it to compiler.cstructs/
				# .cunions/.tagged_unions/.rcclasses itself (only compiler.
				# py's own queue-draining _lower() does that) - confirmed by
				# a real repro: a generic class with an uncalled method
				# returning Result[OtherGeneric[T],E] left OtherGeneric[i32]
				# forward-declared but never given a body, an "incomplete
				# type" C compile error the moment that Result's own
				# payload union (itself scheduled unconditionally by
				# monomorphize_class's TaggedUnion branch) embeds it by
				# value
				self.schedule( substituted )
				return self.monomorphize_class( substituted )
			return substituted
		if isinstance( t, CallableType ):
			# Callable[[Arg1,...],Ret] can mention a type param in either its
			# own arg_types or its return_type (e.g. a generic function's own
			# key: Callable[[T],K] parameter) - same recursive-rebuild-and-
			# intern shape as the Specialization branch above, just through
			# _get_or_create_callable_type (PLAN_CALLABLE.md) instead of
			# _get_or_create_specialization. Never itself a ClassLike (it has
			# no members of its own - see CallableType's own docstring), so
			# no monomorphize_class/schedule() step is needed the way a
			# Specialization's does
			substituted_arg_types = [ self.substitute_type_params( a, type_params, args ) for a in t.arg_types ]
			substituted_return_type = self.substitute_type_params( t.return_type, type_params, args )
			if all( sa is a for sa, a in zip( substituted_arg_types, t.arg_types )) and substituted_return_type is t.return_type:
				return t
			return self.discovery._get_or_create_callable_type( substituted_arg_types, substituted_return_type )
		if isinstance( t, ClosureType ):
			# Closure[[Arg1,...],Ret] (see ClosureType's own docstring) can
			# mention a type param in its own arg_types/return_type the same
			# way CallableType above can - e.g. dict[K,V].with_lock's own
			# `body: Closure[[UnsafeDict[K,V]],None]` parameter. Same
			# recursive-rebuild-and-intern shape, through
			# _get_or_create_closure_type instead of _get_or_create_
			# callable_type. Unlike CallableType, ClosureType IS a real
			# RCClass (fn/self fields, its own destructor) - but neither
			# field depends on arg_types/return_type, and _get_or_create_
			# closure_type already interns/memoizes by (arg_types,
			# return_type) key and wires up its own lazy .resolve, so
			# rebuilding through it is enough; no separate
			# monomorphize_class step needed (mirrors CallableType, which
			# is never itself a ClassLike either).
			substituted_arg_types = [ self.substitute_type_params( a, type_params, args ) for a in t.arg_types ]
			substituted_return_type = self.substitute_type_params( t.return_type, type_params, args )
			if all( sa is a for sa, a in zip( substituted_arg_types, t.arg_types )) and substituted_return_type is t.return_type:
				return t
			return self.discovery._get_or_create_closure_type( substituted_arg_types, substituted_return_type )
		if isinstance( t, TaggedUnion ) and t.file is None:
			# an ANONYMOUS union (T|None, synthesized by discovery.py's own
			# _get_or_create_union - file is None only for these, never for
			# a real, user-declared @union class, which must stay identity-
			# based/never rebuilt this way) can mention a type param directly
			# in one of its own leaves (e.g. Result[T,E].unwrap_or's own
			# declared `T|None` return type) - substitute each leaf and
			# rebuild through the same canonicalizing constructor so the
			# result is the same shared, memoized union any other T|None
			# reference resolves to, not a fresh one-off copy
			leaf_types = [ attr.type for attr in t.attributes ]
			substituted_leaves = [ self.substitute_type_params( lt, type_params, args ) for lt in leaf_types ]
			if all( sl is lt for sl, lt in zip( substituted_leaves, leaf_types )):
				return t
			return self.discovery._get_or_create_union( substituted_leaves )
		return t

	def substituted_field( self, found: Variable, owner_type: Type|None ) -> Variable:
		# a field declared using its owning generic class's own type params
		# (e.g. Result[T,E]'s synthesized `data: Result$data[T,E]`) is stored ONCE,
		# unsubstituted, on the class itself - accessing it through a
		# concrete Specialization (Result[Ptr[u8],OwnershipError]) must
		# substitute T/E with that Specialization's own args, or every
		# access sees the bare TypeVars regardless of which instantiation it
		# went through (this was invisible before match statements: nothing
		# previously read a generic field's type this way - checked-
		# arithmetic/or_return() consume a Result's payload via a dedicated
		# opcode on the whole Result value, never by synthesizing a literal
		# `.field` AST and lowering it)
		type_params = getattr( getattr( owner_type, 'base', None ), 'type_params', None )
		if not isinstance( owner_type, Specialization ) or not type_params:
			return found
		substituted_type = self.substitute_type_params( found.type, type_params, owner_type.args )
		if substituted_type is found.type:
			return found
		return replace( found, type = substituted_type ) # a shallow copy - `found` is the SAME shared Variable object for every access of this field, regardless of specialization, so this must not mutate it in place

	def monomorphized_function( self, spec: Specialization ) -> Function:
		# a distinct compiled unit per explicit generic instantiation
		# (sys.alloc[u8] vs sys.alloc[u32] are two separate functions, each
		# with T bound to a concrete type throughout - not a shared
		# unspecialized body the way a generic CLASS's methods stay today).
		# Built by copying the base Function with every generic-facing
		# field substituted: qualname (so FuncStart/Call get sys.alloc[u8],
		# not the shared sys.alloc), parameters/return_type (via
		# substitute_type_params), and names (T's own entry replaced with
		# the concrete arg, so ordinary name lookups - including
		# compiler.sizeof(T) - resolve it correctly while lowering fn.node.
		# body). Memoized on spec.monomorphized - this only ever builds one
		# copy per distinct instantiation.
		#
		# .node is deep-copied, not shared with base - every specialization
		# gets its own independent body. This is what lets type_resolver.py's
		# generic-call resolution (_ReferenceResolver.visit_Call) tag a
		# DIFFERENT node.resolved_callee per specialization when this
		# function's own body calls another generic function using its own
		# T (e.g. foo[T]'s body calling bar(t: T) - which concrete bar to
		# call depends on which T this copy was bound to, so the two
		# specializations of foo genuinely need independent bodies, not a
		# shared one interpreted two different ways - see resolve_function_
		# body's own docstring in type_resolver.py, which relies on this)
		if spec.monomorphized is not None:
			return spec.monomorphized
		base = spec.base
		if base.resolve is not None:
			base.resolve()
		type_params = base.type_params
		substituted_cls = base.cls
		if not type_params and base.cls is not None and base.cls.type_params:
			# base's own genericity is inherited from its enclosing generic
			# CLASS (Result.Ok/.Err/.is_ok/... referencing Result's own
			# T,E) rather than declared on the function itself (sys.
			# alloc[T]) - substitute against the class's type params
			# instead, and the method's own .cls must become the concrete
			# class specialization too (so e.g. an instance method's self
			# ends up typed as Result[i32,E], not the abstract Result -
			# see Lowering.lower_function's own self-synthesis, which reads
			# fn.cls directly)
			type_params = base.cls.type_params
			substituted_cls = self.discovery._get_or_create_specialization( base.cls, spec.args )
		type_params = type_params or []
		substituted_params = [
			replace( p, type = self.substitute_type_params( p.type, type_params, spec.args ) )
			for p in ( base.parameters or [] )
		]
		substituted_return = self.substitute_type_params( base.return_type, type_params, spec.args )
		substituted_names = dict( base.names )
		for tv, arg in zip( type_params, spec.args ):
			substituted_names[tv.stem] = arg
		for p in substituted_params:
			substituted_names[p.stem] = p
		monomorphized = replace(
			base,
			qualname = spec.qualname,
			cls = substituted_cls,
			parameters = substituted_params,
			return_type = substituted_return,
			names = substituted_names,
			node = copy.deepcopy( base.node ),
			type_params = None,
			resolve = None,
		)
		spec.monomorphized = monomorphized
		return monomorphized

	def _substituted_overload( self, group: Overload, spec: Specialization ) -> Overload:
		# an @overload group declared inside a generic class (e.g. Result
		# [T,E].unwrap_or's `default: T` stub) still carries the class's
		# own bare TypeVars on every member's .parameters, same as an
		# ordinary (non-overloaded) method would before monomorphized_
		# function gets a chance at it - substitute each implementation
		# through monomorphized_function, exactly like the plain-method
		# branch in monomorphize_class already does, just once per member
		# instead of a fresh, throwaway per-call-site copy every time this
		# group is ever dispatched. A member with its OWN additional type
		# params (independently generic beyond the class) is left
		# unsubstituted, same carve-out monomorphize_class's own plain-
		# method loop already applies - it's resolved through the ordinary
		# generic-call machinery when actually invoked, not here.
		def sub_impl( fn: Function ) -> Function:
			if fn.type_params:
				return fn
			method_spec = self.discovery._get_or_create_specialization( fn, spec.args )
			return self.monomorphized_function( method_spec )
		substituted_impls = [ sub_impl( fn ) for fn in group.implementations ]
		# a substituted STUB's own .bound_to (set by discovery.py's stub-
		# binding pass, pointing at the ABSTRACT implementation) has to be
		# re-pointed at the SUBSTITUTED implementation - _lower_call's own
		# winning_stub lookup matches by `s.bound_to is <the resolved
		# implementation>`, which only ever sees the substituted ones
		impl_by_original_id = { id( orig ): sub_fn for orig, sub_fn in zip( group.implementations, substituted_impls ) }

		def sub_stub( stub: Function ) -> Function:
			if stub.type_params:
				return stub
			# deliberately NOT routed through _get_or_create_specialization/
			# monomorphized_function's shared cache, unlike sub_impl above -
			# a stub and the plain implementation it binds to share the
			# EXACT SAME .qualname (discovery.py's _get_qualname has no
			# notion of "which overload candidate"; both are just called
			# `make`), so the cache key those two build would COLLIDE -
			# whichever of the two got monomorphized first would silently
			# be handed back for BOTH, cached under the other's spec too.
			# A stub is never independently scheduled/compiled anyway (no
			# real body, dispatch-only), so it doesn't need that shared,
			# by-qualname identity at all - substitute it directly instead
			if stub.resolve is not None:
				# populates .parameters/.return_type AND runs discovery.
				# py's own stub-binding (_bind_overload_stub) against the
				# ABSTRACT group, setting THIS stub's own .bound_to - relies
				# on group.implementations' own .resolve having already run
				# (see _bind_overload_stub's own "if impl.resolve is not
				# None: impl.resolve()"), which sub_impl above guarantees
				# by running first
				stub.resolve()
			cls_type_params = stub.cls.type_params if stub.cls is not None else None
			substituted_cls = (
				self.discovery._get_or_create_specialization( stub.cls, spec.args )
				if cls_type_params else stub.cls
			)
			type_params = cls_type_params or []
			substituted_params = [
				replace( p, type = self.substitute_type_params( p.type, type_params, spec.args ))
				for p in ( stub.parameters or [] )
			]
			substituted_return = self.substitute_type_params( stub.return_type, type_params, spec.args )
			bound_to = impl_by_original_id.get( id( stub.bound_to ), stub.bound_to ) if stub.bound_to is not None else None
			return replace(
				stub,
				cls = substituted_cls,
				parameters = substituted_params,
				return_type = substituted_return,
				node = copy.deepcopy( stub.node ),
				bound_to = bound_to,
				resolve = None,
			)
		substituted_stubs = [ sub_stub( fn ) for fn in group.stubs ]
		return replace( group, stubs = substituted_stubs, implementations = substituted_impls )

	def monomorphize_class( self, spec: Specialization ) -> ClassLike:
		# gives a concrete generic class specialization (Result[i32,
		# OverflowError]) a real, independent struct/union layout - a
		# shallow copy of the base class with .attributes' own type_params
		# substituted via the SAME substitute_type_params helper
		# monomorphized_function already uses. compiler.py's _lower calls
		# Lowering.monomorphize_class for every ClassLike-based Specialization
		# it schedules (see _enqueue), so the result lands directly in
		# compiler.cstructs/.cunions/.tagged_unions/.rcclasses - stage 3 (the
		# emitter) never has to independently rediscover/resynthesize a
		# concrete specialization itself, it just walks those lists like
		# any other compile unit.
		#
		# .names is built into ONE substituted dict, covering both fields AND
		# the class's own plain (non-Overload, no-own-type_params) methods -
		# .names must never disagree with .attributes about what a field's
		# own type is, which it silently did before this (only the TaggedUnion
		# 'data' field ever got a substituted .names entry; every other
		# field's .names entry stayed the stale, unsubstituted original,
		# pointing at the same shared Variable every OTHER specialization's
		# .names does too). A method is substituted via the exact same
		# monomorphized_function this uses for an explicit generic call -
		# just triggered here, once, memoized, instead of on demand per call
		# site. Overload groups and methods with their OWN additional type
		# params (beyond the class's) are left exactly as today (abstract,
		# unsubstituted) - Lowering._lower_call's own hand-rolled
		# substitution for the Overload branch already handles that
		# correctly and independently; folding overload-group substitution
		# in here too would need a substituted copy of each stub/impl, not
		# just one Function - a separate, bigger piece of work, out of scope
		# here.
		if spec.monomorphized is not None:
			return spec.monomorphized
		# marked for the duration of the build - see _building's own
		# docstring (__init__) for why: a method returning a Specialization
		# of this SAME spec (a generic class method that returns its own
		# enclosing class) would otherwise send substitute_type_params's
		# eager-monomorphize step straight back into monomorphize_class for
		# THIS spec, before .monomorphized is set, forever
		self._building.add( id( spec ))
		try:
			base = spec.base
			if base.resolve is not None:
				base.resolve()
			for attr in base.attributes:
				self._ensure_resolved( attr ) # each field's own .type is lazily resolved, separate from the class itself - same as Lowering._lower_allocate_fields's own identical resolve loop
			if isinstance( base, TaggedUnion ):
				# tag/data are synthesized lazily, the first time the union is
				# actually touched (UnionStorage.get) - trigger that BEFORE
				# snapshotting base.names below, or the snapshot misses 'tag'
				# entirely on a union that's never been constructed/matched
				# against yet (this specialization would be the first reference)
				self._union_storage.get( base )
			type_params = base.type_params or []
			substituted_attrs = [
				replace( attr, type = self.substitute_type_params( attr.type, type_params, spec.args ))
				for attr in base.attributes
			]
			substituted_names = dict( base.names )
			for attr in substituted_attrs:
				substituted_names[attr.stem] = attr
			for member in base.methods:
				if isinstance( member, Overload ):
					substituted_names[member.stem] = self._substituted_overload( member, spec )
					continue
				if not isinstance( member, Function ) or member.type_params:
					continue
				method_spec = self.discovery._get_or_create_specialization( member, spec.args )
				substituted_names[member.stem] = self.monomorphized_function( method_spec )

			extra: dict = {}
			if isinstance( base, TaggedUnion ):
				# base.names['tag']/['data'] (synthesized by UnionStorage.get,
				# already triggered above) are SHARED across every specialization
				# of a generic union - the abstract base's own payload_cls
				# carries bare TypeVar fields (v_Ok: T, v_Err: E), never a real
				# emittable C type. A plain replace() would leave THIS
				# specialization's own .names pointing at that same abstract,
				# TypeVar-typed object - give it its own substituted payload_cls
				# (own qualname, so it doesn't collide with the abstract's or a
				# sibling specialization's), scheduled here since (mirroring
				# UnionStorage.get's own identical comment on the abstract case)
				# nothing else would ever reach it on its own.
				tag_attr, data_attr, payload_cls, _tags = self._union_storage.get( base )
				substituted_payload_fields = [
					replace( f, type = self.substitute_type_params( f.type, type_params, spec.args ))
					for f in payload_cls.attributes
				]
				substituted_payload_cls = CUnion(
					stem = payload_cls.stem,
					qualname = f'{spec.qualname}$data',
					file = payload_cls.file,
					line = payload_cls.line,
					attributes = substituted_payload_fields,
					names = { f.stem: f for f in substituted_payload_fields },
				)
				self.schedule( substituted_payload_cls )
				substituted_data_attr = replace( data_attr, type = substituted_payload_cls )
				substituted_names['data'] = substituted_data_attr
			monomorphized = replace(
				base,
				qualname = spec.qualname,
				attributes = substituted_attrs,
				names = substituted_names,
				type_params = None,
				resolve = None,
				**extra,
			)
			if isinstance( base, TaggedUnion ):
				# the attrs-copy loop above (substituted_names[attr.stem] =
				# attr, for every base.attributes entry) just clobbered
				# union.names[member.stem] back to a plain, substituted
				# attribute Variable for every member - overwriting the real
				# per-member constructor Function base.names[member.stem]
				# ALREADY had (synthesized by the union_storage.get(base)
				# call above, e.g. builtins.Result.Ok(value: T) -> Result[T,E])
				# with e.g. `Ok: str` instead of a callable `Ok(value: str) ->
				# Result[str,MyError]`. A REAL bug, not just a refactor: found
				# via a genuine crash (AttributeError: 'Variable' object has
				# no attribute 'return_type') the moment anything looked up a
				# MONOMORPHIZED union's own member constructor directly
				# (Lowering._coerce_into_union, TODO.txt's own "opportunistic
				# union emission") - Result.Ok(x) written directly in user
				# source never hit this, since `Result.Ok` resolves through
				# the ABSTRACT class's own untouched .names, never through a
				# concrete specialization's copy. Rebuild each member's own
				# constructor here, substituted for the concrete attr/payload
				# types, the exact same way union_storage.py's own
				# build_member_constructor already builds the abstract
				# base's - fn_file/fn_line are base.file/base.line
				# (monomorphized.file/.line, unchanged by replace() above),
				# always real: unlike an anonymous X|Y union, base here is a
				# real, user-declared `@union class Foo[T,E]:`, which always
				# has a genuine file.
				ctor_return_type = monomorphized
				for tag_value, attr in enumerate( substituted_attrs ):
					monomorphized.names[attr.stem] = build_member_constructor(
						monomorphized, attr, tag_value, tag_attr, substituted_data_attr, substituted_payload_cls, ctor_return_type,
						monomorphized.file, monomorphized.line,
					)
		finally:
			self._building.discard( id( spec ))
		spec.monomorphized = monomorphized
		self._origins[ id( monomorphized )] = spec
		return monomorphized

	def origin_of( self, t: object ) -> 'Specialization|None':
		return self._origins.get( id( t ))
