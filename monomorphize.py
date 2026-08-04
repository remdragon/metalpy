# stdlib imports:
from dataclasses import replace
from typing import Callable

# local imports:
from discovery import Discovery
from mpy_types import Type, TypeVar, Specialization, TaggedUnion, CUnion, ClassLike, Function, Variable
from union_storage import UnionStorage

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

	def __init__( self, discovery: Discovery, schedule: Callable[[object],None], union_storage: UnionStorage ) -> None:
		self.discovery = discovery
		self.schedule = schedule
		self._union_storage = union_storage
		# monomorphized Function copies (T substituted with a concrete
		# type), memoized by id(Specialization) - discovery._get_or_create_
		# specialization already dedupes the Specialization itself by its
		# qualname key, so every call to the same instantiation (sys.
		# alloc[u8], from anywhere) reuses the SAME monomorphized Function
		# object, not a fresh copy per call site
		self._monomorphized: dict[int,Function] = {}
		# monomorphized ClassLike copies (a generic class's OWN .attributes
		# with type_params substituted, for a concrete Specialization like
		# Result[i32,OverflowError]) - memoized by id(Specialization), same
		# discipline as _monomorphized above. This is what actually gives a
		# concrete generic specialization a real compile unit/output-list
		# entry (compiler.py's _lower dispatches a ClassLike-based
		# Specialization here) - stage 3 (the emitter) never has to
		# independently rediscover/synthesize one, it just walks
		# compiler.cstructs/.cunions/.tagged_unions/.rcclasses like anything else
		self._monomorphized_classes: dict[int,ClassLike] = {}

	def _ensure_resolved( self, obj: object ) -> None:
		# same discipline as Lowering._ensure_resolved - duplicated here
		# rather than depending back on Lowering, since this class is meant
		# to be usable standalone
		resolve = getattr( obj, 'resolve', None )
		if resolve is not None:
			resolve()
		self.schedule( obj )

	def substitute_type_params( self, t: Type|None, type_params: list[TypeVar], args: list[Type] ) -> Type|None:
		if isinstance( t, TypeVar ):
			for param, arg in zip( type_params, args ):
				if t is param:
					return arg
			return t
		if isinstance( t, Specialization ):
			substituted_args = [ self.substitute_type_params( a, type_params, args ) for a in t.args ]
			if all( sa is a for sa, a in zip( substituted_args, t.args )):
				return t
			return self.discovery._get_or_create_specialization( t.base, substituted_args )
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
		# body, which is otherwise untouched/shared AST). Memoized by
		# id(spec) - discovery._get_or_create_specialization already
		# dedupes the Specialization itself, so this only ever builds one
		# copy per distinct instantiation
		cached = self._monomorphized.get( id( spec ) )
		if cached is not None:
			return cached
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
			type_params = None,
			resolve = None,
		)
		self._monomorphized[ id( spec ) ] = monomorphized
		return monomorphized

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
		# any other compile unit. .methods/.names are copied through
		# UNCHANGED (still referencing the class's abstract, un-monomorphized
		# Function objects) - method lookup keeps working exactly as today
		# (Specialization.names passes through to .base.names), and each
		# individual method call gets its OWN on-demand monomorphization via
		# monomorphized_function/Lowering._lower_class_generic_method_call, not
		# eagerly here.
		cached = self._monomorphized_classes.get( id( spec ) )
		if cached is not None:
			return cached
		base = spec.base
		if base.resolve is not None:
			base.resolve()
		for attr in base.attributes:
			self._ensure_resolved( attr ) # each field's own .type is lazily resolved, separate from the class itself - same as Lowering._lower_allocate_fields's own identical resolve loop
		type_params = base.type_params or []
		substituted_attrs = [
			replace( attr, type = self.substitute_type_params( attr.type, type_params, spec.args ))
			for attr in base.attributes
		]
		extra: dict = {}
		if isinstance( base, TaggedUnion ):
			# base.names['tag']/['data'] (synthesized by UnionStorage.get)
			# are SHARED across every specialization of a generic union - the
			# abstract base's own payload_cls carries bare TypeVar fields
			# (v_Ok: T, v_Err: E), never a real emittable C type. A plain
			# replace() would leave THIS specialization's own .names pointing
			# at that same abstract, TypeVar-typed object - give it its own
			# substituted payload_cls (own qualname, so it doesn't collide
			# with the abstract's or a sibling specialization's), scheduled
			# here since (mirroring UnionStorage.get's own identical
			# comment on the abstract case) nothing else would ever reach it
			# on its own.
			_tag_attr, data_attr, payload_cls, _tags = self._union_storage.get( base )
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
			extra['names'] = dict( base.names )
			extra['names']['data'] = replace( data_attr, type = substituted_payload_cls )
		monomorphized = replace(
			base,
			qualname = spec.qualname,
			attributes = substituted_attrs,
			type_params = None,
			resolve = None,
			**extra,
		)
		self._monomorphized_classes[ id( spec ) ] = monomorphized
		return monomorphized
