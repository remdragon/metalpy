# stdlib imports:
import ast
import copy
from typing import Iterable, NoReturn

# local imports:
import ir
from discovery import is_stub_body
from errors import CompileError
from mpy_types import (
	Name, Type, Variable, Parameter, Function, Overload, ClassLike, Specialization, TaggedUnion, CStruct, CUnion, CEnum, TypeVar, ConditionalDispatch, RCClass, Scalar, int_stem_range, InheritanceChainMixin,
)
import overload_resolution
from type_resolver import TypeResolver

from lowering_shared import _MODE_DUNDER_PREFIX

class ConstructLoweringMixin:
	''' object allocation and construction call lowering (`allocate()`/constructor calls) - mixed into FunctionLowering (lowering.py), which
	see for the shared instance state (self._instructions, self._cfg, self.lowering,
	etc.) every method here reads and writes. Never instantiated on its own;
	split out of lowering.py purely to keep that file to a manageable size - see
	lowering.py's own class docstring and FunctionLowering's base-class list for
	the full set of sibling mixins this one is composed with. '''


	def _construct_generic_instance( self, target_cls: Type, node: ast.AST ) -> ir.Operand:
		''' construct a zero-argument instance of an already-fully-resolved
		class/generic Specialization (target_cls's own type args, if any,
		are already concrete) - used by _expr_List to build the backing
		list[T] instance a list-literal populates via append(). Deliberately
		narrower than _try_lower_construct_call (this file, the general
		ClassName(...) sugar): no fresh AST Call node naming the class is
		synthesized here (that would never have passed through type_
		resolver.py's own pre-pass the way a real call site does, and would
		need its own textual type-argument spelling for an arbitrary
		target_cls) - target_cls is already the concrete type we want, so
		this goes straight to ordinary (non-generic-inference) construction,
		using a synthetic zero-arg Call node purely as the argument-list
		shape _lower_call_args/_match_call_args need (never inspected for
		its own .func) - real default-value expressions (e.g. list[T]'s own
		initial_capacity: usize = 8) are already real AST nodes on the
		Function's own Parameter objects, nothing to fabricate there. Only
		supports a target whose __init__ is present, non-overloaded, and
		non-fallible - list[T]'s own shape; a different caller needing more
		would extend this, not work around it. '''
		resolved_cls = self.lowering._ensure_resolved( target_cls )
		assert isinstance( resolved_cls, ClassLike ), f'internal compiler error: {resolved_cls} is not constructible'
		init = resolved_cls.get_local_or_raise( '__init__' )
		if isinstance( init, Overload ):
			# an overloaded __init__ (e.g. list[T]'s own capacity/iterator/
			# iterable overloads) - this call site only ever constructs
			# ZERO-argument (see this method's own docstring), so resolve_
			# call naturally narrows to whichever ONE candidate accepts no
			# arguments at all (every other candidate's own sole parameter
			# has no default, so _translate_indices excludes it outright) -
			# no move()/kwargs/literal-typing concern here, unlike the
			# general ClassName(...) construction path, since there are
			# never any real arguments to lower in the first place
			try:
				branches, init = overload_resolution.resolve_call(
					init.stubs, init.implementations, [], {},
					qualname = f'{resolved_cls.qualname}.__init__', same_type = self.lowering._type_resolver._same_type,
				)
			except CompileError as e:
				self.lowering.discovery.fail( str( e ), node )
			assert not branches, f'internal compiler error: {resolved_cls.qualname}.__init__() (zero-arg) resolved to a runtime dispatch, not a single candidate'
		assert isinstance( init, Function ), f'internal compiler error: {resolved_cls.qualname} has no usable __init__'
		self.lowering.schedule( resolved_cls )
		self.lowering._ensure_resolved( init )
		synth_call = ast.Call( func = node, args = [], keywords = [] )
		ast.copy_location( synth_call, node )
		args, kwargs = self._lower_call_args( init, synth_call )
		self_temp = self._new_temp( resolved_cls )
		self.lowering._schedule_rcclass_construction( resolved_cls, self_temp.type )
		self._emit( ir.Allocate( dest = self_temp, cls = resolved_cls, fields = {} ))
		self.lowering.schedule( init.return_type )
		for param in init.parameters or []:
			self.lowering.schedule( param.type )
		assert not self.lowering._init_fallibility( init ), f'internal compiler error: {resolved_cls.qualname}.__init__ is fallible'
		self._emit( ir.Call( dest = None, target = init, receiver = self_temp, args = args, kwargs = kwargs ))
		return self_temp

	def _check_rcclass_fully_implemented( self, target_cls: RCClass, node: ast.AST, label: str ) -> None:
		''' RCClass analog of the CStruct-interface stub-body check just
		below (RCClass-subclassing plan Phase 5) - an RCClass with any
		unfulfilled @abstractmethod slot anywhere in its own chain is
		never meant to be constructed directly. Unlike CStruct's implicit
		stub-body-means-unimplemented convention, RCClass uses the
		EXPLICIT is_abstract marker (discovery.py already requires
		@abstractmethod to also be @virtual and have a stub body - so
		checking is_abstract here is equivalent to checking the body
		shape, just reads as what it actually means). Shared by BOTH
		RCClass construction paths (_lower_allocate_fields's own call
		below, and _try_lower_construct_call's __init__-based path) via
		this one helper, so the two can't drift out of sync - matches
		compiler.py's own _schedule_rcclass_vtable_impls, which this stays
		in lockstep with (emitter_c.py's emit_rcclass_vtable_instance
		skips building an instance at all for a class this check would
		reject, the same "None if any slot is unfulfilled" gate CStruct's
		own emit_interface_vtable_instance already uses). '''
		unfulfilled: list[str] = []
		for slot in target_cls.virtual_slots():
			impl = target_cls.chain_lookup( slot.stem )
			assert isinstance( impl, Function ) # virtual_slots()'s own entries always exist somewhere in the chain - at minimum the root's own declaration chain_lookup started from
			if impl.resolve is not None:
				impl.resolve()
			if impl.is_abstract:
				unfulfilled.append( slot.stem )
		if unfulfilled:
			self.lowering.discovery.fail(
				f'{target_cls.qualname}{label} cannot be constructed - abstract method(s) have no implementation: '
				f'{", ".join(unfulfilled)}',
				node,
			)

	def _infer_allocate_type_args(
		self, target_cls: ClassLike, class_type_params: list[TypeVar],
		fields_to_build: dict[str,Variable], fields: dict[str,ir.Operand], node: ast.Call, label: str,
	) -> Specialization:
		# _lower_allocate_fields's no-__init__ field=value sugar has no
		# _lower_generic_construction_args-style argument-based inference of
		# its own (that path infers a generic RCClass's own type args from
		# __init__'s parameters, independent of expected_type) - called only
		# once expected_type has already been ruled out as usable (absent, or
		# - PLAN_RETURN_INFERENCE.md's eager-lowering passes, whose own
		# return_type is deliberately still None while lowering their body -
		# untrustworthy/incompatible, same "rooted at target_cls" discipline
		# the sibling `compatible` check just above this call's own use
		# applies). The fields' own real, already-lowered VALUES
		# (fields[name].type) carry exactly the concrete types needed -
		# unify each field's declared (possibly bare-TypeVar) type against
		# its own real value type, same _unify_type_param every OTHER
		# generic call site already uses. Total/safe on shapes it doesn't
		# recognize (e.g. a TaggedUnion's synthesized tag/data view isn't
		# built from target_cls's own type params in any directly-unifiable
		# way) - it just no-ops rather than crashing, so this applies
		# uniformly across RCClass/CStruct/CUnion/TaggedUnion without
		# needing to special-case any of them out
		bindings: dict[int,Type] = {}
		for name, field in fields_to_build.items():
			self.lowering._unify_type_param( class_type_params, field.type, fields[name].type, bindings, node, target_cls.qualname )
		missing_type_params = [ tv.stem for tv in class_type_params if id( tv ) not in bindings ]
		if missing_type_params:
			self.lowering.discovery.fail(
				f'{target_cls.qualname}{label}: cannot infer type parameter(s) {", ".join(missing_type_params)} '
				f'from these field values or the surrounding expected type: {ast.unparse(node)}',
				node,
			)
		concrete_args = [ bindings[id(tv)] for tv in class_type_params ]
		return self.lowering.discovery._get_or_create_specialization( target_cls, concrete_args )

	def _lower_allocate_fields( self, target_cls: ClassLike, node: ast.Call, expected_type: Type|None, label: str ) -> ir.Temp:
		# shared by both callers of ir.Allocate (Class.__allocate__(...) and
		# bare ClassName(...) sugar for the no-__init__ case) - everything
		# past "which class, and is this call form even allowed here" is
		# identical field-matching/emission logic. `label` is just how the
		# call reads in error messages (".__allocate__(...)" vs "(...)"), so
		# existing callers' error text doesn't change.
		if node.args:
			self.lowering.discovery.fail( f'{target_cls.qualname}{label} takes keyword arguments only: {ast.unparse(node)}', node )
		if any( kw.arg is None for kw in node.keywords ):
			self.lowering.discovery.fail( f'**kwargs not supported for {target_cls.qualname}{label}: {ast.unparse(node)}', node )

		self.lowering._ensure_resolved( target_cls )
		if isinstance( target_cls, ( RCClass, CStruct )):
			# flattened_attributes() (below) resolves NOTHING it returns
			# (see its own docstring) - an ancestor whose own body hasn't
			# been resolved yet contributes ZERO fields to the flattened
			# list instead of erroring, so a base's own fields silently
			# vanish from a subclass's no-__init__ construction sugar
			# unless every ancestor is resolved first. _ensure_resolved
			# above only resolves target_cls ITSELF, not its base chain.
			target_cls.resolve_chain()
		# .attributes alone only ever holds a class's OWN declared fields
		# (discovery.py never merges a base's own fields into a subclass) -
		# flattened_attributes() walks the WHOLE single-inheritance chain
		# (base-first), which is what this no-__init__ field=value sugar
		# needs to see every constructible field, inherited or not. A no-op
		# widening for RCClass/CStruct with no base (returns the same list
		# .attributes would) and for CUnion/CEnum (never have .base at all)
		target_fields = target_cls.flattened_attributes() if isinstance( target_cls, ( RCClass, CStruct )) else target_cls.attributes
		for attr in target_fields:
			self.lowering._ensure_resolved( attr ) # each field's own .type is lazily resolved, separate from the class itself - same as _attr_lookup's found.resolve
		if isinstance( target_cls, CStruct ) and target_cls.is_interface:
			# a "pure interface" (or any @interface class with an unfulfilled
			# @virtual slot anywhere in its chain - a stub body, same shape
			# @overload stubs use) is never meant to be constructed directly -
			# nothing else in this compiler enforces that (there's no separate
			# @abstract marker - see PLAN_SUBCLASSING_VTABLES_COM.md's own
			# "Unimplemented @virtual methods" reasoning), so it's checked
			# here, at the one place a real CStruct value actually gets built
			unfulfilled: list[str] = []
			for slot in target_cls.virtual_slots():
				impl = target_cls.chain_lookup( slot.stem )
				assert isinstance( impl, Function ) # virtual_slots()'s own entries always exist somewhere in the chain - at minimum the root's own declaration chain_lookup started from
				if impl.resolve is not None:
					impl.resolve()
				if is_stub_body( impl.node.body ):
					unfulfilled.append( slot.stem )
			if unfulfilled:
				self.lowering.discovery.fail(
					f'{target_cls.qualname}{label} cannot be constructed - virtual method(s) have no implementation: '
					f'{", ".join(unfulfilled)}',
					node,
				)
		elif isinstance( target_cls, RCClass ):
			self._check_rcclass_fully_implemented( target_cls, node, label )
		# target_cls is always the ABSTRACT class (resolved via
		# _try_resolve_namespace on the shared, unspecialized AST body's
		# own `SomeGeneric.__allocate__` reference - see
		# _try_lower_allocate_call) even from inside a monomorphized
		# generic-class method, whose self._current_fn.cls IS the concrete
		# specialization - substitute target_cls's own field types against
		# that concrete specialization when one's available, same as
		# _substituted_field already does for ordinary attribute reads
		# (_expr_Attribute), or a generic field's expected type here would
		# stay abstract (its own bare TypeVars) forever
		fn_cls = self._current_fn.cls if self._current_fn is not None else None
		if isinstance( target_cls, TaggedUnion ):
			# a TaggedUnion's REAL storage shape is tag+data (synthesized by
			# UnionStorage.get(), registered in .names) - .attributes is the
			# LOGICAL member list (Ok/Err), a completely different thing
			# (used for .leaves()/type-matching, never for real field
			# layout). .__allocate__(tag=.., data=..) - the shape the
			# synthesized per-member constructor's own body uses (see
			# union_storage.py's _build_member_constructor) - must validate
			# against tag/data, not the logical members
			if isinstance( fn_cls, Specialization ) and fn_cls.base is target_cls:
				# target_cls itself is always the ABSTRACT union (see the
				# comment above on why bodies always reference it that way),
				# so target_cls.names['data'].type would stay the ABSTRACT,
				# unsubstituted payload_cls forever - substitute_field can't
				# help here (a payload_cls is a plain CUnion, not a TypeVar/
				# Specialization/anonymous-union shape it knows how to
				# rebuild), so read tag/data from the MONOMORPHIZED copy
				# instead (_ensure_resolved(fn_cls) is exactly monomorphize_
				# class - see its own "fresh payload_cls per specialization"
				# comment for why that copy's own data field is already
				# correctly substituted)
				concrete_union = self.lowering._ensure_resolved( fn_cls )
				tag_field = concrete_union.get_local_or_raise( 'tag' )
				data_field = concrete_union.get_local_or_raise( 'data' )
			else:
				tag_field = target_cls.get_local_or_raise( 'tag' )
				data_field = target_cls.get_local_or_raise( 'data' )
			assert isinstance( tag_field, Variable ) and isinstance( data_field, Variable ), \
				f'{target_cls.qualname}: UnionStorage.get() has not run yet - no real tag/data storage to allocate'
			declared = { tag_field.stem: tag_field, data_field.stem: data_field }
		elif isinstance( fn_cls, Specialization ) and fn_cls.base is target_cls:
			declared = { attr.stem: self.lowering._substituted_field( attr, fn_cls ) for attr in target_fields }
		else:
			declared = { attr.stem: attr for attr in target_fields }
		given = { kw.arg for kw in node.keywords }
		missing = declared.keys() - given
		if isinstance( target_cls, CUnion ):
			# a union's whole point - only ONE member is ever meaningfully
			# set at a time (see UnionStorage.get's identical comment
			# on the synthesized TaggedUnion payload CUnion) - "every OTHER
			# field is missing" isn't an error here the way it is for an
			# ordinary struct/class, unlike ResultPayload(ok=val) never
			# giving err. Pre-existing gap, confirmed unrelated to this
			# pass (reproduces on a clean checkout: Result.Ok(...)/
			# Result.Err(...)'s own bodies were never actually exercised
			# through a full Compiler.run() before, so this went unnoticed)
			if len( given ) != 1:
				self.lowering.discovery.fail( f'{target_cls.qualname}{label} takes exactly one field (only one union member is ever set): {ast.unparse(node)}', node )
		else:
			truly_missing = sorted( name for name in missing if declared[name].init is None )
			if truly_missing:
				self.lowering.discovery.fail( f'{target_cls.qualname}{label} is missing field(s): {", ".join(truly_missing)}', node )
		extra = given - declared.keys()
		if extra:
			self.lowering.discovery.fail( f'{target_cls.qualname}{label} has no field(s): {", ".join(sorted(extra))}', node )

		given_by_name = { kw.arg: kw.value for kw in node.keywords }
		# a CUnion only ever builds the ONE given member - the other
		# declared fields aren't "defaulted", they're simply not part of
		# this particular construction at all (unlike an ordinary struct/
		# class, where every field always exists)
		fields_to_build = { name: declared[name] for name in given_by_name } if isinstance( target_cls, CUnion ) else declared
		fields: dict[str,ir.Operand] = {}
		for name, field in fields_to_build.items():
			if name in given_by_name:
				expr = given_by_name[name]
				field_base = field.type.base if isinstance( field.type, Specialization ) else field.type
				if getattr( expr, 'generator_zero_rc_field', False ) and isinstance( field_base, TaggedUnion ):
					# _expr_Constant's own generator_zero_rc_field
					# exemption only ever helps a plain scalar/RCClass
					# target - a TaggedUnion is unconditionally exempted
					# from THAT check regardless of the tag (see its own
					# comment), so a bare `0` literal here would instead
					# fall through to ordinary union-coercion, which
					# genuinely fails whenever NEITHER leaf happens to be
					# a plain int (e.g. Result[Box,IndexError] - no leaf
					# a bare int literal can ever match). See
					# _build_generator_zero_value's own docstring for why
					# a real leaf-wrap constructor call isn't safe here
					# either (would incref a NULL placeholder).
					value = self._build_generator_zero_value( field.type, expr )
				else:
					value = self._lower_expr( expr, field.type )
			else:
				# omitted at the call site, but declared with a default
				# (`field.init`, already confirmed not None by truly_missing
				# above) - lowered in the CLASS's own scope, not the caller's,
				# matching ordinary Python class-body scoping (a default
				# expression can reference other class-level names, but not
				# anything local to whoever's constructing this instance)
				expr = field.init
				with self.lowering.discovery.module_context( self.lowering._find_module_for( target_cls )):
					with self.lowering.discovery.scope_context( target_cls ):
						value = self._lower_expr( expr, field.type )
			# value.type, not field.type: field.type is the FIELD's declared
			# type, which stays an unsubstituted TypeVar for any field whose
			# type depends on a class's own type params (class methods are
			# never monomorphized per-specialization - see compiler.py's
			# _enqueue) - value.type is always the operand's real, concrete
			# type regardless, since only concrete values ever actually get
			# lowered
			for instr in self._cfg.field_value( value.type, value, is_alias = self.lowering._is_aliasing_expr( expr, value )):
				self._emit( instr )
			fields[name] = value

		if isinstance( target_cls, CStruct ) and target_cls.is_interface:
			# an @interface CStruct is never a plain value type (see
			# PLAN_SUBCLASSING_VTABLES_COM.md) - construction heap-allocates
			# (like RCClass's own ClassName(...), via the same real
			# sys.alloc[T] path - see _schedule_interface_construction) and
			# produces a Ptr[T], not a bare T. Unlike RCClass, this is an
			# EXPLICIT Ptr[T] in the metalpy type system (not an invisible-
			# pointer convention) - self is ALSO always Ptr[T] for the same
			# reason (see lower_function's own self_param construction)
			ptr_cls = self.lowering.discovery.get_intrinsics()['Ptr']
			interface_class_type_params = target_cls.type_params or []
			fallback_cls: ClassLike|Specialization = target_cls
			if expected_type is None and interface_class_type_params:
				fallback_cls = self._infer_allocate_type_args( target_cls, interface_class_type_params, fields_to_build, fields, node, label )
			ptr_type = self.lowering.discovery._get_or_create_specialization( ptr_cls, [ fallback_cls ] )
			dest = self._new_temp( expected_type or ptr_type )
			self.lowering.schedule( dest.type )
			self.lowering._schedule_interface_construction( target_cls )
			self._emit( ir.Allocate( dest = dest, cls = target_cls, fields = fields ))
			return dest

		# expected_type is only trustworthy as dest's type when it's actually
		# ROOTED AT target_cls (itself, a Specialization of it, or - inside a
		# generic class's own method body, e.g. this exact __allocate__ call
		# from Result's own synthesized Err/Ok constructor - the MONOMORPHIZED
		# concrete class fn_cls's own Specialization already substitutes to;
		# substitute_type_params eagerly monomorphizes a ClassLike-shaped
		# type param instead of leaving it wrapped in a Specialization - see
		# monomorphize_class's own "_building" comment - so a generic method's
		# `self._current_fn.return_type` surfaces here already as that bare,
		# concrete ClassLike, not a Specialization wrapping target_cls, even
		# though fn_cls (this same call's own substituted class context,
		# already used just above to resolve tag_field/data_field/declared)
		# IS a Specialization of target_cls). Otherwise, expected_type can be
		# hinted from an unrelated surrounding union context (a generic
		# method call's own type-param unification substituting E in
		# `Result.Err(ParseError(...))` to the inferred `ParseError|OtherError`
		# union BEFORE this argument is even lowered - see
		# _lower_and_infer_call_args). Blindly trusting it there produced a
		# self-inconsistent ir.Allocate (cls=ParseError but dest.type=the
		# union), silently bypassing _lower_expr's own union-coercion check
		# (operand.type is expected_type by accidental identity) and crashing
		# the emitter's isinstance(concrete_cls, RCClass) assert. Mirrors the
		# same "pinning_type.base is target_cls" discipline
		# _lower_generic_construction_args already applies before trusting
		# expected_type for type-param inference.
		compatible = ( expected_type is target_cls
			or ( isinstance( expected_type, Specialization ) and expected_type.base is target_cls )
			or ( isinstance( fn_cls, Specialization ) and fn_cls.base is target_cls
				and expected_type is self.lowering.monomorphize_class( fn_cls ) ))
		# when expected_type isn't usable, falling back to the bare ABSTRACT
		# target_cls (unspecialized type_params and all) is only actually
		# correct when target_cls isn't generic in the first place - see
		# _infer_allocate_type_args. TaggedUnion is included here even though
		# _infer_allocate_type_args can never actually bind anything through
		# it (this branch's own `declared` is always {tag, data} - tag_field.
		# type is a plain intrinsic and data_field.type is a synthesized
		# anonymous CUnion whose own type_params is never set by union_
		# storage.py's UnionStorage.get(), so neither shape is one
		# _unify_type_param can recurse through). That's a harmless no-op
		# today, not a live gap: every REAL generic-union construction goes
		# through union_storage.py's synthesized per-member constructor,
		# whose own fn_cls/expected_type relationship always satisfies
		# `compatible` above BEFORE inference would ever run. If some future
		# path ever did reach here uncompatible, _infer_allocate_type_args's
		# own "cannot infer type parameter(s)" fits this function's
		# established fail-loudly-not-silently-wrong discipline - which is
		# exactly why TaggedUnion stays in this isinstance check rather than
		# being carved out back to the bare-abstract-class fallback
		class_type_params: list[TypeVar] = (
			target_cls.type_params or []
		) if isinstance( target_cls, ( RCClass, CStruct, CUnion, TaggedUnion )) else []
		fallback_cls: ClassLike|Specialization = target_cls
		if not compatible and class_type_params:
			fallback_cls = self._infer_allocate_type_args( target_cls, class_type_params, fields_to_build, fields, node, label )
		dest = self._new_temp( expected_type if compatible else fallback_cls )
		# dest.type can be a concrete Specialization (ResultPayload[i32,
		# OverflowError], inferred from the substituted field type this
		# construction call is being assigned into - see the field.type
		# comment above) even though target_cls itself (this call's own
		# bare `ResultPayload` reference) is always the abstract base - the
		# concrete Specialization needs its own explicit schedule() here,
		# same as _emit_generic_call already does for a generic FUNCTION's
		# own monomorphized return type; nothing else would ever schedule it
		self.lowering.schedule( dest.type )
		if isinstance( target_cls, RCClass ):
			self.lowering._schedule_rcclass_construction( target_cls, dest.type )
		self._emit( ir.Allocate( dest = dest, cls = target_cls, fields = fields ))
		return dest

	def _is_real_field_receiver( self, owner_type: Type|None ) -> bool:
		''' PLAN_THREAD_SAFE_SHARED_STATE.md Part B locking gate: true only
		when owner_type is a genuine, heap-allocated RCClass instance - the
		only kind of receiver that HAS a $header (and therefore a lock) at
		all. Deliberately narrower than _check_field_visibility's own
		InheritanceChainMixin check (RCClass OR CStruct): a CStruct is
		always either embedded BY VALUE inside its own containing object (no
		separate identity/pointer of its own - protected by the CONTAINING
		RCClass's lock, not one of its own) or, for an @interface CStruct,
		heap-allocated WITHOUT an ObjectHeader at all (see emit_c's own
		ir.Allocate codegen: "NO ObjectHeader/refcount init" for that case) -
		neither shape has a $header.lock to acquire. Confirmed as a real bug,
		not a theoretical one: a nested CStruct field's own teardown
		(compiler.decref recursing into a by-value-embedded CStruct's own
		attributes, type_resolver.py's _build_field_teardown_ast) tried to
		reinterpret-cast a whole `struct Wrapper` VALUE (not a pointer) to
		ObjectHeader*, rejected outright by clang. `.tag`/`.data`/
		`.v_<member>` (TaggedUnion's own compiler-SYNTHESIZED storage-view
		accessors - UnionStorage.get()) are excluded for the same underlying
		reason (no ObjectHeader of their own either) and were the first
		confirmed instance of this class of bug. '''
		resolved = owner_type.base if isinstance( owner_type, Specialization ) else owner_type
		return isinstance( resolved, RCClass )

	def _check_field_visibility( self, owner_type: Type|None, attr_var: Variable, attr_name: str, ctx: ast.AST ) -> None:
		''' PLAN_THREAD_SAFE_SHARED_STATE.md's field-visibility-enforcement
		prerequisite: called from every genuine user-facing `obj.field`
		read/write chokepoint (never from an internal synthesized-name
		lookup like a tuple element's `_N` field or a narrowing probe -
		those go through _attr_lookup directly, bypassing this). Resolves
		which class actually DECLARES attr_name (InheritanceChainMixin.
		field_owner - not necessarily owner_type itself, which may be a
		subclass reached through the concrete receiver) and defers the
		actual `_`/`__` check to discovery.check_field_visibility, the same
		accessing-class source of truth (_current_fn.cls) _try_lower_
		allocate_call's own private-access check just below already uses.
		A safe no-op for anything that isn't an RCClass/CStruct field at all
		(CUnion/TaggedUnion/CEnum members, Ptr pointees, ...) - none of
		those have SYNTAX.md's field-privacy concept in the first place. '''
		# unwrap to the abstract TEMPLATE, not _ensure_resolved's monomorphized
		# concrete instantiation - field_owner/in_private_scope/in_protected_
		# scope all compare by object IDENTITY, and _ensure_resolved hands
		# back a fresh, per-instantiation RCClass object for a generic class
		# (confirmed via a real false positive: monomorphized `builtins.
		# list[i32]` methods rejected as unable to access their OWN class's
		# `__inner`/`__lock` fields, because the receiver's and the accessing
		# method's own .cls each independently monomorphized to a
		# DIFFERENT-but-equal-looking object) - the abstract template is the
		# one stable object every instantiation shares, exactly the
		# "whichever template self/scope are each an instance of" contract
		# in_private_scope's own docstring already documents for its `scope`
		# side (see _try_lower_allocate_call's identical target_cls unwrap
		# just below, which resolves this the same way for __allocate__).
		resolved = owner_type.base if isinstance( owner_type, Specialization ) else owner_type
		if not isinstance( resolved, InheritanceChainMixin ):
			return
		defining_cls = resolved.field_owner( attr_name )
		if defining_cls is None:
			return
		if self._current_fn is not None and self._current_fn.is_destructor:
			# $$__destructor__ (type_resolver.py's _synthesize_rcclass_
			# destructor) is built with cls=None (confirmed directly: its own
			# Function(...) construction passes cls=None explicitly) even
			# though it's logically "inside" its own class - its whole job is
			# decref'ing every field, public or private, so it's exempt by
			# construction rather than something accessing_cls=None should
			# reject. Every other compiler-synthesized field toucher
			# (construction/$$__new__) goes through ir.Allocate, never
			# GetAttr/SetAttr, so this is the only synthesized-function case
			# that reaches this check at all.
			return
		accessing_cls = self._current_fn.cls if self._current_fn is not None else None
		self.lowering.discovery.check_field_visibility( attr_var, defining_cls, ctx, accessing_cls )

	def _try_lower_allocate_call( self, node: ast.Call, expected_type: Type|None ) -> ir.Temp|None:
		# Class.__allocate__(field=value, ...) - a compiler-synthesized
		# pseudo-method (SYNTAX.md: "strictly private... can only be called
		# from methods inside the same class"), not a real declared method,
		# so it can never be found via the ordinary _resolve_callee/
		# _attr_lookup_callable path (it's never in any class's .names) -
		# recognized textually here instead, same spirit as defer/errdefer
		# and the arithmetic-mode with-blocks. Returns None (not an error)
		# when this doesn't look like a __allocate__ call at all, so the
		# caller falls through to the normal call path and reports whatever
		# error is actually appropriate (e.g. "not callable")
		if not ( isinstance( node.func, ast.Attribute ) and node.func.attr == '__allocate__' ):
			return None
		target_cls = self.lowering._try_resolve_namespace( node.func.value )
		# explicit generic-class subscript - ClassName[T].__allocate__(...),
		# including a generic class's own method re-applying its OWN type
		# parameter to itself (the bare ClassName.__allocate__(...) spelling
		# already resolves this correctly via in-scope T lookup - this is
		# the same call, just reached through an explicit, redundant [T]).
		# Without this, _try_resolve_namespace's own Subscript branch
		# already correctly resolves ClassName[T] to a Specialization, but
		# this recognizer only accepted a real ClassLike, declining
		# (returning None) and falling through to ordinary receiver-based
		# call resolution - which lowers ClassName[T] as a VALUE expression
		# instead of a type reference, and ClassName (a class, not a
		# Variable/Function) fails there with "'ClassName' is not a value,
		# cannot use it as an expression".
		# Unwrap to .base (the abstract template) rather than monomorphizing
		# to the concrete specialization - _lower_allocate_fields's own
		# comment just below documents that target_cls must always be the
		# ABSTRACT class (it separately substitutes field types against
		# self._current_fn.cls, the concrete specialization, when one's
		# available); handing it an already-monomorphized target_cls here
		# instead breaks that substitution AND in_private_scope's identical
		# "self is always the abstract template" contract just below
		if isinstance( target_cls, Specialization ):
			target_cls = target_cls.base
		if not isinstance( target_cls, ClassLike ):
			return None

		fn = self._current_fn
		if fn is None or not target_cls.in_private_scope( fn.cls ):
			self.lowering.discovery.fail(
				f'{target_cls.qualname}.__allocate__(...) is private - only callable from a method of {target_cls.qualname} itself',
				node,
			)
		return self._lower_allocate_fields( target_cls, node, expected_type, '.__allocate__(...)' )

	def _try_lower_construct_call( self, node: ast.Call, expected_type: Type|None ) -> ir.Operand|None:
		# bare ClassName(...) - SYNTAX.md sugar. A class with no __init__ at
		# all degrades to exactly __allocate__ (field=value sugar). A class
		# WITH __init__: allocate self uninitialized, call __init__ with
		# the call site's own arguments (__init__'s OWN parameter list, NOT
		# the field=value sugar), wrap the result in Result[Foo,E] when
		# __init__ is fallible, dropping self's own refcount (but not
		# calling __del__) on Err - see RCCLASS ATTRIBUTE LIFETIME.md and
		# the approved plan. Scoped to non-subclassed RCClasses only,
		# matching lower_function's own scope check for __init__ itself.
		target_cls = self.lowering._try_resolve_namespace( node.func )

		# explicit generic-class construction via subscript - list[i32](...).
		# _try_resolve_namespace's own Subscript branch resolves this shape
		# to a Specialization (same as it always did for Name[T](...) over a
		# generic FUNCTION) - _ensure_resolved schedules that Specialization
		# the ordinary way (same discipline as node.resolved_callee/
		# resolved_construction below - compiler.py's own pipeline builds
		# the real struct from it once dequeued) and hands back the real,
		# concrete, already-monomorphized class (type_params/resolve both
		# cleared - see Monomorphizer.monomorphize_class), which every check
		# below this point already knows how to treat as an ordinary,
		# non-generic class
		if isinstance( target_cls, Specialization ) and isinstance( target_cls.base, ( RCClass, CStruct, CUnion, TaggedUnion, CEnum )):
			target_cls = self.lowering._ensure_resolved( target_cls )

		# module-level `_x`/`__x` privacy (SYNTAX.md, extended to cover
		# classes too) - checked HERE, not inside the shared _try_resolve_
		# namespace utility itself (self.lowering._try_resolve_namespace is
		# a thin delegate to TypeResolver._try_resolve_namespace, which is
		# ALSO reached by _stmt_diverges's own unrelated NoReturn re-probe -
		# see check_module_visibility's own comment on why enforcing
		# inside that shared utility produced a real false positive for
		# functions). Gated on target_cls actually being a ClassLike -
		# confirmed necessary via a SECOND real false positive, the same
		# shape as _stmt_diverges's: this recognizer is tried against
		# EVERY call node (construction_recognizers, _lower_call's own
		# tuple), including calls to ordinary FUNCTIONS that _try_resolve_
		# namespace happily resolves before this recognizer declines and
		# falls through - checking unconditionally there flagged lib/
		# builtins's own use of sys._assert (a Function, not a class)
		# purely because THIS probe touched it, not because anything
		# actually constructed it.
		if isinstance( target_cls, ClassLike ):
			self.lowering.discovery.check_module_visibility( target_cls, node.func, self._owning_module )

		# T(...) where T defines a static __call__ dispatches to
		# T.__call__(...) instead of construction - rewrite node.func to
		# Attribute(T, '__call__') in place and bail out (returning None
		# here just makes this recognizer decline, same as any other
		# shape mismatch): the remaining recognizers below in _lower_call's
		# own tuple all safely no-op against an Attribute ending in
		# '__call__' (none matches that shape), and _resolve_callee's
		# existing ClassName.static_method(...) path (already used by
		# int.from_str(...)) picks the rewritten node.func up unchanged
		# once the recognizer loop falls through - no new call-resolution
		# logic needed downstream. Reuses target_cls (already resolved just
		# above, including the generic-Specialization case) rather than a
		# second _try_resolve_namespace call - same reasoning as this
		# whole check's own placement, right after that resolution and
		# before the CEnum branch: no-op for the overwhelming common case
		# (target_cls isn't ClassLike, or has no local __call__), so every
		# OTHER class's construction stays byte-for-byte unchanged.
		if isinstance( target_cls, ClassLike ):
			call_member = target_cls.get_local( '__call__' )
			if call_member is not None:
				members = call_member.implementations if isinstance( call_member, Overload ) else [ call_member ]
				if not all( isinstance( m, Function ) and m.is_static for m in members ):
					self.lowering.discovery.fail(
						f'{target_cls.qualname}.__call__ must be declared @staticmethod to be used as {target_cls.qualname}(...): {ast.unparse(node)}',
						node,
					)
					return None
				new_func = ast.Attribute( value = node.func, attr = '__call__', ctx = ast.Load() )
				ast.copy_location( new_func, node.func )
				node.func = new_func
				return None

		# CEnum construction: EnumName(value) is a plain cast to the
		# enum's underlying type — no allocation, no refcounting, just
		# reinterpret the raw integer as the enum type. e.g. OSError(ENOENT)
		if isinstance( target_cls, CEnum ):
			if len( node.args ) != 1 or node.keywords:
				self.lowering.discovery.fail( f'{target_cls.qualname}(...) takes exactly one positional argument: {ast.unparse(node)}', node )
			self.lowering._ensure_resolved( target_cls )
			# a literal argument's magnitude must fit the enum's own
			# underlying type's real range - this is CONSTRUCTION, not a
			# cast, so _lower_scalar_cast's bit-reinterpretation exemption
			# (_allow_literal_bit_reinterpret) doesn't apply here; an out-
			# of-range value is a genuine mistake, unlike u32(-11)'s
			# deliberate WinAPI-style reinterpretation
			arg_node = node.args[0]
			value_type = target_cls.value_type
			if isinstance( arg_node, ast.Constant ) and type( arg_node.value ) is int and isinstance( value_type, Scalar ):
				lo, hi = int_stem_range( value_type )
				if not ( lo <= arg_node.value <= hi ):
					self.lowering.discovery.fail(
						f'{arg_node.value} is out of range for {target_cls.qualname} ({lo}..{hi}): {ast.unparse(node)}',
						node,
					)
			# lower the argument directly — no arithmetic-mode semantics
			# needed here; a CEnum has exactly the same runtime
			# representation as its underlying type, so OSError(42) is
			# just the value 42 with the enum type.
			#
			# The argument's own natural type space is value_type (the
			# underlying scalar), NEVER target_cls (the enum itself) - passing
			# target_cls down as _lower_expr's expected_type here (the
			# pre-fix code) was a category error that only ever incidentally
			# "worked" for a couple of argument shapes, not because it was
			# correct: a bare ast.Name/Variable operand (_expr_Name) ignores
			# expected_type entirely and keeps its own declared type, so
			# _check_assignable's later CEnum<->value_type exemption
			# correctly ran and correctly rejected a genuine kind mismatch
			# (e.g. i32 for a u32-backed enum) - but a Call/BinOp argument's
			# own result-typing tail (_lower_call's final dest allocation,
			# `expected_type or target_return_type`) instead SILENTLY
			# relabeled the destination temp's own type to target_cls
			# directly, with no check at all that the callee's real return
			# type was even a scalar, let alone value_type - confirmed via a
			# real repro: OSError(get_rc()) (get_rc() -> i32) and
			# OSError(rc + 0) both compiled silently, while the exact same
			# value through a bare local, OSError(rc), was rejected as "rc:
			# expected builtins.OSError, got intrinsics.i32" - an
			# inconsistency across argument SHAPE, not a real distinction in
			# what's being constructed. Fixed by routing every shape through
			# the SAME real scalar-cast machinery T(x)/compiler.cast(T,x)
			# already use (_lower_scalar_cast) - value_type is authoritative
			# (matching this construction's own "plain cast to the
			# underlying type" contract), any argument expression a plain
			# scalar cast would accept is accepted here too, and the result
			# is then relabeled (CastWrap, a zero-cost same-bits retag, same
			# as an RCClass upcast/safe scalar widening/pointer interchange
			# elsewhere in this file) to target_cls.
			assert isinstance( value_type, Scalar ), f'{target_cls.qualname}: CEnum value_type must be a scalar, got {value_type!r}'
			if isinstance( arg_node, ast.Constant ) and type( arg_node.value ) is int:
				# the already-range-checked literal fast path: unchanged from
				# before this fix - _expr_Constant already folds a CEnum-
				# expected int literal straight into an ir.Const tagged with
				# target_cls directly (kind+magnitude already validated, here
				# and in _expr_Constant itself), no runtime Temp/CastWrap
				# needed. Keeping this path bypassed by the general fix below
				# preserves that constant-folding (real callers/tests rely on
				# a literal CEnum construction lowering to a bare ir.Const,
				# not a cast instruction).
				return self._lower_expr( node.args[0], target_cls )
			# every NON-literal shape (bare Name, Call, BinOp, ...): the
			# argument's own natural type space is value_type (the underlying
			# scalar), NEVER target_cls (the enum itself) - passing target_cls
			# down as _lower_expr's expected_type here (the pre-fix code, for
			# every argument shape) was a category error that only ever
			# incidentally "worked" for a couple of shapes, not because it was
			# correct: a bare ast.Name/Variable operand (_expr_Name) ignores
			# expected_type entirely and keeps its own declared type, so
			# _check_assignable's later CEnum<->value_type exemption correctly
			# ran and correctly rejected a genuine kind mismatch (e.g. i32 for
			# a u32-backed enum) - but a Call/BinOp argument's own result-
			# typing tail (_lower_call's final dest allocation, `expected_type
			# or target_return_type`) instead SILENTLY relabeled the
			# destination temp's own type to target_cls directly, with no
			# check at all that the callee's real return type was even a
			# scalar, let alone value_type - confirmed via a real repro:
			# OSError(get_rc()) (get_rc() -> i32) and OSError(rc + 0) both
			# compiled silently, while the exact same value through a bare
			# local, OSError(rc), was rejected as "rc: expected
			# builtins.OSError, got intrinsics.i32" - an inconsistency across
			# argument SHAPE, not a real distinction in what's being
			# constructed. Fixed by lowering the argument against value_type
			# (strict=False - a HINT only, e.g. so an untyped nested literal
			# still settles on the right width; the real, authoritative check
			# that the argument is even scalar-shaped happens explicitly
			# below) and relabeling via CastWrap - a plain `(ctype)(operand)`
			# C cast (see emitter_c.py's _emit_cast 'wrap' mode), exactly C's
			# well-defined integer conversion rules, safe for the real cross-
			# signedness/width reinterpretation this construction call
			# promises, the same as an explicit T(x) scalar cast.
			arg_operand = self._lower_expr( node.args[0], value_type, strict = False )
			if not isinstance( arg_operand.type, Scalar ):
				self.lowering.discovery.fail(
					f'{target_cls.qualname}(...) argument must be a scalar value, got '
					f'{arg_operand.type.qualname if arg_operand.type is not None else "?"}: {ast.unparse(node)}',
					node,
				)
			dest = self._new_temp( target_cls )
			self._emit( ir.CastWrap( dest = dest, operand = arg_operand ) )
			return dest

		if not isinstance( target_cls, ClassLike ):
			return None
		# target_cls is already resolved by now - _try_resolve_namespace's
		# own lookup resolves whatever it returns
		assert target_cls.resolve is None, f'internal compiler error, {target_cls=} is not fully resolved'

		resolved_construction = getattr( node, 'resolved_construction', None )
		if resolved_construction is not None:
			# type_resolver.py's own generic-construction resolution
			# (_ReferenceResolver._try_resolve_generic_construction)
			# already inferred target_cls's own concrete type args from
			# this call's arguments and built the real, monomorphized
			# class + __init__ - an ordinary, concrete Function, never a
			# Specialization, same "resolved ahead of time" discipline as
			# node.resolved_callee. Neither gets scheduled here - that
			# happens the ordinary way, right below, exactly like the
			# plain (non-generic) branch already schedules target_cls/
			# init directly
			concrete_cls, init = resolved_construction
			self.lowering.schedule( concrete_cls )
			self.lowering._ensure_resolved( init )
			self_type = concrete_cls
			args, kwargs = self._lower_call_args( init, node )
		else:
			# a CHAIN lookup (self, then base, then base.base, ...), not a
			# flat own-class-only one: `class Bar(Real): pass` with no own
			# __init__ must find and call Real.__init__ on construction,
			# exactly like Python's own "no override -> inherit" MRO
			# semantics, rather than falling through to the no-__init__
			# field=value sugar below and silently dropping both Real's
			# fields AND its constructor call. A subclass that DOES declare
			# its own __init__ still finds THAT one first (chain_lookup
			# checks self before base) - unaffected, still required to
			# chain to its own base via super().__init__(...) itself (see
			# FunctionLowering._lower_super_init_if_required), never both
			# an inherited AND an own __init__ running for the same call.
			# Walking the chain also resolves every ancestor along the way
			# (chain_lookup's own side effect) - load-bearing for the
			# init-is-None case too, since _lower_allocate_fields's own
			# flattened_attributes() call needs that same resolution.
			init = target_cls.chain_lookup( '__init__' ) if isinstance( target_cls, ( RCClass, CStruct )) else target_cls.get_local_or_raise( '__init__' )
			if isinstance( init, Function ) and init.resolve is not None:
				# ordinarily already resolved by now - type_resolver.py's own
				# eager construction-call pre-pass (visit_Call) resolves
				# whatever chain_lookup/get_local finds, and for a generic
				# target_cls, monomorphizing it (the Specialization-swap
				# above) resolves its OWN methods as a side effect of
				# building the substituted copy. Neither covers an INHERITED
				# init reached via chain_lookup through an EXPLICIT
				# ClassName[T](...) subscript call: that pre-pass's own
				# _try_resolve_callable_namespace has no Subscript-over-a-
				# class handling at all (see _try_resolve_generic_
				# construction's own docstring), and the inherited init
				# belongs to a DIFFERENT, un-monomorphized ancestor class,
				# so monomorphizing target_cls never touches it either.
				# Resolve defensively here instead of asserting it must
				# already be true - same "the caller owns making sure it's
				# resolved" discipline _lower_super_init_if_required already
				# uses for this exact "found via chain_lookup" shape.
				self.lowering._ensure_resolved( init )
			if init is None:
				return self._lower_allocate_fields( target_cls, node, expected_type, '(...)' )
			if not isinstance( target_cls, RCClass ):
				# __init__ on a @cstruct/@cunion/@enum - not supported yet
				# (attribute lifetime tracking is scoped to RCClass, matching
				# RCCLASS ATTRIBUTE LIFETIME.md's own title) - falls through to
				# the normal call path, same "not callable" as always
				return None
			if isinstance( init, Overload ):
				# resolve to a single winning Function BEFORE everything
				# below (fallibility detection, field=value sugar, RC
				# attribute-lifetime tracking) - none of that needs to
				# change, it just needs a concrete Function instead of an
				# Overload group. Arguments are lowered TWICE for a generic
				# target_cls (once here, probe-only, again inside _lower_
				# generic_construction_args against the winning candidate's
				# real declared types) - same "probe first, lower for real
				# once the target is fixed" split the ordinary overloaded-
				# call path (_lower_call, below) already uses, just not
				# reusing its lowered operands directly (construction's own
				# downstream generic-inference pass needs to re-lower
				# against the WINNING candidate's parameter types anyway,
				# unlike an ordinary call - see _lower_generic_construction_
				# args). A runtime-dispatched winner (2+ candidates still
				# tied after resolve_call) isn't supported here - construct-
				# ion has no ConditionalDispatch machinery of its own.
				candidates = [ *init.stubs, *init.implementations ]
				if any( kw.arg is None for kw in node.keywords ):
					self.lowering.discovery.fail( f'**kwargs not supported yet: {ast.unparse(node)}', node )
				# the probe below is a REAL lowering (_lower_overload_arg falls
				# through to plain _lower_expr for anything but a bare
				# literal), not a side-effect-free type peek - snapshot first
				# and roll everything it did back once resolve_call has its
				# answer, since _lower_generic_construction_args below is
				# about to lower these SAME arg expressions again for real.
				# Without this, a probe argument with any real side effect
				# (e.g. a generator-returning call, whose own construction
				# increfs its captured receiver) permanently duplicates that
				# side effect - confirmed via a real repro: `list(pattern.
				# finditer(mv))` left `pattern` with one extra, never-released
				# reference, a real RC leak. Mirrors _lower_loop_body_with_
				# ownership_retry's own identical rollback discipline.
				instructions_mark = len( self._instructions )
				defer_flags_mark = len( self._defer_flags )
				for_obj_null_inits_mark = len( self._for_obj_null_inits )
				cancel_flags_mark = self._cfg.cancel_flag_count
				pending_temps_mark = len( self._pending_temps )
				# _current_fn is None for a module-level global initializer
				# (FunctionLowering(self, None).run_global) - no enclosing
				# function, so no per-function local-narrowing state exists
				# to snapshot/restore here at all
				names_snapshot = dict( self._current_fn.names ) if self._current_fn is not None else None
				probe_snapshot = self._cfg.snapshot()
				probe_args = [ self._lower_overload_arg( e, i, None, candidates, node ) for i, e in enumerate( node.args ) ]
				probe_kwargs = { kw.arg: self._lower_overload_arg( kw.value, None, kw.arg, candidates, node ) for kw in node.keywords }
				try:
					branches, resolved_init = overload_resolution.resolve_call(
						init.stubs, init.implementations,
						[ op.type for op in probe_args ], { name: op.type for name, op in probe_kwargs.items() },
						qualname = f'{target_cls.qualname}.__init__', same_type = self.lowering._type_resolver._same_type,
						protocol_conforms = self.lowering._type_conforms_to_protocol,
					)
				except CompileError as e:
					self.lowering.discovery.fail( str( e ), node )
				if branches:
					self.lowering.discovery.fail(
						f'{target_cls.qualname}.__init__: a runtime-dispatched overloaded constructor is not supported yet: {ast.unparse(node)}',
						node,
					)
				init = resolved_init
				del self._instructions[instructions_mark:]
				del self._defer_flags[defer_flags_mark:]
				del self._for_obj_null_inits[for_obj_null_inits_mark:]
				self._cfg.truncate_cancel_flags( cancel_flags_mark )
				del self._pending_temps[pending_temps_mark:]
				if self._current_fn is not None:
					self._current_fn.names.clear()
					self._current_fn.names.update( names_snapshot )
				self._cfg.hard_restore( probe_snapshot )
			if not isinstance( init, Function ):
				self.lowering.discovery.fail( f'{target_cls.qualname}.__init__ is overloaded - not supported yet: {ast.unparse(node)}', node )
			# init may be target_cls's OWN __init__ or an INHERITED one
			# found further up the chain (see the chain_lookup comment
			# above) - either way it's allowed to chain to ITS OWN base via
			# super().__init__(...) (see FunctionLowering._lower_super_
			# init_if_required) - no rejection needed here (RCClass-
			# subclassing plan Phase 2)
			self._check_rcclass_fully_implemented( target_cls, node, '(...)' )

			if target_cls.type_params:
				self_type, init, args, kwargs = self._lower_generic_construction_args( node, target_cls, init, expected_type )
				# see _fill_generic_call_defaults's own docstring - the
				# non-generic branch below gets this for free from
				# _lower_call_args; a generic class's own __init__ needs it
				# applied explicitly, same as any other generic call target
				self._fill_generic_call_defaults( init, args, kwargs, node )
			else:
				self.lowering.schedule( target_cls )
				self.lowering._ensure_resolved( init )
				self_type = target_cls
				args, kwargs = self._lower_call_args( init, node )

		# self_type is either already concrete (plain/resolved_construction
		# branches) or a generic-class Specialization (_lower_generic_
		# construction_args' own cls_spec) - _ensure_resolved is a no-op-
		# ish pass-through for an already-concrete class (same as
		# elsewhere in this file), so this one line handles both uniformly
		self.lowering.schedule( init.return_type )
		for param in init.parameters or []:
			self.lowering.schedule( param.type )
		if isinstance( self_type, Specialization ):
			# self_type was already resolved above (either by this method's
			# own earlier branches - target.schedule/_ensure_resolved(cls_spec)
			# in _lower_generic_construction_args - or by resolved_construction's
			# own pre-resolution) - re-calling _ensure_resolved would just
			# redundantly re-schedule() the same Specialization a second
			# time for no benefit (schedule()'s own id-based _seen dedup
			# makes it harmless, just wasted work) - .monomorphized is the
			# same cache Monomorphizer.monomorphize_class itself reads
			# (mpy_types.py's Specialization), a plain, side-effect-free
			# read of what's already there
			concrete_cls = self_type.monomorphized if self_type.monomorphized is not None else self.lowering._ensure_resolved( self_type )
		else:
			concrete_cls = self_type
		assert isinstance( concrete_cls, RCClass ), f'internal compiler error: {self_type=} did not resolve to a concrete RCClass'
		# synthesize (idempotent, memoized) rather than rely solely on the
		# compiler.py class-registration trigger - schedule() is a
		# deferred queue, so THIS call site needs $$__new__'s live Function
		# object available right now, not whenever it eventually gets
		# dequeued (see _synthesize_rcclass_constructor's own docstring).
		# Uses the RETURN value directly, never a follow-up
		# concrete_cls.get_local('$$__new__') lookup - an overloaded
		# __init__ can synthesize several coexisting $$__new__ wrappers for
		# the SAME concrete_cls (one per distinct init actually constructed
		# with), and cls.add_name('$$__new__', ...) only ever keeps the
		# LAST one under that shared name; the return value is always the
		# one that matches THIS call's own `init`, regardless of how many
		# others have been synthesized for concrete_cls in the meantime.
		new_fn = self.lowering._type_resolver._synthesize_rcclass_constructor( concrete_cls, init )
		assert isinstance( new_fn, Function ), f'internal compiler error: {concrete_cls.qualname} has no synthesized $$__new__'
		self.lowering.schedule( new_fn.return_type )
		dest = self._new_temp( new_fn.return_type )
		# new_fn's own hidden __alloc_loc param (see its own synthesis
		# comment) - THIS call site (node) is the real, per-construction-
		# call-site location dump_live_objects needs; new_fn's own body
		# can't know it (one shared compiled function, every real Foo(...)
		# in the program reuses it). Built directly here, not via the
		# ordinary default-parameter-filling path _lower_call_args already
		# ran above (that matched args/kwargs against `init`'s own
		# parameter list, never new_fn's) - same file/line source
		# _fold_caller_location's own caller_file/caller_line use.
		alloc_loc_module = self.lowering.discovery.module_stack[-1] if self.lowering.discovery.module_stack else None
		alloc_loc_file = str( alloc_loc_module.file ) if alloc_loc_module is not None else '<unknown>'
		u8_cls = self.lowering.discovery.get_intrinsics()['u8']
		const_ptr_cls = self.lowering.discovery.get_intrinsics()['ConstPtr']
		alloc_loc_type = self.lowering.discovery._get_or_create_specialization( const_ptr_cls, [ u8_cls ] )
		kwargs['__alloc_loc'] = ir.Const( type = alloc_loc_type, value = f'{alloc_loc_file}:{node.lineno}' )
		self._emit( ir.Call( dest = dest, target = new_fn, receiver = None, args = args, kwargs = kwargs ))
		return dest

	def _lower_and_infer_call_args(
		self, node: ast.Call, callee: Function, type_params: list[TypeVar], bindings: dict[int,Type], qualname: str,
	) -> tuple[list[ir.Operand],dict[str,ir.Operand]]:
		# shared by _lower_generic_construction_args/_lower_class_generic_
		# method_call - both need to lower a call's arguments against a
		# callee whose own class type params aren't fully bound yet, then
		# use those SAME arguments' real lowered types to refine `bindings`
		# further. Matches call args against callee's own ABSTRACT
		# parameter list, lowers each one with an expected-type hint built
		# from `bindings` as pinned SO FAR (a class type param not yet
		# bound just passes its own bare TypeVar through -
		# _substitute_type_params leaves anything it doesn't recognize
		# alone, so an unbound param position simply gets no useful hint,
		# same as today), applies move hooks, then unifies each argument's
		# own real lowered type against its declared parameter type,
		# mutating `bindings` in place. The caller still owns everything
		# after that (checking for a still-missing binding, building the
		# final concrete arg list/Specialization) - that part differs too
		# much between callers (construction pins from a possibly-
		# Result[_,_]-wrapped expected_type via _result_shape; a class
		# method's own receiver-vs-static distinction) to fold in here too
		positional, keyword = self.lowering._match_call_args( callee, node )
		partial_args = [ bindings.get( id( tv ), tv ) for tv in type_params ]
		# strict=False on both: the substituted hint can still contain an
		# unbound TypeVar nested inside it (a class type param not yet
		# bound) - it's an inference HINT, not a validated requirement; the
		# REAL validation/binding is _unify_type_param below
		args = [
			self._lower_expr( expr, self.lowering._substitute_type_params( param.type, type_params, partial_args ), strict = False )
			for param, expr in positional
		]
		kwargs = {
			param.stem: self._lower_expr( expr, self.lowering._substitute_type_params( param.type, type_params, partial_args ), strict = False )
			for param, expr in keyword
		}
		# strict=False above deliberately skips _coerce_or_check_operand's own
		# case-2 auto-or_throw() hook (a HINT-only lowering, not a real
		# requirement yet - see this method's own comment above) - same as
		# _lower_binary_operands' own identical strict=False lowering, this
		# reinstates it manually right here: an unconsumed Result[T,E]
		# argument (e.g. `Result.Ok(c)` where `c = a + b` left c as a raw
		# Result[u32,OverflowError]) would otherwise reach _unify_type_param
		# below still Result-shaped and get "inferred as both u32 and
		# Result[u32,OverflowError]" instead of just binding T=u32. Skipped
		# whenever the param's own (possibly still-abstract/unbound)
		# declared type is ITSELF Result-shaped - a genuinely nested
		# Result-typed argument (rare, but legal) must keep flowing through
		# raw, exactly like every other case-2 site's identical guard
		args = [
			self._auto_consume_hint_arg( node, operand, param.type )
			for ( param, _expr ), operand in zip( positional, args )
		]
		kwargs = {
			param.stem: self._auto_consume_hint_arg( node, kwargs[param.stem], param.type )
			for param, _expr in keyword
		}
		for ( param, _expr ), operand in zip( positional, args ):
			self._apply_move_hook( param, operand, qualname, node )
		for param, _expr in keyword:
			self._apply_move_hook( param, kwargs[param.stem], qualname, node )
		for ( param, _expr ), operand in zip( positional, args ):
			self.lowering._unify_type_param( type_params, param.type, operand.type, bindings, node, qualname )
		for param, _expr in keyword:
			self.lowering._unify_type_param( type_params, param.type, kwargs[param.stem].type, bindings, node, qualname )
		return args, kwargs

	def _auto_consume_hint_arg( self, node: ast.AST, operand: ir.Operand, declared_param_type: Type|None ) -> ir.Operand:
		''' see _lower_and_infer_call_args' own comment on why this exists -
		reinstates case 2 of the general auto-or_throw() rule for a
		strict=False, hint-only generic-argument lowering, gated on the
		PARAM's own still-abstract declared type (not a substituted/bound
		one - a not-yet-bound bare TypeVar reads as "not Result-shaped"
		here, same as everywhere else, so the common case (inferring T from
		an incidentally-Result-shaped argument) auto-consumes) rather than
		being Result-shaped itself. '''
		if ( self.lowering._type_resolver._result_shape( operand.type ) is None
				or self.lowering._type_resolver._result_shape( declared_param_type ) is not None ):
			return operand
		consumed = self._auto_or_throw( node, operand, self.lowering._AUTO_CONSUME_ALTERNATIVES, want_result = True )
		assert consumed is not None # want_result=True above guarantees this
		return consumed

	def _lower_generic_construction_args( self, node: ast.Call, target_cls: RCClass, init: Function, expected_type: Type|None ) -> tuple[RCClass|Specialization,Function,list[ir.Operand],dict[str,ir.Operand]]:
		# Box(...) where Box is generic: target_cls's own concrete type args
		# have to be pinned down before __init__ can be called - same two-
		# phase strategy _lower_class_generic_method_call's own inference
		# branch uses (unify from expected_type first, then refine from the
		# lowered arguments' own types), since a class constructor's type
		# params are exactly as inferable as a generic method's - working
		# against __init__'s ABSTRACT parameter list throughout (substituting
		# per-parameter via _substitute_type_params) because the concrete,
		# monomorphized __init__ can only be built once the generic type
		# params are inferred from the call's arguments. init's own Function
		# body (parameters/return_type) was already resolved by the caller.
		assert init.resolve is None, f'internal compiler error, {init.qualname} was not resolved before construction'
		class_type_params = target_cls.type_params or []
		# __init__ may declare its OWN extra type param(s) on top of
		# target_cls's (e.g. list[T].__init__[S: IteratorProtocol[T]]) - S
		# and T have to be inferred TOGETHER, in one combined pass, not two
		# separate ones: T never appears directly in init's own (abstract)
		# parameter list at all when init has its own S (only inside S's
		# bound, e.g. IteratorProtocol[T]) - a class-type-params-ONLY unify
		# pass never binds T in that shape, only _unify_type_param's own
		# reverse-bound-unification (through S's parametrized protocol
		# bound) ever does, and that only fires when T is in the SAME
		# type_params list being unified as S itself. Same combined-list
		# technique a free generic function with this identical shape
		# already relies on (max[T, S: Iterable[T]]) - just extended here
		# to cover a class's own type params too, not only a function's.
		own_type_params = init.type_params or []
		type_params = [ *class_type_params, *own_type_params ]
		bindings: dict[int,Type] = {}
		# expected_type pins target_cls's own args directly for a non-
		# fallible __init__ (b: Box[i32] = Box(1)) - but for a FALLIBLE one,
		# Box(...) itself becomes Result[Box[i32],E] (SYNTAX.md), so the
		# surrounding annotation is r: Result[Box[i32],MyError], one level
		# removed from target_cls. Peek through a Result[_,_] wrapper via
		# _result_shape, which speculatively no-ops (rather than hard-
		# failing) when this particular construction isn't Result-shaped
		# at all, same posture as _maybe_consume_result
		pinning_type = expected_type
		shape = self.lowering._type_resolver._result_shape( expected_type )
		if shape is not None:
			pinning_type = shape[0]
		if isinstance( pinning_type, Specialization ) and pinning_type.base is target_cls:
			for tv, arg in zip( class_type_params, pinning_type.args ):
				bindings[ id( tv ) ] = arg

		args, kwargs = self._lower_and_infer_call_args( node, init, type_params, bindings, target_cls.qualname )

		missing = [ tv.stem for tv in type_params if id( tv ) not in bindings ]
		if missing:
			self.lowering.discovery.fail(
				f'{target_cls.qualname}(...): cannot infer type parameter(s) {", ".join(missing)} from these arguments or the surrounding expected type: {ast.unparse(node)}',
				node,
			)
		concrete_args = [ bindings[id(tv)] for tv in class_type_params ]
		own_concrete_args = [ bindings[id(tv)] for tv in own_type_params ]
		self.lowering._check_type_param_bounds( node, type_params, [ *concrete_args, *own_concrete_args ], target_cls.qualname )
		cls_spec = self.lowering.discovery._get_or_create_specialization( target_cls, concrete_args )
		self.lowering._ensure_resolved( cls_spec )

		if own_type_params:
			# build the fully-concrete __init__ directly, substituting BOTH
			# class and own args in one shot (_build_monomorphized_function
			# zips type_params/args positionally, and doesn't care which
			# came from the class vs the method itself) - mirrors
			# Monomorphizer._partial_class_substituted_method's own
			# substitution call, just skipping its "stay generic in S"
			# half (result_type_params stays None/default here): S is
			# already concrete by this point, unlike THAT method's own
			# use case (an ordinary receiver-based method call, where S is
			# only ever known later, at its own separate call site -
			# construction has no such second call site, S is resolved
			# right here from the SAME arguments T is)
			qualname = f'{cls_spec.qualname}.{init.stem}'
			monomorphized_init = self.lowering._monomorphizer._build_monomorphized_function(
				init, type_params, [ *concrete_args, *own_concrete_args ], qualname, cls_spec,
			)
			return cls_spec, monomorphized_init, args, kwargs

		init_spec = self.lowering.discovery._get_or_create_specialization( init, concrete_args )
		monomorphized_init = self.lowering._ensure_resolved( init_spec ) # concrete_cls's own build above (monomorphize_class's method-substitution loop) already populated init_spec.monomorphized as a side effect - same (init, concrete_args) key
		return cls_spec, monomorphized_init, args, kwargs

	def _try_lower_scalar_construct_call( self, node: ast.Call, expected_type: Type|None ) -> ir.Operand|None:
		# ScalarName(x) - Python's own int(x)/float(x)-style constructor-as-
		# cast idiom. Deliberately NOT routed through _try_lower_construct_call
		# (ClassLike-only: its Allocate/self/RC-fallible-construction machinery
		# is meaningless for a scalar - no self to allocate, no attributes, no
		# refcounting)
		target_cls = self.lowering._try_resolve_namespace( node.func )
		if not isinstance( target_cls, Scalar ):
			return None
		if len( node.args ) != 1 or node.keywords:
			self.lowering.discovery.fail( f'{target_cls.qualname}(...) takes exactly one argument: {ast.unparse(node)}', node )
		arg_node = node.args[0]
		if isinstance( arg_node, ast.Constant ):
			return self._lower_scalar_cast( target_cls, arg_node, node )
		operand = self._lower_expr( arg_node, None )
		if isinstance( operand.type, Scalar ):
			# the real motivating case (u32(s.byte_len())) - same
			# arithmetic-mode-respecting logic compiler.cast(...) uses, no
			# dunder dispatch needed: one compiler primitive already
			# covers every Scalar-to-Scalar pair uniformly
			return self._lower_scalar_cast( target_cls, operand, node )
		if isinstance( operand.type, CEnum ) and operand.type.value_type is target_cls:
			# the reverse of CEnum construction (Foo(u8(0)), above): a
			# CEnum's runtime representation IS its own value_type exactly
			# (same comment there), so u8(some_foo) is just as much a bare
			# reinterpret - no range check needed (target_cls IS already
			# the enum's own declared underlying type, so it can never be
			# narrowing), no dunder dispatch needed either.
			dest = self._new_temp( target_cls )
			self._emit( ir.CastWrap( dest = dest, operand = operand ))
			return dest
		# a non-Scalar source (e.g. an RCClass) - this is where library-
		# authored extensibility (Scalar.names, see discovery.py's
		# visit_Assign) actually earns its keep: a `SomeClass.__u32__(self)
		# -> u32: ...` is dispatched here exactly like any other method
		# call - same _mode_qualified_dunder_names/_emit_fallible_method_
		# call pair binop dispatch already uses for __add__/__wrapped_add__/
		# __saturated_add__ (see _MODE_DUNDER_PREFIX's own module-level
		# comment), reused here rather than reimplemented: __i32__'s own
		# value-range check is genuinely mode-INDEPENDENT (SYNTAX.md's
		# .to_T() contract - "no meaningful wrapped/saturated value-range
		# check"), so under wrap/saturate mode the qualified name is tried
		# first and, if the class declares one (e.g. int.__saturated_i32__,
		# infallible - clamps rather than erring, mirroring the compiler's
		# OWN intrinsic cast picking a different, infallible opcode
		# (CastSaturate) under this exact mode), wins; otherwise this falls
		# through to the base name exactly like binop dispatch's identical
		# miss does, so a class with nothing registered under the qualified
		# name needs no special-casing at all.
		for candidate in self._mode_qualified_dunder_names( f'__{target_cls.stem}__' ):
			dunder = self.lowering._find_method( operand.type, candidate )
			if dunder is not None:
				return self._emit_fallible_method_call( node, dunder, operand, [], expected_type )
		self.lowering.discovery.fail(
			f'{operand.type.qualname if operand.type else "?"} has no __{target_cls.stem}__ method - cannot convert to {target_cls.qualname}: {ast.unparse(node)}',
			node,
		)
