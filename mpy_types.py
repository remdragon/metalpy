# stdlib imports:
import ast
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Union

@dataclass( kw_only = True )
class Name:
	stem: str # local name like 'str' instead of 'builtins.str'
	qualname: str # fully qualified name: 'builtins.str' instead of 'str'

	# we don't always know where a name is defined the first time we see it:
	file: Path|None
	line: int|None

@dataclass( kw_only = True )
class Type( Name ):
	''' maybe only use this to distinguish types from values '''
	def leaves( self ) -> list['Type']:
		# a single concrete type is its own only leaf - TaggedUnion overrides
		# this to return its member types instead. shared by overload
		# matching (see Overload.resolve_call and the _is_covered_by/_overlaps
		# primitives below) to treat "a union" and "a plain type" uniformly.
		return [ self ]

class ScopeMixin:
	'''
	shared shape for anything that owns a local namespace (Module, the various
	class kinds, Function) - not a dataclass itself (no fields of its own) so it
	can't interfere with dataclass field collection on whatever it's mixed into.
	'''
	names: dict[str,Name]

	def add_name( self, name: str, name_obj: Name ) -> None:
		self.names[name] = name_obj

	def get_local( self, name: str ) -> Name|None:
		return self.names.get( name )

	def in_private_scope( self, scope: 'Type|None' ) -> bool:
		''' true if `scope` (whatever class the function currently being
		lowered belongs to - Lowering._current_fn.cls, possibly a
		Specialization for a monomorphized generic-class method) IS this
		class itself, i.e. private access (currently just a `__allocate__`
		call - see Lowering._try_lower_allocate_call) from a method of
		this class is allowed. `scope` is unwrapped to its own abstract
		base first - Specialization.base means "the generic template" (NOT
		to be confused with RCClass.base, "parent class in an inheritance
		chain") - a generic class's own body always spells the template
		name bare (Result.Ok's own body says `Result.__allocate__`, never
		`Result[i32,E].__allocate__`), so the comparison has to be against
		whichever template `self` and `scope` are each an instance of, not
		against one specific instantiation of it. Lives on ScopeMixin
		(not just RCClass) because every ClassLike kind can own methods
		and therefore has the same privacy concept - confirmed load-
		bearing for TaggedUnion specifically: every @union member
		constructor's own synthesized body calls the outer union's
		`__allocate__` this way (see union_storage.py's _build_member_
		constructor) '''
		base = scope.base if isinstance( scope, Specialization ) else scope
		return base is self

@dataclass( kw_only = True )
class Scalar( Type, ScopeMixin ):
	'''
	isize, usize, i32, u32, etc - also used for generic pointer intrinsics
	(Ptr, ConstPtr), which is why type_params exists here too. names is
	populated by library source (`usize.__u32__ = some_function` - see
	discovery.py's visit_Assign) rather than a parsed class body, since
	intrinsic scalars aren't declared from any real source file
	'''
	sizeof: int
	type_params: list['TypeVar']|None = None
	names: dict[str,Name] = field( default_factory = dict )

@dataclass( kw_only = True )
class TypeVar( Type ):
	''' a placeholder for one of a generic's type parameters, e.g. T in class Result[T,E] '''

@dataclass( kw_only = True )
class Specialization( Type ):
	''' a generic base type applied to concrete (or still-typevar) type arguments, e.g. Result[i32,IntError] '''
	base: Type
	args: list[Type]
	# populated by Monomorphizer.monomorphized_function/monomorphize_class
	# the first time this Specialization is actually monomorphized - the
	# real, substituted Function (if base is a Function) or ClassLike (if
	# base is one) this Specialization stands in for. None until then; a
	# self-caching slot in the same spirit as `resolve` below, except the
	# substituted value itself is the cache rather than a callback, and it
	# doesn't self-clear. Lives here (not a separate id(spec)-keyed side
	# table) because a Specialization is already the canonical, memoized-
	# by-qualname object for its own (base, args) pair (see discovery.py's
	# _get_or_create_specialization) - every reference to the same
	# instantiation shares this one cache for free
	monomorphized: 'Function|ClassLike|None' = None

	# a Specialization has no members of its own - a generic's methods/
	# attributes live entirely on its base (e.g. Result[T,E]'s .is_err lives
	# on Result, not on any particular Result[i32,Err]) - these passthroughs
	# let callers (lowering.py's _ensure_resolved/_attr_lookup*) treat a
	# Specialization exactly like any other Type without unwrapping it first
	@property
	def resolve( self ) -> Callable[[],None]|None:
		return getattr( self.base, 'resolve', None )

	@property
	def names( self ) -> dict[str,Name]|None:
		return getattr( self.base, 'names', None )

@dataclass( kw_only = True )
class Variable( Name ):
	type: Type|None = None
	# None means already resolved (or never needed resolving); otherwise call
	# it to populate .type, after which it sets itself back to None. Checking
	# "is this resolved" is just `var.resolve is None`.
	resolve: Callable[[],None]|None = None
	# the initializer expression, unresolved (stage 2's concern, same as
	# Function.node's body) - None for a declaration with no initializer
	# (e.g. a bare `x: i32` class attribute)
	init: ast.expr|None = None
	# True only for a genuine module-level global - the same Variable class
	# also represents class attributes and (stage 2's own, lowering.py-built)
	# local variables, neither of which is a standalone compile unit; this is
	# what lets Compiler._enqueue tell them apart without a separate lookup
	is_global: bool = False
	# stage 2's lowered form of `init` (None until Compiler._lower's Variable
	# branch runs) - kept directly on the Variable itself, not only reachable
	# through compiler.globals' own LoweredGlobal list, so a global's own
	# initializer instructions travel with the variable object (see
	# PLAN_GLOBAL_INIT.md). String-quoted to avoid a mpy_types<->ir import
	# cycle (ir.py doesn't need to know about Variable at all).
	init_instructions: list['ir.Instruction']|None = None

@dataclass( kw_only = True )
class Parameter( Variable ):
	'''
	a single function parameter. the kind flags mirror Python's own
	call-site rules (at most one of is_vararg/is_kwarg, mutually exclusive
	with is_posonly/is_kwonly) - stage 2 needs them plus `default` to bind
	keyword/optional call-site arguments down to positional ones.
	'''
	is_posonly: bool = False
	is_kwonly: bool = False
	is_vararg: bool = False # *args
	is_kwarg: bool = False # **kwargs
	default: ast.expr|None = None # unresolved - stage 2's concern, same as Function.node's body

@dataclass( kw_only = True )
class Move( Type ):
	''' `move[T]` in annotation position - ownership of a T is transferred into this binding rather than borrowed/copied. The CFG uses this to know the source binding must be invalidated after the transfer.

	NOTE: this is modeled as a Type wrapper for now, which is arguably wrong -
	see TODO.txt: move[T] is really an ownership status (OWNED vs BORROWED)
	on a binding, not a distinct type from T itself. A move[T]-typed value
	currently can't satisfy a plain T-typed parameter anywhere (including
	overload matching) as a result - left alone rather than patched around
	(e.g. by overriding .leaves()) since the real fix belongs with the
	CFG/incref-decref ownership-tracking work, not a standalone tweak here. '''
	inner: Type

@dataclass( kw_only = True )
class Copy( Type ):
	''' `copy[T]` in annotation position - the callee wants its own
	independent reference (an explicit INCREF in its own prologue, a
	matching DECREF at its own exit), regardless of whatever the caller
	already holds. Unlike move[T], this is a unilateral request: no
	call-site marker is needed/allowed (SYNTAX.md) - the caller's own
	binding is completely unaffected. See TODO.txt/RC MANAGEMENT.md for
	the CFG work this exists for. '''
	inner: Type

@dataclass( kw_only = True )
class CallableType( Type ):
	''' `Callable[[Arg1,Arg2,...], Ret]` in annotation position - a bare
	function SIGNATURE used as a type (see PLAN_CALLABLE.md), for typing a
	function-pointer value: Ptr[Callable[[Ptr[None],Ptr[None]],bool]], not
	a callable OBJECT (no receiver/closure environment - see the plan doc's
	own "deferred" list for why bound-method/capturing-lambda references
	aren't in scope yet). Not a ScopeMixin - has no members of its own,
	purely a shape to type-check a bare function reference or an indirect
	call against. Interned by discovery.py's _get_or_create_callable_type,
	the same way Specialization/TaggedUnion/Move/Copy already are, so two
	annotations spelling the same signature share one object (needed for
	identity-based comparisons elsewhere, e.g. _leaf_is_accepted). '''
	arg_types: list[Type]
	return_type: Type

@dataclass( kw_only = True )
class TupleType( Type ):
	''' `tuple[T0, T1, ..., Tn]` in annotation position (see PLAN_TUPLE.md) -
	a heterogeneous, fixed-arity value group. Unlike list[T]/dict[K,V]
	(ordinary fixed-arity generics, matched against a class's own
	type_params by discovery.py's ordinary generic-subscript path), tuple
	is variadic arity AND heterogeneous, so there's no type_params-bearing
	base class to subscript against - recognized textually in
	visit_Subscript instead, the same way Callable[...]/Closure[...] are.
	Not a ScopeMixin itself (mirrors CallableType - no members of its own
	on the TYPE); unlike CallableType (a bare, receiver-less function-
	pointer SHAPE that's never constructed), a tuple genuinely needs a
	real, constructible, destructible, RC-aware backing class - synthesized
	lazily by tuple_storage.TupleStorage.get() the first time this exact
	TupleType is touched, and cached here afterward (self-caching slot in
	the same spirit as Specialization.monomorphized above, except the
	backing class itself is the cache rather than a callback, and it
	doesn't self-clear). Interned by discovery.py's _get_or_create_tuple_
	type, the same way CallableType already is, so two annotations
	spelling the same element-type list share one object (and therefore
	one backing class/one synthesized __init__, not a fresh one per
	occurrence). '''
	elem_types: list[Type]
	backing: 'RCClass|None' = None

# a class's own body scan (revealing its attribute/method *names*) is
# deferred behind .resolve, exactly like a Function's parameters or a
# Variable's type - nothing about a class's members is known until something
# calls it. type_params is the one exception: it's parsed eagerly, at
# creation time, because external code subscripting this class as a generic
# (Result[i32,usize]) needs to see it before this class's own .resolve ever runs.

# --- shared single-inheritance-chain/vtable helpers, RCClass and CStruct ---
#
# RCClass and CStruct both have an identically-shaped .base/.methods/.names/
# .resolve (single inheritance, own-members-only .names, lazy .resolve) - no
# real shared base class exists to hang one implementation off of (ClassLike,
# below, is a plain Union type alias, not a class), so these live as free
# functions instead, parameterized over 'RCClass|CStruct', and each class's
# own same-named method just delegates to the matching one here. Originally
# CStruct-only (RCClass subclassing/vtables was deferred - see
# PLAN_SUBCLASSING_VTABLES_COM.md); generalized once RCClass subclassing
# work resumed (see PLAN's own RCClass-subclassing follow-up).

def chain_lookup( cls: 'RCClass|CStruct', name: str ) -> Name|None:
	''' walk this class's own single-inheritance chain (self, then base,
	then base.base, ... until None) looking for `name` - .names only ever
	holds a class's OWN declared members (discovery.py never merges a
	base's own names into a subclass), so a subclass needs this to see an
	inherited method/attribute at all. Only ever non-trivial for a class
	that actually has a base (a plain, non-@interface CStruct can't have
	one at all - see discovery.py's _parse_ClassDef_CStruct, which rejects
	bases outright for that case). '''
	node: 'RCClass|CStruct|None' = cls
	while node is not None:
		if node.resolve is not None: # each level's .names is populated lazily, same "None means already resolved" convention as everywhere else - a base's own body may not have run yet just because the derived class's own resolve() (already done by the caller) ran
			node.resolve()
		found = node.names.get( name )
		if found is not None:
			return found
		node = node.base
	return None

def own_new_virtual_slots( cls: 'RCClass|CStruct' ) -> list['Function']:
	''' this class's OWN @virtual methods that AREN'T already a slot
	somewhere in its ancestor chain - i.e. genuinely NEW vtable slots
	introduced here, not overrides of an inherited one. Any level in the
	chain can introduce new slots (not just the root - see vtbl_owner's
	own docstring for why: real interface hierarchies routinely add
	methods at every level, e.g. IUnknown -> ICustom (adds methods) ->
	ConcreteImpl, which a root-only-introduces-slots rule can never
	express). '''
	if cls.resolve is not None:
		cls.resolve()
	inherited_names: set[str] = set()
	node = cls.base
	while node is not None:
		if node.resolve is not None:
			node.resolve()
		inherited_names.update( m.stem for m in node.methods if isinstance( m, Function ) and m.is_virtual )
		node = node.base
	return [ m for m in cls.methods if isinstance( m, Function ) and m.is_virtual and m.stem not in inherited_names ]

def vtbl_owner( cls: 'RCClass|CStruct' ) -> 'RCClass|CStruct':
	''' the nearest class at or above `cls` (cls itself, or walking up
	.base) whose own Vtbl C struct type is the one cls's own $vtable field
	actually points at - the nearest one (starting from cls) that
	introduces at least one genuinely new slot (see own_new_virtual_slots).
	A class that adds nothing of its own (pure overrides, or no @virtual
	methods at all) simply reuses whatever ancestor's Vtbl type is already
	in effect - matches real COM: FooImpl (an ordinary implementation, no
	new capabilities) still has an $vtable field literally typed as
	whichever interface it implements' own IFooVtbl*, not a FooImplVtbl of
	its own. '''
	node = cls
	while node.base is not None and not own_new_virtual_slots( node ):
		node = node.base
	return node

def virtual_slots( cls: 'RCClass|CStruct' ) -> list['Function']:
	''' the full, ordered slot list for THIS class's own EFFECTIVE vtable
	type (vtbl_owner()'s own type) - every new-slot-introducing ancestor's
	own slots, root-first, up to and including vtbl_owner() itself
	(.methods is append-only in source order - see discovery.py's
	_parse_function, so each level's own contribution is already
	declaration-ordered). This is a superset walk, not "only the root" -
	see vtbl_owner's own docstring on why every level can contribute. '''
	owner = vtbl_owner( cls )
	chain: list['RCClass|CStruct'] = []
	node: 'RCClass|CStruct|None' = owner
	while node is not None:
		chain.append( node )
		node = node.base
	slots: list[Function] = []
	for node in reversed( chain ):
		slots.extend( own_new_virtual_slots( node ))
	return slots

def flattened_attributes( cls: 'RCClass|CStruct' ) -> list['Variable']:
	''' every attribute declared anywhere in cls's own single-inheritance
	chain (cls itself, then cls.base, then cls.base.base, ... until None),
	base-first/most-derived-last order - the same order emitter_c.py's own
	emit_rcclass/emit_cstruct field-flattening walk and type_resolver.py's
	_synthesize_rcclass_destructor use for real struct layout. .attributes
	alone only ever holds a class's OWN declared fields (discovery.py never
	merges a base's own fields into a subclass) - this is what a
	subclass's field=value construction sugar (no __init__ at all
	anywhere in the chain) and super().__init__() chaining (the base's
	own portion specifically - see lowering.py's own caller) both need
	instead. '''
	chain: list['RCClass|CStruct'] = []
	node: 'RCClass|CStruct|None' = cls
	while node is not None:
		chain.append( node )
		node = node.base
	attrs: list[Variable] = []
	for node in reversed( chain ):
		attrs.extend( node.attributes )
	return attrs

@dataclass( kw_only = True )
class RCClass( Type, ScopeMixin ): # normal ref-counted class
	# base is resolved eagerly at class-creation time, same as type_params -
	# Python itself requires a base class to already exist when the `class
	# Foo(Base):` statement runs, so there's no forward-reference case to
	# defer here. Multiple inheritance is a compile error (see discovery.py),
	# so this is a single pointer, not a list/MRO.
	base: 'RCClass|None' = None
	type_params: list[TypeVar]|None = None # if not None, this is a generic class
	attributes: list[Variable] = field( default_factory = list )
	methods: list['Function|Overload'] = field( default_factory = list )
	names: dict[str,Name] = field( default_factory = dict )
	resolve: Callable[[],None]|None = None

	def chain_lookup( self, name: str ) -> Name|None:
		return chain_lookup( self, name )

	def own_new_virtual_slots( self ) -> list['Function']:
		return own_new_virtual_slots( self )

	def vtbl_owner( self ) -> 'RCClass':
		return vtbl_owner( self )

	def virtual_slots( self ) -> list['Function']:
		return virtual_slots( self )

	def flattened_attributes( self ) -> list[Variable]:
		return flattened_attributes( self )

@dataclass( kw_only = True )
class ClosureType( RCClass ):
	''' `Closure[[Arg1,Arg2,...], Ret]` - a bound-method VALUE (`worker.run`
	used as a value, not called - see PLAN_CALLABLE.md's own "closure in
	miniature" deferred item, and lowering.py's _expr_Attribute for where
	it's actually constructed). Unlike CallableType (a bare, receiver-less
	function-pointer SHAPE, never RC-managed, always spelled Ptr[Callable
	[...]]), a closure genuinely owns a reference to its captured receiver
	- "creating a closure is by definition creating a new reference to an
	RC object" - so it's a REAL RCClass (subclassed, not wrapped): every
	existing RC mechanism (cfg.py's is_rc/rc_leaves/assign/move,
	emitter_c.py's retain_object/release_object) applies to it completely
	unchanged, no new special-casing needed anywhere past discovery.py.
	Two fields (fn/self, both Ptr[None] - see discovery.py's
	_get_or_create_closure_type), populated lazily via the ordinary
	RCClass.resolve convention. arg_types/return_type are ONLY needed for
	type-checking a call THROUGH a closure value (lowering.py's
	_try_lower_closure_call) and building the right trampoline signature -
	the underlying struct layout never depends on them, only the interning
	key does (two different signatures must never be assignment-
	compatible with each other, even though their runtime representation
	is identical - same reasoning CallableType's own interning already
	uses). '''
	arg_types: list[Type] = field( default_factory = list )
	return_type: Type|None = None

@dataclass( kw_only = True )
class CStruct( Type, ScopeMixin ): # @cstruct class Foo:
	# base is only meaningful for @interface CStructs (single inheritance,
	# same "resolved eagerly at class-creation time" reasoning as
	# RCClass.base above) - a plain (non-@interface) CStruct subclassing
	# another CStruct is out of scope for now (see
	# PLAN_SUBCLASSING_VTABLES_COM.md's "Subclassing mechanics")
	base: 'CStruct|None' = None
	is_interface: bool = False # @interface class Foo: - NOT inherited implicitly, see plan doc
	type_params: list[TypeVar]|None = None
	attributes: list[Variable] = field( default_factory = list )
	methods: list['Function|Overload'] = field( default_factory = list )
	names: dict[str,Name] = field( default_factory = dict )
	resolve: Callable[[],None]|None = None

	def chain_lookup( self, name: str ) -> Name|None:
		return chain_lookup( self, name )

	def own_new_virtual_slots( self ) -> list['Function']:
		return own_new_virtual_slots( self )

	def vtbl_owner( self ) -> 'CStruct':
		return vtbl_owner( self )

	def virtual_slots( self ) -> list['Function']:
		return virtual_slots( self )

	def flattened_attributes( self ) -> list[Variable]:
		return flattened_attributes( self )

@dataclass( kw_only = True )
class CUnion( Type, ScopeMixin ): # @cunion class Foo:
	type_params: list[TypeVar]|None = None
	attributes: list[Variable] = field( default_factory = list )
	methods: list['Function|Overload'] = field( default_factory = list )
	names: dict[str,Name] = field( default_factory = dict )
	resolve: Callable[[],None]|None = None

@dataclass( kw_only = True )
class TaggedUnion( Type, ScopeMixin ): # @union class Foo: ... , also the backing type for synthesized anonymous unions (X|Y)
	# each variant is an attribute: name -> type. Synthesized anonymous
	# unions are built fully-formed directly (never deferred, resolve stays
	# None); a user-declared @union's variants defer like any other class body.
	type_params: list[TypeVar]|None = None # if not None, this is a generic union (e.g. @union class Foo[T]:)
	attributes: list[Variable] = field( default_factory = list )
	methods: list['Function|Overload'] = field( default_factory = list )
	names: dict[str,Name] = field( default_factory = dict )
	resolve: Callable[[],None]|None = None

	def leaves( self ) -> list['Type']:
		result: list[Type] = []
		for attr in self.attributes:
			if attr.resolve is not None:
				attr.resolve()
			result.append( attr.type )
		return result

@dataclass( kw_only = True )
class CEnum( Type, ScopeMixin ): # @enum class Foo:
	value_type: Type
	next_auto: int = 0
	members: dict[str,int] = field( default_factory = dict )
	values: dict[int,str] = field( default_factory = dict )
	names: dict[str,Name] = field( default_factory = dict )
	resolve: Callable[[],None]|None = None

@dataclass( kw_only = True )
class CType( Type ):
	''' a C type defined in an external header, referenced by bare name.
	Used with compiler.c_type('pthread_mutex_t', header='pthread.h') -
	the emitter spells this as the raw C name, and compiler.sizeof(CType)
	emits sizeof(pthread_mutex_t) in the generated C. '''
	c_name: str          # e.g. 'pthread_mutex_t'
	required_header: str  # e.g. 'pthread.h'

# anything that can own methods/be a Function's .cls
ClassLike = Union[ RCClass, CStruct, CUnion, TaggedUnion, CEnum ]

@dataclass( kw_only = True )
class Function( Type, ScopeMixin ):
	cls: ClassLike|None
	node: ast.FunctionDef # whole def - node.args/.returns resolved lazily, node.body untouched until IR generation
	type_params: list[TypeVar]|None = None # if not None, this is a generic function (e.g. def alloc[T](...))
	parameters: list[Parameter]|None = None
	return_type: Type|None = None
	names: dict[str,Name] = field( default_factory = dict )
	# None means already resolved; otherwise call it to populate
	# parameters/return_type, after which it sets itself back to None
	resolve: Callable[[],None]|None = None

	is_static: bool = False
	is_classmethod: bool = False
	is_abstract: bool = False
	is_move: bool = False
	is_private: bool = False
	# @virtual - this method occupies a vtable slot (see
	# PLAN_SUBCLASSING_VTABLES_COM.md). Deliberately not CStruct-specific:
	# lives on Function itself, same as every other decorator flag here, so
	# RCClass can reuse the identical concept once that work resumes. An
	# override of an inherited @virtual method must repeat @virtual on its
	# own re-declaration - matching name+signature alone is not enough.
	is_virtual: bool = False

	# @extern('lib', 'symbol') - a foreign call signature declaration (body
	# must be a stub - see discovery.py's _is_stub_body). extern_lib is the
	# .lib/.so name to link against, except the literal 'c' which means the
	# platform C runtime rather than a real file on disk. Both None for an
	# ordinary function
	extern_lib: str|None = None
	extern_symbol: str|None = None
	extern_header: str|None = None # optional header that declares this @extern function; when included via require_header, the emitter skips the prototype

	is_overload: bool = False # was this def @overload-decorated (whether it ended up a stub or, with a real body, an Overload.implementations entry)
	bound_to: 'Function|None' = None # stubs only: the plain implementation this stub's signature resolves to (see discovery.py's _bind_overload_stub)
	is_destructor: bool = False # synthesized $$__destructor__ body — emitter uses void(void*) signature + cast prologue

	# @inline (PLAN_INLINE.md) - body is arbitrary statements followed by
	# exactly one final, top-level `return <expr>` (no other `return`
	# anywhere else, no defer/errdefer, no reassignment of self/a
	# parameter - discovery.py's _is_inline_eligible_body and its sibling
	# scanners enforce this at parse time); lowering.py splices the whole
	# body directly at each call site instead of ever emitting a real
	# Call/FuncStart/FuncEnd for it
	is_inline: bool = False

def _leaf_is_accepted( leaf: Type, declared: Type ) -> bool:
	# identity-based deliberately, not `==` - Type dataclasses have structural
	# equality (comparing every field, including mutable dicts/lists), which is
	# both wrong and expensive for "is this the same type" here. The existing
	# _get_or_create_union/_specialization/_move dedup caches already guarantee
	# "the same type" is the same object within one Discovery instance.
	return any( leaf is candidate for candidate in declared.leaves() )

def _is_covered_by( narrow: Function, wide: Function ) -> bool:
	'''
	every type `narrow` declares, at every parameter position, is accepted by
	`wide` at that same position. Used both for stub -> implementation binding
	and for @overload-vs-@overload shadowing (is a later one entirely dead
	code because an earlier one already covers everything it declares).
	'''
	if narrow.parameters is None or wide.parameters is None:
		return False # one of them failed to resolve - can't reason about coverage
	if len( narrow.parameters ) != len( wide.parameters ):
		return False
	return all(
		all( _leaf_is_accepted( leaf, wide.parameters[i].type ) for leaf in narrow.parameters[i].type.leaves() )
		for i in range( len( narrow.parameters ))
	)

def _overlaps( a: Function, b: Function ) -> bool:
	''' some concrete call could satisfy both `a` and `b` simultaneously. Used for plain-implementation-vs-plain-implementation ambiguity. '''
	if a.parameters is None or b.parameters is None:
		return False # one of them failed to resolve - can't reason about overlap
	if len( a.parameters ) != len( b.parameters ):
		return False
	return all(
		any( _leaf_is_accepted( leaf, b.parameters[i].type ) for leaf in a.parameters[i].type.leaves() )
		for i in range( len( a.parameters ))
	)

@dataclass( kw_only = True )
class ConditionalDispatch:
	''' one branch of a runtime overload dispatch: if the value(s) bound to each of `conditions` match their expected type, call `function`. '''
	conditions: list[tuple[Parameter,Type]]
	function: Function

@dataclass( kw_only = True )
class Overload( Type ):
	'''
	stands in for a Function when multiple defs share a name in the same scope.

	stubs: @overload-decorated defs with no real body - signature-only,
	never actually called; implementations: real bodies, either one plain
	(non-@overload) fallback that every stub maps to, or several
	@overload-decorated ones distinguished by their own parameters (each such
	member's .is_overload is True even though it lives in .implementations -
	see discovery.py's well-formedness checks, which treat "@overload members"
	as stubs + these combined, ordered by declaration line).

	Resolving an actual call (deciding which member(s) a given call site's
	argument types dispatch to) lives in overload_resolution.py, not here -
	see overload_resolution.resolve_call( group.stubs, group.implementations,
	args, kwargs, qualname = group.qualname ).
	'''
	cls: ClassLike|None = None
	stubs: list[Function] = field( default_factory = list )
	implementations: list[Function] = field( default_factory = list )

@dataclass( kw_only = True )
class Module( Name, ScopeMixin ):
	intrinsics: dict[str,Name]
	builtins: dict[str,Name]|None
	names: dict[str,Name] = field( default_factory = dict )
