# stdlib imports:
import ast
from dataclasses import dataclass, field
from itertools import count
from pathlib import Path
from typing import Callable, Union, TYPE_CHECKING

# local imports:
from errors import RedundantCompilationError
if TYPE_CHECKING:
	import ir
	import type_resolver

# backs Variable.uid - see its own comment
_variable_uid_counter = count()

@dataclass( kw_only = True )
class Name:
	stem: str # local name like 'str' instead of 'builtins.str'
	qualname: str # fully qualified name: 'builtins.str' instead of 'str'

	# we don't always know where a name is defined the first time we see it:
	file: Path|None
	line: int|None

	# set once, permanently, when this name's own creation/resolution
	# raised a CompileError - see Discovery._resolve_guarded and
	# lowering.py's per-kind equivalents. Never cleared: a broken symbol is
	# only ever attempted once (same convention .resolve = None already
	# follows). Checked by ScopeMixin.get_local_or_raise/Discovery.find_name
	# so a later reference raises RedundantCompilationError instead of
	# either using a half-built object or reporting a confusing second
	# error - the real one was already recorded at the point of failure.
	broken: bool = False

	def __deepcopy__( self, memo: dict ) -> 'Name':
		''' Name/Type/Function/Variable/Module/... instances are identity-
		based, process-wide-shared singletons (interned via Discovery.
		_get_or_create_specialization and friends, or simply constructed
		once at their own definition site) - a bare copy.deepcopy() call
		elsewhere in the compiler that happens to reach one of these
		(typically via an AST node's own cached resolved-reference tag,
		e.g. node.resolved_callee) must never clone it: doing so silently
		produces a SECOND, non-identical object with the same qualname,
		corrupting every identity-keyed cache downstream. Confirmed via a
		real repro: compiling two generic instantiations of one generator
		sharing a match-subject promoted RC-typed field crashed with a
		RecursionError inside TaggedUnion's own dataclass __eq__, comparing
		two non-interned but qualname-identical Result[i32,IndexError]
		objects - traced to type_resolver.py's _apply_live_flag_guards,
		whose own copy.deepcopy(s) on the promoted field's assignment
		statement swept along a resolved_callee-tagged Function reference
		and deep-cloned its entire return-type graph (recursively
		duplicating i32/IndexError themselves in the process). Returning
		self unchanged is exactly the correct semantics for a value this
		codebase already treats as immutable/interned everywhere else -
		every AST node itself still deep-copies normally; only a Name (or
		subclass) reached FROM one is short-circuited here. '''
		return self

@dataclass( kw_only = True, repr = False )
class Type( Name ):
	''' maybe only use this to distinguish types from values '''

	# the module this type was DEFINED in - set only for a genuine
	# module-level, user-defined class (RCClass/CStruct/CUnion/CEnum/
	# TaggedUnion parsed from real source, via discovery.py's own
	# _parse_ClassDef_* family) - mirrors Function.module/Variable.module
	# (see either's own comment for the motivating cross-module-
	# resolution use case). Left None for every other Type kind (Scalar,
	# Specialization, CallableType, a synthesized anonymous union, ...) -
	# none of those have a real "defining module" of their own. Used by
	# name/import-visibility enforcement (SYNTAX.md's `_x`/`__x` module/
	# package-privacy rules, extended to cover classes too) to compare the
	# DEFINING module against whichever module is doing the accessing.
	# repr=False/compare=False for the identical reason Variable.module's
	# own field is: two independently-constructed Discovery passes'
	# Module objects for "the same" module differ in identity/content, so
	# including this in dataclass equality/repr would make every Type-
	# comparing pass in this codebase (there are many) spuriously fragile.
	module: 'Module|None' = field( default = None, repr = False, compare = False )

	def __repr__( self ) -> str:
		# every Type subclass below opts out of the dataclass-generated repr
		# (repr=False) and inherits this one instead, deliberately never
		# recursing into another field. dataclasses' auto-repr is only guarded
		# against a field re-entering the SAME object already on the repr call
		# stack (reprlib.recursive_repr, keyed by id(self)) - it does nothing
		# for a DAG where the same object is reachable via multiple sibling
		# fields (e.g. Specialization.base and Specialization.args both
		# pointing at a shared prior type): each convergence re-expands the
		# whole subtree, so a chain of N such diamonds costs O(3^N) - a real,
		# reproduced hang (confirmed: depth 10 already produces a 9.7MB repr
		# in 88ms; the depth seen from a real compiler bug ran the process out
		# of 24+GB of RAM before ever raising). A mistyped Type value reaching
		# an assertion's error message must fail fast, not become a resource-
		# exhaustion trap - so this never walks into another Type's own fields.
		return f'<{type(self).__name__} {self.qualname!r}>'

	def leaves( self ) -> list['Type']:
		# a single concrete type is its own only leaf - TaggedUnion overrides
		# this to return its member types instead. shared by overload
		# matching (see Overload.resolve_call and the _is_covered_by/_overlaps
		# primitives below) to treat "a union" and "a plain type" uniformly.
		return [ self ]

	# --- RC classification -------------------------------------------------
	#
	# These live HERE, on each type kind, rather than as isinstance ladders in
	# whichever pass happens to need them, because the ladder version shipped
	# the same bug three separate times: a new Type kind appeared, the ladder
	# in cfg.py wasn't updated, and the new kind silently defaulted to "not RC"
	# (a nested TaggedUnion leaf -> reference leak; a generic union's
	# unsubstituted TypeVar leaves -> UAF; an unresolved TaggedUnion reporting
	# no leaves depending purely on compile ORDER -> UAF). The default is still
	# "no" below, but it's now a default a new subclass's author is looking
	# straight at, instead of one decided in a file they'd never open.
	#
	# Three DISTINCT questions, deliberately not collapsed into one - tuple[T...]
	# is the row that proves they can't be: it's RC and it's pointer-shaped, but
	# it has no ObjectHeader of its own (its synthesized BACKING class owns
	# that - see TupleType.backing / tuple_storage.py).

	def is_rc( self ) -> bool:
		''' this type's runtime representation carries reference-counted
		references SOMEWHERE inside it - i.e. something has to incref/decref
		when a value of this type is copied or dropped. True for a bare RC
		pointer, but ALSO for an aggregate that merely CONTAINS one (a
		TaggedUnion with any RC member), which is why this is not the same
		question as is_rc_pointer() below. '''
		return False

	def is_result_type( self ) -> bool:
		''' True when this type is a concrete Result[T,E] specialization -
		overridden on TaggedUnion, delegated on Specialization. '''
		return False

	def is_rc_pointer( self ) -> bool:
		''' this type's OWN runtime representation IS a single, bare RC
		pointer - so a Retain/Release can be applied to a value of this type
		DIRECTLY, and it can be spelled `struct <mangled>*` in C.

		Strictly narrower than is_rc(): a TaggedUnion's runtime shape is a
		tag+data VALUE STRUCT, not a pointer at all, so reading one through a
		bare-pointer accessor produces garbage even when it plainly does carry
		RC references. cfg.py's tag-gated refcount path exists precisely for
		the is_rc()-but-not-is_rc_pointer() case. '''
		return False

	def rc_leaves( self ) -> list['Type']:
		''' the distinct runtime slots of this type that need RC treatment.
		At most one (this type itself) for everything except a TaggedUnion,
		which has one per RC member. An empty list means "no RC work at all"
		and is what every automatic incref/decref site in cfg.py gates on. '''
		return [ self ] if self.is_rc() else []

	# --- memory layout (NOT RC - see the note above) ------------------------

	def has_object_header( self ) -> bool:
		''' a value of this type is a heap object that LEADS with
		`ObjectHeader $header` - the refcount field, plus the vtable pointer
		that RCClass's virtual dispatch and its destructor dispatch both read
		through (emitter_c.py's PROLOGUE). This is what separates the two
		vtable-pointer LOCATIONS: an RCClass reads `$header.vtable`, while an
		@interface CStruct has a plain top-level `$vtable` member instead.

		Deliberately False for TupleType even though it IS an RC pointer: the
		header belongs to the backing class tuple_storage.py synthesizes, not
		to the TupleType annotation itself. '''
		return False

	def has_vtable( self ) -> bool:
		''' this type dispatches @virtual methods through SOME vtable -
		true for every RCClass, and for an @interface CStruct (COM model).
		Says nothing about WHERE that vtable pointer lives (see
		has_object_header) nor whether this class needs its own synthesized
		Vtbl STRUCT TYPE (that's own_new_virtual_slots()'s question - see
		emitter_c.py's _rcclass_vtbl_type_name). '''
		return False

	# --- ownership annotations ---------------------------------------------

	def unwrap_ownership( self ) -> 'Type':
		''' the real runtime type this annotation describes - self for an
		ordinary type, and .inner for the move[T]/copy[T] wrappers, which are
		an ownership STATUS on a binding rather than a distinct type at all
		(see Move's own docstring). Call this before asking any of the
		questions above about a PARAMETER's declared type. '''
		return self

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

	def get_local_or_raise( self, name: str ) -> Name|None:
		''' like get_local, but raises RedundantCompilationError instead of
		handing back a name whose own creation/resolution already failed.
		Still returns None (not an error) for a name that's genuinely
		absent - only a caller that needs "this must exist" should keep
		failing on that separately, same as today. Every ordinary "look up
		a specific, known member on an already-in-hand scope object" call
		site should go through this instead of touching .names directly,
		so a broken member doesn't surface as a second, confusing failure
		downstream - get_local itself stays a raw, never-raising accessor,
		since tests rely on it to inspect a deliberately-broken object's
		state directly. '''
		found = self.get_local( name )
		if found is not None and found.broken:
			raise RedundantCompilationError()
		return found

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
		return base is self # type: ignore

	def in_protected_scope( self, scope: 'Type|None' ) -> bool:
		''' the `_protected` counterpart of in_private_scope: true if `scope`
		IS this class, or a (possibly indirect) subclass of it -
		SYNTAX.md's single-underscore field/method tier, visible to the
		defining class and every subclass (any depth). Only RCClass/CStruct
		(InheritanceChainMixin) have a real base chain to walk; every other
		ScopeMixin kind (TaggedUnion, Protocol, CUnion, CEnum - none support
		inheritance) degrades to the identical identity check in_private_
		scope already does, via the loop simply never starting and falling
		through to the plain `base is self` check below. '''
		base = scope.base if isinstance( scope, Specialization ) else scope
		node: 'InheritanceChainMixin|None' = base if isinstance( base, InheritanceChainMixin ) else None
		while node is not None:
			if node is self:
				return True
			node = _next_chain_node( node.base )
		return base is self # type: ignore

@dataclass( kw_only = True, repr = False )
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

def int_stem_range( t: Scalar ) -> tuple[int,int]:
	''' (MIN, MAX), the real inclusive range of integer stem t.stem, as
	Python ints - used to validate a literal's magnitude against its
	declared type (see lowering.py's _expr_Constant and discovery.py's
	_register_enum_member, the two places a literal's value gets checked
	against a concrete integer type). Derived from t.sizeof (already
	resolved to the ACTIVE TARGET's real width by the time either caller
	runs - see discovery.py's active_target-driven sizeof computation -
	isize/usize are NOT hardcoded to 64 here), not a fixed per-stem table,
	so this is correct for every integer stem uniformly, whatever target
	width the compiler was configured for. Signedness is read directly off
	the stem's own first letter (i vs u) - true for every integer stem
	this compiler has (i8/i16/i32/i64/i128/isize vs u8/u16/u32/u64/u128/
	usize) - rather than depending on lowering.py's own _SIGNED_INT_STEMS,
	which this module (mpy_types.py, imported by both discovery.py and
	lowering.py) can't reach without a circular import. '''
	bits = t.sizeof * 8
	if t.stem[0] == 'i':
		return -(2**(bits-1)), 2**(bits-1) - 1
	return 0, 2**bits - 1

@dataclass( kw_only = True, repr = False )
class TypeVar( Type ):
	''' a placeholder for one of a generic's type parameters, e.g. T in class Result[T,E] '''
	# TypeVar(bound=SomeProtocol) - the only bound kind supported (discovery.py's
	# _parse_type_params rejects anything else). Checked at generic
	# instantiation time via bound_satisfied_by below, from every call site
	# that substitutes a concrete type for this TypeVar (lowering.py's
	# _finish_generic_call/_lower_generic_function_call/_lower_class_generic_
	# method_call/_lower_generic_construction_args, and type_resolver.py's
	# _ReferenceResolver, which bails rather than hard-failing so lowering's
	# own copy reports the real error) - deliberately NOT in discovery.py's
	# _get_or_create_specialization, a low-level shared cache/builder with no
	# AST node/error-location available.
	# a Specialization here (e.g. Sequence[T], where T is the SAME
	# generic's own sibling type param - S: Sequence[T]) is a parametrized
	# protocol bound - see bound_satisfied_by below for how the sibling
	# T gets resolved to something concrete before checking.
	bound: 'Protocol|Specialization|None' = None

	def bound_satisfied_by(
		self, concrete: Type,
		type_params: 'list[TypeVar]|None' = None,
		args: 'list[Type]|None' = None,
		resolver: 'type_resolver.TypeResolver|None' = None,
	) -> bool:
		''' True if `concrete` may be substituted for this TypeVar. Always
		true when unbound. When bound to a bare Protocol, `concrete` must be
		an RCClass that has ALREADY had its own conformance verified - this
		is just a membership check against RCClass.protocols, not re-doing
		the structural verification discovery.py's _validate_protocol_
		conformance already did once, at that class's own definition.

		When bound to a Specialization-of-Protocol (S: Sequence[T]) - a
		PARAMETRIZED bound - `type_params`/`args`/`resolver` are required:
		`type_params`/`args` are THIS call site's full, already-resolved
		zip of every sibling type param this generic declares (confirmed
		available at every real call site by the time bound-checking runs -
		see lowering.py's _finish_generic_call), used to substitute the
		bound's own args (e.g. T -> the concrete element type this call
		site bound T to) via resolver.monomorphizer.substitute_type_params -
		the SAME substitution primitive monomorphization itself uses, not
		new machinery. The result is compared structurally (resolver.
		_same_type, not `==` - see that method's own docstring for why a
		bare dataclass equality is unsafe/wrong here) against `concrete`'s
		own declared protocol-conformance Specializations - already
		substituted/concrete themselves (Monomorphizer.monomorphize_class
		substitutes a generic class's OWN declared protocol args, e.g.
		list[i32]'s `Sequence[T]` conformance reads back as `Sequence[i32]`,
		at the same point/via the same mechanism .attributes/.methods are -
		see its own comment), so no second substitution is needed on that
		side. '''
		if self.bound is None:
			return True
		base = concrete.base if isinstance( concrete, Specialization ) else concrete
		if isinstance( base, TaggedUnion ):
			# a union receiver (e.g. `t: bytes|bytearray`) has no .protocols
			# of its own (TaggedUnion never populates that field - only a
			# real RCClass leaf does), so falling straight through to the
			# `not isinstance(base, RCClass): return False` check just below
			# unconditionally rejected EVERY union argument at ANY protocol-
			# bound generic call site, even when every one of its leaves
			# individually conforms - confirmed via a real repro (len(x:
			# bytes|bytearray) inside lib/builtins/__init__.py's own
			# bytes.__init__, once len[T] gained a T: Sized bound: both
			# leaves conform, the bare union doesn't, and used to fail
			# outright instead of checking per-leaf). Mirrors how a union
			# receiver's own METHOD dispatch (lowering.py's
			# _lower_union_receiver_call) already requires every leaf to
			# define the method, not the union as a whole - same principle,
			# applied here to a TypeVar's own protocol bound instead of an
			# attribute/method lookup.
			return all( self.bound_satisfied_by( leaf, type_params, args, resolver ) for leaf in base.leaves() )
		if isinstance( base, TupleType ):
			# tuple[...] isn't an RCClass - its declared protocol
			# conformance (for homogeneous tuples) lives on its lazily-
			# synthesized backing RCClass instead (see tuple_storage.py)
			base = base.backing
		if isinstance( base, TypeVar ):
			# concrete is itself a not-yet-substituted, already-bounded type
			# param (e.g. a generic's own field/parameter, referenced from a
			# call site that hasn't been monomorphized to a concrete type
			# yet) - its own bound already proves conformance, no need to
			# re-derive it structurally (which would always fail: a bare
			# TypeVar is never an RCClass).
			if base.bound is None:
				return False
			return resolver._same_type( base.bound, self.bound ) if resolver is not None else base.bound is self.bound
		if not isinstance( base, RCClass ):
			return False
		if not isinstance( self.bound, Specialization ):
			return self.bound in base.protocols
		assert resolver is not None and type_params is not None and args is not None, (
			f'a parametrized protocol bound ({self.bound.qualname}) requires the caller to pass '
			f'(type_params, args, resolver) - see bound_satisfied_by\'s own docstring'
		)
		concrete_bound = resolver.monomorphizer.substitute_type_params( self.bound, type_params, args )
		# when `concrete` is a Specialization, `base` above is still the
		# ABSTRACT class template (concrete.base) - its own .protocols are
		# still unsubstituted (e.g. Sequence[set.T], not Sequence[i32]).
		# Substitute each candidate entry's own args against concrete.args
		# directly here, rather than going through monomorphize_class(concrete)
		# first to get an already-substituted base (the old approach): that
		# needed a _building reentrancy guard (concrete's own class building
		# its own methods, one of which - e.g. a Sequence-conforming class's
		# own __iter__ - reaches back here to check ITS OWN class's
		# conformance, re-entering monomorphize_class for the same
		# still-mid-construction spec), which fell back to comparing against
		# the UNSUBSTITUTED abstract .protocols and silently failed the bound
		# check for that one call (confirmed by a real repro: builtins.set/
		# tuple's own __iter__ delegating to a shared, protocol-bound generic
		# generator never got past this once reached from within its own
		# class's build). A protocol entry's own .base is always the
		# Protocol, never `concrete`'s class, so substituting just its args
		# can't re-enter monomorphize_class for `concrete` at all - no guard
		# needed, and no eager full build of `concrete` needed either.
		for entry in base.protocols:
			if not isinstance( entry, Specialization ) or entry.base is not concrete_bound.base:
				continue
			entry_args = entry.args
			if isinstance( concrete, Specialization ) and resolver is not None:
				entry_args = [
					resolver.monomorphizer.substitute_type_params( a, base.type_params or [], concrete.args )
					for a in entry_args
				]
			if len( entry_args ) == len( concrete_bound.args ) and all(
				resolver._same_type( a, b ) for a, b in zip( entry_args, concrete_bound.args )
			):
				return True
		return False

@dataclass( kw_only = True, repr = False )
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

	# every RC/layout question about Box[i32] is really a question about Box -
	# a generic CLASS's RC-ness comes from being an RCClass with a header,
	# never from what its type params happen to be bound to, so delegating to
	# self.base is correct there. This delegation is what retires the `base =
	# t.base if isinstance( t, Specialization ) else t` idiom that used to be
	# copy-pasted verbatim in 14 places across cfg/lowering/emitter_c/type_
	# resolver, each site independently deciding whether to ALSO handle Move/
	# Copy/TupleType. Without it, every generic-class/generic-union instance
	# method's own `self` (already typed as a Specialization) wrongly looks
	# untracked.
	#
	# A generic UNION (Result[T,E]) is different: is_rc() GENUINELY depends
	# on what T/E are bound to, not just on being a TaggedUnion - self.base
	# alone (the abstract, unsubstituted union) always reports non-RC
	# (rc_leaves()'s own comment: "a bare TypeVar is never RC" - Type.is_rc()'s
	# own default), regardless of self.args, so a bare `self.base.is_rc()`
	# here made EVERY Result[SomeRCType,SomeRCType]-typed value look non-RC.
	# Confirmed via a real reference leak: a generator's own promoted
	# Result[Box,StopIteration]-typed field (yield-from's own raw next()
	# result, crossing a suspend) was silently excluded from the state-aware
	# destructor's own RC-field cascade entirely, since _build_generator_
	# destructor gates each field on is_rc(). rc_leaves() already does the
	# substitution correctly (same method, see its own docstring) - reuse it
	# rather than duplicating the substitution logic here.
	def is_rc( self ) -> bool:
		if isinstance( self.base, TaggedUnion ):
			return bool( self.rc_leaves() )
		return self.base.is_rc()
	def is_result_type( self ) -> bool: return self.base.is_result_type()
	def is_rc_pointer( self ) -> bool: return self.base.is_rc_pointer()
	def has_object_header( self ) -> bool: return self.base.has_object_header()
	def has_vtable( self ) -> bool: return self.base.has_vtable()

	def rc_leaves( self ) -> list['Type']:
		''' the one place ORDER matters: substitution has to happen BEFORE the
		is_rc() filter.

		A Specialization of a still-GENERIC TaggedUnion (Result[str,MyError])
		has an abstract base whose leaves() returns Result's OWN declared field
		types verbatim - bare TypeVars T/E - and a bare TypeVar is never RC.
		Filtering first therefore returns [] for EVERY generic-union
		instantiation regardless of what T/E were actually bound to, which is
		how a Result[str,E]'s own payload went entirely untracked. Confirmed by
		a real UAF: a temp Result[str,E] receiver of .unwrap()/.unwrap_or() was
		never registered by fresh_temp at all, which is what made the old
		receiver-move workaround in lowering.py's _lower_call look load-bearing
		(it was popping a Temp that had never been inserted - already a no-op)
		while this, the real gap, went unnoticed. '''
		# a NON-union Specialization's leaf is SELF, not self.base: everything
		# downstream (Incref/Decref operands, temp registration) needs the
		# concrete instantiation, never the abstract template
		if not isinstance( self.base, TaggedUnion ):
			return [ self ] if self.is_rc() else []
		leaves = self.base._resolved_leaves()
		if self.base.type_params:
			# Shallow (one level) on purpose: a leaf that's instead e.g.
			# `list[T]` needs no T resolved at all to know list itself is RC
			# (is_rc only reads a Specialization's own .base), and a leaf
			# that's already a fixed concrete type is correct as-is.
			# Identity-keyed deliberately, not by value - Type dataclasses have
			# structural equality (see _leaf_is_accepted's own comment).
			substitution = { id( param ): arg for param, arg in zip( self.base.type_params, self.args ) }
			leaves = [ substitution.get( id( leaf ), leaf ) for leaf in leaves ]
		return [ leaf for leaf in leaves if leaf.is_rc() ]

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
	# the module this Variable was DEFINED in - set from Discovery.module_
	# stack at parse time, mirroring Function.module (see that field's own
	# comment for the motivating cross-module-resolution use case). Only
	# meaningful (and only ever set) for a genuine module-level global
	# (is_global=True); a class attribute or lowering-built local has no
	# owning Module of its own. Used by name/import-visibility enforcement
	# (SYNTAX.md's `_x`/`__x` module/package-privacy rules) to compare the
	# DEFINING module against whichever module is doing the accessing.
	# repr=False/compare=False for the identical reason `uid` below is:
	# Module.names accumulates differently across separately-constructed
	# Discovery passes even for "the same" module, so including it in
	# dataclass equality would make lowering_test.py's exact-IR structural
	# comparisons spuriously fragile - this field is read directly by
	# enforcement logic, never compared/printed.
	module: 'Module|None' = field( default = None, repr = False, compare = False )
	# set by cfg.py's assign() the moment this global is genuinely
	# reassigned from inside a function body (`global X; X = ...`) - never
	# set for a global's own module-level initializer, since lower_global()
	# (lowering.py) emits that ir.Assign directly, bypassing cfg.assign()
	# entirely (see cfg.py's own is_global branch for why). A global that's
	# only ever written by its own initializer never sees this flip to
	# True - PLAN_THREAD_SAFE_SHARED_STATE.md's A.1: provably single-write,
	# no lock needed. Only meaningful for is_global=True RC-typed globals.
	reassigned_outside_init: bool = False
	# per-field analog of the above (PLAN_THREAD_SAFE_SHARED_STATE.md Cost
	# mitigation #2) - flipped by lowering.py's non-construction SetAttr
	# branches (mirroring cfg.py's assign() flipping reassigned_outside_init
	# for a global) the moment a class attribute Variable is written from
	# anywhere OTHER than its own class's __init__. Separate field, not a
	# reused reassigned_outside_init: that one's is_global=True precondition
	# and this one's __private-scoped, __init__-relative precondition are
	# different questions that happen to share a "provably single-writer"
	# shape - conflating them would let one's flip silently satisfy the
	# other's very different soundness requirement. Only meaningful for a
	# class-attribute Variable (never set for is_global=True/local Variables).
	field_reassigned_outside_init: bool = False
	# set only for a local declared `Volatile[T]` (_stmt_AnnAssign) - means
	# its C storage must be qualified `volatile` (see emitter_c._declarator)
	is_volatile: bool = False
	# set only for a @cstruct/@cunion field declared `Aligned[N, T]`
	# (discovery.py's _apply_aligned_annotation) - N overrides just THIS
	# field's own C alignment, independent of the owning struct's
	# CStruct.packed. Works in either direction (N below or above T's own
	# natural alignment) - emitter_c.py's _struct_or_union_body emits a
	# compiler-conditional pair, both verified empirically on real MSVC/
	# clang/gcc: MSVC needs #pragma pack(push,N)/pop (a cap - shrink only)
	# LAYERED WITH __declspec(align(N)) on the field (grow only) since
	# neither alone covers both directions; clang/gcc's single GNU
	# __attribute__((packed,aligned(N))) already covers both on its own.
	c_align: int|None = None
	# stage 2's lowered form of `init` (None until Compiler._lower's Variable
	# branch runs) - kept directly on the Variable itself, not only reachable
	# through compiler.globals' own LoweredGlobal list, so a global's own
	# initializer instructions travel with the variable object (see
	# PLAN_GLOBAL_INIT.md). String-quoted to avoid a mpy_types<->ir import
	# cycle (ir.py doesn't need to know about Variable at all).
	init_instructions: list['ir.Instruction']|None = None
	# assigned once, at construction, from a process-wide counter - lets
	# emitter_c.py give an independently-declared local its own C
	# identifier (stem + '$' + uid) instead of colliding by source name
	# alone, when it actually needs one (see needs_uid_suffix below).
	# repr=False/compare=False: lowering_test.py compares IR instructions
	# both by their exact string repr AND via assertEqual's own structural
	# dataclass equality - a process-wide counter value has no business
	# being part of either (order-dependent, not what those tests are
	# checking; two separately-constructed-but-conceptually-equal Variable
	# objects would otherwise never compare equal again).
	uid: int = field( default_factory = lambda: next( _variable_uid_counter ), repr = False, compare = False )
	# NOT set at construction - mutated in place by emitter_c.py the first
	# time it sees this Variable declared/assigned, if-and-only-if this
	# base C name is already occupied by an INCOMPATIBLE (different type
	# or volatility) prior declaration in the same function (e.g. `del x;
	# x: T2 = ...` redeclaring x with an incompatible type - see del_
	# reuse_and_emitter_naming_bug). Deliberately NOT set just because a
	# Variable is "a local" in the abstract: two independently-declared
	# Variable objects sharing both a stem AND a compatible type (e.g. the
	# same `x: u32 = ...` re-declared, once per arm, across a plain if/
	# elif/else chain - a common, working pattern, see lib/builtins/
	# __File.py's own `creation` local) are INTENDED to share one piece of
	# C storage, exactly as they did before this field existed; suffixing
	# them unconditionally would silently split that one shared C variable
	# into several, only one of which the merged control flow actually
	# initializes on any given path. repr=False/compare=False for the same
	# reasons as uid above (also: this is emission-time-derived, mutated
	# well after lowering.py hands the IR off, not part of the IR's own
	# structural identity).
	needs_uid_suffix: bool = field( default = False, repr = False, compare = False )

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
	# move[T]/copy[T] in the ORIGINAL annotation - an ownership status on
	# this binding, not a distinct type (see Move/Copy's own docstrings).
	# `type` itself is always the unwrapped, real T: discovery.py's own
	# parameter-construction site strips the Move/Copy wrapper and records
	# the ownership fact here instead, so every ordinary consumer (
	# attribute/method lookup, generic inference, assignability checks)
	# sees a plain T like any other binding - only the two call-site-
	# specific concerns (does the caller need to write move(x)? does the
	# CFG owe this binding its own decref?) consult these flags directly.
	is_move: bool = False
	is_copy: bool = False

def _ownership_annotation_error( t: 'Move|Copy', question: str ) -> AssertionError:
	''' move[T]/copy[T] are an ownership STATUS on a binding, not types (see
	Move's own docstring) - asking one whether it's reference-counted, or how
	it's laid out in memory, is a category error, and the only honest answer
	is that whoever asked is holding a PARAMETER's declared annotation where
	they meant to hold a real runtime type.

	Deliberately raises rather than politely delegating to .inner. Delegating
	would make every such call quietly WORK, which permanently hides whether
	any path in the compiler treats an ownership annotation as a runtime type
	- and a wrapper reaching, say, cfg.py's refcount emission is a genuine
	bug worth seeing, not something to paper over. Callers that legitimately
	hold one (cfg.py's _enter_parameter, which needs the Move/Copy-ness
	itself to pick an OwnState) call .unwrap_ownership() first.

	AssertionError, not CompileError: there is no user error to report here -
	CompileError's contract is that the failure is already recorded in an
	ErrorCollector - this is strictly a compiler-internal invariant. '''
	return AssertionError(
		f'{type(t).__name__}[{t.inner.qualname}] was asked {question}() - '
		f'move[T]/copy[T] are ownership annotations, not types. '
		f'Call .unwrap_ownership() first.'
	)

@dataclass( kw_only = True, repr = False )
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

	def unwrap_ownership( self ) -> Type:
		return self.inner

	# see _ownership_annotation_error
	def is_rc( self ) -> bool: raise _ownership_annotation_error( self, 'is_rc' )
	def is_rc_pointer( self ) -> bool: raise _ownership_annotation_error( self, 'is_rc_pointer' )
	def rc_leaves( self ) -> list[Type]: raise _ownership_annotation_error( self, 'rc_leaves' )
	def has_object_header( self ) -> bool: raise _ownership_annotation_error( self, 'has_object_header' )
	def has_vtable( self ) -> bool: raise _ownership_annotation_error( self, 'has_vtable' )

@dataclass( kw_only = True, repr = False )
class Copy( Type ):
	''' `copy[T]` in annotation position - the callee wants its own
	independent reference (an explicit INCREF in its own prologue, a
	matching DECREF at its own exit), regardless of whatever the caller
	already holds. Unlike move[T], this is a unilateral request: no
	call-site marker is needed/allowed (SYNTAX.md) - the caller's own
	binding is completely unaffected. See TODO.txt/RC MANAGEMENT.md for
	the CFG work this exists for. '''
	inner: Type

	def unwrap_ownership( self ) -> Type:
		return self.inner

	# see _ownership_annotation_error
	def is_rc( self ) -> bool: raise _ownership_annotation_error( self, 'is_rc' )
	def is_rc_pointer( self ) -> bool: raise _ownership_annotation_error( self, 'is_rc_pointer' )
	def rc_leaves( self ) -> list[Type]: raise _ownership_annotation_error( self, 'rc_leaves' )
	def has_object_header( self ) -> bool: raise _ownership_annotation_error( self, 'has_object_header' )
	def has_vtable( self ) -> bool: raise _ownership_annotation_error( self, 'has_vtable' )

@dataclass( kw_only = True, repr = False )
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

@dataclass( kw_only = True, repr = False )
class FixedArrayType( Type ):
	''' `ElemType[N]` used as a @cstruct/@cunion FIELD annotation only
	(SYNTAX.md's "Fixed-Size Inline Array": `u16[32]`, `u8[8]`) - a real,
	fixed-size C array embedded inline in the struct body (`uint16_t
	name[32];`), not a heap-allocated/RC sequence the way list[T] is.
	Recognized textually in visit_Subscript (a non-generic base type
	subscripted by a bare positive integer constant, as opposed to a real
	generic type argument - see its own comment), interned by discovery.py's
	_get_or_create_fixed_array the same way CallableType/TupleType already
	are, so two annotations spelling the same (element type, count) share
	one object.

	Deliberately NOT a general-purpose value type: C's own array declarator
	syntax is discontinuous ("TYPE NAME[N]", not a plain prefix type the
	way every other field is spelled) and a bare C array is not assignable
	via `=` at all (only a WHOLE containing struct/union is, or an explicit
	memcpy). A value of this type is never read/written as a whole (`x =
	arr` / `dest->field = arr` are both rejected, matching real C, rather
	than silently emitting invalid C) - only three real operations exist:
	(1) the class-body compound-literal construction path (a `= 0` field
	default or an explicit `ClassName(field=0)` argument, meaning "zero-
	fill the whole array" - the one shape a C designated initializer
	`.field = {0}` can express), (2) element-level indexed read/write
	(`f.arr[i]`, both directions - ir.GetAttrIndex/ir.SetAttrIndex), and
	(3) compiler.addrof(x.arr) -> Ptr[ElemType] at the array's own start,
	via C's own array-to-pointer decay (ir.ArrayFieldPtr). Parameter/
	return/local-variable annotations of this type are rejected outright -
	only a @cstruct/@cunion field. '''
	elem_type: Type
	count: int

@dataclass( kw_only = True, repr = False )
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

	def is_rc( self ) -> bool:
		# PLAN_TUPLE.md: unlike an ordinary generic (list[T]/Result[T,E]/...),
		# where the ABSTRACT template class itself (Specialization.base)
		# already answers "is this RC" without ever needing to monomorphize a
		# specific instantiation, a TupleType has no such template - the only
		# place "is a tuple RC" lives is its own synthesized backing RCClass
		# (tuple_storage.py), which may not have been synthesized yet for this
		# particular TupleType (a local variable's own declared annotation type
		# is never independently re-resolved after discovery.py first builds it
		# - see emitter_c.py's c_type() for the identical "found by a real
		# hang, not anticipated up front" gap this mirrors). No lazy check
		# needed though: EVERY TupleType's backing is unconditionally an
		# RCClass by construction (tuple_storage.TupleStorage.get() never
		# produces anything else), so this is a structural guarantee, not
		# something that depends on whether .backing happens to be populated.
		return True

	def is_rc_pointer( self ) -> bool:
		return True

	# NOT has_object_header: the header belongs to the backing class, not to
	# this annotation. The emitter never allocates a TupleType directly - it
	# allocates the backing RCClass, which answers True on its own behalf.

@dataclass( kw_only = True, repr = False )
class GeneratorType( Type ):
	''' `Iterator[Result[T,E]]` or `Generator[T,E]` (E always includes
	StopIteration - PLAN_GENERATORS.md's StopIteration reversal) in a
	function's own return annotation - both spellings build the exact same
	GeneratorType from here on, they're purely syntactic alternatives (see
	discovery.py's visit_Subscript). Recognized textually, same posture as
	CallableType/TupleType above. Deliberately NOT interned/shared the way
	those are: two unrelated generator functions both declaring the same
	elem_type/error_type still need two independent backing RCClasses (each
	function's own, private state machine/fields) - unifying them by
	elem_type alone would wrongly conflate two functions' unrelated local
	state. A fresh GeneratorType is built for every annotation occurrence;
	.backing is populated once lowering.py recognizes the owning Function
	actually contains a `yield` and synthesizes its backing class (keyed
	to that one Function, not to this type). '''
	elem_type: Type
	# error_type is never actually None for any LEGALLY constructed
	# GeneratorType (discovery.py's visit_Subscript requires StopIteration
	# among its leaves unconditionally) - the field keeps its Optional type/
	# default purely so nothing else in this dataclass's own construction
	# needs simultaneous updating; __next__ always returns
	# Result[elem_type, error_type], never a bare nullable elem_type|None
	error_type: 'Type|None' = None
	send_type: 'Type|None' = None # None: no .send() support; set (PLAN_GENERATORS.md Phase C, Generator[T,SendType,E] - 3 type args): (yield expr) is usable as an EXPRESSION evaluating to plain SendType, delivered via .send(v) - the backing method becomes $$__resume__ instead of $$__next__, with thin __next__()/send(v) wrappers over it
	backing: 'RCClass|None' = None

# a class's own body scan (revealing its attribute/method *names*) is
# deferred behind .resolve, exactly like a Function's parameters or a
# Variable's type - nothing about a class's members is known until something
# calls it. type_params is the one exception: it's parsed eagerly, at
# creation time, because external code subscripting this class as a generic
# (Result[i32,usize]) needs to see it before this class's own .resolve ever runs.

def _next_chain_node( base: 'InheritanceChainMixin|Specialization|None' ) -> 'InheritanceChainMixin|None':
	''' one step of a single-inheritance chain walk - unwraps a still-abstract
	generic base (RCClass.base holding a Specialization, e.g. Real[T] where T
	is the SUBCLASS's own not-yet-bound TypeVar, or even a fully concrete
	Real[i32] that just hasn't been monomorphized in place yet - see
	discovery.py's _parse_ClassDef_RCClass/on_generic_base_resolved) down to
	the underlying abstract template class itself (Real). Every chain-walking
	method in InheritanceChainMixin below only ever needs NAMES/existence
	from an ancestor, never a substituted TYPE (that's Monomorphizer's job -
	see monomorphize_class's own eager .base substitution step, and
	Discovery.on_generic_base_resolved for the already-concrete case) - so
	discarding the Specialization's own .args here and continuing the walk
	from its abstract .base is always correct for what these methods answer.
	A fully-monomorphized/normalized class never has a Specialization here at
	all, so this is a no-op for every chain that doesn't involve an abstract
	generic ancestor. '''
	if isinstance( base, Specialization ):
		assert isinstance( base.base, InheritanceChainMixin ), f'invalid {base.base=}'
		return base.base
	return base

class InheritanceChainMixin( ScopeMixin ):
	'''
	shared single-inheritance-chain / vtable behaviour for the two class kinds
	that have one: RCClass and CStruct. Both grew an identically-shaped
	.base/.methods/.attributes/.names/.resolve (single inheritance,
	own-members-only .names, lazy .resolve) independently, and neither is a
	base of the other - ClassLike, below, is a plain Union type ALIAS, not a
	class - so these five algorithms used to live as module-level free
	functions parameterized over the union type 'RCClass|CStruct', with each
	class carrying a same-named method that did nothing but forward to the
	matching one. That is what a mixin is for; the union annotation was the
	type system being asked to assert a shared shape that no type expressed.

	Not a dataclass, and it declares no fields of its own - exactly the
	discipline ScopeMixin follows above, and for the same reason: RCClass and
	CStruct are @dataclass( kw_only = True ), and a mixin contributing real
	fields would interfere with their own field collection/ordering. The
	attribute lines below are bare ANNOTATIONS (no assignment), purely so the
	methods here can be read without chasing what .base/.methods/.attributes
	are; each real class declares them as actual dataclass fields.

	Mixed in AFTER Type (class RCClass( Type, ScopeMixin, InheritanceChainMixin ))
	to match the existing base order. Nothing here shadows a Type method, so
	that ordering is safe - note it would NOT be if this defined, say,
	has_vtable(): Type comes first in the MRO and would win. has_vtable()
	therefore stays declared on Type and overridden on each class directly,
	which is also the honest place for it - RCClass is always True, a CStruct
	only when @interface, so the two genuinely differ and there is no shared
	answer to hoist.

	RCClass.base specifically may ALSO be a Specialization (Real[T] or
	Real[i32] - a generic ancestor, still abstract or already concrete - see
	discovery.py's _parse_ClassDef_RCClass) rather than a plain RCClass|None
	- CStruct.base never is (CStruct inheritance stays restricted to
	@interface, unrelated to this). Every walk below only ever needs an
	ancestor's NAMES (existence, dispatch), never a substituted TYPE, so
	_next_chain_node's unwrap-to-the-abstract-template is always correct
	here - a real, substituted view of a generic ancestor's ATTRIBUTE TYPES
	is Monomorphizer's job (monomorphize.py's monomorphize_class, which
	substitutes/monomorphizes .base itself, and Discovery.
	on_generic_base_resolved for the already-concrete non-generic-subclass
	case), not something these chain-walking methods ever attempt.
	'''
	base: 'InheritanceChainMixin|Specialization|None'
	names: dict[str,Name]
	methods: list['Function|Overload']
	attributes: list['Variable']
	resolve: Callable[[],None]|None

	def resolve_chain( self ) -> None:
		''' resolve self and every ancestor in the single-inheritance chain
		(self, then .base, then .base.base, ... until None) - needed before
		trusting any level's .attributes/.names are populated (e.g.
		flattened_attributes(), whose own docstring warns it resolves
		nothing it returns) when there's no name being searched for to
		drive the walk the way chain_lookup's own walk does. Same per-level
		resolve() call chain_lookup already performs, factored out so a
		caller that doesn't have (or want) a name to look up can still get
		the walk's resolving side effect. '''
		node: 'InheritanceChainMixin|None' = self
		while node is not None:
			if node.resolve is not None:
				node.resolve()
			node = _next_chain_node( node.base )

	def chain_lookup( self, name: str ) -> Name|None:
		''' walk this class's own single-inheritance chain (self, then base,
		then base.base, ... until None) looking for `name` - .names only ever
		holds a class's OWN declared members (discovery.py never merges a
		base's own names into a subclass), so a subclass needs this to see an
		inherited method/attribute at all. Only ever non-trivial for a class
		that actually has a base (a plain, non-@interface CStruct can't have
		one at all - see discovery.py's _parse_ClassDef_CStruct, which rejects
		bases outright for that case). '''
		node: 'InheritanceChainMixin|None' = self
		while node is not None:
			if node.resolve is not None: # each level's .names is populated lazily, same "None means already resolved" convention as everywhere else - a base's own body may not have run yet just because the derived class's own resolve() (already done by the caller) ran
				node.resolve()
			found = node.get_local_or_raise( name ) # every real InheritanceChainMixin (RCClass/CStruct) is also a ScopeMixin
			if found is not None:
				return found
			node = _next_chain_node( node.base )
		return None

	def field_owner( self, name: str ) -> 'InheritanceChainMixin|None':
		''' the class in this chain whose OWN .names directly declares
		`name` (not merely inherits it) - the "defining class" field-
		visibility enforcement (Discovery.check_field_visibility) checks
		against, since chain_lookup finds a field for any receiver in the
		chain but a `_`/`__` field's privacy is scoped to where it was
		actually DECLARED, not the concrete receiver type it's read
		through. Same walk as chain_lookup, minus the resolved Variable. '''
		node: 'InheritanceChainMixin|None' = self
		while node is not None:
			if node.resolve is not None:
				node.resolve()
			if node.get_local( name ) is not None:
				return node
			node = _next_chain_node( node.base )
		return None

	def own_new_virtual_slots( self ) -> list['Function']:
		''' this class's OWN @virtual methods that AREN'T already a slot
		somewhere in its ancestor chain - i.e. genuinely NEW vtable slots
		introduced here, not overrides of an inherited one. Any level in the
		chain can introduce new slots (not just the root - see vtbl_owner's
		own docstring for why: real interface hierarchies routinely add
		methods at every level, e.g. IUnknown -> ICustom (adds methods) ->
		ConcreteImpl, which a root-only-introduces-slots rule can never
		express). '''
		if self.resolve is not None:
			self.resolve()
		inherited_names: set[str] = set()
		node = _next_chain_node( self.base )
		while node is not None:
			if node.resolve is not None:
				node.resolve()
			inherited_names.update( m.stem for m in node.methods if isinstance( m, Function ) and m.is_virtual )
			node = _next_chain_node( node.base )
		return [ m for m in self.methods if isinstance( m, Function ) and m.is_virtual and m.stem not in inherited_names ]

	def vtbl_owner( self ) -> 'InheritanceChainMixin':
		''' the nearest class at or above self (self itself, or walking up
		.base) whose own Vtbl C struct type is the one self's own $vtable field
		actually points at - the nearest one (starting from self) that
		introduces at least one genuinely new slot (see own_new_virtual_slots).
		A class that adds nothing of its own (pure overrides, or no @virtual
		methods at all) simply reuses whatever ancestor's Vtbl type is already
		in effect - matches real COM: FooImpl (an ordinary implementation, no
		new capabilities) still has an $vtable field literally typed as
		whichever interface it implements' own IFooVtbl*, not a FooImplVtbl of
		its own.

		RCClass and CStruct each override this to narrow the RETURN type to
		themselves - the one thing a shared method genuinely can't express.
		typing.Self would be wrong here rather than merely awkward: ClosureType
		is an RCClass subclass, and walking up .base from one can legitimately
		land on a plain RCClass ancestor, so the result is "the same class
		KIND", not "the same class". Those two narrowing overrides are all that
		remains of what used to be ten forwarding stubs. '''
		node = self
		while (next_node := _next_chain_node( node.base )) is not None and not node.own_new_virtual_slots():
			node = next_node
		return node

	def virtual_slots( self ) -> list['Function']:
		''' the full, ordered slot list for THIS class's own EFFECTIVE vtable
		type (vtbl_owner()'s own type) - every new-slot-introducing ancestor's
		own slots, root-first, up to and including vtbl_owner() itself
		(.methods is append-only in source order - see discovery.py's
		_parse_function, so each level's own contribution is already
		declaration-ordered). This is a superset walk, not "only the root" -
		see vtbl_owner's own docstring on why every level can contribute. '''
		chain: list['InheritanceChainMixin'] = []
		node: 'InheritanceChainMixin|None' = self.vtbl_owner()
		while node is not None:
			chain.append( node )
			node = _next_chain_node( node.base )
		slots: list[Function] = []
		for node in reversed( chain ):
			slots.extend( node.own_new_virtual_slots() )
		return slots

	def flattened_attributes( self ) -> list['Variable']:
		''' every attribute declared anywhere in self's own single-inheritance
		chain (self itself, then .base, then .base.base, ... until None),
		base-first/most-derived-last order - the same order emitter_c.py's own
		emit_rcclass/emit_cstruct field-flattening walk and type_resolver.py's
		_synthesize_rcclass_destructor use for real struct layout. .attributes
		alone only ever holds a class's OWN declared fields (discovery.py never
		merges a base's own fields into a subclass) - this is what a
		subclass's field=value construction sugar (no __init__ at all
		anywhere in the chain) and super().__init__() chaining (the base's
		own portion specifically - see lowering.py's own caller) both need
		instead.

		NOTE: unlike own_new_virtual_slots above, this resolves NOTHING it
		returns - a caller that goes on to ask an attribute about its own
		.type has to resolve it first (see cfg.py's complete_base_construction,
		which does exactly that, and says why).

		NOTE 2: if some level's own .base is still an ABSTRACT generic
		Specialization (Real[T], T not yet bound to anything concrete - see
		discovery.py's _parse_ClassDef_RCClass), the attribute contributed
		from that ancestor is the ABSTRACT, unsubstituted one (Real's own
		bare TypeVar-typed field) - substituting it against a concrete
		instantiation is the caller's job (see monomorphize.py's
		substituted_field), same as it already is for a class's own fields
		declared directly with its own type params. '''
		chain: list['InheritanceChainMixin'] = []
		node: 'InheritanceChainMixin|None' = self
		while node is not None:
			chain.append( node )
			node = _next_chain_node( node.base )
		attrs: list[Variable] = []
		for node in reversed( chain ):
			attrs.extend( node.attributes )
		return attrs

@dataclass( kw_only = True, repr = False )
class Protocol( Type, ScopeMixin ): # @protocol class Foo:
	''' A structural interface contract (SYNTAX.md's `@protocol`). Never
	instantiated, never RC-managed, has no vtable - purely a compile-time
	conformance contract. A conforming class declares "I implement this"
	via ordinary base-class syntax (`class Bar(Foo):`) even though Foo isn't
	a real base - see discovery.py's _parse_ClassDef_RCClass, which routes a
	Protocol-typed entry in node.bases into RCClass.protocols below instead
	of RCClass.base (no vtable/chain_lookup participation at all). Declaring
	this makes it a compile error, right there at Bar's own definition, if
	Bar doesn't fit the contract - see discovery.py's
	_validate_protocol_conformance, which also splices any method the
	protocol supplies a real (non-stub) default body for directly into the
	conforming class's own dispatch table, so ordinary method lookup/dispatch
	needs no new runtime fallback mechanism at all - ITS conformance is
	settled once, at declaration time, not re-checked or scanned per-call. '''
	methods: list['Function|Overload'] = field( default_factory = list )
	names: dict[str,Name] = field( default_factory = dict )
	resolve: Callable[[],None]|None = None
	# @protocol Foo[T]: - None for a non-generic protocol (the overwhelming
	# majority, still fully supported unchanged). Mirrors RCClass/CStruct/
	# CUnion/TaggedUnion's own type_params field - see discovery.py's
	# _parse_ClassDef_Protocol (populates this) and _parse_ClassDef_RCClass's
	# base-list loop (a conforming class now records a Specialization of
	# THIS protocol, carrying its own concrete-or-still-typevar args, in
	# RCClass.protocols instead of the bare Protocol below).
	type_params: 'list[TypeVar]|None' = None

	def is_rc( self ) -> bool: return False
	def is_rc_pointer( self ) -> bool: return False
	def has_object_header( self ) -> bool: return False
	def has_vtable( self ) -> bool: return False

@dataclass( kw_only = True, repr = False )
class RCClass( Type, InheritanceChainMixin ): # normal ref-counted class
	# base is resolved eagerly at class-creation time, same as type_params -
	# Python itself requires a base class to already exist when the `class
	# Foo(Base):` statement runs, so there's no forward-reference case to
	# defer here. Multiple inheritance is a compile error (see discovery.py),
	# so this is a single pointer, not a list/MRO. May ALSO be a Specialization
	# (a generic base, e.g. Real[T] or Real[i32] - see discovery.py's
	# _parse_ClassDef_RCClass) rather than a plain RCClass - see
	# InheritanceChainMixin's own docstring for how the chain walk and
	# monomorphize.py's monomorphize_class/Discovery.on_generic_base_resolved
	# handle that.
	base: 'RCClass|Specialization|None' = None
	# @protocol types this class has explicitly declared conformance to
	# (base-class-list syntax - see Protocol's own docstring). Small and
	# per-class by construction - NOT a global registry, and NOT involved in
	# chain_lookup/vtable at all. discovery.py's _validate_protocol_
	# conformance checks and splices against this list once, at this class's
	# own definition.
	# a Specialization entry here (e.g. Sequence[T], T still this class's OWN
	# unbound type param) is a generic protocol's declared conformance,
	# carrying the args this class declared it with - see Protocol's own
	# type_params field and _parse_ClassDef_RCClass's base-list loop.
	protocols: 'list[Protocol|Specialization]' = field( default_factory = list )
	type_params: list[TypeVar]|None = None # if not None, this is a generic class
	attributes: list[Variable] = field( default_factory = list )
	methods: list['Function|Overload'] = field( default_factory = list )
	names: dict[str,Name] = field( default_factory = dict )
	resolve: Callable[[],None]|None = None

	# chain_lookup/own_new_virtual_slots/virtual_slots/flattened_attributes all
	# come from InheritanceChainMixin unchanged. Only vtbl_owner needs anything
	# here, and only to narrow the RETURN type - callers immediately treat the
	# result as an RCClass (emitter_c's _rcclass_vtbl_type_name takes one), and
	# typing.Self would be a lie, since ClosureType is an RCClass subclass whose
	# vtbl_owner can legitimately be a plain RCClass ancestor.
	def vtbl_owner( self ) -> 'RCClass':
		owner = super().vtbl_owner()
		assert isinstance( owner, RCClass ) # _next_chain_node always unwraps a Specialization down to its real RCClass template, so the walk itself is homogeneous even though .base may hold one
		return owner

	# the whole point of the class - and ClosureType (the only RCClass
	# subclass) inherits every one of these for free, which is exactly why
	# it was made a real subclass rather than a wrapper (see its docstring)
	def is_rc( self ) -> bool: return True
	def is_rc_pointer( self ) -> bool: return True
	def has_object_header( self ) -> bool: return True
	def has_vtable( self ) -> bool: return True

@dataclass( kw_only = True, repr = False )
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

@dataclass( kw_only = True, repr = False )
class CStruct( Type, InheritanceChainMixin ): # @cstruct class Foo:
	# base is only meaningful for @interface CStructs (single inheritance,
	# same "resolved eagerly at class-creation time" reasoning as
	# RCClass.base above) - a plain (non-@interface) CStruct subclassing
	# another CStruct is out of scope for now (see
	# PLAN_SUBCLASSING_VTABLES_COM.md's "Subclassing mechanics")
	base: 'CStruct|None' = None
	is_interface: bool = False # @interface class Foo: - NOT inherited implicitly, see plan doc
	# @cstruct(packed=True) - emitter_c.py wraps the whole struct body in
	# #pragma pack(push,1)/pop, so no field ever gets compiler-inserted
	# padding (verified identical layout across MSVC/clang/gcc). The
	# general, always-portable way to replicate an external ABI's exact
	# byte layout - see Variable.c_align for the narrower per-field knob.
	packed: bool = False
	type_params: list[TypeVar]|None = None
	attributes: list[Variable] = field( default_factory = list )
	methods: list['Function|Overload'] = field( default_factory = list )
	names: dict[str,Name] = field( default_factory = dict )
	resolve: Callable[[],None]|None = None

	# everything except vtbl_owner comes from InheritanceChainMixin unchanged;
	# this narrows the return type only - see RCClass's own copy for why a
	# shared implementation can't do it
	def vtbl_owner( self ) -> 'CStruct':
		owner = super().vtbl_owner()
		assert isinstance( owner, CStruct ) # the chain is homogeneous - .base is typed CStruct|None
		return owner

	def has_vtable( self ) -> bool:
		# only an @interface CStruct dispatches virtually (the COM model) -
		# and through its OWN top-level `$vtable` member, never an
		# ObjectHeader, which is why has_object_header() stays False here.
		# discovery.py's "is a @virtual method even legal on this class"
		# check is exactly this question.
		return self.is_interface

@dataclass( kw_only = True, repr = False )
class CUnion( Type, ScopeMixin ): # @cunion class Foo:
	packed: bool = False # see CStruct.packed
	type_params: list[TypeVar]|None = None
	attributes: list[Variable] = field( default_factory = list )
	methods: list['Function|Overload'] = field( default_factory = list )
	names: dict[str,Name] = field( default_factory = dict )
	resolve: Callable[[],None]|None = None

@dataclass( kw_only = True, repr = False )
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
			assert attr.type is not None
			result.append( attr.type )
		return result

	def _resolved_leaves( self ) -> list['Type']:
		''' leaves(), but forcing this union's own CLASS BODY to resolve first.

		Load-bearing, not caution. leaves() reads self.attributes directly,
		which is populated by the CLASS's own .resolve() (parsing its body) - a
		separate step from leaves()'s own per-ATTRIBUTE attr.resolve() (which
		only resolves each attribute's already-existing .type). Called too
		early - e.g. the very first time any code anywhere references a
		Result[...]-shaped type, before anything else has forced Result's own
		class body to resolve - self.attributes is still empty and leaves()
		silently returns [], which reads as "this union has no RC leaves"
		rather than "this union hasn't been read yet". Confirmed by a real
		UAF: it made a temp Result[str,CodecError] receiver of .unwrap() look
		RC-free depending on ONLY where in the compile that particular call
		site happened to land relative to Result's own first real use
		elsewhere. '''
		if self.resolve is not None:
			self.resolve()
		return self.leaves()

	def is_rc( self ) -> bool:
		# a union is RC whenever ANY of its members are. The recursion matters
		# for a union appearing as a LEAF of an outer type (e.g. Result[T, A|B]):
		# before this existed, a nested union leaf was always reported non-RC
		# (a TaggedUnion is never an RCClass), so rc_leaves() on the OUTER type
		# silently dropped it entirely even when its own members carried real
		# RC payloads - no incref/decref ever fired for that leaf's contents, a
		# genuine reference leak. (A nested union's own TOP-LEVEL rc_leaves()
		# always worked; the gap was specifically one level up, treating A|B as
		# an opaque, always-non-RC leaf of something else.)
		#
		# No type-param substitution needed here, unlike Specialization.rc_leaves:
		# every leaf reaching this point is either a fully concrete anonymous
		# union (never generic/Specialization-wrapped by construction) or has
		# already had its params substituted by whichever caller is asking.
		return any( leaf.is_rc() for leaf in self._resolved_leaves() )

	def is_result_type( self ) -> bool:
		return self.stem == 'Result'

	def is_rc_pointer( self ) -> bool:
		# a union's runtime shape is a tag+data VALUE STRUCT, never a bare
		# pointer - see Type.is_rc_pointer's docstring
		return False

	def rc_leaves( self ) -> list['Type']:
		# str|i32 needs a TAG-GATED incref (only the str arm); str|int (both
		# RC) needs none of that, unconditional instead - see cfg.py's
		# _refcount_instructions, which forks on exactly this
		return [ leaf for leaf in self._resolved_leaves() if leaf.is_rc() ]

def by_value_dependency( t: 'Type|None' ) -> 'CStruct|CUnion|TaggedUnion|None':
	''' the CStruct/CUnion/TaggedUnion `t` embeds BY VALUE, if any - i.e.
	exactly the kind of dependency emitter_c.py's _emit_value_type_bodies
	topologically sorts on (a struct member needs its own type's FULL C
	definition already emitted, unlike an RCClass field, which is always a
	pointer and never forces an ordering - see that function's own
	docstring). Unwraps a Specialization first (a generic field's own
	declared type, e.g. `x: Result[i32,E]`) - same `t.base if isinstance(t,
	Specialization) else t` idiom Type.is_rc_pointer's own delegation
	retired elsewhere, needed again here since this asks a different
	question (layout dependency, not RC-ness). Returns None for anything
	else (a scalar, an RCClass, ...). '''
	base = t.base if isinstance( t, Specialization ) else t
	return base if isinstance( base, ( CStruct, CUnion, TaggedUnion )) else None

@dataclass( kw_only = True, repr = False )
class CEnum( Type, ScopeMixin ): # @enum class Foo:
	value_type: Type
	next_auto: int = 0
	members: dict[str,int] = field( default_factory = dict )
	values: dict[int,str] = field( default_factory = dict )
	methods: list['Function|Overload'] = field( default_factory = list )
	names: dict[str,Name] = field( default_factory = dict )
	resolve: Callable[[],None]|None = None

@dataclass( kw_only = True, repr = False )
class CType( Type ):
	''' a C type defined in an external header, referenced by bare name.
	Used with compiler.c_type('pthread_mutex_t', header='pthread.h') -
	the emitter spells this as the raw C name, and compiler.sizeof(CType)
	emits sizeof(pthread_mutex_t) in the generated C. '''
	c_name: str          # e.g. 'pthread_mutex_t'
	required_header: str  # e.g. 'pthread.h'

# anything that can own methods/be a Function's .cls
ClassLike = Union[ RCClass, CStruct, CUnion, TaggedUnion, CEnum ]

@dataclass( kw_only = True, repr = False )
class Function( Type, ScopeMixin ):
	cls: ClassLike|None
	node: ast.FunctionDef # whole def - node.args/.returns resolved lazily, node.body untouched until IR generation
	# the module this function was DEFINED in (set from Discovery.module_stack
	# at parse time, once, alongside .file - never reassigned afterward).
	# Lets a caller go straight from a Function to its owning Module without
	# a self.modules.values() scan-by-file - see discovery.py's
	# _splice_protocol_default, the motivating caller: a @protocol default
	# method's free/qualified names must resolve against the PROTOCOL's own
	# module, not whichever module happens to be conforming to it.
	module: 'Module|None' = None
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
	# optional runtime DLL(s) (bare filenames, e.g. 'tcl86t.dll') this
	# @extern function needs loadable at runtime - not the same as
	# extern_lib (the .lib/.so linked against at build time, which can
	# live in a different directory than the .dll, or not exist as a
	# separate file at all for a header-only/forwarded symbol). Written as
	# dll='name.dll' or dll=['name.dll', 'other.dll'] - deliberately a
	# fixed, author-supplied list rather than something the compiler
	# derives by scanning a DLL's own import table: the transitive
	# dependency set of a real DLL includes both genuinely-needed vendored
	# files (e.g. tcl86t.dll needs zlib1.dll) AND system components
	# (kernel32.dll, the api-ms-win-crt-*.dll forwarders, ...) that must
	# NEVER be bundled - reliably telling those apart automatically would
	# need either a maintained system-DLL blacklist (a maintenance
	# nightmare, explicitly rejected) or heuristics prone to bundling the
	# wrong thing. An explicit, per-declaration list sidesteps the
	# question entirely: the author decides exactly what ships, including
	# deliberately leaving out something like VCRUNTIME140.dll if it's
	# assumed already present on target machines. See compiler.py's
	# extern_dlls collection and mpy.py's post-link bundling step.
	extern_dlls: tuple[str,...] = ()
	# optional 3rd-party license notice identifier(s) required when this
	# @extern function (and, transitively, whatever it bundles via
	# extern_dlls) ships in a built program's dist/ output. Written as
	# notice='NAME' or notice=['NAME', 'OTHER'] - each NAME resolves to
	# licenses/NAME.txt relative to the metalpy installation itself (see
	# mpy.py's own resolution). Deliberately a separate declaration from
	# extern_dlls, not auto-derived from it: a notice can be shared across
	# unrelated libraries (e.g. 'ZLIB' applies to anything that happens to
	# bundle zlib1.dll, not just Tcl/Tk), so collapsing the two ideas would
	# either duplicate the same license text under every DLL that happens
	# to depend on it, or require guessing which dll= entries share a
	# notice. Same reachability-gated collection as extern_dlls (see
	# compiler.py) and same "explicit author list, fails the build if
	# unresolvable" philosophy - see mpy.py's post-link step, which
	# combines every referenced notice into one dist/THIRD-PARTY-LICENSES
	# file.
	extern_notices: tuple[str,...] = ()
	# PLAN_THREAD_SAFE_SHARED_STATE.md Cost mitigation #1: True only for the
	# real OS-thread-creation syscall boundary (posix.pthread.pthread_create,
	# windows.kernel32.CreateThread) - marks the ONE place Compiler needs to
	# notice to know "this program can ever have a second OS thread", rather
	# than scanning for construction of lib/threading.py's own Thread class
	# (which would miss anything reaching a thread through a different path,
	# and duplicate detection logic every future thread-spawning wrapper
	# would need its own case for). See compiler.py's own requires_crt-
	# adjacent flip site for how this gets consumed - identical pattern.
	extern_spawns_thread: bool = False

	is_overload: bool = False # was this def @overload-decorated (whether it ended up a stub or, with a real body, an Overload.implementations entry)
	bound_to: 'Function|None' = None # stubs only: the plain implementation this stub's signature resolves to (see discovery.py's _bind_overload_stub)
	is_destructor: bool = False # synthesized $$__destructor__ body — emitter uses void(void*) signature + cast prologue
	# PLAN_GENERATORS.md Phase F - synthesized $$__next__ body (a
	# generator's backing class method) - lowering.py gates its own
	# state-check dispatch prologue on this, mirroring is_destructor's
	# identical "synthesized method, special lowering-time treatment"
	# precedent. node.generator_yield_states (a list[tuple[int,str]] tag
	# set once by type_resolver.py's _build_generator_next_function, not
	# a dataclass field - same convention as this file's other AST-level
	# generator tags) carries the (state, resume_label) pairs the
	# prologue dispatches on.
	is_generator_next: bool = False

	# stems of this function's own parameters that a `defer`/`errdefer`
	# block somewhere in its body conditionally incref's (see cfg.py's
	# push_defer) - populated lazily, during THIS function's own lowering,
	# not known up front. A caller passing one of ITS OWN locals into one
	# of these parameter positions may get an extra, compiler-invisible
	# reference back on return - see lowering.py's own Call-site check
	# (mark_possibly_retained) and cfg.py's manually_decreffed().
	errdefer_retained_params: set[str] = field( default_factory = set )
	# lowering.py's _ensure_errdefer_retained_params is lazy (an AST scan,
	# not IR-derived - reachability-ordered lowering means a callee may not
	# have been lowered yet when a caller's own Call needs this, see its
	# own comment) - this distinguishes "not computed yet" from "computed,
	# genuinely empty" so it only ever scans once.
	errdefer_retained_params_computed: bool = False

	# implementations only (never set on a stub - stubs are never scheduled
	# as real compile units, so they never need a C symbol of their own) -
	# the Overload group this Function was appended to group.implementations
	# of, set alongside that same append (see discovery.py's
	# _parse_function_def). Every member of one group shares the group's own
	# .qualname (it's literally "the same named function", just a different
	# signature) - emitter_c.py's mangle_function_qualname uses this back-
	# reference to disambiguate the C symbol when more than one member of
	# the same group actually needs a real body.
	overload_group: 'Overload|None' = None

	# @inline (PLAN_INLINE.md) - body is arbitrary statements followed by
	# exactly one final, top-level `return <expr>` (no other `return`
	# anywhere else, no defer/errdefer, no reassignment of self/a
	# parameter - discovery.py's _is_inline_eligible_body and its sibling
	# scanners enforce this at parse time); lowering.py splices the whole
	# body directly at each call site instead of ever emitting a real
	# Call/FuncStart/FuncEnd for it
	is_inline: bool = False

	# @property - a zero-argument getter that reads like a plain field:
	# `obj.attr` (no call parens) calls this method and returns its result,
	# instead of the ordinary "bound-method closure" _expr_Attribute builds
	# for any other method used as a value (see lowering.py's _expr_
	# Attribute). Read-only only - no @x.setter support yet (would need its
	# own exemption from discovery.py's duplicate-definition check, the same
	# way @overload gets one).
	is_property: bool = False

	# @fallible_arithmetic - a dunder (e.g. int.__floordiv__, or a scalar-
	# registered __add__) whose Result[T,E] return should be consumed by
	# binop dispatch through the SAME ambient-arithmetic-mode machinery
	# scalar Check-mode opcodes already use (_consume_checked_result),
	# instead of being left opaque for the caller to .unwrap()/.or_return()
	# explicitly. Named for what it marks (arithmetic that can fail and
	# should thread through wrap/saturate/panic_arithmetic too), not
	# "checked" - that word already means something narrower and different
	# in this file (ArithmeticChecked/Check-mode opcodes/checked_errors -
	# the DEFAULT mode specifically), and this flag's own behavior spans
	# every mode, not just that one. Only consulted by _lower_binop_values/
	# _classify_leaf_pair_binop for real ast.BinOp operator dispatch - never
	# for explicit method-call syntax. Return-type shape (must be
	# Result[T,E]) is validated where it's used, not here - return_type
	# isn't resolved yet at parse time (see .resolve).
	is_fallible_arithmetic: bool = False

	# @requires_crt - marks a function whose mere reachability (not any
	# @extern('c', ...) call it makes itself) means the whole build must
	# link the real CRT rather than the freestanding Windows entry point -
	# e.g. a library that runs user code on a small/foreign stack and can't
	# rule out that code needing MSVC's __chkstk, which a no_crt build has
	# no way to supply (see msvc_no_crt_missing_chkstk). No effect on
	# non-Windows targets, where there's no freestanding/no_crt distinction
	# to override. compiler.py's Compiler._lower sets self.requires_crt
	# on the Compiler instance the same way it already does for
	# extern_lib/compiler.extern_libs - only when THIS function is actually
	# lowered (reachable from main()), never merely discovered.
	requires_crt: bool = False

def _leaf_is_accepted( leaf: Type, declared: Type|None ) -> bool:
	# identity-based deliberately, not `==` - Type dataclasses have structural
	# equality (comparing every field, including mutable dicts/lists), which is
	# both wrong and expensive for "is this the same type" here. The existing
	# _get_or_create_union/_specialization/_move dedup caches already guarantee
	# "the same type" is the same object within one Discovery instance.
	assert declared is not None
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
	for i in range( len( narrow.parameters )):
		narrow_param_type = narrow.parameters[i].type
		assert narrow_param_type is not None
		for leaf in narrow_param_type.leaves():
			if not _leaf_is_accepted( leaf, wide.parameters[i].type ):
				return False
	return True

def _overlaps( a: Function, b: Function ) -> bool:
	''' some concrete call could satisfy both `a` and `b` simultaneously. Used for plain-implementation-vs-plain-implementation ambiguity. '''
	if a.parameters is None or b.parameters is None:
		return False # one of them failed to resolve - can't reason about overlap
	if len( a.parameters ) != len( b.parameters ):
		return False
	for i in range( len( a.parameters )):
		a_param_type = a.parameters[i].type
		assert a_param_type is not None
		b_param_type = b.parameters[i].type
		if not any( _leaf_is_accepted( leaf, b_param_type ) for leaf in a_param_type.leaves() ):
			return False
	return True

@dataclass( kw_only = True )
class ConditionalDispatch:
	''' one branch of a runtime overload dispatch: if the value(s) bound to each of `conditions` match their expected type, call `function`. '''
	conditions: list[tuple[Parameter,Type]]
	function: Function

@dataclass( kw_only = True, repr = False )
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

	# the package this module lives in, i.e. Python's own __package__ - the
	# base a relative import counts levels up from. Deliberately NOT derivable
	# from qualname (Python's __name__): a module that folds into its package
	# takes the package's own qualname, so slicing a level off qualname would
	# climb one level too far from an __init__.py or a package-private
	# __foo.py, and there is no way to tell from the string alone whether
	# folding happened. '' for a top-level module, which owns no package and
	# from which any relative import is an error
	package: str = ''
