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

# a class's own body scan (revealing its attribute/method *names*) is
# deferred behind .resolve, exactly like a Function's parameters or a
# Variable's type - nothing about a class's members is known until something
# calls it. type_params is the one exception: it's parsed eagerly, at
# creation time, because external code subscripting this class as a generic
# (Result[i32,usize]) needs to see it before this class's own .resolve ever runs.

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

@dataclass( kw_only = True )
class CStruct( Type, ScopeMixin ): # @cstruct class Foo:
	type_params: list[TypeVar]|None = None
	attributes: list[Variable] = field( default_factory = list )
	methods: list['Function|Overload'] = field( default_factory = list )
	names: dict[str,Name] = field( default_factory = dict )
	resolve: Callable[[],None]|None = None

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

	# @extern('lib', 'symbol') - a foreign call signature declaration (body
	# must be a stub - see discovery.py's _is_stub_body). extern_lib is the
	# .lib/.so name to link against, except the literal 'c' which means the
	# platform C runtime rather than a real file on disk. Both None for an
	# ordinary function
	extern_lib: str|None = None
	extern_symbol: str|None = None

	is_overload: bool = False # was this def @overload-decorated (whether it ended up a stub or, with a real body, an Overload.implementations entry)
	bound_to: 'Function|None' = None # stubs only: the plain implementation this stub's signature resolves to (see discovery.py's _bind_overload_stub)

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
