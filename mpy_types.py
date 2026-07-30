# stdlib imports:
import ast
from dataclasses import dataclass, field
import itertools
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

@dataclass( kw_only = True )
class Scalar( Type ):
	'''
	isize, usize, i32, u32, etc - also used for generic pointer intrinsics
	(Ptr, ConstPtr), which is why type_params exists here too
	'''
	type_params: list['TypeVar']|None = None

@dataclass( kw_only = True )
class TypeVar( Type ):
	''' a placeholder for one of a generic's type parameters, e.g. T in class Result[T,E] '''

@dataclass( kw_only = True )
class Specialization( Type ):
	''' a generic base type applied to concrete (or still-typevar) type arguments, e.g. Result[i32,IntError] '''
	base: Type
	args: list[Type]

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
	''' `move[T]` in annotation position - ownership of a T is transferred into this binding rather than borrowed/copied. The CFG uses this to know the source binding must be invalidated after the transfer. '''
	inner: Type


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
	attributes: list[Variable] = field( default_factory = list )
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
	if len( narrow.parameters ) != len( wide.parameters ):
		return False
	return all(
		all( _leaf_is_accepted( leaf, wide.parameters[i].type ) for leaf in narrow.parameters[i].type.leaves() )
		for i in range( len( narrow.parameters ))
	)

def _overlaps( a: Function, b: Function ) -> bool:
	''' some concrete call could satisfy both `a` and `b` simultaneously. Used for plain-implementation-vs-plain-implementation ambiguity. '''
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
	'''
	cls: ClassLike|None = None
	stubs: list[Function] = field( default_factory = list )
	implementations: list[Function] = field( default_factory = list )

	def _overload_members( self ) -> list[Function]:
		return sorted( [ *self.stubs, *( f for f in self.implementations if f.is_overload ) ], key = lambda f: f.line )

	def _plains( self ) -> list[Function]:
		return [ f for f in self.implementations if not f.is_overload ]

	def _target( self, member: Function ) -> Function:
		return member.bound_to if member in self.stubs else member

	def resolve_call( self, args: list[Type], kwargs: dict[str,Type] ) -> tuple[list[ConditionalDispatch],Function]:
		'''
		given a call's argument types (positional and/or keyword - different
		variants may name their own parameters differently), returns either a
		single unconditional target (`([], fn)`) or an ordered list of runtime
		conditions to check before falling through to the trailing default
		Function. Pure function of types - no AST/call-site involved, so this
		is fully unit-testable ahead of stage 2 (which will supply the actual
		call-site argument types).
		'''
		for fn in ( *self.stubs, *self.implementations ):
			if fn.resolve is not None:
				fn.resolve()

		slots: list[tuple[int|str,list[Type]]] = (
			[ ( i, arg.leaves() ) for i, arg in enumerate( args ) ]
			+ [ ( name, typ.leaves() ) for name, typ in kwargs.items() ]
		)
		keys = [ key for key, _ in slots ]
		leaf_lists = [ leaves for _, leaves in slots ]

		def binds( candidate: Function, assignment: dict ) -> bool:
			covered_stems: set[str] = set()
			for key, leaf in assignment.items():
				if isinstance( key, int ):
					if key >= len( candidate.parameters ):
						return False
					param = candidate.parameters[key]
				else:
					param = next( ( p for p in candidate.parameters if p.stem == key ), None )
					if param is None:
						return False
				if not _leaf_is_accepted( leaf, param.type ):
					return False
				covered_stems.add( param.stem )
			return all( param.stem in covered_stems or param.default is not None for param in candidate.parameters )

		overload_members = self._overload_members()
		plains = self._plains()
		# priority order a real call site would check candidates in - stubs
		# and real-bodied @overload members first-match (in declaration
		# order), plain implementations last. Used purely to rank *targets*
		# below, not to decide which one wins for a given assignment.
		priority = { id( m ): i for i, m in enumerate([ *overload_members, *plains ]) }

		def resolve_one( assignment: dict ) -> tuple[Function,Function]:
			for member in overload_members:
				if binds( member, assignment ):
					return self._target( member ), member
			matches = [ p for p in plains if binds( p, assignment ) ]
			assert len( matches ) == 1, (
				f'{self.qualname}: ambiguous call for {assignment!r} - matches {[m.qualname for m in matches]}'
				if matches else
				f'{self.qualname}: no overload matches argument types {assignment!r}'
			)
			return matches[0], matches[0]

		resolved: list[tuple[dict,Function]] = []
		target_rank: dict[int,int] = {} # id(target) -> best (lowest) priority rank it was ever reached through
		for combo in itertools.product( *leaf_lists ):
			assignment = dict( zip( keys, combo ))
			target, matched_via = resolve_one( assignment )
			resolved.append(( assignment, target ))
			rank = priority[ id( matched_via ) ]
			target_rank[ id( target ) ] = min( rank, target_rank.get( id( target ), rank ))

		distinct_targets: list[Function] = []
		for _, target in resolved:
			if not any( target is t for t in distinct_targets ):
				distinct_targets.append( target )
		# order by resolution priority (stub-matched targets are more specific
		# "special cases" and come first; a target only ever reached via the
		# plain-implementation fallback is the most general case, so it sorts
		# last and becomes the trailing default below) rather than by whatever
		# incidental order the leaf decomposition happened to enumerate in
		distinct_targets.sort( key = lambda t: target_rank[ id( t ) ] )

		if len( distinct_targets ) == 1:
			return ( [], distinct_targets[0] )

		default = distinct_targets[-1]
		branches: list[ConditionalDispatch] = []
		for target in distinct_targets[:-1]:
			own = [ a for a, t in resolved if t is target ]
			others = [ a for a, t in resolved if t is not target ]
			conditions: list[tuple[Parameter,Type]] = []
			for key in keys:
				own_leaves = [ a[key] for a in own ]
				if any( leaf is not own_leaves[0] for leaf in own_leaves[1:] ):
					continue # this slot still varies within this branch - can't be a single-value condition
				shared_leaf = own_leaves[0]
				if any( o[key] is not shared_leaf for o in others ):
					param = target.parameters[key] if isinstance( key, int ) else next( p for p in target.parameters if p.stem == key )
					conditions.append(( param, shared_leaf ))
			branches.append( ConditionalDispatch( conditions = conditions, function = target ))

		return ( branches, default )

@dataclass( kw_only = True )
class Module( Name, ScopeMixin ):
	intrinsics: dict[str,Name]
	builtins: dict[str,Name]|None
	names: dict[str,Name] = field( default_factory = dict )
