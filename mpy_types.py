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
	# TODO FIXME: base class for subclassing
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
	parameters: list[Variable]|None = None
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

@dataclass( kw_only = True )
class Overload( Type ):
	'''
	stands in for a Function when multiple defs share a name in the same scope.

	stubs: @overload-decorated defs with no real body - signature-only,
	never actually called; implementations: real bodies, either one plain
	(non-@overload) fallback that every stub maps to, or several
	@overload-decorated ones distinguished by their own parameters.

	NOTE: call-site dispatch (given argument types, which implementation
	applies) is deliberately not implemented here - that's a call-expression-
	compilation concern for stage 2 IR generation, not discovery. Nothing
	about the group itself is deferred (its members list is complete the
	moment it's built) - each member Function still has its own .resolve.
	'''
	cls: ClassLike|None = None
	stubs: list[Function] = field( default_factory = list )
	implementations: list[Function] = field( default_factory = list )

@dataclass( kw_only = True )
class Module( Name, ScopeMixin ):
	intrinsics: dict[str,Name]
	builtins: dict[str,Name]|None
	names: dict[str,Name] = field( default_factory = dict )
