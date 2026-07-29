# stdlib imports:
import ast
from dataclasses import dataclass
from pathlib import Path
import threading
from typing import Iterator

@dataclass( kw_only = True )
class Name:
	stem: str # local name like 'str' instead of 'builtins.str'
	qualname: str # fully qualified name: 'builtins.str' instead of 'str'
	
	# we don't always know where a name is defined the first time we see it:
	file: Path|None
	line: int|None

@dataclass( kw_only = True )
class UnresolvedName( Name ):
	resolution: Name|None = None

@dataclass( kw_only = True )
class Type( Name ):
	''' maybe only use this to distinguish types from values '''

@dataclass( kw_only = True )
class Scalar( Type ):
	''' isize, usize, i32, u32, etc '''

@dataclass( kw_only = True )
class Variable( Name ):
	type: Type # we always (?) know a variable's type ( although it may be an UnresolvedType )


@dataclass( kw_only = True )
class RCClass( Type ): # normal ref-counted class
	# TODO FIXME: base class for subclassing
	type_params: list[str]|None # if not None, this is a generic class
	attributes: list[Variable]
	methods: list[Function]
	names: dict[str,Name]
	body: list[ast.AST]|None
	
	def add_name( self, name: str, name_obj: Name ) -> None:
		self.names[name] = name_obj

@dataclass( kw_only = True )
class CStruct( Type ): # @cstruct class Foo:
	body: list[ast.AST]|None

@dataclass( kw_only = True )
class CUnion( Type ): # @cunion class Foo:
	body: list[ast.AST]|None

@dataclass( kw_only = True )
class TaggedUnion( Type ): # @union class Foo:
	body: list[ast.AST]|None

@dataclass( kw_only = True )
class CEnum( Type ): # @enum class Foo:
	value_type: Type
	next_auto: int = 0
	members: dict[str,int|None]
	values: dict[int|None,str]
	body: list[ast.AST]|None

@dataclass( kw_only = True )
class Function( Type ):
	cls: RCClass|None
	parameters: list[Variable]
	return_type: Type|None
	body_ast: list[ast.stmt]
	names: dict[str,Name]
	
	def add_name( self, name: str, name_obj: Name ) -> None:
		self.names[name] = name_obj

@dataclass( kw_only = True )
class Overload( Type ):
	cls: RCClass|None = None

@dataclass( kw_only = True )
class Module( Name ):
	intrinsics: dict[str,Name]
	builtins: dict[str,Name]|None
	names: dict[str,Name]
	
	def add_name( self, name: str, name_obj: Name ) -> None:
		#print( f'adding {self.qualname}.{name}' )
		self.names[name] = name_obj

@dataclass( kw_only = True )
class PendingTypeInference( Type ):
	resolution: Type|None = None

class NameRegistry:
	def __init__( self ) -> None:
		self._registry: dict[str,Name] = {}
		self.unresolved_names: dict[str,Name] = {}
		self._lock = threading.RLock()
	
	def get_or_create( self, *, qualname: str ) -> Name:
		# Fast read check
		if name_obj := self._registry.get( qualname, None ):
			return name_obj
		
		with self._lock:
			# Double-check inside lock to guarantee single instance
			if ( name_obj := self._registry.get( qualname )) is None:
				self._registry[qualname] = name_obj = UnresolvedName(
					stem = qualname.rpartition( '.' )[2],
					qualname = qualname,
					file = None, # unknown yet
					line = None, # unknown yet
				)
				self.unresolved_names[qualname] = name_obj
			return name_obj
	
	def resolve( self, real_name: Name ) -> Name:
		with self._lock:
			name_obj = self.get_or_create( qualname = real_name.qualname )
			if isinstance( name_obj, UnresolvedName ):
				if name_obj.resolution is None:
					name_obj.resolution = real_name
				else:
					# TODO FIXME: the following can probably be triggered by creating 2 Foo objects, may need to be a compile error
					assert real_name == name_obj.resolution, f'name object mismatch: {real_name=} vs {name_obj.resolution=}'
				self._registry[real_name.qualname] = real_name
				self.unresolved_names.pop( real_name.qualname, None )
			return name_obj
	
	def items( self ) -> Iterator[tuple[str,Name]]:
		return self._registry.items()
