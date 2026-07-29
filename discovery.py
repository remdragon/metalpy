# stdlib imports:
import ast
from contextlib import contextmanager
import itertools
from pathlib import Path
from typing import Any, Callable, Generator

# local imports
from mpy_types import (
	Name, Type, Scalar, Variable, Function,
	CEnum, RCClass, CStruct, CUnion, TaggedUnion,
	Module, NameRegistry,
)

class CompilerModule( Module ):
	' TODO FIXME: put target object here and anything else needed'


class Discovery( ast.NodeVisitor ):
	'''
	This class does a shallow parse of python files and processes all imports on demand
	The goal is to build a complete registry of all potential types in the system
	This class does not parse function bodies at all
	This class only parses class bodies for subclass definitions, but does not process attributes and class member functions
	
	The reason for this is simple. The next stage of the compiler will generate IR.
	It will start with main() and only generate class and functions in IR that are
	actually referenced by main.
	
	Therefore, this class scans just enough to build a complete list of all types
	that main could possibly reference.
	
	IMPORTANT:
		more imports can be uncovered in function bodies
		more types can be created by function bodies ( particularly anonymous unions )
		
		the reason this class doesn't try to uncover those things here is we
		only want to compile things that are actually going to be emitted to the executable
		
		This approach doesn't allow us to cache compilation results for a future compile.
		That's a problem to solve after metalpy is proven viable and starts getting
		used for projects large enough for that capability to be important.
	'''
	log_unhandled: bool = False
	
	builtins: dict[str,Name]|None = None
	_intrinsics: dict[str,Name]|None = None
	
	def __init__( self,
		registry: NameRegistry,
		paths: list[Path] = [],
		import_builtins: bool = True,
	) -> None:
		self.registry = registry
		self.paths = paths
		if not self.paths:
			self.paths.append( Path( __file__ ).parent / 'lib' )
			self.paths.append( Path( '.' ))
		self.compiler_module = CompilerModule(
			stem = 'compiler',
			qualname = 'compiler',
			file = None,
			line = None,
			intrinsics = {},
			builtins = None,
			names = {}, # TODO FIXME: fill this out...
		)
		self.modules: dict[str,Module] = {}
		
		self.module_stack: list[Module] = []
		self.scope_stack: list[Module|CEnum|RCClass|CStruct|CUnion|TaggedUnion|Function] = []
		
		if import_builtins:
			self.builtins = self.import_name( 'builtins' )
	
	@contextmanager
	def scope_context( self, scope: Module|CEnum|RCClass|CStruct|CUnion|TaggedUnion|Function ) -> Generator[None,None,None]:
		self.scope_stack.append( scope )
		try:
			yield
		finally:
			popped = self.scope_stack.pop()
			assert popped == scope
	
	@contextmanager
	def module_context( self, module: Module ) -> Generator[None,None,None]:
		old_scope_stack = self.scope_stack
		self.module_stack.append( module )
		try:
			self.scope_stack = []
			with self.scope_context( module ):
				yield
		finally:
			self.scope_stack = old_scope_stack
			popped = self.module_stack.pop()
			assert popped == module
	
	def import_file( self, filename: Path, scope: str|None = None ) -> Module:
		with filename.open( 'r' ) as f:
			code = f.read()
		return self.import_code( code, filename, scope )
	
	def import_code( self, code: str, filename: Path, scope: str|None = None ) -> Module:
		# NOTE: builtins starts off None and we import builtins when we first instanciate this class
		# that way builtins exists whenever we are ready to parse any other code besides builtins
		stem = filename.stem if filename else ''
		qualname = ( f'{scope}.{stem}' if scope else stem )
		module = Module(
			stem = stem,
			qualname = qualname,
			file = filename,
			line = None,
			intrinsics = self.get_intrinsics(),
			builtins = self.builtins.names if self.builtins else None,
			names = {},
		)
		with self.module_context( module ):
			tree = ast.parse( code )
			self.visit( tree )
		
		return module
	
	def import_name( self,
		package: str,
	) -> Module:
		if package == 'compiler':
			return self.compiler_module
		
		#print( f'{package=}' )
		noisy = False
		#if package == 'codecs':
		#	noisy = True
		
		relpath = package.replace( '.', '/' )
		looked: list[str] = []
		suffixes = [ '.mpy', '.py' ] # TODO FIXME: this kinda sucks for performance...
		for base in self.paths:
			path = base / relpath
			if path.is_dir():
				stem = '__init__'
				qualname = f'{package}.{stem}'
				if noisy:
					print( f'{str(path)!r}.is_dir=True, {stem=} {qualname=}' )
			else:
				stem = path.stem
				path = path.parent
				qualname = '.'.join([ package.rpartition( '.' )[0], stem ]).lstrip( '.' )
				#assert False, f'{package=} {name=} -> {path=} {stem=} {qualname=}'
			if mod := self.modules.get( package, None ):
				return mod
			for suffix in suffixes:
				filename = path / f'{stem}{suffix}'
				if noisy:
					print( f'trying {str(filename)!r}' )
				if filename.is_file():
					if noisy:
						print( f'{str(filename)!r}.is_file()=True' )
					#assert False, f'{filename=} {package=} {scope=}'
					mod = self.import_file( filename, scope = qualname.rpartition( '.' )[0] )
					self.modules[package] = mod
					return mod
				else:
					looked.append( str( filename ))
		e = FileNotFoundError( package )
		e.add_note( f'looked in:\n\t{"\n\t".join(looked)}' )
		raise e
	
	def get_intrinsics( self ) -> dict[str,Name]:
		# TODO FIXME: these might need to be special Name objects because they are intrinsics...
		if self._intrinsics is None:
			intrinsics: dict[str,Name] = {}
			for name in [ 'isize', 'usize', 'i8', 'u8', 'i16', 'u16', 'i32', 'u32', 'i64', 'u64', 'i128', 'u128' ]:
				intrinsics[name] = Scalar(
					stem = name,
					qualname = f'intrinsics.{name}',
					file = None,
					line = None,
				)
			self._intrinsics = intrinsics
		return self._intrinsics
	
	def _get_qualname( self, name: str ) -> str:
		scope = self.scope_stack[-1].qualname
		return f'{scope}.{name}' if scope else name
	
	def find_name( self, name: str, ctx: ast.AST ) -> Name:
		mod = self.module_stack[-1]
		builtins = mod.builtins
		#print( f'{self.scope_stack=}' )
		#print( f'{mod.intrinsics=}' )
		for scope in itertools.chain(
			[ scope.names for scope in self.scope_stack[::-1] ],
			[ builtins.names if builtins else {} ],
			[ mod.intrinsics ],
		):
			assert isinstance( scope, dict ), f'invalid {scope=}'
			if name_obj := scope.get( name ):
				return name_obj
		e = NameError( name )
		e.add_note( f'{str(self.filename)}:{ctx.lineno}' )
		raise e
	
	def visit( self, node: ast.AST ) -> Any:
		method = f'visit_{node.__class__.__name__}'
		handler = getattr( self, method, None )
		if handler is None:
			if self.log_unhandled:
				line = f" (line {node.lineno})" if hasattr(node, "lineno") else ""
				print( f"[Unhandled AST] {node.__class__.__name__}{line}" )
			handler = self.generic_visit
		return handler( node )
	
	def visit_Name( self, node: ast.Name ) -> Name:
		assert isinstance( node.ctx, ast.Load ), f'invalid context on {node=}'
		name = self.find_name( node.id, node )
		assert isinstance( name, Name ), f'invalid {name=} from {node=}'
		return name
	
	def visit_Import( self, node: ast.Import ) -> None:
		# import foo -> Import(names=[alias(name='foo', asname=None)])
		# import foo.bar -> Import(names=[alias(name='foo.bar', asname=None)])
		# import foo as bar -> Import(names=[alias(name='foo', asname='bar')])
		# import foo.bar as baz -> Import(names=[alias(name='foo.bar', asname='baz')])
		scope = self.scope_stack[-1]
		for alias in node.names:
			mod = self.import_name( alias.name )
			scope.add_name( alias.asname or alias.name, mod )
		self.generic_visit( node )
	
	def visit_ImportFrom( self, node: ast.ImportFrom ) -> None:
		# from . import foo -> ImportFrom(module=None, names=[alias(name='foo', asname=None)], level=1)
		# from foo import bar -> ImportFrom(module='foo', names=[alias(name='bar', asname=None)], level=0)
		# from .foo import bar -> ImportFrom(module='foo', names=[alias(name='bar', asname=None)], level=1)
		# from foo import bar as baz -> ImportFrom(module='foo', names=[alias(name='bar', asname='baz')], level=0)
		# from .foo import bar as baz -> ImportFrom(module='foo', names=[alias(name='bar', asname='baz')], level=1)
		# from ..foo import bar -> ImportFrom(module='foo', names=[alias(name='bar', asname=None)], level=2)
		module = node.module or ''
		parts: list[str] = []
		if node.level:
			parts.extend( self.module_stack[-1].qualname.split( '.' )[:-node.level] )
			assert parts, f'unable to relative import from here: {node=} {self.module_stack[-1].qualname=} {self.module_stack[-1].file=}'
		if node.module:
			parts.append( node.module )
		package = '.'.join( parts )
		#print( f'{package=}' )
		scope = self.scope_stack[-1]
		mod = self.import_name( package )
		if not mod:
			raise NameError( package )
		for alias in node.names:
			#print( f'{self.module_stack[-1].qualname=} {package=} {node.level=} {node.module=} {alias.name=}' )
			item = mod.names.get( alias.name )
			if not item:
				raise NameError( f'module {package} does not export {alias}' )
			scope.add_name( alias.asname or alias.name, item )
		#self.generic_visit( node )
	
	def visit_Assign( self, node: ast.Assign ) -> None:
		# this stage does not parse function bodies, so this is either a global variable or a class attribute
		from pprint import pprint
		pprint( node )
	
	def visit_AnnAssign( self, node: ast.AnnAssign ) -> None:
		if isinstance( node.target, ast.Name ):
			type_str = ast.unparse( node.annotation )
			var_type = self.registry.get_or_create( qualname = type_str )
			
			var_obj = Variable(
				stem = node.target.id,
				qualname = self._get_qualname( node.target.id ),
				file = self.module_stack[-1].file,
				line = node.lineno,
				type = var_type,
			)
			scope: Module|CEnum|RCClass|Function = self.scope_stack[-1]
			scope.add_name( var_obj.stem, var_obj )
			self.registry.resolve( var_obj )
	
	def visit_ClassDef( self, node: ast.ClassDef ) -> None:
		qualname = self._get_qualname( node.name )
		
		for decorator in node.decorator_list or []:
			#print( f'{decorator=}' )
			decname: str|None = None
			if isinstance( decorator, ast.Name ):
				decname = decorator.id
			elif isinstance( decorator, ast.Call ):
				func = decorator.func
				if isinstance( func, ast.Name ):
					decname = func.id
			match decname:
				case 'cstruct':
					self._parse_ClassDef_CStruct( node, qualname )
					return
				case 'cunion':
					self._parse_ClassDef_CUnion( node, qualname )
					return
				case 'enum':
					assert isinstance( decorator, ast.Call ), f'invalid @enum {decorator=}'
					assert len( decorator.args ) == 1, f'@enum decorator must have exactly 1 argument'
					#print( f'{decorator.args[0]=}' )
					value_type = self.visit( decorator.args[0] )
					assert isinstance( value_type, Scalar ), f'invalid @enum {value_type=} (must be a scalar like i32)'
					self._parse_ClassDef_CEnum( node, qualname, value_type )
					return
				case 'union':
					self._parse_ClassDef_TaggedUnion( node, qualname )
					return
				case _:
					assert False, f'unsupported class decorator {ast.unparse(decorator)} in {qualname}'
		
		# if we get here, no decorators means this is a normal RC'd class object
		self._parse_ClassDef_RCClass( node, qualname )
	
	def _parse_ClassDef_CEnum( self, node: ast.ClassDef, qualname: str, value_type: Scalar ) -> None:
		#print( f'{node=}' )
		class_obj = CEnum(
			stem = node.name,
			qualname = qualname,
			file = self.module_stack[-1].file,
			line = node.lineno,
			value_type = value_type,
			members = {},
			values = {},
			body = None,
		)
		assert not node.bases, f'@enum {qualname} cannot have a base classes ({node.bases!r})'
		assert not node.keywords, f'@enum {qualname} cannot have keywords ({node.keywords!r})'
		
		scope = self.scope_stack[-1]
		scope.add_name( class_obj.stem, class_obj )
		self.registry.resolve( class_obj )
		
		class_obj.body = self._scan_class_body_for_subclasses( class_obj, node.body )
		
		return class_obj
	
	def _parse_ClassDef_CStruct( self, node: ast.ClassDef, qualname: str ) -> None:
		#print( f'{node=}' )
		class_obj = CStruct(
			stem = node.name,
			qualname = qualname,
			file = self.module_stack[-1].file,
			line = node.lineno,
			body = None,
		)
		assert not node.bases, f'@cstruct {qualname} cannot have a base classes ({node.bases!r})'
		assert not node.keywords, f'@cstruct {qualname} cannot have keywords ({node.keywords!r})'
		
		scope = self.scope_stack[-1]
		scope.add_name( class_obj.stem, class_obj )
		self.registry.resolve( class_obj )
		
		class_obj.body = self._scan_class_body_for_subclasses( class_obj, node.body )
		
		return class_obj
	
	# TODO FIXME: the following code needs to move to the compiler where we start processing class/function bodies on demand
	#def _postprocess_CEnum_body( self, class_obj: CEnum, body: list[ast.AST] ) -> None:
	#	qualname = class_obj.qualname
	#	for node2 in body:
	#		if isinstance( node2, ast.Assign ):
	#			assert len( node2.targets ) == 1, f'multiple targets unsupported in {qualname}: {ast.unparse(node2)}'
	#			target = node2.targets[0]
	#			assert isinstance( target, ast.Name ), f'enum member target must be a Name, not {target=} in {qualname}'
	#			assert isinstance( target.ctx, ast.Store ) # this shouldn't be possible
	#			key = target.id
	#			value_obj = node2.value
	#			if isinstance( value_obj, ast.Name ) and value_obj.id == '_':
	#				value: int|None = None
	#			else:
	#				assert isinstance( value_obj, ast.Constant ), f"enum key {qualname}.{key} must be a '_' or an integer constant, not {value_obj=}"
	#				assert value_obj.kind is None, f'unexpected {value_obj.kind=} in {qualname}.{key}'
	#				value = value_obj.value
	#				assert isinstance( value, int ), f'enum key {qualname}.{key} value must be an integer, not {value_obj.value=}'
	#			# TODO FIXME: assert value fits in value_type... this will require Scalar() to store its min/max values
	#			assert value not in class_obj.values, f'enum {qualname} has duplicated value {value!r} from both {qualname}.{key} and {qualname}.{class_obj.values[value]}'
	#			assert key not in class_obj.members, f'enum {qualname}.{key} is duplicated'
	#			class_obj.members[key] = value
	#			class_obj.values[value] = key
	#		else:
	#			assert False, f'unsupported syntax {node2=} in @enum {qualname}'
	
	def _parse_ClassDef_CUnion( self, node: ast.ClassDef, qualname: str ) -> None:
		#print( f'{node=}' )
		class_obj = CUnion(
			stem = node.name,
			qualname = qualname,
			file = self.module_stack[-1].file,
			line = node.lineno,
			body = None,
		)
		assert not node.bases, f'@cunion {qualname} cannot have a base classes ({node.bases!r})'
		assert not node.keywords, f'@cunion {qualname} cannot have keywords ({node.keywords!r})'
		
		scope = self.scope_stack[-1]
		scope.add_name( class_obj.stem, class_obj )
		self.registry.resolve( class_obj )
		
		class_obj.body = self._scan_class_body_for_subclasses( class_obj, node.body )
		
		return class_obj
	
	def _parse_ClassDef_RCClass( self, node: ast.ClassDef, qualname: str ) -> None:
		class_obj = RCClass(
			stem = node.name,
			qualname = qualname,
			file = self.module_stack[-1].file,
			line = node.lineno,
			type_params = None,
			attributes = [],
			methods = [],
			names = {},
			body = None,
		)
		#assert not node.bases, f'class {qualname} cannot have a base class yet ({node.bases!r})'
		if node.bases:
			print( f'WARNING: subclassing not implemented yet ({qualname} wants to subclass {node.bases[0]})' )
		assert not node.keywords, f'class {qualname} cannot have keywords ({node.keywords!r})'
		
		scope = self.scope_stack[-1]
		scope.add_name( class_obj.stem, class_obj )
		self.registry.resolve( class_obj )
		
		if node.type_params:
			class_obj.type_params = []
			for type_param in node.type_params:
				assert isinstance( type_param, ast.TypeVar ), f'unsupported {type_param=} in {class_obj.qualname}'
				assert type_param.bound is None, f'TypeVar(bound=not None) not supported in {class_obj.qualname}'
				assert type_param.default_value is None, f'TypeVar(default_value=not None) not supported in {class_obj.qualname}'
				class_obj.type_params.append( type_param.name )
		
		class_obj.body = self._scan_class_body_for_subclasses( class_obj, node.body )

	# TODO FIXME: the following logic needs to move to compiler where we start processing class/function bodies on demand
	#def _postprocess_RCClass_body( self, class_obj: RCClass, body: list[ast.AST] ) -> None:
	#	self.scope_stack.append( class_obj )
	#	
	#	for item in body:
	#		if isinstance( item, ast.AnnAssign ) and isinstance( item.target, ast.Name ):
	#			attr_type = self.visit( item.annotation )
	#			assert isinstance( attr_type, Type ), f'class {class_obj.qualname} attribute {item.target.id} has invalid type {attr_type!r}'
	#			# NOTE: we can't punt directly to registry.get_or_create() because type_str isn't a qualname,
	#			# so we have to call find_name() to search through our available namespaces to translate it to a qualname
	#			#attr_type = self.registry.get_or_create( qualname = type_str ) # TODO FIXME: I don't think this is right...
	#			class_obj.attributes.append(
	#				Variable(
	#					stem = item.target.id,
	#					qualname = self._get_qualname( item.target.id ),
	#					file = self.filename,
	#					line = item.lineno,
	#					type = attr_type,
	#				)
	#			)
	#		elif isinstance( item, ast.FunctionDef ):
	#			class_obj.methods.append( self._parse_function( item, class_obj ))
	#	
	#	self.scope_stack.pop()
	
	def _scan_class_body_for_subclasses( self, class_obj: CEnum|RCClass|CStruct|CUnion|TaggedUnion, body: list[ast.AST] ) -> list[ast.AST]:
		# we only do a minimal scan of class bodies for child class definitions
		# we don't want to process attributes or functions yet because we haven't finished collecting type information yet
		with self.scope_context( class_obj ):
			unprocessed: list[ast.AST] = []
			for node in body:
				if isinstance( node, ast.ClassDef ):
					self.visit( node )
				else:
					unprocessed.append( node )
			return unprocessed
	
	def visit_FunctionDef( self, node: ast.FunctionDef ) -> None:
		self._parse_function( node )
	
	def visit_AsyncFunctionDef( self, node: ast.AsyncFunctionDef ) -> None:
		raise SyntaxError( 'async functions not supported' )
	
	def _parse_function(
		self,
		node: ast.FunctionDef,
		class_obj: CEnum|RCClass|CStruct|CUnion|TaggedUnion|None = None,
	) -> Function:
		# NOTE: the name 'main' is special, there can be only one...
		qualname = 'main' if node.name == 'main' else self._get_qualname( node.name )
		
		fn = Function(
			stem = node.name,
			qualname = qualname,
			cls = class_obj,
			file = self.module_stack[-1].file,
			line = node.lineno,
			parameters = [],
			return_type = None,
			body_ast = node.body,
			names = {},
		)
		
		scope = self.scope_stack[-1]
		scope.add_name( fn.stem, fn )
		self.registry.resolve( fn )
		
		with self.scope_context( fn ):
			for arg in node.args.args:
				if arg.arg == 'self':
					continue
				type_str = ast.unparse( arg.annotation ) if arg.annotation else None
				param_type = self.registry.get_or_create( qualname = type_str )
				fn.parameters.append( Variable(
					stem = arg.arg,
					qualname = self._get_qualname( arg.arg ),
					file = self.module_stack[-1].file,
					line = node.lineno,
					type = param_type,
				))
			
			if node.returns:
				fn.return_type = self.registry.get_or_create(
					qualname = ast.unparse( node.returns ),
				)
		
		return fn