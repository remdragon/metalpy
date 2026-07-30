# stdlib imports:
import ast
from contextlib import contextmanager, nullcontext
import itertools
import platform
from pathlib import Path
from typing import Any, Callable, Generator

# local imports
from mpy_types import (
	Name, Type, Scalar, TypeVar, Specialization, Variable, Parameter, Move, Function, Overload,
	CEnum, RCClass, CStruct, CUnion, TaggedUnion, ClassLike,
	Module, _is_covered_by, _overlaps,
)

class CompilerModule( Module ):
	' TODO FIXME: put target object here and anything else needed'


_HOST_OS_TO_TARGET_OS = {
	'Windows': 'windows',
	'Linux': 'linux',
	'Darwin': 'macos',
}
_HOST_MACHINE_TO_TARGET_ARCH = {
	'AMD64': 'x86_64',
	'x86_64': 'x86_64',
	'arm64': 'arm64',
	'aarch64': 'arm64',
}

def _detect_active_target() -> dict[str,str]:
	os_name = _HOST_OS_TO_TARGET_OS.get( platform.system(), platform.system().lower() )
	arch = _HOST_MACHINE_TO_TARGET_ARCH.get( platform.machine(), platform.machine() )
	family = 'windows' if os_name == 'windows' else 'posix'
	return { 'os': os_name, 'arch': arch, 'family': family, 'bits': '64' }


class Discovery( ast.NodeVisitor ):
	'''
	This class does a shallow parse of python files and processes all imports on demand
	The goal is to build a complete registry of all potential types in the system
	This class does not parse function bodies at all

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

	A module body is scanned immediately: every top-level class/function/
	global it directly contains is registered right away (that's just "this
	name exists, in this scope"), so cross-references anywhere in the program
	can always find each other regardless of source order. A class's own
	body is a different story - like a function's parameters or a variable's
	type, scanning it is deferred behind a `.resolve` callable set up when
	the class itself is created, invoked whenever something actually needs
	to know what's inside. `.resolve is None` means already resolved (or
	never needed resolving). The one exception is a generic's `type_params`,
	parsed eagerly at creation time - external code subscripting a class as
	a generic (`Result[i32,usize]`) needs to see it before that class's own
	`.resolve()` has ever run.
	'''
	log_unhandled: bool = False

	_intrinsics: dict[str,Name]|None = None
	_none_type: Scalar|None = None

	def __init__( self,
		paths: list[Path]|None = None,
		import_builtins: bool = True,
		active_target: dict[str,str]|None = None,
	) -> None:
		self.paths: list[Path] = list( paths ) if paths else []
		if not self.paths:
			self.paths.append( Path( __file__ ).parent / 'lib' )
			self.paths.append( Path( '.' ))
		self.active_target = active_target if active_target is not None else _detect_active_target()
		self.compiler_module = CompilerModule(
			stem = 'compiler',
			qualname = 'compiler',
			file = None,
			line = None,
			intrinsics = {},
			builtins = None,
		)
		self.modules: dict[str,Module] = {}
		self.main: Function|None = None

		self.module_stack: list[Module] = []
		self.scope_stack: list[Module|ClassLike|Function] = []

		# dedup caches for compound types built from other types on the fly
		# (anonymous unions, generic specializations, move[T] wrappers) - never
		# looked up by qualname from outside, only reused when the exact same
		# combination is seen again
		self._unions: dict[str,TaggedUnion] = {}
		self._specializations: dict[str,Specialization] = {}
		self._moves: dict[str,Move] = {}

		if import_builtins:
			# just for the side effect of populating self.modules['builtins'] -
			# import_code() looks it up from there directly (see below), so
			# nothing needs to be kept here on the Discovery instance itself
			self.import_name( 'builtins' )

	@contextmanager
	def scope_context( self, scope: Module|ClassLike|Function ) -> Generator[None,None,None]:
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

	def import_file( self, filename: Path, scope: str|None = None, package: str|None = None ) -> Module:
		with filename.open( 'r' ) as f:
			code = f.read()
		return self.import_code( code, filename, scope, package = package )

	def import_code( self, code: str, filename: Path, scope: str|None = None, package: str|None = None ) -> Module:
		stem = filename.stem if filename else ''
		qualname = ( f'{scope}.{stem}' if scope else stem )
		# looked up from self.modules, not a dedicated Discovery.builtins
		# attribute - self.modules['builtins'] is registered (see below)
		# before builtins' own body is scanned, so this is already there by
		# the time anything builtins itself transitively imports (codecs,
		# sys, ...) gets to this point. while builtins is still being
		# imported for the first time ever, this is None - correct, since
		# builtins doesn't need a fallback to itself
		builtins_mod = self.modules.get( 'builtins' )
		module = Module(
			stem = stem,
			qualname = qualname,
			file = filename,
			line = None,
			intrinsics = self.get_intrinsics(),
			builtins = builtins_mod.names if builtins_mod else None,
		)
		if package is not None:
			# register before scanning the body: if this module (transitively)
			# imports itself, that import_name() call must find this same
			# (still being scanned) Module here instead of recursing forever
			self.modules[package] = module
		with self.module_context( module ):
			tree = ast.parse( code )
			# scope_stack[-1] (the module) owns this body - every statement is
			# dispatched through self.visit(), which registers what it finds
			# immediately (structurally) and defers only the deep internals
			for node in tree.body:
				self.visit( node )

		return module

	def import_name( self,
		package: str,
	) -> Module:
		if package == 'compiler':
			return self.compiler_module

		if mod := self.modules.get( package, None ):
			return mod

		relpath = package.replace( '.', '/' )
		looked: list[str] = []
		suffixes = [ '.mpy', '.py' ] # TODO FIXME: this kinda sucks for performance...
		for base in self.paths:
			path = base / relpath
			if path.is_dir():
				stem = '__init__'
				qualname = f'{package}.{stem}'
			else:
				stem = path.stem
				path = path.parent
				qualname = '.'.join([ package.rpartition( '.' )[0], stem ]).lstrip( '.' )
			for suffix in suffixes:
				filename = path / f'{stem}{suffix}'
				if filename.is_file():
					return self.import_file( filename, scope = qualname.rpartition( '.' )[0], package = package )
				else:
					looked.append( str( filename ))
		e = FileNotFoundError( package )
		e.add_note( f'looked in:\n\t{"\n\t".join(looked)}' )
		raise e

	def get_intrinsics( self ) -> dict[str,Name]:
		if self._intrinsics is None:
			intrinsics: dict[str,Name] = {}
			for name in [ 'isize', 'usize', 'i8', 'u8', 'i16', 'u16', 'i32', 'u32', 'i64', 'u64', 'i128', 'u128' ]:
				intrinsics[name] = Scalar(
					stem = name,
					qualname = f'intrinsics.{name}',
					file = None,
					line = None,
				)
			for name in [ 'Ptr', 'ConstPtr' ]: # generic pointer intrinsics: Ptr[T], ConstPtr[T]
				tv = TypeVar(
					stem = 'T',
					qualname = f'intrinsics.{name}.T',
					file = None,
					line = None,
				)
				intrinsics[name] = Scalar(
					stem = name,
					qualname = f'intrinsics.{name}',
					file = None,
					line = None,
					type_params = [ tv ],
				)
			self._intrinsics = intrinsics
		return self._intrinsics

	def get_none_type( self ) -> Scalar:
		if self._none_type is None:
			self._none_type = Scalar(
				stem = 'NoneType',
				qualname = 'intrinsics.NoneType',
				file = None,
				line = None,
			)
		return self._none_type

	def _get_qualname( self, name: str ) -> str:
		scope = self.scope_stack[-1].qualname
		return f'{scope}.{name}' if scope else name

	def find_name( self, name: str, ctx: ast.AST ) -> Name:
		mod = self.module_stack[-1]
		# mod.builtins is already the target module's names dict (see
		# import_code) - not a Module needing a further .names unwrap
		builtins = mod.builtins
		for scope in itertools.chain(
			[ scope.names for scope in self.scope_stack[::-1] ],
			[ builtins if builtins else {} ],
			[ mod.intrinsics ],
		):
			assert isinstance( scope, dict ), f'invalid {scope=}'
			if name_obj := scope.get( name ):
				return name_obj
		e = NameError( name )
		e.add_note( f'{str(mod.file)}:{ctx.lineno}' )
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

	# --- annotation/expression resolution -----------------------------------
	# these are what self.visit(expr) dispatches through; every annotation
	# site (params, returns, AnnAssign) and every simple value-expression site
	# (a bare Assign's rvalue) just calls self.visit() on the expression node
	# and inspects whatever comes back.

	def visit_Name( self, node: ast.Name ) -> Name:
		assert isinstance( node.ctx, ast.Load ), f'invalid context on {node=}'
		name = self.find_name( node.id, node )
		assert isinstance( name, Name ), f'invalid {name=} from {node=}'
		return name

	def visit_Constant( self, node: ast.Constant ) -> Type:
		# in annotation position only None legally appears here (-> None,
		# T|None) - anywhere else this is a value expression's literal, whose
		# type we can resolve to a real builtin the same way
		value = node.value
		if value is None:
			return self.get_none_type()
		if isinstance( value, bool ):
			name = 'bool'
		elif isinstance( value, int ):
			name = 'int'
		elif isinstance( value, float ):
			name = 'float'
		elif isinstance( value, str ):
			name = 'str'
		else:
			assert False, f'cannot resolve type of constant {value!r}'
		return self.find_name( name, node )

	def visit_Attribute( self, node: ast.Attribute ) -> Name:
		base = self.visit( node.value )
		names = getattr( base, 'names', None )
		assert isinstance( names, dict ), f'{base!r} has no members, cannot look up {node.attr!r}'
		name_obj = names.get( node.attr )
		assert name_obj is not None, f'{base.qualname} has no member {node.attr!r}'
		return name_obj

	def visit_BinOp( self, node: ast.BinOp ) -> TaggedUnion:
		assert isinstance( node.op, ast.BitOr ), f'unsupported binary operator in type position: {ast.unparse(node)}'
		operand_nodes = self._flatten_union( node )
		operands = [ self.visit( operand ) for operand in operand_nodes ]
		return self._get_or_create_union( operands )

	def _flatten_union( self, node: ast.expr ) -> list[ast.expr]:
		if isinstance( node, ast.BinOp ) and isinstance( node.op, ast.BitOr ):
			return self._flatten_union( node.left ) + self._flatten_union( node.right )
		return [ node ]

	def _get_or_create_union( self, operands: list[Type] ) -> TaggedUnion:
		# canonicalize per ARCHITECTURE.md: sort operand qualnames asciibetically
		# and join with '|' (str|int -> builtins.int|builtins.str), dedupe on
		# that key. No cname assigned here - that's a stage-2 scheduling concern.
		ordered = sorted( operands, key = lambda t: t.qualname )
		key = '|'.join( t.qualname for t in ordered )
		if union := self._unions.get( key ):
			return union
		union = TaggedUnion(
			stem = key,
			qualname = key,
			file = None,
			line = None,
			attributes = [
				Variable(
					stem = t.stem,
					qualname = f'{key}.{t.stem}',
					file = None,
					line = None,
					type = t,
				)
				for t in ordered
			],
		)
		self._unions[key] = union
		return union

	def visit_Subscript( self, node: ast.Subscript ) -> Specialization|Move:
		# move[T] is compiler syntax, not a real generic lookup - recognized
		# textually here the same way @move is recognized textually as a
		# decorator name in _parse_function, rather than resolved through
		# find_name like an ordinary generic base would be
		if isinstance( node.value, ast.Name ) and node.value.id == 'move':
			assert not isinstance( node.slice, ast.Tuple ), f'move[...] takes exactly one type argument: {ast.unparse(node)}'
			inner = self.visit( node.slice )
			return self._get_or_create_move( inner )

		base = self.visit( node.value )
		type_params = getattr( base, 'type_params', None )
		assert type_params, f'{base.qualname} is not generic, cannot subscript it'

		slice_node = node.slice
		arg_nodes = slice_node.elts if isinstance( slice_node, ast.Tuple ) else [ slice_node ]
		assert len( arg_nodes ) == len( type_params ), (
			f'{base.qualname} expects {len(type_params)} type argument(s), got {len(arg_nodes)}'
		)

		args = [ self.visit( arg_node ) for arg_node in arg_nodes ]
		return self._get_or_create_specialization( base, args )

	def _get_or_create_move( self, inner: Type ) -> Move:
		key = f'move[{inner.qualname}]'
		if mv := self._moves.get( key ):
			return mv
		mv = Move(
			stem = key,
			qualname = key,
			file = inner.file,
			line = inner.line,
			inner = inner,
		)
		self._moves[key] = mv
		return mv

	def _get_or_create_specialization( self, base: Type, args: list[Type] ) -> Specialization:
		key = f'{base.qualname}[{",".join( a.qualname for a in args )}]'
		if spec := self._specializations.get( key ):
			return spec
		spec = Specialization(
			stem = key,
			qualname = key,
			file = base.file,
			line = base.line,
			base = base,
			args = args,
		)
		self._specializations[key] = spec
		return spec

	# --- imports --------------------------------------------------------------

	def visit_Import( self, node: ast.Import ) -> None:
		# import foo -> Import(names=[alias(name='foo', asname=None)])
		# import foo.bar -> Import(names=[alias(name='foo.bar', asname=None)])
		# import foo as bar -> Import(names=[alias(name='foo', asname='bar')])
		# import foo.bar as baz -> Import(names=[alias(name='foo.bar', asname='baz')])
		scope = self.scope_stack[-1]
		for alias in node.names:
			mod = self.import_name( alias.name )
			scope.add_name( alias.asname or alias.name, mod )

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

	# --- globals / attributes ---------------------------------------------------------------

	def visit_AnnAssign( self, node: ast.AnnAssign ) -> Variable:
		assert isinstance( node.target, ast.Name ), f'unsupported AnnAssign target {node.target!r}'
		module = self.module_stack[-1]
		scope = self.scope_stack[-1]
		var_obj = Variable(
			stem = node.target.id,
			qualname = self._get_qualname( node.target.id ),
			file = module.file,
			line = node.lineno,
			init = node.value,
		)
		var_obj.resolve = self._make_annotation_resolver( var_obj, node.annotation, module, scope )
		scope.add_name( var_obj.stem, var_obj )
		if hasattr( scope, 'attributes' ):
			scope.attributes.append( var_obj )
		return var_obj

	def _make_annotation_resolver( self, var_obj: Variable, annotation: ast.expr, module: Module, scope: Module|ClassLike|Function ) -> Callable[[],None]:
		def resolve() -> None:
			with self.module_context( module ):
				with ( self.scope_context( scope ) if scope is not module else nullcontext() ):
					var_obj.type = self.visit( annotation )
			var_obj.resolve = None
		return resolve

	def visit_Assign( self, node: ast.Assign ) -> Name|None:
		scope = self.scope_stack[-1]
		if isinstance( scope, CEnum ):
			self._register_enum_member( scope, node )
			return None

		# this stage does not parse function bodies, so this is either a
		# global variable or a class attribute with no annotation - its type
		# defers to whatever self.visit() resolves the rvalue expression to
		assert len( node.targets ) == 1, f'multiple assignment targets not supported: {ast.unparse(node)}'
		target = node.targets[0]
		assert isinstance( target, ast.Name ), f'unsupported Assign target {target!r}'
		module = self.module_stack[-1]
		var_obj = Variable(
			stem = target.id,
			qualname = self._get_qualname( target.id ),
			file = module.file,
			line = node.lineno,
			init = node.value,
		)
		var_obj.resolve = self._make_value_resolver( var_obj, node.value, module, scope )
		scope.add_name( var_obj.stem, var_obj )
		if hasattr( scope, 'attributes' ):
			scope.attributes.append( var_obj )
		return var_obj

	def _make_value_resolver( self, var_obj: Variable, value: ast.expr, module: Module, scope: Module|ClassLike|Function ) -> Callable[[],None]:
		def resolve() -> None:
			with self.module_context( module ):
				with ( self.scope_context( scope ) if scope is not module else nullcontext() ):
					resolved = self.visit( value )
			assert isinstance( resolved, Name ), (
				f'cannot resolve type of {ast.unparse(value)} (add an annotation instead)'
			)
			if isinstance( resolved, Variable ):
				if resolved.resolve is not None: # e.g. `Y = X` where X hasn't been resolved yet
					resolved.resolve()
				var_obj.type = resolved.type
			else:
				var_obj.type = resolved
			var_obj.resolve = None
		return resolve

	def _register_enum_member( self, cls: CEnum, node: ast.Assign ) -> None:
		# only reached from inside cls's own .resolve (see _make_class_resolver)
		# - so registration is deferred along with the rest of the class body.
		# once that runs, members are self-contained
		# (just integer literals / the '_' auto-increment sentinel), so
		# there's nothing forward-reference-sensitive left needing a further
		# per-member .resolve
		assert len( node.targets ) == 1, f'multiple targets unsupported in {cls.qualname}: {ast.unparse(node)}'
		target = node.targets[0]
		assert isinstance( target, ast.Name ), f'enum member target must be a Name, not {target=} in {cls.qualname}'
		key = target.id
		value_expr = node.value
		if isinstance( value_expr, ast.Name ) and value_expr.id == '_':
			value: int|None = None
		else:
			assert isinstance( value_expr, ast.Constant ), (
				f"enum key {cls.qualname}.{key} must be '_' or an integer constant, not {value_expr=}"
			)
			value = value_expr.value
			assert isinstance( value, int ), f'enum key {cls.qualname}.{key} value must be an integer, not {value_expr.value=}'
		if value is None:
			value = cls.next_auto
		assert value not in cls.values, (
			f'enum {cls.qualname} has duplicated value {value!r} from both {cls.qualname}.{key} and {cls.qualname}.{cls.values[value]}'
		)
		assert key not in cls.members, f'enum {cls.qualname}.{key} is duplicated'
		cls.members[key] = value
		cls.values[value] = key
		cls.next_auto = value + 1

	# --- classes ----------------------------------------------------------------

	def visit_ClassDef( self, node: ast.ClassDef ) -> ClassLike:
		qualname = self._get_qualname( node.name )

		for decorator in node.decorator_list or []:
			decname = self._decorator_name( decorator )
			match decname:
				case 'cstruct':
					return self._parse_ClassDef_CStruct( node, qualname )
				case 'cunion':
					return self._parse_ClassDef_CUnion( node, qualname )
				case 'enum':
					assert isinstance( decorator, ast.Call ), f'invalid @enum {decorator=}'
					assert len( decorator.args ) == 1, f'@enum decorator must have exactly 1 argument'
					value_type = self.visit( decorator.args[0] )
					assert isinstance( value_type, Scalar ), f'invalid @enum {value_type=} (must be a scalar like i32)'
					return self._parse_ClassDef_CEnum( node, qualname, value_type )
				case 'union':
					return self._parse_ClassDef_TaggedUnion( node, qualname )
				case _:
					assert False, f'unsupported class decorator {ast.unparse(decorator)} in {qualname}'

		# if we get here, no decorators means this is a normal RC'd class object
		return self._parse_ClassDef_RCClass( node, qualname )

	def _decorator_name( self, decorator: ast.expr ) -> str|None:
		if isinstance( decorator, ast.Name ):
			return decorator.id
		if isinstance( decorator, ast.Call ) and isinstance( decorator.func, ast.Name ):
			return decorator.func.id
		return None

	def _parse_type_params( self, type_params: list[ast.type_param], owner: RCClass|CStruct|CUnion|Function ) -> None:
		if not type_params:
			return
		owner.type_params = []
		for type_param in type_params:
			assert isinstance( type_param, ast.TypeVar ), f'unsupported {type_param=} in {owner.qualname}'
			assert type_param.bound is None, f'TypeVar(bound=not None) not supported in {owner.qualname}'
			assert type_param.default_value is None, f'TypeVar(default_value=not None) not supported in {owner.qualname}'
			tv = TypeVar(
				stem = type_param.name,
				qualname = f'{owner.qualname}.{type_param.name}',
				file = owner.file,
				line = owner.line,
			)
			owner.type_params.append( tv )
			owner.add_name( type_param.name, tv )
	
	def _shallow_class_body_scan( self,
		class_obj: ClassLike,
		body: list[ast.AST],
	) -> tuple[list[ast.AST],Callable[[],None]]:
		# we only do a minimal scan of class bodies for nested inner class definitions
		# we don't want to process attributes or functions yet because we haven't finished collecting type information yet
		with self.scope_context( class_obj ):
			unprocessed: list[ast.AST] = []
			for node in body:
				if isinstance( node, ast.ClassDef ):
					self.visit( node )
				else:
					unprocessed.append( node )
			return unprocessed

	def _make_class_resolver( self, class_obj: ClassLike, body: list[ast.stmt], module: Module ) -> Callable[[],None]:
		def resolve() -> None:
			with self.module_context( module ):
				with self.scope_context( class_obj ):
					# scope_stack[-1] (class_obj) owns this body - see import_code()
					for node in body:
						self.visit( node )
			class_obj.resolve = None
		return resolve

	def _parse_ClassDef_CEnum( self, node: ast.ClassDef, qualname: str, value_type: Scalar ) -> CEnum:
		module = self.module_stack[-1]
		class_obj = CEnum(
			stem = node.name,
			qualname = qualname,
			file = module.file,
			line = node.lineno,
			value_type = value_type,
		)
		assert not node.bases, f'@enum {qualname} cannot have a base classes ({node.bases!r})'
		assert not node.keywords, f'@enum {qualname} cannot have keywords ({node.keywords!r})'

		scope = self.scope_stack[-1]
		scope.add_name( class_obj.stem, class_obj )

		unresolved = self._shallow_class_body_scan( class_obj, node.body )

		class_obj.resolve = self._make_class_resolver( class_obj, unresolved, module )

		return class_obj

	def _parse_ClassDef_CStruct( self, node: ast.ClassDef, qualname: str ) -> CStruct:
		module = self.module_stack[-1]
		class_obj = CStruct(
			stem = node.name,
			qualname = qualname,
			file = module.file,
			line = node.lineno,
		)
		assert not node.bases, f'@cstruct {qualname} cannot have a base classes ({node.bases!r})'
		assert not node.keywords, f'@cstruct {qualname} cannot have keywords ({node.keywords!r})'

		scope = self.scope_stack[-1]
		scope.add_name( class_obj.stem, class_obj )

		self._parse_type_params( node.type_params, class_obj )

		unresolved = self._shallow_class_body_scan( class_obj, node.body )

		class_obj.resolve = self._make_class_resolver( class_obj, unresolved, module )

		return class_obj

	def _parse_ClassDef_CUnion( self, node: ast.ClassDef, qualname: str ) -> CUnion:
		module = self.module_stack[-1]
		class_obj = CUnion(
			stem = node.name,
			qualname = qualname,
			file = module.file,
			line = node.lineno,
		)
		assert not node.bases, f'@cunion {qualname} cannot have a base classes ({node.bases!r})'
		assert not node.keywords, f'@cunion {qualname} cannot have keywords ({node.keywords!r})'

		scope = self.scope_stack[-1]
		scope.add_name( class_obj.stem, class_obj )

		self._parse_type_params( node.type_params, class_obj )

		unresolved = self._shallow_class_body_scan( class_obj, node.body )

		class_obj.resolve = self._make_class_resolver( class_obj, unresolved, module )

		return class_obj

	def _parse_ClassDef_TaggedUnion( self, node: ast.ClassDef, qualname: str ) -> TaggedUnion:
		module = self.module_stack[-1]
		class_obj = TaggedUnion(
			stem = node.name,
			qualname = qualname,
			file = module.file,
			line = node.lineno,
		)
		assert not node.bases, f'@union {qualname} cannot have a base classes ({node.bases!r})'
		assert not node.keywords, f'@union {qualname} cannot have keywords ({node.keywords!r})'

		scope = self.scope_stack[-1]
		scope.add_name( class_obj.stem, class_obj )

		unresolved = self._shallow_class_body_scan( class_obj, node.body )

		class_obj.resolve = self._make_class_resolver( class_obj, unresolved, module )

		return class_obj

	def _parse_ClassDef_RCClass( self, node: ast.ClassDef, qualname: str ) -> RCClass:
		module = self.module_stack[-1]
		class_obj = RCClass(
			stem = node.name,
			qualname = qualname,
			file = module.file,
			line = node.lineno,
		)
		assert len( node.bases ) <= 1, (
			f'multiple inheritance not supported: class {qualname}({", ".join( ast.unparse(b) for b in node.bases )})'
		)
		assert not node.keywords, f'class {qualname} cannot have keywords ({node.keywords!r})'

		scope = self.scope_stack[-1]
		scope.add_name( class_obj.stem, class_obj )

		if node.bases:
			# resolved eagerly, in the enclosing scope, exactly like Python
			# itself requires the base to already exist when this statement runs
			base = self.visit( node.bases[0] )
			assert isinstance( base, RCClass ), (
				f'{qualname} cannot subclass {base.qualname} (only plain classes support inheritance)'
			)
			class_obj.base = base

		self._parse_type_params( node.type_params, class_obj )
		
		unresolved = self._shallow_class_body_scan( class_obj, node.body )

		class_obj.resolve = self._make_class_resolver( class_obj, unresolved, module )

		return class_obj

	# --- functions ----------------------------------------------------------

	def visit_FunctionDef( self, node: ast.FunctionDef ) -> Function|Overload|None:
		scope = self.scope_stack[-1]
		class_obj = scope if isinstance( scope, ( RCClass, CStruct, CUnion, TaggedUnion, CEnum )) else None
		return self._parse_function( node, class_obj )

	def visit_AsyncFunctionDef( self, node: ast.AsyncFunctionDef ) -> None:
		raise SyntaxError( 'async functions not supported' )

	def _is_compiler_target_call( self, decorator: ast.expr ) -> bool:
		return (
			isinstance( decorator, ast.Call )
			and isinstance( decorator.func, ast.Attribute )
			and decorator.func.attr == 'target'
			and isinstance( decorator.func.value, ast.Name )
			and decorator.func.value.id == 'compiler'
		)

	def _matches_active_target( self, call: ast.Call ) -> bool:
		for kw in call.keywords:
			if kw.arg not in self.active_target:
				continue # unmodeled keyword (e.g. arch, vendor) - inert placeholder for future cross-compilation support
			if not self._target_value_matches( kw.value, self.active_target[kw.arg] ):
				return False
		return True

	def _target_value_matches( self, node: ast.expr, active_value: str ) -> bool:
		if isinstance( node, ast.Constant ) and isinstance( node.value, str ):
			return node.value == active_value
		if isinstance( node, ast.UnaryOp ) and isinstance( node.op, ast.Not ):
			return not self._target_value_matches( node.operand, active_value )
		if isinstance( node, ast.Tuple ):
			return any( self._target_value_matches( elt, active_value ) for elt in node.elts )
		assert False, f'unsupported compiler.target(...) value: {ast.unparse(node)}'

	def _parse_function(
		self,
		node: ast.FunctionDef,
		class_obj: ClassLike|None = None,
	) -> Function|Overload|None:
		# NOTE: the name 'main' is special, there can be only one...
		qualname = 'main' if node.name == 'main' else self._get_qualname( node.name )

		is_overload = False
		is_static = False
		is_classmethod = False
		is_abstract = False
		is_move = False
		is_private = False
		for decorator in node.decorator_list or []:
			if self._is_compiler_target_call( decorator ):
				if not self._matches_active_target( decorator ):
					return None # excluded for this target - not part of the type system at all
				continue
			decname = self._decorator_name( decorator )
			match decname:
				case 'overload':
					is_overload = True
				case 'staticmethod':
					is_static = True
				case 'classmethod':
					is_classmethod = True
				case 'abstractmethod':
					is_abstract = True
				case 'move':
					is_move = True
				case 'private':
					is_private = True
				case _:
					assert False, f'unsupported function decorator @{decname or ast.unparse(decorator)} on {qualname}'

		module = self.module_stack[-1]
		fn = Function(
			stem = node.name,
			qualname = qualname,
			cls = class_obj,
			node = node,
			file = module.file,
			line = node.lineno,
			is_static = is_static,
			is_classmethod = is_classmethod,
			is_abstract = is_abstract,
			is_move = is_move,
			is_private = is_private,
			is_overload = is_overload,
		)
		if node.name == 'main':
			self.main = fn
		self._parse_type_params( node.type_params, fn )

		scope = self.scope_stack[-1]
		existing = scope.names.get( fn.stem )

		# group membership is settled before the resolver is created (below) so
		# it can be threaded straight into the closure, the same way module/
		# class_obj already are - no separate back-reference field needed on
		# Function itself.
		group: Overload|None = None
		if is_overload or isinstance( existing, Overload ):
			if isinstance( existing, Overload ):
				group = existing
			else:
				assert existing is None, f'{qualname} redefines {existing!r} as an overload group'
				group = Overload(
					stem = fn.stem,
					qualname = qualname,
					file = fn.file,
					line = fn.line,
					cls = class_obj,
				)
				scope.add_name( fn.stem, group )
				if class_obj is not None:
					class_obj.methods.append( group )
			if is_overload and self._is_stub_body( node.body ):
				group.stubs.append( fn )
			else:
				group.implementations.append( fn )

		fn.resolve = self._make_function_resolver( fn, module, class_obj, group )

		if group is not None:
			return group

		scope.add_name( fn.stem, fn )
		if class_obj is not None:
			class_obj.methods.append( fn )
		return fn

	def _is_stub_body( self, body: list[ast.stmt] ) -> bool:
		return (
			len( body ) == 1
			and isinstance( body[0], ast.Expr )
			and isinstance( body[0].value, ast.Constant )
			and body[0].value.value is Ellipsis
		)

	def _bind_overload_stub( self, stub: Function, group: Overload ) -> None:
		# a stub has no body of its own - it must resolve to exactly one plain
		# (non-@overload) implementation whose accepted types, at every
		# parameter position, are a superset of what the stub declares
		candidates: list[Function] = []
		for impl in group.implementations:
			if impl.is_overload:
				continue
			if impl.resolve is not None:
				impl.resolve()
			if _is_covered_by( stub, impl ):
				candidates.append( impl )
		assert len( candidates ) == 1, (
			f'{stub.qualname}: ambiguous overload binding - matches {[c.qualname for c in candidates]}'
			if candidates else
			f'{stub.qualname}: no implementation covers this @overload signature'
		)
		stub.bound_to = candidates[0]

	def _check_overload_shadowing( self, fn: Function, group: Overload ) -> None:
		# all @overload-decorated members (stubs and real-bodied ones alike)
		# are tried first-match, in declaration order, at a call site - if an
		# earlier one's domain already fully covers fn's domain, fn can never
		# be reached. only checks "am I shadowed by something earlier" - the
		# reciprocal gets covered when that earlier member's own resolve runs
		overload_members = sorted(
			[ *group.stubs, *( f for f in group.implementations if f.is_overload ) ],
			key = lambda f: f.line,
		)
		for earlier in overload_members:
			if earlier is fn or earlier.line >= fn.line:
				continue
			if earlier.resolve is not None:
				earlier.resolve()
			assert not _is_covered_by( fn, earlier ), (
				f'{fn.qualname} (line {fn.line}) is shadowed by {earlier.qualname} (line {earlier.line}) - '
				f'unreachable, every type it declares is already handled by the earlier overload'
			)

	def _check_overload_ambiguity( self, fn: Function, group: Overload ) -> None:
		# plain (non-@overload) implementations must be pairwise distinguishable
		# by parameter type - any overlap is ambiguous regardless of whether a
		# call ever actually exercises it
		for other in group.implementations:
			if other is fn or other.is_overload:
				continue
			if other.resolve is not None:
				other.resolve()
			assert not _overlaps( fn, other ), (
				f'{fn.qualname} and {other.qualname} are ambiguous - their parameter types overlap'
			)

	def _make_function_resolver( self, fn: Function, module: Module, class_obj: ClassLike|None, group: Overload|None = None ) -> Callable[[],None]:
		def resolve() -> None:
			with self.module_context( module ):
				with ( self.scope_context( class_obj ) if class_obj is not None else nullcontext() ):
					with self.scope_context( fn ):
						args = fn.node.args
						parameters: list[Parameter] = []

						def add_param( arg: ast.arg, default: ast.expr|None, **kind: bool ) -> None:
							if arg.arg == 'self' and not fn.is_static:
								return
							if arg.arg == 'cls' and fn.is_classmethod:
								return
							assert arg.annotation is not None, f'{fn.qualname} parameter {arg.arg!r} has no type annotation'
							param = Parameter(
								stem = arg.arg,
								qualname = self._get_qualname( arg.arg ),
								file = fn.file,
								line = fn.line,
								type = self.visit( arg.annotation ),
								default = default,
								**kind,
							)
							parameters.append( param )
							fn.add_name( param.stem, param )

						# `defaults` applies to the trailing N of posonlyargs+args
						# combined (an ast-module quirk) - left-pad with None so
						# every positional param lines up with its own default
						# (or lack of one)
						positional = [ *args.posonlyargs, *args.args ]
						defaults = [ None ] * ( len( positional ) - len( args.defaults )) + list( args.defaults )
						for i, arg in enumerate( positional ):
							add_param( arg, defaults[i], is_posonly = i < len( args.posonlyargs ))

						if args.vararg is not None:
							add_param( args.vararg, None, is_vararg = True )

						for arg, default in zip( args.kwonlyargs, args.kw_defaults ):
							add_param( arg, default, is_kwonly = True )

						if args.kwarg is not None:
							add_param( args.kwarg, None, is_kwarg = True )

						fn.parameters = parameters
						fn.return_type = self.visit( fn.node.returns ) if fn.node.returns is not None else self.get_none_type()
			# set self done *before* touching any overload siblings below - a
			# sibling's own resolve may need to cross-check back against fn,
			# and seeing fn.resolve is None already tells it not to re-enter
			# this closure (see _make_value_resolver for the same pattern)
			fn.resolve = None
			if group is not None:
				if fn in group.stubs:
					self._bind_overload_stub( fn, group )
					self._check_overload_shadowing( fn, group )
				elif fn.is_overload:
					self._check_overload_shadowing( fn, group )
				else:
					self._check_overload_ambiguity( fn, group )
		return resolve
