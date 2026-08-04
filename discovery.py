# stdlib imports:
import ast
from contextlib import contextmanager, nullcontext
import itertools
import platform
from pathlib import Path
from typing import Any, Callable, Generator, NoReturn

# local imports
import compile_time_transformer
from errors import CompileError, ErrorCollector
from mpy_types import (
	Name, Type, Scalar, TypeVar, Specialization, Variable, Parameter, Move, Copy, Function, Overload,
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

def _detect_active_target() -> dict[str,object]:
	os_name = _HOST_OS_TO_TARGET_OS.get( platform.system(), platform.system().lower() )
	arch = _HOST_MACHINE_TO_TARGET_ARCH.get( platform.machine(), platform.machine() )
	# family is one of SYNTAX.md's FamilySpec literals ('unix'/'windows'/
	# 'wasm') - 'posix' is a *separate* bool field on TargetQuery, not a
	# family value
	family = 'windows' if os_name == 'windows' else 'unix'
	# debug=True by default (matches sys.alloc()'s existing debug-only
	# zeroing behavior) - overridable via Discovery(active_target=...) same
	# as every other key, until a real CLI exposes a release-build flag
	return { 'os': os_name, 'arch': arch, 'family': family, 'bits': 64, 'debug': True, 'posix': family == 'unix' }


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
	parsed eagerly at creation time - external code subscripting this class
	as a generic (`Result[i32,usize]`) needs to see it before this class's
	own `.resolve()` has ever run.

	Errors: user-code problems (unsupported syntax, missing annotations,
	duplicate names, ...) are reported through self.errors (an
	ErrorCollector) via self.fail()/self.fail_loc() rather than raising
	uncaught - see _resolve_guarded and the try/except around the
	module/class body scan loops for where processing picks back up
	afterward. A handful of asserts stay plain asserts where noted - those
	guard invariants that malformed *source* can never actually trigger.
	'''
	log_unhandled: bool = False

	_intrinsics: dict[str,Name]|None = None
	_none_type: Scalar|None = None

	def __init__( self,
		paths: list[Path]|None = None,
		import_builtins: bool = True,
		active_target: dict[str,object]|None = None,
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
		self.errors = ErrorCollector()

		self.module_stack: list[Module] = []
		self.scope_stack: list[Module|ClassLike|Function] = []

		# dedup caches for compound types built from other types on the fly
		# (anonymous unions, generic specializations, move[T] wrappers) - never
		# looked up by qualname from outside, only reused when the exact same
		# combination is seen again
		self._unions: dict[str,TaggedUnion] = {}
		self._specializations: dict[str,Specialization] = {}
		self._moves: dict[str,Move] = {}
		self._copies: dict[str,Copy] = {}

		if import_builtins:
			# just for the side effect of populating self.modules['builtins'] -
			# import_code() looks it up from there directly (see below), so
			# nothing needs to be kept here on the Discovery instance itself
			self.import_name( 'builtins' )

	# --- error reporting -----------------------------------------------------

	def fail( self, message: str, node: ast.AST ) -> NoReturn:
		file = self.module_stack[-1].file if self.module_stack else None
		self.errors.fail( message, file, getattr( node, 'lineno', None ))

	def fail_loc( self, message: str, file: Path|None, line: int|None ) -> NoReturn:
		self.errors.fail( message, file, line )

	def _resolve_guarded( self, target: 'Function|ClassLike|Variable', body: Callable[[],None] ) -> None:
		# the shared recovery boundary every .resolve() closure runs through -
		# a CompileError raised (and already recorded) anywhere inside body()
		# is swallowed here so the caller that triggered this resolve() just
		# gets a partially-resolved object back instead of an uncaught
		# exception. target.resolve is always cleared afterward (even on
		# failure) so a broken symbol is only ever attempted once, not
		# re-attempted (and re-erroring) every time something references it
		try:
			body()
		except CompileError:
			pass
		finally:
			target.resolve = None

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
			# immediately (structurally) and defers only the deep internals.
			# one bad top-level statement doesn't stop the rest of the module
			# from being scanned - see _resolve_guarded for the same idea
			# applied to individual symbols' .resolve()
			for node in tree.body:
				try:
					self.visit( node )
				except CompileError:
					continue

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
			
			# TODO FIXME: make active_target.bits a requirement...
			sizeof_bits = self.active_target.get( 'bits', 64 ) // 8
			
			for name, sizeof in [
				( 'isize', sizeof_bits ), ( 'usize', sizeof_bits ),
				( 'i8', 1 ), ( 'u8', 1 ),
				( 'i16', 2 ), ( 'u16', 2 ),
				( 'i32', 4 ), ( 'u32', 4 ),
				( 'i64', 8 ), ( 'u64', 8 ),
				( 'i128', 16 ), ( 'u128', 16 ),
			]:
				intrinsics[name] = Scalar(
					stem = name,
					qualname = f'intrinsics.{name}',
					file = None,
					line = None,
					sizeof = sizeof,
				)
			# not a fixed-width integer, but the same tier as the numeric
			# scalars above rather than a lib/-defined class: ir.py's own
			# Const.value (bool|int|str|bytes|None) already treats it as a
			# first-class IR-level concept, and it's needed everywhere
			# comparisons/is_ok()/is_err()/defer's flag Variable are lowered,
			# regardless of which module is being compiled
			intrinsics['bool'] = Scalar(
				stem = 'bool',
				qualname = 'intrinsics.bool',
				file = None,
				line = None,
				sizeof = 1,
			)
			# a distinct marker from None/NoneType, not an alias for it - a
			# function declared -> NoReturn behaves exactly like -> None to
			# lowering today (no return value to stow/propagate), but stays
			# a separate, inspectable Scalar so a future emitter can tell
			# the two apart (e.g. to emit C's _Noreturn/[[noreturn]] so the
			# C compiler doesn't warn about a function that never returns,
			# like sys.panic()) - see lowering.py's own none_type-adjacent
			# checks, which treat this identically to NoneType for now
			intrinsics['NoReturn'] = Scalar(
				stem = 'NoReturn',
				qualname = 'intrinsics.NoReturn',
				file = None,
				line = None,
				sizeof = 0, # TODO FIXME: NoReturn isn't a Scalar...
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
					sizeof = sizeof_bits,
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
				sizeof = 0, # TODO FIXME: None isn't a Scalar
			)
		return self._none_type

	def _get_qualname( self, name: str ) -> str:
		scope = self.scope_stack[-1].qualname
		return f'{scope}.{name}' if scope else name

	def find_name_or_none( self, name: str ) -> Name|None:
		mod = self.module_stack[-1]
		# mod.builtins is already the target module's names dict (see
		# import_code) - not a Module needing a further .names unwrap
		builtins = mod.builtins
		for scope in itertools.chain(
			[ scope.names for scope in self.scope_stack[::-1] ],
			[ builtins if builtins else {} ],
			[ mod.intrinsics ],
		):
			assert isinstance( scope, dict ), f'invalid {scope=}' # internal invariant - every scope on the stack always has a .names dict
			if name_obj := scope.get( name ):
				return name_obj
		return None

	def find_name( self, name: str, ctx: ast.AST ) -> Name:
		found = self.find_name_or_none( name )
		if found is None:
			self.fail( f'name {name!r} is not defined', ctx )
		return found

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
		assert isinstance( node.ctx, ast.Load ), f'invalid context on {node=}' # internal invariant - Load is the only context an expression-position Name can have
		name = self.find_name( node.id, node )
		assert isinstance( name, Name ), f'invalid {name=} from {node=}' # internal invariant - every scope entry is a Name
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
			self.fail( f'cannot resolve type of constant {value!r}', node )
		return self.find_name( name, node )

	def visit_Attribute( self, node: ast.Attribute ) -> Name:
		base = self.visit( node.value )
		names = getattr( base, 'names', None )
		if not isinstance( names, dict ):
			self.fail( f'{base!r} has no members, cannot look up {node.attr!r}', node )
		name_obj = names.get( node.attr )
		if name_obj is None:
			self.fail( f'{base.qualname} has no member {node.attr!r}', node )
		return name_obj

	def visit_BinOp( self, node: ast.BinOp ) -> TaggedUnion:
		if not isinstance( node.op, ast.BitOr ):
			self.fail( f'unsupported binary operator in type position: {ast.unparse(node)}', node )
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

	def visit_Subscript( self, node: ast.Subscript ) -> Specialization|Move|Copy:
		# move[T]/copy[T] are compiler syntax, not a real generic lookup -
		# recognized textually here the same way @move is recognized
		# textually as a decorator name in _parse_function, rather than
		# resolved through find_name like an ordinary generic base would be
		if isinstance( node.value, ast.Name ) and node.value.id in ( 'move', 'copy' ):
			if isinstance( node.slice, ast.Tuple ):
				self.fail( f'{node.value.id}[...] takes exactly one type argument: {ast.unparse(node)}', node )
			inner = self.visit( node.slice )
			if node.value.id == 'move':
				return self._get_or_create_move( inner )
			return self._get_or_create_copy( inner )

		base = self.visit( node.value )
		type_params = getattr( base, 'type_params', None )
		if not type_params:
			self.fail( f'{base.qualname} is not generic, cannot subscript it', node )

		slice_node = node.slice
		arg_nodes = slice_node.elts if isinstance( slice_node, ast.Tuple ) else [ slice_node ]
		if len( arg_nodes ) != len( type_params ):
			self.fail( f'{base.qualname} expects {len(type_params)} type argument(s), got {len(arg_nodes)}', node )

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

	def _get_or_create_copy( self, inner: Type ) -> Copy:
		key = f'copy[{inner.qualname}]'
		if cp := self._copies.get( key ):
			return cp
		cp = Copy(
			stem = key,
			qualname = key,
			file = inner.file,
			line = inner.line,
			inner = inner,
		)
		self._copies[key] = cp
		return cp

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
			try:
				mod = self.import_name( alias.name )
			except FileNotFoundError as e:
				self.fail( str( e ), node )
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
			if not parts:
				self.fail( f'unable to relative import from here: {node=} {self.module_stack[-1].qualname=} {self.module_stack[-1].file=}', node )
		if node.module:
			parts.append( node.module )
		package = '.'.join( parts )
		#print( f'{package=}' )
		scope = self.scope_stack[-1]
		try:
			mod = self.import_name( package )
		except FileNotFoundError as e:
			self.fail( str( e ), node )
		if not mod:
			self.fail( f'module {package!r} not found', node )
		for alias in node.names:
			#print( f'{self.module_stack[-1].qualname=} {package=} {node.level=} {node.module=} {alias.name=}' )
			item = mod.names.get( alias.name )
			if not item:
				self.fail( f'module {package} does not export {alias.name!r}', node )
			scope.add_name( alias.asname or alias.name, item )

	# --- globals / attributes ---------------------------------------------------------------

	def visit_AnnAssign( self, node: ast.AnnAssign ) -> Variable|None:
		if not isinstance( node.target, ast.Name ):
			self.fail( f'unsupported AnnAssign target {node.target!r}', node )
		if isinstance( node.annotation, ast.Name ) and node.annotation.id == 'TypeAlias':
			self._parse_type_alias( node )
			return None
		module = self.module_stack[-1]
		scope = self.scope_stack[-1]
		# folded eagerly, same as a function body (_make_function_resolver) -
		# a global/attribute initializer is a bare expression, never
		# otherwise passed through compile_time_transformer at all, so
		# without this `X: u32 = u32(-11)` (a real WinAPI-style constant)
		# would never see its own compile-time-constant argument folded
		init = compile_time_transformer.transform_expr( node.value, self.active_target ) if node.value is not None else None
		var_obj = Variable(
			stem = node.target.id,
			qualname = self._get_qualname( node.target.id ),
			file = module.file,
			line = node.lineno,
			init = init,
			is_global = scope is module,
		)
		var_obj.resolve = self._make_annotation_resolver( var_obj, node.annotation, module, scope )
		scope.add_name( var_obj.stem, var_obj )
		if hasattr( scope, 'attributes' ):
			scope.attributes.append( var_obj )
		return var_obj

	def _parse_type_alias( self, node: ast.AnnAssign ) -> None:
		# X: TypeAlias = <type-expr> - TypeAlias is a compiler-recognized
		# sigil, not a real resolvable name (no `from typing import
		# TypeAlias` needed - recognized purely by AST shape in
		# visit_AnnAssign, same posture as compiler.target/compiler.sizeof).
		# X becomes a genuine alias - the SAME Type object <type-expr>
		# resolves to, registered directly under X's own name - not a new
		# nominal type and not a Variable, matching what TypeAlias means in
		# real Python. Resolved eagerly, right here, unlike an ordinary
		# AnnAssign's deferred _make_annotation_resolver - deliberately
		# simple for now: no forward-referencing a class/alias declared
		# later in the same file. lib/windows/kernel32.py's real motivating
		# case (`HANDLE: TypeAlias = Ptr[None]`) only references an
		# always-available intrinsic, so this covers it; a fully general
		# (deferred, forward-referencing) version is future work if a real
		# case ever needs one
		assert isinstance( node.target, ast.Name ) # already checked by visit_AnnAssign
		if node.value is None:
			self.fail( f'TypeAlias declaration needs a value: {ast.unparse(node)}', node )
		aliased = self.visit( node.value )
		if not isinstance( aliased, Type ):
			self.fail( f'TypeAlias value must be a type expression: {ast.unparse(node)}', node )
		scope = self.scope_stack[-1]
		scope.add_name( node.target.id, aliased )

	def _make_annotation_resolver( self, var_obj: Variable, annotation: ast.expr, module: Module, scope: Module|ClassLike|Function ) -> Callable[[],None]:
		def body() -> None:
			with self.module_context( module ):
				with ( self.scope_context( scope ) if scope is not module else nullcontext() ):
					var_obj.type = self.visit( annotation )
		def resolve() -> None:
			self._resolve_guarded( var_obj, body )
		return resolve

	def visit_Assign( self, node: ast.Assign ) -> Name|None:
		scope = self.scope_stack[-1]
		if isinstance( scope, CEnum ):
			self._register_enum_member( scope, node )
			return None

		# this stage does not parse function bodies, so this is either a
		# global variable or a class attribute with no annotation - its type
		# defers to whatever self.visit() resolves the rvalue expression to
		if len( node.targets ) != 1:
			self.fail( f'multiple assignment targets not supported: {ast.unparse(node)}', node )
		target = node.targets[0]
		if isinstance( target, ast.Attribute ):
			base = self.visit( target.value )
			if isinstance( base, Scalar ):
				# `usize.__u32__ = some_function` - a compiler-recognized
				# sigil (like TypeAlias/compiler.target), not ordinary
				# attribute assignment: registers a real method directly
				# into the shared intrinsic Scalar's own .names, resolved
				# eagerly (the RHS function must already be def'd earlier
				# in the same file, same top-to-bottom limitation
				# _parse_type_alias already has)
				value = self.visit( node.value )
				if not isinstance( value, Function ):
					self.fail( f'{ast.unparse(target)} = ... must assign a function: {ast.unparse(node)}', node )
				base.add_name( target.attr, value )
				return None
			# anything else with an Attribute target falls through to the
			# ordinary failure below, unchanged
		if not isinstance( target, ast.Name ):
			self.fail( f'unsupported Assign target {target!r}', node )
		module = self.module_stack[-1]
		# folded eagerly, same as a function body (_make_function_resolver) -
		# see the identical comment on visit_AnnAssign
		init = compile_time_transformer.transform_expr( node.value, self.active_target )
		var_obj = Variable(
			stem = target.id,
			qualname = self._get_qualname( target.id ),
			file = module.file,
			line = node.lineno,
			init = init,
			is_global = scope is module,
		)
		var_obj.resolve = self._make_value_resolver( var_obj, init, module, scope )
		scope.add_name( var_obj.stem, var_obj )
		if hasattr( scope, 'attributes' ):
			scope.attributes.append( var_obj )
		return var_obj

	def _make_value_resolver( self, var_obj: Variable, value: ast.expr, module: Module, scope: Module|ClassLike|Function ) -> Callable[[],None]:
		def body() -> None:
			with self.module_context( module ):
				with ( self.scope_context( scope ) if scope is not module else nullcontext() ):
					resolved = self.visit( value )
			if not isinstance( resolved, Name ):
				self.fail( f'cannot resolve type of {ast.unparse(value)} (add an annotation instead)', value )
			if isinstance( resolved, Variable ):
				if resolved.resolve is not None: # e.g. `Y = X` where X hasn't been resolved yet
					resolved.resolve()
				var_obj.type = resolved.type
			else:
				var_obj.type = resolved
		def resolve() -> None:
			self._resolve_guarded( var_obj, body )
		return resolve

	def _register_enum_member( self, cls: CEnum, node: ast.Assign ) -> None:
		# only reached from inside cls's own .resolve (see _make_class_resolver)
		# - so registration is deferred along with the rest of the class body.
		# once that runs, members are self-contained
		# (just integer literals / the '_' auto-increment sentinel), so
		# there's nothing forward-reference-sensitive left needing a further
		# per-member .resolve
		if len( node.targets ) != 1:
			self.fail( f'multiple targets unsupported in {cls.qualname}: {ast.unparse(node)}', node )
		target = node.targets[0]
		if not isinstance( target, ast.Name ):
			self.fail( f'enum member target must be a Name, not {target=} in {cls.qualname}', node )
		key = target.id
		value_expr = node.value
		if isinstance( value_expr, ast.Name ) and value_expr.id == '_':
			value: int|None = None
		else:
			if not isinstance( value_expr, ast.Constant ):
				self.fail( f"enum key {cls.qualname}.{key} must be '_' or an integer constant, not {value_expr=}", node )
			value = value_expr.value
			if not isinstance( value, int ):
				self.fail( f'enum key {cls.qualname}.{key} value must be an integer, not {value_expr.value=}', node )
		if value is None:
			value = cls.next_auto
		if value in cls.values:
			self.fail(
				f'enum {cls.qualname} has duplicated value {value!r} from both {cls.qualname}.{key} and {cls.qualname}.{cls.values[value]}',
				node,
			)
		if key in cls.members:
			self.fail( f'enum {cls.qualname}.{key} is duplicated', node )
		cls.members[key] = value
		cls.values[value] = key
		cls.next_auto = value + 1

	# --- classes ----------------------------------------------------------------

	def visit_ClassDef( self, node: ast.ClassDef ) -> ClassLike|None:
		qualname = self._get_qualname( node.name )

		for decorator in node.decorator_list or []:
			if self._is_compiler_target_call( decorator ):
				if not self._matches_active_target( decorator ):
					return None # excluded for this target - not part of the type system at all
				continue
			decname = self._decorator_name( decorator )
			match decname:
				case 'cstruct':
					return self._parse_ClassDef_CStruct( node, qualname )
				case 'cunion':
					return self._parse_ClassDef_CUnion( node, qualname )
				case 'enum':
					if not isinstance( decorator, ast.Call ):
						self.fail( f'invalid @enum {decorator=}', node )
					if len( decorator.args ) != 1:
						self.fail( '@enum decorator must have exactly 1 argument', node )
					value_type = self.visit( decorator.args[0] )
					if not isinstance( value_type, Scalar ):
						self.fail( f'invalid @enum {value_type=} (must be a scalar like i32)', node )
					return self._parse_ClassDef_CEnum( node, qualname, value_type )
				case 'union':
					return self._parse_ClassDef_TaggedUnion( node, qualname )
				case _:
					self.fail( f'unsupported class decorator {ast.unparse(decorator)} in {qualname}', node )

		# if we get here, no decorators means this is a normal RC'd class object
		return self._parse_ClassDef_RCClass( node, qualname )

	def _decorator_name( self, decorator: ast.expr ) -> str|None:
		if isinstance( decorator, ast.Name ):
			return decorator.id
		if isinstance( decorator, ast.Call ) and isinstance( decorator.func, ast.Name ):
			return decorator.func.id
		return None

	def _parse_type_params( self, type_params: list[ast.type_param], owner: RCClass|CStruct|CUnion|TaggedUnion|Function ) -> None:
		if not type_params:
			return
		owner.type_params = []
		for type_param in type_params:
			if not isinstance( type_param, ast.TypeVar ):
				self.fail( f'unsupported {type_param=} in {owner.qualname}', type_param )
			if type_param.bound is not None:
				self.fail( f'TypeVar(bound=not None) not supported in {owner.qualname}', type_param )
			if type_param.default_value is not None:
				self.fail( f'TypeVar(default_value=not None) not supported in {owner.qualname}', type_param )
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
		# we don't want to process attributes or functions yet because we haven't finished collecting type information yet.
		# one bad nested class doesn't stop the rest of this class's shallow
		# scan, same as import_code's top-level statement loop
		with self.scope_context( class_obj ):
			unprocessed: list[ast.AST] = []
			for node in body:
				if isinstance( node, ast.ClassDef ):
					try:
						self.visit( node )
					except CompileError:
						continue
				else:
					unprocessed.append( node )
			return unprocessed

	def _make_class_resolver( self, class_obj: ClassLike, body: list[ast.stmt], module: Module ) -> Callable[[],None]:
		def body_fn() -> None:
			with self.module_context( module ):
				with self.scope_context( class_obj ):
					# scope_stack[-1] (class_obj) owns this body - see import_code()
					for node in body:
						self.visit( node )
		def resolve() -> None:
			self._resolve_guarded( class_obj, body_fn )
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
		if node.bases:
			self.fail( f'@enum {qualname} cannot have a base classes ({node.bases!r})', node )
		if node.keywords:
			self.fail( f'@enum {qualname} cannot have keywords ({node.keywords!r})', node )

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
		if node.bases:
			self.fail( f'@cstruct {qualname} cannot have a base classes ({node.bases!r})', node )
		if node.keywords:
			self.fail( f'@cstruct {qualname} cannot have keywords ({node.keywords!r})', node )

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
		if node.bases:
			self.fail( f'@cunion {qualname} cannot have a base classes ({node.bases!r})', node )
		if node.keywords:
			self.fail( f'@cunion {qualname} cannot have keywords ({node.keywords!r})', node )

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
		if node.bases:
			self.fail( f'@union {qualname} cannot have a base classes ({node.bases!r})', node )
		if node.keywords:
			self.fail( f'@union {qualname} cannot have keywords ({node.keywords!r})', node )

		scope = self.scope_stack[-1]
		scope.add_name( class_obj.stem, class_obj )

		self._parse_type_params( node.type_params, class_obj )

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
		if len( node.bases ) > 1:
			self.fail(
				f'multiple inheritance not supported: class {qualname}({", ".join( ast.unparse(b) for b in node.bases )})',
				node,
			)
		if node.keywords:
			self.fail( f'class {qualname} cannot have keywords ({node.keywords!r})', node )

		scope = self.scope_stack[-1]
		scope.add_name( class_obj.stem, class_obj )

		if node.bases:
			# resolved eagerly, in the enclosing scope, exactly like Python
			# itself requires the base to already exist when this statement runs
			base = self.visit( node.bases[0] )
			if not isinstance( base, RCClass ):
				self.fail( f'{qualname} cannot subclass {base.qualname} (only plain classes support inheritance)', node )
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
		self.fail( 'async functions not supported', node )

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

	def _target_value_matches( self, node: ast.expr, active_value: object ) -> bool:
		if isinstance( node, ast.Constant ):
			return node.value == active_value
		if isinstance( node, ast.UnaryOp ) and isinstance( node.op, ast.Not ):
			return not self._target_value_matches( node.operand, active_value )
		if isinstance( node, ast.Tuple ):
			return any( self._target_value_matches( elt, active_value ) for elt in node.elts )
		self.fail( f'unsupported compiler.target(...) value: {ast.unparse(node)}', node )

	def _parse_extern_decorator( self, decorator: ast.expr, node: ast.FunctionDef, qualname: str ) -> tuple[str,str]:
		# @extern('lib', 'symbol') - a foreign call signature declaration.
		# 'lib' is the .lib/.so name to link against, except the literal
		# 'c' which means the platform C runtime rather than a real file on
		# disk - that distinction is a future emitter/linker's job to act
		# on, not this parse step's
		if not isinstance( decorator, ast.Call ) or len( decorator.args ) != 2 or decorator.keywords:
			self.fail( f'@extern(lib, symbol) requires exactly 2 positional arguments: {ast.unparse(decorator)}', node )
		lib_arg, symbol_arg = decorator.args
		if not ( isinstance( lib_arg, ast.Constant ) and isinstance( lib_arg.value, str )):
			self.fail( f'@extern(...) lib name must be a string literal: {ast.unparse(decorator)}', node )
		if not ( isinstance( symbol_arg, ast.Constant ) and isinstance( symbol_arg.value, str )):
			self.fail( f'@extern(...) symbol name must be a string literal: {ast.unparse(decorator)}', node )
		return lib_arg.value, symbol_arg.value

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
		extern_lib: str|None = None
		extern_symbol: str|None = None
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
				case 'extern':
					extern_lib, extern_symbol = self._parse_extern_decorator( decorator, node, qualname )
				case _:
					self.fail( f'unsupported function decorator @{decname or ast.unparse(decorator)} on {qualname}', node )

		if extern_lib is not None and not self._is_stub_body( node.body ):
			self.fail( f'@extern function {qualname} must have a stub body (...) - it declares a foreign call signature, not a real implementation', node )

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
			extern_lib = extern_lib,
			extern_symbol = extern_symbol,
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
		if is_overload or isinstance( existing, ( Overload, Function )):
			if isinstance( existing, Overload ):
				group = existing
			else:
				if existing is not None and not isinstance( existing, Function ):
					self.fail( f'{qualname} redefines {existing!r} as an overload group', node )
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
				if isinstance( existing, Function ):
					# existing was itself a plain (non-@overload) def, parsed
					# before any sibling gave this name a reason to become a
					# group - fold it in as this group's first implementation
					# rather than losing it to add_name's overwrite below.
					# Its resolver closure was built back when group was still
					# None (parsing is a single forward pass, so existing.resolve
					# is guaranteed not to have run yet) - rebuild it now that
					# group actually exists, so existing's own resolve() also
					# runs _check_overload_ambiguity against its new siblings,
					# the same as every other plain implementation does.
					group.implementations.append( existing )
					if class_obj is not None:
						class_obj.methods.remove( existing )
					existing.resolve = self._make_function_resolver( existing, module, class_obj, group )
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
		if len( candidates ) != 1:
			self.fail_loc(
				f'{stub.qualname}: ambiguous overload binding - matches {[c.qualname for c in candidates]}'
				if candidates else
				f'{stub.qualname}: no implementation covers this @overload signature',
				stub.file, stub.line,
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
			if _is_covered_by( fn, earlier ):
				self.fail_loc(
					f'{fn.qualname} (line {fn.line}) is shadowed by {earlier.qualname} (line {earlier.line}) - '
					f'unreachable, every type it declares is already handled by the earlier overload',
					fn.file, fn.line,
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
			if _overlaps( fn, other ):
				self.fail_loc( f'{fn.qualname} and {other.qualname} are ambiguous - their parameter types overlap', fn.file, fn.line )

	def _make_function_resolver( self, fn: Function, module: Module, class_obj: ClassLike|None, group: Overload|None = None ) -> Callable[[],None]:
		def body() -> None:
			# Phase 2 of @compiler.target support (COMPILER-TARGET.md): fold
			# compile-time-constant expressions/if/while before lowering.py
			# ever walks this body - done once, here, rather than on every
			# lowering attempt
			fn.node.body = compile_time_transformer.transform_function_body( fn.node.body, self.active_target )
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
							if arg.annotation is None:
								self.fail( f'{fn.qualname} parameter {arg.arg!r} has no type annotation', arg )
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
			# this closure (see _make_value_resolver for the same pattern).
			# _resolve_guarded (see resolve() below) clears it again
			# afterward too, harmlessly - that's just the safety net for the
			# case this never got here at all (an error above this point)
			fn.resolve = None
			if group is not None:
				if fn in group.stubs:
					self._bind_overload_stub( fn, group )
					self._check_overload_shadowing( fn, group )
				elif fn.is_overload:
					self._check_overload_shadowing( fn, group )
				else:
					self._check_overload_ambiguity( fn, group )
		def resolve() -> None:
			self._resolve_guarded( fn, body )
		return resolve
