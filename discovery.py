# stdlib imports:
import ast
from contextlib import contextmanager, nullcontext
import itertools
import platform
from pathlib import Path
from typing import Any, Callable, Generator, NoReturn

# local imports
import compile_time_transformer
from errors import CompileError, ErrorCollector, RedundantCompilationError
from mpy_types import (
	Name, Type, Scalar, TypeVar, Specialization, Variable, Parameter, Move, Copy, CallableType, ClosureType, TupleType, FixedArrayType, GeneratorType, Function, Overload,
	CEnum, RCClass, CStruct, CUnion, TaggedUnion, ClassLike, CType,
	Module, _is_covered_by, _overlaps, int_stem_range,
)

def _collect_reachable_returns( stmts: list[ast.stmt] ) -> list[ast.Return]:
	''' every ast.Return reachable anywhere within `stmts` (if/for/while/
	with/try/match bodies included), NOT descending into a nested def/
	lambda - a nested def/lambda's own `return` belongs to IT, not to the
	enclosing body (mirrors lowering.py's _reject_free_variables,
	PLAN_LAMBDA.md). Shared by Discovery._is_inline_eligible_body
	(PLAN_INLINE.md) and Discovery._is_eager_return_inferable_body
	(PLAN_RETURN_INFERENCE.md) - both need exactly this walk, just apply a
	different condition to the result. '''
	returns: list[ast.Return] = []
	class _ReturnCollector( ast.NodeVisitor ):
		def visit_FunctionDef( self, fd: ast.FunctionDef ) -> None:
			pass
		def visit_AsyncFunctionDef( self, fd: ast.AsyncFunctionDef ) -> None:
			pass
		def visit_Lambda( self, lam: ast.Lambda ) -> None:
			pass
		def visit_Return( self, ret: ast.Return ) -> None:
			returns.append( ret )
	collector = _ReturnCollector()
	for stmt in stmts:
		collector.visit( stmt )
	return returns

def _find_inline_body_reserved_name_reassignment( stmts: list[ast.stmt], reserved_names: set[str] ) -> ast.Name|None:
	''' PLAN_INLINE.md multi-statement generalization: a Store-context
	reference to `self` or a declared parameter name anywhere within
	`stmts`, not descending into a nested def/lambda. When such a binding
	was passed into the splice via _lower_inline_call's own zero-overhead
	fast path (reuse the caller's own bare Variable directly, no copy - the
	common case for a bare-name receiver/argument), reassigning it inside
	the spliced body would silently mutate the CALLER's own variable, not
	a private copy - a real aliasing bug, not just an unsupported shape,
	if left unguarded. Returns the first offending Name node, or None. '''
	if not reserved_names:
		return None
	found: list[ast.Name] = []
	class _ReassignmentFinder( ast.NodeVisitor ):
		def visit_FunctionDef( self, fd: ast.FunctionDef ) -> None:
			pass
		def visit_AsyncFunctionDef( self, fd: ast.AsyncFunctionDef ) -> None:
			pass
		def visit_Lambda( self, lam: ast.Lambda ) -> None:
			pass
		def visit_Name( self, node: ast.Name ) -> None:
			if isinstance( node.ctx, ast.Store ) and node.id in reserved_names:
				found.append( node )
	finder = _ReassignmentFinder()
	for stmt in stmts:
		finder.visit( stmt )
	return found[0] if found else None

def is_stub_body( body: list[ast.stmt] ) -> bool:
	''' a bodyless `...`-only declaration - @overload's own stub convention,
	reused elsewhere for "this signature has no real implementation yet"
	(e.g. an @interface CStruct's own unfulfilled @virtual slot - see
	PLAN_SUBCLASSING_VTABLES_COM.md). Module-level (not just Discovery's own
	_is_stub_body method, which now just forwards here) so other modules
	(lowering.py, emitter_c.py) can reuse it without needing a Discovery
	instance - same "shared pure helper" convention as mpy_types.py's own
	_is_covered_by/_overlaps. '''
	return (
		len( body ) == 1
		and isinstance( body[0], ast.Expr )
		and isinstance( body[0].value, ast.Constant )
		and body[0].value.value is Ellipsis
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


def _folds_into_package( stem: str ) -> bool:
	''' true if a module with this stem contributes no namespace level of its
	own - its top-level names take the enclosing package's qualname directly
	(builtins.list, not builtins.__init__.list). The single source of truth
	for that rule: import_code() computes qualnames from it and
	_check_qualname_collisions() decides from it whether a collision is worth
	explaining as a consequence of folding.

	Only ever consulted for a module that actually has an enclosing package -
	see import_code, which requires a non-empty scope before applying this.
	That guard is load-bearing, not incidental: the program entry point is
	compiled as __main__.py with scope=None, and folding it would strip the
	prefix off every top-level name in the user's own program.

	A leading '__' marks a module as package-private, and a package-private
	module is an implementation detail of its package rather than a namespace
	users name: `from .__list import list` in builtins/__init__.py should
	publish builtins.list, not builtins.__list.list. __init__ is the
	degenerate case of the same rule rather than a separate one. '''
	return stem.startswith( '__' )


# every ast.stmt kind Discovery's own module-body/class-body scan loops
# (import_code, _make_class_resolver's body_fn) are prepared to hand to
# self.visit() - anything else must be rejected BEFORE that call, not left
# to fall through to ast.NodeVisitor's own default generic_visit. Without
# this, an unsupported statement containing a Store-context ast.Name (a
# bare `for` loop, `x += 1`) crashes with an uncaught internal
# AssertionError deep inside visit_Name ("Load is the only context an
# expression-position Name can have") the moment generic_visit blindly
# recurses into it - confirmed via a real repro (a bare `for i in
# range(3): pass` at module level), not a theoretical concern. A statement
# kind IS allowed to reach generic_visit directly only when doing so is
# provably a no-op no matter what (ast.Pass has no fields at all to
# recurse into) - every other kind either has its own explicit visit_X
# handler here (which may itself still reject, e.g. visit_AsyncFunctionDef
# - that's a clean, intentional CompileError, not this guard's concern) or
# gets rejected right here.
_SUPPORTED_BODY_STATEMENTS: frozenset[type] = frozenset({
	ast.Expr, ast.AnnAssign, ast.Assign, ast.ClassDef, ast.FunctionDef,
	ast.AsyncFunctionDef, ast.Import, ast.ImportFrom, ast.Pass,
})


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
			# resolved, not bare Path('.') - a module found through this entry
			# would otherwise carry a relative .file while every other module
			# (including one passed an already-.resolve()'d entry filename)
			# carries an absolute one, so plain Path equality (e.g.
			# _check_qualname_collisions' same-file exemption, type_resolver's
			# _find_module_for) silently treats the same file on disk as two
			# different ones
			self.paths.append( Path( '.' ).resolve() )
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
		self.required_headers: set[str] = set()

		self.module_stack: list[Module] = []
		self.scope_stack: list[Module|ClassLike|Function] = []

		# every module-level qualname claimed so far, mapped to the Name that
		# claimed it - see _check_qualname_collisions() for what this is
		# defending against and why the check can't live at the definition
		# sites themselves
		self._claimed_qualnames: dict[str,Name] = {}

		# dedup caches for compound types built from other types on the fly
		# (anonymous unions, generic specializations, move[T] wrappers) - never
		# looked up by qualname from outside, only reused when the exact same
		# combination is seen again
		self._unions: dict[str,TaggedUnion] = {}
		self._specializations: dict[str,Specialization] = {}
		self._moves: dict[str,Move] = {}
		self._copies: dict[str,Copy] = {}
		self._callables: dict[str,CallableType] = {}
		self._closures: dict[str,ClosureType] = {}
		self._tuples: dict[str,TupleType] = {}
		self._fixed_arrays: dict[str,FixedArrayType] = {}

		# lazily detected the first time a has_library(...) check (decorator
		# or compiler.has_library(...) expression - see _matches_has_library/
		# compile_time_transformer.py's own _ConstFolder) actually needs one -
		# detect_cc() does real shutil.which()/subprocess work (slower still
		# for MSVC's vswhere auto-detection), so this is cached on the
		# instance rather than re-run per check. _cc_detected distinguishes
		# "not yet probed" from "probed, found nothing" (self._cc is None
		# either way)
		self._cc: 'linker_c.CcTool | None' = None
		self._cc_detected = False

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

	def _check_qualname_collisions( self, module: Module ) -> None:
		''' a module's qualname is normally its own private prefix, so two
		files can't produce the same qualname for anything. Two cases break
		that: __init__.py takes the package's own qualname, and (see
		import_code) so does any package-private `__foo.py`. Their top-level
		names therefore land directly in the package namespace, where a name
		defined by two different files in the same package mangles to one C
		symbol - a duplicate-definition error from the C compiler, pointing at
		mangled output rather than at either source line.

		Runs once per module, after its body scan, rather than at the
		individual definition sites: _get_qualname() is also reached from
		class bodies, type params and parameter lists, several of them from
		inside lazy resolve() closures, so a check there would need to
		untangle re-entrancy and @overload members legitimately sharing one
		qualname. At this point module.names has already collapsed each
		@overload group into a single Overload entry (see _parse_FunctionDef),
		so what's left is exactly the module-level namespace, and any
		collision found here is necessarily between two different files.

		Records the error and continues (never fail()) - by the time this
		runs, import_code's own per-statement recovery boundary is behind us,
		and raising would leave the half-scanned module cached in
		self.modules for every later importer to trip over. '''
		if module.file is None:
			return # synthesized/test module with no source of its own - nothing to point at
		for name_obj in module.names.values():
			# an imported name is the very same Name object the defining
			# module created (visit_ImportFrom re-binds it rather than copying
			# it), so its .file still points at that module. Re-binding an
			# import into another namespace isn't a claim on the qualname
			if name_obj.file != module.file:
				continue
			claimed = self._claimed_qualnames.setdefault( name_obj.qualname, name_obj )
			# only a claim from a DIFFERENT file is a real collision. Same file
			# means the same source was imported twice into one Discovery
			# (import_code with package=None isn't memoized in self.modules, and
			# several tests re-import one fixture path per assertion) - that
			# rebuilds every Name, so the second pass legitimately re-claims what
			# the first one did. An intra-module duplicate can't reach here at
			# all: module.names holds one entry per stem, already collapsed into
			# a single Overload where that's what the duplicate meant
			if claimed.file == name_obj.file:
				continue
			note = ''
			if _folds_into_package( module.stem ) or ( claimed.file is not None and _folds_into_package( claimed.file.stem )):
				note = (
					f'\n\tnote: a module whose name begins with \'__\' is package-private - its top-level '
					f'names go directly into package {module.qualname!r}, so they collide with names of the '
					f'same stem defined by any sibling private module or by the package\'s own __init__.py'
				)
			self.errors.error(
				f'{name_obj.qualname!r} is already defined'
				f'\n\tfirst defined at {claimed.file}:{claimed.line}'
				f'{note}',
				name_obj.file, name_obj.line,
			)

	def _check_supported_statement( self, node: ast.stmt ) -> None:
		''' called before self.visit(node) at every module-body/class-body
		scan site - see _SUPPORTED_BODY_STATEMENTS' own comment for why
		this can't just be left to generic_visit's default recursion. '''
		if type( node ) not in _SUPPORTED_BODY_STATEMENTS:
			self.fail( f'unsupported statement here: {ast.unparse( node )}', node )

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
			target.broken = True
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
		# explicit encoding='utf-8' - open()'s own default is the OS locale
		# encoding, which on Windows is the system ANSI codepage (e.g.
		# CP1252), not UTF-8. Every non-ASCII source file (a bare string
		# literal like 'straße', or `# -*- comment -*-` text) read on
		# Windows without this would get silently decoded wrong here, then
		# re-encoded as (corrupted) UTF-8 by emitter_c.py's own string
		# literal emission - a real, no-op-on-Linux/macOS Windows-only bug
		with filename.open( 'r', encoding = 'utf-8' ) as f:
			code = f.read()
		return self.import_code( code, filename, scope, package = package )

	def import_code( self, code: str, filename: Path, scope: str|None = None, package: str|None = None ) -> Module:
		stem = filename.stem if filename else ''
		# NB `package` (the parameter) is this module's own dotted path, the
		# key it gets registered under in self.modules ('builtins.__list') -
		# NOT the package it lives in. That one is `scope` ('builtins'), which
		# import_name computes by lopping the last component off the module
		# path, and which is what Module.package below wants
		#
		# a folding module (__init__.py, or any package-private __foo.py)
		# defines the package's own namespace, not a sub-module - see
		# _folds_into_package for the rule and for why it's gated on there
		# actually being an enclosing package
		if scope and _folds_into_package( stem ):
			qualname = scope
		else:
			qualname = f'{scope}.{stem}' if scope else stem
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
			package = scope or '',
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
			# fold compile-time-constant if/else/match at the top level before
			# self.visit() sees any statement - this means a module-level
			# `if compiler.target.os == 'windows': ...` collapses to just the
			# matching branch, same as inside a function body (see
			# compile_time_transformer.transform_stmt_list). one bad top-level
			# statement doesn't stop the rest of the module from being scanned -
			# see _resolve_guarded for the same idea applied to individual
			# symbols' .resolve()
			tree.body = compile_time_transformer.transform_stmt_list( tree.body, self.active_target, self._detect_cc )
			for node in tree.body:
				try:
					self._check_supported_statement( node )
					self.visit( node )
				except CompileError:
					continue

		self._check_qualname_collisions( module )
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
			# __metalpy_wideint/__metalpy_wideuint (i128/u128's own real C
			# type) fall back to plain 64-bit under MSVC, which has no native
			# 128-bit integer type - has_i128 tracks whether the active
			# target's real C compiler actually supports the full 128 bits,
			# same target-width-dependent pattern as sizeof_bits above for
			# isize/usize. Defaults True so callers that build Discovery
			# without going through mpy.py's CLI (most tests) keep today's
			# behavior unless they opt in.
			sizeof_i128 = 16 if self.active_target.get( 'has_i128', True ) else 8

			for name, sizeof in [
				( 'isize', sizeof_bits ), ( 'usize', sizeof_bits ),
				( 'i8', 1 ), ( 'u8', 1 ),
				( 'i16', 2 ), ( 'u16', 2 ),
				( 'i32', 4 ), ( 'u32', 4 ),
				( 'i64', 8 ), ( 'u64', 8 ),
				( 'i128', sizeof_i128 ), ( 'u128', sizeof_i128 ),
			]:
				intrinsics[name] = Scalar(
					stem = name,
					qualname = f'intrinsics.{name}',
					file = None,
					line = None,
					sizeof = sizeof,
				)
			# the two floating-point intrinsics live in the same tier as the
			# fixed-width integers above (not lib/-defined classes) - IEEE 754
			# single/double precision. `float`/`double` are NOT distinct types
			# from f32/f64: they're two spellings of the same type, so the
			# alias entries below point at the very same Scalar object (identity
			# comparisons everywhere then treat `float` and `f32` as one type -
			# a value typed `float` satisfies an `f32` parameter, and vice versa)
			for name, sizeof in [ ( 'f32', 4 ), ( 'f64', 8 ) ]:
				intrinsics[name] = Scalar(
					stem = name,
					qualname = f'intrinsics.{name}',
					file = None,
					line = None,
					sizeof = sizeof,
				)
			intrinsics['float'] = intrinsics['f32']
			intrinsics['double'] = intrinsics['f64']
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
				# 1, NOT 0: a NoneType-typed VALUE (e.g. a generic V=None
				# slot, distinct from a `-> None` return, which is a real
				# C void with no storage at all) genuinely occupies one
				# real byte in generated C - emitter_c.py's c_type() maps
				# it to MetalpyNone (`typedef unsigned char MetalpyNone`),
				# not void. sizeof=0 here used to silently disagree with
				# that: any sys.alloc[u8](compiler.sizeof(V))-then-write
				# call site (e.g. UnsafeDict._store_value/_store_key) would
				# allocate a 0-byte buffer and then write MetalpyNone's one
				# real byte into it - a genuine heap buffer overflow.
				sizeof = 1,
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
		if found.broken:
			# the real error was already recorded once, at the point this
			# name's own creation/resolution failed - see Name.broken
			raise RedundantCompilationError()
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

	def visit_Expr( self, node: ast.Expr ) -> None:
		# module-level (or class-body) compiler directives like
		# `compiler.require_header('pthread.h')` — recognized textually
		# (same pattern as _is_compiler_target_call), not by actually
		# resolving the `compiler` module
		if (
			isinstance( node.value, ast.Call )
			and isinstance( node.value.func, ast.Attribute )
			and isinstance( node.value.func.value, ast.Name )
			and node.value.func.value.id == 'compiler'
			and node.value.func.attr == 'require_header'
		):
			call = node.value
			if len( call.args ) != 1 or call.keywords:
				self.fail( f'compiler.require_header(...) takes exactly one argument: {ast.unparse(node)}', node )
			arg = call.args[0]
			if not ( isinstance( arg, ast.Constant ) and isinstance( arg.value, str )):
				self.fail( f'compiler.require_header(...) argument must be a string literal: {ast.unparse(node)}', node )
			self.required_headers.add( arg.value )
			return
		if isinstance( node.value, ast.Constant ):
			# a bare literal (a module/class docstring, or a `...` stub
			# placeholder) has no side effect either way it's read - silently
			# ignoring it matches ordinary Python, where evaluating a lone
			# literal statement is a no-op
			return
		# neither a recognized directive nor an inert literal: this compiler
		# never executes a module/class body as code (main() is the only
		# real entry point - see ARCHITECTURE.md), so a bare expression
		# statement here - `print(...)`, `some_call()`, `x.y` - can never
		# run. This used to fall through silently (same as generic_visit),
		# so a call like this would compile clean and then vanish from the
		# generated program with zero diagnostic. Reject it instead, same as
		# every other statement kind this scan already rejects for being
		# unreachable/meaningless here (AugAssign, loops, ...)
		self.fail( f'unsupported statement here: {ast.unparse(node)}', node )

	def visit_Name( self, node: ast.Name ) -> Name:
		# node.resolved_type - the same compiler-synthesized-code escape
		# hatch lowering.py's own _lower_compiler_cast/_try_resolve_
		# namespace already use (see their own comments) - lets a
		# synthesized ANNOTATION reference a concrete Type object directly,
		# bypassing ordinary by-name scope resolution entirely. Needed for
		# a monomorphized generic class specifically: its own .stem is
		# still the ABSTRACT template's bare name (e.g. 'Box', not
		# 'Box[i32]') - an ordinary find_name(node.id) lookup there
		# resolves to the WRONG (abstract) class, not this concrete
		# specialization, which has no real source-level spelling at all
		resolved = getattr( node, 'resolved_type', None )
		if resolved is not None:
			return resolved
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
		#
		# file/line MUST stay None here (not "whichever module is currently
		# active", unlike _get_or_create_closure_type's identical-looking
		# case just below) - emitter_c.py's mangle_type() uses "TaggedUnion
		# with file is None" as ITS OWN signal to recognize a synthesized
		# anonymous union (vs. a real user `@union class Foo:`, which always
		# has a genuine file) and mangle it via the special $__u$$... scheme
		# instead of plain mangle_qualname() (which can't handle this
		# object's own qualname containing a literal '|', not a legal C
		# identifier character) - giving the union itself a real file would
		# silently break that detection. See union_storage.py's
		# _build_member_constructor instead for where a REAL file is
		# actually needed (Lowering._find_module_for, once something
		# schedules one of this union's synthesized member constructors,
		# not just reads its tag/data fields directly).
		#
		# operands are flattened here (any operand that is ITSELF an
		# anonymous union - t.file is None, never a real user `@union class`
		# - contributes its own leaves instead of itself) and deduped by
		# qualname before sorting. Needed for e.g. Result[T,E].unwrap_or's
		# own `T|None` return annotation: ordinary AST-level parsing already
		# flattens a literal `T|None` into the two operands [T, NoneType]
		# before either is resolved (this class's own visit_BinOp/
		# _flatten_union above), but that flattening can't see through a
		# TypeVar - monomorphize.py's substitute_type_params substitutes T
		# with a concrete type AFTER that AST-level flattening already ran,
		# so when T is itself bound to an Optional (e.g. i32|None), the
		# substituted operand list becomes [i32|None, NoneType] - one
		# already-a-union operand plus a second, redundant NoneType. Without
		# flattening here, that nested union was kept as a single opaque
		# member whose OWN qualname already contains '|', producing a
		# doubled "NoneType" in the outer key (e.g.
		# "intrinsics.NoneType|intrinsics.NoneType|intrinsics.i32") instead
		# of collapsing to the correct, flat "intrinsics.NoneType|
		# intrinsics.i32" - confirmed via a real compile of
		# Result[i32|None,str].unwrap_or(), which failed with exactly that
		# doubled-NoneType mismatch against the (correctly flat) Ok-payload
		# type before this fix.
		flattened: list[Type] = []
		for operand in operands:
			if isinstance( operand, TaggedUnion ) and operand.file is None:
				flattened.extend( attr.type for attr in operand.attributes )
			else:
				flattened.append( operand )
		deduped: dict[str,Type] = {}
		for operand in flattened:
			deduped.setdefault( operand.qualname, operand )
		ordered = sorted( deduped.values(), key = lambda t: t.qualname )
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

	def visit_Subscript( self, node: ast.Subscript ) -> Specialization|Move|Copy|CallableType|TupleType|FixedArrayType|GeneratorType:
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

		# Volatile[T] is compiler syntax too, but UNLIKE move/copy above it's
		# deliberately not modeled as a wrapper Type: a Volatile[T] value
		# must keep behaving as an ordinary T everywhere (arithmetic,
		# comparisons, overload matching) - only its C declaration differs -
		# so it resolves transparently to T itself here (every caller of
		# discovery.visit() sees a plain T, with nothing further to unwrap).
		# The few sites that need to know a local was declared Volatile[T]
		# (currently just lowering.py's _stmt_AnnAssign) peek at the raw
		# annotation AST node themselves, rather than this method threading
		# a side-channel flag back through its Type-only return type.
		if isinstance( node.value, ast.Name ) and node.value.id == 'Volatile':
			if isinstance( node.slice, ast.Tuple ):
				self.fail( f'Volatile[...] takes exactly one type argument: {ast.unparse(node)}', node )
			return self.visit( node.slice )

		# Callable[[Arg1,Arg2,...], Ret] - also compiler syntax (see
		# PLAN_CALLABLE.md), recognized the same textual way as move/copy
		# above rather than resolved as an ordinary generic base: its own
		# shape (an arg-type LIST nested inside the subscript, not a bare
		# type argument) doesn't fit the ordinary type_params path at all
		if isinstance( node.value, ast.Name ) and node.value.id == 'Callable':
			shape_ok = (
				isinstance( node.slice, ast.Tuple )
				and len( node.slice.elts ) == 2
				and isinstance( node.slice.elts[0], ast.List )
			)
			if not shape_ok:
				self.fail( f"Callable[...] must look like Callable[[ArgType, ...], RetType]: {ast.unparse(node)}", node )
			arg_nodes, ret_node = node.slice.elts
			arg_types = [ self.visit( arg_node ) for arg_node in arg_nodes.elts ]
			return_type = self.visit( ret_node )
			return self._get_or_create_callable_type( arg_types, return_type )

		# Closure[[Arg1,Arg2,...], Ret] - a bound-method VALUE (`worker.run`
		# used as a value - see mpy_types.ClosureType's own docstring),
		# same textual recognition/shape as Callable[...] just above, not a
		# generalization of it - deliberately additive, keeps Callable[...]
		# /Ptr[Callable[...]]'s own existing machinery (dict[K,V]/RawDict's
		# callback erasure) untouched
		if isinstance( node.value, ast.Name ) and node.value.id == 'Closure':
			shape_ok = (
				isinstance( node.slice, ast.Tuple )
				and len( node.slice.elts ) == 2
				and isinstance( node.slice.elts[0], ast.List )
			)
			if not shape_ok:
				self.fail( f"Closure[...] must look like Closure[[ArgType, ...], RetType]: {ast.unparse(node)}", node )
			arg_nodes, ret_node = node.slice.elts
			arg_types = [ self.visit( arg_node ) for arg_node in arg_nodes.elts ]
			return_type = self.visit( ret_node )
			return self._get_or_create_closure_type( arg_types, return_type )

		# tuple[T0, T1, ..., Tn] (n >= 1, arity >= 2) - heterogeneous,
		# fixed-arity value groups (see PLAN_TUPLE.md). Recognized textually
		# here, the same way move/copy/Callable/Closure already are above,
		# rather than resolved as an ordinary generic base: there's no real
		# class with type_params to subscript against - tuple is variadic
		# arity AND heterogeneous, so every distinct element-type list needs
		# its own backing layout, synthesized lazily by tuple_storage.
		# TupleStorage.get() the first time this exact TupleType is actually
		# touched (constructed, or read via a constant index), not here -
		# see _get_or_create_tuple_type's own comment. Lowercase 'tuple'
		# (not 'Tuple') deliberately matches list[T]/dict[K,V]'s own
		# established casing - from a user's perspective this reads as an
		# ordinary builtin, not compiler magic the way Callable/Closure are.
		if isinstance( node.value, ast.Name ) and node.value.id == 'tuple':
			arity_ok = isinstance( node.slice, ast.Tuple ) and len( node.slice.elts ) >= 2
			if not arity_ok:
				# arity 0/1 (bare `tuple[T]`, or no elements at all) is a
				# real Python ast.Tuple parsing ambiguity (a 1-tuple LITERAL
				# needs a trailing comma to disambiguate from a plain
				# parenthesized expression) - deferred rather than guessed
				# at, see PLAN_TUPLE.md's own "Deferred" list
				self.fail( f'tuple[...] needs at least 2 type arguments: {ast.unparse(node)}', node )
			elem_types = [ self.visit( elt ) for elt in node.slice.elts ]
			return self._get_or_create_tuple_type( elem_types )

		# Iterator[T] - PLAN_GENERATORS.md. Recognized textually, same
		# posture as move/copy/Callable/Closure/tuple above - there's no
		# real generic class with type_params to subscript against (a
		# generator function's own backing representation doesn't exist
		# until lowering.py actually finds a `yield` in the function body
		# this annotates). Deliberately NOT interned (see GeneratorType's
		# own docstring) - a fresh instance every occurrence.
		if isinstance( node.value, ast.Name ) and node.value.id == 'Iterator':
			if isinstance( node.slice, ast.Tuple ):
				self.fail( f'Iterator[...] takes exactly one type argument: {ast.unparse(node)}', node )
			elem_type = self.visit( node.slice )
			return GeneratorType(
				stem = f'Iterator[{elem_type.qualname}]',
				qualname = f'Iterator[{elem_type.qualname}]',
				file = elem_type.file, line = elem_type.line,
				elem_type = elem_type,
			)

		# Generator[T,E] - PLAN_GENERATORS.md Phase 4 (roadmap Phase 4), the
		# FALLIBLE sibling of Iterator[T] above - same textual recognition,
		# just two type args instead of one, carried as GeneratorType's own
		# error_type (None for Iterator[T] means infallible). __next__'s
		# return type becomes Result[elem_type|None, error_type] instead of
		# plain elem_type|None once ensure_generator_synthesized sees this
		if isinstance( node.value, ast.Name ) and node.value.id == 'Generator':
			if not isinstance( node.slice, ast.Tuple ) or len( node.slice.elts ) != 2:
				self.fail( f'Generator[...] takes exactly two type arguments (element, error): {ast.unparse(node)}', node )
			elem_type = self.visit( node.slice.elts[0] )
			error_type = self.visit( node.slice.elts[1] )
			return GeneratorType(
				stem = f'Generator[{elem_type.qualname},{error_type.qualname}]',
				qualname = f'Generator[{elem_type.qualname},{error_type.qualname}]',
				file = elem_type.file, line = elem_type.line,
				elem_type = elem_type, error_type = error_type,
			)

		base = self.visit( node.value )
		type_params = getattr( base, 'type_params', None )
		if not type_params:
			# ElemType[N] where N is a bare positive integer constant, and
			# ElemType isn't itself generic - SYNTAX.md's "Fixed-Size Inline
			# Array (inside @struct): u16[32], u8[8]", not a generic type
			# argument (a genuine generic subscript's own slice is always a
			# TYPE expression, an ast.Name/Attribute/Subscript/BinOp, never a
			# bare int Constant - so this can never misfire against a real
			# generic instantiation; every actual one already returned above
			# via the type_params-truthy path this branch is the `else` of).
			# See FixedArrayType's own docstring for why this is recognized
			# here (right where a bad subscript would otherwise unconditionally
			# fail) but restricted to @cstruct/@cunion FIELD position only -
			# enforced by the two call sites that matter (_make_annotation_
			# resolver for class/module-level AnnAssign, and _parse_function's
			# parameter/return-type resolution), not here (this method has no
			# notion of "which position is this annotation in").
			if ( isinstance( base, Type ) and isinstance( node.slice, ast.Constant )
					and isinstance( node.slice.value, int ) and not isinstance( node.slice.value, bool ) ):
				count = node.slice.value
				if count <= 0:
					self.fail( f'fixed-size array count must be a positive integer, got {count}: {ast.unparse(node)}', node )
				return self._get_or_create_fixed_array( base, count )
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

	def _get_or_create_callable_type( self, arg_types: list[Type], return_type: Type ) -> CallableType:
		key = f'Callable[[{",".join( a.qualname for a in arg_types )}],{return_type.qualname}]'
		if fn_type := self._callables.get( key ):
			return fn_type
		fn_type = CallableType(
			stem = key,
			qualname = key,
			file = return_type.file,
			line = return_type.line,
			arg_types = arg_types,
			return_type = return_type,
		)
		self._callables[key] = fn_type
		return fn_type

	def _get_or_create_tuple_type( self, elem_types: list[Type] ) -> TupleType:
		# key mirrors _get_or_create_specialization's own qualname convention
		# (f'{base.qualname}[{args}]') - see PLAN_TUPLE.md and tuple_storage.
		# py's own identical qualname choice for the backing RCClass this
		# interns to. backing is populated lazily by tuple_storage.
		# TupleStorage.get() the first time this exact TupleType is actually
		# touched (constructed, or read via a constant index) - not here,
		# same "type resolves without needing a real runtime representation
		# yet" split CallableType/ClosureType above already use.
		key = f'tuple[{",".join( t.qualname for t in elem_types )}]'
		if tt := self._tuples.get( key ):
			return tt
		tt = TupleType(
			stem = key,
			qualname = key,
			file = None,
			line = None,
			elem_types = elem_types,
		)
		self._tuples[key] = tt
		return tt

	def _get_or_create_fixed_array( self, elem_type: Type, count: int ) -> FixedArrayType:
		# key mirrors _get_or_create_tuple_type's own qualname convention -
		# see FixedArrayType's own docstring for why this is a distinct kind
		# from an ordinary generic Specialization
		key = f'{elem_type.qualname}[{count}]'
		if fa := self._fixed_arrays.get( key ):
			return fa
		fa = FixedArrayType(
			stem = key,
			qualname = key,
			file = None,
			line = None,
			elem_type = elem_type,
			count = count,
		)
		self._fixed_arrays[key] = fa
		return fa

	def _get_or_create_closure_type( self, arg_types: list[Type], return_type: Type ) -> ClosureType:
		key = f'Closure[[{",".join( a.qualname for a in arg_types )}],{return_type.qualname}]'
		if cls := self._closures.get( key ):
			return cls
		# NOT return_type.file: return_type is very often a scalar
		# intrinsic (Closure[[],None] - i.e. return_type is NoneType) with
		# no file of its own, which would leave __del__/$$__destructor__
		# unable to find an owning module later (_find_module_for).
		# module_stack[-1] is "whichever module is currently active" - a
		# real, valid module in both callers (visit_Subscript, parsing a
		# real Closure[...] annotation; lowering.py's own construction,
		# still inside module_context(...) for whatever function is being
		# lowered) - same pattern _make_value_resolver already uses
		file = self.module_stack[-1].file if self.module_stack else None
		line = self.module_stack[-1].line if self.module_stack else None
		cls = ClosureType(
			stem = key, qualname = key, file = file, line = line,
			arg_types = arg_types, return_type = return_type,
		)
		# lazy, exactly like an ordinary RCClass's own .resolve convention -
		# nothing about fn/self/__del__ needs to be known before something
		# actually asks for cls.attributes/.names (a Closure[...] type is
		# never itself further subscripted, so unlike type_params there's
		# no eager-resolution requirement here)
		def resolve() -> None:
			ptr_cls = self.get_intrinsics()['Ptr']
			ptr_none_type = self._get_or_create_specialization( ptr_cls, [ self.get_none_type() ] )
			fn_field = Variable( stem = 'fn', qualname = f'{key}.fn', file = cls.file, line = cls.line, type = ptr_none_type )
			self_field = Variable( stem = 'self', qualname = f'{key}.self', file = cls.file, line = cls.line, type = ptr_none_type )
			cls.attributes = [ fn_field, self_field ]
			cls.add_name( 'fn', fn_field )
			cls.add_name( 'self', self_field )
			del_fn = self._build_closure_destructor( cls )
			cls.methods = [ del_fn ]
			cls.add_name( '__del__', del_fn )
			cls.resolve = None
		cls.resolve = resolve
		self._closures[key] = cls
		return cls

	def _build_closure_destructor( self, cls: ClosureType ) -> Function:
		# releases the captured, type-erased receiver: self.self is
		# Ptr[None] by now (the field just built in _get_or_create_closure_
		# type's own resolve() above), so this goes through compiler.
		# decref_dynamic (reads the destructor off the receiver's own
		# header - see emitter_c.py's ObjectHeader.destructor, Phase 2a),
		# not compiler.decref (which needs a statically RC-typed operand).
		# An ORDINARY method (cls set, is_static False), not a bare
		# function like _synthesize_rcclass_destructor's own $$__destructor
		# __ below it - self is synthesized automatically by lowering.py's
		# own lower_function for any Function shaped this way, same as any
		# hand-written __del__ - nothing extra needed here for it
		line = cls.line or 1
		self_self_expr = ast.Attribute(
			value = ast.Name( id = 'self', ctx = ast.Load(), lineno = line, col_offset = 0 ),
			attr = 'self', ctx = ast.Load(), lineno = line, col_offset = 0,
		)
		call = ast.Call(
			func = ast.Attribute(
				value = ast.Name( id = 'compiler', ctx = ast.Load(), lineno = line, col_offset = 0 ),
				attr = 'decref_dynamic', ctx = ast.Load(), lineno = line, col_offset = 0,
			),
			args = [ self_self_expr ], keywords = [],
			lineno = line, col_offset = 0,
		)
		node = ast.FunctionDef(
			name = '__del__',
			args = ast.arguments(
				posonlyargs = [], args = [], vararg = None,
				kwonlyargs = [], kw_defaults = [], kwarg = None, defaults = [],
			),
			body = [ ast.Expr( call, lineno = line, col_offset = 0 ) ],
			decorator_list = [], returns = None, type_params = [],
			lineno = line, col_offset = 0, end_lineno = line, end_col_offset = 0,
		)
		ast.fix_missing_locations( node )
		return Function(
			stem = '__del__', qualname = f'{cls.qualname}.__del__', file = cls.file, line = cls.line,
			cls = cls, node = node, parameters = [], return_type = self.get_none_type(),
			resolve = None,
		)

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
			# counted from the module's package (Python's __package__), not
			# from its qualname: level=1 means "within my own package", so
			# level N climbs N-1 levels above it. Deriving the package by
			# slicing a level off the qualname instead only works for a module
			# that doesn't fold into its package - see Module.package
			package = self.module_stack[-1].package
			strip = node.level - 1
			parts.extend(( package.split( '.' )[:-strip] if strip else package.split( '.' )) if package else [] )
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
			item = mod.get_local_or_raise( alias.name )
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
		init = compile_time_transformer.transform_expr( node.value, self.active_target, self._detect_cc ) if node.value is not None else None
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
					self._reject_bare_interface_value_type( var_obj.type, annotation, var_obj.qualname )
					self._reject_fixed_array_outside_struct_field( var_obj.type, scope, annotation, var_obj.qualname )
		def resolve() -> None:
			self._resolve_guarded( var_obj, body )
		return resolve

	def _reject_fixed_array_outside_struct_field( self, t: 'Type|None', scope: 'Module|ClassLike|Function', node: ast.AST, context: str ) -> None:
		''' a FixedArrayType (`u8[8]`-style fixed-size inline array - see its
		own docstring) is only a legal field annotation on a plain @cstruct/
		@cunion - never a module-level global, an RCClass/@interface field
		(both are always heap-allocated/pointer-accessed, and this fix's own
		zero-fill-only construction path is scoped to the plain-value stack-
		construction compound-literal shape, not the RCClass/@interface `->
		field = value;` per-field ASSIGNMENT shape, which cannot legally
		target a C array at all), or a @union/@enum member (neither has
		plain data fields in this sense). Checked at every annotation-
		resolution site that can produce a FixedArrayType (this one for
		module/class-level AnnAssign; _parse_function's parameter/return-type
		resolution has its own identical call), rather than letting a bad
		usage silently reach emission and produce invalid C. '''
		if not isinstance( t, FixedArrayType ):
			return
		if isinstance( scope, ( CStruct, CUnion )) and not ( isinstance( scope, CStruct ) and scope.is_interface ):
			return
		self.fail(
			f'{context}: a fixed-size inline array type ({t.qualname}) is only allowed as a plain @cstruct/@cunion field, '
			f'not here: {ast.unparse(node)}',
			node,
		)

	def _reject_bare_interface_value_type( self, t: 'Type|None', node: ast.AST, context: str ) -> None:
		''' an @interface CStruct is never a plain value type - self,
		locals, parameters, return types are all Ptr[T]/ConstPtr[T],
		never bare T (see PLAN_SUBCLASSING_VTABLES_COM.md's REVISION).
		Construction never produces a bare value anymore (always Ptr[T] -
		see lowering.py's _lower_allocate_fields), but a bare-typed
		annotation was still syntactically legal and reachable through
		Ptr[T]'s own [0] escape hatch (p[0] genuinely does produce a bare
		CStruct value) - confirmed to CRASH the compiler outright (an
		internal assertion in emitter_c.py's _emit_self_operand, which
		assumes every @interface CStruct method's receiver is already
		Ptr[T]) rather than fail gracefully, once such a bare value was
		passed to a bare-typed parameter and a method called on it.
		Rejected here instead, at every real value-type-annotation site
		(parameters, return types, variables/attributes) - self is exempt
		(never reaches these call sites at all: add_param skips it
		entirely, matching ordinary Python's own implicit self typing;
		its own Ptr[T] type is set up separately, in lowering.py). '''
		if isinstance( t, CStruct ) and t.is_interface:
			self.fail(
				f'{context}: {t.qualname} is an @interface CStruct - it can never be a plain value type, '
				f'only Ptr[{t.stem}]/ConstPtr[{t.stem}]: {ast.unparse(node)}',
				node,
			)

	def _reject_non_c_type_on_extern_signature( self, t: 'Type|None', node: ast.AST, context: str ) -> None:
		''' an @extern function's foreign C symbol has no notion of this
		compiler's own RC-managed objects or synthesized tagged unions - only
		a genuine plain C value crosses that boundary correctly. Allowlisted
		(not denylisted) deliberately: this codebase's own is_rc()/
		is_rc_pointer() ladder (mpy_types.py's own comment on Type.is_rc)
		documents three separate real bugs from exactly the denylist failure
		mode - a new Type kind silently defaulting to "safe" because nothing
		added it to the reject list. A plain C value is one of: Scalar
		(i32/u8/.../bool/NoneType/...), @cstruct/@cunion (CStruct/CUnion),
		a raw foreign C type (CType, e.g. lib/posix/pthread.py's pthread_t,
		already used by value as a real extern parameter), a C enum (CEnum),
		or a function-pointer signature (CallableType, Ptr[Callable[...]]) -
		anything else (TaggedUnion's tag+payload struct has no foreign-ABI
		counterpart at all; RCClass is heap-allocated with this compiler's
		own ObjectHeader prefix, unsafe even though it happens to already be
		pointer-shaped in the generated C; tuple[...]/Iterator[T]/generic
		containers are all RC-backed the same way) is rejected. Ptr[T]/
		ConstPtr[T] are unwrapped recursively first - a pointer to a bad type
		is exactly as unsafe as the bad type itself (e.g. Ptr[Ptr[str]]). '''
		leaf = t
		while isinstance( leaf, Specialization ) and isinstance( leaf.base, Scalar ) and leaf.base.stem in ( 'Ptr', 'ConstPtr' ):
			leaf = leaf.args[0]
		if isinstance( leaf, ( Scalar, CStruct, CUnion, CType, CEnum, CallableType )):
			return
		qualname = getattr( leaf, 'qualname', leaf )
		self.fail(
			f'{context}: {qualname} cannot cross an @extern boundary - only a plain C value type (a scalar, '
			f'@cstruct/@cunion, a raw C type, a C enum, a function pointer, or Ptr[T]/ConstPtr[T] to one of '
			f'those) is allowed here: {ast.unparse(node)}',
			node,
		)

	def visit_Assign( self, node: ast.Assign ) -> Name|None:
		scope = self.scope_stack[-1]
		if isinstance( scope, CEnum ):
			self._register_enum_member( scope, node )
			return None

		# compiler.c_type('name', header='header.h') — declare a C type
		# defined in an external header, registrable as a normal Name
		if self._is_compiler_c_type_call( node.value ):
			if len( node.targets ) != 1 or not isinstance( node.targets[0], ast.Name ):
				self.fail( f'compiler.c_type(...) must be assigned to a single name: {ast.unparse(node)}', node )
			c_name, header = self._parse_compiler_c_type_call( node.value, node )
			target_name = node.targets[0].id
			module = self.module_stack[-1]
			ctype = CType(
				stem = target_name,
				qualname = self._get_qualname( target_name ),
				file = module.file,
				line = node.lineno,
				c_name = c_name,
				required_header = header,
			)
			# also register the header requirement for the emitter
			self.required_headers.add( header )
			scope.add_name( target_name, ctype )
			return ctype

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
		init = compile_time_transformer.transform_expr( node.value, self.active_target, self._detect_cc )
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
		value_expr = compile_time_transformer.transform_expr( node.value, self.active_target, self._detect_cc )
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
		# a member's value must actually fit the enum's own underlying type's
		# real range - catches both an explicit out-of-range value AND an
		# auto-incremented ('_') one that overflows after enough members.
		# Unlike lowering.py's plain-literal/CEnum-construction range checks,
		# there's no bit-reinterpretation exemption to consider here: an enum
		# member's value is ALWAYS a bare literal (a Call/cast expression is
		# already rejected above, "must be '_' or an integer constant")
		if isinstance( cls.value_type, Scalar ):
			lo, hi = int_stem_range( cls.value_type )
			if not ( lo <= value <= hi ):
				self.fail(
					f'{value} is out of range for {cls.qualname} ({lo}..{hi}): {ast.unparse(node)}',
					node,
				)
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
				case 'interface':
					return self._parse_ClassDef_Interface( node, qualname )
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
						self._check_supported_statement( node )
						self.visit( node )
			if isinstance( class_obj, RCClass ) and class_obj.base is not None:
				self._validate_no_attribute_shadowing( class_obj )
		def resolve() -> None:
			self._resolve_guarded( class_obj, body_fn )
		return resolve

	def _validate_no_attribute_shadowing( self, class_obj: RCClass ) -> None:
		''' a subclass cannot redeclare a name (field or method) already
		declared by an ancestor, matching this codebase's "explicit, not
		accidental" posture around name reuse (e.g. @interface not being
		inherited implicitly). The one exception is a legitimate @virtual
		override - both sides declaring the SAME name as @virtual (RCClass-
		subclassing plan Phase 4) - a subclass overriding an inherited
		@virtual method MUST repeat @virtual on its own re-declaration, no
		method is virtual anywhere without @virtual written on that exact
		declaration, matching CStruct's own identical rule. Strict
		signature matching between the two isn't checked here - that needs
		both sides' parameters/return_type already resolved, which isn't
		guaranteed yet at this (discovery/parse-time) point if a subclass
		is parsed before its base's own methods are individually resolved
		(see compiler.py's _validate_interface_vtable, which re-walks this
		same collision at compile-time once resolution order is no longer
		a concern). `__init__` is exempted - a subclass declaring its own
		__init__ is the ordinary, expected constructor-chaining case
		(super().__init__(), Phase 2), not shadowing in the sense this
		check cares about. Runs once class_obj's own body has been fully
		visited (own .names is only complete once body_fn's loop finishes) -
		the ancestor side is walked via chain_lookup starting at class_obj.
		base specifically (not class_obj itself), since this is deliberately
		checking OWN names against ANCESTOR names only, not self-collisions
		(which .names, a plain dict, already can't have). '''
		base = class_obj.base
		assert base is not None
		for own_name, own in class_obj.names.items():
			if own_name == '__init__':
				continue
			ancestor = base.chain_lookup( own_name )
			if ancestor is None:
				continue
			if isinstance( own, Function ) and own.is_virtual and isinstance( ancestor, Function ) and ancestor.is_virtual:
				continue # a legitimate override - see this method's own docstring
			self.fail_loc(
				f'{class_obj.qualname}.{own_name} shadows {ancestor.qualname} - a subclass cannot '
				f'redeclare an inherited attribute or method name',
				class_obj.file, class_obj.line,
			)

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

		try:
			unresolved = self._shallow_class_body_scan( class_obj, node.body )
		except CompileError:
			# scope.add_name already ran above, so class_obj.resolve would
			# otherwise be left at its dataclass default of None here - the
			# exact value that means "already resolved, nothing to do"
			# everywhere else - indistinguishable from a genuinely fine
			# class to any later reference. broken makes that reference
			# raise RedundantCompilationError instead (see Name.broken).
			class_obj.broken = True
			raise

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

		try:
			self._parse_type_params( node.type_params, class_obj )

			unresolved = self._shallow_class_body_scan( class_obj, node.body )
		except CompileError:
			class_obj.broken = True # see _parse_ClassDef_CEnum's own comment
			raise

		class_obj.resolve = self._make_class_resolver( class_obj, unresolved, module )

		return class_obj

	def _parse_ClassDef_Interface( self, node: ast.ClassDef, qualname: str ) -> CStruct:
		# @interface class IFoo: / @interface class FooImpl(IFoo): - a CStruct
		# that participates in vtable dispatch (see
		# PLAN_SUBCLASSING_VTABLES_COM.md). Its own parse method, not a flag on
		# _parse_ClassDef_CStruct, because the base-class rules are genuinely
		# different from plain @cstruct (which forbids bases outright) -
		# mirrors _parse_ClassDef_RCClass's base handling instead.
		module = self.module_stack[-1]
		class_obj = CStruct(
			stem = node.name,
			qualname = qualname,
			file = module.file,
			line = node.lineno,
			is_interface = True,
		)
		if len( node.bases ) > 1:
			self.fail(
				f'multiple inheritance not supported: class {qualname}({", ".join( ast.unparse(b) for b in node.bases )})',
				node,
			)
		if node.keywords:
			self.fail( f'@interface {qualname} cannot have keywords ({node.keywords!r})', node )

		scope = self.scope_stack[-1]
		scope.add_name( class_obj.stem, class_obj )

		try:
			if node.bases:
				# @interface-ness is NOT inherited implicitly - a CStruct
				# subclassing an @interface CStruct must itself be declared
				# @interface too (this method only runs when it was), and its
				# base must itself already be an @interface CStruct, not a plain
				# one. Deliberately conservative - see "Subclassing mechanics" in
				# PLAN_SUBCLASSING_VTABLES_COM.md.
				base = self.visit( node.bases[0] )
				if not ( isinstance( base, CStruct ) and base.is_interface ):
					self.fail( f'{qualname} cannot subclass {base.qualname} (@interface can only subclass another @interface CStruct)', node )
				class_obj.base = base

			self._parse_type_params( node.type_params, class_obj )

			unresolved = self._shallow_class_body_scan( class_obj, node.body )
		except CompileError:
			# base resolution above runs BEFORE class_obj.resolve is ever
			# assigned below - a failure here would otherwise leave .resolve
			# at its dataclass default of None, indistinguishable from
			# "already resolved fine" to anything checking `.resolve is
			# None` (see _parse_ClassDef_CEnum's own comment for the general
			# shape of this landmine)
			class_obj.broken = True
			raise

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

		try:
			self._parse_type_params( node.type_params, class_obj )

			unresolved = self._shallow_class_body_scan( class_obj, node.body )
		except CompileError:
			class_obj.broken = True # see _parse_ClassDef_CEnum's own comment
			raise

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

		try:
			self._parse_type_params( node.type_params, class_obj )

			unresolved = self._shallow_class_body_scan( class_obj, node.body )
		except CompileError:
			class_obj.broken = True # see _parse_ClassDef_CEnum's own comment
			raise

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

		try:
			if node.bases:
				# resolved eagerly, in the enclosing scope, exactly like Python
				# itself requires the base to already exist when this statement runs
				base = self.visit( node.bases[0] )
				if not isinstance( base, RCClass ):
					self.fail( f'{qualname} cannot subclass {base.qualname} (only plain classes support inheritance)', node )
				class_obj.base = base

			self._parse_type_params( node.type_params, class_obj )

			unresolved = self._shallow_class_body_scan( class_obj, node.body )
		except CompileError:
			# base resolution above runs BEFORE class_obj.resolve is ever
			# assigned below - see _parse_ClassDef_Interface's own comment
			# for why that makes a failure here otherwise invisible
			class_obj.broken = True
			raise

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

	def _is_compiler_c_type_call( self, expr: ast.expr ) -> bool:
		return (
			isinstance( expr, ast.Call )
			and isinstance( expr.func, ast.Attribute )
			and expr.func.attr == 'c_type'
			and isinstance( expr.func.value, ast.Name )
			and expr.func.value.id == 'compiler'
		)

	def _parse_compiler_c_type_call( self, call: ast.Call, node: ast.AST ) -> tuple[str,str]:
		if len( call.args ) != 1 or not isinstance( call.args[0], ast.Constant ) or not isinstance( call.args[0].value, str ):
			self.fail( f'compiler.c_type(name, header=...) requires a string literal name as the first argument: {ast.unparse(node)}', node )
		c_name = call.args[0].value
		header: str|None = None
		for kw in call.keywords:
			if kw.arg == 'header':
				if not isinstance( kw.value, ast.Constant ) or not isinstance( kw.value.value, str ):
					self.fail( f'compiler.c_type(...) header= must be a string literal: {ast.unparse(node)}', node )
				header = kw.value.value
			else:
				self.fail( f'compiler.c_type(...) unexpected keyword argument {kw.arg!r}: {ast.unparse(node)}', node )
		if header is None:
			self.fail( f'compiler.c_type(...) requires header="..." keyword argument: {ast.unparse(node)}', node )
		return c_name, header

	def _matches_active_target( self, call: ast.Call ) -> bool:
		for kw in call.keywords:
			if kw.arg == 'has_library':
				# checked before the active_target dict lookup below,
				# deliberately - has_library=(lib, symbol) is structurally
				# a 2-tuple too, and would otherwise be wrongly read as an
				# OR-list of alternatives against active_target['has_library']
				# (which doesn't exist) by _target_value_matches' own
				# ast.Tuple handling, meant for os=('windows','macos')-style
				# axis alternatives, not a function argument pair
				if not self._matches_has_library( kw.value, call ):
					return False
				continue
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

	def _detect_cc( self ) -> 'linker_c.CcTool | None':
		if not self._cc_detected:
			import linker_c
			self._cc = linker_c.detect_cc()
			self._cc_detected = True
		return self._cc

	def _matches_has_library( self, value: ast.expr, call: ast.Call ) -> bool:
		''' @compiler.target(has_library=('icuuc', 'ucasemap_utf8ToUpper'))
		- eagerly probes (via a real compile+link, cached to disk - see
		linker_c.has_symbol()) whether the given symbol resolves when
		linked against the given library, and filters the decorated
		def/class in or out entirely, the same way os=/arch=/etc already
		do. Deliberately eager (unlike compiler.has_library(...) used as
		an expression inside a function body, which only probes if that
		code path is actually reached during lowering - see compile_time_
		transformer.py's own _ConstFolder) - a whole alternate definition,
		selected by decorator, has no "reached during lowering" moment to
		defer to; the choice has to be made right here, during this early
		scan, same as every other @compiler.target(...) axis. '''
		negate = False
		node = value
		if isinstance( node, ast.UnaryOp ) and isinstance( node.op, ast.Not ):
			negate = True
			node = node.operand
		if not (
			isinstance( node, ast.Tuple ) and len( node.elts ) == 2
			and isinstance( node.elts[0], ast.Constant ) and isinstance( node.elts[0].value, str )
			and isinstance( node.elts[1], ast.Constant ) and isinstance( node.elts[1].value, str )
		):
			self.fail( f'@compiler.target(has_library=(lib, symbol)) requires a 2-tuple of string literals: {ast.unparse(call)}', call )
		lib, symbol = node.elts[0].value, node.elts[1].value
		cc = self._detect_cc()
		if cc is None:
			self.fail(
				f'@compiler.target(has_library=({lib!r}, {symbol!r})) needs a C compiler '
				f'(clang, gcc, or MSVC) to probe with - none was found',
				call,
			)
		import linker_c
		available = linker_c.has_symbol( cc, lib, symbol )
		return available != negate

	def _parse_extern_decorator( self, decorator: ast.expr, node: ast.FunctionDef, qualname: str ) -> tuple[str,str,str|None]:
		# @extern('lib', 'symbol') or @extern('lib', 'symbol', header='<name>')
		# 'lib' is the .lib/.so name to link against, except the literal
		# 'c' which means the platform C runtime rather than a real file on
		# disk - that distinction is a future emitter/linker's job to act
		# on, not this parse step's
		if not isinstance( decorator, ast.Call ) or len( decorator.args ) < 2 or len( decorator.args ) > 3:
			self.fail( f'@extern(lib, symbol[, header=...]) requires 2 or 3 positional arguments: {ast.unparse(decorator)}', node )
		lib_arg, symbol_arg = decorator.args[0], decorator.args[1]
		if not ( isinstance( lib_arg, ast.Constant ) and isinstance( lib_arg.value, str )):
			self.fail( f'@extern(...) lib name must be a string literal: {ast.unparse(decorator)}', node )
		if not ( isinstance( symbol_arg, ast.Constant ) and isinstance( symbol_arg.value, str )):
			self.fail( f'@extern(...) symbol name must be a string literal: {ast.unparse(decorator)}', node )
		header: str|None = None
		for kw in decorator.keywords:
			if kw.arg == 'header':
				if not isinstance( kw.value, ast.Constant ) or not isinstance( kw.value.value, str ):
					self.fail( f'@extern(...) header= must be a string literal: {ast.unparse(decorator)}', node )
				header = kw.value.value
			else:
				self.fail( f'@extern(...) unexpected keyword argument {kw.arg!r}: {ast.unparse(decorator)}', node )
		return lib_arg.value, symbol_arg.value, header

	def _parse_function(
		self,
		node: ast.FunctionDef,
		class_obj: ClassLike|None = None,
	) -> Function|Overload|None:
		# NOTE: the name 'main' is special, there can be only one...
		qualname = 'main' if node.name == 'main' else self._get_qualname( node.name )

		if node.name == 'or_return':
			# 'or_return' is reserved, compiler-implemented-only - Result[T,E]
			# .or_return() is recognized purely by AST shape + the receiver's
			# own type (lowering.py's _lower_call, before ordinary call
			# resolution ever runs), never by looking up a real declared
			# method the way is_ok()/is_err()/unwrap()/unwrap_or() genuinely
			# are (those DO have real, callable bodies - only or_return's own
			# "body" was ever just a spec of the intended behavior, expanded
			# directly to OrReturn/OrJump IR instead - see _lower_or_return's
			# own comment). A user-written `def or_return(...)` - on Result
			# itself or on any other class - can never actually run: nothing
			# ever resolves a real call to it, at ANY receiver type, so
			# accepting one silently would just be dead, misleading code.
			# Checked here unconditionally (independent of any decorator,
			# class, or overload grouping) since the name alone is what's
			# reserved, not any particular shape of definition.
			self.fail(
				f"'or_return' is reserved for the compiler's own Result[T,E].or_return() - it can't be defined as a real "
				f'function or method: {qualname}',
				node,
			)

		is_overload = False
		is_static = False
		is_classmethod = False
		is_abstract = False
		is_move = False
		is_private = False
		is_virtual = False
		is_inline = False
		is_property = False
		is_fallible_arithmetic = False
		extern_lib: str|None = None
		extern_symbol: str|None = None
		extern_header: str|None = None
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
				case 'virtual':
					is_virtual = True
				case 'inline':
					is_inline = True
				case 'property':
					is_property = True
				case 'fallible_arithmetic':
					is_fallible_arithmetic = True
				case 'extern':
					extern_lib, extern_symbol, extern_header = self._parse_extern_decorator( decorator, node, qualname )
				case _:
					self.fail( f'unsupported function decorator @{decname or ast.unparse(decorator)} on {qualname}', node )

		if extern_lib is not None and not self._is_stub_body( node.body ):
			self.fail( f'@extern function {qualname} must have a stub body (...) - it declares a foreign call signature, not a real implementation', node )

		if is_abstract:
			# an abstract method IS a virtual slot with no implementation -
			# @abstractmethod alone is sufficient (implies @virtual, doesn't
			# need it repeated) since there's no OTHER coherent meaning for
			# an unimplemented method to have; unlike @interface not being
			# inherited implicitly or a @virtual OVERRIDE having to repeat
			# @virtual (both genuinely ambiguous without being explicit),
			# there's no ambiguity here to guard against. Set before the
			# is_virtual checks below so they apply uniformly whether
			# @virtual was ALSO explicitly written (redundant, still
			# accepted - harmless, not contradictory) or not. RCClass-
			# subclassing plan Phase 5, revised per user feedback.
			is_virtual = True
			if not self._is_stub_body( node.body ):
				# mirrors @extern's own identical stub-body requirement
				# just above - an abstract method's body IS the "must be
				# overridden" declaration (there's nothing to run), same
				# shape @overload stubs already use
				self.fail( f'@abstractmethod {qualname} must have a stub body (...) - it declares a required override, not a real implementation', node )

		if is_virtual and not ( class_obj is not None and class_obj.has_vtable() ):
			# @interface CStructs and ordinary RCClasses both build a real
			# vtable now (RCClass-subclassing plan Phase 4 generalized this
			# from CStruct-only) - CUnion/TaggedUnion/CEnum/a plain, non-
			# @interface CStruct still never do, and @virtual on one of
			# those would be silently meaningless rather than a real error,
			# which is worse
			self.fail( f'@virtual {qualname} is only supported on @interface classes or ordinary classes right now', node )
		if is_virtual and is_overload:
			# combining the two is a real, separate design question (which
			# overload's signature does the vtable slot use? does each
			# overload get its own slot?) that this plan never addressed -
			# reject rather than silently building something half-right
			self.fail( f'@virtual {qualname} cannot also be @overload - not supported', node )
		if is_virtual and ( is_static or is_classmethod ):
			# no receiver to dispatch through - vtable dispatch is
			# meaningless without a self, same reasoning @virtual+@overload
			# above already uses (reject outright rather than silently
			# building something with no coherent meaning)
			self.fail( f'@virtual {qualname} cannot also be @staticmethod/@classmethod - no receiver to dispatch through', node )

		if is_property:
			# read-only getter only for now - no @x.setter (that needs its
			# own exemption from the "already defined" duplicate-name check
			# below, the same way @overload gets one; no datetime/timedelta
			# need is driving that yet). Must be an ordinary instance method
			# with exactly one parameter (self) - no other positional/
			# keyword/*args/**kwargs params, since `obj.attr` (no call
			# parens) never supplies any.
			if class_obj is None:
				self.fail( f'@property {qualname} is only valid on a method, not a free function', node )
			if is_static or is_classmethod:
				self.fail( f'@property {qualname} cannot also be @staticmethod/@classmethod - a property reads through an instance', node )
			if is_overload:
				self.fail( f'@property {qualname} cannot also be @overload - a property has exactly one signature', node )
			all_params = node.args.posonlyargs + node.args.args + node.args.kwonlyargs
			if len( all_params ) != 1 or node.args.vararg is not None or node.args.kwarg is not None:
				self.fail( f'@property {qualname} must take exactly `self` and no other parameters', node )

		if is_fallible_arithmetic and is_abstract:
			# nothing ever actually runs to produce a Result to consume -
			# same "no coherent meaning" reasoning as @inline+@abstractmethod
			# above
			self.fail( f'@fallible_arithmetic {qualname} cannot also be @abstractmethod - no body to produce a Result', node )

		if is_inline:
			# PLAN_INLINE.md - each of these interacts with the real call
			# boundary (@inline removes it entirely, splicing the body at
			# each call site instead) in a way that hasn't been designed:
			# @overload's "which candidate's body?" is the same open
			# question @virtual+@overload already rejects above; @virtual
			# needs a real vtable slot/indirect call, the opposite of
			# splicing; @abstractmethod has no body to splice; @extern's
			# body is a foreign signature stub, not a real body either;
			# @classmethod would need a `cls` substitution this pass
			# doesn't build; @move's whole-function ownership-transfer
			# semantics at a removed call boundary hasn't been reasoned
			# through
			if is_overload:
				self.fail( f'@inline {qualname} cannot also be @overload - not supported', node )
			# is_abstract checked BEFORE is_virtual: @abstractmethod implies
			# is_virtual=True (set earlier above), so checking is_virtual
			# first would misreport an @inline+@abstractmethod combo as
			# "cannot also be @virtual" - a confusing error for someone who
			# never wrote @virtual at all
			if is_abstract:
				self.fail( f'@inline {qualname} cannot also be @abstractmethod - no body to splice', node )
			if is_virtual:
				self.fail( f'@inline {qualname} cannot also be @virtual - not supported', node )
			if extern_lib is not None:
				self.fail( f'@inline {qualname} cannot also be @extern - no real body to splice', node )
			if is_classmethod:
				self.fail( f'@inline {qualname} cannot also be @classmethod - not supported', node )
			if is_move:
				self.fail( f'@inline {qualname} cannot also be @move - not supported', node )
			if not self._is_inline_eligible_body( node.body ):
				self.fail(
					f'@inline {qualname} must have a body ending in exactly one `return <expr>` '
					f'(optionally preceded by a docstring), with every other reachable `return` (anywhere earlier, including '
					f'nested in if/for/while) also carrying a value - not yet supported for anything else',
					node,
				)
			# multi-statement generalization: everything but the final
			# `return <expr>` (already validated above) gets spliced as
			# real statements at each call site - self/parameter
			# reassignment is rejected here because leaving it unguarded
			# would be silently WRONG (aliasing the caller's own argument),
			# not just unsupported - see the helper's own docstring.
			# defer/errdefer WAS rejected here too (its "runs when this
			# function returns" contract had no real boundary to mean
			# anything against before this pass) - no longer needed: the
			# splice now has a well-defined local epilogue of its own (see
			# lowering.py's _splice_multi_statement_inline_body/cfg.py's
			# push_inline_scope), so defer/errdefer is spliced and replayed
			# there exactly like an ordinary function's own
			pre_return_stmts = node.body[:-1]
			reserved_names = { a.arg for a in ( node.args.posonlyargs + node.args.args + node.args.kwonlyargs ) }
			if class_obj is not None and not is_static and not is_classmethod:
				reserved_names.add( 'self' )
			reassigned = _find_inline_body_reserved_name_reassignment( pre_return_stmts, reserved_names )
			if reassigned is not None:
				self.fail(
					f'@inline {qualname}: reassigning self/a parameter ({reassigned.id!r}) before the final return of a '
					f'multi-statement body is not yet supported - it may alias the caller\'s own argument, not a private '
					f'copy; assign it to a new local first: {ast.unparse(reassigned)}',
					reassigned,
				)

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
			is_virtual = is_virtual,
			is_overload = is_overload,
			is_inline = is_inline,
			is_property = is_property,
			is_fallible_arithmetic = is_fallible_arithmetic,
			extern_lib = extern_lib,
			extern_symbol = extern_symbol,
			extern_header = extern_header,
		)
		if extern_header is not None:
			self.required_headers.add( extern_header )
		if node.name == 'main':
			self.main = fn
		self._parse_type_params( node.type_params, fn )

		scope = self.scope_stack[-1]
		# a raw peek, not a "consume this known-good member" lookup (unlike
		# get_local_or_raise's other call sites) - existing here feeds
		# overload-group-formation branching below, which already handles
		# a plain Function vs an Overload vs nothing at all; a BROKEN
		# existing Function is deliberately left to that same branching
		# rather than special-cased, since folding it into a fresh group as
		# a (broken) sibling implementation is the same "own resolve()
		# fails, doesn't taint the group" behavior _resolve_guarded already
		# gives every other overload member
		existing = scope.get_local( fn.stem )

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
					existing.overload_group = group
					if class_obj is not None:
						class_obj.methods.remove( existing )
					existing.resolve = self._make_function_resolver( existing, module, class_obj, group )
			if is_overload and self._is_stub_body( node.body ):
				group.stubs.append( fn )
			else:
				group.implementations.append( fn )
				fn.overload_group = group

		fn.resolve = self._make_function_resolver( fn, module, class_obj, group )

		if group is not None:
			return group

		scope.add_name( fn.stem, fn )
		if class_obj is not None:
			class_obj.methods.append( fn )
		return fn

	def _is_stub_body( self, body: list[ast.stmt] ) -> bool:
		return is_stub_body( body )

	def _is_inline_eligible_body( self, body: list[ast.stmt] ) -> bool:
		''' PLAN_INLINE.md, generalized for early/nested return: @inline
		accepts a body (after stripping an optional leading docstring - an
		ast.Expr wrapping a string ast.Constant, the same shape ast.
		get_docstring recognizes) of arbitrary statements followed by
		exactly one final, TOP-LEVEL `return <expr>` - but now, unlike the
		original multi-statement generalization, OTHER `return <expr>`
		statements are also allowed anywhere earlier, including nested
		inside if/for/while (lowering.py's _splice_multi_statement_inline_
		body/cfg.py's push_inline_scope give each splice its own local
		early-exit target, so an early return no longer needs to be
		rejected outright the way it once did). Every reachable return -
		the trailing one and any earlier ones alike - must still carry a
		value: a bare `return` has no well-defined meaning for an inline
		function's own overall value, so it's rejected the same way the
		trailing one always has been. The original single-`return <expr>`-
		statement shape is the trivial special case of this (stmts ==
		[Return]) and stays accepted unchanged - every currently-accepted
		body stays accepted. lowering.py's _lower_inline_call relies on
		this having already rejected everything else, it doesn't re-check. '''
		stmts = body
		if stmts and isinstance( stmts[0], ast.Expr ) and isinstance( stmts[0].value, ast.Constant ) and isinstance( stmts[0].value.value, str ):
			stmts = stmts[1:]
		if not stmts or not isinstance( stmts[-1], ast.Return ) or stmts[-1].value is None:
			return False
		returns = _collect_reachable_returns( stmts )
		return all( r.value is not None for r in returns )

	def _is_eager_return_inferable_body( self, body: list[ast.stmt] ) -> bool:
		''' return-only generic type-parameter inference (a generic
		function whose return type is a bare type param appearing in no
		parameter, only knowable by actually lowering the body once every
		OTHER type param is bound - see lowering.py's
		_infer_return_only_type_params) is only attempted on a body with
		EXACTLY ONE reachable `return <expr>` ANYWHERE in it (unlike
		@inline's _is_inline_eligible_body just above, this doesn't
		additionally require it be positionally last) - this sidesteps "do
		all return points agree on the same concrete type" entirely, since
		there's only ever one to agree with. Multi-statement bodies with
		locals/branches/loops are fine; multiple RETURN POINTS are not. '''
		returns = _collect_reachable_returns( body )
		return len( returns ) == 1 and returns[0].value is not None

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
			fn.node.body = compile_time_transformer.transform_function_body( fn.node.body, self.active_target, self._detect_cc )
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
							param_type = self.visit( arg.annotation )
							# move[T]/copy[T] is an ownership status on this
							# binding, not a distinct type from T (see Move/
							# Copy's own docstrings, TODO.txt's own "incref/
							# decref" section) - unwrap here, at the one place
							# a Parameter's real .type gets set, so every
							# ordinary consumer downstream (attribute/method
							# lookup, generic inference, assignability) sees
							# plain T like any other binding; the ownership
							# fact itself is recorded on is_move/is_copy
							# instead, consulted only by the two things that
							# actually care about it (the move(x) call-site
							# syntax check, and the CFG's own decref bookkeeping)
							is_move = isinstance( param_type, Move )
							is_copy = isinstance( param_type, Copy )
							if is_move or is_copy:
								param_type = param_type.inner
								if fn.is_inline:
									# @inline splicing binds self/every parameter
									# zero-copy, always treated as borrowed at the
									# splice boundary (lowering.py's
									# _lower_inline_call - "no _cfg_assign/incref
									# here, deliberately... borrowed, no incref at
									# the boundary") - a move[T]/copy[T] param's
									# real ownership-transfer/prologue-incref
									# semantics have never been reasoned through
									# for that boundary (see inline_splice_
									# aliasing_return_incref_bug_fixed.md: even
									# plain borrowed aliasing returns needed a
									# real fix here). Reject outright rather than
									# risk a silent refcount bug - same "no
									# coherent meaning yet" reasoning the
									# @inline+@move whole-function check below
									# already uses
									self.fail(
										f'@inline {fn.qualname} parameter {arg.arg!r} cannot be move[T]/copy[T] - not supported',
										arg,
									)
							self._reject_bare_interface_value_type( param_type, arg, f'{fn.qualname} parameter {arg.arg!r}' )
							self._reject_fixed_array_outside_struct_field( param_type, fn, arg, f'{fn.qualname} parameter {arg.arg!r}' )
							if fn.extern_lib is not None:
								self._reject_non_c_type_on_extern_signature( param_type, arg, f'{fn.qualname} parameter {arg.arg!r}' )
							param = Parameter(
								stem = arg.arg,
								qualname = self._get_qualname( arg.arg ),
								file = fn.file,
								line = fn.line,
								type = param_type,
								default = default,
								is_move = is_move,
								is_copy = is_copy,
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
						if fn.node.returns is not None:
							fn.return_type = self.visit( fn.node.returns )
							self._reject_bare_interface_value_type( fn.return_type, fn.node.returns, f'{fn.qualname} return type' )
							self._reject_fixed_array_outside_struct_field( fn.return_type, fn, fn.node.returns, f'{fn.qualname} return type' )
							if fn.extern_lib is not None:
								self._reject_non_c_type_on_extern_signature( fn.return_type, fn.node.returns, f'{fn.qualname} return type' )
						else:
							fn.return_type = self.get_none_type()
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
				if fn.is_virtual and ( len( group.stubs ) + len( group.implementations )) > 1:
					# a @virtual method must have EXACTLY one signature - not
					# just the already-rejected @virtual+@overload-on-the-
					# SAME-def combo (checked eagerly at parse time, above in
					# this same method) but ALSO metalpy's OTHER overloading
					# mechanism: multiple plain (non-@overload) defs sharing
					# a name with different signatures silently form an
					# Overload group with no @overload in sight. This can
					# only be checked here, at resolve time - group members
					# may still be getting parsed when THIS def's own parse-
					# time checks ran, so the group's final member count
					# isn't settled until every sibling has been parsed
					# (this codebase's own "single forward pass" parsing
					# discipline guarantees that's true by the time ANY
					# member's own resolve() actually runs - see the
					# existing _check_overload_ambiguity/_check_overload_
					# shadowing calls just above, which rely on the exact
					# same guarantee)
					self.fail_loc(
						f'{fn.qualname}: @virtual method {fn.stem!r} must have exactly one signature - found '
						f'{len(group.stubs) + len(group.implementations)} definitions sharing this name '
						f'(whether declared with @overload or not)',
						fn.file, fn.line,
					)
		def resolve() -> None:
			self._resolve_guarded( fn, body )
		return resolve
