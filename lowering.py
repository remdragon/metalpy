# stdlib imports:
import ast
from contextlib import nullcontext
from dataclasses import replace
from typing import Callable

# local imports:
import arithmetic_mode
import cfg
import ir
from discovery import Discovery, is_stub_body
from errors import CompileError
from mpy_types import (
	Name, Type, Variable, Parameter, Function, Overload, ClassLike, Module, CType,
	Specialization, TaggedUnion, CStruct, CUnion, CEnum, TypeVar, ConditionalDispatch, Move, RCClass, Scalar,
	CallableType, ClosureType,
)
import overload_resolution
from type_resolver import TypeResolver
from union_storage import ReceiverDispatch as _ReceiverDispatch

# compile-error "here's what to do instead" text for a Check-mode opcode's
# checked_error (ir.py's BinOp/UnaryOp.checked_error) - keyed by error name
# rather than owned by ir.py itself, since these are lowering-level compiler
# messages, not IR shape
_ALTERNATIVES_BY_ERROR: dict[str,str] = {
	'OverflowError': (
		'wrap this in `with compiler.wrap_arithmetic:`, `with compiler.saturate_arithmetic:`, '
		'or `with compiler.panic_arithmetic(...):` instead'
	),
	# the primary, expected path for division is the same as any other
	# Check-mode op: the enclosing function returns Result[_,
	# ZeroDivisionError] and the Result propagates via OrReturn/OrJump - no
	# panic involved, and this is what happens even inside wrap_arithmetic/
	# saturate_arithmetic (there's no wrapped/saturated variant of division,
	# so those modes don't change division's checked-ness at all). This
	# message only fires when that requirement ISN'T met - panic_arithmetic
	# is the one remaining alternative to changing the return type, not a
	# default
	'ZeroDivisionError': 'wrap this in `with compiler.panic_arithmetic(...):` instead',
}

# ast.BinOp operator -> the dunder method name to dispatch to for a
# non-scalar left operand (str.__add__, etc.). Scalar operands always
# go through arithmetic mode instead.
_BINOP_DUNDER: dict[type,str] = {
	ast.Add: '__add__',
	ast.Sub: '__sub__',
	ast.Mult: '__mul__',
	ast.FloorDiv: '__floordiv__',
	ast.Mod: '__mod__',
}

# ast comparison operator -> the dunder method name to dispatch to for a
# non-scalar left operand (str.__eq__, etc.). Scalar operands go through
# flat ir.Cmp instead.
_COMP_DUNDER: dict[type,str] = {
	ast.Eq: '__eq__',
	ast.NotEq: '__ne__',
	ast.Lt: '__lt__',
	ast.LtE: '__le__',
	ast.Gt: '__gt__',
	ast.GtE: '__ge__',
}

# ast.UnaryOp operator -> the dunder method name to dispatch to for a
# non-scalar operand (int.__neg__, ...). Scalar operands always go through
# arithmetic mode instead. ast.UAdd/ast.Invert are deliberately not mapped -
# no builtin type defines __pos__/__invert__ today.
_UNARYOP_DUNDER: dict[type,str] = {
	ast.USub: '__neg__',
}


class Lowering:
	'''
	turns one Function body (or one global Variable's initializer) at a time
	into an ir.py instruction list. Reuses Discovery's scope-chain machinery
	(find_name/module_context/scope_context) for identifier resolution -
	function bodies were deliberately left unvisited in stage 1 specifically
	so this could be reused here; what's new here is only the value-expression
	semantics stage 1 never needed (constant values, operator-to-opcode
	mapping, temp allocation, instruction emission).

	Whenever anything that might be a dependency is discovered - a Function, a
	class, a type (possibly a Specialization like Result[i32,E]), a Variable,
	even a Module reached mid-namespace-lookup - `schedule` is called on it
	immediately at the point of discovery, unconditionally; there's no
	separate dependency-scanning pass, and no filtering here either.
	`schedule` (Compiler._enqueue) is the single place that judges what's
	actually a compile unit worth queuing, what decomposes into more of
	those, and what to just quietly ignore - see its own docstring.

	Errors report through self.discovery.errors, the same collector stage 1
	uses (self.discovery.fail()/fail_loc()) - see _lower_stmt's caller in
	lower_function for the recovery boundary (one bad statement doesn't stop
	the rest of that function's body from being lowered).

	Arithmetic (+/-/*) defaults to Check mode (AddCheck/SubCheck/MulCheck,
	producing Result[T,OverflowError]) everywhere - there is no unchecked
	default. A Check op is immediately followed by an OrReturn (like
	Result.or_return()'s own semantics: propagate the error, continue with
	the unwrapped value), which requires the enclosing function to actually
	return Result[_,OverflowError] - using plain arithmetic in a function
	that can't propagate that error is a compile error, unless one of the
	arithmetic-mode with-blocks below is used instead. self._arithmetic_mode
	is a stack of ArithmeticMode objects, pushed/popped by _stmt_With:
		ArithmeticWrap       - `with compiler.wrap_arithmetic:` - plain
		                       AddWrap/SubWrap/MulWrap, no Result involved
		ArithmeticSaturate   - `with compiler.saturate_arithmetic:` - plain
		                       AddSaturate/SubSaturate/MulSaturate, likewise
		ArithmeticChecked    - the default (see above) - Check + OrReturn
		ArithmeticPanic      - `with compiler.panic_arithmetic(errmsg):` -
		                       still Check-mode ops, but consumed with
		                       Unwrap(errmsg) instead of OrReturn, so (unlike
		                       the bare default) this does NOT require the
		                       enclosing function to return Result[_,
		                       OverflowError] - Unwrap panics, it never
		                       needs anywhere to propagate to

	defer/errdefer (SYNTAX.md section 3, either `defer(expr)`/`errdefer(expr)`
	as a single statement or `with defer:`/`with errdefer:` for several) move
	their body to the function's shared epilogue - each registration
	(_register_defer_block) pushes a REAL cfg.Epilogue entry (cfg.py's
	push_defer(), flag set) onto the exact same _epilogue_stack RC bindings
	use, interleaved by declaration order with whatever locals surround it.
	Every `return`/checked-arithmetic-error-path (self._cfg.current_epilogue_
	label()) and the function's own fall-off-the-end funnel through
	build_epilogue_ladder(), which replays the whole stack in reverse
	(deepest/most-recently-pushed first) - a flag-guarded entry's own bool
	flag (False until control passes its registration point) gates whether
	it actually replays; errdefer entries are additionally guarded by
	calling .is_err() on the function's own stowed return value (always a
	Result wherever errdefer is legal) - not a separate signal, so it also
	covers a plain `return Result.Err(x)`, not just the implicit OrJump
	path. defer/errdefer are rejected inside a loop (self._loop_depth) or
	nested inside each other (self._in_deferred_body) - see
	_register_defer_block. A defer/errdefer registered inside an if-branch
	must still be reachable from the function's own single shared epilogue
	regardless of which branch (if either) actually armed it - cfg.py's
	restore() special-cases flag-guarded entries to survive scope-exit
	truncation for exactly this reason (see its own comment).
	'''

	def __init__( self, discovery: Discovery, type_resolver: 'TypeResolver' ) -> None:
		self.discovery = discovery
		# TypeResolver (type_resolver.py) owns the reachable-from-main
		# work queue and the shared UnionStorage/Monomorphizer instances -
		# both already depended on nothing but Discovery and a `schedule`
		# callback, so Lowering just borrows the SAME instances rather than
		# building its own (see TypeResolver's own docstring). schedule/
		# _union_storage/_monomorphizer keep their original names here since
		# they're referenced throughout this file - only construction moved
		self._type_resolver = type_resolver
		self.schedule = type_resolver.schedule
		self._union_storage = type_resolver.union_storage
		self._monomorphizer = type_resolver.monomorphizer
		self._tuple_storage = type_resolver.tuple_storage
		self._closure_trampolines: dict[tuple[int,int],Function] = {} # (id(method), id(owner_type)) -> its one synthesized trampoline, see _get_or_create_closure_trampoline
		self._lambda_counter = 0 # -> f'$$lambda_{n}', unique per compile run - see _expr_Lambda (PLAN_LAMBDA.md)

	def lower_function( self, fn: Function ) -> list[ir.Instruction]:
		# per-function lowering state (_instructions, _current_fn, _cfg, etc.
		# - see FunctionLowering's own docstring) lives on a FRESH instance
		# every call, including a nested/reentrant call made mid-way through
		# lowering an enclosing function (see _expr_Lambda's eager lowering
		# path, PLAN_LAMBDA.md) - there is no shared mutable state between
		# an outer and inner call to worry about saving/restoring at all
		return FunctionLowering( self, fn ).run()

	def lower_global( self, var: Variable ) -> list[ir.Instruction]:
		return FunctionLowering( self, None ).run_global( var )

	# --- __init__ construction (RCCLASS ATTRIBUTE LIFETIME.md) -----------------

	def _init_fallibility( self, fn: Function ) -> bool:
		''' __init__ must return None (non-fallible) or Result[None,E]
		(fallible - per SYNTAX.md, Foo(...) then returns Result[Foo,E]) -
		anything else is a compile error, checked as soon as __init__
		itself is lowered, independent of whether/where it's ever
		constructed from. '''
		none_type = self.discovery.get_none_type()
		if fn.return_type is none_type:
			return False
		shape = self._type_resolver._result_shape( fn.return_type )
		ok = shape is not None and shape[0] is none_type
		if not ok:
			self.discovery.fail(
				f'{fn.qualname} must return None or Result[None,_], got '
				f'{fn.return_type.qualname if fn.return_type else None}',
				fn.node,
			)
		return True

	def _is_result_err_call( self, node: ast.expr | None ) -> str|None:
		# `return Result.Err(...)` - textually recognized, same spirit as
		# _defer_kind_of_call/_defer_kind_of_with - deliberately not
		# attempting deeper type-level inference (see _stmt_Return's own
		# comment on why anything else defaults to "requires completeness").
		# Returns the discriminant ('Result.Err') rather than a bare bool,
		# matching every other textual recognizer in this file
		if (
			isinstance( node, ast.Call )
			and isinstance( node.func, ast.Attribute )
			and node.func.attr == 'Err'
			and isinstance( node.func.value, ast.Name )
			and node.func.value.id == 'Result'
		):
			return 'Result.Err'
		return None

	# --- module lookup ------------------------------------------------------

	def _find_module_for( self, unit: Function|Variable|ClassLike ) -> Module:
		# Function/Variable/ClassLike.file is always set to their owning
		# module's .file (see discovery.py's _parse_function/visit_AnnAssign/
		# visit_Assign/_parse_ClassDef_*) - none of them retain a direct
		# back-reference to the Module itself
		for module in self.discovery.modules.values():
			if module.file == unit.file:
				return module
		self.discovery.fail_loc( f'no module found owning {unit.qualname} (file={unit.file})', unit.file, unit.line )

	def _is_aliasing_expr( self, node: ast.expr, operand_type: Type|None = None ) -> bool:
		# does lowering `node` hand back a reference to a value that
		# already exists independently (needing its own Incref if it's
		# stored into a new binding), vs a genuinely fresh value (Allocate,
		# or a Call - always a fresh owned handoff, whether the callee's
		# own body built it via Allocate or received it as an alias
		# itself, since a well-behaved callee already accounts for that on
		# its own side)? Name/Attribute reads are the only currently-
		# supported expression forms that alias existing state -
		# BinOp/BoolOp/Compare/Constant/UnaryOp never produce RC values at
		# all, and Call is always fresh from the caller's perspective.
		# ast.Subscript is deliberately NOT included here even though it
		# looks like a read: _expr_Subscript's dominant path (a real
		# __getitem__) is a Call underneath (fresh), and its other path
		# (raw ir.GetItem, genuinely aliasing a container element) isn't
		# reachable by any real code yet - no indexable container exists
		# yet (list[T]/dict[K,V] are still first-draft/WIP per TODO.txt) -
		# revisit this once one does.
		#
		# ast.Attribute is genuinely ambiguous now, the same way Subscript
		# already is above: `obj.field` reads an existing field (aliasing),
		# but `worker.run` (a bound-method reference) CONSTRUCTS a fresh
		# closure (an Allocate underneath, via _lower_bound_method_closure)
		# - same "fresh owned handoff" shape as a Call, not a read.
		# Re-inspecting node itself can't tell these apart without
		# re-resolving (and risking double-evaluating) the receiver, so
		# callers instead pass the operand they ALREADY lowered from node -
		# its type alone settles it cheaply and exactly, no re-lowering.
		# Scoped to ast.Attribute specifically, NOT every ClosureType
		# operand - `d = c` (a bare Name reading an EXISTING closure local)
		# is an ordinary aliasing read like any other RC-typed Name, and
		# must still incref (confirmed by a real regression: `d = c` then
		# calling both silently underreferenced the shared closure)
		if isinstance( node, ast.Attribute ) and isinstance( operand_type, ClosureType ):
			return False
		return isinstance( node, ( ast.Name, ast.Attribute ))

	def _is_compiler_attr( self, node: ast.expr ) -> str|None:
		# textual recognition, same as discovery.py's _is_compiler_target_call -
		# `compiler` is a special pseudo-module (Discovery.compiler_module),
		# not something with a real .names dict to resolve this through
		if (
			isinstance( node, ast.Attribute )
			and isinstance( node.value, ast.Name )
			and node.value.id == 'compiler'
		):
			return node.attr
		return None

	def _is_compiler_call( self, node: ast.expr ) -> str|None:
		if (
			isinstance( node, ast.Call )
			and isinstance( node.func, ast.Attribute )
			and isinstance( node.func.value, ast.Name )
			and node.func.value.id == 'compiler'
		):
			return node.func.attr
		else:
			return None

	def _atomic_pointee_type( self, ptr_type: Type|None, node: ast.AST ) -> Type:
		# shared by every compiler.atomic_*(ptr, ...) intrinsic - ptr must be
		# Ptr[T] (not ConstPtr[T]: every op here either writes through the
		# pointer, or (atomic_load) is only meaningful on a location another
		# thread can concurrently write - a genuinely immutable location
		# needs no atomic access at all) with T a plain scalar. RC types are
		# rejected deliberately: atomically swapping an RC pointer without
		# incref/decref bookkeeping is exactly the lock-free-RC rabbit hole
		# this intentionally stays out of (see the plan's own Context).
		if not ( isinstance( ptr_type, Specialization ) and isinstance( ptr_type.base, Scalar ) and ptr_type.base.stem == 'Ptr' ):
			self.discovery.fail( f'compiler.atomic_*(...) argument must be Ptr[T]: {ast.unparse(node)}', node )
		pointee = ptr_type.args[0]
		if not isinstance( pointee, Scalar ):
			self.discovery.fail(
				f'compiler.atomic_*(...) argument must point to a plain scalar, not '
				f'{pointee.qualname if pointee else "?"}: {ast.unparse(node)}',
				node,
			)
		return pointee


	def _eval_cexpr( self, expr: str, header: str, node: ast.AST ) -> int:
		import hashlib
		import os
		import tempfile
		from pathlib import Path
		# cache key derived from (expr, header) — deterministic, so the
		# same expression always hits the same cached value regardless
		# of which compilation or project it appears in
		key = hashlib.sha256( f'{expr}\0{header}'.encode() ).hexdigest()[:16]
		cache_dir = Path( tempfile.gettempdir() ) / 'metalpy' / 'cexpr'
		cache_dir.mkdir( parents = True, exist_ok = True )
		cache_file = cache_dir / key
		if cache_file.is_file():
			return int( cache_file.read_text().strip() )

		# no cached value — compile and run a tiny C program
		import linker_c
		cc = linker_c.detect_cc()
		if cc is None:
			self.discovery.fail(
				f'compiler.cexpr({expr!r}, {header!r}) needs a C compiler '
				f'(clang, gcc, or MSVC) — none was found',
				node,
			)
		c_src = f'#include <{header}>\n#include <stdio.h>\nint main(void) {{ printf("%zu\\n", (size_t)({expr})); return 0; }}\n'
		with tempfile.TemporaryDirectory() as tmp:
			src_path = Path( tmp ) / 'cexpr.c'
			obj_path = Path( tmp ) / 'cexpr.o'
			exe_path = Path( tmp ) / 'cexpr'
			src_path.write_text( c_src, encoding = 'utf-8' )
			cc_result = cc.compile( src_path, obj_path )
			if cc_result.returncode != 0:
				self.discovery.fail(
					f'compiler.cexpr({expr!r}, {header!r}): failed to compile '
					f'the C snippet:\n{cc_result.stdout}',
					node,
				)
			link_result = cc.link( exe_path, [ obj_path ] )
			if link_result.returncode != 0:
				self.discovery.fail(
					f'compiler.cexpr({expr!r}, {header!r}): failed to link '
					f'the C snippet:\n{link_result.stdout}',
					node,
				)
			import subprocess
			run_result = subprocess.run( [ str( exe_path ) ], capture_output = True, text = True )
			if run_result.returncode != 0:
				self.discovery.fail(
					f'compiler.cexpr({expr!r}, {header!r}): C program exited '
					f'{run_result.returncode}',
					node,
				)
			value = int( run_result.stdout.strip() )
		cache_file.write_text( str( value ), encoding = 'utf-8' )
		return value

	_UNICODE_DATA_URL = 'https://www.unicode.org/Public/UCD/latest/ucd/UnicodeData.txt'

	def _fetch_unicode_data_txt( self, node: ast.AST ) -> bytes:
		''' downloads (or reads a locally-cached/overridden copy of)
		UnicodeData.txt - see PLAN_CASE_FOLDING.md's own "Data acquisition"
		section: deliberately NOT version-pinned (the caller's own choice -
		fetches whatever the UCD's own 'latest' alias currently points at),
		cached indefinitely once fetched (same no-expiry philosophy
		compiler.cexpr()'s own cache already uses - see _eval_cexpr), with
		METALPY_UNICODE_DATA_DIR (mirroring METALPY_CC's existing override
		convention) letting an offline/CI build point at a local copy
		instead of ever reaching the network. '''
		import os
		import tempfile
		from pathlib import Path

		override_dir = os.environ.get( 'METALPY_UNICODE_DATA_DIR', '' ).strip()
		if override_dir:
			local_path = Path( override_dir ) / 'UnicodeData.txt'
			if not local_path.is_file():
				self.discovery.fail(
					f'METALPY_UNICODE_DATA_DIR={override_dir!r} is set but {local_path} does not exist',
					node,
				)
			return local_path.read_bytes()

		cache_dir = Path( tempfile.gettempdir() ) / 'metalpy' / 'case_folding'
		cache_dir.mkdir( parents = True, exist_ok = True )
		cache_file = cache_dir / 'UnicodeData.txt'
		if cache_file.is_file():
			return cache_file.read_bytes()

		import urllib.error
		import urllib.request
		request = urllib.request.Request( self._UNICODE_DATA_URL, headers = { 'User-Agent': 'metalpy-compiler' } )
		try:
			with urllib.request.urlopen( request, timeout = 30 ) as response:
				data = response.read()
		except ( urllib.error.URLError, OSError ) as e:
			self.discovery.fail(
				f'compiler.fetch_unicode_table(...): failed to download {self._UNICODE_DATA_URL} ({e}) - '
				f'set METALPY_UNICODE_DATA_DIR to a local directory containing UnicodeData.txt to avoid the network entirely',
				node,
			)
		cache_file.write_bytes( data )
		return data

	def _build_unicode_simple_table( self, data: bytes, which: str, node: ast.AST ) -> bytes:
		''' parses UnicodeData.txt's own semicolon-delimited fields (field 0
		= codepoint, field 12 = simple uppercase mapping, field 13 = simple
		lowercase mapping - both hex, empty when the codepoint has no
		simple mapping in that direction) into a binary-searchable table:
		sorted-by-codepoint pairs of (codepoint: u32 LE, mapped: u32 LE),
		8 bytes per entry, no header/count prefix (the caller already
		knows the byte length via bytes.byte_len(), used as entry_count*8
		directly - see CaseFolding.upper()/lower()'s own comment). Simple-
		mapping ONLY (~1500 entries either direction) - SpecialCasing.txt's
		one-to-many (ß -> SS) and context-sensitive (Greek final sigma)
		entries are a documented v1 scope cut, same posture as the OS-
		backed path this is meant to improve on (see PLAN_CASE_FOLDING.md). '''
		field_index = 12 if which == 'upper' else 13
		entries: list[tuple[int,int]] = []
		for line in data.decode( 'utf-8' ).splitlines():
			if not line or line.startswith( '#' ):
				continue
			fields = line.split( ';' )
			if len( fields ) <= field_index or not fields[field_index]:
				continue
			try:
				codepoint = int( fields[0], 16 )
				mapped = int( fields[field_index], 16 )
			except ValueError:
				continue
			entries.append( ( codepoint, mapped ) )
		entries.sort()
		if not entries:
			self.discovery.fail(
				f"compiler.fetch_unicode_table({which!r}): parsed UnicodeData.txt but found zero entries - "
				f"the file is probably not what was expected (wrong format, truncated download, ...)",
				node,
			)
		table = bytearray()
		for codepoint, mapped in entries:
			table += codepoint.to_bytes( 4, 'little' )
			table += mapped.to_bytes( 4, 'little' )
		return bytes( table )

	def _lower_compiler_fetch_unicode_table( self, node: ast.Call ) -> ir.Operand:
		''' compiler.fetch_unicode_table('upper' | 'lower') - downloads/
		caches UnicodeData.txt (see _fetch_unicode_data_txt) and folds to
		an ir.Const(type=bytes, value=<the encoded table>) - the SAME
		program-wide static-embedding path _emit_string_literals already
		gives any str/bytes-valued ir.Const (see emitter_c.py), so this
		needs no new emitter support at all: the table shows up as an
		ordinary static const byte array, deduplicated the same way two
		identical string literals already are. Only actually reached (and
		only actually pays the download/parse cost) for a program that
		references compiler.fetch_unicode_table(...) itself - nothing in
		builtins does, only case_folding.py - see CaseFolding's own
		comment in lib/builtins/__init__.py. '''
		if len( node.args ) != 1 or node.keywords or not isinstance( node.args[0], ast.Constant ) or node.args[0].value not in ( 'upper', 'lower' ):
			self.discovery.fail(
				f"compiler.fetch_unicode_table(...) takes exactly one literal argument, 'upper' or 'lower': {ast.unparse(node)}",
				node,
			)
		which = node.args[0].value
		data = self._fetch_unicode_data_txt( node )
		table = self._build_unicode_simple_table( data, which, node )
		bytes_cls = self.discovery.find_name( 'bytes', node )
		return ir.Const( type = bytes_cls, value = table )

	def _lower_compiler_cexpr( self, node: ast.Call, expected_type: Type|None ) -> ir.Operand:
		# compiler.cexpr('C expression', 'header.h', [target_type])
		# compiles a tiny C program that printf()'s the expression,
		# runs it, and folds the captured stdout to an ir.Const.
		# The result is cached under $TMPDIR/metalpy/cexpr/.
		if len( node.args ) < 2 or len( node.args ) > 3 or node.keywords:
			self.discovery.fail(
				f'compiler.cexpr(expr, header[, type]) takes 2-3 '
				f'positional arguments: {ast.unparse(node)}', node )
		expr_arg, header_arg = node.args[0], node.args[1]
		if not ( isinstance( expr_arg, ast.Constant ) and isinstance( expr_arg.value, str )):
			self.discovery.fail( f'compiler.cexpr(...) expr must be a string literal: {ast.unparse(node)}', node )
		if not ( isinstance( header_arg, ast.Constant ) and isinstance( header_arg.value, str )):
			self.discovery.fail( f'compiler.cexpr(...) header must be a string literal: {ast.unparse(node)}', node )
		expr, header = expr_arg.value, header_arg.value
		if len( node.args ) == 3:
			user_type = self._try_resolve_namespace( node.args[2] )
			if not isinstance( user_type, Scalar ):
				self.discovery.fail(
					f'compiler.cexpr(...) third argument must be a scalar '
					f'type: {ast.unparse(node)}', node )
			result_type = user_type
		else:
			usize_cls = self.discovery.get_intrinsics()['usize']
			result_type = expected_type or usize_cls
		value = self._eval_cexpr( expr, header, node )
		return ir.Const( type = result_type, value = value )

	# --- defer/errdefer ----------------------------------------------------------

	def _defer_kind_of_with( self, node: ast.expr ) -> str|None:
		# `with defer:` / `with errdefer:` - bare names, unlike the
		# compiler.-prefixed arithmetic-mode context managers
		if isinstance( node, ast.Name ) and node.id in ( 'defer', 'errdefer' ):
			return node.id
		return None

	def _defer_kind_of_call( self, node: ast.expr ) -> str|None:
		# `defer( expr )` / `errdefer( expr )` - the single-statement call form
		if isinstance( node, ast.Call ) and isinstance( node.func, ast.Name ) and node.func.id in ( 'defer', 'errdefer' ):
			return node.func.id
		return None

	def _body_may_fall_off_the_end( self, body: list[ast.stmt] ) -> bool:
		# a simple, deliberately narrow check (not full terminator analysis -
		# same "future work" scope cut as _stmt_If's own true_terminates/
		# false_terminates detection): true whenever the LAST top-level
		# statement isn't itself a `return` (an empty body, or one ending in
		# a plain statement/if/loop/etc, could all still fall through to the
		# function's own closing brace). A body that's actually unreachable
		# past this point (both branches of a trailing if already return,
		# a trailing `while True:` with no break, ...) is a false positive -
		# harmless, since the resulting fall-off unwind+Return is then
		# genuinely dead code, never executed
		return not body or not isinstance( body[-1], ast.Return )

	def _synth_name( self, stem: str, node: ast.AST ) -> ast.Name:
		n = ast.Name( id = stem, ctx = ast.Load() )
		ast.copy_location( n, node )
		return n

	def _find_method( self, owner_type: Type|None, name: str ) -> Function|None:
		# a non-failing probe, unlike _attr_lookup_callable - "this type has
		# no such method" is a normal, expected outcome for callers here
		# (for loop iterability checks, __getitem__'s raw-GetItem fallback),
		# not a real error to report
		owner_type = self._ensure_resolved( owner_type )
		if isinstance( owner_type, ( CStruct, RCClass )):
			found = owner_type.chain_lookup( name )
		else:
			names = getattr( owner_type, 'names', None )
			found = names.get( name ) if isinstance( names, dict ) else None
		return found if isinstance( found, Function ) else None

	def _is_range_call( self, node: ast.expr ) -> str|None:
		# range(...) is textually recognized as compiler sugar, same as
		# compiler.wrap_arithmetic/defer/etc. - there's no real range()
		# function (TODO.txt: a real range()/Iterator needs the generator
		# state-machine transform, which doesn't exist yet). This covers
		# exactly the 1-2 arg counting-loop shape real lib/ code already
		# uses (str.concat's `for i in range(count):`). Returns the
		# discriminant ('range') rather than a bare bool, matching every
		# other textual recognizer in this file
		if isinstance( node, ast.Call ) and isinstance( node.func, ast.Name ) and node.func.id == 'range':
			return 'range'
		return None

	_FOR_LOOP_ALTERNATIVES = 'call .__len__()/.__getitem__() directly and consume their Result yourself instead'

	def _function_ref_operand( self, fn: Function ) -> ir.FunctionRef:
		# fn.parameters/fn.return_type must already be resolved (not None) -
		# shared by _lower_function_ref (a bare reference to an EXISTING
		# Function) and _expr_Lambda/_stmt_FunctionDef (a reference to a
		# freshly synthesized one, PLAN_LAMBDA.md) - both end up needing
		# the exact same Ptr[Callable[...]]-typed FunctionRef operand once
		# they have a resolved Function in hand
		for p in fn.parameters or []:
			self.schedule( p.type )
		self.schedule( fn.return_type )
		fn_type = self.discovery._get_or_create_callable_type( [ p.type for p in fn.parameters or [] ], fn.return_type )
		ptr_cls = self.discovery.get_intrinsics()['Ptr']
		ptr_type = self.discovery._get_or_create_specialization( ptr_cls, [ fn_type ] )
		return ir.FunctionRef( type = ptr_type, fn = fn )

	def _reject_generic_enclosing_scope( self, enclosing: Function, node: ast.AST, what: str ) -> None:
		# shared by _stmt_FunctionDef and _expr_Lambda - a nested def/
		# lambda inside a generic function or a generic class's own method
		# is rejected outright for now (see PLAN_LAMBDA.md's own "deferred"
		# list - sidesteps "which monomorphization does this belong to"
		# entirely). type_params alone isn't enough: a MONOMORPHIZED
		# generic function's own copy has type_params reset to None (it's
		# concrete now, not generic anymore - see Monomorphizer.
		# monomorphized_function) - the qualname's own '[...]' suffix is
		# the one signal that survives substitution (discovery.py's
		# _get_or_create_specialization always spells a Specialization's
		# qualname as f'{base.qualname}[{args}]')
		if enclosing.type_params or '[' in enclosing.qualname or isinstance( enclosing.cls, Specialization ):
			self.discovery.fail(
				f"{what} are not supported inside a generic function or a generic class's own method yet: {ast.unparse(node)}",
				node,
			)

	def _get_or_create_closure_trampoline( self, method: Function, owner_type: Type ) -> Function:
		# one small, memoized, module-level function per (method, owner
		# class): (Ptr[None] erased_self, *rest_args) -> Ret, casting
		# erased_self back to owner_type (compile-time known here, even
		# though the closure's own declared type has erased it - see
		# ClosureType's own docstring) and calling the real method on it.
		# Built the same way type_resolver.py's _synthesize_rcclass_
		# destructor builds $$__destructor__ - a hand-built ast.FunctionDef,
		# scheduled, then lowered completely normally
		key = ( id( method ), id( owner_type ))
		if trampoline := self._closure_trampolines.get( key ):
			return trampoline

		self._ensure_resolved( method )
		if method.parameters is None:
			self.discovery.fail( f'{method.qualname} could not be resolved (see earlier error)', method.node )
		self.schedule( method.return_type )
		for p in method.parameters:
			self.schedule( p.type )

		ptr_cls = self.discovery.get_intrinsics()['Ptr']
		none_type = self.discovery.get_none_type()
		ptr_none_type = self.discovery._get_or_create_specialization( ptr_cls, [ none_type ] )

		line = method.line or 1
		qualname = f'{owner_type.qualname}${method.stem}$$__closure_trampoline__'

		erased_self_arg = ast.arg( arg = 'erased_self', lineno = line, col_offset = 0 )
		rest_args = [ ast.arg( arg = p.stem, lineno = line, col_offset = 0 ) for p in method.parameters ]

		# owner_type has no natural resolvable-by-name spelling from this
		# synthesized function's own lexical context - resolved_type
		# bypasses namespace resolution entirely (see _lower_compiler_
		# cast's own comment on this escape hatch)
		owner_type_ref = ast.Name( id = '<closure_owner>', ctx = ast.Load(), lineno = line, col_offset = 0 )
		owner_type_ref.resolved_type = owner_type
		cast_call = ast.Call(
			func = ast.Attribute(
				value = ast.Name( id = 'compiler', ctx = ast.Load(), lineno = line, col_offset = 0 ),
				attr = 'cast', ctx = ast.Load(), lineno = line, col_offset = 0,
			),
			args = [ owner_type_ref, ast.Name( id = 'erased_self', ctx = ast.Load(), lineno = line, col_offset = 0 ) ],
			keywords = [], lineno = line, col_offset = 0,
		)
		# the cast is inlined DIRECTLY as the call's own receiver expression
		# - deliberately NOT assigned to a named local first (`real_self =
		# compiler.cast(...)`, then `real_self.method(...)`). A named
		# local's own RHS, here a compiler.cast(...) call, is never
		# recognized as aliasing (_is_aliasing_expr only special-cases
		# Name/Attribute/ClosureType - an ordinary Call is always "fresh,
		# owned"), so cfg.assign() would push a REAL epilogue entry for it
		# and decref it at the trampoline's own return - a real, confirmed
		# bug (a real compile+run test showed the receiver's refcount
		# short by one after every closure call): the cast doesn't create
		# a new owned reference, it's a reinterpretation of the SAME
		# pointer the closure's own `self` field already owns, only
		# BORROWED for the duration of this call - exactly like an
		# ordinary method's own self parameter already is (cfg.py's
		# enter_self, OwnState.BORROWED). Inlining the cast as a bare
		# expression sidesteps this entirely: a CastWrap's own temp result
		# is never fresh_temp()-registered (only Call/Allocate results are
		# - see _emit), so nothing ever schedules a decref for it at all,
		# matching the borrowed semantics this needs
		method_call = ast.Call(
			func = ast.Attribute(
				value = cast_call, attr = method.stem, ctx = ast.Load(), lineno = line, col_offset = 0,
			),
			args = [ ast.Name( id = p.stem, ctx = ast.Load(), lineno = line, col_offset = 0 ) for p in method.parameters ],
			keywords = [], lineno = line, col_offset = 0,
		)
		none_return = isinstance( method.return_type, Scalar ) and method.return_type.stem == 'NoneType'
		call_stmt: ast.stmt = (
			ast.Expr( method_call, lineno = line, col_offset = 0 ) if none_return
			else ast.Return( value = method_call, lineno = line, col_offset = 0 )
		)

		node = ast.FunctionDef(
			name = '$$__closure_trampoline__',
			args = ast.arguments(
				posonlyargs = [], args = [ erased_self_arg, *rest_args ],
				vararg = None, kwonlyargs = [], kw_defaults = [], kwarg = None, defaults = [],
			),
			body = [ call_stmt ],
			decorator_list = [], returns = None, type_params = [],
			lineno = line, col_offset = 0, end_lineno = line, end_col_offset = 0,
		)
		ast.fix_missing_locations( node )

		erased_self_param = Parameter( stem = 'erased_self', qualname = f'{qualname}.erased_self', file = method.file, line = method.line, type = ptr_none_type )
		rest_params = [
			Parameter( stem = p.stem, qualname = f'{qualname}.{p.stem}', file = method.file, line = method.line, type = p.type )
			for p in method.parameters
		]
		trampoline = Function(
			stem = '$$__closure_trampoline__', qualname = qualname, file = method.file, line = method.line,
			cls = None, node = node, parameters = [ erased_self_param, *rest_params ], return_type = method.return_type,
			is_static = True, resolve = None,
		)
		trampoline.add_name( 'erased_self', erased_self_param )
		for p in rest_params:
			trampoline.add_name( p.stem, p )

		self.schedule( trampoline )
		self._closure_trampolines[key] = trampoline
		return trampoline

	def find_name_recursive( self, node: ast.Attribute ) -> tuple[object,str]|None:
		''' Resolve a dotted ast.Attribute expression (builtins.OSError.
		FileNotFoundError) to the terminal scope object and the final
		attribute name. Purely walks .names dicts — no ensure_resolved,
		no scheduling, no type-resolving. The chain is assumed to already
		be fully resolved by type_resolver.py before lowering runs.

		Returns (terminal_object, last_attr) on success, None when the
		root isn't an ast.Name or isn't a registered name at all (caller
		falls through to the normal value-lowering path).

		Records a specific error via discovery.fail when an intermediate
		attr is missing so the user gets a clear message rather than the
		generic "not a value" from the value-lowering fallthrough. '''
		# collect attrs right-to-left: builtins.OSError.FileNotFoundError → ['FileNotFoundError', 'OSError']
		attrs: list[str] = []
		cur: ast.expr = node
		while isinstance( cur, ast.Attribute ):
			attrs.append( cur.attr )
			cur = cur.value
		if not isinstance( cur, ast.Name ):
			return None
		root_name = cur.id
		obj: object|None = self.discovery.find_name_or_none( root_name )
		if obj is None:
			return None
		# only walk when the root is a scope-like object (Module, ClassLike,
		# etc.) — a local Variable or bare Function has no .names of its own
		# and should fall through to the normal value-lowering path
		if not isinstance( getattr( obj, 'names', None ), dict ):
			return None
		# walk intermediate scopes (all but the last attr) through .names
		for attr in reversed( attrs[1:] ):
			names = getattr( obj, 'names', None )
			if not isinstance( names, dict ):
				self.discovery.fail( f'{root_name} has no members, cannot look up {attr!r} ({ast.unparse(node)})', node )
				return None
			obj = names.get( attr )
			if obj is None:
				self.discovery.fail( f'{root_name} has no attribute {attr!r} ({ast.unparse(node)})', node )
				return None
		return obj, attrs[0]

	_SUBSCRIPT_ALTERNATIVES = 'call .__getitem__(...) directly and consume its Result yourself instead'

	_CMP_OPCODES: dict[type,'ir.CmpOp'] = {
		ast.Eq: ir.CmpOp.EQ,
		ast.NotEq: ir.CmpOp.NE,
		ast.Lt: ir.CmpOp.LT,
		ast.LtE: ir.CmpOp.LE,
		ast.Gt: ir.CmpOp.GT,
		ast.GtE: ir.CmpOp.GE,
	}

	# --- shared helpers ----------------------------------------------------------

	def _ensure_resolved( self, obj: object ) -> object:
		# moved to TypeResolver.ensure_resolved (type_resolver.py) - kept
		# here as a thin delegate since this file calls it ~15 times and the
		# behavior (resolve now + unconditionally schedule + swap a
		# Specialization for its monomorphized form) is still exactly what
		# every one of those call sites needs. See TypeResolver's own
		# docstring for why this can't wait for schedule()'s work queue.
		return self._type_resolver.ensure_resolved( obj )

	def _resolve_call_target( self, target: Function ) -> None:
		# a @virtual call's STATIC target (whatever chain_lookup found at
		# the call site's own declared receiver type) is NEVER itself
		# directly invoked - real dispatch goes through the vtable at
		# runtime (see emitter_c.py's _emit_virtual_call), reaching whatever
		# concrete override actually applies. compiler.py's own
		# _schedule_interface_vtable_impls already schedules each
		# CONSTRUCTED class's real per-slot implementation independently -
		# scheduling the STATIC target here too would be redundant at best,
		# and actively wrong when it resolves to an unfulfilled root
		# declaration (a stub body, `...` - see PLAN_SUBCLASSING_VTABLES_
		# COM.md's "Unimplemented @virtual methods"): _ensure_resolved
		# unconditionally schedules its target for real lowering, and
		# lowering a stub body as if it were a real function fails outright.
		# Only .resolve() (populating parameters/return_type, needed for
		# THIS call's own type-checking/emission) is needed here - not the
		# scheduling side effect.
		if target.is_virtual:
			if target.resolve is not None:
				target.resolve()
			return
		if target.is_inline:
			# PLAN_INLINE.md - an @inline target is never itself a real
			# compile unit (_lower_inline_call splices its body instead of
			# ever emitting a Call to it) - _ensure_resolved's unconditional
			# scheduling side effect would otherwise still compile it as
			# real, dead, never-called code (confirmed by a real repro:
			# Result[T,E].is_ok, @inline'd and called through a receiver -
			# some_result.is_ok() - reaches this exact branch, since
			# _attr_lookup_callable already hands back an already-
			# monomorphized, non-generic Function for it - see PLAN_
			# INLINE.md's own note on that path). Same shape as the
			# @virtual carve-out just above: resolve the signature (needed
			# to lower args against declared parameter types), skip the
			# scheduling side effect.
			if target.resolve is not None:
				target.resolve()
			return
		self._ensure_resolved( target )

	def _attr_lookup( self, owner_type: Type|None, attr: str, ctx: ast.AST ) -> Variable:
		# _ensure_resolved is the one place a Specialization gets swapped for
		# its real, substituted ClassLike - owner_type past this point is
		# never itself a Specialization, and its .names already has
		# substituted field/method entries (see monomorphize.py), so no
		# separate per-field substitution is needed here anymore
		owner_type = self._ensure_resolved( owner_type )
		if isinstance( owner_type, Specialization ) and isinstance( owner_type.base, Scalar ) and owner_type.base.stem in ( 'Ptr', 'ConstPtr' ):
			# dot-operator on a raw pointer means arrow - `p.attr` looks up
			# `attr` on the POINTEE's own type, same as `p[0].attr` already
			# does (see _expr_Subscript's identical pointee-inference for
			# GetItem) - the OPERAND embedded in the resulting ir.GetAttr
			# stays the pointer itself (unchanged), only the NAME LOOKUP
			# redirects here; emitter_c.py's _member_access_operator reads
			# that same Ptr[T]/ConstPtr[T] type to decide `->` over `.`
			owner_type = self._ensure_resolved( owner_type.args[0] )
		if isinstance( owner_type, TaggedUnion ) and attr in ( 'tag', 'data' ) and owner_type.names.get( attr ) is None:
			# tag/data are synthesized lazily, the first time the union is
			# actually constructed or matched against (UnionStorage.get) -
			# only reachable here for a PLAIN (non-generic) union: a
			# Specialization's own monomorphize_class already triggers this
			# itself before anything reads its .names. A method reading
			# self.tag/self.data directly (e.g. Result.is_ok()) could be
			# scheduled/lowered before anything else in THIS compilation
			# ever triggers that synthesis (the work queue has no ordering
			# guarantee) - trigger it here too, lazily, the moment it's
			# actually needed
			self._union_storage.get( owner_type )
		if isinstance( owner_type, ( CStruct, RCClass )):
			found = owner_type.chain_lookup( attr )
		else:
			names = getattr( owner_type, 'names', None )
			if not isinstance( names, dict ):
				self.discovery.fail( f'{owner_type!r} has no members, cannot look up {attr!r} ({ast.unparse(ctx)})', ctx )
			found = names.get( attr )
		if not isinstance( found, Variable ):
			self.discovery.fail( f'{owner_type.qualname if owner_type else "?"} has no attribute {attr!r}', ctx )
		self._ensure_resolved( found )
		return found

	def _substituted_field( self, found: Variable, owner_type: Type|None ) -> Variable:
		return self._monomorphizer.substituted_field( found, owner_type )

	def _substitute_type_params( self, t: Type|None, type_params: list[TypeVar], args: list[Type] ) -> Type|None:
		return self._monomorphizer.substitute_type_params( t, type_params, args )

	def _monomorphized_function( self, spec: Specialization ) -> Function:
		return self._monomorphizer.monomorphized_function( spec )

	def monomorphize_class( self, spec: Specialization ) -> ClassLike:
		return self._monomorphizer.monomorphize_class( spec )

	def _try_resolve_namespace( self, node: ast.expr ) -> Name|None:
		return self._type_resolver._try_resolve_namespace( node )



	def _attr_lookup_callable( self, owner_type: Type|None, attr: str, ctx: ast.AST ) -> Function|Overload:
		return self._type_resolver._attr_lookup_callable( owner_type, attr, ctx )

	def _match_call_args( self, target: Function, call: ast.Call ) -> tuple[list[tuple[Parameter,ast.expr]],list[tuple[Parameter,ast.expr]]]:
		if target.parameters is None:
			# target's own parameter resolution already failed (and recorded
			# an error - see discovery.py's _resolve_guarded/_make_function_
			# resolver, which can leave .parameters at its None default) -
			# fail cleanly here instead of crashing below on `for p in None`
			self.discovery.fail( f'{target.qualname} could not be resolved (see earlier error): {ast.unparse(call)}', call )
		if any( isinstance( a, ast.Starred ) for a in call.args ):
			self.discovery.fail( f'*args not supported yet: {ast.unparse(call)}', call )
		if any( kw.arg is None for kw in call.keywords ):
			self.discovery.fail( f'**kwargs not supported yet: {ast.unparse(call)}', call )
		positional_params = [ p for p in target.parameters if not p.is_vararg and not p.is_kwarg and not p.is_kwonly ]
		if len( call.args ) > len( positional_params ):
			self.discovery.fail( f'too many positional arguments: {ast.unparse(call)}', call )
		positional = list( zip( positional_params, call.args ))
		keyword: list[tuple[Parameter,ast.expr]] = []
		for kw in call.keywords:
			param = next(( p for p in target.parameters if p.stem == kw.arg and not p.is_vararg and not p.is_kwarg ), None )
			if param is None:
				self.discovery.fail( f'{target.qualname} has no parameter {kw.arg!r}', call )
			keyword.append(( param, kw.value ))
		positional = [ ( param, self._check_move_argument( target, param, expr, call )) for param, expr in positional ]
		keyword = [ ( param, self._check_move_argument( target, param, expr, call )) for param, expr in keyword ]
		return positional, keyword

	def _check_move_argument( self, target: Function, param: Parameter, expr: ast.expr, call: ast.Call ) -> ast.expr:
		# both sides of a move[T] parameter must agree, checked here (once,
		# for every _match_call_args caller - plain calls, generic calls,
		# both explicit-subscript and inferred) rather than downstream:
		# move(x) and plain x lower to an identical Operand once past this
		# point, so this is the only place that can still tell them apart.
		# Unwraps a valid move(expr) down to expr - callers only ever see
		# the real argument expression from here on
		is_move_call = isinstance( expr, ast.Call ) and isinstance( expr.func, ast.Name ) and expr.func.id == 'move'
		if isinstance( param.type, Move ):
			if not is_move_call:
				self.discovery.fail(
					f"{target.qualname}: parameter {param.stem!r} is move[{param.type.inner.qualname}] - "
					f"call site must pass move({ast.unparse(expr)}): {ast.unparse(call)}",
					call,
				)
			if len( expr.args ) != 1 or expr.keywords:
				self.discovery.fail( f'move(...) takes exactly one argument: {ast.unparse(expr)}', call )
			return expr.args[0]
		if is_move_call:
			self.discovery.fail(
				f"{target.qualname}: parameter {param.stem!r} is not move[T] - "
				f"call site must not wrap it in move(...): {ast.unparse(call)}",
				call,
			)
		return expr

	# stems of intrinsic types a Python literal of this exact type could
	# plausibly be lowered as - deliberately coarse (no int-range/value
	# validation exists anywhere yet, see _expr_Constant), just enough to
	# rule out a string literal matching an i32 parameter and vice versa.
	# `type(value) is X`, not isinstance - bool is an int subclass in
	# Python, and ast.Constant.value is only ever bool|int|str|bytes|None
	_LITERAL_COMPATIBLE_STEMS: dict[type,tuple[str,...]] = {
		bool: ( 'bool', ),
		int: ( 'i8', 'u8', 'i16', 'u16', 'i32', 'u32', 'i64', 'u64', 'i128', 'u128', 'isize', 'usize' ),
		str: ( 'str', ),
		bytes: ( 'bytes', ),
	}

	def _schedule_interface_construction( self, target_cls: CStruct ) -> None:
		# heap-allocating an @interface CStruct goes through sys.alloc[T],
		# the same real allocation path everything else in the language
		# uses (see _schedule_rcclass_construction's identical reasoning) -
		# NO automatic destructor scheduling here though (unlike RCClass):
		# there's no automatic refcounting/RC management for an @interface
		# CStruct COM object at all - AddRef/Release are the user's own
		# ordinary virtual methods, never wired into metalpy's automatic
		# Incref/Decref (see PLAN_SUBCLASSING_VTABLES_COM.md's own decision)
		sys_alloc_fn = self._type_resolver._resolve_sys_function( 'alloc' )
		alloc_spec = self.discovery._get_or_create_specialization( sys_alloc_fn, [ target_cls ] )
		self.schedule( alloc_spec )

	def _schedule_rcclass_construction( self, target_cls: RCClass, concrete_type: Type ) -> None:
		# guarantees sys.alloc[concrete_type] is a real, lowered compile unit
		# by the time the emitter sees the resulting ir.Allocate - the
		# emitter independently synthesizes the call to it (mangled
		# qualname, same convention as everything else), so this has to
		# actually exist regardless of whether the user's own program ever
		# wrote `import sys` (same posture as _resolve_sys_function's own
		# doc). Shared by _lower_allocate_fields (the no-__init__/field=value
		# path) and _try_lower_construct_call (the real __init__ path) -
		# both eventually emit an ir.Allocate for a real RCClass and need
		# identical scheduling
		sys_alloc_fn = self._type_resolver._resolve_sys_function( 'alloc' )
		alloc_spec = self.discovery._get_or_create_specialization( sys_alloc_fn, [ concrete_type ])
		self.schedule( alloc_spec )
		# every constructed RCClass needs its own destructor eventually
		# synthesized by the emitter (emit_c walks compiler.rcclasses, one
		# destructor function per entry - see emitter_c.py's Phase 4 work) -
		# that destructor calls sys.free on the object's own backing memory
		# and, if the class declares one, the user's own __del__ - both need
		# to already be real, lowered compile units by the time the emitter
		# needs to call them. Triggered at CONSTRUCTION time (same as
		# sys.alloc above), not merely when the class is referenced as a
		# type - scheduling this for every bare type annotation would drag
		# in sys.free's own transitive dependencies (real HeapFree/crt free
		# externs) for classes that are never actually instantiated
		sys_free_fn = self._type_resolver._resolve_sys_function( 'free' )
		self.schedule( sys_free_fn )
		del_fn = target_cls.get_local( '__del__' ) # target_cls is always the abstract base - methods aren't re-specialized per Specialization (Specialization.names passes through to .base.names)
		if isinstance( del_fn, Function ):
			self.schedule( del_fn )

	_OR_RETURN_ALTERNATIVES = 'or_return() always propagates the error to the caller - there is no other way for the enclosing function to receive it'
	_RESULT_CONSUMING_METHODS = ( 'is_ok', 'is_err', 'unwrap', 'unwrap_or' ) # or_return() is handled separately - see _lower_or_return

	def _unify_type_param( self, type_params: list[TypeVar], declared: Type|None, actual: Type|None, bindings: dict[int,Type], node: ast.AST, context_qualname: str ) -> None:
		# generalized over an explicit type_params list (rather than always
		# reading target.type_params) so this same unification shared by
		# both a generic FREE function's own type params (_lower_inferred_
		# generic_call) and a generic CLASS's type params (_lower_class_
		# generic_method_call - Result.Ok/.Err reached with no receiver to
		# read a concrete Specialization's args from directly)
		if declared is None or actual is None:
			return
		if any( declared is tv for tv in type_params ):
			existing = bindings.get( id( declared ) )
			# _same_type, not a bare `is` - two argument positions can
			# reveal the identical specialization through two different
			# representations (e.g. one already monomorphized, the other
			# a fresh Specialization built from an annotation) - see
			# Monomorphizer.origin_of's own docstring
			if existing is not None and existing is not actual and not self._type_resolver._same_type( existing, actual ):
				self.discovery.fail(
					f'{context_qualname}(...): type parameter {declared.stem!r} is inferred as both '
					f'{existing.qualname} and {actual.qualname} by different arguments: {ast.unparse(node)}',
					node,
				)
			bindings[ id( declared ) ] = actual
			return
		if isinstance( declared, Specialization ):
			# _as_specialization, not a bare isinstance(actual, Specialization)
			# check - actual may already be the real, monomorphized object
			# itself (not a Specialization wrapper) if substitute_type_params's
			# own eager-monomorphize step got to it first - see Monomorphizer.
			# origin_of's own docstring
			actual_spec = self._type_resolver._as_specialization( actual )
			if actual_spec is not None and declared.base is actual_spec.base:
				for d_arg, a_arg in zip( declared.args, actual_spec.args ):
					self._unify_type_param( type_params, d_arg, a_arg, bindings, node, context_qualname )
			return
		if isinstance( declared, CallableType ) and isinstance( actual, CallableType ):
			# Ptr[Callable[[T],K]] reaches here via the Specialization branch
			# above's own recursion (its single type ARG is the CallableType
			# itself) - e.g. a generic key: Ptr[Callable[[T],K]] parameter,
			# matched against a Ptr[Callable[[i32],i32]]-typed argument
			# (a real function/nested-def reference's own FunctionRef type -
			# PLAN_CALLABLE.md), binds T=i32/K=i32 the same way Specialization's
			# own args do
			for d_arg, a_arg in zip( declared.arg_types, actual.arg_types ):
				self._unify_type_param( type_params, d_arg, a_arg, bindings, node, context_qualname )
			self._unify_type_param( type_params, declared.return_type, actual.return_type, bindings, node, context_qualname )
			return

	def _dispatch_operand_for_param( self, node: ast.AST, target: Function, param: Parameter, args: list[ir.Operand], kwargs: dict[str,ir.Operand] ) -> ir.Operand:
		if param.stem in kwargs:
			return kwargs[param.stem]
		index = next( ( i for i, p in enumerate( target.parameters or [] ) if p is param ), None )
		if index is not None and index < len( args ):
			return args[index]
		self.discovery.fail( f'{target.qualname}: cannot locate the call-site argument for parameter {param.stem!r}', node )


class FunctionLowering:
	'''
	Everything Lowering.lower_function/lower_global need that's scoped to ONE
	function body (or one global's initializer) rather than shared across the
	whole compile run - the instruction stream being built, temp/label
	counters, the current CFG, defer/construction bookkeeping, and so on (see
	__init__ for the full field list). A fresh instance is constructed for
	every lower_function/lower_global call, INCLUDING a nested/reentrant call
	made mid-way through lowering an enclosing function (_expr_Lambda's eager
	lowering path, when a lambda's own return type needs to be inferred from
	its body before the enclosing call's own generic type parameters can be
	bound - PLAN_LAMBDA.md) - so an outer and inner lowering never share
	mutable state, and there's no field list to keep in sync by hand the way
	an explicit save/restore around a single shared instance would need.

	self.lowering is the single persistent Lowering instance this was built
	from - discovery, the type resolver, the schedule callback, the shared
	UnionStorage/Monomorphizer, closure-trampoline cache, and the lambda
	counter all live there instead, and are reached through this
	back-reference (self.lowering.discovery, etc.) throughout this class.
	'''

	def __init__( self, lowering: 'Lowering', fn: Function | None ) -> None:
		self.lowering = lowering
		self._instructions: list[ir.Instruction] = []
		self._temp_id = 0
		self._label_id = 0
		self._pending_temps: list[ir.Temp] = []
		self._current_fn = fn
		self._arithmetic_mode: list[arithmetic_mode.ArithmeticMode] = [ arithmetic_mode.ArithmeticChecked() ]
		self._loop_depth = 0
		self._loop_labels: list[tuple[str,str]] = []
		self._in_deferred_body = False
		self._defer_flags: list[Variable] = []
		self._return_value_var = None
		# PLAN_INLINE.md - @inline call splicing (see _lower_inline_call).
		# _inlining_stack (by id(target)) is the reentrancy guard - a target
		# already present means direct or mutual @inline recursion, rejected
		# rather than spliced forever. _inline_binding_id is a monotonic
		# counter giving each splice's synthesized self/parameter bindings
		# their own unique C-safe name, so they never collide with the
		# ENCLOSING function's own real locals/self/parameters of the same
		# name (emitter_c.py's local-declaration tracking is by C name, not
		# by object identity - see _lower_inline_call's own comment).
		self._inlining_stack: list[int] = []
		self._inline_binding_id = 0

	def run( self ) -> list[ir.Instruction]:
		fn = self._current_fn
		module = self.lowering._find_module_for( fn )
		if fn.extern_lib is not None:
			# @extern(lib, symbol) - a foreign call signature declaration,
			# not a real body to lower (discovery.py already required a
			# stub body - see _is_stub_body). No CFG/epilogue/locals
			# machinery applies here at all - just the bare signature, for
			# a future emitter to declare rather than define. Compiler._lower
			# is what actually registers the library dependency (see its
			# extern_libs bookkeeping) - this only has to emit the shape
			self._emit( ir.FuncStart( name = fn.qualname, params = fn.parameters or [], return_type = fn.return_type, extern_lib = fn.extern_lib, extern_symbol = fn.extern_symbol ))
			self._emit( ir.FuncEnd( name = fn.qualname ))
			return self._instructions

		with self.lowering.discovery.module_context( module ):
			with ( self.lowering.discovery.scope_context( fn.cls ) if fn.cls is not None else nullcontext() ):
				with self.lowering.discovery.scope_context( fn ):
					# self is deliberately excluded from fn.parameters/fn.names
					# in discovery.py (_make_function_resolver's add_param) so
					# overload matching never has to think about it - but that
					# means it was never made resolvable at all. The method
					# body obviously needs it, so it's synthesized here,
					# lowering-only, the moment we start lowering a method body
					# RCCLASS ATTRIBUTE LIFETIME.md / the approved plan - scoped
					# to non-subclassed RCClasses only (fn.cls.base is None):
					# subclassing/super()/attribute visibility aren't real
					# features yet, independent of this
					self._construction_self: Variable | None = None
					self._construction_fallible = False
					if fn.cls is not None and not fn.is_static and not fn.is_classmethod:
						self_type: Type|None = fn.cls
						if isinstance( fn.cls, CStruct ) and fn.cls.is_interface:
							# an @interface CStruct is never a plain value
							# type (see PLAN_SUBCLASSING_VTABLES_COM.md) -
							# self is Ptr[T], for every method, virtual or
							# not (consistency, per the plan doc's own
							# decision) - self.x/self.method() still read
							# like ordinary attribute access thanks to the
							# Ptr[T]/ConstPtr[T] dot-operator (see
							# _attr_lookup/_expr_Attribute's own pointee-
							# redirect, and emitter_c.py's matching
							# _member_access_operator rule)
							ptr_cls = self.lowering.discovery.get_intrinsics()['Ptr']
							self_type = self.lowering.discovery._get_or_create_specialization( ptr_cls, [ fn.cls ] )
						self_param = Parameter( stem = 'self', qualname = f'{fn.qualname}.self', file = fn.file, line = fn.line, type = self_type )
						fn.add_name( 'self', self_param )
						# fn.cls may be a Specialization for a monomorphized
						# generic-class __init__ (see Lowering._lower_generic_
						# construction_args) - unwrap to the real RCClass for
						# the isinstance/.base checks below and the field list
						# construction needs further down. NOTE: RCClass.base
						# means "parent class in an inheritance chain" while
						# Specialization.base means "the generic template" -
						# not the same thing, don't conflate them
						self_cls = self.lowering._ensure_resolved( fn.cls ) if isinstance( fn.cls, Specialization ) else fn.cls
						if fn.stem == '__init__' and isinstance( self_cls, RCClass ):
							self._construction_self = self_param
							self._construction_fallible = self.lowering._init_fallibility( fn )

					if '$payload_cls' in fn.names:
						# a synthesized union-member constructor (see
						# union_storage.py's _build_member_constructor).
						# $union_cls stays whatever UnionStorage.get() built
						# at synthesis time - always the ABSTRACT union,
						# which is exactly right: _lower_allocate_fields's
						# own existing substitution (fn_cls.base is
						# target_cls) already handles the OUTER
						# .__allocate__() call correctly once fn.cls is a
						# concrete Specialization, the same way a real
						# hand-written method's body (always textually
						# saying `Result.__allocate__`, never `Result[i32,
						# E].__allocate__`) already relies on. $payload_cls
						# is different: nothing substitutes a bare
						# construct-call's OWN target class, so it's
						# refreshed here to the CONCRETE, correctly-
						# substituted payload class (built by
						# monomorphize_class - see its own "fresh payload_cls
						# per specialization" comment) whenever fn.cls is a
						# Specialization - mirrors how self_cls above is
						# also computed fresh per lowering call rather than
						# baked in once
						if isinstance( fn.cls, Specialization ):
							concrete_union = self.lowering.monomorphize_class( fn.cls )
							payload_cls = concrete_union.names['data'].type
							fn.add_name( '$payload_cls', payload_cls )

					for param in fn.parameters or []:
						self.lowering.schedule( param.type )
					self.lowering.schedule( fn.return_type )

					none_type = self.lowering.discovery.get_none_type()
					noreturn_type = self.lowering.discovery.get_intrinsics()['NoReturn']
					# eagerly created whenever it COULD be needed (whether it
					# actually ends up referenced depends on whether any
					# return ever routes through current_epilogue_label()/
					# OrJump, only known once the body's actually lowered) -
					# harmless when unused: a synthetic Variable, never added
					# to fn.names, that simply never appears in any emitted
					# instruction if nothing ever needs it
					self._return_value_var = (
						Variable( stem = '__return_value', qualname = f'{fn.qualname}.__return_value', file = fn.file, line = fn.line, type = fn.return_type )
						if fn.return_type not in ( none_type, noreturn_type )
						else None
					)

					self._emit( ir.FuncStart( name = fn.qualname, params = fn.parameters or [], return_type = fn.return_type ))
					# constructed AFTER FuncStart - CFGState's own prologue
					# building (a copy[T] union parameter's tag-gated Incref)
					# can call new_temp, which immediately emits its own
					# DeclareTemp, so FuncStart must already be in the stream
					bool_cls = self.lowering.discovery.get_intrinsics()['bool']
					self._cfg = cfg.CFGState(
						fn,
						bool_type = bool_cls,
						new_temp = self._new_temp,
						new_label = self._new_label,
						union_storage = self.lowering._union_storage.get,
					)
					if self._construction_self is not None:
						for attr in self_cls.attributes:
							self.lowering._ensure_resolved( attr ) # each field's own .type is lazily resolved, separate from the class itself - same as _lower_allocate_fields's identical loop
						self._cfg.enter_construction( self_param, self_cls.attributes )
					elif fn.cls is not None and not fn.is_static and not fn.is_classmethod:
						self._cfg.enter_self( self_param, is_move = fn.is_move )
					for instr in self._cfg.prologue_instructions:
						self._emit( instr )
					if self._construction_self is not None:
						self._emit_construction_defaults( self_cls, self_param, module )
					body_start = len( self._instructions )
					# a subclass's own __init__ must open with
					# super().__init__(...) as its literal first statement
					# when its base has one to chain to (RCClass single
					# inheritance, Phase 2 of the RCClass-subclassing plan) -
					# handled once here, before the ordinary per-statement
					# loop below, which then only ever sees whatever's left
					# (unaffected for every other function, and for a
					# construction-only class with no base/no chained
					# __init__, this returns fn.node.body unchanged)
					body_stmts = fn.node.body
					if self._construction_self is not None:
						body_stmts = self._lower_super_init_if_required( self_cls, self_param )
					for stmt in body_stmts:
						# one bad statement doesn't stop the rest of this
						# function's body from being lowered (and error-collected) -
						# mirrors discovery.py's per-.resolve()/per-top-level-statement
						# recovery boundaries
						try:
							self._lower_stmt( stmt )
						except CompileError:
							continue

					# reaching the closing brace with no explicit `return` on
					# this path is __init__'s success path too - every
					# required attribute must already be initialized here.
					# Done BEFORE the branch below is even chosen: it cancels
					# each attribute's own epilogue entry (ownership transfers
					# into the now-complete self), which current_epilogue_label()
					# below has to see already applied - otherwise a
					# construction-only function with nothing else pending
					# would wrongly look like it still has a live entry to
					# jump to. A no-op whenever an explicit return already
					# completed construction on every reachable path (see
					# _stmt_Return's own identical call)
					if self._construction_self is not None:
						self._complete_construction_or_fail( fn )

					# validated unconditionally, whether or not the body
					# actually falls off the end for real: if every path
					# already returned explicitly, merge_if()'s own
					# terminates-reconciliation has already left
					# self._unchecked_results correctly empty/reconciled by
					# this point (and each individual `return` already ran
					# this same check at its own point), so this is a no-op
					# in that case - not a second, redundant error source
					try:
						self._cfg.check_unchecked_results( None )
					except CompileError as e:
						self.lowering.discovery.fail( str( e ), fn.node )

					if self._cfg.current_epilogue_label() is not None:
						# some return (or OrJump) already jumped into the
						# shared epilogue ladder (_stmt_Return/_consume_checked_
						# result, via current_epilogue_label()), or nothing did
						# but entries are still pending at the function's own
						# closing brace (an implicit `return None`/fall-off
						# reaching them the same way) - either way,
						# build_epilogue_ladder() covers whatever's still
						# pending, RC decrefs and defer/errdefer replays alike
						self._emit_epilogue( fn, none_type, body_start )
					elif fn.return_type is none_type and self.lowering._body_may_fall_off_the_end( fn.node.body ):
						# nothing pending to unwind - but falling off the end
						# without an explicit `return` is still a real exit
						# (implicit `return None`, same as Python). Every
						# explicit `return` already does this itself (see
						# _stmt_Return's own else branch) - this only covers
						# the specific case nothing else does: reaching the
						# function's closing brace with no `return` at all
						self._emit( ir.Return( value = None ))

					self._emit( ir.FuncEnd( name = fn.qualname ))

		return self._instructions

	def run_global( self, var: Variable ) -> list[ir.Instruction]:
		module = self.lowering._find_module_for( var )

		with self.lowering.discovery.module_context( module ):
			if var.init is not None:
				# a global's own initializer can itself construct an RC value
				# (e.g. the Unicode case-mapping tables' own lazily-built
				# dict/list globals) - _lower_allocate_fields and friends need
				# a real CFGState the same way an ordinary function body does,
				# just with no fn/self/construction of its own (fn=None - see
				# CFGState's own docstring on this)
				bool_cls = self.lowering.discovery.get_intrinsics()['bool']
				self._cfg = cfg.CFGState(
					None,
					bool_type = bool_cls,
					new_temp = self._new_temp,
					new_label = self._new_label,
					union_storage = self.lowering._union_storage.get,
				)
				self._pending_temps = []
				operand = self._lower_expr( var.init, var.type )
				self._emit( ir.Assign( dest = var, src = operand ))
				for t in reversed( self._pending_temps ):
					self._emit( ir.DeleteTemp( temp = t ))

		return self._instructions

	def _emit_epilogue( self, fn: Function, none_type: Type, body_start: int ) -> None:
		# every return/OrJump/fall-off-the-end that has anything pending
		# (self._cfg.current_epilogue_label() was not None) funnels through
		# here exactly once, at the function's own closing brace -
		# build_epilogue_ladder() replays the WHOLE stack (RC decrefs and
		# defer/errdefer replays interleaved by declaration order, deepest/
		# most-recently-pushed first)
		#
		# flag inits have to run before *any* code that could set them -
		# easiest to guarantee by splicing them in right after FuncStart
		# rather than tracking every branch that could reach a defer statement
		flag_inits = [
			ir.Assign( dest = flag, src = ir.Const( type = flag.type, value = False ))
			for flag in self._defer_flags
		]
		self._instructions[body_start:body_start] = flag_inits

		self._pending_temps = []
		for instr in self._cfg.build_epilogue_ladder( lambda: self._build_is_err_check( fn.node )):
			self._emit( instr )
		for t in reversed( self._pending_temps ):
			for instr in self._cfg.delete_temp( t ):
				self._emit( instr )
			self._emit( ir.DeleteTemp( temp = t ))
		return_value = self._return_value_var if fn.return_type is not none_type else None
		self._emit( ir.Return( value = return_value ))

	def _build_is_err_check( self, node: ast.AST ) -> tuple[list[ir.Instruction],ir.Temp]:
		''' the DeclareTemp+Call that checks self._return_value_var.is_err(),
		built as plain instructions rather than emitted directly - cfg.py's
		_replay() (via this callback) decides exactly where they land. With
		per-Epilogue labels, a single check computed once up front (the old
		design, back when there was only ever one shared epilogue label)
		wouldn't be reached by every jump that might need it - some land
		deeper in the ladder, skipping past it entirely (see
		build_epilogue_ladder()'s own comment) - so this is called fresh,
		deliberately uncached, every time an errdefer entry's own replay
		actually needs it. '''
		bool_cls = self.lowering.discovery.find_name( 'bool', node )
		is_err_fn = self.lowering._attr_lookup_callable( self._return_value_var.type, 'is_err', node )
		# resolve only (NOT _ensure_resolved, which also unconditionally
		# schedules is_err_fn as a compile unit) - is_err's genericity is
		# inherited from Result's own class type params (same as Result.
		# Ok/.Err/.is_ok - see _lower_class_generic_method_call's own
		# identical "receiver already concrete" branch), so the ABSTRACT
		# is_err_fn is never itself the right thing to schedule/call: its
		# synthesized `self` parameter would be typed as bare Result, which
		# has no real C struct body anywhere (only concrete specializations
		# do) - a genuine "incomplete type" compile error confirmed via a
		# real errdefer+clang round trip once _ensure_resolved's own
		# incidental scheduling was scheduling BOTH the abstract AND the
		# correctly-monomorphized version side by side
		assert is_err_fn.resolve is None, f'internal compiler error - {is_err_fn.qualname} was not fully resolved by the type_resolver module'
		return_type = self._return_value_var.type
		if isinstance( return_type, Specialization ) and return_type.base is is_err_fn.cls:
			# the receiver's type (self._return_value_var, always a
			# concrete Result[T,E] specialization by the time this runs)
			# already pins down the concrete args, so this is just an
			# ordinary monomorphization
			method_spec = self.lowering.discovery._get_or_create_specialization( is_err_fn, return_type.args )
			self.lowering.schedule( method_spec )
			is_err_fn = self.lowering._monomorphized_function( method_spec )
		else:
			self.lowering.schedule( is_err_fn )
		temp = ir.Temp( type = bool_cls, id = self._temp_id )
		self._temp_id += 1
		self._pending_temps.append( temp )
		return [
			ir.DeclareTemp( temp = temp ),
			ir.Call( dest = temp, target = is_err_fn, receiver = self._return_value_var, args = [], kwargs = {} ),
		], temp

	def _emit_construction_defaults( self, cls: RCClass, self_param: Variable, module: Module ) -> None:
		''' every defaulted attribute (attr.init is not None) gets an
		unconditional prologue assignment before __init__'s own
		user-written body runs - a later `self.a = ...` in the body (if
		any) then becomes an ordinary attr_assign() replace, decref-ing
		the just-created default. Lowered in the CLASS's own scope, not
		__init__'s - a default expression can reference other class-level
		names, but self isn't in scope for it, matching ordinary Python
		class-body semantics. '''
		for attr in cls.attributes:
			if attr.init is None:
				continue
			with self.lowering.discovery.module_context( module ):
				with self.lowering.discovery.scope_context( cls ):
					default_value = self._lower_expr( attr.init, attr.type )
			for instr in self._cfg.attr_assign( attr, default_value, is_alias = self.lowering._is_aliasing_expr( attr.init, default_value.type )):
				self._emit( instr )
			self._emit( ir.SetAttr( obj = self_param, attr = attr.stem, value = default_value ))

	def _complete_construction_or_fail( self, fn: Function ) -> None:
		try:
			self._cfg.complete_construction( fn.qualname )
		except CompileError as e:
			self.lowering.discovery.fail_loc( str( e ), fn.file, fn.line )

	# --- super().__init__(...) constructor chaining (RCClass, single inheritance) --

	def _super_init_shape( self, node: ast.expr ) -> tuple[ast.Call,ast.Call|None] | None:
		''' recognizes `super().__init__(...)` (base __init__ infallible) or
		`super().__init__(...).or_return()` (base __init__ fallible) as an
		EXACT textual shape - `super` is never a real registered name
		anywhere in this language (there is no builtin/intrinsic for it),
		so this has to be recognized here, before ordinary call resolution
		ever sees it, the same textual-recognition posture as defer/
		errdefer/compiler.X/or_return() itself already uses throughout this
		file. Returns (init_call, or_return_call) - or_return_call is None
		for the plain (infallible) spelling, otherwise the OUTER .or_return()
		ast.Call (handed to _lower_or_return unchanged, so ITS OWN existing
		checked-result propagation logic - OrReturn/OrJump - is reused
		verbatim rather than reimplemented here). Returns None for anything
		that isn't this exact shape - never a compile error by itself,
		callers decide what "not this shape" means in their own context
		(required-and-missing vs. used somewhere it isn't allowed at all). '''
		or_return_call: ast.Call|None = None
		call = node
		if (
			isinstance( call, ast.Call ) and isinstance( call.func, ast.Attribute ) and call.func.attr == 'or_return'
			and not call.args and not call.keywords
		):
			or_return_call = call
			call = call.func.value
		if not ( isinstance( call, ast.Call ) and isinstance( call.func, ast.Attribute ) and call.func.attr == '__init__' ):
			return None
		receiver = call.func.value
		if not (
			isinstance( receiver, ast.Call ) and isinstance( receiver.func, ast.Name ) and receiver.func.id == 'super'
			and not receiver.args and not receiver.keywords
		):
			return None
		return call, or_return_call

	def _lower_super_init_if_required( self, self_cls: RCClass, self_param: Variable ) -> list[ast.stmt]:
		''' called right before a subclass's own __init__ body is lowered
		(self._construction_self is already set) - if self_cls.base has a
		chained __init__ anywhere in ITS OWN chain (RCClass.chain_lookup,
		Phase 1), THIS __init__ must open with super().__init__(...) (or,
		when the base's own __init__ is fallible,
		super().__init__(...).or_return()) as literally its first statement
		- handled specially here rather than through the ordinary
		_stmt_Expr dispatch (see _super_init_shape's own comment on why).
		Returns the REMAINING statements for the ordinary per-statement
		loop to process - fn.node.body[1:] when this consumed statement 0,
		otherwise fn.node.body unchanged.

		A base with no chained __init__ at all needs no super() call - if
		it also has no fields anywhere in its own chain, there is nothing
		for this __init__ to be responsible for on the base's behalf at
		all (matches a root class's own construction exactly, just with a
		harmless base contributing nothing). If it DOES have fields but no
		__init__ to chain to, declaring a subclass __init__ at all is
		rejected outright - a deliberate Phase 2 scope limit (see the
		RCClass-subclassing plan's own Phase 2 notes): no sugar exists yet
		for filling in a field-only ancestor's fields from inside a
		subclass's own __init__, and inventing one isn't this phase's job. '''
		fn = self._current_fn
		if self_cls.base is None:
			return fn.node.body
		base_init = self_cls.base.chain_lookup( '__init__' )
		if base_init is None:
			if self_cls.base.flattened_attributes():
				self.lowering.discovery.fail(
					f'{fn.qualname}: cannot declare __init__ - base {self_cls.base.qualname} has field(s) but no '
					f'__init__ to chain to via super().__init__() (not supported yet)',
					fn.node,
				)
			return fn.node.body
		# resolve AND schedule - base_init might otherwise never become a
		# real compiled unit if nothing else ever calls it directly (an
		# ordinary call's own target already goes through _ensure_resolved/
		# schedule() somewhere upstream; this call site is entirely our own,
		# so it has to do that itself)
		base_init = self.lowering._ensure_resolved( base_init )
		shape = self._super_init_shape( fn.node.body[0].value ) if fn.node.body and isinstance( fn.node.body[0], ast.Expr ) else None
		if shape is None:
			self.lowering.discovery.fail(
				f'{fn.qualname}: must call super().__init__(...) as its first statement '
				f'({self_cls.base.qualname} has its own __init__ to chain to)',
				fn.node.body[0] if fn.node.body else fn.node,
			)
		init_call, or_return_call = shape
		is_fallible = self.lowering._init_fallibility( base_init )
		if is_fallible and or_return_call is None:
			self.lowering.discovery.fail(
				f'super().__init__(...): {self_cls.base.qualname}.__init__ is fallible - must be consumed via '
				f'.or_return(): {ast.unparse(fn.node.body[0])}',
				fn.node.body[0],
			)
		if not is_fallible and or_return_call is not None:
			self.lowering.discovery.fail(
				f'super().__init__(...).or_return(): {self_cls.base.qualname}.__init__ is not fallible - remove '
				f'.or_return(): {ast.unparse(fn.node.body[0])}',
				fn.node.body[0],
			)
		self.lowering.schedule( base_init.return_type )
		for param in base_init.parameters or []:
			self.lowering.schedule( param.type )
		args, kwargs = self._lower_call_args( base_init, init_call )
		dest = self._new_temp( base_init.return_type ) if is_fallible else None
		call = ir.Call( dest = dest, target = base_init, receiver = self_param, args = args, kwargs = kwargs, is_super_init_call = True )
		self._emit( call )
		if or_return_call is not None:
			assert dest is not None
			self._lower_or_return( or_return_call, dest, want_result = False )
		self._cfg.complete_base_construction( self_cls.base.flattened_attributes() )
		return fn.node.body[1:]

	# --- temp/instruction bookkeeping ----------------------------------------

	def _emit( self, instr: ir.Instruction ) -> None:
		# a Call/Allocate's dest is always a genuinely fresh, owned value
		# from the caller's perspective (same rule _is_aliasing_expr already
		# encodes for Call; Allocate is fresh by definition) - registering it
		# here, centrally, at the exact moment it's actually emitted, is what
		# guarantees every one of these sites is covered instead of needing
		# individual fresh_temp() calls hunted down at each of the many
		# places that build a Call/Allocate (plain calls, generic calls,
		# conditional dispatch, union-receiver dispatch, struct/union
		# construction, ...). Gated on self._current_fn - lower_global()
		# never constructs a CFGState at all, and self._cfg would otherwise
		# be whatever function was lowered most recently (this Lowering
		# instance is reused across units), a strictly worse outcome than
		# just skipping it for globals
		if self._current_fn is not None and isinstance( instr, ( ir.Call, ir.Allocate )) and isinstance( instr.dest, ir.Temp ):
			self._cfg.fresh_temp( instr.dest, instr.dest.type )
		if self._current_fn is not None:
			self._check_self_escape_in( instr )
		self._instructions.append( instr )

	def _check_self_escape_in( self, instr: ir.Instruction ) -> None:
		# self can only be used as the receiver of `self.attr`
		# (GetAttr.obj/SetAttr.obj, deliberately excluded here) until
		# __init__ finishes constructing it - see cfg.check_self_escape().
		# Checking every OTHER operand field centrally, at emission time,
		# covers every site that could hand self off somewhere it shouldn't
		# without hunting each one down individually - same centralization
		# fresh_temp() already uses above. A no-op outside __init__
		# (check_self_escape() itself short-circuits when nothing's under
		# construction) - the instruction-type gate below also means this
		# never touches self._cfg before it exists (FuncStart, emitted
		# before CFGState is constructed, matches none of these types)
		operands: list[ir.Operand] = []
		if isinstance( instr, ir.Call ):
			# is_super_init_call's receiver (self, mid-construction) is
			# deliberately excluded too, alongside GetAttr.obj/SetAttr.obj
			# above - see ir.Call.is_super_init_call's own comment. args/
			# kwargs still go through the ordinary check below regardless
			if instr.receiver is not None and not instr.is_super_init_call:
				operands.append( instr.receiver )
			operands += instr.args
			operands += instr.kwargs.values()
		elif isinstance( instr, ir.Assign ):
			operands.append( instr.src )
		elif isinstance( instr, ir.Return ):
			if instr.value is not None:
				operands.append( instr.value )
		elif isinstance( instr, ir.SetItem ):
			operands.append( instr.value )
		elif isinstance( instr, ir.Allocate ):
			operands += instr.fields.values()
		for operand in operands:
			try:
				self._cfg.check_self_escape( operand, self._current_fn.qualname )
			except CompileError as e:
				self.lowering.discovery.fail_loc( str( e ), self._current_fn.file, self._current_fn.line )

	def _new_temp( self, t: Type ) -> ir.Temp:
		temp = ir.Temp( type = t, id = self._temp_id )
		self._temp_id += 1
		self._pending_temps.append( temp )
		self._emit( ir.DeclareTemp( temp = temp ))
		return temp

	def _new_label( self, prefix: str ) -> str:
		label = f'__{prefix}_{self._label_id}__'
		self._label_id += 1
		return label

	# --- statements ------------------------------------------------------------

	def _lower_stmt( self, node: ast.stmt ) -> None:
		# _pending_temps is shared/mutable rather than passed explicitly, so a
		# statement whose own handler recursively lowers nested statements
		# (currently only _stmt_With) must not let those nested calls' own
		# resets/flushes clobber this call's view of it - save/restore around
		# the whole thing, same idea as scope_context's stack push/pop
		outer_pending = self._pending_temps
		self._pending_temps = []
		try:
			method = getattr( self, f'_stmt_{node.__class__.__name__}', None )
			if method is None:
				self.lowering.discovery.fail( f'unsupported statement: {ast.unparse(node)}', node )
			method( node )
			# a no-op by the time this runs for a Return (see _stmt_Return's
			# own comment - it flushes _pending_temps ITSELF, before its own
			# ir.Return/ir.Jump, precisely so this generic post-statement
			# flush - unconditionally emitted AFTER method(node) returns,
			# i.e. AFTER any unconditional terminator that statement itself
			# already emitted - never lands as dead code following it)
			self._flush_pending_temps()
		finally:
			self._pending_temps = outer_pending

	def _flush_pending_temps( self ) -> None:
		''' decref+DeleteTemp every still-pending temp (reverse declaration
		order), then clear the list. A temp genuinely fresh_temp()-
		registered (see _emit) and never consumed by assign()/return_()/
		untrack_temp()/move()/field_value() along the way (e.g. `foo(
		SomeClass() )` where SomeClass() is passed into a plain, non-move[T]
		parameter - nothing ever untracks it) still needs its own decref
		right here, at the natural end of the temporary's own expression-
		scoped lifetime. A no-op for every already-consumed temp (already
		untracked by whichever hook consumed it) and every non-RC temp
		(never registered in the first place). Factored out of _lower_stmt
		so _stmt_Return can call it explicitly BEFORE its own terminator
		(ir.Return/ir.Jump) instead of relying on _lower_stmt's own post-
		method call, which - for every OTHER statement kind, fine, since
		none of them emit an unconditional jump/return of their own - would
		otherwise land as unreachable code right after one. '''
		for t in reversed( self._pending_temps ):
			for instr in self._cfg.delete_temp( t ):
				self._emit( instr )
			self._emit( ir.DeleteTemp( temp = t ))
		self._pending_temps = []

	def _stmt_Return( self, node: ast.Return ) -> None:
		if self._in_deferred_body:
			# a defer/errdefer body's code runs later, replayed inline at the
			# epilogue (see _register_defer_block) - a `return` inside it
			# doesn't have a sensible meaning (it's not really executing at
			# this point in the function, and jumping to __epilogue__ from
			# CODE ALREADY INSIDE the epilogue replay is nonsensical). Same
			# check _register_defer_block already applies to nested defer/
			# errdefer, catches nested cases too (return inside an if/while
			# inside the defer body) since _in_deferred_body stays set for
			# the whole capture, not just the top-level statement
			self.lowering.discovery.fail( f'return is not allowed inside a defer/errdefer body: {ast.unparse(node)}', node )
		value = self._lower_expr( node.value, self._current_fn.return_type ) if node.value is not None else None
		try:
			self._cfg.check_unchecked_results( value )
		except CompileError as e:
			self.lowering.discovery.fail( str( e ), node )
		if self._construction_self is not None:
			# every return in a non-fallible __init__ is unconditionally
			# success (construction_fallible is False, so the `and` below
			# short-circuits) - no legal way to signal failure. In a
			# fallible one, only a return whose value is textually
			# Result.Err(...) is the failure path (partial init expected/
			# legal there, cleaned up normally by the ordinary return_()
			# unwind below - "clean up any that were initialized"); every
			# other shape (Result.Ok(...), or anything else - deliberately
			# not attempting deeper type-level inference here) requires
			# full initialization
			is_success = not ( self._construction_fallible and self.lowering._is_result_err_call( node.value ) is not None )
			if is_success:
				self._complete_construction_or_fail( self._current_fn )
		label = self._cfg.current_epilogue_label( value )
		if label is not None:
			# whatever's still pending (RC decrefs, defer/errdefer replays)
			# gets unwound once, later, by the shared ladder every other
			# return reaching this same label also jumps into
			# (build_epilogue_ladder(), emitted at the function's own
			# closing brace - see _emit_epilogue) - value has to survive
			# the jump some other way than a direct ir.Return
			if self._return_value_var is not None and value is not None:
				self._emit( ir.Assign( dest = self._return_value_var, src = value ))
			# value's own ownership (if it's a bare temp - `return
			# SomeConstructor(...)`, never assigned to a name) just
			# transferred into self._return_value_var above via the plain
			# ir.Assign - untrack it so _flush_pending_temps below doesn't
			# ALSO decref it (return_()'s own docstring explains the
			# identical concern for the other branch)
			self._cfg.untrack_temp( value )
			# flushed HERE, before this branch's own unconditional
			# ir.Jump - not left to _lower_stmt's own post-method flush,
			# which runs strictly after this whole method returns and so
			# would land as dead code following the Jump (see
			# _flush_pending_temps' own docstring for the general shape of
			# this bug: a single-statement function body like `def make()
			# -> Result[str,E]: return Result.Ok('hello'.upper())` used to
			# leave the intermediate str temp's own release permanently
			# unreachable, inflating the returned Result's refcount by one
			# forever)
			self._flush_pending_temps()
			self._emit( ir.Jump( target = label ))
		else:
			# either nothing is pending, or `value` IS itself one of the
			# still-live entries current_epilogue_label() can't route
			# through a shared label (see its own comment) - unwind inline,
			# right here, same as always. Still has to replay any pending
			# defer/errdefer entries itself (return_() does this now too -
			# they're just as "pending" as an RC decref from here)
			for instr in self._cfg.return_( value, lambda: self._build_is_err_check( node )):
				self._emit( instr )
			# same reasoning as the label-is-not-None branch above - flush
			# BEFORE this branch's own unconditional ir.Return, not after
			# (return_() already untracked `value` itself, so this only
			# ever cleans up OTHER still-pending temps - e.g. an
			# intermediate argument consumed into constructing `value`)
			self._flush_pending_temps()
			self._emit( ir.Return( value = value ))

	def _stmt_Pass( self, node: ast.Pass ) -> None:
		pass

	def _stmt_Global( self, node: ast.Global ) -> None:
		# a no-op: an unannotated Assign to a name that already exists
		# anywhere in the scope chain (local, enclosing, or global) always
		# reassigns that same one - find_name_or_none's scope chain already
		# falls through to the module scope on its own, so there's never a
		# separate shadowing local to opt out of. It only introduces a new
		# local when the name is unbound everywhere in the chain (see
		# _stmt_Assign's inference branch), which by definition has nothing
		# to shadow
		pass

	def _stmt_Delete( self, node: ast.Delete ) -> None:
		# del x - ends a local's lifetime early (see TODO.txt/RC MANAGEMENT.md:
		# a local created inside one arm of an if can be referenced only
		# within that arm unless it's del'd before the arm exits, matching
		# the other arm's "never created it either" state). Only a single
		# bare local name is supported - not del a.b, del a[i], or multiple
		# targets. Removing it from fn.names is enough on its own to make a
		# later reference fail (find_name won't find it) - the actual
		# Decref emission is cfg.py's job, wired in alongside its other hooks
		if len( node.targets ) != 1 or not isinstance( node.targets[0], ast.Name ):
			self.lowering.discovery.fail( f'del only supports a single local variable name: {ast.unparse(node)}', node )
		target = node.targets[0]
		fn = self._current_fn
		existing = fn.names.get( target.id )
		if not isinstance( existing, Variable ):
			self.lowering.discovery.fail( f'{target.id!r} is not a local variable, cannot del it', node )
		try:
			instructions = self._cfg.deleted( existing )
		except CompileError as e:
			self.lowering.discovery.fail( str( e ), node )
		for instr in instructions:
			self._emit( instr )
		del fn.names[target.id]

	def _stmt_ImportFrom( self, node: ast.ImportFrom ) -> None:
		# local (in-function) form of discovery.py's own visit_ImportFrom -
		# function bodies are deliberately never walked by discovery.py's
		# own visitor (see this class's docstring: "function bodies were
		# deliberately left unvisited in stage 1"), so an import written
		# inside a function body (lib/sys.py's memzero()/_alloc()/etc. -
		# one FFI declaration per @compiler.target(os=...) branch) only
		# ever reaches here, never discovery.py's version. Registered
		# directly into the current function's own scope (add_name, same
		# as a parameter) rather than resolved/scheduled eagerly - an
		# unused import costs nothing, same posture as _expr_Name's lazy
		# resolve-on-use
		parts: list[str] = []
		if node.level:
			parts.extend( self.lowering.discovery.module_stack[-1].qualname.split( '.' )[:-node.level] )
			if not parts:
				self.lowering.discovery.fail( f'unable to relative import from here: {ast.unparse(node)}', node )
		if node.module:
			parts.append( node.module )
		package = '.'.join( parts )
		try:
			mod = self.lowering.discovery.import_name( package )
		except FileNotFoundError as e:
			self.lowering.discovery.fail( str( e ), node )
		if not mod:
			self.lowering.discovery.fail( f'module {package!r} not found', node )
		for alias in node.names:
			item = mod.names.get( alias.name )
			if item is None:
				self.lowering.discovery.fail( f'module {package} does not export {alias.name!r}', node )
			self._current_fn.add_name( alias.asname or alias.name, item )

	def _stmt_Import( self, node: ast.Import ) -> None:
		for alias in node.names:
			try:
				mod = self.lowering.discovery.import_name( alias.name )
			except FileNotFoundError as e:
				self.lowering.discovery.fail( str( e ), node )
			self._current_fn.add_name( alias.asname or alias.name, mod )

	def _cfg_assign( self, dest: Variable, src: ir.Operand, *, is_alias: bool, node: ast.AST, track_result: bool = True, borrow: bool = False ) -> list[ir.Instruction]:
		# thin wrapper around cfg.assign() - now that it can raise
		# CompileError (see cfg.py's own unchecked-Result overwrite check),
		# every one of its 7 call sites needs the same discovery.fail()
		# conversion _stmt_If/loop_back_edge's own call sites already use,
		# or the raised-but-unrecorded error would just be silently
		# swallowed by the nearest enclosing per-statement `except
		# CompileError: continue` recovery boundary
		try:
			return self._cfg.assign( dest, src, is_alias = is_alias, track_result = track_result, borrow = borrow )
		except CompileError as e:
			self.lowering.discovery.fail( str( e ), node )

	def _stmt_AnnAssign( self, node: ast.AnnAssign ) -> None:
		if not isinstance( node.target, ast.Name ):
			self.lowering.discovery.fail( f'unsupported AnnAssign target: {ast.unparse(node)}', node )
		fn = self._current_fn
		var_type = self.lowering.discovery.visit( node.annotation )
		# var_type starts as whatever discovery.visit() returns - often a
		# bare, un-monomorphized Specialization - and STAYS that way for
		# var's own construction/_lower_expr's expected_type below. Fixed
		# up (see the resolution block after _lower_expr, below) only once
		# the RHS has actually been lowered, and only when node.value is
		# real - see that block's own comment for why both restrictions
		# are load-bearing, not incidental.
		self.lowering.schedule( var_type )
		var = Variable(
			stem = node.target.id,
			qualname = f'{fn.qualname}.{node.target.id}',
			file = fn.file,
			line = node.lineno,
			type = var_type,
		)
		fn.add_name( var.stem, var )
		if node.value is not None:
			operand = self._lower_expr( node.value, var_type )
			# Only NOW, after the RHS is fully lowered, swap var.type for
			# its resolved (monomorphized, if a Specialization) form -
			# ensure_resolved()'s own contract: "the SINGLE place a
			# Specialization gets swapped for the real, substituted thing
			# it stands in for - every caller MUST use the returned value,
			# or they see the abstract, unsubstituted base instead"
			# (Specialization.names/.resolve are raw passthroughs to it).
			# var.type staying an unresolved Specialization here is a real
			# gap: cfg.py's rc_leaves() reads a Specialization's ABSTRACT
			# base.attributes (still bare TypeVars for a generic union like
			# Result[T,E]) and silently concludes an annotated local like
			# `x: Result[SomeRCClass,E]` has no RC leaves at all, skipping
			# its own incref/decref entirely - confirmed directly with
			# AddressSanitizer, not just reasoning.
			#
			# Both restrictions below are load-bearing, found by real
			# regressions, not just caution:
			#
			# 1. Resolving BEFORE lowering the RHS (i.e. swapping var_type
			# up front and reusing it for _lower_expr's own expected_type)
			# regressed a real, unrelated bug into existence: for a
			# generic RCClass's own construction (`b: Box[i32] = Box(1)`),
			# eagerly monomorphizing Box[i32] here - before Box(1)'s own
			# construction-call lowering has had a chance to monomorphize
			# Box.__init__[i32] itself the ordinary way - raced it, and
			# the copy built here cached a version of Box.__init__[i32]
			# whose own `self` parameter was left typed as the abstract
			# Box[T] instead of the concrete Box[i32], which then got
			# scheduled as a bogus extra "Box[Box.T]" compile unit
			# (confirmed with a real repro against compiler.rcclasses,
			# not just a hunch). Resolving only after the RHS's own
			# construction-call machinery has already run first sidesteps
			# it - the resolution here then just reads back whatever it
			# already correctly cached (monomorphized_function's own
			# spec.monomorphized memoization), never racing it.
			#
			# 2. Only when node.value is not None (this whole branch) -
			# skipped for a bare declaration (`c: Box[u32]`, assigned via
			# a later, ordinary Assign statement, not this one) since
			# there's no RHS lowering here to resolve after in the first
			# place, and deferring is always safe: whatever later
			# statement actually assigns/uses c triggers its own
			# resolution through the ordinary paths (e.g. _attr_lookup's
			# own _ensure_resolved call), same as it always has.
			if self.lowering._monomorphizer._is_concrete( var_type ):
				var.type = self.lowering._ensure_resolved( var_type )
			for instr in self._cfg_assign( var, operand, is_alias = self.lowering._is_aliasing_expr( node.value, operand.type ), node = node ):
				self._emit( instr )
			self._emit( ir.Assign( dest = var, src = operand ))

	def _lower_attr_target_obj( self, value_node: ast.expr ) -> tuple[ir.Operand, Callable[[ir.Operand],None]|None]:
		''' the object operand for an attribute assignment target
		(target.attr = ...), plus an optional writeback callback the
		caller must invoke (with the SAME operand, now mutated via
		SetAttr) once the ordinary SetAttr has been emitted.

		Ordinarily there's no writeback needed - the object expression's
		own lowered value (self, an already-Ptr[T] variable, ...) IS the
		lvalue SetAttr writes through (the dot-operator already makes a
		bare Ptr[T] receiver work correctly here - see _attr_lookup's own
		pointee-redirect, emitter_c.py's _member_access_operator). But
		`ptr[idx].attr = value` is different: ptr[idx] ALONE (via
		_expr_Subscript's raw-pointer GetItem fallback, the only shape a
		raw pointer's own subscript has - no real __getitem__ method to
		dispatch to) loads a COPY of the pointee into a fresh temp, and
		writing through that copy silently drops the write entirely -
		confirmed by a real repro, not just reasoning. C has no single
		"address of the idx'th pointee, then ->field = value" primitive
		this compiler emits directly (unlike a bare Ptr[T] receiver,
		which is already the pointer itself) - so this reads the WHOLE
		element via ir.GetItem, returns that temp as the object SetAttr
		mutates (reusing every existing RC-tracking/attr-assignment path
		unchanged), and writes the WHOLE element back via ir.SetItem
		afterward - the same read-modify-write shape `ptr[idx] = value`
		(whole-value replacement) already uses one level up. ptr/index
		are lowered exactly once here (not re-lowered inside
		_expr_Subscript AND again for the writeback) - relowering the
		AST a second time would double-evaluate them, a real correctness
		risk if either expression has side effects (matches the same
		concern _stmt_AugAssign's own Attribute/Subscript-target
		restriction is about). '''
		if isinstance( value_node, ast.Subscript ):
			ptr_obj = self._lower_expr( value_node.value, None )
			if (
				self.lowering._find_method( ptr_obj.type, '__getitem__' ) is None
				and self.lowering._type_resolver._is_ptr_specialization( ptr_obj.type )
			):
				index_type = self.lowering.discovery.get_intrinsics()['usize']
				index = self._lower_expr( value_node.slice, index_type )
				elem_type = ptr_obj.type.args[0]
				elem = self._new_temp( elem_type )
				self._emit( ir.GetItem( dest = elem, obj = ptr_obj, index = index ))
				def writeback( updated: ir.Operand ) -> None:
					self._emit( ir.SetItem( obj = ptr_obj, index = index, value = updated ))
				return elem, writeback
		return self._lower_expr( value_node, None ), None

	def _resolve_narrow_member( self, name: str, member_stem: str, node: ast.AST ) -> Variable:
		# type_resolver.py hands down only a STEM (see its own comment on
		# why - resolved against the TEXTUAL/abstract union at that pass,
		# T/E may still be bare TypeVars there) - re-resolve the real,
		# substituted member against `name`'s own already-monomorphized
		# type here, the same pattern _coerce_into_union already uses. A
		# parameter's own declared type (unlike a local var initialized
		# from a call's already-eagerly-monomorphized return type) stays a
		# genuine Specialization wrapping the ABSTRACT base - .base alone
		# isn't enough, has to go through monomorphize_class same as any
		# other generic-class use site, or the member's own .type resolves
		# to the unsubstituted TypeVar instead of the real leaf (str, not T).
		# Shared by _stmt_Assign's own is_narrowing_bind handling (match/if-
		# desugared narrowing) and _stmt_While's own exit-narrowing (Phase 7).
		subject_var = self.lowering.discovery.find_name( name, node )
		assert isinstance( subject_var, Variable )
		base = self.lowering.monomorphize_class( subject_var.type ) if isinstance( subject_var.type, Specialization ) else subject_var.type
		self.lowering._union_storage.get( base )
		member = next( ( attr for attr in base.attributes if attr.stem == member_stem ), None )
		assert member is not None
		return member

	def _stmt_Assign( self, node: ast.Assign ) -> None:
		if getattr( node, 'is_narrowing_bind', False ):
			# type_resolver.py's _match_pattern: `match x: case T(x):`
			# reusing the subject's own name - x's real Variable/storage is
			# untouched, this is a pure compile-time fact ("reads of x from
			# here until this scope's own restore() may read through the
			# union's own payload instead") - no IR at all, see cfg.py's
			# narrow()/_expr_Name's own comment for the read-side rewrite.
			target_name = node.targets[0]
			assert isinstance( target_name, ast.Name )
			member = self._resolve_narrow_member( target_name.id, node.narrows_member_stem, node )
			self._cfg.narrow( target_name.id, member )
			return
		if len( node.targets ) != 1:
			self.lowering.discovery.fail( f'multiple assignment targets not supported: {ast.unparse(node)}', node )
		target = node.targets[0]
		if isinstance( target, ast.Name ):
			existing = self.lowering.discovery.find_name_or_none( target.id )
			if existing is not None:
				if not isinstance( existing, Variable ):
					self.lowering.discovery.fail( f'{target.id!r} is not a variable, cannot assign to it', node )
				self._cfg.unnarrow( target.id ) # a real reassignment invalidates whatever this name was previously narrowed to - see cfg.py's own comment
				operand = self._lower_expr( node.value, existing.type )
				for instr in self._cfg_assign( existing, operand, is_alias = self.lowering._is_aliasing_expr( node.value, operand.type ), node = node ):
					self._emit( instr )
				self._emit( ir.Assign( dest = existing, src = operand ))
			else:
				# first assignment to a name with no prior declaration - same
				# as an AnnAssign, but the type is inferred from the RHS
				# instead of coming from an explicit annotation
				operand = self._lower_expr( node.value, None )
				fn = self._current_fn
				var = Variable(
					stem = target.id,
					qualname = f'{fn.qualname}.{target.id}',
					file = fn.file,
					line = node.lineno,
					type = operand.type,
				)
				fn.add_name( var.stem, var )
				self.lowering.schedule( var.type )
				# type_resolver.py's visit_Match desugars `match r:` into
				# `__match_subj_N = r; if ...` and marks the synthesized
				# Assign with these two attributes (see its own comment) -
				# is_match_subject means the fresh __match_subj_N temp must
				# never itself become a tracked obligation (the if-chain
				# below only does raw tag Compares, never is_ok()/is_err(),
				# so nothing would ever clear it); match_clears_name carries
				# the ORIGINAL name through when the subject was a bare Name
				# - ordinary aliasing assignment deliberately does NOT clear
				# the source (see cfg.py's "Independent tracking"), but a
				# match statement genuinely IS the inspection of its subject
				is_match_subject = getattr( node, 'is_match_subject', False )
				is_alias = self.lowering._is_aliasing_expr( node.value, operand.type )
				# when the subject is a bare Name (is_alias=True), the
				# ORIGINAL name already owns a live reference for the whole
				# (function-scoped) rest of its lifetime, so __match_subj_N
				# only needs a BORROW, not its own Incref/epilogue-Decref
				# pair - see cfg.py's assign() borrow= doc for the bug this
				# fixes (a real, always-unbalanced-until-function-exit
				# Incref that inflated every compiler.refcount() read taken
				# inside a match arm). A non-Name subject (e.g. `match
				# make():`) has no such original owner, so it keeps full
				# ownership tracking unchanged (borrow=False there).
				for instr in self._cfg_assign( var, operand, is_alias = is_alias, node = node, track_result = not is_match_subject, borrow = is_match_subject and is_alias ):
					self._emit( instr )
				self._emit( ir.Assign( dest = var, src = operand ))
				match_clears_name = getattr( node, 'match_clears_name', None )
				if match_clears_name is not None:
					self._cfg.clear_result( match_clears_name )
		elif isinstance( target, ast.Attribute ):
			obj, writeback = self._lower_attr_target_obj( target.value )
			attr_var = self.lowering._attr_lookup( obj.type, target.attr, target )
			operand = self._lower_expr( node.value, attr_var.type )
			if self._construction_self is not None and obj is self._construction_self:
				# self.<attr> = value, inside __init__ construction itself -
				# tracked for definite-assignment/self-escape purposes (see
				# RCCLASS ATTRIBUTE LIFETIME.md and cfg.attr_assign())
				for instr in self._cfg.attr_assign( attr_var, operand, is_alias = self.lowering._is_aliasing_expr( node.value, operand.type )):
					self._emit( instr )
			elif cfg.rc_leaves( attr_var.type ):
				# ordinary SetAttr on an already-constructed instance -
				# "an RCClass is always complete, so setting an attribute
				# is always a replace" (RCCLASS ATTRIBUTE LIFETIME.md).
				# cfg.py doesn't track arbitrary struct instances' field
				# CONTENTS across statements (v1 scope cut - see cfg.py's
				# module docstring), so the current value is always read
				# fresh here rather than consulted from any tracked state
				old = self._new_temp( attr_var.type )
				self._emit( ir.GetAttr( dest = old, obj = obj, attr = target.attr ))
				for instr in self._cfg.attr_replace( attr_var.type, old, operand, is_alias = self.lowering._is_aliasing_expr( node.value, operand.type )):
					self._emit( instr )
			self._emit( ir.SetAttr( obj = obj, attr = target.attr, value = operand ))
			if writeback is not None:
				writeback( obj )
		elif isinstance( target, ast.Subscript ):
			obj = self._lower_expr( target.value, None )
			setitem_fn = self.lowering._find_method( obj.type, '__setitem__' )
			if setitem_fn is None:
				# no real __setitem__ declared (raw pointers, or any other
				# type that doesn't define subscript assignment as a method)
				# - falls back to the flat SetItem opcode, unconditionally
				# (mirrors _expr_Subscript's own raw-pointer GetItem fallback)
				index = self._lower_expr( target.slice, None )
				operand = self._lower_expr( node.value, None )
				self._emit( ir.SetItem( obj = obj, index = index, value = operand ))
			else:
				# a real __setitem__ - call it like any other method, then if
				# it returns Result[T,E], auto-consume it exactly like
				# _expr_Subscript's own __getitem__ call does: `obj[i] = v`
				# reads as sugar for `obj.__setitem__(i, v).or_return()`
				# whenever __setitem__ can fail
				self.lowering._ensure_resolved( setitem_fn )
				self.lowering.schedule( setitem_fn.return_type )
				index = self._lower_expr( target.slice, setitem_fn.parameters[0].type )
				operand = self._lower_expr( node.value, setitem_fn.parameters[1].type )
				if setitem_fn.return_type is self.lowering.discovery.get_none_type():
					# the ordinary/conventional case (matches Python's own
					# __setitem__ protocol, which always returns None) -
					# a real Temp dest here would try to assign C's void
					# return to a variable, which doesn't compile; no
					# Result to auto-consume either
					self._emit( ir.Call( dest = None, target = setitem_fn, receiver = obj, args = [ index, operand ], kwargs = {} ))
				else:
					call_dest = self._new_temp( setitem_fn.return_type )
					self._emit( ir.Call( dest = call_dest, target = setitem_fn, receiver = obj, args = [ index, operand ], kwargs = {} ))
					self._maybe_consume_result( node, call_dest, self.lowering._SUBSCRIPT_ALTERNATIVES )
		else:
			self.lowering.discovery.fail( f'unsupported Assign target: {ast.unparse(node)}', node )

	def _stmt_AugAssign( self, node: ast.AugAssign ) -> None:
		# x += y desugars to x = x + y (reusing whatever arithmetic mode is
		# active, exactly like a hand-written x = x + y would). A bare Name
		# target reads/writes via the synthesized BinOp+Assign below - safe
		# because a Name lookup has no side effects of its own. Attribute/
		# Subscript targets can't use that same trick (their object/index
		# expression would be evaluated twice - once to read, once to
		# resolve the write - a real correctness risk: `get_obj().x += 1`
		# must only call get_obj() once), so those two branches lower the
		# target's object/index exactly once themselves, then read/compute/
		# write through the SAME already-lowered operand(s) - mirroring
		# _lower_attr_target_obj's own reasoning and _stmt_Assign's own
		# Attribute/Subscript branches, just fused with a read first.
		if isinstance( node.target, ast.Name ):
			read = ast.Name( id = node.target.id, ctx = ast.Load() )
			ast.copy_location( read, node.target )
			binop = ast.BinOp( left = read, op = node.op, right = node.value )
			ast.copy_location( binop, node )
			assign = ast.Assign( targets = [ node.target ], value = binop )
			ast.copy_location( assign, node )
			self._stmt_Assign( assign )
		elif isinstance( node.target, ast.Attribute ):
			obj, writeback = self._lower_attr_target_obj( node.target.value )
			attr_var = self.lowering._attr_lookup( obj.type, node.target.attr, node.target )
			old = self._new_temp( attr_var.type )
			self._emit( ir.GetAttr( dest = old, obj = obj, attr = node.target.attr ))
			usize_cls = self.lowering.discovery.get_intrinsics()['usize']
			right_hint = usize_cls if self.lowering._type_resolver._is_ptr_specialization( old.type ) else old.type
			right = self._lower_expr( node.value, right_hint )
			result = self._lower_binop_values( node, old, right, attr_var.type )
			if self._construction_self is not None and obj is self._construction_self:
				# self.<attr> += value, inside __init__ construction itself -
				# same definite-assignment/self-escape tracking an ordinary
				# self.<attr> = value gets in _stmt_Assign
				for instr in self._cfg.attr_assign( attr_var, result, is_alias = False ):
					self._emit( instr )
			elif cfg.rc_leaves( attr_var.type ):
				# ordinary SetAttr on an already-constructed instance - `old`
				# is exactly the CURRENT value _stmt_Assign's own Attribute
				# branch would otherwise re-read via its own fresh GetAttr;
				# reusing it here avoids a redundant third read
				for instr in self._cfg.attr_replace( attr_var.type, old, result, is_alias = False ):
					self._emit( instr )
			self._emit( ir.SetAttr( obj = obj, attr = node.target.attr, value = result ))
			if writeback is not None:
				writeback( obj )
		elif isinstance( node.target, ast.Subscript ):
			obj = self._lower_expr( node.target.value, None )
			getitem_fn = self.lowering._find_method( obj.type, '__getitem__' )
			if getitem_fn is None:
				# no real __getitem__ declared (raw pointers, or any other
				# type that doesn't define subscript access as a method) -
				# mirrors _expr_Subscript's own raw-pointer GetItem fallback
				# and _stmt_Assign's own raw-pointer SetItem fallback, fused
				# around a single obj/index lowering
				if isinstance( obj.type, Specialization ) and isinstance( obj.type.base, Scalar ) and obj.type.base.stem in ( 'Ptr', 'ConstPtr' ):
					elem_type = obj.type.args[0]
				else:
					self.lowering.discovery.fail( f'cannot infer the element type of {ast.unparse(node.target)} - no expected type available from context', node.target )
				index_type = self.lowering.discovery.get_intrinsics()['usize']
				index = self._lower_expr( node.target.slice, index_type )
				old = self._new_temp( elem_type )
				self._emit( ir.GetItem( dest = old, obj = obj, index = index ))
				right = self._lower_expr( node.value, old.type )
				result = self._lower_binop_values( node, old, right, old.type )
				self._emit( ir.SetItem( obj = obj, index = index, value = result ))
			else:
				# a real __getitem__ - read via it like any other method
				# call, auto-consuming a Result exactly like an ordinary
				# `obj[i]` read already does, then write back via
				# __setitem__ the same way an ordinary `obj[i] = v` already
				# does - index is lowered exactly once, shared by both
				setitem_fn = self.lowering._find_method( obj.type, '__setitem__' )
				if setitem_fn is None:
					self.lowering.discovery.fail( f'{ast.unparse(node.target.value)} defines __getitem__ but not __setitem__ - cannot assign to {ast.unparse(node.target)}', node.target )
				self.lowering._ensure_resolved( getitem_fn )
				self.lowering.schedule( getitem_fn.return_type )
				index = self._lower_expr( node.target.slice, getitem_fn.parameters[0].type )
				get_dest = self._new_temp( getitem_fn.return_type )
				self._emit( ir.Call( dest = get_dest, target = getitem_fn, receiver = obj, args = [ index ], kwargs = {} ))
				old = self._maybe_consume_result( node.target, get_dest, self.lowering._SUBSCRIPT_ALTERNATIVES )
				right = self._lower_expr( node.value, old.type )
				result = self._lower_binop_values( node, old, right, old.type )
				self.lowering._ensure_resolved( setitem_fn )
				self.lowering.schedule( setitem_fn.return_type )
				if setitem_fn.return_type is self.lowering.discovery.get_none_type():
					self._emit( ir.Call( dest = None, target = setitem_fn, receiver = obj, args = [ index, result ], kwargs = {} ))
				else:
					set_dest = self._new_temp( setitem_fn.return_type )
					self._emit( ir.Call( dest = set_dest, target = setitem_fn, receiver = obj, args = [ index, result ], kwargs = {} ))
					self._maybe_consume_result( node.target, set_dest, self.lowering._SUBSCRIPT_ALTERNATIVES )
		else:
			self.lowering.discovery.fail( f'unsupported AugAssign target: {ast.unparse(node)}', node )

	def _stmt_Expr( self, node: ast.Expr ) -> None:
		if self._super_init_shape( node.value ) is not None:
			# only ever consumed specially as literally __init__'s own first
			# statement (_lower_super_init_if_required, called BEFORE this
			# per-statement loop even starts) - reaching here at all means
			# it's either not statement 0, or this isn't even __init__, or
			# there's no base to chain to in the first place
			self.lowering.discovery.fail(
				f'super().__init__(...) is only allowed as the literal first statement of a subclass\'s own __init__: {ast.unparse(node)}',
				node,
			)
		defer_kind = self.lowering._defer_kind_of_call( node.value )
		if defer_kind is not None:
			if len( node.value.args ) != 1 or node.value.keywords:
				self.lowering.discovery.fail( f'{defer_kind}(...) takes exactly one argument: {ast.unparse(node)}', node )
			single_stmt = ast.Expr( value = node.value.args[0] )
			ast.copy_location( single_stmt, node )
			self._register_defer_block( is_err_only = ( defer_kind == 'errdefer' ), body = [ single_stmt ], node = node )
			return
		if isinstance( node.value, ast.Constant ) and isinstance( node.value.value, str ):
			return # a docstring (or any other bare string literal used as a statement) - a no-op, same as _stmt_Pass
		if self.lowering._is_compiler_call( node.value ) == 'early_return':
			self._lower_compiler_early_return( node.value )
			return
		if self.lowering._is_compiler_call( node.value ) == 'decref':
			self._lower_compiler_decref( node.value )
			return
		if self.lowering._is_compiler_call( node.value ) == 'incref':
			self._lower_compiler_incref( node.value )
			return
		if self.lowering._is_compiler_call( node.value ) == 'atomic_store':
			self._lower_compiler_atomic_store( node.value )
			return
		if self.lowering._is_compiler_call( node.value ) == 'decref_dynamic':
			self._lower_compiler_decref_dynamic( node.value )
			return
		if not isinstance( node.value, ast.Call ):
			self.lowering.discovery.fail( f'unsupported expression statement: {ast.unparse(node)}', node )
		self._lower_call( node.value, None, want_result = False )

	def _stmt_With( self, node: ast.With ) -> None:
		if len( node.items ) != 1 or node.items[0].optional_vars is not None:
			self.lowering.discovery.fail( f'unsupported with statement: {ast.unparse(node)}', node )
		context_expr = node.items[0].context_expr

		defer_kind = self.lowering._defer_kind_of_with( context_expr )
		if defer_kind is not None:
			self._register_defer_block( is_err_only = ( defer_kind == 'errdefer' ), body = node.body, node = node )
			return

		attr = self.lowering._is_compiler_attr( context_expr )
		if attr == 'wrap_arithmetic':
			mode: arithmetic_mode.ArithmeticMode = arithmetic_mode.ArithmeticWrap()
		elif attr == 'saturate_arithmetic':
			mode = arithmetic_mode.ArithmeticSaturate()
		elif self.lowering._is_compiler_call( context_expr ) == 'panic_arithmetic':
			if len( context_expr.args ) != 1 or context_expr.keywords:
				self.lowering.discovery.fail( f'compiler.panic_arithmetic(...) takes exactly one argument: {ast.unparse(node)}', node )
			str_cls = self.lowering.discovery.find_name( 'str', node )
			errmsg = self._lower_expr( context_expr.args[0], str_cls )
			mode = arithmetic_mode.ArithmeticPanic( errmsg )
		else:
			self.lowering.discovery.fail( f'unsupported with statement: {ast.unparse(node)}', node )

		self._arithmetic_mode.append( mode )
		try:
			for stmt in node.body:
				# same per-statement recovery boundary as the top-level loop
				# in lower_function - one bad statement inside the with-block
				# doesn't stop the rest of it from being lowered
				try:
					self._lower_stmt( stmt )
				except CompileError:
					continue
		finally:
			self._arithmetic_mode.pop()

	def _static_type_of_value_expr( self, node: ast.expr ) -> Type|None:
		# compile-time-only: the static type of a value-shaped expression
		# (Name/Attribute) - no IR emitted, x itself is never evaluated or
		# lowered (unlike _lower_expr, which would emit a real GetAttr for
		# e.g. self.field, or even execute a call, just to inspect its
		# .type). Used by compiler.sizeof(x)'s value-argument fallback so
		# that e.g. compiler.sizeof(self) never turns self into a real
		# instruction operand - self is read only as self.type here, so
		# cfg.py's check_self_escape (which only inspects instruction
		# operands) never sees it, even mid-__init__ before construction
		# completes
		if isinstance( node, ast.Name ):
			name = self.lowering.discovery.find_name( node.id, node )
			if not isinstance( name, Variable ):
				return None
			member = self._cfg.narrowed_member( node.id )
			return member.type if member is not None else name.type
		if isinstance( node, ast.Attribute ):
			owner_type = self._static_type_of_value_expr( node.value )
			if owner_type is None:
				return None
			return self.lowering._attr_lookup( owner_type, node.attr, node ).type
		return None

	def _lower_compiler_sizeof( self, node: ast.Call, expected_type: Type|None ) -> ir.Operand:
		# compiler.sizeof(T) is a compile-time constant whenever T is
		# already concrete - it folds directly to an ir.Const, no runtime
		# computation involved. T is normally a TYPE reference, resolved
		# via _try_resolve_namespace (same as a generic call's own [T]
		# argument); compiler.sizeof(x) also accepts a plain VALUE
		# expression (self, a local, self.field, ...) - _try_resolve_namespace
		# either fails to resolve those (Attribute chains) or resolves to a
		# Variable/Parameter rather than a Type (bare names, since every
		# declared param/local is registered by discovery.find_name too),
		# so falling back to _static_type_of_value_expr's own non-emitting
		# type lookup covers both without ever lowering/evaluating x itself
		if len( node.args ) != 1 or node.keywords:
			self.lowering.discovery.fail( f'compiler.sizeof(...) takes exactly one type argument: {ast.unparse(node)}', node )
		resolved = self.lowering._try_resolve_namespace( node.args[0] )
		target_type = resolved if isinstance( resolved, Type ) else self._static_type_of_value_expr( node.args[0] )
		if target_type is None:
			self.lowering.discovery.fail( f'compiler.sizeof(...) argument must be a type or a value with a known type: {ast.unparse(node)}', node )
		if isinstance( target_type, TypeVar ):
			self.lowering.discovery.fail(
				f'compiler.sizeof({target_type.stem}) requires a concrete type - {target_type.qualname} is still an '
				f'unbound generic type parameter here (call the enclosing function through an explicit specialization, e.g. foo[SomeType](...))',
				node,
			)
		usize_cls = self.lowering.discovery.get_intrinsics()['usize']
		if size := getattr( target_type, 'sizeof', None ):
			return ir.Const( type = expected_type or usize_cls, value = size )
		# a C type declared via compiler.c_type('pthread_mutex_t', ...) -
		# as opaque to this compiler as a real ClassLike; stays a real
		# ir.SizeOf, letting the C compiler itself compute it
		if isinstance( target_type, CType ):
			self.lowering.discovery.required_headers.add( target_type.required_header )
			dest = self._new_temp( expected_type or usize_cls )
			self._emit( ir.SizeOf( dest = dest, type = target_type ))
			return dest

		# a real class-like type (RCClass/CStruct/CUnion/TaggedUnion, or a
		# concrete Specialization of one) - no field-layout algorithm exists
		# in this compiler (nor should one - that's the C compiler's own
		# job), so unlike an intrinsic scalar's sizeof, this can't fold to a
		# Python int here. Stays a real ir.SizeOf instruction instead - the
		# emitter emits a literal C `sizeof(...)` expression, letting the
		# target C compiler compute the real, layout-dependent size (needed
		# by sys.alloc[T]'s own body, e.g. sys.alloc[SomeRCClass](1) for
		# RCClass construction - see _lower_allocate_fields's RCClass branch)
		base = target_type.base if isinstance( target_type, Specialization ) else target_type
		if not isinstance( base, ( RCClass, CStruct, CUnion, TaggedUnion )):
			self.lowering.discovery.fail( f'compiler.sizeof({target_type.qualname}) is not supported yet - only intrinsic scalar types and real classes have a known size', node )
		self.lowering.schedule( target_type )
		dest = self._new_temp( expected_type or usize_cls )
		self._emit( ir.SizeOf( dest = dest, type = target_type ))
		return dest

	def _lower_compiler_is_rc( self, node: ast.Call, expected_type: Type|None ) -> ir.Operand:
		# compiler.is_rc(T) - a compile-time constant bool, true iff T is an
		# RCClass (possibly wrapped in a Specialization). T is always
		# concrete by the time this lowers (same "no unbound TypeVar"
		# requirement as compiler.sizeof), so this always folds directly to
		# an ir.Const - no runtime check, no emitter support needed at all.
		# Lets generic library code (list[T]'s own per-slot storage width -
		# an RCClass value IS a pointer everywhere else in this compiler,
		# but compiler.sizeof(T) deliberately stays the OBJECT's own struct-
		# body size always, for sys.alloc[T]'s sake - see its own docstring)
		# branch on T's own RC-ness without a new kind of type-level
		# reflection existing anywhere else in the language.
		#
		# Like compiler.sizeof(x), T also accepts a plain VALUE expression
		# (self, a local, ...) - moved here (from the outer Lowering class)
		# specifically so it can share compiler.sizeof(x)'s own non-
		# emitting _static_type_of_value_expr fallback, rather than
		# re-implementing a second, narrowing-unaware type lookup. Before
		# this fix, is_rc(x) on a value silently miscomputed instead of
		# erroring: _try_resolve_namespace(x) returns the VARIABLE (not a
		# Type) for a bare Name, and _is_RC(variable) - `variable.base if
		# isinstance(variable, Specialization) else variable` then
		# `isinstance(that, RCClass)` - is always False for a Variable,
		# regardless of the value's real type (compiler.is_rc(some_rc_var)
		# always folded to False, silently)
		if len( node.args ) != 1 or node.keywords:
			self.lowering.discovery.fail( f'compiler.is_rc(...) takes exactly one type argument: {ast.unparse(node)}', node )
		resolved = self.lowering._try_resolve_namespace( node.args[0] )
		target_type = resolved if isinstance( resolved, Type ) else self._static_type_of_value_expr( node.args[0] )
		if target_type is None:
			self.lowering.discovery.fail( f'compiler.is_rc(...) argument must be a type or a value with a known type: {ast.unparse(node)}', node )
		if isinstance( target_type, TypeVar ):
			self.lowering.discovery.fail(
				f'compiler.is_rc({target_type.stem}) requires a concrete type - {target_type.qualname} is still an '
				f'unbound generic type parameter here (call the enclosing function through an explicit specialization, e.g. foo[SomeType](...))',
				node,
			)
		bool_cls = self.lowering.discovery.get_intrinsics()['bool']
		return ir.Const( type = expected_type or bool_cls, value = self.lowering._type_resolver._is_RC( target_type ))

	def _lower_compiler_refcount( self, node: ast.Call, expected_type: Type|None ) -> ir.Operand:
		# compiler.refcount(x) - unlike compiler.sizeof(T), x is a real
		# VALUE (an RC object), not a type reference, so it's lowered via
		# _lower_expr like any other argument. A genuine runtime read (the
		# header's current count), not a compile-time constant - deliberately
		# opaque at this level (ir.RefCount), same spirit as Incref/Decref;
		# what it actually reads is a codegen/emitter concern, not this pass's
		if len( node.args ) != 1 or node.keywords:
			self.lowering.discovery.fail( f'compiler.refcount(...) takes exactly one argument: {ast.unparse(node)}', node )
		value = self._lower_expr( node.args[0], None )
		if not self.lowering._type_resolver._is_RC( value.type ):
			self.lowering.discovery.fail(
				f'compiler.refcount(...) argument must be a reference-counted value, not '
				f'{value.type.qualname if value.type else "?"}: {ast.unparse(node)}',
				node,
			)
		usize_cls = self.lowering.discovery.get_intrinsics()['usize']
		dest = self._new_temp( expected_type or usize_cls )
		self._emit( ir.RefCount( dest = dest, value = value ))
		return dest

	def _lower_compiler_addrof( self, node: ast.Call, expected_type: Type|None ) -> ir.Operand:
		# compiler.addrof(x) -> Ptr[T], translating directly to C's &x - x
		# must be a bare local variable/parameter name (matches SYNTAX.md's
		# "local variable" wording and C's own lvalue-only restriction on
		# &), not an arbitrary expression. _expr_Name already only ever
		# resolves to a Variable (never a Temp), so requiring the argument's
		# AST shape to be ast.Name is what actually enforces this - lowering
		# it via _lower_expr like any other value would silently accept e.g.
		# compiler.addrof(x.field), which has no address to take here (no
		# field-layout computation exists yet - that's an emitter concern)
		if len( node.args ) != 1 or node.keywords:
			self.lowering.discovery.fail( f'compiler.addrof(...) takes exactly one argument: {ast.unparse(node)}', node )
		arg_node = node.args[0]
		if not isinstance( arg_node, ast.Name ):
			self.lowering.discovery.fail( f'compiler.addrof(...) argument must be a bare local variable, not {ast.unparse(node)}', node )
		value = self._lower_expr( arg_node, None )
		ptr_cls = self.lowering.discovery.get_intrinsics()['Ptr']
		ptr_type = self.lowering.discovery._get_or_create_specialization( ptr_cls, [ value.type ] )
		dest = self._new_temp( expected_type or ptr_type )
		self._emit( ir.AddrOf( dest = dest, value = value ))
		return dest

	def _lower_compiler_atomic_load( self, node: ast.Call, expected_type: Type|None ) -> ir.Operand:
		if len( node.args ) != 1 or node.keywords:
			self.lowering.discovery.fail( f'compiler.atomic_load(...) takes exactly one argument: {ast.unparse(node)}', node )
		ptr = self._lower_expr( node.args[0], None )
		pointee = self.lowering._atomic_pointee_type( ptr.type, node )
		dest = self._new_temp( expected_type or pointee )
		self._emit( ir.AtomicLoad( dest = dest, ptr = ptr ))
		return dest

	def _lower_compiler_atomic_store( self, node: ast.Call ) -> None:
		# statement-only (see _stmt_Expr's own dispatch) - mirrors
		# compiler.incref/decref: no return value, nothing to hand back to
		# an expression context
		if len( node.args ) != 2 or node.keywords:
			self.lowering.discovery.fail( f'compiler.atomic_store(...) takes exactly two arguments: {ast.unparse(node)}', node )
		ptr = self._lower_expr( node.args[0], None )
		pointee = self.lowering._atomic_pointee_type( ptr.type, node )
		value = self._lower_expr( node.args[1], pointee )
		self._emit( ir.AtomicStore( ptr = ptr, value = value ))

	def _lower_compiler_atomic_rmw( self, node: ast.Call, expected_type: Type|None, op: ir.AtomicRMWOp ) -> ir.Operand:
		# shared by atomic_add/atomic_sub/atomic_exchange - same shape
		# (ptr, val), dest gets the value from BEFORE the op (C11
		# atomic_fetch_add/sub/exchange's own convention)
		if len( node.args ) != 2 or node.keywords:
			self.lowering.discovery.fail( f'compiler.atomic_{op.value}(...) takes exactly two arguments: {ast.unparse(node)}', node )
		ptr = self._lower_expr( node.args[0], None )
		pointee = self.lowering._atomic_pointee_type( ptr.type, node )
		value = self._lower_expr( node.args[1], pointee )
		dest = self._new_temp( expected_type or pointee )
		self._emit( ir.AtomicRMW( dest = dest, op = op, ptr = ptr, value = value ))
		return dest

	def _lower_compiler_atomic_compare_exchange( self, node: ast.Call, expected_type: Type|None ) -> ir.Operand:
		# C11 strong CAS: ptr, expected: Ptr[T], desired -> bool. On
		# failure *expected is written with the actual current value - that
		# side effect happens through `expected` itself (an ordinary Ptr[T]
		# the caller already owns), nothing more to hand back for it here
		if len( node.args ) != 3 or node.keywords:
			self.lowering.discovery.fail(
				f'compiler.atomic_compare_exchange(...) takes exactly three arguments (ptr, expected, desired): {ast.unparse(node)}',
				node,
			)
		ptr = self._lower_expr( node.args[0], None )
		pointee = self.lowering._atomic_pointee_type( ptr.type, node )
		expected = self._lower_expr( node.args[1], None )
		expected_pointee = self.lowering._atomic_pointee_type( expected.type, node )
		if expected_pointee is not pointee:
			self.lowering.discovery.fail(
				f'compiler.atomic_compare_exchange(...): ptr and expected must point to the same type: {ast.unparse(node)}',
				node,
			)
		desired = self._lower_expr( node.args[2], pointee )
		bool_cls = self.lowering.discovery.find_name( 'bool', node )
		dest = self._new_temp( expected_type or bool_cls )
		self._emit( ir.AtomicCompareExchange( dest = dest, ptr = ptr, expected = expected, desired = desired ))
		return dest

	def _lower_scalar_cast( self, target_type: Scalar, source: ast.expr|ir.Operand, node: ast.AST ) -> ir.Operand:
		# shared by compiler.cast(T, x) and T(x) construction-sugar - the
		# one place the actual Scalar-to-Scalar conversion logic lives.
		# `source` is EITHER an unlowered ast.expr (a bare literal - always
		# succeeds via bit-reinterpretation, decided at compile time, no
		# Result involved - -11 reinterpreted as u32 is exactly the
		# well-defined two's-complement value real WinAPI constants like
		# STD_OUTPUT_HANDLE rely on) OR an already-lowered ir.Operand (a
		# real runtime value, where "does this fit" is a genuine runtime
		# question - respects self._arithmetic_mode exactly like +/-/*
		# already do, reusing the same Check/Wrap/Saturate/panic_arithmetic
		# machinery, not a separate concept)
		if isinstance( source, ast.expr ):
			return self._lower_expr( source, target_type )
		operand = source
		opcode, extra = self._arithmetic_mode[-1].GetCast()
		return self._lower_arithmetic_op( node, opcode, extra, target_type, { 'operand': operand }, 'cast' )

	def _lower_compiler_cast( self, node: ast.Call, expected_type: Type|None ) -> ir.Operand:
		# compiler.cast(T, x) - T is a TYPE reference (resolved via
		# _try_resolve_namespace, same as compiler.sizeof's argument, not
		# _lower_expr), x is a real value. The call's own target type is
		# always authoritative for the result - unlike an ordinary literal,
		# an explicit cast overrides whatever the ambient expected_type is.
		# node.args[0].resolved_type, if set, bypasses namespace resolution
		# entirely - mirrors resolved_callee's own established escape hatch
		# (type_resolver.py's _synthesize_rcclass_destructor), for
		# compiler-synthesized AST that already knows its target Type
		# object directly and has no natural resolvable-by-name spelling
		# for it (a closure trampoline's own receiver cast, see
		# _get_or_create_closure_trampoline)
		if len( node.args ) != 2 or node.keywords:
			self.lowering.discovery.fail( f'compiler.cast(...) takes exactly two arguments: {ast.unparse(node)}', node )
		target_type = getattr( node.args[0], 'resolved_type', None )
		if target_type is None:
			target_type = self.lowering._try_resolve_namespace( node.args[0] )
		if target_type is None:
			self.lowering.discovery.fail( f'compiler.cast(...) first argument must be a type: {ast.unparse(node)}', node )
		if isinstance( target_type, TypeVar ):
			self.lowering.discovery.fail(
				f'compiler.cast({target_type.stem}, ...) requires a concrete type - {target_type.qualname} is still an '
				f'unbound generic type parameter here (call the enclosing function through an explicit specialization, e.g. foo[SomeType](...))',
				node,
			)
		if self.lowering._type_resolver._is_pointer_representable( target_type ):
			# Ptr[T]/ConstPtr[T], OR a bare RCClass - both are a single
			# machine pointer's worth of bits, just typed differently (see
			# _is_pointer_representable) - a plain reinterpret cast between
			# any of these (RawList's own byte-buffer indexing scheme needs
			# this: a Ptr[None] slot pointer reinterpreted as Ptr[T], OR
			# reinterpreted directly as a bare RC element T itself, since a
			# T-typed SLOT holds exactly T's own handle, not a Ptr[T] to
			# one - see list[T]'s own read/write helpers). Never fails at
			# runtime (unlike a scalar cast, which can lose bits) - no
			# arithmetic-mode concept applies, so this bypasses
			# _lower_scalar_cast/_lower_arithmetic_op entirely and reuses
			# ir.CastWrap directly, purely for its emitter_c.py shape
			# (`dest = (ctype)(operand);`, no overflow check) - not because
			# this is "wrap mode" in the arithmetic sense
			value = self._lower_expr( node.args[1], None )
			if not self.lowering._type_resolver._is_pointer_representable( value.type ):
				self.lowering.discovery.fail(
					f'compiler.cast({target_type.qualname}, ...) second argument must be a pointer or RC value, not '
					f'{value.type.qualname if value.type else "?"}: {ast.unparse(node)}',
					node,
				)
			dest = self._new_temp( expected_type or target_type )
			self._emit( ir.CastWrap( dest = dest, operand = value ))
			return dest
		if not isinstance( target_type, Scalar ):
			self.lowering.discovery.fail( f'compiler.cast({target_type.qualname}, ...) is not supported yet - only Scalar-to-Scalar and pointer-to-pointer casts are, for now', node )
		value_node = node.args[1]
		if isinstance( value_node, ast.Constant ):
			return self._lower_scalar_cast( target_type, value_node, node )
		value = self._lower_expr( value_node, None )
		if not isinstance( value.type, Scalar ):
			self.lowering.discovery.fail(
				f'compiler.cast(...) second argument must be a scalar value, not {value.type.qualname if value.type else "?"}: {ast.unparse(node)}',
				node,
			)
		return self._lower_scalar_cast( target_type, value, node )

	def _lower_compiler_early_return( self, node: ast.Call ) -> None:
		# compiler.early_return(err) - a same-function early bailout: usable
		# anywhere inside a function that itself returns Result[_,_], to
		# return Result.Err(err) immediately without writing the boilerplate
		# out by hand. Desugars to `return Result.Err(err)` and delegates to
		# _stmt_Return so it reuses the epilogue-vs-plain-Return split (and
		# errdefer's is_err() epilogue check, which already treats any
		# Result.Err landing in the return slot uniformly, not just the
		# OrJump path) rather than duplicating either.
		#
		# NOTE: Result.or_return()'s own written body uses this same call
		# (`compiler.early_return(self.data.v_Err)`), but that body is
		# never actually lowered as a real function - it's a spec, not
		# compilable code, because it would need this to trigger a return
		# in ITS CALLER's scope, not or_return()'s own (or_return's declared
		# return type is bare T, not Result[T,E] - `return Result.Err(...)`
		# from inside it could never type-check there). or_return() calls
		# are instead recognized and expanded directly at the call site -
		# see _lower_or_return.
		if len( node.args ) != 1 or node.keywords:
			self.lowering.discovery.fail( f'compiler.early_return(...) takes exactly one argument: {ast.unparse(node)}', node )
		fn = self._current_fn
		return_type = fn.return_type if fn is not None else None
		ok = fn is not None and self.lowering._type_resolver._result_shape( return_type ) is not None
		if not ok:
			where = f'{fn.qualname} returns {return_type.qualname if return_type else None}' if fn is not None else 'this is not inside a function'
			self.lowering.discovery.fail( f'compiler.early_return(...) requires the enclosing function to return Result[_,_] ({where})', node )

		err_call = ast.Call(
			func = ast.Attribute( value = ast.Name( id = 'Result', ctx = ast.Load() ), attr = 'Err', ctx = ast.Load() ),
			args = [ node.args[0] ],
			keywords = [],
		)
		ast.copy_location( err_call, node )
		ast.fix_missing_locations( err_call )
		return_stmt = ast.Return( value = err_call )
		ast.copy_location( return_stmt, node )
		self._stmt_Return( return_stmt )

	def _in_generic_class_method( self ) -> bool:
		# true while lowering a MONOMORPHIZED method of a generic class
		# (list[i32].__del__, ...) - Monomorphizer.monomorphized_function
		# sets a substituted method's own .cls to the concrete class
		# Specialization it was built for (base.cls stays plain/None for
		# an ordinary, non-generic method) - see its own comment on why.
		# Used by compiler.incref/decref to tell "T turned out non-RC this
		# instantiation" (routine, no-op) apart from "this was never RC to
		# begin with" (a real mistake, still rejected) - see their own
		# docstrings
		cls = getattr( self._current_fn, 'cls', None )
		return isinstance( cls, Specialization )

	def _lower_compiler_decref( self, node: ast.Call ) -> None:
		# compiler.decref(x) — emit an ir.Decref for x. Used inside
		# synthesized destructor bodies to tear down each RC field, and by
		# generic containers (list[T]) that need to conditionally RC-manage
		# elements whose T may or may not turn out to be an RC type once
		# monomorphized - a silent no-op for a non-RC T (rather than a hard
		# failure) is allowed ONLY inside a monomorphized generic-class
		# method (_in_generic_class_method), so the SAME generic method
		# body stays correct for both list[SomeRCClass] and list[i32]
		# without the class itself branching on whether T is RC - an
		# ordinary, non-generic call site with a genuinely wrong (always
		# non-RC) argument is still rejected, same as before
		if len( node.args ) != 1 or node.keywords:
			self.lowering.discovery.fail( f'compiler.decref(...) takes exactly one argument: {ast.unparse(node)}', node )
		operand = self._lower_expr( node.args[0], None )
		if operand.type is not None and self.lowering._type_resolver._is_RC( operand.type ):
			self._emit( ir.Decref( value = operand ))
			# stop the scope-exit epilogue from decref'ing operand a SECOND
			# time - see cfg.py's manually_decreffed's own comment for why
			# this is required, not optional (a real, always-on double
			# Decref/use-after-free otherwise, confirmed with ASan)
			self._cfg.manually_decreffed( operand )
			return
		if operand.type is not None and self._in_generic_class_method():
			return
		self.lowering.discovery.fail(
			f'compiler.decref(...) argument must be a reference-counted value, not '
			f'{operand.type.qualname if operand.type else "?"}: {ast.unparse(node)}',
			node,
		)

	def _lower_compiler_incref( self, node: ast.Call ) -> None:
		# compiler.incref(x) — emit an ir.Incref for x. Same conditional
		# no-op-for-non-RC-T posture as _lower_compiler_decref above.
		if len( node.args ) != 1 or node.keywords:
			self.lowering.discovery.fail( f'compiler.incref(...) takes exactly one argument: {ast.unparse(node)}', node )
		operand = self._lower_expr( node.args[0], None )
		if operand.type is not None and self.lowering._type_resolver._is_RC( operand.type ):
			self._emit( ir.Incref( value = operand ))
			return
		if operand.type is not None and self._in_generic_class_method():
			return
		self.lowering.discovery.fail(
			f'compiler.incref(...) argument must be a reference-counted value, not '
			f'{operand.type.qualname if operand.type else "?"}: {ast.unparse(node)}',
			node,
		)

	def _lower_compiler_decref_dynamic( self, node: ast.Call ) -> None:
		# compiler.decref_dynamic(ptr) - releases a TYPE-ERASED Ptr[None]
		# generically, via release_object, which reads its destructor off
		# the object's own header (see emitter_c.py's ObjectHeader) instead
		# of requiring the concrete type statically, unlike compiler.decref
		# above. Internal machinery for compiler-synthesized code (a
		# closure's own __del__, releasing its captured receiver after
		# type erasure) - not meant for ordinary user code, which always
		# has a real static type and should use compiler.decref instead
		if len( node.args ) != 1 or node.keywords:
			self.lowering.discovery.fail( f'compiler.decref_dynamic(...) takes exactly one argument: {ast.unparse(node)}', node )
		operand = self._lower_expr( node.args[0], None )
		none_type = self.lowering.discovery.get_none_type()
		ptr_cls = self.lowering.discovery.get_intrinsics()['Ptr']
		ptr_none_type = self.lowering.discovery._get_or_create_specialization( ptr_cls, [ none_type ] )
		if operand.type is not ptr_none_type:
			self.lowering.discovery.fail(
				f'compiler.decref_dynamic(...) argument must be Ptr[None], not '
				f'{operand.type.qualname if operand.type else "?"}: {ast.unparse(node)}',
				node,
			)
		self._emit( ir.DecrefDynamic( value = operand ))

	def _register_defer_block( self, is_err_only: bool, body: list[ast.stmt], node: ast.AST ) -> None:
		kind = 'errdefer' if is_err_only else 'defer'
		if self._loop_depth > 0:
			self.lowering.discovery.fail( f'{kind} is not allowed inside a loop - call another function and {kind} inside that instead', node )
		if self._in_deferred_body:
			self.lowering.discovery.fail( f'{kind} cannot be nested inside another defer/errdefer', node )

		fn = self._current_fn
		if is_err_only:
			return_type = fn.return_type if fn is not None else None
			ok = fn is not None and self.lowering._type_resolver._result_shape( return_type ) is not None
			if not ok:
				where = f'{fn.qualname} returns {return_type.qualname if return_type else None}' if fn is not None else 'this is not inside a function'
				self.lowering.discovery.fail( f'errdefer requires the enclosing function to return Result[_,_] ({where})', node )

		bool_cls = self.lowering.discovery.find_name( 'bool', node )
		index = len( self._defer_flags )
		flag = Variable(
			stem = f'__defer_flag_{index}',
			qualname = f'{fn.qualname}.__defer_flag_{index}',
			file = fn.file,
			line = getattr( node, 'lineno', None ),
			type = bool_cls,
		)

		# capture the body's instructions instead of emitting them inline -
		# they run later, in the epilogue, not at the with-statement's own
		# position. Lowering happens here, once, right now (not re-lowered at
		# replay time) so identifier resolution and dependency scheduling only
		# ever happen once, same as any other statement
		outer_instructions = self._instructions
		outer_in_deferred_body = self._in_deferred_body
		self._instructions = []
		self._in_deferred_body = True
		try:
			for stmt in body:
				try:
					self._lower_stmt( stmt )
				except CompileError:
					continue
			captured = self._instructions
		finally:
			self._instructions = outer_instructions
			self._in_deferred_body = outer_in_deferred_body

		self._defer_flags.append( flag )
		self._cfg.push_defer( captured, flag, is_err_only )
		# this is what actually runs at the with-statement's/call's position -
		# marks the block "armed" so the epilogue knows to replay it
		self._emit( ir.Assign( dest = flag, src = ir.Const( type = bool_cls, value = True )))

	# --- loops ---------------------------------------------------------------

	def _stmt_While( self, node: ast.While ) -> None:
		if node.orelse:
			self.lowering.discovery.fail( 'while/else is not supported', node )
		bool_cls = self.lowering.discovery.find_name( 'bool', node )
		start_label = self._new_label( 'while_start' )
		end_label = self._new_label( 'while_end' )
		# the test is positioned right after start_label (re-lowered here
		# once, but the resulting instructions physically sit inside the
		# repeated block, same as _stmt_If's test) so it's genuinely
		# re-evaluated every time the bottom Jump loops back
		self._emit( ir.Label( name = start_label ))
		test = self._lower_expr( node.test, bool_cls )
		self._emit( ir.JumpIfFalse( cond = test, target = end_label ))
		loop_snapshot = self._cfg.snapshot()
		break_narrowed = self._lower_loop_body( node.body, continue_label = start_label, break_label = end_label, loop_snapshot = loop_snapshot )
		try:
			back_edge_instructions = self._cfg.loop_back_edge( loop_snapshot.bindings, self._current_fn.qualname, entry_results = loop_snapshot.results )
		except CompileError as e:
			self.lowering.discovery.fail( str( e ), node )
		for instr in back_edge_instructions:
			self._emit( instr )
		self._cfg.restore( loop_snapshot )
		# Phase 7/8: reconcile every way execution can actually reach
		# end_label - the loop's own natural (condition-false) exit, PLUS
		# every break_narrowed record_break_narrowed() collected while
		# lowering the body above. `while True:` (test still the literal
		# Constant(True) it started as - type_resolver.py's visit_While
		# only ever rewrites it away from that shape when it recognizes a
		# real type(x) is T/is not T condition, never for a bare `True`)
		# has NO real condition-false exit at all - passing None here (not
		# a real candidate) is what keeps merge_loop_exits from wrongly
		# treating that unreachable path as competing with, and
		# suppressing, narrowing that only survives via an explicit break.
		# For an ordinary loop, type_resolver.py's own exit_narrows_name/
		# exit_narrows_member_stem (Phase 7 - the ONLY way to reach
		# end_label is via the condition going false, which for a 2-member
		# union, or unconditionally for the `is not` form, uniquely proves
		# x's own type from here on) overlays loop_snapshot's own narrowed
		# state to build the natural-exit candidate.
		is_while_true = isinstance( node.test, ast.Constant ) and node.test.value is True
		natural_exit_narrowed: dict[str,list[Variable]] | None = None
		if not is_while_true:
			natural_exit_narrowed = dict( loop_snapshot.narrowed )
			exit_name = getattr( node, 'exit_narrows_name', None )
			if exit_name is not None:
				member = self._resolve_narrow_member( exit_name, node.exit_narrows_member_stem, node )
				natural_exit_narrowed[exit_name] = [ member ]
		self._cfg.merge_loop_exits( natural_exit_narrowed, break_narrowed )
		self._emit( ir.Jump( target = start_label ))
		self._emit( ir.Label( name = end_label ))

	def _lower_loop_body( self, body: list[ast.stmt], continue_label: str, break_label: str, loop_snapshot: object ) -> list[dict[str,list[Variable]]]:
		self._loop_depth += 1
		self._loop_labels.append(( continue_label, break_label, loop_snapshot ))
		# see cfg.py's CFGState.enter_loop's own docstring: lets
		# current_epilogue_label() recognize an RC entry pushed while
		# lowering THIS body as loop-confined (restore(), called once this
		# body's fully lowered, silently drops it - its own label, if a
		# `return` from inside here ever pointed at it, would never
		# actually get emitted)
		self._cfg.enter_loop( loop_snapshot.stack_depth )
		break_narrowed: list[dict[str,list[Variable]]] = []
		try:
			for stmt in body:
				try:
					self._lower_stmt( stmt )
				except CompileError:
					continue
		finally:
			# Phase 8: every narrowed-state snapshot recorded by a `break`
			# reached while lowering this body (cfg.py's own
			# record_break_narrowed(), called from _stmt_Break below) -
			# handed back to the caller (_stmt_While/for-loop lowerers) to
			# merge with the loop's own natural exit via merge_loop_exits()
			break_narrowed = self._cfg.exit_loop()
			self._loop_labels.pop()
			self._loop_depth -= 1
		return break_narrowed

	def _check_loop_exit_unchecked_results( self, loop_snapshot: object, node: ast.AST ) -> None:
		try:
			self._cfg.check_loop_exit_unchecked_results( loop_snapshot.results, self._current_fn.qualname )
		except CompileError as e:
			self.lowering.discovery.fail( str( e ), node )

	def _stmt_Break( self, node: ast.Break ) -> None:
		if not self._loop_labels:
			self.lowering.discovery.fail( 'break outside a loop', node )
		_, break_label, loop_snapshot = self._loop_labels[-1]
		self._check_loop_exit_unchecked_results( loop_snapshot, node )
		# Phase 8: capture whatever's narrowed RIGHT HERE, at the exact
		# point this break fires - cfg.py's record_break_narrowed() files
		# it under the innermost currently-lowering loop, to be merged
		# with every other break/the loop's own natural exit once that
		# loop's own body is fully lowered. Before unwind_to() below (which
		# doesn't touch _narrowed at all, but ordering it first here keeps
		# this call sitting right next to unwind_to()'s own snapshot read)
		self._cfg.record_break_narrowed()
		for instr in self._cfg.unwind_to( loop_snapshot ):
			self._emit( instr )
		self._emit( ir.Jump( target = break_label ))

	def _stmt_Continue( self, node: ast.Continue ) -> None:
		if not self._loop_labels:
			self.lowering.discovery.fail( 'continue outside a loop', node )
		continue_label, _, loop_snapshot = self._loop_labels[-1]
		self._check_loop_exit_unchecked_results( loop_snapshot, node )
		for instr in self._cfg.unwind_to( loop_snapshot ):
			self._emit( instr )
		self._emit( ir.Jump( target = continue_label ))

	def _declare_hidden_local( self, stem: str, type: Type, node: ast.AST ) -> Variable:
		# compiler-synthesized locals (for-loop scaffolding: the once-
		# evaluated iterable, its length, the hidden index counter) - real
		# named Variables (not anonymous Temps) registered into the
		# function's flat names dict, the same way `self` gets synthesized
		# in lower_function, so synthetic ast.Name references to them
		# resolve normally through the existing _expr_Name/_stmt_Assign
		# machinery instead of duplicating it
		fn = self._current_fn
		var = Variable( stem = stem, qualname = f'{fn.qualname}.{stem}', file = fn.file, line = getattr( node, 'lineno', None ), type = type )
		fn.add_name( stem, var )
		self.lowering.schedule( type )
		return var

	def _maybe_consume_result( self, node: ast.AST, value: ir.Temp, alternatives: str ) -> ir.Operand:
		# if `value` is itself a Result[T,E], auto-consume it via the same
		# OrReturn/OrJump propagation or_return()/checked arithmetic use -
		# unlike _lower_or_return, a non-Result value is passed through
		# unchanged rather than rejected, since not every method this is
		# used for (__getitem__, __len__) is necessarily fallible. Uses
		# find_name_or_none (not find_name) - unlike every other Result
		# lookup in this file, this one runs speculatively for ANY value,
		# so a program that never defines Result at all (or hasn't
		# imported builtins) must not hard-fail here just because this
		# particular value happens not to be Result-shaped
		shape = self.lowering._type_resolver._result_shape( value.type )
		if shape is None:
			return value
		result_type, error_cls = shape
		# find_name, not value.type.base - value.type may already be the
		# real, monomorphized Result object itself (not a Specialization
		# wrapper) by the time _result_shape above succeeds - see
		# Monomorphizer.origin_of's own docstring. Safe to use the raising
		# lookup here (unlike _result_shape's own find_name_or_none) since
		# shape being non-None already proves Result is defined
		result_cls = self.lowering.discovery.find_name( 'Result', node )
		self.lowering._type_resolver._require_result_return( node, result_cls, error_cls, alternatives, fn = self._current_fn )
		return self._consume_checked_result( node, value, result_type, extra = None )

	def _bind_loop_target( self, target: ast.Name, default_type: Type, value_expr: ast.expr, node: ast.AST ) -> Variable:
		# mirrors _stmt_Assign's Name-target "reuse existing, else infer/
		# declare" rule (`for i in range(count):` reuses `i` if a variable
		# of that name already exists - e.g. str.concat in lib/builtins/
		# __init__.py pre-declares `i: usize = 0` before its own for loop)
		# - except a fresh declaration falls back to `default_type` instead
		# of failing outright, since value_expr may be a bare literal
		# (range()'s implicit start=0) with no type of its own to infer from
		existing = self.lowering.discovery.find_name_or_none( target.id )
		if existing is not None and not isinstance( existing, Variable ):
			self.lowering.discovery.fail( f'{target.id!r} is not a variable, cannot use it as a for loop target', node )
		expected = existing.type if existing is not None else default_type
		operand = self._lower_expr( value_expr, expected )
		if existing is not None:
			self._emit( ir.Assign( dest = existing, src = operand ))
			return existing
		fn = self._current_fn
		var = Variable( stem = target.id, qualname = f'{fn.qualname}.{target.id}', file = fn.file, line = getattr( node, 'lineno', None ), type = operand.type )
		fn.add_name( var.stem, var )
		self.lowering.schedule( var.type )
		self._emit( ir.Assign( dest = var, src = operand ))
		return var

	def _stmt_For( self, node: ast.For ) -> None:
		if not isinstance( node.target, ast.Name ):
			self.lowering.discovery.fail( f'for loop target must be a plain name: {ast.unparse(node)}', node )
		if node.orelse:
			self.lowering.discovery.fail( 'for/else is not supported', node )
		if self.lowering._is_range_call( node.iter ) is not None:
			self._lower_for_range( node )
		else:
			self._lower_for_over_indexable( node )

	def _lower_for_range( self, node: ast.For ) -> None:
		call = node.iter
		if call.keywords:
			self.lowering.discovery.fail( f'range(...) does not support keyword arguments: {ast.unparse(call)}', call )
		if len( call.args ) == 1:
			start_expr = ast.Constant( value = 0 )
			ast.copy_location( start_expr, call )
			stop_expr = call.args[0]
		elif len( call.args ) == 2:
			start_expr, stop_expr = call.args
		else:
			self.lowering.discovery.fail( f'range(...) supports 1 or 2 arguments only (no step yet): {ast.unparse(call)}', call )

		usize_cls = self.lowering.discovery.get_intrinsics()['usize']
		bool_cls = self.lowering.discovery.find_name( 'bool', node )

		target_var = self._bind_loop_target( node.target, usize_cls, start_expr, node )

		stop_operand = self._lower_expr( stop_expr, usize_cls )
		stop_var = self._declare_hidden_local( f'__for_stop_{self._label_id}', usize_cls, node )
		self._emit( ir.Assign( dest = stop_var, src = stop_operand ))

		start_label = self._new_label( 'for_start' )
		continue_label = self._new_label( 'for_continue' )
		end_label = self._new_label( 'for_end' )

		self._emit( ir.Label( name = start_label ))
		test = ast.Compare( left = self.lowering._synth_name( target_var.stem, node ), ops = [ ast.Lt() ], comparators = [ self.lowering._synth_name( stop_var.stem, node ) ] )
		ast.copy_location( test, node )
		cond = self._lower_expr( test, bool_cls )
		self._emit( ir.JumpIfFalse( cond = cond, target = end_label ))

		loop_snapshot = self._cfg.snapshot()
		break_narrowed = self._lower_loop_body( node.body, continue_label = continue_label, break_label = end_label, loop_snapshot = loop_snapshot )
		try:
			back_edge_instructions = self._cfg.loop_back_edge( loop_snapshot.bindings, self._current_fn.qualname, entry_results = loop_snapshot.results )
		except CompileError as e:
			self.lowering.discovery.fail( str( e ), node )
		for instr in back_edge_instructions:
			self._emit( instr )
		self._cfg.restore( loop_snapshot )
		# Phase 8: a for-loop has no type(x) is T condition of its own to
		# narrow FROM, but its natural exit (the range simply exhausted,
		# including never having run the body at all - always reachable
		# for any for-loop) is still a real candidate to reconcile against
		# every break_narrowed collected above - same merge_loop_exits()
		# used by _stmt_While
		self._cfg.merge_loop_exits( dict( loop_snapshot.narrowed ), break_narrowed )

		self._emit( ir.Label( name = continue_label ))
		# the increment is a compiler-synthesized implementation detail of
		# the loop, not user-written arithmetic - it's structurally
		# guaranteed safe (target_var < stop_var strictly before every
		# increment), so it bypasses the ambient arithmetic-mode policy
		# entirely (AddWrap directly) rather than imposing a
		# Result[_,OverflowError]/wrap_arithmetic/etc. requirement on
		# ordinary for-loops
		incr = self._new_temp( usize_cls )
		self._emit( ir.AddWrap( dest = incr, left = target_var, right = ir.Const( type = usize_cls, value = 1 ) ))
		self._emit( ir.Assign( dest = target_var, src = incr ))
		self._emit( ir.Jump( target = start_label ))
		self._emit( ir.Label( name = end_label ))

	def _lower_for_over_indexable( self, node: ast.For ) -> None:
		usize_cls = self.lowering.discovery.get_intrinsics()['usize']
		bool_cls = self.lowering.discovery.find_name( 'bool', node )

		obj = self._lower_expr( node.iter, None )
		len_fn = self.lowering._find_method( obj.type, '__len__' )
		getitem_fn = self.lowering._find_method( obj.type, '__getitem__' )
		missing = [ name for name, fn in (( '__len__', len_fn ), ( '__getitem__', getitem_fn )) if fn is None ]
		if missing:
			self.lowering.discovery.fail( f'for loop needs {" and ".join(missing)} on {obj.type.qualname if obj.type else "?"}: {ast.unparse(node)}', node )

		unique = self._label_id
		obj_var = self._declare_hidden_local( f'__for_obj_{unique}', obj.type, node )
		self._emit( ir.Assign( dest = obj_var, src = obj ))

		self.lowering._ensure_resolved( len_fn )
		self.lowering.schedule( len_fn.return_type )
		len_dest = self._new_temp( len_fn.return_type )
		self._emit( ir.Call( dest = len_dest, target = len_fn, receiver = obj_var, args = [], kwargs = {} ))
		len_operand = self._maybe_consume_result( node, len_dest, self.lowering._FOR_LOOP_ALTERNATIVES )
		len_var = self._declare_hidden_local( f'__for_len_{unique}', len_operand.type, node )
		self._emit( ir.Assign( dest = len_var, src = len_operand ))

		index_var = self._declare_hidden_local( f'__for_index_{unique}', usize_cls, node )
		self._emit( ir.Assign( dest = index_var, src = ir.Const( type = usize_cls, value = 0 ) ))

		start_label = self._new_label( 'for_start' )
		continue_label = self._new_label( 'for_continue' )
		end_label = self._new_label( 'for_end' )

		self._emit( ir.Label( name = start_label ))
		test = ast.Compare( left = self.lowering._synth_name( index_var.stem, node ), ops = [ ast.Lt() ], comparators = [ self.lowering._synth_name( len_var.stem, node ) ] )
		ast.copy_location( test, node )
		cond = self._lower_expr( test, bool_cls )
		self._emit( ir.JumpIfFalse( cond = cond, target = end_label ))

		# the snapshot is taken here, BEFORE the loop target's own binding -
		# that binding (e.g. `s2 = obj[index]`) happens fresh every
		# iteration, exactly like any other loop-body statement (matches
		# foo4: a value reassigned each iteration is expected to be stable
		# across the back edge, not confined-and-torn-down)
		loop_snapshot = self._cfg.snapshot()
		subscript = ast.Subscript(
			value = self.lowering._synth_name( obj_var.stem, node ),
			slice = self.lowering._synth_name( index_var.stem, node ),
			ctx = ast.Load(),
		)
		ast.copy_location( subscript, node )
		bind = ast.Assign( targets = [ node.target ], value = subscript )
		ast.copy_location( bind, node )
		self._stmt_Assign( bind )

		break_narrowed = self._lower_loop_body( node.body, continue_label = continue_label, break_label = end_label, loop_snapshot = loop_snapshot )
		try:
			back_edge_instructions = self._cfg.loop_back_edge( loop_snapshot.bindings, self._current_fn.qualname, entry_results = loop_snapshot.results )
		except CompileError as e:
			self.lowering.discovery.fail( str( e ), node )
		for instr in back_edge_instructions:
			self._emit( instr )
		self._cfg.restore( loop_snapshot )
		# Phase 8 - see _lower_for_range's own identical call/comment
		self._cfg.merge_loop_exits( dict( loop_snapshot.narrowed ), break_narrowed )

		self._emit( ir.Label( name = continue_label ))
		incr = self._new_temp( usize_cls )
		self._emit( ir.AddWrap( dest = incr, left = index_var, right = ir.Const( type = usize_cls, value = 1 ) ))
		self._emit( ir.Assign( dest = index_var, src = incr ))
		self._emit( ir.Jump( target = start_label ))
		self._emit( ir.Label( name = end_label ))

	def _stmt_If( self, node: ast.If ) -> None:
		bool_cls = self.lowering.discovery.find_name( 'bool', node )
		test = self._lower_expr( node.test, bool_cls )
		else_label = self._new_label( 'if_else' )
		self._emit( ir.JumpIfFalse( cond = test, target = else_label ))

		# each branch is lowered into its OWN captured instruction list
		# (same technique _register_defer_block already uses) rather than
		# appended directly - a local confined to just one branch needs its
		# own Decref spliced into THAT branch's own code specifically
		# (before its own exit to the join point), never at the shared
		# join point both branches reach, since it only exists on that one
		# path. Known ahead of time only after BOTH branches have been
		# explored (merge_if, below), so neither branch's own instructions
		# can be emitted directly as they're lowered
		entry_snapshot = self._cfg.snapshot()
		outer_instructions = self._instructions
		self._instructions = []
		# see cfg.py's CFGState.enter_branch's own docstring: lets
		# current_epilogue_label() recognize an RC entry pushed while
		# lowering THIS branch (e.g. a match arm's own payload binding) as
		# branch-confined - restore(), called once this branch's fully
		# lowered, silently drops it, so a `return` inside here must never
		# be handed that entry's own label as a shared jump target
		self._cfg.enter_branch( entry_snapshot.stack_depth )
		try:
			for stmt in node.body:
				try:
					self._lower_stmt( stmt )
				except CompileError:
					continue
		finally:
			self._cfg.exit_branch()
		true_captured = self._instructions
		true_end = dict( self._cfg.bindings )
		true_end_results = self._cfg.unchecked_results()
		true_end_narrowed = self._cfg.narrowed_snapshot()
		# return/break/continue as a branch's own last statement means
		# that branch never reaches the if's join point at all - see
		# merge_if()'s own comment on why that has to be treated
		# differently from an ordinary falling-through branch (full
		# terminator/dead-code analysis for anything deeper - nested ifs
		# that both terminate, etc - is future work, not attempted here)
		true_terminates = bool( node.body ) and isinstance( node.body[-1], ( ast.Return, ast.Break, ast.Continue ))

		if node.orelse:
			self._cfg.restore( entry_snapshot )
			self._instructions = []
			self._cfg.enter_branch( entry_snapshot.stack_depth )
			try:
				for stmt in node.orelse:
					try:
						self._lower_stmt( stmt )
					except CompileError:
						continue
			finally:
				self._cfg.exit_branch()
			false_captured = self._instructions
			false_end = dict( self._cfg.bindings )
			false_end_results = self._cfg.unchecked_results()
			false_end_narrowed = self._cfg.narrowed_snapshot()
			false_terminates = bool( node.orelse ) and isinstance( node.orelse[-1], ( ast.Return, ast.Break, ast.Continue ))
		else:
			false_captured = []
			false_end = dict( entry_snapshot.bindings )
			false_end_results = set( entry_snapshot.results )
			false_end_narrowed = dict( entry_snapshot.narrowed )
			false_terminates = False

		self._cfg.restore( entry_snapshot )
		self._instructions = outer_instructions
		try:
			true_extra, false_extra, removed = self._cfg.merge_if(
				entry_snapshot.bindings, true_end, false_end, self._current_fn.qualname,
				entry_results = entry_snapshot.results, true_end_results = true_end_results, false_end_results = false_end_results,
				true_terminates = true_terminates, false_terminates = false_terminates,
				true_end_narrowed = true_end_narrowed, false_end_narrowed = false_end_narrowed,
			)
		except CompileError as e:
			self.lowering.discovery.fail( str( e ), node )
		for name in removed:
			del self._current_fn.names[name]

		for instr in true_captured:
			self._emit( instr )
		for instr in true_extra:
			self._emit( instr )
		if node.orelse:
			end_label = self._new_label( 'if_end' )
			self._emit( ir.Jump( target = end_label ))
			self._emit( ir.Label( name = else_label ))
			for instr in false_captured:
				self._emit( instr )
			for instr in false_extra:
				self._emit( instr )
			self._emit( ir.Label( name = end_label ))
		else:
			self._emit( ir.Label( name = else_label ))

	# --- expressions -----------------------------------------------------------

	def _lower_expr( self, node: ast.expr, expected_type: Type|None ) -> ir.Operand:
		method = getattr( self, f'_expr_{node.__class__.__name__}', None )
		if method is None:
			self.lowering.discovery.fail( f'unsupported expression: {ast.unparse(node)}', node )
		operand = method( node, expected_type )
		# post-hoc, not a pre-emptive override of expected_type before
		# dispatch: a node kind that already produces the right union type
		# on its own (an explicit Result.Ok(x) call, a match-narrowed
		# union-typed value, ...) already satisfies operand.type is
		# expected_type and skips this entirely - only a genuine mismatch
		# (a plain leaf value where the union itself was expected) reaches
		# _coerce_into_union, see its own comment
		if isinstance( expected_type, TaggedUnion ) and operand.type is not expected_type:
			operand = self._coerce_into_union( operand, expected_type, node )
		return operand

	def _coerce_into_union( self, operand: ir.Operand, union: TaggedUnion, node: ast.AST ) -> ir.Operand:
		# operand's own type doesn't match the union it needs to become -
		# TODO.txt's own documented "opportunistic union emission" gap
		# (a: IntStr = 'foo' should become a = IntStr.v_str('foo')). If
		# operand.type is exactly one of the union's own leaves, wrap it
		# through that leaf's own UnionStorage-synthesized member
		# constructor - the SAME Function a real, explicit Result.Ok(x)
		# call already resolves to and calls via ordinary call resolution;
		# this just does that implicitly. A genuine mismatch (operand's
		# type isn't a member of the union at all) is a real compile
		# error, not silently passed through.
		self.lowering._union_storage.get( union ) # ensures union.names[leaf.stem] exists
		leaf = next( ( attr for attr in union.attributes if attr.type is operand.type ), None )
		if leaf is None:
			self.lowering.discovery.fail(
				f'{ast.unparse(node)}: expected {union.qualname}, got a type that is not one of its members',
				node,
			)
		ctor_fn = union.names[leaf.stem]
		self.lowering.schedule( ctor_fn )
		self.lowering.schedule( ctor_fn.return_type )
		for p in ctor_fn.parameters or []:
			self.lowering.schedule( p.type )
		dest = self._new_temp( union )
		self._emit( ir.Call( dest = dest, target = ctor_fn, receiver = None, args = [ operand ], kwargs = {} ))
		return dest

	def _expr_Name( self, node: ast.Name, expected_type: Type|None ) -> ir.Operand:
		name = self.lowering.discovery.find_name( node.id, node )
		if isinstance( name, Function ):
			return self._lower_function_ref( name, node )
		if not isinstance( name, Variable ):
			self.lowering.discovery.fail( f'{node.id!r} is not a value, cannot use it as an expression', node )
		self.lowering._ensure_resolved( name )
		member = self._cfg.narrowed_member( node.id )
		if member is not None and expected_type is not name.type:
			# name is currently proven to hold this union member (cfg.py's
			# narrow(), from a `match x: case T(x):` arm reusing x's own
			# name) - read through the union's own payload instead of
			# returning the raw (still union-typed) operand, unless the
			# caller explicitly wants the whole union back (expected_type
			# is name.type exactly - a rare escape hatch, e.g. passing x
			# through to another T|None-typed parameter unchanged). Same
			# GetAttr(data).GetAttr(v_member) shape cfg.py's own
			# _extract_payload/lowering.py's _maybe_unwrap_union_arg
			# already use elsewhere for the identical operation - a pure
			# read, no incref needed here: name itself still owns the
			# whole union unconditionally the entire time, this is just
			# viewing one field of it (same as any other attribute read);
			# only actually ALIASING this returned operand into a NEW
			# binding needs its own incref, and that already happens the
			# ordinary way wherever this operand is next consumed. Same
			# Specialization gap as _stmt_Assign's own narrowing-bind
			# handling above: a parameter's declared type stays a genuine
			# Specialization (T/E still bare TypeVars) - .base alone would
			# hand _union_storage.get() the ABSTRACT, unsubstituted payload
			# shape instead of the real monomorphized one.
			base = self.lowering.monomorphize_class( name.type ) if isinstance( name.type, Specialization ) else name.type
			_tag_attr, data_attr, payload_cls, _tags = self.lowering._union_storage.get( base )
			payload_dest = self._new_temp( payload_cls )
			self._emit( ir.GetAttr( dest = payload_dest, obj = name, attr = data_attr.stem ))
			leaf_dest = self._new_temp( member.type )
			self._emit( ir.GetAttr( dest = leaf_dest, obj = payload_dest, attr = f'v_{member.stem}' ))
			return leaf_dest
		# when a pointer-typed local flows into a context expecting a
		# differently-typed pointer (e.g. return ptr where ptr: Ptr[u8]
		# but the function returns Ptr[T]), insert a CastWrap — in C all
		# object pointers have the same representation, so this is safe
		if expected_type is not None and name.type is not expected_type and self.lowering._type_resolver._is_ptr_specialization( name.type ) and self.lowering._type_resolver._is_ptr_specialization( expected_type ):
			dest = self._new_temp( expected_type )
			self._emit( ir.CastWrap( dest = dest, operand = name ))
			return dest
		return name

	def _reject_free_variables( self, roots: list[ast.AST], param_names: set[str], node: ast.AST ) -> None:
		# a nested def/lambda may only reference its own parameters/locally
		# -assigned names, module-level names, and builtins - referencing
		# anything from the immediately enclosing function's own scope is a
		# capture, deliberately unsupported for now (see PLAN_LAMBDA.md's
		# own "deferred" list - no representation decision made yet for a
		# captured environment). `roots` is the def's own body (a list of
		# statements) or a lambda's own body wrapped in a single-element
		# list (a bare expression - lambda syntax forbids assignment
		# statements, but NOT ast.NamedExpr/walrus, which also binds via
		# Name(Store) - the same walk covers both shapes uniformly without
		# special-casing). Doesn't recurse into a FURTHER nested def/
		# lambda's own body - that one gets its own independent check when
		# IT gets synthesized (only its OWN name, if it's a def, becomes a
		# local binding at THIS level, same as an ordinary assignment would)
		local_names = set( param_names )
		class _BindingCollector( ast.NodeVisitor ):
			def visit_FunctionDef( self, fd: ast.FunctionDef ) -> None:
				local_names.add( fd.name )
			def visit_Lambda( self, lam: ast.Lambda ) -> None:
				pass
			def visit_Name( self, n: ast.Name ) -> None:
				if isinstance( n.ctx, ast.Store ):
					local_names.add( n.id )
		binder = _BindingCollector()
		for root in roots:
			binder.visit( root )

		free: list[ast.Name] = []
		class _LoadCollector( ast.NodeVisitor ):
			def visit_FunctionDef( self, fd: ast.FunctionDef ) -> None:
				pass
			def visit_Lambda( self, lam: ast.Lambda ) -> None:
				pass
			def visit_Name( self, n: ast.Name ) -> None:
				if isinstance( n.ctx, ast.Load ) and n.id not in local_names:
					free.append( n )
		loader = _LoadCollector()
		for root in roots:
			loader.visit( root )

		enclosing_fn = self._current_fn
		if enclosing_fn is None:
			return
		for free_name in free:
			if free_name.id in enclosing_fn.names:
				self.lowering.discovery.fail(
					f"{ast.unparse(node)}: captures {free_name.id!r} from the enclosing function - nested "
					f"functions/lambdas can only reference their own parameters, module-level names, and "
					f"builtins (no captured variables yet)",
					node,
				)

	def _lower_function_ref( self, fn: Function, node: ast.AST ) -> ir.Operand:
		# a bare reference to a function used AS A VALUE, not called - see
		# PLAN_CALLABLE.md. Only a plain, receiver-less, non-generic,
		# non-overloaded function can become a Ptr[Callable[...]] value:
		# a bound instance method or classmethod has an implicit receiver
		# with nowhere to go in a raw C function pointer (that's a closure,
		# deliberately out of scope for now - see the plan doc's own
		# "deferred" list), a generic function has no single fixed
		# signature to point at (which specialization?), and an overload
		# group's own individual Functions are ambiguous by name alone
		if fn.type_params:
			self.lowering.discovery.fail( f'{fn.qualname} is generic - cannot take a bare reference to it: {ast.unparse(node)}', node )
		if fn.is_overload:
			self.lowering.discovery.fail( f'{fn.qualname} is one of several @overload implementations - cannot take a bare reference to it by name alone: {ast.unparse(node)}', node )
		if fn.cls is not None and not fn.is_static:
			self.lowering.discovery.fail( f'{fn.qualname} is an instance method or classmethod - cannot take a bare reference to it (no receiver to bind): {ast.unparse(node)}', node )
		if fn.cls is not None and fn.cls.type_params:
			# a @staticmethod belonging to a GENERIC class, referenced bare
			# from within another method of that SAME class - discovery.
			# find_name only ever returns the abstract template (fn.cls
			# itself, K/V still bare TypeVars: confirmed - _value_spelling
			# crashes on the unresolved TypeVar downstream in emitter_c.py
			# without this). Ordinary calls (self.method(...)) never hit
			# this because _lower_generic_call_with_receiver substitutes
			# the class type args and calls _monomorphized_function BEFORE
			# emitting the Call - a bare reference has no such call site to
			# hang that on, so it's done here instead, using the CURRENT
			# specialization being lowered (self._current_fn.cls) rather
			# than inferring from arguments (there's nothing to infer from
			# for a value reference - Ptr[Callable[...]]'s own shape
			# carries no class-type-param information at all)
			current_cls = self._current_fn.cls if self._current_fn is not None else None
			current_spec = current_cls if isinstance( current_cls, Specialization ) else None
			if current_spec is None or current_spec.base is not fn.cls:
				self.lowering.discovery.fail(
					f"{fn.qualname} belongs to a generic class - a bare reference to it is only resolvable from "
					f"inside one of {fn.cls.qualname}'s own (already-specialized) methods: {ast.unparse(node)}",
					node,
				)
			method_spec = self.lowering.discovery._get_or_create_specialization( fn, current_spec.args )
			fn = self.lowering._monomorphized_function( method_spec )
		self.lowering._ensure_resolved( fn )
		if fn.parameters is None:
			self.lowering.discovery.fail( f'{fn.qualname} could not be resolved (see earlier error): {ast.unparse(node)}', node )
		return self.lowering._function_ref_operand( fn )

	def _stmt_FunctionDef( self, node: ast.FunctionDef ) -> None:
		# a non-capturing nested function def - see PLAN_LAMBDA.md.
		# Synthesized as an independent, fully real Function (real
		# parameter/return annotations, resolved exactly like an ordinary
		# top-level function via discovery.visit(...) - the same mechanism
		# _stmt_AnnAssign already uses mid-lowering for an ordinary local's
		# own annotation), scheduled and compiled like any other compile
		# unit, and made callable/bare-referenceable by name for the rest
		# of the enclosing function's own body. The statement itself emits
		# no IR - a def only binds a name, same as Python
		enclosing = self._current_fn
		if enclosing is None:
			self.lowering.discovery.fail( f'nested function def outside any function: {ast.unparse(node)}', node )
		self.lowering._reject_generic_enclosing_scope( enclosing, node, 'nested function defs' )
		if node.decorator_list:
			self.lowering.discovery.fail( f'{node.name}: decorators are not supported on a nested function def: {ast.unparse(node)}', node )

		qualname = f'{enclosing.qualname}$$nested_{node.name}'
		synthetic = Function(
			stem = node.name, qualname = qualname,
			file = enclosing.file, line = node.lineno,
			cls = None, node = node,
			parameters = None, return_type = None,
			resolve = None,
		)

		args = node.args
		parameters: list[Parameter] = []
		def add_param( arg: ast.arg, default: ast.expr|None, **kind: bool ) -> None:
			if arg.annotation is None:
				self.lowering.discovery.fail( f'{qualname} parameter {arg.arg!r} has no type annotation: {ast.unparse(node)}', node )
			param_type = self.lowering.discovery.visit( arg.annotation )
			self.lowering.discovery._reject_bare_interface_value_type( param_type, arg, f'{qualname} parameter {arg.arg!r}' )
			param = Parameter(
				stem = arg.arg, qualname = f'{qualname}.{arg.arg}',
				file = enclosing.file, line = node.lineno,
				type = param_type, default = default, **kind,
			)
			parameters.append( param )
			synthetic.add_name( param.stem, param )

		# `defaults` applies to the trailing N of posonlyargs+args combined
		# (an ast-module quirk) - left-pad with None so every positional
		# param lines up with its own default (or lack of one) - same
		# convention discovery.py's own _make_function_resolver uses
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

		synthetic.parameters = parameters
		if node.returns is not None:
			synthetic.return_type = self.lowering.discovery.visit( node.returns )
			self.lowering.discovery._reject_bare_interface_value_type( synthetic.return_type, node.returns, f'{qualname} return type' )
		else:
			synthetic.return_type = self.lowering.discovery.get_none_type()

		self._reject_free_variables( node.body, { p.stem for p in parameters }, node )

		enclosing.add_name( node.name, synthetic )
		self.lowering.schedule( synthetic )

	def _expr_Lambda( self, node: ast.Lambda, expected_type: Type|None ) -> ir.Operand:
		# a non-capturing lambda expression - see PLAN_LAMBDA.md. Lambda
		# syntax carries no type annotations at all, so parameter types are
		# inferred entirely from expected_type (must already be a
		# Ptr[Callable[[ArgTypes],Ret]] shape flowing in from the
		# surrounding context - e.g. a `key: Callable[[T],K]` parameter's
		# own declared type, while lowering the argument expression at a
		# call site)
		fn_type = self.lowering._type_resolver._callable_type_of( expected_type )
		if fn_type is None:
			self.lowering.discovery.fail(
				f'cannot infer lambda parameter types - no expected Callable[...] context: {ast.unparse(node)}',
				node,
			)
		args = node.args
		if args.vararg is not None or args.kwarg is not None or args.kwonlyargs:
			self.lowering.discovery.fail( f'lambda does not support *args/**kwargs/keyword-only parameters yet: {ast.unparse(node)}', node )
		positional = [ *args.posonlyargs, *args.args ]
		if len( positional ) != len( fn_type.arg_types ):
			self.lowering.discovery.fail(
				f'lambda takes {len(positional)} argument(s), the expected Callable[...] type declares {len(fn_type.arg_types)}: {ast.unparse(node)}',
				node,
			)
		if any( isinstance( t, TypeVar ) for t in fn_type.arg_types ):
			# unlike the return type below, a lambda's own PARAMETER types
			# have no body to infer them from - they're needed up front just
			# to lower the body at all (see below), so an unbound arg type
			# here is unrecoverable, not just provisional
			self.lowering.discovery.fail(
				f'cannot infer lambda parameter types - Callable[...] parameter types are not fully concrete: {ast.unparse(node)}',
				node,
			)
		# a lambda passed to a generic function's own Callable[[T],K]-typed
		# parameter (e.g. bisect_right's key=) can have an unbound K here -
		# only knowable from the lambda's OWN body, once lowered (PLAN_LAMBDA.md,
		# "eager lambda lowering"/the circular-inference problem). return_type
		# stays a real Type|None field either way (Function.return_type is
		# already None for any not-yet-resolved function - same convention,
		# see FunctionLowering's own docstring), just resolved eagerly below
		# instead of by the ordinary ast-annotation route
		return_type_provisional = isinstance( fn_type.return_type, TypeVar )

		enclosing = self._current_fn
		if enclosing is None:
			self.lowering.discovery.fail( f'lambda outside any function: {ast.unparse(node)}', node )
		self.lowering._reject_generic_enclosing_scope( enclosing, node, 'lambdas' )

		self.lowering._lambda_counter += 1
		name = f'$$lambda_{self.lowering._lambda_counter}'
		qualname = f'{enclosing.qualname}{name}'
		synthetic_node = ast.FunctionDef(
			name = name,
			args = ast.arguments(
				posonlyargs = [], args = [ ast.arg( arg = p.arg, annotation = None ) for p in positional ],
				vararg = None, kwonlyargs = [], kw_defaults = [], kwarg = None, defaults = [],
			),
			body = [ ast.Return( value = node.body ) ],
			decorator_list = [], returns = None, type_params = [],
			lineno = node.lineno, col_offset = node.col_offset,
		)
		ast.fix_missing_locations( synthetic_node )

		synthetic = Function(
			stem = name, qualname = qualname,
			file = enclosing.file, line = node.lineno,
			cls = None, node = synthetic_node,
			parameters = None, return_type = None if return_type_provisional else fn_type.return_type,
			resolve = None,
		)
		parameters: list[Parameter] = []
		for arg_node, arg_type in zip( positional, fn_type.arg_types ):
			param = Parameter(
				stem = arg_node.arg, qualname = f'{qualname}.{arg_node.arg}',
				file = enclosing.file, line = node.lineno,
				type = arg_type,
			)
			parameters.append( param )
			synthetic.add_name( param.stem, param )
		synthetic.parameters = parameters

		self._reject_free_variables( [ node.body ], { p.arg for p in positional }, node )

		if return_type_provisional:
			# lower the body RIGHT NOW, synchronously, instead of only ever
			# scheduling it onto the work queue for later - the caller's own
			# generic type-parameter binding (_unify_type_param's CallableType
			# recursion) needs the REAL return type immediately, at this call
			# site, not whenever the queue happens to drain it. Reuses
			# Compiler._lower exactly (resolve_function_body + lower_function
			# + extern_libs bookkeeping + registering into compiler.functions)
			# via the _compile_now backreference threaded in Compiler.__init__ -
			# lower_function itself builds a brand-new FunctionLowering
			# instance for this nested call, so there's no shared mutable
			# state with the lowering already in progress for `enclosing` to
			# save/restore around at all
			lowered = self.lowering._compile_now( synthetic )
			return_instr = next( instr for instr in lowered.instructions if isinstance( instr, ir.Return ))
			synthetic.return_type = (
				return_instr.value.type if return_instr.value is not None
				else self.lowering.discovery.get_none_type()
			)
		else:
			self.lowering.schedule( synthetic )
		return self.lowering._function_ref_operand( synthetic )

	def _expr_Constant( self, node: ast.Constant, expected_type: Type|None ) -> ir.Operand:
		# expected_type being a TaggedUnion (e.g. str|None) is treated the
		# same as no expected_type at all: a literal's OWN Python type
		# always determines its natural type (bool/i32/str/NoneType) -
		# blindly typing the Const as the whole union here would be wrong
		# (a literal is never itself union-shaped at the C level), and
		# _lower_expr's own post-hoc coercion (see its comment) is what
		# actually wraps this natural-typed Const into the union afterward
		if expected_type is None or isinstance( expected_type, TaggedUnion ):
			if isinstance( node.value, bool ):
				expected_type = self.lowering.discovery.get_intrinsics()['bool']
			elif isinstance( node.value, int ):
				# integer literals default to i32 when no contextual type is
				# available (bare `x = 1`, generic-call arg inference, etc.)
				# TODO FIXME: for most user code, this should probably be builtins.int and get scheduled as an immortal constant
				expected_type = self.lowering.discovery.get_intrinsics()['i32']
			elif isinstance( node.value, str ):
				expected_type = self.lowering.discovery.find_name_or_none( 'str' )
				if expected_type is None:
					self.lowering.discovery.fail(
						f'cannot infer the type of literal {node.value!r} - no str type available ({ast.unparse(node)})',
						node,
					)
			elif node.value is None:
				expected_type = self.lowering.discovery.get_none_type()
			else:
				self.lowering.discovery.fail(
					f'cannot infer the type of literal {node.value!r} - no expected type available from context ({ast.unparse(node)})',
					node,
				)
		return ir.Const( type = expected_type, value = node.value )

	def _lower_fstring_part( self, node: 'ast.Constant|ast.FormattedValue', str_type: Type ) -> ir.Operand:
		# one element of an f-string's ast.JoinedStr.values - either a
		# literal text segment (ast.Constant, already merged by CPython's
		# own parser) or a {expr} interpolation (ast.FormattedValue).
		# Shared by _expr_JoinedStr's single-part short-circuit and its
		# N-part UnsafeList/slice/str.concat path below - both need the
		# same str-typed operand per element, just assembled differently
		# (PLAN_FSTRINGS.md).
		if isinstance( node, ast.Constant ):
			return self._lower_expr( node, str_type )
		# ast.FormattedValue
		if node.conversion == 97: # '!a' (ascii) - no ascii-escape primitive exists anywhere in this codebase
			self.lowering.discovery.fail( f'f-string !a (ascii) conversion is not supported: {ast.unparse(node)}', node )
		if node.format_spec is not None:
			self.lowering.discovery.fail(
				f'f-string format specs ({{expr:spec}}) are not supported yet - requires str.format() (see TODO.txt): {ast.unparse(node)}',
				node,
			)
		operand = self._lower_expr( node.value, None )
		if operand.type is str_type:
			return operand
		# conversion 114 == '!r'; -1 (none) and 115 ('!s') both want __str__ -
		# matches print()'s own existing "no implicit stringification"
		# convention: a scalar (i32, bool, ...) has no __str__ of its own,
		# only the boxed classes do (int.__str__) - deliberately not
		# auto-boxed here, same reasoning PLAN_FSTRINGS.md's own scope
		# section gives
		method_name = '__repr__' if node.conversion == 114 else '__str__'
		method = self.lowering._find_method( operand.type, method_name )
		if method is None:
			type_name = operand.type.qualname if operand.type is not None else '?'
			self.lowering.discovery.fail(
				f'f-string: {type_name} has no {method_name}() - cannot format {ast.unparse(node.value)} in an f-string',
				node,
			)
		self.lowering._ensure_resolved( method )
		self.lowering.schedule( method.return_type )
		for p in ( method.parameters or [] ):
			self.lowering.schedule( p.type )
		dest = self._new_temp( str_type )
		self._emit( ir.Call( dest = dest, target = method, receiver = operand, args = [], kwargs = {} ))
		return dest

	def _lower_unwrap_result(
		self, result: ir.Operand, errmsg: str, payload_type: Type, error_type: Type, str_type: Type, node: ast.AST, *, want_result: bool = True,
	) -> ir.Operand|None:
		# unwrap()s a Result[T,E] this pass itself just produced (an
		# UnsafeList[str].append()/.get_ptr() call, below). (payload_type,
		# error_type) are passed in explicitly by the caller rather than
		# read back off result.type, since substitute_type_params leaves
		# two visibly different shapes there depending on whether the
		# Result's own structure mentions T (get_ptr's Result[Ptr[T],
		# IndexError] arrives as an already-monomorphized TaggedUnion;
		# append's Result[None,OverflowError], fully concrete already in
		# the abstract declaration, stays a plain Specialization) - the
		# caller already knows both types unambiguously either way.
		#
		# Resolves the CONCRETE Result[payload_type,error_type] CLASS
		# first (_get_or_create_specialization + _ensure_resolved), then
		# reads `unwrap` off ITS OWN .names - the same "always go through
		# the concrete class, never build a method Specialization directly
		# off the abstract one" fix _expr_JoinedStr's own UnsafeList[str]
		# handling above already needed (see its own comment). Building
		# unwrap's Function-Specialization directly against the ABSTRACT
		# Result class (this method's first, abandoned implementation)
		# compiles and runs, but silently ALSO schedules a second, bogus,
		# unspecialized copy of Result.is_ok (called from unwrap's own
		# `if self.is_ok(): ...` body) under the bare, un-mangled C symbol
		# name - a real "conflicting types for 'builtins$Result$is_ok'"
		# link-shape error, confirmed via a real compile attempt and fixed
		# by going through the concrete class first instead, exactly like
		# ordinary source's own `some_result.unwrap(msg)` dispatch already
		# does (Lowering._find_method's own owner_type = self._ensure_
		# resolved(owner_type) is the same "resolve the class, not the
		# method" step). These Results are provably always Ok (the buffer
		# is pre-sized to exactly len(node.values) and never appended to
		# more than that many times, and index 0 is always valid once N >=
		# 2) - unwrap() rather than silently discarding keeps this
		# consistent with the rest of the language's own "a Result is
		# never silently ignored" discipline, and turns a violated
		# invariant into a clear panic instead of undefined behavior.
		result_cls = self.lowering.discovery.find_name( 'Result', node )
		result_spec = self.lowering.discovery._get_or_create_specialization( result_cls, [ payload_type, error_type ])
		concrete_result_cls = self.lowering._ensure_resolved( result_spec )
		unwrap = concrete_result_cls.names.get( 'unwrap' )
		self.lowering._ensure_resolved( unwrap ) # schedules unwrap itself as a compile unit - see _expr_JoinedStr's own identical comment on init/append/get_ptr
		self.lowering.schedule( unwrap.return_type )
		for p in ( unwrap.parameters or [] ):
			self.lowering.schedule( p.type )
		errmsg_const = ir.Const( type = str_type, value = errmsg )
		# want_result=False (append's own Result[None,OverflowError] - the
		# payload is never used for anything, the call is made purely for
		# its panic-on-Err side effect) discards the result rather than
		# storing a None-typed payload in a Temp - a real, narrow, pre-
		# existing emitter gap around a GENERIC Result[T,E].unwrap()
		# monomorphized with T=NoneType (confirmed via a real compile
		# attempt: the emitted unwrap[NoneType,...] function returns C
		# `void`, but a stored dest expects an assignable MetalpyNone
		# value - a mismatch nothing in lib/ has ever hit before, since no
		# existing caller anywhere calls .unwrap() on a Result[None,_] -
		# ListGenericTests' own list.append() usage only ever calls
		# .is_err(), never .unwrap()). Fixing that gap for real belongs to
		# whoever next needs a real None-payload Result value, not this
		# pass - discarding is both correct (nothing here ever reads the
		# payload) and sufficient (Err is still a real panic either way)
		if not want_result:
			self._emit( ir.Call( dest = None, target = unwrap, receiver = result, args = [ errmsg_const ], kwargs = {} ))
			return None
		dest = self._new_temp( unwrap.return_type )
		self._emit( ir.Call( dest = dest, target = unwrap, receiver = result, args = [ errmsg_const ], kwargs = {} ))
		return dest

	def _lower_slice_view( self, ptr: ir.Operand, length: ir.Operand, elem_type: Type, node: ast.AST ) -> ir.Operand:
		# builds a slice[elem_type] value directly via ir.Allocate - the one
		# construction shape in this pass with no prior source-level call
		# site to copy (slice[T] has no user-spellable constructor - see
		# lib/builtins/__init__.py's join() comment, "no array-literal
		# syntax"). Safe precisely because slice is a plain @cstruct, not
		# an RCClass: emitter_c.py's own Allocate handling already treats a
		# plain CStruct as "stack value construction, no header" (same
		# posture _lower_bound_method_closure's own direct Allocate below
		# uses for a ClosureType nothing in source can spell either) - no
		# _schedule_rcclass_construction needed, this isn't heap-allocated
		# or refcounted at all.
		slice_cls = self.lowering.discovery.find_name( 'slice', node )
		slice_spec = self.lowering.discovery._get_or_create_specialization( slice_cls, [ elem_type ])
		concrete_slice_cls = self.lowering._ensure_resolved( slice_spec ) # the real, monomorphized slice[elem_type] - see _expr_JoinedStr's own comment on why the concrete class (not the abstract generic one) is what downstream code needs
		dest = self._new_temp( slice_spec )
		self._emit( ir.Allocate( dest = dest, cls = concrete_slice_cls, fields = { '_ptr': ptr, '__len': length } ))
		return dest

	def _expr_JoinedStr( self, node: ast.JoinedStr, expected_type: Type|None ) -> ir.Operand:
		# f-string (PLAN_FSTRINGS.md). A fully compile-time-known JoinedStr
		# never reaches here at all - compile_time_transformer.py's own
		# _ConstFolder.visit_JoinedStr already collapsed it to a plain
		# ast.Constant(str) before lowering.py ever sees the function body.
		# expected_type is deliberately never used to type the result here,
		# same reasoning _expr_Constant's own comment gives for its own
		# TaggedUnion case: str.concat's return is authoritatively str
		# either way, and _lower_expr's own post-hoc coercion is what wraps
		# a plain str into a wider union afterward, if one was asked for.
		str_type = self.lowering.discovery.find_name_or_none( 'str' )
		if str_type is None:
			self.lowering.discovery.fail( f'f-string requires the str type to be available: {ast.unparse(node)}', node )

		if len( node.values ) == 0:
			return ir.Const( type = str_type, value = '' )
		if len( node.values ) == 1:
			return self._lower_fstring_part( node.values[0], str_type )

		parts = [ self._lower_fstring_part( value, str_type ) for value in node.values ]
		n = len( parts )
		usize_cls = self.lowering.discovery.get_intrinsics()['usize']

		# UnsafeList[str](n) - the escape hatch lib/builtins/__list.py's own
		# module docstring names for exactly this: a fixed-capacity,
		# never-escaping, single-statement-lifetime scratch buffer, with no
		# lock overhead a real list[T] would pay for no reason here (n is
		# fixed at compile time - capacity never grows, so append() below
		# can never actually trigger RawList._grow() at all)
		# _get_or_create_specialization + _ensure_resolved gives back the
		# REAL, concrete, already-monomorphized UnsafeList[str] ClassLike
		# (not the Specialization wrapper - same "swap a Specialization for
		# its monomorphized form" ensure_resolved always does), exactly the
		# way _try_lower_construct_call's own "explicit ClassName[T](...)"
		# branch does before ITS target_cls.type_params check ever runs
		# (type_resolver.py's own _try_resolve_namespace pre-resolves a
		# Subscript callee's Specialization the same way). Using this
		# CONCRETE class from here on (not the abstract UnsafeList) matters
		# for real: its own .names are ALREADY-substituted (T=str bound)
		# methods, no separate per-method Specialization dance needed - and
		# _schedule_rcclass_construction below specifically REQUIRES a
		# concrete class (passing the still-generic abstract one there
		# schedules the ABSTRACT __del__ as a standalone compile unit, T
		# forever unbound - confirmed via a real repro: "compiler.is_rc(T)
		# requires a concrete type" - type_resolver.py's own
		# _schedule_rcclass_destructor_deps documents this exact hazard
		# and guards against it with a cls.type_params check; this is the
		# same hazard from the calling side instead).
		unsafelist_cls = self.lowering.discovery.find_name( 'UnsafeList', node )
		cls_spec = self.lowering.discovery._get_or_create_specialization( unsafelist_cls, [ str_type ])
		concrete_cls = self.lowering._ensure_resolved( cls_spec )

		init = concrete_cls.names.get( '__init__' )
		self.lowering._ensure_resolved( init ) # schedules init ITSELF as a compile unit - monomorphize_class's own per-method substitution loop only builds+caches the substituted Function, it never schedules any of them for real emission on its own (confirmed via a real repro: an unscheduled monomorphized method compiles fine at the CALL SITE but is never actually emitted, producing a C "call to undeclared function" link-time-shaped error)
		self.lowering.schedule( init.return_type )
		for p in ( init.parameters or [] ):
			self.lowering.schedule( p.type )

		buf = self._new_temp( cls_spec )
		self.lowering._schedule_rcclass_construction( concrete_cls, cls_spec )
		self._emit( ir.Allocate( dest = buf, cls = concrete_cls, fields = {} ))
		n_const = ir.Const( type = usize_cls, value = n )
		self._emit( ir.Call( dest = None, target = init, receiver = buf, args = [ n_const ], kwargs = {} ))

		none_type = self.lowering.discovery.get_none_type()
		overflow_error_cls = self.lowering.discovery.find_name( 'OverflowError', node )
		append = concrete_cls.names.get( 'append' )
		self.lowering._ensure_resolved( append ) # see init's own comment on why this is needed
		self.lowering.schedule( append.return_type )
		for p in ( append.parameters or [] ):
			self.lowering.schedule( p.type )
		for part in parts:
			append_result = self._new_temp( append.return_type )
			self._emit( ir.Call( dest = append_result, target = append, receiver = buf, args = [ part ], kwargs = {} ))
			self._lower_unwrap_result(
				append_result, 'f-string: internal append failed (unreachable - buffer is pre-sized exactly)',
				none_type, overflow_error_cls, str_type, node, want_result = False,
			)

		const_ptr_cls = self.lowering.discovery.get_intrinsics()['ConstPtr']
		ptr_cls = self.lowering.discovery.get_intrinsics()['Ptr']
		ptr_str_type = self.lowering.discovery._get_or_create_specialization( ptr_cls, [ str_type ])
		index_error_cls = self.lowering.discovery.find_name( 'IndexError', node )
		get_ptr = concrete_cls.names.get( 'get_ptr' )
		self.lowering._ensure_resolved( get_ptr ) # see init's own comment on why this is needed
		self.lowering.schedule( get_ptr.return_type )
		for p in ( get_ptr.parameters or [] ):
			self.lowering.schedule( p.type )
		zero_const = ir.Const( type = usize_cls, value = 0 )
		get_ptr_result = self._new_temp( get_ptr.return_type )
		self._emit( ir.Call( dest = get_ptr_result, target = get_ptr, receiver = buf, args = [ zero_const ], kwargs = {} ))
		ptr = self._lower_unwrap_result(
			get_ptr_result, 'f-string: internal index failed (unreachable - buffer is non-empty by construction)',
			ptr_str_type, index_error_cls, str_type, node,
		)

		# CastWrap to ConstPtr[None] - a raw, untyped view into the buffer,
		# not ConstPtr[str] - matches slice[T]'s own redesigned _ptr field
		# (see its own comment on why: Ptr[str]/ConstPtr[str] compiles to
		# the exact same C type as a bare str handle, one star, wrong for
		# "array of handles")
		none_type_ptr_target = self.lowering.discovery.get_none_type()
		const_ptr_none = self.lowering.discovery._get_or_create_specialization( const_ptr_cls, [ none_type_ptr_target ])
		const_ptr = self._new_temp( const_ptr_none )
		self._emit( ir.CastWrap( dest = const_ptr, operand = ptr ))

		view = self._lower_slice_view( const_ptr, n_const, str_type, node )

		concat = self.lowering._find_method( str_type, 'concat' )
		self.lowering._ensure_resolved( concat )
		self.lowering.schedule( concat.return_type )
		for p in ( concat.parameters or [] ):
			self.lowering.schedule( p.type )
		dest = self._new_temp( str_type )
		self._emit( ir.Call( dest = dest, target = concat, receiver = None, args = [ view ], kwargs = {} ))
		return dest

	def _lower_bound_method_closure( self, node: ast.Attribute, obj: ir.Operand, method: Function, expected_type: Type|None ) -> ir.Operand:
		# worker.run used as a VALUE (not called) - a bound-method
		# reference, PLAN_CALLABLE.md's own "closure in miniature" deferred
		# item. Builds a real closure value: {fn: Ptr[None] (the memoized
		# trampoline above), self: Ptr[None] (the receiver, increffed -
		# "creating a closure is by definition creating a new reference")}
		# - a compiler-synthesized RCClass (ClosureType), so every existing
		# RC mechanism (cfg.py's is_rc/rc_leaves/assign/move) applies
		# completely unchanged from here on, no special-casing needed
		self.lowering._ensure_resolved( method )
		if method.parameters is None:
			self.lowering.discovery.fail( f'{method.qualname} could not be resolved (see earlier error): {ast.unparse(node)}', node )
		arg_types = [ p.type for p in method.parameters ]
		self.lowering.schedule( method.return_type )
		for t in arg_types:
			self.lowering.schedule( t )

		closure_type = self.lowering.discovery._get_or_create_closure_type( arg_types, method.return_type )
		self.lowering._ensure_resolved( closure_type )
		# same scheduling _lower_allocate_fields/_try_lower_construct_call
		# already do for every OTHER RCClass construction (sys.alloc[cls]/
		# sys.free/__del__ must be real, lowered compile units by the time
		# the emitter sees the ir.Allocate below) - closure_type is already
		# concrete, no Specialization involved, so target_cls == concrete_type
		self.lowering._schedule_rcclass_construction( closure_type, closure_type )
		trampoline = self.lowering._get_or_create_closure_trampoline( method, obj.type )

		ptr_cls = self.lowering.discovery.get_intrinsics()['Ptr']
		none_type = self.lowering.discovery.get_none_type()
		ptr_none_type = self.lowering.discovery._get_or_create_specialization( ptr_cls, [ none_type ] )
		trampoline_callable_type = self.lowering.discovery._get_or_create_callable_type(
			[ p.type for p in ( trampoline.parameters or [] ) ], trampoline.return_type,
		)
		trampoline_ptr_type = self.lowering.discovery._get_or_create_specialization( ptr_cls, [ trampoline_callable_type ] )

		fn_ref = ir.FunctionRef( type = trampoline_ptr_type, fn = trampoline )
		fn_erased = self._new_temp( ptr_none_type )
		self._emit( ir.CastWrap( dest = fn_erased, operand = fn_ref ))

		self_erased = self._new_temp( ptr_none_type )
		self._emit( ir.CastWrap( dest = self_erased, operand = obj ))

		self._emit( ir.Incref( value = obj )) # the closure is a new owner of the receiver

		dest = self._new_temp( expected_type or closure_type )
		self._emit( ir.Allocate( dest = dest, cls = closure_type, fields = { 'fn': fn_erased, 'self': self_erased } ))
		return dest

	def _expr_Attribute( self, node: ast.Attribute, expected_type: Type|None ) -> ir.Operand:
		# CEnum member VALUE expressions (OSError.FileNotFoundError used as
		# a runtime value) — the base is a class, not a runtime value, so
		# the normal _lower_expr path would reject it. Walk the namespace
		# chain through .names dicts (type_resolver.py already resolved
		# and scheduled every link) and fold the member to an ir.Const.
		chain = self.lowering.find_name_recursive( node )
		if chain is not None:
			obj, attr = chain
			if isinstance( obj, CEnum ):
				assert obj.resolve is None, (
					f'CEnum {obj.qualname} reached lowering unresolved — '
					f'type_resolver.py visit_Attribute should have resolved it'
				)
				value = obj.members.get( attr )
				if value is not None:
					return ir.Const( type = obj.value_type, value = value )
			# scope-like terminal (Module, RCClass, etc.) — look up the
			# final attribute as a value directly, without recursing into
			# _lower_expr (which would fail for `sys` when the base is a
			# Module, since a Module is not a value expression). Covers
			# `sys.stdout`, `sys.free`, `builtins.int`, etc. — any
			# module-level Variable/Function/RCClass reached by dotted name.
			names = getattr( obj, 'names', None )
			if isinstance( names, dict ):
				name_obj = names.get( attr )
				if isinstance( name_obj, Variable ):
					self.lowering._ensure_resolved( name_obj )
					return name_obj
		obj = self._lower_expr( node.value, None )
		# worker.run used as a VALUE (no call parens) - _attr_lookup below
		# only ever finds a Variable (a real field); a method is a
		# Function, which it rejects outright ("has no attribute"). Same
		# restrictions _lower_function_ref already enforces for a BARE
		# function reference (no receiver to bind there) minus the
		# receiver-less requirement itself, since THAT'S exactly what a
		# closure is for: a generic/overloaded/static method still has
		# nowhere natural to bind (no single fixed signature, or no
		# receiver at all - closures only wrap a REAL bound instance call)
		method = self.lowering._find_method( obj.type, node.attr )
		if (
			isinstance( method, Function ) and method.cls is not None
			and not method.is_static and not method.is_classmethod
			and not method.type_params and not method.is_overload
		):
			return self._lower_bound_method_closure( node, obj, method, expected_type )
		attr_var = self.lowering._attr_lookup( obj.type, node.attr, node )
		dest = self._new_temp( attr_var.type )
		self._emit( ir.GetAttr( dest = dest, obj = obj, attr = node.attr ))
		return dest

	def _expr_Tuple( self, node: ast.Tuple, expected_type: Type|None ) -> ir.Operand:
		# `(a, b, c)` in value position (PLAN_TUPLE.md) - the first real
		# handling of ast.Tuple as a VALUE anywhere in this file (elsewhere
		# it only ever appears as an annotation-subscript shape, e.g. Dict
		# [K,V]'s own multi-arg slice). No synthesized __init__/construct-
		# call round-trip needed: this builds the backing RCClass directly
		# via ir.Allocate's own field=value shape, the exact same "no real
		# __init__" convention _lower_allocate_fields already applies to any
		# class that doesn't declare one, and the same direct-Allocate shape
		# _lower_bound_method_closure already uses to build a ClosureType
		# value with no __init__ of its own either.
		if len( node.elts ) < 2:
			# arity 0/1 is a real Python ast.Tuple parsing ambiguity (a
			# 1-tuple LITERAL needs a trailing comma to disambiguate from a
			# plain parenthesized expression) - deferred rather than
			# guessed at, see PLAN_TUPLE.md's own "Deferred" list. An empty
			# `()` reaches here too (len 0) - same deferral.
			self.lowering.discovery.fail( f'tuple literals need at least 2 elements: {ast.unparse(node)}', node )
		operands: list[ir.Operand] = []
		for elt in node.elts:
			value = self._lower_expr( elt, None )
			# same per-field RC-retain emission _lower_allocate_fields's own
			# field-value loop uses for every other class's field=value
			# construction sugar - a fresh value (Allocate/Call/Constant)
			# needs no extra incref, an aliasing read of an existing
			# binding (Name/Attribute) does, since the tuple now
			# independently owns a reference alongside whatever binding the
			# element came from
			for instr in self._cfg.field_value( value.type, value, is_alias = self.lowering._is_aliasing_expr( elt, value.type )):
				self._emit( instr )
			operands.append( value )
		tt = self.lowering.discovery._get_or_create_tuple_type( [ op.type for op in operands ] )
		backing_cls = self.lowering._ensure_resolved( tt )
		# every constructed RCClass needs sys.alloc[T]/sys.free (and __del__,
		# if declared - not applicable here) scheduled at the CONSTRUCTION
		# site, same as every other ir.Allocate emission in this file -
		# _ensure_resolved(tt) alone only guarantees the backing class
		# itself is scheduled, not its allocator
		self.lowering._schedule_rcclass_construction( backing_cls, backing_cls )
		fields = { f'_{i}': op for i, op in enumerate( operands ) }
		# resolve expected_type too, not just tt - an annotated declaration
		# (`x: tuple[i32,str] = (1,"a")`) hands down the SAME bare,
		# unresolved TupleType discovery.py's visit_Subscript produced for
		# the annotation (interned - same object as tt above), not yet
		# swapped for backing_cls
		resolved_expected = self.lowering._ensure_resolved( expected_type ) if expected_type is not None else None
		dest = self._new_temp( resolved_expected or backing_cls )
		self._emit( ir.Allocate( dest = dest, cls = backing_cls, fields = fields ))
		return dest

	def _expr_Subscript( self, node: ast.Subscript, expected_type: Type|None ) -> ir.Operand:
		obj = self._lower_expr( node.value, None )
		getitem_fn = self.lowering._find_method( obj.type, '__getitem__' )
		if getitem_fn is None:
			# tuple[...]'s own constant-index-only element access
			# (PLAN_TUPLE.md) - checked ahead of the ordinary Ptr/ConstPtr
			# GetItem fallback below: a heterogeneous tuple has no real
			# __getitem__ (no single return type to give one), so `t[0]`
			# can only ever be resolved to plain attribute access on a
			# COMPILE-TIME-CONSTANT index, never a runtime GetItem
			resolved_obj_type = self.lowering._ensure_resolved( obj.type )
			tuple_type = self.lowering._tuple_storage.tuple_type_for( resolved_obj_type )
			if tuple_type is not None:
				valid_index = (
					isinstance( node.slice, ast.Constant )
					and isinstance( node.slice.value, int )
					and not isinstance( node.slice.value, bool ) # bool is an int subclass in Python's own ast - not a legal tuple index
				)
				if not valid_index:
					self.lowering.discovery.fail(
						f'tuple element access requires a compile-time-constant integer index: {ast.unparse(node)}',
						node,
					)
				index = node.slice.value
				if not ( 0 <= index < len( tuple_type.elem_types )):
					self.lowering.discovery.fail(
						f'tuple index {index} out of range for {resolved_obj_type.qualname} (0..{len(tuple_type.elem_types)-1}): {ast.unparse(node)}',
						node,
					)
				attr_var = self.lowering._attr_lookup( resolved_obj_type, f'_{index}', node )
				dest = self._new_temp( attr_var.type )
				self._emit( ir.GetAttr( dest = dest, obj = obj, attr = f'_{index}' ))
				return dest
			# no real __getitem__ declared (raw pointers, or any other type
			# that doesn't define subscript access as a method) - falls
			# back to the flat GetItem opcode, unconditionally
			if expected_type is None:
				# for Ptr[T]/ConstPtr[T], the pointee type is the natural
				# result of a dereference; for any other type we can't guess
				if isinstance( obj.type, Specialization ) and isinstance( obj.type.base, Scalar ) and obj.type.base.stem in ( 'Ptr', 'ConstPtr' ):
					expected_type = obj.type.args[0]
				else:
					self.lowering.discovery.fail( f'cannot infer the result type of {ast.unparse(node)} - no expected type available from context', node )
			# pointer subscript indices are always usize (pointer arithmetic
			# is defined in terms of the pointer's own element size, not the
			# index's runtime width) — give the index a concrete type so a
			# bare literal 0 in e.g. `ptr[0]` doesn't fail type inference
			index_type = self.lowering.discovery.get_intrinsics()['usize']
			index = self._lower_expr( node.slice, index_type )
			dest = self._new_temp( expected_type )
			self._emit( ir.GetItem( dest = dest, obj = obj, index = index ))
			return dest

		# a real __getitem__ - call it like any other method, then if it
		# returns Result[T,E] (slice.__getitem__'s own real signature, e.g.),
		# auto-consume it exactly like or_return()/checked arithmetic do:
		# `obj[i]` reads as sugar for `obj.__getitem__(i).or_return()`
		# whenever __getitem__ can fail
		self.lowering._ensure_resolved( getitem_fn )
		self.lowering.schedule( getitem_fn.return_type )
		index = self._lower_expr( node.slice, getitem_fn.parameters[0].type )
		call_dest = self._new_temp( getitem_fn.return_type )
		self._emit( ir.Call( dest = call_dest, target = getitem_fn, receiver = obj, args = [ index ], kwargs = {} ))
		return self._maybe_consume_result( node, call_dest, self.lowering._SUBSCRIPT_ALTERNATIVES )

	def _expr_Call( self, node: ast.Call, expected_type: Type|None ) -> ir.Operand:
		return self._lower_call( node, expected_type, want_result = True )

	def _lower_binary_operands( self, left_node: ast.expr, right_node: ast.expr, expected_type: Type|None, *, infer_right_from_left: bool = True ) -> tuple[ir.Operand,ir.Operand]:
		# shared by _expr_BinOp and _expr_Compare: a bare literal constant on
		# either side has no type of its own to offer, so the non-constant
		# side is lowered first and its own inferred type used as the
		# constant's expected_type instead. `infer_right_from_left` captures
		# the one real difference between the two callers when NEITHER side
		# is constant: _expr_BinOp still hints the right operand with the
		# left operand's own inferred type (expected_type or left.type) - but
		# _expr_Compare's expected_type is the comparison's own result type
		# (bool), unrelated to the operands, and never cross-hints one
		# operand from the other outside the constant branches above
		left_is_const = isinstance( left_node, ast.Constant )
		right_is_const = isinstance( right_node, ast.Constant )
		usize_cls = self.lowering.discovery.get_intrinsics()['usize']
		# pointer arithmetic (ptr + offset) is never homogeneous the way
		# ordinary scalar +/- is - hinting the OTHER (non-pointer) side with
		# the pointer's own type here (as every branch below otherwise
		# does) would wrongly propagate into a nested BinOp too (e.g. `ptr +
		# idx * element_size` - the outer Add's own right-hint, if left
		# unguarded, leaks into the INNER Mult's result_type, producing a
		# nonsensical "checked pointer multiply"). usize is the natural
		# offset type - same reasoning _expr_Subscript's own pointer index
		# hint already uses
		if left_is_const and not right_is_const:
			right = self._lower_expr( right_node, expected_type )
			left_hint = usize_cls if self.lowering._type_resolver._is_ptr_specialization( right.type ) else right.type
			left = self._lower_expr( left_node, left_hint )
		elif right_is_const and not left_is_const:
			left = self._lower_expr( left_node, expected_type )
			right_hint = usize_cls if self.lowering._type_resolver._is_ptr_specialization( left.type ) else left.type
			right = self._lower_expr( right_node, right_hint )
		else:
			left = self._lower_expr( left_node, expected_type )
			if infer_right_from_left:
				right_hint = usize_cls if self.lowering._type_resolver._is_ptr_specialization( left.type ) else ( expected_type or left.type )
			else:
				right_hint = expected_type
			right = self._lower_expr( right_node, right_hint )
		return left, right

	def _expr_BinOp( self, node: ast.BinOp, expected_type: Type|None ) -> ir.Operand:
		left, right = self._lower_binary_operands( node.left, node.right, expected_type )
		return self._lower_binop_values( node, left, right, expected_type )

	def _lower_binop_values( self, node: 'ast.BinOp|ast.AugAssign', left: ir.Operand, right: ir.Operand, expected_type: Type|None ) -> ir.Operand:
		# the dunder-dispatch/checked-arithmetic core of _expr_BinOp, split
		# out so _stmt_AugAssign's own Attribute/Subscript-target handling
		# can reuse it with operands it already lowered itself (reading the
		# target's object/index exactly once - see that method's own
		# comment) instead of going through _lower_binary_operands, which
		# always lowers both sides fresh from AST. `node` is only ever
		# read for its `.op` (ast.BinOp and ast.AugAssign both have one)
		# and as an error-reporting location - never for `.left`/`.right`.

		# non-scalar left operand — try the dunder method (str.__add__, ...)
		if not isinstance( left.type, Scalar ):
			method_name = _BINOP_DUNDER.get( type( node.op ))
			if method_name is not None:
				method = self.lowering._find_method( left.type, method_name )
				if method is not None:
					self.lowering._ensure_resolved( method )
					self.lowering.schedule( method.return_type )
					for p in ( method.parameters or [] ):
						self.lowering.schedule( p.type )
					dest = self._new_temp( expected_type or method.return_type )
					self._emit( ir.Call( dest = dest, target = method, receiver = left, args = [ right ], kwargs = {} ))
					return dest

		result_type = expected_type or left.type

		opcode, extra = self._arithmetic_mode[-1].GetBinOp( node )
		return self._lower_arithmetic_op( node, opcode, extra, result_type, { 'left': left, 'right': right }, 'binary' )

	def _lower_arithmetic_op( self, node: ast.AST, opcode: type|None, extra: ir.Operand|None, result_type: Type, operand_kwargs: dict, kind: str ) -> ir.Operand:
		# shared by _lower_scalar_cast/_expr_BinOp/_expr_UnaryOp - each just
		# resolves its own (opcode, extra) via the active ArithmeticMode's
		# GetCast/GetBinOp/GetUnaryOp and hands them here along with its own
		# operand shape (cast/USub take a single `operand`, BinOp takes
		# `left`/`right`). `kind` is only used for the unsupported-operator
		# message below - _lower_scalar_cast's GetCast() never actually
		# returns None (every mode defines a cast opcode), so that branch is
		# unreachable from there, but harmless to share
		if opcode is None:
			self.lowering.discovery.fail( f'unsupported {kind} operator: {ast.unparse(node)}', node )
		if not opcode.checked_error:
			# wrap/saturate, or no overflow concept at all (bitwise/Invert)
			dest = self._new_temp( result_type )
			self._emit( opcode( dest = dest, **operand_kwargs ))
			return dest
		# check mode (the default - see the class docstring): the op itself
		# produces Result[result_type,<opcode.checked_error>]. How that
		# Result gets consumed depends on `extra`: the default (extra is
		# None) uses OrReturn, mirroring Result.or_return()'s own semantics,
		# and needs somewhere for the error to propagate to; `with
		# compiler.panic_arithmetic(msg):` (extra is the lowered msg
		# operand) uses Unwrap instead, which panics immediately and so has
		# no such requirement
		result_cls, error_cls = self.lowering._type_resolver._lookup_result_and_error_types( node, opcode.checked_error )
		if extra is None:
			# validated before anything gets emitted - a mid-statement
			# failure here must not leave partial instructions behind for
			# the per-statement recovery boundary to silently keep
			self.lowering._type_resolver._require_result_return( node, result_cls, error_cls, _ALTERNATIVES_BY_ERROR[opcode.checked_error], fn = self._current_fn )
		return self._emit_checked_op( node, opcode, operand_kwargs, result_type, result_cls, error_cls, extra )

	def _emit_checked_op( self, node: ast.AST, opcode: type, operand_kwargs: dict, result_type: Type, result_cls: ClassLike, error_cls: ClassLike, extra: ir.Operand|None ) -> ir.Temp:
		# shared by Check-mode binops (Add/Sub/Mult/Shl/Div/Mod), USub, and
		# scalar casts - operand_kwargs is however the specific opcode names
		# its operand(s) (left/right for a binop, operand for USub/cast)
		check_type = self.lowering.discovery._get_or_create_specialization( result_cls, [ result_type, error_cls ] )
		# the emitter declares a local variable of this Result type; the
		# struct definition must exist even though the Check op's result
		# is consumed inline (OrReturn/OrJump/Unwrap) — schedule it now
		# so monomorphize_class emits it into compiler.tagged_unions
		self.lowering.schedule( check_type )
		check_dest = self._new_temp( check_type )
		self._emit( opcode( dest = check_dest, **operand_kwargs ))
		return self._consume_checked_result( node, check_dest, result_type, extra )

	def _consume_checked_result( self, node: ast.AST, check_dest: ir.Temp, result_type: Type, extra: ir.Operand|None ) -> ir.Temp:
		# shared by both binop (AddCheck/.../Div/Mod) and unary (NegCheck)
		# Check-mode ops, _maybe_consume_result's __len__/__getitem__ auto-
		# unwrap, and _lower_or_return's own <result_expr>.or_return() - see
		# _expr_BinOp's own comment on the OrReturn/OrJump/Unwrap split.
		# check_dest is sometimes a real, named Variable (or_return()'s own
		# receiver) and sometimes a bare Temp (checked arithmetic, __len__/
		# __getitem__'s auto-unwrap) - isinstance covers both uniformly
		unwrapped = self._new_temp( result_type )
		if extra is None:
			if isinstance( check_dest, Variable ):
				self._cfg.clear_result( check_dest.stem ) # this call IS the inspection of check_dest - clear it before the exit-path check below, or it'd wrongly flag itself
			# the OrReturn/OrJump path below is a second, separate function-
			# exit point alongside plain `return` (see cfg.check_unchecked_
			# results' own docstring) - anything else still unchecked here
			# would otherwise be silently discarded exactly like falling off
			# the end unchecked would be
			try:
				self._cfg.check_unchecked_results( None )
			except CompileError as e:
				self.lowering.discovery.fail( str( e ), node )
			label = self._cfg.current_epilogue_label()
			if label is not None:
				self._emit( ir.OrJump( dest = unwrapped, value = check_dest, target = label, return_slot = self._return_value_var ))
			else:
				self._emit( ir.OrReturn( dest = unwrapped, value = check_dest ))
		else:
			panic_fn = self.lowering._type_resolver._resolve_sys_function( 'panic' )
			self.lowering.schedule( panic_fn )
			self._emit( ir.Unwrap( dest = unwrapped, value = check_dest, errmsg = extra, panic = panic_fn ))
		return unwrapped

	def _expr_UnaryOp( self, node: ast.UnaryOp, expected_type: Type|None ) -> ir.Operand:
		if isinstance( node.op, ast.Not ):
			operand = self._lower_expr( node.operand, expected_type )
			dest = self._new_temp( expected_type or operand.type )
			self._emit( ir.Not( dest = dest, operand = operand ))
			return dest
		operand = self._lower_expr( node.operand, expected_type )

		# non-scalar operand — try the dunder method (int.__neg__, ...),
		# mirroring _expr_BinOp/_expr_Compare's identical dispatch
		if not isinstance( operand.type, Scalar ):
			method_name = _UNARYOP_DUNDER.get( type( node.op ))
			if method_name is not None:
				method = self.lowering._find_method( operand.type, method_name )
				if method is not None:
					self.lowering._ensure_resolved( method )
					self.lowering.schedule( method.return_type )
					for p in ( method.parameters or [] ):
						self.lowering.schedule( p.type )
					dest = self._new_temp( expected_type or method.return_type )
					self._emit( ir.Call( dest = dest, target = method, receiver = operand, args = [], kwargs = {} ))
					return dest

		result_type = expected_type or operand.type

		opcode, extra = self._arithmetic_mode[-1].GetUnaryOp( node )
		return self._lower_arithmetic_op( node, opcode, extra, result_type, { 'operand': operand }, 'unary' )

	def _expr_BoolOp( self, node: ast.BoolOp, expected_type: Type|None ) -> ir.Operand:
		# short-circuit and/or: evaluate operands left to right, each into
		# the same dest temp, stopping early (jump to end) as soon as the
		# result is already decided - `and` stops on the first falsy
		# operand, `or` stops on the first truthy one. Needed by match's
		# nested pattern tests (an outer tag check AND, only if that
		# passes, an inner tag check on the payload - reading the payload
		# before confirming the outer tag would be reading the wrong
		# union member's storage)
		bool_cls = self.lowering.discovery.find_name( 'bool', node )
		is_and = isinstance( node.op, ast.And )
		end_label = self._new_label( 'booland' if is_and else 'boolor' )
		dest = self._new_temp( bool_cls )
		for i, value_node in enumerate( node.values ):
			operand = self._lower_expr( value_node, bool_cls )
			self._emit( ir.Assign( dest = dest, src = operand ))
			if i < len( node.values ) - 1:
				jump_opcode = ir.JumpIfFalse if is_and else ir.JumpIfTrue
				self._emit( jump_opcode( cond = dest, target = end_label ))
		self._emit( ir.Label( name = end_label ))
		return dest

	def _expr_IfExp( self, node: ast.IfExp, expected_type: Type|None ) -> ir.Operand:
		# ternary `x if cond else y` — both branches assign to the same
		# dest temp, then merge at end_label. Use JumpIfTrue so the true
		# branch (body) comes first, avoiding an extra negate.
		bool_cls = self.lowering.discovery.find_name( 'bool', node )
		cond = self._lower_expr( node.test, bool_cls )
		else_label = self._new_label( 'ifexp_else' )
		end_label = self._new_label( 'ifexp_end' )
		dest = self._new_temp( expected_type ) if expected_type is not None else None
		self._emit( ir.JumpIfFalse( cond = cond, target = else_label ))
		# true branch
		true_val = self._lower_expr( node.body, expected_type )
		if dest is None:
			dest = self._new_temp( true_val.type )
		self._emit( ir.Assign( dest = dest, src = true_val ))
		self._emit( ir.Jump( target = end_label ))
		# false branch
		self._emit( ir.Label( name = else_label ))
		false_val = self._lower_expr( node.orelse, dest.type )
		self._emit( ir.Assign( dest = dest, src = false_val ))
		self._emit( ir.Label( name = end_label ))
		return dest

	def _expr_Compare( self, node: ast.Compare, expected_type: Type|None ) -> ir.Operand:
		# ast.In/NotIn are deliberately not handled here - `in`/`not in`
		# need a real container protocol that doesn't exist yet, guessing
		# would bake in the wrong semantics. ast.Is/IsNot ARE handled (see
		# _lower_is_comparison) - identity happens to coincide with value
		# equality for every value kind this language has today
		if len( node.ops ) != 1 or len( node.comparators ) != 1:
			self.lowering.discovery.fail( f'chained comparisons are not yet supported: {ast.unparse(node)}', node )
		if isinstance( node.ops[0], ( ast.Is, ast.IsNot )):
			return self._lower_is_comparison( node, negate = isinstance( node.ops[0], ast.IsNot ))

		# non-scalar left operand — try the dunder method (str.__eq__, ...)
		left = self._lower_expr( node.left, None )
		if not isinstance( left.type, Scalar ):
			method_name = _COMP_DUNDER.get( type( node.ops[0] ))
			if method_name is not None:
				method = self.lowering._find_method( left.type, method_name )
				if method is not None:
					right = self._lower_expr( node.comparators[0], left.type )
					self.lowering._ensure_resolved( method )
					self.lowering.schedule( method.return_type )
					for p in ( method.parameters or [] ):
						self.lowering.schedule( p.type )
					dest = self._new_temp( expected_type or method.return_type )
					self._emit( ir.Call( dest = dest, target = method, receiver = left, args = [ right ], kwargs = {} ))
					return dest
			# non-scalar without a matching dunder — fall through to
			# flat Cmp (pointer comparison), same pre-dunder behavior

		# scalar left operand — flat ir.Cmp
		right = self._lower_expr( node.comparators[0], left.type )
		cmp_op = self.lowering._CMP_OPCODES.get( type( node.ops[0] ))
		if cmp_op is None:
			self.lowering.discovery.fail( f'unsupported comparison operator: {ast.unparse(node)}', node )
		bool_cls = self.lowering.discovery.find_name( 'bool', node )
		dest = self._new_temp( bool_cls )
		self._emit( ir.Cmp( dest = dest, op = cmp_op, left = left, right = right ))
		return dest

	def _lower_is_comparison( self, node: ast.Compare, negate: bool ) -> ir.Operand:
		# `is`/`is not` mean real Python identity - for every value kind
		# this language has today (scalars, pointers, RC handles) identity
		# coincides with value equality, so this is plain Cmp EQ/NE...
		# UNLESS one side is a bare `None` literal being compared against a
		# TaggedUnion-typed value (T|None, e.g. sys._alloc()'s
		# Ptr[u8]|None) - there, "is None" means "the active member is
		# NoneType", which needs a tag check (the same UnionStorage.get
		# machinery match statements/conditional dispatch already use), not
		# a flat Cmp against a synthesized None operand of union type
		# (which wouldn't correspond to any real runtime representation)
		bool_cls = self.lowering.discovery.find_name( 'bool', node )
		cmp_op = ir.CmpOp.NE if negate else ir.CmpOp.EQ
		left_node, right_node = node.left, node.comparators[0]
		left_is_none = isinstance( left_node, ast.Constant ) and left_node.value is None
		right_is_none = isinstance( right_node, ast.Constant ) and right_node.value is None

		if left_is_none and right_is_none:
			return ir.Const( type = bool_cls, value = not negate ) # `None is None` / `None is not None` - degenerate, but not a crash

		if left_is_none or right_is_none:
			# a TaggedUnion-typed operand (T|None) never reaches here anymore -
			# type_resolver.py's _ReferenceResolver already rewrote that
			# shape into a plain tag Eq/NotEq Compare before lowering ever
			# saw this statement (see its own visit_Compare). What's left is
			# a flat Cmp against a real None-typed Const - e.g. a raw
			# Ptr[T]|None never actually applies (still a TaggedUnion), so in
			# practice this is for whatever non-union type this language
			# ever allows a bare `is None` against
			other = self._lower_expr( right_node if left_is_none else left_node, None )
			dest = self._new_temp( bool_cls )
			self._emit( ir.Cmp( dest = dest, op = cmp_op, left = other, right = ir.Const( type = other.type, value = None ) ))
			return dest

		left = self._lower_expr( left_node, None )
		right = self._lower_expr( right_node, left.type )
		dest = self._new_temp( bool_cls )
		self._emit( ir.Cmp( dest = dest, op = cmp_op, left = left, right = right ))
		return dest

	def _resolve_callee( self, func_node: ast.expr ) -> tuple[Function|Overload|Specialization|_ReceiverDispatch,ir.Operand|None]:
		target = self.lowering._type_resolver._resolve_callee_target( func_node )
		if target is not None:
			return target, None

		if not isinstance( func_node, ast.Attribute ):
			self.lowering.discovery.fail( f'cannot call {ast.unparse(func_node)}', func_node )
		receiver = self._lower_expr( func_node.value, None )
		shape = self.lowering._type_resolver._tagged_union_shape( receiver.type )
		if shape is not None:
			base, members = shape
			self.lowering._ensure_resolved( base )
			direct = base.names.get( func_node.attr )
			if not isinstance( direct, ( Function, Overload )):
				return self.lowering._type_resolver._resolve_union_receiver_members( base, members, func_node.attr, func_node ), receiver
		target = self.lowering._type_resolver._attr_lookup_callable( receiver.type, func_node.attr, func_node )
		return target, receiver

	def _apply_move_hook( self, param: Parameter, operand: ir.Operand, target_qualname: str ) -> None:
		# the semantic half of move[T] - _check_move_argument (run earlier,
		# inside _match_call_args) already validated the call-site syntax
		# agrees; this is where the argument's OWN ownership state actually
		# transitions, once its real Operand exists (needs the lowered
		# value, not just the AST expr) - shared by every _match_call_args
		# caller (plain calls, both generic call flavors, union-receiver
		# dispatch), called right after each argument is lowered
		if isinstance( param.type, Move ):
			for instr in self._cfg.move( operand, target_qualname = target_qualname, param_stem = param.stem ):
				self._emit( instr )

	def _lower_overload_arg( self, expr: ast.expr, position: int|None, kw_name: str|None, candidates: list[Function], node: ast.AST ) -> ir.Operand:
		if not isinstance( expr, ast.Constant ):
			return self._lower_expr( expr, None )
		compatible_stems = self.lowering._LITERAL_COMPATIBLE_STEMS.get( type( expr.value ) )
		if compatible_stems is None:
			return self._lower_expr( expr, None ) # a None literal, or something else - falls through to _expr_Constant's own "cannot infer" error, same as before

		candidate_types: list[Type] = []
		for fn in candidates:
			if fn.parameters is None:
				continue
			param = (
				fn.parameters[position] if position is not None and position < len( fn.parameters ) else
				next( ( p for p in fn.parameters if p.stem == kw_name ), None )
			)
			if param is None or param.type is None or getattr( param.type, 'stem', None ) not in compatible_stems:
				continue
			if not any( t is param.type for t in candidate_types ):
				candidate_types.append( param.type )

		if len( candidate_types ) == 1:
			return self._lower_expr( expr, candidate_types[0] )
		if len( candidate_types ) > 1:
			self.lowering.discovery.fail(
				f'ambiguous literal argument {ast.unparse(expr)} - matches more than one overload candidate type '
				f'({", ".join( t.qualname for t in candidate_types )}): {ast.unparse(node)}',
				node,
			)
		return self._lower_expr( expr, None ) # no candidate's parameter type is even plausible for this literal's kind - falls through to the existing error

	def _check_rcclass_fully_implemented( self, target_cls: RCClass, node: ast.AST, label: str ) -> None:
		''' RCClass analog of the CStruct-interface stub-body check just
		below (RCClass-subclassing plan Phase 5) - an RCClass with any
		unfulfilled @abstractmethod slot anywhere in its own chain is
		never meant to be constructed directly. Unlike CStruct's implicit
		stub-body-means-unimplemented convention, RCClass uses the
		EXPLICIT is_abstract marker (discovery.py already requires
		@abstractmethod to also be @virtual and have a stub body - so
		checking is_abstract here is equivalent to checking the body
		shape, just reads as what it actually means). Shared by BOTH
		RCClass construction paths (_lower_allocate_fields's own call
		below, and _try_lower_construct_call's __init__-based path) via
		this one helper, so the two can't drift out of sync - matches
		compiler.py's own _schedule_rcclass_vtable_impls, which this stays
		in lockstep with (emitter_c.py's emit_rcclass_vtable_instance
		skips building an instance at all for a class this check would
		reject, the same "None if any slot is unfulfilled" gate CStruct's
		own emit_interface_vtable_instance already uses). '''
		unfulfilled: list[str] = []
		for slot in target_cls.virtual_slots():
			impl = target_cls.chain_lookup( slot.stem )
			assert isinstance( impl, Function ) # virtual_slots()'s own entries always exist somewhere in the chain - at minimum the root's own declaration chain_lookup started from
			if impl.resolve is not None:
				impl.resolve()
			if impl.is_abstract:
				unfulfilled.append( slot.stem )
		if unfulfilled:
			self.lowering.discovery.fail(
				f'{target_cls.qualname}{label} cannot be constructed - abstract method(s) have no implementation: '
				f'{", ".join(unfulfilled)}',
				node,
			)

	def _lower_allocate_fields( self, target_cls: ClassLike, node: ast.Call, expected_type: Type|None, label: str ) -> ir.Temp:
		# shared by both callers of ir.Allocate (Class.__allocate__(...) and
		# bare ClassName(...) sugar for the no-__init__ case) - everything
		# past "which class, and is this call form even allowed here" is
		# identical field-matching/emission logic. `label` is just how the
		# call reads in error messages (".__allocate__(...)" vs "(...)"), so
		# existing callers' error text doesn't change.
		if node.args:
			self.lowering.discovery.fail( f'{target_cls.qualname}{label} takes keyword arguments only: {ast.unparse(node)}', node )
		if any( kw.arg is None for kw in node.keywords ):
			self.lowering.discovery.fail( f'**kwargs not supported for {target_cls.qualname}{label}: {ast.unparse(node)}', node )

		self.lowering._ensure_resolved( target_cls )
		# .attributes alone only ever holds a class's OWN declared fields
		# (discovery.py never merges a base's own fields into a subclass) -
		# flattened_attributes() walks the WHOLE single-inheritance chain
		# (base-first), which is what this no-__init__ field=value sugar
		# needs to see every constructible field, inherited or not. A no-op
		# widening for RCClass/CStruct with no base (returns the same list
		# .attributes would) and for CUnion/CEnum (never have .base at all)
		target_fields = target_cls.flattened_attributes() if isinstance( target_cls, ( RCClass, CStruct )) else target_cls.attributes
		for attr in target_fields:
			self.lowering._ensure_resolved( attr ) # each field's own .type is lazily resolved, separate from the class itself - same as _attr_lookup's found.resolve
		if isinstance( target_cls, CStruct ) and target_cls.is_interface:
			# a "pure interface" (or any @interface class with an unfulfilled
			# @virtual slot anywhere in its chain - a stub body, same shape
			# @overload stubs use) is never meant to be constructed directly -
			# nothing else in this compiler enforces that (there's no separate
			# @abstract marker - see PLAN_SUBCLASSING_VTABLES_COM.md's own
			# "Unimplemented @virtual methods" reasoning), so it's checked
			# here, at the one place a real CStruct value actually gets built
			unfulfilled: list[str] = []
			for slot in target_cls.virtual_slots():
				impl = target_cls.chain_lookup( slot.stem )
				assert isinstance( impl, Function ) # virtual_slots()'s own entries always exist somewhere in the chain - at minimum the root's own declaration chain_lookup started from
				if impl.resolve is not None:
					impl.resolve()
				if is_stub_body( impl.node.body ):
					unfulfilled.append( slot.stem )
			if unfulfilled:
				self.lowering.discovery.fail(
					f'{target_cls.qualname}{label} cannot be constructed - virtual method(s) have no implementation: '
					f'{", ".join(unfulfilled)}',
					node,
				)
		elif isinstance( target_cls, RCClass ):
			self._check_rcclass_fully_implemented( target_cls, node, label )
		# target_cls is always the ABSTRACT class (resolved via
		# _try_resolve_namespace on the shared, unspecialized AST body's
		# own `SomeGeneric.__allocate__` reference - see
		# _try_lower_allocate_call) even from inside a monomorphized
		# generic-class method, whose self._current_fn.cls IS the concrete
		# specialization - substitute target_cls's own field types against
		# that concrete specialization when one's available, same as
		# _substituted_field already does for ordinary attribute reads
		# (_expr_Attribute), or a generic field's expected type here would
		# stay abstract (its own bare TypeVars) forever
		fn_cls = self._current_fn.cls if self._current_fn is not None else None
		if isinstance( target_cls, TaggedUnion ):
			# a TaggedUnion's REAL storage shape is tag+data (synthesized by
			# UnionStorage.get(), registered in .names) - .attributes is the
			# LOGICAL member list (Ok/Err), a completely different thing
			# (used for .leaves()/type-matching, never for real field
			# layout). .__allocate__(tag=.., data=..) - the shape the
			# synthesized per-member constructor's own body uses (see
			# union_storage.py's _build_member_constructor) - must validate
			# against tag/data, not the logical members
			if isinstance( fn_cls, Specialization ) and fn_cls.base is target_cls:
				# target_cls itself is always the ABSTRACT union (see the
				# comment above on why bodies always reference it that way),
				# so target_cls.names['data'].type would stay the ABSTRACT,
				# unsubstituted payload_cls forever - substitute_field can't
				# help here (a payload_cls is a plain CUnion, not a TypeVar/
				# Specialization/anonymous-union shape it knows how to
				# rebuild), so read tag/data from the MONOMORPHIZED copy
				# instead (_ensure_resolved(fn_cls) is exactly monomorphize_
				# class - see its own "fresh payload_cls per specialization"
				# comment for why that copy's own data field is already
				# correctly substituted)
				concrete_union = self.lowering._ensure_resolved( fn_cls )
				tag_field = concrete_union.names.get( 'tag' )
				data_field = concrete_union.names.get( 'data' )
			else:
				tag_field = target_cls.names.get( 'tag' )
				data_field = target_cls.names.get( 'data' )
			assert isinstance( tag_field, Variable ) and isinstance( data_field, Variable ), \
				f'{target_cls.qualname}: UnionStorage.get() has not run yet - no real tag/data storage to allocate'
			declared = { tag_field.stem: tag_field, data_field.stem: data_field }
		elif isinstance( fn_cls, Specialization ) and fn_cls.base is target_cls:
			declared = { attr.stem: self.lowering._substituted_field( attr, fn_cls ) for attr in target_fields }
		else:
			declared = { attr.stem: attr for attr in target_fields }
		given = { kw.arg for kw in node.keywords }
		missing = declared.keys() - given
		if isinstance( target_cls, CUnion ):
			# a union's whole point - only ONE member is ever meaningfully
			# set at a time (see UnionStorage.get's identical comment
			# on the synthesized TaggedUnion payload CUnion) - "every OTHER
			# field is missing" isn't an error here the way it is for an
			# ordinary struct/class, unlike ResultPayload(ok=val) never
			# giving err. Pre-existing gap, confirmed unrelated to this
			# pass (reproduces on a clean checkout: Result.Ok(...)/
			# Result.Err(...)'s own bodies were never actually exercised
			# through a full Compiler.run() before, so this went unnoticed)
			if len( given ) != 1:
				self.lowering.discovery.fail( f'{target_cls.qualname}{label} takes exactly one field (only one union member is ever set): {ast.unparse(node)}', node )
		else:
			truly_missing = sorted( name for name in missing if declared[name].init is None )
			if truly_missing:
				self.lowering.discovery.fail( f'{target_cls.qualname}{label} is missing field(s): {", ".join(truly_missing)}', node )
		extra = given - declared.keys()
		if extra:
			self.lowering.discovery.fail( f'{target_cls.qualname}{label} has no field(s): {", ".join(sorted(extra))}', node )

		given_by_name = { kw.arg: kw.value for kw in node.keywords }
		# a CUnion only ever builds the ONE given member - the other
		# declared fields aren't "defaulted", they're simply not part of
		# this particular construction at all (unlike an ordinary struct/
		# class, where every field always exists)
		fields_to_build = { name: declared[name] for name in given_by_name } if isinstance( target_cls, CUnion ) else declared
		fields: dict[str,ir.Operand] = {}
		for name, field in fields_to_build.items():
			if name in given_by_name:
				expr = given_by_name[name]
				value = self._lower_expr( expr, field.type )
			else:
				# omitted at the call site, but declared with a default
				# (`field.init`, already confirmed not None by truly_missing
				# above) - lowered in the CLASS's own scope, not the caller's,
				# matching ordinary Python class-body scoping (a default
				# expression can reference other class-level names, but not
				# anything local to whoever's constructing this instance)
				expr = field.init
				with self.lowering.discovery.module_context( self.lowering._find_module_for( target_cls )):
					with self.lowering.discovery.scope_context( target_cls ):
						value = self._lower_expr( expr, field.type )
			# value.type, not field.type: field.type is the FIELD's declared
			# type, which stays an unsubstituted TypeVar for any field whose
			# type depends on a class's own type params (class methods are
			# never monomorphized per-specialization - see compiler.py's
			# _enqueue) - value.type is always the operand's real, concrete
			# type regardless, since only concrete values ever actually get
			# lowered
			for instr in self._cfg.field_value( value.type, value, is_alias = self.lowering._is_aliasing_expr( expr, value.type )):
				self._emit( instr )
			fields[name] = value

		if isinstance( target_cls, CStruct ) and target_cls.is_interface:
			# an @interface CStruct is never a plain value type (see
			# PLAN_SUBCLASSING_VTABLES_COM.md) - construction heap-allocates
			# (like RCClass's own ClassName(...), via the same real
			# sys.alloc[T] path - see _schedule_interface_construction) and
			# produces a Ptr[T], not a bare T. Unlike RCClass, this is an
			# EXPLICIT Ptr[T] in the metalpy type system (not an invisible-
			# pointer convention) - self is ALSO always Ptr[T] for the same
			# reason (see lower_function's own self_param construction)
			ptr_cls = self.lowering.discovery.get_intrinsics()['Ptr']
			ptr_type = self.lowering.discovery._get_or_create_specialization( ptr_cls, [ target_cls ] )
			dest = self._new_temp( expected_type or ptr_type )
			self.lowering.schedule( dest.type )
			self.lowering._schedule_interface_construction( target_cls )
			self._emit( ir.Allocate( dest = dest, cls = target_cls, fields = fields ))
			return dest

		dest = self._new_temp( expected_type or target_cls )
		# dest.type can be a concrete Specialization (ResultPayload[i32,
		# OverflowError], inferred from the substituted field type this
		# construction call is being assigned into - see the field.type
		# comment above) even though target_cls itself (this call's own
		# bare `ResultPayload` reference) is always the abstract base - the
		# concrete Specialization needs its own explicit schedule() here,
		# same as _emit_generic_call already does for a generic FUNCTION's
		# own monomorphized return type; nothing else would ever schedule it
		self.lowering.schedule( dest.type )
		if isinstance( target_cls, RCClass ):
			self.lowering._schedule_rcclass_construction( target_cls, dest.type )
		self._emit( ir.Allocate( dest = dest, cls = target_cls, fields = fields ))
		return dest

	def _try_lower_allocate_call( self, node: ast.Call, expected_type: Type|None ) -> ir.Temp|None:
		# Class.__allocate__(field=value, ...) - a compiler-synthesized
		# pseudo-method (SYNTAX.md: "strictly private... can only be called
		# from methods inside the same class"), not a real declared method,
		# so it can never be found via the ordinary _resolve_callee/
		# _attr_lookup_callable path (it's never in any class's .names) -
		# recognized textually here instead, same spirit as defer/errdefer
		# and the arithmetic-mode with-blocks. Returns None (not an error)
		# when this doesn't look like a __allocate__ call at all, so the
		# caller falls through to the normal call path and reports whatever
		# error is actually appropriate (e.g. "not callable")
		if not ( isinstance( node.func, ast.Attribute ) and node.func.attr == '__allocate__' ):
			return None
		target_cls = self.lowering._try_resolve_namespace( node.func.value )
		if not isinstance( target_cls, ClassLike ):
			return None

		fn = self._current_fn
		if fn is None or not target_cls.in_private_scope( fn.cls ):
			self.lowering.discovery.fail(
				f'{target_cls.qualname}.__allocate__(...) is private - only callable from a method of {target_cls.qualname} itself',
				node,
			)
		return self._lower_allocate_fields( target_cls, node, expected_type, '.__allocate__(...)' )

	def _try_lower_construct_call( self, node: ast.Call, expected_type: Type|None ) -> ir.Operand|None:
		# bare ClassName(...) - SYNTAX.md sugar. A class with no __init__ at
		# all degrades to exactly __allocate__ (field=value sugar). A class
		# WITH __init__: allocate self uninitialized, call __init__ with
		# the call site's own arguments (__init__'s OWN parameter list, NOT
		# the field=value sugar), wrap the result in Result[Foo,E] when
		# __init__ is fallible, dropping self's own refcount (but not
		# calling __del__) on Err - see RCCLASS ATTRIBUTE LIFETIME.md and
		# the approved plan. Scoped to non-subclassed RCClasses only,
		# matching lower_function's own scope check for __init__ itself.
		target_cls = self.lowering._try_resolve_namespace( node.func )

		# explicit generic-class construction via subscript - list[i32](...).
		# _try_resolve_namespace's own Subscript branch resolves this shape
		# to a Specialization (same as it always did for Name[T](...) over a
		# generic FUNCTION) - _ensure_resolved schedules that Specialization
		# the ordinary way (same discipline as node.resolved_callee/
		# resolved_construction below - compiler.py's own pipeline builds
		# the real struct from it once dequeued) and hands back the real,
		# concrete, already-monomorphized class (type_params/resolve both
		# cleared - see Monomorphizer.monomorphize_class), which every check
		# below this point already knows how to treat as an ordinary,
		# non-generic class
		if isinstance( target_cls, Specialization ) and isinstance( target_cls.base, ( RCClass, CStruct, CUnion, TaggedUnion, CEnum )):
			target_cls = self.lowering._ensure_resolved( target_cls )

		# CEnum construction: EnumName(value) is a plain cast to the
		# enum's underlying type — no allocation, no refcounting, just
		# reinterpret the raw integer as the enum type. e.g. OSError(ENOENT)
		if isinstance( target_cls, CEnum ):
			if len( node.args ) != 1 or node.keywords:
				self.lowering.discovery.fail( f'{target_cls.qualname}(...) takes exactly one positional argument: {ast.unparse(node)}', node )
			self.lowering._ensure_resolved( target_cls )
			# lower the argument directly — no arithmetic-mode semantics
			# needed here; a CEnum has exactly the same runtime
			# representation as its underlying type, so OSError(42) is
			# just the value 42 with the enum type
			return self._lower_expr( node.args[0], target_cls )

		if not isinstance( target_cls, ClassLike ):
			return None
		# target_cls is already resolved by now - _try_resolve_namespace's
		# own lookup resolves whatever it returns
		assert target_cls.resolve is None, f'internal compiler error, {target_cls=} is not fully resolved'

		resolved_construction = getattr( node, 'resolved_construction', None )
		if resolved_construction is not None:
			# type_resolver.py's own generic-construction resolution
			# (_ReferenceResolver._try_resolve_generic_construction)
			# already inferred target_cls's own concrete type args from
			# this call's arguments and built the real, monomorphized
			# class + __init__ - an ordinary, concrete Function, never a
			# Specialization, same "resolved ahead of time" discipline as
			# node.resolved_callee. Neither gets scheduled here - that
			# happens the ordinary way, right below, exactly like the
			# plain (non-generic) branch already schedules target_cls/
			# init directly
			concrete_cls, init = resolved_construction
			self.lowering.schedule( concrete_cls )
			self.lowering._ensure_resolved( init )
			self_type = concrete_cls
			args, kwargs = self._lower_call_args( init, node )
		else:
			init = target_cls.names.get( '__init__' )
			if isinstance( init, Function ):
				assert init.resolve is None, f'internal compiler error, {init.qualname} was not resolved before construction'
			if init is None:
				return self._lower_allocate_fields( target_cls, node, expected_type, '(...)' )
			if not isinstance( target_cls, RCClass ):
				# __init__ on a @cstruct/@cunion/@enum - not supported yet
				# (attribute lifetime tracking is scoped to RCClass, matching
				# RCCLASS ATTRIBUTE LIFETIME.md's own title) - falls through to
				# the normal call path, same "not callable" as always
				return None
			if not isinstance( init, Function ):
				self.lowering.discovery.fail( f'{target_cls.qualname}.__init__ is overloaded - not supported yet: {ast.unparse(node)}', node )
			# a subclass's own __init__ (found here via a FLAT, own-class-
			# only lookup - deliberate, matches Python's "an override fully
			# replaces the inherited one, callers never see both" semantics)
			# is allowed to chain to its base via super().__init__(...) now
			# (see FunctionLowering._lower_super_init_if_required) - no
			# rejection needed here anymore (RCClass-subclassing plan Phase 2)
			self._check_rcclass_fully_implemented( target_cls, node, '(...)' )

			if target_cls.type_params:
				self_type, init, args, kwargs = self._lower_generic_construction_args( node, target_cls, init, expected_type )
			else:
				self.lowering.schedule( target_cls )
				self.lowering._ensure_resolved( init )
				self_type = target_cls
				args, kwargs = self._lower_call_args( init, node )

		# self_temp.type is self_type - already scheduled above (schedule
		# (target_cls) for the plain case, _ensure_resolved(cls_spec) for the
		# generic case), so no separate schedule() call is needed here.
		# ir.Allocate's own `cls`, unlike self_temp.type, is always the
		# ABSTRACT target_cls - the emitter only uses it for an RCClass-vs-not
		# check, never field layout (values are in `fields`, and the mangled
		# alloc name comes from dest.type, not cls - see emitter_c.py's own
		# ir.Allocate handling)
		self_temp = self._new_temp( self_type )
		self.lowering._schedule_rcclass_construction( target_cls, self_temp.type )
		self._emit( ir.Allocate( dest = self_temp, cls = target_cls, fields = {} ))

		self.lowering.schedule( init.return_type )
		for param in init.parameters or []:
			self.lowering.schedule( param.type )

		if not self.lowering._init_fallibility( init ):
			self._emit( ir.Call( dest = None, target = init, receiver = self_temp, args = args, kwargs = kwargs ))
			return self_temp
		return self._emit_fallible_construction( node, self_type, init, self_temp, args, kwargs, expected_type )

	def _lower_and_infer_call_args(
		self, node: ast.Call, callee: Function, type_params: list[TypeVar], bindings: dict[int,Type], qualname: str,
	) -> tuple[list[ir.Operand],dict[str,ir.Operand]]:
		# shared by _lower_generic_construction_args/_lower_class_generic_
		# method_call - both need to lower a call's arguments against a
		# callee whose own class type params aren't fully bound yet, then
		# use those SAME arguments' real lowered types to refine `bindings`
		# further. Matches call args against callee's own ABSTRACT
		# parameter list, lowers each one with an expected-type hint built
		# from `bindings` as pinned SO FAR (a class type param not yet
		# bound just passes its own bare TypeVar through -
		# _substitute_type_params leaves anything it doesn't recognize
		# alone, so an unbound param position simply gets no useful hint,
		# same as today), applies move hooks, then unifies each argument's
		# own real lowered type against its declared parameter type,
		# mutating `bindings` in place. The caller still owns everything
		# after that (checking for a still-missing binding, building the
		# final concrete arg list/Specialization) - that part differs too
		# much between callers (construction pins from a possibly-
		# Result[_,_]-wrapped expected_type via _result_shape; a class
		# method's own receiver-vs-static distinction) to fold in here too
		positional, keyword = self.lowering._match_call_args( callee, node )
		partial_args = [ bindings.get( id( tv ), tv ) for tv in type_params ]
		args = [
			self._lower_expr( expr, self.lowering._substitute_type_params( param.type, type_params, partial_args ))
			for param, expr in positional
		]
		kwargs = {
			param.stem: self._lower_expr( expr, self.lowering._substitute_type_params( param.type, type_params, partial_args ))
			for param, expr in keyword
		}
		for ( param, _expr ), operand in zip( positional, args ):
			self._apply_move_hook( param, operand, qualname )
		for param, _expr in keyword:
			self._apply_move_hook( param, kwargs[param.stem], qualname )
		for ( param, _expr ), operand in zip( positional, args ):
			self.lowering._unify_type_param( type_params, param.type, operand.type, bindings, node, qualname )
		for param, _expr in keyword:
			self.lowering._unify_type_param( type_params, param.type, kwargs[param.stem].type, bindings, node, qualname )
		return args, kwargs

	def _lower_generic_construction_args( self, node: ast.Call, target_cls: RCClass, init: Function, expected_type: Type|None ) -> tuple[RCClass|Specialization,Function,list[ir.Operand],dict[str,ir.Operand]]:
		# Box(...) where Box is generic: target_cls's own concrete type args
		# have to be pinned down before __init__ can be called - same two-
		# phase strategy _lower_class_generic_method_call's own inference
		# branch uses (unify from expected_type first, then refine from the
		# lowered arguments' own types), since a class constructor's type
		# params are exactly as inferable as a generic method's - working
		# against __init__'s ABSTRACT parameter list throughout (substituting
		# per-parameter via _substitute_type_params) because the concrete,
		# monomorphized __init__ can only be built once the generic type
		# params are inferred from the call's arguments. init's own Function
		# body (parameters/return_type) was already resolved by the caller.
		assert init.resolve is None, f'internal compiler error, {init.qualname} was not resolved before construction'
		class_type_params = target_cls.type_params or []
		bindings: dict[int,Type] = {}
		# expected_type pins target_cls's own args directly for a non-
		# fallible __init__ (b: Box[i32] = Box(1)) - but for a FALLIBLE one,
		# Box(...) itself becomes Result[Box[i32],E] (SYNTAX.md), so the
		# surrounding annotation is r: Result[Box[i32],MyError], one level
		# removed from target_cls. Peek through a Result[_,_] wrapper via
		# _result_shape, which speculatively no-ops (rather than hard-
		# failing) when this particular construction isn't Result-shaped
		# at all, same posture as _maybe_consume_result
		pinning_type = expected_type
		shape = self.lowering._type_resolver._result_shape( expected_type )
		if shape is not None:
			pinning_type = shape[0]
		if isinstance( pinning_type, Specialization ) and pinning_type.base is target_cls:
			for tv, arg in zip( class_type_params, pinning_type.args ):
				bindings[ id( tv ) ] = arg

		args, kwargs = self._lower_and_infer_call_args( node, init, class_type_params, bindings, target_cls.qualname )

		missing = [ tv.stem for tv in class_type_params if id( tv ) not in bindings ]
		if missing:
			self.lowering.discovery.fail(
				f'{target_cls.qualname}(...): cannot infer type parameter(s) {", ".join(missing)} from these arguments or the surrounding expected type: {ast.unparse(node)}',
				node,
			)
		concrete_args = [ bindings[id(tv)] for tv in class_type_params ]
		cls_spec = self.lowering.discovery._get_or_create_specialization( target_cls, concrete_args )
		init_spec = self.lowering.discovery._get_or_create_specialization( init, concrete_args )
		self.lowering._ensure_resolved( cls_spec ) # also populates init_spec.monomorphized as a side effect - same (init, concrete_args) key monomorphize_class's own method-substitution loop uses
		monomorphized_init = self.lowering._ensure_resolved( init_spec )
		return cls_spec, monomorphized_init, args, kwargs

	def _emit_fallible_construction(
		self, node: ast.Call, concrete_cls: RCClass|Specialization, init: Function, self_temp: ir.Temp,
		args: list[ir.Operand], kwargs: dict[str,ir.Operand], expected_type: Type|None,
	) -> ir.Operand:
		# __init__ is fallible (Result[None,E]) - Foo(...) becomes
		# Result[Foo,E] (SYNTAX.md). The actual Ok/Err wrapping reuses REAL
		# Result.Ok/Result.Err call-lowering (via synthesized AST
		# referencing hidden locals - _declare_hidden_local, the same
		# technique the for-loop scaffolding already uses) rather than
		# hand-building ResultPayload's own internal shape here - only the
		# branch structure itself (and self_temp's own decref on Err, not
		# expressible as source syntax) is raw IR, mirroring
		# _lower_conditional_dispatch's own style
		init_result = self._new_temp( init.return_type )
		self._emit( ir.Call( dest = init_result, target = init, receiver = self_temp, args = args, kwargs = kwargs ))

		# track_result=False throughout this method's own hidden locals -
		# result_var/dest_var are compiler-internal Result-typed scaffolding
		# (see cfg.assign()'s own comment): result_var's is_err-ness is
		# already unconditionally checked right below by the synthesized
		# branch itself (that's the whole point of this method), and
		# dest_var is just a relay for whichever of ok_value/err_value wins -
		# the REAL obligation lands on whatever binding the OUTER `Foo(...)`
		# expression's own result gets assigned into, tracked normally there
		unique = self._label_id
		self_var = self._declare_hidden_local( f'__ctor_self_{unique}', concrete_cls, node )
		for instr in self._cfg_assign( self_var, self_temp, is_alias = False, node = node, track_result = False ):
			self._emit( instr )
		self._emit( ir.Assign( dest = self_var, src = self_temp ))

		result_var = self._declare_hidden_local( f'__ctor_result_{unique}', init.return_type, node )
		for instr in self._cfg_assign( result_var, init_result, is_alias = False, node = node, track_result = False ):
			self._emit( instr )
		self._emit( ir.Assign( dest = result_var, src = init_result ))

		# _result_shape, not init.return_type.args[1] directly -
		# init.return_type may already be the real, monomorphized Result
		# object itself (not a Specialization wrapper) - see Monomorphizer.
		# origin_of's own docstring. Guaranteed to succeed here: the only
		# caller (_try_lower_construct_call) already confirmed init is
		# fallible (Result[None,_]-shaped) via _init_fallibility before
		# ever reaching this method
		error_cls = self.lowering._type_resolver._result_shape( init.return_type )[1]
		result_cls = self.lowering.discovery.find_name( 'Result', node )
		outer_result_type = expected_type or self.lowering.discovery._get_or_create_specialization( result_cls, [ concrete_cls, error_cls ] )
		dest_var = self._declare_hidden_local( f'__ctor_dest_{unique}', outer_result_type, node )

		is_err_fn = self.lowering._attr_lookup_callable( init.return_type, 'is_err', node )
		self.lowering._ensure_resolved( is_err_fn )
		bool_cls = self.lowering.discovery.find_name( 'bool', node )
		is_err_temp = self._new_temp( bool_cls )
		self._emit( ir.Call( dest = is_err_temp, target = is_err_fn, receiver = result_var, args = [], kwargs = {} ))

		err_label = self._new_label( 'ctor_err' )
		end_label = self._new_label( 'ctor_end' )
		# JumpIfTrue, not JumpIfFalse - is_err_temp holds is_err()'s own
		# result, so a jump-to-err has to fire when it's TRUE (a bug found
		# while prototyping Phase 2's fallible super().__init__() chaining -
		# JumpIfFalse here meant "not an error -> jump to the error branch",
		# inverted, for EVERY fallible RCClass __init__ in the language, not
		# just a subclassed one - confirmed on a clean checkout before any
		# RCClass-subclassing work, so unrelated to it beyond being how it
		# was found)
		self._emit( ir.JumpIfTrue( cond = is_err_temp, target = err_label ))

		# Ok branch: self is fully constructed - hand it off
		ok_expr = ast.Call(
			func = ast.Attribute( value = ast.Name( id = 'Result', ctx = ast.Load() ), attr = 'Ok', ctx = ast.Load() ),
			args = [ ast.Name( id = self_var.stem, ctx = ast.Load() ) ], keywords = [],
		)
		ast.copy_location( ok_expr, node )
		ok_value = self._lower_expr( ok_expr, outer_result_type )
		for instr in self._cfg_assign( dest_var, ok_value, is_alias = False, node = node, track_result = False ):
			self._emit( instr )
		self._emit( ir.Assign( dest = dest_var, src = ok_value ))
		self._emit( ir.Jump( target = end_label ))

		# Err branch: self never became valid - drop its own refcount
		# (but __del__ is never invoked on it - SYNTAX.md), propagate the
		# same error, re-wrapped for THIS construction's own Result[Foo,E]
		self._emit( ir.Label( name = err_label ))
		self._emit( ir.Decref( value = self_var ))
		err_expr = ast.Call(
			func = ast.Attribute( value = ast.Name( id = 'Result', ctx = ast.Load() ), attr = 'Err', ctx = ast.Load() ),
			args = [ ast.Attribute(
				value = ast.Attribute( value = ast.Name( id = result_var.stem, ctx = ast.Load() ), attr = 'data', ctx = ast.Load() ),
				attr = 'v_Err', ctx = ast.Load(),
			) ], keywords = [],
		)
		ast.copy_location( err_expr, node )
		err_value = self._lower_expr( err_expr, outer_result_type )
		for instr in self._cfg_assign( dest_var, err_value, is_alias = False, node = node, track_result = False ):
			self._emit( instr )
		self._emit( ir.Assign( dest = dest_var, src = err_value ))
		self._emit( ir.Label( name = end_label ))
		return dest_var

	def _try_lower_scalar_construct_call( self, node: ast.Call, expected_type: Type|None ) -> ir.Operand|None:
		# ScalarName(x) - Python's own int(x)/float(x)-style constructor-as-
		# cast idiom. Deliberately NOT routed through _try_lower_construct_call
		# (ClassLike-only: its Allocate/self/RC-fallible-construction machinery
		# is meaningless for a scalar - no self to allocate, no attributes, no
		# refcounting)
		target_cls = self.lowering._try_resolve_namespace( node.func )
		if not isinstance( target_cls, Scalar ):
			return None
		if len( node.args ) != 1 or node.keywords:
			self.lowering.discovery.fail( f'{target_cls.qualname}(...) takes exactly one argument: {ast.unparse(node)}', node )
		arg_node = node.args[0]
		if isinstance( arg_node, ast.Constant ):
			return self._lower_scalar_cast( target_cls, arg_node, node )
		operand = self._lower_expr( arg_node, None )
		if isinstance( operand.type, Scalar ):
			# the real motivating case (u32(s.byte_len())) - same
			# arithmetic-mode-respecting logic compiler.cast(...) uses, no
			# dunder dispatch needed: one compiler primitive already
			# covers every Scalar-to-Scalar pair uniformly
			return self._lower_scalar_cast( target_cls, operand, node )
		# a non-Scalar source (e.g. an RCClass) - this is where library-
		# authored extensibility (Scalar.names, see discovery.py's
		# visit_Assign) actually earns its keep: a future
		# `SomeClass.__u32__(self) -> u32: ...` is dispatched here exactly
		# like any other method call
		dunder = self.lowering._find_method( operand.type, f'__{target_cls.stem}__' )
		if dunder is None:
			self.lowering.discovery.fail(
				f'{operand.type.qualname if operand.type else "?"} has no __{target_cls.stem}__ method - cannot convert to {target_cls.qualname}: {ast.unparse(node)}',
				node,
			)
		self.lowering._ensure_resolved( dunder )
		self.lowering.schedule( dunder.return_type )
		dest = self._new_temp( expected_type or dunder.return_type )
		self._emit( ir.Call( dest = dest, target = dunder, receiver = operand, args = [], kwargs = {} ))
		return dest

	def _try_lower_indirect_call( self, node: ast.Call, expected_type: Type|None ) -> ir.Operand|None:
		# eq_fn(a, b) where eq_fn: Ptr[Callable[[A,B],R]] - a call THROUGH a
		# function-pointer VALUE, not a named Function/method lookup at all
		# (see PLAN_CALLABLE.md) - _resolve_callee has no way to express
		# this (it only ever returns a Function/Overload/Specialization/
		# _ReceiverDispatch, never an arbitrary Operand), so it's
		# recognized here instead, same "try a shape, None means try the
		# next one" convention as the construction recognizers above.
		# Scoped to a bare Name callee for now - the only shape dict[K,V]/
		# RawDict's own generated code needs (a Ptr[Callable[...]]-typed
		# PARAMETER called directly); a general expression callee (e.g.
		# some_struct.get_callback()(...)) would need care to evaluate it
		# exactly once, deferred until something actually needs it
		if not isinstance( node.func, ast.Name ):
			return None
		name = self.lowering.discovery.find_name_or_none( node.func.id )
		if not isinstance( name, Variable ):
			return None
		self.lowering._ensure_resolved( name )
		fn_type = self.lowering._type_resolver._callable_type_of( name.type )
		if fn_type is None:
			return None
		if any( isinstance( a, ast.Starred ) for a in node.args ):
			self.lowering.discovery.fail( f'*args not supported yet: {ast.unparse(node)}', node )
		if node.keywords:
			self.lowering.discovery.fail( f'a Callable[...] call takes no keyword arguments: {ast.unparse(node)}', node )
		if len( node.args ) != len( fn_type.arg_types ):
			self.lowering.discovery.fail(
				f'{node.func.id}(...) takes {len(fn_type.arg_types)} argument(s), got {len(node.args)}: {ast.unparse(node)}',
				node,
			)
		target = self._lower_expr( node.func, None )
		args = [ self._lower_expr( arg_node, arg_type ) for arg_node, arg_type in zip( node.args, fn_type.arg_types ) ]
		return self._emit_call_indirect( target, args, fn_type.return_type, expected_type )

	def _emit_call_indirect( self, target: ir.Operand, args: list[ir.Operand], return_type: Type, expected_type: Type|None ) -> ir.Operand:
		# shared by _try_lower_indirect_call/_try_lower_closure_call - a
		# NoneType-returning target (Callable[[...],None]/Closure[[...],
		# None]) needs dest=None in the emitted ir.CallIndirect (matching
		# ir.Call's own void-return convention - emitter_c.py's C
		# expression for the call itself has C type void there, and
		# `t = (void)(...)` is a real, confirmed compile error, not just
		# reasoning), but the CALLER here (a construction-sugar recognizer,
		# "None means try the next one") still needs to return a real,
		# non-None ir.Operand to signal "matched" - ir.Const(NoneType,
		# None) is a real value, the same representation an explicit
		# `x = None` already produces, distinct from Python's own None
		none_type = self.lowering.discovery.get_none_type()
		if return_type is none_type:
			self._emit( ir.CallIndirect( dest = None, target = target, args = args ))
			return ir.Const( type = none_type, value = None )
		dest = self._new_temp( expected_type or return_type )
		self._emit( ir.CallIndirect( dest = dest, target = target, args = args ))
		return dest

	def _try_lower_closure_call( self, node: ast.Call, expected_type: Type|None ) -> ir.Operand|None:
		# my_closure(a, b) where my_closure: Closure[[A,B],R] - a call
		# THROUGH a closure VALUE (see _lower_bound_method_closure). Same
		# "try a shape, None means try the next one" convention as
		# _try_lower_indirect_call just above, scoped to a bare Name callee
		# for the identical reason (a general expression callee needs care
		# to evaluate it exactly once - deferred there too)
		if not isinstance( node.func, ast.Name ):
			return None
		name = self.lowering.discovery.find_name_or_none( node.func.id )
		if not isinstance( name, Variable ):
			return None
		self.lowering._ensure_resolved( name )
		closure_type = name.type
		if not isinstance( closure_type, ClosureType ):
			return None
		self.lowering._ensure_resolved( closure_type )
		if any( isinstance( a, ast.Starred ) for a in node.args ):
			self.lowering.discovery.fail( f'*args not supported yet: {ast.unparse(node)}', node )
		if node.keywords:
			self.lowering.discovery.fail( f'a closure call takes no keyword arguments: {ast.unparse(node)}', node )
		if len( node.args ) != len( closure_type.arg_types ):
			self.lowering.discovery.fail(
				f'{node.func.id}(...) takes {len(closure_type.arg_types)} argument(s), got {len(node.args)}: {ast.unparse(node)}',
				node,
			)
		closure_operand = self._lower_expr( node.func, None )
		args = [ self._lower_expr( arg_node, arg_type ) for arg_node, arg_type in zip( node.args, closure_type.arg_types ) ]

		fn_field = closure_type.get_local( 'fn' )
		self_field = closure_type.get_local( 'self' )
		fn_operand = self._new_temp( fn_field.type )
		self._emit( ir.GetAttr( dest = fn_operand, obj = closure_operand, attr = 'fn' ))
		self_operand = self._new_temp( self_field.type )
		self._emit( ir.GetAttr( dest = self_operand, obj = closure_operand, attr = 'self' ))

		# fn is Ptr[None] (type-erased) on the closure struct itself - cast
		# back to the trampoline's real (Ptr[None], *ArgTypes) -> RetType
		# shape before calling through it, exactly mirroring how it was
		# erased going IN (_lower_bound_method_closure's own ir.CastWrap)
		ptr_cls = self.lowering.discovery.get_intrinsics()['Ptr']
		trampoline_callable_type = self.lowering.discovery._get_or_create_callable_type(
			[ fn_field.type, *closure_type.arg_types ], closure_type.return_type,
		)
		trampoline_ptr_type = self.lowering.discovery._get_or_create_specialization( ptr_cls, [ trampoline_callable_type ] )
		fn_cast = self._new_temp( trampoline_ptr_type )
		self._emit( ir.CastWrap( dest = fn_cast, operand = fn_operand ))

		return self._emit_call_indirect( fn_cast, [ self_operand, *args ], closure_type.return_type, expected_type )

	def _lower_or_return( self, node: ast.Call, receiver: ir.Operand, want_result: bool ) -> ir.Operand|None:
		# <result_expr>.or_return() is recognized textually here rather than
		# ever actually calling Result.or_return's own declared body
		# (`if self.is_err(): compiler.early_return(self.data.v_Err)` /
		# `return self.data.v_Ok`) - that body is written as a spec of the
		# intended behavior, not something literally compilable: it needs to
		# trigger a `return Result.Err(...)` in ITS CALLER's scope, not its
		# own (or_return's own declared return type is bare T, not
		# Result[T,E], so `return Result.Err(...)` from inside it could
		# never type-check there - see compiler.early_return's own comment).
		# This expands directly to the same OrReturn/OrJump primitives
		# checked-arithmetic already uses for exactly the same "propagate
		# the error to the enclosing function, continue with the unwrapped
		# value" shape - no new IR needed, and Result.or_return is never
		# scheduled/lowered as a real function as a result.
		if node.args or node.keywords:
			self.lowering.discovery.fail( f'or_return() takes no arguments: {ast.unparse(node)}', node )
		shape = self.lowering._type_resolver._result_shape( receiver.type )
		if shape is None:
			self.lowering.discovery.fail( f'or_return() receiver must be Result[_,_], got {receiver.type.qualname if receiver.type else "?"}', node )
		result_type, error_cls = shape
		# find_name, not receiver.type.base - see _maybe_consume_result's
		# identical comment on why
		result_cls = self.lowering.discovery.find_name( 'Result', node )
		self.lowering._type_resolver._require_result_return( node, result_cls, error_cls, self.lowering._OR_RETURN_ALTERNATIVES, fn = self._current_fn )
		# clearing receiver (when it's a named Variable) and validating that
		# nothing ELSE is still unchecked at this early-exit point both now
		# live in _consume_checked_result itself, shared with checked-
		# arithmetic's own identical OrReturn/OrJump early-exit - see its
		# own comment
		unwrapped = self._consume_checked_result( node, receiver, result_type, extra = None )
		return unwrapped if want_result else None

	def _lower_call_args( self, target: Function, node: ast.Call ) -> tuple[list[ir.Operand],dict[str,ir.Operand]]:
		# shared by the plain call path (_lower_call's own else branch) and
		# _lower_generic_function_call: lowers positional/keyword args
		# straight against target's own already-concrete declared parameter
		# types, applying each param's move hook as it goes. NOT reused by
		# _lower_inferred_generic_call or _lower_class_generic_method_call -
		# both of those still need to INFER target's type params before a
		# parameter type is concrete enough to lower an argument against (in
		# _lower_inferred_generic_call's case, args are lowered with no
		# expected type at all, and move hooks apply in a separate pass
		# afterward instead), so forcing them through this helper would
		# change what expected_type each argument actually gets
		positional, keyword = self.lowering._match_call_args( target, node )
		args = []
		for param, expr in positional:
			operand = self._lower_expr( expr, param.type )
			self._apply_move_hook( param, operand, target.qualname )
			args.append( operand )
		kwargs = {}
		for param, expr in keyword:
			operand = self._lower_expr( expr, param.type )
			self._apply_move_hook( param, operand, target.qualname )
			kwargs[param.stem] = operand
		# fill in default values for any parameter that was not
		# explicitly provided by the call site (e.g. print(msg,
		# end='\n') called as print('hello') — end gets its
		# default lowered here as if the caller had passed it)
		given = { param.stem for param, _ in positional }
		given.update( kwargs.keys() )
		for param in target.parameters or []:
			if param.stem not in given and param.default is not None:
				default_operand = self._lower_expr( param.default, param.type )
				kwargs[param.stem] = default_operand
		return args, kwargs

	def _lower_inline_call( self, node: ast.Call, target: Function, receiver: ir.Operand|None, args: list[ir.Operand], kwargs: dict[str,ir.Operand], expected_type: Type|None, want_result: bool ) -> ir.Operand|None:
		# PLAN_INLINE.md - target.is_inline: splice target's own single
		# `return <expr>` body directly here instead of ever emitting a
		# real ir.Call. `args`/`kwargs` are already-lowered operands (the
		# caller already ran _lower_call_args, or the interleaved generic
		# lower_and_unify - same move-hook/argument-lowering either way,
		# only the tail differs). target may be a plain Function, OR an
		# already-monomorphized one (target.node was deep-copied per
		# Specialization by monomorphize.py - see its own docstring), so
		# target.node.body is always safe to read directly here regardless
		# of which caller reached this
		if id( target ) in self._inlining_stack:
			self.lowering.discovery.fail(
				f'@inline {target.qualname}: recursive inlining (directly or through another @inline function) is not supported: {ast.unparse(node)}',
				node,
			)
		if not want_result and cfg.is_result_type( target.return_type ):
			# same discard check the ordinary call tails already apply -
			# discovery.py's _is_inline_eligible_body already guarantees
			# target.node.body is exactly one `return <expr>`, so this can't
			# be sidestepped by inlining instead of calling for real
			self.lowering.discovery.fail(
				f'{target.qualname}(...) returns a Result that is discarded here - '
				f'assign it to a name and use .is_ok(), .is_err(), .or_return(), .unwrap(msg), or match: {ast.unparse(node)}',
				node,
			)
		bindings: dict[str,ir.Operand] = {} if receiver is None else { 'self': receiver }
		for i, param in enumerate( target.parameters or [] ):
			bindings[param.stem] = args[i] if i < len( args ) else kwargs[param.stem]

		# each binding becomes a REAL local Variable, registered under its
		# ordinary name ('self', a parameter's own stem) directly into
		# target.names - not just an _expr_Name-level shortcut - because
		# discovery.find_name is reached from more than one place while
		# lowering a Call (e.g. _try_resolve_namespace, used by the
		# construction-call recognizers to probe whether `self.foo(...)`
		# might be construction sugar, BEFORE ordinary attribute/method
		# resolution ever runs) - anything less than a real registry entry
		# left those other paths seeing an unresolved 'self'/param name
		# (confirmed by a real repro, not just reasoning: self.__len__()
		# inside an inlined body failed exactly this way, from inside a
		# construction-sugar probe, not from _expr_Name at all).
		#
		# the Variable's own .stem (what emitter_c.py actually declares as
		# a C local, keyed by NAME not by object identity - see its own
		# "declared" set) is deliberately NOT 'self'/the parameter's own
		# stem - reusing those would silently collide with and overwrite
		# the ENCLOSING function's own real `self`/parameter of the same
		# name the moment one method's @inline body gets spliced into
		# another method's own body. _inline_binding_id makes every
		# splice's own bindings unique instead.
		#
		# no _cfg_assign/incref here, deliberately - this must behave
		# exactly like an ordinary (non-@move) function parameter already
		# does at a REAL call boundary: borrowed, no incref at the
		# boundary, no independent decref responsibility (the caller's own
		# argument operand keeps whatever cleanup it already had, e.g. an
		# argument Temp's own DeleteTemp - untouched by any of this). A
		# bare ir.Assign against a fresh Variable is exactly that: a named
		# alias for the call's own duration, nothing more.
		#
		# when the operand is ALREADY a Variable (by far the common case -
		# a bare-name receiver/argument, e.g. b.get_len()/some_result.
		# is_ok()), it's registered directly, no fresh copy and no Assign
		# at all - true zero overhead, and what makes the "compiles
		# identically to writing the callee's body directly at the call
		# site" guarantee exact, not just "close". Only a genuinely
		# computed operand (a Temp from a sub-expression like make_box().
		# get_len(), or a Const) needs the synthesized-local fallback -
		# both to give it a referenceable name at all (Temp/Const aren't
		# Name subtypes, discovery.find_name's registry requires one - see
		# above) and to guarantee it's evaluated exactly once even if the
		# spliced body references self/that parameter more than once
		saved: dict[str,object] = {}
		for stem, operand in bindings.items():
			if isinstance( operand, Variable ):
				fresh = operand
			else:
				fresh = Variable(
					stem = f'$inline{self._inline_binding_id}${stem}',
					qualname = f'{target.qualname}$$inline{self._inline_binding_id}${stem}',
					file = target.file, line = target.line,
					type = operand.type,
				)
				self._inline_binding_id += 1
				self._emit( ir.Assign( dest = fresh, src = operand ))
			saved[stem] = target.names.get( stem )
			target.names[stem] = fresh

		# discovery.py's _is_inline_eligible_body already guaranteed
		# target.node.body is exactly one `return <expr>`, optionally
		# preceded by a docstring - the Return is always the LAST statement
		# either way, so no need to re-strip the docstring here
		return_expr = target.node.body[-1].value
		module = self.lowering._find_module_for( target )
		self._inlining_stack.append( id( target ))
		try:
			with self.lowering.discovery.module_context( module ):
				with self.lowering.discovery.scope_context( target ):
					result = self._lower_expr( return_expr, expected_type or target.return_type )
		finally:
			self._inlining_stack.pop()
			for stem, old in saved.items():
				if old is None:
					target.names.pop( stem, None )
				else:
					target.names[stem] = old
		return result if want_result else None

	def _lower_generic_function_call( self, node: ast.Call, spec: Specialization, receiver: ir.Operand|None, expected_type: Type|None, want_result: bool ) -> ir.Operand|None:
		# sys.alloc[u8](...) - explicit generic instantiation. Matches call
		# args against the MONOMORPHIZED signature (so a literal argument's
		# expected type is already concrete, e.g. usize for alloc[u8]'s
		# count - not the abstract, unsubstituted one)
		monomorphized = self.lowering._monomorphized_function( spec )
		args, kwargs = self._lower_call_args( monomorphized, node )
		if monomorphized.is_inline:
			return self._lower_inline_call( node, monomorphized, receiver, args, kwargs, expected_type, want_result )
		return self._emit_generic_call( node, spec, monomorphized, receiver, args, kwargs, expected_type, want_result )

	def _lower_inferred_generic_call( self, node: ast.Call, target: Function, receiver: ir.Operand|None, expected_type: Type|None, want_result: bool ) -> ir.Operand|None:
		# a BARE call to a generic function (mylen(a), no explicit [T]) -
		# unlike _lower_generic_function_call, there's no already-concrete
		# Specialization to match args against yet: T has to be inferred
		# FROM the arguments themselves first. Each argument is lowered in
		# turn (positional, then keyword - same order _lower_and_infer_
		# call_args uses) with an expected-type hint built from whatever
		# bindings EARLIER arguments in this same call already solved (a
		# still-unbound type param just passes its own bare TypeVar through
		# - _substitute_type_params leaves anything it doesn't recognize
		# alone, same as no hint at all), then immediately unified
		# (_unify_type_param) against its declared parameter type to refine
		# bindings before the NEXT argument is lowered - the inverse of
		# _substitute_type_params, which already handles substituting a
		# SOLVED binding through arbitrarily nested Specializations (list[T]
		# etc), so unification mirrors that same recursive shape instead of
		# only handling a bare `t: T` parameter. This interleaving (rather
		# than lowering everything first, then unifying everything after) is
		# what lets a LATER Callable[[T],K]-typed argument's own lambda body
		# see T already bound from an EARLIER plain argument, even though K
		# itself is still only inferable from the lambda's own body
		# (PLAN_LAMBDA.md, "eager lambda lowering") - a bare TypeVar
		# parameter still offers no useful literal hint on its own, so a
		# literal argument at a position nothing's bound yet still correctly
		# fails via _expr_Constant's own "cannot infer" error, same as before
		assert target.resolve is None, f'internal compiler error - {target=} was not fully resolved by the type_resolver module'
		positional, keyword = self.lowering._match_call_args( target, node )
		type_params = target.type_params or []
		bindings: dict[int,Type] = {} # id(TypeVar) -> the concrete Type it was inferred as

		def lower_and_unify( param: Parameter, expr: ast.expr ) -> ir.Operand:
			partial_args = [ bindings.get( id( tv ), tv ) for tv in type_params ]
			hint = self.lowering._substitute_type_params( param.type, type_params, partial_args )
			if isinstance( hint, TypeVar ) and any( hint is tv for tv in type_params ):
				# substitution left the hint as a BARE, still-unbound type
				# param (nothing bound it yet) - not a real hint, same as no
				# hint at all (_expr_Constant's own "cannot infer" error is
				# the correct outcome for a literal here, exactly as before
				# this method started interleaving lower+unify). A hint that
				# came back PARTIALLY substituted (e.g. Ptr[Callable[[i32],K]]
				# - T bound, K still bare) is fine as-is and reaches here
				# unchanged - only a hint that IS one of type_params, bare,
				# needs this fallback
				hint = None
			operand = self._lower_expr( expr, hint )
			self.lowering._unify_type_param( type_params, param.type, operand.type, bindings, node, target.qualname )
			return operand

		args = [ lower_and_unify( param, expr ) for param, expr in positional ]
		kwargs = { param.stem: lower_and_unify( param, expr ) for param, expr in keyword }
		for ( param, _expr ), operand in zip( positional, args ):
			self._apply_move_hook( param, operand, target.qualname )
		for param, _expr in keyword:
			self._apply_move_hook( param, kwargs[param.stem], target.qualname )

		missing = [ tv.stem for tv in target.type_params or [] if id( tv ) not in bindings ]
		if missing:
			self.lowering.discovery.fail(
				f'{target.qualname}[...]: cannot infer type parameter(s) {", ".join(missing)} from these arguments - '
				f'call it explicitly as {target.qualname}[...](...) instead: {ast.unparse(node)}',
				node,
			)
		inferred_args = [ bindings[id(tv)] for tv in target.type_params or [] ]
		spec = self.lowering.discovery._get_or_create_specialization( target, inferred_args )
		monomorphized = self.lowering._monomorphized_function( spec )
		if monomorphized.is_inline:
			return self._lower_inline_call( node, monomorphized, receiver, args, kwargs, expected_type, want_result )
		return self._emit_generic_call( node, spec, monomorphized, receiver, args, kwargs, expected_type, want_result )
		# else: this parameter position doesn't mention any of type_params
		# (a concrete parameter, or a nested type whose base doesn't even
		# match the argument's) - nothing to infer here. Not an error by
		# itself: a genuine argument-type mismatch isn't checked anywhere
		# yet (no general type-checking pass exists), same as every other
		# call site in this file today

	def _emit_generic_call( self, node: ast.Call, spec: Specialization, monomorphized: Function, receiver: ir.Operand|None, args: list[ir.Operand], kwargs: dict[str,ir.Operand], expected_type: Type|None, want_result: bool ) -> ir.Operand|None:
		# schedules the Specialization itself as the compile unit (see
		# _monomorphized_function/compiler.py's own handling of it), shared
		# tail for both the explicit Name[T](...) and inferred call paths -
		# and, unlike _lower_call's OWN shared tail (which only ever sees a
		# call type_resolver.py's pre-pass could tag with resolved_callee),
		# the ONLY tail a receiver-based generic method call reaches at all
		# (see _lower_class_generic_method_call's own comment on why that
		# one is always left untagged) - so the discard check needs its own
		# copy here too, not just in _lower_call's
		self.lowering.schedule( spec )
		self.lowering.schedule( monomorphized.return_type )
		for param in monomorphized.parameters or []:
			self.lowering.schedule( param.type )
		if not want_result and cfg.is_result_type( monomorphized.return_type ):
			self.lowering.discovery.fail(
				f'{monomorphized.qualname}(...) returns a Result that is discarded here - '
				f'assign it to a name and use .is_ok(), .is_err(), .or_return(), .unwrap(msg), or match: {ast.unparse(node)}',
				node,
			)
		if want_result:
			dest = self._new_temp( expected_type or monomorphized.return_type )
			self._emit( ir.Call( dest = dest, target = monomorphized, receiver = receiver, args = args, kwargs = kwargs ))
			return dest
		self._emit( ir.Call( dest = None, target = monomorphized, receiver = receiver, args = args, kwargs = kwargs ))
		return None

	def _lower_class_generic_method_call( self, node: ast.Call, target: Function, receiver: ir.Operand|None, expected_type: Type|None, want_result: bool ) -> ir.Operand|None:
		# a method whose genericity is inherited from its enclosing class
		# (Result.Ok/.Err/.is_ok/.is_err/... referencing Result's own T,E)
		# rather than declared on the method itself (unlike sys.alloc[T]) -
		# target.type_params is empty, but target.cls.type_params isn't.
		#
		# only ever reached with NO receiver (a static/classmethod reached
		# via bare class name, e.g. Result.Ok(y)) - a receiver whose own
		# type already pins down cls's concrete args never gets here at all:
		# _attr_lookup_callable already hands back an already-substituted
		# Function for that case (see monomorphize.py/_ensure_resolved),
		# whose own .cls is the concrete Specialization, not the abstract
		# cls this dispatch condition (_lower_call) checks .type_params on -
		# confirmed by instrumenting this branch and running the full test
		# suite, not just by this reasoning alone. So the class's own
		# concrete type args always have to be INFERRED here, the same way
		# _lower_inferred_generic_call infers a free function's own type
		# params, with one addition: unify expected_type against the
		# method's still-abstract return type FIRST, before lowering any
		# argument - Result.Ok(val: T) ->
		# Result[T,E] never mentions E in its own parameter list at all
		# (only inferable from context), and even T needs to be known
		# BEFORE a bare literal argument (Result.Ok(5)) can be lowered at
		# all (_expr_Constant needs a real expected type, not a raw
		# TypeVar) - unlike a free generic function, where a literal
		# argument at an inferred position is simply unsupported (see
		# _lower_inferred_generic_call's own comment), the surrounding
		# expected_type is usually enough to resolve every class type
		# param here without needing the arguments' own types at all
		assert target.resolve is None, f'internal compiler error - {target=} was not fully resolved by the type_resolver module'
		cls = target.cls
		class_type_params = cls.type_params or [] if cls is not None else []
		bindings: dict[int,Type] = {}
		if expected_type is not None:
			self.lowering._unify_type_param( class_type_params, target.return_type, expected_type, bindings, node, target.qualname )

		args, kwargs = self._lower_and_infer_call_args( node, target, class_type_params, bindings, target.qualname )

		missing = [ tv.stem for tv in class_type_params if id( tv ) not in bindings ]
		if missing:
			self.lowering.discovery.fail(
				f'{target.qualname}(...): cannot infer {cls.qualname if cls else "?"} type parameter(s) '
				f'{", ".join(missing)} from these arguments or the surrounding expected type: {ast.unparse(node)}',
				node,
			)
		cls_args = [ bindings[id(tv)] for tv in class_type_params ]
		method_spec = self.lowering.discovery._get_or_create_specialization( target, cls_args )
		monomorphized = self.lowering._monomorphized_function( method_spec )
		return self._emit_generic_call( node, method_spec, monomorphized, receiver, args, kwargs, expected_type, want_result )

	def _lower_call( self, node: ast.Call, expected_type: Type|None, want_result: bool ) -> ir.Operand|None:
		match self.lowering._is_compiler_call( node ):
			case 'sizeof':
				result = self._lower_compiler_sizeof( node, expected_type )
				return result if want_result else None

			case 'is_rc':
				result = self._lower_compiler_is_rc( node, expected_type )
				return result if want_result else None

			case 'refcount':
				result = self._lower_compiler_refcount( node, expected_type )
				return result if want_result else None

			case 'cast':
				result = self._lower_compiler_cast( node, expected_type )
				return result if want_result else None

			case 'addrof':
				result = self._lower_compiler_addrof( node, expected_type )
				return result if want_result else None

			case 'atomic_load':
				result = self._lower_compiler_atomic_load( node, expected_type )
				return result if want_result else None

			case 'atomic_add':
				result = self._lower_compiler_atomic_rmw( node, expected_type, ir.AtomicRMWOp.ADD )
				return result if want_result else None

			case 'atomic_sub':
				result = self._lower_compiler_atomic_rmw( node, expected_type, ir.AtomicRMWOp.SUB )
				return result if want_result else None

			case 'atomic_exchange':
				result = self._lower_compiler_atomic_rmw( node, expected_type, ir.AtomicRMWOp.EXCHANGE )
				return result if want_result else None

			case 'atomic_compare_exchange':
				result = self._lower_compiler_atomic_compare_exchange( node, expected_type )
				return result if want_result else None

			case 'cexpr':
				result = self.lowering._lower_compiler_cexpr( node, expected_type )
				return result if want_result else None

			case 'fetch_unicode_table':
				result = self.lowering._lower_compiler_fetch_unicode_table( node )
				return result if want_result else None

		# each recognizer returns None (not an error) when this call doesn't
		# match its own construction-sugar shape at all, falling through to
		# the next; a real error inside a matched shape (e.g. a malformed
		# __allocate__ call) still raises/records normally
		construction_recognizers = (
			self._try_lower_allocate_call,
			self._try_lower_construct_call,
			self._try_lower_scalar_construct_call,
			self._try_lower_indirect_call,
			self._try_lower_closure_call,
		)
		for recognizer in construction_recognizers:
			allocate_dest = recognizer( node, expected_type )
			if allocate_dest is not None:
				return allocate_dest if want_result else None

		# type_resolver.py's own generic-call resolution
		# (_ReferenceResolver.visit_Call) may already have tagged this call
		# with its resolved, monomorphized callee (an ordinary, concrete
		# Function - never a Specialization) - when present, it's
		# authoritative and skips _resolve_callee/the Specialization
		# branches below entirely, so a call resolved there never touches
		# Specialization on this side at all. Absent (any call that pass
		# left untagged, generic or not) falls through to the exact same
		# resolution this always did
		resolved_callee = getattr( node, 'resolved_callee', None )
		if resolved_callee is not None:
			target, receiver = resolved_callee, None
		else:
			target, receiver = self._resolve_callee( node.func )
		if receiver is not None and isinstance( target, Function ) and ( target.is_static or target.is_classmethod ):
			# self.static_method(...) - _resolve_callee's own Attribute
			# fallback always computes a receiver for ANY dotted callee (it
			# has no way to know staticness before resolving the attribute
			# itself), but ir.Call.receiver's own docstring already says a
			# @staticmethod/@classmethod call takes none at all ("None for
			# a free function, staticmethod, or classmethod call") - this
			# is the one place that promise wasn't kept, and it showed up
			# as a real C compile error (an extra `self` argument at the
			# call site that the callee's own prototype never declared).
			# ClassName.static_method(...) never hits this: it already
			# resolves through _resolve_callee_target's namespace-lookup
			# path instead, which never computes a receiver in the first
			# place - only the self.-qualified spelling needs the null-out
			receiver = None
		if receiver is not None:
			self.lowering.schedule( receiver.type )

		if isinstance( target, ( Function, Overload )) and target.stem in self.lowering._RESULT_CONSUMING_METHODS and isinstance( receiver, Variable ):
			# .is_ok()/.is_err()/.unwrap(msg)/.unwrap_or(default) - like
			# or_return() above, these aren't given their own IR shape;
			# they're ordinary method dispatch (unwrap_or in particular
			# resolves through the Overload branch below, for its `default:
			# T` stub), so recognition has to happen here by stem + class
			# identity rather than at a single shared call site the way
			# or_return's own _lower_or_return is. Placed before any of the
			# dispatch branches below (rather than only in the plain-
			# Function "final else" tail) so it applies uniformly regardless
			# of which branch actually ends up lowering the call
			target_cls_base = target.cls.base if isinstance( target.cls, Specialization ) else target.cls
			if target_cls_base is self.lowering.discovery.find_name( 'Result', node ):
				self._cfg.clear_result( receiver.stem )

		if (
			isinstance( target, ( Function, Overload )) and target.stem in ( 'unwrap', 'unwrap_or' )
			and isinstance( receiver, ir.Temp )
		):
			# <chained_call>.unwrap(msg)/.unwrap_or(default) - e.g.
			# xs.__getitem__(0).unwrap(msg), never bound to a name - the
			# receiver is a bare Temp holding a Result[T,E] value whose
			# RC payload (if any) is registered in _temp_states (every
			# Call/Allocate dest with RC leaves is - see _emit's own
			# fresh_temp() call) as still needing its own eventual
			# decref if nothing else claims it first. unwrap()/unwrap_or()
			# return that SAME payload reference (their own declared
			# body is a plain `return self.data.v_Ok`, no incref) - the
			# call's OWN return value inherits ownership of it, so the
			# receiver temp's registration has to be dropped here,
			# silently (no decref emitted - _cfg.move()'s identical Temp
			# branch does exactly this), or the temp's own cleanup
			# (DeleteTemp, once nothing else in this statement still
			# needs it) decrefs the SAME reference a second time while
			# the returned value is ALSO independently tracked as owning
			# it - confirmed with a real repro + AddressSanitizer, not
			# just reasoning: exactly this shape freed a still-referenced
			# int while it was still stored in a list. is_ok()/is_err()
			# don't return the payload, so they're deliberately excluded -
			# the receiver's own eventual cleanup is still correct there.
			self._cfg.move( receiver, target_qualname = target.qualname, param_stem = 'self' )

		if isinstance( target, _ReceiverDispatch ):
			return self._lower_union_receiver_call( node, target, receiver, expected_type, want_result )

		if isinstance( target, Function ) and target.stem == 'or_return':
			# target.cls is a Specialization, not bare Result, whenever the
			# receiver already pinned concrete args (the common case, e.g.
			# some_result.or_return() where some_result: Result[i32,E]) -
			# unwrap before the identity check, or a concrete receiver's own
			# or_return() would stop being recognized at all and fall
			# through to actually CALLING Result.or_return's literal
			# declared body, which is a spec of the intended behavior, not
			# something literally compilable (see _lower_or_return's own
			# comment)
			target_cls_base = target.cls.base if isinstance( target.cls, Specialization ) else target.cls
			if target_cls_base is self.lowering.discovery.find_name( 'Result', node ):
				return self._lower_or_return( node, receiver, want_result )

		if isinstance( target, Specialization ) and isinstance( target.base, Function ):
			return self._lower_generic_function_call( node, target, receiver, expected_type, want_result )

		if isinstance( target, Function ) and target.type_params:
			return self._lower_inferred_generic_call( node, target, receiver, expected_type, want_result )

		# target.cls can legitimately BE a Specialization now (monomorphized_
		# function sets a monomorphized method's own .cls to one) - Specialization
		# has no .type_params of its own, so this must not read it directly;
		# getattr's default (None/falsy) correctly means "not this branch",
		# since a receiver that already pinned down concrete class args (the
		# only way target.cls ends up a Specialization here) already went
		# through _attr_lookup_callable's own substitution - nothing left to
		# infer
		if isinstance( target, Function ) and not target.type_params and target.cls is not None and getattr( target.cls, 'type_params', None ):
			return self._lower_class_generic_method_call( node, target, receiver, expected_type, want_result )

		if isinstance( target, Overload ):
			# a bare literal argument has no type of its own before a
			# specific implementation is chosen - _lower_overload_arg tries
			# each candidate's declared parameter type at that position,
			# using it unambiguously if exactly one is even plausible for
			# the literal's own kind (a string literal never plausibly
			# matches an i32 parameter, etc.) and failing clearly rather
			# than guessing if more than one genuinely could
			candidates = [ *target.stubs, *target.implementations ]
			for fn in candidates:
				assert fn.resolve is None, f'internal compiler error - {fn.qualname} was not resolved before overload dispatch'
			args = [ self._lower_overload_arg( a, i, None, candidates, node ) for i, a in enumerate( node.args ) ]
			if any( kw.arg is None for kw in node.keywords ):
				self.lowering.discovery.fail( f'**kwargs not supported yet: {ast.unparse(node)}', node )
			kwargs = { kw.arg: self._lower_overload_arg( kw.value, None, kw.arg, candidates, node ) for kw in node.keywords }
			arg_types = [ op.type for op in args ]
			kwarg_types = { name: op.type for name, op in kwargs.items() }

			# an @overload group declared inside a generic CLASS (e.g.
			# Result[T,E].unwrap_or's `default: T` stub) is now pre-
			# substituted by monomorphize_class itself whenever `target`
			# was reached through a concrete class specialization - see
			# Monomorphizer._substituted_overload. target.stubs/
			# .implementations are ALREADY the correctly monomorphized
			# Functions in that case (each one's own .cls a Specialization,
			# the same convention monomorphized_function already uses for
			# a single non-overloaded generic method), so no per-call-site
			# substitution is needed here at all anymore - this used to
			# reconstruct that same substitution by hand from the
			# RECEIVER's own type instead (isinstance(receiver.type,
			# Specialization)), which only worked because the receiver
			# hadn't been resolved to its real ClassLike yet; detecting
			# "was this group substituted" now just reads the already-
			# substituted candidate's own .cls, the same signal
			# monomorphized_function already exposes everywhere else
			substituted = bool( candidates ) and isinstance( candidates[0].cls, Specialization )

			def _resolve_original( fn: Function ) -> Function:
				if not substituted:
					return fn
				# when a stub won the overload resolution, `fn` is the
				# stub's own `bound_to` (the real, already-monomorphized
				# implementation) - the stub has a more specific return
				# type than the impl (e.g. T vs T|None), so use the
				# stub's return type while still calling through to the impl
				winning_stub = next( ( s for s in target.stubs if s.bound_to is fn ), None )
				if winning_stub is not None:
					return replace( fn, return_type = winning_stub.return_type )
				return fn

			try:
				branches, resolved = overload_resolution.resolve_call( target.stubs, target.implementations, arg_types, kwarg_types, qualname = target.qualname )
			except CompileError as e:
				# resolve_call is a pure function of types with no
				# AST/Discovery reference by design - it raises unrecorded,
				# this is where a location actually gets attached and it
				# lands in the collector
				self.lowering.discovery.fail( str( e ), node )
			if branches:
				branches = [ ConditionalDispatch( conditions = b.conditions, function = _resolve_original( b.function )) for b in branches ]
				resolved = _resolve_original( resolved )
				return self._lower_conditional_dispatch( node, branches, resolved, args, kwargs, expected_type, want_result )
			target = _resolve_original( resolved )
			self.lowering._ensure_resolved( target ) # resolve_call() already resolved every group member internally - this just schedules the chosen one
		else:
			self.lowering._resolve_call_target( target )
			args, kwargs = self._lower_call_args( target, node )

		if isinstance( target, Function ) and target.is_inline:
			# PLAN_INLINE.md - reaches this shared tail from either the
			# plain (non-generic, non-Overload) `else` branch above, the
			# resolved_callee pre-tag (type_resolver.py's own generic-call
			# resolution, ~ this method's own top), or a receiver-based
			# generic-class method already monomorphized by _attr_lookup_
			# callable before dispatch even started (e.g. some_result.
			# is_ok()/.is_err() - see PLAN_INLINE.md's own "traced through
			# _attr_lookup_callable" note). Never reached with is_inline set
			# from the Overload branch above - @inline+@overload is
			# rejected at discovery time, so a resolved group member is
			# never is_inline
			return self._lower_inline_call( node, target, receiver, args, kwargs, expected_type, want_result )

		self.lowering.schedule( target.return_type )
		for param in target.parameters or []:
			self.lowering.schedule( param.type )

		if not want_result and cfg.is_result_type( target.return_type ):
			# a bare `foo()` statement whose return value is a Result -
			# _stmt_Expr is the only caller that ever passes want_result=
			# False for a call used as a full statement (every other
			# _lower_call caller threads want_result through from ITS OWN
			# caller instead), so this is the "value produced, immediately
			# discarded, never even bound to a name" case from the plan's
			# validation table. v1 gap: this only covers calls that reach
			# this shared tail (plain Function targets, and Overload targets
			# that resolve to one unambiguous implementation without needing
			# _lower_conditional_dispatch) - a bare-statement call to a
			# GENERIC Result-returning function/method, or one requiring
			# runtime union-argument dispatch, isn't covered
			self.lowering.discovery.fail(
				f'{target.qualname}(...) returns a Result that is discarded here - '
				f'assign it to a name and use .is_ok(), .is_err(), .or_return(), .unwrap(msg), or match: {ast.unparse(node)}',
				node,
			)

		if want_result:
			target_return_type = target.return_type
			# a Specialization of a TaggedUnion base (e.g. an unmonomorphized
			# Result[usize,IndexError]) is just as "already the expected
			# union" as a TaggedUnion instance itself - Result's own class
			# body (discovery.py's _parse_ClassDef_TaggedUnion) makes its
			# base a TaggedUnion, so a bare isinstance( _, TaggedUnion )
			# check misses every generic-union return that hasn't been
			# individually monomorphized yet, which target_return_type
			# usually hasn't been at this point (nothing upstream forces it -
			# self.lowering.schedule() below only enqueues it for later
			# compilation)
			target_return_type_base = target_return_type.base if isinstance( target_return_type, Specialization ) else target_return_type
			if isinstance( expected_type, TaggedUnion ) and target_return_type is not None and not isinstance( target_return_type_base, ( TaggedUnion, TypeVar )):
				# a call whose own return type is a plain leaf (e.g. str)
				# flowing into a T|None-typed slot - dest must be typed as
				# target_return_type (what the emitted ir.Call's C signature
				# actually returns), not expected_type, or dest's C
				# declaration wouldn't match the value assigned into it.
				# _lower_expr's own post-hoc coercion (_coerce_into_union,
				# right after this call returns) then wraps it into the union
				# - see its own comment. Deliberately narrower than "prefer
				# target_return_type whenever it's concrete": when
				# target_return_type is ITSELF (a Specialization of) a
				# TaggedUnion (e.g. a Result[usize,IndexError]-returning call
				# assigned into an already Result[usize,IndexError]-typed
				# local), the two are the same union from context but not
				# necessarily the same object - Specialization instances for
				# one generic instantiation aren't interned across
				# independent resolutions, so forcing dest to expected_type
				# there (the `else` below, unchanged from before this fix)
				# keeps dest identity-compatible with whatever already
				# expects it, instead of tripping _coerce_into_union's
				# identity check with a whole (non-leaf) union value it
				# would wrongly treat as a leaf needing wrapping
				dest = self._new_temp( target_return_type )
			else:
				dest = self._new_temp( expected_type or target_return_type )
			self._emit( ir.Call( dest = dest, target = target, receiver = receiver, args = args, kwargs = kwargs ))
			return dest
		else:
			self._emit( ir.Call( dest = None, target = target, receiver = receiver, args = args, kwargs = kwargs ))
			return None

	def _lower_conditional_dispatch( self, node: ast.Call, branches: list[ConditionalDispatch], default: Function, args: list[ir.Operand], kwargs: dict[str,ir.Operand], expected_type: Type|None, want_result: bool ) -> ir.Operand|None:
		# a union-typed argument's runtime tag decides which overload
		# implementation actually runs (e.g. len(copy_from) where
		# copy_from: bytes|bytearray resolves to two candidates, bytes and
		# bytearray). Reuses the same tag/data/v_<member> machinery match
		# statements use (UnionStorage.get) - branches are tried in
		# priority order, falling through to `default` (no test needed -
		# it's whatever's left once every more specific branch is excluded)
		self.lowering._ensure_resolved( default )
		for branch in branches:
			self.lowering._ensure_resolved( branch.function )

		dest = self._new_temp( expected_type or default.return_type ) if want_result else None
		end_label = self._new_label( 'dispatch_end' )
		for branch in branches:
			next_label = self._new_label( 'dispatch_next' )
			self._lower_dispatch_tests( node, branch.function, branch.conditions, args, kwargs, next_label )
			self._emit_dispatch_call( branch.function, args, kwargs, dest, want_result )
			self._emit( ir.Jump( target = end_label ))
			self._emit( ir.Label( name = next_label ))
		self._emit_dispatch_call( default, args, kwargs, dest, want_result )
		self._emit( ir.Label( name = end_label ))
		return dest

	def _lower_dispatch_tests( self, node: ast.AST, target: Function, conditions: list[tuple[Parameter,Type]], args: list[ir.Operand], kwargs: dict[str,ir.Operand], next_label: str ) -> None:
		# a branch's conditions are ANDed together - emits one Cmp +
		# JumpIfFalse per condition, all targeting next_label, which is
		# already a short-circuit AND with no combined boolean value to
		# build at all (same trick _expr_BoolOp uses, just directly in IR
		# since these operands are already lowered)
		bool_cls = self.lowering.discovery.find_name( 'bool', node )
		for param, leaf_type in conditions:
			operand = self.lowering._dispatch_operand_for_param( node, target, param, args, kwargs )
			shape = self.lowering._type_resolver._tagged_union_shape( operand.type )
			if shape is None:
				self.lowering.discovery.fail( f'{target.qualname}: conditional dispatch on a non-union argument: {ast.unparse(node)}', node )
			base, members = shape
			member = next( ( attr for attr in members if attr.type is leaf_type ), None )
			if member is None:
				self.lowering.discovery.fail( f'{target.qualname}: {leaf_type.qualname if leaf_type else "?"} is not a member of {operand.type.qualname}', node )
			tag_attr, _data_attr, _payload_cls, tags = self.lowering._union_storage.get( base )
			tag_dest = self._new_temp( tag_attr.type )
			self._emit( ir.GetAttr( dest = tag_dest, obj = operand, attr = tag_attr.stem ))
			cmp_dest = self._new_temp( bool_cls )
			self._emit( ir.Cmp( dest = cmp_dest, op = ir.CmpOp.EQ, left = tag_dest, right = ir.Const( type = tag_attr.type, value = tags[member.stem] ) ))
			self._emit( ir.JumpIfFalse( cond = cmp_dest, target = next_label ))

	def _emit_dispatch_call( self, target: Function, args: list[ir.Operand], kwargs: dict[str,ir.Operand], dest: ir.Temp|None, want_result: bool ) -> None:
		params = target.parameters or []
		unwrapped_args = [ self._maybe_unwrap_union_arg( a, p.type ) for a, p in zip( args, params ) ]
		unwrapped_kwargs = {
			name: self._maybe_unwrap_union_arg( value, next( p for p in params if p.stem == name ).type )
			for name, value in kwargs.items()
		}
		self.lowering.schedule( target.return_type )
		for p in params:
			self.lowering.schedule( p.type )
		self._emit( ir.Call( dest = dest if want_result else None, target = target, receiver = None, args = unwrapped_args, kwargs = unwrapped_kwargs ))

	def _maybe_unwrap_union_arg( self, operand: ir.Operand, target_type: Type|None ) -> ir.Operand:
		# a union-typed call-site argument (copy_from: bytes|bytearray)
		# must be unwrapped to the concrete leaf type the chosen branch's
		# parameter actually declares before it can be passed as a real
		# argument - mirrors match's own payload extraction
		if target_type is None or operand.type is target_type:
			return operand
		shape = self.lowering._type_resolver._tagged_union_shape( operand.type )
		if shape is None:
			return operand
		base, members = shape
		member = next( ( attr for attr in members if attr.type is target_type ), None )
		if member is None:
			return operand
		tag_attr, data_attr, payload_cls, tags = self.lowering._union_storage.get( base )
		payload_dest = self._new_temp( payload_cls )
		self._emit( ir.GetAttr( dest = payload_dest, obj = operand, attr = data_attr.stem ))
		dest = self._new_temp( target_type )
		self._emit( ir.GetAttr( dest = dest, obj = payload_dest, attr = f'v_{member.stem}' ))
		return dest

	def _lower_union_receiver_call( self, node: ast.Call, dispatch: _ReceiverDispatch, receiver: ir.Operand, expected_type: Type|None, want_result: bool ) -> ir.Operand|None:
		# copy_from.get_const_ptr() where copy_from: bytes|bytearray - unlike
		# _lower_conditional_dispatch (one shared Function, a union-typed
		# ARGUMENT unwrapped per branch), each leaf here has its own
		# unrelated method under this name, so what's dispatched on is the
		# RECEIVER's own tag instead - same tag/data/v_<member> machinery
		# match statements and dispatch already use (UnionStorage.get),
		# just no shared target Function to reuse ConditionalDispatch with
		reference = dispatch.per_leaf[0][1]
		for _member, fn in dispatch.per_leaf:
			self.lowering._ensure_resolved( fn )
		positional, keyword = self.lowering._match_call_args( reference, node )
		args = []
		for param, expr in positional:
			operand = self._lower_expr( expr, param.type )
			self._apply_move_hook( param, operand, dispatch.union.qualname )
			args.append( operand )
		kwargs = {}
		for param, expr in keyword:
			operand = self._lower_expr( expr, param.type )
			self._apply_move_hook( param, operand, dispatch.union.qualname )
			kwargs[param.stem] = operand

		tag_attr, data_attr, payload_cls, tags = self.lowering._union_storage.get( dispatch.union )
		bool_cls = self.lowering.discovery.find_name( 'bool', node )
		dest = self._new_temp( expected_type or reference.return_type ) if want_result else None
		end_label = self._new_label( 'recv_dispatch_end' )
		for i, ( member, fn ) in enumerate( dispatch.per_leaf ):
			is_last = i == len( dispatch.per_leaf ) - 1
			if not is_last:
				next_label = self._new_label( 'recv_dispatch_next' )
				tag_dest = self._new_temp( tag_attr.type )
				self._emit( ir.GetAttr( dest = tag_dest, obj = receiver, attr = tag_attr.stem ))
				cmp_dest = self._new_temp( bool_cls )
				self._emit( ir.Cmp( dest = cmp_dest, op = ir.CmpOp.EQ, left = tag_dest, right = ir.Const( type = tag_attr.type, value = tags[member.stem] )))
				self._emit( ir.JumpIfFalse( cond = cmp_dest, target = next_label ))
			payload_dest = self._new_temp( payload_cls )
			self._emit( ir.GetAttr( dest = payload_dest, obj = receiver, attr = data_attr.stem ))
			narrowed = self._new_temp( member.type )
			self._emit( ir.GetAttr( dest = narrowed, obj = payload_dest, attr = f'v_{member.stem}' ))
			self.lowering.schedule( fn.return_type )
			for p in fn.parameters or []:
				self.lowering.schedule( p.type )
			self._emit( ir.Call( dest = dest, target = fn, receiver = narrowed, args = args, kwargs = kwargs ))
			if not is_last:
				self._emit( ir.Jump( target = end_label ))
				self._emit( ir.Label( name = next_label ))
		self._emit( ir.Label( name = end_label ))
		return dest
