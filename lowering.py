# stdlib imports:
import ast
import copy
import itertools
import math
from contextlib import nullcontext
from dataclasses import dataclass, replace
from typing import Callable, Iterable, NoReturn

# local imports:
import arithmetic_mode
import cfg
import ir
from discovery import Discovery, is_stub_body, reject_reserved_c_identifier
from errors import CompileError, RedundantCompilationError
from fstring_format_spec import FStringFormatSpec, FormatSpecError, parse_format_spec, validate_str_spec, validate_int_spec, validate_float_spec
from mpy_types import (
	Name, Type, Variable, Parameter, Function, Overload, ClassLike, Module, CType,
	Specialization, TaggedUnion, CStruct, CUnion, CEnum, TypeVar, ConditionalDispatch, Move, Copy, RCClass, Scalar,
	CallableType, ClosureType, TupleType, FixedArrayType, int_stem_range, GeneratorType, Protocol, InheritanceChainMixin,
)
import overload_resolution
from type_resolver import TypeResolver
from union_storage import ReceiverDispatch as _ReceiverDispatch
from lowering_shared import _LoopContext, TryContext
from lowering_stmt import StmtLoweringMixin
from lowering_loop import LoopLoweringMixin
from lowering_epilogue import EpilogueLoweringMixin
from lowering_expr import ExprLoweringMixin
from lowering_dunder_dispatch import DunderDispatchLoweringMixin
from lowering_try import TryLoweringMixin
from lowering_generator import GeneratorLoweringMixin
from lowering_closure import ClosureLoweringMixin
from lowering_intrinsics import CompilerIntrinsicsMixin
from lowering_fstring import FStringLoweringMixin
from lowering_construct import ConstructLoweringMixin
from lowering_calls import CallLoweringMixin
from lowering_generic_call import GenericCallLoweringMixin

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
		# return-only generic type-parameter inference - a generic function
		# call whose return type is a bare type param appearing in no
		# parameter, inferred by eagerly lowering the body once every OTHER
		# type param is known (see FunctionLowering._infer_return_only_type_
		# params/_infer_return_only_type_params_inline). Lives HERE, not on
		# FunctionLowering, deliberately: the eager pre-compile (non-@inline
		# case) always builds a BRAND NEW FunctionLowering for the nested
		# call (PLAN_LAMBDA.md's own reentrancy fix), so a guard scoped to
		# one FunctionLowering instance would start empty every time,
		# invisible across exactly the boundary that needs guarding - this
		# has to live on the one object that outlives every nested
		# FunctionLowering. Keyed by id(target) (the abstract base Function)
		# alone, not by which concrete args - simpler and more conservative:
		# ANY reentrant eager-inference attempt on the same target function
		# is rejected outright, regardless of args, rather than trying to
		# precisely distinguish safe from unsafe recursion.
		self._eager_return_inference_stack: list[int] = []
		# id(target) -> the set of id(TypeVar) from target.type_params that
		# appear ANYWHERE in target's own parameter types - a static,
		# per-function-signature property, independent of any call site, so
		# it's computed once and cached here rather than per call
		self._param_referenced_type_params: dict[int,frozenset[int]] = {}

	def lower_function( self, fn: Function ) -> list[ir.Instruction]:
		# per-function lowering state (_instructions, _current_fn, _cfg, etc.
		# - see FunctionLowering's own docstring) lives on a FRESH instance
		# every call, including a nested/reentrant call made mid-way through
		# lowering an enclosing function (see _expr_Lambda's eager lowering
		# path, PLAN_LAMBDA.md) - there is no shared mutable state between
		# an outer and inner call to worry about saving/restoring at all.
		#
		# PLAN_GENERATORS.md: ensure_generator_synthesized is idempotent and a
		# no-op for an ordinary function - for a generator, it MUST have
		# already run by now (every caller resolving this fn's own return
		# type via TypeResolver.ensure_resolved triggers it eagerly, before
		# this fn is ever dequeued for its own lowering - a call site needs
		# the REAL return type immediately, it can't wait for this fn's own
		# turn on the work queue), so fn.node.body is already the rewritten,
		# yield-free constructor body by the time this runs - this call is
		# just a safety net for a generator reached with no earlier caller
		# (e.g. main() itself).
		self._type_resolver.ensure_generator_synthesized( fn )
		return FunctionLowering( self, fn ).run()

	def lower_global( self, var: Variable ) -> list[ir.Instruction]:
		return FunctionLowering( self, None ).run_global( var )

	def lower_deinit_epilogue( self, ordered_vars: list[Variable] ) -> list[ir.Instruction]:
		''' emitter_c.py's own __metalpy_deinit() synthesis - see
		FunctionLowering.run_deinit_epilogue's docstring. '''
		return FunctionLowering( self, None ).run_deinit_epilogue( ordered_vars )

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

	def _is_aliasing_expr( self, node: ast.expr, operand: 'ir.Operand|None' = None ) -> bool:
		# does lowering `node` hand back a reference to a value that
		# already exists independently (needing its own Incref if it's
		# stored into a new binding), vs a genuinely fresh value (Allocate,
		# or a Call - always a fresh owned handoff, whether the callee's
		# own body built it via Allocate or received it as an alias
		# itself, since a well-behaved callee already accounts for that on
		# its own side)? Name/Attribute reads are the only currently-
		# supported expression forms that alias existing state -
		# BinOp/Compare/Constant/UnaryOp never produce RC values at all,
		# and Call is always fresh from the caller's perspective. BoolOp
		# CAN now hold an RC value (it returns the decisive operand's own
		# value, not always bool - see _expr_BoolOp), but still correctly
		# falls through to False below like IfExp already does: both
		# already resolve ownership internally (their own Incref-if-
		# aliasing-else-untrack bookkeeping on whichever operand/branch
		# actually wins) before returning dest, so from THIS function's
		# caller's perspective dest is already a fresh, independently-
		# owned handoff, exactly like an ordinary Call result.
		# ast.Subscript is NOT aliasing in general: _expr_Subscript's
		# dominant path (a real __getitem__) is a Call underneath (fresh).
		# Its other path (tuple[...]'s own constant-index element access,
		# PLAN_TUPLE.md - a raw ir.GetAttr on synthesized fields _0/_1/...,
		# since a heterogeneous tuple has no real __getitem__ to call) IS
		# genuinely aliasing though, the same shape ast.Attribute already
		# is below - this WAS "not reachable by any real code yet" before
		# tuples existed, but tuple-element reads reach it now.
		# _expr_Subscript tags the node itself (node.is_tuple_element_read)
		# when it takes that path, rather than have this function re-
		# inspect/re-resolve node.value's own type to tell the two
		# Subscript shapes apart (the same "risk re-resolving and double-
		# evaluating the receiver" problem ast.Attribute's own is_bound_
		# method_closure tag below avoids the same way). Missing this
		# (confirmed by a real, repeated-real-
		# compile-and-run-verified use-after-free, not just reasoning):
		# reassigning an existing local to another tuple element read
		# (`x = some_tuple[0]`, x already bound) skipped the Incref an
		# aliasing read needs, so the tuple's own eventual teardown
		# (cascading decref of ITS OWN _0/_1 fields) double-released the
		# same object x still pointed to.
		#
		# ast.Attribute is genuinely ambiguous now, the same way Subscript
		# already was above: `obj.field` reads an existing field (aliasing),
		# but `worker.run` (a bound-method reference) CONSTRUCTS a fresh
		# closure (an Allocate underneath, via _lower_bound_method_closure)
		# - same "fresh owned handoff" shape as a Call, not a read.
		# _lower_bound_method_closure tags ITS OWN node (node.
		# is_bound_method_closure) the same way _expr_Subscript tags
		# is_tuple_element_read below - checking the OPERAND's type alone
		# (as this used to) is NOT enough: an ordinary field read whose
		# DECLARED type happens to be ClosureType (e.g. a union payload
		# access like `self.data.v_Ok` for a Result[Closure[...],E], or any
		# user field typed Closure[...]) produces the same ClosureType
		# operand while genuinely aliasing an EXISTING closure, not
		# constructing a fresh one - confirmed by a real heap-use-after-free:
		# `list[Closure[...]]` silently under-referenced every element
		# popped back out, because Result.unwrap()'s `ok: T = self.data.
		# v_Ok` was wrongly treated as fresh (skipping the Incref an
		# aliasing capture-into-local needs) whenever T happened to be a
		# Closure.
		# Scoped to ast.Attribute specifically, NOT every ClosureType
		# operand - `d = c` (a bare Name reading an EXISTING closure local)
		# is an ordinary aliasing read like any other RC-typed Name, and
		# must still incref (confirmed by a real regression: `d = c` then
		# calling both silently underreferenced the shared closure)
		# a value that needed coercing INTO a declared union type (_coerce_
		# into_union, called from _coerce_or_check_operand right after
		# whichever _expr_X method above actually dispatched on `node`) is
		# ALSO genuinely ambiguous the same way: `node` might be a plain
		# Name/Attribute read that looks aliasing on its own, but by the
		# time the caller sees `operand` here it's no longer that read at
		# all - it's the FRESH return value of a synthesized union-member
		# constructor Call (mirrors _coerce_into_union's own emission: `dest
		# = self._new_temp(union); self._emit(ir.Call(dest=dest, ...))`),
		# exactly the "Call is always fresh from the caller's perspective...
		# since a well-behaved callee already accounts for that on its own
		# side" rule this function's own docstring already states for every
		# OTHER Call. That constructor's own body already Increfs the leaf
		# it wraps (the same way any other constructor increfs a BORROWED
		# RC argument it stores into a field - see cfg.py's attr_assign()) -
		# a caller here treating the wrapped result as STILL aliasing the
		# original `node` double-counts that Incref (confirmed by direct
		# compile-and-run: `return b` from a Box|None-returning function,
		# b an ordinary BORROWED parameter, left compiler.refcount(b) two
		# higher than the caller's own new binding plus b's own local
		# should ever account for) - and, wherever the caller's own is_alias
		# branch also skips untrack_temp() (assign()'s is_alias=True path
		# never untracks `src`, only the is_alias=False path does),
		# _flush_pending_temps' later cleanup of the still-tracked union
		# temp decrefs it a SECOND time on top of that, which can net back
		# out to looking "correct" by sheer coincidence (two wrongs) or, in
		# a context where only one of those two extra ops fires, silently
		# under- or over-count for real (confirmed via generated-C
		# inspection, not just reasoning). Checking the OPERAND actually
		# produced (not `node`, which has no idea a coercion happened
		# underneath it) is the only way to tell - same reasoning as the
		# ClosureType check just above, generalized from "a bound-method
		# ast.Attribute" to "any node a coercion silently replaced".
		if getattr( operand, 'is_union_coerce_result', False ):
			return False
		if isinstance( node, ast.Attribute ) and getattr( node, 'is_bound_method_closure', False ):
			return False
		if isinstance( node, ast.Subscript ):
			return getattr( node, 'is_tuple_element_read', False )
		# PLAN_GENERATORS.md Phase C - a captured `(yield expr)` reads an
		# existing field (self.__send_slot, via _expr_Yield) the same way
		# ast.Attribute already does - self.__send_slot independently
		# keeps its own reference regardless of who else reads it, so
		# wherever this value lands needs its own Incref exactly like any
		# other field read. Without this, a captured yield's own consumer
		# (`held = yield i`) was wrongly treated as "fresh" (a Call-shaped
		# handoff needing no Incref of its own) and given its own tracked
		# ownership that never gets balanced - confirmed via a real
		# compiler.refcount() repro.
		return isinstance( node, ( ast.Name, ast.Attribute, ast.Yield ))

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
		# needs no atomic access at all) with T a plain scalar OR a raw
		# Ptr[U]/ConstPtr[U] (same single-machine-word representation as a
		# scalar - e.g. a lock-free lazily-published cache pointer, see
		# str.to_utf16()). Bare RC types are still rejected deliberately:
		# atomically swapping an RC pointer without incref/decref
		# bookkeeping is exactly the lock-free-RC rabbit hole this
		# intentionally stays out of (see the plan's own Context) - a raw
		# Ptr[U]/ConstPtr[U] carries no refcount, so that concern doesn't
		# apply to it.
		if not ( isinstance( ptr_type, Specialization ) and ptr_type.pointer_stem() == 'Ptr' ):
			self.discovery.fail( f'compiler.atomic_*(...) argument must be Ptr[T]: {ast.unparse(node)}', node )
		pointee = ptr_type.args[0]
		if not ( isinstance( pointee, Scalar ) or self._type_resolver._is_ptr_specialization( pointee )):
			self.discovery.fail(
				f'compiler.atomic_*(...) argument must point to a plain scalar or raw pointer, not '
				f'{pointee.qualname if pointee else "?"}: {ast.unparse(node)}',
				node,
			)
		return pointee


	def _eval_cexpr( self, expr: str, header: str, node: ast.AST ) -> int:
		import hashlib
		import os
		import tempfile
		from pathlib import Path
		import linker_c
		# cache key derived from (expr, header) — deterministic, so the
		# same expression always hits the same cached value regardless
		# of which compilation or project it appears in
		key = hashlib.sha256( f'{expr}\0{header}'.encode() ).hexdigest()[:16]
		cache_dir = Path( tempfile.gettempdir() ) / 'metalpy' / 'cexpr'
		cache_file = cache_dir / key
		if linker_c.ensure_cache_dir( cache_dir ) and cache_file.is_file():
			# a torn/half-written entry parses as ValueError, not as a wrong
			# answer - treat it as a miss and re-probe rather than crashing
			# the whole compile (see linker_c.atomic_write_cache). OSError
			# likewise: on Windows this open fails while another process's
			# os.replace of the same path is in flight. Cache contention must
			# never be an error on either side - a lost read costs a re-probe.
			try:
				return int( cache_file.read_text().strip() )
			except ( ValueError, OSError ):
				pass

		# no cached value — compile and run a tiny C program
		cc = linker_c.detect_cc()
		if cc is None:
			self.discovery.fail(
				f'compiler.cexpr({expr!r}, {header!r}) needs a C compiler '
				f'(clang, gcc, or MSVC) — none was found',
				node,
			)
		# _GNU_SOURCE (defined before any include) makes glibc expose the
		# POSIX.1-2008 / GNU-gated identifiers (LC_CTYPE_MASK, CLOCK_MONOTONIC,
		# CLOCK_REALTIME, ...) that -std=c11's implied __STRICT_ANSI__ would
		# otherwise hide - without it these probes fail to compile on Linux even
		# though the very same symbols are perfectly usable in the real build
		# (which declares its own prototypes). Harmless on macOS/Windows
		# toolchains, which ignore it and expose these by default anyway.
		c_src = f'#define _GNU_SOURCE 1\n#include <{header}>\n#include <stdio.h>\nint main(void) {{ printf("%zu\\n", (size_t)({expr})); return 0; }}\n'
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
		linker_c.atomic_write_cache( cache_file, str( value ))
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

		import linker_c
		cache_dir = Path( tempfile.gettempdir() ) / 'metalpy' / 'case_folding'
		cache_file = cache_dir / 'UnicodeData.txt'
		if linker_c.ensure_cache_dir( cache_dir ) and cache_file.is_file():
			# an empty file is a torn write, never a real (multi-MB) table -
			# re-download instead of building casing tables from nothing.
			# Deliberately NOT trying to detect a PARTIAL-but-non-empty file:
			# there's no length/checksum to check against, and a content
			# heuristic (say, "must end in a newline") would risk permanently
			# re-downloading a valid table if upstream ever changed format.
			# Writes are atomic now (see linker_c.atomic_write_cache), so a
			# partial file can only be a leftover from an older build; delete
			# %TEMP%/metalpy/case_folding to clear one.
			# OSError: on Windows this open fails while another process's
			# os.replace of the same path is in flight - a miss, not an error
			try:
				cached = cache_file.read_bytes()
			except OSError:
				cached = b''
			if cached:
				return cached

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
		linker_c.atomic_write_cache( cache_file, data )
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

	_WINDOWS_ZONES_URL = 'https://raw.githubusercontent.com/unicode-org/cldr/main/common/supplemental/windowsZones.xml'

	def _fetch_windows_zones_xml( self, node: ast.AST ) -> bytes:
		''' downloads (or reads a locally-cached/overridden copy of)
		windowsZones.xml - CLDR's Windows-zone-name <-> IANA-zone-name
		mapping table (deliberately the RAW content host, not the
		github.com/.../blob/... viewer URL, which serves an HTML page, not
		XML). Same caching shape as _fetch_unicode_data_txt above: cached
		indefinitely once fetched, with METALPY_WINDOWS_ZONES_DIR (mirroring
		METALPY_UNICODE_DATA_DIR's existing override convention) letting an
		offline/CI build point at a local copy instead of ever reaching the
		network. '''
		import os
		import tempfile
		from pathlib import Path

		override_dir = os.environ.get( 'METALPY_WINDOWS_ZONES_DIR', '' ).strip()
		if override_dir:
			local_path = Path( override_dir ) / 'windowsZones.xml'
			if not local_path.is_file():
				self.discovery.fail(
					f'METALPY_WINDOWS_ZONES_DIR={override_dir!r} is set but {local_path} does not exist',
					node,
				)
			return local_path.read_bytes()

		import linker_c
		cache_dir = Path( tempfile.gettempdir() ) / 'metalpy' / 'windows_zones'
		cache_file = cache_dir / 'windowsZones.xml'
		if linker_c.ensure_cache_dir( cache_dir ) and cache_file.is_file():
			# same "empty file is a torn write, re-download" posture as
			# _fetch_unicode_data_txt - see its own comment
			try:
				cached = cache_file.read_bytes()
			except OSError:
				cached = b''
			if cached:
				return cached

		import urllib.error
		import urllib.request
		request = urllib.request.Request( self._WINDOWS_ZONES_URL, headers = { 'User-Agent': 'metalpy-compiler' } )
		try:
			with urllib.request.urlopen( request, timeout = 30 ) as response:
				data = response.read()
		except ( urllib.error.URLError, OSError ) as e:
			self.discovery.fail(
				f'compiler.fetch_windows_zones_table(): failed to download {self._WINDOWS_ZONES_URL} ({e}) - '
				f'set METALPY_WINDOWS_ZONES_DIR to a local directory containing windowsZones.xml to avoid the network entirely',
				node,
			)
		linker_c.atomic_write_cache( cache_file, data )
		return data

	def _build_windows_zones_table( self, data: bytes, node: ast.AST ) -> bytes:
		''' parses windowsZones.xml's <mapZone other="Win Name"
		territory="001" type="Iana/Name"/> elements - territory="001" only
		(the default/world mapping: one canonical IANA zone per Windows
		key; territory-specific overrides are an explicit v1 scope cut,
		same posture case-folding took on SpecialCasing.txt's one-to-many
		mappings) - into a linear-scan table: repeated [u16 win_len LE]
		[win_name utf-8][u16 iana_len LE][iana_name utf-8] records, packed
		back to back with no count/header prefix - the caller already knows
		the total byte length via bytes.byte_len(), and windows_zones.
		WindowsZoneMap's own runtime lookup (lib/windows_zones.py) just
		scans until it hits that length. ~150 entries at this writing - far
		too few to justify sorting + binary search over a variable-width
		record layout. '''
		import xml.etree.ElementTree as ET
		try:
			root = ET.fromstring( data )
		except ET.ParseError as e:
			self.discovery.fail(
				f'compiler.fetch_windows_zones_table(): failed to parse windowsZones.xml ({e})',
				node,
			)
		entries: list[tuple[str,str]] = []
		for map_zone in root.iter( 'mapZone' ):
			if map_zone.get( 'territory' ) != '001':
				continue
			win_name = map_zone.get( 'other' )
			iana_name = map_zone.get( 'type' )
			if not win_name or not iana_name:
				continue
			entries.append( ( win_name, iana_name ) )
		if not entries:
			self.discovery.fail(
				f"compiler.fetch_windows_zones_table(): parsed windowsZones.xml but found zero territory='001' "
				f"<mapZone> entries - the file is probably not what was expected (wrong format, truncated download, ...)",
				node,
			)
		table = bytearray()
		for win_name, iana_name in entries:
			win_bytes = win_name.encode( 'utf-8' )
			iana_bytes = iana_name.encode( 'utf-8' )
			table += len( win_bytes ).to_bytes( 2, 'little' )
			table += win_bytes
			table += len( iana_bytes ).to_bytes( 2, 'little' )
			table += iana_bytes
		return bytes( table )

	def _lower_compiler_fetch_windows_zones_table( self, node: ast.Call ) -> ir.Operand:
		''' compiler.fetch_windows_zones_table() - downloads/caches
		windowsZones.xml (see _fetch_windows_zones_xml) and folds to an
		ir.Const(type=bytes, value=<the encoded table>) - the SAME program-
		wide static-embedding path compiler.fetch_unicode_table() already
		uses (see its own docstring, and emitter_c.py's _emit_string_
		literals) - no new emitter support needed. Only actually reached
		(and only actually pays the download/parse cost) for a program that
		references compiler.fetch_windows_zones_table() itself - nothing in
		builtins does, only lib/windows_zones.py's own install(). '''
		if len( node.args ) != 0 or node.keywords:
			self.discovery.fail(
				f"compiler.fetch_windows_zones_table() takes no arguments: {ast.unparse(node)}",
				node,
			)
		data = self._fetch_windows_zones_xml( node )
		table = self._build_windows_zones_table( data, node )
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

	def _ensure_errdefer_retained_params( self, fn: Function ) -> set[str]:
		''' fn.errdefer_retained_params (mpy_types.py), computed lazily from
		fn's own AST the first time anyone asks. A plain IR-derived fact
		(scanning cfg.py's own push_defer as it registers each defer/
		errdefer body) would seem simpler, but reachability-ordered lowering
		(compiler.py starts from main() and lowers callees only once first
		discovered reachable) means a callee can easily still be UNLOWERED
		- this field still empty - at the exact moment a caller's own Call
		to it needs the answer; an IR-derived version would silently miss
		exactly the "caller lowered before callee" ordering our own 4-test
		repro cluster hits. The AST is available for every Function up
		front regardless of lowering order, so scan that instead.

		Finds every `defer(EXPR)`/`errdefer(EXPR)` call-form statement and
		every `with defer:`/`with errdefer:` block anywhere in fn's body
		(ast.walk, so nested inside an if/while is still found), then scans
		each one's own registered body for a bare `compiler.incref(<name>)`
		whose <name> is one of fn's own parameters - see cfg.py's
		manually_decreffed()/mark_possibly_retained() for what this list is
		actually used for. Deliberately narrow (a single-level Name, not a
		field/subscript/whatever an incref could in principle be pointed at
		- no test needs more, and a false negative here just means a real
		leak stays undetected-by-this-mechanism, not a new one). '''
		if fn.errdefer_retained_params_computed:
			return fn.errdefer_retained_params
		fn.errdefer_retained_params_computed = True
		param_stems = { p.stem for p in ( fn.parameters or [] ) }
		if not param_stems or fn.node is None:
			return fn.errdefer_retained_params
		for stmt in ast.walk( fn.node ):
			defer_body: list[ast.stmt] | None = None
			if isinstance( stmt, ast.Expr ) and self._defer_kind_of_call( stmt.value ) is not None and stmt.value.args:
				defer_body = [ ast.Expr( value = stmt.value.args[0] ) ]
			elif (
				isinstance( stmt, ast.With ) and len( stmt.items ) == 1 and stmt.items[0].optional_vars is None
				and self._defer_kind_of_with( stmt.items[0].context_expr ) is not None
			):
				defer_body = stmt.body
			if defer_body is None:
				continue
			for sub in defer_body:
				for call in ast.walk( sub ):
					if (
						isinstance( call, ast.Call ) and isinstance( call.func, ast.Attribute )
						and isinstance( call.func.value, ast.Name ) and call.func.value.id == 'compiler'
						and call.func.attr == 'incref' and len( call.args ) == 1
						and isinstance( call.args[0], ast.Name ) and call.args[0].id in param_stems
					):
						fn.errdefer_retained_params.add( call.args[0].id )
		return fn.errdefer_retained_params

	def _body_may_fall_off_the_end( self, body: list[ast.stmt] ) -> bool:
		# a simple, deliberately narrow check (not full terminator analysis):
		# true whenever the LAST top-level statement isn't itself a `return`
		# (an empty body, or one ending in a plain statement/if/loop/etc,
		# could all still fall through to the function's own closing brace).
		# A body that's actually unreachable past this point (both branches
		# of a trailing if already return, a trailing `while True:` with no
		# break, a trailing call to a -> NoReturn function like sys.panic(),
		# ...) is a false positive - harmless, since the resulting fall-off
		# unwind+Return is then genuinely dead code, never executed. _stmt_If
		# and visit_Match's own true_terminates/false_terminates/terminates
		# detection used to share this exact same scope cut but no longer do
		# (see _stmt_diverges, wired into both, and PLAN_COMPILER_BUG_SWEEP.md) -
		# not mirrored here since a false positive at THIS level stays harmless
		# dead code rather than a real narrowing-survival bug, unlike those two
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
		found = self._resolve_scalar_name( found )
		return found if isinstance( found, Function ) else None

	def _find_field( self, owner_type: Type|None, name: str ) -> Variable|None:
		# a non-failing probe, unlike _attr_lookup - "this type has no such
		# FIELD" (either no such attribute at all, or it names a method
		# instead) is a normal outcome for a caller trying a shape (e.g.
		# _try_lower_indirect_call recognizing obj.field(...) as an indirect
		# call through a Ptr[Callable[...]]-typed field, only once it's
		# already ruled out a real method of that name via _find_method),
		# not a real error to report. Same posture/shape as _find_method
		# above, mirroring _attr_lookup's own lookup logic minus the fail().
		owner_type = self._ensure_resolved( owner_type )
		if isinstance( owner_type, Specialization ) and owner_type.pointer_stem() is not None:
			owner_type = self._ensure_resolved( owner_type.args[0] )
		if isinstance( owner_type, ( CStruct, RCClass )):
			found = owner_type.chain_lookup( name )
		else:
			names = getattr( owner_type, 'names', None )
			found = names.get( name ) if isinstance( names, dict ) else None
		if not isinstance( found, Variable ):
			return None
		self._ensure_resolved( found )
		return found

	def _find_iterator_next_method( self, owner_type: Type|None ) -> Function|None:
		# PLAN_GENERATORS.md Phase 3 - a non-failing probe (same posture as
		# _find_method above): "this type has no __next__" is a normal
		# outcome, not an error. Shape validation (does __next__ actually
		# return T|None) happens once, in _lower_for_over_iterator itself,
		# where a real error location is available. Only ever called on a
		# type _stmt_For has ALREADY confirmed conforms to IteratorProtocol[T] (see
		# _type_conforms_to_protocol) - never a raw duck-typing probe.
		return self._find_method( owner_type, '__next__' )

	def _type_conforms_to_protocol( self, owner_type: Type|None, protocol: ClassLike ) -> bool:
		''' does the resolved concrete type declare conformance to
		`protocol` (a bare Protocol/ClassLike, e.g. the real Iterator or
		Iterable class object - not parametrized) - a membership check
		against .protocols, mirroring mpy_types.TypeVar.bound_satisfied_by's
		own bare-Protocol case (discovery.py's _validate_protocol_
		conformance already verified, at that class's own definition, that
		every required method actually exists - this doesn't re-verify
		that, just asks "did this class declare it"). Identity-compares
		each declared protocol's own unparametrized .base, so this matches
		regardless of which concrete T the class happened to parametrize
		it with (list[i32]'s own Iterable[i32] conformance still matches a
		bare `protocol=Iterable` lookup here). '''
		resolved = self._ensure_resolved( owner_type )
		base = resolved.base if isinstance( resolved, Specialization ) else resolved
		if isinstance( base, TupleType ):
			# tuple[...] isn't an RCClass itself - a homogeneous tuple's
			# declared protocol conformance lives on its lazily-synthesized
			# backing RCClass instead (tuple_storage.py); a heterogeneous
			# tuple has no backing at all (never conforms to anything)
			base = base.backing
		if not isinstance( base, RCClass ):
			return False
		for p in base.protocols:
			p_base = p.base if isinstance( p, Specialization ) else p
			if p_base is protocol:
				return True
		return False

	def _is_range_call( self, node: ast.expr ) -> str|None:
		# range(...) is textually recognized as compiler sugar, same as
		# compiler.wrap_arithmetic/defer/etc. - there's no real range()
		# function, and this is DELIBERATE, not a gap: real generator
		# functions exist now (PLAN_GENERATORS.md - a `yield`-containing
		# function becomes a synthesized RCClass + __next__ state machine),
		# but range() specifically stays intrinsic on purpose - it's the
		# single most common loop-counting construct in any real program,
		# and every call site would pay a real heap allocation + atomic-
		# refcount-churn cost for zero functional benefit if it were
		# reimplemented as an ordinary generator (see ARCHITECTURE.md's own
		# "design decision: range() stays a compiler intrinsic" section).
		# Confirmed with the user (2026-08-15): do not convert this. This
		# covers exactly the 1-2 arg counting-loop shape real lib/ code
		# already uses (str.concat's `for i in range(count):`), and a
		# range() call INSIDE a generator body still works (desugared into
		# the equivalent while-loop shape before lowering - see type_
		# resolver.py's _desugar_generator_for_loops, PLAN_GENERATORS.md
		# Phase 4) - this recognizer itself is untouched by that. Returns
		# the discriminant ('range') rather than a bare bool, matching
		# every other textual recognizer in this file
		if isinstance( node, ast.Call ) and isinstance( node.func, ast.Name ) and node.func.id == 'range':
			return 'range'
		return None

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
		if method.broken:
			raise RedundantCompilationError() # already reported at the point method's own resolution failed - see Name.broken
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

	def _build_closure_env_class( self, captures: list[tuple[str,Type]], qualname: str, file: object, line: int|None ) -> RCClass:
		''' the backing RCClass for one capturing lambda/nested-def's captured
		environment - real, un-erased Variable attributes (unlike ClosureType's
		own fn/self, both always Ptr[None]), built exactly like tuple_storage.py's
		TupleStorage.get() builds a tuple's own backing class from nothing: no
		parsed source, no AST body, no __init__ (a construction site builds one
		directly via ir.Allocate's field=value shape, same as a tuple literal
		does). Every existing RC mechanism (cfg.py's is_rc/rc_leaves,
		type_resolver.py's _synthesize_rcclass_destructor/_build_field_teardown_
		ast, emitter_c.py's emit_rcclass) applies to it completely unchanged -
		real typed fields are exactly what makes the automatic, per-field-
		correct (RC pointer/nested CStruct/tag-gated union/nothing) destructor
		synthesis "just work" here with zero new code.

		Deliberately NOT memoized/interned the way TupleStorage.get() is: a
		tuple's backing class is reached from many independent call sites
		across a whole compile run (any annotation spelling the same element
		types), but a lambda/nested-def's own AST node is visited exactly once
		by the ordinary top-to-bottom lowering walk (the same reason
		_lambda_counter is a plain incrementing counter, not a cache key) - a
		cache here would be written once and never read. Two occurrences that
		happen to capture same-typed locals still get two independent classes
		(GeneratorType's "fresh per occurrence" posture, not TupleType's
		cross-occurrence interning - see mpy_types.py's own comment on the
		difference). This is only safe because a nested def/lambda inside a
		generic enclosing function is rejected outright elsewhere
		(_reject_generic_enclosing_scope) - if that restriction is ever lifted,
		a generic function's own capturing closure would be lowered once per
		monomorphization and WOULD need its env class memoized per
		specialization, not built fresh-and-unmemoized like this. '''
		attributes = [
			Variable( stem = name, qualname = f'{qualname}.{name}', file = file, line = line, type = t )
			for name, t in captures
		]
		env_cls = RCClass(
			stem = qualname, qualname = qualname, file = file, line = line,
			base = None, type_params = None,
			attributes = attributes, methods = [],
			names = { a.stem: a for a in attributes },
			resolve = None,
		)
		self.schedule( env_cls )
		return env_cls

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
	_DELITEM_ALTERNATIVES = 'call .__delitem__(...) directly and consume its Result yourself instead'

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
		if isinstance( owner_type, Specialization ) and owner_type.pointer_stem() is not None:
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

	def _resolve_scalar_name( self, found: Name|None ) -> Name|None:
		''' Scalar.names may hold a raw Specialization - a generic dunder/
		method registered via `TypeName.method = generic_fn[T]`, stored
		as-is by discovery.py's visit_Assign since a Monomorphizer isn't
		constructible that early. Every reader of Scalar.names funnels the
		looked-up value through here first so a Specialization transparently
		becomes the real, concrete Function it stands for, instead of
		silently falling through an `isinstance(found, Function)` check
		(what every existing caller already does) as if the name were
		never registered at all. '''
		return self._monomorphized_function( found ) if isinstance( found, Specialization ) else found

	def _resolve_receiver_generic_dunder( self, found: Name|None, owner_type: Type|None ) -> Name|None:
		''' Ptr[T]/ConstPtr[T]'s own dunders (Ptr.__add__ = ptr_add_checked,
		see lib/builtins/__ptr_arith.py) are registered as a BARE generic
		Function (`found` here, still carrying its own unbound type param) -
		unlike an ordinary scalar dunder (i32.__add__ = i_add_checked[i32]),
		there's no concrete pointee type to specialize against AT
		REGISTRATION time, since Ptr's own `.names` dict is shared across
		every Ptr[X] (Specialization.names passes through to .base - see
		mpy_types.py). The pointee type only becomes known at the CALL
		SITE, from the receiver's own owner_type (Ptr[i32], say) - so
		unlike _resolve_scalar_name's Specialization-already-known case,
		this composes the specialization here instead, binding the
		function's own type param to owner_type's pointee arg, then
		monomorphizes it exactly like any other generic instantiation.
		Confirmed via a real spike that skipping this step reaches the
		emitter with a bare, unbound TypeVar and crashes
		(NotImplementedError: c_type: unsupported type <TypeVar ...>) -
		this is not optional defensive padding, it's required for Ptr/
		ConstPtr dunder dispatch to work at all. '''
		if (
			isinstance( found, Function ) and found.type_params
			and isinstance( owner_type, Specialization ) and owner_type.pointer_stem() is not None
		):
			spec = self.discovery._get_or_create_specialization( found, list( owner_type.args ))
			return self._monomorphized_function( spec )
		return found

	def monomorphize_class( self, spec: Specialization ) -> ClassLike:
		return self._monomorphizer.monomorphize_class( spec )

	def _try_resolve_namespace( self, node: ast.expr ) -> Name|None:
		return self._type_resolver._try_resolve_namespace( node )



	def _attr_lookup_callable( self, owner_type: Type|None, attr: str, ctx: ast.AST ) -> Function|Overload:
		return self._type_resolver._attr_lookup_callable( owner_type, attr, ctx )

	def _match_call_args( self, target: Function, call: ast.Call, *, receiver_fills_first_param: bool = False ) -> tuple[list[tuple[Parameter,ast.expr]],list[tuple[Parameter,ast.expr]]]:
		if target.broken:
			raise RedundantCompilationError() # already reported at the point target's own resolution failed - see Name.broken
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
		# a Scalar-registered method's receiver (_lower_call's own "a
		# Scalar-registered method" comment) isn't threaded through call.args
		# at all - it's spliced into target's own first positional parameter
		# directly, later, by the caller - so that parameter is pre-matched
		# here rather than checked against the call site's own args
		receiver_param = None
		if receiver_fills_first_param and positional_params:
			receiver_param = positional_params[0]
			positional_params = positional_params[1:]
		if len( call.args ) > len( positional_params ):
			self.discovery.fail( f'too many positional arguments: {ast.unparse(call)}', call )
		positional = list( zip( positional_params, call.args ))
		keyword: list[tuple[Parameter,ast.expr]] = []
		for kw in call.keywords:
			param = next(( p for p in target.parameters if p.stem == kw.arg and not p.is_vararg and not p.is_kwarg ), None )
			if param is None:
				self.discovery.fail( f'{target.qualname} has no parameter {kw.arg!r}', call )
			keyword.append(( param, kw.value ))
		# "too many positional arguments" above only catches an EXCESS of
		# arguments - nothing previously checked the other direction (a
		# required parameter, no default, never matched by either list),
		# so a call could silently omit one and mis-typecheck downstream
		# instead of failing cleanly here.
		matched_params = { id( p ) for p, _ in positional } | { id( p ) for p, _ in keyword }
		if receiver_param is not None:
			matched_params.add( id( receiver_param ))
		missing = [ p.stem for p in target.parameters if not p.is_vararg and not p.is_kwarg and p.default is None and id( p ) not in matched_params ]
		if missing:
			missing_repr = ', '.join( repr( m ) for m in missing )
			self.discovery.fail( f'{target.qualname} missing required argument(s) {missing_repr}: {ast.unparse(call)}', call )
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
		if param.is_move:
			if not is_move_call:
				self.discovery.fail(
					f"{target.qualname}: parameter {param.stem!r} is move[{param.type.qualname}] - "
					f"call site must pass move({ast.unparse(expr)}): {ast.unparse(call)}",
					call,
				)
			if len( expr.args ) != 1 or expr.keywords:
				self.discovery.fail( f'move(...) takes exactly one argument: {ast.unparse(expr)}', call )
			moved = expr.args[0]
			# cfg.py's ownership tracking (see its own module docstring) only
			# tracks top-level bindings (params/locals/self) - moving a FIELD
			# or SUBSCRIPT target reaches into storage owned by something
			# cfg.py can't invalidate, so the source keeps its pointer after
			# the "move" and the owner's own destructor later double-frees
			# it. A plain Name (an already-tracked binding) or any other
			# expression producing a fresh, unaliased value (e.g. a
			# constructor call like move(bytearray(0))) is fine - reject only
			# the aliasing shapes at the call site instead of silently
			# miscompiling.
			if isinstance( moved, ( ast.Attribute, ast.Subscript )):
				if isinstance( moved, ast.Attribute ):
					reason = ( "a class attribute can't be moved out from under its object "
						"(that would leave the object partially uninitialized)" )
				else:
					reason = "a container element can't be moved out from under its container in place"
				self.discovery.fail(
					f"move(...) cannot move {ast.unparse(moved)!r} directly - {reason}, and the same "
					f"restriction applies to any field or subscript target. Assign it to a local "
					f"variable first, then move() the local, e.g. `tmp = {ast.unparse(moved)}` then "
					f"`move(tmp)`: {ast.unparse(call)}",
					call,
				)
			return moved
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
		float: ( 'f32', 'f64' ),
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
	_OR_THROW_ALTERNATIVES = (
		'or_throw() propagates any leaf not caught by an except clause of any enclosing try (innermost first) to the caller, '
		'exactly like or_return() - there is no other way for the enclosing function to receive it'
	)
	_RAISE_ALTERNATIVES = (
		'raise propagates any leaf not caught by an except clause of any enclosing try (innermost first) to the caller, '
		'exactly like or_return() - there is no other way for the enclosing function to receive it'
	)
	_RESULT_CONSUMING_METHODS = ( 'is_ok', 'is_err', 'unwrap', 'unwrap_or' ) # or_return() is handled separately - see _lower_or_return
	# shared "alternatives" text for the general auto-or_throw() rule (a
	# discarded Result statement, or a Result flowing into a context wanting
	# its own Ok payload directly) - wrap in try/except to catch specific
	# leaves, or consume it explicitly first
	_AUTO_CONSUME_ALTERNATIVES = (
		'wrap this in try/except to catch specific error leaves, or consume it yourself first '
		'via .unwrap()/.or_return()/match'
	)

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
			if isinstance( declared.bound, Specialization ) and isinstance( declared.bound.base, Protocol ):
				# a PARAMETRIZED protocol bound (S: Iterable[T]) - T usually
				# never appears in ANY parameter's own declared type
				# directly (only inside S's bound, and in the return type -
				# e.g. min[T,S:Iterable[T]](seq:S) -> T), so ordinary
				# argument unification alone would never bind it, and
				# return-only inference (the OTHER mechanism that could)
				# breaks down the moment T flows through an intermediate
				# local into a NESTED generic call inside the body (a real,
				# separate gap, confirmed by a real repro: `value: T = ...;
				# value = min(value, t)` inside min[T,S]'s own body hit a
				# false "T inferred as both min.T and i32" conflict, since
				# the eager return-only lowering leaves T as its own
				# abstract placeholder throughout the body, not the value
				# actually flowing through it). Reverse-unify instead, right
				# here: now that `declared` (S) is bound to a concrete
				# `actual`, find actual's OWN declared conformance
				# Specialization for the SAME protocol S's bound names
				# (already substituted/concrete - see Monomorphizer.
				# monomorphize_class's own protocols handling) and unify
				# ITS args against the bound's own args - the same
				# structural walk TypeVar.bound_satisfied_by uses to CHECK
				# this, just repurposed here to also BIND from it.
				base = actual.base if isinstance( actual, Specialization ) else actual
				if isinstance( base, TupleType ):
					base = base.backing
				if isinstance( base, RCClass ):
					for entry in base.protocols:
						if not ( isinstance( entry, Specialization ) and entry.base is declared.bound.base ):
							continue
						entry_args = entry.args
						if isinstance( actual, Specialization ):
							# base here is still the ABSTRACT class template
							# (actual.base) - substitute its declared protocol
							# args against actual.args directly, rather than
							# fully building actual via monomorphize_class
							# first (the old approach, which re-entered
							# monomorphize_class whenever actual was its OWN
							# still-in-progress build - e.g. set[T]'s __iter__
							# body needing set[T]'s own Sequence[T] conformance
							# to reverse-unify T - and had to skip the whole
							# lookup via a _building guard to avoid either a
							# RecursionError or binding to the wrong, still-
							# abstract TypeVar; skipping instead silently left
							# T unbound forever, confirmed by a real repro: the
							# shared _sequence_iter's return type never
							# resolved past its abstract T the first time any
							# caller's __iter__ actually needed it). A
							# protocol entry's own .base is always the
							# Protocol, never `actual`'s class, so
							# substituting just its args can't re-enter
							# monomorphize_class for `actual` at all - no
							# guard needed.
							entry_args = [
								self._type_resolver.monomorphizer.substitute_type_params( a, base.type_params or [], actual.args )
								for a in entry_args
							]
						for b_arg, e_arg in zip( declared.bound.args, entry_args ):
							self._unify_type_param( type_params, b_arg, e_arg, bindings, node, context_qualname )
						break
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
		if isinstance( declared, ClosureType ) and isinstance( actual, ClosureType ):
			# key: Closure[[T],K] - a bound-method/capturing-lambda-typed
			# parameter, unified the same way a Ptr[Callable[[T],K]]
			# parameter's own CallableType branch (just above) does; a
			# capturing lambda argument's own REAL return type is already
			# known by this point (_expr_Lambda's own eager-lowering infers
			# it from the body regardless of any hint - PLAN_LAMBDA.md), so
			# K binds normally from `actual` here rather than needing
			# return-only inference at all. Without this, K was classified
			# as return-only-missing (_type_mentions_param has the same gap -
			# see its own ClosureType branch) and _infer_return_only_type_
			# params tried to re-derive it by eagerly compiling `apply`'s
			# OWN body against `key`'s still-abstract declared type
			# (Closure[[T],K]) - self-referential and unable to ever resolve
			# to anything concrete, since it has no access to the ACTUAL
			# argument's real inferred type at all. That silently bound K to
			# itself (declared IS the bare type param K, unify's own tv-
			# match branch accepts any `actual`, including K right back),
			# reaching emitter_c.py with a still-bare TypeVar - "c_type:
			# unsupported type" - confirmed by a real repro.
			for d_arg, a_arg in zip( declared.arg_types, actual.arg_types ):
				self._unify_type_param( type_params, d_arg, a_arg, bindings, node, context_qualname )
			self._unify_type_param( type_params, declared.return_type, actual.return_type, bindings, node, context_qualname )
			return
		if isinstance( declared, TaggedUnion ) and declared.file is None and not isinstance( actual, TaggedUnion ):
			# anonymous union parameter (X|None) whose non-None leaf can
			# itself mention a type param (e.g. `key: Ptr[Callable[[T],K]]
			# |None`) - type_resolver.py's own speculative pre-pass
			# deliberately declines to drill into a union leaf like this
			# (see its _unify_type_param's own TaggedUnion branch) and
			# leaves it to real argument-lowering time, here, once
			# `actual`'s own concrete (non-union - a caller passing a whole
			# union-typed variable through unchanged isn't this shape at
			# all) type is known. Match against whichever leaf shares
			# actual's own outer shape - a union's leaves are otherwise
			# disjoint types, so at most one can structurally apply.
			# Without this, K here was never bound by ordinary argument
			# unification at all (confirmed by a real repro: it fell
			# through to return-only inference, which then requires a
			# single-return body - a needless restriction this shape
			# should never have hit, since K IS argument-inferable through
			# `key`), and if the body genuinely has multiple returns
			# (as apply_or_default's `if key is not None: return key(x)`/
			# `return default` does), K reached emitter_c.py as a still-
			# bare, unsubstituted TypeVar - "c_type: unsupported type".
			actual_spec = self._type_resolver._as_specialization( actual )
			for leaf in declared.leaves():
				if any( leaf is tv for tv in type_params ):
					self._unify_type_param( type_params, leaf, actual, bindings, node, context_qualname )
					return
				if isinstance( leaf, Specialization ) and actual_spec is not None and leaf.base is actual_spec.base:
					self._unify_type_param( type_params, leaf, actual, bindings, node, context_qualname )
					return
				if isinstance( leaf, CallableType ) and isinstance( actual, CallableType ):
					self._unify_type_param( type_params, leaf, actual, bindings, node, context_qualname )
					return
			return
		if isinstance( declared, GeneratorType ):
			# a declared Generator[T,E]/Iterator[Result[T,E]] return type
			# (e.g. iter[T,S:Iterable[T]](seq:S) -> Generator[T,StopIteration])
			# unified against `actual` - the REAL synthesized backing class a
			# concrete generator call actually produces (never itself a
			# GeneratorType - only the abstract annotation ever is). There's
			# no direct backing-class -> GeneratorType reverse mapping, but
			# the backing class's own __next__ always returns Result[elem_type,
			# error_type] (see type_resolver.py's ensure_generator_synthesized),
			# so read T/E back off THAT instead - same "derive the same fact
			# a different way" approach _same_type's TupleType/Specialization
			# duality already uses.
			next_fn = self._find_iterator_next_method( actual )
			result_args = next_fn.return_type.args if next_fn is not None and isinstance( next_fn.return_type, Specialization ) else None
			if result_args is not None and len( result_args ) == 2:
				self._unify_type_param( type_params, declared.elem_type, result_args[0], bindings, node, context_qualname )
				self._unify_type_param( type_params, declared.error_type, result_args[1], bindings, node, context_qualname )
			return
		if isinstance( declared, TupleType ):
			# a declared tuple[T,T]-shaped parameter - `actual` may be the
			# bare TupleType itself (e.g. built fresh from an annotation) or
			# its resolved backing RCClass (a real argument's already-
			# resolved type carries the backing class, not the annotation
			# object) - same bare-TupleType-vs-backing duality _same_type's
			# own TupleType branch handles, via tuple_type_for's reverse lookup
			actual_tuple = actual if isinstance( actual, TupleType ) else self._tuple_storage.tuple_type_for( actual )
			if actual_tuple is not None and len( declared.elem_types ) == len( actual_tuple.elem_types ):
				for d_elem, a_elem in zip( declared.elem_types, actual_tuple.elem_types ):
					self._unify_type_param( type_params, d_elem, a_elem, bindings, node, context_qualname )
			return

	def _check_type_param_bounds( self, node: ast.AST, type_params: list[TypeVar], concrete_args: list[Type], context_qualname: str ) -> None:
		# every call site that finishes substituting a concrete type for each
		# of type_params (bare inferred call, explicit Name[T](...), generic
		# construction, class-inherited generic method) funnels through here
		# once concrete_args is fully known - see TypeVar.bound's own comment
		# for why this can't live in _get_or_create_specialization instead
		for tv, concrete in zip( type_params, concrete_args ):
			if not tv.bound_satisfied_by( concrete, type_params, concrete_args, self._type_resolver ):
				self.discovery.fail(
					f'{context_qualname}[...]: {concrete.qualname} does not implement protocol {tv.bound.qualname} '
					f'required by type parameter {tv.stem!r}: {ast.unparse(node)}',
					node,
				)

	def _type_mentions_param( self, t: Type|None, tv: TypeVar ) -> bool:
		''' PLAN_RETURN_INFERENCE.md - true if the bare TypeVar `tv` occurs
		anywhere inside `t`, using the SAME structural recursion
		_unify_type_param itself uses (Specialization.args, CallableType/
		ClosureType.arg_types/return_type, anonymous-TaggedUnion leaves):
		"does this parameter type CONTAIN tv" needs to agree exactly with
		"would _unify_type_param actually BIND tv from an argument at this
		position", or a type param that's structurally present but never
		actually unified against would be wrongly classified as argument-
		inferable and never get a chance at return-only inference at all.
		A `key: Closure[[T],K]` parameter is the confirmed real case for
		ClosureType: without that branch, K was wrongly classified as
		return-only-missing, and _infer_return_only_type_params tried to
		re-derive it by eagerly compiling the GENERIC function's own body
		against `key`'s still-abstract declared type - self-referential and
		unable to ever resolve to anything concrete, since it has no access
		to the actual argument's real (already-known, via eager lambda-
		lowering) closure type at all. '''
		if t is None:
			return False
		if t is tv:
			return True
		if isinstance( t, Specialization ):
			return any( self._type_mentions_param( a, tv ) for a in t.args )
		if isinstance( t, ( CallableType, ClosureType )):
			return any( self._type_mentions_param( a, tv ) for a in t.arg_types ) or self._type_mentions_param( t.return_type, tv )
		if isinstance( t, TaggedUnion ) and t.file is None:
			return any( self._type_mentions_param( attr.type, tv ) for attr in t.attributes )
		if isinstance( t, GeneratorType ):
			# Generator[T,E]/Iterator[Result[T,E]]/Generator[T,SendType,E] -
			# a return-only type param can live inside any of these slots
			# (e.g. iter[T,S:Iterable[T]](seq:S) -> Generator[T,StopIteration],
			# T never appears in any parameter's own declared type directly -
			# confirmed by a real repro: without this, T was wrongly
			# classified as genuinely-missing instead of return-only-
			# inferable, since this method previously had no GeneratorType
			# case at all)
			return (
				self._type_mentions_param( t.elem_type, tv )
				or self._type_mentions_param( t.error_type, tv )
				or ( t.send_type is not None and self._type_mentions_param( t.send_type, tv ))
			)
		if isinstance( t, TupleType ):
			return any( self._type_mentions_param( e, tv ) for e in t.elem_types )
		return False

	def _param_referenced_type_params_for( self, target: Function ) -> frozenset[int]:
		''' PLAN_RETURN_INFERENCE.md - the set of id(TypeVar) from target.
		type_params that occur anywhere in target's own PARAMETER types - a
		static property of the function's own signature, independent of any
		call site, cached per id(target) since _lower_inferred_generic_call
		consults it on every under-determined bare call to that function '''
		cached = self._param_referenced_type_params.get( id( target ))
		if cached is not None:
			return cached
		referenced = frozenset(
			id( tv ) for tv in ( target.type_params or [] )
			if any( self._type_mentions_param( p.type, tv ) for p in ( target.parameters or [] ))
		)
		self._param_referenced_type_params[ id( target )] = referenced
		return referenced

	def _dispatch_operand_for_param( self, node: ast.AST, target: Function, param: Parameter, args: list[ir.Operand], kwargs: dict[str,ir.Operand] ) -> ir.Operand:
		if param.stem in kwargs:
			return kwargs[param.stem]
		index = next( ( i for i, p in enumerate( target.parameters or [] ) if p is param ), None )
		if index is not None and index < len( args ):
			return args[index]
		self.discovery.fail( f'{target.qualname}: cannot locate the call-site argument for parameter {param.stem!r}', node )


class FunctionLowering(
	StmtLoweringMixin, LoopLoweringMixin, EpilogueLoweringMixin, ExprLoweringMixin, DunderDispatchLoweringMixin, TryLoweringMixin, GeneratorLoweringMixin, ClosureLoweringMixin, CompilerIntrinsicsMixin, FStringLoweringMixin, ConstructLoweringMixin, CallLoweringMixin, GenericCallLoweringMixin,
):
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
		# every stem ever handed to _mark_fresh_local_declared - unlike
		# fn.names (which loses an entry on del), this NEVER shrinks. Lets a
		# genuinely fresh declaration (one reached with no LIVE binding for
		# its own stem - _declare_local's own callers all pre-check that)
		# tell "truly first-ever declaration of this name" (needs no C-name
		# disambiguation) apart from "a fresh declaration following an
		# earlier del of this same name" (needs its own, uid-suffixed C
		# identifier - see Variable.needs_uid_suffix's own docstring and
		# del_reuse_and_emitter_naming_bug). Pre-seeded with every real
		# parameter's own stem: a parameter IS a Variable (Parameter
		# subclasses it) and del DOES accept one (_stmt_Delete's own
		# isinstance check doesn't exclude Parameter) - `def f(x: i32): del
		# x; x: str = 'hi'` must give the second x its own identifier
		# too, exactly like any other del-then-redeclare, or it would
		# silently collide with the C parameter itself in the function
		# signature. 'self' is added the same way, just below, once its
		# own Parameter exists (not yet constructed this early).
		self._ever_declared_stems: set[str] = { p.stem for p in ( fn.parameters or [] )} if fn is not None else set()
		self._current_fn = fn
		# name -> id(the ast.Match) of whichever match statement's arm most
		# recently declared FRESH storage for this name via a `case T(name):`
		# binding - lets _stmt_Assign tell a SIBLING arm of that SAME match
		# reusing the name (must get its own independent storage, arms are
		# mutually exclusive) apart from an unrelated later reassignment or a
		# totally different match statement reusing the name (ordinary
		# reuse/mismatch rules apply unchanged, see _stmt_Assign's own
		# is_match_binding handling). Never removed on del/reassignment -
		# stale entries are harmless, only ever consulted alongside a live
		# `existing` binding that's ALSO still a match binding for this exact
		# name, which del/an unrelated reassignment already replaces in
		# fn.names by then.
		self._match_binding_origin: dict[str,int] = {}
		# set for real in run()/run_global()/run_deinit_epilogue(), right
		# before each starts lowering anything - None here only covers the
		# brief window before any of those runs (never actually observed by
		# _emit(), same as _owning_module just below)
		self._cfg: 'cfg.CFGState|None' = None
		# ambient "line we're currently lowering", refreshed at the top of
		# _lower_stmt/_lower_expr - debug-info only (Allocate.loc, see _emit),
		# never save/restored, doesn't need to be exact for nested sub-exprs
		self._current_lineno = 0
		# set for real in run()/run_global(), right before either enters its
		# own module_context - see check_module_visibility's own comment on
		# why this (not discovery.module_stack[-1]) is what every privacy
		# check reached from this instance uses. None here only covers the
		# brief window before run()/run_global() itself runs (never actually
		# observed - nothing calls check_module_visibility that early).
		self._owning_module: 'Module|None' = None
		# id()s of Variable objects CURRENTLY bound as an @inline splice's
		# own self/parameter aliases (both _lower_inline_call's single-
		# expression path and _splice_multi_statement_inline_body's own
		# copy add/remove their own bound ids here) - a name resolving to
		# one of these is a pure ALIAS for whatever the ORIGINAL call site
		# already passed in (e.g. `other` inside an inlined scalar_eq[T]
		# resolving straight through to the caller's own operand, zero-
		# copy - see _lower_inline_call's own "when the operand is ALREADY
		# a Variable" comment), not a fresh access in its own right -
		# check_module_visibility's 3 lowering.py call sites skip re-
		# checking these: the ORIGINAL operand expression already got its
		# own, correct check when IT was first lowered at the real call
		# site, before ever being threaded through the splice. Confirmed
		# necessary via a real false positive: lib/csv.py's own `self.
		# __state == _ST_START_FIELD` got flagged as builtins illegally
		# reaching csv's own package-private constant, purely because the
		# generic scalar_eq[i32] dunder this dispatches through re-resolves
		# its own `other` parameter (bound directly to _ST_START_FIELD)
		# while lowering ITS OWN body, under builtins' own module context.
		self._inline_param_alias_ids: set[int] = set()
		self._arithmetic_mode: list[arithmetic_mode.ArithmeticMode] = [ arithmetic_mode.ArithmeticChecked() ]
		self._loop_depth = 0
		self._loop_labels: list[_LoopContext] = []
		self._in_deferred_body = False
		# set (briefly, restored in a finally) only around _lower_scalar_cast's
		# own literal-argument branch - an EXPLICIT cast on a literal
		# (u32(-11), compiler.cast(u8, -1)) is deliberate bit-reinterpretation,
		# exempt from _expr_Constant's own range check below; every other
		# route into _expr_Constant (plain assignment, argument binding,
		# return, CEnum construction) leaves this False and gets validated
		self._allow_literal_bit_reinterpret = False
		self._defer_flags: list[Variable] = []
		# hidden for-loop iterator locals (__for_obj_N) whose real assignment
		# sits at the for-statement's own lexical position, guarded only by
		# whatever ENCLOSING loop/branch reaches it - a for-loop nested
		# inside another loop leaves __for_obj_N genuinely uninitialized on
		# the outer-loop-never-runs path. Safe at runtime regardless (the
		# corresponding _defer_flags entry starts False - see
		# _emit_epilogue's flag_inits - so the guarded release this guards
		# never actually reads it), but MSVC's flow analysis can't see that
		# correlation across two different variables and flags it anyway
		# (C4703). _emit_epilogue splices a real `= 0` default init for each
		# of these right after FuncStart, same trick flag_inits already
		# uses, so the declaration is never skippable-over by a goto
		self._for_obj_null_inits: list[Variable] = []
		# PLAN_GENERATORS.md's defer/errdefer phase (Mechanism 2) - whatever
		# type_resolver.py's _tag_armed_defer_sites tagged the statement
		# CURRENTLY being lowered with (see _lower_stmt's own push/pop),
		# or inherited from an enclosing tagged statement if this one
		# carries no tag of its own. Always [] outside a generator's own
		# $$__next__ - nothing else ever sets the tag this reads
		self._generator_armed_defer_sites: list[tuple[str,bool,list[ast.stmt]]] = []
		self._return_value_var = None
		# `with EXPR [as NAME]: BODY` (general context-manager form, see
		# _lower_with_context_manager) - unique per with-statement in this
		# function, only for the synthesized ctx-holding local's own stem
		self._with_ctx_id = 0
		# try/except/else/finally - .or_throw()/raise textually inside the
		# CURRENT try body (same function) consults this; see TryContext's
		# own docstring. An uncovered leaf walks the WHOLE stack innermost-
		# first (_dispatch_leaves_against_try_stack), so a nested try's own
		# uncovered leaf DOES fall back to an outer try's own handlers.
		self._try_stack: list[TryContext] = []
		# bare `raise` (re-raise) - the innermost enclosing except handler's
		# own TryHandler.raise_value_var, pushed right before that handler's
		# body is lowered and popped right after (_stmt_Try) - a nested
		# handler's own push shadows its enclosing handler's for the
		# duration of ITS body, matching real re-raise scoping.
		self._active_raise_values: list[Variable] = []
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
		# set (briefly, restored in a finally) only around lowering a
		# multi-statement @inline body's own PRE-RETURN statements (see
		# _splice_multi_statement_inline_body) - an .or_return()/checked-
		# arithmetic early exit reached from one of those statements would
		# otherwise jump to the CALLER's own real epilogue mid-splice
		# (self._current_fn is briefly the caller during that window too),
		# silently skipping the rest of the splice AND the rest of the
		# caller's own subsequent statements - _consume_checked_result
		# checks this and fails clearly instead. The trailing return-
		# expression itself is lowered with this already restored to
		# False, unaffected - nothing of the splice remains after it to
		# skip past, so jumping to the caller's own epilogue is correct
		# there, exactly as it always has been.
		# set by _stmt_Expr right before lowering a bare discarded-call
		# statement (`foo(...)` with no assignment) - snapshots len(self.
		# _pending_temps) from BEFORE the call's own args are built, exactly
		# like or_throw()/or_return()'s own receiver_pending_start (see their
		# own comment for why: the CALL's argument temps must be released
		# on _emit_or_throw's early-return propagate path too, not just the
		# receiver). Consumed (and reset) by _finish_call_result the one time
		# it actually reaches the discarded-Result auto-or_throw() branch -
		# see its own comment.
		self._discarded_call_pending_start: int|None = None
		self._in_inline_splice_prelude = False
		# parallel to self._cfg's own _inline_scope_stack (cfg.py), pushed/
		# popped in lockstep by _splice_multi_statement_inline_body - cfg.py's
		# InlineScope only carries the CFG-level boundary_depth/label; these
		# are the LOWERING-level artifacts _stmt_Return/_consume_checked_
		# result need once current_epilogue_label() hands back a splice-local
		# label: (result_var, exited_flag). result_var is where an early exit
		# (return/or_return/checked-arithmetic) inside the splice's pre-
		# return statements stows its value - the splice-local analogue of
		# self._return_value_var. exited_flag is armed (Assign, Const(True))
		# right before jumping there, so the ladder's own tail can tell
		# "early exit vs normal fallthrough" apart and decide whether to
		# still lower the trailing return-expression - see
		# _splice_multi_statement_inline_body's own comment
		self._inline_scope_vars: list[tuple[Variable,Variable]] = []

	def run( self ) -> list[ir.Instruction]:
		fn = self._current_fn
		module = self.lowering._find_module_for( fn )
		# fixed for this WHOLE lowering pass, unlike discovery.module_stack
		# (which type-resolution can transiently re-push mid-lowering, e.g.
		# monomorphizing an @inline generic dunder call - see check_module_
		# visibility's own comment on why that's a real, confirmed false-
		# positive source for module-privacy checks specifically) - every
		# check_module_visibility call site reached from within this
		# FunctionLowering instance uses THIS, not the ambient stack top.
		self._owning_module = module
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
						self._ever_declared_stems.add( 'self' ) # see _ever_declared_stems' own docstring - del accepts self too, same as any other parameter
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
							payload_cls = concrete_union.get_local_or_raise( 'data' ).type
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
						resolve_type = self.lowering._ensure_resolved,
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
					if fn.is_generator_next:
						self._emit_generator_dispatch_prologue( fn )
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

					if self._return_value_var is not None and not ( body_stmts and self._stmt_diverges( body_stmts[-1] )):
						# self._return_value_var is only ever created for a REAL
						# (non-None/NoReturn) declared return type (see its own
						# construction above) - reaching here means fn's own body
						# can fall off the closing brace without ever having set a
						# return value on that path. Before this check existed,
						# nothing caught this at all: emitter_c.py's own ir.Return
						# handling only ever fires from an EXPLICIT return (or the
						# fn.return_type is none_type fallthrough a few lines below,
						# which deliberately does NOT apply here), so the compiled
						# C function's own __return_value local was simply left
						# uninitialized on this path - a real, confirmed bug (lib/
						# builtins/__list.py's list[T].insert(), found via a genuine
						# MSVC /RTC1 uninitialized-variable runtime trap, not a
						# hypothetical - see msvc_toolset_c11atomics memory). Uses
						# _stmt_diverges (not the deliberately conservative _body_
						# may_fall_off_the_end below, which tolerates false positives
						# as harmless dead code for the None-return case) so a
						# genuinely-terminating body (trailing if/else that both
						# return, an exhaustive-by-construction match, a NoReturn
						# call, a `while True:` with no break) is correctly NOT
						# flagged.
						self.lowering.discovery.fail(
							f'{fn.qualname}: not every code path returns a value '
							f'(declared to return {fn.return_type.qualname if fn.return_type is not None else "?"}) - '
							f'add an explicit `return` on every path (a non-exhaustive '
							f'match/if with no matching case counts as a path that can '
							f'still fall through)',
							fn.node,
						)

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

					# current_epilogue_label() (called with no operand below) is
					# only a real question when the body can actually fall off
					# the end into this closing brace (_body_may_fall_off_the_
					# end() - False whenever the last top-level statement is
					# already a literal `return`, which always terminates).
					# Calling it unconditionally is self-fulfilling: merely
					# asking it "is anything still live" captures whatever
					# entry it finds (current_epilogue_label()'s own captured=
					# True/_any_shared_label_used side effect) and manufactures
					# a label for a fall-through that can never happen - e.g. a
					# function whose sole `return x` returns its own live local
					# directly (label=None, return_()'s inline unwind already
					# handled everything, per _stmt_Return) left that local's
					# entry on the stack uncancelled, and this check alone used
					# to conjure a dead "L__epilogue__: release(x); return
					# __return_value;" block after the real, unconditional
					# `return x;` - confirmed via $$__new__ and any ordinary
					# `def f() -> SomeRC: x = SomeRC(...); return x`.
					# used_shared_epilogue_label()/cancel_flags() still catch
					# every case that genuinely needs the ladder built (an
					# EARLIER return already committed a goto into it, or a
					# defer/errdefer flag needs its init spliced in) regardless
					# of whether the body can fall off the end.
					if (
						# mark_captured=False: this is a pure existence probe -
						# the label itself is discarded, never used to emit a
						# goto (the fall-off-the-end path relies on
						# build_epilogue_ladder() being placed immediately
						# after the body, pure fallthrough, no jump needed) -
						# see current_epilogue_label()'s own comment on why
						# marking it captured here would be spurious
						( self.lowering._body_may_fall_off_the_end( fn.node.body ) and self._cfg.current_epilogue_label( mark_captured = False ) is not None )
						or self._cfg.used_shared_epilogue_label() or self._cfg.cancel_flags()
					):
						# some return (or OrJump) already jumped into the
						# shared epilogue ladder (_stmt_Return/_consume_checked_
						# result, via current_epilogue_label()), or nothing did
						# but entries are still pending at the function's own
						# closing brace (an implicit `return None`/fall-off
						# reaching them the same way) - either way,
						# build_epilogue_ladder() covers whatever's still
						# pending, RC decrefs and defer/errdefer replays alike.
						# used_shared_epilogue_label() (not just current_
						# epilogue_label()) is required here: an entry a return
						# ALREADY jumped into, while still live, may since have
						# been cancelled (move()/compiler.decref(x)/del) by the
						# time we reach this closing brace - current_epilogue_
						# label() then correctly reports "nothing NEW needs to
						# unwind here" (None), but that earlier goto still needs
						# its label built, or it's left dangling - see used_
						# shared_epilogue_label()'s own docstring for the real
						# repro this was found from. cancel_flags() (not just
						# the two checks above) is ALSO required: a captured
						# entry that got flag-guarded belonging to an already-
						# popped @inline splice (build_inline_scope_ladder()
						# already consumed and removed it from the stack, and
						# never sets _any_shared_label_used - that flag is
						# function-epilogue-specific, see current_epilogue_
						# label()'s own comment) would otherwise leave this
						# function's own flag_inits below never spliced in at
						# all - a real uninitialized-bool read at the flag's
						# own JumpIfFalse, not merely a missed decref
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
		self._owning_module = module # see run()'s own identical comment

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
					resolve_type = self.lowering._ensure_resolved,
				)
				self._pending_temps = []
				operand = self._lower_expr( var.init, var.type )
				# PLAN_THREAD_SAFE_SHARED_STATE.md Part A: this global's own
				# initializing write needs the SAME critical section an
				# ordinary `global X; X = ...` reassignment gets, in case the
				# initializer (or something it calls) spawns a thread that
				# reassigns this global before this write itself runs - see
				# cfg.py's assign_global_initializer's own docstring for the
				# confirmed race this closes. The Assign has to stay INSIDE
				# the same critical section as the Acquire/decref (not
				# reconstructed after the fact) for the identical reason
				# _cfg_assign's own write-side split already documents: it's
				# its own separate textual read/write of `var` in the
				# generated C, not covered by whatever the decref touched.
				for instr in self._cfg.assign_global_initializer( var ):
					self._emit( instr )
				self._emit( ir.Assign( dest = var, src = operand ))
				# operand is now owned by var - untrack it (no Decref) BEFORE
				# the flush below, same as _stmt_Return's identical hand-off
				# pattern, so _flush_pending_temps' own delete_temp(operand)
				# correctly no-ops instead of releasing the value just
				# stored. Without this, any OTHER still-pending temp from
				# lowering var.init (e.g. a fallible initializer's own raw
				# Result[T,E], still holding its own internal reference to
				# the SAME payload after .unwrap()'s narrowed extraction)
				# was never flushed at all - the previous bare DeleteTemp-
				# only loop below never called _cfg.delete_temp() to begin
				# with, silently skipping every real Decref a fallible
				# global initializer's own scaffolding needed. Confirmed via
				# a real repro: `r_eol: re.Pattern = re.compile(...).unwrap(
				# ...)` at module scope left the compile()'s own raw Result
				# permanently retaining the Pattern it had already handed
				# off, one leaked reference for the whole process lifetime.
				self._cfg.untrack_temp( operand )
				if cfg.rc_leaves( var.type ):
					self._emit( ir.ReleaseGlobalLock( var = var ))
				self._flush_pending_temps()

		return self._instructions

	def run_deinit_epilogue( self, ordered_vars: list[Variable] ) -> list[ir.Instruction]:
		''' debug-mode automatic leak-check epilogue (emitter_c.py's
		__metalpy_deinit()) - decrefs every global RC variable in
		`ordered_vars` (caller passes them in REVERSE dependency order,
		undoing __metalpy_init()'s own construction order - see emitter_c.
		py's own __metalpy_deinit assembly). Same "real CFGState, fn=None"
		shape as run_global above (no self/construction of its own), reused
		here purely for its union-aware decref() (cfg.py's
		_tag_gated_refcount_instructions) - a nested-union RC global needs
		the identical runtime tag dispatch an ordinary local/field release
		already gets, not a hand-rolled duplicate. '''
		bool_cls = self.lowering.discovery.get_intrinsics()['bool']
		self._cfg = cfg.CFGState(
			None,
			bool_type = bool_cls,
			new_temp = self._new_temp,
			new_label = self._new_label,
			union_storage = self.lowering._union_storage.get,
			resolve_type = self.lowering._ensure_resolved,
		)
		for var in ordered_vars:
			for instr in self._cfg.decref( var.type, var ):
				self._emit( instr )
		for t in reversed( self._pending_temps ):
			self._emit( ir.DeleteTemp( temp = t ))
		return self._instructions
