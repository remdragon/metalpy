# stdlib imports:
import ast
import copy
import itertools
import math
from contextlib import nullcontext
from dataclasses import dataclass, replace
from typing import Callable, Iterable

# local imports:
import arithmetic_mode
import cfg
import ir
from discovery import Discovery, is_stub_body
from errors import CompileError, RedundantCompilationError
from fstring_format_spec import FStringFormatSpec, FormatSpecError, parse_format_spec, validate_str_spec, validate_int_spec, validate_float_spec
from mpy_types import (
	Name, Type, Variable, Parameter, Function, Overload, ClassLike, Module, CType,
	Specialization, TaggedUnion, CStruct, CUnion, CEnum, TypeVar, ConditionalDispatch, Move, Copy, RCClass, Scalar,
	CallableType, ClosureType, TupleType, FixedArrayType, int_stem_range,
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
	# float inf/nan faults - the primary path is the same as OverflowError's
	# (the enclosing function returns Result[_,FloatingPointError] and the
	# Result propagates via OrReturn/OrJump); wrap_arithmetic/saturate_arithmetic
	# make float arithmetic raw IEEE (inf/nan produced silently, never an error)
	'FloatingPointError': (
		'wrap this in `with compiler.wrap_arithmetic:`, `with compiler.saturate_arithmetic:`, '
		'or `with compiler.panic_arithmetic(...):` instead'
	),
}

# ast.BinOp operator -> the dunder method name to dispatch to for a
# non-scalar left operand (str.__add__, etc.). Scalar operands always
# go through arithmetic mode instead. The three bitwise entries exist
# purely for set[T]'s own algebra (__or__/__and__/__xor__ - union/
# intersection/symmetric_difference); ast.Sub (__sub__, difference) was
# already here for str/int's own use.
_BINOP_DUNDER: dict[type,str] = {
	ast.Add: '__add__',
	ast.Sub: '__sub__',
	ast.Mult: '__mul__',
	ast.FloorDiv: '__floordiv__',
	ast.Mod: '__mod__',
	ast.Div: '__truediv__', # float-only in practice (see lib/builtins/__scalar_dunders.py) - int has no `/`, only `//`
	ast.BitOr: '__or__',
	ast.BitAnd: '__and__',
	ast.BitXor: '__xor__',
	ast.LShift: '__lshift__',
	ast.RShift: '__rshift__',
}

# forward binop dunder name -> its REFLECTED counterpart, mirroring Python's
# real protocol: `a + b` tries `a.__add__(b)` first, and - if that isn't
# applicable (no such dunder on left.type at all, OR left.type is Scalar and
# so has no dunder mechanism of its own to begin with, e.g. `5 + some_vector`)
# - falls back to `b.__radd__(a)`. Unlike `==`/`!=` (see _LeafPairEq's own
# docstring on why equality reflects onto the SAME method name, just with
# receiver/argument swapped), every one of these is a genuinely asymmetric
# operator - Python gives each one its own DIFFERENTLY-NAMED reflected
# method, not a reflected call to the same name, since e.g. `a - b` and
# `b - a` are never interchangeable the way `a == b`/`b == a` are.
_REFLECTED_BINOP_DUNDER: dict[str,str] = {
	'__add__': '__radd__',
	'__sub__': '__rsub__',
	'__mul__': '__rmul__',
	'__floordiv__': '__rfloordiv__',
	'__mod__': '__rmod__',
	'__truediv__': '__rtruediv__',
	'__or__': '__ror__',
	'__and__': '__rand__',
	'__xor__': '__rxor__',
}

# ArithmeticMode subclass -> the mode-qualified dunder name prefix binop
# dispatch tries FIRST, before falling back to the base name (__add__ ->
# __wrapped_add__ under ArithmeticWrap, __saturated_add__ under
# ArithmeticSaturate). ArithmeticChecked/ArithmeticPanic aren't here - they
# always use the base name directly (no separate "panicked" spelling; a
# @fallible_arithmetic dunder's own is_fallible_arithmetic consumption
# already handles the panic-vs-propagate distinction). This is the whole
# mechanism that lets a class like `int` (which never registers
# __wrapped_add__) stay mode-independent with zero isinstance(Scalar)-style
# special-casing anywhere: the qualified lookup just misses and falls
# through to __add__.
_MODE_DUNDER_PREFIX: dict[type,str] = {
	arithmetic_mode.ArithmeticWrap: 'wrapped',
	arithmetic_mode.ArithmeticSaturate: 'saturated',
}

# (operation kind, mode) -> the ir.BinOp opcode compiler.checked_*/wrapped_*/
# saturated_* resolve to for INTEGER operands - see _lower_compiler_checked_
# binop. 'floordiv'/'mod' stay fallible (ZeroDivisionError) in every mode -
# see ir.py's own DivWrap/DivSaturate/ModWrap/ModSaturate comments - unlike
# add/sub/mul, which are only fallible under 'checked'.
_CHECKED_BINOP_OPCODES: dict[tuple[str,str],type] = {
	( 'add', 'checked' ): ir.AddCheck, ( 'add', 'wrapped' ): ir.AddWrap, ( 'add', 'saturated' ): ir.AddSaturate,
	( 'sub', 'checked' ): ir.SubCheck, ( 'sub', 'wrapped' ): ir.SubWrap, ( 'sub', 'saturated' ): ir.SubSaturate,
	( 'mul', 'checked' ): ir.MulCheck, ( 'mul', 'wrapped' ): ir.MulWrap, ( 'mul', 'saturated' ): ir.MulSaturate,
	( 'floordiv', 'checked' ): ir.Div, ( 'floordiv', 'wrapped' ): ir.DivWrap, ( 'floordiv', 'saturated' ): ir.DivSaturate,
	( 'mod', 'checked' ): ir.Mod, ( 'mod', 'wrapped' ): ir.ModWrap, ( 'mod', 'saturated' ): ir.ModSaturate,
	( 'shl', 'checked' ): ir.ShlCheck, ( 'shl', 'wrapped' ): ir.ShlWrap, ( 'shl', 'saturated' ): ir.ShlSaturate,
}

# same idea for FLOAT operands - wrap/saturate raw-IEEE add/sub/mul reuse the
# integer Wrap opcodes (their emitter codegen is already type-generic, see
# _emit_wrap_arith); there's no float 'saturated' opcode distinct from
# 'wrapped' at all (arithmetic_mode.py's own ArithmeticSaturate.GetFloatBinOp
# delegates to _raw_float_binop, identically to ArithmeticWrap) - the
# library-level saturated_add/etc dunders for f32/f64 call compiler.
# wrapped_add directly instead of a separate saturated intrinsic (see
# lib/builtins/__scalar_dunders.py), so 'saturated' is deliberately absent
# here. No 'floordiv'/'mod' entries either - float has no // or % in this
# language's arithmetic-mode system (arithmetic_mode.py's GetFloatBinOp has
# no case for either).
_CHECKED_FLOAT_BINOP_OPCODES: dict[tuple[str,str],type] = {
	( 'add', 'checked' ): ir.FAddCheck, ( 'add', 'wrapped' ): ir.AddWrap,
	( 'sub', 'checked' ): ir.FSubCheck, ( 'sub', 'wrapped' ): ir.SubWrap,
	( 'mul', 'checked' ): ir.FMulCheck, ( 'mul', 'wrapped' ): ir.MulWrap,
	( 'truediv', 'checked' ): ir.FloatDivCheck, ( 'truediv', 'wrapped' ): ir.FloatDiv,
}

# ast comparison operator -> the dunder method name to dispatch to
# (str.__eq__, i32.__lt__, etc. - lib/builtins/__scalar_dunders.py registers
# these for every scalar type too, so this isn't gated on left operand
# scalar-ness anymore). A type with nothing registered under the name (a
# class that never defined it, or NoneType) falls through to flat ir.Cmp.
_COMP_DUNDER: dict[type,str] = {
	ast.Eq: '__eq__',
	ast.NotEq: '__ne__',
	ast.Lt: '__lt__',
	ast.LtE: '__le__',
	ast.Gt: '__gt__',
	ast.GtE: '__ge__',
}

@dataclass( frozen = True )
class _LeafPairEq:
	''' _lower_eq_dispatch's own PASS 1 classification of one (left leaf
	type, right leaf type) grid cell - see that method's own docstring for
	the full per-kind rule. `method`/`reflected` are only meaningful for
	kind == 'cross_dunder': `reflected = False` means the match was found
	via left_type's own __eq__/__ne__ (receiver=left, arg=right);
	`reflected = True` means it was found via the REFLECTED call instead -
	right_type's own __eq__/__ne__ (receiver=right, arg=left). Both
	attempts look up the SAME method name (mirrors Python's real equality
	protocol: unlike an asymmetric operator such as `+`, which falls back
	to a DIFFERENTLY-NAMED `__radd__` on the right side, `==`/`!=` has no
	separate reflected-name method - only __eq__/__ne__ itself, with
	receiver/argument roles swapped for the second attempt). '''
	kind: str   # 'none_true' | 'none_false' | 'same_type' | 'cross_dunder' | 'error'
	method: Function|None = None
	reflected: bool = False

@dataclass( frozen = True )
class _LeafPairBinop:
	''' _lower_binop_dispatch's own PASS 1 classification of one (left leaf
	type, right leaf type) grid cell for a union-involving +-*//%|&^ - see
	that method's own docstring for the full per-kind rule. `success_type`
	is None only for 'error' (no valid operation for this leaf pair at
	all); `error_type` is None whenever this cell CAN'T fail as far as the
	OUTER aggregate is concerned - either the dunder's own return type
	isn't Result[T,E] at all, or it IS but `method.is_fallible_arithmetic`
	(a scalar-registered arithmetic dunder, or int.__floordiv__/__mod__)
	means its Result gets auto-consumed via the ambient arithmetic mode
	instead, exactly like the non-union path already does - see
	_emit_binop_cell's own handling. `method`/`reflected`, same meaning as
	_LeafPairEq's own pair - `reflected = True` means this was found via
	right_type's own REFLECTED, DIFFERENTLY-NAMED method (__radd__ etc,
	not __eq__/__ne__'s same-name-swapped-roles reflection - see
	_REFLECTED_BINOP_DUNDER's own comment on why binops need the
	different convention). No separate 'scalar' kind anymore - a Scalar
	operand's own arithmetic is just another dunder lookup now
	(i32.__add__ = ..., see lib/builtins/__scalar_dunders.py), found via the
	exact same _find_dunder_for_arg/_mode_qualified_dunder_names machinery
	the non-union path already uses - one source of truth for "how do I
	resolve a mode-qualified arithmetic dunder", not a second, independent
	reimplementation of GetBinOp/_resolve_checked_error. '''
	kind: str   # 'dunder' | 'error'
	success_type: Type|None = None
	error_type: Type|None = None
	method: Function|None = None
	reflected: bool = False

# ast.UnaryOp operator -> the dunder method name to dispatch to for a
# non-scalar operand (int.__neg__, ...). Scalar operands always go through
# arithmetic mode instead. ast.UAdd/ast.Invert are deliberately not mapped -
# no builtin type defines __pos__/__invert__ today.
_UNARYOP_DUNDER: dict[type,str] = {
	ast.USub: '__neg__',
}

# the two floating-point scalar stems. A float operand/target routes arithmetic,
# unary negate, and casts through the ArithmeticMode's GetFloat* methods (plain
# IEEE / inf-nan-checked, per mode) instead of the integer GetBinOp/GetUnaryOp/
# GetCast - see _lower_binop_values / _expr_UnaryOp / _lower_scalar_cast.
# `float`/`double` resolve to the same f32/f64 Scalar objects, so checking .stem
# covers all four spellings.
def _is_float_scalar( t: Type|None ) -> bool:
	return isinstance( t, Scalar ) and t.stem in ( 'f32', 'f64' )

# signed integer scalar stems - used to decide whether a checked Div/Mod can
# also raise OverflowError (signed INT_MIN/-1), see ir.BinOp.signed_only and
# _lower_arithmetic_op's error-set resolution
_SIGNED_INT_STEMS: frozenset[str] = frozenset([ 'i8', 'i16', 'i32', 'i64', 'i128', 'isize' ])

def _is_signed_scalar( t: Type|None ) -> bool:
	return isinstance( t, Scalar ) and t.stem in _SIGNED_INT_STEMS


# ast binary operators that have no floating-point meaning - bitwise/shift and
# floor-div/mod (Python's float // and % exist but aren't in this first pass).
# Rejected with a clear message before float arithmetic routing.
_FLOAT_UNSUPPORTED_BINOPS: dict[type,str] = {
	ast.BitAnd: '&', ast.BitOr: '|', ast.BitXor: '^',
	ast.LShift: '<<', ast.RShift: '>>',
	ast.FloorDiv: '//', ast.Mod: '%',
}

def _dedup_types( types: Iterable[Type] ) -> list[Type]:
	''' first-seen-order dedup by qualname, for _lower_binop_dispatch's own
	success/error type aggregation - discovery._get_or_create_union already
	dedupes internally, but the CALLER needs a deduped Python list first to
	decide whether to collapse to a single plain type (len == 1) or actually
	synthesize a union (len > 1). '''
	seen: dict[str,Type] = {}
	for t in types:
		seen.setdefault( t.qualname, t )
	return list( seen.values() )


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
		# BinOp/BoolOp/Compare/Constant/UnaryOp never produce RC values at
		# all, and Call is always fresh from the caller's perspective.
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

	def _find_iterator_next_method( self, owner_type: Type|None ) -> Function|None:
		# PLAN_GENERATORS.md Phase 3 - a non-failing probe (same posture as
		# _find_method above): "this type has no __next__" is a normal
		# outcome (falls through to the __len__/__getitem__ indexable path),
		# not an error. Shape validation (does __next__ actually return
		# T|None) happens once, in _lower_for_over_iterator itself, where a
		# real error location is available.
		return self._find_method( owner_type, '__next__' )

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
			and isinstance( owner_type, Specialization ) and isinstance( owner_type.base, Scalar )
			and owner_type.base.stem in ( 'Ptr', 'ConstPtr' )
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
	_FALLIBLE_METHOD_ALTERNATIVES = 'wrap this in `with compiler.panic_arithmetic(...):` instead'
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

	def _type_mentions_param( self, t: Type|None, tv: TypeVar ) -> bool:
		''' PLAN_RETURN_INFERENCE.md - true if the bare TypeVar `tv` occurs
		anywhere inside `t`, using the SAME structural recursion
		_unify_type_param itself uses (Specialization.args, CallableType.
		arg_types/return_type) - deliberately not the broader shape
		Monomorphizer.substitute_type_params uses (which also recurses into
		ClosureType/anonymous-TaggedUnion leaves): "does this parameter
		type CONTAIN tv" needs to agree exactly with "would _unify_type_param
		actually BIND tv from an argument at this position", or a type param
		that's structurally present but never actually unified against
		would be wrongly classified as argument-inferable and never get a
		chance at return-only inference at all. '''
		if t is None:
			return False
		if t is tv:
			return True
		if isinstance( t, Specialization ):
			return any( self._type_mentions_param( a, tv ) for a in t.args )
		if isinstance( t, CallableType ):
			return any( self._type_mentions_param( a, tv ) for a in t.arg_types ) or self._type_mentions_param( t.return_type, tv )
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


@dataclass
class _LoopContext:
	''' one entry of FunctionLowering._loop_labels - a for/while loop
	currently being lowered. continue_captured tracks whether a `continue`
	anywhere in the body actually emitted a Jump into continue_label - a
	for-loop's continue_label is a synthesized fallthrough point (the
	increment/back-edge code), not something the loop's own control flow
	ever jumps to on its own, so if no `continue` ever captured it, the
	label itself must be omitted rather than emitted-then-unused (a bare
	`goto`-less C label triggers -Wunused-label/C4102 on every compiler).
	A while-loop's continue_label is start_label instead, always already
	captured by the loop's own back edge, so this only matters for the
	three for-loop lowerers. '''
	continue_label: str
	break_label: str
	loop_snapshot: object
	continue_captured: bool = False


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

	def _emit_generator_dispatch_prologue( self, fn: Function ) -> None:
		''' PLAN_GENERATORS.md Phase F - a real state-check-and-goto
		dispatch, built directly as IR rather than synthesized AST like
		everything else in a generator's assembled $$__next__ body still
		is (see type_resolver.py's _build_generator_next_function's own
		docstring for why: Python's ast module has no goto statement to
		spell this with). For every (state, resume_label) TypeResolver.
		_assign_generator_yield_dispatch tagged onto fn.node (cached as
		node.generator_yield_states): `if self.__state == state: goto
		resume_label`. Reuses ordinary comparison lowering (a synthesized
		ast.Compare fed through _lower_expr) rather than hand-building
		the GetAttr/comparison IR directly - the exact same "borrow the
		real expression-lowering pipeline for a tiny synthesized
		snippet" trick this file's other generator hooks already use
		(_build_generator_error_defer_replay, etc.).

		State 0 ("not yet started") and the DONE sentinel (checked
		separately, by the assembled body's own leading ast.If - see
		_build_generator_next_function) both simply fail every check
		here and fall through into the body's own ordinary top, exactly
		as intended - this only ever needs to actively dispatch on a
		real mid-body suspend state. '''
		states = getattr( fn.node, 'generator_yield_states', None )
		if not states:
			return
		bool_cls = self.lowering.discovery.get_intrinsics()['bool']
		for state, resume_label in states:
			self_attr = ast.Attribute( value = ast.Name( id = 'self', ctx = ast.Load() ), attr = '__state', ctx = ast.Load() )
			compare = ast.Compare( left = self_attr, ops = [ ast.Eq() ], comparators = [ ast.Constant( value = state ) ] )
			ast.fix_missing_locations( ast.copy_location( compare, fn.node ) )
			cond = self._lower_expr( compare, bool_cls )
			skip_label = self._new_label( 'gen_dispatch_skip' )
			self._emit( ir.JumpIfFalse( cond = cond, target = skip_label ))
			self._emit( ir.Jump( target = resume_label ))
			self._emit( ir.Label( name = skip_label ))

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
					resolve_type = self.lowering._ensure_resolved,
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
		# or a captured-then-cancelled epilogue entry (cancel_flags() - see
		# cfg.py's _neutralize()). A defer flag starts False (disarmed until
		# the defer statement itself runs); a cancel flag starts the other
		# way, True (still needs releasing until whichever of move()/
		# deleted()/manually_decreffed() actually neutralizes its entry runs)
		flag_inits = [
			ir.Assign( dest = flag, src = ir.Const( type = flag.type, value = False ))
			for flag in self._defer_flags
		] + [
			ir.Assign( dest = flag, src = ir.Const( type = flag.type, value = True ))
			for flag in self._cfg.cancel_flags()
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
			for instr in self._cfg.attr_assign( attr, default_value, is_alias = self.lowering._is_aliasing_expr( attr.init, default_value )):
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

	def _emit_captured( self, instr: ir.Instruction ) -> None:
		# splices ONE instruction from a branch's own true_captured/false_
		# captured list (_stmt_If/_lower_binary_branch's own "lower this
		# branch into a SEPARATE instruction list first, decide true_extra/
		# false_extra via merge_if, THEN splice everything into the real
		# stream" technique) back into self._instructions - deliberately
		# NOT through self._emit() below: that instruction was ALREADY
		# _emit()'d once, when it was first captured (self._instructions
		# was redirected to the branch's own list at the time, but _emit()
		# itself ran, including its own fresh_temp() registration) - _emit()
		# has no way to tell "first time" from "being re-spliced", so
		# calling it a SECOND time here for the same ir.Call/Allocate
		# instruction RE-registers its own dest temp into cfg.py's
		# _temp_states, silently UNDOING whatever untracked it in between
		# (e.g. cfg.assign()'s own "ownership transferred into a named
		# binding, untrack the source temp" branch, if the branch's own
		# body assigned this Call's result into an EXISTING binding, like
		# `if flag: r = make_ok(b) else: r = make_err()` reassigning a
		# pre-declared `r`) - confirmed via a real reference leak (refcount
		# one too high after either branch of exactly that shape ran).
		# Every OTHER instruction kind is unaffected (self._emit()'s own
		# registration is gated on isinstance(instr, (Call, Allocate)), and
		# _check_self_escape_in is idempotent - re-running it on an
		# already-checked instruction is harmless, just redundant), so
		# this only needs to skip the ONE non-idempotent side effect,
		# not reimplement self._emit() from scratch.
		if self._current_fn is not None:
			self._check_self_escape_in( instr )
		self._instructions.append( instr )

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
		# PLAN_GENERATORS.md's defer/errdefer phase (Mechanism 2) - only
		# OVERRIDES self._generator_armed_defer_sites when `node` itself
		# carries type_resolver.py's own tag (_tag_armed_defer_sites tags
		# only the top-level statement of each preamble/segment slice, not
		# every nested descendant); otherwise this statement's own
		# recursive lowering (e.g. an ordinary nested if's own body)
		# simply inherits whatever an ENCLOSING tagged statement already
		# pushed, exactly like _arithmetic_mode's own stack semantics
		outer_armed = self._generator_armed_defer_sites
		tagged = getattr( node, 'generator_armed_defer_sites', None )
		if tagged is not None:
			self._generator_armed_defer_sites = tagged
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
			self._generator_armed_defer_sites = outer_armed

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

	def _incref_aliasing_return( self, node_expr: ast.expr, value: 'ir.Operand|None', *, force: bool = False ) -> None:
		''' shared by _stmt_Return and @inline splicing (_lower_inline_call/
		_splice_multi_statement_inline_body): an ALIASING return expression
		(self.lowering._is_aliasing_expr - `return self`/`return self.x`)
		hands back a reference someone else still independently owns, so the
		caller needs its own +1 - regardless of whether the return happens
		through a real call boundary or is spliced in directly. Skipping this
		for the spliced case (confirmed by a real refcount() repro) silently
		drops the Incref an @inline'd `return self` would otherwise get from
		a real, non-inlined call to the same function.

		`force` bypasses the has_live_entry() check below - needed by the
		@inline splice callers specifically: self/a parameter is bound
		zero-copy (SAME Variable identity as whatever the caller passed in -
		see _lower_inline_call's own "no _cfg_assign/incref here, deliberately"
		comment), so has_live_entry(value) would answer "does the CALLER's own
		operand happen to be a live owned local in the OUTER scope" instead of
		"is this splice's self/parameter borrowed" - the wrong question
		whenever the caller's argument was itself a plain owned local (exactly
		the str(s) repro: s has its own live entry in main(), so an unforced
		check wrongly concluded "already a move, no Incref needed"). @inline
		splice callers already know from their own binding loop that self/
		every parameter is always treated as borrowed at the splice boundary
		(same loop, same comment), so they pass force=True for those; a
		multi-statement splice's own pre-return-declared local (a real,
		splice-scoped self._cfg entry, not aliased to any outer identity)
		still needs the ordinary has_live_entry check, so force stays False
		for those. '''
		if value is None or not self.lowering._is_aliasing_expr( node_expr, value ):
			return
		if force or not self._cfg.has_live_entry( value ):
			for instr in self._cfg.incref( value.type, value ):
				self._emit( instr )

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
		# strict=False: this method already has its OWN, more complete
		# compatibility check just below (monomorphized-Specialization
		# comparison, CEnum-to-underlying, and _maybe_widen_return_result's
		# error-union widening) - _lower_expr's own general _check_assignable
		# would otherwise fire first and incorrectly reject exactly the
		# widening case this method exists to allow (`return x` where x:
		# Result[T,NarrowE] inside a function declared -> Result[T,WideE]).
		# The pre-existing TaggedUnion/RCClass-upcast coercions this method's
		# own comment below already expects still apply regardless (not
		# gated on strict - see _lower_expr's own comment)
		value = self._lower_expr( node.value, self._current_fn.return_type, strict = False ) if node.value is not None else None
		# an ALIASING return expression (self._is_aliasing_expr - a plain
		# Name/Attribute read, or a tuple-element Subscript) that does NOT
		# correspond to a live, skippable epilogue entry (self._cfg.
		# has_live_entry) needs its own Incref right here, before it's
		# handed off below: `return self.x` (an attribute read) and
		# `return self`/`return some_borrowed_param` (a BORROWED Name,
		# never pushed onto the epilogue stack - see cfg.py's
		# _enter_parameter()) both alias a reference that SOMEONE ELSE
		# still independently owns and will decref on their own schedule,
		# so the caller needs a genuinely separate +1, not a bare pointer
		# copy. An OWNED/COPY local (or a copy[T]/move[T] parameter) DOES
		# have a live entry - that's a real move (its own decref is what
		# current_epilogue_label()/return_() skip below, by this same
		# identity), and must NOT also get an Incref here, or the moved-
		# out reference would be permanently over-counted by one.
		# Confirmed by direct compile-and-run testing with compiler.
		# refcount(): `Holder.get(self) -> Box: return self.x` previously
		# hung onto only 2 references (the field + the caller's own new
		# holder of the returned value, double-counted as the SAME
		# reference) where 3 are live once the caller's copy exists,
		# leading to a premature free the moment either one dropped.
		self._incref_aliasing_return( node.value, value )
		# what actually gets returned/assigned into the return-value slot
		# below - defaults to `value` itself, reassigned to a widened temp
		# further down when the covered-Result-error-widening case applies.
		# `value` itself stays UNCHANGED throughout this whole method after
		# this point - every other use of it (check_unchecked_results,
		# current_epilogue_label, return_(), untrack_temp) needs the ORIGINAL
		# operand's own identity, not the widened temp's, to correctly
		# recognize "this tracked binding's own epilogue entry is the one
		# being moved out here" (see ir.WidenResult's own docstring)
		return_value = value
		# self._current_fn.return_type is Python None ONLY as a deliberate
		# sentinel (return-only generic type-param inference and lambda
		# eager-lowering both temporarily set it to None specifically so
		# _lower_expr(node.value, None) lets the return expression take its
		# OWN natural type, unconstrained, which then gets read back as the
		# inferred return type - see _infer_return_only_type_params/
		# _expr_Lambda's own "return_type_provisional" comments). A GENUINE
		# `-> None`-declared (or unannotated) function's own return_type is
		# always the real NoneType Scalar object (discovery.py's own
		# get_none_type()), never Python None - so this check only ever
		# skips during that provisional, not-yet-resolved state, never for
		# an actual declared return type
		if value is not None and self._current_fn.return_type is not None:
			# _lower_expr already applies every coercion it legitimately can
			# (union-leaf-wrap via _coerce_into_union, RCClass base-upcast via
			# _is_rcclass_upcast) - if value.type STILL doesn't match the
			# function's own declared return type afterward, this is a
			# genuine, uncaught mismatch that would otherwise emit a `return`
			# of the wrong C type (confirmed: `return 5` inside a `-> str`
			# function, or `return x` where x: Result[T,NarrowE] inside a
			# function declared -> Result[T,WideE] even though NarrowE is
			# covered by WideE, both previously compiled with zero errors and
			# produced C a real compiler rejects outright). The covered-but-
			# narrower-Result case specifically is a real, intentional gap for
			# now - it's handled by widening instead of rejection, added
			# separately (see the auto-widening this same function grows next
			# to this check).
			fn_type = self._current_fn.return_type
			# a generic ClassLike return type (Result[T,E], list[T], any
			# user generic class/@union) stays the ORIGINAL, un-monomorphized
			# Specialization on fn.return_type forever (it's parsed once,
			# from the function's own annotation, never re-resolved) - but a
			# local Variable's own .type (e.g. `result: list[str] = ...`)
			# DOES get monomorphized to the real, concrete ClassLike at some
			# point before this runs (confirmed: `result.type` here is
			# already a concrete RCClass, not the Specialization wrapping
			# list's abstract template) - so value.type and a bare fn_type
			# can legitimately be the "same type" while being different
			# objects. Monomorphize fn_type the same way before comparing -
			# mirrors _lower_expr's own identical "monomorphize a generic
			# union expected_type before comparing" step, generalized here
			# to every ClassLike kind (not just TaggedUnion), since this
			# same divergence isn't union-specific. Ptr[T]/ConstPtr[T]
			# (Scalar-based generics, not ClassLike) are deliberately
			# excluded - those are already consistently interned via plain
			# Specialization identity, confirmed by direct inspection, and
			# monomorphize_class doesn't apply to them anyway
			expected_concrete = fn_type
			if isinstance( fn_type, Specialization ) and isinstance( fn_type.base, ( RCClass, CStruct, CUnion, TaggedUnion, CEnum )):
				expected_concrete = self.lowering.monomorphize_class( fn_type )
			elif isinstance( fn_type, TupleType ) and fn_type.backing is not None:
				# same divergence, different shape: a tuple LITERAL (`return
				# (a, b, c)`) lowers to its own synthesized backing RCClass
				# directly (tuple_storage.TupleStorage.get()'s own real,
				# constructible representation), not the bare TupleType
				# wrapper the function's own `-> tuple[str,str,str]`
				# annotation stays as
				expected_concrete = fn_type.backing
			# a CEnum value flowing into a context expecting its OWN
			# underlying scalar type is also legitimate - "a CEnum has
			# exactly the same runtime representation as its underlying
			# type" (see the CEnum construction-call comment above), so
			# returning one where the underlying type is declared is a
			# value-preserving reinterpretation, not a mismatch. Both
			# directions, mirroring _check_assignable's own bidirectional
			# CEnum<->value_type exemption (lines ~4169/4171) - this method
			# can't just delegate to _check_assignable itself (see this
			# method's own strict=False comment above, on why that would
			# incorrectly reject the covered-Result-error widening case
			# before it's even attempted), so it has to re-derive every
			# exemption _check_assignable would apply; PLAN_COMPILER_BUG_
			# SWEEP.md's own audit found this had only ever re-derived ONE
			# of the two directions - `return raw_scalar` from a function
			# declared `-> SomeCEnum` (the OTHER direction) was wrongly
			# rejected, confirmed via a real repro
			is_cenum_to_underlying = isinstance( value.type, CEnum ) and value.type.value_type is fn_type
			is_underlying_to_cenum = isinstance( fn_type, CEnum ) and value.type is fn_type.value_type
			# the mirror image of expected_concrete above: value.type can
			# ALSO still be a raw, un-monomorphized Specialization here (a
			# compiler.checked_add(...)-style intrinsic's own check_dest,
			# see _lower_compiler_checked_binop's own comment on why IT
			# can't be pre-monomorphized either - the emitter needs its
			# Specialization .args) - monomorphize it the same way before
			# the identity comparison, rather than requiring every producer
			# of a same-statement-escaping Result value to guess which form
			# the compare side wants
			value_concrete = value.type
			if isinstance( value.type, Specialization ) and isinstance( value.type.base, ( RCClass, CStruct, CUnion, TaggedUnion, CEnum )):
				value_concrete = self.lowering.monomorphize_class( value.type )
			if (
				value.type is not fn_type and value.type is not expected_concrete
				and value_concrete is not fn_type and value_concrete is not expected_concrete
				and not is_cenum_to_underlying and not is_underlying_to_cenum
			):
				widened = self._maybe_widen_return_result( node, value, fn_type )
				if widened is None:
					self.lowering.discovery.fail(
						f'{ast.unparse(node)}: function returns '
						f'{fn_type.qualname if fn_type else "None"}, not {value.type.qualname if value.type else "?"}',
						node,
					)
				return_value = widened
		try:
			self._cfg.check_unchecked_results( value )
		except CompileError as e:
			self.lowering.discovery.fail( str( e ), node )
		# a fallible __init__'s own Err-path return, before construction
		# completes - forces the INLINE return_() unwind below rather than
		# ever letting current_epilogue_label() hand out one of the
		# function's shared closing-brace labels. That shared ladder is
		# built ONCE, using self._epilogue_stack's FINAL cancelled-state as
		# of the function's own closing brace - a LATER return in this same
		# __init__ that reaches complete_construction() (this construction's
		# eventual success path, cancelling every attribute entry so
		# ownership transfers cleanly into the now-complete self) would
		# retroactively wipe out the very decref this EARLIER Err return's
		# already-committed jump depends on, since Epilogue.cancelled is one
		# mutable flag shared by every jump into that entry's label, not a
		# per-jump-site snapshot. Confirmed by a real repro: an RC attribute
		# assigned before a later-failing validation, on the Err path,
		# silently stopped being released the moment a later Ok-path return
		# in the same __init__ completed construction - masked as a leak
		# (not a crash) only because the CALL SITE's own release of self
		# used to fall back to the generic per-class destructor's
		# unconditional field cascade, independently releasing the same
		# attribute again; that fallback is gone now (see
		# _lower_compiler_raw_free's own comment (used by the synthesized $$__new__'s Err branch) on why it had to be
		# removed - it also unconditionally touched attributes that were
		# NEVER assigned at all, reading uninitialized memory), so this
		# construction's own inline unwind is now the ONLY place whichever
		# attributes it assigned ever get released on this path
		construction_err_path = False
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
			else:
				construction_err_path = True
		label = None if construction_err_path else self._cfg.current_epilogue_label( value )
		# the innermost active multi-statement @inline splice, if this
		# return is reached from one of its own pre-return statements (see
		# _splice_multi_statement_inline_body/self._inline_scope_vars' own
		# comment) - value-computation/widening above is already correct
		# unchanged (self._current_fn.return_type is provisional's, i.e.
		# the INLINED function's own declared type), only the TERMINAL
		# emission below needs to redirect: into the scope's own result_var
		# instead of self._return_value_var, arming its exited_flag, and
		# (inline-unwind branch only) jumping to the scope's own merge_label
		# instead of emitting a real ir.Return - this early return must
		# never become the CALLER's own return
		inline_scope = self._inline_scope_vars[-1] if self._in_inline_splice_prelude and self._inline_scope_vars else None
		if label is not None:
			# whatever's still pending (RC decrefs, defer/errdefer replays)
			# gets unwound once, later, by the shared ladder every other
			# return reaching this same label also jumps into
			# (build_epilogue_ladder(), emitted at the function's own
			# closing brace - see _emit_epilogue; or, inside a splice, the
			# scope's own local ladder - see _splice_multi_statement_
			# inline_body) - value has to survive the jump some other way
			# than a direct ir.Return
			if inline_scope is not None:
				result_var, exited_flag, _merge_label = inline_scope
				if result_var is not None and value is not None:
					self._emit( ir.Assign( dest = result_var, src = return_value ))
				self._emit( ir.Assign( dest = exited_flag, src = ir.Const( type = exited_flag.type, value = True )))
			elif self._return_value_var is not None and value is not None:
				self._emit( ir.Assign( dest = self._return_value_var, src = return_value ))
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
			# right here, same as always (bounded to the splice's own
			# portion of the stack when inline_scope is set - see cfg.py's
			# return_() own comment). Still has to replay any pending
			# defer/errdefer entries itself (return_() does this now too -
			# they're just as "pending" as an RC decref from here)
			for instr in self._cfg.return_( value, lambda: self._build_is_err_check( node )):
				self._emit( instr )
			# same reasoning as the label-is-not-None branch above - flush
			# BEFORE this branch's own unconditional terminator, not after
			# (return_() already untracked `value` itself, so this only
			# ever cleans up OTHER still-pending temps - e.g. an
			# intermediate argument consumed into constructing `value`)
			self._flush_pending_temps()
			if inline_scope is not None:
				result_var, exited_flag, merge_label = inline_scope
				if result_var is not None and value is not None:
					self._emit( ir.Assign( dest = result_var, src = return_value ))
				self._emit( ir.Assign( dest = exited_flag, src = ir.Const( type = exited_flag.type, value = True )))
				# jumps PAST the scope's own ladder (already replayed
				# inline, right above - re-entering it via its own label
				# would replay the same entries a second time) straight to
				# where the early-exit-vs-normal-fallthrough merge begins
				self._cfg.mark_inline_scope_captured()
				self._emit( ir.Jump( target = merge_label ))
			else:
				self._emit( ir.Return( value = return_value ))

	def _maybe_widen_return_result( self, node: ast.Return, value: ir.Operand, fn_type: Type ) -> ir.Temp|None:
		''' `return x` where x is Result[T,NarrowE] and this function is
		declared -> Result[T,WideE] - if WideE genuinely COVERS NarrowE (every
		leaf of NarrowE is also a leaf of WideE - the identical leaves-
		containment rule type_resolver._require_result_return already applies
		for .or_return()/checked-arithmetic propagation), emit ir.WidenResult
		and return its dest temp for the caller to actually return/assign
		instead of `value`. Returns None (no widening applies) for every other
		shape of mismatch - the caller's own existing error message covers
		those. '''
		tr = self.lowering._type_resolver
		op_shape = tr._result_shape( value.type )
		fn_shape = tr._result_shape( fn_type )
		if op_shape is None or fn_shape is None:
			return None
		op_t, op_e = op_shape
		fn_t, fn_e = fn_shape
		if op_t is not fn_t or op_e is fn_e:
			return None # different Ok type entirely, or errors already match (not this method's concern)
		fn_e_leaves = tr._atomic_leaves( fn_e )
		if not all( leaf in fn_e_leaves for leaf in tr._atomic_leaves( op_e )):
			return None # op's error isn't covered by fn's - a genuine mismatch, not widenable
		# fn_type is guaranteed Result-shaped here (fn_shape matched), so
		# schedule/monomorphize it the same way _stmt_Return's own caller
		# already does for every OTHER ClassLike return type - dest.type
		# must be the real, concrete Result[T,WideE] the function actually
		# returns in C, not the abstract Specialization
		dest_type = self.lowering.monomorphize_class( fn_type ) if isinstance( fn_type, Specialization ) else fn_type
		self.lowering.schedule( dest_type )
		dest = self._new_temp( dest_type )
		self._emit( ir.WidenResult( dest = dest, src = value ))
		return dest

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
		existing = fn.get_local_or_raise( target.id )
		if not isinstance( existing, Variable ):
			self.lowering.discovery.fail( f'{target.id!r} is not a local variable, cannot del it', node )
		try:
			instructions = self._cfg.deleted( existing, self._current_fn.qualname )
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
			# same package-relative counting as discovery.py's own
			# visit_ImportFrom - see the comment there. This site previously
			# sliced the qualname without discovery's compensation for a
			# folded module, so a relative import written inside a function
			# body in a package's __init__.py climbed one level too far
			package = self.lowering.discovery.module_stack[-1].package
			strip = node.level - 1
			parts.extend(( package.split( '.' )[:-strip] if strip else package.split( '.' )) if package else [] )
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
			item = mod.get_local_or_raise( alias.name )
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
		# Volatile[T] resolves transparently to plain T (discovery.py's
		# visit_Subscript strips it) - detected here, separately, by peeking
		# at the raw annotation AST so this ONE call site (the only thing
		# that needs to know) can tag the resulting Variable's storage.
		is_volatile = ( isinstance( node.annotation, ast.Subscript ) and isinstance( node.annotation.value, ast.Name )
				and node.annotation.value.id == 'Volatile' )
		var_type = self.lowering.discovery.visit( node.annotation )
		if is_volatile and var_type.is_rc():
			self.lowering.discovery.fail( f'Volatile[...] does not support refcounted types: {ast.unparse(node)}', node )
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
			is_volatile = is_volatile,
		)
		fn.add_name( var.stem, var ) # scoped to the whole function body regardless of node.value (no block scoping - see cfg.py's own module docstring) - a bare declaration (node.value is None) deliberately does NOT mark it live in self._cfg (see below); a later real assignment does, via assign()'s own unconditional self._live.add()
		if node.value is not None:
			try:
				operand = self._lower_expr( node.value, var_type )
			except CompileError:
				# var is already registered (above) with a plausible
				# declared type but no real value/instructions behind it -
				# mark it broken so a later reference raises
				# RedundantCompilationError instead of using it as if it
				# were genuinely initialized (see Name.broken)
				var.broken = True
				raise
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
			for instr in self._cfg_assign( var, operand, is_alias = self.lowering._is_aliasing_expr( node.value, operand ), node = node ):
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

	def _static_field_type_or_none( self, node: ast.expr ) -> Type|None:
		''' like _static_type_of_value_expr, but NEVER calls discovery.fail()
		- any lookup miss (unknown name, a method/property instead of a
		plain field, a scope with no .names, ...) just returns None instead
		of hard-erroring. Needed specifically for _fixed_array_index_target's
		speculative "is this a FixedArrayType field?" peek: unlike sizeof(x)'s
		fallback (always a genuine error if it misses), a MISS here is the
		expected, common case - e.g. `obj.some_list_field[i]` or
		`obj.some_property[i]` (a property returning something indexable)
		must fall through to the ordinary _expr_Attribute/_lower_attr_target_obj
		handling unchanged, not be mistaken for a real error. '''
		if isinstance( node, ast.Name ):
			name = self.lowering.discovery.find_name_or_none( node.id )
			if not isinstance( name, Variable ) or name.broken:
				return None
			member = self._cfg.narrowed_member( node.id )
			return member.type if member is not None else name.type
		if isinstance( node, ast.Attribute ):
			owner_type = self._static_field_type_or_none( node.value )
			if owner_type is None:
				return None
			owner_type = self.lowering._ensure_resolved( owner_type )
			if isinstance( owner_type, Specialization ) and isinstance( owner_type.base, Scalar ) and owner_type.base.stem in ( 'Ptr', 'ConstPtr' ):
				owner_type = self.lowering._ensure_resolved( owner_type.args[0] )
			if isinstance( owner_type, ( CStruct, RCClass )):
				try:
					found = owner_type.chain_lookup( node.attr )
				except RedundantCompilationError:
					# a broken inherited member is exactly as much "not
					# statically known" as a genuine miss, for this
					# speculative-peek contract - never raise here
					return None
			else:
				names = getattr( owner_type, 'names', None )
				found = names.get( node.attr ) if isinstance( names, dict ) else None
			if not isinstance( found, Variable ) or found.broken:
				return None
			# resolve the VARIABLE itself first, same as _attr_lookup's own
			# `self._ensure_resolved( found ); return found` - found.type can
			# still be a lazily-deferred placeholder until found itself is
			# resolved (confirmed by a real repro: `g: Foo = make()` then
			# `g.b[i]`, Foo only reached indirectly through make()'s return
			# type rather than a direct Foo() construction in the same
			# statement, left found.type unresolved here and silently
			# misidentified a genuine FixedArrayType field as "not one")
			self.lowering._ensure_resolved( found )
			return found.type
		return None

	def _fixed_array_index_target( self, attr_node: ast.Attribute, index_node: ast.expr ) -> tuple[ir.Operand,str,FixedArrayType,ir.Operand]|None:
		''' `f.arr[i]` where `f.arr` (attr_node) statically resolves to a
		FixedArrayType field - returns (root_obj, attr_name, array_type,
		index_operand) ready for ir.GetAttrIndex/SetAttrIndex, or None if it
		doesn't (every other subscript shape is handled unchanged by the
		ordinary paths in _expr_Subscript/_stmt_Assign). Checked via
		_static_field_type_or_none FIRST, before lowering attr_node.value
		for real - that helper emits no IR, never fails, and never evaluates
		attr_node, so a non-match here doesn't double-evaluate the root
		object (the same "lower it exactly once" concern
		_lower_attr_target_obj's own docstring explains) and doesn't
		misfire a spurious error for some other legitimate attribute shape
		(property, method-value, ...). This is also why deeper/non-Name
		roots (e.g. `make().arr[i]`) fall through unrecognized rather than
		being specially rejected here: the helper only walks Name/Attribute
		chains, so anything else just resolves to None and reaches the
		ordinary whole-value-read rejection below, same as before this
		feature existed.

		A literal constant index out of [0, count) is rejected at compile
		time, same as tuple's own compile-time-constant index check - the
		one bit of free bounds checking available here (FixedArrayType's
		count, unlike a raw Ptr[T], is always known at compile time). A
		non-constant (runtime) index is otherwise unchecked, matching
		Ptr[T]/ConstPtr[T]'s own GetItem convention - this is a raw inline
		C array field, not a general-purpose bounds-checked container (see
		FixedArrayType's own docstring). '''
		array_type = self._static_field_type_or_none( attr_node )
		if not isinstance( array_type, FixedArrayType ):
			return None
		if ( isinstance( index_node, ast.Constant ) and isinstance( index_node.value, int )
				and not isinstance( index_node.value, bool ) and not ( 0 <= index_node.value < array_type.count )):
			self.lowering.discovery.fail(
				f'index {index_node.value} out of range for {array_type.qualname} (0..{array_type.count-1}): {ast.unparse(index_node)}',
				index_node,
			)
		root = self._lower_expr( attr_node.value, None )
		index_type = self.lowering.discovery.get_intrinsics()['usize']
		index = self._lower_expr( index_node, index_type )
		return root, attr_node.attr, array_type, index

	def _existing_local_or_none( self, target_id: str, node: ast.AST, context: str ) -> Variable|None:
		''' find_name_or_none, but treating a previously-BROKEN entry as if
		it weren't there at all - free to redeclare cleanly via
		_declare_local below, same as a genuinely first assignment, since
		nothing usable was ever produced for it. Shared by _stmt_Assign,
		_expr_NamedExpr (walrus), and _bind_loop_target - all three mirror
		the same "reuse existing, else declare fresh" rule (see their own
		comments). `context` only feeds the not-a-variable error message,
		which differs slightly per caller. '''
		existing = self.lowering.discovery.find_name_or_none( target_id )
		if existing is None or existing.broken:
			return None
		if not isinstance( existing, Variable ):
			self.lowering.discovery.fail( f'{target_id!r} is not a variable, {context}', node )
		return existing

	def _declare_local( self, target_id: str, node: ast.AST, lower_rhs: Callable[[Type|None],ir.Operand], *, default_type: Type|None = None ) -> tuple[Variable,ir.Operand]:
		''' the RHS is lowered BEFORE target_id is registered - not the
		other way around - because it may reference target_id itself: a
		bare Name target's own read/write desugaring (_stmt_AugAssign's
		`x += 1` -> `x = x + 1`, threaded straight through _stmt_Assign)
		relies on a genuinely-undeclared x's OWN read, inside that
		synthesized RHS, still failing "not defined" normally - registering
		x first would make that read find a freshly-declared, empty x
		instead. Only on FAILURE is a (broken) placeholder registered here,
		in the except clause - the actual fix for the gap that otherwise let
		a later, separate reference to target_id report a spurious "not
		defined" cascade on top of the real, original error (see
		Name.broken). Shared by the "no prior declaration" branches of
		_stmt_Assign, _expr_NamedExpr (walrus), and _bind_loop_target.
		`default_type` is only meaningful for _bind_loop_target's own
		fallback when value_expr has no type of its own to infer from (e.g.
		range()'s implicit literal start) - every other caller leaves it
		None, inferring purely from the RHS. '''
		fn = self._current_fn
		try:
			operand = lower_rhs( default_type )
		except CompileError:
			broken = Variable( stem = target_id, qualname = f'{fn.qualname}.{target_id}', file = fn.file, line = getattr( node, 'lineno', None ), type = None, broken = True )
			fn.add_name( broken.stem, broken )
			raise
		var = Variable( stem = target_id, qualname = f'{fn.qualname}.{target_id}', file = fn.file, line = getattr( node, 'lineno', None ), type = operand.type )
		fn.add_name( var.stem, var )
		self.lowering.schedule( var.type )
		return var, operand

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
			existing = self._existing_local_or_none( target.id, node, 'cannot assign to it' )
			if existing is not None:
				self._cfg.unnarrow( target.id ) # a real reassignment invalidates whatever this name was previously narrowed to - see cfg.py's own comment
				operand = self._lower_expr( node.value, existing.type )
				for instr in self._cfg_assign( existing, operand, is_alias = self.lowering._is_aliasing_expr( node.value, operand ), node = node ):
					self._emit( instr )
				self._emit( ir.Assign( dest = existing, src = operand ))
			else:
				# first assignment to a name with no prior declaration - same
				# as an AnnAssign, but the type is inferred from the RHS
				# instead of coming from an explicit annotation
				var, operand = self._declare_local( target.id, node, lambda expected: self._lower_expr( node.value, expected ))
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
				is_alias = self.lowering._is_aliasing_expr( node.value, operand )
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
				if getattr( node, 'is_match_binding', False ):
					# type_resolver.py's _match_pattern: a `case T(name):`
					# extracted payload. This language has no wildcard/discard
					# binding syntax (no Rust-style `case T(_):`), so a case
					# body that never reads `name` (`case Result.Err(e): pass`,
					# or one that builds a fresh, unrelated error instead of
					# reusing e) is entirely ordinary, expected code, not an
					# oversight - unlike a bare user-declared local's own
					# genuinely-forgotten unused value, there's no syntax the
					# user could have written instead to signal "discard this"
					# and silence a real warning. Mark it read unconditionally
					# (harmless when the arm DOES go on to use it - a real
					# later read isn't affected either way) rather than
					# leaving every such arm's own real, confirmed
					# -Wunused-variable/C4189 unfixable from the language side
					self._emit( ir.MarkUsed( operand = var ))
				match_clears_name = getattr( node, 'match_clears_name', None )
				if match_clears_name is not None:
					self._cfg.clear_result( match_clears_name )
		elif isinstance( target, ast.Attribute ):
			obj, writeback = self._lower_attr_target_obj( target.value )
			attr_var = self.lowering._attr_lookup( obj.type, target.attr, target )
			if isinstance( attr_var.type, FixedArrayType ):
				# same restriction as the GetAttr (read) side - a bare C array
				# member is never assignable via `=` (only a whole containing
				# struct/union is, via the compound-literal construction path
				# _lower_allocate_fields already handles) - see
				# FixedArrayType's own docstring
				self.lowering.discovery.fail(
					f'{ast.unparse(target)}: {attr_var.type.qualname} fields cannot be assigned after construction '
					f'(no element-level array access is implemented)',
					target,
				)
			operand = self._lower_expr( node.value, attr_var.type )
			if self._construction_self is not None and obj is self._construction_self:
				# self.<attr> = value, inside __init__ construction itself -
				# tracked for definite-assignment/self-escape purposes (see
				# RCCLASS ATTRIBUTE LIFETIME.md and cfg.attr_assign())
				for instr in self._cfg.attr_assign( attr_var, operand, is_alias = self.lowering._is_aliasing_expr( node.value, operand )):
					self._emit( instr )
			elif getattr( node, 'generator_first_rc_assign', False ):
				# PLAN_GENERATORS.md Phase 5 (roadmap Phase 5) - a generator's
				# own $$__next__ reassigning an RC-typed promoted local for
				# the FIRST time ever (type_resolver.py's _rename_and_track_
				# liveness splits every such reassignment into `if self.__
				# <stem>_live: <ordinary decref-old assign via attr_replace,
				# below> else: <THIS tagged assign>; self.__<stem>_live =
				# True`). The field's CURRENT value is still the construction-
				# time placeholder (a bare `0`, see _expr_Constant's own
				# generator_zero_rc_field exemption elsewhere in this file) -
				# reading it back and decref-ing it the ordinary way below
				# would compute &(NULL)->$header, a real, confirmed UBSan trap
				# (member access through a null pointer is UB even when the
				# member is at offset 0, and release_object's own runtime
				# null-check would otherwise make it harmless anyway) - so
				# there is no old value read/decref here at all, unlike both
				# branches above/below. Deliberately NOT routed through
				# attr_assign (unlike the construction-time branch just above)
				# even though the underlying need - "a fresh value, no prior
				# one to decref" - is the same: attr_assign also pushes a
				# 'self.<attr>'-keyed entry onto self.bindings/the epilogue
				# stack, a mechanism scoped to (and only ever reconciled
				# correctly by merge_if/cfg.py for) an actual __init__ under
				# construction - reusing it here, inside the ordinary if/else
				# type_resolver.py synthesizes around this branch, made
				# merge_if() see a 'self.<attr>' binding fresh on only one
				# branch and try to `del self._current_fn.names['self.<attr>']`
				# - a real KeyError, confirmed via a real repro, since no local
				# named 'self.<attr>' is ever registered in fn.names (that key
				# format is attr_assign's own bindings-dict convention, not a
				# real name lookup key). Only the two ordinary halves of what
				# a fresh RC value assignment needs are reproduced directly
				# instead, via the same public helpers _stmt_Return already
				# uses for an analogous "move ownership in, no bindings
				# tracking" need: incref the new value if it's an alias of an
				# existing tracked binding (mirrors attr_replace/attr_assign's
				# own is_alias branch), else untrack_temp() so the fresh temp's
				# own end-of-statement cleanup doesn't ALSO decref it now that
				# the field owns it.
				if self.lowering._is_aliasing_expr( node.value, operand ):
					for instr in self._cfg.incref( attr_var.type, operand ):
						self._emit( instr )
				else:
					self._cfg.untrack_temp( operand )
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
				for instr in self._cfg.attr_replace( attr_var.type, old, operand, is_alias = self.lowering._is_aliasing_expr( node.value, operand )):
					self._emit( instr )
			self._emit( ir.SetAttr( obj = obj, attr = target.attr, value = operand ))
			if writeback is not None:
				writeback( obj )
		elif isinstance( target, ast.Subscript ):
			if isinstance( target.value, ast.Attribute ):
				fixed = self._fixed_array_index_target( target.value, target.slice )
				if fixed is not None:
					root, attr, array_type, index = fixed
					operand = self._lower_expr( node.value, array_type.elem_type )
					self._emit( ir.SetAttrIndex( obj = root, attr = attr, index = index, value = operand ))
					return
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
		elif isinstance( target, ast.Tuple ):
			# `(a, b) = t` / `a, b = t` - both spellings parse to the same
			# ast.Assign(targets=[ast.Tuple(...)]) shape. node.value is
			# lowered exactly ONCE (not per-element) since it may be
			# side-effecting (sock.accept().or_return()) - each element is
			# then a raw GetAttr off the tuple's own _0/_1/... fields,
			# genuinely aliasing the tuple's own storage, the identical
			# shape _expr_Subscript's tuple-constant-index read already
			# established (and already fixed a real use-after-free for -
			# see its own is_tuple_element_read comment) - is_alias=True
			# unconditionally for a fresh declaration, re-derived from the
			# coercion result for a reassignment (a union-widening coerce
			# already increfs the leaf it wraps internally; treating that
			# as still-aliasing would double-incref).
			if any( isinstance( elt, ast.Starred ) for elt in target.elts ):
				self.lowering.discovery.fail( f'starred unpacking targets are not supported: {ast.unparse(node)}', node )
			if not all( isinstance( elt, ast.Name ) for elt in target.elts ):
				self.lowering.discovery.fail( f'unpacking targets must be plain names (nested tuple targets are not supported): {ast.unparse(node)}', node )
			fn = self._current_fn
			try:
				value = self._lower_expr( node.value, None )
				resolved_value_type = self.lowering._ensure_resolved( value.type )
				tuple_type = self.lowering._tuple_storage.tuple_type_for( resolved_value_type )
				if tuple_type is None:
					self.lowering.discovery.fail( f'cannot unpack a non-tuple value: {ast.unparse(node)}', node )
				if len( tuple_type.elem_types ) != len( target.elts ):
					self.lowering.discovery.fail(
						f'unpacking target has {len(target.elts)} name(s), value has {len(tuple_type.elem_types)}: {ast.unparse(node)}',
						node,
					)
				for i, elt in enumerate( target.elts ):
					assert isinstance( elt, ast.Name )
					attr_var = self.lowering._attr_lookup( resolved_value_type, f'_{i}', node )
					elem = self._new_temp( attr_var.type )
					self._emit( ir.GetAttr( dest = elem, obj = value, attr = f'_{i}' ))
					existing = self._existing_local_or_none( elt.id, node, 'cannot assign to it' )
					if existing is not None:
						self._cfg.unnarrow( elt.id )
						final = self._coerce_or_check_operand( elem, existing.type, node )
						is_alias = not getattr( final, 'is_union_coerce_result', False )
						for instr in self._cfg_assign( existing, final, is_alias = is_alias, node = node ):
							self._emit( instr )
						self._emit( ir.Assign( dest = existing, src = final ))
					else:
						var = Variable( stem = elt.id, qualname = f'{fn.qualname}.{elt.id}', file = fn.file, line = getattr( node, 'lineno', None ), type = elem.type )
						fn.add_name( var.stem, var )
						self.lowering.schedule( var.type )
						for instr in self._cfg_assign( var, elem, is_alias = True, node = node ):
							self._emit( instr )
						self._emit( ir.Assign( dest = var, src = elem ))
			except CompileError:
				for elt in target.elts:
					if isinstance( elt, ast.Name ) and self.lowering.discovery.find_name_or_none( elt.id ) is None:
						broken = Variable( stem = elt.id, qualname = f'{fn.qualname}.{elt.id}', file = fn.file, line = getattr( node, 'lineno', None ), type = None, broken = True )
						fn.add_name( broken.stem, broken )
				raise
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
		if self.lowering._is_compiler_call( node.value ) == '__raw_free__':
			self._lower_compiler_raw_free( node.value )
			return
		if isinstance( node.value, ast.Yield ):
			# PLAN_GENERATORS.md Phase F - a bare (statement-position)
			# `yield expr` inside a generator's $$__next__ body
			# (type_resolver.py only ever emits ast.Yield in a
			# is_generator_next function - see ensure_generator_
			# synthesized's own top-of-file docstring)
			self._lower_generator_yield( node.value )
			return
		if isinstance( node.value, ast.YieldFrom ):
			# PLAN_GENERATORS.md A.4a follow-up - type_resolver.py's
			# _desugar_generator_yield_from always rewrites a real `yield
			# from` into an ordinary for-loop before lowering ever runs,
			# so reaching HERE means the SAME "generator whose own
			# ensure_generator_synthesized run aborted partway through,
			# for an unrelated already-reported reason" situation _lower_
			# generator_yield's own identical check handles - see its
			# docstring for the full explanation. A graceful discovery.
			# fail() here too, instead of falling through to the generic
			# "unsupported expression statement" message below (which is
			# technically correct but confusingly generic for what's
			# really a cascading secondary error, not a new one).
			self.lowering.discovery.fail(
				f'{self._current_fn.qualname}: yield from reached outside a successfully-synthesized generator '
				f'(an earlier, already-reported error left this generator only partially built) - see PLAN_GENERATORS.md',
				node,
			)
			return
		if not isinstance( node.value, ast.Call ):
			self.lowering.discovery.fail( f'unsupported expression statement: {ast.unparse(node)}', node )
		self._lower_call( node.value, None, want_result = False )

	def _lower_generator_yield( self, node: ast.Yield ) -> None:
		''' PLAN_GENERATORS.md Phase F - a real `yield` suspend point:
		`self.__state = state` (an ordinary SetAttr - __state is a scalar
		usize field, never RC-typed, so this needs none of _stmt_Assign's
		decref-old-value machinery), the yielded value coerced against
		this function's own declared return type (mirrors _stmt_Return's
		identical coercion + _incref_aliasing_return call, deliberately
		WITHOUT everything else Return does - a yield doesn't unwind or
		jump to any epilogue; locals persist across a suspend by
		construction, see type_resolver.py's own live-flag-field
		mechanism for what actually makes that safe at the value level),
		then ir.Yield itself, then the resume Label.

		Composes for free with arbitrary nesting (if/while/for/with),
		unlike the old AST-synthesis unit-matcher this replaced: nothing
		here touches self._cfg's bindings/live-set at all, so lowering
		the WHOLE generator body once, in ordinary program order (exactly
		like any non-generator function's body), already leaves the
		compiler's own static RC-ownership view exactly where a real
		fall-through would - regardless of how many times a given
		textual point is actually reached at RUNTIME via a resume jump.
		No merge_if-style reconciliation is needed for the dispatch
		prologue's own jumps either, for the same reason: they're
		alternate ENTRY points into one linear lowering pass, not a fork
		requiring two independently-lowered branches to be reconciled.

		state/resume_label can genuinely be missing here (not just a
		theoretical "should never happen"): a generator whose own
		ensure_generator_synthesized run aborted partway through, for an
		unrelated already-reported reason (e.g. a `return` inside a
		defer/errdefer body - _reject_return_inside_generator_defer_body,
		called from _desugar_generator_defer_sites, BEFORE _assign_
		generator_yield_dispatch ever runs), leaves fn.node.body's own
		yields untagged AND fn.return_type never rewritten to the
		synthesized backing class - id(fn) is already memoized as
		"synthesized" by then (see ensure_generator_synthesized's own
		top), so nothing retries it, and the original, still-yield-
		bearing body can still reach real lowering via a later, unrelated
        call site. A graceful discovery.fail() here, not a raw crash - a
		single already-broken generator shouldn't be able to take down
		the whole compile run. '''
		self._emit_generator_yield_suspend( node )

	def _emit_generator_yield_suspend( self, node: ast.Yield ) -> None:
		''' PLAN_GENERATORS.md Phase F/Phase C - the suspend itself, shared
		by both the discarded (statement-position, `_lower_generator_
		yield` above) and captured (expression-position, `_expr_Yield`
		below) cases: `self.__state = state` (an ordinary SetAttr -
		__state is a scalar usize field, never RC-typed, so this needs
		none of _stmt_Assign's decref-old-value machinery), the yielded
		value coerced against this function's own declared return type
		(mirrors _stmt_Return's identical coercion + _incref_aliasing_
		return call, deliberately WITHOUT everything else Return does - a
		yield doesn't unwind or jump to any epilogue; locals persist
		across a suspend by construction, see type_resolver.py's own
		live-flag-field mechanism for what actually makes that safe at
		the value level), then ir.Yield itself, then the resume Label.
		`_expr_Yield` picks up from here to read back whatever `.send()`
		delivered - this method itself has no idea whether that's even
		possible (SendType may not be declared at all), that's entirely
		its own caller's concern.

		Composes for free with arbitrary nesting (if/while/for/with),
		unlike the old AST-synthesis unit-matcher this replaced: nothing
		here touches self._cfg's bindings/live-set at all, so lowering
		the WHOLE generator body once, in ordinary program order (exactly
		like any non-generator function's body), already leaves the
		compiler's own static RC-ownership view exactly where a real
		fall-through would - regardless of how many times a given
		textual point is actually reached at RUNTIME via a resume jump.
		No merge_if-style reconciliation is needed for the dispatch
		prologue's own jumps either, for the same reason: they're
		alternate ENTRY points into one linear lowering pass, not a fork
		requiring two independently-lowered branches to be reconciled.

		state/resume_label can genuinely be missing here (not just a
		theoretical "should never happen"): a generator whose own
		ensure_generator_synthesized run aborted partway through, for an
		unrelated already-reported reason (e.g. a `return` inside a
		defer/errdefer body - _reject_return_inside_generator_defer_body,
		called from _desugar_generator_defer_sites, BEFORE _assign_
		generator_yield_dispatch ever runs), leaves fn.node.body's own
		yields untagged AND fn.return_type never rewritten to the
		synthesized backing class - id(fn) is already memoized as
		"synthesized" by then (see ensure_generator_synthesized's own
		top), so nothing retries it, and the original, still-yield-
		bearing body can still reach real lowering via a later, unrelated
        call site. A graceful discovery.fail() here, not a raw crash - a
		single already-broken generator shouldn't be able to take down
		the whole compile run. '''
		state = getattr( node, 'generator_yield_state', None )
		resume_label = getattr( node, 'generator_resume_label', None )
		if state is None or resume_label is None:
			self.lowering.discovery.fail(
				f'{self._current_fn.qualname}: yield reached outside a successfully-synthesized generator '
				f'(an earlier, already-reported error left this generator only partially built) - see PLAN_GENERATORS.md',
				node,
			)
			return

		self_name = ast.Name( id = 'self', ctx = ast.Load() )
		ast.copy_location( self_name, node )
		self_obj, writeback = self._lower_attr_target_obj( self_name )
		usize_cls = self.lowering.discovery.get_intrinsics()['usize']
		self._emit( ir.SetAttr( obj = self_obj, attr = '__state', value = ir.Const( type = usize_cls, value = state )))
		if writeback is not None:
			writeback( self_obj )

		value_node = node.value if node.value is not None else ast.Constant( value = None )
		value = self._lower_expr( value_node, self._current_fn.return_type, strict = False )
		# strict=False (mirrors _stmt_Return's own identical call) skips
		# _lower_expr's own built-in final rejection - so, same as _stmt_
		# Return, this needs its OWN explicit check afterward: _lower_
		# expr already applies every coercion it legitimately can (union-
		# leaf-wrap via _coerce_into_union, RCClass base-upcast); if
		# value.type STILL doesn't match this function's own declared
		# return type, that's a genuine, uncaught mismatch that would
		# otherwise silently emit a `return` of the wrong C type - a
		# scoped-down version of _stmt_Return's own check (no CEnum
		# bidirectional exemption, no Result-error widening - neither
		# realistically arises for a generator's own elem_type|None/
		# Result[elem_type|None,error_type] return shape), confirmed via
		# a real repro: `for x in range(n): yield x * 10` inside an
		# Iterator[i32] generator (range()'s own loop counter is usize,
		# not i32) compiled with zero errors and produced C a real
		# compiler rejects outright (`return $tN;` returning a bare
		# uintptr_t where the union struct is expected) - found while
		# testing A.4a's own nested-for-loop generalization, but
		# reproduces identically with no nesting involved at all.
		if value.type is not None and self._current_fn.return_type is not None:
			fn_type = self._current_fn.return_type
			expected_concrete = fn_type
			if isinstance( fn_type, Specialization ) and isinstance( fn_type.base, ( RCClass, CStruct, CUnion, TaggedUnion, CEnum )):
				expected_concrete = self.lowering.monomorphize_class( fn_type )
			value_concrete = value.type
			if isinstance( value.type, Specialization ) and isinstance( value.type.base, ( RCClass, CStruct, CUnion, TaggedUnion, CEnum )):
				value_concrete = self.lowering.monomorphize_class( value.type )
			if (
				value.type is not fn_type and value.type is not expected_concrete
				and value_concrete is not fn_type and value_concrete is not expected_concrete
			):
				self.lowering.discovery.fail(
					f'{ast.unparse(node)}: yield produces {fn_type.qualname if fn_type else "None"}, '
					f'not {value.type.qualname if value.type else "?"}',
					node,
				)
		self._incref_aliasing_return( value_node, value )
		# mirrors _stmt_Return's own identical call (its simpler, no-
		# shared-epilogue-label branch - a yield needs none of that
		# branch's OTHER machinery, cfg.return_()'s own epilogue-style
		# unwind of every other still-live local, since a yield's own
		# suspend must NOT decref anything else - locals persist across
		# it by construction): `value`'s own ownership is about to
		# transfer into ir.Yield below (a bare temp, if the yielded
		# expression needed coercing into this function's own union
		# return type - `yield self.<field>`, the common case). Without
		# this, _flush_pending_temps right after ALSO decrefs it,
		# silently cancelling out the coercion's own incref (confirmed
		# via a real refcount() repro: a captured RC parameter yielded
		# back through a match arm read compiler.refcount() one lower
		# than expected).
		self._cfg.untrack_temp( value )
		self._flush_pending_temps()
		self._emit( ir.Yield( value = value, state = state, resume_label = resume_label ))
		self._emit( ir.Label( name = resume_label ))

	def _expr_Yield( self, node: ast.Yield, expected_type: Type|None ) -> ir.Operand:
		''' PLAN_GENERATORS.md Phase C - `(yield expr)` used as an
		EXPRESSION: the suspend itself is identical to the ordinary
		statement-position case (_emit_generator_yield_suspend, shared) -
		what's new is what happens AFTER resuming. Only legal inside a
		generator that declared a SendType (Generator[T,SendType,E]);
		Iterator[T]/the 2-arg Generator[T,E] form have no SendType to
		ever deliver a captured yield's own value through, so this fails
		clearly instead of reaching lowering with nothing to read back
		(mirrors the pre-Phase-C behavior exactly - _lower_expr's own
		getattr-based dispatch already rejected any expression-position
		yield as "unsupported expression" before this method existed at
		all, just with a much less specific message).

		Reads back self.__send_ready: True means send(v) armed __send_
		slot since this exact suspend point - clears the flag (consumed,
		one-shot) and reads __send_slot as this expression's own value; a
		PLAIN field read, no incref of its own - _is_aliasing_expr now
		recognizes a captured ast.Yield as aliasing (same category
		ast.Attribute already is - it reads an existing field, self.
		__send_slot's own reference, independent of whatever __send_slot
		itself keeps), so wherever this operand ultimately lands (`held =
		yield i`, an argument, ...) gets its own Incref through the SAME
		ordinary consumption-time machinery any other field read already
		relies on - no special-casing needed here beyond that one
		recognition fix. False means this resume came from a bare
		__next__()/for-loop consumption instead (send_ready never armed) -
		panics with a message pointing at .send(), mirroring Python's own
		runtime behavior for resuming a captured yield without sending a
		value. Composes for FREE with anything wrapping the yield (`x =
		yield v`, `x = (yield v).or_return()`, ...) since this is ordinary
		expression lowering - no special-casing needed for nesting, same
		reason Phase F's own dispatch mechanism generalized nesting/
		multiplicity for free. '''
		send_type = getattr( self._current_fn.node, 'generator_send_type', None )
		if send_type is None:
			self.lowering.discovery.fail(
				f'{self._current_fn.qualname}: yield can only be used as an expression inside a generator that declares a '
				f'SendType (Generator[T,SendType,E]) - a bare `yield expr` statement is still supported everywhere else - '
				f'see PLAN_GENERATORS.md',
				node,
			)
			send_type = self._current_fn.return_type # keep going with SOME type so the rest of this method still produces a well-typed (if meaningless) operand, same "don't cascade into confusing follow-on errors" posture the rest of this file uses after a reported failure

		self._emit_generator_yield_suspend( node )

		self_name = ast.Name( id = 'self', ctx = ast.Load() )
		ast.copy_location( self_name, node )
		bool_cls = self.lowering.discovery.find_name( 'bool', node )
		ready_read = ast.Attribute( value = self_name, attr = '__send_ready', ctx = ast.Load() )
		ast.copy_location( ready_read, node )
		ready = self._lower_expr( ready_read, bool_cls )

		panic_label = self._new_label( 'gen_send_required' )
		end_label = self._new_label( 'gen_send_end' )
		self._emit( ir.JumpIfFalse( cond = ready, target = panic_label ))

		# true branch: a pending send exists - consume it (clear the flag,
		# one-shot) and deliver __send_slot as this expression's value
		self_obj, writeback = self._lower_attr_target_obj( self_name )
		self._emit( ir.SetAttr( obj = self_obj, attr = '__send_ready', value = ir.Const( type = bool_cls, value = False )))
		if writeback is not None:
			writeback( self_obj )
		slot_read = ast.Attribute( value = ast.Name( id = 'self', ctx = ast.Load() ), attr = '__send_slot', ctx = ast.Load() )
		ast.copy_location( slot_read, node )
		ast.copy_location( slot_read.value, node )
		slot_val = self._lower_expr( slot_read, send_type, strict = False )
		dest = self._new_temp( send_type )
		self._emit( ir.Assign( dest = dest, src = slot_val ))
		self._emit( ir.Jump( target = end_label ))

		# false branch: resumed via __next__()/for-loop consumption
		# instead of .send() - panic, same statement-position AST-
		# synthesis + swap-buffer technique _build_generator_error_defer_
		# replay/_build_generator_pessimistic_done_pin already use to
		# reuse ordinary _lower_call/_lower_stmt machinery for a call this
		# file has no other reason to hand-build IR for directly
		self._emit( ir.Label( name = panic_label ))
		panic_call = ast.Call(
			func = ast.Attribute( value = ast.Name( id = 'sys', ctx = ast.Load() ), attr = 'panic', ctx = ast.Load() ),
			args = [ ast.Constant( value = f'{self._current_fn.qualname}: resumed via __next__()/for-loop consumption at a captured yield - use .send() instead' ) ],
			keywords = [],
		)
		panic_stmt = ast.Expr( value = panic_call )
		ast.copy_location( panic_call, node ); ast.copy_location( panic_stmt, node )
		ast.fix_missing_locations( panic_stmt )
		self._lower_stmt( panic_stmt )
		self._emit( ir.Label( name = end_label ))
		return dest

	def _stmt_With( self, node: ast.With ) -> None:
		if len( node.items ) != 1:
			self.lowering.discovery.fail( f'unsupported with statement: {ast.unparse(node)}', node )
		item = node.items[0]
		context_expr = item.context_expr

		if item.optional_vars is None:
			defer_kind = self.lowering._defer_kind_of_with( context_expr )
			if defer_kind is not None:
				self._register_defer_block( is_err_only = ( defer_kind == 'errdefer' ), body = node.body, node = node )
				return

			attr = self.lowering._is_compiler_attr( context_expr )
			mode: arithmetic_mode.ArithmeticMode|None = None
			if attr == 'wrap_arithmetic':
				mode = arithmetic_mode.ArithmeticWrap()
			elif attr == 'saturate_arithmetic':
				mode = arithmetic_mode.ArithmeticSaturate()
			elif self.lowering._is_compiler_call( context_expr ) == 'panic_arithmetic':
				if len( context_expr.args ) != 1 or context_expr.keywords:
					self.lowering.discovery.fail( f'compiler.panic_arithmetic(...) takes exactly one argument: {ast.unparse(node)}', node )
				str_cls = self.lowering.discovery.find_name( 'str', node )
				errmsg = self._lower_expr( context_expr.args[0], str_cls )
				mode = arithmetic_mode.ArithmeticPanic( errmsg )

			if mode is not None:
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
				return

		# general context-manager form: with EXPR [as NAME]: BODY - EXPR's
		# type supplies __enter__(self)/__exit__(self), neither of the
		# special-cased shapes above (defer/errdefer, compiler.*_arithmetic)
		self._lower_with_context_manager( node, item )

	def _lower_with_context_manager( self, node: ast.With, item: ast.withitem ) -> None:
		''' `with EXPR [as NAME]: BODY` for a user-defined context manager -
		EXPR's type must supply __enter__(self)->T and __exit__(self)->None.
		Desugars to (as plain AST, fed back through the ordinary statement
		pipeline, same idiom type_resolver.py's own desugaring passes use):
			__with_ctx_N = EXPR
			[NAME = ]__with_ctx_N.__enter__()
			with defer: __with_ctx_N.__exit__()
			BODY
		reusing _register_defer_block for the guaranteed-once-per-entry,
		runs-on-every-exit-path contract - same "not allowed inside a loop"
		restriction defer/errdefer already have (see the check below), and
		the same "not inside a generator's own body" restriction (Mechanism
		2's defer-replay is a SEPARATE, generator-specific path this doesn't
		integrate with yet - a clean rejection, not attempted here). Real
		per-iteration loop scoping (`with timeout(...): read(...)` inside a
		request loop, PLAN_NON_BLOCKING_IO's own motivating case) is real,
		separate future work, same as it would be for defer/errdefer.
		__exit__ always runs unconditionally on every exit path - this
		compiler has no Python-style exception propagation for __exit__ to
		observe or suppress, so there's no exc_type/exc_value/traceback
		parameter, unlike Python's own protocol; a Result-returning
		__enter__/__exit__ works the same as any other bare/bound call
		(the ordinary "an unconsumed Result is a compile error" rule
		applies exactly as it would to hand-written code, nothing special
		here consumes or requires one).

		__with_ctx_N/NAME (`as NAME`) are ORDINARY, function-scoped locals,
		exactly like any other name introduced anywhere in this compiler
		(cfg.py's own module docstring: no block scoping at all) - NOT torn
		down early at this with-statement's own textual end. An earlier
		version of this feature DID scope them (reusing _stmt_If's own
		branch-confinement machinery, treating the whole with-block as an
		unconditionally-taken "if branch") - wrong, and reverted: `with`
		does not introduce a lifetime scope (real Python's own `with EXPR as
		NAME:` doesn't either - NAME stays bound and alive for the rest of
		the enclosing function/scope there too), and a with-statement's
		BODY - unlike an if-branch's body - always executes exactly once
		when reached, so ANY local BODY itself declares needs to survive
		past the with-statement's own end the same way it would if the same
		statements were written with no with-statement wrapping them at
		all. Confirmed via a real repro: `with Ctx(): x: i32 = 5` followed
		by `return x` wrongly reported `'x' is not initialized on all code
		branches` under the branch-confinement version - `merge_if`'s
		confinement applies to EVERY binding newly introduced inside the
		window, not just __with_ctx_N/NAME, so there was no way to confine
		only those two without also breaking every ordinary local BODY
		declares. `with x:` on an EXISTING object (not a fresh construction)
		is the other motivating case: __with_ctx_N then aliases x rather
		than owning a fresh construction, but ordinary aliasing assignment
		in this language (`ctx = x`, a plain Name read) still takes its own
		independent Incref - same as any other `y = x` - so it stays
		correctly balanced by its own (now function-scoped, not block-
		scoped) eventual release; nothing here needs to special-case that
		case, it just needs to NOT be forced into an artificial block scope
		that has no basis in either this language's own "no block scoping"
		design or real Python's own `with` semantics (which also introduces
		no new scope - NAME/x stay bound and alive for the rest of the
		enclosing scope there too). '''
		if self._loop_depth > 0:
			self.lowering.discovery.fail(
				'with-statement (context manager) is not allowed inside a loop - call another function and use the '
				f'with-statement inside that instead: {ast.unparse(node)}', node,
			)
		if self._current_fn.is_generator_next:
			self.lowering.discovery.fail(
				f'with-statement (context manager) is not supported inside a generator body yet: {ast.unparse(node)}', node,
			)

		index = self._with_ctx_id
		self._with_ctx_id += 1
		ctx_name = f'__with_ctx_{index}'

		context_expr = item.context_expr
		ctx_assign = ast.Assign( targets = [ ast.Name( id = ctx_name, ctx = ast.Store() ) ], value = context_expr )
		ast.fix_missing_locations( ast.copy_location( ctx_assign, node ))
		self._lower_stmt( ctx_assign )

		ctx_var = self._existing_local_or_none( ctx_name, node, 'with-statement context expression' )
		assert ctx_var is not None # just declared immediately above - _lower_stmt would have raised on failure
		ctx_type = ctx_var.type

		if self.lowering._find_method( ctx_type, '__enter__' ) is None or self.lowering._find_method( ctx_type, '__exit__' ) is None:
			type_name = ctx_type.qualname if ctx_type is not None else '?'
			self.lowering.discovery.fail(
				f'with-statement requires {type_name} to define both __enter__(self) and __exit__(self): {ast.unparse(node)}', node,
			)

		def _ctx_read() -> ast.Name:
			n = ast.Name( id = ctx_name, ctx = ast.Load() )
			ast.fix_missing_locations( ast.copy_location( n, node ))
			return n

		enter_call = ast.Call( func = ast.Attribute( value = _ctx_read(), attr = '__enter__', ctx = ast.Load() ), args = [], keywords = [] )
		ast.fix_missing_locations( ast.copy_location( enter_call, node ))
		enter_stmt: ast.stmt
		if item.optional_vars is not None:
			assert isinstance( item.optional_vars, ast.Name )
			enter_stmt = ast.Assign( targets = [ ast.Name( id = item.optional_vars.id, ctx = ast.Store() ) ], value = enter_call )
		else:
			enter_stmt = ast.Expr( value = enter_call )
		ast.fix_missing_locations( ast.copy_location( enter_stmt, node ))
		self._lower_stmt( enter_stmt )

		def _make_exit_stmt() -> ast.stmt:
			# a FRESH node every call, never reused across the two sites
			# below - lowering attaches mutable per-occurrence attributes to
			# a node as it processes it (same reason _build_defer_replay_
			# guards' own resume_call lambda in type_resolver.py rebuilds
			# fresh each time, not once and shared)
			call = ast.Call( func = ast.Attribute( value = _ctx_read(), attr = '__exit__', ctx = ast.Load() ), args = [], keywords = [] )
			ast.fix_missing_locations( ast.copy_location( call, node ))
			stmt = ast.Expr( value = call )
			ast.fix_missing_locations( ast.copy_location( stmt, node ))
			return stmt

		# _register_defer_block gives FUNCTION-scoped semantics (runs once,
		# from here to wherever the function actually ends, regardless of
		# what code follows this with-statement) - exactly right for an
		# early return/break/continue reached from INSIDE this with-block's
		# own body, but too broad on its own: a with-statement's __exit__
		# must run when THIS BLOCK is left, not merely "sometime before the
		# function ends". So: register it as a defer (covers every early-
		# exit path from inside the body below), then - only if the body
		# can actually fall through to its own natural end (_stmt_diverges:
		# false unless every path through the body already returns/breaks/
		# continues) - explicitly DISARM that defer's flag and call
		# __exit__() directly, right here, matching the block's real
		# lexical extent. Without the disarm, a with-statement followed by
		# more code in the same function would see __exit__ fire twice:
		# once here (if this were a plain second call with no disarm) AND
		# again later at the function's own eventual exit - confirmed via a
		# real compile+run repro (list.append() call counts, __exit__'s own
		# side effects observably running at the wrong point in the
		# program's actual output order, not just "eventually").
		self._register_defer_block( is_err_only = False, body = [ _make_exit_stmt() ], node = node )
		exit_flag = self._defer_flags[-1] # the one push_defer above just armed

		for stmt in node.body:
			# same per-statement recovery boundary as the arithmetic-mode/
			# defer-body cases above
			try:
				self._lower_stmt( stmt )
			except CompileError:
				continue

		if not node.body or not self._stmt_diverges( node.body[-1] ):
			bool_cls = self.lowering.discovery.find_name( 'bool', node )
			self._emit( ir.Assign( dest = exit_flag, src = ir.Const( type = bool_cls, value = False )))
			self._lower_stmt( _make_exit_stmt() )

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
		# `is not None`, NOT a truthy `:=` check - NoneType's own sizeof is
		# a legitimate 0 (see discovery.py's get_none_type()), and 0 is
		# falsy, so a truthy check here wrongly fell through to the
		# RCClass/CStruct/CUnion/TaggedUnion-only branch below and failed
		# with "compiler.sizeof(NoneType) is not supported yet" - see that
		# type's own sizeof field for why 0 there is real, not a "missing"
		# sentinel
		sizeof_attr = getattr( target_type, 'sizeof', None )
		if sizeof_attr is not None:
			return ir.Const( type = expected_type or usize_cls, value = sizeof_attr )
		# Ptr[T]/ConstPtr[T] is always exactly one machine pointer wide, whatever
		# T is - fold to the Ptr/ConstPtr intrinsic's own sizeof. A Specialization
		# carries no sizeof of its own, so the plain getattr above misses it;
		# without this, sys.alloc[Ptr[None]]( 1 ) on the POSIX threading path
		# (Thread.__init__'s pthread_create out-slot) fails to compile with
		# "compiler.sizeof(Ptr[None]) is not supported yet".
		if self.lowering._type_resolver._is_ptr_specialization( target_type ):
			return ir.Const( type = expected_type or usize_cls, value = target_type.base.sizeof )
		# a C type declared via compiler.c_type('pthread_mutex_t', ...) -
		# as opaque to this compiler as a real ClassLike; stays a real
		# ir.SizeOf, letting the C compiler itself compute it
		if isinstance( target_type, CType ):
			self.lowering.discovery.required_headers.add( target_type.required_header )
			dest = self._new_temp( expected_type or usize_cls )
			self._emit( ir.SizeOf( dest = dest, type = target_type ))
			return dest

		# a FixedArrayType field (u8[N]/u16[N]-style) - folds to a compile-
		# time constant the same way a plain Scalar's sizeof already does,
		# PROVIDED the element type itself has a plain-int sizeof (true for
		# every real element type this compiler's FixedArrayType is
		# actually exercised with today - u8/u16/etc). An element type
		# without one (a real class-like type, e.g. a hypothetical
		# SomeStruct[N] field) would need the C compiler's own sizeof(...)
		# to size correctly, same as any other class-like type below - not
		# attempted here since no real FixedArrayType field with a non-
		# scalar element type exists anywhere in this codebase yet; falls
		# through to the same "not supported yet" error below rather than
		# silently computing a wrong Python-int size for a type with no
		# sizeof of its own.
		if isinstance( target_type, FixedArrayType ):
			elem_sizeof = getattr( target_type.elem_type, 'sizeof', None )
			if elem_sizeof is not None:
				return ir.Const( type = expected_type or usize_cls, value = elem_sizeof * target_type.count )

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

	def _lower_compiler_checked_binop( self, node: ast.Call, intrinsic_name: str, kind: str, mode: str, expected_type: Type|None ) -> ir.Operand:
		# compiler.checked_add(a, b)/wrapped_add(a, b)/saturated_add(a, b)/
		# checked_sub(...)/.../checked_truediv(a, b)/wrapped_truediv(a, b) -
		# FIXED primitives (unlike bare `+`, which reads self._arithmetic_
		# mode[-1] to pick BETWEEN AddCheck/AddWrap/AddSaturate): each of
		# these always resolves to exactly one opcode PER OPERAND TYPE (int
		# vs float pick a different opcode for the same kind/mode pair -
		# see _CHECKED_BINOP_OPCODES/_CHECKED_FLOAT_BINOP_OPCODES below -
		# but which one is picked never depends on ambient arithmetic mode).
		# Intended body for a scalar-registered, @inline'd `__add__`/
		# `__wrapped_add__`/`__saturated_add__`/etc (see lib/builtins/
		# __scalar_dunders.py) - which opcode a given SOURCE `+`/`//`/etc
		# actually gets still comes from ambient mode picking which of
		# these dunders binop dispatch resolves to (mode-qualified name
		# lookup in _lower_binop_values), not from anything read here.
		if len( node.args ) != 2 or node.keywords:
			self.lowering.discovery.fail( f'compiler.{intrinsic_name}(...) takes exactly two positional arguments: {ast.unparse(node)}', node )
		left = self._lower_expr( node.args[0], None )
		right = self._lower_expr( node.args[1], None )
		is_float = _is_float_scalar( left.type )
		opcode = ( _CHECKED_FLOAT_BINOP_OPCODES if is_float else _CHECKED_BINOP_OPCODES ).get(( kind, mode ))
		if not isinstance( left.type, Scalar ) or left.type is not right.type or opcode is None:
			self.lowering.discovery.fail(
				f'compiler.{intrinsic_name}(...) arguments must both be the same scalar type supporting {kind!r} - got '
				f'{left.type.qualname if left.type else "?"} and {right.type.qualname if right.type else "?"}: {ast.unparse(node)}',
				node,
			)
		result_type = expected_type if expected_type is not None and isinstance( expected_type, Scalar ) else left.type
		if not opcode.checked_errors:
			# wrapped_*/saturated_* on add/sub/mul, wrapped_truediv - fully
			# infallible, _lower_arithmetic_op's own infallible branch
			# (single instruction, no Result at all) is exactly right
			return self._lower_arithmetic_op( node, opcode, None, result_type, { 'left': left, 'right': right }, 'binary' )
		# checked_*, and wrapped_/saturated_floordiv/mod (still fallible -
		# ZeroDivisionError persists in every mode, see ir.py's own
		# DivWrap/DivSaturate/ModWrap/ModSaturate comments) - deliberately
		# does NOT auto-consume the way a bare checked `+` would
		# (_lower_arithmetic_op's own check-mode branch always does,
		# regardless of caller context - arithmetic modes don't translate
		# through a function call boundary, inline or not, by design). This
		# returns the RAW Result[result_type,error_type] value instead,
		# matching a @fallible_arithmetic dunder's own declared return type
		# exactly - so whatever calls/inlines this dunder can uniformly
		# consume it via ambient mode at the DISPATCH site
		# (_emit_binop_dunder_call's is_fallible_arithmetic handling), the
		# same way whether this call ends up inlined or not.
		result_cls = self.lowering.discovery.find_name( 'Result', node )
		error_type, _alternatives = self._resolve_checked_error( node, opcode, result_type )
		check_type = self.lowering.discovery._get_or_create_specialization( result_cls, [ result_type, error_type ] )
		self.lowering.schedule( check_type )
		# NOT monomorphized - _emit_check_arith (emitter_c.py) needs this
		# temp's type to stay a Specialization (reads .args[0] for the
		# success type) exactly like _emit_checked_op's identical, older
		# check_dest already relies on. This value deliberately escapes as
		# a real `return` value though (unlike _emit_checked_op's own
		# check_dest, always consumed same-statement via _consume_checked_
		# result) - see _stmt_Return's own comment on why it tries
		# monomorphize_class(value.type) too, not just value.type itself,
		# to match a Specialization-typed return against a function's
		# already-monomorphized declared return type.
		check_dest = self._new_temp( check_type )
		self._emit( opcode( dest = check_dest, left = left, right = right ))
		return check_dest

	def _lower_compiler_checked_convert( self, node: ast.Call, expected_type: Type|None ) -> ir.Operand:
		# compiler.checked_convert(T, x) - the single fixed-mode intrinsic
		# behind every scalar .to_T() conversion method (lib/builtins/
		# __scalar_dunders.py) - a genuine numeric VALUE-range check against
		# T's own [MIN,MAX], independent of bit width (unlike compiler.
		# cast(T,x)/T(x) construct-cast syntax, which only range-checks a
		# NARROWING conversion - see _lower_scalar_cast). Always fallible
		# (ir.ConvertCheck), unlike checked_add/etc's wrapped_*/saturated_*
		# siblings - there's no "wrapped"/"saturated" variant of a value-
		# range check that means anything different, so .to_T() only ever
		# needs this one intrinsic (see mode-consumption at the DISPATCH
		# site - ambient mode still governs how the Result gets consumed,
		# same as any other @fallible_arithmetic method, just never changes
		# the check itself).
		if len( node.args ) != 2 or node.keywords:
			self.lowering.discovery.fail( f'compiler.checked_convert(...) takes exactly two arguments (target type, value): {ast.unparse(node)}', node )
		target_type = getattr( node.args[0], 'resolved_type', None )
		if target_type is None:
			target_type = self.lowering._try_resolve_namespace( node.args[0] )
		if not isinstance( target_type, Scalar ) or target_type.stem in ( 'f32', 'f64' ):
			self.lowering.discovery.fail(
				f'compiler.checked_convert(...) first argument must be an integer scalar type: {ast.unparse(node)}', node,
			)
		operand = self._lower_expr( node.args[1], None )
		if not isinstance( operand.type, Scalar ) or operand.type.stem in ( 'f32', 'f64' ):
			self.lowering.discovery.fail(
				f'compiler.checked_convert(...) second argument must be an integer scalar value, got '
				f'{operand.type.qualname if operand.type else "?"}: {ast.unparse(node)}',
				node,
			)
		result_cls = self.lowering.discovery.find_name( 'Result', node )
		overflow_cls = self.lowering.discovery.find_name( 'OverflowError', node )
		# NOT monomorphized - same reasoning as _lower_compiler_checked_binop's
		# own check_dest above: _emit_convert_check (emitter_c.py) needs
		# this temp's type to stay a Specialization (reads .args[0] for the
		# target type)
		check_type = self.lowering.discovery._get_or_create_specialization( result_cls, [ target_type, overflow_cls ] )
		self.lowering.schedule( check_type )
		check_dest = self._new_temp( check_type )
		self._emit( ir.ConvertCheck( dest = check_dest, operand = operand ))
		return check_dest

	# (kind, mode) -> opcode for Ptr[T]/ConstPtr[T] +/- usize -> Ptr[T].
	# Deliberately reuses the SAME AddCheck/AddWrap/SubCheck/SubWrap opcodes
	# _CHECKED_BINOP_OPCODES already uses for plain scalar add/sub - both
	# _emit_check_arith and _emit_wrap_arith (emitter_c.py) already branch
	# on a POINTER-typed dest_type correctly (byte-offset uintptr_t round-
	# trip, never sizeof(T)-scaled) - confirmed via Spike B, this already
	# works today via the (about-to-be-retired) fallback path, just needs
	# wiring through dunders now. No 'saturated' entry - saturating pointer
	# arithmetic is a clean compile-time rejection instead (confirmed with
	# the user: an address isn't a bounded numeric range the way an int is,
	# "clamp to min/max" has no coherent meaning) - see
	# _lower_compiler_ptr_binop's own handling of that mode.
	_PTR_BINOP_OPCODES: dict[tuple[str,str],type] = {
		( 'add', 'checked' ): ir.AddCheck, ( 'add', 'wrapped' ): ir.AddWrap,
		( 'sub', 'checked' ): ir.SubCheck, ( 'sub', 'wrapped' ): ir.SubWrap,
	}

	def _lower_compiler_ptr_binop( self, node: ast.Call, intrinsic_name: str, kind: str, mode: str, expected_type: Type|None ) -> ir.Operand:
		# compiler.checked_ptr_add(p, offset)/wrapped_ptr_add(...)/
		# saturated_ptr_add(...)/checked_ptr_sub(...)/wrapped_ptr_sub(...)/
		# saturated_ptr_sub(...) - the fixed-mode intrinsics behind Ptr[T]/
		# ConstPtr[T]'s own __add__/__sub__ dunders (lib/builtins/
		# __ptr_arith.py) for the Ptr[T] +/- usize -> Ptr[T] shape (pointer
		# MINUS pointer, yielding a distance, is a separate shape/dunder -
		# see compiler.ptr_sub_dist). Parallel to, not sharing code with,
		# _lower_compiler_checked_binop - that method hard-requires
		# left.type is right.type, but here the two operand types genuinely
		# differ (Ptr[T], usize).
		if len( node.args ) != 2 or node.keywords:
			self.lowering.discovery.fail( f'compiler.{intrinsic_name}(...) takes exactly two positional arguments: {ast.unparse(node)}', node )
		left = self._lower_expr( node.args[0], None )
		right = self._lower_expr( node.args[1], None )
		if not self.lowering._type_resolver._is_ptr_specialization( left.type ):
			self.lowering.discovery.fail(
				f'compiler.{intrinsic_name}(...) first argument must be a Ptr[T]/ConstPtr[T]: {ast.unparse(node)}', node,
			)
		usize_cls = self.lowering.discovery.get_intrinsics()['usize']
		if right.type is not usize_cls:
			self.lowering.discovery.fail(
				f'compiler.{intrinsic_name}(...) second argument must be usize, got '
				f'{right.type.qualname if right.type else "?"}: {ast.unparse(node)}',
				node,
			)
		if mode == 'saturated':
			self.lowering.discovery.fail(
				f'saturating pointer arithmetic is not supported (an address is not a bounded numeric range - '
				f'use checked or wrap mode instead): {ast.unparse(node)}',
				node,
			)
		opcode = self._PTR_BINOP_OPCODES[( kind, mode )]
		result_type = left.type
		if not opcode.checked_errors:
			return self._lower_arithmetic_op( node, opcode, None, result_type, { 'left': left, 'right': right }, 'binary' )
		result_cls = self.lowering.discovery.find_name( 'Result', node )
		error_type, _alternatives = self._resolve_checked_error( node, opcode, result_type )
		check_type = self.lowering.discovery._get_or_create_specialization( result_cls, [ result_type, error_type ] )
		self.lowering.schedule( check_type )
		check_dest = self._new_temp( check_type )
		self._emit( opcode( dest = check_dest, left = left, right = right ))
		return check_dest

	def _lower_compiler_ptr_diff( self, node: ast.Call, expected_type: Type|None ) -> ir.Operand:
		# compiler.ptr_sub_dist(a, b) - Ptr[T]/ConstPtr[T] - Ptr[T]/ConstPtr[T]
		# -> isize, the fixed-opcode intrinsic behind Ptr[T]/ConstPtr[T]'s own
		# __sub__ dunder for the pointer-MINUS-pointer shape (ptr_sub_dist[T]
		# in lib/builtins/__ptr_arith.py) - distinct from the Ptr[T]-usize ->
		# Ptr[T] offset-subtraction shape _lower_compiler_ptr_binop handles
		# above (same __sub__ name, disambiguated at the dunder-lookup level
		# by _find_dunder_for_arg's arg-type matching, not here). Infallible,
		# single opcode - see ir.PtrDiff's own comment for why there's no
		# wrap/check/saturate split for a pointer distance.
		if len( node.args ) != 2 or node.keywords:
			self.lowering.discovery.fail( f'compiler.ptr_sub_dist(...) takes exactly two positional arguments: {ast.unparse(node)}', node )
		left = self._lower_expr( node.args[0], None )
		right = self._lower_expr( node.args[1], None )
		if not self.lowering._type_resolver._is_ptr_specialization( left.type ) or left.type is not right.type:
			self.lowering.discovery.fail(
				f'compiler.ptr_sub_dist(...) arguments must both be the same Ptr[T]/ConstPtr[T] type - got '
				f'{left.type.qualname if left.type else "?"} and {right.type.qualname if right.type else "?"}: {ast.unparse(node)}',
				node,
			)
		isize_cls = self.lowering.discovery.get_intrinsics()['isize']
		return self._lower_arithmetic_op( node, ir.PtrDiff, None, isize_cls, { 'left': left, 'right': right }, 'binary' )

	# ir.Shr (>>) joins these deliberately: right-shift by a valid amount is
	# always well-defined (this compiler doesn't check shift-amount-exceeds-
	# width for either direction - a separate, pre-existing, out-of-scope
	# concern), so it's single-opcode/infallible exactly like the bitwise
	# ops - unlike << (Shl), which DOES have real Wrap/Check/Saturate
	# variants (shifting bits out the top is a real, already-modeled
	# concern) and flows through _lower_compiler_checked_binop instead,
	# parallel to add/sub/mul.
	_BITWISE_OPCODES: dict[str,type] = { 'bitand': ir.BitAnd, 'bitor': ir.BitOr, 'bitxor': ir.BitXor, 'rshift': ir.Shr }

	def _lower_compiler_bitwise( self, node: ast.Call, intrinsic_name: str, expected_type: Type|None ) -> ir.Operand:
		# compiler.bitand/bitor/bitxor/rshift(a, b) - the fixed-opcode
		# intrinsics behind every scalar __and__/__or__/__xor__/__rshift__
		# dunder (lib/builtins/__scalar_dunders.py). Unlike checked_add/etc,
		# there is only ONE variant each - ir.BitAnd/BitOr/BitXor/Shr have
		# no Wrap/Check/Saturate forms at all - always infallible, plain
		# T-returning, no Result involved.
		if len( node.args ) != 2 or node.keywords:
			self.lowering.discovery.fail( f'compiler.{intrinsic_name}(...) takes exactly two positional arguments: {ast.unparse(node)}', node )
		left = self._lower_expr( node.args[0], None )
		right = self._lower_expr( node.args[1], None )
		opcode = self._BITWISE_OPCODES[intrinsic_name]
		if not isinstance( left.type, Scalar ) or left.type is not right.type or _is_float_scalar( left.type ):
			self.lowering.discovery.fail(
				f'compiler.{intrinsic_name}(...) arguments must both be the same integer scalar type - got '
				f'{left.type.qualname if left.type else "?"} and {right.type.qualname if right.type else "?"}: {ast.unparse(node)}',
				node,
			)
		result_type = expected_type if expected_type is not None and isinstance( expected_type, Scalar ) else left.type
		return self._lower_arithmetic_op( node, opcode, None, result_type, { 'left': left, 'right': right }, 'binary' )

	# comparison never overflows - single variant each, same "always
	# infallible, plain bool-returning" shape as the bitwise ops above, no
	# ambient arithmetic mode to disambiguate. Backs every scalar comparison
	# dunder (lib/builtins/__scalar_dunders.py's scalar_eq/scalar_ne/
	# scalar_lt/scalar_le/scalar_gt/scalar_ge) - the same fixed-opcode-
	# intrinsic pattern _BITWISE_OPCODES/_lower_compiler_bitwise already use,
	# letting ordinary comparison dispatch (_expr_Compare/_lower_eq_or_ne/
	# _lower_operand_compare/_classify_leaf_pair_eq) find a real dunder for
	# Scalar operands too, instead of hardcoding a flat ir.Cmp as the only
	# possible outcome for a Scalar left operand.
	_CMP_INTRINSIC_OPCODES: dict[str,'ir.CmpOp'] = {
		'cmp_eq': ir.CmpOp.EQ, 'cmp_ne': ir.CmpOp.NE,
		'cmp_lt': ir.CmpOp.LT, 'cmp_le': ir.CmpOp.LE,
		'cmp_gt': ir.CmpOp.GT, 'cmp_ge': ir.CmpOp.GE,
	}

	def _lower_compiler_cmp( self, node: ast.Call, intrinsic_name: str, expected_type: Type|None ) -> ir.Operand:
		# compiler.cmp_eq/cmp_ne/cmp_lt/cmp_le/cmp_gt/cmp_ge(a, b) -> bool -
		# emits the exact same ir.Cmp _expr_Compare's own flat-Cmp fallback
		# does (see its own docstring), just reachable as a real callable
		# intrinsic so a Scalar-, Ptr[T]/ConstPtr[T]-, or CEnum-registered
		# dunder body can delegate to it. Ptr[T]/ConstPtr[T] (a Specialization
		# wrapping a Scalar base, not itself `isinstance(_, Scalar)` - see
		# _is_ptr_specialization) needs the same plain address comparison a
		# bare scalar gets (a C pointer compares natively with ==/!=/</etc,
		# same ir.Cmp opcode, no different codegen) - backs Ptr.__eq__/etc
		# (lib/builtins/__ptr_arith.py). CEnum lowers to a plain C
		# typedef'd int (emitter_c.py's c_type - never struct/union-
		# prefixed), so it compares exactly the same native way too - backs
		# the auto-synthesized CEnum __eq__/etc (type_resolver.py's
		# _synthesize_cenum_comparisons).
		if len( node.args ) != 2 or node.keywords:
			self.lowering.discovery.fail( f'compiler.{intrinsic_name}(...) takes exactly two positional arguments: {ast.unparse(node)}', node )
		left = self._lower_expr( node.args[0], None )
		right = self._lower_expr( node.args[1], None )
		is_ptr = self.lowering._type_resolver._is_ptr_specialization( left.type )
		if not ( isinstance( left.type, ( Scalar, CEnum ) ) or is_ptr ) or left.type is not right.type:
			self.lowering.discovery.fail(
				f'compiler.{intrinsic_name}(...) arguments must both be the same scalar, Ptr[T]/ConstPtr[T], or CEnum type - got '
				f'{left.type.qualname if left.type else "?"} and {right.type.qualname if right.type else "?"}: {ast.unparse(node)}',
				node,
			)
		cmp_op = self._CMP_INTRINSIC_OPCODES[intrinsic_name]
		bool_cls = self.lowering.discovery.find_name( 'bool', node )
		dest = self._new_temp( bool_cls )
		self._emit( ir.Cmp( dest = dest, op = cmp_op, left = left, right = right ))
		return dest

	def _lower_compiler_addrof( self, node: ast.Call, expected_type: Type|None ) -> ir.Operand:
		# compiler.addrof(x) -> Ptr[T], translating directly to C's &x - x
		# must be a bare local variable/parameter name (matches SYNTAX.md's
		# "local variable" wording and C's own lvalue-only restriction on
		# &), OR exactly one level of plain field access ROOTED at one
		# (x.field, e.g. compiler.addrof(addr.sin_addr) - the common "fill
		# one field of a stack struct via a C out-parameter" FFI idiom,
		# e.g. inet_pton(af, str, &addr.sin_addr)). Investigated and
		# confirmed safe for exactly this shape: a bare Name's own operand
		# (_expr_Name) is always a genuine, stable-lifetime Variable (never
		# a Temp) - requiring the CHAIN'S ROOT to be one too is what
		# preserves that same "well-defined, stable lvalue" guarantee one
		# level deeper (a plain field of an already-stable object is itself
		# just as stable/addressable - C's own `&x.field`/`&x->field`).
		# Deliberately NOT extended to a Call/Subscript/deeper chain root
		# (e.g. compiler.addrof(get_foo().field) or compiler.addrof(a.b.c)):
		# a Call's own result is a TEMPORARY whose lifetime this compiler's
		# RC discipline doesn't guarantee outlives the current statement (a
		# real dangling-pointer risk, not just an implementation gap), and a
		# deeper chain is un-analyzed extra scope, not needed by the one
		# real motivating case - both rejected with a clear message rather
		# than silently mishandled.
		if len( node.args ) != 1 or node.keywords:
			self.lowering.discovery.fail( f'compiler.addrof(...) takes exactly one argument: {ast.unparse(node)}', node )
		arg_node = node.args[0]
		if isinstance( arg_node, ast.Subscript ) and isinstance( arg_node.value, ast.Attribute ):
			# compiler.addrof(x.field[i]) where field is a FixedArrayType ->
			# Ptr[ElemType] at element i specifically (unlike the bare
			# compiler.addrof(x.field) case above, which only ever gives
			# element 0 via array-to-pointer decay). Same root-must-be-a-
			# bare-local safety requirement as every other addrof shape
			# here, checked explicitly since _fixed_array_index_target
			# itself doesn't enforce it (by design - GetAttrIndex/
			# SetAttrIndex's own callers only ever need a VALUE, which
			# doesn't share addrof's dangling-pointer concern about the
			# root's lifetime).
			attr_node = arg_node.value
			if not isinstance( attr_node.value, ast.Name ):
				self.lowering.discovery.fail(
					f'compiler.addrof(...) indexed field-access argument must be rooted at a bare local variable, '
					f'not {ast.unparse(node)} (only one level of field access is supported)',
					node,
				)
			fixed = self._fixed_array_index_target( attr_node, arg_node.slice )
			if fixed is None:
				self.lowering.discovery.fail(
					f'compiler.addrof(...) subscript argument must index a fixed-size array field: {ast.unparse(node)}',
					node,
				)
			root, attr, array_type, index = fixed
			ptr_cls = self.lowering.discovery.get_intrinsics()['Ptr']
			elem_ptr_type = self.lowering.discovery._get_or_create_specialization( ptr_cls, [ array_type.elem_type ] )
			dest = self._new_temp( elem_ptr_type )
			self._emit( ir.AddrOfArrayIndex( dest = dest, obj = root, attr = attr, index = index ))
			return dest
		if isinstance( arg_node, ast.Attribute ):
			if not isinstance( arg_node.value, ast.Name ):
				self.lowering.discovery.fail(
					f'compiler.addrof(...) field-access argument must be rooted at a bare local variable, '
					f'not {ast.unparse(node)} (only one level of field access is supported)',
					node,
				)
			root = self._lower_expr( arg_node.value, None )
			attr_var = self.lowering._attr_lookup( root.type, arg_node.attr, arg_node )
			ptr_cls = self.lowering.discovery.get_intrinsics()['Ptr']
			if isinstance( attr_var.type, FixedArrayType ):
				# compiler.addrof(x.field) where field is ElemType[N] ->
				# Ptr[ElemType], via C's own array-to-pointer decay - NOT
				# &(x.field), which would be a pointer TO the array
				# (ElemType(*)[N]), a different C type than the declared
				# Ptr[ElemType] destination even though the address value
				# is identical. Safe for the same reason ordinary field
				# addrof is: `obj` is a real, stable-lifetime lvalue (a
				# bare local, or one level of field access rooted at one),
				# and a fixed-size array member of a stable object is
				# itself just as stable. See ir.ArrayFieldPtr's own
				# docstring for the emission this builds.
				elem_ptr_type = self.lowering.discovery._get_or_create_specialization( ptr_cls, [ attr_var.type.elem_type ] )
				dest = self._new_temp( elem_ptr_type )
				self._emit( ir.ArrayFieldPtr( dest = dest, obj = root, attr = arg_node.attr ))
				return dest
			pointee = attr_var.type
			# same RC-pointee-depth rule the bare-Name path below applies -
			# see its own comment for why (Ptr[Foo] already spells `Foo*`,
			# so &-ing an RC-typed FIELD needs the same extra Ptr level)
			if self.lowering._type_resolver._is_RC( pointee ):
				pointee = self.lowering.discovery._get_or_create_specialization( ptr_cls, [ pointee ] )
			ptr_type = self.lowering.discovery._get_or_create_specialization( ptr_cls, [ pointee ] )
			dest = self._new_temp( ptr_type )
			self._emit( ir.AddrOfField( dest = dest, obj = root, attr = arg_node.attr ))
			return dest
		if not isinstance( arg_node, ast.Name ):
			self.lowering.discovery.fail( f'compiler.addrof(...) argument must be a bare local variable, not {ast.unparse(node)}', node )
		value = self._lower_expr( arg_node, None )
		ptr_cls = self.lowering.discovery.get_intrinsics()['Ptr']
		# &value's C type is one pointer level deeper than value's OWN storage.
		# For a scalar/CStruct a variable stores the value directly, so that is
		# Ptr[T]. But an RCClass value is itself stored as a pointer (a `Foo`
		# variable holds a `Foo*`), so &value is `Foo**` - which in this type
		# system is Ptr[Ptr[Foo]], because Ptr[Foo] already spells `Foo*` (see
		# emitter_c._value_spelling's "single pointer even when RCClass" rule,
		# which sys.alloc[Foo] -> Ptr[Foo] relies on and must stay). Getting this
		# level right is what makes the emitted `dest = &value` type-check under
		# GCC and Linux-clang; Windows clang/MSVC only warned, so the old
		# Ptr[Foo] result (`Foo*` assigned from `Foo**`) slipped through. The one
		# caller that hits the RC path - list/UnsafeList.append/insert via
		# compiler.cast( Ptr[None], compiler.addrof( val ) ) - casts straight to
		# void*, so the extra level is invisible downstream. (expected_type is
		# intentionally NOT consulted: addrof's result type is determined solely
		# by the operand; callers wanting another pointer type cast explicitly.)
		pointee = value.type
		if self.lowering._type_resolver._is_RC( pointee ):
			pointee = self.lowering.discovery._get_or_create_specialization( ptr_cls, [ pointee ] )
		ptr_type = self.lowering.discovery._get_or_create_specialization( ptr_cls, [ pointee ] )
		dest = self._new_temp( ptr_type )
		self._emit( ir.AddrOf( dest = dest, value = value ))
		return dest

	def _c_field_name_literal( self, name_node: ast.expr, fn_name: str, node: ast.Call ) -> str:
		# one level of field access ONLY, same restriction compiler.addrof's
		# own field-access shape already has (see its own comment) - a
		# dotted path ('uc_stack.ss_sp') can't reuse GetAttr/SetAttr/
		# AddrOfField's existing 'attr: str' emission unchanged
		# (mangle_qualname corrupts a literal '.' into '$', producing wrong
		# C) - reaching a second-level opaque field means composing two
		# single-level compiler.c_field*() calls instead (see
		# _lower_compiler_c_field_addr's own comment)
		if not ( isinstance( name_node, ast.Constant ) and isinstance( name_node.value, str ) and name_node.value.isidentifier() ):
			self.lowering.discovery.fail(
				f'compiler.{fn_name}(...) field name must be a plain identifier string literal '
				f'(one level of field access, no dots): {ast.unparse(node)}',
				node,
			)
		return name_node.value

	def _require_c_type_pointer( self, ptr_type: Type|None, fn_name: str, node: ast.Call ) -> None:
		if not (
			isinstance( ptr_type, Specialization )
			and isinstance( ptr_type.base, Scalar )
			and ptr_type.base.stem in ( 'Ptr', 'ConstPtr' )
			and isinstance( ptr_type.args[0], CType )
		):
			self.lowering.discovery.fail(
				f'compiler.{fn_name}(...) first argument must be Ptr[T]/ConstPtr[T] where T is a '
				f'compiler.c_type(...): {ast.unparse(node)}',
				node,
			)

	def _lower_compiler_c_field( self, node: ast.Call, expected_type: Type|None ) -> ir.Operand:
		# compiler.c_field(ptr, 'field_name', T) -> T - reads a field of an
		# OPAQUE compiler.c_type(...) struct through ptr (Ptr[SomeCType]/
		# ConstPtr[SomeCType]), trusting the caller's asserted type T the
		# same way compiler.cexpr's/compiler.sizeof's own type arguments are
		# trusted - nothing here verifies field_name or T against the real
		# header's actual layout; a mismatch is caught by the C compiler
		# once it sees the real struct definition, exactly like an
		# @extern(header=...) signature mismatch already is. See
		# lib/posix/pthread.py's own comment on why ucontext_t needs this:
		# CType has no known field layout at all (unlike a @cstruct/
		# @interface), so ordinary x.field access has nowhere to look the
		# field up (_attr_lookup requires chain_lookup or a .names dict,
		# neither of which CType has).
		if len( node.args ) != 3 or node.keywords:
			self.lowering.discovery.fail( f'compiler.c_field(ptr, field_name, type) takes exactly 3 positional arguments: {ast.unparse(node)}', node )
		ptr_node, name_node, type_node = node.args
		ptr = self._lower_expr( ptr_node, None )
		field_name = self._c_field_name_literal( name_node, 'c_field', node )
		self._require_c_type_pointer( ptr.type, 'c_field', node )
		field_type = self.lowering._try_resolve_namespace( type_node )
		if not isinstance( field_type, Type ):
			self.lowering.discovery.fail( f'compiler.c_field(...) third argument must be a type: {ast.unparse(node)}', node )
		dest = self._new_temp( field_type )
		self._emit( ir.GetAttr( dest = dest, obj = ptr, attr = field_name ))
		return dest

	def _lower_compiler_c_field_set( self, node: ast.Call, expected_type: Type|None ) -> None:
		# compiler.c_field_set(ptr, 'field_name', value) - writes a field of
		# an opaque compiler.c_type(...) struct through ptr. Same trust
		# model as compiler.c_field's own read side (see its comment) -
		# value's own type is whatever it already lowers to, unchecked
		# against the real field's type beyond what the C compiler itself
		# catches.
		if len( node.args ) != 3 or node.keywords:
			self.lowering.discovery.fail( f'compiler.c_field_set(ptr, field_name, value) takes exactly 3 positional arguments: {ast.unparse(node)}', node )
		ptr_node, name_node, value_node = node.args
		ptr = self._lower_expr( ptr_node, None )
		field_name = self._c_field_name_literal( name_node, 'c_field_set', node )
		self._require_c_type_pointer( ptr.type, 'c_field_set', node )
		value = self._lower_expr( value_node, None )
		self._emit( ir.SetAttr( obj = ptr, attr = field_name, value = value ))
		return None

	def _lower_compiler_c_field_addr( self, node: ast.Call, expected_type: Type|None ) -> ir.Operand:
		# compiler.c_field_addr(ptr, 'field_name', Ptr[T]) -> Ptr[T] -
		# address of a NESTED value-typed field of an opaque c_type struct
		# (e.g. ucontext_t's uc_stack, itself a stack_t VALUE, not a
		# pointer) - lets a second-level opaque type (stack_t, its own
		# compiler.c_type(...)) be reached by composing two single-level
		# accesses (this, then compiler.c_field/c_field_set on the result)
		# rather than needing a dotted field path - see
		# _c_field_name_literal's own comment for why dotted paths aren't
		# supported directly. Third argument is the FULL result type
		# (Ptr[T], not just T) - unlike compiler.addrof, which computes the
		# pointer level itself from a known operand type, everything here is
		# already caller-asserted, so there's no "known type" to compute a
		# level from.
		if len( node.args ) != 3 or node.keywords:
			self.lowering.discovery.fail( f'compiler.c_field_addr(ptr, field_name, type) takes exactly 3 positional arguments: {ast.unparse(node)}', node )
		ptr_node, name_node, type_node = node.args
		ptr = self._lower_expr( ptr_node, None )
		field_name = self._c_field_name_literal( name_node, 'c_field_addr', node )
		self._require_c_type_pointer( ptr.type, 'c_field_addr', node )
		result_type = self.lowering._try_resolve_namespace( type_node )
		if not isinstance( result_type, Type ):
			self.lowering.discovery.fail( f'compiler.c_field_addr(...) third argument must be a type: {ast.unparse(node)}', node )
		dest = self._new_temp( result_type )
		self._emit( ir.AddrOfField( dest = dest, obj = ptr, attr = field_name ))
		return dest

	def _lower_compiler_atomic_load( self, node: ast.Call, expected_type: Type|None ) -> ir.Operand:
		if len( node.args ) != 1 or node.keywords:
			self.lowering.discovery.fail( f'compiler.atomic_load(...) takes exactly one argument: {ast.unparse(node)}', node )
		ptr = self._lower_expr( node.args[0], None )
		pointee = self.lowering._atomic_pointee_type( ptr.type, node )
		dest = self._new_temp( expected_type or pointee )
		self._emit( ir.AtomicLoad( dest = dest, ptr = ptr ))
		return dest

	def _lower_compiler_format_f64( self, node: ast.Call, expected_type: Type|None ) -> ir.Operand:
		# compiler.format_f64(buf, size, precision, type_char, alt, value) ->
		# i32 - writes value's fixed-precision decimal digits (magnitude
		# only, no sign - lib/builtins/__float.py's own callers split the
		# sign out first, the same split int's __str__/_to_radix_digits/
		# _decimal_digits_with_grouping already keep) into buf[0:size),
		# returns the byte count written. type_char is a printf-style
		# conversion character's ASCII code ('f'/'F'/'e'/'E'/'g'/'G' - see
		# fstring_format_spec.FORMAT_SPEC_TYPE_CHARS; '%' is handled entirely
		# in metalpy source instead, by scaling the value and formatting as
		# 'f' - see lib/builtins/__float.py's _percent_digits). alt is the
		# '#' flag (always show the decimal point for 'f'/'F'/'e'/'E', keep
		# trailing zeros for 'g'/'G' - real snprintf's own '#' flag already
		# matches Python's semantics for every one of these exactly, so it's
		# passed straight through rather than needing its own post-
		# processing the way grouping does). Backed by a hand-written C
		# helper in emitter_c.py's PROLOGUE (real snprintf/msvcrt _snprintf,
		# called there with its true variadic prototype) - deliberately NOT
		# an ordinary @extern binding: emitter_c.py's extern codegen only
		# ever emits fixed-arity C prototypes, which is an ABI hazard for a
		# genuinely variadic callee, and tagging this under the 'c' extern
		# lib would flip compiler.extern_libs and break the no-crt Windows
		# build (float_test.py's own no_crt = 'c' not in compiler.
		# extern_libs).
		if len( node.args ) != 6 or node.keywords:
			self.lowering.discovery.fail( f'compiler.format_f64(...) takes exactly 6 arguments (buf, size, precision, type_char, alt, value): {ast.unparse(node)}', node )
		intrinsics = self.lowering.discovery.get_intrinsics()
		ptr_cls = intrinsics['Ptr']
		buf_type = self.lowering.discovery._get_or_create_specialization( ptr_cls, [ intrinsics['u8'] ] )
		buf = self._lower_expr( node.args[0], buf_type )
		size = self._lower_expr( node.args[1], intrinsics['usize'] )
		precision = self._lower_expr( node.args[2], intrinsics['i32'] )
		type_char = self._lower_expr( node.args[3], intrinsics['i32'] )
		alt = self._lower_expr( node.args[4], intrinsics['bool'] )
		value = self._lower_expr( node.args[5], intrinsics['f64'] )
		dest = self._new_temp( expected_type or intrinsics['i32'] )
		self._emit( ir.FormatFloat( dest = dest, buf = buf, size = size, precision = precision, type_char = type_char, alt = alt, value = value ))
		return dest

	def _lower_compiler_is_nan_or_inf( self, node: ast.Call, expected_type: Type|None, name: str ) -> ir.Operand:
		# compiler.is_nan(x)/compiler.is_inf(x) - x: f32|f64 -> bool. Reuses
		# __metalpy_isnan/__metalpy_isinf (emitter_c.py's PROLOGUE, already
		# there for checked/panic-mode float arithmetic) - exposed directly
		# so f-string format specs can special-case inf/nan display
		# (lib/builtins/__float.py), since real snprintf/msvcrt don't
		# reliably produce "inf"/"nan" text for these themselves (confirmed:
		# legacy msvcrt's own _snprintf gives outright garbage like "1.$"
		# for +infinity, not "inf" - unlike the exponent-padding/missing-'F'
		# quirks found earlier, this one isn't even close to right).
		if len( node.args ) != 1 or node.keywords:
			self.lowering.discovery.fail( f'compiler.{name}(...) takes exactly one argument: {ast.unparse(node)}', node )
		intrinsics = self.lowering.discovery.get_intrinsics()
		value = self._lower_expr( node.args[0], None )
		if value.type is not intrinsics.get( 'f32' ) and value.type is not intrinsics.get( 'f64' ):
			type_name = value.type.qualname if value.type is not None else '?'
			self.lowering.discovery.fail( f'compiler.{name}(...) argument must be f32 or f64, not {type_name}: {ast.unparse(node)}', node )
		dest = self._new_temp( expected_type or intrinsics['bool'] )
		ir_cls = ir.IsNan if name == 'is_nan' else ir.IsInf
		self._emit( ir_cls( dest = dest, value = value ))
		return dest

	def _lower_compiler_parse_f64( self, node: ast.Call, expected_type: Type|None ) -> ir.Operand:
		# compiler.parse_f64(buf) - buf: ConstPtr[u8] (null-terminated C
		# text) -> f64. The inverse of compiler.format_f64 - needed for the
		# shortest-round-trip repr search (lib/builtins/__float.py's
		# _f64_repr_digits_raw: try increasing precision, re-parse each
		# candidate, stop at the first exact round-trip). Backed by a
		# hand-written C helper in emitter_c.py's PROLOGUE (real strtod on
		# POSIX; msvcrt.dll's own strtod, resolved dynamically via
		# GetModuleHandleA/LoadLibraryA/GetProcAddress, on Windows -
		# verified correct against this system's own msvcrt.dll, unlike
		# some of its other legacy quirks found earlier) - deliberately
		# NOT an ordinary @extern binding even though strtod's own
		# signature is perfectly ordinary (non-variadic, no ABI hazard
		# like compiler.format_f64 has): tagging it under the 'c' extern
		# lib would still wrongly flip compiler.extern_libs and break the
		# no-crt Windows build, the same reason compiler.format_f64 itself
		# isn't a plain @extern binding either.
		if len( node.args ) != 1 or node.keywords:
			self.lowering.discovery.fail( f'compiler.parse_f64(...) takes exactly one argument: {ast.unparse(node)}', node )
		intrinsics = self.lowering.discovery.get_intrinsics()
		ptr_cls = intrinsics['ConstPtr']
		buf_type = self.lowering.discovery._get_or_create_specialization( ptr_cls, [ intrinsics['u8'] ] )
		buf = self._lower_expr( node.args[0], buf_type )
		dest = self._new_temp( expected_type or intrinsics['f64'] )
		self._emit( ir.ParseFloat( dest = dest, buf = buf ))
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
		# `source` is EITHER an unlowered ast.expr (a bare literal) OR an
		# already-lowered ir.Operand (a real runtime value). For a pure
		# int<->int conversion (float involved is handled separately below,
		# unchanged), the rule is purely a bit-width question, identical for
		# a literal and a runtime value alike (see the width comparison
		# below and its own comment): same-width or widening ALWAYS
		# succeeds, unconditionally, in every arithmetic mode (a pure bit-
		# reinterpretation - -1 reinterpreted as u32 is exactly the well-
		# defined two's-complement value real WinAPI constants like
		# STD_OUTPUT_HANDLE rely on); only a genuinely NARROWING conversion
		# can ever fail, and that stays mode-aware (respects self.
		# _arithmetic_mode exactly like +/-/* already do, reusing the same
		# Check/Wrap/Saturate/panic_arithmetic machinery) - see SYNTAX.md's
		# own T(x) section for the full rationale, including why this is
		# deliberately a DIFFERENT, narrower rule than x.to_T() (a value-
		# range check, independent of width - see compiler.checked_convert).
		if isinstance( source, ast.expr ):
			# an EXPLICIT float-literal cast to a non-float scalar (i32(1.5),
			# compiler.cast(u8, 3.9)) truncates toward zero at compile time -
			# this is the deliberate float->int conversion mechanism, so it
			# bypasses _expr_Constant's implicit-hint guard (which only rejects a
			# float literal being SILENTLY coerced to an int, e.g. `i + 1.5`).
			# A float->float or int/other literal keeps its existing behavior
			# (float value preserved, or int bit-reinterpretation via _lower_expr)
			if isinstance( source, ast.Constant ) and isinstance( source.value, float ) and not _is_float_scalar( target_type ):
				return ir.Const( type = target_type, value = int( source.value ) )
			# a bare int literal's own natural type is i32 (_expr_Constant's
			# own "a literal's OWN Python type always determines its natural
			# type" rule, used whenever no expected_type applies) - applying
			# the SAME same-width/widening-vs-narrowing rule the runtime
			# branch below uses, relative to THAT natural i32 width, is what
			# makes a literal and an equivalent runtime i32 argument behave
			# IDENTICALLY through T(x) - closing the exact inconsistency
			# this whole mechanism exists for (previously: u32(-1) always
			# succeeded via a blanket exemption, but u32(some_i32_var
			# holding -1) went through the real, then still mode-aware,
			# checked-cast path - now both are simply never fallible in the
			# first place, since i32->u32 is same-width). Only a genuinely
			# NARROWING literal (target narrower than i32) still needs
			# range-checking - e.g. u8(-10000) is still a hard, mode-
			# independent compile-time error, matching how `x: u8 = -10000`
			# already behaves everywhere else literals flow into a
			# concrete integer type.
			allow_bit_reinterpret = True
			if isinstance( source, ast.Constant ) and type( source.value ) is int and not _is_float_scalar( target_type ):
				i32 = self.lowering.discovery.get_intrinsics()['i32']
				allow_bit_reinterpret = target_type.sizeof >= i32.sizeof
			prev_allow_bit_reinterpret = self._allow_literal_bit_reinterpret
			self._allow_literal_bit_reinterpret = allow_bit_reinterpret
			try:
				return self._lower_expr( source, target_type )
			finally:
				self._allow_literal_bit_reinterpret = prev_allow_bit_reinterpret
		operand = source
		# a cast that touches a float on either side (int<->float, float<->float)
		# takes the GetFloatCast path: floats have no integer-overflow concept,
		# so checked/panic mode checks the result/source for inf/nan/range and
		# wrap/saturate does a plain-or-clamping conversion - never the unsigned-
		# roundtrip integer cast machinery. A pure int<->int cast is unchanged
		target_is_float = _is_float_scalar( target_type )
		source_is_float = _is_float_scalar( operand.type )
		if target_is_float or source_is_float:
			opcode, extra = self._arithmetic_mode[-1].GetFloatCast( target_is_float = target_is_float, source_is_float = source_is_float )
			return self._lower_arithmetic_op( node, opcode, extra, target_type, { 'operand': operand }, 'cast' )
		if target_type.sizeof >= operand.type.sizeof:
			# same-width or widening int<->int - always succeeds, in every
			# mode, no Result involved: matches CastWrap's own bare-C-cast
			# emitter code (a plain `(ctype)(operand)`), just no longer
			# gated behind wrap-mode specifically. A real, deliberate
			# behavior change from this conversion's own prior semantics
			# (confirmed via a real repro: u32(some_i32_holding_a_negative_
			# value) previously REQUIRED an enclosing Result under checked/
			# panic mode, only wrap mode was unconditional) - see this
			# function's own top-of-method comment and SYNTAX.md.
			return self._lower_arithmetic_op( node, ir.CastWrap, None, target_type, { 'operand': operand }, 'cast' )
		opcode, extra = self._arithmetic_mode[-1].GetCast()
		return self._lower_arithmetic_op( node, opcode, extra, target_type, { 'operand': operand }, 'cast' )

	def _lower_compiler_raw_alloc( self, node: ast.Call, expected_type: Type|None ) -> ir.Operand:
		# compiler.__raw_alloc__(T) - allocate a fresh RCClass instance with
		# every field left UNINITIALIZED: no field validation, no __init__
		# call. This is the exact alloc _try_lower_construct_call used to
		# emit inline (dest = new temp, schedule sys.alloc[T], ir.Allocate
		# with an empty fields dict) - factored out here so a synthesized
		# $$__new__ body (type_resolver.py's
		# _synthesize_rcclass_constructor) can spell "allocate self, THEN
		# call __init__ myself" as ordinary AST rather than raw IR. Not
		# meant for ordinary user code (there's no field-completeness check
		# at all - the caller is on the hook for calling __init__ or
		# compiler.__raw_free__'ing it before it ever escapes), same
		# internal-only posture as compiler.decref_dynamic.
		#
		# node.args[0].resolved_type, like compiler.cast's first argument,
		# lets compiler-synthesized AST hand over a concrete RCClass object
		# directly (including a monomorphized generic with no user-
		# spellable name), bypassing ordinary namespace resolution.
		if len( node.args ) != 1 or node.keywords:
			self.lowering.discovery.fail( f'compiler.__raw_alloc__(...) takes exactly one argument: {ast.unparse(node)}', node )
		target_type = getattr( node.args[0], 'resolved_type', None )
		if target_type is None:
			target_type = self.lowering._try_resolve_namespace( node.args[0] )
		if not isinstance( target_type, RCClass ):
			self.lowering.discovery.fail( f'compiler.__raw_alloc__(...) argument must be a concrete RCClass: {ast.unparse(node)}', node )
		dest = self._new_temp( target_type )
		self.lowering._schedule_rcclass_construction( target_type, dest.type )
		self._emit( ir.Allocate( dest = dest, cls = target_type, fields = {} ))
		return dest

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
		# compiler.decref(x) — emit the real Decref sequence for x, via
		# cfg.py's own union-aware decref() (NOT a bare ir.Decref emitted
		# directly here - that's only correct for a plain RC pointer; a
		# TaggedUnion operand with RC leaves needs the tag-gated release
		# ladder instead, exactly like every other decref site in this
		# compiler - see cfg.py's _refcount_instructions). Used inside
		# synthesized destructor bodies to tear down each RC field, and by
		# generic containers (list[T]) that need to conditionally RC-manage
		# elements whose T may or may not turn out to be an RC type once
		# monomorphized - a silent no-op for a non-RC T (rather than a hard
		# failure) is allowed ONLY inside a monomorphized generic-class
		# method (_in_generic_class_method), so the SAME generic method
		# body stays correct for both list[SomeRCClass] and list[i32]
		# without the class itself branching on whether T is RC - an
		# ordinary, non-generic call site with a genuinely wrong (always
		# non-RC) argument is still rejected, same as before.
		#
		# Gated on cfg.rc_leaves(operand.type), not the narrower
		# type_resolver._is_RC (is_rc_pointer) - _is_RC is False for a
		# TaggedUnion with RC members (its runtime shape is a tag+data value
		# struct, never a bare pointer), which used to make this whole
		# branch treat "T monomorphized to a union with RC leaves" exactly
		# like "T monomorphized to a genuinely non-RC scalar" - a silent
		# no-op inside _in_generic_class_method(), the SAME no-op posture
		# that's actually correct for list[i32]. Confirmed via a real repro
		# (list[T].append/__getitem__ with T a @union whose RC-carrying leaf
		# is a plain RCClass like the builtin int): the missing incref/decref
		# left every such element under-retained by exactly one reference,
		# a real heap-corruption-on-free bug - masked whenever the leaf
		# happened to be an IMMORTAL-refcount value (a string literal),
		# which is why this surfaced as "str leaves work, int leaves crash"
		# rather than an unconditional failure.
		if len( node.args ) != 1 or node.keywords:
			self.lowering.discovery.fail( f'compiler.decref(...) takes exactly one argument: {ast.unparse(node)}', node )
		operand = self._lower_expr( node.args[0], None )
		if operand.type is not None and cfg.rc_leaves( operand.type ):
			for instr in self._cfg.decref( operand.type, operand ):
				self._emit( instr )
			# stop the scope-exit epilogue from decref'ing operand a SECOND
			# time - see cfg.py's manually_decreffed's own comment for why
			# this is required, not optional (a real, always-on double
			# Decref/use-after-free otherwise, confirmed with ASan). Usually
			# returns nothing more to emit - only non-empty when operand's
			# own entry was already captured by an earlier return, in which
			# case this is the flag-disarm that keeps that earlier return's
			# own shared-ladder decref from silently going missing (see
			# manually_decreffed's own docstring)
			for instr in self._cfg.manually_decreffed( operand ):
				self._emit( instr )
			return
		if operand.type is not None and self._in_generic_class_method():
			# a genuine no-op for THIS monomorphization (see the comment
			# above), but operand may have no other reader at all in that
			# case (e.g. list[i32].__del__'s `val`, only ever passed to
			# decref) - mark it read so the no-op doesn't turn its own
			# already-emitted definition into -Wunused-variable/C4189
			self._emit( ir.MarkUsed( operand = operand ))
			return
		self.lowering.discovery.fail(
			f'compiler.decref(...) argument must be a reference-counted value, not '
			f'{operand.type.qualname if operand.type else "?"}: {ast.unparse(node)}',
			node,
		)

	def _lower_compiler_raw_free( self, node: ast.Call ) -> None:
		# compiler.__raw_free__(x) - free a raw, not-yet-fully-alive RCClass
		# allocation (compiler.__raw_alloc__'s own product, once __init__
		# has failed) WITHOUT running the class's real destructor. An
		# ordinary decref-to-zero would call the synthesized
		# $$__destructor__, which reads every field as though __init__ had
		# already populated them - on a raw, not-yet-initialized alloc
		# that's still garbage, a real heap-corruption bug (see
		# type_resolver.py's _synthesize_rcclass_constructor, fallible
		# body, and _emit_fallible_construction's own former Err branch,
		# whose logic this generalizes). Frees the backing storage directly
		# via sys.free - the same thing _synthesize_rcclass_destructor's
		# own step 3 does, skipping its __del__/field-cascade steps 1/2
		# entirely - then cancels x's pending automatic scope-exit release
		# via manually_decreffed, the same pairing compiler.decref(x) uses
		# just above, minus the real Decref that precedes it there.
		if len( node.args ) != 1 or node.keywords:
			self.lowering.discovery.fail( f'compiler.__raw_free__(...) takes exactly one argument: {ast.unparse(node)}', node )
		operand = self._lower_expr( node.args[0], None )
		if not isinstance( operand.type, RCClass ):
			self.lowering.discovery.fail(
				f'compiler.__raw_free__(...) argument must be a bare RCClass value, not '
				f'{operand.type.qualname if operand.type else "?"}: {ast.unparse(node)}',
				node,
			)
		sys_module = self.lowering.discovery.modules['sys']
		free_overload = sys_module.get_local( 'free' )
		free_fn = free_overload.implementations[0] if isinstance( free_overload, Overload ) else free_overload
		self.lowering._ensure_resolved( free_fn )
		cast_dest = self._new_temp( free_fn.parameters[0].type )
		self._emit( ir.CastWrap( dest = cast_dest, operand = operand ))
		self._emit( ir.Call( dest = None, target = free_fn, receiver = None, args = [ cast_dest ], kwargs = {} ))
		for instr in self._cfg.manually_decreffed( operand ):
			self._emit( instr )

	def _lower_compiler_incref( self, node: ast.Call ) -> None:
		# compiler.incref(x) — emit the real Incref sequence for x, via
		# cfg.py's own union-aware incref(). Same conditional no-op-for-
		# non-RC-T posture, and the same rc_leaves(...)-vs-_is_RC fix, as
		# _lower_compiler_decref above.
		if len( node.args ) != 1 or node.keywords:
			self.lowering.discovery.fail( f'compiler.incref(...) takes exactly one argument: {ast.unparse(node)}', node )
		operand = self._lower_expr( node.args[0], None )
		if operand.type is not None and cfg.rc_leaves( operand.type ):
			for instr in self._cfg.incref( operand.type, operand ):
				self._emit( instr )
			return
		if operand.type is not None and self._in_generic_class_method():
			# see _lower_compiler_decref's own identical comment
			self._emit( ir.MarkUsed( operand = operand ))
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
		# continue_captured unused here - start_label (this loop's own
		# continue target) is always jumped to by the back edge below
		# regardless of whether the body itself ever uses `continue`
		break_narrowed, break_live, _ = self._lower_loop_body( node.body, continue_label = start_label, break_label = end_label, loop_snapshot = loop_snapshot )
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
		natural_exit_live: set[str] | None = None
		if not is_while_true:
			natural_exit_narrowed = dict( loop_snapshot.narrowed )
			exit_name = getattr( node, 'exit_narrows_name', None )
			if exit_name is not None:
				member = self._resolve_narrow_member( exit_name, node.exit_narrows_member_stem, node )
				natural_exit_narrowed[exit_name] = [ member ]
			natural_exit_live = set( loop_snapshot.live )
		self._cfg.merge_loop_exits( natural_exit_narrowed, break_narrowed, natural_exit_live, break_live )
		self._emit( ir.Jump( target = start_label ))
		self._emit( ir.Label( name = end_label ))

	def _lower_loop_body( self, body: list[ast.stmt], continue_label: str, break_label: str, loop_snapshot: object ) -> tuple[list[dict[str,list[Variable]]],list[set[str]],bool]:
		self._loop_depth += 1
		ctx = _LoopContext( continue_label = continue_label, break_label = break_label, loop_snapshot = loop_snapshot )
		self._loop_labels.append( ctx )
		# see cfg.py's CFGState.enter_loop's own docstring: lets
		# current_epilogue_label() recognize an RC entry pushed while
		# lowering THIS body as loop-confined (restore(), called once this
		# body's fully lowered, silently drops it - its own label, if a
		# `return` from inside here ever pointed at it, would never
		# actually get emitted)
		self._cfg.enter_loop( loop_snapshot.stack_depth )
		break_narrowed: list[dict[str,list[Variable]]] = []
		break_live: list[set[str]] = []
		try:
			for stmt in body:
				try:
					self._lower_stmt( stmt )
				except CompileError:
					continue
		finally:
			# Phase 8: every narrowed-state/live-state snapshot recorded by a
			# `break` reached while lowering this body (cfg.py's own
			# record_break_narrowed()/record_break_live(), called from
			# _stmt_Break below) - handed back to the caller (_stmt_While/
			# for-loop lowerers) to merge with the loop's own natural exit
			# via merge_loop_exits()
			break_narrowed, break_live = self._cfg.exit_loop()
			self._loop_labels.pop()
			self._loop_depth -= 1
		return break_narrowed, break_live, ctx.continue_captured

	def _check_loop_exit_unchecked_results( self, loop_snapshot: object, node: ast.AST ) -> None:
		try:
			self._cfg.check_loop_exit_unchecked_results( loop_snapshot.results, self._current_fn.qualname )
		except CompileError as e:
			self.lowering.discovery.fail( str( e ), node )

	def _stmt_Break( self, node: ast.Break ) -> None:
		if not self._loop_labels:
			self.lowering.discovery.fail( 'break outside a loop', node )
		ctx = self._loop_labels[-1]
		break_label, loop_snapshot = ctx.break_label, ctx.loop_snapshot
		self._check_loop_exit_unchecked_results( loop_snapshot, node )
		# Phase 8: capture whatever's narrowed RIGHT HERE, at the exact
		# point this break fires - cfg.py's record_break_narrowed() files
		# it under the innermost currently-lowering loop, to be merged
		# with every other break/the loop's own natural exit once that
		# loop's own body is fully lowered. Before unwind_to() below (which
		# doesn't touch _narrowed at all, but ordering it first here keeps
		# this call sitting right next to unwind_to()'s own snapshot read).
		# record_break_live() is the definite-assignment analogue, captured
		# alongside it for the identical reason.
		self._cfg.record_break_narrowed()
		self._cfg.record_break_live()
		for instr in self._cfg.unwind_to( loop_snapshot ):
			self._emit( instr )
		self._emit( ir.Jump( target = break_label ))

	def _stmt_Continue( self, node: ast.Continue ) -> None:
		if not self._loop_labels:
			self.lowering.discovery.fail( 'continue outside a loop', node )
		ctx = self._loop_labels[-1]
		continue_label, loop_snapshot = ctx.continue_label, ctx.loop_snapshot
		ctx.continue_captured = True
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
		# every caller unconditionally assigns this right after declaring it
		# (no user code runs in between - see each call site's own next
		# line/statement), so it's live from here on, same bucket as a
		# parameter - see cfg.py's mark_live() docstring
		self._cfg.mark_live( stem )
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
		existing = self._existing_local_or_none( target.id, node, 'cannot use it as a for loop target' )
		if existing is not None:
			operand = self._lower_expr( value_expr, existing.type )
			self._emit( ir.Assign( dest = existing, src = operand ))
			self._cfg.mark_live( existing.stem ) # this bypasses _cfg_assign like the rest of this function does (pre-existing, not touched here) - liveness alone still needs marking, since it's unconditionally assigned right here regardless
			return existing
		var, operand = self._declare_local( target.id, node, lambda expected: self._lower_expr( value_expr, expected ), default_type = default_type )
		self._emit( ir.Assign( dest = var, src = operand ))
		self._cfg.mark_live( var.stem )
		return var

	def _stmt_For( self, node: ast.For ) -> None:
		if not isinstance( node.target, ast.Name ):
			self.lowering.discovery.fail( f'for loop target must be a plain name: {ast.unparse(node)}', node )
		if node.orelse:
			self.lowering.discovery.fail( 'for/else is not supported', node )
		if self.lowering._is_range_call( node.iter ) is not None:
			self._lower_for_range( node )
			return
		# PLAN_GENERATORS.md Phase 3 - node.iter is lowered exactly ONCE here
		# (not once per candidate path) and the resulting operand handed to
		# whichever real consumption path applies, so an iterable expression
		# with a side effect (most commonly: a generator CONSTRUCTOR call)
		# is never evaluated twice - _lower_for_over_indexable used to lower
		# node.iter itself; it now takes the already-lowered operand instead,
		# the same way _lower_for_over_iterator does
		obj = self._lower_expr( node.iter, None )
		next_fn = self.lowering._find_iterator_next_method( obj.type )
		if next_fn is not None:
			self._lower_for_over_iterator( node, obj, next_fn )
		else:
			self._lower_for_over_indexable( node, obj )

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
		break_narrowed, break_live, continue_captured = self._lower_loop_body( node.body, continue_label = continue_label, break_label = end_label, loop_snapshot = loop_snapshot )
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
		self._cfg.merge_loop_exits( dict( loop_snapshot.narrowed ), break_narrowed, set( loop_snapshot.live ), break_live )

		# continue_label is purely a fallthrough landing (the increment
		# below) unless some `continue` in the body actually jumped to it -
		# an un-goto'd label triggers -Wunused-label/C4102
		if continue_captured:
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

	def _lower_for_over_indexable( self, node: ast.For, obj: ir.Operand ) -> None:
		usize_cls = self.lowering.discovery.get_intrinsics()['usize']
		bool_cls = self.lowering.discovery.find_name( 'bool', node )

		len_fn = self.lowering._find_method( obj.type, '__len__' )
		getitem_fn = self.lowering._find_method( obj.type, '__getitem__' )
		missing = [ name for name, fn in (( '__len__', len_fn ), ( '__getitem__', getitem_fn )) if fn is None ]
		if missing:
			self.lowering.discovery.fail(
				f'for loop needs {" and ".join(missing)} (or a __next__() returning T|None) on '
				f'{obj.type.qualname if obj.type else "?"}: {ast.unparse(node)}',
				node,
			)

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

		break_narrowed, break_live, continue_captured = self._lower_loop_body( node.body, continue_label = continue_label, break_label = end_label, loop_snapshot = loop_snapshot )
		try:
			back_edge_instructions = self._cfg.loop_back_edge( loop_snapshot.bindings, self._current_fn.qualname, entry_results = loop_snapshot.results )
		except CompileError as e:
			self.lowering.discovery.fail( str( e ), node )
		for instr in back_edge_instructions:
			self._emit( instr )
		self._cfg.restore( loop_snapshot )
		# Phase 8 - see _lower_for_range's own identical call/comment
		self._cfg.merge_loop_exits( dict( loop_snapshot.narrowed ), break_narrowed, set( loop_snapshot.live ), break_live )

		# see _lower_for_range's own identical comment on continue_captured
		if continue_captured:
			self._emit( ir.Label( name = continue_label ))
		incr = self._new_temp( usize_cls )
		self._emit( ir.AddWrap( dest = incr, left = index_var, right = ir.Const( type = usize_cls, value = 1 ) ))
		self._emit( ir.Assign( dest = index_var, src = incr ))
		self._emit( ir.Jump( target = start_label ))
		self._emit( ir.Label( name = end_label ))

	def _lower_binary_branch( self, cond: ir.Operand, node: ast.AST, true_thunk: 'Callable[[],bool]', false_thunk: 'Callable[[],bool]' ) -> None:
		''' hand-rolled version of _stmt_If's own branch-then-merge
		machinery (JumpIfFalse/snapshot/enter_branch/exit_branch/merge_if/
		splice), for a caller building its own condition and branch bodies
		directly at the IR level rather than lowering a real ast.If -
		_lower_for_over_iterator's own Result[T,E]/Err(StopIteration)-vs-
		real-error dispatch, which has no ast.If to lower (the condition is
		a synthesized tag comparison, and each branch's own body is a
		narrow()+bind pair, not user-written statements).

		Each thunk is called with no arguments inside its own branch-
		confined region (self._cfg.enter_branch/exit_branch, self.
		_instructions redirected to a captured list - identical setup
		_stmt_If uses for node.body/node.orelse), expected to emit
		whatever IR it needs via self._emit/self._cfg directly, and return
		True if it's a dead end that never reaches the merge point (a raw
		jump elsewhere - the same "terminates" concept _stmt_If's own
		true_terminates/false_terminates track for a branch ending in
		return/break/continue, just decided by the thunk itself instead of
		_stmt_diverges since there's no real AST statement to inspect). '''
		else_label = self._new_label( 'branch_else' )
		self._emit( ir.JumpIfFalse( cond = cond, target = else_label ))

		entry_snapshot = self._cfg.snapshot()
		outer_instructions = self._instructions
		self._instructions = []
		self._cfg.enter_branch( entry_snapshot.stack_depth )
		true_temps_start = len( self._pending_temps )
		try:
			true_terminates = true_thunk()
		finally:
			self._cfg.exit_branch()
		# each thunk's own PURELY INTERMEDIATE temps (e.g. leaf_bind_thunk's
		# E'-union coercion wrapper, built to pass a narrowed leaf value into
		# Result.Err(e: E')) must be flushed HERE, still inside this branch's
		# own captured instruction list - _flush_branch_temps' own comment
		# documents the exact same crash class this recreates otherwise:
		# left pending, they'd survive into the ENCLOSING statement's single
		# unconditional end-of-statement flush (this method builds raw IR,
		# never routes a thunk's own statements through _lower_stmt, so nothing
		# else ever flushes them) and get decref'd there even for whichever
		# branch never ran at runtime - reading tag/payload data off an
		# uninitialized C local (confirmed via a real crash: multi_leaf-style
		# two-leaf error dispatch, MSVC access violation)
		self._flush_branch_temps( true_temps_start )
		true_captured = self._instructions
		true_end = dict( self._cfg.bindings )
		true_end_results = self._cfg.unchecked_results()
		true_end_narrowed = self._cfg.narrowed_snapshot()
		true_end_live = self._cfg.live_snapshot()

		self._cfg.restore( entry_snapshot )
		self._instructions = []
		self._cfg.enter_branch( entry_snapshot.stack_depth )
		false_temps_start = len( self._pending_temps )
		try:
			false_terminates = false_thunk()
		finally:
			self._cfg.exit_branch()
		self._flush_branch_temps( false_temps_start )
		false_captured = self._instructions
		false_end = dict( self._cfg.bindings )
		false_end_results = self._cfg.unchecked_results()
		false_end_narrowed = self._cfg.narrowed_snapshot()
		false_end_live = self._cfg.live_snapshot()

		self._cfg.restore( entry_snapshot )
		self._instructions = outer_instructions
		try:
			true_extra, false_extra, removed = self._cfg.merge_if(
				entry_snapshot.bindings, true_end, false_end, self._current_fn.qualname,
				entry_results = entry_snapshot.results, true_end_results = true_end_results, false_end_results = false_end_results,
				true_terminates = true_terminates, false_terminates = false_terminates,
				true_end_narrowed = true_end_narrowed, false_end_narrowed = false_end_narrowed,
				true_end_live = true_end_live, false_end_live = false_end_live,
			)
		except CompileError as e:
			self.lowering.discovery.fail( str( e ), node )

		for instr in true_captured:
			self._emit_captured( instr )
		for instr in true_extra:
			self._emit( instr )
		end_label = self._new_label( 'branch_end' )
		self._emit( ir.Jump( target = end_label ))
		self._emit( ir.Label( name = else_label ))
		for instr in false_captured:
			self._emit_captured( instr )
		for instr in false_extra:
			self._emit( instr )
		self._emit( ir.Label( name = end_label ))

	def _lower_for_over_iterator( self, node: ast.For, obj: ir.Operand, next_fn: Function ) -> None:
		''' PLAN_GENERATORS.md's StopIteration reversal - `for x in <expr
		with a __next__() returning Result[T,E]>:` (E always including
		StopIteration). Structurally the same shape as _lower_for_over_
		indexable (once-evaluated iterable, start/continue/end labels,
		snapshot-bind-lower_loop_body-back_edge-restore-merge), except the
		"is there another element" test is __next__()'s own Result[T,E]
		tag rather than an index/length comparison, and the element
		binding needs the union's payload extracted rather than a plain
		__getitem__ call.

		Binding rule (matches type_resolver.py's own _desugar_iterator_for,
		the in-generator-body mirror of this method - see its own docstring
		for the full reasoning, confirmed directly with the user): if E is
		JUST StopIteration, x binds to plain T. If E has any OTHER error,
		x binds to Result[T,E'] (E' = E minus StopIteration) - the caller
		handles the real error explicitly inside the loop body (match/
		.is_err()/.or_return()/.unwrap()), no auto-propagation. Unlike the
		in-generator-body desugaring (which builds fresh AST processed by a
		LATER type-checking/desugaring pass with its own narrowing
		machinery), this method lowers directly to IR - constructing a
		NARROWER Result[T,E'] value from a payload read out of the WIDER
		Result[T,E] needs a genuine N-way tag dispatch per E' leaf (cfg.py's
		narrow() only yields a concrete single type when exactly one
		candidate remains - Result.Err(e)'s own generic T/E inference
		disagrees between the assignment target's declared E' and e's own
		un-narrowed E otherwise, confirmed via a real repro identical to
		type_resolver.py's own _desugar_iterator_for hitting the same
		trap), built here via _lower_binary_branch - a hand-rolled
		_stmt_If-style branch-then-merge helper for exactly this situation
		(a condition + branch bodies with no real ast.If to lower).

		The payload extraction reuses cfg.py's REAL narrowing mechanism
		(narrow()/narrowed_member(), the same machinery a `match x: case
		T(x):` arm - reusing the subject's own name - already relies on;
		confirmed working via a standalone repro, since a plain `if x is
		None: ... else: ...`-narrowed read does NOT currently work anywhere
		in this compiler, generator-unrelated - see PLAN_GENERATORS.md's own
		STATUS section) directly at the IR level: __next__()'s result is
		bound into a hidden local, narrow()'d to the union's own Ok leaf,
		then read back through node.target's own ordinary _stmt_Assign -
		_expr_Name's existing narrowed-read branch does the rest (extracts
		through GetAttr(data)/GetAttr(v_<leaf>) automatically), no new
		extraction code needed here at all.

		The "was this an error" test can't be spelled `x.is_err()` in the
		synthesized AST the way user source would (that's a real method
		call, more machinery than needed) - the same direct .tag comparison
		this method has always built by hand (see its own historical
		docstring on why is/is-not-None doesn't work here either - the same
		reasoning applies to any comparison against a TaggedUnion tag). '''
		self.lowering._ensure_resolved( next_fn )
		result_type = next_fn.return_type
		shape = self.lowering._type_resolver._result_shape( result_type )
		stop_iteration_cls = self.lowering.discovery.find_name_or_none( 'StopIteration' )
		if shape is None or stop_iteration_cls is None or stop_iteration_cls not in self.lowering._type_resolver._atomic_leaves( shape[1] ):
			self.lowering.discovery.fail(
				f'for loop needs __next__() to return Result[T,E] (E including StopIteration) on '
				f'{obj.type.qualname if obj.type else "?"}: {ast.unparse(node)}',
				node,
			)
		elem_type, full_error_type = shape
		remaining_leaves = [ leaf for leaf in self.lowering._type_resolver._atomic_leaves( full_error_type ) if leaf is not stop_iteration_cls ]
		remaining_error_type: Type|None
		if not remaining_leaves:
			remaining_error_type = None
		elif len( remaining_leaves ) == 1:
			remaining_error_type = remaining_leaves[0]
		else:
			remaining_error_type = self.lowering.discovery._get_or_create_union( remaining_leaves )
		self.lowering.schedule( result_type )
		# result_type is routinely a Result[T,E] SPECIALIZATION, not a bare
		# TaggedUnion - _tagged_union_shape gives back the abstract base's
		# own member list (substituted for THIS instantiation), and _union_
		# storage.get needs a monomorphized concrete union (the abstract
		# Result class's own payload union has no real C definition, bare
		# unsubstituted T/E TypeVars) - same "monomorphize_class if
		# Specialization else itself" pattern _emit_binop_fallible_check's
		# own identical situation already uses (see its own comment)
		tagged_shape = self.lowering._type_resolver._tagged_union_shape( result_type )
		assert tagged_shape is not None
		_result_base, result_members = tagged_shape
		err_member = next( a for a in result_members if a.stem == 'Err' )
		ok_member = next( a for a in result_members if a.stem == 'Ok' )
		concrete_result_union = self.lowering.monomorphize_class( result_type ) if isinstance( result_type, Specialization ) else result_type
		tag_attr, _data_attr, _payload_cls, tags = self.lowering._union_storage.get( concrete_result_union )
		bool_cls = self.lowering.discovery.find_name( 'bool', node )

		unique = self._label_id
		obj_var = self._declare_hidden_local( f'__for_obj_{unique}', obj.type, node )
		self._emit( ir.Assign( dest = obj_var, src = obj ))
		next_var = self._declare_hidden_local( f'__for_next_{unique}', result_type, node )

		if remaining_error_type is not None:
			# the loop target's own type is now Result[elem_type,
			# remaining_error_type], not bare elem_type - pre-declared
			# EXPLICITLY (rather than left to _stmt_Assign's own "first
			# assignment infers the type from the RHS" path, which the
			# bare-T case below relies on) because NEITHER Result.Ok(v)
			# nor Result.Err(e) can infer their own OTHER type parameter
			# from their single argument alone (Ok's own call never
			# mentions E at all; Err's never mentions T) - needs an
			# expected_type to resolve against, which _stmt_Assign only
			# supplies for an ALREADY-declared target (existing.type) -
			# confirmed via a real repro otherwise ("type parameter E is
			# inferred as both remaining_error_type and full_error_type").
			result_cls = self.lowering.discovery.find_name( 'Result', node )
			target_type = self.lowering.discovery._get_or_create_specialization( result_cls, [ elem_type, remaining_error_type ] )
			self._declare_hidden_local( node.target.id, target_type, node )

		start_label = self._new_label( 'for_start' )
		continue_label = self._new_label( 'for_continue' )
		end_label = self._new_label( 'for_end' )

		self._emit( ir.Label( name = start_label ))
		# snapshot at the VERY TOP of the loop, before next_var's own
		# per-iteration rebind AND before the loop target's own binding -
		# same reasoning _lower_for_over_indexable's identical comment gives
		# (both bindings are fresh every iteration, not confined-and-torn-
		# down across it) - taken here, before ANY of that, so loop_back_
		# edge()'s later reconciliation sees next_var (like the target) as
		# absent from entry/present at the back edge and decrefs its stale
		# value right before jumping back to start_label. A raw ir.Assign
		# for next_var (the original shape here) would never decref what it
		# held from the PRIOR iteration before overwriting it - leaking one
		# reference per iteration for any RC-typed elem_type (confirmed via
		# a real multi-iteration compiler.refcount() repro). Tried gating
		# next_var's own decref through a plain cfg.assign() call instead of
		# loop_snapshot placement first - doesn't work: cfg.assign() only
		# emits a decref-of-the-old-value when a binding ALREADY exists in
		# self.bindings, but next_var's assignment here is lowered exactly
		# ONCE at compile time (this call happens on every RUNTIME
		# iteration via the goto back-edge, but cfg only ever sees it as the
		# textually-first assignment) - loop_back_edge() is the mechanism
		# actually built for "this binding's value is torn down and
		# replaced every iteration", not a second compile-time assign() call
		loop_snapshot = self._cfg.snapshot()
		next_dest = self._new_temp( result_type )
		self._emit( ir.Call( dest = next_dest, target = next_fn, receiver = obj_var, args = [], kwargs = {} ))
		# track_result=False: next_var's own Result-ness is scaffolding
		# inspected via the raw .tag comparison below, not is_ok()/is_err()/
		# match - same reasoning as the match-subject temp's own call site
		for instr in self._cfg.assign( next_var, next_dest, is_alias = False, track_result = False ):
			self._emit( instr )
		self._emit( ir.Assign( dest = next_var, src = next_dest ))

		tag_expr = ast.Attribute( value = self.lowering._synth_name( next_var.stem, node ), attr = tag_attr.stem, ctx = ast.Load() )
		ast.copy_location( tag_expr, node )
		is_err_test = ast.Compare( left = tag_expr, ops = [ ast.Eq() ], comparators = [ ast.Constant( value = tags[ err_member.stem ] ) ] )
		ast.copy_location( is_err_test, node )
		is_err_cond = self._lower_expr( is_err_test, bool_cls )

		if remaining_error_type is None:
			self._emit( ir.JumpIfTrue( cond = is_err_cond, target = end_label ))
			self._cfg.narrow( next_var.stem, ok_member )
			bind = ast.Assign( targets = [ node.target ], value = self.lowering._synth_name( next_var.stem, node ))
			ast.copy_location( bind, node )
			self._stmt_Assign( bind )
		else:
			self._lower_for_over_iterator_fallible_bind(
				node, next_var, err_member, ok_member, is_err_cond, elem_type, full_error_type,
				remaining_leaves, remaining_error_type, stop_iteration_cls, end_label, unique,
			)

		break_narrowed, break_live, continue_captured = self._lower_loop_body( node.body, continue_label = continue_label, break_label = end_label, loop_snapshot = loop_snapshot )
		try:
			back_edge_instructions = self._cfg.loop_back_edge( loop_snapshot.bindings, self._current_fn.qualname, entry_results = loop_snapshot.results )
		except CompileError as e:
			self.lowering.discovery.fail( str( e ), node )
		for instr in back_edge_instructions:
			self._emit( instr )
		self._cfg.restore( loop_snapshot )
		# Phase 8 - see _lower_for_range's own identical call/comment
		self._cfg.merge_loop_exits( dict( loop_snapshot.narrowed ), break_narrowed, set( loop_snapshot.live ), break_live )

		# see _lower_for_range's own identical comment on continue_captured
		if continue_captured:
			self._emit( ir.Label( name = continue_label ))
		self._emit( ir.Jump( target = start_label ))
		self._emit( ir.Label( name = end_label ))

	def _lower_for_over_iterator_fallible_bind(
		self, node: ast.For, next_var: Variable, err_member: Variable, ok_member: Variable, is_err_cond: ir.Operand,
		elem_type: Type, full_error_type: Type, remaining_leaves: list[Type], remaining_error_type: Type,
		stop_iteration_cls: Type, end_label: str, unique: int,
	) -> None:
		''' _lower_for_over_iterator's own Result[T,E'] loop-target binding
		(E' = the generator's declared error type minus StopIteration) -
		split out into its own method purely for readability, not reused
		elsewhere. node.target's own type (Result[elem_type,remaining_
		error_type]) is already pre-declared by the caller before this
		runs (see its own comment on why - neither Result.Ok(v) nor
		Result.Err(e) can infer their own OTHER type parameter alone).

		Three-way outcome (Ok / Err-StopIteration / Err-real-error), but
		StopIteration is a pure early exit (jumps straight to end_label,
		contributing nothing to the loop body's own entry state) - so this
		is built as ONE binary branch (Ok vs Err) via _lower_binary_branch,
		with the Err side recursing into its OWN binary branch (StopIteration-
		exit vs real-error-bind), and the real-error side recursing into a
		right-nested CHAIN of one more binary branch PER remaining leaf
		(dispatch, below) when there's more than one - each arm narrow()s
		the extracted error to that ONE concrete leaf (cfg.py's narrow()
		only yields a concrete type when exactly one candidate remains),
		needed for Result.Err(e)'s own generic inference to resolve E
		correctly (matches type_resolver.py's own _desugar_iterator_for,
		which hit the identical trap with a single wildcard arm instead of
		one explicit arm per leaf - see its own comment). '''
		bool_cls = self.lowering.discovery.find_name( 'bool', node )
		if full_error_type.resolve is not None:
			full_error_type.resolve()
		for attr in full_error_type.attributes:
			if attr.resolve is not None:
				attr.resolve()
		e_tag_attr, _e_data_attr, _e_payload_cls, e_tags = self.lowering._union_storage.get( full_error_type )
		# identity-keyed, not by value - Type dataclasses aren't all
		# hashable (e.g. RCClass), and every leaf here is interned anyway
		# (the same _atomic_leaves-derived Type object backs both this
		# union's own member and remaining_leaves/stop_iteration_cls)
		e_members = { id( m.type ): m for m in full_error_type.attributes }

		def rewrap_bind( ctor_attr: str, bind_name: str, payload_type: Type ) -> None:
			rewrap = ast.Call(
				func = ast.Attribute( value = ast.Name( id = 'Result', ctx = ast.Load() ), attr = ctor_attr, ctx = ast.Load() ),
				args = [ ast.Name( id = bind_name, ctx = ast.Load() ) ], keywords = [],
			)
			ast.copy_location( rewrap, node ); ast.copy_location( rewrap.func, node ); ast.copy_location( rewrap.func.value, node )
			bind = ast.Assign( targets = [ node.target ], value = rewrap )
			ast.copy_location( bind, node )
			self._stmt_Assign( bind )
			if payload_type.is_rc():
				# bind_name's own extraction (above) is an aliasing-read
				# incref, and Result.Ok(...)/Result.Err(...) increfs its own
				# argument AGAIN when wrapping it - bind_name's own
				# extracted reference is never otherwise consumed (node.
				# target holds the REWRAPPED value's own, independent
				# reference), so without this it's a real per-iteration
				# leak - confirmed via a real repro, matches type_resolver.
				# py's _desugar_iterator_for's identical situation (its own
				# comment has the full reasoning)
				decref_call = ast.Expr( value = ast.Call(
					func = ast.Attribute( value = ast.Name( id = 'compiler', ctx = ast.Load() ), attr = 'decref', ctx = ast.Load() ),
					args = [ ast.Name( id = bind_name, ctx = ast.Load() ) ], keywords = [],
				))
				ast.copy_location( decref_call, node ); ast.copy_location( decref_call.value, node )
				ast.copy_location( decref_call.value.func, node ); ast.copy_location( decref_call.value.func.value, node )
				self._lower_stmt( decref_call )

		def ok_thunk() -> bool:
			self._cfg.narrow( next_var.stem, ok_member )
			ok_bind_name = f'__for_ok_{unique}'
			self._declare_hidden_local( ok_bind_name, elem_type, node )
			extract = ast.Assign( targets = [ ast.Name( id = ok_bind_name, ctx = ast.Store() ) ], value = self.lowering._synth_name( next_var.stem, node ))
			ast.copy_location( extract, node )
			self._stmt_Assign( extract )
			rewrap_bind( 'Ok', ok_bind_name, elem_type )
			return False

		def err_thunk() -> bool:
			self._cfg.narrow( next_var.stem, err_member )
			err_bind_name = f'__for_err_{unique}'
			self._declare_hidden_local( err_bind_name, full_error_type, node )
			extract = ast.Assign( targets = [ ast.Name( id = err_bind_name, ctx = ast.Store() ) ], value = self.lowering._synth_name( next_var.stem, node ))
			ast.copy_location( extract, node )
			self._stmt_Assign( extract )

			def leaf_test_cond( leaf_type: Type ) -> ir.Operand:
				leaf_member = e_members[ id( leaf_type ) ]
				tag_expr = ast.Attribute( value = self.lowering._synth_name( err_bind_name, node ), attr = e_tag_attr.stem, ctx = ast.Load() )
				ast.copy_location( tag_expr, node )
				test = ast.Compare( left = tag_expr, ops = [ ast.Eq() ], comparators = [ ast.Constant( value = e_tags[ leaf_member.stem ] ) ] )
				ast.copy_location( test, node )
				return self._lower_expr( test, bool_cls )

			def leaf_bind_thunk( leaf_type: Type ) -> bool:
				# narrow()'d reads of err_bind_name aren't cached - EVERY
				# ast.Name(id=err_bind_name) read after this narrow() re-
				# extracts through the payload independently (a fresh
				# aliasing-read incref each time, same mechanism as any
				# other narrowed read), unlike an ordinary plain-variable
				# read which always refers to the SAME already-extracted
				# value. rewrap_bind reads bind_name TWICE (the rewrap's
				# own argument, then the decref) - reading err_bind_name
				# itself directly for both would incref it twice while
				# only ever decref-ing one of those extractions, leaking
				# the other - confirmed via a real repro. Extract ONCE
				# into a fresh, real (non-narrowed) local instead, matching
				# ok_thunk's own identical pattern - both of rewrap_bind's
				# own reads then correctly refer to the SAME extraction.
				leaf_bind_name = f'__for_err_{leaf_type.stem}_{unique}'
				self._declare_hidden_local( leaf_bind_name, leaf_type, node )
				self._cfg.narrow( err_bind_name, e_members[ id( leaf_type ) ] )
				leaf_extract = ast.Assign( targets = [ ast.Name( id = leaf_bind_name, ctx = ast.Store() ) ], value = self.lowering._synth_name( err_bind_name, node ))
				ast.copy_location( leaf_extract, node )
				self._stmt_Assign( leaf_extract )
				rewrap_bind( 'Err', leaf_bind_name, leaf_type )
				return False

			def dispatch_real_error( leaves: list[Type] ) -> bool:
				if len( leaves ) == 1:
					return leaf_bind_thunk( leaves[0] )
				leaf = leaves[0]
				cond = leaf_test_cond( leaf )
				self._lower_binary_branch(
					cond, node,
					lambda: leaf_bind_thunk( leaf ),
					lambda: dispatch_real_error( leaves[1:] ),
				)
				return False

			is_stop_iteration_cond = leaf_test_cond( stop_iteration_cls )

			def stop_iteration_thunk() -> bool:
				self._emit( ir.Jump( target = end_label ))
				return True

			def real_error_thunk() -> bool:
				return dispatch_real_error( remaining_leaves )

			self._lower_binary_branch( is_stop_iteration_cond, node, stop_iteration_thunk, real_error_thunk )
			return False

		self._lower_binary_branch( is_err_cond, node, err_thunk, ok_thunk )

	def _stmt_diverges( self, stmt: ast.stmt ) -> bool:
		''' true if `stmt` never falls through to the statement after it -
		either structurally (return/break/continue) or because it's a bare
		call expression to a function declared -> NoReturn (sys.panic, most
		commonly). Used by _stmt_If (true_terminates/false_terminates) to
		decide whether a branch's own ending narrowed/bindings state can
		reach the if's join point at all - see merge_if()'s own docstring.
		Resolved via _resolve_callee_target rather than a full _lower_call -
		this only needs the CALLEE's declared return type, not a real
		lowered call (the statement was already lowered by the caller's own
		loop before this runs), and _resolve_callee_target is a pure lookup
		with no scheduling side effects beyond _resolve_callable's ordinary
		signature-resolution. A receiver call (x.method()) or anything
		_resolve_callee_target can't resolve without a receiver just isn't
		recognized here - NoReturn is overwhelmingly a free-function/sys.*
		shape (sys.panic, sys.exit, ...), and misses just fall back to
		today's existing (safe, if incomplete) behavior. '''
		if isinstance( stmt, ( ast.Return, ast.Break, ast.Continue )):
			return True
		if isinstance( stmt, ast.If ):
			# an if/else BOTH of whose own branches diverge is itself
			# terminating, even though it isn't literally a Return/Break/
			# Continue - the shape a `match` statement desugars to
			# (type_resolver.py's visit_Match, chained ast.If via
			# tail.orelse) whenever every case returns. Recursing through
			# _stmt_diverges itself (rather than a one-level check) handles
			# arbitrarily long desugared case chains, each nested one level
			# deeper than the last. No orelse at all means the false path
			# always falls through, so it can never qualify.
			return (
				bool( stmt.body ) and self._stmt_diverges( stmt.body[-1] )
				and bool( stmt.orelse ) and self._stmt_diverges( stmt.orelse[-1] )
			)
		if not ( isinstance( stmt, ast.Expr ) and isinstance( stmt.value, ast.Call )):
			return False
		if self.lowering._defer_kind_of_call( stmt.value ) is not None:
			# defer(...)/errdefer(...) as an if-branch's LAST statement (a
			# natural, common shape - arm a cleanup right before the branch
			# falls through) is not itself a real call: `defer`/`errdefer`
			# are recognized purely by this AST shape (_stmt_Expr, before
			# ordinary call resolution ever runs - see this file's own
			# module docstring), never registered as an actual Name anywhere.
			# Falling through to _resolve_callee_target below tried to look
			# up 'defer' as an ordinary callable and failed outright ("name
			# 'defer' is not defined") - a real, confirmed compile error on
			# every defer/errdefer that happens to be the last statement of
			# a non-terminating if-branch (found via lib/builtins/__str.py's
			# `if loc is not None: defer(freelocale(loc))`, the only defer
			# call in this codebase shaped that way - every other call site
			# happens to sit at the top level or right after an early-return
			# guard, never as an if-branch's own last statement, which is
			# why this went unnoticed until real POSIX-target compilation
			# actually exercised it). defer/errdefer always falls through
			# (arms a flag, never diverges) - never NoReturn-shaped
			return False
		target = self.lowering._type_resolver._resolve_callee_target( stmt.value.func )
		fn = target.base if isinstance( target, Specialization ) else target
		if not isinstance( fn, Function ):
			return False
		return isinstance( fn.return_type, Scalar ) and fn.return_type.stem == 'NoReturn'

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
		true_end_live = self._cfg.live_snapshot()
		# return/break/continue as a branch's own last statement means
		# that branch never reaches the if's join point at all - see
		# merge_if()'s own comment on why that has to be treated
		# differently from an ordinary falling-through branch (full
		# terminator/dead-code analysis for anything deeper - nested ifs
		# that both terminate, etc - is future work, not attempted here)
		true_terminates = bool( node.body ) and self._stmt_diverges( node.body[-1] )

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
			false_end_live = self._cfg.live_snapshot()
			false_terminates = bool( node.orelse ) and self._stmt_diverges( node.orelse[-1] )
		else:
			false_captured = []
			false_end = dict( entry_snapshot.bindings )
			false_end_results = set( entry_snapshot.results )
			false_end_narrowed = dict( entry_snapshot.narrowed )
			false_end_live = set( entry_snapshot.live )
			false_terminates = False

		self._cfg.restore( entry_snapshot )
		self._instructions = outer_instructions
		try:
			true_extra, false_extra, removed = self._cfg.merge_if(
				entry_snapshot.bindings, true_end, false_end, self._current_fn.qualname,
				entry_results = entry_snapshot.results, true_end_results = true_end_results, false_end_results = false_end_results,
				true_terminates = true_terminates, false_terminates = false_terminates,
				true_end_narrowed = true_end_narrowed, false_end_narrowed = false_end_narrowed,
				true_end_live = true_end_live, false_end_live = false_end_live,
			)
		except CompileError as e:
			self.lowering.discovery.fail( str( e ), node )
		# `removed` only drives the RC Decref instructions above (already
		# spliced into true_extra/false_extra) - it must NOT also remove
		# these names from fn.names. This language has no block scoping (see
		# cfg.py's own module docstring): a name introduced anywhere in the
		# function body stays a real name for the rest of it. Whether a
		# later reference is actually valid is now the new _live mechanism's
		# job (_expr_Name's own liveness gate) - it correctly reports "not
		# initialized on all code branches" for exactly this case, instead
		# of the misleading "is not defined" this loop used to produce by
		# deleting the name out from under a later reference entirely.

		for instr in true_captured:
			self._emit_captured( instr )
		for instr in true_extra:
			self._emit( instr )
		if node.orelse:
			end_label = self._new_label( 'if_end' )
			self._emit( ir.Jump( target = end_label ))
			self._emit( ir.Label( name = else_label ))
			for instr in false_captured:
				self._emit_captured( instr )
			for instr in false_extra:
				self._emit( instr )
			self._emit( ir.Label( name = end_label ))
		else:
			self._emit( ir.Label( name = else_label ))
			# false_extra used to always be empty here (merge_if's "fresh on
			# exactly one branch" case can only ever populate
			# false_instructions when a real ast.orelse existed, since
			# false_end is otherwise just entry_bindings copied verbatim -
			# see false_end's own fallback above) - but merge_if's
			# ownership-disagreement flag reconciliation (the "if x is
			# None: x = Owned(...)" idiom, no else needed) DOES need to land
			# a disarm Assign on exactly this implicit "condition was
			# false" path - dropping it here would silently leave the flag
			# permanently armed, decref'ing a merely-borrowed value at the
			# eventual epilogue.
			for instr in false_extra:
				self._emit( instr )

	# --- expressions -----------------------------------------------------------

	def _lower_expr( self, node: ast.expr, expected_type: Type|None, *, strict: bool = True ) -> ir.Operand:
		''' `strict=False` (default True): `expected_type` here is a HINT for
		inference (e.g. _lower_binary_operands passing the left operand's own
		type down to help type an untyped literal, or to help a nested
		generic call bind its type params) rather than a real requirement the
		lowered operand must satisfy - skips the safe-scalar-widening
		coercion and the final _check_assignable rejection (both new; see
		their own comments below), but still applies the pre-existing,
		unconditionally-safe TaggedUnion/RCClass-upcast/pointer-cast
		coercions. Every ordinary call site (assignment, call argument,
		return, ...) leaves this at its default True. '''
		method = getattr( self, f'_expr_{node.__class__.__name__}', None )
		if method is None:
			self.lowering.discovery.fail( f'unsupported expression: {ast.unparse(node)}', node )
		operand = method( node, expected_type )
		return self._coerce_or_check_operand( operand, expected_type, node, strict = strict )

	def _coerce_or_check_operand( self, operand: ir.Operand, expected_type: Type|None, node: ast.AST, *, strict: bool = True, context: str|None = None ) -> ir.Operand:
		''' the shared post-dispatch tail: given an operand (freshly produced
		by one of the _expr_X dispatch methods above, OR - unlike every
		other caller - already-lowered and handed in directly, with no AST
		node of its own left to re-dispatch) and an expected_type, applies
		every legitimate coercion in turn and, if none apply and `strict`,
		rejects a genuine mismatch. Factored out of _lower_expr (which calls
		this immediately after dispatch, `node` there being the same node
		method() was just given) specifically so _lower_union_receiver_call
		can call this a SECOND time, once per union leaf, against an
		operand it already has - never re-lowering/re-evaluating the
		original argument expression (which would double its side effects
		once per leaf) while still getting the exact same coercion-or-
		rejection treatment an ordinary call argument gets. `context`, if
		given, only affects _check_assignable's own failure message (see
		its own docstring) - it plays no role in which coercion, if any,
		applies. '''
		# post-hoc, not a pre-emptive override of expected_type before
		# dispatch: a node kind that already produces the right union type
		# on its own (an explicit Result.Ok(x) call, a match-narrowed
		# union-typed value, ...) already satisfies operand.type is
		# expected_type and skips this entirely - only a genuine mismatch
		# (a plain leaf value where the union itself was expected) reaches
		# _coerce_into_union, see its own comment
		#
		# unwrap a Specialization first - a GENERIC union return/annotation
		# type (Result[T,E], or any user @union class Foo[T]:) is always a
		# Specialization wrapping the abstract TaggedUnion, never a bare
		# TaggedUnion instance itself, so a plain isinstance check here
		# (unlike is_rc/rc_leaves/is_result_type/_as_specialization elsewhere
		# in this codebase, all of which unwrap first) silently never fired
		# for one - a bare leaf value flowing into a generic union-typed
		# context (e.g. `return some_T` where T|None is declared) never got
		# coerced at all. Monomorphized (not the bare abstract class) once
		# it's actually needed below - the abstract class's own .attributes
		# hold bare TypeVars (T/E), never a real leaf's type, so
		# _coerce_into_union needs the CONCRETE, substituted union to have
		# any chance of matching operand.type - mirrors the identical
		# "monomorphize_class if Specialization else itself" pattern already
		# used elsewhere for this same reason (e.g. _expr_Name's own
		# union-narrowing branch)
		expected_union: TaggedUnion|None = None
		if operand.type is not expected_type: # cheap identity check first - skip monomorphize_class entirely on the common "already matches" path
			if isinstance( expected_type, TaggedUnion ):
				expected_union = expected_type
			elif isinstance( expected_type, Specialization ) and isinstance( expected_type.base, TaggedUnion ):
				expected_union = self.lowering.monomorphize_class( expected_type )
		if expected_union is not None and operand.type is not expected_union:
			# operand being ITSELF union-shaped does NOT automatically rule out
			# coercion - a nominal @union (e.g. HTTPError) is exactly as valid a
			# member of a WIDER union (OSError|HTTPError) as any plain leaf type
			# is. But it doesn't automatically qualify either: operand can ALSO be
			# a genuinely unrelated Result[T,OtherE] flowing into a Result[T,E]-
			# expected context (Result is itself a TaggedUnion-based
			# Specialization) - that shape is handled by a SEPARATE mechanism
			# entirely (_stmt_Return's own _maybe_widen_return_result, emitting
			# ir.WidenResult), which must get the first chance to run, not be
			# preempted by a hard failure here. So this only actually invokes
			# _coerce_into_union (which hard-fails on a genuine mismatch) when
			# operand's whole type already verbatim matches one of expected_
			# union's own leaf attribute types - the same check _coerce_into_
			# union would use to decide to wrap it anyway, just performed as a
			# non-failing probe first so a genuine non-member (e.g. that
			# unrelated Result[T,OtherE]) is left untouched for its own,
			# more-specific caller-side handling instead
			if any(
				self.lowering._type_resolver._same_type( attr.type, operand.type )
				for attr in expected_union.attributes
			):
				was_fresh = self._cfg.is_fresh_temp( operand )
				pre_coerce = operand
				operand = self._coerce_into_union( operand, expected_union, node )
				if was_fresh:
					# _coerce_into_union's own ctor call unconditionally
					# increfs whatever it wraps (needed when the wrapped
					# value is a BORROWED reference its own caller still
					# needs afterward) - but a value that was already a
					# fresh, solely-owned temp here (a Call/Allocate
					# result) doesn't need that extra reference kept
					# alive too: release it right here, in place, rather
					# than leaving it as a dangling pending-temp
					# obligation for whatever later flush would otherwise
					# decref it unconditionally. Confirmed as a real bug
					# via a ternary branch specifically (ASAN: SEGV
					# reading uninitialized stack memory on the OTHER
					# branch, where this temp was never even created) -
					# every other caller of this shared coercion tail has
					# the identical gap, just silently masked there by
					# dumb luck (a non-branching statement's own
					# unconditional flush still nets the right refcount
					# when there's no branch boundary for it to straddle).
					for instr in self._cfg.decref( pre_coerce.type, pre_coerce ):
						self._emit( instr )
					self._cfg.untrack_temp( pre_coerce )
		# a derived RCClass value flowing into a base-class context (arg, return,
		# assignment) is an upcast: struct Derived* -> struct Base*, which C
		# rejects without an explicit cast. A CastWrap is a borrowed reinterpret
		# (its temp is never RC-registered - see _lower_bound_method_closure's own
		# note), so this adds no incref/decref, exactly right for passing the
		# SAME object under its base type. Restricted to a genuine strict-subclass
		# relationship so it never masks an unrelated type mismatch.
		elif ( expected_type is not None and operand.type is not expected_type
				and self._is_rcclass_upcast( operand.type, expected_type ) ):
			dest = self._new_temp( expected_type )
			self._emit( ir.CastWrap( dest = dest, operand = operand ) )
			operand = dest
		# a same-signedness, strictly-wider scalar (i32 -> i64, u8 -> u32,
		# f32 -> f64) is a safe, value-preserving widening - allowed
		# implicitly, same CastWrap shape as the RCClass upcast above, but
		# this one is a REAL value conversion (C's own sign-/zero-extension,
		# not a pointer reinterpret) - see _is_safe_scalar_widening's own
		# docstring for exactly which pairs qualify and why isize/usize are
		# deliberately excluded. Gated on `strict` (see _lower_expr's own
		# parameter comment) - _lower_binary_operands' own cross-operand
		# HINTING must never trigger this: `c: f64 = a + b` (a: f64, b: f32)
		# needs to keep hitting _lower_binop_values' own deliberately-
		# stricter "floating-point operation requires both operands to be
		# the SAME type, cast explicitly" rule, not have b silently widened
		# to f64 here first.
		elif ( strict and expected_type is not None and operand.type is not expected_type
				and self._is_safe_scalar_widening( operand.type, expected_type ) ):
			dest = self._new_temp( expected_type )
			self._emit( ir.CastWrap( dest = dest, operand = operand ) )
			operand = dest
		# interchangeable object-pointer types (Ptr[T]/ConstPtr[T], any T/U) -
		# generalized here from _expr_Name/_expr_Attribute's own narrower,
		# node-kind-specific calls to the same helper, so a pointer value
		# reaching this point through ANY expression kind (a Call result, a
		# ternary, ...) gets the identical treatment - _maybe_castwrap_pointer
		# re-checks its own guard and is a no-op once operand.type already
		# matches, so calling it unconditionally here is safe even when
		# _expr_Name/_expr_Attribute already tried it themselves. NOT gated
		# on `strict` - this coercion predates this change and is always
		# safe (any two object pointers share one C representation)
		elif expected_type is not None and operand.type is not expected_type:
			operand = self._maybe_castwrap_pointer( operand, expected_type )
		# the single choke point: every legitimate coercion above already had
		# its chance to rewrite `operand` into something matching expected_type -
		# anything still mismatched past this point is a genuine type error
		# (lowering.py's own longstanding gap: "a genuine argument-type
		# mismatch isn't checked anywhere yet" - see _check_assignable).
		# Skipped when strict=False (see _lower_expr's own `strict` param
		# comment) - a HINT passed down for inference purposes only, not a
		# real requirement the operand must satisfy; the caller (e.g.
		# _lower_binop_values' own float-same-type check) is responsible for
		# validating the ACTUAL requirement itself in that case.
		if strict:
			self._check_assignable( operand, expected_type, node, context = context )
		return operand

	def _is_rcclass_upcast( self, sub: Type|None, sup: Type|None ) -> bool:
		''' True if `sub` is a strict subclass (transitively) of `sup`, both being
		RCClasses (or specializations of one) - i.e. a derived->base upcast.

		Deliberately NOT expressed with Type.is_rc()/is_rc_pointer(): this asks
		about INHERITANCE, not reference counting, and it needs the real RCClass
		OBJECT to walk .base with. A tuple[T...] is every bit as much an RC
		pointer as an RCClass but has no inheritance chain at all, so widening
		this guard to is_rc_pointer() would let one into an upcast test it can
		never meaningfully participate in. The isinstance is the right check
		here - see mpy_types.Type's own note on the RC vs layout vs class-kind
		distinction. '''
		def rc_of( t: Type|None ) -> Type|None:
			base = t.base if isinstance( t, Specialization ) else t
			return base if isinstance( base, RCClass ) else None
		sub_c, sup_c = rc_of( sub ), rc_of( sup )
		if sub_c is None or sup_c is None:
			return False
		c = getattr( sub_c, 'base', None )
		while c is not None:
			if c is sup_c:
				return True
			c = getattr( c, 'base', None )
		return False

	# same-signedness scalar widening chains - isize/usize deliberately
	# excluded (their concrete width is target-dependent, see discovery.py's
	# get_intrinsics: sizeof = active_target['bits'] - so treating them as
	# interchangeable with a same-width fixed type would be non-portable
	# across targets; usize(x)/isize(x) stay explicit-cast-only), as are
	# bool/NoneType (never "numeric" for this purpose)
	_SIGNED_INT_WIDENING_ORDER = ( 'i8', 'i16', 'i32', 'i64', 'i128' )
	_UNSIGNED_INT_WIDENING_ORDER = ( 'u8', 'u16', 'u32', 'u64', 'u128' )
	_FLOAT_WIDENING_ORDER = ( 'f32', 'f64' )

	def _is_safe_scalar_widening( self, operand_type: Type|None, expected_type: Type|None ) -> bool:
		''' True if operand_type -> expected_type is a same-family (signed
		int / unsigned int / float), strictly-growing-width scalar
		conversion - i32->i64, u8->u32, f32->f64, etc. Narrowing, sign-
		changing (i32->u32), bool<->numeric, and anything involving isize/
		usize are all deliberately NOT safe widenings here - those require
		an explicit T(x) cast, matching this language's own "no implicit
		conversion" design (SYNTAX.md) everywhere else. `float`/`double` are
		the SAME Scalar object as f32/f64 (see discovery.py's get_intrinsics),
		so comparing by .stem already handles both spellings uniformly. '''
		if not isinstance( operand_type, Scalar ) or not isinstance( expected_type, Scalar ):
			return False
		for order in ( self._SIGNED_INT_WIDENING_ORDER, self._UNSIGNED_INT_WIDENING_ORDER, self._FLOAT_WIDENING_ORDER ):
			if operand_type.stem in order and expected_type.stem in order:
				return order.index( expected_type.stem ) > order.index( operand_type.stem )
		return False

	def _check_assignable( self, operand: ir.Operand, expected_type: Type|None, node: ast.AST, *, context: str|None = None ) -> None:
		''' the single choke point for lowering.py's own longstanding,
		self-documented gap ("a genuine argument-type mismatch isn't
		checked anywhere yet (no general type-checking pass exists)") -
		called last from _coerce_or_check_operand (in turn called from both
		_lower_expr and, a second time per leaf, _lower_union_receiver_call),
		after every legitimate coercion (TaggedUnion wrap, RCClass upcast,
		safe scalar widening, interchangeable pointer cast) already had its
		chance to rewrite `operand` into something matching expected_type.
		`context`, if given, is prefixed onto the failure message - used by
		union-receiver dispatch to name which leaf/parameter disagreed,
		since a bare "expected X, got Y" doesn't otherwise say WHICH of
		several call targets is the one that actually declared X. Uses
		_same_type,
		not raw `is`, for the equality check: a bare Specialization and its
		own already-monomorphized form (or a bare TupleType and its own
		resolved backing RCClass) are the SAME type reached through two
		different representations, not a real conflict - see
		TypeResolver._same_type's own docstring. Also exempts two shapes
		_expr_Constant's own pre-existing literal-kind check already treats
		as non-mismatches for the identical reason (same precedent, applied
		here too since this is the general case, not just the literal one):
		a still-unbound generic TypeVar as expected_type (e.g.
		_lower_allocate_fields lowering a generic no-__init__ field's own
		value against its still-abstract, unsubstituted field.type - nothing
		concrete exists yet to validate against, T is what's BEING inferred,
		not what to check the value against), and a CEnum on either side
		matched against its own declared underlying value_type (e.g.
		`x: builtins.OSError = ...; return x` from a function declared -> u32
		- "a CEnum has exactly the same runtime representation as its
		underlying type", per _try_lower_construct_call's own CEnum-
		construction comment, so the two are interchangeable both directions,
		not just at construction time). Also unwraps move[T]/copy[T] on the
		expected side: both are pure compile-time annotation wrappers (see
		Move/Copy's own docstrings) - a real call-site VALUE is always
		plain T, never literally typed as Move[T]/Copy[T] itself (that
		ownership transfer is tracked separately, by _apply_move_hook/cfg.py,
		not by the value's own .type). '''
		if expected_type is None or operand.type is None:
			return
		if self.lowering._type_resolver._same_type( operand.type, expected_type ):
			return
		if isinstance( expected_type, TypeVar ):
			return
		if isinstance( expected_type, CEnum ) and operand.type is expected_type.value_type:
			return
		if isinstance( operand.type, CEnum ) and expected_type is operand.type.value_type:
			return
		if isinstance( expected_type, ( Move, Copy )):
			self._check_assignable( operand, expected_type.inner, node, context = context )
			return
		prefix = f'{context}: ' if context is not None else ''
		self.lowering.discovery.fail(
			f'{prefix}{ast.unparse(node)}: expected {expected_type.qualname}, got {operand.type.qualname} - '
			f'these are different types; convert explicitly if this is intentional '
			f'(e.g. {expected_type.stem}(...) for a scalar target)',
			node,
		)

	def _maybe_castwrap_pointer( self, operand: ir.Operand, expected_type: Type|None ) -> ir.Operand:
		''' When a pointer-typed value flows into a context expecting a DIFFERENT
		pointer type (e.g. self.__metadata: Ptr[_FastListMetadata] passed to
		sys.memcpy's src: ConstPtr[u8]), insert a CastWrap - in C all object
		pointers share one representation, so this is safe, and without it GCC/
		clang reject the call as -Wincompatible-pointer-types. Function pointers
		(Ptr[Callable[...]]) are deliberately excluded: casting them interferes
		with generic-parameter inference through callable arguments, and object<->
		function pointer casts are not the interchangeable case above. This is the
		same coercion _expr_Name applies to bare locals, factored out so the
		attribute path (self.field reads) gets it too. '''
		tr = self.lowering._type_resolver
		if ( expected_type is not None and operand.type is not expected_type
				and tr._is_ptr_specialization( operand.type ) and tr._is_ptr_specialization( expected_type )
				and tr._callable_type_of( operand.type ) is None and tr._callable_type_of( expected_type ) is None ):
			dest = self._new_temp( expected_type )
			self._emit( ir.CastWrap( dest = dest, operand = operand ) )
			return dest
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
		# _same_type, not raw `is` - a leaf's declared type (e.g. list[Op]
		# substituted into a generic union's own attributes) and operand's own
		# type can be two different Specialization objects for the identical
		# instantiation (one already-monomorphized, one freshly built from an
		# annotation) - see TypeResolver._same_type's own docstring, the exact
		# same duality _unify_type_param/_check_assignable already guard
		# against elsewhere. Without this, a bare `list[Op]` return against a
		# declared `list[Op]|None` return type wrongly fell through to the
		# "not one of its members" failure below.
		leaf = next( ( attr for attr in union.attributes if self.lowering._type_resolver._same_type( attr.type, operand.type ) ), None )
		if leaf is None:
			self.lowering.discovery.fail(
				f'{ast.unparse(node)}: expected {union.qualname}, got a type that is not one of its members',
				node,
			)
		ctor_fn = union.get_local_or_raise( leaf.stem )
		self.lowering.schedule( ctor_fn )
		self.lowering.schedule( ctor_fn.return_type )
		for p in ctor_fn.parameters or []:
			self.lowering.schedule( p.type )
		dest = self._new_temp( union )
		self._emit( ir.Call( dest = dest, target = ctor_fn, receiver = None, args = [ operand ], kwargs = {} ))
		# see _is_aliasing_expr's own comment on this flag: `dest` is a
		# fresh, already-Increfed Call result (the ctor's own body increfs
		# the leaf it wraps), never still-aliasing whatever `node` (the
		# original, pre-coercion expression) looked like to a caller that
		# only has the ast around, not this operand
		dest.is_union_coerce_result = True
		return dest

	def _expr_Name( self, node: ast.Name, expected_type: Type|None ) -> ir.Operand:
		name = self.lowering.discovery.find_name( node.id, node )
		if isinstance( name, Function ):
			return self._lower_function_ref( name, node )
		if not isinstance( name, Variable ):
			self.lowering.discovery.fail( f'{node.id!r} is not a value, cannot use it as an expression', node )
		# definite-assignment gate: a local (never a global - those aren't
		# tracked by this function's own _live set at all, see cfg.py's
		# is_live() docstring) that's in scope (fn.names, so find_name above
		# already succeeded) but not provably assigned on every path that
		# reaches here - e.g. a bare `x: T` declaration only assigned inside
		# one if-branch, then read unconditionally after it
		if not name.is_global and not self._cfg.is_live( node.id ):
			self.lowering.discovery.fail( f'{node.id!r} is not initialized on all code branches', node )
		self.lowering._ensure_resolved( name )
		member = self._cfg.narrowed_member( node.id )
		# _same_type, not raw `is` - same PLAN_COMPILER_BUG_SWEEP.md audit
		# that found the other Shape 1 candidates flagged this escape-hatch
		# comparison too, on the theory that expected_type and name.type
		# could be two different objects for the identical union
		# instantiation. No repro could be constructed for it despite
		# several attempts (generic-substituted vs fresh-annotation
		# parameter types, local-variable-annotation vs fresh-annotation) -
		# unlike the OTHER unconfirmed Shape 1 candidates (gated by a
		# separate, already-known upstream bug), this one looks like it may
		# genuinely be unreachable: _get_or_create_union caches by a
		# qualname-text key (ARCHITECTURE.md), which is insensitive to
		# whether the union's own MEMBER Specializations are identical
		# objects, so two structurally-identical union ANNOTATIONS seem to
		# always land on the same cached union object regardless. Applied
		# anyway for consistency/defense-in-depth - strictly safer than
		# `is` (accepts everything `is` did, plus more), so this can only
		# widen when the escape hatch correctly fires, never narrow it.
		if member is not None and not self.lowering._type_resolver._same_type( expected_type, name.type ):
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
		# a pointer-typed local flowing into a context expecting a
		# differently-typed pointer (e.g. return ptr where ptr: Ptr[u8] but
		# the function returns Ptr[T]) used to get a CastWrap inserted right
		# here - now handled uniformly by _lower_expr's own generalized
		# pointer-coercion branch instead (which correctly excludes function
		# pointers, unlike this now-removed inline copy did), so every
		# expression kind gets identical treatment, not just a bare Name
		return name

	def _expr_NamedExpr( self, node: ast.NamedExpr, expected_type: Type|None ) -> ir.Operand:
		''' walrus (`x := expr`): the same two ast.Name-target branches
		_stmt_Assign uses (reassignment vs first declaration - `target` is
		always a bare ast.Name per Python's own grammar), except this is an
		EXPRESSION, so it hands back the assigned operand as its own value
		instead of emitting a void statement. Doesn't thread expected_type
		into the RHS lowering below - _lower_expr's own wrapper already
		re-applies _coerce_or_check_operand to whatever this returns, so
		outer-context coercion (e.g. `x: i64 = (y := 5)`) happens for free,
		same as every other _expr_* method. All locals here are function-
		scoped unconditionally (not block-scoped), so a walrus-bound name
		stays visible after its enclosing if/while exactly like an ordinary
		preceding assignment would - no special escape-the-block handling
		needed, unlike real Python's own comprehension-scoping nuance
		(moot anyway - this language has no comprehensions). '''
		target = node.target
		assert isinstance( target, ast.Name )
		existing = self._existing_local_or_none( target.id, node, 'cannot assign to it' )
		if existing is not None:
			self._cfg.unnarrow( target.id )
			operand = self._lower_expr( node.value, existing.type )
			for instr in self._cfg_assign( existing, operand, is_alias = self.lowering._is_aliasing_expr( node.value, operand ), node = node ):
				self._emit( instr )
			self._emit( ir.Assign( dest = existing, src = operand ))
			return existing
		var, operand = self._declare_local( target.id, node, lambda expected: self._lower_expr( node.value, expected ))
		is_alias = self.lowering._is_aliasing_expr( node.value, operand )
		for instr in self._cfg_assign( var, operand, is_alias = is_alias, node = node ):
			self._emit( instr )
		self._emit( ir.Assign( dest = var, src = operand ))
		return var

	def _collect_free_variables( self, roots: list[ast.AST], param_names: set[str], node: ast.AST ) -> list[tuple[str,Variable]]:
		# a nested def/lambda's own parameters/locally-assigned names,
		# module-level names, and builtins resolve normally; anything else
		# reaching into the immediately enclosing function's own scope is a
		# CAPTURE - collected and returned here (as (name, Variable) pairs,
		# ready for _build_closure_env_class/the AST rewrite - see the
		# capturing-closures plan) rather than rejected, now that a
		# representation decision has been made (real closures - see
		# ClosureType/PLAN_CALLABLE.md's own bound-method precedent,
		# generalized). `roots` is the def's own body (a list of statements)
		# or a lambda's own body wrapped in a single-element list (a bare
		# expression - lambda syntax forbids assignment statements, but NOT
		# ast.NamedExpr/walrus, which also binds via Name(Store) - the same
		# walk covers both shapes uniformly without special-casing).
		# Doesn't recurse into a FURTHER nested def/lambda's own body - that
		# one gets its own independent capture collection when IT gets
		# synthesized (only its OWN name, if it's a def, becomes a local
		# binding at THIS level, same as an ordinary assignment would); this
		# is also the reason a closure capturing another closure's own
		# capture isn't supported yet (see the plan's "Deferred" section) -
		# by the time an inner nested def/lambda's own free-variable walk
		# runs, THIS level's own rewrite has already turned any name it
		# captured into an Attribute expression, never a resolvable name.
		#
		# `x = x + 1` inside the body never reaches `free` at all: any
		# ast.Store anywhere in the body makes that name local for the
		# WHOLE body (mirroring real Python's own hoisting rule), so this
		# is also what makes captures strictly immutable snapshots (no
		# nonlocal write-back) - a captured-and-reassigned name is simply a
		# local read of its own not-yet-assigned local, not a capture,
		# enforced with no extra checking needed here.
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
			return []
		captures: list[tuple[str,Variable]] = []
		seen: set[str] = set()
		for free_name in free:
			if free_name.id in seen:
				continue
			resolved = enclosing_fn.names.get( free_name.id )
			if resolved is None:
				continue # genuinely undefined - falls through to the ordinary "not defined" error once the (rewritten) body is actually lowered
			if not isinstance( resolved, Variable ):
				self.lowering.discovery.fail(
					f"{ast.unparse(node)}: cannot capture {free_name.id!r} - only local variables and "
					f"parameters can be captured, not {type(resolved).__name__.lower()}s",
					node,
				)
			seen.add( free_name.id )
			captures.append( ( free_name.id, resolved ) )
		return captures

	def _rewrite_captures_into_env_reads( self, roots: list[ast.AST], env_cls: RCClass, erased_param: str, captured_names: set[str] ) -> list[ast.AST]:
		''' replaces every captured-name ast.Name(Load) reference inside
		`roots` with an inline, never-named env-field read:
		compiler.cast(<env>, erased_param).name - mirrors
		_get_or_create_closure_trampoline's own inline compiler.cast(
		<closure_owner>, erased_self) receiver expression, for the
		identical reason: binding the cast to a named local first would
		make _is_aliasing_expr treat it as a fresh, owned value needing its
		own scope-exit decref, over-releasing the env object this is only
		ever a BORROWED reinterpretation of (the closure's own `self`
		field already owns it - see _construct_capturing_closure). Doesn't
		recurse into a FURTHER nested def/lambda's own body - same posture
		_collect_free_variables's own walk takes, for the same reason (see
		its own comment on why multi-level capture-of-a-capture isn't
		supported yet). Mutates/replaces in place - each occurrence's own
		AST is synthesized and lowered exactly once, never reused for
		anything else afterward, so there's no aliasing hazard in doing so. '''
		def env_field_read( name: str, line: int, col: int ) -> ast.Attribute:
			env_type_ref = ast.Name( id = '<closure_env>', ctx = ast.Load(), lineno = line, col_offset = col )
			env_type_ref.resolved_type = env_cls
			cast_call = ast.Call(
				func = ast.Attribute(
					value = ast.Name( id = 'compiler', ctx = ast.Load(), lineno = line, col_offset = col ),
					attr = 'cast', ctx = ast.Load(), lineno = line, col_offset = col,
				),
				args = [ env_type_ref, ast.Name( id = erased_param, ctx = ast.Load(), lineno = line, col_offset = col ) ],
				keywords = [], lineno = line, col_offset = col,
			)
			return ast.Attribute( value = cast_call, attr = name, ctx = ast.Load(), lineno = line, col_offset = col )

		class _CaptureRewriter( ast.NodeTransformer ):
			def visit_FunctionDef( self, fd: ast.FunctionDef ) -> ast.FunctionDef:
				return fd # don't recurse into a further nested def's own body
			def visit_Lambda( self, lam: ast.Lambda ) -> ast.Lambda:
				return lam # ditto for a further nested lambda
			def visit_Name( self, n: ast.Name ) -> ast.AST:
				if isinstance( n.ctx, ast.Load ) and n.id in captured_names:
					return env_field_read( n.id, n.lineno, n.col_offset )
				return n

		rewriter = _CaptureRewriter()
		return [ rewriter.visit( root ) for root in roots ]

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
		if fn.broken:
			raise RedundantCompilationError() # already reported at the point fn's own resolution failed - see Name.broken
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

		captures = self._collect_free_variables( node.body, { p.stem for p in parameters }, node )

		if not captures:
			# the common, zero-cost case: nothing to capture, so `node.name`
			# resolves straight to a real Function - callers reach it via
			# _lower_function_ref (a bare reference) or ordinary Call
			# resolution, exactly as before this feature existed
			enclosing.add_name( node.name, synthetic )
			self.lowering.schedule( synthetic )
			return

		# a CAPTURING nested def - unlike the non-capturing case above, this
		# genuinely emits IR at the def statement's own position (this IS
		# "closure creation time" for a nested def, the same point Python
		# itself creates the function object each time the statement
		# executes - so one inside a loop correctly rebuilds a fresh env +
		# closure every iteration). `node.name` can no longer resolve to a
		# bare Function: calling it must route through _try_lower_closure_
		# call, which only matches a Variable of ClosureType - see the
		# closures plan
		ptr_cls = self.lowering.discovery.get_intrinsics()['Ptr']
		none_type = self.lowering.discovery.get_none_type()
		ptr_none_type = self.lowering.discovery._get_or_create_specialization( ptr_cls, [ none_type ] )

		env_cls = self.lowering._build_closure_env_class(
			[ ( name, resolved.type ) for name, resolved in captures ], f'{qualname}$$env', enclosing.file, node.lineno,
		)
		captured_names = { name for name, _ in captures }
		node.body = self._rewrite_captures_into_env_reads( node.body, env_cls, 'erased_env', captured_names )

		erased_env_param = Parameter( stem = 'erased_env', qualname = f'{qualname}.erased_env', file = enclosing.file, line = node.lineno, type = ptr_none_type )
		synthetic.parameters = [ erased_env_param, *parameters ]
		synthetic.add_name( 'erased_env', erased_env_param )

		operand = self._construct_capturing_closure( captures, env_cls, synthetic, node )

		# bind node.name as a real Variable (not a Function) - the same
		# "first assignment to a name with no prior declaration" tail
		# _stmt_Assign's own no-prior-declaration branch uses, minus
		# _declare_local's callback-based lowering (operand is already
		# lowered above). is_alias=False: operand is a fresh Allocate
		# result, same as any other first-time construction
		var = Variable( stem = node.name, qualname = f'{enclosing.qualname}.{node.name}', file = enclosing.file, line = node.lineno, type = operand.type )
		enclosing.add_name( var.stem, var )
		self.lowering.schedule( var.type )
		for instr in self._cfg_assign( var, operand, is_alias = False, node = node ):
			self._emit( instr )
		self._emit( ir.Assign( dest = var, src = operand ))

	def _expr_Lambda( self, node: ast.Lambda, expected_type: Type|None ) -> ir.Operand:
		# a lambda expression, capturing or not - see PLAN_LAMBDA.md/the
		# closures plan. Lambda syntax carries no type annotations at all,
		# so parameter types are inferred entirely from expected_type -
		# either a Ptr[Callable[[ArgTypes],Ret]] shape (the non-capturing
		# case's own established route - e.g. a `key: Callable[[T],K]`
		# parameter's own declared type) or, now, a bare ClosureType (a
		# capturing lambda's own actual result type - `c: Closure[[Args],
		# Ret] = lambda ...: ...`, the natural way to write one). ClosureType
		# already exposes the identical arg_types/return_type shape
		# CallableType does (see its own docstring), so no wrapper object is
		# needed - just falling back to expected_type itself when it's
		# already the right shape. Deliberately NOT folded into
		# TypeResolver._callable_type_of itself - that function's other
		# callers (_try_lower_indirect_call in particular) mean specifically
		# "a bare function-pointer value", not "anything callable"
		fn_type = self.lowering._type_resolver._callable_type_of( expected_type )
		if fn_type is None and isinstance( expected_type, ClosureType ):
			fn_type = expected_type
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

		captures = self._collect_free_variables( [ node.body ], { p.arg for p in positional }, node )

		env_cls: RCClass|None = None
		if captures:
			# a capturing lambda - build the env class and rewrite the body
			# BEFORE any eager lowering below, so a still-unbound return
			# type is inferred from the ALREADY-REWRITTEN (fully closed, no
			# free names left) body - see the closures plan's own note on
			# why this ordering is one-directional
			ptr_cls = self.lowering.discovery.get_intrinsics()['Ptr']
			none_type = self.lowering.discovery.get_none_type()
			ptr_none_type = self.lowering.discovery._get_or_create_specialization( ptr_cls, [ none_type ] )

			env_cls = self.lowering._build_closure_env_class(
				[ ( cap_name, resolved.type ) for cap_name, resolved in captures ], f'{qualname}$$env', enclosing.file, node.lineno,
			)
			captured_names = { cap_name for cap_name, _ in captures }
			synthetic_node.body = self._rewrite_captures_into_env_reads( synthetic_node.body, env_cls, 'erased_env', captured_names )

			erased_env_param = Parameter( stem = 'erased_env', qualname = f'{qualname}.erased_env', file = enclosing.file, line = node.lineno, type = ptr_none_type )
			synthetic.parameters = [ erased_env_param, *parameters ]
			synthetic.add_name( 'erased_env', erased_env_param )

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
			# save/restore around at all. Post-rewrite (if capturing), the
			# body is already fully closed (every capture is now an
			# ordinary attribute-chain expression rooted at erased_env) -
			# eager lowering genuinely cannot tell a capturing lambda from a
			# hand-written one at this point, so nothing here needs to change
			lowered = self.lowering._compile_now( synthetic )
			return_instr = next( instr for instr in lowered.instructions if isinstance( instr, ir.Return ))
			synthetic.return_type = (
				return_instr.value.type if return_instr.value is not None
				else self.lowering.discovery.get_none_type()
			)
		elif not captures:
			self.lowering.schedule( synthetic )

		if captures:
			# _construct_capturing_closure schedules `synthetic` itself
			# (see its own comment) - not done separately here
			return self._construct_capturing_closure( captures, env_cls, synthetic, node, expected_type )
		return self.lowering._function_ref_operand( synthetic )

	def _expr_Constant( self, node: ast.Constant, expected_type: Type|None ) -> ir.Operand:
		# ArrType[N] field's own supported literal (see FixedArrayType's own
		# docstring): a bare `0` means "zero-fill the whole array" - the one
		# value this construction path knows how to emit (a C11 `{0}`
		# designated-initializer payload, valid only inside the class-body
		# compound-literal construction shape - see emitter_c.py's
		# _emit_const). Handled first/separately since none of the ordinary
		# scalar-literal validation below (int range checks, CEnum duality,
		# ...) applies to this type at all.
		if isinstance( expected_type, FixedArrayType ):
			if type( node.value ) is not int or node.value != 0:
				self.lowering.discovery.fail(
					f'{ast.unparse(node)}: {expected_type.qualname} only supports a 0 (zero-fill) literal here - '
					f'per-element array construction is not implemented',
					node,
				)
			return ir.Const( type = expected_type, value = 0 )
		# a floating-point literal can only be typed as a float. If context
		# hints it toward a non-float scalar (an integer), that's a silent-
		# truncation trap - reject it. Catches `i + 1.5` (the literal hinted to
		# the int operand's type by _lower_binary_operands), `x: i32 = 1.5`, and
		# `i32(1.5)`. A float target (f32/f64), no expected type (defaults f64),
		# and a union expected type all fall through unaffected.
		if isinstance( node.value, float ) and isinstance( expected_type, Scalar ) and not _is_float_scalar( expected_type ):
			self.lowering.discovery.fail(
				f'a floating-point literal cannot be used where {expected_type.qualname} is expected - '
				f'write an explicit cast (e.g. {expected_type.stem}(...)) or use an integer literal: {ast.unparse(node)}',
				node,
			)
		# a literal being lowered against a CONCRETE, non-union expected type -
		# verify the literal's own Python value kind could plausibly represent
		# it at all (the same coarse stem-compatibility _LITERAL_COMPATIBLE_
		# STEMS already uses for overload-argument matching below). Without
		# this, expected_type was blindly trusted for anything other than the
		# None/TaggedUnion cases just below - `return 5` inside a function
		# declared -> str tagged the resulting Const as type=str while its own
		# .value stayed the Python int 5 (invalid/nonsensical downstream:
		# emitter_c.py's _emit_const dispatches purely on c.value's own Python
		# type, producing a bare `5` where a struct builtins$str* was
		# expected - confirmed compiling with zero errors, a real C compiler
		# then rejecting the mismatched return type outright). A union
		# expected_type (possibly Specialization-wrapped, e.g. Result[T,E])
		# is exempted here for the same reason the block below exempts it -
		# a literal is never itself union-shaped, _lower_expr's own post-hoc
		# coercion handles that case separately. A POINTER expected_type
		# (Ptr[T]/ConstPtr[T]) is also exempted - a bare int literal assigned
		# to one is a deliberate, pre-existing, well-defined bit-
		# reinterpretation idiom real WinAPI constants rely on (e.g.
		# lib/windows/kernel32.py's `INVALID_HANDLE_VALUE: HANDLE = -1`),
		# already handled downstream by emitter_c.py's own _emit_const
		# pointer-const branch - not a type error. A bare, still-unbound
		# TypeVar (a generic class's own T, not yet substituted with a
		# concrete type - e.g. a field=value construction argument for a
		# still-generic Box[T]) is exempted too: there's no real type here
		# yet to validate against at all, this literal's own natural type
		# is what will eventually get UNIFIED to solve T, not compared
		# against it. A CEnum expected_type validates against its OWN
		# underlying scalar's stem instead of being exempted outright
		# (cenum_value_type below) - "a CEnum has exactly the same
		# runtime representation as its underlying type" (EnumName(42)'s
		# own construction call, see _try_lower_construct_call's CEnum
		# branch, is exactly what this is for), but that's only true
		# when the literal is actually compatible with the underlying
		# SCALAR - a raw int literal for an i32-backed CEnum, not
		# literally anything. Unconditionally exempting every CEnum
		# expected_type from validation here (the original code) let a
		# kind-mismatched literal (e.g. a string) sail through
		# unchecked, tagging the resulting Const with the CEnum type
		# while its own .value stayed the mismatched Python value -
		# confirmed to crash emitter_c.py's _emit_const with an
		# uncaught Python NotImplementedError instead of a clean
		# CompileError (PLAN_COMPILER_BUG_SWEEP.md)
		cenum_value_type = expected_type.value_type if isinstance( expected_type, CEnum ) else None
		expected_base = expected_type.base if isinstance( expected_type, Specialization ) else expected_type
		if (
			expected_type is not None and not isinstance( expected_base, TaggedUnion )
			and not isinstance( expected_type, TypeVar )
			and not self.lowering._type_resolver._is_ptr_specialization( expected_type )
			# PLAN_GENERATORS.md Phase 5 (roadmap Phase 5) - a compiler-
			# synthesized `0` standing in for "this RC-typed generator
			# field isn't assigned yet" (type_resolver.py's
			# _build_generator_backing_class/_rewrite_generator_
			# constructor - a live-flag-gated field, never read before
			# its own first real assignment, so the actual zero bit
			# pattern here is never observed by anything but the
			# generator's own state/flag-gated destructor deciding NOT
			# to decref it). Deliberately NOT a general "int literal into
			# any RCClass" language relaxation (unlike the pre-existing
			# Ptr[T]/ConstPtr[T] exemption just above, which IS meant for
			# ordinary user code) - gated on this compiler-internal tag
			# only, exactly like resolved_type/resolved_callee elsewhere
			# in this codebase, so ordinary user code still can't write
			# `b: Box = 0` as a novel "null RCClass" idiom
			and not getattr( node, 'generator_zero_rc_field', False )
		):
			compatible_stems = self.lowering._LITERAL_COMPATIBLE_STEMS.get( type( node.value ) )
			expected_stem = cenum_value_type.stem if cenum_value_type is not None else getattr( expected_type, 'stem', None )
			# an int literal implicitly widening into a float scalar (`x: f64
			# = 1`, `f64(1)`) is a pre-existing, legitimate pattern - NOT
			# folded into _LITERAL_COMPATIBLE_STEMS[int] itself, since that
			# dict is shared with _lower_overload_arg's own OVERLOAD
			# resolution, where allowing int literals to also match a float
			# overload would introduce new ambiguity there. The reverse
			# (float literal -> int scalar) stays rejected by the dedicated
			# guard at the top of this method - this is one-directional
			is_int_into_float = type( node.value ) is int and expected_stem in ( 'f32', 'f64' )
			# a None literal is deliberately NOT a key in _LITERAL_COMPATIBLE_
			# STEMS at all - _lower_overload_arg's own identical dict lookup
			# relies on exactly that miss to skip straight to _lower_expr(expr,
			# None) for overload candidate matching (a None literal's "which
			# overload accepts it" question needs real per-candidate nullable-
			# type matching, not this coarse stem-list check) - adding a
			# NoneType entry there would silently break that. Validated here
			# instead, separately: only NoneType itself is a legitimate non-
			# pointer target (Ptr[T]/ConstPtr[T] is already exempted above -
			# the real, common nullable-pointer idiom, e.g. `p: Ptr[u8] = None`)
			if node.value is None and expected_stem != 'NoneType':
				self.lowering.discovery.fail(
					f'{ast.unparse(node)}: None cannot be used where {expected_type.qualname} is expected',
					node,
				)
			if compatible_stems is not None and expected_stem not in compatible_stems and not is_int_into_float:
				kind = type( node.value ).__name__
				article = 'an' if kind[0] in 'aeiou' else 'a'
				self.lowering.discovery.fail(
					f'{ast.unparse(node)}: {article} {kind} literal cannot be used where '
					f'{expected_type.qualname} is expected',
					node,
				)
			# a PLAIN literal (not the argument of an explicit T(...)/
			# compiler.cast(T,...) - see _lower_scalar_cast's own
			# _allow_literal_bit_reinterpret handling, which exempts THAT
			# case as deliberate bit-reinterpretation) flowing into a
			# concrete integer scalar type must actually FIT that type's
			# real range - compatible_stems above only checked the
			# literal's KIND (int vs float/str/...), never its magnitude
			if (
				not self._allow_literal_bit_reinterpret and type( node.value ) is int
				and ( isinstance( expected_type, Scalar ) or cenum_value_type is not None )
				and expected_stem in self.lowering._LITERAL_COMPATIBLE_STEMS[int]
			):
				lo, hi = int_stem_range( cenum_value_type if cenum_value_type is not None else expected_type )
				if not ( lo <= node.value <= hi ):
					self.lowering.discovery.fail(
						f'{node.value} is out of range for {expected_type.qualname} ({lo}..{hi}): {ast.unparse(node)}',
						node,
					)
		# expected_type being a TaggedUnion (e.g. str|None) is treated the
		# same as no expected_type at all: a literal's OWN Python type
		# always determines its natural type (bool/i32/str/NoneType) -
		# blindly typing the Const as the whole union here would be wrong
		# (a literal is never itself union-shaped at the C level), and
		# _lower_expr's own post-hoc coercion (see its comment) is what
		# actually wraps this natural-typed Const into the union afterward.
		# A bare, still-unbound TypeVar is treated the same way, matching
		# the validation exemption above - the literal's own natural type
		# is what UNIFIES to solve T, so tagging the Const with the bare
		# TypeVar itself (leaving it unsubstituted downstream) is wrong.
		if expected_type is None or isinstance( expected_type, TaggedUnion ) or isinstance( expected_type, TypeVar ):
			if isinstance( node.value, bool ):
				expected_type = self.lowering.discovery.get_intrinsics()['bool']
			elif isinstance( node.value, float ):
				# float literals default to f64 when no contextual type is
				# available (bare `x = 3.14`) - matches Python, whose float is
				# 64-bit. Checked before int is unnecessary (float/int are
				# disjoint, unlike bool/int) but placed here for clarity. A
				# hinted literal (`y: f32 = 1.5`) never reaches here with
				# expected_type None, so this is purely the no-context default
				expected_type = self.lowering.discovery.get_intrinsics()['f64']
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
			elif isinstance( node.value, bytes ):
				expected_type = self.lowering.discovery.find_name_or_none( 'bytes' )
				if expected_type is None:
					self.lowering.discovery.fail(
						f'cannot infer the type of literal {node.value!r} - no bytes type available ({ast.unparse(node)})',
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

	def _lower_method_call( self, receiver: ir.Operand, method_name: str, args: list[ir.Operand], result_type: Type, node: ast.AST ) -> ir.Operand:
		# shared _find_method + resolve/schedule + emit Call boilerplate -
		# every f-string dunder-dispatch/format-spec call site below uses
		# this same shape (receiver already lowered, method looked up by
		# plain name). str/int/f32/f64 are never generic, so this comment
		# used to end there - but Ptr[T]/ConstPtr[T]'s own __str__/__repr__
		# (lib/builtins/__ptr_arith.py) ARE bare generic Functions with an
		# unbound type param T (same registration shape as their __add__/
		# __sub__/comparison dunders - see _resolve_receiver_generic_dunder's
		# own docstring), so _find_method alone isn't enough here anymore:
		# without also resolving T from the receiver's own concrete pointee
		# type, `method` still carries the bare TypeVar, and emitter_c.py's
		# c_type crashes on it at prototype-emission time (confirmed via a
		# real repro building f'{some_ptr}'/some_ptr.__str__()) - same fix
		# _find_dunder_for_arg's own tail already applies for operator-
		# dispatched Ptr dunders, just needed here too for this SEPARATE,
		# plain-method-name dispatch path.
		method = self.lowering._resolve_receiver_generic_dunder(
			self.lowering._find_method( receiver.type, method_name ), receiver.type,
		)
		if method is None:
			type_name = receiver.type.qualname if receiver.type is not None else '?'
			self.lowering.discovery.fail( f'f-string requires {type_name}.{method_name}() to be available: {ast.unparse(node)}', node )
		self.lowering._ensure_resolved( method )
		self.lowering.schedule( method.return_type )
		for p in ( method.parameters or [] ):
			self.lowering.schedule( p.type )
		dest = self._new_temp( result_type )
		if method.cls is None:
			# a Scalar-registered method (`SomeScalar.method = some_free_
			# function` - discovery.py's visit_Assign, e.g. this file's own
			# float format-spec dispatch onto f64._sign_prefix/_fixed_digits,
			# lib/builtins/__float.py) is a genuine free Function, unlike a
			# real CStruct/RCClass method - discovery never strips a "self"
			# off its .parameters the way _make_function_resolver does for
			# an actual class body (there IS no class body here), so
			# emitter_c.py's _emit_call_args (which walks target.parameters
			# assuming it already excludes the receiver) would double-count
			# the receiver against the first declared parameter otherwise -
			# confirmed by a real KeyError crash while wiring this up.
			# ir.Call's own receiver field is for real bound-method calls
			# only; a free function just takes the receiver as an ordinary
			# leading positional argument instead.
			self._emit( ir.Call( dest = dest, target = method, receiver = None, args = [ receiver ] + args, kwargs = {} ))
		else:
			self._emit( ir.Call( dest = dest, target = method, receiver = receiver, args = args, kwargs = {} ))
		return dest

	def _const_usize( self, value: int ) -> ir.Const:
		return ir.Const( type = self.lowering.discovery.get_intrinsics()['usize'], value = value )

	def _const_i32( self, value: int ) -> ir.Const:
		return ir.Const( type = self.lowering.discovery.get_intrinsics()['i32'], value = value )

	def _const_bool( self, value: bool ) -> ir.Const:
		return ir.Const( type = self.lowering.discovery.get_intrinsics()['bool'], value = value )

	def _is_literal_format_spec( self, format_spec: ast.JoinedStr ) -> bool:
		return all( isinstance( v, ast.Constant ) for v in format_spec.values )

	def _literal_format_spec_text( self, format_spec: ast.JoinedStr ) -> str:
		return ''.join( v.value for v in format_spec.values ) # each v is ast.Constant(str) - _is_literal_format_spec already confirmed this

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
		parsed_spec = None
		if node.format_spec is not None:
			if not self._is_literal_format_spec( node.format_spec ):
				self.lowering.discovery.fail(
					f'f-string format specs must be a literal string for now - dynamic format specs are not supported yet: {ast.unparse(node)}',
					node,
				)
			try:
				parsed_spec = parse_format_spec( self._literal_format_spec_text( node.format_spec ))
			except FormatSpecError as e:
				self.lowering.discovery.fail( f'{e} ({ast.unparse(node)})', node )

		operand = self._lower_expr( node.value, None )

		if node.conversion == -1 and parsed_spec is not None:
			# no explicit !conversion - the format spec dispatches against
			# the value's OWN type directly (int's own radix/width/sign
			# handling, e.g.), matching Python's own format(x, spec) ==
			# type(x).__format__(x, spec) - as opposed to format(str(x),
			# spec) or format(repr(x), spec), which is what an EXPLICIT
			# !s/!r/!a conversion means instead (handled below)
			return self._lower_dispatch_format_spec( operand, parsed_spec, str_type, node )

		# conversion 114 == '!r' or 97 == '!a' (ascii wants a repr-shaped
		# text, ascii-escaped below via _lower_ascii_escape - see its own
		# comment on why this does NOT add surrounding quotes, unlike
		# Python's real ascii(): str has no __repr__() of its own here for
		# !a to match the quoting behavior of either, so !a just escapes
		# whatever !r's own resolution already produces) both want
		# __repr__; -1 (none) and 115 ('!s') want __str__ - just an ordinary
		# method lookup, same as any other type: every fixed-width int
		# scalar (i8/u8/.../isize/usize) has a real __str__/__repr__
		# (lib/builtins/__scalar_dunders.py's i_str_signed/i_str_unsigned),
		# same mechanism f64/f32's own __str__/__repr__ use (__float.py) -
		# a scalar WITHOUT one (bool, currently) still fails cleanly here
		# with a plain "method not found" error rather than being auto-boxed
		if operand.type is str_type:
			value_as_str = operand
		else:
			method_name = '__repr__' if node.conversion in ( 114, 97 ) else '__str__'
			value_as_str = self._lower_method_call( operand, method_name, [], str_type, node )
		if node.conversion == 97:
			value_as_str = self._lower_ascii_escape( value_as_str, str_type, node )

		if parsed_spec is not None:
			# an explicit !s/!r/!a conversion (or the operand's own type
			# needing __str__) already reduced the value to plain str - the
			# spec now formats THAT text (fill/align/width/precision-as-
			# truncation only, str's own branch below) rather than
			# dispatching against the original value's own type again
			return self._lower_dispatch_format_spec( value_as_str, parsed_spec, str_type, node )
		return value_as_str

	def _lower_ascii_escape( self, operand: ir.Operand, str_type: Type, node: ast.AST ) -> ir.Operand:
		# f-string !a conversion's second half - operand is already str-
		# typed (whatever the !r-equivalent resolution above produced);
		# this just calls str._ascii_escape() (lib/builtins/__init__.py)
		# on it.
		return self._lower_method_call( operand, '_ascii_escape', [], str_type, node )

	def _lower_dispatch_format_spec( self, operand: ir.Operand, spec: FStringFormatSpec, str_type: Type, node: ast.AST ) -> ir.Operand:
		if operand.type is str_type:
			return self._lower_str_format_spec( operand, spec, str_type, node )
		int_type = self.lowering.discovery.find_name_or_none( 'int' )
		if int_type is not None and operand.type is int_type:
			return self._lower_int_format_spec( operand, spec, str_type, node )
		intrinsics = self.lowering.discovery.get_intrinsics()
		if operand.type is intrinsics.get( 'f32' ) or operand.type is intrinsics.get( 'f64' ):
			return self._lower_float_format_spec( operand, spec, str_type, node )
		type_name = operand.type.qualname if operand.type is not None else '?'
		if spec.type in ( 'f', 'F', 'e', 'E', 'g', 'G', '%' ):
			self.lowering.discovery.fail(
				f"f-string format spec: {spec.type!r} needs a real float type with formatting support, which doesn't exist yet ({type_name}): {ast.unparse(node)}",
				node,
			)
		self.lowering.discovery.fail(
			f'f-string format spec: {type_name} does not support format specs yet (only str, int, and float do): {ast.unparse(node)}',
			node,
		)

	def _lower_pad_by_align( self, operand: ir.Operand, align: str, fill: str, width: int, str_type: Type, node: ast.AST ) -> ir.Operand:
		method_name = { '<': 'ljust', '>': 'rjust', '^': 'center' }[align]
		args = [ self._const_usize( width ), ir.Const( type = str_type, value = fill ) ]
		return self._lower_method_call( operand, method_name, args, str_type, node )

	def _lower_str_format_spec( self, operand: ir.Operand, spec: FStringFormatSpec, str_type: Type, node: ast.AST ) -> ir.Operand:
		try:
			validate_str_spec( spec )
		except FormatSpecError as e:
			self.lowering.discovery.fail( f'{e} ({ast.unparse(node)})', node )
		value = operand
		if spec.precision is not None:
			value = self._lower_method_call( value, '_truncate_codepoints', [ self._const_usize( spec.precision ) ], str_type, node )
		if spec.width is not None:
			value = self._lower_pad_by_align( value, spec.align or '<', spec.fill, spec.width, str_type, node ) # str's own default align is left, unlike numeric types' right
		return value

	_RADIX_BY_TYPE_CHAR = { 'b': 2, 'o': 8, 'x': 16, 'X': 16 }
	_RADIX_PREFIX_BY_TYPE_CHAR = { 'b': '0b', 'o': '0o', 'x': '0x', 'X': '0X' }

	def _lower_int_format_spec( self, operand: ir.Operand, spec: FStringFormatSpec, str_type: Type, node: ast.AST ) -> ir.Operand:
		try:
			validate_int_spec( spec )
		except FormatSpecError as e:
			self.lowering.discovery.fail( f'{e} ({ast.unparse(node)})', node )
		type_char = spec.type

		if type_char in ( 'b', 'o', 'x', 'X' ):
			base = self._RADIX_BY_TYPE_CHAR[type_char]
			uppercase = type_char == 'X'
			raw_digits = self._lower_method_call( operand, '_to_radix_digits', [ self._const_i32( base ), self._const_bool( uppercase ) ], str_type, node )
			prefix_text = self._RADIX_PREFIX_BY_TYPE_CHAR[type_char] if spec.alt else ''
			sep_text = '' # grouping is never valid for a radix type char (validate_int_spec)
		else:
			raw_digits = self._lower_method_call( operand, '_decimal_digits', [], str_type, node )
			prefix_text = ''
			sep_text = spec.grouping or ''
		sep = ir.Const( type = str_type, value = sep_text ) # '' still groups correctly - see str._insert_thousands_sep's own comment

		sign_char = self._lower_method_call( operand, '_sign_prefix', [ ir.Const( type = str_type, value = spec.sign ) ], str_type, node )
		if prefix_text:
			sign_and_prefix = self._lower_str_add( sign_char, ir.Const( type = str_type, value = prefix_text ), str_type, node )
		else:
			sign_and_prefix = sign_char

		if spec.width is not None and spec.align == '=':
			# the '0' shorthand - zero-padding goes BETWEEN sign/prefix and
			# digits, grouping-aware (str._pad_and_group_after_prefix - a
			# plain "group first, then _pad_after_prefix" two-step gives
			# the wrong answer once grouping is combined with zero-pad, see
			# its own comment) - needs the RAW, ungrouped digits, not the
			# _insert_thousands_sep'd ones the other two branches below want
			return self._lower_method_call(
				raw_digits, '_pad_and_group_after_prefix',
				[ sign_and_prefix, self._const_usize( spec.width ), ir.Const( type = str_type, value = spec.fill ), sep ],
				str_type, node,
			)
		digits = self._lower_method_call( raw_digits, '_insert_thousands_sep', [ sep ], str_type, node )
		if spec.width is None:
			return self._lower_str_add( sign_and_prefix, digits, str_type, node )
		body = self._lower_str_add( sign_and_prefix, digits, str_type, node )
		return self._lower_pad_by_align( body, spec.align or '>', spec.fill, spec.width, str_type, node ) # numeric types' own default align is right, unlike str's left

	def _lower_float_format_spec( self, operand: ir.Operand, spec: FStringFormatSpec, str_type: Type, node: ast.AST ) -> ir.Operand:
		# 'f'/'F'/'e'/'E'/'g'/'G'/'%' (PLAN_STR_FORMAT.md item 4 - every
		# float type char fstring_format_spec.FORMAT_SPEC_TYPE_CHARS
		# recognizes). Same sign+digits+pad assembly shape as
		# _lower_int_format_spec above (no radix/grouping prefix to worry
		# about here, so it's simpler), calling into lib/builtins/
		# __float.py's own _sign_prefix/_fixed_digits/_percent_digits
		# methods - real control flow lives there, not hand-built IR here,
		# matching int's own _sign_prefix/_to_radix_digits split.
		try:
			validate_float_spec( spec )
		except FormatSpecError as e:
			self.lowering.discovery.fail( f'{e} ({ast.unparse(node)})', node )
		precision = spec.precision if spec.precision is not None else 6 # Python's own f"{x:f}"/f"{x:e}"/f"{x:g}"/f"{x:%}" all share this default
		alt = self._const_bool( spec.alt )
		sep = ir.Const( type = str_type, value = spec.grouping or '' ) # '' still groups correctly - see str._insert_thousands_sep's own comment
		is_percent = spec.type == '%'
		# None type char WITH an explicit precision behaves like 'g' (plus
		# its own "always show a fractional digit in fixed form" tweak) -
		# real Python's own "None" presentation, not plain 'f' (see
		# validate_float_spec's own comment and lib/builtins/__float.py's
		# _none_type_digits_raw). None type char with NO precision either
		# (f"{x:10}") needs Python's real shortest-round-trip repr
		# algorithm instead - _repr_digits/_repr_digits_raw (lib/builtins/
		# __float.py), the same machinery bare f"{x}" uses via __str__/
		# __repr__ (_lower_fstring_part's own dispatch, unrelated to this
		# function - reached before a format spec is even considered).
		is_none_type_with_precision = spec.type is None and spec.precision is not None and not is_percent
		is_none_type_no_precision = spec.type is None and spec.precision is None and not is_percent
		type_char = (
			self._const_i32( ord( spec.type or 'f' ) )
			if not is_percent and not is_none_type_with_precision and not is_none_type_no_precision
			else None
		)
		sign_char = self._lower_method_call( operand, '_sign_prefix', [ ir.Const( type = str_type, value = spec.sign ) ], str_type, node )

		if is_percent:
			digits_method, digits_args = '_percent_digits', [ self._const_usize( precision ), alt ]
		elif is_none_type_with_precision:
			digits_method, digits_args = '_none_type_digits', [ self._const_usize( precision ), alt ]
		elif is_none_type_no_precision:
			digits_method, digits_args = '_repr_digits', []
		else:
			digits_method, digits_args = '_fixed_digits', [ self._const_usize( precision ), type_char, alt ]

		if spec.width is not None and spec.align == '=':
			# the '0' shorthand - zero-padding goes BETWEEN sign and
			# digits, grouping-aware AND special-value-aware (str._pad_
			# maybe_special - a plain "group first, then _pad_after_prefix"
			# two-step gives the wrong answer once grouping is combined
			# with zero-pad, and "nan"/"inf" text needs to skip grouping
			# entirely even when requested - see str._pad_and_group_after_
			# prefix's own comment and _pad_maybe_special's own comment) -
			# needs the RAW, ungrouped digits (the '_raw' variant of
			# whichever digits_method was picked above), not the already-
			# grouped ones the other branch below wants
			raw = self._lower_method_call( operand, digits_method + '_raw', digits_args, str_type, node )
			if is_percent:
				# str._pad_maybe_special has no notion of '%' - reserve 1
				# char of the nominal width for it here, then append it
				# after, the same "caller reserves room for what this
				# method doesn't know about" convention _pad_and_group_
				# before_dot's own comment documents
				inner_width = max( spec.width - 1, 0 )
				padded = self._lower_method_call(
					raw, '_pad_maybe_special',
					[ sign_char, self._const_usize( inner_width ), ir.Const( type = str_type, value = spec.fill ), sep ],
					str_type, node,
				)
				return self._lower_str_add( padded, ir.Const( type = str_type, value = '%' ), str_type, node )
			return self._lower_method_call(
				raw, '_pad_maybe_special',
				[ sign_char, self._const_usize( spec.width ), ir.Const( type = str_type, value = spec.fill ), sep ],
				str_type, node,
			)

		digits = self._lower_method_call( operand, digits_method, digits_args + [ sep ], str_type, node )
		if spec.width is None:
			return self._lower_str_add( sign_char, digits, str_type, node )
		body = self._lower_str_add( sign_char, digits, str_type, node )
		return self._lower_pad_by_align( body, spec.align or '>', spec.fill, spec.width, str_type, node ) # numeric types' own default align is right, unlike str's left

	def _lower_str_add( self, left: ir.Operand, right: ir.Operand, str_type: Type, node: ast.AST ) -> ir.Operand:
		return self._lower_method_call( left, '__add__', [ right ], str_type, node )

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
		unwrap = concrete_result_cls.get_local_or_raise( 'unwrap' )
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

		init = concrete_cls.get_local_or_raise( '__init__' )
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
		append = concrete_cls.get_local_or_raise( 'append' )
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
		get_ptr = concrete_cls.get_local_or_raise( 'get_ptr' )
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
		# is_bound_method_closure tags this SPECIFIC node so
		# _is_aliasing_expr can tell it apart from an ordinary field read
		# whose declared type just happens to be ClosureType (e.g.
		# `self.data.v_Ok` for a Result[Closure[...],E]) - see
		# _is_aliasing_expr's own comment on why the operand's type alone
		# isn't enough
		node.is_bound_method_closure = True
		self.lowering._ensure_resolved( method )
		if method.broken:
			raise RedundantCompilationError() # already reported at the point method's own resolution failed - see Name.broken
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

	def _construct_capturing_closure(
		self, captures: list[tuple[str,Variable]], env_cls: RCClass, synthetic: Function,
		node: ast.AST, expected_type: Type|None = None,
	) -> ir.Operand:
		''' builds a real, capturing Closure[[Args],Ret] value - the general
		case of _lower_bound_method_closure just above, generalized from one
		erased receiver field to N real captured fields collapsed behind one
		erased env pointer. Shared by _stmt_FunctionDef (a capturing nested
		def) and _expr_Lambda (a capturing lambda) - see the closures plan.
		`synthetic` IS the trampoline here (unlike the bound-method case's
		separate, shared, memoized-per-(method,owner) trampoline function) -
		a lambda/nested-def's own synthesized body is never called any other
		way than through its own closure's fn field, and never shared across
		construction sites, so there's no benefit to a second indirection
		layer; its own first parameter is already `erased_env: Ptr[None]`. '''
		arg_types = [ p.type for p in synthetic.parameters[1:] ] # skip erased_env
		self.lowering.schedule( synthetic.return_type )
		for t in arg_types:
			self.lowering.schedule( t )

		closure_type = self.lowering.discovery._get_or_create_closure_type( arg_types, synthetic.return_type )
		self.lowering._ensure_resolved( closure_type )
		self.lowering._schedule_rcclass_construction( closure_type, closure_type )
		self.lowering._schedule_rcclass_construction( env_cls, env_cls ) # the env is a real, constructed RCClass too - needs its own sys.alloc/__del__ scheduled, same as any other constructed class

		# each capture's CURRENT value, read in the ENCLOSING function's own
		# scope (an ordinary ast.Name read) - field_value() is the same
		# generic per-field embedding every ordinary SomeClass(field=value)
		# construction already uses (cfg.py, shared by _lower_allocate_fields):
		# an aliasing Name read gets exactly one Incref if its type is RC,
		# nothing otherwise - no hand-rolled ir.Incref needed here, unlike
		# the bound-method case above, precisely BECAUSE these fields keep
		# their real declared types instead of erasing to Ptr[None]
		fields: dict[str,ir.Operand] = {}
		for name, resolved in captures:
			name_node = ast.Name( id = name, ctx = ast.Load(), lineno = node.lineno, col_offset = node.col_offset )
			value = self._lower_expr( name_node, resolved.type )
			is_alias = self.lowering._is_aliasing_expr( name_node, value )
			for instr in self._cfg.field_value( value.type, value, is_alias = is_alias ):
				self._emit( instr )
			fields[name] = value

		env_dest = self._new_temp( env_cls )
		self._emit( ir.Allocate( dest = env_dest, cls = env_cls, fields = fields ))

		ptr_cls = self.lowering.discovery.get_intrinsics()['Ptr']
		none_type = self.lowering.discovery.get_none_type()
		ptr_none_type = self.lowering.discovery._get_or_create_specialization( ptr_cls, [ none_type ] )

		env_erased = self._new_temp( ptr_none_type )
		self._emit( ir.CastWrap( dest = env_erased, operand = env_dest ))
		# env_dest is a fresh ir.Allocate result, so it's fresh_temp()-
		# tracked as a pending obligation for THIS statement's own cleanup
		# (see _emit) - erasing it via CastWrap doesn't transfer that
		# tracking (a CastWrap's own dest is never fresh_temp()-registered,
		# but its OPERAND's existing tracking is untouched), so without this
		# the per-statement pending-temp flush would decref env_dest right
		# out from under the closure that's about to become its only real
		# owner - a real, confirmed premature free (heap corruption at
		# runtime, not just reasoning). untrack_temp mirrors exactly what
		# cfg.field_value's own is_alias=False branch already does for any
		# other fresh value handed into a new field - ownership transfers
		# into the closure's own `self` field, so the original temp needs
		# no independent decref of its own
		self._cfg.untrack_temp( env_dest )

		trampoline_callable_type = self.lowering.discovery._get_or_create_callable_type(
			[ p.type for p in synthetic.parameters ], synthetic.return_type,
		)
		trampoline_ptr_type = self.lowering.discovery._get_or_create_specialization( ptr_cls, [ trampoline_callable_type ] )
		fn_ref = ir.FunctionRef( type = trampoline_ptr_type, fn = synthetic )
		fn_erased = self._new_temp( ptr_none_type )
		self._emit( ir.CastWrap( dest = fn_erased, operand = fn_ref ))
		# unlike the bound-method trampoline (scheduled inside _get_or_create_
		# closure_trampoline, which BUILDS it), `synthetic` here is built by
		# the caller (_stmt_FunctionDef/_expr_Lambda) - this is the one place
		# both paths funnel through, so scheduling it here (not at either
		# call site) is what actually makes it a real, emitted function
		self.lowering.schedule( synthetic )

		dest = self._new_temp( expected_type or closure_type )
		self._emit( ir.Allocate( dest = dest, cls = closure_type, fields = { 'fn': fn_erased, 'self': env_erased } ))
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
					# tag the Const with the CEnum's own nominal type, not its
					# underlying scalar, whenever the surrounding context already
					# expects exactly that CEnum (e.g. a generic type param
					# already bound to it by the enclosing return-type context -
					# see _unify_type_param, which has no CEnum<->value_type
					# exemption the way _check_assignable does at line ~4169/4171
					# and so would wrongly see this as a conflicting inference).
					# Falls back to the scalar for every other context (None, the
					# raw value_type itself, an unrelated/unbound TypeVar) -
					# _check_assignable's own bidirectional CEnum<->value_type
					# exemption already makes both spellings interchangeable
					# there, so this only changes behavior where the exemption
					# doesn't already exist.
					const_type = obj if expected_type is obj else obj.value_type
					return ir.Const( type = const_type, value = value )
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
		if method is None:
			# _find_method itself can't distinguish "no such attribute at
			# all" from "found a real Overload group, silently discarded
			# it" - see its own isinstance(found, Function) check. Re-probe
			# here (identical chain_lookup/names shape) specifically to
			# give a clear diagnostic for the latter case, which
			# _attr_lookup below would otherwise ALSO reject, but with the
			# misleading "no such attribute" message - an overloaded
			# method genuinely has nowhere to bind as a bare value (no
			# single fixed signature to close over), same restriction the
			# comment above already states for generic/overloaded methods;
			# this just reports it accurately instead of via the wrong
			# error text. Purely diagnostic - both paths were already a
			# compile error either way, nothing here changes what compiles
			resolved_owner = self.lowering._ensure_resolved( obj.type )
			if isinstance( resolved_owner, ( CStruct, RCClass )):
				maybe_overload = resolved_owner.chain_lookup( node.attr )
			else:
				owner_names = getattr( resolved_owner, 'names', None )
				maybe_overload = owner_names.get( node.attr ) if isinstance( owner_names, dict ) else None
			if isinstance( maybe_overload, Overload ):
				self.lowering.discovery.fail(
					f'{node.attr!r} is an overloaded method and cannot be referenced as a value - call it directly instead: {ast.unparse(node)}',
					node,
				)
		if (
			isinstance( method, Function ) and method.cls is not None
			and not method.is_static and not method.is_classmethod
			and not method.type_params and not method.is_overload
		):
			if method.is_property:
				# @property - `obj.attr` (no call parens) means "call this
				# zero-arg getter and use its result", not "bind a callable
				# closure to it" (the ordinary method-as-value meaning just
				# below, which a property never uses - there is no coherent
				# "callable referring to this property" the way there is for
				# an ordinary method). _ensure_resolved first - method.
				# return_type is still None until then (same reason the
				# dunder-dispatch BinOp/Compare paths above resolve their own
				# method before ever reading .return_type)
				self.lowering._ensure_resolved( method )
				return self._lower_method_call( obj, node.attr, [], expected_type or method.return_type, node )
			return self._lower_bound_method_closure( node, obj, method, expected_type )
		attr_var = self.lowering._attr_lookup( obj.type, node.attr, node )
		if isinstance( attr_var.type, FixedArrayType ):
			# a bare C array member isn't assignable via `=` at all (only a
			# whole containing struct/union is, or an explicit memcpy) - see
			# FixedArrayType's own docstring. Reading it out as an ordinary
			# value (`x = f.b`) would need ir.GetAttr's emission to do
			# something other than a plain `dest = (obj).field;` assignment,
			# which isn't implemented (no element-level access exists yet
			# either, the same gap this repo's own bytearray has today) -
			# rejected here with a clear message rather than silently
			# reaching emitter_c.py and producing invalid C.
			self.lowering.discovery.fail(
				f'{ast.unparse(node)}: {attr_var.type.qualname} fields cannot be read as a whole value yet '
				f'(no element-level array access is implemented)',
				node,
			)
		dest = self._new_temp( attr_var.type )
		self._emit( ir.GetAttr( dest = dest, obj = obj, attr = node.attr ))
		# a pointer-typed field passed into a differently-typed pointer parameter
		# (e.g. sys.memcpy( ..., self.__metadata, ... ) where src is ConstPtr[u8])
		# needs the same CastWrap coercion _expr_Name does for bare locals.
		return self._maybe_castwrap_pointer( dest, expected_type )

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
		# per-element expected types, threaded down the same way
		# _lower_allocate_fields threads field.type into each field's own
		# _lower_expr call - without this, a leaf value destined for a
		# union-typed element (e.g. `bytes` into a declared
		# tuple[bytes|None, str|None]) never goes through
		# _coerce_or_check_operand's union-coercion, AND the tuple type
		# inferred below from the elements' own NATURAL (uncoerced) types
		# would differ from expected_type - two distinct backing RCClasses
		# for what's supposed to be one tuple type, with only the natural
		# one's allocator actually scheduled (_schedule_rcclass_
		# construction below) while dest ends up typed as the OTHER
		# (expected) one - exactly the "call to undeclared function"/
		# "assigning incompatible type" emitter bug this comment is here
		# to prevent regressing. Only applied when arity matches - a
		# genuine arity mismatch is a real type error better left to
		# whatever assignment/return-type check already reports it
		# clearly, not guessed at here.
		#
		# expected_type isn't always the bare TupleType itself - a generic
		# call's own type-param inference (monomorphize.py's substitute_
		# type_params) eagerly resolves a TupleType bound to a TypeVar into
		# its backing RCClass before handing it down as an expected-type
		# hint (Result.Ok((a, b)) against a declared Result[tuple[T1|None,
		# T2|None],E] return type reaches here with expected_type already
		# the tuple's backing RCClass, not the bare TupleType) - falling
		# back to tuple_type_for's reverse lookup recovers the original
		# elem_types (with their union members) in that case too, instead
		# of silently skipping per-element coercion and inferring the
		# tuple's own NATURAL (non-union) type, which then disagrees with
		# the outer expected type and fails generic inference.
		expected_tuple_type = (
			expected_type if isinstance( expected_type, TupleType )
			else self.lowering._tuple_storage.tuple_type_for( expected_type )
		)
		expected_elem_types: list[Type]|None = (
			expected_tuple_type.elem_types
			if expected_tuple_type is not None and len( expected_tuple_type.elem_types ) == len( node.elts )
			else None
		)
		operands: list[ir.Operand] = []
		for i, elt in enumerate( node.elts ):
			elem_expected = expected_elem_types[i] if expected_elem_types is not None else None
			value = self._lower_expr( elt, elem_expected )
			# same per-field RC-retain emission _lower_allocate_fields's own
			# field-value loop uses for every other class's field=value
			# construction sugar - a fresh value (Allocate/Call/Constant)
			# needs no extra incref, an aliasing read of an existing
			# binding (Name/Attribute) does, since the tuple now
			# independently owns a reference alongside whatever binding the
			# element came from
			for instr in self._cfg.field_value( value.type, value, is_alias = self.lowering._is_aliasing_expr( elt, value )):
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
		# swapped for backing_cls. Only trust it when it actually resolves to
		# THIS tuple's own backing class though - expected_type can just as
		# easily be an unrelated OUTER context (e.g. a `tuple[T,T]|None`
		# parameter's own union, handed down so _coerce_or_check_operand can
		# wrap the result into it afterward) rather than a description of the
		# tuple itself; interning guarantees identity in the genuine-match
		# case, so anything else must fall back to backing_cls, not be
		# trusted as dest's real type (a real bug: a tuple literal passed as
		# a `tuple[str,str]|None` argument used to set dest.type to the
		# UNION's own TaggedUnion, which isn't an RCClass, crashing emitter_c
		# .py's Allocate emission with `assert isinstance(concrete_cls,
		# RCClass)` since the union coercion never got a chance to run).
		resolved_expected = self.lowering._ensure_resolved( expected_type ) if expected_type is not None else None
		dest = self._new_temp( resolved_expected if resolved_expected is backing_cls else backing_cls )
		self._emit( ir.Allocate( dest = dest, cls = backing_cls, fields = fields ))
		return dest

	def _construct_generic_instance( self, target_cls: Type, node: ast.AST ) -> ir.Operand:
		''' construct a zero-argument instance of an already-fully-resolved
		class/generic Specialization (target_cls's own type args, if any,
		are already concrete) - used by _expr_List to build the backing
		list[T] instance a list-literal populates via append(). Deliberately
		narrower than _try_lower_construct_call (this file, the general
		ClassName(...) sugar): no fresh AST Call node naming the class is
		synthesized here (that would never have passed through type_
		resolver.py's own pre-pass the way a real call site does, and would
		need its own textual type-argument spelling for an arbitrary
		target_cls) - target_cls is already the concrete type we want, so
		this goes straight to ordinary (non-generic-inference) construction,
		using a synthetic zero-arg Call node purely as the argument-list
		shape _lower_call_args/_match_call_args need (never inspected for
		its own .func) - real default-value expressions (e.g. list[T]'s own
		initial_capacity: usize = 8) are already real AST nodes on the
		Function's own Parameter objects, nothing to fabricate there. Only
		supports a target whose __init__ is present, non-overloaded, and
		non-fallible - list[T]'s own shape; a different caller needing more
		would extend this, not work around it. '''
		resolved_cls = self.lowering._ensure_resolved( target_cls )
		assert isinstance( resolved_cls, ClassLike ), f'internal compiler error: {resolved_cls} is not constructible'
		init = resolved_cls.get_local_or_raise( '__init__' )
		assert isinstance( init, Function ), f'internal compiler error: {resolved_cls.qualname} has no usable __init__'
		self.lowering.schedule( resolved_cls )
		self.lowering._ensure_resolved( init )
		synth_call = ast.Call( func = node, args = [], keywords = [] )
		ast.copy_location( synth_call, node )
		args, kwargs = self._lower_call_args( init, synth_call )
		self_temp = self._new_temp( resolved_cls )
		self.lowering._schedule_rcclass_construction( resolved_cls, self_temp.type )
		self._emit( ir.Allocate( dest = self_temp, cls = resolved_cls, fields = {} ))
		self.lowering.schedule( init.return_type )
		for param in init.parameters or []:
			self.lowering.schedule( param.type )
		assert not self.lowering._init_fallibility( init ), f'internal compiler error: {resolved_cls.qualname}.__init__ is fallible'
		self._emit( ir.Call( dest = None, target = init, receiver = self_temp, args = args, kwargs = kwargs ))
		return self_temp

	def _expr_List( self, node: ast.List, expected_type: Type|None ) -> ir.Operand:
		''' [a, b, c] - requires expected_type to already be a concrete
		list[T] Specialization (inferring T from the elements themselves
		when no annotation/return-type is available is deferred - every
		real site in lib/ already has one, matching _expr_Tuple's own
		precedent of deferring an unforced generalization (arity 0/1)
		rather than guessing). Builds one list[T] instance via
		_construct_generic_instance, then a real append(elt).unwrap(...)
		method-call chain per element - list[T] has a real __init__/append,
		unlike tuple, so this can't reuse _expr_Tuple's single-ir.Allocate
		shape. A wrong-typed element is rejected the ordinary way by the
		_lower_expr(elt, elem_type) call below - the general assignability
		check already covers it, nothing extra needed here. '''
		# _as_specialization, not a bare isinstance(expected_type,
		# Specialization) check - expected_type may already have been
		# eagerly monomorphized to the real list[T] RCClass by
		# TypeResolver.resolve_declared_types (a function's own declared
		# return type, a variable's own annotation, ...) by the time this
		# runs, same duality _same_type exists to handle elsewhere -
		# without this, a perfectly valid list[T]-typed context wrongly
		# fails with "needs a known list[T] target type"
		spec = self.lowering._type_resolver._as_specialization( expected_type )
		resolved = self.lowering._ensure_resolved( expected_type ) if expected_type is not None else None
		if not ( spec is not None and isinstance( resolved, RCClass )
				and spec.base.stem == 'list' and len( spec.args ) == 1 ):
			self.lowering.discovery.fail(
				f'list literal needs a known list[T] target type from context (e.g. an annotation or return type): {ast.unparse(node)}',
				node,
			)
		elem_type = spec.args[0]
		dest = self._construct_generic_instance( expected_type, node )
		if not node.elts:
			return dest
		append_fn = self.lowering._find_method( dest.type, 'append' )
		assert append_fn is not None, 'internal compiler error: list[T] has no append method'
		self.lowering._ensure_resolved( append_fn )
		self.lowering.schedule( append_fn.return_type )
		unwrap_fn = self.lowering._find_method( append_fn.return_type, 'unwrap' )
		assert unwrap_fn is not None, 'internal compiler error: list[T].append does not return a Result with unwrap()'
		self.lowering._ensure_resolved( unwrap_fn )
		self.lowering.schedule( unwrap_fn.return_type )
		errmsg_node = ast.Constant( value = 'list literal: append failed' )
		ast.copy_location( errmsg_node, node )
		for elt in node.elts:
			operand = self._lower_expr( elt, elem_type )
			append_dest = self._new_temp( append_fn.return_type )
			self._emit( ir.Call( dest = append_dest, target = append_fn, receiver = dest, args = [ operand ], kwargs = {} ))
			errmsg = self._lower_expr( errmsg_node, unwrap_fn.parameters[0].type )
			# unwrap()'s own return value (T=None here, list[T].append's own
			# Result[None,OverflowError]) is never read - only its side
			# effect (panic on Err) matters, so no destination temp: T=None
			# compiles to a real C `void` return, and a real ir.Call dest
			# expects an actual value to assign, not void - same "dest=None
			# for a call whose result isn't used" convention _stmt_Expr's
			# own bare-call-statement handling already relies on
			self._emit( ir.Call( dest = None, target = unwrap_fn, receiver = append_dest, args = [ errmsg ], kwargs = {} ))
		return dest

	def _expr_Set( self, node: ast.Set, expected_type: Type|None ) -> ir.Operand:
		''' {a, b, c} - mirrors _expr_List's own shape (requires expected_type
		to already be a concrete set[T] Specialization - element-driven
		inference deferred, same precedent as list/tuple literals above).
		Builds one set[T] instance via _construct_generic_instance, then a
		real add(elt) call per element - unlike list[T].append, set[T].add
		returns plain None (no Result[None,OverflowError] to unwrap), so
		this skips _expr_List's errmsg/unwrap dance entirely. '''
		# _as_specialization, not a bare isinstance(expected_type,
		# Specialization) check - see _expr_List's own identical comment
		spec = self.lowering._type_resolver._as_specialization( expected_type )
		resolved = self.lowering._ensure_resolved( expected_type ) if expected_type is not None else None
		if not ( spec is not None and isinstance( resolved, RCClass )
				and spec.base.stem == 'set' and len( spec.args ) == 1 ):
			self.lowering.discovery.fail(
				f'set literal needs a known set[T] target type from context (e.g. an annotation or return type): {ast.unparse(node)}',
				node,
			)
		elem_type = spec.args[0]
		dest = self._construct_generic_instance( expected_type, node )
		if not node.elts:
			# the standard parser never actually produces an empty ast.Set
			# from source text (`{}` always parses as ast.Dict) - kept for
			# robustness against a synthetically-built empty node, same
			# defensive guard _expr_List keeps for its own analogous case
			return dest
		add_fn = self.lowering._find_method( dest.type, 'add' )
		assert add_fn is not None, 'internal compiler error: set[T] has no add method'
		self.lowering._ensure_resolved( add_fn )
		self.lowering.schedule( add_fn.return_type )
		for elt in node.elts:
			operand = self._lower_expr( elt, elem_type )
			# dest=None: add()'s return value (None) is never read, only its
			# side effect - same "dest=None for a call whose result isn't
			# used" convention _expr_List's own unwrap() call above relies on
			self._emit( ir.Call( dest = None, target = add_fn, receiver = dest, args = [ operand ], kwargs = {} ))
		return dest

	# obj.type.stem -> its own length-accessor method name, for slice
	# syntax's own default-stop resolution (_lower_slice_subscript below).
	# str and bytearray genuinely expose differently-named length
	# accessors (str.__len__() is a Unicode codepoint count - see its own
	# docstring - not the byte length _byte_slice's own byte-offset
	# contract needs; bytearray has no such split, __len__() IS its real
	# byte length) - not a uniform dunder lookup, so a small fixed table
	# for the two currently-supported types is the honest shape here,
	# same posture as the tuple-index/pointer-fallback cases elsewhere in
	# _expr_Subscript already hardcoding per concrete type family rather
	# than inventing a protocol for two callers
	_SLICE_LENGTH_METHOD = { 'str': 'byte_len', 'bytearray': '__len__' }

	def _lower_slice_subscript( self, node: ast.Subscript, obj: ir.Operand ) -> ir.Operand:
		''' x[a:b] / x[:b] / x[a:] - str/bytearray only (PLAN_POSIX_FEATURE.md's
		scope; list[T] slicing deferred - no real caller, and would need new
		RC-aware bulk-copy machinery list[T] doesn't have yet). Byte-offset
		semantics, not Python's real Unicode-codepoint offsets - deliberate:
		the one real caller (lib/posix/time.py's target_path[idx+9:]) slices
		from str.find()'s own byte offset, and str already has exactly the
		right byte-offset primitive (_byte_slice, also used by split()) -
		distinct from str.__len__()'s codepoint count. No special RC/
		aliasing tagging needed (unlike the tuple-index case's node.
		is_tuple_element_read) - this goes through an ordinary ir.Call,
		which the general Call-result convention already treats as a fresh,
		owned value by default. '''
		node_slice = node.slice
		assert isinstance( node_slice, ast.Slice )
		if node_slice.step is not None:
			self.lowering.discovery.fail( f'slice step is not supported: {ast.unparse(node)}', node )
		slice_fn = self.lowering._find_method( obj.type, '_byte_slice' )
		length_method_name = self._SLICE_LENGTH_METHOD.get( getattr( obj.type, 'stem', None ) )
		if slice_fn is None or length_method_name is None:
			self.lowering.discovery.fail(
				f'slicing is not supported for {obj.type.qualname} (only str and bytearray support slice syntax): {ast.unparse(node)}',
				node,
			)
		self.lowering._ensure_resolved( slice_fn )
		self.lowering.schedule( slice_fn.return_type )
		start_type = slice_fn.parameters[0].type
		stop_type = slice_fn.parameters[1].type
		if node_slice.lower is not None:
			start = self._lower_expr( node_slice.lower, start_type )
		else:
			start = ir.Const( type = start_type, value = 0 )
		if node_slice.upper is not None:
			stop = self._lower_expr( node_slice.upper, stop_type )
		else:
			length_fn = self.lowering._find_method( obj.type, length_method_name )
			self.lowering._ensure_resolved( length_fn )
			self.lowering.schedule( length_fn.return_type )
			len_dest = self._new_temp( length_fn.return_type )
			self._emit( ir.Call( dest = len_dest, target = length_fn, receiver = obj, args = [], kwargs = {} ))
			stop = len_dest
		dest = self._new_temp( slice_fn.return_type )
		self._emit( ir.Call( dest = dest, target = slice_fn, receiver = obj, args = [ start, stop ], kwargs = {} ))
		return self._maybe_consume_result( node, dest, self.lowering._SUBSCRIPT_ALTERNATIVES )

	def _expr_Subscript( self, node: ast.Subscript, expected_type: Type|None ) -> ir.Operand:
		if isinstance( node.value, ast.Attribute ) and not isinstance( node.slice, ast.Slice ):
			fixed = self._fixed_array_index_target( node.value, node.slice )
			if fixed is not None:
				root, attr, array_type, index = fixed
				dest = self._new_temp( array_type.elem_type )
				self._emit( ir.GetAttrIndex( dest = dest, obj = root, attr = attr, index = index ))
				return self._maybe_castwrap_pointer( dest, expected_type )
		obj = self._lower_expr( node.value, None )
		if isinstance( node.slice, ast.Slice ):
			return self._lower_slice_subscript( node, obj )
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
				# genuinely aliasing (a GetAttr borrow of the tuple's own
				# field, NOT a fresh Call-owned value) - _is_aliasing_expr's
				# own comment already anticipated exactly this path
				# ("raw ir.GetItem/GetAttr, genuinely aliasing a container
				# element... revisit this once [an indexable container]
				# does [exist]") but was never revisited once tuples landed.
				# Tag the node here (same posture as node.resolved_callee/
				# node.is_narrowing_bind elsewhere) rather than have
				# _is_aliasing_expr re-inspect node.value's own type itself,
				# which is exactly the "risk re-resolving/double-evaluating
				# the receiver" its own comment already rules out - real
				# repro: `q: int = some_tuple[0]` (T RC) then reassigning an
				# existing local to another tuple-index-read, in a loop,
				# with the tuple itself released each iteration, silently
				# skipped this Incref, so the tuple's own destructor's
				# cascading decref of its OWN fields double-released the
				# SAME object the reassigned local still pointed to - a
				# real, confirmed (via generated-C inspection and repeated
				# real-compile-and-run trials) use-after-free, not a
				# hypothetical.
				node.is_tuple_element_read = True
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
		# expected_type is the OUTER statement's own target (e.g. `r:
		# Result[i32|int,E] = x + y`'s Result[...]) - hinting an OPERAND's
		# own lowering with it directly is wrong whenever expected_type is
		# a union: if the operand's OWN natural type happens to exactly
		# equal one of its leaves (x: i32|int here, matching Result's own
		# Ok leaf exactly), _coerce_or_check_operand's union-wrap coercion
		# silently wraps x into Result.Ok(x) BEFORE _lower_binop_values
		# ever runs - x's type is then Result[...], not i32|int, breaking
		# both ordinary dunder lookup and _lower_binop_dispatch's own
		# union-shape detection. Same bug CLASS _lower_eq_or_ne's own
		# right_hint fix already covers for left.type being a union - this
		# is the expected_type-cascades-from-the-caller counterpart,
		# confirmed via a real repro (`r: Result[i32|int,E] = x + y`,
		# x/y: i32|int). None here lets the operand infer its own natural
		# type instead, exactly as if no hint had been given at all.
		# _tagged_union_shape, not a raw isinstance check - expected_type
		# here can still be a Specialization wrapping a TaggedUnion base
		# (a generic Result[T,E] annotation not yet monomorphized) rather
		# than a bare TaggedUnion already - same Specialization-vs-plain
		# duality _tagged_union_shape already exists to paper over
		# everywhere else in this file
		operand_hint = expected_type if self.lowering._type_resolver._tagged_union_shape( expected_type ) is None else None
		# pointer arithmetic (ptr + offset) is never homogeneous the way
		# ordinary scalar +/- is - hinting the OTHER (non-pointer) side with
		# the pointer's own type here (as every branch below otherwise
		# does) would wrongly propagate into a nested BinOp too (e.g. `ptr +
		# idx * element_size` - the outer Add's own right-hint, if left
		# unguarded, leaks into the INNER Mult's result_type, producing a
		# nonsensical "checked pointer multiply"). usize is the natural
		# offset type - same reasoning _expr_Subscript's own pointer index
		# hint already uses
		# every _lower_expr call below is strict=False: these are all HINTS
		# (helping an untyped literal or a nested generic call infer its own
		# type), never a real requirement the operand must satisfy - the
		# real requirement, if any, is validated separately, either by _lower_
		# binop_values' own explicit checks (e.g. its float-same-type rule)
		# or by the OUTER _lower_expr call that invoked _expr_BinOp/_expr_
		# Compare in the first place, re-checking the whole expression's own
		# RESULT afterward. Without this, e.g. `c: f64 = a + b` (a: f64, b:
		# f32) would have b silently widened to f64 right here, bypassing
		# _lower_binop_values' own deliberately stricter "both operands must
		# already be the SAME float type, cast explicitly" rule entirely.
		if left_is_const and not right_is_const:
			right = self._lower_expr( right_node, operand_hint, strict = False )
			# only hint the literal toward right.type when that's actually a
			# meaningful target for a literal to become (a scalar, or the
			# existing Ptr-offset special case) - hinting toward an arbitrary
			# non-scalar CLASS (e.g. `5 + some_vector`, right.type=Vector) hits
			# _expr_Constant's own literal-compatibility check ("an int literal
			# cannot be used where Vector is expected") before this expression
			# ever reaches _lower_binop_values' own dunder/reflected-dunder
			# dispatch - confirmed via a real repro. None here lets the
			# literal infer its own natural type instead, same as it would
			# with no hint at all, so the reflected-dunder lookup below can
			# still find e.g. Vector.__radd__(other: i32) matching it.
			if self.lowering._type_resolver._is_ptr_specialization( right.type ):
				left_hint = usize_cls
			elif isinstance( right.type, Scalar ):
				left_hint = right.type
			else:
				left_hint = None
			left = self._lower_expr( left_node, left_hint, strict = False )
		elif right_is_const and not left_is_const:
			left = self._lower_expr( left_node, operand_hint, strict = False )
			if self.lowering._type_resolver._is_ptr_specialization( left.type ):
				right_hint = usize_cls
			elif isinstance( left.type, Scalar ):
				right_hint = left.type
			else:
				right_hint = None
			right = self._lower_expr( right_node, right_hint, strict = False )
		else:
			left = self._lower_expr( left_node, operand_hint, strict = False )
			if infer_right_from_left:
				if self.lowering._type_resolver._is_ptr_specialization( left.type ):
					right_hint = usize_cls
				elif operand_hint is not None:
					right_hint = operand_hint
				elif self.lowering._type_resolver._tagged_union_shape( left.type ) is not None:
					# same "don't hint an operand's own lowering with a
					# union" principle as operand_hint above, just via a
					# DIFFERENT path: left.type here is the fallback hint
					# for right when nothing else applies, but if left
					# happens to be a union operand (x: Vector|i32),
					# hinting right (a PLAIN Vector-typed operand, no
					# union of its own) with it wrongly wraps right into
					# THAT union too - widening the leaf-pair grid with a
					# cell the source never actually expressed. Confirmed
					# via a real repro (Boxed|int + Boxed wrongly widened
					# right into Boxed|int too, inventing an (int,int)
					# grid cell that doesn't exist in the source at all)
					right_hint = None
				else:
					right_hint = left.type
			else:
				right_hint = operand_hint
			right = self._lower_expr( right_node, right_hint, strict = False )
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

		# either operand union-typed (`x: i32|int; y: i32|int; x + y`) - a
		# whole separate dispatch, mirroring _lower_eq_or_ne's identical
		# fork for ==/!=: a union operand has no dunder/scalar-arithmetic
		# shape of its OWN, only its individual LEAVES do, so every
		# (left leaf, right leaf) pairing needs its own classification -
		# see _lower_binop_dispatch's own docstring. Left completely
		# untouched below when neither operand is a union - existing,
		# already-verified dunder/reflected-dunder/scalar-arithmetic
		# tail keeps handling that case exactly as it does today.
		left_shape = self.lowering._type_resolver._tagged_union_shape( left.type )
		right_shape = self.lowering._type_resolver._tagged_union_shape( right.type )
		if left_shape is not None or right_shape is not None:
			return self._lower_binop_dispatch( node, left, left_shape, right, right_shape )

		# try the dunder method (str.__add__, ...) first on left.type, then -
		# mirroring Python's real protocol - the REFLECTED, differently-named
		# dunder on right.type (str.__radd__, ...) if the forward one isn't
		# applicable. Unlike the old code, this ISN'T gated on left.type
		# being non-Scalar: a Scalar left operand has no dunder of its own to
		# try (skip straight to reflected), but that must NOT also skip
		# giving right.type's own reflected dunder a chance - e.g. `5 +
		# some_vector` needs some_vector's own __radd__(other: i32), the
		# exact same "Scalar operand can't have a forward dunder, but the
		# OTHER side's own dunder is still a real candidate" gap _lower_eq_
		# or_ne's own unconditional-Eq/NotEq-dispatch fix already closed for
		# ==/!=; this is the binop counterpart, with a differently-named
		# reflected method instead of the same name reflected.
		method_name = _BINOP_DUNDER.get( type( node.op ))
		if method_name is not None:
			# NOT gated on isinstance(left.type, Scalar)/isinstance(right.type,
			# Scalar) anymore - _find_dunder_for_arg already resolves a
			# scalar-registered dunder (i32.__add__ = ...; see lib/builtins)
			# identically to a real class's own method (both just read
			# .names - see _find_method's own owner-kind branch). A Scalar
			# with nothing registered under this name just misses, exactly
			# like a class that doesn't define the dunder at all - no special
			# case needed for "this operand has no dunder mechanism".
			for candidate in self._mode_qualified_dunder_names( method_name ):
				method = self._find_dunder_for_arg( left.type, candidate, right.type )
				if method is not None:
					return self._emit_binop_dunder_call( node, method, left, right, expected_type )
			reflected_name = _REFLECTED_BINOP_DUNDER.get( method_name )
			if reflected_name is not None:
				for candidate in self._mode_qualified_dunder_names( reflected_name ):
					reflected_method = self._find_dunder_for_arg( right.type, candidate, left.type )
					if reflected_method is not None:
						return self._emit_binop_dunder_call( node, reflected_method, right, left, expected_type )

		# a float on EITHER side gets two float-specific rejections dunder
		# dispatch above can't produce itself (it only ever MISSES silently,
		# never explains why): strict same-type (a bare literal on either
		# side has already been hinted to the other's type by _lower_binary_
		# operands, so `f + 1.5`/`f + 1` still work; only a float mixed with
		# an int VARIABLE, or two different float widths, reaches this error.
		# NOTE an int VARIABLE combined with a bare float LITERAL (`i + 1.5`)
		# is not caught here - the literal is hinted to the int's type and
		# truncated, an accepted first-pass edge), and bitwise/shift/floordiv/
		# mod, which have no floating-point meaning and no dunder at all.
		# Every remaining same-type float shape (+-*/`/`) already has a real
		# dunder (lib/builtins/__scalar_dunders.py's f_*_checked/wrapped/
		# saturated, __truediv__) and dispatched through it above - nothing
		# legitimate reaches past these two checks.
		if _is_float_scalar( left.type ) or _is_float_scalar( right.type ):
			if left.type is not right.type:
				float_type = left.type if _is_float_scalar( left.type ) else right.type
				self.lowering.discovery.fail(
					f'floating-point operation requires both operands to be the same type - '
					f'got {left.type.qualname if left.type else "?"} and {right.type.qualname if right.type else "?"}; cast one explicitly '
					f'(e.g. {float_type.stem}(x)): {ast.unparse(node)}',
					node,
				)
			bad = _FLOAT_UNSUPPORTED_BINOPS.get( type( node.op ))
			if bad is not None:
				self.lowering.discovery.fail( f'operator {bad!r} is not supported on floating-point values: {ast.unparse(node)}', node )

		# every operator this language actually supports has a dunder mapping
		# in _BINOP_DUNDER above (dispatched through it, or through the float
		# checks just above, before ever reaching here) - what's left is a
		# genuinely unsupported operator (`/` on an int - only `//` exists;
		# `**`/`@`, never mapped to anything at all)
		self.lowering.discovery.fail( f'unsupported binary operator: {ast.unparse(node)}', node )

	def _mode_qualified_dunder_names( self, base_name: str ) -> list[str]:
		# see _MODE_DUNDER_PREFIX's own module-level comment for the full
		# rationale - the candidate dunder name(s) to try, in order, for
		# binop dispatch given the CURRENT ambient arithmetic mode
		prefix = _MODE_DUNDER_PREFIX.get( type( self._arithmetic_mode[-1] ))
		if prefix is None:
			return [ base_name ]
		return [ f'__{prefix}_{base_name.strip( "_" )}__', base_name ]

	def _emit_binop_dunder_call( self, node: 'ast.BinOp|ast.AugAssign', method: Function, receiver: ir.Operand, arg: ir.Operand, expected_type: Type|None ) -> ir.Operand:
		# shared tail for the forward/reflected dunder-call cases in
		# _lower_binop_values above - a thin, binop-shaped wrapper over
		# _emit_fallible_method_call (a two-operand call: receiver + one
		# arg), which does the real work and is reused by any OTHER call
		# needing the identical "resolve+schedule, splice-or-Call, consume
		# via ambient mode if @fallible_arithmetic" treatment (e.g. .to_T()
		# conversion dispatch - a one-operand call, no second arg)
		return self._emit_fallible_method_call( node, method, receiver, [ arg ], expected_type )

	def _emit_fallible_method_call( self, node: ast.AST, method: Function, receiver: ir.Operand|None, args: list[ir.Operand], expected_type: Type|None ) -> ir.Operand:
		# handles a plain class method (int.__add__, ...) and a
		# scalar-registered one (i32.__add__ = ..., see lib/builtins)
		# identically, since the caller's own dunder/method lookup already
		# resolved both the same way. `method.cls is None` means a genuine
		# free function was registered onto a Scalar (never had `self`
		# stripped by discovery) - same fix _lower_method_call/the general
		# _lower_call already apply: the receiver becomes a plain leading
		# positional arg instead of ir.Call.receiver.
		#
		# is_inline is checked BEFORE is_fallible_arithmetic, not instead of it: @inline
		# splicing (_lower_inline_call) never auto-consumes anything on its
		# own - arithmetic modes don't translate through a function call
		# boundary just because it happens to be inlined away (that's a
		# deliberate design choice, not a gap - see compiler.checked_add's
		# own comment). A @fallible_arithmetic+@inline method's spliced trailing return
		# (typically a compiler.checked_add/wrapped_add/saturated_add/
		# checked_convert intrinsic call) hands back its raw, real return
		# value - for is_fallible_arithmetic methods that's a genuine,
		# unconsumed Result[T,E], exactly matching the declared signature.
		# So is_fallible_arithmetic consumption below runs uniformly on
		# whatever came back, whether that value was produced by a real
		# ir.Call or by a splice - this is what makes `with compiler.
		# panic_arithmetic(...): a // b` (a, b: int) auto-panic, and
		# default-mode `a // b` auto-propagate, exactly like a bare scalar
		# `+` already does.
		# _resolve_call_target, NOT _ensure_resolved - the latter
		# unconditionally schedules its target as a real compile unit as a
		# side effect (see its own docstring), which for an @inline target
		# means compiling it as real, dead, never-called code (confirmed by
		# a real repro - see _resolve_call_target's own identical carve-out
		# and comment, already relied on by the general _lower_call path;
		# this is the same fix, needed again here since dunder-dispatch
		# resolves its own target independently rather than going through
		# that shared path)
		self.lowering._resolve_call_target( method )
		self.lowering.schedule( method.return_type )
		for p in ( method.parameters or [] ):
			self.lowering.schedule( p.type )
		call_receiver = None if method.cls is None else receiver
		call_args = ( [ receiver ] + args ) if method.cls is None else args
		if method.is_inline:
			result = self._lower_inline_call( node, method, call_receiver, call_args, {}, method.return_type, True )
			assert result is not None # want_result=True above guarantees this
		else:
			# expected_type describes the FINAL, post-consumption value (e.g.
			# `q1: int = a // b`'s expected_type is the success type `int`,
			# not the intermediate Result[int,E]) - for an is_fallible_
			# arithmetic method the dest here is that raw, unconsumed Result,
			# so it must always be typed as method.return_type exactly, never
			# expected_type. Using expected_type here silently mistyped the
			# Call's dest, which downstream Unwrap/OrJump lowering then
			# tried to treat as the wrong Result shape - a real, confirmed
			# bug (an assertion in _result_tag_data_names, whose own error-
			# message formatting then hit an unrelated circular-repr hang
			# instead of failing cleanly).
			dest_type = method.return_type if method.is_fallible_arithmetic else ( expected_type or method.return_type )
			dest = self._new_temp( dest_type )
			self._emit( ir.Call( dest = dest, target = method, receiver = call_receiver, args = call_args, kwargs = {} ))
			result = dest
		if not method.is_fallible_arithmetic:
			return result
		shape = self.lowering._type_resolver._tagged_union_shape( method.return_type )
		assert shape is not None and len( shape[1] ) == 2, f'@fallible_arithmetic {method.qualname} must declare a Result[T,E] return type'
		success_type = shape[1][0].type
		error_type = shape[1][1].type
		mode = self._arithmetic_mode[-1]
		extra = mode.extra if isinstance( mode, arithmetic_mode.ArithmeticPanic ) else None
		if extra is None:
			# same requirement _lower_arithmetic_op's own Check-mode opcodes
			# already enforce before emitting OrReturn/OrJump - missing here
			# let an @fallible_arithmetic dunder call (int.__floordiv__/
			# __mod__ via `//`/`%`, or a Scalar-registered arithmetic dunder)
			# silently emit an OrReturn/OrJump into a function whose return
			# type can't represent the error at all, crashing at C emission
			# time instead of failing to compile cleanly - confirmed via a
			# real repro (`r: int = a // b` inside a function declared -> i32)
			result_cls = self.lowering.discovery.find_name( 'Result', node )
			self.lowering._type_resolver._require_result_return(
				node, result_cls, error_type, self.lowering._FALLIBLE_METHOD_ALTERNATIVES, fn = self._current_fn,
			)
		return self._consume_checked_result( node, result, success_type, extra )

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
		if not opcode.checked_errors:
			# wrap/saturate, or no overflow concept at all (bitwise/Invert)
			dest = self._new_temp( result_type )
			self._emit( opcode( dest = dest, **operand_kwargs ))
			return dest
		# check mode (the default - see the class docstring): the op itself
		# produces Result[result_type,<error_type>], where <error_type> is a
		# single marker class (most ops) or the anonymous UNION of several
		# (signed Div/Mod -> ZeroDivisionError|OverflowError; checked float / ->
		# ZeroDivisionError|FloatingPointError). How that Result gets consumed
		# depends on `extra`: the default (extra is None) uses OrReturn,
		# mirroring Result.or_return()'s own semantics, and needs somewhere for
		# the error to propagate to; `with compiler.panic_arithmetic(msg):`
		# (extra is the lowered msg operand) uses Unwrap instead, which panics
		# immediately and so has no such requirement
		result_cls = self.lowering.discovery.find_name( 'Result', node )
		error_type, alternatives = self._resolve_checked_error( node, opcode, result_type )
		if extra is None:
			# validated before anything gets emitted - a mid-statement
			# failure here must not leave partial instructions behind for
			# the per-statement recovery boundary to silently keep
			self.lowering._type_resolver._require_result_return( node, result_cls, error_type, alternatives, fn = self._current_fn )
		return self._emit_checked_op( node, opcode, operand_kwargs, result_type, result_cls, error_type, extra )

	def _resolve_checked_error( self, node: ast.AST, opcode: type, result_type: Type ) -> tuple[ClassLike,str]:
		# the error TYPE a checked opcode's Result is against, plus the "how to
		# avoid needing to propagate it" message. opcode.checked_errors lists
		# the possible error class names; opcode.signed_only names the ones that
		# only apply to a signed integer result (OverflowError on Div/Mod's
		# INT_MIN/-1) - filtered out for unsigned operands. A single remaining
		# name -> that marker class (unchanged from before); several -> the
		# anonymous union of them (interned by _get_or_create_union, so a user's
		# own `A | B` annotation on the enclosing function's return type is the
		# SAME object - identity holds for _require_result_return's coverage
		# check and for OrReturn's error copy).
		signed = _is_signed_scalar( result_type )
		names = [ e for e in opcode.checked_errors if e not in opcode.signed_only or signed ]
		classes = [ self.lowering.discovery.find_name( name, node ) for name in names ]
		if len( classes ) == 1:
			# single-error path: preserve the exact per-error message (existing
			# tests assert e.g. 'Result[_,ZeroDivisionError]' + 'panic_arithmetic')
			return classes[0], _ALTERNATIVES_BY_ERROR[names[0]]
		error_union = self.lowering.discovery._get_or_create_union( classes )
		# the union's tag/data storage must exist by emit time (OrReturn's
		# widening and the division emitter both read the inner variant tag)
		self.lowering.schedule( error_union )
		self.lowering._union_storage.get( error_union )
		alternatives = (
			f'change the enclosing function to return Result[_,{" | ".join(sorted(c.stem for c in classes))}] '
			'(or a wider union covering those), or wrap this in `with compiler.panic_arithmetic(...):`'
		)
		return error_union, alternatives

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

	def _build_generator_error_defer_replay( self ) -> list[ir.Instruction]:
		''' PLAN_GENERATORS.md's defer/errdefer phase (Mechanism 2) - lowers
		every site in self._generator_armed_defer_sites (LIFO - deepest/
		most-recently-armed first, same convention Mechanism 1's own
		_build_defer_replay_guards uses), wrapped in `if self.
		__defer_armed_N: <body>`, for splicing into an OrReturn's own
		epilogue (see this method's one call site). A no-op (empty list,
		no lowering work at all) whenever the list is empty - true for
		EVERY ordinary, non-generator function, and for a generator body
		with no armed defer/errdefer site reaching this exact position.

		Reuses AST synthesis + the swap-the-instruction-buffer technique
		_register_defer_block already established (lower once into a
		fresh buffer, splice the result) rather than hand-building IR
		directly - the body statements are ALREADY self.<field>-qualified
		(type_resolver.py's _build_generator_next_function renamed them
		once, via the same renamer used for everything else in $$__next__,
		before ever tagging a node with them), so ordinary statement
		lowering already does the right thing with zero new machinery.

		Each site's own body is deep-copied FRESH here (same reasoning as
		_build_defer_replay_guards - lowering attaches mutable per-
		occurrence attributes like resolved_* that would corrupt a shared
		node if two OrReturn sites, or this site and a Mechanism-1 normal-
		exit site, shared one). self._generator_armed_defer_sites is reset
		to empty while lowering each body - a defer/errdefer body is
		expected to be simple cleanup, not itself something needing its
		OWN error-defer replay; without this, a fallible operation nested
		inside a defer body would recurse into this same method against
		the SAME still-armed site, unboundedly.

		Each guard ALSO unsets its own flag right after replaying (self.
		__defer_armed_N = False) - same reasoning as _build_defer_replay_
		guards' own identical unset: this error exit permanently pins
		self.__state to done, but the generator OBJECT itself often isn't
		freed until later (whatever reference the caller still holds), at
		which point $$__destructor__'s own Mechanism-1 replay would
		otherwise see this SAME flag still True and fire the (non-
		errdefer) body a second time. '''
		if not self._generator_armed_defer_sites:
			return []
		instructions: list[ir.Instruction] = []
		outer_armed = self._generator_armed_defer_sites
		self._generator_armed_defer_sites = []
		try:
			for flag_stem, _is_errdefer, body_stmts in reversed( outer_armed ):
				body_copy = [ copy.deepcopy( s ) for s in body_stmts ]
				unset = ast.Assign(
					targets = [ ast.Attribute( value = ast.Name( id = 'self', ctx = ast.Load() ), attr = flag_stem, ctx = ast.Store() ) ],
					value = ast.Constant( value = False ),
				)
				guard = ast.If(
					test = ast.Attribute( value = ast.Name( id = 'self', ctx = ast.Load() ), attr = flag_stem, ctx = ast.Load() ),
					body = ( body_copy or [ ast.Pass() ] ) + [ unset ], orelse = [],
				)
				if body_stmts:
					ast.copy_location( guard, body_stmts[0] )
				else:
					guard.lineno = 1; guard.col_offset = 0
				ast.fix_missing_locations( guard )
				outer_instructions = self._instructions
				self._instructions = []
				try:
					self._lower_stmt( guard )
				finally:
					captured = self._instructions
					self._instructions = outer_instructions
				instructions += captured
		finally:
			self._generator_armed_defer_sites = outer_armed
		return instructions

	def _build_generator_pessimistic_done_pin( self ) -> list[ir.Instruction]:
		''' PLAN_GENERATORS.md Phase 4/Phase F - a fallible generator's
		$$__next__ needs "permanently done" set on ANY early error exit
		(or_return()'s own Err branch, or checked-arithmetic under Check
		mode consumed the same way) - OrReturn's own error exit returns
		directly out of $$__next__ WITHOUT running whatever would
		normally advance self.__state afterward, so without this, self.
		__state stays at whatever it was BEFORE the failing statement,
		and a later .__next__() call would wrongly re-enter and re-run
		the same (possibly already-consumed-a-moved-value) code from
		scratch.

		Phase F re-derives this at the LOWERING level (this hook,
		spliced into ir.OrReturn's own epilogue right alongside
		Mechanism 2's error-defer replay - see this method's one call
		site) instead of the old AST-level pre-write (_pessimistic_
		done_prefix, inserted ahead of every block of user code that
		MIGHT fail, deleted along with the rest of the unit-matcher):
		reaching this exact point during lowering already means an
		early Err-branch exit is really happening, so the pin only ever
		needs building once per OrReturn site, not speculatively ahead
		of every fallible-eligible block regardless of whether it's
		even generator code. A no-op outside a generator
		(self._current_fn.is_generator_next False for every ordinary
		function) - correct, since only a generator's own $$__next__
		has a self.__state field to pin at all. '''
		if not self._current_fn.is_generator_next:
			return []
		done_state = len( self._current_fn.node.generator_yield_states ) + 1
		pin = ast.Assign(
			targets = [ ast.Attribute( value = ast.Name( id = 'self', ctx = ast.Load() ), attr = '__state', ctx = ast.Store() ) ],
			value = ast.Constant( value = done_state ),
		)
		ast.fix_missing_locations( ast.copy_location( pin, self._current_fn.node ) )
		outer_instructions = self._instructions
		self._instructions = []
		try:
			self._lower_stmt( pin )
		finally:
			captured = self._instructions
			self._instructions = outer_instructions
		return captured

	def _consume_checked_result( self, node: ast.AST, check_dest: ir.Temp, result_type: Type, extra: ir.Operand|None ) -> ir.Temp:
		# shared by both binop (AddCheck/.../Div/Mod) and unary (NegCheck)
		# Check-mode ops, _maybe_consume_result's __len__/__getitem__ auto-
		# unwrap, and _lower_or_return's own <result_expr>.or_return() - see
		# _expr_BinOp's own comment on the OrReturn/OrJump/Unwrap split.
		# check_dest is sometimes a real, named Variable (or_return()'s own
		# receiver) and sometimes a bare Temp (checked arithmetic, __len__/
		# __getitem__'s auto-unwrap) - isinstance covers both uniformly
		# the innermost active multi-statement @inline splice, if this
		# early-exit-shaped construct (.or_return(), checked arithmetic
		# under the default Check mode, or the __len__/__getitem__ auto-
		# consume path) is reached from one of a spliced body's own pre-
		# return statements (self._current_fn is briefly the caller during
		# this window too - see _splice_multi_statement_inline_body's own
		# comment). Left unredirected, the OrReturn/OrJump path below would
		# jump to/return from the CALLER's own real epilogue - a real
		# correctness bug (silently skipping the rest of THIS splice AND
		# the caller's own subsequent statements), not just an unsupported
		# case - so both branches below stow into the SPLICE's own result
		# var/exited flag instead of self._return_value_var/a real return
		# whenever this is set. The trailing return-EXPRESSION itself is
		# lowered with this restored to None first, so it's unaffected -
		# nothing of the splice remains after it to skip past there, so
		# jumping to the caller's own epilogue is already correct, exactly
		# as the single-statement case already relies on
		if self._in_inline_splice_prelude and not self._inline_scope_vars:
			# PLAN_RETURN_INFERENCE.md's own @inline variant reached here
			# with target.return_type still the "infer it" sentinel (see
			# _splice_multi_statement_inline_body's own top-of-function
			# comment) - no inline scope exists to redirect into (result_
			# var's type isn't known yet, by construction), so this narrow
			# combination stays rejected, exactly as the single, blanket
			# guard this method used to have always rejected every
			# multi-statement splice's own pre-return statements
			self.lowering.discovery.fail(
				f'@inline: .or_return()/checked arithmetic that could propagate an error is not yet supported before the '
				f'final return of a multi-statement body whose own return type is still being inferred: {ast.unparse(node)}',
				node,
			)
		inline_scope = self._inline_scope_vars[-1] if self._in_inline_splice_prelude and self._inline_scope_vars else None
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
			# pass check_dest (when it's a named, tracked Variable - e.g.
			# `x.or_return()`, as opposed to a bare Temp from `foo().or_return()`)
			# into current_epilogue_label(), mirroring _stmt_Return's identical
			# call for `return x` - this lets the identity-skip guard recognize
			# "the operand being propagated out IS itself one of the still-live
			# tracked bindings" and route through the safe inline-replay path
			# below instead of jumping into check_dest's OWN epilogue label (which
			# would double-decref its payload right after copying it into the
			# return value, unretained - a real, confirmed use-after-free: check_
			# dest's Err payload gets moved into the returned struct, then the
			# shared epilogue ladder at that SAME label unconditionally decrefs
			# check_dest's own copy of it too). A bare Temp has no epilogue entry
			# of its own, so this is a no-op for that case, matching today's
			# already-correct behavior exactly (identical to passing None).
			tracked_operand = check_dest if isinstance( check_dest, Variable ) else None
			label = self._cfg.current_epilogue_label( tracked_operand )
			if label is not None:
				if inline_scope is not None:
					result_var, exited_flag, _merge_label = inline_scope
					self._emit( ir.OrJump( dest = unwrapped, value = check_dest, target = label, return_slot = result_var, exited_flag = exited_flag ))
				else:
					self._emit( ir.OrJump( dest = unwrapped, value = check_dest, target = label, return_slot = self._return_value_var ))
			else:
				# either check_dest's own entry needed excluding (the bug above),
				# or (matching _stmt_Return's own inline path for the identical
				# reasons - a confined entry, or simply nothing pending) there's no
				# shared label to jump to at all - return_() handles both uniformly:
				# it replays every OTHER still-live binding/defer/errdefer
				# obligation inline, skipping only check_dest's own entry (a no-op
				# skip when check_dest has no entry in the first place). Its
				# DeclareTemp side effects (via the injected new_temp callback)
				# land unconditionally in the main instruction stream right here,
				# same as _stmt_Return's identical call - only the rest (the
				# actual conditional replay logic) is embedded below, to run
				# strictly inside the Err branch
				replay = self._cfg.return_( tracked_operand, lambda: self._build_is_err_check( node ))
				# PLAN_GENERATORS.md's defer/errdefer phase (Mechanism 2) -
				# ir.OrReturn.epilogue is ALREADY Err-branch-exclusive by
				# construction (unlike an ordinary function's own shared
				# epilogue, which is reached by success AND error paths
				# alike, needing its own is_err() guard) - so both `defer`
				# and `errdefer` sites currently armed at this fallible
				# operation's own position just get appended here, no extra
				# guard needed. See _build_generator_error_defer_replay's
				# own docstring for why this is a no-op outside a generator.
				# The pessimistic-done pin runs FIRST, ahead of any defer/
				# errdefer replay - both are no-ops outside a generator,
				# order between them doesn't affect correctness inside one
				# (see _build_generator_pessimistic_done_pin's own docstring)
				replay = replay + self._build_generator_pessimistic_done_pin() + self._build_generator_error_defer_replay()
				if inline_scope is not None:
					# always `goto`s merge_label directly (see emitter_c.py's
					# own ir.OrReturn.inline_exit handling) - same bypass
					# shape _stmt_Return's identical branch marks captured for
					self._cfg.mark_inline_scope_captured()
					self._emit( ir.OrReturn( dest = unwrapped, value = check_dest, epilogue = replay, inline_exit = inline_scope ))
				else:
					self._emit( ir.OrReturn( dest = unwrapped, value = check_dest, epilogue = replay ))
		else:
			panic_fn = self.lowering._type_resolver._resolve_sys_function( 'panic' )
			self.lowering.schedule( panic_fn )
			self._emit( ir.Unwrap( dest = unwrapped, value = check_dest, errmsg = extra, panic = panic_fn ))
		# OrReturn/OrJump/Unwrap all extract the Ok payload as a raw struct-
		# field copy (emitter_c.py's _emit_or_return/_emit_or_jump/ir.Unwrap
		# handling) - a BORROW of check_dest's own payload, not a fresh
		# reference, exactly like Result.unwrap()'s old bare `return
		# self.data.v_Ok` was. check_dest's own payload gets its OWN eventual
		# decref (it's an ordinary tracked temp/binding like any other Result
		# value - see cfg.py's rc_leaves()/lowering.py's CFGState resolve_type
		# wiring for the fix that makes that actually happen now), so without
		# this incref `unwrapped` and check_dest's own decref would fight over
		# the SAME single reference - confirmed with a real UAF repro
		# (int.__floordiv__'s `self.divmod(other).or_return()`, caught by
		# AddressSanitizer). A no-op for a non-RC result_type (plain
		# arithmetic's own scalar Check ops), so safe to call unconditionally
		# regardless of which of this function's three callers reached here.
		for instr in self._cfg.incref( unwrapped.type, unwrapped ):
			self._emit( instr )
		return unwrapped

	def _expr_UnaryOp( self, node: ast.UnaryOp, expected_type: Type|None ) -> ir.Operand:
		# expected_type is the OUTER expression's own target (e.g. `return
		# -i` from a function declared -> i32|None) - hinting node.operand's
		# own lowering with it directly is wrong whenever expected_type is a
		# union: _lower_expr's own union-wrap coercion would silently wrap
		# the OPERAND into Some(i) BEFORE the unary operator ever runs,
		# leaving `-`/`~`/`not` trying to operate on a union value instead
		# of the scalar it actually needs - confirmed via a real repro
		# (`return -i` from a function declared -> i32|None generated
		# invalid C, assigning a bare int into the union struct directly;
		# NegWrap's own dest/operand had both silently become union-typed).
		# Same fix _lower_binary_operands' own operand_hint already applies
		# for +-*/etc - None here lets node.operand infer its own natural
		# type instead, exactly as if no hint had been given at all; the
		# unary operator's own RESULT still gets coerced into expected_type
		# normally, by the ordinary _lower_expr call that invoked this
		# method in the first place.
		operand_hint = expected_type if self.lowering._type_resolver._tagged_union_shape( expected_type ) is None else None
		if isinstance( node.op, ast.Not ):
			# strict=False: `not x` applies C-style truthiness to WHATEVER
			# scalar x already is (ir.Not/emitter_c.py's own `!operand` C
			# emission handles any scalar type, not just bool) - expected_type
			# here is only ever a hint for the RARE case node.operand itself
			# still needs inference (an untyped literal/generic call), never a
			# real requirement that x must already BE expected_type's own type
			operand = self._lower_expr( node.operand, operand_hint, strict = False )
			# `not x`'s own result is ALWAYS bool, never expected_type itself
			# (which, same reasoning as operand_hint above, might be a union
			# wrapping bool - `return not flag` from a function declared ->
			# bool|None) - dest must stay plain bool here so ir.Not's own
			# `dest = !(operand);` codegen matches its declared C type; the
			# post-dispatch _coerce_or_check_operand tail (_lower_expr's own,
			# back in the caller) is what wraps this plain bool into the
			# union afterward, same as every other _expr_X method relies on
			# it to. Confirmed via a real repro: using expected_type directly
			# here made dest itself union-typed, which _coerce_or_check_
			# operand's own "operand.type is expected_type already" identity
			# check then wrongly treated as "nothing to coerce", leaving a
			# bare `!(flag)` assigned straight into a union struct in C
			bool_cls = self.lowering.discovery.find_name( 'bool', node )
			dest = self._new_temp( bool_cls )
			self._emit( ir.Not( dest = dest, operand = operand ))
			return dest
		operand = self._lower_expr( node.operand, operand_hint )

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

		result_type = expected_type if _is_float_scalar( expected_type ) else operand.type

		# a float operand takes the GetFloatUnaryOp path: unary `-` is plain
		# negation (never faults), `~` falls through to the unsupported-operator
		# error - see ArithmeticMode.GetFloatUnaryOp
		if _is_float_scalar( operand.type ):
			opcode, extra = self._arithmetic_mode[-1].GetFloatUnaryOp( node )
		else:
			opcode, extra = self._arithmetic_mode[-1].GetUnaryOp( node )
		return self._lower_arithmetic_op( node, opcode, extra, result_type, { 'operand': operand }, 'unary' )

	def _flush_branch_temps( self, start_idx: int, *keep: ir.Operand ) -> None:
		# shared by _expr_BoolOp and _expr_IfExp: both lower a SEQUENCE of
		# conditionally-skippable sub-expressions (BoolOp operands after a
		# short-circuit jump; the IfExp branch that didn't run) into the
		# same straight-line instruction stream. A sub-expression that
		# chains multiple calls (e.g. `field.find(x)` receiver consumed by
		# `.is_ok()`, or `prefix + str('.') + k`'s two chained __add__s)
		# DeclareTemp's its own intermediate temps via the ordinary
		# self._new_temp() path - every one of those lands in
		# self._pending_temps exactly like any other temp. Neither method
		# ever touches these purely-intermediate temps itself (only the
		# construct's own final value - `operand`/`true_val`/`false_val` -
		# gets special handling), so left alone they'd survive in
		# _pending_temps all the way to the ENCLOSING STATEMENT's own
		# _flush_pending_temps() (e.g. _stmt_Return's, called once after
		# every operand/branch has already merged at end_label) - which
		# then decref's them UNCONDITIONALLY, including for whichever
		# operand/branch never actually ran (skipped by an earlier
		# operand's short-circuit jump, or the branch not taken), reading
		# tag/payload data off an uninitialized C local. Confirmed as a
		# real, reproducible crash in both shapes - not just a leak/UAF:
		# _expr_IfExp's case releases a garbage object pointer through a
		# garbage vtable (a stack-overflow crash); _expr_BoolOp's case
		# reads a garbage union tag and, whenever it happens to look like
		# the RC leaf, releases a garbage pointer straight from the stack
		# (a STATUS_BREAKPOINT crash under MSVC - confirmed with as few as
		# 2 chained `field.find(x).is_ok() or field.find(y).is_ok()`
		# operands, whenever the first one short-circuits).
		#
		# The fix: flush each operand/branch's OWN intermediate temps
		# (added to _pending_temps since `start_idx`, i.e. everything
		# DeclareTemp'd while lowering just this one) right here, before
		# moving on to the next operand or the branch's own Jump/merge -
		# exactly mirroring how the already-correct if/else STATEMENT form
		# gets this right for free (each branch is its own statement, so
		# _lower_stmt's per-statement pending_temps save/flush/restore
		# already scopes it correctly). `keep` (the construct's own
		# dest/final-value temps) is excluded - their ownership is already
		# fully resolved by the incref/untrack_temp() decision made just
		# above each call site (IfExp) or is simply never RC to begin with
		# (BoolOp's `operand` is always bool), and dest in particular is
		# still actively in use afterward (assigned into, then read again
		# once every operand/branch merges) so it must not be DeleteTemp'd
		# here even though the actual decref side would already be a safe
		# no-op for it.
		branch_temps = self._pending_temps[ start_idx: ]
		self._pending_temps = self._pending_temps[ : start_idx ]
		keep_ids = { k.id for k in keep if isinstance( k, ir.Temp ) }
		for t in reversed( branch_temps ):
			if t.id in keep_ids:
				continue
			for instr in self._cfg.delete_temp( t ):
				self._emit( instr )
			self._emit( ir.DeleteTemp( temp = t ))

	def _expr_BoolOp( self, node: ast.BoolOp, expected_type: Type|None ) -> ir.Operand:
		# short-circuit and/or: evaluate operands left to right, each into
		# the same dest temp, stopping early (jump to end) as soon as the
		# result is already decided - `and` stops on the first falsy
		# operand, `or` stops on the first truthy one. Needed by match's
		# nested pattern tests (an outer tag check AND, only if that
		# passes, an inner tag check on the payload - reading the payload
		# before confirming the outer tag would be reading the wrong
		# union member's storage)
		#
		# Each operand's own intermediate temps (e.g. `field.find(x)`'s
		# Result temp, consumed as `.is_ok()`'s receiver) are flushed via
		# _flush_branch_temps right after that operand's own code runs,
		# before any later operand's short-circuit jump could skip past a
		# temp this operand already finished with - see that method's own
		# comment for the confirmed crash this fixes.
		bool_cls = self.lowering.discovery.find_name( 'bool', node )
		is_and = isinstance( node.op, ast.And )
		end_label = self._new_label( 'booland' if is_and else 'boolor' )
		dest = self._new_temp( bool_cls )
		for i, value_node in enumerate( node.values ):
			operand_start = len( self._pending_temps )
			operand = self._lower_expr( value_node, bool_cls )
			self._emit( ir.Assign( dest = dest, src = operand ))
			self._flush_branch_temps( operand_start, dest, operand )
			if i < len( node.values ) - 1:
				jump_opcode = ir.JumpIfFalse if is_and else ir.JumpIfTrue
				self._emit( jump_opcode( cond = dest, target = end_label ))
		self._emit( ir.Label( name = end_label ))
		return dest

	def _expr_IfExp( self, node: ast.IfExp, expected_type: Type|None ) -> ir.Operand:
		# ternary `x if cond else y` — both branches assign to the same
		# dest temp, then merge at end_label. Use JumpIfTrue so the true
		# branch (body) comes first, avoiding an extra negate.
		#
		# RC bookkeeping mirrors cfg.assign()'s own is_alias split, done
		# per-branch since node.body/node.orelse can differ in aliasing-ness
		# (e.g. `x if cond else str('literal')`): an ALIASING branch value
		# (a plain Name/GetAttr read of an already-live binding) needs its
		# own Incref before being merged into dest, since dest becomes an
		# independent, longer-lived holder of the same reference; a FRESH
		# branch value (a Call/Allocate result, already registered via
		# fresh_temp() by whatever lowered it) has its ownership MOVED into
		# dest via the plain ir.Assign below, so it must be untrack_temp()'d
		# - otherwise _flush_pending_temps' later decref of the branch's own
		# temp double-frees the exact same object dest (and whatever dest
		# gets assigned into) still holds. dest itself only becomes tracked
		# once, after both branches (fresh_temp() is idempotent per id) -
		# confirmed as a real, reproducible UAF/double-free via direct
		# testing (`str('-') if cond else str('+')` corrupted/crashed
		# before this fix), not just reasoning from the code shape.
		#
		# Each branch's own PURELY INTERMEDIATE temps (e.g. every temp a
		# chained `prefix + str('.') + k` concatenation DeclareTemp's along
		# the way, none of which is `true_val`/`false_val` itself) are
		# flushed inside that branch via _flush_branch_temps - see its own
		# comment for why: left to the enclosing statement's normal
		# end-of-statement flush, they leak past end_label and get
		# unconditionally decref'd even in the branch that never ran,
		# releasing an uninitialized C local - a real, reproducible stack-
		# overflow crash (release_object on stack garbage), not just a
		# leak/UAF, confirmed via direct testing.
		bool_cls = self.lowering.discovery.find_name( 'bool', node )
		cond = self._lower_expr( node.test, bool_cls )
		else_label = self._new_label( 'ifexp_else' )
		end_label = self._new_label( 'ifexp_end' )
		dest = self._new_temp( expected_type ) if expected_type is not None else None
		self._emit( ir.JumpIfFalse( cond = cond, target = else_label ))
		# true branch
		true_branch_start = len( self._pending_temps )
		true_val = self._lower_expr( node.body, expected_type )
		if dest is None:
			dest = self._new_temp( true_val.type )
		if self.lowering._is_aliasing_expr( node.body, true_val ):
			for instr in self._cfg.incref( dest.type, true_val ):
				self._emit( instr )
		else:
			self._cfg.untrack_temp( true_val )
		self._flush_branch_temps( true_branch_start, dest, true_val )
		self._emit( ir.Assign( dest = dest, src = true_val ))
		self._emit( ir.Jump( target = end_label ))
		# false branch
		self._emit( ir.Label( name = else_label ))
		false_branch_start = len( self._pending_temps )
		false_val = self._lower_expr( node.orelse, dest.type )
		if self.lowering._is_aliasing_expr( node.orelse, false_val ):
			for instr in self._cfg.incref( dest.type, false_val ):
				self._emit( instr )
		else:
			self._cfg.untrack_temp( false_val )
		self._flush_branch_temps( false_branch_start, dest, false_val )
		self._emit( ir.Assign( dest = dest, src = false_val ))
		self._emit( ir.Label( name = end_label ))
		self._cfg.fresh_temp( dest, dest.type )
		return dest

	def _expr_Compare( self, node: ast.Compare, expected_type: Type|None ) -> ir.Operand:
		# ast.Is/IsNot ARE handled (see _lower_is_comparison) - identity
		# happens to coincide with value equality for every value kind this
		# language has today. ast.In/NotIn ARE ALSO handled (see
		# _lower_in_comparison) but needed their own dispatch method rather
		# than falling through _COMP_DUNDER below - see that method's own
		# comment for why
		if len( node.ops ) != 1 or len( node.comparators ) != 1:
			self.lowering.discovery.fail( f'chained comparisons are not yet supported: {ast.unparse(node)}', node )
		if isinstance( node.ops[0], ( ast.Is, ast.IsNot )):
			return self._lower_is_comparison( node, negate = isinstance( node.ops[0], ast.IsNot ))
		if isinstance( node.ops[0], ( ast.In, ast.NotIn )):
			return self._lower_in_comparison( node, negate = isinstance( node.ops[0], ast.NotIn ))

		left = self._lower_expr( node.left, None )
		if isinstance( node.ops[0], ( ast.Eq, ast.NotEq )):
			# Eq/NotEq get their OWN unified path (_lower_eq_or_ne), tried
			# BEFORE the scalar-vs-non-scalar fork below (unlike every other
			# operator) - a union can appear on EITHER side regardless of
			# whether the OTHER side happens to be scalar: `None == x` (a
			# bare None literal - NoneType is itself Scalar, see discovery.
			# py's get_none_type - has no dunder of its own at all) or `5 ==
			# n` (a bare int literal, also Scalar) both need the same
			# union-on-the-right handling as `"hi" == x`. The generic
			# method-lookup-on-non-scalar-left dispatch below can never
			# cover either shape, since it never even runs when left is
			# Scalar.
			return self._lower_eq_or_ne( node, left, expected_type, negate = isinstance( node.ops[0], ast.NotEq ))

		# try the dunder method (<, >, <=, >=, ...) - NOT gated on
		# isinstance(left.type, Scalar) anymore: _find_dunder_for_arg
		# already resolves a Scalar-registered dunder (i32.__lt__ = ...;
		# see lib/builtins/__scalar_dunders.py) identically to a real
		# class's own method (both just read .names). A Scalar with
		# nothing registered under this name (NoneType, in practice) just
		# misses, exactly like a class that doesn't define the dunder at
		# all - falls through to flat Cmp below either way, same
		# precedent as _lower_binop_values' own arithmetic dispatch
		method_name = _COMP_DUNDER.get( type( node.ops[0] ))
		if method_name is not None:
			# _find_dunder_for_arg, not the plain _find_method - see
			# its own docstring: right, lowered with strict=True just
			# below, is already guaranteed to end up exactly left.type
			# (coerced or rejected) before this dunder lookup even
			# matters, so the wanted implementation is whichever one
			# declares its own parameter as exactly left.type - same
			# "caller already knows the wanted arg type" shape as
			# _lower_eq_or_ne's own same-type fast path
			method = self._find_dunder_for_arg( left.type, method_name, left.type )
			if method is not None:
				right = self._lower_expr( node.comparators[0], left.type )
				# _emit_fallible_method_call, not a hand-rolled ir.Call -
				# a Scalar-registered dunder (method.cls is None) needs its
				# receiver threaded as a plain leading positional arg
				# instead of ir.Call.receiver (discovery.py never strips
				# "self" off a free function's own parameter list the way
				# it does for a real class method - same fix _lower_call/
				# _lower_method_call/_emit_binop_dunder_call already apply)
				return self._emit_fallible_method_call( node, method, left, [ right ], expected_type )
		# no matching dunder - a hard compile error, not a silent flat-Cmp
		# fallback (pointer identity comparison, or a plain scalar compare):
		# confirmed with the user - there's no sensible default for
		# comparing two arbitrary values (an RCClass's own == is meaningless
		# unless the class defines it), so every comparable type needs a
		# real dunder now, no exceptions. Scalars/Ptr[T]/ConstPtr[T]/CEnum
		# all have one (gen_scalar_dunders.py, lib/builtins/__ptr_arith.py,
		# discovery.py's _synthesize_cenum_comparison_methods) - this is
		# reached only by a genuine RCClass/CStruct/CUnion/TaggedUnion (or a
		# leftover corner) with no __<op>__ of its own.
		right = self._lower_expr( node.comparators[0], left.type )
		cmp_op = self.lowering._CMP_OPCODES.get( type( node.ops[0] ))
		if cmp_op is None:
			self.lowering.discovery.fail( f'unsupported comparison operator: {ast.unparse(node)}', node )
		type_name = left.type.qualname if left.type is not None else '?'
		self.lowering.discovery.fail(
			f'{type_name} has no {method_name}() defined - comparison requires an explicit dunder: {ast.unparse(node)}',
			node,
		)
		bool_cls = self.lowering.discovery.find_name( 'bool', node )
		dest = self._new_temp( bool_cls )
		self._emit( ir.Cmp( dest = dest, op = cmp_op, left = left, right = right ))
		return dest

	def _find_dunder_for_arg( self, owner_type: Type|None, name: str, arg_type: Type ) -> Function|None:
		''' like self.lowering._find_method, but Overload-aware: if `name`
		resolves to a real Overload group on owner_type (multiple defs
		sharing the name - e.g. int.__eq__(other: int) alongside a second
		int.__eq__(other: i32) cross-dunder overload), picks the ONE
		implementation whose single declared parameter type exactly matches
		arg_type, rather than silently treating the whole group as "no such
		method" the way a bare _find_method does (a real, confirmed gap:
		_find_method's own `isinstance(found, Function)` check returns None
		for an Overload - before this fix, adding a second int.__eq__
		overload SILENTLY broke the pre-existing int==int comparison too,
		since it fell through to comparing by raw pointer identity instead
		of calling __eq__ at all, confirmed via a real repro).

		Deliberately narrow, not a general replacement for _find_method
		everywhere: every caller here already knows the exact concrete
		argument type it wants to match against (comparison dispatch and
		binop/reflected-binop dispatch, not a call site needing real
		runtime dispatch across multiple candidate argument shapes), so a
		simple single-parameter-type scan over the Overload's own
		implementations suffices - no need for overload_resolution.py's
		own general ConditionalDispatch machinery. Used by: _lower_eq_or_ne
		and _classify_leaf_pair_eq (==/!=, both directions),
		_expr_Compare's </>/<=/>= dispatch, _lower_operand_compare, and
		_lower_binop_values' forward/reflected dunder dispatch (+-*//%|&^ and
		their __r<op>__ counterparts) - every one of these already forces
		(or already knows) the argument operand's exact type before dispatch
		is even reached, so the same "caller already knows the wanted arg
		type" precondition holds throughout. _find_method's other ~20
		remaining call sites elsewhere in this file (container-protocol
		dunders like __getitem__/__contains__/__next__, unary dispatch, ...)
		are unrelated and stay untouched - unary in particular can't
		meaningfully be Overloaded on argument type at all (no second
		operand to disambiguate against) - the rest are a separate,
		wider-scoped Overload-blindness gap, not fixed here. '''
		owner_type = self.lowering._ensure_resolved( owner_type )
		if isinstance( owner_type, ( CStruct, RCClass ) ):
			found = owner_type.chain_lookup( name )
		else:
			names = getattr( owner_type, 'names', None )
			found = names.get( name ) if isinstance( names, dict ) else None
		found = self.lowering._resolve_scalar_name( found )
		# a plain (non-Overload) Function is checked against arg_type here
		# too, NOT returned unconditionally the way a bare _find_method
		# would - a real bug caught during development: str only has ONE
		# __eq__(other: str), so this branch used to hand it back for ANY
		# arg_type (even i32), and the caller then emitted a Call passing
		# a mismatched scalar where struct builtins$str* was expected
		candidates = found.implementations if isinstance( found, Overload ) else ( [ found ] if isinstance( found, Function ) else [] )
		for impl in candidates:
			if impl.resolve is not None:
				impl.resolve()
			params = impl.parameters or []
			# a Scalar-registered dunder (impl.cls is None - a free function
			# whose receiver was never stripped by discovery, unlike a real
			# class method) declares its receiver as an ORDINARY leading
			# parameter (`def i32__add__i32(value: i32, other: i32)`), so
			# the operand to match against arg_type is params[1], not
			# params[0] - a real, confirmed bug found via a real repro
			# (`return a + b` from a function declared to return exactly
			# Result[i32,OverflowError] double-wrapped the Result, because
			# this check unconditionally required exactly ONE parameter and
			# so NEVER matched any Scalar-registered dunder at all, silently
			# falling through to the older, pre-dunder-dispatch direct-
			# opcode path below instead - which mistypes result_type as the
			# outer expected_type instead of falling back to left.type,
			# specifically when expected_type happens to already BE the
			# checked-Result shape the binop's own dunder dispatch was
			# supposed to produce). The existing test suite never caught
			# this because every existing test assigns the binop to an
			# unannotated local first (`c = a + b; return Result.Ok(c)`),
			# never returning the binop expression directly.
			arg_index = 1 if impl.cls is None else 0
			if len( params ) != arg_index + 1:
				continue
			param_type = params[arg_index].type
			# either an exact match (the ordinary case - e.g. usize against
			# a `other: usize` param), or a wildcard match against a still-
			# generic candidate's OWN type-param-typed parameter (e.g.
			# ptr_sub_dist[T]'s `other: Ptr[T]` against a concrete Ptr[i32]
			# arg_type - _same_type can't structurally match an unbound
			# TypeVar, so this is a separate, narrower check: same base,
			# and the param's own type arg is one of impl's own type_params -
			# any Ptr[whatever] arg_type counts, since T gets bound from the
			# RECEIVER below via _resolve_receiver_generic_dunder anyway,
			# which is what actually pins this parameter's concrete type).
			is_wildcard = (
				isinstance( param_type, Specialization ) and isinstance( arg_type, Specialization )
				and param_type.base is arg_type.base
				and any( isinstance( a, TypeVar ) and any( a is tv for tv in impl.type_params or [] ) for a in param_type.args )
			)
			if is_wildcard or self.lowering._type_resolver._same_type( param_type, arg_type ):
				return self.lowering._resolve_receiver_generic_dunder( impl, owner_type )
		return None

	def _lower_eq_or_ne( self, node: ast.Compare, left: ir.Operand, expected_type: Type|None, negate: bool ) -> ir.Operand:
		''' `==`/`!=`, for ANY left operand (scalar or not) - unlike every
		other comparison operator, Eq/NotEq are the one shape a union can
		ever meaningfully participate in (see _build_union_leaf_eq), and a
		union can show up on either side regardless of the OTHER side's own
		scalar-ness, so this doesn't share the non-scalar-left gate the rest
		of _expr_Compare's dunder dispatch still uses.
		Lowers node.comparators[0] EXACTLY ONCE (hinted toward left.type,
		non-strict - see the strict=False comment below), then tries, in
		order:
		  1. a real user __eq__/__ne__ on left's own type, when the
		     comparator's own natural type already matches left.type (the
		     ordinary/common case - unchanged behavior, byte-for-byte the
		     same dunder Call this used to emit before this method existed);
		  2. left is a union and the comparator is one of its own leaves
		     (`x == "hi"` where x: str|None - union on the LEFT);
		  3. the comparator is ITSELF a union containing left.type as a
		     member (`"hi" == x` or `None == x` - union on the RIGHT, the
		     mirror image of (2); NoneType in particular has no __eq__ of
		     its own at all, so a bare `None == x` never even reaches a
		     dunder lookup, and a bare int literal defaults to i32 - also
		     Scalar - so neither shape can be caught by gating on "left is
		     non-scalar" the way every other operator still does; only this
		     unified method, entered unconditionally for Eq/NotEq regardless
		     of left's own scalar-ness, sees both operands together and can
		     recognize the shape);
		  4. neither a matching dunder nor a recognized union - the
		     ORIGINAL pre-union-support behavior: a safe scalar widening
		     (i32->i64, ...) if one applies, else flat Cmp (pointer
		     comparison, or an ordinary same-type scalar compare) when the
		     comparator's natural type already matches left.type, or a
		     genuine type-mismatch compile error otherwise - reported via
		     _check_assignable the same way strict=True used to reject it
		     (before this method existed, that rejection - and the scalar
		     widening - ran INSIDE the right-hand _lower_expr call itself,
		     earlier than the dunder-vs-flat-Cmp fork could even be reached;
		     both are reinstated explicitly here since strict=False below
		     skips them). '''
		method_name = '__ne__' if negate else '__eq__'
		# _find_dunder_for_arg, not the plain _find_method: this is the
		# "receiver and argument end up the SAME type" fast path (right,
		# once lowered below, is hinted toward left.type when left isn't a
		# union - the common case), so the wanted implementation is
		# whichever one declares its own parameter as exactly left.type -
		# see _find_dunder_for_arg's own docstring for why a plain
		# _find_method silently breaks this once a class ever declares a
		# SECOND __eq__/__ne__ overload (e.g. int.__eq__(other: i32)
		# alongside the pre-existing int.__eq__(other: int)). NOT gated on
		# isinstance(left.type, Scalar) anymore - a Scalar-registered
		# __eq__/__ne__ (i32.__eq__ = ...; see lib/builtins/
		# __scalar_dunders.py) resolves identically here, same precedent
		# as _lower_binop_values' own arithmetic dispatch. NoneType (also
		# Scalar) has none registered, so `None == x` below still misses
		# and falls through to the union-recognition/flat-Cmp tail exactly
		# as before.
		method = self._find_dunder_for_arg( left.type, method_name, left.type )
		left_shape = self.lowering._type_resolver._tagged_union_shape( left.type )
		# the hint handed to the comparator's own lowering below: left.type,
		# EXCEPT when left.type is ITSELF a union - hinting a plain leaf
		# comparator toward a union expected_type would trigger _coerce_or_
		# check_operand's own (unconditional, not strict-gated) union-WRAP
		# coercion, turning a bare `"hi"` into a FULLY WRAPPED str|None
		# value before this method ever gets a look at it - defeating the
		# whole point of the union checks below (right.type would already
		# equal left.type by then, both union structs, and the code would
		# fall straight through to a flat Cmp comparing two STRUCTS
		# directly - confirmed by a real regression while developing this
		# fix). None here (natural inference only) matches exactly what the
		# union-on-the-left case always did, pre-unification.
		right_hint = None if left_shape is not None else left.type
		# strict=False: skips _coerce_or_check_operand's final
		# _check_assignable rejection AND its scalar-widening coercion
		# (both gated on strict) specifically so a still-mismatched right
		# (after every OTHER, unconditional coercion - union-wrap, RCClass
		# upcast, pointer cast - already had its chance) can be checked
		# HERE, against BOTH operands' own shapes, instead of failing (or
		# silently widening) blind - both are reinstated manually below in
		# the same order _coerce_or_check_operand itself would try them.
		right = self._lower_expr( node.comparators[0], right_hint, strict = False )
		if right.type is not left.type:
			if self._is_safe_scalar_widening( right.type, left.type ):
				widened = self._new_temp( left.type )
				self._emit( ir.CastWrap( dest = widened, operand = right ))
				right = widened
			else:
				right_shape = self.lowering._type_resolver._tagged_union_shape( right.type )
				return self._lower_eq_dispatch( node, left, left_shape, right, right_shape, negate )
		if method is not None:
			# _emit_fallible_method_call, not a hand-rolled ir.Call - see
			# _expr_Compare's own identical comment on why (Scalar-
			# registered dunder receiver threading)
			return self._emit_fallible_method_call( node, method, left, [ right ], expected_type )
		if left_shape is not None:
			# right.type IS left.type (the fast-path check above), and both
			# are the exact SAME union - genuinely no dunder of its own was
			# found. A flat Cmp below would compare two STRUCTS directly (C
			# rejects this) even though both operands are already known to
			# be the identical union type - route through the general
			# dispatch instead of assuming "same type -> flat Cmp is safe",
			# which only holds for scalars/pointers, never for a TaggedUnion
			return self._lower_eq_dispatch( node, left, left_shape, right, left_shape, negate )
		# no matching dunder - a hard compile error, not a silent flat-Cmp
		# fallback - see _expr_Compare's own identical comment for the full
		# rationale (confirmed with the user: no sensible default exists for
		# comparing two arbitrary values)
		type_name = left.type.qualname if left.type is not None else '?'
		self.lowering.discovery.fail(
			f'{type_name} has no {method_name}() defined - comparison requires an explicit dunder: {ast.unparse(node)}',
			node,
		)
		bool_cls = self.lowering.discovery.find_name( 'bool', node )
		dest = self._new_temp( bool_cls )
		self._emit( ir.Cmp( dest = dest, op = ir.CmpOp.NE if negate else ir.CmpOp.EQ, left = left, right = right ))
		return dest

	def _lower_eq_dispatch(
		self, node: ast.Compare, left: ir.Operand, left_shape: tuple[TaggedUnion,list[Variable]]|None,
		right: ir.Operand, right_shape: tuple[TaggedUnion,list[Variable]]|None, negate: bool,
	) -> ir.Operand:
		''' `==`/`!=` once the ordinary fast path (a real dunder whose
		declared parameter already matches right's natural type, or a safe
		scalar widening) has already been tried and didn't apply - the general
		case, covering all three arities uniformly: neither operand is a
		union, exactly one is, or both are. A non-union operand is treated as
		a degenerate single-leaf "shape" (its own type, no tag dispatch
		needed) - this is what lets one mechanism replace what used to be two
		separate ones (a leaf-vs-union binary tag test, and a plain
		_check_assignable rejection for two ordinary non-union types), not
		just generalize a new third case alongside them.

		Builds a full (left leaf type x right leaf type) grid via
		_classify_leaf_pair_eq - EVERY cell, not just whichever member happens
		to be "the matching one": comparing a union's CURRENTLY-INACTIVE
		member against the other side is not automatically "not equal" just
		because the tag doesn't match right now - it still needs the exact
		same same-type/cross-dunder/error classification as any other pairing
		(confirmed as a real gap in the narrower predecessor of this method,
		which only ever tested T|None-shaped unions - there, the "wrong"
		member was always NoneType, so its own hardcoded "wrong tag = not
		equal" shortcut happened to coincide with the correct answer by luck,
		not by construction; a union with two non-None members would have
		gotten this wrong).

		Per-cell rule (final, confirmed): both leaves NoneType -> trivially
		equal. Exactly one is NoneType -> trivially not equal (comparing
		anything against None is always well-defined - the Optional-check
		idiom - never an error, regardless of whether the OTHER side's
		declared type actually includes None as a possible member). Same
		concrete non-None type on both sides -> the existing
		_lower_operand_compare (dunder-or-flat-Cmp), unchanged. Different
		concrete non-None types -> Python's real equality protocol: try
		left_type's own __eq__/__ne__ first, then the REFLECTED call
		(right_type's own __eq__/__ne__, receiver/arg swapped - see
		_LeafPairEq's own docstring for why this isn't the same asymmetry as
		__radd__). Neither applies -> TypeError.

		Deliberately NOT gated on whether a union is actually involved: ANY
		'error' cell makes the whole comparison's result Result[bool,
		TypeError] uniformly, even when NEITHER side is a union at all (a
		single, statically-certain mismatch, e.g. two unrelated classes) -
		the user's own call: in practice a bare `if a == b:` without
		explicitly consuming the Result still hits an ordinary "expected bool,
		got Result[...]" mismatch either way, so nothing is lost by not
		special-casing the no-union case into a bespoke, immediate compile
		error the way the narrower predecessor of this method did - and the
		user gains the ability to explicitly consume/inspect a genuine
		type-confusion at runtime if they actually want to (`(a == b).unwrap_or(...)`,
		a match, ...), using the SAME Result[T,E] machinery every other
		fallible operation already provides, no special-casing needed.

		Deliberately does NOT reuse arithmetic's/subscript's own
		_maybe_consume_result auto-`.or_return()` consumption - explicitly
		rejected (no new "compiler binop mode" concept wanted): a fallible
		comparison's Result[bool,TypeError] is simply the expression's own
		real value, returned as-is. '''
		left_types = [ m.type for m in left_shape[1] ] if left_shape is not None else [ left.type ]
		right_types = [ m.type for m in right_shape[1] ] if right_shape is not None else [ right.type ]
		method_name = '__ne__' if negate else '__eq__'
		grid = [
			[ self._classify_leaf_pair_eq( lt, rt, node, method_name ) for rt in right_types ]
			for lt in left_types
		]
		fallible = any( cell.kind == 'error' for row in grid for cell in row )
		bool_cls = self.lowering.discovery.find_name( 'bool', node )
		if fallible:
			result_cls = self.lowering.discovery.find_name( 'Result', node )
			type_error_cls = self.lowering.discovery.find_name( 'TypeError', node )
			# result_union (the monomorphized, real TaggedUnion with concrete
			# .attributes) is what _coerce_into_union needs to find Ok/Err's own
			# synthesized member constructors, AND what dest itself must be
			# typed as - unlike _emit_checked_op's own Result[T,E] (used only
			# as a Check-mode opcode's dest, never directly compared against a
			# function's own declared return type by IDENTITY), a fully-
			# concrete generic annotation like `-> Result[bool,TypeError]`
			# (every type arg already concrete, no TypeVars left to bind) gets
			# EAGERLY monomorphized by the time _current_fn.return_type is
			# read (confirmed via a real repro) - _stmt_Return's own identity
			# check then requires dest.type to be that SAME monomorphized
			# object, not the bare Specialization
			check_type = self.lowering.discovery._get_or_create_specialization( result_cls, [ bool_cls, type_error_cls ] )
			self.lowering.schedule( check_type )
			result_union = self.lowering.monomorphize_class( check_type )
			self.lowering._union_storage.get( result_union )
			dest = self._new_temp( result_union )
		else:
			result_union = None
			dest = self._new_temp( bool_cls )
		self._emit_eq_dispatch_tree( node, left, left_shape, right, right_shape, grid, dest, negate, bool_cls, result_union )
		return dest

	def _classify_leaf_pair_eq( self, left_type: Type, right_type: Type, node: ast.AST, method_name: str ) -> _LeafPairEq:
		''' one grid cell of _lower_eq_dispatch's own classification - see
		that method's docstring for the full rule. Pure type-level, no IR. '''
		none_type = self.lowering.discovery.get_none_type()
		left_is_none = left_type is none_type
		right_is_none = right_type is none_type
		if left_is_none and right_is_none:
			return _LeafPairEq( 'none_true' )
		if left_is_none or right_is_none:
			return _LeafPairEq( 'none_false' )
		if self.lowering._type_resolver._same_type( left_type, right_type ):
			return _LeafPairEq( 'same_type' )
		# _find_dunder_for_arg, not the plain _find_method - see its own
		# docstring: a class declaring TWO __eq__/__ne__ signatures (the
		# same-type one plus a genuine cross-type one, e.g. int.__eq__
		# (other: i32) alongside int.__eq__(other: int)) registers as a
		# real Overload, which a bare _find_method silently treats as "no
		# such method" - the exact shape this whole 'cross_dunder' branch
		# exists to use. NOT gated on isinstance(Scalar) anymore - same
		# precedent as everywhere else this file's comparison/binop
		# dispatch dropped that gate; in practice this cross-type branch
		# still never matches for two DIFFERENT Scalar types today (no
		# cross-type scalar comparison dunders are registered, only same-
		# type ones - see lib/builtins/__scalar_dunders.py), so dropping
		# the gate is a no-op for Scalar pairs right now, not a behavior
		# change - just no longer special-cased for no reason.
		method = self._find_dunder_for_arg( left_type, method_name, right_type )
		if method is not None:
			return _LeafPairEq( 'cross_dunder', method = method, reflected = False )
		reflected_method = self._find_dunder_for_arg( right_type, method_name, left_type )
		if reflected_method is not None:
			return _LeafPairEq( 'cross_dunder', method = reflected_method, reflected = True )
		return _LeafPairEq( 'error' )

	def _emit_eq_dispatch_tree(
		self, node: ast.Compare, left: ir.Operand, left_shape: tuple[TaggedUnion,list[Variable]]|None,
		right: ir.Operand, right_shape: tuple[TaggedUnion,list[Variable]]|None,
		grid: list[list[_LeafPairEq]], dest: ir.Temp, negate: bool, bool_cls: Type, result_union: TaggedUnion|None,
	) -> None:
		''' nested 2-level tag dispatch shared by every arity _lower_eq_
		dispatch handles - disambiguates LEFT's active member first (skipped
		entirely when left_shape is None - a non-union operand is used
		directly, no tag test, matching a real union's own "last candidate
		needs no test either" shape), then WITHIN each left branch,
		disambiguates RIGHT's the same way. Reuses the identical
		union_storage.get(base) -> (tag_attr, data_attr, payload_cls, tags)
		primitive and GetAttr(tag)+Cmp EQ+JumpIfFalse / GetAttr(data)+
		GetAttr(v_<member>) IR shape already used identically in
		_lower_dispatch_tests/_maybe_unwrap_union_arg/_match_union_member
		(type_resolver.py) - not reinvented here. Union members are never
		themselves further unions (nested unions are flattened at discovery
		time), so this recursion is bounded to exactly 2 levels regardless of
		how many members either side has. '''
		none_type = self.lowering.discovery.get_none_type()
		end_label = self._new_label( 'eq_dispatch_end' )
		if left_shape is not None:
			left_base, left_members = left_shape
			left_tag_attr, left_data_attr, left_payload_cls, left_tags = self.lowering._union_storage.get( left_base )
			n_left = len( left_members )
		else:
			n_left = 1
		if right_shape is not None:
			right_base, right_members = right_shape
			right_tag_attr, right_data_attr, right_payload_cls, right_tags = self.lowering._union_storage.get( right_base )
			n_right = len( right_members )
		else:
			n_right = 1

		for i in range( n_left ):
			is_last_left = ( i == n_left - 1 )
			if left_shape is not None:
				lm = left_members[i]
				if not is_last_left:
					next_left_label = self._new_label( 'eq_dispatch_left_next' )
					tag_dest = self._new_temp( left_tag_attr.type )
					self._emit( ir.GetAttr( dest = tag_dest, obj = left, attr = left_tag_attr.stem ))
					match = self._new_temp( bool_cls )
					self._emit( ir.Cmp( dest = match, op = ir.CmpOp.EQ, left = tag_dest, right = ir.Const( type = left_tag_attr.type, value = left_tags[lm.stem] )))
					self._emit( ir.JumpIfFalse( cond = match, target = next_left_label ))
				# _emit_leaf_pair_eq_value only ever reads narrowed_left/
				# narrowed_right for a 'same_type'/'cross_dunder' cell - an
				# 'error' cell builds a TypeError instance instead, and a
				# 'none_true'/'none_false' cell returns a bare Const, neither
				# ever touching the operand at all. Extracting the payload
				# regardless is dead code (a real -Wunused-but-set-variable,
				# confirmed suite-wide). Skip it whenever NO cell in this row
				# can ever read it.
				row_has_reader_cell = any( c.kind in ( 'same_type', 'cross_dunder' ) for c in grid[i] )
				if lm.type is none_type or not row_has_reader_cell:
					narrowed_left = left   # never read
				else:
					narrowed_left = self._extract_union_payload( left, left_data_attr, left_payload_cls, lm )
			else:
				narrowed_left = left

			for j in range( n_right ):
				is_last_right = ( j == n_right - 1 )
				if right_shape is not None:
					rm = right_members[j]
					if not is_last_right:
						next_right_label = self._new_label( 'eq_dispatch_right_next' )
						tag_dest2 = self._new_temp( right_tag_attr.type )
						self._emit( ir.GetAttr( dest = tag_dest2, obj = right, attr = right_tag_attr.stem ))
						match2 = self._new_temp( bool_cls )
						self._emit( ir.Cmp( dest = match2, op = ir.CmpOp.EQ, left = tag_dest2, right = ir.Const( type = right_tag_attr.type, value = right_tags[rm.stem] )))
						self._emit( ir.JumpIfFalse( cond = match2, target = next_right_label ))
					# same reasoning as narrowed_left above, but per-cell: this
					# one cell (i,j) is the ONLY reader of narrowed_right
					if rm.type is none_type or grid[i][j].kind not in ( 'same_type', 'cross_dunder' ):
						narrowed_right = right
					else:
						narrowed_right = self._extract_union_payload( right, right_data_attr, right_payload_cls, rm )
				else:
					narrowed_right = right

				# cell_start brackets this ONE cell's own intermediate temps
				# (error_instance, and _coerce_into_union's own Call result
				# for `value` when result_union is set) - this whole per-cell
				# block is only ONE branch of a larger dispatch tree, and the
				# enclosing statement's natural end-of-statement flush fires
				# unconditionally for EVERY temp still tracked regardless of
				# which cell actually ran at runtime (first found via a real
				# ASAN SEGV - release_object() on an uninitialized C local
				# from a cell that was never taken; then a real ASAN LEAK
				# from an earlier fix that untracked with no decref at all).
				# _flush_branch_temps below is the general form of the fix
				# this cell used to apply by hand (see _lower_binary_branch's
				# own identical use for the same reason) - flushes every
				# temp created since cell_start, keeping only dest/value
				cell_start = len( self._pending_temps )
				cell = grid[i][j]
				if cell.kind == 'error':
					value: ir.Operand = self._build_type_error_instance( node )
				else:
					value = self._emit_leaf_pair_eq_value( node, narrowed_left, narrowed_right, cell, negate, bool_cls )
				if result_union is not None:
					value = self._coerce_into_union( value, result_union, node )
				self._flush_branch_temps( cell_start, dest, value )
				self._emit( ir.Assign( dest = dest, src = value ))
				self._emit( ir.Jump( target = end_label ))
				if right_shape is not None and not is_last_right:
					self._emit( ir.Label( name = next_right_label ))
			if left_shape is not None and not is_last_left:
				self._emit( ir.Label( name = next_left_label ))
		self._emit( ir.Label( name = end_label ))
		self._cfg.fresh_temp( dest, dest.type )

	def _extract_union_payload( self, union_operand: ir.Operand, data_attr: Variable, payload_cls: CUnion, member: Variable ) -> ir.Temp:
		''' the two-GetAttr "read one member's own payload out of a union's
		data storage" shape _build_union_leaf_eq/_maybe_unwrap_union_arg both
		used to duplicate independently - factored out here since
		_emit_eq_dispatch_tree now needs it on both axes. Both dests are bare
		GetAttr reads, never fresh_temp()-registered (see _emit's own comment
		- only Call/Allocate results are), so callers need no incref/decref
		bookkeeping around the returned value. '''
		payload_dest = self._new_temp( payload_cls )
		self._emit( ir.GetAttr( dest = payload_dest, obj = union_operand, attr = data_attr.stem ))
		narrowed = self._new_temp( member.type )
		self._emit( ir.GetAttr( dest = narrowed, obj = payload_dest, attr = f'v_{member.stem}' ))
		return narrowed

	def _emit_leaf_pair_eq_value( self, node: ast.AST, narrowed_left: ir.Operand, narrowed_right: ir.Operand, cell: _LeafPairEq, negate: bool, bool_cls: Type ) -> ir.Operand:
		''' produces a plain bool operand for one grid cell whose kind isn't
		'error' (Result-wrapping, if any, is _emit_eq_dispatch_tree's own job,
		kept out of here so this stays usable for both the infallible and
		fallible codegen shapes unchanged). '''
		if cell.kind == 'none_true':
			return ir.Const( type = bool_cls, value = not negate )
		if cell.kind == 'none_false':
			return ir.Const( type = bool_cls, value = negate )
		if cell.kind == 'same_type':
			return self._lower_operand_compare( narrowed_left, narrowed_right, negate, node )
		assert cell.kind == 'cross_dunder' and cell.method is not None
		method = cell.method
		receiver, arg = ( narrowed_right, narrowed_left ) if cell.reflected else ( narrowed_left, narrowed_right )
		self.lowering._ensure_resolved( method )
		self.lowering.schedule( method.return_type )
		for p in ( method.parameters or [] ):
			self.lowering.schedule( p.type )
		dest = self._new_temp( method.return_type )
		self._emit( ir.Call( dest = dest, target = method, receiver = receiver, args = [ arg ], kwargs = {} ))
		return dest

	def _build_type_error_instance( self, node: ast.AST ) -> ir.Temp:
		''' constructs a bare TypeError() instance - the first internal
		(non-AST-driven) construction site for a trivial marker-error class
		anywhere in this file (every existing one, OverflowError() etc., is
		only ever written in real library .py source). Mirrors the general
		class-construction tail's own RCClass branch
		(_schedule_rcclass_construction + bare ir.Allocate), simplified since
		TypeError is guaranteed zero-field. dest is a fresh, Allocate-
		registered temp (see _emit's own fresh_temp() rule) - immediately
		consumed by the caller's own _coerce_into_union, whose synthesized
		member-ctor Call increfs it into the Result's Err payload; the
		ordinary end-of-scope decref of dest itself brings the refcount back
		down to the single reference the Result now owns - same fresh-temp +
		union-ctor-incref shape any ordinary Result.Err(SomeClass()) already
		uses, no new RC mechanism. '''
		type_error_cls = self.lowering.discovery.find_name( 'TypeError', node )
		self.lowering._ensure_resolved( type_error_cls )
		self.lowering.schedule( type_error_cls )
		dest = self._new_temp( type_error_cls )
		assert isinstance( type_error_cls, RCClass )
		self.lowering._schedule_rcclass_construction( type_error_cls, dest.type )
		self._emit( ir.Allocate( dest = dest, cls = type_error_cls, fields = {} ))
		return dest

	def _lower_binop_dispatch(
		self, node: 'ast.BinOp|ast.AugAssign', left: ir.Operand, left_shape: tuple[TaggedUnion,list[Variable]]|None,
		right: ir.Operand, right_shape: tuple[TaggedUnion,list[Variable]]|None,
	) -> ir.Operand:
		''' +-*//%|&^ once at least one operand is union-typed - the
		arithmetic counterpart of _lower_eq_dispatch, covering all three
		arities the same way (a non-union operand is a degenerate
		single-leaf "shape", no tag dispatch needed on that axis). Builds a
		full (left leaf x right leaf) grid via _classify_leaf_pair_binop,
		then SYNTHESIZES the expression's own result type from whatever the
		grid actually produces - unlike equality (always plain bool),
		different leaf pairs here can produce genuinely different concrete
		types (i32+i32 -> i32 vs Vector+Vector -> Vector), and not every
		call site has an expected_type to coerce into, so a fresh union of
		the DISTINCT success types is synthesized via discovery.
		_get_or_create_union - collapsing to a single plain type when every
		reachable pair happens to agree (e.g. int|i32 + int where both
		int.__add__(int) and int.__radd__(i32) return plain int - must NOT
		become a degenerate 1-member union).

		Three independent sources of fallibility all fold into ONE
		synthesized error-type union the same way: (1) a leaf pair with no
		valid operation at all -> TypeError (mirrors equality's own
		'error' cell); (2) plain scalar arithmetic's own EXISTING checked-
		arithmetic fallibility, respecting the CURRENT arithmetic mode per
		cell (_classify_leaf_pair_binop calls the exact same
		_resolve_checked_error the non-union scalar path already uses -
		wrap_arithmetic/saturate_arithmetic/panic_arithmetic are honored
		exactly as they are today, never bypassed); (3) a resolved
		dunder's own declared return type can ITSELF be Result[T,E] - its
		E folds in too, not just its T used as-is (detected via cfg.
		is_result_type + _tagged_union_shape, the same Result-shape
		detection the rest of the compiler already uses).

		Deliberately does NOT reuse the plain scalar path's own auto-
		`.or_return()` consumption (_consume_checked_result) for a
		fallible cell - same "no compiler binop modes" principle
		_lower_eq_dispatch already settled: a fallible union binop's
		Result[...] is simply the expression's own real value, returned
		as-is (see _emit_binop_fallible_split). '''
		left_types = [ m.type for m in left_shape[1] ] if left_shape is not None else [ left.type ]
		right_types = [ m.type for m in right_shape[1] ] if right_shape is not None else [ right.type ]
		method_name = _BINOP_DUNDER.get( type( node.op ))
		reflected_name = _REFLECTED_BINOP_DUNDER.get( method_name ) if method_name is not None else None
		grid = [
			[ self._classify_leaf_pair_binop( node, method_name, reflected_name, lt, rt ) for rt in right_types ]
			for lt in left_types
		]
		success_types = _dedup_types( cell.success_type for row in grid for cell in row if cell.success_type is not None )
		error_types = _dedup_types( cell.error_type for row in grid for cell in row if cell.error_type is not None )
		fallible = bool( error_types )
		success_type = success_types[0] if len( success_types ) == 1 else self.lowering.discovery._get_or_create_union( success_types )
		self.lowering.schedule( success_type )
		if isinstance( success_type, TaggedUnion ):
			self.lowering._union_storage.get( success_type )
		error_type: Type|None = None
		if fallible:
			error_type = error_types[0] if len( error_types ) == 1 else self.lowering.discovery._get_or_create_union( error_types )
			self.lowering.schedule( error_type )
			if isinstance( error_type, TaggedUnion ):
				self.lowering._union_storage.get( error_type )
			result_cls = self.lowering.discovery.find_name( 'Result', node )
			check_type = self.lowering.discovery._get_or_create_specialization( result_cls, [ success_type, error_type ] )
			self.lowering.schedule( check_type )
			result_union = self.lowering.monomorphize_class( check_type )
			self.lowering._union_storage.get( result_union )
			dest = self._new_temp( result_union )
		else:
			result_union = None
			dest = self._new_temp( success_type )
		self._emit_binop_dispatch_tree( node, left, left_shape, right, right_shape, grid, dest, success_type, error_type, result_union )
		return dest

	def _classify_leaf_pair_binop( self, node: 'ast.BinOp|ast.AugAssign', method_name: str|None, reflected_name: str|None, left_type: Type, right_type: Type ) -> _LeafPairBinop:
		''' one grid cell of _lower_binop_dispatch's own classification -
		see that method's docstring for the full rule. Dunder lookup only,
		mode-qualified exactly like the non-union path's own forward/
		reflected loop in _lower_binop_values - no isinstance(Scalar)
		branch at all: a Scalar operand's own arithmetic is registered as
		a real (if @inline, zero-overhead) dunder now (see lib/builtins/
		__scalar_dunders.py), found via _find_dunder_for_arg identically to
		any class's own method. A leaf pair that can't type-check at all
		(mismatched float types - no matching dunder is ever registered
		for that pairing, so lookup just misses; an operator with no
		dunder mapping; or neither side has a usable method) becomes an
		'error' cell (contributing TypeError) rather than an immediate
		compile failure - mirrors _classify_leaf_pair_eq's own identical
		choice: a single bad pairing doesn't reject the WHOLE union
		expression, it becomes one runtime-checkable branch of it. '''
		type_error_cls = self.lowering.discovery.find_name( 'TypeError', node )
		method: Function|None = None
		reflected = False
		if method_name is not None:
			for candidate in self._mode_qualified_dunder_names( method_name ):
				method = self._find_dunder_for_arg( left_type, candidate, right_type )
				if method is not None:
					break
			if method is None and reflected_name is not None:
				for candidate in self._mode_qualified_dunder_names( reflected_name ):
					method = self._find_dunder_for_arg( right_type, candidate, left_type )
					if method is not None:
						reflected = True
						break
		if method is None:
			return _LeafPairBinop( 'error', error_type = type_error_cls )
		# _resolve_call_target, not _ensure_resolved - the latter
		# unconditionally schedules its target as a real compile unit,
		# which for an @inline scalar-arithmetic dunder means compiling
		# it as real, dead, never-called code - same gotcha
		# _emit_fallible_method_call's own comment documents, needed
		# again here since classification resolves the method before
		# codegen ever reaches that shared call site
		self.lowering._resolve_call_target( method )
		self.lowering.schedule( method.return_type )
		for p in ( method.parameters or [] ):
			self.lowering.schedule( p.type )
		if method.is_fallible_arithmetic:
			# consumed via the ambient arithmetic mode at emission time
			# (_emit_binop_cell), exactly like this SAME dunder already
			# behaves reached from the non-union path - never folds into
			# this expression's own aggregate error union
			shape = self.lowering._type_resolver._tagged_union_shape( method.return_type )
			assert shape is not None and len( shape[1] ) == 2, f'@fallible_arithmetic {method.qualname} must declare a Result[T,E] return type'
			return _LeafPairBinop( 'dunder', success_type = shape[1][0].type, method = method, reflected = reflected )
		if cfg.is_result_type( method.return_type ):
			shape = self.lowering._type_resolver._tagged_union_shape( method.return_type )
			assert shape is not None and len( shape[1] ) == 2
			return _LeafPairBinop( 'dunder', success_type = shape[1][0].type, error_type = shape[1][1].type, method = method, reflected = reflected )
		return _LeafPairBinop( 'dunder', success_type = method.return_type, method = method, reflected = reflected )

	def _emit_binop_dispatch_tree(
		self, node: 'ast.BinOp|ast.AugAssign', left: ir.Operand, left_shape: tuple[TaggedUnion,list[Variable]]|None,
		right: ir.Operand, right_shape: tuple[TaggedUnion,list[Variable]]|None,
		grid: list[list[_LeafPairBinop]], dest: ir.Temp, success_type: Type, error_type: Type|None, result_union: TaggedUnion|None,
	) -> None:
		''' nested 2-level tag dispatch - see _emit_eq_dispatch_tree's own
		docstring, this is the identical shape (same union_storage.get /
		GetAttr+Cmp+JumpIfFalse / _extract_union_payload primitive, N-1
		members tested per axis, last is the untested default). Differs
		only in the per-cell body: a binop cell can itself be
		independently fallible (checked scalar arithmetic under the
		current mode, or a dunder declaring Result[T,E]) - see
		_emit_binop_fallible_split for that inner Ok/Err decomposition,
		reached only from cells that need it. '''
		none_type = self.lowering.discovery.get_none_type()
		end_label = self._new_label( 'binop_dispatch_end' )
		bool_cls = self.lowering.discovery.find_name( 'bool', node )
		if left_shape is not None:
			left_base, left_members = left_shape
			# not _union_storage.get(left_base) - see _emit_binop_fallible_
			# split's own comment on why a Specialization's abstract base
			# (e.g. left is itself a Result[T,E]-typed operand) needs the
			# concrete, monomorphized union instead for storage purposes
			left_concrete = self.lowering.monomorphize_class( left.type ) if isinstance( left.type, Specialization ) else left_base
			left_tag_attr, left_data_attr, left_payload_cls, left_tags = self.lowering._union_storage.get( left_concrete )
			n_left = len( left_members )
		else:
			n_left = 1
		if right_shape is not None:
			right_base, right_members = right_shape
			right_concrete = self.lowering.monomorphize_class( right.type ) if isinstance( right.type, Specialization ) else right_base
			right_tag_attr, right_data_attr, right_payload_cls, right_tags = self.lowering._union_storage.get( right_concrete )
			n_right = len( right_members )
		else:
			n_right = 1

		for i in range( n_left ):
			is_last_left = ( i == n_left - 1 )
			if left_shape is not None:
				lm = left_members[i]
				if not is_last_left:
					next_left_label = self._new_label( 'binop_dispatch_left_next' )
					tag_dest = self._new_temp( left_tag_attr.type )
					self._emit( ir.GetAttr( dest = tag_dest, obj = left, attr = left_tag_attr.stem ))
					match = self._new_temp( bool_cls )
					self._emit( ir.Cmp( dest = match, op = ir.CmpOp.EQ, left = tag_dest, right = ir.Const( type = left_tag_attr.type, value = left_tags[lm.stem] )))
					self._emit( ir.JumpIfFalse( cond = match, target = next_left_label ))
				# an 'error' cell's own codegen (_emit_binop_cell) never reads
				# narrowed_left/narrowed_right - only a TypeError instance is
				# built. Extracting the payload anyway is dead code (a real
				# -Wunused-but-set-variable, confirmed suite-wide): skip it
				# whenever NO cell in this row can ever read it, i.e. every
				# right-side pairing for this left leaf is itself 'error'.
				row_has_dunder_cell = any( c.kind != 'error' for c in grid[i] )
				narrowed_left = left if lm.type is none_type or not row_has_dunder_cell else self._extract_union_payload( left, left_data_attr, left_payload_cls, lm )
			else:
				narrowed_left = left

			for j in range( n_right ):
				is_last_right = ( j == n_right - 1 )
				if right_shape is not None:
					rm = right_members[j]
					if not is_last_right:
						next_right_label = self._new_label( 'binop_dispatch_right_next' )
						tag_dest2 = self._new_temp( right_tag_attr.type )
						self._emit( ir.GetAttr( dest = tag_dest2, obj = right, attr = right_tag_attr.stem ))
						match2 = self._new_temp( bool_cls )
						self._emit( ir.Cmp( dest = match2, op = ir.CmpOp.EQ, left = tag_dest2, right = ir.Const( type = right_tag_attr.type, value = right_tags[rm.stem] )))
						self._emit( ir.JumpIfFalse( cond = match2, target = next_right_label ))
					# same reasoning as narrowed_left above, but per-cell: this
					# one cell (i,j) is the ONLY reader of narrowed_right, so
					# an 'error' cell alone is enough to skip it
					narrowed_right = right if rm.type is none_type or grid[i][j].kind == 'error' else self._extract_union_payload( right, right_data_attr, right_payload_cls, rm )
				else:
					narrowed_right = right

				self._emit_binop_cell( node, narrowed_left, narrowed_right, grid[i][j], dest, success_type, error_type, result_union, end_label )

				if right_shape is not None and not is_last_right:
					self._emit( ir.Label( name = next_right_label ))
			if left_shape is not None and not is_last_left:
				self._emit( ir.Label( name = next_left_label ))
		self._emit( ir.Label( name = end_label ))
		self._cfg.fresh_temp( dest, dest.type )

	def _emit_binop_cell(
		self, node: 'ast.BinOp|ast.AugAssign', narrowed_left: ir.Operand, narrowed_right: ir.Operand,
		cell: _LeafPairBinop, dest: ir.Temp, success_type: Type, error_type: Type|None, result_union: TaggedUnion|None, end_label: str,
	) -> None:
		''' one grid cell's codegen - see _lower_binop_dispatch's own
		docstring for what each kind means. An 'error' cell needs a fresh
		TypeError() instance; a 'dunder' cell is emitted via the SAME
		_emit_fallible_method_call the non-union path uses (one source of
		truth for "resolve+schedule, splice-or-Call, consume via ambient
		mode if @fallible_arithmetic") - passing expected_type=None always,
		since a non-fallible-arithmetic dunder's raw return value is what
		THIS dispatch's own _coerce_binop_value/_emit_binop_fallible_split
		need to see, never pre-coerced at the call site. For an
		is_fallible_arithmetic method, _emit_fallible_method_call has
		ALREADY consumed its Result via the ambient mode by the time it
		returns here (cell.error_type is None in that case, by
		construction - see _classify_leaf_pair_binop) - so `value` is
		simply the cell's own final success value either way; only a
		cell with error_type set (a REGULAR dunder whose own declared
		return type is Result[T,E], e.g. int.__add__/Vector.__add__)
		still needs the extra Ok/Err decomposition. branch_start (snapshotted
		here, at this cell's own entry, before anything below) is threaded
		through every tail this cell can reach - _finish_binop_cell's own
		_flush_branch_temps call uses it to release every intermediate this
		ONE cell created (error_instance, _coerce_binop_value's own
		`intermediate`), same reasoning _lower_binary_branch's identical
		snapshot-then-flush already documents. Safe to reuse ONE snapshot
		across a cell's own Ok/Err/nested-unwrap sub-branches too, even
		though those are themselves mutually exclusive at runtime -
		_flush_branch_temps trims self._pending_temps back to the snapshot
		on every call, so whichever sub-branch's flush actually runs first
        (in compile-time emission order) only ever sees temps created SO
        FAR, never a later sub-branch's not-yet-emitted ones. '''
		branch_start = len( self._pending_temps )
		if cell.kind == 'error':
			error_instance = self._build_type_error_instance( node )
			assert error_type is not None and result_union is not None   # an 'error' cell always contributes TypeError, so the whole expression is always fallible whenever one exists
			value = self._coerce_binop_value( error_instance, error_type, result_union, node )
			self._finish_binop_result_branch( branch_start, error_instance, value, dest, end_label )
			return
		assert cell.kind == 'dunder' and cell.method is not None
		receiver, arg = ( narrowed_right, narrowed_left ) if cell.reflected else ( narrowed_left, narrowed_right )
		value = self._emit_fallible_method_call( node, cell.method, receiver, [ arg ], None )
		if cell.error_type is None:
			value = self._coerce_binop_value( value, success_type, result_union, node )
			self._finish_binop_cell( branch_start, dest, value, end_label )
			return
		self._emit_binop_fallible_split( node, branch_start, value, dest, success_type, error_type, result_union, end_label )

	def _coerce_binop_value( self, value: ir.Operand, axis_type: Type, result_union: TaggedUnion|None, node: ast.AST ) -> ir.Operand:
		''' two-step coercion for ONE axis (success or error) of the
		synthesized dispatch: first into that axis's own aggregate type
		(success_type/error_type - itself a TaggedUnion only when more
		than one distinct type is actually possible on this axis; a
		cell's own produced value only ever matches ONE LEAF of it, never
		axis_type itself directly, whenever axis_type genuinely is a
		union), THEN - only when the whole expression is fallible - into
		result_union's own matching Ok/Err slot (a cell's value NEVER
		already matches result_union directly: result_union's own two
		leaves are success_type/error_type as a WHOLE, never one leaf's
		own concrete type - so this second step, unlike the first, is
		never skippable once result_union is present). Callers pass
		axis_type = success_type for a success-axis value, error_type
		for an error-axis value (always non-None whenever reached, since
		producing an error value at all implies the whole expression is
		fallible).

		When BOTH steps run, the intermediate axis_type-wrapped value is
		itself a fresh, independently fresh_temp()-tracked Call result
		(same shape as every other branch-local temp this whole dispatch
		tree produces) - the second _coerce_into_union call makes its OWN
		independent embedded reference via its own ctor's incref (a
		tag-gated copy of whichever member is active, since axis_type is
		itself RC-carrying whenever this path is taken), so the
		intermediate's OWN reference is now redundant. Left tracked and
		pending here deliberately (no inline decref/untrack) - the caller's
		own _finish_binop_cell (reached via _finish_binop_result_branch or
		directly) always flushes everything created since ITS OWN
		branch_start right before returning, which correctly sweeps this up
		either way: when result_union is None `intermediate` becomes the
		return value itself (kept alive - see the `if result_union is None:
		return intermediate` branch below, matches whatever `value`
		_finish_binop_cell was called with); when result_union is set,
		`intermediate` is a genuine throwaway distinct from the SECOND
		coercion's own result, correctly released by that same flush. '''
		if isinstance( axis_type, TaggedUnion ) and not self.lowering._type_resolver._same_type( value.type, axis_type ):
			intermediate = self._coerce_into_union( value, axis_type, node )
			if result_union is None:
				return intermediate
			value = self._coerce_into_union( intermediate, result_union, node )
			return value
		if result_union is not None:
			value = self._coerce_into_union( value, result_union, node )
		return value

	def _finish_binop_cell( self, branch_start: int, dest: ir.Temp, value: ir.Operand, end_label: str ) -> None:
		# mirrors _emit_eq_dispatch_tree's own identical per-cell tail -
		# flush everything this cell (or cell sub-branch, for the fallible
		# split path) created since branch_start, keeping dest/value - see
		# _emit_binop_cell's own docstring for why ONE snapshot correctly
		# scopes every sub-branch a cell can reach, not just the top level
		self._flush_branch_temps( branch_start, dest, value )
		self._emit( ir.Assign( dest = dest, src = value ))
		self._emit( ir.Jump( target = end_label ))

	def _finish_binop_result_branch( self, branch_start: int, raw_temp: ir.Temp, value: ir.Operand, dest: ir.Temp, end_label: str ) -> None:
		''' shared tail for every branch of _emit_binop_fallible_split (and
		the 'error' cell above, whose own error_instance is the identical
		shape) - raw_temp is a branch-local, independently fresh_temp()-
		tracked value (a checked op's check_dest, a dunder Call's own
		dest, or _build_type_error_instance's own Allocate result) whose
		OWN payload `value` was just extracted from (via _coerce_into_
		union, whose synthesized ctor already increfs `value` - see
		_build_type_error_instance's own docstring for why this specific
		incref-then-decref pairing is correctly balanced). raw_temp PRE-
		DATES branch_start (it's created once, shared across every
		Ok/Err/nested-unwrap sub-branch reachable from here, released on
		exactly whichever one actually runs) - _flush_branch_temps' own
		since-a-checkpoint model can't express "release a value that
		already existed before the checkpoint", so this stays a dedicated,
		explicit decref + untrack, unlike everything created AFTER
		branch_start (which _finish_binop_cell's own flush call, below,
		handles generically). '''
		for instr in self._cfg.decref( raw_temp.type, raw_temp ):
			self._emit( instr )
		self._cfg.untrack_temp( raw_temp )
		self._finish_binop_cell( branch_start, dest, value, end_label )

	def _emit_binop_fallible_split(
		self, node: ast.AST, branch_start: int, raw_temp: ir.Temp, dest: ir.Temp, success_type: Type, error_type: Type, result_union: TaggedUnion|None, end_label: str,
	) -> None:
		''' raw_temp is a not-yet-consumed Result[T,E] value (a checked
		scalar op's own check_dest, or a dunder's own Call result whose
		declared return type is itself Result[T,E]) - decomposed via ONE
		more nested Ok/Err tag-check (bounded, always exactly 2 members -
		Result's own shape), coercing whichever branch fires into the
		OUTER dest, deliberately WITHOUT _consume_checked_result's own
		auto-`.or_return()` consumption (see _lower_binop_dispatch's own
		docstring - this fallible value IS the expression's own real
		value here, not propagated to the enclosing function). Extracted
		via _extract_union_payload, which returns a bare, un-incref'd
		BORROW (unlike every other value this whole dispatch tree
		produces, which are always already-owned fresh Call/opcode
		results) - _coerce_binop_value's own _coerce_into_union call(s)
		are what give it a real +1 reference; result_union is guaranteed
		non-None whenever this is reached (a cell only gets here when its
		own error_type is set, which is exactly what makes the WHOLE
		expression fallible). '''
		assert result_union is not None
		shape = self.lowering._type_resolver._tagged_union_shape( raw_temp.type )
		assert shape is not None and len( shape[1] ) == 2
		base, members = shape
		# NOT _union_storage.get(base): base is _tagged_union_shape's own
		# deliberately-ABSTRACT return (shared tag values across every
		# instantiation of the same generic union) - raw_temp.type here is
		# routinely a Result[T,E] SPECIALIZATION (a scalar op's own
		# check_type, or a dunder's declared Result[T,E] return type), and
		# the ABSTRACT Result class's own payload union has no real C
		# definition at all (its fields are bare, unsubstituted T/E
		# TypeVars) - confirmed via a real repro ("incomplete type 'union
		# builtins$Result$data'" from clang). union_storage needs THIS
		# instantiation's own concrete, monomorphized union instead - same
		# "monomorphize_class if Specialization else itself" pattern
		# _coerce_or_check_operand already uses for the identical reason.
		concrete_union = self.lowering.monomorphize_class( raw_temp.type ) if isinstance( raw_temp.type, Specialization ) else raw_temp.type
		tag_attr, data_attr, payload_cls, tags = self.lowering._union_storage.get( concrete_union )
		ok_member, err_member = members[0], members[1]
		bool_cls = self.lowering.discovery.find_name( 'bool', node )
		err_label = self._new_label( 'binop_result_err' )
		tag_dest = self._new_temp( tag_attr.type )
		self._emit( ir.GetAttr( dest = tag_dest, obj = raw_temp, attr = tag_attr.stem ))
		match = self._new_temp( bool_cls )
		self._emit( ir.Cmp( dest = match, op = ir.CmpOp.EQ, left = tag_dest, right = ir.Const( type = tag_attr.type, value = tags[ok_member.stem] )))
		self._emit( ir.JumpIfFalse( cond = match, target = err_label ))
		# Ok branch
		ok_payload = self._extract_union_payload( raw_temp, data_attr, payload_cls, ok_member )
		ok_value = self._coerce_binop_value( ok_payload, success_type, result_union, node )
		self._finish_binop_result_branch( branch_start, raw_temp, ok_value, dest, end_label )
		self._emit( ir.Label( name = err_label ))
		# Err branch - err_member's own type might ITSELF be a multi-member
		# ANONYMOUS union (signed Div/Mod's own ZeroDivisionError|
		# OverflowError - see _resolve_checked_error) - one more bounded
		# nested unwrap to reach a concrete leaf class before coercing
		# into the OUTER error union. Gated on file is None (the same
		# "synthesized, not a real declared type" marker
		# discovery._get_or_create_union's own flattening logic already
		# uses) - a NOMINAL @union error type (e.g. a real `@union class
		# IntError: DivideByZero: None; ...`) is just as much an opaque
		# LEAF as any plain marker class, exactly like a nominal @union
		# leaf flowing into a WIDER union elsewhere in this compiler (see
		# _coerce_into_union's own "nominal @union is exactly as valid a
		# member... as any plain leaf type" comment) - unwrapping ITS OWN
		# variants here would be wrong, confirmed via a real repro
		# (int.__add__'s own Result[int,IntError] wrongly tried to
		# decompose IntError's OWN internal DivideByZero/... variants
		# instead of treating the whole IntError value as one leaf)
		err_payload = self._extract_union_payload( raw_temp, data_attr, payload_cls, err_member )
		err_shape = self.lowering._type_resolver._tagged_union_shape( err_payload.type )
		if err_shape is not None and err_shape[0].file is None and len( err_shape[1] ) > 1:
			self._emit_nested_error_unwrap( node, branch_start, err_payload, err_shape, raw_temp, dest, error_type, result_union, end_label )
		else:
			err_value = self._coerce_binop_value( err_payload, error_type, result_union, node )
			self._finish_binop_result_branch( branch_start, raw_temp, err_value, dest, end_label )

	def _emit_nested_error_unwrap(
		self, node: ast.AST, branch_start: int, err_union_operand: ir.Operand, err_shape: tuple[TaggedUnion,list[Variable]],
		raw_temp: ir.Temp, dest: ir.Temp, error_type: Type, result_union: TaggedUnion, end_label: str,
	) -> None:
		''' one leaf pair's own checked error type can itself be a
		multi-member anonymous union (signed Div/Mod's ZeroDivisionError|
		OverflowError) - unwraps it down to the one concrete leaf class
		before coercing into the OUTER (already-flattened, individual-
		classes) error union. Bounded to exactly this one extra level -
		_resolve_checked_error never produces a nested union of unions. '''
		base, members = err_shape
		tag_attr, data_attr, payload_cls, tags = self.lowering._union_storage.get( base )
		bool_cls = self.lowering.discovery.find_name( 'bool', node )
		n = len( members )
		for i, member in enumerate( members ):
			is_last = ( i == n - 1 )
			if not is_last:
				next_label = self._new_label( 'binop_error_unwrap_next' )
				tag_dest = self._new_temp( tag_attr.type )
				self._emit( ir.GetAttr( dest = tag_dest, obj = err_union_operand, attr = tag_attr.stem ))
				match = self._new_temp( bool_cls )
				self._emit( ir.Cmp( dest = match, op = ir.CmpOp.EQ, left = tag_dest, right = ir.Const( type = tag_attr.type, value = tags[member.stem] )))
				self._emit( ir.JumpIfFalse( cond = match, target = next_label ))
			concrete = self._extract_union_payload( err_union_operand, data_attr, payload_cls, member )
			value = self._coerce_binop_value( concrete, error_type, result_union, node )
			self._finish_binop_result_branch( branch_start, raw_temp, value, dest, end_label )
			if not is_last:
				self._emit( ir.Label( name = next_label ))

	def _lower_operand_compare( self, left: ir.Operand, right: ir.Operand, negate: bool, node: ast.AST ) -> ir.Operand:
		''' Eq/NotEq between two ALREADY-LOWERED operands of the SAME
		(non-union) type - the same dunder-or-flat-Cmp choice _lower_eq_or_ne
		makes from AST nodes, reimplemented against operands directly since
		_build_union_leaf_eq's own narrowed payload has no AST node of its
		own to re-dispatch through (mirrors _coerce_or_check_operand's
		identical "no node to re-evaluate" posture). '''
		bool_cls = self.lowering.discovery.find_name( 'bool', node )
		method_name = '__ne__' if negate else '__eq__'
		# _find_dunder_for_arg, not the plain _find_method - both
		# operands are already known to share the SAME type (this
		# method's own docstring), so the wanted implementation is
		# whichever one declares its own parameter as exactly left.type.
		# NOT gated on isinstance(left.type, Scalar) anymore - a Scalar-
		# registered __eq__/__ne__ resolves identically here (see lib/
		# builtins/__scalar_dunders.py), same precedent as everywhere else
		# this file's comparison dispatch dropped that gate.
		method = self._find_dunder_for_arg( left.type, method_name, left.type )
		if method is not None:
			# _emit_fallible_method_call, not a hand-rolled ir.Call - see
			# _expr_Compare's own identical comment on why (Scalar-
			# registered dunder receiver threading)
			return self._emit_fallible_method_call( node, method, left, [ right ], None )
		# no matching dunder - a hard compile error, not a silent flat-Cmp
		# fallback - see _expr_Compare's own identical comment for the full
		# rationale (confirmed with the user: no sensible default exists for
		# comparing two arbitrary values). No source AST node for this
		# specific comparison (this method's own docstring - a narrowed
		# union-leaf payload has none of its own), so `node` here is
		# whatever the caller passed for error-reporting purposes only.
		type_name = left.type.qualname if left.type is not None else '?'
		self.lowering.discovery.fail( f'{type_name} has no {method_name}() defined - comparison requires an explicit dunder', node )
		dest = self._new_temp( bool_cls )
		self._emit( ir.Cmp( dest = dest, op = ir.CmpOp.NE if negate else ir.CmpOp.EQ, left = left, right = right ))
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

	def _lower_in_comparison( self, node: ast.Compare, negate: bool ) -> ir.Operand:
		# `x in y` / `x not in y` mean `y.__contains__(x)` (negated for
		# NotIn) - the REVERSE of every other _COMP_DUNDER-driven comparison
		# (==, <, ...), where the LEFT operand is always the receiver. That
		# reversal is exactly why In/NotIn can't just be added as two more
		# _COMP_DUNDER entries and fall through the generic left-operand
		# dispatch above: this lowers the RIGHT operand first and dispatches
		# on ITS type instead.
		right = self._lower_expr( node.comparators[0], None )
		bool_cls = self.lowering.discovery.find_name( 'bool', node )
		if not isinstance( right.type, Scalar ):
			method = self.lowering._find_method( right.type, '__contains__' )
			if method is not None:
				self.lowering._ensure_resolved( method )
				self.lowering.schedule( method.return_type )
				for p in ( method.parameters or [] ):
					self.lowering.schedule( p.type )
				param_type = method.parameters[0].type if method.parameters else None
				left = self._lower_expr( node.left, param_type )
				call_dest = self._new_temp( method.return_type )
				self._emit( ir.Call( dest = call_dest, target = method, receiver = right, args = [ left ], kwargs = {} ))
				if not negate:
					return call_dest
				# NotIn: negate __contains__'s plain bool result - ir.Not
				# (same as _expr_UnaryOp's `not x`), NOT
				# _lower_is_comparison's tagged-union-aware EQ/NE flip,
				# which solves an unrelated problem (`is None` narrowing)
				dest = self._new_temp( bool_cls )
				self._emit( ir.Not( dest = dest, operand = call_dest ))
				return dest
		# no __contains__ on a non-scalar right operand, or a scalar right
		# operand entirely (e.g. `x in 5`) - unlike ==, there's no sane
		# degraded fallback (a raw pointer/value compare is never what `in`
		# means), so this is a hard error rather than a silent Cmp fallback
		self.lowering.discovery.fail(
			f'{"not " if negate else ""}in requires a __contains__ method on '
			f'{right.type.qualname if right.type else "?"}: {ast.unparse(node)}',
			node,
		)

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
			direct = base.get_local_or_raise( func_node.attr )
			if not isinstance( direct, ( Function, Overload )):
				return self.lowering._type_resolver._resolve_union_receiver_members( base, members, func_node.attr, func_node ), receiver
		target = self.lowering._type_resolver._attr_lookup_callable( receiver.type, func_node.attr, func_node )
		return target, self._maybe_deref_arrow_receiver( receiver, target )

	def _maybe_deref_arrow_receiver( self, receiver: ir.Operand, target: Function|Overload|Specialization|_ReceiverDispatch ) -> ir.Operand:
		''' `p.method()` where p: Ptr[T]/ConstPtr[T] is arrow-sugar -
		_attr_lookup_callable already redirected the NAME LOOKUP to T's own
		method, but `receiver` above is still evaluated against p's own,
		un-redirected Ptr[T] type. A plain (non-interface) CStruct or an
		RCClass method's self expects a real T value (T's own by-value
		struct, or the RCClass's own bare pointer respectively - see
		lower_function's own self_param construction), not the raw Ptr[T] -
		dereference it here, the same ir.GetItem the ordinary `p[0]`
		subscript uses (Ptr[T]/ConstPtr[T]'s own dereference convention).
		An @interface CStruct's self is ALWAYS Ptr[T] itself though - there
		the pointer already IS what self expects, so this leaves receiver
		untouched (confirmed: dereferencing there produced invalid C -
		passing a by-value struct where the generated prototype declares a
		pointer). '''
		if not self.lowering._type_resolver._is_ptr_specialization( receiver.type ):
			return receiver
		target_cls = getattr( target, 'cls', None )
		if target_cls is None or ( isinstance( target_cls, CStruct ) and target_cls.is_interface ):
			return receiver
		index_type = self.lowering.discovery.get_intrinsics()['usize']
		index = ir.Const( type = index_type, value = 0 )
		dest = self._new_temp( receiver.type.args[0] )
		self._emit( ir.GetItem( dest = dest, obj = receiver, index = index ))
		return dest

	def _apply_move_hook( self, param: Parameter, operand: ir.Operand, target_qualname: str ) -> None:
		# the semantic half of move[T] - _check_move_argument (run earlier,
		# inside _match_call_args) already validated the call-site syntax
		# agrees; this is where the argument's OWN ownership state actually
		# transitions, once its real Operand exists (needs the lowered
		# value, not just the AST expr) - shared by every _match_call_args
		# caller (plain calls, both generic call flavors, union-receiver
		# dispatch), called right after each argument is lowered
		if param.is_move:
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
			# a stub (fn.bound_to is not None) is "signature-only, never
			# actually called" (see mpy_types.Overload's own docstring) - the
			# real, emitted call always targets its bound_to implementation,
			# whose OWN parameter type is what the literal argument actually
			# needs to satisfy at the C level, not the stub's own (often
			# narrower) declared one. Without this redirect, a stub bound to
			# a WIDER implementation (e.g. Result[T,E].unwrap_or's own
			# `default: T` stub, bound to the plain `default: T|None = None`
			# impl) contributed its own narrower scalar type as a SEPARATE
			# candidate alongside the impl's own wider union type - two
			# genuinely different Type objects for what is, at the real call
			# site, the exact same parameter slot - which the magnitude-based
			# disambiguation just below (correctly designed for choosing
			# between truly independent overload arms, e.g. i8 vs i32) then
			# resolved by picking the narrower SCALAR one, since only a
			# Scalar passes its own isinstance(t, Scalar) filter. The literal
			# was then lowered against that narrower type while the actual,
			# real call still targets the wider union-typed implementation -
			# a genuine "passing 'int' to parameter of incompatible type
			# 'struct $__u$$...'" C mismatch, confirmed via a real compile of
			# Result[i32,str]'s own .unwrap_or(5). Redirecting first makes a
			# bound stub and its own implementation contribute the SAME
			# (impl's) type, deduping to one real candidate - matching what
			# actually gets called
			real_fn = fn.bound_to if fn.bound_to is not None else fn
			if real_fn.parameters is None:
				continue
			param = (
				real_fn.parameters[position] if position is not None and position < len( real_fn.parameters ) else
				next( ( p for p in real_fn.parameters if p.stem == kw_name ), None )
			)
			if param is None or param.type is None:
				continue
			if getattr( param.type, 'stem', None ) in compatible_stems:
				matched_type = param.type
			else:
				# not DIRECTLY a compatible scalar - but a union-typed param
				# (e.g. Result[T,E].unwrap_or's own `default: T|None`, once T
				# itself substitutes to a scalar) can still unambiguously
				# accept this literal, through exactly one of its own leaves.
				# Without this, a union-typed candidate was always silently
				# skipped here regardless of whether it fit, so a literal
				# argument to an overloaded call never got the union-
				# coercion _lower_expr's own expected_type machinery
				# (_coerce_into_union) already does correctly for an ORDINARY
				# (non-overloaded) call - the literal fell through to the
				# "no candidate's parameter type is even plausible" path
				# below, lowered with expected_type=None, and reached
				# emitter_c as a bare scalar handed to a C parameter whose
				# real type is the whole union struct: a genuine, confirmed
				# "passing 'int' to parameter of incompatible type 'struct
				# $__u$$...'" C compile error (Result[i32|None,str].
				# unwrap_or(42), a fallible generator's own g.__next__().
				# unwrap_or(default) - any T|None-shaped default param at
				# all). len(...)==1 (not >=1) mirrors this method's own
				# existing ambiguity discipline just below: a literal that
				# plausibly fits more than one leaf of the SAME union is
				# exactly as ambiguous as fitting more than one candidate
				# scalar directly would be, and is left for the ordinary
				# ambiguous-candidate error path rather than silently
				# guessing one.
				#
				# the matched LEAF itself (not the whole union) is what gets
				# added as this candidate's type below - lowering the literal
				# with the union as its expected_type would make the
				# resulting operand's own STATIC type the whole union (both
				# members "possible" as far as any TYPE-based reasoning can
				# tell), even though a fresh literal's value is obviously,
				# unambiguously never the None variant. overload_resolution.
				# resolve_call is a pure function of TYPES with no way to see
				# that extra fact - an operand whose type is the whole union
				# made it build a genuine, unnecessary RUNTIME conditional
				# dispatch (checking the literal's own freshly-constructed
				# tag at runtime, only ever one way) instead of a single
				# unconditional target, and that dispatch path (_emit_
				# dispatch_call) has no notion of a bound method RECEIVER at
				# all - confirmed via a real compile of Result[i32,str]'s own
				# .unwrap_or(5): a bound method call's own `self` argument
				# went missing from the emitted C call entirely. Using the
				# narrow leaf here instead keeps the operand's own static
				# type exactly as unambiguous as the literal itself is, and
				# the mismatch against the real (union-typed) parameter this
				# eventually needs to satisfy is resolved afterward, once the
				# winning target is actually known - see the Overload
				# dispatch's own post-resolution coercion step, below
				leaves = param.type.leaves()
				compatible_leaves = [ leaf for leaf in leaves if getattr( leaf, 'stem', None ) in compatible_stems ]
				if len( compatible_leaves ) != 1:
					continue
				matched_type = compatible_leaves[0]
			if not any( t is matched_type for t in candidate_types ):
				candidate_types.append( matched_type )

		if len( candidate_types ) > 1 and type( expr.value ) is int:
			# kind alone left more than one candidate (e.g. i8 AND i32 both
			# accept an int literal) - narrow further by whether the
			# literal's own MAGNITUDE actually fits each candidate's real
			# range (f(300) between f(x: i8)/f(x: i32) has only one answer,
			# not an ambiguity). Only ever NARROWS candidate_types when this
			# lands on exactly one match - if it eliminates every candidate,
			# or still leaves more than one (genuinely ambiguous even by
			# magnitude, e.g. two same-range types), candidate_types is left
			# untouched and the existing ambiguous/fallback paths below are
			# completely unaffected
			in_range = [ t for t in candidate_types if isinstance( t, Scalar ) and int_stem_range( t )[0] <= expr.value <= int_stem_range( t )[1] ]
			if len( in_range ) == 1:
				candidate_types = in_range
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

	def _infer_allocate_type_args(
		self, target_cls: ClassLike, class_type_params: list[TypeVar],
		fields_to_build: dict[str,Variable], fields: dict[str,ir.Operand], node: ast.Call, label: str,
	) -> Specialization:
		# _lower_allocate_fields's no-__init__ field=value sugar has no
		# _lower_generic_construction_args-style argument-based inference of
		# its own (that path infers a generic RCClass's own type args from
		# __init__'s parameters, independent of expected_type) - called only
		# once expected_type has already been ruled out as usable (absent, or
		# - PLAN_RETURN_INFERENCE.md's eager-lowering passes, whose own
		# return_type is deliberately still None while lowering their body -
		# untrustworthy/incompatible, same "rooted at target_cls" discipline
		# the sibling `compatible` check just above this call's own use
		# applies). The fields' own real, already-lowered VALUES
		# (fields[name].type) carry exactly the concrete types needed -
		# unify each field's declared (possibly bare-TypeVar) type against
		# its own real value type, same _unify_type_param every OTHER
		# generic call site already uses. Total/safe on shapes it doesn't
		# recognize (e.g. a TaggedUnion's synthesized tag/data view isn't
		# built from target_cls's own type params in any directly-unifiable
		# way) - it just no-ops rather than crashing, so this applies
		# uniformly across RCClass/CStruct/CUnion/TaggedUnion without
		# needing to special-case any of them out
		bindings: dict[int,Type] = {}
		for name, field in fields_to_build.items():
			self.lowering._unify_type_param( class_type_params, field.type, fields[name].type, bindings, node, target_cls.qualname )
		missing_type_params = [ tv.stem for tv in class_type_params if id( tv ) not in bindings ]
		if missing_type_params:
			self.lowering.discovery.fail(
				f'{target_cls.qualname}{label}: cannot infer type parameter(s) {", ".join(missing_type_params)} '
				f'from these field values or the surrounding expected type: {ast.unparse(node)}',
				node,
			)
		concrete_args = [ bindings[id(tv)] for tv in class_type_params ]
		return self.lowering.discovery._get_or_create_specialization( target_cls, concrete_args )

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
				tag_field = concrete_union.get_local_or_raise( 'tag' )
				data_field = concrete_union.get_local_or_raise( 'data' )
			else:
				tag_field = target_cls.get_local_or_raise( 'tag' )
				data_field = target_cls.get_local_or_raise( 'data' )
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
			for instr in self._cfg.field_value( value.type, value, is_alias = self.lowering._is_aliasing_expr( expr, value )):
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
			interface_class_type_params = target_cls.type_params or []
			fallback_cls: ClassLike|Specialization = target_cls
			if expected_type is None and interface_class_type_params:
				fallback_cls = self._infer_allocate_type_args( target_cls, interface_class_type_params, fields_to_build, fields, node, label )
			ptr_type = self.lowering.discovery._get_or_create_specialization( ptr_cls, [ fallback_cls ] )
			dest = self._new_temp( expected_type or ptr_type )
			self.lowering.schedule( dest.type )
			self.lowering._schedule_interface_construction( target_cls )
			self._emit( ir.Allocate( dest = dest, cls = target_cls, fields = fields ))
			return dest

		# expected_type is only trustworthy as dest's type when it's actually
		# ROOTED AT target_cls (itself, a Specialization of it, or - inside a
		# generic class's own method body, e.g. this exact __allocate__ call
		# from Result's own synthesized Err/Ok constructor - the MONOMORPHIZED
		# concrete class fn_cls's own Specialization already substitutes to;
		# substitute_type_params eagerly monomorphizes a ClassLike-shaped
		# type param instead of leaving it wrapped in a Specialization - see
		# monomorphize_class's own "_building" comment - so a generic method's
		# `self._current_fn.return_type` surfaces here already as that bare,
		# concrete ClassLike, not a Specialization wrapping target_cls, even
		# though fn_cls (this same call's own substituted class context,
		# already used just above to resolve tag_field/data_field/declared)
		# IS a Specialization of target_cls). Otherwise, expected_type can be
		# hinted from an unrelated surrounding union context (a generic
		# method call's own type-param unification substituting E in
		# `Result.Err(ParseError(...))` to the inferred `ParseError|OtherError`
		# union BEFORE this argument is even lowered - see
		# _lower_and_infer_call_args). Blindly trusting it there produced a
		# self-inconsistent ir.Allocate (cls=ParseError but dest.type=the
		# union), silently bypassing _lower_expr's own union-coercion check
		# (operand.type is expected_type by accidental identity) and crashing
		# the emitter's isinstance(concrete_cls, RCClass) assert. Mirrors the
		# same "pinning_type.base is target_cls" discipline
		# _lower_generic_construction_args already applies before trusting
		# expected_type for type-param inference.
		compatible = ( expected_type is target_cls
			or ( isinstance( expected_type, Specialization ) and expected_type.base is target_cls )
			or ( isinstance( fn_cls, Specialization ) and fn_cls.base is target_cls
				and expected_type is self.lowering.monomorphize_class( fn_cls ) ))
		# when expected_type isn't usable, falling back to the bare ABSTRACT
		# target_cls (unspecialized type_params and all) is only actually
		# correct when target_cls isn't generic in the first place - see
		# _infer_allocate_type_args. TaggedUnion is included here even though
		# _infer_allocate_type_args can never actually bind anything through
		# it (this branch's own `declared` is always {tag, data} - tag_field.
		# type is a plain intrinsic and data_field.type is a synthesized
		# anonymous CUnion whose own type_params is never set by union_
		# storage.py's UnionStorage.get(), so neither shape is one
		# _unify_type_param can recurse through). That's a harmless no-op
		# today, not a live gap: every REAL generic-union construction goes
		# through union_storage.py's synthesized per-member constructor,
		# whose own fn_cls/expected_type relationship always satisfies
		# `compatible` above BEFORE inference would ever run. If some future
		# path ever did reach here uncompatible, _infer_allocate_type_args's
		# own "cannot infer type parameter(s)" fits this function's
		# established fail-loudly-not-silently-wrong discipline - which is
		# exactly why TaggedUnion stays in this isinstance check rather than
		# being carved out back to the bare-abstract-class fallback
		class_type_params: list[TypeVar] = (
			target_cls.type_params or []
		) if isinstance( target_cls, ( RCClass, CStruct, CUnion, TaggedUnion )) else []
		fallback_cls: ClassLike|Specialization = target_cls
		if not compatible and class_type_params:
			fallback_cls = self._infer_allocate_type_args( target_cls, class_type_params, fields_to_build, fields, node, label )
		dest = self._new_temp( expected_type if compatible else fallback_cls )
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

	def _try_lower_generator_allocate_call( self, node: ast.Call, expected_type: Type|None ) -> ir.Temp|None:
		# PLAN_GENERATORS.md - Lowering._rewrite_generator_constructor's own
		# synthesized `BackingClass(__state=0, ...)` call, tagged directly
		# with the target RCClass object itself (generator_backing_cls)
		# rather than something resolvable by name through any real scope -
		# same escape-hatch spirit as node.resolved_callee/resolved_
		# construction elsewhere in this file, just for a class instead of
		# a function.
		target_cls = getattr( node, 'generator_backing_cls', None )
		if target_cls is None:
			return None
		return self._lower_allocate_fields( target_cls, node, expected_type, '(...)' )

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
		# explicit generic-class subscript - ClassName[T].__allocate__(...),
		# including a generic class's own method re-applying its OWN type
		# parameter to itself (the bare ClassName.__allocate__(...) spelling
		# already resolves this correctly via in-scope T lookup - this is
		# the same call, just reached through an explicit, redundant [T]).
		# Without this, _try_resolve_namespace's own Subscript branch
		# already correctly resolves ClassName[T] to a Specialization, but
		# this recognizer only accepted a real ClassLike, declining
		# (returning None) and falling through to ordinary receiver-based
		# call resolution - which lowers ClassName[T] as a VALUE expression
		# instead of a type reference, and ClassName (a class, not a
		# Variable/Function) fails there with "'ClassName' is not a value,
		# cannot use it as an expression".
		# Unwrap to .base (the abstract template) rather than monomorphizing
		# to the concrete specialization - _lower_allocate_fields's own
		# comment just below documents that target_cls must always be the
		# ABSTRACT class (it separately substitutes field types against
		# self._current_fn.cls, the concrete specialization, when one's
		# available); handing it an already-monomorphized target_cls here
		# instead breaks that substitution AND in_private_scope's identical
		# "self is always the abstract template" contract just below
		if isinstance( target_cls, Specialization ):
			target_cls = target_cls.base
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

		# T(...) where T defines a static __call__ dispatches to
		# T.__call__(...) instead of construction - rewrite node.func to
		# Attribute(T, '__call__') in place and bail out (returning None
		# here just makes this recognizer decline, same as any other
		# shape mismatch): the remaining recognizers below in _lower_call's
		# own tuple all safely no-op against an Attribute ending in
		# '__call__' (none matches that shape), and _resolve_callee's
		# existing ClassName.static_method(...) path (already used by
		# int.from_str(...)) picks the rewritten node.func up unchanged
		# once the recognizer loop falls through - no new call-resolution
		# logic needed downstream. Reuses target_cls (already resolved just
		# above, including the generic-Specialization case) rather than a
		# second _try_resolve_namespace call - same reasoning as this
		# whole check's own placement, right after that resolution and
		# before the CEnum branch: no-op for the overwhelming common case
		# (target_cls isn't ClassLike, or has no local __call__), so every
		# OTHER class's construction stays byte-for-byte unchanged.
		if isinstance( target_cls, ClassLike ):
			call_member = target_cls.get_local( '__call__' )
			if call_member is not None:
				members = call_member.implementations if isinstance( call_member, Overload ) else [ call_member ]
				if not all( isinstance( m, Function ) and m.is_static for m in members ):
					self.lowering.discovery.fail(
						f'{target_cls.qualname}.__call__ must be declared @staticmethod to be used as {target_cls.qualname}(...): {ast.unparse(node)}',
						node,
					)
					return None
				new_func = ast.Attribute( value = node.func, attr = '__call__', ctx = ast.Load() )
				ast.copy_location( new_func, node.func )
				node.func = new_func
				return None

		# CEnum construction: EnumName(value) is a plain cast to the
		# enum's underlying type — no allocation, no refcounting, just
		# reinterpret the raw integer as the enum type. e.g. OSError(ENOENT)
		if isinstance( target_cls, CEnum ):
			if len( node.args ) != 1 or node.keywords:
				self.lowering.discovery.fail( f'{target_cls.qualname}(...) takes exactly one positional argument: {ast.unparse(node)}', node )
			self.lowering._ensure_resolved( target_cls )
			# a literal argument's magnitude must fit the enum's own
			# underlying type's real range - this is CONSTRUCTION, not a
			# cast, so _lower_scalar_cast's bit-reinterpretation exemption
			# (_allow_literal_bit_reinterpret) doesn't apply here; an out-
			# of-range value is a genuine mistake, unlike u32(-11)'s
			# deliberate WinAPI-style reinterpretation
			arg_node = node.args[0]
			value_type = target_cls.value_type
			if isinstance( arg_node, ast.Constant ) and type( arg_node.value ) is int and isinstance( value_type, Scalar ):
				lo, hi = int_stem_range( value_type )
				if not ( lo <= arg_node.value <= hi ):
					self.lowering.discovery.fail(
						f'{arg_node.value} is out of range for {target_cls.qualname} ({lo}..{hi}): {ast.unparse(node)}',
						node,
					)
			# lower the argument directly — no arithmetic-mode semantics
			# needed here; a CEnum has exactly the same runtime
			# representation as its underlying type, so OSError(42) is
			# just the value 42 with the enum type.
			#
			# The argument's own natural type space is value_type (the
			# underlying scalar), NEVER target_cls (the enum itself) - passing
			# target_cls down as _lower_expr's expected_type here (the
			# pre-fix code) was a category error that only ever incidentally
			# "worked" for a couple of argument shapes, not because it was
			# correct: a bare ast.Name/Variable operand (_expr_Name) ignores
			# expected_type entirely and keeps its own declared type, so
			# _check_assignable's later CEnum<->value_type exemption
			# correctly ran and correctly rejected a genuine kind mismatch
			# (e.g. i32 for a u32-backed enum) - but a Call/BinOp argument's
			# own result-typing tail (_lower_call's final dest allocation,
			# `expected_type or target_return_type`) instead SILENTLY
			# relabeled the destination temp's own type to target_cls
			# directly, with no check at all that the callee's real return
			# type was even a scalar, let alone value_type - confirmed via a
			# real repro: OSError(get_rc()) (get_rc() -> i32) and
			# OSError(rc + 0) both compiled silently, while the exact same
			# value through a bare local, OSError(rc), was rejected as "rc:
			# expected builtins.OSError, got intrinsics.i32" - an
			# inconsistency across argument SHAPE, not a real distinction in
			# what's being constructed. Fixed by routing every shape through
			# the SAME real scalar-cast machinery T(x)/compiler.cast(T,x)
			# already use (_lower_scalar_cast) - value_type is authoritative
			# (matching this construction's own "plain cast to the
			# underlying type" contract), any argument expression a plain
			# scalar cast would accept is accepted here too, and the result
			# is then relabeled (CastWrap, a zero-cost same-bits retag, same
			# as an RCClass upcast/safe scalar widening/pointer interchange
			# elsewhere in this file) to target_cls.
			assert isinstance( value_type, Scalar ), f'{target_cls.qualname}: CEnum value_type must be a scalar, got {value_type!r}'
			if isinstance( arg_node, ast.Constant ) and type( arg_node.value ) is int:
				# the already-range-checked literal fast path: unchanged from
				# before this fix - _expr_Constant already folds a CEnum-
				# expected int literal straight into an ir.Const tagged with
				# target_cls directly (kind+magnitude already validated, here
				# and in _expr_Constant itself), no runtime Temp/CastWrap
				# needed. Keeping this path bypassed by the general fix below
				# preserves that constant-folding (real callers/tests rely on
				# a literal CEnum construction lowering to a bare ir.Const,
				# not a cast instruction).
				return self._lower_expr( node.args[0], target_cls )
			# every NON-literal shape (bare Name, Call, BinOp, ...): the
			# argument's own natural type space is value_type (the underlying
			# scalar), NEVER target_cls (the enum itself) - passing target_cls
			# down as _lower_expr's expected_type here (the pre-fix code, for
			# every argument shape) was a category error that only ever
			# incidentally "worked" for a couple of shapes, not because it was
			# correct: a bare ast.Name/Variable operand (_expr_Name) ignores
			# expected_type entirely and keeps its own declared type, so
			# _check_assignable's later CEnum<->value_type exemption correctly
			# ran and correctly rejected a genuine kind mismatch (e.g. i32 for
			# a u32-backed enum) - but a Call/BinOp argument's own result-
			# typing tail (_lower_call's final dest allocation, `expected_type
			# or target_return_type`) instead SILENTLY relabeled the
			# destination temp's own type to target_cls directly, with no
			# check at all that the callee's real return type was even a
			# scalar, let alone value_type - confirmed via a real repro:
			# OSError(get_rc()) (get_rc() -> i32) and OSError(rc + 0) both
			# compiled silently, while the exact same value through a bare
			# local, OSError(rc), was rejected as "rc: expected
			# builtins.OSError, got intrinsics.i32" - an inconsistency across
			# argument SHAPE, not a real distinction in what's being
			# constructed. Fixed by lowering the argument against value_type
			# (strict=False - a HINT only, e.g. so an untyped nested literal
			# still settles on the right width; the real, authoritative check
			# that the argument is even scalar-shaped happens explicitly
			# below) and relabeling via CastWrap - a plain `(ctype)(operand)`
			# C cast (see emitter_c.py's _emit_cast 'wrap' mode), exactly C's
			# well-defined integer conversion rules, safe for the real cross-
			# signedness/width reinterpretation this construction call
			# promises, the same as an explicit T(x) scalar cast.
			arg_operand = self._lower_expr( node.args[0], value_type, strict = False )
			if not isinstance( arg_operand.type, Scalar ):
				self.lowering.discovery.fail(
					f'{target_cls.qualname}(...) argument must be a scalar value, got '
					f'{arg_operand.type.qualname if arg_operand.type is not None else "?"}: {ast.unparse(node)}',
					node,
				)
			dest = self._new_temp( target_cls )
			self._emit( ir.CastWrap( dest = dest, operand = arg_operand ) )
			return dest

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
			init = target_cls.get_local_or_raise( '__init__' )
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

		# self_type is either already concrete (plain/resolved_construction
		# branches) or a generic-class Specialization (_lower_generic_
		# construction_args' own cls_spec) - _ensure_resolved is a no-op-
		# ish pass-through for an already-concrete class (same as
		# elsewhere in this file), so this one line handles both uniformly
		self.lowering.schedule( init.return_type )
		for param in init.parameters or []:
			self.lowering.schedule( param.type )
		if isinstance( self_type, Specialization ):
			# self_type was already resolved above (either by this method's
			# own earlier branches - target.schedule/_ensure_resolved(cls_spec)
			# in _lower_generic_construction_args - or by resolved_construction's
			# own pre-resolution) - re-calling _ensure_resolved would just
			# redundantly re-schedule() the same Specialization a second
			# time for no benefit (schedule()'s own id-based _seen dedup
			# makes it harmless, just wasted work) - .monomorphized is the
			# same cache Monomorphizer.monomorphize_class itself reads
			# (mpy_types.py's Specialization), a plain, side-effect-free
			# read of what's already there
			concrete_cls = self_type.monomorphized if self_type.monomorphized is not None else self.lowering._ensure_resolved( self_type )
		else:
			concrete_cls = self_type
		assert isinstance( concrete_cls, RCClass ), f'internal compiler error: {self_type=} did not resolve to a concrete RCClass'
		# synthesize (idempotent, memoized) rather than rely solely on the
		# compiler.py class-registration trigger - schedule() is a
		# deferred queue, so THIS call site needs $$__new__'s live Function
		# object available right now, not whenever it eventually gets
		# dequeued (see _synthesize_rcclass_constructor's own docstring)
		self.lowering._type_resolver._synthesize_rcclass_constructor( concrete_cls, init )
		new_fn = concrete_cls.get_local( '$$__new__' )
		assert isinstance( new_fn, Function ), f'internal compiler error: {concrete_cls.qualname} has no synthesized $$__new__'
		self.lowering.schedule( new_fn.return_type )
		dest = self._new_temp( new_fn.return_type )
		self._emit( ir.Call( dest = dest, target = new_fn, receiver = None, args = args, kwargs = kwargs ))
		return dest

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
		# strict=False on both: the substituted hint can still contain an
		# unbound TypeVar nested inside it (a class type param not yet
		# bound) - it's an inference HINT, not a validated requirement; the
		# REAL validation/binding is _unify_type_param below
		args = [
			self._lower_expr( expr, self.lowering._substitute_type_params( param.type, type_params, partial_args ), strict = False )
			for param, expr in positional
		]
		kwargs = {
			param.stem: self._lower_expr( expr, self.lowering._substitute_type_params( param.type, type_params, partial_args ), strict = False )
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

	def _narrowed_type_of_name( self, var_id: str, declared_type: Type ) -> Type:
		''' declared_type, unless var_id is currently narrowed (cfg.py's
		narrow(), e.g. inside an `if x is not None:`/`match x: case _:` arm) -
		then the narrowed member's own type instead. Shared by
		_try_lower_indirect_call/_try_lower_closure_call so a Ptr[Callable[...]]
		|None or Closure[...]|None parameter is recognized as callable once
		narrowed to its non-None leaf, not just when declared bare - mirrors
		_static_type_of_value_expr's identical narrowed_member lookup. '''
		member = self._cfg.narrowed_member( var_id )
		return member.type if member is not None else declared_type

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
		effective_type = self._narrowed_type_of_name( node.func.id, name.type )
		fn_type = self.lowering._type_resolver._callable_type_of( effective_type )
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
		closure_type = self._narrowed_type_of_name( node.func.id, name.type )
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
		if not want_result:
			# unwrapped's own extraction is bundled into OrReturn/OrJump's IR
			# shape (see _consume_checked_result) - can't be skipped even
			# though a bare `expr.or_return()` statement (validate-only,
			# error propagation is the only wanted effect) never reads it.
			# Non-RC-leaf receivers (e.g. Result[u8,E]) get no decref of their
			# own either, leaving a real -Wunused-but-set-variable - confirmed
			# suite-wide (lib/urllib/parse.py's own _unquote_impl first-pass
			# validation scan).
			self._emit( ir.MarkUsed( operand = unwrapped ))
			return None
		return unwrapped

	def _lower_call_args( self, target: Function, node: ast.Call, *, receiver_fills_first_param: bool = False ) -> tuple[list[ir.Operand],dict[str,ir.Operand]]:
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
		positional, keyword = self.lowering._match_call_args( target, node, receiver_fills_first_param = receiver_fills_first_param )
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
				# a construction call embedded in the default (`x: Foo =
				# Foo()`) needs the same eager __init__ pre-resolution an
				# ordinary body statement gets from _ReferenceResolver -
				# defaults live on fn.node.args, never walked by resolve_
				# function_body's fn.node.body loop, and are lowered here,
				# often before target's own turn on the compile queue ever
				# comes up - see resolve_parameter_default's own docstring
				self.lowering._type_resolver.resolve_parameter_default( target, param )
				# lowered in the CALLEE's own module/scope, not the
				# caller's (matching the identical field-default pattern
				# above in _lower_allocate_fields) - a default expression
				# can reference names visible where the function/class was
				# DEFINED, and errors inside it should be located there too
				with self.lowering.discovery.module_context( self.lowering._find_module_for( target )):
					with self.lowering.discovery.scope_context( target ):
						default_operand = self._lower_expr( param.default, param.type )
				kwargs[param.stem] = default_operand
		return args, kwargs

	def _lower_inline_call( self, node: ast.Call, target: Function, receiver: ir.Operand|None, args: list[ir.Operand], kwargs: dict[str,ir.Operand], expected_type: Type|None, want_result: bool ) -> ir.Operand|None:
		# PLAN_INLINE.md, generalized for multi-statement bodies - target.
		# is_inline: splice target's own body directly here instead of
		# ever emitting a real ir.Call. `args`/`kwargs` are already-lowered
		# operands (the caller already ran _lower_call_args, or the
		# interleaved generic lower_and_unify - same move-hook/argument-
		# lowering either way, only the tail differs). target may be a
		# plain Function, OR an already-monomorphized one (target.node was
		# deep-copied per Specialization by monomorphize.py - see its own
		# docstring), so target.node.body is always safe to read directly
		# here regardless of which caller reached this
		if id( target ) in self._inlining_stack:
			self.lowering.discovery.fail(
				f'@inline {target.qualname}: recursive inlining (directly or through another @inline function) is not supported: {ast.unparse(node)}',
				node,
			)
		if not want_result and cfg.is_result_type( target.return_type ):
			# same discard check the ordinary call tails already apply -
			# discovery.py's _is_inline_eligible_body already guarantees
			# target.node.body ends in exactly one `return <expr>`, so this
			# can't be sidestepped by inlining instead of calling for real
			self.lowering.discovery.fail(
				f'{target.qualname}(...) returns a Result that is discarded here - '
				f'assign it to a name and use .is_ok(), .is_err(), .or_return(), .unwrap(msg), or match: {ast.unparse(node)}',
				node,
			)
		stmts = target.node.body
		if stmts and isinstance( stmts[0], ast.Expr ) and isinstance( stmts[0].value, ast.Constant ) and isinstance( stmts[0].value.value, str ):
			stmts = stmts[1:] # strip a leading docstring, same shape discovery.py's _is_inline_eligible_body already validated
		# the reentrancy guard wraps the WHOLE call - both branches below,
		# not just the single-expression case's own return-expression -
		# so a recursive @inline call reached from a pre-return statement
		# in the multi-statement path is caught identically
		self._inlining_stack.append( id( target ))
		try:
			if len( stmts ) > 1:
				# the splice's own pre-return statements bind self/params
				# (and declare their own alpha-renamed locals) directly on
				# the shared self._cfg - unlike provisional.names (a fresh,
				# single-use, discarded Function), self._cfg._live is NOT
				# reverted internally by _splice_multi_statement_inline_body
				# itself, so it's done here instead: wholesale snapshot/
				# restore around the whole call - see set_live()'s own
				# docstring for why a full revert is safe here
				saved_live = self._cfg.live_snapshot()
				try:
					return self._splice_multi_statement_inline_body( node, target, receiver, args, kwargs, expected_type, want_result, stmts )
				finally:
					self._cfg.set_live( saved_live )

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
			saved_live: dict[str,bool] = {}
			bound_ids: set[int] = set() # see _incref_aliasing_return's own `force` doc - every self/parameter binding here is always treated as borrowed
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
				bound_ids.add( id( fresh ))
				saved[stem] = target.names.get( stem )
				target.names[stem] = fresh
				# liveness is keyed by `stem` (the literal 'self'/parameter
				# name the spliced body's own ast.Name nodes reference it
				# by, via target.names) NOT fresh.stem (which differs for
				# the synthesized-local fallback above, and even for the
				# zero-copy case is the CALLER's own variable's stem, e.g.
				# 'b' for a `b.get_len()` receiver, not 'self'). Bypasses
				# _cfg_assign like the rest of this binding deliberately
				# does (see this method's own "no _cfg_assign/incref here"
				# comment above) - still unconditionally bound right here.
				# Saved/restored the same shadow-and-restore way target.
				# names itself is, in case `stem` collides with an outer
				# name that wasn't actually live before this splice.
				saved_live[stem] = self._cfg.is_live( stem )
				self._cfg.mark_live( stem )

			return_expr = stmts[-1].value
			module = self.lowering._find_module_for( target )
			try:
				with self.lowering.discovery.module_context( module ):
					with self.lowering.discovery.scope_context( target ):
						result = self._lower_expr( return_expr, expected_type or target.return_type )
						self._incref_aliasing_return( return_expr, result, force = id( result ) in bound_ids )
			finally:
				for stem, was_live in saved_live.items():
					if not was_live:
						self._cfg.unmark_live( stem )
				for stem, old in saved.items():
					if old is None:
						target.names.pop( stem, None )
					else:
						target.names[stem] = old
			return result if want_result else None
		finally:
			self._inlining_stack.pop()

	def _splice_multi_statement_inline_body( self, node: ast.Call, target: Function, receiver: ir.Operand|None, args: list[ir.Operand], kwargs: dict[str,ir.Operand], expected_type: Type|None, want_result: bool, stmts: list[ast.stmt] ) -> ir.Operand|None:
		# PLAN_INLINE.md multi-statement generalization - target's own
		# body (already docstring-stripped by _lower_inline_call, the only
		# caller) has more than the single `return <expr>` statement the
		# original @inline design handled. discovery.py's _is_inline_
		# eligible_body already guarantees `stmts` ends in exactly one,
		# un-nested `return <expr>`, no other Return anywhere else in it,
		# no defer/errdefer, and no reassignment of self/a parameter
		# anywhere among the pre-return statements - this method doesn't
		# re-check any of that. The reentrancy guard was already pushed by
		# _lower_inline_call, covering this whole splice.
		bindings: dict[str,ir.Operand] = {} if receiver is None else { 'self': receiver }
		for i, param in enumerate( target.parameters or [] ):
			bindings[param.stem] = args[i] if i < len( args ) else kwargs[param.stem]

		# a fresh, per-call-site provisional Function - independent deep-
		# copied .node, independent .names dict, never touching `target`
		# itself (unlike the single-statement path's target.names
		# monkeypatch above - no save/restore needed anywhere in this
		# path, provisional is single-use, discarded once this call
		# returns). type_params=[]/args=[] is a no-op substitution: target
		# is already type-parameter-free by the time it reaches here
		# regardless of whether it was originally generic (monomorphize.py
		# already did that substitution before _lower_inline_call was ever
		# reached - see PLAN_RETURN_INFERENCE.md/monomorphize_function)
		provisional = self.lowering._monomorphizer._build_monomorphized_function( target, [], [], target.qualname )

		# run BEFORE alpha-renaming: match-statement desugaring (rewrite 2
		# - there is no _stmt_Match anywhere in this file, so a match
		# statement in a spliced body can only ever lower after this runs)
		# needs to see the ORIGINAL names (the alpha-renamer below only
		# understands plain ast.Name, never match-pattern capture shapes);
		# generic-call tagging (rewrite 3) is genuinely per-copy already -
		# needed for a nested generic call inside the pre-return
		# statements to resolve against THIS call site's own bindings, not
		# some other call site's
		self.lowering._type_resolver.resolve_function_body( provisional )

		provisional_stmts = provisional.node.body
		if provisional_stmts and isinstance( provisional_stmts[0], ast.Expr ) and isinstance( provisional_stmts[0].value, ast.Constant ) and isinstance( provisional_stmts[0].value.value, str ):
			provisional_stmts = provisional_stmts[1:]
		pre_return_stmts = provisional_stmts[:-1]
		return_stmt = provisional_stmts[-1]

		# alpha-rename every local the pre-return statements themselves
		# declare (a Store-context Name that isn't already self/a
		# parameter - those are bound via the names-dict substitution
		# below instead, never renamed here, since that mechanism still
		# keys off the literal original name) to a fresh, globally-unique
		# name, reusing the same $inline{id}$stem convention self/param
		# bindings already use (so the two can never collide). No
		# shadowing subtlety needed: this language has no block scoping
		# (cfg.py's own docstring: "structural, not a reference scan"),
		# and a nested def/lambda's own free variables are already
		# rejected elsewhere (PLAN_LAMBDA.md) - a flat, uniform rename
		# across every occurrence, Store and Load alike, is exactly
		# correct here, not an approximation. In-place mutation of ast.
		# Name.id suffices (no NodeTransformer needed) since provisional.
		# node is already a private, freshly-deep-copied-per-call-site
		# object - this only ever changes a string field, never
		# restructures the tree
		fl = self
		rename_map: dict[str,str] = {}
		class _LocalCollector( ast.NodeVisitor ):
			def visit_FunctionDef( self, fd: ast.FunctionDef ) -> None:
				pass
			def visit_AsyncFunctionDef( self, fd: ast.AsyncFunctionDef ) -> None:
				pass
			def visit_Lambda( self, lam: ast.Lambda ) -> None:
				pass
			def visit_Name( self, n: ast.Name ) -> None:
				if isinstance( n.ctx, ast.Store ) and n.id not in bindings and n.id not in rename_map:
					rename_map[n.id] = f'$inline{fl._inline_binding_id}${n.id}'
					fl._inline_binding_id += 1
		collector = _LocalCollector()
		for stmt in pre_return_stmts:
			collector.visit( stmt )

		if rename_map:
			class _LocalRenamer( ast.NodeVisitor ):
				def visit_FunctionDef( self, fd: ast.FunctionDef ) -> None:
					pass
				def visit_AsyncFunctionDef( self, fd: ast.AsyncFunctionDef ) -> None:
					pass
				def visit_Lambda( self, lam: ast.Lambda ) -> None:
					pass
				def visit_Name( self, n: ast.Name ) -> None:
					if n.id in rename_map:
						n.id = rename_map[n.id]
			renamer = _LocalRenamer()
			for stmt in pre_return_stmts:
				renamer.visit( stmt )
			renamer.visit( return_stmt ) # a later pre-return statement, or the return-expression itself, may reference an earlier pre-return-declared local

		# bind self/params into the PROVISIONAL's own names dict - same
		# logic the single-statement path above uses for target.names,
		# just no save/restore needed (provisional is single-use)
		bound_ids: set[int] = set() # see _incref_aliasing_return's own `force` doc - every self/parameter binding here is always treated as borrowed
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
			bound_ids.add( id( fresh ))
			provisional.names[stem] = fresh
			# liveness keyed by `stem` (see _lower_inline_call's own
			# identical single-statement-path comment) - the caller
			# (_lower_inline_call) reverts self._cfg's ENTIRE live set once
			# this whole splice returns, so no per-stem save/restore is
			# needed here, unlike provisional.names/that other path
			self._cfg.mark_live( stem )

		# early/nested-return + defer/errdefer/.or_return() generalization -
		# a splice-local "epilogue" scope for the pre-return statements: an
		# early return, or a .or_return()/checked-arithmetic early exit,
		# reached from one of them must never jump into/return from the
		# CALLER's own real epilogue - it needs its OWN local landing point.
		# result_var carries whichever value flowed through an early exit
		# (the splice-local analogue of self._return_value_var); exited_flag
		# (armed alongside it, same mechanism defer/errdefer's own flags
		# use) lets the tail below tell "early exit vs normal fallthrough"
		# apart once everything converges - see cfg.py's push_inline_scope()
		# and current_epilogue_label()/return_()'s own comments for the CFG
		# half of this, and ir.OrReturn.inline_exit/ir.OrJump.exited_flag
		# for how or_return()/checked-arithmetic feed into it
		# PLAN_RETURN_INFERENCE.md's own @inline variant (_infer_return_
		# only_type_params_inline) reaches here with target.return_type set
		# to Python None as a DELIBERATE SENTINEL (not none_type - the real
		# NoneType class), specifically so the trailing return-expression's
		# own _lower_expr(..., None) call can take its own natural type,
		# later read back via result.type to discover R. None of the new
		# early-exit machinery below can run in that state - result_var/
		# result would need a REAL type up front, which is exactly the one
		# thing not known yet. This is safe to skip entirely rather than
		# work around: _is_eager_return_inferable_body (the ONLY gate that
		# lets return-only inference even be attempted) already requires
		# EXACTLY ONE reachable return, so a body reaching here with this
		# sentinel can never have an early return to support in the first
		# place - only .or_return()/checked-arithmetic in a pre-return
		# statement remains a real (if narrow) hazard, still explicitly
		# rejected below, exactly as the single, blanket guard this
		# replaces always did for every multi-statement splice
		none_type = self.lowering.discovery.get_none_type()
		noreturn_type = self.lowering.discovery.get_intrinsics()['NoReturn']
		supports_early_exit = target.return_type is not None
		result_var: Variable|None = None
		exited_flag: Variable|None = None
		merge_label: str|None = None
		bool_cls: Type|None = None
		if supports_early_exit:
			result_var = (
				Variable(
					stem = f'$inline{self._inline_binding_id}$result', qualname = f'{target.qualname}$$inline{self._inline_binding_id}$result',
					file = target.file, line = target.line, type = target.return_type,
				)
				if target.return_type not in ( none_type, noreturn_type )
				else None
			)
			bool_cls = self.lowering.discovery.find_name( 'bool', node )
			exited_flag = Variable(
				stem = f'$inline{self._inline_binding_id}$exited', qualname = f'{target.qualname}$$inline{self._inline_binding_id}$exited',
				file = target.file, line = target.line, type = bool_cls,
			)
			self._inline_binding_id += 1
			merge_label = self._new_label( 'inline_merge' )

		module = self.lowering._find_module_for( target )
		with self.lowering.discovery.module_context( module ):
			with self.lowering.discovery.scope_context( provisional ):
				# the ONE place self._current_fn is ever reassigned in this
				# file - narrowly scoped to this window, restored in a
				# finally. Needed because _stmt_AnnAssign/_stmt_Assign's
				# fresh-declaration branch registers a new local into
				# self._current_fn (see their own code) while their
				# "already exists?" check instead goes through discovery.
				# find_name_or_none (walking discovery.scope_stack, which
				# module_context/scope_context above already point at
				# `provisional`) - without this reassignment those two
				# would disagree: a pre-return local would silently
				# register into the CALLER's own namespace (self._current_
				# fn, unless reassigned, stays whatever the caller's own
				# top-level function is - confirmed by grep, it's assigned
				# exactly once, in __init__, and never touched anywhere
				# else in this file), corrupting any later caller-side
				# reference to a same-named local; and reassigning that
				# same pre-return local a second time within the SAME
				# spliced body would fail to find its own first
				# registration, creating a second, independent binding
				# instead of a replace (a silent decref/leak, not just a
				# cosmetic issue - cfg.py's own fresh-vs-replace machinery
				# depends on finding the SAME Variable object both times).
				# Making self._current_fn and the active scope_context
				# point at the same `provisional` object for this whole
				# window fixes both at once.
				#
				# result_var/exited_flag are both given a real, flat,
				# unconditional declaration/init RIGHT HERE - before the
				# pre-return statements (and therefore before any .or_
				# return()/checked-arithmetic early exit nested inside
				# emitter_c.py's own hand-emitted C `{ }` blocks - see ir.
				# DeclareLocal's own docstring) could otherwise become
				# result_var's first, block-scoped-and-therefore-unsafe
				# write. exited_flag has a trivial default (False) an
				# ordinary ir.Assign already declares safely; result_var's
				# type has no generic default, hence DeclareLocal
				scope_label: str|None = None
				if supports_early_exit:
					assert exited_flag is not None and bool_cls is not None and merge_label is not None
					if result_var is not None:
						self._emit( ir.DeclareLocal( variable = result_var ))
					self._emit( ir.Assign( dest = exited_flag, src = ir.Const( type = bool_cls, value = False )))
					scope_label = self._cfg.push_inline_scope()
					self._inline_scope_vars.append(( result_var, exited_flag, merge_label ))
				outer_fn = self._current_fn
				outer_prelude = self._in_inline_splice_prelude
				self._current_fn = provisional
				self._in_inline_splice_prelude = True
				try:
					for stmt in pre_return_stmts:
						# mirrors FunctionLowering.run()'s own identical
						# per-statement recovery boundary - one bad
						# statement doesn't stop the rest of this splice
						# from being lowered (and error-collected)
						try:
							self._lower_stmt( stmt )
						except CompileError:
							continue
				finally:
					self._current_fn = outer_fn
					self._in_inline_splice_prelude = outer_prelude

				if not supports_early_exit:
					# PLAN_RETURN_INFERENCE.md's own @inline variant - see
					# this method's own top-of-function comment. No scope
					# was pushed, nothing to merge - the trailing return-
					# expression's own natural type IS the answer being
					# discovered here, exactly as the pre-existing
					# single-statement/original multi-statement code always
					# computed it
					result = self._lower_expr( return_stmt.value, expected_type )
					self._incref_aliasing_return( return_stmt.value, result, force = id( result ) in bound_ids )
					return result if want_result else None

				assert scope_label is not None and exited_flag is not None and merge_label is not None
				# current_epilogue_label()'s own fallback target once
				# nothing shallower within THIS splice qualified (push_
				# inline_scope()'s own label) - an inline-unwind return_()
				# call reached during the splice already replayed
				# everything itself and jumps straight past this, to
				# merge_label below (see _stmt_Return/_consume_checked_
				# result's own splice branches). Neither label is a real
				# jump target unless the splice body actually contained an
				# early exit reaching one of those two branches (a plain
				# multi-statement @inline body with none, e.g. Ptr.__str__,
				# never goes near either) - gated on InlineScope.captured,
				# same "don't declare a label nothing goto's" reasoning
				# build_epilogue_ladder() already uses for a real function's
				# own shared epilogue (a real, confirmed -Wunused-label
				# otherwise, suite-wide)
				if self._cfg.inline_scope_captured():
					self._emit( ir.Label( name = scope_label ))
				for instr in self._cfg.build_inline_scope_ladder( lambda: self._build_is_err_check( node )):
					self._emit( instr )
				was_captured = self._cfg.pop_inline_scope()
				self._inline_scope_vars.pop()

				# early exit vs normal fallthrough - both converge into ONE
				# result operand from here, same "shared dest temp, two
				# Assign sites, converge at one label" shape _expr_IfExp
				# already uses for Python's own ternary. self._current_fn/
				# _in_inline_splice_prelude are already restored to the
				# REAL caller above, before this point - the trailing
				# return-expression's own .or_return()/checked-arithmetic
				# behavior is therefore unchanged from the single-statement
				# case (validates and jumps against the CALLER's own
				# epilogue/return type, exactly as already tested), while
				# scope_context(provisional) stays active so it can still
				# resolve pre-return-declared locals it references
				if was_captured:
					self._emit( ir.Label( name = merge_label ))
				result = self._new_temp( target.return_type )
				normal_label = self._new_label( 'inline_normal' )
				converge_label = self._new_label( 'inline_converge' )
				self._emit( ir.JumpIfFalse( cond = exited_flag, target = normal_label ))
				if result_var is not None:
					self._emit( ir.Assign( dest = result, src = result_var ))
				self._emit( ir.Jump( target = converge_label ))
				self._emit( ir.Label( name = normal_label ))
				trailing_value = self._lower_expr( return_stmt.value, target.return_type )
				self._incref_aliasing_return( return_stmt.value, trailing_value, force = id( trailing_value ) in bound_ids )
				self._emit( ir.Assign( dest = result, src = trailing_value ))
				# trailing_value's own ownership (if it's a bare temp - e.g.
				# the Result.Ok(x) construction temp a trailing `return
				# Result.Ok(x)` produces) just transferred into `result`
				# above via the plain ir.Assign - untrack it, or whatever
				# later cleans up STILL-pending temps (_flush_pending_temps,
				# called by _lower_stmt's own post-statement wrapper once
				# this whole splice call returns) would emit a SECOND,
				# unconditional RC-check for it outside the "normal" arm's
				# own guard - reading trailing_value's memory even on the
				# early-exit path, where it was never assigned at all (a
				# real uninitialized-read bug, not just a redundant decref -
				# confirmed by a real repro under MSVC's /RTC1). Exactly the
				# same concern _stmt_Return's own identical transfer already
				# guards against via this same call
				self._cfg.untrack_temp( trailing_value )
				self._emit( ir.Label( name = converge_label ))
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
			# strict=False: `hint` can still contain an unbound TypeVar nested
			# inside it (e.g. Ptr[Callable[[i32],K]] with K not yet bound) -
			# it's an inference HINT here, not a validated requirement; the
			# REAL validation/binding is _unify_type_param below, not
			# _lower_expr's own general assignability check
			operand = self._lower_expr( expr, hint, strict = False )
			self.lowering._unify_type_param( type_params, param.type, operand.type, bindings, node, target.qualname )
			return operand

		args = [ lower_and_unify( param, expr ) for param, expr in positional ]
		kwargs = { param.stem: lower_and_unify( param, expr ) for param, expr in keyword }
		for ( param, _expr ), operand in zip( positional, args ):
			self._apply_move_hook( param, operand, target.qualname )
		for param, _expr in keyword:
			self._apply_move_hook( param, kwargs[param.stem], target.qualname )

		return self._finish_generic_call( node, target, type_params, bindings, receiver, args, kwargs, expected_type, want_result )
		# else: this parameter position doesn't mention any of type_params
		# (a concrete parameter, or a nested type whose base doesn't even
		# match the argument's) - nothing to infer here. Not an error by
		# itself: a genuine argument-type mismatch isn't checked anywhere
		# yet (no general type-checking pass exists), same as every other
		# call site in this file today

	def _finish_generic_call( self, node: ast.Call, target: Function, type_params: list[TypeVar], bindings: dict[int,Type], receiver: ir.Operand|None, args: list[ir.Operand], kwargs: dict[str,ir.Operand], expected_type: Type|None, want_result: bool ) -> ir.Operand|None:
		# shared tail of _lower_inferred_generic_call (extracted verbatim,
		# unchanged) and _lower_overload_generic_call below - once `bindings`
		# holds every type param inferable from the ARGUMENTS alone (however
		# they were obtained: interleaved lower+unify for a bare generic-
		# function call, or unified against already-lowered operands for a
		# resolved Overload-group generic candidate), monomorphizing and
		# emitting the call is identical either way.
		missing = [ tv for tv in target.type_params or [] if id( tv ) not in bindings ]
		if missing:
			# return-only inference: a still-unbound type param that never
			# appears in any PARAMETER type (so ordinary argument unification,
			# above, could never have bound it no matter what was passed) but
			# DOES appear in the function's own return type is inferable by
			# actually lowering the body, once every other (argument-bound)
			# type param is known - see _infer_return_only_type_params/
			# _infer_return_only_type_params_inline. A param appearing in
			# NEITHER any parameter nor the return type is a degenerate,
			# vacuous case - left to fail below like anything else genuinely
			# missing, not silently accepted.
			referenced = self.lowering._param_referenced_type_params_for( target )
			return_only_missing = [
				tv for tv in missing
				if id( tv ) not in referenced and self.lowering._type_mentions_param( target.return_type, tv )
			]
			genuinely_missing = [ tv for tv in missing if tv not in return_only_missing ]
			if genuinely_missing:
				# a call can't be "partially" rescued by the return-only path
				# while some other param remains genuinely stuck from the
				# arguments alone - lists every still-unbound name (not just
				# the genuinely-missing ones), same as before this branch
				# existed at all, since an explicit spelling would need to
				# supply ALL of them anyway (no partial explicit subscript
				# exists - see PLAN_RETURN_INFERENCE's own "Deferred")
				self.lowering.discovery.fail(
					f'{target.qualname}[...]: cannot infer type parameter(s) {", ".join(tv.stem for tv in missing)} from these arguments - '
					f'call it explicitly as {target.qualname}[...](...) instead: {ast.unparse(node)}',
					node,
				)
			if target.is_inline:
				return self._infer_return_only_type_params_inline(
					node, target, type_params, bindings, return_only_missing, receiver, args, kwargs, expected_type, want_result,
				)
			monomorphized = self._infer_return_only_type_params( node, target, type_params, bindings, return_only_missing )
			inferred_args = [ bindings[id(tv)] for tv in target.type_params or [] ]
			spec = self.lowering.discovery._get_or_create_specialization( target, inferred_args )
			return self._emit_generic_call( node, spec, monomorphized, receiver, args, kwargs, expected_type, want_result, already_compiled = True )
		inferred_args = [ bindings[id(tv)] for tv in target.type_params or [] ]
		spec = self.lowering.discovery._get_or_create_specialization( target, inferred_args )
		monomorphized = self.lowering._monomorphized_function( spec )
		if monomorphized.is_inline:
			return self._lower_inline_call( node, monomorphized, receiver, args, kwargs, expected_type, want_result )
		return self._emit_generic_call( node, spec, monomorphized, receiver, args, kwargs, expected_type, want_result )

	def _lower_overload_generic_call( self, node: ast.Call, target: Function, receiver: ir.Operand|None, args: list[ir.Operand], kwargs: dict[str,ir.Operand], expected_type: Type|None, want_result: bool ) -> ir.Operand|None:
		# an Overload group's own overload_resolution.resolve_call picked a
		# GENERIC candidate (target.type_params truthy - e.g. a `[T](x: T)`
		# alternative sharing a name with one or more concrete overloads) as
		# either the sole unconditional match or the trailing default of a
		# runtime ConditionalDispatch. Unlike _lower_inferred_generic_call,
		# args/kwargs are ALREADY lowered operands here (resolve_call needed
		# their real types before it could even pick this candidate - see the
		# Overload branch's own _lower_overload_arg call, above) - so there's
		# no interleaved lower+unify to do, just unification directly against
		# each already-known operand's own .type, then the identical
		# monomorphize-and-emit tail _lower_inferred_generic_call itself
		# funnels into via _finish_generic_call.
		assert target.resolve is None, f'internal compiler error - {target=} was not fully resolved by overload_resolution.resolve_call'
		type_params = target.type_params or []
		bindings: dict[int,Type] = {}
		for param, operand in zip( target.parameters or [], args ):
			self.lowering._unify_type_param( type_params, param.type, operand.type, bindings, node, target.qualname )
		for param in target.parameters or []:
			if param.stem in kwargs:
				self.lowering._unify_type_param( type_params, param.type, kwargs[param.stem].type, bindings, node, target.qualname )
		return self._finish_generic_call( node, target, type_params, bindings, receiver, args, kwargs, expected_type, want_result )

	# defensive cap on how many distinct per-tag monomorphizations
	# _expand_dispatch_target will synthesize for ONE generic branch -
	# mirrors overload_resolution.py's own _MAX_TRACKED_STATES spirit (a
	# named, trivially-adjustable constant, not a hard architectural limit).
	# Every real lib/ overload group is 1-2 params/2-4 leaves - nowhere near
	# this before it'd be a genuine sign of a mis-scoped overload group
	# rather than a legitimate need for more combos
	_MAX_DISPATCH_COMBOS = 64

	def _dispatch_remaining_leaves(
		self, node: ast.Call, target: Function, param: Parameter,
		known: dict[int,Type], args: list[ir.Operand], kwargs: dict[str,ir.Operand], claimed: dict[int,list[Type]],
	) -> list[Type]:
		# every leaf `param` could still be, once THIS branch's own runtime
		# tag check(s) (if any) have already excluded every other candidate -
		# see _expand_dispatch_target's own comment. `known` (keyed by
		# id(param)) covers whatever this branch's own condition (or an
		# earlier _expand_dispatch_target combo) already pinned down
		# explicitly - returned as the sole element, no further narrowing
		# needed. Every OTHER parameter is narrowed from the real call-site
		# operand's own (possibly still union) type, minus whatever leaves
		# `claimed` (built from every OTHER branch's own conditions - only
		# ever non-empty for the trailing default) already accounts for
		# elsewhere. Exactly one leaf here means this parameter is statically
		# resolvable (_monomorphize_dispatch_target); more than one is what
		# _expand_dispatch_target splits into distinct per-leaf branches.
		if id( param ) in known:
			return [ known[ id( param ) ] ]
		operand = self.lowering._dispatch_operand_for_param( node, target, param, args, kwargs )
		already = claimed.get( id( operand ), [] )
		return [
			leaf for leaf in operand.type.leaves()
			if not any( self.lowering._type_resolver._same_type( leaf, c ) for c in already )
		]

	def _monomorphize_dispatch_target(
		self, node: ast.Call, target: Function,
		known: dict[int,Type], args: list[ir.Operand], kwargs: dict[str,ir.Operand], claimed: dict[int,list[Type]],
	) -> Function:
		# monomorphizes a GENERIC branch/default of a runtime-dispatched
		# Overload call in place, so the rest of _lower_conditional_dispatch
		# (which schedules every branch's target as an ordinary, concrete
		# compile unit) never sees a bare TypeVar parameter. Unlike
		# _lower_overload_generic_call (which unifies T against the FULL,
		# possibly-union call-site operand type, for the "sole unconditional
		# match" case), a ConditionalDispatch branch/default only ever runs
		# once every OTHER branch's own runtime tag check has excluded its own
		# leaf(s) - _dispatch_remaining_leaves resolves the real, narrower
		# type(s) reaching THIS target at each parameter. Caller contract:
		# every parameter must already resolve to EXACTLY one leaf here (see
		# _expand_dispatch_target, the only real caller) - asserted, not
		# re-validated, since by the time this is called any genuine
		# ambiguity has already been split into a separate combo.
		type_params = target.type_params or []
		bindings: dict[int,Type] = {}
		for param in target.parameters or []:
			remaining = self._dispatch_remaining_leaves( node, target, param, known, args, kwargs, claimed )
			assert len( remaining ) == 1, (
				f'internal compiler error - {target.qualname} parameter {param.stem!r} not resolved to exactly '
				f'one leaf before monomorphizing ({len(remaining)} remaining) - _expand_dispatch_target caller contract violated'
			)
			self.lowering._unify_type_param( type_params, param.type, remaining[0], bindings, node, target.qualname )
		missing = [ tv for tv in type_params if id( tv ) not in bindings ]
		if missing:
			# every real lib/ generic overload binds every type param
			# directly off a parameter (see this module's own note on
			# real-world overload group shapes) - return-only inference
			# (_infer_return_only_type_params) is a materially bigger
			# feature to wire through a runtime-dispatched branch (it needs
			# to actually lower the body to infer the return type) and isn't
			# attempted here; fails clearly rather than silently
			self.lowering.discovery.fail(
				f'a generic overload of {target.qualname} cannot be one branch of a runtime-dispatched call '
				f'(type parameter(s) {", ".join(tv.stem for tv in missing)} aren\'t bound by any parameter - '
				f'return-only inference isn\'t supported here yet): {ast.unparse(node)}',
				node,
			)
		inferred_args = [ bindings[id(tv)] for tv in type_params ]
		spec = self.lowering.discovery._get_or_create_specialization( target, inferred_args )
		return self.lowering._monomorphized_function( spec )

	def _remap_conditions(
		self, original_params: list[Parameter], new_params: list[Parameter], conditions: list[tuple[Parameter,Type]],
	) -> list[tuple[Parameter,Type]]:
		# a monomorphized Function has its own, distinct Parameter objects
		# (same count/order as the generic original - substitution never
		# reorders or drops parameters) - a runtime condition built against
		# the ORIGINAL generic function's own Parameter identity (from
		# resolve_call, or from an earlier _expand_dispatch_target combo)
		# has to be re-pointed at the monomorphized function's corresponding
		# one before _lower_dispatch_tests/_dispatch_operand_for_param can
		# find it there (identity lookup, not structural equality - see
		# their own docstrings)
		return [
			( new_params[ next( i for i, op in enumerate( original_params ) if op is p ) ], leaf_type )
			for p, leaf_type in conditions
		]

	def _expand_dispatch_target(
		self, node: ast.Call, target: Function,
		known: dict[int,Type], args: list[ir.Operand], kwargs: dict[str,ir.Operand], claimed: dict[int,list[Type]],
	) -> list[tuple[list[tuple[Parameter,Type]],Function]]:
		'''
		Resolves a GENERIC branch/default of a runtime-dispatched Overload
		call into one or more concrete (extra_conditions, monomorphized
		Function) pairs. Most of the time this is exactly one entry with no
		extra conditions - the EASY case (_monomorphize_dispatch_target's own
		docstring): every parameter's real leaf is already pinned to exactly
		one value here, either explicitly (`known`, from this branch's own
		runtime condition) or by elimination (`claimed`, only ever populated
		for the trailing default).

		When one or more parameters can still legitimately be more than one
		leaf here - the HARD case this dispatch machinery used to reject
		outright - this ONE branch is split into one synthetic entry PER
		combination of those parameters' remaining leaves, each monomorphized
		with its own distinct T binding and given its own extra runtime
		condition(s) pinning exactly that combination. This turns "this one
		generic branch might need any of N different C functions at runtime,
		selected by a tag no single Call target can express" into N ordinary,
		individually-concrete branches - exactly the shape
		_lower_conditional_dispatch already knows how to schedule, just more
		of them. The caller (the Overload branch of _lower_call) is
		responsible for splicing these into the overall branches/default
		list and picking exactly one overall entry to remain the trailing,
		unconditioned default.
		'''
		params = target.parameters or []
		remaining_by_id = {
			id( p ): self._dispatch_remaining_leaves( node, target, p, known, args, kwargs, claimed )
			for p in params
		}
		ambiguous = [ p for p in params if len( remaining_by_id[ id( p ) ] ) != 1 ]
		if not ambiguous:
			return [ ( [], self._monomorphize_dispatch_target( node, target, known, args, kwargs, claimed )) ]
		combo_count = math.prod( len( remaining_by_id[ id( p ) ] ) for p in ambiguous )
		if combo_count > self._MAX_DISPATCH_COMBOS:
			self.lowering.discovery.fail(
				f'a generic overload of {target.qualname} used as a runtime-dispatched branch would need '
				f'{combo_count} separate per-type monomorphizations here (more than {self._MAX_DISPATCH_COMBOS}) - '
				f'narrow the overloaded parameter types: {ast.unparse(node)}',
				node,
			)
		results: list[tuple[list[tuple[Parameter,Type]],Function]] = []
		for combo in itertools.product( *( remaining_by_id[ id( p ) ] for p in ambiguous )):
			combo_known = dict( known )
			combo_known.update({ id( p ): leaf for p, leaf in zip( ambiguous, combo ) })
			monomorphized = self._monomorphize_dispatch_target( node, target, combo_known, args, kwargs, claimed )
			new_params = monomorphized.parameters or []
			extra_conditions = self._remap_conditions( params, new_params, list( zip( ambiguous, combo )))
			results.append(( extra_conditions, monomorphized ))
		return results

	def _infer_return_only_type_params( self, node: ast.Call, target: Function, type_params: list[TypeVar], bindings: dict[int,Type], return_only: list[TypeVar] ) -> Function:
		# PLAN_RETURN_INFERENCE.md - non-@inline variant: a bare generic
		# call left one or more type params unbound after ordinary
		# argument unification, each appearing ONLY in target's own return
		# type (never in any parameter) - determined by actually, eagerly,
		# synchronously compiling the function's body (with every OTHER,
		# argument-bound type param already substituted) and reading the
		# real return type back off it, generalizing _expr_Lambda's own
		# eager-lowering trick (PLAN_LAMBDA.md) from an unbound Callable's
		# own return type to a named generic function's own return type.
		# Mutates `bindings` in place to add the newly-discovered args, and
		# returns the (now fully concrete) monomorphized Function - already
		# a real, compiled unit (appended to compiler.functions by
		# _compile_now), never rebuilt - for the caller to finish emitting
		# a Call against.
		pending_args = [ bindings.get( id( tv ), tv ) for tv in type_params ]
		pending_spec = self.lowering.discovery._get_or_create_specialization( target, pending_args )
		if pending_spec.monomorphized is not None:
			# another call site with the SAME known args already discovered
			# the return-only bindings and compiled this - no new compile,
			# just recover the bindings THIS call site's own `bindings`
			# dict still needs filled in (the caller reads it right after
			# this returns to build its own Specialization args)
			self.lowering._unify_type_param( type_params, target.return_type, pending_spec.monomorphized.return_type, bindings, node, target.qualname )
			return pending_spec.monomorphized

		if id( target ) in self.lowering._eager_return_inference_stack:
			self.lowering.discovery.fail(
				f'{target.qualname}[...]: cannot infer return-only type parameter(s) {", ".join(tv.stem for tv in return_only)} - '
				f'the body (directly, or through another generic function) recursively calls itself before its own return type is '
				f'known - call it explicitly as {target.qualname}[...](...) instead: {ast.unparse(node)}',
				node,
			)
		if not self.lowering.discovery._is_eager_return_inferable_body( target.node.body ):
			self.lowering.discovery.fail(
				f'{target.qualname}[...]: cannot infer return-only type parameter(s) {", ".join(tv.stem for tv in return_only)} - '
				f'its body must have exactly one reachable `return <expr>` for this to work - call it explicitly as '
				f'{target.qualname}[...](...) instead: {ast.unparse(node)}',
				node,
			)

		self.lowering._eager_return_inference_stack.append( id( target ))
		try:
			# qualname deliberately the PENDING spec's own (not target's bare
			# abstract name) - already unambiguous per known-args binding,
			# and gives compile errors reached WHILE eagerly lowering this
			# body a sensible name too, not the shared, unspecialized one
			provisional = self.lowering._monomorphizer._build_monomorphized_function( target, type_params, pending_args, pending_spec.qualname )
			# forces _stmt_Return's own _lower_expr(node.value, self.
			# _current_fn.return_type) to lower the return expression with
			# NO hint, taking its own natural type - _expr_Lambda's
			# identical return_type_provisional convention, reused verbatim
			provisional.return_type = None
			lowered = self.lowering._compile_now( provisional ) # == Compiler._lower - this IS the final compiled unit, never rebuilt
			# NOT a bare next(...) (unlike _expr_Lambda's own identical-
			# looking lookup, PLAN_LAMBDA.md line ~3555 - a real, pre-
			# existing gap there too, not fixed here, out of scope for this
			# plan): the ONE statement in `provisional`'s own body can
			# itself fail to lower (e.g. the reentrancy guard above,
			# triggered one level deeper by a recursive call inside the
			# body) - FunctionLowering.run()'s own per-statement recovery
			# (try/except CompileError: continue) SWALLOWS that failure
			# silently rather than propagating it here, leaving `lowered.
			# instructions` with no ir.Return at all. A bare next(...) would
			# crash with an unhandled StopIteration instead of a clean
			# compile error - confirmed by a real repro (direct/mutual
			# recursion through this same path). The real error is already
			# recorded in discovery.errors by whatever failed deeper in the
			# body (the reentrancy guard's own fail(), most commonly) -
			# this fail() call is a second, redundant-but-harmless report,
			# same accepted pattern resolve_function_body's own docstring
			# already documents ("gets reported again... so nothing is
			# silently swallowed")
			return_instr = next( ( instr for instr in lowered.instructions if isinstance( instr, ir.Return ) ), None )
			if return_instr is None:
				self.lowering.discovery.fail(
					f'{target.qualname}[...]: cannot infer return-only type parameter(s) {", ".join(tv.stem for tv in return_only)} - '
					f'the body failed to compile (see earlier error): {ast.unparse(node)}',
					node,
				)
			actual_return_type = (
				return_instr.value.type if return_instr.value is not None
				else self.lowering.discovery.get_none_type()
			)
			self.lowering._unify_type_param( type_params, target.return_type, actual_return_type, bindings, node, target.qualname )
		finally:
			self.lowering._eager_return_inference_stack.pop()

		still_missing = [ tv.stem for tv in return_only if id( tv ) not in bindings ]
		if still_missing:
			self.lowering.discovery.fail(
				f'{target.qualname}[...]: cannot infer type parameter(s) {", ".join(still_missing)} - the function\'s own declared '
				f'return type does not structurally match what its body actually returns: {ast.unparse(node)}',
				node,
			)
		full_args = [ bindings[id(tv)] for tv in type_params ]
		real_spec = self.lowering.discovery._get_or_create_specialization( target, full_args )
		# patch the SAME object in place afterward, not rebuilt - mirrors
		# _expr_Lambda's own "reused afterward" convention exactly
		provisional.return_type = self.lowering._substitute_type_params( target.return_type, type_params, full_args )
		provisional.qualname = real_spec.qualname
		real_spec.monomorphized = provisional # the real key, for a FUTURE fully-concrete lookup (e.g. an explicit foo[Concrete,Other](...) call elsewhere)
		pending_spec.monomorphized = provisional # the pending key, for a FUTURE bare call with the same already-known args
		return provisional

	def _infer_return_only_type_params_inline( self, node: ast.Call, target: Function, type_params: list[TypeVar], bindings: dict[int,Type], return_only: list[TypeVar], receiver: ir.Operand|None, args: list[ir.Operand], kwargs: dict[str,ir.Operand], expected_type: Type|None, want_result: bool ) -> ir.Operand|None:
		# PLAN_RETURN_INFERENCE.md - @inline variant: cheaper than
		# _infer_return_only_type_params above because splicing the body IS
		# the eager compile already - no separate FuncStart/CFG/real
		# compiled unit needed for target itself here (matches PLAN_INLINE.
		# md's own "never a real compiled unit" invariant), just the
		# AST-level generic-call-resolution rewrite (resolve_function_body)
		# a NESTED generic call inside the body would need, same as the
		# ordinary is_inline path already gets via _monomorphized_function.
		# _is_eager_return_inferable_body is NOT re-checked here - @inline's
		# own decorator-time _is_inline_eligible_body (exactly one TOP-LEVEL
		# `return <expr>`) is strictly stronger; every @inline-eligible body
		# already trivially satisfies it.
		pending_args = [ bindings.get( id( tv ), tv ) for tv in type_params ]
		pending_spec = self.lowering.discovery._get_or_create_specialization( target, pending_args )
		if pending_spec.monomorphized is not None:
			# another call site already discovered R for these known args -
			# splice against the cached provisional directly. `bindings`
			# doesn't need filling in here (unlike the non-inline variant) -
			# this call RETURNS the final operand straight to _lower_call's
			# own caller, nothing downstream reads `bindings` again
			return self._lower_inline_call( node, pending_spec.monomorphized, receiver, args, kwargs, expected_type, want_result )

		if id( target ) in self.lowering._eager_return_inference_stack:
			self.lowering.discovery.fail(
				f'@inline {target.qualname}[...]: cannot infer return-only type parameter(s) {", ".join(tv.stem for tv in return_only)} - '
				f'the body (directly, or through another generic function) recursively calls itself before its own return type is '
				f'known - call it explicitly as {target.qualname}[...](...) instead: {ast.unparse(node)}',
				node,
			)
		self.lowering._eager_return_inference_stack.append( id( target ))
		try:
			provisional = self.lowering._monomorphizer._build_monomorphized_function( target, type_params, pending_args, pending_spec.qualname )
			provisional.return_type = None
			# @inline never needs a real compiled unit for its own target -
			# only the same AST-rewrite pass the ordinary is_inline path
			# already runs on a monomorphized copy before splicing it, so a
			# nested generic call inside the body resolves against the
			# concrete, already-substituted T
			self.lowering._type_resolver.resolve_function_body( provisional )
			# forced True regardless of the real want_result - the real
			# operand (and its .type) is needed to discover the return-only
			# bindings even when the CALLER's own want_result is False; the
			# discard-Result check _lower_inline_call would otherwise apply
			# internally is skipped by forcing this (target.return_type is
			# still None here, so it would've been a no-op anyway) and
			# re-applied below instead, once the real return type is known
			result = self._lower_inline_call( node, provisional, receiver, args, kwargs, expected_type, want_result = True )
			actual_return_type = result.type if result is not None else self.lowering.discovery.get_none_type()
			self.lowering._unify_type_param( type_params, target.return_type, actual_return_type, bindings, node, target.qualname )
		finally:
			self.lowering._eager_return_inference_stack.pop()

		still_missing = [ tv.stem for tv in return_only if id( tv ) not in bindings ]
		if still_missing:
			self.lowering.discovery.fail(
				f'{target.qualname}[...]: cannot infer type parameter(s) {", ".join(still_missing)} - the function\'s own declared '
				f'return type does not structurally match what its body actually returns: {ast.unparse(node)}',
				node,
			)
		full_args = [ bindings[id(tv)] for tv in type_params ]
		real_spec = self.lowering.discovery._get_or_create_specialization( target, full_args )
		provisional.return_type = self.lowering._substitute_type_params( target.return_type, type_params, full_args )
		provisional.qualname = real_spec.qualname
		real_spec.monomorphized = provisional # caches the PROVISIONAL BODY for reuse by _lower_inline_call at a future call site - provisional is never independently scheduled/appended to compiler.functions by this variant, matching PLAN_INLINE.md's invariant
		pending_spec.monomorphized = provisional

		if not want_result and cfg.is_result_type( provisional.return_type ):
			# same discard check _lower_inline_call itself applies - done
			# here instead since target.return_type wasn't known yet when
			# _lower_inline_call ran above (want_result was forced True)
			self.lowering.discovery.fail(
				f'{target.qualname}(...) returns a Result that is discarded here - '
				f'assign it to a name and use .is_ok(), .is_err(), .or_return(), .unwrap(msg), or match: {ast.unparse(node)}',
				node,
			)
		return result if want_result else None

	def _emit_generic_call( self, node: ast.Call, spec: Specialization, monomorphized: Function, receiver: ir.Operand|None, args: list[ir.Operand], kwargs: dict[str,ir.Operand], expected_type: Type|None, want_result: bool, *, already_compiled: bool = False ) -> ir.Operand|None:
		# schedules the Specialization itself as the compile unit (see
		# _monomorphized_function/compiler.py's own handling of it), shared
		# tail for both the explicit Name[T](...) and inferred call paths -
		# and, unlike _lower_call's OWN shared tail (which only ever sees a
		# call type_resolver.py's pre-pass could tag with resolved_callee),
		# the ONLY tail a receiver-based generic method call reaches at all
		# (see _lower_class_generic_method_call's own comment on why that
		# one is always left untagged) - so the discard check needs its own
		# copy here too, not just in _lower_call's
		#
		# already_compiled=True (return-only type-parameter inference's own
		# eager-compile path - see _infer_return_only_type_params) means
		# `monomorphized` was already lowered for real, synchronously, via
		# Lowering._compile_now, and is already sitting in compiler.functions
		# - scheduling `spec` again here would re-enqueue it onto the
		# ordinary work queue, and Compiler._lower's Specialization+Function
		# branch has no "already lowered" check of its own: it would
		# unconditionally re-run resolve_function_body (which mutates
		# monomorphized.node.body IN PLACE - a second pass over an already-
		# AST-rewritten body) and lower_function a second time, producing a
		# duplicate LoweredFunction entry for the same qualname (a real
		# duplicate-symbol C compile error), not just wasted work. Every
		# other type/parameter still gets scheduled normally here - those
		# are idempotent registrations of TYPES, unrelated to re-lowering
		# monomorphized's own body
		if not already_compiled:
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

			case 'checked_add' | 'wrapped_add' | 'saturated_add' | 'checked_sub' | 'wrapped_sub' | 'saturated_sub' | \
				'checked_mul' | 'wrapped_mul' | 'saturated_mul' | 'checked_floordiv' | 'wrapped_floordiv' | 'saturated_floordiv' | \
				'checked_mod' | 'wrapped_mod' | 'saturated_mod' | 'checked_truediv' | 'wrapped_truediv' | \
				'checked_shl' | 'wrapped_shl' | 'saturated_shl':
				# every compiler.<mode>_<kind>(a, b) intrinsic shares one
				# lowering - see _lower_compiler_checked_binop and
				# _CHECKED_BINOP_OPCODES/_CHECKED_FLOAT_BINOP_OPCODES for how
				# (kind, mode) picks the actual opcode per operand type
				name = self.lowering._is_compiler_call( node )
				mode, _sep, kind = name.partition( '_' )
				result = self._lower_compiler_checked_binop( node, name, kind, mode, expected_type )
				return result if want_result else None

			case 'cast':
				result = self._lower_compiler_cast( node, expected_type )
				return result if want_result else None

			case 'checked_convert':
				result = self._lower_compiler_checked_convert( node, expected_type )
				return result if want_result else None

			case 'bitand' | 'bitor' | 'bitxor' | 'rshift':
				result = self._lower_compiler_bitwise( node, self.lowering._is_compiler_call( node ), expected_type )
				return result if want_result else None

			case 'cmp_eq' | 'cmp_ne' | 'cmp_lt' | 'cmp_le' | 'cmp_gt' | 'cmp_ge':
				result = self._lower_compiler_cmp( node, self.lowering._is_compiler_call( node ), expected_type )
				return result if want_result else None

			case 'checked_ptr_add' | 'wrapped_ptr_add' | 'saturated_ptr_add' | \
				'checked_ptr_sub' | 'wrapped_ptr_sub' | 'saturated_ptr_sub':
				name = self.lowering._is_compiler_call( node )
				mode, _sep, rest = name.partition( '_' )
				kind = 'add' if rest == 'ptr_add' else 'sub'
				result = self._lower_compiler_ptr_binop( node, name, kind, mode, expected_type )
				return result if want_result else None

			case 'ptr_sub_dist':
				result = self._lower_compiler_ptr_diff( node, expected_type )
				return result if want_result else None

			case '__raw_alloc__':
				result = self._lower_compiler_raw_alloc( node, expected_type )
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

			case 'c_field':
				result = self._lower_compiler_c_field( node, expected_type )
				return result if want_result else None

			case 'c_field_set':
				self._lower_compiler_c_field_set( node, expected_type )
				return None

			case 'c_field_addr':
				result = self._lower_compiler_c_field_addr( node, expected_type )
				return result if want_result else None

			case 'fetch_unicode_table':
				result = self.lowering._lower_compiler_fetch_unicode_table( node )
				return result if want_result else None

			case 'fetch_windows_zones_table':
				result = self.lowering._lower_compiler_fetch_windows_zones_table( node )
				return result if want_result else None

			case 'format_f64':
				result = self._lower_compiler_format_f64( node, expected_type )
				return result if want_result else None

			case 'is_nan':
				result = self._lower_compiler_is_nan_or_inf( node, expected_type, 'is_nan' )
				return result if want_result else None

			case 'is_inf':
				result = self._lower_compiler_is_nan_or_inf( node, expected_type, 'is_inf' )
				return result if want_result else None

			case 'parse_f64':
				result = self._lower_compiler_parse_f64( node, expected_type )
				return result if want_result else None

		if isinstance( node.func, ast.Attribute ) and node.func.attr == 'or_return':
			# <result_expr>.or_return() - recognized by AST shape alone,
			# BEFORE _resolve_callee/_attr_lookup_callable ever look for a
			# real declared 'or_return' method on the receiver's class -
			# there is none to find (discovery.py's _parse_function now
			# rejects any user-written `def or_return(...)` outright, on
			# ANY class, since one could never actually be called - see its
			# own comment). Without this, a receiver whose class has no
			# such method (the ordinary, correct case - nobody is expected
			# to write one) failed to resolve at all ("'or_return' is not
			# callable on ..."), confirmed by a real repro: this bug
			# predates and is unrelated to that new rejection, which just
			# makes the fix here airtight instead of merely "the common
			# case works". _lower_or_return itself already validates the
			# receiver is actually Result[_,_]-shaped (and that no
			# arguments were given) - a non-Result receiver correctly still
			# fails there, with the same message as before.
			receiver = self._lower_expr( node.func.value, None )
			return self._lower_or_return( node, receiver, want_result )

		# each recognizer returns None (not an error) when this call doesn't
		# match its own construction-sugar shape at all, falling through to
		# the next; a real error inside a matched shape (e.g. a malformed
		# __allocate__ call) still raises/records normally
		construction_recognizers = (
			self._try_lower_generator_allocate_call,
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

		if receiver is not None and isinstance( target, Function ) and target.is_move:
			# @move on a method means calling it consumes/invalidates self -
			# cfg.py's own move() (already the exact mechanism _apply_move_hook
			# uses for move[T] PARAMETER arguments) needs to run here too, for
			# the RECEIVER: nothing else ever transitions the CALLER's own
			# ownership-tracking state for a receiver on an @move call -
			# confirmed via a real double-free (bytearray.release(), called
			# through str.from_cstr's own move[bytearray] parameter: release()
			# only invalidates ITS OWN self.__data sentinel, guarding against
			# a double-free of the byte buffer, but does nothing about the
			# CALLER's own binding, which still got an ordinary Decref at
			# scope exit on top of that - two teardown paths for one struct).
			# Also correctly rejects calling an @move method through a merely
			# BORROWED receiver (move()'s own OWNED/COPY precondition), which
			# was never checked before either.
			for instr in self._cfg.move( receiver, target_qualname = target.qualname, param_stem = 'self' ):
				self._emit( instr )

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

		# NOTE: no receiver move for unwrap/unwrap_or here (this used to move a
		# Temp receiver - xs.__getitem__(0).unwrap(msg) - so its RC payload
		# wasn't decref'd twice, back when unwrap/unwrap_or returned
		# self.data.v_Ok as a BORROW, AND back when a temp Result's own payload
		# happened to never actually get tracked/decref'd at all - see cfg.py's
		# rc_leaves()/_refcount_instructions (the resolve_type wiring above,
		# in this file's own CFGState construction) for that separate,
		# previously-missing half. Both are fixed now: unwrap/unwrap_or COPY
		# the payload into a local before returning it (lib/builtins/
		# __init__.py), which increfs it, so the returned value is a
		# genuinely owned reference, and the
		# receiver temp's ordinary end-of-expression decref is exactly right -
		# moving it here would suppress that decref and leak the payload. The
		# same incref is also what fixes the persistent-Variable-receiver
		# double-free (g0 = xs.__getitem__(0); g0.unwrap(msg)) that the old
		# Temp-only move never covered - so both receiver shapes now balance.

		if isinstance( target, _ReceiverDispatch ):
			return self._lower_union_receiver_call( node, target, receiver, expected_type, want_result )

		# `.or_return()` no longer reaches here at all - it's recognized and
		# fully handled at the top of this method, before target/receiver
		# were even resolved (see that check's own comment for why)

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

		# set only by the Overload branch below, when a winning stub's own
		# more specific return type applies to THIS call's real arguments
		# (see the Overload branch's own comment on stub_covers_call) - a
		# genuinely narrower Type than target.return_type's own real,
		# emittable return type. Applied AFTER the call is emitted (see the
		# shared tail below, "narrowed_return_type is not None"): the call
		# itself always still targets the ORIGINAL, shared Function (never a
		# replace()'d copy with a mismatched declared return type), and the
		# narrowing is realized by statically extracting the matching leaf
		# out of the call's real, wide-typed result - sound specifically
		# because stub_covers_call already proved this call's own arguments
		# can never actually produce the OTHER (narrowed-away) member at
		# runtime, so no tag check is needed, mirroring
		# _maybe_unwrap_union_arg's identical no-runtime-check extraction
		# for a union-typed CALL ARGUMENT already known to match one leaf
		narrowed_return_type: Type|None = None
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
			if any( fn.is_move for fn in candidates ):
				# the receiver-move-hook (below, gated on isinstance(target,
				# Function)) never fires for an Overload target at all -
				# calling an @move-decorated overload alternative would
				# neither track receiver ownership correctly nor error, so
				# it's rejected outright, matching this same plan's
				# identical policy for a union-typed receiver
				self.lowering.discovery.fail(
					f'calling an @move-decorated overload of {target.qualname} is not supported: {ast.unparse(node)}',
					node,
				)

			def _peel_move( expr: ast.expr ) -> tuple[ast.expr,bool]:
				# move(...) sugar isn't a real name anywhere - _check_move_
				# argument (only reachable once a single concrete Function
				# target is already chosen, never for an Overload group)
				# already recognizes this exact shape; mirrored here so it
				# at least PARSES against an overload group too, before a
				# winning candidate is even known. Which positions/kwargs
				# were wrapped is remembered (moved_pos/moved_kw below) so
				# it can be validated/applied once resolve_call picks a
				# single concrete winner, below.
				if isinstance( expr, ast.Call ) and isinstance( expr.func, ast.Name ) and expr.func.id == 'move':
					if len( expr.args ) != 1 or expr.keywords:
						self.lowering.discovery.fail( f'move(...) takes exactly one argument: {ast.unparse(expr)}', node )
					return expr.args[0], True
				return expr, False

			peeled_args = [ _peel_move( a ) for a in node.args ]
			args = [ self._lower_overload_arg( e, i, None, candidates, node ) for i, ( e, _ ) in enumerate( peeled_args ) ]
			moved_pos = [ was_moved for _, was_moved in peeled_args ]
			if any( kw.arg is None for kw in node.keywords ):
				self.lowering.discovery.fail( f'**kwargs not supported yet: {ast.unparse(node)}', node )
			peeled_kwargs = { kw.arg: _peel_move( kw.value ) for kw in node.keywords }
			kwargs = { name: self._lower_overload_arg( e, None, name, candidates, node ) for name, ( e, _ ) in peeled_kwargs.items() }
			moved_kw = { name: was_moved for name, ( _, was_moved ) in peeled_kwargs.items() }
			arg_types = [ op.type for op in args ]
			kwarg_types = { name: op.type for name, op in kwargs.items() }
			call_slots: list[int|str] = [ *range( len( arg_types )), *kwarg_types.keys() ]
			arg_leaves: dict[int|str,tuple[Type,...]] = {
				**{ i: tuple( t.leaves() ) for i, t in enumerate( arg_types ) },
				**{ name: tuple( t.leaves() ) for name, t in kwarg_types.items() },
			}

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
				# stub's return type while still calling through to the impl.
				# `bound_to` is a static, unconditional relationship (one
				# stub always resolves to the same implementation), so it
				# alone can't tell whether THIS call's own arguments
				# actually matched the stub's narrower signature or fell
				# through to the implementation's own wider one (e.g.
				# unwrap_or()'s zero-argument form only ever matches the
				# plain `default: T|None = None` impl, never the `default:
				# T` stub bound to it) - stub_covers_call re-checks that
				# against this call's real argument types before narrowing
				winning_stub = next( ( s for s in target.stubs if s.bound_to is fn ), None )
				if (
					winning_stub is not None and winning_stub.return_type is not fn.return_type
					and overload_resolution.stub_covers_call(
						winning_stub, call_slots, arg_leaves, self.lowering._type_resolver._same_type,
					)
				):
					# only actually narrow (build a distinct replace()'d copy)
					# when the stub's own return type is genuinely a DIFFERENT
					# object from fn's own - when T is already itself Optional
					# (e.g. Result[i32|None,E].unwrap_or), the stub's `T` and
					# the impl's `T|None` substitute to the exact SAME interned
					# union object (discovery.py's _get_or_create_union
					# memoizes by flattened/deduped leaf set - see its own
					# flattening fix), so there's nothing to narrow. Skipping
					# the copy in that case matters for more than avoiding
					# useless work: the copy returned here is handed straight
					# to self.lowering._ensure_resolved()/schedule() below as
					# if it were its own real, independent compile unit - fully
					# separately LOWERED (type_resolver.py's
					# resolve_function_body + Lowering.lower_function) under
					# the narrowed return_type - but mangle_function_qualname
					# (emitter_c.py) mangles purely off fn.qualname +
					# fn.overload_group, with no notion of "this Function
					# object is a distinct return-type view of another one" -
					# so a call site combining a zero-argument call (schedules
					# the original, wide-return-type fn) with an explicit-
					# argument call to the same group previously always built
					# and scheduled a SEPARATE replace()'d copy, even on the
					# many calls where there was nothing left to actually
					# narrow, producing two independently-lowered Functions
					# sharing one mangled C symbol - a genuine duplicate-
					# definition/argument-type-mismatch at the real C compile
					# stage. Confirmed via a real compile of
					# Result[i32|None,str]'s own .unwrap_or(42): before this
					# fix, a fresh identical-looking copy was always built and
					# independently scheduled regardless, producing exactly
					# that redefinition.
					return replace( fn, return_type = winning_stub.return_type )
				return fn

			try:
				branches, resolved = overload_resolution.resolve_call(
				target.stubs, target.implementations, arg_types, kwarg_types,
				qualname = target.qualname, same_type = self.lowering._type_resolver._same_type,
			)
			except CompileError as e:
				# resolve_call is a pure function of types with no
				# AST/Discovery reference by design - it raises unrecorded,
				# this is where a location actually gets attached and it
				# lands in the collector
				self.lowering.discovery.fail( str( e ), node )
			if branches:
				if any( moved_pos ) or any( moved_kw.values() ):
					# which branch actually runs is a RUNTIME decision
					# (ConditionalDispatch) - move(...)'s ownership transfer
					# needs a single, statically-known target (matching this
					# plan's own policy on @move through a union receiver/
					# overload group elsewhere) - not attempted here
					self.lowering.discovery.fail(
						f'move(...) through a runtime-dispatched overload group is not supported: {ast.unparse(node)}',
						node,
					)
				branches = [ ConditionalDispatch( conditions = b.conditions, function = _resolve_original( b.function )) for b in branches ]
				resolved = _resolve_original( resolved )
				if resolved.type_params or any( b.function.type_params for b in branches ):
					# a GENERIC candidate as one branch (or the trailing
					# default) of a runtime-dispatched overload group: which
					# concrete C symbol to call has to be fixed at compile
					# time (this compiler has no vtable/runtime-polymorphic
					# dispatch concept anywhere) - reachable only when the
					# call's own union-typed argument has leaves routing to
					# more than one overload, at least one of them generic.
					#
					# Whatever leaf(s) of the union still reach one particular
					# generic branch/default get resolved STATICALLY, either
					# explicitly (a non-default branch always has its own
					# runtime condition(s) - see overload_resolution.
					# resolve_call) or by elimination (the trailing default:
					# the call's real leaves at that slot, minus whatever
					# every OTHER branch's own condition already claims
					# there). The common case is exactly ONE leaf - T is
					# knowable at compile time same as any other generic
					# call, and this branch is monomorphized in place before
					# ever reaching _lower_conditional_dispatch (which
					# unconditionally schedules every branch's target as an
					# ordinary, already-concrete compile unit).
					#
					# When 2+ leaves can still legitimately reach ONE generic
					# branch (e.g. a 3+-member union where only one member
					# has a concrete overload - every OTHER member falls
					# through to the SAME generic default, each needing its
					# own distinct monomorphization) - _expand_dispatch_target
					# splits that ONE branch into one new, individually-
					# concrete branch PER leaf, each with its own extra
					# runtime condition pinning exactly that leaf. Exactly
					# one overall entry (see `default_fn` below) stays the
					# trailing, unconditioned default - by construction, once
					# every OTHER entry's own condition has been tested and
					# excluded, only that one's own territory can remain, so
					# it never needs a check of its own either way.
					# every leaf already spoken for by a CONCRETE candidate,
					# keyed by the real call-site operand it came from - not
					# just whichever branches happen to carry an explicit
					# runtime condition: a concrete candidate that ends up as
					# the trailing default has its own condition computed
					# then discarded by resolve_call (see its own "own
					# leaves" comment - a default never needs one), so
					# reading conditions alone under-counts. A concrete
					# function's own declared parameter type unambiguously
					# IS the one leaf it handles, condition or not - reading
					# .parameters directly instead is both simpler and
					# correct for every concrete candidate, branch or default.
					claimed: dict[int,list[Type]] = {}
					for b in ( *branches, ConditionalDispatch( conditions = [], function = resolved )):
						if b.function.type_params:
							continue
						for p in b.function.parameters or []:
							if p.type is None:
								continue
							operand = self.lowering._dispatch_operand_for_param( node, b.function, p, args, kwargs )
							claimed.setdefault( id( operand ), [] ).append( p.type )
					new_branches: list[ConditionalDispatch] = []
					for b in branches:
						if not b.function.type_params:
							new_branches.append( b )
							continue
						original_params = b.function.parameters or []
						known = { id( p ): t for p, t in b.conditions }
						for extra_conditions, monomorphized in self._expand_dispatch_target( node, b.function, known, args, kwargs, claimed ):
							new_params = monomorphized.parameters or []
							remapped = self._remap_conditions( original_params, new_params, b.conditions )
							new_branches.append( ConditionalDispatch( conditions = remapped + extra_conditions, function = monomorphized ))
					branches = new_branches
					if resolved.type_params:
						# the LAST expansion becomes the new trailing default
						# (its own extra_conditions are dropped - see the
						# comment above); every OTHER expansion is a genuine
						# new conditioned branch, appended after the ones
						# above (lowest priority, matching resolve_call's own
						# "default is whatever's left once every real branch
						# is excluded" convention)
						*extra, ( _, default_fn ) = self._expand_dispatch_target( node, resolved, {}, args, kwargs, claimed )
						for extra_conditions, monomorphized in extra:
							new_branches.append( ConditionalDispatch( conditions = extra_conditions, function = monomorphized ))
						resolved = default_fn
				return self._lower_conditional_dispatch( node, branches, resolved, receiver, args, kwargs, expected_type, want_result )
			winning_stub = next( ( s for s in target.stubs if s.bound_to is resolved ), None )
			if (
				winning_stub is not None and winning_stub.return_type is not resolved.return_type
				and overload_resolution.stub_covers_call(
					winning_stub, call_slots, arg_leaves, self.lowering._type_resolver._same_type,
				)
			):
				narrowed_return_type = winning_stub.return_type
			target = resolved
			if target.type_params:
				# resolve_call picked a GENERIC candidate (see overload_
				# resolution.py's own wildcard/TypeVar handling) - target is
				# still the abstract, unspecialized Function here, never a
				# real compile unit of its own (only ITS monomorphized
				# Specialization ever is - see _emit_generic_call's own
				# scheduling). Deliberately skips _ensure_resolved(target)
				# below (unlike the concrete case) - resolve_call() already
				# resolved every group member internally, and
				# _ensure_resolved's own unconditional schedule() would
				# register this bare abstract Function as a real compile
				# unit, reaching the emitter with a still-bare TypeVar
				# parameter (confirmed via a real repro: emitter_c.py's
				# c_type() crashes with NotImplementedError on the TypeVar
				# itself). Route through the same inference+monomorphization
				# machinery a bare generic-function call uses instead of
				# falling into the rest of this branch, which assumes a
				# concrete target.parameters (union-coercion, move
				# validation, default-arg filling) and the shared call-
				# emission tail below, neither of which apply to an
				# unspecialized generic target.
				return self._lower_overload_generic_call( node, target, receiver, args, kwargs, expected_type, want_result )
			self.lowering._ensure_resolved( target ) # resolve_call() already resolved every group member internally - this just schedules the chosen one

			# args/kwargs were lowered by _lower_overload_arg BEFORE target was
			# known, each literal deliberately typed as narrowly as possible
			# (see _lower_overload_arg's own comment on why: so resolve_call's
			# own dispatch decision, above, sees the literal's true, single-
			# leaf type rather than a whole union "either member is possible"
			# type). Now that target is fixed, coerce any operand that's still
			# narrower than target's own real declared parameter type UP into
			# it - mirrors the identical coercion an ORDINARY, non-overloaded
			# call gets for free by lowering its arguments directly against
			# the (already statically known) target's parameter types; an
			# Overload target never goes through that path (see _lower_call_
			# args, only used by the plain-target `else` branch below), so it
			# needs this equivalent applied explicitly, once, here
			for i, param in enumerate( target.parameters or [] ):
				if i >= len( args ):
					break
				if args[i].type is not param.type and isinstance( param.type, TaggedUnion ):
					args[i] = self._coerce_into_union( args[i], param.type, node )
			for param in target.parameters or []:
				if param.stem in kwargs and kwargs[param.stem].type is not param.type and isinstance( param.type, TaggedUnion ):
					kwargs[param.stem] = self._coerce_into_union( kwargs[param.stem], param.type, node )

			# now that a single concrete winner is known, validate move(...)
			# usage against ITS OWN parameters (mirroring _check_move_
			# argument's identical checks) and actually transition
			# ownership (mirroring _apply_move_hook) - both were previously
			# unreachable for an Overload target, see this plan's own item 3
			for i, param in enumerate( target.parameters or [] ):
				if i >= len( args ):
					break
				was_moved = moved_pos[i] if i < len( moved_pos ) else False
				if param.is_move and not was_moved:
					self.lowering.discovery.fail(
						f"{target.qualname}: parameter {param.stem!r} is move[{param.type.qualname}] - "
						f"call site must pass move(...): {ast.unparse(node)}",
						node,
					)
				elif was_moved and not param.is_move:
					self.lowering.discovery.fail(
						f"{target.qualname}: parameter {param.stem!r} is not move[T] - "
						f"call site must not wrap it in move(...): {ast.unparse(node)}",
						node,
					)
				if param.is_move:
					self._apply_move_hook( param, args[i], target.qualname )
			for param in target.parameters or []:
				if param.stem not in moved_kw:
					continue
				was_moved = moved_kw[param.stem]
				if param.is_move and not was_moved:
					self.lowering.discovery.fail(
						f"{target.qualname}: parameter {param.stem!r} is move[{param.type.qualname}] - "
						f"call site must pass move(...): {ast.unparse(node)}",
						node,
					)
				elif was_moved and not param.is_move:
					self.lowering.discovery.fail(
						f"{target.qualname}: parameter {param.stem!r} is not move[T] - "
						f"call site must not wrap it in move(...): {ast.unparse(node)}",
						node,
					)
				if param.is_move:
					self._apply_move_hook( param, kwargs[param.stem], target.qualname )

			# fill in default values for any of target's OWN parameters the
			# call site didn't supply - mirrors _lower_call_args's identical
			# tail for the plain (non-Overload) path just below, which this
			# branch never goes through (an Overload target builds args/
			# kwargs itself, above, straight from node.args/node.keywords,
			# with no equivalent step). Without this, a zero-argument
			# unwrap_or() call (its own `default: T|None = None` impl
			# parameter never supplied) reached real emission with no
			# 'default' entry in instr.kwargs at all, crashing emitter_c.py's
			# _emit_call_args with a bare KeyError
			given = { p.stem for i, p in enumerate( target.parameters or [] ) if i < len( args ) }
			given.update( kwargs.keys() )
			for param in target.parameters or []:
				if param.stem not in given and param.default is not None:
					# see _lower_call_args's identical call for why this is
					# needed - a construction call embedded in this default
					# otherwise never gets its __init__ eagerly pre-resolved
					self.lowering._type_resolver.resolve_parameter_default( target, param )
					default_operand = self._lower_expr( param.default, param.type )
					kwargs[param.stem] = default_operand
		else:
			self.lowering._resolve_call_target( target )
			# a Scalar-registered method's receiver isn't threaded through
			# call.args at all - _match_call_args's own
			# receiver_fills_first_param excludes target's first positional
			# parameter from call-site matching accordingly (see its own
			# comment); the receiver is spliced back in as an ordinary
			# leading positional argument just below.
			receiver_fills_first_param = receiver is not None and isinstance( target, Function ) and target.cls is None
			args, kwargs = self._lower_call_args( target, node, receiver_fills_first_param = receiver_fills_first_param )

		if receiver is not None and isinstance( target, Function ) and target.cls is None:
			# ir.Call's own receiver field is for real bound-method calls
			# only (an RCClass/CStruct method with "self" already excluded
			# from .parameters) - a Scalar-attached free function (see the
			# comment above) takes the receiver as an ordinary LEADING
			# positional argument instead, confirmed by a real KeyError
			# crash on ordinary `f.__str__()` call syntax before this fix.
			# _lower_method_call above (used by f-string dunder-dispatch/
			# format-spec call sites) already carries this exact fix for
			# its own narrower set of callers; this is the same fix for the
			# general call-lowering path every other Scalar-attached-method
			# call site (including ordinary user-written `receiver.method()`
			# syntax) actually goes through.
			args = [ receiver ] + args
			receiver = None

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
			if narrowed_return_type is not None:
				# the raw call always targets `target`'s own REAL, wide return
				# type (see narrowed_return_type's own comment, above where
				# it's declared) - sizing dest directly to expected_type/
				# narrowed_return_type here, the way the ordinary branches
				# below do, would declare dest with a C type the actual
				# callee never produces (its real, emitted C prototype still
				# returns the whole union struct). The narrowing is instead
				# realized AFTER the call: extract the matching leaf out of
				# the wide result via the SAME no-runtime-check extraction
				# _maybe_unwrap_union_arg already uses for a union-typed call
				# ARGUMENT known to match one leaf - sound here for the
				# identical reason (stub_covers_call already proved this
				# call's arguments can never produce the other member)
				dest = self._new_temp( target_return_type )
				self._emit( ir.Call( dest = dest, target = target, receiver = receiver, args = args, kwargs = kwargs ))
				return self._maybe_unwrap_union_arg( dest, narrowed_return_type )
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
				# a bare TypeVar expected_type is never a legitimate coercion
				# target here - by this point `target` is always already a
				# concrete, resolved Function (never itself mid-generic-
				# dispatch - see the "already tagged with its resolved,
				# monomorphized callee" comment above), so target_return_type
				# is the call's own real, concrete return type. A stray bare
				# TypeVar hint (e.g. _lower_allocate_fields threading a
				# generic field's own still-abstract declared type down as
				# the expected_type for lowering that field's VALUE
				# expression - _infer_allocate_type_args then needs that
				# value's own REAL type back, not this hint bounced straight
				# through) would otherwise silently override dest with a
				# meaningless, unresolvable type instead of what the callee
				# actually returns.
				#
				# Beyond the TypeVar case, expected_type is only safe to use
				# for dest when it's the SAME type as target_return_type (just
				# possibly a different Specialization/TupleType representation
				# of the identical instantiation - _same_type's own docstring)
				# - using it whenever merely non-None, regardless of whether it
				# actually matches, let a genuinely mismatched declared local
				# type (`x: i32 = a_result_returning_call()`) silently retype
				# dest to i32 while the callee's real C prototype still returns
				# the whole Result struct, producing invalid C AND skipping
				# _coerce_or_check_operand's own mismatch rejection below
				# entirely (operand.type came back already equal to
				# expected_type, so its "still mismatched past this point"
				# check never even saw a mismatch to catch). A genuine
				# mismatch here now falls through with dest correctly typed as
				# target_return_type instead, so the coercion-or-rejection
				# tail gets an honest look at it.
				if isinstance( expected_type, TypeVar ):
					dest = self._new_temp( target_return_type )
				elif expected_type is not None and target_return_type is not None and not self.lowering._type_resolver._same_type( target_return_type, expected_type ):
					dest = self._new_temp( target_return_type )
				else:
					dest = self._new_temp( expected_type or target_return_type )
			self._emit( ir.Call( dest = dest, target = target, receiver = receiver, args = args, kwargs = kwargs ))
			return dest
		else:
			self._emit( ir.Call( dest = None, target = target, receiver = receiver, args = args, kwargs = kwargs ))
			return None

	def _lower_conditional_dispatch( self, node: ast.Call, branches: list[ConditionalDispatch], default: Function, receiver: ir.Operand|None, args: list[ir.Operand], kwargs: dict[str,ir.Operand], expected_type: Type|None, want_result: bool ) -> ir.Operand|None:
		# a union-typed argument's runtime tag decides which overload
		# implementation actually runs (e.g. len(copy_from) where
		# copy_from: bytes|bytearray resolves to two candidates, bytes and
		# bytearray). Reuses the same tag/data/v_<member> machinery match
		# statements use (UnionStorage.get) - branches are tried in
		# priority order, falling through to `default` (no test needed -
		# it's whatever's left once every more specific branch is excluded).
		# `receiver` is the SAME instance operand for every branch (an
		# Overload group is either entirely bound methods sharing one
		# receiver, or entirely receiver-less free functions/statics - never
		# a mix) - already scheduled/move-tracked by _lower_call before ever
		# reaching here, so it's just threaded through unchanged into each
		# branch's own ir.Call, same as `args`/`kwargs` already are.
		self.lowering._ensure_resolved( default )
		for branch in branches:
			self.lowering._ensure_resolved( branch.function )

		dest = self._new_temp( expected_type or default.return_type ) if want_result else None
		end_label = self._new_label( 'dispatch_end' )
		for branch in branches:
			next_label = self._new_label( 'dispatch_next' )
			self._lower_dispatch_tests( node, branch.function, branch.conditions, args, kwargs, next_label )
			self._emit_dispatch_call( branch.function, receiver, args, kwargs, dest, want_result )
			self._emit( ir.Jump( target = end_label ))
			self._emit( ir.Label( name = next_label ))
		self._emit_dispatch_call( default, receiver, args, kwargs, dest, want_result )
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
			# _same_type, not raw `is` - leaf_type (from overload_resolution.
			# py's own call-site-argument-derived condition) and a member's
			# own .type (re-derived here from operand.type via
			# _tagged_union_shape) can be two different Specialization
			# objects for the identical generic instantiation - same
			# duality _check_assignable/_unify_type_param/_coerce_into_union
			# already guard against elsewhere (see PLAN_COMPILER_BUG_SWEEP.md)
			member = next( ( attr for attr in members if self.lowering._type_resolver._same_type( attr.type, leaf_type ) ), None )
			if member is None:
				self.lowering.discovery.fail( f'{target.qualname}: {leaf_type.qualname if leaf_type else "?"} is not a member of {operand.type.qualname}', node )
			tag_attr, _data_attr, _payload_cls, tags = self.lowering._union_storage.get( base )
			tag_dest = self._new_temp( tag_attr.type )
			self._emit( ir.GetAttr( dest = tag_dest, obj = operand, attr = tag_attr.stem ))
			cmp_dest = self._new_temp( bool_cls )
			self._emit( ir.Cmp( dest = cmp_dest, op = ir.CmpOp.EQ, left = tag_dest, right = ir.Const( type = tag_attr.type, value = tags[member.stem] ) ))
			self._emit( ir.JumpIfFalse( cond = cmp_dest, target = next_label ))

	def _emit_dispatch_call( self, target: Function, receiver: ir.Operand|None, args: list[ir.Operand], kwargs: dict[str,ir.Operand], dest: ir.Temp|None, want_result: bool ) -> None:
		params = target.parameters or []
		unwrapped_args = [ self._maybe_unwrap_union_arg( a, p.type ) for a, p in zip( args, params ) ]
		unwrapped_kwargs = {
			name: self._maybe_unwrap_union_arg( value, next( p for p in params if p.stem == name ).type )
			for name, value in kwargs.items()
		}
		self.lowering.schedule( target.return_type )
		for p in params:
			self.lowering.schedule( p.type )
		self._emit( ir.Call( dest = dest if want_result else None, target = target, receiver = receiver, args = unwrapped_args, kwargs = unwrapped_kwargs ))

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
		# _same_type, not raw `is` - same duality as _lower_dispatch_tests'
		# own identical fix just above (target_type and a member's own
		# .type can be two different Specialization objects for the same
		# generic instantiation). Silently returning operand UNCHANGED
		# when no member matches (rather than failing loudly) makes this
		# one worse than _lower_dispatch_tests' own version if it ever
		# misfires - a wrong, still-union-typed argument passed through
		# to a call expecting a concrete leaf, not a compile error
		member = next( ( attr for attr in members if self.lowering._type_resolver._same_type( attr.type, target_type ) ), None )
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
			# per-leaf argument coercion/validation - `args`/`kwargs` above
			# were built ONCE, lowered against `reference`'s own declared
			# parameter types only; a leaf whose own parameter type
			# genuinely differs (Box[i32]|Box[u32]'s own two `set(x: T)`
			# instantiations) needs the SAME coercion-or-rejection chain
			# _lower_expr's own dispatch would already have given it, run
			# again here against THIS leaf's own type - reusing the already-
			# lowered operand (never re-lowering/re-evaluating the original
			# argument expression, which would double its side effects once
			# per leaf; see _coerce_or_check_operand's own docstring)
			leaf_temps_start = len( self._pending_temps )
			leaf_args = []
			for ( ref_param, expr ), operand in zip( positional, args ):
				leaf_param = self._corresponding_leaf_param( reference, fn, ref_param )
				context = f'{fn.qualname}(...): parameter {leaf_param.stem!r}'
				leaf_args.append( self._coerce_or_check_operand( operand, leaf_param.type, expr, context = context ))
			leaf_kwargs = {}
			for ref_param, expr in keyword:
				leaf_param = self._corresponding_leaf_param( reference, fn, ref_param )
				context = f'{fn.qualname}(...): parameter {leaf_param.stem!r}'
				leaf_kwargs[leaf_param.stem] = self._coerce_or_check_operand( kwargs[ref_param.stem], leaf_param.type, expr, context = context )
			self._emit( ir.Call( dest = dest, target = fn, receiver = narrowed, args = leaf_args, kwargs = leaf_kwargs ))
			# a leaf whose own parameter type needs real union-widening
			# coercion (not just a borrowed CastWrap - see _coerce_or_check_
			# operand's own comment) leaves a fresh, independently
			# fresh_temp()-tracked wrapped value in leaf_args/leaf_kwargs,
			# passed to the Call above as an ordinary BORROWED argument (no
			# ownership transfer, same convention every other call site
			# uses) - the caller still owns releasing it. In non-branching
			# code the enclosing statement's own end-of-statement flush does
			# that correctly; here this whole per-leaf block is only ONE
			# branch of a larger dispatch tree (skippable via an earlier
			# leaf's own tag match), so that flush fires unconditionally for
			# EVERY leaf regardless of which one's Call actually ran -
			# reading tag/payload data off an uninitialized C local for
			# whichever leaf never executed. Confirmed via a real repro
			# (union receiver dispatch, one leaf declaring a plain parameter
			# type, the other a wider union needing _coerce_into_union) -
			# same bug class _flush_branch_temps' own docstring documents
			# for _expr_BoolOp/_expr_IfExp, and _emit_eq_dispatch_tree's/
			# _coerce_or_check_operand's own hand-rolled decref+untrack
			# fixes cover elsewhere in this file. dest is excluded (it's
			# this whole call's own merge point, must survive to the next
			# leaf/end_label)
			self._flush_branch_temps( leaf_temps_start, *( [ dest ] if dest is not None else [] ))
			if not is_last:
				self._emit( ir.Jump( target = end_label ))
				self._emit( ir.Label( name = next_label ))
		self._emit( ir.Label( name = end_label ))
		return dest

	def _corresponding_leaf_param( self, reference: Function, fn: Function, ref_param: Parameter ) -> Parameter:
		''' the Parameter in `fn`'s own parameter list at the SAME POSITION
		as `ref_param` in `reference`'s - used by _lower_union_receiver_call
		to find each leaf's own declared type for an argument that was
		matched (once, against `reference` only) by _match_call_args.
		Index-based, not name-based: type_resolver.py's own _resolve_union_
		receiver_members already guarantees every leaf has the SAME
		parameter COUNT as reference, but not (yet - a real, smaller,
		separate gap, not attempted here) the same names/kinds at each
		position, so position is the only correspondence available. '''
		index = next( i for i, p in enumerate( reference.parameters ) if p is ref_param )
		return fn.parameters[index]
