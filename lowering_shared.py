# stdlib imports:
import ast
from dataclasses import dataclass
from typing import Iterable

# local imports:
import arithmetic_mode
import ir
from mpy_types import Function, Type, Variable

''' module-level constants/small dataclasses shared by lowering.py's Lowering/
FunctionLowering and the lowering_*.py mixins FunctionLowering is composed
from - split out to a leaf module (no lowering.py/lowering_*.py imports of
its own) so every one of those files can import from here directly without
risking a circular import back into lowering.py itself. '''

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

# ast.operator -> the IN-PLACE dunder name (__iadd__, ...) _stmt_AugAssign
# tries before falling back to the ordinary out-of-place _BINOP_DUNDER path.
# No reflected counterpart - Python's own data model has no such concept for
# in-place operators either (there's no "b.__riadd__(a)"). Gated (see
# _find_iplace_dunder) on the receiver being an RC-class pointer: only then
# does mutating self in place have any effect the caller can observe (a
# CStruct/Scalar receiver's self is passed BY VALUE), so this table is only
# ever consulted for RC-class receivers.
_IPLACE_BINOP_DUNDER: dict[type,str] = {
	ast.Add: '__iadd__',
	ast.Sub: '__isub__',
	ast.Mult: '__imul__',
	ast.FloorDiv: '__ifloordiv__',
	ast.Mod: '__imod__',
	ast.Div: '__itruediv__',
	ast.BitOr: '__ior__',
	ast.BitAnd: '__iand__',
	ast.BitXor: '__ixor__',
	ast.LShift: '__ilshift__',
	ast.RShift: '__irshift__',
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

@dataclass
class TryHandler:
	''' one `except T:`/`except (A,B) as e:` clause of some enclosing try
	(FunctionLowering._try_stack) - `leaves` are the resolved error-class
	leaves this clause covers (each may itself be one class, or several
	for a tuple-of-classes clause). `bind` is the real, already-registered
	local Variable `as NAME` binds (see _stmt_Try), or None for a bare
	`except T:` - kept None in that case since nothing user-visible should
	depend on it existing. `raise_value_var` is ALWAYS a real registered
	local of the same type `bind` would have (equal to `bind` itself when
	the clause wrote `as NAME`, otherwise a hidden compiler-synthesized
	one) - a bare `raise` inside this handler's own body re-raises
	whatever this holds (see FunctionLowering._active_raise_values), so
	every handler needs one regardless of whether the user named it.
	`matched` is set True the moment ANY leaf of this handler is actually
	selected by a .or_throw()/raise dispatch anywhere inside this
	handler's own try body (see _dispatch_leaves_against_try_stack) - an
	except clause left False once its own try's body is fully lowered is
	unreachable dead code, a compile error (_stmt_Try's own check, after
	the try_stack pop). '''
	leaves: list[Type]
	label: str
	bind: 'Variable|None'
	raise_value_var: 'Variable'
	matched: bool = False

@dataclass
class TryContext:
	''' one entry of FunctionLowering._try_stack - a try statement
	currently being lowered (nested trys push one entry each, innermost
	last). Consulted by .or_throw()/raise (_dispatch_leaves_against_try_stack)
	textually inside its own body, in the SAME function - an uncovered
	leaf walks the WHOLE stack innermost-first, checking every textually
	enclosing try's own handlers before falling back to propagating to
	the caller. `entry_stack_depth` is this try's own entry_snapshot.
	stack_depth (the try body is UNCONFINED - see TryHandler's own
	docstring - so this is the one place that boundary is still recorded):
	a covered `raise`/or_throw() dispatch straight into this try's own
	handler is a goto that never otherwise unwinds anything (see ir.
	ThrowLeaf.epilogue's own docstring) - everything the try BODY pushed
	since this depth is about to become unreachable from the handler and
	must be released right before the jump, same as unwind_to() already
	does for break/continue leaving a loop. '''
	handlers: list[TryHandler]
	end_label: str
	entry_stack_depth: int
