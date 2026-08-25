# stdlib imports:
from dataclasses import dataclass, field
from enum import Enum
from typing import ClassVar, Union

# local imports:
from mpy_types import Type, Variable, Parameter, ClassLike, Function

@dataclass( kw_only = True )
class Value:
	type: Type

@dataclass( kw_only = True )
class Temp( Value ):
	''' compiler-generated register/temporary - stage 2 numbers these (t0, t1, ...) '''
	id: int

@dataclass( kw_only = True )
class Const( Value ):
	value: bool|int|str|bytes|float|None # float: an f32/f64 literal (3.14, 1.5, ...)

@dataclass( kw_only = True )
class FunctionRef( Value ):
	''' a bare reference to a plain, receiver-less function (free function
	or @staticmethod) used as a VALUE - see PLAN_CALLABLE.md. .type is
	always Ptr[CallableType(...)] (never the bare CallableType - a function
	reference is a pointer, same as every other "address of" operation in
	this language). Emitted only by lowering.py's _expr_Name (there's no
	syntax to construct one any other way yet - no lambdas/closures, see
	the plan doc's own "deferred" list). '''
	fn: Function

Operand = Union[ Temp, Const, Variable, FunctionRef ] # Variable already covers params/locals/globals via mpy_types

@dataclass( kw_only = True )
class Instruction:
	def test_repr( self ) -> str:
		cls = type( self )
		raise NotImplementedError( f'{cls.__module__}.{cls.__qualname__}.test_repr()' )

@dataclass( kw_only = True )
class FuncStart( Instruction ):
	name: str
	params: list[Parameter]
	return_type: Type|None
	# @extern('lib','symbol') (see mpy_types.Function) - both None for an
	# ordinary function. A future emitter declares this as a foreign call
	# signature (no body follows - see lowering.py's lower_function) rather
	# than defining it
	extern_lib: str|None = None
	extern_symbol: str|None = None

	def test_repr( self ) -> str:
		extern = f', extern_lib={self.extern_lib!r}, extern_symbol={self.extern_symbol!r}' if self.extern_lib is not None else ''
		return f'FuncStart( name={self.name!r}, params={self.params!r}, return_type={self.return_type!r}{extern} )'

@dataclass( kw_only = True )
class FuncEnd( Instruction ):
	name: str

	def test_repr( self ) -> str:
		return f'FuncEnd( name={self.name!r} )'

@dataclass( kw_only = True )
class DeclareTemp( Instruction ):
	temp: Temp

	def test_repr( self ) -> str:
		return f'DeclareTemp( temp={self.temp!r} )'

@dataclass( kw_only = True )
class DeleteTemp( Instruction ): # temp is no longer valid after this instruction
	temp: Temp

	def test_repr( self ) -> str:
		return f'DeleteTemp( temp={self.temp!r} )'

@dataclass( kw_only = True )
class DeclareLocal( Instruction ):
	''' PLAN_INLINE.md early-return generalization - a bare "TYPE name;"
	declaration (no initializer) for a synthesized, splice-local NAMED
	Variable, emitted flat/unconditionally wherever this instruction itself
	is placed - the DeclareTemp of Variables (which are keyed by .stem, not
	an id, so DeclareTemp itself doesn't fit). Specifically for _splice_
	multi_statement_inline_body's own result_var: relying on ir.Assign's own
	lazy "declare on first use" (Lowering._emit_instruction) risks the
	declaration landing INSIDE one of emitter_c.py's own hand-emitted C `{
	}` blocks (_emit_or_return's Err branch, _emit_or_jump's tag check) if
	an early exit via .or_return()/checked-arithmetic happens to be result_
	var's first write - C then scopes the declaration to just that block,
	making it an undeclared identifier anywhere used afterward (confirmed by
	a real repro, not just reasoning). No such hazard for a type that
	already has a trivial default value (exited_flag: bool, initialized via
	a plain, always-flat ir.Assign(..., Const(False)) instead) - this is
	only needed for result_var, whose type has no generic zero/default
	representation to assign. '''
	variable: Variable

	def test_repr( self ) -> str:
		return f'DeclareLocal( variable={self.variable!r} )'

@dataclass( kw_only = True )
class Assign( Instruction ):
	dest: Variable|Temp
	src: Operand

	def test_repr( self ) -> str:
		return f'Assign( dest={self.dest!r}, src={self.src!r} )'

@dataclass( kw_only = True )
class GetAttr( Instruction ):
	dest: Temp
	obj: Operand
	attr: str

	def test_repr( self ) -> str:
		return f'GetAttr( dest={self.dest!r}, obj={self.obj!r}, attr={self.attr!r} )'

@dataclass( kw_only = True )
class SetAttr( Instruction ):
	obj: Operand
	attr: str
	value: Operand

	def test_repr( self ) -> str:
		return f'SetAttr( obj={self.obj!r}, attr={self.attr!r}, value={self.value!r} )'

@dataclass( kw_only = True )
class GetItem( Instruction ): # a[i]
	dest: Temp
	obj: Operand
	index: Operand

	def test_repr( self ) -> str:
		return f'GetItem( dest={self.dest!r}, obj={self.obj!r}, index={self.index!r} )'

@dataclass( kw_only = True )
class SetItem( Instruction ):
	obj: Operand
	index: Operand
	value: Operand

	def test_repr( self ) -> str:
		return f'SetItem( obj={self.obj!r}, index={self.index!r}, value={self.value!r} )'

@dataclass( kw_only = True )
class GetAttrIndex( Instruction ):
	# f.arr[i] - element-level read of a FixedArrayType field (mpy_types.
	# FixedArrayType, `u8[8]`-style inline C array). obj is always the ROOT
	# object holding the field (never itself a GetAttr result), the same
	# "obj+attr, not obj already reduced to the field's own value" shape
	# AddrOfField uses and for the same reason: a real C array member
	# decays to a pointer on use, but is never itself a loadable VALUE (no
	# `dest = (obj).field;` exists to build on) - so this is a distinct
	# instruction rather than GetAttr+GetItem composed, letting emission
	# spell one flat `(obj)OP field[index]` expression directly against the
	# field's real storage. Unchecked (no bounds check emitted), matching
	# Ptr[T]/ConstPtr[T]'s own GetItem convention - see lowering.py's
	# _lower_fixed_array_index for the one bit of free compile-time
	# checking a LITERAL constant index still gets, same as tuple indexing.
	dest: Temp
	obj: Operand
	attr: str
	index: Operand

	def test_repr( self ) -> str:
		return f'GetAttrIndex( dest={self.dest!r}, obj={self.obj!r}, attr={self.attr!r}, index={self.index!r} )'

@dataclass( kw_only = True )
class SetAttrIndex( Instruction ):
	# f.arr[i] = value - element-level write, the SetItem-shaped sibling of
	# GetAttrIndex above (see its own docstring). Targets the field's REAL
	# storage in place, same "obj is always the root, one flat `(obj)OP
	# field[index] = value` expression" reasoning as AddrOfField.
	obj: Operand
	attr: str
	index: Operand
	value: Operand

	def test_repr( self ) -> str:
		return f'SetAttrIndex( obj={self.obj!r}, attr={self.attr!r}, index={self.index!r}, value={self.value!r} )'

# Arithmetic (mode-specific opcodes) + bitwise + unary.
#
# dest's type differs by mode: Wrap/Saturate produce a plain T; Check
# produces Result[T,OverflowError]; Div/Mod (always checked, but against
# zero rather than overflow) produce Result[T,ZeroDivisionError] - matching
# Python's own exception name. It's lowering's responsibility to pick the
# right T/Result[T,E] type when it builds the Temp, this module just
# distinguishes the opcodes.

@dataclass( kw_only = True )
class BinOp( Instruction ):
	dest: Temp
	left: Operand
	right: Operand

	# lowering support - class-level tags, never per-instance data (see
	# arithmetic_mode.py / lowering.py's _lower_arithmetic_op, the only readers):
	#
	# checked_errors: the error class name(s) a Check-mode opcode's dest.type is
	# Result[T, <error>] against. Empty for a plain (non-Result) opcode. When
	# more than one, the op's error type is the anonymous UNION of them (built +
	# interned by discovery._get_or_create_union). ClassVars so subclasses can
	# override them as bare class attributes without becoming dataclass __init__
	# parameters whose base-class default would otherwise shadow them.
	#
	# signed_only: the subset of checked_errors that applies ONLY when the
	# result type is a SIGNED integer - lowering filters these out for unsigned
	# operands. Currently just OverflowError on signed Div/Mod (INT_MIN/-1);
	# unsigned division can only ever raise ZeroDivisionError.
	checked_errors: ClassVar[tuple[str,...]] = ()
	signed_only: ClassVar[frozenset[str]] = frozenset()

	# test support:
	def test_repr( self ) -> str:
		# lives on the base class - every op x mode subclass below adds no
		# fields of its own, so type(self).__name__ is all that needs to vary
		return f'{type(self).__name__}( dest={self.dest!r}, left={self.left!r}, right={self.right!r} )'

class AddWrap( BinOp ): pass
class AddCheck( BinOp ): checked_errors = ( 'OverflowError', ) # dest.type is Result[T,OverflowError]
class AddSaturate( BinOp ): pass

class SubWrap( BinOp ): pass
class SubCheck( BinOp ): checked_errors = ( 'OverflowError', ) # dest.type is Result[T,OverflowError]
class SubSaturate( BinOp ): pass

class MulWrap( BinOp ): pass
class MulCheck( BinOp ): checked_errors = ( 'OverflowError', ) # dest.type is Result[T,OverflowError]
class MulSaturate( BinOp ): pass

class ShlWrap( BinOp ): pass
class ShlCheck( BinOp ): checked_errors = ( 'OverflowError', ) # dest.type is Result[T,OverflowError]
class ShlSaturate( BinOp ): pass

# integer division/modulo. Mode-specific because signed INT_MIN/-1 (the one
# value that overflows a division, UB in C) is handled differently per mode:
# checked/panic -> OverflowError (a second error alongside the always-present
# ZeroDivisionError, hence the union - but ONLY for signed operands, see
# signed_only); wrap/saturate handle it inline in the emitter (wrapped/saturated
# result, no error), so they only ever raise ZeroDivisionError.
class Div( BinOp ): checked_errors = ( 'ZeroDivisionError', 'OverflowError' ); signed_only = frozenset({ 'OverflowError' })
class Mod( BinOp ): checked_errors = ( 'ZeroDivisionError', 'OverflowError' ); signed_only = frozenset({ 'OverflowError' })
class DivWrap( BinOp ): checked_errors = ( 'ZeroDivisionError', ) # INT_MIN/-1 wraps to INT_MIN inline; only r==0 is an error
class DivSaturate( BinOp ): checked_errors = ( 'ZeroDivisionError', ) # INT_MIN/-1 saturates to INT_MAX inline
class ModWrap( BinOp ): checked_errors = ( 'ZeroDivisionError', ) # INT_MIN%-1 == 0 inline
class ModSaturate( BinOp ): checked_errors = ( 'ZeroDivisionError', ) # INT_MIN%-1 == 0 inline

# --- floating-point (f32/f64) arithmetic ---
# Floats have no integer overflow concept - IEEE 754 yields inf/nan, never
# traps - so they don't reuse the Wrap/Check/Saturate integer opcodes. Instead:
# checked/panic mode uses the *Check variants below (result inf/nan -> Result[T,
# FloatingPointError]); wrap/saturate mode uses the plain variants (raw IEEE, no
# error). See arithmetic_mode.py (GetFloatBinOp/GetFloatUnaryOp/GetFloatCast)
# for the mode->opcode mapping, and lowering.py's _is_float_scalar for routing.
# Checked float division (FloatDivCheck) can raise EITHER ZeroDivisionError
# (divisor 0) OR FloatingPointError (result inf/nan, incl. from a non-finite
# operand that flowed in from a prior wrap-mode block) - hence a union; wrap/
# saturate float / uses the plain FloatDiv (no error at all).
class FloatDiv( BinOp ): pass # plain IEEE l/r, no zero-check (wrap/saturate)
class FloatDivCheck( BinOp ): checked_errors = ( 'ZeroDivisionError', 'FloatingPointError' ) # dest.type is Result[T, ZeroDivisionError|FloatingPointError]
class FAddCheck( BinOp ): checked_errors = ( 'FloatingPointError', ) # dest.type is Result[T,FloatingPointError]
class FSubCheck( BinOp ): checked_errors = ( 'FloatingPointError', ) # dest.type is Result[T,FloatingPointError]
class FMulCheck( BinOp ): checked_errors = ( 'FloatingPointError', ) # dest.type is Result[T,FloatingPointError]

class BitAnd( BinOp ): pass
class BitOr( BinOp ): pass
class BitXor( BinOp ): pass
class Shr( BinOp ): pass

# Ptr[T]/ConstPtr[T] - Ptr[T]/ConstPtr[T] -> isize: raw byte distance between
# two pointers (never sizeof(T)-scaled, consistent with this compiler's other
# pointer arithmetic - see emitter_c.py's _is_pointer_type). Infallible: an
# address difference can't meaningfully overflow/underflow the way pointer
# ADDITION can against a fixed-size buffer, so unlike Add/Sub there's no
# Wrap/Check/Saturate split here, just one opcode.
class PtrDiff( BinOp ): pass

@dataclass( kw_only = True )
class UnaryOp( Instruction ):
	dest: Temp
	operand: Operand

	# lowering support - see BinOp.checked_errors/signed_only above, same reasoning
	checked_errors: ClassVar[tuple[str,...]] = ()
	signed_only: ClassVar[frozenset[str]] = frozenset()

	# test support:
	def test_repr( self ) -> str:
		return f'{type(self).__name__}( dest={self.dest!r}, operand={self.operand!r} )'

class Invert( UnaryOp ): pass
class NegWrap( UnaryOp ): pass
class NegCheck( UnaryOp ): checked_errors = ( 'OverflowError', ) # dest.type is Result[T,OverflowError]
class NegSaturate( UnaryOp ): pass

class CastWrap( UnaryOp ): pass
class CastCheck( UnaryOp ): checked_errors = ( 'OverflowError', ) # dest.type is Result[T,OverflowError]
class CastSaturate( UnaryOp ): pass

# compiler.checked_convert(T, x) - deliberately separate from CastCheck, not
# a reuse: CastCheck's own range check only applies to a WIDTH-CHANGING
# (narrowing) conversion - same-width/widening always succeed unconditionally
# (T(x) construct-cast syntax, see _lower_scalar_cast). ConvertCheck's own
# check is a genuine VALUE-range comparison against the target type's own
# [MIN,MAX], independent of width - it can fail even for a same-width,
# cross-signedness conversion (i8(-1).to_u8() must fail; u8(i8(-1)) via
# construct-cast syntax never does). See SYNTAX.md's own T(x)-vs-.to_T()
# section for the full rationale.
class ConvertCheck( UnaryOp ): checked_errors = ( 'OverflowError', ) # dest.type is Result[T,OverflowError]

# float-involving scalar casts (see the float-arithmetic note above). Unary `-`
# on a float always reuses the plain NegWrap opcode (negation never introduces
# inf/nan, so there's nothing to check), so no float negate opcode is needed.
# FloatCastCheck covers BOTH directions (to-float: result inf/nan check; float->
# int: source out-of-range/nan check) - the emitter branches on target/source.
# FloatToIntClamp is the wrap/saturate float->int cast (clamps nan->0 and out-of-
# range->MIN/MAX so it's never UB - wrap==saturate here per the plan).
class FloatCastCheck( UnaryOp ): checked_errors = ( 'FloatingPointError', ) # dest.type is Result[T,FloatingPointError]
class FloatToIntClamp( UnaryOp ): pass # plain clamping float->int, no error

@dataclass( kw_only = True )
class Not( Instruction ): # boolean negation: dest = !operand
	dest: Temp
	operand: Operand

	def test_repr( self ) -> str:
		return f'Not( dest={self.dest!r}, operand={self.operand!r} )'

@dataclass( kw_only = True )
class MarkUsed( Instruction ):
	''' no runtime effect - marks operand as read without actually reading
	it, purely to keep the C compiler from flagging its already-completed
	definition as dead (-Wunused-variable/-Wunused-but-set-variable/
	C4189). Emitted only where lowering.py already knows operand's real
	definition happened and won't be read again through any other path -
	e.g. compiler.decref(x)/compiler.incref(x) silently no-op for a non-RC
	x inside a monomorphized generic-class method (_in_generic_class_
	method) rather than failing, so x can end up with no other reader at
	all in that specific instantiation (list[i32].__del__'s `val`, never
	RC, only ever passed to decref). NEVER emit this before operand's real
	definition - that would silence a genuine uninitialized-value bug
	instead of a spurious warning. '''
	operand: Operand

	def test_repr( self ) -> str:
		return f'MarkUsed( operand={self.operand!r} )'

# Result-consuming ops - Check-mode arithmetic and Div/Mod hand back a
# Result[T,OverflowError] rather than panicking inline. These mirror the real
# methods already defined on builtins.Result (or_return, unwrap, unwrap_or in
# lib/builtins/__init__.py) - they're the IR-level primitives those library
# methods lower to, not new behavior.

@dataclass( kw_only = True )
class OrReturn( Instruction ): # Result.or_return(): Err -> return Err from the current function; Ok -> dest = payload
	dest: Temp
	value: Operand # a Result[T,E]
	# extra cleanup to replay on the Err branch, BEFORE the return - only
	# non-empty when `value` is itself a named, tracked local binding whose own
	# ordinary scope-exit decref must be excluded from replay here (it's being
	# moved into the return value, not independently released) while every
	# OTHER still-live binding/defer/errdefer obligation still needs its normal
	# cleanup on this early-exit path. Built via the same CFGState.return_()
	# primitive _stmt_Return already uses for the identical "returning a
	# tracked operand" situation - see lowering.py's _consume_checked_result.
	# Emitted inside emitter_c.py's own `if (value.tag==1) {...}` block, after
	# the error-widening lines and before the actual `return`.
	epilogue: list['Instruction'] = field( default_factory = list )
	# PLAN_INLINE.md early-return generalization: (result_var, exited_flag,
	# merge_label), or None (default - preserves today's exact behavior:
	# widen and emit a real C `return`). When set, this OrReturn is
	# .or_return()/checked-arithmetic's own "no shared label" inline-unwind
	# path (lowering.py's _consume_checked_result), reached from inside a
	# multi-statement @inline splice's pre-return statements - a literal C
	# `return` there would incorrectly return from the CALLER, not just
	# exit the splice (self._current_fn is briefly the caller during that
	# window too). The Err branch instead widens into result_var (typed
	# from the INLINED target's own return type, not the enclosing C
	# function's - see emitter_c.py's own comment on why function.
	# return_type can't be used here), arms exited_flag, and `goto`s
	# merge_label - see _splice_multi_statement_inline_body's own comment
	# for what happens there.
	inline_exit: 'tuple[Variable,Variable,str]|None' = None

	def test_repr( self ) -> str:
		return f'OrReturn( dest={self.dest!r}, value={self.value!r} )'

@dataclass( kw_only = True )
class WidenResult( Instruction ):
	''' a bare `return x` where x is Result[T,NarrowE] and the enclosing
	function is declared -> Result[T,WideE], with WideE covering NarrowE
	(type_resolver's leaves-containment check - see lowering.py's
	_stmt_Return) - unlike OrReturn/OrJump (which only ever widen the ERR
	branch, since the OK branch means "continue executing, not return"), a
	bare return exits unconditionally on EITHER branch, so dest.type
	(Result[T,WideE]) gets built from src on BOTH: Ok is a plain field copy
	(same T on both sides, nothing to widen), Err reuses emitter_c.py's
	_emit_widen_error. dest is then what the surrounding (otherwise
	UNCHANGED) _stmt_Return return-emission logic actually returns/assigns
	into the return-value slot - src itself is what still gets passed to
	CFGState.current_epilogue_label()/return_() for the identity-based
	"this operand's own epilogue entry is being moved out" exclusion, exactly
	as for an ordinary, non-widened return of a tracked binding. '''
	dest: Temp
	src: Operand # a Result[T,NarrowE]

	def test_repr( self ) -> str:
		return f'WidenResult( dest={self.dest!r}, src={self.src!r} )'

@dataclass( kw_only = True )
class OrJump( Instruction ):
	''' like OrReturn, but jumps to an epilogue label instead of returning
	directly - used once the current function has any defer/errdefer active.
	Its (opaque, stage-3-implemented) error branch stows Result::Err(...)
	into return_slot and jumps to target instead of returning; its success
	branch (dest = payload) is identical to OrReturn's. errdefer doesn't need
	a separate signal for "was this an error exit" - the epilogue checks
	return_slot's own tag directly (via .is_err()), which this naturally sets. '''
	dest: Temp
	value: Operand # a Result[T,E]
	target: str # epilogue label
	return_slot: Variable|None # where the wrapped error gets stowed before jumping; None if the function returns None
	# PLAN_INLINE.md early-return generalization: set only when `target` is a
	# multi-statement @inline splice's own local epilogue label (rather than
	# the enclosing real function's) - armed (set true) right before the
	# `goto`, alongside return_slot, so the splice's own ladder tail (see
	# _splice_multi_statement_inline_body) can tell "early exit vs normal
	# fallthrough" apart. None (the default) for every ordinary, non-spliced
	# OrJump - unchanged behavior.
	exited_flag: Variable|None = None

	def test_repr( self ) -> str:
		return f'OrJump( dest={self.dest!r}, value={self.value!r}, target={self.target!r}, return_slot={self.return_slot!r} )'

@dataclass( kw_only = True )
class ThrowLeaf:
	''' one covered leaf of an ir.OrThrow's own error type - matched by
	identity against the Err branch's runtime tag (same identity-leaf
	convention _atomic_leaves/_union_member already use). `bind` is a real
	local Variable (the handler's own TryHandler.raise_value_var - `except
	T as e:` binds it to `e` itself, a bare `except T:` gets a hidden
	compiler-synthesized one instead, needed for bare `raise` re-raise
	support even without a user-facing name) - emitter_c.py assigns the
	narrowed payload into it, as an ordinary already-registered local,
	before jumping to `label`. `epilogue` replays whatever's still
	pending CONFINED to the covering try's own body (RC decrefs, defer/
	errdefer) - everything pushed since that try's own entry snapshot,
	excluding the leaf's own payload (which transfers into `bind`
	instead) - a goto straight into a handler never otherwise unwinds
	any of that (unlike a real function-level return/propagation - see
	ir.Raise's own docstring), so without this it silently leaks:
	confirmed by a real repro, an ordinary RC local declared earlier in
	the try body, never touched again, leaking every time a later
	covered `raise`/or_throw() dispatches past it. Emitted (see
	emitter_c.py's own _emit_leaf_dispatch_case) right before the
	assignment into `bind`. '''
	leaf: Type
	bind: 'Variable|None'
	label: str
	epilogue: 'list[Instruction]' = field( default_factory = list )

@dataclass( kw_only = True )
class OrThrow( Instruction ):
	''' Result.or_throw(): like OrReturn/OrJump, but each leaf of the Err
	branch's error type is checked against `dispatch` first - a leaf
	matched there jumps straight into that except handler (binding its
	payload if the clause names one), NEVER touching the enclosing
	function's own return type at all. Only a leaf with NO entry in
	dispatch falls back to exactly OrReturn's (target is None) or OrJump's
	(target is a real epilogue label) own propagate-to-caller behavior, OR
	(inside a multi-statement @inline splice's pre-return statements) the
	splice-local inline_exit shape below - `epilogue`/`target`/
	`return_slot` mirror OrReturn/OrJump exactly, and are only ever
	consulted along that uncovered-leaf path (built and legality-checked
	by lowering.py only when at least one leaf is actually uncovered - see
	_lower_or_throw). '''
	dest: Temp
	value: Operand # a Result[T,E]
	dispatch: list[ThrowLeaf]
	epilogue: list['Instruction'] = field( default_factory = list )
	target: str|None = None # None -> real C `return`, matching OrReturn; a real label -> `goto`, matching OrJump
	return_slot: 'Variable|None' = None # only meaningful when target is not None
	# PLAN_INLINE.md early-return generalization, same shape as OrReturn's own
	# inline_exit field - set only for the uncovered-leaf fallback, reached
	# from inside a multi-statement @inline splice's pre-return statements:
	# widens into result_var (the SPLICE TARGET's own return type, not
	# target/return_slot's enclosing-function one), arms exited_flag, and
	# `goto`s merge_label instead of returning/jumping to a real epilogue.
	# Mutually exclusive with target/return_slot being meaningfully set - see
	# lowering.py's _emit_or_throw.
	inline_exit: 'tuple[Variable,Variable,str]|None' = None

	def test_repr( self ) -> str:
		return f'OrThrow( dest={self.dest!r}, value={self.value!r}, dispatch={self.dispatch!r}, target={self.target!r} )'

@dataclass( kw_only = True )
class Raise( Instruction ):
	''' `raise EXPR` (lowering.py's _stmt_Raise) - same per-leaf dispatch
	shape as ir.OrThrow's own Err branch, but `value` here IS the error
	itself (never a Result - contrast OrThrow.value), so there's no outer
	Ok/Err tag to check first and no `dest`/Ok-arm at all: a raise never
	falls through, there's no "otherwise" value. A covered leaf narrows
	`value`'s payload into its handler's own bind and jumps straight to
	`label` - no Result is ever built. An uncovered leaf propagates via a
	REAL function return, built directly from the enclosing function's own
	declared Result[T,E] return type (or return_slot's, when target is a
	real epilogue label) - emitter_c.py reuses the exact same
	_emit_widen_error-based machinery ir.OrThrow's own uncovered-leaf arm
	already uses, just seeded from `value` directly instead of `(receiver).
	data.err`. `dispatch`/`epilogue`/`target`/`return_slot`/`inline_exit`
	mirror ir.OrThrow's own identical fields. '''
	value: Operand # the raised error value itself - NOT a Result
	dispatch: list[ThrowLeaf]
	epilogue: list['Instruction'] = field( default_factory = list )
	target: str|None = None # None -> real C `return`; a real label -> `goto`
	return_slot: 'Variable|None' = None # only meaningful when target is not None
	inline_exit: 'tuple[Variable,Variable,str]|None' = None # see ir.OrThrow's own identical field

	def test_repr( self ) -> str:
		return f'Raise( value={self.value!r}, dispatch={self.dispatch!r}, target={self.target!r} )'

@dataclass( kw_only = True )
class Unwrap( Instruction ): # Result.unwrap(errmsg): Err -> panic(errmsg); Ok -> dest = payload
	dest: Temp
	value: Operand # a Result[T,E]
	errmsg: Operand
	# the resolved sys.panic(message: str) -> NoReturn to call on the Err
	# branch - a real Function reference (same posture as Call.target),
	# not a name an emitter has to know/invent on its own. Populated by
	# lowering.py's _consume_checked_result via Lowering._resolve_sys_function
	panic: Function

	def test_repr( self ) -> str:
		return f'Unwrap( dest={self.dest!r}, value={self.value!r}, errmsg={self.errmsg!r}, panic={self.panic.qualname!r} )'

@dataclass( kw_only = True )
class UnwrapOr( Instruction ): # Result.unwrap_or(default): Err -> dest = default; Ok -> dest = payload
	dest: Temp
	value: Operand # a Result[T,E]
	default: Operand

	def test_repr( self ) -> str:
		return f'UnwrapOr( dest={self.dest!r}, value={self.value!r}, default={self.default!r} )'

class CmpOp( Enum ):
	EQ = 'eq'
	NE = 'ne'
	LT = 'lt'
	LE = 'le'
	GT = 'gt'
	GE = 'ge'

@dataclass( kw_only = True )
class Cmp( Instruction ):
	dest: Temp
	op: CmpOp
	left: Operand
	right: Operand

	def test_repr( self ) -> str:
		return f'Cmp( dest={self.dest!r}, op={self.op!r}, left={self.left!r}, right={self.right!r} )'

@dataclass( kw_only = True )
class Label( Instruction ):
	name: str

	def test_repr( self ) -> str:
		return f'Label( name={self.name!r} )'

@dataclass( kw_only = True )
class Jump( Instruction ):
	target: str

	def test_repr( self ) -> str:
		return f'Jump( target={self.target!r} )'

@dataclass( kw_only = True )
class JumpIfFalse( Instruction ):
	cond: Operand
	target: str

	def test_repr( self ) -> str:
		return f'JumpIfFalse( cond={self.cond!r}, target={self.target!r} )'

@dataclass( kw_only = True )
class JumpIfTrue( Instruction ):
	cond: Operand
	target: str

	def test_repr( self ) -> str:
		return f'JumpIfTrue( cond={self.cond!r}, target={self.target!r} )'

@dataclass( kw_only = True )
class Call( Instruction ):
	dest: Temp|None # None for a call whose result is discarded
	target: Function
	# the bound instance for a method call (self/cls are excluded from
	# Function.parameters entirely - see discovery.py's _make_function_resolver
	# - so there's nothing in args/kwargs to carry it). None for a free
	# function, staticmethod, or classmethod call.
	receiver: Operand|None = None
	args: list[Operand]
	kwargs: dict[str,Operand]
	# super().__init__(...) specifically (lowering.py's own
	# _lower_super_init_if_required) - the receiver here is `self`,
	# mid-construction, which cfg.py's check_self_escape would otherwise
	# reject (self can't be handed to an ordinary call before construction
	# completes) - this one narrow exemption lets lowering.py's own
	# _check_self_escape_in skip just the RECEIVER of exactly this call
	# (args/kwargs still go through the ordinary check), matching how
	# GetAttr.obj/SetAttr.obj are already structurally exempted rather
	# than individually special-cased at every call site
	is_super_init_call: bool = False

	def test_repr( self ) -> str:
		return (
			f'Call( dest={self.dest!r}, target={self.target.qualname!r}, receiver={self.receiver!r}, '
			f'args={self.args!r}, kwargs={self.kwargs!r} )'
		)

@dataclass( kw_only = True )
class CallIndirect( Instruction ):
	''' calling THROUGH a Ptr[Callable[...]]-typed value, as opposed to
	Call's own "target is a real, named Function" shape - target here is
	just an Operand (typically a Parameter/Variable holding a function
	pointer, or a FunctionRef taken and called in the same expression) with
	no qualname/resolve() of its own to reason about. No kwargs (a
	CallableType's own shape is purely positional - see PLAN_CALLABLE.md). '''
	dest: Temp|None
	target: Operand
	args: list[Operand]

	def test_repr( self ) -> str:
		return f'CallIndirect( dest={self.dest!r}, target={self.target!r}, args={self.args!r} )'

@dataclass( kw_only = True )
class Allocate( Instruction ): # Foo.__allocate__( field = value, ... )
	dest: Temp
	cls: ClassLike
	fields: dict[str,Operand]
	loc: str|None = None # 'file:line' of the allocating source, debug-mode object tracking only (see emitter_c.py's dump_live_objects support) - set centrally by Lowering._emit, not by individual construction sites

	def test_repr( self ) -> str:
		return f'Allocate( dest={self.dest!r}, cls={self.cls.qualname!r}, fields={self.fields!r} )'

@dataclass( kw_only = True )
class Incref( Instruction ):
	value: Operand

	def test_repr( self ) -> str:
		return f'Incref( value={self.value!r} )'

@dataclass( kw_only = True )
class Decref( Instruction ):
	value: Operand

	def test_repr( self ) -> str:
		return f'Decref( value={self.value!r} )'

@dataclass( kw_only = True )
class DecrefDynamic( Instruction ): # compiler.decref_dynamic(ptr) - releases a type-erased Ptr[None] via release_object, which reads the destructor from the object's own header (see emitter_c.py's ObjectHeader.destructor) rather than a compile-time-known type - used by a synthesized closure's own __del__ to release its captured, type-erased receiver
	value: Operand

	def test_repr( self ) -> str:
		return f'DecrefDynamic( value={self.value!r} )'

@dataclass( kw_only = True )
class AcquireGlobalLock( Instruction ):
	''' PLAN_THREAD_SAFE_SHARED_STATE.md Part A - marks the START of the one
	critical section a protected global's reassignment needs: releasing its
	CURRENT value and overwriting it with the new one must happen as one
	atomic-with-respect-to-other-threads unit, never as two separate lock
	acquisitions (a reader could interleave in the gap otherwise - see that
	plan's own worked reader-vs-writer interleaving). Emitted directly by
	cfg.py's assign() (`dest.is_global` branch), wrapping whatever
	_decref_instructions(dest.type, dest) produces - which for a union-typed
	global (e.g. ZoneInfo|None) is a multi-instruction tag-gated sequence,
	not a single bare Decref, so the boundary of "everything that needs
	protecting" can only be known by whoever is CONSTRUCTING that sequence,
	not reconstructed later by pattern-matching the emitted instructions
	(confirmed unsound via a real test: the naive "look for an adjacent
	Decref+Assign" version this replaced silently never matched a union-
	typed global at all - exactly localtz()'s own shape, the bug that
	motivated this whole mechanism). '''
	var: Variable

	def test_repr( self ) -> str:
		return f'AcquireGlobalLock( var={self.var.qualname!r} )'

@dataclass( kw_only = True )
class ReleaseGlobalLock( Instruction ):
	''' the matching END marker for AcquireGlobalLock - emitted by lowering.py's
	_cfg_assign, immediately after the ir.Assign that overwrites the
	global's slot (the actual store cfg.assign() itself never emits - see
	_cfg_assign's own docstring for why both the RC-bookkeeping instructions
	and this trailing Assign have to come from one function body). '''
	var: Variable

	def test_repr( self ) -> str:
		return f'ReleaseGlobalLock( var={self.var.qualname!r} )'

@dataclass( kw_only = True )
class AcquireFieldLock( Instruction ):
	''' PLAN_THREAD_SAFE_SHARED_STATE.md Part B - the AcquireGlobalLock/
	ReleaseGlobalLock pair's per-OBJECT counterpart: marks the START of the
	one critical section a single instance-field READ or WRITE needs (B.3's
	"lock the access, not the statement" - never spans more than one field
	access, so two accesses to the same object's fields in the same
	statement/method get two separate critical sections, not one wrapping
	both - this is what keeps B.4's same-thread reentrancy hazard from ever
	materializing for straight-line code). `obj` is the RECEIVER operand
	(not a Variable, unlike AcquireGlobalLock's `var` - the lock lives in
	the object's OWN ObjectHeader, keyed off whichever expression currently
	holds the reference, not off any particular binding of it). '''
	obj: Operand

	def test_repr( self ) -> str:
		return f'AcquireFieldLock( obj={self.obj!r} )'

@dataclass( kw_only = True )
class ReleaseFieldLock( Instruction ):
	''' the matching END marker for AcquireFieldLock. '''
	obj: Operand

	def test_repr( self ) -> str:
		return f'ReleaseFieldLock( obj={self.obj!r} )'

@dataclass( kw_only = True )
class RefCount( Instruction ): # compiler.refcount(x) - reads x's current header refcount
	dest: Temp
	value: Operand

	def test_repr( self ) -> str:
		return f'RefCount( dest={self.dest!r}, value={self.value!r} )'

@dataclass( kw_only = True )
class AddrOf( Instruction ): # compiler.addrof(x) - yields &x, x a local variable/parameter
	dest: Temp
	value: Operand

	def test_repr( self ) -> str:
		return f'AddrOf( dest={self.dest!r}, value={self.value!r} )'

@dataclass( kw_only = True )
class AddrOfField( Instruction ):
	# compiler.addrof(x.field) - yields &(x.field)/&(x->field) directly, one
	# level of field access on a bare local/parameter x (see lowering.py's
	# _lower_compiler_addrof for why deeper chains/non-Name roots aren't
	# accepted). Distinct from AddrOf(GetAttr(...)) - GetAttr loads a COPY of
	# the field's value into a fresh temp, whose address would be the copy's,
	# not the real field's (useless for the FFI out-parameter idiom this
	# exists for, e.g. inet_pton(af, str, &addr.sin_addr) needs the callee to
	# write into `addr` itself). obj is always the ROOT object (never itself
	# a GetAttr result) so emission can spell one flat `&(obj)OP field`
	# expression, OP chosen the same way GetAttr/SetAttr already choose it
	# (_member_access_operator - '.' for a plain value, '->' for an RCClass
	# instance or a raw Ptr[T]/ConstPtr[T]).
	dest: Temp
	obj: Operand
	attr: str

	def test_repr( self ) -> str:
		return f'AddrOfField( dest={self.dest!r}, obj={self.obj!r}, attr={self.attr!r} )'

@dataclass( kw_only = True )
class ArrayFieldPtr( Instruction ):
	# compiler.addrof(x.field) where field is a FixedArrayType (mpy_types.
	# FixedArrayType, `u8[8]`-style inline C array) - yields Ptr[ElemType]
	# pointing at the array's first element via C's own array-to-pointer
	# decay, e.g. `dest = (x.field);` / `dest = (x->field);` - deliberately
	# NOT `dest = &(x.field);` (that would be AddrOfField's own emission,
	# giving a DIFFERENT C type, ElemType(*)[N] - pointer-TO-array, not
	# pointer-to-element - a real type mismatch against the declared
	# Ptr[ElemType] destination, even though the underlying address value
	# is identical). Same "obj is always the ROOT object" shape as
	# AddrOfField/GetAttrIndex/SetAttrIndex, for the same reason.
	dest: Temp
	obj: Operand
	attr: str

	def test_repr( self ) -> str:
		return f'ArrayFieldPtr( dest={self.dest!r}, obj={self.obj!r}, attr={self.attr!r} )'

@dataclass( kw_only = True )
class AddrOfArrayIndex( Instruction ):
	# compiler.addrof(x.field[i]) where field is a FixedArrayType - yields
	# Ptr[ElemType] pointing at element i specifically (not the array's
	# start the way ArrayFieldPtr does), e.g. `dest = &(x.field[i]);` /
	# `dest = &(x->field[i]);`. A real, well-defined C operation (indexing
	# then &-ing gives ElemType* directly, no decay-vs-pointer-to-array
	# ambiguity the way ArrayFieldPtr's own bare-array-decay case has).
	# Same "obj is always the ROOT object" shape as AddrOfField/
	# ArrayFieldPtr/GetAttrIndex/SetAttrIndex.
	dest: Temp
	obj: Operand
	attr: str
	index: Operand

	def test_repr( self ) -> str:
		return f'AddrOfArrayIndex( dest={self.dest!r}, obj={self.obj!r}, attr={self.attr!r}, index={self.index!r} )'

class AtomicRMWOp( Enum ): # compiler.atomic_add/atomic_sub/atomic_exchange - fetch-and-op, dest gets the value BEFORE the op
	ADD = 'add'
	SUB = 'sub'
	EXCHANGE = 'exchange'

@dataclass( kw_only = True )
class AtomicLoad( Instruction ): # compiler.atomic_load(ptr) - ptr: Ptr[T], T a scalar (see lowering.py's _lower_compiler_atomic_load)
	dest: Temp
	ptr: Operand

	def test_repr( self ) -> str:
		return f'AtomicLoad( dest={self.dest!r}, ptr={self.ptr!r} )'

@dataclass( kw_only = True )
class AtomicStore( Instruction ): # compiler.atomic_store(ptr, val)
	ptr: Operand
	value: Operand

	def test_repr( self ) -> str:
		return f'AtomicStore( ptr={self.ptr!r}, value={self.value!r} )'

@dataclass( kw_only = True )
class AtomicRMW( Instruction ): # compiler.atomic_add/atomic_sub/atomic_exchange(ptr, val)
	dest: Temp
	op: AtomicRMWOp
	ptr: Operand
	value: Operand

	def test_repr( self ) -> str:
		return f'AtomicRMW( dest={self.dest!r}, op={self.op!r}, ptr={self.ptr!r}, value={self.value!r} )'

@dataclass( kw_only = True )
class MarkUnused( Instruction ): # `(void)value;` - explicitly discards an intentionally-unread value (e.g. compiler.atomic_sub()'s fetch-before-op result, called as a bare statement for its side effect only) so -Wunused-but-set-variable/C4189 doesn't fire on the temp that holds it
	value: Operand

	def test_repr( self ) -> str:
		return f'MarkUnused( value={self.value!r} )'

@dataclass( kw_only = True )
class AtomicCompareExchange( Instruction ): # compiler.atomic_compare_exchange(ptr, expected, desired) -> bool - C11 strong CAS; *expected is updated to the current value on failure
	dest: Temp
	ptr: Operand
	expected: Operand # Ptr[T] - the lvalue C11 atomic_compare_exchange_strong writes the actual current value into on failure
	desired: Operand

	def test_repr( self ) -> str:
		return f'AtomicCompareExchange( dest={self.dest!r}, ptr={self.ptr!r}, expected={self.expected!r}, desired={self.desired!r} )'

@dataclass( kw_only = True )
class FormatFloat( Instruction ): # compiler.format_f64(buf, size, precision, type_char, alt, value) - writes value's MAGNITUDE (no sign) into buf per a printf-style type_char ('f'/'F'/'e'/'E'/'g'/'G', as its ASCII code), with `precision` meaning fractional digits for 'f'/'F'/'e'/'E' or significant digits for 'g'/'G', and `alt` the '#' flag (always show the decimal point / keep trailing zeros) - returns the byte count written (see lowering.py's _lower_compiler_format_f64)
	dest: Temp
	buf: Operand
	size: Operand
	precision: Operand
	type_char: Operand
	alt: Operand
	value: Operand

	def test_repr( self ) -> str:
		return f'FormatFloat( dest={self.dest!r}, buf={self.buf!r}, size={self.size!r}, precision={self.precision!r}, type_char={self.type_char!r}, alt={self.alt!r}, value={self.value!r} )'

@dataclass( kw_only = True )
class IsNan( Instruction ): # compiler.is_nan(x) - x: f32|f64, dest: bool - true iff x is NaN. Reuses __metalpy_isnan (emitter_c.py's PROLOGUE), already there for checked/panic-mode float arithmetic - this just exposes it to metalpy source directly (needed so f-string format specs can special-case NaN display, which real snprintf/msvcrt don't reliably produce "nan" text for - see lib/builtins/__float.py's own comment)
	dest: Temp
	value: Operand

	def test_repr( self ) -> str:
		return f'IsNan( dest={self.dest!r}, value={self.value!r} )'

@dataclass( kw_only = True )
class IsInf( Instruction ): # compiler.is_inf(x) - x: f32|f64, dest: bool - true iff x is +/-infinity. Reuses __metalpy_isinf (emitter_c.py's PROLOGUE), same rationale as IsNan above
	dest: Temp
	value: Operand

	def test_repr( self ) -> str:
		return f'IsInf( dest={self.dest!r}, value={self.value!r} )'

@dataclass( kw_only = True )
class ParseFloat( Instruction ): # compiler.parse_f64(ptr) - ptr: ConstPtr[u8] (null-terminated), dest: f64 - the inverse of compiler.format_f64: parses C text back into a double. Needed for the shortest-round-trip repr search (lib/builtins/__float.py's _f64_repr_digits_raw tries increasing precision and re-parses each candidate to check for an exact round-trip) - same "hand-written C helper in PROLOGUE, dynamically resolved on Windows, never an ordinary @extern binding" shape as compiler.format_f64 (see its own comment), here because strtod is a genuinely ordinary (non-variadic) function but tagging it 'c' would still wrongly flip the no-crt Windows build
	dest: Temp
	buf: Operand

	def test_repr( self ) -> str:
		return f'ParseFloat( dest={self.dest!r}, buf={self.buf!r} )'

@dataclass( kw_only = True )
class SizeOf( Instruction ): # compiler.sizeof(T) for a real ClassLike T - no field-layout
	# algorithm exists in this compiler (nor should one - that's the C
	# compiler's job), so unlike an intrinsic scalar's sizeof (which folds
	# straight to ir.Const), this stays a real instruction: the emitter emits
	# a literal C `sizeof(...)` expression, letting the target C compiler
	# compute the real, field-layout-dependent size
	dest: Temp
	type: Type

	def test_repr( self ) -> str:
		return f'SizeOf( dest={self.dest!r}, type={self.type.qualname!r} )'

# --- debug-mode alloc-site tracking (dump_live_objects, PLAN in
# i-want-to-investigate-kind-garden.md) - raw sys.alloc[T] buffers have no
# ObjectHeader of their own. Tracked via a SIDE TABLE (a small tracking node,
# allocated straight from the OS allocator, holding just {link, ptr, size})
# rather than a hidden prefix header in front of the real block: sys.alloc[T]
# must keep returning the EXACT pointer the OS allocator gave it, unchanged -
# confirmed necessary by a real regression, not just caution: an earlier
# version of this feature offset the returned pointer past a prefix header,
# which broke sys_free_mempoison_test.py's direct HeapSize(ptr) query (HeapSize
# requires the literal block-start pointer HeapAlloc returned; any offset
# pointer is a hard crash, not just a wrong answer) - some existing code
# legitimately queries the OS allocator directly on a sys.alloc'd pointer, so
# that pointer's identity has to stay exactly what the OS handed back.
# Threaded into a SEPARATE global list from the RC one (not the RC objects'
# list - the two header shapes differ, so keeping them apart avoids any
# runtime type-tag/reinterpret-cast dance when dump_live_objects walks
# either). debug-only; compile_time_transformer folds every call site of
# these away entirely in a release build (same `if compiler.target.debug:`
# guard mempoison already uses), so emitter_c.py only ever sees these when
# _target_debug is True. ---

@dataclass( kw_only = True )
class DebugRawTrack( Instruction ): # compiler.__debug_raw_track__(ptr, size) -> None - records a freshly allocated raw sys.alloc[T] buffer in the side-table tracking list (best-effort: silently does nothing if the side allocation itself fails - never crashes the real allocation path)
	ptr: Operand
	size: Operand

	def test_repr( self ) -> str:
		return f'DebugRawTrack( ptr={self.ptr!r}, size={self.size!r} )'

@dataclass( kw_only = True )
class DebugRawUntrack( Instruction ): # compiler.__debug_raw_untrack__(ptr) -> None - removes ptr's side-table tracking entry (a no-op if ptr was never tracked, e.g. a release-mode-allocated pointer reaching a debug-mode free somehow - shouldn't happen, but this stays a safe no-op rather than a crash either way)
	ptr: Operand

	def test_repr( self ) -> str:
		return f'DebugRawUntrack( ptr={self.ptr!r} )'

@dataclass( kw_only = True )
class DebugQuarantine( Instruction ): # compiler.__debug_quarantine__(ptr) -> same Ptr[T] as ptr - the leak tracker's own pit (emitter_c.py's __metalpy_debug_pit_push): holds a just-freed block instead of handing it straight back to the allocator, so a double-free/UAF on it lands on reliably-poisoned memory instead of silently-reused memory. Returns whatever the pit evicted to make room for ptr (a genuine, no-longer-protected block the caller must now actually free) - null if the pit wasn't full yet. See lib/sys.py's free().
	dest: Temp
	ptr: Operand

	def test_repr( self ) -> str:
		return f'DebugQuarantine( dest={self.dest!r}, ptr={self.ptr!r} )'

@dataclass( kw_only = True )
class DumpLiveObjects( Instruction ): # compiler.dump_live_objects() - walks both debug-tracking lists (RC objects + raw sys.alloc buffers), aggregates by (type_name, alloc_loc), prints counts/bytes via _Stdout.write - see emitter_c.py's __metalpy_dump_live_objects
	def test_repr( self ) -> str:
		return 'DumpLiveObjects()'

@dataclass( kw_only = True )
class DebugUntrackRC( Instruction ): # debug-mode alloc tracking only (see DumpLiveObjects) - untracks an RC object's own debug_link WITHOUT going through release_object's normal refcount-hits-zero path. Needed by compiler.__raw_free__'s own codegen (Lowering._lower_compiler_raw_free): a not-yet-fully-alive RCClass whose __init__ failed is freed DIRECTLY via sys.free(), bypassing release_object entirely - confirmed as a real bug otherwise (not just theoretical): the object's own ir.Allocate already tracked it into the global RC list, so skipping this leaves a dangling entry pointing at memory that's about to be freed, which corrupts the list the moment anything else touches it (a real MSVC-only crash this fixed, root-caused via bisection - clang/gcc happened not to reorder/reuse the freed block in a way that tripped it, in the same debug-mode test run)
	value: Operand

	def test_repr( self ) -> str:
		return f'DebugUntrackRC( value={self.value!r} )'

@dataclass( kw_only = True )
class Return( Instruction ):
	value: Operand|None

	def test_repr( self ) -> str:
		return f'Return( value={self.value!r} )'

@dataclass( kw_only = True )
class Yield( Instruction ):
	''' PLAN_GENERATORS.md Phase F - a real `yield` suspend point inside a
	generator's $$__next__ body (Function.is_generator_next). Codegen
	(emitter_c.py) is deliberately trivial - `return value;`, the exact
	same shape ir.Return's own non-void/non-entry-point branch already
	emits ($$__next__ always has a concrete, non-void return type, and is
	never the program's entry point) - the state store (self.__state =
	state) is an ordinary, separate ir.SetAttr emitted immediately BEFORE
	this instruction, and the resume point is an ordinary, separate
	ir.Label(name=resume_label) emitted immediately AFTER it
	(lowering.py's own yield-lowering emits all three, back to back) -
	both already-proven, unmodified machinery, reused as-is rather than
	reimplemented inside this instruction's own codegen. A LATER call,
	dispatched via the function's own state-check prologue jumping
	straight to that label, resumes execution there. state/resume_label
	are carried here anyway (not read back by codegen at all) purely so a
	dumped/test_repr'd instruction stream is self-describing - state is
	this yield's own dispatch discriminant (unique per textual yield
	site, assigned by TypeResolver._assign_generator_yield_dispatch,
	starting at 1 - state 0 means "not yet started"). '''
	value: Operand # already coerced/wrapped to match the function's own declared return type (elem_type|None, or Result[elem_type|None,error_type] when fallible) - same convention ir.Return's own `value` field expects
	state: int
	resume_label: str

	def test_repr( self ) -> str:
		return f'Yield( value={self.value!r}, state={self.state}, resume_label={self.resume_label!r} )'
