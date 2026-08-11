# stdlib imports:
from dataclasses import dataclass
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
	value: bool|int|str|bytes|None # no float type exists in the language today

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

	# lowering support - a class-level tag, never per-instance data (see
	# arithmetic_mode.py, the only reader): which error a Check-mode opcode's
	# dest.type is Result[T,<checked_error>] against, or None for a plain
	# (non-Result) opcode. ClassVar so subclasses can override it as a bare
	# class attribute without it becoming a dataclass __init__ parameter
	# whose default (None, from this base class) would otherwise shadow it
	# on every instance
	checked_error: ClassVar[str|None] = None

	# test support:
	def test_repr( self ) -> str:
		# lives on the base class - every op x mode subclass below adds no
		# fields of its own, so type(self).__name__ is all that needs to vary
		return f'{type(self).__name__}( dest={self.dest!r}, left={self.left!r}, right={self.right!r} )'

class AddWrap( BinOp ): pass
class AddCheck( BinOp ): checked_error = 'OverflowError' # dest.type is Result[T,OverflowError]
class AddSaturate( BinOp ): pass

class SubWrap( BinOp ): pass
class SubCheck( BinOp ): checked_error = 'OverflowError' # dest.type is Result[T,OverflowError]
class SubSaturate( BinOp ): pass

class MulWrap( BinOp ): pass
class MulCheck( BinOp ): checked_error = 'OverflowError' # dest.type is Result[T,OverflowError]
class MulSaturate( BinOp ): pass

class ShlWrap( BinOp ): pass
class ShlCheck( BinOp ): checked_error = 'OverflowError' # dest.type is Result[T,OverflowError]
class ShlSaturate( BinOp ): pass

class Div( BinOp ): checked_error = 'ZeroDivisionError' # dest.type is Result[T,ZeroDivisionError]
class Mod( BinOp ): checked_error = 'ZeroDivisionError' # dest.type is Result[T,ZeroDivisionError]

class BitAnd( BinOp ): pass
class BitOr( BinOp ): pass
class BitXor( BinOp ): pass
class Shr( BinOp ): pass

@dataclass( kw_only = True )
class UnaryOp( Instruction ):
	dest: Temp
	operand: Operand

	# lowering support - see BinOp.checked_error above, same reasoning
	checked_error: ClassVar[str|None] = None

	# test support:
	def test_repr( self ) -> str:
		return f'{type(self).__name__}( dest={self.dest!r}, operand={self.operand!r} )'

class Invert( UnaryOp ): pass
class NegWrap( UnaryOp ): pass
class NegCheck( UnaryOp ): checked_error = 'OverflowError' # dest.type is Result[T,OverflowError]
class NegSaturate( UnaryOp ): pass

class CastWrap( UnaryOp ): pass
class CastCheck( UnaryOp ): checked_error = 'OverflowError' # dest.type is Result[T,OverflowError]
class CastSaturate( UnaryOp ): pass

@dataclass( kw_only = True )
class Not( Instruction ): # boolean negation: dest = !operand
	dest: Temp
	operand: Operand

	def test_repr( self ) -> str:
		return f'Not( dest={self.dest!r}, operand={self.operand!r} )'

# Result-consuming ops - Check-mode arithmetic and Div/Mod hand back a
# Result[T,OverflowError] rather than panicking inline. These mirror the real
# methods already defined on builtins.Result (or_return, unwrap, unwrap_or in
# lib/builtins/__init__.py) - they're the IR-level primitives those library
# methods lower to, not new behavior.

@dataclass( kw_only = True )
class OrReturn( Instruction ): # Result.or_return(): Err -> return Err from the current function; Ok -> dest = payload
	dest: Temp
	value: Operand # a Result[T,E]

	def test_repr( self ) -> str:
		return f'OrReturn( dest={self.dest!r}, value={self.value!r} )'

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

	def test_repr( self ) -> str:
		return f'OrJump( dest={self.dest!r}, value={self.value!r}, target={self.target!r}, return_slot={self.return_slot!r} )'

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

@dataclass( kw_only = True )
class Return( Instruction ):
	value: Operand|None
	
	def test_repr( self ) -> str:
		return f'Return( value={self.value!r} )'
