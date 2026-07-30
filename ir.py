# stdlib imports:
from dataclasses import dataclass
from enum import Enum
from typing import Union

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

Operand = Union[ Temp, Const, Variable ] # Variable already covers params/locals/globals via mpy_types

@dataclass( kw_only = True )
class Instruction:
	pass

@dataclass( kw_only = True )
class FuncStart( Instruction ):
	name: str
	params: list[Parameter]
	return_type: Type|None

@dataclass( kw_only = True )
class FuncEnd( Instruction ):
	name: str

@dataclass( kw_only = True )
class DeclareTemp( Instruction ):
	temp: Temp

@dataclass( kw_only = True )
class DeleteTemp( Instruction ): # temp is no longer valid after this instruction
	temp: Temp

@dataclass( kw_only = True )
class Assign( Instruction ):
	dest: Variable|Temp
	src: Operand

@dataclass( kw_only = True )
class GetAttr( Instruction ):
	dest: Temp
	obj: Operand
	attr: str

@dataclass( kw_only = True )
class SetAttr( Instruction ):
	obj: Operand
	attr: str
	value: Operand

@dataclass( kw_only = True )
class GetItem( Instruction ): # a[i]
	dest: Temp
	obj: Operand
	index: Operand

@dataclass( kw_only = True )
class SetItem( Instruction ):
	obj: Operand
	index: Operand
	value: Operand

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

class AddWrap( BinOp ): pass
class AddCheck( BinOp ): pass # dest.type is Result[T,OverflowError]
class AddSaturate( BinOp ): pass

class SubWrap( BinOp ): pass
class SubCheck( BinOp ): pass # dest.type is Result[T,OverflowError]
class SubSaturate( BinOp ): pass

class MulWrap( BinOp ): pass
class MulCheck( BinOp ): pass # dest.type is Result[T,OverflowError]
class MulSaturate( BinOp ): pass

class ShlWrap( BinOp ): pass
class ShlCheck( BinOp ): pass # dest.type is Result[T,OverflowError]
class ShlSaturate( BinOp ): pass

class Div( BinOp ): pass # dest.type is Result[T,ZeroDivisionError]
class Mod( BinOp ): pass # dest.type is Result[T,ZeroDivisionError]

class BitAnd( BinOp ): pass
class BitOr( BinOp ): pass
class BitXor( BinOp ): pass
class Shr( BinOp ): pass

@dataclass( kw_only = True )
class UnaryOp( Instruction ):
	dest: Temp
	operand: Operand

class Invert( UnaryOp ): pass
class NegWrap( UnaryOp ): pass
class NegCheck( UnaryOp ): pass # dest.type is Result[T,OverflowError]
class NegSaturate( UnaryOp ): pass

# Result-consuming ops - Check-mode arithmetic and Div/Mod hand back a
# Result[T,OverflowError] rather than panicking inline. These mirror the real
# methods already defined on builtins.Result (or_return, unwrap, unwrap_or in
# lib/builtins/__init__.py) - they're the IR-level primitives those library
# methods lower to, not new behavior.

@dataclass( kw_only = True )
class OrReturn( Instruction ): # Result.or_return(): Err -> return Err from the current function; Ok -> dest = payload
	dest: Temp
	value: Operand # a Result[T,E]

@dataclass( kw_only = True )
class Unwrap( Instruction ): # Result.unwrap(errmsg): Err -> panic(errmsg); Ok -> dest = payload
	dest: Temp
	value: Operand # a Result[T,E]
	errmsg: Operand

@dataclass( kw_only = True )
class UnwrapOr( Instruction ): # Result.unwrap_or(default): Err -> dest = default; Ok -> dest = payload
	dest: Temp
	value: Operand # a Result[T,E]
	default: Operand

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

@dataclass( kw_only = True )
class Label( Instruction ):
	name: str

@dataclass( kw_only = True )
class Jump( Instruction ):
	target: str

@dataclass( kw_only = True )
class JumpIfFalse( Instruction ):
	cond: Operand
	target: str

@dataclass( kw_only = True )
class JumpIfTrue( Instruction ):
	cond: Operand
	target: str

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

@dataclass( kw_only = True )
class Allocate( Instruction ): # Foo.__allocate__( field = value, ... )
	dest: Temp
	cls: ClassLike
	fields: dict[str,Operand]

@dataclass( kw_only = True )
class Incref( Instruction ):
	value: Operand

@dataclass( kw_only = True )
class Decref( Instruction ):
	value: Operand

@dataclass( kw_only = True )
class Return( Instruction ):
	value: Operand|None
