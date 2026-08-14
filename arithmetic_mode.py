# stdlib imports:
import ast
from typing import Type as PyType

# local imports:
import ir

'''
self._arithmetic_mode (lowering.py) is a stack of these, pushed/popped by
_stmt_With - see Lowering's own docstring for the with-block syntax each one
corresponds to. Each mode answers the same question three ways (GetCast/
GetUnaryOp/GetBinOp): given the AST node being lowered, which ir opcode
implements it under this mode, and what extra operand (if any - only
ArithmeticPanic ever supplies one) does a Check-mode opcode's Result get
consumed with. Split out of both ir.py (a pure IR-shape module with no
business inspecting `ast` nodes or picking opcodes) and lowering.py (per
explicit request - anything ir.py doesn't need to know about doesn't need to
live in lowering.py either).
'''

class ArithmeticMode:
	def GetCast( self ) -> tuple[PyType[ir.UnaryOp],str|None]:
		raise NotImplementedError()

	def GetUnaryOp( self, node: ast.UnaryOp ) -> tuple[PyType[ir.UnaryOp]|None,str|None]:
		if isinstance( node.op, ast.Invert ):
			return ir.Invert, None
		return None, None

	def GetBinOp( self, node: ast.BinOp ) -> tuple[PyType[ir.BinOp]|None,str|None]:
		# no overflow concept - always the same opcode, independent of the
		# active arithmetic mode (that only governs Add/Sub/Mult/Shl/casts)
		match type( node.op ):
			case ast.BitAnd:
				return ir.BitAnd, None
			case ast.BitOr:
				return ir.BitOr, None
			case ast.BitXor:
				return ir.BitXor, None
			case ast.RShift:
				return ir.Shr, None
			case _:
				return None, None

	# --- floating-point (f32/f64) variants -----------------------------------
	# The three Get* methods above pick INTEGER opcodes; these pick the FLOAT
	# ones. lowering.py calls these instead (never the integer versions) once it
	# sees a float operand/target (see _is_float_scalar). Floats have no integer
	# overflow, so the mode meaning shifts: checked/panic -> inf/nan is an error;
	# wrap/saturate -> raw IEEE, inf/nan produced silently. Bitwise/shift/
	# floordiv/mod on floats are rejected in lowering before these are reached.

	def GetFloatUnaryOp( self, node: ast.UnaryOp ) -> tuple[PyType[ir.UnaryOp]|None,str|None]:
		# unary `-` on a float is the same in every mode: plain negation can
		# never introduce inf/nan (negating inf/nan just flips the sign bit), so
		# there's nothing to check - always the plain NegWrap opcode. `~` (Invert)
		# on a float falls through to None -> lowering's "unsupported operator"
		if isinstance( node.op, ast.USub ):
			return ir.NegWrap, None
		return None, None

	def GetFloatBinOp( self, node: ast.BinOp ) -> tuple[PyType[ir.BinOp]|None,str|None]:
		raise NotImplementedError() # each concrete mode defines +,-,*,/ for floats

	def GetFloatCast( self, *, target_is_float: bool, source_is_float: bool ) -> tuple[PyType[ir.UnaryOp]|None,str|None]:
		raise NotImplementedError() # each concrete mode defines float-involving casts

class ArithmeticChecked( ArithmeticMode ):
	''' `with compiler.panic_arithmetic(...):` is layered directly on top of
	this one (ArithmeticPanic below) - same opcodes, the only difference is
	what `extra` the Result gets consumed with '''
	def GetCast( self ) -> tuple[PyType[ir.UnaryOp],str|None]:
		return ir.CastCheck, None

	def GetUnaryOp( self, node: ast.UnaryOp ) -> tuple[PyType[ir.UnaryOp]|None,str|None]:
		if isinstance( node.op, ast.USub ):
			return ir.NegCheck, None
		return super().GetUnaryOp( node )

	def GetBinOp( self, node: ast.BinOp ) -> tuple[PyType[ir.BinOp]|None,str|None]:
		match type( node.op ):
			case ast.Add:
				return ir.AddCheck, None
			case ast.Sub:
				return ir.SubCheck, None
			case ast.Mult:
				return ir.MulCheck, None
			case ast.LShift:
				return ir.ShlCheck, None
			case ast.FloorDiv:
				# checked/panic-mode division: signed INT_MIN/-1 -> OverflowError
				# alongside the always-present ZeroDivisionError (see ir.Div's
				# own checked_errors/signed_only). Wrap/Saturate use their own
				# DivWrap/DivSaturate below, which handle INT_MIN/-1 inline
				return ir.Div, None
			case ast.Mod:
				return ir.Mod, None
			case _:
				return super().GetBinOp( node )

	def GetFloatBinOp( self, node: ast.BinOp ) -> tuple[PyType[ir.BinOp]|None,str|None]:
		match type( node.op ):
			case ast.Add:
				return ir.FAddCheck, None
			case ast.Sub:
				return ir.FSubCheck, None
			case ast.Mult:
				return ir.FMulCheck, None
			case ast.Div:
				# checked/panic float `/` raises ZeroDivisionError (divisor 0)
				# OR FloatingPointError (result inf/nan) - a union, so it needs
				# its own opcode, NOT the integer Div (which raises Overflow-
				# Error for INT_MIN/-1, meaningless for floats)
				return ir.FloatDivCheck, None
			case _:
				return None, None # float //, %, bitwise: rejected in lowering, but be safe

	def GetFloatCast( self, *, target_is_float: bool, source_is_float: bool ) -> tuple[PyType[ir.UnaryOp]|None,str|None]:
		# a single checked opcode covers both directions - the emitter branches
		# on target/source (to-float: result inf/nan; float->int: source range/nan)
		return ir.FloatCastCheck, None

class ArithmeticWrap( ArithmeticMode ):
	def GetCast( self ) -> tuple[PyType[ir.UnaryOp],str|None]:
		return ir.CastWrap, None

	def GetUnaryOp( self, node: ast.UnaryOp ) -> tuple[PyType[ir.UnaryOp]|None,str|None]:
		if isinstance( node.op, ast.USub ):
			return ir.NegWrap, None
		return super().GetUnaryOp( node )

	def GetBinOp( self, node: ast.BinOp ) -> tuple[PyType[ir.BinOp]|None,str|None]:
		match type( node.op ):
			case ast.Add:
				return ir.AddWrap, None
			case ast.Sub:
				return ir.SubWrap, None
			case ast.Mult:
				return ir.MulWrap, None
			case ast.LShift:
				return ir.ShlWrap, None
			case ast.FloorDiv:
				return ir.DivWrap, None # INT_MIN/-1 wraps to INT_MIN inline (no error); only r==0 raises
			case ast.Mod:
				return ir.ModWrap, None
			case _:
				return super().GetBinOp( node )

	def GetFloatBinOp( self, node: ast.BinOp ) -> tuple[PyType[ir.BinOp]|None,str|None]:
		return _raw_float_binop( node ) # raw IEEE, identical for wrap & saturate

	def GetFloatCast( self, *, target_is_float: bool, source_is_float: bool ) -> tuple[PyType[ir.UnaryOp]|None,str|None]:
		return _raw_float_cast( target_is_float = target_is_float ) # identical for wrap & saturate

class ArithmeticSaturate( ArithmeticMode ):
	def GetCast( self ) -> tuple[PyType[ir.UnaryOp],str|None]:
		return ir.CastSaturate, None

	def GetUnaryOp( self, node: ast.UnaryOp ) -> tuple[PyType[ir.UnaryOp]|None,str|None]:
		if isinstance( node.op, ast.USub ):
			return ir.NegSaturate, None
		return super().GetUnaryOp( node )

	def GetBinOp( self, node: ast.BinOp ) -> tuple[PyType[ir.BinOp]|None,str|None]:
		match type( node.op ):
			case ast.Add:
				return ir.AddSaturate, None
			case ast.Sub:
				return ir.SubSaturate, None
			case ast.Mult:
				return ir.MulSaturate, None
			case ast.LShift:
				return ir.ShlSaturate, None
			case ast.FloorDiv:
				return ir.DivSaturate, None # INT_MIN/-1 saturates to INT_MAX inline (no error); only r==0 raises
			case ast.Mod:
				return ir.ModSaturate, None
			case _:
				return super().GetBinOp( node )

	def GetFloatBinOp( self, node: ast.BinOp ) -> tuple[PyType[ir.BinOp]|None,str|None]:
		return _raw_float_binop( node ) # raw IEEE, identical for wrap & saturate

	def GetFloatCast( self, *, target_is_float: bool, source_is_float: bool ) -> tuple[PyType[ir.UnaryOp]|None,str|None]:
		return _raw_float_cast( target_is_float = target_is_float ) # identical for wrap & saturate

class ArithmeticPanic( ArithmeticChecked ):
	''' `with compiler.panic_arithmetic(errmsg):` - identical opcode choices
	to the bare Check-mode default (ArithmeticChecked), just consumed with
	Unwrap(errmsg) instead of OrReturn - see Lowering's own docstring '''
	def __init__( self, extra: str|None = None ) -> None:
		self.extra = extra

	def GetCast( self ) -> tuple[PyType[ir.UnaryOp],str|None]:
		opcode, _ = super().GetCast()
		return opcode, self.extra

	def GetUnaryOp( self, node: ast.UnaryOp ) -> tuple[PyType[ir.UnaryOp]|None,str|None]:
		opcode, _ = super().GetUnaryOp( node )
		return opcode, self.extra

	def GetBinOp( self, node: ast.BinOp ) -> tuple[PyType[ir.BinOp]|None,str|None]:
		opcode, _ = super().GetBinOp( node )
		return opcode, self.extra

	def GetFloatBinOp( self, node: ast.BinOp ) -> tuple[PyType[ir.BinOp]|None,str|None]:
		# same checked opcodes as ArithmeticChecked (FAddCheck/.../ir.Div), just
		# consumed with Unwrap(errmsg) instead of OrReturn. GetFloatUnaryOp is
		# NOT overridden: it returns the plain NegWrap, which never faults, so
		# there's nothing to panic on
		opcode, _ = super().GetFloatBinOp( node )
		return opcode, self.extra

	def GetFloatCast( self, *, target_is_float: bool, source_is_float: bool ) -> tuple[PyType[ir.UnaryOp]|None,str|None]:
		opcode, _ = super().GetFloatCast( target_is_float = target_is_float, source_is_float = source_is_float )
		return opcode, self.extra

# shared by ArithmeticWrap and ArithmeticSaturate - float arithmetic is raw IEEE
# in both (no wrap/saturate distinction for a bare float result, since there's
# no clamp target), so both delegate here rather than duplicating the bodies.
# The two modes only differ for INTEGER arithmetic and for float->int casts...
# except float->int also clamps identically in both (wrap==saturate here, so it's
# never UB - see the plan), so even the cast is shared.
def _raw_float_binop( node: ast.BinOp ) -> tuple[PyType[ir.BinOp]|None,str|None]:
	match type( node.op ):
		case ast.Add:
			return ir.AddWrap, None # emits plain `l + r` for a float stem
		case ast.Sub:
			return ir.SubWrap, None
		case ast.Mult:
			return ir.MulWrap, None
		case ast.Div:
			return ir.FloatDiv, None # plain IEEE l/r, no zero-check
		case _:
			return None, None # float //, %, bitwise: rejected in lowering, but be safe

def _raw_float_cast( *, target_is_float: bool ) -> tuple[PyType[ir.UnaryOp]|None,str|None]:
	# to-float (int->float, f64->f32): a plain C cast (CastWrap emits
	# `(ctype)(x)`). float->int: clamp (FloatToIntClamp) so out-of-range/nan is
	# never UB
	return ( ir.CastWrap, None ) if target_is_float else ( ir.FloatToIntClamp, None )
