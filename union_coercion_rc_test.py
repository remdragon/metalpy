# Real-compile-and-run regression tests for two related RC bugs in the code
# that coerces a plain leaf value INTO a declared union type (T|None, or any
# other @union/Result[...]-shaped type) - lowering.py's _coerce_into_union,
# called from _coerce_or_check_operand right after every _expr_X dispatch
# (return, AnnAssign, Assign, attr default, attr_assign/attr_replace outside
# __init__, ternary branches, tuple elements, ...).
#
# _coerce_into_union wraps the leaf value by calling a synthesized union-
# member constructor (the same Function an explicit `Result.Ok(x)` call
# already resolves to) - `dest = self._new_temp(union); self._emit(ir.Call(
# dest=dest, target=ctor_fn, args=[operand]))`. That constructor's own C body
# ALREADY Increfs the leaf it wraps (the same way any other constructor
# increfs a BORROWED RC argument it stores into a field - see cfg.py's
# attr_assign()) - `dest` is therefore a genuinely FRESH, already-owned Call
# result, exactly like any other Call's return value.
#
# Root cause (see lowering.py's _is_aliasing_expr, now fixed): every caller
# deciding "does this returned/assigned operand need ITS OWN extra Incref"
# based on whether the ORIGINAL, pre-coercion ast node "looks aliasing" (a
# bare Name/Attribute read) - `_is_aliasing_expr(node.value, ...)` - had no
# way to know the operand it was actually handed had since been REPLACED by
# _coerce_into_union's fresh Call result. Two distinct symptoms followed from
# the same blind spot:
#
#   (1) _stmt_Return's own aliasing-return Incref check (added for a
#   different, EARLIER field-read-return bug - see attribute_return_rc_test.
#   py) fired a SECOND, spurious Incref on top of the union constructor's own
#   Incref for `return <borrowed_name>` where the function's declared return
#   type is a union - an over-count (confirmed: `def maybe(b: Box) -> Box|
#   None: return b` left compiler.refcount(b) TWO higher after the call than
#   before, not one).
#
#   (2) an AnnAssign/Assign's own is_alias branch (cfg.py's assign()) ALSO
#   fired a spurious extra Incref on the coerced union temp - but that
#   branch, unlike the (correct) fresh-Call branch, never untrack_temp()'s
#   the source operand, so _flush_pending_temps' later cleanup of the still-
#   tracked union temp emits a spurious Decref right after, canceling the
#   spurious Incref back out. Whether this nets out to "looking correct" by
#   coincidence (two wrongs, e.g. the plain field-read case below) or
#   silently under-counts for real depends on exactly which of the two
#   spurious ops actually fires in a given context - confirmed via a real
#   generator-machinery repro that crashed outright (see PLAN_GENERATORS.md's
#   own note on type_resolver.py's _maybe_route_yield_through_temp, which
#   works around both this and (1) above without fixing either).
#
# The fix tags _coerce_into_union's own returned operand (dest.
# is_union_coerce_result = True) and has _is_aliasing_expr check that flag
# before falling back to its ORIGINAL ast-node-shape guess - the exact same
# "the operand actually produced disagrees with what the ast node alone
# would suggest" pattern _is_aliasing_expr already used for a bound-method
# closure reference (`worker.run`, ast.Attribute but NOT aliasing - see its
# own ClosureType check).

import unittest

import test_support
from test_support import RealCompileMixin

# bug (1): return b where b is an ordinary BORROWED parameter and the
# function's own declared return type is a union - confirmed via direct
# compile-and-run that this used to leave compiler.refcount(b) inflated by
# TWO after the call (the union constructor's own correct Incref, PLUS
# _stmt_Return's spurious extra one), not one.
_RETURN_BORROWED_THROUGH_UNION = '''
class Box:
	v: i32
	def __init__( self, v: i32 ) -> None:
		self.v = v

def maybe( b: Box ) -> Box|None:
	return b

def main() -> i32:
	with compiler.wrap_arithmetic:
		b: Box = Box( v = 42 )
		before: usize = compiler.refcount( b )
		r = maybe( b )
		after: usize = compiler.refcount( b )
		if after != before + 1:
			return 1
		match r:
			case Box( got ):
				if got.v != 42:
					return 2
			case None:
				return 3
	return 0
'''

# bug (2): coercing a FIELD READ (an untracked expression - no epilogue
# entry, no already-tracked binding of its own) directly into a union-typed
# local. Confirmed via generated-C inspection that the pre-fix code emitted
# a spurious Incref immediately canceled by a spurious Decref of the same
# intermediate union temp right after - this standalone shape happened to
# still net to the correct total (the union constructor's own Incref
# survives either way), but the fix removes both spurious ops entirely
# rather than relying on them continuing to cancel by coincidence (see the
# del-and-still-valid check below, which the pre-fix double-op would NOT
# have survived in every context - the generator machinery's own real crash
# confirms that).
_UNION_COERCE_FIELD_READ = '''
class Box:
	v: i32
	def __init__( self, v: i32 ) -> None:
		self.v = v

class Holder:
	x: Box
	def __init__( self, x: Box ) -> None:
		self.x = x
	def get( self ) -> Box|None:
		wrapped: Box|None = self.x
		return wrapped

def main() -> i32:
	with compiler.wrap_arithmetic:
		b: Box = Box( v = 42 )
		h: Holder = Holder( b )
		before: usize = compiler.refcount( b )
		r = h.get()
		after: usize = compiler.refcount( b )
		if after != before + 1:
			return 1
		del h
		match r:
			case Box( got ):
				if got.v != 42:
					return 2
			case None:
				return 3
	return 0
'''

# same coercion shape, but the source is an already-tracked LOCAL (not a
# field read) - this was the one case the pre-existing generator workaround
# (_maybe_route_yield_through_temp) confirmed already correctly Increfed, so
# the fix must not regress it into a double-Incref.
_UNION_COERCE_TRACKED_LOCAL = '''
class Box:
	v: i32
	def __init__( self, v: i32 ) -> None:
		self.v = v

def wrap( b: Box ) -> Box|None:
	local: Box = b
	wrapped: Box|None = local
	return wrapped

def main() -> i32:
	with compiler.wrap_arithmetic:
		b: Box = Box( v = 7 )
		before: usize = compiler.refcount( b )
		r = wrap( b )
		after: usize = compiler.refcount( b )
		if after != before + 1:
			return 1
		match r:
			case Box( got ):
				if got.v != 7:
					return 2
			case None:
				return 3
	return 0
'''

# bug (4): a tuple LITERAL passed directly as a generic call's own argument
# (Result.Ok((a, b)), not merely a return value/AnnAssign RHS already fixed
# by 98c2010 "tuple: coerce elements into their declared union types during
# construction") against a declared Result[tuple[T1|None,T2|None],E] return
# type used to fail generic inference outright: "type parameter 'T' is
# inferred as both tuple[str|None,str|None] and tuple[str,str]". Root cause:
# monomorphize.py's substitute_type_params eagerly resolves a TupleType bound
# to a TypeVar into its backing RCClass before handing it down the call's own
# argument-lowering as an expected-type hint (needed so the emitter, which
# has no ensure_resolved of its own, never sees a bare unresolved TupleType -
# see that function's own comment) - but _expr_Tuple's own per-element
# union-coercion only recognized a BARE TupleType, not its already-resolved
# backing RCClass, so it silently skipped coercing the tuple's own elements
# into their declared union types, inferring the tuple's plain NATURAL type
# instead - which then disagreed with the return type's own binding.
# _expr_Tuple now falls back to TupleStorage's own reverse lookup (backing
# RCClass -> the TupleType it backs) to recover the original elem_types
# (with their union members) in that case too.
_RESULT_OK_TUPLE_LITERAL_UNION_ELEMENTS = '''
@union
class SomeErr:
	Bad: None

def f( a: str, b: str|None ) -> Result[tuple[str|None,str|None], SomeErr]:
	return Result.Ok(( a, b ))

def main() -> i32:
	r = f( 'hello', None )
	if r.is_err():
		return 1
	t: tuple[str|None,str|None] = r.unwrap( 'f: expected Ok' )
	match t[0]:
		case str( s ):
			if s != 'hello':
				return 2
		case None:
			return 3
	match t[1]:
		case str( s ):
			return 4
		case None:
			pass
	return 0
'''

_UNARY_OP_INTO_UNION = '''
def maybe_neg( i: i32 ) -> i32|None:
	with compiler.wrap_arithmetic:
		return -i

def maybe_invert( i: i32 ) -> i32|None:
	with compiler.wrap_arithmetic:
		return ~i

def maybe_not( flag: bool ) -> bool|None:
	return not flag

def yield_neg( i: i32 ) -> Iterator[Result[i32, StopIteration]]:
	with compiler.wrap_arithmetic:
		yield -i

def main() -> i32:
	a = maybe_neg( 5 )
	if a is None or a != -5:
		return 1
	b = maybe_invert( 5 )
	if b is None or b != -6:
		return 2
	c = maybe_not( True )
	if c is None or c != False:
		return 3
	d = maybe_not( False )
	if d is None or d != True:
		return 4
	g = yield_neg( 7 )
	r = g.__next__()
	match r:
		case Result.Err( _ ):
			return 5
		case Result.Ok( e ):
			if e != -7:
				return 5
	return 0
'''


@unittest.skipUnless( test_support.HAS_CC, 'no C compiler (clang/gcc/msvc) found - skipping real-compile RC tests' )
class UnionCoercionRCTests( RealCompileMixin, unittest.TestCase ):
	def test_union_coercion_increfs_exactly_once( self ) -> None:
		self.assert_programs_run([
			( 'return_borrowed_through_union', _RETURN_BORROWED_THROUGH_UNION ),
			( 'union_coerce_field_read', _UNION_COERCE_FIELD_READ ),
			( 'union_coerce_tracked_local', _UNION_COERCE_TRACKED_LOCAL ),
		])

	def test_tuple_literal_union_elements_via_generic_call( self ) -> None:
		self.assert_programs_run([
			( 'result_ok_tuple_literal_union_elements', _RESULT_OK_TUPLE_LITERAL_UNION_ELEMENTS ),
		])

	def test_unary_op_coerces_into_union_correctly( self ) -> None:
		# the scalar (non-RC) analog of this file's own bug class: a UnaryOp
		# (-x/~x/not x) whose OUTER context expects a union return/yield type
		# (e.g. `return -i` from a function declared -> i32|None) used to hint
		# node.operand's own lowering with that union type DIRECTLY, silently
		# wrapping the operand into Some(i) BEFORE the operator ever ran, and
		# (the `not` case specifically) forcing the RESULT temp itself to be
		# union-typed too - both produced invalid C (assigning a bare
		# int/bool straight into a union struct), confirmed via a real
		# compile failure, not just reasoning. Found while rebuilding
		# generator Phase F (`yield -i` hit the identical bug) - see
		# _expr_UnaryOp's own operand_hint/dest comments for the fix.
		self.assert_programs_run([
			( 'unary_op_into_union', _UNARY_OP_INTO_UNION ),
		])


if __name__ == '__main__':
	unittest.main()
