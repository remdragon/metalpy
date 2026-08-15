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


@unittest.skipUnless( test_support.HAS_CC, 'no C compiler (clang/gcc/msvc) found - skipping real-compile RC tests' )
class UnionCoercionRCTests( RealCompileMixin, unittest.TestCase ):
	def test_union_coercion_increfs_exactly_once( self ) -> None:
		self.assert_programs_run([
			( 'return_borrowed_through_union', _RETURN_BORROWED_THROUGH_UNION ),
			( 'union_coerce_field_read', _UNION_COERCE_FIELD_READ ),
			( 'union_coerce_tracked_local', _UNION_COERCE_TRACKED_LOCAL ),
		])


if __name__ == '__main__':
	unittest.main()
