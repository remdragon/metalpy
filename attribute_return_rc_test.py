# Real-compile-and-run regression tests for a silent RC under-count: `return
# self.<field>` (or any other attribute-read expression, including a nested
# `self.a.b` chain) handed back the field's own bare pointer with NO Incref,
# even though the field itself keeps its own reference afterward. The caller
# then holds what LOOKS like an independent reference but isn't - once the
# original holder (or the field itself) is torn down, the "returned" value is
# a dangling pointer.
#
# Root cause (see lowering.py's _stmt_Return and cfg.py's has_live_entry()):
# _stmt_Return's existing "skip this binding's own decref, ownership just
# moved to the caller" identity-match logic (current_epilogue_label()/
# return_()) only ever matches a LOCAL binding that has its own live epilogue
# entry (an OWNED/COPY local, or a copy[T]/move[T] parameter). An attribute
# read (`self.x`) lowers to a fresh ir.GetAttr temp that was never pushed onto
# the epilogue stack in the first place (struct/union fields are never
# separately RC-tracked - see cfg.py's field_value() docstring), so the
# identity match trivially fails and _stmt_Return fell through treating it as
# an already-owned handoff - a bare pointer copy, no Incref, no Decref. A
# BORROWED parameter/`self` returned directly (`return self`) has exactly the
# same shape (never pushed either) and was affected identically.
#
# The workaround that was already known to correctly incref - assigning the
# field into a fresh local first (`local: Box = self.x; return local`) - goes
# through cfg.assign()'s own is_alias branch instead, entirely independent of
# _stmt_Return, which is why it was unaffected. One of the tests below
# confirms the fix doesn't turn that already-correct path into a double-incref.

import unittest

import test_support
from test_support import RealCompileMixin

_ATTR_RETURN_REFCOUNT_DELTA = '''
class Box:
	v: i32
	def __init__( self, v: i32 ) -> None:
		self.v = v

class Holder:
	x: Box
	def __init__( self, x: Box ) -> None:
		self.x = x
	def get( self ) -> Box:
		return self.x

def main() -> i32:
	with compiler.wrap_arithmetic:
		b: Box = Box( v = 42 )
		h: Holder = Holder( b )
		before: usize = compiler.refcount( b )
		got: Box = h.get()
		after: usize = compiler.refcount( b )
		# h.x still holds its own reference AND got is now a second,
		# independent one - the bug reported `after == before` (no Incref at
		# all)
		if after != before + 1:
			return 1
		if got.v != 42:
			return 2
	return 0
'''

# behavioral counterpart to the refcount-delta test above: drop the ORIGINAL
# holder (`del h`) right after extracting `got`, then read through `got`. If
# `got` were sharing h.x's own reference count (no separate Incref), h's own
# teardown would free the Box out from under `got` - reading got.v afterward
# is a genuine use-after-free, not just a miscounted number.
_ATTR_RETURN_SURVIVES_HOLDER_DROP = '''
class Box:
	v: i32
	def __init__( self, v: i32 ) -> None:
		self.v = v

class Holder:
	x: Box
	def __init__( self, x: Box ) -> None:
		self.x = x
	def get( self ) -> Box:
		return self.x

def main() -> i32:
	with compiler.wrap_arithmetic:
		h: Holder = Holder( Box( v = 99 ) )
		got: Box = h.get()
		del h
		if got.v != 99:
			return 1
	return 0
'''

# nested field-of-a-field: `self.middle.inner` returned directly from a
# method on Outer. Only the LEAF (Inner) needs its own extra Incref here -
# the intermediate `self.middle` GetAttr is a transient receiver used only to
# reach `.inner`, never itself stored/returned, so it correctly gets none.
_NESTED_ATTR_RETURN_REFCOUNT_DELTA = '''
class Inner:
	v: i32
	def __init__( self, v: i32 ) -> None:
		self.v = v

class Middle:
	inner: Inner
	def __init__( self, inner: Inner ) -> None:
		self.inner = inner

class Outer:
	middle: Middle
	def __init__( self, middle: Middle ) -> None:
		self.middle = middle
	def get_inner( self ) -> Inner:
		return self.middle.inner

def main() -> i32:
	with compiler.wrap_arithmetic:
		i: Inner = Inner( v = 7 )
		m: Middle = Middle( i )
		o: Outer = Outer( m )
		before: usize = compiler.refcount( i )
		got: Inner = o.get_inner()
		after: usize = compiler.refcount( i )
		if after != before + 1:
			return 1
		if got.v != 7:
			return 2
	return 0
'''

# `return self` (a BORROWED receiver, never pushed onto the epilogue stack -
# same shape as an attribute read) needs its own Incref too, exactly like
# `self.x` does.
_BORROWED_SELF_RETURN_REFCOUNT_DELTA = '''
class Widget:
	v: i32
	def __init__( self, v: i32 ) -> None:
		self.v = v
	def identity( self ) -> Widget:
		return self

def main() -> i32:
	with compiler.wrap_arithmetic:
		w: Widget = Widget( v = 5 )
		before: usize = compiler.refcount( w )
		got: Widget = w.identity()
		after: usize = compiler.refcount( w )
		if after != before + 1:
			return 1
		if got.v != 5:
			return 2
	return 0
'''

# regression guard: the pre-existing workaround (assign the field into a
# fresh local first, THEN return the local) already increfs correctly via
# cfg.assign()'s own is_alias branch - `local` is an OWNED binding with a
# live epilogue entry, so _stmt_Return's new incref check must recognize the
# identity match (has_live_entry) and skip its own Incref here, or this would
# now double-count instead of fixing the bare-attribute-return case.
_LOCAL_WORKAROUND_NOT_DOUBLE_INCREFED = '''
class Box:
	v: i32
	def __init__( self, v: i32 ) -> None:
		self.v = v

class Holder:
	x: Box
	def __init__( self, x: Box ) -> None:
		self.x = x
	def get_via_local( self ) -> Box:
		local: Box = self.x
		return local

def main() -> i32:
	with compiler.wrap_arithmetic:
		b: Box = Box( v = 42 )
		h: Holder = Holder( b )
		before: usize = compiler.refcount( b )
		got: Box = h.get_via_local()
		after: usize = compiler.refcount( b )
		if after != before + 1:
			return 1
		if got.v != 42:
			return 2
	return 0
'''


@unittest.skipUnless( test_support.HAS_CC, 'no C compiler (clang/gcc/msvc) found - skipping real-compile RC tests' )
class AttributeReturnRCTests( RealCompileMixin, unittest.TestCase ):
	def test_attribute_return_increfs( self ) -> None:
		self.assert_programs_run([
			( 'attr_return_refcount_delta', _ATTR_RETURN_REFCOUNT_DELTA ),
			( 'attr_return_survives_holder_drop', _ATTR_RETURN_SURVIVES_HOLDER_DROP ),
			( 'nested_attr_return_refcount_delta', _NESTED_ATTR_RETURN_REFCOUNT_DELTA ),
			( 'borrowed_self_return_refcount_delta', _BORROWED_SELF_RETURN_REFCOUNT_DELTA ),
			( 'local_workaround_not_double_increfed', _LOCAL_WORKAROUND_NOT_DOUBLE_INCREFED ),
		])


if __name__ == '__main__':
	unittest.main()
