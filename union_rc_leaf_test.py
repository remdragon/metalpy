# Real-compile-and-run regression tests for a heap-corruption bug in
# compiler.incref(x)/compiler.decref(x) (lowering.py's _lower_compiler_incref/
# _lower_compiler_decref) and in UnsafeDict[K,V]'s own key/value ownership
# helpers (lib/builtins/__init__.py's _owned_value/_owned_key/_store_key/
# _store_value/_release_key/_release_value).
#
# Both used type_resolver._is_RC(t) - deliberately t.is_rc_pointer(), true
# only when t's OWN runtime representation IS a single bare RC pointer - to
# decide whether there was any RC work to do at all. That's the right
# question for choosing a storage LAYOUT (a bare handle vs a real heap-
# allocated struct copy - see UnsafeList[T]/UnsafeDict[K,V]'s own __init__/
# _read_element comments), but the WRONG question for "does this value need
# incref/decref treatment" - a TaggedUnion (a `@union class Foo:`) is never
# is_rc_pointer() (its runtime shape is a tag+data VALUE STRUCT, never a bare
# pointer - see mpy_types.TaggedUnion.is_rc_pointer's own docstring) even
# when one of its members carries a real RC leaf.
#
# Concretely: T monomorphized to a @union whose RC-carrying member is a
# plain RCClass (the builtin `int`, or an ordinary user-defined class) made
# compiler.incref(val)/compiler.decref(val) inside UnsafeList[T].append/
# __getitem__/etc silently no-op (the SAME no-op posture that's correctly a
# no-op for list[i32]), and made UnsafeDict[K,V]'s _owned_value/_store_value/
# _release_value skip RC bookkeeping for V entirely. Every element read back
# out of the container ended up under-retained, corrupting the heap on free
# (Windows STATUS_HEAP_CORRUPTION). This didn't show up with a str/list/dict
# leaf because those tests all happened to use IMMORTAL-refcount string
# literals (METALPY_IMMORTAL_REFCOUNT - see emitter_c.py's literal struct
# emission), whose retain/release are guarded no-ops regardless of count -
# masking the exact same under-count that corrupts a REAL heap allocation
# (int's own digit buffer, a plain class's own fields) on double-free.
#
# Fix: cfg.py grew a public decref() (mirroring its existing incref()), and
# _lower_compiler_incref/_lower_compiler_decref now gate on cfg.rc_leaves(t)
# (the general is_rc() question, TaggedUnion-aware) and emit through
# cfg.py's own union-aware incref()/decref() - the same tag-gated codegen
# every other RC site in this compiler already uses for a TaggedUnion -
# instead of a bare ir.Incref/ir.Decref (only ever correct for a genuine
# pointer). UnsafeDict[K,V]'s helpers now call compiler.incref/decref
# unconditionally in both branches, relying on the intrinsic's own (now
# correct) no-op-for-genuinely-non-RC-T posture, instead of gating the call
# itself on is_rc(V)/is_rc(K).

import unittest

import test_support
from test_support import RealCompileMixin

# list[T], T a @union whose RC leaf is the builtin int (arbitrary-precision,
# heap-managed - its own __del__ frees a separately-allocated digit buffer,
# which is what makes a double-release of it a real, deterministic heap
# corruption rather than a silent no-op like an immortal string literal)
_LIST_UNION_RC_LEAF_BUILTIN_INT = '''
@union
class Val:
	Nothing: None
	Number: int

def main() -> i32:
	with compiler.wrap_arithmetic:
		n: int = int.from_str( '42' ).unwrap( 'parse' )
		before: usize = compiler.refcount( n )
		items: list[Val] = list[Val]()
		items.append( Val.Number( n ))
		got: Val = items.__getitem__( 0 ).unwrap( 'get' )
		match got:
			case Val.Number( m ):
				if m.__str__() != '42':
					return 1
				del m
			case Val.Nothing( _ ):
				return 2
		del got
		del items
		after: usize = compiler.refcount( n )
		if after != before:
			return 3
	return 0
'''

# same shape, RC leaf is an ordinary user-defined class instead of a builtin
_LIST_UNION_RC_LEAF_USER_CLASS = '''
class Box:
	v: i32
	def __init__( self, v: i32 ) -> None:
		self.v = v

@union
class Val:
	Nothing: None
	Boxed: Box

def main() -> i32:
	with compiler.wrap_arithmetic:
		b: Box = Box( v = 99 )
		before: usize = compiler.refcount( b )
		items: list[Val] = list[Val]()
		items.append( Val.Boxed( b ))
		got: Val = items.__getitem__( 0 ).unwrap( 'get' )
		match got:
			case Val.Boxed( boxed ):
				if boxed.v != 99:
					return 1
				del boxed
			case Val.Nothing( _ ):
				return 2
		del got
		del items
		after: usize = compiler.refcount( b )
		if after != before:
			return 3
	return 0
'''

# dict[K,V], V a @union whose RC leaf is the builtin int - exercises
# UnsafeDict[K,V]'s own _store_value/_owned_value/_release_value, a
# different code path from list[T]'s append/__getitem__/UnsafeList.__del__
_DICT_UNION_RC_LEAF_BUILTIN_INT = '''
@union
class Val:
	Nothing: None
	Number: int

def main() -> i32:
	with compiler.wrap_arithmetic:
		n: int = int.from_str( '7' ).unwrap( 'parse' )
		before: usize = compiler.refcount( n )
		d: dict[str, Val] = dict[str, Val]()
		d.__setitem__( 'k', Val.Number( n ))
		got: Val = d.__getitem__( 'k' ).unwrap( 'get' )
		match got:
			case Val.Number( m ):
				if m.__str__() != '7':
					return 1
				del m
			case Val.Nothing( _ ):
				return 2
		del got
		del d
		after: usize = compiler.refcount( n )
		if after != before:
			return 3
	return 0
'''


@unittest.skipUnless( test_support.HAS_CC, 'no C compiler (clang/gcc/msvc) found - skipping real-compile RC tests' )
class UnionRCLeafContainerTests( RealCompileMixin, unittest.TestCase ):
	def test_plain_rcclass_union_leaf_round_trips_through_containers( self ) -> None:
		self.assert_programs_run([
			( 'list_union_rc_leaf_builtin_int', _LIST_UNION_RC_LEAF_BUILTIN_INT ),
			( 'list_union_rc_leaf_user_class', _LIST_UNION_RC_LEAF_USER_CLASS ),
			( 'dict_union_rc_leaf_builtin_int', _DICT_UNION_RC_LEAF_BUILTIN_INT ),
		])


if __name__ == '__main__':
	unittest.main()
