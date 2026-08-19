# Real-compile-and-run behavioral test for Ptr[T]/ConstPtr[T] __str__/
# __repr__ (lib/builtins/__ptr_arith.py's _ptr_str[T]/_const_ptr_str[T]) -
# the natural numeric address in hex, most-significant digit first (e.g.
# "0x7ff6a1234560"), matching how a debugger or C's %p shows a pointer.
# Deliberately NOT binascii.hexlify() of the pointer's own raw storage bytes
# (considered and rejected - confirmed with the user): that would byte-dump
# the pointer's in-memory representation, giving REVERSED digit order on
# every little-endian target this compiler actually runs on. _ptr_addr
# reads the address as a real usize VALUE first (a native typed dereference
# through a reinterpret cast, not a raw memory copy), which is what keeps
# the result endian-correct.
#
# Two real, independent, pre-existing gaps were found and fixed while
# building this:
#
# 1. lowering.py's _lower_method_call (the f-string dunder-dispatch path)
#    found a Ptr[T]-registered generic dunder (_find_method alone) but never
#    bound its own type param T from the receiver - every OTHER caller of a
#    Ptr[T] dunder goes through operator dispatch (_find_dunder_for_arg),
#    which already does this binding (_resolve_receiver_generic_dunder);
#    __str__/__repr__ are the first Ptr[T] dunders ever reached through the
#    plain-method-name f-string path, since f32/f64/int's own __str__/
#    __repr__ (the only prior users of this path) are never generic.
#
# 2. type_resolver.py's _attr_lookup_callable had a `dot-operator on a raw
#    pointer means arrow` rule (`p.method()` dereferences and looks up
#    `method` on p's own POINTEE type, matching C's `->`) with no exception
#    for a dunder Ptr[T]/ConstPtr[T] registers on ITSELF - so `p.__str__()`
#    (and, transitively, `str(p)`, since builtins.str.__call__'s own body is
#    `return x.__str__()`) used to find the POINTEE's own __str__ instead
#    (e.g. u8's, for Ptr[u8]) and pass the raw pointer where a plain scalar
#    value was expected - a real type-confusion compile error. Fixed by
#    checking Ptr[T]/ConstPtr[T]'s own dunders first, falling through to the
#    arrow redirect only when the dunder isn't found there - matching how
#    Python itself always prioritizes a protocol dunder over attribute
#    lookup through the object's own contents.
#
# A separate, unrelated, pre-existing bug was found (not fixed here, out of
# scope) and spawned as its own follow-up: calling an ORDINARY (non-dunder)
# method through a Ptr[SomeCStruct] passes the raw pointer instead of a
# dereferenced value, failing to compile - confirmed present identically on
# the commit before this file's own work started.

import unittest

import emitter_c
import test_support
from test_support import RealCompileMixin


@unittest.skipUnless( test_support.HAS_CC, 'no C compiler (clang/gcc/msvc) found - skipping real-compile Ptr str/repr tests' )
class PtrStrBehaviorTests( RealCompileMixin, unittest.TestCase ):
	def test_bare_fstring_interpolation( self ) -> None:
		compiler = self._compile_source( '''
import sys
def main() -> i32:
	p: Ptr[u8] = sys.alloc[u8]( 4 )
	s: str = f'{p}'
	sys.free( p )
	if s._byte_slice( usize(0), usize(2) ) != '0x':
		return 1
	if s.byte_len() <= usize(2): # at least one hex digit after "0x"
		return 2
	return 0
''' )
		self._assert_compiles_and_runs( emitter_c.emit_c( compiler ), expected_exit = 0, compiler = compiler )

	def test_direct_str_and_repr_calls( self ) -> None:
		compiler = self._compile_source( '''
import sys
def main() -> i32:
	p: Ptr[u8] = sys.alloc[u8]( 4 )
	s: str = p.__str__()
	r: str = p.__repr__()
	sys.free( p )
	if s != r:
		return 1
	if s._byte_slice( usize(0), usize(2) ) != '0x':
		return 2
	return 0
''' )
		self._assert_compiles_and_runs( emitter_c.emit_c( compiler ), expected_exit = 0, compiler = compiler )

	def test_str_builtin_call( self ) -> None:
		compiler = self._compile_source( '''
import sys
def main() -> i32:
	p: Ptr[u8] = sys.alloc[u8]( 4 )
	s: str = str( p )
	sys.free( p )
	if s._byte_slice( usize(0), usize(2) ) != '0x':
		return 1
	return 0
''' )
		self._assert_compiles_and_runs( emitter_c.emit_c( compiler ), expected_exit = 0, compiler = compiler )

	def test_const_ptr_str_and_repr( self ) -> None:
		compiler = self._compile_source( '''
import sys
def main() -> i32:
	p: Ptr[u8] = sys.alloc[u8]( 4 )
	cp: ConstPtr[u8] = p
	s: str = cp.__str__()
	r: str = cp.__repr__()
	fs: str = f'{cp}'
	sys.free( p )
	if s != r or s != fs:
		return 1
	if s._byte_slice( usize(0), usize(2) ) != '0x':
		return 2
	return 0
''' )
		self._assert_compiles_and_runs( emitter_c.emit_c( compiler ), expected_exit = 0, compiler = compiler )

	def test_hex_digits_match_the_real_address_value( self ) -> None:
		# a real end-to-end round trip: two DIFFERENT allocations must
		# produce two DIFFERENT hex strings, and the same allocation must
		# always produce the SAME hex string, twice
		compiler = self._compile_source( '''
import sys
def main() -> i32:
	a: Ptr[u8] = sys.alloc[u8]( 4 )
	b: Ptr[u8] = sys.alloc[u8]( 4 )
	sa1: str = f'{a}'
	sa2: str = f'{a}'
	sb: str = f'{b}'
	sys.free( a )
	sys.free( b )
	if sa1 != sa2:
		return 1
	if sa1 == sb:
		return 2
	return 0
''' )
		self._assert_compiles_and_runs( emitter_c.emit_c( compiler ), expected_exit = 0, compiler = compiler )


if __name__ == '__main__':
	unittest.main()
