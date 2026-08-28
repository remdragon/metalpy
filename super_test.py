# super().<method>(...) - real compile+link+run coverage, following
# builtins_iteration_test.py's own template (RealCompileMixin +
# assert_programs_run, print()-free, exit code 0 = every check passed,
# distinct nonzero i32 per failing check).

import unittest

import test_support
from test_support import RealCompileMixin

_ORDINARY_VIRTUAL_METHOD_CHAIN = '''
import compiler
from atomic import Atomic

_order: Atomic[i32] = Atomic[i32]( 0 )
_base_ran_at: Atomic[i32] = Atomic[i32]( -1 )
_derived_ran_at: Atomic[i32] = Atomic[i32]( -1 )

class Base:
	@virtual
	def greet( self ) -> i32:
		_base_ran_at.store( _order.fetch_add( 1 ) )
		return 1

class Derived( Base ):
	@virtual
	def greet( self ) -> i32:
		_derived_ran_at.store( _order.fetch_add( 1 ) )
		# super().greet() must reach Base.greet directly - NOT re-dispatch
		# back into Derived.greet through the vtable (infinite recursion)
		with compiler.wrap_arithmetic:
			return super().greet() + 10

def main() -> i32:
	d: Derived = Derived()
	result: i32 = d.greet()
	if result != 11:
		return 1
	if _derived_ran_at.load() != 0:
		return 2
	if _base_ran_at.load() != 1:
		return 3
	return 0
'''

_DEL_CHAIN_WITH_REAL_RC_FIELDS = '''
from atomic import Atomic

_resource_live: Atomic[i32] = Atomic[i32]( 0 )
_base_del_ran: Atomic[i32] = Atomic[i32]( 0 )
_derived_del_ran: Atomic[i32] = Atomic[i32]( 0 )

class Resource:
	def __init__( self ) -> None:
		_resource_live.fetch_add( 1 )
	@virtual
	def __del__( self ) -> None:
		_resource_live.fetch_add( -1 )

class Base:
	base_res: Resource
	def __init__( self ) -> None:
		self.base_res = Resource()
	@virtual
	def __del__( self ) -> None:
		# must run exactly once - never double-counted whether reached via
		# super().__del__() or the ordinary top-level destructor dispatch
		_base_del_ran.fetch_add( 1 )

class Derived( Base ):
	extra_res: Resource
	def __init__( self ) -> None:
		super().__init__()
		self.extra_res = Resource()
	@virtual
	def __del__( self ) -> None:
		_derived_del_ran.fetch_add( 1 )
		super().__del__()

def make_and_drop() -> None:
	d: Derived = Derived()
	return

def main() -> i32:
	make_and_drop()
	if _base_del_ran.load() != 1:
		return 1
	if _derived_del_ran.load() != 1:
		return 2
	# both Resource fields (base_res, extra_res) must be released exactly
	# once each - the field cascade is automatic/independent of __del__, but
	# a broken super() chain (e.g. double dispatch) could still double-free
	# one of them or otherwise corrupt this count
	if _resource_live.load() != 0:
		return 3
	return 0
'''

_INHERITED_DEL_RUNS_WITHOUT_OVERRIDE = '''
from atomic import Atomic

_base_del_ran: Atomic[i32] = Atomic[i32]( 0 )

class Base:
	@virtual
	def __del__( self ) -> None:
		_base_del_ran.fetch_add( 1 )

class Derived( Base ):
	y: i32
	def __init__( self, y: i32 ) -> None:
		self.y = y

def make_and_drop() -> None:
	d: Derived = Derived( 1 )
	return

def main() -> i32:
	make_and_drop()
	# regression: a subclass with no __del__ of its own must still run its
	# base's __del__ automatically - no super() call needed or possible here
	if _base_del_ran.load() != 1:
		return 1
	return 0
'''


@unittest.skipUnless( test_support.HAS_CC, 'no C compiler (clang/gcc/msvc) found - skipping real-compile super() tests' )
class SuperCallTests( RealCompileMixin, unittest.TestCase ):
	def test_super_call_on_ordinary_virtual_method( self ) -> None:
		self.assert_programs_run([
			( 'ordinary_virtual_method_chain', _ORDINARY_VIRTUAL_METHOD_CHAIN ),
		])

	def test_super_del_chains_and_frees_exactly_once( self ) -> None:
		self.assert_programs_run([
			( 'del_chain_with_real_rc_fields', _DEL_CHAIN_WITH_REAL_RC_FIELDS ),
		])

	def test_inherited_del_runs_without_subclass_override( self ) -> None:
		self.assert_programs_run([
			( 'inherited_del_runs_without_override', _INHERITED_DEL_RUNS_WITHOUT_OVERRIDE ),
		])


if __name__ == '__main__':
	unittest.main()
