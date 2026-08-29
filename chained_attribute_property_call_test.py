# Real-compile-and-run regression tests for a chained-property-into-method-
# call compile failure: `named_local.some_property.some_method(...)` in one
# expression failed with a misleading "<Type> has no attribute '<property>'",
# even though `some_property` is a real @property and the split-out form
# (`x = named_local.some_property; x.some_method(...)`) always worked.
#
# Root cause (lowering.py's _static_type_of_value_expr): the indirect-call/
# closure-call probes (_try_lower_indirect_call/_try_lower_closure_call) use
# this function to get a receiver's static type WITHOUT emitting/evaluating
# anything, so they can decide "is this even a field-typed callable" before
# committing to lowering node.func for real. Its own ast.Attribute branch
# routed through _attr_lookup, which is fatal (calls discovery.fail()) for
# anything that isn't a plain stored field - a @property getter is a
# Function, not a Variable, so probing a chain ending in one aborted the
# whole compile instead of gracefully declining (None) the way it already
# does for e.g. a Call-shaped receiver (Box().prop.method() worked fine,
# since _static_type_of_value_expr can't even start walking a Call node).
# Fixed by using the existing non-failing _find_field probe instead.
#
# The trigger has nothing to do with or_return()/Result/if-else branching
# (the a downstream project repro that surfaced this, pathlib.Path's `.parent.mkdir()`,
# happened to have all three) - it reproduces with a plain single-branch
# function and a directly-constructed local. Tests below cover both the
# minimal shape and the original multi-branch/or_return shape, to make sure
# the fix isn't accidentally narrow.

import unittest

import test_support
from test_support import RealCompileMixin

_MINIMAL_PROPERTY_CHAIN_INTO_CALL = '''
class Inner:
	def go( self ) -> i32:
		return 42

class Box:
	@property
	def inner( self ) -> Inner:
		return Inner()

def main() -> i32:
	b: Box = Box()
	if b.inner.go() != 42:
		return 1
	return 0
'''

# construction-expression receiver (no named local at all) - confirmed this
# shape already worked even before the fix, kept as a not-a-regression check.
_PROPERTY_CHAIN_ON_CONSTRUCTOR_RESULT = '''
class Inner:
	def go( self ) -> i32:
		return 42

class Box:
	@property
	def inner( self ) -> Inner:
		return Inner()

def main() -> i32:
	if Box().inner.go() != 42:
		return 1
	return 0
'''

# the original a downstream project-shaped trigger: an if/else-branching function
# returning Result[T,E], unwrapped via .or_return() in a DIFFERENT function,
# then the chained property-into-call.
_OR_RETURN_MULTI_BRANCH_PROPERTY_CHAIN = '''
class Inner:
	def go( self ) -> i32:
		return 42

class Box:
	@property
	def inner( self ) -> Inner:
		return Inner()

def make_box( flag: bool ) -> Result[Box, str]:
	if flag:
		return Result.Ok( Box() )
	else:
		return Result.Ok( Box() )

def use_box( flag: bool ) -> Result[i32, str]:
	b: Box = make_box( flag ).or_return()
	return Result.Ok( b.inner.go() )

def main() -> i32:
	match use_box( True ):
		case Result.Ok( v ):
			return 0 if v == 42 else 1
		case Result.Err( _ ):
			return 2
'''

# single-branch source function (no if/else) still unwrapped via or_return -
# confirms the if/else branching itself was never the trigger.
_OR_RETURN_SINGLE_BRANCH_PROPERTY_CHAIN = '''
class Inner:
	def go( self ) -> i32:
		return 42

class Box:
	@property
	def inner( self ) -> Inner:
		return Inner()

def make_box() -> Result[Box, str]:
	return Result.Ok( Box() )

def use_box() -> Result[i32, str]:
	b: Box = make_box().or_return()
	return Result.Ok( b.inner.go() )

def main() -> i32:
	match use_box():
		case Result.Ok( v ):
			return 0 if v == 42 else 1
		case Result.Err( _ ):
			return 2
'''

# split-into-two-statements form (the known-working workaround) must keep
# working unchanged.
_SPLIT_WORKAROUND_STILL_WORKS = '''
class Inner:
	def go( self ) -> i32:
		return 42

class Box:
	@property
	def inner( self ) -> Inner:
		return Inner()

def main() -> i32:
	b: Box = Box()
	i: Inner = b.inner
	if i.go() != 42:
		return 1
	return 0
'''


@unittest.skipUnless( test_support.HAS_CC, 'no C compiler (clang/gcc/msvc) found - skipping real-compile chained-property tests' )
class ChainedAttributePropertyCallTests( RealCompileMixin, unittest.TestCase ):
	def test_property_chained_into_method_call( self ) -> None:
		self.assert_programs_run([
			( 'minimal_property_chain_into_call', _MINIMAL_PROPERTY_CHAIN_INTO_CALL ),
			( 'property_chain_on_constructor_result', _PROPERTY_CHAIN_ON_CONSTRUCTOR_RESULT ),
			( 'or_return_multi_branch_property_chain', _OR_RETURN_MULTI_BRANCH_PROPERTY_CHAIN ),
			( 'or_return_single_branch_property_chain', _OR_RETURN_SINGLE_BRANCH_PROPERTY_CHAIN ),
			( 'split_workaround_still_works', _SPLIT_WORKAROUND_STILL_WORKS ),
		])


if __name__ == '__main__':
	unittest.main()
