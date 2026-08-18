# stdlib imports:
from pathlib import Path
import unittest

# local imports:
import ir
from discovery import Discovery
from compiler import Compiler
from mpy_types import Module, Variable, RCClass, Specialization

class CompilerTestCase( unittest.TestCase ):
	# compiler.run() force-enqueues windows/_console.py's own console-codepage
	# global on every Windows target, and sys.exit() whenever no_crt (see
	# Compiler.force_reachable's own comment) - real but incidental to what
	# these tests are actually checking, and absent entirely on non-Windows
	# targets (or on CRT-linked Windows targets, for sys.exit specifically),
	# so every helper below that turns compiler.functions/.extern_libs into a
	# comparable value filters them back out first, keeping assertions
	# host-OS-independent
	_CONSOLE_INIT_QUALNAMES = frozenset({
		'windows._console._init_console', 'windows.kernel32.SetConsoleOutputCP',
		'sys.exit', 'windows.kernel32.ExitProcess',
	})

	def setUp( self ) -> None:
		self.discovery = Discovery( import_builtins = False )
		self.compiler = Compiler( self.discovery )

	def _run( self, code: str ) -> None:
		self.compiler.import_code( code, Path( '__main__.py' ), scope = None )
		self.compiler.run()

	def _function_names( self ) -> list[str]:
		return [ f.function.qualname for f in self.compiler.functions if f.function.qualname not in self._CONSOLE_INIT_QUALNAMES ]

	def _extern_libs( self ) -> dict[str, set[str]]:
		libs = { lib: set( syms ) for lib, syms in self.compiler.extern_libs.items() }
		if 'kernel32' in libs:
			libs['kernel32'].discard( 'SetConsoleOutputCP' )
			libs['kernel32'].discard( 'ExitProcess' )
			if not libs['kernel32']:
				del libs['kernel32']
		return libs

	def _instructions_for( self, qualname: str ) -> list[ir.Instruction]:
		for f in self.compiler.functions:
			if f.function.qualname == qualname:
				return f.instructions
		self.fail( f'{qualname!r} was never lowered' )

class ArchitectureExampleTests( CompilerTestCase ):
	''' hand-verifies ARCHITECTURE.md's own shape: FuncStart, DeclareTemp, AddWrap, Call, DeleteTemp, Return, FuncEnd '''

	def test_foo_sequence_and_schedule_order( self ) -> None:
		# x + 1 dispatches through i32.__wrapped_add__ now - the literal `1`
		# isn't already a Variable, so it's spliced into a synthesized local
		# (an extra Assign) before AddWrap, same as lowering_test.py's own
		# migrated arithmetic tests
		self.discovery.import_name( 'builtins' )
		self._run( '''
def main() -> None:
	foo( 3 )

def foo( x: i32 ) -> None:
	with compiler.wrap_arithmetic:
		echo( x + 1 )
	return

def echo( x: i32 ) -> None:
	pass
''' )
		self.assertEqual( self._function_names(), [ 'main', '__main__.foo', '__main__.echo' ] )

		instructions = self._instructions_for( '__main__.foo' )
		kinds = [ type( instr ) for instr in instructions ]
		self.assertEqual( kinds, [ ir.FuncStart, ir.Assign, ir.DeclareTemp, ir.AddWrap, ir.Call, ir.DeleteTemp, ir.Return, ir.FuncEnd ] )

		add_wrap = instructions[3]
		self.assertIsInstance( add_wrap, ir.AddWrap )
		self.assertEqual( add_wrap.right, instructions[1].dest ) # the spliced $inline0$other local
		self.assertEqual( instructions[1].src, ir.Const( type = add_wrap.left.type, value = 1 ))

		call = instructions[4]
		self.assertIsInstance( call, ir.Call )
		self.assertEqual( call.target.qualname, '__main__.echo' )
		self.assertIsNone( call.dest )
		self.assertIsNone( call.receiver )
		self.assertEqual( call.args, [ add_wrap.dest ] )

class DeadCodeTests( CompilerTestCase ):
	def test_uncalled_function_is_excluded( self ) -> None:
		self._run( '''
def main() -> None:
	pass

def dead() -> None:
	pass
''' )
		self.assertEqual( self._function_names(), [ 'main' ] )

class MutualRecursionTests( CompilerTestCase ):
	def test_terminates_and_schedules_each_once( self ) -> None:
		self._run( '''
def main() -> None:
	a()

def a() -> None:
	b()

def b() -> None:
	a()
''' )
		names = self._function_names()
		self.assertEqual( sorted( names ), sorted([ 'main', '__main__.a', '__main__.b' ]))
		self.assertEqual( len( names ), len( set( names )))

class ClassDependencyTests( CompilerTestCase ):
	def test_class_referenced_only_via_local_annotation_is_scheduled( self ) -> None:
		self._run( '''
class Base:
	pass

class Foo( Base ):
	pass

def main() -> None:
	f: Foo
''' )
		rcclass_names = [ cls.qualname for cls in self.compiler.rcclasses ]
		self.assertIn( '__main__.Foo', rcclass_names )
		self.assertIn( '__main__.Base', rcclass_names )

	def test_class_referenced_via_parameter_type_is_scheduled( self ) -> None:
		self._run( '''
class Foo:
	x: i32
	def bump( self ) -> None:
		self.x = self.x

def use( f: Foo ) -> None:
	f.bump()

def main() -> None:
	f: Foo
	use( f )
''' )
		self.assertIn( '__main__.use', self._function_names() )
		self.assertIn( '__main__.Foo.bump', self._function_names() )
		self.assertIn( '__main__.Foo', [ cls.qualname for cls in self.compiler.rcclasses ] )

		bump_instructions = self._instructions_for( '__main__.Foo.bump' )
		kinds = [ type( instr ) for instr in bump_instructions ]
		# self.x = self.x -> GetAttr( t0, self, 'x' ), SetAttr( self, 'x', t0 ), DeleteTemp( t0 )
		self.assertIn( ir.GetAttr, kinds )
		self.assertIn( ir.SetAttr, kinds )

		use_instructions = self._instructions_for( '__main__.use' )
		call = next( i for i in use_instructions if isinstance( i, ir.Call ) )
		self.assertEqual( call.target.qualname, '__main__.Foo.bump' )
		self.assertIsNotNone( call.receiver )

	def test_class_reached_only_as_an_intermediate_attribute_chain_link_is_scheduled( self ) -> None:
		# Inner never appears as its own parameter/return/local-variable
		# annotation anywhere - the only way to learn it exists at all is by
		# resolving Outer.inner's field type while chasing o.inner.value.
		# _attr_lookup used to resolve() the owner type (so the lookup itself
		# succeeds) without ever scheduling it, so Inner would silently never
		# make it into compiler.rcclasses even though get_value depends on it
		self._run( '''
class Inner:
	value: i32

class Outer:
	inner: Inner

def get_value( o: Outer ) -> i32:
	return o.inner.value

def main() -> None:
	o: Outer
	x: i32 = get_value( o )
''' )
		rcclass_names = [ cls.qualname for cls in self.compiler.rcclasses ]
		self.assertIn( '__main__.Outer', rcclass_names )
		self.assertIn( '__main__.Inner', rcclass_names )

	def test_class_reached_only_via_a_staticmethod_call_is_scheduled( self ) -> None:
		# same gap, different path: Foo is never used as a parameter/return/
		# local-variable type anywhere - the only way to reach it is resolving
		# it while walking the Foo.make() namespace path in _try_resolve_namespace
		self._run( '''
class Foo:
	@staticmethod
	def make() -> i32:
		return 1

def main() -> None:
	x: i32 = Foo.make()
''' )
		self.assertIn( '__main__.Foo', [ cls.qualname for cls in self.compiler.rcclasses ] )

class FunctionSchedulingViaEnsureResolvedTests( CompilerTestCase ):
	''' _lower_call and _emit_is_err_check both schedule their call target
	through _ensure_resolved rather than a separate explicit self.schedule(...)
	call - these confirm the call target still actually ends up lowered by a
	full Compiler.run(), not just resolved for the immediate lookup '''

	def test_plain_call_target_is_scheduled( self ) -> None:
		self._run( '''
def main() -> None:
	helper()

def helper() -> None:
	pass
''' )
		self.assertEqual( self._function_names(), [ 'main', '__main__.helper' ] )

	def test_errdefer_is_err_call_target_is_scheduled( self ) -> None:
		# _build_is_err_check resolves+schedules Result.is_err purely as a
		# side effect of building the epilogue - is_err is never called
		# from user source at all, so it's the one call site here
		# self.schedule() can't be reached via the ordinary _lower_call
		# path. is_err's genericity is inherited from Result's own class
		# type params (like Result.Ok/.Err) - the receiver's type already
		# pins down the concrete args, so what actually gets scheduled is
		# the MONOMORPHIZED copy (Result.is_err[...]), not the bare,
		# never-emitted abstract Result.is_err (which has no real C struct
		# body anywhere - only concrete specializations do)
		self._run( '''
class bool: pass
class OverflowError: pass

@cstruct
class Result[T,E]:
	def is_err( self ) -> bool:
		pass
	@staticmethod
	def Err( e: E ) -> Result[T,E]:
		pass

def checked( e: OverflowError ) -> Result[None,OverflowError]:
	with errdefer:
		pass
	return Result.Err( e )

def main() -> None:
	e: OverflowError
	checked( e )
''' )
		names = self._function_names()
		self.assertIn( '__main__.checked', names )
		self.assertTrue( any( 'Result.is_err[' in name for name in names ), names )

class EnqueueFilteringTests( CompilerTestCase ):
	''' lowering.py's _ensure_resolved hands _enqueue literally anything it
	comes across, unconditionally - these are the direct, white-box checks
	that _enqueue itself is the one drawing the line: real compile units get
	queued, a Specialization decomposes into its schedulable pieces, and
	everything else is silently dropped rather than corrupting compiler.
	globals or crashing on drain '''

	def test_module_is_silently_ignored( self ) -> None:
		fake_module = Module( stem = 'fake', qualname = 'fake', file = None, line = None, intrinsics = {}, builtins = None )
		self.compiler._enqueue( fake_module )
		self.assertTrue( self.compiler.queue.empty())

	def test_none_is_silently_ignored( self ) -> None:
		self.compiler._enqueue( None )
		self.assertTrue( self.compiler.queue.empty())

	def test_non_global_variable_is_silently_ignored( self ) -> None:
		field = Variable( stem = 'x', qualname = 'Foo.x', file = None, line = None, is_global = False )
		self.compiler._enqueue( field )
		self.assertTrue( self.compiler.queue.empty())

	def test_global_variable_is_enqueued_and_lowered( self ) -> None:
		self._run( '''
X: i32 = 1

def main() -> None:
	pass
''' )
		x = self.discovery.modules['__main__'].get_local( 'X' )
		self.assertTrue( x.is_global )
		self.compiler._enqueue( x )
		self.compiler.run()
		self.assertIn( 'X', [ g.variable.stem for g in self.compiler.globals ] )

	def test_global_read_only_by_bare_name_is_still_lowered( self ) -> None:
		# _expr_Name used to just look the Variable up and hand it back
		# without ever calling _ensure_resolved on it - a global reached only
		# by plain reference (never via an annotation/attribute chain
		# elsewhere) would silently never make it into compiler.globals
		self._run( '''
X: i32 = 1

def main() -> None:
	y: i32 = X
''' )
		self.assertIn( 'X', [ g.variable.stem for g in self.compiler.globals ] )

	def test_classlike_specialization_is_queued_directly_plus_its_args( self ) -> None:
		# a concrete generic CLASS specialization (Result[Ok,Err]) is a real
		# compile unit in its own right now (see Lowering.monomorphize_class/
		# compiler.py's own _lower dispatch) - queued directly, same as a
		# Function-based Specialization already was, NOT decomposed away.
		# Its own concrete args are ALSO independently enqueued (unlike the
		# Function case) since a class specialization's substituted field
		# types have no "body" of their own to walk for that - see _enqueue's
		# own comment
		base = RCClass( stem = 'Result', qualname = '__main__.Result', file = None, line = None, type_params = [] )
		arg1 = RCClass( stem = 'Ok', qualname = '__main__.Ok', file = None, line = None )
		arg2 = RCClass( stem = 'Err', qualname = '__main__.Err', file = None, line = None )
		spec = Specialization( stem = 'Result[Ok,Err]', qualname = '__main__.Result[Ok,Err]', file = None, line = None, base = base, args = [ arg1, arg2 ] )
		self.compiler._enqueue( spec )
		queued = []
		while not self.compiler.queue.empty():
			queued.append( self.compiler.queue.get_nowait())
		self.assertEqual( len( queued ), 3 )
		self.assertTrue( any( q is spec for q in queued ))
		self.assertTrue( any( q is arg1 for q in queued ))
		self.assertTrue( any( q is arg2 for q in queued ))

class OverloadCallSiteTests( CompilerTestCase ):
	def test_len_style_plain_overload_group_dispatches_by_argument_type( self ) -> None:
		# mirrors the real lib/builtins/__init__.py len() shape: a free
		# function with no @overload anywhere, one plain implementation per
		# concrete type, dispatched purely by the call's own argument type
		self._run( '''
class usize: pass

class Foo:
	def __len__( self ) -> usize:
		return 1

class Bar:
	def __len__( self ) -> usize:
		return 2

def len( x: Foo ) -> usize:
	return x.__len__()

def len( x: Bar ) -> usize:
	return x.__len__()

def main( f: Foo, b: Bar ) -> None:
	x: usize = len( f )
	y: usize = len( b )
''' )
		self.assertEqual( self.discovery.errors.errors, [] )
		# each len(...) call site picks a *distinct* implementation (both
		# share the same qualname - they're only distinguishable by object
		# identity/their own parameter type) - confirm both actually got
		# scheduled+lowered, and that each one's own body calls the right
		# receiver's __len__, not the other one's
		len_fns = [ f for f in self.compiler.functions if f.function.stem == 'len' ]
		self.assertEqual( len( len_fns ), 2 )
		receiver_classes = sorted( lf.function.parameters[0].type.stem for lf in len_fns )
		self.assertEqual( receiver_classes, [ 'Bar', 'Foo' ])
		for lf in len_fns:
			expected_receiver_cls = lf.function.parameters[0].type.stem
			call = next( i for i in lf.instructions if isinstance( i, ir.Call ))
			self.assertEqual( call.target.qualname, f'__main__.{expected_receiver_cls}.__len__' )

	def test_unconditional_target_schedules_only_that_target( self ) -> None:
		self._run( '''
class int: pass
class str: pass

@overload
def foo( x: int ) -> None:
	...

def foo( x: int|None = None ) -> None:
	pass

def foo( x: str ) -> None:
	pass

def main( x: int ) -> None:
	foo( x )
''' )
		names = self._function_names()
		self.assertIn( 'main', names )
		self.assertIn( '__main__.foo', names )
		# 3, not 2: exactly one of the two plain implementations was
		# scheduled (never the whole group and never the stub - stubs have
		# no body to lower), PLUS the union's own synthesized 'int' member
		# constructor - x's own plain `int` type doesn't match the winning
		# implementation's real declared parameter type (int|None, a union),
		# so it must be coerced into it first (see lowering_test.py's
		# test_overload_call_resolves_to_unconditional_target for the exact
		# IR shape this produces). Before this fix, that coercion was
		# skipped entirely for this exact case (a non-literal argument whose
		# plain type is a LEAF of an overloaded call's winning target's own
		# union-typed parameter) - confirmed via a real compile of the
		# equivalent real-builtins shape, which produced a genuine "passing
		# 'int32_t' to parameter of incompatible type 'struct $__u$$...'" C
		# mismatch
		self.assertEqual( len( names ), 3 )

	def test_multi_branch_dispatch_resolves_via_runtime_tag_check( self ) -> None:
		# a union-typed argument (x: int|str) makes foo(x) ambiguous at
		# compile time - resolve_call returns real ConditionalDispatch
		# branches, which now lower to a runtime tag check (ir.Cmp against
		# the synthesized anonymous union's tag) rather than failing
		self.compiler.import_code( '''
class int: pass
class str: pass

@overload
def foo( x: int ) -> None:
	...

@overload
def foo( x: str ) -> None:
	...

def foo( x: int ) -> None:
	pass

def foo( x: str ) -> None:
	pass

def main( x: int|str ) -> None:
	foo( x )
''', Path( '__main__.py' ), scope = None )
		self.compiler.run()
		self.assertEqual( self.discovery.errors.errors, [] )
		instructions = self._instructions_for( 'main' )
		kinds = [ type( i ).__name__ for i in instructions ]
		self.assertIn( 'Cmp', kinds )
		self.assertEqual( kinds.count( 'Call' ), 2 ) # one per possible target (the branch + the default) - only one runs at runtime

	def test_generic_function_call_schedules_only_the_specialization( self ) -> None:
		# alloc[u32](...) must compile exactly one function - the
		# monomorphized alloc[u32] - never the shared, unspecialized alloc
		# itself (T never gets bound there, so it can't actually compile)
		self.discovery.import_name( 'builtins' )
		self._run( '''
def alloc[T]( count: usize ) -> usize:
	with compiler.wrap_arithmetic:
		return count * compiler.sizeof( T )

def main() -> None:
	x: usize = alloc[u32]( 10 )
''' )
		self.assertEqual( self.discovery.errors.errors, [] )
		names = self._function_names()
		self.assertEqual( set( names ), { 'main', '__main__.alloc[intrinsics.u32]' } )

class LocalVariableTests( CompilerTestCase ):
	def test_annassign_then_reassign( self ) -> None:
		self._run( '''
def main() -> None:
	x: i32 = 1
	x = 2
''' )
		instructions = self._instructions_for( 'main' )
		assigns = [ i for i in instructions if isinstance( i, ir.Assign ) ]
		self.assertEqual( len( assigns ), 2 )
		self.assertEqual( assigns[0].dest.stem, 'x' )
		self.assertIs( assigns[0].dest, assigns[1].dest )
		self.assertEqual( assigns[0].src, ir.Const( type = assigns[0].dest.type, value = 1 ))
		self.assertEqual( assigns[1].src, ir.Const( type = assigns[0].dest.type, value = 2 ))

class ErrorRecoveryTests( CompilerTestCase ):
	def test_broken_statement_does_not_block_sibling_statements_or_functions( self ) -> None:
		# 'broken's one statement fails (a bare name expression isn't
		# supported) - lower_function's own per-statement recovery boundary
		# means 'broken' still gets lowered (just missing that statement -
		# FuncStart/[fall-off Return]/FuncEnd, since its -> None body has
		# nothing left that ends in an explicit `return`) rather than being
		# dropped entirely, and 'fine' - unrelated - isn't affected at all
		self._run( '''
def broken() -> None:
	undefined_name

def fine() -> None:
	pass

def main() -> None:
	broken()
	fine()
''' )
		names = self._function_names()
		self.assertIn( 'main', names )
		self.assertIn( '__main__.fine', names )
		self.assertIn( '__main__.broken', names )
		self.assertEqual( [ type( i ).__name__ for i in self._instructions_for( '__main__.broken' ) ], [ 'FuncStart', 'Return', 'FuncEnd' ] )
		self.assertTrue( self.discovery.errors.errors )

class ExternLibraryDependencyTests( CompilerTestCase ):
	def test_called_extern_functions_register_their_library( self ) -> None:
		self._run( '''
@extern( 'c', 'malloc' )
def malloc( size: usize ) -> Ptr[u8]:
	...

@extern( 'c', 'free' )
def free( ptr: Ptr[u8] ) -> None:
	...

@extern( 'kernel32', 'HeapAlloc' )
def HeapAlloc( h: usize, flags: u32, size: usize ) -> Ptr[u8]:
	...

def main() -> None:
	p = malloc( 4 )
	free( p )
''' )
		# HeapAlloc is declared but never called - never scheduled/lowered,
		# so it never registers, matching how any other unused Function is
		# quietly dropped by _enqueue's filtering
		self.assertEqual( self._extern_libs(), { 'c': { 'malloc', 'free' } } )

	def test_no_extern_calls_leaves_the_registry_empty( self ) -> None:
		self._run( '''
def main() -> None:
	pass
''' )
		self.assertEqual( self._extern_libs(), {} )

class ExternDllDependencyTests( CompilerTestCase ):
	''' compiler.extern_dlls - populated only from @extern(..., dll=...)
	declarations on functions actually reached/lowered, the same
	reachability gate ExternLibraryDependencyTests above verifies for
	extern_libs (see compiler.py's Function-lowering branch: both are
	registered together, from the same `if unit.extern_lib is not None:`
	check). Drives mpy.py's post-link DLL-bundling step. '''

	def test_called_extern_function_registers_its_dll( self ) -> None:
		self._run( '''
@extern( 'tcl86t', 'Tcl_CreateInterp', dll = 'tcl86t.dll' )
def Tcl_CreateInterp() -> Ptr[None]:
	...

def main() -> None:
	Tcl_CreateInterp()
''' )
		self.assertEqual( self.compiler.extern_dlls, { 'tcl86t.dll' } )

	def test_declared_but_uncalled_extern_function_does_not_register_its_dll( self ) -> None:
		self._run( '''
@extern( 'tcl86t', 'Tcl_CreateInterp', dll = 'tcl86t.dll' )
def Tcl_CreateInterp() -> Ptr[None]:
	...

def main() -> None:
	pass
''' )
		self.assertEqual( self.compiler.extern_dlls, set() )

	def test_extern_without_dll_leaves_the_registry_empty( self ) -> None:
		self._run( '''
@extern( 'c', 'malloc' )
def malloc( size: usize ) -> Ptr[u8]:
	...

def main() -> None:
	malloc( 4 )
''' )
		self.assertEqual( self.compiler.extern_dlls, set() )

	def test_dll_list_registers_every_entry( self ) -> None:
		self._run( '''
@extern( 'tcl86t', 'Tcl_CreateInterp', dll = [ 'tcl86t.dll', 'zlib1.dll' ] )
def Tcl_CreateInterp() -> Ptr[None]:
	...

def main() -> None:
	Tcl_CreateInterp()
''' )
		self.assertEqual( self.compiler.extern_dlls, { 'tcl86t.dll', 'zlib1.dll' } )

	def test_dlls_union_across_multiple_reached_functions( self ) -> None:
		self._run( '''
@extern( 'tcl86t', 'Tcl_CreateInterp', dll = 'tcl86t.dll' )
def Tcl_CreateInterp() -> Ptr[None]:
	...

@extern( 'tk86t', 'Tk_Init', dll = 'tk86t.dll' )
def Tk_Init( interp: Ptr[None] ) -> i32:
	...

def main() -> None:
	Tcl_CreateInterp()
	Tk_Init( None )
''' )
		self.assertEqual( self.compiler.extern_dlls, { 'tcl86t.dll', 'tk86t.dll' } )

class ExternNoticeDependencyTests( CompilerTestCase ):
	''' compiler.extern_notices - populated only from
	@extern(..., notice=...) declarations on functions actually
	reached/lowered, same reachability gate and same registration point as
	extern_dlls above. Drives mpy.py's post-link THIRD-PARTY-LICENSES
	combination step. Deliberately independent of extern_dlls (see
	mpy_types.Function.extern_notices's own comment) - covered explicitly
	below, not assumed. '''

	def test_called_extern_function_registers_its_notice( self ) -> None:
		self._run( '''
@extern( 'tcl86t', 'Tcl_CreateInterp', notice = 'TCL' )
def Tcl_CreateInterp() -> Ptr[None]:
	...

def main() -> None:
	Tcl_CreateInterp()
''' )
		self.assertEqual( self.compiler.extern_notices, { 'TCL' } )

	def test_declared_but_uncalled_extern_function_does_not_register_its_notice( self ) -> None:
		self._run( '''
@extern( 'tcl86t', 'Tcl_CreateInterp', notice = 'TCL' )
def Tcl_CreateInterp() -> Ptr[None]:
	...

def main() -> None:
	pass
''' )
		self.assertEqual( self.compiler.extern_notices, set() )

	def test_notice_list_registers_every_entry( self ) -> None:
		self._run( '''
@extern( 'tcl86t', 'Tcl_CreateInterp', notice = [ 'TCL', 'ZLIB' ] )
def Tcl_CreateInterp() -> Ptr[None]:
	...

def main() -> None:
	Tcl_CreateInterp()
''' )
		self.assertEqual( self.compiler.extern_notices, { 'TCL', 'ZLIB' } )

	def test_notice_independent_of_dll( self ) -> None:
		''' a notice can be declared with no dll= at all (e.g. a header-only
		or statically-linked dependency that still needs attribution), and
		a dll= with no notice= (the author's call) - the two registries
		never imply each other. '''
		self._run( '''
@extern( 'tcl86t', 'Tcl_CreateInterp', dll = 'tcl86t.dll' )
def Tcl_CreateInterp() -> Ptr[None]:
	...

@extern( 'tk86t', 'Tk_Init', notice = 'TCL' )
def Tk_Init( interp: Ptr[None] ) -> i32:
	...

def main() -> None:
	Tcl_CreateInterp()
	Tk_Init( None )
''' )
		self.assertEqual( self.compiler.extern_dlls, { 'tcl86t.dll' } )
		self.assertEqual( self.compiler.extern_notices, { 'TCL' } )

	def test_notices_union_across_multiple_reached_functions( self ) -> None:
		self._run( '''
@extern( 'tcl86t', 'Tcl_CreateInterp', notice = 'TCL' )
def Tcl_CreateInterp() -> Ptr[None]:
	...

@extern( 'tcl86t', 'Tcl_Eval', notice = [ 'TCL', 'ZLIB' ] )
def Tcl_Eval( interp: Ptr[None], script: ConstPtr[u8] ) -> i32:
	...

def main() -> None:
	Tcl_CreateInterp()
	Tcl_Eval( None, None )
''' )
		self.assertEqual( self.compiler.extern_notices, { 'TCL', 'ZLIB' } )

if __name__ == '__main__':
	unittest.main()
