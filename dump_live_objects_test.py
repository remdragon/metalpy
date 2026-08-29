# Real-compile-and-run tests for debug-mode alloc-site tracking +
# sys.dump_live_objects() (emitter_c.py's ObjectHeader.alloc_loc/debug_link,
# __metalpy_ObjectVtbl.type_name, the __metalpy_debug_* intrusive-list
# PROLOGUE, and lib/sys.py's alloc[T]/free/dump_live_objects). Verifies real
# aggregation behavior (not just "compiles"): multiple objects from the SAME
# alloc_loc (including several from inside a loop) must collapse into one
# grouped line with the right count, a dropped reference must vanish from
# the dump, and a different CLASS must report as its own separate group.
#
# Per-call-site attribution: an ordinary `Foo(...)` allocates through Foo's
# own synthesized $$__new__/__init__ constructor, built ONCE per class and
# shared by every `Foo(...)` call site in the whole program - $$__new__'s
# own hidden __alloc_loc parameter (type_resolver.py's
# _synthesize_rcclass_constructor) is what lets alloc_loc still name the
# REAL textual call site rather than $$__new__'s own (useless - shared by
# every caller) location, baked in per call by lowering.py's
# _try_lower_construct_call.

from pathlib import Path
import subprocess
import tempfile
import unittest

import emitter_c
import linker_c
import test_support
from test_support import RealCompileMixin
from compiler import Compiler
from discovery import Discovery
from targets import detect

_DUMP_LIVE_OBJECTS_AGGREGATES_BY_SITE = '''
import compiler
import sys

class Foo:
	x: i32
	def __init__(self, x: i32) -> None:
		self.x = x

class Bar:
	y: i32
	def __init__(self, y: i32) -> None:
		self.y = y

def make_three() -> list[Foo]:
	result: list[Foo] = list[Foo]()
	i: i32 = 0
	with compiler.panic_arithmetic( 'test bound, cannot overflow' ):
		while i < 3:
			result.append( Foo( i )) # loop - ONE call site, 3 live instances, must still aggregate into ONE group (count=3)
			i += 1
	return result

def main() -> i32:
	a = Foo( 1 ) # a DIFFERENT call site than make_three's loop - own separate group now (see this file's own top comment)
	items = make_three()
	dropped = Foo( 2 ) # dropped before the dump, must not inflate the live count
	del dropped
	b = Bar( 1 ) # a DIFFERENT class - must report as its own separate group
	sys.dump_live_objects()
	# program exits immediately after - no need to clean up a/items/b for the
	# dump's own correctness, the OS reclaims everything on exit either way
	return 0
'''

_DUMP_LIVE_OBJECTS_IS_A_NOOP_IN_RELEASE = '''
import sys

class Foo:
	x: i32
	def __init__(self, x: i32) -> None:
		self.x = x

def main() -> i32:
	a = Foo( 1 )
	sys.dump_live_objects() # release mode: compiler.target.debug folds False, this is a no-op
	if a.x == 1:
		return 0
	return 1
'''


@unittest.skipUnless( test_support.HAS_CC, 'no C compiler (clang/gcc/msvc) found - skipping real-compile RC tests' )
class DumpLiveObjectsTests( RealCompileMixin, unittest.TestCase ):
	def test_dump_live_objects_aggregates_by_class_and_site( self ) -> None:
		# debug is the default active target (targets.detect()'s own
		# comment) - self._compile_source's Discovery() uses it unmodified
		compiler = self._compile_source( _DUMP_LIVE_OBJECTS_AGGREGATES_BY_SITE )
		import emitter_c
		c_source = emitter_c.emit_c( compiler )
		result = self._build_and_run( compiler, c_source, timeout = 10 )
		self.assertEqual( result.returncode, 0,
			f'program crashed (exit {result.returncode}):\nstdout: {result.stdout}\nstderr: {result.stderr}'
			f'{test_support.c_source_on_failure( c_source )}' )
		out = result.stdout.decode( 'utf-8', errors = 'replace' )

		# TWO __main__.Foo groups now (per-call-site attribution) - main's
		# own `a = Foo(1)` (count=1) and make_three's loop (count=3: one
		# call SITE, 3 runtime instances, proving same-site instances still
		# aggregate together rather than becoming 3 separate entries). The
		# already-dropped `dropped = Foo(2)` instance (main's own, a
		# DIFFERENT call site than `a`'s) is correctly excluded entirely -
		# a real leak there would show as a spurious THIRD Foo group.
		foo_lines = [ line for line in out.splitlines() if '__main__.Foo @' in line ]
		self.assertEqual( len( foo_lines ), 2, f'expected exactly 2 __main__.Foo groups (one per real call site), got:\n{out}' )
		counts = sorted( int( line.rsplit( 'count=', 1 )[1].split()[0] ) for line in foo_lines )
		self.assertEqual( counts, [ 1, 3 ], f'expected one group of 1 (main\'s own `a`) and one of 3 (make_three\'s loop):\n{out}' )

		# Bar is a DIFFERENT class - must be its own separate group, not
		# merged with Foo's
		bar_lines = [ line for line in out.splitlines() if '__main__.Bar @' in line ]
		self.assertEqual( len( bar_lines ), 1, f'expected exactly 1 __main__.Bar group, got:\n{out}' )
		self.assertIn( 'count=1', bar_lines[0] )

		# each __main__.Foo instance is the same size - a group's own byte
		# total must be an exact multiple of its own count (not pinning an
		# exact ObjectHeader layout size here), and every group must agree
		# on the SAME per-instance size (all Foo, regardless of call site)
		def _bytes_of( line: str ) -> int:
			return int( line.rsplit( 'bytes=', 1 )[1].strip() )
		def _count_of( line: str ) -> int:
			return int( line.rsplit( 'count=', 1 )[1].split()[0] )
		per_instance_sizes = { _bytes_of( line ) // _count_of( line ) for line in foo_lines }
		self.assertEqual( len( per_instance_sizes ), 1, f'expected every Foo group to agree on the same per-instance size:\n{out}' )
		self.assertGreater( next( iter( per_instance_sizes )), 0 )

		# a live raw sys.alloc buffer group is present too (list[Foo]'s own
		# backing storage, ... - see lib/sys.py's alloc[T])
		self.assertIn( 'live raw sys.alloc buffers', out )

	def test_dump_live_objects_is_a_noop_in_release( self ) -> None:
		# --release: compiler.target.debug folds False, so sys.dump_live_
		# objects()'s own body is dead-code-eliminated to nothing - this
		# must still compile and run cleanly (the whole point of gating it
		# behind an ordinary runtime `if`, not a compiler.target()-selected
		# overload - a bug there would only show up in a release build).
		# Needs an explicit release active_target - RealCompileMixin's own
		# _compile_source/assert_programs_run always build the debug
		# default (targets.detect()'s own comment), so this compiles by
		# hand instead, mirroring _compile_source's own steps.
		release_target = dict( detect() )
		release_target['debug'] = False
		discovery = Discovery( import_builtins = True, active_target = release_target )
		compiler = Compiler( discovery )
		compiler.import_code( _DUMP_LIVE_OBJECTS_IS_A_NOOP_IN_RELEASE, Path( '__main__.py' ), scope = None )
		compiler.run()
		self.assertEqual( discovery.errors.errors, [],
			'compile errors:\n' + '\n'.join( str( e ) for e in discovery.errors.errors ))
		c_source = emitter_c.emit_c( compiler )
		self.assertNotIn( '__metalpy_dump_live_objects', c_source )
		result = self._build_and_run( compiler, c_source, timeout = 10 )
		self.assertEqual( result.returncode, 0,
			f'exe exited {result.returncode}, expected 0 (stderr: {result.stderr})'
			f'{test_support.c_source_on_failure( c_source )}' )


# --- automatic debug-mode leak-check epilogue (emitter_c.py's own
# __metalpy_deinit()) - decref every global RC variable, then dump whatever's
# still tracked, all automatically at exit, no explicit sys.dump_live_
# objects() call needed anywhere in the program itself.

# survivor: never reassigned - a global-owned object still reachable at exit
# through its own declaring global must NOT appear in the leak report.
# reassigned: reassigned during main() (via a helper - a pre-existing,
# unrelated compiler crash reassigning a global directly inline inside
# main() itself, `AttributeError: 'NoneType' object has no attribute
# 'rc_leaves'` in cfg.py's rc_leaves() - is flagged separately, not fixed
# here) to a fresh Foo kept alive through exit - the false-positive case a
# simpler "just wipe the tracking registry" alternative would have gotten
# wrong (the OLD Foo(10) value is already auto-decref'd by the reassignment
# itself, well before the epilogue ever runs - ordinary `global x; x = ...`
# semantics, nothing epilogue-specific). Both globals are also read from
# (not just written) - an entirely unread global hits the SAME pre-existing
# crash (TypeResolver apparently never resolves a write-only global's type).
# leaked: a genuine leak, an extra manual incref() with no matching decref -
# metalpy's own ordinary scope-exit destruction already releases a plain
# local when it goes out of scope normally (confirmed: a plain, otherwise-
# untouched local does NOT show up here, unlike the OLDER, still-existing
# manual sys.dump_live_objects() test above, which calls the dump WHILE
# main() is still on the stack, before ITS OWN scope-exit cleanup has had a
# chance to run - this automatic epilogue only ever runs AFTER
# __metalpy_user_main() has fully returned, by which point every one of its
# own plain locals is already gone) - so the extra reference this holds
# alive has no real owner left anywhere and MUST appear in the report.
_AUTOMATIC_LEAK_CHECK_EPILOGUE = '''
import compiler

class Foo:
	x: i32
	def __init__(self, x: i32) -> None:
		self.x = x

survivor: Foo = Foo( 100 )
reassigned: Foo = Foo( 10 )

def swap() -> None:
	global reassigned
	reassigned = Foo( 20 )

def main() -> i32:
	leaked = Foo( 1 )
	compiler.incref( leaked ) # extra reference, never balanced by a decref - the genuine leak
	swap()
	with compiler.wrap_arithmetic:
		if survivor.x == 100 and reassigned.x == 20 and leaked.x == 1:
			return 0
		return 1
'''


@unittest.skipUnless( test_support.HAS_CC, 'no C compiler (clang/gcc/msvc) found - skipping real-compile RC tests' )
class AutomaticLeakCheckEpilogueTests( RealCompileMixin, unittest.TestCase ):
	def _compiled( self ) -> Compiler:
		discovery = Discovery( import_builtins = True ) # debug is the default active target
		compiler = Compiler( discovery )
		compiler.import_code( _AUTOMATIC_LEAK_CHECK_EPILOGUE, Path( '__main__.py' ), scope = None )
		compiler.run()
		self.assertEqual( discovery.errors.errors, [],
			'compile errors:\n' + '\n'.join( str( e ) for e in discovery.errors.errors ))
		return compiler

	def test_globals_excluded_reassignment_handled_leak_reported( self ) -> None:
		compiler = self._compiled()
		c_source = emitter_c.emit_c( compiler ) # leak_check defaults to True
		self.assertIn( '__metalpy_deinit', c_source )
		self.assertIn( '__metalpy_deinit();', c_source ) # actually called from main(), not just defined
		result = self._build_and_run( compiler, c_source, timeout = 10 )
		self.assertEqual( result.returncode, 0,
			f'program crashed (exit {result.returncode}):\nstdout: {result.stdout}\nstderr: {result.stderr}'
			f'{test_support.c_source_on_failure( c_source )}' )
		out = result.stdout.decode( 'utf-8', errors = 'replace' )
		foo_lines = [ line for line in out.splitlines() if '__main__.Foo @' in line ]
		# `survivor` (never reassigned) and `reassigned`'s own final value
		# (the fresh Foo(20), correctly decref'd by the epilogue itself just
		# like any other global) are both excluded - only `leaked` remains
		self.assertEqual( len( foo_lines ), 1, f'expected exactly 1 __main__.Foo group (only the genuine leak), got:\n{out}' )
		self.assertIn( 'count=1', foo_lines[0], f'globals must be excluded from the leak report:\n{out}' )

	def test_no_leak_check_flag_suppresses_the_epilogue_entirely( self ) -> None:
		compiler = self._compiled()
		c_source = emitter_c.emit_c( compiler, leak_check = False )
		self.assertNotIn( '__metalpy_deinit', c_source )
		result = self._build_and_run( compiler, c_source, timeout = 10 )
		self.assertEqual( result.returncode, 0,
			f'exe exited {result.returncode}, expected 0 (stderr: {result.stderr})'
			f'{test_support.c_source_on_failure( c_source )}' )
		out = result.stdout.decode( 'utf-8', errors = 'replace' )
		self.assertNotIn( '__main__.Foo @', out, 'no dump should have run at all' )

	def test_release_build_never_defines_the_epilogue( self ) -> None:
		# _target_debug gates __metalpy_deinit() the same way every other
		# debug-only PROLOGUE piece is gated - a release build must not even
		# define it, leak_check=True default notwithstanding
		release_target = dict( detect() )
		release_target['debug'] = False
		discovery = Discovery( import_builtins = True, active_target = release_target )
		compiler = Compiler( discovery )
		compiler.import_code( _AUTOMATIC_LEAK_CHECK_EPILOGUE, Path( '__main__.py' ), scope = None )
		compiler.run()
		self.assertEqual( discovery.errors.errors, [] )
		c_source = emitter_c.emit_c( compiler )
		self.assertNotIn( '__metalpy_deinit', c_source )
		result = self._build_and_run( compiler, c_source, timeout = 10 )
		self.assertEqual( result.returncode, 0,
			f'exe exited {result.returncode}, expected 0 (stderr: {result.stderr})'
			f'{test_support.c_source_on_failure( c_source )}' )

	def test_automatic_leak_check_epilogue_under_no_crt_windows_build( self ) -> None:
		# mirrors sys_argv_test.py's own test_argv_correct_under_freestanding_
		# no_crt_build - a freestanding build's own mainCRTStartup calls the
		# real (thin, synthesized) main() directly, which is what actually
		# runs __metalpy_init()/__metalpy_user_main()/__metalpy_deinit() -
		# confirms the whole epilogue (including the dump's own raw
		# WriteFile path, no CRT involved) works identically under no_crt.
		compiler = self._compiled()
		self.assertFalse( compiler.requires_crt )
		no_crt = 'c' not in compiler.extern_libs and not compiler.requires_crt
		if compiler.disco.active_target['os'] == 'windows':
			self.assertNotIn( 'c', compiler.extern_libs )
			self.assertTrue( no_crt )
		c_source = emitter_c.emit_c( compiler, no_crt = no_crt )
		self.assertIn( '__metalpy_deinit', c_source )

		with tempfile.TemporaryDirectory() as tmp:
			src_path = Path( tmp ) / 'generated.c'
			obj_path = Path( tmp ) / 'generated.o'
			exe_path = Path( tmp ) / 'test_exe.exe'
			src_path.write_text( c_source, encoding = 'utf-8' )

			cc = test_support._CC
			cc_result = cc.compile( src_path, obj_path, no_crt = no_crt )
			self.assertEqual( cc_result.returncode, 0,
				f'{cc.name} compile failed:\n{cc_result.stdout}{test_support.c_source_on_failure( c_source )}' )

			ldflags = ''
			for lib in sorted( compiler.extern_libs ):
				if lib == 'c':
					continue
				flag = linker_c.resolve_lib_ldflag( cc, lib, compiler.extern_libs[lib], no_crt = no_crt )
				ldflags = ldflags + f' {flag}' if ldflags else flag

			link_result = cc.link( exe_path, [ obj_path ], ldflags = ldflags, no_crt = no_crt )
			self.assertEqual( link_result.returncode, 0, f'{cc.name} link failed:\n{link_result.stdout}' )

			result = subprocess.run( [ str( exe_path ) ], capture_output = True, cwd = tmp )
			self.assertEqual( result.returncode, 0, f'exe exited {result.returncode}, expected 0 (stderr: {result.stderr})' )
			out = result.stdout.decode( 'utf-8', errors = 'replace' )
			foo_lines = [ line for line in out.splitlines() if '__main__.Foo @' in line ]
			self.assertEqual( len( foo_lines ), 1, f'expected exactly 1 __main__.Foo group, got:\n{out}' )
			self.assertIn( 'count=1', foo_lines[0], f'globals must be excluded from the leak report:\n{out}' )


# lowering.py's Lowering.run_global()/FunctionLowering._emit() - a module-
# level global's own initializer expression is lowered with fn=None (no
# enclosing function - run_global()'s own "no fn/self/construction of its
# own" comment), but DOES build a real CFGState up front, same as an
# ordinary function body. _emit()'s own fresh_temp() registration (needed
# for ANY intermediate RC temp - e.g. a fallible initializer's own raw
# Result[T,E], still holding its own internal reference after .unwrap()'s
# narrowed extraction - to ever get flushed/released) used to be gated on
# self._current_fn is not None instead of self._cfg is not None, a stale
# leftover from before run_global() built its own CFGState - silently
# skipped fresh_temp() registration for EVERY module-level global
# initializer in the program, leaking any such intermediate temp
# permanently (not just past one statement - for the whole process
# lifetime, since nothing ever flushed it). Real repro: `R_EOL: Foo =
# make_fallible().unwrap('msg')` at module scope.
_FALLIBLE_GLOBAL_INITIALIZER_LEAK = '''
import compiler
import sys

class Foo:
	x: i32
	def __init__(self, x: i32) -> None:
		self.x = x

class MyError:
	pass

def make_fallible() -> Result[Foo, MyError]:
	return Result.Ok( Foo( 7 ))

R: Foo = make_fallible().unwrap( 'should not fail' )

def main() -> i32:
	ok = R.x == 7
	sys.dump_live_objects() # BEFORE returning - R itself is still live, only the Result's own intermediate payload reference should be gone
	with compiler.wrap_arithmetic:
		return 0 if ok else 1
'''


@unittest.skipUnless( test_support.HAS_CC, 'no C compiler (clang/gcc/msvc) found - skipping real-compile RC tests' )
class FallibleGlobalInitializerLeakTests( RealCompileMixin, unittest.TestCase ):
	def test_fallible_initializer_leaves_no_intermediate_result_alive( self ) -> None:
		discovery = Discovery( import_builtins = True )
		compiler = Compiler( discovery )
		compiler.import_code( _FALLIBLE_GLOBAL_INITIALIZER_LEAK, Path( '__main__.py' ), scope = None )
		compiler.run()
		self.assertEqual( discovery.errors.errors, [],
			'compile errors:\n' + '\n'.join( str( e ) for e in discovery.errors.errors ))
		c_source = emitter_c.emit_c( compiler )
		result = self._build_and_run( compiler, c_source, timeout = 10 )
		self.assertEqual( result.returncode, 0,
			f'program crashed (exit {result.returncode}):\nstdout: {result.stdout}\nstderr: {result.stderr}'
			f'{test_support.c_source_on_failure( c_source )}' )
		out = result.stdout.decode( 'utf-8', errors = 'replace' )
		# a leaked intermediate Foo (the Result's own internal payload
		# reference, never released) would show up here too, count=2 -
		# only R's own single, final instance should survive to be reported
		foo_lines = [ line for line in out.splitlines() if '__main__.Foo @' in line ]
		self.assertEqual( len( foo_lines ), 1, f'expected exactly 1 __main__.Foo group (R itself), got:\n{out}' )
		self.assertIn( 'count=1', foo_lines[0], f'the Result[Foo,MyError] intermediate must not leak a second Foo reference:\n{out}' )


# lowering.py's _stmt_If - `if cond: x = Owned(...)` with NO explicit else,
# x a borrowed parameter reassigned to a fresh owned value only inside the
# if-body: merge_if()'s own ownership-disagreement reconciliation mints a
# cancel flag for x's new epilogue-release obligation, default-armed True
# at function entry, meant to be DISARMED (set False) only on the implicit
# "condition was false" path (x stays borrowed, nothing to release). The
# disarm Assign is correctly emitted right at else_label - but _stmt_If's
# own no-explicit-orelse branch never emitted a Jump to SKIP else_label
# from the end of the if-body's own code, so the if-body fell straight
# through into the disarm unconditionally, on BOTH paths - permanently
# disarming the flag even when the if-branch DID run and DID make x owned,
# leaking whatever it was reassigned to. Confirmed via a real repro:
# grap.mpy's own `if not filespecs: filespecs = ['*']`.
_BORROWED_PARAM_REASSIGNED_IN_IF_LEAK = '''
import sys

class Foo:
	x: i32
	def __init__(self, x: i32) -> None:
		self.x = x

def make() -> Foo:
	return Foo( 99 )

def helper( y: Foo ) -> i32:
	if y.x == 0:
		y = make()
	return y.x

def main() -> i32:
	y: Foo = Foo( 0 )
	result = helper( y )
	sys.dump_live_objects() # BEFORE returning - y (the caller's own borrowed original) is still live; make()'s own Foo must not also still be live
	with compiler.wrap_arithmetic:
		return 0 if result == 99 else 1
'''


@unittest.skipUnless( test_support.HAS_CC, 'no C compiler (clang/gcc/msvc) found - skipping real-compile RC tests' )
class BorrowedParamReassignedInIfLeakTests( RealCompileMixin, unittest.TestCase ):
	def test_reassigning_a_borrowed_param_inside_an_elseless_if_does_not_leak( self ) -> None:
		discovery = Discovery( import_builtins = True )
		compiler = Compiler( discovery )
		compiler.import_code( _BORROWED_PARAM_REASSIGNED_IN_IF_LEAK, Path( '__main__.py' ), scope = None )
		compiler.run()
		self.assertEqual( discovery.errors.errors, [],
			'compile errors:\n' + '\n'.join( str( e ) for e in discovery.errors.errors ))
		c_source = emitter_c.emit_c( compiler )
		result = self._build_and_run( compiler, c_source, timeout = 10 )
		self.assertEqual( result.returncode, 0,
			f'program crashed (exit {result.returncode}):\nstdout: {result.stdout}\nstderr: {result.stderr}'
			f'{test_support.c_source_on_failure( c_source )}' )
		out = result.stdout.decode( 'utf-8', errors = 'replace' )
		# only main()'s own `y` (Foo(0)) should survive to be reported -
		# helper()'s own local reassignment (y = make()) must be fully
		# released before helper() returns; a leak there would show a
		# SECOND __main__.Foo group (make()'s own instance, never released)
		foo_lines = [ line for line in out.splitlines() if '__main__.Foo @' in line ]
		self.assertEqual( len( foo_lines ), 1, f'expected exactly 1 __main__.Foo group (main\'s own y), got:\n{out}' )
		self.assertIn( 'count=1', foo_lines[0], f'helper()\'s own reassigned y (make()\'s Foo) must not leak:\n{out}' )


# lowering.py's _stmt_While - `while x := f():` re-executes the walrus's own
# fresh-declare lowering (no release of the PRIOR iteration's value) on every
# pass through the back edge, even though it's only a true first-time
# declaration on iteration 1. Confirmed via a real repro: grap.mpy's own
# `while root := paths.pop(''):`.
_WALRUS_IN_WHILE_CONDITION_LEAK = '''
import sys

class Foo:
	x: i32
	def __init__(self, x: i32) -> None:
		self.x = x

def make_two() -> list[Foo]:
	result: list[Foo] = list[Foo]()
	result.append( Foo( 1 ))
	result.append( Foo( 2 ))
	return result

def helper() -> i32:
	items = make_two()
	while item := items.pop( None ):
		pass
	return 0

def main() -> i32:
	helper()
	sys.dump_live_objects() # every Foo popped off items across BOTH iterations must be released, not just the final one
	return 0
'''


@unittest.skipUnless( test_support.HAS_CC, 'no C compiler (clang/gcc/msvc) found - skipping real-compile RC tests' )
class WalrusInWhileConditionLeakTests( RealCompileMixin, unittest.TestCase ):
	def test_walrus_reassigned_every_iteration_of_a_while_condition_does_not_leak( self ) -> None:
		discovery = Discovery( import_builtins = True )
		compiler = Compiler( discovery )
		compiler.import_code( _WALRUS_IN_WHILE_CONDITION_LEAK, Path( '__main__.py' ), scope = None )
		compiler.run()
		self.assertEqual( discovery.errors.errors, [],
			'compile errors:\n' + '\n'.join( str( e ) for e in discovery.errors.errors ))
		c_source = emitter_c.emit_c( compiler )
		result = self._build_and_run( compiler, c_source, timeout = 10 )
		self.assertEqual( result.returncode, 0,
			f'program crashed (exit {result.returncode}):\nstdout: {result.stdout}\nstderr: {result.stderr}'
			f'{test_support.c_source_on_failure( c_source )}' )
		out = result.stdout.decode( 'utf-8', errors = 'replace' )
		foo_lines = [ line for line in out.splitlines() if '__main__.Foo @' in line ]
		self.assertEqual( foo_lines, [], f'both walrus-rebound Foo instances must be released each iteration, got:\n{out}' )


# KNOWN GAP (see _stmt_While's own comment on fresh_rc_walrus_names): the
# release above only covers the loop's NORMAL (body-completed) back edge - an
# explicit `continue` inside the body jumps straight to start_label
# (continue_label == start_label for a while loop) and bypasses it entirely.
# Same shape as _WALRUS_IN_WHILE_CONDITION_LEAK but with the body replaced by
# a bare `continue` - expected to leak both walrus-rebound Foo instances until
# that gap is closed. Marked expectedFailure so this documents the gap
# without failing the suite; flip to a real assertion (and drop the
# decorator) once fixed.
_WALRUS_IN_WHILE_CONDITION_CONTINUE_LEAK = '''
import sys

class Foo:
	x: i32
	def __init__(self, x: i32) -> None:
		self.x = x

def make_two() -> list[Foo]:
	result: list[Foo] = list[Foo]()
	result.append( Foo( 1 ))
	result.append( Foo( 2 ))
	return result

def helper() -> i32:
	items = make_two()
	while item := items.pop( None ):
		continue
	return 0

def main() -> i32:
	helper()
	sys.dump_live_objects() # every Foo popped off items across BOTH iterations must be released, not just the final one
	return 0
'''


@unittest.skipUnless( test_support.HAS_CC, 'no C compiler (clang/gcc/msvc) found - skipping real-compile RC tests' )
class WalrusInWhileConditionContinueLeakTests( RealCompileMixin, unittest.TestCase ):
	@unittest.expectedFailure
	def test_continue_inside_a_while_walrus_loop_bypasses_the_per_iteration_release( self ) -> None:
		discovery = Discovery( import_builtins = True )
		compiler = Compiler( discovery )
		compiler.import_code( _WALRUS_IN_WHILE_CONDITION_CONTINUE_LEAK, Path( '__main__.py' ), scope = None )
		compiler.run()
		self.assertEqual( discovery.errors.errors, [],
			'compile errors:\n' + '\n'.join( str( e ) for e in discovery.errors.errors ))
		c_source = emitter_c.emit_c( compiler )
		result = self._build_and_run( compiler, c_source, timeout = 10 )
		self.assertEqual( result.returncode, 0,
			f'program crashed (exit {result.returncode}):\nstdout: {result.stdout}\nstderr: {result.stderr}'
			f'{test_support.c_source_on_failure( c_source )}' )
		out = result.stdout.decode( 'utf-8', errors = 'replace' )
		foo_lines = [ line for line in out.splitlines() if '__main__.Foo @' in line ]
		self.assertEqual( foo_lines, [], f'both walrus-rebound Foo instances must be released each iteration, got:\n{out}' )


# cfg.py's promote_borrowed_for_loop() - a for-loop that reassigns a borrowed
# parameter mints a ONE-TIME incref before the loop starts (promoting the
# parameter from BORROWED to OWNED for the rest of the function), but the
# pushed Epilogue entry was indistinguishable from an ordinary loop-body-local
# push: restore() (called once the loop body's retried lowering succeeds)
# truncated it away as if it were block-scoped, so the promoted incref was
# never balanced by a release at the function's own epilogue - a permanent
# +1 leak, present even when the loop's own iterable is empty (0 runtime
# iterations). Confirmed via a real repro: grap.mpy's own
# `for file in self._list_dir(root):` reassigning `path`.
_FOR_LOOP_PROMOTED_PARAM_LEAK = '''
import sys

class Foo:
	x: i32
	def __init__(self, x: i32) -> None:
		self.x = x

def make_empty() -> list[Foo]:
	return list[Foo]()

def helper( item: Foo ) -> i32:
	for other in make_empty(): # never runs - the promoted incref still fires unconditionally
		if other.x == 0:
			item = other
		else:
			item = other
	return item.x

def main() -> i32:
	f = Foo( 1 )
	helper( f )
	sys.dump_live_objects() # only main's own f may still be live - the promoted-but-never-released copy must not also show up
	return 0
'''


@unittest.skipUnless( test_support.HAS_CC, 'no C compiler (clang/gcc/msvc) found - skipping real-compile RC tests' )
class ForLoopPromotedParamLeakTests( RealCompileMixin, unittest.TestCase ):
	def test_promoting_a_borrowed_param_for_an_empty_for_loop_does_not_leak( self ) -> None:
		discovery = Discovery( import_builtins = True )
		compiler = Compiler( discovery )
		compiler.import_code( _FOR_LOOP_PROMOTED_PARAM_LEAK, Path( '__main__.py' ), scope = None )
		compiler.run()
		self.assertEqual( discovery.errors.errors, [],
			'compile errors:\n' + '\n'.join( str( e ) for e in discovery.errors.errors ))
		c_source = emitter_c.emit_c( compiler )
		result = self._build_and_run( compiler, c_source, timeout = 10 )
		self.assertEqual( result.returncode, 0,
			f'program crashed (exit {result.returncode}):\nstdout: {result.stdout}\nstderr: {result.stderr}'
			f'{test_support.c_source_on_failure( c_source )}' )
		out = result.stdout.decode( 'utf-8', errors = 'replace' )
		foo_lines = [ line for line in out.splitlines() if '__main__.Foo @' in line ]
		self.assertEqual( len( foo_lines ), 1, f'expected exactly 1 __main__.Foo group (main\'s own f), got:\n{out}' )
		self.assertIn( 'count=1', foo_lines[0], f'the promoted-but-unreleased copy of item must not leak:\n{out}' )


_INLINE_ALIASING_RETURN_USED_INLINE_LEAK = '''
def sink( s: str ) -> None:
	print( s )

def main() -> i32:
	n: i32 = 5
	s: str = f'attacked {n} times'
	sink( s.__str__() ) # __str__ is @inline `return self` - result used inline, never bound to a local
	return 0
'''

_LIST_REPR_STR_ELEMENT_LEAK = '''
def main() -> i32:
	n: i32 = 5
	s: str = f'attacked {n} times' # dynamic (non-literal) interpolation - a real heap str, unlike a literal
	hooks: list[str] = list[str]()
	hooks.append( s )
	print( f'{hooks}' ) # whole-list interpolation -> list.__repr__ -> str(val) per element
	return 0
'''


@unittest.skipUnless( test_support.HAS_CC, 'no C compiler (clang/gcc/msvc) found - skipping real-compile RC tests' )
class InlineAliasingReturnUsedInlineLeakTests( RealCompileMixin, unittest.TestCase ):
	''' an @inline function whose body is a single aliasing return (`return
	self`, e.g. str.__str__/str.__call__) needs its own extra Incref spliced
	in (lowering_stmt.py's _incref_aliasing_return) since the returned
	reference is still independently owned elsewhere. That Incref used to
	attach to the SAME pre-existing operand identity (self/a parameter),
	with nothing registering the extra unit of ownership for release unless
	the caller happened to bind the call's result to a fresh named local
	first (which independently tracks it) - used directly as an inline,
	unbound call argument, the extra reference had nothing left to release
	it at all. Fixed by materializing a genuinely fresh, cfg.fresh_temp()-
	registered temp for the wrapped result (see _incref_aliasing_return's
	own `wrap_fresh` parameter) so it participates in the ordinary pending-
	temp release machinery an un-inlined call's own result already gets. '''

	def _run_and_get_output( self, source: str ) -> str:
		discovery = Discovery( import_builtins = True )
		compiler = Compiler( discovery )
		compiler.import_code( source, Path( '__main__.py' ), scope = None )
		compiler.run()
		self.assertEqual( discovery.errors.errors, [],
			'compile errors:\n' + '\n'.join( str( e ) for e in discovery.errors.errors ))
		c_source = emitter_c.emit_c( compiler )
		result = self._build_and_run( compiler, c_source, timeout = 10 )
		self.assertEqual( result.returncode, 0,
			f'program crashed (exit {result.returncode}):\nstdout: {result.stdout}\nstderr: {result.stderr}'
			f'{test_support.c_source_on_failure( c_source )}' )
		return result.stdout.decode( 'utf-8', errors = 'replace' )

	def test_inline_dunder_result_used_as_a_bare_call_argument_does_not_leak( self ) -> None:
		out = self._run_and_get_output( _INLINE_ALIASING_RETURN_USED_INLINE_LEAK )
		self.assertNotIn( 'builtins.str @', out, f'leaked str from an inlined `return self` used inline:\n{out}' )

	def test_whole_list_fstring_interpolation_of_dynamic_str_elements_does_not_leak( self ) -> None:
		# the original repro: list.__repr__'s own `parts.append(str(val))`
		# hits the identical shape - str(val) rewrites to str.__call__(val),
		# @inline splices to `val.__str__()`, @inline splices again to
		# `return self` - the leaked reference used to only show up for a
		# DYNAMIC (non-literal) element: a string literal is a static const,
		# whose incref/decref are no-ops the leak checker never sees either
		# way, so this must use a real f-string-built str to actually exercise it
		out = self._run_and_get_output( _LIST_REPR_STR_ELEMENT_LEAK )
		self.assertIn( '[attacked 5 times]', out )
		self.assertNotIn( 'builtins.str @', out, f'leaked str element from whole-list f-string interpolation:\n{out}' )


if __name__ == '__main__':
	unittest.main()
