# Real-compile-and-run tests for debug-mode alloc-site tracking +
# sys.dump_live_objects() (emitter_c.py's ObjectHeader.alloc_loc/debug_link,
# __metalpy_ObjectVtbl.type_name, the __metalpy_debug_* intrusive-list
# PROLOGUE, and lib/sys.py's alloc[T]/free/dump_live_objects). Verifies real
# aggregation behavior (not just "compiles"): multiple objects from the SAME
# alloc_loc (including several from inside a loop) must collapse into one
# grouped line with the right count, a dropped reference must vanish from
# the dump, and a different CLASS must report as its own separate group.
#
# Known granularity limitation (see the plan's own report): ir.Allocate.loc
# is stamped from the ambient line the Lowering pass happens to be on when
# it emits that ONE Allocate instruction - for an ordinary `Foo(...)` this is
# inside Foo's own synthesized $$__new__/__init__ constructor, which is
# built ONCE per class and called from every `Foo(...)` site in the whole
# program. So alloc_loc ends up naming the CLASS's own constructor location,
# not each individual textual call expression - confirmed directly against
# the generated C (a single `$header.alloc_loc = "...";` line assigned from
# two separate call sites below). Real, useful info (still separates
# distinct classes and still counts+aggregates correctly), just coarser
# than "per call site" for user RCClass construction specifically.

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
			result.append( Foo( i )) # loop - three more live Foo instances, same alloc_loc as the one below (see this file's own top comment)
			i += 1
	return result

def main() -> i32:
	a = Foo( 1 ) # same alloc_loc as make_three's loop - both must aggregate into ONE Foo group
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

		# exactly ONE __main__.Foo group (both call sites share one alloc_loc
		# - see this file's own top comment) with count=4: the 1 top-level
		# instance + the 3 built inside make_three's loop - proves the loop
		# instances really do aggregate together (not 3 separate entries),
		# and that the already-dropped instance is correctly excluded (would
		# be count=5 otherwise)
		foo_lines = [ line for line in out.splitlines() if '__main__.Foo @' in line ]
		self.assertEqual( len( foo_lines ), 1, f'expected exactly 1 __main__.Foo group, got:\n{out}' )
		self.assertIn( 'count=4', foo_lines[0], f'expected count=4 (1 + the loop instances=3, dropped instance excluded):\n{out}' )

		# Bar is a DIFFERENT class - must be its own separate group, not
		# merged with Foo's
		bar_lines = [ line for line in out.splitlines() if '__main__.Bar @' in line ]
		self.assertEqual( len( bar_lines ), 1, f'expected exactly 1 __main__.Bar group, got:\n{out}' )
		self.assertIn( 'count=1', bar_lines[0] )

		# each __main__.Foo instance is the same size - byte total must be
		# an exact 4x multiple of a single instance's own size (not pinning
		# an exact ObjectHeader layout size here)
		def _bytes_of( line: str ) -> int:
			return int( line.rsplit( 'bytes=', 1 )[1].strip() )
		self.assertEqual( _bytes_of( foo_lines[0] ) % 4, 0 )
		self.assertGreater( _bytes_of( foo_lines[0] ), 0 )

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


if __name__ == '__main__':
	unittest.main()
