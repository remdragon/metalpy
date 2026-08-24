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
import unittest

import emitter_c
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
	compiler.decref( dropped )
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


if __name__ == '__main__':
	unittest.main()
