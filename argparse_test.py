# Real-compile-and-run tests for lib/argparse.py.
#
# parse_args() calls sys.exit() on --help and on a bad command line (real
# argparse's own top-level behavior) - those cases can't share the merged
# multi-case binary assert_programs_run() uses (sys.exit() would kill the
# whole merged process before later cases run), so they're each their own
# standalone compile+run, like termcolor_test.py's own
# test_unknown_color_panics.

import unittest

import test_support
from compiler import Compiler
from discovery import Discovery


class ArgparseTests( test_support.RealCompileMixin, unittest.TestCase ):
	def setUp( self ) -> None:
		self.discovery = Discovery( import_builtins = True )
		self.compiler = Compiler( self.discovery )

	@unittest.skipUnless( test_support.HAS_CC, 'no C compiler (clang/gcc/msvc) found - skipping' )
	def test_programs_compile_and_run( self ) -> None:
		import emitter_c
		self.assert_programs_run([
			( 'positional_and_optionals_parsed', '''
import argparse

def main() -> i32:
	p: argparse.ArgumentParser = argparse.ArgumentParser( prog = 'myprog' )
	p.add_argument( ['input'], help = 'input file' )
	p.add_argument( ['-v', '--verbose'], action = argparse.ArgAction.StoreTrue, help = 'verbose' )
	p.add_argument( ['-o', '--output'], default = 'out.txt', help = 'output file' )
	args: list[str] = ['file.txt', '-v', '--output', 'result.txt']
	ns: argparse.Namespace = p.parse_args( args )
	if ns.get_str( 'input' ) != 'file.txt':
		return 1
	if not ns.get_bool( 'verbose' ):
		return 2
	if ns.get_str( 'output' ) != 'result.txt':
		return 3
	return 0
''' ),
			( 'default_used_when_option_omitted', '''
import argparse

def main() -> i32:
	p: argparse.ArgumentParser = argparse.ArgumentParser( prog = 'myprog' )
	p.add_argument( ['input'] )
	p.add_argument( ['-o', '--output'], default = 'out.txt' )
	ns: argparse.Namespace = p.parse_args( ['file.txt'] )
	if ns.get_str( 'output' ) != 'out.txt':
		return 1
	if ns.get_bool( 'verbose' ):  # never registered - should read as false, not crash
		return 2
	return 0
''' ),
			( 'store_true_defaults_false_until_present', '''
import argparse

def main() -> i32:
	p: argparse.ArgumentParser = argparse.ArgumentParser( prog = 'myprog' )
	p.add_argument( ['-v', '--verbose'], action = argparse.ArgAction.StoreTrue )
	ns1: argparse.Namespace = p.parse_args( list[str]() )
	if ns1.get_bool( 'verbose' ):
		return 1
	ns2: argparse.Namespace = p.parse_args( ['-v'] )
	if not ns2.get_bool( 'verbose' ):
		return 2
	return 0
''' ),
			( 'store_false_defaults_true_until_present', '''
import argparse

def main() -> i32:
	p: argparse.ArgumentParser = argparse.ArgumentParser( prog = 'myprog' )
	p.add_argument( ['--no-cache'], action = argparse.ArgAction.StoreFalse, dest = 'cache' )
	ns1: argparse.Namespace = p.parse_args( list[str]() )
	if not ns1.get_bool( 'cache' ):
		return 1
	ns2: argparse.Namespace = p.parse_args( ['--no-cache'] )
	if ns2.get_bool( 'cache' ):
		return 2
	return 0
''' ),
			( 'append_action_collects_repeats_in_order', '''
import argparse

def main() -> i32:
	p: argparse.ArgumentParser = argparse.ArgumentParser( prog = 'myprog' )
	p.add_argument( ['--tag'], action = argparse.ArgAction.Append )
	ns: argparse.Namespace = p.parse_args( ['--tag', 'a', '--tag', 'b', '--tag', 'c'] )
	tags: list[str] = ns.get_list( 'tag' )
	if tags.__len__() != usize( 3 ):
		return 1
	if tags.__getitem__( 0 ).unwrap( '' ) != 'a':
		return 2
	if tags.__getitem__( 1 ).unwrap( '' ) != 'b':
		return 3
	if tags.__getitem__( 2 ).unwrap( '' ) != 'c':
		return 4
	return 0
''' ),
			( 'append_action_absent_is_empty_list', '''
import argparse

def main() -> i32:
	p: argparse.ArgumentParser = argparse.ArgumentParser( prog = 'myprog' )
	p.add_argument( ['--tag'], action = argparse.ArgAction.Append )
	ns: argparse.Namespace = p.parse_args( list[str]() )
	tags: list[str] = ns.get_list( 'tag' )
	if tags.__len__() != usize( 0 ):
		return 1
	return 0
''' ),
			( 'short_and_long_option_both_bind_same_dest', '''
import argparse

def main() -> i32:
	p: argparse.ArgumentParser = argparse.ArgumentParser( prog = 'myprog' )
	p.add_argument( ['-v', '--verbose'], action = argparse.ArgAction.StoreTrue )
	ns_short: argparse.Namespace = p.parse_args( ['-v'] )
	ns_long: argparse.Namespace = p.parse_args( ['--verbose'] )
	if not ns_short.get_bool( 'verbose' ):
		return 1
	if not ns_long.get_bool( 'verbose' ):
		return 2
	return 0
''' ),
			( 'explicit_dest_overrides_derived_name', '''
import argparse

def main() -> i32:
	p: argparse.ArgumentParser = argparse.ArgumentParser( prog = 'myprog' )
	p.add_argument( ['--output-file'], dest = 'outfile', default = 'a.out' )
	ns: argparse.Namespace = p.parse_args( ['--output-file', 'b.out'] )
	if ns.get_str( 'outfile' ) != 'b.out':
		return 1
	return 0
''' ),
			( 'multiple_positionals_assigned_in_order', '''
import argparse

def main() -> i32:
	p: argparse.ArgumentParser = argparse.ArgumentParser( prog = 'myprog' )
	p.add_argument( ['src'] )
	p.add_argument( ['dst'] )
	ns: argparse.Namespace = p.parse_args( ['a.txt', 'b.txt'] )
	if ns.get_str( 'src' ) != 'a.txt':
		return 1
	if ns.get_str( 'dst' ) != 'b.txt':
		return 2
	return 0
''' ),
		])

	@unittest.skipUnless( test_support.HAS_CC, 'no C compiler (clang/gcc/msvc) found - skipping' )
	def test_help_flag_prints_usage_and_exits_zero( self ) -> None:
		import emitter_c
		compiler = self._compile_source( '''
import argparse

def main() -> i32:
	p: argparse.ArgumentParser = argparse.ArgumentParser( prog = 'myprog' )
	p.add_argument( ['input'], help = 'input file' )
	ns: argparse.Namespace = p.parse_args( ['-h'] )
	return 99
''' )
		result = self._build_and_run( compiler, emitter_c.emit_c( compiler ), None )
		self.assertEqual( result.returncode, 0 )
		self.assertIn( b'usage: myprog', result.stdout )

	@unittest.skipUnless( test_support.HAS_CC, 'no C compiler (clang/gcc/msvc) found - skipping' )
	def test_missing_required_positional_exits_two( self ) -> None:
		import emitter_c
		compiler = self._compile_source( '''
import argparse

def main() -> i32:
	p: argparse.ArgumentParser = argparse.ArgumentParser( prog = 'myprog' )
	p.add_argument( ['input'], help = 'input file' )
	ns: argparse.Namespace = p.parse_args( list[str]() )
	return 99
''' )
		result = self._build_and_run( compiler, emitter_c.emit_c( compiler ), None )
		self.assertEqual( result.returncode, 2 )
		self.assertIn( b'error: the following arguments are required: input', result.stderr )

	@unittest.skipUnless( test_support.HAS_CC, 'no C compiler (clang/gcc/msvc) found - skipping' )
	def test_unrecognized_option_exits_two( self ) -> None:
		import emitter_c
		compiler = self._compile_source( '''
import argparse

def main() -> i32:
	p: argparse.ArgumentParser = argparse.ArgumentParser( prog = 'myprog' )
	ns: argparse.Namespace = p.parse_args( ['--nope'] )
	return 99
''' )
		result = self._build_and_run( compiler, emitter_c.emit_c( compiler ), None )
		self.assertEqual( result.returncode, 2 )
		self.assertIn( b'error: unrecognized arguments: --nope', result.stderr )

	@unittest.skipUnless( test_support.HAS_CC, 'no C compiler (clang/gcc/msvc) found - skipping' )
	def test_option_missing_its_value_exits_two( self ) -> None:
		import emitter_c
		compiler = self._compile_source( '''
import argparse

def main() -> i32:
	p: argparse.ArgumentParser = argparse.ArgumentParser( prog = 'myprog' )
	p.add_argument( ['-o', '--output'] )
	ns: argparse.Namespace = p.parse_args( ['-o'] )
	return 99
''' )
		result = self._build_and_run( compiler, emitter_c.emit_c( compiler ), None )
		self.assertEqual( result.returncode, 2 )
		self.assertIn( b'expected one argument', result.stderr )

	@unittest.skipUnless( test_support.HAS_CC, 'no C compiler (clang/gcc/msvc) found - skipping' )
	def test_missing_required_optional_exits_two( self ) -> None:
		import emitter_c
		compiler = self._compile_source( '''
import argparse

def main() -> i32:
	p: argparse.ArgumentParser = argparse.ArgumentParser( prog = 'myprog' )
	p.add_argument( ['--must-have'], required = True )
	ns: argparse.Namespace = p.parse_args( list[str]() )
	return 99
''' )
		result = self._build_and_run( compiler, emitter_c.emit_c( compiler ), None )
		self.assertEqual( result.returncode, 2 )
		self.assertIn( b'error: the following arguments are required', result.stderr )


if __name__ == '__main__':
	unittest.main()
