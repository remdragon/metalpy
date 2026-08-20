# Real-compile-and-run tests for lib/os.py: listdir + os.path helpers
# (join/isdir/splitext/normpath/abspath). Uses REAL temp directories/files
# created via Python's own tempfile - each fixture path is baked into the
# generated MetalPy source as a string literal (repr() of a Python str is
# also valid MetalPy source syntax, since both share the same ast.parse/
# ast.unparse pipeline - same escaping rules apply automatically).
#
# abspath('.') can't be checked against a known expected string: the
# compiled EXE's own cwd is an ephemeral tempdir created internally by
# RealCompileMixin._build_and_run, not something this file controls or can
# predict ahead of compiling. So abspath is verified by CONTRACT instead
# (idempotent on an already-absolute path; always produces an absolute
# result) rather than by comparing against one hardcoded "the" cwd string.

# stdlib imports:
import os
from pathlib import Path
import unittest

# local imports:
import test_support


class OsPathPureTests( test_support.RealCompileMixin, unittest.TestCase ):
	''' join/splitext/normpath - no filesystem I/O, so these run as ordinary
	string-manipulation checks. '''

	def setUp( self ) -> None:
		from discovery import Discovery
		from compiler import Compiler
		self.discovery = Discovery( import_builtins = True )
		self.compiler = Compiler( self.discovery )

	@unittest.skipUnless( test_support.HAS_CC, 'no C compiler (clang/gcc/msvc) found - skipping' )
	def test_programs_compile_and_run( self ) -> None:
		# doubled, not a single backslash: this gets interpolated INSIDE a
		# quoted string literal in the generated MetalPy source below, where
		# a lone backslash would be parsed as (the start of) an escape
		# sequence rather than a literal separator byte
		sep = '\\\\' if os.name == 'nt' else '/'
		cases = [
			( 'join_no_trailing_sep', f'''
import os

def main() -> i32:
	joined: str = os.path.join( 'a', 'b' )
	if joined != 'a{sep}b':
		return 1
	return 0
''' ),
			( 'join_already_has_trailing_sep', f'''
import os

def main() -> i32:
	joined: str = os.path.join( 'a{sep}', 'b' )
	if joined != 'a{sep}b':
		return 1
	return 0
''' ),
			( 'join_empty_first_arg', '''
import os

def main() -> i32:
	joined: str = os.path.join( '', 'b' )
	if joined != 'b':
		return 1
	return 0
''' ),
			( 'splitext_simple', '''
import os

def main() -> i32:
	root: str
	ext: str
	root, ext = os.path.splitext( 'file.txt' )
	if root != 'file':
		return 1
	if ext != '.txt':
		return 2
	return 0
''' ),
			( 'splitext_multiple_dots_keeps_only_last', '''
import os

def main() -> i32:
	root: str
	ext: str
	root, ext = os.path.splitext( 'archive.tar.gz' )
	if root != 'archive.tar':
		return 1
	if ext != '.gz':
		return 2
	return 0
''' ),
			( 'splitext_leading_dot_is_not_an_extension', '''
import os

def main() -> i32:
	root: str
	ext: str
	root, ext = os.path.splitext( '.bashrc' )
	if root != '.bashrc':
		return 1
	if ext != '':
		return 2
	return 0
''' ),
			( 'splitext_no_extension', '''
import os

def main() -> i32:
	root: str
	ext: str
	root, ext = os.path.splitext( 'noext' )
	if root != 'noext':
		return 1
	if ext != '':
		return 2
	return 0
''' ),
			( 'splitext_ignores_dot_in_directory_component', f'''
import os

def main() -> i32:
	root: str
	ext: str
	root, ext = os.path.splitext( 'dir.with.dots{sep}file' )
	if root != 'dir.with.dots{sep}file':
		return 1
	if ext != '':
		return 2
	return 0
''' ),
			( 'splitext_dot_after_last_separator_only', f'''
import os

def main() -> i32:
	root: str
	ext: str
	root, ext = os.path.splitext( 'a.b{sep}c.d' )
	if root != 'a.b{sep}c':
		return 1
	if ext != '.d':
		return 2
	return 0
''' ),
			( 'normpath_collapses_dot_segments', f'''
import os

def main() -> i32:
	result: str = os.path.normpath( 'a{sep}.{sep}b' )
	if result != 'a{sep}b':
		return 1
	return 0
''' ),
			( 'normpath_resolves_dotdot', f'''
import os

def main() -> i32:
	result: str = os.path.normpath( 'a{sep}b{sep}..{sep}c' )
	if result != 'a{sep}c':
		return 1
	return 0
''' ),
			( 'normpath_leading_dotdot_on_relative_path_is_kept', f'''
import os

def main() -> i32:
	result: str = os.path.normpath( '..{sep}a' )
	if result != '..{sep}a':
		return 1
	return 0
''' ),
			( 'abspath_is_idempotent_on_an_already_absolute_path', f'''
import os

def main() -> i32:
	base: str = os.path.normpath( os.path.abspath( '.' ))
	once: str = os.path.abspath( base )
	twice: str = os.path.abspath( once )
	if once != twice:
		return 1
	return 0
''' ),
			( 'abspath_of_relative_path_is_absolute', '''
import os

def main() -> i32:
	result: str = os.path.abspath( 'somefile.txt' )
	if result == 'somefile.txt':
		return 1
	if compiler.target.os == 'windows':
		match result.__getitem__( 1 ):
			case Result.Ok( colon ):
				if colon != ':':
					return 2
			case Result.Err( _ ):
				return 3
	else:
		if not result.startswith( '/' ):
			return 2
	return 0
''' ),
		]
		if os.name == 'nt':
			cases.append(( 'normpath_normalizes_forward_slash_to_backslash', '''
import os

def main() -> i32:
	result: str = os.path.normpath( 'a/./b' )
	if result != 'a\\\\b':
		return 1
	return 0
''' ))
		self.assert_programs_run( cases )


class OsListdirIsdirTests( test_support.RealCompileMixin, unittest.TestCase ):
	''' listdir + isdir - real filesystem I/O against a fixture directory
	tree created in setUp/tearDown. '''

	def setUp( self ) -> None:
		from discovery import Discovery
		from compiler import Compiler
		self.discovery = Discovery( import_builtins = True )
		self.compiler = Compiler( self.discovery )

		import tempfile
		self._tmpdir_ctx = tempfile.TemporaryDirectory()
		self.tmpdir = self._tmpdir_ctx.name
		for name in ( 'alpha.txt', 'beta.txt' ):
			with open( os.path.join( self.tmpdir, name ), 'w' ) as f:
				f.write( 'x' )
		os.mkdir( os.path.join( self.tmpdir, 'subdir' ))

	def tearDown( self ) -> None:
		self._tmpdir_ctx.cleanup()

	@unittest.skipUnless( test_support.HAS_CC, 'no C compiler (clang/gcc/msvc) found - skipping' )
	def test_listdir_and_isdir( self ) -> None:
		tmpdir_literal = repr( self.tmpdir )
		alpha_literal = repr( os.path.join( self.tmpdir, 'alpha.txt' ))
		subdir_literal = repr( os.path.join( self.tmpdir, 'subdir' ))
		self.assert_programs_run([
			( 'listdir_sees_every_entry_no_dot_entries', f'''
import os

def main() -> i32:
	entries: list[str] = os.listdir( {tmpdir_literal} ).unwrap( 'listdir failed' )
	if len( entries ) != 3:
		return 1
	saw_alpha: bool = False
	saw_beta: bool = False
	saw_subdir: bool = False
	saw_dot: bool = False
	i: usize = 0
	with compiler.wrap_arithmetic:
		while i < len( entries ):
			name: str = entries.__getitem__( i ).unwrap( 'idx' )
			if name == 'alpha.txt':
				saw_alpha = True
			elif name == 'beta.txt':
				saw_beta = True
			elif name == 'subdir':
				saw_subdir = True
			elif name == '.' or name == '..':
				saw_dot = True
			i += 1
	if not saw_alpha:
		return 2
	if not saw_beta:
		return 3
	if not saw_subdir:
		return 4
	if saw_dot:
		return 5
	return 0
''' ),
			( 'isdir_true_for_directory_false_for_file', f'''
import os

def main() -> i32:
	if not os.path.isdir( {subdir_literal} ):
		return 1
	if os.path.isdir( {alpha_literal} ):
		return 2
	return 0
''' ),
			( 'isdir_false_for_nonexistent_path', '''
import os

def main() -> i32:
	if os.path.isdir( 'this_path_should_not_exist_anywhere_12345' ):
		return 1
	return 0
''' ),
		])

	@unittest.skipUnless( test_support.HAS_CC, 'no C compiler (clang/gcc/msvc) found - skipping' )
	def test_listdir_of_missing_directory_is_an_error( self ) -> None:
		self.compiler.import_code( '''
import os

def main() -> i32:
	match os.listdir( 'this_path_should_not_exist_anywhere_12345' ):
		case Result.Ok( _ ):
			return 1
		case Result.Err( _ ):
			return 0
''', Path( '__main__.py' ), scope = None )
		self.compiler.run()
		self.assertEqual( self.discovery.errors.errors, [] )
		import emitter_c
		self._assert_compiles_and_runs( emitter_c.emit_c( self.compiler ), expected_exit = 0 )


if __name__ == '__main__':
	unittest.main()
