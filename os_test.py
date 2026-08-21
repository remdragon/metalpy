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
		# same doubled-backslash-for-embedding trick as `sep` above, for an
		# OS-appropriate absolute-path prefix (drive letter on Windows,
		# root slash on POSIX) - lets one case cover both platforms
		abs_prefix = 'C:\\\\' if os.name == 'nt' else '/'
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
		colon: str = result.__getitem__( 1 ).unwrap_or( '' )
		if colon != ':':
			return 2
	else:
		if not result.startswith( '/' ):
			return 2
	return 0
''' ),
			( 'basename_with_separators', f'''
import os

def main() -> i32:
	result: str = os.path.basename( 'a{sep}b{sep}c' )
	if result != 'c':
		return 1
	return 0
''' ),
			( 'basename_trailing_separator', f'''
import os

def main() -> i32:
	result: str = os.path.basename( 'a{sep}b{sep}' )
	if result != '':
		return 1
	return 0
''' ),
			( 'basename_no_separator', '''
import os

def main() -> i32:
	result: str = os.path.basename( 'file.txt' )
	if result != 'file.txt':
		return 1
	return 0
''' ),
			( 'dirname_with_separators', f'''
import os

def main() -> i32:
	result: str = os.path.dirname( 'a{sep}b{sep}c' )
	if result != 'a{sep}b':
		return 1
	return 0
''' ),
			( 'dirname_trailing_separator', f'''
import os

def main() -> i32:
	result: str = os.path.dirname( 'a{sep}b{sep}' )
	if result != 'a{sep}b':
		return 1
	return 0
''' ),
			( 'dirname_root_is_kept_whole', f'''
import os

def main() -> i32:
	result: str = os.path.dirname( '{sep}a' )
	if result != '{sep}':
		return 1
	return 0
''' ),
			( 'dirname_no_separator', '''
import os

def main() -> i32:
	result: str = os.path.dirname( 'a' )
	if result != '':
		return 1
	return 0
''' ),
			( 'isabs_true_for_platform_absolute_path', f'''
import os

def main() -> i32:
	if not os.path.isabs( '{abs_prefix}a' ):
		return 1
	return 0
''' ),
			( 'isabs_false_for_relative_path', '''
import os

def main() -> i32:
	if os.path.isabs( 'a' ):
		return 1
	return 0
''' ),
			( 'os_sep_matches_platform_separator', f'''
import os

def main() -> i32:
	if os.sep != '{sep}':
		return 1
	return 0
''' ),
			( 'commonpath_relative', f'''
import os

def main() -> i32:
	paths: list[str] = list[str]()
	paths.append( 'a{sep}b{sep}c' ).unwrap( 'append failed' )
	paths.append( 'a{sep}b{sep}d' ).unwrap( 'append failed' )
	result: str = os.path.commonpath( paths ).unwrap( 'commonpath failed' )
	if result != 'a{sep}b':
		return 1
	return 0
''' ),
			( 'commonpath_absolute', f'''
import os

def main() -> i32:
	paths: list[str] = list[str]()
	paths.append( '{abs_prefix}a{sep}b{sep}c' ).unwrap( 'append failed' )
	paths.append( '{abs_prefix}a{sep}b{sep}d' ).unwrap( 'append failed' )
	result: str = os.path.commonpath( paths ).unwrap( 'commonpath failed' )
	if result != '{abs_prefix}a{sep}b':
		return 1
	return 0
''' ),
			( 'commonpath_mixed_absolute_and_relative_is_an_error', f'''
import os

def main() -> i32:
	paths: list[str] = list[str]()
	paths.append( 'a{sep}b' ).unwrap( 'append failed' )
	paths.append( '{abs_prefix}a{sep}b' ).unwrap( 'append failed' )
	match os.path.commonpath( paths ):
		case Result.Ok( _ ):
			return 1
		case Result.Err( _ ):
			return 0
''' ),
			( 'commonpath_empty_list_is_an_error', '''
import os

def main() -> i32:
	paths: list[str] = list[str]()
	match os.path.commonpath( paths ):
		case Result.Ok( _ ):
			return 1
		case Result.Err( _ ):
			return 0
''' ),
			( 'relpath_descends_into_child', f'''
import os

def main() -> i32:
	result: str = os.path.relpath( 'a{sep}b{sep}c', 'a{sep}b' ).unwrap( 'relpath failed' )
	if result != 'c':
		return 1
	return 0
''' ),
			( 'relpath_ascends_to_parent', f'''
import os

def main() -> i32:
	result: str = os.path.relpath( 'a', 'a{sep}b' ).unwrap( 'relpath failed' )
	if result != '..':
		return 1
	return 0
''' ),
			( 'relpath_of_identical_paths_is_dot', f'''
import os

def main() -> i32:
	result: str = os.path.relpath( 'a{sep}b', 'a{sep}b' ).unwrap( 'relpath failed' )
	if result != '.':
		return 1
	return 0
''' ),
			( 'is_relative_to_identical_paths', f'''
import os

def main() -> i32:
	if not os.path.is_relative_to( 'a{sep}b', 'a{sep}b' ):
		return 1
	return 0
''' ),
			( 'is_relative_to_child_of_parent', f'''
import os

def main() -> i32:
	if not os.path.is_relative_to( 'a{sep}b', 'a' ):
		return 1
	return 0
''' ),
			( 'is_relative_to_sibling_directories_is_false', f'''
import os

def main() -> i32:
	if os.path.is_relative_to( 'a{sep}b', 'a{sep}c' ):
		return 1
	return 0
''' ),
			( 'is_relative_to_trailing_slash_both_directions', f'''
import os

def main() -> i32:
	if not os.path.is_relative_to( 'a{sep}b{sep}', 'a{sep}b' ):
		return 1
	if not os.path.is_relative_to( 'a{sep}b', 'a{sep}b{sep}' ):
		return 2
	return 0
''' ),
			( 'is_relative_to_mismatched_relative_vs_absolute_is_false', f'''
import os

def main() -> i32:
	if os.path.is_relative_to( 'a', '{abs_prefix}a' ):
		return 1
	return 0
''' ),
			( 'is_relative_to_agrees_with_manual_dirname_walk', f'''
import os

def _is_ancestor_via_dirname_walk( child: str, candidate_parent: str ) -> bool:
	current: str = os.path.normpath( child )
	target: str = os.path.normpath( candidate_parent )
	with compiler.wrap_arithmetic:
		i: i32 = 0
		while i < 64:
			if current == target:
				return True
			parent: str = os.path.dirname( current )
			if parent == current:
				return False
			current = parent
			i += 1
	return False

def main() -> i32:
	# webchat Method 4 (manual dirname-walk), reimplemented here with the
	# now-real os.path.dirname primitive, cross-checked against
	# os.path.is_relative_to on the same pairs
	true_parent: bool = _is_ancestor_via_dirname_walk( 'a{sep}b{sep}c', 'a{sep}b' )
	if not true_parent:
		return 1
	if true_parent != os.path.is_relative_to( 'a{sep}b{sep}c', 'a{sep}b' ):
		return 2
	sibling_mismatch: bool = _is_ancestor_via_dirname_walk( 'a{sep}b{sep}c', 'a{sep}x' )
	if sibling_mismatch:
		return 3
	if sibling_mismatch != os.path.is_relative_to( 'a{sep}b{sep}c', 'a{sep}x' ):
		return 4
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
			cases.append(( 'dirname_drive_root_is_kept_whole', '''
import os

def main() -> i32:
	result: str = os.path.dirname( 'C:\\\\a' )
	if result != 'C:\\\\':
		return 1
	return 0
''' ))
			cases.append(( 'commonpath_cross_drive_is_an_error', '''
import os

def main() -> i32:
	paths: list[str] = list[str]()
	paths.append( 'C:\\\\a' ).unwrap( 'append failed' )
	paths.append( 'D:\\\\b' ).unwrap( 'append failed' )
	match os.path.commonpath( paths ):
		case Result.Ok( _ ):
			return 1
		case Result.Err( _ ):
			return 0
''' ))
			cases.append(( 'relpath_cross_drive_is_an_error', '''
import os

def main() -> i32:
	match os.path.relpath( 'D:\\\\a', 'C:\\\\a' ):
		case Result.Ok( _ ):
			return 1
		case Result.Err( _ ):
			return 0
''' ))
			cases.append(( 'is_relative_to_cross_drive_is_false', '''
import os

def main() -> i32:
	if os.path.is_relative_to( 'D:\\\\a', 'C:\\\\a' ):
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
