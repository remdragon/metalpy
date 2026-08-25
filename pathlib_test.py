# Real-compile-and-run tests for lib/pathlib.py: pure parsing (both flavors,
# via Path.posix()/Path.windows() so the same test process exercises both
# regardless of which platform it's actually compiled on) + concrete
# filesystem ops (native flavor only, real temp files/dirs via Python's own
# tempfile - same fixture convention as os_test.py).

# stdlib imports:
import os
from pathlib import Path
import unittest

# local imports:
import test_support


class PathlibPureTests( test_support.RealCompileMixin, unittest.TestCase ):
	''' parsing/joining/naming - no filesystem I/O. Both PathFlavor.Posix and
	PathFlavor.Windows are exercised explicitly via Path.posix()/
	Path.windows(), independent of which platform this test process itself
	runs on. '''

	def setUp( self ) -> None:
		from discovery import Discovery
		from compiler import Compiler
		self.discovery = Discovery( import_builtins = True )
		self.compiler = Compiler( self.discovery )

	@unittest.skipUnless( test_support.HAS_CC, 'no C compiler (clang/gcc/msvc) found - skipping' )
	def test_programs_compile_and_run( self ) -> None:
		cases = [
			( 'posix_name_suffix_stem', '''
from pathlib import Path

def main() -> i32:
	p: Path = Path.posix( 'a/b/c.tar.gz' )
	if p.name != 'c.tar.gz':
		return 1
	if p.suffix != '.gz':
		return 2
	if p.stem != 'c.tar':
		return 3
	return 0
''' ),
			( 'posix_leading_dot_is_not_a_suffix', '''
from pathlib import Path

def main() -> i32:
	p: Path = Path.posix( '.bashrc' )
	if p.suffix != '':
		return 1
	if p.stem != '.bashrc':
		return 2
	return 0
''' ),
			( 'posix_duplicate_separators_and_dot_segments_collapse', '''
from pathlib import Path

def main() -> i32:
	p: Path = Path.posix( 'a//./b/./c' )
	if str( p ) != 'a/b/c':
		return 1
	return 0
''' ),
			( 'posix_dotdot_segments_are_kept_literally', '''
from pathlib import Path

def main() -> i32:
	p: Path = Path.posix( 'a/../b' )
	if str( p ) != 'a/../b':
		return 1
	return 0
''' ),
			( 'posix_empty_string_is_dot', '''
from pathlib import Path

def main() -> i32:
	p: Path = Path.posix( '' )
	if str( p ) != '.':
		return 1
	if p.name != '':
		return 2
	return 0
''' ),
			( 'posix_root_name_and_parent', '''
from pathlib import Path

def main() -> i32:
	root: Path = Path.posix( '/' )
	if root.name != '':
		return 1
	if root.parent != root:
		return 2
	if not root.is_absolute():
		return 3
	return 0
''' ),
			( 'posix_dot_parent_is_itself', '''
from pathlib import Path

def main() -> i32:
	p: Path = Path.posix( '.' )
	if p.parent != p:
		return 1
	if p.name != '':
		return 2
	return 0
''' ),
			( 'posix_parent_of_relative_single_segment_is_dot', '''
from pathlib import Path

def main() -> i32:
	p: Path = Path.posix( 'a' )
	if p.parent != Path.posix( '.' ):
		return 1
	return 0
''' ),
			( 'posix_parents_list', '''
from pathlib import Path

def main() -> i32:
	p: Path = Path.posix( 'a/b/c' )
	parents: list[Path] = p.parents
	if len( parents ) != 3:
		return 1
	if parents.__getitem__( 0 ).unwrap( 'idx' ) != Path.posix( 'a/b' ):
		return 2
	if parents.__getitem__( 1 ).unwrap( 'idx' ) != Path.posix( 'a' ):
		return 3
	if parents.__getitem__( 2 ).unwrap( 'idx' ) != Path.posix( '.' ):
		return 4
	return 0
''' ),
			( 'posix_parts', '''
from pathlib import Path

def main() -> i32:
	p: Path = Path.posix( '/a/b' )
	parts: list[str] = p.parts
	if len( parts ) != 3:
		return 1
	if parts.__getitem__( 0 ).unwrap( 'idx' ) != '/':
		return 2
	if parts.__getitem__( 1 ).unwrap( 'idx' ) != 'a':
		return 3
	if parts.__getitem__( 2 ).unwrap( 'idx' ) != 'b':
		return 4
	return 0
''' ),
			( 'posix_joinpath_and_truediv', '''
from pathlib import Path

def main() -> i32:
	p: Path = Path.posix( 'a' ) / 'b' / 'c'
	if str( p ) != 'a/b/c':
		return 1
	if p != Path.posix( 'a' ).joinpath( 'b/c' ):
		return 2
	return 0
''' ),
			( 'posix_joinpath_absolute_replaces_self', '''
from pathlib import Path

def main() -> i32:
	p: Path = Path.posix( '/a/b' ) / '/c/d'
	if str( p ) != '/c/d':
		return 1
	return 0
''' ),
			( 'posix_with_name', '''
from pathlib import Path

def main() -> i32:
	p: Path = Path.posix( 'a/b/c.txt' ).with_name( 'd.csv' )
	if str( p ) != 'a/b/d.csv':
		return 1
	return 0
''' ),
			( 'posix_with_suffix', '''
from pathlib import Path

def main() -> i32:
	p: Path = Path.posix( 'a/b.txt' ).with_suffix( '.csv' )
	if str( p ) != 'a/b.csv':
		return 1
	q: Path = Path.posix( 'a/b' ).with_suffix( '.csv' )
	if str( q ) != 'a/b.csv':
		return 2
	return 0
''' ),
			( 'posix_relative_path_is_not_absolute', '''
from pathlib import Path

def main() -> i32:
	if Path.posix( 'a/b' ).is_absolute():
		return 1
	return 0
''' ),
			( 'posix_equality_is_case_sensitive', '''
from pathlib import Path

def main() -> i32:
	if Path.posix( 'A' ) == Path.posix( 'a' ):
		return 1
	if Path.posix( 'a/b' ) != Path.posix( 'a/b' ):
		return 2
	return 0
''' ),
			( 'windows_drive_root_is_absolute', '''
from pathlib import Path

def main() -> i32:
	p: Path = Path.windows( 'C:\\\\a\\\\b' )
	if not p.is_absolute():
		return 1
	if p.name != 'b':
		return 2
	if p.anchor != 'C:\\\\':
		return 3
	return 0
''' ),
			( 'windows_drive_relative_is_not_absolute', '''
from pathlib import Path

def main() -> i32:
	if Path.windows( 'C:foo' ).is_absolute():
		return 1
	return 0
''' ),
			( 'windows_forward_slash_is_recognized_as_separator', '''
from pathlib import Path

def main() -> i32:
	p: Path = Path.windows( 'a/b/c' )
	if str( p ) != 'a\\\\b\\\\c':
		return 1
	if p.name != 'c':
		return 2
	return 0
''' ),
			( 'windows_unc_prefix_is_absolute', '''
from pathlib import Path

def main() -> i32:
	p: Path = Path.windows( '\\\\\\\\server\\\\share\\\\x' )
	if not p.is_absolute():
		return 1
	return 0
''' ),
			( 'windows_equality_is_case_insensitive', '''
from pathlib import Path

def main() -> i32:
	if Path.windows( 'C:\\\\A' ) != Path.windows( 'C:\\\\a' ):
		return 1
	return 0
''' ),
			( 'windows_and_posix_flavors_never_compare_equal_even_with_same_text', '''
from pathlib import Path

def main() -> i32:
	if Path.posix( 'a' ) == Path.windows( 'a' ):
		return 1
	return 0
''' ),
		]
		self.assert_programs_run( cases )


class PathlibFsTests( test_support.RealCompileMixin, unittest.TestCase ):
	''' concrete filesystem ops (native flavor) - real temp directory,
	created fresh per test method via Python's own tempfile. '''

	def setUp( self ) -> None:
		from discovery import Discovery
		from compiler import Compiler
		self.discovery = Discovery( import_builtins = True )
		self.compiler = Compiler( self.discovery )

		import tempfile
		self._tmpdir_ctx = tempfile.TemporaryDirectory()
		self.tmpdir = self._tmpdir_ctx.name

	def tearDown( self ) -> None:
		self._tmpdir_ctx.cleanup()

	@unittest.skipUnless( test_support.HAS_CC, 'no C compiler (clang/gcc/msvc) found - skipping' )
	def test_mkdir_exists_is_dir_rmdir( self ) -> None:
		sub_literal = repr( os.path.join( self.tmpdir, 'newdir' ))
		self.assert_programs_run([
			( 'mkdir_then_exists_and_is_dir', f'''
from pathlib import Path

def main() -> i32:
	p: Path = Path( {sub_literal} )
	if p.exists():
		return 1
	p.mkdir().unwrap( 'mkdir failed' )
	if not p.exists():
		return 2
	if not p.is_dir():
		return 3
	if p.is_file():
		return 4
	p.rmdir().unwrap( 'rmdir failed' )
	if p.exists():
		return 5
	return 0
''' ),
			( 'mkdir_exist_ok_on_existing_dir_succeeds', f'''
from pathlib import Path

def main() -> i32:
	p: Path = Path( {sub_literal} )
	p.mkdir().unwrap( 'mkdir failed' )
	p.mkdir( exist_ok = True ).unwrap( 'mkdir exist_ok=True should not fail on an existing dir' )
	match p.mkdir():
		case Result.Ok( _ ):
			return 1
		case Result.Err( _ ):
			pass
	p.rmdir().unwrap( 'rmdir failed' )
	return 0
''' ),
			( 'mkdir_parents_creates_intermediate_dirs', f'''
from pathlib import Path

def main() -> i32:
	base: Path = Path( {sub_literal} )
	nested: Path = base / 'a' / 'b'
	nested.mkdir( parents = True ).unwrap( 'mkdir parents=True failed' )
	if not nested.is_dir():
		return 1
	nested.rmdir().unwrap( 'rmdir failed' )
	( base / 'a' ).rmdir().unwrap( 'rmdir failed' )
	base.rmdir().unwrap( 'rmdir failed' )
	return 0
''' ),
		])

	@unittest.skipUnless( test_support.HAS_CC, 'no C compiler (clang/gcc/msvc) found - skipping' )
	def test_read_write_text_and_bytes_roundtrip( self ) -> None:
		file_literal = repr( os.path.join( self.tmpdir, 'hello.txt' ))
		self.assert_programs_run([
			( 'write_text_then_read_text_roundtrip', f'''
from pathlib import Path

def main() -> i32:
	p: Path = Path( {file_literal} )
	p.write_text( 'hello, world' ).unwrap( 'write_text failed' )
	content: str = p.read_text().unwrap( 'read_text failed' )
	if content != 'hello, world':
		return 1
	p.unlink().unwrap( 'unlink failed' )
	return 0
''' ),
			( 'write_bytes_then_read_bytes_roundtrip', f'''
from pathlib import Path

def main() -> i32:
	p: Path = Path( {file_literal} )
	data: bytearray = bytearray( 3 )
	data.__setitem__( 0, 65 )
	data.__setitem__( 1, 66 )
	data.__setitem__( 2, 67 )
	p.write_bytes( data ).unwrap( 'write_bytes failed' )
	back: bytearray = p.read_bytes().unwrap( 'read_bytes failed' )
	if len( back ) != 3:
		return 1
	if back.__getitem__( 0 ).unwrap( 'idx' ) != 65:
		return 2
	if back.__getitem__( 2 ).unwrap( 'idx' ) != 67:
		return 3
	p.unlink().unwrap( 'unlink failed' )
	return 0
''' ),
			( 'unlink_missing_ok_on_missing_file_succeeds', f'''
from pathlib import Path

def main() -> i32:
	p: Path = Path( {file_literal} )
	p.unlink( missing_ok = True ).unwrap( 'unlink missing_ok=True should not fail on a missing file' )
	match p.unlink():
		case Result.Ok( _ ):
			return 1
		case Result.Err( _ ):
			pass
	return 0
''' ),
		])

	@unittest.skipUnless( test_support.HAS_CC, 'no C compiler (clang/gcc/msvc) found - skipping' )
	def test_iterdir_and_rename( self ) -> None:
		tmpdir_literal = repr( self.tmpdir )
		src_literal = repr( os.path.join( self.tmpdir, 'src.txt' ))
		dst_literal = repr( os.path.join( self.tmpdir, 'dst.txt' ))
		self.assert_programs_run([
			( 'iterdir_sees_created_file', f'''
from pathlib import Path

def main() -> i32:
	d: Path = Path( {tmpdir_literal} )
	f: Path = d / 'entry.txt'
	f.write_text( 'x' ).unwrap( 'write_text failed' )
	entries: list[Path] = d.iterdir().unwrap( 'iterdir failed' )
	if len( entries ) != 1:
		return 1
	if entries.__getitem__( 0 ).unwrap( 'idx' ).name != 'entry.txt':
		return 2
	f.unlink().unwrap( 'unlink failed' )
	return 0
''' ),
			( 'rename_moves_file_and_returns_new_path', f'''
from pathlib import Path

def main() -> i32:
	src: Path = Path( {src_literal} )
	dst: Path = Path( {dst_literal} )
	src.write_text( 'moved' ).unwrap( 'write_text failed' )
	result: Path = src.rename( {dst_literal} ).unwrap( 'rename failed' )
	if result != dst:
		return 1
	if src.exists():
		return 2
	if not dst.exists():
		return 3
	if dst.read_text().unwrap( 'read_text failed' ) != 'moved':
		return 4
	dst.unlink().unwrap( 'unlink failed' )
	return 0
''' ),
			( 'rename_fails_when_dst_exists', f'''
from pathlib import Path

def main() -> i32:
	src: Path = Path( {src_literal} )
	dst: Path = Path( {dst_literal} )
	src.write_text( 'src' ).unwrap( 'write src failed' )
	dst.write_text( 'dst' ).unwrap( 'write dst failed' )

	match src.rename( {dst_literal} ):
		case Result.Ok( _ ):
			return 1
		case Result.Err( _ ):
			pass

	if not src.exists():
		return 2
	if dst.read_text().unwrap( 'read_text failed' ) != 'dst':
		return 3

	src.unlink().unwrap( 'unlink src failed' )
	dst.unlink().unwrap( 'unlink dst failed' )
	return 0
''' ),
		])

	@unittest.skipUnless( test_support.HAS_CC, 'no C compiler (clang/gcc/msvc) found - skipping' )
	def test_cwd_and_foreign_flavor_fs_ops( self ) -> None:
		self.compiler.import_code( '''
from pathlib import Path, PathFlavor

def main() -> i32:
	cwd: Path = Path.cwd().unwrap( 'cwd failed' )
	if not cwd.is_absolute():
		return 1

	# a foreign-flavor Path (opposite of the native platform) must refuse
	# every filesystem op rather than hand a wrong-syntax string to a
	# native OS call
	foreign: Path
	if compiler.target.os == 'windows':
		foreign = Path.posix( '/definitely/not/a/real/path/12345' )
	else:
		foreign = Path.windows( 'C:\\\\definitely\\\\not\\\\a\\\\real\\\\path\\\\12345' )
	if foreign.exists():
		return 2
	if foreign.is_file():
		return 3
	if foreign.is_dir():
		return 4
	match foreign.mkdir():
		case Result.Ok( _ ):
			return 5
		case Result.Err( e ):
			if e != OSError.Invalid:
				return 6
	return 0
''', Path( '__main__.py' ), scope = None )
		self.compiler.run()
		self.assertEqual( self.discovery.errors.errors, [] )
		import emitter_c
		self._assert_compiles_and_runs( emitter_c.emit_c( self.compiler ), expected_exit = 0 )


if __name__ == '__main__':
	unittest.main()
