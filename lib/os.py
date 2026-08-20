'''
os — directory listing plus a handful of os.path helpers.

listdir: Windows FindFirstFileA/FindNextFileA/FindClose; POSIX opendir(3)/
	readdir(3)/closedir(3) (lib/posix/dirent.py). Excludes '.'/'..', matching
	Python's own os.listdir().

path.join/isdir/splitext/abspath/normpath: built entirely on str's own
	public methods (find/rfind/partition/rpartition/split/join/removeprefix/
	removesuffix/strip) - no raw byte-offset slicing, so these stay correct
	on non-ASCII path components.
'''

import compiler
import sys

# str.__getitem__(idx).unwrap_or('') instead of a plain '' default: the
# explicit-default unwrap_or(...) overload (Result[T,E].unwrap_or(default:
# T), lib/builtins/__init__.py's @overload-marked stub) is currently
# broken - it returns garbage/uninitialized memory at runtime rather than
# dispatching to the real implementation below it (task_d43b94e5, a
# separately-tracked bug, confirmed with a completely unrelated repro too -
# not something specific to str). match/case sidesteps it entirely.
def _char_at_or_empty( s: str, idx: usize ) -> str:
	match s.__getitem__( idx ):
		case Result.Ok( c ):
			return c
		case Result.Err( _ ):
			return ''

# ---------------------------------------------------------------------------
# listdir
# ---------------------------------------------------------------------------

@compiler.target( os = 'windows' )
def listdir( dirpath: str ) -> Result[list[str], OSError]:
	from windows.kernel32 import (
		WIN32_FIND_DATAA, FindFirstFileA, FindNextFileA, FindClose, GetLastError, INVALID_HANDLE_VALUE,
	)
	pattern: str = dirpath + '\\*'
	data: WIN32_FIND_DATAA = WIN32_FIND_DATAA()
	handle: Ptr[None] = FindFirstFileA( pattern.get_cstr(), compiler.addrof( data ))
	if handle == INVALID_HANDLE_VALUE:
		return Result.Err( OSError( GetLastError() ))
	defer( FindClose( handle ))

	entries: list[str] = list[str]()
	while True:
		name_ptr: ConstPtr[u8] = compiler.addrof( data.cFileName )
		name_len: usize = sys.cstrlen( name_ptr, 260 )
		with compiler.panic_arithmetic( 'bounded by the 260-byte cFileName buffer, cannot overflow' ):
			name_size: usize = name_len + 1
		name: str = str.from_cstr( name_ptr, name_size ).unwrap( 'os.listdir: invalid UTF-8 in filename' )
		if name != '.' and name != '..':
			entries.append( name ).unwrap( 'os.listdir: too many directory entries' )
		if not FindNextFileA( handle, compiler.addrof( data )):
			break
	return Result.Ok( entries )


@compiler.target( os = not 'windows' )
def listdir( dirpath: str ) -> Result[list[str], OSError]:
	from posix.dirent import opendir, readdir, closedir
	from crt import get_errno
	dirp = opendir( compiler.cast( ConstPtr[None], dirpath.get_cstr() ))
	if dirp is None:
		return Result.Err( OSError( get_errno() ))
	defer( closedir( dirp ))

	entries: list[str] = list[str]()
	while True:
		entry = readdir( dirp )
		if entry is None:
			break
		# d_name's real type is `char[N]` - c_field_addr requests Ptr[None]
		# (not Ptr[u8]) for the same char*-vs-unsigned-char* reason as
		# opendir/stat above, then casts back to ConstPtr[u8] for
		# sys.cstrlen/str.from_cstr below
		name_ptr: ConstPtr[u8] = compiler.cast( ConstPtr[u8], compiler.c_field_addr( entry, 'd_name', Ptr[None] ))
		name_len: usize = sys.cstrlen( name_ptr, 4096 )
		with compiler.panic_arithmetic( 'bounded by the 4096-byte cstrlen cap, cannot overflow' ):
			name_size: usize = name_len + 1
		name: str = str.from_cstr( name_ptr, name_size ).unwrap( 'os.listdir: invalid UTF-8 in filename' )
		if name != '.' and name != '..':
			entries.append( name ).unwrap( 'os.listdir: too many directory entries' )
	return Result.Ok( entries )


# ---------------------------------------------------------------------------
# os.path
# ---------------------------------------------------------------------------

@compiler.target( os = 'windows' )
def _getcwd() -> Result[str, OSError]:
	from windows.kernel32 import GetCurrentDirectoryA, GetLastError
	buf_size: u32 = 4096
	buf: Ptr[u8] = sys.alloc[u8]( usize( buf_size ))
	defer( sys.free( buf ))
	n: u32 = GetCurrentDirectoryA( buf_size, buf )
	if n == 0 or n >= buf_size:
		return Result.Err( OSError( GetLastError() ))
	with compiler.panic_arithmetic( 'bounded by n < buf_size, cannot overflow' ):
		size: usize = usize( n ) + 1
	return Result.Ok( str.from_cstr( compiler.cast( ConstPtr[u8], buf ), size ).unwrap( 'os.path.abspath: invalid UTF-8 in cwd '
		'(GetCurrentDirectoryA uses the system ANSI code page, not UTF-8 - see CreateFileA\'s own caveat)' ))


@compiler.target( os = not 'windows' )
def _getcwd() -> Result[str, OSError]:
	from crt import getcwd, get_errno
	buf_size: usize = 4096
	buf: Ptr[u8] = sys.alloc[u8]( buf_size )
	defer( sys.free( buf ))
	result: Ptr[u8] = getcwd( buf, buf_size )
	if result is None:
		return Result.Err( OSError( get_errno() ))
	n: usize = sys.cstrlen( buf, buf_size )
	with compiler.panic_arithmetic( 'bounded by cstrlen < buf_size, cannot overflow' ):
		size: usize = n + 1
	return Result.Ok( str.from_cstr( compiler.cast( ConstPtr[u8], buf ), size ).unwrap( 'os.path.abspath: invalid UTF-8 in cwd' ))


@compiler.target( os = 'windows' )
def _is_abs( p: str ) -> bool:
	# drive-letter ("C:\..." or "C:/...") or UNC ("\\server\share") - good
	# enough for real-world usage; doesn't recognize a drive-RELATIVE path
	# like "C:foo" as absolute, but neither does Python's own ntpath.isabs.
	if p.startswith( '\\\\' ) or p.startswith( '//' ):
		return True
	if len( p ) < 3:
		return False
	c1: str = _char_at_or_empty( p, 1 )
	c2: str = _char_at_or_empty( p, 2 )
	return c1 == ':' and ( c2 == '\\' or c2 == '/' )


@compiler.target( os = not 'windows' )
def _is_abs( p: str ) -> bool:
	return p.startswith( '/' )


class path:

	@staticmethod
	def join( a: str, b: str ) -> str:
		if a == '':
			return b
		if a.endswith( '/' ) or a.endswith( '\\' ):
			return a + b
		sep: str = '\\' if compiler.target.os == 'windows' else '/'
		return a + sep + b

	@compiler.target( os = 'windows' )
	@staticmethod
	def isdir( p: str ) -> bool:
		from windows.kernel32 import GetFileAttributesA, FILE_ATTRIBUTE_DIRECTORY, INVALID_FILE_ATTRIBUTES
		attrs: u32 = GetFileAttributesA( p.get_cstr() )
		if attrs == INVALID_FILE_ATTRIBUTES:
			return False
		return ( attrs & FILE_ATTRIBUTE_DIRECTORY ) != 0

	@compiler.target( os = not 'windows' )
	@staticmethod
	def isdir( p: str ) -> bool:
		from posix.stat import stat_t, stat, S_IFMT, S_IFDIR
		buf: Ptr[stat_t] = sys.alloc[stat_t]( 1 )
		if buf is None:
			return False
		defer( sys.free( compiler.cast( Ptr[None], buf )))
		rc: i32 = stat( compiler.cast( ConstPtr[None], p.get_cstr() ), buf )
		if rc != 0:
			return False
		mode: u32 = compiler.c_field( buf, 'st_mode', u32 )
		return ( mode & S_IFMT ) == S_IFDIR

	@staticmethod
	def splitext( p: str ) -> tuple[str, str]:
		# basename = whatever comes after the last '/' or '\\' (both
		# recognized on every platform, harmless on POSIX where '\\' is
		# just an ordinary filename byte that will never match) - found via
		# two rpartitions rather than manual byte-offset indexing, so this
		# stays correct on non-ASCII path components.
		_, _, after_slash = p.rpartition( '/' )
		_, backslash_sep, after_backslash = after_slash.rpartition( '\\' )
		basename: str = after_backslash if backslash_sep != '' else after_slash

		before_dot, dot_sep, ext_after_dot = basename.rpartition( '.' )
		# no dot in the basename at all, OR the basename is nothing but
		# leading dots ('.bashrc', '..') - Python's splitext requires at
		# least one non-dot byte before the extension's own dot
		if dot_sep == '' or before_dot.strip( '.' ) == '':
			return ( p, '' )
		ext: str = '.' + ext_after_dot
		return ( p.removesuffix( ext ), ext )

	@staticmethod
	def normpath( p: str ) -> str:
		if compiler.target.os == 'windows':
			sep: str = '\\'
			norm: str = p.replace( '/', '\\' )
		else:
			sep: str = '/'
			norm: str = p

		root: str = ''
		rest: str = norm
		if compiler.target.os == 'windows':
			if norm.startswith( '\\\\' ):
				# UNC - keep "\\server\share" intact, never collapse '..'
				# past the share root (matches CPython's own ntpath)
				root = '\\\\'
				rest = norm.removeprefix( '\\\\' )
			elif len( norm ) >= 2 and _char_at_or_empty( norm, 1 ) == ':':
				drive: str = _char_at_or_empty( norm, 0 )
				root = drive + ':\\'
				rest = norm.removeprefix( drive + ':' ).removeprefix( '\\' )
		elif norm.startswith( '/' ):
			root = '/'
			rest = norm.removeprefix( '/' )

		parts: list[str] = rest.split( sep )
		kept: list[str] = list[str]()
		i: usize = 0
		with compiler.panic_arithmetic( 'bounded by len(parts), cannot overflow' ):
			while i < len( parts ):
				part: str = parts.__getitem__( i ).unwrap( 'os.path.normpath: index in bounds by construction' )
				i += 1
				if part == '' or part == '.':
					continue
				if part == '..':
					if len( kept ) > 0:
						kept.pop().unwrap( 'os.path.normpath: pop failed' )
					elif root == '':
						kept.append( part ).unwrap( 'os.path.normpath: append failed' )
					continue
				kept.append( part ).unwrap( 'os.path.normpath: append failed' )

		joined: str = sep.join( kept )
		if root != '':
			return root + joined
		if joined == '':
			return '.'
		return joined

	@staticmethod
	def abspath( p: str ) -> str:
		if _is_abs( p ):
			return path.normpath( p )
		cwd: str = _getcwd().unwrap( 'os.path.abspath: could not determine current directory' )
		return path.normpath( path.join( cwd, p ))
