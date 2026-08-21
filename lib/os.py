'''
os — directory listing plus a handful of os.path helpers.

listdir: Windows FindFirstFileA/FindNextFileA/FindClose; POSIX opendir(3)/
	readdir(3)/closedir(3) (lib/posix/dirent.py). Excludes '.'/'..', matching
	Python's own os.listdir().

path.join/isdir/splitext/abspath/normpath/basename/dirname/isabs/commonpath/
	relpath/is_relative_to: built entirely on str's own public methods
	(find/rfind/partition/rpartition/split/join/removeprefix/removesuffix/
	strip) - no raw byte-offset slicing, so these stay correct on non-ASCII
	path components.

path.is_relative_to is purely lexical (normpath-based) - it never touches
	the filesystem or cwd, and never fails; a root/drive mismatch just
	compares unequal and returns False. This matches real
	pathlib.PurePath.is_relative_to()'s own actual behavior, and is meant as
	the primitive a future pathlib module can delegate to. commonpath/
	relpath mirror CPython's own functions (which raise ValueError on
	mismatched absolute/relative input or cross-drive Windows paths) via
	Result[str, ValueError], per this codebase's Result convention.
'''

import compiler
import sys

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
	c1: str = p.__getitem__( 1 ).unwrap_or( '' )
	c2: str = p.__getitem__( 2 ).unwrap_or( '' )
	return c1 == ':' and ( c2 == '\\' or c2 == '/' )


@compiler.target( os = not 'windows' )
def _is_abs( p: str ) -> bool:
	return p.startswith( '/' )


if compiler.target.os == 'windows':
	sep: str = '\\'
else:
	sep: str = '/'


@compiler.target( os = 'windows' )
def _splitroot( p: str ) -> tuple[str, str]:
	# same root-detection shape as normpath's own inline block below,
	# factored out for reuse by dirname/commonpath/relpath/is_relative_to.
	# Used on both raw paths (dirname) and already-normpath'd ones (every
	# other caller, via _path_parts) - so unlike normpath's own inline
	# block, this recognizes forward-slash UNC/drive separators too,
	# matching _is_abs above.
	if p.startswith( '\\\\' ):
		return ( '\\\\', p.removeprefix( '\\\\' ))
	if p.startswith( '//' ):
		return ( '//', p.removeprefix( '//' ))
	if len( p ) >= 2 and p.__getitem__( 1 ).unwrap_or( '' ) == ':':
		drive: str = p.__getitem__( 0 ).unwrap_or( '' )
		return ( drive + ':\\', p.removeprefix( drive + ':' ).removeprefix( '\\' ).removeprefix( '/' ))
	return ( '', p )


@compiler.target( os = not 'windows' )
def _splitroot( p: str ) -> tuple[str, str]:
	if p.startswith( '/' ):
		return ( '/', p.removeprefix( '/' ))
	return ( '', p )


def _path_parts( p: str ) -> tuple[str, list[str]]:
	''' root + ordered list of normalized path segments, built entirely on
	normpath - lexical only, never touches the filesystem/cwd. e.g.
	"a/b/../c" -> ("", ["a","c"]); "C:\\a\\b" -> ("C:\\", ["a","b"]). '''
	normalized: str = path.normpath( p )
	root, rest = _splitroot( normalized )
	if rest == '' or rest == '.':
		return ( root, list[str]() )
	return ( root, rest.split( sep ))


class path:

	@staticmethod
	def join( a: str, b: str ) -> str:
		if a == '':
			return b
		if a.endswith( '/' ) or a.endswith( '\\' ):
			return a + b
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
			norm: str = p.replace( '/', '\\' )
		else:
			norm: str = p

		root: str = ''
		rest: str = norm
		if compiler.target.os == 'windows':
			if norm.startswith( '\\\\' ):
				# UNC - keep "\\server\share" intact, never collapse '..'
				# past the share root (matches CPython's own ntpath)
				root = '\\\\'
				rest = norm.removeprefix( '\\\\' )
			elif len( norm ) >= 2 and norm.__getitem__( 1 ).unwrap_or( '' ) == ':':
				drive: str = norm.__getitem__( 0 ).unwrap_or( '' )
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

	@staticmethod
	def basename( p: str ) -> str:
		# same double-rpartition shape as splitext's own basename step above
		_, _, after_slash = p.rpartition( '/' )
		_, backslash_sep, after_backslash = after_slash.rpartition( '\\' )
		return after_backslash if backslash_sep != '' else after_slash

	@staticmethod
	def dirname( p: str ) -> str:
		root, rest = _splitroot( p )
		before_slash, slash_sep, after_slash = rest.rpartition( '/' )
		before_backslash, backslash_sep, after_backslash = after_slash.rpartition( '\\' )
		if backslash_sep != '':
			head: str = before_slash + slash_sep + before_backslash + backslash_sep
		else:
			head = before_slash + slash_sep
		if head == '':
			return root
		if head.strip( '/\\' ) == '':
			# head is nothing but separator characters (POSIX '/' root, or
			# the separator right after a drive/UNC root) - keep it whole,
			# matching CPython's own "don't rstrip a bare root to nothing"
			return root + head
		return root + head.rstrip( '/\\' )

	@staticmethod
	def isabs( p: str ) -> bool:
		return _is_abs( p )

	@staticmethod
	def commonpath( paths: list[str] ) -> Result[str, ValueError]:
		if len( paths ) == 0:
			return Result.Err( ValueError() )

		first: str = paths.__getitem__( 0 ).unwrap( 'os.path.commonpath: index in bounds by construction' )
		common_root, common_parts = _path_parts( first )

		idx: usize = 1
		with compiler.panic_arithmetic( 'bounded by len(paths), cannot overflow' ):
			while idx < len( paths ):
				p: str = paths.__getitem__( idx ).unwrap( 'os.path.commonpath: index in bounds by construction' )
				idx += 1
				root, parts = _path_parts( p )
				if root != common_root:
					# covers both "mixed absolute/relative" (one root is ''
					# and the other isn't) and a differing Windows drive/
					# UNC share - CPython raises ValueError for both; one
					# check here covers both too
					return Result.Err( ValueError() )

				limit: usize = len( common_parts ) if len( common_parts ) < len( parts ) else len( parts )
				j: usize = 0
				while j < limit:
					a: str = common_parts.__getitem__( j ).unwrap( 'os.path.commonpath: index in bounds by construction' )
					b: str = parts.__getitem__( j ).unwrap( 'os.path.commonpath: index in bounds by construction' )
					if a != b:
						break
					j += 1

				new_common: list[str] = list[str]()
				k: usize = 0
				while k < j:
					new_common.append( common_parts.__getitem__( k ).unwrap( 'os.path.commonpath: index in bounds by construction' )).unwrap( 'os.path.commonpath: append failed' )
					k += 1
				common_parts = new_common

		return Result.Ok( common_root + sep.join( common_parts ))

	@staticmethod
	def relpath( target: str, start: str = '.' ) -> Result[str, ValueError]:
		# unlike is_relative_to, relpath legitimately resolves against cwd
		# (matches real CPython os.path.relpath's own behavior)
		start_root, start_parts = _path_parts( path.abspath( start ))
		target_root, target_parts = _path_parts( path.abspath( target ))
		if start_root != target_root:
			return Result.Err( ValueError() )  # e.g. cross-drive on Windows

		limit: usize = len( start_parts ) if len( start_parts ) < len( target_parts ) else len( target_parts )
		i: usize = 0
		with compiler.panic_arithmetic( 'bounded by limit, cannot overflow' ):
			while i < limit:
				a: str = start_parts.__getitem__( i ).unwrap( 'os.path.relpath: index in bounds by construction' )
				b: str = target_parts.__getitem__( i ).unwrap( 'os.path.relpath: index in bounds by construction' )
				if a != b:
					break
				i += 1

		rel_parts: list[str] = list[str]()
		with compiler.panic_arithmetic( 'bounded by i <= len(start_parts), cannot underflow or overflow' ):
			up_count: usize = len( start_parts ) - i
			k: usize = 0
			while k < up_count:
				rel_parts.append( '..' ).unwrap( 'os.path.relpath: append failed' )
				k += 1
			j: usize = i
			while j < len( target_parts ):
				rel_parts.append( target_parts.__getitem__( j ).unwrap( 'os.path.relpath: index in bounds by construction' )).unwrap( 'os.path.relpath: append failed' )
				j += 1

		if len( rel_parts ) == 0:
			return Result.Ok( '.' )
		return Result.Ok( sep.join( rel_parts ))

	@staticmethod
	def is_relative_to( target: str, start: str ) -> bool:
		# purely lexical (normpath-based, via _path_parts) - never touches
		# the filesystem/cwd, never raises; a root/drive mismatch just
		# compares unequal and returns False. Matches real
		# pathlib.PurePath.is_relative_to()'s own actual behavior.
		target_root, target_parts = _path_parts( target )
		start_root, start_parts = _path_parts( start )
		if target_root != start_root:
			return False
		if len( start_parts ) > len( target_parts ):
			return False
		i: usize = 0
		with compiler.panic_arithmetic( 'bounded by len(start_parts) <= len(target_parts), cannot overflow' ):
			while i < len( start_parts ):
				a: str = start_parts.__getitem__( i ).unwrap( 'os.path.is_relative_to: index in bounds by construction' )
				b: str = target_parts.__getitem__( i ).unwrap( 'os.path.is_relative_to: index in bounds by construction' )
				if a != b:
					return False
				i += 1
		return True
