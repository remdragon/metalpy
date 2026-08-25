'''
pathlib — a pathlib.Path-inspired path class.

A single Path class (not a PurePosixPath/PureWindowsPath/PosixPath/WindowsPath
hierarchy): path *flavor* (separator/case rules) is a runtime field rather
than tied to the compile target, so a build can still construct and
manipulate foreign-syntax paths - e.g. POSIX-style paths for a remote SSH/FTP
target while compiled for Windows - via Path.posix()/Path.windows(). Plain
Path(...) uses the native flavor. Filesystem-touching methods (exists/mkdir/
read_text/...) require the path's own flavor to match the native platform;
called on a foreign-flavor Path they fail immediately (Result.Err(OSError.
Invalid), or False for the bool-shaped exists/is_file/is_dir) rather than
handing a wrong-syntax string to a native OS call.

Purely lexical parsing: '.' segments are dropped, '..' segments are kept
literally (matching real PurePath - collapsing '..' requires knowing whether
an intermediate component is a symlink, which needs the filesystem). No
glob/rglob, no symlink-specific ops, no full stat metadata, no chmod/
permissions - see PLAN_PATHLIB.md for the scope this first pass covers.
'''

import compiler
import os
import sys


@enum( u8 )
class PathFlavor:
	Posix = 0
	Windows = 1


if compiler.target.os == 'windows':
	_NATIVE_FLAVOR: PathFlavor = PathFlavor.Windows
else:
	_NATIVE_FLAVOR: PathFlavor = PathFlavor.Posix


# ---------------------------------------------------------------------------
# Private, flavor-parameterized parsing helpers
# ---------------------------------------------------------------------------

def _native_sep( flavor: PathFlavor ) -> str:
	if flavor == PathFlavor.Windows:
		return '\\'
	return '/'


def _split_anchor( raw: str, flavor: PathFlavor ) -> tuple[str, str]:
	''' (anchor, rest) - raw is assumed already separator-normalized (only
	the flavor's own native separator remains) by the time this is called;
	rest never has a leading separator. '''
	if flavor == PathFlavor.Windows:
		if raw.startswith( '\\\\' ):
			# UNC ("\\server\share\...") - simplified: the anchor is just
			# the leading "\\", server/share fall through as ordinary parts
			# rather than being folded into one anchor unit the way real
			# PureWindowsPath does (a documented v1 simplification).
			return ( '\\\\', raw.removeprefix( '\\\\' ))
		if len( raw ) >= 2 and raw.__getitem__( 1 ).unwrap_or( '' ) == ':':
			drive: str = raw.__getitem__( 0 ).unwrap_or( '' )
			rest: str = raw.removeprefix( drive + ':' )
			if rest.startswith( '\\' ):
				return ( drive + ':\\', rest.removeprefix( '\\' ))
			# drive-relative ("C:foo") - has a drive but no root, so it's
			# NOT absolute (matches real ntpath.isabs's own surprising rule)
			return ( drive + ':', rest )
		return ( '', raw )
	else:
		if raw.startswith( '/' ):
			return ( '/', raw.removeprefix( '/' ))
		return ( '', raw )


def _split_parts( rest: str, sep: str ) -> list[str]:
	''' rest split on sep, dropping empty segments (collapses doubled
	separators) and literal '.' segments; '..' is kept. '''
	raw_parts: list[str] = rest.split( sep )
	parts: list[str] = list[str]()
	i: usize = 0
	with compiler.panic_arithmetic( 'bounded by len(raw_parts), cannot overflow' ):
		while i < len( raw_parts ):
			part: str = raw_parts.__getitem__( i ).unwrap( 'pathlib: index in bounds by construction' )
			i += 1
			if part == '' or part == '.':
				continue
			parts.append( part )
	return parts


def _normalize( raw: str, flavor: PathFlavor ) -> str:
	pre: str = raw.replace( '/', '\\' ) if flavor == PathFlavor.Windows else raw
	anchor, rest = _split_anchor( pre, flavor )
	sep: str = _native_sep( flavor )
	parts: list[str] = _split_parts( rest, sep )
	joined: str = sep.join( parts )
	if anchor != '':
		return anchor + joined
	if joined == '':
		return '.'
	return joined


# ---------------------------------------------------------------------------
# Path
# ---------------------------------------------------------------------------

class Path:
	_raw: str
	_flavor: PathFlavor

	def __init__( self, raw: str = '.', flavor: PathFlavor = _NATIVE_FLAVOR ) -> None:
		self._flavor = flavor
		self._raw = _normalize( raw, flavor )

	@staticmethod
	def posix( raw: str ) -> Path:
		return Path( raw, PathFlavor.Posix )

	@staticmethod
	def windows( raw: str ) -> Path:
		return Path( raw, PathFlavor.Windows )

	@private
	@staticmethod
	def _from_raw( raw: str, flavor: PathFlavor ) -> Path:
		''' raw must already be normalized - used internally once a result
		string has already been built via _normalize/str concatenation of
		already-normalized pieces. '''
		return Path.__allocate__( _raw = raw, _flavor = flavor )

	# ---- pure parsing (never touches the filesystem) ---------------------

	@property
	def parts( self ) -> list[str]:
		anchor, rest = _split_anchor( self._raw, self._flavor )
		sep: str = _native_sep( self._flavor )
		segments: list[str] = _split_parts( rest, sep )
		result: list[str] = list[str]()
		if anchor != '':
			result.append( anchor )
		i: usize = 0
		with compiler.panic_arithmetic( 'bounded by len(segments), cannot overflow' ):
			while i < len( segments ):
				result.append( segments.__getitem__( i ).unwrap( 'pathlib: index in bounds by construction' ))
				i += 1
		return result

	@property
	def anchor( self ) -> str:
		a, _ = _split_anchor( self._raw, self._flavor )
		return a

	@property
	def name( self ) -> str:
		_, rest = _split_anchor( self._raw, self._flavor )
		if rest == '' or rest == '.':
			return ''
		sep: str = _native_sep( self._flavor )
		_, _, last = rest.rpartition( sep )
		return last

	@property
	def suffix( self ) -> str:
		n: str = self.name
		before_dot, dot_sep, after_dot = n.rpartition( '.' )
		if dot_sep == '' or before_dot.strip( '.' ) == '':
			# no dot at all, or the name is nothing but leading dots
			# ('.bashrc', '..') - Python requires a non-dot byte before the
			# extension's own dot for it to count as a real suffix
			return ''
		return '.' + after_dot

	@property
	def stem( self ) -> str:
		n: str = self.name
		suf: str = self.suffix
		return n.removesuffix( suf ) if suf != '' else n

	@property
	def parent( self ) -> Path:
		anchor, rest = _split_anchor( self._raw, self._flavor )
		if rest == '' or rest == '.':
			return self   # root's (and '.'s) parent is itself
		sep: str = _native_sep( self._flavor )
		head, sep_found, _ = rest.rpartition( sep )
		if sep_found == '':
			if anchor != '':
				return Path._from_raw( anchor, self._flavor )
			return Path._from_raw( '.', self._flavor )
		return Path._from_raw( anchor + head, self._flavor )

	@property
	def parents( self ) -> list[Path]:
		''' eager (materializes real pathlib's lazy Path.parents sequence),
		nearest ancestor first. Terminates because each step's rest is
		strictly shorter, until parent() reaches its own fixed point
		('.' or an anchor-only root). '''
		result: list[Path] = list[Path]()
		current: Path = self
		while True:
			p: Path = current.parent
			if p._raw == current._raw:
				break
			result.append( p )
			current = p
		return result

	def is_absolute( self ) -> bool:
		anchor, _ = _split_anchor( self._raw, self._flavor )
		if self._flavor == PathFlavor.Windows:
			return anchor.endswith( '\\' )
		return anchor == '/'

	def joinpath( self, other: str ) -> Path:
		other_path: Path = Path( other, self._flavor )
		if other_path.is_absolute():
			return other_path
		if self._raw == '.':
			combined: str = other_path._raw
		else:
			combined = self._raw + _native_sep( self._flavor ) + other_path._raw
		return Path._from_raw( _normalize( combined, self._flavor ), self._flavor )

	def joinpath( self, other: Path ) -> Path:
		return self.joinpath( other._raw )

	def __truediv__( self, other: str ) -> Path:
		return self.joinpath( other )

	def __truediv__( self, other: Path ) -> Path:
		return self.joinpath( other._raw )

	def with_name( self, name: str ) -> Path:
		anchor, rest = _split_anchor( self._raw, self._flavor )
		if rest == '' or rest == '.':
			sys.panic( 'Path.with_name: path has no name component to replace' )
		sep: str = _native_sep( self._flavor )
		head, sep_found, _ = rest.rpartition( sep )
		new_raw: str
		if sep_found == '':
			new_raw = anchor + name if anchor != '' else name
		else:
			new_raw = anchor + head + sep + name
		return Path._from_raw( _normalize( new_raw, self._flavor ), self._flavor )

	def with_suffix( self, suffix: str ) -> Path:
		if suffix != '' and not suffix.startswith( '.' ):
			sys.panic( 'Path.with_suffix: suffix must be empty or start with a dot' )
		return self.with_name( self.stem + suffix )

	def __str__( self ) -> str:
		return self._raw

	def __repr__( self ) -> str:
		return f"Path({self._raw!r})"

	def __eq__( self, other: Path ) -> bool:
		if self._flavor != other._flavor:
			return False
		if self._flavor == PathFlavor.Windows:
			return self._raw.lower() == other._raw.lower()
		return self._raw == other._raw

	def __ne__( self, other: Path ) -> bool:
		return not self.__eq__( other )

	def __hash__( self ) -> u64:
		if self._flavor == PathFlavor.Windows:
			return self._raw.lower().__hash__()
		return self._raw.__hash__()

	# ---- concrete filesystem ops (native flavor only) ---------------------

	def exists( self ) -> bool:
		if self._flavor != _NATIVE_FLAVOR:
			return False
		return os.path.exists( self._raw )

	def is_file( self ) -> bool:
		if self._flavor != _NATIVE_FLAVOR:
			return False
		return os.path.isfile( self._raw )

	def is_dir( self ) -> bool:
		if self._flavor != _NATIVE_FLAVOR:
			return False
		return os.path.isdir( self._raw )

	def mkdir( self, parents: bool = False, exist_ok: bool = False ) -> Result[None, OSError]:
		if self._flavor != _NATIVE_FLAVOR:
			return Result.Err( OSError.Invalid )
		if parents:
			parent: Path = self.parent
			if parent._raw != self._raw and not parent.exists():
				parent.mkdir( parents = True, exist_ok = True ).or_return()
		match os.mkdir( self._raw ):
			case Result.Ok( _ ):
				return Result.Ok( None )
			case Result.Err( e ):
				if exist_ok and e == OSError.AlreadyExists and self.is_dir():
					return Result.Ok( None )
				return Result.Err( e )

	def rmdir( self ) -> Result[None, OSError]:
		if self._flavor != _NATIVE_FLAVOR:
			return Result.Err( OSError.Invalid )
		return os.rmdir( self._raw )

	def unlink( self, missing_ok: bool = False ) -> Result[None, OSError]:
		if self._flavor != _NATIVE_FLAVOR:
			return Result.Err( OSError.Invalid )
		match os.unlink( self._raw ):
			case Result.Ok( _ ):
				return Result.Ok( None )
			case Result.Err( e ):
				if missing_ok and e == OSError.FileNotFoundError:
					return Result.Ok( None )
				return Result.Err( e )

	def rename( self, target: str ) -> Result[Path, OSError]:
		if self._flavor != _NATIVE_FLAVOR:
			return Result.Err( OSError.Invalid )
		os.rename( self._raw, target ).or_return()
		return Result.Ok( Path( target, self._flavor ))

	def iterdir( self ) -> Result[list[Path], OSError]:
		if self._flavor != _NATIVE_FLAVOR:
			return Result.Err( OSError.Invalid )
		names: list[str] = os.listdir( self._raw ).or_return()
		result: list[Path] = list[Path]()
		i: usize = 0
		with compiler.panic_arithmetic( 'bounded by len(names), cannot overflow' ):
			while i < len( names ):
				name: str = names.__getitem__( i ).unwrap( 'pathlib: index in bounds by construction' )
				result.append( self / name )
				i += 1
		return Result.Ok( result )

	@staticmethod
	def cwd() -> Result[Path, OSError]:
		raw: str = os.getcwd().or_return()
		return Result.Ok( Path( raw, _NATIVE_FLAVOR ))

	@staticmethod
	def home() -> Result[Path, OSError]:
		var_name: str = 'USERPROFILE' if compiler.target.os == 'windows' else 'HOME'
		match os.getenv( var_name ):
			case None:
				# real pathlib raises RuntimeError here (not an OSError at
				# all) - OSError.Other is a forced fit, a consequence of
				# reusing OSError everywhere rather than inventing a new
				# error type just for this one case.
				return Result.Err( OSError.Other )
			case home_str:
				return Result.Ok( Path( home_str, _NATIVE_FLAVOR ))

	def read_bytes( self ) -> Result[bytearray, OSError]:
		if self._flavor != _NATIVE_FLAVOR:
			return Result.Err( OSError.Invalid )
		from fs import SEEK_END, SEEK_SET
		reader: BinaryReader = File.binary_reader( self._raw ).or_return()
		defer( reader.close() )
		size: i64 = reader.seek( 0, SEEK_END ).or_return()
		reader.seek( 0, SEEK_SET ).or_return()
		with compiler.wrap_arithmetic:
			total: usize = usize( size )
		buf: bytearray = bytearray( total )
		got: usize = 0
		while got < total:
			with compiler.wrap_arithmetic:
				remaining: usize = total - got
			with compiler.panic_arithmetic( 'bounded by got < total <= buf\'s own allocated size, cannot overflow' ):
				dest: Ptr[u8] = buf.get_ptr() + got
			n: usize = reader.read( dest, remaining ).or_return()
			if n == 0:
				break   # file shrank concurrently - stop rather than loop forever
			with compiler.wrap_arithmetic:
				got += n
		return Result.Ok( buf )

	def write_bytes( self, data: bytearray ) -> Result[None, OSError]:
		if self._flavor != _NATIVE_FLAVOR:
			return Result.Err( OSError.Invalid )
		from fs import write_all
		writer: BinaryWriter = File.binary_writer( self._raw, truncate = True ).or_return()
		defer( writer.close() )
		write_all( writer.fileno(), data.get_const_ptr(), len( data )).or_return()
		return Result.Ok( None )

	def read_text( self, codec: Codec = utf8 ) -> Result[str, OSError|CodecError]:
		if self._flavor != _NATIVE_FLAVOR:
			invalid: OSError|CodecError = OSError( OSError.Invalid )
			return Result.Err( invalid )
		raw: bytearray = self.read_bytes().or_return()
		return raw.decode( codec )

	def write_text( self, data: str, codec: Codec = utf8 ) -> Result[None, OSError|CodecError]:
		if self._flavor != _NATIVE_FLAVOR:
			invalid: OSError|CodecError = OSError( OSError.Invalid )
			return Result.Err( invalid )
		encoded: bytes = data.encode( codec ).or_return()
		from fs import write_all
		writer: BinaryWriter = File.binary_writer( self._raw, truncate = True ).or_return()
		defer( writer.close() )
		write_all( writer.fileno(), encoded.get_const_ptr(), len( encoded )).or_return()
		return Result.Ok( None )
