'''
File I/O handle types and the File namespace.

BinaryReader      — read-only binary file handle
BinaryWriter      — write-only binary file handle
BinaryReadWriter  — read+write binary file handle
File              — namespace with static factory methods
'''

import compiler
from fs import (
	FD, INVALID_FD, SEEK_END,
	close_raw, open_raw, read_raw, seek_raw, truncate_raw, write_raw,
)
from io import Reader, Writer, Seekable


# FileOpsInterface - injection point for reactor-aware file I/O (lib/
# asyncfile.py) - a small, deliberately dependency-free ABSTRACT interface
# (no reactor/socket/threading import - just two method shapes), so
# BinaryReader/BinaryWriter/BinaryReadWriter's own read()/write() never
# need to know reactor.py exists at all. Only whoever constructs a
# reactor-aware instance (lib/asyncfile.py's AsyncFile.binary_reader() et
# al.) needs to import the module that actually implements this and
# inject it - every OTHER program using plain File.binary_reader() etc.
# never pulls in the reactor/socket/threading dependency graph at all.
# read()/write() take the raw fd directly (not a BinaryReader/Writer
# instance) so the implementer stays equally decoupled from this module.
class FileOpsInterface:
	@abstractmethod
	def do_read( self, fd: FD, buf: Ptr[u8], count: usize ) -> Result[usize, OSError]:
		...
	@abstractmethod
	def do_write( self, fd: FD, buf: ConstPtr[u8], count: usize ) -> Result[usize, OSError]:
		...


# _SyncFileOps - the default FileOpsInterface every BinaryReader/BinaryWriter/
# BinaryReadWriter starts with: a null-object implementation that just
# calls read_raw()/write_raw() directly, so read()/write() below can be a
# single unconditional virtual-dispatch call with NO `is not None` branch
# at all
class _SyncFileOps( FileOpsInterface ):
	@virtual
	def do_read( self, fd: FD, buf: Ptr[u8], count: usize ) -> Result[usize, OSError]:
		return read_raw( fd, buf, count )
	@virtual
	def do_write( self, fd: FD, buf: ConstPtr[u8], count: usize ) -> Result[usize, OSError]:
		return write_raw( fd, buf, count )

_sync_ops: _SyncFileOps = _SyncFileOps()

# Pull platform constants into scope.  The module-level if/else is folded
# by compile_time_transformer, so each target sees exactly one set.
if compiler.target.os == 'windows':
	from fs import (
		GENERIC_READ, GENERIC_WRITE,
		CREATE_NEW, CREATE_ALWAYS, OPEN_EXISTING, OPEN_ALWAYS, TRUNCATE_EXISTING,
	)
else:
	from fs import (
		O_RDONLY, O_WRONLY, O_RDWR,
		O_CREAT, O_EXCL, O_TRUNC, O_APPEND,
	)


# ---------------------------------------------------------------------------
# BinaryReader — read-only binary file handle
# ---------------------------------------------------------------------------

class BinaryReader( Reader, Seekable ):
	__fd: FD
	__ops: FileOpsInterface   # defaults to _sync_ops - see its own docstring

	def __del__( self ) -> None:
		if self.__fd != INVALID_FD:
			close_raw( self.__fd ).is_ok() # a destructor can't propagate close() failure - deliberately ignored, not silently unchecked

	def read( self, buf: Ptr[u8], count: usize ) -> Result[usize, OSError]:
		return self.__ops.do_read( self.__fd, buf, count )

	def seek( self, offset: i64, whence: i32 ) -> Result[i64, OSError]:
		return seek_raw( self.__fd, offset, whence )

	def close( self ) -> None:
		if self.__fd != INVALID_FD:
			close_raw( self.__fd ).is_ok() # see __del__'s own comment
			self.__fd = INVALID_FD

	def fileno( self ) -> FD:
		return self.__fd

	def _set_ops( self, aio: FileOpsInterface ) -> None:
		self.__ops = aio

	@private
	@staticmethod
	def _from_fd( fd: FD ) -> BinaryReader:
		return BinaryReader.__allocate__( __fd = fd, __ops = _sync_ops )


# ---------------------------------------------------------------------------
# BinaryWriter — write-only binary file handle
# ---------------------------------------------------------------------------

class BinaryWriter( Writer, Seekable ):
	__fd: FD
	__ops: FileOpsInterface   # defaults to _sync_ops - see its own docstring

	def __del__( self ) -> None:
		if self.__fd != INVALID_FD:
			close_raw( self.__fd ).is_ok() # see BinaryReader.__del__'s own comment

	def write( self, buf: ConstPtr[u8], count: usize ) -> Result[usize, OSError]:
		return self.__ops.do_write( self.__fd, buf, count )

	def seek( self, offset: i64, whence: i32 ) -> Result[i64, OSError]:
		return seek_raw( self.__fd, offset, whence )

	def close( self ) -> None:
		if self.__fd != INVALID_FD:
			close_raw( self.__fd ).is_ok() # see BinaryReader.__del__'s own comment
			self.__fd = INVALID_FD

	def fileno( self ) -> FD:
		return self.__fd

	def _set_ops( self, aio: FileOpsInterface ) -> None:
		self.__ops = aio

	@private
	@staticmethod
	def _from_fd( fd: FD ) -> BinaryWriter:
		return BinaryWriter.__allocate__( __fd = fd, __ops = _sync_ops )


# ---------------------------------------------------------------------------
# BinaryReadWriter — read+write binary file handle
# ---------------------------------------------------------------------------

class BinaryReadWriter( Reader, Writer, Seekable ):
	__fd: FD
	__ops: FileOpsInterface   # defaults to _sync_ops - see its own docstring

	def __del__( self ) -> None:
		if self.__fd != INVALID_FD:
			close_raw( self.__fd ).is_ok() # see BinaryReader.__del__'s own comment

	def read( self, buf: Ptr[u8], count: usize ) -> Result[usize, OSError]:
		return self.__ops.do_read( self.__fd, buf, count )

	def write( self, buf: ConstPtr[u8], count: usize ) -> Result[usize, OSError]:
		return self.__ops.do_write( self.__fd, buf, count )

	def seek( self, offset: i64, whence: i32 ) -> Result[i64, OSError]:
		return seek_raw( self.__fd, offset, whence )

	def close( self ) -> None:
		if self.__fd != INVALID_FD:
			close_raw( self.__fd ).is_ok() # see BinaryReader.__del__'s own comment
			self.__fd = INVALID_FD

	def fileno( self ) -> FD:
		return self.__fd

	def _set_ops( self, aio: FileOpsInterface ) -> None:
		self.__ops = aio

	@private
	@staticmethod
	def _from_fd( fd: FD ) -> BinaryReadWriter:
		return BinaryReadWriter.__allocate__( __fd = fd, __ops = _sync_ops )


# ---------------------------------------------------------------------------
# File — namespace with static factory methods
# ---------------------------------------------------------------------------

class File:

	# ---- binary_reader ---------------------------------------------------

	@compiler.target( os = 'windows' )
	@staticmethod
	def binary_reader( path: str ) -> Result[BinaryReader, OSError]:
		fd: FD = open_raw( path.get_cstr(), GENERIC_READ, OPEN_EXISTING ).or_return()
		return Result.Ok( BinaryReader._from_fd( fd ))

	@compiler.target( os = not 'windows' )
	@staticmethod
	def binary_reader( path: str ) -> Result[BinaryReader, OSError]:
		fd: FD = open_raw( path.get_cstr(), O_RDONLY, 0 ).or_return()
		return Result.Ok( BinaryReader._from_fd( fd ))

	# ---- binary_writer ---------------------------------------------------

	@compiler.target( os = 'windows' )
	@staticmethod
	def binary_writer(
		path: str,
		append: bool = False,
		truncate: bool = True,
		exists: bool|None = None,
	) -> Result[BinaryWriter, OSError]:
		access: u32 = GENERIC_WRITE
		creation: u32
		if exists is None:
			if truncate:
				creation = CREATE_ALWAYS
			else:
				creation = OPEN_ALWAYS
		elif exists:
			if truncate:
				creation = TRUNCATE_EXISTING
			else:
				creation = OPEN_EXISTING
		else:
			creation = CREATE_NEW
		fd: FD = open_raw( path.get_cstr(), access, creation ).or_return()
		if append:
			seek_raw( fd, 0, SEEK_END ).or_return()
		return Result.Ok( BinaryWriter._from_fd( fd ))

	@compiler.target( os = not 'windows' )
	@staticmethod
	def binary_writer(
		path: str,
		append: bool = False,
		truncate: bool = True,
		exists: bool|None = None,
	) -> Result[BinaryWriter, OSError]:
		flags: i32 = O_WRONLY
		if exists is None:
			flags |= O_CREAT
			if truncate:
				flags |= O_TRUNC
		elif exists:
			if truncate:
				flags |= O_TRUNC
		else:
			flags |= O_CREAT | O_EXCL
		if append:
			flags |= O_APPEND
		fd: FD = open_raw( path.get_cstr(), flags, 0o644 ).or_return()
		return Result.Ok( BinaryWriter._from_fd( fd ))

	# ---- binary_read_writer ----------------------------------------------

	@compiler.target( os = 'windows' )
	@staticmethod
	def binary_read_writer(
		path: str,
		append: bool = False,
		truncate: bool = True,
		exists: bool|None = None,
	) -> Result[BinaryReadWriter, OSError]:
		access: u32 = GENERIC_READ | GENERIC_WRITE
		creation: u32
		if exists is None:
			if truncate:
				creation = CREATE_ALWAYS
			else:
				creation = OPEN_ALWAYS
		elif exists:
			if truncate:
				creation = TRUNCATE_EXISTING
			else:
				creation = OPEN_EXISTING
		else:
			creation = CREATE_NEW
		fd: FD = open_raw( path.get_cstr(), access, creation ).or_return()
		if append:
			seek_raw( fd, 0, SEEK_END ).or_return()
		return Result.Ok( BinaryReadWriter._from_fd( fd ))

	@compiler.target( os = not 'windows' )
	@staticmethod
	def binary_read_writer(
		path: str,
		append: bool = False,
		truncate: bool = True,
		exists: bool|None = None,
	) -> Result[BinaryReadWriter, OSError]:
		flags: i32 = O_RDWR
		if exists is None:
			flags |= O_CREAT
			if truncate:
				flags |= O_TRUNC
		elif exists:
			if truncate:
				flags |= O_TRUNC
		else:
			flags |= O_CREAT | O_EXCL
		if append:
			flags |= O_APPEND
		fd: FD = open_raw( path.get_cstr(), flags, 0o644 ).or_return()
		return Result.Ok( BinaryReadWriter._from_fd( fd ))
