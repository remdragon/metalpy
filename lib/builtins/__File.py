'''
File I/O handle types and the File namespace.

BinaryReader      — read-only binary file handle
BinaryWriter      — write-only binary file handle
BinaryReadWriter  — read+write binary file handle
File              — namespace with static factory methods
'''

import compiler
from fs import (
	FD, INVALID_FD,
	close_raw, open_raw, read_raw, seek_raw, truncate_raw, write_raw,
)

# Pull platform constants into scope.  The module-level if/else is folded
# by compile_time_transformer, so each target sees exactly one set.
if compiler.target.os == 'windows':
	from fs import (
		GENERIC_READ, GENERIC_WRITE,
		CREATE_ALWAYS, OPEN_ALWAYS, OPEN_EXISTING,
		FILE_END,
	)
else:
	from fs import (
		O_RDONLY, O_WRONLY, O_RDWR,
		O_CREAT, O_TRUNC, O_APPEND,
	)


# ---------------------------------------------------------------------------
# BinaryReader — read-only binary file handle
# ---------------------------------------------------------------------------

class BinaryReader:
	__fd: FD

	def __del__( self ) -> None:
		if self.__fd != INVALID_FD:
			close_raw( self.__fd )

	def read( self, buf: Ptr[u8], count: usize ) -> Result[usize, OSError]:
		return read_raw( self.__fd, buf, count )

	def close( self ) -> None:
		if self.__fd != INVALID_FD:
			close_raw( self.__fd )
			self.__fd = INVALID_FD

	def fd( self ) -> FD:
		return self.__fd

	@private
	@staticmethod
	def _from_fd( fd: FD ) -> BinaryReader:
		return BinaryReader.__allocate__( __fd = fd )


# ---------------------------------------------------------------------------
# BinaryWriter — write-only binary file handle
# ---------------------------------------------------------------------------

class BinaryWriter:
	__fd: FD

	def __del__( self ) -> None:
		if self.__fd != INVALID_FD:
			close_raw( self.__fd )

	def write( self, buf: ConstPtr[u8], count: usize ) -> Result[usize, OSError]:
		return write_raw( self.__fd, buf, count )

	def close( self ) -> None:
		if self.__fd != INVALID_FD:
			close_raw( self.__fd )
			self.__fd = INVALID_FD

	def fd( self ) -> FD:
		return self.__fd

	@private
	@staticmethod
	def _from_fd( fd: FD ) -> BinaryWriter:
		return BinaryWriter.__allocate__( __fd = fd )


# ---------------------------------------------------------------------------
# BinaryReadWriter — read+write binary file handle
# ---------------------------------------------------------------------------

class BinaryReadWriter:
	__fd: FD

	def __del__( self ) -> None:
		if self.__fd != INVALID_FD:
			close_raw( self.__fd )

	def read( self, buf: Ptr[u8], count: usize ) -> Result[usize, OSError]:
		return read_raw( self.__fd, buf, count )

	def write( self, buf: ConstPtr[u8], count: usize ) -> Result[usize, OSError]:
		return write_raw( self.__fd, buf, count )

	def close( self ) -> None:
		if self.__fd != INVALID_FD:
			close_raw( self.__fd )
			self.__fd = INVALID_FD

	def fd( self ) -> FD:
		return self.__fd

	@private
	@staticmethod
	def _from_fd( fd: FD ) -> BinaryReadWriter:
		return BinaryReadWriter.__allocate__( __fd = fd )


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
	) -> Result[BinaryWriter, OSError]:
		access: u32 = GENERIC_WRITE
		if truncate:
			creation: u32 = CREATE_ALWAYS
		else:
			creation: u32 = OPEN_ALWAYS
		fd: FD = open_raw( path.get_cstr(), access, creation ).or_return()
		if append:
			seek_raw( fd, 0, FILE_END ).or_return()
		return Result.Ok( BinaryWriter._from_fd( fd ))

	@compiler.target( os = not 'windows' )
	@staticmethod
	def binary_writer(
		path: str,
		append: bool = False,
		truncate: bool = True,
	) -> Result[BinaryWriter, OSError]:
		flags: i32 = O_WRONLY | O_CREAT
		if truncate:
			flags |= O_TRUNC
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
	) -> Result[BinaryReadWriter, OSError]:
		access: u32 = GENERIC_READ | GENERIC_WRITE
		if truncate:
			creation: u32 = CREATE_ALWAYS
		else:
			creation: u32 = OPEN_ALWAYS
		fd: FD = open_raw( path.get_cstr(), access, creation ).or_return()
		if append:
			seek_raw( fd, 0, FILE_END ).or_return()
		return Result.Ok( BinaryReadWriter._from_fd( fd ))

	@compiler.target( os = not 'windows' )
	@staticmethod
	def binary_read_writer(
		path: str,
		append: bool = False,
		truncate: bool = True,
	) -> Result[BinaryReadWriter, OSError]:
		flags: i32 = O_RDWR | O_CREAT
		if truncate:
			flags |= O_TRUNC
		if append:
			flags |= O_APPEND
		fd: FD = open_raw( path.get_cstr(), flags, 0o644 ).or_return()
		return Result.Ok( BinaryReadWriter._from_fd( fd ))
