'''
mmap — memory-mapped file access.

ACCESS_READ maps the file read-only. ACCESS_WRITE maps it read-write, and
writes go back to the underlying file (MAP_SHARED / FILE_MAP_WRITE).
ACCESS_COPY maps it read-write, but writes stay private to this mapping,
never reaching the file (MAP_PRIVATE / FILE_MAP_COPY) - Python's own
copy-on-write semantics.

mmap.mmap(fileno, length, access=ACCESS_READ) is a FALLIBLE constructor
(SYNTAX.md's "Fallible __init__() Construction" - `mmap(...)` itself
returns Result[mmap, OSError]) rather than raising, unlike real Python's
own mmap.mmap(). length=0 means "map the whole file" (its current size,
looked up via GetFileSizeEx/fstat), matching real Python.

fileno is fs.FD (lib/fs.py) - the same platform-split file-descriptor type
File.binary_reader()'s own BinaryReader.fileno() returns, so
`mmap.mmap(f.fileno(), 0)` lines up directly.
'''

import compiler
from fs import FD

ACCESS_READ:  i32 = 1
ACCESS_WRITE: i32 = 2
ACCESS_COPY:  i32 = 3


class mmap( Sequence[u8], Iterable[u8] ):
	__ptr: Ptr[u8]
	__len: usize

	@compiler.target( os = 'windows' )
	def __init__( self, fileno: FD, length: usize, access: i32 = ACCESS_READ ) -> Result[None, OSError]:
		from windows.kernel32 import (
			CreateFileMappingA, MapViewOfFile, UnmapViewOfFile, CloseHandle, GetFileSizeEx, GetLastError,
			PAGE_READONLY, PAGE_READWRITE, FILE_MAP_READ, FILE_MAP_WRITE, FILE_MAP_COPY,
		)
		real_length: usize = length
		if real_length == 0:
			size: i64 = 0
			if not GetFileSizeEx( fileno, compiler.addrof( size )):
				return Result.Err( OSError( GetLastError() ))
			with compiler.panic_arithmetic( 'file size does not fit in usize' ):
				real_length = usize( size )
		protect: u32
		map_access: u32
		if access == ACCESS_WRITE:
			protect = PAGE_READWRITE
			map_access = FILE_MAP_WRITE
		elif access == ACCESS_COPY:
			protect = PAGE_READWRITE
			map_access = FILE_MAP_COPY
		else:
			protect = PAGE_READONLY
			map_access = FILE_MAP_READ
		mapping: Ptr[None] = CreateFileMappingA( fileno, None, protect, 0, 0, None )
		if mapping is None:
			return Result.Err( OSError( GetLastError() ))
		view: Ptr[None] = MapViewOfFile( mapping, map_access, 0, 0, real_length )
		if view is None:
			err: u32 = GetLastError()
			CloseHandle( mapping )
			return Result.Err( OSError( err ))
		# the VIEW keeps the mapping alive once created (documented Win32
		# behavior - CreateFileMapping's own HANDLE isn't needed past this
		# point), so the handle closes immediately rather than being kept
		# around for close()/__del__ to close later
		CloseHandle( mapping )
		self.__ptr = compiler.cast( Ptr[u8], view )
		self.__len = real_length
		return Result.Ok( None )

	@compiler.target( os = not 'windows' )
	def __init__( self, fileno: FD, length: usize, access: i32 = ACCESS_READ ) -> Result[None, OSError]:
		from posix.mman import mmap as _mmap, MAP_FAILED, PROT_READ, PROT_WRITE, MAP_PRIVATE, MAP_SHARED
		from posix.stat import stat_t, fstat
		from crt import get_errno
		real_length: usize = length
		if real_length == 0:
			buf: Ptr[stat_t] = sys.alloc[stat_t]( 1 )
			if buf is None:
				return Result.Err( OSError( get_errno() ))
			defer( sys.free( compiler.cast( Ptr[None], buf )))
			if fstat( fileno, buf ) != 0:
				err: i32 = get_errno()
				return Result.Err( OSError( err ))
			size: i64 = compiler.c_field( buf, 'st_size', i64 )
			with compiler.panic_arithmetic( 'file size does not fit in usize' ):
				real_length = usize( size )
		prot: i32
		flags: i32
		if access == ACCESS_READ:
			prot = PROT_READ
			flags = MAP_SHARED
		elif access == ACCESS_WRITE:
			prot = PROT_READ | PROT_WRITE
			flags = MAP_SHARED
		else:
			prot = PROT_READ | PROT_WRITE
			flags = MAP_PRIVATE
		result: Ptr[None] = _mmap( None, real_length, prot, flags, fileno, 0 )
		if result == MAP_FAILED:
			return Result.Err( OSError( get_errno() ))
		self.__ptr = compiler.cast( Ptr[u8], result )
		self.__len = real_length
		return Result.Ok( None )

	def __enter__( self ) -> mmap:
		return self

	def __exit__( self ) -> None:
		self.close()

	@compiler.target( os = 'windows' )
	def close( self ) -> None:
		if self.__ptr is not None:
			from windows.kernel32 import UnmapViewOfFile
			UnmapViewOfFile( compiler.cast( ConstPtr[None], self.__ptr ))
			self.__ptr = None

	@compiler.target( os = not 'windows' )
	def close( self ) -> None:
		if self.__ptr is not None:
			from posix.mman import munmap
			munmap( compiler.cast( Ptr[None], self.__ptr ), self.__len )
			self.__ptr = None

	def __del__( self ) -> None:
		self.close()

	def __len__( self ) -> usize:
		return self.__len

	def __getitem__( self, index: usize ) -> Result[u8,IndexError]:
		if index >= self.__len:
			return Result.Err( IndexError() )
		return Result.Ok( self.__ptr[index] )

	def __iter__( self ) -> Generator[u8, StopIteration]:
		return _sequence_iter( self )

	def get_ptr( self ) -> Ptr[u8]:
		return self.__ptr

	def get_const_ptr( self ) -> ConstPtr[u8]:
		return self.__ptr
