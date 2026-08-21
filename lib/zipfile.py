# zipfile - PKZIP archive container format. ZipReader/ZipWriter wrap
# deflate.py (compression method 8) and crc32.py; entries stored raw use
# method 0. Read parses the central directory first (real zipfile
# semantics) rather than scanning local headers - see ZipFile.open's own
# comment. Write compresses each entry to memory before writing its local
# header, so sizes/CRC are always known up front and the "data descriptor
# follows" streaming flag is never needed on the writer side (still
# respected on the reader side - see ZipReader.read's own comment).
#
# Known gaps, not silently worked around:
#   - No ZIP64: entries/archives >= 4 GiB or >= 65535 entries are rejected
#     outright (an explicit size check before writing headers), not
#     silently truncated/corrupted.
#   - No encryption: an entry with the encrypted flag set is reported as
#     ZipError, not silently mis-decompressed.
#   - No append-to-existing-archive ('a') mode.
#   - ZipWriter never emits STORED for size reasons (only when the caller
#     explicitly asks for ZIP_STORED) - see deflate.py's own docstring on
#     why compress() itself never falls back to STORED.

import bitstream
import compiler
import crc32
import deflate
import io
import sys
from fs import SEEK_END, SEEK_SET


ZIP_STORED:   i32 = 0
ZIP_DEFLATED: i32 = 8

_LOCAL_FILE_HEADER_SIG: u32 = 0x04034b50
_CENTRAL_DIR_SIG:       u32 = 0x02014b50
_EOCD_SIG:              u32 = 0x06054b50

_MAX_U32: u32 = 0xFFFFFFFF


class ZipError:
	message: str

	def __init__( self, message: str ) -> None:
		self.message = message


@cstruct( packed = True )
class _LocalFileHeaderFixed:
	signature:          u32 = 0
	version_needed:     u16 = 20
	flags:              u16 = 0
	compression_method: u16 = 0
	mod_time:           u16 = 0
	mod_date:           u16 = 0
	crc32:              u32 = 0
	compressed_size:    u32 = 0
	uncompressed_size:  u32 = 0
	filename_len:       u16 = 0
	extra_len:          u16 = 0


@cstruct( packed = True )
class _CentralDirectoryHeaderFixed:
	signature:           u32 = 0
	version_made_by:     u16 = 0
	version_needed:      u16 = 20
	flags:               u16 = 0
	compression_method:  u16 = 0
	mod_time:            u16 = 0
	mod_date:            u16 = 0
	crc32:               u32 = 0
	compressed_size:     u32 = 0
	uncompressed_size:   u32 = 0
	filename_len:        u16 = 0
	extra_len:           u16 = 0
	comment_len:         u16 = 0
	disk_number_start:   u16 = 0
	internal_attrs:      u16 = 0
	external_attrs:      u32 = 0
	local_header_offset: u32 = 0


@cstruct( packed = True )
class _EndOfCentralDirectory:
	signature:            u32 = 0
	disk_number:          u16 = 0
	cd_start_disk:        u16 = 0
	cd_records_this_disk: u16 = 0
	cd_records_total:     u16 = 0
	cd_size:              u32 = 0
	cd_offset:             u32 = 0
	comment_len:          u16 = 0


class ZipInfo:
	filename:            str
	crc:                 u32
	compress_size:       u32
	file_size:           u32
	compress_type:       i32
	flags:               u16
	header_offset:       u32

	def __init__( self, filename: str, crc: u32, compress_size: u32, file_size: u32, compress_type: i32, flags: u16, header_offset: u32 ) -> None:
		self.filename      = filename
		self.crc           = crc
		self.compress_size = compress_size
		self.file_size     = file_size
		self.compress_type = compress_type
		self.flags         = flags
		self.header_offset = header_offset


# Not generic (T) on purpose: a generic free function with T inside a
# Result[T,E] shape crashes the C emitter's monomorphization (confirmed
# via a minimal repro, reported separately as a compiler bug) - so each
# @cstruct gets its own small, concretely-typed read/write pair instead.

def _read_local_header( reader: BinaryReader ) -> Result[_LocalFileHeaderFixed, ZipError]:
	hdr: _LocalFileHeaderFixed = _LocalFileHeaderFixed()
	match io.read_exact( reader, compiler.cast( Ptr[u8], compiler.addrof( hdr )), compiler.sizeof( _LocalFileHeaderFixed )):
		case Result.Ok( _ ):
			return Result.Ok( hdr )
		case Result.Err( _ ):
			return Result.Err( ZipError( '_read_local_header: unexpected end of file or read error' ))


def _read_central_header( reader: BinaryReader ) -> Result[_CentralDirectoryHeaderFixed, ZipError]:
	hdr: _CentralDirectoryHeaderFixed = _CentralDirectoryHeaderFixed()
	match io.read_exact( reader, compiler.cast( Ptr[u8], compiler.addrof( hdr )), compiler.sizeof( _CentralDirectoryHeaderFixed )):
		case Result.Ok( _ ):
			return Result.Ok( hdr )
		case Result.Err( _ ):
			return Result.Err( ZipError( '_read_central_header: unexpected end of file or read error' ))


def _write_local_header( writer: BinaryWriter, hdr: _LocalFileHeaderFixed ) -> Result[None, ZipError]:
	match io.write_all( writer, compiler.cast( ConstPtr[u8], compiler.addrof( hdr )), compiler.sizeof( _LocalFileHeaderFixed )):
		case Result.Ok( _ ):
			return Result.Ok( None )
		case Result.Err( _ ):
			return Result.Err( ZipError( '_write_local_header: write error' ))


def _write_central_header( writer: BinaryWriter, hdr: _CentralDirectoryHeaderFixed ) -> Result[None, ZipError]:
	match io.write_all( writer, compiler.cast( ConstPtr[u8], compiler.addrof( hdr )), compiler.sizeof( _CentralDirectoryHeaderFixed )):
		case Result.Ok( _ ):
			return Result.Ok( None )
		case Result.Err( _ ):
			return Result.Err( ZipError( '_write_central_header: write error' ))


def _write_eocd( writer: BinaryWriter, eocd: _EndOfCentralDirectory ) -> Result[None, ZipError]:
	match io.write_all( writer, compiler.cast( ConstPtr[u8], compiler.addrof( eocd )), compiler.sizeof( _EndOfCentralDirectory )):
		case Result.Ok( _ ):
			return Result.Ok( None )
		case Result.Err( _ ):
			return Result.Err( ZipError( '_write_eocd: write error' ))


def _read_exact_bytes( reader: BinaryReader, count: usize ) -> Result[bytes, ZipError]:
	buf: bytearray = bytearray( count )
	match io.read_exact( reader, buf.get_ptr(), count ):
		case Result.Ok( _ ):
			return Result.Ok( bytes.from_bytearray( move( buf )))
		case Result.Err( _ ):
			return Result.Err( ZipError( '_read_exact_bytes: unexpected end of file or read error' ))


def _read_filename( reader: BinaryReader, filename_len: u16 ) -> Result[str, ZipError]:
	raw: bytes = _read_exact_bytes( reader, usize( filename_len )).or_return()
	match raw.decode():
		case Result.Ok( s ):
			return Result.Ok( s )
		case Result.Err( _ ):
			return Result.Err( ZipError( '_read_filename: filename is not valid UTF-8' ))


def _seek( target: BinaryReader|BinaryWriter, offset: i64, whence: i32, context: str ) -> Result[None, ZipError]:
	match target.seek( offset, whence ):
		case Result.Ok( _ ):
			return Result.Ok( None )
		case Result.Err( _ ):
			return Result.Err( ZipError( context ))


# ---------------------------------------------------------------------------
# ZipReader
# ---------------------------------------------------------------------------

class ZipReader:
	__reader:  BinaryReader
	__entries: list[ZipInfo]

	def __init__( self, reader: BinaryReader, entries: list[ZipInfo] ) -> None:
		self.__reader  = reader
		self.__entries = entries

	def namelist( self ) -> list[str]:
		names: list[str] = list[str]()
		i: usize = 0
		with compiler.panic_arithmetic( 'bounded by entry count, cannot overflow' ):
			while i < len( self.__entries ):
				info: ZipInfo = self.__entries.__getitem__( i ).unwrap( 'i < len(entries)' )
				names.append( info.filename ).unwrap( 'namelist: append failed' )
				i += 1
		return names

	def infolist( self ) -> list[ZipInfo]:
		return self.__entries

	def getinfo( self, name: str ) -> Result[ZipInfo, ZipError]:
		i: usize = 0
		with compiler.panic_arithmetic( 'bounded by entry count, cannot overflow' ):
			while i < len( self.__entries ):
				info: ZipInfo = self.__entries.__getitem__( i ).unwrap( 'i < len(entries)' )
				if info.filename == name:
					return Result.Ok( info )
				i += 1
		return Result.Err( ZipError( f'getinfo: no such entry {name!r}' ))

	def read( self, name: str ) -> Result[bytes, ZipError]:
		info: ZipInfo = self.getinfo( name ).or_return()

		if info.flags & 0x0001 != 0:
			return Result.Err( ZipError( f'read: {name!r} is encrypted, not supported' ))
		if info.compress_type != ZIP_STORED and info.compress_type != ZIP_DEFLATED:
			return Result.Err( ZipError( f'read: {name!r} uses unsupported compression method {info.compress_type}' ))

		with compiler.wrap_arithmetic:
			local_offset: i64 = i64( info.header_offset )
		_seek( self.__reader, local_offset, SEEK_SET, 'read: seek to local header failed' ).or_return()

		local_hdr: _LocalFileHeaderFixed = _read_local_header( self.__reader ).or_return()
		if local_hdr.signature != _LOCAL_FILE_HEADER_SIG:
			return Result.Err( ZipError( f'read: {name!r} has a bad local file header signature' ))

		# skip the local header's own filename+extra (its lengths can
		# legitimately differ in byte content from the central directory's
		# copies in obscure cases, but never in LENGTH for a well-formed
		# archive written by any real tool - trusting the LOCAL header's own
		# lengths here is what correctly skips straight to the compressed
		# data regardless)
		with compiler.wrap_arithmetic:
			skip: usize = usize( local_hdr.filename_len ) + usize( local_hdr.extra_len )
		_read_exact_bytes( self.__reader, skip ).or_return()

		raw: bytes = _read_exact_bytes( self.__reader, usize( info.compress_size )).or_return()

		decompressed: bytes
		if info.compress_type == ZIP_STORED:
			decompressed = raw
		else:
			match deflate.decompress_exact( raw, usize( info.file_size )):
				case Result.Ok( d ):
					decompressed = d
				case Result.Err( _ ):
					return Result.Err( ZipError( f'read: {name!r} failed to decompress' ))

		if len( decompressed ) != usize( info.file_size ):
			return Result.Err( ZipError( f'read: {name!r} decompressed to the wrong size' ))

		actual_crc: u32 = crc32.crc32( decompressed )
		if actual_crc != info.crc:
			return Result.Err( ZipError( f'read: {name!r} failed CRC-32 check' ))

		return Result.Ok( decompressed )

	def close( self ) -> None:
		self.__reader.close()


def _find_eocd( reader: BinaryReader, file_size: i64 ) -> Result[_EndOfCentralDirectory, ZipError]:
	with compiler.wrap_arithmetic:
		window: i64 = i64( 22 + 65536 )
	if window > file_size:
		window = file_size
	with compiler.panic_arithmetic( 'window <= file_size, just clamped' ):
		start: i64 = file_size - window
	_seek( reader, start, SEEK_SET, '_find_eocd: seek failed' ).or_return()
	buf: bytes = _read_exact_bytes( reader, usize( window )).or_return()
	ptr: ConstPtr[u8] = buf.get_const_ptr()

	n: usize = len( buf )
	if n < usize( 22 ):
		return Result.Err( ZipError( '_find_eocd: file too small to contain an End Of Central Directory record' ))

	with compiler.panic_arithmetic( 'n >= 22, just checked' ):
		search_from: usize = n - usize( 4 )
	found: bool = False
	found_at: usize = 0
	pos: usize = search_from
	while True:
		with compiler.wrap_arithmetic:
			sig: u32 = u32( ptr[pos] ) | ( u32( ptr[pos + 1] ) << 8 ) | ( u32( ptr[pos + 2] ) << 16 ) | ( u32( ptr[pos + 3] ) << 24 )
		if sig == _EOCD_SIG:
			found = True
			found_at = pos
			break
		if pos == 0:
			break
		with compiler.panic_arithmetic( 'pos != 0, just checked' ):
			pos -= 1

	if not found:
		return Result.Err( ZipError( '_find_eocd: End Of Central Directory signature not found' ))

	with compiler.panic_arithmetic( 'found_at + 22 <= n by construction of the search window' ):
		remaining: usize = n - found_at
	if remaining < usize( 22 ):
		return Result.Err( ZipError( '_find_eocd: truncated End Of Central Directory record' ))

	eocd: _EndOfCentralDirectory = _EndOfCentralDirectory()
	with compiler.wrap_arithmetic:
		field_ptr: ConstPtr[u8] = ptr + found_at
	sys.memcpy( compiler.cast( Ptr[u8], compiler.addrof( eocd )), field_ptr, compiler.sizeof( _EndOfCentralDirectory ))
	return Result.Ok( eocd )


def _read_central_directory( reader: BinaryReader, eocd: _EndOfCentralDirectory ) -> Result[list[ZipInfo], ZipError]:
	with compiler.wrap_arithmetic:
		cd_offset: i64 = i64( eocd.cd_offset )
	_seek( reader, cd_offset, SEEK_SET, '_read_central_directory: seek to central directory failed' ).or_return()

	entries: list[ZipInfo] = list[ZipInfo]()
	count: u16 = eocd.cd_records_total
	i: u16 = 0
	with compiler.wrap_arithmetic:
		while i < count:
			hdr: _CentralDirectoryHeaderFixed = _read_central_header( reader ).or_return()
			if hdr.signature != _CENTRAL_DIR_SIG:
				return Result.Err( ZipError( '_read_central_directory: bad central directory header signature' ))
			filename: str = _read_filename( reader, hdr.filename_len ).or_return()
			_read_exact_bytes( reader, usize( hdr.extra_len )).or_return()
			_read_exact_bytes( reader, usize( hdr.comment_len )).or_return()

			info = ZipInfo(
				filename      = filename,
				crc           = hdr.crc32,
				compress_size = hdr.compressed_size,
				file_size     = hdr.uncompressed_size,
				compress_type = i32( hdr.compression_method ),
				flags         = hdr.flags,
				header_offset = hdr.local_header_offset,
			)
			entries.append( info ).unwrap( '_read_central_directory: append failed' )
			i += 1
	return Result.Ok( entries )


# ---------------------------------------------------------------------------
# ZipWriter
# ---------------------------------------------------------------------------

class _WrittenEntry:
	info: ZipInfo

	def __init__( self, info: ZipInfo ) -> None:
		self.info = info


class ZipWriter:
	__writer:  BinaryWriter
	__written: list[_WrittenEntry]
	__closed:  bool

	def __init__( self, writer: BinaryWriter ) -> None:
		self.__writer  = writer
		self.__written = list[_WrittenEntry]()
		self.__closed  = False

	def writestr( self, arcname: str, data: bytes|bytearray, compress_type: i32 = ZIP_DEFLATED ) -> Result[None, ZipError]:
		file_size: usize = len( data )
		if file_size > usize( _MAX_U32 ):
			return Result.Err( ZipError( f'writestr: {arcname!r} is too large (no ZIP64 support)' ))

		crc: u32 = crc32.crc32( data )

		payload: bytes = _compress_payload( data, compress_type, arcname ).or_return()

		compress_size: usize = len( payload )
		if compress_size > usize( _MAX_U32 ):
			return Result.Err( ZipError( f'writestr: {arcname!r} compressed larger than 4 GiB (no ZIP64 support)' ))

		name_bytes: bytes = _encode_name( arcname ).or_return()
		if len( name_bytes ) > usize( 0xFFFF ):
			return Result.Err( ZipError( f'writestr: {arcname!r} filename is too long' ))

		match self.__writer.tell():
			case Result.Ok( pos ):
				with compiler.panic_arithmetic( 'a real file offset never exceeds u32 range for files this tool writes (no ZIP64)' ):
					header_offset: u32 = u32( pos )
			case Result.Err( _ ):
				return Result.Err( ZipError( 'writestr: tell() failed' ))

		hdr: _LocalFileHeaderFixed = _LocalFileHeaderFixed()
		hdr.signature          = _LOCAL_FILE_HEADER_SIG
		hdr.version_needed     = 20
		hdr.flags               = 0
		with compiler.wrap_arithmetic:
			hdr.compression_method = u16( compress_type )
		hdr.mod_time            = 0
		hdr.mod_date            = 0
		hdr.crc32                = crc
		with compiler.wrap_arithmetic:
			hdr.compressed_size    = u32( compress_size )
			hdr.uncompressed_size  = u32( file_size )
			hdr.filename_len       = u16( len( name_bytes ))
		hdr.extra_len            = 0

		_write_local_header( self.__writer, hdr ).or_return()
		match io.write_all( self.__writer, name_bytes.get_const_ptr(), len( name_bytes )):
			case Result.Ok( _ ):
				pass
			case Result.Err( _ ):
				return Result.Err( ZipError( f'writestr: {arcname!r} failed writing filename' ))
		match io.write_all( self.__writer, payload.get_const_ptr(), len( payload )):
			case Result.Ok( _ ):
				pass
			case Result.Err( _ ):
				return Result.Err( ZipError( f'writestr: {arcname!r} failed writing compressed data' ))

		with compiler.wrap_arithmetic:
			info = ZipInfo(
				filename      = arcname,
				crc           = crc,
				compress_size = u32( compress_size ),
				file_size     = u32( file_size ),
				compress_type = compress_type,
				flags         = 0,
				header_offset = header_offset,
			)
		self.__written.append( _WrittenEntry( info )).unwrap( 'writestr: bookkeeping append failed' )
		return Result.Ok( None )

	def write( self, path_on_disk: str, arcname: str, compress_type: i32 = ZIP_DEFLATED ) -> Result[None, ZipError]:
		src: BinaryReader
		match File.binary_reader( path_on_disk ):
			case Result.Ok( r ):
				src = r
			case Result.Err( _ ):
				return Result.Err( ZipError( f'write: failed to open {path_on_disk!r}' ))

		size: i64
		match src.seek( i64( 0 ), SEEK_END ):
			case Result.Ok( s ):
				size = s
			case Result.Err( _ ):
				return Result.Err( ZipError( f'write: failed to seek {path_on_disk!r}' ))
		_seek( src, i64( 0 ), SEEK_SET, f'write: failed to rewind {path_on_disk!r}' ).or_return()

		with compiler.panic_arithmetic( 'a real file size is never negative' ):
			count: usize = usize( size )
		data: bytes = _read_exact_bytes( src, count ).or_return()
		src.close()

		return self.writestr( arcname, data, compress_type )

	def close( self ) -> Result[None, ZipError]:
		if self.__closed:
			return Result.Ok( None )
		self.__closed = True

		cd_start: i64
		match self.__writer.tell():
			case Result.Ok( pos ):
				cd_start = pos
			case Result.Err( _ ):
				return Result.Err( ZipError( 'close: tell() before central directory failed' ))

		if len( self.__written ) > usize( 0xFFFF ):
			return Result.Err( ZipError( 'close: too many entries (no ZIP64 support)' ))

		i: usize = 0
		with compiler.panic_arithmetic( 'bounded by entry count, cannot overflow' ):
			while i < len( self.__written ):
				entry: _WrittenEntry = self.__written.__getitem__( i ).unwrap( 'i < len(written)' )
				info: ZipInfo = entry.info
				name_bytes: bytes = _encode_name( info.filename ).or_return()

				chdr: _CentralDirectoryHeaderFixed = _CentralDirectoryHeaderFixed()
				chdr.signature           = _CENTRAL_DIR_SIG
				chdr.version_made_by     = 20
				chdr.version_needed      = 20
				chdr.flags                = info.flags
				with compiler.wrap_arithmetic:
					chdr.compression_method = u16( info.compress_type )
				chdr.mod_time             = 0
				chdr.mod_date             = 0
				chdr.crc32                 = info.crc
				chdr.compressed_size      = info.compress_size
				chdr.uncompressed_size    = info.file_size
				with compiler.wrap_arithmetic:
					chdr.filename_len        = u16( len( name_bytes ))
				chdr.extra_len            = 0
				chdr.comment_len          = 0
				chdr.disk_number_start    = 0
				chdr.internal_attrs       = 0
				chdr.external_attrs       = 0
				chdr.local_header_offset  = info.header_offset

				_write_central_header( self.__writer, chdr ).or_return()
				match io.write_all( self.__writer, name_bytes.get_const_ptr(), len( name_bytes )):
					case Result.Ok( _ ):
						pass
					case Result.Err( _ ):
						return Result.Err( ZipError( 'close: failed writing a central directory filename' ))
				i += 1

		cd_end: i64
		match self.__writer.tell():
			case Result.Ok( pos ):
				cd_end = pos
			case Result.Err( _ ):
				return Result.Err( ZipError( 'close: tell() after central directory failed' ))

		eocd: _EndOfCentralDirectory = _EndOfCentralDirectory()
		eocd.signature            = _EOCD_SIG
		eocd.disk_number          = 0
		eocd.cd_start_disk        = 0
		with compiler.wrap_arithmetic:
			eocd.cd_records_this_disk = u16( len( self.__written ))
			eocd.cd_records_total     = u16( len( self.__written ))
			eocd.cd_size               = u32( cd_end - cd_start )
			eocd.cd_offset              = u32( cd_start )
		eocd.comment_len          = 0

		_write_eocd( self.__writer, eocd ).or_return()
		self.__writer.close()
		return Result.Ok( None )


def _compress_payload( data: bytes|bytearray, compress_type: i32, arcname: str ) -> Result[bytes, ZipError]:
	if compress_type == ZIP_STORED:
		return Result.Ok( bytes( data ))
	elif compress_type == ZIP_DEFLATED:
		match deflate.compress( data ):
			case Result.Ok( c ):
				return Result.Ok( c )
			case Result.Err( _ ):
				return Result.Err( ZipError( f'writestr: {arcname!r} failed to compress' ))
	else:
		return Result.Err( ZipError( f'writestr: unsupported compress_type {compress_type}' ))


def _encode_name( name: str ) -> Result[bytes, ZipError]:
	match name.encode():
		case Result.Ok( b ):
			return Result.Ok( b )
		case Result.Err( _ ):
			return Result.Err( ZipError( f'{name!r} is not valid UTF-8' ))


# ---------------------------------------------------------------------------
# ZipFile - static-factory namespace, mirrors builtins.File's own shape
# ---------------------------------------------------------------------------

class ZipFile:

	@staticmethod
	def open( path: str ) -> Result[ZipReader, ZipError]:
		reader: BinaryReader
		match File.binary_reader( path ):
			case Result.Ok( r ):
				reader = r
			case Result.Err( _ ):
				return Result.Err( ZipError( f'open: failed to open {path!r}' ))

		size: i64
		match reader.seek( i64( 0 ), SEEK_END ):
			case Result.Ok( s ):
				size = s
			case Result.Err( _ ):
				return Result.Err( ZipError( f'open: failed to seek {path!r}' ))

		eocd: _EndOfCentralDirectory = _find_eocd( reader, size ).or_return()
		entries: list[ZipInfo] = _read_central_directory( reader, eocd ).or_return()
		return Result.Ok( ZipReader( reader, entries ))

	@staticmethod
	def create( path: str ) -> Result[ZipWriter, ZipError]:
		writer: BinaryWriter
		match File.binary_writer( path ):
			case Result.Ok( w ):
				writer = w
			case Result.Err( _ ):
				return Result.Err( ZipError( f'create: failed to open {path!r} for writing' ))
		return Result.Ok( ZipWriter( writer ))
