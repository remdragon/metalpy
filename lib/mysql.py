# lib/mysql.py — MySQL/MariaDB client: wire protocol + a PEP-249-shaped
# DB-API surface. See database_client_investigation.md for the full design
# (paramstyle, autocommit default, rowcount semantics, MySQLError, thread-
# safety enforcement, cursor lifecycle, type mapping, prepared-statement-
# only safety rule).
#
# v1 scope: handshake v10 + CLIENT_PROTOCOL_41, mysql_native_password auth
# only, plain TCP (no CLIENT_SSL - the design doc's own §1 calls this
# "recommended... but not force-mandated in v1"; lib/ssl.py's SSLSocket has
# a different send/recv/close shape than socket.Socket and wiring it in
# would mean either a generic Connection[T] (which can't cross a runtime
# plain-vs-TLS branch as ONE returned type, unlike lib/http/client.py's
# request/response calls which stay inside one branch) or a second
# transport field + branch on every I/O call - deferred, not attempted,
# given v1's time budget). Binary protocol only (COM_STMT_PREPARE/EXECUTE/
# CLOSE) for every caller query - COM_QUERY (text protocol) is used ONLY
# for the driver's own fixed, zero-caller-data admin statements (SET
# autocommit=0, COMMIT, ROLLBACK) - see _com_query()'s own comment for why
# that's consistent with, not a violation of, design doc §10's safety rule.

import compiler
import sys
import socket
import threading
from sha1 import sha1
from datetime import date, datetime, timedelta
from zoneinfo import ZoneInfo


@enum( i32 )
class MySQLError:
	ConnectFailed         = 1  # TCP connect failed
	TLSFailed             = 2  # reserved - no TLS backend wired in v1 (see header comment)
	AuthFailed            = 3  # server rejected credentials (ERR_Packet during handshake)
	UnsupportedAuthPlugin = 4  # server asked for a plugin other than mysql_native_password
	ProtocolError         = 5  # malformed packet, unexpected packet type, sequence-id mismatch
	QueryError            = 6  # ERR_Packet in response to a real statement - see Connection.last_server_errno/last_sqlstate/last_server_message
	TypeMismatch          = 7  # bound parameter count/type didn't match the statement
	Closed                = 8  # operation attempted on a closed Connection/Cursor
	WrongThread           = 9  # Connection/Cursor used from a thread other than the one that created it
	Other = _


# ---------------------------------------------------------------------------
# Capability flags (numeric values confirmed against dev.mysql.com's real
# group__group__cs__capabilities__flags.html, not guessed) and MYSQL_TYPE_*
# field type codes (confirmed against the real binary-protocol resultset row
# and column-definition packet docs - these ARE this module's public column
# type codes, exposed on ColumnDescription.type_code).
# ---------------------------------------------------------------------------

_CLIENT_LONG_PASSWORD:     u32 = 0x00000001
_CLIENT_PROTOCOL_41:       u32 = 0x00000200
_CLIENT_SECURE_CONNECTION: u32 = 0x00008000
_CLIENT_MULTI_RESULTS:     u32 = 0x00020000
_CLIENT_PLUGIN_AUTH:       u32 = 0x00080000
_CLIENT_CONNECT_WITH_DB:   u32 = 0x00000008
# CLIENT_FOUND_ROWS and CLIENT_DEPRECATE_EOF are deliberately never set -
# see design doc §4 (rowcount semantics) and this file's EOF-packet reads.
_CLIENT_CAPS: u32 = _CLIENT_LONG_PASSWORD | _CLIENT_PROTOCOL_41 | _CLIENT_SECURE_CONNECTION | _CLIENT_PLUGIN_AUTH | _CLIENT_MULTI_RESULTS

MYSQL_TYPE_DECIMAL:    u8 = 0
MYSQL_TYPE_TINY:       u8 = 1
MYSQL_TYPE_SHORT:      u8 = 2
MYSQL_TYPE_LONG:       u8 = 3
MYSQL_TYPE_FLOAT:      u8 = 4
MYSQL_TYPE_DOUBLE:     u8 = 5
MYSQL_TYPE_NULL:       u8 = 6
MYSQL_TYPE_TIMESTAMP:  u8 = 7
MYSQL_TYPE_LONGLONG:   u8 = 8
MYSQL_TYPE_INT24:      u8 = 9
MYSQL_TYPE_DATE:       u8 = 10
MYSQL_TYPE_TIME:       u8 = 11
MYSQL_TYPE_DATETIME:   u8 = 12
MYSQL_TYPE_YEAR:       u8 = 13
MYSQL_TYPE_VARCHAR:    u8 = 15
MYSQL_TYPE_BIT:        u8 = 16
MYSQL_TYPE_JSON:       u8 = 245
MYSQL_TYPE_NEWDECIMAL: u8 = 246
MYSQL_TYPE_ENUM:       u8 = 247
MYSQL_TYPE_SET:        u8 = 248
MYSQL_TYPE_TINY_BLOB:  u8 = 249
MYSQL_TYPE_MEDIUM_BLOB: u8 = 250
MYSQL_TYPE_LONG_BLOB:  u8 = 251
MYSQL_TYPE_BLOB:       u8 = 252
MYSQL_TYPE_VAR_STRING: u8 = 253
MYSQL_TYPE_STRING:     u8 = 254
MYSQL_TYPE_GEOMETRY:   u8 = 255

_UNSIGNED_PARAM_FLAG: u16 = 0x8000  # COM_STMT_EXECUTE per-param type marker: unsigned flag in the type word's MSB
_COLUMN_UNSIGNED_FLAG: u16 = 32     # column-definition `flags` field: UNSIGNED_FLAG (distinct bit space from the above)
_COLUMN_NOT_NULL_FLAG: u16 = 1      # column-definition `flags` field: NOT_NULL_FLAG

_COM_QUIT:          u8 = 0x01
_COM_QUERY:         u8 = 0x03
_COM_STMT_PREPARE:  u8 = 0x16
_COM_STMT_EXECUTE:  u8 = 0x17
_COM_STMT_CLOSE:    u8 = 0x19

# a naive placeholder tz for DATE/DATETIME/TIMESTAMP decode - lib/datetime.py
# requires every datetime to carry a real ZoneInfo (see that module's own
# docstring: no naive datetime exists there), but MySQL's DATETIME/TIMESTAMP
# carry no tz at all (design doc §9: "always naive (server-local)... no
# implicit tz conversion is performed"). Resolution: tag every value with
# this fixed +00:00 zone purely to satisfy the mandatory-tzinfo field - it is
# NOT a claim the value is actually UTC, just a neutral placeholder. Noted in
# database_client_investigation.md as a resolved design-doc/lib-datetime.py
# mismatch.
_NAIVE_TZ: ZoneInfo = ZoneInfo.fixed_offset( 0, 'naive' )


@union
class MySQLValue:
	''' one bound parameter OR one fetched column value - see design doc §9's
	type table (this is that table's "metalpy type" column, collapsed into
	one tagged union so a row can hold heterogeneous column types and a
	params list can hold heterogeneous argument types). Int/UInt cover every
	SQL integer width (TINYINT..BIGINT) - width is a column-metadata detail,
	not a distinct metalpy type, matching §9. '''
	Null:     None
	Int:      i64
	UInt:     u64
	Float32:  f32
	Float64:  f64
	Str:      str
	Bytes:    bytes
	Date:     date
	DateTime: datetime
	Time:     timedelta


class ColumnDescription:
	''' PEP-249-shaped column metadata - a plain RC class (not @cstruct),
	matching lib/socket.py's SocketAddr precedent: a struct-typed field
	holding a str would silently escape this compiler's CFG-driven RC decref
	walk (@struct/@cstruct fields aren't tracked by it). '''
	__name:     str
	__type_code: u8
	__size:     u32
	__scale:    u8
	__nullable: bool
	__flags:    u16  # column-definition flags word - UNSIGNED_FLAG read via _flags(), not otherwise exposed
	__charset:  u16  # column charset id - 63 ("binary") disambiguates BLOB from TEXT for the shared wire type codes, see _read_column_value

	@property
	def name( self ) -> str:
		return self.__name

	@property
	def type_code( self ) -> u8:
		return self.__type_code

	@property
	def size( self ) -> u32:
		return self.__size

	@property
	def scale( self ) -> u8:
		return self.__scale

	@property
	def nullable( self ) -> bool:
		return self.__nullable

	@private
	def _flags( self ) -> u16:
		return self.__flags

	@private
	def _charset( self ) -> u16:
		return self.__charset

	@private
	@staticmethod
	def _make( name: str, type_code: u8, size: u32, scale: u8, nullable: bool, flags: u16, charset: u16 ) -> ColumnDescription:
		return ColumnDescription.__allocate__( __name = name, __type_code = type_code, __size = size, __scale = scale, __nullable = nullable, __flags = flags, __charset = charset )


# ---------------------------------------------------------------------------
# _Buf - a growable write buffer for building packet payloads. Same
# grow/append-only shape as lib/socket.py's RecvBuffer / lib/ssl.py's
# _CipherBuf, plus typed little-endian/length-encoded write helpers this
# module's packet builders need repeatedly.
# ---------------------------------------------------------------------------

class _Buf:
	__data: Ptr[u8]
	__len:  usize
	__cap:  usize

	def __init__( self, initial_cap: usize = 256 ) -> None:
		self.__cap = initial_cap
		self.__data = sys.alloc[u8]( self.__cap )
		self.__len = 0

	def __del__( self ) -> None:
		sys.free( self.__data )

	def len( self ) -> usize:
		return self.__len

	def get_const_ptr( self ) -> ConstPtr[u8]:
		return self.__data

	def _grow( self, min_additional: usize ) -> None:
		with compiler.panic_arithmetic( 'irrational buffer growth' ):
			needed: usize = self.__len + min_additional
		if needed <= self.__cap:
			return
		new_cap: usize = self.__cap
		with compiler.panic_arithmetic( 'irrational buffer growth' ):
			while new_cap < needed:
				new_cap = new_cap * 2
		new_data: Ptr[u8] = sys.alloc[u8]( new_cap )
		sys.memcpy( new_data, self.__data, self.__len )
		sys.free( self.__data )
		self.__data = new_data
		self.__cap = new_cap

	def write_u8( self, v: u8 ) -> None:
		self._grow( usize( 1 ))
		with compiler.wrap_arithmetic:
			self.__data[self.__len] = v
			self.__len += usize( 1 )

	def write_u16le( self, v: u16 ) -> None:
		self._grow( usize( 2 ))
		with compiler.wrap_arithmetic:
			p: Ptr[u8] = self.__data + self.__len
			p[0] = u8( v & 0xFF )
			p[1] = u8(( v >> 8 ) & 0xFF )
			self.__len += usize( 2 )

	def write_u24le( self, v: u32 ) -> None:
		self._grow( usize( 3 ))
		with compiler.wrap_arithmetic:
			p: Ptr[u8] = self.__data + self.__len
			p[0] = u8(  v         & 0xFF )
			p[1] = u8(( v >> 8  ) & 0xFF )
			p[2] = u8(( v >> 16 ) & 0xFF )
			self.__len += usize( 3 )

	def write_u32le( self, v: u32 ) -> None:
		self._grow( usize( 4 ))
		with compiler.wrap_arithmetic:
			p: Ptr[u8] = self.__data + self.__len
			p[0] = u8(  v         & 0xFF )
			p[1] = u8(( v >> 8  ) & 0xFF )
			p[2] = u8(( v >> 16 ) & 0xFF )
			p[3] = u8(( v >> 24 ) & 0xFF )
			self.__len += usize( 4 )

	def write_u64le( self, v: u64 ) -> None:
		self._grow( usize( 8 ))
		with compiler.wrap_arithmetic:
			p: Ptr[u8] = self.__data + self.__len
			i: u64 = 0
			while i < 8:
				p[usize( i )] = u8(( v >> ( i * 8 )) & 0xFF )
				i += 1
			self.__len += usize( 8 )

	def write_bytes( self, src: ConstPtr[u8], n: usize ) -> None:
		self._grow( n )
		with compiler.wrap_arithmetic:
			dst: Ptr[u8] = self.__data + self.__len
			sys.memcpy( dst, src, n )
			self.__len += n

	def write_zeros( self, n: usize ) -> None:
		self._grow( n )
		with compiler.wrap_arithmetic:
			dst: Ptr[u8] = self.__data + self.__len
			sys.memset( dst, 0, n )
			self.__len += n

	def write_lenenc_int( self, v: u64 ) -> None:
		with compiler.wrap_arithmetic:
			if v < 251:
				self.write_u8( u8( v ))
			elif v < 65536:
				self.write_u8( 0xFC )
				self.write_u16le( u16( v ))
			elif v < 16777216:
				self.write_u8( 0xFD )
				self.write_u24le( u32( v ))
			else:
				self.write_u8( 0xFE )
				self.write_u64le( v )

	def write_lenenc_bytes( self, src: ConstPtr[u8], n: usize ) -> None:
		self.write_lenenc_int( u64( n ))
		self.write_bytes( src, n )

	def write_lenenc_str( self, s: str ) -> None:
		self.write_lenenc_bytes( s.get_const_ptr(), s.byte_len() )

	def write_cstr( self, s: str ) -> None:
		''' NUL-terminated (username/database/plugin-name fields) - NOT a
		length-encoded string. '''
		self.write_bytes( s.get_const_ptr(), s.byte_len() )
		self.write_u8( 0 )


# ---------------------------------------------------------------------------
# _Reader - a bounds-checked cursor over one already-fully-received packet
# payload. Every read returns Result[T, MySQLError] (ProtocolError on a
# short/malformed read) rather than panicking - a server bug or a MITM'd
# connection is a runtime condition to report, not a programmer error.
# ---------------------------------------------------------------------------

class _Reader:
	__data: ConstPtr[u8]
	__len:  usize
	__pos:  usize

	def __init__( self, data: ConstPtr[u8], length: usize ) -> None:
		self.__data = data
		self.__len = length
		self.__pos = 0

	def remaining( self ) -> usize:
		with compiler.wrap_arithmetic:
			return self.__len - self.__pos

	def peek_u8( self ) -> Result[u8, MySQLError]:
		if self.remaining() < usize( 1 ):
			return Result.Err( MySQLError.ProtocolError )
		return Result.Ok( self.__data[self.__pos] )

	def read_u8( self ) -> Result[u8, MySQLError]:
		v: u8 = self.peek_u8().or_return()
		with compiler.wrap_arithmetic:
			self.__pos += usize( 1 )
		return Result.Ok( v )

	def read_u16le( self ) -> Result[u16, MySQLError]:
		if self.remaining() < usize( 2 ):
			return Result.Err( MySQLError.ProtocolError )
		with compiler.wrap_arithmetic:
			p: ConstPtr[u8] = self.__data + self.__pos
			v: u16 = u16( p[0] ) | ( u16( p[1] ) << 8 )
			self.__pos += usize( 2 )
		return Result.Ok( v )

	def read_u24le( self ) -> Result[u32, MySQLError]:
		if self.remaining() < usize( 3 ):
			return Result.Err( MySQLError.ProtocolError )
		with compiler.wrap_arithmetic:
			p: ConstPtr[u8] = self.__data + self.__pos
			v: u32 = u32( p[0] ) | ( u32( p[1] ) << 8 ) | ( u32( p[2] ) << 16 )
			self.__pos += usize( 3 )
		return Result.Ok( v )

	def read_u32le( self ) -> Result[u32, MySQLError]:
		if self.remaining() < usize( 4 ):
			return Result.Err( MySQLError.ProtocolError )
		with compiler.wrap_arithmetic:
			p: ConstPtr[u8] = self.__data + self.__pos
			v: u32 = u32( p[0] ) | ( u32( p[1] ) << 8 ) | ( u32( p[2] ) << 16 ) | ( u32( p[3] ) << 24 )
			self.__pos += usize( 4 )
		return Result.Ok( v )

	def read_u64le( self ) -> Result[u64, MySQLError]:
		if self.remaining() < usize( 8 ):
			return Result.Err( MySQLError.ProtocolError )
		with compiler.wrap_arithmetic:
			p: ConstPtr[u8] = self.__data + self.__pos
			v: u64 = 0
			i: usize = 0
			while i < 8:
				v |= u64( p[i] ) << ( u64( i ) * 8 )
				i += 1
			self.__pos += usize( 8 )
		return Result.Ok( v )

	def skip( self, n: usize ) -> Result[None, MySQLError]:
		if self.remaining() < n:
			return Result.Err( MySQLError.ProtocolError )
		with compiler.wrap_arithmetic:
			self.__pos += n
		return Result.Ok( None )

	def read_lenenc_int( self ) -> Result[u64, MySQLError]:
		first: u8 = self.read_u8().or_return()
		if first < 0xFB:
			return Result.Ok( u64( first ))
		if first == 0xFC:
			return Result.Ok( u64( self.read_u16le().or_return() ))
		if first == 0xFD:
			return Result.Ok( u64( self.read_u24le().or_return() ))
		if first == 0xFE:
			return Result.Ok( self.read_u64le().or_return() )
		# 0xFB is the text-protocol NULL sentinel - never legal here since
		# the binary protocol always signals NULL via the row's own
		# null-bitmap instead (design doc §9's "never encoded/decoded as a
		# sentinel value" rule, applied to length-encoding itself).
		return Result.Err( MySQLError.ProtocolError )

	def read_fixed_bytes( self, n: usize ) -> Result[bytes, MySQLError]:
		if self.remaining() < n:
			return Result.Err( MySQLError.ProtocolError )
		buf = bytearray( n )
		with compiler.wrap_arithmetic:
			sys.memcpy( buf.get_ptr(), self.__data + self.__pos, n )
			self.__pos += n
		return Result.Ok( bytes.from_bytearray( move( buf )))

	def read_lenenc_bytes( self ) -> Result[bytes, MySQLError]:
		n: u64 = self.read_lenenc_int().or_return()
		with compiler.saturate_arithmetic:
			count: usize = usize( n )
		return self.read_fixed_bytes( count )

	def read_lenenc_str( self ) -> Result[str, MySQLError]:
		raw: bytes = self.read_lenenc_bytes().or_return()
		match raw.decode():
			case Result.Ok( s ):
				return Result.Ok( s )
			case Result.Err( _ ):
				return Result.Err( MySQLError.ProtocolError )

	def read_null_term_str( self ) -> Result[str, MySQLError]:
		start: usize = self.__pos
		p: ConstPtr[u8] = self.__data
		i: usize = start
		with compiler.wrap_arithmetic:
			while i < self.__len:
				if p[i] == 0:
					break
				i += 1
		if i >= self.__len:
			return Result.Err( MySQLError.ProtocolError )
		with compiler.wrap_arithmetic:
			n: usize = i - start
			after: usize = i + usize( 1 )
			str_ptr: ConstPtr[u8] = p + start
			str_len: usize = n + usize( 1 )
		match str.from_cstr( str_ptr, str_len ):
			case Result.Ok( s ):
				self.__pos = after
				return Result.Ok( s )
			case Result.Err( _ ):
				return Result.Err( MySQLError.ProtocolError )

	def read_rest_bytes( self ) -> bytes:
		n: usize = self.remaining()
		buf = bytearray( n )
		with compiler.wrap_arithmetic:
			src: ConstPtr[u8] = self.__data + self.__pos
		sys.memcpy( buf.get_ptr(), src, n )
		self.__pos = self.__len
		return bytes.from_bytearray( move( buf ))


# ---------------------------------------------------------------------------
# mysql_native_password scramble - SHA1(password) XOR SHA1(seed +
# SHA1(SHA1(password))). Verified against the real MariaDB protocol docs
# (mariadb.com/docs, connection phase) before writing this, not recalled
# from memory alone - getting this formula wrong would silently break auth
# rather than fail to compile.
# ---------------------------------------------------------------------------

def _scramble_password( password: str, seed: bytes ) -> bytes:
	if password.byte_len() == usize( 0 ):
		return bytes.from_bytearray( move( bytearray( 0 )))
	pw_bytes: bytes = password.encode().unwrap( 'mysql: password must be valid UTF-8' )
	stage1: bytes = sha1( pw_bytes )
	stage2: bytes = sha1( stage1 )

	with compiler.wrap_arithmetic:
		combined_len: usize = seed.__len__() + stage2.__len__()
	combined = bytearray( combined_len )
	cp: Ptr[u8] = combined.get_ptr()
	with compiler.wrap_arithmetic:
		combined_tail: Ptr[u8] = cp + seed.__len__()
	sys.memcpy( cp, seed.get_const_ptr(), seed.__len__() )
	sys.memcpy( combined_tail, stage2.get_const_ptr(), stage2.__len__() )
	stage3: bytes = sha1( bytes.from_bytearray( move( combined )))

	n: usize = stage1.__len__()
	out = bytearray( n )
	op: Ptr[u8] = out.get_ptr()
	s1: ConstPtr[u8] = stage1.get_const_ptr()
	s3: ConstPtr[u8] = stage3.get_const_ptr()
	i: usize = 0
	with compiler.wrap_arithmetic:
		while i < n:
			op[i] = s1[i] ^ s3[i]
			i += 1
	return bytes.from_bytearray( move( out ))


# ---------------------------------------------------------------------------
# Packet framing - 3-byte little-endian length + 1-byte sequence id, per the
# real MySQL client/server protocol header (confirmed against dev.mysql.com).
# v1 never splits a payload across multiple 0xFFFFFF-boundary packets - every
# payload this driver ever builds or expects (queries, small result sets) is
# far below 16MB.
# ---------------------------------------------------------------------------

def _recv_exact( sock: socket.Socket, buf: Ptr[u8], count: usize ) -> Result[None, MySQLError]:
	got: usize = 0
	with compiler.wrap_arithmetic:
		while got < count:
			n: usize = 0
			match sock.recv( buf + got, count - got ):
				case Result.Ok( k ):
					n = k
				case Result.Err( _ ):
					return Result.Err( MySQLError.ProtocolError )
			if n == 0:
				return Result.Err( MySQLError.ProtocolError )  # peer closed mid-packet
			got += n
	return Result.Ok( None )


def _write_packet( sock: socket.Socket, seq: u8, payload: ConstPtr[u8], payload_len: usize ) -> Result[None, MySQLError]:
	with compiler.saturate_arithmetic:
		n: u32 = u32( payload_len )
	header = bytearray( usize( 4 ))
	hp: Ptr[u8] = header.get_ptr()
	with compiler.wrap_arithmetic:
		hp[0] = u8(  n         & 0xFF )
		hp[1] = u8(( n >> 8  ) & 0xFF )
		hp[2] = u8(( n >> 16 ) & 0xFF )
	hp[3] = seq
	match sock.send_all( header.get_const_ptr(), usize( 4 )):
		case Result.Err( _ ):
			return Result.Err( MySQLError.ProtocolError )
		case Result.Ok( _ ):
			pass
	if payload_len > usize( 0 ):
		match sock.send_all( payload, payload_len ):
			case Result.Err( _ ):
				return Result.Err( MySQLError.ProtocolError )
			case Result.Ok( _ ):
				pass
	return Result.Ok( None )


def _read_packet( sock: socket.Socket, expected_seq: u8 ) -> Result[tuple[bytes, u8], MySQLError]:
	header = bytearray( usize( 4 ))
	_recv_exact( sock, header.get_ptr(), usize( 4 )).or_return()
	hp: ConstPtr[u8] = header.get_const_ptr()
	with compiler.wrap_arithmetic:
		length: u32 = u32( hp[0] ) | ( u32( hp[1] ) << 8 ) | ( u32( hp[2] ) << 16 )
	seq: u8 = hp[3]
	if seq != expected_seq:
		return Result.Err( MySQLError.ProtocolError )
	with compiler.saturate_arithmetic:
		plen: usize = usize( length )
	payload = bytearray( plen )
	if plen > usize( 0 ):
		_recv_exact( sock, payload.get_ptr(), plen ).or_return()
	with compiler.wrap_arithmetic:
		next_seq: u8 = seq + 1
	return Result.Ok(( bytes.from_bytearray( move( payload )), next_seq ))


# ---------------------------------------------------------------------------
# OK_Packet / ERR_Packet parsing - shared by the post-auth OK, COM_QUERY
# responses, and COM_STMT_EXECUTE's own OK-for-no-resultset path. Byte
# layouts confirmed against dev.mysql.com's real protocol docs.
# ---------------------------------------------------------------------------

class _OkResult:
	affected_rows: u64
	last_insert_id: u64

	@staticmethod
	def _make( affected_rows: u64, last_insert_id: u64 ) -> _OkResult:
		return _OkResult.__allocate__( affected_rows = affected_rows, last_insert_id = last_insert_id )


def _parse_ok_packet( payload: bytes ) -> Result[_OkResult, MySQLError]:
	r = _Reader( payload.get_const_ptr(), payload.__len__() )
	header: u8 = r.read_u8().or_return()
	if header != 0x00:
		return Result.Err( MySQLError.ProtocolError )
	affected_rows: u64 = r.read_lenenc_int().or_return()
	last_insert_id: u64 = r.read_lenenc_int().or_return()
	return Result.Ok( _OkResult._make( affected_rows, last_insert_id ))


class _ErrInfo:
	errno:   i32
	sqlstate: str
	message:  str

	@staticmethod
	def _make( errno: i32, sqlstate: str, message: str ) -> _ErrInfo:
		return _ErrInfo.__allocate__( errno = errno, sqlstate = sqlstate, message = message )


def _parse_err_packet( payload: bytes ) -> Result[_ErrInfo, MySQLError]:
	r = _Reader( payload.get_const_ptr(), payload.__len__() )
	header: u8 = r.read_u8().or_return()
	if header != 0xFF:
		return Result.Err( MySQLError.ProtocolError )
	code: u16 = r.read_u16le().or_return()
	with compiler.wrap_arithmetic:
		errno: i32 = i32( code )
	r.skip( usize( 1 )).or_return()  # '#' sql_state_marker (CLIENT_PROTOCOL_41 always set here)
	state_bytes: bytes = r.read_fixed_bytes( usize( 5 )).or_return()
	sqlstate: str = state_bytes.decode().unwrap( 'sqlstate is always ASCII' )
	msg_bytes: bytes = r.read_rest_bytes()
	message: str = msg_bytes.decode_lossy()
	return Result.Ok( _ErrInfo._make( errno, sqlstate, message ))


def _expect_eof( payload: bytes ) -> Result[None, MySQLError]:
	if len( payload ) == usize( 0 ):
		return Result.Err( MySQLError.ProtocolError )
	first: u8 = payload.__getitem__( usize( 0 )).unwrap( 'non-empty packet' )
	if first != 0xFE or len( payload ) >= usize( 9 ):
		return Result.Err( MySQLError.ProtocolError )
	return Result.Ok( None )


def _send_stmt_close( sock: socket.Socket, stmt_id: u32 ) -> Result[None, MySQLError]:
	''' COM_STMT_CLOSE has no response at all (fire-and-forget, per protocol) -
	the next command's own sequence starts fresh at 0 regardless. '''
	buf = _Buf()
	buf.write_u8( _COM_STMT_CLOSE )
	buf.write_u32le( stmt_id )
	return _write_packet( sock, 0, buf.get_const_ptr(), buf.len() )


# ---------------------------------------------------------------------------
# IEEE-754 bit reinterpretation for FLOAT/DOUBLE wire values - this compiler
# has no bitcast/reinterpret intrinsic for scalars (confirmed: lib/builtins/
# __float.py's own _f64_sign_prefix docstring says so directly, re -0.0
# sign-bit detection). Worked around with the SAME pointer-reinterpret idiom
# already established elsewhere in this codebase (lib/socket.py's inet_pton/
# inet_ntop casts, cited by lib/builtins/__ptr_arith.py's own comment as the
# precedent) - stage into a local, take its address, reinterpret the pointer
# type, dereference. Not a new risky pattern, just this one applied to f32/
# f64 specifically.
# ---------------------------------------------------------------------------

def _f32_to_bits( v: f32 ) -> u32:
	tmp: f32 = v
	p: Ptr[u32] = compiler.cast( Ptr[u32], compiler.addrof( tmp ))
	return p[0]

def _bits_to_f32( v: u32 ) -> f32:
	tmp: u32 = v
	p: Ptr[f32] = compiler.cast( Ptr[f32], compiler.addrof( tmp ))
	return p[0]

def _f64_to_bits( v: f64 ) -> u64:
	tmp: f64 = v
	p: Ptr[u64] = compiler.cast( Ptr[u64], compiler.addrof( tmp ))
	return p[0]

def _bits_to_f64( v: u64 ) -> f64:
	tmp: u64 = v
	p: Ptr[f64] = compiler.cast( Ptr[f64], compiler.addrof( tmp ))
	return p[0]


# ---------------------------------------------------------------------------
# Binary-protocol value encode (COM_STMT_EXECUTE params) / decode (resultset
# rows) - see design doc §9's full SQL<->metalpy type table. `binary` charset
# (collation id 63 - the real, well-known MySQL/MariaDB constant for it) is
# how VARCHAR/VAR_STRING/STRING/*_BLOB column defs distinguish an actual
# BLOB/BINARY/VARBINARY column from a TEXT-family one sharing the same wire
# type code - confirmed real MySQL client behavior (mysqlnd/libmysqlclient
# both key off this), not a guess.
# ---------------------------------------------------------------------------

_BINARY_CHARSET_ID: u16 = 63

def _build_execute_payload( stmt_id: u32, params: list[MySQLValue] ) -> _Buf:
	''' COM_STMT_EXECUTE - always sends new_params_bind_flag=1 and a fresh
	set of per-param type markers (design doc §10: no cached/reused prepared
	statements, so there's never a reason to omit them). NULL-bitmap size
	formula and bit offset (0, unlike a resultset row's own offset-2 bitmap)
	confirmed against dev.mysql.com's real COM_STMT_EXECUTE docs. '''
	buf = _Buf()
	buf.write_u8( _COM_STMT_EXECUTE )
	buf.write_u32le( stmt_id )
	buf.write_u8( 0 )    # flags - CURSOR_TYPE_NO_CURSOR
	buf.write_u32le( 1 ) # iteration_count

	n: usize = len( params )
	if n > usize( 0 ):
		with compiler.panic_arithmetic( 'divisor is the nonzero literal 8' ):
			bitmap_len: usize = ( n + 7 ) // 8
		bitmap = bytearray( bitmap_len )
		bp: Ptr[u8] = bitmap.get_ptr()
		i: usize = 0
		with compiler.panic_arithmetic( 'i < n, divisor is the nonzero literal 8' ):
			while i < n:
				v: MySQLValue = params.__getitem__( i ).unwrap( 'i < n' )
				match v:
					case MySQLValue.Null( _ ):
						byte_idx: usize = i // 8
						bit_idx: usize = i % 8
						bp[byte_idx] = bp[byte_idx] | u8( 1 << bit_idx )
					case _:
						pass
				i += 1
		buf.write_bytes( bitmap.get_const_ptr(), bitmap_len )
		buf.write_u8( 1 )  # new_params_bind_flag

		j: usize = 0
		with compiler.panic_arithmetic( 'j < n, cannot overflow' ):
			while j < n:
				_write_param_type( buf, params.__getitem__( j ).unwrap( 'j < n' ))
				j += 1

		k: usize = 0
		with compiler.panic_arithmetic( 'k < n, cannot overflow' ):
			while k < n:
				_write_param_value( buf, params.__getitem__( k ).unwrap( 'k < n' ))
				k += 1
	return buf


def _write_param_type( buf: _Buf, v: MySQLValue ) -> None:
	match v:
		case MySQLValue.Null( _ ):
			buf.write_u16le( u16( MYSQL_TYPE_NULL ))
		case MySQLValue.Int( _ ):
			buf.write_u16le( u16( MYSQL_TYPE_LONGLONG ))
		case MySQLValue.UInt( _ ):
			buf.write_u16le( u16( MYSQL_TYPE_LONGLONG ) | _UNSIGNED_PARAM_FLAG )
		case MySQLValue.Float32( _ ):
			buf.write_u16le( u16( MYSQL_TYPE_FLOAT ))
		case MySQLValue.Float64( _ ):
			buf.write_u16le( u16( MYSQL_TYPE_DOUBLE ))
		case MySQLValue.Str( _ ):
			buf.write_u16le( u16( MYSQL_TYPE_VAR_STRING ))
		case MySQLValue.Bytes( _ ):
			buf.write_u16le( u16( MYSQL_TYPE_BLOB ))
		case MySQLValue.Date( _ ):
			buf.write_u16le( u16( MYSQL_TYPE_DATE ))
		case MySQLValue.DateTime( _ ):
			buf.write_u16le( u16( MYSQL_TYPE_DATETIME ))
		case MySQLValue.Time( _ ):
			buf.write_u16le( u16( MYSQL_TYPE_TIME ))


def _write_param_value( buf: _Buf, v: MySQLValue ) -> None:
	''' NULL writes nothing - it's signaled entirely by the null-bitmap
	(design doc §9: "never encoded/decoded as a sentinel value"). '''
	match v:
		case MySQLValue.Null( _ ):
			pass
		case MySQLValue.Int( iv ):
			with compiler.wrap_arithmetic:
				buf.write_u64le( u64( iv ))
		case MySQLValue.UInt( uv ):
			buf.write_u64le( uv )
		case MySQLValue.Float32( fv ):
			buf.write_u32le( _f32_to_bits( fv ))
		case MySQLValue.Float64( dv ):
			buf.write_u64le( _f64_to_bits( dv ))
		case MySQLValue.Str( sv ):
			buf.write_lenenc_str( sv )
		case MySQLValue.Bytes( bv ):
			buf.write_lenenc_bytes( bv.get_const_ptr(), len( bv ))
		case MySQLValue.Date( d ):
			buf.write_u8( 4 )
			with compiler.wrap_arithmetic:
				buf.write_u16le( u16( d.year ))
				buf.write_u8( u8( d.month ))
				buf.write_u8( u8( d.day ))
		case MySQLValue.DateTime( dt ):
			buf.write_u8( 11 )
			with compiler.wrap_arithmetic:
				buf.write_u16le( u16( dt.year ))
				buf.write_u8( u8( dt.month ))
				buf.write_u8( u8( dt.day ))
				buf.write_u8( u8( dt.hour ))
				buf.write_u8( u8( dt.minute ))
				buf.write_u8( u8( dt.second ))
				buf.write_u32le( u32( dt.microsecond ))
		case MySQLValue.Time( td ):
			_write_time_value( buf, td )


_US_PER_DAY_U64:  u64 = 86_400_000_000
_US_PER_HOUR_U64: u64 = 3_600_000_000
_US_PER_MIN_U64:  u64 = 60_000_000
_US_PER_SEC_U64:  u64 = 1_000_000

def _write_time_value( buf: _Buf, td: timedelta ) -> None:
	''' timedelta.total_us (a plain public field - see lib/datetime.py) is
	the exact signed-microsecond total this needs; timedelta's own .days/
	.seconds/.microseconds properties are Python-normalized ("only .days can
	be negative") which is the wrong shape for TIME's is_negative+magnitude
	wire form, so this reads total_us directly instead. '''
	total_us: i64 = td.total_us
	is_neg: bool = total_us < 0
	mag: u64
	with compiler.wrap_arithmetic:
		mag = u64( -total_us ) if is_neg else u64( total_us )
	with compiler.panic_arithmetic( 'divisors are nonzero literals' ):
		days: u64 = mag // _US_PER_DAY_U64
		rem: u64 = mag % _US_PER_DAY_U64
		hour: u64 = rem // _US_PER_HOUR_U64
		rem2: u64 = rem % _US_PER_HOUR_U64
		minute: u64 = rem2 // _US_PER_MIN_U64
		rem3: u64 = rem2 % _US_PER_MIN_U64
		second: u64 = rem3 // _US_PER_SEC_U64
		micro: u64 = rem3 % _US_PER_SEC_U64
	buf.write_u8( 12 )
	buf.write_u8( 1 if is_neg else 0 )
	with compiler.saturate_arithmetic:
		buf.write_u32le( u32( days ))
		buf.write_u8( u8( hour ))
		buf.write_u8( u8( minute ))
		buf.write_u8( u8( second ))
		buf.write_u32le( u32( micro ))


def _read_date_value( r: _Reader ) -> Result[date, MySQLError]:
	length: u8 = r.read_u8().or_return()
	if length == 0:
		# zero date ('0000-00-00') - unrepresentable by lib/datetime.py's
		# date (year must be 1..9999) - accepted v1 gap, see
		# database_client_investigation.md. Requires NO_ZERO_DATE off,
		# which isn't this machine's default sql_mode.
		return Result.Err( MySQLError.ProtocolError )
	year: u16 = r.read_u16le().or_return()
	month: u8 = r.read_u8().or_return()
	day: u8 = r.read_u8().or_return()
	with compiler.wrap_arithmetic:
		return Result.Ok( date( i32( year ), i32( month ), i32( day )).unwrap( 'server-supplied DATE is always valid' ))


def _read_datetime_value( r: _Reader ) -> Result[datetime, MySQLError]:
	length: u8 = r.read_u8().or_return()
	if length == 0:
		return Result.Err( MySQLError.ProtocolError )  # zero datetime - see _read_date_value's own note
	year: u16 = r.read_u16le().or_return()
	month: u8 = r.read_u8().or_return()
	day: u8 = r.read_u8().or_return()
	hour: u8 = 0
	minute: u8 = 0
	second: u8 = 0
	micro: u32 = 0
	if length >= 7:
		hour = r.read_u8().or_return()
		minute = r.read_u8().or_return()
		second = r.read_u8().or_return()
	if length == 11:
		micro = r.read_u32le().or_return()
	with compiler.wrap_arithmetic:
		return Result.Ok( datetime( i32( year ), i32( month ), i32( day ), i32( hour ), i32( minute ), i32( second ), i32( micro ), tzinfo = _NAIVE_TZ ).unwrap( 'server-supplied DATETIME is always valid' ))


def _read_time_value( r: _Reader ) -> Result[timedelta, MySQLError]:
	length: u8 = r.read_u8().or_return()
	if length == 0:
		return Result.Ok( timedelta() )
	is_neg: u8 = r.read_u8().or_return()
	days: u32 = r.read_u32le().or_return()
	hour: u8 = 0
	minute: u8 = 0
	second: u8 = 0
	micro: u32 = 0
	if length >= 8:
		hour = r.read_u8().or_return()
		minute = r.read_u8().or_return()
		second = r.read_u8().or_return()
	if length == 12:
		micro = r.read_u32le().or_return()
	sign: i32 = -1 if is_neg != 0 else 1
	with compiler.wrap_arithmetic:
		return Result.Ok( timedelta( days = sign * i32( days ), hours = sign * i32( hour ), minutes = sign * i32( minute ), seconds = sign * i32( second ), microseconds = sign * i32( micro )))


def _read_int_value( r: _Reader, type_code: u8, unsigned: bool ) -> Result[MySQLValue, MySQLError]:
	if type_code == MYSQL_TYPE_TINY:
		v: u8 = r.read_u8().or_return()
		if unsigned:
			return Result.Ok( MySQLValue.UInt( u64( v )))
		with compiler.wrap_arithmetic:
			return Result.Ok( MySQLValue.Int( i64( i8( v ))))
	if type_code == MYSQL_TYPE_SHORT or type_code == MYSQL_TYPE_YEAR:
		v2: u16 = r.read_u16le().or_return()
		if unsigned or type_code == MYSQL_TYPE_YEAR:
			with compiler.wrap_arithmetic:
				return Result.Ok( MySQLValue.Int( i64( v2 )))
		with compiler.wrap_arithmetic:
			return Result.Ok( MySQLValue.Int( i64( i16( v2 ))))
	if type_code == MYSQL_TYPE_LONG or type_code == MYSQL_TYPE_INT24:
		v3: u32 = r.read_u32le().or_return()
		if unsigned:
			return Result.Ok( MySQLValue.UInt( u64( v3 )))
		with compiler.wrap_arithmetic:
			return Result.Ok( MySQLValue.Int( i64( i32( v3 ))))
	if type_code == MYSQL_TYPE_LONGLONG:
		v4: u64 = r.read_u64le().or_return()
		if unsigned:
			return Result.Ok( MySQLValue.UInt( v4 ))
		with compiler.wrap_arithmetic:
			return Result.Ok( MySQLValue.Int( i64( v4 )))
	return Result.Err( MySQLError.ProtocolError )


def _is_text_family( type_code: u8 ) -> bool:
	return ( type_code == MYSQL_TYPE_VARCHAR or type_code == MYSQL_TYPE_VAR_STRING or type_code == MYSQL_TYPE_STRING
		or type_code == MYSQL_TYPE_TINY_BLOB or type_code == MYSQL_TYPE_MEDIUM_BLOB
		or type_code == MYSQL_TYPE_LONG_BLOB or type_code == MYSQL_TYPE_BLOB )


def _read_column_value( r: _Reader, col: ColumnDescription ) -> Result[MySQLValue, MySQLError]:
	t: u8 = col.type_code
	charset: u16 = col._charset()
	unsigned: bool = ( col._flags() & _COLUMN_UNSIGNED_FLAG ) != 0
	if t == MYSQL_TYPE_TINY or t == MYSQL_TYPE_SHORT or t == MYSQL_TYPE_YEAR or t == MYSQL_TYPE_LONG or t == MYSQL_TYPE_INT24 or t == MYSQL_TYPE_LONGLONG:
		return _read_int_value( r, t, unsigned )
	if t == MYSQL_TYPE_FLOAT:
		return Result.Ok( MySQLValue.Float32( _bits_to_f32( r.read_u32le().or_return() )))
	if t == MYSQL_TYPE_DOUBLE:
		return Result.Ok( MySQLValue.Float64( _bits_to_f64( r.read_u64le().or_return() )))
	if t == MYSQL_TYPE_DECIMAL or t == MYSQL_TYPE_NEWDECIMAL or t == MYSQL_TYPE_ENUM or t == MYSQL_TYPE_SET or t == MYSQL_TYPE_JSON:
		return Result.Ok( MySQLValue.Str( r.read_lenenc_str().or_return() ))
	if _is_text_family( t ):
		if charset == _BINARY_CHARSET_ID:
			return Result.Ok( MySQLValue.Bytes( r.read_lenenc_bytes().or_return() ))
		return Result.Ok( MySQLValue.Str( r.read_lenenc_str().or_return() ))
	if t == MYSQL_TYPE_DATE:
		return Result.Ok( MySQLValue.Date( _read_date_value( r ).or_return() ))
	if t == MYSQL_TYPE_DATETIME or t == MYSQL_TYPE_TIMESTAMP:
		return Result.Ok( MySQLValue.DateTime( _read_datetime_value( r ).or_return() ))
	if t == MYSQL_TYPE_TIME:
		return Result.Ok( MySQLValue.Time( _read_time_value( r ).or_return() ))
	return Result.Err( MySQLError.ProtocolError )


def _parse_binary_row( payload: bytes, cols: list[ColumnDescription] ) -> Result[list[MySQLValue], MySQLError]:
	r = _Reader( payload.get_const_ptr(), len( payload ))
	header: u8 = r.read_u8().or_return()
	if header != 0x00:
		return Result.Err( MySQLError.ProtocolError )
	n: usize = len( cols )
	with compiler.panic_arithmetic( 'divisor is the nonzero literal 8' ):
		bitmap_len: usize = ( n + 7 + 2 ) // 8
	bitmap: bytes = r.read_fixed_bytes( bitmap_len ).or_return()
	bp: ConstPtr[u8] = bitmap.get_const_ptr()
	row: list[MySQLValue] = list[MySQLValue]()
	i: usize = 0
	with compiler.panic_arithmetic( 'i < n, divisor is the nonzero literal 8' ):
		while i < n:
			bit_pos: usize = i + 2
			byte_idx: usize = bit_pos // 8
			bit_idx: usize = bit_pos % 8
			is_null: bool = ( bp[byte_idx] & u8( 1 << bit_idx )) != 0
			if is_null:
				row.append( MySQLValue.Null( None ))
			else:
				col: ColumnDescription = cols.__getitem__( i ).unwrap( 'i < n' )
				row.append( _read_column_value( r, col ).or_return() )
			i += 1
	return Result.Ok( row )


# ---------------------------------------------------------------------------
# Column Definition (41) packet parsing - byte layout confirmed against
# dev.mysql.com's real ColumnDefinition41 docs.
# ---------------------------------------------------------------------------

def _parse_column_def( payload: bytes ) -> Result[ColumnDescription, MySQLError]:
	r = _Reader( payload.get_const_ptr(), len( payload ))
	r.read_lenenc_bytes().or_return()  # catalog - always "def", unused
	r.read_lenenc_bytes().or_return()  # schema
	r.read_lenenc_bytes().or_return()  # table
	r.read_lenenc_bytes().or_return()  # org_table
	name: str = r.read_lenenc_str().or_return()
	r.read_lenenc_bytes().or_return()  # org_name
	r.read_lenenc_int().or_return()    # length_of_fixed_length_fields - fixed 0x0c, unused
	charset: u16 = r.read_u16le().or_return()
	column_length: u32 = r.read_u32le().or_return()
	type_code: u8 = r.read_u8().or_return()
	flags: u16 = r.read_u16le().or_return()
	decimals: u8 = r.read_u8().or_return()
	r.skip( usize( 2 )).or_return()  # reserved
	nullable: bool = ( flags & _COLUMN_NOT_NULL_FLAG ) == 0
	return Result.Ok( ColumnDescription._make( name, type_code, column_length, decimals, nullable, flags, charset ))


def _read_column_defs( sock: socket.Socket, start_seq: u8, count: u64 ) -> Result[tuple[list[ColumnDescription], u8], MySQLError]:
	''' reads `count` Column Definition packets, then the trailing EOF packet
	if count > 0 (CLIENT_DEPRECATE_EOF is never set - see this file's header
	comment - so that EOF always follows, matching both the COM_STMT_PREPARE
	param/column-definition blocks and a real resultset's own header). '''
	cols: list[ColumnDescription] = list[ColumnDescription]()
	seq: u8 = start_seq
	i: u64 = 0
	with compiler.panic_arithmetic( 'i < count, cannot overflow' ):
		while i < count:
			( payload, s ) = _read_packet( sock, seq ).or_return()
			seq = s
			cols.append( _parse_column_def( payload ).or_return() )
			i += 1
	if count > 0:
		( eof_payload, s2 ) = _read_packet( sock, seq ).or_return()
		seq = s2
		_expect_eof( eof_payload ).or_return()
	return Result.Ok(( cols, seq ))


# ---------------------------------------------------------------------------
# Handshake v10 + HandshakeResponse41 + mysql_native_password auth. Byte
# layouts confirmed against dev.mysql.com's real protocol docs (see
# database_client_investigation.md for the fetch citations) - this is the
# highest-risk part of this file (a wrong offset here is silent auth
# corruption, not a compile error), so every field width/order below was
# checked against those docs while writing this, not recalled from memory.
# ---------------------------------------------------------------------------

_AUTH_PLUGIN_NAME: str = 'mysql_native_password'

class _HandshakeInfo:
	seed: bytes  # 20-byte combined auth-plugin-data (part 1 + part 2, trailing NUL dropped)

	@staticmethod
	def _make( seed: bytes ) -> _HandshakeInfo:
		return _HandshakeInfo.__allocate__( seed = seed )

def _parse_handshake_v10( payload: bytes ) -> Result[_HandshakeInfo, MySQLError]:
	r = _Reader( payload.get_const_ptr(), len( payload ))
	protocol_version: u8 = r.read_u8().or_return()
	if protocol_version != 10:
		return Result.Err( MySQLError.ProtocolError )
	r.read_null_term_str().or_return()  # server_version - unused
	r.skip( usize( 4 )).or_return()     # thread_id
	auth_data1: bytes = r.read_fixed_bytes( usize( 8 )).or_return()
	r.skip( usize( 1 )).or_return()     # filler (0x00)
	r.read_u16le().or_return()          # capability_flags_1 - assumed CLIENT_PROTOCOL_41-capable, not branched on
	r.read_u8().or_return()             # character_set
	r.read_u16le().or_return()          # status_flags
	r.read_u16le().or_return()          # capability_flags_2 - unused (v1 never needs to gate on optional server capabilities)
	auth_data_len: u8 = r.read_u8().or_return()
	r.skip( usize( 10 )).or_return()    # reserved, all-zero
	with compiler.saturate_arithmetic:
		part2_len: usize = usize( auth_data_len ) - usize( 8 ) if usize( auth_data_len ) > usize( 8 ) else usize( 13 )
	if part2_len < usize( 13 ):
		part2_len = usize( 13 )
	auth_data2: bytes = r.read_fixed_bytes( part2_len ).or_return()  # includes a trailing NUL byte, dropped below
	plugin_name: str = r.read_null_term_str().or_return()
	if plugin_name != _AUTH_PLUGIN_NAME:
		return Result.Err( MySQLError.UnsupportedAuthPlugin )

	seed_buf = bytearray( usize( 20 ))
	sp: Ptr[u8] = seed_buf.get_ptr()
	sys.memcpy( sp, auth_data1.get_const_ptr(), usize( 8 ))
	with compiler.wrap_arithmetic:
		sys.memcpy( sp + usize( 8 ), auth_data2.get_const_ptr(), usize( 12 ))  # first 12 of part2's 13 bytes - the 13th is the NUL terminator
	seed: bytes = bytes.from_bytearray( move( seed_buf ))
	return Result.Ok( _HandshakeInfo._make( seed ))


def _build_handshake_response( user: str, password: str, database: str|None, seed: bytes ) -> _Buf:
	token: bytes = _scramble_password( password, seed )
	client_flags: u32 = _CLIENT_CAPS
	if database is not None:
		client_flags |= _CLIENT_CONNECT_WITH_DB

	buf = _Buf()
	buf.write_u32le( client_flags )
	buf.write_u32le( 0x01000000 )  # max_packet_size - 16MB, matches this file's own single-packet-payload assumption
	buf.write_u8( 45 )             # character_set - utf8mb4_general_ci
	buf.write_zeros( usize( 23 ))  # filler/reserved
	buf.write_cstr( user )
	with compiler.saturate_arithmetic:
		buf.write_u8( u8( len( token )))
	buf.write_bytes( token.get_const_ptr(), len( token ))
	if database is not None:
		db: str = database
		buf.write_cstr( db )
	buf.write_cstr( _AUTH_PLUGIN_NAME )
	return buf


def _do_handshake_and_auth( sock: socket.Socket, user: str, password: str, database: str|None ) -> Result[None, MySQLError]:
	( hs_payload, seq ) = _read_packet( sock, 0 ).or_return()
	first: u8 = hs_payload.__getitem__( usize( 0 )).unwrap( 'non-empty handshake packet' )
	if first == 0xFF:
		_parse_err_packet( hs_payload ).or_return()
		return Result.Err( MySQLError.AuthFailed )
	info2: _HandshakeInfo = _parse_handshake_v10( hs_payload ).or_return()

	resp: _Buf = _build_handshake_response( user, password, database, info2.seed )
	_write_packet( sock, seq, resp.get_const_ptr(), resp.len() ).or_return()

	with compiler.wrap_arithmetic:
		next_seq: u8 = seq + 1
	( auth_payload, _seq2 ) = _read_packet( sock, next_seq ).or_return()
	result_byte: u8 = auth_payload.__getitem__( usize( 0 )).unwrap( 'non-empty auth response packet' )
	if result_byte == 0x00:
		return Result.Ok( None )
	if result_byte == 0xFF:
		return Result.Err( MySQLError.AuthFailed )
	# AuthSwitchRequest (0xFE) or AuthMoreData (0x01) - both imply the server
	# wants a different plugin (e.g. caching_sha2_password) - explicitly
	# out of v1 scope (design doc §7/§11).
	return Result.Err( MySQLError.UnsupportedAuthPlugin )


class _QueryResult:
	rowcount:    i64
	description: list[ColumnDescription]
	rows:        list[list[MySQLValue]]

	@staticmethod
	def _make( rowcount: i64, description: list[ColumnDescription], rows: list[list[MySQLValue]] ) -> _QueryResult:
		return _QueryResult.__allocate__( rowcount = rowcount, description = description, rows = rows )


# ---------------------------------------------------------------------------
# Connection / Cursor - the public DB-API surface (design doc §5-§10).
#
# threadsafety=0 enforcement (§6): a threading.ThreadLocal[_ThreadOwner] slot
# set once at construction, read by _check_thread() on every public call -
# real, not aspirational (a genuinely different OS thread sees .get() come
# back None, since a ThreadLocal slot is per-thread by construction).
# ThreadLocal[T] requires T to be an RC type (threading.py's own __init__
# panics otherwise) - _ThreadOwner is a trivial one-field RC marker class
# for exactly this, since bool itself doesn't qualify. No raw numeric
# thread-id primitive exists in this codebase's threading.py (checked
# directly) - this sidesteps needing one.
#
# ThreadLocal.set() does NOT incref its argument (confirmed by a real
# double-free crash while testing this: threading.py's set() just casts the
# RC pointer to Ptr[None] and stores it raw) - Connection.__owner_marker
# keeps the SAME _ThreadOwner instance alive for the Connection's whole
# lifetime, since nothing else does. Passing a bare temporary straight into
# set() (no other reference anywhere) gets released the moment set()
# returns, leaving the TLS slot dangling.

class _ThreadOwner:
	present: bool

	@staticmethod
	def _make() -> _ThreadOwner:
		return _ThreadOwner.__allocate__( present = True )
#
# Cursor lifecycle (§7): Cursor holds a normal (strong) reference to its
# Connection - the only way for Cursor.execute()/fetch*() to reach the
# shared socket - and Connection does NOT hold a reference back to any
# Cursor it created. This is deliberately NOT the "Connection walks an
# owned list of Cursors" shape the design doc describes literally: a
# Connection<->Cursor cycle (Cursor -> Connection strong, Connection ->
# Cursor strong) would leak both objects forever under this compiler's
# pure-refcounting model (no cycle collector). The observable contract is
# identical either way - a Cursor whose Connection has been closed always
# reports MySQLError.Closed - because Cursor checks Connection's own
# _is_closed()/_check_thread() on every call rather than a private flag of
# its own being flipped externally. Noted as a resolved design-doc/RC-model
# tension in database_client_investigation.md.
# ---------------------------------------------------------------------------

class Connection:
	__sock:                socket.Socket
	__closed:               bool
	__owner:                threading.ThreadLocal[_ThreadOwner]
	__owner_marker:         _ThreadOwner  # ThreadLocal.set() doesn't take ownership (no incref) - this keeps the SAME object alive for as long as the Connection is, see this file's threadsafety-enforcement header comment
	__last_server_errno:    i32
	__last_sqlstate:        str
	__last_server_message:  str

	def __del__( self ) -> None:
		self._close_impl()

	def _close_impl( self ) -> None:
		if not self.__closed:
			quit_payload = bytearray( usize( 1 ))
			quit_payload.get_ptr()[0] = _COM_QUIT
			_write_packet( self.__sock, 0, quit_payload.get_const_ptr(), usize( 1 )).is_ok()
			self.__sock.close()
			self.__closed = True

	def close( self ) -> Result[None, MySQLError]:
		self._check_thread().or_return()
		self._close_impl()
		return Result.Ok( None )

	@private
	def _check_thread( self ) -> Result[None, MySQLError]:
		if self.__owner.get() is None:
			return Result.Err( MySQLError.WrongThread )
		return Result.Ok( None )

	@private
	def _is_closed( self ) -> bool:
		return self.__closed

	@private
	def _set_err( self, info: _ErrInfo ) -> None:
		self.__last_server_errno = info.errno
		self.__last_sqlstate = info.sqlstate
		self.__last_server_message = info.message

	@property
	def last_server_errno( self ) -> i32:
		return self.__last_server_errno

	@property
	def last_sqlstate( self ) -> str:
		return self.__last_sqlstate

	@property
	def last_server_message( self ) -> str:
		return self.__last_server_message

	@staticmethod
	def connect( host: str, port: u16, user: str, password: str, database: str|None = None ) -> Result[Connection, MySQLError]:
		sock: socket.Socket
		match socket.Socket.tcp():
			case Result.Ok( s ):
				sock = s
			case Result.Err( _ ):
				return Result.Err( MySQLError.ConnectFailed )
		match sock.connect( host, port ):
			case Result.Ok( _ ):
				pass
			case Result.Err( _ ):
				return Result.Err( MySQLError.ConnectFailed )
		_do_handshake_and_auth( sock, user, password, database ).or_return()

		owner: threading.ThreadLocal[_ThreadOwner] = threading.ThreadLocal[_ThreadOwner]()
		marker: _ThreadOwner = _ThreadOwner._make()
		owner.set( marker )
		conn: Connection = Connection.__allocate__(
			__sock = sock, __closed = False, __owner = owner, __owner_marker = marker,
			__last_server_errno = 0, __last_sqlstate = '', __last_server_message = '',
		)
		# autocommit default is OFF (design doc §3) - the server itself
		# defaults to ON, so this actively overrides it right after auth,
		# before returning control to the caller.
		conn._com_query_admin( 'SET autocommit=0' ).or_return()
		return Result.Ok( conn )

	@private
	def _com_query_admin( self, sql: str ) -> Result[None, MySQLError]:
		''' COM_QUERY (text protocol) for a FIXED, zero-caller-data driver
		statement only (SET autocommit=0 / COMMIT / ROLLBACK - see this
		file's header comment). Design doc §10's prepared-statement-only
		rule is about never folding a CALLER-supplied value into query text;
		none of these three statements ever contain one, so this doesn't
		reopen the injection surface §10 closes for Cursor.execute(). '''
		payload = _Buf()
		payload.write_u8( _COM_QUERY )
		payload.write_bytes( sql.get_const_ptr(), sql.byte_len() )
		_write_packet( self.__sock, 0, payload.get_const_ptr(), payload.len() ).or_return()
		( resp, _seq ) = _read_packet( self.__sock, 1 ).or_return()
		first: u8 = resp.__getitem__( usize( 0 )).unwrap( 'non-empty admin-query response' )
		if first == 0x00:
			return Result.Ok( None )
		if first == 0xFF:
			info: _ErrInfo = _parse_err_packet( resp ).or_return()
			self._set_err( info )
			return Result.Err( MySQLError.QueryError )
		return Result.Err( MySQLError.ProtocolError )

	def commit( self ) -> Result[None, MySQLError]:
		''' always forwarded to the server as a real COMMIT, even with
		nothing pending (design doc §8 - autocommit is always off, so
		there's always some transaction open, even an empty one). '''
		self._check_thread().or_return()
		if self.__closed:
			return Result.Err( MySQLError.Closed )
		return self._com_query_admin( 'COMMIT' )

	def rollback( self ) -> Result[None, MySQLError]:
		self._check_thread().or_return()
		if self.__closed:
			return Result.Err( MySQLError.Closed )
		return self._com_query_admin( 'ROLLBACK' )

	def cursor( self ) -> Result[Cursor, MySQLError]:
		self._check_thread().or_return()
		if self.__closed:
			return Result.Err( MySQLError.Closed )
		return Result.Ok( Cursor._make( self ))

	@private
	def _run_prepared( self, sql: str, params: list[MySQLValue] ) -> Result[_QueryResult, MySQLError]:
		''' COM_STMT_PREPARE -> COM_STMT_EXECUTE -> COM_STMT_CLOSE, for EVERY
		call (design doc §10 - no cached/reused prepared statements across
		execute() calls, even a zero-param query goes through this same
		single path). '''
		prep = _Buf()
		prep.write_u8( _COM_STMT_PREPARE )
		prep.write_bytes( sql.get_const_ptr(), sql.byte_len() )
		_write_packet( self.__sock, 0, prep.get_const_ptr(), prep.len() ).or_return()
		( prep_resp, prep_seq ) = _read_packet( self.__sock, 1 ).or_return()
		prep_first: u8 = prep_resp.__getitem__( usize( 0 )).unwrap( 'non-empty COM_STMT_PREPARE response' )
		if prep_first == 0xFF:
			info: _ErrInfo = _parse_err_packet( prep_resp ).or_return()
			self._set_err( info )
			return Result.Err( MySQLError.QueryError )
		if prep_first != 0x00:
			return Result.Err( MySQLError.ProtocolError )

		pr = _Reader( prep_resp.get_const_ptr(), len( prep_resp ))
		pr.skip( usize( 1 )).or_return()
		stmt_id: u32 = pr.read_u32le().or_return()
		num_columns: u16 = pr.read_u16le().or_return()
		num_params: u16 = pr.read_u16le().or_return()

		( _param_defs, seq_a ) = _read_column_defs( self.__sock, prep_seq, u64( num_params )).or_return()
		( col_defs, _seq_b ) = _read_column_defs( self.__sock, seq_a, u64( num_columns )).or_return()

		if usize( num_params ) != len( params ):
			_send_stmt_close( self.__sock, stmt_id ).or_return()
			return Result.Err( MySQLError.TypeMismatch )

		exec_buf: _Buf = _build_execute_payload( stmt_id, params )
		_write_packet( self.__sock, 0, exec_buf.get_const_ptr(), exec_buf.len() ).or_return()
		( exec_resp, exec_seq ) = _read_packet( self.__sock, 1 ).or_return()
		exec_first: u8 = exec_resp.__getitem__( usize( 0 )).unwrap( 'non-empty COM_STMT_EXECUTE response' )

		if exec_first == 0xFF:
			info2: _ErrInfo = _parse_err_packet( exec_resp ).or_return()
			self._set_err( info2 )
			_send_stmt_close( self.__sock, stmt_id ).or_return()
			return Result.Err( MySQLError.QueryError )

		if exec_first == 0x00:
			ok: _OkResult = _parse_ok_packet( exec_resp ).or_return()
			_send_stmt_close( self.__sock, stmt_id ).or_return()
			with compiler.wrap_arithmetic:
				rc: i64 = i64( ok.affected_rows )
			return Result.Ok( _QueryResult._make( rc, col_defs, list[list[MySQLValue]]() ))

		rr = _Reader( exec_resp.get_const_ptr(), len( exec_resp ))
		col_count: u64 = rr.read_lenenc_int().or_return()
		( result_cols, seq_c ) = _read_column_defs( self.__sock, exec_seq, col_count ).or_return()

		rows: list[list[MySQLValue]] = list[list[MySQLValue]]()
		seq_d: u8 = seq_c
		while True:
			( row_payload, seq_e ) = _read_packet( self.__sock, seq_d ).or_return()
			seq_d = seq_e
			row_first: u8 = row_payload.__getitem__( usize( 0 )).unwrap( 'non-empty row/EOF packet' )
			if row_first == 0xFE and len( row_payload ) < usize( 9 ):
				break
			if row_first == 0xFF:
				info3: _ErrInfo = _parse_err_packet( row_payload ).or_return()
				self._set_err( info3 )
				_send_stmt_close( self.__sock, stmt_id ).or_return()
				return Result.Err( MySQLError.QueryError )
			rows.append( _parse_binary_row( row_payload, result_cols ).or_return() )

		_send_stmt_close( self.__sock, stmt_id ).or_return()
		with compiler.wrap_arithmetic:
			row_count: i64 = i64( len( rows ))
		return Result.Ok( _QueryResult._make( row_count, result_cols, rows ))


class Cursor:
	__conn:        Connection
	__closed:      bool
	__rowcount:    i64
	__description: list[ColumnDescription]
	__rows:        list[list[MySQLValue]]
	__pos:         usize

	@private
	@staticmethod
	def _make( conn: Connection ) -> Cursor:
		return Cursor.__allocate__(
			__conn = conn, __closed = False, __rowcount = -1,
			__description = list[ColumnDescription](), __rows = list[list[MySQLValue]](), __pos = 0,
		)

	def close( self ) -> Result[None, MySQLError]:
		self.__conn._check_thread().or_return()
		self.__closed = True
		return Result.Ok( None )

	@property
	def rowcount( self ) -> i64:
		return self.__rowcount

	@property
	def description( self ) -> list[ColumnDescription]:
		return self.__description

	def _guard( self ) -> Result[None, MySQLError]:
		self.__conn._check_thread().or_return()
		if self.__closed or self.__conn._is_closed():
			return Result.Err( MySQLError.Closed )
		return Result.Ok( None )

	def execute( self, sql: str, params: list[MySQLValue] ) -> Result[None, MySQLError]:
		self._guard().or_return()
		result: _QueryResult = self.__conn._run_prepared( sql, params ).or_return()
		self.__rowcount = result.rowcount
		self.__description = result.description
		self.__rows = result.rows
		self.__pos = 0
		return Result.Ok( None )

	def fetchone( self ) -> Result[list[MySQLValue]|None, MySQLError]:
		self._guard().or_return()
		if self.__pos >= len( self.__rows ):
			return Result.Ok( None )
		row: list[MySQLValue] = self.__rows.__getitem__( self.__pos ).unwrap( 'pos < len(rows)' )
		with compiler.wrap_arithmetic:
			self.__pos += usize( 1 )
		return Result.Ok( row )

	def fetchall( self ) -> Result[list[list[MySQLValue]], MySQLError]:
		self._guard().or_return()
		out: list[list[MySQLValue]] = list[list[MySQLValue]]()
		while self.__pos < len( self.__rows ):
			out.append( self.__rows.__getitem__( self.__pos ).unwrap( 'pos < len(rows)' ))
			with compiler.wrap_arithmetic:
				self.__pos += usize( 1 )
		return Result.Ok( out )
