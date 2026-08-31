# lib/mysql/protocol.py - MySQL/MariaDB wire protocol: packet framing,
# handshake v10 + mysql_native_password auth, COM_STMT_PREPARE/EXECUTE/CLOSE,
# OK/ERR/column-definition parsing, binary-protocol row encode/decode.
#
# See database_client_investigation.md for the full design (this file
# implements SS1/SS9/SS10 of that doc). Byte layouts below were looked up
# against dev.mysql.com/doc/dev/mysql-server (Protocol::HandshakeV10,
# Protocol::HandshakeResponse41, COM_STMT_PREPARE, COM_STMT_EXECUTE,
# OK_Packet/ERR_Packet, Binary Protocol Resultset Row, length-encoded
# integer rules), not reconstructed from memory - a wrong byte offset here
# is a silent-corruption bug, not a compile error.
#
# No CLIENT_SSL/TLS upgrade in this file (see doc SS11's open-items list) -
# every connection in v1 is plain TCP. mysql_native_password only -
# caching_sha2_password (MySQL 8's default) is explicitly out of scope.

import compiler
import sys
from socket import Socket
from sha1 import sha1
from datetime import date, datetime, timedelta, DateError
from zoneinfo import ZoneInfo


# ---------------------------------------------------------------------------
# Client capability flags (mysql_com.h - stable, well-known values)
# ---------------------------------------------------------------------------

CLIENT_LONG_PASSWORD:                  u32 = 0x00000001
CLIENT_FOUND_ROWS:                     u32 = 0x00000002  # never set - see doc SS4
CLIENT_CONNECT_WITH_DB:                u32 = 0x00000008
CLIENT_PROTOCOL_41:                    u32 = 0x00000200
CLIENT_SSL:                            u32 = 0x00000800
CLIENT_TRANSACTIONS:                   u32 = 0x00002000
CLIENT_SECURE_CONNECTION:              u32 = 0x00008000
CLIENT_MULTI_RESULTS:                  u32 = 0x00020000
CLIENT_PLUGIN_AUTH:                    u32 = 0x00080000

_CLIENT_FLAGS: u32 = ( CLIENT_LONG_PASSWORD | CLIENT_PROTOCOL_41 | CLIENT_SECURE_CONNECTION
	| CLIENT_TRANSACTIONS | CLIENT_MULTI_RESULTS | CLIENT_PLUGIN_AUTH )

_UTF8MB4_GENERAL_CI: u8 = 45


# ---------------------------------------------------------------------------
# MYSQL_TYPE_* binary-protocol type codes (mysql_com.h enum_field_types)
# ---------------------------------------------------------------------------

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

_COL_FLAG_UNSIGNED: u16 = 0x0020
_BINARY_CHARSET:    u16 = 63  # "binary" collation id - distinguishes BLOB from TEXT


@enum( i32 )
class MySQLError:
	ConnectFailed         = 1
	TLSFailed             = 2
	AuthFailed            = 3
	UnsupportedAuthPlugin = 4
	ProtocolError         = 5
	QueryError            = 6
	TypeMismatch          = 7
	Closed                = 8
	WrongThread           = 9
	Other = _


def _map_os( e: OSError ) -> MySQLError:
	return MySQLError.ConnectFailed


# ---------------------------------------------------------------------------
# Thread identity - for threadsafety=0 enforcement (doc SS6). pthread_self()
# on POSIX is opaque (not necessarily the kernel TID) but stable/unique per
# thread for the lifetime of the thread, which is all a same-thread identity
# check needs - never displayed/logged, only compared.
# ---------------------------------------------------------------------------

if compiler.target.os == 'windows':
	@extern( 'kernel32', 'GetCurrentThreadId' )
	def _get_current_thread_id_raw() -> u32:
		...

	def current_thread_id() -> usize:
		with compiler.wrap_arithmetic:
			return usize( _get_current_thread_id_raw() )
else:
	@extern( 'c', 'pthread_self' )
	def _pthread_self_raw() -> usize:
		...

	def current_thread_id() -> usize:
		return _pthread_self_raw()


# ---------------------------------------------------------------------------
# _ByteBuf - growable write buffer for building packet payloads. Same grow/
# shape as lib/ssl.py's _CipherBuf / lib/socket.py's RecvBuffer.
# ---------------------------------------------------------------------------

class _ByteBuf:
	__data: Ptr[u8]
	__len:  usize
	__cap:  usize

	def __init__( self, initial_cap: usize = 64 ) -> None:
		self.__cap = initial_cap
		self.__data = sys.alloc[u8]( self.__cap )
		self.__len = 0

	def __del__( self ) -> None:
		sys.free( self.__data )

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

	def append_u8( self, v: u8 ) -> None:
		self._grow( 1 )
		with compiler.wrap_arithmetic:
			self.__data[self.__len] = v
			self.__len += 1

	def append_u16_le( self, v: u16 ) -> None:
		with compiler.wrap_arithmetic:
			self.append_u8( u8( v & 0xFF ))
			self.append_u8( u8(( v >> 8 ) & 0xFF ))

	def append_u24_le( self, v: u32 ) -> None:
		with compiler.wrap_arithmetic:
			self.append_u8( u8( v & 0xFF ))
			self.append_u8( u8(( v >> 8 ) & 0xFF ))
			self.append_u8( u8(( v >> 16 ) & 0xFF ))

	def append_u32_le( self, v: u32 ) -> None:
		with compiler.wrap_arithmetic:
			self.append_u8( u8( v & 0xFF ))
			self.append_u8( u8(( v >> 8 ) & 0xFF ))
			self.append_u8( u8(( v >> 16 ) & 0xFF ))
			self.append_u8( u8(( v >> 24 ) & 0xFF ))

	def append_u64_le( self, v: u64 ) -> None:
		with compiler.wrap_arithmetic:
			i: u64 = 0
			while i < 8:
				self.append_u8( u8(( v >> ( i * 8 )) & 0xFF ))
				i += 1

	def append_bytes_raw( self, ptr: ConstPtr[u8], count: usize ) -> None:
		self._grow( count )
		with compiler.wrap_arithmetic:
			dest: Ptr[u8] = self.__data + self.__len
		sys.memcpy( dest, ptr, count )
		with compiler.wrap_arithmetic:
			self.__len += count

	def append_bytes( self, data: bytes|bytearray ) -> None:
		self.append_bytes_raw( data.get_const_ptr(), len( data ))

	def append_cstr( self, s: str ) -> None:
		''' raw utf8 bytes + a NUL terminator - NOT null-safe (a NUL byte
		inside s would truncate the field on the wire); fine here since this
		is only ever used for driver-controlled or already-validated
		identifiers (username/database), never an arbitrary query value. '''
		encoded: bytes = s.encode().unwrap( 'driver-controlled string is valid utf8' )
		self.append_bytes( encoded )
		self.append_u8( 0 )

	def append_lenenc_int( self, v: u64 ) -> None:
		if v < u64( 251 ):
			with compiler.panic_arithmetic( 'v < 251, fits u8 by construction' ):
				self.append_u8( u8( v ))
		elif v < u64( 65536 ):
			self.append_u8( 0xFC )
			with compiler.wrap_arithmetic:
				self.append_u16_le( u16( v ))
		elif v < u64( 16777216 ):
			self.append_u8( 0xFD )
			with compiler.wrap_arithmetic:
				self.append_u24_le( u32( v ))
		else:
			self.append_u8( 0xFE )
			self.append_u64_le( v )

	def append_lenenc_bytes( self, data: bytes|bytearray ) -> None:
		with compiler.wrap_arithmetic:
			self.append_lenenc_int( u64( len( data )))
		self.append_bytes( data )

	def to_bytes( self ) -> bytes:
		out = bytearray( self.__len )
		sys.memcpy( out.get_ptr(), self.__data, self.__len )
		return bytes.from_bytearray( move( out ))


# ---------------------------------------------------------------------------
# _PacketReader - read cursor over one already-fully-buffered packet payload.
# ---------------------------------------------------------------------------

class _PacketReader:
	__data: bytes
	__pos:  usize

	def __init__( self, data: bytes ) -> None:
		self.__data = data
		self.__pos = 0

	def remaining( self ) -> usize:
		with compiler.panic_arithmetic( 'pos never exceeds data length - every read below checks first' ):
			return len( self.__data ) - self.__pos

	def peek_u8( self ) -> Result[u8, MySQLError]:
		if self.remaining() < usize( 1 ):
			return Result.Err( MySQLError.ProtocolError )
		return self.__data.__getitem__( self.__pos ).or_return( _idx_to_protocol_err )

	def read_u8( self ) -> Result[u8, MySQLError]:
		v: u8 = self.peek_u8().or_return()
		with compiler.wrap_arithmetic:
			self.__pos += 1
		return Result.Ok( v )

	def read_u16_le( self ) -> Result[u16, MySQLError]:
		lo: u8 = self.read_u8().or_return()
		hi: u8 = self.read_u8().or_return()
		with compiler.wrap_arithmetic:
			return Result.Ok( u16( lo ) | ( u16( hi ) << 8 ))

	def read_u24_le( self ) -> Result[u32, MySQLError]:
		b0: u8 = self.read_u8().or_return()
		b1: u8 = self.read_u8().or_return()
		b2: u8 = self.read_u8().or_return()
		with compiler.wrap_arithmetic:
			return Result.Ok( u32( b0 ) | ( u32( b1 ) << 8 ) | ( u32( b2 ) << 16 ))

	def read_u32_le( self ) -> Result[u32, MySQLError]:
		b0: u8 = self.read_u8().or_return()
		b1: u8 = self.read_u8().or_return()
		b2: u8 = self.read_u8().or_return()
		b3: u8 = self.read_u8().or_return()
		with compiler.wrap_arithmetic:
			return Result.Ok( u32( b0 ) | ( u32( b1 ) << 8 ) | ( u32( b2 ) << 16 ) | ( u32( b3 ) << 24 ))

	def read_u64_le( self ) -> Result[u64, MySQLError]:
		with compiler.wrap_arithmetic:
			v: u64 = 0
			i: u64 = 0
			while i < 8:
				b: u8 = self.read_u8().or_return()
				v = v | ( u64( b ) << ( i * 8 ))
				i += 1
			return Result.Ok( v )

	def read_bytes( self, n: usize ) -> Result[bytes, MySQLError]:
		if self.remaining() < n:
			return Result.Err( MySQLError.ProtocolError )
		out = bytearray( n )
		with compiler.wrap_arithmetic:
			src: ConstPtr[u8] = self.__data.get_const_ptr() + self.__pos
		sys.memcpy( out.get_ptr(), src, n )
		with compiler.wrap_arithmetic:
			self.__pos += n
		return Result.Ok( bytes.from_bytearray( move( out )))

	def read_rest( self ) -> bytes:
		n: usize = self.remaining()
		return self.read_bytes( n ).unwrap( 'remaining() bytes are always readable' )

	def skip( self, n: usize ) -> Result[None, MySQLError]:
		if self.remaining() < n:
			return Result.Err( MySQLError.ProtocolError )
		with compiler.wrap_arithmetic:
			self.__pos += n
		return Result.Ok( None )

	def read_null_str( self ) -> Result[str, MySQLError]:
		start: usize = self.__pos
		with compiler.wrap_arithmetic:
			data_len: usize = len( self.__data )
			p: usize = start
			while p < data_len:
				b: u8 = self.__data.__getitem__( p ).unwrap( 'p < data_len, just checked' )
				if b == 0:
					break
				p += 1
		if p >= data_len:
			return Result.Err( MySQLError.ProtocolError )
		with compiler.panic_arithmetic( 'p >= start, computed by the scan above' ):
			piece_len: usize = p - start
		text: bytes = self.read_bytes( piece_len ).or_return()
		self.skip( usize( 1 )).or_return()  # the NUL terminator itself
		return Result.Ok( text.decode_lossy() )

	def read_lenenc_int( self ) -> Result[u64|None, MySQLError]:
		''' None represents the wire's 0xFB NULL marker - only meaningful
		when this lenenc appears as a ROW VALUE length prefix, never for a
		lenenc used as a plain count (num columns, etc) - callers that only
		expect a count should treat a None here as MySQLError.ProtocolError. '''
		first: u8 = self.read_u8().or_return()
		if first < 251:
			with compiler.wrap_arithmetic:
				return Result.Ok( u64( first ))
		if first == 0xFB:
			return Result.Ok( None )
		if first == 0xFC:
			v: u16 = self.read_u16_le().or_return()
			with compiler.wrap_arithmetic:
				return Result.Ok( u64( v ))
		if first == 0xFD:
			v3: u32 = self.read_u24_le().or_return()
			with compiler.wrap_arithmetic:
				return Result.Ok( u64( v3 ))
		if first == 0xFE:
			v8: u64 = self.read_u64_le().or_return()
			return Result.Ok( v8 )
		return Result.Err( MySQLError.ProtocolError )

	def read_lenenc_count( self ) -> Result[u64, MySQLError]:
		v: u64|None = self.read_lenenc_int().or_return()
		if v is None:
			return Result.Err( MySQLError.ProtocolError )
		return Result.Ok( v )

	def read_lenenc_bytes( self ) -> Result[bytes, MySQLError]:
		n: u64 = self.read_lenenc_count().or_return()
		with compiler.saturate_arithmetic:
			nn: usize = usize( n )
		return self.read_bytes( nn )

	def read_lenenc_str( self ) -> Result[str, MySQLError]:
		data: bytes = self.read_lenenc_bytes().or_return()
		return Result.Ok( data.decode_lossy() )


def _idx_to_protocol_err( e: IndexError ) -> MySQLError:
	return MySQLError.ProtocolError


def bytes_decode_lossy( data: bytes ) -> str:
	return data.decode_lossy()


# ---------------------------------------------------------------------------
# Packet framing - 3-byte little-endian length + 1-byte sequence id + payload.
# A payload >= 0xFFFFFF is split across several physical packets (the same
# sequence-id counter keeps incrementing per physical packet); read_packet
# reassembles a multi-packet payload back into one buffer transparently.
# ---------------------------------------------------------------------------

def _send_all( sock: Socket, ptr: ConstPtr[u8], n: usize ) -> Result[None, MySQLError]:
	match sock.send_all( ptr, n ):
		case Result.Ok( _ ):
			return Result.Ok( None )
		case Result.Err( _ ):
			return Result.Err( MySQLError.ConnectFailed )


def _recv_exact( sock: Socket, n: usize ) -> Result[bytes, MySQLError]:
	buf = bytearray( n )
	ptr: Ptr[u8] = buf.get_ptr()
	got: usize = 0
	with compiler.wrap_arithmetic:
		while got < n:
			dst: Ptr[u8] = ptr + got
			room: usize = n - got
			m: usize = sock.recv( dst, room ).or_return( _map_os )
			if m == 0:
				return Result.Err( MySQLError.ConnectFailed )  # peer closed mid-packet
			got += m
	return Result.Ok( bytes.from_bytearray( move( buf )))


def read_packet( sock: Socket ) -> Result[tuple[u8, bytes], MySQLError]:
	payload_buf: _ByteBuf = _ByteBuf()
	last_seq: u8 = 0
	with compiler.wrap_arithmetic:
		while True:
			header: bytes = _recv_exact( sock, usize( 4 )).or_return()
			hr: _PacketReader = _PacketReader( header )
			length: u32 = hr.read_u24_le().or_return()
			seq: u8 = hr.read_u8().or_return()
			last_seq = seq
			with compiler.saturate_arithmetic:
				length_usize: usize = usize( length )
			if length_usize > 0:
				body: bytes = _recv_exact( sock, length_usize ).or_return()
				payload_buf.append_bytes( body )
			if length != 0xFFFFFF:
				break
	return Result.Ok(( last_seq, payload_buf.to_bytes() ))


def send_packet( sock: Socket, seq: u8, payload: bytes ) -> Result[u8, MySQLError]:
	''' sends payload as one or more physical packets; returns the sequence
	id the NEXT packet in this exchange should use. '''
	total: usize = len( payload )
	ptr: ConstPtr[u8] = payload.get_const_ptr()
	offset: usize = 0
	cur_seq: u8 = seq
	with compiler.wrap_arithmetic:
		while True:
			remaining: usize = total - offset
			chunk: usize = remaining if remaining < usize( 0xFFFFFF ) else usize( 0xFFFFFF )
			hdr: _ByteBuf = _ByteBuf( 4 )
			hdr.append_u24_le( u32( chunk ))
			hdr.append_u8( cur_seq )
			hdr_bytes: bytes = hdr.to_bytes()
			_send_all( sock, hdr_bytes.get_const_ptr(), usize( 4 )).or_return()
			if chunk > 0:
				chunk_ptr: ConstPtr[u8] = ptr + offset
				_send_all( sock, chunk_ptr, chunk ).or_return()
			offset += chunk
			cur_seq += 1
			if chunk < usize( 0xFFFFFF ):
				break
	return Result.Ok( cur_seq )


# ---------------------------------------------------------------------------
# Handshake v10 + mysql_native_password auth (doc SS1/SS9)
# ---------------------------------------------------------------------------

class HandshakeInfo:
	auth_plugin_data: bytes  # the full scramble, 20 bytes
	auth_plugin_name: str
	capabilities:      u32

	def __init__( self, auth_plugin_data: bytes, auth_plugin_name: str, capabilities: u32 ) -> None:
		self.auth_plugin_data = auth_plugin_data
		self.auth_plugin_name = auth_plugin_name
		self.capabilities = capabilities


def parse_handshake_v10( payload: bytes ) -> Result[HandshakeInfo, MySQLError]:
	r: _PacketReader = _PacketReader( payload )
	proto_version: u8 = r.read_u8().or_return()
	if proto_version != 10:
		return Result.Err( MySQLError.ProtocolError )
	r.read_null_str().or_return()  # server_version - informational only
	r.skip( usize( 4 )).or_return()  # thread id
	scramble1: bytes = r.read_bytes( usize( 8 )).or_return()
	r.skip( usize( 1 )).or_return()  # filler (0x00)
	cap_lo: u16 = r.read_u16_le().or_return()
	with compiler.wrap_arithmetic:
		capabilities: u32 = u32( cap_lo )
	if r.remaining() == usize( 0 ):
		# pre-4.1 server - not supported (no CLIENT_PROTOCOL_41 to negotiate)
		return Result.Err( MySQLError.ProtocolError )
	r.skip( usize( 1 )).or_return()  # character_set
	r.skip( usize( 2 )).or_return()  # status_flags
	cap_hi: u16 = r.read_u16_le().or_return()
	with compiler.wrap_arithmetic:
		capabilities = capabilities | ( u32( cap_hi ) << 16 )
	auth_data_len: u8 = r.read_u8().or_return()
	r.skip( usize( 10 )).or_return()  # reserved

	plugin_name: str = ''
	scramble2: bytes = bytes.from_bytearray( move( bytearray( 0 )))
	if ( capabilities & CLIENT_PLUGIN_AUTH ) != 0:
		with compiler.saturate_arithmetic:
			part2_len_i: i32 = i32( auth_data_len ) - 8
		part2_len: usize = usize( 13 )
		if part2_len_i > 13:
			with compiler.wrap_arithmetic:
				part2_len = usize( part2_len_i )
		scramble2 = r.read_bytes( part2_len ).or_return()
		plugin_name = r.read_null_str().or_return()
	else:
		return Result.Err( MySQLError.UnsupportedAuthPlugin )

	# scramble2's last byte is a 0x00 pad byte (part2_len is MAX(13, len-8),
	# and the true scramble is always exactly 20 bytes: 8 + 12) - drop it.
	with compiler.panic_arithmetic( 'scramble2 always has at least 1 byte (part2_len >= 13)' ):
		scramble2_trimmed: bytes = scramble2[0:len( scramble2 ) - usize( 1 )]

	full_scramble: _ByteBuf = _ByteBuf( 20 )
	full_scramble.append_bytes( scramble1 )
	full_scramble.append_bytes( scramble2_trimmed )

	return Result.Ok( HandshakeInfo( full_scramble.to_bytes(), plugin_name, capabilities ))


def scramble_native_password( password: bytes, seed: bytes ) -> bytes:
	''' mysql_native_password's challenge-response:
	SHA1(password) XOR SHA1(seed + SHA1(SHA1(password)))
	(dev.mysql.com's Native Authentication page - verified via WebFetch, not
	reconstructed from memory, since a wrong formula here silently breaks
	auth against every server rather than failing loudly). An empty
	password sends an empty auth-response (no scrambling at all) - matches
	every real client's own special-case for a passwordless account. '''
	if len( password ) == 0:
		return bytes.from_bytearray( move( bytearray( 0 )))

	stage1: bytes = sha1( password )               # SHA1(password)
	stage2: bytes = sha1( stage1 )                  # SHA1(SHA1(password))

	combined: _ByteBuf = _ByteBuf( 40 )
	combined.append_bytes( seed )
	combined.append_bytes( stage2 )
	stage3: bytes = sha1( combined.to_bytes() )     # SHA1(seed + SHA1(SHA1(password)))

	out = bytearray( usize( 20 ))
	out_ptr: Ptr[u8] = out.get_ptr()
	i: usize = 0
	with compiler.panic_arithmetic( 'both operands are always exactly 20 bytes' ):
		while i < usize( 20 ):
			a: u8 = stage1.__getitem__( i ).unwrap( 'i < 20' )
			b: u8 = stage3.__getitem__( i ).unwrap( 'i < 20' )
			out_ptr[i] = a ^ b
			i += 1
	return bytes.from_bytearray( move( out ))


def build_handshake_response41( user: str, auth_response: bytes, database: str ) -> bytes:
	flags: u32 = _CLIENT_FLAGS
	if len( database ) > 0:
		flags = flags | CLIENT_CONNECT_WITH_DB

	buf: _ByteBuf = _ByteBuf( 128 )
	buf.append_u32_le( flags )
	buf.append_u32_le( 0x01000000 )  # max_packet_size - 16MB
	buf.append_u8( _UTF8MB4_GENERAL_CI )
	i: i32 = 0
	with compiler.panic_arithmetic( 'i < 23, cannot overflow i32' ):
		while i < 23:
			buf.append_u8( 0 )  # filler
			i += 1
	buf.append_cstr( user )
	# CLIENT_SECURE_CONNECTION (always set here) means a 1-byte length prefix,
	# not the lenenc form CLIENT_PLUGIN_AUTH_LENENC_CLIENT_DATA would need -
	# fine since the scramble is always exactly 20 bytes, well under 251.
	with compiler.wrap_arithmetic:
		buf.append_u8( u8( len( auth_response )))
	buf.append_bytes( auth_response )
	if len( database ) > 0:
		buf.append_cstr( database )
	buf.append_cstr( 'mysql_native_password' )
	return buf.to_bytes()


# ---------------------------------------------------------------------------
# OK_Packet / ERR_Packet (doc SS5's .last_server_errno/.last_sqlstate/
# .last_server_message come from parse_err_packet's ErrInfo)
# ---------------------------------------------------------------------------

class OkPacket:
	affected_rows:  u64
	last_insert_id: u64
	status_flags:   u16
	warnings:       u16

	def __init__( self, affected_rows: u64, last_insert_id: u64, status_flags: u16, warnings: u16 ) -> None:
		self.affected_rows = affected_rows
		self.last_insert_id = last_insert_id
		self.status_flags = status_flags
		self.warnings = warnings


def parse_ok_packet( payload: bytes ) -> Result[OkPacket, MySQLError]:
	r: _PacketReader = _PacketReader( payload )
	header: u8 = r.read_u8().or_return()
	if header != 0x00 and header != 0xFE:
		return Result.Err( MySQLError.ProtocolError )
	affected_rows: u64 = r.read_lenenc_count().or_return()
	last_insert_id: u64 = r.read_lenenc_count().or_return()
	status_flags: u16 = r.read_u16_le().or_return()
	warnings: u16 = r.read_u16_le().or_return()
	return Result.Ok( OkPacket( affected_rows, last_insert_id, status_flags, warnings ))


class ErrInfo:
	code:     u16
	sqlstate: str
	message:  str

	def __init__( self, code: u16, sqlstate: str, message: str ) -> None:
		self.code = code
		self.sqlstate = sqlstate
		self.message = message


def parse_err_packet( payload: bytes ) -> Result[ErrInfo, MySQLError]:
	r: _PacketReader = _PacketReader( payload )
	header: u8 = r.read_u8().or_return()
	if header != 0xFF:
		return Result.Err( MySQLError.ProtocolError )
	code: u16 = r.read_u16_le().or_return()
	r.skip( usize( 1 )).or_return()  # '#' sql_state_marker (CLIENT_PROTOCOL_41 always negotiated here)
	sqlstate_bytes: bytes = r.read_bytes( usize( 5 )).or_return()
	message_bytes: bytes = r.read_rest()
	return Result.Ok( ErrInfo( code, sqlstate_bytes.decode_lossy(), message_bytes.decode_lossy() ))


# ---------------------------------------------------------------------------
# Column definition (Protocol::ColumnDefinition41) - used both for prepared-
# statement parameter placeholders (values discarded, only the count
# matters) and for result-set columns (name/type/flags/charset kept for
# row decoding + cursor.description).
# ---------------------------------------------------------------------------

class ColumnDef:
	name:     str
	col_type: u8
	flags:    u16
	charset:  u16
	decimals: u8

	def __init__( self, name: str, col_type: u8, flags: u16, charset: u16, decimals: u8 ) -> None:
		self.name = name
		self.col_type = col_type
		self.flags = flags
		self.charset = charset
		self.decimals = decimals

	@property
	def is_unsigned( self ) -> bool:
		return ( self.flags & _COL_FLAG_UNSIGNED ) != 0


def parse_column_def( payload: bytes ) -> Result[ColumnDef, MySQLError]:
	r: _PacketReader = _PacketReader( payload )
	r.read_lenenc_bytes().or_return()  # catalog
	r.read_lenenc_bytes().or_return()  # schema
	r.read_lenenc_bytes().or_return()  # table
	r.read_lenenc_bytes().or_return()  # org_table
	name_bytes: bytes = r.read_lenenc_bytes().or_return()
	r.read_lenenc_bytes().or_return()  # org_name
	r.read_u8().or_return()            # length-of-fixed-fields, always 0x0c
	charset: u16 = r.read_u16_le().or_return()
	r.skip( usize( 4 )).or_return()    # column_length
	col_type: u8 = r.read_u8().or_return()
	flags: u16 = r.read_u16_le().or_return()
	decimals: u8 = r.read_u8().or_return()
	return Result.Ok( ColumnDef( name_bytes.decode_lossy(), col_type, flags, charset, decimals ))


# ---------------------------------------------------------------------------
# Float/double bit reinterpretation - MYSQL_TYPE_FLOAT/DOUBLE are IEEE-754
# bit patterns on the wire, not a value conversion (5u32 must become the
# 4-byte pattern of 5.0f32, not a widening 5->5.0 numeric cast). No bitcast
# intrinsic exists in this compiler - compiler.addrof() + a pointer-type
# compiler.cast() is the established trick (lib/socket.py's own
# _build_sockaddr_in uses the identical pattern to punch a u32 through
# inet_pton as a byte buffer).
# ---------------------------------------------------------------------------

def _f32_from_bits( bits: u32 ) -> f32:
	b: u32 = bits
	p: Ptr[f32] = compiler.cast( Ptr[f32], compiler.addrof( b ))
	return p[0]

def _bits_from_f32( v: f32 ) -> u32:
	f: f32 = v
	p: Ptr[u32] = compiler.cast( Ptr[u32], compiler.addrof( f ))
	return p[0]

def _f64_from_bits( bits: u64 ) -> f64:
	b: u64 = bits
	p: Ptr[f64] = compiler.cast( Ptr[f64], compiler.addrof( b ))
	return p[0]

def _bits_from_f64( v: f64 ) -> u64:
	f: f64 = v
	p: Ptr[u64] = compiler.cast( Ptr[u64], compiler.addrof( f ))
	return p[0]


# ---------------------------------------------------------------------------
# MySQLValue - the SS9 type-mapping table as one tagged union, used both for
# bound parameters (COM_STMT_EXECUTE) and decoded result rows. `MySQLParam`
# is the same type under a different name - the value shapes going in and
# coming out are identical, so there is no separate encode-only union.
# ---------------------------------------------------------------------------

@union
class MySQLValue:
	Null:        None
	Int32:       i32
	UInt32:      u32
	Int64:       i64
	UInt64:      u64
	Float32:     f32
	Float64:     f64
	Text:        str
	Blob:        bytes
	Decimal:     str
	DateVal:     date
	TimeVal:     timedelta
	DateTimeVal: datetime

MySQLParam: TypeAlias = MySQLValue

# the tzinfo attached to every DATETIME/TIMESTAMP value decoded off the wire
# - lib/datetime.py's datetime has no naive variant (tzinfo is mandatory),
# and MySQL's own DATETIME/TIMESTAMP wall-clock values carry no zone
# information at all (see doc SS9's own note on this) - a zero fixed offset
# is a documented placeholder with no real meaning, never adjusted against.
_WIRE_TZ: ZoneInfo = ZoneInfo.fixed_offset( 0 )


def _protocol_err_from_date( e: DateError ) -> MySQLError:
	return MySQLError.ProtocolError


# ---------------------------------------------------------------------------
# Binary Protocol Resultset Row decoding. The NULL-bitmap bit offset is 2
# (not 0, unlike COM_STMT_EXECUTE's own null-bitmap) - dev.mysql.com's own
# binary-resultset page documents this offset explicitly; getting it wrong
# silently shifts every null-check by two columns.
# ---------------------------------------------------------------------------

def _decode_value( r: _PacketReader, col: ColumnDef ) -> Result[MySQLValue, MySQLError]:
	t: u8 = col.col_type
	if t == MYSQL_TYPE_TINY:
		v: u8 = r.read_u8().or_return()
		if col.is_unsigned:
			with compiler.wrap_arithmetic:
				return Result.Ok( MySQLValue.UInt32( u32( v )))
		with compiler.wrap_arithmetic:
			s8: i8 = i8( v )
			return Result.Ok( MySQLValue.Int32( i32( s8 )))
	if t == MYSQL_TYPE_SHORT or t == MYSQL_TYPE_YEAR:
		v16: u16 = r.read_u16_le().or_return()
		if col.is_unsigned or t == MYSQL_TYPE_YEAR:
			with compiler.wrap_arithmetic:
				return Result.Ok( MySQLValue.Int32( i32( v16 )) if t == MYSQL_TYPE_YEAR else MySQLValue.UInt32( u32( v16 )))
		with compiler.wrap_arithmetic:
			s16: i16 = i16( v16 )
			return Result.Ok( MySQLValue.Int32( i32( s16 )))
	if t == MYSQL_TYPE_LONG or t == MYSQL_TYPE_INT24:
		v32: u32 = r.read_u32_le().or_return()
		if col.is_unsigned:
			return Result.Ok( MySQLValue.UInt32( v32 ))
		with compiler.wrap_arithmetic:
			return Result.Ok( MySQLValue.Int32( i32( v32 )))
	if t == MYSQL_TYPE_LONGLONG:
		v64: u64 = r.read_u64_le().or_return()
		if col.is_unsigned:
			return Result.Ok( MySQLValue.UInt64( v64 ))
		with compiler.wrap_arithmetic:
			return Result.Ok( MySQLValue.Int64( i64( v64 )))
	if t == MYSQL_TYPE_FLOAT:
		fb: u32 = r.read_u32_le().or_return()
		return Result.Ok( MySQLValue.Float32( _f32_from_bits( fb )))
	if t == MYSQL_TYPE_DOUBLE:
		db: u64 = r.read_u64_le().or_return()
		return Result.Ok( MySQLValue.Float64( _f64_from_bits( db )))
	if t == MYSQL_TYPE_NEWDECIMAL or t == MYSQL_TYPE_DECIMAL:
		dtext: bytes = r.read_lenenc_bytes().or_return()
		return Result.Ok( MySQLValue.Decimal( dtext.decode_lossy() ))
	if ( t == MYSQL_TYPE_VARCHAR or t == MYSQL_TYPE_VAR_STRING or t == MYSQL_TYPE_STRING
			or t == MYSQL_TYPE_JSON ):
		text_bytes: bytes = r.read_lenenc_bytes().or_return()
		if col.charset == _BINARY_CHARSET:
			return Result.Ok( MySQLValue.Blob( text_bytes ))
		return Result.Ok( MySQLValue.Text( text_bytes.decode_lossy() ))
	if ( t == MYSQL_TYPE_BLOB or t == MYSQL_TYPE_TINY_BLOB or t == MYSQL_TYPE_MEDIUM_BLOB
			or t == MYSQL_TYPE_LONG_BLOB or t == MYSQL_TYPE_BIT or t == MYSQL_TYPE_GEOMETRY
			or t == MYSQL_TYPE_ENUM or t == MYSQL_TYPE_SET ):
		blob_bytes: bytes = r.read_lenenc_bytes().or_return()
		if col.charset == _BINARY_CHARSET:
			return Result.Ok( MySQLValue.Blob( blob_bytes ))
		return Result.Ok( MySQLValue.Text( blob_bytes.decode_lossy() ))
	if t == MYSQL_TYPE_DATE:
		dlen: u8 = r.read_u8().or_return()
		if dlen == 0:
			return Result.Err( MySQLError.ProtocolError )  # zero-date - see doc SS9's note
		year: u16 = r.read_u16_le().or_return()
		month: u8 = r.read_u8().or_return()
		day: u8 = r.read_u8().or_return()
		with compiler.wrap_arithmetic:
			d: date = date( i32( year ), i32( month ), i32( day )).or_return( _protocol_err_from_date )
		return Result.Ok( MySQLValue.DateVal( d ))
	if t == MYSQL_TYPE_DATETIME or t == MYSQL_TYPE_TIMESTAMP:
		dtlen: u8 = r.read_u8().or_return()
		if dtlen == 0:
			return Result.Err( MySQLError.ProtocolError )  # zero-datetime - see doc SS9's note
		dyear: u16 = r.read_u16_le().or_return()
		dmonth: u8 = r.read_u8().or_return()
		dday: u8 = r.read_u8().or_return()
		hour: u8 = 0
		minute: u8 = 0
		second: u8 = 0
		micro: u32 = 0
		if dtlen >= 7:
			hour = r.read_u8().or_return()
			minute = r.read_u8().or_return()
			second = r.read_u8().or_return()
		if dtlen >= 11:
			micro = r.read_u32_le().or_return()
		with compiler.wrap_arithmetic:
			dt: datetime = datetime(
				i32( dyear ), i32( dmonth ), i32( dday ),
				i32( hour ), i32( minute ), i32( second ), i32( micro ),
				tzinfo = _WIRE_TZ,
			).or_return( _protocol_err_from_date )
		return Result.Ok( MySQLValue.DateTimeVal( dt ))
	if t == MYSQL_TYPE_TIME:
		tlen: u8 = r.read_u8().or_return()
		if tlen == 0:
			return Result.Ok( MySQLValue.TimeVal( timedelta() ))
		is_neg: u8 = r.read_u8().or_return()
		tdays: u32 = r.read_u32_le().or_return()
		thour: u8 = r.read_u8().or_return()
		tminute: u8 = r.read_u8().or_return()
		tsecond: u8 = r.read_u8().or_return()
		tmicro: u32 = 0
		if tlen >= 12:
			tmicro = r.read_u32_le().or_return()
		with compiler.wrap_arithmetic:
			mag: timedelta = timedelta(
				days = i32( tdays ), hours = i32( thour ), minutes = i32( tminute ),
				seconds = i32( tsecond ), microseconds = i32( tmicro ),
			)
		td: timedelta = -mag if is_neg != 0 else mag
		return Result.Ok( MySQLValue.TimeVal( td ))
	return Result.Err( MySQLError.ProtocolError )


def decode_binary_row( payload: bytes, columns: list[ColumnDef] ) -> Result[list[MySQLValue], MySQLError]:
	r: _PacketReader = _PacketReader( payload )
	header: u8 = r.read_u8().or_return()
	if header != 0x00:
		return Result.Err( MySQLError.ProtocolError )
	ncols: usize = len( columns )
	with compiler.panic_arithmetic( 'divisor is a non-zero literal' ):
		bitmap_len: usize = ( ncols + usize( 7 ) + usize( 2 )) // usize( 8 )
	bitmap: bytes = r.read_bytes( bitmap_len ).or_return()

	values: list[MySQLValue] = list[MySQLValue]()
	i: usize = 0
	with compiler.wrap_arithmetic:
		while i < ncols:
			bit_index: usize = i + usize( 2 )
			byte_index: usize = 0
			bit_in_byte: usize = 0
			with compiler.panic_arithmetic( 'divisor is a non-zero literal' ):
				byte_index = bit_index // usize( 8 )
				bit_in_byte = bit_index % usize( 8 )
			byte_val: u8 = bitmap.__getitem__( byte_index ).unwrap( 'byte_index < bitmap_len by construction' )
			is_null: bool = (( byte_val >> u8( bit_in_byte )) & u8( 1 )) != 0
			if is_null:
				values.append( MySQLValue.Null( None ))
			else:
				col: ColumnDef = columns.__getitem__( i ).unwrap( 'i < ncols' )
				values.append( _decode_value( r, col ).or_return() )
			i += 1
	return Result.Ok( values )


# ---------------------------------------------------------------------------
# Parameter binding for COM_STMT_EXECUTE - the reverse direction of
# _decode_value above.
# ---------------------------------------------------------------------------

def _param_type_and_unsigned( p: MySQLValue ) -> tuple[u8, bool]:
	match p:
		case MySQLValue.Null( _ ):
			return ( MYSQL_TYPE_NULL, False )
		case MySQLValue.Int32( _ ):
			return ( MYSQL_TYPE_LONG, False )
		case MySQLValue.UInt32( _ ):
			return ( MYSQL_TYPE_LONG, True )
		case MySQLValue.Int64( _ ):
			return ( MYSQL_TYPE_LONGLONG, False )
		case MySQLValue.UInt64( _ ):
			return ( MYSQL_TYPE_LONGLONG, True )
		case MySQLValue.Float32( _ ):
			return ( MYSQL_TYPE_FLOAT, False )
		case MySQLValue.Float64( _ ):
			return ( MYSQL_TYPE_DOUBLE, False )
		case MySQLValue.Text( _ ):
			return ( MYSQL_TYPE_VARCHAR, False )
		case MySQLValue.Blob( _ ):
			return ( MYSQL_TYPE_BLOB, False )
		case MySQLValue.Decimal( _ ):
			return ( MYSQL_TYPE_NEWDECIMAL, False )
		case MySQLValue.DateVal( _ ):
			return ( MYSQL_TYPE_DATE, False )
		case MySQLValue.TimeVal( _ ):
			return ( MYSQL_TYPE_TIME, False )
		case MySQLValue.DateTimeVal( _ ):
			return ( MYSQL_TYPE_DATETIME, False )


def _append_param_value( buf: _ByteBuf, p: MySQLValue ) -> None:
	match p:
		case MySQLValue.Null( _ ):
			pass  # no value bytes - covered by the null bitmap
		case MySQLValue.Int32( v_i32 ):
			with compiler.wrap_arithmetic:
				buf.append_u32_le( u32( v_i32 ))
		case MySQLValue.UInt32( v_u32 ):
			buf.append_u32_le( v_u32 )
		case MySQLValue.Int64( v_i64 ):
			with compiler.wrap_arithmetic:
				buf.append_u64_le( u64( v_i64 ))
		case MySQLValue.UInt64( v_u64 ):
			buf.append_u64_le( v_u64 )
		case MySQLValue.Float32( v_f32 ):
			buf.append_u32_le( _bits_from_f32( v_f32 ))
		case MySQLValue.Float64( v_f64 ):
			buf.append_u64_le( _bits_from_f64( v_f64 ))
		case MySQLValue.Text( v_text ):
			buf.append_lenenc_bytes( v_text.encode().unwrap( 'str is always valid utf8' ))
		case MySQLValue.Blob( v_blob ):
			buf.append_lenenc_bytes( v_blob )
		case MySQLValue.Decimal( v_dec ):
			buf.append_lenenc_bytes( v_dec.encode().unwrap( 'str is always valid utf8' ))
		case MySQLValue.DateVal( v_date ):
			buf.append_u8( 4 )
			with compiler.wrap_arithmetic:
				buf.append_u16_le( u16( v_date.year ))
				buf.append_u8( u8( v_date.month ))
				buf.append_u8( u8( v_date.day ))
		case MySQLValue.TimeVal( v_time ):
			us: i64 = v_time.total_us
			is_neg: bool = us < i64( 0 )
			with compiler.wrap_arithmetic:
				mag: i64 = -us if is_neg else us
			with compiler.panic_arithmetic( 'every divisor here is a non-zero literal' ):
				days: i64 = mag // 86_400_000_000
				rem: i64 = mag % 86_400_000_000
				hour: i64 = rem // 3_600_000_000
				rem2: i64 = rem % 3_600_000_000
				minute: i64 = rem2 // 60_000_000
				rem3: i64 = rem2 % 60_000_000
				second: i64 = rem3 // 1_000_000
				micro: i64 = rem3 % 1_000_000
			buf.append_u8( 12 )
			buf.append_u8( 1 if is_neg else 0 )
			with compiler.wrap_arithmetic:
				buf.append_u32_le( u32( days ))
				buf.append_u8( u8( hour ))
				buf.append_u8( u8( minute ))
				buf.append_u8( u8( second ))
				buf.append_u32_le( u32( micro ))
		case MySQLValue.DateTimeVal( v_dt ):
			buf.append_u8( 11 )
			with compiler.wrap_arithmetic:
				buf.append_u16_le( u16( v_dt.year ))
				buf.append_u8( u8( v_dt.month ))
				buf.append_u8( u8( v_dt.day ))
				buf.append_u8( u8( v_dt.hour ))
				buf.append_u8( u8( v_dt.minute ))
				buf.append_u8( u8( v_dt.second ))
				buf.append_u32_le( u32( v_dt.microsecond ))


def _is_null_param( p: MySQLValue ) -> bool:
	match p:
		case MySQLValue.Null( _ ):
			return True
		case _:
			return False


# ---------------------------------------------------------------------------
# COM_* payload builders
# ---------------------------------------------------------------------------

_COM_QUERY:        u8 = 0x03
_COM_STMT_PREPARE: u8 = 0x16
_COM_STMT_EXECUTE: u8 = 0x17
_COM_STMT_CLOSE:   u8 = 0x19

_CURSOR_TYPE_NO_CURSOR: u8 = 0x00


def build_com_query( sql: str ) -> bytes:
	''' internal use only - see doc SS10: this is never used for anything
	touching a caller-supplied value, only fixed, driver-controlled
	statements (SET autocommit=0, COMMIT, ROLLBACK). '''
	buf: _ByteBuf = _ByteBuf( 32 )
	buf.append_u8( _COM_QUERY )
	buf.append_bytes( sql.encode().unwrap( 'driver-controlled sql is valid utf8' ))
	return buf.to_bytes()


def build_com_stmt_prepare( sql: str ) -> bytes:
	with compiler.panic_arithmetic( 'a capacity hint - never actually indexed, just sized generously' ):
		cap: usize = usize( 32 ) + len( sql )
	buf: _ByteBuf = _ByteBuf( cap )
	buf.append_u8( _COM_STMT_PREPARE )
	buf.append_bytes( sql.encode().unwrap( 'sql is valid utf8' ))
	return buf.to_bytes()


def build_com_stmt_close( stmt_id: u32 ) -> bytes:
	buf: _ByteBuf = _ByteBuf( 5 )
	buf.append_u8( _COM_STMT_CLOSE )
	buf.append_u32_le( stmt_id )
	return buf.to_bytes()


def build_com_stmt_execute( stmt_id: u32, params: list[MySQLValue] ) -> bytes:
	buf: _ByteBuf = _ByteBuf( 32 )
	buf.append_u8( _COM_STMT_EXECUTE )
	buf.append_u32_le( stmt_id )
	buf.append_u8( _CURSOR_TYPE_NO_CURSOR )
	buf.append_u32_le( 1 )  # iteration_count - always 1

	nparams: usize = len( params )
	if nparams > 0:
		with compiler.panic_arithmetic( 'divisor is a non-zero literal' ):
			bitmap_len: usize = ( nparams + usize( 7 )) // usize( 8 )
		bitmap = bytearray( bitmap_len )
		i: usize = 0
		with compiler.wrap_arithmetic:
			while i < nparams:
				p: MySQLValue = params.__getitem__( i ).unwrap( 'i < nparams' )
				if _is_null_param( p ):
					with compiler.panic_arithmetic( 'divisor is a non-zero literal' ):
						byte_index: usize = i // usize( 8 )
						bit_in_byte: usize = i % usize( 8 )
					cur: u8 = bitmap.__getitem__( byte_index ).unwrap( 'byte_index < bitmap_len' )
					bitmap.__setitem__( byte_index, cur | ( u8( 1 ) << u8( bit_in_byte )))
				i += 1
		buf.append_bytes( bitmap )
		buf.append_u8( 1 )  # new_params_bind_flag - always re-send types

		j: usize = 0
		with compiler.wrap_arithmetic:
			while j < nparams:
				pj: MySQLValue = params.__getitem__( j ).unwrap( 'j < nparams' )
				( type_code, is_unsigned ) = _param_type_and_unsigned( pj )
				buf.append_u8( type_code )
				buf.append_u8( 0x80 if is_unsigned else 0x00 )
				j += 1

		k: usize = 0
		with compiler.wrap_arithmetic:
			while k < nparams:
				pk: MySQLValue = params.__getitem__( k ).unwrap( 'k < nparams' )
				if not _is_null_param( pk ):
					_append_param_value( buf, pk )
				k += 1

	return buf.to_bytes()


def read_lenenc_count_from( payload: bytes ) -> Result[u64, MySQLError]:
	''' reads a single lenenc int off the front of an already-buffered
	packet payload - used for a resultset header packet's leading
	column-count field (see mysql.client.Cursor.execute). '''
	return _PacketReader( payload ).read_lenenc_count()


def read_column_defs( sock: Socket, count: u64 ) -> Result[list[ColumnDef], MySQLError]:
	''' reads `count` Protocol::ColumnDefinition41 packets followed by the
	trailing EOF packet (CLIENT_DEPRECATE_EOF is never negotiated - see
	build_handshake_response41 - so every column-def run still ends in a
	real EOF packet on every server version this driver talks to). '''
	cols: list[ColumnDef] = list[ColumnDef]()
	with compiler.saturate_arithmetic:
		n: usize = usize( count )
	i: usize = 0
	with compiler.wrap_arithmetic:
		while i < n:
			( _, payload ) = read_packet( sock ).or_return()
			cols.append( parse_column_def( payload ).or_return() )
			i += 1
	read_packet( sock ).or_return()  # trailing EOF - discarded
	return Result.Ok( cols )


class PrepareResult:
	statement_id: u32
	num_columns:  u16
	num_params:   u16

	def __init__( self, statement_id: u32, num_columns: u16, num_params: u16 ) -> None:
		self.statement_id = statement_id
		self.num_columns = num_columns
		self.num_params = num_params


def parse_stmt_prepare_ok( payload: bytes ) -> Result[PrepareResult, MySQLError]:
	r: _PacketReader = _PacketReader( payload )
	header: u8 = r.read_u8().or_return()
	if header != 0x00:
		return Result.Err( MySQLError.ProtocolError )
	statement_id: u32 = r.read_u32_le().or_return()
	num_columns: u16 = r.read_u16_le().or_return()
	num_params: u16 = r.read_u16_le().or_return()
	r.skip( usize( 1 )).or_return()   # reserved
	r.read_u16_le().or_return()       # warning_count - unused
	return Result.Ok( PrepareResult( statement_id, num_columns, num_params ))


class AuthSwitch:
	plugin_name: str
	plugin_data: bytes

	def __init__( self, plugin_name: str, plugin_data: bytes ) -> None:
		self.plugin_name = plugin_name
		self.plugin_data = plugin_data


def parse_auth_switch_request( payload: bytes ) -> Result[AuthSwitch, MySQLError]:
	r: _PacketReader = _PacketReader( payload )
	r.read_u8().or_return()  # 0xFE marker
	name: str = r.read_null_str().or_return()
	data: bytes = r.read_rest()
	# the scramble handed to a switch-request is already the full 20 bytes
	# (no 8+12 split like the initial handshake's own auth-plugin-data) -
	# trailing NUL some servers still append is harmless: scramble_native_
	# password only ever reads the first 20 bytes it's given.
	return Result.Ok( AuthSwitch( name, data ))
