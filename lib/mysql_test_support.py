# lib/mysql_test_support.py — thin public wrappers around lib/mysql.py's
# package-private internals (_Buf/_Reader/_scramble_password/_write_packet/
# _read_packet), so mysql_test.py's compiled programs (which run as their
# own `__main__` package, outside `lib`'s package boundary) can unit-test
# the wire-format/scramble logic directly instead of only indirectly via a
# live server round trip. Test-only - not part of the public DB-API surface
# (see database_client_investigation.md).

import mysql
import socket


def lenenc_int_roundtrips( v: u64 ) -> bool:
	buf = mysql._Buf()
	buf.write_lenenc_int( v )
	r = mysql._Reader( buf.get_const_ptr(), buf.len() )
	match r.read_lenenc_int():
		case Result.Ok( got ):
			return got == v
		case Result.Err( _ ):
			return False


def lenenc_str_and_cstr_roundtrip() -> Result[None, str]:
	buf = mysql._Buf()
	buf.write_lenenc_str( 'hello world' )
	buf.write_cstr( 'user42' )
	r = mysql._Reader( buf.get_const_ptr(), buf.len() )
	s1: str = r.read_lenenc_str().unwrap( 'lenenc str' )
	if s1 != 'hello world':
		return Result.Err( 'lenenc str mismatch' )
	s2: str = r.read_null_term_str().unwrap( 'cstr' )
	if s2 != 'user42':
		return Result.Err( 'cstr mismatch' )
	if r.remaining() != usize( 0 ):
		return Result.Err( 'trailing bytes' )
	return Result.Ok( None )


def packet_framing_roundtrip() -> Result[None, str]:
	( a, b ) = socket.make_loopback_pair()
	payload = mysql._Buf()
	payload.write_u8( 7 )
	payload.write_lenenc_str( 'SELECT 1' )
	mysql._write_packet( a, 3, payload.get_const_ptr(), payload.len() ).unwrap( 'write' )
	( got, next_seq ) = mysql._read_packet( b, 3 ).unwrap( 'read' )
	a.close()
	b.close()
	if next_seq != 4:
		return Result.Err( 'sequence id not incremented' )
	r = mysql._Reader( got.get_const_ptr(), len( got ))
	if r.read_u8().unwrap( 'byte' ) != 7:
		return Result.Err( 'first byte mismatch' )
	if r.read_lenenc_str().unwrap( 'str' ) != 'SELECT 1':
		return Result.Err( 'payload string mismatch' )
	return Result.Ok( None )


def compute_scramble( password: str, seed: bytes ) -> bytes:
	return mysql._scramble_password( password, seed )


def make_column_description( name: str, type_code: u8, size: u32, scale: u8, nullable: bool, flags: u16, charset: u16 ) -> mysql.ColumnDescription:
	return mysql.ColumnDescription._make( name, type_code, size, scale, nullable, flags, charset )
