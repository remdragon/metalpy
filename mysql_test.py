# Phase-0 tests for lib/mysql/{protocol,client}.py: packet encode/decode,
# scramble computation, MySQLError/MySQLValue shape - no network I/O. See
# mysql_live_test.py for the real-server leg and database_client_
# investigation.md for the design.

import unittest

import test_support
from compiler import Compiler
from discovery import Discovery


class MySQLPhase0Tests( test_support.RealCompileMixin, unittest.TestCase ):
	def setUp( self ) -> None:
		self.discovery = Discovery( import_builtins = True )
		self.compiler = Compiler( self.discovery )

	@unittest.skipUnless( test_support.HAS_CC, 'no C compiler (clang/gcc/msvc) found - skipping' )
	def test_programs_compile_and_run( self ) -> None:
		self.assert_programs_run([
			( 'mysqlerror_values_distinct', '''
from mysql.protocol import MySQLError

def main() -> i32:
	if MySQLError.ConnectFailed == MySQLError.AuthFailed:
		return 1
	if MySQLError.QueryError == MySQLError.ProtocolError:
		return 2
	if MySQLError.Closed == MySQLError.WrongThread:
		return 3
	if MySQLError.Other == MySQLError.ConnectFailed:
		return 4
	return 0
''' ),
			( 'mysqlerror_match_dispatch', '''
from mysql.protocol import MySQLError

def classify( e: MySQLError ) -> i32:
	match e:
		case MySQLError.AuthFailed:
			return 100
		case MySQLError.QueryError:
			return 101
		case MySQLError.TypeMismatch:
			return 102
		case _:
			return 999

def main() -> i32:
	if classify( MySQLError.AuthFailed ) != 100:
		return 1
	if classify( MySQLError.QueryError ) != 101:
		return 2
	if classify( MySQLError.TypeMismatch ) != 102:
		return 3
	if classify( MySQLError.Closed ) != 999:
		return 4
	return 0
''' ),
			# COM_STMT_PREPARE payload: 1 command byte + raw sql text, no
			# length prefix/terminator (EOF-string per the wire spec).
			( 'stmt_prepare_payload_shape', '''
from mysql.protocol import build_com_stmt_prepare

def main() -> i32:
	sql: str = "SELECT 1"
	p: bytes = build_com_stmt_prepare( sql )
	with compiler.wrap_arithmetic:
		expected: usize = usize( 1 ) + len( sql )
	if len( p ) != expected:
		return 1
	first: u8 = p.__getitem__( usize( 0 )).unwrap( 'x' )
	if first != u8( 0x16 ):
		return 2
	return 0
''' ),
			( 'stmt_close_payload_shape', '''
from mysql.protocol import build_com_stmt_close

def main() -> i32:
	p: bytes = build_com_stmt_close( u32( 7 ))
	if len( p ) != usize( 5 ):
		return 1
	first: u8 = p.__getitem__( usize( 0 )).unwrap( 'x' )
	if first != u8( 0x19 ):
		return 2
	return 0
''' ),
			( 'com_query_payload_shape', '''
from mysql.protocol import build_com_query

def main() -> i32:
	sql: str = "COMMIT"
	q: bytes = build_com_query( sql )
	with compiler.wrap_arithmetic:
		expected: usize = usize( 1 ) + len( sql )
	if len( q ) != expected:
		return 1
	first: u8 = q.__getitem__( usize( 0 )).unwrap( 'x' )
	if first != u8( 0x03 ):
		return 2
	return 0
''' ),
			# COM_STMT_EXECUTE with 2 params, one of them NULL: fixed header
			# (1+4+1+4=10 bytes) + 1-byte null bitmap ((2+7)/8=1) + 1-byte
			# new-params-bind-flag + 2*2 type bytes + the one non-null value
			# (a 4-byte i32).
			( 'stmt_execute_null_bitmap_and_types', '''
from mysql.protocol import build_com_stmt_execute, MySQLValue

def main() -> i32:
	params: list[MySQLValue] = list[MySQLValue]()
	params.append( MySQLValue.Int32( 42 ))
	params.append( MySQLValue.Null( None ))
	ex: bytes = build_com_stmt_execute( u32( 1 ), params )
	if len( ex ) != usize( 10 + 1 + 1 + 4 + 4 ):
		return 1
	# null bitmap byte is at offset 10 - bit 1 set (2nd param is NULL)
	bitmap_byte: u8 = ex.__getitem__( usize( 10 )).unwrap( 'x' )
	if bitmap_byte != u8( 0x02 ):
		return 2
	# first param's type bytes (MYSQL_TYPE_LONG=3, signed) at offset 12
	type0: u8 = ex.__getitem__( usize( 12 )).unwrap( 'x' )
	if type0 != u8( 3 ):
		return 3
	unsigned0: u8 = ex.__getitem__( usize( 13 )).unwrap( 'x' )
	if unsigned0 != u8( 0 ):
		return 4
	# second param's type bytes (MYSQL_TYPE_NULL=6) at offset 14
	type1: u8 = ex.__getitem__( usize( 14 )).unwrap( 'x' )
	if type1 != u8( 6 ):
		return 5
	return 0
''' ),
			( 'stmt_execute_unsigned_flag', '''
from mysql.protocol import build_com_stmt_execute, MySQLValue

def main() -> i32:
	params: list[MySQLValue] = list[MySQLValue]()
	params.append( MySQLValue.UInt64( u64( 9 )))
	ex: bytes = build_com_stmt_execute( u32( 1 ), params )
	# fixed header(10) + null bitmap(1) + bind-flag(1) = offset 12 for type
	type0: u8 = ex.__getitem__( usize( 12 )).unwrap( 'x' )
	if type0 != u8( 8 ):  # MYSQL_TYPE_LONGLONG
		return 1
	unsigned0: u8 = ex.__getitem__( usize( 13 )).unwrap( 'x' )
	if unsigned0 != u8( 0x80 ):
		return 2
	return 0
''' ),
			# mysql_native_password's scramble formula, checked against an
			# independently computed reference (Python hashlib, see this
			# test's own commit message / database_client_investigation.md).
			( 'scramble_matches_reference_vector', '''
from mysql.protocol import scramble_native_password
import base64

def main() -> i32:
	password: bytes = "MetalpyTestPw1".encode().unwrap( 'x' )
	seed: bytes = "01234567890123456789".encode().unwrap( 'x' )
	scramble: bytes = scramble_native_password( password, seed )
	if len( scramble ) != usize( 20 ):
		return 1
	hex: str = base64.b16encode( scramble ).decode().unwrap( 'x' )
	if hex != '2335D098D1764352CBD53B64A70129280FA78524'[0:40]:
		return 2
	return 0
''' ),
			( 'scramble_empty_password_is_empty', '''
from mysql.protocol import scramble_native_password

def main() -> i32:
	password: bytes = bytes.from_bytearray( move( bytearray( 0 )))
	seed: bytes = "01234567890123456789".encode().unwrap( 'x' )
	scramble: bytes = scramble_native_password( password, seed )
	if len( scramble ) != usize( 0 ):
		return 1
	return 0
''' ),
			# MySQLValue round-trips through match dispatch for every
			# variant in the SS9 type-mapping table - a compile-time
			# exhaustiveness check as much as a runtime one.
			( 'mysqlvalue_variants_roundtrip', '''
from mysql.protocol import MySQLValue
from datetime import date, datetime, timedelta
from zoneinfo import ZoneInfo

def describe( v: MySQLValue ) -> i32:
	match v:
		case MySQLValue.Null( _ ):
			return 0
		case MySQLValue.Int32( x0 ):
			return 1
		case MySQLValue.UInt32( x1 ):
			return 2
		case MySQLValue.Int64( x2 ):
			return 3
		case MySQLValue.UInt64( x3 ):
			return 4
		case MySQLValue.Float32( x4 ):
			return 5
		case MySQLValue.Float64( x5 ):
			return 6
		case MySQLValue.Text( x6 ):
			return 7
		case MySQLValue.Blob( x7 ):
			return 8
		case MySQLValue.Decimal( x8 ):
			return 9
		case MySQLValue.DateVal( x9 ):
			return 10
		case MySQLValue.TimeVal( x10 ):
			return 11
		case MySQLValue.DateTimeVal( x11 ):
			return 12

def main() -> i32:
	if describe( MySQLValue.Null( None )) != 0:
		return 1
	if describe( MySQLValue.Int32( -5 )) != 1:
		return 2
	if describe( MySQLValue.UInt32( u32( 5 ))) != 2:
		return 3
	if describe( MySQLValue.Text( "hi" )) != 7:
		return 4
	blob: bytes = "hi".encode().unwrap( 'x' )
	if describe( MySQLValue.Blob( blob )) != 8:
		return 5
	d: date = date( 2024, 1, 1 ).unwrap( 'x' )
	if describe( MySQLValue.DateVal( d )) != 10:
		return 6
	td: timedelta = timedelta( hours = 1 )
	if describe( MySQLValue.TimeVal( td )) != 11:
		return 7
	dt: datetime = datetime( 2024, 1, 1, tzinfo = ZoneInfo.fixed_offset( 0 )).unwrap( 'x' )
	if describe( MySQLValue.DateTimeVal( dt )) != 12:
		return 8
	return 0
''' ),
		] )


if __name__ == '__main__':
	unittest.main()
