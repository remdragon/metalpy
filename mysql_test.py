# Real-compile-and-run tests for lib/mysql.py.
#
# MySQLPhase0Tests - packet encode/decode, the mysql_native_password
#   scramble formula, and the MySQLError/MySQLValue types - zero I/O, runs
#   everywhere a C compiler is available (same split as ssl_test.py's
#   SSLPhase0Tests vs its live-handshake classes).
# MySQLLiveServerTests - a real connect/auth/query/fetch/error-path round
#   trip against this machine's actual XAMPP MariaDB instance
#   (C:\xampp\mysql\, mysqld already running on port 3306,
#   mysql_native_password auth). Requires a `metalpy_test` database/user to
#   already exist - see database_client_investigation.md for the exact
#   `mysql.exe` commands used to create it; skipped if that database can't
#   be reached (METALPY_TEST_MYSQL=0 to explicitly opt out).

import os
import unittest

import test_support


class MySQLPhase0Tests( test_support.RealCompileMixin, unittest.TestCase ):
	def setUp( self ) -> None:
		from discovery import Discovery
		from compiler import Compiler
		self.discovery = Discovery( import_builtins = True )
		self.compiler = Compiler( self.discovery )

	@unittest.skipUnless( test_support.HAS_CC, 'no C compiler (clang/gcc/msvc) found - skipping real-compile mysql tests' )
	def test_programs_compile_and_run( self ) -> None:
		self.assert_programs_run([
			( 'mysqlerror_values_distinct', '''
from mysql import MySQLError

def main() -> i32:
	if MySQLError.ConnectFailed == MySQLError.AuthFailed:
		return 1
	if MySQLError.ProtocolError == MySQLError.QueryError:
		return 2
	if MySQLError.Closed == MySQLError.WrongThread:
		return 3
	if MySQLError.Other == MySQLError.ConnectFailed:
		return 4
	return 0
''' ),
			( 'mysqlvalue_construct_and_match', '''
from mysql import MySQLValue

def classify( v: MySQLValue ) -> i32:
	match v:
		case MySQLValue.Null( _ ):
			return 0
		case MySQLValue.Int( iv ):
			if iv == 42:
				return 1
			return -1
		case MySQLValue.UInt( uv ):
			if uv == u64( 7 ):
				return 2
			return -1
		case MySQLValue.Float32( _ ):
			return 3
		case MySQLValue.Float64( _ ):
			return 4
		case MySQLValue.Str( s ):
			if s == 'hi':
				return 5
			return -1
		case MySQLValue.Bytes( _ ):
			return 6
		case MySQLValue.Date( _ ):
			return 7
		case MySQLValue.DateTime( _ ):
			return 8
		case MySQLValue.Time( _ ):
			return 9

def main() -> i32:
	if classify( MySQLValue.Null( None )) != 0:
		return 1
	if classify( MySQLValue.Int( i64( 42 ))) != 1:
		return 2
	if classify( MySQLValue.UInt( u64( 7 ))) != 2:
		return 3
	if classify( MySQLValue.Float32( f32( 1.5 ))) != 3:
		return 4
	if classify( MySQLValue.Float64( 1.5 )) != 4:
		return 5
	if classify( MySQLValue.Str( 'hi' )) != 5:
		return 6
	if classify( MySQLValue.Bytes( 'x'.encode().unwrap( 'x' ))) != 6:
		return 7
	return 0
''' ),
			# round-trips every length-encoded-integer boundary the real wire
			# format defines (0xFB/0xFC/0xFD/0xFE thresholds), plus lenenc-str/
			# cstr and full packet framing - these are the exact byte layouts
			# a COM_STMT_EXECUTE/binary-resultset-row misparse would corrupt
			# silently rather than fail to compile. Routed through
			# lib/mysql_test_support.py's thin wrappers, NOT mysql._Buf/
			# _Reader/etc directly - those are package-private (a single
			# leading underscore is enforced, not just conventional: a real
			# compile error confirms a `__main__`-scope test program cannot
			# see them), and mysql_test_support.py lives inside the same
			# top-level `lib` package so it can.
			( 'lenenc_int_roundtrip', '''
import mysql_test_support as mts

def main() -> i32:
	if not mts.lenenc_int_roundtrips( u64( 0 )):
		return 1
	if not mts.lenenc_int_roundtrips( u64( 250 )):
		return 2
	if not mts.lenenc_int_roundtrips( u64( 251 )):
		return 3
	if not mts.lenenc_int_roundtrips( u64( 65535 )):
		return 4
	if not mts.lenenc_int_roundtrips( u64( 65536 )):
		return 5
	if not mts.lenenc_int_roundtrips( u64( 16777215 )):
		return 6
	if not mts.lenenc_int_roundtrips( u64( 16777216 )):
		return 7
	if not mts.lenenc_int_roundtrips( u64( 4294967296 )):
		return 8
	return 0
''' ),
			( 'lenenc_str_and_cstr_roundtrip', '''
import mysql_test_support as mts

def main() -> i32:
	match mts.lenenc_str_and_cstr_roundtrip():
		case Result.Ok( _ ):
			return 0
		case Result.Err( _ ):
			return 1
''' ),
			( 'packet_framing_roundtrip', '''
import mysql_test_support as mts

def main() -> i32:
	match mts.packet_framing_roundtrip():
		case Result.Ok( _ ):
			return 0
		case Result.Err( _ ):
			return 1
''' ),
			# real mysql_native_password formula: SHA1(password) XOR
			# SHA1(seed + SHA1(SHA1(password))) - independently computed via
			# Python's hashlib for password="secret", seed=bytes(range(20))
			# while writing this test, not derived circularly from the
			# implementation under test.
			( 'mysql_native_password_scramble_known_answer', '''
import mysql_test_support as mts
import base64

def main() -> i32:
	seed_buf: bytearray = bytearray( usize( 20 ))
	sp: Ptr[u8] = seed_buf.get_ptr()
	i: usize = 0
	with compiler.wrap_arithmetic:
		while i < usize( 20 ):
			sp[i] = u8( i )
			i += 1
	seed: bytes = bytes.from_bytearray( move( seed_buf ))
	token: bytes = mts.compute_scramble( 'secret', seed )
	if len( token ) != usize( 20 ):
		return 1
	expected_hex: str = '21B3FF405F32CBE4AAFFF291396046EA29FA3A4D'
	got_hex: str = base64.b16encode( token ).decode().unwrap( 'x' )
	if got_hex != expected_hex:
		return 2
	return 0
''' ),
			( 'mysql_native_password_scramble_empty_password', '''
import mysql_test_support as mts

def main() -> i32:
	seed: bytes = bytes.from_bytearray( move( bytearray( usize( 20 ))))
	token: bytes = mts.compute_scramble( '', seed )
	if len( token ) != usize( 0 ):
		return 1
	return 0
''' ),
			( 'column_description_flags', '''
import mysql_test_support as mts
from mysql import MYSQL_TYPE_LONG

def main() -> i32:
	col = mts.make_column_description( 'id', MYSQL_TYPE_LONG, u32( 11 ), u8( 0 ), False, u16( 1 ), u16( 63 ))
	if col.name != 'id':
		return 1
	if col.type_code != MYSQL_TYPE_LONG:
		return 2
	if col.nullable:
		return 3
	return 0
''' ),
		] )


_MYSQL_HOST = '127.0.0.1'
_MYSQL_PORT = 3306
_MYSQL_USER = 'metalpy_test'
_MYSQL_PASSWORD = 'metalpy_test_pw'
_MYSQL_DB = 'metalpy_test'

def _mysql_server_reachable() -> bool:
	if os.environ.get( 'METALPY_TEST_MYSQL' ) == '0':
		return False
	import socket as pysocket
	try:
		s = pysocket.create_connection(( _MYSQL_HOST, _MYSQL_PORT ), timeout = 1.0 )
		s.close()
		return True
	except OSError:
		return False


@unittest.skipUnless( test_support.HAS_CC, 'no C compiler (clang/gcc/msvc) found - skipping real-compile mysql tests' )
@unittest.skipUnless( _mysql_server_reachable(), 'no MariaDB/MySQL server reachable on 127.0.0.1:3306 - set METALPY_TEST_MYSQL=0 to acknowledge, or start XAMPP mysqld' )
class MySQLLiveServerTests( test_support.RealCompileMixin, unittest.TestCase ):
	''' Drives lib/mysql.py's real wire protocol against this machine's real
	XAMPP MariaDB (C:\\xampp\\mysql\\), using a dedicated metalpy_test
	database/user created specifically for this suite (see
	database_client_investigation.md for the exact `mysql.exe` commands) -
	never the real app_service_account account or any of its data. '''

	def setUp( self ) -> None:
		from discovery import Discovery
		from compiler import Compiler
		self.discovery = Discovery( import_builtins = True )
		self.compiler = Compiler( self.discovery )

	def test_programs_compile_and_run( self ) -> None:
		connect_args = f"'{_MYSQL_HOST}', u16( {_MYSQL_PORT} ), '{_MYSQL_USER}', '{_MYSQL_PASSWORD}', '{_MYSQL_DB}'"
		self.assert_programs_run([
			( 'connect_and_close', f'''
import mysql

def main() -> i32:
	conn: mysql.Connection = mysql.Connection.connect( {connect_args} ).unwrap( 'connect' )
	conn.close().unwrap( 'close' )
	return 0
''' ),
			( 'insert_select_fetchone_fetchall', f'''
import mysql
from mysql import MySQLValue

def main() -> i32:
	conn: mysql.Connection = mysql.Connection.connect( {connect_args} ).unwrap( 'connect' )
	cur: mysql.Cursor = conn.cursor().unwrap( 'cursor' )

	cur.execute( 'DROP TABLE IF EXISTS metalpy_smoke', list[MySQLValue]() ).unwrap( 'drop' )
	cur.execute( 'CREATE TABLE metalpy_smoke (id INT PRIMARY KEY, name VARCHAR(64))', list[MySQLValue]() ).unwrap( 'create' )

	ins_params: list[MySQLValue] = list[MySQLValue]()
	ins_params.append( MySQLValue.Int( i64( 1 )))
	ins_params.append( MySQLValue.Str( 'alice' ))
	cur.execute( 'INSERT INTO metalpy_smoke (id, name) VALUES (?, ?)', ins_params ).unwrap( 'insert' )
	if cur.rowcount != 1:
		return 1

	sel_params: list[MySQLValue] = list[MySQLValue]()
	sel_params.append( MySQLValue.Int( i64( 1 )))
	cur.execute( 'SELECT id, name FROM metalpy_smoke WHERE id = ?', sel_params ).unwrap( 'select' )
	if cur.rowcount != 1:
		return 2
	if len( cur.description ) != usize( 2 ):
		return 3

	row: list[MySQLValue]|None = cur.fetchone().unwrap( 'fetchone' )
	match row:
		case None:
			return 4
		case _:
			pass
	row_vals: list[MySQLValue] = row.unwrap( 'row' )
	match row_vals.__getitem__( usize( 1 )).unwrap( 'x' ):
		case MySQLValue.Str( name ):
			if name != 'alice':
				return 5
		case _:
			return 6

	none_row: list[MySQLValue]|None = cur.fetchone().unwrap( 'fetchone again' )
	match none_row:
		case None:
			pass
		case _:
			return 7

	cur.execute( 'SELECT id FROM metalpy_smoke', list[MySQLValue]() ).unwrap( 'select all' )
	all_rows: list[list[MySQLValue]] = cur.fetchall().unwrap( 'fetchall' )
	if len( all_rows ) != usize( 1 ):
		return 8

	cur.execute( 'DROP TABLE metalpy_smoke', list[MySQLValue]() ).unwrap( 'drop again' )
	cur.close().unwrap( 'cursor close' )
	conn.close().unwrap( 'conn close' )
	return 0
''' ),
			# the exact rowcount claim-pattern from design doc §4: a matched-
			# but-unmodified UPDATE reports rowcount 0 (CLIENT_FOUND_ROWS is
			# never set), so `rowcount == 1` unambiguously means THIS caller
			# won the single-use-claim race.
			( 'update_claim_pattern_rowcount', f'''
import mysql
from mysql import MySQLValue

def main() -> i32:
	conn: mysql.Connection = mysql.Connection.connect( {connect_args} ).unwrap( 'connect' )
	cur: mysql.Cursor = conn.cursor().unwrap( 'cursor' )

	cur.execute( 'DROP TABLE IF EXISTS metalpy_tokens', list[MySQLValue]() ).unwrap( 'drop' )
	cur.execute( 'CREATE TABLE metalpy_tokens (id INT PRIMARY KEY, used_at INT NULL)', list[MySQLValue]() ).unwrap( 'create' )

	ins: list[MySQLValue] = list[MySQLValue]()
	ins.append( MySQLValue.Int( i64( 1 )))
	cur.execute( 'INSERT INTO metalpy_tokens (id, used_at) VALUES (?, NULL)', ins ).unwrap( 'insert' )

	claim: list[MySQLValue] = list[MySQLValue]()
	claim.append( MySQLValue.Int( i64( 1234 )))
	claim.append( MySQLValue.Int( i64( 1 )))
	cur.execute( 'UPDATE metalpy_tokens SET used_at = ? WHERE id = ? AND used_at IS NULL', claim ).unwrap( 'claim 1' )
	if cur.rowcount != 1:
		return 1

	claim2: list[MySQLValue] = list[MySQLValue]()
	claim2.append( MySQLValue.Int( i64( 5678 )))
	claim2.append( MySQLValue.Int( i64( 1 )))
	cur.execute( 'UPDATE metalpy_tokens SET used_at = ? WHERE id = ? AND used_at IS NULL', claim2 ).unwrap( 'claim 2' )
	if cur.rowcount != 0:
		return 2

	cur.execute( 'DROP TABLE metalpy_tokens', list[MySQLValue]() ).unwrap( 'drop again' )
	conn.close().unwrap( 'close' )
	return 0
''' ),
			( 'bad_auth_maps_to_auth_failed', f'''
import mysql
from mysql import MySQLError

def main() -> i32:
	match mysql.Connection.connect( '{_MYSQL_HOST}', u16( {_MYSQL_PORT} ), '{_MYSQL_USER}', 'definitely-the-wrong-password' ):
		case Result.Ok( _ ):
			return 1
		case Result.Err( e ):
			if e != MySQLError.AuthFailed:
				return 2
	return 0
''' ),
			( 'syntax_error_maps_to_query_error', f'''
import mysql
from mysql import MySQLValue, MySQLError

def main() -> i32:
	conn: mysql.Connection = mysql.Connection.connect( {connect_args} ).unwrap( 'connect' )
	cur: mysql.Cursor = conn.cursor().unwrap( 'cursor' )
	match cur.execute( 'SELECT THIS IS NOT VALID SQL !!!', list[MySQLValue]() ):
		case Result.Ok( _ ):
			return 1
		case Result.Err( e ):
			if e != MySQLError.QueryError:
				return 2
	if conn.last_server_errno == 0:
		return 3
	conn.close().unwrap( 'close' )
	return 0
''' ),
		], timeout = 30.0 )


if __name__ == '__main__':
	unittest.main()
