# Real-server tests for lib/mysql/client.py, against this machine's live
# MariaDB (XAMPP, C:\xampp\mysql\, port 3306). Uses a DEDICATED database +
# account created for this purpose only (see database_client_investigation.
# md SS11) - never touches app_service_account or any other real account/db:
#
#   CREATE DATABASE metalpy_test_aa43;
#   CREATE USER 'metalpy_test_aa43'@'localhost' IDENTIFIED BY 'MetalpyTestPw1';
#   GRANT ALL ON metalpy_test_aa43.* TO 'metalpy_test_aa43'@'localhost';
#
# Table setup is done once per test run by setUpClass via the mysql.exe CLI
# (not through the driver being tested - keeps the fixture independent of
# the code under test). Requires a real server; set
# METALPY_TEST_MYSQL_LIVE=0 to skip if one isn't available.

import os
import subprocess
import unittest

import test_support
from compiler import Compiler
from discovery import Discovery

_MYSQL_CLI = r'C:\xampp\mysql\bin\mysql.exe'
_LIVE_OK = ( os.environ.get( 'METALPY_TEST_MYSQL_LIVE', '1' ) not in ( '0', 'false', 'False' )
	and os.path.exists( _MYSQL_CLI ))


def _run_sql( sql: str ) -> None:
	subprocess.run(
		[ _MYSQL_CLI, '-u', 'metalpy_test_aa43', '-pMetalpyTestPw1', 'metalpy_test_aa43', '-e', sql ],
		check = True, capture_output = True,
	)


@unittest.skipUnless( _LIVE_OK, 'no local MariaDB (C:\\xampp\\mysql\\bin\\mysql.exe) found - set METALPY_TEST_MYSQL_LIVE=0 to acknowledge' )
class MySQLLiveTests( test_support.RealCompileMixin, unittest.TestCase ):
	@classmethod
	def setUpClass( cls ) -> None:
		_run_sql( '''
			CREATE TABLE IF NOT EXISTS kv (id INT PRIMARY KEY, name VARCHAR(64), val DOUBLE);
			CREATE TABLE IF NOT EXISTS tokens (id INT PRIMARY KEY, used_at DATETIME NULL);
			CREATE TABLE IF NOT EXISTS types_probe (
				id INT PRIMARY KEY, d DECIMAL(10,2), dt DATE, tm TIME, yr YEAR,
				big BIGINT, ubig BIGINT UNSIGNED, flt FLOAT, blb BLOB
			);
			DELETE FROM kv;
			DELETE FROM tokens;
			DELETE FROM types_probe;
			INSERT INTO tokens (id, used_at) VALUES (1, NULL);
		''' )

	def setUp( self ) -> None:
		self.discovery = Discovery( import_builtins = True )
		self.compiler = Compiler( self.discovery )

	@unittest.skipUnless( test_support.HAS_CC, 'no C compiler (clang/gcc/msvc) found - skipping' )
	def test_programs_compile_and_run( self ) -> None:
		self.assert_programs_run([
			( 'connect_insert_select_fetch', '''
import compiler
from mysql.client import Connection, Cursor
from mysql.protocol import MySQLValue

def main() -> i32:
	conn: Connection = Connection.connect( "127.0.0.1", u16( 3306 ), "metalpy_test_aa43", "MetalpyTestPw1", "metalpy_test_aa43" ).unwrap( "connect" )
	cur: Cursor = conn.cursor().unwrap( "cursor" )

	ins_params: list[MySQLValue] = list[MySQLValue]()
	ins_params.append( MySQLValue.Int32( 1 ))
	ins_params.append( MySQLValue.Text( "hello" ))
	ins_params.append( MySQLValue.Float64( 3.5 ))
	cur.execute( "INSERT INTO kv (id, name, val) VALUES (?, ?, ?)", ins_params ).unwrap( "insert" )
	if cur.rowcount != 1:
		return 1

	sel_params: list[MySQLValue] = list[MySQLValue]()
	sel_params.append( MySQLValue.Int32( 1 ))
	cur.execute( "SELECT id, name, val FROM kv WHERE id=?", sel_params ).unwrap( "select" )
	if cur.rowcount != 1:
		return 2

	maybe_row = cur.fetchone().unwrap( "fetchone" )
	if maybe_row is None:
		return 3
	row: list[MySQLValue] = maybe_row

	match row.__getitem__( usize( 0 )).unwrap( "col0" ):
		case MySQLValue.Int32( v0 ):
			if v0 != 1:
				return 4
		case _:
			return 5
	match row.__getitem__( usize( 1 )).unwrap( "col1" ):
		case MySQLValue.Text( v1 ):
			if v1 != "hello":
				return 6
		case _:
			return 7
	match row.__getitem__( usize( 2 )).unwrap( "col2" ):
		case MySQLValue.Float64( v2 ):
			if v2 < 3.4 or v2 > 3.6:
				return 8
		case _:
			return 9

	second = cur.fetchone().unwrap( "fetchone2" )
	if second is not None:
		return 10

	conn.commit().unwrap( "commit" )
	cur.close()
	conn.close()
	return 0
''' ),
			( 'fetchall_multiple_rows', '''
import compiler
from mysql.client import Connection, Cursor
from mysql.protocol import MySQLValue

def main() -> i32:
	conn: Connection = Connection.connect( "127.0.0.1", u16( 3306 ), "metalpy_test_aa43", "MetalpyTestPw1", "metalpy_test_aa43" ).unwrap( "connect" )
	cur: Cursor = conn.cursor().unwrap( "cursor" )

	del_params: list[MySQLValue] = list[MySQLValue]()
	cur.execute( "DELETE FROM kv", del_params ).unwrap( "clear" )
	conn.commit().unwrap( "commit0" )

	i: i32 = 0
	with compiler.wrap_arithmetic:
		while i < 3:
			ins: list[MySQLValue] = list[MySQLValue]()
			ins.append( MySQLValue.Int32( i ))
			ins.append( MySQLValue.Null( None ))
			ins.append( MySQLValue.Float64( f64( i )))
			cur.execute( "INSERT INTO kv (id, name, val) VALUES (?, ?, ?)", ins ).unwrap( "insert" )
			i += 1
	conn.commit().unwrap( "commit1" )

	sel: list[MySQLValue] = list[MySQLValue]()
	cur.execute( "SELECT id FROM kv ORDER BY id", sel ).unwrap( "select" )
	if cur.rowcount != 3:
		return 1
	rows: list[list[MySQLValue]] = cur.fetchall().unwrap( "fetchall" )
	if len( rows ) != usize( 3 ):
		return 2

	# NULL param/column round-trip - the second column was bound as NULL above
	sel2: list[MySQLValue] = list[MySQLValue]()
	cur.execute( "SELECT name FROM kv WHERE id=0", sel2 ).unwrap( "select2" )
	row2 = cur.fetchone().unwrap( "fetchone" )
	if row2 is None:
		return 3
	match row2.__getitem__( usize( 0 )).unwrap( "col0" ):
		case MySQLValue.Null( _ ):
			pass
		case _:
			return 4

	cur.close()
	conn.close()
	return 0
''' ),
			# the exact claim-pattern rowcount semantics from database_client_
			# investigation.md SS4: UPDATE ... WHERE used_at IS NULL claims
			# once (rowcount 1), a second attempt sees rowcount 0 - never
			# ambiguous with "matched but no-op" since CLIENT_FOUND_ROWS is
			# never negotiated.
			( 'update_claim_pattern_rowcount', '''
import compiler
from mysql.client import Connection, Cursor
from mysql.protocol import MySQLValue
from datetime import datetime
from zoneinfo import ZoneInfo

def main() -> i32:
	conn: Connection = Connection.connect( "127.0.0.1", u16( 3306 ), "metalpy_test_aa43", "MetalpyTestPw1", "metalpy_test_aa43" ).unwrap( "connect" )
	cur: Cursor = conn.cursor().unwrap( "cursor" )

	stamp: datetime = datetime( 2026, 1, 1, 12, 0, 0, 0, tzinfo = ZoneInfo.fixed_offset( 0 )).unwrap( "x" )

	p1: list[MySQLValue] = list[MySQLValue]()
	p1.append( MySQLValue.DateTimeVal( stamp ))
	p1.append( MySQLValue.Int32( 1 ))
	cur.execute( "UPDATE tokens SET used_at=? WHERE id=? AND used_at IS NULL", p1 ).unwrap( "update1" )
	first: i32 = cur.rowcount

	p2: list[MySQLValue] = list[MySQLValue]()
	p2.append( MySQLValue.DateTimeVal( stamp ))
	p2.append( MySQLValue.Int32( 1 ))
	cur.execute( "UPDATE tokens SET used_at=? WHERE id=? AND used_at IS NULL", p2 ).unwrap( "update2" )
	second: i32 = cur.rowcount

	conn.commit().unwrap( "commit" )
	cur.close()
	conn.close()

	if first != 1:
		return 1
	if second != 0:
		return 2
	return 0
''' ),
			# broader SS9 type-table coverage against a real server: DECIMAL
			# (as str, never through f64), DATE, TIME, YEAR, signed/unsigned
			# BIGINT, FLOAT, and a binary-charset BLOB (distinguished from
			# TEXT by charset=63 - see protocol._decode_value).
			( 'type_table_round_trip', '''
import compiler
from mysql.client import Connection, Cursor
from mysql.protocol import MySQLValue
from datetime import date, timedelta

def main() -> i32:
	conn: Connection = Connection.connect( "127.0.0.1", u16( 3306 ), "metalpy_test_aa43", "MetalpyTestPw1", "metalpy_test_aa43" ).unwrap( "connect" )
	cur: Cursor = conn.cursor().unwrap( "cursor" )

	d: date = date( 2026, 8, 29 ).unwrap( "x" )
	tm: timedelta = timedelta( hours = 13, minutes = 5, seconds = 9 )
	blob_val: bytes = "rawbytes".encode().unwrap( "x" )

	ins: list[MySQLValue] = list[MySQLValue]()
	ins.append( MySQLValue.Int32( 1 ))
	ins.append( MySQLValue.Decimal( "1234.50" ))
	ins.append( MySQLValue.DateVal( d ))
	ins.append( MySQLValue.TimeVal( tm ))
	ins.append( MySQLValue.Int32( 2024 ))
	ins.append( MySQLValue.Int64( i64( -123456789012 )))
	ins.append( MySQLValue.UInt64( u64( 18446744073709551615 )))
	ins.append( MySQLValue.Float32( f32( 2.5 )))
	ins.append( MySQLValue.Blob( blob_val ))
	cur.execute( "INSERT INTO types_probe (id, d, dt, tm, yr, big, ubig, flt, blb) VALUES (?,?,?,?,?,?,?,?,?)", ins ).unwrap( "insert" )
	conn.commit().unwrap( "commit" )

	sel: list[MySQLValue] = list[MySQLValue]()
	sel.append( MySQLValue.Int32( 1 ))
	cur.execute( "SELECT d, dt, tm, yr, big, ubig, flt, blb FROM types_probe WHERE id=?", sel ).unwrap( "select" )
	maybe_row = cur.fetchone().unwrap( "fetchone" )
	if maybe_row is None:
		return 1
	row: list[MySQLValue] = maybe_row

	match row.__getitem__( usize( 0 )).unwrap( "d" ):
		case MySQLValue.Decimal( dv ):
			if dv != "1234.50":
				return 2
		case _:
			return 3
	match row.__getitem__( usize( 1 )).unwrap( "dt" ):
		case MySQLValue.DateVal( dtv ):
			if dtv.year != 2026 or dtv.month != 8 or dtv.day != 29:
				return 4
		case _:
			return 5
	match row.__getitem__( usize( 2 )).unwrap( "tm" ):
		case MySQLValue.TimeVal( tv ):
			with compiler.wrap_arithmetic:
				expected_us: i64 = i64( 13 * 3600 + 5 * 60 + 9 ) * 1_000_000
			if tv.total_us != expected_us:
				return 6
		case _:
			return 7
	match row.__getitem__( usize( 3 )).unwrap( "yr" ):
		case MySQLValue.Int32( yv ):
			if yv != 2024:
				return 8
		case _:
			return 9
	match row.__getitem__( usize( 4 )).unwrap( "big" ):
		case MySQLValue.Int64( bv ):
			if bv != i64( -123456789012 ):
				return 10
		case _:
			return 11
	match row.__getitem__( usize( 5 )).unwrap( "ubig" ):
		case MySQLValue.UInt64( uv ):
			if uv != u64( 18446744073709551615 ):
				return 12
		case _:
			return 13
	match row.__getitem__( usize( 6 )).unwrap( "flt" ):
		case MySQLValue.Float32( fv ):
			if fv < 2.4 or fv > 2.6:
				return 14
		case _:
			return 15
	match row.__getitem__( usize( 7 )).unwrap( "blb" ):
		case MySQLValue.Blob( blv ):
			if len( blv ) != len( blob_val ):
				return 16
		case _:
			return 17

	cur.close()
	conn.close()
	return 0
''' ),
			# error paths: bad auth -> AuthFailed, syntax error -> QueryError
			# with real server detail populated (doc SS5).
			( 'error_paths', '''
import compiler
from mysql.client import Connection, Cursor
from mysql.protocol import MySQLValue, MySQLError

def main() -> i32:
	match Connection.connect( "127.0.0.1", u16( 3306 ), "metalpy_test_aa43", "wrong_password_entirely", "metalpy_test_aa43" ):
		case Result.Ok( _ ):
			return 1
		case Result.Err( e ):
			if e != MySQLError.AuthFailed:
				return 2

	conn: Connection = Connection.connect( "127.0.0.1", u16( 3306 ), "metalpy_test_aa43", "MetalpyTestPw1", "metalpy_test_aa43" ).unwrap( "connect" )
	cur: Cursor = conn.cursor().unwrap( "cursor" )
	no_params: list[MySQLValue] = list[MySQLValue]()
	match cur.execute( "SELEKT GARBAGE FROM NOWHERE", no_params ):
		case Result.Ok( _ ):
			return 3
		case Result.Err( e2 ):
			if e2 != MySQLError.QueryError:
				return 4
			if conn.last_server_errno == 0:
				return 5
			if len( conn.last_sqlstate ) != usize( 5 ):
				return 6

	cur.close()
	conn.close()
	return 0
''' ),
			# cursor/connection lifecycle: using a cursor after its
			# connection closed returns Closed, not a crash (doc SS7).
			( 'closed_connection_cursor_returns_closed_error', '''
import compiler
from mysql.client import Connection, Cursor
from mysql.protocol import MySQLValue, MySQLError

def main() -> i32:
	conn: Connection = Connection.connect( "127.0.0.1", u16( 3306 ), "metalpy_test_aa43", "MetalpyTestPw1", "metalpy_test_aa43" ).unwrap( "connect" )
	cur: Cursor = conn.cursor().unwrap( "cursor" )
	conn.close()
	conn.close()  # closing twice is a no-op, not an error

	no_params: list[MySQLValue] = list[MySQLValue]()
	match cur.execute( "SELECT 1", no_params ):
		case Result.Ok( _ ):
			return 1
		case Result.Err( e ):
			if e != MySQLError.Closed:
				return 2
	return 0
''' ),
		], timeout = 30.0 )


if __name__ == '__main__':
	unittest.main()
