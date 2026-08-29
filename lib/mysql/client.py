# lib/mysql/client.py - DB-API-shaped Connection/Cursor on top of
# mysql.protocol's wire-protocol primitives. See database_client_
# investigation.md for the full design this implements (paramstyle,
# autocommit, rowcount semantics, threadsafety=0 enforcement, cursor
# lifecycle, commit/rollback behavior).

import compiler
from socket import Socket
from mysql.protocol import (
	MySQLError, MySQLValue, MySQLParam, ColumnDef, ErrInfo,
	read_packet, send_packet, parse_handshake_v10, scramble_native_password,
	build_handshake_response41, parse_ok_packet, parse_err_packet,
	build_com_query, build_com_stmt_prepare, build_com_stmt_close,
	build_com_stmt_execute, parse_stmt_prepare_ok, read_column_defs,
	decode_binary_row, current_thread_id, parse_auth_switch_request,
	read_lenenc_count_from,
)


def _map_os( e: OSError ) -> MySQLError:
	return MySQLError.ConnectFailed


# ---------------------------------------------------------------------------
# Connection
# ---------------------------------------------------------------------------

class Connection:
	__sock:    Socket
	__closed:  bool
	__thread_id: usize

	# populated on QueryError/AuthFailed reached via a live Connection (doc
	# SS5) - connect()-time AuthFailed has no Connection yet to attach this
	# to, so that one specific path carries no server detail (a documented
	# v1 gap, see database_client_investigation.md SS11).
	last_server_errno:   i32
	last_sqlstate:       str
	last_server_message: str

	def __init__( self, sock: Socket ) -> None:
		self.__sock = sock
		self.__closed = False
		self.__thread_id = current_thread_id()
		self.last_server_errno = 0
		self.last_sqlstate = ''
		self.last_server_message = ''

	def __del__( self ) -> None:
		self.close()

	def close( self ) -> None:
		if not self.__closed:
			self.__sock.close()
			self.__closed = True

	@private
	def is_closed( self ) -> bool:
		return self.__closed

	@private
	def sock( self ) -> Socket:
		return self.__sock

	@private
	def _guard( self ) -> Result[None, MySQLError]:
		if current_thread_id() != self.__thread_id:
			return Result.Err( MySQLError.WrongThread )
		if self.__closed:
			return Result.Err( MySQLError.Closed )
		return Result.Ok( None )

	@private
	def _record_err( self, e: ErrInfo ) -> None:
		with compiler.wrap_arithmetic:
			self.last_server_errno = i32( e.code )
		self.last_sqlstate = e.sqlstate
		self.last_server_message = e.message

	def cursor( self ) -> Result[Cursor, MySQLError]:
		self._guard().or_return()
		return Result.Ok( Cursor( self ))

	@private
	def _exec_fixed_statement( self, sql: str ) -> Result[None, MySQLError]:
		''' COM_QUERY with a fixed, driver-controlled statement only (SET
		autocommit=0 / COMMIT / ROLLBACK) - never a caller-supplied query,
		per doc SS10. '''
		self._guard().or_return()
		send_packet( self.__sock, 0, build_com_query( sql )).or_return()
		( _seq, payload ) = read_packet( self.__sock ).or_return()
		header: u8 = payload.__getitem__( usize( 0 )).unwrap( 'a real response packet is never empty' )
		if header == 0x00 or header == 0xFE:
			parse_ok_packet( payload ).or_return()
			return Result.Ok( None )
		if header == 0xFF:
			err = parse_err_packet( payload ).or_return()
			self._record_err( err )
			return Result.Err( MySQLError.QueryError )
		return Result.Err( MySQLError.ProtocolError )

	def commit( self ) -> Result[None, MySQLError]:
		''' always forwarded to the server, even with nothing pending - see
		doc SS8 (autocommit is always off, so there's always a transaction,
		possibly empty, and MySQL treats COMMIT with nothing pending as a
		harmless no-op already). '''
		return self._exec_fixed_statement( 'COMMIT' )

	def rollback( self ) -> Result[None, MySQLError]:
		return self._exec_fixed_statement( 'ROLLBACK' )

	@staticmethod
	def connect( host: str, port: u16, user: str, password: str, database: str = '' ) -> Result[Connection, MySQLError]:
		sock: Socket = Socket.tcp().or_return( _map_os )
		sock.connect( host, port ).or_return( _map_os )

		( hs_seq, hs_payload ) = read_packet( sock ).or_return()
		hinfo = parse_handshake_v10( hs_payload ).or_return()
		if hinfo.auth_plugin_name != 'mysql_native_password':
			return Result.Err( MySQLError.UnsupportedAuthPlugin )

		password_bytes = password.encode().unwrap( 'password is valid utf8' )
		scramble = scramble_native_password( password_bytes, hinfo.auth_plugin_data )
		resp = build_handshake_response41( user, scramble, database )
		with compiler.wrap_arithmetic:
			resp_seq: u8 = hs_seq + 1
		send_packet( sock, resp_seq, resp ).or_return()

		( _seq2, payload2 ) = read_packet( sock ).or_return()
		header: u8 = payload2.__getitem__( usize( 0 )).unwrap( 'a real response packet is never empty' )

		if header == 0xFE and len( payload2 ) > usize( 1 ):
			# AuthSwitchRequest - the server wants a different plugin than it
			# originally advertised. Only re-attempted once, and only if the
			# new plugin is still mysql_native_password (the one scramble
			# algorithm this driver implements) - see doc SS1/SS11.
			switch = parse_auth_switch_request( payload2 ).or_return()
			if switch.plugin_name != 'mysql_native_password':
				return Result.Err( MySQLError.UnsupportedAuthPlugin )
			scramble2 = scramble_native_password( password_bytes, switch.plugin_data )
			with compiler.wrap_arithmetic:
				switch_seq: u8 = _seq2 + 1
			send_packet( sock, switch_seq, scramble2 ).or_return()
			( _seq3, payload3 ) = read_packet( sock ).or_return()
			payload2 = payload3
			header = payload2.__getitem__( usize( 0 )).unwrap( 'a real response packet is never empty' )

		if header == 0xFF:
			# no Connection object exists yet to attach server detail to -
			# see this class's own last_server_errno/etc doc comment.
			return Result.Err( MySQLError.AuthFailed )
		if header != 0x00:
			return Result.Err( MySQLError.ProtocolError )
		parse_ok_packet( payload2 ).or_return()

		conn: Connection = Connection( sock )
		conn._exec_fixed_statement( 'SET autocommit=0' ).or_return()
		return Result.Ok( conn )


# ---------------------------------------------------------------------------
# Cursor
# ---------------------------------------------------------------------------

class Cursor:
	__conn:    Connection
	__thread_id: usize
	__closed:  bool
	__rowcount: i32
	__description: list[ColumnDef]
	__has_description: bool
	__rows: list[list[MySQLValue]]
	__pos:  usize

	def __init__( self, conn: Connection ) -> None:
		self.__conn = conn
		self.__thread_id = current_thread_id()
		self.__closed = False
		self.__rowcount = -1
		self.__description = list[ColumnDef]()
		self.__has_description = False
		self.__rows = list[list[MySQLValue]]()
		self.__pos = 0

	def close( self ) -> None:
		self.__closed = True

	def _guard( self ) -> Result[Socket, MySQLError]:
		if current_thread_id() != self.__thread_id:
			return Result.Err( MySQLError.WrongThread )
		if self.__closed:
			return Result.Err( MySQLError.Closed )
		if self.__conn.is_closed():
			return Result.Err( MySQLError.Closed )
		return Result.Ok( self.__conn.sock() )

	@property
	def rowcount( self ) -> i32:
		return self.__rowcount

	@property
	def description( self ) -> list[ColumnDef]|None:
		if self.__has_description:
			return self.__description
		return None

	def execute( self, sql: str, params: list[MySQLValue] ) -> Result[None, MySQLError]:
		sock: Socket = self._guard().or_return()

		send_packet( sock, 0, build_com_stmt_prepare( sql )).or_return()
		( _seq, first_payload ) = read_packet( sock ).or_return()
		first_byte: u8 = first_payload.__getitem__( usize( 0 )).unwrap( 'a real response packet is never empty' )
		if first_byte == 0xFF:
			err = parse_err_packet( first_payload ).or_return()
			self.__conn._record_err( err )
			return Result.Err( MySQLError.QueryError )
		prep = parse_stmt_prepare_ok( first_payload ).or_return()

		# both param and column definition runs (if present) must be fully
		# drained off the wire before this exchange can move on to EXECUTE
		# or CLOSE, regardless of whether the param count below turns out to
		# match - leaving either one half-read would desync framing for
		# every packet after it.
		if prep.num_params > 0:
			read_column_defs( sock, u64( prep.num_params )).or_return()
		result_columns: list[ColumnDef] = list[ColumnDef]()
		if prep.num_columns > 0:
			result_columns = read_column_defs( sock, u64( prep.num_columns )).or_return()

		with compiler.wrap_arithmetic:
			nparams_given: usize = len( params )
			nparams_expected: usize = usize( prep.num_params )
		if nparams_given != nparams_expected:
			send_packet( sock, 0, build_com_stmt_close( prep.statement_id )).or_return()
			return Result.Err( MySQLError.TypeMismatch )

		send_packet( sock, 0, build_com_stmt_execute( prep.statement_id, params )).or_return()
		( _seq2, exec_payload ) = read_packet( sock ).or_return()
		exec_header: u8 = exec_payload.__getitem__( usize( 0 )).unwrap( 'a real response packet is never empty' )

		if exec_header == 0xFF:
			err2 = parse_err_packet( exec_payload ).or_return()
			self.__conn._record_err( err2 )
			send_packet( sock, 0, build_com_stmt_close( prep.statement_id )).or_return()
			return Result.Err( MySQLError.QueryError )

		if exec_header == 0x00:
			ok = parse_ok_packet( exec_payload ).or_return()
			with compiler.saturate_arithmetic:
				self.__rowcount = i32( ok.affected_rows )
			self.__has_description = False
			self.__description = list[ColumnDef]()
			self.__rows = list[list[MySQLValue]]()
			self.__pos = 0
			send_packet( sock, 0, build_com_stmt_close( prep.statement_id )).or_return()
			return Result.Ok( None )

		# a real result set - exec_payload is the column-count lenenc int,
		# then column_count column-definition packets + EOF (a full, self-
		# contained resultset header, same shape COM_QUERY's own resultset
		# response has - NOT reusing the column defs already read right
		# after PREPARE above, which are informational-only), THEN the rows.
		col_count: u64 = read_lenenc_count_from( exec_payload ).or_return()
		read_column_defs( sock, col_count ).or_return()

		rows: list[list[MySQLValue]] = list[list[MySQLValue]]()
		with compiler.wrap_arithmetic:
			while True:
				( _seq3, row_payload ) = read_packet( sock ).or_return()
				row_header: u8 = row_payload.__getitem__( usize( 0 )).unwrap( 'a real row packet is never empty' )
				if row_header == 0xFE and len( row_payload ) < usize( 9 ):
					break  # EOF
				if row_header == 0xFF:
					err3 = parse_err_packet( row_payload ).or_return()
					self.__conn._record_err( err3 )
					send_packet( sock, 0, build_com_stmt_close( prep.statement_id )).or_return()
					return Result.Err( MySQLError.QueryError )
				rows.append( decode_binary_row( row_payload, result_columns ).or_return() )

		self.__description = result_columns
		self.__has_description = True
		self.__rows = rows
		self.__pos = 0
		with compiler.saturate_arithmetic:
			self.__rowcount = i32( len( rows ))
		send_packet( sock, 0, build_com_stmt_close( prep.statement_id )).or_return()
		return Result.Ok( None )

	def fetchone( self ) -> Result[list[MySQLValue]|None, MySQLError]:
		self._guard().or_return()
		if self.__pos >= len( self.__rows ):
			return Result.Ok( None )
		row = self.__rows.__getitem__( self.__pos ).unwrap( 'pos < len(rows), just checked' )
		with compiler.wrap_arithmetic:
			self.__pos += 1
		return Result.Ok( row )

	def fetchall( self ) -> Result[list[list[MySQLValue]], MySQLError]:
		self._guard().or_return()
		out: list[list[MySQLValue]] = list[list[MySQLValue]]()
		with compiler.wrap_arithmetic:
			while self.__pos < len( self.__rows ):
				out.append( self.__rows.__getitem__( self.__pos ).unwrap( 'pos < len(rows), just checked' ))
				self.__pos += 1
		return Result.Ok( out )
