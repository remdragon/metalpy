'''
CSV reading/writing, interface shaped after Python's built-in csv module.

Two deviations from Python's ergonomics, both confirmed necessary by real
compiler experiments (see PLAN csv.py, Step 0), not design choices:

1. A Result-returning __next__() can never be driven by `for` in this
   language, whether on a hand-written class or a real generator, in any
   context. Since CSV parsing must be able to signal errors (malformed
   input, I/O failure) without crashing the whole program, Reader/DictReader
   cannot support `for row in csv.reader(...):` - consumers drive them with
   an explicit loop instead.
2. `list[T]|None` (and any other generic type used as a T|None union member)
   does not type-check in this compiler at all - confirmed with a minimal
   repro; a plain non-generic RC class in the same union position works
   fine. So instead of Ok(None)/Ok(Some(row)), a "row or end of input"
   result is carried by a small non-optional, non-generic wrapper (MaybeRow
   below): callers check `.has_row` instead of matching on None.

	match csv.reader( 'data.csv' ):
		case Result.Ok( r ):
			while True:
				match r.__next__():
					case Result.Ok( m ):
						if not m.has_row:
							break
						... use m.row ...
					case Result.Err( e ):
						... handle e ...
		case Result.Err( e ):
			... failed to open ...

v1 scope: delimiter/quotechar (each exactly one byte), doubled-quotechar
escaping, "\r\n"/"\n" line endings on read, "\r\n" always written
(QUOTE_MINIMAL-equivalent: a field is quoted only if it contains the
delimiter, quotechar, or a newline). Deferred: Dialect objects, Sniffer,
QUOTE_ALL/QUOTE_NONNUMERIC/QUOTE_NONE, escapechar, skipinitialspace, custom
lineterminator, DictReader's restkey.
'''

import compiler
import sys
from builtins import BinaryReader, BinaryWriter


class CsvError:
	message: str

	def __init__( self, message: str ) -> None:
		self.message = message


class MaybeRow:
	''' row-or-end-of-input carrier - see the module docstring's point 2 on
	why this exists instead of list[str]|None. row is only meaningful when
	has_row is True; an empty list[str]() otherwise. '''
	has_row: bool
	row: list[str]

	def __init__( self, has_row: bool, row: list[str] ) -> None:
		self.has_row = has_row
		self.row = row


class MaybeLine:
	''' line-or-end-of-input carrier. str|None has the SAME problem as
	list[str]|None above in practice, even though str isn't generic: a
	str|None value narrows fine for a direct comparison/method call on the
	same expression (confirmed working), but not when reassigned into
	another variable or passed as a function argument (confirmed failing,
	real compile errors either way) - and _LineReader.next_line()'s result
	needs to be passed on to RowParser.feed_line(), so the carrier class
	sidesteps the whole question. line is only meaningful when has_line is
	True; an empty str otherwise. '''
	has_line: bool
	line: str

	def __init__( self, has_line: bool, line: str ) -> None:
		self.has_line = has_line
		self.line = line


# ---------------------------------------------------------------------------
# _substr - [start, end) byte range of s, as a new str. str has no public
# slicing primitive (str._byte_slice exists but is @private to str itself),
# so this rebuilds the same "alloc, memcpy the range, revalidate as UTF-8"
# shape using only str's PUBLIC surface (get_const_ptr/byte_len/from_cstr)
# rather than reaching into str's private internals.
# ---------------------------------------------------------------------------

def _substr( s: str, start: usize, end: usize ) -> str:
	with compiler.panic_arithmetic( 'bounded by the caller\'s own scan of s, cannot overflow' ):
		piece_len: usize = end - start
		buf_size: usize = piece_len + 1
	buf: Ptr[u8] = sys.alloc[u8]( buf_size )
	defer( sys.free( buf ))
	with compiler.wrap_arithmetic:
		src: ConstPtr[u8] = s.get_const_ptr() + start
	sys.memcpy( buf, src, piece_len )
	buf[piece_len] = 0
	const_buf: ConstPtr[u8] = compiler.cast( ConstPtr[u8], buf )
	return str.from_cstr( const_buf, buf_size ).unwrap( 'invalid UTF-8 in csv._substr' )


# ---------------------------------------------------------------------------
# RowParser - pure, no I/O. Consumes one already-newline-stripped line per
# feed_line() call, accumulating state across calls only when a quoted
# field's content spans multiple lines (an embedded literal newline).
# ---------------------------------------------------------------------------

_ST_START_FIELD: i32 = 0     # about to read the first byte of a new field
_ST_IN_UNQUOTED: i32 = 1     # reading an unquoted field's content
_ST_IN_QUOTED: i32 = 2       # reading inside an open quote
_ST_QUOTE_IN_QUOTED: i32 = 3 # just saw quotechar while IN_QUOTED - ambiguous
                              # (closing quote, or the first half of a
                              # doubled-quotechar escape) until the next byte

class RowParser:
	__delimiter: u8
	__quotechar: u8
	__quotechar_str: str
	__state: i32
	__resuming: bool          # True if the previous feed_line() call ended
	                          # mid-quoted-field (MaybeRow(False,...) -
	                          # needs another line)
	__fields: list[str]
	__current_field: str

	def __init__( self, delimiter: str = ',', quotechar: str = '"' ) -> None:
		if delimiter.byte_len() != 1:
			sys.panic( 'RowParser: delimiter must be exactly one byte' )
		if quotechar.byte_len() != 1:
			sys.panic( 'RowParser: quotechar must be exactly one byte' )
		self.__delimiter = delimiter.get_const_ptr()[0]
		self.__quotechar = quotechar.get_const_ptr()[0]
		self.__quotechar_str = quotechar
		self.__state = _ST_START_FIELD
		self.__resuming = False
		self.__fields = list[str]()
		self.__current_field = ''

	def _finish_field( self ) -> None:
		self.__fields.append( self.__current_field ).unwrap( 'RowParser: field append failed' )
		self.__current_field = ''

	def feed_line( self, line: str ) -> Result[MaybeRow, CsvError]:
		''' line has its own trailing line-ending already stripped by the
		caller. Returns Ok(MaybeRow(True, row)) when a full record completed
		on this line, Ok(MaybeRow(False, ...)) when the line ended
		mid-quoted-field (feed the next line to this SAME RowParser instance
		to continue the record), or Err on malformed input (a bare,
		non-doubled quotechar appearing right after a field's closing quote,
		before the next delimiter). '''
		if self.__resuming:
			self.__current_field = self.__current_field + '\n'
			self.__state = _ST_IN_QUOTED
			self.__resuming = False
		else:
			self.__fields = list[str]()
			self.__current_field = ''
			self.__state = _ST_START_FIELD

		data: ConstPtr[u8] = line.get_const_ptr()
		n: usize = line.byte_len()
		i: usize = 0
		piece_start: usize = 0

		while i < n:
			b: u8 = data[i]
			if self.__state == _ST_START_FIELD:
				if b == self.__quotechar:
					self.__state = _ST_IN_QUOTED
					with compiler.wrap_arithmetic:
						i += 1
					piece_start = i
				elif b == self.__delimiter:
					self._finish_field()
					with compiler.wrap_arithmetic:
						i += 1
					piece_start = i
				else:
					self.__state = _ST_IN_UNQUOTED
					piece_start = i
			elif self.__state == _ST_IN_UNQUOTED:
				if b == self.__delimiter:
					self.__current_field = self.__current_field + _substr( line, piece_start, i )
					self._finish_field()
					self.__state = _ST_START_FIELD
					with compiler.wrap_arithmetic:
						i += 1
					piece_start = i
				else:
					with compiler.wrap_arithmetic:
						i += 1
			elif self.__state == _ST_IN_QUOTED:
				if b == self.__quotechar:
					self.__current_field = self.__current_field + _substr( line, piece_start, i )
					self.__state = _ST_QUOTE_IN_QUOTED
					with compiler.wrap_arithmetic:
						i += 1
				else:
					with compiler.wrap_arithmetic:
						i += 1
			else: # _ST_QUOTE_IN_QUOTED
				if b == self.__quotechar:
					self.__current_field = self.__current_field + self.__quotechar_str
					self.__state = _ST_IN_QUOTED
					with compiler.wrap_arithmetic:
						i += 1
					piece_start = i
				elif b == self.__delimiter:
					self._finish_field()
					self.__state = _ST_START_FIELD
					with compiler.wrap_arithmetic:
						i += 1
					piece_start = i
				else:
					return Result.Err( CsvError( 'unexpected character after closing quote' ))

		if self.__state == _ST_IN_QUOTED:
			self.__current_field = self.__current_field + _substr( line, piece_start, n )
			self.__resuming = True
			pending: list[str] = list[str]()
			return Result.Ok( MaybeRow( False, pending ))
		elif self.__state == _ST_QUOTE_IN_QUOTED:
			self._finish_field()
			return Result.Ok( MaybeRow( True, self.__fields ))
		elif self.__state == _ST_IN_UNQUOTED:
			self.__current_field = self.__current_field + _substr( line, piece_start, n )
			self._finish_field()
			return Result.Ok( MaybeRow( True, self.__fields ))
		else: # _ST_START_FIELD
			if n == 0:
				return Result.Ok( MaybeRow( True, self.__fields )) # already empty - reset at feed_line's own top
			self._finish_field() # trailing empty field after a final delimiter
			return Result.Ok( MaybeRow( True, self.__fields ))


# ---------------------------------------------------------------------------
# format_row - pure, no I/O. Symmetric to RowParser above.
# ---------------------------------------------------------------------------

def _format_field( field: str, delimiter: str, quotechar: str ) -> str:
	# deliberately if/elif, NOT one `or`-chained boolean expression: a chain
	# of 3+ `field.find(...) != isize(-1)` comparisons joined by `or`
	# crashes at runtime under MSVC specifically (STATUS_BREAKPOINT, heap-
	# corruption-flavored - confirmed via a minimal repro isolated down to
	# exactly this shape; 2-deep `or` chains are fine). Real compiler bug
	# in short-circuit-chain temporary cleanup, not something csv.py can
	# fix - this is the workaround.
	needs_quote: bool = False
	if field.find( delimiter ) != isize( -1 ):
		needs_quote = True
	elif field.find( quotechar ) != isize( -1 ):
		needs_quote = True
	elif field.find( '\r' ) != isize( -1 ):
		needs_quote = True
	elif field.find( '\n' ) != isize( -1 ):
		needs_quote = True
	if not needs_quote:
		return field
	doubled_quote: str = quotechar + quotechar
	escaped: str = field.replace( quotechar, doubled_quote )
	return quotechar + escaped + quotechar

def format_row( row: list[str], delimiter: str = ',', quotechar: str = '"' ) -> str:
	''' one CSV record line, WITHOUT a trailing line terminator (callers
	writing to a file append "\r\n"). Quotes a field iff it contains
	delimiter, quotechar, '\r', or '\n' (QUOTE_MINIMAL-equivalent); doubles
	any embedded quotechar. '''
	line: str = ''
	n: usize = row.__len__()
	i: usize = 0
	while i < n:
		field: str = row.__getitem__( i ).unwrap( 'format_row: index in range' )
		line = line + _format_field( field, delimiter, quotechar )
		with compiler.wrap_arithmetic:
			next_i: usize = i + 1
		if next_i < n:
			line = line + delimiter
		i = next_i
	return line


# ---------------------------------------------------------------------------
# _LineReader - chunked, buffered line reading over a BinaryReader. No
# text-mode file abstraction exists anywhere in lib/ today (only raw byte
# reads) - this builds line-splitting + UTF-8 decode from scratch. An
# ordinary class, not a generator (see the module docstring's point 1).
# Accepts bare "\n" or "\r\n" as a line ending; strips it from the returned
# line either way.
# ---------------------------------------------------------------------------

_LF: u8 = 0x0A
_CR: u8 = 0x0D
_LINE_READER_INITIAL_CAP: usize = 4096

class _LineReader:
	__src: BinaryReader
	__buf: Ptr[u8]
	__cap: usize
	__start: usize # offset of the first unconsumed byte in __buf
	__fill: usize  # offset one past the last valid byte in __buf
	__eof: bool    # True once __src.read() returned 0 bytes

	def __init__( self, src: BinaryReader ) -> None:
		self.__src = src
		self.__cap = _LINE_READER_INITIAL_CAP
		self.__buf = sys.alloc[u8]( self.__cap )
		self.__start = 0
		self.__fill = 0
		self.__eof = False

	def __del__( self ) -> None:
		sys.free( self.__buf )

	def _find_lf( self ) -> usize|None:
		i: usize = self.__start
		while i < self.__fill:
			if self.__buf[i] == _LF:
				return i
			with compiler.wrap_arithmetic:
				i += 1
		return None

	def _refill( self ) -> Result[None, CsvError]:
		if self.__start > 0:
			with compiler.wrap_arithmetic:
				unconsumed: usize = self.__fill - self.__start
			with compiler.wrap_arithmetic:
				tail: ConstPtr[u8] = compiler.cast( ConstPtr[u8], self.__buf + self.__start )
			sys.memmove( self.__buf, tail, unconsumed )
			self.__start = 0
			self.__fill = unconsumed
		if self.__fill == self.__cap:
			with compiler.panic_arithmetic( 'a single csv line exceeded the line-buffer growth ceiling' ):
				new_cap: usize = self.__cap * 2
			new_buf: Ptr[u8] = sys.alloc[u8]( new_cap )
			old_view: ConstPtr[u8] = compiler.cast( ConstPtr[u8], self.__buf )
			sys.memcpy( new_buf, old_view, self.__fill )
			sys.free( self.__buf )
			self.__buf = new_buf
			self.__cap = new_cap
		with compiler.wrap_arithmetic:
			space: usize = self.__cap - self.__fill
		with compiler.wrap_arithmetic:
			dest: Ptr[u8] = self.__buf + self.__fill
		match self.__src.read( dest, space ):
			case Result.Ok( n ):
				if n == 0:
					self.__eof = True
				else:
					with compiler.wrap_arithmetic:
						self.__fill += n
				return Result.Ok( None )
			case Result.Err( e ):
				return Result.Err( CsvError( 'csv._LineReader: read failed' ))

	def _decode_range( self, start: usize, end: usize ) -> Result[str, CsvError]:
		# NOT `defer( sys.free( out ))`: this function has two return points
		# (inside the match below), and duplicating a deferred sys.free(out)
		# across both crashed gcc's compile specifically - under a non-
		# Windows target, sys.free takes Ptr[None] (not Ptr[u8] like on
		# Windows, see lib/sys.py), so out: Ptr[u8] needs an implicit cast
		# to reconcile, and something about that cast being synthesized
		# once per duplicated defer site produced a genuine "redeclaration
		# of '$t5' with no linkage" - confirmed via a real gcc compile,
		# clang/MSVC never caught it (Ptr[u8] matches sys.free's Windows
		# overload exactly, no cast, no duplication-sensitive codegen path
		# there). A single explicit sys.free(out) before one unified return
		# sidesteps it entirely - a real compiler bug, not something this
		# module can fix at the source level.
		with compiler.panic_arithmetic( 'bounded by buffer fill, cannot overflow' ):
			piece_len: usize = end - start
			buf_size: usize = piece_len + 1
		out: Ptr[u8] = sys.alloc[u8]( buf_size )
		with compiler.wrap_arithmetic:
			src: ConstPtr[u8] = compiler.cast( ConstPtr[u8], self.__buf + start )
		sys.memcpy( out, src, piece_len )
		out[piece_len] = 0
		const_out: ConstPtr[u8] = compiler.cast( ConstPtr[u8], out )
		ok: bool = False
		decoded: str = ''
		match str.from_cstr( const_out, buf_size ):
			case Result.Ok( s ):
				ok = True
				decoded = s
			case Result.Err( e ):
				pass
		sys.free( out )
		if ok:
			return Result.Ok( decoded )
		return Result.Err( CsvError( 'csv._LineReader: invalid UTF-8 in input' ))

	def next_line( self ) -> Result[MaybeLine, CsvError]:
		''' Ok(MaybeLine(True, line)) with one line, its own trailing
		"\r\n"/"\n" stripped, Ok(MaybeLine(False, ...)) at clean
		end-of-file, or Err wrapping an underlying read failure or a UTF-8
		decode failure. '''
		while True:
			found: usize|None = self._find_lf()
			if found is None:
				if self.__eof:
					if self.__start < self.__fill:
						end: usize = self.__fill
						if end > self.__start:
							with compiler.wrap_arithmetic:
								last: usize = end - 1
							if self.__buf[last] == _CR:
								end = last
						line: str = self._decode_range( self.__start, end ).or_return()
						self.__start = self.__fill
						return Result.Ok( MaybeLine( True, line ))
					return Result.Ok( MaybeLine( False, '' ))
				self._refill().or_return()
				continue
			end2: usize = found
			if end2 > self.__start:
				with compiler.wrap_arithmetic:
					last2: usize = end2 - 1
				if self.__buf[last2] == _CR:
					end2 = last2
			line2: str = self._decode_range( self.__start, end2 ).or_return()
			with compiler.wrap_arithmetic:
				self.__start = found + 1
			return Result.Ok( MaybeLine( True, line2 ))


# ---------------------------------------------------------------------------
# Reader - combines _LineReader + RowParser into row-at-a-time reading over
# a real BinaryReader. Consumers drive it with an explicit loop, per the
# module docstring's point 1 (Result-returning __next__ is never for-loop
# compatible in this language).
# ---------------------------------------------------------------------------

class Reader:
	__lines: _LineReader
	__parser: RowParser

	def __init__( self, src: BinaryReader, delimiter: str = ',', quotechar: str = '"' ) -> None:
		self.__lines = _LineReader( src )
		self.__parser = RowParser( delimiter, quotechar )

	def __next__( self ) -> Result[MaybeRow, CsvError]:
		''' Ok(MaybeRow(True, row)) for each record, Ok(MaybeRow(False, ...))
		once the underlying source is exhausted, or Err on a read failure,
		a UTF-8 decode failure, or malformed CSV syntax. '''
		while True:
			match self.__lines.next_line():
				case Result.Ok( ml ):
					if not ml.has_line:
						empty_row: list[str] = list[str]()
						return Result.Ok( MaybeRow( False, empty_row ))
					match self.__parser.feed_line( ml.line ):
						case Result.Ok( m ):
							if m.has_row:
								return Result.Ok( m )
							# else: line ended mid-quoted-field - loop again
						case Result.Err( e2 ):
							return Result.Err( e2 )
				case Result.Err( e ):
					return Result.Err( e )


def reader( path: str, delimiter: str = ',', quotechar: str = '"' ) -> Result[Reader, OSError]:
	''' opens path for reading and wraps it in a Reader. '''
	src: BinaryReader = File.binary_reader( path ).or_return()
	return Result.Ok( Reader( src, delimiter, quotechar ))


# ---------------------------------------------------------------------------
# Writer - row-at-a-time writing over a real BinaryWriter. No iteration
# involved on the write side, so none of the Reader/DictReader constraints
# above apply here.
# ---------------------------------------------------------------------------

class Writer:
	__sink: BinaryWriter # NOT named __out - collides with the legacy Win32
	                      # SAL annotation macro of that name and corrupts
	                      # the generated C struct field (confirmed via a
	                      # real clang compile failure: "expected member
	                      # name or ';'" on `struct ...* __out;`)
	__delimiter: str
	__quotechar: str

	def __init__( self, out: BinaryWriter, delimiter: str = ',', quotechar: str = '"' ) -> None:
		self.__sink = out
		self.__delimiter = delimiter
		self.__quotechar = quotechar

	def _write_all( self, s: str ) -> Result[None, OSError]:
		total: usize = s.byte_len()
		data: ConstPtr[u8] = s.get_const_ptr()
		written: usize = 0
		while written < total:
			with compiler.wrap_arithmetic:
				remaining: usize = total - written
				offset: ConstPtr[u8] = data + written
			n: usize = self.__sink.write( offset, remaining ).or_return()
			with compiler.wrap_arithmetic:
				written += n
		return Result.Ok( None )

	def writerow( self, row: list[str] ) -> Result[None, OSError]:
		line: str = format_row( row, self.__delimiter, self.__quotechar ) + '\r\n'
		return self._write_all( line )

	def writerows( self, rows: list[list[str]] ) -> Result[None, OSError]:
		n: usize = rows.__len__()
		i: usize = 0
		while i < n:
			row: list[str] = rows.__getitem__( i ).unwrap( 'writerows: index in range' )
			self.writerow( row ).or_return()
			with compiler.wrap_arithmetic:
				i += 1
		return Result.Ok( None )

	def close( self ) -> None:
		self.__sink.close()


def writer( path: str, delimiter: str = ',', quotechar: str = '"' ) -> Result[Writer, OSError]:
	''' opens path for writing (truncating any existing file) and wraps it
	in a Writer. '''
	out: BinaryWriter = File.binary_writer( path ).or_return()
	return Result.Ok( Writer( out, delimiter, quotechar ))


# ---------------------------------------------------------------------------
# DictReader / DictWriter - dict[str,str] rows on top of Reader/Writer.
# ---------------------------------------------------------------------------

class MaybeDictRow:
	''' row-or-end-of-input carrier - dict[K,V] is generic, same union
	restriction as MaybeRow/MaybeLine above. row is only meaningful when
	has_row is True. '''
	has_row: bool
	row: dict[str,str]

	def __init__( self, has_row: bool, row: dict[str,str] ) -> None:
		self.has_row = has_row
		self.row = row


class DictReader:
	__rows: Reader
	__fieldnames: list[str]

	def __init__( self, rows: Reader, fieldnames: list[str] ) -> None:
		''' takes an ALREADY-CONSTRUCTED Reader (not a raw path/BinaryReader)
		so dict_reader() below can consume the header row itself first and
		hand over the SAME Reader (and its already-buffered _LineReader
		state) rather than opening a second, independent Reader over the
		same file that would silently skip whatever the first one had
		already buffered past the header line. '''
		self.__rows = rows
		self.__fieldnames = fieldnames

	def __next__( self ) -> Result[MaybeDictRow, CsvError]:
		''' zips the next row against fieldnames (insertion-ordered, so
		enumeration order matches fieldnames order) via __setitem__. Short
		rows: missing trailing keys get '' (restval). Long rows: extra
		values beyond len(fieldnames) are dropped (v1 gap - Python's
		restkey extras-as-list is deferred). '''
		match self.__rows.__next__():
			case Result.Ok( m ):
				if not m.has_row:
					empty: dict[str,str] = dict[str,str]()
					return Result.Ok( MaybeDictRow( False, empty ))
				d: dict[str,str] = dict[str,str]()
				n: usize = self.__fieldnames.__len__()
				row_len: usize = m.row.__len__()
				i: usize = 0
				while i < n:
					key: str = self.__fieldnames.__getitem__( i ).unwrap( 'DictReader: fieldname index in range' )
					value: str = ''
					if i < row_len:
						value = m.row.__getitem__( i ).unwrap( 'DictReader: row index in range' )
					d.__setitem__( key, value )
					with compiler.wrap_arithmetic:
						i += 1
				return Result.Ok( MaybeDictRow( True, d ))
			case Result.Err( e ):
				return Result.Err( e )


def dict_reader( path: str, delimiter: str = ',', quotechar: str = '"' ) -> Result[DictReader, CsvError]:
	''' opens path, reads its first row as the field names, and returns a
	DictReader positioned at the first DATA row. '''
	src: BinaryReader
	match File.binary_reader( path ):
		case Result.Ok( s ):
			src = s
		case Result.Err( e ):
			return Result.Err( CsvError( 'csv.dict_reader: could not open file' ))

	r = Reader( src, delimiter, quotechar )
	fieldnames: list[str] = list[str]()
	match r.__next__():
		case Result.Ok( m ):
			if m.has_row:
				fieldnames = m.row
			# else: empty file - fieldnames stays an empty list
		case Result.Err( e2 ):
			return Result.Err( e2 )

	return Result.Ok( DictReader( r, fieldnames ))


class DictWriter:
	__rows: Writer
	__fieldnames: list[str]

	def __init__( self, rows: Writer, fieldnames: list[str] ) -> None:
		self.__rows = rows
		self.__fieldnames = fieldnames

	def writeheader( self ) -> Result[None, OSError]:
		return self.__rows.writerow( self.__fieldnames )

	def writerow( self, row: dict[str,str] ) -> Result[None, OSError]:
		''' projects row through fieldnames order into a list[str]. A key
		missing from row gets '' (restval), matching DictReader's own
		short-row handling; extra keys in row beyond fieldnames are
		ignored. '''
		n: usize = self.__fieldnames.__len__()
		values: list[str] = list[str]()
		i: usize = 0
		while i < n:
			key: str = self.__fieldnames.__getitem__( i ).unwrap( 'DictWriter: fieldname index in range' )
			value: str = ''
			match row.__getitem__( key ):
				case Result.Ok( v ):
					value = v
				case Result.Err( e ):
					pass # key missing from row - value stays ''
			values.append( value ).unwrap( 'DictWriter: value append failed' )
			with compiler.wrap_arithmetic:
				i += 1
		return self.__rows.writerow( values )

	def close( self ) -> None:
		self.__rows.close()


def dict_writer( path: str, fieldnames: list[str], delimiter: str = ',', quotechar: str = '"' ) -> Result[DictWriter, OSError]:
	''' opens path for writing (truncating any existing file) and wraps it
	in a DictWriter. Does NOT write the header row automatically - call
	writeheader() explicitly, matching Python's own DictWriter. '''
	w: Writer = writer( path, delimiter, quotechar ).or_return()
	return Result.Ok( DictWriter( w, fieldnames ))
