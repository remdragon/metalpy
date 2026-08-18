import unittest

import test_support
from compiler import Compiler
from discovery import Discovery


class CsvRowParserTests( test_support.RealCompileMixin, unittest.TestCase ):
	''' PLAN csv.py, Step 1+2 - RowParser.feed_line and format_row are pure
	string logic (no I/O, no generators - see the module docstring in
	lib/csv.py for why generators aren't used, and why MaybeRow exists
	instead of list[str]|None). '''

	def setUp( self ) -> None:
		self.discovery = Discovery( import_builtins = True )
		self.compiler = Compiler( self.discovery )

	@unittest.skipUnless( test_support.HAS_CC, 'no C compiler (clang/gcc/msvc) found - skipping' )
	def test_programs_compile_and_run( self ) -> None:
		self.assert_programs_run([
			( 'simple_unquoted_row', '''
import csv

def main() -> i32:
	p = csv.RowParser()
	match p.feed_line( 'a,b,c' ):
		case Result.Ok( m ):
			if not m.has_row:
				return 1
			if m.row.__len__() != 3:
				return 2
			if m.row.__getitem__( 0 ).unwrap( 'x' ) != 'a':
				return 3
			if m.row.__getitem__( 1 ).unwrap( 'x' ) != 'b':
				return 4
			if m.row.__getitem__( 2 ).unwrap( 'x' ) != 'c':
				return 5
			return 0
		case Result.Err( e ):
			return 6
''' ),
			( 'empty_line_yields_empty_row', '''
import csv

def main() -> i32:
	p = csv.RowParser()
	match p.feed_line( '' ):
		case Result.Ok( m ):
			if not m.has_row:
				return 1
			if m.row.__len__() != 0:
				return 2
			return 0
		case Result.Err( e ):
			return 3
''' ),
			( 'trailing_empty_field', '''
import csv

def main() -> i32:
	p = csv.RowParser()
	match p.feed_line( 'a,b,' ):
		case Result.Ok( m ):
			if not m.has_row:
				return 1
			if m.row.__len__() != 3:
				return 2
			if m.row.__getitem__( 2 ).unwrap( 'x' ) != '':
				return 3
			return 0
		case Result.Err( e ):
			return 4
''' ),
			( 'leading_and_consecutive_empty_fields', '''
import csv

def main() -> i32:
	p = csv.RowParser()
	match p.feed_line( ',a,,b' ):
		case Result.Ok( m ):
			if not m.has_row:
				return 1
			if m.row.__len__() != 4:
				return 2
			if m.row.__getitem__( 0 ).unwrap( 'x' ) != '':
				return 3
			if m.row.__getitem__( 1 ).unwrap( 'x' ) != 'a':
				return 4
			if m.row.__getitem__( 2 ).unwrap( 'x' ) != '':
				return 5
			if m.row.__getitem__( 3 ).unwrap( 'x' ) != 'b':
				return 6
			return 0
		case Result.Err( e ):
			return 7
''' ),
			( 'quoted_field_with_embedded_delimiter', '''
import csv

def main() -> i32:
	p = csv.RowParser()
	match p.feed_line( '"a,b",c' ):
		case Result.Ok( m ):
			if not m.has_row:
				return 1
			if m.row.__len__() != 2:
				return 2
			if m.row.__getitem__( 0 ).unwrap( 'x' ) != 'a,b':
				return 3
			if m.row.__getitem__( 1 ).unwrap( 'x' ) != 'c':
				return 4
			return 0
		case Result.Err( e ):
			return 5
''' ),
			( 'doubled_quotechar_escape', '''
import csv

def main() -> i32:
	p = csv.RowParser()
	match p.feed_line( '"a""b",c' ):
		case Result.Ok( m ):
			if not m.has_row:
				return 1
			if m.row.__len__() != 2:
				return 2
			if m.row.__getitem__( 0 ).unwrap( 'x' ) != 'a"b':
				return 3
			return 0
		case Result.Err( e ):
			return 4
''' ),
			( 'quoted_field_spans_two_lines_embedded_newline', '''
import csv

def main() -> i32:
	p = csv.RowParser()
	match p.feed_line( 'a,"b' ):
		case Result.Ok( m ):
			if m.has_row:
				return 1 # should not complete yet - still inside the quote
		case Result.Err( e ):
			return 2
	match p.feed_line( 'c",d' ):
		case Result.Ok( m ):
			if not m.has_row:
				return 3
			if m.row.__len__() != 3:
				return 4
			if m.row.__getitem__( 0 ).unwrap( 'x' ) != 'a':
				return 5
			if m.row.__getitem__( 1 ).unwrap( 'x' ) != 'b\\nc':
				return 6
			if m.row.__getitem__( 2 ).unwrap( 'x' ) != 'd':
				return 7
			return 0
		case Result.Err( e ):
			return 8
''' ),
			( 'unexpected_char_after_closing_quote_is_error', '''
import csv

def main() -> i32:
	p = csv.RowParser()
	match p.feed_line( '"a"b,c' ):
		case Result.Ok( m ):
			return 1
		case Result.Err( e ):
			return 0
''' ),
			( 'row_parser_reusable_across_records', '''
import csv

def main() -> i32:
	p = csv.RowParser()
	match p.feed_line( 'a,b' ):
		case Result.Ok( m ):
			if not m.has_row or m.row.__len__() != 2:
				return 1
		case Result.Err( e ):
			return 2
	match p.feed_line( 'c,d,e' ):
		case Result.Ok( m ):
			if not m.has_row:
				return 3
			if m.row.__len__() != 3:
				return 4
			if m.row.__getitem__( 0 ).unwrap( 'x' ) != 'c':
				return 5
			return 0
		case Result.Err( e ):
			return 6
''' ),
			( 'format_row_no_quoting_needed', '''
import csv

def main() -> i32:
	row: list[str] = list[str]()
	row.append( 'a' ).unwrap( 'x' )
	row.append( 'b' ).unwrap( 'x' )
	row.append( 'c' ).unwrap( 'x' )
	line: str = csv.format_row( row )
	if line != 'a,b,c':
		return 1
	return 0
''' ),
			( 'format_row_quotes_field_with_delimiter_and_doubles_quotechar', '''
import csv

def main() -> i32:
	row: list[str] = list[str]()
	row.append( 'a,b' ).unwrap( 'x' )
	row.append( 'has"quote' ).unwrap( 'x' )
	row.append( 'plain' ).unwrap( 'x' )
	line: str = csv.format_row( row )
	if line != '"a,b","has""quote",plain':
		return 1
	return 0
''' ),
			( 'format_row_then_parse_round_trip', '''
import csv

def main() -> i32:
	row: list[str] = list[str]()
	row.append( 'simple' ).unwrap( 'x' )
	row.append( 'with,comma' ).unwrap( 'x' )
	row.append( 'with"quote' ).unwrap( 'x' )
	row.append( '' ).unwrap( 'x' )
	line: str = csv.format_row( row )
	p = csv.RowParser()
	match p.feed_line( line ):
		case Result.Ok( m ):
			if not m.has_row:
				return 1
			if m.row.__len__() != 4:
				return 2
			if m.row.__getitem__( 0 ).unwrap( 'x' ) != 'simple':
				return 3
			if m.row.__getitem__( 1 ).unwrap( 'x' ) != 'with,comma':
				return 4
			if m.row.__getitem__( 2 ).unwrap( 'x' ) != 'with"quote':
				return 5
			if m.row.__getitem__( 3 ).unwrap( 'x' ) != '':
				return 6
			return 0
		case Result.Err( e ):
			return 7
''' ),
			# investigation repro for the compiler-bug comment at
			# lib/csv.py:244-250 (_format_field's if/elif workaround) -
			# NOT wired into _format_field itself, that shape must stay as
			# the shipped if/elif regardless of this test's outcome. This
			# is the exact rejected shape: a single `or`-chain of 4
			# Result-returning `.is_ok()` calls, joined all the way instead
			# of split into separate if/elif branches - uses index()
			# (not find(), which now returns a plain isize sentinel, not a
			# Result, since the str.find()/index() convention swap - see
			# lib/builtins/__init__.py) so this still genuinely exercises
			# an RC-leaf @union (Result[usize,IndexError]) BoolOp operand,
			# the actual bug shape this test needs. Cases cover every
			# short-circuit position (no match at all - every operand
			# actually runs; match on the 1st/2nd/3rd/4th operand - all
			# preceding operands run, everything after is skipped) since a
			# short-circuit-cleanup bug specifically needs operands that
			# are genuinely skipped at runtime, not just present in the
			# source.
			( 'boolop_or_chain_of_find_is_ok_repro', '''
def needs_quote_or_chain( field: str, delimiter: str, quotechar: str ) -> bool:
	return field.index( delimiter ).is_ok() or field.index( quotechar ).is_ok() or field.index( '\\r' ).is_ok() or field.index( '\\n' ).is_ok()

def main() -> i32:
	if needs_quote_or_chain( 'plain', ',', '"' ) != False: # no match - all 4 operands run
		return 1
	if needs_quote_or_chain( 'a,b', ',', '"' ) != True: # 1st operand matches - short-circuits immediately
		return 2
	if needs_quote_or_chain( 'a"b', ',', '"' ) != True: # 2nd operand matches - 1st ran and was false, then short-circuits
		return 3
	if needs_quote_or_chain( 'a\\rb', ',', '"' ) != True: # 3rd operand matches - 1st/2nd ran false, then short-circuits
		return 4
	if needs_quote_or_chain( 'a\\nb', ',', '"' ) != True: # 4th (last) operand matches - all 4 operands run
		return 5
	return 0
''' ),
		])


if __name__ == '__main__':
	unittest.main()
