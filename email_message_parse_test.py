import unittest

import test_support
from compiler import Compiler
from discovery import Discovery


class EmailMessageParseTests( test_support.RealCompileMixin, unittest.TestCase ):
	''' lib/email/message.py - message_from_string: header parsing, RFC 5322
	folded continuation lines, and Content-Type/-Disposition parameter
	parsing (quoted and unquoted). '''

	def setUp( self ) -> None:
		self.discovery = Discovery( import_builtins = True )
		self.compiler = Compiler( self.discovery )

	@unittest.skipUnless( test_support.HAS_CC, 'no C compiler (clang/gcc/msvc) found - skipping' )
	def test_programs_compile_and_run( self ) -> None:
		self.assert_programs_run([
			( 'simple_headers_and_body', '''
import compiler
from email.message import message_from_string, Message

def main() -> i32:
	raw: str = 'Subject: hello\\r\\nFrom: a@b.com\\r\\nTo: c@d.com\\r\\n\\r\\nbody text\\r\\n'
	msg: Message
	match message_from_string( raw ):
		case Result.Ok( m ):
			msg = m
		case Result.Err( e ):
			return 1
	subj: str|None = msg.get( 'Subject' )
	if subj is None:
		return 2
	if subj != 'hello':
		return 3
	frm: str|None = msg.get( 'From' )
	if frm is None:
		return 4
	if frm != 'a@b.com':
		return 5
	if msg.get_payload() != 'body text\\r\\n':
		return 6
	if msg.is_multipart():
		return 7
	return 0
''' ),
			( 'folded_continuation_line', '''
import compiler
from email.message import message_from_string, Message

def main() -> i32:
	raw: str = 'X-Folded: line one\\r\\n continued with a tab next\\r\\n\\tand more\\r\\nSubject: s\\r\\n\\r\\nbody\\r\\n'
	msg: Message
	match message_from_string( raw ):
		case Result.Ok( m ):
			msg = m
		case Result.Err( e ):
			return 1
	folded: str|None = msg.get( 'X-Folded' )
	if folded is None:
		return 2
	if folded != 'line one continued with a tab next and more':
		return 3
	subj: str|None = msg.get( 'Subject' )
	if subj is None:
		return 4
	if subj != 's':
		return 5
	return 0
''' ),
			( 'malformed_header_line_is_an_error', '''
import compiler
from email.message import message_from_string, Message

def main() -> i32:
	raw: str = 'this is not a header line\\r\\n\\r\\nbody\\r\\n'
	match message_from_string( raw ):
		case Result.Ok( m ):
			return 1 # should have failed to parse
		case Result.Err( e ):
			return 0
''' ),
			( 'no_headers_at_all', '''
import compiler
from email.message import message_from_string, Message

def main() -> i32:
	raw: str = '\\r\\njust a body, no headers'
	msg: Message
	match message_from_string( raw ):
		case Result.Ok( m ):
			msg = m
		case Result.Err( e ):
			return 1
	if msg.keys().__len__() != 0:
		return 2
	if msg.get_payload() != 'just a body, no headers':
		return 3
	return 0
''' ),
			( 'content_type_params_quoted_and_unquoted', '''
import compiler
from email.message import Message

def main() -> i32:
	m: Message = Message()
	m.add_header( 'Content-Type', 'multipart/mixed; boundary="abc;123"; charset=utf-8' )
	b: str|None = m.get_boundary()
	if b is None:
		return 1
	if b != 'abc;123':
		return 2
	cs: str|None = m.get_charset()
	if cs is None:
		return 3
	if cs != 'utf-8':
		return 4
	return 0
''' ),
			( 'content_disposition_filename', '''
import compiler
from email.message import Message

def main() -> i32:
	m: Message = Message()
	m.add_header( 'Content-Disposition', 'attachment; filename="report.pdf"' )
	fn: str|None = m.get_filename()
	if fn is None:
		return 1
	if fn != 'report.pdf':
		return 2
	return 0
''' ),
			( 'content_type_name_fallback_for_filename', '''
import compiler
from email.message import Message

def main() -> i32:
	m: Message = Message()
	m.add_header( 'Content-Type', 'application/octet-stream; name="data.bin"' )
	fn: str|None = m.get_filename()
	if fn is None:
		return 1
	if fn != 'data.bin':
		return 2
	return 0
''' ),
		])
