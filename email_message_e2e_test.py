import unittest

import test_support
from compiler import Compiler
from discovery import Discovery


class EmailMessageEndToEndTests( test_support.RealCompileMixin, unittest.TestCase ):
	''' lib/email/message.py - a full build -> as_string() -> parse round
	trip through the public API: a multipart message with a plain text
	part, a base64 Content-Transfer-Encoding part, and an RFC 2047 encoded
	Subject header, mirroring csv_e2e_test.py's write-then-read-back
	shape. '''

	def setUp( self ) -> None:
		self.discovery = Discovery( import_builtins = True )
		self.compiler = Compiler( self.discovery )

	@unittest.skipUnless( test_support.HAS_CC, 'no C compiler (clang/gcc/msvc) found - skipping' )
	def test_programs_compile_and_run( self ) -> None:
		self.assert_programs_run([
			( 'build_serialize_reparse_multipart_with_cte_and_rfc2047', '''
import compiler
from email.message import Message, message_from_string, decode_header

def main() -> i32:
	outer: Message = Message()
	outer.add_header( 'Subject', '=?utf-8?B?aGVsbG8=?= world' )
	outer.add_header( 'From', 'sender@example.com' )
	outer.add_header( 'To', 'recipient@example.com' )

	text_part: Message = Message()
	text_part.add_header( 'Content-Type', 'text/plain; charset=utf-8' )
	text_part.set_payload( 'This is the plain text body.' )
	outer.attach( text_part )

	binary_part: Message = Message()
	binary_part.add_header( 'Content-Type', 'application/octet-stream' )
	binary_part.add_header( 'Content-Transfer-Encoding', 'base64' )
	binary_part.add_header( 'Content-Disposition', 'attachment; filename="note.txt"' )
	binary_part.set_payload( 'aGVsbG8sIGJpbmFyeSB3b3JsZCE=' ) # base64("hello, binary world!")
	outer.attach( binary_part )

	wire_text: str = outer.as_string()

	parsed: Message
	match message_from_string( wire_text ):
		case Result.Ok( m ):
			parsed = m
		case Result.Err( e ):
			return 1

	# --- headers survived the round trip ---
	subj_raw: str|None = parsed.get( 'Subject' )
	if subj_raw is None:
		return 2
	subj_decoded: str
	match decode_header( subj_raw ):
		case Result.Ok( s ):
			subj_decoded = s
		case Result.Err( e2 ):
			return 3
	if subj_decoded != 'hello world':
		return 4

	frm: str|None = parsed.get( 'From' )
	if frm is None:
		return 5
	if frm != 'sender@example.com':
		return 6

	# --- structure survived the round trip ---
	if not parsed.is_multipart():
		return 7
	parts: list[Message] = parsed.get_parts()
	if parts.__len__() != 2:
		return 8

	p1: Message = parts.__getitem__( 0 ).unwrap( 'p1' )
	if p1.get_content_type() != 'text/plain':
		return 9
	if p1.get_payload() != 'This is the plain text body.':
		return 10

	p2: Message = parts.__getitem__( 1 ).unwrap( 'p2' )
	if p2.get_content_type() != 'application/octet-stream':
		return 11
	fn: str|None = p2.get_filename()
	if fn is None:
		return 12
	if fn != 'note.txt':
		return 13

	decoded_bytes: bytes
	match p2.get_payload_decoded():
		case Result.Ok( d ):
			decoded_bytes = d
		case Result.Err( e3 ):
			return 14
	decoded_str: str = decoded_bytes.decode().unwrap( 'decoded_str' )
	if decoded_str != 'hello, binary world!':
		return 15

	# --- re-serializing the reparsed message reparses identically again
	# (idempotent round trip - not necessarily byte-identical to wire_text,
	# since header folding/boundary tokens may format differently, but
	# structurally equivalent) ---
	wire_text2: str = parsed.as_string()
	reparsed2: Message
	match message_from_string( wire_text2 ):
		case Result.Ok( m2 ):
			reparsed2 = m2
		case Result.Err( e4 ):
			return 16
	if not reparsed2.is_multipart():
		return 17
	if reparsed2.get_parts().__len__() != 2:
		return 18

	return 0
''' ),
		])
