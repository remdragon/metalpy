import unittest

import test_support
from compiler import Compiler
from discovery import Discovery


class EmailMessageEncodingTests( test_support.RealCompileMixin, unittest.TestCase ):
	''' lib/email/quoprimime.py's quoted-printable codec, Message's
	Content-Transfer-Encoding decode (base64 / quoted-printable), and RFC
	2047 encoded-word header decoding (decode_header). '''

	def setUp( self ) -> None:
		self.discovery = Discovery( import_builtins = True )
		self.compiler = Compiler( self.discovery )

	@unittest.skipUnless( test_support.HAS_CC, 'no C compiler (clang/gcc/msvc) found - skipping' )
	def test_programs_compile_and_run( self ) -> None:
		self.assert_programs_run([
			( 'quoted_printable_round_trip', '''
import compiler
import sys
from email.quoprimime import encode as qp_encode, decode as qp_decode

def main() -> i32:
	src: str = 'Hello, W\\x99rld! =foo=\\r\\nLine2\\ttab and space at end  '
	buf: bytearray = bytearray( src.byte_len() )
	sys.memcpy( buf.get_ptr(), src.get_const_ptr(), src.byte_len() )
	data: bytes = bytes.from_bytearray( move( buf ))
	encoded: str = qp_encode( data )
	decoded: bytes = qp_decode( encoded )
	if len( decoded ) != len( data ):
		return 1
	decoded_str: str = decoded.decode().unwrap( 'd' )
	if decoded_str != src:
		return 2
	return 0
''' ),
			( 'quoted_printable_decode_leniency', '''
import compiler
from email.quoprimime import decode as qp_decode

def main() -> i32:
	d1: bytes = qp_decode( 'abc=\\r\\ndef' )
	if d1.decode().unwrap( 'd1' ) != 'abcdef':
		return 1
	d2: bytes = qp_decode( 'abc=3Ddef' )
	if d2.decode().unwrap( 'd2' ) != 'abc=def':
		return 2
	d3: bytes = qp_decode( 'trailing=' )
	if d3.decode().unwrap( 'd3' ) != 'trailing=':
		return 3
	return 0
''' ),
			( 'message_cte_base64_decode', '''
import compiler
from email.message import Message

def main() -> i32:
	m: Message = Message()
	m.add_header( 'Content-Transfer-Encoding', 'base64' )
	m.set_payload( 'aGVsbG8=' ) # base64("hello")
	match m.get_payload_decoded():
		case Result.Ok( d ):
			if d.decode().unwrap( 'd' ) != 'hello':
				return 1
			return 0
		case Result.Err( e ):
			return 2
''' ),
			( 'message_cte_base64_decode_tolerates_wrapped_lines', '''
import compiler
from email.message import Message

def main() -> i32:
	m: Message = Message()
	m.add_header( 'Content-Transfer-Encoding', 'BASE64' )
	# real MIME bodies wrap base64 at 76 cols with CRLF - validate=False
	# leniency must tolerate the embedded line break
	m.set_payload( 'aGVs\\r\\nbG8=' )
	match m.get_payload_decoded():
		case Result.Ok( d ):
			if d.decode().unwrap( 'd' ) != 'hello':
				return 1
			return 0
		case Result.Err( e ):
			return 2
''' ),
			( 'message_cte_quoted_printable_decode', '''
import compiler
from email.message import Message

def main() -> i32:
	m: Message = Message()
	m.add_header( 'Content-Transfer-Encoding', 'quoted-printable' )
	m.set_payload( 'caf=E9 au lait' )
	match m.get_payload_decoded():
		case Result.Ok( d ):
			if d.__len__() != 12: # "caf" (3) + 0xE9 (1) + " au lait" (8) = 12
				return 1
			return 0
		case Result.Err( e ):
			return 2
''' ),
			( 'message_cte_identity_passthrough', '''
import compiler
from email.message import Message

def main() -> i32:
	m: Message = Message()
	m.set_payload( 'plain text body' )
	match m.get_payload_decoded():
		case Result.Ok( d ):
			if d.decode().unwrap( 'd' ) != 'plain text body':
				return 1
			return 0
		case Result.Err( e ):
			return 2
''' ),
			( 'rfc2047_base64_decode', '''
import compiler
from email.message import decode_header

def main() -> i32:
	match decode_header( '=?utf-8?B?aGVsbG8=?=' ):
		case Result.Ok( s ):
			if s != 'hello':
				return 1
			return 0
		case Result.Err( e ):
			return 2
''' ),
			( 'rfc2047_quoted_printable_decode', '''
import compiler
from email.message import decode_header

def main() -> i32:
	match decode_header( '=?utf-8?Q?Hello=20World?=' ):
		case Result.Ok( s ):
			if s != 'Hello World':
				return 1
			return 0
		case Result.Err( e ):
			return 2
''' ),
			( 'rfc2047_q_underscore_means_space', '''
import compiler
from email.message import decode_header

def main() -> i32:
	match decode_header( '=?utf-8?Q?Hello_World?=' ):
		case Result.Ok( s ):
			if s != 'Hello World':
				return 1
			return 0
		case Result.Err( e ):
			return 2
''' ),
			( 'rfc2047_adjacent_encoded_words_concatenate', '''
import compiler
from email.message import decode_header

def main() -> i32:
	match decode_header( '=?utf-8?Q?Hello?= =?utf-8?Q?World?=' ):
		case Result.Ok( s ):
			if s != 'HelloWorld':
				return 1
			return 0
		case Result.Err( e ):
			return 2
''' ),
			( 'rfc2047_plain_text_passthrough', '''
import compiler
from email.message import decode_header

def main() -> i32:
	match decode_header( 'plain text, no encoding' ):
		case Result.Ok( s ):
			if s != 'plain text, no encoding':
				return 1
			return 0
		case Result.Err( e ):
			return 2
''' ),
			( 'rfc2047_mixed_plain_and_encoded', '''
import compiler
from email.message import decode_header

def main() -> i32:
	match decode_header( 'plain =?utf-8?Q?encoded?= tail' ):
		case Result.Ok( s ):
			if s != 'plain encoded tail':
				return 1
			return 0
		case Result.Err( e ):
			return 2
''' ),
		])
