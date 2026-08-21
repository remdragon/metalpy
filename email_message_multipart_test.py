import unittest

import test_support
from compiler import Compiler
from discovery import Discovery


class EmailMessageMultipartTests( test_support.RealCompileMixin, unittest.TestCase ):
	''' lib/email/message.py - attach()/walk()/get_part(), boundary
	auto-generation, and multipart round-tripping through as_string() /
	message_from_string(), including one level of nesting. '''

	def setUp( self ) -> None:
		self.discovery = Discovery( import_builtins = True )
		self.compiler = Compiler( self.discovery )

	@unittest.skipUnless( test_support.HAS_CC, 'no C compiler (clang/gcc/msvc) found - skipping' )
	def test_programs_compile_and_run( self ) -> None:
		self.assert_programs_run([
			( 'attach_marks_multipart_and_sets_boundary', '''
import compiler
from email.message import Message

def main() -> i32:
	outer: Message = Message()
	if outer.is_multipart():
		return 1
	part: Message = Message()
	part.set_payload( 'hi' )
	outer.attach( part )
	if not outer.is_multipart():
		return 2
	if outer.get_content_type() != 'multipart/mixed':
		return 3
	b: str|None = outer.get_boundary()
	if b is None:
		return 4
	if b.byte_len() == 0:
		return 5
	return 0
''' ),
			( 'two_parts_round_trip', '''
import compiler
from email.message import Message, message_from_string

def main() -> i32:
	outer: Message = Message()
	outer.add_header( 'Subject', 'two parts' )
	p1: Message = Message()
	p1.add_header( 'Content-Type', 'text/plain' )
	p1.set_payload( 'first part body' )
	outer.attach( p1 )
	p2: Message = Message()
	p2.add_header( 'Content-Type', 'text/plain' )
	p2.set_payload( 'second part body' )
	outer.attach( p2 )

	out_str: str = outer.as_string()
	parsed: Message
	match message_from_string( out_str ):
		case Result.Ok( m ):
			parsed = m
		case Result.Err( e ):
			return 1

	if not parsed.is_multipart():
		return 2
	parts: list[Message] = parsed.get_parts()
	if parts.__len__() != 2:
		return 3
	first: Message = parts.__getitem__( 0 ).unwrap( 'first' )
	if first.get_payload() != 'first part body':
		return 4
	second: Message = parts.__getitem__( 1 ).unwrap( 'second' )
	if second.get_payload() != 'second part body':
		return 5

	via_get_part: Message
	part_result: Result[Message, IndexError] = parsed.get_part( 1 )
	match part_result:
		case Result.Ok( gp ):
			via_get_part = gp
		case Result.Err( e2 ):
			return 6
	if via_get_part.get_payload() != 'second part body':
		return 7

	out_of_range: bool = False
	out_of_range_result: Result[Message, IndexError] = parsed.get_part( 5 )
	match out_of_range_result:
		case Result.Ok( _unused ):
			pass
		case Result.Err( e3 ):
			out_of_range = True
	if not out_of_range:
		return 8

	return 0
''' ),
			( 'walk_flattens_pre_order', '''
import compiler
from email.message import Message

def main() -> i32:
	outer: Message = Message()
	p1: Message = Message()
	p1.set_payload( 'p1' )
	outer.attach( p1 )
	p2: Message = Message()
	p2.set_payload( 'p2' )
	outer.attach( p2 )
	walked: list[Message] = outer.walk()
	if walked.__len__() != 3:
		return 1
	first: Message = walked.__getitem__( 0 ).unwrap( 'w0' )
	if not first.is_multipart():
		return 2
	second: Message = walked.__getitem__( 1 ).unwrap( 'w1' )
	if second.get_payload() != 'p1':
		return 3
	third: Message = walked.__getitem__( 2 ).unwrap( 'w2' )
	if third.get_payload() != 'p2':
		return 4
	return 0
''' ),
			( 'nested_multipart_round_trip', '''
import compiler
from email.message import Message, message_from_string

def main() -> i32:
	outer: Message = Message()
	outer.add_header( 'Subject', 'nested' )

	inner: Message = Message()
	ip1: Message = Message()
	ip1.set_payload( 'inner part 1' )
	inner.attach( ip1 )
	ip2: Message = Message()
	ip2.set_payload( 'inner part 2' )
	inner.attach( ip2 )
	outer.attach( inner )

	op2: Message = Message()
	op2.set_payload( 'outer part 2' )
	outer.attach( op2 )

	out_str: str = outer.as_string()
	parsed: Message
	match message_from_string( out_str ):
		case Result.Ok( m ):
			parsed = m
		case Result.Err( e ):
			return 1

	walked: list[Message] = parsed.walk()
	if walked.__len__() != 5: # outer, inner, ip1, ip2, op2
		return 2

	outer_parts: list[Message] = parsed.get_parts()
	if outer_parts.__len__() != 2:
		return 3
	first: Message = outer_parts.__getitem__( 0 ).unwrap( 'first' )
	if not first.is_multipart():
		return 4
	first_parts: list[Message] = first.get_parts()
	if first_parts.__len__() != 2:
		return 5
	fp1: Message = first_parts.__getitem__( 0 ).unwrap( 'fp1' )
	if fp1.get_payload() != 'inner part 1':
		return 6
	fp2: Message = first_parts.__getitem__( 1 ).unwrap( 'fp2' )
	if fp2.get_payload() != 'inner part 2':
		return 7
	second: Message = outer_parts.__getitem__( 1 ).unwrap( 'second' )
	if second.get_payload() != 'outer part 2':
		return 8

	return 0
''' ),
			( 'missing_boundary_is_an_error', '''
import compiler
from email.message import message_from_string, Message

def main() -> i32:
	raw: str = 'Content-Type: multipart/mixed\\r\\n\\r\\nno boundary param\\r\\n'
	match message_from_string( raw ):
		case Result.Ok( m ):
			return 1 # should have failed - multipart/* with no boundary param
		case Result.Err( e ):
			return 0
''' ),
		])
