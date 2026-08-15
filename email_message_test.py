import unittest

import test_support
from compiler import Compiler
from discovery import Discovery


class EmailMessageHeaderTests( test_support.RealCompileMixin, unittest.TestCase ):
	''' lib/email/message.py - the header multimap (add/get/get_all/set/
	delete/keys/values/items), case-insensitivity, and duplicate
	preservation. Modeled on lib/http/client.py's HTTPHeaders. '''

	def setUp( self ) -> None:
		self.discovery = Discovery( import_builtins = True )
		self.compiler = Compiler( self.discovery )

	@unittest.skipUnless( test_support.HAS_CC, 'no C compiler (clang/gcc/msvc) found - skipping' )
	def test_programs_compile_and_run( self ) -> None:
		self.assert_programs_run([
			( 'add_and_get', '''
import compiler
from email.message import Message

def main() -> i32:
	m: Message = Message()
	m.add_header( 'Subject', 'hello' )
	v: str|None = m.get( 'Subject' )
	if v is None:
		return 1
	if v != 'hello':
		return 2
	missing: str|None = m.get( 'X-Nope' )
	if missing is not None:
		return 3
	return 0
''' ),
			( 'case_insensitive_lookup', '''
import compiler
from email.message import Message

def main() -> i32:
	m: Message = Message()
	m.add_header( 'Content-Type', 'text/plain' )
	v: str|None = m.get( 'content-type' )
	if v is None:
		return 1
	if v != 'text/plain':
		return 2
	v2: str|None = m.get( 'CONTENT-TYPE' )
	if v2 is None:
		return 3
	if v2 != 'text/plain':
		return 4
	if not m.__contains__( 'content-TYPE' ):
		return 5
	return 0
''' ),
			( 'get_all_preserves_duplicates_in_order', '''
import compiler
from email.message import Message

def main() -> i32:
	m: Message = Message()
	m.add_header( 'X-Multi', 'one' )
	m.add_header( 'X-Multi', 'two' )
	m.add_header( 'X-Multi', 'three' )
	all_vals: list[str] = m.get_all( 'x-multi' )
	if all_vals.__len__() != 3:
		return 1
	if all_vals.__getitem__( 0 ).unwrap( 'a' ) != 'one':
		return 2
	if all_vals.__getitem__( 1 ).unwrap( 'a' ) != 'two':
		return 3
	if all_vals.__getitem__( 2 ).unwrap( 'a' ) != 'three':
		return 4
	return 0
''' ),
			( 'setitem_adds_not_replaces', '''
import compiler
from email.message import Message

def main() -> i32:
	m: Message = Message()
	m.__setitem__( 'X-Foo', 'a' )
	m.__setitem__( 'X-Foo', 'b' )
	# __setitem__ ADDS, matching Python's own Message.__setitem__ semantics
	if m.get_all( 'X-Foo' ).__len__() != 2:
		return 1
	return 0
''' ),
			( 'replace_header_overwrites_first_and_drops_rest', '''
import compiler
from email.message import Message

def main() -> i32:
	m: Message = Message()
	m.add_header( 'X-Foo', 'a' )
	m.add_header( 'X-Foo', 'b' )
	m.replace_header( 'X-Foo', 'c' )
	all_vals: list[str] = m.get_all( 'X-Foo' )
	if all_vals.__len__() != 1:
		return 1
	if all_vals.__getitem__( 0 ).unwrap( 'a' ) != 'c':
		return 2
	return 0
''' ),
			( 'delitem_removes_all_matching', '''
import compiler
from email.message import Message

def main() -> i32:
	m: Message = Message()
	m.add_header( 'X-Foo', 'a' )
	m.add_header( 'X-Foo', 'b' )
	m.add_header( 'Subject', 'keep me' )
	m.__delitem__( 'x-foo' )
	if m.get_all( 'X-Foo' ).__len__() != 0:
		return 1
	subj: str|None = m.get( 'Subject' )
	if subj is None:
		return 2
	if subj != 'keep me':
		return 3
	return 0
''' ),
			( 'keys_values_items_insertion_order', '''
import compiler
from email.message import Message

def main() -> i32:
	m: Message = Message()
	m.add_header( 'A', '1' )
	m.add_header( 'B', '2' )
	m.add_header( 'A', '3' )
	keys: list[str] = m.keys()
	values: list[str] = m.values()
	items: list[tuple[str,str]] = m.items()
	if keys.__len__() != 3 or values.__len__() != 3 or items.__len__() != 3:
		return 1
	if keys.__getitem__( 0 ).unwrap( 'k' ) != 'A':
		return 2
	if keys.__getitem__( 1 ).unwrap( 'k' ) != 'B':
		return 3
	if keys.__getitem__( 2 ).unwrap( 'k' ) != 'A':
		return 4
	if values.__getitem__( 2 ).unwrap( 'v' ) != '3':
		return 5
	pair: tuple[str,str] = items.__getitem__( 1 ).unwrap( 'i' )
	if pair[0] != 'B' or pair[1] != '2':
		return 6
	return 0
''' ),
			( 'content_type_maintype_subtype_charset_default', '''
import compiler
from email.message import Message

def main() -> i32:
	m: Message = Message()
	# no Content-Type header at all - Python's own Message defaults to text/plain
	if m.get_content_type() != 'text/plain':
		return 1
	if m.get_content_maintype() != 'text':
		return 2
	if m.get_content_subtype() != 'plain':
		return 3
	cs: str|None = m.get_charset()
	if cs is not None:
		return 4

	m.add_header( 'Content-Type', 'text/html; charset=UTF-8' )
	if m.get_content_type() != 'text/html':
		return 5
	if m.get_content_subtype() != 'html':
		return 6
	cs2: str|None = m.get_charset()
	if cs2 is None:
		return 7
	if cs2 != 'utf-8':
		return 8
	return 0
''' ),
		])
