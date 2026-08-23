# xml.py — well-formed XML parsing (not DTD-validating) into a DOM-style
# tree, plus serialization back to text.
#
# XMLNode is a @union (a value type, same shape as JSONValue in lib/json.py -
# see that module's own comment on why a TaggedUnion with RC leaves is
# correctly incref/decref tracked). Element's payload is ElementData, an RC
# class (not the union itself) because an element needs several named
# fields at once (tag, resolved namespace, attributes, children) - unlike
# json.py's Array/Object, which hold list[JSONValue]/dict[str,JSONValue]
# directly as the variant's own payload. ElementData.children: list[XMLNode]
# closes the self-reference one level removed (union -> RC class -> list of
# the same union, rather than json.py's union -> list of the same union
# directly). This exact shape - including ElementData being declared BEFORE
# XMLNode and forward-referencing it in a field annotation - was confirmed
# to actually compile and run via two standalone spikes before any of this
# module was written (see xml_test.py's own first test, mirroring json.py's
# "nothing else in this file should be trusted until this passes" spike).
#
# Namespace scope during parsing is a plain list[tuple[str,str]] of
# (prefix, uri) pairs, "" as the prefix meaning the default namespace -
# NOT a dedicated linked-list class - matching lib/http/client.py's own
# precedent of using list[tuple[str,str]] instead of a dedicated small
# class for an ordered (key,value)-shaped collection. Each element that
# declares new xmlns/xmlns:prefix bindings gets a freshly appended-onto
# copy passed down to its children; an element with no such declarations
# just shares its parent's list by reference. Lookup walks from the end
# (innermost/most-recent declaration wins). This sidesteps the need for
# any Optional-self-referential class field entirely.
#
# Similarly, an attribute/element's resolved namespace URI and an
# unprefixed name's "prefix" are both plain str, with "" as the sentinel
# for "none" rather than str|None - this is not just a convenience, it is
# exactly what the XML namespaces spec itself says: xmlns="" explicitly
# UNDECLARES the default namespace, so "no namespace in scope" and "bound
# to the empty string" are the same state on the wire, not two states this
# module would need to distinguish.
#
# v1 scope, deliberate gaps (not bugs):
#   - not a validating parser: no DTD/entity-declaration processing, no
#     external entity resolution, no XInclude.
#   - Comments and CDATA sections are preserved as tree nodes, but ONLY
#     when they appear inside the element tree (as a child of some
#     element) - there is no Document-level container node, so a comment
#     in the prolog or epilog (outside the root element) is scanned past
#     (so it doesn't break parsing) and then discarded, same as the XML
#     declaration and DOCTYPE are.
#   - attribute-uniqueness is checked on the raw qualified name, not the
#     resolved (namespace_uri, local_name) pair - two differently-prefixed
#     attributes that happen to resolve to the same expanded name are not
#     flagged as duplicates.
#   - the "xmlns"/"xmlns:*" attributes themselves are not resolved into
#     the reserved http://www.w3.org/2000/xmlns/ namespace URI - they keep
#     namespace_uri "" like any other unprefixed/reserved name. They ARE
#     still kept in .attributes for round-trip serialization fidelity.
#   - no pretty-printing: dumps() always produces compact output.
#
# dumps() builds output as a list[str] of pieces + ''.join(pieces), same
# as json.py's dumps() - this codebase has no growable StringBuilder/byte-
# buffer type.

import builtins
import compiler
import sys

# ---------------------------------------------------------------------------
# ASCII byte constants - the XML grammar's structural tokens are themselves
# ASCII-only; only name/text/attribute-value CONTENT needs real UTF-8
# handling (via builtins.utf8_encoded_len/encode_utf8_at below - this module
# never needs to DECODE input UTF-8, only re-encode decoded entity/char-ref
# codepoints, since everything else is passed through byte-for-byte).
# ---------------------------------------------------------------------------

_ASCII_TAB:       u8 = 0x09
_ASCII_LF:        u8 = 0x0A
_ASCII_CR:        u8 = 0x0D
_ASCII_SPACE:     u8 = 0x20
_ASCII_DQUOTE:    u8 = 0x22 # '"'
_ASCII_HASH:      u8 = 0x23
_ASCII_AMP:       u8 = 0x26
_ASCII_SQUOTE:    u8 = 0x27 # "'"
_ASCII_DASH:      u8 = 0x2D
_ASCII_DOT:       u8 = 0x2E
_ASCII_ZERO:      u8 = 0x30
_ASCII_NINE:      u8 = 0x39
_ASCII_COLON:     u8 = 0x3A
_ASCII_SEMICOLON: u8 = 0x3B
_ASCII_LT:        u8 = 0x3C
_ASCII_EQUALS:    u8 = 0x3D
_ASCII_GT:        u8 = 0x3E
_ASCII_QUESTION:  u8 = 0x3F
_ASCII_UPPER_A:   u8 = 0x41
_ASCII_UPPER_F:   u8 = 0x46
_ASCII_UPPER_Z:   u8 = 0x5A
_ASCII_LBRACKET:  u8 = 0x5B
_ASCII_RBRACKET:  u8 = 0x5D
_ASCII_UNDERSCORE: u8 = 0x5F
_ASCII_LOWER_A:   u8 = 0x61
_ASCII_LOWER_F:   u8 = 0x66
_ASCII_LOWER_X:   u8 = 0x78
_ASCII_LOWER_Z:   u8 = 0x7A
_ASCII_SLASH:     u8 = 0x2F

def _is_ascii_digit( c: u8 ) -> bool:
	return c >= _ASCII_ZERO and c <= _ASCII_NINE

def _is_hex_digit( c: u8 ) -> bool:
	return _is_ascii_digit( c ) or ( c >= _ASCII_LOWER_A and c <= _ASCII_LOWER_F ) or ( c >= _ASCII_UPPER_A and c <= _ASCII_UPPER_F )

def _hex_digit_value( c: u8 ) -> u32:
	if c >= _ASCII_ZERO and c <= _ASCII_NINE:
		with compiler.panic_arithmetic( 'a decimal digit byte minus the ascii zero offset cannot overflow u32' ):
			return u32( c - _ASCII_ZERO )
	if c >= _ASCII_LOWER_A and c <= _ASCII_LOWER_F:
		with compiler.panic_arithmetic( 'a lowercase hex digit byte minus the ascii lowercase-a offset, plus 10, cannot overflow u32' ):
			return u32( c - _ASCII_LOWER_A ) + 10
	with compiler.panic_arithmetic( 'an uppercase hex digit byte minus the ascii uppercase-A offset, plus 10, cannot overflow u32' ):
		return u32( c - _ASCII_UPPER_A ) + 10

def _is_name_start_byte( c: u8 ) -> bool:
	# permissive: any byte with the high bit set (part of a multi-byte UTF-8
	# sequence) is accepted as a name byte without validating it against the
	# real Unicode XML Name production - a deliberate v1 simplification.
	return ( c >= _ASCII_UPPER_A and c <= _ASCII_UPPER_Z ) or ( c >= _ASCII_LOWER_A and c <= _ASCII_LOWER_Z ) or c == _ASCII_UNDERSCORE or c >= 0x80

def _is_name_byte( c: u8 ) -> bool:
	return _is_name_start_byte( c ) or _is_ascii_digit( c ) or c == _ASCII_DASH or c == _ASCII_DOT or c == _ASCII_COLON


# ---------------------------------------------------------------------------
# 1. XMLNode / ElementData / XMLError
# ---------------------------------------------------------------------------

class ElementData:
	tag:             str                    # raw qualified name, e.g. "ns1:foo" - kept verbatim for round-tripping
	namespace_uri:   str                    # "" means no namespace in scope for this element
	attributes:      list[tuple[str,str,str]] # (qualified_name, namespace_uri, value); namespace_uri "" = no namespace
	children:        list[XMLNode]

	def __init__( self, tag: str, namespace_uri: str, attributes: list[tuple[str,str,str]], children: list[XMLNode] ) -> None:
		self.tag = tag
		self.namespace_uri = namespace_uri
		self.attributes = attributes
		self.children = children

	def local_name( self ) -> str:
		parts: tuple[str,str] = _split_qname( self.tag )
		return parts[1]

	def prefix( self ) -> str:
		parts: tuple[str,str] = _split_qname( self.tag )
		return parts[0]

	def get_attr( self, local_name: str, namespace_uri: str = '' ) -> Result[str, XMLError]:
		i: usize = 0
		n: usize = len( self.attributes )
		while i < n:
			entry: tuple[str,str,str] = self.attributes.__getitem__( i ).unwrap( 'get_attr: index in bounds by construction' )
			qparts: tuple[str,str] = _split_qname( entry[0] )
			if qparts[1] == local_name and entry[1] == namespace_uri:
				return Result.Ok( entry[2] )
			with compiler.wrap_arithmetic:
				i += 1
		return Result.Err( XMLError.AttributeNotFound( None ))


@union
class XMLNode:
	Element: ElementData
	Text:    str
	Comment: str
	CData:   str

	def as_element( self ) -> Result[ElementData, XMLError]:
		match self:
			case XMLNode.Element( e ):
				return Result.Ok( e )
			case _:
				return Result.Err( XMLError.WrongType( None ))

	def as_text( self ) -> Result[str, XMLError]:
		match self:
			case XMLNode.Text( t ):
				return Result.Ok( t )
			case _:
				return Result.Err( XMLError.WrongType( None ))

	def as_comment( self ) -> Result[str, XMLError]:
		match self:
			case XMLNode.Comment( c ):
				return Result.Ok( c )
			case _:
				return Result.Err( XMLError.WrongType( None ))

	def as_cdata( self ) -> Result[str, XMLError]:
		match self:
			case XMLNode.CData( cd ):
				return Result.Ok( cd )
			case _:
				return Result.Err( XMLError.WrongType( None ))

	def child_elements( self ) -> list[ElementData]:
		e: ElementData = self.as_element().unwrap( 'child_elements: called on a non-Element node' )
		out: list[ElementData] = list[ElementData]()
		i: usize = 0
		n: usize = len( e.children )
		while i < n:
			c: XMLNode = e.children.__getitem__( i ).unwrap( 'child_elements: index in bounds by construction' )
			match c:
				case XMLNode.Element( ed ):
					out.append( ed ).unwrap( 'child_elements: append failed' )
				case _:
					pass
			with compiler.wrap_arithmetic:
				i += 1
		return out

	def text_content( self ) -> str:
		# concatenates DIRECT Text-child values only - not recursive into
		# nested elements (a documented v1 limitation, cheap to extend).
		e: ElementData = self.as_element().unwrap( 'text_content: called on a non-Element node' )
		out: str = ''
		i: usize = 0
		n: usize = len( e.children )
		while i < n:
			c: XMLNode = e.children.__getitem__( i ).unwrap( 'text_content: index in bounds by construction' )
			match c:
				case XMLNode.Text( t ):
					out = out + t
				case _:
					pass
			with compiler.wrap_arithmetic:
				i += 1
		return out


@union
class XMLError:
	# accessor errors
	WrongType:          None
	AttributeNotFound:  None
	# parse errors - payload is the byte offset INTO THE XML TEXT
	UnexpectedEnd:       usize
	UnexpectedChar:      usize
	InvalidName:         usize
	DuplicateAttribute:  usize
	InvalidEntity:       usize
	InvalidComment:      usize
	UnboundPrefix:       usize
	MismatchedEndTag:    usize
	MultipleRootElements: usize
	NoRootElement:       None
	TrailingGarbage:     usize


def _split_qname( tag: str ) -> tuple[str,str]:
	# (prefix, local_name) - prefix is "" when tag has no ':'.
	found: isize = tag.find( ':' )
	if found == isize( -1 ):
		return ( '', tag )
	with compiler.panic_arithmetic( 'find() never returns a negative offset once the -1/not-found case is excluded' ):
		i: usize = usize( found )
	with compiler.wrap_arithmetic:
		after: usize = i + 1
	return ( tag[0:i], tag[after:] )


def _lookup_scope( scope: list[tuple[str,str]], prefix: str ) -> Result[str, None]:
	# walks from the END of the list - the most-recently-appended (i.e.
	# innermost/closest-declared) binding for `prefix` wins.
	n: usize = len( scope )
	i: usize = n
	while i > 0:
		with compiler.wrap_arithmetic:
			i -= 1
		entry: tuple[str,str] = scope.__getitem__( i ).unwrap( '_lookup_scope: index in bounds by construction' )
		if entry[0] == prefix:
			return Result.Ok( entry[1] )
	return Result.Err( None )


# ---------------------------------------------------------------------------
# 2. parse() - recursive-descent parser
#
# Each _parse_X takes the input string plus a byte cursor and returns
# Result[tuple[X,usize], XMLError] - the parsed value/text paired with the
# cursor position just past what it consumed, same tuple[T,usize] idiom
# lib/json.py uses.
# ---------------------------------------------------------------------------

def parse( s: str ) -> Result[XMLNode, XMLError]:
	length: usize = s.byte_len()
	pos: usize = _skip_misc( s, 0, True ).or_return()
	if pos >= length:
		return Result.Err( XMLError.NoRootElement( None ))
	data: ConstPtr[u8] = s.get_cstr()
	if data[pos] != _ASCII_LT or s.startswith( '</', pos ):
		return Result.Err( XMLError.UnexpectedChar( pos ))
	empty_scope: list[tuple[str,str]] = list[tuple[str,str]]()
	root_parsed = _parse_element( s, pos, empty_scope ).or_return()
	root: XMLNode = root_parsed[0]
	end_pos: usize = root_parsed[1]
	end_pos = _skip_misc( s, end_pos, False ).or_return()
	if end_pos != length:
		data2: ConstPtr[u8] = s.get_cstr()
		if data2[end_pos] == _ASCII_LT and not s.startswith( '</', end_pos ):
			return Result.Err( XMLError.MultipleRootElements( end_pos ))
		return Result.Err( XMLError.TrailingGarbage( end_pos ))
	return Result.Ok( root )


def _skip_ws( s: str, pos: usize ) -> usize:
	data: ConstPtr[u8] = s.get_cstr()
	length: usize = s.byte_len()
	i: usize = pos
	while i < length:
		ch: u8 = data[i]
		if ch != _ASCII_SPACE and ch != _ASCII_TAB and ch != _ASCII_LF and ch != _ASCII_CR:
			return i
		with compiler.wrap_arithmetic:
			i += 1
	return i


def _skip_misc( s: str, pos: usize, allow_doctype: bool ) -> Result[usize, XMLError]:
	# skips whitespace, "<?...?>" (XML declaration or any other processing
	# instruction), "<!--...-->" comments, and - only when allow_doctype -
	# a single "<!DOCTYPE ...>". Stops at the first byte that is none of
	# these (expected to be the root element's '<', or end of input).
	p: usize = pos
	while True:
		p = _skip_ws( s, p )
		length: usize = s.byte_len()
		if p >= length:
			return Result.Ok( p )
		if s.startswith( '<?', p ):
			found: isize = s.find( '?>', p )
			if found == isize( -1 ):
				return Result.Err( XMLError.UnexpectedEnd( p ))
			with compiler.panic_arithmetic( 'find() never returns a negative offset once the -1/not-found case is excluded' ):
				e: usize = usize( found )
			with compiler.wrap_arithmetic:
				p = e + 2
		elif s.startswith( '<!--', p ):
			comment_parsed = _parse_comment( s, p ).or_return()
			p = comment_parsed[1]
		elif allow_doctype and s.startswith( '<!DOCTYPE', p ):
			p = _skip_doctype( s, p ).or_return()
		else:
			return Result.Ok( p )


def _skip_doctype( s: str, pos: usize ) -> Result[usize, XMLError]:
	# pos points at the '<' of "<!DOCTYPE". Scans to the matching top-level
	# '>' (tracking '['/']' depth for the internal subset, and skipping
	# quoted spans so a '>' inside a PUBLIC/SYSTEM literal doesn't
	# prematurely terminate) WITHOUT validating the internal subset itself.
	data: ConstPtr[u8] = s.get_cstr()
	length: usize = s.byte_len()
	with compiler.wrap_arithmetic:
		i: usize = pos + 9 # len( '<!DOCTYPE' )
	depth: usize = 0
	while i < length:
		ch: u8 = data[i]
		if ch == _ASCII_DQUOTE or ch == _ASCII_SQUOTE:
			q: u8 = ch
			with compiler.wrap_arithmetic:
				i += 1
			while i < length and data[i] != q:
				with compiler.wrap_arithmetic:
					i += 1
			if i >= length:
				return Result.Err( XMLError.UnexpectedEnd( i ))
			with compiler.wrap_arithmetic:
				i += 1
		elif ch == _ASCII_LBRACKET:
			with compiler.wrap_arithmetic:
				depth += 1
				i += 1
		elif ch == _ASCII_RBRACKET:
			if depth > 0:
				with compiler.wrap_arithmetic:
					depth -= 1
			with compiler.wrap_arithmetic:
				i += 1
		elif ch == _ASCII_GT and depth == 0:
			with compiler.wrap_arithmetic:
				return Result.Ok( i + 1 )
		else:
			with compiler.wrap_arithmetic:
				i += 1
	return Result.Err( XMLError.UnexpectedEnd( i ))


def _parse_comment( s: str, pos: usize ) -> Result[tuple[str,usize], XMLError]:
	# pos points at the '<' of "<!--".
	with compiler.wrap_arithmetic:
		content_start: usize = pos + 4
	found: isize = s.find( '-->', content_start )
	if found == isize( -1 ):
		return Result.Err( XMLError.UnexpectedEnd( pos ))
	with compiler.panic_arithmetic( 'find() never returns a negative offset once the -1/not-found case is excluded' ):
		end: usize = usize( found )
	content: str = s[content_start:end]
	if content.find( '--' ) != isize( -1 ):
		return Result.Err( XMLError.InvalidComment( pos ))
	with compiler.wrap_arithmetic:
		return Result.Ok(( content, end + 3 ))


def _parse_cdata( s: str, pos: usize ) -> Result[tuple[str,usize], XMLError]:
	# pos points at the '<' of "<![CDATA[".
	with compiler.wrap_arithmetic:
		content_start: usize = pos + 9
	found: isize = s.find( ']]>', content_start )
	if found == isize( -1 ):
		return Result.Err( XMLError.UnexpectedEnd( pos ))
	with compiler.panic_arithmetic( 'find() never returns a negative offset once the -1/not-found case is excluded' ):
		end: usize = usize( found )
	content: str = s[content_start:end]
	with compiler.wrap_arithmetic:
		return Result.Ok(( content, end + 3 ))


def _parse_name( s: str, pos: usize ) -> Result[tuple[str,usize], XMLError]:
	data: ConstPtr[u8] = s.get_cstr()
	length: usize = s.byte_len()
	if pos >= length or not _is_name_start_byte( data[pos] ):
		return Result.Err( XMLError.InvalidName( pos ))
	with compiler.wrap_arithmetic:
		i: usize = pos + 1
	while i < length and _is_name_byte( data[i] ):
		with compiler.wrap_arithmetic:
			i += 1
	return Result.Ok(( s[pos:i], i ))


# --- entity/char-ref decoding + attribute-value/text-run scanning:
# two-pass size-then-fill, mirroring lib/json.py's _measure_string/
# _fill_string. A single pair of functions serves BOTH text content (stop
# byte '<') and attribute values (stop byte the opening quote char) -
# whichever stop byte is passed, a literal '<' encountered before it is
# always rejected (a no-op for the content case, since '<' IS the stop
# byte there and so is never seen as "still inside" the run). ------------

def _decode_entity_at( s: str, pos: usize ) -> Result[tuple[u32,usize], XMLError]:
	# pos points at the '&'.
	data: ConstPtr[u8] = s.get_cstr()
	length: usize = s.byte_len()
	with compiler.wrap_arithmetic:
		i: usize = pos + 1
	if i >= length:
		return Result.Err( XMLError.UnexpectedEnd( i ))
	if data[i] == _ASCII_HASH:
		with compiler.wrap_arithmetic:
			i += 1
		if i >= length:
			return Result.Err( XMLError.UnexpectedEnd( i ))
		is_hex: bool = False
		if data[i] == _ASCII_LOWER_X:
			is_hex = True
			with compiler.wrap_arithmetic:
				i += 1
		digit_start: usize = i
		value: u32 = 0
		digit_count: usize = 0
		while i < length and (( is_hex and _is_hex_digit( data[i] )) or ( not is_hex and _is_ascii_digit( data[i] ))):
			if digit_count >= 8:
				return Result.Err( XMLError.InvalidEntity( pos ))
			if is_hex:
				with compiler.panic_arithmetic( 'accumulating at most 8 hex digits (capped by the guard above) into value cannot overflow u32' ):
					value = ( value << 4 ) | _hex_digit_value( data[i] )
			else:
				with compiler.panic_arithmetic( 'accumulating at most 8 decimal digits (capped by the guard above) into value cannot overflow u32' ):
					value = value * 10 + u32( data[i] - _ASCII_ZERO )
			with compiler.wrap_arithmetic:
				digit_count += 1
				i += 1
		if i == digit_start:
			return Result.Err( XMLError.InvalidEntity( pos ))
		if value >= 0xD800 and value <= 0xDFFF:
			return Result.Err( XMLError.InvalidEntity( pos ))
		if value > 0x10FFFF:
			return Result.Err( XMLError.InvalidEntity( pos ))
		if i >= length or data[i] != _ASCII_SEMICOLON:
			return Result.Err( XMLError.InvalidEntity( pos ))
		with compiler.wrap_arithmetic:
			return Result.Ok(( value, i + 1 ))
	name_start: usize = i
	while i < length and data[i] != _ASCII_SEMICOLON:
		with compiler.wrap_arithmetic:
			i += 1
	if i >= length:
		return Result.Err( XMLError.UnexpectedEnd( i ))
	name: str = s[name_start:i]
	with compiler.wrap_arithmetic:
		end: usize = i + 1
	if name == 'amp':
		return Result.Ok(( 0x26, end ))
	if name == 'lt':
		return Result.Ok(( 0x3C, end ))
	if name == 'gt':
		return Result.Ok(( 0x3E, end ))
	if name == 'quot':
		return Result.Ok(( 0x22, end ))
	if name == 'apos':
		return Result.Ok(( 0x27, end ))
	return Result.Err( XMLError.InvalidEntity( pos ))


def _measure_decoded( s: str, start: usize, stop: u8 ) -> Result[tuple[usize,usize], XMLError]:
	data: ConstPtr[u8] = s.get_cstr()
	length: usize = s.byte_len()
	i: usize = start
	total: usize = 0
	while i < length:
		ch: u8 = data[i]
		if ch == stop:
			return Result.Ok(( total, i ))
		if ch == _ASCII_LT:
			return Result.Err( XMLError.UnexpectedChar( i ))
		if ch == _ASCII_AMP:
			decoded = _decode_entity_at( s, i ).or_return()
			with compiler.panic_arithmetic( 'a decoded codepoints own utf8 width added to a running usize total cannot overflow' ):
				total += builtins.utf8_encoded_len( decoded[0] )
			i = decoded[1]
		else:
			with compiler.panic_arithmetic( 'a single-byte passthrough increment cannot overflow usize' ):
				total += 1
			with compiler.wrap_arithmetic:
				i += 1
	return Result.Err( XMLError.UnexpectedEnd( i ))


def _fill_decoded( s: str, start: usize, stop: u8, buf: Ptr[u8] ) -> None:
	data: ConstPtr[u8] = s.get_cstr()
	length: usize = s.byte_len()
	i: usize = start
	out: usize = 0
	while i < length:
		ch: u8 = data[i]
		if ch == stop:
			return
		if ch == _ASCII_AMP:
			decoded = _decode_entity_at( s, i ).unwrap( '_fill_decoded: re-decoding an entity already validated by _measure_decoded cannot fail' )
			with compiler.panic_arithmetic( 'a decoded codepoints own utf8-encode increment cannot overflow usize' ):
				out += builtins.encode_utf8_at( buf, out, decoded[0] )
			i = decoded[1]
		else:
			buf[out] = ch
			with compiler.panic_arithmetic( 'a single-byte passthrough increment cannot overflow usize' ):
				out += 1
				i += 1


def _parse_decoded_run( s: str, start: usize, stop: u8 ) -> Result[tuple[str,usize], XMLError]:
	measured = _measure_decoded( s, start, stop ).or_return()
	new_size: usize = measured[0]
	end_pos: usize = measured[1]
	with compiler.panic_arithmetic( 'a measured decoded-run size plus one zero terminator cannot overflow usize' ):
		buf_size: usize = new_size + 1
	buf: Ptr[u8] = sys.alloc[u8]( buf_size )
	_fill_decoded( s, start, stop, buf )
	buf[new_size] = 0
	built: str = str.from_cstr( compiler.cast( ConstPtr[u8], buf ), buf_size ).unwrap( 'xml._parse_decoded_run: internal buffer was not valid UTF-8 (unreachable - only input UTF-8 bytes and computed entity codepoints are ever written)' )
	sys.free( buf )
	return Result.Ok(( built, end_pos ))


def _parse_element( s: str, pos: usize, scope: list[tuple[str,str]] ) -> Result[tuple[XMLNode,usize], XMLError]:
	# pos points at the '<'.
	with compiler.wrap_arithmetic:
		name_start: usize = pos + 1
	tag_parsed = _parse_name( s, name_start ).or_return()
	tag: str = tag_parsed[0]
	p: usize = tag_parsed[1]

	raw_attrs: list[tuple[str,str,usize]] = list[tuple[str,str,usize]]() # (qualified_name, value, name_pos)
	length: usize = s.byte_len()
	data: ConstPtr[u8] = s.get_cstr()
	ch: u8 = 0
	while True:
		p = _skip_ws( s, p )
		if p >= length:
			return Result.Err( XMLError.UnexpectedEnd( p ))
		ch = data[p]
		if ch == _ASCII_SLASH or ch == _ASCII_GT:
			break
		attr_name_pos: usize = p
		aname_parsed = _parse_name( s, p ).or_return()
		aname: str = aname_parsed[0]
		p = aname_parsed[1]
		p = _skip_ws( s, p )
		if p >= length or data[p] != _ASCII_EQUALS:
			return Result.Err( XMLError.UnexpectedChar( p ))
		with compiler.wrap_arithmetic:
			p += 1
		p = _skip_ws( s, p )
		if p >= length:
			return Result.Err( XMLError.UnexpectedEnd( p ))
		quote: u8 = data[p]
		if quote != _ASCII_DQUOTE and quote != _ASCII_SQUOTE:
			return Result.Err( XMLError.UnexpectedChar( p ))
		with compiler.wrap_arithmetic:
			value_start: usize = p + 1
		value_parsed = _parse_decoded_run( s, value_start, quote ).or_return()
		avalue: str = value_parsed[0]
		with compiler.wrap_arithmetic:
			p = value_parsed[1] + 1 # past the closing quote

		dup: bool = False
		j: usize = 0
		rn: usize = len( raw_attrs )
		while j < rn:
			existing: tuple[str,str,usize] = raw_attrs.__getitem__( j ).unwrap( '_parse_element: index in bounds by construction' )
			if existing[0] == aname:
				dup = True
			with compiler.wrap_arithmetic:
				j += 1
		if dup:
			return Result.Err( XMLError.DuplicateAttribute( attr_name_pos ))
		raw_attrs.append(( aname, avalue, attr_name_pos )).unwrap( '_parse_element: raw_attrs append failed' )

	self_closing: bool = False
	if ch == _ASCII_SLASH:
		with compiler.wrap_arithmetic:
			p += 1
		if p >= length or data[p] != _ASCII_GT:
			return Result.Err( XMLError.UnexpectedChar( p ))
		self_closing = True
	with compiler.wrap_arithmetic:
		p += 1 # past '>'

	# --- build the child namespace scope from any xmlns/xmlns:* attrs ---
	new_bindings: list[tuple[str,str]] = list[tuple[str,str]]()
	total_attrs: usize = len( raw_attrs )
	k: usize = 0
	while k < total_attrs:
		entry: tuple[str,str,usize] = raw_attrs.__getitem__( k ).unwrap( '_parse_element: index in bounds by construction' )
		aname2: str = entry[0]
		avalue2: str = entry[1]
		if aname2 == 'xmlns':
			new_bindings.append(( '', avalue2 )).unwrap( '_parse_element: new_bindings append failed' )
		elif aname2.startswith( 'xmlns:' ):
			new_bindings.append(( aname2[6:], avalue2 )).unwrap( '_parse_element: new_bindings append failed' )
		with compiler.wrap_arithmetic:
			k += 1

	child_scope: list[tuple[str,str]] = scope
	if len( new_bindings ) > 0:
		combined: list[tuple[str,str]] = list[tuple[str,str]]()
		m: usize = 0
		sn: usize = len( scope )
		while m < sn:
			combined.append( scope.__getitem__( m ).unwrap( '_parse_element: index in bounds by construction' )).unwrap( '_parse_element: combined append failed' )
			with compiler.wrap_arithmetic:
				m += 1
		m2: usize = 0
		nn: usize = len( new_bindings )
		while m2 < nn:
			combined.append( new_bindings.__getitem__( m2 ).unwrap( '_parse_element: index in bounds by construction' )).unwrap( '_parse_element: combined append failed' )
			with compiler.wrap_arithmetic:
				m2 += 1
		child_scope = combined

	# --- resolve the element's own namespace ---
	qparts: tuple[str,str] = _split_qname( tag )
	elem_prefix: str = qparts[0]
	elem_uri: str = ''
	if elem_prefix == '':
		match _lookup_scope( child_scope, '' ):
			case Result.Ok( u ):
				elem_uri = u
			case Result.Err( _ ):
				pass
	else:
		match _lookup_scope( child_scope, elem_prefix ):
			case Result.Ok( u2 ):
				elem_uri = u2
			case Result.Err( _ ):
				return Result.Err( XMLError.UnboundPrefix( pos ))

	# --- resolve each attribute's namespace (unprefixed attrs never
	# inherit the default namespace, per the XML namespaces spec) ---
	resolved_attrs: list[tuple[str,str,str]] = list[tuple[str,str,str]]()
	k2: usize = 0
	while k2 < total_attrs:
		entry2: tuple[str,str,usize] = raw_attrs.__getitem__( k2 ).unwrap( '_parse_element: index in bounds by construction' )
		aname3: str = entry2[0]
		avalue3: str = entry2[1]
		apos3: usize = entry2[2]
		auri: str = ''
		if aname3 != 'xmlns' and not aname3.startswith( 'xmlns:' ):
			aqparts: tuple[str,str] = _split_qname( aname3 )
			apfx: str = aqparts[0]
			if apfx != '':
				match _lookup_scope( child_scope, apfx ):
					case Result.Ok( u3 ):
						auri = u3
					case Result.Err( _ ):
						return Result.Err( XMLError.UnboundPrefix( apos3 ))
		resolved_attrs.append(( aname3, auri, avalue3 )).unwrap( '_parse_element: resolved_attrs append failed' )
		with compiler.wrap_arithmetic:
			k2 += 1

	children: list[XMLNode] = list[XMLNode]()
	if not self_closing:
		children_parsed = _parse_children( s, p, tag, child_scope ).or_return()
		children = children_parsed[0]
		p = children_parsed[1]

	elem_data: ElementData = ElementData( tag, elem_uri, resolved_attrs, children )
	return Result.Ok(( XMLNode.Element( elem_data ), p ))


def _parse_children( s: str, pos: usize, expected_tag: str, scope: list[tuple[str,str]] ) -> Result[tuple[list[XMLNode],usize], XMLError]:
	data: ConstPtr[u8] = s.get_cstr()
	length: usize = s.byte_len()
	nodes: list[XMLNode] = list[XMLNode]()
	p: usize = pos
	while True:
		if p >= length:
			return Result.Err( XMLError.UnexpectedEnd( p ))
		ch: u8 = data[p]
		if ch == _ASCII_LT:
			if s.startswith( '</', p ):
				with compiler.wrap_arithmetic:
					end_name_start: usize = p + 2
				end_name_parsed = _parse_name( s, end_name_start ).or_return()
				end_name: str = end_name_parsed[0]
				p2: usize = end_name_parsed[1]
				p2 = _skip_ws( s, p2 )
				if p2 >= length or data[p2] != _ASCII_GT:
					return Result.Err( XMLError.UnexpectedChar( p2 ))
				if end_name != expected_tag:
					return Result.Err( XMLError.MismatchedEndTag( end_name_start ))
				with compiler.wrap_arithmetic:
					return Result.Ok(( nodes, p2 + 1 ))
			elif s.startswith( '<!--', p ):
				comment_parsed = _parse_comment( s, p ).or_return()
				nodes.append( XMLNode.Comment( comment_parsed[0] )).unwrap( '_parse_children: append failed' )
				p = comment_parsed[1]
			elif s.startswith( '<![CDATA[', p ):
				cdata_parsed = _parse_cdata( s, p ).or_return()
				nodes.append( XMLNode.CData( cdata_parsed[0] )).unwrap( '_parse_children: append failed' )
				p = cdata_parsed[1]
			elif s.startswith( '<?', p ):
				found2: isize = s.find( '?>', p )
				if found2 == isize( -1 ):
					return Result.Err( XMLError.UnexpectedEnd( p ))
				with compiler.panic_arithmetic( 'find() never returns a negative offset once the -1/not-found case is excluded' ):
					e2: usize = usize( found2 )
				with compiler.wrap_arithmetic:
					p = e2 + 2
			else:
				child_parsed = _parse_element( s, p, scope ).or_return()
				nodes.append( child_parsed[0] ).unwrap( '_parse_children: append failed' )
				p = child_parsed[1]
		else:
			text_parsed = _parse_decoded_run( s, p, _ASCII_LT ).or_return()
			nodes.append( XMLNode.Text( text_parsed[0] )).unwrap( '_parse_children: append failed' )
			p = text_parsed[1]


# ---------------------------------------------------------------------------
# 3. dumps() - serializer. Builds output as a list[str] of pieces + join,
# same idiom as lib/json.py's dumps() (no growable StringBuilder exists in
# this language). Infallible - unlike json.py's dumps(), there is no failure
# mode here (no NaN/Inf-shaped values in this data model), so dumps()
# returns a plain str, not a Result.
# ---------------------------------------------------------------------------

def dumps( node: XMLNode ) -> str:
	return _dump_node( node )


def _dump_node( node: XMLNode ) -> str:
	match node:
		case XMLNode.Text( t ):
			return _escape_text( t )
		case XMLNode.Comment( c ):
			return '<!--' + c + '-->'
		case XMLNode.CData( cd ):
			return '<![CDATA[' + cd + ']]>'
		case XMLNode.Element( e ):
			pieces: list[str] = list[str]()
			pieces.append( '<' + e.tag ).unwrap( '_dump_node: append failed' )
			i: usize = 0
			an: usize = len( e.attributes )
			while i < an:
				attr: tuple[str,str,str] = e.attributes.__getitem__( i ).unwrap( '_dump_node: index in bounds by construction' )
				pieces.append( ' ' + attr[0] + '="' + _escape_attr( attr[2] ) + '"' ).unwrap( '_dump_node: append failed' )
				with compiler.wrap_arithmetic:
					i += 1
			cn: usize = len( e.children )
			if cn == 0:
				pieces.append( '/>' ).unwrap( '_dump_node: append failed' )
				return ''.join( pieces )
			pieces.append( '>' ).unwrap( '_dump_node: append failed' )
			j: usize = 0
			while j < cn:
				child: XMLNode = e.children.__getitem__( j ).unwrap( '_dump_node: index in bounds by construction' )
				pieces.append( _dump_node( child )).unwrap( '_dump_node: append failed' )
				with compiler.wrap_arithmetic:
					j += 1
			pieces.append( '</' + e.tag + '>' ).unwrap( '_dump_node: append failed' )
			return ''.join( pieces )


def _escape_text( s: str ) -> str:
	return _escape_bytes( s, False )


def _escape_attr( s: str ) -> str:
	return _escape_bytes( s, True )


def _escape_bytes( s: str, quote_too: bool ) -> str:
	data: ConstPtr[u8] = s.get_cstr()
	length: usize = s.byte_len()
	new_size: usize = _measure_escaped( data, length, quote_too )
	with compiler.panic_arithmetic( 'a measured escape size plus one zero terminator cannot overflow usize' ):
		buf_size: usize = new_size + 1
	buf: Ptr[u8] = sys.alloc[u8]( buf_size )
	_fill_escaped( data, length, quote_too, buf )
	buf[new_size] = 0
	built: str = str.from_cstr( compiler.cast( ConstPtr[u8], buf ), buf_size ).unwrap( 'xml._escape_bytes: internal buffer was not valid UTF-8 (unreachable - only input UTF-8 bytes and fixed ASCII escape sequences are ever written)' )
	sys.free( buf )
	return built


def _escaped_width( ch: u8, quote_too: bool ) -> usize:
	if ch == _ASCII_AMP:
		return 5 # &amp;
	if ch == _ASCII_LT:
		return 4 # &lt;
	if ch == _ASCII_GT:
		return 4 # &gt;
	if quote_too and ch == _ASCII_DQUOTE:
		return 6 # &quot;
	return 1


def _measure_escaped( data: ConstPtr[u8], length: usize, quote_too: bool ) -> usize:
	total: usize = 0
	i: usize = 0
	while i < length:
		with compiler.panic_arithmetic( 'a bounded per-byte escape width added to a running usize total cannot overflow' ):
			total += _escaped_width( data[i], quote_too )
		with compiler.wrap_arithmetic:
			i += 1
	return total


def _write_literal( buf: Ptr[u8], out: usize, lit: str ) -> usize:
	ldata: ConstPtr[u8] = lit.get_cstr()
	llen: usize = lit.byte_len()
	j: usize = 0
	while j < llen:
		with compiler.wrap_arithmetic:
			buf[out + j] = ldata[j]
			j += 1
	with compiler.panic_arithmetic( 'advancing by a literal escape sequences own fixed byte width cannot overflow usize' ):
		return out + llen


def _fill_escaped( data: ConstPtr[u8], length: usize, quote_too: bool, buf: Ptr[u8] ) -> None:
	i: usize = 0
	out: usize = 0
	while i < length:
		ch: u8 = data[i]
		if ch == _ASCII_AMP:
			out = _write_literal( buf, out, '&amp;' )
		elif ch == _ASCII_LT:
			out = _write_literal( buf, out, '&lt;' )
		elif ch == _ASCII_GT:
			out = _write_literal( buf, out, '&gt;' )
		elif quote_too and ch == _ASCII_DQUOTE:
			out = _write_literal( buf, out, '&quot;' )
		else:
			buf[out] = ch
			with compiler.panic_arithmetic( 'a single-byte passthrough increment cannot overflow usize' ):
				out += 1
		with compiler.wrap_arithmetic:
			i += 1
