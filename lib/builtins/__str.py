# __str.py: str's own UTF-8 codec + OS-specific case-mapping backends,
# pulled out of the str class itself (lib/builtins/__init__.py) - none of
# this is actually str-specific. Every function here works on a raw
# (data: ConstPtr[u8], byte_len: usize) view of UTF-8 bytes and, where it
# allocates, hands back a freshly owned buffer (an out-param Ptr[usize]
# receives its size, the same out-param convention _decode_utf8_at already
# used before the move) - str.upper()/lower()/chr()/ord() (see __init__.py)
# do the one thing that DOES need to live on str itself (wrapping a
# returned buffer back into a real str via the private, __allocate__-based
# _from_owned_cstr, or reading str's own private fields through its public
# accessors) and nothing else needs to be a method at all. Same raw/type-
# specific split __RawDict.py/__list.py already use, just for str.

import compiler
import sys

def decode_utf8_at( data: ConstPtr[u8], i: usize, consumed: Ptr[usize] ) -> u32:
	''' decodes one codepoint starting at data[i], writing the number of
	bytes consumed (1-4) to *consumed - used by ord() and by the POSIX
	case_map's own decode pass below (towupper_l/towlower_l take one
	codepoint at a time, unlike Windows' whole-buffer LCMapStringEx).
	Assumes well-formed UTF-8 (str's own construction-time invariant - see
	__init__.py's str._from_owned_cstr, the one place that's actually
	checked) - not re-validated here, same trust boundary str.byte_len()/
	__len__ already rely on. '''
	byte1: u8 = data[i]
	if ( byte1 & 0x80 ) == 0x00:
		consumed[0] = 1
		with compiler.wrap_arithmetic:
			return u32( byte1 )
	if ( byte1 & 0xE0 ) == 0xC0:
		consumed[0] = 2
		with compiler.wrap_arithmetic:
			return ( u32( byte1 & 0x1F ) << 6 ) | u32( data[i+1] & 0x3F )
	if ( byte1 & 0xF0 ) == 0xE0:
		consumed[0] = 3
		with compiler.wrap_arithmetic:
			return ( u32( byte1 & 0x0F ) << 12 ) | ( u32( data[i+1] & 0x3F ) << 6 ) | u32( data[i+2] & 0x3F )
	# 4-byte sequence - the only shape left once 1/2/3-byte are ruled out
	consumed[0] = 4
	with compiler.wrap_arithmetic:
		return ( u32( byte1 & 0x07 ) << 18 ) | ( u32( data[i+1] & 0x3F ) << 12 ) | ( u32( data[i+2] & 0x3F ) << 6 ) | u32( data[i+3] & 0x3F )

def utf8_encoded_len( cp: u32 ) -> usize:
	''' how many UTF-8 bytes `cp` would take - the "size" half of the
	size-then-fill two-pass convention str.concat/the POSIX case_map
	(below) use, needed because towupper_l/towlower_l can move a codepoint
	across a UTF-8 length boundary (e.g. U+00FF -> U+0178 is 2 bytes -> 2
	bytes, but plenty of other codepoints cross a boundary), so an output
	buffer can't just reuse the input's own byte size the way an
	ASCII-only implementation could. '''
	if cp < 0x80:
		return 1
	if cp < 0x800:
		return 2
	if cp < 0x10000:
		return 3
	return 4

def encode_utf8_at( dest: Ptr[u8], i: usize, cp: u32 ) -> usize:
	''' encodes `cp` as UTF-8 into dest starting at dest[i], returns the
	number of bytes written (1-4) - the "fill" half, paired with
	utf8_encoded_len just above. '''
	if cp < 0x80:
		with compiler.wrap_arithmetic:
			dest[i] = u8( cp )
		return 1
	if cp < 0x800:
		with compiler.wrap_arithmetic:
			dest[i]   = u8( 0xC0 | ( cp >> 6 ))
			dest[i+1] = u8( 0x80 | ( cp & 0x3F ))
		return 2
	if cp < 0x10000:
		with compiler.wrap_arithmetic:
			dest[i]   = u8( 0xE0 | ( cp >> 12 ))
			dest[i+1] = u8( 0x80 | (( cp >> 6 ) & 0x3F ))
			dest[i+2] = u8( 0x80 | ( cp & 0x3F ))
		return 3
	with compiler.wrap_arithmetic:
		dest[i]   = u8( 0xF0 | ( cp >> 18 ))
		dest[i+1] = u8( 0x80 | (( cp >> 12 ) & 0x3F ))
		dest[i+2] = u8( 0x80 | (( cp >> 6 ) & 0x3F ))
		dest[i+3] = u8( 0x80 | ( cp & 0x3F ))
	return 4

@compiler.target( os = 'windows' )
def case_map( data: ConstPtr[u8], byte_len: usize, is_upper: bool, out_size: Ptr[usize] ) -> Ptr[u8]:
	''' shared by str.upper()/lower() on Windows - same name/signature as
	the POSIX case_map below (see its own comment) so str's own
	_upper_os_native/_lower_os_native (lib/builtins/__init__.py) can call
	ONE name without needing to be @compiler.target-gated themselves;
	whichever of the two case_map definitions matches the current build
	target is the one that actually gets compiled/imported, same
	resolution lib/sys.py's own OS-gated cstrlen/memcpy/... pairs already
	rely on. Converts UTF-8 -> UTF-16 (MultiByteToWideChar),
	maps case on the whole UTF-16 buffer at once (LCMapStringEx), then
	converts back (WideCharToMultiByte). Every step follows the standard
	Win32 "call once with a null buffer to get the required size, allocate,
	call again to fill it" idiom - case mapping CAN change the byte length
	in general (surrogate-pair-widening codepoints, etc), even though it
	doesn't for any of the cases this implementation actually improves on
	over the old ASCII-only one.

	Confirmed by an actual compiled-and-run test, not just Win32 docs:
	LCMapStringEx (even with LCMAP_LINGUISTIC_CASING) only does SIMPLE
	(one-codepoint-in, one-codepoint-out) Unicode case mapping, the same
	ceiling the POSIX case_map's towupper_l/towlower_l have - 'ß' stays 'ß'
	(not 'SS'), and Greek final sigma isn't applied ('ΣΊΣΥΦΟΣ'.lower() ends
	in plain 'σ', not the contextually-correct 'ς'). Despite what
	PLAN_STR_UPPER_LOWER.md's own research suggested, there's no evidence
	LCMAP_LINGUISTIC_CASING implements Unicode's SpecialCasing.txt
	one-to-many/context-sensitive rules at all - only ICU reliably does,
	which this project deliberately isn't linking against (see crt.py's
	own comment on why). What this DOES still improve on over the old
	byte-range-only implementation: full simple-mapping coverage across
	every script Windows' own NLS Unicode data covers, not just ASCII
	'a'-'z'/'A'-'Z'. The empty locale name (a bare zero u16, not NULL/
	LOCALE_NAME_USER_DEFAULT) requests locale-INVARIANT behavior -
	matching real Python's own str.upper()/lower(), which never depend on
	the process locale (no Turkish dotless-i surprises). '''
	from windows.kernel32 import MultiByteToWideChar, WideCharToMultiByte, LCMapStringEx, CP_UTF8, LCMAP_LINGUISTIC_CASING, LCMAP_UPPERCASE, LCMAP_LOWERCASE
	flags: u32 = LCMAP_UPPERCASE if is_upper else LCMAP_LOWERCASE
	if byte_len == 0:
		empty_buf: Ptr[u8] = sys.alloc[u8]( 1 )
		empty_buf[0] = 0
		out_size[0] = 1
		return empty_buf
	with compiler.panic_arithmetic( 'string too long for a Win32 API call' ):
		src_len: i32 = i32( byte_len )

	wide_len: i32 = MultiByteToWideChar( CP_UTF8, 0, data, src_len, None, 0 )
	if wide_len <= 0:
		sys.panic( 'MultiByteToWideChar failed' )
	with compiler.panic_arithmetic( 'string too long for a Win32 API call' ):
		wide_buf: Ptr[u16] = sys.alloc[u16]( usize( wide_len ))
	defer( sys.free( wide_buf ))
	MultiByteToWideChar( CP_UTF8, 0, data, src_len, wide_buf, wide_len )

	empty_locale: u16 = 0
	map_flags: u32 = flags | LCMAP_LINGUISTIC_CASING
	mapped_len: i32 = LCMapStringEx( compiler.addrof( empty_locale ), map_flags, wide_buf, wide_len, None, 0, None, None, 0 )
	if mapped_len <= 0:
		sys.panic( 'LCMapStringEx failed' )
	with compiler.panic_arithmetic( 'string too long for a Win32 API call' ):
		mapped_buf: Ptr[u16] = sys.alloc[u16]( usize( mapped_len ))
	defer( sys.free( mapped_buf ))
	LCMapStringEx( compiler.addrof( empty_locale ), map_flags, wide_buf, wide_len, mapped_buf, mapped_len, None, None, 0 )

	out_len: i32 = WideCharToMultiByte( CP_UTF8, 0, mapped_buf, mapped_len, None, 0, None, None )
	if out_len <= 0:
		sys.panic( 'WideCharToMultiByte failed' )
	with compiler.panic_arithmetic( 'string too long for a Win32 API call' ):
		out_count: usize = usize( out_len )
		out_buf_size: usize = out_count + 1
	out_buf: Ptr[u8] = sys.alloc[u8]( out_buf_size )
	WideCharToMultiByte( CP_UTF8, 0, mapped_buf, mapped_len, out_buf, out_len, None, None )
	out_buf[out_count] = 0
	out_size[0] = out_buf_size
	return out_buf

@compiler.target( os = not 'windows' )
def _case_codepoint_posix( cp: u32, is_upper: bool, loc: Ptr[None] ) -> u32:
	''' cases one codepoint via towupper_l/towlower_l, or passes it through
	unchanged if loc is None (newlocale('C.UTF-8') failed - see case_map's
	own comment on why that's a graceful fallback, not a hard error). '''
	if loc is None:
		return cp
	from crt import towupper_l, towlower_l
	with compiler.panic_arithmetic( 'codepoint out of range for wint_t - impossible for valid Unicode (max U+10FFFF)' ):
		wc: i32 = i32( cp )
	cased: i32
	if is_upper:
		cased = towupper_l( wc, loc )
	else:
		cased = towlower_l( wc, loc )
	with compiler.panic_arithmetic( 'towupper_l/towlower_l returned a negative/out-of-range codepoint' ):
		return u32( cased )

@compiler.target( os = not 'windows' )
def case_map( data: ConstPtr[u8], byte_len: usize, is_upper: bool, out_size: Ptr[usize] ) -> Ptr[u8]:
	''' shared by str.upper()/lower() on Linux/macOS/BSD - same name/
	signature as the Windows case_map above (see its own comment on why:
	str's own _upper_os_native/_lower_os_native call ONE name, not gated
	themselves). towupper_l/towlower_l only, no ICU (see
	PLAN_STR_UPPER_LOWER.md and crt.py's own comment on why): correctly
	cased single codepoints, but NOT one-to-many
	expansions (ß stays ß, not SS) or context-sensitive rules (Greek final
	sigma). newlocale('C.UTF-8') is looked up once per call (not cached
	process-wide) specifically to avoid the data race a shared/global
	locale object would need locking around - see crt.py's own comment; a
	NULL result (locale unavailable - possible on older systems) falls
	back to towupper_l/towlower_l's own "C"-locale behavior (ASCII-only,
	same ceiling an old implementation would have had, everything else
	passed through unchanged) rather than failing outright.

	Two passes over the codepoints (see str.concat's own identical shape):
	the first sums the cased codepoints' own encoded byte lengths (which
	can differ from the input's - U+00E9 (2 bytes) uppercases to U+00C9
	(also 2 bytes), but not everything stays put), the second actually
	encodes into the freshly, exactly sized buffer. '''
	from crt import newlocale, freelocale, LC_CTYPE_MASK
	if byte_len == 0:
		empty_buf: Ptr[u8] = sys.alloc[u8]( 1 )
		empty_buf[0] = 0
		out_size[0] = 1
		return empty_buf
	loc: Ptr[None] = newlocale( LC_CTYPE_MASK, 'C.UTF-8'.get_cstr(), None )
	if loc is not None:
		defer( freelocale( loc ))

	new_size: usize = 1 # zero terminator
	i: usize = 0
	consumed: usize = 0
	while i < byte_len:
		cp: u32 = decode_utf8_at( data, i, compiler.addrof( consumed ))
		cased: u32 = _case_codepoint_posix( cp, is_upper, loc )
		with compiler.panic_arithmetic( 'irrational string length' ):
			new_size += utf8_encoded_len( cased )
		with compiler.wrap_arithmetic:
			i += consumed

	new_buf: Ptr[u8] = sys.alloc[u8]( new_size )
	out_i: usize = 0
	i = 0
	while i < byte_len:
		cp = decode_utf8_at( data, i, compiler.addrof( consumed ))
		cased = _case_codepoint_posix( cp, is_upper, loc )
		with compiler.wrap_arithmetic:
			out_i += encode_utf8_at( new_buf, out_i, cased )
			i += consumed
	new_buf[out_i] = 0

	out_size[0] = new_size
	return new_buf
