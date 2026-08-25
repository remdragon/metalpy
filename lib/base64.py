# base64 - RFC 4648 base64/urlsafe-base64/base16 encode+decode, mirroring
# Python's base64 module. See PLAN_HTTP_CLIENT.md, which names this as a
# zero-prerequisite piece needed for auth= (HTTP Basic -> base64 Authorization
# header).
#
# Scope for this pass (agreed with the user): b64encode/b64decode,
# urlsafe_b64encode/urlsafe_b64decode, b16encode/b16decode. Base32,
# Ascii85/Base85, and the legacy encodebytes()/decodebytes() (MIME
# line-wrapped) functions are out of scope.
#
# Decoding takes a `validate` flag for API-shape parity with Python, but
# DEFAULTS TO True (strict) - the opposite of Python's own lenient default -
# matching this codebase's general preference (guid.py, ascii.py) for
# erroring on malformed input rather than silently discarding bad
# characters. validate=False still implements Python's lenient behavior
# (non-alphabet bytes, e.g. embedded whitespace, are discarded first).

class Base64Error:
	message: str

	def __init__( self, message: str ) -> None:
		self.message = message

	def __str__( self ) -> str:
		return self.message

	def __repr__( self ) -> str:
		return f"Base64Error({self.message!r})"


# 64-entry alphabets: 'A'-'Z', 'a'-'z', '0'-'9', then the two symbols that
# differ between standard and urlsafe. Spelled out as byte literals - this
# language has no ord()/chr() (confirmed absent, see lib/builtins/__int.py),
# matching guid.py's own established idiom.
_B64_STD_ALPHABET: list[u8] = [
	0x41, 0x42, 0x43, 0x44, 0x45, 0x46, 0x47, 0x48, 0x49, 0x4A, 0x4B, 0x4C, 0x4D,
	0x4E, 0x4F, 0x50, 0x51, 0x52, 0x53, 0x54, 0x55, 0x56, 0x57, 0x58, 0x59, 0x5A,
	0x61, 0x62, 0x63, 0x64, 0x65, 0x66, 0x67, 0x68, 0x69, 0x6A, 0x6B, 0x6C, 0x6D,
	0x6E, 0x6F, 0x70, 0x71, 0x72, 0x73, 0x74, 0x75, 0x76, 0x77, 0x78, 0x79, 0x7A,
	0x30, 0x31, 0x32, 0x33, 0x34, 0x35, 0x36, 0x37, 0x38, 0x39, 0x2B, 0x2F,
]

_B64_URLSAFE_ALPHABET: list[u8] = [
	0x41, 0x42, 0x43, 0x44, 0x45, 0x46, 0x47, 0x48, 0x49, 0x4A, 0x4B, 0x4C, 0x4D,
	0x4E, 0x4F, 0x50, 0x51, 0x52, 0x53, 0x54, 0x55, 0x56, 0x57, 0x58, 0x59, 0x5A,
	0x61, 0x62, 0x63, 0x64, 0x65, 0x66, 0x67, 0x68, 0x69, 0x6A, 0x6B, 0x6C, 0x6D,
	0x6E, 0x6F, 0x70, 0x71, 0x72, 0x73, 0x74, 0x75, 0x76, 0x77, 0x78, 0x79, 0x7A,
	0x30, 0x31, 0x32, 0x33, 0x34, 0x35, 0x36, 0x37, 0x38, 0x39, 0x2D, 0x5F,
]

_HEX_UPPER: list[u8] = [
	0x30, 0x31, 0x32, 0x33, 0x34, 0x35, 0x36, 0x37,
	0x38, 0x39, 0x41, 0x42, 0x43, 0x44, 0x45, 0x46,
]

_PAD: u8 = 0x3D # '='


def b64encode( data: bytes|bytearray ) -> bytes:
	return _b64encode( data, _B64_STD_ALPHABET )


def urlsafe_b64encode( data: bytes|bytearray ) -> bytes:
	return _b64encode( data, _B64_URLSAFE_ALPHABET )


def b64decode( data: bytes|bytearray, validate: bool = True ) -> Result[bytes,Base64Error]:
	return _b64decode( data, 0x2B, 0x2F, validate ) # '+', '/'


def urlsafe_b64decode( data: bytes|bytearray, validate: bool = True ) -> Result[bytes,Base64Error]:
	return _b64decode( data, 0x2D, 0x5F, validate ) # '-', '_'


def b16encode( data: bytes|bytearray ) -> bytes:
	n: usize = len( data )
	in_ptr: ConstPtr[u8] = data.get_const_ptr()

	with compiler.panic_arithmetic( 'output length is exactly double input length' ):
		out_len: usize = n * 2
	out = bytearray( out_len )
	out_ptr: Ptr[u8] = out.get_ptr()

	i: usize = 0
	with compiler.panic_arithmetic( 'bounded by n, cannot overflow' ):
		while i < n:
			b: u8 = in_ptr[i]
			out_ptr[i * 2] = _HEX_UPPER.__getitem__( usize( b >> 4 ) ).unwrap( 'nibble always < 16' )
			out_ptr[i * 2 + 1] = _HEX_UPPER.__getitem__( usize( b & 0x0F ) ).unwrap( 'nibble always < 16' )
			i += 1

	return bytes.from_bytearray( move( out ) )


def b16decode( data: bytes|bytearray, casefold: bool = False ) -> Result[bytes,Base64Error]:
	n: usize = len( data )
	in_ptr: ConstPtr[u8] = data.get_const_ptr()

	with compiler.panic_arithmetic( 'divisor is a nonzero literal' ):
		is_odd: bool = n % 2 != 0
	if is_odd:
		return Result.Err( Base64Error( 'invalid base16-encoded string: odd length' ) )

	with compiler.panic_arithmetic( 'output length is exactly half input length' ):
		out_len: usize = n // 2
	out = bytearray( out_len )
	out_ptr: Ptr[u8] = out.get_ptr()

	i: usize = 0
	with compiler.panic_arithmetic( 'bounded by n, cannot overflow' ):
		while i < n:
			hi: u8 = _hex_nibble_value( in_ptr[i], casefold ).or_return()
			lo: u8 = _hex_nibble_value( in_ptr[i + 1], casefold ).or_return()
			out_ptr[i // 2] = ( hi << 4 ) | lo
			i += 2

	return Result.Ok( bytes.from_bytearray( move( out ) ) )


def _b64encode( data: bytes|bytearray, alphabet: list[u8] ) -> bytes:
	''' Shared by b64encode/urlsafe_b64encode - only the alphabet table
	differs. Output length is exactly computable up front (unlike a codec's
	variable-collapse case, e.g. lib/codecs/cp437.py's UTF-8<->CP437), so a
	single exactly-sized bytearray suffices - no worst-case-alloc-then-
	copy-down step needed. '''
	n: usize = len( data )
	in_ptr: ConstPtr[u8] = data.get_const_ptr()

	with compiler.panic_arithmetic( 'output length is a fixed function of input length' ):
		full_groups: usize = n // 3
		remainder: usize = n % 3
		out_len: usize = full_groups * 4
		if remainder > 0:
			out_len = out_len + 4

	out = bytearray( out_len )
	out_ptr: Ptr[u8] = out.get_ptr()

	in_idx: usize = 0
	out_idx: usize = 0
	with compiler.panic_arithmetic( 'bounded by n, cannot overflow' ):
		while in_idx + 3 <= n:
			b0: u8 = in_ptr[in_idx]
			b1: u8 = in_ptr[in_idx + 1]
			b2: u8 = in_ptr[in_idx + 2]
			v: u32 = ( u32( b0 ) << 16 ) | ( u32( b1 ) << 8 ) | u32( b2 )
			out_ptr[out_idx] = alphabet.__getitem__( usize( ( v >> 18 ) & 0x3F ) ).unwrap( 'sextet always < 64' )
			out_ptr[out_idx + 1] = alphabet.__getitem__( usize( ( v >> 12 ) & 0x3F ) ).unwrap( 'sextet always < 64' )
			out_ptr[out_idx + 2] = alphabet.__getitem__( usize( ( v >> 6 ) & 0x3F ) ).unwrap( 'sextet always < 64' )
			out_ptr[out_idx + 3] = alphabet.__getitem__( usize( v & 0x3F ) ).unwrap( 'sextet always < 64' )
			in_idx += 3
			out_idx += 4

		# tb0/tv declared bare here, once - a variable's type is only ever
		# declared once per function, even across mutually exclusive
		# if/elif arms (see del_reuse_and_emitter_naming_bug)
		tb0: u8
		tv: u32
		if remainder == 1:
			tb0 = in_ptr[in_idx]
			tv = u32( tb0 ) << 16
			out_ptr[out_idx] = alphabet.__getitem__( usize( ( tv >> 18 ) & 0x3F ) ).unwrap( 'sextet always < 64' )
			out_ptr[out_idx + 1] = alphabet.__getitem__( usize( ( tv >> 12 ) & 0x3F ) ).unwrap( 'sextet always < 64' )
			out_ptr[out_idx + 2] = _PAD
			out_ptr[out_idx + 3] = _PAD
		elif remainder == 2:
			tb0 = in_ptr[in_idx]
			tb1: u8 = in_ptr[in_idx + 1]
			tv = ( u32( tb0 ) << 16 ) | ( u32( tb1 ) << 8 )
			out_ptr[out_idx] = alphabet.__getitem__( usize( ( tv >> 18 ) & 0x3F ) ).unwrap( 'sextet always < 64' )
			out_ptr[out_idx + 1] = alphabet.__getitem__( usize( ( tv >> 12 ) & 0x3F ) ).unwrap( 'sextet always < 64' )
			out_ptr[out_idx + 2] = alphabet.__getitem__( usize( ( tv >> 6 ) & 0x3F ) ).unwrap( 'sextet always < 64' )
			out_ptr[out_idx + 3] = _PAD

	return bytes.from_bytearray( move( out ) )


def _b64_char_value( c: u8, sym62: u8, sym63: u8 ) -> Result[u8,Base64Error]:
	''' Maps one base64 alphabet character to its 6-bit value (0-63), or Err
	if `c` isn't part of the alphabet - including '=', which never matches
	any of these ranges/symbols, so mid-stream padding is rejected for
	free. The two non-alphanumeric symbols differ between the standard
	('+'/'/') and urlsafe ('-'/'_') alphabets, so the caller passes which
	pair applies. Same range-check idiom as guid.py's own
	_hex_digit_value, extended to base64's 4 contiguous ranges. Unlike
	_hex_digit_value this returns Result rather than panicking - this
	decodes attacker/user-controlled input, not a hardcoded literal, so
	malformed input is genuinely recoverable (same distinction guid.py's
	own docstring draws). '''
	if c >= 0x41 and c <= 0x5A: # 'A'-'Z'
		with compiler.wrap_arithmetic:
			return Result.Ok( c - 0x41 )
	if c >= 0x61 and c <= 0x7A: # 'a'-'z'
		with compiler.wrap_arithmetic:
			return Result.Ok( c - 0x61 + 26 )
	if c >= 0x30 and c <= 0x39: # '0'-'9'
		with compiler.wrap_arithmetic:
			return Result.Ok( c - 0x30 + 52 )
	if c == sym62:
		return Result.Ok( 62 )
	if c == sym63:
		return Result.Ok( 63 )
	return Result.Err( Base64Error( 'invalid base64 character' ) )


def _hex_nibble_value( c: u8, casefold: bool ) -> Result[u8,Base64Error]:
	if c >= 0x30 and c <= 0x39: # '0'-'9'
		with compiler.wrap_arithmetic:
			return Result.Ok( c - 0x30 )
	if c >= 0x41 and c <= 0x46: # 'A'-'F'
		with compiler.wrap_arithmetic:
			return Result.Ok( c - 0x41 + 10 )
	if casefold and c >= 0x61 and c <= 0x66: # 'a'-'f'
		with compiler.wrap_arithmetic:
			return Result.Ok( c - 0x61 + 10 )
	return Result.Err( Base64Error( 'invalid base16 character' ) )


def _b64decode( data: bytes|bytearray, sym62: u8, sym63: u8, validate: bool ) -> Result[bytes,Base64Error]:
	n: usize = len( data )
	in_ptr: ConstPtr[u8] = data.get_const_ptr()

	if validate:
		return _b64decode_core( in_ptr, n, sym62, sym63 )

	# Lenient mode (matches Python's own validate=False default): discard
	# any byte that isn't part of the alphabet (or '=') before decoding,
	# e.g. embedded whitespace/newlines. Count valid bytes first, then
	# copy them down into an exactly-sized compacted buffer - same
	# "count first, allocate exact, fill second" shape used elsewhere in
	# this codebase, rather than allocating worst-case and copying down
	# after.
	valid_count: usize = 0
	i: usize = 0
	with compiler.panic_arithmetic( 'bounded by n, cannot overflow' ):
		while i < n:
			c: u8 = in_ptr[i]
			if c == _PAD or _b64_char_value( c, sym62, sym63 ).is_ok():
				valid_count += 1
			i += 1

	filtered = bytearray( valid_count )
	f_ptr: Ptr[u8] = filtered.get_ptr()
	f_idx: usize = 0
	i = 0
	with compiler.panic_arithmetic( 'bounded by n, cannot overflow' ):
		while i < n:
			# distinct name from the counting loop's own `c` above - a
			# variable's type is only ever declared once per function
			fc: u8 = in_ptr[i]
			if fc == _PAD or _b64_char_value( fc, sym62, sym63 ).is_ok():
				f_ptr[f_idx] = fc
				f_idx += 1
			i += 1

	return _b64decode_core( filtered.get_const_ptr(), valid_count, sym62, sym63 )


def _b64decode_core( in_ptr: ConstPtr[u8], n: usize, sym62: u8, sym63: u8 ) -> Result[bytes,Base64Error]:
	if n == 0:
		return Result.Ok( bytes( bytearray( 0 ) ) )

	with compiler.panic_arithmetic( 'divisor is a nonzero literal' ):
		is_misaligned: bool = n % 4 != 0
	if is_misaligned:
		return Result.Err( Base64Error( 'invalid base64-encoded string: length is not a multiple of 4' ) )

	pad_count: usize = 0
	with compiler.panic_arithmetic( 'n is a positive multiple of 4, checked above' ):
		if in_ptr[n - 1] == _PAD:
			pad_count += 1
			if in_ptr[n - 2] == _PAD:
				pad_count += 1

	with compiler.panic_arithmetic( 'n is a positive multiple of 4, checked above' ):
		out_len: usize = ( n // 4 ) * 3 - pad_count
		full_group_end: usize = n - 4

	out = bytearray( out_len )
	out_ptr: Ptr[u8] = out.get_ptr()

	in_idx: usize = 0
	out_idx: usize = 0
	with compiler.panic_arithmetic( 'bounded by n, cannot overflow' ):
		while in_idx < full_group_end:
			c0: u8 = in_ptr[in_idx]
			c1: u8 = in_ptr[in_idx + 1]
			c2: u8 = in_ptr[in_idx + 2]
			c3: u8 = in_ptr[in_idx + 3]
			v0: u8 = _b64_char_value( c0, sym62, sym63 ).or_return()
			v1: u8 = _b64_char_value( c1, sym62, sym63 ).or_return()
			v2: u8 = _b64_char_value( c2, sym62, sym63 ).or_return()
			v3: u8 = _b64_char_value( c3, sym62, sym63 ).or_return()
			v: u32 = ( u32( v0 ) << 18 ) | ( u32( v1 ) << 12 ) | ( u32( v2 ) << 6 ) | u32( v3 )
			out_ptr[out_idx] = u8( ( v >> 16 ) & 0xFF )
			out_ptr[out_idx + 1] = u8( ( v >> 8 ) & 0xFF )
			out_ptr[out_idx + 2] = u8( v & 0xFF )
			in_idx += 4
			out_idx += 3

		# Final group (exactly 4 chars): the only place padding may appear.
		# f-prefixed names throughout - distinct from the main loop's own
		# c0-c3/v0-v3 above (this language has no block scoping, so those
		# are still live here even though the loop itself has exited) and
		# from each other's own if/elif/else arm below (v2/pv declared
		# bare once, since a variable's type is only ever declared once
		# per function, even across mutually exclusive branches)
		fc0: u8 = in_ptr[in_idx]
		fc1: u8 = in_ptr[in_idx + 1]
		fc2: u8 = in_ptr[in_idx + 2]
		fc3: u8 = in_ptr[in_idx + 3]
		fv0: u8 = _b64_char_value( fc0, sym62, sym63 ).or_return()
		fv1: u8 = _b64_char_value( fc1, sym62, sym63 ).or_return()
		fv2: u8
		pv: u32

		if pad_count == 2:
			if fc2 != _PAD or fc3 != _PAD:
				return Result.Err( Base64Error( 'invalid base64 padding' ) )
			pv = ( u32( fv0 ) << 18 ) | ( u32( fv1 ) << 12 )
			out_ptr[out_idx] = u8( ( pv >> 16 ) & 0xFF )
		elif pad_count == 1:
			if fc3 != _PAD or fc2 == _PAD:
				return Result.Err( Base64Error( 'invalid base64 padding' ) )
			fv2 = _b64_char_value( fc2, sym62, sym63 ).or_return()
			pv = ( u32( fv0 ) << 18 ) | ( u32( fv1 ) << 12 ) | ( u32( fv2 ) << 6 )
			out_ptr[out_idx] = u8( ( pv >> 16 ) & 0xFF )
			out_ptr[out_idx + 1] = u8( ( pv >> 8 ) & 0xFF )
		else:
			if fc2 == _PAD or fc3 == _PAD:
				return Result.Err( Base64Error( 'invalid base64 padding' ) )
			fv2 = _b64_char_value( fc2, sym62, sym63 ).or_return()
			fv3: u8 = _b64_char_value( fc3, sym62, sym63 ).or_return()
			pv = ( u32( fv0 ) << 18 ) | ( u32( fv1 ) << 12 ) | ( u32( fv2 ) << 6 ) | u32( fv3 )
			out_ptr[out_idx] = u8( ( pv >> 16 ) & 0xFF )
			out_ptr[out_idx + 1] = u8( ( pv >> 8 ) & 0xFF )
			out_ptr[out_idx + 2] = u8( pv & 0xFF )

	return Result.Ok( bytes.from_bytearray( move( out ) ) )
