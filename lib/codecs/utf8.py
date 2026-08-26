import sys
from . import Codec, CodecError, DecodeErrors, _emit_lossy_unit

class Utf8( Codec ):
	@virtual
	def names( self ) -> list[str]:
		return [ 'utf8', 'utf-8', 'UTF8', 'UTF-8' ]

	@virtual
	def encode( self, s: str ) -> Result[bytes,CodecError]:
		'''
		this function can't actually generate a CodecError because str's underlying data is guaranteed to be utf-8
		However, the base class Codec.encode() function definition requires it
		'''
		length: usize = s.byte_len()
		out = bytearray( length )
		sys.memcpy( out.get_ptr(), s.get_const_ptr(), length )
		return Result.Ok( bytes.from_bytearray( move( out )))

	@virtual
	def decode( self, b: bytes|bytearray ) -> Result[str,CodecError]:
		# NOTE: there's no need to check for valid utf-8 encoding here, because
		# str._from_owned_cstr() has to do it anyway. b's own raw bytes carry
		# no guaranteed trailing zero terminator (e.g. a syscall-filled buffer,
		# or a slice) - same "allocate len+1, memcpy, terminate explicitly"
		# shape str._byte_slice/str.__add__/etc already use, not str.from_cstr's
		# own (buf, size_including_zero_terminator) overload, which REQUIRES
		# the caller's own buffer to already end in a real \0 at size-1
		length: usize = len( b )
		with compiler.panic_arithmetic( 'irrational byte length' ):
			buf_size: usize = length + 1
		new_buf: Ptr[u8] = sys.alloc[u8]( buf_size )
		sys.memcpy( new_buf, b.get_const_ptr(), length )
		new_buf[length] = 0
		return str._from_owned_cstr( new_buf, buf_size )

	@virtual
	def decode_lossy( self, b: bytes|bytearray, errors: DecodeErrors = DecodeErrors.BackslashReplace ) -> str:
		''' same sequence-validity scan as decode()/_from_owned_cstr, but
		malformed bytes are handled one at a time per `errors` instead of
		bailing on the first bad byte - so this can't just delegate to
		_from_owned_cstr, it has to build the output itself. '''
		length: usize = len( b )
		ptr: ConstPtr[u8] = b.get_const_ptr()

		# worst case: every byte is malformed and backslash-escaped (4 out bytes each)
		with compiler.panic_arithmetic( 'irrational byte length' ):
			out = bytearray( length * 4 )
		out_ptr: Ptr[u8] = out.get_ptr()
		out_idx: usize = 0

		i: usize = 0
		with compiler.panic_arithmetic( 'bounded by length, cannot overflow' ):
			while i < length:
				byte1 = ptr[i]
				seq_len: usize = 0

				if ( byte1 & 0x80 ) == 0x00:
					seq_len = 1
				elif ( byte1 & 0xE0 ) == 0xC0 and byte1 >= 0xC2 \
						and i + 1 < length and ( ptr[i + 1] & 0xC0 ) == 0x80:
					seq_len = 2
				elif ( byte1 & 0xF0 ) == 0xE0 and i + 2 < length \
						and ( ptr[i + 1] & 0xC0 ) == 0x80 and ( ptr[i + 2] & 0xC0 ) == 0x80 \
						and not ( byte1 == 0xE0 and ptr[i + 1] < 0xA0 ) \
						and not ( byte1 == 0xED and ptr[i + 1] >= 0xA0 ):
					seq_len = 3
				elif ( byte1 & 0xF8 ) == 0xF0 and byte1 <= 0xF4 and i + 3 < length \
						and ( ptr[i + 1] & 0xC0 ) == 0x80 and ( ptr[i + 2] & 0xC0 ) == 0x80 and ( ptr[i + 3] & 0xC0 ) == 0x80 \
						and not ( byte1 == 0xF0 and ptr[i + 1] < 0x90 ) \
						and not ( byte1 == 0xF4 and ptr[i + 1] >= 0x90 ):
					seq_len = 4

				if seq_len == 0:
					out_idx = _emit_lossy_unit( out_ptr, out_idx, byte1, errors )
					i += 1
					continue

				j: usize = 0
				with compiler.panic_arithmetic( 'seq_len bounded by loop condition above' ):
					while j < seq_len:
						out_ptr[out_idx] = ptr[i + j]
						out_idx += 1
						j += 1
				i += seq_len

		with compiler.panic_arithmetic( 'irrational byte length' ):
			buf_size: usize = out_idx + 1
		new_buf: Ptr[u8] = sys.alloc[u8]( buf_size )
		sys.memcpy( new_buf, out_ptr, out_idx )
		new_buf[out_idx] = 0
		return str._from_owned_cstr( new_buf, buf_size ).unwrap( 'decode_lossy always produces valid utf-8 by construction' )

# Utf8 is stateless (no __init__, no fields) - one shared instance is safe
# and avoids constructing a fresh one at every default-parameter-value site
# (lib/posix/fs.py's readlink, bytes.decode/bytearray.decode/str.encode's
# own codec: Codec = utf8 defaults, lib/codecs/__init__.py's _build_registry).
# Deliberately named the SAME as the old class identifier: every existing
# bare `utf8` reference (imports, defaults, utf8.decode(self)-shaped calls)
# keeps working completely unchanged now that it resolves to a real
# instance instead of the class - only call sites that explicitly try to
# CONSTRUCT it (utf8()) need to change, since it's no longer callable.
utf8: Utf8 = Utf8()
