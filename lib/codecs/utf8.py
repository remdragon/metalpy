import sys
from . import Codec, CodecError

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
