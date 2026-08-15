import sys
from . import Codec, CodecError

class ascii( Codec ):
	@virtual
	def names( self ) -> list[str]:
		return [ 'ascii', 'us-ascii', 'US-ASCC', 'cp646' ]

	@virtual
	def encode( self, s: str ) -> Result[bytes,CodecError]:
		length: usize = s.byte_len()
		out = bytearray( length )
		
		s_ptr: ConstPtr[u8] = s.get_const_ptr()
		out_ptr: Ptr[u8] = out.get_ptr()

		i: usize = 0
		with compiler.panic_arithmetic( 'bounded by length, cannot overflow' ):
			while i < length:
				byte: u8 = s_ptr[i]
				if byte > 0x7F:
					return Result.Err(
						CodecError( 'ascii', 'Ordinal out of range(128)' )
					)
				out_ptr[i] = byte
				i += 1

		# NOTE: bytes(out) (copy) rather than bytes.from_bytearray(move(out))
		# - a real, general, pre-existing compiler bug (confirmed via a
		# minimal repro, not specific to this file): an RC-tracked local
		# consumed via move() on the function's fall-through success path,
		# with an earlier `return` still reachable while that local is live,
		# leaves the early return's own epilogue-cleanup label unemitted
		# (current_epilogue_label() hands the return a label whose backing
		# _epilogue_stack entry the later move() consumption then silently
		# drops, instead of leaving a decref-less "cancelled" entry behind
		# the way every other consumption path does) - "use of undeclared
		# label" at the C level. Reported separately; this file just avoids
		# the trigger shape.
		return Result.Ok( bytes( out ))
	
	@virtual
	def decode( self, b: bytes|bytearray ) -> Result[str,CodecError]:
		length: usize = len( b )
		ptr: ConstPtr[u8] = b.get_const_ptr()
		
		i: usize = 0
		with compiler.panic_arithmetic( 'bounded by length, cannot overflow' ):
			while i < length:
				if ptr[i] > 0x7F:
					return Result.Err(
						CodecError( 'ascii', 'Ordinal out of range(128)' )
					)
				i += 1

		with compiler.panic_arithmetic( 'irrational byte length' ):
			buf_size: usize = length + 1
		new_buf: Ptr[u8] = sys.alloc[u8]( buf_size )
		sys.memcpy( new_buf, ptr, length )
		new_buf[length] = 0
		return str._from_owned_cstr( new_buf, buf_size )
