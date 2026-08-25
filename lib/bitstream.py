# bitstream - LSB-first bit-level I/O (RFC 1951 SS3.1.1), the primitive
# lib/deflate.py's block framing and lib/huffman.py's code packing are both
# built on. Knows nothing about Huffman codes or DEFLATE block structure -
# just fixed-width bit fields and raw byte alignment.

import compiler
import sys


class BitstreamError:
	message: str

	def __init__( self, message: str ) -> None:
		self.message = message

	def __str__( self ) -> str:
		return self.message

	def __repr__( self ) -> str:
		return f"BitstreamError({self.message!r})"


# ---------------------------------------------------------------------------
# BitWriter
# ---------------------------------------------------------------------------

class BitWriter:
	__buf:       bytearray  # amortized-doubling backing store - see _push_byte
	__used:      usize      # bytes actually written so far ( <= len( __buf ))
	__bitbuf:    u32        # pending bits not yet flushed to a whole byte
	__bitcount:  u32        # number of valid low bits in __bitbuf ( < 8 between calls )

	def __init__( self ) -> None:
		self.__buf      = bytearray( 64 )
		self.__used     = 0
		self.__bitbuf   = 0
		self.__bitcount = 0

	# Writes the low `nbits` bits of `value`, LSB-first (bit 0 of value is
	# transmitted first). nbits must be <= 16 - covers every fixed-width
	# DEFLATE field (BTYPE, HLIT/HDIST/HCLEN, length/distance extra bits);
	# Huffman codes themselves are written via huffman.py, which reverses
	# the code's bits before calling this (see that module's own comment).
	def write_bits( self, value: u32, nbits: u32 ) -> None:
		with compiler.wrap_arithmetic:
			mask: u32 = ( u32( 1 ) << nbits ) - 1
			self.__bitbuf |= ( value & mask ) << self.__bitcount
			self.__bitcount += nbits
			while self.__bitcount >= 8:
				self._push_byte( u8( self.__bitbuf & 0xFF ))
				self.__bitbuf >>= 8
				self.__bitcount -= 8

	def _push_byte( self, b: u8 ) -> None:
		if self.__used == len( self.__buf ):
			with compiler.wrap_arithmetic:
				new_cap: usize = len( self.__buf ) * 2
			self.__buf.resize( new_cap )
		self.__buf.get_ptr()[ self.__used ] = b
		with compiler.wrap_arithmetic:
			self.__used += 1

	# Pads any partial byte with zero bits and flushes it. A no-op if
	# already byte-aligned. Required before a STORED block's LEN/~LEN/data,
	# and before write_aligned_bytes.
	def align_to_byte( self ) -> None:
		if self.__bitcount > 0:
			with compiler.wrap_arithmetic:
				self._push_byte( u8( self.__bitbuf & 0xFF ))
			self.__bitbuf   = 0
			self.__bitcount = 0

	# Writes raw bytes with no bit packing - caller must already be
	# byte-aligned (STORED blocks only).
	def write_aligned_bytes( self, buf: ConstPtr[u8], count: usize ) -> None:
		i: usize = 0
		with compiler.panic_arithmetic( 'bounded by count, cannot overflow' ):
			while i < count:
				self._push_byte( buf[i] )
				i += 1

	def byte_len( self ) -> usize:
		return self.__used

	# Pads the final partial byte and returns everything written so far as
	# an owned bytes. Consumes the writer's backing buffer - swap in a
	# fresh empty one first (self.__buf is an RC field; move() only tracks
	# plain local/param bindings, not fields - see this module's own
	# spawned follow-up task on that compiler gap) so bytes.from_bytearray
	# can validly move() a LOCAL binding instead.
	def finish( self ) -> bytes:
		self.align_to_byte()
		self.__buf.resize( self.__used )
		out: bytearray = self.__buf
		self.__buf = bytearray( 0 )
		return bytes.from_bytearray( move( out ))


# ---------------------------------------------------------------------------
# BitReader
# ---------------------------------------------------------------------------

class BitReader:
	__data:      ConstPtr[u8]
	__len:       usize
	__pos:       usize      # next unread byte index
	__bitbuf:    u32        # pending bits already pulled from __data
	__bitcount:  u32        # number of valid low bits in __bitbuf

	def __init__( self, data: bytes|bytearray ) -> None:
		self.__data      = data.get_const_ptr()
		self.__len       = len( data )
		self.__pos       = 0
		self.__bitbuf    = 0
		self.__bitcount  = 0

	# Reads `nbits` bits LSB-first (mirrors write_bits - the first bit
	# read becomes bit 0 of the result, etc). nbits must be <= 16.
	def read_bits( self, nbits: u32 ) -> Result[u32, BitstreamError]:
		with compiler.wrap_arithmetic:
			while self.__bitcount < nbits:
				if self.__pos == self.__len:
					return Result.Err( BitstreamError( 'read_bits: unexpected end of stream' ))
				self.__bitbuf |= u32( self.__data[ self.__pos ] ) << self.__bitcount
				self.__pos += 1
				self.__bitcount += 8
			mask: u32 = ( u32( 1 ) << nbits ) - 1
			result: u32 = self.__bitbuf & mask
			self.__bitbuf >>= nbits
			self.__bitcount -= nbits
			return Result.Ok( result )

	# Discards any partial byte of already-buffered bits, resuming reads
	# at the next whole byte boundary. Mirrors BitWriter.align_to_byte.
	def align_to_byte( self ) -> None:
		self.__bitbuf   = 0
		self.__bitcount = 0

	# Reads `count` raw bytes with no bit unpacking - caller must already
	# be byte-aligned (STORED blocks only).
	def read_aligned_bytes( self, buf: Ptr[u8], count: usize ) -> Result[None, BitstreamError]:
		with compiler.panic_arithmetic( 'bounded by __len, cannot overflow' ):
			if self.__pos + count > self.__len:
				return Result.Err( BitstreamError( 'read_aligned_bytes: unexpected end of stream' ))
			src: ConstPtr[u8] = self.__data + self.__pos
			sys.memcpy( buf, src, count )
			self.__pos += count
		return Result.Ok( None )

	def bits_remaining( self ) -> usize:
		with compiler.wrap_arithmetic:
			whole_bytes_left: usize = self.__len - self.__pos
			return whole_bytes_left * 8 + usize( self.__bitcount )
