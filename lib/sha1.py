# sha1 - FIPS 180-1 SHA-1. One-shot digest: sha1(data) -> 20 raw digest
# bytes. No streaming update()/digest() split, same posture as lib/
# sha256.py - the only known caller is mysql.py's mysql_native_password
# scramble (small fixed-size buffers).
#
# SHA-1 is cryptographically broken for collision resistance but still the
# literal wire-protocol requirement for MySQL/MariaDB's mysql_native_password
# auth plugin - not a choice, a compatibility requirement.

import compiler


_H0: list[u32] = [ 0x67452301, 0xEFCDAB89, 0x98BADCFE, 0x10325476, 0xC3D2E1F0 ]


def _rotl( x: u32, n: u32 ) -> u32:
	with compiler.wrap_arithmetic:
		return ( x << n ) | ( x >> ( 32 - n ))


def sha1( data: bytes|bytearray ) -> bytes:
	n: usize = len( data )
	in_ptr: ConstPtr[u8] = data.get_const_ptr()

	# same padding scheme as sha256.sha256(): msg || 0x80 || zero pad ||
	# 8-byte big-endian bit length, total a multiple of 64.
	with compiler.panic_arithmetic( 'padded length is a small bounded function of input length' ):
		pad_len: usize = (( n + 9 + 63 ) // 64 ) * 64

	msg = bytearray( pad_len )
	msg_ptr: Ptr[u8] = msg.get_ptr()

	i: usize = 0
	with compiler.panic_arithmetic( 'bounded by n < pad_len' ):
		while i < n:
			msg_ptr[i] = in_ptr[i]
			i += 1
	msg_ptr[n] = 0x80

	with compiler.wrap_arithmetic:
		bit_len: u64 = u64( n ) << 3
		length_at: usize = pad_len - 8
		msg_ptr[length_at]     = u8( ( bit_len >> 56 ) & 0xFF )
		msg_ptr[length_at + 1] = u8( ( bit_len >> 48 ) & 0xFF )
		msg_ptr[length_at + 2] = u8( ( bit_len >> 40 ) & 0xFF )
		msg_ptr[length_at + 3] = u8( ( bit_len >> 32 ) & 0xFF )
		msg_ptr[length_at + 4] = u8( ( bit_len >> 24 ) & 0xFF )
		msg_ptr[length_at + 5] = u8( ( bit_len >> 16 ) & 0xFF )
		msg_ptr[length_at + 6] = u8( ( bit_len >> 8  ) & 0xFF )
		msg_ptr[length_at + 7] = u8(   bit_len         & 0xFF )

	h: UnsafeList[u32] = UnsafeList[u32]( usize( 5 ))
	hi: usize = 0
	with compiler.panic_arithmetic( 'hi < 5, cannot overflow' ):
		while hi < 5:
			h.append( _H0.__getitem__( hi ).unwrap( 'hi < 5' ))
			hi += 1

	block: usize = 0
	with compiler.wrap_arithmetic:
		while block < pad_len:
			w: UnsafeList[u32] = UnsafeList[u32]( usize( 80 ))
			t: usize = 0
			while t < 16:
				bo: usize = block + t * 4
				w.append(
					( u32( msg_ptr[bo] )     << 24 ) |
					( u32( msg_ptr[bo + 1] ) << 16 ) |
					( u32( msg_ptr[bo + 2] ) << 8  ) |
					  u32( msg_ptr[bo + 3] )
				)
				t += 1
			while t < 80:
				wm3:  u32 = w.__getitem__( t - 3 ).unwrap( 't >= 16' )
				wm8:  u32 = w.__getitem__( t - 8 ).unwrap( 't >= 16' )
				wm14: u32 = w.__getitem__( t - 14 ).unwrap( 't >= 16' )
				wm16: u32 = w.__getitem__( t - 16 ).unwrap( 't >= 16' )
				w.append( _rotl( wm3 ^ wm8 ^ wm14 ^ wm16, 1 ))
				t += 1

			a: u32 = h.__getitem__( 0 ).unwrap( 'h has 5 entries' )
			b: u32 = h.__getitem__( 1 ).unwrap( 'h has 5 entries' )
			c: u32 = h.__getitem__( 2 ).unwrap( 'h has 5 entries' )
			d: u32 = h.__getitem__( 3 ).unwrap( 'h has 5 entries' )
			e: u32 = h.__getitem__( 4 ).unwrap( 'h has 5 entries' )

			rt: usize = 0
			while rt < 80:
				f: u32 = 0
				k: u32 = 0
				if rt < 20:
					f = ( b & c ) | (( ~b ) & d )
					k = 0x5A827999
				elif rt < 40:
					f = b ^ c ^ d
					k = 0x6ED9EBA1
				elif rt < 60:
					f = ( b & c ) | ( b & d ) | ( c & d )
					k = 0x8F1BBCDC
				else:
					f = b ^ c ^ d
					k = 0xCA62C1D6

				wt: u32 = w.__getitem__( rt ).unwrap( 'rt < 80' )
				temp: u32 = _rotl( a, 5 ) + f + e + k + wt
				e = d
				d = c
				c = _rotl( b, 30 )
				b = a
				a = temp
				rt += 1

			h.__setitem__( 0, h.__getitem__( 0 ).unwrap( 'h has 5 entries' ) + a ).unwrap( 'idx < 5' )
			h.__setitem__( 1, h.__getitem__( 1 ).unwrap( 'h has 5 entries' ) + b ).unwrap( 'idx < 5' )
			h.__setitem__( 2, h.__getitem__( 2 ).unwrap( 'h has 5 entries' ) + c ).unwrap( 'idx < 5' )
			h.__setitem__( 3, h.__getitem__( 3 ).unwrap( 'h has 5 entries' ) + d ).unwrap( 'idx < 5' )
			h.__setitem__( 4, h.__getitem__( 4 ).unwrap( 'h has 5 entries' ) + e ).unwrap( 'idx < 5' )

			block += 64

	out = bytearray( usize( 20 ))
	out_ptr: Ptr[u8] = out.get_ptr()
	oi: usize = 0
	with compiler.wrap_arithmetic:
		while oi < 5:
			hv: u32 = h.__getitem__( oi ).unwrap( 'oi < 5' )
			ob: usize = oi * 4
			out_ptr[ob]     = u8( ( hv >> 24 ) & 0xFF )
			out_ptr[ob + 1] = u8( ( hv >> 16 ) & 0xFF )
			out_ptr[ob + 2] = u8( ( hv >> 8  ) & 0xFF )
			out_ptr[ob + 3] = u8(   hv         & 0xFF )
			oi += 1

	return bytes.from_bytearray( move( out ))
