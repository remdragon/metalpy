# sha256 - FIPS 180-4 SHA-256. One-shot digest: sha256(data) -> 32 raw
# digest bytes. No streaming update()/digest() split - the only known
# caller hashes small whole buffers (enrollment tokens); add streaming
# later if a large-file/incremental need shows up.

import compiler


_K: list[u32] = [
	0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5,
	0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174,
	0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,
	0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7, 0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967,
	0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85,
	0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
	0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
	0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2,
]

_H0: list[u32] = [
	0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a,
	0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19,
]


def _rotr( x: u32, n: u32 ) -> u32:
	with compiler.wrap_arithmetic:
		return ( x >> n ) | ( x << ( 32 - n ))


def sha256( data: bytes|bytearray ) -> bytes:
	n: usize = len( data )
	in_ptr: ConstPtr[u8] = data.get_const_ptr()

	# pad: msg || 0x80 || zero bytes || 8-byte big-endian bit length, total
	# a multiple of 64. pad_len is the smallest multiple of 64 >= n + 9.
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

	h: UnsafeList[u32] = UnsafeList[u32]( usize( 8 ))
	hi: usize = 0
	with compiler.panic_arithmetic( 'hi < 8, cannot overflow' ):
		while hi < 8:
			h.append( _H0.__getitem__( hi ).unwrap( 'hi < 8' ))
			hi += 1

	block: usize = 0
	with compiler.wrap_arithmetic:
		while block < pad_len:
			# message schedule: 16 words straight from the block, then 48
			# more expanded per FIPS 180-4 SS6.2.2 step 1
			w: UnsafeList[u32] = UnsafeList[u32]( usize( 64 ))
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
			while t < 64:
				wm15: u32 = w.__getitem__( t - 15 ).unwrap( 't >= 16' )
				wm2:  u32 = w.__getitem__( t - 2 ).unwrap( 't >= 16' )
				wm16: u32 = w.__getitem__( t - 16 ).unwrap( 't >= 16' )
				wm7:  u32 = w.__getitem__( t - 7 ).unwrap( 't >= 16' )
				s0: u32 = _rotr( wm15, 7 ) ^ _rotr( wm15, 18 ) ^ ( wm15 >> 3 )
				s1: u32 = _rotr( wm2, 17 ) ^ _rotr( wm2, 19 ) ^ ( wm2 >> 10 )
				w.append( wm16 + s0 + wm7 + s1 )
				t += 1

			a:  u32 = h.__getitem__( 0 ).unwrap( 'h has 8 entries' )
			b:  u32 = h.__getitem__( 1 ).unwrap( 'h has 8 entries' )
			c:  u32 = h.__getitem__( 2 ).unwrap( 'h has 8 entries' )
			d:  u32 = h.__getitem__( 3 ).unwrap( 'h has 8 entries' )
			e:  u32 = h.__getitem__( 4 ).unwrap( 'h has 8 entries' )
			f:  u32 = h.__getitem__( 5 ).unwrap( 'h has 8 entries' )
			g:  u32 = h.__getitem__( 6 ).unwrap( 'h has 8 entries' )
			hh: u32 = h.__getitem__( 7 ).unwrap( 'h has 8 entries' )

			rt: usize = 0
			while rt < 64:
				big_s1: u32 = _rotr( e, 6 ) ^ _rotr( e, 11 ) ^ _rotr( e, 25 )
				ch:     u32 = ( e & f ) ^ (( ~e ) & g )
				kt:     u32 = _K.__getitem__( rt ).unwrap( 'rt < 64' )
				wt:     u32 = w.__getitem__( rt ).unwrap( 'rt < 64' )
				temp1:  u32 = hh + big_s1 + ch + kt + wt
				big_s0: u32 = _rotr( a, 2 ) ^ _rotr( a, 13 ) ^ _rotr( a, 22 )
				maj:    u32 = ( a & b ) ^ ( a & c ) ^ ( b & c )
				temp2:  u32 = big_s0 + maj

				hh = g
				g = f
				f = e
				e = d + temp1
				d = c
				c = b
				b = a
				a = temp1 + temp2
				rt += 1

			h.__setitem__( 0, h.__getitem__( 0 ).unwrap( 'h has 8 entries' ) + a ).unwrap( 'idx < 8' )
			h.__setitem__( 1, h.__getitem__( 1 ).unwrap( 'h has 8 entries' ) + b ).unwrap( 'idx < 8' )
			h.__setitem__( 2, h.__getitem__( 2 ).unwrap( 'h has 8 entries' ) + c ).unwrap( 'idx < 8' )
			h.__setitem__( 3, h.__getitem__( 3 ).unwrap( 'h has 8 entries' ) + d ).unwrap( 'idx < 8' )
			h.__setitem__( 4, h.__getitem__( 4 ).unwrap( 'h has 8 entries' ) + e ).unwrap( 'idx < 8' )
			h.__setitem__( 5, h.__getitem__( 5 ).unwrap( 'h has 8 entries' ) + f ).unwrap( 'idx < 8' )
			h.__setitem__( 6, h.__getitem__( 6 ).unwrap( 'h has 8 entries' ) + g ).unwrap( 'idx < 8' )
			h.__setitem__( 7, h.__getitem__( 7 ).unwrap( 'h has 8 entries' ) + hh ).unwrap( 'idx < 8' )

			block += 64

	out = bytearray( usize( 32 ))
	out_ptr: Ptr[u8] = out.get_ptr()
	oi: usize = 0
	with compiler.wrap_arithmetic:
		while oi < 8:
			hv: u32 = h.__getitem__( oi ).unwrap( 'oi < 8' )
			ob: usize = oi * 4
			out_ptr[ob]     = u8( ( hv >> 24 ) & 0xFF )
			out_ptr[ob + 1] = u8( ( hv >> 16 ) & 0xFF )
			out_ptr[ob + 2] = u8( ( hv >> 8  ) & 0xFF )
			out_ptr[ob + 3] = u8(   hv         & 0xFF )
			oi += 1

	return bytes.from_bytearray( move( out ))
