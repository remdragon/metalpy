# sha512 - FIPS 180-4 SHA-512. One-shot digest: sha512(data) -> 64 raw digest
# bytes. Only building block Ed25519 needs (RFC 8032's own internal hashing) -
# no streaming update()/digest() split, mirrors sha256.py's own scope call.
#
# Message length is tracked as a plain u64 bit count (not the full 128-bit
# field FIPS 180-4 allows) - correct for any message under 2^61 bytes, i.e.
# every real caller; the high 64 bits of the length field are always written
# as zero.

import compiler


_K: list[u64] = [
	0x428a2f98d728ae22, 0x7137449123ef65cd, 0xb5c0fbcfec4d3b2f, 0xe9b5dba58189dbbc,
	0x3956c25bf348b538, 0x59f111f1b605d019, 0x923f82a4af194f9b, 0xab1c5ed5da6d8118,
	0xd807aa98a3030242, 0x12835b0145706fbe, 0x243185be4ee4b28c, 0x550c7dc3d5ffb4e2,
	0x72be5d74f27b896f, 0x80deb1fe3b1696b1, 0x9bdc06a725c71235, 0xc19bf174cf692694,
	0xe49b69c19ef14ad2, 0xefbe4786384f25e3, 0x0fc19dc68b8cd5b5, 0x240ca1cc77ac9c65,
	0x2de92c6f592b0275, 0x4a7484aa6ea6e483, 0x5cb0a9dcbd41fbd4, 0x76f988da831153b5,
	0x983e5152ee66dfab, 0xa831c66d2db43210, 0xb00327c898fb213f, 0xbf597fc7beef0ee4,
	0xc6e00bf33da88fc2, 0xd5a79147930aa725, 0x06ca6351e003826f, 0x142929670a0e6e70,
	0x27b70a8546d22ffc, 0x2e1b21385c26c926, 0x4d2c6dfc5ac42aed, 0x53380d139d95b3df,
	0x650a73548baf63de, 0x766a0abb3c77b2a8, 0x81c2c92e47edaee6, 0x92722c851482353b,
	0xa2bfe8a14cf10364, 0xa81a664bbc423001, 0xc24b8b70d0f89791, 0xc76c51a30654be30,
	0xd192e819d6ef5218, 0xd69906245565a910, 0xf40e35855771202a, 0x106aa07032bbd1b8,
	0x19a4c116b8d2d0c8, 0x1e376c085141ab53, 0x2748774cdf8eeb99, 0x34b0bcb5e19b48a8,
	0x391c0cb3c5c95a63, 0x4ed8aa4ae3418acb, 0x5b9cca4f7763e373, 0x682e6ff3d6b2b8a3,
	0x748f82ee5defb2fc, 0x78a5636f43172f60, 0x84c87814a1f0ab72, 0x8cc702081a6439ec,
	0x90befffa23631e28, 0xa4506cebde82bde9, 0xbef9a3f7b2c67915, 0xc67178f2e372532b,
	0xca273eceea26619c, 0xd186b8c721c0c207, 0xeada7dd6cde0eb1e, 0xf57d4f7fee6ed178,
	0x06f067aa72176fba, 0x0a637dc5a2c898a6, 0x113f9804bef90dae, 0x1b710b35131c471b,
	0x28db77f523047d84, 0x32caab7b40c72493, 0x3c9ebe0a15c9bebc, 0x431d67c49c100d4c,
	0x4cc5d4becb3e42b6, 0x597f299cfc657e2a, 0x5fcb6fab3ad6faec, 0x6c44198c4a475817,
]

_H0: list[u64] = [
	0x6a09e667f3bcc908, 0xbb67ae8584caa73b, 0x3c6ef372fe94f82b, 0xa54ff53a5f1d36f1,
	0x510e527fade682d1, 0x9b05688c2b3e6c1f, 0x1f83d9abfb41bd6b, 0x5be0cd19137e2179,
]


def _rotr( x: u64, n: u64 ) -> u64:
	with compiler.wrap_arithmetic:
		return ( x >> n ) | ( x << ( 64 - n ))


def sha512( data: bytes|bytearray ) -> bytes:
	n: usize = len( data )
	in_ptr: ConstPtr[u8] = data.get_const_ptr()

	# pad: msg || 0x80 || zero bytes || 16-byte big-endian bit length, total
	# a multiple of 128. pad_len is the smallest multiple of 128 >= n + 17.
	with compiler.panic_arithmetic( 'padded length is a small bounded function of input length' ):
		pad_len: usize = (( n + 17 + 127 ) // 128 ) * 128

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
		# high 8 bytes of the 128-bit length field (msg_ptr[length_at-8 .. length_at)) are already zero from bytearray()'s own zero-init

	h: UnsafeList[u64] = UnsafeList[u64]( usize( 8 ))
	hi: usize = 0
	with compiler.panic_arithmetic( 'hi < 8, cannot overflow' ):
		while hi < 8:
			h.append( _H0.__getitem__( hi ).unwrap( 'hi < 8' ))
			hi += 1

	block: usize = 0
	with compiler.wrap_arithmetic:
		while block < pad_len:
			# message schedule: 16 words straight from the block, then 64
			# more expanded per FIPS 180-4 SS6.4.2 step 1
			w: UnsafeList[u64] = UnsafeList[u64]( usize( 80 ))
			t: usize = 0
			while t < 16:
				bo: usize = block + t * 8
				w.append(
					( u64( msg_ptr[bo] )     << 56 ) |
					( u64( msg_ptr[bo + 1] ) << 48 ) |
					( u64( msg_ptr[bo + 2] ) << 40 ) |
					( u64( msg_ptr[bo + 3] ) << 32 ) |
					( u64( msg_ptr[bo + 4] ) << 24 ) |
					( u64( msg_ptr[bo + 5] ) << 16 ) |
					( u64( msg_ptr[bo + 6] ) << 8  ) |
					  u64( msg_ptr[bo + 7] )
				)
				t += 1
			while t < 80:
				wm15: u64 = w.__getitem__( t - 15 ).unwrap( 't >= 16' )
				wm2:  u64 = w.__getitem__( t - 2 ).unwrap( 't >= 16' )
				wm16: u64 = w.__getitem__( t - 16 ).unwrap( 't >= 16' )
				wm7:  u64 = w.__getitem__( t - 7 ).unwrap( 't >= 16' )
				s0: u64 = _rotr( wm15, 1 ) ^ _rotr( wm15, 8 ) ^ ( wm15 >> 7 )
				s1: u64 = _rotr( wm2, 19 ) ^ _rotr( wm2, 61 ) ^ ( wm2 >> 6 )
				w.append( wm16 + s0 + wm7 + s1 )
				t += 1

			a:  u64 = h.__getitem__( 0 ).unwrap( 'h has 8 entries' )
			b:  u64 = h.__getitem__( 1 ).unwrap( 'h has 8 entries' )
			c:  u64 = h.__getitem__( 2 ).unwrap( 'h has 8 entries' )
			d:  u64 = h.__getitem__( 3 ).unwrap( 'h has 8 entries' )
			e:  u64 = h.__getitem__( 4 ).unwrap( 'h has 8 entries' )
			f:  u64 = h.__getitem__( 5 ).unwrap( 'h has 8 entries' )
			g:  u64 = h.__getitem__( 6 ).unwrap( 'h has 8 entries' )
			hh: u64 = h.__getitem__( 7 ).unwrap( 'h has 8 entries' )

			rt: usize = 0
			while rt < 80:
				big_s1: u64 = _rotr( e, 14 ) ^ _rotr( e, 18 ) ^ _rotr( e, 41 )
				ch:     u64 = ( e & f ) ^ (( ~e ) & g )
				kt:     u64 = _K.__getitem__( rt ).unwrap( 'rt < 80' )
				wt:     u64 = w.__getitem__( rt ).unwrap( 'rt < 80' )
				temp1:  u64 = hh + big_s1 + ch + kt + wt
				big_s0: u64 = _rotr( a, 28 ) ^ _rotr( a, 34 ) ^ _rotr( a, 39 )
				maj:    u64 = ( a & b ) ^ ( a & c ) ^ ( b & c )
				temp2:  u64 = big_s0 + maj

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

			block += 128

	out = bytearray( usize( 64 ))
	out_ptr: Ptr[u8] = out.get_ptr()
	oi: usize = 0
	with compiler.wrap_arithmetic:
		while oi < 8:
			hv: u64 = h.__getitem__( oi ).unwrap( 'oi < 8' )
			ob: usize = oi * 8
			out_ptr[ob]     = u8( ( hv >> 56 ) & 0xFF )
			out_ptr[ob + 1] = u8( ( hv >> 48 ) & 0xFF )
			out_ptr[ob + 2] = u8( ( hv >> 40 ) & 0xFF )
			out_ptr[ob + 3] = u8( ( hv >> 32 ) & 0xFF )
			out_ptr[ob + 4] = u8( ( hv >> 24 ) & 0xFF )
			out_ptr[ob + 5] = u8( ( hv >> 16 ) & 0xFF )
			out_ptr[ob + 6] = u8( ( hv >> 8  ) & 0xFF )
			out_ptr[ob + 7] = u8(   hv         & 0xFF )
			oi += 1

	return bytes.from_bytearray( move( out ))
