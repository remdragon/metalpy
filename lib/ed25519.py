# ed25519 - RFC 8032 EdDSA over Curve25519 (keypair generation, sign, verify).
#
# Approach: implemented from scratch in MetalPy, not bound to a platform
# crypto library. Investigated both routes first:
#   - Linux: OpenSSL (libcrypto, already linked by lib/ssl.py) has clean
#     Ed25519 support (EVP_PKEY_new_raw_private_key + one-shot
#     EVP_DigestSign/EVP_DigestVerify).
#   - Windows: CNG's BCRYPT_ECC_CURVE_25519 only covers ECDH (curve25519,
#     used via BCryptSecretAgreement) - NOT EdDSA/Ed25519 signing. There is
#     no BCryptSignHash path that produces a real RFC 8032 signature.
#     Confirmed independently: .NET's own Ed25519 API proposal
#     (dotnet/runtime#63174) was still open, explicitly blocked on the same
#     underlying CNG gap - this isn't a one-off doc-reading miss, the
#     platform primitive genuinely doesn't exist yet.
# A binding-only implementation would therefore need a from-scratch fallback
# on Windows anyway (2 of this project's 3 real compilers), so the only way
# to get one real, uniformly-tested implementation across all three
# compilers is to implement Ed25519 directly and not depend on either OS's
# crypto stack at all. Trade-off: this is real, auditable-but-unaudited
# field/curve arithmetic - it has been checked against the official RFC 8032
# test vectors (see ed25519_test.py) but has not had the kind of side-channel
# review a production signing library would need (no attempt is made at
# constant-time execution - see _mod_pow's own note).
#
# No native 128-bit integer support to build on (u128/i128 silently fall back
# to 64-bit under MSVC - cl.exe has no __int128 - see lib/builtins/__int.py's
# own _to_i64 comment), so field/scalar elements are plain 4x64-bit-limb
# (256-bit) values with schoolbook multiply (portable 64x64->128 done by hand
# via 32-bit splitting - see _mul64) and generic bit-serial mod-reduction
# (binary long division, not a Solinas-style fast reduction) - simpler to get
# right than a curve25519-specific reduction trick, at the cost of speed
# nothing here actually needs (a handful of sign/verify calls, not a hot loop).
#
# Curve constants (d, the base point, sqrt(-1)) are deliberately NOT
# hardcoded as giant decimal magic numbers - they're derived at runtime from
# the small well-known integers RFC 8032 defines them from (d = -121665/121666,
# base point y = 4/5, sqrt(-1) = 2^((p-1)/4) mod p), reusing the same
# mod-inverse/mod-pow/point-decode machinery every other operation already
# exercises, rather than trusting a copy-pasted 77-digit constant.

import compiler
import sha512
import random
import sys


@enum( i32 )
class Ed25519Error:
	InvalidEncoding = 1
	InvalidSignature = 2
	RandomSourceFailed = 3
	Other = _


# --- portable 64x64->128 multiply (no u128 - see this file's header) -------

def _addc( a: u64, b: u64, carry_in: u64 ) -> tuple[u64, u64]:
	''' a + b + carry_in (carry_in is 0 or 1) -> (sum mod 2^64, carry_out).
	carry_out is always 0 or 1 for a valid carry_in - standard add-with-carry
	identity (three <2^64 terms sum to <2^65). '''
	with compiler.wrap_arithmetic:
		s1: u64 = a + b
		c1: u64 = u64( 1 ) if s1 < a else u64( 0 )
		s2: u64 = s1 + carry_in
		c2: u64 = u64( 1 ) if s2 < s1 else u64( 0 )
		return ( s2, c1 + c2 )


def _subb( a: u64, b: u64, borrow_in: u64 ) -> tuple[u64, u64]:
	''' a - b - borrow_in (borrow_in is 0 or 1) -> (diff mod 2^64, borrow_out). '''
	with compiler.wrap_arithmetic:
		d1: u64 = a - b
		b1: u64 = u64( 1 ) if a < b else u64( 0 )
		d2: u64 = d1 - borrow_in
		b2: u64 = u64( 1 ) if d1 < borrow_in else u64( 0 )
		return ( d2, b1 + b2 )


def _mul64( a: u64, b: u64 ) -> tuple[u64, u64]:
	''' a * b -> (hi, lo), the exact 128-bit product, built from four
	32x32->64 partial products (every real 64-bit target has native u64
	multiply, so only the WIDENING needs manual carry propagation). '''
	with compiler.wrap_arithmetic:
		a_lo: u64 = a & 0xFFFFFFFF
		a_hi: u64 = a >> 32
		b_lo: u64 = b & 0xFFFFFFFF
		b_hi: u64 = b >> 32

		lo_lo: u64 = a_lo * b_lo
		hi_lo: u64 = a_hi * b_lo
		lo_hi: u64 = a_lo * b_hi
		hi_hi: u64 = a_hi * b_hi

		cross:  u64 = hi_lo + ( lo_lo >> 32 )
		cross2: u64 = lo_hi + ( cross & 0xFFFFFFFF )

		result_lo: u64 = ( cross2 << 32 ) | ( lo_lo & 0xFFFFFFFF )
		result_hi: u64 = hi_hi + ( cross >> 32 ) + ( cross2 >> 32 )
		return ( result_hi, result_lo )


# --- fixed-width unsigned bignums -------------------------------------------
# _U256: 256 bits, little-endian limbs (limbs[0] = least significant).
# _U512: 512 bits, only ever a product (two _U256 factors) or a SHA-512
# digest being reduced - never itself a modulus.

@cstruct
class _U256:
	limbs: u64[4]

@cstruct
class _U512:
	limbs: u64[8]


def _u512_add_at( r: _U512, pos: usize, val: u64 ) -> _U512:
	''' r.limbs[pos] += val, propagating carry into higher limbs - the
	"accumulate one partial product word" primitive _u256_mul builds on.
	pos+carry-chain never reaches 8 for this file's own callers (a 256x256
	product's highest partial-product word lands at index 7). '''
	carry: u64 = val
	p: usize = pos
	with compiler.wrap_arithmetic:
		while carry != 0 and p < 8:
			( new_val, new_carry ) = _addc( r.limbs[p], carry, 0 )
			r.limbs[p] = new_val
			carry = new_carry
			p += 1
	return r


def _u256_zero() -> _U256:
	return _U256( limbs = 0 )

def _u256_small( v: u64 ) -> _U256:
	r: _U256 = _U256( limbs = 0 )
	r.limbs[0] = v
	return r

def _u256_is_zero( a: _U256 ) -> bool:
	return a.limbs[0] == 0 and a.limbs[1] == 0 and a.limbs[2] == 0 and a.limbs[3] == 0

def _u256_cmp( a: _U256, b: _U256 ) -> i32:
	i: usize = 4
	with compiler.wrap_arithmetic:
		while i > 0:
			i -= 1
			if a.limbs[i] != b.limbs[i]:
				return 1 if a.limbs[i] > b.limbs[i] else -1
	return 0

def _u256_bit( a: _U256, bit_index: usize ) -> u64:
	with compiler.wrap_arithmetic:
		limb: usize = bit_index >> 6
		off: u64 = u64( bit_index & 63 )
		return ( a.limbs[limb] >> off ) & 1

def _u256_add( a: _U256, b: _U256 ) -> _U256:
	''' (a + b) mod 2^256 - callers only ever use this on values already
	known to fit (field/scalar elements bounded well under 2^256). '''
	r: _U256 = _U256( limbs = 0 )
	carry: u64 = 0
	i: usize = 0
	with compiler.wrap_arithmetic:
		while i < 4:
			( s, new_carry ) = _addc( a.limbs[i], b.limbs[i], carry )
			carry = new_carry
			r.limbs[i] = s
			i += 1
	return r

def _u256_sub( a: _U256, b: _U256 ) -> _U256:
	''' (a - b) mod 2^256 - wraps on underflow, matching _u256_add. '''
	r: _U256 = _U256( limbs = 0 )
	borrow: u64 = 0
	i: usize = 0
	with compiler.wrap_arithmetic:
		while i < 4:
			( d, new_borrow ) = _subb( a.limbs[i], b.limbs[i], borrow )
			borrow = new_borrow
			r.limbs[i] = d
			i += 1
	return r

def _u256_from_bytes_le( buf: ConstPtr[u8] ) -> _U256:
	r: _U256 = _U256( limbs = 0 )
	limb: usize = 0
	with compiler.wrap_arithmetic:
		while limb < 4:
			v: u64 = 0
			byte: usize = 0
			while byte < 8:
				off: usize = limb * 8 + byte
				v = v | ( u64( buf[off] ) << u64( byte * 8 ))
				byte += 1
			r.limbs[limb] = v
			limb += 1
	return r

def _u256_to_bytes_le( a: _U256, out: Ptr[u8] ) -> None:
	limb: usize = 0
	with compiler.wrap_arithmetic:
		while limb < 4:
			v: u64 = a.limbs[limb]
			byte: usize = 0
			while byte < 8:
				off: usize = limb * 8 + byte
				out[off] = u8(( v >> u64( byte * 8 )) & 0xFF )
				byte += 1
			limb += 1

def _u256_from_decimal( s: str ) -> _U256:
	''' parses a plain (small, no sign) decimal literal into a _U256 -
	multiply-by-10-and-add-digit, same shape as lib/builtins/__int.py's own
	str->int accumulation. Only used for constants derived from RFC 8032
	text at module-load time, never on attacker-controlled input. '''
	r: _U256 = _u256_zero()
	ten: _U256 = _u256_small( 10 )
	cstr: ConstPtr[u8] = s.get_cstr()
	i: usize = 0
	n: usize = s.byte_len()
	with compiler.wrap_arithmetic:
		while i < n:
			c: u8 = cstr[i]
			digit: u64 = u64( c ) - u64( 0x30 )
			r = _u256_mul_small_add( r, ten, digit )
			i += 1
	return r

def _u256_mul_small_add( a: _U256, small: _U256, add: u64 ) -> _U256:
	''' a*small + add, truncated to 256 bits - small helper for
	_u256_from_decimal only (small is always 10, add always a single digit,
	so this never needs to be a full _U256 x _U256 multiply). '''
	prod: _U512 = _u256_mul( a, small )
	r: _U256 = _U256( limbs = 0 )
	i: usize = 0
	with compiler.wrap_arithmetic:
		while i < 4:
			r.limbs[i] = prod.limbs[i]
			i += 1
		( v0, carry ) = _addc( r.limbs[0], add, 0 )
		r.limbs[0] = v0
		i = 1
		while carry != 0 and i < 4:
			( vi, new_carry ) = _addc( r.limbs[i], 0, carry )
			r.limbs[i] = vi
			carry = new_carry
			i += 1
	return r


def _u256_mul( a: _U256, b: _U256 ) -> _U512:
	''' schoolbook 256x256->512 multiply: each of the 16 partial 64x64->128
	products is added into the result independently via _u512_add_at (its
	lo word at position i+j, its hi word at position i+j+1) - two separate
	carry-propagating adds per partial product rather than a single
	hand-rolled running-carry variable, since positions i+j+1 receive
	contributions from multiple (i,j) pairs and must simply be summed, not
	overwritten. '''
	r: _U512 = _U512( limbs = 0 )
	i: usize = 0
	with compiler.wrap_arithmetic:
		while i < 4:
			j: usize = 0
			while j < 4:
				( hi, lo ) = _mul64( a.limbs[i], b.limbs[j] )
				idx: usize = i + j
				r = _u512_add_at( r, idx, lo )
				r = _u512_add_at( r, idx + 1, hi )
				j += 1
			i += 1
	return r


# --- _U512 helpers (only used inside the generic mod-reduction below) ------

def _u512_zero() -> _U512:
	return _U512( limbs = 0 )

def _u512_from_u256( a: _U256 ) -> _U512:
	r: _U512 = _U512( limbs = 0 )
	i: usize = 0
	with compiler.wrap_arithmetic:
		while i < 4:
			r.limbs[i] = a.limbs[i]
			i += 1
	return r

def _u512_truncate( a: _U512 ) -> _U256:
	r: _U256 = _U256( limbs = 0 )
	i: usize = 0
	with compiler.wrap_arithmetic:
		while i < 4:
			r.limbs[i] = a.limbs[i]
			i += 1
	return r

def _u512_cmp( a: _U512, b: _U512 ) -> i32:
	i: usize = 8
	with compiler.wrap_arithmetic:
		while i > 0:
			i -= 1
			if a.limbs[i] != b.limbs[i]:
				return 1 if a.limbs[i] > b.limbs[i] else -1
	return 0

def _u512_sub( a: _U512, b: _U512 ) -> _U512:
	r: _U512 = _U512( limbs = 0 )
	borrow: u64 = 0
	i: usize = 0
	with compiler.wrap_arithmetic:
		while i < 8:
			( d, new_borrow ) = _subb( a.limbs[i], b.limbs[i], borrow )
			borrow = new_borrow
			r.limbs[i] = d
			i += 1
	return r

def _u512_shl1_or( a: _U512, bit: u64 ) -> _U512:
	''' (a << 1) | bit, truncated to 512 bits - the top bit dropped here is
	always known-zero (the remainder in _reduce512's binary long division
	never exceeds 2*modulus < 2^257, far under the 512-bit container). '''
	r: _U512 = _U512( limbs = 0 )
	carry: u64 = bit
	i: usize = 0
	with compiler.wrap_arithmetic:
		while i < 8:
			v: u64 = a.limbs[i]
			r.limbs[i] = ( v << 1 ) | carry
			carry = v >> 63
			i += 1
	return r

def _u512_bit( a: _U512, bit_index: usize ) -> u64:
	with compiler.wrap_arithmetic:
		limb: usize = bit_index >> 6
		off: u64 = u64( bit_index & 63 )
		return ( a.limbs[limb] >> off ) & 1


def _reduce512( x: _U512, m: _U256 ) -> _U256:
	''' x mod m, via bit-serial binary long division (512 shift/compare/
	subtract steps). Not the fastest possible reduction (a curve25519-
	specific Solinas trick would be far fewer steps) but the whole algorithm
	is 15 lines long and needs no per-modulus special-casing - m is either p
	or the group order L, both handled by the exact same code. Performance
	is a non-issue: a handful of these run per sign/verify call, not a hot
	loop. '''
	m_wide: _U512 = _u512_from_u256( m )
	remainder: _U512 = _u512_zero()
	i: usize = 512
	with compiler.wrap_arithmetic:
		while i > 0:
			i -= 1
			bit: u64 = _u512_bit( x, i )
			remainder = _u512_shl1_or( remainder, bit )
			if _u512_cmp( remainder, m_wide ) >= 0:
				remainder = _u512_sub( remainder, m_wide )
	return _u512_truncate( remainder )


def _mod_add( a: _U256, b: _U256, m: _U256 ) -> _U256:
	r: _U256 = _u256_add( a, b )
	# a,b < m < 2^256 so a+b < 2^257, but _u256_add only tracks 256 bits -
	# detect the dropped carry bit by noticing the wrapped sum is smaller
	# than either operand, same overflow tell _addc itself uses.
	overflowed: bool = _u256_cmp( r, a ) < 0
	if overflowed or _u256_cmp( r, m ) >= 0:
		r = _u256_sub( r, m )
	return r

def _mod_sub( a: _U256, b: _U256, m: _U256 ) -> _U256:
	if _u256_cmp( a, b ) >= 0:
		return _u256_sub( a, b )
	return _u256_sub( _u256_add( a, m ), b )

def _mod_mul( a: _U256, b: _U256, m: _U256 ) -> _U256:
	return _reduce512( _u256_mul( a, b ), m )

def _mod_pow( base: _U256, exp: _U256, m: _U256 ) -> _U256:
	''' base^exp mod m via square-and-multiply, MSB to LSB. Deliberately NOT
	constant-time (the multiply-by-base step is skipped on a zero exponent
	bit) - fine for this file's own use (field inversion/sqrt, both driven
	by the fixed public constant p, and per-signature scalar k which is
	itself a public hash output, never a secret) but this is NOT a general-
	purpose constant-time modpow; don't reuse it for a secret exponent. '''
	result: _U256 = _u256_small( 1 )
	b: _U256 = _mod_reduce_once( base, m )
	i: usize = 256
	with compiler.wrap_arithmetic:
		while i > 0:
			i -= 1
			result = _mod_mul( result, result, m )
			if _u256_bit( exp, i ) != 0:
				result = _mod_mul( result, b, m )
	return result

def _mod_reduce_once( a: _U256, m: _U256 ) -> _U256:
	if _u256_cmp( a, m ) < 0:
		return a
	return _reduce512( _u512_from_u256( a ), m )


# --- field (mod p = 2^255 - 19) and scalar (mod L, base-point order) -------

def _p() -> _U256:
	''' p = 2^255 - 19 (bit 255 set, i.e. limbs[3] bit 63). '''
	r: _U256 = _u256_zero()
	with compiler.wrap_arithmetic:
		r.limbs[3] = u64( 1 ) << 63
	return _u256_sub( r, _u256_small( 19 ))

_P: _U256 = _p()

def _l() -> _U256:
	two_252: _U256 = _u256_zero()
	with compiler.wrap_arithmetic:
		two_252.limbs[3] = u64( 1 ) << 60
	addend: _U256 = _u256_from_decimal( '27742317777372353535851937790883648493' )
	return _u256_add( two_252, addend )

_L: _U256 = _l()

def _curve_d() -> _U256:
	''' d = -121665/121666 mod p, per RFC 8032 SS5.1. '''
	num: _U256 = _mod_sub( _P, _u256_small( 121665 ), _P )  # -121665 mod p
	den: _U256 = _u256_small( 121666 )
	inv_den: _U256 = _mod_pow( den, _u256_sub( _P, _u256_small( 2 )), _P )  # Fermat inverse
	return _mod_mul( num, inv_den, _P )

_D: _U256 = _curve_d()

def _sqrt_m1() -> _U256:
	''' 2^((p-1)/4) mod p - a square root of -1 mod p (exists since p = 5
	mod 8), needed by point decompression's sqrt step. '''
	return _mod_pow( _u256_small( 2 ), _sqrt_m1_exponent(), _P )

def _sqrt_m1_exponent() -> _U256:
	''' (p-1)/4, computed properly (not approximated) - p-1 = 2^255-20,
	divided by 4 is 2^253-5. '''
	p_minus_1: _U256 = _u256_sub( _P, _u256_small( 1 ))
	# divide by 4: shift right 2. p_minus_1's bottom 2 bits are 0 (p is odd,
	# p-1 even; p == 5 mod 8 makes p-1 == 4 mod 8, so p-1 is divisible by 4
	# exactly) so this is an exact shift, no remainder lost.
	r: _U256 = _u256_zero()
	borrow_bits: u64 = 0
	j: usize = 4
	with compiler.wrap_arithmetic:
		while j > 0:
			j -= 1
			v: u64 = p_minus_1.limbs[j]
			r.limbs[j] = ( v >> 2 ) | ( borrow_bits << 62 )
			borrow_bits = v & 0x3
	return r

_SQRT_M1: _U256 = _sqrt_m1()


def _f_add( a: _U256, b: _U256 ) -> _U256:
	return _mod_add( a, b, _P )

def _f_sub( a: _U256, b: _U256 ) -> _U256:
	return _mod_sub( a, b, _P )

def _f_mul( a: _U256, b: _U256 ) -> _U256:
	return _mod_mul( a, b, _P )

def _f_sqr( a: _U256 ) -> _U256:
	return _mod_mul( a, a, _P )

def _f_neg( a: _U256 ) -> _U256:
	return _mod_sub( _P, a, _P ) if not _u256_is_zero( a ) else a

def _f_inv( a: _U256 ) -> _U256:
	return _mod_pow( a, _u256_sub( _P, _u256_small( 2 )), _P )


def _s_reduce512( x: _U512 ) -> _U256:
	return _reduce512( x, _L )

def _s_add( a: _U256, b: _U256 ) -> _U256:
	return _mod_add( a, b, _L )

def _s_mul( a: _U256, b: _U256 ) -> _U256:
	return _mod_mul( a, b, _L )


# --- extended twisted Edwards curve points ----------------------------------
# -x^2 + y^2 = 1 + d*x^2*y^2 (mod p), a = -1. (X,Y,Z,T) with x=X/Z, y=Y/Z,
# x*y=T/Z. Formulas are add-2008-hwcd-3 (unified, complete for a=-1) and
# dbl-2008-hwcd from the Explicit-Formulas Database - the standard formula
# set behind essentially every Ed25519 implementation.

@cstruct
class _Point:
	x: _U256
	y: _U256
	z: _U256
	t: _U256


def _point_identity() -> _Point:
	return _Point( x = _u256_zero(), y = _u256_small( 1 ), z = _u256_small( 1 ), t = _u256_zero() )


def _point_double( p: _Point ) -> _Point:
	a: _U256 = _f_sqr( p.x )
	b: _U256 = _f_sqr( p.y )
	c: _U256 = _f_add( _f_sqr( p.z ), _f_sqr( p.z ))
	d: _U256 = _f_neg( a )
	xy_sum: _U256 = _f_add( p.x, p.y )
	e: _U256 = _f_sub( _f_sub( _f_sqr( xy_sum ), a ), b )
	g: _U256 = _f_add( d, b )
	f: _U256 = _f_sub( g, c )
	h: _U256 = _f_sub( d, b )
	return _Point(
		x = _f_mul( e, f ),
		y = _f_mul( g, h ),
		z = _f_mul( f, g ),
		t = _f_mul( e, h ),
	)


def _point_add( p1: _Point, p2: _Point ) -> _Point:
	a: _U256 = _f_mul( _f_sub( p1.y, p1.x ), _f_sub( p2.y, p2.x ))
	b: _U256 = _f_mul( _f_add( p1.y, p1.x ), _f_add( p2.y, p2.x ))
	two_d: _U256 = _f_add( _D, _D )
	c: _U256 = _f_mul( _f_mul( p1.t, two_d ), p2.t )
	d: _U256 = _f_mul( _f_add( p1.z, p1.z ), p2.z )
	e: _U256 = _f_sub( b, a )
	f: _U256 = _f_sub( d, c )
	g: _U256 = _f_add( d, c )
	h: _U256 = _f_add( b, a )
	return _Point(
		x = _f_mul( e, f ),
		y = _f_mul( g, h ),
		z = _f_mul( f, g ),
		t = _f_mul( e, h ),
	)


def _point_eq( p1: _Point, p2: _Point ) -> bool:
	''' affine equality via cross-multiplication - avoids inversion. '''
	lhs_x: _U256 = _f_mul( p1.x, p2.z )
	rhs_x: _U256 = _f_mul( p2.x, p1.z )
	lhs_y: _U256 = _f_mul( p1.y, p2.z )
	rhs_y: _U256 = _f_mul( p2.y, p1.z )
	return _u256_cmp( lhs_x, rhs_x ) == 0 and _u256_cmp( lhs_y, rhs_y ) == 0


def _scalar_mult( k: _U256, base: _Point ) -> _Point:
	''' double-and-add, MSB to LSB - not constant-time (see _mod_pow's own
	note; same posture applies here). '''
	result: _Point = _point_identity()
	i: usize = 256
	with compiler.wrap_arithmetic:
		while i > 0:
			i -= 1
			result = _point_double( result )
			if _u256_bit( k, i ) != 0:
				result = _point_add( result, base )
	return result


def _point_affine_xy( p: _Point ) -> tuple[_U256, _U256]:
	z_inv: _U256 = _f_inv( p.z )
	return ( _f_mul( p.x, z_inv ), _f_mul( p.y, z_inv ))


def _point_compress( p: _Point ) -> bytes:
	( x, y ) = _point_affine_xy( p )
	out: bytearray = bytearray( usize( 32 ))
	_u256_to_bytes_le( y, out.get_ptr() )
	with compiler.wrap_arithmetic:
		sign_bit: u8 = u8( _u256_bit( x, 0 ))
		last: u8 = out.__getitem__( 31 ).unwrap( 'out has 32 bytes' )
		out[31] = last | ( sign_bit << 7 )
	return bytes.from_bytearray( move( out ))


def _point_from_y( y: _U256, sign_bit: u8 ) -> Result[_Point, Ed25519Error]:
	''' RFC 8032 SS5.1.3 point decompression, given y already parsed (top
	bit of the encoding stripped into `sign_bit`). '''
	if _u256_cmp( y, _P ) >= 0:
		return Result.Err( Ed25519Error.InvalidEncoding )  # non-canonical y
	y2: _U256 = _f_sqr( y )
	u: _U256 = _f_sub( y2, _u256_small( 1 ))
	v: _U256 = _f_add( _f_mul( _D, y2 ), _u256_small( 1 ))
	v_inv: _U256 = _f_inv( v )
	uv: _U256 = _f_mul( u, v_inv )
	# candidate = uv^((p+3)/8) mod p
	x: _U256 = _mod_pow( uv, _decode_exponent(), _P )
	x2: _U256 = _f_sqr( x )
	if _u256_cmp( x2, uv ) != 0:
		if _u256_cmp( x2, _f_neg( uv )) == 0:
			x = _f_mul( x, _SQRT_M1 )
		else:
			return Result.Err( Ed25519Error.InvalidEncoding )
	if _u256_is_zero( x ) and sign_bit != 0:
		return Result.Err( Ed25519Error.InvalidEncoding )
	if u64( _u256_bit( x, 0 )) != u64( sign_bit ):
		x = _f_neg( x )
	return Result.Ok( _Point( x = x, y = y, z = _u256_small( 1 ), t = _f_mul( x, y )))


def _decode_exponent() -> _U256:
	''' (p+3)/8 - the exponent point decompression's sqrt step uses. '''
	p_plus_3: _U256 = _u256_add( _P, _u256_small( 3 ))
	# divide by 8: p == 5 mod 8, so p+3 == 0 mod 8 - exact shift.
	r: _U256 = _u256_zero()
	borrow_bits: u64 = 0
	j: usize = 4
	with compiler.wrap_arithmetic:
		while j > 0:
			j -= 1
			v: u64 = p_plus_3.limbs[j]
			r.limbs[j] = ( v >> 3 ) | ( borrow_bits << 61 )
			borrow_bits = v & 0x7
	return r


def _point_decompress( encoded: ConstPtr[u8] ) -> Result[_Point, Ed25519Error]:
	sign_bit: u8 = ( encoded[31] >> 7 ) & 1
	y_bytes: bytearray = bytearray( usize( 32 ))
	y_ptr: Ptr[u8] = y_bytes.get_ptr()
	i: usize = 0
	with compiler.wrap_arithmetic:
		while i < 32:
			y_ptr[i] = encoded[i]
			i += 1
		last: u8 = y_ptr[31] & 0x7F
		y_ptr[31] = last
	y: _U256 = _u256_from_bytes_le( y_bytes.get_const_ptr() )
	return _point_from_y( y, sign_bit )


def _base_point() -> _Point:
	''' RFC 8032's base point B: y = 4/5 mod p, x recovered via the same
	decompression formula as any other point (sign bit 0, matching the
	standard base point's known-even x). '''
	y: _U256 = _f_mul( _u256_small( 4 ), _f_inv( _u256_small( 5 )))
	match _point_from_y( y, 0 ):
		case Result.Ok( p ):
			return p
		case Result.Err( _ ):
			sys.panic( 'base point derivation failed - curve constants are wrong' )

_B: _Point = _base_point()


# --- public API --------------------------------------------------------------

def _clamp( a_bytes: Ptr[u8] ) -> None:
	with compiler.wrap_arithmetic:
		b0: u8 = a_bytes[0] & 0xF8
		a_bytes[0] = b0
		b31: u8 = ( a_bytes[31] & 0x7F ) | 0x40
		a_bytes[31] = b31


def _hash_to_scalar2( a: ConstPtr[u8], a_len: usize, b: ConstPtr[u8], b_len: usize ) -> _U256:
	''' SHA-512(a || b) mod L - the r = H(prefix || M) step. '''
	buf: bytearray
	with compiler.wrap_arithmetic:
		total: usize = a_len + b_len
		buf = bytearray( total )
		buf_ptr: Ptr[u8] = buf.get_ptr()
		i: usize = 0
		while i < a_len:
			buf_ptr[i] = a[i]
			i += 1
		i = 0
		while i < b_len:
			buf_ptr[a_len + i] = b[i]
			i += 1
	return _hash_bytes_to_scalar( move( buf ))


def _hash_to_scalar3( a: ConstPtr[u8], a_len: usize, b: ConstPtr[u8], b_len: usize, c: ConstPtr[u8], c_len: usize ) -> _U256:
	''' SHA-512(a || b || c) mod L - the k = H(R || A || M) step. '''
	buf: bytearray
	with compiler.wrap_arithmetic:
		total: usize = a_len + b_len + c_len
		buf = bytearray( total )
		buf_ptr: Ptr[u8] = buf.get_ptr()
		i: usize = 0
		while i < a_len:
			buf_ptr[i] = a[i]
			i += 1
		i = 0
		while i < b_len:
			buf_ptr[a_len + i] = b[i]
			i += 1
		i = 0
		ab_len: usize = a_len + b_len
		while i < c_len:
			buf_ptr[ab_len + i] = c[i]
			i += 1
	return _hash_bytes_to_scalar( move( buf ))


def _hash_bytes_to_scalar( buf: move[bytearray] ) -> _U256:
	digest: bytes = sha512.sha512( bytes.from_bytearray( move( buf )))
	digest_u512: _U512 = _u512_from_bytes_le( digest.get_const_ptr() )
	return _s_reduce512( digest_u512 )


def _u512_from_bytes_le( buf: ConstPtr[u8] ) -> _U512:
	r: _U512 = _U512( limbs = 0 )
	limb: usize = 0
	with compiler.wrap_arithmetic:
		while limb < 8:
			v: u64 = 0
			byte: usize = 0
			while byte < 8:
				off: usize = limb * 8 + byte
				v = v | ( u64( buf[off] ) << u64( byte * 8 ))
				byte += 1
			r.limbs[limb] = v
			limb += 1
	return r


def generate_keypair() -> Result[tuple[bytes, bytes], Ed25519Error]:
	''' -> (public_key, private_key). private_key is the raw 32-byte seed
	(RFC 8032's own "private key" encoding - not the expanded/clamped
	scalar); sign() re-derives everything from it each call, matching
	RFC 8032 SS5.1.5/SS5.1.6 exactly. '''
	seed: bytearray
	match random.random_bytes( usize( 32 )):
		case Result.Ok( s ):
			seed = s
		case Result.Err( _ ):
			return Result.Err( Ed25519Error.RandomSourceFailed )
	seed_bytes: bytes = bytes.from_bytearray( move( seed ))
	pub: bytes = public_key_from_seed( seed_bytes )
	return Result.Ok(( pub, seed_bytes ))


def public_key_from_seed( seed: bytes ) -> bytes:
	''' re-derives the 32-byte public key from a 32-byte private key seed -
	RFC 8032 SS5.1.5, exposed separately from generate_keypair() for a
	caller that only has a stored seed (or, as here, the official RFC 8032
	test vectors, which give a fixed seed rather than letting one be drawn
	from the CSPRNG). '''
	h: bytes = sha512.sha512( seed )
	a_bytes: bytearray = bytearray( usize( 32 ))
	a_ptr: Ptr[u8] = a_bytes.get_ptr()
	i: usize = 0
	with compiler.wrap_arithmetic:
		while i < 32:
			a_ptr[i] = h.__getitem__( i ).unwrap( 'sha512 digest is 64 bytes' )
			i += 1
	_clamp( a_ptr )
	a: _U256 = _u256_from_bytes_le( a_bytes.get_const_ptr() )
	pub_point: _Point = _scalar_mult( a, _B )
	return _point_compress( pub_point )


def sign( private_key: bytes, message: bytes ) -> Result[bytes, Ed25519Error]:
	if len( private_key ) != usize( 32 ):
		return Result.Err( Ed25519Error.InvalidEncoding )
	h: bytes = sha512.sha512( private_key )
	a_bytes: bytearray = bytearray( usize( 32 ))
	a_ptr: Ptr[u8] = a_bytes.get_ptr()
	prefix: bytearray = bytearray( usize( 32 ))
	prefix_ptr: Ptr[u8] = prefix.get_ptr()
	i: usize = 0
	with compiler.wrap_arithmetic:
		while i < 32:
			a_ptr[i] = h.__getitem__( i ).unwrap( 'sha512 digest is 64 bytes' )
			prefix_ptr[i] = h.__getitem__( i + 32 ).unwrap( 'sha512 digest is 64 bytes' )
			i += 1
	_clamp( a_ptr )
	a: _U256 = _u256_from_bytes_le( a_bytes.get_const_ptr() )
	a_point: _Point = _scalar_mult( a, _B )
	a_encoded: bytes = _point_compress( a_point )

	msg_ptr: ConstPtr[u8] = message.get_const_ptr()
	msg_len: usize = len( message )
	r: _U256 = _hash_to_scalar2( prefix.get_const_ptr(), usize( 32 ), msg_ptr, msg_len )
	r_point: _Point = _scalar_mult( r, _B )
	r_encoded: bytes = _point_compress( r_point )

	k: _U256 = _hash_to_scalar3( r_encoded.get_const_ptr(), usize( 32 ), a_encoded.get_const_ptr(), usize( 32 ), msg_ptr, msg_len )
	s: _U256 = _s_add( r, _s_mul( k, a ))

	out: bytearray = bytearray( usize( 64 ))
	out_ptr: Ptr[u8] = out.get_ptr()
	with compiler.wrap_arithmetic:
		i = 0
		while i < 32:
			out_ptr[i] = r_encoded.__getitem__( i ).unwrap( 'r_encoded is 32 bytes' )
			i += 1
		s_dest: Ptr[u8] = out_ptr + 32
	_u256_to_bytes_le( s, s_dest )
	return Result.Ok( bytes.from_bytearray( move( out )))


def verify( public_key: bytes, message: bytes, signature: bytes ) -> bool:
	''' plain bool, not Result - every failure mode here (bad encoding, bad
	signature, non-canonical S) means exactly one thing to a caller
	authenticating a peer: reject. There's no separate "couldn't attempt
	verification" case worth distinguishing (unlike sign(), this never
	touches the CSPRNG or any other fallible OS resource). '''
	if len( public_key ) != usize( 32 ) or len( signature ) != usize( 64 ):
		return False
	sig_ptr: ConstPtr[u8] = signature.get_const_ptr()
	with compiler.wrap_arithmetic:
		s_src: ConstPtr[u8] = sig_ptr + 32
	s: _U256 = _u256_from_bytes_le( s_src )
	if _u256_cmp( s, _L ) >= 0:
		return False  # non-canonical S - RFC 8032 SS5.1.7's recommended check

	a_point: _Point
	match _point_decompress( public_key.get_const_ptr() ):
		case Result.Ok( p ):
			a_point = p
		case Result.Err( _ ):
			return False
	r_point: _Point
	match _point_decompress( sig_ptr ):
		case Result.Ok( p ):
			r_point = p
		case Result.Err( _ ):
			return False

	msg_ptr: ConstPtr[u8] = message.get_const_ptr()
	k: _U256 = _hash_to_scalar3( sig_ptr, usize( 32 ), public_key.get_const_ptr(), usize( 32 ), msg_ptr, len( message ))

	lhs: _Point = _scalar_mult( s, _B )
	rhs: _Point = _point_add( r_point, _scalar_mult( k, a_point ))
	return _point_eq( lhs, rhs )
