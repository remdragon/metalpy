'''
CSPRNG - cryptographically secure random bytes.

Windows: BCryptGenRandom (bcrypt.dll), hAlgorithm=NULL +
	BCRYPT_USE_SYSTEM_PREFERRED_RNG - no algorithm handle to open/close,
	Microsoft's own recommended shape for a one-off random fill.
Linux: getrandom(2) as the primary path (has_library-gated - glibc >= 2.25;
	older glibc/musl builds without the symbol fall back to reading
	/dev/urandom instead, same trust level).
macOS: not implemented - poison pill, same convention as lib/ssl.py's own
	macOS section (a real def, not an absent one, so anything that merely
	imports this module still compiles on macOS as long as it never calls in).
'''

import compiler
import fs
import sys
import threading

if compiler.target.os == 'windows':
	BCRYPT_USE_SYSTEM_PREFERRED_RNG: u32 = 0x00000002


@compiler.target( os = 'windows' )
@extern( 'bcrypt', 'BCryptGenRandom' )
def BCryptGenRandom(
	hAlgorithm: Ptr[None],
	pbBuffer:   Ptr[u8],
	cbBuffer:   u32,
	dwFlags:    u32,
) -> i32:  # NTSTATUS
	...

@compiler.target( os = 'windows' )
def _fill_random_raw( buf: Ptr[u8], count: usize ) -> Result[usize, OSError]:
	with compiler.saturate_arithmetic:
		to_fill: u32 = u32( count )
	status: i32 = BCryptGenRandom( None, buf, to_fill, BCRYPT_USE_SYSTEM_PREFERRED_RNG )
	if status != 0:  # STATUS_SUCCESS
		return Result.Err( OSError.Other )  # NTSTATUS isn't a Win32 error code, nothing better to map to
	with compiler.wrap_arithmetic:
		return Result.Ok( usize( to_fill ))


@compiler.target( os = not ( 'windows', 'macos' ), has_library = ( 'c', 'getrandom' ) )
@extern( 'c', 'getrandom' )
def getrandom( buf: Ptr[None], buflen: usize, flags: u32 ) -> isize:
	...

@compiler.target( os = not ( 'windows', 'macos' ), has_library = ( 'c', 'getrandom' ) )
def _fill_random_raw( buf: Ptr[u8], count: usize ) -> Result[usize, OSError]:
	''' getrandom(2) can do a short read for count > 256 if interrupted -
	caller (fill_random) loops. EINTR itself is retried here rather than
	surfaced, matching write_all/read_exact's own "keep going" posture. '''
	from crt import get_errno
	while True:
		n: isize = getrandom( compiler.cast( Ptr[None], buf ), count, u32( 0 ))
		if n < isize( 0 ):
			err: OSError = OSError( get_errno() )
			if err == OSError.Interrupted:
				continue
			return Result.Err( err )
		with compiler.wrap_arithmetic:
			return Result.Ok( usize( n ))

# no getrandom symbol - fall back to reading /dev/urandom directly.
@compiler.target( os = not ( 'windows', 'macos' ), has_library = not ( 'c', 'getrandom' ) )
def _fill_random_raw( buf: Ptr[u8], count: usize ) -> Result[usize, OSError]:
	path: str = "/dev/urandom"
	fd: fs.FD = fs.open_raw( path.get_cstr(), fs.O_RDONLY, 0 ).or_return()
	result: Result[usize, OSError] = fs.read_raw( fd, buf, count )
	fs.close_raw( fd ).is_ok()  # best-effort - a read that already succeeded still counts
	return result


@compiler.target( os = 'macos' )
def _fill_random_raw( buf: Ptr[u8], count: usize ) -> Result[usize, OSError]:
	return _MACOS_RANDOM_NOT_YET_IMPLEMENTED()


def fill_random( buf: Ptr[u8], count: usize ) -> Result[None, OSError]:
	''' fills buf[0:count) with cryptographically secure random bytes,
	looping the OS primitive since a single call isn't guaranteed to fill
	an arbitrarily large buffer in one shot. '''
	filled: usize = 0
	with compiler.panic_arithmetic( 'bounded by count, cannot overflow' ):
		while filled < count:
			with compiler.wrap_arithmetic:
				dest: Ptr[u8] = buf + filled
				remaining: usize = count - filled
			n: usize = _fill_random_raw( dest, remaining ).or_return()
			if n == 0:
				return Result.Err( OSError.Other )
			with compiler.wrap_arithmetic:
				filled += n
	return Result.Ok( None )


def random_bytes( count: usize ) -> Result[bytearray, OSError]:
	''' count cryptographically secure random bytes, as a fresh bytearray. '''
	out: bytearray = bytearray( count )
	fill_random( out.get_ptr(), count ).or_return()
	return Result.Ok( out )


# ---------------------------------------------------------------------------
# Random: a seedable, deterministic, NON-cryptographic PRNG - xoshiro256**
# (Blackman/Vigna), seeded via SplitMix64 (the algorithm's own recommended
# seed expansion, avoiding an all-zero or low-entropy initial state). Not
# CSPRNG-backed by default - reproducibility (same seed -> same sequence,
# for simulations/tests) is the point; a caller that needs cryptographic
# randomness wants random_bytes()/fill_random() above instead.
# ---------------------------------------------------------------------------

def _rotl64( x: u64, k: u64 ) -> u64:
	with compiler.wrap_arithmetic:
		return ( x << k ) | ( x >> ( u64( 64 ) - k ))


def _expand_seed( resolved: u64 ) -> UnsafeList[u64]:
	''' SplitMix64: state advances by the golden-ratio increment before each
	mix step, expanding one u64 seed into 4 well-distributed xoshiro256**
	state words (an all-zero state is xoshiro256**'s one forbidden input -
	SplitMix64 output is never all-zero for any of the 2^64 possible seeds,
	by construction). A free function (not inlined into __init__/seed()
	directly) since __init__ can't call a method on self before its own
	fields are initialized - both call this and assign its 4 results. '''
	with compiler.wrap_arithmetic:
		out: UnsafeList[u64] = UnsafeList[u64]( usize( 4 ))
		s: u64 = resolved
		s += u64( 0x9E3779B97F4A7C15 ); out.append( _splitmix64_mix( s ))
		s += u64( 0x9E3779B97F4A7C15 ); out.append( _splitmix64_mix( s ))
		s += u64( 0x9E3779B97F4A7C15 ); out.append( _splitmix64_mix( s ))
		s += u64( 0x9E3779B97F4A7C15 ); out.append( _splitmix64_mix( s ))
		return out


def _resolve_seed( value: u64|None ) -> u64:
	''' value=None draws a fresh seed from the CSPRNG above - same "reseed
	from real entropy" convention as Python's own random.seed(None). '''
	if value is None:
		drawn: bytearray = random_bytes( usize( 8 )).unwrap( 'Random.seed: CSPRNG draw' )
		return _u64_from_bytes( drawn )
	return value


def _splitmix64_mix( z: u64 ) -> u64:
	''' SplitMix64's own output mixing step (not the state increment - see
	Random.seed()'s own call site for that half). '''
	with compiler.wrap_arithmetic:
		m: u64 = z
		m = ( m ^ ( m >> 30 )) * u64( 0xBF58476D1CE4E5B9 )
		m = ( m ^ ( m >> 27 )) * u64( 0x94D049BB133111EB )
		return m ^ ( m >> 31 )


def _u64_from_bytes( b: bytearray ) -> u64:
	''' combines 8 raw bytes into a u64 - endianness doesn't matter here
	(only used to turn a CSPRNG draw into a seed), same byte-combining shape
	as lib/sha1.py's own message-word assembly. '''
	with compiler.wrap_arithmetic:
		return (
			( u64( b.__getitem__( 0 ).unwrap( 'b has 8 entries' )) << 56 ) |
			( u64( b.__getitem__( 1 ).unwrap( 'b has 8 entries' )) << 48 ) |
			( u64( b.__getitem__( 2 ).unwrap( 'b has 8 entries' )) << 40 ) |
			( u64( b.__getitem__( 3 ).unwrap( 'b has 8 entries' )) << 32 ) |
			( u64( b.__getitem__( 4 ).unwrap( 'b has 8 entries' )) << 24 ) |
			( u64( b.__getitem__( 5 ).unwrap( 'b has 8 entries' )) << 16 ) |
			( u64( b.__getitem__( 6 ).unwrap( 'b has 8 entries' )) << 8  ) |
			  u64( b.__getitem__( 7 ).unwrap( 'b has 8 entries' ))
		)


class Random:
	__s0: u64
	__s1: u64
	__s2: u64
	__s3: u64

	def __init__( self, seed: u64|None = None ) -> None:
		''' seed=None (default) draws a fresh seed from the CSPRNG above -
		same "reseed from real entropy" convention as Python's own
		random.seed(None). Pass an explicit seed for a reproducible
		sequence (simulations/tests). '''
		words: UnsafeList[u64] = _expand_seed( _resolve_seed( seed ))
		self.__s0 = words.__getitem__( 0 ).unwrap( '_expand_seed always returns 4 words' )
		self.__s1 = words.__getitem__( 1 ).unwrap( '_expand_seed always returns 4 words' )
		self.__s2 = words.__getitem__( 2 ).unwrap( '_expand_seed always returns 4 words' )
		self.__s3 = words.__getitem__( 3 ).unwrap( '_expand_seed always returns 4 words' )

	def seed( self, value: u64|None = None ) -> None:
		''' reseeds this generator - same value always produces the same
		subsequent sequence. value=None draws a fresh CSPRNG seed instead
		(see __init__'s own docstring). '''
		words: UnsafeList[u64] = _expand_seed( _resolve_seed( value ))
		self.__s0 = words.__getitem__( 0 ).unwrap( '_expand_seed always returns 4 words' )
		self.__s1 = words.__getitem__( 1 ).unwrap( '_expand_seed always returns 4 words' )
		self.__s2 = words.__getitem__( 2 ).unwrap( '_expand_seed always returns 4 words' )
		self.__s3 = words.__getitem__( 3 ).unwrap( '_expand_seed always returns 4 words' )

	def _next_u64( self ) -> u64:
		''' xoshiro256** (Blackman/Vigna, public domain) - one 64-bit draw,
		advancing internal state. '''
		with compiler.wrap_arithmetic:
			result: u64 = _rotl64( self.__s1 * u64( 5 ), u64( 7 )) * u64( 9 )
			t: u64 = self.__s1 << u64( 17 )
			self.__s2 ^= self.__s0
			self.__s3 ^= self.__s1
			self.__s1 ^= self.__s2
			self.__s0 ^= self.__s3
			self.__s2 ^= t
			self.__s3 = _rotl64( self.__s3, u64( 45 ))
			return result

	def random( self ) -> f64:
		''' a float uniformly in [0.0, 1.0) - top 53 bits of a draw (f64's
		full mantissa precision), matching Python's own random.random()
		range/precision. '''
		with compiler.wrap_arithmetic:
			bits: u64 = self._next_u64() >> u64( 11 )
			return f64( bits ) * ( 1.0 / 9007199254740992.0 )  # 1 / 2**53

	def uniform( self, a: f64, b: f64 ) -> f64:
		''' a float uniformly in [a, b] (matching Python's own
		random.uniform, including its "b is inclusive, subject to float
		rounding" caveat). '''
		with compiler.wrap_arithmetic:
			return a + ( b - a ) * self.random()

	def randint( self, a: i64, b: i64 ) -> i64:
		''' an int uniformly in [a, b], BOTH ends inclusive - matches
		Python's own random.randint (not the half-open randrange). Uses
		rejection sampling against a modulo, not a naive unconditional
		modulo (a naive modulo biases toward smaller values whenever the
		range doesn't evenly divide 2**64) - see the loop below for why
		this doesn't use the faster widening-multiply (Lemire) method. '''
		if a > b:
			sys.panic( 'Random.randint: a must be <= b' )
		# number of representable values; wraps to 0 exactly when [a,b]
		# spans the full i64 range (a=i64 MIN, b=i64 MAX) - handled below
		# as "every u64 is in range", not a real error (a<=b already
		# checked above).
		with compiler.wrap_arithmetic:
			span: u64 = u64( b - a ) + u64( 1 )
		if span == u64( 0 ):
			with compiler.wrap_arithmetic:
				return i64( self._next_u64() )
		# rejection sampling against a modulo (NOT Lemire's widening-multiply
		# method - that needs a real 128-bit multiply, and this compiler's
		# u128 is a plain-64-bit fallback under MSVC, cl.exe having no
		# native 128-bit integer type - see emitter_c.py's own
		# __metalpy_wideint comment). threshold rejects the partial top
		# bucket of u64 space that a plain modulo would otherwise bias
		# toward (2**64 % span values reserved, via 2**64-span computed as
		# an unsigned wraparound subtraction rather than literally 2**64).
		with compiler.wrap_arithmetic:
			neg_span: u64 = u64( 0 ) - span  # 2**64 - span, via unsigned wraparound
		with compiler.panic_arithmetic( 'Random.randint: span is nonzero, checked above' ):
			threshold: u64 = neg_span % span
		while True:
			x: u64 = self._next_u64()
			if x >= threshold:
				with compiler.panic_arithmetic( 'Random.randint: span is nonzero, checked above' ):
					return a + i64( x % span )

	def choice[T]( self, seq: list[T] ) -> Result[T, IndexError]:
		''' a uniformly random element of seq - Result, not a raise, since
		an empty seq has no element to return (matches this codebase's own
		Result-not-exception convention; Python's random.choice raises
		IndexError for the same case). '''
		n: usize = seq.__len__()
		if n == usize( 0 ):
			return Result.Err( IndexError() )
		with compiler.wrap_arithmetic:
			idx: i64 = i64( n ) - i64( 1 )
		return seq.__getitem__( usize( self.randint( i64( 0 ), idx )))

	def shuffle[T]( self, seq: list[T] ) -> None:
		''' in-place Fisher-Yates shuffle, uniform over all permutations. '''
		n: usize = seq.__len__()
		if n < usize( 2 ):
			return
		with compiler.wrap_arithmetic:
			i: usize = n - usize( 1 )
			while i > usize( 0 ):
				j: usize = usize( self.randint( i64( 0 ), i64( i )))
				vi: T = seq.__getitem__( i ).unwrap( 'i < n by construction' )
				vj: T = seq.__getitem__( j ).unwrap( 'j <= i < n by construction' )
				seq.__setitem__( i, vj ).unwrap( 'i < n by construction' )
				seq.__setitem__( j, vi ).unwrap( 'j <= i < n by construction' )
				i -= usize( 1 )


# ---------------------------------------------------------------------------
# Module-level convenience API, mirroring Python's random.seed()/random()/
# randint()/choice()/shuffle()/uniform() - a single shared, lock-guarded
# Random instance, lazily CSPRNG-seeded on first use.
#
# Lock-guarded, not a bare is-None check: see lib/datetime.py's own
# localtz() for why check-then-set on a lazily-constructed module global is
# a real data race under concurrent first touch (confirmed there via a real
# crash repro), not just a theoretical concern.
# ---------------------------------------------------------------------------

__default: Random|None = None
__default_lock: threading.FastLock = threading.FastLock()

def _default() -> Random:
	''' caller must already hold __default_lock. '''
	global __default
	if __default is None:
		__default = Random()
	return __default

def seed( value: u64|None = None ) -> None:
	with __default_lock:
		_default().seed( value )

def random() -> f64:
	with __default_lock:
		return _default().random()

def uniform( a: f64, b: f64 ) -> f64:
	with __default_lock:
		return _default().uniform( a, b )

def randint( a: i64, b: i64 ) -> i64:
	with __default_lock:
		return _default().randint( a, b )

def choice[T]( seq: list[T] ) -> Result[T, IndexError]:
	with __default_lock:
		return _default().choice( seq )

def shuffle[T]( seq: list[T] ) -> None:
	with __default_lock:
		_default().shuffle( seq )
