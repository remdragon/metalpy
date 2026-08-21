# lz77 - sliding-window match finding (encode) and match replay (decode),
# RFC 1951's own limits: matches are length 3..258, distance 1..32768.
# Doesn't know about Huffman coding or DEFLATE block framing - lib/deflate.py
# feeds this module's token stream through lib/huffman.py and packs it via
# lib/bitstream.py.

import compiler

MIN_MATCH:    usize = 3
MAX_MATCH:    usize = 258
MAX_DISTANCE: usize = 32768
HASH_BITS:    u32   = 15
HASH_SIZE:    usize = 32768  # 1 << HASH_BITS


class LZ77Error:
	message: str

	def __init__( self, message: str ) -> None:
		self.message = message


@cstruct
class LZ77Match:
	length:   u16
	distance: u16


@union
class LZ77Token:
	Literal: u8
	Match:   LZ77Match


def _hash3( ptr: ConstPtr[u8], pos: usize ) -> usize:
	''' cheap multiplicative hash over 3 bytes - doesn't need to be
	cryptographic, just cheap and reasonably well distributed. '''
	with compiler.wrap_arithmetic:
		h: u32 = u32( ptr[pos] )
		h = ( h << 5 ) ^ u32( ptr[pos + 1] )
		h = ( h << 5 ) ^ u32( ptr[pos + 2] )
		return usize( h & u32( HASH_SIZE - 1 ))


def _match_length( ptr: ConstPtr[u8], a: usize, b: usize, n: usize ) -> usize:
	''' length of the common prefix of ptr[a:] and ptr[b:], capped at
	MAX_MATCH and at the end of the buffer. a < b always holds (a is an
	earlier position from the hash chain) - both point into the SAME
	already-fully-known input buffer, so unlike decode-side copy_match,
	there's no self-overlap subtlety here even when b - a < the eventual
	match length. '''
	with compiler.panic_arithmetic( 'b <= n, cannot underflow' ):
		max_len: usize = n - b
	if max_len > MAX_MATCH:
		max_len = MAX_MATCH
	length: usize = 0
	with compiler.panic_arithmetic( 'bounded by max_len, cannot overflow' ):
		while length < max_len and ptr[a + length] == ptr[b + length]:
			length += 1
	return length


# ---------------------------------------------------------------------------
# find_matches - encode side
# ---------------------------------------------------------------------------

def find_matches( data: bytes|bytearray, level: i32 = 6 ) -> Result[UnsafeList[LZ77Token], OverflowError]:
	''' level (0..9, like zlib's own) controls how many hash-chain
	candidates get examined per position - a deeper search finds better
	matches but costs more time. Affects compression ratio only: every
	level produces a token stream that decodes back to the same bytes. '''
	n: usize = len( data )
	ptr: ConstPtr[u8] = data.get_const_ptr()
	tokens: UnsafeList[LZ77Token] = UnsafeList[LZ77Token]()

	if n < MIN_MATCH:
		i: usize = 0
		with compiler.panic_arithmetic( 'bounded by n, cannot overflow' ):
			while i < n:
				tokens.append( LZ77Token.Literal( ptr[i] )).or_return()
				i += 1
		return Result.Ok( tokens )

	with compiler.wrap_arithmetic:
		max_chain: usize = usize( level ) * 8 + 8

	head: UnsafeList[i32] = UnsafeList[i32]( HASH_SIZE )
	hi: usize = 0
	with compiler.wrap_arithmetic:
		while hi < HASH_SIZE:
			head.append( i32( -1 )).unwrap( 'sized to HASH_SIZE, never overflows' )
			hi += 1

	prev: UnsafeList[i32] = UnsafeList[i32]( n )
	pi: usize = 0
	with compiler.panic_arithmetic( 'bounded by n, cannot overflow' ):
		while pi < n:
			prev.append( i32( -1 )).unwrap( 'sized to n, never overflows' )
			pi += 1

	pos: usize = 0
	with compiler.panic_arithmetic( 'bounded by n, cannot overflow' ):
		while pos < n:
			with compiler.panic_arithmetic( 'bounded by n, cannot overflow' ):
				have_anchor: bool = pos + MIN_MATCH <= n
			if not have_anchor:
				tokens.append( LZ77Token.Literal( ptr[pos] )).or_return()
				pos += 1
				continue

			h: usize = _hash3( ptr, pos )
			candidate: i32 = head.__getitem__( h ).unwrap( 'h < HASH_SIZE' )
			best_len:  usize = 0
			best_dist: usize = 0
			chain_len: usize = 0
			while candidate != -1 and chain_len < max_chain:
				cand_pos: usize = usize( candidate )
				with compiler.panic_arithmetic( 'cand_pos <= pos, cand_pos is an earlier position' ):
					dist: usize = pos - cand_pos
				if dist <= MAX_DISTANCE:
					match_len: usize = _match_length( ptr, cand_pos, pos, n )
					if match_len > best_len:
						best_len  = match_len
						best_dist = dist
						if best_len >= MAX_MATCH:
							candidate = -1
							break
				candidate = prev.__getitem__( cand_pos ).unwrap( 'cand_pos < n' )
				chain_len += 1

			prev.__setitem__( pos, head.__getitem__( h ).unwrap( 'h < HASH_SIZE' )).unwrap( 'pos < n' )
			head.__setitem__( h, i32( pos )).unwrap( 'h < HASH_SIZE' )

			if best_len >= MIN_MATCH:
				with compiler.panic_arithmetic( 'both bounded by RFC 1951 limits, fit in u16' ):
					tok: LZ77Token = LZ77Token.Match( LZ77Match( length = u16( best_len ), distance = u16( best_dist )))
				tokens.append( tok ).or_return()

				# insert every position the match covers into the hash
				# chains too - otherwise future matches that would
				# reference bytes INSIDE this match are silently missed
				j: usize = pos + 1
				with compiler.panic_arithmetic( 'bounded by n, cannot overflow' ):
					match_end: usize = pos + best_len
				with compiler.panic_arithmetic( 'bounded by n, cannot overflow' ):
					while j < match_end and j + MIN_MATCH <= n:
						hj: usize = _hash3( ptr, j )
						prev.__setitem__( j, head.__getitem__( hj ).unwrap( 'hj < HASH_SIZE' )).unwrap( 'j < n' )
						head.__setitem__( hj, i32( j )).unwrap( 'hj < HASH_SIZE' )
						j += 1
				pos = match_end
			else:
				tokens.append( LZ77Token.Literal( ptr[pos] )).or_return()
				pos += 1

	return Result.Ok( tokens )


# ---------------------------------------------------------------------------
# copy_match - decode side
# ---------------------------------------------------------------------------

def copy_match( out: UnsafeList[u8], distance: usize, length: usize ) -> Result[None, LZ77Error]:
	''' Replays a (length, distance) match by copying from EARLIER IN THE
	SAME output buffer being built - NOT sys.memcpy/memmove: distance <
	length is the normal case (e.g. a run of one repeated byte), and the
	"source" region is still being written as the copy proceeds, so this
	must be byte-at-a-time (each byte becomes readable to the next
	iteration the instant it's appended). '''
	out_len: usize = len( out )
	if distance == 0 or distance > out_len:
		return Result.Err( LZ77Error( 'copy_match: distance out of range' ))

	with compiler.panic_arithmetic( 'distance <= out_len, just checked' ):
		start: usize = out_len - distance

	i: usize = 0
	with compiler.panic_arithmetic( 'bounded by length, cannot overflow' ):
		while i < length:
			src_idx: usize = start + i
			b: u8 = out.__getitem__( src_idx ).unwrap( 'src_idx always already-written: either original data (src_idx < out_len) or already appended earlier this call (distance >= 1 keeps src_idx behind the write cursor)' )
			match out.append( b ):
				case Result.Ok( _ ):
					pass
				case Result.Err( _ ):
					return Result.Err( LZ77Error( 'copy_match: output buffer overflow' ))
			i += 1
	return Result.Ok( None )
