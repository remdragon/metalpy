# deflate - RFC 1951 block framing: composes lz77.py's token stream through
# huffman.py's code tables, packed via bitstream.py. zipfile.py only ever
# calls compress()/decompress_exact() - never touches lz77/huffman/bitstream
# directly.
#
# compress() always emits a single BFINAL=1 block (splitting into several
# smaller blocks is a size optimization, never a correctness requirement -
# RFC 1951 imposes no limit on a single block's token count). It never emits
# a STORED block either (a real gap for incompressible/tiny input - see this
# module's own docstring at the bottom) - only FIXED or DYNAMIC, falling
# back to FIXED whenever DYNAMIC's own Huffman table construction reports
# CodeTooLong (see huffman.build_code_lengths_from_frequencies - vanishingly
# rare for real data, but a real possibility for adversarial input).
#
# decompress_exact()/decompress_unbounded() decode all three block types
# (STORED/FIXED/DYNAMIC, multi-block streams too) - this direction is the
# hard interop requirement (must read files produced by other tools), so it
# doesn't get to skip anything compress() skips.
#
# Every fallible sub-call below goes through one of the small `_xxx` wrapper
# functions further down (_read_bits, _build_encoder, _decode_symbol, ...)
# rather than an inline `match ...: case Ok(v): x = v; case Err: return ...`
# at the use site - a bare `x: T` declaration whose value comes from a
# LATER match arm crashes/misinfers as "not initialized on all code
# branches" once more than one such match appears in the same block (a real
# compiler bug in the definite-assignment checker, confirmed via a minimal
# repro and reported separately). Each wrapper is a single match with a
# direct `return` in every arm - safe - and callers just do
# `x: T = _wrapper(...).or_return()`.

import bitstream
import huffman
import lz77
import compiler


class DeflateError:
	message: str

	def __init__( self, message: str ) -> None:
		self.message = message

	def __str__( self ) -> str:
		return self.message

	def __repr__( self ) -> str:
		return f"DeflateError({self.message!r})"


BTYPE_STORED:  u32 = 0
BTYPE_FIXED:   u32 = 1
BTYPE_DYNAMIC: u32 = 2


# ---------------------------------------------------------------------------
# error-converting wrappers - see this module's own header comment
# ---------------------------------------------------------------------------

def _read_bits( reader: bitstream.BitReader, nbits: u32, context: str ) -> Result[u32, DeflateError]:
	match reader.read_bits( nbits ):
		case Result.Ok( v ):
			return Result.Ok( v )
		case Result.Err( _ ):
			return Result.Err( DeflateError( context ))


def _build_encoder( lengths: UnsafeList[u8], context: str ) -> Result[huffman.HuffmanEncoder, DeflateError]:
	match huffman.HuffmanEncoder( lengths ):
		case Result.Ok( e ):
			return Result.Ok( e )
		case Result.Err( _ ):
			return Result.Err( DeflateError( context ))


def _build_decoder( lengths: UnsafeList[u8], context: str ) -> Result[huffman.HuffmanDecoder, DeflateError]:
	match huffman.HuffmanDecoder( lengths ):
		case Result.Ok( d ):
			return Result.Ok( d )
		case Result.Err( _ ):
			return Result.Err( DeflateError( context ))


def _decode_symbol( dec: huffman.HuffmanDecoder, reader: bitstream.BitReader, context: str ) -> Result[u16, DeflateError]:
	match dec.decode( reader ):
		case Result.Ok( s ):
			return Result.Ok( s )
		case Result.Err( _ ):
			return Result.Err( DeflateError( context ))


def _write_symbol( enc: huffman.HuffmanEncoder, writer: bitstream.BitWriter, symbol: u16, context: str ) -> Result[None, DeflateError]:
	match enc.write_symbol( writer, symbol ):
		case Result.Ok( _ ):
			return Result.Ok( None )
		case Result.Err( _ ):
			return Result.Err( DeflateError( context ))


def _build_lengths( freqs: UnsafeList[u32], max_len: u8, context: str ) -> Result[UnsafeList[u8], DeflateError]:
	match huffman.build_code_lengths_from_frequencies( freqs, max_len ):
		case Result.Ok( v ):
			return Result.Ok( v )
		case Result.Err( _ ):
			return Result.Err( DeflateError( context ))


def _do_find_matches( data: bytes|bytearray, level: i32 ) -> UnsafeList[lz77.LZ77Token]:
	return lz77.find_matches( data, level )


def _do_copy_match( out: UnsafeList[u8], distance: usize, length: usize, context: str ) -> Result[None, DeflateError]:
	match lz77.copy_match( out, distance, length ):
		case Result.Ok( _ ):
			return Result.Ok( None )
		case Result.Err( _ ):
			return Result.Err( DeflateError( context ))


# ---------------------------------------------------------------------------
# constant tables
# ---------------------------------------------------------------------------

# RFC 1951 SS3.2.5 - the code-length alphabet's own fixed wire order for
# transmitting a dynamic block's HCLEN code lengths. Index i in this list is
# the CODE-LENGTH-ALPHABET SYMBOL whose 3-bit length is transmitted i-th -
# deliberately not ascending order, a well-known easy-to-transpose trap.
_CL_ORDER_DATA: list[i32] = [ 16,17,18, 0,8,7,9,6,10,5,11,4,12,3,13,2,14,1,15 ]


def _build_cl_order() -> UnsafeList[u8]:
	# NOTE: the wire-order data is a LOCAL here, not a module-level global -
	# a module-level list[T]/UnsafeList[T] global read (even via
	# __getitem__) from inside a function used to initialize a DIFFERENT
	# module-level global crashes at runtime (confirmed minimal repro,
	# reported separately) - a plain scalar-to-scalar global dependency is
	# fine, and reading a list[T] global directly from an ordinary
	# function (never from within another global's own initializer) is
	# also fine, so keeping each of these tables' source data local to its
	# own builder avoids the bug entirely rather than working around it
	# with cross-global reads.
	data: list[i32] = [ 16,17,18, 0,8,7,9,6,10,5,11,4,12,3,13,2,14,1,15 ]
	order: UnsafeList[u8] = UnsafeList[u8]( usize( 19 ))
	i: usize = 0
	with compiler.panic_arithmetic( 'fixed 19-entry literal, cannot overflow' ):
		while i < usize( len( data )):
			order.append( u8( data.__getitem__( i ).unwrap( 'i in range' )))
			i += 1
	return order


CL_ORDER: UnsafeList[u8] = _build_cl_order()


class _U16U8Table:
	base:  UnsafeList[u16]
	extra: UnsafeList[u8]

	def __init__( self, base: UnsafeList[u16], extra: UnsafeList[u8] ) -> None:
		self.base  = base
		self.extra = extra


# RFC 1951 SS3.2.5 - length codes 257..285 (index 0..28 here), base length
# and extra-bit count each carries. Base/extra for symbols 257..285 in
# order. See _build_cl_order's own comment: the source literal is kept
# LOCAL to this one function (not a module-level global) to avoid the
# cross-global-initializer-dependency crash.
def _build_length_table() -> _U16U8Table:
	data: list[i32] = [
		3,0, 4,0, 5,0, 6,0, 7,0, 8,0, 9,0, 10,0,
		11,1, 13,1, 15,1, 17,1,
		19,2, 23,2, 27,2, 31,2,
		35,3, 43,3, 51,3, 59,3,
		67,4, 83,4, 99,4, 115,4,
		131,5, 163,5, 195,5, 227,5,
		258,0,
	]
	bases: UnsafeList[u16] = UnsafeList[u16]()
	extra: UnsafeList[u8]  = UnsafeList[u8]()
	i: usize = 0
	with compiler.panic_arithmetic( 'fixed 58-entry literal, cannot overflow' ):
		while i < usize( len( data )):
			bases.append( u16( data.__getitem__( i ).unwrap( 'i in range' )))
			extra.append( u8( data.__getitem__( i + 1 ).unwrap( 'i+1 in range' )))
			i += 2
	return _U16U8Table( bases, extra )


_LENGTH_TABLE: _U16U8Table = _build_length_table()
LENGTH_BASE:  UnsafeList[u16] = _LENGTH_TABLE.base
LENGTH_EXTRA: UnsafeList[u8]  = _LENGTH_TABLE.extra


# RFC 1951 SS3.2.5 - distance codes 0..29, base distance and extra-bit
# count each carries.
def _build_dist_table() -> _U16U8Table:
	data: list[i32] = [
		1,0, 2,0, 3,0, 4,0,
		5,1, 7,1,
		9,2, 13,2,
		17,3, 25,3,
		33,4, 49,4,
		65,5, 97,5,
		129,6, 193,6,
		257,7, 385,7,
		513,8, 769,8,
		1025,9, 1537,9,
		2049,10, 3073,10,
		4097,11, 6145,11,
		8193,12, 12289,12,
		16385,13, 24577,13,
	]
	bases: UnsafeList[u16] = UnsafeList[u16]()
	extra: UnsafeList[u8]  = UnsafeList[u8]()
	i: usize = 0
	with compiler.panic_arithmetic( 'fixed 60-entry literal, cannot overflow' ):
		while i < usize( len( data )):
			bases.append( u16( data.__getitem__( i ).unwrap( 'i in range' )))
			extra.append( u8( data.__getitem__( i + 1 ).unwrap( 'i+1 in range' )))
			i += 2
	return _U16U8Table( bases, extra )


_DIST_TABLE: _U16U8Table = _build_dist_table()
DIST_BASE:  UnsafeList[u16] = _DIST_TABLE.base
DIST_EXTRA: UnsafeList[u8]  = _DIST_TABLE.extra


def _length_to_symbol( length: u16 ) -> tuple[u16, u32, u32]:
	''' -> (litlen symbol 257..285, extra-bits value, extra-bit count) '''
	idx: usize = 0
	with compiler.panic_arithmetic( 'bounded by LENGTH_BASE size, cannot overflow' ):
		while idx + 1 < len( LENGTH_BASE ):
			next_base: u16 = LENGTH_BASE.__getitem__( idx + 1 ).unwrap( 'idx+1 in range' )
			if length < next_base:
				break
			idx += 1
	base: u16 = LENGTH_BASE.__getitem__( idx ).unwrap( 'idx in range' )
	extra_bits: u8 = LENGTH_EXTRA.__getitem__( idx ).unwrap( 'idx in range' )
	with compiler.wrap_arithmetic:
		extra_val: u32 = u32( length ) - u32( base )
		symbol: u16 = u16( 257 ) + u16( idx )
	return ( symbol, extra_val, u32( extra_bits ))


def _distance_to_symbol( distance: u16 ) -> tuple[u16, u32, u32]:
	''' -> (distance symbol 0..29, extra-bits value, extra-bit count) '''
	idx: usize = 0
	with compiler.panic_arithmetic( 'bounded by DIST_BASE size, cannot overflow' ):
		while idx + 1 < len( DIST_BASE ):
			next_base: u16 = DIST_BASE.__getitem__( idx + 1 ).unwrap( 'idx+1 in range' )
			if distance < next_base:
				break
			idx += 1
	base: u16 = DIST_BASE.__getitem__( idx ).unwrap( 'idx in range' )
	extra_bits: u8 = DIST_EXTRA.__getitem__( idx ).unwrap( 'idx in range' )
	with compiler.wrap_arithmetic:
		extra_val: u32 = u32( distance ) - u32( base )
		symbol: u16 = u16( idx )
	return ( symbol, extra_val, u32( extra_bits ))


# ---------------------------------------------------------------------------
# fixed Huffman tables (RFC 1951 SS3.2.6) - built once, reused by every
# compress()/decompress call that hits a fixed block
# ---------------------------------------------------------------------------

def _build_fixed_litlen_lengths() -> UnsafeList[u8]:
	lengths: UnsafeList[u8] = UnsafeList[u8]( usize( 288 ))
	sym: usize = 0
	with compiler.panic_arithmetic( 'bounded by 288, cannot overflow' ):
		while sym < usize( 288 ):
			if sym <= 143:
				lengths.append( u8( 8 ))
			elif sym <= 255:
				lengths.append( u8( 9 ))
			elif sym <= 279:
				lengths.append( u8( 7 ))
			else:
				lengths.append( u8( 8 ))
			sym += 1
	return lengths


def _build_fixed_dist_lengths() -> UnsafeList[u8]:
	lengths: UnsafeList[u8] = UnsafeList[u8]( usize( 32 ))
	sym: usize = 0
	with compiler.panic_arithmetic( 'bounded by 32, cannot overflow' ):
		while sym < usize( 32 ):
			lengths.append( u8( 5 ))
			sym += 1
	return lengths


_FIXED_LITLEN_LENGTHS: UnsafeList[u8] = _build_fixed_litlen_lengths()
_FIXED_DIST_LENGTHS:   UnsafeList[u8] = _build_fixed_dist_lengths()


# ---------------------------------------------------------------------------
# compress
# ---------------------------------------------------------------------------

def compress( data: bytes|bytearray, level: i32 = 6 ) -> Result[bytes,DeflateError]:
	tokens: UnsafeList[lz77.LZ77Token] = _do_find_matches( data, level )

	litlen_freqs: UnsafeList[u32] = UnsafeList[u32]( usize( 286 ))
	fi: usize = 0
	with compiler.panic_arithmetic( 'bounded by 286, cannot overflow' ):
		while fi < usize( 286 ):
			litlen_freqs.append( u32( 0 ))
			fi += 1
	dist_freqs: UnsafeList[u32] = UnsafeList[u32]( usize( 30 ))
	di: usize = 0
	with compiler.panic_arithmetic( 'bounded by 30, cannot overflow' ):
		while di < usize( 30 ):
			dist_freqs.append( u32( 0 ))
			di += 1

	_tally_frequencies( tokens, litlen_freqs, dist_freqs )

	writer = bitstream.BitWriter()
	writer.write_bits( u32( 1 ), u32( 1 ))  # BFINAL=1: always a single block

	dynamic_lengths: DynamicLengths|None = _try_build_dynamic_lengths( litlen_freqs, dist_freqs )
	if dynamic_lengths is not None:
		writer.write_bits( BTYPE_DYNAMIC, u32( 2 ))
		_emit_dynamic_header( writer, dynamic_lengths.litlen, dynamic_lengths.dist ).or_return()
		litlen_enc: huffman.HuffmanEncoder = _build_encoder( dynamic_lengths.litlen, 'compress: dynamic litlen encoder build failed' ).or_return()
		dist_enc: huffman.HuffmanEncoder = _build_encoder( dynamic_lengths.dist, 'compress: dynamic dist encoder build failed' ).or_return()
		_emit_tokens( writer, tokens, litlen_enc, dist_enc ).or_return()
	else:
		writer.write_bits( BTYPE_FIXED, u32( 2 ))
		litlen_enc2: huffman.HuffmanEncoder = _build_encoder( _FIXED_LITLEN_LENGTHS, 'compress: fixed litlen encoder build failed' ).or_return()
		dist_enc2: huffman.HuffmanEncoder = _build_encoder( _FIXED_DIST_LENGTHS, 'compress: fixed dist encoder build failed' ).or_return()
		_emit_tokens( writer, tokens, litlen_enc2, dist_enc2 ).or_return()

	return writer.finish()


def _tally_frequencies( tokens: UnsafeList[lz77.LZ77Token], litlen_freqs: UnsafeList[u32], dist_freqs: UnsafeList[u32] ) -> None:
	ti: usize = 0
	with compiler.panic_arithmetic( 'bounded by len(tokens), cannot overflow' ):
		while ti < len( tokens ):
			tok: lz77.LZ77Token = tokens.__getitem__( ti ).unwrap( 'ti < len(tokens)' )
			match tok:
				case lz77.LZ77Token.Literal( b ):
					idx: usize = usize( b )
					prev: u32 = litlen_freqs.__getitem__( idx ).unwrap( 'idx < 286' )
					with compiler.wrap_arithmetic:
						litlen_freqs.__setitem__( idx, prev + 1 ).unwrap( 'idx < 286' )
				case lz77.LZ77Token.Match( m ):
					lsym: tuple[u16, u32, u32] = _length_to_symbol( m.length )
					with compiler.panic_arithmetic( 'lsym[0] in 257..285, litlen_freqs sized 286' ):
						lidx: usize = usize( lsym[0] )
					lprev: u32 = litlen_freqs.__getitem__( lidx ).unwrap( 'lidx < 286' )
					with compiler.wrap_arithmetic:
						litlen_freqs.__setitem__( lidx, lprev + 1 ).unwrap( 'lidx < 286' )
					dsym: tuple[u16, u32, u32] = _distance_to_symbol( m.distance )
					didx: usize = usize( dsym[0] )
					dprev: u32 = dist_freqs.__getitem__( didx ).unwrap( 'didx < 30' )
					with compiler.wrap_arithmetic:
						dist_freqs.__setitem__( didx, dprev + 1 ).unwrap( 'didx < 30' )
			ti += 1
	# EOB always appears exactly once
	eob_prev: u32 = litlen_freqs.__getitem__( usize( 256 )).unwrap( 'x' )
	with compiler.wrap_arithmetic:
		litlen_freqs.__setitem__( usize( 256 ), eob_prev + 1 ).unwrap( 'x' )

	# RFC 1951 SS3.2.7: HDIST can never be zero - a block with no matches at
	# all still transmits one (unused) distance code, conventionally symbol
	# 0 with length 1 (matches every other real DEFLATE encoder's behavior)
	any_dist: bool = False
	adi: usize = 0
	with compiler.panic_arithmetic( 'bounded by 30, cannot overflow' ):
		while adi < usize( 30 ):
			if dist_freqs.__getitem__( adi ).unwrap( 'adi < 30' ) > 0:
				any_dist = True
			adi += 1
	if not any_dist:
		dist_freqs.__setitem__( usize( 0 ), u32( 1 )).unwrap( 'x' )


class DynamicLengths:
	litlen: UnsafeList[u8]
	dist:   UnsafeList[u8]

	def __init__( self, litlen: UnsafeList[u8], dist: UnsafeList[u8] ) -> None:
		self.litlen = litlen
		self.dist   = dist


def _try_build_dynamic_lengths( litlen_freqs: UnsafeList[u32], dist_freqs: UnsafeList[u32] ) -> DynamicLengths|None:
	match huffman.build_code_lengths_from_frequencies( litlen_freqs, u8( 15 )):
		case Result.Ok( litlen_lengths ):
			match huffman.build_code_lengths_from_frequencies( dist_freqs, u8( 15 )):
				case Result.Ok( dist_lengths ):
					return DynamicLengths( litlen_lengths, dist_lengths )
				case Result.Err( _ ):
					return None
		case Result.Err( _ ):
			return None


def _emit_tokens(
	writer: bitstream.BitWriter,
	tokens: UnsafeList[lz77.LZ77Token],
	litlen_enc: huffman.HuffmanEncoder,
	dist_enc: huffman.HuffmanEncoder,
) -> Result[None, DeflateError]:
	ti: usize = 0
	with compiler.panic_arithmetic( 'bounded by len(tokens), cannot overflow' ):
		while ti < len( tokens ):
			tok: lz77.LZ77Token = tokens.__getitem__( ti ).unwrap( 'ti < len(tokens)' )
			match tok:
				case lz77.LZ77Token.Literal( b ):
					_write_symbol( litlen_enc, writer, u16( b ), '_emit_tokens: literal symbol not in table' ).or_return()
				case lz77.LZ77Token.Match( m ):
					lsym: tuple[u16, u32, u32] = _length_to_symbol( m.length )
					_write_symbol( litlen_enc, writer, lsym[0], '_emit_tokens: length symbol not in table' ).or_return()
					if lsym[2] > 0:
						writer.write_bits( lsym[1], lsym[2] )
					dsym: tuple[u16, u32, u32] = _distance_to_symbol( m.distance )
					_write_symbol( dist_enc, writer, dsym[0], '_emit_tokens: distance symbol not in table' ).or_return()
					if dsym[2] > 0:
						writer.write_bits( dsym[1], dsym[2] )
			ti += 1
	_write_symbol( litlen_enc, writer, u16( 256 ), '_emit_tokens: end-of-block symbol not in table' ).or_return()  # end-of-block
	return Result.Ok( None )


def _emit_dynamic_header( writer: bitstream.BitWriter, litlen_lengths: UnsafeList[u8], dist_lengths: UnsafeList[u8] ) -> Result[None, DeflateError]:
	''' Always transmits the FULL 286/30-entry tables (HLIT=286, HDIST=30)
	and the full 19-entry code-length alphabet (HCLEN=19), rather than
	trimming trailing zero lengths - simpler, always correct, costs a few
	extra bytes per block. RLE-encodes the combined length array through
	the 19-symbol code-length alphabet per RFC 1951 SS3.2.7 (repeat-
	previous / repeat-zero codes 16/17/18). '''
	combined: UnsafeList[u8] = UnsafeList[u8]()
	ci: usize = 0
	with compiler.panic_arithmetic( 'bounded by 286, cannot overflow' ):
		while ci < len( litlen_lengths ):
			combined.append( litlen_lengths.__getitem__( ci ).unwrap( 'ci in range' ))
			ci += 1
	di: usize = 0
	with compiler.panic_arithmetic( 'bounded by 30, cannot overflow' ):
		while di < len( dist_lengths ):
			combined.append( dist_lengths.__getitem__( di ).unwrap( 'di in range' ))
			di += 1

	cl_freqs: UnsafeList[u32] = UnsafeList[u32]( usize( 19 ))
	cfi: usize = 0
	with compiler.panic_arithmetic( 'bounded by 19, cannot overflow' ):
		while cfi < usize( 19 ):
			cl_freqs.append( u32( 0 ))
			cfi += 1

	rle_symbols: UnsafeList[u8] = UnsafeList[u8]()
	rle_extra:   UnsafeList[u8] = UnsafeList[u8]()

	n: usize = len( combined )
	i: usize = 0
	with compiler.panic_arithmetic( 'bounded by n, cannot overflow' ):
		while i < n:
			val: u8 = combined.__getitem__( i ).unwrap( 'i < n' )
			run: usize = 1
			while i + run < n and combined.__getitem__( i + run ).unwrap( 'i+run < n' ) == val and run < 138:
				run += 1
			if val == 0:
				while run >= 11:
					take: usize = run if run < 138 else usize( 138 )
					rle_symbols.append( u8( 18 ))
					with compiler.wrap_arithmetic:
						rle_extra.append( u8( take - 11 ))
						bump18: u32 = cl_freqs.__getitem__( usize( 18 )).unwrap( 'x' ) + 1
					cl_freqs.__setitem__( usize( 18 ), bump18 ).unwrap( 'x' )
					with compiler.panic_arithmetic( 'take <= run, cannot underflow' ):
						run -= take
					i += take
				while run >= 3:
					take2: usize = run if run < 10 else usize( 10 )
					rle_symbols.append( u8( 17 ))
					with compiler.wrap_arithmetic:
						rle_extra.append( u8( take2 - 3 ))
						bump17: u32 = cl_freqs.__getitem__( usize( 17 )).unwrap( 'x' ) + 1
					cl_freqs.__setitem__( usize( 17 ), bump17 ).unwrap( 'x' )
					with compiler.panic_arithmetic( 'take2 <= run, cannot underflow' ):
						run -= take2
					i += take2
				while run > 0:
					rle_symbols.append( u8( 0 ))
					rle_extra.append( u8( 0 ))
					with compiler.wrap_arithmetic:
						bump0: u32 = cl_freqs.__getitem__( usize( 0 )).unwrap( 'x' ) + 1
					cl_freqs.__setitem__( usize( 0 ), bump0 ).unwrap( 'x' )
					run -= 1
					i += 1
			else:
				rle_symbols.append( val )
				rle_extra.append( u8( 0 ))
				with compiler.wrap_arithmetic:
					bumpv: u32 = cl_freqs.__getitem__( usize( val )).unwrap( 'x' ) + 1
				cl_freqs.__setitem__( usize( val ), bumpv ).unwrap( 'x' )
				i += 1
				with compiler.panic_arithmetic( 'run >= 1, just consumed one' ):
					run -= 1
				while run >= 3:
					take3: usize = run if run < 6 else usize( 6 )
					rle_symbols.append( u8( 16 ))
					with compiler.wrap_arithmetic:
						rle_extra.append( u8( take3 - 3 ))
						bump16: u32 = cl_freqs.__getitem__( usize( 16 )).unwrap( 'x' ) + 1
					cl_freqs.__setitem__( usize( 16 ), bump16 ).unwrap( 'x' )
					with compiler.panic_arithmetic( 'take3 <= run, cannot underflow' ):
						run -= take3
					i += take3
				while run > 0:
					rle_symbols.append( val )
					rle_extra.append( u8( 0 ))
					with compiler.wrap_arithmetic:
						bumpv2: u32 = cl_freqs.__getitem__( usize( val )).unwrap( 'x' ) + 1
					cl_freqs.__setitem__( usize( val ), bumpv2 ).unwrap( 'x' )
					run -= 1
					i += 1

	cl_lengths: UnsafeList[u8] = _build_lengths( cl_freqs, u8( 7 ), '_emit_dynamic_header: code-length alphabet table build failed' ).or_return()
	cl_enc: huffman.HuffmanEncoder = _build_encoder( cl_lengths, '_emit_dynamic_header: code-length alphabet encoder build failed' ).or_return()

	writer.write_bits( u32( 286 - 257 ), u32( 5 ))  # HLIT
	writer.write_bits( u32( 30 - 1 ), u32( 5 ))      # HDIST
	writer.write_bits( u32( 19 - 4 ), u32( 4 ))      # HCLEN

	oi: usize = 0
	with compiler.panic_arithmetic( 'bounded by 19, cannot overflow' ):
		while oi < usize( 19 ):
			sym_idx: u8 = CL_ORDER.__getitem__( oi ).unwrap( 'oi < 19' )
			cl_len: u8 = cl_lengths.__getitem__( usize( sym_idx )).unwrap( 'sym_idx < 19' )
			writer.write_bits( u32( cl_len ), u32( 3 ))
			oi += 1

	ri: usize = 0
	with compiler.panic_arithmetic( 'bounded by len(rle_symbols), cannot overflow' ):
		while ri < len( rle_symbols ):
			rsym: u8 = rle_symbols.__getitem__( ri ).unwrap( 'ri in range' )
			rextra: u8 = rle_extra.__getitem__( ri ).unwrap( 'ri in range' )
			_write_symbol( cl_enc, writer, u16( rsym ), '_emit_dynamic_header: rle symbol not in cl table' ).or_return()
			if rsym == 16:
				writer.write_bits( u32( rextra ), u32( 2 ))
			elif rsym == 17:
				writer.write_bits( u32( rextra ), u32( 3 ))
			elif rsym == 18:
				writer.write_bits( u32( rextra ), u32( 7 ))
			ri += 1

	return Result.Ok( None )


# ---------------------------------------------------------------------------
# decompress
# ---------------------------------------------------------------------------

def decompress_exact( data: bytes|bytearray, uncompressed_size: usize ) -> Result[bytes, DeflateError]:
	out: UnsafeList[u8] = UnsafeList[u8]( uncompressed_size )
	reader = bitstream.BitReader( data )
	_inflate_into( reader, out ).or_return()
	if len( out ) != uncompressed_size:
		return Result.Err( DeflateError( 'decompress_exact: decoded size does not match uncompressed_size' ))
	return Result.Ok( _unsafe_list_u8_to_bytes( out ))


def decompress_unbounded( data: bytes|bytearray ) -> Result[bytes, DeflateError]:
	out: UnsafeList[u8] = UnsafeList[u8]()
	reader = bitstream.BitReader( data )
	_inflate_into( reader, out ).or_return()
	return Result.Ok( _unsafe_list_u8_to_bytes( out ))


def _unsafe_list_u8_to_bytes( src: UnsafeList[u8] ) -> bytes:
	n: usize = len( src )
	out: bytearray = bytearray( n )
	op: Ptr[u8] = out.get_ptr()
	i: usize = 0
	with compiler.panic_arithmetic( 'bounded by n, cannot overflow' ):
		while i < n:
			op[i] = src.__getitem__( i ).unwrap( 'i < n' )
			i += 1
	return bytes.from_bytearray( move( out ))


def _inflate_into( reader: bitstream.BitReader, out: UnsafeList[u8] ) -> Result[None, DeflateError]:
	while True:
		bfinal: u32 = _read_bits( reader, u32( 1 ), '_inflate_into: unexpected end of stream reading BFINAL' ).or_return()
		btype: u32 = _read_bits( reader, u32( 2 ), '_inflate_into: unexpected end of stream reading BTYPE' ).or_return()

		if btype == BTYPE_STORED:
			_inflate_stored( reader, out ).or_return()
		elif btype == BTYPE_FIXED:
			_inflate_huffman_block( reader, out, _FIXED_LITLEN_LENGTHS, _FIXED_DIST_LENGTHS ).or_return()
		elif btype == BTYPE_DYNAMIC:
			_inflate_dynamic_block( reader, out ).or_return()
		else:
			return Result.Err( DeflateError( '_inflate_into: invalid BTYPE 3 (reserved)' ))

		if bfinal == 1:
			break
	return Result.Ok( None )


def _inflate_stored( reader: bitstream.BitReader, out: UnsafeList[u8] ) -> Result[None, DeflateError]:
	reader.align_to_byte()
	len_val: u32 = _read_bits( reader, u32( 16 ), '_inflate_stored: unexpected end of stream reading LEN' ).or_return()
	nlen_val: u32 = _read_bits( reader, u32( 16 ), '_inflate_stored: unexpected end of stream reading NLEN' ).or_return()
	with compiler.wrap_arithmetic:
		complement: u32 = ( ~len_val ) & 0xFFFF
	if complement != nlen_val:
		return Result.Err( DeflateError( '_inflate_stored: LEN/NLEN complement mismatch' ))

	count: usize = usize( len_val )
	i: usize = 0
	with compiler.panic_arithmetic( 'bounded by count, cannot overflow' ):
		while i < count:
			b: u32 = _read_bits( reader, u32( 8 ), '_inflate_stored: unexpected end of stream reading data' ).or_return()
			out.append( u8( b ))
			i += 1
	return Result.Ok( None )


def _inflate_huffman_block( reader: bitstream.BitReader, out: UnsafeList[u8], litlen_lengths: UnsafeList[u8], dist_lengths: UnsafeList[u8] ) -> Result[None, DeflateError]:
	litlen_dec: huffman.HuffmanDecoder = _build_decoder( litlen_lengths, '_inflate_huffman_block: invalid literal/length table' ).or_return()
	dist_dec: huffman.HuffmanDecoder = _build_decoder( dist_lengths, '_inflate_huffman_block: invalid distance table' ).or_return()

	while True:
		sym: u16 = _decode_symbol( litlen_dec, reader, '_inflate_huffman_block: failed to decode literal/length symbol' ).or_return()

		if sym < 256:
			with compiler.panic_arithmetic( 'sym < 256, fits in u8' ):
				lit_byte: u8 = u8( sym )
			out.append( lit_byte )
		elif sym == 256:
			break
		else:
			with compiler.panic_arithmetic( 'sym in 257..285, LENGTH_BASE sized 29' ):
				lidx: usize = usize( sym ) - usize( 257 )
			if lidx >= len( LENGTH_BASE ):
				return Result.Err( DeflateError( '_inflate_huffman_block: length symbol out of range' ))
			lbase: u16 = LENGTH_BASE.__getitem__( lidx ).unwrap( 'lidx in range' )
			lextra_bits: u8 = LENGTH_EXTRA.__getitem__( lidx ).unwrap( 'lidx in range' )
			length: u32 = u32( lbase )
			if lextra_bits > 0:
				ev: u32 = _read_bits( reader, u32( lextra_bits ), '_inflate_huffman_block: unexpected end of stream reading length extra bits' ).or_return()
				with compiler.wrap_arithmetic:
					length += ev

			dsym: u16 = _decode_symbol( dist_dec, reader, '_inflate_huffman_block: failed to decode distance symbol' ).or_return()
			didx: usize = usize( dsym )
			if didx >= len( DIST_BASE ):
				return Result.Err( DeflateError( '_inflate_huffman_block: distance symbol out of range' ))
			dbase: u16 = DIST_BASE.__getitem__( didx ).unwrap( 'didx in range' )
			dextra_bits: u8 = DIST_EXTRA.__getitem__( didx ).unwrap( 'didx in range' )
			distance: u32 = u32( dbase )
			if dextra_bits > 0:
				ev2: u32 = _read_bits( reader, u32( dextra_bits ), '_inflate_huffman_block: unexpected end of stream reading distance extra bits' ).or_return()
				with compiler.wrap_arithmetic:
					distance += ev2

			_do_copy_match( out, usize( distance ), usize( length ), '_inflate_huffman_block: invalid match (bad distance)' ).or_return()
	return Result.Ok( None )


def _inflate_dynamic_block( reader: bitstream.BitReader, out: UnsafeList[u8] ) -> Result[None, DeflateError]:
	hlit_raw: u32 = _read_bits( reader, u32( 5 ), '_inflate_dynamic_block: unexpected end of stream reading HLIT' ).or_return()
	hdist_raw: u32 = _read_bits( reader, u32( 5 ), '_inflate_dynamic_block: unexpected end of stream reading HDIST' ).or_return()
	hclen_raw: u32 = _read_bits( reader, u32( 4 ), '_inflate_dynamic_block: unexpected end of stream reading HCLEN' ).or_return()

	with compiler.wrap_arithmetic:
		hlit:  usize = usize( hlit_raw ) + usize( 257 )
		hdist: usize = usize( hdist_raw ) + usize( 1 )
		hclen: usize = usize( hclen_raw ) + usize( 4 )

	cl_lengths: UnsafeList[u8] = UnsafeList[u8]( usize( 19 ))
	zi: usize = 0
	with compiler.panic_arithmetic( 'bounded by 19, cannot overflow' ):
		while zi < usize( 19 ):
			cl_lengths.append( u8( 0 ))
			zi += 1
	ci: usize = 0
	with compiler.panic_arithmetic( 'bounded by hclen <= 19, cannot overflow' ):
		while ci < hclen:
			v3: u32 = _read_bits( reader, u32( 3 ), '_inflate_dynamic_block: unexpected end of stream reading a CL length' ).or_return()
			sym_idx: u8 = CL_ORDER.__getitem__( ci ).unwrap( 'ci < 19' )
			cl_lengths.__setitem__( usize( sym_idx ), u8( v3 )).unwrap( 'sym_idx < 19' )
			ci += 1

	cl_dec: huffman.HuffmanDecoder = _build_decoder( cl_lengths, '_inflate_dynamic_block: invalid code-length alphabet table' ).or_return()

	with compiler.wrap_arithmetic:
		total: usize = hlit + hdist
	combined: UnsafeList[u8] = UnsafeList[u8]( total )
	with compiler.panic_arithmetic( 'bounded by total, cannot overflow' ):
		while len( combined ) < total:
			csym: u16 = _decode_symbol( cl_dec, reader, '_inflate_dynamic_block: failed to decode a code-length symbol' ).or_return()

			if csym < 16:
				combined.append( u8( csym ))
			elif csym == 16:
				if len( combined ) == 0:
					return Result.Err( DeflateError( '_inflate_dynamic_block: repeat-previous code with no previous length' ))
				with compiler.panic_arithmetic( 'len(combined) > 0, just checked' ):
					prev_len: u8 = combined.__getitem__( len( combined ) - 1 ).unwrap( 'x' )
				extra16: u32 = _read_bits( reader, u32( 2 ), '_inflate_dynamic_block: unexpected end of stream reading repeat-previous extra bits' ).or_return()
				with compiler.wrap_arithmetic:
					repeat_count: u32 = extra16 + 3
				ri16: u32 = 0
				with compiler.panic_arithmetic( 'repeat_count <= 6, cannot overflow' ):
					while ri16 < repeat_count:
						combined.append( prev_len )
						ri16 += 1
			elif csym == 17:
				extra17: u32 = _read_bits( reader, u32( 3 ), '_inflate_dynamic_block: unexpected end of stream reading repeat-zero(short) extra bits' ).or_return()
				with compiler.wrap_arithmetic:
					zrun: u32 = extra17 + 3
				ri17: u32 = 0
				with compiler.panic_arithmetic( 'zrun <= 10, cannot overflow' ):
					while ri17 < zrun:
						combined.append( u8( 0 ))
						ri17 += 1
			elif csym == 18:
				extra18: u32 = _read_bits( reader, u32( 7 ), '_inflate_dynamic_block: unexpected end of stream reading repeat-zero(long) extra bits' ).or_return()
				with compiler.wrap_arithmetic:
					zrun2: u32 = extra18 + 11
				ri18: u32 = 0
				with compiler.panic_arithmetic( 'zrun2 <= 138, cannot overflow' ):
					while ri18 < zrun2:
						combined.append( u8( 0 ))
						ri18 += 1
			else:
				return Result.Err( DeflateError( '_inflate_dynamic_block: invalid code-length symbol' ))

	if len( combined ) != total:
		return Result.Err( DeflateError( '_inflate_dynamic_block: combined length table overran HLIT+HDIST' ))

	litlen_lengths: UnsafeList[u8] = UnsafeList[u8]( hlit )
	li: usize = 0
	with compiler.panic_arithmetic( 'bounded by hlit, cannot overflow' ):
		while li < hlit:
			litlen_lengths.append( combined.__getitem__( li ).unwrap( 'li < total' ))
			li += 1
	dist_lengths: UnsafeList[u8] = UnsafeList[u8]( hdist )
	dj: usize = 0
	with compiler.panic_arithmetic( 'bounded by hdist, cannot overflow' ):
		while dj < hdist:
			with compiler.wrap_arithmetic:
				src_idx: usize = hlit + dj
			dist_lengths.append( combined.__getitem__( src_idx ).unwrap( 'src_idx < total' ))
			dj += 1

	return _inflate_huffman_block( reader, out, litlen_lengths, dist_lengths )
