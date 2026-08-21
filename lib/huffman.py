# huffman - canonical Huffman code tables (RFC 1951 SS3.2.2): building codes
# from a code-length array (both directions), and building a length array
# from symbol frequencies for DEFLATE's dynamic blocks.
#
# Doesn't know about DEFLATE block framing - just alphabets and code-length
# arrays. See lib/deflate.py for fixed/dynamic block tables, the literal/
# length/distance alphabets, and the length/distance extra-bits tables.

import bitstream
import compiler

# Covers every DEFLATE alphabet: literal/length and distance codes are
# capped at 15 bits by RFC 1951 itself; the 19-symbol code-length alphabet
# (used to transmit dynamic tables) only ever needs <=7.
MAX_BITS: u32 = 15

# per-length tables are indexed 0..MAX_BITS inclusive, so they need
# MAX_BITS+1 slots - a separate constant instead of repeating `MAX_BITS + 1`
# (itself a runtime add needing its own arithmetic-context wrap) at every
# call site
TABLE_SIZE: usize = 16


@union
class HuffmanError:
	OversubscribedCode:     None  # more codes at some length than fit
	InvalidCodeLength:      None  # a length > MAX_BITS
	UnexpectedEndOfStream:  None  # decode() ran out of bits mid-symbol
	NoSymbolMatched:        None  # decode() exhausted all lengths with no match - implies a validated-but-empty table
	CodeTooLong:            None  # build_code_lengths_from_frequencies only: see its own docstring


# ---------------------------------------------------------------------------
# shared: canonical code assignment from a code-length array (RFC 1951 SS3.2.2)
# ---------------------------------------------------------------------------

def _count_per_length( code_lengths: UnsafeList[u8] ) -> Result[UnsafeList[u16], HuffmanError]:
	n: usize = len( code_lengths )
	count: UnsafeList[u16] = UnsafeList[u16]( TABLE_SIZE )
	i: usize = 0
	with compiler.wrap_arithmetic:
		while i < TABLE_SIZE:
			count.append( u16( 0 )).unwrap( 'fixed TABLE_SIZE entries, never overflows' )
			i += 1
	sym: usize = 0
	with compiler.panic_arithmetic( 'bounded by n, cannot overflow' ):
		while sym < n:
			length: u8 = code_lengths.__getitem__( sym ).unwrap( 'sym < n' )
			if u32( length ) > MAX_BITS:
				return Result.Err( HuffmanError.InvalidCodeLength( None ))
			if length > 0:
				with compiler.wrap_arithmetic:
					idx: usize = usize( length )
					prev: u16 = count.__getitem__( idx ).unwrap( 'idx <= MAX_BITS' )
					count.__setitem__( idx, prev + 1 ).unwrap( 'idx <= MAX_BITS' )
			sym += 1
	# oversubscription check (puff.c's own "left" bookkeeping): a valid
	# code can never use more code space at any prefix than 2^bits allows
	left: i32 = 1
	bits: u32 = 1
	with compiler.wrap_arithmetic:
		while bits <= MAX_BITS:
			left <<= 1
			left -= i32( count.__getitem__( usize( bits )).unwrap( 'bits <= MAX_BITS' ))
			if left < 0:
				return Result.Err( HuffmanError.OversubscribedCode( None ))
			bits += 1
	return Result.Ok( count )


def _canonical_codes_from_lengths( code_lengths: UnsafeList[u8] ) -> Result[UnsafeList[u16], HuffmanError]:
	n: usize = len( code_lengths )
	count: UnsafeList[u16] = _count_per_length( code_lengths ).or_return()

	first_code: UnsafeList[u32] = UnsafeList[u32]( TABLE_SIZE )
	i: usize = 0
	with compiler.wrap_arithmetic:
		while i < TABLE_SIZE:
			first_code.append( u32( 0 )).unwrap( 'fixed TABLE_SIZE entries, never overflows' )
			i += 1
	code: u32 = 0
	bits: u32 = 1
	with compiler.wrap_arithmetic:
		while bits <= MAX_BITS:
			prev_count: u16 = count.__getitem__( usize( bits - 1 )).unwrap( 'bits-1 <= MAX_BITS' )
			code = ( code + u32( prev_count )) << 1
			first_code.__setitem__( usize( bits ), code ).unwrap( 'bits <= MAX_BITS' )
			bits += 1

	next_code: UnsafeList[u32] = UnsafeList[u32]( TABLE_SIZE )
	j: usize = 0
	with compiler.wrap_arithmetic:
		while j < TABLE_SIZE:
			next_code.append( first_code.__getitem__( j ).unwrap( 'j < TABLE_SIZE' )).unwrap( 'fixed TABLE_SIZE entries, never overflows' )
			j += 1

	codes: UnsafeList[u16] = UnsafeList[u16]( n )
	sym: usize = 0
	with compiler.panic_arithmetic( 'bounded by n, cannot overflow' ):
		while sym < n:
			length: u8 = code_lengths.__getitem__( sym ).unwrap( 'sym < n' )
			if length == 0:
				codes.append( u16( 0 )).unwrap( 'bounded by n, never overflows' )
			else:
				with compiler.wrap_arithmetic:
					len_idx: usize = usize( length )
					c: u32 = next_code.__getitem__( len_idx ).unwrap( 'len_idx <= MAX_BITS' )
					next_code.__setitem__( len_idx, c + 1 ).unwrap( 'len_idx <= MAX_BITS' )
				codes.append( u16( c )).unwrap( 'bounded by n, never overflows' )
			sym += 1
	return Result.Ok( codes )


# ---------------------------------------------------------------------------
# HuffmanEncoder
# ---------------------------------------------------------------------------

class HuffmanEncoder:
	__codes:   UnsafeList[u16]
	__lengths: UnsafeList[u8]

	def __init__( self, code_lengths: UnsafeList[u8] ) -> Result[None, HuffmanError]:
		self.__codes   = _canonical_codes_from_lengths( code_lengths ).or_return()
		self.__lengths = code_lengths
		return Result.Ok( None )

	def code_for( self, symbol: u16 ) -> Result[tuple[u16, u8], HuffmanError]:
		idx: usize = usize( symbol )
		length: u8 = self.__lengths.__getitem__( idx ).unwrap( 'symbol within alphabet' )
		if length == 0:
			return Result.Err( HuffmanError.NoSymbolMatched( None ))
		code: u16 = self.__codes.__getitem__( idx ).unwrap( 'symbol within alphabet' )
		return Result.Ok(( code, length ))

	# Writes `symbol`'s canonical code to `writer`. Huffman codes are
	# packed MOST-significant-bit-first (RFC 1951 SS3.1.1) - the opposite
	# of every other fixed-width DEFLATE field, which packs LSB-first (see
	# bitstream.BitWriter.write_bits) - so the code's bits are reversed
	# before handing them to the same LSB-first primitive.
	def write_symbol( self, writer: bitstream.BitWriter, symbol: u16 ) -> Result[None, HuffmanError]:
		code_and_len: tuple[u16, u8] = self.code_for( symbol ).or_return()
		code: u16 = code_and_len[0]
		length: u8 = code_and_len[1]
		reversed_code: u32 = 0
		c: u32 = u32( code )
		i: u8 = 0
		with compiler.wrap_arithmetic:
			while i < length:
				reversed_code = ( reversed_code << 1 ) | ( c & 1 )
				c >>= 1
				i += 1
		writer.write_bits( reversed_code, u32( length ))
		return Result.Ok( None )


# ---------------------------------------------------------------------------
# HuffmanDecoder
# ---------------------------------------------------------------------------

class HuffmanDecoder:
	__count:   UnsafeList[u16]  # count[len] = number of codes of that length, len 0..MAX_BITS
	__symbols: UnsafeList[u16]  # symbols sorted by (length, symbol value) - see __init__

	def __init__( self, code_lengths: UnsafeList[u8] ) -> Result[None, HuffmanError]:
		n: usize = len( code_lengths )
		count: UnsafeList[u16] = _count_per_length( code_lengths ).or_return()

		# offs[len] = starting index into __symbols for length `len`,
		# built as a running total of count[] (a standard counting sort) -
		# mutated into a cursor below as each symbol claims its slot
		offs: UnsafeList[u16] = UnsafeList[u16]( TABLE_SIZE )
		i: usize = 0
		with compiler.wrap_arithmetic:
			while i < TABLE_SIZE:
				offs.append( u16( 0 )).unwrap( 'fixed TABLE_SIZE entries, never overflows' )
				i += 1
		bits: u32 = 1
		with compiler.wrap_arithmetic:
			while bits < MAX_BITS:
				prev_offs: u16 = offs.__getitem__( usize( bits )).unwrap( 'bits <= MAX_BITS' )
				prev_count: u16 = count.__getitem__( usize( bits )).unwrap( 'bits <= MAX_BITS' )
				offs.__setitem__( usize( bits + 1 ), prev_offs + prev_count ).unwrap( 'bits+1 <= MAX_BITS' )
				bits += 1

		total_used: usize = 0
		with compiler.wrap_arithmetic:
			b2: u32 = 1
			while b2 <= MAX_BITS:
				total_used += usize( count.__getitem__( usize( b2 )).unwrap( 'b2 <= MAX_BITS' ))
				b2 += 1

		symbols: UnsafeList[u16] = UnsafeList[u16]( total_used if total_used > 0 else usize( 1 ))
		k: usize = 0
		with compiler.wrap_arithmetic:
			while k < total_used:
				symbols.append( u16( 0 )).unwrap( 'sized to total_used, never overflows' )
				k += 1

		sym: usize = 0
		with compiler.panic_arithmetic( 'bounded by n, cannot overflow' ):
			while sym < n:
				length: u8 = code_lengths.__getitem__( sym ).unwrap( 'sym < n' )
				if length > 0:
					len_idx: usize = usize( length )
					slot: u16 = offs.__getitem__( len_idx ).unwrap( 'len_idx <= MAX_BITS' )
					symbols.__setitem__( usize( slot ), u16( sym )).unwrap( 'slot < total_used by construction' )
					with compiler.wrap_arithmetic:
						offs.__setitem__( len_idx, slot + 1 ).unwrap( 'len_idx <= MAX_BITS' )
				sym += 1

		self.__count   = count
		self.__symbols = symbols
		return Result.Ok( None )

	# Reads one symbol from `reader`, one bit at a time - the standard
	# RFC 1951 "counting" decode algorithm (same shape as zlib's own
	# reference decoder, puff.c): accumulates `code = (code<<1)|bit` and
	# compares against the canonical [first_code[len], first_code[len]+
	# count[len]) range at each length, without ever materializing actual
	# code VALUES (only counts + a sorted symbol table) - correct for any
	# valid canonical assignment by construction, not just ones this
	# module itself built.
	def decode( self, reader: bitstream.BitReader ) -> Result[u16, HuffmanError]:
		code:  u32 = 0
		first: u32 = 0
		index: u32 = 0
		bits:  u32 = 1
		with compiler.wrap_arithmetic:
			while bits <= MAX_BITS:
				bit: u32 = 0
				match reader.read_bits( u32( 1 )):
					case Result.Ok( b ):
						bit = b
					case Result.Err( _ ):
						return Result.Err( HuffmanError.UnexpectedEndOfStream( None ))
				code |= bit
				cnt: u16 = self.__count.__getitem__( usize( bits )).unwrap( 'bits <= MAX_BITS' )
				with compiler.panic_arithmetic( 'code >= first always holds within a validated table' ):
					offset: u32 = code - first
				if offset < u32( cnt ):
					idx: usize = usize( index ) + usize( offset )
					sym: u16 = self.__symbols.__getitem__( idx ).unwrap( 'idx within symbols bounds by construction' )
					return Result.Ok( sym )
				index += u32( cnt )
				first += u32( cnt )
				first <<= 1
				code <<= 1
				bits += 1
		return Result.Err( HuffmanError.NoSymbolMatched( None ))


# ---------------------------------------------------------------------------
# build_code_lengths_from_frequencies - encoder-side only, for dynamic blocks
# ---------------------------------------------------------------------------

def build_code_lengths_from_frequencies( freqs: UnsafeList[u32], max_code_length: u8 ) -> Result[UnsafeList[u8], HuffmanError]:
	''' Builds a standard (non length-limited) Huffman tree from per-symbol
	frequencies and returns its code lengths. If the natural tree depth
	exceeds max_code_length (only possible with a heavily-skewed,
	Fibonacci-like frequency distribution - rare for real LZ77 token
	streams, but not impossible), returns Err(CodeTooLong) rather than
	attempting length-limiting: lib/deflate.py's compress() falls back to
	a FIXED Huffman block for that data in that case, which is always
	valid (hardcoded lengths, <=9 bits) - correctness over optimality,
	since DEFLATE never requires dynamic blocks to be used at all. '''
	n: usize = len( freqs )
	used: UnsafeList[u16] = UnsafeList[u16]()
	sym: usize = 0
	with compiler.panic_arithmetic( 'bounded by n, cannot overflow' ):
		while sym < n:
			f: u32 = freqs.__getitem__( sym ).unwrap( 'sym < n' )
			if f > 0:
				used.append( u16( sym )).unwrap( 'at most n symbols, n fits comfortably' )
			sym += 1

	lengths: UnsafeList[u8] = UnsafeList[u8]( n )
	zi: usize = 0
	with compiler.panic_arithmetic( 'bounded by n, cannot overflow' ):
		while zi < n:
			lengths.append( u8( 0 )).unwrap( 'bounded by n, never overflows' )
			zi += 1

	used_count: usize = len( used )
	if used_count == 0:
		return Result.Ok( lengths )
	if used_count == 1:
		# RFC 1951's degenerate single-symbol case
		only_sym: u16 = used.__getitem__( 0 ).unwrap( 'used_count == 1' )
		lengths.__setitem__( usize( only_sym ), u8( 1 )).unwrap( 'only_sym < n' )
		return Result.Ok( lengths )

	with compiler.panic_arithmetic( 'used_count >= 2, capacity fits comfortably within a usize' ):
		capacity: usize = 2 * used_count - 1

	weight:      UnsafeList[u32]  = UnsafeList[u32]( capacity )
	left_child:  UnsafeList[i32]  = UnsafeList[i32]( capacity )
	right_child: UnsafeList[i32]  = UnsafeList[i32]( capacity )
	active:      UnsafeList[bool] = UnsafeList[bool]( capacity )

	li: usize = 0
	with compiler.panic_arithmetic( 'bounded by used_count, cannot overflow' ):
		while li < used_count:
			leaf_sym: u16 = used.__getitem__( li ).unwrap( 'li < used_count' )
			leaf_freq: u32 = freqs.__getitem__( usize( leaf_sym )).unwrap( 'leaf_sym < n' )
			weight.append( leaf_freq ).unwrap( 'sized to capacity, never overflows' )
			left_child.append( i32( -1 )).unwrap( 'sized to capacity, never overflows' )
			right_child.append( i32( -1 )).unwrap( 'sized to capacity, never overflows' )
			active.append( True ).unwrap( 'sized to capacity, never overflows' )
			li += 1

	node_count: usize = used_count
	step: usize = 0
	with compiler.panic_arithmetic( 'bounded by used_count, cannot overflow' ):
		while step < used_count - 1:
			a: i32 = -1
			b: i32 = -1
			a_w: u32 = 0xFFFFFFFF
			b_w: u32 = 0xFFFFFFFF
			scan: usize = 0
			while scan < node_count:
				if active.__getitem__( scan ).unwrap( 'scan < node_count' ):
					w: u32 = weight.__getitem__( scan ).unwrap( 'scan < node_count' )
					if w < a_w:
						b = a
						b_w = a_w
						a = i32( scan )
						a_w = w
					elif w < b_w:
						b = i32( scan )
						b_w = w
				scan += 1
			active.__setitem__( usize( a ), False ).unwrap( 'a valid index' )
			active.__setitem__( usize( b ), False ).unwrap( 'b valid index' )
			with compiler.wrap_arithmetic:
				new_weight: u32 = a_w + b_w
			weight.append( new_weight ).unwrap( 'sized to capacity, never overflows' )
			left_child.append( a ).unwrap( 'sized to capacity, never overflows' )
			right_child.append( b ).unwrap( 'sized to capacity, never overflows' )
			active.append( True ).unwrap( 'sized to capacity, never overflows' )
			node_count += 1
			step += 1

	with compiler.panic_arithmetic( 'node_count == capacity, both derived from used_count' ):
		root: usize = node_count - 1

	depth: UnsafeList[u8] = UnsafeList[u8]( capacity )
	di: usize = 0
	with compiler.panic_arithmetic( 'bounded by capacity, cannot overflow' ):
		while di < capacity:
			depth.append( u8( 0 )).unwrap( 'sized to capacity, never overflows' )
			di += 1

	node_stack:  UnsafeList[i32] = UnsafeList[i32]()
	depth_stack: UnsafeList[u8]  = UnsafeList[u8]()
	with compiler.panic_arithmetic( 'root < capacity <= 2*used_count-1, well within i32 range' ):
		node_stack.append( i32( root )).unwrap( 'x' )
	depth_stack.append( u8( 0 )).unwrap( 'x' )
	max_depth: u8 = 0
	while len( node_stack ) > 0:
		with compiler.panic_arithmetic( 'len > 0, just checked' ):
			last: usize = len( node_stack ) - 1
		idx: i32 = node_stack.__getitem__( last ).unwrap( 'last valid' )
		d: u8 = depth_stack.__getitem__( last ).unwrap( 'last valid' )
		node_stack.erase_at( last ).unwrap( 'last valid' )
		depth_stack.erase_at( last ).unwrap( 'last valid' )
		lc: i32 = left_child.__getitem__( usize( idx )).unwrap( 'idx valid' )
		rc: i32 = right_child.__getitem__( usize( idx )).unwrap( 'idx valid' )
		if lc == -1:
			depth.__setitem__( usize( idx ), d ).unwrap( 'idx valid' )
			if d > max_depth:
				max_depth = d
		else:
			with compiler.wrap_arithmetic:
				child_depth: u8 = d + 1
			node_stack.append( lc ).unwrap( 'x' )
			depth_stack.append( child_depth ).unwrap( 'x' )
			node_stack.append( rc ).unwrap( 'x' )
			depth_stack.append( child_depth ).unwrap( 'x' )

	if max_depth > max_code_length:
		return Result.Err( HuffmanError.CodeTooLong( None ))

	fi: usize = 0
	with compiler.panic_arithmetic( 'bounded by used_count, cannot overflow' ):
		while fi < used_count:
			leaf_sym2: u16 = used.__getitem__( fi ).unwrap( 'fi < used_count' )
			leaf_depth: u8 = depth.__getitem__( fi ).unwrap( 'fi < used_count, leaves are indices [0,used_count)' )
			lengths.__setitem__( usize( leaf_sym2 ), leaf_depth ).unwrap( 'leaf_sym2 < n' )
			fi += 1

	return Result.Ok( lengths )
