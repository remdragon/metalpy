import sys
from . import Codec, CodecError, DecodeErrors, _emit_lossy_unit

class Utf16( Codec ):
	@virtual
	def names(self) -> list[str]:
		return [
			'utf-16', 'UTF-16', 'utf16', 'UTF16'
		]

	@virtual
	def encode( self, s: str ) -> Result[bytes, CodecError]:
		s_len: usize = s.byte_len()
		s_ptr: ConstPtr[u8] = s.get_const_ptr()

		# Worst-case: every input byte is ASCII -> 2 bytes per byte.
		with compiler.panic_arithmetic('a real string can never be within 2x of usize::MAX bytes'):
			out = bytearray(s_len * 2)
		# Cast to u16 pointer – writes are native endianness.
		out_u16: Ptr[u16] = compiler.cast(Ptr[u16], out.get_ptr())

		out_idx: usize = 0   # number of u16 units written
		in_idx: usize = 0

		with compiler.panic_arithmetic('overflow impossible with bounds checks'):
			while in_idx < s_len:
				b0: u8 = s_ptr[in_idx]

				# Decode UTF-8 sequence to codepoint.
				if b0 <= 0x7F:
					cp: u32 = b0
					bytes_read: usize = 1
				else:
					if (b0 & 0xE0) == 0xC0:          # 2-byte
						if in_idx + 1 >= s_len:
							return Result.Err(CodecError('utf-16le',
								'Truncated UTF-8 sequence'))
						cp = (u32(b0 & 0x1F) << 6) | (s_ptr[in_idx + 1] & 0x3F)
						bytes_read = 2
					elif (b0 & 0xF0) == 0xE0:        # 3-byte
						if in_idx + 2 >= s_len:
							return Result.Err(CodecError('utf-16le',
								'Truncated UTF-8 sequence'))
						cp = (u32(b0 & 0x0F) << 12) | \
							(u32(s_ptr[in_idx + 1] & 0x3F) << 6) | \
							(s_ptr[in_idx + 2] & 0x3F)
						bytes_read = 3
					elif (b0 & 0xF8) == 0xF0:        # 4-byte
						if in_idx + 3 >= s_len:
							return Result.Err(CodecError('utf-16le',
								'Truncated UTF-8 sequence'))
						cp = (u32(b0 & 0x07) << 18) | \
							(u32(s_ptr[in_idx + 1] & 0x3F) << 12) | \
							(u32(s_ptr[in_idx + 2] & 0x3F) << 6) | \
							(s_ptr[in_idx + 3] & 0x3F)
						bytes_read = 4
					else:
						return Result.Err(CodecError('utf-16le',
							'Invalid UTF-8 sequence'))

				# Write codepoint as UTF‑16 (native endianness).
				if cp <= 0xFFFF:
					out_u16[out_idx] = u16(cp)
					out_idx += 1
				else:
					# Surrogate pair.
					cp -= 0x10000
					high: u16 = 0xD800 | u16(cp >> 10)
					low:  u16 = 0xDC00 | u16(cp & 0x3FF)
					out_u16[out_idx]     = high
					out_u16[out_idx + 1] = low
					out_idx += 2

				in_idx += bytes_read

		# Shrink to exact byte length.
		with compiler.panic_arithmetic('out_idx counts u16 units of out, which is already 2x-sized'):
			byte_len: usize = out_idx * 2
		final = bytearray(byte_len)
		sys.memcpy(final.get_ptr(), out.get_ptr(), byte_len)
		return Result.Ok(bytes.from_bytearray(move(final)))

	@virtual
	def decode(self, b: bytes | bytearray) -> Result[str, CodecError]:
		b_len: usize = len(b)
		with compiler.panic_arithmetic('divisor is the literal 2, never zero'):
			is_odd: usize = b_len % 2
		if is_odd != 0:
			return Result.Err(CodecError('utf-16le',
				'UTF-16 data must have even length'))
		b_ptr: ConstPtr[u8] = b.get_const_ptr()
		# Cast input to u16* – reads are native endianness.
		in_u16: ConstPtr[u16] = compiler.cast(ConstPtr[u16], b_ptr)
		with compiler.panic_arithmetic('divisor is the literal 2, never zero'):
			num_units: usize = b_len // 2

		# Worst-case: each u16 unit may become up to 3 UTF-8 bytes.
		with compiler.panic_arithmetic('irrational byte length'):
			out = bytearray(num_units * 3)
		out_ptr: Ptr[u8] = out.get_ptr()

		out_idx: usize = 0
		i: usize = 0

		with compiler.panic_arithmetic('bounded by num_units'):
			while i < num_units:
				cp: u32 = in_u16[i]
				i += 1

				# Surrogate pair detection (native endianness).
				if cp >= 0xD800 and cp <= 0xDBFF:   # High surrogate
					if i >= num_units:
						return Result.Err(CodecError('utf-16le',
							'Incomplete surrogate pair'))
					low_surr: u32 = in_u16[i]
					i += 1
					if not ( low_surr >= 0xDC00 and low_surr <= 0xDFFF ):
						return Result.Err(CodecError('utf-16le',
							'Invalid low surrogate'))
					cp = 0x10000 + ((cp - 0xD800) << 10) + (low_surr - 0xDC00)
				elif cp >= 0xDC00 and cp <= 0xDFFF:   # Low surrogate without high
					return Result.Err(CodecError('utf-16le',
						'Unexpected low surrogate'))

				# Encode codepoint to UTF-8.
				if cp <= 0x7F:
					out_ptr[out_idx] = u8(cp)
					out_idx += 1
				elif cp <= 0x7FF:
					out_ptr[out_idx]     = 0xC0 | u8(cp >> 6)
					out_ptr[out_idx + 1] = 0x80 | u8(cp & 0x3F)
					out_idx += 2
				elif cp <= 0xFFFF:
					out_ptr[out_idx]     = 0xE0 | u8(cp >> 12)
					out_ptr[out_idx + 1] = 0x80 | u8((cp >> 6) & 0x3F)
					out_ptr[out_idx + 2] = 0x80 | u8(cp & 0x3F)
					out_idx += 3
				else:   # cp > 0xFFFF
					out_ptr[out_idx]     = 0xF0 | u8(cp >> 18)
					out_ptr[out_idx + 1] = 0x80 | u8((cp >> 12) & 0x3F)
					out_ptr[out_idx + 2] = 0x80 | u8((cp >> 6) & 0x3F)
					out_ptr[out_idx + 3] = 0x80 | u8(cp & 0x3F)
					out_idx += 4

		# Create null-terminated C string.
		with compiler.panic_arithmetic('irrational byte length'):
			buf_size: usize = out_idx + 1
		new_buf: Ptr[u8] = sys.alloc[u8](buf_size)
		sys.memcpy(new_buf, out_ptr, out_idx)
		new_buf[out_idx] = 0
		return str._from_owned_cstr(new_buf, buf_size)

	@virtual
	def decode_lossy(self, b: bytes | bytearray, errors: DecodeErrors = DecodeErrors.BackslashReplace) -> str:
		b_len: usize = len(b)
		b_ptr: ConstPtr[u8] = b.get_const_ptr()
		in_u16: ConstPtr[u16] = compiler.cast(ConstPtr[u16], b_ptr)
		with compiler.panic_arithmetic('divisor is the literal 2, never zero'):
			num_units: usize = b_len // 2

		# worst case: every byte is malformed and backslash-escaped (4 out bytes each)
		with compiler.panic_arithmetic('irrational byte length'):
			out = bytearray(b_len * 4)
		out_ptr: Ptr[u8] = out.get_ptr()
		out_idx: usize = 0

		i: usize = 0
		with compiler.panic_arithmetic('bounded by num_units'):
			while i < num_units:
				cp: u32 = in_u16[i]
				unit_idx: usize = i
				i += 1

				if cp >= 0xD800 and cp <= 0xDBFF:   # High surrogate
					if i >= num_units:
						out_idx = _emit_lossy_code_unit(out_ptr, out_idx, in_u16[unit_idx], errors)
						continue
					low_surr: u32 = in_u16[i]
					if not (low_surr >= 0xDC00 and low_surr <= 0xDFFF):
						out_idx = _emit_lossy_code_unit(out_ptr, out_idx, in_u16[unit_idx], errors)
						continue
					i += 1
					cp = 0x10000 + ((cp - 0xD800) << 10) + (low_surr - 0xDC00)
				elif cp >= 0xDC00 and cp <= 0xDFFF:   # Low surrogate without high
					out_idx = _emit_lossy_code_unit(out_ptr, out_idx, in_u16[unit_idx], errors)
					continue

				if cp <= 0x7F:
					out_ptr[out_idx] = u8(cp)
					out_idx += 1
				elif cp <= 0x7FF:
					out_ptr[out_idx]     = 0xC0 | u8(cp >> 6)
					out_ptr[out_idx + 1] = 0x80 | u8(cp & 0x3F)
					out_idx += 2
				elif cp <= 0xFFFF:
					out_ptr[out_idx]     = 0xE0 | u8(cp >> 12)
					out_ptr[out_idx + 1] = 0x80 | u8((cp >> 6) & 0x3F)
					out_ptr[out_idx + 2] = 0x80 | u8(cp & 0x3F)
					out_idx += 3
				else:
					out_ptr[out_idx]     = 0xF0 | u8(cp >> 18)
					out_ptr[out_idx + 1] = 0x80 | u8((cp >> 12) & 0x3F)
					out_ptr[out_idx + 2] = 0x80 | u8((cp >> 6) & 0x3F)
					out_ptr[out_idx + 3] = 0x80 | u8(cp & 0x3F)
					out_idx += 4

		with compiler.panic_arithmetic('divisor is the literal 2, never zero'):
			is_odd: usize = b_len % 2
		if is_odd != 0:   # trailing odd byte has no partner code unit
			with compiler.panic_arithmetic('is_odd != 0 implies b_len >= 1'):
				out_idx = _emit_lossy_unit(out_ptr, out_idx, b_ptr[b_len - 1], errors)

		with compiler.panic_arithmetic('irrational byte length'):
			buf_size: usize = out_idx + 1
		new_buf: Ptr[u8] = sys.alloc[u8](buf_size)
		sys.memcpy(new_buf, out_ptr, out_idx)
		new_buf[out_idx] = 0
		return str._from_owned_cstr(new_buf, buf_size).unwrap('decode_lossy always produces valid utf-8 by construction')

def _emit_lossy_code_unit(out_ptr: Ptr[u8], out_idx: usize, unit: u16, errors: DecodeErrors) -> usize:
	# a bad UTF-16 code unit is 2 raw bytes - apply the per-byte policy to
	# each (native-endian split, matching decode()'s own u16* cast)
	with compiler.panic_arithmetic('masked/shifted into u8 range, cannot overflow'):
		out_idx = _emit_lossy_unit(out_ptr, out_idx, u8(unit & 0xFF), errors)
		out_idx = _emit_lossy_unit(out_ptr, out_idx, u8(unit >> 8), errors)
	return out_idx

utf16: Utf16 = Utf16()
