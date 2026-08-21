# crc32 - table-driven CRC-32 (IEEE 802.3 / reflected polynomial 0xEDB88320),
# the checksum ZIP's local/central-directory headers require per entry.

import compiler


def _build_table() -> UnsafeList[u32]:
	table: UnsafeList[u32] = UnsafeList[u32]( usize( 256 ))
	i: u32 = 0
	with compiler.wrap_arithmetic:
		while i < 256:
			v: u32 = i
			bit: i32 = 0
			while bit < 8:
				if v & 1 != 0:
					v = ( v >> 1 ) ^ 0xEDB88320
				else:
					v = v >> 1
				bit += 1
			table.append( v ).unwrap( 'fixed 256-entry table, never overflows' )
			i += 1
	return table


_TABLE: UnsafeList[u32] = _build_table()


def crc32( data: bytes|bytearray, initial: u32 = 0 ) -> u32:
	n: usize = len( data )
	in_ptr: ConstPtr[u8] = data.get_const_ptr()

	with compiler.wrap_arithmetic:
		crc: u32 = initial ^ 0xFFFFFFFF
		i: usize = 0
		while i < n:
			index: usize = usize(( crc ^ u32( in_ptr[i] )) & 0xFF )
			crc = _TABLE.__getitem__( index ).unwrap( 'index always < 256' ) ^ ( crc >> 8 )
			i += 1
		return crc ^ 0xFFFFFFFF
