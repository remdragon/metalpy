# src/posix/zoneinfo_rules.py
#
# Parses the real on-disk TZif binary file at /usr/share/zoneinfo/<key> -
# the same mechanism CPython's own zoneinfo module uses by default. No
# network, no embedding: this file ships with virtually every POSIX system,
# so this is the "ask the OS" path for real DST/offset rules on POSIX.
#
# TZif format (RFC 8536 / tzfile(5)): a legacy 32-bit-time_t header+data
# block, optionally followed (version byte '2' or '3') by an authoritative
# 64-bit-time_t header+data block in the same shape. This parser reads the
# 64-bit block when present (skipping straight past the 32-bit one, which
# exists only for old 32-bit-time_t readers) and falls back to the 32-bit
# block for the rare version-0 file. The trailing POSIX-TZ-string footer
# (for extrapolating past the last explicit transition) is NOT parsed here -
# every date within the file's explicit transition range (which for any
# current tzdata release extends years into the future) resolves correctly
# without it; this is a documented v1 scope cut, not an oversight.

import compiler
import sys
from zoneinfo import ZoneInfo, TTInfo

_ZONEINFO_DIR = '/usr/share/zoneinfo/'
_HEADER_SIZE: usize = 44  # 4 magic + 1 version + 15 reserved + 6x4 counts
_MAX_TZIF_SIZE: usize = 65536


def load_zone( zone: ZoneInfo, key: str ) -> None:
	_validate_key( key )
	path: str = _ZONEINFO_DIR + key
	data: bytes = _read_whole_file( path )
	_parse_tzif( zone, data )


def _validate_key( key: str ) -> None:
	''' key becomes a filesystem path (/usr/share/zoneinfo/<key>) - reject
	anything that could escape that directory before ever touching the
	filesystem. IANA zone keys are always relative, forward-slash-separated
	path segments (e.g. 'America/New_York') - a leading '/' or any '..'
	segment is never legitimate. '''
	if key.byte_len() == 0:
		sys.panic( 'zoneinfo: invalid key (empty)' )
	if key.startswith( '/' ):
		sys.panic( 'zoneinfo: invalid key (must be relative, not absolute)' )
	if key.find( '..' ) != isize( -1 ):
		sys.panic( "zoneinfo: invalid key (must not contain '..')" )


def _read_whole_file( path: str ) -> bytes:
	# TZif files are always small (a few hundred bytes to a handful of KB,
	# even for the zones with the most historical transitions) - one
	# sufficiently large fixed read, same "guess a generous max size" shape
	# posix/time.py's own _read_etc_timezone_file already uses (just a much
	# bigger buffer, since this needs to hold a whole binary rule table, not
	# one line of text).
	match File.binary_reader( path ):
		case Result.Ok( reader ):
			buf: bytearray = bytearray( _MAX_TZIF_SIZE )
			match reader.read( buf.get_ptr(), _MAX_TZIF_SIZE ):
				case Result.Ok( n ):
					if n == 0:
						sys.panic( 'zoneinfo: empty tzdata file: ' + path )
					if n == _MAX_TZIF_SIZE:
						sys.panic( 'zoneinfo: tzdata file unexpectedly large (>64KB): ' + path )
					sliced: bytearray = buf[:n]
					return bytes.from_bytearray( move( sliced ))
				case Result.Err( _ ):
					sys.panic( 'zoneinfo: failed to read tzdata file: ' + path )
		case Result.Err( _ ):
			sys.panic( 'zoneinfo: no tzdata file for key (not a recognized IANA zone?): ' + path )


# --- big-endian primitive reads (TZif is network-byte-order throughout) ----

def _read_u32_be( data: ConstPtr[u8], i: usize ) -> u32:
	with compiler.wrap_arithmetic:
		return ( u32( data[i] ) << 24 ) | ( u32( data[i+1] ) << 16 ) | ( u32( data[i+2] ) << 8 ) | u32( data[i+3] )

def _read_i32_be( data: ConstPtr[u8], i: usize ) -> i32:
	with compiler.wrap_arithmetic:
		return compiler.cast( i32, _read_u32_be( data, i ))

def _read_i64_be( data: ConstPtr[u8], i: usize ) -> i64:
	with compiler.wrap_arithmetic:
		hi: u64 = u64( _read_u32_be( data, i ))
		lo: u64 = u64( _read_u32_be( data, i + 4 ))
		combined: u64 = ( hi << 32 ) | lo
		return compiler.cast( i64, combined )

def _read_cstr_at( pool: ConstPtr[u8], pool_len: usize, start: usize ) -> str:
	''' NUL-terminated string starting at byte offset `start` within a
	charcnt-byte designation-string pool, stopping at pool_len even if no
	NUL is found first (a malformed/truncated file shouldn't read past the
	buffer this pointer actually owns). '''
	end: usize = start
	with compiler.wrap_arithmetic:
		while end < pool_len and pool[end] != 0:
			end += 1
		length: usize = end - start
		buf_size: usize = length + 1
	buf: Ptr[u8] = sys.alloc[u8]( buf_size )
	i: usize = 0
	with compiler.wrap_arithmetic:
		while i < length:
			buf[i] = pool[start + i]
			i += 1
		buf[length] = 0
	return str._from_owned_cstr( buf, buf_size ).unwrap( 'zoneinfo: invalid UTF-8 in tzdata designation string' )


class _TzifHeader:
	version: u8
	isutcnt: usize
	isstdcnt: usize
	leapcnt: usize
	timecnt: usize
	typecnt: usize
	charcnt: usize

	def __init__(
		self,
		version: u8,
		isutcnt: usize, isstdcnt: usize, leapcnt: usize,
		timecnt: usize, typecnt: usize, charcnt: usize,
	) -> None:
		self.version = version
		self.isutcnt = isutcnt
		self.isstdcnt = isstdcnt
		self.leapcnt = leapcnt
		self.timecnt = timecnt
		self.typecnt = typecnt
		self.charcnt = charcnt


class _TzifBlock:
	''' one parsed 32-bit-or-64-bit TZif data block. '''
	timecnt: usize
	typecnt: usize
	transition_times: list[i64]
	transition_type_idx: list[u8]
	ttinfos: list[TTInfo]
	next_offset: usize  # byte offset just past this block (start of whatever follows)

	def __init__(
		self,
		timecnt: usize,
		typecnt: usize,
		transition_times: list[i64],
		transition_type_idx: list[u8],
		ttinfos: list[TTInfo],
		next_offset: usize,
	) -> None:
		self.timecnt = timecnt
		self.typecnt = typecnt
		self.transition_times = transition_times
		self.transition_type_idx = transition_type_idx
		self.ttinfos = ttinfos
		self.next_offset = next_offset


def _parse_header( data: ConstPtr[u8], data_len: usize, offset: usize, path_for_errors: str ) -> _TzifHeader:
	with compiler.panic_arithmetic( 'zoneinfo: tzdata file truncated (no room for header)' ):
		header_end: usize = offset + _HEADER_SIZE
	if header_end > data_len:
		sys.panic( 'zoneinfo: tzdata file truncated (header): ' + path_for_errors )
	with compiler.wrap_arithmetic:
		magic_ok: bool = ( data[offset] == 0x54 and data[offset+1] == 0x5A and data[offset+2] == 0x69 and data[offset+3] == 0x66 )  # "TZif"
	if not magic_ok:
		sys.panic( 'zoneinfo: not a TZif file (bad magic): ' + path_for_errors )
	with compiler.wrap_arithmetic:
		version_offset: usize = offset + 4
	version: u8 = data[version_offset]
	with compiler.wrap_arithmetic:
		isutcnt: usize = usize( _read_u32_be( data, offset + 20 ))
		isstdcnt: usize = usize( _read_u32_be( data, offset + 24 ))
		leapcnt: usize = usize( _read_u32_be( data, offset + 28 ))
		timecnt: usize = usize( _read_u32_be( data, offset + 32 ))
		typecnt: usize = usize( _read_u32_be( data, offset + 36 ))
		charcnt: usize = usize( _read_u32_be( data, offset + 40 ))
	return _TzifHeader(
		version = version,
		isutcnt = isutcnt, isstdcnt = isstdcnt, leapcnt = leapcnt,
		timecnt = timecnt, typecnt = typecnt, charcnt = charcnt,
	)


def _parse_data_block(
	data: ConstPtr[u8],
	data_len: usize,
	offset: usize,
	header: _TzifHeader,
	time_width: usize,  # 4 for the legacy 32-bit block, 8 for the 64-bit block
	path_for_errors: str,
) -> _TzifBlock:
	timecnt: usize = header.timecnt
	typecnt: usize = header.typecnt
	charcnt: usize = header.charcnt
	pos: usize = offset

	transition_times: list[i64] = []
	i: usize = 0
	while i < timecnt:
		with compiler.panic_arithmetic( 'zoneinfo: tzdata file truncated (transition times)' ):
			entry_off: usize = pos + i * time_width
		t: i64
		if time_width == 8:
			t = _read_i64_be( data, entry_off )
		else:
			with compiler.wrap_arithmetic:
				t = i64( _read_i32_be( data, entry_off ))
		transition_times.append( t )
		with compiler.wrap_arithmetic:
			i += 1
	with compiler.wrap_arithmetic:
		pos += timecnt * time_width

	transition_type_idx: list[u8] = []
	i = 0
	while i < timecnt:
		with compiler.panic_arithmetic( 'zoneinfo: tzdata file truncated (transition types)' ):
			idx_off: usize = pos + i
		transition_type_idx.append( data[idx_off] )
		with compiler.wrap_arithmetic:
			i += 1
	with compiler.wrap_arithmetic:
		pos += timecnt

	# ttinfo records: 6 bytes each (i32 BE utoff, u8 isdst, u8 desigidx)
	pool_offset: usize
	with compiler.wrap_arithmetic:
		pool_offset = pos + typecnt * 6
	pool_end: usize
	with compiler.wrap_arithmetic:
		pool_end = pool_offset + charcnt
	if pool_end > data_len:
		sys.panic( 'zoneinfo: tzdata file truncated (ttinfo/designations): ' + path_for_errors )

	ttinfos: list[TTInfo] = []
	i = 0
	while i < typecnt:
		with compiler.panic_arithmetic( 'zoneinfo: tzdata file truncated (ttinfo)' ):
			rec_off: usize = pos + i * 6
		utoff: i32 = _read_i32_be( data, rec_off )
		with compiler.wrap_arithmetic:
			isdst_off: usize = rec_off + 4
			desigidx_off: usize = rec_off + 5
		isdst: bool = data[isdst_off] != 0
		desigidx: u8 = data[desigidx_off]
		with compiler.wrap_arithmetic:
			desig_start: usize = pool_offset + usize( desigidx )
		abbr: str = _read_cstr_at( data, pool_end, desig_start )
		ttinfos.append( TTInfo( utcoffset = utoff, is_dst = isdst, abbr = abbr ))
		with compiler.wrap_arithmetic:
			i += 1
	pos = pool_end

	# leap-second records: (time_width + 4) bytes each (occur, corr) - not
	# needed for offset/abbr queries, skipped over.
	with compiler.wrap_arithmetic:
		pos += header.leapcnt * ( time_width + 4 )
	# standard/wall and UT/local indicators: 1 byte each - also unused.
	with compiler.wrap_arithmetic:
		pos += header.isstdcnt + header.isutcnt

	if pos > data_len:
		sys.panic( 'zoneinfo: tzdata file truncated (trailing indicators): ' + path_for_errors )

	return _TzifBlock(
		timecnt = timecnt,
		typecnt = typecnt,
		transition_times = transition_times,
		transition_type_idx = transition_type_idx,
		ttinfos = ttinfos,
		next_offset = pos,
	)


def _default_rule( block: _TzifBlock ) -> TTInfo:
	''' the rule in effect before the first explicit transition - per
	tzfile(5): the first non-DST ttinfo, or ttinfo[0] if every type is DST
	(or there are no transitions at all, e.g. a fixed-offset zone like
	Etc/UTC). '''
	i: usize = 0
	while i < block.typecnt:
		ttinfo: TTInfo = block.ttinfos.__getitem__( i ).unwrap( 'zoneinfo: index in bounds by construction' )
		if not ttinfo.is_dst:
			return ttinfo
		with compiler.wrap_arithmetic:
			i += 1
	return block.ttinfos.__getitem__( 0 ).unwrap( 'zoneinfo: tzdata file has zero ttinfo records' )


def _parse_tzif( zone: ZoneInfo, data: bytes ) -> None:
	data_len: usize = len( data )
	raw: ConstPtr[u8] = data.get_const_ptr()

	v1_header: _TzifHeader = _parse_header( raw, data_len, 0, zone.name )
	v1_block: _TzifBlock = _parse_data_block( raw, data_len, _HEADER_SIZE, v1_header, 4, zone.name )

	block: _TzifBlock = v1_block
	if v1_header.version != 0:
		# a '2'/'3' file repeats the header+data block using 64-bit
		# transition times right after the 32-bit block just parsed - THAT
		# one is authoritative (RFC 8536 S3.2), the 32-bit block above
		# exists only so pre-2005 32-bit-time_t readers still get something
		# usable.
		v2_header: _TzifHeader = _parse_header( raw, data_len, v1_block.next_offset, zone.name )
		with compiler.wrap_arithmetic:
			v2_data_offset: usize = v1_block.next_offset + _HEADER_SIZE
		block = _parse_data_block( raw, data_len, v2_data_offset, v2_header, 8, zone.name )

	zone.default_rule = _default_rule( block )

	transition_times: list[i64] = []
	transition_rules: list[TTInfo] = []
	i: usize = 0
	while i < block.timecnt:
		t: i64 = block.transition_times.__getitem__( i ).unwrap( 'zoneinfo: index in bounds by construction' )
		type_idx: u8 = block.transition_type_idx.__getitem__( i ).unwrap( 'zoneinfo: index in bounds by construction' )
		with compiler.wrap_arithmetic:
			type_idx_usize: usize = usize( type_idx )
		rule: TTInfo = block.ttinfos.__getitem__( type_idx_usize ).unwrap( 'zoneinfo: transition type index out of range (malformed tzdata)' )
		transition_times.append( t )
		transition_rules.append( rule )
		with compiler.wrap_arithmetic:
			i += 1

	zone.transition_times = transition_times
	zone.transition_rules = transition_rules
