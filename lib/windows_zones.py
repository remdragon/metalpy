# lib/windows_zones.py — optional Windows-zone-name <-> IANA-zone-name
# translation, embedded from CLDR's windowsZones.xml.
#
# Importing this module by itself does nothing; call install() once (e.g.
# at the top of main()) to fetch/embed the table and point windows_zone_map
# at it - mirrors lib/case_folding.py exactly (see its own comment for the
# full "opt-in via reachability" rationale: a program that never imports
# this module never references the table at all, so it's never linked in).
#
# This table carries ONLY name translation, no DST/offset rule data -
# ZoneInfo('America/New_York') gets its actual rules from the OS either way
# (see lib/windows/zoneinfo_rules.py); windows_zones.install() only changes
# whether an IANA-style key can be used at all on Windows, vs. requiring the
# native Windows zone name (e.g. 'Eastern Standard Time') directly.

import compiler
import sys

@cstruct
class WindowsZoneMap:
	# zero-initialized by default (table=None, table_len=0) - "not
	# installed" needs no init call, same reasoning as CaseFolding's own
	# cstruct-not-RCClass choice (lib/builtins/__init__.py's case_folder
	# comment: a global RCClass's own init function is never actually
	# wired up to run, so it would stay a null pointer forever).
	#
	# table format: a sequence of length-prefixed variable-width records
	# packed back-to-back until table_len bytes are consumed (no separate
	# count - unlike CaseFolding's fixed-8-bytes-per-entry table, entries
	# here are variable length, so "read until you hit the end" is simpler
	# than maintaining a redundant count):
	#   [u16 win_name_len LE][win_name UTF-8 bytes]
	#   [u16 iana_name_len LE][iana_name UTF-8 bytes]
	# repeated. ~150 entries total (CLDR's windowsZones.xml, territory="001"
	# entries only) - small enough that a linear scan at lookup time is the
	# right tradeoff over sorting + binary search.
	table: ConstPtr[u8]
	table_len: usize

	def to_iana( self, win_name: str ) -> str|None:
		return self._lookup( win_name, want_iana = True )

	def to_windows( self, iana_name: str ) -> str|None:
		return self._lookup( iana_name, want_iana = False )

	@private
	def _lookup( self, needle: str, want_iana: bool ) -> str|None:
		if self.table_len == 0:
			return None
		offset: usize = 0
		while offset < self.table_len:
			with compiler.wrap_arithmetic:
				win_len: usize = usize( WindowsZoneMap._read_u16_le( self.table, offset ) )
				win_start: usize = offset + 2
				win_end: usize = win_start + win_len
				iana_len: usize = usize( WindowsZoneMap._read_u16_le( self.table, win_end ) )
				iana_start: usize = win_end + 2
				iana_end: usize = iana_start + iana_len

			if want_iana:
				win_name: str = WindowsZoneMap._decode_utf8_range( self.table, win_start, win_len )
				if win_name == needle:
					return WindowsZoneMap._decode_utf8_range( self.table, iana_start, iana_len )
			else:
				iana_name: str = WindowsZoneMap._decode_utf8_range( self.table, iana_start, iana_len )
				if iana_name == needle:
					return WindowsZoneMap._decode_utf8_range( self.table, win_start, win_len )

			offset = iana_end
		return None

	@private
	@staticmethod
	def _read_u16_le( data: ConstPtr[u8], i: usize ) -> u16:
		with compiler.wrap_arithmetic:
			return u16( data[i] ) | ( u16( data[i+1] ) << 8 )

	@private
	@staticmethod
	def _decode_utf8_range( data: ConstPtr[u8], start: usize, length: usize ) -> str:
		with compiler.wrap_arithmetic:
			buf_size: usize = length + 1
		buf: Ptr[u8] = sys.alloc[u8]( buf_size )
		i: usize = 0
		with compiler.wrap_arithmetic:
			while i < length:
				buf[i] = data[start + i]
				i += 1
			buf[length] = 0
		return str._from_owned_cstr( buf, buf_size ).unwrap( 'windows_zones table: invalid UTF-8 (corrupt embedded table)' )


windows_zone_map: WindowsZoneMap = WindowsZoneMap( table = None, table_len = 0 )


def install() -> None:
	table: bytes = compiler.fetch_windows_zones_table()
	windows_zone_map.table_len = len( table )
	windows_zone_map.table = table.get_const_ptr()


def to_iana( win_name: str ) -> str|None:
	return windows_zone_map.to_iana( win_name )

def to_windows( iana_name: str ) -> str|None:
	return windows_zone_map.to_windows( iana_name )
