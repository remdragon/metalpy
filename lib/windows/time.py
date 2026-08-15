# src/windows/time.py

import compiler
import sys


def _decode_ascii_utf16z( ptr: Ptr[u16], max_len: usize ) -> str:
	''' decode up to max_len UTF-16 code units starting at ptr, stopping at
	the first zero code unit (Win32's usual nul-terminated-within-a-fixed-
	buffer convention). ASCII-only, not a general UTF-16 decoder: lib/codecs
	has no utf16 module (codecs.utf16 was referenced here before but never
	existed anywhere in the repo), and every real Windows time zone display/
	key name (e.g. 'Eastern Standard Time') is plain ASCII in practice, so a
	general surrogate-pair-aware decoder isn't needed for this narrow input. '''
	n: usize = 0
	with compiler.wrap_arithmetic:
		while n < max_len and ptr[n] != 0:
			n += 1
		buf_size: usize = n + 1
	buf: Ptr[u8] = sys.alloc[u8]( buf_size )
	i: usize = 0
	with compiler.wrap_arithmetic:
		while i < n:
			buf[i] = u8( ptr[i] )  # truncates to the low byte - safe for real ASCII input
			i += 1
		buf[n] = 0
	return str._from_owned_cstr( buf, buf_size ).unwrap( 'windows zone name: unexpectedly non-ASCII' )


def _field_ptr_u16( struct_ptr: Ptr[None], byte_offset: usize ) -> Ptr[u16]:
	''' compiler.addrof(...) only accepts a bare local-variable name, not a
	field expression (e.g. compiler.addrof(tz_info.TimeZoneKeyName_0) is a
	hard compile error - verified directly this session), so there is no
	direct way to get a pointer AT one of DynamicTimeZoneInformation's
	decomposed u16 fields. Callers instead already hold a Ptr[None] to the
	whole struct (see get_local_timezone_name's own tz_info, which is
	heap-allocated rather than a bare local for an unrelated reason - see
	its own comment) - this just advances it by byte_offset. See kernel32.
	py's own _TZNAME_OFFSET/_TZKEYNAME_OFFSET comment for how each offset
	is derived and kept in sync with the struct. '''
	raw: Ptr[u8] = compiler.cast( Ptr[u8], struct_ptr )
	with compiler.wrap_arithmetic:
		field_bytes: Ptr[u8] = raw + byte_offset
	return compiler.cast( Ptr[u16], field_bytes )


def get_local_timezone_name() -> str:
	from windows.kernel32 import (
		DynamicTimeZoneInformation,
		GetDynamicTimeZoneInformation,
		_TZNAME_SIZE,
		_TZNAME_OFFSET,
		_TZKEYNAME_SIZE,
		_TZKEYNAME_OFFSET,
	)

	# DynamicTimeZoneInformation is large (432 bytes, decomposed from three
	# WCHAR arrays - see kernel32.py's own comment). Constructing it as an
	# ordinary local (`tz_info = DynamicTimeZoneInformation()`, relying on
	# its all-"= 0" field defaults) makes clang synthesize calls to bare
	# memset/memcpy for the zero-init - confirmed directly this session, a
	# real LNK2019 "unresolved external symbol memset/memcpy" under this
	# project's deliberately CRT-free Windows link setup (no libc linked;
	# sys.memzero/memcpy route through RtlZeroMemory/RtlCopyMemory instead -
	# see lib/sys.py). Sidestepped by heap-allocating raw bytes and zeroing
	# them via sys.memzero (which DOES stay CRT-free) instead of ever
	# materializing an in-source all-zero struct literal this large. The
	# struct's contents are entirely OS-written moments later by
	# GetDynamicTimeZoneInformation anyway, so no meaningful initial value
	# is being skipped by not using the real constructor here.
	struct_size: usize = compiler.sizeof( DynamicTimeZoneInformation )
	raw: Ptr[u8] = sys.alloc[u8]( struct_size )
	sys.memzero( raw, struct_size )
	tz_info = compiler.cast( Ptr[DynamicTimeZoneInformation], raw )

	status: u32 = GetDynamicTimeZoneInformation( tz_info )
	# status values:
	# TIME_ZONE_ID_INVALID: u32   = 0xFFFFFFFF
	# TIME_ZONE_ID_UNKNOWN: u32   = 0
	# TIME_ZONE_ID_STANDARD: u32  = 1
	# TIME_ZONE_ID_DAYLIGHT: u32  = 2

	result: str = 'UTC'
	if status != 0xFFFFFFFF:
		opaque_ptr: Ptr[None] = compiler.cast( Ptr[None], tz_info )
		if tz_info.TimeZoneKeyName_0 != 0:
			key_ptr: Ptr[u16] = _field_ptr_u16( opaque_ptr, _TZKEYNAME_OFFSET )
			win_name: str = _decode_ascii_utf16z( key_ptr, _TZKEYNAME_SIZE )
			# windows_zones.to_iana() was always meant to back this lookup
			# (see lib/windows_zones.py) - opt-in: a program that never
			# calls windows_zones.install() gets None back and this
			# returns the raw Windows key name unchanged, which is a
			# valid ZoneInfo() key on Windows on its own (see
			# lib/windows/zoneinfo_rules.py).
			import windows_zones
			iana_name: str|None = windows_zones.to_iana( win_name )
			if iana_name is not None:
				result = iana_name
			else:
				result = win_name
		else:
			name_ptr: Ptr[u16] = _field_ptr_u16( opaque_ptr, _TZNAME_OFFSET )
			result = _decode_ascii_utf16z( name_ptr, _TZNAME_SIZE )

	sys.free( raw )
	return result
