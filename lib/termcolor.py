'''
termcolor - ANSI/CSI helpers for colored terminal output.

Mirrors the real termcolor package's (pip: termcolor) colored() signature
closely enough to run existing code written against it unmodified:
	termcolor.colored( text, 'cyan' )
	termcolor.colored( text, 'red', attrs = ['bold'] )
'''

import compiler
import sys
import threading

_color_codes: dict[str,str]|None = None
_color_codes_lock: threading.FastLock = threading.FastLock()

def _codes() -> dict[str,str]:
	''' lock-guarded lazy init, not a bare is-None check - see lib/
	datetime.py's localtz() for why an unguarded check-then-set on a
	module-global cache is a real data race once called from more than
	one OS thread. '''
	global _color_codes
	with _color_codes_lock:
		if _color_codes is None:
			codes: dict[str,str] = dict[str,str]()
			codes['black']   = '30'
			codes['red']     = '31'
			codes['green']   = '32'
			codes['yellow']  = '33'
			codes['blue']    = '34'
			codes['magenta'] = '35'
			codes['cyan']    = '36'
			codes['white']   = '37'
			# 'grey' maps to bright-black (90), not plain black (30) - plain
			# black is invisible on the common dark-terminal-background case
			# this is meant for (dim status/debug output).
			codes['grey']    = '90'
			_color_codes = codes
		return _color_codes

def colored( text: str, color: str|None = None, attrs: list[str]|None = None ) -> str:
	parts: list[str] = []
	if color is not None:
		match _codes().__getitem__( color ):
			case Result.Ok( code ):
				parts.append( code ).unwrap( 'termcolor.colored: append failed' )
			case Result.Err( e ):
				sys.panic( f'termcolor.colored: unknown color {color}' )
	if attrs is not None:
		attr_list: list[str] = attrs
		i: usize = 0
		with compiler.wrap_arithmetic:
			while i < len( attr_list ):
				attr: str = attr_list.__getitem__( i ).unwrap( 'termcolor.colored: attrs index in range' )
				if attr == 'bold':
					parts.append( '1' ).unwrap( 'termcolor.colored: append failed' )
				else:
					sys.panic( 'termcolor.colored: unknown attr ' + attr )
				i += 1
	if len( parts ) == 0:
		return text
	_parts_ = ';'.join( parts )
	prefix: str = f'\x1b[{_parts_}m'
	return f'{prefix}{text}\x1b[0m'
