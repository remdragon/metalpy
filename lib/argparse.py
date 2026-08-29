# lib/argparse.py — command-line argument parsing, adapted from Python's
# argparse for this compiler's static typing and no-exceptions error model:
#
#   - add_argument() takes a `names: list[str]` instead of Python's *args
#     spelling (`add_argument('-v', '--verbose', ...)`) - this compiler has
#     no variadic parameters at all (see SYNTAX.md's own print() note).
#   - Namespace has no dynamic attributes (Python's `args.foo`) - values are
#     typed via ArgValue (a tagged union: Str/Bool/List) and read back via
#     Namespace.get_str()/get_bool()/get_list().
#   - parse_args() still calls sys.exit(2) on a bad command line and
#     sys.exit(0) after printing help, matching real argparse's own
#     top-level behavior - a CLI program expects the process to just stop
#     on a bad invocation, not thread a Result through main().
#
# Deliberately NOT ported: subparsers, mutually exclusive groups, custom
# `type=`/`choices=` validators, `nargs` beyond the implicit single-value
# (Store/Append) vs zero-value (StoreTrue/StoreFalse) split. Add if a real
# caller needs them.

import compiler
import sys


@enum( i32 )
class ArgAction:
	Store      = 0  # single string value (default) - last occurrence wins if repeated
	StoreTrue  = 1  # boolean flag: False unless present, then True
	StoreFalse = 2  # boolean flag: True unless present, then False
	Append     = 3  # collects into a list[str]; can repeat


@union
class ArgValue:
	Str:  str
	Bool: bool
	List: list[str]


class Namespace:
	__values: dict[str, ArgValue]

	def __init__( self ) -> None:
		self.__values = dict[str, ArgValue]()

	def _set( self, name: str, value: ArgValue ) -> None:
		self.__values[name] = value

	def _append_str( self, name: str, item: str ) -> None:
		match self.__values.__getitem__( name ):
			case Result.Ok( ArgValue.List( existing ) ):
				existing.append( item )
				self.__values[name] = ArgValue.List( existing )
			case _:
				fresh: list[str] = list[str]()
				fresh.append( item )
				self.__values[name] = ArgValue.List( fresh )

	def has( self, name: str ) -> bool:
		return self.__values.__contains__( name )

	def get_str( self, name: str, default: str = '' ) -> str:
		match self.__values.__getitem__( name ):
			case Result.Ok( ArgValue.Str( s ) ):
				return s
			case _:
				return default

	def get_bool( self, name: str ) -> bool:
		match self.__values.__getitem__( name ):
			case Result.Ok( ArgValue.Bool( b ) ):
				return b
			case _:
				return False

	def get_list( self, name: str ) -> list[str]:
		match self.__values.__getitem__( name ):
			case Result.Ok( ArgValue.List( l ) ):
				return l
			case _:
				return list[str]()


class _ArgSpec:
	names:         list[str]
	dest:          str
	action:        ArgAction
	required:      bool
	default_str:   str|None  # NOT `default` - a C reserved word, breaks struct field emission
	help:          str
	is_positional: bool

	def __init__(
		self,
		names:         list[str],
		dest:          str,
		action:        ArgAction,
		required:      bool,
		default_str:   str|None,
		help:          str,
		is_positional: bool,
	) -> None:
		self.names = names
		self.dest = dest
		self.action = action
		self.required = required
		self.default_str = default_str
		self.help = help
		self.is_positional = is_positional

	def display_name( self ) -> str:
		if self.is_positional:
			return self.dest
		return '/'.join( self.names ) if self.names.__len__() > usize( 1 ) else self.names.__getitem__( 0 ).unwrap( 'names non-empty by construction' )

	def default_value( self ) -> ArgValue:
		if self.action == ArgAction.StoreTrue:
			return ArgValue.Bool( False )
		if self.action == ArgAction.StoreFalse:
			return ArgValue.Bool( True )
		if self.action == ArgAction.Append:
			return ArgValue.List( list[str]() )
		if self.default_str is not None:
			return ArgValue.Str( self.default_str )
		return ArgValue.Str( '' )


def _list_contains_str( items: list[str], target: str ) -> bool:
	''' list[T] has no __contains__ - a plain linear scan (option lists are
	always tiny, 1-2 entries). '''
	i: usize = 0
	with compiler.wrap_arithmetic:
		while i < items.__len__():
			if items.__getitem__( i ).unwrap( 'i < len by construction' ) == target:
				return True
			i += usize( 1 )
	return False


def _derive_dest( names: list[str], is_positional: bool ) -> str:
	first: str = names.__getitem__( 0 ).unwrap( 'add_argument: names must be non-empty' )
	if is_positional:
		return first.replace( '-', '_' )
	best: str = first.lstrip( '-' ).replace( '-', '_' )
	i: usize = 0
	with compiler.wrap_arithmetic:
		while i < names.__len__():
			n: str = names.__getitem__( i ).unwrap( 'i < len by construction' )
			if n.startswith( '--' ):
				best = n[2:].replace( '-', '_' )
				break
			i += usize( 1 )
	return best


class ArgumentParser:
	__prog:        str
	__description: str
	__specs:       list[_ArgSpec]

	def __init__( self, prog: str = '', description: str = '' ) -> None:
		self.__prog = prog if prog != '' else _program_name()
		self.__description = description
		self.__specs = list[_ArgSpec]()
		help_names: list[str] = ['-h', '--help']
		self.__specs.append( _ArgSpec( help_names, 'help', ArgAction.StoreTrue, False, None, 'show this help message and exit', False ))

	def add_argument(
		self,
		names:    list[str],
		action:   ArgAction = ArgAction.Store,
		required: bool = False,
		default:  str|None = None,
		help:     str = '',
		dest:     str|None = None,
	) -> None:
		''' names: e.g. ['-v', '--verbose'] for an optional, or ['input']
		for a positional (no leading '-'). Positionals are always required
		(matching real argparse - it rejects required= on a positional
		outright; this just ignores the passed value instead). '''
		is_positional: bool = not names.__getitem__( 0 ).unwrap( 'add_argument: names must be non-empty' ).startswith( '-' )
		resolved_dest: str
		if dest is not None:
			resolved_dest = dest
		else:
			resolved_dest = _derive_dest( names, is_positional )
		self.__specs.append( _ArgSpec( names, resolved_dest, action, required or is_positional, default, help, is_positional ))

	def __positional_specs( self ) -> list[_ArgSpec]:
		out: list[_ArgSpec] = list[_ArgSpec]()
		i: usize = 0
		with compiler.wrap_arithmetic:
			while i < self.__specs.__len__():
				s: _ArgSpec = self.__specs.__getitem__( i ).unwrap( 'i < len by construction' )
				if s.is_positional:
					out.append( s )
				i += usize( 1 )
		return out

	def __find_optional( self, token: str ) -> _ArgSpec|None:
		i: usize = 0
		with compiler.wrap_arithmetic:
			while i < self.__specs.__len__():
				s: _ArgSpec = self.__specs.__getitem__( i ).unwrap( 'i < len by construction' )
				if not s.is_positional and _list_contains_str( s.names, token ):
					return s
				i += usize( 1 )
		return None

	def __apply_defaults( self, ns: Namespace ) -> None:
		i: usize = 0
		with compiler.wrap_arithmetic:
			while i < self.__specs.__len__():
				s: _ArgSpec = self.__specs.__getitem__( i ).unwrap( 'i < len by construction' )
				ns._set( s.dest, s.default_value() )
				i += usize( 1 )

	def format_usage( self ) -> str:
		parts: str = f'usage: {self.__prog}'
		i: usize = 0
		with compiler.wrap_arithmetic:
			while i < self.__specs.__len__():
				s: _ArgSpec = self.__specs.__getitem__( i ).unwrap( 'i < len by construction' )
				if s.is_positional:
					parts = f'{parts} {s.dest}'
				else:
					name: str = s.names.__getitem__( 0 ).unwrap( 'names non-empty by construction' )
					shape: str = name if s.action == ArgAction.StoreTrue or s.action == ArgAction.StoreFalse else f'{name} {s.dest.upper()}'
					parts = f'{parts} [{shape}]' if not s.required else f'{parts} {shape}'
				i += usize( 1 )
		return f'{parts}\n'

	def format_help( self ) -> str:
		out: str = self.format_usage()
		if self.__description != '':
			out = f'{out}\n{self.__description}\n'
		out = f'{out}\noptions:\n'
		i: usize = 0
		with compiler.wrap_arithmetic:
			while i < self.__specs.__len__():
				s: _ArgSpec = self.__specs.__getitem__( i ).unwrap( 'i < len by construction' )
				out = f'{out}  {s.display_name()}\n\t{s.help}\n'
				i += usize( 1 )
		return out

	def print_help( self ) -> None:
		sys.stdout.write( self.format_help() ).is_ok()

	def error( self, message: str ) -> NoReturn:
		sys.stderr.write( self.format_usage() ).is_ok()
		sys.stderr.write( f'{self.__prog}: error: {message}\n' ).is_ok()
		sys.exit( i32( 2 ))

	def parse_args( self, args: list[str]|None = None ) -> Namespace:
		actual: list[str]
		if args is not None:
			actual = args
		else:
			actual = _default_args()
		ns: Namespace = Namespace()
		self.__apply_defaults( ns )

		positionals: list[_ArgSpec] = self.__positional_specs()
		pos_idx: usize = 0
		seen: dict[str, bool] = dict[str, bool]()

		i: usize = 0
		with compiler.wrap_arithmetic:
			while i < actual.__len__():
				tok: str = actual.__getitem__( i ).unwrap( 'i < len by construction' )
				if tok.startswith( '-' ) and tok != '-':
					found: _ArgSpec|None = self.__find_optional( tok )
					if found is None:
						self.error( f'unrecognized arguments: {tok}' )
						continue  # unreachable - error() never returns, but satisfies flow analysis
					spec: _ArgSpec = found
					if spec.action == ArgAction.StoreTrue:
						ns._set( spec.dest, ArgValue.Bool( True ))
						i += usize( 1 )
					elif spec.action == ArgAction.StoreFalse:
						ns._set( spec.dest, ArgValue.Bool( False ))
						i += usize( 1 )
					else:
						i += usize( 1 )
						if i >= actual.__len__():
							self.error( f'argument {tok}: expected one argument' )
						val: str = actual.__getitem__( i ).unwrap( 'checked above' )
						if spec.action == ArgAction.Append:
							ns._append_str( spec.dest, val )
						else:
							ns._set( spec.dest, ArgValue.Str( val ))
						i += usize( 1 )
					seen[spec.dest] = True
					if spec.dest == 'help' and ns.get_bool( 'help' ):
						self.print_help()
						sys.exit( i32( 0 ))
				else:
					if pos_idx >= positionals.__len__():
						self.error( f'unrecognized arguments: {tok}' )
					pspec: _ArgSpec = positionals.__getitem__( pos_idx ).unwrap( 'checked above' )
					ns._set( pspec.dest, ArgValue.Str( tok ))
					seen[pspec.dest] = True
					pos_idx += usize( 1 )
					i += usize( 1 )

			if pos_idx < positionals.__len__():
				missing: _ArgSpec = positionals.__getitem__( pos_idx ).unwrap( 'pos_idx < len by construction' )
				self.error( f'the following arguments are required: {missing.dest}' )

			j: usize = 0
			while j < self.__specs.__len__():
				s: _ArgSpec = self.__specs.__getitem__( j ).unwrap( 'j < len by construction' )
				if s.required and not s.is_positional and not seen.__contains__( s.dest ):
					self.error( f'the following arguments are required: {s.display_name()}' )
				j += usize( 1 )

		return ns


def _program_name() -> str:
	if sys.argv.__len__() > usize( 0 ):
		return sys.argv.__getitem__( 0 ).unwrap( 'checked above' )
	return 'prog'


def _default_args() -> list[str]:
	''' sys.argv with argv[0] dropped, matching Python's own
	parse_args(None) default of sys.argv[1:]. '''
	out: list[str] = list[str]()
	i: usize = 1
	with compiler.wrap_arithmetic:
		while i < sys.argv.__len__():
			out.append( sys.argv.__getitem__( i ).unwrap( 'i < len by construction' ))
			i += usize( 1 )
	return out
