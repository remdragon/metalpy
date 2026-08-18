# lib/logging.py — a logging module that works as similarly to Python's
# `logging` as metalpy's language currently allows.
#
# Deliberate departures from CPython's logging, forced by the language:
#   - No *args/%-formatting: metalpy has no varargs at call sites and no
#     %/​.format() (lowering.py rejects *args/**kwargs unpacking today).
#     Log calls take a single, already-built `str` - callers use an
#     f-string themselves (`logger.info(f'...')`), same escape hatch
#     Python's own docs recommend for expensive messages. A lazy
#     Closure[[],str] overload was considered and deliberately left out of
#     v1 - t-strings are the intended real fix once they land.
#   - No exceptions: everything fallible returns Result[T,E]. There is no
#     Handler.handleError - a handler's emit() failure is just a Result the
#     caller (Logger._handle) discards, same as a destructor discarding
#     close() (see lib/builtins/__File.py's own convention).
#   - No call-stack introspection: LogRecord has no filename/lineno/funcName.
#   - No datetime/strftime: LogRecord.created is a raw time.time() epoch
#     float; Formatter.format is a virtual method (not a % format string)
#     precisely so callers CAN still get custom timestamp rendering by
#     subclassing.
#
# Formatter.format and Handler.emit are @virtual/@abstractmethod, so real
# subclassing + polymorphic dispatch works here exactly like lib/codecs's
# Codec base class - Handler/Formatter values are held and called through
# their base type (list[Handler], Handler.formatter: Formatter) and real
# vtable dispatch reaches the override.

import compiler
import sys
import time
import threading

# FileHandler is built directly on fs's raw FD primitives, NOT on
# lib/builtins/__File.py's File/BinaryWriter - storing a BinaryWriter as a
# class field (as opposed to a local variable) crashes the compiler's
# destructor-synthesis pass (type_resolver.py:_build_field_teardown_ast,
# a confirmed pre-existing bug, unrelated to logging's own design - see the
# commit message). A raw FD field, mirroring how lib/threading.py's
# FastLock/lib/atomic.py's Atomic[T] already hold their own raw handles
# directly, sidesteps it entirely.
from fs import FD, INVALID_FD, close_raw, open_raw, seek_raw, write_all
if compiler.target.os == 'windows':
	from fs import GENERIC_WRITE, OPEN_ALWAYS, FILE_END
else:
	from fs import O_WRONLY, O_CREAT, O_APPEND

# ---------------------------------------------------------------------------
# Levels - plain i32 constants (not an @enum), matching Python's own
# logging module allowing arbitrary custom integer levels.
# ---------------------------------------------------------------------------

NOTSET: i32 = 0
DEBUG: i32 = 10
INFO: i32 = 20
WARNING: i32 = 30
ERROR: i32 = 40
CRITICAL: i32 = 50

def _level_name( level: i32 ) -> str:
	if level == DEBUG:
		return 'DEBUG'
	if level == INFO:
		return 'INFO'
	if level == WARNING:
		return 'WARNING'
	if level == ERROR:
		return 'ERROR'
	if level == CRITICAL:
		return 'CRITICAL'
	if level == NOTSET:
		return 'NOTSET'
	return f'Level {int( level )}' # i32 itself has no __str__ - only the boxed int does


# ---------------------------------------------------------------------------
# LogRecord
# ---------------------------------------------------------------------------

class LogRecord:
	name: str
	level: i32
	message: str
	created: f64

	def __init__( self, name: str, level: i32, message: str, created: f64 ) -> None:
		self.name = name
		self.level = level
		self.message = message
		self.created = created


# ---------------------------------------------------------------------------
# Formatter - subclass and override format() for custom output shapes;
# there is no % format-string mini-language to configure instead.
# ---------------------------------------------------------------------------

class Formatter:
	@virtual
	def format( self, record: LogRecord ) -> str:
		return f'{record.created} {record.name} {_level_name( record.level )} {record.message}'


# ---------------------------------------------------------------------------
# Handler
# ---------------------------------------------------------------------------

class Handler:
	level: i32
	formatter: Formatter

	def __init__( self ) -> None:
		self.level = NOTSET
		self.formatter = Formatter()

	def setLevel( self, level: i32 ) -> None:
		self.level = level

	def setFormatter( self, fmt: Formatter ) -> None:
		self.formatter = fmt

	@abstractmethod
	def emit( self, record: LogRecord ) -> Result[None, OSError]:
		...


class StreamHandler( Handler ):
	''' writes formatted records to sys.stderr (Python's own StreamHandler
	default) or sys.stdout. '''
	__to_stderr: bool

	def __init__( self, to_stderr: bool = True ) -> None:
		super().__init__()
		self.__to_stderr = to_stderr

	@virtual
	def emit( self, record: LogRecord ) -> Result[None, OSError]:
		line: str = self.formatter.format( record ) + '\n'
		if self.__to_stderr:
			return sys.stderr.write( line )
		return sys.stdout.write( line )


class FileHandler( Handler ):
	''' appends formatted records to a file, opened once at construction. '''
	__fd: FD

	@compiler.target( os = 'windows' )
	def __init__( self, path: str ) -> Result[None, OSError]:
		super().__init__()
		fd: FD = open_raw( path.get_cstr(), GENERIC_WRITE, OPEN_ALWAYS ).or_return()
		seek_raw( fd, 0, FILE_END ).or_return()
		self.__fd = fd
		return Result.Ok( None )

	@compiler.target( os = not 'windows' )
	def __init__( self, path: str ) -> Result[None, OSError]:
		super().__init__()
		self.__fd = open_raw( path.get_cstr(), O_WRONLY | O_CREAT | O_APPEND, 0o644 ).or_return()
		return Result.Ok( None )

	def __del__( self ) -> None:
		if self.__fd != INVALID_FD:
			close_raw( self.__fd ).is_ok() # a destructor can't propagate close() failure - deliberately ignored, not silently unchecked

	@virtual
	def emit( self, record: LogRecord ) -> Result[None, OSError]:
		line: str = self.formatter.format( record ) + '\n'
		write_all( self.__fd, line.get_cstr(), line.byte_len() ).or_return()
		return Result.Ok( None )

	def close( self ) -> None:
		if self.__fd != INVALID_FD:
			close_raw( self.__fd ).is_ok() # see __del__'s own comment
			self.__fd = INVALID_FD


# ---------------------------------------------------------------------------
# Logger - dotted-name hierarchy: a child's parent is looked up BY NAME
# through the registry on every call, rather than stored as a
# `parent: Logger` field - simpler ownership (no possibility of a reference
# cycle through the registry) at the cost of one extra registry lookup per
# ancestor per log call, same tradeoff Python's C implementation doesn't
# have to make but a pure-registry design does.
# ---------------------------------------------------------------------------

def _parent_name( name: str ) -> str|None:
	''' 'a.b.c' -> 'a.b'; 'a' -> '' (root); '' -> None (root has no parent).
	Built on str.split('.')/str.join rather than rfind + a substring slice -
	str has no public substring-by-byte-offset operation (str._byte_slice is
	@private to the str class itself), so the dotted-name hierarchy is
	implemented on the public split/join surface instead. '''
	if name == '':
		return None
	parts: list[str] = name.split( '.' )
	count: usize = len( parts )
	if count <= 1:
		return ''
	with compiler.panic_arithmetic( 'unreachable: count > 1, just checked above' ):
		parent_count: usize = count - 1
	parent_parts: list[str] = list[str]()
	for i in range( parent_count ):
		part: str = parts.__getitem__( i ).unwrap( 'unreachable: i bounded by parent_count < count' )
		parent_parts.append( part ).unwrap( 'unreachable: fresh list, cannot overflow' )
	sep: str = '.'
	return sep.join( parent_parts )


class Logger:
	name: str
	level: i32          # NOTSET means "inherit from the parent logger"
	handlers: list[Handler]
	propagate: bool

	def __init__( self, name: str ) -> None:
		self.name = name
		self.level = NOTSET
		self.handlers = list[Handler]()
		self.propagate = True

	def setLevel( self, level: i32 ) -> None:
		self.level = level

	def addHandler( self, handler: Handler ) -> None:
		self.handlers.append( handler ).unwrap( 'Logger.addHandler: append failed' )

	def _parent( self ) -> Logger|None:
		parent_name: str|None = _parent_name( self.name )
		if parent_name is None:
			return None # already the root
		return getLogger( parent_name )

	def getEffectiveLevel( self ) -> i32:
		# `while logger is not None:` doesn't narrow logger's type inside
		# the loop body (unlike an `if` guard) - walk with an always-
		# Logger-typed `current`, re-narrowed via an if-guard each
		# iteration instead, matching __str.py's own `if loc is None:
		# return ...` / use-loc-unconditionally-after pattern.
		current: Logger = self
		while True:
			if current.level != NOTSET:
				return current.level
			parent: Logger|None = current._parent()
			if parent is None:
				return WARNING # root always has an explicit level - unreachable in practice
			current = parent

	def isEnabledFor( self, level: i32 ) -> bool:
		return level >= self.getEffectiveLevel()

	def log( self, level: i32, msg: str ) -> None:
		if not self.isEnabledFor( level ):
			return
		record: LogRecord = LogRecord( self.name, level, msg, time.time() )
		self._handle( record )

	def debug( self, msg: str ) -> None:
		self.log( DEBUG, msg )

	def info( self, msg: str ) -> None:
		self.log( INFO, msg )

	def warning( self, msg: str ) -> None:
		self.log( WARNING, msg )

	def error( self, msg: str ) -> None:
		self.log( ERROR, msg )

	def critical( self, msg: str ) -> None:
		self.log( CRITICAL, msg )

	def _handle( self, record: LogRecord ) -> None:
		current: Logger = self
		while True:
			for i in range( len( current.handlers )):
				handler: Handler = current.handlers.__getitem__( i ).unwrap( 'unreachable: bounded by len()' )
				if record.level >= handler.level:
					handler.emit( record ).is_ok() # no Handler.handleError - deliberately discarded, not silently unchecked
			if not current.propagate:
				return
			parent: Logger|None = current._parent()
			if parent is None:
				return
			current = parent


# ---------------------------------------------------------------------------
# Registry + root logger + module-level convenience functions
#
# The registry itself (an EMPTY UnsafeDict) is a safe module-level global -
# confirmed by a standalone repro. Eagerly constructing a `root: Logger`
# singleton by calling getLogger('') at global-init time is NOT safe -
# inserting a fresh RC-class value into a module-level UnsafeDict from
# global static-init (as opposed to from within a real function call
# happening after main() has started) reliably crashes/corrupts state, a
# separate, confirmed compiler bug outside this module's own scope (global-
# init ordering, not a str/dict/logging-specific issue). root() is
# therefore a function, not a bare attribute like Python's `logging.root`
# - it just calls getLogger('') lazily, on first REAL use.
# ---------------------------------------------------------------------------

_registry: UnsafeDict[str, Logger] = UnsafeDict[str, Logger]()
_registry_lock: threading.FastLock = threading.FastLock()

def getLogger( name: str = '' ) -> Logger:
	_registry_lock.acquire().unwrap( 'logging.getLogger: lock failed' )
	defer( _registry_lock.release() )
	existing: Result[Logger,KeyError] = _registry.__getitem__( name )
	if existing.is_ok():
		return existing.unwrap( 'unreachable: is_ok just confirmed' )
	logger: Logger = Logger( name )
	if name == '':
		logger.level = WARNING # root's own explicit default level, matches Python
	_registry[name] = logger
	return logger

def root() -> Logger:
	return getLogger( '' )

def basicConfig( level: i32 = WARNING ) -> None:
	r: Logger = root()
	r.setLevel( level )
	if len( r.handlers ) == 0:
		r.addHandler( StreamHandler() )

def debug( msg: str ) -> None:
	root().debug( msg )

def info( msg: str ) -> None:
	root().info( msg )

def warning( msg: str ) -> None:
	root().warning( msg )

def error( msg: str ) -> None:
	root().error( msg )

def critical( msg: str ) -> None:
	root().critical( msg )
