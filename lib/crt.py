@compiler.target( os = not ( 'windows', 'macos' ))
@extern( 'c', '__errno_location' )
def __error() -> Ptr[i32]:
	...

@compiler.target( os = 'macos' )
@extern( 'c', '__error' )
def __error() -> Ptr[i32]:
	...

@extern( 'c', '_exit' )
def _exit(
	status: i32,
) -> None:
	...

@extern( 'c', 'free' )
def free(
	ptr: Ptr[None],
) -> None:
	# void* (Ptr[None] -> void* - see emitter.py's map_type_to_c), matching
	# libc's real `void free(void*)` exactly - a Ptr[u8] declaration here
	# compiles fine (void* implicitly converts, so callers are unaffected
	# either way) but triggers GCC's -Wbuiltin-declaration-mismatch ("conflicting
	# types for built-in function 'free'") since GCC knows free's real
	# signature and compares against whatever's declared here.
	...

@compiler.target( os = not 'windows' )
def get_errno() -> i32:
	ptr = __error()
	return ptr[0]

@extern( 'c', 'malloc' )
def malloc(
	size: usize,
) -> Ptr[u8]:
	...

@extern( 'c', 'memset' )
def memset(
	ptr: Ptr[u8],
	value: u8,
	length: usize,
) -> None:
	...

@extern( 'c', 'memcpy' )
def memcpy(
	dest: Ptr[u8],
	src: ConstPtr[u8],
	n: usize,
) -> Ptr[u8]:
	...

@extern( 'c', 'memcmp' )
def memcmp(
	a: ConstPtr[u8],
	b: ConstPtr[u8],
	n: usize,
) -> i32:
	...

@extern( 'c', 'memmove' )
def memmove(
	dest: Ptr[u8],
	src: ConstPtr[u8],
	n: usize,
) -> Ptr[u8]:
	...

@extern( 'c', 'readlink' )
def readlink(
	path: ConstPtr[u8],
	buf: Ptr[u8],
	bufsize: usize,
) -> usize:
	...

@extern( 'c', 'strerror' )
def strerror( errnum: 32 ) -> ConstPtr[u8]|None:
	...

@extern( 'c', 'strnlen' )
def strnlen(
	str: ConstPtr[u8],
	maxlen: usize,
) -> usize:
	...

@extern( 'c', 'write' )
def write(
	fd: i32,
	buf: ConstPtr[u8],
	count: usize,
) -> isize:
	...

@extern( 'c', 'open' )
def open(
	path: ConstPtr[u8],
	flags: i32,
	mode: i32,
) -> i32:
	...

@extern( 'c', 'close' )
def close(
	fd: i32,
) -> i32:
	...

@extern( 'c', 'read' )
def read(
	fd: i32,
	buf: Ptr[u8],
	count: usize,
) -> isize:
	...

@extern( 'c', 'lseek' )
def lseek(
	fd: i32,
	offset: i64,
	whence: i32,
) -> i64:
	...

@extern( 'c', 'ftruncate' )
def ftruncate(
	fd: i32,
	length: i64,
) -> i32:
	...
