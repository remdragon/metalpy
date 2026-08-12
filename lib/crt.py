import compiler

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

# ---------------------------------------------------------------------------
# Unicode-correct case mapping - str.upper()/str.lower() (see PLAN_STR_UPPER_
# LOWER.md). towupper_l/towlower_l (the explicit-locale, thread-safe POSIX.1-
# 2008 variants) rather than plain setlocale()+towupper()/towlower() - a bare
# setlocale() mutates process-global state, which is a real data race against
# any other thread calling .upper()/.lower() (or anything else locale-
# sensitive) concurrently - this codebase has lib/threading.py, so that's a
# real scenario, not a hypothetical one. 'C.UTF-8' gives Unicode-aware casing
# without any language-specific tailoring (no Turkish dotless-i surprises) -
# the POSIX equivalent of Windows' LOCALE_NAME_INVARIANT. Only handles
# single-codepoint mappings (towupper_l/towlower_l are inherently 1-in-1-out)
# - no ICU here, so one-to-many expansions (ß -> SS) and context-sensitive
# rules (Greek final sigma) aren't covered on this path - see
# PLAN_STR_UPPER_LOWER.md's own notes on why (ICU's availability isn't
# guaranteed on any of these platforms, and compiler.has_library()-gating it
# would mean every program that merely imports builtins pays an eager probe).
# ---------------------------------------------------------------------------

locale_t: TypeAlias = Ptr[None]

# LC_CTYPE_MASK's numeric value isn't ABI-stable across glibc/musl/macOS's
# libc (each defines it as 1 << their own LC_CTYPE, which aren't guaranteed
# to agree) - compiler.cexpr fetches the real one from the actual headers
# this build will compile against, rather than hardcoding a guess
LC_CTYPE_MASK: i32 = compiler.cexpr( 'LC_CTYPE_MASK', 'locale.h', i32 )

@extern( 'c', 'newlocale' )
def newlocale(
	category_mask: i32,
	locale: ConstPtr[u8],
	base: locale_t,
) -> locale_t:
	...

@extern( 'c', 'freelocale' )
def freelocale(
	locobj: locale_t,
) -> None:
	...

@extern( 'c', 'towupper_l' )
def towupper_l(
	wc: i32,
	loc: locale_t,
) -> i32:
	...

@extern( 'c', 'towlower_l' )
def towlower_l(
	wc: i32,
	loc: locale_t,
) -> i32:
	...

# ---------------------------------------------------------------------------
# Unicode codepoint classification - str.isalpha()/isdigit()/isspace()/
# isupper()/islower()/isalnum()/isprintable() (see __str.py's is_*_cp
# primitives, TODO.txt's str-methods plan). Same explicit-locale,
# 'C.UTF-8', per-call newlocale/freelocale convention as towupper_l/
# towlower_l above, for the identical data-race reasoning - a plain
# iswalpha()/etc would depend on process-global locale state.
# ---------------------------------------------------------------------------

@extern( 'c', 'iswalpha_l' )
def iswalpha_l(
	wc: i32,
	loc: locale_t,
) -> i32:
	...

@extern( 'c', 'iswdigit_l' )
def iswdigit_l(
	wc: i32,
	loc: locale_t,
) -> i32:
	...

@extern( 'c', 'iswspace_l' )
def iswspace_l(
	wc: i32,
	loc: locale_t,
) -> i32:
	...

@extern( 'c', 'iswupper_l' )
def iswupper_l(
	wc: i32,
	loc: locale_t,
) -> i32:
	...

@extern( 'c', 'iswlower_l' )
def iswlower_l(
	wc: i32,
	loc: locale_t,
) -> i32:
	...

@extern( 'c', 'iswalnum_l' )
def iswalnum_l(
	wc: i32,
	loc: locale_t,
) -> i32:
	...

@extern( 'c', 'iswprint_l' )
def iswprint_l(
	wc: i32,
	loc: locale_t,
) -> i32:
	...
