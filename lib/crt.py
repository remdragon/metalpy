import compiler

@compiler.target( os = not ( 'windows', 'macos' ))
@extern( 'c', '__errno_location' )
def __error() -> Ptr[i32]:
	...

@compiler.target( os = 'macos' )
@extern( 'c', '__error' )
def __error() -> Ptr[i32]:
	...

# split by target (unlike this file's other externs) so header='unistd.h'
# can apply to the branch that's genuinely POSIX - a single ungated def here
# would get discovered (and its header= registered) on EVERY target, since
# nothing else in this file gates on @compiler.target the way its own
# callers (lib/sys.py) do; confirmed via a real "Cannot open include file:
# 'unistd.h'" error on Windows from an earlier, ungated attempt at this same
# fix. See write's own identical comment below - this is the same pattern,
# repeated for every unistd.h-shaped extern in this file.
@compiler.target( os = not 'windows' )
@extern( 'c', '_exit', header = 'unistd.h' )
def _exit(
	status: i32,
) -> NoReturn:
	...

# no real Windows caller today (lib/sys.py's own exit() only ever reaches
# this via its `os = not 'windows'` branch) - kept for symmetry with every
# other os-split pair in this file, and so `from crt import _exit` still
# resolves on every target. Deliberately no header= (unistd.h doesn't exist
# on Windows).
@compiler.target( os = 'windows' )
@extern( 'c', '_exit' )
def _exit(
	status: i32,
) -> NoReturn:
	...

# the size of a live malloc'd block, for sys.free()'s own debug-only
# mempoison-before-free - glibc/Linux and macOS expose this under two
# different names (macOS's <malloc/malloc.h> malloc_size vs glibc's
# malloc_usable_size), same os split as __error/__errno_location above.
# Both may return a genuinely LARGER size than what was originally
# requested (allocator rounding) - fine here, poisoning a few extra
# trailing bytes inside the same live block is harmless.
@compiler.target( os = not ( 'windows', 'macos' ))
@extern( 'c', 'malloc_usable_size' )
def malloc_usable_size( ptr: ConstPtr[None] ) -> usize:
	...

@compiler.target( os = 'macos' )
@extern( 'c', 'malloc_size' )
def malloc_usable_size( ptr: ConstPtr[None] ) -> usize:
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
	# void*/int/void* (not Ptr[u8]/u8) to match libc's real `void*
	# memset(void*, int, size_t)` exactly - see free()'s comment above for
	# why (GCC's -Wbuiltin-declaration-mismatch)
	ptr: Ptr[None],
	value: i32,
	length: usize,
) -> Ptr[None]:
	...

@extern( 'c', 'memcpy' )
def memcpy(
	dest: Ptr[None],
	src: ConstPtr[None],
	n: usize,
) -> Ptr[None]:
	...

@extern( 'c', 'memcmp' )
def memcmp(
	a: ConstPtr[None],
	b: ConstPtr[None],
	n: usize,
) -> i32:
	...

@extern( 'c', 'memmove' )
def memmove(
	dest: Ptr[None],
	src: ConstPtr[None],
	n: usize,
) -> Ptr[None]:
	...

# deliberately NOT header='unistd.h' (unlike write/close/read/lseek/
# ftruncate/getcwd/rmdir/unlink just below/above - all real POSIX.1 base
# functions, visible in glibc's <unistd.h> under plain -std=c11): readlink
# is XSI/BSD-gated in glibc, hidden under -std=c11's implied __STRICT_ANSI__
# unless _GNU_SOURCE (or similar) is defined before the FIRST #include of
# anything - confirmed via a real "implicit declaration of function
# 'readlink'" GCC error even with unistd.h genuinely included. Defining
# _GNU_SOURCE process-wide was tried and reverted: it also newly exposes
# glibc's OWN clock_gettime through <pthread.h>'s transitively-included
# <time.h> (previously hidden the same way), conflicting with lib/posix/
# time.py's own hand-declared one (confirmed via a real "conflicting types
# for 'clock_gettime'" error) - likely not the only such collision waiting
# in every other hand-rolled POSIX extern in this codebase. Given that
# blast radius, kept private/hand-declared here instead - the ConstPtr[None]/
# Ptr[None] param types (not ConstPtr[u8]/Ptr[u8]) are still real readlink's
# own signature (matching mkdir's void*-sidesteps-signedness posture above),
# kept correct in case a future, more targeted fix (e.g. a readlink-specific
# feature-test macro scoped narrower than _GNU_SOURCE) makes header=
# viable again. Callers (lib/posix/fs.py's readlink) cast their
# ConstPtr[u8]/Ptr[u8] arguments accordingly.
@extern( 'c', 'readlink' )
def readlink(
	path: ConstPtr[None],
	buf: Ptr[None],
	bufsize: usize,
) -> isize:
	...

@extern( 'c', 'strerror' )
def strerror( errnum: i32 ) -> ConstPtr[u8]:
	...

@extern( 'c', 'strnlen' )
def strnlen(
	str: ConstPtr[u8],
	maxlen: usize,
) -> usize:
	...

# ConstPtr[None] (not ConstPtr[u8]) on the POSIX branch - see readlink's own
# comment above for why (real write's `const void*` buffer, made visible by
# header='unistd.h', conflicts with unsigned u8's pointee signedness).
# Callers (lib/fs.py's write_raw) cast their ConstPtr[u8] argument
# accordingly. emitter_c.py's own _PROLOGUE_CRASH_HANDLER now `#include
# <unistd.h>` directly on the POSIX side instead of hand-declaring its own
# write()/_exit() - safe now that this file's own prototype for both defers
# to the same real header rather than guessing at an incompatible one.
@compiler.target( os = not 'windows' )
@extern( 'c', 'write', header = 'unistd.h' )
def write(
	fd: i32,
	buf: ConstPtr[None],
	count: usize,
) -> isize:
	...

@compiler.target( os = 'windows' )
@extern( 'c', 'write' )
def write(
	fd: i32,
	buf: ConstPtr[u8],
	count: usize,
) -> isize:
	...

# POSIX-only: lib/sys.py's stdout/stderr buffering uses this to pick
# line-buffered (a real terminal) vs block-buffered (redirected file/pipe) -
# same real() header/split posture as write's own POSIX branch above.
@compiler.target( os = not 'windows' )
@extern( 'c', 'isatty', header = 'unistd.h' )
def isatty(
	fd: i32,
) -> i32:
	...

@extern( 'c', 'open' )
def open(
	path: ConstPtr[u8],
	flags: i32,
	mode: i32,
) -> i32:
	...

# int/int32_t are the same underlying type on every real target here, so
# this split exists purely to carry header='unistd.h' without also
# registering it for a Windows build (see _exit's own comment above) - not
# because close's own signature needed correcting.
@compiler.target( os = not 'windows' )
@extern( 'c', 'close', header = 'unistd.h' )
def close(
	fd: i32,
) -> i32:
	...

@compiler.target( os = 'windows' )
@extern( 'c', 'close' )
def close(
	fd: i32,
) -> i32:
	...

# Ptr[None] (not Ptr[u8]) on the POSIX branch - see write's own comment
# above for why. Callers (lib/fs.py's read_raw) cast their Ptr[u8] argument
# accordingly.
@compiler.target( os = not 'windows' )
@extern( 'c', 'read', header = 'unistd.h' )
def read(
	fd: i32,
	buf: Ptr[None],
	count: usize,
) -> isize:
	...

@compiler.target( os = 'windows' )
@extern( 'c', 'read' )
def read(
	fd: i32,
	buf: Ptr[u8],
	count: usize,
) -> isize:
	...

# i64 already matches real off_t (a 64-bit `long`/`long long` on every
# 64-bit target this codebase supports) - this split exists purely to carry
# header='unistd.h' without registering it on Windows too, same as close's
# own comment above.
@compiler.target( os = not 'windows' )
@extern( 'c', 'lseek', header = 'unistd.h' )
def lseek(
	fd: i32,
	offset: i64,
	whence: i32,
) -> i64:
	...

@compiler.target( os = 'windows' )
@extern( 'c', 'lseek' )
def lseek(
	fd: i32,
	offset: i64,
	whence: i32,
) -> i64:
	...

# i64 already matches real off_t - see lseek's own comment just above.
@compiler.target( os = not 'windows' )
@extern( 'c', 'ftruncate', header = 'unistd.h' )
def ftruncate(
	fd: i32,
	length: i64,
) -> i32:
	...

@compiler.target( os = 'windows' )
@extern( 'c', 'ftruncate' )
def ftruncate(
	fd: i32,
	length: i64,
) -> i32:
	...

# Ptr[None] (not Ptr[u8]) param AND return on the POSIX branch - real
# getcwd's `char*` conflicts with unsigned u8 both ways once header=
# 'unistd.h' makes the real prototype visible. Callers (lib/os.py's
# _getcwd) cast their Ptr[u8] argument accordingly; the None-vs-not-None
# check on the returned pointer needs no cast either way.
@compiler.target( os = not 'windows' )
@extern( 'c', 'getcwd', header = 'unistd.h' )
def getcwd(
	buf: Ptr[None],
	size: usize,
) -> Ptr[None]:
	# NULL on failure (e.g. ERANGE if buf is too small for the real cwd) -
	# callers must check get_errno() to distinguish the failure reason.
	...

@compiler.target( os = 'windows' )
@extern( 'c', 'getcwd' )
def getcwd(
	buf: Ptr[u8],
	size: usize,
) -> Ptr[u8]:
	...

@extern( 'c', 'mkdir', header = 'sys/stat.h' )
def mkdir(
	# ConstPtr[None], not ConstPtr[u8] - lib/posix/stat.py's own stat()
	# extern already pulls in sys/stat.h (real prototype: mkdir(const
	# char*, mode_t)), and once a header makes the real prototype visible
	# gcc hard-errors on a char*/unsigned-char* mismatch
	# (-Wincompatible-pointer-types) - void* sidesteps the signedness
	# distinction entirely, same posture as opendir/stat's own path params.
	path: ConstPtr[None],
	mode: i32,
) -> i32:
	...

# ConstPtr[None] (not ConstPtr[u8]) on the POSIX branch - see mkdir's own
# comment above for why. Callers (lib/os.py's rmdir) cast their ConstPtr[u8]
# argument accordingly.
@compiler.target( os = not 'windows' )
@extern( 'c', 'rmdir', header = 'unistd.h' )
def rmdir(
	path: ConstPtr[None],
) -> i32:
	...

@compiler.target( os = 'windows' )
@extern( 'c', 'rmdir' )
def rmdir(
	path: ConstPtr[u8],
) -> i32:
	...

# ConstPtr[None] (not ConstPtr[u8]) on the POSIX branch - see mkdir's own
# comment above for why. Callers (lib/os.py's unlink) cast their
# ConstPtr[u8] argument accordingly.
@compiler.target( os = not 'windows' )
@extern( 'c', 'unlink', header = 'unistd.h' )
def unlink(
	path: ConstPtr[None],
) -> i32:
	...

@compiler.target( os = 'windows' )
@extern( 'c', 'unlink' )
def unlink(
	path: ConstPtr[u8],
) -> i32:
	...

@extern( 'c', 'rename' )
def rename(
	old: ConstPtr[u8],
	new: ConstPtr[u8],
) -> i32:
	...

# POSIX-only: os.rename()'s own fail-if-exists semantics (matching Windows)
# are built on link()+unlink() rather than plain rename(2), which replaces
# an existing destination - see lib/os.py's rename(). ConstPtr[None], not
# ConstPtr[u8] - same char*/unsigned-char* mismatch as mkdir/rmdir/unlink's
# own POSIX bindings above, once header= makes the real prototype visible.
@compiler.target( os = not 'windows' )
@extern( 'c', 'link', header = 'unistd.h' )
def link(
	oldpath: ConstPtr[None],
	newpath: ConstPtr[None],
) -> i32:
	...

@extern( 'c', 'getenv' )
def getenv(
	name: ConstPtr[u8],
) -> ConstPtr[u8]:
	# the returned pointer is owned by the CRT (valid only until the next
	# environment mutation) - callers must copy it into a str immediately,
	# never hold onto it.
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

# ---------------------------------------------------------------------------
# sqrt/sqrtf - lib/builtins/__float.py's f64.sqrt()/f32.sqrt(). A single
# hardware instruction on every real target (SQRTSD/SQRTSS on x86, FSQRT on
# ARM) - this still goes through an ordinary libm/CRT call (no builtin-call
# spelling reaches a compiler intrinsic from this codebase's own extern
# mechanism), but real linkers/optimizers fold the call down to that one
# instruction anyway. POSIX splits out 'm' (glibc keeps math functions in a
# separate libm archive pre-2.34; -lm still resolves fine on newer glibc and
# on macOS, where it's just an alias into libSystem) - Windows has no
# separate libm, math functions live directly in the CRT ('c'), same as
# every other Windows extern in this file.
@compiler.target( os = not 'windows' )
@extern( 'm', 'sqrt', header = 'math.h' )
def sqrt( x: f64 ) -> f64:
	...

@compiler.target( os = 'windows' )
@extern( 'c', 'sqrt', header = 'math.h' )
def sqrt( x: f64 ) -> f64:
	...

@compiler.target( os = not 'windows' )
@extern( 'm', 'sqrtf', header = 'math.h' )
def sqrtf( x: f32 ) -> f32:
	...

@compiler.target( os = 'windows' )
@extern( 'c', 'sqrtf', header = 'math.h' )
def sqrtf( x: f32 ) -> f32:
	...
