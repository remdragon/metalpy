# SYNTAX.md - Metal Py Language Specification & Syntax Guide

This document defines the language constructs, syntax rules, type system, error handling mechanisms, memory management annotations, and low-level C FFI primitives for **Metal Py**.

---

## 1. Core Principles & Paradigm

* **Syntax**: Native Python 3.12+ AST compatible syntax.
* **Compilation**: Directly compiled to static C (No VM, No Interpreter, No JIT). Maybe other backends in the future
* **Control Flow**: 100% deterministic control flow with **Zero Exceptions**. All fallible operations return `Result[T, E]`.
* **Code Formatting Directive**: Hard Tabs (`\t`) must be used for all Metal Py source code.

---

## 2. Type System & Memory Representation

### Fixed-Width & Arbitrary-Precision Primitives
* **Unsigned Integers**: `u8`, `u16`, `u32`, `u64`, `u128`, `usize`
* **Signed Integers**: `i8`, `i16`, `i32`, `i64`, `i128`, `isize`
* **Floating-Point**: `f32`, `f64`
* **Boolean**: `bool`
* **Arbitrary-Precision Integer**: `int` (Heap-managed, supports standard `+`, `-`, `*` operators; no implicit conversion to/from fixed-width integer types).

### builtin types
* str: immutable strings, guaranteed to be valid utf8 and null-terminated.
* bytes: immutable sequence of bytes (u8)
* bytearray: mutable sequence of bytes (u8)
* list[T]: threadsafe mutable list of objects
* dict[K,V]: threadsafe mutable map/dictionary of key/value pairs
* set[T]: threadsafe mutable collection of unique values
* tuple: immutable collection of values

### Pointers, Slices & Fixed-Size Inline Arrays
* **Mutable Raw Pointer**: `Ptr[T]` (e.g., `Ptr[u8]`, `Ptr[T]`)
* **Const Raw Pointer**: `ConstPtr[T]` (e.g., `ConstPtr[u8]`)
* **Canonical Pointer Indexing Syntax**: `ptr[i]` (e.g., `ptr[0]`, `ptr[i] = val`) is the canonical pointer dereference and indexing syntax.
* **Type Alias**: `FooPtr: TypeAlias = Ptr[Foo]`
* **Generic Slice View**: `slice[T]` (Lowercase `slice` matches Python 3.9+ built-in generic collection casing: `list[T]`, `dict[K, V]`, `slice[T]`).
* **Fixed-Size Inline Array** (inside `@struct`): `u16[32]`, `u8[8]`
* **Address-Of**: `compiler.addrof(x) -> Ptr[T]` yields a pointer to a local variable `x: T`, translating directly to C `&x`. Used for C out-parameters (e.g. Win32 `WriteFile`'s `lpNumberOfBytesWritten`), since Python has no `&` operator of its own.
* **Packed Struct** (`@cstruct(packed = True)` / `@cunion(packed = True)`): no compiler-inserted padding anywhere in the body (`#pragma pack(push,1)` around the whole struct/union) - the general way to replicate an external ABI's exact byte layout when field types alone don't produce it.
* **Per-Field Alignment** (`Aligned[N, T]`, `@cstruct`/`@cunion` field only): overrides just that field's own C alignment, either below or above `T`'s natural alignment. Resolves transparently to plain `T` everywhere else (arithmetic, comparisons, construction). Cannot be combined with `packed = True` on the same struct (confirmed to diverge between MSVC and clang/gcc - rejected as a compile error).

```metalpy
@cstruct
class WIN32_FIND_DATA:
	dwFileAttributes: u32 = 0
	ftCreationTime: Aligned[4, u64] = 0   # FILETIME's real ABI alignment is 4, not u64's natural 8
	ftLastAccessTime: Aligned[4, u64] = 0
	ftLastWriteTime: Aligned[4, u64] = 0
	nFileSizeHigh: u32 = 0
	nFileSizeLow: u32 = 0
	cFileName: u8[260] = 0
```

```metalpy
written: u32 = 0
WriteFile( handle, buf, count, compiler.addrof( written ), None )
```

```metalpy
HANDLE: TypeAlias = Ptr[None]

@struct
class DynamicTimeZoneInformation:
	Bias: i32
	StandardName: u16[32]
	StandardDate: u16[8]
	StandardBias: i32
	DaylightName: u16[32]
	DaylightDate: u16[8]
	DaylightBias: i32
	TimeZoneKeyName: u16[32]
	DynamicDaylightTimeDisabled: u8
	_pad: u8[3]
```

### Lowering `str` to `ConstPtr[u8]`
When passing a `str` to an FFI function or low-level API parameter expecting `ConstPtr[u8]`, user str.get_cstr().
Because Metal Py `str` is internally stored as a null-terminated C-string, this lowering is safe and zero-cost.

### String, Bytes & Bytearray Methods

`str`/`bytes`/`bytearray` support Python-like search/manipulation methods, built directly on byte-level scanning — real callable methods, not sugar.

* **`str`** has the full set: `find`/`rfind`, `index`/`rindex`, `split`/`rsplit`, `startswith`/`endswith`, `removeprefix`/`removesuffix`, `strip`/`lstrip`/`rstrip`, `replace`, `join`, `partition`/`rpartition`, `isascii`, plus `__contains__` (`sub in s`).
* **`bytes`/`bytearray`** support a smaller subset so far: `find`, `split`, `startswith`, `endswith` (plus `decode(codec)`, `len()`, and indexing/slicing). `strip`/`index`/etc. are not yet implemented for these two types.
* **`find`/`rfind` return `isize`, with `-1` meaning "not found"** (Python's own `str.find()` convention) — they never fail or panic.
* **`index`/`rindex` return `Result[usize, IndexError]`** instead — for callers that consider "not found" itself an error worth propagating via `match`/`.or_return()`/`.is_err()`, rather than a plain `!= -1` conditional. They never panic either.
* `bytes`/`bytearray`'s `find`/`startswith`/`endswith`/`split` all take their needle/prefix/suffix/separator argument as `bytes|bytearray` — either type works interchangeably on either side (a `bytearray` needle against a `bytes` haystack, and vice versa), including bare literal receivers/arguments (`b''.split(...)`, `data.find(b'\r\n')`) directly, with no typed-local workaround needed.

```metalpy
request: bytes = b'GET /hello HTTP/1.1'
if request.find( b' ' ) != -1:
	parts: list[bytes] = request.split( b' ' )   # [b'GET', b'/hello', b'HTTP/1.1']

if not request.startswith( b'GET' ):
	...

s: str = 'deadbeef-dead-beef-dead-beefdeadbeef'
if s.find( 'beef' ) == -1:        # find() never fails - check against -1
	...
match s.index( 'beef' ):          # index() returns a real Result instead
	case Result.Ok( offset ):
		...
	case Result.Err( _ ):
		...
```

### `str()` vs. f-strings — converting values to text

`str` is fully immutable and has no public constructor of its own — not even a copy constructor (`str` never needs deep-copying, since nothing can mutate it; a function that used to "return a copy" of an existing `str` just returns/reuses that same value directly). `str(x)` still works for any `x` with a `__str__` method, though — it's ordinary generic dispatch (`str.__call__[T](x: T) -> str: return x.__str__()`), not a copy constructor.

F-string interpolation (`f"{x}"`) is the more common stringification mechanism: with no format spec (or `!s`), it calls `x.__str__()`; with `!r`/`!a`, `x.__repr__()`; a `str` value is used as-is. Every scalar type has its own `__str__`/`__repr__` except `bool` — fixed-width ints (`i8`/`u8`/`i16`/`u16`/`i32`/`u32`/`i64`/`u64`/`i128`/`u128`/`isize`/`usize`) and `f32`/`f64` all support it directly, alongside the boxed arbitrary-precision `int` and any other class that defines its own `__str__`/`__repr__`.

```metalpy
n: int = int( 42 )
msg: str = f'count: {n}'          # OK - int has __str__

port: u16 = 8080
msg2: str = f'port: {port}'       # OK - u16 has __str__ too
```

### Structs, Unions & Monomorphization
* **Value Struct**: `@struct` decorator creates value-typed structs with fixed byte offsets.
	undefined (yet) future work: struct member alignment
* **Raw Union Type**: `@cunion` decorator creates untagged value layout unions - the low-level primitive, shared storage for every declared field, no discriminant of its own.
* **Enum-with-payload sugar**: `@union` decorator takes Enum-like body syntax (`Variant: Type` or `Variant: Type = tag`) and expands it into a `@struct` wrapping a synthesized `@cunion` payload, plus one `@staticmethod` constructor per variant (`ClassName.Variant(value)`). Tag values are explicit or auto-assigned starting at 0. See `PLAN_UNIONS.md` section J.
* **Union Monomorphization**: Parameters with union types (e.g., `copy_from: bytes | bytearray`) are wrapped
    with an anonymous tagged value layout union (generated via the `@union` sugar above) which implements the methods required by a function
* **Field Visibility**: fields use Python's own leading-underscore convention, not a decorator (a decorator can only precede `def`/`class`, so it can't legally decorate a bare field annotation): a single leading underscore (`_x`) is **protected** (defining class + subclasses), a double leading underscore (`__x`) is **private** (defining class only, `Class.__allocate__()`-style single-pass construction included). `@private` remains valid as a **method** decorator (e.g. `@private def _from_owned_cstr(...)`), since decorating a `def` is legal syntax.
* **Enums with Fallback**: `@enum(T)` supports explicit enumeration with `Other = _` catch-all fallback.

```metalpy
@enum( i32 )
class OwnershipError:
	SharedReference = 0
	AlreadyBorrowed = 1
	UseAfterFree = 2
	DanglingReference = 3
	Other = _

# Result[T, E] is actually compiler-special-cased today (not real compiled
# @union sugar - see PLAN_UNIONS.md), but the sugar shape it's modeled on:
@union
class Result[T, E]:
	Ok: T = 1
	Err: E = 0
```

---

## 3. Ownership, Lifetimes & Memory Management

### Parameter Passing Semantics
* `T`: Borrowed reference (default parameter passing syntax).
* `move[T]`: Explicit ownership transfer; invalidates the caller's binding.
* `copy[T]`: Forced refcount increment/decrement on call boundary.

	callsite must move(t) a move[T] parameter. This isn't needed for copy[T] parameters

### Method Ownership Transfers (`@move`)
The `@move` decorator on a method invalidates the instance (`self`) on invocation and transfers ownership to the method execution context.

Note that bytearray.release() can fail if the caller didn't pass it exclusive ownership, but the @move is unconditional.
In order for the caller to get the object back, the reference is passed back via OwnershipError.SharedReference holding it

```metalpy
class bytearray:
	__data: Ptr[u8]
	__len: usize
	__cap: usize
	
	@move
	def release( self ) -> Result[Ptr[u8],OwnershipError]:
		if compiler.refcount( self ) != 1:
			return Result.Err( OwnershipError.SharedReference( self ))
		ptr = self.__data
		self.__len = 0
		self.__cap = 0
		self.__data = BYTEARRAY_INVALID
		return Result.Ok( ptr )
```

### Destructors (`__del__`) & Memory Hooks
* **Deterministic `__del__`**: Executed immediately when an RC object's refcount reaches `0`.
* **Scope Cleanup Hooks**:
  - `defer`: Executes code block on function exit across all return paths.
  - `errdefer`: Executes code block on function exit only when returning an error (`Result.Err`).
  - There are 2 ways to invoke defer/errdefer (both are handled specially by the compiler):
    1) as a function call for single statements
    2) as a with block (required for multiple statements)

```metalpy
def concat( parts: slice[str] ) -> Result[str, OverflowError]:
	new_len: usize = 0
	# ... calculate length ...
	new_buf: Ptr[u8] = sys.alloc[u8]( new_len + 1 )
	errdefer( sys.free( new_buf ))  # Clean up buffer if subsequent operations return Err

def concat( parts: slice[str] ) -> Result[str, OverflowError]:
	new_len: usize = 0
	# ... calculate length ...
	new_buf: Ptr[u8] = sys.alloc[u8]( new_len + 1 )
	with errdefer:
		sys.free( new_buf )  # Clean up buffer if subsequent operations return Err

def read_and_close( sock: Socket ) -> Result[str, OSError]:
	buf: bytearray = bytearray( 4096 )
	with defer:
		sock.close()  # always runs on exit, whether Ok or Err - multiple statements allowed
		print( 'connection closed' )
	n: usize = sock.recv( buf.get_ptr(), 4096 ).or_return()
	return buf[:n].decode( utf8 ).unwrap( 'utf8' )

each defer/errdefer must only execute once, which means it is a syntax error to put one inside a loop

If users need defer in a loop, they need to move the logic into a different function to get the defer out of the loop

```

---

## 4. Error Handling & Result Semantics

### Result[T, E] & Panic Semantics
Control flow does not use exceptions. Fallible functions return `Result[T, E]`.

`E` has **no constraint on its shape** — it doesn't have to be an `@enum`/`@union` error type. A plain `str` (or any other type) works fine as `E`, though an enum-like error type is the more common/idiomatic choice for anything beyond a quick prototype:

```metalpy
def parse_port( s: str ) -> Result[u16, str]:
	# ... 
	return Result.Err( 'not a valid port number' )  # E can be str, not just an enum
```

* **`unwrap(errmsg: str) -> T`**:
  Checks if `Result` is `Ok`. If `Ok`, returns the value `T`. If `Err`, **triggers a panic** by calling `sys.panic(errmsg)`.
  *Usage*: Used when a failure is logically impossible in context or for early developer assertion failures (aligning with Rust `unwrap()` conventions).
* **`or_return() -> T`**:
  Checks if `Result` is `Ok`. If `Ok`, returns `T`. If `Err`, early-returns from the containing function with the error payload (`compiler.early_return(self._payload.err)`).
* **`unwrap_or(default: T) -> T`**:
  Returns `ok` payload if `is_ok()`, otherwise returns the `default` fallback value.
* **`unwrap_or(default: T|None = None) -> T|None`**:
  alternative unwrap_or() that can return None

```metalpy
# Error Propagation Example:
def encode( s: str, encoding: str ) -> Result[bytes, CodecError]:
	codec = Codec.get( encoding ).or_return()  # Early returns CodecError if Err
	return codec.encode( s )

# Logically Impossible Failure / Panic Example:
def get_assert( self, index: usize ) -> T:
	if index >= self._len:
		sys.panic( 'bad slice index' )
	return self.get_unchecked( index )
```

### Pattern Matching on Results
```metalpy
match src.release():
	case Result.Ok( ptr ):
		return bytes.__allocate__( _data = ptr, _len = length )
	case Result.Err( OwnershipError.SharedReference( src2 )):
		return bytes( src2 )
```

### Tuple Destructuring

A `tuple[T0, T1, ...]`-typed value can be unpacked into individual names, either as a plain assignment (`(a, b) = t` or bare `a, b = t` — both spellings are equivalent) or as a `match`/`case` pattern (`case (a, b):`, including nested inside a class pattern like `case Result.Ok((a, b)):`). This composes with `or_return()`: since `or_return()` on a `Result[tuple[...], E]` returns the tuple, it can be destructured directly.

```metalpy
def accept_one( server: Socket ) -> Result[i32, OSError]:
	( conn, addr ) = server.accept().or_return()   # Result[tuple[Socket, SocketAddr], OSError]
	print( addr.host )
	conn.close()
	return Result.Ok( 0 )

match make_pair():
	case Result.Ok(( a, b )):
		...
	case Result.Err( _ ):
		...
```

Destructuring targets must be plain names — nested tuple targets (`((a,b), c) = t`) and starred targets (`a, *rest = t`) are not supported.

---

## 5. Arithmetic Operators & Context Overrides

Standard arithmetic operators (`+`, `-`, `*`) are fully supported on fixed-width integer types (`u8`, `i32`, `usize`, etc.), but their compiler code generation is governed by **arithmetic contexts**.

### Operator Translation Rules

1. **Default Context (No context block)**:
   Standard arithmetic operations automatically translate to checked operations combined with `.or_return()`.
   - Expression: `a + b`
   - Generated Code: `a.checked_add(b).or_return()`
   - **Compile-Time Error Requirement**: If `a + b` is evaluated in a function whose return type signature does **not** include `OverflowError` in its `Result[T, E]` error payload, compilation fails.

2. **Panicking Checked Context (`with compiler.panic_arithmetic:` / `panic_arithmetic(errmsg)`)**:
   Standard operators translate to checked operations combined with `.unwrap()`.
   - Expression: `a + b`
   - Generated Code: `a.checked_add(b).unwrap(errmsg)`

3. **Wrapping Context (`with compiler.wrap_arithmetic:`)**:
   Standard operators translate to explicit modular wrapping arithmetic methods.
   - Expression: `a + b`
   - Generated Code: `a.wrapped_add(b)`
   NOTE: this is the highest performance option, but we force users to opt-in so they understand what will happen

4. **Saturating Context (`with compiler.saturate_arithmetic:`)**:
   Standard operators translate to saturating arithmetic methods (clamping at boundary min/max values).
   - Expression: `a + b`
   - Generated Code: `a.saturated_add(b)`

```metalpy
# Default context: Requires function to return Result[..., OverflowError]
def add_bounds( a: usize, b: usize ) -> Result[usize, OverflowError]:
	return Result.Ok( a + b )  # Translates to a.checked_add( b ).or_return()

# Wrapping context: Overrides + to wrapped_add
def compute_hash( a: u32, b: u32 ) -> u32:
	with compiler.wrap_arithmetic:
		return a + b  # Translates to a.wrapped_add( b )

# Unwrapped context: Panics on overflow
def alloc_buffer( count: usize ) -> Ptr[u8]:
	with compiler.panic_arithmetic( 'allocation size overflow' ):
		byte_count: usize = count * compiler.sizeof( u32 )  # Translates to checked_mul(...).unwrap(...)
	return sys.alloc[u8]( byte_count )

# Saturating context: Clamps at boundary min/max
def adjust_volume( vol: u8, delta: i8 ) -> u8:
	with compiler.saturate_arithmetic:
		return vol + delta  # Translates to vol.saturating_add( delta )
```

### Scalar Conversions: `T(x)` vs. `x.to_T()`

Converting between scalar types has two spellings with **different
semantics** — they are not interchangeable, and are not two names for the
same operation.

**`T(x)` (constructor-call syntax, also `compiler.cast(T, x)`)** is a
bit-width operation: it succeeds unconditionally whenever the target is the
same width or wider than the source — same-width is a pure bit
reinterpretation, widening sign/zero-extends — in **every** arithmetic
context, including the default checked context. Only a genuinely
*narrowing* conversion (target bit-width smaller than the source's) can
fail, and that stays context-aware exactly like `+`/`-`/`*`:

```metalpy
def widen_and_reinterpret() -> None:
	x: i32 = -1
	y: u32 = u32( x )      # same-width: always 4294967295, every arithmetic context
	z: i32 = i32( y )      # same-width: round-trips back to -1

	small: u8 = 200
	w: i32 = i32( small )  # widening: always succeeds, every context

def narrow_it( big: i32 ) -> Result[u8, OverflowError]:
	return u8( big )       # narrowing: checked, propagates/panics/wraps/clamps like +/-/*
```

A literal argument is checked the same way, using its own natural type
(`i32` for an int literal — the type an unannotated literal always has
elsewhere in the language) as the source width: `u32(-1)` succeeds
(same width as `i32`), but `u8(300)`/`u8(-10000)` are compile-time errors
(narrowing, out of `u8`'s real value range) — even inside an explicit cast.
This is deliberate: an out-of-range literal is treated as a caught bug, not
a silent truncation. If you need a specific narrower bit pattern a literal
can't spell directly, compose two casts — `i8(u8(128))` first fits `128`
into `u8` (in range), then reinterprets that same-width value as `i8`,
giving `i8::MIN`.

**`x.to_T()`** is a value-preserving numeric conversion: it succeeds **iff
the source's mathematical value fits within `T`'s own `[MIN, MAX]`**,
independent of bit width. This is a genuinely different check from `T(x)` —
it can fail for a same-width conversion `T(x)` never fails for, and succeed
for a narrowing conversion whose value happens to fit:

```metalpy
def compare_the_two() -> None:
	neg: i8 = -1
	a: u8 = u8( neg )         # T(x): same-width, always succeeds -> 255
	b = neg.to_u8()           # .to_T(): -1 isn't a valid u8 VALUE -> Err(OverflowError)

	fits: i32 = 200
	c = fits.to_u8()          # .to_T(): narrowing, but 200 fits u8's range -> Ok(200)
```

`.to_T()` returns `Result[T, OverflowError]`, consumed by ambient
arithmetic context exactly like any other `Result`-returning arithmetic
expression (auto-propagate under the default/wrap/saturate contexts,
`.unwrap()`/panic under `panic_arithmetic`) — but unlike `+`/`-`/`*`, the
check itself never varies by context: there's no meaningful "wrapped" or
"saturated" value-range check, the same way `int`'s own `.__floordiv__()`
only ever has one divide-by-zero check regardless of context. Currently
implemented for integer-to-integer conversions only (`i8`/`i16`/`i32`/
`i64`/`i128`/`isize`/`u8`/`u16`/`u32`/`u64`/`u128`/`usize`, every ordered
pair plus self-conversion) — float conversions are a planned follow-up.

---

## 6. Foreign Function Interface (FFI) & Target Conditioning

### C & Dynamic Library External Declarations (`@extern`)
Functions declared with `@extern` map directly to native C symbols or library symbols. The function body is omitted using an ellipsis `...`.

```metalpy
# System C Runtime FFI:
@extern( 'c', 'malloc' )
def malloc( size: usize ) -> Ptr[u8]:
	...

# Platform DLL FFI:
@extern( 'kernel32', 'HeapAlloc' )
def HeapAlloc( hHeap: HANDLE, dwFlags: u32, dwBytes: usize ) -> Ptr[u8]:
	...

# Vendored/3rd-party DLL FFI, with a runtime bundling hint (single DLL,
# or a list when the vendored library has its own further DLL
# dependencies that also need to ship) and a 3rd-party license notice
# hint (same single-or-list shape):
@extern( 'tcl86t', 'Tcl_CreateInterp', dll = [ 'tcl86t.dll', 'zlib1.dll' ], notice = [ 'TCL', 'ZLIB' ] )
def Tcl_CreateInterp() -> Ptr[None]:
	...
```

`dll=` is optional and independent from the `lib` argument: `lib` ('tcl86t' above) is the
import library linked against at build time, while `dll=` names the bare runtime DLL
filename(s) that must be loadable when the built program actually runs - these can live in
a different directory than `lib` on the build machine (e.g. a vendored library's `.lib` and
`.dll` shipped separately), and a real DLL commonly has its own further DLL dependencies
(e.g. `tcl86t.dll` also needs `zlib1.dll`) that must be listed explicitly too if they need
bundling - the compiler deliberately never scans a DLL's own import table to discover these
automatically, since a real dependency list mixes genuinely-vendored files with system
components (`kernel32.dll`, various `api-ms-win-crt-*.dll` forwarders, ...) that must never
be bundled, and reliably telling those apart without a maintained blacklist isn't possible.
An explicit, author-supplied list sidesteps the question entirely - including the freedom to
deliberately leave something like `VCRUNTIME140.dll` off the list if it's assumed already
present on target machines. When a function declaring `dll=` is actually reached and
compiled into the program, `mpy`'s build step locates each named DLL on `PATH` and copies it
next to the built executable; a function that's declared but never called contributes
nothing, and a declared DLL that can't be found anywhere on `PATH` fails the build. System
DLLs simply never declare `dll=` in the first place.

`notice=` is likewise optional and independent - both from `lib` and from `dll=`. Each
identifier (`'TCL'`, `'ZLIB'` above) resolves to a `licenses/<NAME>.txt` file at the metalpy
installation root, containing that dependency's actual license text. It's a separate
declaration from `dll=` on purpose: a notice can apply to several otherwise-unrelated DLL
dependencies (e.g. `'ZLIB'` covers `zlib1.dll` regardless of which library happens to bundle
it, not just Tcl/Tk), so collapsing the two ideas would either duplicate license text per
dependency or force guessing which `dll=` entries share a notice. When a function declaring
`notice=` is actually reached and compiled in, `mpy`'s build step combines every referenced
notice file into one `dist/THIRD-PARTY-LICENSES.txt` alongside the bundled DLLs; a notice
that can't be found fails the build, the same as a missing `dll=` entry.

### Forcing CRT Linking (`@requires_crt`)
On Windows, a build with no reachable `@extern('c', ...)` call links freestanding by default
(no CRT, a hand-rolled entry point - see `mpy.py`'s `--crt` flag to override this from the
command line). `@requires_crt` lets a library function force CRT linking for the whole build
from *inside* the language instead, whenever that function is itself actually reachable -
useful for something whose need for the CRT isn't expressed as an ordinary `@extern('c', ...)`
call at all (e.g. MSVC's `__chkstk` stack-probing support routine, silently required by any
function with a large enough local stack frame, which a freestanding MSVC build has no way to
supply):

```metalpy
@requires_crt
def uses_a_large_stack_frame() -> i32:
	...
```

Reachability-gated the same way `@extern`'s own `lib`/`dll`/`notice` declarations are: a
`@requires_crt` function that's never called from anything reachable from `main()` has no
effect. Not supported on an `@inline` function - it's spliced directly into each call site
and never becomes its own reachable unit, so the flag would never actually fire. No effect on
non-Windows targets, which have no freestanding/no-CRT build mode to override in the first
place.

### Target Platform Conditioning (`@compiler.target` & `compiler.target`)

Target conditioning supports fine-grained keyword filters at the function level, as well as compile-time property queries inside function scopes.

A python declaration of compiler.target could look like this:

# Base specifier types
ArchSpec = Literal['x86_64', 'x86', 'aarch64', 'arm', 'riscv64', 'riscv32']
OSSpec = Literal['windows', 'linux', 'darwin', 'freebsd', 'openbsd', 'none']
EnvSpec = Literal['msvc', 'gnu', 'musl', 'none']
VendorSpec = Literal['pc', 'apple', 'unknown']
FamilySpec = Literal['unix', 'windows', 'wasm']
BitsSpec = Literal[16, 32, 64]
EndianSpec = Literal['little', 'big']

# Parameter type helper accepting single value, tuple, or boolean (produced by `not`)
TargetValue[T] = Union[T, tuple[T, ...], bool, None]

class TargetQuery:
	arch: ArchSpec
	os: OSSpec
	env: EnvSpec
	vendor: VendorSpec
	family: FamilySpec
	bits: BitsSpec
	endianness: EndianSpec
	posix: bool

class compiler:
	target: TargetQuery
	
	@staticmethod
	def target(
		*,
		arch: TargetValue[ArchSpec] = None,
		os: TargetValue[OSSpec] = None,
		env: TargetValue[EnvSpec] = None,
		vendor: TargetValue[VendorSpec] = None,
		family: TargetValue[FamilySpec] = None,
		bits: TargetValue[BitsSpec] = None,
		endianness: TargetValue[EndianSpec] = None,
		posix: bool | None = None,
	) -> None:
		...

```metalpy
@compiler.target( os = not 'windows' )
def free( ptr: Ptr[u8] ) -> None:
	from crt import free as _crt_free
	_crt_free( ptr )

@compiler.target( os = not ( 'freebsd', 'openbsd' ) )
def get_thread_id() -> u64:
	...

@compiler.target( family = 'unix', arch = not ( 'x86', 'arm' ), bits = 64 )
def sys_mmap_64(...) -> Ptr[u8]:
	...

```

---

## 7. Class Construction & Inheritance Model

* **Single Implementation Inheritance**: Classes support single inheritance only (`class CodecError( sys.Error ):`).
* **Interfaces & Abstract Methods**: Interface contracts use `@abstractmethod` for method stubs.
* **Fallible `__init__()` Construction**:
  - Memory is allocated **before** `__init__()` is called so that instance properties can be set via `self`.
  - If `__init__()` is declared to return `Result[None, E]`, object construction syntax `Foo(...)` returns `Result[Foo, E]`.
  - If `__init__()` fails and returns `Result.Err(e)`, the reference count of the allocated instance drops to `0` upon function exit, triggering immediate resource cleanup. However, __del__() is ***NOT*** called if __init__() returns Result.Err().
* **Private `Class.__allocate__()` Single-Pass Allocation**:
  - `Class.__allocate__()` allocates memory and initializes fields in a single pass.
  - **Access Restriction**: `Class.__allocate__()` is **strictly private** and can **only** be called from methods inside the same class (e.g., static factory methods). External calls from outside the class are compile errors.

```metalpy
class Result[T, E]:
	__payload: ResultPayload[T, E]
	__tag: u8
	
	@staticmethod
	def Ok( val: T ) -> Result[T, E]:
		# Allowed: __allocate__() is called from inside a method of Result class
		res: Result[T, E] = Result.__allocate__(
			__payload = ResultPayload( ok = val ),
			__tag = 0,
		)
		return res
```

---

## 8. Program Entry Points & Standard Library Basics

### Module-Level (Top-Level) Execution Rules

At module scope, `import`/`from...import`, `class`/`def` definitions (including `@compiler.target`-decorated ones), `pass`, and global variable declarations/initializations are allowed. A **non**-constant `if`/`match`, loops, `AugAssign` (`x += 1`), and any bare expression-statement that isn't a literal (a call with no assignment target, e.g. a top-level `print('hi')` line) are all compile errors at module scope ("unsupported statement here") — a bare call like that can never actually run, so it's rejected rather than silently compiled away. A compile-time-constant `if`/`match` is folded down to just its taken branch's statements before anything else runs, so it's effectively allowed too.

A global variable's initializer is **not** restricted to a compile-time constant — it can call ordinary functions at real program-startup time; the compiler synthesizes a `__metalpy_init_<name>()` function per initializer, called before `main()` runs.

```metalpy
MAX_RETRIES: i32 = 3               # compile-time constant - fine
HANDLE: TypeAlias = Ptr[None]      # TypeAlias - just an ordinary AnnAssign

def compute_default() -> i32:
	return 42

DEFAULT: i32 = compute_default()   # a real function call, evaluated at program startup - also fine

@compiler.target( os = 'windows' )
class WindowsSpecific:
	pass
```

A bare literal statement (a module/class docstring, or a `...` stub placeholder) is still silently accepted and is a genuine no-op, same as ordinary Python. There is still no working idiom for "run this one statement at module scope" — the only way to run code at program-startup time (before `main()`) is a global variable's own initializer, above.

### `main()` Entry Point

The program entry point is a function named `main`. Its return type must be `None` or a scalar integer type (`i32`, `u8`, `u32`, etc.) — never `Result[...]` or any other shape. `main` compiles directly to C's real `int main(...)` — the OS/CRT invokes it exactly like any C program's `main`, with **no** metalpy-level driver call needed (a top-level `sys.exit(main())`-style statement is not part of this codebase's actual convention, and per the module-scope rule just above, a bare one would be a compile error anyway). A `None`-returning `main()` synthesizes `return 0;`; any other declared return type has its `Return`'s operand emitted as the C `int main`'s return value directly. The return-type restriction isn't (yet) enforced as a dedicated, friendly compiler diagnostic — declaring `main() -> Result[...]` compiles cleanly through this compiler's own IR and only fails once the *generated C* is compiled, with a much less friendly C-level type error.

```metalpy
def main() -> i32:
	print( 'hello' )
	return 0
```

### `print()`

`def print(msg: str, end: str = '\n') -> None` — a single positional `str` argument only, no `*args`, no `sep`. Combine multiple values into one string via an f-string first.

```metalpy
def log_connection( host: str ) -> None:
	print( f'Connection from {host}' )   # not print( 'Connection from', host )
```

### `sys.exit()`

`def exit(code: u32) -> NoReturn` — takes a `u32` exit code, not `i32`.

### `codecs` — encoding/decoding text

`from codecs.utf8 import utf8` imports the (already-constructed, stateless, shared) `utf8` codec instance. `str.encode(codec: Codec = utf8) -> Result[bytes, CodecError]` and `bytes.decode(codec: Codec = utf8) -> Result[str, CodecError]` / `bytearray.decode(codec: Codec = utf8) -> Result[str, CodecError]` both default to it already, so the explicit import is only needed to pass it by name or use a non-default codec.

```metalpy
from codecs.utf8 import utf8

body: str = 'hello\r\n'
encoded: bytes = body.encode( utf8 ).unwrap( 'ASCII text is always valid utf-8' )
sock.send_all( encoded.get_const_ptr(), encoded.__len__() ).or_return()
```

---

## 9. Generics

each specialization of generics produces distinct code in the executable. Therefore,
large complicated generic class can explode executable size. stdlib generics
like list[T] and dict[K,V] have an underlying *raw* implementation where most
of the code lives and the generics are thin type-safe wrappers. A user doesn't
have to follow this pattern, but its how we are trying to keep the stdlib small.

Note in the Bar[T] example, a specialization's type signature can be inferred
from parameters if possible. This isn't possible with the Foo example.

```metalpy
class Foo[T]:
	pass

foo1 = Foo[u8]()
foo2 = Foo[i32]()

class Bar[T]:
	def __init__( self, value: T ) -> None:
		...

bar1 = Bar( u32( 17 ))
bar2 = Bar( str( 'foo' ))
```

---

## 10. Function Overloads

Its possible to have functions with the same name.

Two functions can have the same name as long as they have distinct parameters/types.

For example:

```metalpy
def panic( msg: ConstPtr[u8] ) -> None:
	sys.stderr.write( msg )

def panic( msg: str ) -> None:
	sys.stderr.write( msg.get_cstr() )
```

The compiler chooses which one to call based on the parameter types at compile time.

In the case of anonymous unions, the compiler will generate runtime logic to determine
which variant to call.

There is also an @overload operator

The purpose of this is to give more flexibility in directing the compiler to the correct
function implementation. @overload methods have slightly lighter requirements but
we recognize they may be indistinguishable from non-@overload methods.

@overload methods:
	* They are chosen first-come first-serve instead of exact pattern match
		** this changes the logic of how they're chosen, which may be useful in some situations
		** including in the runtime disambiguation logic
	* if there's no body, they are bound to a non-@overload version
		** their parameter type signature must match exactly one non-@overload or it is a compilation error
	* must not completely shadow a following @overload
