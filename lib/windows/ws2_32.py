# lib/windows/ws2_32.py — Winsock (ws2_32.dll) extern declarations
#
# Structured the same way lib/windows/kernel32.py is: hand-declared prototypes
# (no header= — sockaddr* out-params stay opaque Ptr[None], same posture
# kernel32.py already takes for its own structured out-params like
# lpNumberOfBytesWritten), one real DLL export per @extern. mpy.py's existing
# compiler.extern_libs → linker-flag logic auto-adds ws2_32.lib once any
# @extern('ws2_32', ...) below is lowered — no separate linker wiring needed.
#
# inet_pton/inet_ntop (not InetPtonA/InetNtopA) — confirmed by a real
# compile+link: the A-suffixed ANSI names fail to resolve against this SDK's
# ws2_32.lib, while the lowercase POSIX-style names (exported by ws2_32.dll
# since Vista) link and work identically. Using the same names as the POSIX
# side of lib/socket.py is a bonus, not just a workaround.

import compiler

# SOCKET is UINT_PTR on real Winsock (pointer-sized unsigned) — usize matches
# that on every target this compiles for.
SOCKET: TypeAlias = usize

INVALID_SOCKET: SOCKET = usize( -1 )
SOCKET_ERROR: i32 = -1

@extern( 'ws2_32', 'WSAStartup' )
def WSAStartup(
	wVersionRequested: u16,
	# WSADATA is a large, version-varying struct (its shape differs between
	# the ANSI/Unicode Winsock headers) nothing here ever reads a single
	# field from — only its address is ever passed, and never touched again.
	# The natural "opaque C type" tool (compiler.c_type(header='winsock2.h'),
	# same one lib/posix/pthread.py's pthread_t uses) does NOT work here —
	# confirmed by a real compile: winsock2.h transitively drags in
	# windows.h, whose GetModuleHandleA/GetProcAddress/LoadLibraryA
	# declarations conflict with prototypes this compiler's own generated C
	# already emits, a hard compile error. So lpWSAData stays a bare
	# Ptr[None]; the caller (lib/socket.py's _ensure_wsa_started) just needs
	# to hand WSAStartup a scratch buffer comfortably larger than any real
	# WSADATA (well under 512 bytes on any Windows version).
	lpWSAData: Ptr[None],
) -> i32:
	...

@extern( 'ws2_32', 'WSACleanup' )
def WSACleanup() -> i32:
	...

# NOT kernel32.GetLastError — Winsock functions report failure through this,
# their own documented error-retrieval API (see lib/socket.py's callers).
@extern( 'ws2_32', 'WSAGetLastError' )
def WSAGetLastError() -> i32:
	...

@extern( 'ws2_32', 'socket' )
def socket(
	af: i32,
	type: i32,
	protocol: i32,
) -> SOCKET:
	...

@extern( 'ws2_32', 'closesocket' )
def closesocket(
	s: SOCKET,
) -> i32:
	...

@extern( 'ws2_32', 'bind' )
def bind(
	s: SOCKET,
	name: Ptr[None],
	namelen: i32,
) -> i32:
	...

@extern( 'ws2_32', 'listen' )
def listen(
	s: SOCKET,
	backlog: i32,
) -> i32:
	...

@extern( 'ws2_32', 'accept' )
def accept(
	s: SOCKET,
	addr: Ptr[None],
	addrlen: Ptr[i32],
) -> SOCKET:
	...

@extern( 'ws2_32', 'connect' )
def connect(
	s: SOCKET,
	name: Ptr[None],
	namelen: i32,
) -> i32:
	...

@extern( 'ws2_32', 'send' )
def send(
	s: SOCKET,
	buf: ConstPtr[u8],
	len: i32,
	flags: i32,
) -> i32:
	...

@extern( 'ws2_32', 'recv' )
def recv(
	s: SOCKET,
	buf: Ptr[u8],
	len: i32,
	flags: i32,
) -> i32:
	...

@extern( 'ws2_32', 'sendto' )
def sendto(
	s: SOCKET,
	buf: ConstPtr[u8],
	len: i32,
	flags: i32,
	to: Ptr[None],
	tolen: i32,
) -> i32:
	...

@extern( 'ws2_32', 'recvfrom' )
def recvfrom(
	s: SOCKET,
	buf: Ptr[u8],
	len: i32,
	flags: i32,
	from_: Ptr[None],
	fromlen: Ptr[i32],
) -> i32:
	...

@extern( 'ws2_32', 'shutdown' )
def shutdown(
	s: SOCKET,
	how: i32,
) -> i32:
	...

@extern( 'ws2_32', 'getsockname' )
def getsockname(
	s: SOCKET,
	name: Ptr[None],
	namelen: Ptr[i32],
) -> i32:
	...

@extern( 'ws2_32', 'setsockopt' )
def setsockopt(
	s: SOCKET,
	level: i32,
	optname: i32,
	optval: ConstPtr[u8],
	optlen: i32,
) -> i32:
	...

@extern( 'ws2_32', 'inet_pton' )
def inet_pton(
	family: i32,
	pszAddrString: ConstPtr[u8],
	pAddrBuf: Ptr[None],
) -> i32:
	...

# Return type is a plain ConstPtr[u8], NOT ConstPtr[u8]|None - a real,
# reproducible compiler bug (confirmed with a minimal repro outside this
# file): a `T|None` union as an @extern function's RETURN type corrupts the
# actual pointer value (it no longer equals the buffer that was passed in),
# while a bare Ptr/ConstPtr return still supports `is None`/`== None`
# directly (same as lib/sys.py's `_alloc(...) -> Ptr[u8]`, checked with a
# plain `if ptr is None:`) since raw pointers are inherently nullable - no
# union wrapper needed or safe to use here.
@extern( 'ws2_32', 'inet_ntop' )
def inet_ntop(
	family: i32,
	pAddr: Ptr[None],
	pStringBuf: Ptr[u8],
	StringBufSize: usize,
) -> ConstPtr[u8]:
	...

# getaddrinfo/freeaddrinfo — hostname resolution. Real declaration lives in
# ws2tcpip.h (confirmed against the real SDK header), but header='ws2tcpip.h'
# is NOT usable here - a real compile confirms it drags in windows.h (same
# transitive-include problem WSAStartup's own comment above already hit),
# which collides with prototypes this compiler's own generated C emits for
# unrelated functions (memset/memcpy/WriteFile/SetConsoleOutputCP etc), a
# hard compile error. So this stays a plain @extern with no header=, exactly
# like every other Winsock function in this file: hints/res are opaque
# Ptr[None]/Ptr[Ptr[None]] on our side (lib/socket.py owns the actual
# ADDRINFOA field layout via its own _AddrInfo cstruct, since that's memory
# this compiler's own generated code reads directly) - the linker resolves
# the real ws2_32.dll export regardless of our prototype's exact spelling,
# same as inet_pton/inet_ntop above already rely on.
@extern( 'ws2_32', 'getaddrinfo' )
def getaddrinfo(
	pNodeName: ConstPtr[u8],
	pServiceName: ConstPtr[u8],
	pHints: Ptr[None],
	ppResult: Ptr[Ptr[None]],
) -> i32:
	...

@extern( 'ws2_32', 'freeaddrinfo' )
def freeaddrinfo(
	pAddrInfo: Ptr[None],
) -> None:
	...

# FIONBIO - winsock2.h's _IOW('f', 126, u_long), 0x8004667E. i32 (not u32):
# ioctlsocket's real signature takes `long cmd`, matching every other
# Winsock parameter here already using i32/u32 per the real Win32 type,
# not a POSIX-style unsigned assumption.
FIONBIO: i32 = -2147195266   # 0x8004667E as a signed i32 - see comment above

@extern( 'ws2_32', 'ioctlsocket' )
def ioctlsocket(
	s: SOCKET,
	cmd: i32,
	argp: Ptr[u32],
) -> i32:
	...

# WSAPoll - fdArray stays opaque Ptr[None] (same posture as every sockaddr*
# out-param above): WSAPOLLFD's own field layout is owned by whichever
# application module actually reads/writes it (lib/poller.py), cast to
# Ptr[None] at the call site - not declared here, avoiding a circular
# import (poller.py already needs to import THIS module for WSAPoll
# itself).
@extern( 'ws2_32', 'WSAPoll' )
def WSAPoll(
	fdArray: Ptr[None],
	fds: u32,
	timeout: i32,
) -> i32:
	...
