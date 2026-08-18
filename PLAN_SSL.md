SSL/TLS library (lib/ssl.py) — scope and roadmap

Context

The user wants TLS/SSL support eventually available in MetalPy's stdlib, following
directly on PLAN_HTTP_CLIENT.md (which scoped lib/http/client.py). That plan
explicitly punted HTTPS/TLS out of scope as "a large separate undertaking...
deferred to its own future plan doc," and speculated without committing that the
approach would be "schannel on Windows / some TLS lib on POSIX."

This started as a scoping/roadmap-only pass (mirroring PLAN_HTTP_CLIENT.md's own
early scoping session), written before lib/socket.py existed. Phase 0 and Phase 1
(Windows/Schannel) have since landed for real - see "What actually landed" below.

Backend strategy — native OS TLS, not a bundled OpenSSL

The user's opening assumption was that this requires bundling a vendored copy of
OpenSSL with MetalPy. Investigation of the existing lib/ convention
(lib/windows/kernel32.py, lib/crt.py, lib/windows/ntdll.py, lib/posix/pthread.py)
shows every native binding in this codebase @extern's directly into a library
that is already present on the target system — always-loaded Windows DLLs
(kernel32, ntdll, secur32, ...) via plain extern function declarations, or the
system's own libc/pthread on POSIX. There is no precedent anywhere in this
codebase for vendoring or statically bundling third-party C source. lib/crt.py's
own comment on ICU (str.upper()/.lower() unicode casing) explicitly reasons
about why an optional system library should be probed with
compiler.has_library(...) rather than bundled — the exact mechanism this plan
leans on for the one platform (Linux) that has no OS-native TLS API.

This was raised with the user directly, against their original OpenSSL-bundling
assumption. They confirmed the recommended, codebase-consistent direction:

  - Windows: Schannel via SSPI (secur32.dll) — always present, matches
    kernel32.py's "CRT-free, nothing extra needed" story exactly. LANDED - see
    below.

  - macOS: Secure Transport or Network.framework — always present, no bundling.
    Flagged as needing a research spike at implementation time: this plan did
    not verify current API shape or deprecation status, and Network.framework's
    async-callback design may not map cleanly onto the blocking synchronous
    socket contract lib/socket.py actually shipped with. Treat this plan's macOS
    section as provisional, not final. NOT STARTED.

  - Linux: no OS-native TLS API exists, so link against the system's own
    OpenSSL via the existing @compiler.target(has_library=('ssl', 'SSL_new'))
    / compiler.has_library(...) probing mechanism — already implemented and
    tested (discovery.py's _matches_has_library / discovery_test.py's
    has_library tests, linker_c.py's has_symbol). SSL_CTX_new, SSL_new,
    SSL_set_fd, SSL_connect, SSL_read, SSL_write, SSL_get_error, SSL_shutdown,
    SSL_free, SSL_CTX_free. This genuinely requires libssl-dev on the build
    machine. NOT STARTED.

  - Linking against system OpenSSL uniformly on all three platforms (skipping
    Schannel/Secure Transport) was considered and rejected: unlike Linux,
    Windows and macOS don't ship OpenSSL by default, so that would regress
    kernel32.py's CRT-free "nothing extra needed" Windows story for no benefit.

What actually landed (lib/ssl.py, ssl_test.py)

Phase 0 and Phase 1 (Windows/Schannel) landed together, once lib/socket.py
(commit 863bfc8 and its follow-ups) made a real client possible to build and
test end-to-end.

  Phase 0 — SSLError, a single portable @enum (HandshakeFailed,
    CertificateVerifyFailed, CertificateExpired, HostnameMismatch,
    ProtocolError, Closed, Other = _). Not OS-split like builtins.__errors.
    OSError - these concepts are backend-agnostic even though three different
    native backends will eventually produce them. Covered by
    ssl_test.py's SSLPhase0Tests (enum-value and match-dispatch checks, zero
    prerequisites).

  Phase 1 — the Windows Schannel/SSPI backend: SSLContext.create_default_context()
    and SSLSocket.wrap_socket()/send()/send_all()/recv()/close(), wrapping an
    already-connected socket.Socket. Every struct this needed (SecHandle/
    CredHandle/CtxtHandle, SecBuffer, SecBufferDesc, SCHANNEL_CRED,
    SecPkgContext_StreamSizes) was NOT hand-derived from memory - every
    sizeof()/offsetof() was cross-checked with a throwaway C probe compiled
    against the real Windows SDK headers (schannel.h/sspi.h via cl.exe) before
    being encoded as @cstruct declarations, and the full handshake/encrypt/
    decrypt call sequence (the InitializeSecurityContext loop, EncryptMessage/
    DecryptMessage buffer wrangling, SECBUFFER_EXTRA handling) was validated
    end-to-end with a standalone C client first - a real TCP connection, a real
    Schannel handshake, and a real decrypted HTTP response, against
    example.com - before any of it was transcribed into MetalPy. Certificate-
    failure error mapping (SEC_E_CERT_EXPIRED -> CertificateExpired,
    SEC_E_WRONG_PRINCIPAL -> HostnameMismatch, SEC_E_UNTRUSTED_ROOT ->
    CertificateVerifyFailed) was confirmed the same way, against
    badssl.com's expired/wrong-host/self-signed fixtures - not guessed.

    Covered by ssl_test.py's SSLWindowsHandshakeTests: a real handshake +
    encrypted HTTP/1.1 round trip against example.com (exercising multiple
    DecryptMessage calls across a chunked response), and the three
    certificate-failure paths against badssl.com. These are the only tests in
    this codebase that dial out to the real network rather than looping back
    locally - there is no local TLS server to loop back against without also
    implementing server-side Schannel (out of scope; lib/ssl.py is client-only),
    and badssl.com exists specifically as a public fixture for this kind of
    testing. Gated on os.name == 'nt' (no other backend exists yet) and
    METALPY_TEST_NETWORK (set to '0' to skip if network egress isn't
    available).

  A private _CipherBuf class (local to lib/ssl.py, not shared with
    lib/socket.py's RecvBuffer) accumulates raw ciphertext read off the socket
    and supports consume() (shift an already-processed prefix off the front) -
    needed by both the handshake loop and the post-handshake decrypt loop.
    Kept private rather than extending RecvBuffer, matching this codebase's own
    established precedent of each stdlib consumer keeping its own private
    growable buffer (see lib/http/client.py's _GrowableBuffer per
    PLAN_HTTP_CLIENT.md) rather than sharing one - keeps lib/ssl.py isolated
    from lib/socket.py's already-shipped, tested code.

  One real gap versus the original draft sketch: wrap_socket()'s
    server_hostname is a required `str`, not `str|None` - the |None case (skip
    SNI/hostname verification) has no real caller yet (HTTPSConnection will
    always have a real hostname) and was dropped to avoid adding untested,
    unused surface. Can be added back if a real caller needs it.

  One real gotcha hit while landing this, worth recording: a struct field
    literally named `__in` silently vanished during C compilation (MSVC/
    Windows SDK headers define `__in` as an empty SAL annotation macro via
    windows.h/specstrings.h, so `struct Foo* __in;` preprocessed down to
    `struct Foo* ;` - a syntax error whose downstream fallout looked like the
    whole enclosing struct was never defined). Renamed to `__cipher`. Worth
    keeping in mind for any future Windows-facing MetalPy code: `__in`/`__out`/
    `__inout` (and their `_opt`-suffixed and `_In_`/`_Out_`-style SAL 2.0
    cousins) are unsafe field/parameter names on Windows targets.

Scope for this pass

In scope (now landed for Windows):
  - lib/ssl.py's SSLError, SSLContext, SSLSocket (Windows/Schannel backend).
  - ssl_test.py covering both Phase 0 (pure) and Phase 1 (real network).

Out of scope (still deferred):
  - Linux (has_library-gated OpenSSL) and macOS (Secure Transport/
    Network.framework) backends - see "Backend strategy" above.
  - Client-certificate authentication (load_cert_chain) - stretch goal, same
    treatment PLAN_HTTP_CLIENT.md gave multipart files= uploads.
  - Wiring into lib/http/client.py's reserved verify=/HTTPSConnection path -
    a separate follow-up once this is stable.
  - Any change to the compiler/linker pipeline - has_library/has_symbol already
    do everything Linux support will need; nothing new required there.

Draft API (as implemented)

Result[T,E]-style throughout, no exceptions — matching every other stdlib
module and PLAN_HTTP_CLIENT.md's own Response/Session sketch.

  @enum( i32 )
  class SSLError:
      HandshakeFailed         = 1
      CertificateVerifyFailed = 2
      CertificateExpired      = 3
      HostnameMismatch        = 4
      ProtocolError           = 5
      Closed                  = 6
      Other = _

  class SSLContext:
      @staticmethod
      def create_default_context() -> Result[SSLContext, SSLError]: ...
      # loads the platform trust store implicitly (Windows: system cert store
      # via Schannel's own automatic validation)

  class SSLSocket:
      @staticmethod
      def wrap_socket(
          ctx: SSLContext,
          sock: socket.Socket,
          server_hostname: str,
      ) -> Result[SSLSocket, SSLError]: ...

      def send( self, buf: ConstPtr[u8], count: usize ) -> Result[usize, SSLError]: ...
      def send_all( self, buf: ConstPtr[u8], count: usize ) -> Result[None, SSLError]: ...
      def recv( self, buf: Ptr[u8], count: usize ) -> Result[usize, SSLError]: ...
      def close( self ) -> None: ...
      # __del__ auto-closes, same idiom as BinaryReader/BinaryWriter in
      # lib/builtins/__File.py and socket.Socket itself

This is what makes PLAN_HTTP_CLIENT.md's reserved verify=True kwarg and its
deferred HTTPSConnection real, once wired up (still a follow-up) — HTTPSConnection
becomes a thin wrapper: connect a raw socket.Socket, then SSLSocket.wrap_socket()
it before handing the result to the existing request/response read/write path.

Remaining phased roadmap

  Phase 2 — Linux backend: has_library-gated OpenSSL extern bindings
    (SSL_CTX_new/SSL_new/SSL_set_fd/SSL_connect/SSL_read/SSL_write/
    SSL_get_error/SSL_shutdown/SSL_free/SSL_CTX_free).

  Phase 3 — macOS backend, prefixed by the research spike flagged above
    (confirm Secure Transport vs Network.framework's actual current shape
    before committing to bindings).

  Phase 4 (deferred/future plan doc) — wire into lib/http/client.py's reserved
    verify=/HTTPSConnection path. Also: client-certificate auth
    (load_cert_chain), session resumption/ticket caching, ALPN (relevant for a
    future HTTP/2 story PLAN_HTTP_CLIENT.md already marked out of scope).

Testing approach

  Same idiom as PLAN_HTTP_CLIENT.md: root-level *_test.py files compiled and
  run for real via test_support.RealCompileMixin.assert_programs_run (pattern
  in emitter_c_test.py) — exit-code based, no lib/tests/ directory.

  Phase 0 needed no network dependency, same as any other pure-data test in
  this codebase.

  Phase 1's tests are a deliberate departure from every other *_test.py here:
  they dial out to real public hosts (example.com, badssl.com) rather than
  looping back locally, because there's no local TLS server to loop back
  against without implementing server-side Schannel too (out of scope for a
  client-only library). Gated behind METALPY_TEST_NETWORK=0 for environments
  without network egress. Future Linux/macOS backend tests should follow the
  same pattern.
