SSL/TLS library (lib/ssl.py) — scope and roadmap

Context

The user wants TLS/SSL support eventually available in MetalPy's stdlib, following
directly on PLAN_HTTP_CLIENT.md (which scoped lib/http/client.py). That plan
explicitly punted HTTPS/TLS out of scope as "a large separate undertaking...
deferred to its own future plan doc," and speculated without committing that the
approach would be "schannel on Windows / some TLS lib on POSIX."

This started as a scoping/roadmap-only pass (mirroring PLAN_HTTP_CLIENT.md's own
early scoping session), written before lib/socket.py existed. Phase 0, Phase 1
(Windows/Schannel), and Phase 2 (Linux/OpenSSL) have since landed for real - see
"What actually landed" below.

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
    section as provisional, not final. NOT STARTED - no macOS machine available
    to this session; deferred until one is.

  - Linux: no OS-native TLS API exists, so link against the system's own
    OpenSSL via the existing @compiler.target(has_library=('ssl', 'SSL_new'))
    / compiler.has_library(...) probing mechanism — already implemented and
    tested (discovery.py's _matches_has_library / discovery_test.py's
    has_library tests, linker_c.py's has_symbol). SSL_CTX_new, SSL_new,
    SSL_set_fd, SSL_connect, SSL_read, SSL_write, SSL_get_error, SSL_shutdown,
    SSL_free, SSL_CTX_free. This genuinely requires libssl-dev on the build
    machine. LANDED - see below.

  - Linking against system OpenSSL uniformly on all three platforms (skipping
    Schannel/Secure Transport) was considered and rejected: unlike Linux,
    Windows and macOS don't ship OpenSSL by default, so that would regress
    kernel32.py's CRT-free "nothing extra needed" Windows story for no benefit.

What actually landed (lib/ssl.py, ssl_test.py)

Phase 0 and Phase 1 (Windows/Schannel) landed together, once lib/socket.py
(commit 863bfc8 and its follow-ups) made a real client possible to build and
test end-to-end. Phase 2 (Linux/OpenSSL) landed in a follow-up session.

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

  Phase 2 — the Linux OpenSSL backend, same SSLContext/SSLSocket public shape
    as Windows (SSLSocket.wrap_socket()/send()/send_all()/recv()/close()) but
    a completely different internal shape: every OpenSSL type touched here
    (SSL_CTX*, SSL*, SSL_METHOD*) is fully opaque from MetalPy's side - no
    struct layouts to derive at all, only function signatures and integer
    constants, both cross-checked against the real openssl/ssl.h and
    openssl/x509_vfy.h (OpenSSL 3.5, Debian 13 trixie, via WSL) rather than
    guessed. The same "validate with a standalone C client first" discipline
    applied here too: a real TCP connection, a real OpenSSL handshake with
    peer verification actually turned on (SSL_CTX_set_verify(...,
    SSL_VERIFY_PEER, ...) - OFF by default in raw OpenSSL, a well-known
    footgun this wrapper doesn't expose), and a real decrypted HTTP response
    against example.com, plus the same three badssl.com fixtures, before any
    of it was transcribed into MetalPy.

    One real OpenSSL-specific gotcha worth recording: SSL_get_error() alone
    does NOT distinguish which certificate problem occurred - it returns the
    same generic SSL_ERROR_SSL (1) for all three badssl.com failures
    (confirmed empirically with the C reference, not assumed from docs). The
    actual reason lives in a SEPARATE call, SSL_get_verify_result(), which
    returns a real X509_V_ERR_* code (X509_V_ERR_CERT_HAS_EXPIRED = 10,
    X509_V_ERR_HOSTNAME_MISMATCH = 62, and a third value for the self-signed
    case that isn't either of those - hence _map_ssl_error()'s trailing
    catch-all to CertificateVerifyFailed rather than a third named check).

    SSL_set_tlsext_host_name (SNI) is a macro in real OpenSSL headers, not an
    exported symbol - it expands to SSL_ctrl(ssl, 55, 0, name) (confirmed
    against openssl/tls1.h) - so lib/ssl.py calls SSL_ctrl directly with
    those constants rather than @extern'ing a symbol that doesn't exist.

    A real, load-bearing infrastructure bug was found and fixed while
    landing this (not part of lib/ssl.py itself): linker_c.py's CcTool.link()
    placed ldflags (e.g. `-lssl`) BEFORE the object files being linked on the
    gcc/clang command line. GNU ld only pulls a symbol from a `-l<name>`
    library if there's already a pending undefined reference for it AT THE
    POINT ld reaches that flag - a library listed before the object that
    needs it is silently a no-op, so every @compiler.target(has_library=(lib,
    symbol))-gated def/class in a program that genuinely CALLED that symbol
    would just never resolve, even though has_symbol()'s own probe (compile +
    link, with the same broken ordering) consistently and self-consistently
    reported the symbol as unavailable. This was invisible until now because
    every prior has_library/extern_libs use on Linux was libc ('c'), which
    every compiler driver links implicitly regardless of -l position - lib/
    ssl.py's has_library=('ssl', 'SSL_new') is the first real non-libc shared
    library dependency this mechanism has ever been exercised against on
    Linux. Fixed by moving `extra` (ldflags) after `obj_args` in link()'s
    argument list; verified with linker_c_test.py's full suite plus this
    project's whole test suite (tests.py) on both Windows (MSVC and clang)
    and Linux (gcc, via WSL) - all green, no regressions from the reorder.

    lib/socket.py also gained one small, purely additive method:
    Socket.fileno() -> SOCKET, returning the raw OS handle (POSIX fd / Windows
    SOCKET) - needed because SSL_set_fd() drives its own socket I/O directly
    against the raw fd rather than going through Socket.send()/recv() the way
    the Windows Schannel backend does (Schannel only ever needs byte buffers
    handed through the existing Socket API; OpenSSL's SSL_set_fd() approach
    is the standard, simplest way to use it and is what the validated C
    reference does too).

    Covered by ssl_test.py's SSLLinuxHandshakeTests - the exact same MetalPy
    source (_HANDSHAKE_ROUND_TRIP / _CERTIFICATE_FAILURE_MAPPING) that
    SSLWindowsHandshakeTests runs, just compiled against a different backend
    per host OS. Verified for real via WSL (Debian 13, gcc, real libssl-dev).

Scope for this pass

In scope (now landed for Windows and Linux):
  - lib/ssl.py's SSLError, SSLContext, SSLSocket (Schannel backend on
    Windows, OpenSSL backend on Linux).
  - lib/socket.py's new Socket.fileno() accessor.
  - The linker_c.py ldflags-ordering fix (infrastructure, not ssl.py-specific,
    but found and required by this work).
  - ssl_test.py covering Phase 0 (pure) and both real-network backend classes.

Out of scope (still deferred):
  - macOS (Secure Transport/Network.framework) backend - see "Backend
    strategy" above; no macOS machine available to verify against right now.
    A deliberate poison pill stands in its place instead of either a silent
    gap or a blind implementation - see "macOS poison pill" below.
  - Client-certificate authentication (load_cert_chain) - stretch goal, same
    treatment PLAN_HTTP_CLIENT.md gave multipart files= uploads.
  - Wiring into lib/http/client.py's reserved verify=/HTTPSConnection path -
    a separate follow-up once this is stable.

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
      # via Schannel's own automatic validation; Linux: OpenSSL's default CA
      # bundle/directory search, with peer verification explicitly turned on)

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

This is what makes lib/http/client.py's HTTPSConnection real - wired up as
Phase 4 below: connect a raw socket.Socket, then SSLSocket.wrap_socket() it
before handing the result to the existing request/response read/write path.

Remaining phased roadmap

  Phase 3 — macOS backend, prefixed by the research spike flagged above
    (confirm Secure Transport vs Network.framework's actual current shape
    before committing to bindings). Blocked on access to a macOS machine to
    verify against - do the binding/struct work but hold off calling it done
    without a real handshake test, same discipline Phase 1/2 were held to.
    NOT started - user explicitly decided against implementing this blind
    (no way to verify a real handshake), and asked for a poison pill instead
    (see "macOS poison pill" below) so the gap is loud, not silent.

macOS poison pill (in lieu of Phase 3)

  Rather than an absent SSLContext/SSLSocket on macOS (which was the default,
  do-nothing state before this), lib/ssl.py now defines both as REAL classes
  under @compiler.target(os='macos'), with every method that would need to
  actually do something routing through a single deliberately-undefined name,
  _MACOS_SSL_NOT_YET_IMPLEMENTED__SEE_PLAN_SSL_MD. MetalPy has no
  compiler.error(...)/compiler.static_assert(...) intrinsic for a custom
  compile-time message (confirmed absent from discovery.py/compile_time_
  transformer.py) - referencing an undefined name is the only mechanism that
  exists, so this leans on it deliberately rather than inventing something.

  Why real classes, not just an absent module member: tested directly
  against a simulated macOS compile (Discovery(active_target={'os': 'macos',
  ...}) - not a real Mac, but enough to exercise discovery/type resolution).
  Two things were confirmed, not assumed:

    - With NO macOS definition at all (the prior state), referencing
      ssl.SSLContext from a MetalPy program produces a confusing, target-
      agnostic error ('ssl' is not a value, cannot use it as an expression)
      that gives no hint this is a known, deliberate gap.
    - Worse: lib/http/client.py's own _Transport @union declares
      `Secure: ssl.SSLSocket` as a field type UNCONDITIONALLY (a union needs
      one concrete type per variant, not a per-target one) - with ssl.
      SSLSocket entirely absent, type_resolver.py's RC-class destructor
      synthesis CRASHES outright (AttributeError: 'NoneType' object has no
      attribute 'is_rc_pointer', in _build_field_teardown_ast) the moment
      ANY program merely imports lib/http/client.py on macOS - even one that
      only ever uses plain http://, never touches TLS. So SSLContext/
      SSLSocket need to exist as real, structurally valid types on every
      target lib/http/client.py might compile for, whether or not that
      target's TLS backend is finished - not optional polish.

  A program that merely imports ssl (or http.client) without ever calling
  into TLS compiles clean on macOS - the undefined-name reference only gets
  type-checked once something actually reaches/calls that method (MetalPy
  compiles from main() outward, per ARCHITECTURE.md's stage 2). Confirmed
  directly: `import ssl` alone, and `from http.client import get` alone
  (unused), both produce zero errors under the simulated macOS target.

  One real, broader consequence worth being explicit about, also confirmed
  directly rather than assumed: because lib/http/client.py's _transport_send/
  _transport_recv/_transport_close each pattern-match BOTH _Transport
  variants in one shared function body (`case _Transport.Secure(tls):
  tls.send(...)`), and a function's full body - every match arm, not just the
  ones a given call's runtime value takes - gets compiled as a unit, the
  poison pill fires for ANY http.client usage on macOS, including plain
  http:// with no TLS involved at all. This is NOT a regression this poison
  pill introduces - the crash described above already blocked plain http://
  on macOS before this change, for the identical structural reason (_Transport
  needing ssl.SSLSocket to be a real type regardless of which variant a
  specific call site uses). The poison pill turns that crash into a clear,
  self-explanatory compile error instead - it doesn't narrow the blast radius,
  because narrowing it would need decoupling HTTPConnection's shared
  transport-dispatch helpers per scheme, a real lib/http/client.py redesign
  question, not a "make ssl.py fail loudly" one - out of scope here.

  Verified: full lib/ssl.py + http_client_test.py test suites (Windows -
  MSVC and clang; Linux - gcc via WSL) plus a full tests.py run on both
  platforms all stayed green after adding this - the macOS-only class
  definitions are inert everywhere else.

  Phase 4 — landed: wired into lib/http/client.py as HTTPSConnection (see
    PLAN_HTTP_CLIENT.md's own "Phase 4b — landed" entry for the details -
    _Transport union, HTTPError.TLSError, https:// URL support). Still
    deferred: client-certificate auth (load_cert_chain), session
    resumption/ticket caching, ALPN (relevant for a future HTTP/2 story
    PLAN_HTTP_CLIENT.md already marked out of scope).

Testing approach

  Same idiom as PLAN_HTTP_CLIENT.md: root-level *_test.py files compiled and
  run for real via test_support.RealCompileMixin.assert_programs_run (pattern
  in emitter_c_test.py) — exit-code based, no lib/tests/ directory.

  Phase 0 needed no network dependency, same as any other pure-data test in
  this codebase.

  Phase 1/2's tests (SSLWindowsHandshakeTests / SSLLinuxHandshakeTests) are a
  deliberate departure from every other *_test.py here: they dial out to real
  public hosts (example.com, badssl.com) rather than looping back locally,
  because there's no local TLS server to loop back against without
  implementing server-side Schannel/OpenSSL too (out of scope for a
  client-only library). Both classes run the identical MetalPy source -
  only the compiled backend differs by host OS (os.name=='nt' vs
  sys.platform.startswith('linux')). Gated behind METALPY_TEST_NETWORK=0 for
  environments without network egress. A future macOS backend's tests should
  follow the same pattern - a third class, same shared source, gated on
  sys.platform=='darwin'.

  This phase's Linux work was verified end-to-end via WSL (Debian 13 trixie,
  gcc, real libssl-dev) rather than a native Linux machine - both the
  standalone C reference client and the final ssl_test.py suite were compiled
  and run there for real, including a full tests.py run (1387 tests, 0
  failures) to confirm the linker_c.py fix didn't regress anything else. This
  is a reasonable stand-in for "a real Linux machine" (same kernel/libc/ld
  family, same libssl-dev package), not a shortcut - every claim in this doc
  about what works on Linux is backed by an actual run, not an assumption.
