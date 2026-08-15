HTTP client library (lib/http/client.py) — scope and roadmap

Context

The user wants an HTTP client eventually available in MetalPy's stdlib, but flagged
up front that this has real prerequisites: a socket library (currently being scoped
in a separate, parallel planning session — not merged, not started in code) and an
email.message-style interface for header encoding (not started at all). A repo-wide
search confirms README.md ("Features that are planned but not built yet": socket
library, http client/server classes, smtp library, email parsing, json library) and
TODO.txt (bare "socket class" stub entry) treat this as fully greenfield — zero
existing socket, email, http, or json code anywhere in lib/ today.

The user explicitly wants the *interface* to mirror the 3rd-party Python `requests`
library (Session/Response/get/post/request ergonomics: status_code, headers, text,
json(), params=/data=/json=/headers=/cookies=/timeout=/allow_redirects=), not
MetalPy's own low-level BinaryReader/BinaryWriter file-handle idiom. They also
pointed at their own PHP `fetch()` helper (a requests-like wrapper built on curl)
as a reference for a few ergonomic details worth carrying over: a Session-level
cookie jar that auto-harvests Set-Cookie headers and replays them, and content-
type-driven request body encoding.

So this session's job is not to write working code (it can't compile yet — nothing
to connect a socket to), but to produce a roadmap that sequences the real
prerequisite work, pins down a requests-shaped API surface, and hands the socket-
planning session a concrete contract for what http.client needs from it.

Scope for this pass

In scope:
  - This roadmap document, following the existing PLAN_*.md convention (plain-text
    section titles, no markdown headers).
  - Non-compiling draft class/method signatures for lib/http/client.py, shaped
    after `requests` (Session, Response, module-level get/post/request/...),
    adapted to Result[T,E]/no-exceptions instead of requests' exception-raising
    style.
  - The minimal socket surface http.client needs, as a handoff contract for the
    parallel socket-planning session.
  - Identifying which pieces can be built and tested TODAY with zero prerequisites
    (pure wire-format parsing, URL/form encoding, base64), vs. genuinely blocked
    pieces.

Out of scope (this pass):
  - Actually implementing lib/http/client.py, lib/socket*, or lib/email* — none of
    the prerequisites exist yet.
  - HTTPS/TLS support — needs a TLS library, itself a large separate undertaking
    (schannel on Windows / some TLS lib on POSIX). `verify=`/https:// URLs are
    reserved in the API shape now but no-op/error until TLS lands; deferred to its
    own future plan doc.
  - A full email.message clone (MIME, RFC 2822 header folding, multipart). See
    "Header representation" below.
  - multipart/form-data file uploads (`files=`) — even the referenced PHP fetch()
    just delegates this to curl natively rather than hand-rolling multipart
    encoding; MetalPy would have to hand-roll it. Deferred as a stretch goal.
  - Connection pooling/keep-alive reuse across requests, HTTP/2, proxies.

Current state of prerequisites

  socket: not started in code anywhere in the repo. Being scoped in a separate
    parallel session right now. This plan defines the exact surface http.client
    needs (see below) so that work can target it.

  buffered I/O / streams: TODO.txt (line ~707) confirms even file I/O has "no real
    stream object (buffering, flush, read)" yet — only raw single-shot read_raw/
    write_raw (lib/fs.py) and write_all's retry-loop exist as precedent. http.client
    will need a small buffered reader (peek/read-until-length, read-until-chunk-
    boundary) to parse responses off a raw socket. Recommend this be scoped as a
    private helper local to lib/http/client.py, NOT a general lib/io module.

  email.message: does not exist. See "Header representation" below — v1 doesn't
    need it.

  json: also listed in README as planned-but-not-built, and nothing in lib/ does
    JSON encode/decode. This is a prerequisite the user didn't mention but that
    `requests` parity pulls in directly (`Response.json()`, `request(json=...)`).
    Recommend treating `.json()`/`json=` as deferred until a json library exists,
    while `.content`/`.text`/`data=` (raw bytes/str) work from day one.

  URL parsing / form encoding / base64: none exist (no urllib, no percent-encoding
    helper, no base64 anywhere in lib/). All three are pure string/byte transforms
    with zero I/O dependency — str already has split/find/index/strip
    (lib/builtins/__init__.py) to build a minimal scheme://host:port/path?query
    splitter and a percent-encoder on top of. These are cheap, buildable today, and
    needed for `params=`, form-encoded `data=`, and `auth=` (HTTP Basic → base64).

  Error codes: lib/posix/errors.py's PosixError already defines
    ConnectionRefused = 111 (ECONNREFUSED) — staged in advance for this. lib/builtins/
    __errors.py's OSError (both @compiler.target variants) has no network-specific
    variants yet. The socket library will need to add ConnectionReset, TimedOut,
    HostUnreachable, and NameResolutionFailed equivalents on both platforms.

Minimal socket surface required (handoff contract for the socket-planning session)

http.client only needs blocking, synchronous TCP stream sockets — no UDP, no
async/select. Concretely:

  Socket.connect(host: str, port: u16, timeout_ms: u32|None = None)
      -> Result[Socket, SocketError]
    Must resolve hostnames (DNS), not just accept literal IPs.

  socket.send(buf: ConstPtr[u8], count: usize) -> Result[usize, SocketError]
  socket.recv(buf: Ptr[u8], count: usize) -> Result[usize, SocketError]
    Same shape as fs.py's write_raw/read_raw. recv returning 0 means peer closed.

  socket.close(), and __del__ auto-closing — same idiom as BinaryReader/BinaryWriter
    in lib/builtins/__File.py.

  A SocketError @enum (Windows/POSIX @compiler.target pair, same pattern as OSError)
    with at least: ConnectionRefused, ConnectionReset, TimedOut, HostUnreachable,
    NameResolutionFailed, Other.

Anything beyond this (SO_REUSEADDR, non-blocking mode, UDP, raw sockets) is not
needed by http.client v1.

Header representation — skip email.message for v1

`requests` itself doesn't use email.message for its public API either (it's a
case-insensitive dict-like object). Define a small purpose-built HTTPHeaders type
in lib/http/client.py — an ordered, case-insensitive string multimap, matching
`requests.structures.CaseInsensitiveDict` ergonomics (get/set/iterate). It covers
everything an HTTP client needs without pulling in MIME semantics. If a real
email.message ever lands for the smtp/email-parsing stdlib goals, http.client can
be revisited to reuse it, but shouldn't block on that landing first.

Draft API sketch (lib/http/client.py — NOT compilable yet, pins the surface only)

Shaped after `requests`' get/post/Session/Response ergonomics. Every call that
`requests` would raise an exception for instead returns Result[T, HTTPError] —
including what `requests` does in `Response.raise_for_status()`.

  @enum( i32 )
  class HTTPError:
      MalformedStatusLine = 1
      MalformedHeader      = 2
      UnexpectedEOF         = 3
      ChunkSizeInvalid      = 4
      TooManyRedirects      = 5
      BadStatus             = 6   # for a requests-style raise_for_status() check
      Other = _

  class HTTPHeaders:                       # ordered, case-insensitive multimap
      def get( self, name: str ) -> str|None: ...
      def get_all( self, name: str ) -> list[str]: ...
      def set( self, name: str, value: str ) -> None: ...
      def add( self, name: str, value: str ) -> None: ...
      # iterate yields (name, value) in wire/insertion order

  class Response:                          # mirrors requests.Response
      status_code: u16
      reason: str
      url: str                             # final URL, after any redirects
      headers: HTTPHeaders
      content: bytes
      # text: str                          # decoded per Content-Type charset (utf-8 default)
      #   -- exposed as a method, not a stored field, since decoding is fallible:
      def text( self ) -> Result[str, HTTPError]: ...
      def json( self ) -> Result[JSONValue, HTTPError]: ...   # deferred until json lib lands
      def ok( self ) -> bool: ...                              # status_code < 400
      def raise_for_status( self ) -> Result[None, HTTPError]: ...  # Err(BadStatus) on 4xx/5xx

  class Session:                           # mirrors requests.Session
      headers: HTTPHeaders                 # default headers merged into every request
      __cookies: dict[str, str]             # simple name->value jar, harvested from
                                            # Set-Cookie on every response and replayed
                                            # on subsequent requests (same approach as
                                            # the referenced PHP fetch() Session)

      def request( self, method: str, url: str,
          params: dict[str,str]|None = None,
          data: bytes|str|dict[str,str]|None = None,
          json: JSONValue|None = None,           # deferred until json lib lands
          headers: HTTPHeaders|None = None,
          cookies: dict[str,str]|None = None,
          auth: tuple[str,str]|None = None,       # HTTP Basic -> base64 Authorization header
          timeout_ms: u32|None = None,
          allow_redirects: bool = True,
          verify: bool = True,                    # reserved; no-op until TLS lands
      ) -> Result[Response, HTTPError]: ...

      def get( self, url: str, **kwargs ) -> Result[Response, HTTPError]: ...
      def post( self, url: str, **kwargs ) -> Result[Response, HTTPError]: ...
      def put( self, url: str, **kwargs ) -> Result[Response, HTTPError]: ...
      def patch( self, url: str, **kwargs ) -> Result[Response, HTTPError]: ...
      def delete( self, url: str, **kwargs ) -> Result[Response, HTTPError]: ...
      def head( self, url: str, **kwargs ) -> Result[Response, HTTPError]: ...
      def options( self, url: str, **kwargs ) -> Result[Response, HTTPError]: ...

  # module-level convenience, mirrors requests.get()/requests.post()/etc. — each
  # is a one-off Session() underneath, same as the referenced PHP fetch() function
  # wrapping its own one-off Session
  def get( url: str, **kwargs ) -> Result[Response, HTTPError]: ...
  def post( url: str, **kwargs ) -> Result[Response, HTTPError]: ...
  def put( url: str, **kwargs ) -> Result[Response, HTTPError]: ...
  def patch( url: str, **kwargs ) -> Result[Response, HTTPError]: ...
  def delete( url: str, **kwargs ) -> Result[Response, HTTPError]: ...
  def head( url: str, **kwargs ) -> Result[Response, HTTPError]: ...
  def options( url: str, **kwargs ) -> Result[Response, HTTPError]: ...
  def request( method: str, url: str, **kwargs ) -> Result[Response, HTTPError]: ...

Whether MetalPy's compiler actually supports `**kwargs`-style forwarding is worth
confirming during implementation — if not, the module-level functions and
Session methods will need to spell out the full parameter list instead of
forwarding kwargs (same shape either way, more verbose call sites).

Body encoding follows `data`'s runtime type, same dispatch idea as the referenced
PHP fetch(): str/bytes sent as-is; dict[str,str] form-encoded as
application/x-www-form-urlencoded (needs the percent-encoder from "Current state
of prerequisites" above). `json=` follows once a json encoder exists. Redirects
(3xx + Location header) are followed by request() itself when allow_redirects is
set — unlike CPython's stdlib http.client (which leaves that to urllib), matching
`requests`' and the reference PHP fetch()'s behavior of handling it internally.

Implementation plan (phased, for once this moves from scoping to real work)

  Phase 0 — pure functions, zero prerequisites, buildable NOW: HTTP status-line
    parsing, header-line parsing into HTTPHeaders, chunked transfer-encoding
    decode, request-line/header serialization, minimal URL splitting +
    percent-encoding (for params=/form data=), base64 (for auth=).

  Phase 1 — blocked on socket library landing (parallel session, contract above).

  Phase 2 — small private buffered-read helper over a raw socket (peek/read-exact/
    read-until-chunk-boundary), local to lib/http/client.py.

  Phase 3 — wire Phase 0 + 1 + 2 into Session.request()/Response, including the
    cookie jar and redirect-following loop.

  Phase 4 (deferred/future plan doc) — HTTPSConnection/TLS, `json=`/`.json()` once
    a json library exists, multipart `files=`, connection reuse.

Testing approach

  No lib/tests/ directory exists in this project — stdlib coverage lives in
  root-level *_test.py files compiled and run for real via
  test_support.RealCompileMixin.assert_programs_run (see emitter_c_test.py for the
  pattern: setUp() builds a Discovery+Compiler, then one test method feeds several
  (case_name, mpy_source) tuples that each define main() -> i32, 0 = pass).

  Phase 0 (wire-format/URL/base64 parsing) can get a new http_client_test.py TODAY:
  feed in-memory byte strings representing status lines, headers, chunked/
  Content-Length bodies, and known encode/decode vectors; assert via exit code —
  no socket dependency.

  Phase 3 (full Session.request()/Response round-trip, redirects, cookie jar) needs
  a real loopback test once socket lands: spin up a minimal raw-socket listener in
  the test and have Session talk to it over real localhost TCP, asserting via exit
  code.

  A manual smoke test (compile a small program that GETs a local fixture and prints
  status_code/text) is worth running by hand once Phase 3 lands, per this project's
  existing practice of verifying compiled programs via print()/exit codes.
