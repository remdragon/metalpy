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

  URL parsing / form encoding / base64: written when none of this existed yet —
    since landed as real, general-purpose modules, and http.client migrated onto
    both: lib/base64.py (see the base64_encode note elsewhere in this doc) and
    lib/urllib/parse.py (quote/unquote, urlencode/parse_qsl, urlsplit/urlunsplit,
    urljoin — commit 23baa97). http.client's own hand-rolled ParsedURL/
    _parse_url/_merge_query_params/_form_encode were replaced with thin wrappers
    around urlsplit()/parse_qsl()/urlencode() — see "urllib.parse migration"
    further down for what that changed, including a real feature gain (relative
    redirect Location headers, via urljoin(), previously unsupported).

  Error codes: written when none of this existed yet - since landed. lib/builtins/
    __errors.py's OSError (both @compiler.target variants) now has ConnectionRefused,
    ConnectionReset, TimedOut, AddressInUse, WouldBlock, and NameResolutionFailed,
    added alongside lib/socket.py itself. HTTPConnection.connect() distinguishes
    HTTPError.NameResolutionFailed from every other connect() failure (which still
    collapses to HTTPError.Other()) - see "Socket-facing OSError" note further down.

Socket surface — what actually landed (lib/socket.py, commit 863bfc8)

The handoff contract below is superseded by this section - kept for history,
not as the current source of truth. lib/socket.py landed with a slightly
different shape than requested, close enough to build on directly:

  Socket.tcp( family: i32 = AF_INET ) -> Result[Socket, OSError]
    Two-step construction (create, then connect), not a single
    Socket.connect(host,port) factory.
  socket.connect( host: str, port: u16 ) -> Result[None, OSError]
  socket.send( buf: ConstPtr[u8], count: usize ) -> Result[usize, OSError]
  socket.recv( buf: Ptr[u8], count: usize ) -> Result[usize, OSError]
  socket.close(), __del__ auto-close — same idiom as BinaryReader/BinaryWriter.

Two real gaps versus what was asked for, both accepted as-is rather than
reworked, for the reasons below:

  - No SocketError - errors are plain OSError, same as lib/fs.py. STALE as of
    lib/socket.py's own later growth: OSError gained ConnectionRefused,
    ConnectionReset, TimedOut, AddressInUse, WouldBlock, and
    NameResolutionFailed variants (added alongside lib/socket.py itself, not
    part of its original landing). http.client itself still collapses every
    connect() failure to HTTPError.Other() EXCEPT NameResolutionFailed, which
    HTTPConnection.connect() distinguishes explicitly (`os_err ==
    OSError.NameResolutionFailed` - confirmed this comparison against a
    caught, non-bare-reference OSError value works via a real compile,
    untested territory before this) - the one case a caller is likely to
    want to handle differently (retry vs. give up on a typo'd hostname).
    Further granularity (refused vs. reset vs. timed out) remains
    unexposed - not needed by anything built here yet, easy to add the same
    way if a real caller needs it.
  - No timeout_ms parameter at all (blocking-only, no timeout support
    anywhere in lib/socket.py yet). http.client's own timeout_ms= parameter
    (see the Session.request() sketch below) stays reserved/no-op until
    lib/socket.py itself grows timeout support - not blocking on it now.
  - No DNS/getaddrinfo - lib/socket.py's own header comment states this
    outright: "host a pre-resolved IPv4/IPv6 literal ... a self-contained
    follow-up." Real hostnames (not IP literals) don't work yet. Flagged as
    its own follow-up task (see task_a8b4e7c3 / "Add DNS/getaddrinfo
    resolution to lib/socket.py"). Not a blocker for building/testing
    HTTPConnection today: loopback testing against 127.0.0.1 (a literal)
    works fine without it - only real-hostname support is blocked.

Header representation — skip email.message for v1

`requests` itself doesn't use email.message for its public API either (it's a
case-insensitive dict-like object). Define a small purpose-built HTTPHeaders type
in lib/http/client.py — an ordered, case-insensitive string multimap, matching
`requests.structures.CaseInsensitiveDict` ergonomics (get/set/iterate). It covers
everything an HTTP client needs without pulling in MIME semantics. If a real
email.message ever lands for the smtp/email-parsing stdlib goals, http.client can
be revisited to reuse it, but shouldn't block on that landing first.

HTTPConnection/Response (Phase 3a — this is the layer actually being implemented
now that Phase 1/2 are unblocked; not in the original sketch below, which jumped
straight to Session):

  class HTTPConnection:
      __sock: Socket
      __host: str
      __port: u16

      def __del__( self ) -> None: ...
      @staticmethod
      def connect( host: str, port: u16 = 80 ) -> Result[HTTPConnection, OSError]: ...
      def request( self, method: str, path: str, headers: HTTPHeaders|None = None,
          body: bytes|None = None ) -> Result[None, OSError]: ...
      def getresponse( self ) -> Result[Response, HTTPError]: ...
      def close( self ) -> None: ...

  class Response:
      status_code: u16
      reason: str
      headers: HTTPHeaders
      content: bytes
      def text( self ) -> Result[str, CodecError]: ...
      def ok( self ) -> bool: ...

Body reading (inside getresponse()) picks Content-Length, chunked (via Phase 0's
decode_chunked), or read-until-close, matching HTTP/1.1 semantics for how a
response body's own length is determined. host is an IP literal only for now
(see "Socket surface" above) - real hostnames wait on task_a8b4e7c3.

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

  Phase 0 — landed: HTTPError, HTTPHeaders, status-line/header-line parsing,
    percent-encoding, base64 encoding (now a thin wrapper around lib/base64.py,
    which landed after this file's own hand-rolled version - see that module),
    chunked transfer-encoding decode. Covered by http_client_test.py's
    HTTPClientPhase0Tests.

  Phase 1 — landed: lib/socket.py (commit 863bfc8). IP-literal-only (no DNS yet -
    see "Socket surface" above and task_a8b4e7c3), sufficient for loopback testing.

  Phase 2 — landed: _GrowableBuffer, a private doubling byte buffer local to
    lib/http/client.py that accumulates recv() output across multiple calls
    (find_double_crlf/slice_bytes/slice_str). Folded into Phase 3a below rather
    than landing separately - the two were implemented and tested together.

  Phase 3a — landed: HTTPConnection (connect/request/getresponse/close) and
    Response (status_code, reason, headers, content, text(), ok()). Wires
    Phase 0 + 1 + 2 together; body reading picks Content-Length, chunked, or
    read-until-close per RFC 7230. Every public method returns a bare
    Result[_, HTTPError] (see "A real compiler gap found while landing Phase
    3a" below - OSError from lib/socket.py collapses into HTTPError.Other()
    rather than being part of the public error type). Does NOT include
    Session's own ergonomics (params=/data=/json=/cookies=/auth=/redirects,
    module-level get/post/...) - see Phase 3b. Covered by http_client_test.py's
    HTTPConnectionLoopbackTests: a real loopback TCP round trip (background
    thread plays a minimal server via lib/socket.py directly) for both a
    Content-Length body and a chunked body, plus a connection-refused error
    path. host is still an IP literal only (see Phase 1).

  Phase 3b — landed: Session (cookie jar, redirect-following loop with a
    10-redirect cap, params=/form=/auth= encoding) and module-level get()/
    post()/put()/patch()/delete()/head()/options()/request(), each a one-off
    Session() underneath. One API deviation from the original sketch below:
    `data=` is bytes|str|None only - a separate `form=` dict[str,str]
    parameter handles application/x-www-form-urlencoded bodies, instead of
    one requests-style bytes|str|dict|None union (a real compiler gap made
    `form=` a required workaround at the time - see "Four real compiler
    gaps" below; `auth=` hit an analogous gap and was temporarily a
    BasicAuth(user, password) class instead of a bare tuple, since reverted
    back to `tuple[str,str]|None` once that gap was fixed - `form=` was kept
    as its own parameter rather than reverted, since match-based dispatch
    across a real 3-member bytes|str|dict union remains genuinely untested
    territory even now, per _encode_body's own comment).
    `json=`/`.json()` remain deferred (no json library yet, unchanged from
    the original plan). Relative Location headers on a redirect ARE now
    resolved (via lib/urllib/parse.py's urljoin() - see "urllib.parse
    migration" below; this was originally a gap, fixed once urljoin()
    landed). Covered by http_client_test.py's SessionLoopbackTests:
    cookie-jar harvest+replay across two real requests, a real 302 redirect
    followed transparently (with params= merged + percent-encoded into the
    pre-redirect request), a relative-Location redirect resolved correctly,
    and a form POST with Basic auth - all verified by having the fake
    server inspect the raw bytes it actually received, not just checking
    the client-side response.

  urllib.parse migration (lib/urllib/parse.py, commit 23baa97) — http.client's
    own hand-rolled ParsedURL/_parse_url/_merge_query_params/_form_encode were
    replaced with thin wrappers around urlsplit()/parse_qsl()/urlencode()/
    urljoin(). Two real, positive behavior changes came with it, not just a
    refactor:
      - Query-string/form encoding now goes through urlencode() (quote_plus:
        space -> '+'), matching requests' own params=/data= dict encoding
        exactly - the old hand-rolled version used quote()-style %20, a
        subtle mismatch with what it was supposed to mirror.
      - Redirect Location headers may now be relative, resolved against the
        request URL via urljoin() (RFC 3986 5.3) - previously only absolute
        http:// Location values were followed; a relative one silently
        wasn't treated as a redirect at all. Covered by a new loopback test,
        session_follows_relative_redirect.

Four real compiler gaps found while landing this plan - all now fixed

Each was flagged as its own follow-up task while lib/http/client.py worked
around it; all four have since landed on master, and every workaround below
has been reverted back to the originally-intended shape. Kept here as a
historical record (worth knowing if similar tuple/union code elsewhere in
this codebase was hitting the same walls before these fixes landed).

1. No DNS/getaddrinfo in lib/socket.py (task_a8b4e7c3) - host had to be a
   pre-resolved IP literal. Fixed by commit f794edb ("socket: add
   getaddrinfo-based hostname resolution") - Socket.connect() now resolves
   real hostnames transparently. No lib/http/client.py changes were needed
   either way - HTTPConnection.connect()/Session already just called
   sock.connect(host, port) and got hostname support for free once the fix
   landed underneath them.

2. Widening a bare @union error type (HTTPError) into a WIDER declared union
   return type (e.g. OSError|HTTPError) was broken three related ways -
   .or_return(), a direct return Result.Err(e), and a staged explicitly-
   typed local all failed (task_ef51cec6). Worked around by never declaring
   a union return type at all - every public HTTPConnection method returns
   a bare Result[_, HTTPError], with small `_*_or_http_err` helpers
   collapsing any OSError from lib/socket.py into HTTPError.Other() at the
   call site. Fixed by commit a4ca6a3 ("Fix union widening: nominal @union
   error types can now widen into a bigger union"). NOT reverted -
   collapsing every Socket-facing OSError into HTTPError.Other() is still
   arguably better API design on its own merits (HTTPConnection's own
   public error type stays a single, simple HTTPError instead of leaking
   lib/socket.py's OSError), so the design was kept deliberately once fixed,
   not just left as a stale workaround - see lib/http/client.py's own
   comment above `_connect_or_http_err`.

3. tuple[T|None, ...] (a union as a tuple's own ELEMENT type) generated C
   that didn't compile (task_34251c9f) - `_encode_body()` originally
   returned tuple[bytes|None, str|None], and the generated C called an
   allocator function that was never declared anywhere in the translation
   unit, plus assigned raw pointers directly into fields that should have
   been tagged-union structs. Worked around with a small dedicated
   _EncodedBody class (two fields) instead of a tuple return type. Fixed by
   commit 98c2010 ("tuple: coerce elements into their declared union types
   during construction") - reverted back to `tuple[bytes|None, str|None]`,
   _EncodedBody removed.

4. tuple[...] as a MEMBER of an outer union (the inverse of #3) crashed the
   emitter outright - not a bad-compile-error, an uncaught Python
   AssertionError inside emit_c() itself, the moment a real tuple[str,str]
   value flowed through a tuple[str,str]|None-typed parameter (auth=
   ('user','pass') in this case) (task_827c2650). Worked around by giving
   auth= a dedicated BasicAuth(user, password) class instead of requests'
   own bare-tuple ergonomics. Fixed by commit 0c3ab27 ("Fix tuple-in-union
   Allocate crash: don't trust expected_type blindly for dest's type") -
   reverted back to `auth: tuple[str,str]|None`, BasicAuth removed. auth=
   now matches requests' own `auth=(user, password)` ergonomics exactly.

  Phase 4a — landed: `json=` (on Session/post/put/patch and their module-level
    counterparts) and `Response.json()`, built on lib/json.py (commit cc116c9)
    - `json=` serializes via json.dumps() and sets Content-Type: application/
    json if not already present; `.json()` parses `.content` via json.loads(),
    collapsing either a UTF-8 decode failure or a JSON parse failure into
    HTTPError.InvalidJSON. `json=`/`data=`/`form=` remain mutually exclusive,
    checked in that priority order (json= wins if more than one is somehow
    given). Covered by http_client_test.py's session_json_request_and_response:
    a real loopback POST with json=, server verifies the raw Content-Type and
    JSON body bytes, response comes back as its own JSON body, parsed back via
    .json() and read through object_get()/as_str()/as_int(). Two real compiler
    gaps found while landing this - see "Compiler gaps found while landing
    Phase 4a" below.

  Phase 4b (deferred/future plan doc) — HTTPSConnection/TLS (no TLS library
    exists at all yet - a large separate undertaking), multipart `files=`,
    connection reuse/keep-alive.

Compiler gaps found while landing Phase 4a

1. A tuple literal passed DIRECTLY as Result.Ok(...)'s own argument, where
   the enclosing function's declared return type wraps a tuple with union
   element types (Result[tuple[bytes|None,str|None], HTTPError] here), left
   T ambiguous - "inferred as both tuple[bytes|None,str|None] and
   tuple[bytes,str]" (a real compile error). This is a narrower case than
   task_34251c9f (a union AS a tuple's own element - fixed by 98c2010): here
   the tuple/union shape itself is fine on its own (it's exactly what
   task_34251c9f fixed), the NEW gap is specifically Result.Ok(...) inferring
   its own T from a bare tuple-literal argument rather than the function's
   declared return type. Worked around by staging every such tuple literal
   through an explicitly `tuple[bytes|None,str|None]`-typed local first, then
   passing THAT to Result.Ok() - see _encode_body's own comment. Flagged as
   task_ffb0bdb5.

2. A nested `match` (every arm returning) directly inside an `if x is not
   None:` block, immediately followed by a plain `if` checking a DIFFERENT
   parameter, produced a nonsensical diagnostic on the unrelated parameter -
   "'form' is not initialized on all code branches", where `form` is an
   ordinary always-bound parameter never touched by the preceding block.
   Worked around by extracting the nested-match branch into its own small
   single-return-statement helper function (_encode_json_body) - matches
   this file's own established "extract into a plain helper" pattern for
   narrowing-related compiler gaps (_build_request_headers/_next_redirect_url).
   Flagged as task_ccb9f3d6.

3. (Not a compiler bug - a real bug in this file, found and fixed the same
   way): a match-arm capture bound to the name `text` inside Response.json()
   collided with Response's own text() method, "'text' is not a variable,
   cannot assign to it". Fixed by renaming the capture to `decoded`.

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
