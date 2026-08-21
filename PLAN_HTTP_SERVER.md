HTTP/1.1 server (lib/http/server.py) — scope

Context

The user wants a POC async web server written in MetalPy, purely to get a real
target for stress-testing the language/runtime (throughput, RC overhead under
concurrency, reactor scheduling behavior) - not a production server. Before this,
lib/http/client.py existed (client-only: connect/request/getresponse) and the
networking/reactor layer (lib/socket.py, lib/tcp.py, lib/reactor.py) was already
server-ready (TcpListener.accept(), reactor-aware retry loops, tcp_test.py's own
accept-loop-plus-Reactor.spawn() pattern) - the actual gap was entirely
HTTP-protocol-level: there was no server anywhere in the repo.

Scope (v1, this pass)

In:
  - Content-Length request/response bodies.
  - HTTP/1.1 keep-alive: a connection stays open across multiple requests unless
    either side sends "Connection: close", or the request isn't HTTP/1.1.
  - A single handler-closure API (`serve(listener, handler, reactor)`) rather than
    a routing table - matches this codebase's bias against premature abstraction;
    a caller wanting routing builds its own dispatch inside one handler.
  - Reuse of lib/http/client.py's already-public wire-format helpers (HTTPHeaders,
    parse_headers) rather than duplicating header-parsing logic.

Out (deliberately, not gaps to feel bad about):
  - Chunked request/response transfer encoding (decode_chunked exists client-side;
    no chunked *encode* anywhere yet either).
  - gzip/deflate content encoding.
  - Caching (ETag/If-Modified-Since/Cache-Control).
  - HTTP/2, HTTP/3.
  - Server-side TLS (lib/ssl.py's own PLAN_SSL.md is client-oriented; HTTPS
    serving would need a real server-side handshake path that doesn't exist).
  - A routing/middleware framework - see the handler-closure decision above.

Design notes

  Built directly on tcp.TcpListener/tcp.TcpConnection + reactor, NOT on
  socket.Socket the way http.client's HTTPConnection is - the whole point here is
  genuine reactor-driven concurrency, and TcpConnection already retries through
  reactor.wait_for_signal() on WouldBlock, so the exact same handler code runs
  cooperatively under a Reactor or as an ordinary blocking accept loop with none.

  http.client's own internal buffering helper (_GrowableBuffer) is typed against
  Socket.recv() directly, not the io.Reader protocol, so it can't be reused as-is.
  lib/http/server.py's _RequestBuffer is the server-side equivalent: same
  fill_from/find_double_crlf/slice_bytes/slice_str shape, generic over io.Reader
  (calls .read()) instead. It adds one capability the client-side buffer never
  needed: compact(consumed), which discards a fully-read request's bytes and
  shifts any trailing bytes (a pipelined next request already sitting in the same
  buffer) down to offset 0 - required for the keep-alive loop to reuse one buffer
  across multiple requests on the same connection. Not worth factoring the shared
  parts into a third module for a POC - the two buffers differ in which transport
  method they call, and ~80 lines of duplication matches this codebase's own
  stated preference over a premature shared abstraction.

  serve(listener, handler, r) does not block - it spawns the accept loop onto r
  and returns; the caller still calls r.run() itself, mirroring tcp_test.py's own
  spawn-then-run() shape rather than serve() taking over the whole process.

  Handler calling convention: `handler: Closure[[Request], Response]` must be a
  real closure - a bound method (e.g. `app.handle`) or a capturing lambda - NOT a
  bare top-level function reference. Passing a bare function (or explicitly
  converting one via `Closure[[Request],Response](my_fn)`) crashes the compiler
  itself (AssertionError in lowering.py's _try_lower_construct_call: "target_cls
  ... is not fully resolved") when the closure's argument/return types are
  user-defined classes rather than scalars - confirmed via a real compile attempt
  while building this module. This is a pre-existing compiler gap (every other
  Closure[[...],...]-typed call site in this codebase - reactor.py's own
  Worker.schedule/Reactor.spawn callers - already only ever passes bound methods
  or capturing lambdas, never a bare function), not something introduced here;
  flagged for a future fix rather than blocking this POC, since the bound-method/
  lambda calling convention is already this codebase's norm.

  Errors: a distinct `http.server.HTTPError` @union (not shared with
  http.client's own HTTPError, a different nominal type despite the same name) -
  MalformedRequestLine, MalformedHeader, UnexpectedEOF (peer closed mid-request:
  a real error), ConnectionClosed (peer closed cleanly between requests: expected,
  not an error - lets _handle_connection tell the two apart via ordinary
  Result.Err(HTTPError.X(_)) matching, the same "match a specific Err variant"
  shape http.client's own _read_chunked_body already uses for ChunkSizeInvalid).
  Unlike http.client's getresponse() (which .unwrap()s a UTF-8 decode of a
  response it already trusts), a request head comes straight from the network -
  a malformed one returns a proper HTTPError instead of panicking the whole
  worker OS thread.
