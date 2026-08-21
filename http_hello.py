'''
POC async HTTP server, for real-world stress testing of MetalPy's
reactor/fiber runtime under concurrent I/O - see PLAN_HTTP_SERVER.md.

GET /                -> dynamic text, regenerated every request (proves
                         nothing is cached) via lib/datetime.py's
                         datetime.now().
GET <anything else>  -> resolved against the www/ subfolder next to this
                         file and served via lib/asyncfile.py's AsyncFile -
                         exercises async file I/O under the reactor, not
                         just sockets.
'''

import compiler
import io
import reactor
import tcp
import asyncfile
from datetime import datetime
from fs import SEEK_SET, SEEK_END
from http.server import Request, Response, serve


def _is_safe_relative_path( path: str ) -> bool:
	''' rejects any '..' path segment outright - simplest correct check for
	a POC, not full canonicalization. '''
	segments: list[str] = path.split( '/' )
	n: usize = segments.__len__()
	i: usize = 0
	for i in range( n ):
		segment: str = segments.__getitem__( i ).unwrap( '_is_safe_relative_path: index in bounds by construction' )
		if segment == '..':
			return False
	return True

def _read_static_file( file_path: str ) -> Result[bytes, OSError]:
	reader: BinaryReader = asyncfile.AsyncFile.binary_reader( file_path ).or_return()
	size: i64 = reader.seek( i64( 0 ), SEEK_END ).or_return()
	reader.seek( i64( 0 ), SEEK_SET ).or_return()
	with compiler.panic_arithmetic( 'a real static file fits well within usize' ):
		n: usize = usize( size )
	buf: bytearray = bytearray( n )
	io.read_exact( reader, buf.get_ptr(), n ).or_return()
	return Result.Ok( bytes.from_bytearray( move( buf )))

def _serve_static( path: str ) -> Response:
	if not _is_safe_relative_path( path ):
		return Response.text( 'Forbidden\n', 403, 'Forbidden' )
	file_path: str = 'www' + path
	match _read_static_file( file_path ):
		case Result.Ok( content ):
			return Response.bytes_( content, 'text/html; charset=utf-8', 200, 'OK' )
		case Result.Err( _ ):
			return Response.text( 'Not Found\n', 404, 'Not Found' )

class App:
	def handle( self, req: Request ) -> Response:
		if req.path == '/':
			now: datetime = datetime.now()
			return Response.text( f'Hello from MetalPy {now}\n' )
		return _serve_static( req.path )

def main() -> i32:
	listener: tcp.TcpListener = tcp.TcpListener.bind( '0.0.0.0', u16( 8080 )).unwrap( 'bind 0.0.0.0:8080' )
	# 1 worker, not 2+ - a confirmed, pre-existing reactor.py bug (task
	# task_0904047c): a task spawned by a running fiber onto ANOTHER worker
	# that started with an empty queue never runs, so multi-worker isn't
	# safe here yet - see http_server_test.py's own note on the same issue.
	r: reactor.Reactor = reactor.Reactor( 1 )
	app: App = App()
	serve( listener, app.handle, r )
	print( 'listening on http://0.0.0.0:8080' )
	r.run()
	return 0
