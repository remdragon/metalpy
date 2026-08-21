================================================================================

I want to explore opening up global init to allow all top-level instructions.

We're already so close with creating global objects, I think it would be trivial
just to move all top-level code into __metalpy_init() and enable all top-level
code.

I do *NOT* want to introduce the `if __name__ == '__main__'` concept, top-level
execution is not the same as python, so this mechanism doesn't make sense here

The reason this could be useful is I have another exploration task to allow
user-created decorators and combined with this feature would allow us to create
path registration decorators for web servers, for example.

================================================================================

in the stdlib, I see usage like `s: str; c = s.__getitem__( n )`. Investigate
why this is happening and if this code can be cleaned up to a simple `c = s[n]`

Also, fmt.__len__() instead of the more readable len(fmt)

================================================================================

Explore the possibility of enabling defer/errdefer in a loop by scheduling it
for the end of the loop body instead of the end of the function.

================================================================================

Looking at the difference between http_server_demo_sync.py and
http_server_demo_async.py, do we need a synchronous TcpServer that implements
BaseServer and _ConnectionHandler (with better names) so that the sync and async
versions of the http server are more similar?

I think python has something like this already too.

Additionally, the sync version appears to be serving on only one thread.
Should we have an option to service requests with multiple threads? Do we need
a separate reusable ThreadPool class for this?

================================================================================

studying the http server demo, I see it appears to be hard-coding the www path
relative to cwd. We need a way for the program to know the path to it's own
executable so it can calculate paths relative to that (I think this might
not be possible on all OS's. If that's the case, we'll need some kind of
Result indicating it's not available)

================================================================================

more http server analysis, we are reading entire files into a bytearray,
converting to a bytes object, then returning the entire thing to the server.
We need an ergonomic way to stream a response so that we can mmap a source file
and stream it instead of allocating it in memory twice

================================================================================

I want to explore the possibility of user-created decorators. I've done some
research and I believe this may be possible with the infrastructure we've built

================================================================================

demo http server has Response.bytes_(...). This is ugly. Other http apis have
more rich method names, but in this case maybe we implement Response.text() and
have it take a str|bytes type?

================================================================================

explore what's the different between Callable[] and Closure[]. Is there a
reason to have both?

================================================================================

create an open() method for people that want to be lazy and not use the more
type-efficient methods. This will require returning a
BinaryReadWriter|StrReadWriter since we can't know at compile time which one
they're choosing. We could however, implement a special case in type_resolver.py
to catch literal strings passed to open() and swap it out with a more type-
specific File opener method.

================================================================================

we need min() and max(). each needs to be an overload, for example:

def min[T]( a: T, b: T ) -> T:
	return a if a < b else b

def min[T]( seq: Sequence[T] ) -> T:
	it = iter( seq )
	value: T = next( it )
	for t in it:
		value = min( value, t )
	return value

I've already added the first form, but the second form needs some design work:

This is going to require defining a @protocol class Sequence
that can be implemented by list[T], tuple[T], slice[T], and any user classes
that want to participate in this.

Additionaly, iter() is not implemented yet, maybe something like:

def iter[T]( seq: Sequence[T] ) -> Iterator[T,StopIteration]:
	for item in seq:
		yield item

we also need the following functions (I'm not sure if they are all correct):

def any[T]( seq: Sequence[T] ) -> bool:
	for item in seq:
		if item:
			return True
	return False

def all[T]( seq: Sequence[T] ) -> bool:
	for item in seq:
		if not item:
			return False
	return True

def enumerate[T]( seq: Sequence[T], start: isize = 0 ) -> Iterator[tuple[isize,T],StopIteration]:
	with compiler.wrap_arithmetic:
		for item in seq:
			yield start, item
			start += 1

def map[T,U]( fn: Callable[[T],U], seq: Sequence[T] ) -> Iterator[U,StopIteration]:
	for item in seq:
		yield fn( item )

def reduce[T]( fn: Callable[[T,T],U], seq: Sequence[T] ) -> Iterator[T,StopIterator]:
	it = iter( seq )
	value1 = next( it )
	for value2 in it:
		value1 = fn( value1, value2 )
	return value1

def sum[T]( seq: Sequence[T], start: T = 0 ) -> T:
	for item in seq:
		start += item
	return start

================================================================================

need ssl server support, including self-signed certs

================================================================================

there seems to be a bug with list. The following code works fine:

x: list[str] = [ 'foo' ]
print( x.__getitem__( 0 ).unwrap() )

but the following code requires the function it's in to return a Result[IndexError]:

x: list[str] = [ 'foo' ]
print( x[0].unwrap() )

These two things should be the same, why is the 2nd producing an automatic .or_return()?

