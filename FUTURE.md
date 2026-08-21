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
executable so it can calculate paths relative to that (not saying this is a
good thing but sometimes users want this)

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