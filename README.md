Welcome to MetalPy

MetalPy is something I've wanted to build for years. It is a love
letter to the Python programming language. It compiles Python source
(with a few significant exceptions) down to C source code and generates
an executable.

As I said, I love the Python programming language. I can express ideas
and implement them very rapidly.

There's a few reasons for MetalPy, a few things about Python that are
less than ideal:

	* it's hard to know what Exceptions might interrupt your code
	* performance is rather poor.
	* building and distributing a hello world windows executable is 10's of megabytes.

In order to make something that fixes these issues, but remains as similar
to Python as possible, here are the solutions that I came up with:

	* Exceptions are out
		- Result objects that can't be ignored, inspired by Rust
		- defer and errdefer for resource cleanup, inspired by Zig
		- local try/except/finally syntax is supported as an alternate way to
			handle Result objects (via .or_throw()), but it's not real
			exceptions/unwinding - it only reaches a handler within the same
			function, not across calls
	* "everything is an object" is not supported
	* Multiple Inheritance is not supported
		- very hard to get right for very little benefit
	* deterministic __del__(), the __del__() method fires immediately when an
		object's refcount goes to 0, no more work-arounds to make sure resources
		clean up in a tiny manner

Features that exist now:

	* native integer types with different arithmetic modes
		default checked - creates a OverflowError or ZeroDivisionError
		wrap arithmetic - overflows just wrap (ZeroDivision still checked)
		saturate arithmetic - overflows clamp to the min/max (ZeroDivision still checked)
		panic arithmetic - overflows and ZeroDivision panic (error message and abort
	* memory-management through automatic ref-counting
	* unbounded int class
	* threadsafe list
	* threadsafe dict
	* threadsafe set
	* tuple
	* anonymous unions like int|str with both compile time and runtime type narrowing
	* subclasses (single inheritance only)
	* file i/o
		- we broke significantly away from python's open() method here, because
		  the mode string is difficult for type-safety and cannot express all
		  valid combinations if file i/o
	* generic functions and classes
	* threading, non-reentrant Lock, atomic primitives
	* same-named functions chosen at either compile-time or run-time based on parameters
		- 2 layers of resolution, normal functions must have an exact type match
		- @overload functions are first-match wins
	* ability to create actual C structs (can interact directly with COM interfaces)
	* on Windows, the stdlib avoids linking against msvcrt. However, use of any
		feature that requires the C runtime will bring it in.
	* lazily scan source file and only compile things actually needed
	* f-strings
		most common use-cases are implemented and working
	* inline functions
	* blocking TCP/UDP sockets over IPv4/IPv6 (lib/socket.py)
	* global object initialization on startup
	* re library
	* dependency report showing every object included in the compilation and
		what triggered its inclusion, useful for troubleshooting executable
		bloat.
	* local try/except/else/finally syntax over Result objects (via
		.or_throw() / raise EXPR), still not real exceptions/unwinding

Features that are being scoped and built right now:

	* generators (mostly functional, currently researching the ability to
		support inline generators)
	* url parser
	* email.message (needed by http client)
	* http client
	* json library
	* tkinter library

Features that are planned but not built yet:

	* with statements (there is a compiler hack for defer/errdefer using with
		statement syntax, but this isn't general with support yet)
	* threadsafe Queue
	* http server
	* smtp library
	* email parsing
	* parser library
	* unblocking socket i/o and file i/o

Features that would be nice to have:

	* pip-like package manager

Python functionality that I have no plans to implement:

	* async/await
		- I'm open to others wanting to do this work, no interest in it myself
	* multiple inheritance
		- too many footguns for too little benefit
	* everything is an object
		- performance nightmare
	* monkey-patching
		- requires everything is an object
	* Exceptions
		- rust-inspired Result objects for mandatory and deterministic error handling

Other thoughts / features:

Right now, MetalPy uses Python's own AST parser and uses a C compiler to generate
the actual executable. This gives MetalPy a lot of portability. While I plan to build
a parser library, I don't have any plan to make MetalPy self-hosting. There are some
small things about MetalPy's syntax that could be made nicer by stepping away from
Python's parser, but restricting to Python's AST parser opens up a lot of opportunities
to use existing tools for source code analysis.

MetalPy does not create .obj files. It does a lazy evaluation of everything main
could possibly touch, and then only fully evaluates items that are actually
needed for the executable. If a function is never called by main or by any other
function reached indirectly, the MetalPy compiler never evens walks through the
body of the unused function.

MetalPy tries very hard to avoid msvcrt on Windows builds, but if users need
to link to it for whatever reason, MetalPy doesn't prevent, it just doesn't
need it for the stdlib (yet).

License:

MetalPy itself is licensed under Apache 2.0 (see LICENSE/NOTICE). Programs
compiled by MetalPy may bundle third-party runtime components (e.g. Tcl/Tk,
zlib) under their own licenses - see licenses/ and the generated
dist/THIRD-PARTY-LICENSES.txt for those.
