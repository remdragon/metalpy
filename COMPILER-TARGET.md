okay, we need to fix @compiler.target support.

I want to do this as clean as possible.

There are 2 phases to this.

Phase 1:

	discovery.py needs to recognized @compiler.target( ... ) on classes and functions.
	
	If the condition is not met, the object is discarded immediately.
	
Phase 2:

	When discovery.py is asked to resolve a function, it needs to pass that function's
	body through a new file: compile_time_transformer.py.
	
	This new file will be an AST transformer.
	
	It will simplify compile-time constant expressions.
	
		`1 + 1` gets simplified to `1`
		
		`compiler.target.os` gets simplified to `"windows"`
		
		`"windows" == "windows"` gets simplified to `True`
	
	Any constructs that have a compile-time constant condition will be simplified:
	
		```
		if True:
			print( 'foo' )
		else:
			print( 'bar' )
		```
		
		becomes just: `print('foo')`
		
		whereas:
		
		```
		if False:
			print( 'foo' )
		else:
			print( 'bar' )
		```
		
		becomes just: `print('bar')`
		
		this applies to loops too:
		
		```
		while True:
			print( 'foo' )
		```
		
		becomes:
		
		```
		_label_0:
			print( 'foo' )
			goto _label_0
		```
		
		whereas this gets eliminated entirely:
		
		```
		while False:
			print( 'foo' )
		```

FYI, here is a snippet from a previous version of metalpy that you can use to
bootstrap the compiler.target state:

_HOST_OS_TO_TARGET_OS = {
	'Windows': 'windows',
	'Linux': 'linux',
	'Darwin': 'macos',
}
_HOST_MACHINE_TO_TARGET_ARCH = {
	'x86_64': 'x86_64', 'amd64': 'x86_64',
	'arm64': 'arm64', 'aarch64': 'arm64',
}

def _detect_active_target() -> dict:
	host_os = platform.system()
	target_os = _HOST_OS_TO_TARGET_OS.get( host_os )
	if target_os is None:
		raise RuntimeError(
			f'unsupported host OS {host_os!r} (platform.system()) - MetalPy only knows how to '
			f"target one of {sorted(_HOST_OS_TO_TARGET_OS.values())!r} right now"
		)
	target_arch = _HOST_MACHINE_TO_TARGET_ARCH.get( platform.machine().lower(), platform.machine().lower() )
	return {
		'os': target_os,
		'arch': target_arch,
		'family': 'windows' if target_os == 'windows' else 'posix',
		'bits': 64,
	}
