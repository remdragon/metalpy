# src/unix/time.py

def get_local_timezone_name() -> str:
	# sys_readlink syscall or libc wrapper
	from posix.fs import readlink
	target_path = readlink( '/etc/localtime' ).unwrap_or()
	
	if target_path:
		idx = target_path.find( 'zoneinfo/' )
		if idx != -1:
			return target_path[idx+9:]
	
	return _read_etc_timezone_file()


def _read_etc_timezone_file() -> str:
	if f := open( '/etc/timezone', 'r' ).unwrap_or():
		with defer:
			f.close()
		if s := f.read( 128 ):
			return s.strip()
	
	return 'UTC'
