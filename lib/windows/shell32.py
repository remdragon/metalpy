# LPWSTR* CommandLineToArgvW(LPCWSTR lpCmdLine, int *pNumArgs) - splits a
# command line into argv-style pieces using the same quoting/escaping rules
# real Windows programs expect (backslash-escaped quotes, etc.) - see
# sys.py's own _build_argv (windows branch), the only caller. The returned
# array (and the strings it points to, allocated as one block) is freed with
# a single kernel32.LocalFree(argv) call, not per-element.
@extern( 'shell32', 'CommandLineToArgvW' )
def CommandLineToArgvW(
	lpCmdLine: ConstPtr[u16],
	pNumArgs: Ptr[i32],
) -> Ptr[Ptr[u16]]:
	...
