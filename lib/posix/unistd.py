# lib/posix/unistd.py — sysconf(3), just enough for CPU count (sys.cpu_count).
#
# Declared WITHOUT header='unistd.h' (unlike compiler.cexpr's own constant
# probe below, a separate throwaway compile that never touches the real
# generated.c) - actually #include-ing unistd.h into generated.c conflicts
# with this codebase's own hand-declared write()/read()/etc. externs
# (lib/crt.py): ABI-identical but spelled with different C type names
# (intptr_t/uint8_t/uintptr_t here vs ssize_t/void*/size_t in the real
# header), which C treats as a hard "conflicting types" redeclaration
# error, not a warning - confirmed via a real gcc compile attempt. Same
# reasoning fcntl.py's own header-less fcntl() extern already documents.

import compiler

_SC_NPROCESSORS_ONLN: i32 = compiler.cexpr( '_SC_NPROCESSORS_ONLN', 'unistd.h', i32 )

@extern( 'c', 'sysconf' )
def sysconf( name: i32 ) -> i64:
	...
