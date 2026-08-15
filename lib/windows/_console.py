# Compiler-internal only - never imported by user code. compiler.py's own
# Compiler.run() force-imports this module and enqueues _console_init below
# unconditionally on every Windows target (the same way it unconditionally
# enqueues main() itself), so the ordinary global-init machinery
# (emitter_c.py's __metalpy_init synthesis) calls SetConsoleOutputCP(CP_UTF8)
# once before main() runs, on every Windows build, regardless of whether the
# user's own program ever touches kernel32 - replaces the old hand-rolled
# #ifdef _WIN32 declaration + call that used to live directly in emitter_c.py's
# PROLOGUE/__metalpy_init synthesis.
#
# A plain function-initialized bool, NOT a class instance: this module gets
# force-imported into every Discovery, including the many `import_builtins =
# False` test harnesses that deliberately never pull in lib/builtins (see
# emitter_c_test.py's CompilerTestCase). A real `_ConsoleInit()` construction
# would go through lib/sys.py's own RC-allocation path (alloc[T]), which
# needs `str` (its own panic message) - unavailable without builtins, and a
# real, confirmed failure caught by EmitGlobalTests.test_trivial_global_is_a_
# real_static_initializer. bool/u32 are compiler intrinsics, always available
# regardless of import_builtins, so this stays allocation-free.
from windows.kernel32 import SetConsoleOutputCP, CP_UTF8

def _init_console() -> bool:
	return SetConsoleOutputCP( CP_UTF8 )

_console_init: bool = _init_console()
