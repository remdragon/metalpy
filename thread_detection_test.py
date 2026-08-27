# Real-compile tests for PLAN_THREAD_SAFE_SHARED_STATE.md's Cost mitigation
# #1 - the whole-program on/off switch. If a compiled program never reaches
# posix.pthread.pthread_create()/windows.kernel32.CreateThread() (directly or
# transitively, e.g. through lib/threading.py's Thread/ThreadPool, lib/
# reactor.py, lib/tcpserver.py's dispatcher), nothing can ever race a global/
# field, so Part A/B's real lock codegen is skipped entirely.
#
# Detection: @extern(..., spawns_thread=True) tags the two real OS-thread-
# creation syscall bindings (discovery.py's _parse_extern_decorator);
# Compiler.spawns_threads flips the moment such a function is actually
# reached and lowered (compiler.py, right beside the identical requires_crt
# precedent) - no scan needed. emitter_c.py's emit_c() reads this into
# _program_uses_threads, which gates every AcquireGlobalLock/ReleaseGlobalLock/
# AcquireFieldLock/ReleaseFieldLock marker's real-vs-no-op decision (same
# emission-time-decided pattern _global_lock_supported() already uses).
#
# thread_safe_globals_test.py/thread_safe_fields_test.py's own stress tests
# (which all DO spawn real threads) are the sabotage-and-confirm evidence
# that this flag is load-bearing, not a no-op gate: forcing
# emitter_c._program_uses_threads off unconditionally reproduced a real
# STATUS_ILLEGAL_INSTRUCTION crash in 15/15 runs, restored immediately after
# confirming that. This file covers the OTHER half: that the flag correctly
# stays OFF (no lock codegen at all) for a program that never spawns a
# thread, and correctly turns ON for one that does, including transitively
# through lib/threading.py's own wrappers.

import unittest

import discovery
import compiler as compiler_module
import emitter_c
import test_support
from pathlib import Path


@unittest.skipUnless( test_support.HAS_CC, 'no C compiler (clang/gcc/msvc) found - skipping real-compile tests' )
class ThreadDetectionTests( unittest.TestCase ):
	def _compile( self, source: str ) -> tuple[compiler_module.Compiler, str]:
		disco = discovery.Discovery( import_builtins = True )
		compiler = compiler_module.Compiler( disco )
		compiler.import_code( source, Path( '__main__.py' ), scope = None )
		compiler.run()
		self.assertEqual( disco.errors.errors, [], disco.errors.errors )
		c_source = emitter_c.emit_c( compiler )
		return compiler, c_source

	def test_no_thread_spawned_emits_no_lock_codegen( self ) -> None:
		# a genuinely reassigned global AND a shared instance field, neither
		# ever touched from a second thread - both should compile with zero
		# real lock/acquire codegen
		source = '\n'.join([
			'import compiler',
			'class Box:',
			'	x: i32',
			'	def __init__( self, x: i32 ) -> None:',
			'		self.x = x',
			'',
			'g: Box = Box( 0 )',
			'',
			'def swap( n: i32 ) -> None:',
			'	global g',
			'	g = Box( n )',
			'',
			'class Holder:',
			'	b: Box',
			'	def __init__( self, b: Box ) -> None:',
			'		self.b = b',
			'',
			'def main() -> i32:',
			'	swap( 1 )',
			'	h: Holder = Holder( Box( 2 ) )',
			'	h.b = Box( 3 )',
			'	with compiler.wrap_arithmetic:',
			'		return h.b.x + g.x',
		])
		compiler, c_source = self._compile( source )
		self.assertFalse( compiler.spawns_threads )
		# acquire_field_lock/release_field_lock/pthread_mutex_init are
		# unique to Part A/B's own mechanism - unlike AcquireSRWLockExclusive/
		# pthread_mutex_lock, which a debug build's own UNRELATED
		# __metalpy_debug_lock (alloc-site tracking) also legitimately uses,
		# so checking those would be a false positive against a completely
		# different lock this compiler already has.
		for needle in ( 'acquire_field_lock', 'release_field_lock', 'pthread_mutex_init' ):
			self.assertNotIn( needle, c_source, f'{needle!r} found in generated C for a program that never spawns a thread' )

	def test_thread_construction_sets_spawns_threads( self ) -> None:
		source = '\n'.join([
			'import threading',
			'class Worker:',
			'	def run( self ) -> None:',
			'		pass',
			'',
			'def main() -> i32:',
			'	w: Worker = Worker()',
			'	t: threading.Thread = threading.Thread( w.run )',
			'	t.join()',
			'	return 0',
		])
		compiler, _c_source = self._compile( source )
		self.assertTrue( compiler.spawns_threads )

	def test_thread_spawned_emits_real_lock_codegen( self ) -> None:
		# the positive-space counterpart of test_no_thread_spawned_emits_no_
		# lock_codegen above - a program that DOES spawn a thread and DOES
		# touch a shared field must still get the real lock codegen, not
		# have it silently dropped
		source = '\n'.join([
			'import threading',
			'class Box:',
			'	x: i32',
			'	def __init__( self, x: i32 ) -> None:',
			'		self.x = x',
			'class Holder:',
			'	b: Box',
			'	def __init__( self, b: Box ) -> None:',
			'		self.b = b',
			'class Worker:',
			'	def run( self ) -> None:',
			'		pass',
			'',
			'def main() -> i32:',
			'	h: Holder = Holder( Box( 1 ) )',
			'	h.b = Box( 2 )',
			'	w: Worker = Worker()',
			'	t: threading.Thread = threading.Thread( w.run )',
			'	t.join()',
			'	return h.b.x',
		])
		compiler, c_source = self._compile( source )
		self.assertTrue( compiler.spawns_threads )
		# real CALL sites, not just the helper function definitions (which
		# Cost mitigation #2 can cause to be emitted even when every actual
		# per-field access at this specific site is a no-op - see
		# thread_detection_test.py's own exemption test for why)
		self.assertIn( 'acquire_field_lock( (ObjectHeader*)(', c_source )
		self.assertIn( 'release_field_lock( (ObjectHeader*)(', c_source )

	def test_assume_threaded_override_forces_lock_codegen( self ) -> None:
		# mpy.py's own --assume-threaded escape hatch (PLAN_THREAD_SAFE_
		# SHARED_STATE.md's own Open Question #5 - a program that reaches a
		# second OS thread some other way this compiler can't see, e.g. a
		# raw signal handler or an externally-invoked C callback) - setting
		# compiler.spawns_threads = True directly is exactly what mpy.py
		# itself does when the flag is passed, before ever calling emit_c()
		source = '\n'.join([
			'class Box:',
			'	x: i32',
			'	def __init__( self, x: i32 ) -> None:',
			'		self.x = x',
			'class Holder:',
			'	b: Box',
			'	def __init__( self, b: Box ) -> None:',
			'		self.b = b',
			'',
			'def main() -> i32:',
			'	h: Holder = Holder( Box( 1 ) )',
			'	h.b = Box( 2 )',
			'	return h.b.x',
		])
		disco = discovery.Discovery( import_builtins = True )
		compiler = compiler_module.Compiler( disco )
		compiler.import_code( source, Path( '__main__.py' ), scope = None )
		compiler.run()
		self.assertEqual( disco.errors.errors, [], disco.errors.errors )
		self.assertFalse( compiler.spawns_threads ) # nothing spawns a thread here
		compiler.spawns_threads = True # the override, applied exactly like mpy.py's --assume-threaded
		c_source = emitter_c.emit_c( compiler )
		self.assertIn( 'acquire_field_lock( (ObjectHeader*)(', c_source )
		self.assertIn( 'release_field_lock( (ObjectHeader*)(', c_source )

	def test_threadpool_construction_sets_spawns_threads( self ) -> None:
		# ThreadPool is a lib/threading.py wrapper, not a direct Thread() -
		# confirms the extern-tag-based detection catches it TRANSITIVELY
		# (it bottoms out at pthread_create/CreateThread internally), the
		# same way it would catch any future thread-spawning wrapper with
		# no detection code of its own needed
		source = '\n'.join([
			'import threading',
			'def main() -> i32:',
			'	pool: threading.ThreadPool = threading.ThreadPool( 1 )',
			'	return 0',
		])
		compiler, _c_source = self._compile( source )
		self.assertTrue( compiler.spawns_threads )


	def test_private_write_once_field_exempt_even_when_threaded( self ) -> None:
		# PLAN_THREAD_SAFE_SHARED_STATE.md Cost mitigation #2 - a `__private`
		# field written only from its own class's __init__ (Holder.__b) needs
		# no lock at all, even in a program that DOES spawn a real thread
		# elsewhere (compiler.spawns_threads True) - construction happens
		# single-threaded, before the object can be published anywhere.
		source = '\n'.join([
			'import compiler',
			'import threading',
			'class Box:',
			'	x: i32',
			'	def __init__( self, x: i32 ) -> None:',
			'		self.x = x',
			'class Holder:',
			'	__b: Box',
			'	def __init__( self, b: Box ) -> None:',
			'		self.__b = b',
			'	def get( self ) -> i32:',
			'		return self.__b.x',
			'class Worker:',
			'	def run( self ) -> None:',
			'		pass',
			'',
			'def main() -> i32:',
			'	h: Holder = Holder( Box( 1 ) )',
			'	w: Worker = Worker()',
			'	t: threading.Thread = threading.Thread( w.run )',
			'	t.join()',
			'	return h.get()',
		])
		compiler, c_source = self._compile( source )
		self.assertTrue( compiler.spawns_threads ) # program DOES spawn a thread elsewhere
		# the helper FUNCTIONS themselves (acquire_field_lock/release_field_
		# lock's own `static inline void ...` definitions) are emitted
		# whenever ANY field lock marker exists anywhere in the program - see
		# emitter_c.py's uses_field_lock, presence-based on purpose (Cost
		# mitigation #2's exemption is a per-ACCESS emission-time decision,
		# not a whole-program one). Checking for actual CALL sites instead -
		# their unique `(ObjectHeader*)(` cast, which no definition line has.
		for needle in ( 'acquire_field_lock( (ObjectHeader*)(', 'release_field_lock( (ObjectHeader*)(' ):
			self.assertNotIn( needle, c_source, f'{needle!r} found for a __private write-once field, even though it should be exempt' )

	def test_private_field_reassigned_outside_init_still_gets_lock_codegen( self ) -> None:
		# the positive-space counterpart of the exemption test above - a
		# `__private` field reassigned from a method OTHER than __init__
		# (even one still inside the same class, a legitimate private access)
		# must NOT be treated as exempt - confirms the detector actually
		# checks field_reassigned_outside_init, not just the '__' name shape.
		source = '\n'.join([
			'import compiler',
			'import threading',
			'class Box:',
			'	x: i32',
			'	def __init__( self, x: i32 ) -> None:',
			'		self.x = x',
			'class Holder:',
			'	__b: Box',
			'	def __init__( self, b: Box ) -> None:',
			'		self.__b = b',
			'	def __swap( self, b: Box ) -> None:',
			'		self.__b = b',
			'	def trigger_swap( self, b: Box ) -> None:',
			'		self.__swap( b )',
			'	def get( self ) -> i32:',
			'		return self.__b.x',
			'class Worker:',
			'	def run( self ) -> None:',
			'		pass',
			'',
			'def main() -> i32:',
			'	h: Holder = Holder( Box( 1 ) )',
			'	h.trigger_swap( Box( 2 ) )',
			'	w: Worker = Worker()',
			'	t: threading.Thread = threading.Thread( w.run )',
			'	t.join()',
			'	return h.get()',
		])
		compiler, c_source = self._compile( source )
		self.assertTrue( compiler.spawns_threads )
		# real CALL sites, not just the always-present helper function
		# definitions - see the exemption test's own identical comment above
		self.assertIn( 'acquire_field_lock( (ObjectHeader*)(', c_source )
		self.assertIn( 'release_field_lock( (ObjectHeader*)(', c_source )


if __name__ == '__main__':
	unittest.main()
