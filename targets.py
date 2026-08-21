import platform
from typing import get_args, get_type_hints, Literal, TypedDict

_HOST_OS_TO_TARGET_OS = {
	'Windows': 'windows',
	'Linux': 'linux',
	'Darwin': 'macos',
}
_HOST_MACHINE_TO_TARGET_ARCH = {
	'AMD64': 'x86_64',
	'x86_64': 'x86_64',
	'arm64': 'arm64',
	'aarch64': 'arm64',
}

class ActiveTarget( TypedDict ):
	os: Literal['windows','linux','macos']
	arch: Literal['x86_64','arm64']
	family: Literal['windows','unix']
	bits: Literal[64]
	debug: bool
	posix: bool
	has_i128: bool

_OS_NAMES = get_args(get_type_hints(ActiveTarget)['os'])
_ARCH_NAMES = get_args(get_type_hints(ActiveTarget)['arch'])
_FAMILY_NAMES = get_args(get_type_hints(ActiveTarget)['family'])

def detect() -> ActiveTarget:
	os_name = _HOST_OS_TO_TARGET_OS.get( platform.system(), platform.system().lower() )
	arch = _HOST_MACHINE_TO_TARGET_ARCH.get( platform.machine(), platform.machine() )
	# family is one of SYNTAX.md's FamilySpec literals ('unix'/'windows'/
	# 'wasm') - 'posix' is a *separate* bool field on TargetQuery, not a
	# family value
	family = 'windows' if os_name == 'windows' else 'unix'
	# debug=True by default (matches sys.alloc()'s existing debug-only
	# zeroing behavior) - overridable via Discovery(active_target=...) same
	# as every other key, until a real CLI exposes a release-build flag
	assert os_name in _OS_NAMES, f'invalid {os_name=}'
	assert arch in _ARCH_NAMES, f'invalid {arch=}'
	assert family in _FAMILY_NAMES, f'invalid {family=}'
	return ActiveTarget(
		os = os_name, # type: ignore[typeddict-item]
		arch = arch, # type: ignore[typeddict-item]
		family = family, # type: ignore[typeddict-item]
		bits = 64,
		debug = True,
		posix = family == 'unix',
		has_i128 = True, # just default to true
	)
