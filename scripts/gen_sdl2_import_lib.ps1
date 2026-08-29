# Regenerates scripts/sdl2_import_lib/SDL2.lib from pip-installed pysdl2-dll's
# bare SDL2.dll (no .lib shipped) - see sdl2_test.py's module docstring.
# Requires: pip install pysdl2-dll, and vcvars64 (MSVC dumpbin/lib.exe) on PATH
# or discoverable via vswhere.

$ErrorActionPreference = 'Stop'

$dll = python -c "import sdl2dll, pathlib; print(pathlib.Path(sdl2dll.__file__).parent / 'dll' / 'SDL2.dll')"
$outDir = Join-Path $PSScriptRoot 'sdl2_import_lib'
New-Item -ItemType Directory -Force -Path $outDir | Out-Null
$defPath = Join-Path $outDir 'SDL2.def'
$libOut  = Join-Path $outDir 'SDL2.lib'

$vswhere = "${env:ProgramFiles(x86)}\Microsoft Visual Studio\Installer\vswhere.exe"
$vcvars = & $vswhere -latest -products '*' -all -find 'VC\Auxiliary\Build\vcvars64.bat'

# needed symbols come from lib/windows/sdl2.py's @extern declarations -
# extend this list (and re-run) if that file grows more bindings.
$needed = @('SDL_Init','SDL_Quit','SDL_GetError','SDL_CreateWindow','SDL_DestroyWindow',
  'SDL_CreateRenderer','SDL_DestroyRenderer','SDL_SetRenderDrawColor','SDL_RenderClear',
  'SDL_RenderPresent','SDL_PollEvent','SDL_Delay')

"LIBRARY SDL2" | Out-File -Encoding ascii $defPath
"EXPORTS" | Out-File -Encoding ascii -Append $defPath
foreach ($n in $needed) { $n | Out-File -Encoding ascii -Append $defPath }

cmd.exe /c "call `"$vcvars`" >nul && lib.exe /DEF:`"$defPath`" /OUT:`"$libOut`" /MACHINE:X64"
Write-Host "wrote $libOut"
