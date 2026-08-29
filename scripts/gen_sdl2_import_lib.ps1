# Regenerates scripts/sdl2_import_lib/{SDL2,SDL2_image}.lib from pip-installed
# pysdl2-dll's bare DLLs (no .lib shipped) - see sdl2_test.py's module
# docstring. Requires: pip install pysdl2-dll, and vcvars64 (MSVC dumpbin/
# lib.exe) on PATH or discoverable via vswhere.

$ErrorActionPreference = 'Stop'

$dllDir = python -c "import sdl2dll, pathlib; print(pathlib.Path(sdl2dll.__file__).parent / 'dll')"
$outDir = Join-Path $PSScriptRoot 'sdl2_import_lib'
New-Item -ItemType Directory -Force -Path $outDir | Out-Null

$vswhere = "${env:ProgramFiles(x86)}\Microsoft Visual Studio\Installer\vswhere.exe"
$vcvars = & $vswhere -latest -products '*' -all -find 'VC\Auxiliary\Build\vcvars64.bat'

function New-ImportLib( [string]$libName, [string[]]$needed ) {
	# needed symbols come from lib/windows/$libName.py's @extern declarations -
	# extend the caller's list (and re-run) if that file grows more bindings.
	$defPath = Join-Path $outDir "$libName.def"
	$libOut  = Join-Path $outDir "$libName.lib"

	"LIBRARY $libName" | Out-File -Encoding ascii $defPath
	"EXPORTS" | Out-File -Encoding ascii -Append $defPath
	foreach ($n in $needed) { $n | Out-File -Encoding ascii -Append $defPath }

	cmd.exe /c "call `"$vcvars`" >nul && lib.exe /DEF:`"$defPath`" /OUT:`"$libOut`" /MACHINE:X64"
	Write-Host "wrote $libOut"
}

New-ImportLib 'SDL2' @('SDL_Init','SDL_Quit','SDL_GetError','SDL_CreateWindow','SDL_DestroyWindow',
  'SDL_CreateRenderer','SDL_DestroyRenderer','SDL_SetRenderDrawColor','SDL_RenderClear',
  'SDL_RenderPresent','SDL_PollEvent','SDL_Delay','SDL_RenderCopy','SDL_DestroyTexture')

New-ImportLib 'SDL2_image' @('IMG_Init','IMG_Quit','IMG_LoadTexture')
