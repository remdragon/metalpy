Based on the following chat with DeepSeek, let's look into implement str case
folder in metalpy via:

	* windows: LCMapStringEx from kernel32
		* we'll have to convert to UTF16 with MultiByteToWideChar
		* and back to UTF8 afterwards with WideCharToMultiByte
	* linux: ICU
		* fallback to towupper and towlower if LCU not available



are there OS functions I can rely on to perform proper unicode-correct upper and lower casing for all the over 600 translations that exist in unicode? I'm looking for something in the crt for linux and something in ntdll/kernel32/etc in windows (not msvcrt)
Search is unavailable in Expert Mode. Please use Instant Mode.
Yes, both Linux and Windows provide OS-level functions that handle full Unicode casing correctly, including locale-specific rules. Here's what you can rely on:

Linux (glibc)
towupper/towlower with proper locale:

c
#include <wctype.h>
#include <locale.h>

setlocale(LC_ALL, "en_US.UTF-8");  // or your locale
wint_t upper = towupper(L'ß');     // becomes ẞ (U+1E9E)
These functions handle all Unicode casing rules including:

One-to-many mappings (e.g., ß → SS uppercased)

Context-sensitive casing (e.g., Greek final sigma)

Locale-specific rules (Turkish I, Lithuanian dot preservation)

For full string casing, use ICU if available, but if you must stay in libc:

c
// Manual string processing using towupper/towlower per character
// Note: This won't handle 1-to-many expansions properly
The real OS-level solution on Linux is actually ICU (commonly present):

c
#include <unicode/ucasemap.h>
UCaseMap *csm = ucasemap_open("tr", 0, &status);
int32_t len = ucasemap_utf8ToUpper(csm, dest, destCapacity, src, srcLength, &status);
Windows (ntdll/kernel32)
LCMapStringEx from kernel32.dll:

c
#include <windows.h>

// For UTF-16 strings (native Windows encoding):
wchar_t src[] = L"straße";
wchar_t dest[100];
int result = LCMapStringEx(
    LOCALE_NAME_USER_DEFAULT,  // or specific locale like L"tr-TR"
    LCMAP_UPPERCASE | LCMAP_LINGUISTIC_CASING,
    src, wcslen(src),
    dest, ARRAYSIZE(dest),
    NULL, NULL, 0
);
Key flags:

LCMAP_LINGUISTIC_CASING - Uses full linguistic rules instead of simple mapping

LCMAP_UPPERCASE / LCMAP_LOWERCASE - Direction

LCMAP_TITLECASE - Title case

LOCALE_NAME_USER_DEFAULT - Current user locale (handles Turkish I correctly)

For UTF-8 input, convert first:

c
MultiByteToWideChar(CP_UTF8, 0, utf8src, -1, widesrc, capacity);
LCMapStringEx(...);
WideCharToMultiByte(CP_UTF8, 0, widedest, -1, utf8dest, capacity, NULL, NULL);
Advanced: NORM_LINGUISTIC_CASING flag (Vista+):
This handles special cases like:

German ß → SS

Greek final sigma at word boundaries

Lithuanian dot preservation

Turkish/Azeri dotted/dotless I

Important Notes
No single CRT function handles this correctly - towupper/towlower work character-by-character but can't handle expansion cases (1 char → multiple chars) in a string context.

Locale matters: Turkish casing rules (I→ı, İ→i) require the correct locale set. Both LCMapStringEx with locale parameter and setlocale + towupper handle this.

String context required: Some rules depend on context (Greek sigma at word end), which towupper alone can't handle. Windows' LCMapStringEx with LCMAP_LINGUISTIC_CASING does.

For Linux without ICU: You're limited to character-by-character mappings with towupper/towlower, which handles most cases but misses string-level context and expansion.

The NLS files in Windows: These functions use the same Unicode data as ICU through Windows' NLS system files in C:\Windows\System32\*.nls - they're comprehensive and cover all 600+ translations.

The bottom line: LCMapStringEx from kernel32 on Windows and ICU on Linux are your best bets for truly correct, full Unicode casing. CRT functions alone won't cut it for all cases.