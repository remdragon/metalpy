# lib/case_folding.py — optional full-Unicode str.upper()/str.lower() casing
#
# Importing this module by itself does nothing; call install() once (e.g. at
# the top of main()) to point builtins.case_folder's tables at the real
# Unicode simple-casing data (UnicodeData.txt's own simple uppercase/
# lowercase mappings, fetched/cached at compile time - see
# PLAN_CASE_FOLDING.md), which redirects every subsequent str.upper()/
# str.lower() call in the program away from the OS's own casing
# (LCMapStringEx / towupper_l+towlower_l - ASCII-correct, but wrong past
# ASCII on POSIX systems without a full Unicode-aware locale) to this table.
#
# A program that never imports case_folding never references the table at
# all, so the data is never linked in - compiler.fetch_unicode_table() is
# only ever called from here, so it only runs (and only gets embedded) for
# programs that actually import this module and call install().

import builtins
import compiler

def install() -> None:
	upper: bytes = compiler.fetch_unicode_table( 'upper' )
	lower: bytes = compiler.fetch_unicode_table( 'lower' )
	with compiler.panic_arithmetic( 'unreachable: fetch_unicode_table() always returns a whole number of 8-byte entries' ):
		builtins.case_folder.upper_count = len( upper ) // 8
		builtins.case_folder.lower_count = len( lower ) // 8
	builtins.case_folder.upper_table = upper.get_const_ptr()
	builtins.case_folder.lower_table = lower.get_const_ptr()
