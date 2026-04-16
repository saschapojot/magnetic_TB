import re
import sys
import os
import pickle
import numpy as np
from pathlib import Path
import copy


project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))
from name_conventions import symmetry_matrices_file_name
#this script parse .cif file to get atom positions, symmetry operations
#.cif file is generated from https://iso.byu.edu/findsym.php


paramErrCode = 3         # Wrong command-line parameters
fileNotExistErrCode = 4  # Configuration file doesn't exist
tol=1e-3
if len(sys.argv) != 2:
    print("wrong number of arguments.", file=sys.stderr)
    print("usage: python parse_cif.py /path/to/xxx.cif", file=sys.stderr)
    exit(paramErrCode)



cif_file_name = str(sys.argv[1])
# Check if configuration file exists
if not os.path.exists(cif_file_name):
    print(f"file not found: {cif_file_name}", file=sys.stderr)
    exit(fileNotExistErrCode)


def remove_comments_and_empty_lines_cif(file):
    """
    Remove comments and empty lines from a CIF-like configuration file.

    - Comments start with # and continue to end of line.
    - # inside single (') or double (") quotes are preserved.
    - Empty lines (or lines with only whitespace) are removed.

    :param file: conf file path
    :return: list of cleaned lines
    """
    with open(file, "r") as fptr:
        lines = fptr.readlines()

    linesToReturn = []
    # Regex explanation:
    # 1. (["\'])  -> Capture group 1: Match a quote (single or double)
    # 2. (?:      -> Non-capturing group for content inside quotes
    #    \\.      -> Match escaped characters (like \")
    #    |        -> OR
    #    [^\\]    -> Match any character that isn't a backslash
    #    )*?      -> Match as few times as possible (lazy)
    # 3. \1       -> Match the closing quote (same as group 1)
    # 4. |        -> OR
    # 5. (#.*$)   -> Capture group 2: Match a real comment (starts with #, goes to end)
    # This pattern finds quoted strings OR comments.
    # We will use a callback to decide what to do.
    pattern = re.compile(r'(["\'])(?:\\.|[^\\])*?\1|(#.*$)')
    for oneLine in lines:
        # We use a callback function for re.sub
        # If group 2 (the comment) exists, replace it with empty string.
        # If group 1 (the quote) matches, return the whole match (preserve it).
        def replace_func(match):
            if match.group(2):
                return ""  # It's a comment, remove it
            else:
                return match.group(0)  # It's a quoted string, keep it
        # Apply the regex
        cleaned_line = pattern.sub(replace_func, oneLine).strip()
        # Only add non-empty lines
        if cleaned_line:
            linesToReturn.append(cleaned_line)

    return linesToReturn



coef_pattern = re.compile(r"([+-]?)\s*([xyz])", re.IGNORECASE)
# Matches: optional sign, optional space, number (integer, decimal, or fraction)
# Example: "+1/2", "-0.5", "1/4", "+1"
translation_pattern = re.compile(r"([+-]?)\s*(\d+(?:[./]\d+)?)")