#!/usr/bin/env python3
"""
Filter out lines matching unwanted patterns using regex rules.
Usage: python 15_regex_filter_bad_lines.py input_file.txt output_file.txt
"""

import re
import sys
import argparse
from typing import List, Pattern

def load_default_patterns() -> List[str]:
    """Load default regex patterns for filtering."""
    return [
        r'\b(fuck|shit|damn|hell)\b',           # Basic profanity
        r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',  # Email addresses
        r'\{\{.*?\}\}',                         # Template variables {{variable}}
        r'<.*?>',                               # HTML tags
        r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+',  # URLs
        r'\$\{.*?\}',                           # Shell-style variables ${variable}
        r'__.*?__',                             # Double underscore patterns
    ]

def compile_patterns(patterns: List[str]) -> List[Pattern]:
    """Compile regex patterns for better performance."""
    compiled = []
    for pattern in patterns:
        try:
            compiled.append(re.compile(pattern, re.IGNORECASE))
        except re.error as e:
            print(f"Warning: Invalid regex pattern '{pattern}': {e}")
    return compiled

def filter_lines(input_file: str, output_file: str, patterns: List[str], 
                invert: bool = False, verbose: bool = False) -> None:
    """
    Filter lines from input file and write results to output file.
    
    Args:
        input_file: Path to input file
        output_file: Path to output file
        patterns: List of regex patterns to filter
        invert: If True, keep only matching lines (opposite behavior)
        verbose: If True, print filtering statistics
    """
    compiled_patterns = compile_patterns(patterns)
    
    if verbose:
        print(f"Loaded {len(compiled_patterns)} filtering patterns")
    
    try:
        with open(input_file, 'r', encoding='utf-8', errors='ignore') as infile:
            lines = infile.readlines()
    except FileNotFoundError:
        print(f"Error: Input file '{input_file}' not found")
        return
    except Exception as e:
        print(f"Error reading input file: {e}")
        return
    
    kept_lines = 0
    filtered_lines = 0
    
    with open(output_file, 'w', encoding='utf-8') as outfile:
        for line_num, line in enumerate(lines, 1):
            # Check if line matches any unwanted pattern
            matched = any(pattern.search(line) for pattern in compiled_patterns)
            
            # Keep line if it doesn't match (or matches if inverted)
            should_keep = not matched if not invert else matched
            
            if should_keep:
                outfile.write(line)
                kept_lines += 1
            else:
                filtered_lines += 1
                if verbose:
                    print(f"Filtered line {line_num}: {line.strip()}")
    
    if verbose:
        print(f"\nProcessing complete:")
        print(f"  - Lines kept: {kept_lines}")
        print(f"  - Lines filtered: {filtered_lines}")
        print(f"  - Output written to: {output_file}")

def main():
    parser = argparse.ArgumentParser(description="Filter out lines matching unwanted patterns")
    parser.add_argument("input_file", help="Input file to process")
    parser.add_argument("output_file", help="Output file for filtered results")
    parser.add_argument("-p", "--pattern", action="append", dest="patterns",
                       help="Custom regex pattern to filter (can be used multiple times)")
    parser.add_argument("-f", "--pattern-file", 
                       help="File containing regex patterns (one per line)")
    parser.add_argument("-i", "--invert", action="store_true",
                       help="Invert matching (keep only matching lines)")
    parser.add_argument("-v", "--verbose", action="store_true",
                       help="Show detailed filtering information")
    parser.add_argument("--no-defaults", action="store_true",
                       help="Don't use default filtering patterns")
    
    args = parser.parse_args()
    
    # Determine which patterns to use
    patterns = []
    
    if not args.no_defaults:
        patterns.extend(load_default_patterns())
    
    if args.patterns:
        patterns.extend(args.patterns)
    
    if args.pattern_file:
        try:
            with open(args.pattern_file, 'r') as f:
                file_patterns = [line.strip() for line in f if line.strip() and not line.startswith('#')]
                patterns.extend(file_patterns)
        except FileNotFoundError:
            print(f"Error: Pattern file '{args.pattern_file}' not found")
            return
        except Exception as e:
            print(f"Error reading pattern file: {e}")
            return
    
    if not patterns:
        print("Warning: No patterns specified for filtering")
        patterns = [r'^$']  # Default to removing empty lines if nothing else
    
    filter_lines(
        input_file=args.input_file,
        output_file=args.output_file,
        patterns=patterns,
        invert=args.invert,
        verbose=args.verbose
    )

if __name__ == "__main__":
    main()