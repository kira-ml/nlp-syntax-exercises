import re
import logging
from typing import List, Set, Pattern, Optional, Iterator, Union
from pathlib import Path
import json

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class RegexLineFilter:
    """
    A robust regex-based line filter for removing unwanted content from text files.
    
    Features:
    - Precompiled regex patterns for performance
    - Configurable pattern loading from files or dictionaries
    - Support for both inclusion and exclusion patterns
    - Performance optimizations for large files
    - Comprehensive error handling
    """
    
    def __init__(self, 
                 exclude_patterns: Optional[List[str]] = None,
                 include_patterns: Optional[List[str]] = None,
                 case_sensitive: bool = False):
        """
        Initialize the filter with regex patterns.
        
        Args:
            exclude_patterns: List of regex patterns to exclude
            include_patterns: List of regex patterns to require (whitelist)
            case_sensitive: Whether pattern matching should be case-sensitive
        """
        self.case_sensitive = case_sensitive
        self.flags = 0 if case_sensitive else re.IGNORECASE
        
        # Precompile patterns for performance
        self.exclude_regexes: List[Pattern] = [
            re.compile(pattern, self.flags) 
            for pattern in (exclude_patterns or [])
        ]
        
        self.include_regexes: List[Pattern] = [
            re.compile(pattern, self.flags) 
            for pattern in (include_patterns or [])
        ]
        
        logger.info(f"Initialized filter with {len(self.exclude_regexes)} exclude "
                   f"and {len(self.include_regexes)} include patterns")
    
    @classmethod
    def from_config_file(cls, config_path: Union[str, Path]) -> 'RegexLineFilter':
        """
        Create filter instance from JSON configuration file.
        
        Expected JSON format:
        {
            "exclude_patterns": ["pattern1", "pattern2"],
            "include_patterns": ["pattern3"],
            "case_sensitive": false
        }
        """
        config_path = Path(config_path)
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                config = json.load(f)
            
            return cls(
                exclude_patterns=config.get('exclude_patterns'),
                include_patterns=config.get('include_patterns'),
                case_sensitive=config.get('case_sensitive', False)
            )
        except (FileNotFoundError, json.JSONDecodeError) as e:
            logger.error(f"Failed to load config from {config_path}: {e}")
            raise
    
    def should_exclude_line(self, line: str) -> bool:
        """
        Determine if a line should be excluded based on patterns.
        
        Args:
            line: Input text line to evaluate
            
        Returns:
            True if line matches any exclude pattern or fails include patterns
        """
        # Check exclude patterns first (faster short-circuit)
        if any(regex.search(line) for regex in self.exclude_regexes):
            return True
            
        # If include patterns exist, line must match at least one
        if self.include_regexes and not any(
            regex.search(line) for regex in self.include_regexes
        ):
            return True
            
        return False
    
    def filter_lines(self, lines: Iterator[str]) -> Iterator[str]:
        """
        Filter lines using configured regex patterns.
        
        Args:
            lines: Iterator of text lines to filter
            
        Yields:
            Lines that pass all filtering criteria
        """
        filtered_count = 0
        total_count = 0
        
        for line_num, line in enumerate(lines, 1):
            total_count += 1
            line_content = line.rstrip('\n\r')  # Preserve line endings in output
            
            if not self.should_exclude_line(line_content):
                yield line
            else:
                filtered_count += 1
                
                # Log every 1000 filtered lines to avoid log spam
                if filtered_count % 1000 == 0:
                    logger.debug(f"Filtered {filtered_count} lines so far (current: {line_num})")
        
        logger.info(f"Filtering complete: {filtered_count}/{total_count} lines filtered")
    
    def filter_file(self, 
                   input_path: Union[str, Path], 
                   output_path: Union[str, Path],
                   encoding: str = 'utf-8') -> int:
        """
        Filter lines from input file and write results to output file.
        
        Args:
            input_path: Path to input file
            output_path: Path to output file
            encoding: File encoding to use
            
        Returns:
            Number of lines written to output file
        """
        input_path = Path(input_path)
        output_path = Path(output_path)
        
        if not input_path.exists():
            raise FileNotFoundError(f"Input file not found: {input_path}")
            
        # Ensure output directory exists
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(input_path, 'r', encoding=encoding) as infile, \
             open(output_path, 'w', encoding=encoding) as outfile:
            
            written_count = 0
            for filtered_line in self.filter_lines(infile):
                outfile.write(filtered_line)
                written_count += 1
                
        logger.info(f"Wrote {written_count} lines to {output_path}")
        return written_count

# Example usage and common patterns
def create_default_filter() -> RegexLineFilter:
    """Create a filter with common unwanted content patterns."""
    exclude_patterns = [
        # Email addresses
        r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',
        
        # URLs
        r'https?://(?:[-\w.])+(?:[:\d]+)?(?:/(?:[\w/_.])*(?:\?(?:[\w&=%.])*)?(?:#(?:[\w.])*)?)?',
        
        # Profanity (example - in practice use comprehensive list)
        r'\b(fuck|shit|damn)\b',
        
        # Template artifacts
        r'\{\{.*?\}\}',  # Handlebars-style placeholders
        r'<%.*?%>',      # ERB-style placeholders
        r'\$\{.*?\}',    # Shell-style placeholders
        
        # Common spam patterns
        r'\b(?:casino|viagra|lottery)\b',
        
        # Excessive whitespace
        r'^\s*$'  # Empty or whitespace-only lines
    ]
    
    return RegexLineFilter(exclude_patterns=exclude_patterns)

# Main execution function
def main():
    """Main execution function demonstrating usage."""
    import argparse
    import sys
    
    parser = argparse.ArgumentParser(description="Filter unwanted lines from text files")
    parser.add_argument('input_file', help='Input file path')
    parser.add_argument('output_file', help='Output file path')
    parser.add_argument('--config', help='JSON config file with patterns')
    parser.add_argument('--encoding', default='utf-8', help='File encoding')
    parser.add_argument('--verbose', '-v', action='store_true', help='Verbose output')
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    try:
        # Initialize filter
        if args.config:
            filter_instance = RegexLineFilter.from_config_file(args.config)
        else:
            filter_instance = create_default_filter()
            logger.info("Using default filter patterns")
        
        # Process file
        lines_written = filter_instance.filter_file(
            args.input_file, 
            args.output_file, 
            args.encoding
        )
        
        print(f"Successfully filtered {args.input_file} -> {args.output_file}")
        print(f"Lines written: {lines_written}")
        
    except Exception as e:
        logger.error(f"Processing failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()