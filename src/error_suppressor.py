# src/error_suppressor.py

import os
import sys
import contextlib
import io
import re
import logging

logger = logging.getLogger("cupcake.error_suppressor")

class ErrorSuppressor:
    """
    Utility class for suppressing and redirecting error messages.
    Particularly useful for intercepting libraw error messages.
    """
    
    # Known error patterns to suppress
    SUPPRESSION_PATTERNS = [
        r"File format not recognized",
        r"Cannot decode file",
        r"Unable to process file",
        r"PNG file does not have exif data",
    ]
    
    @staticmethod
    @contextlib.contextmanager
    def suppress_stderr():
        """
        Context manager to temporarily redirect stderr to suppress output.
        
        Usage:
            with ErrorSuppressor.suppress_stderr():
                # Code that produces unwanted stderr output
        """
        original_stderr = sys.stderr
        sys.stderr = io.StringIO()
        
        try:
            yield
        finally:
            # Restore stderr but collect what was captured
            captured = sys.stderr.getvalue()
            sys.stderr = original_stderr
            
            # Log any important errors that aren't in suppression patterns
            for line in captured.splitlines():
                if line.strip() and not any(re.search(pattern, line) for pattern in ErrorSuppressor.SUPPRESSION_PATTERNS):
                    logger.debug(f"Suppressed message: {line}")
    
    @staticmethod
    @contextlib.contextmanager
    def redirect_stderr(to_file=None):
        """
        Context manager to redirect stderr to a file or to a custom handler.
        
        Args:
            to_file: Path to a file for redirecting stderr, or None to discard
            
        Usage:
            with ErrorSuppressor.redirect_stderr("/path/to/errors.log"):
                # Code that produces stderr output to be redirected
        """
        original_stderr = sys.stderr
        
        if to_file:
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(os.path.abspath(to_file)), exist_ok=True)
            sys.stderr = open(to_file, 'a')
        else:
            # Discard output
            sys.stderr = io.StringIO()
        
        try:
            yield
        finally:
            # Restore stderr
            if to_file:
                sys.stderr.close()
            sys.stderr = original_stderr
    
    @staticmethod
    def filter_output(func):
        """
        Decorator to suppress stderr output from a function.
        
        Usage:
            @ErrorSuppressor.filter_output
            def function_with_unwanted_stderr():
                # Code that produces unwanted stderr output
        """
        def wrapper(*args, **kwargs):
            with ErrorSuppressor.suppress_stderr():
                return func(*args, **kwargs)
        return wrapper