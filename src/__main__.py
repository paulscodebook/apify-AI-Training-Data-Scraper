"""
Main entry point for the AI Training Data Scraper Actor.
"""

import asyncio
import sys

# Surgical monkey-patch for browserforge/crawlee compatibility
# This MUST run before anything else is imported
try:
    import browserforge.download as bfd
    if not hasattr(bfd, 'DATA_FILES'):
        # Map DATA_DIRS to DATA_FILES if available, otherwise use a safe default
        bfd.DATA_FILES = getattr(bfd, 'DATA_DIRS', {'headers': []})
    
    # Ensure the 'headers' key exists as crawlee specifically checks for it
    if isinstance(bfd.DATA_FILES, dict) and 'headers' not in bfd.DATA_FILES:
        bfd.DATA_FILES['headers'] = []
    
    # Inject it into sys.modules to ensure all subsequent imports see the patch
    sys.modules['browserforge.download'] = bfd
except Exception:
    pass

from .main import main

if __name__ == "__main__":
    asyncio.run(main())
