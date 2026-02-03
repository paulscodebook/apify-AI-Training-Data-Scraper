"""
Main entry point for the AI Training Data Scraper Actor.
"""

import asyncio
import sys

# Surgical monkey-patch for browserforge/crawlee compatibility
# This MUST run before anything else is imported
try:
    import browserforge.download as bfd
    from pathlib import Path
    
    # Ensure DATA_FILES exists and has the required keys
    if not hasattr(bfd, 'DATA_FILES'):
        bfd.DATA_FILES = {}
    
    # Ensure DATA_DIRS exists and has the required keys
    if not hasattr(bfd, 'DATA_DIRS'):
        bfd.DATA_DIRS = {}

    # Setup safe defaults for keys crawlee expects
    for d in [bfd.DATA_FILES, bfd.DATA_DIRS]:
        if 'headers' not in d:
            d['headers'] = [] if d is bfd.DATA_FILES else Path("/tmp")
        if 'fingerprints' not in d:
            # For DATA_FILES, it needs to be a dict-like object to support [path.name]
            class MockDict(dict):
                def __getitem__(self, key):
                    return super().get(key, key)
            d['fingerprints'] = MockDict() if d is bfd.DATA_FILES else Path("/tmp")
    
    # Inject it into sys.modules
    sys.modules['browserforge.download'] = bfd
except Exception:
    pass

from .main import main

if __name__ == "__main__":
    asyncio.run(main())
