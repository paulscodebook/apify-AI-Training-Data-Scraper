"""
Main entry point for the AI Training Data Scraper Actor.
"""

import asyncio

# Monkey-patch browserforge for crawlee compatibility
try:
    import browserforge.download as bfd
    if not hasattr(bfd, 'DATA_FILES') and hasattr(bfd, 'DATA_DIRS'):
        bfd.DATA_FILES = bfd.DATA_DIRS
except ImportError:
    pass

from .main import main

if __name__ == "__main__":
    asyncio.run(main())
