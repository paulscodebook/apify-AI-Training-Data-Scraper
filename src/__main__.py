"""
Main entry point for the AI Training Data Scraper Actor.
"""

import asyncio
from .main import main

if __name__ == "__main__":
    asyncio.run(main())
