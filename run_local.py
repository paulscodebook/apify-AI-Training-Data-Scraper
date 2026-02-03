"""
Local development script for testing the actor.

Usage:
    python run_local.py [input_file.json]

If no input file is specified, uses sample_input.json
"""

import asyncio
import json
import sys
from pathlib import Path


async def run_actor(input_data: dict):
    """Run the actor with given input data."""
    # Add src to path
    src_path = Path(__file__).parent / "src"
    sys.path.insert(0, str(src_path.parent))
    
    from src.main import AITrainingDataScraper
    
    # Create a mock Actor for local testing
    class MockActor:
        @staticmethod
        def log_info(msg):
            print(f"[INFO] {msg}")
        
        @staticmethod
        def log_warning(msg):
            print(f"[WARN] {msg}")
        
        @staticmethod
        def log_error(msg):
            print(f"[ERROR] {msg}")
        
        @staticmethod
        def log_exception(msg):
            print(f"[EXCEPTION] {msg}")
        
        @staticmethod
        async def push_data(data):
            print(f"\n{'='*60}")
            print("DATA OUTPUT:")
            print(json.dumps(data, indent=2, default=str)[:2000])
            if len(json.dumps(data)) > 2000:
                print("... (truncated)")
            print(f"{'='*60}\n")
        
        @staticmethod
        async def get_input():
            return input_data
        
        @staticmethod
        async def set_value(key, value):
            print(f"[KV Store] Set {key}")
    
    # Patch the Actor
    import src.main as main_module
    
    # Create a mock log class
    class MockLog:
        @staticmethod
        def info(msg):
            print(f"[INFO] {msg}")
        
        @staticmethod
        def warning(msg):
            print(f"[WARN] {msg}")
        
        @staticmethod
        def error(msg):
            print(f"[ERROR] {msg}")
        
        @staticmethod
        def exception(msg):
            print(f"[EXCEPTION] {msg}")
        
        @staticmethod
        def debug(msg):
            print(f"[DEBUG] {msg}")
    
    # Create mock Actor with log attribute
    MockActor.log = MockLog
    
    # Run scraper
    print("\n" + "="*60)
    print("AI TRAINING DATA SCRAPER - LOCAL TEST")
    print("="*60 + "\n")
    
    print("Input configuration:")
    print(json.dumps(input_data, indent=2))
    print("\n")
    
    # Create scraper instance
    scraper = AITrainingDataScraper(input_data)
    
    # We can't fully run without Apify environment, but we can test components
    print("Testing content extraction and chunking components...\n")
    
    # Test chunking engine
    from src.chunking import ChunkingEngine
    
    test_text = """
    Python is a high-level, general-purpose programming language.
    Its design philosophy emphasizes code readability with the use of significant indentation.
    Python is dynamically typed and garbage-collected.
    It supports multiple programming paradigms, including structured, object-oriented and functional programming.
    
    Python was conceived in the late 1980s by Guido van Rossum at Centrum Wiskunde & Informatica (CWI).
    The language was released in 1991 and has since become one of the most popular programming languages.
    """
    
    engine = ChunkingEngine(
        strategy=input_data.get("chunkingStrategy", "semantic"),
        chunk_size=input_data.get("chunkSize", 512),
        overlap=input_data.get("chunkOverlap", 100)
    )
    
    chunks = engine.chunk(test_text, url="https://example.com/test", title="Python Overview")
    
    print(f"Generated {len(chunks)} chunks using '{input_data.get('chunkingStrategy', 'semantic')}' strategy:\n")
    
    for chunk in chunks:
        print(f"Chunk {chunk['chunk_index']}:")
        print(f"  Token count: {chunk['token_count']}")
        print(f"  Word count: {chunk['word_count']}")
        print(f"  Content preview: {chunk['content'][:100]}...")
        print()
    
    print("\n" + "="*60)
    print("LOCAL TEST COMPLETE")
    print("="*60)
    print("\nTo run the full actor, deploy to Apify and run with:")
    print("  apify run")
    print("\nOr use the Apify Console.")


def main():
    # Determine input file
    if len(sys.argv) > 1:
        input_file = Path(sys.argv[1])
    else:
        input_file = Path(__file__).parent / "sample_input.json"
    
    if not input_file.exists():
        print(f"Error: Input file not found: {input_file}")
        sys.exit(1)
    
    # Load input
    with open(input_file, "r") as f:
        input_data = json.load(f)
    
    # Run
    asyncio.run(run_actor(input_data))


if __name__ == "__main__":
    main()
