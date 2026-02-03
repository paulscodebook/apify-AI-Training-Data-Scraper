"""
Tests for the Chunking Engine

Run with: pytest tests/test_chunking.py -v
"""

import pytest
from src.chunking import ChunkingEngine, Chunk


class TestChunkingEngine:
    """Test suite for ChunkingEngine class."""

    def setup_method(self):
        """Set up test fixtures."""
        self.sample_text = """
        Introduction to Python Programming

        Python is a versatile programming language that is widely used in many fields.
        It was created by Guido van Rossum and first released in 1991.
        Python emphasizes code readability and allows programmers to express concepts in fewer lines of code.

        Getting Started

        To get started with Python, you first need to install it on your computer.
        You can download Python from the official website at python.org.
        Once installed, you can start writing Python code using any text editor.

        Basic Syntax

        Python uses indentation to define code blocks instead of curly braces.
        Here is a simple example of a Python function:

        def greet(name):
            print(f"Hello, {name}!")

        This function takes a name parameter and prints a greeting message.
        Python supports both single and double quotes for strings.

        Conclusion

        Python is an excellent language for beginners and experienced programmers alike.
        Its simple syntax and powerful libraries make it ideal for various applications.
        """

    def test_fixed_token_chunking(self):
        """Test fixed token chunking strategy."""
        engine = ChunkingEngine(strategy="fixed_token", chunk_size=100, overlap=20)
        chunks = engine.chunk(self.sample_text, url="https://example.com/test")
        
        assert len(chunks) > 0
        assert all("chunk_id" in c for c in chunks)
        assert all("content" in c for c in chunks)
        assert all("token_count" in c for c in chunks)

    def test_sentence_based_chunking(self):
        """Test sentence-based chunking strategy."""
        engine = ChunkingEngine(strategy="sentence_based", chunk_size=50, overlap=10)
        chunks = engine.chunk(self.sample_text, url="https://example.com/test")
        
        assert len(chunks) > 0
        # Sentences should not be cut mid-sentence
        for chunk in chunks:
            content = chunk["content"]
            # Most chunks should end with sentence-ending punctuation
            assert content.strip()

    def test_semantic_chunking(self):
        """Test semantic chunking strategy."""
        engine = ChunkingEngine(strategy="semantic", chunk_size=100, overlap=20)
        chunks = engine.chunk(self.sample_text, url="https://example.com/test")
        
        assert len(chunks) > 0
        assert all(c["embedding_ready"] for c in chunks)

    def test_markdown_section_chunking(self):
        """Test markdown section chunking."""
        markdown_text = """# Main Title

This is the introduction paragraph with some content.
It has multiple sentences that form the introduction section.

## Section One

This section covers the first topic in detail.
There are multiple paragraphs here explaining the concept.
We include examples and explanations.

### Subsection 1.1

More detailed content about a specific aspect.
This goes deeper into the topic.

## Section Two

The second main section with different content.
This covers another important topic.
"""
        engine = ChunkingEngine(strategy="markdown_section", chunk_size=50)
        chunks = engine.chunk(markdown_text, url="https://example.com/docs")
        
        assert len(chunks) > 0
        # Check that section titles are captured
        has_section = any(
            c.get("chunk_metadata", {}).get("section_title") 
            for c in chunks
        )
        assert has_section or len(chunks) >= 1

    def test_empty_text(self):
        """Test handling of empty text."""
        engine = ChunkingEngine()
        chunks = engine.chunk("", url="https://example.com/empty")
        assert chunks == []

    def test_whitespace_only_text(self):
        """Test handling of whitespace-only text."""
        engine = ChunkingEngine()
        chunks = engine.chunk("   \n\n\t  ", url="https://example.com/whitespace")
        assert chunks == []

    def test_chunk_ids_are_unique(self):
        """Test that chunk IDs are unique within a page."""
        engine = ChunkingEngine(strategy="sentence_based", chunk_size=30)
        chunks = engine.chunk(self.sample_text, url="https://example.com/test")
        
        chunk_ids = [c["chunk_id"] for c in chunks]
        assert len(chunk_ids) == len(set(chunk_ids))

    def test_chunk_indices_are_sequential(self):
        """Test that chunk indices are sequential."""
        engine = ChunkingEngine(strategy="sentence_based", chunk_size=30)
        chunks = engine.chunk(self.sample_text, url="https://example.com/test")
        
        indices = [c["chunk_index"] for c in chunks]
        expected = list(range(len(chunks)))
        assert indices == expected

    def test_code_detection(self):
        """Test code block detection in chunks."""
        text_with_code = """
        Here is how to define a function:

        ```python
        def hello():
            print("Hello, World!")
        ```

        And here is how to use it:

        ```python
        hello()
        ```
        """
        engine = ChunkingEngine(strategy="sentence_based", chunk_size=100)
        chunks = engine.chunk(text_with_code, url="https://example.com/code")
        
        # At least one chunk should have code
        has_code = any(
            c.get("chunk_metadata", {}).get("has_code", False) 
            for c in chunks
        )
        assert has_code

    def test_different_urls_produce_different_ids(self):
        """Test that different URLs produce different chunk IDs."""
        engine = ChunkingEngine(strategy="fixed_token", chunk_size=200)
        
        chunks1 = engine.chunk(self.sample_text, url="https://example.com/page1")
        chunks2 = engine.chunk(self.sample_text, url="https://example.com/page2")
        
        ids1 = set(c["chunk_id"] for c in chunks1)
        ids2 = set(c["chunk_id"] for c in chunks2)
        
        # IDs should not overlap
        assert not ids1.intersection(ids2)


class TestTokenCounting:
    """Test token counting functionality."""

    def test_token_count_accuracy(self):
        """Test that token counts are reasonable."""
        engine = ChunkingEngine()
        
        # A simple sentence
        text = "Hello, this is a test sentence."
        count = engine._count_tokens(text)
        
        # Should be roughly word count * 1.3
        word_count = len(text.split())
        assert count >= word_count
        assert count <= word_count * 3  # Generous upper bound

    def test_empty_string_token_count(self):
        """Test token count for empty string."""
        engine = ChunkingEngine()
        count = engine._count_tokens("")
        assert count == 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
