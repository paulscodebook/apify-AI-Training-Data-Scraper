"""
Tests for the Content Extractor

Run with: pytest tests/test_content_extractor.py -v
"""

import pytest
from bs4 import BeautifulSoup
from src.content_extractor import ContentExtractor


class TestContentExtractor:
    """Test suite for ContentExtractor class."""

    def setup_method(self):
        """Set up test fixtures."""
        self.extractor = ContentExtractor()
        
        self.sample_html = """
        <!DOCTYPE html>
        <html lang="en">
        <head>
            <title>Test Page - Site Name</title>
            <meta name="description" content="This is a test page">
            <meta name="author" content="Test Author">
        </head>
        <body>
            <nav>
                <a href="/">Home</a>
                <a href="/about">About</a>
            </nav>
            <header>
                <h1>Main Header</h1>
            </header>
            <main>
                <article>
                    <h1>Article Title</h1>
                    <p>This is the first paragraph with some content.</p>
                    <h2>Section One</h2>
                    <p>More content in section one.</p>
                    <pre><code class="language-python">def hello():
    print("Hello, World!")</code></pre>
                    <h2>Section Two</h2>
                    <p>Content in section two.</p>
                    <ul>
                        <li>Item one</li>
                        <li>Item two</li>
                        <li>Item three</li>
                    </ul>
                </article>
            </main>
            <footer>
                <p>Copyright 2026</p>
            </footer>
        </body>
        </html>
        """

    def test_extract_basic_content(self):
        """Test basic content extraction."""
        soup = BeautifulSoup(self.sample_html, "lxml")
        result = self.extractor.extract(soup, "https://example.com/test")
        
        assert "content" in result
        assert "title" in result
        assert len(result["content"]) > 0

    def test_extract_title(self):
        """Test title extraction."""
        soup = BeautifulSoup(self.sample_html, "lxml")
        result = self.extractor.extract(soup, "https://example.com/test")
        
        # Should extract title without site suffix
        assert result["title"]
        assert "Test" in result["title"] or "Article" in result["title"]

    def test_remove_navigation(self):
        """Test that navigation is removed."""
        soup = BeautifulSoup(self.sample_html, "lxml")
        result = self.extractor.extract(soup, "https://example.com/test")
        
        content = result["content"].lower()
        # Nav links should be removed
        assert "home" not in content or "about" not in content

    def test_remove_footer(self):
        """Test that footer is removed."""
        soup = BeautifulSoup(self.sample_html, "lxml")
        result = self.extractor.extract(soup, "https://example.com/test")
        
        content = result["content"].lower()
        assert "copyright" not in content

    def test_extract_headings(self):
        """Test heading extraction."""
        soup = BeautifulSoup(self.sample_html, "lxml")
        result = self.extractor.extract(soup, "https://example.com/test")
        
        assert "headings" in result
        headings = result["headings"]
        
        # Should have some headings
        assert len(headings) > 0
        
        # Each heading should have level and text
        for h in headings:
            assert "level" in h
            assert "text" in h

    def test_extract_code_blocks(self):
        """Test code block extraction."""
        soup = BeautifulSoup(self.sample_html, "lxml")
        result = self.extractor.extract(soup, "https://example.com/test")
        
        assert "code_blocks" in result
        assert result["has_code"] == True
        
        code_blocks = result["code_blocks"]
        assert len(code_blocks) > 0
        
        # Should detect Python language
        languages = [b.get("language", "") for b in code_blocks]
        assert "python" in languages or any("python" in l for l in languages)

    def test_extract_lists(self):
        """Test list extraction."""
        soup = BeautifulSoup(self.sample_html, "lxml")
        result = self.extractor.extract(soup, "https://example.com/test")
        
        assert "lists" in result
        lists = result["lists"]
        
        assert len(lists) > 0
        
        # Should find the unordered list
        ul = next((l for l in lists if l["type"] == "unordered"), None)
        assert ul is not None
        assert len(ul["items"]) == 3

    def test_custom_remove_selectors(self):
        """Test custom element removal."""
        html = """
        <html>
        <body>
            <div class="advertisement">Buy now!</div>
            <main>
                <p>Main content here.</p>
            </main>
            <aside class="sidebar">Sidebar content</aside>
        </body>
        </html>
        """
        
        extractor = ContentExtractor(remove_selectors=[".advertisement", ".sidebar"])
        soup = BeautifulSoup(html, "lxml")
        result = extractor.extract(soup, "https://example.com/test")
        
        content = result["content"].lower()
        assert "buy now" not in content
        assert "sidebar content" not in content
        assert "main content" in content

    def test_markdown_conversion(self):
        """Test markdown conversion."""
        soup = BeautifulSoup(self.sample_html, "lxml")
        result = self.extractor.extract(soup, "https://example.com/test")
        
        assert "markdown" in result
        markdown = result["markdown"]
        
        # Should have markdown formatting
        assert len(markdown) > 0

    def test_empty_page(self):
        """Test handling of empty page."""
        html = "<html><body></body></html>"
        soup = BeautifulSoup(html, "lxml")
        result = self.extractor.extract(soup, "https://example.com/empty")
        
        assert result["content"] == "" or result["content"] is None or len(result["content"]) < 10

    def test_title_from_og_tag(self):
        """Test title extraction from Open Graph tag."""
        html = """
        <html>
        <head>
            <meta property="og:title" content="OG Title">
            <title>Regular Title</title>
        </head>
        <body><p>Content</p></body>
        </html>
        """
        
        soup = BeautifulSoup(html, "lxml")
        result = self.extractor.extract(soup, "https://example.com/test")
        
        assert result["title"] == "OG Title"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
