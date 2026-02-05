"""
Content Extractor Module

Handles extraction of main content from web pages, removing navigation,
ads, and boilerplate while preserving semantic structure.
"""

import re
from typing import Dict, Any, List, Optional
from bs4 import BeautifulSoup, Tag, NavigableString
import html2text
from readability import Document


class ContentExtractor:
    """
    Extracts and cleans main content from HTML pages.
    Uses readability algorithms and custom cleaning rules.
    """

    def __init__(self, remove_selectors: Optional[List[str]] = None):
        """
        Initialize content extractor.
        
        Args:
            remove_selectors: CSS selectors for elements to remove
        """
        self.remove_selectors = remove_selectors or [
            "nav", "header", "footer", ".advertisement", 
            "#cookie-banner", ".sidebar", ".nav", ".menu",
            ".ads", ".social-share", ".comments", ".related-posts",
            "script", "style", "noscript", "iframe", "form",
            ".breadcrumb", ".pagination", ".newsletter-signup"
        ]
        
        # Initialize html2text converter
        self.html_converter = html2text.HTML2Text()
        self.html_converter.ignore_links = False
        self.html_converter.ignore_images = True
        self.html_converter.ignore_emphasis = False
        self.html_converter.body_width = 0  # Don't wrap lines
        self.html_converter.unicode_snob = True
        self.html_converter.skip_internal_links = True
        self.html_converter.fenced_code = True

    def extract(self, soup: BeautifulSoup, url: str) -> Dict[str, Any]:
        """
        Extract clean main content from a BeautifulSoup object.
        
        Args:
            soup: BeautifulSoup object of the page
            url: The page URL
            
        Returns:
            Dictionary with extracted content and structure
        """
        # Get a copy to avoid modifying original
        soup_copy = BeautifulSoup(str(soup), "lxml")
        
        # Resolve all relative links to absolute URLs first
        self._resolve_relative_links(soup_copy, url)
        
        # Extract title first
        title = self._extract_title(soup_copy)
        
        # Calculate link density before removing elements to detect "index" pages
        link_density, total_links = self._calculate_link_density(soup_copy)
        
        # Remove unwanted elements
        self._remove_elements(soup_copy)
        
        # Try to find main content container
        main_content = self._find_main_content(soup_copy)
        
        # If no main content found, use readability
        if not main_content or len(main_content.get_text(strip=True)) < 100:
            main_content = self._use_readability(str(soup), url)
        
        if not main_content:
            return {
                "title": title,
                "content": "",
                "markdown": "",
                "headings": [],
                "code_blocks": [],
                "lists": [],
                "has_code": False,
                "metadata": {
                    "link_density": link_density,
                    "page_type": "empty"
                }
            }
        
        # Extract structured elements
        headings = self._extract_headings(main_content)
        code_blocks = self._extract_code_blocks(main_content)
        lists = self._extract_lists(main_content)
        
        # Convert to markdown
        markdown = self._convert_to_markdown(main_content)
        
        # Get plain text
        plain_text = self._get_plain_text(main_content)
        
        # Detect page type
        page_type = self._classify_page(url, title, plain_text, link_density, total_links)
        
        return {
            "title": title,
            "content": plain_text,
            "markdown": markdown,
            "html": str(main_content),
            "headings": headings,
            "code_blocks": code_blocks,
            "lists": lists,
            "has_code": len(code_blocks) > 0,
            "metadata": {
                "link_density": round(link_density, 3),
                "total_links": total_links,
                "page_type": page_type
            }
        }

    def _resolve_relative_links(self, soup: BeautifulSoup, base_url: str) -> None:
        """Resolve all relative links in the soup to absolute URLs."""
        from urllib.parse import urljoin
        for a in soup.find_all('a', href=True):
            a['href'] = urljoin(base_url, a['href'])
        for img in soup.find_all('img', src=True):
            img['src'] = urljoin(base_url, img['src'])

    def _calculate_link_density(self, soup: BeautifulSoup) -> tuple[float, int]:
        """Calculate the ratio of link text to total text."""
        all_text = soup.get_text()
        if not all_text.strip():
            return 0.0, 0
        
        link_text = ""
        total_links = 0
        for a in soup.find_all('a'):
            link_text += a.get_text()
            total_links += 1
            
        ratio = len(link_text) / len(all_text) if len(all_text) > 0 else 0
        return ratio, total_links

    def _classify_page(self, url: str, title: str, text: str, density: float, link_count: int) -> str:
        """Classify the page type based on content markers."""
        url_lower = url.lower()
        title_lower = title.lower()
        
        # Index/Navigation detection
        if density > 0.5 and link_count > 20:
             return "index_page"
        if any(x in url_lower for x in ["/index", "/toc", "/contents"]) or "table of contents" in title_lower:
             if density > 0.3:
                 return "index_page"
        
        # Documentation detection
        if any(x in url_lower for x in ["/docs", "/documentation", "/guide"]):
             return "documentation"
             
        # API Reference detection
        if any(x in url_lower for x in ["/api", "/reference", "api-docs"]):
             return "api_reference"
             
        # Tutorial detection
        if any(x in url_lower for x in ["/tutorial", "/course", "/learn"]):
             return "tutorial"
             
        # Blog detection
        if any(x in url_lower for x in ["/blog", "/posts"]):
             return "blog_post"
             
        # Landing page detection (heuristic)
        if len(text.split()) < 300 and link_count < 15 and ("welcome" in title_lower or "home" in title_lower):
             return "landing_page"
             
        return "general_content"

    def _extract_title(self, soup: BeautifulSoup) -> str:
        """Extract page title from various sources."""
        # Try og:title first
        og_title = soup.find("meta", attrs={"property": "og:title"})
        if og_title and og_title.get("content"):
            return og_title["content"].strip()
        
        # Try title tag
        title_tag = soup.find("title")
        if title_tag:
            title = title_tag.get_text(strip=True)
            # Clean common suffixes
            title = re.sub(r'\s*[\|\-–—]\s*[^|\-–—]+$', '', title)
            return title.strip()
        
        # Try h1
        h1 = soup.find("h1")
        if h1:
            return h1.get_text(strip=True)
        
        return ""

    def _remove_elements(self, soup: BeautifulSoup) -> None:
        """Remove unwanted elements from the soup."""
        for selector in self.remove_selectors:
            try:
                for element in soup.select(selector):
                    element.decompose()
            except Exception:
                pass  # Skip invalid selectors
        
        # Remove empty elements
        for tag in soup.find_all():
            if len(tag.get_text(strip=True)) == 0 and tag.name not in ['br', 'hr', 'img']:
                # Check if it has no children
                if not tag.find_all():
                    tag.decompose()

    def _find_main_content(self, soup: BeautifulSoup) -> Optional[Tag]:
        """Find the main content container using common patterns."""
        # Try standard content containers
        content_selectors = [
            "main",
            "article",
            "[role='main']",
            "#main-content",
            "#content",
            ".content",
            ".post-content",
            ".article-content",
            ".entry-content",
            ".page-content",
            ".documentation",
            ".docs-content",
            ".markdown-body",
            ".prose"
        ]
        
        for selector in content_selectors:
            try:
                element = soup.select_one(selector)
                if element and len(element.get_text(strip=True)) > 100:
                    return element
            except Exception:
                continue
        
        # Fallback: find the largest text container
        return self._find_largest_text_container(soup)

    def _find_largest_text_container(self, soup: BeautifulSoup) -> Optional[Tag]:
        """Find the container with the most text content."""
        candidates = []
        
        for tag in soup.find_all(['div', 'section', 'article', 'main']):
            text = tag.get_text(strip=True)
            if len(text) > 200:
                # Score by text length and number of paragraphs
                paragraphs = len(tag.find_all('p'))
                score = len(text) + (paragraphs * 100)
                candidates.append((tag, score))
        
        if candidates:
            candidates.sort(key=lambda x: x[1], reverse=True)
            return candidates[0][0]
        
        return soup.body if soup.body else soup

    def _use_readability(self, html: str, url: str) -> Optional[Tag]:
        """Use readability algorithm to extract main content."""
        try:
            doc = Document(html)
            content_html = doc.summary()
            return BeautifulSoup(content_html, "lxml").body
        except Exception:
            return None

    def _extract_headings(self, content: Tag) -> List[Dict[str, Any]]:
        """Extract all headings with their hierarchy."""
        headings = []
        
        for level in range(1, 7):
            for h in content.find_all(f'h{level}'):
                text = h.get_text(strip=True)
                if text:
                    headings.append({
                        "level": level,
                        "text": text,
                        "id": h.get("id", "")
                    })
        
        return headings

    def _extract_code_blocks(self, content: Tag) -> List[Dict[str, Any]]:
        """Extract code blocks with language detection."""
        code_blocks = []
        
        # Find <pre><code> blocks
        for pre in content.find_all('pre'):
            code = pre.find('code')
            if code:
                code_text = code.get_text()
                language = self._detect_code_language(code, pre)
            else:
                code_text = pre.get_text()
                language = self._detect_code_language(pre, pre)
            
            if code_text.strip():
                code_blocks.append({
                    "code": code_text.strip(),
                    "language": language
                })
        
        # Find inline code (but only if substantial)
        for code in content.find_all('code'):
            if code.parent and code.parent.name != 'pre':
                # Skip inline code for now, only track block code
                pass
        
        return code_blocks

    def _detect_code_language(self, element: Tag, parent: Tag) -> str:
        """Detect programming language from code block."""
        # Check class names for language hints
        classes = element.get('class', []) + parent.get('class', [])
        
        for cls in classes:
            if isinstance(cls, str):
                # Common patterns: language-python, lang-js, highlight-python
                match = re.match(r'(?:language-|lang-|highlight-)?(\w+)', cls)
                if match:
                    lang = match.group(1).lower()
                    if lang in ['python', 'javascript', 'js', 'typescript', 'ts',
                               'java', 'c', 'cpp', 'csharp', 'go', 'rust',
                               'ruby', 'php', 'swift', 'kotlin', 'scala',
                               'bash', 'shell', 'sql', 'html', 'css', 'json',
                               'yaml', 'xml', 'markdown', 'md']:
                        return lang
        
        return "unknown"

    def _extract_lists(self, content: Tag) -> List[Dict[str, Any]]:
        """Extract list items from the content."""
        lists = []
        
        for list_tag in content.find_all(['ul', 'ol']):
            items = []
            for li in list_tag.find_all('li', recursive=False):
                text = li.get_text(strip=True)
                if text:
                    items.append(text)
            
            if items:
                lists.append({
                    "type": "ordered" if list_tag.name == 'ol' else "unordered",
                    "items": items
                })
        
        return lists

    def _convert_to_markdown(self, content: Tag) -> str:
        """Convert HTML content to Markdown."""
        try:
            markdown = self.html_converter.handle(str(content))
            # Clean up excessive newlines
            markdown = re.sub(r'\n{3,}', '\n\n', markdown)
            return markdown.strip()
        except Exception:
            return self._get_plain_text(content)

    def _get_plain_text(self, content: Tag) -> str:
        """Get clean plain text from content."""
        # Get text
        text = content.get_text(separator='\n')
        
        # Clean up
        lines = []
        for line in text.split('\n'):
            line = line.strip()
            if line:
                lines.append(line)
        
        text = '\n'.join(lines)
        
        # Remove excessive whitespace
        text = re.sub(r'[ \t]+', ' ', text)
        text = re.sub(r'\n{3,}', '\n\n', text)
        
        return text.strip()
