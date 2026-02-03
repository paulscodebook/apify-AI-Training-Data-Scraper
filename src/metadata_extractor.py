"""
Metadata Extractor Module

Extracts comprehensive metadata from web pages for better retrieval and filtering.
"""

import re
from datetime import datetime
from typing import Dict, Any, Optional, List
from bs4 import BeautifulSoup


class MetadataExtractor:
    """Extracts rich metadata from web pages."""

    def extract(self, soup: BeautifulSoup, url: str, content: Dict[str, Any]) -> Dict[str, Any]:
        """Extract comprehensive metadata from page."""
        metadata = {
            "url": url,
            "title": self._extract_title(soup, content),
            "description": self._extract_description(soup),
            "author": self._extract_author(soup),
            "published_date": self._extract_date(soup, "published"),
            "modified_date": self._extract_date(soup, "modified"),
            "language": self._detect_language(soup, content),
            "keywords": self._extract_keywords(soup),
            "word_count": len(content.get("content", "").split()),
            "estimated_reading_time": max(1, len(content.get("content", "").split()) // 250),
            "content_type": self._detect_content_type(url, soup),
            "has_code_blocks": content.get("has_code", False),
            "heading_structure": [h["level"] for h in content.get("headings", [])],
            "heading_count": len(content.get("headings", [])),
            "code_languages": self._get_code_languages(content),
            "og_image": self._extract_og_image(soup),
            "canonical_url": self._extract_canonical(soup, url),
        }
        return {k: v for k, v in metadata.items() if v is not None}

    def _extract_title(self, soup: BeautifulSoup, content: Dict) -> str:
        if content.get("title"):
            return content["title"]
        og = soup.find("meta", attrs={"property": "og:title"})
        if og and og.get("content"):
            return og["content"].strip()
        title = soup.find("title")
        return title.get_text(strip=True) if title else ""

    def _extract_description(self, soup: BeautifulSoup) -> Optional[str]:
        for prop in ["og:description", "description", "twitter:description"]:
            meta = soup.find("meta", attrs={"property": prop}) or soup.find("meta", attrs={"name": prop})
            if meta and meta.get("content"):
                return meta["content"].strip()[:500]
        return None

    def _extract_author(self, soup: BeautifulSoup) -> Optional[str]:
        for prop in ["author", "article:author", "twitter:creator"]:
            meta = soup.find("meta", attrs={"name": prop}) or soup.find("meta", attrs={"property": prop})
            if meta and meta.get("content"):
                return meta["content"].strip()
        author_elem = soup.find(class_=re.compile(r"author|byline", re.I))
        if author_elem:
            return author_elem.get_text(strip=True)[:100]
        return None

    def _extract_date(self, soup: BeautifulSoup, date_type: str) -> Optional[str]:
        props = {
            "published": ["article:published_time", "datePublished", "date", "pubdate"],
            "modified": ["article:modified_time", "dateModified", "lastmod"],
        }
        for prop in props.get(date_type, []):
            meta = soup.find("meta", attrs={"property": prop}) or soup.find("meta", attrs={"name": prop})
            if meta and meta.get("content"):
                return self._parse_date(meta["content"])
        time_elem = soup.find("time", attrs={"datetime": True})
        if time_elem:
            return self._parse_date(time_elem["datetime"])
        return None

    def _parse_date(self, date_str: str) -> Optional[str]:
        try:
            from dateutil import parser
            return parser.parse(date_str).isoformat()
        except:
            return date_str[:25] if date_str else None

    def _detect_language(self, soup: BeautifulSoup, content: Dict) -> str:
        html = soup.find("html")
        if html and html.get("lang"):
            return html["lang"][:5]
        try:
            from langdetect import detect
            text = content.get("content", "")[:1000]
            if text:
                return detect(text)
        except:
            pass
        return "en"

    def _extract_keywords(self, soup: BeautifulSoup) -> List[str]:
        meta = soup.find("meta", attrs={"name": "keywords"})
        if meta and meta.get("content"):
            return [k.strip().lower() for k in meta["content"].split(",") if k.strip()][:20]
        return []

    def _detect_content_type(self, url: str, soup: BeautifulSoup) -> str:
        url_lower = url.lower()
        patterns = [
            (r'/docs?/', "documentation"), (r'/api/', "api_reference"),
            (r'/blog/', "blog_post"), (r'/tutorial', "tutorial"),
            (r'/guide', "guide"), (r'/help/', "help_article"),
            (r'/forum/', "forum_post"), (r'/faq', "faq"),
            (r'/reference/', "reference"), (r'/example', "example"),
        ]
        for pattern, content_type in patterns:
            if re.search(pattern, url_lower):
                return content_type
        og_type = soup.find("meta", attrs={"property": "og:type"})
        if og_type and og_type.get("content"):
            return og_type["content"]
        return "general"

    def _get_code_languages(self, content: Dict) -> List[str]:
        blocks = content.get("code_blocks", [])
        langs = set()
        for b in blocks:
            lang = b.get("language", "").lower()
            if lang and lang != "unknown":
                langs.add(lang)
        return list(langs)[:10]

    def _extract_og_image(self, soup: BeautifulSoup) -> Optional[str]:
        og = soup.find("meta", attrs={"property": "og:image"})
        return og["content"] if og and og.get("content") else None

    def _extract_canonical(self, soup: BeautifulSoup, url: str) -> str:
        link = soup.find("link", attrs={"rel": "canonical"})
        return link["href"] if link and link.get("href") else url
