"""
Utility Functions

Helper functions for URL handling, validation, and common operations.
"""

import re
from urllib.parse import urlparse, urljoin, urlunparse
from typing import List, Optional
import fnmatch


def is_valid_url(url: str) -> bool:
    """Check if URL is valid and has http/https scheme."""
    try:
        result = urlparse(url)
        return all([result.scheme in ('http', 'https'), result.netloc])
    except:
        return False


def normalize_url(url: str) -> str:
    """Normalize URL by removing fragments and trailing slashes."""
    try:
        parsed = urlparse(url)
        path = parsed.path.rstrip('/') or '/'
        normalized = urlunparse((
            parsed.scheme.lower(),
            parsed.netloc.lower(),
            path,
            parsed.params,
            parsed.query,
            ''  # Remove fragment
        ))
        return normalized
    except:
        return url


def get_domain(url: str) -> str:
    """Extract domain from URL."""
    try:
        parsed = urlparse(url)
        return parsed.netloc.lower()
    except:
        return ""


def get_base_domain(url: str) -> str:
    """Extract base domain (without subdomain) from URL."""
    domain = get_domain(url)
    parts = domain.split('.')
    if len(parts) > 2:
        return '.'.join(parts[-2:])
    return domain


def should_exclude_url(url: str, patterns: List[str]) -> bool:
    """Check if URL matches any exclusion pattern."""
    for pattern in patterns:
        pattern = pattern.replace('**', '*')
        if fnmatch.fnmatch(url.lower(), pattern.lower()):
            return True
        if fnmatch.fnmatch(urlparse(url).path.lower(), pattern.lower()):
            return True
    return False


def should_include_url(url: str, patterns: List[str]) -> bool:
    """Check if URL matches any inclusion pattern."""
    if not patterns:
        return True
    for pattern in patterns:
        pattern = pattern.replace('**', '*')
        if fnmatch.fnmatch(url.lower(), pattern.lower()):
            return True
        if fnmatch.fnmatch(urlparse(url).path.lower(), pattern.lower()):
            return True
    return False


def clean_text(text: str) -> str:
    """Clean and normalize text content."""
    text = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]', '', text)
    text = re.sub(r'[ \t]+', ' ', text)
    text = re.sub(r'\n{3,}', '\n\n', text)
    return text.strip()


def truncate_text(text: str, max_length: int = 1000, suffix: str = "...") -> str:
    """Truncate text to max length, preferring word boundaries."""
    if len(text) <= max_length:
        return text
    truncated = text[:max_length - len(suffix)]
    last_space = truncated.rfind(' ')
    if last_space > max_length // 2:
        truncated = truncated[:last_space]
    return truncated + suffix


def generate_chunk_id(url: str, index: int) -> str:
    """Generate a unique chunk ID."""
    import hashlib
    url_hash = hashlib.md5(url.encode()).hexdigest()[:8]
    return f"{url_hash}_chunk_{index}"
