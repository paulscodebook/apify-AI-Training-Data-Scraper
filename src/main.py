"""
AI Training Data Scraper - Main Actor Module

This module orchestrates the web crawling and content extraction process,
producing clean, chunked data optimized for LLMs and vector databases.
"""

import asyncio
from datetime import datetime, timezone, timedelta
from urllib.parse import urlparse, urljoin
from typing import Optional, Dict, Any, List
import re

from apify import Actor
from crawlee.crawlers import BeautifulSoupCrawler, BeautifulSoupCrawlingContext
from crawlee.crawlers import PlaywrightCrawler, PlaywrightCrawlingContext
from crawlee.browsers import BrowserPool, PlaywrightBrowserPlugin

from .content_extractor import ContentExtractor
from .chunking import ChunkingEngine
from .metadata_extractor import MetadataExtractor
from .output_formatter import OutputFormatter
from .utils import (
    is_valid_url,
    normalize_url,
    should_exclude_url,
    should_include_url,
    get_domain,
    get_base_domain,
    generate_chunk_id,
)


class AITrainingDataScraper:
    """
    Main scraper class that orchestrates crawling and content extraction.
    """

    def __init__(self, actor_input: Dict[str, Any]):
        """Initialize the scraper with actor input configuration."""
        self.input = actor_input
        self.start_urls = actor_input.get("startUrls", [])
        self.crawler_type = actor_input.get("crawlerType", "cheerio")
        self.max_pages = actor_input.get("maxCrawlPages", 100)
        self.max_depth = actor_input.get("maxCrawlDepth", 20)
        self.chunking_strategy = actor_input.get("chunkingStrategy", "semantic")
        self.chunk_size = actor_input.get("chunkSize", 512)
        self.chunk_overlap = actor_input.get("chunkOverlap", 100)
        self.output_format = actor_input.get("outputFormat", "vector_ready")
        self.remove_elements = actor_input.get("removeElements", [
            "nav", "header", "footer", ".advertisement", 
            "#cookie-banner", ".sidebar"
        ])
        self.include_metadata = actor_input.get("includeMetadata", True)
        self.extract_links = actor_input.get("extractLinks", False)
        self.save_screenshots = actor_input.get("saveScreenshots", False)
        self.respect_robots = actor_input.get("respectRobotsTxt", True)
        self.max_concurrency = actor_input.get("maxConcurrency", 5)
        self.request_timeout = actor_input.get("requestTimeout", 30)
        self.proxy_config = actor_input.get("proxyConfiguration", {"useApifyProxy": True})
        self.url_patterns = actor_input.get("urlPatterns", [])
        self.exclude_patterns = actor_input.get("excludeUrlPatterns", [
            "**/login**", "**/signup**", "**/register**", 
            "**/cart**", "**/checkout**", "**/account**"
        ])

        # Initialize components
        self.content_extractor = ContentExtractor(self.remove_elements)
        self.chunking_engine = ChunkingEngine(
            strategy=self.chunking_strategy,
            chunk_size=self.chunk_size,
            overlap=self.chunk_overlap
        )
        self.metadata_extractor = MetadataExtractor()
        self.output_formatter = OutputFormatter(self.output_format)

        # Track crawled URLs and stats
        self.crawled_count = 0
        self.total_chunks = 0
        self.errors_count = 0
        self.start_domains = set()

        # Extract domains from start URLs
        for url_item in self.start_urls:
            url = url_item.get("url") if isinstance(url_item, dict) else url_item
            if url:
                self.start_domains.add(get_domain(url))

    def _validate_input(self):
        """Validate actor input configuration."""
        if self.max_pages <= 0:
            raise ValueError(f"maxCrawlPages must be positive, got {self.max_pages}")
        
        if self.max_depth <= 0:
            raise ValueError(f"maxCrawlDepth must be positive, got {self.max_depth}")
            
        if self.crawler_type not in ["cheerio", "playwright"]:
            raise ValueError(f"Invalid crawlerType: {self.crawler_type}. Must be 'cheerio' or 'playwright'")
            
        # Warning regarding crawler selection
        if self.crawler_type == "cheerio":
             Actor.log.warning("⚠️ Using 'cheerio' crawler. This may not work for SPA/JavaScript-heavy sites (e.g. React/Vue docs). Consider using 'playwright' if you see no links extracted.")

        # Clarify extractLinks
        if self.input.get("extractLinks") is not None:
             Actor.log.info(f"ℹ️ extractLinks set to {self.extract_links}. This controls whether discovered links are INCLUDED IN THE OUTPUT dataset. It does NOT control crawling behavior (use maxCrawlDepth for that).")

    async def run(self):
        """Run the scraper."""
        self._validate_input()
        
        Actor.log.info(f"🚀 Starting AI Training Data Scraper")
        Actor.log.info(f"📌 Start URLs: {len(self.start_urls)}")
        Actor.log.info(f"🔧 Crawler type: {self.crawler_type}")
        Actor.log.info(f"📄 Max pages: {self.max_pages}")
        Actor.log.info(f"🌊 Max depth: {self.max_depth}")
        Actor.log.info(f"📊 Chunking strategy: {self.chunking_strategy}")
        Actor.log.info(f"📏 Chunk size: {self.chunk_size}, Overlap: {self.chunk_overlap}")
        Actor.log.info(f"📦 Output format: {self.output_format}")
        Actor.log.info(f"🔗 Link Extraction in Output: {'Enabled' if self.extract_links else 'Disabled'}")

        # Build crawler based on type
        if self.crawler_type == "playwright":
            crawler = await self._build_playwright_crawler()
        else:
            crawler = await self._build_beautifulsoup_crawler()

        # Prepare start URLs
        start_requests = []
        for url_item in self.start_urls:
            url = url_item.get("url") if isinstance(url_item, dict) else url_item
            if url and is_valid_url(url):
                start_requests.append(url)
                Actor.log.info(f"📍 Added start URL: {url}")
            else:
                Actor.log.warning(f"⚠️ Invalid start URL skipped: {url_item}")

        if not start_requests:
            Actor.log.error("❌ No valid start URLs provided")
            return

        # Run the crawler
        await crawler.run(start_requests)

        # Log final stats
        Actor.log.info("=" * 60)
        Actor.log.info("✅ Crawl completed!")
        Actor.log.info(f"📊 Pages crawled: {self.crawled_count}")
        Actor.log.info(f"📦 Total chunks generated: {self.total_chunks}")
        Actor.log.info(f"❌ Errors: {self.errors_count}")
        Actor.log.info("=" * 60)
        
        if self.crawled_count <= 1 and self.max_pages > 1:
             Actor.log.warning("⚠️ Only 1 page was crawled despite maxCrawlPages > 1. This usually means no links were found or matched. \nCHECK: 1) Is the site an SPA? Use 'playwright'. \n2) Are there valid links on the page? \n3) Do your urlPatterns/excludeUrlPatterns filter everything?")

    async def _build_beautifulsoup_crawler(self) -> BeautifulSoupCrawler:
        """Build and configure BeautifulSoup crawler for static sites."""
        
        crawler = BeautifulSoupCrawler(
            max_requests_per_crawl=self.max_pages,
            max_request_retries=3,
            request_handler_timeout=timedelta(seconds=self.request_timeout),
            max_crawl_depth=self.max_depth,
        )

        @crawler.router.default_handler
        async def request_handler(context: BeautifulSoupCrawlingContext) -> None:
            await self._process_page(context)

        return crawler

    async def _build_playwright_crawler(self) -> PlaywrightCrawler:
        """Build and configure Playwright crawler for JavaScript-heavy sites."""
        
        # Configure browser plugin with sandbox flags
        plugin = PlaywrightBrowserPlugin(
            browser_type='chromium',
            browser_launch_options={
                'headless': True,
                'args': ['--no-sandbox', '--disable-setuid-sandbox']
            }
        )
        
        # Create browser pool with the plugin
        pool = BrowserPool(plugins=[plugin])

        crawler = PlaywrightCrawler(
            max_requests_per_crawl=self.max_pages,
            max_request_retries=3,
            request_handler_timeout=timedelta(seconds=self.request_timeout),
            max_crawl_depth=self.max_depth,
            browser_pool=pool,
        )

        @crawler.router.default_handler
        async def request_handler(context: PlaywrightCrawlingContext) -> None:
            # Get page content
            html = await context.page.content()
            
            # Save screenshot if enabled
            if self.save_screenshots:
                try:
                    screenshot = await context.page.screenshot()
                    key = f"screenshot_{self.crawled_count}.png"
                    await Actor.set_value(key, screenshot)
                except Exception as e:
                    Actor.log.warning(f"Failed to save screenshot: {e}")

            # Create a BeautifulSoup-like context for processing
            from bs4 import BeautifulSoup
            soup = BeautifulSoup(html, "lxml")
            
            await self._process_page_content(context.request, soup, context)

        return crawler

    async def _process_page(self, context: BeautifulSoupCrawlingContext) -> None:
        """Process a page from BeautifulSoup crawler."""
        try:
            await self._process_page_content(context.request, context.soup, context)
        except Exception as e:
            self.errors_count += 1
            Actor.log.exception(f"❌ Error processing {context.request.url}: {e}")
            
            # Push error record
            await Actor.push_data({
                "error": True,
                "url": context.request.url,
                "error_message": str(e),
                "error_type": type(e).__name__,
                "crawl_info": {
                    "crawled_at": datetime.now(timezone.utc).isoformat(),
                    "crawl_depth": getattr(context.request, 'user_data', {}).get('depth', 0)
                }
            })

    async def _process_page_content(self, request, soup, context) -> None:
        """Process page content and extract data."""
        url = request.url
        self.crawled_count += 1

        Actor.log.info(f"📄 [{self.crawled_count}/{self.max_pages}] Processing: {url}")

        # Extract main content (this now includes page classification and absolute link resolution)
        extracted = self.content_extractor.extract(soup, url)
        
        page_type = extracted.get("metadata", {}).get("page_type")
        link_density = extracted.get("metadata", {}).get("link_density", 0)
        
        # Enqueue more links EARLY to ensure multi-page crawling works even if extraction fails
        await self._enqueue_links(context, url, soup)

        if not extracted.get("content"):
            Actor.log.warning(f"⚠️ No content extracted from: {url}")
            return

        # Skip granular chunking for pure index/navigation pages to save tokens,
        # but ALWAYS produce at least one chunk if there is text (fixes Priority 2)
        is_index = page_type == "index_page" or link_density > 0.7
        
        if is_index:
             Actor.log.info(f"   ℹ️ Page classified as {page_type} (Link density: {link_density}). Using single-chunk fallback.")
             # Single chunk fallback for index pages
             chunks = [{
                "chunk_id": generate_chunk_id(url, 0),
                "chunk_index": 0,
                "content": extracted["markdown"] or extracted["content"],
                "token_count": len(self.chunking_engine.encoder.encode(extracted["markdown"] or extracted["content"])) if extracted.get("content") else 0,
                "word_count": len(extracted["content"].split()),
                "char_count": len(extracted["content"]),
                "embedding_ready": True,
                "chunk_metadata": {
                    "page_type": page_type,
                    "is_index": True
                }
             }]
        else:
            # Apply standard chunking
            chunks = self.chunking_engine.chunk(
                extracted["markdown"] or extracted["content"],
                url=url,
                title=extracted.get("title", "")
            )
        
        self.total_chunks += len(chunks)
        Actor.log.info(f"   📊 [{page_type}] Generated {len(chunks)} chunks")

        # Extract metadata if enabled
        metadata = {}
        if self.include_metadata:
            metadata = self.metadata_extractor.extract(soup, url, extracted)
            # Merge classification metadata
            metadata.update(extracted.get("metadata", {}))

        # Format output
        output = self.output_formatter.format(
            url=url,
            title=extracted.get("title", ""),
            content=extracted.get("content", ""),
            chunks=chunks,
            metadata=metadata,
            extract_links=self.extract_links,
            crawl_depth=getattr(request, 'user_data', {}).get('depth', 0)
        )
        
        output["page_type"] = page_type

        # Push to dataset
        await Actor.push_data(output)

    async def _enqueue_links(self, context, current_url: str, soup=None) -> None:
        """Enqueue discovered links for crawling with explicit depth and pattern control."""
        try:
            if hasattr(context, 'enqueue_links'):
                current_depth = getattr(context.request, 'user_data', {}).get('depth', 0)
                
                if current_depth >= self.max_depth:
                    Actor.log.debug(f"Reached max depth ({self.max_depth}) at {current_url}")
                    return

                Actor.log.info(f"🔍 Searching for links on {current_url} (Current depth: {current_depth})")
                
                # Use a more explicit enqueue strategy to avoid "same-domain" misses
                # and ensure our patterns are respected
                await context.enqueue_links(
                    selector='a[href]',
                    strategy='same-domain',
                    exclude=self.exclude_patterns,
                    transform_request_function=self._transform_request
                )
        except Exception as e:
            Actor.log.error(f"❌ Link extraction failed on {current_url}: {e}")

    def _transform_request(self, request):
        """Prepare request with correct user data (depth)."""
        # Note: Crawlee handles depth incrementing if we use enqueue_links correctly,
        # but we can explicitly manage it in user_data if needed.
        return request

    def _should_follow_link(self, url: str) -> bool:
        """Determine if a URL should be followed."""
        # Check if URL is valid
        if not is_valid_url(url):
            return False

        # Check if URL is within allowed domains
        url_domain = get_domain(url)
        if url_domain not in self.start_domains:
            # Check if it's a subdomain of a start domain
            base_domains = {get_base_domain(d) for d in self.start_domains}
            if get_base_domain(url) not in base_domains:
                return False

        # Check exclude patterns
        if should_exclude_url(url, self.exclude_patterns):
            return False
            
        # Check include patterns
        if self.url_patterns and not should_include_url(url, self.url_patterns):
            return False

        return True


async def main():
    """Main entry point for the actor."""
    async with Actor:
        # Get input
        actor_input = await Actor.get_input() or {}
        
        if not actor_input.get("startUrls"):
            Actor.log.error("❌ No start URLs provided. Please provide at least one URL.")
            raise ValueError("startUrls is required")

        # Create and run scraper
        scraper = AITrainingDataScraper(actor_input)
        
        try:
            await scraper.run()
        except Exception as e:
            Actor.log.exception(f"❌ Actor failed with unhandled exception: {e}")
            
        # Final Summary for visibility (Priority 3)
        if scraper.crawled_count == 0:
             Actor.log.error("❌ Crawl completed but 0 pages were processed. Check your startUrls and proxy settings.")
        elif scraper.total_chunks == 0:
             Actor.log.warning(f"⚠️ Crawled {scraper.crawled_count} pages but generated 0 chunks. \n"
                               "Possible causes: \n"
                               "1) removeElements is too broad (e.g. removing 'body') \n"
                               "2) Content classification skipped all pages as 'index' (check link_density) \n"
                               "3) Pages had no text content.")
        else:
             Actor.log.info(f"📊 Final Summary: {scraper.crawled_count} pages crawled, {scraper.total_chunks} chunks generated.")
