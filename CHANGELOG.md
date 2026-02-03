# Changelog

All notable changes to AI Training Data Scraper will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.0.0] - 2026-02-02

### Added
- Initial release of AI Training Data Scraper
- 4 chunking strategies: fixed_token, sentence_based, semantic, markdown_section
- Dual crawler support: BeautifulSoup (fast) and Playwright (JS rendering)
- Rich metadata extraction (15+ fields)
- 4 output formats: markdown, plain_text, json_structured, vector_ready
- Vector database optimization for Pinecone, Qdrant, Weaviate, ChromaDB
- LangChain and LlamaIndex integration support
- Configurable content cleaning with CSS selectors
- URL pattern inclusion/exclusion filters
- Comprehensive error handling and logging
- Full documentation with integration examples

### Technical Details
- Built with Apify SDK 1.7+ and Crawlee 0.3+
- Uses tiktoken for accurate GPT-4 token counting
- Sentence transformers for semantic chunking
- NLTK for natural language processing
- Readability algorithm for main content extraction

---

## Future Roadmap

### [1.1.0] - Planned
- [ ] Built-in embedding generation (OpenAI, Cohere, Voyage)
- [ ] Direct vector database push (Pinecone, Qdrant)
- [ ] PDF and document file support
- [ ] Custom extraction rules (XPath, JSON-LD)

### [1.2.0] - Planned
- [ ] Multi-language optimization
- [ ] Table extraction and formatting
- [ ] Image alt-text extraction
- [ ] Sitemap-based crawling
