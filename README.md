# 🤖 AI Training Data Scraper

[![Apify Actor](https://img.shields.io/badge/Apify-Actor-blue)](https://apify.com)
[![Python 3.11](https://img.shields.io/badge/Python-3.11-green)](https://python.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**Extract clean, semantically-chunked content from websites optimized for LLMs, RAG pipelines, and vector databases.**

Transform raw web content into AI-ready training data with intelligent chunking, rich metadata extraction, and vector database optimization.

---

## ✨ Key Features

- 🎯 **4 Smart Chunking Strategies**: Fixed token, sentence-based, semantic, and markdown section
- 🧹 **Intelligent Content Cleaning**: Removes navigation, ads, and boilerplate automatically
- 📊 **Rich Metadata Extraction**: Author, dates, keywords, language detection, content type
- 🔗 **Deep Recursive Crawling**: Crawl entire documentation sites with configurable depth
- ⚡ **Vector Database Ready**: Output formatted for Pinecone, Qdrant, Weaviate, ChromaDB
- 🦜 **LangChain & LlamaIndex Compatible**: Direct integration with popular AI frameworks
- 🎨 **Multiple Output Formats**: Markdown, plain text, JSON structured, vector-ready
- 🚀 **Dual Crawler Support**: Fast HTTP for static sites, Playwright for JS-heavy sites
- 🔒 **Respectful Crawling**: Respects robots.txt, configurable rate limiting

---

## 📋 Use Cases

| Use Case | Description |
|----------|-------------|
| **RAG Applications** | Build accurate retrieval-augmented generation systems with clean documentation |
| **AI Chatbots** | Train domain-specific chatbots on your knowledge base |
| **Code Assistants** | Extract technical documentation for programming assistants |
| **LLM Fine-tuning** | Collect high-quality training data for domain-specific models |
| **Semantic Search** | Populate vector databases for intelligent search systems |
| **Knowledge Management** | Structure and organize documentation for AI consumption |

---

## 🚀 Quick Start

### 1. Basic Usage

```json
{
  "startUrls": [{"url": "https://docs.python.org/3/"}],
  "crawlerType": "cheerio",
  "maxCrawlPages": 100,
  "chunkingStrategy": "semantic",
  "outputFormat": "vector_ready"
}
```

### 2. Advanced Configuration

```json
{
  "startUrls": [
    {"url": "https://docs.example.com/"},
    {"url": "https://api.example.com/docs"}
  ],
  "crawlerType": "playwright",
  "maxCrawlPages": 500,
  "maxCrawlDepth": 10,
  "chunkingStrategy": "semantic",
  "chunkSize": 512,
  "chunkOverlap": 100,
  "outputFormat": "vector_ready",
  "removeElements": ["nav", "header", "footer", ".sidebar", ".ads"],
  "includeMetadata": true,
  "proxyConfiguration": {"useApifyProxy": true}
}
```

---

## 📦 Chunking Strategies

Choose the right chunking strategy for your use case:

### 1. Fixed Token (`fixed_token`)
- **Best for**: Consistent chunk sizes for embedding models with token limits
- **How it works**: Splits content into fixed-size token chunks (default: 512 tokens)
- **Use when**: You need precise control over chunk sizes for OpenAI/Anthropic embeddings

### 2. Sentence-Based (`sentence_based`)
- **Best for**: Preserving natural language boundaries
- **How it works**: Groups complete sentences until reaching target size
- **Use when**: You want readable chunks that never cut mid-sentence

### 3. Semantic (`semantic`) ⭐ Recommended
- **Best for**: Optimal RAG performance
- **How it works**: Uses NLP to detect topic boundaries and group related content
- **Use when**: Building RAG systems where context preservation is critical

### 4. Markdown Section (`markdown_section`)
- **Best for**: Documentation and structured content
- **How it works**: Splits by heading hierarchy (## Section, ### Subsection)
- **Use when**: Scraping markdown-based documentation or wikis

---

## 📊 Output Schema

Each extracted page produces structured output:

```json
{
  "url": "https://docs.example.com/guide",
  "title": "Getting Started Guide",
  "content_format": "vector_ready",
  "full_content": "Complete page text...",
  "chunks": [
    {
      "id": "a1b2c3d4_chunk_0",
      "text": "Introduction to the framework...",
      "metadata": {
        "source_url": "https://docs.example.com/guide",
        "page_title": "Getting Started Guide",
        "chunk_index": 0,
        "token_count": 487,
        "has_code": true,
        "section_title": "Introduction",
        "language": "en",
        "content_type": "documentation"
      }
    }
  ],
  "metadata": {
    "author": "John Doe",
    "published_date": "2025-01-15T00:00:00Z",
    "language": "en",
    "keywords": ["python", "tutorial", "getting-started"],
    "word_count": 2500,
    "estimated_reading_time": 10,
    "content_type": "documentation",
    "has_code_blocks": true
  },
  "embedding_info": {
    "chunk_count": 8,
    "total_tokens": 3200,
    "ready_for_embedding": true,
    "recommended_model": "text-embedding-3-small"
  },
  "crawl_info": {
    "crawled_at": "2026-02-02T12:00:00Z",
    "crawl_depth": 2
  }
}
```

---

## 🔗 Integration Examples

### LangChain

```python
from langchain.document_loaders import ApifyDatasetLoader
from langchain.schema import Document

def transform_dataset_item(item):
    documents = []
    for chunk in item.get("chunks", []):
        documents.append(Document(
            page_content=chunk["text"],
            metadata=chunk["metadata"]
        ))
    return documents

loader = ApifyDatasetLoader(
    dataset_id="your_dataset_id",
    dataset_mapping_function=transform_dataset_item
)

documents = loader.load()
```

### LlamaIndex

```python
from llama_index import Document
from apify_client import ApifyClient

client = ApifyClient("your_api_token")
dataset = client.dataset("your_dataset_id").list_items().items

documents = []
for item in dataset:
    for chunk in item.get("chunks", []):
        documents.append(Document(
            text=chunk["text"],
            metadata=chunk["metadata"]
        ))
```

### Pinecone

```python
import pinecone
from openai import OpenAI

# Initialize
pinecone.init(api_key="your-api-key")
index = pinecone.Index("your-index")
openai = OpenAI()

# Upsert chunks
for item in dataset:
    for chunk in item["chunks"]:
        # Generate embedding
        response = openai.embeddings.create(
            input=chunk["text"],
            model="text-embedding-3-small"
        )
        embedding = response.data[0].embedding
        
        # Upsert to Pinecone
        index.upsert([(
            chunk["id"],
            embedding,
            chunk["metadata"]
        )])
```

### Qdrant

```python
from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct

client = QdrantClient("localhost", port=6333)

points = []
for item in dataset:
    for chunk in item["chunks"]:
        embedding = embed_model.encode(chunk["text"])
        points.append(PointStruct(
            id=hash(chunk["id"]),
            vector=embedding.tolist(),
            payload=chunk["metadata"]
        ))

client.upsert(collection_name="docs", points=points)
```

---

## ⚙️ Configuration Reference

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `startUrls` | array | *required* | URLs to begin crawling |
| `crawlerType` | string | `"cheerio"` | `"cheerio"` (fast) or `"playwright"` (JS support) |
| `maxCrawlPages` | integer | `100` | Maximum pages to crawl |
| `maxCrawlDepth` | integer | `20` | Maximum crawl depth from start URLs |
| `chunkingStrategy` | string | `"semantic"` | Chunking algorithm to use |
| `chunkSize` | integer | `512` | Target chunk size (tokens/words) |
| `chunkOverlap` | integer | `100` | Overlap between chunks |
| `outputFormat` | string | `"vector_ready"` | Output format |
| `removeElements` | array | *see below* | CSS selectors to remove |
| `includeMetadata` | boolean | `true` | Extract page metadata |
| `excludeUrlPatterns` | array | *see below* | URL patterns to skip |

### Default Remove Elements
```json
["nav", "header", "footer", ".advertisement", "#cookie-banner", ".sidebar"]
```

### Default Exclude Patterns
```json
["**/login**", "**/signup**", "**/register**", "**/cart**", "**/checkout**"]
```

---

## 💡 Pro Tips

### For Best RAG Performance
- Use **semantic chunking** for intelligent topic grouping
- Set **chunk overlap to 15-20%** of chunk size (e.g., 100 for 512-token chunks)
- Enable **metadata extraction** for better filtering during retrieval
- Target **400-600 tokens** per chunk for most embedding models

### For Large Documentation Sites
- Start with **maxCrawlPages = 50** to test configuration
- Use **Cheerio crawler** (10x faster) unless site requires JavaScript
- Set **exclude patterns** for login, user profiles, and dynamic pages
- Enable **Apify Proxy** to avoid rate limiting

### For Code-Heavy Content
- **Markdown section chunking** preserves code block structure
- Extracted metadata includes **code languages detected**
- Code blocks are **never split** mid-block

---

## 🔧 Troubleshooting

### Getting blocked by website
**Solution**: Enable Apify Proxy in configuration. For aggressive blocking, use residential proxies.

### Missing content on JavaScript sites
**Solution**: Switch to `"crawlerType": "playwright"` for full JavaScript rendering.

### Chunks too large for embeddings
**Solution**: Reduce `chunkSize` to 384-512 tokens. Most models have 8192-token limits.

### Empty chunks generated
**Solution**: Check `removeElements` - you may be removing content containers. Reduce selectors.

### Slow crawling speed
**Solution**: Increase `maxConcurrency` (carefully) or use Cheerio crawler for static sites.

---

## 📈 Performance

| Metric | Cheerio Crawler | Playwright Crawler |
|--------|-----------------|-------------------|
| Speed | ~10 pages/sec | ~1 page/sec |
| JavaScript Support | ❌ No | ✅ Yes |
| Memory Usage | Low | Medium |
| Best For | Documentation, Blogs | SPAs, Dynamic Sites |

---

## 🏆 Why This Actor?

Compared to generic web scrapers, AI Training Data Scraper offers:

| Feature | Generic Scrapers | This Actor |
|---------|-----------------|------------|
| Token-Aware Chunking | ❌ | ✅ Uses tiktoken |
| Semantic Chunking | ❌ | ✅ NLP-based |
| Vector DB Ready | ❌ | ✅ Pre-formatted |
| Code Block Handling | ❌ | ✅ Never splits |
| Metadata Extraction | Basic | 15+ fields |
| RAG Optimization | ❌ | ✅ Purpose-built |

---

## 📞 Support

- **Issues**: Report bugs via Apify Console
- **Feature Requests**: Submit through Apify feedback
- **Documentation**: [Full API Reference](https://docs.apify.com)

---

## 📄 License

MIT License - See [LICENSE](LICENSE) file for details.

---

**Built for developers, optimized for AI.** ⚡

*Transform the web into training data.*
