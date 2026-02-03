"""
Output Formatter Module

Formats extracted data for different output formats optimized for LLMs and vector databases.
"""

from datetime import datetime, timezone
from typing import Dict, Any, List


class OutputFormatter:
    """Formats output in various formats for different use cases."""

    def __init__(self, output_format: str = "vector_ready"):
        self.output_format = output_format

    def format(
        self,
        url: str,
        title: str,
        content: str,
        chunks: List[Dict[str, Any]],
        metadata: Dict[str, Any],
        extract_links: bool = False,
        crawl_depth: int = 0
    ) -> Dict[str, Any]:
        """Format the extracted data based on output format setting."""
        base = {
            "url": url,
            "title": title,
            "crawl_info": {
                "crawled_at": datetime.now(timezone.utc).isoformat(),
                "crawl_depth": crawl_depth,
            }
        }

        if self.output_format == "markdown":
            return self._format_markdown(base, content, chunks, metadata)
        elif self.output_format == "plain_text":
            return self._format_plain_text(base, content, chunks, metadata)
        elif self.output_format == "json_structured":
            return self._format_json_structured(base, content, chunks, metadata)
        else:  # vector_ready
            return self._format_vector_ready(base, content, chunks, metadata)

    def _format_markdown(self, base: Dict, content: str, chunks: List, metadata: Dict) -> Dict:
        """Format with markdown content preserved."""
        return {
            **base,
            "content_format": "markdown",
            "content": content,
            "chunks": [{"chunk_id": c["chunk_id"], "chunk_index": c["chunk_index"], "content": c["content"], "token_count": c["token_count"]} for c in chunks],
            "metadata": metadata,
            "chunk_count": len(chunks),
            "total_tokens": sum(c.get("token_count", 0) for c in chunks),
        }

    def _format_plain_text(self, base: Dict, content: str, chunks: List, metadata: Dict) -> Dict:
        """Format as clean plain text."""
        return {
            **base,
            "content_format": "plain_text",
            "content": content,
            "chunks": [{"chunk_id": c["chunk_id"], "chunk_index": c["chunk_index"], "content": c["content"], "word_count": c.get("word_count", 0)} for c in chunks],
            "metadata": metadata,
            "chunk_count": len(chunks),
        }

    def _format_json_structured(self, base: Dict, content: str, chunks: List, metadata: Dict) -> Dict:
        """Format with structured JSON output."""
        return {
            **base,
            "content_format": "json_structured",
            "content": {"full_text": content, "word_count": len(content.split())},
            "chunks": chunks,
            "metadata": metadata,
            "statistics": {
                "chunk_count": len(chunks),
                "total_tokens": sum(c.get("token_count", 0) for c in chunks),
                "avg_chunk_size": sum(c.get("token_count", 0) for c in chunks) // max(1, len(chunks)),
            }
        }

    def _format_vector_ready(self, base: Dict, content: str, chunks: List, metadata: Dict) -> Dict:
        """Format optimized for vector database ingestion."""
        vector_chunks = []
        for c in chunks:
            vector_chunks.append({
                "id": c["chunk_id"],
                "text": c["content"],
                "metadata": {
                    "source_url": base["url"],
                    "page_title": base["title"],
                    "chunk_index": c["chunk_index"],
                    "token_count": c.get("token_count", 0),
                    **c.get("chunk_metadata", {}),
                    **{k: v for k, v in metadata.items() if k in ["author", "language", "content_type", "keywords"]},
                }
            })
        return {
            **base,
            "content_format": "vector_ready",
            "full_content": content,
            "chunks": vector_chunks,
            "metadata": metadata,
            "embedding_info": {
                "chunk_count": len(chunks),
                "total_tokens": sum(c.get("token_count", 0) for c in chunks),
                "ready_for_embedding": True,
                "recommended_model": "text-embedding-3-small"
            }
        }
