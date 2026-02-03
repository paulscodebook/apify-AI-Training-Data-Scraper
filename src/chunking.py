"""
Chunking Engine Module

Implements 4 intelligent chunking strategies for optimal embedding and retrieval.
"""

import re
import hashlib
from typing import List, Dict, Any, Optional
from dataclasses import dataclass


def get_tiktoken():
    import tiktoken
    return tiktoken

def get_nltk():
    import nltk
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        nltk.download('punkt', quiet=True)
    try:
        nltk.data.find('tokenizers/punkt_tab')
    except LookupError:
        nltk.download('punkt_tab', quiet=True)
    return nltk


@dataclass
class Chunk:
    content: str
    chunk_index: int
    token_count: int
    word_count: int
    start_position: int
    end_position: int
    section_title: Optional[str] = None
    has_code: bool = False
    heading_level: Optional[int] = None


class ChunkingEngine:
    def __init__(self, strategy: str = "semantic", chunk_size: int = 512, overlap: int = 100):
        self.strategy = strategy
        self.chunk_size = chunk_size
        self.overlap = overlap
        self._encoder = None
        self._nltk = None
        self._sentence_model = None

    @property
    def encoder(self):
        if self._encoder is None:
            tiktoken = get_tiktoken()
            self._encoder = tiktoken.get_encoding("cl100k_base")
        return self._encoder

    @property
    def nltk(self):
        if self._nltk is None:
            self._nltk = get_nltk()
        return self._nltk

    def _protect_code_blocks(self, text: str) -> tuple[str, List[str]]:
        """Identify and protect code blocks from being split."""
        code_blocks = []
        
        # 1. Fenced code blocks (triple backticks)
        def fenced_repl(match):
            code_blocks.append(match.group(0))
            return f"\n___CODE_BLOCK_{len(code_blocks)-1}___\n"
        
        text = re.sub(r'```[\s\S]*?```', fenced_repl, text)
        
        # 2. Python REPL blocks (>>>)
        # Match a line starting with >>> followed by lines starting with ... or just indented lines
        # This is a heuristic that captures typical Python REPL output
        def repl_repl(match):
            code_blocks.append(match.group(0))
            return f"\n___CODE_BLOCK_{len(code_blocks)-1}___\n"
        
        # Regex for lines starting with >>> and following lines starting with ... or indentation
        repl_pattern = r'(?:^|\n)>>> (?:.*\n(?:(?:\.\.\. | {4}).*\n)*)'
        text = re.sub(repl_pattern, repl_repl, text)
        
        return text, code_blocks

    def _restore_code_blocks(self, text: str, code_blocks: List[str]) -> str:
        """Restore protected code blocks."""
        for i, block in enumerate(code_blocks):
            text = text.replace(f"___CODE_BLOCK_{i}___", block)
        return text

    def _segment_text(self, text: str, max_tokens: int = 800) -> List[str]:
        """Pre-segment text if it's too long to reduce semantic similarity complexity."""
        # Simple split by paragraphs first to avoid breaking them
        paragraphs = text.split('\n\n')
        segments = []
        current_segment = []
        current_tokens = 0
        
        for p in paragraphs:
            p_tokens = self._count_tokens(p)
            if current_tokens + p_tokens > max_tokens and current_segment:
                segments.append('\n\n'.join(current_segment))
                current_segment = [p]
                current_tokens = p_tokens
            else:
                current_segment.append(p)
                current_tokens += p_tokens
        
        if current_segment:
            segments.append('\n\n'.join(current_segment))
            
        return segments

    def chunk(self, text: str, url: str = "", title: str = "") -> List[Dict[str, Any]]:
        if not text or not text.strip():
            return []

        # Protect code blocks
        protected_text, code_blocks = self._protect_code_blocks(text)

        if self.strategy == "semantic":
            # Apply hard cutoff/segmentation for semantic strategy
            segments = self._segment_text(protected_text, max_tokens=800)
            all_chunks = []
            
            for seg in segments:
                seg_chunks = self._chunk_semantically(seg)
                all_chunks.extend(seg_chunks)
            chunks = all_chunks
        else:
            if self.strategy == "fixed_token":
                chunks = self._chunk_by_tokens(protected_text)
            elif self.strategy == "sentence_based":
                chunks = self._chunk_by_sentences(protected_text)
            elif self.strategy == "markdown_section":
                chunks = self._chunk_by_markdown(protected_text)
            else:
                chunks = self._chunk_by_sentences(protected_text)

        result = []
        url_hash = hashlib.md5(url.encode()).hexdigest()[:8]
        
        for i, chunk in enumerate(chunks):
            # Restore code blocks
            final_content = self._restore_code_blocks(chunk.content, code_blocks)
            
            result.append({
                "chunk_id": f"{url_hash}_chunk_{i}",
                "chunk_index": i,
                "content": final_content.strip(),
                "token_count": self._count_tokens(final_content),
                "word_count": len(final_content.split()),
                "char_count": len(final_content),
                "embedding_ready": True,
                "chunk_metadata": {
                    "start_position": chunk.start_position,
                    "end_position": chunk.end_position,
                    "has_code": self._has_code(final_content),
                    "section_title": chunk.section_title,
                    "heading_level": chunk.heading_level
                }
            })
        return result

    def _count_tokens(self, text: str) -> int:
        try:
            return len(self.encoder.encode(text))
        except:
            return int(len(text.split()) * 1.3)

    def _has_code(self, text: str) -> bool:
        patterns = [r'```', r'`[^`]+`', r'>>>', r'def\s+\w+\(', r'function\s+\w+\(', r'class\s+\w+', r'import\s+\w+']
        return any(re.search(p, text) for p in patterns)

    def _chunk_by_tokens(self, text: str) -> List[Chunk]:
        try:
            tokens = self.encoder.encode(text)
        except:
            return self._chunk_by_words(text)
        chunks = []
        start = 0
        while start < len(tokens):
            end = min(start + self.chunk_size, len(tokens))
            chunk_tokens = tokens[start:end]
            try:
                chunk_text = self.encoder.decode(chunk_tokens)
            except:
                chunk_text = ""
            chunks.append(Chunk(content=chunk_text.strip(), chunk_index=len(chunks), token_count=len(chunk_tokens), word_count=len(chunk_text.split()), start_position=start, end_position=end))
            start += max(1, self.chunk_size - self.overlap)
        return chunks

    def _chunk_by_words(self, text: str) -> List[Chunk]:
        words = text.split()
        chunks = []
        start = 0
        while start < len(words):
            end = min(start + self.chunk_size, len(words))
            chunk_text = ' '.join(words[start:end])
            chunks.append(Chunk(content=chunk_text, chunk_index=len(chunks), token_count=int(len(chunk_text.split()) * 1.3), word_count=end - start, start_position=start, end_position=end))
            start += max(1, self.chunk_size - self.overlap)
        return chunks

    def _chunk_by_sentences(self, text: str) -> List[Chunk]:
        try:
            sentences = self.nltk.sent_tokenize(text)
        except:
            sentences = re.split(r'(?<=[.!?])\s+', text)
        chunks = []
        current = []
        word_count = 0
        pos = 0
        for s in sentences:
            s = s.strip()
            if not s:
                continue
            s_words = len(s.split())
            if word_count + s_words > self.chunk_size * 1.5 and current:
                chunk_text = ' '.join(current)
                chunks.append(Chunk(content=chunk_text, chunk_index=len(chunks), token_count=self._count_tokens(chunk_text), word_count=word_count, start_position=pos, end_position=pos + len(chunk_text)))
                pos += len(chunk_text)
                current = [s]
                word_count = s_words
            else:
                current.append(s)
                word_count += s_words
        if current:
            chunk_text = ' '.join(current)
            chunks.append(Chunk(content=chunk_text, chunk_index=len(chunks), token_count=self._count_tokens(chunk_text), word_count=word_count, start_position=pos, end_position=pos + len(chunk_text)))
        return chunks

    def _chunk_semantically(self, text: str) -> List[Chunk]:
        try:
            sentences = self.nltk.sent_tokenize(text)
        except:
            sentences = re.split(r'(?<=[.!?])\s+', text)
        if len(sentences) <= 3:
            return self._chunk_by_sentences(text)
        try:
            return self._semantic_with_embeddings(sentences)
        except:
            return self._semantic_heuristic(sentences, text)

    def _semantic_with_embeddings(self, sentences: List[str]) -> List[Chunk]:
        import numpy as np
        from sentence_transformers import SentenceTransformer
        if self._sentence_model is None:
            self._sentence_model = SentenceTransformer('all-MiniLM-L6-v2')
        embeddings = self._sentence_model.encode(sentences)
        
        # Sliding window approach: compare each sentence with its neighbors
        # User specified "Cap semantic similarity scope" to sliding window
        sims = []
        for i in range(1, len(embeddings)):
            sim = np.dot(embeddings[i-1], embeddings[i]) / (np.linalg.norm(embeddings[i-1]) * np.linalg.norm(embeddings[i]))
            sims.append(sim)
            
        threshold = np.percentile(sims, 25)
        chunks = []
        current = [sentences[0]]
        word_count = len(sentences[0].split())
        pos = 0
        for s, sim in zip(sentences[1:], sims):
            # Dynamic reasoning: break if similarity is low OR we are over the target size
            should_break = (sim < threshold and word_count >= self.chunk_size // 2) or word_count >= self.chunk_size * 1.5
            if should_break and current:
                chunk_text = ' '.join(current)
                chunks.append(Chunk(content=chunk_text, chunk_index=len(chunks), token_count=self._count_tokens(chunk_text), word_count=word_count, start_position=pos, end_position=pos + len(chunk_text)))
                pos += len(chunk_text)
                current = [s]
                word_count = len(s.split())
            else:
                current.append(s)
                word_count += len(s.split())
        if current:
            chunk_text = ' '.join(current)
            chunks.append(Chunk(content=chunk_text, chunk_index=len(chunks), token_count=self._count_tokens(chunk_text), word_count=word_count, start_position=pos, end_position=pos + len(chunk_text)))
        return chunks

    def _semantic_heuristic(self, sentences: List[str], text: str) -> List[Chunk]:
        patterns = [r'^(?:however|but|although)', r'^(?:first|second|finally)', r'^(?:for example)', r'^(?:note:|warning:)', r'^\d+\.', r'^[-•*]\s']
        chunks = []
        current = []
        word_count = 0
        pos = 0
        for s in sentences:
            if not s.strip():
                continue
            is_break = any(re.match(p, s.strip(), re.I) for p in patterns)
            should_break = (is_break and word_count >= self.chunk_size // 3) or word_count >= self.chunk_size
            if should_break and current:
                chunk_text = ' '.join(current)
                chunks.append(Chunk(content=chunk_text, chunk_index=len(chunks), token_count=self._count_tokens(chunk_text), word_count=word_count, start_position=pos, end_position=pos + len(chunk_text)))
                pos += len(chunk_text)
                current = [s]
                word_count = len(s.split())
            else:
                current.append(s)
                word_count += len(s.split())
        if current:
            chunk_text = ' '.join(current)
            chunks.append(Chunk(content=chunk_text, chunk_index=len(chunks), token_count=self._count_tokens(chunk_text), word_count=word_count, start_position=pos, end_position=pos + len(chunk_text)))
        return chunks

    def _chunk_by_markdown(self, text: str) -> List[Chunk]:
        lines = text.split('\n')
        chunks = []
        current = []
        heading = None
        level = None
        pos = 0
        for line in lines:
            m = re.match(r'^(#{1,6})\s+(.+)$', line)
            if m:
                if current:
                    chunk_text = '\n'.join(current)
                    if len(chunk_text.split()) >= 50:
                        chunks.append(Chunk(content=chunk_text, chunk_index=len(chunks), token_count=self._count_tokens(chunk_text), word_count=len(chunk_text.split()), start_position=pos, end_position=pos + len(chunk_text), section_title=heading, heading_level=level))
                        pos += len(chunk_text)
                current = [line]
                heading = m.group(2).strip()
                level = len(m.group(1))
            else:
                current.append(line)
        if current:
            chunk_text = '\n'.join(current)
            if len(chunk_text.split()) >= 20:
                chunks.append(Chunk(content=chunk_text, chunk_index=len(chunks), token_count=self._count_tokens(chunk_text), word_count=len(chunk_text.split()), start_position=pos, end_position=pos + len(chunk_text), section_title=heading, heading_level=level))
        return chunks if chunks else self._chunk_by_sentences(text)
