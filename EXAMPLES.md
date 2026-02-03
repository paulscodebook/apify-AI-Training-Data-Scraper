# Integration Examples

Complete integration examples for popular AI/ML frameworks and vector databases.

---

## Table of Contents

1. [LangChain Integration](#langchain-integration)
2. [LlamaIndex Integration](#llamaindex-integration)
3. [Vector Databases](#vector-databases)
   - [Pinecone](#pinecone)
   - [Qdrant](#qdrant)
   - [Weaviate](#weaviate)
   - [ChromaDB](#chromadb)
4. [Embedding Models](#embedding-models)
5. [RAG Pipeline Example](#rag-pipeline-example)

---

## LangChain Integration

### Basic Document Loading

```python
from langchain.document_loaders import ApifyDatasetLoader
from langchain.schema import Document
from apify_client import ApifyClient

# First, run the actor
client = ApifyClient("your_api_token")
run = client.actor("your-username/ai-training-data-scraper").call(
    run_input={
        "startUrls": [{"url": "https://docs.example.com/"}],
        "chunkingStrategy": "semantic",
        "outputFormat": "vector_ready"
    }
)

# Load results
def transform_item(item):
    return [
        Document(
            page_content=chunk["text"],
            metadata=chunk["metadata"]
        )
        for chunk in item.get("chunks", [])
    ]

loader = ApifyDatasetLoader(
    dataset_id=run["defaultDatasetId"],
    dataset_mapping_function=transform_item
)

documents = loader.load()
print(f"Loaded {len(documents)} documents")
```

### With Vector Store

```python
from langchain.vectorstores import Chroma
from langchain.embeddings import OpenAIEmbeddings

# Create embeddings and vector store
embeddings = OpenAIEmbeddings()
vectorstore = Chroma.from_documents(
    documents=documents,
    embedding=embeddings,
    collection_name="documentation"
)

# Query
results = vectorstore.similarity_search(
    "How do I get started?",
    k=5
)
```

### RAG Chain

```python
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI

llm = ChatOpenAI(model="gpt-4")
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=vectorstore.as_retriever(search_kwargs={"k": 5})
)

answer = qa_chain.run("What are the main features?")
print(answer)
```

---

## LlamaIndex Integration

### Basic Usage

```python
from llama_index import Document, VectorStoreIndex
from apify_client import ApifyClient

client = ApifyClient("your_api_token")
dataset = client.dataset("your_dataset_id").list_items().items

# Convert to LlamaIndex documents
documents = []
for item in dataset:
    for chunk in item.get("chunks", []):
        documents.append(Document(
            text=chunk["text"],
            extra_info=chunk["metadata"]
        ))

# Create index
index = VectorStoreIndex.from_documents(documents)

# Query
query_engine = index.as_query_engine()
response = query_engine.query("Explain the installation process")
print(response)
```

---

## Vector Databases

### Pinecone

```python
import pinecone
from openai import OpenAI

# Initialize
pinecone.init(api_key="your-pinecone-key", environment="us-west1-gcp")
openai = OpenAI(api_key="your-openai-key")

# Create index
if "docs" not in pinecone.list_indexes():
    pinecone.create_index("docs", dimension=1536, metric="cosine")

index = pinecone.Index("docs")

# Process and upsert
for item in dataset:
    vectors = []
    for chunk in item["chunks"]:
        response = openai.embeddings.create(
            input=chunk["text"],
            model="text-embedding-3-small"
        )
        vectors.append({
            "id": chunk["id"],
            "values": response.data[0].embedding,
            "metadata": chunk["metadata"]
        })
    
    index.upsert(vectors=vectors, batch_size=100)
```

### Qdrant

```python
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct

client = QdrantClient("localhost", port=6333)

# Create collection
client.create_collection(
    collection_name="docs",
    vectors_config=VectorParams(size=1536, distance=Distance.COSINE)
)

# Upsert
points = []
for item in dataset:
    for chunk in item["chunks"]:
        embedding = embed_model.encode(chunk["text"])
        points.append(PointStruct(
            id=abs(hash(chunk["id"])) % (10 ** 10),
            vector=embedding.tolist(),
            payload=chunk["metadata"]
        ))

client.upsert(collection_name="docs", points=points)
```

### Weaviate

```python
import weaviate

client = weaviate.Client("http://localhost:8080")

# Create schema
class_obj = {
    "class": "Documentation",
    "vectorizer": "text2vec-openai",
    "properties": [
        {"name": "content", "dataType": ["text"]},
        {"name": "source_url", "dataType": ["string"]},
        {"name": "page_title", "dataType": ["string"]},
    ]
}
client.schema.create_class(class_obj)

# Import data
for item in dataset:
    for chunk in item["chunks"]:
        client.data_object.create(
            class_name="Documentation",
            data_object={
                "content": chunk["text"],
                **chunk["metadata"]
            }
        )
```

### ChromaDB

```python
import chromadb

client = chromadb.Client()
collection = client.create_collection("docs")

for item in dataset:
    ids = [c["id"] for c in item["chunks"]]
    documents = [c["text"] for c in item["chunks"]]
    metadatas = [c["metadata"] for c in item["chunks"]]
    
    collection.add(
        ids=ids,
        documents=documents,
        metadatas=metadatas
    )
```

---

## Embedding Models

### OpenAI

```python
from openai import OpenAI

client = OpenAI()

def embed_text(text: str) -> list:
    response = client.embeddings.create(
        input=text,
        model="text-embedding-3-small"  # or text-embedding-3-large
    )
    return response.data[0].embedding
```

### Cohere

```python
import cohere

co = cohere.Client("your-api-key")

def embed_texts(texts: list) -> list:
    response = co.embed(
        texts=texts,
        model="embed-english-v3.0",
        input_type="search_document"
    )
    return response.embeddings
```

### Sentence Transformers (Local)

```python
from sentence_transformers import SentenceTransformer

model = SentenceTransformer('all-MiniLM-L6-v2')

def embed_texts(texts: list) -> list:
    return model.encode(texts).tolist()
```

---

## RAG Pipeline Example

Complete end-to-end RAG pipeline:

```python
from apify_client import ApifyClient
from openai import OpenAI
import chromadb

# 1. Scrape documentation
apify = ApifyClient("your-apify-token")
run = apify.actor("ai-training-data-scraper").call(
    run_input={
        "startUrls": [{"url": "https://docs.example.com/"}],
        "chunkingStrategy": "semantic",
        "chunkSize": 512,
        "outputFormat": "vector_ready"
    }
)

dataset = apify.dataset(run["defaultDatasetId"]).list_items().items

# 2. Create embeddings and store
openai = OpenAI()
chroma = chromadb.Client()
collection = chroma.create_collection("docs")

for item in dataset:
    for chunk in item["chunks"]:
        embedding = openai.embeddings.create(
            input=chunk["text"],
            model="text-embedding-3-small"
        ).data[0].embedding
        
        collection.add(
            ids=[chunk["id"]],
            embeddings=[embedding],
            documents=[chunk["text"]],
            metadatas=[chunk["metadata"]]
        )

# 3. Query function
def query_docs(question: str, n_results: int = 5) -> str:
    q_embedding = openai.embeddings.create(
        input=question,
        model="text-embedding-3-small"
    ).data[0].embedding
    
    results = collection.query(
        query_embeddings=[q_embedding],
        n_results=n_results
    )
    
    context = "\n\n".join(results["documents"][0])
    
    response = openai.chat.completions.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": f"Answer based on context:\n{context}"},
            {"role": "user", "content": question}
        ]
    )
    
    return response.choices[0].message.content

# 4. Use it
answer = query_docs("How do I install the package?")
print(answer)
```

---

## Tips for Best Results

1. **Use semantic chunking** for most RAG applications
2. **Match chunk size** to your embedding model's optimal input length
3. **Include metadata** for filtering (by date, author, content type)
4. **Use overlap** to prevent context loss at boundaries
5. **Test with small datasets** first to validate configuration
