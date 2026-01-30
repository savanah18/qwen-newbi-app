# Extracting Embeddings from Qwen3-VL (VLM2Vec)

## Overview

This guide explains how to extract hidden states from Qwen3-VL to create embeddings for RAG. Instead of using a separate embedding model, we leverage the existing Qwen3-VL model's internal representations.

## How It Works

### 1. **Model Architecture**

Qwen3-VL has multiple transformer layers. Each layer produces "hidden states" - intermediate representations of the input:

```
Input (text + image) 
    → Tokenization & Image Encoding
    → Layer 1 (hidden state 1)
    → Layer 2 (hidden state 2)
    → ...
    → Layer N (hidden state N)  ← We extract this!
    → Language Modeling Head
    → Generated Text
```

### 2. **Extracting Hidden States**

We stop before the generation head and extract the last hidden state:

```python
# Normal generation (what we currently do)
output = model.generate(**inputs)

# Embedding extraction (what we'll add)
output = model(**inputs, output_hidden_states=True, return_dict=True)
hidden_states = output.hidden_states[-1]  # Last layer
```

### 3. **Pooling to Fixed Size**

Hidden states have shape `(batch_size, sequence_length, hidden_size)`. We need `(batch_size, hidden_size)`:

**Pooling Strategies:**

- **Mean Pooling** (Recommended): Average across sequence length
  - Best for capturing overall semantic meaning
  - Most common in sentence embeddings

- **CLS Token**: Use first token representation
  - Common in BERT-style models
  - Fast, but loses sequence information

- **Max Pooling**: Take maximum value across sequence
  - Captures strongest features
  - Can be noisy

- **Last Token**: Use last non-padding token
  - Good for decoder models
  - May miss earlier context

### 4. **Normalization**

L2-normalize embeddings for cosine similarity:

```python
embedding = embedding / np.linalg.norm(embedding)
```

## Implementation

### Basic Usage

```python
from agent.memory.embeddings import VLM2VecEmbeddings
from transformers import AutoModelForCausalLM, AutoProcessor

# Load your existing model
model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen3-VL-8B-Instruct",
    torch_dtype="auto",
    device_map="auto",
    load_in_4bit=True
)
processor = AutoProcessor.from_pretrained("Qwen/Qwen3-VL-8B-Instruct")

# Create embedding service
embedder = VLM2VecEmbeddings(
    model=model,
    processor=processor,
    pooling_strategy="mean",
    normalize=True
)

# Get embedding dimension
print(f"Embedding dimension: {embedder.get_embedding_dimension()}")
# Output: 4096 (for Qwen3-VL-8B)

# Generate text embedding
text = "Explain bubble sort algorithm"
embedding = embedder.embed_text(text)
print(embedding.shape)  # (4096,)

# Generate multimodal embedding
from PIL import Image
image = Image.open("algorithm_diagram.png")
embedding = embedder.embed_multimodal(
    text="Binary search tree visualization",
    image=image
)
print(embedding.shape)  # (4096,)
```

### Batch Processing

```python
# Prepare batch
items = [
    {"text": "Quicksort algorithm"},
    {"text": "Merge sort explained"},
    {"text": "Binary tree", "image": tree_image},
]

# Generate embeddings efficiently
embeddings = embedder.embed_batch(items, batch_size=8)

# Use in vector database
import chromadb
client = chromadb.Client()
collection = client.create_collection(
    name="knowledge_base",
    metadata={"dimension": embedder.get_embedding_dimension()}
)

# Add to collection
collection.add(
    embeddings=[emb.tolist() for emb in embeddings],
    documents=[item["text"] for item in items],
    ids=[f"doc_{i}" for i in range(len(items))]
)
```

### Integration with Inference

```python
# In your inference_engine.py
class InferenceEngine:
    def __init__(self, ...):
        # ... existing code ...
        
        # Add embedding service
        from agent.memory.embeddings import VLM2VecEmbeddings
        self.embedder = VLM2VecEmbeddings(
            model=model,
            processor=processor,
            pooling_strategy="mean"
        )
    
    async def embed_query(self, text: str, image: Optional[Image] = None):
        """Generate embedding for user query."""
        if image:
            return self.embedder.embed_multimodal(text, image)
        return self.embedder.embed_text(text)
```

## Technical Details

### Memory Considerations

**Embedding Extraction vs Generation:**
- Generation: Processes token-by-token, generates sequence
- Embedding: Single forward pass, no generation
- **Memory savings**: ~30-40% less memory (no KV cache for generation)

**Batch Size:**
- Text-only: Can batch 16-32 items
- With images: Batch 4-8 items (images use more memory)

### Performance

**Latency Comparison (on A100):**
```
Text generation (512 tokens):     ~2.0s
Embedding extraction (text-only):  ~0.1s
Embedding extraction (multimodal): ~0.3s
```

**Throughput:**
- Can generate ~100-200 embeddings/second (text-only)
- Can generate ~30-50 embeddings/second (multimodal)

### Embedding Quality

**Factors Affecting Quality:**

1. **Pooling Strategy**:
   - Mean pooling: Best for general semantic similarity
   - CLS token: Fast but may miss nuances
   - Test on your data to find best strategy

2. **Layer Selection**:
   - Last layer (`hidden_states[-1]`): Best for task-specific embeddings
   - Middle layers: More general representations
   - We use last layer by default

3. **Normalization**:
   - Always normalize for cosine similarity
   - Enables consistent distance metrics

## Comparison with Alternatives

### Option 1: Qwen3-VL Hidden States (Our Approach)

**Pros:**
- ✅ No additional model to load
- ✅ Multimodal support (text + images)
- ✅ Consistent with generation model
- ✅ High-quality contextualized representations

**Cons:**
- ❌ Large embedding dimension (4096)
- ❌ Requires GPU for efficient extraction
- ❌ Slower than dedicated embedding models

### Option 2: Sentence Transformers (Baseline)

**Pros:**
- ✅ Optimized for embeddings
- ✅ Smaller dimensions (384-768)
- ✅ Fast on CPU
- ✅ Many pre-trained models

**Cons:**
- ❌ Text-only (no vision)
- ❌ Separate model to maintain
- ❌ May not align with Qwen3-VL semantics

### Option 3: Dedicated VLM Embedding Models

**Pros:**
- ✅ Optimized for embedding task
- ✅ Potentially higher quality
- ✅ Smaller dimensions

**Cons:**
- ❌ Additional model to load (~1-2GB)
- ❌ More complexity
- ❌ May not align with Qwen3-VL

### Recommendation

**Start with Qwen3-VL embeddings (Option 1):**
1. No new dependencies
2. Multimodal from day one
3. Can always switch later

**Consider alternatives if:**
- Vector DB storage becomes expensive (use dimension reduction)
- CPU-only deployment needed (use sentence-transformers)
- Need extreme performance (use optimized embedding models)

## Advanced: Dimension Reduction

If 4096 dimensions is too large:

```python
from sklearn.decomposition import PCA

# Train PCA on sample embeddings
pca = PCA(n_components=768)  # Reduce to 768 dims
pca.fit(sample_embeddings)

# Transform new embeddings
reduced_embedding = pca.transform(embedding.reshape(1, -1))
```

Or use **Product Quantization** in vector DB:
```python
# ChromaDB with HNSW + PQ
collection = client.create_collection(
    name="knowledge_base",
    metadata={
        "hnsw:space": "cosine",
        "hnsw:construction_ef": 200,
        "hnsw:M": 16,
    }
)
```

## Using Triton for Production Embeddings

For production systems, use **Triton Inference Server** instead of loading the model directly. This provides:

- ✅ GPU sharing across multiple models
- ✅ Dynamic batching (up to 32 concurrent requests)
- ✅ Automatic scaling
- ✅ HTTP/gRPC APIs
- ✅ Model versioning
- ✅ Built-in monitoring

### Triton Setup

```bash
# Start Triton server with Qwen3-VL model
docker compose up triton-server

# Verify it's ready
curl http://localhost:8000/v2/models/qwen3-vl
```

### Python Client (Triton)

```python
from agent.memory.embeddings import TritonEmbeddings

# Connect to Triton server
embedder = TritonEmbeddings(
    triton_url="localhost:8000",
    model_name="qwen3-vl",
    use_grpc=False  # Use HTTP (or True for gRPC)
)

# Single text embedding
text = "Explain quicksort algorithm"
embedding = embedder.embed_text(text)
print(f"Embedding shape: {embedding.shape}")  # (3584,)

# Batch embeddings (efficient with dynamic batching)
texts = [
    {"text": "What is merge sort?"},
    {"text": "Explain binary search"},
    {"text": "How does heap sort work?"}
]
embeddings = embedder.embed_batch(texts, batch_size=16)
print(f"Generated {len(embeddings)} embeddings")

# Multimodal (text + image)
from PIL import Image
image = Image.open("algorithm_diagram.png")
embedding = embedder.embed_multimodal("Explain this structure:", image)
```

### RAG System with Triton

```python
from agent.memory import create_rag_system

# Initialize with Triton (recommended for production)
rag = await create_rag_system(
    use_triton=True,
    triton_url="localhost:8000"
)

# Or use local model (development only)
from agent.serving.fastapi.src.model_loader import ModelLoader
model, processor = ModelLoader.load_native(...)
rag = await create_rag_system(
    model=model,
    processor=processor,
    use_triton=False
)

# Retrieve context
results = await rag.retrieve_context("What is quicksort?", top_k=3)
for result in results:
    print(f"Score: {result['score']:.3f}")
    print(f"Text: {result['text'][:100]}...")
```

### Triton Embedding Modes

The Triton model supports two modes:

| Mode | Input | Output | Use Case |
|------|-------|--------|----------|
| `generate` | text (+ optional image) | Generated text response | Question answering |
| `embed` | text (+ optional image) | 3584-dim embedding vector | RAG/similarity search |

**API Call:**
```python
# Generate mode (Q&A)
response_text, response_time = embedder.client.chat(
    "What is quicksort?",
    mode="generate"
)

# Embed mode (RAG)
embedding, response_time = embedder.client.chat(
    "Quicksort is a divide-and-conquer sorting algorithm...",
    mode="embed"
)
```

## Debugging & Validation

### Test Embedding Quality

```python
# Test semantic similarity
texts = [
    "Bubble sort algorithm",
    "Bubble sort implementation",
    "Binary search tree",
]

embeddings = [embedder.embed_text(t) for t in texts]

# Compute similarities
from sklearn.metrics.pairwise import cosine_similarity
similarities = cosine_similarity(embeddings)
print(similarities)
# [[1.00, 0.85, 0.42],  ← High similarity between sort texts
#  [0.85, 1.00, 0.39],
#  [0.42, 0.39, 1.00]]
```

### Verify Multimodal Fusion

```python
# Test that images affect embeddings
text = "Binary tree structure"
emb_text_only = embedder.embed_text(text)
emb_with_image = embedder.embed_multimodal(text, tree_diagram)

# Should be different
similarity = cosine_similarity(
    emb_text_only.reshape(1, -1),
    emb_with_image.reshape(1, -1)
)
print(f"Similarity: {similarity[0][0]:.3f}")
# Should be ~0.7-0.9 (similar but not identical)
```

### Monitor Performance (Triton)

```python
# Check Triton health
import requests
health = requests.get("http://localhost:8000/v2/health/ready")
print(f"Triton status: {health.status_code}")

# Benchmark with dynamic batching
import time
import asyncio

async def benchmark():
    texts = ["Query " + str(i) for i in range(100)]
    
    start = time.time()
    embeddings = embedder.embed_batch(
        [{"text": t} for t in texts],
        batch_size=32  # Triton handles batching
    )
    duration = time.time() - start
    
    print(f"Processed 100 items in {duration:.2f}s")
    print(f"Throughput: {100/duration:.1f} items/sec")
    # With dynamic batching: 50-100 items/sec on GPU
```

## Next Steps

1. **Implement in RAG pipeline** - See [RAG_INTEGRATION_PLAN.md](../RAG_INTEGRATION_PLAN.md) Phase 2
2. **Test with sample content** - Evaluate retrieval quality
3. **Optimize for production** - Batch processing, caching
4. **Monitor in production** - Track latency, quality metrics

## Resources

- [Transformers Output Documentation](https://huggingface.co/docs/transformers/main_classes/output)
- [Sentence Embeddings Guide](https://www.sbert.net/)
- [ChromaDB Python Client](https://docs.trychroma.com/usage-guide)
