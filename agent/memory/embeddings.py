"""
Embedding service for generating vector representations using Qwen3-VL.
Extracts hidden states from the vision-language model for RAG.
"""

from typing import List, Optional, Dict, Any, Union
import torch
import numpy as np
from PIL import Image
import base64
import io
from transformers import Qwen3VLForConditionalGeneration, AutoProcessor


class VLM2VecEmbeddings:
    """
    Extract embeddings from Qwen3-VL by accessing hidden states.
    
    This class provides methods to:
    1. Extract text embeddings from the language model encoder
    2. Extract multimodal (text + image) embeddings from the vision-text encoder
    3. Pool hidden states to fixed-size vectors suitable for vector databases
    """
    
    def __init__(
        self,
        model: Qwen3VLForConditionalGeneration,
        processor: AutoProcessor,
        pooling_strategy: str = "mean",  # Options: "mean", "cls", "max", "last"
        normalize: bool = True
    ):
        """
        Initialize embedding service.
        
        Args:
            model: Loaded Qwen3-VL model
            processor: Qwen3-VL processor for tokenization
            pooling_strategy: How to pool hidden states into single vector
            normalize: Whether to L2-normalize embeddings
        """
        self.model = model
        self.processor = processor
        self.pooling_strategy = pooling_strategy
        self.normalize = normalize
        self.device = model.device
        
        # Get embedding dimension from model config
        self.embedding_dim = model.config.hidden_size
        print(f"Embedding dimension: {self.embedding_dim}")
    
    def embed_text(self, text: str) -> np.ndarray:
        """
        Generate embedding for text only.
        
        Args:
            text: Input text to embed
            
        Returns:
            numpy array of shape (embedding_dim,)
        """
        return self._extract_embedding(text=text, image=None)
    
    def embed_multimodal(self, text: str, image: Image.Image) -> np.ndarray:
        """
        Generate embedding for text + image pair.
        
        Args:
            text: Input text
            image: PIL Image
            
        Returns:
            numpy array of shape (embedding_dim,)
        """
        return self._extract_embedding(text=text, image=image)
    
    def embed_batch(
        self, 
        items: List[Dict[str, Any]], 
        batch_size: int = 8
    ) -> List[np.ndarray]:
        """
        Generate embeddings for multiple items in batches.
        
        Args:
            items: List of dicts with 'text' and optional 'image' keys
            batch_size: Number of items to process at once
            
        Returns:
            List of numpy arrays, one per item
        """
        embeddings = []
        
        for i in range(0, len(items), batch_size):
            batch = items[i:i + batch_size]
            batch_embeddings = self._extract_batch_embeddings(batch)
            embeddings.extend(batch_embeddings)
        
        return embeddings
    
    def _extract_embedding(
        self, 
        text: str, 
        image: Optional[Image.Image] = None
    ) -> np.ndarray:
        """
        Core method to extract hidden states from model.
        
        This uses the model's encoder to get contextualized representations,
        then pools them into a single vector.
        """
        # Prepare conversation format (Qwen3-VL expects chat format)
        content = [{"type": "text", "text": text}]
        if image is not None:
            content.append({"type": "image", "image": image})
        
        messages = [{"role": "user", "content": content}]
        
        # Process inputs
        inputs = self.processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=False,  # We don't need generation
            return_dict=True,
            return_tensors="pt"
        )
        
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # Extract hidden states without generating tokens
        with torch.no_grad():
            # Get model outputs with hidden states
            outputs = self.model(
                **inputs,
                output_hidden_states=True,
                return_dict=True
            )
            
            # Extract last hidden state
            # Shape: (batch_size, sequence_length, hidden_size)
            hidden_states = outputs.hidden_states[-1]  # Last layer
            
            # Pool to single vector
            embedding = self._pool_hidden_states(
                hidden_states, 
                inputs.get('attention_mask')
            )
        
        # Convert to numpy and normalize if needed
        embedding = embedding.cpu().numpy().squeeze()
        
        if self.normalize:
            embedding = embedding / (np.linalg.norm(embedding) + 1e-9)
        
        return embedding
    
    def _extract_batch_embeddings(
        self, 
        batch: List[Dict[str, Any]]
    ) -> List[np.ndarray]:
        """
        Extract embeddings for a batch of items.
        Note: Batching with mixed modalities (some with images, some without)
        is tricky. For simplicity, we process items individually.
        """
        embeddings = []
        for item in batch:
            text = item.get('text', '')
            image = item.get('image', None)
            embedding = self._extract_embedding(text, image)
            embeddings.append(embedding)
        
        return embeddings
    
    def _pool_hidden_states(
        self, 
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Pool sequence of hidden states into single vector.
        
        Args:
            hidden_states: Shape (batch_size, seq_len, hidden_size)
            attention_mask: Shape (batch_size, seq_len)
            
        Returns:
            Pooled embedding of shape (batch_size, hidden_size)
        """
        if self.pooling_strategy == "mean":
            # Mean pooling over sequence length (excluding padding)
            if attention_mask is not None:
                # Expand mask to match hidden states dimensions
                mask_expanded = attention_mask.unsqueeze(-1).expand(hidden_states.size())
                sum_embeddings = torch.sum(hidden_states * mask_expanded, dim=1)
                sum_mask = torch.clamp(mask_expanded.sum(dim=1), min=1e-9)
                pooled = sum_embeddings / sum_mask
            else:
                pooled = torch.mean(hidden_states, dim=1)
                
        elif self.pooling_strategy == "cls":
            # Use first token (CLS token) representation
            pooled = hidden_states[:, 0, :]
            
        elif self.pooling_strategy == "max":
            # Max pooling over sequence
            pooled = torch.max(hidden_states, dim=1)[0]
            
        elif self.pooling_strategy == "last":
            # Use last non-padding token
            if attention_mask is not None:
                # Find last non-zero position for each batch item
                seq_lengths = attention_mask.sum(dim=1) - 1
                pooled = hidden_states[torch.arange(hidden_states.size(0)), seq_lengths]
            else:
                pooled = hidden_states[:, -1, :]
        else:
            raise ValueError(f"Unknown pooling strategy: {self.pooling_strategy}")
        
        return pooled
    
    def embed_from_base64(self, text: str, image_base64: Optional[str] = None) -> np.ndarray:
        """
        Convenience method for API endpoints that receive base64 images.
        
        Args:
            text: Input text
            image_base64: Base64-encoded image string
            
        Returns:
            Embedding vector
        """
        image = None
        if image_base64:
            try:
                image_data = base64.b64decode(image_base64)
                image = Image.open(io.BytesIO(image_data))
            except Exception as e:
                print(f"Warning: Failed to decode image: {e}")
        
        return self._extract_embedding(text, image)
    
    def get_embedding_dimension(self) -> int:
        """Return the dimensionality of embeddings."""
        return self.embedding_dim


class SentenceTransformerEmbeddings:
    """
    Alternative embedding service using sentence-transformers.
    Useful as a baseline or for text-only embeddings.
    """
    
    def __init__(
        self, 
        model_name: str = "all-MiniLM-L6-v2",
        device: Optional[str] = None
    ):
        """
        Initialize sentence-transformers embedding service.
        
        Args:
            model_name: HuggingFace model identifier
            device: Device to run on (cuda/cpu)
        """
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError:
            raise ImportError(
                "sentence-transformers not installed. "
                "Install with: pip install sentence-transformers"
            )
        
        self.model = SentenceTransformer(model_name, device=device)
        self.embedding_dim = self.model.get_sentence_embedding_dimension()
    
    def embed_text(self, text: str) -> np.ndarray:
        """Generate text embedding."""
        return self.model.encode(text, convert_to_numpy=True)
    
    def embed_batch(self, texts: List[str], batch_size: int = 32) -> List[np.ndarray]:
        """Generate embeddings for multiple texts."""
        embeddings = self.model.encode(
            texts, 
            batch_size=batch_size,
            convert_to_numpy=True,
            show_progress_bar=True
        )
        return list(embeddings)
    
    def get_embedding_dimension(self) -> int:
        """Return embedding dimension."""
        return self.embedding_dim


# Factory function to create appropriate embedding service
def create_embedding_service(
    strategy: str = "vlm2vec",
    model = None,
    processor = None,
    **kwargs
) -> Union[VLM2VecEmbeddings, SentenceTransformerEmbeddings]:
    """
    Factory to create embedding service.
    
    Args:
        strategy: "vlm2vec" for Qwen3-VL, "sentence-transformer" for text-only
        model: Qwen3-VL model (required for vlm2vec)
        processor: Qwen3-VL processor (required for vlm2vec)
        **kwargs: Additional arguments for embedding service
        
    Returns:
        Embedding service instance
    """
    if strategy == "vlm2vec":
        if model is None or processor is None:
            raise ValueError("model and processor required for vlm2vec strategy")
        return VLM2VecEmbeddings(model, processor, **kwargs)
    
    elif strategy == "sentence-transformer":
        return SentenceTransformerEmbeddings(**kwargs)
    
    else:
        raise ValueError(f"Unknown embedding strategy: {strategy}")
