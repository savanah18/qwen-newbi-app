"""
Model loading logic for different strategies.
Handles both native transformers and vLLM loading.
"""

from pathlib import Path
from typing import Optional, Tuple
import torch
from transformers import (
    Qwen3VLForConditionalGeneration,
    AutoProcessor,
    BitsAndBytesConfig
)
from enum import Enum


class LoadingStrategy(str, Enum):
    NATIVE = "native"
    VLLM = "vllm"


class QuantizationType(str, Enum):
    NONE = "none"
    INT4 = "int4"
    INT8 = "int8"
    AWQ = "awq"
    GPTQ = "gptq"


class AttentionImplementation(str, Enum):
    EAGER = "eager"
    SDPA = "sdpa"
    FLASH_ATTENTION_2 = "flash_attention_2"


class ModelLoader:
    """Handles model loading for different strategies."""
    
    @staticmethod
    def load_native(
        model_path: str,
        quantization_type: QuantizationType,
        attention_impl: AttentionImplementation
    ) -> Tuple:
        """
        Load model using native transformers with optional quantization.
        
        Returns:
            Tuple of (model, processor)
        """
        print(f"Loading model with native transformers...")
        print(f"  - Quantization: {quantization_type.value}")
        print(f"  - Attention: {attention_impl.value}")
        
        # Configure quantization
        quantization_config = None
        if quantization_type == QuantizationType.INT4:
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4"
            )
        elif quantization_type == QuantizationType.INT8:
            quantization_config = BitsAndBytesConfig(
                load_in_8bit=True
            )
        
        model = Qwen3VLForConditionalGeneration.from_pretrained(
            model_path,
            quantization_config=quantization_config,
            device_map="auto",
            attn_implementation=attention_impl.value,
            torch_dtype=torch.bfloat16 if quantization_config is None else None
        ).eval()
        
        processor = AutoProcessor.from_pretrained(model_path)
        print("✓ Model loaded successfully with native transformers!")
        
        return model, processor
    
    @staticmethod
    def load_vllm(
        model_path: str,
        quantization_type: QuantizationType,
        gpu_memory_utilization: float,
        tensor_parallel_size: int
    ):
        """
        Load model using vLLM for high-performance inference.
        
        Returns:
            vLLM LLM instance
        """
        try:
            from vllm import LLM
        except ImportError:
            raise ImportError("vLLM not installed. Install with: pip install vllm")
        
        print(f"Initializing vLLM engine...")
        print(f"  - GPU Memory Utilization: {gpu_memory_utilization}")
        print(f"  - Tensor Parallel Size: {tensor_parallel_size}")
        
        # Configure vLLM quantization
        quantization = None
        if quantization_type in [QuantizationType.AWQ, QuantizationType.GPTQ]:
            quantization = quantization_type.value
        
        llm = LLM(
            model=model_path,
            gpu_memory_utilization=gpu_memory_utilization,
            tensor_parallel_size=tensor_parallel_size,
            trust_remote_code=True,
            max_model_len=4096,
            limit_mm_per_prompt={"image": 10},
            quantization=quantization
        )
        
        print("✓ vLLM engine initialized successfully!")
        return llm
