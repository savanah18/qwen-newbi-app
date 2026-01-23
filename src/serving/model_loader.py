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
    INT4 = "int4"      # BitsAndBytes 4-bit (native only)
    INT8 = "int8"      # BitsAndBytes 8-bit (native only)
    AWQ = "awq"        # Activation-aware Weight Quantization (vLLM, requires pre-quantized model)
    GPTQ = "gptq"      # GPTQ quantization (vLLM, requires pre-quantized model)
    FP8 = "fp8"        # FP8 quantization (vLLM, H100+ GPUs)
    SQUEEZELLM = "squeezellm"  # SqueezeLLM (vLLM, requires pre-quantized model)


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
        # vLLM supports: awq, gptq, squeezellm, fp8
        # AWQ, GPTQ, SqueezeLLM require pre-quantized models
        quantization = None
        if quantization_type in [QuantizationType.AWQ, QuantizationType.GPTQ, 
                                 QuantizationType.SQUEEZELLM]:
            quantization = quantization_type.value
            print(f"  - Quantization: {quantization}")
            print(f"  ⚠️  WARNING: {quantization.upper()} requires a pre-quantized model!")
            print(f"     Current model: {model_path}")
            print(f"     Make sure this is a {quantization.upper()}-quantized checkpoint.")
        elif quantization_type == QuantizationType.FP8:
            quantization = quantization_type.value
            print(f"  - Quantization: {quantization} (H100+ GPU required)")
        
        try:
            llm = LLM(
                model=model_path,
                gpu_memory_utilization=gpu_memory_utilization, ## Use 75% of available GP memory to KV cache
                tensor_parallel_size=tensor_parallel_size,
                trust_remote_code=True,
                max_model_len=2048,   # Reduced from 4096 - saves ~2-3GB GPU memory
                limit_mm_per_prompt={"image": 10},
                quantization=quantization,
                enforce_eager=True,   # Skip compilation - faster init (3 min vs 10+ min)
                max_num_seqs=4,       # Reduced from 16 - saves RAM and GPU memory
            )
        except Exception as e:
            error_msg = str(e)
            if "Cannot find the config file" in error_msg and quantization:
                raise ValueError(
                    f"Failed to load {quantization.upper()}-quantized model. "
                    f"The model at '{model_path}' does not appear to be quantized with {quantization.upper()}. "
                    f"\n\nTo use {quantization.upper()} quantization:"
                    f"\n1. Download a pre-quantized {quantization.upper()} model, or"
                    f"\n2. Quantize your model using the {quantization.upper()} method, or"
                    f"\n3. Set QUANTIZATION_TYPE=none to use the model without quantization"
                ) from e
            raise
        
        print("✓ vLLM engine initialized successfully!")
        return llm
