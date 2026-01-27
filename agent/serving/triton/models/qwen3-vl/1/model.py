"""
Triton Python backend for Qwen3-VL model.
Implements TritonPythonModel interface for Triton Inference Server.
"""

import triton_python_backend_utils as pb_utils
import json
import numpy as np
from typing import Optional, List, Dict, Any
import base64
import io
import time
from datetime import datetime
from pathlib import Path
import os
import torch
from transformers import (
    Qwen3VLForConditionalGeneration,
    AutoProcessor,
    BitsAndBytesConfig
)
from PIL import Image


class TritonPythonModel:
    """Triton Python backend model for Qwen3-VL inference."""
    
    def initialize(self, args):
        """Initialize the model when Triton starts."""
        print("=" * 50)
        print("Initializing Qwen3-VL Triton Model")
        print("=" * 50)
        
        self.model_config = json.loads(args['model_config'])
        self.model_name = args['model_name']
        
        # Get configuration from environment
        model_path = os.getenv(
            "MODEL_PATH", 
            "/root/workspace/lnd/aiops/vlm/Qwen/Qwen3-VL-8B-Instruct"
        )
        quantization_type = os.getenv("QUANTIZATION_TYPE", "int4")
        attention_impl = os.getenv("ATTENTION_IMPL", "sdpa")
        
        print(f"Model Path: {model_path}")
        print(f"Quantization: {quantization_type}")
        print(f"Attention Implementation: {attention_impl}")
        
        # Load model
        self.model, self.processor = self._load_model(
            model_path=model_path,
            quantization_type=quantization_type,
            attention_impl=attention_impl
        )
        
        # Initialize conversation history (per-instance state)
        self.conversation_history: List[Dict[str, Any]] = []
        self.max_history = int(os.getenv("MAX_HISTORY", "5"))
        self.max_new_tokens = int(os.getenv("MAX_NEW_TOKENS", "512"))
        self.temperature = float(os.getenv("TEMPERATURE", "0.7"))
        self.top_p = float(os.getenv("TOP_P", "0.9"))
        
        print("=" * 50)
        print("✓ Model initialized successfully!")
        print("=" * 50)
    
    def _load_model(self, model_path: str, quantization_type: str, attention_impl: str):
        """Load Qwen3-VL model with optional quantization."""
        print(f"Loading model with transformers...")
        
        # Configure quantization
        quantization_config = None
        if quantization_type == "int4":
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4"
            )
        elif quantization_type == "int8":
            quantization_config = BitsAndBytesConfig(
                load_in_8bit=True
            )
        
        model = Qwen3VLForConditionalGeneration.from_pretrained(
            model_path,
            quantization_config=quantization_config,
            device_map="auto",
            attn_implementation=attention_impl,
            torch_dtype=torch.bfloat16 if quantization_config is None else None
        ).eval()
        
        processor = AutoProcessor.from_pretrained(model_path)
        print("✓ Model loaded successfully!")
        
        return model, processor
    
    def execute(self, requests):
        """Process inference requests from Triton."""
        responses = []
        
        for request in requests:
            try:
                # Extract inputs
                message = pb_utils.get_input_tensor_by_name(request, "message")
                image_input = pb_utils.get_input_tensor_by_name(request, "image")
                
                # Convert to Python strings
                message_str = message.as_numpy()[0].decode('utf-8') if message else ""
                image_base64 = None
                if image_input is not None:
                    image_data = image_input.as_numpy()
                    if image_data.size > 0:
                        image_base64 = image_data[0].decode('utf-8')
                
                # Process inference
                start_time = time.time()
                response_text = self._inference(message_str, image_base64)
                response_time = time.time() - start_time
                
                # Prepare output tensors
                response_tensor = pb_utils.Tensor(
                    "response",
                    np.array([response_text.encode('utf-8')], dtype=object)
                )
                
                time_tensor = pb_utils.Tensor(
                    "response_time",
                    np.array([[response_time]], dtype=np.float32)
                )
                
                # Create response
                response = pb_utils.InferenceResponse(
                    output_tensors=[response_tensor, time_tensor]
                )
                responses.append(response)
                
            except Exception as e:
                print(f"Error in inference: {str(e)}")
                import traceback
                traceback.print_exc()
                
                # Return error response
                error_response = pb_utils.InferenceResponse(
                    output_tensors=[
                        pb_utils.Tensor(
                            "response",
                            np.array([f"Error: {str(e)}".encode('utf-8')], dtype=object)
                        ),
                        pb_utils.Tensor(
                            "response_time",
                            np.array([[0.0]], dtype=np.float32)
                        )
                    ]
                )
                responses.append(error_response)
        
        return responses
    
    def _inference(self, message: str, image_base64: Optional[str]) -> str:
        """Perform inference on the message."""
        if not message.strip():
            return "Error: Message cannot be empty"
        
        # Build content
        content = [{"type": "text", "text": message}]
        
        # Decode image if provided
        image = None
        if image_base64:
            try:
                image_data = base64.b64decode(image_base64)
                image = Image.open(io.BytesIO(image_data))
                content.append({"type": "image", "image": image})
            except Exception as e:
                print(f"Warning: Failed to decode image: {str(e)}")
        
        # Add user message to history
        self.conversation_history.append({
            "role": "user",
            "content": content
        })
        
        # Slide context window
        if len(self.conversation_history) > self.max_history:
            self.conversation_history = self.conversation_history[-self.max_history:]
        
        # Prepare inputs
        try:
            inputs = self.processor.apply_chat_template(
                self.conversation_history,
                tokenize=True,
                add_generation_prompt=True,
                return_dict=True,
                return_tensors="pt"
            )
        except Exception as e:
            print(f"Error preparing inputs: {str(e)}")
            return f"Error preparing inputs: {str(e)}"
        
        inputs = inputs.to(self.model.device)
        
        # Inference
        try:
            with torch.no_grad():
                generated_ids = self.model.generate(
                    **inputs,
                    max_new_tokens=self.max_new_tokens,
                    do_sample=True,
                    temperature=self.temperature,
                    top_p=self.top_p
                )
            
            generated_ids_trimmed = [
                out_ids[len(in_ids):] 
                for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
            ]
            
            output_text = self.processor.batch_decode(
                generated_ids_trimmed,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False
            )
            
            response = output_text[0]
        except Exception as e:
            print(f"Error during generation: {str(e)}")
            return f"Error during generation: {str(e)}"
        
        # Add assistant response to history
        self.conversation_history.append({
            "role": "assistant",
            "content": [{"type": "text", "text": response}]
        })
        
        return response
    
    def finalize(self):
        """Cleanup when Triton shuts down."""
        print("Finalizing Qwen3-VL model")
        if hasattr(self, 'model'):
            del self.model
        if hasattr(self, 'processor'):
            del self.processor
        self.conversation_history.clear()
