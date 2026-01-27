"""
Inference engine for processing chat requests.
Handles both native transformers and vLLM inference.
"""

from typing import Optional, List, Dict, Any
import base64
import io
from PIL import Image
from fastapi import HTTPException


class InferenceEngine:
    """Handles inference for different loading strategies."""
    
    def __init__(
        self,
        loading_strategy: str,
        max_new_tokens: int = 512,
        temperature: float = 0.7,
        top_p: float = 0.9
    ):
        self.loading_strategy = loading_strategy
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.top_p = top_p
    
    async def process_native(
        self,
        message: str,
        image_base64: Optional[str],
        model,
        processor,
        conversation_history: List[Dict[str, Any]]
    ) -> str:
        """Process chat using native transformers."""
        content = [{"type": "text", "text": message}]
        
        # Decode image if provided
        image = None
        if image_base64:
            try:
                image_data = base64.b64decode(image_base64)
                image = Image.open(io.BytesIO(image_data))
                content.append({"type": "image", "image": image})
            except Exception as e:
                raise HTTPException(status_code=400, detail=f"Invalid image data: {str(e)}")
        
        # Add user message to history
        conversation_history.append({
            "role": "user",
            "content": content
        })
        
        # Prepare inputs
        inputs = processor.apply_chat_template(
            conversation_history,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt"
        )
        
        inputs = inputs.to(model.device)
        
        # Inference
        generated_ids = model.generate(
            **inputs,
            max_new_tokens=self.max_new_tokens,
            do_sample=True,
            temperature=self.temperature,
            top_p=self.top_p
        )
        
        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        
        output_text = processor.batch_decode(
            generated_ids_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False
        )
        
        response = output_text[0]
        
        # Add assistant response to history
        conversation_history.append({
            "role": "assistant",
            "content": [{"type": "text", "text": response}]
        })
        
        return response
    
    async def process_vllm(
        self,
        message: str,
        image_base64: Optional[str],
        llm,
        conversation_history: List[Dict[str, Any]],
        max_history: int
    ) -> str:
        """Process chat using vLLM."""
        try:
            from vllm import SamplingParams
        except ImportError:
            raise ImportError("vLLM not installed")
        
        # Build the prompt with conversation history
        messages = []
        
        # Add conversation history
        for hist in conversation_history[-max_history:]:
            messages.append(hist)
        
        # Add current user message
        content = [{"type": "text", "text": message}]
        
        # Decode and add image if provided
        image_data_list = []
        if image_base64:
            try:
                image_data = base64.b64decode(image_base64)
                image = Image.open(io.BytesIO(image_data))
                content.append({"type": "image"})
                image_data_list.append(image)
            except Exception as e:
                raise HTTPException(status_code=400, detail=f"Invalid image data: {str(e)}")
        
        messages.append({
            "role": "user",
            "content": content
        })
        
        # Apply chat template using vLLM's tokenizer
        prompt = llm.get_tokenizer().apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        
        # Configure sampling parameters
        sampling_params = SamplingParams(
            max_tokens=self.max_new_tokens,
            temperature=self.temperature,
            top_p=self.top_p,
            stop_token_ids=None
        )
        
        # Run inference with vLLM
        if image_data_list:
            outputs = llm.generate(
                {"prompt": prompt, "multi_modal_data": {"image": image_data_list}},
                sampling_params=sampling_params
            )
        else:
            outputs = llm.generate(prompt, sampling_params=sampling_params)
        
        # Extract response
        response = outputs[0].outputs[0].text.strip()
        
        # Update conversation history
        conversation_history.append({
            "role": "user",
            "content": content
        })
        
        conversation_history.append({
            "role": "assistant",
            "content": [{"type": "text", "text": response}]
        })
        
        return response
