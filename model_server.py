"""
FastAPI backend for Qwen3-VL model inference.
Serves model operations via REST API.
"""

from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import Optional, List
from pathlib import Path
from datetime import datetime
import torch
from transformers import (
    Qwen3VLForConditionalGeneration,
    AutoProcessor,
    BitsAndBytesConfig
)
import io
from PIL import Image
import base64
import uvicorn

# Model configuration
MODEL_PATH = Path("/root/workspace/lnd/aiops/vlm/Qwen/Qwen3-VL-8B-Instruct")
MAX_HISTORY = 5
MAX_NEW_TOKENS = 512

# Global model and processor
model = None
processor = None
conversation_history = []

# Initialize FastAPI app
app = FastAPI(
    title="Qwen3-VL Model Server",
    description="Backend service for Qwen3-VL model inference",
    version="1.0.0"
)


@app.on_event("startup")
async def startup_event():
    """Load model on server startup."""
    global model, processor
    print("=" * 50)
    print("Qwen3-VL Model Server Starting Up")
    print("=" * 50)
    print("Loading model on startup...")
    
    try:
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4"
        )
        
        model = Qwen3VLForConditionalGeneration.from_pretrained(
            str(MODEL_PATH),
            quantization_config=quantization_config,
            device_map="auto",
            attn_implementation="sdpa",
        ).eval()
        
        processor = AutoProcessor.from_pretrained(str(MODEL_PATH))
        
        print("✓ Model loaded successfully on startup!")
        print("=" * 50)
    except Exception as e:
        print(f"❌ Failed to load model on startup: {str(e)}")
        print("=" * 50)


# Pydantic models
class LoadModelRequest(BaseModel):
    model_path: Optional[str] = None


class LoadModelResponse(BaseModel):
    status: str
    message: str


class ChatRequest(BaseModel):
    message: str
    image_base64: Optional[str] = None


class ChatResponse(BaseModel):
    response: str
    response_time: float
    history_length: int


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool


class ClearHistoryResponse(BaseModel):
    status: str
    message: str


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Check server health and model status."""
    return {
        "status": "ok",
        "model_loaded": model is not None and processor is not None
    }


@app.post("/load_model", response_model=LoadModelResponse)
async def load_model_endpoint(request: LoadModelRequest):
    """Load the Qwen3-VL model."""
    global model, processor
    
    if model is not None and processor is not None:
        return {
            "status": "already_loaded",
            "message": "Model is already loaded"
        }
    
    try:
        model_path = request.model_path or str(MODEL_PATH)
        print(f"Loading model from: {model_path}")
        
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4"
        )
        
        model = Qwen3VLForConditionalGeneration.from_pretrained(
            model_path,
            quantization_config=quantization_config,
            device_map="auto",
            attn_implementation="sdpa",
        ).eval()
        
        processor = AutoProcessor.from_pretrained(model_path)
        
        print("Model loaded successfully!")
        return {
            "status": "success",
            "message": "Model loaded successfully"
        }
    
    except Exception as e:
        error_msg = f"Failed to load model: {str(e)}"
        print(error_msg)
        raise HTTPException(status_code=500, detail=error_msg)


@app.post("/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest):
    """Process chat message with optional image."""
    global model, processor, conversation_history
    
    if model is None or processor is None:
        raise HTTPException(status_code=400, detail="Model not loaded. Call /load_model first.")
    
    if not request.message.strip():
        raise HTTPException(status_code=400, detail="Message cannot be empty")
    
    start_time = datetime.now()
    
    try:
        content = [{"type": "text", "text": request.message}]
        
        # Decode image if provided
        if request.image_base64:
            try:
                image_data = base64.b64decode(request.image_base64)
                image = Image.open(io.BytesIO(image_data))
                content.append({"type": "image", "image": image})

            except Exception as e:
                raise HTTPException(status_code=400, detail=f"Invalid image data: {str(e)}")
        else:
            pass
        
        # Build conversation - text and optional image
        # Processor handles image placement based on images parameter
        
        # Add user message to history (text only)
        conversation_history.append({
            "role": "user",
            "content": content
        })
        
        # Slide context
        if len(conversation_history) > MAX_HISTORY:
            conversation_history = conversation_history[-MAX_HISTORY:]
        
        # Prepare inputs - conditionally pass images like the reference implementation
        
        try:
            # Only pass images if we have them - don't pass None
            if image is not None:
                inputs = processor.apply_chat_template(
                    conversation_history,
                    tokenize=True,
                    add_generation_prompt=True,
                    return_dict=True,
                    return_tensors="pt"
                )
            else:
                inputs = processor.apply_chat_template(
                    conversation_history,
                    tokenize=True,
                    add_generation_prompt=True,
                    return_dict=True,
                    return_tensors="pt"
                )
            
        except TypeError as e:
            raise
        
        inputs = inputs.to(model.device)
        
        # Inference
        generated_ids = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=True,
            temperature=0.7,
            top_p=0.9
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
        
        end_time = datetime.now()
        response_time = (end_time - start_time).total_seconds()
        
        return {
            "response": response,
            "response_time": response_time,
            "history_length": len(conversation_history)
        }
    
    except HTTPException:
        raise
    except Exception as e:
        error_msg = f"Error processing chat: {str(e)}"
        print(f"Error details: {str(e)}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=error_msg)


@app.post("/clear_history", response_model=ClearHistoryResponse)
async def clear_history_endpoint():
    """Clear conversation history."""
    global conversation_history
    conversation_history = []
    return {
        "status": "success",
        "message": "Conversation history cleared"
    }


@app.get("/history")
async def get_history():
    """Get current conversation history."""
    return {
        "history": conversation_history,
        "length": len(conversation_history)
    }


if __name__ == "__main__":
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        log_level="info"
    )
