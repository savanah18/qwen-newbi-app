"""FastAPI backend for Qwen3-VL model inference.
REST API layer - business logic separated to model_loader and inference_engine.
"""

from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import Optional, List
from pathlib import Path
from datetime import datetime
import os
import uvicorn

# Import business logic
from model_loader import (
    ModelLoader,
    LoadingStrategy,
    QuantizationType,
    AttentionImplementation
)
from inference_engine import InferenceEngine

# Environment variables configuration
LOADING_STRATEGY = LoadingStrategy(os.getenv("LOADING_STRATEGY", "native"))
QUANTIZATION_TYPE = QuantizationType(os.getenv("QUANTIZATION_TYPE", "int4"))
ATTENTION_IMPL = AttentionImplementation(os.getenv("ATTENTION_IMPL", "sdpa"))
VLLM_GPU_MEMORY_UTILIZATION = float(os.getenv("VLLM_GPU_MEMORY_UTILIZATION", "0.75"))
VLLM_TENSOR_PARALLEL_SIZE = int(os.getenv("VLLM_TENSOR_PARALLEL_SIZE", "1"))

# Model configuration
MODEL_PATH = Path(os.getenv("MODEL_PATH", "/root/workspace/lnd/aiops/vlm/Qwen/Qwen3-VL-8B-Instruct"))
MAX_HISTORY = int(os.getenv("MAX_HISTORY", "5"))
MAX_NEW_TOKENS = int(os.getenv("MAX_NEW_TOKENS", "512"))

# Global model objects
model = None
processor = None
llm = None  # For vLLM
conversation_history = []

# Initialize inference engine
inference_engine = InferenceEngine(
    loading_strategy=LOADING_STRATEGY.value,
    max_new_tokens=MAX_NEW_TOKENS,
    temperature=0.7,
    top_p=0.9
)


def load_model_native(model_path: str):
    """Load model using native transformers - delegates to ModelLoader."""
    global model, processor
    model, processor = ModelLoader.load_native(
        model_path=model_path,
        quantization_type=QUANTIZATION_TYPE,
        attention_impl=ATTENTION_IMPL
    )


def load_model_vllm(model_path: str):
    """Load model using vLLM - delegates to ModelLoader."""
    global llm
    llm = ModelLoader.load_vllm(
        model_path=model_path,
        quantization_type=QUANTIZATION_TYPE,
        gpu_memory_utilization=VLLM_GPU_MEMORY_UTILIZATION,
        tensor_parallel_size=VLLM_TENSOR_PARALLEL_SIZE
    )


# Initialize FastAPI app
app = FastAPI(
    title="Qwen3-VL Model Server",
    description=f"Backend service for Qwen3-VL model inference (Strategy: {LOADING_STRATEGY.value})",
    version="1.0.0"
)


@app.on_event("startup")
async def startup_event():
    """Load model on server startup based on loading strategy."""
    global model, processor, llm
    print("=" * 50)
    print(f"Qwen3-VL Model Server Starting Up")
    print(f"Loading Strategy: {LOADING_STRATEGY.value.upper()}")
    print("=" * 50)
    
    try:
        if LOADING_STRATEGY == LoadingStrategy.NATIVE:
            load_model_native(str(MODEL_PATH))
        elif LOADING_STRATEGY == LoadingStrategy.VLLM:
            load_model_vllm(str(MODEL_PATH))
        else:
            raise ValueError(f"Unknown loading strategy: {LOADING_STRATEGY}")
        
        print("=" * 50)
    except Exception as e:
        print(f"❌ Failed to load model: {str(e)}")
        import traceback
        traceback.print_exc()
        print("=" * 50)


# Pydantic models
class LoadModelRequest(BaseModel):
    model_path: Optional[str] = None
    loading_strategy: Optional[LoadingStrategy] = None
    quantization_type: Optional[QuantizationType] = None


class LoadModelResponse(BaseModel):
    status: str
    message: str
    loading_strategy: str
    quantization: str


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
    loading_strategy: str
    quantization: str


class ClearHistoryResponse(BaseModel):
    status: str
    message: str


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Check server health and model status."""
    is_loaded = False
    if LOADING_STRATEGY == LoadingStrategy.NATIVE:
        is_loaded = model is not None and processor is not None
    elif LOADING_STRATEGY == LoadingStrategy.VLLM:
        is_loaded = llm is not None
    
    return {
        "status": "ok",
        "model_loaded": is_loaded,
        "loading_strategy": LOADING_STRATEGY.value,
        "quantization": QUANTIZATION_TYPE.value
    }


@app.post("/load_model", response_model=LoadModelResponse)
async def load_model_endpoint(request: LoadModelRequest):
    """Load the Qwen3-VL model with specified strategy."""
    global model, processor, llm, LOADING_STRATEGY, QUANTIZATION_TYPE
    
    # Check if model is already loaded
    is_loaded = (LOADING_STRATEGY == LoadingStrategy.NATIVE and model is not None) or \
                (LOADING_STRATEGY == LoadingStrategy.VLLM and llm is not None)
    
    if is_loaded:
        return {
            "status": "already_loaded",
            "message": "Model is already loaded",
            "loading_strategy": LOADING_STRATEGY.value,
            "quantization": QUANTIZATION_TYPE.value
        }
    
    try:
        model_path = request.model_path or str(MODEL_PATH)
        
        # Override strategy if provided
        strategy = request.loading_strategy or LOADING_STRATEGY
        quantization = request.quantization_type or QUANTIZATION_TYPE
        
        # Update global config if different
        if request.loading_strategy:
            LOADING_STRATEGY = strategy
        if request.quantization_type:
            QUANTIZATION_TYPE = quantization
        
        print(f"Loading model from: {model_path}")
        
        if strategy == LoadingStrategy.NATIVE:
            load_model_native(model_path)
        elif strategy == LoadingStrategy.VLLM:
            load_model_vllm(model_path)
        else:
            raise ValueError(f"Unknown loading strategy: {strategy}")
        
        return {
            "status": "success",
            "message": "Model loaded successfully",
            "loading_strategy": strategy.value,
            "quantization": quantization.value
        }
    
    except Exception as e:
        error_msg = f"Failed to load model: {str(e)}"
        print(error_msg)
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=error_msg)


@app.post("/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest):
    """Process chat message with optional image - delegates to InferenceEngine."""
    global model, processor, llm, conversation_history
    
    # Check if model is loaded
    if LOADING_STRATEGY == LoadingStrategy.NATIVE:
        if model is None or processor is None:
            raise HTTPException(status_code=400, detail="Model not loaded. Call /load_model first.")
    elif LOADING_STRATEGY == LoadingStrategy.VLLM:
        if llm is None:
            raise HTTPException(status_code=400, detail="vLLM engine not initialized. Call /load_model first.")
    
    if not request.message.strip():
        raise HTTPException(status_code=400, detail="Message cannot be empty")
    
    start_time = datetime.now()
    
    try:
        # Slide context window before processing
        if len(conversation_history) > MAX_HISTORY:
            conversation_history = conversation_history[-MAX_HISTORY:]
        
        # Delegate to inference engine
        if LOADING_STRATEGY == LoadingStrategy.NATIVE:
            response = await inference_engine.process_native(
                message=request.message,
                image_base64=request.image_base64,
                model=model,
                processor=processor,
                conversation_history=conversation_history
            )
        elif LOADING_STRATEGY == LoadingStrategy.VLLM:
            response = await inference_engine.process_vllm(
                message=request.message,
                image_base64=request.image_base64,
                llm=llm,
                conversation_history=conversation_history,
                max_history=MAX_HISTORY
            )
        else:
            raise ValueError(f"Unknown loading strategy: {LOADING_STRATEGY}")
        
        # Slide context window after processing (for vLLM)
        if LOADING_STRATEGY == LoadingStrategy.VLLM:
            if len(conversation_history) > MAX_HISTORY * 2:
                conversation_history = conversation_history[-(MAX_HISTORY * 2):]
        
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
