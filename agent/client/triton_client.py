"""
Triton client for Qwen3-VL inference.
Can be used instead of the FastAPI client for higher performance.
"""

import tritonclient.http as httpclient
import tritonclient.grpc as grpcclient
from typing import Optional, Tuple
import base64
import numpy as np
from pathlib import Path


class TritonClient:
    """Client for Triton Inference Server."""
    
    def __init__(
        self,
        triton_url: str = "localhost:8001",
        use_grpc: bool = False,
        model_name: str = "qwen3-vl"
    ):
        """
        Initialize Triton client.
        
        Args:
            triton_url: Triton server URL (host:port)
            use_grpc: Use gRPC protocol instead of HTTP
            model_name: Model name in Triton repository
        """
        self.triton_url = triton_url
        self.model_name = model_name
        self.use_grpc = use_grpc
        
        if use_grpc:
            self.client = grpcclient.InferenceServerClient(triton_url)
        else:
            self.client = httpclient.InferenceServerClient(triton_url)
        
        print(f"Connected to Triton at {triton_url}")
        print(f"Model: {model_name}")
    
    def check_health(self) -> bool:
        """Check if Triton server is healthy."""
        try:
            if self.use_grpc:
                return self.client.is_server_live()
            else:
                return self.client.is_server_live()
        except Exception as e:
            print(f"Health check failed: {str(e)}")
            return False
    
    def chat(
        self,
        message: str,
        image_path: Optional[str] = None
    ) -> Tuple[str, float]:
        """
        Send a chat message to Triton for inference.
        
        Args:
            message: Chat message
            image_path: Optional path to image file
            
        Returns:
            Tuple of (response, response_time)
        """
        # Prepare image if provided
        image_base64 = None
        if image_path:
            try:
                with open(image_path, "rb") as f:
                    image_data = f.read()
                    image_base64 = base64.b64encode(image_data).decode('utf-8')
            except Exception as e:
                print(f"Warning: Failed to read image: {str(e)}")
        
        # Create input tensors
        message_input = httpclient.InferInput(
            "message",
            [1],
            "BYTES"
        )
        message_input.set_data_from_numpy(
            np.array([message.encode('utf-8')], dtype=object)
        )
        
        inputs = [message_input]
        
        # Add image input if provided
        if image_base64:
            image_input = httpclient.InferInput(
                "image",
                [1],
                "BYTES"
            )
            image_input.set_data_from_numpy(
                np.array([image_base64.encode('utf-8')], dtype=object)
            )
            inputs.append(image_input)
        
        # Request inference
        try:
            response = self.client.infer(
                model_name=self.model_name,
                inputs=inputs
            )
            
            # Extract outputs
            response_text = response.as_numpy("response")[0].decode('utf-8')
            response_time = float(response.as_numpy("response_time")[0][0])
            
            return response_text, response_time
        
        except Exception as e:
            print(f"Inference failed: {str(e)}")
            raise


class TritonHttpClient(TritonClient):
    """HTTP-based Triton client."""
    
    def __init__(self, url: str = "localhost:8001", model_name: str = "qwen3-vl"):
        super().__init__(url, use_grpc=False, model_name=model_name)


class TritonGrpcClient(TritonClient):
    """gRPC-based Triton client (faster for low-latency)."""
    
    def __init__(self, url: str = "localhost:8001", model_name: str = "qwen3-vl"):
        super().__init__(url, use_grpc=True, model_name=model_name)


if __name__ == "__main__":
    # Example usage
    client = TritonHttpClient("localhost:8001")
    
    if client.check_health():
        print("Triton server is healthy!")
        
        # Example inference
        try:
            response, response_time = client.chat("What are data structures?")
            print(f"\nResponse: {response}")
            print(f"Response Time: {response_time:.2f}s")
        except Exception as e:
            print(f"Error: {str(e)}")
    else:
        print("Triton server is not healthy!")
