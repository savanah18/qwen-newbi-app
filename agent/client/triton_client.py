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
    
    @staticmethod
    def debug_infer_result(result):
        """Debug helper to inspect InferResult structure."""
        print("=== InferResult Debug Info ===")
        print(f"Type: {type(result)}")
        print(f"Available methods: {[x for x in dir(result) if not x.startswith('_')]}")
        
        # Use get_response to get the underlying protobuf message
        try:
            proto_response = result.get_response()
            print(f"Proto response outputs: {[o.name for o in proto_response.outputs]}")
            
            # Try as_numpy for known outputs
            for output in proto_response.outputs:
                try:
                    print(output.name)
                    output_data = result.as_numpy(output.name)
                    print(f"\n  Output: {output.name}")
                    print(f"    Shape: {output_data.shape}")
                    print(f"    Dtype: {output_data.dtype}")
                    print(f"    Data: {output_data}")
                except Exception as e:
                    print(f"  Error getting {output.name}: {e}")
        except Exception as e:
            print(f"Error inspecting response: {e}")
    
    
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
        # Prepare inputs separately
        message_bytes = message.encode('utf-8')
        image_bytes = None
        
        if image_path:
            try:
                with open(image_path, "rb") as f:
                    image_data = f.read()
                    image_bytes = base64.b64encode(image_data)
            except Exception as e:
                print(f"Warning: Failed to read image: {str(e)}")
        
        # Use appropriate client based on protocol
        InferInput = grpcclient.InferInput if self.use_grpc else httpclient.InferInput
        
        # Create input tensors with batch dimension
        message_input = InferInput("message", [1, 1], "BYTES")
        message_input.set_data_from_numpy(np.array([[message_bytes]], dtype=object))
        
        inputs = [message_input]
        
        # Add image input if provided
        if image_bytes:
            image_input = InferInput("image", [1, 1], "BYTES")
            image_input.set_data_from_numpy(np.array([[image_bytes]], dtype=object))
            inputs.append(image_input)
        
        # Request inference
        try:
            response = self.client.infer(
                model_name=self.model_name,
                inputs=inputs
            )
            # Uncomment below to debug response structure
            # self.debug_infer_result(response)
            
            # Extract response - handle various numpy output formats
            response_array = response.as_numpy("response")
            response_time_array = response.as_numpy("response_time")
            
            # Get first element regardless of shape
            response_bytes = response_array.flat[0]
            if isinstance(response_bytes, np.bytes_):
                response_bytes = bytes(response_bytes)
            response_text = response_bytes.decode('utf-8')
            response_time = float(response_time_array.flat[0])
            
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
    import sys
    from pathlib import Path
    
    print("=" * 70)
    print("Triton Qwen3-VL Inference Tests")
    print("=" * 70)
    
    # Initialize client (use HTTP by default, can switch to gRPC)
    use_grpc = "--grpc" in sys.argv
    port = "8001" if use_grpc else "8000"
    
    client = TritonGrpcClient(f"localhost:{port}") if use_grpc else TritonHttpClient(f"localhost:{port}")
    print(f"Using {'gRPC' if use_grpc else 'HTTP'} protocol on port {port}")
    
    # Health check
    print("\n[1/4] Health Check...")
    if not client.check_health():
        print("❌ Triton server is not healthy!")
        sys.exit(1)
    print("✓ Triton server is healthy!")
    
    # Test 1: Text-only inference
    print("\n[2/4] Test: Text-only DSA Question")
    print("-" * 70)
    try:
        question = "Hi, are you there?"
        print(f"Query: {question}")
        response, response_time = client.chat(question)
        print(f"\nResponse:\n{response}")
        print(f"\n⏱️  Response Time: {response_time:.2f}s")
    except Exception as e:
        print(f"❌ Error: {str(e)}")
    
    # Test 2: Image inference with VS Code extension screenshot
    print("\n[3/4] Test: Multimodal (Text + Image) - VS Code Extension")
    print("-" * 70)
    image_path = Path("docs/figures/poc-vscode-ext.png")
    if image_path.exists():
        try:
            question = "What do you see in this image? Describe the VS Code interface shown."
            print(f"Query: {question}")
            print(f"Image: {image_path}")
            response, response_time = client.chat(question, str(image_path))
            print(f"\nResponse:\n{response}")
            print(f"\n⏱️  Response Time: {response_time:.2f}s")
        except Exception as e:
            print(f"❌ Error: {str(e)}")
    else:
        print(f"⚠️  Image not found: {image_path}")
    
    # Test 3: Image inference with POC screenshot
    print("\n[4/4] Test: Multimodal (Text + Image) - POC Interface")
    print("-" * 70)
    image_path = Path("docs/figures/poc.png")
    if image_path.exists():
        try:
            question = "Analyze this interface. What features and components can you identify?"
            print(f"Query: {question}")
            print(f"Image: {image_path}")
            response, response_time = client.chat(question, str(image_path))
            print(f"\nResponse:\n{response}")
            print(f"\n⏱️  Response Time: {response_time:.2f}s")
        except Exception as e:
            print(f"❌ Error: {str(e)}")
    else:
        print(f"⚠️  Image not found: {image_path}")
    
    print("\n" + "=" * 70)
    print("Tests Complete!")
    print("=" * 70)
    print("\nUsage:")
    print("  python agent/client/triton_client.py          # Use HTTP (port 8000)")
    print("  python agent/client/triton_client.py --grpc   # Use gRPC (port 8001)")
