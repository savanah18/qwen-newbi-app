"""
Gradio frontend for Qwen3-VL Chat application.
Communicates with model_server via REST API.
"""

import gradio as gr
from PIL import Image
from typing import Optional
import requests
import base64
import io
import time

# Model server configuration
MODEL_SERVER_URL = "http://localhost:8000"
LOAD_MODEL_ENDPOINT = f"{MODEL_SERVER_URL}/load_model"
CHAT_ENDPOINT = f"{MODEL_SERVER_URL}/chat"
CLEAR_HISTORY_ENDPOINT = f"{MODEL_SERVER_URL}/clear_history"
HEALTH_ENDPOINT = f"{MODEL_SERVER_URL}/health"

# Global state
chat_history = []


def check_server_health():
    """Check if model server is running."""
    try:
        response = requests.get(HEALTH_ENDPOINT, timeout=5)
        if response.status_code == 200:
            data = response.json()
            return data["model_loaded"], data["status"]
        return False, "Server error"
    except requests.exceptions.ConnectionError:
        return False, "Server not running"
    except Exception as e:
        return False, f"Error: {str(e)}"


def load_model_remote():
    """Load model on the remote server."""
    try:
        response = requests.post(LOAD_MODEL_ENDPOINT, json={}, timeout=300)
        if response.status_code == 200:
            data = response.json()
            return data["message"]
        return f"Error: {response.json().get('detail', 'Unknown error')}"
    except requests.exceptions.Timeout:
        return "Error: Model loading timed out"
    except requests.exceptions.ConnectionError:
        return "Error: Cannot connect to model server"
    except Exception as e:
        return f"Error: {str(e)}"


def process_chat_and_image(
    message: str,
    image: Optional[Image.Image] = None,
    chat_history: list = None
) -> tuple:
    """
    Send chat message and image to model server.
    
    Args:
        message: User's chat message
        image: Optional image
        chat_history: Conversation history
    
    Returns:
        Updated chat history and empty input
    """
    if chat_history is None:
        chat_history = []
    
    if not message.strip():
        error_msg = "Error: Please enter a message"
        chat_history.append({"role": "user", "content": message})
        chat_history.append({"role": "assistant", "content": error_msg})
        return chat_history, ""
    
    # Check server health
    model_loaded, status = check_server_health()
    if not model_loaded:
        error_msg = "Error: Model not loaded on server. Click 'Load Model' first."
        chat_history.append({"role": "user", "content": message})
        chat_history.append({"role": "assistant", "content": error_msg})
        return chat_history, ""
    
    try:
        # Prepare image data
        image_base64 = None
        if image is not None:
            buffered = io.BytesIO()
            image.save(buffered, format="PNG")
            image_base64 = base64.b64encode(buffered.getvalue()).decode()
        
        # Send request to model server
        start_time = time.time()
        response = requests.post(
            CHAT_ENDPOINT,
            json={
                "message": message,
                "image_base64": image_base64
            },
            timeout=120
        )
        
        if response.status_code == 200:
            data = response.json()
            response_text = data["response"]
            response_time = data["response_time"]
            
            # Format response with timing
            display_response = f"{response_text}\n\n⏱️ Response time: {response_time:.2f}s"
            
            # Add to chat history
            chat_history.append({"role": "user", "content": message})
            chat_history.append({"role": "assistant", "content": display_response})
            
            return chat_history, ""
        else:
            error_msg = f"Server error: {response.json().get('detail', 'Unknown error')}"
            chat_history.append({"role": "user", "content": message})
            chat_history.append({"role": "assistant", "content": error_msg})
            return chat_history, ""
    
    except requests.exceptions.Timeout:
        error_msg = "Error: Request timed out"
        chat_history.append({"role": "user", "content": message})
        chat_history.append({"role": "assistant", "content": error_msg})
        return chat_history, ""
    except requests.exceptions.ConnectionError:
        error_msg = "Error: Cannot connect to model server. Make sure it's running."
        chat_history.append({"role": "user", "content": message})
        chat_history.append({"role": "assistant", "content": error_msg})
        return chat_history, ""
    except Exception as e:
        error_msg = f"Error: {str(e)}"
        chat_history.append({"role": "user", "content": message})
        chat_history.append({"role": "assistant", "content": error_msg})
        return chat_history, ""


def clear_chat():
    """Clear chat history on both frontend and server."""
    global chat_history
    try:
        requests.post(CLEAR_HISTORY_ENDPOINT, timeout=5)
    except:
        pass
    chat_history = []
    return []


def main():
    """Create and launch the Gradio interface."""
    global chat_history
    
    with gr.Blocks(title="Qwen3-VL Chat") as app:
        gr.Markdown("# Qwen3-VL Chat & Image Processing")
        gr.Markdown("Chat with Qwen3-VL model via REST API")
        
        # Server status
        with gr.Row():
            server_status = gr.Textbox(
                label="Server Status",
                value="Checking...",
                interactive=False
            )
            load_model_btn = gr.Button("Load Model", variant="primary")
        
        gr.Markdown("---")
        
        with gr.Row():
            with gr.Column(scale=3):
                chatbot = gr.Chatbot(
                    label="Chat History",
                    height=500,
                    show_label=True
                )
            
            with gr.Column(scale=1):
                image_input = gr.Image(
                    type="pil",
                    label="Upload Image",
                    height=500
                )
        
        # Input section
        with gr.Row():
            message_input = gr.Textbox(
                placeholder="Enter your message...",
                label="Message",
                lines=3
            )
        
        with gr.Row():
            submit_btn = gr.Button("Send", variant="primary", scale=2)
            clear_btn = gr.Button("Clear Chat", scale=1)
        
        # Load model button
        def on_load_model():
            status = load_model_remote()
            return status
        
        load_model_btn.click(
            fn=on_load_model,
            outputs=[server_status]
        )
        
        # Check server status on load
        def check_status():
            model_loaded, status = check_server_health()
            return f"Status: {status} | Model: {'Loaded' if model_loaded else 'Not loaded'}"
        
        app.load(check_status, outputs=[server_status])
        
        # Chat submission
        submit_btn.click(
            fn=process_chat_and_image,
            inputs=[message_input, image_input, chatbot],
            outputs=[chatbot, message_input]
        )
        
        # Clear chat
        clear_btn.click(
            fn=clear_chat,
            outputs=[chatbot]
        )
        
        gr.Markdown("---")
        gr.Markdown(
            "**Note:** Model server must be running on http://localhost:8000\n\n"
            "Start it with: `python model_server.py`"
        )
    
    app.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        show_error=True,
        theme=gr.themes.Soft()
    )


if __name__ == "__main__":
    main()
