"""
Gradio app with chat and image interface using Qwen3-VL model.
Uses the aiops-py312 conda environment.
"""

import gradio as gr
from PIL import Image
from pathlib import Path
from datetime import datetime
from typing import Optional
import torch
from transformers import (
    Qwen3VLForConditionalGeneration,
    AutoProcessor,
    BitsAndBytesConfig
)

# Model configuration
MODEL_PATH = Path("/root/workspace/lnd/aiops/vlm/Qwen/Qwen3-VL-8B-Instruct")
MAX_HISTORY = 5
MAX_NEW_TOKENS = 512

# Global variables for model and processor
model = None
processor = None
conversation_history = []  # For Qwen model
gradio_chat_history = []   # For Gradio UI (messages format)


def load_model():
    """Load the Qwen3-VL model with quantization."""
    global model, processor
    
    if model is not None:
        return "Model already loaded"
    
    print("Loading Qwen3-VL model...")
    
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
    
    print("Model loaded successfully!")
    return "Model loaded successfully"


def process_chat_and_image(
    message: str,
    image: Optional[Image.Image] = None,
    chat_history: list = None
) -> tuple:
    """
    Process chat message and image input using Qwen3-VL model.
    
    Args:
        message: User's chat message
        image: Optional image uploaded by user
        chat_history: Conversation history (Gradio format - messages)
    
    Returns:
        Updated chat history and response
    """
    global model, processor, conversation_history, gradio_chat_history
    
    if model is None or processor is None:
        error_response = "Error: Model not loaded. Please load the model first."
        if chat_history is None:
            chat_history = []
        chat_history.append({"role": "user", "content": message})
        chat_history.append({"role": "assistant", "content": error_response})
        gradio_chat_history = chat_history
        return chat_history, ""
    
    if not message.strip():
        error_response = "Error: Please enter a message"
        if chat_history is None:
            chat_history = []
        chat_history.append({"role": "user", "content": message})
        chat_history.append({"role": "assistant", "content": error_response})
        gradio_chat_history = chat_history
        return chat_history, ""
    
    if chat_history is None:
        chat_history = []
    
    start_time = datetime.now()
    
    try:
        # Build the content with text and optional image
        content = [{"type": "text", "text": message}]
        
        if image is not None:
            content.append({"type": "image"})
        
        # Add user message to internal conversation history (Qwen format)
        conversation_history.append({
            "role": "user",
            "content": content
        })
        
        # Slide context: keep only recent messages
        if len(conversation_history) > MAX_HISTORY:
            conversation_history = conversation_history[-MAX_HISTORY:]
        
        # Prepare inputs - handle image if provided
        if image is not None:
            inputs = processor.apply_chat_template(
                conversation_history,
                images=[image],
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
        end_time = datetime.now()
        response_time = (end_time - start_time).total_seconds()
        
        # Add assistant response to internal conversation history
        conversation_history.append({
            "role": "assistant",
            "content": [{"type": "text", "text": response}]
        })
        
        # Add timing info to display
        display_response = f"{response}\n\n⏱️ Response time: {response_time:.2f}s"
        
        # Add to Gradio chat history (messages format)
        chat_history.append({"role": "user", "content": message})
        chat_history.append({"role": "assistant", "content": display_response})
        gradio_chat_history = chat_history
        
        return chat_history, ""
    
    except Exception as e:
        error_msg = f"Error processing request: {str(e)}"
        print(f"Error: {error_msg}")
        import traceback
        traceback.print_exc()
        chat_history.append({"role": "user", "content": message})
        chat_history.append({"role": "assistant", "content": error_msg})
        gradio_chat_history = chat_history
        return chat_history, ""


def clear_chat():
    """Clear chat history."""
    global conversation_history, gradio_chat_history
    conversation_history = []
    gradio_chat_history = []
    return []


def main():
    """Create and launch the Gradio interface."""
    global model, processor
    
    with gr.Blocks(title="Qwen3-VL Chat & Image App") as app:
        gr.Markdown("# Qwen3-VL Chat & Image Processing")
        gr.Markdown("An advanced multimodal AI assistant that understands both text and images.")
        
        # Model status
        with gr.Row():
            model_status = gr.Textbox(label="Model Status", value="Not loaded", interactive=False)
            load_model_btn = gr.Button("Load Model", variant="primary")
        
        gr.Markdown("---")
        
        with gr.Row():
            with gr.Column(scale=3):
                # Chat interface
                chatbot = gr.Chatbot(
                    label="Chat History",
                    height=500,
                    show_label=True
                )
                
            with gr.Column(scale=1):
                # Image interface
                image_input = gr.Image(
                    type="pil",
                    label="Upload Image",
                    height=500
                )
        
        # Input section
        with gr.Row():
            message_input = gr.Textbox(
                placeholder="Enter your message... You can also upload an image to describe it.",
                label="Message",
                lines=3
            )
        
        with gr.Row():
            submit_btn = gr.Button("Send", variant="primary", scale=2)
            clear_btn = gr.Button("Clear Chat", scale=1)
        
        # Model loading
        def load_model_clicked():
            status = load_model()
            return status
        
        load_model_btn.click(
            fn=load_model_clicked,
            outputs=[model_status]
        )
        
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
        
        # Example usage
        gr.Examples(
            examples=[
                ["What can you do?", None],
                ["Hello! How are you today?", None],
                ["Can you help me understand this image?", None],
            ],
            inputs=[message_input, image_input]
        )
        
        gr.Markdown("---")
        gr.Markdown(
            "**Note:** Load the model first by clicking 'Load Model'. "
            "The model supports both text and image inputs for multimodal understanding."
        )
    
    # Launch the app
    app.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        show_error=True,
        theme=gr.themes.Soft()
    )


if __name__ == "__main__":
    main()
