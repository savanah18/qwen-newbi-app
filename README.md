# Qwen3-VL Chat & Image Processing App

A simple multimodal AI chat application built with [Gradio](https://www.gradio.app/) and [Qwen3-VL](https://huggingface.co/Qwen/Qwen3-VL-8B-Instruct) that can process text messages and images.

> ⚠️ **Learning Opportunity**: This application is created for educational and learning purposes only. It demonstrates how to integrate multimodal AI models with web interfaces using Gradio.

## Features

- Text and image chat interface
- Gradio web UI
- Conversation history (last 5 messages)
- Response timing information

## Requirements

- **Python**: 3.12+
- **Conda Environment**: `aiops-py312`
- **GPU**: NVIDIA GPU with sufficient VRAM (recommended 8GB+ for smooth operation)

## Dependencies

All dependencies are listed in `requirements.txt`:

## Setup

### 1. Activate Conda Environment

```bash
conda activate aiops-py312
```

### 2. Install Dependencies (if not already installed)

```bash
pip install -r requirements.txt
```

### 3. Verify Qwen Model Path

The app expects the Qwen3-VL model at:
```
/root/workspace/lnd/aiops/vlm/Qwen/Qwen3-VL-8B-Instruct
```

If your model is located elsewhere, update the `MODEL_PATH` variable in `app.py`:

```python
MODEL_PATH = Path("/path/to/your/Qwen3-VL-8B-Instruct")
```

## Usage

### Start the Application

```bash
conda run -n aiops-py312 python app.py
```

The app will start and display:
```
* Running on local URL:  http://0.0.0.0:7860
```

### Access the Web Interface

Open your browser and navigate to:
- **Local**: `http://localhost:7860`
- **Remote**: `http://<server-ip>:7860`

### Using the App

1. **Load the Model**: Click the "Load Model" button to initialize Qwen3-VL
   - This may take 30-60 seconds on first load
   - Status will update when complete

2. **Send Messages**: 
   - Type your message in the text input box
   - Optionally upload an image using the image upload panel
   - Click "Send" to process

3. **Chat Features**:
   - **Text Only**: Send text messages for general conversation
   - **With Image**: Upload an image and ask questions about it
   - **Conversation Context**: The app maintains the last 5 exchanges for context
   - **Clear Chat**: Click "Clear Chat" to reset the conversation

## Architecture

### Model Configuration

The app uses the following optimizations:

- **Quantization**: 4-bit quantization with BFloat16 compute dtype
- **Attention**: SDPA (Scaled Dot Product Attention) for faster inference
- **Device Mapping**: Automatic GPU/CPU distribution (`device_map="auto"`)

### Conversation Management

- **Internal History**: Uses Qwen-compatible message format for the model
- **Display History**: Uses Gradio messages format for the web UI
- **Context Sliding**: Keeps only the last 5 message pairs to reduce token count

## Configuration

Key parameters you can adjust in `app.py`:

```python
MAX_HISTORY = 5              # Number of recent message pairs to keep
MAX_NEW_TOKENS = 512         # Maximum tokens to generate per response
temperature = 0.7            # Sampling temperature (0.0-1.0)
top_p = 0.9                  # Nucleus sampling parameter
```

## Troubleshooting

### Model Not Loading
- Verify the model path exists
- Check GPU memory availability: `nvidia-smi`
- Ensure PyTorch is installed correctly

### Out of Memory (OOM)
- Reduce `MAX_NEW_TOKENS`
- Reduce `MAX_HISTORY`
- Try running on a machine with more VRAM

### Slow Responses
- Normal for first inference (warm-up)
- Subsequent responses should be faster
- Check GPU utilization: `nvidia-smi`

### Port Already in Use
- Change the port in the launch settings in `app.py`
- Or kill the process: `lsof -ti:7860 | xargs kill -9`

## Example Usage

### Text Conversation
```
User: What are the benefits of machine learning?
Assistant: [Detailed response about ML benefits with timing]
```

### Image Understanding
```
User: [Upload an image] What's in this image?
Assistant: [Description of the image content]
```

## Performance Notes

- **First Load**: 25-60 seconds (model initialization)
- **First Inference**: 10-20 seconds (warm-up)
- **Subsequent Inference**: 3-8 seconds (depends on response length)

Response times are displayed at the end of each message.

## File Structure

```
newbie-app/
├── app.py              # Main Gradio application
├── requirements.txt    # Python dependencies
└── README.md          # This file
```

## References

- [Qwen3-VL Model](https://huggingface.co/Qwen/Qwen3-VL-8B-Instruct)
- [Gradio Documentation](https://www.gradio.app/docs)
- [Hugging Face Transformers](https://huggingface.co/docs/transformers)
- [BitsAndBytes Quantization](https://github.com/TimDettmers/bitsandbytes)

## License

This project uses models and libraries under their respective licenses. Please refer to the model card and library documentation for details.

## Support

For issues or questions:
1. Check the Troubleshooting section
2. Review the error messages in the terminal
3. Verify all dependencies are installed correctly

---

**Happy chatting! 🚀**
