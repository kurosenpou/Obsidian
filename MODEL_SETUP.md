# Model Setup Guide

This guide explains how to set up the AI models for the Obsidian AI MVP application.

## Overview

The application supports three AI models:
1. **Llama3** - General purpose conversational AI
2. **matscibert** (m3rg-iitd/matscibert) - Materials science BERT model
3. **MaterialsBERT** (pranav-s/MaterialsBERT) - Alternative materials science BERT

## Quick Start

The app will work out of the box by downloading models from Hugging Face when needed. However, for better performance and offline usage, you can download models locally.

## Local Model Setup (Recommended)

### Prerequisites

1. Install required dependencies:
```bash
pip install -r requirements_mvp.txt
```

2. Set up Hugging Face token (optional, for faster downloads):
```bash
# Windows
set HF_TOKEN=your_hugging_face_token_here

# Linux/Mac
export HF_TOKEN=your_hugging_face_token_here
```

### Download Models Locally

Create the models directory structure:

```
models/
├── matscibert/           # m3rg-iitd/matscibert
└── MaterialsBERT/        # pranav-s/MaterialsBERT
```

#### Option 1: Using Hugging Face CLI (Recommended)

```bash
# Install huggingface-hub
pip install huggingface-hub

# Download matscibert
huggingface-cli download m3rg-iitd/matscibert --local-dir ./models/matscibert

# Download MaterialsBERT
huggingface-cli download pranav-s/MaterialsBERT --local-dir ./models/MaterialsBERT
```

#### Option 2: Using Python Script

Create a script to download models:

```python
from transformers import AutoTokenizer, AutoModel, BertTokenizer, BertForMaskedLM

# Download matscibert
tokenizer = AutoTokenizer.from_pretrained("m3rg-iitd/matscibert")
model = AutoModel.from_pretrained("m3rg-iitd/matscibert")
tokenizer.save_pretrained("./models/matscibert")
model.save_pretrained("./models/matscibert")

# Download MaterialsBERT
tokenizer = BertTokenizer.from_pretrained("pranav-s/MaterialsBERT")
model = BertForMaskedLM.from_pretrained("pranav-s/MaterialsBERT")
tokenizer.save_pretrained("./models/MaterialsBERT")
model.save_pretrained("./models/MaterialsBERT")
```

#### Option 3: Using Git LFS

```bash
# Install git-lfs if not already installed
git lfs install

# Clone matscibert
git clone https://huggingface.co/m3rg-iitd/matscibert ./models/matscibert

# Clone MaterialsBERT
git clone https://huggingface.co/pranav-s/MaterialsBERT ./models/MaterialsBERT
```

## Model File Structure

After downloading, your models directory should look like:

```
models/
├── matscibert/
│   ├── config.json
│   ├── model.safetensors
│   ├── pytorch_model.bin
│   ├── tokenizer_config.json
│   ├── tokenizer.json
│   ├── vocab.txt
│   └── README.md
└── MaterialsBERT/
    ├── config.json
    ├── pytorch_model.bin
    ├── vocab.txt
    ├── training_DOI.txt
    └── README.md
```

## GPU Support

The application automatically detects and uses GPU if available:

- **CUDA GPU**: Models will run in mixed precision (float16) for better performance
- **CPU Only**: Models will run in float32 with good performance on modern CPUs

## Llama3 Setup

For Llama3, you have several options:

1. **Ollama** (Recommended for local use):
   ```bash
   # Install Ollama from https://ollama.ai
   ollama pull llama3
   ```

2. **Hugging Face Transformers**:
   ```bash
   # Download Llama3 model (requires approval)
   huggingface-cli download meta-llama/Llama-3-8B-Instruct --local-dir ./models/llama3
   ```

3. **Cloud API**: Use OpenAI-compatible API endpoints

## Troubleshooting

### Model Loading Issues

1. **OutOfMemoryError**: Reduce batch size or switch to CPU
2. **Network Issues**: Download models locally using the methods above
3. **Permission Errors**: Ensure you have access to the Hugging Face models

### Performance Tips

1. **Use GPU**: Significant speedup for model inference
2. **Local Models**: Faster loading and offline usage
3. **Model Caching**: Models are cached after first download

### Common Errors

1. **"Model not found"**: The app will automatically fallback to online download
2. **"CUDA out of memory"**: Switch to CPU mode or use a smaller model
3. **"Token limit exceeded"**: Break down your input into smaller chunks

## File Sizes

Be aware of model file sizes:

- **matscibert**: ~440MB
- **MaterialsBERT**: ~440MB
- **Llama3-8B**: ~8GB (if downloaded locally)

## Security Notes

- Models are excluded from git tracking (see `.gitignore`)
- Use environment variables for API tokens
- Never commit large model files to version control

## Support

If you encounter issues:

1. Check the console output for detailed error messages
2. Ensure all dependencies are installed
3. Verify model files are correctly downloaded
4. Check available disk space and memory
