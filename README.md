# llm-local 🤖

> Run open-source language models locally with Python & Hugging Face - Privacy-first, production-ready LLM runner

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

## ✨ Features

- 🖥️ **4 Interfaces**: One-off script, Classic CLI, Rich Terminal UI, REST API
- ⚡ **Real-time Streaming**: Live token-by-token generation with visual feedback
- 🎨 **Beautiful UI**: Markdown rendering with syntax highlighting (200+ languages)
- 💾 **Persistent History**: Save conversations, search with Ctrl+R, auto-suggestions
- 🔧 **Highly Configurable**: Environment-based settings with smart defaults
- 🚀 **Production Ready**: FastAPI server with validation, error handling, token counting
- 🔒 **Privacy First**: 100% local execution, zero cloud dependencies
- 🎯 **Easy Setup**: Virtual environment, one command installation

## 🚀 Quick Start

### Installation

```bash
# Clone repository
git clone <your-repo-url>
cd llm-local

# Create and activate virtual environment
python3 -m venv .venv
source .venv/bin/activate  # On Linux/Mac/WSL

# Install dependencies
pip install -r requirements.txt

# Configure (optional - works with defaults)
cp .env.example .env
nano .env  # Edit if needed
```

### First Run

```bash
# Enhanced Rich UI (recommended)
./chat-rich.sh

# Or use the classic CLI
./chat.sh

# Or start the API server
./api.sh
```

## 📚 Usage Guide

### 1. Enhanced Rich CLI (Recommended)

Beautiful terminal interface with Claude Code-like experience.

```bash
./chat-rich.sh
```

**Features:**
- 🎨 Markdown rendering with monokai syntax highlighting
- ⚡ Streaming responses with live updates (10 FPS)
- ⌨️ Multiline input - press **Meta+Enter** (ESC+Enter) or **Alt+Enter** to submit
- 📜 Persistent history saved to `.llm_chat_history`
- 🔍 Search history with **Ctrl+R**
- 💡 Auto-suggestions from previous prompts
- 📤 Export conversations with `/export filename.md`
- 🎨 Rich panels and beautiful formatting

**Keyboard Shortcuts:**
- **Meta+Enter** / **Alt+Enter** - Submit message
- **Ctrl+R** - Fuzzy search history
- **Ctrl+C** - Cancel current input
- **Arrow Up/Down** - Navigate history

**Commands:**
- `/help` - Show help menu
- `/history` - View conversation history
- `/export [file]` - Export to markdown (default: conversation.md)
- `/reset` - Clear conversation
- `/exit` - Quit

**Example:**
```bash
# With custom system prompt
./chat-rich.sh --system-prompt "You are a Python expert and helpful coding assistant"

# With custom parameters
./chat-rich.sh --max-tokens 1024 --temperature 0.9
```

### 2. Classic CLI

Simple text-based interface for minimal environments.

```bash
./chat.sh
```

**Features:**
- Simple REPL interface
- Conversation history (in-memory, up to 50 messages)
- Basic commands (/help, /reset, /exit, /history)
- System prompt support
- Keyboard interrupt handling

**Example:**
```bash
./chat.sh --system-prompt "You are helpful" --no-history
```

### 3. REST API Server

Production-ready FastAPI server.

```bash
# Start server (default: http://0.0.0.0:8000)
./api.sh

# Custom host/port
python3 -m src.api_server
# or
uvicorn src.api_server:app --host 127.0.0.1 --port 8080 --reload
```

#### API Endpoints

**GET /** - API information
```bash
curl http://localhost:8000/
```
```json
{
  "name": "Local LLM API",
  "version": "0.1.0",
  "model": "Qwen/Qwen2.5-1.5B-Instruct",
  "status": "running"
}
```

**GET /health** - Health check
```bash
curl http://localhost:8000/health
```
```json
{
  "status": "healthy",
  "model_loaded": true
}
```

**GET /info** - Model details
```bash
curl http://localhost:8000/info
```
```json
{
  "model_id": "Qwen/Qwen2.5-1.5B-Instruct",
  "device": "cpu",
  "dtype": "torch.bfloat16",
  "num_parameters": 1543714304,
  "num_parameters_millions": 1543.71,
  "config": {...}
}
```

**POST /generate** - Text completion
```bash
curl -X POST http://localhost:8000/generate \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "Explain Python in simple terms",
    "max_new_tokens": 200,
    "temperature": 0.7
  }'
```
```json
{
  "output_text": "Python is a high-level programming language...",
  "prompt_tokens": 6,
  "completion_tokens": 45,
  "total_tokens": 51,
  "elapsed_seconds": 2.864
}
```

**POST /chat** - Chat completion
```bash
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [
      {"role": "system", "content": "You are a helpful coding assistant"},
      {"role": "user", "content": "How do I reverse a list in Python?"}
    ],
    "temperature": 0.7
  }'
```
```json
{
  "role": "assistant",
  "content": "To reverse a list in Python...",
  "elapsed_seconds": 1.523
}
```

### 4. One-off Generation Script

Single text generation without persistence.

```bash
# Direct prompt
./run.sh scripts/run_once.py --prompt "What is Docker?"

# From stdin
echo "Explain quantum computing" | ./run.sh scripts/run_once.py

# With options
python3 scripts/run_once.py \
  --prompt "Write a Python function to calculate fibonacci" \
  --max-tokens 300 \
  --temperature 0.8 \
  --verbose \
  --show-prompt
```

**Options:**
- `--prompt TEXT` - The prompt (or use stdin)
- `--system-prompt TEXT` - System instruction
- `--model MODEL_ID` - Override model
- `--max-tokens INT` - Max tokens to generate
- `--temperature FLOAT` - Sampling temperature (0.0-2.0)
- `--top-p FLOAT` - Nucleus sampling (0.0-1.0)
- `--verbose` - Enable debug logging
- `--show-prompt` - Display formatted prompt before generation

## ⚙️ Configuration

### Environment Variables

Create `.env` file (copy from `.env.example`):

```env
# Model Selection
MODEL_ID=Qwen/Qwen2.5-1.5B-Instruct

# Device Settings
USE_GPU=true              # Set to false for CPU-only
TORCH_DTYPE=auto          # Options: auto, float16, bfloat16, float32
DEVICE_MAP=auto           # Options: auto, cpu, cuda, cuda:0

# Generation Parameters
MAX_NEW_TOKENS=512        # Maximum tokens per generation
TEMPERATURE=0.7           # Sampling temperature (0.0-2.0)
TOP_P=0.9                 # Nucleus sampling (0.0-1.0)
TOP_K=50                  # Top-k sampling
DO_SAMPLE=true            # Enable sampling vs greedy

# API Server
API_HOST=0.0.0.0         # Bind address (use 127.0.0.1 for local-only)
API_PORT=8000            # Port number

# Logging
LOG_LEVEL=INFO           # Options: DEBUG, INFO, WARNING, ERROR
```

### Supported Models

The system works with any Hugging Face causal language model. Pre-tested models:

| Model | Size | VRAM | Speed | Use Case |
|-------|------|------|-------|----------|
| `Qwen/Qwen2.5-1.5B-Instruct` | 1.5B | ~3GB | Fast | General, CPU-friendly |
| `Qwen/Qwen2.5-7B-Instruct` | 7B | ~14GB | Medium | Advanced reasoning |
| `meta-llama/Llama-3.2-1B-Instruct` | 1B | ~2GB | Fastest | Quick responses |
| `microsoft/Phi-3.5-mini-instruct` | 3.8B | ~8GB | Medium | Balanced |

**Changing Models:**
```env
# In .env file
MODEL_ID=meta-llama/Llama-3.2-1B-Instruct

# Or via command line
./chat-rich.sh --model "Qwen/Qwen2.5-7B-Instruct"
```

### Recommended Configurations

**For CPU (8GB RAM):**
```env
MODEL_ID=Qwen/Qwen2.5-1.5B-Instruct
USE_GPU=false
DEVICE_MAP=cpu
TORCH_DTYPE=float32
MAX_NEW_TOKENS=256
```

**For GPU (6GB+ VRAM):**
```env
MODEL_ID=Qwen/Qwen2.5-1.5B-Instruct
USE_GPU=true
TORCH_DTYPE=float16
DEVICE_MAP=auto
MAX_NEW_TOKENS=512
```

**For GPU (14GB+ VRAM):**
```env
MODEL_ID=Qwen/Qwen2.5-7B-Instruct
USE_GPU=true
TORCH_DTYPE=float16
DEVICE_MAP=auto
MAX_NEW_TOKENS=1024
```

**For Creative Writing:**
```env
TEMPERATURE=0.9
TOP_P=0.95
DO_SAMPLE=true
```

**For Factual/Deterministic:**
```env
TEMPERATURE=0.1
TOP_P=0.9
DO_SAMPLE=true
```

## 🐍 Python API

### Basic Usage

```python
from src.config import config
from src.model_loader import load_tokenizer_and_model, get_model_info
from src.generator import TextGenerator

# Load model
tokenizer, model = load_tokenizer_and_model()

# Get model info
info = get_model_info(model)
print(f"Model: {info['num_parameters_millions']}M parameters")
print(f"Device: {info['device']}, Dtype: {info['dtype']}")

# Create generator
generator = TextGenerator(model, tokenizer)

# Generate text
response = generator.generate_text(
    prompt="Explain Python in simple terms",
    system_prompt="You are a helpful teacher",
    max_new_tokens=200,
    temperature=0.7
)
print(response)
```

### Chat Mode

```python
# Multi-turn conversation
messages = [
    {"role": "system", "content": "You are a helpful coding assistant"},
    {"role": "user", "content": "How do I reverse a list in Python?"}
]

response = generator.generate_chat(
    messages=messages,
    max_new_tokens=300,
    temperature=0.7
)
print(response)

# Add to conversation
messages.append({"role": "assistant", "content": response})
messages.append({"role": "user", "content": "Can you show an example?"})

response = generator.generate_chat(messages)
print(response)
```

### Streaming

```python
# Real-time streaming generation
for chunk in generator.generate_chat_stream(messages):
    print(chunk, end='', flush=True)
print()  # New line after streaming
```

### Configuration

```python
# Override config at runtime
config.MAX_NEW_TOKENS = 1024
config.TEMPERATURE = 0.9

# Or pass directly to generator
response = generator.generate_text(
    prompt="Write a story",
    max_new_tokens=2000,
    temperature=0.95,
    top_p=0.98
)
```

## 🏗️ Architecture

### Project Structure

```
llm-local/
├── src/                    # Source code
│   ├── __init__.py        # Package initialization
│   ├── config.py          # Configuration management
│   ├── model_loader.py    # Model/tokenizer loading
│   ├── generator.py       # Text generation engine
│   ├── chat_cli.py        # Classic CLI interface
│   ├── chat_cli_rich.py   # Enhanced Rich UI interface
│   └── api_server.py      # FastAPI REST API server
│
├── scripts/               # Utility scripts
│   └── run_once.py       # One-off generation script
│
├── tests/                # Test suite (to be added)
│
├── *.sh                  # Convenience launchers
├── .env.example          # Configuration template
├── .envrc                # direnv auto-activation
├── .gitignore            # Git ignore rules
├── LICENSE               # MIT License
├── requirements.txt      # Python dependencies
└── README.md            # This file
```

### Technology Stack

- **ML Framework**: PyTorch 2.0+
- **Model Hub**: Hugging Face Transformers, Accelerate
- **API**: FastAPI + Uvicorn + Pydantic
- **Terminal UI**: Rich + prompt-toolkit
- **Config**: python-dotenv
- **Testing**: pytest, pytest-asyncio, httpx

### Module Dependencies

```
config.py (standalone)
    ↓
model_loader.py → config
    ↓
generator.py → config
    ↓
├── chat_cli.py → config, model_loader, generator
├── chat_cli_rich.py → config, model_loader, generator
├── api_server.py → config, model_loader, generator
└── run_once.py → config, model_loader, generator
```

## 📋 Requirements

### System Requirements

- **Python**: 3.10 or higher
- **OS**: Linux, macOS, Windows (with WSL recommended)
- **RAM**: 8GB minimum, 16GB recommended
- **VRAM**: 
  - 3GB for 1.5B models
  - 8GB for 4B models
  - 14GB+ for 7B models
- **Disk**: 5-20GB for model storage
- **GPU**: Optional but recommended (CUDA-compatible NVIDIA GPU)

### Python Dependencies

Core dependencies (from `requirements.txt`):
- `torch>=2.0.0`
- `transformers>=4.35.0`
- `accelerate>=0.25.0`
- `fastapi>=0.104.0`
- `uvicorn[standard]>=0.24.0`
- `pydantic>=2.5.0`
- `python-dotenv>=1.0.0`
- `rich>=13.0.0`
- `prompt-toolkit>=3.0.0`

## 🐛 Troubleshooting

### GPU Not Detected

```bash
# Check CUDA availability
python3 -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
python3 -c "import torch; print(f'CUDA version: {torch.version.cuda}')"
python3 -c "import torch; print(f'GPU: {torch.cuda.get_device_name(0)}')"
```

**Solution**: If CUDA is not available:
1. Install CUDA toolkit for your system
2. Install PyTorch with CUDA support: `pip install torch --index-url https://download.pytorch.org/whl/cu118`
3. Or use CPU mode: Set `USE_GPU=false` in `.env`

### Out of Memory

**Symptoms**: `RuntimeError: CUDA out of memory` or system freezes

**Solutions**:
1. Use a smaller model (1.5B instead of 7B)
2. Reduce `MAX_NEW_TOKENS` in `.env` (try 256 or 128)
3. Use lower precision: `TORCH_DTYPE=float16` or `bfloat16`
4. Enable CPU fallback: `USE_GPU=false`
5. Close other GPU-using applications

### Slow Generation

**On CPU**: Generation is slower, expect 3-10 seconds per response for 1.5B model

**Solutions**:
1. Ensure GPU is enabled: `USE_GPU=true`
2. Use float16: `TORCH_DTYPE=float16`
3. Reduce output length: `MAX_NEW_TOKENS=256`
4. Use smaller model for faster responses
5. Check system load: `htop` or `nvidia-smi`

### Module Not Found Errors

**Error**: `ModuleNotFoundError: No module named 'dotenv'` or similar

**Solution**: Activate virtual environment first!
```bash
source .venv/bin/activate  # Linux/Mac/WSL
# Then run your command
```

**Or use wrapper scripts** (automatically activate venv):
```bash
./run.sh scripts/run_once.py --prompt "Hello"
```

### Model Download Issues

**Error**: Download fails or times out

**Solutions**:
1. Check internet connection
2. Try again (downloads resume automatically)
3. Set HuggingFace cache: `export HF_HOME=/path/to/cache`
4. Use HuggingFace token for gated models: `huggingface-cli login`

### Permission Errors

**Error**: Permission denied on `.sh` scripts

**Solution**:
```bash
chmod +x *.sh
```

## 🎯 Performance

### Benchmarks

Tested on: Intel i7 CPU, 16GB RAM, No GPU (CPU mode)

| Operation | Time | Tokens |
|-----------|------|--------|
| Model Loading | ~2-3 seconds | - |
| Generate (CPU) | ~3 seconds | ~10 tokens |
| Generate (API) | ~2.9 seconds | ~10 tokens |
| Streaming Start | ~0.5 seconds | First token |

With GPU (NVIDIA RTX 3060, 12GB):
| Operation | Time | Tokens |
|-----------|------|--------|
| Model Loading | ~5 seconds | - |
| Generate (GPU) | ~0.5 seconds | ~10 tokens |
| Tokens/second | ~50-100 | - |

### Optimization Tips

1. **Use GPU** if available (10-20x faster)
2. **Use float16** precision on GPU
3. **Keep model loaded** (use API server for multiple requests)
4. **Use smaller models** for faster responses
5. **Reduce max_tokens** for quicker completions

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

### Development Setup

```bash
# Clone and setup
git clone <repo-url>
cd llm-local
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Install development dependencies
pip install black isort autoflake pytest pytest-cov

# Run tests (when available)
pytest

# Format code
black src/ scripts/
isort src/ scripts/

# Remove unused imports
autoflake --remove-all-unused-imports --in-place src/*.py scripts/*.py
```

### Code Style

- Use **Black** for formatting (line length: 100)
- Use **isort** for import sorting
- Add type hints to all functions
- Write docstrings for public functions
- Keep functions focused and small

## 📄 License

MIT License - see [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- [Hugging Face](https://huggingface.co/) for Transformers library
- [Qwen Team](https://github.com/QwenLM/Qwen) for excellent open-source models
- [Rich](https://github.com/Textualize/rich) for beautiful terminal output
- [FastAPI](https://fastapi.tiangolo.com/) for modern Python APIs
- [prompt-toolkit](https://github.com/prompt-toolkit/python-prompt-toolkit) for advanced terminal input

## 📞 Support

- **Issues**: [GitHub Issues](https://github.com/yourusername/llm-local/issues)
- **Discussions**: [GitHub Discussions](https://github.com/yourusername/llm-local/discussions)

## 🗺️ Roadmap

See [PLAN.md](PLAN.md) for detailed development roadmap.

**Coming Soon:**
- Comprehensive test suite
- API streaming endpoints (Server-Sent Events)
- Docker containerization
- Model quantization (4-bit/8-bit)
- RAG (Retrieval Augmented Generation)
- Web UI

---

**Made with ❤️ for the open-source community**
