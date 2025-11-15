# Ego Proxy

Run powerful AI language models on your computer completely offline. No cloud dependencies, no data shared externally. Perfect for chatting with AI while keeping everything private.

## What You Can Do

Chat interactively with AI using a beautiful, feature-rich terminal interface. Ego Proxy provides a modern chat experience with markdown rendering, syntax highlighting, streaming responses, and conversation history.

## What You Need

Python 3.10 or newer. A GPU helps but isn't required. Models download automatically from Hugging Face (default is about 1.5GB). That's it.

## Getting Started

```bash
python3 -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Using It

**Interactive Chat:**
```bash
./chat-rich.sh
```

## Configuration

Create a `.env` file to customize settings:
```
MODEL_ID=Qwen/Qwen2.5-1.5B-Instruct
USE_GPU=true
MAX_NEW_TOKENS=512
TEMPERATURE=0.7
```

## Troubleshooting

**Out of memory?** Lower `MAX_NEW_TOKENS` or try `USE_GPU=false`

**Very slow?** You're probably on CPU. GPU is much faster.

**Script won't run?** Make it executable: `chmod +x chat-rich.sh`

## That's It

Your data stays local. No tracking. No cloud calls. Just you and your AI model running on your machine.

Want more details? Check out `PLAN.md` for the development roadmap.
