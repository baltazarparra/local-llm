# Personal Assistant with Memory

An intelligent personal work assistant that remembers everything you tell it and provides context-aware advice based on your conversation history.

## Overview

The Personal Assistant extends ego_proxy with persistent memory, semantic search, and intelligent context retrieval. It's designed to help you track daily work activities, remember important details about people and projects, and provide advice based on past interactions.

## Features

### 🧠 Perfect Memory
- **Automatic Storage**: Every conversation is automatically saved to a local SQLite database
- **Rich Metadata**: Extracts people, topics, dates, sentiment, and categories from your messages
- **Semantic Search**: Uses sentence transformers to find relevant past conversations
- **Privacy First**: 100% local storage, no cloud dependencies

### 🔍 Intelligent Retrieval
- **Context-Aware**: Automatically retrieves relevant past conversations when responding
- **Multi-Strategy Search**: Combines semantic similarity, entity matching, and temporal ranking
- **Smart Filtering**: Natural language time filters (e.g., "last week", "recent")
- **Person & Topic Tracking**: Search by people mentioned or topics discussed

### 💬 Natural Interaction
- Beautiful terminal UI with markdown rendering
- Streaming responses with live updates
- Persistent command history
- Multiple search and timeline commands

## Quick Start

### Installation

1. **Install dependencies** (if not already done):
```bash
pip install -r requirements.txt
```

This includes:
- `sentence-transformers` (for semantic embeddings)
- `numpy` (for vector operations)
- All existing ego_proxy dependencies

2. **Launch the assistant**:
```bash
./assistant.sh
```

### First Use

The assistant will create a database file (`assistant_memory.db`) on first run. Every conversation is automatically saved and enriched with metadata.

## Usage Examples

### Natural Logging

Just speak naturally about your work:

```
You> I have a pair programming session with Matthew today. 
     Last time he was feeling sad about the project delays.

Assistant> I'll help you with that session. Since Matthew was feeling down about the 
          project delays, it might be good to start by checking in on how he's feeling 
          and discussing ways to get back on track...
```

### Getting Advice

Ask for context-aware advice:

```
You> What should I discuss with Matthew today?

Assistant> Based on your history:
          - Last session (3 days ago): Matthew was sad about project delays
          - Previous discussion: You were working on the API project
          
          I suggest:
          1. Check in on his mood and offer support
          2. Discuss concrete next steps to address the delays
          3. Break down remaining tasks into manageable pieces
```

### Searching History

```
You> /search matthew project

Search Results:
1. [2024-11-15 10:30] user: "Matthew was sad about project delays..."
2. [2024-11-12 14:15] user: "Discussed API project with Matthew..."
3. [2024-11-10 09:00] user: "Matthew asked about the deadline..."
```

## Available Commands

### Memory Commands
- `/search <query>` - Search past conversations semantically
- `/timeline [person]` - Show chronological conversation history
- `/people` - List all people mentioned in conversations
- `/topics` - List all topics discussed
- `/stats` - Show memory statistics (total conversations, people, topics, etc.)

### Session Commands
- `/help` - Show help message
- `/reset` - Clear working memory (database preserved)
- `/export [file]` - Export current session to markdown
- `/exit` - Exit the assistant

### Keyboard Shortcuts
- **Meta+Enter** (ESC+Enter) or **Alt+Enter** - Submit message
- **Ctrl+R** - Search command history
- **Ctrl+C** - Cancel current input

## How It Works

### Message Flow

```
1. You type a message
   ↓
2. Saved to database (conversations table)
   ↓
3. Generate semantic embedding
   ↓
4. Extract metadata using LLM:
   - People mentioned
   - Topics discussed
   - Dates/times mentioned
   - Sentiment (positive, negative, neutral, etc.)
   - Category (work, meeting, technical, etc.)
   ↓
5. Store metadata + embedding
```

### Response Flow

```
1. You ask a question
   ↓
2. Generate query embedding
   ↓
3. Search database:
   - Semantic similarity
   - Entity matching (people, topics)
   - Temporal filtering (recent conversations prioritized)
   ↓
4. Retrieve top 5 relevant conversations
   ↓
5. Inject context into LLM prompt
   ↓
6. Generate context-aware response
   ↓
7. Display response + save to database
```

## Architecture

### Database (SQLite)
- **conversations**: Stores all messages with timestamps
- **metadata**: Extracted entities, embeddings, sentiment

### Semantic Search
- Uses `sentence-transformers` (all-MiniLM-L6-v2 model)
- 384-dimensional embeddings
- Cosine similarity for search
- Fast on CPU, no GPU required

### Metadata Extraction
- Uses the loaded LLM to extract structured data
- JSON-based extraction with validation
- Fallback to simple heuristics if needed

### Context Retrieval
- Multi-strategy ranking:
  - Semantic similarity (base score)
  - Time decay (recent = higher score)
  - Role boost (user messages prioritized)
  - Sentiment boost (emotional content important)

## Configuration

### Database Location

Change the database file location:
```bash
./assistant.sh --db /path/to/your/memory.db
```

### Model Selection

Use a different model:
```bash
./assistant.sh --model Qwen/Qwen2.5-3B-Instruct
```

### Generation Parameters

Adjust temperature and token limits:
```bash
./assistant.sh --temperature 0.8 --max-tokens 1024
```

## Files Created

The assistant creates these files:
- `assistant_memory.db` - SQLite database with conversations and metadata
- `.assistant_chat_history` - Command history for prompt toolkit
- `assistant_session.md` - Exported session (if you use `/export`)

## Performance

### Speed
- **Embedding generation**: ~50-100ms per message (CPU)
- **Search**: <100ms for thousands of conversations
- **Metadata extraction**: ~2-3 seconds per message (depends on LLM)

### Storage
- Database grows ~1-2KB per conversation
- Embeddings are ~1.5KB each
- 10,000 conversations = ~15-20MB database

### Memory Usage
- sentence-transformers model: ~80MB RAM
- Minimal overhead beyond base ego_proxy

## Tips & Best Practices

### Logging Work Activities

Be descriptive and natural:
```
✓ Good: "I had a great pair programming session with Sarah. We fixed the 
        authentication bug and she taught me about JWT tokens."

✗ Too Brief: "Paired with Sarah"
```

### Getting Better Context

Include time references:
```
"What did I discuss with Sarah last week?"
"Show me recent conversations about the authentication project"
```

### Managing Memory

- Use `/stats` regularly to see memory growth
- Use `/timeline` to review interactions with specific people
- Export important sessions with `/export`

## Troubleshooting

### "sentence-transformers not installed"

Install dependencies:
```bash
pip install sentence-transformers numpy
```

### Slow first response

The embedding model loads on first use (~80MB download). Subsequent responses are fast.

### Database locked error

Only one assistant instance can access the database at a time. Close other instances.

### Out of memory

The assistant is lightweight, but if you have memory constraints:
- Use a smaller base model
- Reduce `--max-tokens`
- Set lower temperature

## Advanced Usage

### Multiple Databases

Maintain separate memories for different contexts:
```bash
./assistant.sh --db work_memory.db        # Work conversations
./assistant.sh --db personal_memory.db    # Personal notes
```

### Programmatic Access

You can use the components directly in Python:

```python
from src.database import AssistantDatabase
from src.retrieval import ContextRetriever

# Open database
db = AssistantDatabase("assistant_memory.db")

# Search conversations
retriever = ContextRetriever(db)
results = retriever.retrieve_by_person("Matthew", limit=10)

# Get statistics
stats = db.get_stats()
print(f"Total conversations: {stats['total_conversations']}")
```

## Privacy & Security

- **100% Local**: All data stored locally in SQLite
- **No Cloud Calls**: No telemetry, no external APIs (except Google Calendar if configured)
- **Your Control**: You own the database file
- **Portable**: Copy the `.db` file to backup or move

## Key Features

| Feature | Description |
|---------|-------------|
| **Permanent Memory** | All conversations automatically saved |
| **Context Awareness** | Full history retrieval for informed responses |
| **People Tracking** | Automatically identifies and tracks people mentioned |
| **Topic Tracking** | Categorizes conversations by topic |
| **Semantic Search** | Find past conversations using natural language |
| **Timeline View** | Browse conversations chronologically |
| **Metadata Extraction** | Auto-extracts dates, sentiment, categories |
| **Google Calendar** | Natural language event creation and upcoming events display |
| **Export** | Export conversations to markdown |

## Future Enhancements

Potential improvements (not yet implemented):
- Web UI for browsing conversation history
- Export to various formats (JSON, CSV)
- Scheduled reminders based on history
- Multi-user support
- Conversation tagging and categorization
- Advanced analytics and insights
- Email integration

## Support

For issues or questions:
1. Check this documentation
2. Review `PLAN.md` for architecture details
3. Check logs for error messages
4. Ensure all dependencies are installed

## License

Same as ego_proxy - MIT License

