# Offline Adaptive DSA Tutor (MLML)

Training status: Trained 3 times.

An advanced, fully offline Data Structures & Algorithms tutor powered by state-of-the-art LLMs, adaptive ELO skill tracking, and retrieval-augmented generation. Runs on a single GPU (RTX 2060/6GB VRAM) with no internet required after setup.

---

## System Architecture

```mermaid
graph TD
  A[User (React Frontend)] -- Ask Question --> B(FastAPI Backend)
  B -- Validate Domain --> C[Domain Validator]
  B -- Retrieve Context --> D[RAG Retriever (FAISS + MiniLM)]
  B -- Generate Answer --> E[Model Loader (Mistral-7B-Instruct-v0.3, 4-bit)]
  B -- Store Interaction --> F[SQLite DB]
  B -- Update Skill --> G[Skill Engine (ELO)]
  B -- Stream Events --> A
  F -- Export Data --> H[Finetune Pipeline (LoRA/PEFT)]
  H -- Train Adapter --> E
  E -- Load Adapter --> B
```

### Engineering Stack
- **Frontend**: React 19 + Vite 7 (SSE streaming, collapsible thinking mode UI)
- **Backend**: FastAPI (async), Uvicorn, Python 3.14
- **Database**: SQLite (aiosqlite, async)
- **Model**: Mistral-7B-Instruct-v0.3 (4-bit NF4 quantization, ~4.1GB VRAM)
- **Embeddings**: sentence-transformers/all-MiniLM-L6-v2 (384-dim, FAISS)
- **Vector Store**: FAISS IndexFlatIP (cosine similarity)
- **Skill Engine**: ELO rating system (K-factor schedule)
- **Fine-tuning**: LoRA/PEFT, TRL, adapters hot-swappable
- **RAG Corpus**: 10 DSA markdown files, 43 vectors

---

## Project Structure

```
MLML/
├── app/                # Backend (FastAPI)
│   ├── config.py       # Pydantic settings
│   ├── database.py     # SQLite async layer
│   ├── main.py         # FastAPI app + lifespan
│   ├── model_loader.py # 4-bit LLM loader (Mistral-7B)
│   ├── retriever.py    # FAISS RAG retriever
│   ├── routes.py       # API endpoints
│   ├── tutor_routes.py # Thinking mode endpoints (SSE)
│   ├── services.py     # Business logic orchestration
│   ├── skill_engine.py # ELO rating system
│   ├── domain_validator.py # DSA keyword validation
│   └── reasoning.py    # Single-pass reasoning pipeline
├── frontend-react/     # React + Vite frontend
│   ├── src/            # Components, App.jsx, api.js, index.css
│   └── vite.config.js  # Proxy config
├── rag/                # RAG corpus & index builder
│   ├── build_index.py  # FAISS index builder
│   └── corpus/         # 10 DSA .md files
├── finetune/           # LoRA fine-tuning pipeline
│   ├── prepare_data.py # Export training data from DB
│   ├── train_lora.py   # LoRA/PEFT training
│   ├── merge_adapters.py # Merge LoRA into base
│   └── run_seed_finetune.py # Quick-start with seed data
├── faiss_index/        # Generated FAISS index
├── data/               # SQLite database
├── models/             # Downloaded models (MiniLM, Mistral-7B)
├── .env                # Config/env vars
└── requirements.txt
```

---

## Detailed Workflow

### 1. Ask a Question
- User submits a DSA question via the React frontend
- Frontend streams events (SSE) for thinking mode

### 2. Domain Validation
- Backend checks question for DSA keywords and blocks prompt injection
- Only valid DSA queries are allowed

### 3. Retrieval-Augmented Generation (RAG)
- FAISS index retrieves top-3 relevant DSA context chunks (from 10 .md files)
- Embeddings: all-MiniLM-L6-v2 (384-dim, CPU)

### 4. Reasoning Pipeline
- Single-pass: Model outputs <think>...</think> analysis + answer in one call
- Model: Mistral-7B-Instruct-v0.3 (4-bit NF4, ~4.1GB VRAM)
- Prompt includes user skill rating, topic, and retrieved context

### 5. Skill Engine
- ELO rating system tracks user skill per topic
- K-factor schedule adapts learning rate
- Feedback (helpful/not helpful) updates ELO

### 6. Storing Interactions
- All questions, answers, analysis, feedback stored in SQLite
- Used for fine-tuning pipeline

### 7. Fine-tuning Pipeline
- Export positive interactions to JSONL
- LoRA/PEFT fine-tuning (memory-safe for 6GB VRAM)
- Adapters hot-swappable at runtime
- Optionally merge adapter into base model

---

## Models Used

- **Mistral-7B-Instruct-v0.3**: Main LLM, 4-bit quantized, loaded from local directory
- **sentence-transformers/all-MiniLM-L6-v2**: Embedding model for RAG, cached locally
- **LoRA Adapters**: Fine-tuned adapters for DSA tutor, loaded via PEFT

---

## Training & Fine-tuning

### Data Export
- Export positive interactions from SQLite:
  ```bash
  python -m finetune.prepare_data --db-path data/mlml.db --output finetune/data/train.jsonl
  ```

### LoRA Fine-tuning
- Train LoRA adapter (memory-safe for RTX 2060 6GB):
  ```bash
  python -m finetune.train_lora --data finetune/data/train.jsonl --output adapters/v1 --model-id models/Mistral-7B-Instruct-v0.3
  ```
- Hot-swap adapter at runtime or merge for deployment

### Seed Training
- Quick-start with bundled seed DSA QA pairs:
  ```bash
  python -m finetune.run_seed_finetune
  ```

---

## Engineering Highlights

- **Single-pass DeepSeek-style thinking mode**: Model outputs <think>...</think> analysis + answer in one call
- **SSE Streaming**: Frontend displays collapsible "Thought" panel with spinner, updates in real time
- **ELO Skill Engine**: Adaptive difficulty, tracks per-topic skill
- **RAG**: FAISS + MiniLM, 43 vectors from 10 DSA corpus files
- **SQLite**: Zero-config, async, portable
- **LoRA/PEFT**: Fine-tuning pipeline, adapters hot-swappable
- **GPU Efficient**: 4-bit quantization, fits Mistral-7B in 4.1GB VRAM

---

## API Endpoints

| Method | Path                   | Description                        |
|--------|------------------------|------------------------------------|
| POST   | `/ask`                 | Submit a DSA question (non-stream) |
| POST   | `/tutor/ask/stream`    | Streaming thinking mode (SSE)      |
| POST   | `/feedback`            | Rate a response (ELO updates)      |
| GET    | `/skills/{user}`       | Get user's skill ratings           |
| GET    | `/health`              | System health check                |
| POST   | `/finetune/trigger`    | Start fine-tuning job              |
| POST   | `/finetune/adapter`    | Hot-swap LoRA adapter              |
| GET    | `/finetune/status`     | Fine-tuning job status             |

---

## DSA Topics Covered

Arrays, Linked Lists, Stacks & Queues, Hash Tables, Trees, Graphs, Sorting & Searching, Dynamic Programming, Recursion & Backtracking, Strings

---

## Quick Start

### Prerequisites
- Python 3.14+
- Node.js 18+
- NVIDIA GPU with CUDA (6GB+ VRAM)

### 1. Install Python dependencies
```bash
pip install -r requirements.txt
```

### 2. Download models (optional, auto-download on first run)
```bash
# Download Mistral-7B-Instruct-v0.3 (~15GB)
python -c "from huggingface_hub import snapshot_download; snapshot_download('mistralai/Mistral-7B-Instruct-v0.3', local_dir='models/Mistral-7B-Instruct-v0.3', ignore_patterns=['consolidated.safetensors', 'consolidated/*'])"
# Download all-MiniLM-L6-v2 (~90MB)
python -c "from huggingface_hub import snapshot_download; snapshot_download('sentence-transformers/all-MiniLM-L6-v2', local_dir='models/all-MiniLM-L6-v2')"
```

### 3. Build FAISS index (one-time)
```bash
python -m rag.build_index
```

### 4. Start backend
```bash
python -m uvicorn app.main:app --host 0.0.0.0 --port 8000
```

### 5. Start frontend
```bash
cd frontend-react
npm install
npm run dev
```
Open http://localhost:3000

---

## License
MIT
```json
{
  "username": "student",
  "topic": "arrays",
  "question": "What is the two pointer technique?"
}
```

### POST /feedback
```json
{
  "interaction_id": 1,
  "feedback": 1
}
```
`feedback`: 1 = helpful (ELO up), -1 = not helpful (ELO down)

## Project Structure
```
MLML/
├── app/
│   ├── config.py          # Pydantic settings
│   ├── database.py        # SQLite async layer
│   ├── main.py            # FastAPI app + lifespan
│   ├── model_loader.py    # 4-bit TinyLlama loader
│   ├── retriever.py       # FAISS RAG retriever
│   ├── routes.py          # API endpoints
│   ├── services.py        # Business logic orchestration
│   └── skill_engine.py    # ELO rating system
├── frontend-react/
│   ├── src/
│   │   ├── App.jsx        # Main React app
│   │   ├── api.js         # API client
│   │   ├── components/    # React components
│   │   └── index.css      # Styles
│   ├── package.json
│   └── vite.config.js     # Vite config with proxy
├── rag/
│   ├── build_index.py     # FAISS index builder
│   └── corpus/            # DSA topic files (.md)
├── finetune/
│   ├── prepare_data.py    # Export training data
│   ├── train_lora.py      # LoRA fine-tuning
│   └── merge_adapters.py  # Merge LoRA into base
├── sql/
│   └── init_db.sql        # SQLite schema
├── tests/
│   ├── test_skill_engine.py
│   ├── test_rag.py
│   └── test_db_smoke.py
├── faiss_index/           # Generated FAISS index
├── data/                  # SQLite database
└── requirements.txt
```

## How It Works

1. **Ask a question** → Backend retrieves relevant DSA context via FAISS RAG
2. **Generate response** → TinyLlama (4-bit) generates an answer calibrated to your skill level
3. **Rate the response** → Your feedback updates your ELO rating for that topic
4. **Adaptive difficulty** → Future responses are tailored to your rating (beginner/intermediate/advanced)

## Fine-tuning (Optional)

After collecting positive interactions, you can fine-tune the model with LoRA:

```bash
# Export training data
python -m finetune.prepare_data --db-path data/mlml.db --output finetune/data/train.jsonl

# Train LoRA adapter
python -m finetune.train_lora --data-path finetune/data/train.jsonl --output-dir adapters/v1

# Set MLML_ADAPTER_DIR=adapters/v1 and restart the server
```

## DSA Topics Covered
Arrays, Linked Lists, Stacks & Queues, Hash Tables, Trees, Graphs, Sorting & Searching, Dynamic Programming, Recursion & Backtracking, Strings
# MLML
