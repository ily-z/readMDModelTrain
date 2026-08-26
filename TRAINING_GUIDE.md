# MD-Deployer: Complete Training Guide

> A 37M parameter decoder-only transformer that reads any .md file and extracts a complete deployment manifest (source code, configs, dependencies, env vars, build/start commands).

---

## Table of Contents

1. Architecture
2. Precision and Hardware
3. Dataset Format
4. Dataset Generation Pipeline
5. Training (3 Phases)
6. Tokenizer Training
7. GGUF Conversion and Deployment
8. Inference

---

## 1. Architecture

**Decoder-only transformer. ~37-38M parameters.**

```
d_model     = 512
n_layers    = 8
n_heads     = 8
d_ff        = 2048
block_size  = 512
vocab_size  = 8192 (custom BPE)
dropout     = 0.1
norm        = RMSNorm
activation  = SwiGLU
positional  = RoPE
tie_embeddings = True
```

### Parameter Count

```
Token embedding:    8192 x 512           =  4.2M (tied with output)
Per layer:
  Attention Q/K/V/O: 4 x 512 x 512      =  1.0M
  MLP (SwiGLU):      3 x 512 x 2048     =  3.1M
  Norms + misc:                        ~  0.01M
  Per-layer total:                     ~  4.1M
8 layers:                                ~ 32.8M
Final norm + head:                       ~  0.5M
-----------------------------------------------
TOTAL:                                   ~ 37-38M
```

## 2. Precision and Hardware

### Training Precision

| Setting | Value | Why |
|---|---|---|
| Forward pass | bfloat16 | 2x throughput on RTX 3060, zero quality loss |
| Gradient accumulation | float32 | Stable loss accumulation |
| Optimizer states | float32 | AdamW requires fp32 states |
| Mixed precision | torch.cuda.amp | Automatic management |

### Hardware Requirements

| Component | Minimum | Recommended |
|---|---|---|
| GPU VRAM | 8 GB | 12 GB (RTX 3060) |
| System RAM | 16 GB | 32 GB |
| Storage | 50 GB | 100 GB SSD |
| Training time | 15-25 hours | 8-14 hours |

### VRAM Budget (RTX 3060 12GB)

```
Model weights (bf16):            ~75 MB   (38M params x 2 bytes)
Adam optimizer states (fp32):   ~300 MB   (2 x 38M x 4 bytes)
Gradients (bf16):                ~75 MB
Activations (batch=32, seq=512): ~1.5-3 GB
----------------------------------------------
Total:                           ~2-4 GB (comfortable with 12GB)
```

### Hyperparameters

```yaml
training:
  pretrain:
    batch_size: 64
    learning_rate: 3.0e-4
    lr_schedule: cosine
    lr_min: 3.0e-5
    warmup_steps: 500
    max_steps: 20000
    weight_decay: 0.1
    grad_clip: 1.0
    bf16: true

  distill:
    batch_size: 32
    learning_rate: 1.0e-4
    max_steps: 10000
    weight_decay: 0.1
    grad_clip: 1.0
    bf16: true

  sft:
    batch_size: 16
    learning_rate: 5.0e-5
    warmup_steps: 100
    max_steps: 5000
    weight_decay: 0.1
    grad_clip: 1.0
    bf16: true
```

## 3. Dataset Format

### Input: Markdown

```markdown
# FastAPI Todo App

A simple REST API for managing todos.

## Installation

pip install fastapi uvicorn sqlalchemy

## Usage

from fastapi import FastAPI
from sqlalchemy import create_engine

app = FastAPI()
engine = create_engine("sqlite:///./todos.db")

@app.get("/todos")
def get_todos():
    return [{"id": 1, "task": "Buy milk"}]

## Environment Variables

- DATABASE_URL: Database connection string (default: sqlite:///./todos.db)
- PORT: Server port (default: 8000)

## Running

uvicorn main:app --host 0.0.0.0 --port $PORT
```

### Output: JSON Deployment Manifest

```json
{
  "project_name": "fastapi-todo-app",
  "language": "python",
  "framework": "fastapi",
  "files": [
    {
      "path": "main.py",
      "content": "from fastapi import FastAPI\nfrom sqlalchemy import create_engine\n\napp = FastAPI()\nengine = create_engine(\"sqlite:///./todos.db\")\n\n@app.get(\"/todos\")\ndef get_todos():\n    return [{\"id\": 1, \"task\": \"Buy milk\"}]",
      "type": "source",
      "language": "python",
      "is_entrypoint": true,
      "confidence": 0.95
    }
  ],
  "dependencies": {
    "package_manager": "pip",
    "packages": [
      {"name": "fastapi", "version": null, "is_dev": false},
      {"name": "uvicorn", "version": null, "is_dev": false},
      {"name": "sqlalchemy", "version": null, "is_dev": false}
    ],
    "system_deps": []
  },
  "environment_variables": [
    {"name": "DATABASE_URL", "description": "Database connection string", "required": false, "default": "sqlite:///./todos.db"},
    {"name": "PORT", "description": "Server port", "required": false, "default": "8000"}
  ],
  "build_commands": ["pip install -r requirements.txt"],
  "start_command": "uvicorn main:app --host 0.0.0.0 --port 8000",
  "port": 8000
}
```

### Training JSONL Format

Each line is a chat-formatted training example:

```json
{"messages": [{"role": "system", "content": "You are a deployment manifest extractor. Given markdown documentation, output valid JSON."}, {"role": "user", "content": "Extract deployment manifest from this markdown:\n\n<markdown_content>"}, {"role": "assistant", "content": "<json_output>"}]}
```

## 4. Dataset Generation Pipeline

### Step 1: Collect Markdown Files

```python
import os, json, glob, requests
from pathlib import Path

def fetch_github_readmes(query, max_results=100, token=None):
    headers = {"Accept": "application/vnd.github.v3+json"}
    if token:
        headers["Authorization"] = f"token {token}"
    readmes = []
    per_page = min(max_results, 100)
    pages = (max_results + per_page - 1) // per_page
    for page in range(1, pages + 1):
        url = "https://api.github.com/search/repositories"
        params = {"q": query, "sort": "stars", "per_page": per_page, "page": page}
        try:
            resp = requests.get(url, headers=headers, params=params, timeout=30)
            resp.raise_for_status()
            for repo in resp.json().get("items", []):
                full_name = repo["full_name"]
                for branch in ["main", "master"]:
                    readme_url = f"https://raw.githubusercontent.com/{full_name}/{branch}/README.md"
                    try:
                        r = requests.get(readme_url, headers=headers, timeout=10)
                        if r.status_code == 200 and len(r.text) > 100:
                            readmes.append({
                                "repo_name": full_name,
                                "content": r.text,
                                "stars": repo.get("stargazers_count", 0),
                            })
                            break
                    except Exception:
                        continue
        except Exception as e:
            print(f"GitHub API error: {e}")
            break
    return readmes[:max_results]

def collect_local_markdown(data_dir):
    results = []
    for fp in glob.glob(os.path.join(data_dir, "**", "*.md"), recursive=True):
        try:
            with open(fp, "r", encoding="utf-8", errors="ignore") as f:
                content = f.read()
            if len(content) > 100:
                results.append({"repo_name": Path(fp).stem, "content": content})
        except Exception:
            continue
    return results
```

### Step 2: Generate Ground Truth with LLM

```python
import ollama

SYSTEM_PROMPT = """You are a deployment manifest extractor. Given markdown documentation,
extract all deployable project files and metadata into a JSON object.

Rules:
1. Every fenced code block with a recognized language is a candidate file
2. Infer file paths from: language tags, surrounding context, section headings
3. Extract dependency lists from prose or code examples
4. Identify build/start commands from prose instructions
5. Capture environment variables mentioned anywhere
6. Output ONLY valid JSON matching the schema. No explanations."""

def generate_ground_truth(markdown, model="qwen2.5:7b"):
    response = ollama.chat(
        model=model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"Extract deployment manifest as JSON:\n\n{markdown[:8000]}"},
        ],
        options={"temperature": 0.1},
    )
    content = response["message"]["content"]
    return json.loads(content)
```

### Step 3: Validate Pairs

```python
def extract_code_blocks(markdown_text):
    import re
    blocks = []
    for match in re.finditer(r"```(\w*)\n(.*?)```", markdown_text, re.DOTALL):
        blocks.append({"lang": match.group(1).strip().lower(), "content": match.group(2).strip()})
    return blocks

def validate_manifest(markdown, manifest):
    warnings = []
    if "files" not in manifest or not manifest["files"]:
        return False, ["No files extracted"]
    if not manifest.get("project_name"):
        warnings.append("Missing project_name")
    if not manifest.get("language"):
        warnings.append("Missing language")
    code_blocks = extract_code_blocks(markdown)
    extracted_count = len(manifest.get("files", []))
    if extracted_count < len(code_blocks) * 0.3:
        warnings.append(f"Only {extracted_count}/{len(code_blocks)} code blocks extracted")
    is_valid = len(warnings) == 0 or (len(warnings) <= 2 and extracted_count > 0)
    return is_valid, warnings
```

### Step 4: Format for Training

```python
SYSTEM_MESSAGE = "You are a deployment manifest extractor. Given markdown documentation, output valid JSON with extracted files, dependencies, and deployment configuration."

def format_training_example(markdown, manifest):
    return {
        "messages": [
            {"role": "system", "content": SYSTEM_MESSAGE},
            {"role": "user", "content": f"Extract deployment manifest from this markdown:\n\n{markdown}"},
            {"role": "assistant", "content": json.dumps(manifest, indent=2)},
        ]
    }

def create_training_file(pairs, output_path):
    with open(output_path, "w", encoding="utf-8") as f:
        for markdown, manifest in pairs:
            example = format_training_example(markdown, manifest)
            f.write(json.dumps(example, ensure_ascii=False) + "\n")
    print(f"Created: {output_path} ({len(pairs)} examples)")
```

### Step 5: Run the Pipeline

```python
from tqdm import tqdm

def run_pipeline(github_queries=None, local_dir=None, output_dir="data",
                 max_items=200, api_key=None, model="gpt-4o-mini"):
    os.makedirs(os.path.join(output_dir, "pairs"), exist_ok=True)
    all_md = []
    if github_queries:
        for q in github_queries:
            all_md.extend(fetch_github_readmes(q, max_results=min(max_items, 100)))
    if local_dir:
        all_md.extend(collect_local_markdown(local_dir))
    all_md = all_md[:max_items]

    valid_pairs = []
    for item in tqdm(all_md, desc="Generating ground truth"):
        try:
            manifest = generate_ground_truth(item["content"], api_key=api_key, model=model)
            ok, _ = validate_manifest(item["content"], manifest)
            if ok:
                valid_pairs.append((item["content"], manifest))
        except Exception:
            continue

    create_training_file(valid_pairs, os.path.join(output_dir, "pairs", "training.jsonl"))
    return valid_pairs

if __name__ == "__main__":
    run_pipeline(
        github_queries=["python fastapi tutorial", "nodejs express api", "react vite starter"],
        output_dir="data", max_items=200,
        api_key=os.environ.get("OPENAI_API_KEY"),
    )
```

### Data Sources

| Source | Volume | Method |
|---|---|---|
| GitHub READMEs with code | ~50M repos | GitHub API |
| Official framework docs | ~10K docs | Scrape |
| Dev.to / Medium tutorials | ~5M articles | RSS |
| Local project docs | Unlimited | File scan |

### Quality Filters

1. **Structural validation**: Output parses as valid JSON
2. **Completeness check**: All fenced code blocks accounted for
3. **Path plausibility**: File paths match language extensions
4. **Minimum size**: At least 1 source file extracted

## 5. Training (3 Phases)

### Phase 1: Pretrain (learn language structure)

- **Data**: 100M-500M tokens from READMEs, docs, tutorials, code
- **Duration**: 8-14 hours on RTX 3060
- **Purpose**: Learn markdown formatting, JSON syntax, code patterns
- **Data format**: Raw text (concatenated .md, .json, .py files)

### Phase 2: Distill (learn from larger model)

- **Data**: 50M-100M tokens of (markdown -> manifest) pairs generated by teacher
- **Duration**: 2-5 hours on RTX 3060
- **Purpose**: Learn the extraction task from a capable teacher (GPT-4o/Qwen2.5-7B)
- **Data format**: JSONL chat format

### Phase 3: SFT (task-specific tuning)

- **Data**: 5K-20K curated (markdown -> manifest) pairs
- **Duration**: 30min-2 hours on RTX 3060
- **Purpose**: Lock in exact output format, improve accuracy on edge cases
- **Data format**: JSONL chat format

### Training Loop

See train.py. Key components:

```python
def cosine_lr(step, config):
    warmup = config.get('warmup_steps', 500)
    max_steps = config['max_steps']
    lr_min = config.get('lr_min', 3e-5)
    lr_max = config['learning_rate']
    if step < warmup:
        return lr_max * step / warmup
    progress = (step - warmup) / (max_steps - warmup)
    return lr_min + 0.5 * (lr_max - lr_min) * (1 + math.cos(math.pi * progress))
```

### Checkpoint Save

```python
def save_checkpoint(model, optimizer, step, path, config):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'step': step,
        'config': {k: getattr(config, k) for k in config.__dataclass_fields__},
    }, path)
```

---

## 6. Tokenizer Training

### Approach

- **Type**: Byte-level BPE (GPT-2 style)
- **Vocab size**: 8192 (optimal for 37M param model)
- **Special tokens**: SOT, EOT, PAD, UNK

### tokenizer.py

```python
import os, glob
from tokenizers import Tokenizer, models, trainers, pre_tokenizers, decoders

def collect_texts(data_dir):
    texts = []
    for pat in ['*.md','*.txt','*.json','*.py','*.js','*.ts']:
        for fp in glob.glob(os.path.join(data_dir, '**', pat), recursive=True):
            try:
                with open(fp, 'r', encoding='utf-8', errors='ignore') as f:
                    texts.append(f.read())
            except: pass
    return texts

def train_tokenizer(texts, vocab_size=8192, output='tokenizer.json'):
    tok = Tokenizer(models.BPE())
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tok.decoder = decoders.ByteLevel()
    special = ['<sot>', '<eot>', '<pad>', '<unk>']
    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size, min_frequency=2,
        special_tokens=special, show_progress=True
    )
    tok.train_from_iterator(texts, trainer=trainer)
    tok.save(output)
    return tok

if __name__ == '__main__':
    texts = collect_texts('data/raw')
    train_tokenizer(texts)
```

---

## 7. GGUF Conversion and Deployment

### Step 1: Save as HuggingFace Format

```python
from transformers import LlamaConfig, LlamaForCausalLM

def save_hf_format(model, tokenizer, output_dir):
    config = LlamaConfig(
        vocab_size=model.config.vocab_size,
        hidden_size=model.config.d_model,
        intermediate_size=model.config.d_ff,
        num_hidden_layers=model.config.n_layers,
        num_attention_heads=model.config.n_heads,
        max_position_embeddings=model.config.block_size,
        tie_word_embeddings=model.config.tie_embeddings,
        rms_norm_eps=1e-6,
    )
    hf_model = LlamaForCausalLM(config)
    hf_model.load_state_dict(remap_keys(model))
    hf_model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
```

### Step 2: Convert to GGUF

```bash
git clone https://github.com/ggerganov/llama.cpp.git
pip install -r llama.cpp/requirements/requirements-convert_hf_to_gguf.txt

# Convert to FP16 GGUF
python llama.cpp/convert_hf_to_gguf.py \
    ./my-model-hf --outfile my-model-f16.gguf --outtype f16

# Quantize to Q4_K_M (~22MB)
llama-quantize my-model-f16.gguf my-model-Q4_K_M.gguf Q4_K_M
```

### Step 3: Deploy with Ollama

Create Modelfile:

```
FROM ./my-model-Q4_K_M.gguf

SYSTEM You are a deployment manifest extractor. Given markdown, output valid JSON.

PARAMETER temperature 0.1
PARAMETER top_p 0.9
```

```bash
ollama create md-deployer -f Modelfile
ollama run md-deployer
```

### Quantization Sizes (38M model)

| Format | Size | Quality |
|---|---|---|
| F16 | 75 MB | Reference |
| Q8_0 | 40 MB | Negligible loss |
| Q5_K_M | 28 MB | Very small loss |
| Q4_K_M | 22 MB | Small loss (recommended) |
| Q3_K_M | 17 MB | Noticeable loss |

---

## 8. Inference

### Python API

```python
import ollama

def extract_manifest(markdown_text):
    response = ollama.chat(
        model='md-deployer',
        messages=[{
            'role': 'user',
            'content': f'Extract deployment manifest from this markdown:\n\n{markdown_text}'
        }],
    )
    return json.loads(response['message']['content'])
```

### Integration with PVE Auto Deploy

```python
async def analyze_documentation(repo_url: str) -> dict:
    md_files = await fetch_markdown_files(repo_url)
    manifests = []
    for md in md_files:
        manifest = extract_manifest(md)
        manifests.append(manifest)
    return merge_manifests(manifests)
```

---

## Quick Start

```bash
pip install torch tokenizers transformers pyyaml requests tqdm ollama

# 1. Train tokenizer
python tokenizer.py data/raw 8192 tokenizer.json

# 2. Generate dataset (requires Ollama with qwen2.5:7b)
ollama pull qwen2.5:7b
python dataset.py

# 3. Train model
python train.py

# 4. Convert to GGUF
python convert.py checkpoints/sft_best.pt ./my-model-hf

# 5. Deploy to Ollama
ollama create md-deployer -f Modelfile
ollama run md-deployer
```

