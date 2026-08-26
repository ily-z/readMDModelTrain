# MD-Deployer

> A custom 37M parameter transformer that reads any `.md` file and extracts a complete deployment manifest — source code, configs, dependencies, environment variables, build/start commands — everything needed to make a web app run as-is.

---

## What Is This?

MD-Deployer is a **small, purpose-built language model** trained to understand markdown documentation and extract deployable project files from it. Give it a README, a tutorial, or any `.md` file, and it outputs a structured JSON manifest that tells you exactly how to deploy the app.

### Why Build a Custom Model?

| Approach | RAM | Cost | Speed | Accuracy |
|---|---|---|---|---|
| GPT-4o API | 0 MB (cloud) | $0.01-0.10/query | 2-5s | High |
| Qwen2.5-0.5B (Ollama) | ~500 MB | Free | 10-30 tok/s | Fair |
| **MD-Deployer (this)** | **~75 MB** | **Free** | **50-100+ tok/s** | **Good (narrow task)** |

MD-Deployer runs in **75 MB of RAM** (Q4 quantized), making it viable for edge deployments, CI/CD pipelines, and constrained environments where a 500MB+ model is too heavy.

---

## How It Works

```
Input:  Any .md file (README, tutorial, docs)
         |
         v
   MD-Deployer (37M params)
         |
         v
Output: JSON Deployment Manifest
        {
          "project_name": "fastapi-todo",
          "language": "python",
          "framework": "fastapi",
          "files": [
            {"path": "main.py", "content": "from fastapi import ...", "type": "source"},
            {"path": "requirements.txt", "content": "fastapi\nuvicorn", "type": "config"}
          ],
          "dependencies": {"package_manager": "pip", "packages": [...]},
          "environment_variables": [{"name": "DATABASE_URL", ...}],
          "build_commands": ["pip install -r requirements.txt"],
          "start_command": "uvicorn main:app --host 0.0.0.0 --port 8000",
          "port": 8000
        }
```

---

## Architecture

```
Model:     37-38M parameter decoder-only transformer
Training:  3 phases (pretrain -> distill -> SFT)
Tokenizer: Custom BPE, vocab 8192
Precision: bfloat16 (training), Q4_K_M GGUF (deployment)
```

| Component | Value |
|---|---|
| d_model | 512 |
| n_layers | 8 |
| n_heads | 8 |
| d_ff | 2048 |
| block_size | 512 |
| vocab_size | 8192 |
| Norm | RMSNorm |
| Activation | SwiGLU |
| Positional | RoPE |
| Total params | ~37.8M |

---

## Project Structure

```
mdModelTrain/
  model.py            # Transformer architecture (RMSNorm, RoPE, SwiGLU)
  tokenizer.py        # Custom BPE tokenizer training
  dataset.py          # Data collection + LLM ground truth generation
  train.py            # 3-phase training loop (pretrain/distill/SFT)
  convert.py          # HuggingFace format conversion for GGUF
  inference.py        # Ollama inference API
  config.yaml         # All hyperparameters
  requirements.txt    # Python dependencies
  TRAINING_GUIDE.md   # Complete 600-line training documentation
  README.md           # This file
```

---

## Quick Start

### Prerequisites

- Python 3.11+
- CUDA GPU with 8GB+ VRAM (RTX 3060 or better)
- [Ollama](https://ollama.ai) installed

### Step 1: Install Dependencies

```bash
cd mdModelTrain
pip install -r requirements.txt
```

### Step 2: Pull the Teacher Model

```bash
ollama pull qwen2.5:7b
```

This is the larger model that generates training data for MD-Deployer. You only need it during training, not at inference time.

### Step 3: Collect Training Data

```bash
# Option A: Fetch READMEs from GitHub
python dataset.py

# Option B: Use your own markdown files
python dataset.py /path/to/your/docs
```

This will:
1. Fetch READMEs from GitHub (Python, Node.js, React tutorials)
2. Use Qwen2.5-7b (local Ollama) to generate ground truth JSON manifests
3. Validate and save training pairs to `data/pairs/training.jsonl`

### Step 4: Train the Tokenizer

```bash
python tokenizer.py data/raw 8192 tokenizer.json
```

### Step 5: Train the Model (3 Phases)

```bash
python train.py
```

Training takes **8-14 hours on an RTX 3060** (12GB VRAM). The three phases:

| Phase | Purpose | Data | Duration |
|---|---|---|---|
| Pretrain | Learn markdown + JSON structure | Raw text | 8-14h |
| Distill | Learn extraction from teacher | LLM-generated pairs | 2-5h |
| SFT | Lock in exact output format | Curated pairs | 30min-2h |

Checkpoints are saved to `checkpoints/` every 2000 steps.

### Step 6: Convert to GGUF

```bash
python convert.py checkpoints/sft_best.pt ./hf_model
```

Then quantize and deploy:

```bash
# Clone llama.cpp
git clone https://github.com/ggerganov/llama.cpp.git
pip install -r llama.cpp/requirements/requirements-convert_hf_to_gguf.txt

# Convert to GGUF
python llama.cpp/convert_hf_to_gguf.py ./hf_model --outfile md-deployer-f16.gguf --outtype f16

# Quantize to Q4_K_M (~22MB)
llama-quantize md-deployer-f16.gguf md-deployer-Q4_K_M.gguf Q4_K_M
```

### Step 7: Deploy to Ollama

Create a `Modelfile`:

```
FROM ./md-deployer-Q4_K_M.gguf

SYSTEM You are a deployment manifest extractor. Given markdown, output valid JSON.

PARAMETER temperature 0.1
PARAMETER top_p 0.9
```

```bash
ollama create md-deployer -f Modelfile
ollama run md-deployer
```

### Step 8: Use It

```python
import ollama

response = ollama.chat(
    model="md-deployer",
    messages=[{
        "role": "user",
        "content": "Extract deployment manifest from this markdown:\n\n# My FastAPI App\n\npip install fastapi uvicorn\n\nuvicorn main:app --port 8000"
    }]
)

import json
manifest = json.loads(response["message"]["content"])
print(json.dumps(manifest, indent=2))
```

---

## Hardware Requirements

| Component | Minimum | Recommended |
|---|---|---|
| GPU VRAM | 8 GB | 12 GB (RTX 3060) |
| System RAM | 16 GB | 32 GB |
| Storage | 50 GB | 100 GB SSD |
| Training time | 15-25 hours | 8-14 hours |

### VRAM Budget

```
Model weights (bf16):            ~75 MB
Adam optimizer states (fp32):   ~300 MB
Gradients (bf16):                ~75 MB
Activations (batch=32, seq=512): ~1.5-3 GB
----------------------------------------------
Total:                           ~2-4 GB (fits in 8GB)
```

---

## Output Schema

The model outputs JSON matching this structure:

```json
{
  "project_name": "string",
  "language": "python|javascript|typescript|go|rust|java|ruby|shell",
  "framework": "string or null",
  "files": [
    {
      "path": "relative/file/path.py",
      "content": "file content here",
      "type": "source|config|environment|script|test",
      "language": "python",
      "is_entrypoint": true,
      "confidence": 0.95
    }
  ],
  "dependencies": {
    "package_manager": "pip|npm|yarn|cargo|go",
    "packages": [{"name": "fastapi", "version": null, "is_dev": false}],
    "system_deps": ["ffmpeg"]
  },
  "environment_variables": [
    {"name": "DATABASE_URL", "description": "...", "required": true, "default": null}
  ],
  "build_commands": ["pip install -r requirements.txt"],
  "start_command": "uvicorn main:app --port 8000",
  "port": 8000,
  "docker": null
}
```

---

## Integration with PVE Auto Deploy

MD-Deployer replaces the LLM doc analyzer in Phase 3 of the PVE Auto Deploy system:

```python
# app/modules/doc_analyzer.py
import ollama

async def analyze_documentation(repo_url: str) -> dict:
    md_files = await fetch_markdown_files(repo_url)
    manifests = []
    for md in md_files:
        response = ollama.chat(
            model="md-deployer",
            messages=[{
                "role": "user",
                "content": f"Extract deployment manifest from this markdown:\n\n{md}"
            }]
        )
        manifest = json.loads(response["message"]["content"])
        manifests.append(manifest)
    return merge_manifests(manifests)
```

---

## Training Data Sources

| Source | Volume | Method |
|---|---|---|
| GitHub READMEs with code | ~50M repos | GitHub API |
| Official framework docs | ~10K docs | Scrape |
| Dev.to / Medium tutorials | ~5M articles | RSS |
| Local project docs | Unlimited | File scan |

---

## Troubleshooting

### "CUDA out of memory"
- Reduce batch size in `config.yaml` (e.g., `batch_size: 32` instead of 64)
- Reduce `block_size` from 512 to 256

### "No texts found in data/raw"
- Run `python dataset.py` first to collect training data
- Or manually add `.md`, `.json`, `.py` files to `data/raw/`

### "Ollama not responding"
- Make sure Ollama is running: `ollama serve`
- Check the model is pulled: `ollama list`

### Low accuracy on output
- Increase training data (run `dataset.py` with `max_items=500`)
- Increase SFT steps in `config.yaml`
- Try a larger teacher model: `ollama pull qwen2.5:14b`

---

## License

Research use. Built for the PVE Auto Deploy project.

---

## Credits

- Architecture inspired by [NanoGPT](https://github.com/karpathy/nanoGPT) and [SmolLM2](https://huggingface.co/HuggingFaceTB/SmolLM2-135M)
- Tokenizer trained with [HuggingFace tokenizers](https://github.com/huggingface/tokenizers)
- GGUF conversion via [llama.cpp](https://github.com/ggerganov/llama.cpp)
- Inference via [Ollama](https://ollama.ai)
