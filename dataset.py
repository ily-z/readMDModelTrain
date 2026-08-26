"""Dataset generation pipeline for MD-Deployer.

Collects markdown files, generates ground truth deployment manifests
using a larger LLM, validates pairs, and formats for training.
"""

import os
import json
import glob
import re
from pathlib import Path

import requests
from tqdm import tqdm


# --- Output Schema ---

DEPLOYMENT_MANIFEST_SCHEMA = {
    "type": "object",
    "required": ["project_name", "language", "files"],
    "properties": {
        "project_name": {"type": "string"},
        "project_description": {"type": "string"},
        "language": {
            "type": "string",
            "enum": ["python", "javascript", "typescript", "go", "rust", "java", "ruby", "shell", "unknown"],
        },
        "framework": {"type": ["string", "null"]},
        "files": {
            "type": "array",
            "items": {
                "type": "object",
                "required": ["path", "content", "type"],
                "properties": {
                    "path": {"type": "string"},
                    "content": {"type": "string"},
                    "type": {
                        "type": "string",
                        "enum": ["source", "config", "environment", "documentation", "script", "test"],
                    },
                    "language": {"type": ["string", "null"]},
                    "is_entrypoint": {"type": "boolean"},
                    "confidence": {"type": "number"},
                },
            },
        },
        "dependencies": {
            "type": "object",
            "properties": {
                "package_manager": {"type": "string"},
                "packages": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "name": {"type": "string"},
                            "version": {"type": ["string", "null"]},
                            "is_dev": {"type": "boolean"},
                        },
                    },
                },
                "system_deps": {"type": "array", "items": {"type": "string"}},
            },
        },
        "environment_variables": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "description": {"type": "string"},
                    "required": {"type": "boolean"},
                    "default": {"type": ["string", "null"]},
                    "example": {"type": ["string", "null"]},
                },
            },
        },
        "build_commands": {"type": "array", "items": {"type": "string"}},
        "start_command": {"type": ["string", "null"]},
        "test_command": {"type": ["string", "null"]},
        "port": {"type": ["integer", "null"]},
        "docker": {
            "type": ["object", "null"],
            "properties": {
                "dockerfile": {"type": ["string", "null"]},
                "docker_compose": {"type": ["string", "null"]},
                "base_image": {"type": ["string", "null"]},
            },
        },
    },
}


# --- Markdown Parsing ---

def extract_code_blocks(markdown_text: str) -> list:
    """Extract fenced code blocks from markdown."""
    blocks = []
    for match in re.finditer(r"```(\w*)\n(.*?)```", markdown_text, re.DOTALL):
        lang = match.group(1).strip().lower()
        content = match.group(2).strip()
        blocks.append({"lang": lang, "content": content})
    return blocks


def extract_headings(markdown_text: str) -> list:
    """Extract markdown headings with their content."""
    headings = []
    lines = markdown_text.split("\n")
    current = None
    content_lines = []
    for line in lines:
        heading_match = re.match(r"^(#{1,6})\s+(.*)", line)
        if heading_match:
            if current:
                headings.append({
                    "level": len(current["hashes"]),
                    "text": current["text"],
                    "content": "\n".join(content_lines),
                })
            current = {"hashes": heading_match.group(1), "text": heading_match.group(2)}
            content_lines = []
        else:
            content_lines.append(line)
    if current:
        headings.append({
            "level": len(current["hashes"]),
            "text": current["text"],
            "content": "\n".join(content_lines),
        })
    return headings


# --- Data Collection ---

def fetch_github_readmes(query: str, max_results: int = 100, token: str = None) -> list:
    """Fetch READMEs from GitHub search results."""
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
                                "url": repo.get("html_url", ""),
                            })
                            break
                    except Exception:
                        continue
        except Exception as e:
            print(f"GitHub API error on page {page}: {e}")
            break

    return readmes[:max_results]


def collect_local_markdown(data_dir: str) -> list:
    """Collect local markdown files."""
    results = []
    for fp in glob.glob(os.path.join(data_dir, "**", "*.md"), recursive=True):
        try:
            with open(fp, "r", encoding="utf-8", errors="ignore") as f:
                content = f.read()
            if len(content) > 100:
                results.append({
                    "repo_name": Path(fp).stem,
                    "content": content,
                    "source_path": fp,
                })
        except Exception:
            continue
    return results


# --- Ground Truth Generation ---

SYSTEM_PROMPT = """You are a deployment manifest extractor. Given markdown documentation (README, tutorial, or docs), extract all deployable project files and metadata into a JSON object.

Rules:
1. Every fenced code block with a recognized language is a candidate file
2. Infer file paths from: language tags, surrounding context, section headings, comments in code
3. Extract dependency lists (requirements.txt, package.json, etc.) from prose or code examples
4. Identify build/start commands from prose instructions
5. Capture environment variables mentioned anywhere
6. Output ONLY valid JSON matching the schema. No explanations.
7. Set confidence based on how certain you are about each extraction"""


def generate_ground_truth(
    markdown: str,
    model: str = "qwen2.5:7b",
    base_url: str = None,
) -> dict:
    """Generate a ground truth deployment manifest from markdown using Ollama."""
    import ollama

    schema_str = json.dumps(DEPLOYMENT_MANIFEST_SCHEMA, indent=2)
    user_msg = f"Extract deployment manifest as JSON.\n\nSchema:\n{schema_str}\n\nMarkdown:\n{markdown[:8000]}"

    response = ollama.chat(
        model=model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_msg},
        ],
        options={"temperature": 0.1},
    )

    content = response["message"]["content"]
    return json.loads(content)


# --- Validation ---

def validate_manifest(markdown: str, manifest: dict) -> tuple:
    """Validate a deployment manifest. Returns (is_valid, list_of_warnings)."""
    warnings = []

    if "files" not in manifest or not manifest["files"]:
        warnings.append("No files extracted")
        return False, warnings

    if not manifest.get("project_name"):
        warnings.append("Missing project_name")

    if not manifest.get("language"):
        warnings.append("Missing language")

    code_blocks = extract_code_blocks(markdown)
    extracted_count = len(manifest.get("files", []))
    if extracted_count < len(code_blocks) * 0.3:
        warnings.append(f"Only {extracted_count}/{len(code_blocks)} code blocks extracted")

    for f in manifest.get("files", []):
        if not f.get("path"):
            warnings.append(f"File missing path: {f.get('content', '')[:50]}")
        if not f.get("content"):
            warnings.append(f"File {f.get('path', '?')} has empty content")

    is_valid = len(warnings) == 0 or (len(warnings) <= 2 and extracted_count > 0)
    return is_valid, warnings


# --- Training Data Formatting ---

SYSTEM_MESSAGE = "You are a deployment manifest extractor. Given markdown documentation, output valid JSON with extracted files, dependencies, and deployment configuration."


def format_training_example(markdown: str, manifest: dict) -> dict:
    """Create a JSONL training example in chat format."""
    return {
        "messages": [
            {"role": "system", "content": SYSTEM_MESSAGE},
            {"role": "user", "content": f"Extract deployment manifest from this markdown:\n\n{markdown}"},
            {"role": "assistant", "content": json.dumps(manifest, indent=2)},
        ]
    }


def create_training_file(pairs: list, output_path: str):
    """Write training pairs to JSONL file."""
    with open(output_path, "w", encoding="utf-8") as f:
        for markdown, manifest in pairs:
            example = format_training_example(markdown, manifest)
            f.write(json.dumps(example, ensure_ascii=False) + "\n")
    print(f"Created training file: {output_path} ({len(pairs)} examples)")


# --- Main Pipeline ---

def run_pipeline(
    sources: list = None,
    github_queries: list = None,
    local_dir: str = None,
    output_dir: str = "data",
    max_items: int = 1000,
    model: str = "qwen2.5:7b",
):
    """Full dataset generation pipeline.

    Args:
        sources: List of markdown dicts with 'repo_name' and 'content' keys.
        github_queries: GitHub search queries for README collection.
        local_dir: Local directory to scan for .md files.
        output_dir: Where to save output files.
        max_items: Max items to process.
        model: Ollama model to use for ground truth generation.
    """
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, "raw"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "pairs"), exist_ok=True)

    all_markdown = []

    if sources:
        all_markdown.extend(sources)

    if github_queries:
        for query in github_queries:
            print(f"Fetching GitHub repos for: {query}")
            readmes = fetch_github_readmes(
                query, max_results=min(max_items, 100),
                token=os.environ.get("GITHUB_TOKEN"),
            )
            all_markdown.extend(readmes)
            print(f"  Collected {len(readmes)} READMEs")

    if local_dir:
        local = collect_local_markdown(local_dir)
        all_markdown.extend(local)
        print(f"Collected {len(local)} local markdown files")

    all_markdown = all_markdown[:max_items]
    print(f"\nTotal markdown files: {len(all_markdown)}")

    valid_pairs = []
    invalid_count = 0

    for item in tqdm(all_markdown, desc="Generating ground truth"):
        markdown = item["content"]
        try:
            manifest = generate_ground_truth(markdown, model=model)
            is_valid, warnings = validate_manifest(markdown, manifest)
            if is_valid:
                valid_pairs.append((markdown, manifest))
            else:
                invalid_count += 1
        except Exception:
            invalid_count += 1
            continue

    print(f"\nValid pairs: {len(valid_pairs)}, Invalid: {invalid_count}")

    create_training_file(valid_pairs, os.path.join(output_dir, "pairs", "training.jsonl"))

    with open(os.path.join(output_dir, "raw", "manifest.json"), "w") as f:
        json.dump({
            "total_markdown": len(all_markdown),
            "valid_pairs": len(valid_pairs),
            "invalid": invalid_count,
        }, f, indent=2)

    return valid_pairs


if __name__ == "__main__":
    import sys

    queries = [
        "python fastapi tutorial",
        "nodejs express api",
        "react vite starter",
        "python flask app",
        "django rest api",
    ]

    model = os.environ.get("GT_MODEL", "qwen2.5:7b")

    pairs = run_pipeline(
        github_queries=queries,
        local_dir=sys.argv[1] if len(sys.argv) > 1 else None,
        output_dir="data",
        max_items=200,
        model=model,
    )
    print(f"\nDone. {len(pairs)} training pairs saved to data/pairs/training.jsonl")
