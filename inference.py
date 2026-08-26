import json
import sys
import os

try:
    import ollama
except ImportError:
    ollama = None

MODEL_NAME = "md-deployer"


def extract_manifest(markdown_text: str) -> dict:
    if ollama is None:
        raise ImportError(
            "ollama package not installed. Run: pip install ollama"
        )

    response = ollama.chat(
        model=MODEL_NAME,
        messages=[
            {
                "role": "user",
                "content": f"Extract deployment manifest from this markdown:\n\n{markdown_text}",
            }
        ],
    )

    content = response["message"]["content"]

    try:
        manifest = json.loads(content)
    except json.JSONDecodeError:
        manifest = {"raw_response": content}

    return manifest


def extract_manifest_from_file(file_path: str) -> dict:
    with open(file_path, "r", encoding="utf-8") as f:
        markdown_text = f.read()

    return extract_manifest(markdown_text)


def analyze_repository(repo: str) -> dict:
    if os.path.isdir(repo):
        return _analyze_local_repo(repo)
    elif repo.startswith("http://") or repo.startswith("https://"):
        return _analyze_remote_repo(repo)
    else:
        raise ValueError(f"Invalid repo argument: {repo}")


def _analyze_local_repo(repo_path: str) -> dict:
    manifests = []
    candidates = []

    for root, _dirs, files in os.walk(repo_path):
        for fname in files:
            lower = fname.lower()
            if lower == "readme.md" or lower.endswith(".md") and "doc" in root.lower():
                candidates.append(os.path.join(root, fname))

    if not candidates:
        for root, _dirs, files in os.walk(repo_path):
            for fname in files:
                if fname.lower().endswith(".md"):
                    candidates.append(os.path.join(root, fname))

    for md_file in candidates[:5]:
        try:
            manifest = extract_manifest_from_file(md_file)
            manifest["_source_file"] = md_file
            manifests.append(manifest)
        except Exception as e:
            print(f"Warning: failed to process {md_file}: {e}", file=sys.stderr)

    return _merge_manifests(manifests)


def _analyze_remote_repo(repo_url: str) -> dict:
    print(f"Remote repo analysis not yet implemented: {repo_url}", file=sys.stderr)
    return {"error": "remote repo analysis not implemented", "repo": repo_url}


def _merge_manifests(manifests: list) -> dict:
    if not manifests:
        return {}

    merged = {}
    for m in manifests:
        for key, value in m.items():
            if key == "_source_file":
                continue
            if key in merged:
                if isinstance(merged[key], list) and isinstance(value, list):
                    merged[key].extend(value)
                elif isinstance(merged[key], dict) and isinstance(value, dict):
                    merged[key].update(value)
                else:
                    merged[key] = value
            else:
                merged[key] = value

    merged["_sources"] = [m.get("_source_file", "unknown") for m in manifests]
    return merged


def main():
    if len(sys.argv) < 2:
        print("Usage: python inference.py <markdown_file_or_repo_url>")
        sys.exit(1)

    target = sys.argv[1]

    try:
        if os.path.isfile(target):
            result = extract_manifest_from_file(target)
        elif os.path.isdir(target) or target.startswith("http"):
            result = analyze_repository(target)
        else:
            print(f"Error: '{target}' is not a valid file or directory.", file=sys.stderr)
            sys.exit(1)

        print(json.dumps(result, indent=2))

    except ImportError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
