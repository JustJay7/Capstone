import os
from pathlib import Path

OUTPUT_FILE = Path(r"C:\Users\RDX\largest_files_report.txt")

# Directories to skip to avoid permission and endless recursion issues
SKIP_DIRS = [
    r"C:\Windows",
    r"C:\Program Files",
    r"C:\Program Files (x86)",
    r"C:\ProgramData",
    r"C:\Recovery",
    r"C:\System Volume Information",
    r"C:\$Recycle.Bin"
]

# Keywords that signal AI / coding / Python / VS Code relevance
AI_KEYWORDS = [
    "python", "pytorch", "tensorflow", "huggingface", "transformers",
    "ai", "ml", "deep", "neural", "vscode", "vs code", "notebook",
    "conda", "torch", "cuda", "checkpoint", "model", "weights",
    "yolo", "openai", "langchain", "gemini", "llm", "fastapi",
    "venv", "research_env", "googleads_env"
]

def is_skipped(path: str) -> bool:
    """Return True if path should be skipped."""
    for skip in SKIP_DIRS:
        if path.lower().startswith(skip.lower()):
            return True
    return False

def scan_files(root: str):
    """Yield (path, size_bytes) for all files under root, safely."""
    for dirpath, _, filenames in os.walk(root, topdown=True):
        if is_skipped(dirpath):
            # prevent walking into system dirs
            continue
        for name in filenames:
            fpath = os.path.join(dirpath, name)
            try:
                size = os.path.getsize(fpath)
                yield fpath, size
            except (OSError, PermissionError):
                continue

def categorize(path: str) -> str:
    """Return 'AI/Coding Related' or 'Other'."""
    lower = path.lower()
    return "AI/Coding Related" if any(k in lower for k in AI_KEYWORDS) else "Other"

def main():
    print("🔍 Scanning drive C: ... this may take several minutes.")
    all_files = list(scan_files("C:\\"))
    print(f"✅ Scanned {len(all_files):,} files. Sorting by size...")

    # Sort and get top 100
    largest = sorted(all_files, key=lambda x: x[1], reverse=True)[:100]

    # Write report
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        f.write("==== Largest Files on C: (Top 100) ====\n\n")
        f.write(f"{'Size (MB)':>10} | {'Category':<20} | Path\n")
        f.write("-" * 120 + "\n")
        for path, size in largest:
            size_mb = round(size / (1024 * 1024), 2)
            category = categorize(path)
            f.write(f"{size_mb:>10} | {category:<20} | {path}\n")

    print(f"📁 Report saved at: {OUTPUT_FILE}")

if __name__ == "__main__":
    main()
