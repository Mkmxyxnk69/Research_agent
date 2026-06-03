import json
from pathlib import Path

CORPUS_DIR = Path("corpus")
PDF_DIR = CORPUS_DIR / "pdfs"
META_PATH = CORPUS_DIR / "metadata.jsonl"

def main():
    if not PDF_DIR.exists():
        print("No corpus/pdfs directory found.")
        return

    pdf_files = sorted(PDF_DIR.glob("*.pdf"))
    if not pdf_files:
        print("No PDFs found in corpus/pdfs.")
        return

    print(f"Found {len(pdf_files)} PDFs. Rebuilding minimal metadata.jsonl ...")

    with open(META_PATH, "w") as f:
        for pdf_path in pdf_files:
            arxiv_id = pdf_path.stem
            record = {
                "arxiv_id": arxiv_id,
                "title": "",
                "abstract": "",
                "published": "",
                "authors": [],
                "pdf_url": ""
            }
            f.write(json.dumps(record) + "\n")

    print(f"Written minimal metadata for {len(pdf_files)} papers to {META_PATH}")

if __name__ == "__main__":
    main()