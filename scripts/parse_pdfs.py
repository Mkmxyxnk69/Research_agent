import pymupdf as fitz
import json
from pathlib import Path

CORPUS_DIR = Path("corpus")
PDF_DIR = CORPUS_DIR / "pdfs"
META_PATH = CORPUS_DIR / "metadata.jsonl"
OUTPUT_PATH = Path("data") / "chunks.jsonl"
CHUNK_SIZE = 600      # words per chunk
CHUNK_OVERLAP = 100   # words overlap between chunks

Path("data").mkdir(exist_ok=True)

def load_metadata():
    papers = {}
    with open(META_PATH) as f:
        for line in f:
            p = json.loads(line)
            papers[p["arxiv_id"]] = p
    return papers

def extract_text(pdf_path):
    try:
        doc = fitz.open(pdf_path)
        full_text = ""
        for page in doc:
            full_text += page.get_text()
        doc.close()
        return full_text.strip()
    except Exception as e:
        print(f"  Error reading {pdf_path.name}: {e}")
        return ""

def make_chunks(text, paper_id):
    words = text.split()
    chunks = []
    start = 0
    chunk_id = 0
    while start < len(words):
        end = start + CHUNK_SIZE
        chunk_words = words[start:end]
        chunk_text = " ".join(chunk_words)
        if len(chunk_text.strip()) > 100:  # skip tiny chunks
            chunks.append({
                "paper_id": paper_id,
                "chunk_id": chunk_id,
                "text": chunk_text
            })
            chunk_id += 1
        start += CHUNK_SIZE - CHUNK_OVERLAP
    return chunks

def main():
    papers = load_metadata()
    print(f"Loaded metadata for {len(papers)} papers")

    all_chunks = []
    success, fail = 0, 0

    pdf_files = list(PDF_DIR.glob("*.pdf"))
    print(f"Found {len(pdf_files)} PDFs\n")

    for i, pdf_path in enumerate(pdf_files):
        paper_id = pdf_path.stem
        print(f"[{i+1}/{len(pdf_files)}] Parsing {paper_id}")

        text = extract_text(pdf_path)
        if not text:
            fail += 1
            continue

        chunks = make_chunks(text, paper_id)
        all_chunks.extend(chunks)
        success += 1
        print(f"  → {len(chunks)} chunks")

    # Save all chunks
    with open(OUTPUT_PATH, "w") as f:
        for chunk in all_chunks:
            f.write(json.dumps(chunk) + "\n")

    print(f"\nDone!")
    print(f"Parsed: {success} PDFs, Failed: {fail}")
    print(f"Total chunks: {len(all_chunks)}")
    print(f"Saved to {OUTPUT_PATH}")

if __name__ == "__main__":
    main()