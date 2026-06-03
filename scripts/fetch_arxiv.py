import requests
import json
import time
import os
import xml.etree.ElementTree as ET
from pathlib import Path

# Config
CATEGORIES = ["cs.CL", "cs.AI", "cs.LG"]
KEYWORDS = [
    "LLM agent", "agentic", "tool use", "agent memory",
    "computer-use agent", "agentic RAG", "agent benchmark",
    "autonomous agent", "multi-agent", "language agent"
]
START_DATE = "20240101"
END_DATE = "20260430"
MAX_PAPERS = 700
OUTPUT_DIR = Path("corpus")
PDF_DIR = OUTPUT_DIR / "pdfs"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
PDF_DIR.mkdir(parents=True, exist_ok=True)

ARXIV_API = "http://export.arxiv.org/api/query"
NS = "{http://www.w3.org/2005/Atom}"

def build_query():
    cat_query = " OR ".join([f"cat:{c}" for c in CATEGORIES])
    kw_query = " OR ".join([f'ti:"{k}" OR abs:"{k}"' for k in KEYWORDS])
    date_filter = f"submittedDate:[{START_DATE}0000 TO {END_DATE}2359]"
    return f"({cat_query}) AND ({kw_query}) AND {date_filter}"

def fetch_batch(query, start, max_results=100):
    params = {
        "search_query": query,
        "start": start,
        "max_results": max_results,
        "sortBy": "submittedDate",
        "sortOrder": "descending"
    }
    resp = requests.get(ARXIV_API, params=params, timeout=30)
    resp.raise_for_status()
    return resp.text

def parse_feed(xml_text):
    root = ET.fromstring(xml_text)
    papers = []
    for entry in root.findall(f"{NS}entry"):
        arxiv_id_raw = entry.find(f"{NS}id").text
        arxiv_id = arxiv_id_raw.split("/abs/")[-1].replace("/", "_")
        title = entry.find(f"{NS}title").text.strip().replace("\n", " ")
        abstract = entry.find(f"{NS}summary").text.strip().replace("\n", " ")
        published = entry.find(f"{NS}published").text[:10]
        authors = [a.find(f"{NS}name").text for a in entry.findall(f"{NS}author")]
        pdf_url = None
        for link in entry.findall(f"{NS}link"):
            if link.attrib.get("title") == "pdf":
                pdf_url = link.attrib["href"]
                break
        if pdf_url is None:
            pdf_url = arxiv_id_raw.replace("/abs/", "/pdf/")
        papers.append({
            "arxiv_id": arxiv_id,
            "title": title,
            "abstract": abstract,
            "published": published,
            "authors": authors,
            "pdf_url": pdf_url
        })
    return papers

def download_pdf(paper, retries=3, min_size_kb=20):
    pdf_path = PDF_DIR / f"{paper['arxiv_id']}.pdf"

    # If file already exists and looks valid, skip
    if pdf_path.exists():
        size_kb = pdf_path.stat().st_size / 1024
        if size_kb >= min_size_kb:
            return True
        else:
            print(f"  Found incomplete file for {paper['arxiv_id']} ({size_kb:.1f} KB), re-downloading...")
            pdf_path.unlink()  # delete broken small file

    for attempt in range(retries):
        try:
            resp = requests.get(paper["pdf_url"], timeout=30, stream=True)
            if resp.status_code == 200:
                with open(pdf_path, "wb") as f:
                    for chunk in resp.iter_content(chunk_size=8192):
                        if chunk:
                            f.write(chunk)

                size_kb = pdf_path.stat().st_size / 1024
                if size_kb < min_size_kb:
                    print(f"  Downloaded file too small for {paper['arxiv_id']} ({size_kb:.1f} KB), retrying...")
                    pdf_path.unlink(missing_ok=True)
                    continue

                return True

        except Exception as e:
            print(f"  Retry {attempt+1} failed for {paper['arxiv_id']}: {e}")
            time.sleep(2)

    return False

def main():
    query = build_query()
    print(f"Query: {query[:120]}...")
    all_papers = []
    seen_ids = set()
    start = 0
    batch_size = 100

    while len(all_papers) < MAX_PAPERS:
        print(f"Fetching batch start={start} ...")
        try:
            xml_text = fetch_batch(query, start, batch_size)
        except Exception as e:
            print(f"API error: {e}")
            break

        batch = parse_feed(xml_text)
        if not batch:
            print("No more results.")
            break

        new = 0
        for p in batch:
            if p["arxiv_id"] not in seen_ids:
                seen_ids.add(p["arxiv_id"])
                all_papers.append(p)
                new += 1

        print(f"  Got {new} new papers. Total so far: {len(all_papers)}")
        start += batch_size
        time.sleep(3)

    meta_path = OUTPUT_DIR / "metadata.jsonl"
    with open(meta_path, "w") as f:
        for p in all_papers:
            f.write(json.dumps(p) + "\n")
    print(f"\nSaved {len(all_papers)} papers to {meta_path}")

    print("\nDownloading PDFs...")
    success, fail = 0, 0
    for i, p in enumerate(all_papers):
        print(f"[{i+1}/{len(all_papers)}] {p['arxiv_id']} - {p['title'][:60]}")
        if download_pdf(p):
            success += 1
        else:
            fail += 1
            print(f"  FAILED: {p['arxiv_id']}")
        time.sleep(1)

    print(f"\nDone. Downloaded: {success}, Failed: {fail}")

if __name__ == "__main__":
    main()