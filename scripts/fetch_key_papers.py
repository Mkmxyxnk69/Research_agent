import requests
import time
from pathlib import Path

PDF_DIR = Path("corpus/pdfs")
PDF_DIR.mkdir(parents=True, exist_ok=True)

KEY_PAPERS = [
    ("2504.19413", "Mem0"),
    ("2404.07972", "OSWorld"),
    ("2405.15793", "SWE-agent"),
    ("2407.18901", "AppWorld"),
    ("2407.16741", "OpenHands"),
    ("2501.12326", "UI-TARS"),
    ("2504.02119", "UI-TARS-2"),
    ("2502.12110", "A-MEM"),
    ("2406.12045", "Tau-bench"),
    ("2501.12273", "OS-MAP"),
]

def download(arxiv_id, name):
    pdf_path = PDF_DIR / f"{arxiv_id}v1.pdf"
    if pdf_path.exists() and pdf_path.stat().st_size > 10000:
        print(f"  Already exists: {name}")
        return True
    url = f"https://arxiv.org/pdf/{arxiv_id}"
    print(f"Downloading {name} ({arxiv_id})...")
    try:
        resp = requests.get(url, timeout=60, stream=True)
        if resp.status_code == 200:
            with open(pdf_path, "wb") as f:
                for chunk in resp.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
            size_kb = pdf_path.stat().st_size / 1024
            print(f"  Done: {size_kb:.1f} KB")
            return True
        else:
            print(f"  Failed: HTTP {resp.status_code}")
            return False
    except Exception as e:
        print(f"  Error: {e}")
        return False

def main():
    success, fail = 0, 0
    for arxiv_id, name in KEY_PAPERS:
        if download(arxiv_id, name):
            success += 1
        else:
            fail += 1
        time.sleep(3)
    print(f"\nDone! Downloaded: {success}, Failed: {fail}")

if __name__ == "__main__":
    main()
