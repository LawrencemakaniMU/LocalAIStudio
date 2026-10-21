from pathlib import Path
import sys
import chromadb
from pypdf import PdfReader

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "knowledge" / "raw"
DB = ROOT / "data" / "chroma"
client = chromadb.PersistentClient(path=str(DB))
collection = client.get_or_create_collection(name="local_knowledge")

def extract(path: Path) -> str:
    if path.suffix.lower() == ".pdf":
        reader = PdfReader(str(path))
        return "\n".join(page.extract_text() or "" for page in reader.pages)
    if path.suffix.lower() in {".txt", ".md", ".py", ".c", ".cpp", ".h", ".json", ".csv"}:
        return path.read_text(encoding="utf-8", errors="ignore")
    return ""

def chunks(text: str, size=1000, overlap=150):
    text = " ".join(text.split())
    if not text:
        return []
    out, start = [], 0
    while start < len(text):
        end = min(start + size, len(text))
        out.append(text[start:end])
        if end == len(text):
            break
        start = max(end - overlap, start + 1)
    return out

def main():
    paths = [p for p in RAW.rglob("*") if p.is_file() and p.suffix.lower() in
             {".pdf", ".txt", ".md", ".py", ".c", ".cpp", ".h", ".json", ".csv"}]
    if not paths:
        print(f"No supported files found. Add PDF/TXT/MD/code files to: {RAW}")
        return
    # This starter rebuilds the collection each time to avoid duplicate chunks.
    existing = collection.get()
    if existing["ids"]:
        collection.delete(ids=existing["ids"])
    total = 0
    for path in paths:
        text = extract(path)
        pieces = chunks(text)
        ids = [f"{path.relative_to(RAW).as_posix()}::{i}" for i in range(len(pieces))]
        metas = [{"source": path.relative_to(RAW).as_posix(), "chunk": i} for i in range(len(pieces))]
        if pieces:
            collection.add(ids=ids, documents=pieces, metadatas=metas)
            total += len(pieces)
        print(f"{path.relative_to(RAW)}: {len(pieces)} chunks")
    print(f"Done. Indexed {total} chunks from {len(paths)} files.")
    print("Note: scanned PDF text only; scanned-image PDFs need OCR.")

if __name__ == "__main__":
    main()
