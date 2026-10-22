"""Index the PDFs in knowledge/raw into Chroma (Sentence Transformers embeddings).

  python scripts\\ingest.py            add NEW books only (already-indexed books are skipped)
  python scripts\\ingest.py --rebuild  wipe the index and re-index every book

Also maintains config/book_manifest.json (one entry per book; fill in the blank
author/edition/year/permission fields by hand - re-running never overwrites them).
"""
from pathlib import Path
import hashlib
import json
import sys

import chromadb
import pymupdf
from sentence_transformers import SentenceTransformer

ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = ROOT / "knowledge" / "raw"
DB_DIR = ROOT / "data" / "chroma"
MANIFEST = ROOT / "config" / "book_manifest.json"
COLLECTION = "local_knowledge"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

CHUNK_SIZE, CHUNK_OVERLAP = 1000, 150
BATCH = 64
TARGET_BOOKS = 15


def split_text(text):
    text = " ".join(text.split())
    chunks, start = [], 0
    while start < len(text):
        end = min(start + CHUNK_SIZE, len(text))
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end == len(text):
            break
        start = end - CHUNK_OVERLAP
    return chunks


def load_manifest():
    if MANIFEST.exists():
        return {b["source"]: b for b in json.loads(MANIFEST.read_text(encoding="utf-8"))}
    return {}


def save_manifest(entries):
    MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    MANIFEST.write_text(json.dumps(sorted(entries.values(), key=lambda b: b["source"]),
                                   indent=2, ensure_ascii=False), encoding="utf-8")


def flush(collection, model, ids, docs, metas):
    if not ids:
        return
    emb = model.encode(docs, batch_size=32, normalize_embeddings=True,
                       convert_to_numpy=True, show_progress_bar=False)
    collection.upsert(ids=ids, documents=docs, embeddings=emb.tolist(), metadatas=metas)
    ids.clear(); docs.clear(); metas.clear()


def main():
    rebuild = "--rebuild" in sys.argv
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    DB_DIR.mkdir(parents=True, exist_ok=True)
    pdfs = sorted(RAW_DIR.glob("*.pdf"))
    if not pdfs:
        print(f"No PDF files found in: {RAW_DIR}")
        return

    client = chromadb.PersistentClient(path=str(DB_DIR))
    if rebuild:
        try:
            client.delete_collection(COLLECTION)
        except Exception:
            pass
        MANIFEST.unlink(missing_ok=True)
    collection = client.get_or_create_collection(name=COLLECTION, metadata={"hnsw:space": "cosine"})
    manifest = load_manifest()

    # Books already in the index (counts come from stored metadata).
    indexed = {}
    for m in collection.get(include=["metadatas"])["metadatas"]:
        indexed[m["source"]] = indexed.get(m["source"], 0) + 1
    for src, n in indexed.items():  # index built before the manifest existed
        manifest.setdefault(src, {
            "book_id": hashlib.sha1(src.encode()).hexdigest()[:12], "source": src,
            "title": Path(src).stem, "author": "", "edition": "", "year": "",
            "permission_status": "", "processing_status": "processed",
            "chunks": n, "pages_without_text": None})

    todo = [p for p in pdfs if p.name not in indexed or p.name not in manifest]
    print(f"{len(pdfs)} PDFs in knowledge/raw | {len(pdfs) - len(todo)} already indexed | {len(todo)} to process")

    if todo:
        print("Loading the Sentence Transformers embedding model...")
        model = SentenceTransformer(EMBEDDING_MODEL, device="cpu")

    for pdf in todo:
        print(f"\nReading: {pdf.name}")
        collection.delete(where={"source": pdf.name})  # clear leftovers from an interrupted run
        try:
            doc = pymupdf.open(pdf)
        except Exception as exc:
            print(f"  Could not open {pdf.name}: {exc}")
            manifest[pdf.name] = {**manifest.get(pdf.name, {}), "source": pdf.name,
                                  "processing_status": f"failed: {exc}"}
            continue

        book_id = hashlib.sha1(pdf.name.encode()).hexdigest()[:12]
        ids, docs, metas, total, empty = [], [], [], 0, 0
        for i in range(len(doc)):
            page_no = i + 1
            chunks = split_text(doc[i].get_text("text"))
            if not chunks:
                empty += 1
            for c, chunk in enumerate(chunks, start=1):
                cid = f"{book_id}_P{page_no:04d}_C{c:03d}"
                ids.append(cid); docs.append(chunk)
                metas.append({"source": pdf.name, "title": pdf.stem, "book_id": book_id,
                              "page": page_no, "chunk_id": cid})
                total += 1
            if len(ids) >= BATCH:
                flush(collection, model, ids, docs, metas)
            if page_no % 100 == 0:
                print(f"  page {page_no}/{len(doc)}")
        flush(collection, model, ids, docs, metas)
        n_pages = len(doc)
        doc.close()

        old = manifest.get(pdf.name, {})
        manifest[pdf.name] = {
            "book_id": book_id, "source": pdf.name, "title": old.get("title", pdf.stem),
            "author": old.get("author", ""), "edition": old.get("edition", ""),
            "year": old.get("year", ""), "permission_status": old.get("permission_status", ""),
            "processing_status": "processed" if total else "needs OCR (no text found)",
            "chunks": total, "pages": n_pages, "pages_without_text": empty}
        save_manifest(manifest)  # saved after every book, so an interruption loses little
        warn = "  <-- mostly scanned? needs OCR or another copy" if n_pages and empty / n_pages > 0.10 else ""
        print(f"  done: {total} chunks, {empty}/{n_pages} pages without text{warn}")

    save_manifest(manifest)
    ok = [b for b in manifest.values() if b.get("processing_status") == "processed"]
    print(f"\nBooks indexed: {len(ok)} (target: at least {TARGET_BOOKS}) | total chunks: {collection.count()}")
    print("Restart the app so it picks up the new books.")


if __name__ == "__main__":
    main()
