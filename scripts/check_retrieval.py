"""Type questions and see the top passages with similarity scores.
Use it to choose min_similarity in config/settings.json."""
from pathlib import Path
import chromadb
from sentence_transformers import SentenceTransformer

ROOT = Path(__file__).resolve().parent.parent
client = chromadb.PersistentClient(path=str(ROOT / "data" / "chroma"))
collection = client.get_collection("local_knowledge")
model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", device="cpu")
print(f"Loaded {collection.count()} chunks. Type a question (empty line to quit).\n")

while True:
    q = input("Question> ").strip()
    if not q:
        break
    emb = model.encode(q, normalize_embeddings=True).tolist()
    res = collection.query(query_embeddings=[emb], n_results=5,
                           include=["documents", "metadatas", "distances"])
    for doc, meta, dist in zip(res["documents"][0], res["metadatas"][0], res["distances"][0]):
        text = " ".join(doc.split())[:150]
        print(f"  {1 - dist:.3f} | {meta.get('source')} p.{meta.get('page')} | {text}...")
    print()
