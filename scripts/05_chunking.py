import os
import re
import numpy as np
import pandas as pd
from tqdm import tqdm
from pinecone import Pinecone, ServerlessSpec
from sentence_transformers import SentenceTransformer

MODEL_NAME = "allenai/specter2_base"
VECTOR_DIM = 768
BATCH_SIZE = 100
N_PAPERS   = 30
CHUNK_WORDS = 60
OVERLAP_WORDS = 15
MAX_SEMANTIC_WORDS = 60
INDEX_FIXED    = "arxiv-chunks-fixed"
INDEX_SEMANTIC = "arxiv-chunks-semantic"

pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"])
model = SentenceTransformer(MODEL_NAME)
df = pd.read_parquet("/content/data/arxiv_subset.parquet")

df["abstract_len"] = df["abstract"].str.split().str.len()
longest = df.nlargest(N_PAPERS, "abstract_len").reset_index(drop=True)
print(f"Відібрано {len(longest)} статей")

def fixed_size_chunks(text, size=CHUNK_WORDS, overlap=OVERLAP_WORDS):
    words = text.split()
    chunks = []
    step = size - overlap
    for start in range(0, len(words), step):
        chunk = " ".join(words[start:start + size])
        if chunk:
            chunks.append(chunk)
        if start + size >= len(words):
            break
    return chunks

def semantic_chunks(text, max_words=MAX_SEMANTIC_WORDS):
    sentences = re.split(r"(?<=[.!?])\s+", text)
    chunks, current, cl = [], [], 0
    for sent in sentences:
        n = len(sent.split())
        if current and cl + n > max_words:
            chunks.append(" ".join(current))
            current, cl = [], 0
        current.append(sent)
        cl += n
    if current:
        chunks.append(" ".join(current))
    return chunks

def ensure_index(name):
    if name not in pc.list_indexes().names():
        pc.create_index(
            name=name, dimension=VECTOR_DIM, metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-east-1"),
        )
        print(f"Індекс {name} створено")
    return pc.Index(name)

def build_and_upload(index_name, chunk_fn, label):
    index = ensure_index(index_name)
    records = []
    for _, row in longest.iterrows():
        for chunk_no, chunk_text in enumerate(chunk_fn(row["abstract"])):
            records.append({
                "arxiv_id": str(row["id"]),
                "title":    str(row["title"]),
                "chunk":    chunk_text,
                "chunk_no": chunk_no,
                "year":     int(row["year"]),
                "category": str(row["category"]),
            })
    print(f"\n[{label}] Всього чанків: {len(records)}")
    embs = model.encode(
        [r["chunk"] for r in records],
        batch_size=64, show_progress_bar=True, normalize_embeddings=True,
    )
    for start in tqdm(range(0, len(records), BATCH_SIZE), desc=f"Завантаження {index_name}"):
        batch = []
        for offset in range(start, min(start + BATCH_SIZE, len(records))):
            r = records[offset]
            batch.append({
                "id": f"{r['arxiv_id']}_chunk_{r['chunk_no']}",
                "values": embs[offset].tolist(),
                "metadata": {
                    "arxiv_id": r["arxiv_id"],
                    "title":    r["title"],
                    "chunk":    r["chunk"][:500],
                    "chunk_no": r["chunk_no"],
                    "year":     r["year"],
                    "category": r["category"],
                },
            })
        index.upsert(vectors=batch)
    print(f"[{label}] В індексі: {index.describe_index_stats()['total_vector_count']} векторів")
    return index

index_fixed    = build_and_upload(INDEX_FIXED,    fixed_size_chunks, "FIXED-SIZE")
index_semantic = build_and_upload(INDEX_SEMANTIC, semantic_chunks,   "SEMANTIC")

def search_chunks(index, query, top_k=5):
    q = model.encode(query, normalize_embeddings=True).tolist()
    res = index.query(vector=q, top_k=top_k, include_metadata=True)
    for m in res["matches"]:
        md = m["metadata"]
        print(f"  score={m['score']:.4f} | чанк #{int(md['chunk_no'])} | {md['title'][:60]}")
        print(f"    {md['chunk'][:150]}...")
    return res

test_queries = [
    "quantum entanglement and information theory",
    "numerical methods for differential equations",
    "statistical analysis of galaxy formation",
]

for query in test_queries:
    print(f"\n{'='*80}\nЗАПИТ: '{query}'\n{'='*80}")
    print("\n--- FIXED-SIZE CHUNKS ---")
    search_chunks(index_fixed, query)
    print("\n--- SEMANTIC CHUNKS ---")
    search_chunks(index_semantic, query)
