import os
import numpy as np
import pandas as pd
from pinecone import Pinecone
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi

INDEX_NAME = "arxiv-papers"
MODEL_NAME = "allenai/specter2_base"
TOP_K = 10
SHOW_K = 5
RRF_K = 60

pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"])
index = pc.Index(INDEX_NAME)
model = SentenceTransformer(MODEL_NAME)
df = pd.read_parquet("/content/data/arxiv_subset.parquet").reset_index(drop=True)

corpus = (df["title"] + " " + df["abstract"]).tolist()
tokenized_corpus = [doc.lower().split() for doc in corpus]
bm25 = BM25Okapi(tokenized_corpus)
print(f"BM25-індекс побудовано за {len(corpus)} документами")

def bm25_search(query, top_k=TOP_K):
    scores = bm25.get_scores(query.lower().split())
    return np.argsort(scores)[::-1][:top_k].tolist()

def vector_search(query, top_k=TOP_K):
    q = model.encode(query, normalize_embeddings=True).tolist()
    res = index.query(vector=q, top_k=top_k, include_metadata=False)
    return [int(m["id"].split("_")[1]) for m in res["matches"]]

def rrf_fuse(ranked_lists, k=RRF_K):
    scores = {}
    for ranked in ranked_lists:
        for rank, doc_id in enumerate(ranked, start=1):
            scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (k + rank)
    return sorted(scores.items(), key=lambda x: x[1], reverse=True)

def hybrid_search(query, top_k=TOP_K):
    return rrf_fuse([bm25_search(query, top_k), vector_search(query, top_k)])

def show(doc_id, score):
    row = df.iloc[doc_id]
    print(f"  {score} | {row['year']} | {row['category']} | {row['title'][:70]}")

queries = [
    "BERT fine-tuning",
    "Yann LeCun convolutional networks",
    "making computers understand human emotions from text",
]

for query in queries:
    print(f"\n{'='*80}\nЗАПИТ: '{query}'\n{'='*80}")

    bm25_ids = bm25_search(query)
    bm25_scores = bm25.get_scores(query.lower().split())
    print(f"\n--- BM25 (топ-{SHOW_K}) ---")
    for doc_id in bm25_ids[:SHOW_K]:
        show(doc_id, f"{bm25_scores[doc_id]:.3f}")

    vec_ids = vector_search(query)
    print(f"\n--- VECTOR (топ-{SHOW_K}) ---")
    for rank, doc_id in enumerate(vec_ids[:SHOW_K], start=1):
        show(doc_id, f"rank {rank}")

    hybrid = hybrid_search(query)
    print(f"\n--- HYBRID/RRF (топ-{SHOW_K}) ---")
    for doc_id, rrf_score in hybrid[:SHOW_K]:
        in_bm25 = "B" if doc_id in bm25_ids[:SHOW_K] else "-"
        in_vec  = "V" if doc_id in vec_ids[:SHOW_K] else "-"
        show(doc_id, f"RRF={rrf_score:.4f} [{in_bm25}{in_vec}]")

    only_hybrid = [d for d, _ in hybrid[:SHOW_K]
                   if d not in bm25_ids[:SHOW_K] and d not in vec_ids[:SHOW_K]]
    if only_hybrid:
        print(f"\n  Тільки в hybrid топ-5: {[f'paper_{d}' for d in only_hybrid]}")

# Вплив параметра k
query = "BERT fine-tuning"
print(f"\n{'='*80}\nВПЛИВ k У RRF (запит: '{query}')\n{'='*80}")
for k in (60, 1):
    fused = rrf_fuse([bm25_search(query), vector_search(query)], k=k)
    print(f"\nk={k}, топ-5:")
    for doc_id, score in fused[:5]:
        show(doc_id, f"RRF={score:.4f}")
