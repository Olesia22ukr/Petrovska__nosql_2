import os
import numpy as np
import pandas as pd
from pinecone import Pinecone
from sentence_transformers import SentenceTransformer

INDEX_NAME = "arxiv-papers"
MODEL_NAME = "allenai/specter2_base"
TOP_K = 5

pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"])
index = pc.Index(INDEX_NAME)
model = SentenceTransformer(MODEL_NAME)
df = pd.read_parquet("/content/data/arxiv_subset.parquet")

def encode_query(query):
    return model.encode(query, normalize_embeddings=True).tolist()

def print_matches(matches):
    for m in matches:
        md = m["metadata"]
        print(f"  score={m['score']:.4f} | {int(md['year'])} | {md['category']}")
        print(f"  {md['title']}")
        print(f"  {md['abstract'][:200]}...")
        print("  " + "-" * 76)

# 1. Семантичний пошук
query = "teaching machines to recognize objects in pictures"
print(f"\n{'='*80}\nСЕМАНТИЧНИЙ ПОШУК: '{query}'\n{'='*80}")
results = index.query(vector=encode_query(query), top_k=TOP_K, include_metadata=True)
print_matches(results["matches"])

# 2. Фільтр A: cs.LG, після 2021
query_a = "reinforcement learning"
print(f"\n{'='*80}\nФІЛЬТР A: year >= 2021, category == cs.LG\n{'='*80}")
results_a = index.query(
    vector=encode_query(query_a), top_k=TOP_K, include_metadata=True,
    filter={"year": {"$gte": 2021}, "category": {"$eq": "cs.LG"}},
)
print_matches(results_a["matches"])

# 3. Фільтр B: до 2015
print(f"\n{'='*80}\nФІЛЬТР B: year < 2015\n{'='*80}")
results_b = index.query(
    vector=encode_query(query_a), top_k=TOP_K, include_metadata=True,
    filter={"year": {"$lt": 2015}},
)
print_matches(results_b["matches"])

# 4. Порівняння метрик локально
print(f"\n{'='*80}\nПОРІВНЯННЯ МЕТРИК\n{'='*80}")
embeddings = np.load("/content/embeddings/embeddings.npy")
q = np.array(model.encode(query, normalize_embeddings=True))

dot_scores = embeddings @ q
norms = np.linalg.norm(embeddings, axis=1) * np.linalg.norm(q)
cosine_scores = dot_scores / norms
l2_dist = np.linalg.norm(embeddings - q, axis=1)

top_cosine = np.argsort(cosine_scores)[::-1][:TOP_K]
top_dot    = np.argsort(dot_scores)[::-1][:TOP_K]
top_l2     = np.argsort(l2_dist)[:TOP_K]

for name, ids, scores in [
    ("COSINE", top_cosine, cosine_scores),
    ("DOT",    top_dot,    dot_scores),
]:
    print(f"\n--- {name} ---")
    for i in ids:
        print(f"  {scores[i]:.4f} | {df.iloc[i]['title'][:80]}")

print(f"\n--- L2 ---")
for i in top_l2:
    print(f"  {l2_dist[i]:.4f} | {df.iloc[i]['title'][:80]}")

print(f"\nТоп-5 cosine == топ-5 dot: {list(top_cosine) == list(top_dot)}")
print(f"Топ-5 cosine == топ-5 L2:  {list(top_cosine) == list(top_l2)}")
