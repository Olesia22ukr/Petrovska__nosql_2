import os
import numpy as np
import pandas as pd
from tqdm import tqdm
from pinecone import Pinecone, ServerlessSpec

INPUT_PARQUET    = "/content/data/arxiv_subset.parquet"
INPUT_EMBEDDINGS = "/content/embeddings/embeddings.npy"
INDEX_NAME       = "arxiv-papers"
VECTOR_DIM       = 768
BATCH_SIZE       = 200

pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"])

if INDEX_NAME not in pc.list_indexes().names():
    pc.create_index(
        name=INDEX_NAME,
        dimension=VECTOR_DIM,
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1"),
    )
    print(f"Індекс {INDEX_NAME} створено")
else:
    print(f"Індекс {INDEX_NAME} вже існує")

index = pc.Index(INDEX_NAME)

df = pd.read_parquet(INPUT_PARQUET)
embeddings = np.load(INPUT_EMBEDDINGS)
print(f"Записів: {len(df)}, ембеддингів: {embeddings.shape}")

for start in tqdm(range(0, len(df), BATCH_SIZE), desc="Завантаження в Pinecone"):
    batch_df  = df.iloc[start:start + BATCH_SIZE]
    batch_emb = embeddings[start:start + BATCH_SIZE]
    vectors = []
    for offset, (_, row) in enumerate(batch_df.iterrows()):
        i = start + offset
        vectors.append({
            "id": f"paper_{i}",
            "values": batch_emb[offset].tolist(),
            "metadata": {
                "arxiv_id": str(row["id"]),
                "title":    str(row["title"]),
                "abstract": str(row["abstract"])[:500],
                "authors":  str(row["authors"])[:200],
                "year":     int(row["year"]),
                "category": str(row["category"]),
            },
        })
    index.upsert(vectors=vectors)

stats = index.describe_index_stats()
print(f"\nЗагальна кількість векторів в індексі: {stats['total_vector_count']}")
