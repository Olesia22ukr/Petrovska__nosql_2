import os
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

INPUT_PARQUET = "/content/data/arxiv_subset.parquet"
OUTPUT_FILE   = "/content/embeddings/embeddings.npy"
MODEL_NAME    = "allenai/specter2_base"
BATCH_SIZE    = 64

df = pd.read_parquet(INPUT_PARQUET)
print(f"Завантажено записів: {len(df)}")

texts = (df["title"] + " [SEP] " + df["abstract"]).tolist()

model = SentenceTransformer(MODEL_NAME)
print(f"Модель {MODEL_NAME} завантажена")

embeddings = model.encode(
    texts,
    batch_size=BATCH_SIZE,
    show_progress_bar=True,
    normalize_embeddings=True,
)

print(f"\nОброблено текстів: {len(texts)}")
print(f"Розмірність ембеддингів: {embeddings.shape[1]}")
print(f"Норма першого ембеддингу: {np.linalg.norm(embeddings[0]):.6f}")

os.makedirs("/content/embeddings", exist_ok=True)
np.save(OUTPUT_FILE, embeddings)
print(f"Збережено в {OUTPUT_FILE}, форма масиву: {embeddings.shape}")
