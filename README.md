# Vector Database Homework

## Comprehensive Survey on Vector Database

Course: NoSQL & Vector Databases

Author: Olesia Petrovska

---

# Project Overview

This project demonstrates the implementation of a semantic search system using vector databases.

The project includes:

- arXiv dataset processing;
- text embeddings generation using SPECTER2 model;
- vector storage in Pinecone;
- semantic search using cosine similarity;
- keyword search;
- hybrid search using vector search and BM25;
- chunking strategies comparison.

---

# Part 1. Dataset Processing and Model Selection

## Dataset

The project uses the arXiv dataset containing scientific paper abstracts.

The dataset was selected because scientific documents require semantic understanding rather than simple keyword matching.

## Embedding Model Selection

SPECTER2 embeddings model was chosen because it is designed for scientific papers and captures relationships between research documents.

The model converts text documents into numerical vectors that represent their semantic meaning.

## Vector Database Selection

Pinecone was selected as the vector database because:

- it provides fast similarity search;
- it supports large-scale vector storage;
- it allows metadata storage together with vectors;
- it is optimized for Retrieval-Augmented Generation (RAG) systems.

---

# Part 2. Vector Database Index Creation

## Index Configuration

The Pinecone index was created with the following parameters:

- Dimension: 384
- Metric: cosine similarity
- Total vectors uploaded: 1000

## Metadata

Each stored vector contains metadata:

```text
{
"text": original document text
}
# Part 3. Search Methods

Implemented search methods:

## 1. Semantic Search

Description:
...

Example queries:

| Query | Result |
|---|---|
| machine learning neural networks | ... |

## 2. Keyword Search (BM25)

Description:
...

## 3. Hybrid Search

Description:
Combines BM25 and vector similarity.

Results were compared using three queries.

# Part 4. Chunking Strategies

Two approaches were tested:

## Strategy 1: Fixed-size chunks

Description...

## Strategy 2: Semantic chunks

Description...

Comparison:

| Method | Advantages | Disadvantages |
|---|---|---|
| Fixed | simple | may split meaning |
| Semantic | better context | more complex |

# Part 5. Hybrid Search and RRF

Hybrid search combines:

- BM25 keyword relevance
- vector semantic similarity


## Comparison

| Query | BM25 | Vector Search | Hybrid Search |
|---|---|---|---|
| machine learning neural networks | ... | ... | ... |
| artificial intelligence machine learning | ... | ... | ... |
| computer vision deep learning | ... | ... | ... |

# Part 6. Analysis and Conclusions

Vector search advantages:

- understands semantic meaning;
- finds related concepts.

BM25 advantages:

- works well with exact keywords.

Hybrid search:

- combines both approaches;
- improves retrieval quality.

Cosine similarity measures the angle between vectors.
Higher similarity means closer semantic meaning.

Main trade-off:

Vector search provides better semantic understanding,
while keyword search provides better exact matching.
