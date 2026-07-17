# Vector Database Homework

## Comprehensive Survey on Vector Database

**Course:** NoSQL & Vector Databases  
**Author:** Olesia Petrovska

---

# Project Overview

This project demonstrates the implementation of a semantic search system using vector databases.

The goal of the project is to compare different search approaches for scientific documents:

- semantic vector search;
- keyword-based search;
- hybrid search combining both approaches.

The project uses the arXiv scientific papers dataset and includes:

- dataset processing;
- text embeddings generation using SPECTER2 model;
- vector storage in Pinecone;
- semantic search using cosine similarity;
- keyword search using BM25;
- hybrid search using Reciprocal Rank Fusion (RRF);
- comparison of chunking strategies.

---

# Part 1. Dataset Processing and Model Selection

## Dataset

The project uses the **arXiv dataset** containing scientific paper abstracts.

Scientific documents were selected because they require semantic understanding. Simple keyword matching is often insufficient because related concepts may be described using different words.

## Embedding Model Selection

The **SPECTER2 embedding model** was selected because it is designed specifically for scientific documents.

The model converts text documents into numerical vectors that represent semantic meaning and relationships between research papers.

Generated embeddings have:

- dimension: **384**
- total generated vectors: **1000**

## Vector Database Selection

**Pinecone** was selected as the vector database.

Reasons:

- fast similarity search;
- scalable vector storage;
- metadata support;
- suitable for Retrieval-Augmented Generation (RAG) systems.

---

# Part 2. Vector Database Index Creation

## Pinecone Index Configuration

The vector index was created with the following parameters:

| Parameter | Value |
|---|---|
| Index name | arxiv-search |
| Dimension | 384 |
| Similarity metric | cosine |
| Stored vectors | 1000 |

## Metadata Storage

Each vector contains metadata with the original document text:

```text
{
"text": original document text
}
```

Vectors were uploaded to Pinecone using batch upload.

---

# Part 3. Search Methods

Three search approaches were implemented and compared.

---

## 1. Semantic Vector Search

Semantic search uses embeddings and cosine similarity.

Instead of looking only for exact words, it searches for documents with similar meaning.

Example query:

```
machine learning neural networks
```

Results are ranked by vector similarity score.

---

## 2. Keyword Search (BM25)

BM25 is a traditional keyword-based ranking algorithm.

It searches documents based on word occurrence and relevance.

Advantages:

- good for exact keyword matching;
- simple and fast.

Example:

```
computer vision deep learning
```

---

## 3. Hybrid Search

Hybrid search combines:

- vector similarity search;
- BM25 keyword ranking.

The final ranking is created using:

**Reciprocal Rank Fusion (RRF)**

This approach combines semantic understanding with exact keyword matching.

---

# Part 4. Chunking Strategies

Two chunking approaches were tested.

## Fixed-size Chunking

Documents were divided into equal-size text blocks.

Advantages:

- simple implementation;
- predictable chunk size.

Disadvantages:

- may split important context.

Example result:

```
Fixed chunks: 5771
```

---

## Overlapping Semantic Chunking

Chunks were created with overlap between text segments.

Advantages:

- preserves more context;
- reduces information loss.

Example result:

```
Semantic chunks: 6828
```

---

## Chunking Comparison

| Method | Advantages | Disadvantages |
|---|---|---|
| Fixed chunks | Simple and fast | Can split meaning |
| Semantic chunks | Better context preservation | More complex |

---

# Part 5. Search Results Comparison

Three queries were tested:

| Query |
|---|
| machine learning neural networks |
| artificial intelligence machine learning |
| computer vision deep learning |

Search methods compared:

- Vector Search
- BM25 Search
- Hybrid Search

Hybrid search combines advantages of both approaches:

- semantic similarity from embeddings;
- keyword relevance from BM25.

---

# Part 6. Results Screenshots

The project execution results are provided in:

```
results_screenshots/
```

Included screenshots:

- dataset processing;
- embeddings generation;
- Pinecone index creation;
- vector search results;
- BM25 search results;
- hybrid search results;
- chunking comparison.

---

# Part 7. Analysis and Conclusions

## Vector Search

Advantages:

- understands semantic meaning;
- finds conceptually related documents;
- works even when exact keywords are missing.

## BM25 Search

Advantages:

- effective for exact keyword matching;
- simple and interpretable.

## Hybrid Search

Advantages:

- combines semantic and keyword approaches;
- improves retrieval quality;
- provides more balanced results.

## Final Conclusion

Vector search provides better semantic understanding, while BM25 provides stronger exact keyword matching.

The hybrid approach combines both methods and provides the most flexible search strategy for scientific documents.

The project demonstrates how vector databases can be used for semantic retrieval and Retrieval-Augmented Generation (RAG) systems.
