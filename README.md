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
