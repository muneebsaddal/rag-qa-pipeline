# Mini RAG Question Answering Bot (SQuAD)

This repository contains an implementation of a Retrieval-Augmented Generation (RAG) pipeline designed to perform accurate, source-grounded Question Answering (QA) using a structured knowledge base derived from the SQuAD dataset.

The system works by intelligently retrieving relevant document chunks (Contexts) using vector search and then feeding those chunks to a smaller Large Language Model (LLM) to generate a precise answer, mitigating the risks of hallucination.

![1](https://github.com/user-attachments/assets/92d3ddd5-8aec-4714-b59d-223bcce8f2e1)

## Architecture and Components

The pipeline is built on the following key components:

- Knowledge Base: The SQuAD dataset is processed and chunked into small, manageable text segments.

- Vector Store: The text chunks are transformed into high-dimensional vectors (embeddings).

- FAISS Index: A highly efficient index is used for rapid similarity search (retrieval) of the most relevant chunks given a user query.

- Generative Model: A smaller, faster language model (flan-t5-small) is prompted with the retrieved context to synthesize the final answer.

## Technologies & Libraries

- Core Language: Python

- Embedding Model: all-MiniLM-L6-v2 (Sentence Transformers)

- Vector Database: FAISS (for vector indexing and fast similarity search)

- Generative Model: flan-t5-small (or similar Hugging Face model)

## Dataset: SQuAD v2.0

## ML Framework: PyTorch / Hugging Face ecosystem
