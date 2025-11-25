# Mini RAG Question Answering Bot (Using SQuAD)

This project implements a **Retrieval-Augmented Generation (RAG) Question Answering Bot** using the SQuAD dataset. It demonstrates an end-to-end pipeline for processing raw text, building embeddings, retrieving relevant context, and generating answers using an LLM.

---

## Features

- Load and preprocess SQuAD dataset (v1.1 or v2.0)
- Chunk context documents into manageable pieces
- Generate embeddings and build a FAISS vector store
- Retrieve top-k relevant context chunks for any question
- Generate context-aware answers using an LLM (e.g., `flan-t5-small`)
- (Planned) Simple Gradio UI for interactive querying

---
