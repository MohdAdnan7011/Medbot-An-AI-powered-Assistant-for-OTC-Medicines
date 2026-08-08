# Medbot-An-AI-powered-Assistant-for-OTC-Medicines

# 💊 HealthMedBot – AI-Powered OTC Medicine Assistant

HealthMedBot is a **safety-aware Retrieval-Augmented Generation (RAG) chatbot** designed to provide information about Over-the-Counter (OTC) medicines.

It combines a curated medicine dataset, semantic search, a vector database, and a local LLM to generate context-grounded responses.

---

## 🎯 Objectives

- Provide OTC medicine information through a conversational interface.
- Use a curated OTC medicine dataset as the knowledge source.
- Reduce hallucinations using RAG.
- Retrieve relevant information using semantic search.
- Add a safety validation layer for safer responses.
- Run the LLM locally for privacy and cost efficiency.

---

## 🏗️ System Architecture

```text
                 DATA INGESTION
                      
Medicine Dataset
      ↓
Data Cleaning & OTC Filtering
      ↓
Text Chunking
      ↓
MiniLM Embeddings
      ↓
ChromaDB Vector Database
             │
             │
             ▼
        QUERY PIPELINE

User Query (Streamlit)
      ↓
Query Embedding
      ↓
Semantic Similarity Search
      ↓
Relevant Medical Context
      ↓
Prompt + Context
      ↓
Phi-3 Mini (Ollama)
      ↓
Safety Validation
      ↓
Final Response


🛠️ Technologies Used
Technology	Purpose
Python	Backend & data processing
Streamlit	Chatbot interface
LangChain	RAG orchestration
Hugging Face	Embeddings
MiniLM-L6-v2	384-dimensional embeddings
ChromaDB	Vector database
Ollama	Local LLM execution
Phi-3 Mini	Response generation
Pandas	Dataset processing
