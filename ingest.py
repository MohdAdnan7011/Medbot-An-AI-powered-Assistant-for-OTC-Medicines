# ingest.py - Improved ingestion pipeline for HealthMedBot

import os
import pandas as pd
from dotenv import load_dotenv

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

# ----------------------------
# Load environment variables
# ----------------------------
load_dotenv()

# ----------------------------
# Configuration
# ----------------------------
DATA_PATH = os.path.join("data", "medication_facts.csv")
CHROMA_PATH = "chroma_db"

EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"

# Allowed OTC medicines
SAFE_OTC_MEDICINES = [
    "paracetamol",
    "acetaminophen",
    "ibuprofen",
    "aspirin",
    "loperamide",
    "antacid",
    "cetirizine",
    "loratadine",
    "diphenhydramine"
]

# ----------------------------
# Load and Clean Dataset
# ----------------------------
def load_dataset(file_path):

    print(f"\n Loading dataset from: {file_path}")

    if not os.path.exists(file_path):
        raise FileNotFoundError(
            f"Dataset not found at {file_path}. Please place medication_facts.csv in the data folder."
        )

    df = pd.read_csv(file_path)

    # Normalize columns
    df.columns = df.columns.str.strip().str.lower()

    # Fill missing values
    df = df.fillna("unknown")

    # Remove duplicates
    df = df.drop_duplicates()

    print(f" Dataset loaded with {len(df)} rows")

    return df


# ----------------------------
# Add OTC classification
# ----------------------------
def classify_otc_status(drug_name):

    drug_name = str(drug_name).lower()

    for safe_drug in SAFE_OTC_MEDICINES:
        if safe_drug in drug_name:
            return "OTC"

    return "Prescription_or_Unknown"


# ----------------------------
# Build Context for RAG
# ----------------------------
def build_context(df):

    contexts = []

    for _, row in df.iterrows():

        drug = row.get("drug_name", "unknown")
        category = row.get("category", "unknown")
        fact_type = row.get("fact_type", "unknown")
        detail = row.get("fact_detail", "unknown")

        otc_status = classify_otc_status(drug)

        context = f"""
MEDICINE NAME: {drug}

CATEGORY: {category}

OTC STATUS: {otc_status}

FACT TYPE: {fact_type}

DETAILS:
{detail}

MEDICAL NOTE:
This information is for educational purposes. Only OTC medicines should be recommended.
"""

        contexts.append({
            "text": context,
            "metadata": {
                "drug_name": drug,
                "category": category,
                "fact_type": fact_type,
                "otc_status": otc_status
            }
        })

    return contexts


# ----------------------------
# Chunk Text for Embeddings
# ----------------------------
def chunk_documents(contexts):

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=80,
        length_function=len
    )

    texts = []
    metadatas = []

    for item in contexts:

        chunks = splitter.split_text(item["text"])

        for chunk in chunks:

            if chunk.strip():

                texts.append(chunk)

                metadatas.append(item["metadata"])

    print(f" Created {len(texts)} text chunks for embedding")

    return texts, metadatas


# ----------------------------
# Create Chroma Vector DB
# ----------------------------
def create_chroma_index(texts, metadatas):

    print("\n Generating embeddings...")

    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL_NAME
    )

    # Remove existing DB if present
    if os.path.exists(CHROMA_PATH):
        print(" Removing old vector database...")
        import shutil
        shutil.rmtree(CHROMA_PATH)

    db = Chroma.from_texts(
        texts=texts,
        embedding=embeddings,
        metadatas=metadatas,
        persist_directory=CHROMA_PATH
    )

    db.persist()

    print(f" Successfully stored {db._collection.count()} vectors")
    print(" Vector database ready")


# ----------------------------
# Main pipeline
# ----------------------------
def run_ingestion():

    print("\n Starting ingestion pipeline...")

    df = load_dataset(DATA_PATH)

    contexts = build_context(df)

    texts, metadatas = chunk_documents(contexts)

    create_chroma_index(texts, metadatas)

    print("\n INGESTION COMPLETED SUCCESSFULLY")


# ----------------------------
# Script entry
# ----------------------------
if __name__ == "__main__":

    try:
        run_ingestion()

    except Exception as e:
        print("\n ERROR DURING INGESTION")
        print(str(e))