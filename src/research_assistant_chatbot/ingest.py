"""
File: ingest.py
Description: This module handles ingestion of publication documents into the RAG vector database.
Author: Gizachew Kassa
Date Created: 19/10/2025
"""

from services.chunker import Chunker
from services.embeddings import EmbeddingService
from services.vector_database import VectorDatabase
from utils.helpers import load_publications
from core.rag_pipeline import RAGPipeline

def main():
    """Runs the ingestion pipeline to process and store documents."""
    # Initialize core services
    chunker = Chunker(chunk_size=1000, chunk_overlap=200)
    embedder = EmbeddingService(device='cpu')
    vectordb = VectorDatabase()

    # Create RAG pipeline
    rag = RAGPipeline(chunker=chunker, embedder=embedder, vectordb=vectordb)

    # Load raw documents
    documents = load_publications()

    # Ingest documents
    rag.ingest_publications(documents)
    print(f"✅ Successfully ingested {len(documents)} documents into the vector database.")

if __name__ == "__main__":
    main()
