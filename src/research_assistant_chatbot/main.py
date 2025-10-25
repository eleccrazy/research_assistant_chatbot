"""
File: main.py
Description: Central application orchestrator for managing RAG chatbot interfaces and utilities.
Author: Gizachew Kassa
Date Created: 25/10/2025
"""

import argparse
from services.chunker import Chunker
from services.embeddings import EmbeddingService
from services.vector_database import VectorDatabase
from services.llms import get_llm
from core.rag_pipeline import RAGPipeline
from core.chatbot import Chatbot
from core.memory import MemoryManager
from utils.paths import PROMPT_CONFIG_FPATH
from utils.helpers import load_yaml_config, load_publications


def initialize_chatbot() -> Chatbot:
    """Initializes and returns a configured Chatbot instance."""
    chunker = Chunker(chunk_size=1000, chunk_overlap=200)
    embedder = EmbeddingService(device="cpu")
    vectordb = VectorDatabase()
    memory_manager = MemoryManager(window_size=6, token_limit=2500)

    rag = RAGPipeline(chunker=chunker, embedder=embedder, vectordb=vectordb)
    config = load_yaml_config(PROMPT_CONFIG_FPATH)
    llm = get_llm("gemini-2.5-flash")

    chatbot = Chatbot(
        rag_pipeline=rag,
        system_prompt_config=config.get("chatbot_prompt", ""),
        llm_client=llm,
        memory_manager=memory_manager,
    )
    return chatbot


def ingest_documents():
    """Rebuilds the vector database by chunking, embedding, and storing all documents."""
    print("Starting document ingestion...")
    chunker = Chunker(chunk_size=1000, chunk_overlap=200)
    embedder = EmbeddingService(device="cpu")
    vectordb = VectorDatabase()
    rag = RAGPipeline(chunker=chunker, embedder=embedder, vectordb=vectordb)
    documents = load_publications()
    rag.ingest_publications(documents)
    print("Ingestion complete! Vector store updated.")


def main():
    """Handles all high-level project operations."""
    parser = argparse.ArgumentParser(description="Research Assistant Chatbot Manager")
    parser.add_argument("--ingest", action="store_true", help="Rebuild and ingest publications into the vector store")
    parser.add_argument("--cli", action="store_true", help="Launch the interactive CLI chat interface")

    args = parser.parse_args()

    if args.ingest:
        ingest_documents()
    elif args.cli:
        from interfaces.cli import run_cli
        print("Launching CLI Chat Interface...\n")
        chatbot = initialize_chatbot()
        run_cli(chatbot)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
