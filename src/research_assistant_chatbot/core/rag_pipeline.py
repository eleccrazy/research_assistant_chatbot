"""
File: rag_pipeline.py
Description: This module defines the RAGPipeline class, which orchestrates the
             Retrieval-Augmented Generation (RAG) workflow.
Author: Gizachew Kassa
Date: 05/10/2025
"""

from typing import List, Dict
from utils.helpers import load_publications
from services.chunker import Chunker
from services.embeddings import EmbeddingService
from services.vector_database import VectorDatabase


class RAGPipeline:
    """
    A high-level pipeline for managing document ingestion and retrieval
    in a (RAG) system.

    This class integrates the chunking, embedding, and vector database services
    to provide an end-to-end workflow — from loading publications to performing
    similarity-based document retrieval for queries.

    Attributes:
        chunker (Chunker): Instance responsible for splitting documents into chunks.
        embedder (EmbeddingService): Service used to generate embeddings for chunks and queries.
        vectordb (VectorDatabase): Storage layer for chunk embeddings and metadata.

    Methods:
        ingest_publications(publications: list[dict]) -> None:
            Splits documents into chunks, generates embeddings, and stores them in the vector database.

        query(query_text: str, top_k: int = 5) -> list[dict]:
            Embeds a user query, retrieves the most similar chunks from the vector database,
            and formats the search results.
    """

    def __init__(self,
                 chunker: Chunker,
                 embedder: EmbeddingService,
                 vectordb: VectorDatabase):
        """
        Initialize the RAGPipeline with its core components.

        Args:
            chunker (Chunker): The text chunking utility.
            embedder (EmbeddingService): The embedding generator for text data.
            vectordb (VectorDatabase): The vector database for storing and querying embeddings.
        """
        self.chunker = chunker
        self.embedder = embedder
        self.vectordb = vectordb

    def ingest_publications(self, publications: list[dict]) -> None:
        """
        Ingests a list of publications into the RAG system.

        This method splits publications into chunks, generates embeddings for each chunk,
        and stores both the embeddings and metadata in the vector database.

        Args:
            publications (list[dict]): A list of publication dictionaries containing content,
            metadata, and chunk ids.

        Returns:
            None
        """
        if not publications:
            print("No documents found to ingest.")
            return

        # Generate chunks from input publications
        chunks = self.chunker.split_publications(publications)

        # Extract textual content and metadata
        texts = [chunk.get('content') for chunk in chunks]
        metadatas = [chunk.get('metadata') for chunk in chunks]
        ids = [metadata.get('chunk_id') for metadata in metadatas]

        # Generate embeddings for the chunk texts
        embeddings = self.embedder.embed_documents(texts)

        # Store chunks and their embeddings in the vector database
        self.vectordb.add_chunks(
            chunk_ids=ids,
            documents=texts,
            metadatas=metadatas,
            embeddings=embeddings
        )

        print(f"Ingested {len(chunks)} chunks to vector DB.")

    def query(self, query_text: str, top_k: int = 5) -> List[Dict]:
        """
        Perform a semantic similarity search against the vector database.

        This method embeds the user’s query, retrieves the most similar chunks
        from the vector store, and formats the results for downstream use.

        Args:
            query_text (str): The natural language query text.
            top_k (int, optional): Number of top similar chunks to retrieve. Defaults to 5.

        Returns:
            list[dict]: A list of retrieved chunks, each containing:
                - content (str): The chunk text.
                - metadata (dict): Associated metadata (e.g., title, author, chunk_id).
                - similarity (float): The cosine similarity score (1.0 - distance).
        """
        # Embed the query text
        qv = self.embedder.embed_query(query_text)

        # Query the vector database for most similar chunks
        results = self.vectordb.query_by_vector(query_vector=qv, top_k=top_k)

        formatted = []
        docs = results.get("documents", [[]])[0]
        metadatas = results.get("metadatas", [[]])[0]
        distances = results.get("distances", [[]])[0]

        # Format the retrieved results with similarity scores
        for i, content in enumerate(docs):
            formatted.append({
                "content": content,
                "metadata": metadatas[i] if i < len(metadatas) else {},
                "similarity": 1.0 - distances[i] if i < len(distances) else None
            })

        return formatted
