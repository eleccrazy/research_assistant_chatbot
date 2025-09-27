"""
File: vector_database.py
Description: Wrapper around ChromaDB for storing and querying vector embeddings.
Author: Gizachew Kassa
Date: 23/09/2025
"""

from typing import List
import chromadb


class VectorDatabase:
    """Wrapper class for managing a ChromaDB collection."""

    def __init__(self, persist_path: str = "./research_db", collection_name: str = "ml_publications", distance_metric: str = "cosine"):
        """
        Initialize a persistent ChromaDB collection.

        Args:
            persist_path (str): Directory where the database is stored.
            collection_name (str): Name of the collection to create/use.
            distance_metric (str): Similarity metric (e.g., "cosine", "l2", "ip").
        """
        self.client = chromadb.PersistentClient(path=persist_path)
        self.collection = self.client.get_or_create_collection(
            name=collection_name, metadata={"hnsw:space": distance_metric}
        )

    def add_chunks(self, chunk_ids: List[str], documents: List[str], metadatas: List[dict], embeddings: List[List[float]]):
        """
        Add a batch of chunks to the collection.

        Args:
            chunk_ids (List[str]): Unique IDs for each chunk.
            documents (List[str]): Chunk text content.
            metadatas (List[dict]): Metadata dictionaries for each chunk.
            embeddings (List[List[float]]): Vector embeddings for each chunk.
        """
        self.collection.add(
            ids=chunk_ids,
            documents=documents,
            metadatas=metadatas,
            embeddings=embeddings,
        )

    def query_by_vector(self, query_vector: List[float], top_k: int = 5) -> dict:
        """
        Search for the most similar chunks to a given vector.

        Args:
            query_vector (List[float]): Embedding of the query.
            top_k (int, optional): Number of results to return. Defaults to 5.

        Returns:
            dict: Query results including documents and metadata.
        """
        return self.collection.query(query_embeddings=[query_vector], n_results=top_k)
