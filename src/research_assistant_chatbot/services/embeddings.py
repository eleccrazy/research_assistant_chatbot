"""
File: embeddings.py
Description: This module provides functionality for generating embeddings of text
documents and queries using Hugging Face models.
Author: Gizachew Kassa
Date: 27/09/2025
"""

from typing import List
import torch
from langchain_huggingface import HuggingFaceEmbeddings


class EmbeddingService:
    """
    A utility class for embedding documents and queries.

    This class wraps LangChain's `HuggingFaceEmbeddings` and provides a simple
    interface for converting texts into vector representations, which can then
    be used in similarity search and retrieval pipelines.

    Attributes:
        model (HuggingFaceEmbeddings): The underlying embedding model.

    Methods:
        embed_documents(texts: List[str]) -> List[List[float]]:
            Generate embeddings for a list of text documents.

        embed_query(text: str) -> List[float]:
            Generate an embedding for a single query string.
    """

    def __init__(self, model_name="sentence-transformers/all-MiniLM-L6-v2", device: str | None = None):
        self.model = HuggingFaceEmbeddings(
            model_name=model_name,
            model_kwargs={
                "device": device
                or ("cuda" if torch.cuda.is_available() else (
                    "mps" if torch.backends.mps.is_available() else "cpu"
                ))
            },
        )

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """
        Generate embeddings for a list of text documents.

        Args:
            texts (List[str]): Input documents.

        Returns:
            List[List[float]]: Embeddings for each document.
        """
        return self.model.embed_documents(texts)

    def embed_query(self, text: str) -> List[float]:
        """
        Generate an embedding for a single query string.

        Args:
            text (str): Input query.

        Returns:
            List[float]: Embedding vector for the query.
        """
        return self.model.embed_query(text)
