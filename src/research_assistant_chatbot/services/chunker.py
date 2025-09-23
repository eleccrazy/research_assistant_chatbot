"""
File: chunker.py
Description: This module provides functionality for splitting large text documents into
smaller, manageable chunks
Author: Gizachew Kassa
Date: 23/09/2025
"""

from langchain_text_splitters import RecursiveCharacterTextSplitter


class Chunker:
    """
    A utility class for splitting documents into context-preserving chunks.

    This class wraps LangChain's `RecursiveCharacterTextSplitter` and is designed
    for use in research assistant pipelines. Each chunk is assigned metadata,
    including the original publication title and a unique chunk ID, which makes
    it easier to trace search results back to their source documents.

    Attributes:
        chunk_size (int): Maximum number of characters in each chunk.
        chunk_overlap (int): Number of overlapping characters between chunks.
        text_splitter (RecursiveCharacterTextSplitter): Internal splitter instance.

    Methods:
        split_publication(publication: dict) -> list[dict]:
            Splits a single publication (title + description) into chunks with metadata.
    """
    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200):
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", ". ", " ", ""],
        )

    def split_publications(self, publications: list[dict]) -> list[dict]:
        """Split JSON publications into chunks with metadata.

        Args:
            publications (list[dict]): Each publication dict should have keys:
                - id
                - username
                - license
                - title
                - publication_description

        Returns:
            list[dict]: Chunks with content and metadata.
        """
        all_chunks = []

        for pub in publications:
            pub_id = pub.get("id", "unknown")
            title = pub.get("title", "Untitled")
            description = pub.get("publication_description", "")

            # Split description into chunks
            chunks = self.text_splitter.split_text(description)

            for i, chunk in enumerate(chunks):
                all_chunks.append(
                    {
                        "content": chunk,
                        "metadata": {
                            "id": pub_id,
                            "username": pub.get("username"),
                            "license": pub.get("license"),
                            "title": title,
                            "chunk_id": f"{pub_id}_{i}",
                        },
                    }
                )

        return all_chunks
