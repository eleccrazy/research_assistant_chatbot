"""
File: chatbot.py
Description: This module defines the Chatbot class that orchestrates the RAG-based conversational workflow between the user, LLM, memory, and retrieval components.
Author: Gizachew Kassa
Date Created: 19/10/2025
"""
from core.prompt_builder import build_system_prompt_from_config
from core.rag_pipeline import RAGPipeline
from services.llms import get_llm
from typing import Dict, Any, Union
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_groq import ChatGroq
from datetime import datetime


class Chatbot:
    """
    """
    def __init__(
            self,
            rag_pipeline: RAGPipeline,
            system_prompt_config: Dict[str, Any],
            llm_client: Union[ChatGoogleGenerativeAI, ChatGroq]
            ) -> None:

        self.rag = rag_pipeline
        self.system_prompt = build_system_prompt_from_config(system_prompt_config)
        self.llm = llm_client

    def _build_user_message(self, retrieved_chunks: list, user_query: str) -> str:
        """
        Builds the user message by combining the retrieved document chunks
        with the user's query into a single formatted message for the LLM.

        Args:
            retrieved_chunks (list): Top-K text chunks retrieved by the RAG pipeline.
            user_query (str): The user's current input or question.

        Returns:
            str: Formatted user message to be passed to the LLM.
        """
        # Combine retrieved chunks into a readable format
        formatted_chunks = "\n\n".join(
            f"Chunk {i + 1}:\n{chunk}" for i, chunk in enumerate(retrieved_chunks)
        ) if retrieved_chunks else "No relevant chunks retrieved."

        # Construct the final user message
        user_message = (
            f"Retrieved context:\n{formatted_chunks}\n\n"
            f"User question:\n{user_query}"
        )

        return user_message

    def ask(self, user_query: str, user_id=None) -> Dict[str,Any]:
        """
        """
        # Retrieve relevant chunks
        chunks = self.rag.query(user_query, top_k=5)

        # Build user message
        user_message = self._build_user_message(chunks, user_query)

        # Call LLM
        response = self.llm.invoke(
            [
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": user_message}
            ]
        )

        # Return structured result
        return {
            "query": user_query,
            "context": chunks,
            "response": response.content,
            "metadata": {
                "timestamp": datetime.now().isoformat(),
                "num_chunks_used": len(chunks)
            }
        }
