"""
File: chatbot.py
Description: This module defines the Chatbot class that orchestrates the RAG-based conversational workflow between the user, LLM, memory, and retrieval components.
Author: Gizachew Kassa
Date Created: 19/10/2025
"""
from core.prompt_builder import build_system_prompt_from_config
from core.rag_pipeline import RAGPipeline
from datetime import datetime
from langchain_core.runnables import RunnableSequence, RunnableLambda
from langchain_core.prompts import ChatPromptTemplate
from typing import Dict, Any
from langchain_core.language_models import BaseChatModel


class Chatbot:
    """
        Chatbot class that orchestrates a multi-step RAG workflow using a language model,
        retrieval pipeline, and structured prompt generation.

        The chatbot handles the end-to-end process of:
        - Retrieving relevant document chunks from the vector database.
        - Formatting the retrieved context.
        - Building a complete prompt that combines system instructions, context, and user query.
        - Sending the final prompt to the LLM for response generation.
    """
    def __init__(
            self,
            rag_pipeline: RAGPipeline,
            system_prompt_config: str,
            llm_client: BaseChatModel
            ) -> None:
        """
                Initializes the Chatbot with RAG components, system prompt, and an LLM client.

                Args:
                    rag_pipeline (RAGPipeline): The RAG pipeline instance for document retrieval.
                    system_prompt (str): The system-level instruction prompt.
                    llm_client (BaseChatModel): The chat language model client instance.
        """
        self.rag = rag_pipeline
        self.system_prompt = build_system_prompt_from_config(system_prompt_config)
        self.llm = llm_client
        self.chain = self._build_runnable_chain()

    def _build_runnable_chain(self) -> RunnableSequence:
        """
        Builds a multi-step RAG chain using LangChain's RunnableSequence.

        This method defines a composable, declarative pipeline that connects
        document retrieval, context formatting, prompt construction, and LLM
        invocation into a single runnable sequence. Each step’s output is passed
        to the next, enabling a clear and modular RAG workflow.

        Returns:
            RunnableSequence: A runnable RAG chain that can be invoked with a user query
                              to perform retrieval, prompt generation, and LLM inference.
        """

        # Document retrieval
        retrieve_docs = RunnableLambda(lambda inputs: {
            "query": inputs["query"],
            "system_prompt": inputs.get("system_prompt", ""),
            "context": self.rag.query(inputs["query"], top_k=5)
        })

        # Format retrieved context
        format_context = RunnableLambda(lambda inputs: {
            "query": inputs["query"],
            "system_prompt": inputs.get("system_prompt", ""),
            "formatted_context": "\n\n".join(
                f"Chunk {i + 1}:\n{chunk}" for i, chunk in enumerate(inputs["context"])
            )
        })

        # Build prompt template
        prompt = ChatPromptTemplate.from_template("""
        System instructions:
        {system_prompt}

        Retrieved context:
        {formatted_context}

        User question:
        {query}
        """)

        # Combine all into a RunnableSequence chain
        chain = RunnableSequence(retrieve_docs | format_context | prompt | self.llm)

        return chain


    def ask(self, user_query: str, user_id=None) -> Dict[str,Any]:
        """
             Handles a single user query using the RAG chain.

             Executes the multi-step RAG process: retrieves context, formats it,
             builds the full LLM prompt, and generates a response through the
             configured language model.

             Args:
                 user_query (str): The user's input question or request.

             Returns:
                 dict: A structured response containing the query, model output,
                       timestamp, and additional metadata.
             """
        result = self.chain.invoke({
            "query": user_query,
            "system_prompt": self.system_prompt
        })

        return {
            "query": user_query,
            "response": result.content,
            "metadata": {
                "timestamp": datetime.now().isoformat(),
                "model": getattr(self.llm, "model", "unknown")
            }
        }
