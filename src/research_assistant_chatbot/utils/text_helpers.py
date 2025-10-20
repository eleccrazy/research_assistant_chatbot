"""
File: text_helpers.py
Description: Provides text utility functions such as token counting, message formatting, and safe fallbacks for LLM processing.
Author: Gizachew Kassa
Date Created: 20/10/2025
"""

import tiktoken
from typing import List, Dict, Union
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage


def count_tokens(text: str, model: str = "gemini-1.5-flash") -> int:
    """
    Estimates the number of tokens in a given text for a specific model.

    Attempts to use the tiktoken library for accurate tokenization, with a fallback
    approximation if the model encoding is unavailable.

    Args:
        text (str): The input text to tokenize.
        model (str): The name of the model for which tokenization is estimated.

    Returns:
        int: Estimated number of tokens.
    """
    try:
        encoding = tiktoken.encoding_for_model(model)
        return len(encoding.encode(text))
    except Exception:
        # Fallback estimation based on word count
        return int(len(text.split()) * 1.3)


def messages_to_string(
    messages: List[Union[SystemMessage, HumanMessage, AIMessage]],
) -> str:
    """
    Converts a list of LangChain message objects into a readable plain-text conversation string.

    Designed for RAG-based assistants that work on document chunks rather than full publication texts.

    Args:
        messages (list): List of LangChain message objects (SystemMessage, HumanMessage, AIMessage).

    Returns:
        str: Readable text representation of the conversation history.
    """
    content = []
    user_question_count = 0

    for i, msg in enumerate(messages):
        # System message
        if isinstance(msg, SystemMessage):
            content.append(f"SYSTEM: {msg.content}")

        # User message
        elif isinstance(msg, HumanMessage):
            user_question_count += 1
            if i > 0:
                content.append("=" * 80)
            content.append(f"USER Q{user_question_count}: {msg.content}")

        # Assistant message
        elif isinstance(msg, AIMessage):
            content.append(f"ASSISTANT: {msg.content}")

        else:
            # Fallback for unknown types
            content.append(f"UNKNOWN ({type(msg).__name__}): {getattr(msg, 'content', str(msg))}")

    return "\n\n".join(content)
