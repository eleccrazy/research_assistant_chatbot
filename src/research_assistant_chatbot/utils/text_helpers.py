"""
File: text_helpers.py
Description: Provides text utility functions such as token counting, message formatting, and safe fallbacks for LLM processing.
Author: Gizachew Kassa
Date Created: 20/10/2025
"""

import tiktoken
from typing import List, Dict, Union


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


def messages_to_string(messages: List[Dict[str, str]]) -> str:
    """
    Converts a list of role-based message dictionaries into a readable text transcript.

    Designed for RAG-based assistants that store chat messages as simple dictionaries
    with 'role' and 'content' keys.

    Args:
        messages (list[dict]): List of message dictionaries (e.g., {"role": "user", "content": "..."}).

    Returns:
        str: Readable text representation of the conversation history.
    """
    content = []
    user_question_count = 0

    for i, msg in enumerate(messages):
        role = msg.get("role", "").lower()
        text = msg.get("content", "")

        if role == "system":
            content.append(f"SYSTEM: {text}")
        elif role == "user":
            user_question_count += 1
            if i > 0:
                content.append("=" * 80)
            content.append(f"USER Q{user_question_count}: {text}")
        elif role == "assistant":
            content.append(f"ASSISTANT: {text}")
        else:
            content.append(f"UNKNOWN ({role}): {text}")

    return "\n\n".join(content)
