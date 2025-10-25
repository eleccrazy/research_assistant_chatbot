"""
File: memory.py
Description: Handles conversation history, summarization, and context persistence for the chatbot.
Author: Gizachew Kassa
Date Created: 19/10/2025
"""
from collections import deque
from typing import List, Dict, Any

from utils.text_helpers import messages_to_string, count_tokens


class MemoryManager:
    """
    Manages short-term chat history and long-term summarized memory for context retention.
    """
    def __init__(self, window_size: int = 4, token_limit: int = 2500):
        """
        Initializes the memory manager.

        Args:
            window_size (int): Number of recent message pairs to retain.
            token_limit (int): Max tokens to include in summarization input.
        """
        self.history = deque(maxlen=window_size * 2)  # store user+assistant turns
        self.summary = ""  # rolling summary for older messages
        self.token_limit = token_limit

    def add_turn(self, user_message: str, assistant_response: str):
        """Add one interaction (user + assistant) to memory."""
        self.history.append({"role": "user", "content": user_message})
        self.history.append({"role": "assistant", "content": assistant_response})

    def get_recent_history(self) -> List[Dict[str, str]]:
        """Return recent exchanges."""
        return list(self.history)

    def get_combined_context(self) -> str:
        """
        Combines long-term summary and recent chat history into a readable text block.
        """
        formatted_history = messages_to_string(list(self.history))
        return (
            f"Conversation Summary:\n{self.summary or 'No summary yet.'}\n\n"
            f"Recent Conversation:\n{formatted_history}"
        )

    def update_summary(self, llm, summarization_prompt: str):
        """
        Summarizes the current history using an LLM, keeping total tokens within limit.

        Args:
            llm: LLM client instance.
            summarization_prompt (str): Template for summarization prompt.
                                         Must include `{chat}` placeholder.

        Returns:
            str | None: Generated summary text if performed, otherwise None.
        """
        # Combine messages into text
        text_to_summarize = messages_to_string(list(self.history))
        token_count = count_tokens(text_to_summarize, getattr(llm, "model", "gemini-1.5-flash"))

        if token_count > self.token_limit:
            # Trim oldest messages to fit within limit
            while token_count > self.token_limit and len(self.history) > 2:
                self.history.popleft()
                text_to_summarize = messages_to_string(list(self.history))
                token_count = count_tokens(text_to_summarize, getattr(llm, "model", "gemini-1.5-flash"))

        # If no history left, skip summarization
        if not self.history:
            return None

        # Request summary from LLM
        summary_response = llm.invoke([
            {"role": "system", "content": "You are a summarization assistant."},
            {"role": "user", "content": f"""Provide a concise summary of this conversation history: 
                {summarization_prompt.format(chat=text_to_summarize)}

            Focus on main topics and key information. Keep under 200 words."""
             }
        ])

        # Update long-term summary
        new_summary = summary_response.content.strip()
        self.summary += f"\n{new_summary}\n"
        self.history.clear()

        return new_summary

    def to_dict(self) -> Dict[str, Any]:
        """
        Returns the full memory state as a dictionary for debugging or persistence.
        Useful for logging intermediate states during conversation.

        Returns:
            dict: Current summary, chat history, and timestamp.
        """
        from datetime import datetime

        return {
            "summary": self.summary,
            "history": list(self.history),
            "timestamp": datetime.now().isoformat(),
        }

