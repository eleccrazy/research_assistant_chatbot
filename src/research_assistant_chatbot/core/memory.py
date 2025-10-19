"""
File: memory.py
Description: Handles conversation history, summarization, and context persistence for the chatbot.
Author: Gizachew Kassa
Date Created: 19/10/2025
"""
from collections import deque
from typing import List, Dict, Any

class MemoryManager:
    """
    Manages short-term chat history and long-term summarized memory for context retention.
    """
    def __init__(self, window_size: int = 4):
        self.history = deque(maxlen=window_size * 2)  # store user+assistant turns
        self.summary = ""  # rolling summary for older messages

    def add_turn(self, user_message: str, assistant_response: str):
        """Add one interaction (user + assistant) to memory."""
        self.history.append({"role": "user", "content": user_message})
        self.history.append({"role": "assistant", "content": assistant_response})

    def get_recent_history(self) -> List[Dict[str, str]]:
        """Return recent exchanges."""
        return list(self.history)

    def get_combined_context(self) -> str:
        """Combine summary and recent exchanges into a text context."""
        formatted_history = "\n".join(
            f"{m['role'].capitalize()}: {m['content']}" for m in self.history
        )
        return f"Summary:\n{self.summary or 'None'}\n\nRecent exchanges:\n{formatted_history}"

    def update_summary(self, llm, summarization_prompt: str):
        """Summarize old messages using the LLM and store as long-term memory."""
        if len(self.history) < self.history.maxlen:
            return  # not enough history yet to summarize

        # Combine old messages
        text_to_summarize = "\n".join(
            f"{m['role']}: {m['content']}" for m in self.history
        )

        summary = llm.invoke([
            {"role": "system", "content": "You are a summarization assistant."},
            {"role": "user", "content": summarization_prompt.format(chat=text_to_summarize)},
        ])

        self.summary += f"\n{summary.content}"
        self.history.clear()
