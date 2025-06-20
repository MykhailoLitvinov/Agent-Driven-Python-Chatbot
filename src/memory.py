import os
from typing import List, Dict, Any

from src.summarizer import Summarizer

MAX_RAW_MESSAGES = os.getenv("MAX_MEMORY_MESSAGES", 10)


class ConversationMemory:
    """Manage conversation memory using the Summary Buffer approach"""

    def __init__(self, summarizer: Summarizer, max_raw_messages: int = MAX_RAW_MESSAGES):
        self.max_raw_messages = max_raw_messages
        self.messages: List[Dict[str, str]] = []
        self.summary: str = ""
        self.summarizer = summarizer

    def add_message(self, role: str, content: str):
        """Add a message to the memory"""
        message = {"role": role, "content": content}
        self.messages.append(message)

        # If message limit exceeded – summarize older ones
        if len(self.messages) > self.max_raw_messages:
            # Identify messages to summarize
            messages_to_summarize = self.messages[: -self.max_raw_messages // 2]

            # Update summary using LLM
            self.summary = self.summarizer.update_summary(self.summary, messages_to_summarize)

            # Keep only the most recent messages
            self.messages = self.messages[-self.max_raw_messages // 2 :]

    def get_context(self) -> Dict[str, Any]:
        """Get context for the LLM"""
        formatted_messages = self.messages[-self.max_raw_messages :]
        return {
            "messages": formatted_messages,
            "summary": self.summary,
        }

    def reset(self):
        """Reset the memory"""
        self.messages = []
        self.summary = ""
