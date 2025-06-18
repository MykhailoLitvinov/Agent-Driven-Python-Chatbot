from datetime import datetime
from typing import List, Dict, Any


class ConversationMemory:
    """Manage conversation memory using the Summary Buffer approach"""

    def __init__(self, max_raw_messages: int = 10):
        self.max_raw_messages = max_raw_messages
        self.messages: List[Dict] = []
        self.summary: str = ""  # TODO Add logic for summarizing
        self.conversation_id = f"conv_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    def add_message(self, role: str, content: str, agent: str = None):
        """Add a message to the memory"""
        message = {"role": role, "content": content, "timestamp": datetime.now().isoformat(), "agent": agent}

        self.messages.append(message)

        # If the limit is exceeded - trim the messages (simplified version without summary)
        if len(self.messages) > self.max_raw_messages:
            # Keep only the latest messages
            self.messages = self.messages[-self.max_raw_messages // 2 :]

    def get_context(self) -> Dict[str, Any]:
        """Get context for the LLM"""
        # Format messages for the LLM
        formatted_messages = []
        for msg in self.messages[-8:]:  # Last 8 messages
            formatted_messages.append({"role": msg["role"], "content": msg["content"]})

        return {
            "messages": formatted_messages,
            "summary": self.summary,
            "conversation_id": self.conversation_id,
            "message_count": len(self.messages),
        }

    def reset(self):
        """Reset the memory"""
        self.messages = []
        self.summary = ""
        self.conversation_id = f"conv_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
