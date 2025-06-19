import os
from typing import List, Dict

import openai
from dotenv import load_dotenv

load_dotenv()


class LLMClient:
    """Client for interacting with LLM"""

    def __init__(self):
        self.client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    def generate_response(
        self, system_prompt: str, messages: List[Dict], model: str, temperature: float, max_tokens: int
    ) -> str:
        """Generate a response using the OpenAI API"""
        try:
            # Prepare messages for the API
            api_messages = [{"role": "system", "content": system_prompt}]
            api_messages.extend(messages)

            response = self.client.chat.completions.create(
                model=model, messages=api_messages, max_tokens=max_tokens, temperature=temperature
            )

            return response.choices[0].message.content.strip()

        except openai.OpenAIError as e:
            return f"I apologize, but I'm having trouble generating a response right now. Error: {str(e)}"

        except Exception as e:
            return f"Internal error: {str(e)}"
