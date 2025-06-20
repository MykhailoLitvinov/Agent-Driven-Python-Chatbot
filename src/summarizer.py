import json
from typing import List, Dict
from src.llm_client import LLMClient


class Summarizer:
    def __init__(self, llm_client: LLMClient):
        self.llm_client = llm_client
        self.model = "gpt-4o-mini"
        self.temperature = 0.3
        self.max_tokens = 300
        self.stream = False
        self.response_format = {
            "type": "json_schema",
            "json_schema": {
                "name": "summary_response",
                "schema": {
                    "type": "object",
                    "properties": {
                        "summary": {"type": "string"},
                    },
                    "required": ["summary"],
                    "additionalProperties": False,
                },
            },
        }

    def update_summary(self, previous_summary: str, messages: List[Dict[str, str]]) -> str:
        prompt = self._build_prompt(previous_summary, messages)
        generated_response = self.llm_client.generate_response(
            system_prompt=prompt,
            messages=[],
            model=self.model,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            stream=self.stream,
            response_format=self.response_format,
        )
        generated_response = json.loads(generated_response)

        return generated_response["summary"]

    @staticmethod
    def _build_prompt(summary: str, messages: List[Dict[str, str]]) -> str:
        latest_messages = "\n".join(f"{msg['role']}: {msg['content']}" for msg in messages)

        prompt = f"""You are expert in summarizing conversation. 
            Your goal is to provide compact and clear summary based on latest messages and previous summary.
        
            Previous summary:
            {summary or 'None'}
        
            Latest messages:
            {latest_messages}
        
            Provide an updated summary below:"""

        return prompt.strip()
