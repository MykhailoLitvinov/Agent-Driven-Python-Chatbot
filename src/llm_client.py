import os
from typing import List, Dict

import openai
from openai.types.chat import ChatCompletionMessageParam
from dotenv import load_dotenv

load_dotenv()


class LLMClient:
    """Client for interacting with an OpenAI-compatible LLM"""

    def __init__(self):
        self.client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    def generate_response(
        self,
        system_prompt: str,
        messages: List[Dict[str, str]],
        model: str,
        temperature: float,
        max_tokens: int,
        stream=True,
    ) -> str:
        """Generate a streamed response using the OpenAI API"""

        try:
            # Prepare the full message list
            full_messages: List[ChatCompletionMessageParam] = [{"role": "system", "content": system_prompt}]
            full_messages.extend(messages)

            # Initiate streaming request
            response = self.client.chat.completions.create(
                model=model,
                messages=full_messages,
                max_completion_tokens=max_tokens,
                temperature=temperature,
                stream=stream,
            )

            if not stream:
                return response.choices[0].message.content.strip()

            response_text = ""
            printed_first_token = False

            for chunk in response:
                token = chunk.choices[0].delta.content
                if token:
                    response_text += token
                    if not printed_first_token:
                        token = f"🤖 {token}"
                        printed_first_token = True
                    print(token, end="", flush=True)

            print()  # Newline after full response
            return response_text

        except openai.OpenAIError as e:
            return f"I apologize, but I'm having trouble generating a response right now. Error: {str(e)}"

        except Exception as e:
            return f"Internal error: {str(e)}"
