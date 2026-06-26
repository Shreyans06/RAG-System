from collections.abc import Generator

import anthropic
from anthropic.types import MessageParam, TextBlock

class LLMClient:
    """
    A client for interacting with the Anthropic API.
    """
    def __init__(
            self,
            api_key: str,
            model: str = "claude-sonnet-4-5-20251001",
            max_tokens: int = 2048,
            temperature: float = 0.0,
    ) -> None:
        self.api_key = api_key
        self.model = model
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.client = anthropic.Anthropic(api_key=api_key)
    
    def generate(
            self,
            system: str,
            messages: list[MessageParam]
    ) -> str:
        """
        Generate a response from the LLM based on the provided system prompt and messages.
        """
        response = self.client.messages.create(
            model = self.model,
            max_tokens = self.max_tokens,
            temperature = self.temperature,
            system = system,
            messages = messages,
        )
        text_block = next(b for b in response.content if isinstance(b, TextBlock))
        return text_block.text.strip()
    
    def stream(
            self,
            system: str,
            messages: list[MessageParam]
    ) -> Generator[str , None , None]:
        """
        Stream a response from the LLM based on the provided system prompt and messages.
        Yields chunks of text as they are generated.
        """
        with self.client.messages.stream(
            model = self.model,
            max_tokens = self.max_tokens,
            temperature = self.temperature,
            system = system,
            messages = messages,
        ) as stream:
            yield from stream.text_stream
       