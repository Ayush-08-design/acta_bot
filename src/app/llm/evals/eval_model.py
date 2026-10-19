import openai
from deepeval.models.base_model import DeepEvalBaseLLM


class AsyncLLMJudge(DeepEvalBaseLLM):
    def __init__(self, model_name: str, base_url: str, api_key: str = "ollama"):
        self.model_name = model_name
        self.client = openai.AsyncOpenAI(base_url=base_url, api_key=api_key)

    def load_model(self):
        return self.client

    def generate(self, prompt: str) -> str:
        sync_client = openai.OpenAI(base_url=self.client.base_url, api_key=self.client.api_key)
        response = sync_client.chat.completions.create(
            model=self.model_name,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0
        )
        return response.choices.message.content

    async def a_generate(self, prompt: str) -> str:
        client = self.load_model()
        response = await client.chat.completions.create(
            model=self.model_name,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0
        )
        return response.choices.message.content

    def get_model_name(self) -> str:
        return self.model_name


async_judge = AsyncLLMJudge(
    model_name="qwen2.5:0.5b",
    base_url="http://localhost:1234/v1"
)

