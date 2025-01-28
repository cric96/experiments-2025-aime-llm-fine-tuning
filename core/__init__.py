from abc import ABC
from google import genai
from google import generativeai as old_genai
import ollama
class LangaugeModelService(ABC):
    def embed_content(self, model: str, content: list[str]) -> list[list[float]]:
        pass

    def generate_content(self, model: str, content: str) -> str:
        pass

    def generate_contents(self, model: str, contents: list[str]) -> list[str]:
        return [self.generate_content(model, content) for content in contents]

class OllamaService(LangaugeModelService):

    def embed_contents(self, model: str, content: list[str]):
        return ollama.embed(
            model = model,
            input=content,
        ).embeddings

    def generate_content(self, model: str, content: str):
        return ollama.generate(
            model = model,
            prompt = content,
        ).response
MAX_BATCH = 100
class GeminiService(LangaugeModelService):
    def __init__(self, api_key):
        self.client = genai.Client(api_key=api_key)

    def embed_contents(self, model: str, content: list[str]):
        embeddings = []
        for i in range(0, len(content), MAX_BATCH):
            start_index = i
            end_index = min(i + MAX_BATCH, len(content))
            print(f"Processing batch {i // MAX_BATCH + 1} of {len(content) // MAX_BATCH + 1}, start index {start_index}, end index {end_index}")
            batch_embedding = self.client.models.embed_content(
                model=model,
                contents=content[start_index:end_index],
            )
            embeddings.extend(batch_embedding.embeddings)
        return [embedding.values for embedding in embeddings]

    def generate_content(self, model: str, content: str):
        return self.client.models.generate_content(
            model=model,
            contents=content,
        ).text

class GeminiOldService(LangaugeModelService):
    def __init__(self, api_key):
        old_genai.configure(api_key=api_key)


    def generate_content(self, model: str, content: str) -> str:
        # Create the model
        generation_config = {
            "temperature": 1,
            "top_p": 0.95,
            "top_k": 40,
            "max_output_tokens": 8192,
            "response_mime_type": "text/plain",
        }

        model = old_genai.GenerativeModel(
            model_name="tunedModels/generated-b4ba5xhxlhk0",
            generation_config=generation_config,
        )

        chat_session = model.start_chat(
            history=[]
        )

        response = chat_session.send_message(content)

        return response.text