# from langchain_community.embeddings import GPT4AllEmbeddings, HuggingFaceEmbeddings
import ollama
# from sentence_transformers import SentenceTransformer
from abc import ABC, abstractmethod
from typing import Optional
import logging


logger = logging.getLogger(__name__)


class Embedder(ABC):
    @abstractmethod
    def embed_query(self, query: str) -> list:
        pass

class OllamaEmbedder(Embedder):
    def __init__(self, hostname, model_name):
        self.__client = ollama.Client(host=hostname)
        self.__model_name = model_name

    def embed_query(self, query: str) -> list:
        response = self.__client.embed(model=self.__model_name, input=query)
        logger.debug(f"Got embedings response from ollama: {response}")
        return response["embeddings"][0]

# class GPT4AllEmbedder(Embedder):
#     def __init__(self):
#         self.__embedder = GPT4AllEmbeddings()
#
#     def embed_query(self, query: str) -> list:
#         return self.__embedder.embed_query(query)
#
#
# class E5Embedder(Embedder):
#     def __init__(self):
#         self.__embedder = SentenceTransformer('intfloat/e5-large-v2')
#
#     def embed_query(self, query: str) -> list:
#         return self.__embedder.encode([query], show_progress_bar=False)[0]
#
#
# class HuggingFaceEmbedder(Embedder):
#     def __init__(self):
#         self.__embedder = HuggingFaceEmbeddings()
#
#     def embed_query(self, query: str) -> list:
#         return self.__embedder.embed_query(query)

def get_embedder(embedder_name: Optional[str] = None, config: Optional[dict] = None) -> Embedder:
    embedders = {
        # "gpt4all" : GPT4AllEmbedder,
        # "e5" : E5Embedder,
        # "hugging_face": HuggingFaceEmbedder,
    }
    try:
        if embedder_name in ("ollama", None) and config:
            return OllamaEmbedder(hostname=config["embedder_hostname"], model_name=config["embedder_model_name"])
        embedder = embedders.get(embedder_name, lambda: None)()

        if embedder is None:
            raise RuntimeError(f"Failed to create embedder, name not found {embedder_name}")
    except Exception as ex:
        logger.error(f"Failed to created embedder {embedder_name}: {ex}")
        raise

    logger.info(f"Created embedder {embedder_name}")
    return embedder
