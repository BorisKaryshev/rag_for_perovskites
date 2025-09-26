from langchain.prompts.chat import ChatPromptTemplate
from typing import Optional
import logging

import ollama

logger = logging.getLogger(__name__)


class OllamaWrapper:
    def __init__(self, model_name, model_host, prompt_template: Optional[str] = None):
        logger.info("Creating Ollama client")
        self.__model_name = model_name
        self.__client = ollama.Client(host=model_host)

        if prompt_template:
            self.__chat_prompt = ChatPromptTemplate.from_messages([
                ("system", prompt_template),
                ("human", "{text}"),
            ])
        else:
            self.__chat_prompt = ChatPromptTemplate.from_messages([
                ("human", "{text}"),
            ])

    def ask_question(self, question: str, context: str) -> str:
        prompt = self.__chat_prompt.format_messages(
            context = context,
            text = question
        )
        prompt = [{"role": "user" if msg.type == "human" else "assistant" if msg.type == "ai" else msg.type, "content": msg.content} for msg in prompt]
        logger.debug(f"Invoking chat model with prompt: {prompt}")
        ollama_response = self.__client.chat(model=self.__model_name, messages=prompt)
        logger.info(f"Got response from ollama: {ollama_response}")
        return ollama_response["message"]["content"]
