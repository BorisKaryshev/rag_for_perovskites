from langchain.prompts.chat import ChatPromptTemplate
from typing import Optional
import logging


logger = logging.getLogger(__name__)


class ChatWrapper:
    def __init__(self, chat_model, prompt_template: Optional[str] = None):
        logger.info("Creating ModelWrapper")
        self.__chat_model = chat_model
        
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
        logger.debug(f"Invoking chat model with prompt: {prompt}")
        return self.__chat_model.invoke(prompt).content
