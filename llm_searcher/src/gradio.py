from .search import Searcher

import gradio as gr
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


HELP_MESSAGE = (
    "This chat bot supports commands:\n"
    "\thelp - view this message\n"
    "\tchange_config <config name> - change config for chat bot\n"
)


class SearcherForGradio:
    def __init__(self, config: dict, searcher_name: str):
        self.__config = config
        self.__searcher = None
        self.change_searcher(searcher_name)

    def change_searcher(self, searcher_name: str):
        try:
            searcher_config = self.__config[searcher_name]
        except Exception:
            logger.error(f"Could not load searcher config {searcher_name}")
            return

        self.__searcher =  Searcher(searcher_config)

    def ask_question(self, query: str) -> str:
        return self.__searcher.ask_question(query)

    def add_document(self, document_path: str) -> None:
        self.__searcher.add_document(Path(document_path))


class StopServerException(Exception):
    pass

class GradioLLMSearcher:
    def __init__(self, config: dict, searcher_name: str) -> None:
        self.__searcher = SearcherForGradio(config, searcher_name)

    def __call__(self, query: str, history) -> str:
        command = query.strip().split()[0].lower()

        if command == "change_config":
            searcher_name = query.strip().split()[1]
            return self.__searcher.change_searcher(searcher_name)
        if command == "help":
            return HELP_MESSAGE
        if command == "exit":
            raise StopServerException()

        answer = self.__searcher.ask_question(query)
        history.append((query, answer))
        return history

    def add_document(self, files):
        if isinstance(files, str):
            files = [files]
        for file in files:
            self.__searcher.add_document(file)

def gradio_main(config: dict, searcher_name: str, publish_link_to_web: bool = False):
    searcher = GradioLLMSearcher(config, searcher_name)
    with gr.Blocks() as demo:
        chatbot = gr.Chatbot(label='Виртуальный ассистент')
        msg = gr.Textbox(scale=1)
        with gr.Row():
            clear = gr.ClearButton([msg, chatbot], scale=1)
        with gr.Row():
            file = gr.File()
            upload = gr.UploadButton("Click to upload a document")
            upload.upload(lambda paths: searcher.add_document(paths), upload, file)
        msg.submit(searcher, [msg, chatbot], [chatbot])
        try:
            if publish_link_to_web:
                demo.launch(share=True)
            else:
                demo.launch(server_name="0.0.0.0", share=False)
        except StopServerException:
            pass
        except Exception as ex:
            logger.error(f"Exception occurred: {ex}")
