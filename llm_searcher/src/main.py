from .search import Searcher

from typing import Callable
import sys
import time
import json
import logging

logger = logging.getLogger(__name__)


def ask_question_from_file(
    searcher: Searcher, path: str, outstream: Callable[[str], None] = sys.stdout.write
):
    questions = None
    try:
        with open(path, "r") as f:
            questions = json.load(f)["questions"]
    except Exception as ex:
        logger.error(f"Failed while loading questions. Reading from stdin: {ex}. ")
    else:
        for question in questions:
            begining = time.time()
            res = searcher.ask_question(question)
            end = time.time()
            d = json.dumps(
                {
                    "question": question,
                    "answer": res,
                    "time_elapsed_s": (end - begining),
                }
            )
            logger.info(f"Got answer to file: {d}")
            outstream(d)


def main(config: dict, searcher_name: str) -> None:

    searcher_config = config.get(searcher_name)
    if not searcher_config:
        logger.error("Could not load searcher config")

    searcher = Searcher(searcher_config)

    path_to_questions = searcher_config.get("questions")
    if path_to_questions:
        ask_question_from_file(searcher, path_to_questions)

    question = input("Your questions:\n")
    while question.upper() != "EXIT":
        res = searcher.ask_question(question)
        print(res)
        question = input()
