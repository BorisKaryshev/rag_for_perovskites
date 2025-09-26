from pandas import DataFrame, read_csv
from multiprocessing import Process, Queue, cpu_count
from time import time
import os
import re
import logging
from pathlib import Path
from typing import List

from .PdfReader_impl import read_pdf


logger = logging.getLogger(__name__)


class TextReplacer:
    def __init__(self, pattern: str, replace_on: str):
        self.__pattern = pattern
        self.__replace_on = replace_on

    def __call__(self, text: str, flag = 0) -> str:
        return re.sub(
            self.__pattern,
            self.__replace_on,
            text,
            flags=flag
        )


def clean_text(text: str) -> str:
    regexes = (
        TextReplacer(r"[\n]+", r"\n"),
        TextReplacer(r"[ |]+", r" "),
        TextReplacer(r"\n+\s+", r"\n"),
        TextReplacer(r"\s+\n+", r"\n"),
        TextReplacer(r"None", r""),
    )

    for replace in regexes:
        text = replace(text)

    return text

def read_file(path: Path):
    extension = path.suffix
    if extension == ".txt":
        with open(str(path), 'r') as file:
            return file.read()
    if extension == ".pdf":
        return read_pdf(str(path.resolve()))



def process_worker(
    filename_queue: Queue,
    folder_name: str,
    result_queue: Queue,
) -> None:
    for file in iter(filename_queue.get, None):
        begin = time()
        text = read_file(Path(folder_name) / Path(file))
        end = time()
        result_queue.put({
            "name": str(file),
            "text": clean_text(text),
            "embedding" : None
        })

        logger.info(
            f'Finished reading "{file}" it took {end - begin}'
        )
    result_queue.put(None)


def load_pdfs(folder_with_pdf: str, database: DataFrame | str, num_of_jobs: int = cpu_count()) -> DataFrame:
    data = database if isinstance(database, DataFrame) else read_csv(database)

    directory = os.fsencode(folder_with_pdf)

    filename_queue = Queue()
    result_queue = Queue()

    for file in os.listdir(directory):
        filename = os.fsdecode(file)
        if data.empty or filename not in data['name'].unique():
            logger.info(f'Reading file {filename}')
            filename_queue.put(filename)

    for _ in range(num_of_jobs):
        filename_queue.put(None)

    jobs = []
    for _ in range(num_of_jobs - 1):
        p = Process(
                target=process_worker,
                args=(filename_queue, folder_with_pdf, result_queue)
            )
        p.start()
        jobs.append(p)

    process_worker(filename_queue, folder_with_pdf, result_queue)

    result_data = {
        "name": [],
        "text": [],
        "embedding" : []
    }
    item = result_queue.get()
    num_of_nones = 1 if item is None else 0

    while num_of_nones < num_of_jobs:
        if item is not None:
            for key, value in item.items():
                result_data[key].append(value)
        item = result_queue.get()
        if item is None:
            num_of_nones += 1

    for job in jobs:
        job.join()

    if data.empty:
        data = DataFrame(result_data)
    else:
        for (name, text) in zip(result_data["name"], result_data["text"]):
            data.loc[len(data)] = [name, text, None]

    if isinstance(database, str):
        data.to_csv(database, index=False)
    return data
