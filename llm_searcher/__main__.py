from src.main import main
from src.gradio import gradio_main
from src.logger import setup_default_logger, setup_logger
import json
import logging

import argparse

import sys

logger = logging.getLogger(__name__)

DEFAULT_PATH_TO_CONFIG = "./datasheets.json"


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config"
                        , default=DEFAULT_PATH_TO_CONFIG
                        , help="Path to config")
    parser.add_argument("--mode", choices=["stdio", "gradio"]
                        , default="stdio"
                        , help="Working mode")
    parser.add_argument("--searcher", required=True, help="Which config to use")
    parser.add_argument("--publish_to_web", action='store_true')
    return parser.parse_args()


def remove_stream_log_handlers():
    for handler in logger.handlers:
        if isinstance(handler, logging.StreamHandler):
            logger.removeHandler(handler)


if __name__ == "__main__":
    setup_default_logger()
    args = parse_arguments()

    configs = None

    try:
        with open(args.config, 'r', encoding='utf8') as config_file:
            configs = json.load(config_file)
    except Exception as ex:
        logger.error(f"Failed to read configs: {ex}")
        exit(1)
    setup_logger(configs.get("logging"))

    try:
        if args.mode == "gradio":
            gradio_main(configs, args.searcher, publish_link_to_web=args.publish_to_web)
        else:
            remove_stream_log_handlers()
            main(configs, args.searcher)
    except Exception as ex:
        logger.error(ex)
        raise
