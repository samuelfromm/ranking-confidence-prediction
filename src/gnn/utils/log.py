import logging
import os
import re
import multiprocessing


def init_logger(log_file=None, log_tag=""):
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    log_format = logging.Formatter(f"[%(asctime)s {log_tag}] %(message)s")
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(log_format)
    logger.addHandler(console_handler)

    if log_file:
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(log_format)
        logger.addHandler(file_handler)

    return logger


def close_logger(logger):
    handlers = logger.handlers[:]
    for handler in handlers:
        handler.close()
        logger.removeHandler(handler)
