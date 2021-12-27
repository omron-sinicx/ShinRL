""" Functions for logging.
Author: Toshinori Kitamura
Affiliation: NAIST & OSX
"""
import logging

import structlog
from structlog.processors import *
from structlog.stdlib import *


def initialize_log_style() -> None:
    shinrl_logger = logging.getLogger()
    shinrl_logger.handlers = []

    stream_handler = logging.StreamHandler()
    stream_formatter = structlog.stdlib.ProcessorFormatter(
        processor=structlog.dev.ConsoleRenderer(
            level_styles={
                "info:": "\033[31m",
            }
        )
    )
    stream_handler.setFormatter(stream_formatter)
    stream_handler.setLevel(logging.INFO)
    shinrl_logger.addHandler(stream_handler)
    shinrl_logger.setLevel(logging.INFO)
    structlog.configure(
        processors=[
            TimeStamper(fmt="iso"),
            ExceptionPrettyPrinter(),
            structlog.stdlib.ProcessorFormatter.wrap_for_formatter,
        ],
        logger_factory=structlog.stdlib.LoggerFactory(),
    )


def add_logfile_handler(file_path: str) -> None:
    shinrl_logger = logging.getLogger()
    file_handler = logging.FileHandler(file_path)
    file_formatter = structlog.stdlib.ProcessorFormatter(
        processor=structlog.processors.JSONRenderer(),
    )
    file_handler.setFormatter(file_formatter)
    shinrl_logger.addHandler(file_handler)
