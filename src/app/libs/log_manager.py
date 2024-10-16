import logging
import socket
import sys
from logging.handlers import TimedRotatingFileHandler

FORMATTER = logging.Formatter(
    "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
LOG_FILE = "access.log"

def get_console_handler():
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(FORMATTER)
    return console_handler


def get_file_handler():
    file_handler = TimedRotatingFileHandler(LOG_FILE, when="midnight")
    file_handler.setFormatter(FORMATTER)
    return file_handler


def get_full_logger_name(logger_name):
    return socket.gethostname() + "-" + logger_name


def get_logger(logger_name):
    logger: logging.Logger = logging.getLogger(
        get_full_logger_name(logger_name)
    )

    logger.setLevel(
        logging.DEBUG
    ) 

    logger.addHandler(get_console_handler())
    logger.addHandler(get_file_handler())

    # with this pattern, it's rarely necessary to propagate the error up to parent
    logger.propagate = False

    return logger


