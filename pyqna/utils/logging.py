"""
All logging related helper functions.
"""

import logging
from typing import Callable


def create_logger(name: str) -> logging.Logger:
    """
    Creates a logger with the given name and level.
    """
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    return logger


def raise_if(
    condition: bool,
    message: str,
    exception_cls: Callable[[Exception], None] = ValueError,
    logger: logging.Logger = create_logger("base_logger"),
) -> None:
    """
    Raises an exception if the given condition is true.
    """
    if condition:
        logger.error(message)
        raise exception_cls(message)


def raise_if_not(
    condition: bool,
    message: str,
    exception_cls: Exception = ValueError,
    logger: logging.Logger = create_logger("base_logger"),
) -> None:
    """
    Raises an exception if the given condition is false.
    """
    raise_if(not condition, message, exception_cls, logger)
