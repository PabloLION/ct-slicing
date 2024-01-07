import os
from rich.logging import RichHandler
import logging

from ct_slicing.config.dev_config import DEV_MODE

logging.basicConfig(
    level="INFO",
    format="%(message)s",
    datefmt="-",
    # date takes indent datefmt="[%X]",
    handlers=[RichHandler()],
)

logger = logging.getLogger("rich")


# change logger level based on environment variable
if DEV_MODE:
    logger.setLevel(logging.DEBUG)
    logger.critical("Running in dev mode")

if __name__ == "__main__":
    logger.info("This is an info message")
    logger.warning("This is a warning message")
    logger.error("This is an error message")
    logger.critical("This is a critical message")
    logger.debug("This is a debug message")  # only shown in dev mode
    logger.setLevel("DEBUG")  # to show debug message
    logger.debug("This is a debug message")  # shown
