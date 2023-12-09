import os
from rich.logging import RichHandler
import logging

logging.basicConfig(
    level="INFO",
    format="%(message)s",
    datefmt="-",
    # date takes indent datefmt="[%X]",
    handlers=[RichHandler()],
)

logger = logging.getLogger("rich")


# change logger level based on environment variable
DEV_MODE = os.environ.get("DEV_MODE")
if DEV_MODE:
    logger.setLevel(logging.DEBUG)

if __name__ == "__main__":
    logger.info("This is an info message")
    logger.warning("This is a warning message")
    logger.error("This is an error message")
    logger.critical("This is a critical message")
    logger.debug("This is a debug message")  # only shown in dev mode
    logger.setLevel("DEBUG")  # to show debug message
    logger.debug("This is a debug message")  # shown
