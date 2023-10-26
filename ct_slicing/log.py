import logging
import os

DEV_MODE = os.environ.get("DEV_MODE")

# set the logger for the package
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

if DEV_MODE:
    logger.setLevel(logging.DEBUG)
