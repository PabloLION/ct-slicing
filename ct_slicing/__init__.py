import sys
import logging
import os

DEV_MODE = os.environ.get("DEV_MODE")


# change maximum recursion limit
sys.setrecursionlimit(100000)

# set the logger for the package
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

if DEV_MODE:
    logging.basicConfig(level=logging.DEBUG)
