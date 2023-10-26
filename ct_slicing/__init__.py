import sys
import logging

# change maximum recursion limit
sys.setrecursionlimit(100000)
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
