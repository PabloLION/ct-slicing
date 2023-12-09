from pathlib import Path

from ct_slicing.ct_logger import logger


REPO_ROOT = Path(__file__).parent.parent.parent
DATA_FOLDER = REPO_ROOT / "data"
OUTPUT_FOLDER = REPO_ROOT / "output"

logger.debug(f"DATA_FOLDER: {DATA_FOLDER}")
