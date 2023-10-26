from pathlib import Path
from log import logger


DATA_FOLDER = Path(__file__).parent.parent.parent / "data"

logger.debug(f"DATA_FOLDER: {DATA_FOLDER}")
