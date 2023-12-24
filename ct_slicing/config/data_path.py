from pathlib import Path

REPO_ROOT = Path(__file__).parent.parent.parent
DATA_FOLDER = REPO_ROOT / "data"
OUTPUT_FOLDER = REPO_ROOT / "output"
META_DATA_PICKLE = REPO_ROOT / "data" / "metadata.pickle"
NODULE_METADATA_PICKLE = REPO_ROOT / "data" / "nodule_metadata.pickle"
