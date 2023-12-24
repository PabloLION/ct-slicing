from pathlib import Path

REPO_ROOT = Path(__file__).parent.parent.parent
DATA_FOLDER = REPO_ROOT / "data"
OUTPUT_FOLDER = REPO_ROOT / "output"
META_DATA_PICKLE = REPO_ROOT / "data" / "metadata.pickle"
NODULE_METADATA_PICKLE = REPO_ROOT / "data" / "nodule_metadata.pickle"
META_DATA_PATH = DATA_FOLDER / "MetadatabyNoduleMaxVoting.xlsx"
DEFAULT_EXPORT_XLSX_PATH = OUTPUT_FOLDER / "features.xlsx"
RADIOMICS_DEFAULT_PARAMS_PATH = REPO_ROOT / "ct_slicing" / "config" / "Params.yaml"
RADIOMICS_CUSTOM_PARAMS_PATH = REPO_ROOT / "ct_slicing" / "config" / "Params.yaml"
# intermediate data
NODULE_ID_PICKLE = REPO_ROOT / "ct_slicing" / "data_util" / "nodule_id.pkl"
