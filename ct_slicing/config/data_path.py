from pathlib import Path

REPO_ROOT = Path(__file__).parent.parent.parent
# data
DATA_FOLDER = REPO_ROOT / "data"
METADATA_PATH = DATA_FOLDER / "MetadatabyNoduleMaxVoting.xlsx"
CT_FOLDER = DATA_FOLDER / "CT"
VOI_FOLDER = DATA_FOLDER / "VOIs"

# config
RADIOMICS_DEFAULT_PARAMS_PATH = REPO_ROOT / "ct_slicing" / "config" / "Params.yaml"
RADIOMICS_CUSTOM_PARAMS_PATH = (
    REPO_ROOT / "ct_slicing" / "config" / "FeaturesExtraction_Params.yaml"
)

# intermediate data
NODULE_ID_PICKLE = REPO_ROOT / "output" / "intermediate-results" / "nodule_id.pickle"
METADATA_JSON_GZIP = REPO_ROOT / "output" / "intermediate-results" / "metadata.json.gz"


def extracted_features_npy_path_with_threshold(threshold) -> Path:
    return REPO_ROOT / "output" / f"extracted_features_{threshold}.npy"


# output
OUTPUT_FOLDER = REPO_ROOT / "output"
DEFAULT_EXPORT_XLSX_PATH = OUTPUT_FOLDER / "features.xlsx"
