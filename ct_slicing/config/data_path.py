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

# intermediate results (git-ignored)
OUTPUT_FOLDER = REPO_ROOT / "output"

NODULE_ID_PICKLE = OUTPUT_FOLDER / "intermediate-results" / "nodule_id.pkl"
METADATA_JSON_GZIP = OUTPUT_FOLDER / "intermediate-results" / "metadata.json.gz"
UNCALIBRATED_CLASSIFIER_JOBLIB = (
    OUTPUT_FOLDER / "intermediate-results" / "uncalibrated_classifier.joblib"
)
CALIBRATED_CLASSIFIER_JOBLIB = (
    OUTPUT_FOLDER / "intermediate-results" / "calibrated_classifier.joblib"
)


def extracted_features_npy_path_with_threshold(threshold) -> Path:
    return (
        REPO_ROOT
        / "output"
        / "intermediate-results"
        / f"extracted_features_{threshold}.npy"
    )


MODEL_PATH = OUTPUT_FOLDER / "trained-model" / "resnet152_model.pth"
# #TODO: remove MODEL_PATH
MODEL_OPTIMIZER_PATH = OUTPUT_FOLDER / "trained-model" / "resnet152_model_optimizer.pth"
SLICE_IMAGE_FOLDER = OUTPUT_FOLDER / "png-slice-images"

# output (git-ignored)
DEFAULT_EXPORT_XLSX_PATH = OUTPUT_FOLDER / "features.xlsx"

for folder in [OUTPUT_FOLDER, SLICE_IMAGE_FOLDER]:
    if not folder.exists():
        folder.mkdir()
