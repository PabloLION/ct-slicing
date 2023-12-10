"""
The result shows that no VOI file is same as the corresponding CT file.
So we cannot remove the CT folder.
"""
from pathlib import Path
from ct_slicing.config.data_path import DATA_FOLDER
import filecmp

CT_FOLDER = DATA_FOLDER / "CT"
VOI_FOLDER = DATA_FOLDER / "VOIs"


def iter_files(base_path: Path):
    """Iterate over all files in the given directory recursively.
    Return the relative path to each file.
    """
    for file_path in base_path.glob("**/*"):
        if file_path.is_file():
            yield file_path.relative_to(base_path)


if __name__ == "__main__":
    for file_rel_path in iter_files(CT_FOLDER):
        if (file_rel_path.name) == ".DS_Store":
            continue
        corresponding_voi = VOI_FOLDER / file_rel_path
        if not corresponding_voi.exists():
            print(f"VOI file not found for {file_rel_path}")
            continue
        if not filecmp.cmp(CT_FOLDER / file_rel_path, corresponding_voi):
            print(f"VOI file not equal for {file_rel_path}")
            continue
        print(f"VOI file equal for {file_rel_path}")
