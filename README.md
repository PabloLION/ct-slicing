Source code for CT slicing volume visualization

Computer Vision Center
Universitat Autonoma de Barcelona
Bellaterra, Barcelona, Spain

## How to use

I changed the packager to [poetry](https://python-poetry.org/), [install](https://python-poetry.org/docs/#installation) it if needed (also possible to install with `brew`, `pip`). So now you can install the dependencies with:

```bash
poetry install
```

To activate the virtual environment created by poetry:

```bash
poetry shell
```

### Known Installation Issues

Poetry cannot install `PyRadiomics = "^3.0.1"`, use pip instead:

```bash
# PyRadiomics does not support PEP-517 https://github.com/AIM-Harvard/pyradiomics/issues/787
pip install PyRadiomics==3.0.1
```

### Backup of old version with pip

Set up and activate virtual environment (Optional)

```bash
python -m venv .venv
source .venv/bin/activate
```

Install dependencies

```bash
pip install -r requirements.txt
```

Run the script

```bash
python __main__.py
```

## File Naming Convention

- `__main__.py`: Main entry point of the program
- `__init__.py`: Python package initializer
- `*.py`: Python modules
- `lX_Y_file_name.py`: Python modules for session X, part Y. Both X and Y are 1-indexed order in the plan (not actually fulfilled).

## Contribute

### Development Mode

Run the code in dev mode.

```bash
# On Unix-like systems, including macOS and Linux
export DEV_MODE=1
# On Windows
set DEV_MODE=1
```

### Known issues

- [ ] Universal name across all codes. IMG_PATH, MASK_PATH in lower case; img for ITK image, img_np for numpy array, etc.
