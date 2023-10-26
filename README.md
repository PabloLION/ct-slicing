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

## Contribute

Run the code in dev mode.

```bash
export DEV_MODE=1  # On Unix-like systems
set DEV_MODE=1     # On Windows
```
