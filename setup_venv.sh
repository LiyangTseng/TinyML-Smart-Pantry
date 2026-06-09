#!/usr/bin/env bash
set -euo pipefail

# Create and activate a local virtual environment in .venv and install requirements.
# Usage: ./setup_venv.sh

HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
VENV_DIR="$HERE/.venv"

if [ -d "$VENV_DIR" ]; then
  echo "Using existing virtualenv at $VENV_DIR"
else
  echo "Creating virtualenv at $VENV_DIR"
  python3 -m venv "$VENV_DIR"
fi

echo "Upgrading pip and installing dependencies..."
"$VENV_DIR/bin/python" -m pip install --upgrade pip setuptools wheel
"$VENV_DIR/bin/python" -m pip install -r "$HERE/requirements.txt"

echo "Virtualenv ready. To activate it run:"
echo "  source $VENV_DIR/bin/activate"
#!/usr/bin/env bash
# Create and populate a Python virtual environment for this project
set -euo pipefail

HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
VENV_DIR="$HERE/.venv"

if [ -d "$VENV_DIR" ]; then
  echo "Using existing virtualenv at $VENV_DIR"
else
  echo "Creating virtualenv at $VENV_DIR"
  python3 -m venv "$VENV_DIR"
fi

echo "Upgrading pip inside virtualenv..."
"$VENV_DIR/bin/python" -m pip install --upgrade pip setuptools wheel

echo "Installing project requirements into virtualenv..."
"$VENV_DIR/bin/python" -m pip install -r "$HERE/requirements.txt"

cat <<'USAGE'
Virtual environment created.

To activate locally:
  source .venv/bin/activate

Then run commands normally, e.g.:
  python data/download_food101.py \
  --output-dir artifacts/food101_full \
  --label-map data/label_map.json \
  --max-per-class 1000

To remove the virtualenv:
  rm -rf .venv
USAGE
