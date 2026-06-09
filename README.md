# TinyML Smart Pantry & Waste Reducer

This folder contains the implementation scaffold for the TinyML final project.

## What is here

- `data/` contains dataset assembly helpers and label mapping.
- `models/` contains training and TFLite conversion scripts.
- `src/host/` contains the host-side receiver and USDA shelf-life lookup.
- `src/mcu/` contains the Arduino / MCU integration scaffold.
- `tools/` contains an evaluation harness for in-situ predictions.

## Quick start
Create a local virtual environment so collaborators use consistent deps:

```bash
cd /Users/li-yangtseng/Codes/uw/tinyml/project
./setup_venv.sh
source .venv/bin/activate
```

After activation you can run the tools without installing system-wide packages.

## Where to run things

- Dataset download and manifest creation are documented in [data/README.md](data/README.md).
- Model training and conversion use the manifest produced by the data step. The default recipe in `models/train.py` now uses pretrained MobileNetV2 and a fine-tuning stage.
- MCU integration notes live in [src/mcu/README.md](src/mcu/README.md).

### Model training
Set up environment.
```bash
source .venv/bin/activate
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export TF_NUM_INTRAOP_THREADS=1
export TF_NUM_INTEROP_THREADS=1
export TF_CPP_MIN_LOG_LEVEL=2
```
Then run training with the manifest created from the data assembly step. Adjust hyperparameters as needed.
```bash
python models/train.py \
  --manifest artifacts/manifest.jsonl \
  --output-dir artifacts/model_pretrained/debug_run \
  --architecture mobilenetv2 \
  --epochs 6 \
  --fine-tune-epochs 4 \
  --batch-size 16 \
  --image-size 96 \
  --validation-split 0.2
```

### Model quantization and TFLite conversion
```bash
python models/convert_to_tflite.py \
  --model-path artifacts/model_pretrained/debug_run/best_model.keras \
  --representative-manifest artifacts/manifest.jsonl \
  --representative-samples 240 \
  --output-path artifacts/model/model.tflite \
  --output-cpp artifacts/model/model.cc
```

Why this matters:
- Using too few representative samples can cause large quantization accuracy drops.
- The manifest-based path above produces a larger, balanced calibration set and is recommended for reproducible results.

### QAT (Quantization-Aware Training)
If full-int8 PTQ drops accuracy too much, run QAT from your best float checkpoint.

Install dependencies (once):
```bash
pip install -r requirements.txt
```

Run the recommended QAT-friendly higher-accuracy pass (separable CNN path):
```bash
python models/train.py \
  --manifest artifacts/manifest.jsonl \
  --output-dir artifacts/model_sepqat2 \
  --architecture separable_cnn \
  --epochs 12 \
  --fine-tune-epochs 0 \
  --qat \
  --qat-epochs 6 \
  --qat-learning-rate 1e-5 \
  --batch-size 16 \
  --image-size 96 \
  --validation-split 0.2 \
  --seed 42
```

QAT training now emits an int8 TFLite artifact directly at `artifacts/model_sepqat2/qat_model.tflite` (or the matching output directory you pass to `--output-dir`).

If you want to re-convert a compatible SavedModel manually, you can still use `models/convert_to_tflite.py`, but the default reproducible path is the direct export from `models/train.py`.

Evaluate QAT int8 model:
```bash
python tools/evaluate_tflite.py \
  --tflite artifacts/model_sepqat2/qat_model.tflite \
  --manifest artifacts/manifest.jsonl \
  --image-size 96 \
  --split-mode train_compatible \
  --validation-split 0.2 \
  --seed 42
```

Notes:
- Current recommended QAT path uses `--architecture separable_cnn` in this pipeline.
- `--qat-from` is only needed for legacy experiments; the supported path now exports `artifacts/model_qat/qat_saved_model` for conversion.

### Quantized TFLite evaluation
After converting to a fully-quantized int8 TFLite, use the provided evaluation script to measure validation accuracy. The quantized model expects integer inputs (INT8/UINT8) — the script handles input quantization and output dequantization automatically.

Run the quick evaluation:
```bash
python tools/evaluate_tflite.py \
  --tflite artifacts/model/model.tflite \
  --manifest artifacts/manifest.jsonl \
  --image-size 96
```

Notes:
- If your TFLite model uses INT8 inputs/outputs the script will quantize the input images using the interpreter's quantization parameters (scale, zero-point). Do not feed raw float32 arrays directly to an INT8 model.
- The script defaults to `--split-mode train_compatible`, which reproduces `models/train.py` stratified split using `--validation-split 0.2 --seed 42`.
- You can switch to tail-based validation with `--split-mode tail --val-size 300`.
- Re-run `models/convert_to_tflite.py` after each new training run so `artifacts/model/model.tflite` is not stale.

## Notes

- The MCU integration is intentionally scaffolded, not yet hardware-complete.
- The current class taxonomy is a six-class Food-101 dish set: `apple_pie`, `bread_pudding`, `caesar_salad`, `cheesecake`, `deviled_eggs`, and `pizza`.
- The host receiver stores entries in JSONL so the format stays simple and append-only.
- The project is now scoped as dish classification rather than raw ingredient detection.
