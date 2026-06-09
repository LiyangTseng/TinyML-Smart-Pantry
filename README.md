# TinyML Smart Pantry & Waste Reducer

This folder contains the current implementation workflow for the TinyML final project.

Current deployed model target:
- Architecture: micro_cnn (QAT)
- Input: 96x96x1 grayscale, int8
- Classes: apple_pie, bread_pudding, caesar_salad, cheesecake, deviled_eggs, pizza
- MCU sketch: src/mcu/mcu.ino

## What is here

- `data/` contains dataset assembly helpers and label mapping.
- `models/` contains training scripts.
- `src/host/` contains the host-side receiver and USDA shelf-life lookup.
- `src/mcu/` contains the Arduino / MCU inference sketch and model header.
- `tools/` contains the TFLite evaluation harness.

## Quick start
Create a local virtual environment so collaborators use consistent deps:

```bash
cd /Users/li-yangtseng/Codes/uw/tinyml/project
./setup_venv.sh
source .venv/bin/activate
```

After activation you can run the tools without installing system-wide packages.

## End-to-end workflow (micro_cnn -> Arduino)

### 1) Prepare dataset
Download Food-101 subset into class folders:

```bash
source .venv/bin/activate
python data/download_food101.py \
  --output-dir artifacts/source_root \
  --label-map data/label_map.json \
  --max-per-class 1000
```

Build manifest from that source root:

```bash
python data/assemble_dataset.py \
  --source-root artifacts/source_root \
  --output-manifest artifacts/manifest.jsonl
```

### 2) Train micro_cnn with QAT (recommended path)

```bash
source .venv/bin/activate
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export TF_NUM_INTRAOP_THREADS=1
export TF_NUM_INTEROP_THREADS=1
export TF_CPP_MIN_LOG_LEVEL=2

python models/train.py \
  --manifest artifacts/manifest.jsonl \
  --output-dir artifacts/micro_cnn_qat \
  --architecture micro_cnn \
  --epochs 20 \
  --fine-tune-epochs 0 \
  --qat \
  --qat-epochs 8 \
  --qat-learning-rate 1e-5 \
  --batch-size 16 \
  --image-size 96 \
  --validation-split 0.2 \
  --seed 42
```

Training emits the deployment model directly at:
- artifacts/micro_cnn_qat/qat_model.tflite

### 3) Evaluate TFLite model

```bash
python tools/evaluate_tflite.py \
  --tflite artifacts/micro_cnn_qat/qat_model.tflite \
  --manifest artifacts/manifest.jsonl \
  --image-size 96 \
  --split-mode train_compatible \
  --validation-split 0.2 \
  --seed 42
```

- Do not pass .keras files to this evaluator; it expects a .tflite model.

### 4) Export TFLite to MCU header

Convert the trained model to src/mcu/model_data.h with the symbols expected by mcu.ino:

```bash
xxd -i artifacts/micro_cnn_qat/qat_model.tflite > src/mcu/model_data_raw.h
{
  echo '#pragma once'
  echo
  sed 's/^unsigned char artifacts_micro_cnn_qat_qat_model_tflite\[\] = {/alignas(16) const unsigned char g_model[] = {/' src/mcu/model_data_raw.h
  sed -n 's/^unsigned int artifacts_micro_cnn_qat_qat_model_tflite_len = /const unsigned int g_model_len = /p' src/mcu/model_data_raw.h
} > src/mcu/model_data.h
rm src/mcu/model_data_raw.h
```

### 5) Arduino setup and upload

1. In Arduino IDE, install board package: Arduino Mbed OS Nano Boards.
2. Select board: Arduino Nano 33 BLE Sense.
3. Install libraries:
   - Arduino_TensorFlowLite
   - Harvard_TinyMLx (TinyMLShield.h)
4. Open and upload src/mcu/mcu.ino.
5. Open Serial Monitor at 115200 baud.

Expected startup logs include:
- camera_ready
- model_bytes,<n>
- smart_pantry_ready

If startup fails, check these diagnostics:
- allocate_tensors_failed
- unexpected_input_shape
- camera_init_failed

## Notes

- src/mcu/main.ino is an older variant. Use src/mcu/mcu.ino as the active deployment sketch.
- The current class taxonomy is a six-class Food-101 dish set: `apple_pie`, `bread_pudding`, `caesar_salad`, `cheesecake`, `deviled_eggs`, and `pizza`.
- The host receiver stores entries in JSONL so the format stays simple and append-only.
- The project is scoped as dish classification rather than raw ingredient detection.
