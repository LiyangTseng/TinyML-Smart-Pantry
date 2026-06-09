# Implementation Plan — TinyML Smart Pantry & Waste Reducer

Last updated: 2026-05-30

This plan translates the research findings into a phased, actionable implementation roadmap. It defines architecture, engineering trade-offs, exact repository locations for artifacts, conceptual code snippets, evaluation plans, and a granular TODO list broken into phases.

**Scope & Assumptions**
- Edge device: Arduino Nano 33 BLE Sense (ARM Cortex-M4F) or equivalent.
- Camera: OV7675 (or compatible small CMOS module) attached via supported shield/driver.
- Communication: initial implementation uses Serial (USB) for simplicity; BLE optional in Phase 3.
- Class set: configurable; initial prototype targets 6 Food-101 dish classes (apple_pie, bread_pudding, caesar_salad, cheesecake, deviled_eggs, pizza).

**High-Level Architecture**
- Edge (MCU): camera capture → lightweight preprocessing → TFLite Micro int8 inference → emit event (label id, confidence, timestamp) over Serial/BLE.
- Host (desktop/phone): receive events → apply USDA FoodKeeper lookup → persist inventory with expiration date → optional UI for viewing/confirming entries.
- Training pipeline: assemble datasets → train compact model (MobileNetV2 small or custom 3-layer CNN) → post-training quantize to int8 → validate with in-situ images → produce TFLite Micro artifact and microcontroller build files.

**Repository layout (files to create/modify)**
- `data/assemble_dataset.py` — script to download, filter, and canonicalize Food-101 and other datasets into the project taxonomy.
- `data/README.md` — dataset sources, label mapping JSON (e.g., `data/label_map.json`) and preprocessing notes.
- `models/train.py` — training script for prototyping (TensorFlow + Keras) with configurable architecture and augmentations.
- `models/convert_to_tflite.py` — conversion and quantization script; produces `models/model.tflite` and `models/model_micro.cc` skeleton.
- `models/representative_data/` — samples used for int8 calibration.
- `src/host/receiver.py` — host-side Python script to receive events over Serial/BLE and persist inventory to `data/inventory.json`.
- `src/host/usda_lookup.py` — small module encoding USDA FoodKeeper shelf-life mapping and expiration calculation.
- `src/mcu/` — MCU project folder (PlatformIO or Arduino) containing micro inference glue, camera driver integration, and `src/mcu/main.ino` or `src/mcu/main.cpp`.
- `tools/evaluate_in_situ.py` — evaluation scripts to compute Top-1 accuracy and confusion matrices using captured fridge images.

**Conceptual Code Snippets**

1) Host receiver (minimal Python, conceptual)

```python
# src/host/receiver.py
import serial, json, time
from usda_lookup import get_shelf_life

ser = serial.Serial('/dev/ttyUSB0', 115200, timeout=1)
while True:
    line = ser.readline().decode().strip()
    if not line: continue
    # expected format: LABEL_ID,CONFIDENCE,TS
    label, conf, ts = line.split(',')
    shelf_days = get_shelf_life(label)
    expire = int(ts) + shelf_days * 24*3600
    # append to inventory.json
    with open('data/inventory.json','a') as f:
        f.write(json.dumps({'label':label,'conf':float(conf),'ts':int(ts),'expire':expire})+'\n')
```

2) TFLite conversion snippet (conceptual)

```python
# models/convert_to_tflite.py
import tensorflow as tf

model = tf.keras.models.load_model('models/checkpoint/best.h5')
converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.representative_dataset = lambda: (x for x in tf.data.Dataset.from_tensor_slices(rep_samples).batch(1))
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
converter.inference_input_type = tf.uint8
converter.inference_output_type = tf.uint8
tflite_model = converter.convert()
open('models/model.tflite','wb').write(tflite_model)
```

3) MCU inference loop (pseudocode)

```c
// src/mcu/main.cpp (conceptual)
setup() {
  camera_init();
  tflite_micro_init(model_data);
}
loop() {
  if (door_open_trigger()) {
    capture_frame(buf);
    preprocess(buf, input_tensor);
    tflite_invoke();
    int label = argmax(output_tensor);
    float conf = output_tensor[label];
    send_serial(label, conf, now());
  }
}
```

**Engineering Trade-offs and Rationale**

- Model choice: `MobileNetV2` small provides higher accuracy for similar parameter budget vs a tiny custom CNN, but risk higher activation memory; start with MobileNetV2 (width multiplier 0.35) and fall back to custom CNN if memory exceeds MCU limits.
- Input size vs accuracy: `96×96` recommended as baseline; lowering to `64×64` reduces memory and compute with accuracy trade-offs—test both.
- Quantization: prefer full-integer `int8` conversion (supported by TFLite Micro) for size and speed; requires representative calibration data.
- Communication: Serial during development simplifies debugging. BLE adds UX benefits but increases complexity (pairing, security) and code size.
- On-device processing vs host: keep identity classification on-device for privacy. Host handles USDA mapping and longer-term storage.

**Evaluation Plan**

- Baseline (off-device): train and evaluate model on held-out test split from assembled dataset; measure Top-1, precision/recall, and confusion matrix.
- In-situ: collect ≥50 labeled images per class in target fridge lighting; run `tools/evaluate_in_situ.py` to compute in-situ accuracy.
- MCU profiling: instrument inference time and peak RAM usage; verify model binary size and stack/heap budgets.

**Risk Matrix and Mitigations (short)**
- Memory overrun: mitigate by reducing width multiplier, input size, or class count; use streaming capture to reduce buffer usage.
- Poor real-world accuracy: collect representative in-situ images for fine-tuning and calibration; add flash/controlled lighting.
- Camera driver incompatibility: research/choose camera modules with existing Arduino examples or use an external camera + host-assisted capture for initial prototype.

**Phase-based Implementation Plan (granular TODOs)**

Phase 0 — Design confirmation (1–2 days)
- Task 0.1: Confirm final class list (six Food-101 dish classes). (Owner: team)
- Task 0.2: Confirm hardware & camera module choice. (Owner: team)
  - Edited: confirmed that we would use Arduino Nano 33 BLE Sense Rev2 + OV7675 camera module
- Task 0.3: Choose initial communication channel (Serial for dev). (Owner: team)
  - Edited: I think we can first have a working food classification prototype, then add inventory checking support in Phase 3 if time allows. This will let us focus on the core ML and integration challenges first. (Owner: team)

Phase 1 — Data assembly & preprocessing (3–5 days)
- Task 1.1: Create `data/assemble_dataset.py` to pull Food-101 subset and map to taxonomy.
- Task 1.2: Create `data/label_map.json` and `data/README.md` documenting sources and license.
- Task 1.3: Create augmentation pipeline and representative calibration set `models/representative_data/`.

Phase 2 — Model prototyping & conversion (5–10 days)
- Task 2.1: Implement `models/train.py` with MobileNetV2 (configurable width/input size).
- Task 2.2: Run experiments on Colab; record hyperparameters and checkpoints.
- Task 2.3: Convert best checkpoint using `models/convert_to_tflite.py` to int8; produce `models/model.tflite` and `models/model_micro.cc`.
- Task 2.4: Run off-device evaluation and iterate until acceptable accuracy.

Phase 3 — Host-side receiver & USDA integration (2–3 days)
- Task 3.1: Implement `src/host/receiver.py` and `src/host/usda_lookup.py`.
- Task 3.2: Implement `data/inventory.json` storage format and simple CLI to list items.

Phase 4 — MCU integration & testing (5–10 days)
- Task 4.1: Set up ~~PlatformIO/~~Arduino project at `src/mcu/` and integrate TFLite Micro example.
- Task 4.2: Integrate camera driver and implement capture pipeline with minimal buffering.
- Task 4.3: Integrate quantized model into MCU build and implement inference loop.
- Task 4.4: Measure RAM, flash, and inference time; iterate on model if resource limits hit.

Phase 5 — In-situ evaluation, UX polish & documentation (3–5 days)
- Task 5.1: Collect in-situ dataset; run `tools/evaluate_in_situ.py` and generate report.
- Task 5.2: Add BLE support (optional) and pairing flow if requested.
- Task 5.3: Finalize reproducibility artifacts and write `README.md` with build instructions and evaluation results.

Phase 6 — Buffer & final deliverables (2–3 days)
- Task 6.1: Fix remaining issues from integration testing.
- Task 6.2: Package model and MCU firmware for submission.

**Estimated Timeline Summary**
- Minimum viable prototype (Serial-based host, working classification on dev hardware): 3–4 weeks (parallelizable: training + host development).
- Full integrated system (BLE, polished UI, robust in-situ accuracy): additional 2–3 weeks.

---

Granular TODO checklist (copyable)

- [x] Phase 0: Confirm class list, hardware, comms.
- [x] Phase 1: `data/assemble_dataset.py`, `data/label_map.json`, representative calibration set.
- [x] Phase 2: `models/train.py`, training experiments, `models/convert_to_tflite.py`.
- [ ] Phase 3: `src/host/receiver.py`, `src/host/usda_lookup.py`, `data/inventory.json`.
- [ ] Phase 4: `src/mcu/` PlatformIO/Arduino project, camera driver integration, MCU inference loop.
- [ ] Phase 5: Collect in-situ images, `tools/evaluate_in_situ.py`, final report.

If you approve this plan I will begin Phase 0 tasks: (1) confirm the final class list and (2) confirm the exact hardware/camera choice. After you answer those, I'll start Phase 1 and create `data/assemble_dataset.py` and `data/label_map.json`.
