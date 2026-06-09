# TinyML Smart Pantry & Waste Reducer — Research Notes

> Historical research and design notes.
> For current executable workflow and deployment instructions, use README.md and src/mcu/README.md.

Authors: Miles Chin (2577815), Daniel Tseng (2577912)

Last updated: 2026-05-30

## Project Overview

A low-power, privacy-preserving TinyML device mounted at a fridge or pantry entrypoint that captures short RGB images of items being placed into storage, classifies the item (from a constrained set of classes), and emits a simple event (item label + timestamp). A host-side component stores the item with an estimated expiration date (using USDA FoodKeeper shelf-life mappings). The edge device is based on an Arduino Nano 33 BLE Sense (or equivalent) with a small camera (OV7675 or similar) running a highly optimized, quantized TensorFlow Lite Micro model.

This document consolidates all research findings, non-actionable requirements, constraints, potential pitfalls, and open questions required to decide whether and how to proceed to implementation.

## Objectives and Success Criteria

- Primary objective: Demonstrate a TinyML pipeline that correctly classifies a selected set of common fridge items (e.g., apple, banana, milk, eggs, orange, bread) on-device with usable accuracy and fits the memory/compute envelope of the Arduino Nano 33 BLE Sense.
- Success metrics (proposed):
  - Top-1 accuracy >= 85% on an in-situ test set (>50 images captured under target lighting conditions) for the chosen class subset.
  - Model binary (flash image) and runtime memory footprint that fits within available device resources (target: program + data + runtime <= flash + 256 KB RAM working set for inference).
  - End-to-end event latency (capture → inference → host notification) acceptable for interactive use (heuristic target: < 1 s on-device inference, host handshake variable).
  - Privacy: No images leave the device; only labels and timestamps are transmitted.

## Constraints and Non-Functional Requirements

- Hardware memory: Arduino Nano 33 BLE Sense typical RAM ≈ 256 KB (usable working RAM for model activation will be a fraction of that) and flash ≈ 1 MB — exact usable amounts depend on other firmware.
- Power: Device should be low-power; inference should be fast and infrequent (triggered on door opening). Keep model and runtime energy low.
- Dataset / class scope: limit to 5–10 classes to reduce final layer size and class confusion.
- Input size: target image size ~96×96 RGB to balance accuracy vs compute.
- Model format: TensorFlow Lite Micro with int8 post-training quantization (or full-integer quantization) for runtime on MCU.

## Datasets (inventory & notes)

- Food-101 (TensorFlow Datasets): large, real-food dataset. Use a small, targeted subset of 5–10 classes that match common fridge items. Note: Food-101 contains diverse, web-scraped images; domain mismatch to fridge photos is expected.
- Food Freshness Dataset (Kaggle): useful for variability in appearance vs freshness; may help robustness but may not be strictly required for classification of item identity.
- USDA FoodKeeper: non-image dataset mapping generic food items to shelf-life ranges; used as a lookup/metadata table on the host to determine recommended expiration dates.
- Local in-situ photos: Small dataset of images taken in the actual target environment (fridge with LED flash) is critical for evaluation and possibly for fine-tuning; aim for at least 50–100 images per class for a credible in-situ test set.

Data quality considerations:
- Domain gap: web images vs fridge images (lighting, occlusion, packaging). Expect accuracy drop unless in-domain examples are included in training or fine-tuning.
- Label granularity: Food-101 has fine-grained labels; mapping to fridge-relevant labels is required (e.g., many apple varieties → unify to `apple`).
- Class imbalance: choose classes with enough examples and real-world frequency to avoid poor performance on underrepresented classes.

## Data Preparation & Augmentation (analysis only)

- Input normalization: consistent resizing and center-cropping to the model input size (96×96) with per-channel mean/std normalization appropriate for training framework.
- Augmentations to improve robustness (recommended to consider): brightness/contrast jitter to simulate fridge lighting variability, small rotations, translations, random occlusion (partially obscured items), and synthetic packaging variance.
- Label mapping: canonicalize labels to the chosen class set; create a mapping table from dataset taxonomy → project taxonomy.

## Model Architecture Options (analysis)

- Scaled-down MobileNetV2 (width multiplier 0.35–0.5) — tradeoff: good baseline accuracy with efficient building blocks. Pros: proven on edge, supported by TFLite conversion. Cons: may still be large; requires careful memory budgeting.
- Custom lightweight 3-layer CNN — tradeoff: very small flash/RAM usage but likely lower accuracy. May suffice for small class sets and constrained domain.
- TinyML-specific optimized architectures (e.g., MobileNetV1 small, EfficientNet-lite variants) — evaluate model size vs accuracy.

Memory & compute budgeting (estimations, cross-check during implementation):
- Quantized int8 model size often reduces binary size by ~4× vs float32 weights; activations still require working RAM. Typical microcontroller deployments aim for model weights < 200 KB and peak activation memory < 100 KB where possible.
- Final layer size scales with number of classes: keep classes minimal to avoid large dense layers.

## Optimization Techniques (analysis)

- Post-training quantization to int8 (TFLite full-integer): reduces model size and enables integer-only inference on MCU.
- Pruning can reduce parameter count pre-quantization but may require retraining/fine-tuning; pruning benefits on MCU can be limited unless combined with weight clustering or sparsity-aware code generation.
- Operator selection: ensure chosen ops are supported by TFLite Micro or have fallbacks (depthwise conv, pointwise conv, etc.). Avoid exotic ops that complicate micro builds.

## Input & Inference Pipeline (data flow analysis)

High-level dataflow (conceptual):
- Trigger (door open / light change sensor) → capture small RGB frame from camera → basic preprocessing (resize, normalize) → pass tensor to on-device model → obtain class logits → apply argmax + confidence → send minimal event (label id, confidence, timestamp) to host via Serial or BLE.

Key observations:
- Preprocessing must be computationally cheap and deterministic on MCU codepath; heavy preprocessing should be done on host if necessary, but that conflicts with privacy and offline goals.
- Confidence threshold: decide a safe threshold below which the device should either send `unknown` or request host-side validation to avoid incorrect automatic logging.

## Hardware & Integration Notes

- Target board: Arduino Nano 33 BLE Sense (ARM Cortex-M4F) — confirm exact RAM/flash available in firmware image.
- Camera module: OV7675 or equivalent small CMOS camera with SPI/I2C or parallel interface compatible with chosen Arduino shield. Verify driver availability for the chosen camera on the board.
- Trigger sensors: light sensor (detect door opening by light delta) or door switch. Onboard IMU or microphone could be alternate triggers.
- Host communication: Serial (USB) or BLE Low Energy to a small host script (desktop/phone) that receives events and stores metadata + expiration from USDA lookup.

Hardware integration risks:
- Camera driver availability and memory usage for frame buffering can dominate RAM demands; confirm native camera driver behavior (frame buffering vs streaming).
- Using onboard LED as flash increases power draw and changes exposure characteristics; test for image quality.

## Evaluation Strategy & Metrics

- Core metrics: Top-1 accuracy, confusion matrix across classes, per-class precision/recall, false positive rate for `unknown`/misclassifications.
- Robustness tests: evaluate under variable lighting, partial occlusion, and with common packaging (boxes, bottles) present.
- On-device performance: measure peak RAM usage during inference, model size in flash, inference time (ms), and energy per inference if possible.

## Edge Cases & Failure Modes

- Ambiguous items and multi-item captures: scene may contain multiple items or partially visible item — device may misclassify.
- Packaging vs unwrapped produce: classifier trained only on unwrapped items may mislabel packaged goods.
- Similar-looking classes (e.g., orange vs tangerine) — if these distinctions are irrelevant, merge labels to reduce confusion.
- Low-confidence predictions: need a policy for host handling (e.g., host prompts user or logs as `unknown`).

## Privacy, Security & Ethics

- Privacy: maintain the principle that no images leave the device. Only transmit labels, timestamps, and optional low-entropy metadata.
- Data retention: store only minimal metadata on host; if images must be stored for debugging, make this opt-in and encrypted.
- Security: protect BLE/Serial endpoints against unauthorized write commands; use pairing/UUID filtering for BLE and authenticated host channels where possible.

## Reproducibility & Artifacts to Produce

The following artifacts should be produced (for reproducibility and auditing):
- Training dataset snapshot and label mapping files (or exact code pointers for dataset assembly).
- Model training configuration (architecture, hyperparameters, augmentation pipeline, epochs, optimizer, loss), training logs, and final model checkpoints.
- TFLite conversion script + quantization calibration data (representative samples used for int8 calibration).
- Exact microcontroller build recipe: toolchain version, board variant, TFLite Micro commit/version, and any compile-time flags.
- Small in-situ evaluation image set and evaluation script used to compute the final reported metrics.

## Dependencies & Tooling (candidates)

- Training and conversion: Python 3.x, TensorFlow (2.x), TensorFlow Lite Converter, TFLite Micro tools.
- Host-side: Python 3.x script for receiving events, storing items, and applying USDA lookup.
- Development: Arduino IDE or PlatformIO for MCU build; confirm required support for chosen camera driver and TFLite Micro port.

## Risks and Mitigations (analysis)

- Risk: Model and buffers exceed MCU memory budget.
  - Mitigation: reduce model capacity, reduce input resolution, reduce class count, or move some logic to host (tradeoffs with privacy).
- Risk: Camera driver or frame buffering consumes too much RAM.
  - Mitigation: use streaming capture, reduce temporary buffers, or select camera modules/drivers tested for MCU targets.
- Risk: Accuracy too low in real fridge lighting.
  - Mitigation: collect representative in-situ images and either fine-tune or calibrate preprocessing; consider adding on-device flash.

## Open Questions / Decisions Required

1. Final class list (5–10 items). The proposal suggests common items (apple, banana, milk, eggs); confirm final taxonomy.
2. Exact hardware selection: confirm Arduino model and camera module; verify driver availability and memory characteristics.
3. Communication channel preference: Serial vs BLE (BLE adds pairing complexity but enables phone-only workflows).
4. Acceptance threshold for confidence values and host policy on low-confidence events.
5. Tolerance for on-device vs host-side processing tradeoffs (privacy vs feasibility).

## Estimated Deliverables (for user decision)

- `research.md` (this document) — requirements and analysis.
- A list of reproducibility artifacts (datasets, mapping, training configs) to produce if proceeding.
- A proposed minimal class taxonomy and an in-situ data-collection plan (if approved to move forward).

---

Please review this `research.md` and confirm which open questions you want to resolve (class list, hardware choice, communication channel, and privacy tradeoffs). Once you confirm, the next step would be to assemble the reproducibility artifacts and collect an in-situ evaluation set.
