# MCU Scaffold

This directory holds the Arduino / MCU side of the TinyML project.

## Active deployment path

- Active sketch: `mcu.ino`
- Target board: Arduino Nano 33 BLE Sense
- Camera path: TinyMLShield OV7675 grayscale capture
- Model format: int8 TFLite, embedded as `model_data.h` (`g_model`, `g_model_len`)
- Model input: `[1, 96, 96, 1]` (grayscale)

`main.ino` is an older variant and is not the recommended deployment target.

## Required Arduino setup

1. Install board package: **Arduino Mbed OS Nano Boards**.
2. Select board: **Arduino Nano 33 BLE Sense**.
3. Install libraries:
   - **Arduino_TensorFlowLite**
   - **Harvard_TinyMLx** (provides `TinyMLShield.h`)
4. Use **115200 baud** in Serial Monitor.

## Deploy trained micro_cnn model

From project root, export the trained model into the header used by `mcu.ino`:

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

Notes:
- `mcu.ino` expects `src/mcu/model_data.h` with symbols `g_model` and `g_model_len`.
- Re-generate `model_data.h` every time you produce a new `qat_model.tflite`.

## Runtime behavior in mcu.ino

- Camera starts in QQVGA grayscale, then center-crops to 96x96.
- Input tensor is int8 grayscale with quantization:
  - `q = round((pixel/255.0) / input_scale + input_zero_point)`
  - clamp to `[-128, 127]`
- Inference triggers on door-open signal (active low input on pin 2).
- Output emits CSV: `label,confidence,uptime_s`.

## Bring-up checklist

1. Upload `mcu.ino`.
2. Confirm startup log includes:
   - `camera_ready`
   - `model_bytes,<n>`
   - `smart_pantry_ready`
3. Trigger the door input and verify predictions print as CSV.

If boot fails, check diagnostics:
- `allocate_tensors_failed` -> increase arena size slightly.
- `unexpected_input_shape` -> exported model does not match expected 96x96x1.
- `camera_init_failed` -> camera stack/library mismatch.

## Kit reference links

- Kit product page (official):
  - https://store.arduino.cc/products/arduino-tiny-machine-learning-kit
- Kit hardware docs page:
  - https://docs.arduino.cc/hardware/tiny-machine-learning-kit/
- Machine Learning Carrier schematic PDF:
  - https://content.arduino.cc/assets/MachineLearningCarrierV1.0.pdf

## person_detection reference

`Arduino_TensorFlowLite > person_detection` is useful for camera stack examples,
but this project differs in two important ways:
- Uses your own six-class dish model instead of person/no-person.
- Uses int8 grayscale `96x96x1` input path in `mcu.ino`.

