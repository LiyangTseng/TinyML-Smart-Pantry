# MCU Scaffold

This directory holds the Arduino / MCU side of the TinyML project.

## Current state

- The sketch is a starter scaffold that boots, opens Serial, and leaves hooks for camera capture, door trigger detection, and TFLite Micro inference.
- The hardware-specific camera and model integration will be filled in once the final board and camera module are confirmed.

## Next integration points

- wire up the camera driver for the chosen module
- embed the quantized model generated from `models/convert_to_tflite.py`
- replace the stub trigger and inference hooks in the sketch

## Deploy current best model (sweep_b)

From project root, export the best int8 model into a header used by `main.ino`:

```bash
xxd -i artifacts/sweep_b/qat_model.tflite > src/mcu/model_data_raw.h
{
	echo '#pragma once'
	echo
	sed 's/^unsigned char artifacts_sweep_b_qat_model_tflite\[\] = {/alignas(16) const unsigned char g_model[] = {/' src/mcu/model_data_raw.h
	sed -n 's/^unsigned int artifacts_sweep_b_qat_model_tflite_len = /const unsigned int g_model_len = /p' src/mcu/model_data_raw.h
} > src/mcu/model_data.h
rm src/mcu/model_data_raw.h
```

Notes:
- `main.ino` now expects `src/mcu/model_data.h` with symbols `g_model` and `g_model_len`.
- The sketch initializes TFLite Micro and decodes int8 outputs, but camera capture and door trigger are still TODO hooks for board-specific integration.
- Install Arduino library dependency: **Arduino_TensorFlowLite**.

## Nano 33 BLE Sense final configuration checklist

Target board confirmed: Arduino Nano 33 BLE Sense.

### 1) Arduino IDE settings

- Install **Arduino Mbed OS Nano Boards** via Boards Manager.
- Select board: **Arduino Nano 33 BLE Sense**.
- Install library: **Arduino_TensorFlowLite**.
- Keep Serial Monitor at **115200 baud** to match the sketch.

### 2) Reuse what worked from previous labs

- Keep the TFLM initialization flow from `main.ino`: model load, schema check, tensor allocation, input/output tensor binding.
- From `labs/lab8/lab8-classfier/lab8-classfier.ino`, keep the simple run loop pattern: trigger -> fill input tensor -> invoke -> print outputs.
- From `hw2/.../network_data.ino`, keep these robustness habits:
	- check input/output tensor type before writing/reading
	- use quantization params (`scale`, `zero_point`) for int8 conversion
	- fail fast on invoke or tensor errors with explicit Serial logs

### 3) Camera integration (OV7675) path

Important: OV7675 support depends on the exact camera board and driver style.

- If your OV7675 module requires raw parallel capture (no FIFO), Nano 33 BLE Sense integration can be difficult.
- If your camera board exposes a FIFO/shielded interface with an Arduino driver, use that library and adapt `fillInputTensorFromCamera(...)`.

Minimum camera implementation tasks in `main.ino`:

- initialize camera in `setup()` and print an error if camera init fails
- capture one frame when trigger is active
- resize/crop to model input (`96x96x3`)
- write int8 RGB values into `gInputTensor->data.int8[]` using input quantization params

Quantization write rule for each pixel channel value `x` in `[0,255]`:

- `q = round(x / input_scale) + input_zero_point`
- clamp `q` to `[-128, 127]`
- store into `input->data.int8[idx]`

### 4) Door trigger completion

- Wire reed switch (or chosen trigger pin) to a digital input.
- Implement `doorOpenTriggered()` as a debounced digital read.
- Keep the `delay(250)` guard to avoid repeated duplicate events.

### 5) Bring-up sequence

1. Upload sketch and confirm `smart_pantry_ready` appears on Serial.
2. Verify model boot logs (`model_bytes,...`) and no tensor/schema errors.
3. Add camera init log and verify successful camera startup.
4. Add a temporary synthetic input path in `fillInputTensorFromCamera(...)` to test `Invoke()` before wiring real capture.
5. Switch to real camera frame preprocessing and check that labels/confidence stream correctly.

## Where to find exact kit information

If you are using the Arduino Tiny Machine Learning Kit (AKX00028), verify camera and shield details from these sources:

- Kit product page (official):
	- https://store.arduino.cc/products/arduino-tiny-machine-learning-kit
	- Check the Tech specs section for the included board and camera module.
- Kit hardware docs page:
	- https://docs.arduino.cc/hardware/tiny-machine-learning-kit/
	- If this page changes, use the product page Documentation links instead.
- Shield schematic PDF (pin mapping and electrical connections):
	- https://content.arduino.cc/assets/MachineLearningCarrierV1.0.pdf

Practical checks on physical hardware:

- Verify board marking: Nano 33 BLE Sense Lite variant used in the kit.
- Verify shield marking: Machine Learning Carrier version printed on PCB.
- Verify camera board marking: OV7675 and any carrier/FIFO identifiers printed on the module.

In Arduino IDE, the fastest way to confirm required software pieces is:

- File > Examples > Arduino_TensorFlowLite > person_detection
- Read that example and its included headers to see expected camera provider path and supported camera stack for your installed library version.

## What person_detection teaches us

Useful patterns to copy into this project:

- Use a minimal TFLM op resolver instead of `AllOpsResolver` once the model ops are known.
- Keep the camera read logic in a separate `GetImage`-style function, which maps well to `fillInputTensorFromCamera(...)`.
- Keep model boot, inference, and result reporting separated into small functions instead of one large loop.
- Allocate a large enough tensor arena and trim it only after the camera path is working.

Important differences from person_detection:

- Their example uses an older camera/image pipeline and a different model layout.
- Their output handling uses `uint8` scores, while your project should use the int8 quantized model from `artifacts/sweep_b/qat_model.tflite`.
- Your deployment path should write RGB pixels into a `96x96x3` int8 input tensor, not IMU data.

Practical next step for your kit:

- Use the `person_detection` sketch as a library reference for the camera stack only.
- Keep your current `main.ino` as the inference shell.
- Once the camera driver compiles, replace the stub in `fillInputTensorFromCamera(...)` with the kit's capture API and int8 quantization writeback.
