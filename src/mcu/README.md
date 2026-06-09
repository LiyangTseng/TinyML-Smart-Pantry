# MCU Scaffold

This directory holds the Arduino / MCU side of the TinyML project.

## Current state

- The sketch is a starter scaffold that boots, opens Serial, and leaves hooks for camera capture, door trigger detection, and TFLite Micro inference.
- The hardware-specific camera and model integration will be filled in once the final board and camera module are confirmed.

## Next integration points

- wire up the camera driver for the chosen module
- embed the quantized model generated from `models/convert_to_tflite.py`
- replace the stub trigger and inference hooks in the sketch
