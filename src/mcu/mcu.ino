/*
 * Smart Pantry & Waste Reducer — MCU Inference Sketch
 *
 * Board  : Arduino Nano 33 BLE Sense Lite  (Mbed OS Nano Boards package)
 * Camera : OV7675 on Arduino Machine Learning Carrier Shield (AKX00028)
 * Model  : INT8 QAT micro-CNN, 96x96x1 grayscale, 6 food classes
 *
 * Required libraries (in Arduino IDE / Library Manager)
 *   Arduino_TensorFlowLite — manual install from:
 *     https://github.com/tensorflow/tflite-micro-arduino-examples  (ZIP)
 *   Harvard_TinyMLx        — search "TinyMLx" in Library Manager (includes
 *                            TinyMLShield.h with OV7675 96×96 capture support)
 *
 * Serial protocol (115200 baud, 8N1)
 *   On boot : "smart_pantry_ready"
 *   Per result: "<label>,<confidence>,<uptime_s>"
 *   Diagnostics: comma-free error tokens, e.g. "camera_init_failed"
 */

#include <Arduino.h>
#include <TinyMLShield.h>  // Harvard_TinyMLx — OV7675 96×96 native capture
#include <TensorFlowLite.h>
#include "model_data.h"
#include "tensorflow/lite/micro/micro_error_reporter.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/schema/schema_generated.h"

// ---------------------------------------------------------------------------
// Compile-time constants — model / image
// ---------------------------------------------------------------------------

// Image dimensions: must match the model's input shape [1, H, W, C].
static constexpr int kImgH = 96;
static constexpr int kImgW = 96;
static constexpr int kImgC = 1;  // Grayscale

// Class labels in the same order as the training manifest.
static constexpr int kNumLabels = 6;
static const char* const kLabels[kNumLabels] = {
    "apple_pie",
    "bread_pudding",
    "caesar_salad",
    "cheesecake",
    "deviled_eggs",
    "pizza",
};

// ---------------------------------------------------------------------------
// Compile-time constants — hardware
// ---------------------------------------------------------------------------

// Baud rate for Serial output.
static constexpr unsigned long kBaud = 115200;

// Digital pin wired to the door sensor (reed switch or IR break-beam).
// The sensor must pull this pin LOW when the door opens. The internal
// pull-up keeps it HIGH when the door is closed.
static constexpr int kDoorPin = 2;

// Debounce window: the pin must stay LOW for this long before we accept it.
static constexpr unsigned long kDebounceMsec = 50;

// Cooldown between successive captures (prevents burst-triggering).
static constexpr unsigned long kCooldownMsec = 2000;

// How long to keep the built-in LED on for the capture flash.
static constexpr unsigned long kFlashMsec = 40;

// OV7675 capture size used by the TinyML_EEP595 library.
static constexpr int kCaptureW = 160;
static constexpr int kCaptureH = 120;


// ---------------------------------------------------------------------------
// TFLite Micro arena
// The separable-CNN activation buffers require ~184 KB at runtime.
// 186 KB is the minimum working target for this model on Nano 33 BLE.
// If you see "allocate_tensors_failed", bump to 187 KB or 188 KB.
//
// The library only exposes Camera.readFrame(...), so we use a temporary
// 160x120 capture buffer on the stack and crop that down to 96x96 for the
// model.
// ---------------------------------------------------------------------------
static constexpr size_t kArenaSize = 186 * 1024;
alignas(16) static uint8_t gTensorArena[kArenaSize];

// ---------------------------------------------------------------------------
// TFLite Micro globals
// ---------------------------------------------------------------------------
static tflite::MicroErrorReporter gErrorReporter;
static const tflite::Model*       gModel       = nullptr;
static tflite::MicroInterpreter*  gInterpreter = nullptr;
static TfLiteTensor*              gInput       = nullptr;
static TfLiteTensor*              gOutput      = nullptr;

// ---------------------------------------------------------------------------
// Result type — declared here so Arduino's auto-prototype generator sees it
// before any function signature that references it.
// ---------------------------------------------------------------------------
struct InferenceResult {
    const char* label;
    float       confidence;
};

// ---------------------------------------------------------------------------
// Door-trigger state (debounce + cooldown)
// ---------------------------------------------------------------------------
static bool          gDoorLastState    = HIGH;
static unsigned long gDoorStableAt     = 0;
static unsigned long gLastCaptureMsec  = 0;

// ===========================================================================
// initModel()
// Load the embedded TFLite flatbuffer, allocate tensors, and bind I/O pointers.
// Returns false (and prints a diagnostic) on any failure.
// ===========================================================================
static bool initModel() {
    gModel = tflite::GetModel(g_model);
    if (gModel == nullptr) {
        Serial.println("model_null");
        return false;
    }
    // Schema version check omitted: TFLITE_SCHEMA_VERSION is not exported by
    // Arduino_TensorFlowLite 2.4.0-ALPHA. The flatbuffer is verified implicitly
    // by AllocateTensors() which will fail fast on a corrupt or mismatched model.

    // Use only the kernels required by artifacts/micro_cnn_qat/qat_model.tflite.
    static tflite::MicroMutableOpResolver<5> resolver;
    static bool resolverInitialized = false;
    if (!resolverInitialized) {
        resolver.AddConv2D();
        resolver.AddMaxPool2D();
        resolver.AddReshape();
        resolver.AddFullyConnected();
        resolver.AddSoftmax();
        resolverInitialized = true;
    }

    static tflite::MicroInterpreter interp(
        gModel, resolver, gTensorArena, kArenaSize, &gErrorReporter);
    gInterpreter = &interp;

    if (gInterpreter->AllocateTensors() != kTfLiteOk) {
        Serial.println("allocate_tensors_failed");
        return false;
    }

    gInput  = gInterpreter->input(0);
    gOutput = gInterpreter->output(0);
    if (gInput == nullptr || gOutput == nullptr) {
        Serial.println("tensor_bind_failed");
        return false;
    }

    // Guard: both tensors must be INT8 (QAT export guarantee).
    if (gInput->type != kTfLiteInt8 || gOutput->type != kTfLiteInt8) {
        Serial.println("expected_int8_tensors");
        return false;
    }

    // Guard: input shape must be [1, kImgH, kImgW, kImgC].
    const TfLiteIntArray* dims = gInput->dims;
    if (dims->size != 4
        || dims->data[1] != kImgH
        || dims->data[2] != kImgW
        || dims->data[3] != kImgC) {
        Serial.println("unexpected_input_shape");
        return false;
    }

    Serial.print("model_bytes,");
    Serial.println(g_model_len);
    return true;
}

// ===========================================================================
// initCamera()
// Initialise the OV7675 via the TinyML_EEP595 TinyMLShield library.
// The library supports QQVGA grayscale capture on the OV7675.
// ===========================================================================
static bool initCamera() {
    initializeShield();
    if (!Camera.begin(QQVGA, GRAYSCALE, 5, OV7675)) {
        Serial.println("camera_init_failed");
        return false;
    }
    Serial.println("camera_ready");
    return true;
}

// ===========================================================================
// fillInputTensor()
// Capture a QQVGA grayscale frame and center-crop to the model input size.
//
// Pixel bytes are normalized to [0, 1] before quantization:
//   v_norm = pixel_byte / 255.0
//   q      = round(v_norm / input_scale + input_zero_point)
//   q      = clamp(q, -128, 127)
// ===========================================================================
static bool fillInputTensor(TfLiteTensor* input) {
    // GRAYSCALE capture writes one byte per pixel.
    byte cameraFrame[kCaptureW * kCaptureH];
    Camera.readFrame(cameraFrame);

    const float   inScale = input->params.scale;
    const int32_t inZP    = input->params.zero_point;

    auto quantize = [&](uint8_t v) -> int8_t {
        const float   vn = static_cast<float>(v) / 255.0f;
        const int32_t q  = static_cast<int32_t>(roundf(vn / inScale + inZP));
        return static_cast<int8_t>(constrain(q, -128, 127));
    };

    const int xOff = (kCaptureW - kImgW) / 2;
    const int yOff = (kCaptureH - kImgH) / 2;

    for (int y = 0; y < kImgH; ++y) {
        for (int x = 0; x < kImgW; ++x) {
            const int src = (y + yOff) * kCaptureW + (x + xOff);
            const uint8_t v = cameraFrame[src];
            const int8_t q = quantize(v);
            const int dst = (y * kImgW + x);
            input->data.int8[dst] = q;
        }
    }
    return true;
}

// ===========================================================================
// doorOpenTriggered()
// Returns true exactly once per door-open event (falling edge on kDoorPin),
// subject to debounce and cooldown constraints.
// ===========================================================================
static bool doorOpenTriggered() {
    const bool pinLow = (digitalRead(kDoorPin) == LOW);

    // Track edge: restart the debounce timer whenever the raw reading changes.
    if (pinLow != gDoorLastState) {
        gDoorStableAt  = millis();
        gDoorLastState = pinLow;
    }

    // Require pin to be stably LOW for at least kDebounceMsec.
    if (!pinLow || (millis() - gDoorStableAt) < kDebounceMsec) {
        return false;
    }

    // Apply inter-capture cooldown.
    if ((millis() - gLastCaptureMsec) < kCooldownMsec) {
        return false;
    }

    gLastCaptureMsec = millis();
    return true;
}

// ===========================================================================
// runInference()
// Flash the onboard LED, capture + fill the input tensor, invoke the model,
// argmax the INT8 output, and dequantise the winning logit to a [0, 1]
// confidence score.
// ===========================================================================
static bool runInference(InferenceResult* result) {
    // Brief LED flash for consistent ambient lighting.
    digitalWrite(LED_BUILTIN, HIGH);
    delay(kFlashMsec);
    const bool ok = fillInputTensor(gInput);
    digitalWrite(LED_BUILTIN, LOW);

    if (!ok) {
        Serial.println("camera_read_failed");
        return false;
    }

    if (gInterpreter->Invoke() != kTfLiteOk) {
        Serial.println("invoke_failed");
        return false;
    }

    // Argmax over INT8 logits.
    const int nOut   = static_cast<int>(gOutput->bytes / sizeof(int8_t));
    int       bestI  = 0;
    int8_t    bestQ  = gOutput->data.int8[0];
    for (int i = 1; i < nOut; ++i) {
        if (gOutput->data.int8[i] > bestQ) {
            bestQ = gOutput->data.int8[i];
            bestI = i;
        }
    }

    // Dequantise: float = (q − zero_point) * scale.
    const float conf = (static_cast<float>(bestQ) - gOutput->params.zero_point)
                       * gOutput->params.scale;

    result->label      = (bestI < kNumLabels) ? kLabels[bestI] : "unknown";
    result->confidence = constrain(conf, 0.0f, 1.0f);
    return true;
}

// ===========================================================================
// sendResult()
// Emit a CSV line: label,confidence,uptime_s
// ===========================================================================
static void sendResult(const InferenceResult& r) {
    Serial.print(r.label);
    Serial.print(',');
    Serial.print(r.confidence, 4);
    Serial.print(',');
    Serial.println(millis() / 1000.0f, 3);
}

// ===========================================================================
// Arduino entry points
// ===========================================================================

void setup() {
    Serial.begin(kBaud);
    while (!Serial) { delay(10); }

    pinMode(LED_BUILTIN, OUTPUT);
    digitalWrite(LED_BUILTIN, LOW);

    // Active-LOW input with internal pull-up (reed switch / IR beam to GND).
    pinMode(kDoorPin, INPUT_PULLUP);

    if (!initModel()) {
        Serial.println("smart_pantry_model_init_failed");
        while (true) { delay(1000); }
    }

    if (!initCamera()) {
        Serial.println("smart_pantry_camera_init_failed");
        while (true) { delay(1000); }
    }

    Serial.println("smart_pantry_ready");
}

void loop() {
    if (!doorOpenTriggered()) {
        delay(20);
        return;
    }

    InferenceResult result = { "unknown", 0.0f };
    if (runInference(&result)) {
        sendResult(result);
    }
}
