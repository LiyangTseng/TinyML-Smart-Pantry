/*
 * Smart Pantry & Waste Reducer — MCU Inference Sketch
 *
 * Board  : Arduino Nano 33 BLE Sense Lite  (Mbed OS Nano Boards package)
 * Camera : OV7675 on Arduino Machine Learning Carrier Shield (AKX00028)
 * Model  : INT8 QAT micro-CNN, 96x96x1 grayscale, 6 food classes
 *
 * Serial protocol (115200 baud, 8N1)
 * On boot : "smart_pantry_ready"
 * Per result: "<label>,<confidence>,<uptime_s>"
 * Diagnostics: comma-free error tokens, e.g. "camera_init_failed"
 */

#include <Arduino.h>
#include <TinyMLShield.h>  // Harvard_TinyMLx — OV7675 native capture
#include <TensorFlowLite.h>
#include "model_data.h"
#include "tensorflow/lite/micro/micro_error_reporter.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/schema/schema_generated.h"

// ---------------------------------------------------------------------------
// Compile-time constants — model / image
// ---------------------------------------------------------------------------
static constexpr int kImgH = 96;
static constexpr int kImgW = 96;
static constexpr int kImgC = 3;  // RGB

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
static constexpr unsigned long kBaud = 115200;
static constexpr int           kDoorPin = 2;
static constexpr unsigned long kDebounceMsec = 50;
static constexpr unsigned long kCooldownMsec = 2000;
static constexpr unsigned long kFlashMsec = 40;

// QQVGA dimensions used by the Harvard library driver configuration
static constexpr int kCaptureW = 160;
static constexpr int kCaptureH = 120;

// ---------------------------------------------------------------------------
// TFLite Micro arena
// ---------------------------------------------------------------------------
static constexpr size_t kArenaSize = 186 * 1024;
alignas(16) static uint8_t gTensorArena[kArenaSize];

static tflite::MicroErrorReporter gErrorReporter;
static const tflite::Model* gModel       = nullptr;
static tflite::MicroInterpreter* gInterpreter = nullptr;
static TfLiteTensor* gInput       = nullptr;
static TfLiteTensor* gOutput      = nullptr;

struct InferenceResult {
    const char* label;
    float       confidence;
};

static bool          gDoorLastState    = HIGH;
static unsigned long gDoorStableAt     = 0;
static unsigned long gLastCaptureMsec  = 0;

// ===========================================================================
// initModel()
// ===========================================================================
static bool initModel() {
    gModel = tflite::GetModel(g_model);
    if (gModel == nullptr) {
        Serial.println("model_null");
        return false;
    }

    static tflite::MicroMutableOpResolver<7> resolver;
    static bool resolverInitialized = false;
    if (!resolverInitialized) {
        resolver.AddConv2D();
        resolver.AddMaxPool2D();
        resolver.AddReshape();
        resolver.AddFullyConnected();
        resolver.AddSoftmax();
        resolver.AddMean();
        resolver.AddAveragePool2D();
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

    if (gInput->type != kTfLiteInt8 || gOutput->type != kTfLiteInt8) {
        Serial.println("expected_int8_tensors");
        return false;
    }

    return true;
}

// ===========================================================================
// initCamera()
// ===========================================================================
static bool initCamera() {
    initializeShield();
    if (!Camera.begin(QQVGA, RGB565, 5, OV7675)) {
        Serial.println("camera_init_failed");
        return false;
    }
    Serial.println("camera_ready");
    return true;
}

// ===========================================================================
// fillInputTensor()
// ===========================================================================
static bool fillInputTensor(TfLiteTensor* input) {
    // RGB565 uses 2 bytes per pixel
    byte cameraFrame[kCaptureW * kCaptureH * 2];
    Camera.readFrame(cameraFrame);

    const float   inScale = input->params.scale;
    const int32_t inZP    = input->params.zero_point;

    // Direct center tracking
    int i = 0;
    for (int y = 0; y < kCaptureH; y++) {
        for (int x = 0; x < kCaptureW; x++) {
            // Check if current camera pixel falls inside the central 96x96 box
            if (x >= 32 && x < 128 && y >= 12 && y < 108) {
                int pixelIdx = (y * kCaptureW + x) * 2;
                uint8_t byte1 = cameraFrame[pixelIdx];
                uint8_t byte2 = cameraFrame[pixelIdx + 1];
                uint16_t pixel = (byte1 << 8) | byte2;
                
                uint8_t r = ((pixel >> 11) & 0x1F);
                uint8_t g = ((pixel >> 5) & 0x3F);
                uint8_t b = (pixel & 0x1F);
                
                // Convert to [0.0, 1.0] range
                float r_norm = static_cast<float>((r * 255) / 31) / 255.0f;
                float g_norm = static_cast<float>((g * 255) / 63) / 255.0f;
                float b_norm = static_cast<float>((b * 255) / 31) / 255.0f;
                
                int32_t qr = static_cast<int32_t>(roundf(r_norm / inScale) + inZP);
                int32_t qg = static_cast<int32_t>(roundf(g_norm / inScale) + inZP);
                int32_t qb = static_cast<int32_t>(roundf(b_norm / inScale) + inZP);
                
                input->data.int8[i++] = static_cast<int8_t>(constrain(qr, -128, 127));
                input->data.int8[i++] = static_cast<int8_t>(constrain(qg, -128, 127));
                input->data.int8[i++] = static_cast<int8_t>(constrain(qb, -128, 127));
            }
        }
    }
    return true;
}

// ===========================================================================
// doorOpenTriggered()
// ===========================================================================
static bool doorOpenTriggered() {
    const bool pinLow = (digitalRead(kDoorPin) == LOW);

    if (pinLow != gDoorLastState) {
        gDoorStableAt  = millis();
        gDoorLastState = pinLow;
    }

    if (!pinLow || (millis() - gDoorStableAt) < kDebounceMsec) {
        return false;
    }

    if ((millis() - gLastCaptureMsec) < kCooldownMsec) {
        return false;
    }

    gLastCaptureMsec = millis();
    return true;
}

// ===========================================================================
// visualDebugPrint()
// ===========================================================================
static void visualDebugPrint(TfLiteTensor* input) {
    Serial.println("--- START ARDUINO MATRIX DUMP ---");
    // Sub-sample down to a 24x24 text block so it prints quickly over Serial
    for (int y = 0; y < kImgH; y += 4) {
        for (int x = 0; x < kImgW; x += 4) {
            int dst = (y * kImgW + x) * 3;
            int32_t r = input->data.int8[dst];
            int32_t g = input->data.int8[dst + 1];
            int32_t b = input->data.int8[dst + 2];
            int32_t pixelVal = (r + g + b) / 3;
            
            // Map the INT8 ranges to ASCII characters based on brightness
            if (pixelVal < -64)       Serial.print(" ");  // Darkest
            else if (pixelVal < 0)    Serial.print(".");
            else if (pixelVal < 64)   Serial.print("x");
            else                      Serial.print("#");  // Brightest
        }
        Serial.println();
    }
    Serial.println("--- END ARDUINO MATRIX DUMP ---");
}

// ===========================================================================
// runInference()
// ===========================================================================
static bool runInference(InferenceResult* result) {
    // digitalWrite(LED_BUILTIN, HIGH);
    // delay(kFlashMsec);
    const bool ok = fillInputTensor(gInput);
    // digitalWrite(LED_BUILTIN, LOW);

    if (!ok) {
        Serial.println("camera_read_failed");
        return false;
    }

    // --- CALL THE VISUALIZER HERE ---
    visualDebugPrint(gInput);
    // --------------------------------

    if (gInterpreter->Invoke() != kTfLiteOk) {
        Serial.println("invoke_failed");
        return false;
    }
    const int nOut   = static_cast<int>(gOutput->bytes / sizeof(int8_t));
    int       bestI  = 0;
    int8_t    bestQ  = gOutput->data.int8[0];
    for (int i = 1; i < nOut; ++i) {
        if (gOutput->data.int8[i] > bestQ) {
            bestQ = gOutput->data.int8[i];
            bestI = i;
        }
    }

    const float conf = (static_cast<float>(bestQ) - gOutput->params.zero_point)
                       * gOutput->params.scale;

    result->label      = (bestI < kNumLabels) ? kLabels[bestI] : "unknown";
    result->confidence = constrain(conf, 0.0f, 1.0f);
    return true;
}

// ===========================================================================
// sendResult()
// ===========================================================================
static void sendResult(const InferenceResult& r) {
    Serial.print(r.label);
    Serial.print(',');
    Serial.print(r.confidence, 4);
    Serial.print(',');
    Serial.println(millis() / 1000.0f, 3);
}

// ===========================================================================
// Entry Points
// ===========================================================================
void setup() {
    Serial.begin(kBaud);
    while (!Serial) { delay(10); }

    pinMode(LED_BUILTIN, OUTPUT);
    digitalWrite(LED_BUILTIN, LOW);
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