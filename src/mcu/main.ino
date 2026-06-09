#include <Arduino.h>

namespace {
constexpr unsigned long kBaudRate = 115200;

struct ClassificationResult {
  String label;
  float confidence;
};

bool doorOpenTriggered() {
  return false;
}

bool captureAndRunInference(ClassificationResult *result) {
  if (result == nullptr) {
    return false;
  }

  result->label = "unknown";
  result->confidence = 0.0f;
  return false;
}

void sendResult(const ClassificationResult &result) {
  Serial.print(result.label);
  Serial.print(',');
  Serial.print(result.confidence, 4);
  Serial.print(',');
  Serial.println(millis() / 1000.0f, 3);
}
}  // namespace

void setup() {
  Serial.begin(kBaudRate);
  while (!Serial) {
    delay(10);
  }

  Serial.println("smart_pantry_ready");
}

void loop() {
  if (!doorOpenTriggered()) {
    delay(50);
    return;
  }

  ClassificationResult result;
  if (captureAndRunInference(&result)) {
    sendResult(result);
  }

  delay(250);
}
