# TinyML Smart Pantry & Waste Reducer
Miles Chin (2577815), Daniel Tseng (2577912)

> Historical proposal document.
> Some assumptions here are superseded by the implemented workflow.
> Use README.md and src/mcu/README.md for current commands and deployment steps.

## 1. Problem Statement
### What problem are you solving?
Food waste is a significant global issue, with much of it occurring at the consumer level due
to simple forgetfulness. Items are placed in the fridge, pushed to the back, and forgotten until
they have already expired. Current solutions require manual entry into apps, which is
high-friction and often abandoned.
### Why is it relevant for TinyML?
A "Smart Fridge" doesn't need a power-hungry GPU or a constant cloud connection to
identify an apple or a carton of eggs. TinyML allows for an "always-on,
" low-power, and
private sensor (the Arduino Nano BLE Sense) to sit at the fridge entry point. It can process
images locally, identify the item, and log the entry date without sending private photos of
your home to the cloud.
## 2. Dataset(s)
To ensure this project is not "time-occupying,
" we will use a Hybrid Dataset approach:
- Food-101 (TensorFlow Datasets): We will use a subset of this public dataset
(focusing on 5–10 common fridge items like apples, bananas, milk, and eggs) to
pre-train our model.
- Food Freshness Dataset (Kaggle): To provide variety in how "fresh" items look
when first purchased.
- USDA FoodKeeper Data: This is a non-image dataset. We will use it as a "Lookup
Table" where the model’s output (e.g.,
"Label: Banana") is matched to a shelf-life
constant (e.g.,
"7 Days").
## 3. Modeling Approach
- Model Type: A Convolutional Neural Network (CNN). Given the memory
constraints of the Arduino Nano BLE Sense, we will likely use a scaled-down
MobileNetV2 or a custom 3-layer CNN.
●
- Input: Vision-based (RGB images from the ArduCam).
- Compression: * Post-Training Quantization: Converting the model from float32 to
int8 using TensorFlow Lite Micro to fit the 256KB RAM limit.
  - Input Resizing: Downsampling images to $96 \times 96$ pixels to reduce the
computational load.
## 4. Deployment Plan
- Hardware: Arduino Nano 33 BLE Sense with an OV7675 Camera module.
- Workflow:
  - 1. The camera triggers when it detects a change in light (fridge door
opening).
  - 2. The model classifies the item.
  - 3. The Arduino sends the item name via Serial/Bluetooth to a simple script that adds
the "Expiration Date" (Current Date + USDA Shelf Life).
- Evaluation: We will measure "Top-1 Accuracy" on a test set of 50 images taken in
the actual lighting conditions of your kitchen.
## 5. Challenges & Solutions
- Challenge (Memory): Image models are often too large for Arduinos.
  - Solution: We will limit the scope to only 10 classes of food to keep the final
layer small and use aggressive quantization.
- Challenge (Lighting): The inside of a fridge or a kitchen pantry has inconsistent
lighting.
  - Solution: We will use the Arduino’s built-in on-board LED as a flash to
normalize the lighting for every photo.
## 6. Estimated Timeline & Division of Work
Since you want to preserve time for other projects, this timeline is front-loaded.
Week Task Primary Responsibility
Week 1 Proposal & Dataset Cleaning (Subset Food-101) Daniel
Week 2 Model Training (Google Colab) & Quantization Daniel
Week 3 Arduino Camera Integration & Deployment Miles
Week 4 Final Testing & Documentation Miles