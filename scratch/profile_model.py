import tensorflow as tf
import numpy as np

model_path = "artifacts/micro_cnn_qat/qat_model.tflite"
interpreter = tf.lite.Interpreter(model_path=model_path)
interpreter.allocate_tensors()

print("Model loaded successfully.")
tensor_details = interpreter.get_tensor_details()

print(f"\nTotal tensors: {len(tensor_details)}")
print(f"{'Index':<5} | {'Name':<40} | {'Shape':<15} | {'Dtype':<10} | {'Bytes':<10} | {'BufferIdx':<10}")
print("-" * 95)

total_activation_bytes = 0
for detail in tensor_details:
    idx = detail['index']
    name = detail['name']
    shape = detail['shape']
    dtype = detail['dtype']
    
    num_elements = int(np.prod(shape)) if len(shape) > 0 else 1
    element_size = np.dtype(dtype).itemsize
    n_bytes = num_elements * element_size
    
    buffer_idx = detail.get('buffer', 0)
    is_weight = (buffer_idx != 0)
    
    print(f"{idx:<5} | {name:<40} | {str(shape):<15} | {str(dtype):<10} | {n_bytes:<10} | {str(buffer_idx):<10}")
    
    if not is_weight:
        total_activation_bytes += n_bytes

print("-" * 95)
print(f"Total activation tensor bytes (sum of non-static tensors): {total_activation_bytes} bytes (~{total_activation_bytes/1024:.2f} KB)")
