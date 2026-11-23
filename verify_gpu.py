import tensorflow as tf
import sys

print("=" * 60)
print("TensorFlow GPU Verification Script")
print("=" * 60)

# TensorFlow version
print(f"\nTensorFlow Version: {tf.__version__}")

# Check if GPU is available
print(f"\nGPU Available: {tf.config.list_physical_devices('GPU')}")

# List all physical devices
print("\nAll Physical Devices:")
for device in tf.config.list_physical_devices():
    print(f"  - {device}")

# GPU details
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    print(f"\nNumber of GPUs: {len(gpus)}")
    for i, gpu in enumerate(gpus):
        print(f"\nGPU {i}: {gpu}")
        try:
            # Get GPU details
            gpu_details = tf.config.experimental.get_device_details(gpu)
            print(f"  Details: {gpu_details}")
        except Exception as e:
            print(f"  Could not get details: {e}")
    
    # Test GPU computation
    print("\n" + "=" * 60)
    print("Testing GPU Computation...")
    print("=" * 60)
    
    try:
        with tf.device('/GPU:0'):
            # Create random matrices
            a = tf.random.normal([1000, 1000])
            b = tf.random.normal([1000, 1000])
            
            # Perform matrix multiplication
            c = tf.matmul(a, b)
            
        print("[SUCCESS] GPU computation successful!")
        print(f"  Result shape: {c.shape}")
    except Exception as e:
        print(f"[FAILED] GPU computation failed: {e}")
else:
    print("\n" + "=" * 60)
    print("WARNING: No GPU detected!")
    print("=" * 60)
    print("\nTensorFlow will run on CPU only.")
    print("\nTo enable GPU support, ensure you have:")
    print("  1. NVIDIA GPU with CUDA Compute Capability 3.5 or higher")
    print("  2. CUDA Toolkit 12.x installed")
    print("  3. cuDNN 8.9 installed")
    print("\nDownload from:")
    print("  - CUDA: https://developer.nvidia.com/cuda-downloads")
    print("  - cuDNN: https://developer.nvidia.com/cudnn")

print("\n" + "=" * 60)
print("Verification Complete")
print("=" * 60)
