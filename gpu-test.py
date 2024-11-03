import tensorflow as tf

# Check TensorFlow version
print("TensorFlow version:", tf.__version__)

# Check available devices
print("Available devices:", tf.config.list_physical_devices())

# Check for a GPU
if tf.test.gpu_device_name():
    print('Default GPU Device: {}'.format(tf.test.gpu_device_name()))
else:
    print("Please install GPU version of TF")