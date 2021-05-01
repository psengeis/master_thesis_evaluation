from tensorflow.python.client import device_lib
import tensorflow as tf
import math


def PlotGPUInfo():
    print(tf.config.list_physical_devices('GPU'))


def RestrictMaxGPUMemoryAllocation(maxGB):
    gpus = tf.config.list_physical_devices('GPU')

    if gpus:
        # Restrict TensorFlow to only allocate 6GB of memory on the first GPU
        try:
            tf.config.experimental.set_virtual_device_configuration(
                gpus[0],
                [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=math.floor(maxGB * 1024))])
            logical_gpus = tf.config.experimental.list_logical_devices('GPU')
            print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
        except RuntimeError as e:
            # Virtual devices must be set before GPUs have been initialized
            print(e)
