# -*- coding: utf-8 -*-
"""
converting loaded model to tflite 

@author: IURII
"""
import numpy as np

import argparse
import tensorflow.keras
from tensorflow.keras.models import Sequential, load_model, Model
import tensorflow as tf

import os

os.environ['CUDA_VISIBLE_DEVICES']= "-1"
    
# tf.config.threading.set_intra_op_parallelism_threads(0)
# tf.config.threading.set_inter_op_parallelism_threads(0)

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        # Currently, memory growth needs to be the same across GPUs
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    except RuntimeError as e:
        # Memory growth must be set before GPUs have been initialized
        print(e)


def main():
    
    print("tf version ", tf.__version__)
    
    #Loading Model
    model = load_model(os.path.join('./fr_model.h5'))
    
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    tflite_save = converter.convert()
    open("fr_model.tflite", "wb").write(tflite_save)


if __name__ == '__main__':
    main()
