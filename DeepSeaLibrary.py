# -*- coding: utf-8 -*-
"""
Created on Thu May 29 18:44:41 2025

@author: spbki
"""

import os
import math
import random
import pathlib
import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf
import matplotlib.pyplot as plt
from collections import Counter
from emoji import emojize as emo
from matplotlib.colors import Normalize
from tensorflow.keras.layers import Layer
from tensorflow.keras import layers, models
from sklearn.metrics import confusion_matrix
from scipy.spatial.distance import squareform
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.callbacks import ReduceLROnPlateau
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster

Augmentation = tf.keras.Sequential([
    layers.RandomFlip("horizontal"), #lustrzane odbicie
    layers.RandomRotation(0.5),  # losowy obrót nawet o 180 stopni
    layers.RandomZoom(0.1), # losowy zoom +/- 10%
    layers.RandomTranslation(0.1, 0.1), #losowe przesunięcie
    layers.RandomContrast(0.1), #losowa zmiana jasności obrazka
])

def PlantktonPlot(image, label):
    plt.imshow(image, cmap='binary')
    plt.title(label)
    #plt.show()

def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def image_example(image_string, label):
    feature = {
        'image': _bytes_feature(image_string),
        'label': _int64_feature(label),
    }
    return tf.train.Example(features=tf.train.Features(feature=feature))

def create_tfrecord(image_paths, labels, tfrecord_filename):
    with tf.io.TFRecordWriter(tfrecord_filename) as writer:
        for img_path, label in zip(image_paths, labels):
            with open(img_path, 'rb') as f:
                img_bytes = f.read()
            tf_example = image_example(img_bytes, label)
            writer.write(tf_example.SerializeToString())

def parse_tfrecord(example_proto):
    feature_description = {
        'image': tf.io.FixedLenFeature([], tf.string),
        'label': tf.io.FixedLenFeature([], tf.int64),
    }
    parsed = tf.io.parse_single_example(example_proto, feature_description)
    
    # Decode the JPEG image
    image = tf.io.decode_jpeg(parsed['image'], channels=1)  # grayscale images
    image = tf.image.resize(image, [128, 128])
    image = tf.cast(image, tf.float32) / 255.0  # normalize to [0,1]
    
    label = parsed['label']
    return image, label

def load_dataset(tfrecord_path, batch_size=32, training=False):
    dataset = tf.data.TFRecordDataset(tfrecord_path)
    dataset = dataset.map(parse_tfrecord, num_parallel_calls=tf.data.AUTOTUNE)
    
    if training:
        # Apply augmentation only for training set
        dataset = dataset.map(lambda x, y: (Augmentation(x, training=True), y), num_parallel_calls=tf.data.AUTOTUNE)
        dataset = dataset.shuffle(1000)
        
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)
    return dataset


def Model(p, c, d, num_classes):
    layer_list = [
        layers.Input(shape=(128, 128, 1)),

        layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Dropout(p * 0.1),

        layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Dropout(p * 0.1),
    ]

    if c:
        layer_list += [
            layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.MaxPooling2D(pool_size=(2, 2)),
            layers.Dropout(p * 0.1),
        ]

    layer_list += [
        layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Dropout(p * 0.1),

        layers.Flatten(),
        layers.Dense(256, activation='relu'),
        layers.Dropout(p * 0.4),

        layers.Dense(128, activation='relu'),
        layers.Dropout(p * 0.4),
    ]

    if d:
        layer_list += [
            layers.Dense(128, activation='relu'),
            layers.Dropout(p * 0.4),
        ]

    layer_list += [
        layers.Dense(num_classes, activation='softmax')
    ]

    return models.Sequential(layer_list)

def Metryka(class_weights):
    class_weights_tensor = tf.constant(class_weights, dtype=tf.float32)
    
    def loss_fn(y_true, y_pred):
        y_true = tf.cast(y_true, tf.int32)
        weights = tf.gather(class_weights_tensor, y_true)
        loss = tf.keras.losses.sparse_categorical_crossentropy(y_true, y_pred)
        return loss * weights
    return loss_fn