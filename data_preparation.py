import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
import tensorflow as tf
import os



low_data = "D://BrighteningTrain/BrighteningTrain/low/"
high_data = "D://BrighteningTrain/BrighteningTrain/high/"


def load_dataset(low_data , high_data):
    path_1 = []
    path_2 = []
    low_light_image = [img for img in os.listdir(low_data)]

    for img in low_light_image:
        low_path = low_data + img
        path_1.append(low_path)

    high_light_image = [img for img in os.listdir(high_data)]

    for img in high_light_image:
        high_path = high_data + img
        path_2.append(high_path)
    return np.array(path_1), np.array(path_2)

def image_preprocessing(path1, path2):
    img1 = tf.io.read_file(path1)
    img1 = tf.image.decode_png(img1, channels=3)
    img1 = tf.image.resize(img1, (128, 128)) / 255.0
    img1 = tf.cast(img1, dtype=tf.float32)

    img2 = tf.io.read_file(path2)
    img2 = tf.image.decode_png(img2, channels=3)
    img2 = tf.image.resize(img2, (128, 128)) / 255.0
    img2 = tf.cast(img2, dtype=tf.float32)

    return img1, img2

def train_dataset(path1,path2):
    train_dataset = tf.data.Dataset.from_tensor_slices((path1,path2))
    train_dataset = train_dataset.map(image_preprocessing).shuffle(32).batch(8,drop_remainder=True).prefetch(4)
    return train_dataset



path1,path2 = load_dataset(low_data,high_data)

train = train_dataset(path1,path2)



