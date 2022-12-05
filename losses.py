import numpy as np
import tensorflow as tf
from model import enhancement_model,encoder,global_average_block
import keras.backend as K


def get_FeatureMap(image):
    out = encoder(image)
    out = global_average_block(out)
    content_component, luminance_component = tf.split(out, [384, 128], -1)
    return content_component,luminance_component


def euclidean_distance(x,y):
    distance = K.sum(tf.square(x - y), axis=(1,2,3))
    distance = K.maximum(distance, K.epsilon())
    return distance

def reconstruction_loss(reconstructed_image, reference_image):
    # measure the distance between the prediction and ground truth
    Lr = K.abs(K.sum(reconstructed_image - reference_image, axis=(1, 2, 3)))
    Lr = K.mean(Lr)
    return Lr

def content_feature_loss(low_light_content, prediction_content):
    l2_loss = K.sum(K.square(prediction_content - low_light_content),axis=(1,2,3))
    return K.mean(l2_loss)

def luminance_feature_loss(low_light_luminance, reference_luminance, prediction_luminance) :
    post_dict = euclidean_distance(prediction_luminance - reference_luminance)
    neg_dict = euclidean_distance(prediction_luminance - low_light_luminance)
    basic_loss = K.maximum(post_dict-neg_dict+0.08 ,0.0)
    return K.mean(basic_loss)

def feature_loss(low_light_content, prediction_content,low_light_luminance, reference_luminance, prediction_luminance):
    loss_1 = content_feature_loss(low_light_content, prediction_content)
    loss_2 = luminance_feature_loss(low_light_luminance, reference_luminance, prediction_luminance)
    total_loss = loss_1+loss_2
    return total_loss

def findCosineSimilarity_s(low_light_image, prediction):
    s_low_light = tf.image.rgb_to_hsv(low_light_image)[..., 1]
    s_prediction = tf.image.rgb_to_hsv(prediction)[..., 1]

    s_low_light = tf.reshape(h_low_light, (-1, 128 * 128))
    s_prediction = tf.reshape(h_prediction, (-1, 128 * 128))

    a = tf.matmul(s_low_light,tf.transpose(s_prediction))
    arr2 = tf.linalg.diag_part(a)
    b = tf.reduce_sum(tf.multiply(s_low_light,s_low_light),axis=-1)
    c = tf.reduce_sum(tf.multiply(s_prediction ,s_prediction ), axis=-1)
    loss = 1 - (arr2 / (np.sqrt(b) * np.sqrt(c)))
    return K.mean(loss)

def findCosineSimilarity_h(low_light_image, prediction):
    h_low_light = tf.image.rgb_to_hsv(low_light_image)[..., 0]
    h_prediction = tf.image.rgb_to_hsv(prediction)[..., 0]

    h_low_light = tf.reshape(h_low_light, (-1, 128 * 128))
    h_prediction = tf.reshape(h_prediction, (-1, 128 * 128))

    a = tf.matmul(h_low_light,tf.transpose(h_prediction))
    arr2 = tf.linalg.diag_part(a)
    b = tf.reduce_sum(tf.multiply(h_low_light,h_low_light),axis=-1)
    c = tf.reduce_sum(tf.multiply(h_prediction ,h_prediction ), axis=-1)
    loss = 1 - (arr2 / (np.sqrt(b) * np.sqrt(c)))
    return K.mean(loss)

def content_consistency_loss(low_light_image, prediction):
    loss_1 = findCosineSimilarity_h(low_light_image,prediction)
    loss_2 = findCosineSimilarity_s(low_light_image,prediction)
    total_loss = loss_1 + loss_2
    return total_loss