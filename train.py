import matplotlib.pyplot as plt
import tensorflow as tf
from model import enhancement_model
from losses import content_consistency_loss, feature_loss, reconstruction_loss, get_FeatureMap
from argument import args
from data_preparation import load_dataset,train_dataset,image_preprocessing
from tensorflow.keras.optimizers import Adam

model_optimizer = Adam(learning_rate=args.learning_rate)


path1,path2 = load_dataset(args.low_data,args.high_data)

low_light,reference = image_preprocessing(path1[500],path2[500])

low_light=tf.expand_dims(low_light,axis=0)
reference= tf.expand_dims(reference,axis=0)


with tf.GradientTape() as model_tape:
    reconstructed_image = enhancement_model([low_light,reference])

    content_low,luminance_low = get_FeatureMap(low_light)
    content_ref,luminance_ref = get_FeatureMap(reference)
    content_pred,luminance_pred = get_FeatureMap(reconstructed_image)

    loss1 = reconstruction_loss(reconstructed_image, reference)
    loss2 = feature_loss(content_low, content_pred,luminance_low, luminance_ref, luminance_pred )
    loss3 = content_consistency_loss(low_light,reconstructed_image)
    total_loss = loss1 + args.weight * loss2 + loss3

model_gradients = model_tape.gradient(total_loss,enhancement_model.trainable_variables)
model_optimizer.apply_gradients(zip(model_gradients, enhancement_model.trainable_variables))

