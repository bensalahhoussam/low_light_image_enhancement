import tensorflow as tf
from tensorflow.keras.layers import Conv2D, Dropout, BatchNormalization, Activation, MaxPool2D, Input, \
    GlobalAveragePooling2D, Concatenate, Conv2DTranspose
from tensorflow.keras.models import Model



def conv_block(x_input, num_filters):
    x = Conv2D(num_filters, kernel_size=3, padding="same", kernel_initializer="he_normal")(x_input)
    x = BatchNormalization()(x)
    x = Activation("elu")(x)
    x = Conv2D(num_filters, kernel_size=3, padding="same", kernel_initializer="he_normal")(x)
    x = BatchNormalization()(x)
    x = Activation("elu")(x)
    return x


def encoder_block(x_input, num_filters):
    x = conv_block(x_input, num_filters)
    p = MaxPool2D(pool_size=(2, 2))(x)
    output = Dropout(0.3)(p)
    return output


def encoder(inputs):
    output = encoder_block(inputs, num_filters=64)
    output = encoder_block(output, num_filters=128)
    output = encoder_block(output, num_filters=256)
    output = encoder_block(output, num_filters=512)
    return output


def global_average_block(inputs):
    global_average = GlobalAveragePooling2D(keepdims=True)(inputs)
    return global_average


def feature_concatenation_model(low_light, reference):
    global_average_low_light_img = global_average_block(low_light)
    global_average_reference_img = global_average_block(reference)
    content_component_low_light, luminance_component_low_light = tf.split(global_average_low_light_img, [384, 128], -1)
    content_component_ref, luminance_component_ref = tf.split(global_average_reference_img, [384, 128], -1)
    reconstructed_image = Concatenate(axis=-1)([content_component_low_light, luminance_component_ref])
    reconstructed_image = Conv2DTranspose(512, kernel_size=3, strides=2, padding="same", kernel_initializer="he_normal",
                                          use_bias=False)(reconstructed_image)
    reconstructed_image = Conv2DTranspose(512, kernel_size=3, strides=2, padding="same", kernel_initializer="he_normal",
                                          use_bias=False)(reconstructed_image)
    reconstructed_image = Conv2DTranspose(512, kernel_size=3, strides=2, padding="same", kernel_initializer="he_normal",
                                          use_bias=False)(reconstructed_image)

    return reconstructed_image


def decoder_block(input_x, num_filters):
    x = Conv2DTranspose(num_filters, (3, 3), strides=2, padding="same", kernel_initializer="he_normal",
                        use_bias=False)(input_x)
    x = Dropout(0.3)(x)
    x = conv_block(x, num_filters)
    return x


def decoder(inputs):
    c6 = decoder_block(inputs, num_filters=512)
    c7 = decoder_block(c6, num_filters=256)
    c8 = decoder_block(c7, num_filters=128)
    c9 = decoder_block(c8, num_filters=64)
    outputs = Conv2D(3, 1, kernel_initializer="he_normal", activation='softmax')(c9)
    return outputs


def model():
    low_light_image = Input(shape=(128, 128, 3))
    reference_image = Input(shape=(128, 128, 3))

    feature_map_low_light_image = encoder(low_light_image)
    feature_map_reference_image = encoder(reference_image)

    reconstructed_image = feature_concatenation_model(feature_map_low_light_image, feature_map_reference_image)

    output = decoder(reconstructed_image)

    image_enhancement_model = Model(inputs=[low_light_image, reference_image], outputs=output)

    return image_enhancement_model


enhancement_model = model()
print(enhancement_model.summary())
