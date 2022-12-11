import matplotlib.pyplot as plt
import tensorflow as tf
from model import enhancement_model
from losses import content_consistency_loss, feature_loss, reconstruction_loss, get_FeatureMap
from argument import args
from data_preparation import load_dataset,train_dataset,image_preprocessing
from tensorflow.keras.optimizers import Adam
from tqdm import tqdm
import time




model_optimizer = Adam(learning_rate=args.learning_rate)
path1,path2 = load_dataset(args.low_data,args.high_data)

train_dataset = train_dataset(path1,path2)


def apply_gradient(low_light,reference):
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

    return total_loss

def train_data_for_one_epoch():
    losses = []
    pbar = tqdm(total=len(list(enumerate(train_dataset))), position=0, leave=True,bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} ')

    for step ,(low_light,reference) in enumerate(train_dataset):
        total_loss = apply_gradient(low_light,reference)
        losses.append(total_loss.numpy())
        pbar.set_description(f"Training loss for step_num {step} ,total_loss:{total_loss:0.4f}")
        pbar.update()

    return losses

def plot_history(losses,num_epoch):
    x=list(range(num_epoch))
    plt.figure(figsize=(10,10))
    plt.plot(x, losses,label='loss',color='green',marker='o')
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.savefig('loss.png')
    
    
def evaluation(model_path,epoch):
    path1, path2 = load_dataset(low_data, high_data)
    model = load_model(model_path)
    numper = np.random.randint(0, 1000,size=2)

    low_light ,reference = image_preprocessing(path1[numper[0]], path2[numper[1]])
    low_light = tf.expand_dims(low_light, axis=0)
    reference  = tf.expand_dims(reference , axis=0)
    reconstructed_image  = model([low_light, reference])

    low_light=(tf.squeeze(low_light,axis=0))
    reference=(tf.squeeze(reference,axis=0))
    reconstructed_image=(tf.squeeze(reconstructed_image,axis=0))

    plt.figure(figsize=(10,10))
    plt.subplot(1,3,1)
    plt.imshow(low_light)
    plt.xlabel("low_light")
    plt.subplot(1, 3, 2)
    plt.imshow(reference)
    plt.xlabel("reference")
    plt.subplot(1, 3, 3)
    plt.imshow(reconstructed_image)
    plt.xlabel("reconstructed_image")
    plt.savefig(f"/content/model_weights/epoch_{epoch}/image_{epoch}.png")
    
    
def training_fit(epochs):
    epochs_losses = []
    for epoch in range(epochs):
        print(f"Start of epoch number : {epoch}")
        start_time = time.time()
        losses = train_data_for_one_epoch()
        elapsed = time.time() - start_time
        minutes = int(elapsed // 60)
        seconds = int(elapsed % 60)
        epochs_losses.append(np.mean(losses))
        print(f'Epoch {epoch}: total_loss: {np.mean(losses):0.3f}  time:({minutes} min {seconds} sec)')
        if epoch % 10 == 0 :
            mkdir(f"/content/model_weights/epoch_{epoch}")
            enhancement_model.save(f"/content/model_weights/epoch_{epoch}/enhancement_model_{epoch}.h5")
            evaluation(model_path=f"/content/model_weights/epoch_{epoch}/enhancement_model_{epoch}.h5",epoch=epoch)

    plot_history(epochs_losses, epochs)


