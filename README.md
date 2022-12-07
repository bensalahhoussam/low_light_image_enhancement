# low_light_image_enhancement
This repository contains the code for low light image enhancement.
## Methodology

It's known that the image often suffer degradations due to the environment and equipment limitations, such as low contrast,so The goal of low_light image enhancement is to learn a mapping from an image to a normal-light version.

We can assume that the image can be decomposed into two feature components in the latent space, namely the content component and the luminance component,
In other words pairs of images low light image and reference image are encoded by an encoder E to generate feature vectors, it decomposed into (Ci,Li) and (Cj,Lj) ,next, Ci and Lj are concatenated to form a new feature vector ,after the reconstructed image are generated by a decoder G is the same in content as low light image and same in luminance as reference image.

![image](https://user-images.githubusercontent.com/112108580/205706246-455815f9-104d-41db-a88e-9c01e3648099.png)


## Overall Architecture
The model is designed to enhance a low-light image to corresponding normal-light versions.
It consists of an encoder E, a feature concatenation module and a decoder G.
The network employs down-sampling part of U-Net as the encoder E, followed by a global average pooling, which respectively encodes Ii and Ir as feature vectors Fi and Fr. Correspondingly, the decoder G is up-sampling part of U-Net to reconstruct the feature vector.

![image](https://user-images.githubusercontent.com/112108580/205706419-333fb383-f22f-4419-8875-c416f9f1827a.png)

## Dependencies
* Python 3
* Tensorflow 2.x
* Numpy
* Matplotlib
* TQDM
* OpenCV
## Data Preparation 
LOL dataset is involved in training. It consists of 1000 image pairs, where each pair contains a low-light image and its corresponding normal-light image. 

### Download 
You can also download the dataset from 
* Low link:https://drive.google.com/file/d/1jiO810sQgmkRDMxwRmSWImJOKh9gG3-p/view?usp=sharing
* High link:https://drive.google.com/file/d/1QjsfY7fODsM_RkJxMO_no9VGachuivhk/view?usp=sharing
## Training 
To train the model, modify the training parameters and path folder for the dataset in `argument.py`

Run the code for training: 
`python train.py`



