import argparse


low_data = "D://BrighteningTrain/BrighteningTrain/low/"
high_data = "D://BrighteningTrain/BrighteningTrain/high/"
batch_size = 8
learning_rate = 1e-4
epochs = 1000
weight = 2

parser = argparse.ArgumentParser()
parser.add_argument('--low_data', type = str,default = low_data, help = 'path of low dataset')
parser.add_argument('--high_data', type = str,default = high_data, help = 'path of high dataset')
parser.add_argument('--batch_size', type = int,default = batch_size, help = 'batch size ')
parser.add_argument('--weight', type = int,default = weight, help = 'weight')
parser.add_argument('--learning_rate', type = float,default = learning_rate, help = 'learning rate ')
parser.add_argument('--epochs', type = int,default = epochs, help = 'total epoch number ')
args = parser.parse_args()