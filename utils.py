# . . GPUtil for memory management
import platform
import numpy as np
import torch
import os
# . . opencv
import cv2
import random
import argparse
import itertools
import matplotlib.pyplot as plt 

# . . do not import on MacOSX
if platform.system() is not 'Darwin':
    import GPUtil
# . . parse the command line parameters
def parse_args():

    parser = argparse.ArgumentParser(description='physics informed neural networks for 2D AWE solutions.')   
    # . . data directory
    default_dir = os.path.abspath(os.path.join(os.getcwd(), os.pardir)) +'/data/'
    parser.add_argument('--datapath',     type=str,   default=default_dir, help='data path')
    # . . hyperparameters
    parser.add_argument('--lr',           type=float, default=1e-3,  help='learning rate')
    parser.add_argument('--batch_size',   type=int,   default=64,    help='training batch size')
    parser.add_argument('--train_size',   type=float, default=0.8,   help='fraction of grid points to train')

    # . . training parameters
    parser.add_argument('--epochs',       type=int,   default=10,   help='number of epochs to train')

    # . . parameters for early stopping
    parser.add_argument('--patience',     type=int,   default=10,      help='number of epochs to wait for improvement')
    parser.add_argument('--min_delta',    type=float, default=0.0005,  help='min loss function reduction to consider as improvement')
    
    # . . parameters for data loaders
    parser.add_argument('--num_workers',    type=int,  default=8,      help='number of workers to use inf data loader')
    parser.add_argument('--pin_memory' ,    type=bool, default=False,  help='use pin memory for faster cpu-to-gpu transfer')

    parser.add_argument('--jprint',       type=int,   default=1,   help='print interval')

    # . . parse the arguments
    args = parser.parse_args()

    return args


def gpuinfo(msg=""):
    print("------------------")
    print(msg)
    print("------------------")
    GPUtil.showUtilization()
    print("------------------")

def devinfo(device):
    print("------------------")
    print("torch.device: ", device)
    print("------------------")
   
def batchinfo(loader, label=True):

    print("------------------")
    print("There are {} batches in the dataset".format(len(loader)))
    if label:
        for x, y in loader:
            print("For one iteration (batch), there are:")
            print("Data:    {}".format(x.shape))
            print("Label:   {}".format(y.shape))
            break   
    else:
        for x in loader:
            print("For one iteration (batch), there are:")
            print("Data:    {}".format(x.shape))
            break  
    print("------------------")

def conv_output_shape(h_w, kernel_size=1, stride=1, padding=0, dilation=1):
    """
    Utility function for computing output of convolutions
    takes a tuple of (h,w) and returns a tuple of (h,w)
    taken from: https://discuss.pytorch.org/t/utility-function-for-calculating-the-shape-of-a-conv-output/11173/5
    """
    
    if type(h_w) is not tuple:
        h_w = (h_w, h_w)
    
    if type(kernel_size) is not tuple:
        kernel_size = (kernel_size, kernel_size)
    
    if type(stride) is not tuple:
        stride = (stride, stride)
    
    if type(padding) is not tuple:
        padding = (padding, padding)
    
    h = (h_w[0] + (2 * padding[0]) - (dilation * (kernel_size[0] - 1)) - 1)// stride[0] + 1
    w = (h_w[1] + (2 * padding[1]) - (dilation * (kernel_size[1] - 1)) - 1)// stride[1] + 1
    
    return h, w


# . . forward passes a given (VGG) network and returns outputs
def forward_pass(model, loader, num_data, num_features, device='cuda'):
    X = np.zeros((num_data, num_features))
    y = np.zeros((num_data, 1))

    sample = 0
    with torch.no_grad():
        for inputs, targets in loader:
            # . . move data to device
            inputs = inputs.to(device)

            # . . forward pass
            outputs = model(inputs)

            # . . get the size of the current batch
            bsize = len(outputs)

            # . . assign the output to new training data
            X[sample : sample + bsize] = outputs.cpu().detach().numpy()
            y[sample : sample + bsize] = targets.view(-1, 1).numpy()

            # . . increase the number of samples processed
            sample += bsize

    return X, y