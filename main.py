# . . import libraries
import sys,os
from pathlib import Path
# . . pytorch modules
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils import data
from torch.utils.data import DataLoader
import torchvision
from torchvision import datasets, transforms, models

# . . numpy
import numpy as np
# . . scikit-learn
from sklearn.preprocessing import StandardScaler
# . . matplotlib
import matplotlib.pyplot as plt
import matplotlib.image as npimg
# . .  set this to be able to see the figure axis labels in a dark theme
from matplotlib import style
#style.use('dark_background')
# . . to see the available options
# print(plt.style.available)
from glob import glob
import imageio
from torchsummary import summary

# . . import libraries by tugrulkonuk
import utils
from utils import parse_args, forward_pass
from model import *
from trainer import Trainer
from callbacks import ReturnBestModel, EarlyStopping

# . . parse the command-line arguments
args = parse_args()

# . . set the device
if torch.cuda.is_available():  
    device = torch.device("cuda")  
else:  
    device = torch.device("cpu")      

# . . set the default precision
dtype = torch.float32

# . . use cudnn backend for performance
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.enabled = True

# . . parameters
# . . user-defined
num_epochs    = args.epochs
batch_size    = args.batch_size
learning_rate = args.lr
train_size    = args.train_size
min_delta     = args.min_delta
patience      = args.patience 
num_workers   = args.num_workers
pin_memory    = args.pin_memory
jprint        = args.jprint
# . . computed
test_size     = 1.0 - train_size


# . . import the dataset
# . . standard transforms for ImageNet:
# . . normalize with the mean and standard deviation: image = (image - mean) / std
train_transform = transforms.Compose([
                  transforms.RandomResizedCrop(size=256, scale=(0.8, 1.0)),
                  transforms.RandomRotation(degrees=15),
                  transforms.ColorJitter(),
                  transforms.CenterCrop(size=224),
                  transforms.RandomHorizontalFlip(),
                  transforms.ToTensor(),
                  transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

valid_transform = transforms.Compose([
                  transforms.Resize(size=256),
                  transforms.CenterCrop(size=224),
                  transforms.ToTensor(),
                  transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])


# . . use the ImageFolder property of torchvision to create dataset objects
# . . the train set
train_dataset = datasets.ImageFolder(
    root='data/train',
    transform=train_transform)

# . . the validation set
valid_dataset = datasets.ImageFolder(
    root='data/validation',
    transform=valid_transform)

# . . the test set: same transform as the validation dataset
test_dataset = datasets.ImageFolder(
    root='data/test',
    transform=valid_transform)  

# . . data loaders
# . . the training loader: shuffle
trainloader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True, 
                         num_workers=num_workers, pin_memory=pin_memory)

# . . the validation loader: no shuffle
validloader = DataLoader(dataset=valid_dataset, batch_size=batch_size, shuffle=False,
                         num_workers=num_workers, pin_memory=pin_memory)

# . . the test loader: no shuffle
testloader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False,
                         num_workers=num_workers, pin_memory=pin_memory)     


# . . define the pretrained model to be used for the transfer learning
vgg = models.vgg16(pretrained=True)

# . . freeze the VGG weights
for param in vgg.parameters():
    param.requires_grad = False

# . . create the vgg features class
vggf = VGGFeatures(vgg)

# . . forward pass the data to the VGG network
# . . the idea is that instead of running multiple (expensive) forward passes 
# . . of the vgg network, we just do it once and use the output as the input 
# . . for the logistic regression network
# . . number of training, validation, and test images
num_train = len(train_dataset)
num_valid = len(valid_dataset)
num_test  = len(test_dataset)

# . .  compute the output shape of the VGG network: run on a single image (array)
dim_vgg = vggf(torch.rand(1, 3, 224, 224)).shape[1]

# . . send VGG network to device
vggf.to(device)

# . . pass training, validation, and test datasets trough the pretrained VGG network
X_train, y_train = forward_pass(vggf, trainloader, num_train, dim_vgg, device)
X_valid, y_valid = forward_pass(vggf, validloader, num_valid, dim_vgg, device)
X_test , y_test  = forward_pass(vggf, testloader , num_test , dim_vgg, device)

# . . scale the data for the logistic regression
# . . output VGG values are high because of the ReLU activation

scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_valid = scaler.fit_transform(X_valid)
X_test  = scaler.fit_transform(X_test)

# . . create new datasets using VGG outputs
# . . overwrite the precious ones
training_dataset = data.TensorDataset(
    torch.from_numpy(X_train.astype(np.float32)),
    torch.from_numpy(y_train.astype(np.float32)))

valid_dataset = data.TensorDataset(
    torch.from_numpy(X_valid.astype(np.float32)),
    torch.from_numpy(y_valid.astype(np.float32)))    

test_dataset = data.TensorDataset(
    torch.from_numpy(X_test.astype(np.float32)),
    torch.from_numpy(y_test.astype(np.float32)))   

# . . new data loaders for VGG outputs
# . . the training loader: shuffle
trainloader = DataLoader(dataset=training_dataset, batch_size=batch_size, shuffle=True, 
                         num_workers=num_workers, pin_memory=pin_memory)

# . . the validation loader: no shuffle
validloader = DataLoader(dataset=valid_dataset, batch_size=batch_size, shuffle=False,
                         num_workers=num_workers, pin_memory=pin_memory)

# . . the test loader: no shuffle
testloader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False,
                         num_workers=num_workers, pin_memory=pin_memory)          


# . . the logistic regression model in pytorch
model = nn.Linear(dim_vgg, 1)

# . . send the model to device
model.to(device)

# . . create the trainer
trainer = Trainer(model, device)

# . . compile the trainer
# . . define the loss
criterion = nn.BCEWithLogitsLoss()

# . . define the optimizer
optimparams = {'lr':learning_rate
              }

# . . define the callbacks
cb=[ReturnBestModel(), EarlyStopping(min_delta=min_delta, patience=patience)]

trainer.compile(optimizer='adam', criterion=criterion, callbacks=cb, jprint=jprint, **optimparams)

# . . the learning-rate scheduler
schedulerparams = {'factor':0.5,
                   'patience':50,
                   'threshold':1e-5,
                   'cooldown':5,
                   'min_lr':1e-5,                
                   'verbose':True               
                  }
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(trainer.optimizer, **schedulerparams)

# . . train the network
train_loss, valid_loss = trainer.fit(trainloader, validloader, scheduler=None, num_epochs=num_epochs)

# . . plot the training and validation losses
plt.plot(train_loss)
plt.plot(valid_loss)
plt.legend(['train_loss', 'valid_loss'])

# . . evaluate the accuracy of the trained network
training_accuracy, test_accuracy = trainer.evaluate(trainloader, testloader)

#
# . . save the model
torch.save(trainer.model.state_dict(), 'models/final_model.pt')
