import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
import torchvision.models as models
import torch.optim as optim
from itertools import chain

from tqdm import tnrange, tqdm

import os
from PIL import Image, ImageOps
from skimage.color import rgb2lab, lab2rgb, rgb2gray, xyz2lab
from tqdm import tqdm
import h5py
import torch.utils.data as data
from model import ColorizeClassifier, ResNextColorizeClassifier
from color_utils import ColorLoss, lab2pil, ab2bin, bin2ab, idx_to_lab
from dataset import (torch_softmax2image, hd52numpy, hd52numpy, img2hdf5, 
     ColorizeHD5Dataset,CategoricalColorizeDataSet )

import wandb
wandb.init()


train_dir = '/home/ec2-user/data/train/'#lukas'
test_dir = '/home/ec2-user/data/test/'#lukas'


ds_train = CategoricalColorizeDataSet(train_dir,  transform=transforms.Compose([
                            transforms.RandomRotation(15, expand=False),
                            transforms.RandomResizedCrop(256),
                            transforms.RandomHorizontalFlip(),
                           ]))
train_loader = data.DataLoader(ds_train,batch_size=16, shuffle=True, num_workers=6)
ds_test = CategoricalColorizeDataSet(test_dir,  transform=transforms.Compose([
                            transforms.Resize(299),
                            transforms.CenterCrop(256),
                           ]))
test_loader = data.DataLoader(ds_test,batch_size=16, shuffle=False)

color_weights = np.load('color_weights.npy')
y_soft_encode = torch.FloatTensor(np.load('soft_encoding.npy')).cuda()


def log_wandb_images(inputs, labels, output):
    input_images = []
    output_images = []
    reference_images = []
    
    inps_cpu = inputs.cpu().numpy()
    for i in range(inps_cpu.shape[0]):
        x = inps_cpu[i,0,:,:]
        x_image = Image.fromarray(np.uint8(np.round((x + 0.5)*255)))
        target_image = ds_test.get_pil(i)
        recolored = torch_softmax2image(inputs, output, i, 0.2)
        recolored_image = Image.fromarray(np.uint8(np.round(recolored*255)), 'RGB')
        input_images.append(wandb.Image(x_image, grouping=3))
        output_images.append(wandb.Image(recolored_image))
        reference_images.append(wandb.Image(target_image))
    payload = list(chain.from_iterable(zip(input_images, output_images, reference_images)))
    wandb.log({"examples": payload}, commit=False)
    
    
def get_validation_error(model):
    model.eval()
    loss = 0.0
    num = 0
    for inputs, labels in test_loader:
        inputs, labels = inputs.to('cuda'), labels.to('cuda')
        with torch.no_grad():
            a_out = model(inputs)
        # we only log stuff to wandb on first iteration
        if num == 0:
            log_wandb_images(inputs, labels, a_out)
        _, yh = torch.max(a_out, 1)
        soft_output = y_soft_encode[labels].permute(0,3,1,2)
        
        loss += criterion(a_out, soft_output).item() * labels.shape[0]
        num += labels.shape[0]
        

    print(f"Validation Loss: {loss/num}")
    return loss/num

def train(model, optimizer, criterion, epochs=10, lrs=None):
    val_loss, t_loss = [], []
    best_loss_so_far = 10000000.0
    for e in range(epochs):
        print(f"epoch: {e}")
        running_loss = 0.0
        running_num = 1e-8
        if lrs is not None: lrs.step()
        pbar = tqdm(train_loader)
            
        for inputs, labels in pbar:
            pbar.set_description(f"loss: {running_loss/running_num}")
            model.train()
            inputs, labels = inputs.to('cuda'), labels.to('cuda') 

            optimizer.zero_grad()
            logits = model(inputs)
            soft_output = y_soft_encode[labels].permute(0,3,1,2)
            loss = criterion(logits, soft_output)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * inputs.shape[0]
            running_num += inputs.shape[0]
        tloss = running_loss/running_num
        vloss = get_validation_error(model)
        val_loss.append(vloss)
        t_loss.append(tloss)
        torch.save({
            'epoch': e + 1,
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict()
        }, f"epoch_{e}.pth")
        
        # wandb log loss
        wandb.log({'epoch': e + 1, 'loss': tloss, 'val_loss': vloss})
        
        if vloss < best_loss_so_far:
            best_loss_so_far = vloss
            torch.save({
                'epoch': e + 1,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict()
            }, "trained_models/best_so_far.pth")    
        
    return pd.DataFrame({'loss': val_loss, 't_loss': t_loss})

def save_files_to_wandb():
    files_to_copy = ['best_so_far.pth', 'color_utils.py', 'run.py']
    for fname in files_to_copy:
        filepath = os.path.join(wandb.run.dir, 'fname')
        

if __name__ == '__main__':
    model = ResNextColorizeClassifier() #ColorizeClassifier(feature_cascade=(512, 256, 64, 64))
    model.freeze_ft()
    model.cuda()
    
    optimizer = optim.Adam(model.parameters(), lr=0.002)
    lrs = optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)
    criterion = ColorLoss(torch.FloatTensor(color_weights).cuda())
    epochs = 10
    wandb.config.epochs = epochs
    wandb.config.batch_size = 16
    wandb.config.width = 256
    wandb.config.height = 256
    wandb.config.library = "PyTorch 0.4" 
    train(model, optimizer, criterion, epochs, lrs)
    save_files_to_wandb() 
