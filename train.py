import numpy as np
import shutil
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
from models.model import ColorizeClassifier, ResNextColorizeClassifier
from color_utils import ColorLoss, lab2pil, ab2bin, bin2ab, idx_to_lab
from dataset import (torch_softmax2image, hd52numpy, hd52numpy, img2hdf5, 
     ColorizeHD5Dataset,CategoricalColorizeDataSet )

import wandb
wandb.init()


train_dir = '/home/ec2-user/data/train/aggregate/'
test_dir = '/home/ec2-user/data/test/lukas'

val_bs = 12
ds_train = CategoricalColorizeDataSet(train_dir,  transform=transforms.Compose([
                            transforms.RandomRotation(15, expand=False),
                            transforms.RandomResizedCrop(256),
                            transforms.RandomHorizontalFlip(),
                           ]))
train_loader = data.DataLoader(ds_train,batch_size=12, shuffle=True, num_workers=8)
ds_test = CategoricalColorizeDataSet(test_dir,  transform=transforms.Compose([
                            transforms.Resize(299),
                            transforms.CenterCrop(256),
                           ]))
test_loader = data.DataLoader(ds_test,batch_size=val_bs, shuffle=False)

color_weights = np.load('color_weights.npy')
y_soft_encode = torch.FloatTensor(np.load('soft_encoding.npy')).cuda()


def log_wandb_images(inputs, labels, output, idx):
    input_images = []
    output_images = []
    reference_images = []
    inps_cpu = inputs.cpu().numpy()
    for i in range(inps_cpu.shape[0]):
        x = inps_cpu[i,0,:,:]
        x_image = Image.fromarray(np.uint8(np.round((x + 0.5)*255)))
        target_image = ds_test.get_pil(i + idx * val_bs)
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
    idx = np.random.choice(range(len(test_loader)))
    for i, (inputs, labels) in enumerate(test_loader):
        inputs, labels = inputs.to('cuda'), labels.to('cuda')
        with torch.no_grad():
            a_out = model(inputs)
        # we only log stuff to wandb on first iteration
        if idx == i:
            log_wandb_images(inputs, labels, a_out, idx)
        _, yh = torch.max(a_out, 1)
        soft_output = y_soft_encode[labels].permute(0,3,1,2)
        
        loss += criterion(a_out, soft_output).item() * labels.shape[0]
        num += labels.shape[0]
        

    print(f"Validation Loss: {loss/num}")
    return loss/num

def train(model, optimizer, criterion, epochs=10, lrs=None, report_every=100):
    val_loss, t_loss = [], []
    best_loss_so_far = float('inf')
    
    rloss, uloss = 0.0, 0.0
    rcount = 0
    
    report_count = 0
    
    for e in range(epochs):
        print(f"epoch: {e}")
        
        if lrs is not None: lrs.step()
        pbar = tqdm(train_loader)
            
        for inputs, labels in pbar:
            pbar.set_description(f"loss: {uloss}")
            model.train()
            inputs, labels = inputs.to('cuda'), labels.to('cuda') 

            optimizer.zero_grad()
            logits = model(inputs)
            soft_output = y_soft_encode[labels].permute(0,3,1,2)
            loss = criterion(logits, soft_output)
            loss.backward()
            if rcount < 1000000:
                rcount += 1
            rloss = 0.9*rloss + 0.1*loss.item()
            uloss = rloss / (1 - 0.9**rcount)
            optimizer.step()
            
            report_count += 1
            if report_count % report_every == 0:
                vloss = get_validation_error(model)
                val_loss.append(vloss)
                t_loss.append(uloss)
                # wandb log loss
                wandb.log({'epoch': report_count, 'loss': uloss, 'val_loss': vloss})
                if vloss < best_loss_so_far:
                    best_loss_so_far = vloss
                    torch.save({
                        'epoch': e + 1,
                        'state_dict': model.state_dict(),
                        'optimizer': optimizer.state_dict()
                    }, "trained_models/best_so_far.pth")
        
        torch.save({
            'epoch': e + 1,
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict()
        }, f"epoch_{e}.pth")
        
            
        
    return pd.DataFrame({'loss': val_loss, 't_loss': t_loss})

def save_files_to_wandb():
    os.mkdir(os.path.join(wandb.run.dir, 'trained_models'))
    files_to_copy = ['trained_models/best_so_far.pth', 'color_utils.py', 'run.py', 'models', 'README.md']
    for fname in files_to_copy:
        dest_path = os.path.join(wandb.run.dir, fname)
        if os.path.isdir(fname):
            shutil.copytree(fname, dest_path)
        else:
            shutil.copy2(fname, dest_path)          
        

if __name__ == '__main__':
    model = ResNextColorizeClassifier(training=None) #ColorizeClassifier(feature_cascade=(512, 256, 64, 64))
    state = torch.load('trained_models/resnext-frozen-imagenet.pth')
    model.load_state_dict(state['state_dict'])
    
    #model.freeze_ft()
    model.cuda()
    optimizer = optim.Adam(model.parameters(), lr=0.0001)
    lrs = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)
    criterion = ColorLoss(torch.FloatTensor(color_weights).cuda())
    epochs = 8
    wandb.config.epochs = epochs
    wandb.config.batch_size = 16
    wandb.config.width = 256
    wandb.config.height = 256
    wandb.config.library = "PyTorch 0.4" 
    train(model, optimizer, criterion, epochs, lrs)
    save_files_to_wandb() 
