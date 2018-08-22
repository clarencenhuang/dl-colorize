import torch.utils.data as data
from torchvision import transforms
from PIL import Image, ImageOps
from skimage.color import rgb2lab, lab2rgb, rgb2gray, xyz2lab
from tqdm import tqdm
import os
import numpy as np
import h5py
import torch
from torchvision.datasets.folder import has_file_allowed_extension, default_loader, IMG_EXTENSIONS
from color_utils import (torch_softmax2image, lab2pil, ab2bin, pil2lab,
    bin2ab, idx_to_lab)

def img2hdf5(input_dir, output_file, sz=64):
    files = []
    for rootdir, _, fnames in sorted(os.walk(input_dir)):
        for fname in sorted(fnames):
            if fname[0] == '.': continue
            files.append(os.path.join(rootdir, fname))
    num_files = len(files)
    t_x, t_y = np.zeros((num_files, sz, sz, 1)), np.zeros((num_files, sz, sz, 2))
    for i, fpath in enumerate(tqdm(files)):
        img = Image.open(fpath)
        img = ImageOps.fit(img, (sz,sz), 0, 0, (0.5, 0.5))
        rgb_hwc = np.asarray(img)
        l_hw, ab_hwc = pil2lab(rgb_hwc)
        t_x[i, :, :, 0] = l_hw
        t_y[i, :, :, :] = ab_hwc
    f = h5py.File(output_file, "w")
    f.create_dataset("inputs", data=t_x)
    f.create_dataset("labels", data=t_y)
    f.close()
    
    
def hd52numpy(input_file):
    f = h5py.File(input_file, 'r')
    inputs = np.array(f['inputs'])
    labels = np.array(f['labels'])
    f.close()
    return inputs, labels


class ColorizeHD5Dataset(data.Dataset):
    
    def __init__(self, hd5file, classification=True):
        super(ColorizeHD5Dataset, self).__init__()
        self.inputs, self.labels = hd52numpy(hd5file)
        self.classification = classification
        
    def __getitem__(self, index):
        l, ab = self.inputs[index,:], self.labels[index, :]
        if self.classification:
            hist = ab2bin(ab)
            return torch.FloatTensor(l.transpose(2,0,1)), torch.LongTensor(hist)
        else:
            return torch.FloatTensor(l.transpose(2,0,1)), torch.FloatTensor(ab.transpose(2,0,1))

    def __len__(self):
        return self.inputs.shape[0]
    
class ColorizeDataSet(data.Dataset):
    
    def __init__(self, root, loader=default_loader, 
                 extensions=IMG_EXTENSIONS, transform=None, target_transform=None,
                normalize_lab=None):
        self.images = []
        self.transform = transform
        self.target_transform = target_transform
        self.loader = loader
        self.normalize_lab = normalize_lab
        root = os.path.expanduser(root)
        for rootdir, _, fnames in sorted(os.walk(root)):
            for fname in sorted(fnames):
                if fname[0] == '.':
                    continue
                if has_file_allowed_extension(fname, extensions):
                    path = os.path.join(rootdir, fname)
                    self.images.append(path)
        if len(self.images) == 0:
            raise(RuntimeError("Found 0 files in folder of: " + root + "\n"
                               "Supported extensions are: " + ",".join(extensions)))
            
        if normalize_lab is not None:
            m, std = torch.Tensor(normalize_lab[0]).float(), torch.Tensor(normalize_lab[1]).float()
            self.normalize_lab = (m, std)

    
    def get_pil(self, index):
        path = self.images[index]
        sample = self.loader(path)
        if self.transform is not None:
            sample = self.transform(sample)
        return sample
    
        
    def __getitem__(self, index):
        sample = self.get_pil(index)
        rgb_hwc = np.asarray(sample)
        l_hw, ab_hwc = pil2lab(rgb_hwc)
        l_hw = l_hw[:,:,None]
        return torch.FloatTensor(l_hw.transpose(2,0,1)), torch.FloatTensor(ab_hwc.transpose(2,0,1))
    
    def __len__(self):
        return len(self.images)
    
    
class CategoricalColorizeDataSet(ColorizeDataSet):
    
    
    def __getitem__(self, index):
        sample = self.get_pil(index)
        rgb_hwc = np.asarray(sample)
        l_hw, ab_hwc = pil2lab(rgb_hwc)
        l_hw = l_hw[:,:,None]
        hist = ab2bin(ab_hwc)
        return torch.FloatTensor(l_hw.transpose(2,0,1)), torch.LongTensor(hist)
    
    def to_image(self, xb, yb, idx):
        x = (xb.cpu().numpy() * 100.0)[idx,0,:]
        y = (yb.cpu().numpy())[idx,:]
        idx_ab = y.argmax(axis=0)
        
        img_ab = self.enc.decode_histo_to_ab(idx_ab)
        img_l = x[:,:,None]

        img_lab = np.concatenate((img_l, img_ab), 2)
        rgb_img = lab2rgb(img_lab)
        return rgb_img
    

