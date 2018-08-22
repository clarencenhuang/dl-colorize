from skimage.color import rgb2lab, lab2rgb, rgb2gray, xyz2lab
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

def pil2lab(rgb_hwc):
    lab_hwc = rgb2lab(rgb_hwc)
    l_hw = (lab_hwc[:,:,0] - 50)/100.0
    ab_hwc = (lab_hwc[:,:,1:])/255.0
    return l_hw, ab_hwc


def lab2pil(l, ab):
    l = l * 100.0 + 50
    ab = ab * 255.0
    lab = np.concatenate((l, ab), 2)
    rgb = lab2rgb(lab)
    return rgb

# logic for color encoding

stride = len(range(-128, 128, 10))
idx_to_bin_idx = np.array([[4, 17],  [4, 18],  [4, 19],  [4, 20],  [4, 21],  [5, 15],  [5, 16],  [5, 17],  [5, 18],  [5, 19],  [5, 20],  [5, 21],  [6, 13],  [6, 14],  [6, 15],  [6, 16],  [6, 17],  [6, 18],  [6, 19],  [6, 20],  [6, 21],  [7, 11],  [7, 12],  [7, 13],  [7, 14],  [7, 15],  [7, 16],  [7, 17],  [7, 18],  [7, 19],  [7, 20],  [7, 21],  [8, 10],  [8, 11],  [8, 12],  [8, 13],  [8, 14],  [8, 15],  [8, 16],  [8, 17],  [8, 18],  [8, 19],  [8, 20],  [8, 21],  [9, 9],  [9, 10],  [9, 11],  [9, 12],  [9, 13],  [9, 14],  [9, 15],  [9, 16],  [9, 17],  [9, 18],  [9, 19],  [9, 20],  [9, 21],  [9, 22],  [10, 8],  [10, 9],  [10, 10],  [10, 11],  [10, 12],  [10, 13],  [10, 14],  [10, 15],  [10, 16],  [10, 17],  [10, 18],  [10, 19],  [10, 20],  [10, 21],  [10, 22],  [11, 7],  [11, 8],  [11, 9],  [11, 10],  [11, 11],  [11, 12],  [11, 13],  [11, 14],  [11, 15],  [11, 16],  [11, 17],  [11, 18],  [11, 19],  [11, 20],  [11, 21],  [11, 22],  [12, 6],  [12, 7],  [12, 8],  [12, 9],  [12, 10],  [12, 11],  [12, 12],  [12, 13],  [12, 14],  [12, 15],  [12, 16],  [12, 17],  [12, 18],  [12, 19],  [12, 20],  [12, 21],  [13, 6],  [13, 7],  [13, 8],  [13, 9],  [13, 10],  [13, 11],  [13, 12],  [13, 13],  [13, 14],  [13, 15],  [13, 16],  [13, 17],  [13, 18],  [13, 19],  [13, 20],  [13, 21],  [14, 5],  [14, 6],  [14, 7],  [14, 8],  [14, 9],  [14, 10],  [14, 11],  [14, 12],  [14, 13],  [14, 14],  [14, 15],  [14, 16],  [14, 17],  [14, 18],  [14, 19],  [14, 20],  [14, 21],  [15, 4],  [15, 5],  [15, 6],  [15, 7],  [15, 8],  [15, 9],  [15, 10],  [15, 11],  [15, 12],  [15, 13],  [15, 14],  [15, 15],  [15, 16],  [15, 17],  [15, 18],  [15, 19],  [15, 20],  [16, 4],  [16, 5],  [16, 6],  [16, 7],  [16, 8],  [16, 9],  [16, 10],  [16, 11],  [16, 12],  [16, 13],  [16, 14],  [16, 15],  [16, 16],  [16, 17],  [16, 18],  [16, 19],  [16, 20],  [17, 3],  [17, 4],  [17, 5],  [17, 6],  [17, 7],  [17, 8],  [17, 9],  [17, 10],  [17, 11],  [17, 12],  [17, 13],  [17, 14],  [17, 15],  [17, 16],  [17, 17],  [17, 18],  [17, 19],  [17, 20],  [18, 2],  [18, 3],  [18, 4],  [18, 5],  [18, 6],  [18, 7],  [18, 8],  [18, 9],  [18, 10],  [18, 11],  [18, 12],  [18, 13],  [18, 14],  [18, 15],  [18, 16],  [18, 17],  [18, 18],  [18, 19],  [19, 2],  [19, 3],  [19, 4],  [19, 5],  [19, 6],  [19, 7],  [19, 8],  [19, 9],  [19, 10],  [19, 11],  [19, 12],  [19, 13],  [19, 14],  [19, 15],  [19, 16],  [19, 17],  [19, 18],  [19, 19],  [20, 2],  [20, 3],  [20, 4],  [20, 5],  [20, 6],  [20, 7],  [20, 8],  [20, 9],  [20, 10],  [20, 11],  [20, 12],  [20, 13],  [20, 14],  [20, 15],  [20, 16],  [20, 17],  [20, 18],  [20, 19],  [21, 3],  [21, 4],  [21, 5],  [21, 6],  [21, 7],  [21, 8],  [21, 9],  [21, 10],  [21, 11],  [21, 12],  [21, 13],  [21, 14],  [21, 15],  [22, 5],  [22, 6],  [22, 7],  [22, 8],  [22, 9]], dtype=np.uint8)
idx_to_lab = np.double(idx_to_bin_idx) * 10 - 128 + 5
bin_idx_to_idx = np.zeros((stride, stride))
bin_idx_to_idx[idx_to_bin_idx[:,0], idx_to_bin_idx[:,1]] = np.arange(idx_to_bin_idx.shape[0]) 

def ab2bin(ab_hwc):
    assert ab_hwc.shape[2] == 2, 'need shape (H x W X 2)'
    ab_hwc = ab_hwc * 255.0
    a_bin_idx = np.uint8((ab_hwc[:,:,0] + 128) / 10)
    b_bin_idx = np.uint8((ab_hwc[:,:,1] + 128) / 10)
    return bin_idx_to_idx[a_bin_idx, b_bin_idx]

def bin2ab(idx):
    #assert len(idx.shape) == 2, 'need shape (H x W)'
    return idx_to_lab[idx, :] / 255.0

def torch_softmax2image(xb, yb, idx, annealing_term=None):
    x = (xb.cpu().numpy())[idx,0,:]
    y = (yb.cpu().numpy())[idx,:]
    if annealing_term is None:
        idx_ab = y.argmax(axis=0)
        img_ab = bin2ab(idx_ab)
    else:
        yc = y.transpose(1,2,0)
        yexp = np.exp(yc)
        probs = yexp / yexp.sum(axis=2)[:,:,None]
        t = 0.4
        z = np.exp(np.log(probs+1e-8)/t)
        z = z / z.sum(axis=2)[:,:,None]
        probs_expanded = np.repeat(z[:,:,:,None], 2, 3)
        weighted = probs_expanded * idx_to_lab / 255.0
        img_ab = np.sum(weighted, axis=2)
    img_l = x[:,:,None]
    return lab2pil(img_l, img_ab)

class ColorLoss(nn.Module):
    
    def  __init__(self, weights=None):
        super(ColorLoss, self).__init__()
        self.weights = weights #/ weights.mean()
        self.weights.requires_grad = False
    
    def forward(self, label, target):
        log_sm_lbls = F.log_softmax(label, dim=1) # log softmax has nicer numerical properties
        bs, c, h, w = label.shape
        expanded_weights = self.weights[:,None,None].expand([262, h, w])
        #print("expanded_weights ", expanded_weights.shape, " target ", target.shape)
        divisor = torch.sum(expanded_weights * target)
        return -torch.sum( expanded_weights * target * log_sm_lbls) / divisor