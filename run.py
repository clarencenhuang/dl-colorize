# This file processes an image
# If you change the training code, add normalization or something different
# Please make sure this works.
#
# Usage: run.py <model_file_path> <input_image> <output_image>
#
# model_file_path should initially be a path to an h5 file.  By default wandb
# will save a model in wandb/<run_id>/model-best.h5 

import argparse
import torch
from PIL import Image, ImageOps
import numpy as np
from model import ColorizeClassifier
from color_utils import torch_softmax2image

parser = argparse.ArgumentParser(description='Run model file on input_image file and output out_image.')
parser.add_argument('model', type=str)
parser.add_argument('input_image')
parser.add_argument('output_image')

height=256
width=256

args = parser.parse_args()

train_state = torch.load(args.model)
model = ColorizeClassifier(feature_cascade=(512, 256, 64, 64), training=False)
model.load_state_dict(train_state['state_dict'])

img = Image.open(args.input_image)
sized_img = ImageOps.fit(img, (256,256), 0, 0, (0.5, 0.5))
bw_image = np.array(sized_img.convert('L'))
normalized =((np.double(bw_image) - 127.5)/255)
x = torch.FloatTensor(normalized).unsqueeze(0).unsqueeze(0)

if torch.cuda.is_available():
    x = x.cuda()
    model.cuda()

with torch.no_grad():
    model.eval()
    y = model(x)

recolored = torch_softmax2image(x, y, 0, 0.2)
new_image = Image.fromarray(np.uint8(np.round(recolored*255)), 'RGB')

# new_image is the output from the model
new_image.save(args.output_image)