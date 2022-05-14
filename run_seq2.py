import argparse
import os
from shutil import copyfile

import torch
import torch.nn as nn
from models.models import DIFNet2

from PIL import Image
import numpy as np
import pdb
from tqdm import tqdm

# python run_seq2.py --cuda --n_iter 3 --skip 2


def load_tensor(filename):
    image = Image.open(filename).convert("RGB")
    return torch.cuda.FloatTensor(np.array(image).transpose(2, 0, 1).astype(np.float32)[None, :, :, :] / 255.0)


parser = argparse.ArgumentParser()
parser.add_argument("--model1", default="./trained_models/DIFNet2.pth")  ####2
parser.add_argument("--in_file", default="images/")
parser.add_argument("--out_file", default="output/")
parser.add_argument("--n_iter", type=int, default=1, help="number of stabilization interations")
parser.add_argument("--skip", type=int, default=1, help="number of frame skips for interpolation")
parser.add_argument("--cuda", action="store_true", help="use GPU computation")
opt = parser.parse_args()
print(opt)

if torch.cuda.is_available() and not opt.cuda:
    print("WARNING: You have a CUDA device, so you should probably run with --cuda")

##########################################################

### Networks
model = DIFNet2()

# Place Network in cuda memory
if opt.cuda:
    model.cuda()

### DataParallel
model = nn.DataParallel(model)
model.load_state_dict(torch.load(opt.model1))
model.eval()
model = model.module


##########################################################

filename_list = os.listdir(opt.in_file)
filename_list.sort()
filename_list = filename_list[0:10]
# filename_list[0:5] -- ['001.png', '002.png', '003.png', '004.png', '005.png']


if not os.path.exists(opt.out_file):
    os.makedirs(opt.out_file)
copyfile(opt.in_file + filename_list[0], opt.out_file + filename_list[0])
copyfile(opt.in_file + filename_list[-1], opt.out_file + filename_list[-1])

### Generate output sequence
destion = opt.out_file
min_i = 0
max_i = len(filename_list) - 1
for n in range(opt.n_iter):
    if n == 0:
        source = opt.in_file
    else:
        source = opt.out_file

    for i in tqdm(range(1, max_i)):
        prev_filename = destion + filename_list[i - opt.skip if i - opt.skip else min_i]
        prev_frame = load_tensor(prev_filename)

        curr_filename = source + filename_list[i]
        curr_frame = load_tensor(curr_filename)

        next_filename = source + filename_list[i + opt.skip if i + opt.skip < max_i else max_i]
        next_frame = load_tensor(next_filename)

        with torch.no_grad():
            fhat, I_int = model(prev_frame, curr_frame, next_frame, 0.5)  # Notice 0.5

        # Save image
        output_filename = destion + filename_list[i]
        # print("Saving ", prev_filename, curr_filename, next_filename, "==>", output_filename, "...")
        img = Image.fromarray(np.uint8(fhat.cpu().squeeze().permute(1, 2, 0) * 255))
        img.save(output_filename)


### Make video
# print('\nMaking video...')
# frame2vid(src=opt.out_file, vidDir=opt.out_file[:-1] + '.avi')

### Assess with metrics
# print('\nComputing metrics...')
# metrics(in_src=opt.in_file, out_src=opt.out_file)
