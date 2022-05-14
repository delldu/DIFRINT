import argparse
import os
from shutil import copyfile

import torch
import torch.nn as nn
from models.models import DIFNet2
# from models.pwcNet import PwcNet
# from metrics import metrics
# from frame2vid import frame2vid

from PIL import Image
import numpy as np
import pdb
from tqdm import tqdm
#python run_seq2.py --cuda --n_iter 3 --skip 2

def load_tensor(filename):
	image = Image.open(filename).convert("RGB")	
	return torch.cuda.FloatTensor(np.array(image).transpose(2, 0, 1).astype(np.float32)[None,:,:,:] / 255.0)


parser = argparse.ArgumentParser()
parser.add_argument('--model1', default='./trained_models/DIFNet2.pth') ####2
parser.add_argument('--in_file', default='images/')
parser.add_argument('--out_file', default='output/')
parser.add_argument('--n_iter', type=int, default=1, help='number of stabilization interations')
parser.add_argument('--skip', type=int, default=1, help='number of frame skips for interpolation')
#parser.add_argument('--input_nc', type=int, default=3, help='number of channels of input data')
#parser.add_argument('--output_nc', type=int, default=3, help='number of channels of output data')
#parser.add_argument('--height', type=int, default=720, help='size of the data crop (squared assumed)')
#parser.add_argument('--width', type=int, default=1280, help='size of the data crop (squared assumed)')
parser.add_argument('--cuda', action='store_true', help='use GPU computation')
opt = parser.parse_args()
print(opt)

if torch.cuda.is_available() and not opt.cuda:
	print("WARNING: You have a CUDA device, so you should probably run with --cuda")

##########################################################

### Networks
DIFNet = DIFNet2()

# Place Network in cuda memory
if opt.cuda:
	DIFNet.cuda()

### DataParallel
DIFNet = nn.DataParallel(DIFNet)
DIFNet.load_state_dict(torch.load(opt.model1))
DIFNet.eval()

##########################################################

filename_list = os.listdir(opt.in_file)
filename_list.sort()
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
		if i - opt.skip >= 0:
			fn_o_p = destion + filename_list[i - opt.skip]
		else:
			fn_o_p = destion + filename_list[min_i]
		ft_o_p = load_tensor(fn_o_p)

		fn_i_c = source + filename_list[i]	
		ft_i_c = load_tensor(fn_i_c)

		if i + opt.skip < max_i:
			fn_i_n = source + filename_list[i + opt.skip]
		else:
			fn_i_n = source + filename_list[max_i]
		ft_i_n = load_tensor(fn_i_n)

		with torch.no_grad():
			fhat, I_int = DIFNet(ft_o_p, ft_i_n, ft_i_c, ft_i_n, ft_o_p, 0.5) # Notice 0.5

		# Save image
		fn_o_c = destion + filename_list[i]
		# print("Saving ", fn_o_p, fn_i_c, fn_i_n, "==>", fn_o_c, "...")
		img = Image.fromarray(np.uint8(fhat.cpu().squeeze().permute(1,2,0)*255))
		img.save(fn_o_c)


### Make video
#print('\nMaking video...')
#frame2vid(src=opt.out_file, vidDir=opt.out_file[:-1] + '.avi')

### Assess with metrics
#print('\nComputing metrics...')
#metrics(in_src=opt.in_file, out_src=opt.out_file)
