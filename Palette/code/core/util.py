import random
import os
from os.path import join
import cv2
import numpy as np
import math
import torch
from torch.nn.parallel import DistributedDataParallel as DDP
from torchvision.utils import make_grid


def tensor2img(tensor, out_type=np.uint8, min_max=(-1, 1)):
    '''
    Converts a torch Tensor into an image Numpy array
    Input: 4D(B,(3/1),H,W), 3D(C,H,W), or 2D(H,W), any range, RGB channel order
    Output: 3D(H,W,C) or 2D(H,W), [0,255], np.uint8 (default)
    '''
    tensor = tensor.clamp_(*min_max)  # clamp
    n_dim = tensor.dim()
    if n_dim == 4:
        n_img = len(tensor)
        img_np = make_grid(tensor, nrow=int(math.sqrt(n_img)), normalize=False).numpy()
        img_np = np.transpose(img_np, (1, 2, 0))  # HWC, RGB
    elif n_dim == 3:
        img_np = tensor.numpy()
        img_np = np.transpose(img_np, (1, 2, 0))  # HWC, RGB
    elif n_dim == 2:
        img_np = tensor.numpy()
    else:
        raise TypeError('Only support 4D, 3D and 2D tensor. But received with dimension: {:d}'.format(n_dim))
    if out_type == np.uint8:
        img_np = ((img_np+1) * 127.5).round()
        # Important. Unlike matlab, numpy.unit8() WILL NOT round by default.
    return img_np.astype(out_type).squeeze()

def tensor2CTimg(tensor, out_type=np.uint8, min_max=(-1, 1)):
    '''
    Converts a torch Tensor into an image Numpy array whose elems in [-2000, 2000]
    Input: 4D(B,(3/1),H,W), 3D(C,H,W), or 2D(H,W), any range, RGB channel order
    Output: 3D(H,W,C) or 2D(H,W), [0,255], np.uint8 (default)
    '''
    tensor = tensor.clamp_(*min_max)  # clamp
    n_dim = tensor.dim()
    if n_dim == 4:
        n_img = len(tensor)
        img_np = make_grid(tensor, nrow=int(math.sqrt(n_img)), normalize=False).numpy()
        img_np = np.transpose(img_np, (1, 2, 0))  # HWC, RGB
    elif n_dim == 3:
        img_np = tensor.numpy()
        img_np = np.transpose(img_np, (1, 2, 0))  # HWC, RGB
    elif n_dim == 2:
        img_np = tensor.numpy()
    else:
        raise TypeError('Only support 4D, 3D and 2D tensor. But received with dimension: {:d}'.format(n_dim))
    if out_type == np.int16:
        img_np = ((img_np+1) * 2000 - 2000).round()
        # Important. Unlike matlab, numpy.unit8() WILL NOT round by default.
    return img_np.astype(out_type).squeeze()

def postprocess(images):
	return [tensor2img(image) for image in images]
def postprocess4CT(images):
	return [tensor2CTimg(image, out_type=np.int16) for image in images]


def set_seed(seed, gl_seed=0):
	"""  set random seed, gl_seed used in worker_init_fn function """
	if seed >=0 and gl_seed>=0:
		seed += gl_seed
		torch.manual_seed(seed)
		torch.cuda.manual_seed_all(seed)
		np.random.seed(seed)
		random.seed(seed)

	''' change the deterministic and benchmark maybe cause uncertain convolution behavior. 
		speed-reproducibility tradeoff https://pytorch.org/docs/stable/notes/randomness.html '''
	if seed >=0 and gl_seed>=0:  # slower, more reproducible
		torch.backends.cudnn.deterministic = True
		torch.backends.cudnn.benchmark = False
	else:  # faster, less reproducible
		torch.backends.cudnn.deterministic = False
		torch.backends.cudnn.benchmark = True

def set_gpu(args, distributed=False, rank=0):
	""" set parameter to gpu or ddp """
	if args is None:
		return None
	if distributed and isinstance(args, torch.nn.Module):
		return DDP(args.cuda(), device_ids=[rank], output_device=rank, broadcast_buffers=True, find_unused_parameters=True)
	else:
		return args.cuda()
		
def set_device(args, distributed=False, rank=0):
	""" set parameter to gpu or cpu """
	if torch.cuda.is_available():
		if isinstance(args, list):
			return (set_gpu(item, distributed, rank) for item in args)
		elif isinstance(args, dict):
			return {key:set_gpu(args[key], distributed, rank) for key in args}
		else:
			args = set_gpu(args, distributed, rank)
	return args


def make_mirflickr25k_traindata(data_root, flist_root):
	gray_imgs_root = data_root.replace('color', "gray")
	os.makedirs(gray_imgs_root, exist_ok=True)
	jpg_files = [f for f in os.listdir(data_root) if f.endswith('.png')]
	random.shuffle(jpg_files)

	split_point = int(len(jpg_files) * 0.8)
	train_files = jpg_files[:split_point]
	test_files = jpg_files[split_point:]

	def convert_to_gray_save(file_list, output_flist):
		with open(output_flist, 'w') as f:
			for file in file_list:
				image = cv2.imread(join(data_root, file))
				gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
				print(gray_image.shape)
				gray_image_path = join(gray_imgs_root ,file)
				cv2.imwrite(gray_image_path, gray_image)
				f.write(file + '\n')
	
	convert_to_gray_save(train_files, join(flist_root, "train.flist"))
	convert_to_gray_save(test_files, join(flist_root, "test.flist"))

if __name__ == "__main__":
	# color_image_root = "datasets/mirflickr25k/images/color"
	# flist_root = "datasets/mirflickr25k/flist"
	# make_mirflickr25k_traindata(color_image_root, flist_root)
	dir = "datasets/mirflickr25k-kaggle/flist/train.flist"
	images = [i for i in np.genfromtxt(dir, dtype=np.str, encoding='utf-8')]