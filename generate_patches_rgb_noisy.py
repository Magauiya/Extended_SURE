import numpy as np
import argparse
import glob
import os

parser = argparse.ArgumentParser(description='')
parser.add_argument('--gt_dir', dest='gt_dir', default='./Dataset/trainset/', help='dir of data')
parser.add_argument('--save_dir', dest='save_dir', default='./Dataset/trainset/', help='dir of patches')
parser.add_argument('--std_imp_min', dest='gt_min', type=float, default=10., help='minimum noise std. of Imperfect ground-truth')
parser.add_argument('--std_imp_max', dest='gt_max', type=float, default=10., help='maximum noise std. of Imperfect ground-truth')

parser.add_argument('--std_noisy_max', dest='ny_max', type=float, default=55., help='maximum noise std. of Noisy patches')

args = parser.parse_args()

# Reading ground-truth patches
gt = np.load(os.path.join(args.gt_dir, 'rgb_clean_patches.npy'), mmap_mode='r')
patches, h, w, ch = np.shape(gt)

# Unique names
if args.gt_min == args.gt_max:
	name = str(int(args.gt_min)) + '-' + str(int(args.gt_max))
else:
	name = str(int(args.gt_min))

# Generating Imperfect Ground-Truth
sigma_imperfect = np.random.uniform(args.gt_min, args.gt_max, (patches, )).astype(np.float32)
gt_noisy        = np.array(gt).astype(np.float32) + np.array([np.random.normal(0, sigma, (50,50,3)) for sigma in sigma_imperfect], dtype='float32')  
gt_path_data    = os.path.join(args.save_dir, 'gt_rgb_noisy_' + name + '_patches.npy')
gt_path_sigma   = os.path.join(args.save_dir, 'sigma_rgb_vector_imperfect_' + name + '.npy')

np.save(gt_path_data, gt_noisy.astype(np.float32))
np.save(gt_path_sigma, sigma_imperfect.astype(np.float32))

print('Imperfect GT patch shape: %s range: [%.3f-%.3f]' % (np.shape(gt_noisy), np.amin(gt_noisy), np.amax(gt_noisy)))
print('Imperfect GT sigma shape: %s range: [%.3f-%.3f]' % (np.shape(sigma_imperfect), np.amin(sigma_imperfect), np.amax(sigma_imperfect)))

# Generating Noisy patches using Imperfect Ground-Truth ---> Correlated noise
sigmavec     = np.random.uniform(gt_max + 0.1, args.ny_max, (patches, )).astype(np.float32)
sigma_dif    = np.sqrt(np.square(sigmavec) - np.square(sigma_imperfect))
noisy        = gt_noisy + np.array([np.random.normal(0, s, (50,50,3)) for s in sigma_dif], dtype='float32')  
ny_datapath  = os.path.join(args.save_dir, 'rgb_noisy_' + name + '_patches.npy')
ny_sigmapath = os.path.join(args.save_dir, 'sigma_rgb_vector_blind_' + name + '.npy')

np.save(ny_datapath, noisy.astype(np.float32))
np.save(ny_sigmapath, sigmavec.astype(np.float32))

print('Noisy patch shape: %s range: [%.3f-%.3f]' % (np.shape(noisy), np.amin(noisy), np.amax(noisy)))
print('Noisy sigma shape: %s range: [%.3f-%.3f]' % (np.shape(sigmavec), np.amin(sigmavec), np.amax(sigmavec)))
























