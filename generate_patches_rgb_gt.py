import argparse
import glob
from PIL import Image
import PIL
import random
from utils import *

# the pixel value range is '0-255'(uint8) of clean patches

# macro
DATA_AUG_TIMES = 1  # transform a sample to a different sample for DATA_AUG_TIMES times

parser = argparse.ArgumentParser(description='')
parser.add_argument('--src_dir', dest='src_dir', default='./Dataset/CBSD432', help='dir of data')
parser.add_argument('--save_dir', dest='save_dir', default='./Dataset/trainset/', help='dir of patches')
parser.add_argument('--patch_size', dest='pat_size', type=int, default=50, help='patch size')
parser.add_argument('--stride', dest='stride', type=int, default=24, help='stride')
parser.add_argument('--step', dest='step', type=int, default=0, help='step')
parser.add_argument('--batch_size', dest='bat_size', type=int, default=128, help='batch size')
args = parser.parse_args()


def generate_patches(isDebug=False):
    global DATA_AUG_TIMES
    count = 0
    filepaths = glob.glob(args.src_dir + '/*.jpg')
    if isDebug:
        filepaths = filepaths[:10]
    print "number of training data %d" % len(filepaths)
    
    scales = [1, 0.9, 0.8, 0.7]

    # calculate the number of patches
    for i in xrange(len(filepaths)):
        img = Image.open(filepaths[i]).convert('RGB')  # convert RGB to gray
        for s in xrange(len(scales)):
            newsize = (int(img.size[0] * scales[s]), int(img.size[1] * scales[s]))
            img_s = img.resize(newsize, resample=PIL.Image.BICUBIC)  # do not change the original img
            im_h, im_w = img_s.size
            for x in range(0 + args.step, (im_h - args.pat_size), args.stride):
                for y in range(0 + args.step, (im_w - args.pat_size), args.stride):
                    count += 1
    origin_patch_num = count * DATA_AUG_TIMES
    
    if origin_patch_num % args.bat_size != 0:
        numPatches = (origin_patch_num / args.bat_size + 1) * args.bat_size
    else:
        numPatches = origin_patch_num
    print "total patches = %d , batch size = %d, total batches = %d" % \
          (numPatches, args.bat_size, numPatches / args.bat_size)
    
    # data matrix 4-D
    inputs = np.zeros((numPatches, args.pat_size, args.pat_size, 3), dtype="uint8")
    
    count = 0
    # generate patches
    for i in xrange(len(filepaths)):
        img = Image.open(filepaths[i]).convert('RGB')
        for s in xrange(len(scales)):
            newsize = (int(img.size[0] * scales[s]), int(img.size[1] * scales[s]))
            # print newsize
            img_s = img.resize(newsize, resample=PIL.Image.BICUBIC)
            img_s = np.array(img_s)  # extend one dimension
            
            for j in xrange(DATA_AUG_TIMES):
                im_h, im_w, _ = img_s.shape
                for x in range(0 + args.step, im_h - args.pat_size, args.stride):
                    for y in range(0 + args.step, im_w - args.pat_size, args.stride):
                        inputs[count, :, :, :] = data_augmentation(img_s[x:x + args.pat_size, y:y + args.pat_size, :], \
                                                                   random.randint(0, 7))
                        count += 1
    # pad the batch
    if count < numPatches:
        to_pad = numPatches - count
        inputs[-to_pad:, :, :, :] = inputs[:to_pad, :, :, :]
    
    if not os.path.exists(args.save_dir):
        os.mkdir(args.save_dir)
    np.save(args.save_dir, inputs) 
    print "size of inputs tensor = " + str(inputs.shape)

if __name__ == '__main__':
    generate_patches()
