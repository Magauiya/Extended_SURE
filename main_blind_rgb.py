from __future__ import print_function
from __future__ import division
import argparse
from glob import glob

import tensorflow as tf
import natsort

from model_blind_rgb import denoiser
from utils import *
import os

parser = argparse.ArgumentParser(description='')
parser.add_argument('--epoch', dest='epoch', type=int, default=50, help='# of epoch')
parser.add_argument('--batch_size', dest='batch_size', type=int, default=128, help='# images in batch')
parser.add_argument('--lr', dest='lr', type=float, default=0.001, help='initial learning rate for sgd')
parser.add_argument('--sigma', dest='sigma', type=float, default=25.0, help='noise level (for evaluation and testing)')


parser.add_argument('--data', dest='data', default='./Dataset/trainset', help='training data path')
parser.add_argument('--checkpoint_dir', dest='ckpt_dir', default='./checkpoint', help='models are saved here')
parser.add_argument('--sample_dir', dest='sample_dir', default='./sample', help='sample are saved here')
parser.add_argument('--log_dir', dest='log_dir', default='./logs', help='tensorboard logs are saved here')
parser.add_argument('--test_dir', dest='test_dir', default='./test', help='test sample are saved here')
parser.add_argument('--eval_set', dest='eval_set', default='CBSD68', help='dataset for eval in training')
parser.add_argument('--test_set', dest='test_set', default='CBSD68', help='dataset for testing')

parser.add_argument('--cost', dest='cost', default='sure', help='cost to minimize: MSE, SURE, e-SURE')
parser.add_argument('--gpu', dest='gpu', default='1', help='which gpu to use')
parser.add_argument('--type', dest='type', default='', help='arg to give unique names to realizations')
parser.add_argument('--gt_type', dest='gt_type', default='', help='arg to give unique names to realizations')
parser.add_argument('--phase', dest='phase', default='train', help='train or test')
parser.add_argument('--use_gpu', dest='use_gpu', type=int, default=1, help='gpu flag, 1 for GPU and 0 for CPU')

args = parser.parse_args()


def denoiser_train(denoiser, lr):
    eval_files = glob('./Dataset/testset/{}/*.png'.format(args.eval_set))
    denoiser.train(args.data, eval_files, batch_size=args.batch_size, epoch=args.epoch, lr=lr, gt_type=args.gt_type)


def denoiser_test(denoiser, save_dir):

    # List of paths to noisy testset images
    if args.sigma==25.0:
        noisy_files = natsort.natsorted(glob('./Dataset/testset/{}/sigma25/*.npy'.format(args.test_set)))
    elif args.sigma==50.0:
        noisy_files = natsort.natsorted(glob('./Dataset/testset/{}/sigma50/*.npy'.format(args.test_set)))
    print('Testing on {} dataset'.format(args.test_set))

    # List of paths to ground-truth testset images
    test_files = natsort.natsorted(glob('./Dataset/testset/{}/*.png'.format(args.test_set)))
    
    denoiser.test(test_files, noisy_files, save_dir)


def main(_):
    
    #the following string is attached to checkpoint, log and image folder names
    name = "CDnCNN_" + args.cost + '_' + str(args.type)
    
    ckpt_dir = args.ckpt_dir + "/" + name
    sample_dir = args.sample_dir + "/" + name
    test_dir = args.test_dir + "/" + name
    log_dir = args.log_dir + "/" + name
    print('CKPT path: ', ckpt_dir)
    
    if not os.path.exists(args.ckpt_dir):
        os.makedirs(args.ckpt_dir)

    lr = args.lr * np.ones([args.epoch])
    lr[40:] = lr[0] / 10.0 #lr decay

    
    if args.use_gpu:
        # added to control the gpu memory
        print("GPU\n")
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.9)
        with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
            model = denoiser(sess, sigma=args.sigma, cost_str=args.cost, ckpt_dir=ckpt_dir, sample_dir=sample_dir, log_dir=log_dir)
            if args.phase == 'train':
                denoiser_train(model, lr=lr)
            elif args.phase == 'test':
                denoiser_test(model, test_dir)
            else:
                print('[!]Unknown phase')
                exit(0)
    else:
        print("CPU\n")
        with tf.Session() as sess:
            model = denoiser(sess, sigma=args.sigma, cost_str=args.cost, ckpt_dir=ckpt_dir, sample_dir=sample_dir, log_dir=log_dir)
            if args.phase == 'train':
                denoiser_train(model, lr=lr)
            elif args.phase == 'test':
                denoiser_test(model, test_dir)
            else:
                print('[!]Unknown phase')
                exit(0)


if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES']=str(args.gpu)
    tf.app.run()
