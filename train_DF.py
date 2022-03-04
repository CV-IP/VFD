"""
Anonymous release of VFD
Part of the framework is borrowed from
https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix
Many thanks to these authors!
"""
import time
from options.train_options import TrainOptions
from data import create_dataset
from models import create_model
import os
import math
import cv2
from PIL import Image
from util import util
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
import pylab as pl
import torch
import random


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
setup_seed(20)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def auc(real, fake):
    label_all = []
    target_all = []
    for ind in real:
        target_all.append(1)
        label_all.append(ind)

    for ind in fake:
        target_all.append(0)
        label_all.append(ind)

    from sklearn.metrics import roc_auc_score
    return roc_auc_score(target_all, label_all)

if __name__ == '__main__':
    opt = TrainOptions().parse()
    dataset = create_dataset(opt)
    dataset_size = len(dataset)

    opt.mode = 'val'
    opt.serial_batches = False
    dataset_val = create_dataset(opt)
    dataset_val_size = len(dataset_val)
    print('The number of training images dir = %d' % dataset_size)
    print('The number of val images dir = %d' % dataset_val_size)

    model = create_model(opt)  # create a model given opt.model and other options
    model.setup(opt)  # regular setup: load and print networks; create schedulers
    total_iters = 0  # the total number of training iterations

    loss_x = []
    loss_y_g = []
    loss_y_l = []
    loss_y_t = []
    loss_y_f = []
    loss_epo = 0
    for epoch in range(opt.epoch_count,
                       opt.n_epochs + opt.n_epochs_decay + 1):
        epoch_start_time = time.time()
        iter_data_time = time.time()
        time_start = epoch_start_time
        epoch_iter = 0
        iter_start_time = time.time()
        if total_iters % opt.print_freq == 0:
            t_data = iter_start_time - iter_data_time
        loss_G_AV_all = 0
        for i, data in enumerate(dataset):
            input_data = {'img_real': data['img_real'],
                          'img_fake': data['img_fake'],
                          'aud_real': data['aud_real'],
                          'aud_fake': data['aud_fake'],
                          }
            model.set_input(input_data)
            loss_G_AV = model.optimize_parameters()

            loss_G_AV_all += loss_G_AV
            loss_epo += loss_G_AV
            total_iters += 1

            if total_iters % 10 == 0:
                print('epoch %d, total_iters %d: loss_G_AV: %.3f(%.3f), time cost: %.2f s' %
                      (epoch, total_iters, loss_G_AV, loss_G_AV_all / total_iters, time.time() - time_start))
                time_start = time.time()

            if total_iters % opt.save_latest_freq == 0:
                print('saving the latest model (epoch %d, total_iters %d)' % (epoch, total_iters))
                save_suffix = 'iter_%d' % total_iters if opt.save_by_iter else 'latest'
                model.save_networks(save_suffix)
                model.eval()
                real = []
                fake = []
                with tqdm(total=len(dataset_val)) as pbar:
                    for i, data in enumerate(dataset_val):
                        input_data = {'img_real': data['img_real'],
                                      'img_fake': data['img_fake'],
                                      'aud_real': data['aud_real'],
                                      'aud_fake': data['aud_fake'],
                                      }
                        model.set_input(input_data)
                        dist_AV, dist_VA = model.val()
                        real.append(dist_AV.item())
                        for i in dist_VA:
                            fake.append(i.item())
                        pbar.update()
                _auc = auc(real, fake)
                print('Val auc (for refer) %.4f'%(_auc))
                model.train()

        iter_data_time = time.time()
        if epoch % opt.save_epoch_freq == 0:
            print('saving the model at the end of epoch %d, iters %d' % (epoch, total_iters))
            model.save_networks('latest')
            model.save_networks(epoch)
            model.eval()
            real = []
            fake = []
            for i, data in enumerate(dataset_val):
                input_data = {'img_real': data['img_real'],
                              'img_fake': data['img_fake'],
                              'aud_real': data['aud_real'],
                              'aud_fake': data['aud_fake'],
                              }
                model.set_input(input_data)
                dist_AV, dist_VA = model.val()
                real.append(dist_AV.item())
                for i in dist_VA:
                    fake.append(i.item())
            _auc = auc(real, fake)
            model.train()
            loss_epo = 0

        print('End of epoch %d / %d \t Time Taken: %d sec' % (
        epoch, opt.n_epochs + opt.n_epochs_decay, time.time() - epoch_start_time))
        model.update_learning_rate()