import time
from options.test_options import TestOptions
from data import create_dataset
from models import create_model
import os
import math
import cv2
from PIL import Image
from util import util
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import pylab as pl
import torch
import random
from torchvision import transforms
from torchcam.cams import SmoothGradCAMpp
from torchvision.transforms.functional import to_pil_image
from torchcam.utils import overlay_mask

def auc(real, fake):
    label_all = []
    target_all = []
    for ind in real:
        target_all.append(1)
        label_all.append(-ind)
    for ind in fake:
        target_all.append(0)
        label_all.append(-ind)

    from sklearn.metrics import roc_auc_score
    return roc_auc_score(target_all, label_all)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if __name__ == '__main__':
    opt = TestOptions().parse()  # get test options
    # hard-code some parameters for test
    opt.num_threads = 4  # test code only supports num_threads = 1
    opt.batch_size = 1  # test code only supports batch_size = 1
    opt.serial_batches = False  #r disable data shuffling; comment this line if results on randomly chosen images ae needed.
    opt.no_flip = True  # no flip; comment this line if results on flipped images are needed.
    opt.display_id = -1  # no visdom display; the test code saves the results to a HTML file.
    opt.mode = 'test'
    dataset = create_dataset(opt)  # create a dataset given opt.dataset_mode and other options
    model = create_model(opt)  # create a model given opt.model and other options
    model.setup(opt)  # regular setup: load and print networks; create schedulers

    if opt.eval:
        model.eval()

    dataset_size = len(dataset)
    print('The number of test images dir = %d' % dataset_size)


    total_iters = 0
    label = None
    real = []
    fake = []

    with tqdm(total=dataset_size) as pbar:
        for i, data in enumerate(dataset):
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
            total_iters += 1
            pbar.update()

    print(auc(real, fake))