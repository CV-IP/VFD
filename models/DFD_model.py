import torch
from .base_model import BaseModel
from . import networks
import numpy as np
import torchvision
from PIL import Image
import torch.nn as nn
import torch.nn.functional as F
import os
from util import util
import torchvision.transforms as transforms

class DFDModel(BaseModel):

    @staticmethod
    def modify_commandline_options(parser, is_train=True):

        parser.set_defaults(norm='batch', netG='unet_af', dataset_mode='aligned')
        if is_train:
            parser.set_defaults(pool_size=0, gan_mode='vanilla')
            parser.add_argument('--lambda_L1', type=float, default=10.0, help='weight for L1 loss')

        return parser

    def __init__(self, opt):
        BaseModel.__init__(self, opt)

        if self.isTrain:
            self.model_names = ['G_audio', 'G_video']
        else:
            self.model_names = ['G_audio', 'G_video']

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.batch_size = opt.batch_size

        self.netG_video = networks.define_G(3, 3, opt.ngf, 'transformer_video', opt.norm,
                                            not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids).to(self.device)
        self.netG_audio = networks.define_G(3, 3, opt.ngf, 'transformer_audio', opt.norm,
                                            not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids).to(self.device)

        if self.isTrain:
            self.triplet_loss = nn.TripletMarginLoss(margin=100.0, p=2)
            self.pdist = nn.PairwiseDistance(p=2)
            self.optimizer_G = torch.optim.Adam(list(self.netG_audio.parameters())
                                                + list(self.netG_video.parameters()), lr=opt.lr, betas=(opt.beta1, 0.999),
                                                )
            self.optimizers.append(self.optimizer_G)

    def set_input(self, input_data):
        self.img_real = input_data['img_real']
        self.img_fake = input_data['img_fake']
        self.aud_real = input_data['aud_real']
        self.aud_fake = input_data['aud_fake']

    def set_test_input(self, input_data):
        self.target = input_data['label']
        self.img_real = input_data['img']
        self.aud_real = input_data['aud']

    def forward(self):
        if torch.cuda.is_available():
            self.img_real = torch.autograd.Variable(self.img_real).cuda()
            self.img_fake = torch.autograd.Variable(self.img_fake).cuda()
            self.aud_real = torch.autograd.Variable(self.aud_real).cuda()
            self.aud_fake = torch.autograd.Variable(self.aud_fake).cuda()
        else:
            self.img_real = torch.autograd.Variable(self.img_real)
            self.img_fake = torch.autograd.Variable(self.img_fake)
            self.aud_real = torch.autograd.Variable(self.aud_real)
            self.aud_fake = torch.autograd.Variable(self.aud_fake)

        self.aud_real_feat = self.netG_audio(self.aud_real.squeeze(0))
        self.aud_fake_feat = self.netG_audio(self.aud_fake.squeeze(0))
        self.img_fake_feat = self.netG_video(self.img_fake.squeeze(0))
        self.img_real_feat = self.netG_video(self.img_real.squeeze(0))

    def forward_test(self):
        if torch.cuda.is_available():
            self.img_real = torch.autograd.Variable(self.img_real).cuda()
            self.aud_real = torch.autograd.Variable(self.aud_real).cuda()
        else:
            self.img_real = torch.autograd.Variable(self.img_real)
            self.aud_real = torch.autograd.Variable(self.aud_real)

        self.aud_real_feat = self.netG_audio(self.aud_real.squeeze(0))
        self.img_real_feat = self.netG_video(self.img_real.squeeze(0))

    def backward_G(self):
        audio_real = self.aud_real_feat
        video_real = self.img_real_feat
        video_fake = self.img_fake_feat

        self.loss_A_V = None
        for i in video_fake:
            if self.loss_A_V is None:
                self.loss_A_V = self.triplet_loss(audio_real, video_real, i)
            else:
                self.loss_A_V += self.triplet_loss(audio_real, video_real, i)
        self.loss = self.loss_A_V
        self.loss.backward()
        return {
            'loss_A_V': self.loss_A_V.detach().item(),
        }

    def optimize_parameters(self):
        self.forward()
        self.optimizer_G.zero_grad()
        loss_pack = self.backward_G()
        loss_G_AV = loss_pack['loss_A_V']
        self.optimizer_G.step()
        return loss_G_AV

    def val(self):
        with torch.no_grad():
            self.forward()
            audio_real = self.aud_real_feat
            audio_fake = self.aud_fake_feat
            video_real = self.img_real_feat
            video_fake = self.img_fake_feat
            self.sim_A_V = self.simi(audio_real, video_real)
            self.sim_V_A = []
            for i, j in zip(video_fake, audio_fake):
                self.sim_V_A.append(self.simi(i.unsqueeze(0), j.unsqueeze(0)))
        return self.sim_A_V, self.sim_V_A

    def simi(self, anchor, pos):
        pdist = nn.PairwiseDistance(p=2)
        dist = pdist(anchor, pos)
        return dist