import os
from data.base_dataset import BaseDataset, get_transform
from data.image_folder import make_dataset
from PIL import Image
import random
from util import util
import torch
import json
import cv2
import numpy as np
from tqdm import tqdm
import torchaudio
import warnings
import librosa
warnings.filterwarnings('ignore')


class VoxImageDataset(BaseDataset):

    def __init__(self, opt):

        BaseDataset.__init__(self, opt)
        self.opt = opt
        self.stage = opt.mode
        if self.stage == 'train':
            self.video_dataset_path = os.path.join(self.opt.dataroot, 'face')
            self.audio_dataset_path = os.path.join(self.opt.dataroot, 'voice')
        elif self.stage == 'val':
            self.video_dataset_path = os.path.join(self.opt.dataroot, 'face')
            self.audio_dataset_path = os.path.join(self.opt.dataroot, 'voice')
        else:
            self.video_dataset_path = os.path.join(self.opt.dataroot, 'face')
            self.audio_dataset_path = os.path.join(self.opt.dataroot, 'voice')
        self.video_list, self.audio_list = self.get_video_list(self.video_dataset_path, self.audio_dataset_path, self.stage)
        random.seed(3)
        self.transform = get_transform(self.opt)

    def __getitem__(self, index):

        skip = self.__len__() // 2
        sample_exa = 3
        video_path = self.video_list[index]
        audio_path = self.audio_list[index]

        img_real = []
        aud_real = []
        img_fake = []
        aud_fake = []

        image_input = Image.open(video_path).convert('RGB')
        img_d = self.transform(image_input)
        audio = np.load(audio_path)
        audio_d = librosa.util.normalize(audio)
        img_real.append(img_d)
        aud_real.append(audio_d)

        min = index + skip
        max = (index + 2*skip)%self.__len__()
        if min > max:
            min = 0
            max = min + skip
        sample = random.sample(range(min, max), sample_exa)
        for ind in sample:
            image_input = Image.open(self.video_list[ind]).convert('RGB')
            img_d = self.transform(image_input)
            img_fake.append(img_d)

            audio = np.load(self.audio_list[ind])
            audio_d = librosa.util.normalize(audio)
            aud_fake.append(audio_d)

        aud_real = np.stack(aud_real, axis=0)
        aud_fake = np.stack(aud_fake, axis=0)
        img_real = np.stack(img_real, axis=0)
        img_fake = np.stack(img_fake, axis=0)
        aud_real = np.expand_dims(aud_real, 1)
        aud_fake = np.expand_dims(aud_fake, 1)
        return {
            'img_real': img_real,
            'img_fake': img_fake,
            'aud_real': aud_real,
            'aud_fake': aud_fake,
        }

    def __len__(self):
        return len(self.video_list)

    def get_video_list(self, dataset_path, audio_dataset_path, mode):
        video_feat_path = os.path.join(dataset_path)
        audio_feat_path = os.path.join(audio_dataset_path)
        id_list = [i for i in os.listdir(video_feat_path) if i.startswith('id')]
        video_path = []
        audio_path = []

        # Too little data, train and val set are the same in the demo code,
        # they will be distinguished in the official version.
        if mode == 'train':
            id_list_new = id_list[0:1]
            len_max = 100000
        elif mode == 'val':
            id_list_new = id_list[0:1]
            len_max = 15000
        else:
            id_list_new = id_list
            len_max = 15000
        cnt = 0
        id_list_new.sort()
        print(id_list_new)
        for id in tqdm(id_list_new):
            id_video_split_path = os.path.join(video_feat_path, id)
            video_list = os.listdir(id_video_split_path)
            cnt_video = 0
            for video in video_list:
                image_list_path = os.path.join(id_video_split_path, video)
                image_list = os.listdir(image_list_path)
                if cnt_video > 50:
                    break
                for image in image_list:
                    if cnt_video > 40:
                        break
                    audio_p = os.path.join(id, video, image.replace('.jpg','.npy')).replace('/', '_')
                    if image.endswith('.jpg') and os.path.exists(os.path.join(audio_feat_path, audio_p)):
                        video_path.append(os.path.join(image_list_path, image))
                        audio_path.append(os.path.join(audio_feat_path, audio_p))
                        cnt += 1
                        cnt_video += 1
            if cnt > len_max:
                break
        return video_path, audio_path
