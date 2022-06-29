from torch.utils.data.dataset import Dataset

import os
import torch
import fnmatch
import numpy as np
import pdb
import torchvision.transforms as transforms
from PIL import Image
import random
import torch.nn.functional as F



class RandomScaleCrop(object):
    """
    Credit to Jialong Wu from https://github.com/lorenmt/mtan/issues/34.
    """
    def __init__(self, scale=[1.0, 1.2, 1.5]):
        self.scale = scale

    def __call__(self, img, label, depth, normal):
        height, width = img.shape[-2:]
        sc = self.scale[random.randint(0, len(self.scale) - 1)]
        h, w = int(height / sc), int(width / sc)
        i = random.randint(0, height - h)
        j = random.randint(0, width - w)
        # pdb.set_trace()
        img_ = F.interpolate(img[None, :, i:i + h, j:j + w], size=(height, width), mode='bilinear', align_corners=True).squeeze(0)
        label_ = F.interpolate(label[None, None, i:i + h, j:j + w], size=(height, width), mode='nearest').squeeze(0).squeeze(0)
        depth_ = F.interpolate(depth[None, :, i:i + h, j:j + w], size=(height, width), mode='nearest').squeeze(0)
        normal_ = F.interpolate(normal[None, :, i:i + h, j:j + w], size=(height, width), mode='bilinear', align_corners=True).squeeze(0)
        _sc = sc
        _h, _w, _i, _j = h, w, i, j

        return img_, label_, depth_ / sc, normal_, torch.tensor([_sc, _h, _w, _i, _j, height, width])


class NYUv2(Dataset):
    """
    This file is directly modified from https://pytorch.org/docs/stable/torchvision/datasets.html
    """
    def __init__(self, root, train=True, index=None):
        self.train = train
        self.root = os.path.expanduser(root)

        # R\read the data file
        if train:
            self.data_path = root + '/train'
        else:
            self.data_path = root + '/val'

        # calculate data length
        self.data_len = len(fnmatch.filter(os.listdir(self.data_path + '/image'), '*.npy'))

    def __getitem__(self, index):
        # get image name from the pandas df
        image = torch.from_numpy(np.moveaxis(np.load(self.data_path + '/image/{:d}.npy'.format(index)), -1, 0))
        semantic = torch.from_numpy(np.load(self.data_path + '/label/{:d}.npy'.format(index)))
        depth = torch.from_numpy(np.moveaxis(np.load(self.data_path + '/depth/{:d}.npy'.format(index)), -1, 0))
        normal = torch.from_numpy(np.moveaxis(np.load(self.data_path + '/normal/{:d}.npy'.format(index)), -1, 0))

        if self.train:
            return image.type(torch.FloatTensor), semantic.type(torch.FloatTensor), depth.type(torch.FloatTensor), normal.type(torch.FloatTensor), index
        else:
            return image.type(torch.FloatTensor), semantic.type(torch.FloatTensor), depth.type(torch.FloatTensor), normal.type(torch.FloatTensor)
    def __len__(self):
        return self.data_len


class NYUv2_crop(Dataset):
    """
    This file is directly modified from https://pytorch.org/docs/stable/torchvision/datasets.html
    """
    def __init__(self, root, train=True, index=None, augmentation=False, aug_twice=False):
        self.train = train
        self.root = os.path.expanduser(root)
        self.augmentation = augmentation
        self.aug_twice = aug_twice


        # R\read the data file
        if train:
            self.data_path = root + '/train'
        else:
            self.data_path = root + '/val'

        # calculate data length
        self.data_len = len(fnmatch.filter(os.listdir(self.data_path + '/image'), '*.npy'))

    def __getitem__(self, index):
        # get image name from the pandas df
        image = torch.from_numpy(np.moveaxis(np.load(self.data_path + '/image/{:d}.npy'.format(index)), -1, 0))
        semantic = torch.from_numpy(np.load(self.data_path + '/label/{:d}.npy'.format(index)))
        depth = torch.from_numpy(np.moveaxis(np.load(self.data_path + '/depth/{:d}.npy'.format(index)), -1, 0))
        normal = torch.from_numpy(np.moveaxis(np.load(self.data_path + '/normal/{:d}.npy'.format(index)), -1, 0))
        
        if self.augmentation and self.aug_twice == False:
            image, semantic, depth, normal, _ = RandomScaleCrop()(image, semantic, depth, normal)
            return image.type(torch.FloatTensor), semantic.type(torch.FloatTensor), depth.type(torch.FloatTensor), normal.type(torch.FloatTensor), index

        elif self.augmentation and self.aug_twice:
            image, semantic, depth, normal, _ = RandomScaleCrop()(image, semantic, depth, normal)
            image1, semantic1, depth1, normal1, trans_params = RandomScaleCrop()(image, semantic, depth, normal)
            return image.type(torch.FloatTensor), semantic.type(torch.FloatTensor), depth.type(torch.FloatTensor), normal.type(torch.FloatTensor), index, image1.type(torch.FloatTensor), semantic1.type(torch.FloatTensor), depth1.type(torch.FloatTensor), normal1.type(torch.FloatTensor), trans_params
        if self.train:
            return image.type(torch.FloatTensor), semantic.type(torch.FloatTensor), depth.type(torch.FloatTensor), normal.type(torch.FloatTensor), index
        else:
            return image.type(torch.FloatTensor), semantic.type(torch.FloatTensor), depth.type(torch.FloatTensor), normal.type(torch.FloatTensor)
    def __len__(self):
        return self.data_len
