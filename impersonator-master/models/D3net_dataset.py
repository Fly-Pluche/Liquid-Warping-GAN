import torch
import numpy as np
from PIL import Image
from torchvision import transforms
from torch.utils.data.dataset import Dataset
import random
import matplotlib.pyplot as plt
import os
import math
import matplotlib.pyplot as plt


def random_crop(HR, LR, patch_size_lr, scale_factor): # HR: N*H*W
    _, _, h_hr, w_hr = HR.shape
    h_lr = h_hr // scale_factor
    w_lr = w_hr // scale_factor
    h_start_lr = random.randint(5, h_lr - patch_size_lr - 5)
    h_end_lr = h_start_lr + patch_size_lr
    w_start_lr = random.randint(5, w_lr - patch_size_lr - 5)
    w_end_lr = w_start_lr + patch_size_lr

    h_start = h_start_lr * scale_factor
    h_end = h_end_lr * scale_factor
    w_start = w_start_lr * scale_factor
    w_end = w_end_lr * scale_factor

    HR = HR[:, :, h_start:h_end, w_start:w_end]
    LR = LR[:, :, h_start_lr:h_end_lr, w_start_lr:w_end_lr]

    return HR, LR

def add_noise(img, n_std):
    return img + np.random.normal(0, n_std, img.shape)

def add_light(img, light, *paras, mode):
    if mode == 'point':
        x0, y0, radius = paras
        light_res = np.zeros(3, radius, radius)
        for i in range(radius):
            for j in range(radius):
                light_res[0, i, j, 0] = light * (1-math.sqrt((i-radius//2)**2 + (j-radius//2)**2)/radius)
                light_res[1, i, j, 0] = light * (1-math.sqrt((i-radius//2)**2 + (j-radius//2)**2)/radius)
                light_res[2, i, j, 0] = light * (1-math.sqrt((i-radius//2)**2 + (j-radius//2)**2)/radius)

        light_res = np.clip(light_res + img[:, x0-radius//2:x0+1+radius//2, y0-radius//2:y0+1+radius//2, :], 0, 255)
        img[:, x0-radius//2:x0+1+radius//2, y0-radius//2:y0+1+radius//2, :] = light_res
    return img

def rgb2ycbcr(img, only_y=True):
    '''same as matlab rgb2ycbcr
    only_y: only return Y channel
    Input:
        uint8, [0, 255]
        float, [0, 1]
    '''

    in_img_type = img.dtype
    img.astype(np.float32)
    if in_img_type != np.uint8:
        img *= 255.
    # convert
    if only_y:
        rlt = np.dot(img, [65.481, 128.553, 24.966]) / 255.0 + 16.0
    else:
        rlt = np.matmul(img, [[65.481, -37.797, 112.0], [128.553, -74.203, -93.786],
                              [24.966, 112.0, -18.214]]) / 255.0 + [16, 128, 128]
    if in_img_type == np.uint8:
        rlt = rlt.round()
    else:
        rlt /= 255.
    return rlt.astype(in_img_type)

def ycbcr2rgb(ycbcr_img):
    ycbcr_img = ycbcr_img.numpy()
    in_img_type = ycbcr_img.dtype
    if in_img_type != np.uint8:
        ycbcr_img *= 255.
    mat = np.array(
        [[65.481, 128.553, 24.966],
         [-37.797, -74.203, 112.0],
         [112.0, -93.786, -18.214]])
    mat_inv = np.linalg.inv(mat)
    offset = np.array([16, 128, 128])
    rgb_img = np.zeros(ycbcr_img.shape)
    for x in range(ycbcr_img.shape[0]):
        for y in range(ycbcr_img.shape[1]):
            rgb_img[x, y, :] = np.maximum(0, np.minimum(255,np.round(np.dot(mat_inv, ycbcr_img[x, y, :] - offset) * 255.0)))
    return torch.from_numpy(np.ascontiguousarray(rgb_img.astype(np.float32)/255))


class TrainSetLoader(Dataset):
    def __init__(self, dataset_dir, scale_factor, inType='y'):
        super(TrainSetLoader).__init__()
        self.scale_factor = scale_factor
        self.dir = dataset_dir
        with open(dataset_dir+'/sep_trainlist.txt', 'r') as f:
            self.train_list = f.read().splitlines()
        self.tranform = augumentation()
        self.inType = inType
    def __getitem__(self, idx):
        HR = []
        LR = []
        for i in range(7):
            img_hr = Image.open(self.dir + '/sequences/' + self.train_list[idx] + '/im' + str(i + 1) + '.png')
            img_lr = Image.open(self.dir + '/LR_x4/' + self.train_list[idx] + '/im' + str(i + 1) + '.png')
            img_hr = np.array(img_hr, dtype=np.float32)/255.0
            img_lr = np.array(img_lr, dtype=np.float32)/255.0
            if self.inType == 'y':
                img_hr = rgb2ycbcr(img_hr, only_y=True)[np.newaxis,:]
                img_lr = rgb2ycbcr(img_lr, only_y=True)[np.newaxis,:]
            if self.inType == 'RGB':
                img_hr = img_hr.transpose(2,0,1)
                img_lr = img_lr.transpose(2,0,1)
            HR.append(img_hr)
            LR.append(img_lr)

        HR = np.stack(HR, 1)
        LR = np.stack(LR, 1)

        HR, LR = random_crop(HR, LR, 32, self.scale_factor)
        HR, LR = self.tranform(HR, LR)

        HR = torch.from_numpy(np.ascontiguousarray(HR))
        LR = torch.from_numpy(np.ascontiguousarray(LR))

        return LR, HR
    def __len__(self):
        return len(self.train_list)

class ValidSetLoader(Dataset):
    def __init__(self, dataset_dir, scale_factor, inType='y'):
        super(TrainSetLoader).__init__()
        self.scale_factor = scale_factor
        self.dir = dataset_dir
        with open(dataset_dir+'/sep_testlist.txt', 'r') as f:
            self.train_list = f.read().splitlines()
        self.tranform = augumentation()
        self.inType = inType
    def __getitem__(self, idx):
        HR = []
        LR = []
        for i in range(7):
            img_hr = Image.open(self.dir + '/sequences/' + self.train_list[idx] + '/im' + str(i + 1) + '.png')
            img_lr = Image.open(self.dir + '/LR_x4/' + self.train_list[idx] + '/im' + str(i + 1) + '.png')
            img_hr = np.array(img_hr, dtype=np.float32)/255.0
            img_lr = np.array(img_lr, dtype=np.float32)/255.0
            if self.inType == 'y':
                img_hr = rgb2ycbcr(img_hr, only_y=True)[np.newaxis,:]
                img_lr = rgb2ycbcr(img_lr, only_y=True)[np.newaxis,:]
            if self.inType == 'RGB':
                img_hr = img_hr.transpose(2,0,1)
                img_lr = img_lr.transpose(2,0,1)
            HR.append(img_hr)
            LR.append(img_lr)

        HR = np.stack(HR, 1)
        LR = np.stack(LR, 1)

        HR, LR = random_crop(HR, LR, 32, self.scale_factor)
        HR, LR = self.tranform(HR, LR)

        HR = torch.from_numpy(np.ascontiguousarray(HR))
        LR = torch.from_numpy(np.ascontiguousarray(LR))

        return LR, HR
    def __len__(self):
        return len(self.train_list)

class TestSetLoader(Dataset):
    def __init__(self, img_list, scale_factor):
        super(TestSetLoader).__init__()
        self.dataset_dir = ''
        self.upscale_factor = scale_factor
        self.img_list = img_list
        self.totensor = transforms.ToTensor()
    def __getitem__(self, idx):
        dataset_dir='/'.join(i for i in self.img_list[0].split('/')[:-1])
        LR = []
        for idx_frame in range(idx - 3, idx + 4):
            if idx_frame < 0:
                idx_frame = 0
            if idx_frame > len(self.img_list) - 1:
                idx_frame = len(self.img_list) - 1
            img_LR = Image.open(dataset_dir + '/pred_' + str(idx_frame).rjust(8, '0') + '.jpg')
            if idx_frame == idx:
                # h, w, c = 256,256,3
                SR_buicbic = np.array(img_LR, dtype=np.float32) / 255.0
                SR_buicbic = rgb2ycbcr(SR_buicbic, only_y=False).transpose(2, 0, 1)
            img_LR = np.array(img_LR, dtype=np.float32) / 255.0
            img_LR = rgb2ycbcr(img_LR, only_y=True)[np.newaxis,:]

            LR.append(img_LR)

        LR = np.stack(LR, 1)

        C, N, H, W= LR.shape
        H = math.floor(H / self.upscale_factor / 4) * self.upscale_factor * 4
        W = math.floor(W / self.upscale_factor / 4) * self.upscale_factor * 4

        SR_buicbic = SR_buicbic[:, :H, :W]
        LR = LR[:, :, :H // self.upscale_factor, :W // self.upscale_factor]

        LR = torch.from_numpy(np.ascontiguousarray(LR))
        SR_buicbic = torch.from_numpy(np.ascontiguousarray(SR_buicbic))
        return LR,SR_buicbic

    def __len__(self):
        return len(self.img_list)

class InferLoader(Dataset):

    def __init__(self, img_list, scale_factor):
        super(TestSetLoader).__init__()
        self.dataset_dir = ''
        self.upscale_factor = scale_factor
        self.img_list = img_list
        self.totensor = transforms.ToTensor()
    def __getitem__(self, idx):
        dataset_dir='/'.join(i for i in self.img_list[0].split('/')[:-1])
        LR = []
        for idx_frame in range(idx - 3, idx + 4):
            if idx_frame < 0:
                idx_frame = 0
            if idx_frame > len(self.img_list) - 1:
                idx_frame = len(self.img_list) - 1
            img_LR_o =Image.open(dataset_dir + '/pred_' + str(idx_frame).rjust(8, '0') + '.jpg')
            img_LR = np.array(img_LR_o, dtype=np.float32) / 255.0
            if idx_frame == idx:
                h, w, c = img_LR.shape
                SR_buicbic = np.array(img_LR_o.resize((w*self.upscale_factor, h*self.upscale_factor), Image.BICUBIC), dtype=np.float32) / 255.0
                SR_buicbic = rgb2ycbcr(SR_buicbic, only_y=False).transpose(2, 0, 1)
            img_LR = rgb2ycbcr(img_LR, only_y=True)[np.newaxis,:]

            LR.append(img_LR)
        LR = np.stack(LR, 1)


        LR = torch.from_numpy(np.ascontiguousarray(LR))
        SR_buicbic = torch.from_numpy(np.ascontiguousarray(SR_buicbic))
        return LR, SR_buicbic

    def __len__(self):
        return len(self.img_list)

class TestSetLoader_Vimeo(Dataset):
    def __init__(self, dataset_dir, video_name, scale_factor, inType='y'):
        super(TestSetLoader).__init__()
        self.upscale_factor = scale_factor
        self.dir = dataset_dir
        self.video_name = video_name
        self.img_list = os.listdir(self.dir + '/sequences/' + self.video_name)
        self.inType = inType

    def __getitem__(self, idx):
        HR = []
        LR = []
        for idx_frame in range(idx - 3, idx + 4):
            if idx_frame < 0:
                idx_frame = 0
            if idx_frame > len(self.img_list) - 1:
                idx_frame = len(self.img_list) - 1
            img_hr = Image.open(self.dir + '/sequences/' + self.video_name + '/im' + str(idx_frame + 1) + '.png')
            img_lr = Image.open(self.dir + '/LR_x4/' + self.video_name + '/im' + str(idx_frame + 1) + '.png')
            img_hr = np.array(img_hr, dtype=np.float32) / 255.0
            if idx_frame == idx:
                h, w, c = img_hr.shape
                SR_buicbic = np.array(img_lr.resize((w, h), Image.BICUBIC), dtype=np.float32) / 255.0
                SR_buicbic = rgb2ycbcr(SR_buicbic, only_y=False).transpose(2, 0, 1)
            img_lr = np.array(img_lr, dtype=np.float32) / 255.0
            if self.inType == 'y':
                img_hr = rgb2ycbcr(img_hr, only_y=True)[np.newaxis, :]
                img_lr = rgb2ycbcr(img_lr, only_y=True)[np.newaxis, :]
            if self.inType == 'RGB':
                img_hr = img_hr.transpose(2, 0, 1)
                img_lr = img_lr.transpose(2, 0, 1)
            HR.append(img_hr)
            LR.append(img_lr)

        HR = np.stack(HR, 1)
        LR = np.stack(LR, 1)

        C, N, H, W = HR.shape
        H = math.floor(H / self.upscale_factor / 4) * self.upscale_factor * 4
        W = math.floor(W / self.upscale_factor / 4) * self.upscale_factor * 4
        HR = HR[:, :, :H, :W]
        SR_buicbic = SR_buicbic[:, :H, :W]
        LR = LR[:, :, :H // self.upscale_factor, :W // self.upscale_factor]

        HR = torch.from_numpy(np.ascontiguousarray(HR))
        LR = torch.from_numpy(np.ascontiguousarray(LR))
        SR_buicbic = torch.from_numpy(np.ascontiguousarray(SR_buicbic))

        return LR, HR, SR_buicbic

    def __len__(self):
        return len(self.img_list)

class augumentation(object):
    def __call__(self, input, target):
        if random.random()<0.5:
            input = input[::-1, :, :]
            target = target[::-1, :, :]
        if random.random()<0.5:
            input = input[:, ::-1, :]
            target = target[:, ::-1, :]
        if random.random()<0.5:
            input = input.transpose(0, 1, 3, 2)#C N H W
            target = target.transpose(0, 1, 3, 2)
        return input, target
