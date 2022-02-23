import os

import cv2
import torch
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
from .models import BaseModel
from networks.networks import NetworksFactory, HumanModelRecovery
from utils.nmr import SMPLRenderer
from utils.detectors import PersonMaskRCNNDetector
import utils.cv_utils as cv_utils
import utils.util as util
from networks import U2GE_net


def save_img(body_mask, i, c=1):
    bg_img2 = body_mask.squeeze(0)
    if c == 3:
        bg_img2 = bg_img2.permute(1, 2, 0)
        bg_img2 = np.array(bg_img2.cpu())
        preds = cv2.cvtColor(bg_img2, cv2.COLOR_RGB2BGR)
    else:
        bg_img2 = bg_img2.squeeze(0)
        preds = np.array(bg_img2.cpu())
    preds = (preds + 1) / 2.0 * 255
    bg_img2 = preds.astype(np.uint8)
    cv2.imwrite(f'{i}.png', bg_img2)

class Imitator(BaseModel):
    def __init__(self, opt):
        super(Imitator, self).__init__(opt)
        self._name = 'Imitator'

        # prefetch variables
        self.src_info = None
        self.tsf_info = None
        self.first_cam = None

    def _create_networks(self):
        # 0. create bgnet
        if self._opt.bg_model != 'ORIGINAL':
            if torch.cuda.is_available():
                self.bgnet = self._create_bgnet().cuda()
            else:
                self.bgnet = self._create_bgnet()
        else:
            self.bgnet = self.generator.bg_model

    def _create_bgnet(self):
        net = NetworksFactory.get_by_name('deepfillv2', c_dim=4)
        self._load_params(net, './outputs/checkpoints/deepfillv2/net_epoch_50_id_G.pth', need_module=False)
        net.eval()
        return net


imitator=Imitator()
bg_net=imitator._create_bgnet().cuda()

ori_img = cv_utils.read_cv2_img(src_path)

# resize image and convert the color space from [0, 255] to [-1, 1]
img = cv_utils.transform_img(ori_img, (256,256), transpose=True) * 2 - 1.0
img = torch.tensor(img, dtype=torch.float32).cuda()[None, ...]


final_img = bg_net(img, masks=body_mask, only_x=True)


save_img(final_img,3,3)