from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import transforms
from models import D3net_dataset,D3net_model
import torch

import numpy as np
import matplotlib.pyplot as plt
import torch.backends.cudnn as cudnn
import argparse, os

parser = argparse.ArgumentParser(description="PyTorch D3Dnet")
parser.add_argument("--scale_factor", type=int, default=4, help="scale")
parser.add_argument("--test_dataset_dir", default='./data', type=str, help="test_dataset dir")
parser.add_argument("--model", default='./log/D3Dnet.pth.tar', type=str, help="checkpoint")
parser.add_argument("--inType", type=str, default='y', help="RGB input or y input")
parser.add_argument("--batchSize", type=int, default=1, help="Test batch size")
parser.add_argument("--gpu", type=int, default=0, help="Test batch size")
parser.add_argument("--datasets", type=str, default='', help="Test batch size")

global opt, model
opt = parser.parse_args()
torch.cuda.set_device(opt.gpu)

def demo_test(net, test_loader, scale_factor, save_path):

    with torch.no_grad():
        for idx_iter, (LR, SR_buicbic) in enumerate(test_loader):
            LR = Variable(LR).cuda()
            SR = net(LR)
            SR = torch.clamp(SR, 0, 1)

            if not os.path.exists(save_path):
                os.makedirs(save_path)
            ## save y images
            # SR_img = transforms.ToPILImage()(SR[0, :, :, :].cpu())
            # SR_img.save(save_path + '/pred2_' + str(idx_iter).rjust(8, '0') + '.jpg')

            ## save rgb images
            SR_buicbic[:, 0, :, :] = SR[:, 0, :, :].cpu()
            SR_rgb = (D3net_dataset.ycbcr2rgb(SR_buicbic[0,:,:,:].permute(2,1,0))).permute(2,1,0)
            SR_rgb = torch.clamp(SR_rgb, 0, 1)
            SR_img = transforms.ToPILImage()(SR_rgb)
            SR_img.save(save_path+ '/pred_' + str(idx_iter).rjust(8, '0') + '.jpg')


def D3net(img_path_list):
    net = D3net_model.Net(opt.scale_factor).cuda()
    model = torch.load(opt.model)
    net.load_state_dict(model['state_dict'])

    test_set = D3net_dataset.InferLoader(img_path_list, scale_factor=opt.scale_factor)
    test_loader = DataLoader(dataset=test_set, num_workers=1, batch_size=1, shuffle=False)


    save_path='/'.join(i for i in img_path_list[0].split('/')[:-1])+'_final'
    demo_test(net, test_loader, opt.scale_factor, save_path)

    return save_path