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


class Imitator(BaseModel):
    def __init__(self, opt):
        super(Imitator, self).__init__(opt)
        self._name = 'Imitator'
        self._create_networks()
        # prefetch variables
        self.src_info = None
        self.tsf_info = None
        self.first_cam = None

    def _create_networks(self):
        # 0. create generator
        if torch.cuda.is_available():
            self.generator = self._create_generator().cuda()
        else:
            self.generator = self._create_generator()

        # 0. create bgnet
        if self._opt.bg_model != 'ORIGINAL':
            if torch.cuda.is_available():
                self.bgnet = self._create_bgnet().cuda()
            else:
                self.bgnet = self._create_bgnet()
        else:
            self.bgnet = self.generator.bg_model

        # 2. create hmr

        if torch.cuda.is_available():
            self.hmr = self._create_hmr().cuda()
        else:
            self.hmr = self._create_hmr()

        # 3. create render
        self.render = SMPLRenderer(image_size=self._opt.image_size, tex_size=self._opt.tex_size,
                                   has_front=self._opt.front_warp, fill_back=False).cuda()
        # 4. pre-processor
        if self._opt.has_detector:
            self.detector = PersonMaskRCNNDetector(ks=self._opt.bg_ks, threshold=0.5, to_gpu=True)
        else:
            self.detector = None

    def load_consistent_state_dict(self,pretrained_dict, model):
        model_dict = model.state_dict()
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict and 'coarse' in k}
        # 2. overwrite entries in the existing state dict
        model_dict.update(pretrained_dict)
        # 3. load the new state dict
        model.load_state_dict(model_dict)

    def _create_bgnet(self):

        whole_model_path = 'outputs/checkpoints/deepfillv2/epoch_0_ckpt.pth'
        # whole_model_path = 'latest_ckpt.pth'
        net = torch.load(whole_model_path)['netG_state_dict']
        net.eval()
        return net

    def _create_generator(self):
        net = NetworksFactory.get_by_name(self._opt.gen_name, bg_dim=4, src_dim=3+self._G_cond_nc,
                                          tsf_dim=3+self._G_cond_nc, repeat_num=self._opt.repeat_num)

        if self._opt.load_path:
            self._load_params(net, self._opt.load_path)
        elif self._opt.load_epoch > 0:
            self._load_network(net, 'G', self._opt.load_epoch)
        else:
            raise ValueError('load_path {} is empty and load_epoch {} is 0'.format(
                self._opt.load_path, self._opt.load_epoch))

        net.eval()
        return net

    def _create_hmr(self):
        hmr = HumanModelRecovery(self._opt.smpl_model)
        saved_data = torch.load(self._opt.hmr_model)
        hmr.load_state_dict(saved_data)
        hmr.eval()
        return hmr

    def visualize(self, *args, **kwargs):
        visualizer = args[0]
        if visualizer is not None:
            for key, value in kwargs.items():
                visualizer.vis_named_img(key, value)

    @torch.no_grad()
    def personalize(self, src_path, src_smpl=None, output_path='', visualizer=None):

        ori_img = cv_utils.read_cv2_img(src_path)

        # resize image and convert the color space from [0, 255] to [-1, 1]
        img = cv_utils.transform_img(ori_img, self._opt.image_size, transpose=True) * 2 - 1.0
        img = torch.tensor(img, dtype=torch.float32).cuda()[None, ...]

        if src_smpl is None:
            img_hmr = cv_utils.transform_img(ori_img, 224, transpose=True) * 2 - 1.0
            img_hmr = torch.tensor(img_hmr, dtype=torch.float32).cuda()[None, ...]
            src_smpl = self.hmr(img_hmr)   # src_smpl = [1,85]
        else:
            src_smpl = torch.tensor(src_smpl, dtype=torch.float32).cuda()[None, ...]

        # source process, {'theta', 'cam', 'pose', 'shape', 'verts', 'j2d', 'j3d'}
        src_info = self.hmr.get_details(src_smpl)
        src_f2verts, src_fim, src_wim = self.render.render_fim_wim(src_info['cam'], src_info['verts'])
        # src_f2pts = src_f2verts[:, :, :, 0:2]
        src_info['fim'] = src_fim
        src_info['wim'] = src_wim
        src_info['cond'], _ = self.render.encode_fim(src_info['cam'], src_info['verts'], fim=src_fim, transpose=True)
        src_info['f2verts'] = src_f2verts
        src_info['p2verts'] = src_f2verts[:, :, :, 0:2]
        src_info['p2verts'][:, :, :, 1] *= -1

        if self._opt.only_vis:
            src_info['p2verts'] = self.render.get_vis_f2pts(src_info['p2verts'], src_fim)
        # add image to source info
        src_info['img'] = img
        src_info['image'] = ori_img

        """original 得到body_mask"""
        # 2. process the src inputs
        if self.detector is not None:
            bbox, body_mask = self.detector.inference(img[0])
            bg_mask = 1 - body_mask
        else:
            # bg is 1, ft is 0
            bg_mask = util.morph(src_info['cond'][:, -1:, :, :], ks=self._opt.bg_ks, mode='erode')
            body_mask = 1 - bg_mask

        def save_img(body_mask,i,c=1):
            bg_img2 = body_mask.squeeze(0)
            if c==3:
                bg_img2 = bg_img2.permute(1, 2, 0)
                bg_img2 = np.array(bg_img2.cpu())
                preds = cv2.cvtColor(bg_img2, cv2.COLOR_RGB2BGR)
            else:
                bg_img2=bg_img2.squeeze(0)
                preds = np.array(bg_img2.cpu())
            # preds = (preds + 1) / 2.0 * 255
            preds = (preds + 1)*127.5
            bg_img2 = preds.astype(np.uint8)
            cv2.imwrite(f'{i}.png',bg_img2)

        if self._opt.bg_model != 'ORIGINAL':

            """U2GE-net"""
            ft_mask_img=U2GE_net.net(src_path)
            ft_mask_img[ft_mask_img<0.1]=0
            ft_mask_img[ft_mask_img>=0.1]=1

            """膨胀"""
            body_mask=ft_mask_img*255
            body_mask=np.array(body_mask.cpu())
            body_mask = cv2.dilate(body_mask, None, iterations=10)/255
            body_mask=torch.from_numpy(body_mask)

            ft_mask_img=body_mask.unsqueeze(0).cuda()
            recon_imgs,_ = self.bgnet(img, masks=ft_mask_img)
            src_info['bg']=recon_imgs * ft_mask_img + img * (1 - ft_mask_img)
            """分块恢复"""
            # img_1=img[:,:,:128,:128]
            # img_2=img[:,:,128:,128:]
            # img_3=img[:,:,:128,128:]
            # img_4=img[:,:,128:,:128]
            #
            #
            # body_mask_1=body_mask[:,:,:128,:128]
            # body_mask_2=body_mask[:,:,128:,128:]
            # body_mask_3=body_mask[:,:,:128,128:]
            # body_mask_4=body_mask[:,:,128:,:128]
            #
            # # img= self.bgnet(img, masks=body_mask, only_x=True)
            # img_1= self.bgnet(img_1, masks=body_mask_1, only_x=True)
            # img_2 = self.bgnet(img_2, masks=body_mask_2, only_x=True)
            # img_3 = self.bgnet(img_3, masks=body_mask_3, only_x=True)
            # img_4 = self.bgnet(img_4, masks=body_mask_4, only_x=True)
            #
            # img_5=torch.cat((img_1,img_3),3)
            # img_6=torch.cat((img_4,img_2),3)
            #
            #
            # src_info['bg']=torch.cat((img_5,img_6),2)
            # save_img(src_info['bg'],7)
            #
            # body_mask_1=src_info['bg']*255
            # body_mask_1=body_mask_1.permute(0,2,3,1)
            # body_mask_1=np.array(body_mask_1.cpu())
            # body_mask_1=body_mask_1.squeeze(0)
            # a=body_mask_1.astype(np.uint8)
            # cv2.imwrite('背景图.png',a)

            """腐蚀"""
            # body_mask=torch.from_numpy(body_mask/255)
            # body_mask=body_mask.unsqueeze(0).unsqueeze(0).cuda()
            # src_info['bg'] = self.bgnet(src_info['bg'], masks=body_mask,only_x=True)
            # body_mask=body_mask*255
            # body_mask=np.array(body_mask.cpu())
            # body_mask=body_mask.squeeze(0).squeeze(0)
            # body_mask = cv2.erode(body_mask, None, iterations=7)
            # a=cv2.erode(body_mask, None, iterations=5)
            # a=a.astype(np.uint8)
            # cv2.imwrite('腐蚀2.png',a)
            # body_mask=torch.from_numpy(body_mask/255)
            # body_mask=body_mask.unsqueeze(0).unsqueeze(0).cuda()
            # src_info['bg'] = self.bgnet(src_info['bg'], masks=body_mask,only_x=True)

        """U2GE-net"""
        # ft_mask_img=U2GE_net.net(src_path)
        # ft_mask_img[ft_mask_img<0.1]=0
        # ft_mask_img[ft_mask_img>=0.1]=1
        # ft_mask_img=ft_mask_img.unsqueeze(0)
        # ft_mask_img = ft_mask_img.cuda()
        # ft_mask_img = util.morph(ft_mask_img, ks=self._opt.ft_ks, mode='erode')
        # src_inputs = torch.cat([img * ft_mask_img, src_info['cond']], dim=1)

        """Original"""
        ft_mask = 1 - util.morph(src_info['cond'][:, -1:, :, :], ks=self._opt.ft_ks, mode='erode')
        src_inputs = torch.cat([img * ft_mask, src_info['cond']], dim=1)

        src_info['feats'] = self.generator.encode_src(src_inputs) # return img_outs, mask_outs

        self.src_info = src_info

        if visualizer is not None:
            visualizer.vis_named_img('src', img)
            visualizer.vis_named_img('bg', src_info['bg'])

        if output_path:
            cv_utils.save_cv2_img(src_info['image'], output_path, image_size=self._opt.image_size)

    @torch.no_grad()
    def _extract_smpls(self, input_file):
        img = cv_utils.read_cv2_img(input_file)
        img = cv_utils.transform_img(img, image_size=224) * 2 - 1.0  # hmr receive [-1, 1]
        img = img.transpose((2, 0, 1))
        img = torch.tensor(img, dtype=torch.float32).cuda()[None, ...]
        theta = self.hmr(img)[-1]

        return theta

    @torch.no_grad()
    def inference(self, tgt_paths, tgt_smpls=None, cam_strategy='smooth',
                  output_dir='./output/result', visualizer=None, verbose=True,new_bg=None):

        length = len(tgt_paths)

        process_bar = tqdm(range(length)) if verbose else range(length)
        outputs=[]

        for t in process_bar:
            tgt_path = tgt_paths[t]
            tgt_smpl = tgt_smpls[t] if tgt_smpls is not None else None

            tsf_inputs = self.transfer_params(tgt_path, tgt_smpl, cam_strategy, t=t)
            ## 得到转化的图片
            if new_bg:
                preds = self.forward(tsf_inputs, self.tsf_info['T'], new_bg)
            else:
                preds = self.forward(tsf_inputs, self.tsf_info['T'])

            if visualizer is not None:
                gt = cv_utils.transform_img(self.tsf_info['image'], image_size=self._opt.image_size, transpose=True)
                visualizer.vis_named_img('pred_' + cam_strategy, preds)
                visualizer.vis_named_img('gt', gt[None, ...], denormalize=False)

            preds = preds[0].permute(1, 2, 0)
            preds = preds.cpu().numpy()
            outputs.append(preds)

            if output_dir:
                filename = os.path.split(tgt_path)[-1]

                # if new_bg != None:
                #     preds=(preds+1)*127.5
                #     preds=preds.astype(np.int8)
                #     cv2.imwrite(os.path.join(output_dir, 'pred_' + filename),preds)
                # else:
                cv_utils.save_cv2_img(preds, os.path.join(output_dir, 'pred_' + filename), normalize=True)

        return outputs,output_dir
    
    @torch.no_grad()
    def inference_by_smpls(self, tgt_smpls, cam_strategy='smooth', output_dir='', visualizer=None,new_bg=None):
        length = len(tgt_smpls)

        outputs = []
        for t in tqdm(range(length)):
            tgt_smpl = tgt_smpls[t] if tgt_smpls is not None else None

            tsf_inputs = self.transfer_params_by_smpl(tgt_smpl, cam_strategy, t=t)
            if new_bg:
                preds = self.forward(tsf_inputs, self.tsf_info['T'],new_bg)
            else:
                preds = self.forward(tsf_inputs, self.tsf_info['T'])

            if visualizer is not None:
                gt = cv_utils.transform_img(self.tsf_info['image'], image_size=self._opt.image_size, transpose=True)
                visualizer.vis_named_img('pred_' + cam_strategy, preds)
                visualizer.vis_named_img('gt', gt[None, ...], denormalize=False)

            preds = preds[0].permute(1, 2, 0)
            preds = preds.cpu().detach().numpy()
            outputs.append(preds)

            if output_dir:
                cv_utils.save_cv2_img(preds, os.path.join(output_dir, 'pred_%.8d.jpg' % t), normalize=True)

        return outputs

    def swap_smpl(self, src_cam, src_shape, tgt_smpl, cam_strategy='smooth'):
        tgt_cam = tgt_smpl[:, 0:3].contiguous()
        pose = tgt_smpl[:, 3:75].contiguous()

        # TODO, need more tricky ways
        if cam_strategy == 'smooth':

            cam = src_cam.clone()
            delta_xy = tgt_cam[:, 1:] - self.first_cam[:, 1:]
            cam[:, 1:] += delta_xy

        elif cam_strategy == 'source':
            cam = src_cam
        else:
            cam = tgt_cam

        tsf_smpl = torch.cat([cam, pose, src_shape], dim=1)

        return tsf_smpl

    def transfer_params_by_smpl(self, tgt_smpl, cam_strategy='smooth', t=0):
        # get source info
        src_info = self.src_info

        if isinstance(tgt_smpl, np.ndarray):
            tgt_smpl = torch.tensor(tgt_smpl).float().cuda()[None, ...]

        if t == 0 and cam_strategy == 'smooth':
            self.first_cam = tgt_smpl[:, 0:3].clone()

        # get transfer smpl
        tsf_smpl = self.swap_smpl(src_info['cam'], src_info['shape'], tgt_smpl, cam_strategy=cam_strategy)
        # transfer process, {'theta', 'cam', 'pose', 'shape', 'verts', 'j2d', 'j3d'}
        tsf_info = self.hmr.get_details(tsf_smpl)

        tsf_f2verts, tsf_fim, tsf_wim = self.render.render_fim_wim(tsf_info['cam'], tsf_info['verts'])
        # src_f2pts = src_f2verts[:, :, :, 0:2]
        tsf_info['fim'] = tsf_fim
        tsf_info['wim'] = tsf_wim
        tsf_info['cond'], _ = self.render.encode_fim(tsf_info['cam'], tsf_info['verts'], fim=tsf_fim, transpose=True)
        # tsf_info['sil'] = util.morph((tsf_fim != -1).float(), ks=self._opt.ft_ks, mode='dilate')

        T = self.render.cal_bc_transform(src_info['p2verts'], tsf_fim, tsf_wim)
        tsf_img = F.grid_sample(src_info['img'], T)
        tsf_inputs = torch.cat([tsf_img, tsf_info['cond']], dim=1)

        # add target image to tsf info
        tsf_info['tsf_img'] = tsf_img
        tsf_info['T'] = T

        self.tsf_info = tsf_info

        return tsf_inputs

    def img_center(self,tgt_path):
        mask=U2GE_net.net(tgt_path,1)
        mask = mask.astype(np.uint8)
        # mask=0~255
        mask = mask[:, :, 0]*255
        # mask两个维度
        cnts = (cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE))[1]
        # print(cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE))

        c = sorted(cnts, key=cv2.contourArea, reverse=True)[0]  # contourArea这计算图像轮廓的面积  从大到小排,取最大

        rect = cv2.boundingRect(c)  # minAreaRect就是求出在上述点集下的最小面积矩形

        x = rect[0] - 10
        y = rect[1] - 10
        wight = rect[2]
        height = rect[3]
        if height >= wight:
            x = x - (height - wight) / 2
            wight = height

        else:
            y = y - (wight - height) / 2
            height = wight

        if y < 0:
            y = 0
        if x < 0:
            x = 0

        img = cv2.imread(tgt_path)
        img = img[int(y):int(y + height), int(x):int(x + wight), :]
        ori_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return ori_img

    def transfer_params(self, tgt_path, tgt_smpl=None, cam_strategy='smooth', t=0):

        ori_img = cv_utils.read_cv2_img(tgt_path)
        # ori_img=self.img_center(tgt_path)

        if tgt_smpl is None:
            img_hmr = cv_utils.transform_img(ori_img, 224, transpose=True) * 2 - 1.0
            img_hmr = torch.tensor(img_hmr, dtype=torch.float32).cuda()[None, ...]
            tgt_smpl = self.hmr(img_hmr)
        else:
            if isinstance(tgt_smpl, np.ndarray):
                tgt_smpl = torch.tensor(tgt_smpl, dtype=torch.float32).cuda()[None, ...]

        tsf_inputs = self.transfer_params_by_smpl(tgt_smpl=tgt_smpl, cam_strategy=cam_strategy, t=t)
        self.tsf_info['image'] = ori_img

        return tsf_inputs

    # @torch.no_grad()
    # def transfer_params(self, tgt_path, tgt_smpl=None, cam_strategy='smooth', t=0):
    #     # get source info
    #     src_info = self.src_info
    #
    #     ori_img = cv_utils.read_cv2_img(tgt_path)
    #     if tgt_smpl is None:
    #         img_hmr = cv_utils.transform_img(ori_img, 224, transpose=True) * 2 - 1.0
    #         img_hmr = torch.tensor(img_hmr, dtype=torch.float32).cuda()[None, ...]
    #         tgt_smpl = self.hmr(img_hmr)
    #     else:
    #         tgt_smpl = torch.tensor(tgt_smpl, dtype=torch.float32).cuda()[None, ...]
    #
    #     if t == 0 and cam_strategy == 'smooth':
    #         self.first_cam = tgt_smpl[:, 0:3].clone()
    #
    #     # get transfer smpl
    #     tsf_smpl = self.swap_smpl(src_info['cam'], src_info['shape'], tgt_smpl, cam_strategy=cam_strategy)
    #     # transfer process, {'theta', 'cam', 'pose', 'shape', 'verts', 'j2d', 'j3d'}
    #     tsf_info = self.hmr.get_details(tsf_smpl)
    #
    #     tsf_f2verts, tsf_fim, tsf_wim = self.render.render_fim_wim(tsf_info['cam'], tsf_info['verts'])
    #     # src_f2pts = src_f2verts[:, :, :, 0:2]
    #     tsf_info['fim'] = tsf_fim
    #     tsf_info['wim'] = tsf_wim
    #     tsf_info['cond'], _ = self.render.encode_fim(tsf_info['cam'], tsf_info['verts'], fim=tsf_fim, transpose=True)
    #     # tsf_info['sil'] = util.morph((tsf_fim != -1).float(), ks=self._opt.ft_ks, mode='dilate')
    #
    #     T = self.render.cal_bc_transform(src_info['p2verts'], tsf_fim, tsf_wim)
    #     tsf_img = F.grid_sample(src_info['img'], T)
    #     tsf_inputs = torch.cat([tsf_img, tsf_info['cond']], dim=1)
    #
    #     # add target image to tsf info
    #     tsf_info['tsf_img'] = tsf_img
    #     tsf_info['image'] = ori_img
    #     tsf_info['T'] = T
    #
    #     self.tsf_info = tsf_info
    #
    #     return tsf_inputs


    def forward(self, tsf_inputs, T,new_bg=None):
        if new_bg:
            _,_,h,w=tsf_inputs.size()
            new_bg=cv2.imread(new_bg)
            new_bg=cv2.cvtColor(new_bg,cv2.COLOR_BGR2RGB)
            new_bg=(cv2.resize(new_bg,(h,w))/127.5)-1
            bg_img=torch.from_numpy(new_bg).permute(2,0,1).unsqueeze(0).cuda()
        else:
            bg_img = self.src_info['bg']

        src_encoder_outs, src_resnet_outs = self.src_info['feats']

        tsf_color, tsf_mask = self.generator.inference(src_encoder_outs, src_resnet_outs, tsf_inputs, T)

        pred_imgs = tsf_mask * bg_img + (1 - tsf_mask) * tsf_color
        if self._opt.front_warp:
            pred_imgs = self.warp_front(pred_imgs, tsf_mask)

        return pred_imgs


    def warp_front(self, preds, mask):
        front_mask = self.render.encode_front_fim(self.tsf_info['fim'], transpose=True, front_fn=True)
        preds = (1 - front_mask) * preds + self.tsf_info['tsf_img'] * front_mask * (1 - mask)
        # preds = torch.clamp(preds + self.tsf_info['tsf_img'] * front_mask, -1, 1)
        return preds

    def post_personalize(self, out_dir, data_loader, visualizer, verbose=True):
        from networks.networks import FaceLoss

        bg_inpaint = self.src_info['bg']

        @torch.no_grad()
        def set_gen_inputs(sample):
            j2ds = sample['j2d'].cuda()  # (N, 4)
            T = sample['T'].cuda()  # (N, h, w, 2)
            T_cycle = sample['T_cycle'].cuda()  # (N, h, w, 2)
            src_inputs = sample['src_inputs'].cuda()  # (N, 6, h, w)
            tsf_inputs = sample['tsf_inputs'].cuda()  # (N, 6, h, w)
            src_fim = sample['src_fim'].cuda()
            tsf_fim = sample['tsf_fim'].cuda()
            init_preds = sample['preds'].cuda()
            images = sample['images']
            images = torch.cat([images[:, 0, ...], images[:, 1, ...]], dim=0).cuda()  # (2N, 3, h, w)
            pseudo_masks = sample['pseudo_masks']
            pseudo_masks = torch.cat([pseudo_masks[:, 0, ...], pseudo_masks[:, 1, ...]],
                                     dim=0).cuda()  # (2N, 1, h, w)

            return src_fim, tsf_fim, j2ds, T, T_cycle, \
                   src_inputs, tsf_inputs, images, init_preds, pseudo_masks

        def set_cycle_inputs(fake_tsf_imgs, src_inputs, tsf_inputs, T_cycle):
            # set cycle src inputs
            cycle_src_inputs = torch.cat([fake_tsf_imgs * tsf_inputs[:, -1:, ...], tsf_inputs[:, 3:]], dim=1)

            # set cycle tsf inputs
            cycle_tsf_img = F.grid_sample(fake_tsf_imgs, T_cycle,align_corners=True)
            cycle_tsf_inputs = torch.cat([cycle_tsf_img, src_inputs[:, 3:]], dim=1)

            return cycle_src_inputs, cycle_tsf_inputs

        def warp(preds, tsf, fim, fake_tsf_mask):
            front_mask = self.render.encode_front_fim(fim, transpose=True)
            preds = (1 - front_mask) * preds + tsf * front_mask * (1 - fake_tsf_mask)
            # preds = torch.clamp(preds + tsf * front_mask, -1, 1)
            return preds

        def inference(src_inputs, tsf_inputs, T, T_cycle, src_fim, tsf_fim):
            fake_src_color, fake_src_mask, fake_tsf_color, fake_tsf_mask = \
                self.generator.infer_front(src_inputs, tsf_inputs, T=T)

            fake_src_imgs = fake_src_mask * bg_inpaint + (1 - fake_src_mask) * fake_src_color
            fake_tsf_imgs = fake_tsf_mask * bg_inpaint + (1 - fake_tsf_mask) * fake_tsf_color

            if self._opt.front_warp:
                fake_tsf_imgs = warp(fake_tsf_imgs, tsf_inputs[:, 0:3], tsf_fim, fake_tsf_mask)

            cycle_src_inputs, cycle_tsf_inputs = set_cycle_inputs(
                fake_tsf_imgs, src_inputs, tsf_inputs, T_cycle)

            cycle_src_color, cycle_src_mask, cycle_tsf_color, cycle_tsf_mask = \
                self.generator.infer_front(cycle_src_inputs, cycle_tsf_inputs, T=T_cycle)

            cycle_src_imgs = cycle_src_mask * bg_inpaint + (1 - cycle_src_mask) * cycle_src_color
            cycle_tsf_imgs = cycle_tsf_mask * bg_inpaint + (1 - cycle_tsf_mask) * cycle_tsf_color

            if self._opt.front_warp:
                cycle_tsf_imgs = warp(cycle_tsf_imgs, src_inputs[:, 0:3], src_fim, fake_src_mask)

            return fake_src_imgs, fake_tsf_imgs, cycle_src_imgs, cycle_tsf_imgs, fake_src_mask, fake_tsf_mask

        def create_criterion():
            face_criterion = FaceLoss(pretrained_path=self._opt.face_model).cuda()
            idt_criterion = torch.nn.L1Loss()
            mask_criterion = torch.nn.BCELoss()

            return face_criterion, idt_criterion, mask_criterion

        init_lr = 0.0002
        nodecay_epochs = 5
        optimizer = torch.optim.Adam(self.generator.parameters(), lr=init_lr, betas=(0.5, 0.999))
        face_cri, idt_cri, msk_cri = create_criterion()

        step = 0
        logger = tqdm(range(nodecay_epochs))
        for epoch in logger:
            for i, sample in enumerate(data_loader):
                src_fim, tsf_fim, j2ds, T, T_cycle, src_inputs, tsf_inputs, \
                images, init_preds, pseudo_masks = set_gen_inputs(sample)

                # print(bg_inputs.shape, src_inputs.shape, tsf_inputs.shape)
                bs = tsf_inputs.shape[0]
                src_imgs = images[0:bs]
                fake_src_imgs, fake_tsf_imgs, cycle_src_imgs, cycle_tsf_imgs, fake_src_mask, fake_tsf_mask = inference(
                    src_inputs, tsf_inputs, T, T_cycle, src_fim, tsf_fim)

                # cycle reconstruction loss
                cycle_loss = idt_cri(src_imgs, fake_src_imgs) + idt_cri(src_imgs, cycle_tsf_imgs)

                # structure loss
                bg_mask = src_inputs[:, -1:]
                body_mask = 1 - bg_mask
                str_src_imgs = src_imgs * body_mask
                cycle_warp_imgs = F.grid_sample(fake_tsf_imgs, T_cycle)
                back_head_mask = 1 - self.render.encode_front_fim(tsf_fim, transpose=True, front_fn=False)
                struct_loss = idt_cri(init_preds, fake_tsf_imgs) + \
                              2 * idt_cri(str_src_imgs * back_head_mask, cycle_warp_imgs * back_head_mask)

                fid_loss = face_cri(src_imgs, cycle_tsf_imgs, kps1=j2ds[:, 0], kps2=j2ds[:, 0]) + \
                           face_cri(init_preds, fake_tsf_imgs, kps1=j2ds[:, 1], kps2=j2ds[:, 1])

                # mask loss
                # mask_loss = msk_cri(fake_tsf_mask, tsf_inputs[:, -1:]) + msk_cri(fake_src_mask, src_inputs[:, -1:])
                mask_loss = msk_cri(torch.cat([fake_src_mask, fake_tsf_mask], dim=0), pseudo_masks)

                loss = 10 * cycle_loss + 10 * struct_loss + fid_loss + 5 * mask_loss
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                if verbose:
                    logger.set_description(
                        (
                            f'epoch: {epoch + 1}; step: {step}; '
                            f'total: {loss.item():.6f}; cyc: {cycle_loss.item():.6f}; '
                            f'str: {struct_loss.item():.6f}; fid: {fid_loss.item():.6f}; '
                            f'msk: {mask_loss.item():.6f}'
                        )
                    )

                if verbose and step % 5 == 0:
                    self.visualize(visualizer, input_imgs=images, tsf_imgs=fake_tsf_imgs, cyc_imgs=cycle_tsf_imgs)

                step += 1

        self.generator.eval()
