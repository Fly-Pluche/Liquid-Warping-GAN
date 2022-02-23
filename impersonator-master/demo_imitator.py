import numpy as np
from tqdm import tqdm
import os
import glob

from models.imitator import Imitator
from models.D3net import D3net
from options.test_options import TestOptions
from utils.util import mkdir
import pickle
from utils.video import make_video
import time

from run_imitator import adaptive_personalize

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
mixamo_root_path = './assets/samples/refs/mixamo'
# MIXAMO_DANCE_ACTION_IDX_LIST = [102]
# MIXAMO_BASE_ACTION_IDX_LIST = [20, 22, 32, 70]
# MIXAMO_ACROBAT_ACTION_IDX_LIST = [7, 31, 83, 131, 145]

MIXAMO_DANCE_ACTION_IDX_LIST = [102]
MIXAMO_BASE_ACTION_IDX_LIST = [20]
MIXAMO_ACROBAT_ACTION_IDX_LIST = [7]


def load_mixamo_smpl(mixamo_idx):
    global mixamo_root_path

    dir_name = '%.4d' % mixamo_idx
    pkl_path = os.path.join(mixamo_root_path, dir_name, 'result.pkl')

    with open(pkl_path, 'rb') as f:
        result = pickle.load(f)

    anim_len = result['anim_len']
    pose_array = result['smpl_array'].reshape(anim_len, -1)
    cam_array = result['cam_array']
    shape_array = np.ones((anim_len, 10))
    smpl_array = np.concatenate((cam_array, pose_array, shape_array), axis=1)

    return smpl_array


def generate_actor_result(test_opt, src_img_path):
    ''''生成器'''
    imitator = Imitator(test_opt)

    src_img_name = os.path.split(src_img_path)[-1][:-4]
    test_opt.src_path = src_img_path

    if test_opt.post_tune:
        ##
        adaptive_personalize(test_opt, imitator, visualizer=None)
    else:
        imitator.personalize(test_opt.src_path, visualizer=None)

    action_list_dict = {'dance': MIXAMO_DANCE_ACTION_IDX_LIST,
                        'base': MIXAMO_BASE_ACTION_IDX_LIST,
                        'acrobat': MIXAMO_ACROBAT_ACTION_IDX_LIST}

    for action_type in ['dance']:
    # for action_type in ['dance', 'base', 'acrobat']:
        for i in action_list_dict[action_type]:

            """创立文件夹"""
            if test_opt.output_dir:
                pred_output_dir = os.path.join(test_opt.output_dir, 'mixamo_preds')
                if os.path.exists(pred_output_dir):
                    os.system("rm -r %s" % pred_output_dir)
                mkdir(pred_output_dir)
                print(pred_output_dir)
            else:
                pred_output_dir = None

            """加载 SMPL"""
            tgt_smpls = load_mixamo_smpl(i)
            """ 推理 """
            imitator.inference_by_smpls(tgt_smpls, cam_strategy='smooth', output_dir=pred_output_dir, visualizer=None)

            save_dir = os.path.join(test_opt.output_dir, src_img_name, action_type)
            mkdir(save_dir)

            """ 生成动图 """
            output_mp4_path = os.path.join(save_dir, 'mixamo_%.4d_%s.mp4' % (i, src_img_name))
            img_path_list = sorted(glob.glob('%s/*.jpg' % pred_output_dir))
            # './outputs/results/demos/imitators/mixamo_preds/pred_00000100.jpg'
            # save_path=D3net(img_path_list)
            # img_path_list=sorted(glob.glob(save_path+'/*.jpg'))
            make_video(output_mp4_path, img_path_list, save_frames_dir=None, fps=30)


def clean(output_dir):

    for item in ['imgs', 'pairs', 'mixamo_preds', 'pairs_meta.pkl']:
        filepath = os.path.join(output_dir, item)
        if os.path.exists(filepath):
            os.system("rm -r %s" % filepath)


def main():
    # meta imitator
    test_opt = TestOptions().parse()
    test_opt.bg_ks = 25
    test_opt.front_warp = False
    test_opt.post_tune = True

    test_opt.output_dir = mkdir('./outputs/results/demos/imitators')
    # source images from iPER

    images_paths=['./assets/src_imgs/internet/ins3.jpg',
                  './assets/src_imgs/internet/ins8.jpg',
                  './assets/src_imgs/internet/men1_256.jpg',
                  './assets/src_imgs/internet/women1_256.jpg',
                  './assets/src_imgs/imper_A_Pose/009_5_1_000.jpg',
                  './assets/src_imgs/imper_A_Pose/024_8_2_0000.jpg',
                  './assets/src_imgs/fashion_woman/Sweaters-id_0000088807_4_full.jpg']
    images_paths=['./assets/src_imgs/internet/men1_256.jpg']
    for src_img_path in tqdm(images_paths):
        generate_actor_result(test_opt, src_img_path)

    """对预测结果使用超分辨率"""
    # a='./outputs/results/demos/imitators/mixamo_preds/'
    # img_path_list = sorted(glob.glob(a+'/*.jpg'))
    # # './outputs/results/demos/imitators/mixamo_preds/pred_00000100.jpg'
    # save_path = D3net(img_path_list)
    # save_path='/home/ray/workspace/Fly_Pluche/impersonator-master/outputs/results/demos/imitators/mixamo_preds_final'
    # img_path_list = sorted(glob.glob(save_path + '/*.jpg'))
    # output_mp4_path='/home/ray/workspace/Fly_Pluche/impersonator-master/outputs/results/demos/imitators/1.mp4'
    # make_video(output_mp4_path, img_path_list, save_frames_dir=None, fps=30)


    # clean other files

    clean(test_opt.output_dir)
    print("---"*88)
    print('Completed! All demo videos are save in {}'.format(test_opt.output_dir))


if __name__ == "__main__":
    time1=time.time()

    main()

    time2=time.time()
    print('---'*88)
    print('Finally finish！')
    second=time2-time1
    minute=second//60

    print(f'Spend {minute}m,{60*(second/60-minute)}s')
