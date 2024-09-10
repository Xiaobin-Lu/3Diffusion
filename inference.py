#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Power by Zongsheng Yue 2022-07-02 20:43:41

import os
import torch
import argparse
import numpy as np
from pathlib import Path
from einops import rearrange
from omegaconf import OmegaConf
from skimage import img_as_ubyte

from utils import util_opts
from utils import util_image
from utils import util_common

from sampler import DifIRSampler
from ResizeRight.resize_right import resize
from basicsr.utils.download_util import load_file_from_url

from models.adapter import Adapter
# PYTHONPATH="./:${PYTHONPATH}" CUDA_VISIBLE_DEVICES=1  python inference.py  --aligned --in_path /mnt/sda1/luxb/luxb_move/experiment_image_result/data/lq_lowQuality  --out_path ./res

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--gpu_id",
        type=str,
        default='3',
        help="GPU Index",
    )
    parser.add_argument(
        "-s",
        "--started_timesteps",
        type=int,
        default='100',
    )
    parser.add_argument(
        "--aligned",
        action='store_true',
        help='Input are alinged faces',
    )
    parser.add_argument(
        "--draw_box",
        action='store_true',
        help='Draw box for face in the unaligned case',
    )
    parser.add_argument(
        "-t",
        "--timestep_respacing",
        type=str,
        default='250',
        # default='ddim50',
        help='Sampling steps for Improved DDPM, parameter T in out paper (default 250)',
    )
    parser.add_argument(
        "--in_path",
        type=str,
        default='/mnt/sdb/face/data/lq',
        help='Folder to save the low quality image',
    )
    parser.add_argument(
        "--out_path",
        type=str,
        default='/home/wz/luxb/CelebA/6ka_test_80',
        help='Folder to save the restored results',
    )
    args = parser.parse_args()

    cfg_path = 'configs/train_with_0112.yaml'
    configs = OmegaConf.load(cfg_path)
    configs.gpu_id = "1"
    configs.aligned = args.aligned
    configs.diffusion.params.timestep_respacing = args.timestep_respacing

    # prepare the checkpoint
    print(configs.model.ckpt_path)
    if not Path(configs.model_ir.ckpt_path).exists():
        load_file_from_url(
            url="https://github.com/zsyOAOA/DifFace/releases/download/V1.0/General_Face_ffhq512.pth",
            model_dir=str(Path(configs.model_ir.ckpt_path).parent),
            progress=True,
            file_name=Path(configs.model_ir.ckpt_path).name,
        )

    # build the sampler for diffusion
    sampler_dist = DifIRSampler(configs)

    # prepare low quality images
    exts_all = ('jpg', 'png', 'jpeg', 'JPG', 'JPEG', 'bmp')
    if args.in_path.endswith(exts_all):
        im_path_list = [Path(args.in_path), ]
    else:  # for folder
        im_path_list = []
        for ext in exts_all:
            im_path_list.extend([x for x in Path(args.in_path).glob(f'*.{ext}')])

    # prepare result path
    if not Path(args.out_path).exists():
        Path(args.out_path).mkdir(parents=True)
    restored_face_dir = Path(args.out_path)
    if not restored_face_dir.exists():
        restored_face_dir.mkdir()
    if not args.aligned:
        cropped_face_dir = Path(args.out_path) / 'cropped_faces'
        if not cropped_face_dir.exists():
            cropped_face_dir.mkdir()
        restored_image_dir = Path(args.out_path)
        if not restored_image_dir.exists():
            restored_image_dir.mkdir()

    for ii, im_path in enumerate(im_path_list):
        if (ii + 1) % 5 == 0:
            print(f"Processing: {ii + 1}/{len(im_path_list)}...")
        im_lq = util_image.imread(im_path, chn='bgr', dtype='uint8')

        
        im_3d = util_image.imread("/mnt/sda1/luxb/luxb_move/experiment_image_result/data/3d_sr_1/"+str(im_path).split('/')[-1], chn='bgr', dtype='uint8')

        if args.aligned:
            face_restored, hr_out = sampler_dist.sample_func_ir_aligned(
                y0=im_lq,
                threeD=im_3d,
                start_timesteps=args.started_timesteps,
                need_restoration=True,
            )  # [0,1], 'rgb'
            face_restored = util_image.tensor2img(
                face_restored,
                rgb2bgr=True,
                min_max=(0.0, 1.0),
            )  # uint8, BGR
            hr_out = util_image.tensor2img(
                hr_out,
                rgb2bgr=True,
                min_max=(0.0, 1.0),
            )  # uint8, BGR
            save_path = restored_face_dir / im_path.name
            util_image.imwrite(face_restored, save_path, chn='bgr', dtype_in='uint8')
        else:
            image_restored, face_restored, face_cropped = sampler_dist.sample_func_bfr_unaligned(
                y0=im_lq,
                start_timesteps=args.started_timesteps,
                need_restoration=True,
                draw_box=args.draw_box,
            )

            # save the whole image
            save_path = restored_image_dir / im_path.name
            util_image.imwrite(image_restored, save_path, chn='bgr', dtype_in='uint8')

            # save the cropped and restored faces
            assert len(face_cropped) == len(face_restored)
            for jj, face_cropped_current in enumerate(face_cropped):
                face_restored_current = face_restored[jj]

                save_path = cropped_face_dir / f"{im_path.stem}_{jj}.png"
                util_image.imwrite(face_cropped_current, save_path, chn='bgr', dtype_in='uint8')

                save_path = restored_face_dir / f"{im_path.stem}_{jj}.png"
                util_image.imwrite(face_restored_current, save_path, chn='bgr', dtype_in='uint8')


if __name__ == '__main__':
    main()
