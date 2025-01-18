import os
import argparse
import cv2
import numpy as np
import torch
import utils.cropface as cropface
from utils.umeyama import umeyama
from face_detection.yolov5_face import YoloFace
from face_d3dfr.preprocess import D3DFR_preprocess_class
from face_d3dfr.D3DFR_ReconModel import model_FR3D_wrapper, model_3DMM_wrapper

class FaceRestoration(object):
    def __init__(self, args):
        # face det
        self.yoloface = YoloFace(pt_path=args["ckpt_facedet"], device=args["device"])
        # 3dmm
        self.model_3dmm = model_3DMM_wrapper(args["ckpt_3dmm"], device=args["device"])
        # preprocess
        self.d3dfrpreprocess = D3DFR_preprocess_class(args["ckpt_3dmm"])
        # d3dfr
        self.d3dfr = model_FR3D_wrapper().eval().to(args["device"])
        info = self.d3dfr.model_FR3d.load_state_dict(
            torch.load(args['ckpt_fr3d'], map_location=lambda storage, loc: storage))

        self.in_size = args['in_size']
        self.device = args['device']

        # the mask for pasting restored faces back
        self.mask = np.zeros((args['in_size'], args['in_size']), np.float32)
        cv2.rectangle(self.mask, (26, 26), (args['in_size']-26, args['in_size']-26), (1, 1, 1), -1, cv2.LINE_AA)
        self.mask = cv2.GaussianBlur(self.mask, (101, 101), 4)
        self.mask = cv2.GaussianBlur(self.mask, (101, 101), 4)
        self.mask = torch.from_numpy(self.mask).to(args['device']).view(1, 1, args['in_size'], args['in_size'])

        # get the reference 5 landmarks position in the crop settings
        default_square = True
        inner_padding_factor = 0.25
        outer_padding = (0, 0)
        self.reference_5pts = cropface.get_reference_facial_points((args['in_size'], args['in_size']), inner_padding_factor,
                                                                   outer_padding, default_square)

    @torch.no_grad()
    def infer(self, img_crop_cv2, crop_pts5):
        # convert tensor
        crop_rgb = cv2.cvtColor(img_crop_cv2, cv2.COLOR_BGR2RGB)
        crop_rgb_normed = (torch.from_numpy(crop_rgb).permute(2, 0, 1).float() - 127.5) / 127.5
        crop_t = crop_rgb_normed.view(1, 3, self.in_size, self.in_size).to(self.device)

        # warp
        d3dfr_5pts = self.d3dfrpreprocess.get_D3DFR_target5p(crop_pts5)
        tfm_d3dfr = umeyama(src=crop_pts5, dst=d3dfr_5pts, estimate_scale=True)[0:2]
        tfm_inv_d3dfr = umeyama(src=d3dfr_5pts, dst=crop_pts5, estimate_scale=True)[0:2]
        tfm_d3dfr_tensor = torch.tensor(np.array(tfm_d3dfr).astype(np.float32)).unsqueeze(0).to(self.device)
        tfm_inv_d3dfr_tensor = torch.tensor(np.array(tfm_inv_d3dfr).astype(np.float32)).unsqueeze(0).to(self.device)

        lr_coeff = self.d3dfr(im=crop_t, warpmat=tfm_d3dfr_tensor)
        recon_result_dict = self.model_3dmm(D3D_coeff=lr_coeff, inverse_warpmat=tfm_inv_d3dfr_tensor)
        crop_3drec_t = recon_result_dict['im_3DRec']
        
        return crop_3drec_t


    @torch.no_grad()
    def enhance(self, img_cv2, real_image=False):
        return self.enhance_crop(img_cv2)

    @torch.no_grad()
    def enhance_crop(self, img_crop_cv2):
        img_crop_cv2 = img_crop_cv2.squeeze(0).permute(1, 2, 0).cpu().numpy()
        img_crop_cv2 = cv2.cvtColor(img_crop_cv2, cv2.COLOR_RGB2BGR)
        img_crop_cv2 = ((img_crop_cv2)*255.).astype(np.uint8)
        img_crop_cv2 = cv2.resize(img_crop_cv2, (self.in_size, self.in_size), cv2.INTER_CUBIC)
        _, kpss, _ = self.yoloface.detect(img_crop_cv2)

        crop_pts5 = kpss[0]
        crop_3drec_t = self.infer(img_crop_cv2, crop_pts5)
        crop_3d_bgr = cv2.cvtColor((crop_3drec_t[0] * 127.5 + 127.5).cpu().numpy().astype('uint8').transpose(1, 2, 0),
                                   cv2.COLOR_RGB2BGR)
                                   
        crop_3d_bgr = cv2.cvtColor(crop_3d_bgr, cv2.COLOR_RGB2BGR)
        crop_3d_bgr = torch.from_numpy(crop_3d_bgr).permute(2, 0, 1).unsqueeze(0).float().to(self.device)/255
        return crop_3d_bgr
