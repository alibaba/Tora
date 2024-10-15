# -*- encoding: utf-8 -*-
'''
@File    :   patch_completion.py
@Time    :   2022/04/02 21:42:14
@Author  :   Ming Ding 
@Contact :   dm18@mails.tsinghua.edu.cn
'''

# here put the import lib
import os
import sys
import math
import random

import numpy as np
import torch
import torch.nn.functional as F
from icetk import icetk
from torchvision.io import read_image
from torchvision import transforms

from .base_completion import BaseCompletion

class PatchCompletion:
    def __init__(self, model, strategy, srg, log_attention_weight=3.5, max_inference_batch_size=8):
        self.base_comp = BaseCompletion(model, strategy, log_attention_weight, max_inference_batch_size)
        self.srg = srg
        self.device = model.parameters().__next__().device

    @staticmethod
    def _surrounding_box_crop(orig_mask):
        assert orig_mask.shape == (480, 480)
        nz = orig_mask.nonzero()
        h0, w0 = nz.min(dim=0)[0].tolist()
        h1, w1 = nz.max(dim=0)[0].tolist()
        inner_box = (h0, w0, h1+1, w1+1)
        hc, wc = (h0+h1)/2, (w0+w1)/2
        lenh, lenw = int((h1-h0+1)*20/14), int((w1-w0+1)*20/14)
        lenm = max(lenh, lenw)
        if lenm >= 480: # use whole image
            return (0, 0, 480, 480), orig_mask, inner_box
        # upper left corner
        h0, w0 = int(hc - lenm/2), int(wc - lenm/2)
        if h0 < 0:
            h0 = 0
        if int(hc + lenm/2) >= 480:
            h0 = 480 - lenm
        if w0 < 0:
            w0 = 0
        if int(wc + lenm/2) >= 480:
            w0 = 480 - lenm
        return (h0, w0, h0+lenm, w0+lenm), orig_mask[h0:h0+lenm, w0:w0+lenm], inner_box

    def __call__(self, image, orig_mask, text, batch_size=1):
        if isinstance(image, str):
            image = read_image(image).to(self.device)
            tr = transforms.Compose([
                    transforms.Resize(480),
                    transforms.CenterCrop(480),
                ])
            image = tr(image).float() / 255
        if isinstance(text, str):
            txt_tokens = torch.tensor(icetk.encode(text), dtype=torch.long, device=self.device)
        else:
            txt_tokens = text
        assert len(image.shape) == 3 and image.shape[-1] == 480
        (h0, w0, h1, w1), crop_mask, inner_box = self._surrounding_box_crop(orig_mask)
        crop_image = image[:, h0:h1, w0:w1]
        small_img_tokens = icetk.encode(image_torch=crop_image.unsqueeze(0), image_size=160)
        small_mask = (F.interpolate(crop_mask.float()[None,None,...], size=20) >= 0.5).view(20, 20)
        base_comp_tokens = self.base_comp(small_img_tokens, small_mask, txt_tokens, batch_size)
        selected_base_comp_tokens = base_comp_tokens # TODO multiple sort
        transformed_crops = []

        if h1 - h0 <= 160: # base_comp is enough
            for comp_token in selected_base_comp_tokens:
                transformed_crop = icetk.decode(image_ids=comp_token)
                transformed_crops.append(F.interpolate(transformed_crop, size=h1-h0))
        else: # need sr
            # (1) dsr the base_comp results
            if len(txt_tokens.shape) == 1:
                txt_tokens = txt_tokens.unsqueeze(0).expand(batch_size, txt_tokens.shape[-1])
            dsred_tokens = self.srg.dsr(txt_tokens, selected_base_comp_tokens) # [bz, 3600]
            # (2) upsample to 480 and tokenize the crop, maybe blured but conherence is more important
            upsampled_crop = F.interpolate(crop_image.unsqueeze(0), size=480, mode='bilinear')
            upsampled_crop_tokens = icetk.encode(image_torch=upsampled_crop)
            # (3) change the unmask region in (1) to (2)
            big_mask = (F.interpolate(crop_mask.float()[None,None,...], size=60) >= 0.5).view(3600)
            dsred_tokens[:, big_mask.logical_not()] = upsampled_crop_tokens[:, big_mask.logical_not()] 
            # (4) itersr via new mask api
            itersred_tokens = self.srg.itersr(txt_tokens, dsred_tokens, input_mask=big_mask) # will change dsred_tokens
            # (5) downsample to crop size
            for comp_token in itersred_tokens:
                transformed_crop = icetk.decode(image_ids=comp_token)
                transformed_crops.append(F.interpolate(transformed_crop, size=h1-h0))
        transformed_crops = torch.cat(transformed_crops, dim=0)
        # find the position of inner box in the crop
        h0i, w0i, h1i, w1i = inner_box
        h0i -= h0; h1i -= h0; w0i -= w0; w1i -= w0
        
        # replace the inner box
        ret = image.clone()[None, ...].expand(batch_size, -1, -1, -1).contiguous()
        for i in range(batch_size):
            # ret[i, :, inner_box[0]:inner_box[2], inner_box[1]:inner_box[3]] = transformed_crops[i, :, h0i:h1i, w0i:w1i]
            ret[i, :, h0:h1, w0:w1] = transformed_crops[i]
            ret[i] = icetk.decode(image_ids=icetk.encode(image_torch=ret[i], image_size=480))
        return ret
