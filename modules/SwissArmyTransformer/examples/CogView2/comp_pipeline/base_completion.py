# -*- encoding: utf-8 -*-
'''
@File    :   base_completion.py
@Time    :   2022/04/02 16:52:19
@Author  :   Ming Ding 
@Contact :   dm18@mails.tsinghua.edu.cn
'''

# here put the import lib
import os
import sys
import math
import random
from functools import partial

import numpy as np
import torch
import torch.nn.functional as F
from icetk import icetk
from SwissArmyTransformer.generation.autoregressive_sampling import filling_sequence

def get_masks_and_position_ids_comp(seq, pos_ids, txt_len, context_length):
    tokens = seq.unsqueeze(0)

    attention_mask = torch.ones((1, len(seq), len(seq)), device=tokens.device)
    attention_mask.tril_()
    attention_mask[..., txt_len+1:, txt_len+1:context_length] = 1
    attention_mask[..., context_length+1:, :txt_len] = 1 # TODO
    attention_mask[..., :txt_len, :txt_len] = 1
    # attention_mask[..., :, :txt_len] = 1
    attention_mask.unsqueeze_(1)

    position_ids = pos_ids

    position_ids = position_ids.unsqueeze(0)
    return tokens, attention_mask, position_ids

class BaseCompletion:
    def __init__(self, model, strategy, log_attention_weight=3.5, max_inference_batch_size=8):
        self.model = model
        self.log_attention_weight = log_attention_weight
        self.max_inference_batch_size = max_inference_batch_size
        self.strategy = strategy
        pass

    def __call__(self, img_tokens, mask, txt_tokens, batch_size):
        assert img_tokens.shape == (1, 400)
        img_tokens = img_tokens.view(20, 20)
        assert mask.shape == (20, 20)
        mask = mask.view(400)
        device = img_tokens.device
        # reorder sequence
        txt_len = len(txt_tokens)
        pos0, pos1, token0, token1 = [], [], [], []
        for i in range(20):
            for j in range(20):
                if mask[i*20+j]: # regenerate => change to -1
                    pos1.append(513+j+i*20)
                    token1.append(-1)
                elif j < 19 and mask[i*20+j+1]: # the first of the block
                    pos1.append(513+j+i*20)
                    token1.append(img_tokens[i,j])
                else: # context
                    pos0.append(513+j+i*20)
                    token0.append(img_tokens[i,j])
        seq = torch.cat(
            (
                txt_tokens, 
                torch.tensor([icetk['<start_of_image>']], device=device),
                torch.tensor(token0, device=device),
                torch.tensor(token1, device=device)
            ))
        pos_ids = torch.cat(
            (
                torch.arange(txt_len, device=device),
                torch.tensor([512], device=device),
                torch.tensor(pos0, device=device),
                torch.tensor(pos1, device=device)
            )
        )
        context_length = len(pos_ids) - len(pos1)

        log_attention_weights = torch.zeros(len(seq), len(seq), 
            device=device, dtype=self.model.parameters().__next__().dtype)
        log_attention_weights[..., :txt_len] = self.log_attention_weight

        # generation
        mbz = self.max_inference_batch_size
        assert batch_size < mbz or batch_size % mbz == 0
        get_func = partial(get_masks_and_position_ids_comp, pos_ids=pos_ids, txt_len=txt_len, context_length=context_length)
        output_list = []
        for tim in range(max(batch_size // mbz, 1)):
            output_list.append(
                filling_sequence(self.model, seq.clone(),
                    batch_size=min(batch_size, mbz),
                    strategy=self.strategy,
                    log_attention_weights=log_attention_weights,
                    get_masks_and_position_ids=get_func
                    )[0]
                )
        output_tokens = torch.cat(output_list, dim=0)
        # decoding
        small_imgs = []
        for seq in output_tokens:
            # ---
            im = seq[-400:].clone()
            for j in range(txt_len+1, len(pos_ids)):
                im[pos_ids[j]-513] = seq[j]
            # ---
            small_imgs.append(im)
            
        small_imgs = torch.stack(small_imgs)
        return small_imgs