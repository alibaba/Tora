# -*- encoding: utf-8 -*-
'''
@File    :   pretrain_coglm.py
@Time    :   2021/12/30 15:56:40
@Author  :   Ming Ding 
@Contact :   dm18@mails.tsinghua.edu.cn
'''

# here put the import lib
import os
import sys
import math
import random

import torch
import argparse
import numpy as np
from icetk import icetk as tokenizer
tokenizer.add_special_tokens(['<start_of_image>', '<start_of_english>', '<start_of_chinese>'])

from SwissArmyTransformer import mpu, get_args, get_tokenizer
from SwissArmyTransformer.model.base_model import BaseModel, BaseMixin
from SwissArmyTransformer.training.deepspeed_training import training_main
from SwissArmyTransformer.data_utils import BinaryDataset

def get_batch(data_iterator, args, timers):
    # Items and their type.
    keys = ['text']
    datatype = torch.int64

    # Broadcast data.
    timers('data loader').start()
    if data_iterator is not None:
        data = next(data_iterator)
    else:
        data = None
    timers('data loader').stop()

    data_b = mpu.broadcast_data(keys, data, datatype)
    # Unpack.
    tokens_ = data_b['text'].long()
    
    tokens, position_ids, labels, loss_masks, attention_mask = [], [], [], [], []
    # dispatch text-image, text samples
    for sample in tokens_:
        pad_masks = (sample == tokenizer['<pad>'])
        if pad_masks.any(): # text-image
            if random.random() < args.image_caption_ratio:
                token, pos, label, attn_mask, loss_mask = make_image_text_understanding(sample)
            else:
                token, pos, label, attn_mask, loss_mask = make_text_image_generation(sample)
        else:
            if sample[0] >= tokenizer.num_image_tokens:
                token, pos, label, attn_mask, loss_mask = make_text_understanding(sample)
            else:
                raise ValueError('temporally not support pure image samples')
        tokens.append(token)
        position_ids.append(pos)
        labels.append(label)
        attention_mask.append(attn_mask)
        loss_masks.append(loss_mask)
        
    tokens = torch.stack(tokens)
    position_ids = torch.stack(position_ids)
    labels = torch.stack(labels)
    attention_mask = torch.stack(attention_mask).unsqueeze(1)
    if args.fp16:
        attention_mask = attention_mask.half()
    loss_masks = torch.stack(loss_masks)
    
    return tokens, position_ids, labels, attention_mask, loss_masks

def make_text_understanding(x, poisson_rate=3, mask_ratio=0.15):
    token = x
    pos = torch.arange(len(x), device=x.device, dtype=torch.long)
    label = x.clone()
    label[:-1] = x[1:]
    attn_mask = torch.ones(len(x), len(x), device=x.device).tril() # TODO FP16
    loss_mask = torch.zeros_like(x)

    m = torch.distributions.Poisson(poisson_rate)
    span_lens = m.sample(
        torch.Size((int(len(x) / poisson_rate * mask_ratio) + 10,))
        ).clip_(1).long() # +10 to ensure enough
    lensum = 0
    for i in range(len(span_lens)): 
        lensum += span_lens[i]
        if lensum > len(x) * mask_ratio:
            break
    lensum -= span_lens[i]
    span_lens = span_lens[:i]
    prev = sorted(random.sample(range(len(x) - lensum), len(span_lens)))
    
    cum_lens, end_of_last_span = 0, 0
    full_attn_slices = []
    for i in range(len(prev)):
        prev[i] += cum_lens
        cum_lens += span_lens[i]
        attn_mask[end_of_last_span: prev[i]] = 0
        full_attn_slices.append(slice(end_of_last_span, prev[i]))
        loss_mask[prev[i]: prev[i] + span_lens[i]] = 1
        # the uni-direction region contains the previous one for each span
        end_of_last_span = prev[i] + span_lens[i] + 1
    for s in full_attn_slices:
        attn_mask[:, s] = 1

    return token, pos, label, attn_mask, loss_mask 

def make_text_image_generation(x):
    pad_indices = (x == tokenizer['<pad>']).nonzero().squeeze(1)
    if pad_indices.shape[0] == 1 or pad_indices[1] - pad_indices[0] == 1:
        # no sep pad, only single language
        selected_text_slice = slice(0, pad_indices[0])
        unselected_text_slice = None
    elif pad_indices[1] - pad_indices[0] > 1:
        # select one language at random
        selected_text_slice, unselected_text_slice = slice(0, pad_indices[0]), slice(pad_indices[0] + 1, pad_indices[1])
        if random.random() < 0.5:
            tmp = selected_text_slice
            selected_text_slice = unselected_text_slice
            unselected_text_slice = tmp
    else:
        print('warning: non-text image.')
    assert len(x) <= 512
    
    token = x.clone()
    token[-401] = tokenizer['<start_of_image>']
    
    pos = torch.zeros_like(x)
    pos[selected_text_slice] = torch.arange(len(pos[selected_text_slice]), device=x.device)
    pos[-401:] = torch.arange(512, 512 + 401, device=x.device)
    
    label = x.clone()
    label[:-1] = x[1:]
    
    loss_mask = torch.zeros_like(x)
    loss_mask[-401:-1] = 1
    
    attn_mask = torch.zeros(len(x), len(x), device=x.device) # TODO FP16
    attn_mask[:, selected_text_slice] = 1
    attn_mask[-401:, -401:] = 1
    attn_mask[-401:, -401:].tril_()
    
    # whether text unidirectional
    # attn_mask[selected_text_slice, selected_text_slice].tril_()
    # loss_mask[selected_text_slice.start: selected_text_slice.stop - 1] = 1
    
    return token, pos, label, attn_mask, loss_mask 

def make_image_text_understanding(x, patch_size=4, pseudo_mask_ratio=0.8):
    token = x.clone()
    text_lr = []
    pad_indices = (x == tokenizer['<pad>']).nonzero().squeeze(1)
    if pad_indices.shape[0] == 1 or pad_indices[1] - pad_indices[0] == 1:
        # no sep pad, only single language
        text_lr.append((0+1, pad_indices[0]+1))
    elif pad_indices[1] - pad_indices[0] > 1:
        # both zh and en
        text_lr.extend([(0+1, pad_indices[0]+1), (pad_indices[0] + 1+1, pad_indices[1]+1)])
    token[1:-400] = x[:-401] # move right to set aside start tokens
    
    for l, r in text_lr:
        is_en = (token[l:r] < 83823).all()
        token[l-1] = tokenizer['<start_of_english>'] if is_en else tokenizer['<start_of_chinese>']
    
    pos = torch.zeros_like(x)
    for l, r in text_lr:
        pos[l-1:r] = torch.arange(r-l+1, device=x.device)
    pos[-400:] = torch.arange(512 +1, 512 + 401, device=x.device)
    
    label = token.clone()
    label[:-1] = token[1:]
    
    loss_mask = torch.zeros_like(x)
    attn_mask = torch.zeros(len(x), len(x), device=x.device) # TODO FP16
    for l, r in text_lr:
        loss_mask[l-1:r-1] = 1
        attn_mask[l-1:r, l-1:r] = 1
        attn_mask[l-1:r, l-1:r].tril_()
        attn_mask[l-1:r, -400:] = 1
    
    # sample patch_size * patch_size patches
    n = int(pseudo_mask_ratio * 400 / patch_size ** 2)
    lu_corners = random.sample(range((20 - patch_size)**2), k=n)

    region_mask = torch.zeros_like(x, dtype=torch.bool)
    region_mask[-400:] = True
    
    for c in lu_corners:
        x, y = c // (20-patch_size), c % (20-patch_size)
        loss_mask[-400:].view(20, 20)[x: x + patch_size, y: y + patch_size - 1] = 1.
        region_mask[-400:].view(20, 20)[x: x + patch_size, y: y + patch_size] = False
    attn_mask[-400:, -400:] = 1.
    attn_mask[-400:, -400:].tril_()
    attn_mask.masked_fill_(region_mask.unsqueeze(1), 0)
    attn_mask.masked_fill_(region_mask.unsqueeze(0), 1)
    
    return token, pos, label, attn_mask, loss_mask 

def forward_step(data_iterator, model, args, timers):
    """Forward step."""

    # Get the batch.
    timers('batch generator').start()
    tokens, position_ids, labels, attention_mask, loss_mask = get_batch(
        data_iterator, args, timers)
    timers('batch generator').stop()
    # Forward model. 
    logits, *mems = model(tokens, position_ids, attention_mask)
    losses = mpu.vocab_parallel_cross_entropy(logits.contiguous().float(), labels)
    # scaling loss mask
    loss_mask = loss_mask.reshape(-1)
    losses = losses.view(-1) * loss_mask
    loss = torch.sum(losses) / loss_mask.sum()
    # split loss 
    is_image_pos = (labels<20000).view(-1).logical_and(loss_mask)
    is_english_pos = ((labels>20100) & (labels < 83823)).view(-1).logical_and(loss_mask)
    is_chinese_pos = ((labels>=83823) & (labels < 145653)).view(-1).logical_and(loss_mask)
    image_loss = losses[is_image_pos].sum() / max(is_image_pos.sum(), 1)
    english_loss = losses[is_english_pos].sum() / max(is_english_pos.sum(), 1)
    chinese_loss = losses[is_chinese_pos].sum() / max(is_chinese_pos.sum(), 1)
    return loss, {'image_loss': image_loss, 'english_loss': english_loss, 'chinese_loss': chinese_loss}

def create_dataset_function(path, args):
    def process_fn(row):
        row = row.astype(np.int64)
        return {'text': row}
    return BinaryDataset(path, process_fn, length_per_sample=512)

if __name__ == '__main__':    
    py_parser = argparse.ArgumentParser(add_help=False)
    py_parser.add_argument('--new_hyperparam', type=str, default=None)
    py_parser.add_argument('--mask_ratio', type=float, default=0.15)
    py_parser.add_argument('--image_caption_ratio', type=float, default=0.3)
    known, args_list = py_parser.parse_known_args()
    args = get_args(args_list)
    args = argparse.Namespace(**vars(args), **vars(known))
    
    training_main(args, model_cls=BaseModel, forward_step_function=forward_step, create_dataset_function=create_dataset_function)