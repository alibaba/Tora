# -*- encoding: utf-8 -*-
'''
@File    :   inference_coglm.py
@Time    :   2021/10/09 19:41:58
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
from functools import partial
import numpy as np

from SwissArmyTransformer import get_args, get_tokenizer
from SwissArmyTransformer.model import CachedAutoregressiveModel
from SwissArmyTransformer.generation.sampling_strategies import BaseStrategy
from SwissArmyTransformer.generation.autoregressive_sampling import filling_sequence, evaluate_perplexity
from SwissArmyTransformer.generation.utils import timed_name, save_multiple_images, generate_continually

from coglm_strategy import CoglmStrategy

from icetk import icetk as tokenizer
tokenizer.add_special_tokens(['<start_of_image>', '<start_of_english>', '<start_of_chinese>'])

def get_masks_and_position_ids_coglm(seq, context_length):
    tokens = seq.unsqueeze(0)

    attention_mask = torch.ones((1, len(seq), len(seq)), device=tokens.device)
    attention_mask.tril_()
    attention_mask[..., :context_length] = 1
    attention_mask.unsqueeze_(1)

    position_ids = torch.zeros(len(seq), device=tokens.device, dtype=torch.long)
    torch.arange(0, context_length, out=position_ids[:context_length])
    torch.arange(512, 512 + len(seq) - context_length, 
            out=position_ids[context_length:]
    )

    position_ids = position_ids.unsqueeze(0)
    return tokens, attention_mask, position_ids

def main(args):
    model, args = InferenceModel.from_pretrained(args, 'coglm')
    text_model = CachedAutoregressiveModel(args, transformer=model.transformer)
    
    query_template = args.query_template
    invalid_slices = [slice(tokenizer.num_image_tokens, None)]
    strategy = CoglmStrategy(invalid_slices,
                            temperature=args.temp_all_gen, top_k=args.topk_gen, top_k_cluster=args.temp_cluster_gen)
    
    from sr_pipeline import SRGroup 
    if not args.only_first_stage:
        srg = SRGroup(args)
        
    def process(raw_text):
        if args.with_id:
            query_id, raw_text = raw_text.split('\t')
        print('raw text: ', raw_text)
        text = query_template.format(raw_text)
        seq = tokenizer.encode(text)
        if len(seq) > 110:
            raise ValueError('text too long.')
        
        txt_len = len(seq) - 1
        seq = torch.tensor(seq + [-1]*400, device=args.device)
        # calibrate text length
        log_attention_weights = torch.zeros(len(seq), len(seq), 
            device=args.device, dtype=torch.half if args.fp16 else torch.float32)
        log_attention_weights[:, :txt_len] = args.attn_plus
        # generation
        mbz = args.max_inference_batch_size
        assert args.batch_size < mbz or args.batch_size % mbz == 0
        get_func = partial(get_masks_and_position_ids_coglm, context_length=txt_len)
        output_list, score_list = [], []

        for tim in range(max(args.batch_size // mbz, 1)):
            strategy.start_pos = txt_len + 1
            coarse_samples = filling_sequence(model, seq.clone(),
                    batch_size=min(args.batch_size, mbz),
                    strategy=strategy,
                    log_attention_weights=log_attention_weights,
                    get_masks_and_position_ids=get_func
                    )[0]
            
            # get ppl for inverse prompting
            if args.inverse_prompt:
                image_text_seq = torch.cat(
                    (
                        coarse_samples[:, -400:],
                        torch.tensor([tokenizer['<start_of_chinese>']]+tokenizer.encode(raw_text), device=args.device).expand(coarse_samples.shape[0], -1)
                    ), dim=1)
                seqlen = image_text_seq.shape[1]
                attention_mask = torch.zeros(seqlen, seqlen, device=args.device, dtype=torch.long)
                attention_mask[:, :400] = 1
                attention_mask[400:, 400:] = 1
                attention_mask[400:, 400:].tril_()
                position_ids = torch.zeros(seqlen, device=args.device, dtype=torch.long)
                torch.arange(513, 513+400, out=position_ids[:400])
                torch.arange(0, seqlen-400, out=position_ids[400:])
                loss_mask = torch.zeros(seqlen, device=args.device, dtype=torch.long)
                loss_mask[401:] = 1
                scores = evaluate_perplexity(
                    text_model, image_text_seq, attention_mask,
                    position_ids, loss_mask#, invalid_slices=[slice(0, 20000)], reduction='mean'
                )
                score_list.extend(scores.tolist())
                # ---------------------
            
            output_list.append(
                    coarse_samples
                )
        output_tokens = torch.cat(output_list, dim=0)
        
        if args.inverse_prompt:
            order_list = np.argsort(score_list)[::-1]
            print(sorted(score_list))
        else:
            order_list = range(output_tokens.shape[0])
            
        imgs, txts = [], []
        if args.only_first_stage:
            for i in order_list:
                seq = output_tokens[i]
                decoded_img = tokenizer.decode(image_ids=seq[-400:])
                decoded_img = torch.nn.functional.interpolate(decoded_img, size=(480, 480))
                imgs.append(decoded_img) # only the last image (target)
        if not args.only_first_stage: # sr
            iter_tokens = srg.sr_base(output_tokens[:, -400:], seq[:txt_len])
            for seq in iter_tokens:
                decoded_img = tokenizer.decode(image_ids=seq[-3600:])
                decoded_img = torch.nn.functional.interpolate(decoded_img, size=(480, 480))
                imgs.append(decoded_img) # only the last image (target)

        # save 
        if args.with_id:
            full_path = os.path.join(args.output_path, query_id)
            os.makedirs(full_path, exist_ok=True)
            save_multiple_images(imgs, full_path, False)
        else: 
            prefix = raw_text.replace('/', '')[:20]
            full_path = timed_name(prefix, '.jpeg', args.output_path)
            imgs = torch.cat(imgs, dim=0)
            print("\nSave to: ", full_path, flush=True)
            from PIL import Image
            from torchvision.utils import make_grid
            grid = make_grid(imgs, nrow=3, padding=0)
            # Add 0.5 after unnormalizing to [0, 255] to round to nearest integer
            ndarr = grid.mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to('cpu', torch.uint8).numpy()
            im = Image.fromarray(ndarr)
            im.save(full_path, quality=100, subsampling=0)
    
    os.makedirs(args.output_path, exist_ok=True)
    generate_continually(process, args.input_source)

class InferenceModel(CachedAutoregressiveModel):
    def final_forward(self, logits, **kwargs):
        logits_parallel = logits
        logits_parallel = torch.nn.functional.linear(logits_parallel.float(), self.transformer.word_embeddings.weight[:20000].float())
        return logits_parallel

def get_recipe(name):
    r = {
        'attn_plus': 1.4,
        'temp_all_gen': 1.15,
        'topk_gen': 16,
        'temp_cluster_gen': 1.,

        'temp_all_dsr': 1.5,
        'topk_dsr': 100,
        'temp_cluster_dsr': 0.89,

        'temp_all_itersr': 1.3,
        'topk_itersr': 16,
        'query_template': '{}<start_of_image>'
    }
    if name == 'none':
        pass
    elif name == 'mainbody':
        r['query_template'] = '{} 高清摄影 隔绝<start_of_image>'
        
    elif name == 'photo':
        r['query_template'] = '{} 高清摄影<start_of_image>'
        
    elif name == 'flat':
        r['query_template'] = '{} 平面风格<start_of_image>'
        # r['attn_plus'] = 1.8
        # r['temp_cluster_gen'] = 0.75
        r['temp_all_gen'] = 1.1
        r['topk_dsr'] = 5
        r['temp_cluster_dsr'] = 0.4

        r['temp_all_itersr'] = 1
        r['topk_itersr'] = 5
    elif name == 'comics':
        r['query_template'] = '{} 漫画 隔绝<start_of_image>'
        r['topk_dsr'] = 5
        r['temp_cluster_dsr'] = 0.4
        r['temp_all_gen'] = 1.1
        r['temp_all_itersr'] = 1
        r['topk_itersr'] = 5
    elif name == 'oil':
        r['query_template'] = '{} 油画风格<start_of_image>'
        pass
    elif name == 'sketch':
        r['query_template'] = '{} 素描风格<start_of_image>'
        r['temp_all_gen'] = 1.1
    elif name == 'isometric':
        r['query_template'] = '{} 等距矢量图<start_of_image>'
        r['temp_all_gen'] = 1.1
    elif name == 'chinese':
        r['query_template'] = '{} 水墨国画<start_of_image>'
        r['temp_all_gen'] = 1.12
    elif name == 'watercolor':
        r['query_template'] = '{} 水彩画风格<start_of_image>'
    return r

if __name__ == "__main__":
    py_parser = argparse.ArgumentParser(add_help=False)
    py_parser.add_argument('--img-size', type=int, default=160)
    py_parser.add_argument('--only-first-stage', action='store_true')
    py_parser.add_argument('--inverse-prompt', action='store_true')
    py_parser.add_argument('--style', type=str, default='mainbody', 
        choices=['none', 'mainbody', 'photo', 'flat', 'comics', 'oil', 'sketch', 'isometric', 'chinese', 'watercolor'])
    known, args_list = py_parser.parse_known_args()
    args = get_args(args_list)
    args = argparse.Namespace(**vars(args), **vars(known), **get_recipe(known.style))
    with torch.no_grad():
        main(args)
