import os
import re
import random
import numpy as np
import torch
import torch.nn as nn
import datetime
import math
import transformers
from peft import PeftModel
from transformers import LlamaForCausalLM
from model.modeling_llama import (
    LlamaForCausalLM as ReduLlamaForCausalLM,
    LlamaDecoderLayer as ReduLlamaDecoderLayer,
    LlamaMLP,
    LlamaAttention,
)
from transformers.models.llama.configuration_llama import LlamaConfig
from datasets import Dataset, DatasetDict, load_from_disk
from typing import Optional, Dict, List, Callable, Any, Union, Tuple
from collections import defaultdict
from utils.piqatrainer import LlamaTrainer

if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"

class LlamaUtils:

    def __init__(self):
        self.LIMIT = 512

    def collect(self,
        data_args,
        model_args,
        model: LlamaForCausalLM,
        trainer: LlamaTrainer,
    ):
        config: LlamaConfig = model.config
        handlers = []
        norm_dict = defaultdict(list)

        def init_fn(name, collector, collect_input=True):
            if name not in collector:
                collector[name] = []

            def fn(module, inputs, outputs):
                if collect_input:
                    data: torch.Tensor = inputs[0].detach().cpu().squeeze(0)
                else:
                    data: torch.Tensor = outputs[0].detach().cpu().squeeze(0)
                size = min(data_args.per_sample_token_num, data.shape[0])
                indices: np.ndarray = np.random.choice(
                    data.shape[0], size, replace=False).tolist()
                collector[name].append(data[indices, ...])

            return fn

        block_pat = re.compile('model\.layers\.(\d+)')

        for name, module in model.named_modules():
            if block_pat.fullmatch(name):

                # RMS:
                self_attn_norm_name = name + '.post_attention_layernorm'
                ffn_norm_name = name + '.input_layernorm'

                handlers.append(module.post_attention_layernorm.register_forward_hook(
                    init_fn(self_attn_norm_name, norm_dict)))
                handlers.append(module.input_layernorm.register_forward_hook(
                    init_fn(ffn_norm_name, norm_dict)))

        trainer.collect()

        def dump_norm():
            token_sample_num = int(
                math.ceil(data_args.token_sample_num / config.num_hidden_layers))
            hiddens = []
            for key, value in norm_dict.items():
                assert isinstance(key, str)
                print('key = {}'.format(key))
                hidden = torch.concat(value, dim=0).transpose(0, 1)
                size = min(hidden.shape[1], token_sample_num)
                indices = np.random.choice(
                    hidden.shape[1], size=size, replace=False)
                hiddens.append(hidden[:, indices])
            hidden = torch.cat(hiddens, dim=-1).to(torch.float)

            norm_proj, _, _ = torch.linalg.svd(hidden)
            norm_proj = norm_proj[:, :model_args.r_model]
            return norm_proj

        norm_proj = dump_norm()

        Llama_comp_params = {
            "norm_proj": norm_proj,
        }
        return Llama_comp_params

    @torch.no_grad()
    def load_model_params(
        self,
        model: ReduLlamaForCausalLM,
        state_dict: Dict[str, torch.Tensor],
        compression_params: Dict[str, torch.Tensor]
    ):
        norm_proj: torch.Tensor = compression_params["norm_proj"]

        for n, p in model.named_parameters():
            if n not in state_dict:
                continue
            if p.shape == state_dict[n].shape:
                p.copy_(state_dict[n])

        config: LlamaConfig = model.config
        block_pat = re.compile('model\.layers\.(\d+)')

        def get_norm_params(block_id: int, perfix: str):
            path = "model.layers.{}.{}".format(
                block_id, perfix)
            return state_dict[path + ".weight"]

        def load_qk_params(
                name: str,
                linear: torch.nn.Linear,
                norm_w: torch.Tensor,
                norm_proj: torch.Tensor,
        ):
            lin_w = state_dict[name + ".weight"]
            diag = torch.diag(norm_w)
            factor = math.sqrt(config.hidden_size / config.sub_size)

            n_lin_w = factor * (lin_w @ diag @ norm_proj)
            linear.weight.copy_(n_lin_w)

        def load_afternorm_linear(
                name: str,
                linear: torch.nn.Linear,
                norm_w: torch.Tensor,
                norm_proj: torch.Tensor,
        ):
            lin_w = state_dict[name + ".weight"]
            diag = torch.diag(norm_w)
            factor = math.sqrt(config.hidden_size / config.sub_size)

            n_lin_w = factor * (lin_w @ diag @ norm_proj)
            linear.weight.copy_(n_lin_w)

        def load_beforenorm_linear(
                name: str,
                linear: torch.nn.Linear,
                norm_proj: torch.Tensor,
        ):
            lin_w = state_dict[name + ".weight"]
            n_lin_w = norm_proj.T @ lin_w

            linear.weight.copy_(n_lin_w)

        for name, module in model.named_modules():
            match_block = block_pat.fullmatch(name)
            if match_block is not None:
                assert isinstance(module, ReduLlamaDecoderLayer)
                self_attn: LlamaAttention = module.self_attn
                ffn: LlamaMLP = module.mlp

                block_id = int(match_block.group(1))
                att_norm_w = get_norm_params(
                    block_id, "input_layernorm")
                ffn_norm_w = get_norm_params(
                    block_id, 'post_attention_layernorm')
                # self_attn
                load_qk_params(
                    name + ".self_attn.q_proj",
                    self_attn.q_proj,
                    att_norm_w,
                    norm_proj,
                )
                load_qk_params(
                    name + ".self_attn.k_proj",
                    self_attn.k_proj,
                    att_norm_w,
                    norm_proj,
                )
                load_afternorm_linear(
                    name + ".self_attn.v_proj",
                    self_attn.v_proj,
                    att_norm_w,
                    norm_proj,
                )
                load_beforenorm_linear(
                    name + ".self_attn.o_proj",
                    self_attn.o_proj,
                    norm_proj,
                )

                # FFN
                load_afternorm_linear(
                    name + ".mlp.gate_proj",
                    ffn.gate_proj,
                    ffn_norm_w,
                    norm_proj,
                )
                load_afternorm_linear(
                    name + ".mlp.up_proj",
                    ffn.up_proj,
                    ffn_norm_w,
                    norm_proj,
                )
                load_beforenorm_linear(
                    name + ".mlp.down_proj",
                    ffn.down_proj,
                    norm_proj,
                )

                module.post_attention_layernorm.weight.copy_(
                    torch.ones_like(module.post_attention_layernorm.weight)
                )
                module.input_layernorm.weight.copy_(
                    torch.ones_like(module.input_layernorm.weight)
                )
        
        model.model.down_linear.weight.copy_(norm_proj.T)
        model.model.up_linear.weight.copy_(norm_proj)
