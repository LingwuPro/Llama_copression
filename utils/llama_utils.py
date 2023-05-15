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
from transformers import GenerationConfig, LlamaForCausalLM, LlamaTokenizer, LlamaTokenizerFast
from transformers.models.llama.modeling_llama import (
    LlamaRMSNorm,
    LlamaRotaryEmbedding,
    LlamaMLP,  # -ffn
    LlamaAttention,
    LlamaDecoderLayer,  # -wholeBlock
)
from transformers.models.llama.configuration_llama import LlamaConfig
from datasets import Dataset, DatasetDict, load_from_disk, load_metric
from typing import Optional, Dict, List, Callable, Any, Union, Tuple
from collections import defaultdict
import pickle

if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"


class LlamaUtils:
    def __init__(self):
        self.LIMIT = 512
        ...

    def collect(self,
                data_args,
                model_args,
                model,
                trainer
                ):
        config: LlamaConfig = model.config
        handlers = []
        norm_dict = defaultdict(list)
        attn_dict = defaultdict(list)
        ffn_dict = defaultdict(list)
        wo_dict = defaultdict(list)

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
                if len(collector[name]) == self.LIMIT and data_args.use_disk:
                    count = self.file_count[name]
                    self.file_count[name] += 1
                    file_name = "tmp/{}_{}.pkl".format(name, count)
                    with open(file_name, "wb") as f:
                        pickle.dump(collector[name], f)
                    collector[name] = []

            return fn

        def init_wo_fn(name, collector):
            def fn(module: nn.Linear, inputs, outputs):
                data: torch.Tensor = inputs[0].detach().cpu().squeeze(0)
                size = min(data_args.per_sample_token_num, data.shape[0])
                indices = np.random.choice(
                    data.shape[0], size, replace=False).tolist()
                collector[name].append(data[indices, ...])
            return fn

        def init_ffn_fn(name, collector, collect_input=True):
            if name not in collector:
                collector[name] = []

            def fn(module, inputs, outputs):
                if collect_input:
                    data = inputs[0].detach().cpu().squeeze(0)
                else:
                    data = outputs[0].detach().cpu().squeeze(0)
                collector[name].append(
                    (data.shape[0], data.abs().mean(dim=0), data.pow(2).mean(dim=0)))
                if len(collector[name]) == self.LIMIT and data_args.use_disk:
                    count = self.file_count[name]
                    self.file_count[name] += 1
                    file_name = "tmp/{}_{}.pkl".format(name, count)
                    with open(file_name, "wb") as f:
                        pickle.dump(collector[name], f)
                    collector[name] = []

            return fn

        def dump_value(collector: Dict):
            if data_args.use_disk:
                for name in collector.keys():
                    if len(collector[name]) > 0:
                        count = self.file_count[name]
                        self.file_count[name] += 1
                        file_name = "tmp/{}_{}.pkl".format(name, count)
                        with open(file_name, "wb") as f:
                            pickle.dump(collector[name], f)
                        collector[name] = []

        def load_value(name: str) -> List:
            if data_args.use_disk:
                value = []
                for count in range(self.file_count[name]):
                    file_name = "tmp/{}_{}.pkl".format(name, count)
                    if os.path.exists(file_name):
                        with open(file_name, "rb") as f:
                            value.extend(pickle.load(f))
                return value
            else:
                for collector in [norm_dict, attn_dict, wo_dict, ffn_dict]:
                    if name in collector:
                        return collector[name]
                raise ValueError

        block_pat = re.compile('model\.layers\.(\d+)')

        for name, module in model.named_modules():
            # print("name:", name)
            if block_pat.fullmatch(name):
                # assert isinstance(module, LlamaDecoderLayer)
                self_attn: LlamaAttention = module.self_attn
                ffn: LlamaMLP = module.mlp

                # RMS:
                self_attn_norm_name = name + '.post_attention_layernorm'
                ffn_norm_name = name + '.input_layernorm'

                handlers.append(module.post_attention_layernorm.register_forward_hook(
                    init_fn(self_attn_norm_name, norm_dict)))
                handlers.append(module.input_layernorm.register_forward_hook(
                    init_fn(ffn_norm_name, norm_dict)))

                # ffn:
                w1_name = name + '.mlp.w1'
                w2_name = name + '.mlp.w2'
                w3_name = name + '.mlp.w3'

        trainer.collect()

        for collector in [norm_dict, ffn_dict]:
            dump_value(collector)

        def dump_norm():
            token_sample_num = int(
                math.ceil(data_args.token_sample_num/config.num_hidden_layers))
            hiddens = []
            for key in norm_dict.keys():
                value = load_value(key)
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

        def shape(states):
            return states.view(-1, config.num_heads, config.d_kv).permute(1, 2, 0)

        norm_proj = dump_norm()
        model_args.comp_mode = 0

        if model_args.comp_mode == 0:
            attn_proj = None
            wo_proj = None
            ffn_value = None
        elif model_args.comp_mode == 1:
            attn_proj = dump_attn()
            wo_proj = dump_wo()
            ffn_value = dump_ffn()
        else:
            raise ValueError

        Llama_comp_params = {
            "norm_proj": norm_proj,
        }
        return Llama_comp_params

    @torch.no_grad()
    def load_model_params(
        self,
        model,
        teacher,
        state_dict,
        compression_params
    ):
        device = "cpu"
        norm_proj: torch.Tensor = compression_params["norm_proj"]
        norm_proj.to(dtype=torch.float)

        config: LlamaConfig = model.config
        block_pat = re.compile('model\.layers\.(\d+)')

        def get_norm_params(block_id: int, perfix: str):
            path = "base_model.model.model.layers.{}.{}".format(
                block_id, perfix)
            return state_dict[path + ".weight"]

        def load_qk_params(
                name: str,
                linear: torch.nn.Linear,
                norm_w: torch.Tensor,
                norm_proj: torch.Tensor,
                qk_proj: Optional[torch.Tensor] = None,
        ):
            lin_w = state_dict[name + ".weight"]
            diag0 = torch.diag(norm_w).to(device, dtype=torch.float)
            part = norm_proj.to(device, dtype=torch.float)

            n_lin_w = (lin_w @ diag0 @ part) * \
                math.sqrt(config.hidden_size / config.sub_size)
            n_lin_w.to(dtype=torch.float16)
            linear.weight.copy_(n_lin_w)

        def load_afternorm_linear(
                name: str,
                linear: torch.nn.Linear,
                norm_w: torch.Tensor,
                norm_proj: torch.Tensor,
                retain_indices: Optional[torch.Tensor] = None,
                prune_dim: Optional[int] = None,
                v_proj: Optional[torch.Tensor] = None,
        ):
            lin_w = state_dict[name + ".weight"]

            lin_w = lin_w.to(device, dtype=torch.float)
            diag0 = torch.diag(norm_w).to(device, dtype=torch.float)
            part = norm_proj.to(device, dtype=torch.float)
            n_lin_w = (lin_w @ diag0 @ part) * \
                math.sqrt(config.hidden_size / config.sub_size)
            n_lin_w.to(dtype=torch.float16)

            linear.weight.copy_(n_lin_w)

        def load_beforenorm_linear(
                name: str,
                linear: torch.nn.Linear,
                norm_proj: torch.Tensor,
                retain_indices: Optional[torch.Tensor] = None,
                prune_dim: Optional[int] = None,
                o_proj: Optional[torch.Tensor] = None,
        ):
            lin_w = state_dict[name + ".weight"]

            trans = norm_proj.T.to(device, dtype=torch.float)
            lin_w = lin_w.to(device, dtype=torch.float)
            n_lin_w = trans @ lin_w
            n_lin_w.to(dtype=torch.float16)

            linear.weight.copy_(n_lin_w)

        for n, p in model.named_parameters():
            if n not in state_dict:
                continue
            if 'lora' in n:
                continue
            if p.shape == state_dict[n].shape:
                p.copy_(state_dict[n])

        for name, module in model.named_modules():
            match_block = block_pat.fullmatch(name)
            if match_block is not None:
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
                    prune_dim=0,
                )
                load_afternorm_linear(
                    name + ".mlp.up_proj",
                    ffn.up_proj,
                    ffn_norm_w,
                    norm_proj,
                    prune_dim=0,
                )
                load_beforenorm_linear(
                    name + ".mlp.down_proj",
                    ffn.down_proj,
                    norm_proj,
                    prune_dim=1,
                )

                module.post_attention_layernorm.weight.copy_(
                    torch.ones_like(module.post_attention_layernorm.weight)
                )
                module.input_layernorm.weight.copy_(
                    torch.ones_like(module.input_layernorm.weight)
                )

        model.base_model.model.model.norm.weight.copy_(
            teacher.base_model.model.model.norm.weight)
        model.base_model.model.model.down_linear.weight.copy_(norm_proj.T)
        model.base_model.model.model.up_linear.weight.copy_(norm_proj)
