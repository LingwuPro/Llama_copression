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

        block_pat = re.compile('base_model\.model\.model\.layers\.(\d+)')
        self_attn_pat = re.compile(
            'base_model\.model\.model\.layers\.(\d+)\.input_layernorm')
        ffn_pat = re.compile(
            'base_model\.model\.model\.layers\.(\d+)\.post_attention_layernorm')

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

                # # attn:
                # q_name = name + '.self_attn.q_proj'
                # k_name = name + '.self_attn.k_proj'
                # o_name = name + '.self_attn.o_proj'

                # handlers.append(self_attn.q_proj.register_forward_hook(
                #     init_fn(q_name, attn_dict, collect_input=False)))
                # handlers.append(self_attn.k_proj.register_forward_hook(
                #     init_fn(k_name, attn_dict, collect_input=False)))
                # handlers.append(self_attn.o_proj.register_forward_hook(
                #     init_fn(k_name, attn_dict)))

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

        # def dump_attn():
        #     attn_value = {}
        #     r_kv = model_args.r_kv
        #     keys = sorted(list(set(x.replace('.q_proj', '').replace('.k_proj', '')
        #                            for x in attn_dict.keys())))

        #     for key in keys:
        #         print('key={}'.format(key))
        #         hidden_query = load_value(key + '.q_proj')
        #         hidden_key = load_value(key + '.k_proj')

        #         hidden_query = torch.concat(hidden_query, dim=0)
        #         hidden_key = torch.concat(hidden_key, dim=0)
        #         size = min(hidden_query.shape[0], data_args.token_sample_num)
        #         indices = np.random.choice(
        #             hidden_query.shape[0], size=size, replace=False)
        #         hidden_query = shape(hidden_query[indices, :])
        #         hidden_key = shape(hidden_key[indices, :])
        #         Uq, Sq, _ = torch.linalg.svd(hidden_query)
        #         Uk, Sk, _ = torch.linalg.svd(hidden_key)

        #         M = torch.diag_embed(Sq) @ Uq.transpose(1,
        #                                                 2) @ Uk @ torch.diag_embed(Sk)
        #         Um, Sm, VmT = torch.linalg.svd(M)
        #         UT = Uq @ torch.diag_embed(1.0 / Sq) @ Um[..., :r_kv] @ torch.diag_embed(
        #             torch.sqrt(Sm[:, :r_kv]))
        #         V = torch.diag_embed(torch.sqrt(Sm[:, :r_kv])) @ VmT[:, :r_kv, :] @ torch.diag_embed(
        #             1.0 / Sk) @ Uk.transpose(1, 2)
        #         attn_value[key + '.wq_proj'] = UT.transpose(1, 2)
        #         attn_value[key + '.wk_proj'] = V
        #     return attn_value

        # def dump_wo():
        #     wo_value = {}
        #     for key in wo_dict.keys():
        #         print("key={}".format(key))
        #         value: List[torch.Tensor] = load_value(key)
        #         hidden = torch.concat(value, dim=0)
        #         size = min(hidden.shape[0], data_args.token_sample_num)
        #         indices = np.random.choice(
        #             hidden.shape[0], size=size, replace=False)
        #         hidden = shape(hidden[indices, :])
        #         U, S, _ = torch.linalg.svd(hidden)
        #         wo_value[key] = U[..., :model_args.r_kv]
        #     return wo_value

        # def dump_ffn():
            ffn_value = {}
            model_state_dict = model.state_dict()
            for key in ffn_dict.keys():
                print("key={}".format(key))
                value: List[Tuple] = load_value(key)
                ws, hs, h2s = zip(*value)
                ws = torch.tensor(ws).unsqueeze(-1)
                hs = torch.stack(hs)
                h2s = torch.stack(h2s)

                h_mean = (ws * hs).sum(dim=0) / ws.sum()
                h2_mean = (ws * h2s).sum(dim=0) / ws.sum()
                h_std = (h2_mean - h_mean * h_mean + 1e-5).sqrt()
                weight: torch.Tensor = model_state_dict[key +
                                                        ".weight"].transpose(0, 1).detach().cpu()
                norm = weight.norm(dim=1)
                h_value: torch.Tensor = (h_mean + h_std) * norm
                ffn_value[key] = h_value
            return ffn_value

        norm_proj = dump_norm()
        # ffn_value = dump_ffn()
        # print(norm_proj)
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
            # 'attn_proj': attn_proj,
            # 'wo_proj': wo_proj,
            'ffn_value': ffn_value
        }
        return Llama_comp_params

    @torch.no_grad()
    def load_model_params(
        self,
        model,
        state_dict,  # finetune
        compression_params  # collect
    ):
        norm_proj: torch.Tensor = compression_params["norm_proj"]
        # print(norm_proj)
        # attn_proj: Optional[Dict[str, torch.Tensor]
        #                     ] = compression_params["attn_proj"]
        # wo_proj: Optional[Dict[str, torch.Tensor]
        #                   ] = compression_params["wo_proj"]
        # ffn_value: Optional[Dict[str, torch.Tensor]
        #                     ] = compression_params["ffn_value"]

        # if attn_proj is None:
        attn_proj = defaultdict(lambda: None)
        # if wo_proj is None:
        wo_proj = defaultdict(lambda: None)
        # if act_value is None:
        act_value = defaultdict(lambda: None)

        config: LlamaConfig = model.config
        block_pat = re.compile('base_model\.model\.model\.layers\.(\d+)')

        def get_norm_params(block_id: int, perfix: str):
            path = "base_model.model.model.layers.{}.{}".format(
                block_id, perfix)
            return state_dict[path].weight

        def load_qk_params(
                name: str,
                linear: torch.nn.Linear,
                norm_w: torch.Tensor,
                norm_proj: torch.Tensor,
                qk_proj: Optional[torch.Tensor] = None,
        ):
            # num_heads = config.num_heads
            # d_kv = config.d_kv
            # r_kv = config.r_kv
            # print(state_dict[name])
            lin_w_drop = state_dict[name].weight.to(torch.float)
            # print("Dropout: ", lin_w_drop)
            lin_w_A = state_dict[name +
                                 '.lora_A']['default'].weight.to(torch.float)
            lin_w_B = state_dict[name +
                                 '.lora_B']['default'].weight.to(torch.float)
            # print(lin_w_A)
            # exit(0)
            lin_w = (lin_w_drop + lin_w_B @
                     lin_w_A).to(device, dtype=torch.float)
            diag0 = torch.diag(norm_w).to(device, dtype=torch.float)
            part = norm_proj.to(device, dtype=torch.float)

            # lin_w = state_dict[name + '.weight']
            n_lin_w = lin_w @ diag0 @ part
            n_lin_w.to(dtype=torch.float16)
            # if qk_proj is not None:
            #     n_lin_w = n_lin_w.view(num_heads, d_kv, config.r_model)
            #     n_lin_w = (qk_proj @ n_lin_w).view(num_heads *
            #                                        r_kv, config.r_model)

            linear.weight.copy_(n_lin_w)

        def load_afternorm_linear(
                name: str,
                linear: torch.nn.Linear,
                norm_w: torch.Tensor,
                norm_proj: torch.Tensor,
                retain_indices: Optional[torch.Tensor] = None,
                prune_dim: Optional[int] = None,
                v_proj: Optional[torch.Tensor] = None,
                isFFN=False
        ):
            if not isFFN:
                lin_w_A = state_dict[name +
                                     '.lora_A']['default'].weight.to(torch.float)
                lin_w_B = state_dict[name +
                                     '.lora_B']['default'].weight.to(torch.float)
                lin_w_drop = state_dict[name].weight.to(torch.float)
                lin_w = lin_w_drop + lin_w_B @ lin_w_A
            else:
                lin_w = state_dict[name].weight

            lin_w = lin_w.to(device, dtype=torch.float)
            diag0 = torch.diag(norm_w).to(device, dtype=torch.float)
            part = norm_proj.to(device, dtype=torch.float)

            n_lin_w = (lin_w @ diag0 @ part) * \
                math.sqrt(config.hidden_size / 3072)

            n_lin_w.to(dtype=torch.float16)

            # if retain_indices is not None:
            #     n_lin_w = prune_ffn_weight(n_lin_w, retain_indices, prune_dim)

            # if v_proj is not None:
            #     n_lin_w = n_lin_w.view(
            #         config.num_heads, config.d_kv, config.r_model)
            #     n_lin_w = torch.matmul(v_proj.transpose(1, 2), n_lin_w)
            #     n_lin_w = n_lin_w.reshape(
            #         config.num_heads * config.r_kv, config.r_model)

            linear.weight.copy_(n_lin_w)

        def load_beforenorm_linear(
                name: str,
                linear: torch.nn.Linear,
                norm_proj: torch.Tensor,
                retain_indices: Optional[torch.Tensor] = None,
                prune_dim: Optional[int] = None,
                o_proj: Optional[torch.Tensor] = None,
                isFFN=False
        ):
            if not isFFN:
                lin_w_A = state_dict[name +
                                     '.lora_A']['default'].weight.to(torch.float)
                lin_w_B = state_dict[name +
                                     '.lora_B']['default'].weight.to(torch.float)
                lin_w_drop = state_dict[name].weight.to(torch.float)
                lin_w = lin_w_drop + lin_w_B @ lin_w_A
            else:
                lin_w = state_dict[name].weight

            trans = norm_proj.T.to(device, dtype=torch.float)
            lin_w = lin_w.to(device, dtype=torch.float)
            n_lin_w = trans @ lin_w
            n_lin_w.to(dtype=torch.float16)

            # if retain_indices is not None:
            #     n_lin_w = prune_ffn_weight(n_lin_w, retain_indices, prune_dim)

            # if o_proj is not None:
            #     n_lin_w = n_lin_w.view(
            #         config.r_model, config.num_heads, config.d_kv).permute(1, 0, 2)
            #     n_lin_w = torch.matmul(n_lin_w, o_proj)
            #     n_lin_w = n_lin_w.permute(1, 0, 2).reshape(
            #         config.r_model, config.num_heads * config.r_kv)

            linear.weight.copy_(n_lin_w)

        # def load_act_indices(
        #         name: str,
        # ):
        #     h_value = act_value[name]
        #     act_indices = h_value.sort(
        #     )[1][-config.r_ff:] if h_value is not None else None
        #     return act_indices

        for n, p in model.named_parameters():
            if n not in state_dict:
                continue
            if p.shape == state_dict[n].shape:
                p.copy_(state_dict[n])

        for name, module in model.named_modules():
            # print("name: ", name)
            match_block = block_pat.fullmatch(name)
            if match_block is not None:
                # assert isinstance(module, LlamaDecoderLayer)
                self_attn: LlamaAttention = module.self_attn
                ffn: LlamaMLP = module.mlp

                block_id = int(match_block.group(1))
                att_norm_w = get_norm_params(
                    block_id, "post_attention_layernorm")
                ffn_norm_w = get_norm_params(block_id, 'input_layernorm')

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
                    isFFN=True
                )
                load_afternorm_linear(
                    name + ".mlp.up_proj",
                    ffn.up_proj,
                    ffn_norm_w,
                    norm_proj,
                    prune_dim=0,
                    isFFN=True
                )
                load_beforenorm_linear(
                    name + ".mlp.down_proj",
                    ffn.down_proj,
                    norm_proj,
                    prune_dim=1,
                    isFFN=True
                )

                module.post_attention_layernorm.weight.copy_(
                    torch.ones_like(module.post_attention_layernorm.weight)
                )
                module.input_layernorm.weight.copy_(
                    torch.ones_like(module.input_layernorm.weight)
                )

        model.base_model.model.model.norm.weight.copy_(
            torch.ones_like(model.base_model.model.model.norm.weight))

        model.base_model.model.model.down_linear.weight.copy_(norm_proj.T)

        temp0 = norm_proj.to(dtype=torch.float)
        temp1 = torch.diag(
            model.base_model.model.model.norm.weight).to(dtype=torch.float)
        temp2 = temp1.T @ temp0
        temp2 = temp2.to(dtype=torch.float16)
        model.base_model.model.model.up_linear.weight.copy_(temp2)
