import random
import argparse
import numpy as np
import torch
import torch.nn as nn
import transformers
from transformers import LlamaConfig, LlamaTokenizer
from model.modeling_llama import LlamaForCausalLM as ReduLlamaForCausalLM
from peft import PeftModel, get_peft_config, set_peft_model_state_dict
from typing import Optional, Dict, List
from utils.eval_util import evaluate
from transformers import AutoTokenizer, AutoModelForCausalLM
if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--model_name', type=str,
                        default="AlexWortega/LLama2-7b")
    parser.add_argument('--base_model_path', type=str,
                        default="./checkpoint/base")
    parser.add_argument('--r_model', type=int, default=3584)
    parser.add_argument('--lora_model_path', type=str,
                        default="./checkpoint/lora")
    return parser.parse_args()


def setup_seed(seed: int):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)


def run():

    args = parse_args()
    setup_seed(args.seed)

    model_config = LlamaConfig.from_pretrained(args.model_name)
    model_config.r_model = args.r_model
    model: ReduLlamaForCausalLM = ReduLlamaForCausalLM.from_pretrained(
        args.base_model_path, config=model_config)
    model: PeftModel = PeftModel.from_pretrained(model, args.lora_model_path)
    model.to(device)

    tokenizer = LlamaTokenizer.from_pretrained(args.model_name)
    tokenizer.pad_token_id = 0

    for dataset_name in ["piqa", "hella", "winogrande"]:
        evaluate(
            model=model,
            tokenizer=tokenizer,
            dataset_name=dataset_name,
            device=device
        )


if __name__ == '__main__':
    run()
