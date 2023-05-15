import re
import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"


from torch.utils.data import Dataset
import torch
import torch.nn as nn
import transformers
import numpy as np
import random
from sklearn.model_selection import train_test_split
from typing import Optional, Dict, List
import argparse


import json
from utils.llama_utils import LlamaUtils
from utils.prompter import Prompter
from transformers import GenerationConfig, LlamaTokenizer, LlamaConfig, LlamaForCausalLM
from model.modeling_llama import LlamaForCausalLM as ReduLlamaForCausalLM
from peft import PeftModel, get_peft_config, set_peft_model_state_dict


from datasets import load_dataset
from transformers import DataCollatorForSeq2Seq
from tqdm import tqdm


if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"
input_filepath = './metric/piqa/inputs/train.jsonl'
label_filepath = './metric/piqa/label/train-labels.lst'

data_path = 'alpaca_data_gpt4.json'
model_name: str = "decapoda-research/llama-7b-hf"
lora_weights: str = "./lora-alpaca"

# The prompt template to use, will default to alpaca.
prompt_template: str = "alpaca"
cutoff_len: int = 256

output_dir = './checkpoint'

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--r_model', type=int, default=3072)
    return parser.parse_args()

def setup_seed(seed: int):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)


def batch2dict(input_filepath, label_filepath=None):
    with open(input_filepath, encoding="utf-8") as input_file:
        inputs = input_file.read().splitlines()

    if label_filepath is not None:
        with open(label_filepath, encoding="utf-8") as label_file:
            labels = label_file.read().splitlines()
    else:
        # Labels are not available for the test set.
        # Filling the `label` column with -1 by default
        labels = [-1] * len(inputs)

    ans = []
    instruction = "choose the coherent one of the sol, output 0 for sol1, or 1 for sol2."

    for idx, (row, lab) in enumerate(zip(inputs, labels)):
        data = json.loads(row)
        instructions = (instruction)
        inputss = ('sol1: ' + data["goal"] + data["sol1"][:64] +
                   '; sol2: ' + data["goal"] + data["sol2"][:64])
        labeling = (lab)
        raw_dataset = {"instruction": instructions,
                       "input": inputss,
                       "output": labeling}
        ans.append(raw_dataset)

    return ans

def evaluate(
    model,
    instruction,
    input=None,
    temperature=0.1,
    top_p=0.75,
    top_k=40,
    num_beams=1,
    max_new_tokens=10,  # 256
    **kwargs,
):
    prompt = prompter.generate_prompt(instruction, input)
    inputs = tokenizer(prompt, return_tensors="pt")
    input_ids = inputs["input_ids"].to(device)
    generation_config = GenerationConfig(
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        num_beams=num_beams,
        **kwargs,
    )

    with torch.no_grad():
        generation_output = model.generate(
            input_ids=input_ids,
            generation_config=generation_config,
            return_dict_in_generate=True,
            output_scores=True,
            max_new_tokens=max_new_tokens,
        )
    s = generation_output.sequences[0]
    output = tokenizer.decode(s)
    return prompter.get_response(output)

def run():
    global prompter
    global tokenizer

    prompter = Prompter(prompt_template)
    tokenizer = LlamaTokenizer.from_pretrained(model_name)

    args = parse_args()
    comp_utils = LlamaUtils()

    setup_seed(args.seed)

    # compression_params = torch.load(
    #     f"./svd_results/piqa-256/alpaca_{args.r_model}_params.pt", map_location='cpu')

    teacher: LlamaForCausalLM = LlamaForCausalLM.from_pretrained(model_name)
    teacher: PeftModel = PeftModel.from_pretrained(teacher, lora_weights)
    teacher: LlamaForCausalLM = teacher.merge_and_unload()
    teacher.config.pad_token_id = 0  # unk
    teacher.config.bos_token_id = 1
    teacher.config.eos_token_id = 2

    teacher.to(device)

    dev_dataset = batch2dict('./metric/piqa/inputs/valid.jsonl')

    answer = []

    for data in tqdm(dev_dataset, total=len(dev_dataset)):
        answer.append(evaluate(teacher, data['instruction'], input=data['input']))

    print(answer)

    with open("answer.txt", 'w') as f:
        for i in answer:
            f.write(i + '\n')

if __name__ == '__main__':
    run()
