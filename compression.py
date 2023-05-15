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
    instruction = "choose answer for goal from sol1 and sol2."

    for idx, (row, lab) in enumerate(zip(inputs, labels)):
        data = json.loads(row)
        instructions = instruction
        inputss = ('goal: ' + data["goal"] + 
                   '; sol1: ' + data["sol1"][:64] +
                   '; sol2: ' + data["sol2"][:64])
        labeling = (lab)
        raw_dataset = {"instruction": instructions,
                       "input": inputss}
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
    max_new_tokens=256,  # 256
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

    args = parse_args()
    comp_utils = LlamaUtils()

    setup_seed(args.seed)

    compression_params = torch.load(
        f"./svd_results/piqa-256/alpaca_{args.r_model}_params.pt", map_location='cpu')
    
    model_config = LlamaConfig.from_pretrained(model_name)
    model_config.r_model = args.r_model
    model: ReduLlamaForCausalLM = ReduLlamaForCausalLM(model_config)
    model.config.pad_token_id = 0  # unk
    model.config.bos_token_id = 1
    model.config.eos_token_id = 2

    teacher: LlamaForCausalLM = LlamaForCausalLM.from_pretrained(model_name)
    teacher: PeftModel = PeftModel.from_pretrained(teacher, lora_weights)
    teacher: LlamaForCausalLM = teacher.merge_and_unload()
    teacher.config.pad_token_id = 0  # unk
    teacher.config.bos_token_id = 1
    teacher.config.eos_token_id = 2

    state_dict: Dict[str, torch.Tensor] = teacher.state_dict()

    comp_utils.load_model_params(
        model, state_dict, compression_params)
    
    peft_dict = {
        "base_model_name_or_path": "llama-7b-hf",
        "bias": "none",
        "fan_in_fan_out": False,
        "lora_alpha": 16,
        "lora_dropout": 0.05,
        "modules_to_save": None,
        "peft_type": "LORA",
        "r": 16,
        "target_modules": [
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "down_proj",
            "up_proj"
        ],
        "task_type": "CAUSAL_LM"
    }
    peft_config = get_peft_config(peft_dict)
    model = PeftModel(model, peft_config)

    prompter = Prompter(prompt_template)
    tokenizer = LlamaTokenizer.from_pretrained(model_name)

    tokenizer.pad_token_id = 0

    def tokenize(prompt, add_eos_token=True):
        # there's probably a way to do this with the tokenizer settings
        # but again, gotta move fast
        result = tokenizer(
            prompt,
            truncation=True,
            max_length=cutoff_len,
            padding=False,
            return_tensors=None,
        )
        if (
            result["input_ids"][-1] != tokenizer.eos_token_id
            and len(result["input_ids"]) < cutoff_len
            and add_eos_token
        ):
            result["input_ids"].append(tokenizer.eos_token_id)
            result["attention_mask"].append(1)

        result["labels"] = result["input_ids"].copy()

        return result

    def generate_and_tokenize_prompt(data_point):
        full_prompt = prompter.generate_prompt(
            data_point["instruction"],
            data_point["input"],
            data_point["output"],
        )
        tokenized_full_prompt = tokenize(full_prompt)
        return tokenized_full_prompt


    if data_path.endswith(".json") or data_path.endswith(".jsonl"):
        data = load_dataset("json", data_files=data_path)
    else:
        data = load_dataset(data_path)

    train_val = data['train'].train_test_split(
        test_size=0.1, shuffle=True, seed=42
    )
    train_data = (
        train_val["train"].shuffle().map(generate_and_tokenize_prompt)
    )
    val_data = (
        train_val["test"].shuffle().map(generate_and_tokenize_prompt)
    )
    save_columns = ['input_ids', 'attention_mask', 'labels']

    train_data = train_data.remove_columns(
        set(data['train'].column_names) - set(save_columns))
    val_data = val_data.remove_columns(
        set(data['train'].column_names) - set(save_columns))

    trainer = transformers.Trainer(
        model=model,
        train_dataset=train_data,
        eval_dataset=None,
        args=transformers.TrainingArguments(
            per_device_train_batch_size=4,
            gradient_accumulation_steps=32,
            warmup_steps=100,
            num_train_epochs=1,
            learning_rate=3e-4,
            logging_steps=10,
            optim="adamw_torch",
            evaluation_strategy="no",
            eval_steps=None,
            save_strategy="steps",
            save_steps=30,
            output_dir=output_dir,
            save_total_limit=3,
            load_best_model_at_end=False,
            ddp_find_unused_parameters=False if int(
                os.environ.get("WORLD_SIZE", 1)) != 1 else None,
            group_by_length=False,
            remove_unused_columns=False
        ),
        data_collator=transformers.DataCollatorForSeq2Seq(
            tokenizer, pad_to_multiple_of=8, return_tensors="pt", padding=True
        ),
    )

    # trainer.train()
    # model.save_pretrained(output_dir)
    
    model.to(device)

    dev_dataset = batch2dict('./metric/piqa/inputs/valid.jsonl')

    answer = []

    for data in tqdm(dev_dataset, total=len(dev_dataset)):
        answer.append(evaluate(model, data['instruction'], input=data['input']))

    print(answer)

    with open("answer.txt", 'w') as f:
        for i in answer:
            f.write(i + '\n')

if __name__ == '__main__':
    run()
