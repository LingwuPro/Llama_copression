import os
import sys
import random
import numpy as np
import torch
import datetime
import transformers
from transformers import DataCollatorForSeq2Seq
from peft import PeftModel
from transformers import LlamaForCausalLM, LlamaTokenizer
from datasets import load_dataset
from utils.prompter import Prompter
from trainer import LlamaTrainer
from utils.llama_utils import LlamaUtils
from typing import Optional, List
import datasets
import json
import argparse

format_time = datetime.datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"

def setup_seed(seed: int):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=str, default="cuda:0")
    parser.add_argument('--seed', type=int, default=42)

    # for collect
    parser.add_argument('--data_sample_num', type=int, default=256)
    parser.add_argument('--token_sample_num', type=int, default=4096)
    parser.add_argument('--per_sample_token_num', type=int, default=32)
    parser.add_argument('--svd_results_dir', type=str, default="svd_results")
    parser.add_argument('--r_model', type=int, default=3072)

    return parser.parse_args()

def run(
    # model/data params
    model_name: str = "decapoda-research/llama-7b-hf",  # the only required argument  # yahma/llama-7b-hf
    data_path: str = "alpaca_data_gpt4.json",
    lora_dir: str = "./lora-alpaca",
    # training hyperparams
    batch_size: int = 128,
    micro_batch_size: int = 4,
    num_epochs: int = 3,
    learning_rate: float = 3e-4,
    cutoff_len: int = 256,
    val_set_size: int = 2000,
    # lora hyperparams
    # llm hyperparams
    train_on_inputs: bool = True,  # if False, masks out inputs in loss
    group_by_length: bool = False,  # faster, but produces an odd training loss curve
    # either training checkpoint or final adapter
    resume_from_checkpoint: str = None,
    # The prompt template to use, will default to alpaca.
    prompt_template_name: str = "alpaca",
):
    gradient_accumulation_steps = batch_size // micro_batch_size

    args = parse_args()
    setup_seed(args.seed)

    comp_utils = LlamaUtils()

    prompter = Prompter(prompt_template_name)
    tokenizer = LlamaTokenizer.from_pretrained(model_name)

    model = LlamaForCausalLM.from_pretrained(model_name)
    model = PeftModel.from_pretrained(model, lora_dir)
    model = model.merge_and_unload()

    # unwind broken decapoda-research config
    model.config.pad_token_id = tokenizer.pad_token_id = 0  # unk
    model.config.bos_token_id = 1
    model.config.eos_token_id = 2

    model.eval()
    model.to(device)

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
        if not train_on_inputs:
            user_prompt = prompter.generate_prompt(
                data_point["instruction"], data_point["input"]
            )
            tokenized_user_prompt = tokenize(user_prompt, add_eos_token=False)
            user_prompt_len = len(tokenized_user_prompt["input_ids"])

            tokenized_full_prompt["labels"] = [
                -100
            ] * user_prompt_len + tokenized_full_prompt["labels"][
                user_prompt_len:
            ]  # could be sped up, probably
        return tokenized_full_prompt

    data = load_dataset("json", data_files=data_path)

    val_data = data["train"]\
        .shuffle()\
        .select(range(args.data_sample_num))\
        .map(generate_and_tokenize_prompt)

    trainer = LlamaTrainer(
        model=model,
        train_dataset=None,
        eval_dataset=val_data,
        args=transformers.Seq2SeqTrainingArguments(
            output_dir="./collect_results",
            per_device_eval_batch_size=1
        ),
        data_collator=transformers.DataCollatorForSeq2Seq(
            tokenizer, pad_to_multiple_of=8, return_tensors="pt", padding=True
        ),
    )

    data_args = args
    model_args = args
    params = comp_utils.collect(data_args, model_args, model, trainer)
    # print(labels, goal, sol1, sol2)
    dir_path = f"{args.svd_results_dir}/piqa-{data_args.data_sample_num}"
    os.makedirs(dir_path, exist_ok=True)

    param_path = "alpaca_{}_params.pt".format(model_args.r_model)
    torch.save(params, os.path.join(dir_path, param_path))

if __name__ == '__main__':
    run()
