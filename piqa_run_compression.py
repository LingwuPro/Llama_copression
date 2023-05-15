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
from utils.piqautils import PiqaUtils
from utils.llama_utils import LlamaUtils
from utils.piqatrainer import LlamaPiqaTrainer
from utils.prompter import Prompter


from utils.callbacks import Iteratorize, Stream
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

data_path = './train.jsonl'
base_model: str = "decapoda-research/llama-7b-hf"
lora_weights: str = "./lora-alpaca"
# The prompt template to use, will default to alpaca.
prompt_template: str = "alpaca"

prompter = Prompter(prompt_template)
tokenizer = LlamaTokenizer.from_pretrained(base_model)

output_dir = './checkpoint'


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
    # goal = []
    # sol1 = []
    # sol2 = []
    # inputss = []
    # labeling = []
    # instructions = []
    ans = []
    instruction = "choose the coherent one of the sol, output 0 for sol1,or     1 for sol2."

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

    # raw_dataset = MyDataset(raw_dataset)

    return ans


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default="glue")
    parser.add_argument('--task_name', type=str, default=None)
    parser.add_argument('--model_name', type=str, default="t5-base")
    parser.add_argument('--weight_decay', type=float, default=0)
    parser.add_argument('--adam_epsilon', type=float, default=1e-8)
    parser.add_argument('--num_warump_steps', type=int, default=0)
    parser.add_argument('--max_grad_norm', type=float, default=1.0)
    parser.add_argument('--shuffle', action='store_true')
    parser.add_argument('--device', type=str, default="cuda:0")
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--lambda_logits', type=float, default=0.5)
    parser.add_argument('--lambda_hiddens', type=float, default=0.5)

    parser.add_argument('--lr', type=float, default=1e-5)
    parser.add_argument('--num_epochs', type=int, default=2)
    parser.add_argument('--batch_size', type=int, default=16)

    parser.add_argument('--comp_mode', type=int, default=0)
    parser.add_argument('--use_subset', action="store_true")
    parser.add_argument('--data_sample_num', type=int, default=20)
    parser.add_argument('--r_model', type=int, default=3072)
    parser.add_argument('--r_kv', type=int, default=48)
    parser.add_argument('--r_ff', type=int, default=2304)
    parser.add_argument('--en_num_layers', type=int, default=12)
    parser.add_argument('--de_num_layers', type=int, default=12)

    parser.add_argument('--svd_results_dir', type=str, default="svd_results")
    parser.add_argument('--use_random', action="store_true")
    return parser.parse_args()


def evaluate(
    models,
    instruction,
    input=None,
    temperature=0.1,
    top_p=0.75,
    top_k=40,
    num_beams=4,
    max_new_tokens=128,
    stream_output=False,
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

    generate_params = {
        "input_ids": input_ids,
        "generation_config": generation_config,
        "return_dict_in_generate": True,
        "output_scores": True,
        "max_new_tokens": max_new_tokens,
    }

    with torch.no_grad():
        generation_output = models.generate(
            input_ids=input_ids,
            generation_config=generation_config,
            return_dict_in_generate=True,
            output_scores=True,
            max_new_tokens=max_new_tokens,
        )
    s = generation_output.sequences[0]
    output = tokenizer.decode(s)
    return prompter.get_response(output)


def tokenize(prompt, add_eos_token=True):
    # there's probably a way to do this with the tokenizer settings
    # but again, gotta move fast
    result = tokenizer(
        prompt,
        truncation=True,
        max_length=256,
        padding=False,
        return_tensors=None,
    )
    if (
        result["input_ids"][-1] != tokenizer.eos_token_id
        and len(result["input_ids"]) < 256
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
    # if not train_on_inputs:
    #     user_prompt = prompter.generate_prompt(
    #         data_point["instruction"], data_point["input"]
    #     )
    #     tokenized_user_prompt = tokenize(user_prompt, add_eos_token=False)
    #     user_prompt_len = len(tokenized_user_prompt["input_ids"])

    #     tokenized_full_prompt["labels"] = [
    #         -100
    #     ] * user_prompt_len + tokenized_full_prompt["labels"][
    #         user_prompt_len:
    #     ]  # could be sped up, probably
    return tokenized_full_prompt


def run():
    args = parse_args()
    comp_utils = LlamaUtils()
    # piqa_utils = piqa_utils()

    setup_seed(args.seed)

    compression_params = torch.load(
        "./svd_results/piqa-4096/alpaca_3072_48_0_params.pt", map_location='cpu')

    model_config = LlamaConfig.from_pretrained(base_model)
    model = ReduLlamaForCausalLM(model_config)
    # model = LlamaForCausalLM(model_config)

    teacher = LlamaForCausalLM.from_pretrained(
        base_model,
        load_in_8bit=False,
        torch_dtype=torch.float16,
        device_map="auto",
    )

    teacher = PeftModel.from_pretrained(
        teacher,
        lora_weights,
        torch_dtype=torch.float16,
    )

    teacher.config.pad_token_id = tokenizer.pad_token_id = 0  # unk
    teacher.config.bos_token_id = 1
    teacher.config.eos_token_id = 2

    teacher.half()

    teacher.eval()
    teacher.merge_adapter()
    teacher.to("cpu")
    state_dict: Dict[str, torch.Tensor] = {}

    state_dict = teacher.state_dict()
    # print(state_dict)

    peft_dict = {
        "base_model_name_or_path": "decapoda-research/llama-7b-hf",
        "bias": "none",
        # "enable_lora": None,
        "fan_in_fan_out": False,
        # "inference_mode": True,
        "lora_alpha": 16,
        "lora_dropout": 0.05,
        # "merge_weights": False,
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

    comp_utils.load_model_params(
        model, teacher, state_dict, compression_params)

    # print(model)
    # exit(0)

    # unwind broken decapoda-research config
    model.config.pad_token_id = tokenizer.pad_token_id = 0  # unk
    model.config.bos_token_id = 1
    model.config.eos_token_id = 2

    model.half()
    model.eval()

    model.to(device)

    resume_from_checkpoint = './checkpoint'
    if resume_from_checkpoint:

        checkpoint_name = os.path.join(
            resume_from_checkpoint, "pytorch_model.bin"
        )
        if not os.path.exists(checkpoint_name):
            checkpoint_name = os.path.join(
                resume_from_checkpoint, "adapter_model.bin"
            )  # only LoRA model - LoRA config above has to fit
            resume_from_checkpoint = (
                False  # So the trainer won't try loading its state
            )
        # The two files above have a different name depending on how they were saved, but are actually the same.
        if os.path.exists(checkpoint_name):
            print(f"Restarting from {checkpoint_name}")
            adapters_weights = torch.load(checkpoint_name)
            # print(adapters_weights)
            set_peft_model_state_dict(model, adapters_weights)
        else:
            print(
                f"Checkpoint {checkpoint_name} not found.\nWe will start a new trainer")

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
                model=teacher,
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

            trainer.train()
            model.save_pretrained(output_dir)

    dev_dataset = batch2dict('./metric/piqa/inputs/tests-tiny.jsonl')

    answer = []

    for data in tqdm(dev_dataset, total=len(dev_dataset)):
        answer.append(
            evaluate(teacher, data['instruction'], input=data['input']))

    print(answer)

    with open("answer.txt", 'w') as f:
        for i in answer:
            f.write(i + '\n')


run()
