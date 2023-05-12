import os
import sys
import random
import numpy as np
import torch
import datetime
from transformers import DataCollatorForSeq2Seq
from peft import PeftModel
from transformers import GenerationConfig, LlamaForCausalLM, LlamaTokenizer
from utils.callbacks import Iteratorize, Stream
from utils.prompter import Prompter
from utils.piqatrainer import LlamaPiqaTrainer
from utils.llama_utils import LlamaUtils
from utils.piqautils import PiqaUtils
import datasets
import json
import argparse

format_time = datetime.datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"

base_model: str = "decapoda-research/llama-7b-hf"
lora_weights: str = "./lora-alpaca"
# The prompt template to use, will default to alpaca.
prompt_template: str = "alpaca"

prompter = Prompter(prompt_template)
tokenizer = LlamaTokenizer.from_pretrained(base_model)
model = LlamaForCausalLM.from_pretrained(
    base_model,
    load_in_8bit=False,
    torch_dtype=torch.float16,
    device_map="auto",
)
model = PeftModel.from_pretrained(
    model,
    lora_weights,
    torch_dtype=torch.float16,
)

# unwind broken decapoda-research config
model.config.pad_token_id = tokenizer.pad_token_id = 0  # unk
model.config.bos_token_id = 1
model.config.eos_token_id = 2

model.half()  # seems to fix bugs for some users.

model.eval()
# if torch.__version__ >= "2" and sys.platform != "win32":
#     model = torch.compile(model)


def setup_seed(seed: int):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default="piqa")
    parser.add_argument('--task_name', type=str, default=None)
    parser.add_argument('--model_name', type=str, default="alpaca")
    parser.add_argument('--device', type=str, default="cuda:0")
    parser.add_argument('--seed', type=int, default=42)

    # for collect
    parser.add_argument('--use_disk', action="store_true")
    parser.add_argument('--use_tmp',  action="store_true")
    parser.add_argument('--data_sample_num', type=int, default=4096)
    parser.add_argument('--token_sample_num', type=int, default=4096)
    parser.add_argument('--per_sample_token_num', type=int, default=10)
    parser.add_argument('--svd_results_dir', type=str, default="svd_results")
    parser.add_argument('--comp_mode', type=int, default=0)
    parser.add_argument('--r_model', type=int, default=3072)
    parser.add_argument('--r_kv', type=int, default=48)

    return parser.parse_args()


def evaluate(
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
    args = parse_args()
    comp_utils = LlamaUtils()
    piqa_utils = PiqaUtils(args, "goal", "sol1", "sol2", tokenizer)

    input_filepath = './metric/piqa/inputs/train.jsonl'
    label_filepath = './metric/piqa/label/train-labels.lst'

    setup_seed(args.seed)

    with open(input_filepath, encoding="utf-8") as input_file:
        inputs = input_file.read().splitlines()

    if label_filepath is not None:
        with open(label_filepath, encoding="utf-8") as label_file:
            labels = label_file.read().splitlines()
    else:
        # Labels are not available for the test set.
        # Filling the `label` column with -1 by default
        labels = [-1] * len(inputs)
    goal = []
    sol1 = []
    sol2 = []
    for idx, (row, lab) in enumerate(zip(inputs, labels)):
        data = json.loads(row)
        goal.append(data["goal"][:args.data_sample_num])
        sol1.append(data["sol1"][:args.data_sample_num])
        sol2.append(data["sol2"][:args.data_sample_num])
    example = {"goal": goal,
               "sol1": sol1,
               "sol2": sol2}

    data_collator = DataCollatorForSeq2Seq(tokenizer)
    args.data_sample_num = min(args.data_sample_num, len(goal))

    data_args = args
    model_args = args

    trainer = LlamaPiqaTrainer(
        args, model, example, prompter, tokenizer, data_collator)
    params = comp_utils.collect(data_args, model_args, model, trainer)
    # print(labels, goal, sol1, sol2)
    dir_path = f"{args.svd_results_dir}/{args.dataset}-{data_args.data_sample_num}"
    os.makedirs(dir_path, exist_ok=True)

    param_path = "{}_{}_{}_{}_params.pt".format(
        model_args.model_name,
        model_args.r_model,
        model_args.r_kv,
        model_args.comp_mode
    )
    torch.save(params, os.path.join(dir_path, param_path))


if __name__ == '__main__':
    run()
