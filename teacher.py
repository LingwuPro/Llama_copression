import random
import argparse
import numpy as np
import torch
import torch.nn as nn
import transformers
from transformers import LlamaTokenizer, LlamaConfig, LlamaForCausalLM

from typing import Optional, Dict, List
from utils.eval_util import evaluate

if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--model_name', type=str, default="yahma/llama-7b-hf")
    return parser.parse_args()

def setup_seed(seed: int):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)

def run():

    args = parse_args()
    setup_seed(args.seed)

    # compression_params = torch.load(
    #     f"./svd_results/piqa-256/alpaca_{args.r_model}_params.pt", map_location='cpu')

    teacher: LlamaForCausalLM = LlamaForCausalLM.from_pretrained(args.model_name)
    # teacher: PeftModel = PeftModel.from_pretrained(teacher, lora_weights)
    # teacher: LlamaForCausalLM = teacher.merge_and_unload()
    teacher.to(device)

    tokenizer = LlamaTokenizer.from_pretrained(args.model_name)
    tokenizer.pad_token_id = 0

    evaluate(
        model=teacher,
        tokenizer=tokenizer,
        dataset_name="piqa",
        device=device
    )


if __name__ == '__main__':
    run()


"""

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
"""