
import os
import random
import numpy as np
import torch
import datetime
import evaluate

from transformers import GenerationConfig, LlamaForCausalLM, LlamaTokenizer

from utils.callbacks import Iteratorize, Stream
from utils.prompter import Prompter
from utils.llama_utils import LlamaUtils

from transformers.utils import logging
from datasets import DatasetDict, Dataset, load_from_disk, load_metric
from typing import Optional, Dict, List
from loguru import logger

import argparse

format_time = datetime.datetime.now().strftime("%Y-%m-%d_%H:%M:%S")


def setup_seed(seed: int):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default="glue")
    parser.add_argument('--task_name', type=str, default=None)
    parser.add_argument('--model_name', type=str, default="t5-base")
    parser.add_argument('--device', type=str, default="cuda:0")
    parser.add_argument('--seed', type=int, default=42)

    # for train
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--weight_decay', type=float, default=0)
    parser.add_argument('--adam_epsilon', type=float, default=1e-8)
    parser.add_argument('--num_warump_steps', type=int, default=0)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--num_epochs', type=int, default=2)
    parser.add_argument('--max_grad_norm', type=float, default=1.0)
    parser.add_argument('--shuffle', action='store_true')

    return parser.parse_args()


def run():
    args = parse_args()
    assert args.dataset == "glue"

    setup_seed(args.seed)
    try:
        raw_datasets = load_from_disk(
            '../../.cache/datasets/{}/{}'.format(args.dataset, args.task_name))
    except:
        logger.error('dataset: [{}] does not exist'.format(args.dataset))
        raise
    assert isinstance(raw_datasets, DatasetDict)

    tokenizer = T5Tokenizer.from_pretrained(
        args.model_name, cache_dir=f"../../.cache/models")
    config = T5Config.from_pretrained(
        args.model_name, cache_dir=f"../../.cache/models")
    model = T5ForConditionalGeneration(config)

    checkpoint_path = "../../.cache/checkpoint/{}-{}-{}.pth".format(
        args.dataset, args.task_name, args.model_name)
    state_dict: Dict[str, torch.Tensor] = torch.load(
        checkpoint_path, map_location='cpu')
    model.load_state_dict(state_dict)

    glue_utils = GlueUtils()
    preprocess_function = glue_utils.init_glue_preprocess_function(
        args, tokenizer)

    for key in list(raw_datasets.keys()):
        if key.startswith("test"):
            del raw_datasets[key]

    raw_datasets = raw_datasets.map(
        preprocess_function,
        batched=True,
        desc="Running tokenizer on dataset",
    )
    for key, dataset in raw_datasets.items():
        raw_datasets[key] = dataset.remove_columns(
            set(dataset.column_names) - set(glue_utils.columns))

    datasets: Dict[str, Dataset] = {
        'train': raw_datasets['train'],
        'validation': raw_datasets["validation_matched" if args.task_name == "mnli" else "validation"]
    }

    data_collator = DataCollatorForSeq2Seq(tokenizer)

    # metric = evaluate.load(args.dataset, args.task_name)
    metric = load_metric("metric/" + args.dataset + ".py", args.task_name)
    trainer = T5GlueTrainer(args, model, datasets,
                            tokenizer, data_collator, metric)

    validation_resutls = trainer.evaluate(
        "validation",
        datasets["validation"]
    )
    for metric, rest in validation_resutls.items():
        logger.info("teacher-[validation] {:<6}: {:.5}".format(metric, rest))


if __name__ == '__main__':
    run()
