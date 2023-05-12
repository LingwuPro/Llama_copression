import random
import numpy as np
import math
import torch
from transformers import GenerationConfig, LlamaForCausalLM, LlamaTokenizer
from tqdm import tqdm
from utils.piqautils import PiqaUtils
from torch.utils.data import Subset

from typing import Dict, List, Any, Tuple, Callable, Union, Optional, Sequence


class LlamaPiqaTrainer:
    def __init__(self,
                 args,
                 model,
                 dataset: Dict[str, List],
                 prompter,
                 tokenizer,
                 data_collator: Callable
                 ) -> None:
        self.args = args
        self.model = model
        self.datasets = dataset
        self.prompter = prompter
        self.tokenizer = tokenizer
        self.data_collator = data_collator
        if torch.cuda.is_available():
            self.device = "cuda"
        else:
            self.device = "cpu"

    @torch.no_grad()
    def collect(self):
        args = self.args
        dataset_utils = PiqaUtils(
            self.args, "goal", "sol1", "sol2")

        dataset = self.datasets
        indices = min(len(dataset["goal"]), args.data_sample_num)
        dataset["goal"] = dataset["goal"][:indices]
        dataset["sol1"] = dataset["sol1"][:indices]
        dataset["sol2"] = dataset["sol2"][:indices]

        instructions, inputs = dataset_utils.preprocess_function(
            dataset, "goal", "sol1", "sol2")
        generation_config = GenerationConfig(
            temperature=0.1,
            top_p=0.75,
            top_k=40,
            num_beams=4,
        )
        for instruction, input in tqdm(zip(instructions, inputs)):
            prompt = self.prompter.generate_prompt(instruction, input)
            inputs = self.tokenizer(
                prompt, return_tensors="pt").to(self.device)
            generation_output = self.model(
                **inputs
            )
            # print("generation_output_loss: ", generation_output.loss)
