import json
import transformers
import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score
from transformers import LlamaForCausalLM, LlamaTokenizer
from typing import Dict, List, Tuple, Any, Union, Callable

from torch.utils.data import Dataset, DataLoader
from collections import defaultdict
from tqdm import tqdm
from loguru import logger


class EvalDataset(Dataset):
    def __init__(self, samples: List, samples_input: List) -> None:
        super().__init__()
        self.samples = samples
        self.samples_input = samples_input
    
    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        return self.samples[index], self.samples_input[index]


def load_piqa() -> Dict[str, str]:
    data_path = './metric/piqa/inputs/valid.jsonl'
    label_path = './metric/piqa/label/valid-labels.lst'
    with open(data_path, encoding="utf-8") as f:
        _samples = f.readlines()
    with open(label_path, encoding="utf-8") as f:
        _labels = f.readlines()

    samples = []
    for idx, (sample_str, label_str) in enumerate(zip(_samples, _labels)):
        _sample = json.loads(sample_str)
        sample = {
            "id": idx,
            "goal": _sample["goal"],
            "choices": [_sample["sol1"], _sample["sol2"]],
            "gold": int(label_str)
        }
        samples.append(sample)
    return samples

def load_dataset(name: str) -> Dict[str, str]:
    dataset: List[Dict] = None
    task_dict: Dict[str, Callable] = {
        "piqa": load_piqa
    }
    if name in task_dict:
        dataset = task_dict[name]()
    else:
        raise ValueError
    return dataset


@torch.no_grad()
def evaluate(
    model: LlamaForCausalLM,
    tokenizer: LlamaTokenizer,
    dataset_name: str,
    device: str
):
    dataset = load_dataset(dataset_name)
    tokenizer.padding_side = "left"

    def tokenize(docs: List[Dict[str, Union[str, List[str]]]]) -> List[Dict[str, str]]:
        samples = []
        for doc in tqdm(docs):
            for choice_id, choice in enumerate(doc["choices"]):
                labels = tokenizer(choice).input_ids[1:] + [tokenizer.eos_token_id]
                samples.append({
                    "sample_id": doc["id"],
                    "choice_id": choice_id,
                    "text": doc["goal"] + " " + choice,
                    "labels": labels,
                    "gold": doc["gold"]
                })
        samples_input = []
        for sample in tqdm(samples):
            samples_input.append(tokenizer(sample["text"]))
        return samples, samples_input
    
    samples, samples_input = tokenize(dataset)

    data_collator=transformers.DataCollatorForSeq2Seq(
        tokenizer, pad_to_multiple_of=8, return_tensors="pt", padding=True
    )
    def collate_fn(batch: List):
        batch_samples, batch_inputs = zip(*batch)
        return batch_samples, data_collator(batch_inputs)

    eval_dataset = EvalDataset(samples, samples_input)
    eval_dataloader = DataLoader(eval_dataset, shuffle=False, collate_fn=collate_fn)

    results = []
    pbar = tqdm(eval_dataloader, total=len(eval_dataloader), desc="evaluate")
    for batch_samples, batch_inputs in pbar:
        batch_inputs = batch_inputs.to(device)
        batch_logits = model(**batch_inputs).logits

        batch_results = []
        for i in range(len(batch_samples)):
            logits = batch_logits[i][-len(batch_samples[i]["labels"]):]
            labels = torch.tensor(
                batch_samples[i]["labels"], 
                dtype=torch.long
            ).to(device)
            score = F.cross_entropy(logits, labels)
            batch_results.append(score.item())
        results.extend(batch_results)

    eval_dict = defaultdict(list)
    for sample, result in zip(samples, results):
        eval_dict[sample["sample_id"]].append(
            (result, sample["choice_id"], sample["gold"])
        )
    y_pred = []
    y_true = []
    for value in eval_dict.values():
        value = sorted(value, key=lambda t: t[0])
        y_pred.append(value[0][1])
        y_true.append(value[0][2])
    accuracy = accuracy_score(y_true=y_true, y_pred=y_pred)
    logger.info("accuracy = {}".format(accuracy))
