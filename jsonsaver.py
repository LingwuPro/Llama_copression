import json
import sys
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

input_filepath = './metric/piqa/inputs/train.jsonl'
label_filepath = './metric/piqa/label/train-labels.lst'


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
    instruction = "choose the coherent one of the sol, output 0 for sol1,or 1 for sol2."

    for idx, (row, lab) in enumerate(zip(inputs, labels)):
        data = json.loads(row)
        instructions = (instruction)
        inputss = ('sol1: ' + data["goal"] + data["sol1"][:64] +
                   '; sol2: ' + data["goal"] + data["sol2"][:64])
        if lab == '1':
            output = data["goal"] + data["sol2"]
        elif lab == '0':
            output = data["goal"] + data["sol1"]
        raw_dataset = {"instruction": instructions,
                       "input": inputss,
                       "output": output}
        ans.append(raw_dataset)

    # raw_dataset = MyDataset(raw_dataset)

    return ans


raw_data = batch2dict(input_filepath, label_filepath)

with open('./train.jsonl', 'w') as json_file:
    json_file.write(json.dumps(raw_data, ensure_ascii=False, indent=4))
