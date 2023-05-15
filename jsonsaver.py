import json
import sys
import os

#piqa
input_filepath = './metric/piqa/inputs/train.jsonl'
label_filepath = './metric/piqa/label/train-labels.lst'
def batch2dict(input_filepath, label_filepath=None):
    with open(input_filepath, encoding="utf-8") as input_file:
        inputs = input_file.read().splitlines()
    if label_filepath is not None:
        with open(label_filepath, encoding="utf-8") as label_file:
            labels = label_file.read().splitlines()
    else:
        labels = [-1] * len(inputs)
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

    return ans

#  winogrande_1.1
input_filepath = "./metric/winogrande_1.1/dev.jsonl"
def batch2dict(input_filepath, label_filepath=None):
    with open(input_filepath, encoding="utf-8") as input_file:
        inputs = input_file.read().splitlines()
    ans = []
    instruction = "complete the sentence's '_' with option1 or option2, and only output '1' for option1 or output '2' for option2."

    for idx, row in enumerate(inputs):
        data = json.loads(row)
        instructions = instruction
        inputss = ('sentence: ' + data['sentence'] + 
                   '; option1: ' + data["option1"] +
                   '; option2: ' + data["option2"])
        raw_dataset = {"instruction": instructions,
                       "input": inputss,
                       "output": data['answer']}
        ans.append(raw_dataset)


    return ans

# hellaswag-train-dev
input_filepath = "./metric/hellaswag-train-dev/valid.jsonl"
def batch2dict(input_filepath, label_filepath=None):
    with open(input_filepath, encoding="utf-8") as input_file:
        inputs = input_file.read().splitlines()
    ans = []
    instruction = "Choose the most suitable option for continuation based on the beginning of the sentence, and output the number of the option."

    for idx, row in enumerate(inputs):
        data = json.loads(row)
        instructions = instruction
        inputss = ('beginning: ' + data['activity_label'] + 
                   '; option0: ' + data["ending_options"][0] +
                   '; option1: ' + data["ending_options"][1] + 
                   '; option2: ' + data["ending_options"][2] + 
                   '; option3: ' + data["ending_options"][3])
        raw_dataset = {"instruction": instructions,
                       "input": inputss}
        ans.append(raw_dataset)


    return ans