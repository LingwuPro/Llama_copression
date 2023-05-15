# coding=utf-8
# Copyright 2022 EleutherAI and the HuggingFace Inc. team. All rights reserved.
#
# This code is based on EleutherAI's GPT-NeoX library and the GPT-NeoX
# and OPT implementations in this library. It has been modified from its
# original forms to accommodate minor architectural differences compared
# to GPT-NeoX and OPT used by the Meta AI team that trained the model.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
""" PyTorch LLaMA model."""
import math
import re
from typing import List, Optional, Tuple, Union

import torch
import torch.utils.checkpoint
from torch import nn
import transformers.utils
# from . import doc
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss

from transformers.activations import ACT2FN
from transformers.modeling_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast, SequenceClassifierOutputWithPast
from transformers.modeling_utils import PreTrainedModel
from transformers.utils import logging
from .configuration_llama import LlamaConfig

import functools
import types


def add_start_docstrings(*docstr):
    def docstring_decorator(fn):
        fn.__doc__ = "".join(docstr) + \
            (fn.__doc__ if fn.__doc__ is not None else "")
        return fn

    return docstring_decorator


def add_start_docstrings_to_model_forward(*docstr):
    def docstring_decorator(fn):
        docstring = "".join(docstr) + \
            (fn.__doc__ if fn.__doc__ is not None else "")
        class_name = f"[`{fn.__qualname__.split('.')[0]}`]"
        intro = f"   The {class_name} forward method, overrides the `__call__` special method."
        note = r"""

    <Tip>

    Although the recipe for forward pass needs to be defined within this function, one should call the [`Module`]
    instance afterwards instead of this since the former takes care of running the pre and post processing steps while
    the latter silently ignores them.

    </Tip>
"""

        fn.__doc__ = intro + note + docstring
        return fn

    return docstring_decorator


def add_end_docstrings(*docstr):
    def docstring_decorator(fn):
        fn.__doc__ = (
            fn.__doc__ if fn.__doc__ is not None else "") + "".join(docstr)
        return fn

    return docstring_decorator


PT_RETURN_INTRODUCTION = r"""
    Returns:
        [`{full_output_type}`] or `tuple(torch.FloatTensor)`: A [`{full_output_type}`] or a tuple of
        `torch.FloatTensor` (if `return_dict=False` is passed or when `config.return_dict=False`) comprising various
        elements depending on the configuration ([`{config_class}`]) and inputs.

"""


TF_RETURN_INTRODUCTION = r"""
    Returns:
        [`{full_output_type}`] or `tuple(tf.Tensor)`: A [`{full_output_type}`] or a tuple of `tf.Tensor` (if
        `return_dict=False` is passed or when `config.return_dict=False`) comprising various elements depending on the
        configuration ([`{config_class}`]) and inputs.

"""


def _get_indent(t):
    """Returns the indentation in the first line of t"""
    search = re.search(r"^(\s*)\S", t)
    return "" if search is None else search.groups()[0]


def _convert_output_args_doc(output_args_doc):
    """Convert output_args_doc to display properly."""
    # Split output_arg_doc in blocks argument/description
    indent = _get_indent(output_args_doc)
    blocks = []
    current_block = ""
    for line in output_args_doc.split("\n"):
        # If the indent is the same as the beginning, the line is the name of new arg.
        if _get_indent(line) == indent:
            if len(current_block) > 0:
                blocks.append(current_block[:-1])
            current_block = f"{line}\n"
        else:
            # Otherwise it's part of the description of the current arg.
            # We need to remove 2 spaces to the indentation.
            current_block += f"{line[2:]}\n"
    blocks.append(current_block[:-1])

    # Format each block for proper rendering
    for i in range(len(blocks)):
        blocks[i] = re.sub(r"^(\s+)(\S+)(\s+)", r"\1- **\2**\3", blocks[i])
        blocks[i] = re.sub(r":\s*\n\s*(\S)", r" -- \1", blocks[i])

    return "\n".join(blocks)


def _prepare_output_docstrings(output_type, config_class, min_indent=None):
    """
    Prepares the return part of the docstring using `output_type`.
    """
    output_docstring = output_type.__doc__

    # Remove the head of the docstring to keep the list of args only
    lines = output_docstring.split("\n")
    i = 0
    while i < len(lines) and re.search(r"^\s*(Args|Parameters):\s*$", lines[i]) is None:
        i += 1
    if i < len(lines):
        params_docstring = "\n".join(lines[(i + 1):])
        params_docstring = _convert_output_args_doc(params_docstring)

    # Add the return introduction
    full_output_type = f"{output_type.__module__}.{output_type.__name__}"
    intro = TF_RETURN_INTRODUCTION if output_type.__name__.startswith(
        "TF") else PT_RETURN_INTRODUCTION
    intro = intro.format(full_output_type=full_output_type,
                         config_class=config_class)
    result = intro + params_docstring

    # Apply minimum indent if necessary
    if min_indent is not None:
        lines = result.split("\n")
        # Find the indent of the first nonempty line
        i = 0
        while len(lines[i]) == 0:
            i += 1
        indent = len(_get_indent(lines[i]))
        # If too small, add indentation to all nonempty lines
        if indent < min_indent:
            to_add = " " * (min_indent - indent)
            lines = [(f"{to_add}{line}" if len(line) > 0 else line)
                     for line in lines]
            result = "\n".join(lines)

    return result


FAKE_MODEL_DISCLAIMER = """
    <Tip warning={true}>

    This example uses a random model as the real ones are all very big. To get proper results, you should use
    {real_checkpoint} instead of {fake_checkpoint}. If you get out-of-memory when loading that checkpoint, you can try
    adding `device_map="auto"` in the `from_pretrained` call.

    </Tip>
"""


PT_TOKEN_CLASSIFICATION_SAMPLE = r"""
    Example:

    ```python
    >>> from transformers import AutoTokenizer, {model_class}
    >>> import torch

    >>> tokenizer = AutoTokenizer.from_pretrained("{checkpoint}")
    >>> model = {model_class}.from_pretrained("{checkpoint}")

    >>> inputs = tokenizer(
    ...     "HuggingFace is a company based in Paris and New York", add_special_tokens=False, return_tensors="pt"
    ... )

    >>> with torch.no_grad():
    ...     logits = model(**inputs).logits

    >>> predicted_token_class_ids = logits.argmax(-1)

    >>> # Note that tokens are classified rather then input words which means that
    >>> # there might be more predicted token classes than words.
    >>> # Multiple token classes might account for the same word
    >>> predicted_tokens_classes = [model.config.id2label[t.item()] for t in predicted_token_class_ids[0]]
    >>> predicted_tokens_classes
    {expected_output}

    >>> labels = predicted_token_class_ids
    >>> loss = model(**inputs, labels=labels).loss
    >>> round(loss.item(), 2)
    {expected_loss}
    ```
"""

PT_QUESTION_ANSWERING_SAMPLE = r"""
    Example:

    ```python
    >>> from transformers import AutoTokenizer, {model_class}
    >>> import torch

    >>> tokenizer = AutoTokenizer.from_pretrained("{checkpoint}")
    >>> model = {model_class}.from_pretrained("{checkpoint}")

    >>> question, text = "Who was Jim Henson?", "Jim Henson was a nice puppet"

    >>> inputs = tokenizer(question, text, return_tensors="pt")
    >>> with torch.no_grad():
    ...     outputs = model(**inputs)

    >>> answer_start_index = outputs.start_logits.argmax()
    >>> answer_end_index = outputs.end_logits.argmax()

    >>> predict_answer_tokens = inputs.input_ids[0, answer_start_index : answer_end_index + 1]
    >>> tokenizer.decode(predict_answer_tokens, skip_special_tokens=True)
    {expected_output}

    >>> # target is "nice puppet"
    >>> target_start_index = torch.tensor([{qa_target_start_index}])
    >>> target_end_index = torch.tensor([{qa_target_end_index}])

    >>> outputs = model(**inputs, start_positions=target_start_index, end_positions=target_end_index)
    >>> loss = outputs.loss
    >>> round(loss.item(), 2)
    {expected_loss}
    ```
"""

PT_SEQUENCE_CLASSIFICATION_SAMPLE = r"""
    Example of single-label classification:

    ```python
    >>> import torch
    >>> from transformers import AutoTokenizer, {model_class}

    >>> tokenizer = AutoTokenizer.from_pretrained("{checkpoint}")
    >>> model = {model_class}.from_pretrained("{checkpoint}")

    >>> inputs = tokenizer("Hello, my dog is cute", return_tensors="pt")

    >>> with torch.no_grad():
    ...     logits = model(**inputs).logits

    >>> predicted_class_id = logits.argmax().item()
    >>> model.config.id2label[predicted_class_id]
    {expected_output}

    >>> # To train a model on `num_labels` classes, you can pass `num_labels=num_labels` to `.from_pretrained(...)`
    >>> num_labels = len(model.config.id2label)
    >>> model = {model_class}.from_pretrained("{checkpoint}", num_labels=num_labels)

    >>> labels = torch.tensor([1])
    >>> loss = model(**inputs, labels=labels).loss
    >>> round(loss.item(), 2)
    {expected_loss}
    ```

    Example of multi-label classification:

    ```python
    >>> import torch
    >>> from transformers import AutoTokenizer, {model_class}

    >>> tokenizer = AutoTokenizer.from_pretrained("{checkpoint}")
    >>> model = {model_class}.from_pretrained("{checkpoint}", problem_type="multi_label_classification")

    >>> inputs = tokenizer("Hello, my dog is cute", return_tensors="pt")

    >>> with torch.no_grad():
    ...     logits = model(**inputs).logits

    >>> predicted_class_ids = torch.arange(0, logits.shape[-1])[torch.sigmoid(logits).squeeze(dim=0) > 0.5]

    >>> # To train a model on `num_labels` classes, you can pass `num_labels=num_labels` to `.from_pretrained(...)`
    >>> num_labels = len(model.config.id2label)
    >>> model = {model_class}.from_pretrained(
    ...     "{checkpoint}", num_labels=num_labels, problem_type="multi_label_classification"
    ... )

    >>> labels = torch.sum(
    ...     torch.nn.functional.one_hot(predicted_class_ids[None, :].clone(), num_classes=num_labels), dim=1
    ... ).to(torch.float)
    >>> loss = model(**inputs, labels=labels).loss
    ```
"""

PT_MASKED_LM_SAMPLE = r"""
    Example:

    ```python
    >>> from transformers import AutoTokenizer, {model_class}
    >>> import torch

    >>> tokenizer = AutoTokenizer.from_pretrained("{checkpoint}")
    >>> model = {model_class}.from_pretrained("{checkpoint}")

    >>> inputs = tokenizer("The capital of France is {mask}.", return_tensors="pt")

    >>> with torch.no_grad():
    ...     logits = model(**inputs).logits

    >>> # retrieve index of {mask}
    >>> mask_token_index = (inputs.input_ids == tokenizer.mask_token_id)[0].nonzero(as_tuple=True)[0]

    >>> predicted_token_id = logits[0, mask_token_index].argmax(axis=-1)
    >>> tokenizer.decode(predicted_token_id)
    {expected_output}

    >>> labels = tokenizer("The capital of France is Paris.", return_tensors="pt")["input_ids"]
    >>> # mask labels of non-{mask} tokens
    >>> labels = torch.where(inputs.input_ids == tokenizer.mask_token_id, labels, -100)

    >>> outputs = model(**inputs, labels=labels)
    >>> round(outputs.loss.item(), 2)
    {expected_loss}
    ```
"""

PT_BASE_MODEL_SAMPLE = r"""
    Example:

    ```python
    >>> from transformers import AutoTokenizer, {model_class}
    >>> import torch

    >>> tokenizer = AutoTokenizer.from_pretrained("{checkpoint}")
    >>> model = {model_class}.from_pretrained("{checkpoint}")

    >>> inputs = tokenizer("Hello, my dog is cute", return_tensors="pt")
    >>> outputs = model(**inputs)

    >>> last_hidden_states = outputs.last_hidden_state
    ```
"""

PT_MULTIPLE_CHOICE_SAMPLE = r"""
    Example:

    ```python
    >>> from transformers import AutoTokenizer, {model_class}
    >>> import torch

    >>> tokenizer = AutoTokenizer.from_pretrained("{checkpoint}")
    >>> model = {model_class}.from_pretrained("{checkpoint}")

    >>> prompt = "In Italy, pizza served in formal settings, such as at a restaurant, is presented unsliced."
    >>> choice0 = "It is eaten with a fork and a knife."
    >>> choice1 = "It is eaten while held in the hand."
    >>> labels = torch.tensor(0).unsqueeze(0)  # choice0 is correct (according to Wikipedia ;)), batch size 1

    >>> encoding = tokenizer([prompt, prompt], [choice0, choice1], return_tensors="pt", padding=True)
    >>> outputs = model(**{{k: v.unsqueeze(0) for k, v in encoding.items()}}, labels=labels)  # batch size is 1

    >>> # the linear classifier still needs to be trained
    >>> loss = outputs.loss
    >>> logits = outputs.logits
    ```
"""

PT_CAUSAL_LM_SAMPLE = r"""
    Example:

    ```python
    >>> import torch
    >>> from transformers import AutoTokenizer, {model_class}

    >>> tokenizer = AutoTokenizer.from_pretrained("{checkpoint}")
    >>> model = {model_class}.from_pretrained("{checkpoint}")

    >>> inputs = tokenizer("Hello, my dog is cute", return_tensors="pt")
    >>> outputs = model(**inputs, labels=inputs["input_ids"])
    >>> loss = outputs.loss
    >>> logits = outputs.logits
    ```
"""

PT_SPEECH_BASE_MODEL_SAMPLE = r"""
    Example:

    ```python
    >>> from transformers import AutoProcessor, {model_class}
    >>> import torch
    >>> from datasets import load_dataset

    >>> dataset = load_dataset("hf-internal-testing/librispeech_asr_demo", "clean", split="validation")
    >>> dataset = dataset.sort("id")
    >>> sampling_rate = dataset.features["audio"].sampling_rate

    >>> processor = AutoProcessor.from_pretrained("{checkpoint}")
    >>> model = {model_class}.from_pretrained("{checkpoint}")

    >>> # audio file is decoded on the fly
    >>> inputs = processor(dataset[0]["audio"]["array"], sampling_rate=sampling_rate, return_tensors="pt")
    >>> with torch.no_grad():
    ...     outputs = model(**inputs)

    >>> last_hidden_states = outputs.last_hidden_state
    >>> list(last_hidden_states.shape)
    {expected_output}
    ```
"""

PT_SPEECH_CTC_SAMPLE = r"""
    Example:

    ```python
    >>> from transformers import AutoProcessor, {model_class}
    >>> from datasets import load_dataset
    >>> import torch

    >>> dataset = load_dataset("hf-internal-testing/librispeech_asr_demo", "clean", split="validation")
    >>> dataset = dataset.sort("id")
    >>> sampling_rate = dataset.features["audio"].sampling_rate

    >>> processor = AutoProcessor.from_pretrained("{checkpoint}")
    >>> model = {model_class}.from_pretrained("{checkpoint}")

    >>> # audio file is decoded on the fly
    >>> inputs = processor(dataset[0]["audio"]["array"], sampling_rate=sampling_rate, return_tensors="pt")
    >>> with torch.no_grad():
    ...     logits = model(**inputs).logits
    >>> predicted_ids = torch.argmax(logits, dim=-1)

    >>> # transcribe speech
    >>> transcription = processor.batch_decode(predicted_ids)
    >>> transcription[0]
    {expected_output}

    >>> inputs["labels"] = processor(text=dataset[0]["text"], return_tensors="pt").input_ids

    >>> # compute loss
    >>> loss = model(**inputs).loss
    >>> round(loss.item(), 2)
    {expected_loss}
    ```
"""

PT_SPEECH_SEQ_CLASS_SAMPLE = r"""
    Example:

    ```python
    >>> from transformers import AutoFeatureExtractor, {model_class}
    >>> from datasets import load_dataset
    >>> import torch

    >>> dataset = load_dataset("hf-internal-testing/librispeech_asr_demo", "clean", split="validation")
    >>> dataset = dataset.sort("id")
    >>> sampling_rate = dataset.features["audio"].sampling_rate

    >>> feature_extractor = AutoFeatureExtractor.from_pretrained("{checkpoint}")
    >>> model = {model_class}.from_pretrained("{checkpoint}")

    >>> # audio file is decoded on the fly
    >>> inputs = feature_extractor(dataset[0]["audio"]["array"], sampling_rate=sampling_rate, return_tensors="pt")

    >>> with torch.no_grad():
    ...     logits = model(**inputs).logits

    >>> predicted_class_ids = torch.argmax(logits, dim=-1).item()
    >>> predicted_label = model.config.id2label[predicted_class_ids]
    >>> predicted_label
    {expected_output}

    >>> # compute loss - target_label is e.g. "down"
    >>> target_label = model.config.id2label[0]
    >>> inputs["labels"] = torch.tensor([model.config.label2id[target_label]])
    >>> loss = model(**inputs).loss
    >>> round(loss.item(), 2)
    {expected_loss}
    ```
"""


PT_SPEECH_FRAME_CLASS_SAMPLE = r"""
    Example:

    ```python
    >>> from transformers import AutoFeatureExtractor, {model_class}
    >>> from datasets import load_dataset
    >>> import torch

    >>> dataset = load_dataset("hf-internal-testing/librispeech_asr_demo", "clean", split="validation")
    >>> dataset = dataset.sort("id")
    >>> sampling_rate = dataset.features["audio"].sampling_rate

    >>> feature_extractor = AutoFeatureExtractor.from_pretrained("{checkpoint}")
    >>> model = {model_class}.from_pretrained("{checkpoint}")

    >>> # audio file is decoded on the fly
    >>> inputs = feature_extractor(dataset[0]["audio"]["array"], return_tensors="pt", sampling_rate=sampling_rate)
    >>> with torch.no_grad():
    ...     logits = model(**inputs).logits

    >>> probabilities = torch.sigmoid(logits[0])
    >>> # labels is a one-hot array of shape (num_frames, num_speakers)
    >>> labels = (probabilities > 0.5).long()
    >>> labels[0].tolist()
    {expected_output}
    ```
"""


PT_SPEECH_XVECTOR_SAMPLE = r"""
    Example:

    ```python
    >>> from transformers import AutoFeatureExtractor, {model_class}
    >>> from datasets import load_dataset
    >>> import torch

    >>> dataset = load_dataset("hf-internal-testing/librispeech_asr_demo", "clean", split="validation")
    >>> dataset = dataset.sort("id")
    >>> sampling_rate = dataset.features["audio"].sampling_rate

    >>> feature_extractor = AutoFeatureExtractor.from_pretrained("{checkpoint}")
    >>> model = {model_class}.from_pretrained("{checkpoint}")

    >>> # audio file is decoded on the fly
    >>> inputs = feature_extractor(
    ...     [d["array"] for d in dataset[:2]["audio"]], sampling_rate=sampling_rate, return_tensors="pt", padding=True
    ... )
    >>> with torch.no_grad():
    ...     embeddings = model(**inputs).embeddings

    >>> embeddings = torch.nn.functional.normalize(embeddings, dim=-1).cpu()

    >>> # the resulting embeddings can be used for cosine similarity-based retrieval
    >>> cosine_sim = torch.nn.CosineSimilarity(dim=-1)
    >>> similarity = cosine_sim(embeddings[0], embeddings[1])
    >>> threshold = 0.7  # the optimal threshold is dataset-dependent
    >>> if similarity < threshold:
    ...     print("Speakers are not the same!")
    >>> round(similarity.item(), 2)
    {expected_output}
    ```
"""

PT_VISION_BASE_MODEL_SAMPLE = r"""
    Example:

    ```python
    >>> from transformers import AutoImageProcessor, {model_class}
    >>> import torch
    >>> from datasets import load_dataset

    >>> dataset = load_dataset("huggingface/cats-image")
    >>> image = dataset["test"]["image"][0]

    >>> image_processor = AutoImageProcessor.from_pretrained("{checkpoint}")
    >>> model = {model_class}.from_pretrained("{checkpoint}")

    >>> inputs = image_processor(image, return_tensors="pt")

    >>> with torch.no_grad():
    ...     outputs = model(**inputs)

    >>> last_hidden_states = outputs.last_hidden_state
    >>> list(last_hidden_states.shape)
    {expected_output}
    ```
"""

PT_VISION_SEQ_CLASS_SAMPLE = r"""
    Example:

    ```python
    >>> from transformers import AutoImageProcessor, {model_class}
    >>> import torch
    >>> from datasets import load_dataset

    >>> dataset = load_dataset("huggingface/cats-image")
    >>> image = dataset["test"]["image"][0]

    >>> image_processor = AutoImageProcessor.from_pretrained("{checkpoint}")
    >>> model = {model_class}.from_pretrained("{checkpoint}")

    >>> inputs = image_processor(image, return_tensors="pt")

    >>> with torch.no_grad():
    ...     logits = model(**inputs).logits

    >>> # model predicts one of the 1000 ImageNet classes
    >>> predicted_label = logits.argmax(-1).item()
    >>> print(model.config.id2label[predicted_label])
    {expected_output}
    ```
"""


PT_SAMPLE_DOCSTRINGS = {
    "SequenceClassification": PT_SEQUENCE_CLASSIFICATION_SAMPLE,
    "QuestionAnswering": PT_QUESTION_ANSWERING_SAMPLE,
    "TokenClassification": PT_TOKEN_CLASSIFICATION_SAMPLE,
    "MultipleChoice": PT_MULTIPLE_CHOICE_SAMPLE,
    "MaskedLM": PT_MASKED_LM_SAMPLE,
    "LMHead": PT_CAUSAL_LM_SAMPLE,
    "BaseModel": PT_BASE_MODEL_SAMPLE,
    "SpeechBaseModel": PT_SPEECH_BASE_MODEL_SAMPLE,
    "CTC": PT_SPEECH_CTC_SAMPLE,
    "AudioClassification": PT_SPEECH_SEQ_CLASS_SAMPLE,
    "AudioFrameClassification": PT_SPEECH_FRAME_CLASS_SAMPLE,
    "AudioXVector": PT_SPEECH_XVECTOR_SAMPLE,
    "VisionBaseModel": PT_VISION_BASE_MODEL_SAMPLE,
    "ImageClassification": PT_VISION_SEQ_CLASS_SAMPLE,
}


TF_TOKEN_CLASSIFICATION_SAMPLE = r"""
    Example:

    ```python
    >>> from transformers import AutoTokenizer, {model_class}
    >>> import tensorflow as tf

    >>> tokenizer = AutoTokenizer.from_pretrained("{checkpoint}")
    >>> model = {model_class}.from_pretrained("{checkpoint}")

    >>> inputs = tokenizer(
    ...     "HuggingFace is a company based in Paris and New York", add_special_tokens=False, return_tensors="tf"
    ... )

    >>> logits = model(**inputs).logits
    >>> predicted_token_class_ids = tf.math.argmax(logits, axis=-1)

    >>> # Note that tokens are classified rather then input words which means that
    >>> # there might be more predicted token classes than words.
    >>> # Multiple token classes might account for the same word
    >>> predicted_tokens_classes = [model.config.id2label[t] for t in predicted_token_class_ids[0].numpy().tolist()]
    >>> predicted_tokens_classes
    {expected_output}
    ```

    ```python
    >>> labels = predicted_token_class_ids
    >>> loss = tf.math.reduce_mean(model(**inputs, labels=labels).loss)
    >>> round(float(loss), 2)
    {expected_loss}
    ```
"""

TF_QUESTION_ANSWERING_SAMPLE = r"""
    Example:

    ```python
    >>> from transformers import AutoTokenizer, {model_class}
    >>> import tensorflow as tf

    >>> tokenizer = AutoTokenizer.from_pretrained("{checkpoint}")
    >>> model = {model_class}.from_pretrained("{checkpoint}")

    >>> question, text = "Who was Jim Henson?", "Jim Henson was a nice puppet"

    >>> inputs = tokenizer(question, text, return_tensors="tf")
    >>> outputs = model(**inputs)

    >>> answer_start_index = int(tf.math.argmax(outputs.start_logits, axis=-1)[0])
    >>> answer_end_index = int(tf.math.argmax(outputs.end_logits, axis=-1)[0])

    >>> predict_answer_tokens = inputs.input_ids[0, answer_start_index : answer_end_index + 1]
    >>> tokenizer.decode(predict_answer_tokens)
    {expected_output}
    ```

    ```python
    >>> # target is "nice puppet"
    >>> target_start_index = tf.constant([{qa_target_start_index}])
    >>> target_end_index = tf.constant([{qa_target_end_index}])

    >>> outputs = model(**inputs, start_positions=target_start_index, end_positions=target_end_index)
    >>> loss = tf.math.reduce_mean(outputs.loss)
    >>> round(float(loss), 2)
    {expected_loss}
    ```
"""

TF_SEQUENCE_CLASSIFICATION_SAMPLE = r"""
    Example:

    ```python
    >>> from transformers import AutoTokenizer, {model_class}
    >>> import tensorflow as tf

    >>> tokenizer = AutoTokenizer.from_pretrained("{checkpoint}")
    >>> model = {model_class}.from_pretrained("{checkpoint}")

    >>> inputs = tokenizer("Hello, my dog is cute", return_tensors="tf")

    >>> logits = model(**inputs).logits

    >>> predicted_class_id = int(tf.math.argmax(logits, axis=-1)[0])
    >>> model.config.id2label[predicted_class_id]
    {expected_output}
    ```

    ```python
    >>> # To train a model on `num_labels` classes, you can pass `num_labels=num_labels` to `.from_pretrained(...)`
    >>> num_labels = len(model.config.id2label)
    >>> model = {model_class}.from_pretrained("{checkpoint}", num_labels=num_labels)

    >>> labels = tf.constant(1)
    >>> loss = model(**inputs, labels=labels).loss
    >>> round(float(loss), 2)
    {expected_loss}
    ```
"""

TF_MASKED_LM_SAMPLE = r"""
    Example:

    ```python
    >>> from transformers import AutoTokenizer, {model_class}
    >>> import tensorflow as tf

    >>> tokenizer = AutoTokenizer.from_pretrained("{checkpoint}")
    >>> model = {model_class}.from_pretrained("{checkpoint}")

    >>> inputs = tokenizer("The capital of France is {mask}.", return_tensors="tf")
    >>> logits = model(**inputs).logits

    >>> # retrieve index of {mask}
    >>> mask_token_index = tf.where((inputs.input_ids == tokenizer.mask_token_id)[0])
    >>> selected_logits = tf.gather_nd(logits[0], indices=mask_token_index)

    >>> predicted_token_id = tf.math.argmax(selected_logits, axis=-1)
    >>> tokenizer.decode(predicted_token_id)
    {expected_output}
    ```

    ```python
    >>> labels = tokenizer("The capital of France is Paris.", return_tensors="tf")["input_ids"]
    >>> # mask labels of non-{mask} tokens
    >>> labels = tf.where(inputs.input_ids == tokenizer.mask_token_id, labels, -100)

    >>> outputs = model(**inputs, labels=labels)
    >>> round(float(outputs.loss), 2)
    {expected_loss}
    ```
"""

TF_BASE_MODEL_SAMPLE = r"""
    Example:

    ```python
    >>> from transformers import AutoTokenizer, {model_class}
    >>> import tensorflow as tf

    >>> tokenizer = AutoTokenizer.from_pretrained("{checkpoint}")
    >>> model = {model_class}.from_pretrained("{checkpoint}")

    >>> inputs = tokenizer("Hello, my dog is cute", return_tensors="tf")
    >>> outputs = model(inputs)

    >>> last_hidden_states = outputs.last_hidden_state
    ```
"""

TF_MULTIPLE_CHOICE_SAMPLE = r"""
    Example:

    ```python
    >>> from transformers import AutoTokenizer, {model_class}
    >>> import tensorflow as tf

    >>> tokenizer = AutoTokenizer.from_pretrained("{checkpoint}")
    >>> model = {model_class}.from_pretrained("{checkpoint}")

    >>> prompt = "In Italy, pizza served in formal settings, such as at a restaurant, is presented unsliced."
    >>> choice0 = "It is eaten with a fork and a knife."
    >>> choice1 = "It is eaten while held in the hand."

    >>> encoding = tokenizer([prompt, prompt], [choice0, choice1], return_tensors="tf", padding=True)
    >>> inputs = {{k: tf.expand_dims(v, 0) for k, v in encoding.items()}}
    >>> outputs = model(inputs)  # batch size is 1

    >>> # the linear classifier still needs to be trained
    >>> logits = outputs.logits
    ```
"""

TF_CAUSAL_LM_SAMPLE = r"""
    Example:

    ```python
    >>> from transformers import AutoTokenizer, {model_class}
    >>> import tensorflow as tf

    >>> tokenizer = AutoTokenizer.from_pretrained("{checkpoint}")
    >>> model = {model_class}.from_pretrained("{checkpoint}")

    >>> inputs = tokenizer("Hello, my dog is cute", return_tensors="tf")
    >>> outputs = model(inputs)
    >>> logits = outputs.logits
    ```
"""

TF_SPEECH_BASE_MODEL_SAMPLE = r"""
    Example:

    ```python
    >>> from transformers import AutoProcessor, {model_class}
    >>> from datasets import load_dataset

    >>> dataset = load_dataset("hf-internal-testing/librispeech_asr_demo", "clean", split="validation")
    >>> dataset = dataset.sort("id")
    >>> sampling_rate = dataset.features["audio"].sampling_rate

    >>> processor = AutoProcessor.from_pretrained("{checkpoint}")
    >>> model = {model_class}.from_pretrained("{checkpoint}")

    >>> # audio file is decoded on the fly
    >>> inputs = processor(dataset[0]["audio"]["array"], sampling_rate=sampling_rate, return_tensors="tf")
    >>> outputs = model(**inputs)

    >>> last_hidden_states = outputs.last_hidden_state
    >>> list(last_hidden_states.shape)
    {expected_output}
    ```
"""

TF_SPEECH_CTC_SAMPLE = r"""
    Example:

    ```python
    >>> from transformers import AutoProcessor, {model_class}
    >>> from datasets import load_dataset
    >>> import tensorflow as tf

    >>> dataset = load_dataset("hf-internal-testing/librispeech_asr_demo", "clean", split="validation")
    >>> dataset = dataset.sort("id")
    >>> sampling_rate = dataset.features["audio"].sampling_rate

    >>> processor = AutoProcessor.from_pretrained("{checkpoint}")
    >>> model = {model_class}.from_pretrained("{checkpoint}")

    >>> # audio file is decoded on the fly
    >>> inputs = processor(dataset[0]["audio"]["array"], sampling_rate=sampling_rate, return_tensors="tf")
    >>> logits = model(**inputs).logits
    >>> predicted_ids = tf.math.argmax(logits, axis=-1)

    >>> # transcribe speech
    >>> transcription = processor.batch_decode(predicted_ids)
    >>> transcription[0]
    {expected_output}
    ```

    ```python
    >>> inputs["labels"] = processor(text=dataset[0]["text"], return_tensors="tf").input_ids

    >>> # compute loss
    >>> loss = model(**inputs).loss
    >>> round(float(loss), 2)
    {expected_loss}
    ```
"""

TF_VISION_BASE_MODEL_SAMPLE = r"""
    Example:

    ```python
    >>> from transformers import AutoImageProcessor, {model_class}
    >>> from datasets import load_dataset

    >>> dataset = load_dataset("huggingface/cats-image")
    >>> image = dataset["test"]["image"][0]

    >>> image_processor = AutoImageProcessor.from_pretrained("{checkpoint}")
    >>> model = {model_class}.from_pretrained("{checkpoint}")

    >>> inputs = image_processor(image, return_tensors="tf")
    >>> outputs = model(**inputs)

    >>> last_hidden_states = outputs.last_hidden_state
    >>> list(last_hidden_states.shape)
    {expected_output}
    ```
"""

TF_VISION_SEQ_CLASS_SAMPLE = r"""
    Example:

    ```python
    >>> from transformers import AutoImageProcessor, {model_class}
    >>> import tensorflow as tf
    >>> from datasets import load_dataset

    >>> dataset = load_dataset("huggingface/cats-image")
    >>> image = dataset["test"]["image"][0]

    >>> image_processor = AutoImageProcessor.from_pretrained("{checkpoint}")
    >>> model = {model_class}.from_pretrained("{checkpoint}")

    >>> inputs = image_processor(image, return_tensors="tf")
    >>> logits = model(**inputs).logits

    >>> # model predicts one of the 1000 ImageNet classes
    >>> predicted_label = int(tf.math.argmax(logits, axis=-1))
    >>> print(model.config.id2label[predicted_label])
    {expected_output}
    ```
"""

TF_SAMPLE_DOCSTRINGS = {
    "SequenceClassification": TF_SEQUENCE_CLASSIFICATION_SAMPLE,
    "QuestionAnswering": TF_QUESTION_ANSWERING_SAMPLE,
    "TokenClassification": TF_TOKEN_CLASSIFICATION_SAMPLE,
    "MultipleChoice": TF_MULTIPLE_CHOICE_SAMPLE,
    "MaskedLM": TF_MASKED_LM_SAMPLE,
    "LMHead": TF_CAUSAL_LM_SAMPLE,
    "BaseModel": TF_BASE_MODEL_SAMPLE,
    "SpeechBaseModel": TF_SPEECH_BASE_MODEL_SAMPLE,
    "CTC": TF_SPEECH_CTC_SAMPLE,
    "VisionBaseModel": TF_VISION_BASE_MODEL_SAMPLE,
    "ImageClassification": TF_VISION_SEQ_CLASS_SAMPLE,
}


FLAX_TOKEN_CLASSIFICATION_SAMPLE = r"""
    Example:

    ```python
    >>> from transformers import AutoTokenizer, {model_class}

    >>> tokenizer = AutoTokenizer.from_pretrained("{checkpoint}")
    >>> model = {model_class}.from_pretrained("{checkpoint}")

    >>> inputs = tokenizer("Hello, my dog is cute", return_tensors="jax")

    >>> outputs = model(**inputs)
    >>> logits = outputs.logits
    ```
"""

FLAX_QUESTION_ANSWERING_SAMPLE = r"""
    Example:

    ```python
    >>> from transformers import AutoTokenizer, {model_class}

    >>> tokenizer = AutoTokenizer.from_pretrained("{checkpoint}")
    >>> model = {model_class}.from_pretrained("{checkpoint}")

    >>> question, text = "Who was Jim Henson?", "Jim Henson was a nice puppet"
    >>> inputs = tokenizer(question, text, return_tensors="jax")

    >>> outputs = model(**inputs)
    >>> start_scores = outputs.start_logits
    >>> end_scores = outputs.end_logits
    ```
"""

FLAX_SEQUENCE_CLASSIFICATION_SAMPLE = r"""
    Example:

    ```python
    >>> from transformers import AutoTokenizer, {model_class}

    >>> tokenizer = AutoTokenizer.from_pretrained("{checkpoint}")
    >>> model = {model_class}.from_pretrained("{checkpoint}")

    >>> inputs = tokenizer("Hello, my dog is cute", return_tensors="jax")

    >>> outputs = model(**inputs)
    >>> logits = outputs.logits
    ```
"""

FLAX_MASKED_LM_SAMPLE = r"""
    Example:

    ```python
    >>> from transformers import AutoTokenizer, {model_class}

    >>> tokenizer = AutoTokenizer.from_pretrained("{checkpoint}")
    >>> model = {model_class}.from_pretrained("{checkpoint}")

    >>> inputs = tokenizer("The capital of France is {mask}.", return_tensors="jax")

    >>> outputs = model(**inputs)
    >>> logits = outputs.logits
    ```
"""

FLAX_BASE_MODEL_SAMPLE = r"""
    Example:

    ```python
    >>> from transformers import AutoTokenizer, {model_class}

    >>> tokenizer = AutoTokenizer.from_pretrained("{checkpoint}")
    >>> model = {model_class}.from_pretrained("{checkpoint}")

    >>> inputs = tokenizer("Hello, my dog is cute", return_tensors="jax")
    >>> outputs = model(**inputs)

    >>> last_hidden_states = outputs.last_hidden_state
    ```
"""

FLAX_MULTIPLE_CHOICE_SAMPLE = r"""
    Example:

    ```python
    >>> from transformers import AutoTokenizer, {model_class}

    >>> tokenizer = AutoTokenizer.from_pretrained("{checkpoint}")
    >>> model = {model_class}.from_pretrained("{checkpoint}")

    >>> prompt = "In Italy, pizza served in formal settings, such as at a restaurant, is presented unsliced."
    >>> choice0 = "It is eaten with a fork and a knife."
    >>> choice1 = "It is eaten while held in the hand."

    >>> encoding = tokenizer([prompt, prompt], [choice0, choice1], return_tensors="jax", padding=True)
    >>> outputs = model(**{{k: v[None, :] for k, v in encoding.items()}})

    >>> logits = outputs.logits
    ```
"""

FLAX_CAUSAL_LM_SAMPLE = r"""
    Example:

    ```python
    >>> from transformers import AutoTokenizer, {model_class}

    >>> tokenizer = AutoTokenizer.from_pretrained("{checkpoint}")
    >>> model = {model_class}.from_pretrained("{checkpoint}")

    >>> inputs = tokenizer("Hello, my dog is cute", return_tensors="np")
    >>> outputs = model(**inputs)

    >>> # retrieve logts for next token
    >>> next_token_logits = outputs.logits[:, -1]
    ```
"""

FLAX_SAMPLE_DOCSTRINGS = {
    "SequenceClassification": FLAX_SEQUENCE_CLASSIFICATION_SAMPLE,
    "QuestionAnswering": FLAX_QUESTION_ANSWERING_SAMPLE,
    "TokenClassification": FLAX_TOKEN_CLASSIFICATION_SAMPLE,
    "MultipleChoice": FLAX_MULTIPLE_CHOICE_SAMPLE,
    "MaskedLM": FLAX_MASKED_LM_SAMPLE,
    "BaseModel": FLAX_BASE_MODEL_SAMPLE,
    "LMHead": FLAX_CAUSAL_LM_SAMPLE,
}


def filter_outputs_from_example(docstring, **kwargs):
    """
    Removes the lines testing an output with the doctest syntax in a code sample when it's set to `None`.
    """
    for key, value in kwargs.items():
        if value is not None:
            continue

        doc_key = "{" + key + "}"
        docstring = re.sub(rf"\n([^\n]+)\n\s+{doc_key}\n", "\n", docstring)

    return docstring


def add_code_sample_docstrings(
    *docstr,
    processor_class=None,
    checkpoint=None,
    output_type=None,
    config_class=None,
    mask="[MASK]",
    qa_target_start_index=14,
    qa_target_end_index=15,
    model_cls=None,
    modality=None,
    expected_output=None,
    expected_loss=None,
    real_checkpoint=None,
):
    def docstring_decorator(fn):
        # model_class defaults to function's class if not specified otherwise
        model_class = fn.__qualname__.split(
            ".")[0] if model_cls is None else model_cls

        if model_class[:2] == "TF":
            sample_docstrings = TF_SAMPLE_DOCSTRINGS
        elif model_class[:4] == "Flax":
            sample_docstrings = FLAX_SAMPLE_DOCSTRINGS
        else:
            sample_docstrings = PT_SAMPLE_DOCSTRINGS

        # putting all kwargs for docstrings in a dict to be used
        # with the `.format(**doc_kwargs)`. Note that string might
        # be formatted with non-existing keys, which is fine.
        doc_kwargs = {
            "model_class": model_class,
            "processor_class": processor_class,
            "checkpoint": checkpoint,
            "mask": mask,
            "qa_target_start_index": qa_target_start_index,
            "qa_target_end_index": qa_target_end_index,
            "expected_output": expected_output,
            "expected_loss": expected_loss,
            "real_checkpoint": real_checkpoint,
            "fake_checkpoint": checkpoint,
            # For <Tip warning={true}> syntax that conflicts with formatting.
            "true": "{true}",
        }

        if ("SequenceClassification" in model_class or "AudioClassification" in model_class) and modality == "audio":
            code_sample = sample_docstrings["AudioClassification"]
        elif "SequenceClassification" in model_class:
            code_sample = sample_docstrings["SequenceClassification"]
        elif "QuestionAnswering" in model_class:
            code_sample = sample_docstrings["QuestionAnswering"]
        elif "TokenClassification" in model_class:
            code_sample = sample_docstrings["TokenClassification"]
        elif "MultipleChoice" in model_class:
            code_sample = sample_docstrings["MultipleChoice"]
        elif "MaskedLM" in model_class or model_class in ["FlaubertWithLMHeadModel", "XLMWithLMHeadModel"]:
            code_sample = sample_docstrings["MaskedLM"]
        elif "LMHead" in model_class or "CausalLM" in model_class:
            code_sample = sample_docstrings["LMHead"]
        elif "CTC" in model_class:
            code_sample = sample_docstrings["CTC"]
        elif "AudioFrameClassification" in model_class:
            code_sample = sample_docstrings["AudioFrameClassification"]
        elif "XVector" in model_class and modality == "audio":
            code_sample = sample_docstrings["AudioXVector"]
        elif "Model" in model_class and modality == "audio":
            code_sample = sample_docstrings["SpeechBaseModel"]
        elif "Model" in model_class and modality == "vision":
            code_sample = sample_docstrings["VisionBaseModel"]
        elif "Model" in model_class or "Encoder" in model_class:
            code_sample = sample_docstrings["BaseModel"]
        elif "ImageClassification" in model_class:
            code_sample = sample_docstrings["ImageClassification"]
        else:
            raise ValueError(
                f"Docstring can't be built for model {model_class}")

        code_sample = filter_outputs_from_example(
            code_sample, expected_output=expected_output, expected_loss=expected_loss
        )
        if real_checkpoint is not None:
            code_sample = FAKE_MODEL_DISCLAIMER + code_sample
        func_doc = (fn.__doc__ or "") + "".join(docstr)
        output_doc = "" if output_type is None else _prepare_output_docstrings(
            output_type, config_class)
        built_doc = code_sample.format(**doc_kwargs)
        fn.__doc__ = func_doc + output_doc + built_doc
        return fn

    return docstring_decorator


def replace_return_docstrings(output_type=None, config_class=None):
    def docstring_decorator(fn):
        func_doc = fn.__doc__
        lines = func_doc.split("\n")
        i = 0
        while i < len(lines) and re.search(r"^\s*Returns?:\s*$", lines[i]) is None:
            i += 1
        if i < len(lines):
            indent = len(_get_indent(lines[i]))
            lines[i] = _prepare_output_docstrings(
                output_type, config_class, min_indent=indent)
            func_doc = "\n".join(lines)
        else:
            raise ValueError(
                f"The function {fn} should have an empty 'Return:' or 'Returns:' in its docstring as placeholder, "
                f"current docstring is:\n{func_doc}"
            )
        fn.__doc__ = func_doc
        return fn

    return docstring_decorator


def copy_func(f):
    """Returns a copy of a function f."""
    # Based on http://stackoverflow.com/a/6528148/190597 (Glenn Maynard)
    g = types.FunctionType(f.__code__, f.__globals__, name=f.__name__,
                           argdefs=f.__defaults__, closure=f.__closure__)
    g = functools.update_wrapper(g, f)
    g.__kwdefaults__ = f.__kwdefaults__
    return g


def add_start_docstrings(*docstr):
    def docstring_decorator(fn):
        fn.__doc__ = "".join(docstr) + \
            (fn.__doc__ if fn.__doc__ is not None else "")
        return fn

    return docstring_decorator


def add_start_docstrings_to_model_forward(*docstr):
    def docstring_decorator(fn):
        docstring = "".join(docstr) + \
            (fn.__doc__ if fn.__doc__ is not None else "")
        class_name = f"[`{fn.__qualname__.split('.')[0]}`]"
        intro = f"   The {class_name} forward method, overrides the `__call__` special method."
        note = r"""
        
        <Tip>

        Although the recipe for forward pass needs to be defined within this function, one should call the [`Module`]
        instance afterwards instead of this since the former takes care of running the pre and post processing steps while
        the latter silently ignores them.

        </Tip>
        """

        fn.__doc__ = intro + note + docstring
        return fn

    return docstring_decorator


def replace_return_docstrings(output_type=None, config_class=None):
    def docstring_decorator(fn):
        func_doc = fn.__doc__
        lines = func_doc.split("\n")
        i = 0
        while i < len(lines) and re.search(r"^\s*Returns?:\s*$", lines[i]) is None:
            i += 1
        if i < len(lines):
            indent = len(_get_indent(lines[i]))
            lines[i] = _prepare_output_docstrings(
                output_type, config_class, min_indent=indent)
            func_doc = "\n".join(lines)
        else:
            raise ValueError(
                f"The function {fn} should have an empty 'Return:' or 'Returns:' in its docstring as placeholder, "
                f"current docstring is:\n{func_doc}"
            )
        fn.__doc__ = func_doc
        return fn

    return docstring_decorator


logger = logging.get_logger(__name__)

_CONFIG_FOR_DOC = "LlamaConfig"


# Copied from transformers.models.bart.modeling_bart._make_causal_mask
def _make_causal_mask(
    input_ids_shape: torch.Size, dtype: torch.dtype, device: torch.device, past_key_values_length: int = 0
):
    """
    Make causal mask used for bi-directional self-attention.
    """
    bsz, tgt_len = input_ids_shape
    mask = torch.full((tgt_len, tgt_len), torch.tensor(
        torch.finfo(dtype).min, device=device), device=device)
    mask_cond = torch.arange(mask.size(-1), device=device)
    mask.masked_fill_(mask_cond < (mask_cond + 1).view(mask.size(-1), 1), 0)
    mask = mask.to(dtype)

    if past_key_values_length > 0:
        mask = torch.cat([torch.zeros(
            tgt_len, past_key_values_length, dtype=dtype, device=device), mask], dim=-1)
    return mask[None, None, :, :].expand(bsz, 1, tgt_len, tgt_len + past_key_values_length)


# Copied from transformers.models.bart.modeling_bart._expand_mask
def _expand_mask(mask: torch.Tensor, dtype: torch.dtype, tgt_len: Optional[int] = None):
    """
    Expands attention_mask from `[bsz, seq_len]` to `[bsz, 1, tgt_seq_len, src_seq_len]`.
    """
    bsz, src_len = mask.size()
    tgt_len = tgt_len if tgt_len is not None else src_len

    expanded_mask = mask[:, None, None, :].expand(
        bsz, 1, tgt_len, src_len).to(dtype)

    inverted_mask = 1.0 - expanded_mask

    return inverted_mask.masked_fill(inverted_mask.to(torch.bool), torch.finfo(dtype).min)


class LlamaRMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        """
        LlamaRMSNorm is equivalent to T5LayerNorm
        """
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        variance = hidden_states.to(torch.float32).pow(
            2).mean(-1, keepdim=True)
        hidden_states = hidden_states * \
            torch.rsqrt(variance + self.variance_epsilon)

        # convert into half-precision if necessary
        if self.weight.dtype in [torch.float16, torch.bfloat16]:
            hidden_states = hidden_states.to(self.weight.dtype)

        return self.weight * hidden_states


class LlamaRotaryEmbedding(torch.nn.Module):
    def __init__(self, dim, max_position_embeddings=2048, base=10000, device=None):
        super().__init__()
        inv_freq = 1.0 / \
            (base ** (torch.arange(0, dim, 2).float().to(device) / dim))
        self.register_buffer("inv_freq", inv_freq)

        # Build here to make `torch.jit.trace` work.
        self.max_seq_len_cached = max_position_embeddings
        t = torch.arange(self.max_seq_len_cached,
                         device=self.inv_freq.device, dtype=self.inv_freq.dtype)
        freqs = torch.einsum("i,j->ij", t, self.inv_freq)
        # Different from paper, but it uses a different permutation in order to obtain the same calculation
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos()[
                             None, None, :, :], persistent=False)
        self.register_buffer("sin_cached", emb.sin()[
                             None, None, :, :], persistent=False)

    def forward(self, x, seq_len=None):
        # x: [bs, num_attention_heads, seq_len, head_size]
        # This `if` block is unlikely to be run after we build sin/cos in `__init__`. Keep the logic here just in case.
        if seq_len > self.max_seq_len_cached:
            self.max_seq_len_cached = seq_len
            t = torch.arange(self.max_seq_len_cached,
                             device=x.device, dtype=self.inv_freq.dtype)
            freqs = torch.einsum("i,j->ij", t, self.inv_freq)
            # Different from paper, but it uses a different permutation in order to obtain the same calculation
            emb = torch.cat((freqs, freqs), dim=-1).to(x.device)
            self.register_buffer("cos_cached", emb.cos()[
                                 None, None, :, :], persistent=False)
            self.register_buffer("sin_cached", emb.sin()[
                                 None, None, :, :], persistent=False)
        return (
            self.cos_cached[:, :, :seq_len, ...].to(dtype=x.dtype),
            self.sin_cached[:, :, :seq_len, ...].to(dtype=x.dtype),
        )


def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2:]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q, k, cos, sin, position_ids):
    # The first two dimensions of cos and sin are always 1, so we can `squeeze` them.
    cos = cos.squeeze(1).squeeze(0)  # [seq_len, dim]
    sin = sin.squeeze(1).squeeze(0)  # [seq_len, dim]
    cos = cos[position_ids].unsqueeze(1)  # [bs, 1, seq_len, dim]
    sin = sin[position_ids].unsqueeze(1)  # [bs, 1, seq_len, dim]
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


class LlamaMLP(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        hidden_act: str
    ):
        super().__init__()
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.act_fn = ACT2FN[hidden_act]

    def forward(self, x):
        return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))


class LlamaAttention(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(self, config: LlamaConfig):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.r_model = config.r_model
        self.num_heads = config.num_attention_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.max_position_embeddings = config.max_position_embeddings

        if (self.head_dim * self.num_heads) != self.hidden_size:
            raise ValueError(
                f"hidden_size must be divisible by num_heads (got `hidden_size`: {self.hidden_size}"
                f" and `num_heads`: {self.num_heads})."
            )
        self.q_proj = nn.Linear(
            self.r_model, self.num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(
            self.r_model, self.num_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(
            self.r_model, self.num_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(
            self.num_heads * self.head_dim, self.r_model, bias=False)
        self.rotary_emb = LlamaRotaryEmbedding(
            self.head_dim, max_position_embeddings=self.max_position_embeddings)

    def _shape(self, tensor: torch.Tensor, seq_len: int, bsz: int):
        return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        bsz, q_len, _ = hidden_states.size()

        query_states = self.q_proj(hidden_states).view(
            bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = self.k_proj(hidden_states).view(
            bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        value_states = self.v_proj(hidden_states).view(
            bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)

        kv_seq_len = key_states.shape[-2]
        if past_key_value is not None:
            kv_seq_len += past_key_value[0].shape[-2]
        cos, sin = self.rotary_emb(value_states, seq_len=kv_seq_len)
        query_states, key_states = apply_rotary_pos_emb(
            query_states, key_states, cos, sin, position_ids)
        # [bsz, nh, t, hd]

        if past_key_value is not None:
            # reuse k, v, self_attention
            key_states = torch.cat([past_key_value[0], key_states], dim=2)
            value_states = torch.cat([past_key_value[1], value_states], dim=2)

        past_key_value = (key_states, value_states) if use_cache else None

        attn_weights = torch.matmul(
            query_states, key_states.transpose(2, 3)) / math.sqrt(self.head_dim)

        if attn_weights.size() != (bsz, self.num_heads, q_len, kv_seq_len):
            raise ValueError(
                f"Attention weights should be of size {(bsz, self.num_heads, q_len, kv_seq_len)}, but is"
                f" {attn_weights.size()}"
            )

        if attention_mask is not None:
            if attention_mask.size() != (bsz, 1, q_len, kv_seq_len):
                raise ValueError(
                    f"Attention mask should be of size {(bsz, 1, q_len, kv_seq_len)}, but is {attention_mask.size()}"
                )
            attn_weights = attn_weights + attention_mask
            attn_weights = torch.max(attn_weights, torch.tensor(
                torch.finfo(attn_weights.dtype).min))

        # upcast attention to fp32
        attn_weights = nn.functional.softmax(
            attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        attn_output = torch.matmul(attn_weights, value_states)

        if attn_output.size() != (bsz, self.num_heads, q_len, self.head_dim):
            raise ValueError(
                f"`attn_output` should be of size {(bsz, self.num_heads, q_len, self.head_dim)}, but is"
                f" {attn_output.size()}"
            )

        attn_output = attn_output.transpose(1, 2)
        attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)

        attn_output = self.o_proj(attn_output)

        if not output_attentions:
            attn_weights = None

        return attn_output, attn_weights, past_key_value


class LlamaDecoderLayer(nn.Module):
    def __init__(self, config: LlamaConfig):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.self_attn = LlamaAttention(config=config)
        self.mlp = LlamaMLP(
            hidden_size=config.r_model,
            intermediate_size=config.intermediate_size,
            hidden_act=config.hidden_act,
        )
        self.input_layernorm = LlamaRMSNorm(
            config.r_model, eps=config.rms_norm_eps)
        self.post_attention_layernorm = LlamaRMSNorm(
            config.r_model, eps=config.rms_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = False,
    ) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:
        """
        Args:
            hidden_states (`torch.FloatTensor`): input to the layer of shape `(batch, seq_len, embed_dim)`
            attention_mask (`torch.FloatTensor`, *optional*): attention mask of size
                `(batch, 1, tgt_len, src_len)` where padding elements are indicated by very large negative values.
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
            use_cache (`bool`, *optional*):
                If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding
                (see `past_key_values`).
            past_key_value (`Tuple(torch.FloatTensor)`, *optional*): cached past key and value projection states
        """

        residual = hidden_states

        # hidden_states = self.down_linear(hidden_states)
        hidden_states = self.input_layernorm(hidden_states)
        # hidden_states = self.up_linear(hidden_states)

        # Self Attention
        hidden_states, self_attn_weights, present_key_value = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
        )
        hidden_states = residual + hidden_states

        # Fully Connected
        residual = hidden_states
        # hidden_states = self.down_linear(hidden_states)
        hidden_states = self.post_attention_layernorm(hidden_states)
        # hidden_states = self.up_linear(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (self_attn_weights,)

        if use_cache:
            outputs += (present_key_value,)

        return outputs


LLAMA_START_DOCSTRING = r"""
    This model inherits from [`PreTrainedModel`]. Check the superclass documentation for the generic methods the
    library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
    etc.)

    This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
    Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
    and behavior.

    Parameters:
        config ([`LlamaConfig`]):
            Model configuration class with all the parameters of the model. Initializing with a config file does not
            load the weights associated with the model, only the configuration. Check out the
            [`~PreTrainedModel.from_pretrained`] method to load the model weights.
"""


@add_start_docstrings(
    "The bare LLaMA Model outputting raw hidden-states without any specific head on top.",
    LLAMA_START_DOCSTRING,
)
class LlamaPreTrainedModel(PreTrainedModel):
    config_class = LlamaConfig
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _no_split_modules = ["LlamaDecoderLayer"]
    _keys_to_ignore_on_load_unexpected = [r"decoder\.version"]

    def _init_weights(self, module):
        std = self.config.initializer_range
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()

    def _set_gradient_checkpointing(self, module, value=False):
        if isinstance(module, LlamaModel):
            module.gradient_checkpointing = value


LLAMA_INPUTS_DOCSTRING = r"""
    Args:
        input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
            Indices of input sequence tokens in the vocabulary. Padding will be ignored by default should you provide
            it.

            Indices can be obtained using [`AutoTokenizer`]. See [`PreTrainedTokenizer.encode`] and
            [`PreTrainedTokenizer.__call__`] for details.

            [What are input IDs?](../glossary#input-ids)
        attention_mask (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
            Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:

            - 1 for tokens that are **not masked**,
            - 0 for tokens that are **masked**.

            [What are attention masks?](../glossary#attention-mask)

            Indices can be obtained using [`AutoTokenizer`]. See [`PreTrainedTokenizer.encode`] and
            [`PreTrainedTokenizer.__call__`] for details.

            If `past_key_values` is used, optionally only the last `decoder_input_ids` have to be input (see
            `past_key_values`).

            If you want to change padding behavior, you should read [`modeling_opt._prepare_decoder_attention_mask`]
            and modify to your needs. See diagram 1 in [the paper](https://arxiv.org/abs/1910.13461) for more
            information on the default strategy.

            - 1 indicates the head is **not masked**,
            - 0 indicates the head is **masked**.
        position_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Indices of positions of each input sequence tokens in the position embeddings. Selected in the range `[0,
            config.n_positions - 1]`.

            [What are position IDs?](../glossary#position-ids)
        past_key_values (`tuple(tuple(torch.FloatTensor))`, *optional*, returned when `use_cache=True` is passed or when `config.use_cache=True`):
            Tuple of `tuple(torch.FloatTensor)` of length `config.n_layers`, with each tuple having 2 tensors of shape
            `(batch_size, num_heads, sequence_length, embed_size_per_head)`) and 2 additional tensors of shape
            `(batch_size, num_heads, encoder_sequence_length, embed_size_per_head)`.

            Contains pre-computed hidden-states (key and values in the self-attention blocks and in the cross-attention
            blocks) that can be used (see `past_key_values` input) to speed up sequential decoding.

            If `past_key_values` are used, the user can optionally input only the last `decoder_input_ids` (those that
            don't have their past key value states given to this model) of shape `(batch_size, 1)` instead of all
            `decoder_input_ids` of shape `(batch_size, sequence_length)`.
        inputs_embeds (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*):
            Optionally, instead of passing `input_ids` you can choose to directly pass an embedded representation. This
            is useful if you want more control over how to convert `input_ids` indices into associated vectors than the
            model's internal embedding lookup matrix.
        use_cache (`bool`, *optional*):
            If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding (see
            `past_key_values`).
        output_attentions (`bool`, *optional*):
            Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned
            tensors for more detail.
        output_hidden_states (`bool`, *optional*):
            Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for
            more detail.
        return_dict (`bool`, *optional*):
            Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.
"""


@add_start_docstrings(
    "The bare LLaMA Model outputting raw hidden-states without any specific head on top.",
    LLAMA_START_DOCSTRING,
)
class LlamaModel(LlamaPreTrainedModel):
    """
    Transformer decoder consisting of *config.num_hidden_layers* layers. Each layer is a [`LlamaDecoderLayer`]

    Args:
        config: LlamaConfig
    """

    def __init__(self, config: LlamaConfig):
        super().__init__(config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        self.embed_tokens = nn.Embedding(
            config.vocab_size, config.hidden_size, self.padding_idx)

        self.down_linear = nn.Linear(
            config.hidden_size, config.r_model, bias=False)

        self.layers = nn.ModuleList(
            [LlamaDecoderLayer(config) for _ in range(config.num_hidden_layers)])

        self.up_linear = nn.Linear(
            config.r_model, config.hidden_size, bias=False)

        self.norm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        self.gradient_checkpointing = False
        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.embed_tokens

    def set_input_embeddings(self, value):
        self.embed_tokens = value

    # Copied from transformers.models.bart.modeling_bart.BartDecoder._prepare_decoder_attention_mask
    def _prepare_decoder_attention_mask(self, attention_mask, input_shape, inputs_embeds, past_key_values_length):
        # create causal mask
        # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
        combined_attention_mask = None
        if input_shape[-1] > 1:
            combined_attention_mask = _make_causal_mask(
                input_shape,
                inputs_embeds.dtype,
                device=inputs_embeds.device,
                past_key_values_length=past_key_values_length,
            )

        if attention_mask is not None:
            # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
            expanded_attn_mask = _expand_mask(attention_mask, inputs_embeds.dtype, tgt_len=input_shape[-1]).to(
                inputs_embeds.device
            )
            combined_attention_mask = (
                expanded_attn_mask if combined_attention_mask is None else expanded_attn_mask +
                combined_attention_mask
            )

        return combined_attention_mask

    @add_start_docstrings_to_model_forward(LLAMA_INPUTS_DOCSTRING)
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, BaseModelOutputWithPast]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # retrieve input_ids and inputs_embeds
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError(
                "You cannot specify both decoder_input_ids and decoder_inputs_embeds at the same time")
        elif input_ids is not None:
            batch_size, seq_length = input_ids.shape
        elif inputs_embeds is not None:
            batch_size, seq_length, _ = inputs_embeds.shape
        else:
            raise ValueError(
                "You have to specify either decoder_input_ids or decoder_inputs_embeds")

        seq_length_with_past = seq_length
        past_key_values_length = 0

        if past_key_values is not None:
            past_key_values_length = past_key_values[0][0].shape[2]
            seq_length_with_past = seq_length_with_past + past_key_values_length

        if position_ids is None:
            device = input_ids.device if input_ids is not None else inputs_embeds.device
            position_ids = torch.arange(
                past_key_values_length, seq_length + past_key_values_length, dtype=torch.long, device=device
            )
            position_ids = position_ids.unsqueeze(0).view(-1, seq_length)
        else:
            position_ids = position_ids.view(-1, seq_length).long()

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)
        # embed positions
        if attention_mask is None:
            attention_mask = torch.ones(
                (batch_size, seq_length_with_past), dtype=torch.bool, device=inputs_embeds.device
            )
        attention_mask = self._prepare_decoder_attention_mask(
            attention_mask, (batch_size,
                             seq_length), inputs_embeds, past_key_values_length
        )

        hidden_states = inputs_embeds

        if self.gradient_checkpointing and self.training:
            if use_cache:
                logger.warning_once(
                    "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`..."
                )
                use_cache = False

        # decoder layers
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        next_decoder_cache = () if use_cache else None

        hidden_states = self.down_linear(hidden_states)

        for idx, decoder_layer in enumerate(self.layers):
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            past_key_value = past_key_values[idx] if past_key_values is not None else None

            if self.gradient_checkpointing and self.training:

                def create_custom_forward(module):
                    def custom_forward(*inputs):
                        # None for past_key_value
                        return module(*inputs, output_attentions, None)

                    return custom_forward

                layer_outputs = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(decoder_layer),
                    hidden_states,
                    attention_mask,
                    position_ids,
                    None,
                )
            else:
                layer_outputs = decoder_layer(
                    hidden_states,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    past_key_value=past_key_value,
                    output_attentions=output_attentions,
                    use_cache=use_cache,
                )

            hidden_states = layer_outputs[0]

            if use_cache:
                next_decoder_cache += (
                    layer_outputs[2 if output_attentions else 1],)

            if output_attentions:
                all_self_attns += (layer_outputs[1],)
                
        hidden_states = self.up_linear(hidden_states)

        hidden_states = self.norm(hidden_states)

        # add hidden states from the last decoder layer
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        next_cache = next_decoder_cache if use_cache else None
        if not return_dict:
            return tuple(v for v in [hidden_states, next_cache, all_hidden_states, all_self_attns] if v is not None)
        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=next_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
        )


class LlamaForCausalLM(LlamaPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.model = LlamaModel(config)

        self.lm_head = nn.Linear(
            config.hidden_size, config.vocab_size, bias=False)

        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.model.embed_tokens

    def set_input_embeddings(self, value):
        self.model.embed_tokens = value

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def set_decoder(self, decoder):
        self.model = decoder

    def get_decoder(self):
        return self.model

    @add_start_docstrings_to_model_forward(LLAMA_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=CausalLMOutputWithPast, config_class=_CONFIG_FOR_DOC)
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        r"""
        Args:
            labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
                Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
                config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
                (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.

        Returns:

        Example:

        ```python
        >>> from transformers import AutoTokenizer, LlamaForCausalLM

        >>> model = LlamaForCausalLM.from_pretrained(PATH_TO_CONVERTED_WEIGHTS)
        >>> tokenizer = AutoTokenizer.from_pretrained(PATH_TO_CONVERTED_TOKENIZER)

        >>> prompt = "Hey, are you consciours? Can you talk to me?"
        >>> inputs = tokenizer(prompt, return_tensors="pt")

        >>> # Generate
        >>> generate_ids = model.generate(inputs.input_ids, max_length=30)
        >>> tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        "Hey, are you consciours? Can you talk to me?\nI'm not consciours, but I can talk to you."
        ```"""

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        hidden_states = outputs[0]
        # logits = self.lm_head(self.up_linear(hidden_states))
        logits = self.lm_head(hidden_states)

        loss = None
        if labels is not None:
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = CrossEntropyLoss()
            shift_logits = shift_logits.view(-1, self.config.vocab_size)
            shift_labels = shift_labels.view(-1)
            # Enable model parallelism
            shift_labels = shift_labels.to(shift_logits.device)
            loss = loss_fct(shift_logits, shift_labels)

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    def prepare_inputs_for_generation(
        self, input_ids, past_key_values=None, attention_mask=None, inputs_embeds=None, **kwargs
    ):
        if past_key_values:
            input_ids = input_ids[:, -1:]

        position_ids = kwargs.get("position_ids", None)
        if attention_mask is not None and position_ids is None:
            # create position_ids on the fly for batch generation
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)
            if past_key_values:
                position_ids = position_ids[:, -1].unsqueeze(-1)

        # if `inputs_embeds` are passed, we only want to use them in the 1st generation step
        if inputs_embeds is not None and past_key_values is None:
            model_inputs = {"inputs_embeds": inputs_embeds}
        else:
            model_inputs = {"input_ids": input_ids}

        model_inputs.update(
            {
                "position_ids": position_ids,
                "past_key_values": past_key_values,
                "use_cache": kwargs.get("use_cache"),
                "attention_mask": attention_mask,
            }
        )
        return model_inputs

    @staticmethod
    def _reorder_cache(past_key_values, beam_idx):
        reordered_past = ()
        for layer_past in past_key_values:
            reordered_past += (tuple(past_state.index_select(0, beam_idx)
                               for past_state in layer_past),)
        return reordered_past


@add_start_docstrings(
    """
    The LLaMa Model transformer with a sequence classification head on top (linear layer).

    [`LlamaForSequenceClassification`] uses the last token in order to do the classification, as other causal models
    (e.g. GPT-2) do.

    Since it does classification on the last token, it requires to know the position of the last token. If a
    `pad_token_id` is defined in the configuration, it finds the last token that is not a padding token in each row. If
    no `pad_token_id` is defined, it simply takes the last value in each row of the batch. Since it cannot guess the
    padding tokens when `inputs_embeds` are passed instead of `input_ids`, it does the same (take the last value in
    each row of the batch).
    """,
    LLAMA_START_DOCSTRING,
)
class LlamaForSequenceClassification(LlamaPreTrainedModel):
    _keys_to_ignore_on_load_missing = [r"lm_head.weight"]

    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.model = LlamaModel(config)
        self.score = nn.Linear(config.hidden_size, self.num_labels, bias=False)

        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.model.embed_tokens

    def set_input_embeddings(self, value):
        self.model.embed_tokens = value

    @add_start_docstrings_to_model_forward(LLAMA_INPUTS_DOCSTRING)
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, SequenceClassifierOutputWithPast]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the sequence classification/regression loss. Indices should be in `[0, ...,
            config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
            `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        transformer_outputs = self.model(
            input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        hidden_states = transformer_outputs[0]
        logits = self.score(hidden_states)

        if input_ids is not None:
            batch_size = input_ids.shape[0]
        else:
            batch_size = inputs_embeds.shape[0]

        if self.config.pad_token_id is None and batch_size != 1:
            raise ValueError(
                "Cannot handle batch sizes > 1 if no padding token is defined.")
        if self.config.pad_token_id is None:
            sequence_lengths = -1
        else:
            if input_ids is not None:
                sequence_lengths = (
                    torch.ne(input_ids, self.config.pad_token_id).sum(-1) - 1).to(logits.device)
            else:
                sequence_lengths = -1

        pooled_logits = logits[torch.arange(
            batch_size, device=logits.device), sequence_lengths]

        loss = None
        if labels is not None:
            labels = labels.to(logits.device)
            if self.config.problem_type is None:
                if self.num_labels == 1:
                    self.config.problem_type = "regression"
                elif self.num_labels > 1 and (labels.dtype == torch.long or labels.dtype == torch.int):
                    self.config.problem_type = "single_label_classification"
                else:
                    self.config.problem_type = "multi_label_classification"

            if self.config.problem_type == "regression":
                loss_fct = MSELoss()
                if self.num_labels == 1:
                    loss = loss_fct(pooled_logits.squeeze(), labels.squeeze())
                else:
                    loss = loss_fct(pooled_logits, labels)
            elif self.config.problem_type == "single_label_classification":
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(
                    pooled_logits.view(-1, self.num_labels), labels.view(-1))
            elif self.config.problem_type == "multi_label_classification":
                loss_fct = BCEWithLogitsLoss()
                loss = loss_fct(pooled_logits, labels)
        if not return_dict:
            output = (pooled_logits,) + transformer_outputs[1:]
            return ((loss,) + output) if loss is not None else output

        return SequenceClassifierOutputWithPast(
            loss=loss,
            logits=pooled_logits,
            past_key_values=transformer_outputs.past_key_values,
            hidden_states=transformer_outputs.hidden_states,
            attentions=transformer_outputs.attentions,
        )
