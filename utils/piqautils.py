from transformers import LlamaTokenizerFast
from typing import List, Tuple, Union, Optional, Any

prompt_add = ',choose the only sol for goal.'


class PiqaUtils:
    def __init__(self,
                 data_args: Any,
                 goal_columns: str,
                 sol1_columns: str,
                 sol2_columns: str,
                 tokenizer: Optional[LlamaTokenizerFast] = None,
                 #  max_seq_length: Optional[int],
                 #  max_answer_length: Optional[int],
                 #  padding: Optional[Union[str, bool]],
                 ):
        self.data_args = data_args
        self.goal_columns = goal_columns
        self.sol1_columns = sol1_columns
        self.sol2_columns = sol2_columns
        self.tokenizer = tokenizer
        # self.max_seq_length = max_seq_length
        # self.max_answer_length = max_answer_length
        # self.padding = padding

    def preprocess_function(self,
                            examples,
                            goal_columns: str,
                            sol1_columns: str,
                            sol2_columns: str,
                            ):
        goal = examples[goal_columns]
        sol1 = examples[sol1_columns]
        sol2 = examples[sol2_columns]

        def generate_instruction(_goal):
            return " ".join(["\"goal:\"", _goal.lstrip(), prompt_add])

        def generate_input(_sol1, _sol2):
            return " ".join(["\"sol1:\"", _sol1, "\"sol2:\"", _sol2])

        inputs = [generate_input(_sol1, _sol2)
                  for _sol1, _sol2 in zip(sol1, sol2)]
        instruction = [generate_instruction(_goal) for _goal in goal]

        return instruction, inputs

    # def preprocess_function(self, examples):
    #     instruction, inputs = preprocess_piqa_batch(
    #         examples, self.goal_columns, self.sol1_columns, self.sol2_columns
    #     )

    #     self.tokenizer.pad_token_id = 0
    #     model_instruction = self.tokenizer()
