import time
import math
import numpy as np
import torch
from transformers import Seq2SeqTrainer
from transformers.trainer_utils import PredictionOutput, speed_metrics
from typing import Dict, List, Any, Tuple, Callable, Union, Optional, Sequence

class LlamaTrainer(Seq2SeqTrainer):

    def __init__(self, *args, **kw_args):
        super().__init__(*args, **kw_args)
    
    def collect(
        self, ignore_keys=None, metric_key_prefix: str = "test", **gen_kwargs
    ):
        gen_kwargs = gen_kwargs.copy()
        gen_kwargs["max_length"] = (
            gen_kwargs["max_length"] if gen_kwargs.get("max_length") is not None else self.args.generation_max_length
        )
        gen_kwargs["num_beams"] = (
            gen_kwargs["num_beams"] if gen_kwargs.get("num_beams") is not None else self.args.generation_num_beams
        )
        self._gen_kwargs = gen_kwargs

        eval_dataset = self.eval_dataset
        eval_dataloader = self.get_eval_dataloader(eval_dataset)

        # Temporarily disable metric computation, we will do it in the loop here.
        compute_metrics = self.compute_metrics
        self.compute_metrics = None
        start_time = time.time()
        eval_loop = self.prediction_loop if self.args.use_legacy_prediction_loop else self.evaluation_loop
        try:
            output = eval_loop(
                eval_dataloader,
                description="Evaluation",
                prediction_loss_only=True,
                ignore_keys=ignore_keys,
                metric_key_prefix=metric_key_prefix,
            )
        finally:
            self.compute_metrics = compute_metrics
        total_batch_size = self.args.eval_batch_size * self.args.world_size
        if f"{metric_key_prefix}_jit_compilation_time" in output.metrics:
            start_time += output.metrics[f"{metric_key_prefix}_jit_compilation_time"]
        output.metrics.update(
            speed_metrics(
                metric_key_prefix,
                start_time,
                num_samples=output.num_samples,
                num_steps=math.ceil(output.num_samples / total_batch_size),
            )
        )
        metrics = output.metrics

        if self.args.should_log:
            # Only the main node log the results by default
            self.log(metrics)

        self.control = self.callback_handler.on_evaluate(self.args, self.state, self.control, metrics)
        return metrics
