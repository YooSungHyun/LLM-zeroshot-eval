import torch
import pandas as pd
from datasets import load_dataset
import os
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, HfArgumentParser, TrainingArguments
from copy import deepcopy
from utils.Collator import DataCollatorForSupervisedDataset
from typing import Dict, List, Union
import logging
from transformers.trainer_utils import is_main_process
import argparse
from trl import (
    ModelConfig,
    ScriptArguments,
    SFTConfig,
    SFTTrainer,
    TrlParser,
    DataCollatorForCompletionOnlyLM,
    get_kbit_device_map,
    get_peft_config,
    get_quantization_config,
)

os.environ["TORCHDYNAMO_DISABLE"] = "1"
IGNORE_INDEX = -100


def main(script_args, training_args, model_args):
    # 1. 교사 모델 및 토크나이저 로드
    system_prompt = "당신은 Alibaba Cloud에 의해 만들어진 Qwen입니다. 당신은 도움이 되는 어시스턴트입니다."
    quantization_config = get_quantization_config(model_args)
    model_kwargs = dict(
        revision=model_args.model_revision,
        trust_remote_code=model_args.trust_remote_code,
        attn_implementation=model_args.attn_implementation,
        torch_dtype=model_args.torch_dtype,
        use_cache=False if training_args.gradient_checkpointing else True,
        device_map=get_kbit_device_map() if quantization_config is not None else None,
        quantization_config=quantization_config,
    )
    training_args.model_init_kwargs = model_kwargs
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.model_name_or_path, trust_remote_code=model_args.trust_remote_code, use_fast=True
    )
    dataset = load_dataset("json", data_files="")["train"]
    dataset = dataset.shuffle(seed=42).select(range(250))

    def raw_to_sft_format(row):
        """input, instruction, output 컬럼만 남기고 전처리하는 함수"""
        row["input"] = (row.get("input", "")).strip()

        # instruction은 여러 필드 중 하나를 선택
        row["instruction"] = (row.get("instruction") or row.get("question") or row.get("prompt") or "").strip()

        # output은 여러 필드 중 하나를 선택
        row["output"] = (row.get("output") or row.get("chosen") or row.get("response") or "").strip()

        return row

    def get_remove_columns_names(total_column_names):
        sft_column_names = ["input", "instruction", "output"]
        return list(set(total_column_names).difference(sft_column_names))

    def instruct_dataset_preprocess(row, tokenizer: AutoTokenizer):
        if not row["input"] == "":
            row["instruction"] += "\n" + row["input"]

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": row["instruction"]},
            {"role": "assistant", "content": row["output"]},
        ]
        tokenized_text = tokenizer.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=False, truncation=False
        )

        row["input_ids"] = tokenized_text
        return row

    dataset = dataset.map(raw_to_sft_format, remove_columns=get_remove_columns_names(dataset.column_names), num_proc=8)
    dataset = dataset.map(
        instruct_dataset_preprocess,
        remove_columns=dataset.column_names,
        num_proc=8,
        fn_kwargs={"tokenizer": tokenizer},
    )
    total_result = list()

    def compute_metrics(pred: Dict[str, Union[List[int], torch.Tensor]], compute_result) -> Dict[str, float]:
        """compute_metrics eval_loop, pred_loop에서 사용되는 compute_metrics
        Args:
            pred (Dict[str, Union[List[int], torch.Tensor]]): 예측값
        Returns:
            metrics (Dict[str, float]): 계산된 Metrics dict
        """
        logits = pred.predictions.to("cpu")
        labels = pred.label_ids.to("cpu")
        probs = torch.softmax(logits, dim=-1)
        mask = labels != -100
        indices = torch.argmax(mask.int(), dim=1)
        has_non_negative_100 = mask.any(dim=1)  # 각 행에 -100이 아닌 값이 있는지 여부
        indices[~has_non_negative_100] = -1
        batch_size, seq_len, embed_size = logits.shape
        log_probs = []
        entropies = []
        accuracies = []
        for batch_idx in range(batch_size):
            # 각 배치의 True 시작과 끝 위치 찾기
            true_indices = torch.nonzero(mask[batch_idx], as_tuple=True)[0]
            start_idx = true_indices[0] - 1
            end_idx = true_indices[-1] - 1

            # 인덱스 범위를 클리핑 (0 이상, seq_len 미만으로 제한)
            start_idx = max(start_idx, 0)
            end_idx = min(end_idx, seq_len - 1)

            # 해당 구간의 텐서를 추가
            shifted_probs = probs[batch_idx, start_idx : end_idx + 1, :]
            shifted_labels = labels[batch_idx, true_indices]
            target_probs = shifted_probs.gather(1, shifted_labels.unsqueeze(-1)).squeeze(-1)
            # label index에 대한 vocab의 평균 log-prob
            log_prob = torch.log(target_probs + 1e-12).mean().item()

            # 엔트로피 계산
            entropy = -(shifted_probs * torch.log(shifted_probs + 1e-12)).sum(dim=-1).mean().item()

            # 정확도 계산
            predictions = torch.argmax(shifted_probs, dim=-1)
            correct = (predictions == shifted_labels).sum().item()
            correctness = correct / shifted_labels.numel()
            log_probs.append(log_prob)
            entropies.append(entropy)
            accuracies.append(correctness)
            total_result.append({"log_prob": log_prob, "entropy": entropy, "accuracy": correctness})
        metrics = {"log_prob": log_probs, "entropy": entropies, "accuracy": accuracies}
        return metrics

    collator = DataCollatorForCompletionOnlyLM(
        tokenizer=tokenizer, instruction_template="<|im_start|>user\n", response_template="<|im_start|>assistant\n"
    )
    trainer = SFTTrainer(
        model=model_args.model_name_or_path,
        args=training_args,
        processing_class=tokenizer,
        data_collator=collator,
        compute_metrics=compute_metrics,
        peft_config=get_peft_config(model_args),
    )
    pred_output = trainer.evaluate(eval_dataset=dataset)
    df = pd.DataFrame(total_result, columns=["log_prob", "entropy", "accuracy"])
    # 열별 통계 계산
    mean_per_column = df.mean()
    # 일반적으로 log-prob > -2.0과 entropy < 1.0이면 적절한 생성 결과로 간주됩니다. log(1/vocabs)의 균등확률을 비교해보자.
    quantiles_per_column = df.quantile([0.25, 0.50, 0.75])

    # 결과 출력
    pd.set_option("display.max_columns", None)
    print("Mean per column:\n", mean_per_column)
    print("\nQuantiles per column:\n", quantiles_per_column)


def make_parser(subparsers: argparse._SubParsersAction = None):
    dataclass_types = (ScriptArguments, SFTConfig, ModelConfig)
    if subparsers is not None:
        parser = subparsers.add_parser("sft", help="Run the SFT training script", dataclass_types=dataclass_types)
    else:
        parser = TrlParser(dataclass_types)
    return parser


if __name__ == "__main__":
    parser = make_parser()
    script_args, training_args, model_args = parser.parse_args_and_config()
    main(script_args, training_args, model_args)
