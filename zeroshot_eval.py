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

os.environ["TORCHDYNAMO_DISABLE"] = "1"
IGNORE_INDEX = -100


def main(training_args: TrainingArguments):
    # 1. 교사 모델 및 토크나이저 로드
    system_prompt = "You are a helpful chatbot. Respond in the same language as the user's question."
    model_name = "microsoft/phi-4"  # 원하는 모델 이름으로 변경 가능
    model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto")
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(model_name, model_max_length=8192, truncation_side="left")
    instruct_dataset1 = load_dataset("json", data_files="")["train"]
    instruct_dataset1 = instruct_dataset1.shuffle(seed=42).select(range(250))

    def raw_to_sft_fromat(row):
        # 각종 포맷을 input, instruction, output의 컬럼만 남기고 전처리 하는 함수
        inputs = ""
        instruction = ""
        output = ""
        if "input" in row:
            inputs = row["input"]

        if "instruction" in row:
            instruction = row["instruction"]
        if "question" in row:
            instruction = row["question"]
        if "prompt" in row:
            instruction = row["prompt"]

        if "output" in row:
            output = row["output"]
        if "chosen" in row:
            output = row["chosen"]
        if "response" in row:
            output = row["response"]

        row["input"] = inputs
        row["instruction"] = instruction
        row["output"] = output
        return row

    def get_remove_columns_names(total_column_names):
        sft_column_names = ["input", "instruction", "output"]
        return list(set(total_column_names).difference(sft_column_names))

    instruct_dataset1 = instruct_dataset1.map(
        raw_to_sft_fromat, remove_columns=get_remove_columns_names(instruct_dataset1.column_names), num_proc=8
    )

    def instruct_dataset_preprocess(row):
        if not row["input"] == "":
            row["instruction"] += "\n" + row["input"]

        input_text = row["instruction"]
        messages = [{"role": "system", "content": system_prompt}, {"role": "user", "content": input_text}]
        input_template_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        label_text = f"{row['output']}<|im_end|>"

        total_text = input_template_text + label_text

        input_seq_token_len = len(tokenizer(input_template_text, add_special_tokens=False)["input_ids"])
        tokenized_text = tokenizer(
            total_text, return_token_type_ids=False, return_tensors="pt", add_special_tokens=False
        )

        labels_ids = deepcopy(tokenized_text["input_ids"][0])
        labels_ids[:input_seq_token_len] = IGNORE_INDEX

        row["input_ids"] = tokenized_text["input_ids"][0]
        row["attention_mask"] = tokenized_text["attention_mask"][0]
        row["labels"] = labels_ids
        row["lengths"] = len(tokenized_text["input_ids"][0])
        return row

    instruct_dataset1 = instruct_dataset1.map(
        instruct_dataset_preprocess, remove_columns=instruct_dataset1.column_names, num_proc=8
    )
    asis_len = len(instruct_dataset1)
    dataset = instruct_dataset1.filter(lambda x: x["lengths"] <= 8192)
    logging.info(f"dataset length to {asis_len} -> {len(dataset)}")
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

    collator = DataCollatorForSupervisedDataset(tokenizer=tokenizer)
    trainer = Trainer(model=model, data_collator=collator, args=training_args, compute_metrics=compute_metrics)
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


if __name__ == "__main__":
    parser = HfArgumentParser((TrainingArguments))
    training_args = parser.parse_args_into_dataclasses()
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO if is_main_process(training_args[0].local_rank) else logging.WARN,
    )
    main(training_args=training_args[0])
