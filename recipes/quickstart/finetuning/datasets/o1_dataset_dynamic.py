# coding=utf-8

import os
from pathlib import Path
import random

import datasets
import torch

class o1preview(torch.utils.data.Dataset):
    def __init__(
        self,
        dataset_config,
        tokenizer,
        split,
        num_recycles,
    ):
        if dataset_config and dataset_config.data_path:
            data_dir = dataset_config.data_path
        else:
            data_dir = Path(__file__).parent / "o1_dataset"
        self.dataset = datasets.load_dataset("json", split=split, data_dir=data_dir)

        self.tokenizer = tokenizer
        self.num_recycles = num_recycles

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        def apply_prompt_template(sample):
            def submission_filter(submission):
                return submission.get("code")

            submissions = list(filter(submission_filter, sample.get("submission", [])))
            if not submissions:
                return None

            input_text = f"帮我解决算法题。\n{sample['problem']}\n"

            # 随机挑选一些答案充当中间的推理步骤
            num_submissions = random.randint(0, min(self.num_recycles, len(submissions)))
            if num_submissions > 0:
                selected_submissions = random.sample(submissions, num_submissions)


                submission_template = "答案{idx}，得分：{score}, 提交状态：{status}，代码: \n {code} \n"
                if selected_submissions:
                    def apply_submition_template(submission):
                        idx, submission = submission
                        score = submission.get("score")
                        if score is None:
                            score = 0

                        status = submission["status"]
                        code = submission["code"]
                        return submission_template.format(idx=idx + 1, score=score, status=status, code=code)

                    selected_submissions = list(map(apply_submition_template, enumerate(selected_submissions)))
                    input_text += f"它有以下可能的答案：\n\n"
                    input_text += "\n".join(selected_submissions)
                    input_text += "\n\n"
            input_text += "最优答案是:\n"

            # 选择得分最高且代码字符最少的提交作为终极答案
            def submission_key(submission):
                score = submission.get("score")
                return score if score else 0
            best_submission = max(submissions, key=submission_key)

            return {
                "input": input_text,
                "output": best_submission["code"]
            }

        def tokenize_add_label(sample):
            input_ids = self.tokenizer.encode(self.tokenizer.bos_token + sample["input"], add_special_tokens=False)
            output_ids = self.tokenizer.encode(sample["output"] + self.tokenizer.eos_token, add_special_tokens=False)

            sample = {
                "input_ids": input_ids + output_ids,
                "attention_mask": [1] * (len(input_ids) + len(output_ids)),
                "labels": [-100] * len(input_ids) + output_ids,
            }

            return sample

        sample = self.dataset[int(index)]
        sample = apply_prompt_template(sample)
        if self.tokenizer is not None:
            sample = tokenize_add_label(sample)
        return sample

def get_custom_dataset(dataset_config, tokenizer, split):
    random.seed(None)

    num_recycles = int(os.environ.get("NUM_RECYCLES", 2))
    return o1preview(dataset_config, tokenizer, split, num_recycles)

if __name__ == "__main__":
    import inspect
    import fire
    from llama_recipes.configs import datasets as dataset_list
    from llama_recipes.utils import config_utils

    def main(**kwargs):
        dataset_config = {k:v for k, v in inspect.getmembers(dataset_list)}["custom_dataset"]()
        config_utils.update_config(dataset_config, **kwargs)
        split = kwargs.get("split", "train")
        print("=================================")
        print(f"data_path: {dataset_config.data_path}, split: {split}")
        print("=================================")
        dataset = get_custom_dataset(dataset_config=dataset_config, tokenizer=None, split=split)
        for idx, sample in enumerate(dataset):
            print("-------------------------------------")
            print(f"INDEX: {idx}")
            print("-------------------------------------")
            print(f"INPUT: {sample['input']}")
            print(f"OUTPUT: {sample['output']}")
    fire.Fire(main)
