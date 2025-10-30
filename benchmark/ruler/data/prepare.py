import os

"""
Prepare jsonl with field `input` and `outputs`.
{
    "index" int,
    "input": str,
    "outputs": [str],
}

python prepare.py     --save_dir ./     --benchmark synthetic     --task niah_single_1     --tokenizer_path tokenizer.model     --tokenizer_type nemo     --max_seq_length 4096     --model_template_type base     --num_samples 10 """
import argparse
import importlib
import json
import math
import re
import subprocess
import time
from pathlib import Path

import nltk
import yaml

try:
    nltk.data.find("tokenizers/punkt")
except LookupError:
    nltk.download("punkt")
Templates = {
    "base": "{task_template}",
    "meta-chat": "[INST] {task_template} [/INST]",
    "vicuna-chat": "A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions. USER: {task_template} ASSISTANT:",
    "lwm-chat": "You are a helpful assistant. USER: {task_template} ASSISTANT: ",
    "command-r-chat": "<BOS_TOKEN><|START_OF_TURN_TOKEN|><|USER_TOKEN|>{task_template}<|END_OF_TURN_TOKEN|><|START_OF_TURN_TOKEN|><|CHATBOT_TOKEN|>",
    "chatglm-chat": """[gMASK]sop<|user|> 
 {task_template}<|assistant|> 
 """,
    "RWKV": """User: hi

Assistant: Hi. I am your assistant and I will provide expert full response in full details. Please feel free to ask any question and I will always answer it

User: {task_template}

Assistant:""",
}


def main(args):
    start_time = time.time()
    curr_folder = os.path.dirname(os.path.abspath(__file__))
    try:
        module = importlib.import_module(f"{args.benchmark}.constants")
    except ImportError:
        print(f"Module data.{args.benchmark}.constants not found.")
    tasks_base = module.TASKS
    with open(os.path.join(curr_folder, f"../{args.benchmark}.yaml"), "r") as f:
        tasks_customized = yaml.safe_load(f)
    if args.task not in tasks_customized:
        raise ValueError(f"{args.task} is not found in config_tasks.yaml")
    config = tasks_customized.get(args.task)
    config.update(tasks_base[config["task"]])
    assert args.model_template_type in Templates, print(
        f"{args.model_template_type} is not found in {Templates.keys()}"
    )
    model_template = Templates[args.model_template_type]
    task_template = config["template"]
    answer_prefix = config["answer_prefix"] if "answer_prefix" in config else ""
    config["template"] = (
        model_template.format(task_template=task_template) + answer_prefix
    )
    chunks = [
        (
            args.num_samples // args.chunk_amount
            + (1 if i < args.num_samples % args.chunk_amount else 0)
        )
        for i in range(args.chunk_amount)
    ]
    num_samples = chunks[args.chunk_idx]
    pre_samples = sum(chunks[: args.chunk_idx])
    random_seed = 42 + args.chunk_idx
    try:
        script = os.path.join(curr_folder, args.benchmark, f"{config['task']}.py")
        additional_args = " ".join([f"--{k} {v}" for k, v in config["args"].items()])
        command = f"""python -u {script}         --save_dir  {args.save_dir}         --save_name {args.task}         --subset {args.subset}         --tokenizer_path {args.tokenizer_path}         --tokenizer_type {args.tokenizer_type}         --max_seq_length {args.max_seq_length}         --tokens_to_generate {config['tokens_to_generate']}         --num_samples {num_samples}         --random_seed {random_seed}         {additional_args}         {f'--remove_newline_tab' if args.remove_newline_tab else ''}         {f'--pre_samples {pre_samples}' if config['task'] == 'qa' else ''}         --template "{config['template']}"
        """
        print(command)
        result = subprocess.run(
            command,
            shell=True,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        if result.returncode == 0:
            print("Output:")
            print(result.stdout)
        else:
            print("Error:")
            print(result.stderr)
    except subprocess.CalledProcessError as e:
        print("Error output:", e.stderr)
    save_file = args.save_dir / args.task / f"{args.subset}.jsonl"
    print(f"Prepare {args.task} with lines: {args.num_samples} to {save_file}")
    print(f"Used time: {round((time.time() - start_time) / 60, 1)} minutes")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--save_dir", type=Path, required=True, help="dataset folder to save dataset"
    )
    parser.add_argument(
        "--benchmark", type=str, default="synthetic", help="Options: [synthetic]"
    )
    parser.add_argument("--task", type=str, required=True, help="tasks in benchmark")
    parser.add_argument(
        "--tokenizer_path", type=str, required=True, help="path to the tokenizer model"
    )
    parser.add_argument(
        "--tokenizer_type", type=str, default="nemo", help="[Options] nemo, hf, openai."
    )
    parser.add_argument(
        "--max_seq_length",
        type=int,
        required=True,
        help="max sequence length including all input tokens and generated tokens.",
    )
    parser.add_argument(
        "--model_template_type",
        type=str,
        default="base",
        help="Options in `template.py`",
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=500,
        help="maximum number of samples we want to test",
    )
    parser.add_argument(
        "--remove_newline_tab",
        action="store_true",
        help="""remove `
` and `	` in all strings.""",
    )
    parser.add_argument(
        "--chunk_idx", type=int, default=0, help="index of current split chunk"
    )
    parser.add_argument(
        "--chunk_amount", type=int, default=1, help="size of split chunk"
    )
    parser.add_argument(
        "--subset", type=str, default="validation", help="Options: validation or test"
    )
    parser.add_argument("--random_seed", type=int, default=42)
    args = parser.parse_args()
    main(args)
