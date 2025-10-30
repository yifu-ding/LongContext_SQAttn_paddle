import os

import paddle
from paddle_utils import *
os.environ["USE_TORCH"] = "0"

"""
Prepare prediction jsonl with field `pred` .
dataset jsonl:
{
    "index" int,
    "input": str,
    "outputs": [str],
}

prediction jsonl: 
{
    "index" int,
    "input": str,
    "outputs": [str],
    "pred": str,
}
"""
import argparse
import importlib
import json
import random
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import yaml
from tqdm import tqdm
from utils import load_data
from model_hub import QwenModel, LlamaModel
SERVER_TYPES = "trtllm", "vllm", "openai", "gemini", "hf", "mamba"

from sqattn.compress import compress_model
from sqattn.attn_replacer import process_model

def seed_everything(seed):
    paddle.seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    PaddleFlag.cudnn_benchmark = False
    PaddleFlag.cudnn_deterministic = True


class HuggingFaceModel:
    def __init__(
        self,
        model_name,
        max_len,
        max_new_len,
        attn_type,
        dtype,
        device,
        synthetic_len,
    ) -> None:
        if "llama" in model_name.lower():
            llm = LlamaModel(
                model_name,
                max_length=max_len + max_new_len,
                dtype=dtype,
                device_map=device,
                attention_type=attn_type,
            )
        elif "qwen" in model_name.lower():
            llm = QwenModel(
                model_name,
                max_length=max_len + max_new_len,
                dtype=dtype,
                device_map=device,
                attention_type=attn_type,
            )
        else:
            raise ValueError(f"Unsupported model: {model_name}")
        self.llm = llm
        self.max_new_len = max_new_len
        self.attn_type = attn_type
        self.model_name = model_name
        self.synthetic_len = synthetic_len

    def __call__(self, prompt: str, **kwargs) -> Dict[str, List[str]]:
        generated_text = get_pred(
            self.llm,
            input_text=prompt,
            max_new_tokens=self.max_new_len,
            attn_type=self.attn_type,
        )
        return {"text": [generated_text]}


class ServerAction(argparse.Action):
    def __call__(self, parser, namespace, values, option_string=None):
        namespace.server_type = values


def get_llm(
    model_name,
    max_len,
    max_new_len,
    attn_type,
    dtype,
    device,
    synthetic_len,
):
    if args.server_type == "hf":
        llm = HuggingFaceModel(
            model_name=model_name,
            max_len=max_len,
            max_new_len=max_new_len,
            attn_type=attn_type,
            dtype=dtype,
            device=device,
            synthetic_len=synthetic_len,
        )
    else:
        raise RuntimeError(f"Unsupported server type {args.server_type}")
    return llm


def get_pred(
    llm,
    input_text: str,
    max_new_tokens: int,
    attn_type: str,
) -> str:
    llm.tokenizer.pad_token = llm.tokenizer.eos_token
    llm.tokenizer.padding_side = "left"
    inputs = llm.tokenizer([input_text], return_tensors="pd", padding=True)
    input_ids = inputs.input_ids
    attention_masks = inputs.attention_mask
    out = llm.generate(
        attention_type=attn_type,
        inputs_ids=input_ids,
        attention_masks=attention_masks,
        max_new_length=max_new_tokens,
    )
    output = llm.tokenizer.batch_decode(out, skip_special_tokens=True)
    print("Chunked generation:", output[0])
    return output[0]


def get_output(
    llm, outputs_parallel, idx, index, input, outputs, others, truncation, length
):
    pred = llm(prompt=input)
    if len(pred["text"]) > 0:
        outputs_parallel[idx] = {
            "index": index,
            "pred": pred["text"][0],
            "input": input,
            "outputs": outputs,
            "others": others,
            "truncation": truncation,
            "length": length,
        }
    return outputs_parallel[idx]


def main(args):
    start_time = time.time()
    curr_folder = os.path.dirname(os.path.abspath(__file__))
    try:
        sys.path.append(os.path.dirname(curr_folder))
        module = importlib.import_module(f"data.{args.benchmark}.constants")
    except ImportError:
        print(f"Module data.{args.benchmark}.constants not found.")
    tasks_base = module.TASKS
    with open(os.path.join(curr_folder, f"../{args.benchmark}.yaml"), "r") as f:
        tasks_customized = yaml.safe_load(f)
    if args.task not in tasks_customized:
        raise ValueError(f"{args.task} is not found in config_tasks.yaml")
    config = tasks_customized.get(args.task)
    config.update(tasks_base[config["task"]])
    task_file = args.data_dir / args.task / f"{args.subset}.jsonl"
    if args.chunk_amount > 1:
        pred_file = args.save_dir / f"{args.task}-{args.chunk_idx}.jsonl"
    else:
        pred_file = args.save_dir / f"{args.task}.jsonl"
    print(f"Predict {args.task} \nfrom {task_file}\nto {pred_file}")
    pred_file.parent.mkdir(parents=True, exist_ok=True)
    if os.path.exists(pred_file):
        pred_index = [sample["index"] for sample in load_data(pred_file)]
        data = [
            sample
            for sample in load_data(task_file)
            if sample["index"] not in pred_index
        ]
    else:
        data = load_data(task_file)
    dtype = paddle.float16 if args.dtype == "fp16" else paddle.bfloat16
    llm = get_llm(
        args.model_name,
        args.max_len,
        config["tokens_to_generate"],
        args.attn_type,
        dtype,
        args.device,
        synthetic_len=args.synthetic_len,
    )
    
    if paddle.device.is_compiled_with_cuda():
        device = "gpu"
    else:
        device = "cpu"
    # compress model using sqattn
    if args.attn_type == "SQAttn":
        bits_alloc = compress_model(llm.llm, llm.llm.tokenizer, device, args)
        llm.llm = process_model(llm.llm, args.attn_type, bits_alloc, args)
    else:
        bits_alloc = None
    
    outputs_parallel = [{} for _ in range(len(data))]
    with open(pred_file, "at", encoding="utf-8", buffering=1) as fout:
        for idx, data_point in tqdm(enumerate(data), total=len(data)):
            outputs_parallel[idx] = get_output(llm, 
                                               outputs_parallel, 
                                               idx, 
                                               data_point["index"], 
                                               data_point["input"], 
                                               data_point["outputs"], 
                                               data_point.get("others", {}), 
                                               data_point.get("truncation", -1), 
                                               data_point.get("length", -1))
            if len(outputs_parallel[idx]) > 0:
                fout.write(json.dumps(outputs_parallel[idx]) + "\n")
    print(f"Used time: {round((time.time() - start_time) / 60, 1)} minutes")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_dir",
        type=Path,
        required=True,
        help="path to load the dataset jsonl files",
    )
    parser.add_argument(
        "--save_dir",
        type=Path,
        required=True,
        help="path to save the prediction jsonl files",
    )
    parser.add_argument(
        "--benchmark", type=str, default="synthetic", help="Options: [synthetic]"
    )
    parser.add_argument(
        "--task", type=str, required=True, help="Options: tasks in benchmark"
    )
    parser.add_argument(
        "--subset", type=str, default="validation", help="Options: validation or test"
    )
    parser.add_argument(
        "--chunk_idx", type=int, default=0, help="index of current split chunk"
    )
    parser.add_argument(
        "--chunk_amount", type=int, default=1, help="size of split chunk"
    )
    parser.add_argument(
        "--server_type", default="nemo", action=ServerAction, choices=SERVER_TYPES
    )
    parser.add_argument("--server_host", type=str, default="127.0.0.1")
    parser.add_argument("--server_port", type=str, default="5000")
    parser.add_argument("--ssh_server", type=str)
    parser.add_argument("--ssh_key_path", type=str)
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument(
        "--attn_type",
        type=str,
        default="Full_Flash_Attn",
        choices=["Full_Flash_Attn", "SQAttn", "SDPA"],
        help="Attention method",
    )
    parser.add_argument("--max_len", type=int, default=128000)
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size")
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--dtype", type=str, default="fp16", choices=["fp16", "bf16"])
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top_k", type=int, default=32)
    parser.add_argument("--top_p", type=float, default=1.0)
    parser.add_argument("--random_seed", type=int, default=0)
    parser.add_argument("--sliding_window_size", type=int)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--synthetic_len", type=int, required=True)

    # SQAttn
    parser.add_argument(
        "--calib_dataset",
        type=str,
        default="pileval",
        choices=["wikitext2", "ptb", "c4", "mix", "pileval", "gsm8k", "longbench"],
        help="Where to extract calibration data from.",
    )
    parser.add_argument(
        "--nsamples", type=int, default=128, help="Number of calibration data samples."
    )
    parser.add_argument("--seqlen", type=int, default=2048, help="seqlen.")
    parser.add_argument("--eval_ppl", action="store_true", help="eval wikitext2 ppl")
    parser.add_argument("--eval_gsm8k", action="store_true", help="eval gsm8k")
    parser.add_argument("--multigpu", action="store_true", help="use multigpu for eval")
    parser.add_argument("--tasks", default=None, type=str)
    parser.add_argument("--num_fewshot", type=int, default=0)
    parser.add_argument("--limit", type=int, default=-1)
    parser.add_argument(
        "--dynamic_shape", action="store_true", help="use dynamic shape for flashattn"
    )
    parser.add_argument(
        "--quant", action="store_true", help="use causal mask for flashattn"
    )
    parser.add_argument(
        "--qk_qtype",
        type=str,
        default="int",
        choices=["int", "e4m3", "e5m2"],
        help="quantization type for qk",
    )
    parser.add_argument(
        "--v_qtype",
        type=str,
        default="int",
        choices=["int", "e4m3", "e5m2"],
        help="quantization type for v",
    )
    parser.add_argument(
        "--bit8_thres", type=float, default=0.95, help="threshold for bit8"
    )
    parser.add_argument(
        "--bit4_thres", type=float, default=0.98, help="threshold for bit4"
    )
    parser.add_argument(
        "--sample_output_file",
        type=str,
        default="gsm8k_res.jsonl",
        help="file for saving sample output",
    )
    parser.add_argument(
        "--use_relative_distance", action="store_true", help="use relative window"
    )
    parser.add_argument(
        "--max_new_tokens", type=int, default=128, help="max new tokens"
    )
    
    args = parser.parse_args()
    print(args)
    if args.server_type == "hf" or args.server_type == "gemini":
        args.threads = 1
    seed_everything(2025)
    main(args)
