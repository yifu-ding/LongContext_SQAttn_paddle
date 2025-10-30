import json
import pickle

import paddle
from paddle_utils import *
from paddlenlp import datasets

def build_chat(tokenizer, prompt):
    messages = [{"role": "user", "content": prompt}]
    prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    return prompt

def get_calib_dataset(
    data="pileval",
    model=None,
    tokenizer=None,
    n_samples=512,
    seq_len=512,
    device="cuda",
    args=None,
):
    if data == "pileval":
        dataset = datasets.load_dataset(
            "mit-han-lab/pile-val-backup", split="validation"
        )
    elif data == "gsm8k":
        return get_calib_dataset_gsm8k(tokenizer, device, args.gsm8k_prompt)
    elif data == "longbench":
        return get_calib_dataset_longbench(model, tokenizer, device, True)
    else:
        raise NotImplementedError
    dataset = dataset.shuffle(seed=42)
    samples = []
    n_run = 0
    for data in dataset:
        line = data["text"]
        line = line.strip()
        line_encoded = tokenizer.encode(line)
        if len(line_encoded) > 512:
            continue
        sample = paddle.tensor([line_encoded])
        if sample.size == 0:
            continue
        samples.append(sample)
        n_run += 1
        if n_run == n_samples:
            break
    samples = paddle.cat(samples, dim=1)
    n_split = samples.shape[1] // seq_len
    samples = [samples[:, i * seq_len : (i + 1) * seq_len] for i in range(n_split)]
    samples = paddle.cat(samples, dim=0)
    samples = samples[0:1]
    return samples, None


def doc_to_text(doc, fewshot_prompt):
    return (
        fewshot_prompt
        + "\nQuestion: "
        + doc["question"]
        + "\nLet's think step by step\n"
    )


def get_calib_dataset_gsm8k(tokenizer=None, device="cuda", gsm8k_prompt=None):
    fewshot_prompt = open(gsm8k_prompt).read()
    config = datasets.DownloadConfig(resume_download=True, max_retries=100)
    dataset = datasets.load_dataset("gsm8k", "main", download_config=config)
    dataset = dataset["train"].select(range(1))
    texts = []
    for doc in dataset:
        context = doc_to_text(doc, fewshot_prompt)
        texts.append(context)
    tokenizer.pad_token = tokenizer.eos_token
    encodings = tokenizer(texts, return_tensors="pt", padding=True, truncation=True)
    input_ids = encodings["input_ids"].to(device)
    attention_mask = encodings["attention_mask"].to(device)
    return input_ids, attention_mask


def get_calib_dataset_longbench(
    model=None, tokenizer=None, device="cuda", e=False, static=True
):
    if static:
        with open("longbench_samples.pkl", "rb") as f:
            samples = []
            loaded_data = pickle.load(f)
            for data in loaded_data:
                tokenized_data = tokenizer(data, truncation=False, return_tensors="pd")
                samples.append(tokenized_data["input_ids"])
            return samples, None
    max_length = model.config.max_position_embeddings - 500
    if e:
        datasets = [
            "qasper",
            "multifieldqa_en",
            "hotpotqa",
            "2wikimqa",
            "gov_report",
            "multi_news",
            "trec",
            "triviaqa",
            "samsum",
            "passage_count",
            "passage_retrieval_en",
            "lcc",
            "repobench-p",
        ]
    else:
        datasets = [
            "narrativeqa",
            "qasper",
            "multifieldqa_en",
            "multifieldqa_zh",
            "hotpotqa",
            "2wikimqa",
            "musique",
            "dureader",
            "gov_report",
            "qmsum",
            "multi_news",
            "vcsum",
            "trec",
            "triviaqa",
            "samsum",
            "lsht",
            "passage_count",
            "passage_retrieval_en",
            "passage_retrieval_zh",
            "lcc",
            "repobench-p",
        ]
    dataset2prompt = json.load(
        open("sparse_quant_attn/eval/longbench/config/dataset2prompt.json", "r")
    )
    dataset2maxlen = json.load(
        open("sparse_quant_attn/eval/longbench/config/dataset2maxlen.json", "r")
    )
    import pdb; pdb.set_trace()
    samples = []
    model.cuda()
    for dataset in datasets:
        if e:
            data = datasets.load_dataset(
                "THUDM/LongBench", f"{dataset}_e", split="test"
            )
        else:
            data = datasets.load_dataset("THUDM/LongBench", dataset, split="test")
        data = data.select(range(1))
        data = [data_sample for data_sample in data][0]
        prompt_format = dataset2prompt[dataset]
        max_gen = dataset2maxlen[dataset]
        prompt = prompt_format.format(**data)
        tokenized_prompt = tokenizer(
            prompt, truncation=False, return_tensor="pd"
        ).input_ids
        if len(tokenized_prompt) > max_length:
            half = int(max_length / 2)
            prompt = tokenizer.decode(
                tokenized_prompt[:half], skip_special_tokens=True
            ) + tokenizer.decode(tokenized_prompt[-half:], skip_special_tokens=True)
        if dataset not in ["trec", "triviaqa", "samsum", "lsht", "lcc", "repobench-p"]:
            prompt = build_chat(tokenizer, prompt)
        input = tokenizer(prompt, truncation=False, return_tensors="pt").to(device)
        context_length = input.input_ids.shape[-1]
        if dataset == "samsum":
            output = model.generate(
                **input,
                max_new_tokens=max_gen,
                num_beams=1,
                do_sample=False,
                temperature=1.0,
                min_length=context_length + 1,
                eos_token_id=[
                    tokenizer.eos_token_id,
                    tokenizer.encode("n", add_special_tokens=False)[-1],
                ],
            )[0].detach()
        else:
            output = model.generate(
                **input,
                max_new_tokens=max_gen,
                num_beams=1,
                do_sample=False,
                temperature=1.0,
            )[0].detach()
        pred = tokenizer.decode(output[context_length:], skip_special_tokens=True)
        samples.append(prompt + pred)
    with open("longbench_samples.pkl", "wb") as f:
        pickle.dump(samples, f)
    model.cpu()
    return samples, None
