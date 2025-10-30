import gc
import json
from dataclasses import dataclass
from typing import Dict
import types

import numpy as np
import paddle
from loguru import logger
from paddle_utils import *
from sqattn.calibration import get_calib_dataset
from sqattn.window_search import calibrate_layer_windows
from sqattn.model_utils import (batch_layer_infer, get_blocks)
from model_hub import LlamaModel, QwenModel
from tqdm import tqdm
from sqattn.attn_replacer import mp_triton_wrapper

paddle.set_device("gpu" if paddle.device.is_compiled_with_cuda() else "cpu")
device = "gpu" if paddle.device.is_compiled_with_cuda() else "cpu"


@dataclass
class CalibrationConfig:
    """Calibration configuration"""

    use_calibration: bool = True
    energy_threshold: float = 0.95
    save_checkpoints: bool = True
    checkpoint_interval: int = 4
    default_bit8_window: int = 128
    default_bit4_window: int = 256
    default_sink_window: int = 16


def get_layer_inputs(model, sample, args):
    """
    Get inputs to first layer using Catcher mechanism
    """
    inps = []
    layer_kwargs = {}

    if model.__class__.__name__ == "LlavaLlamaModel":
        model.llm(sample)
    elif isinstance(model, (QwenModel, LlamaModel)):
        setattr(model, "capture_layer_inps", inps)
        model.tokenizer.pad_token = model.tokenizer.eos_token
        model.tokenizer.padding_side = "left"
        input_ids = sample
        model.generate(
            attention_type=args.attn_type,
            inputs_ids=input_ids,
            attention_masks=None,
            max_new_length=0,
        )
    else:
        model(sample)
    setattr(model, "capture_layer_inps", None)
    paddle.device.synchronize()
    gc.collect()
    return inps[0], layer_kwargs


def save_checkpoint(layer_idx: int, windows_dict: Dict, bits_alloc: Dict):
    """Save checkpoint for recovery"""
    checkpoint = {
        "layer_idx": layer_idx,
        "windows": {f"{k[0]}_{k[1]}": v for k, v in windows_dict.items()},
        "bits_alloc": bits_alloc,
    }
    path = f"checkpoint_layer_{layer_idx}.json"
    with open(path, "w") as f:
        json.dump(checkpoint, f, indent=2)
    logger.info(f"Checkpoint saved: {path}")


def save_final_results(
    windows_dict: Dict, bits_alloc: Dict, config: CalibrationConfig, output_path: str
):
    """Save final calibration results with statistics"""
    all_bit8 = []
    all_bit4 = []
    for layer_config in bits_alloc.values():
        if isinstance(layer_config, dict):
            all_bit8.append(layer_config.get("bit8", 0))
            all_bit4.append(layer_config.get("bit4", 0))
    results = {
        "configuration": {
            "energy_threshold": config.energy_threshold,
            "use_calibration": config.use_calibration,
        },
        "bits_allocation": bits_alloc,
        "statistics": {
            "bit8": {
                "mean": np.mean(all_bit8) if all_bit8 else 0,
                "std": np.std(all_bit8) if all_bit8 else 0,
                "min": np.min(all_bit8) if all_bit8 else 0,
                "max": np.max(all_bit8) if all_bit8 else 0,
            },
            "bit4": {
                "mean": np.mean(all_bit4) if all_bit4 else 0,
                "std": np.std(all_bit4) if all_bit4 else 0,
                "min": np.min(all_bit4) if all_bit4 else 0,
                "max": np.max(all_bit4) if all_bit4 else 0,
            },
        },
    }
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    logger.info(f"Results saved: {output_path}")


def print_gpu_mem(tag=""):
    device = paddle.device.get_device()
    if device.startswith("gpu"):
        gpu_id = int(device.split(":")[-1])
        alloc = paddle.device.cuda.memory_allocated(gpu_id) / 1024**3
        resv = paddle.device.cuda.memory_reserved(gpu_id) / 1024**3
        print(f"[{tag}] allocated={alloc:.2f} GB, reserved={resv:.2f} GB on {device}")
    else:
        print(f"[{tag}] running on {device}, no GPU memory info.")
    return alloc, resv

def compress_model(model, tokenizer, device, args):
    """
    Compress model with integrated layer-by-layer calibration
    """
    layers = get_blocks(model)
    logger.info(f"Starting compression with {len(layers)} layers")
    logger.info("Phase 1: Preparing calibration data...")
    raw_samples, _ = get_calib_dataset(
        data=args.calib_dataset,
        model=model,
        tokenizer=tokenizer,
        n_samples=args.nsamples,
        seq_len=args.seqlen,
        device=device,
        args=args,
    )
    if not isinstance(raw_samples, list):
        raw_samples = [raw_samples]
    max_len = max([sample.shape[1] for sample in raw_samples])
    samples_inps, samples_layer_kwargs = [], []

    for i, sample in enumerate(raw_samples):
        sample = sample[:, :5000] 
        inps, layer_kwargs = get_layer_inputs(model, sample, args)
        
        samples_inps.append(inps)
        samples_layer_kwargs.append(layer_kwargs)
        del inps, layer_kwargs
        paddle.device.cuda.empty_cache()

    del raw_samples
    paddle.device.cuda.empty_cache()
    gc.collect()

    logger.info("Phase 2: Processing layers...")
    bits_alloc = {}
    
    for layer_idx in tqdm(range(len(layers)), desc="Compressing"):
        layer = layers[layer_idx]
        if layer_idx not in [0, len(layers) - 1]:
            bit8_window, bit4_window, head_windows = calibrate_layer_windows(
                layer, layer_idx, samples_inps, samples_layer_kwargs, max_len, args, model
            )
            bit8_windows, bit4_windows = [], []
            for hw in head_windows:
                bit8_windows.append(hw["bit8_relative"]) if 'bit8_relative' in hw else bit8_windows.append(hw["bit8"])
                bit4_windows.append(hw["bit4_relative"]) if 'bit4_relative' in hw else bit4_windows.append(hw["bit4"])
            bits_alloc[layer_idx] = {
                "bit8": bit8_windows,
                "bit4": bit4_windows,
                "sink": 256,
            }
            layer.prefill_attention = types.MethodType(
                mp_triton_wrapper(layer_idx, args=args,
                                bit8_window_sizes=bit8_windows,
                                bit4_window_sizes=bit4_windows,
                                sink_window_size=256),
                layer
            )
            model.layers[layer_idx] = layer
            
        samples_inps = batch_layer_infer(
            layer, samples_inps, samples_layer_kwargs, args, model, layer_idx
        )
        
        paddle.device.cuda.empty_cache()

    if hasattr(args, "current_attention"):
        del args.current_attention
    paddle.device.cuda.empty_cache()
    logger.info("Compression complete!")
    return bits_alloc
