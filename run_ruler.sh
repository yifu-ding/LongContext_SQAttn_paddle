#!/bin/bash
# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
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

# Root Directories
export LD_PRELOAD=$CONDA_PREFIX/lib/libstdc++.so.6
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH
# export CUDA_LAUNCH_BLOCKING=1

# Ensure Python can import from project root (for paddle_utils.py, etc.)
export PYTHONPATH="./:$PYTHONPATH"

export CUDA_VISIBLE_DEVICES=1
ROOT_DIR="./benchmark/ruler/ruler_eval_result" # the path that stores generated task samples and model predictions.

NUM_SAMPLES=1
MAX_SEQ_LENGTH=131072 # 1048576: 1024k, 131072: 128k
ATTN_TYPE=Full_Flash_Attn
DEVICE=cuda:0
TASK=vt
DTYPE=bf16
BIT8_THRES=0.75
BIT4_THRES=0.80

# Model and Tokenizer
source benchmark/ruler/ruler_config_models.sh
MODEL_NAME=qwen2.5-7b

LOG_FILE=logs/${ATTN_TYPE}_${MAX_SEQ_LENGTH}_${MODEL_NAME}.log

MODEL_CONFIG=$(MODEL_SELECT ${MODEL_NAME})
IFS=":" read MODEL_NAME MODEL_TEMPLATE_TYPE MODEL_FRAMEWORK TOKENIZER_PATH TOKENIZER_TYPE <<< "$MODEL_CONFIG"
if [ -z "${MODEL_NAME}" ]; then
    echo "Model: ${MODEL_NAME} is not supported"
    exit 1
fi

# Benchmark and Tasks
source benchmark/ruler/ruler_config_tasks.sh
BENCHMARK=synthetic
declare -n TASKS=$BENCHMARK
if [ -z "${TASKS}" ]; then
    echo "Benchmark: ${BENCHMARK} is not supported"
    exit 1
fi

RESULTS_DIR="${ROOT_DIR}/${MODEL_NAME}/${BENCHMARK}/${MAX_SEQ_LENGTH}/${ATTN_TYPE}"
DATA_DIR="${RESULTS_DIR}/data"
PRED_DIR="${RESULTS_DIR}/pred"
mkdir -p ${DATA_DIR}
mkdir -p ${PRED_DIR}

python -u benchmark/ruler/data/prepare.py \
    --save_dir ${DATA_DIR} \
    --benchmark ${BENCHMARK} \
    --task ${TASK} \
    --tokenizer_path ${TOKENIZER_PATH} \
    --tokenizer_type ${TOKENIZER_TYPE} \
    --max_seq_length ${MAX_SEQ_LENGTH} \
    --model_template_type ${MODEL_TEMPLATE_TYPE} \
    --num_samples ${NUM_SAMPLES} \
    ${REMOVE_NEWLINE_TAB}

python -u benchmark/ruler/pred/call_api.py \
    --model_name ${MODEL_NAME} \
    --attn_type ${ATTN_TYPE} \
    --max_len ${MAX_SEQ_LENGTH} \
    --batch_size 1 \
    --data_dir ${DATA_DIR} \
    --save_dir ${PRED_DIR} \
    --benchmark ${BENCHMARK} \
    --task ${TASK} \
    --dtype ${DTYPE} \
    --server_type ${MODEL_FRAMEWORK} \
    --device ${DEVICE} \
    --synthetic_len ${MAX_SEQ_LENGTH} \
    --calib_dataset longbench \
    --qk_qtype int \
    --v_qtype int \
    --eval_ppl \
    --quant \
    --bit8_thres ${BIT8_THRES} \
    --bit4_thres ${BIT4_THRES} \
    --use_relative_distance \
    2>&1 | tee ${LOG_FILE}

python -u benchmark/ruler/eval/evaluate.py \
    --data_dir ${PRED_DIR} \
    --benchmark ${BENCHMARK}