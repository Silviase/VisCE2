#!/bin/bash

# 必須の引数を取得
DATASET_ID=$1
MODEL_ID=$2
PROMPT_ID=$3
DEVICES=$4

# シフトして引数を4つ進める
shift 4

# 残りの引数をすべて取得
EXTRA_ARGS="$@"

# コマンドを実行
CUDA_VISIBLE_DEVICES=$DEVICES python src/eval.py \
                --dataset_id=$DATASET_ID \
                --model_id=$MODEL_ID \
                --prompt_path=prompts/$PROMPT_ID.txt \
                --source_model_id=$MODEL_ID \
                $EXTRA_ARGS