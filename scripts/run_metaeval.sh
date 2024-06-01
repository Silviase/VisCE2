# Usage: bash scripts/run_metaeval.sh <dataset_id> <model_id> <prompt_id>
# Example: bash scripts/run_metaeval.sh thumb liuhaotian/llava-v1.5-7b base
dataset_id=$1
model_id=$2
prompt_id=$3
python src/metaeval.py --dataset-id $dataset_id --model-id $model_id --prompt-id $prompt_id