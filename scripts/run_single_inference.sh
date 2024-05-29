DATASET_ID=$1
MODEL_ID=$2
PROMPT_ID=$3
RESULT_KEY=$4
DEVICES=$5

CUDA_VISIBLE_DEVICES=$DEVICES python src/inference.py \
                --dataset_id=$DATASET_ID \
                --model_id=$MODEL_ID \
                --prompt_path=prompts/$PROMPT_ID.txt \
                --result_key=$RESULT_KEY \
                --debug