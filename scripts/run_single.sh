DATASET_ID=$1
MODEL_ID=$2
PROMPT_ID=$3
MODE=$4
RESULT_DIR=$5
RESULT_KEY=$6
DEVICES=$7

CUDA_VISIBLE_DEVICES=$DEVICES python src/eval.py \
                --dataset_id=$DATASET_ID \
                --model_id=$MODEL_ID \
                --prompt_path=prompts/$PROMPT_ID.txt \
                --eval_results_dir=results/$RESULT_DIR \
                --split=-1 \
                --mode=$MODE \
                --result_key=$RESULT_KEY \
                --debug