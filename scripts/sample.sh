CUDA_VISIBLE_DEVICES=1 python src/eval.py \
    --dataset_id=flickr8k-expert \
    --model_id=liuhaotian/llava-v1.5-7b \
    --prompt_path=prompts/base.txt \
    --split=0 \
    --result_key=model \
    --use_cand \
    --debug