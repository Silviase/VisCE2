# スタート値
start=0
# エンド値
end=120
# バッチサイズ
batch_size=4

while [ $start -le $end ]; do
    # バッチ内で4つのプロセスを並列に実行
    for ((i=0; i<$batch_size; i++)); do
        c=$((start + i))
        if [ $c -le $end ]; then
        (CUDA_VISIBLE_DEVICES=1 python src/eval.py \
            --dataset_id=flickr8k-expert \
            --model_id=liuhaotian/llava-v1.5-7b \
            --prompt_path=prompts/visual_context.txt \
            --eval_results_dir results/inference \
            --split=$c \
            --mode=extract \
            --result_key=context) &
        fi
    done

    # すべてのバックグラウンドジョブが終了するのを待つ
    wait

    # 次のバッチに進む
    start=$((start + batch_size))
done

echo "All tasks completed."

