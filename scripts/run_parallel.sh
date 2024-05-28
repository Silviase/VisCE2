DATASET_ID=$1
MODEL_ID=$2
PROMPT_ID=$3
MODE=$4
START=$5
END=$6
BATCH_SIZE=$7
RESULT_DIR=$8
RESULT_KEY=$9
DEVICES=${10}

# F8KEXP_NUM=170
# F8KCF_NUM=479
# CMP_NUM=187

while [ $START -le $END ]; do
    # バッチ内で4つのプロセスを並列に実行
    for ((i=0; i<$BATCH_SIZE; i++)); do
        c=$((START + i))
        if [ $c -le $END ]; then
            (CUDA_VISIBLE_DEVICES=$DEVICES python src/eval.py \
                --dataset_id=$DATASET_ID \
                --model_id=$MODEL_ID \
                --prompt_path=prompts/$PROMPT_ID.txt \
                --eval_results_dir=results/$RESULT_DIR \
                --split=$c \
                --mode=$MODE \
                --result_key=$RESULT_KEY \
                --debug) &
        fi
    done

    # すべてのバックグラウンドジョブが終了するのを待つ
    wait

    # 次のバッチに進む
    START=$((START + BATCH_SIZE))
done

echo "All tasks completed."