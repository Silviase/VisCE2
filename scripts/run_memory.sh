# DATASET_ID=$1
# MODEL_ID=$2
# PROMPT_ID=$3
# MODE=$4
# RESULT_DIR=$5
# RESULT_KEY=$6
# DEVICES=$7
# 5/28 
### thumb -- llava
bash scripts/run_single.sh thumb liuhaotian/llava-v1.5-7b visual_context extract inference ____ 5
bash scripts/run_single.sh thumb liuhaotian/llava-v1.5-13b visual_context extract inference ____ 5
bash scripts/run_single.sh thumb liuhaotian/llava-v1.5-7b caption extract inference ____ 6
bash scripts/run_single.sh thumb liuhaotian/llava-v1.5-13b caption extract inference ____ 6
### f8kexp -- llava
bash scripts/run_single.sh flickr8k-expert liuhaotian/llava-v1.5-7b visual_context extract inference ____ 1
bash scripts/run_single.sh flickr8k-expert liuhaotian/llava-v1.5-13b visual_context extract inference ____ 1
bash scripts/run_single.sh flickr8k-expert liuhaotian/llava-v1.5-7b caption extract inference ____ 4
bash scripts/run_single.sh flickr8k-expert liuhaotian/llava-v1.5-13b caption extract inference ____ 4
### composite -- llava
bash scripts/run_single.sh composite liuhaotian/llava-v1.5-7b visual_context extract inference ____ 1
bash scripts/run_single.sh composite liuhaotian/llava-v1.5-13b visual_context extract inference ____ 4
bash scripts/run_single.sh composite liuhaotian/llava-v1.5-7b caption extract inference ____ 5
bash scripts/run_single.sh composite liuhaotian/llava-v1.5-13b caption extract inference ____ 6
# 未着手=================================================================================================

### f8kcf -- llava
### pascal50s -- llava
### thumb -- yi
### f8kexp -- yi
### composite -- yi
### f8kcf -- yi
### pascal50s -- yi

# conda activate llava16
# clear