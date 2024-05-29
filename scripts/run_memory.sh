# DATASET_ID=$1
# MODEL_ID=$2
# PROMPT_ID=$3
# RESULT_DIR=$4
# DEVICES=$5
# 5/28 
### extract -- thumb -- llava
bash scripts/run_single_inference.sh thumb liuhaotian/llava-v1.5-7b visual_context inference 5
bash scripts/run_single_inference.sh thumb liuhaotian/llava-v1.5-13b visual_context inference 5
bash scripts/run_single_inference.sh thumb liuhaotian/llava-v1.5-7b caption inference 6
bash scripts/run_single_inference.sh thumb liuhaotian/llava-v1.5-13b caption inference 6
### extract -- f8kexp -- llava
bash scripts/run_single_inference.sh flickr8k-expert liuhaotian/llava-v1.5-7b visual_context inference 1
bash scripts/run_single_inference.sh flickr8k-expert liuhaotian/llava-v1.5-13b visual_context inference 1
bash scripts/run_single_inference.sh flickr8k-expert liuhaotian/llava-v1.5-7b caption inference 4
bash scripts/run_single_inference.sh flickr8k-expert liuhaotian/llava-v1.5-13b caption inference 4
### extract -- composite -- llava
bash scripts/run_single_inference.sh composite liuhaotian/llava-v1.5-7b visual_context inference 1
bash scripts/run_single_inference.sh composite liuhaotian/llava-v1.5-13b visual_context inference 4
bash scripts/run_single_inference.sh composite liuhaotian/llava-v1.5-7b caption inference 5
bash scripts/run_single_inference.sh composite liuhaotian/llava-v1.5-13b caption inference 6
### extract -- thumb -- yi-6b
bash scripts/run_single_inference.sh thumb Yi-VL-6B visual_context inference 7
bash scripts/run_single_inference.sh thumb Yi-VL-6B caption inference 7
### extract -- f8kexp -- yi-6b
bash scripts/run_single_inference.sh flickr8k-expert Yi-VL-6B visual_context inference 5
bash scripts/run_single_inference.sh flickr8k-expert Yi-VL-6B caption inference 5
### extract -- composite -- yi-6b
bash scripts/run_single_inference.sh composite Yi-VL-6B visual_context inference 6
bash scripts/run_single_inference.sh composite Yi-VL-6B caption inference 6
# 未着手=================================================================================================

### extract -- thumb -- yi-34b
bash scripts/run_single_inference.sh thumb Yi-VL-34B visual_context inference 5
bash scripts/run_single_inference.sh thumb Yi-VL-34B visual_context inference 6

### extract -- f8kcf -- llava
### extract -- pascal50s -- llava

### extract -- f8kcf -- yi
### extract -- pascal50s -- yi

# conda activate llava16
# clear