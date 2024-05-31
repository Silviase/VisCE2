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

# 5/29
### extract -- thumb -- llava -- (visce / caps / refs)
bash scripts/run_single_eval_simple.sh thumb liuhaotian/llava-v1.5-7b visce 4 --debug --use_cand --use_context 
bash scripts/run_single_eval_simple.sh thumb liuhaotian/llava-v1.5-13b visce 4 --debug --use_cand --use_context 
bash scripts/run_single_eval_simple.sh thumb liuhaotian/llava-v1.5-7b captions 5 --debug --use_cand --use_caption
bash scripts/run_single_eval_simple.sh thumb liuhaotian/llava-v1.5-13b captions 5 --debug --use_cand --use_caption 
bash scripts/run_single_eval_simple.sh thumb liuhaotian/llava-v1.5-7b references 6 --debug --use_cand --use_refs 
bash scripts/run_single_eval_simple.sh thumb liuhaotian/llava-v1.5-13b references 6 --debug --use_cand --use_refs 
### extract -- flickr8k-expert -- llava -- (visce / caps / refs)
bash scripts/run_single_eval_simple.sh flickr8k-expert liuhaotian/llava-v1.5-7b visce 4 --debug --use_cand --use_context 
bash scripts/run_single_eval_simple.sh flickr8k-expert liuhaotian/llava-v1.5-13b visce 5 --debug --use_cand --use_context 
bash scripts/run_single_eval_simple.sh flickr8k-expert liuhaotian/llava-v1.5-7b captions 6 --debug --use_cand --use_caption
bash scripts/run_single_eval_simple.sh flickr8k-expert liuhaotian/llava-v1.5-13b captions 4 --debug --use_cand --use_caption 
bash scripts/run_single_eval_simple.sh flickr8k-expert liuhaotian/llava-v1.5-7b references 5 --debug --use_cand --use_refs 
bash scripts/run_single_eval_simple.sh flickr8k-expert liuhaotian/llava-v1.5-13b references 6 --debug --use_cand --use_refs 

# 5/30 1,4,5,6が空いてる 
### extract -- pascal-50s -- llava
bash scripts/run_single_inference.sh pascal-50s liuhaotian/llava-v1.5-7b visual_context inference 1
bash scripts/run_single_inference.sh pascal-50s liuhaotian/llava-v1.5-13b visual_context inference 1
bash scripts/run_single_inference.sh pascal-50s liuhaotian/llava-v1.5-7b caption inference 4
bash scripts/run_single_inference.sh pascal-50s liuhaotian/llava-v1.5-13b caption inference 4
### extract -- pascal50s -- yi
bash scripts/run_single_inference.sh pascal-50s Yi-VL-6B visual_context inference 5
bash scripts/run_single_inference.sh pascal-50s Yi-VL-6B caption inference 5

### extract -- thumb/flickr/composite/pascal -- llava16-7b -- visual_context/caption
bash scripts/run_single_inference.sh thumb liuhaotian/llava-v1.6-vicuna-7b visual_context inference 6
bash scripts/run_single_inference.sh thumb liuhaotian/llava-v1.6-vicuna-7b caption inference 6
bash scripts/run_single_inference.sh flickr8k-expert liuhaotian/llava-v1.6-vicuna-7b visual_context inference 6
bash scripts/run_single_inference.sh flickr8k-expert liuhaotian/llava-v1.6-vicuna-7b caption inference 6
bash scripts/run_single_inference.sh composite liuhaotian/llava-v1.6-vicuna-7b visual_context inference 1
bash scripts/run_single_inference.sh composite liuhaotian/llava-v1.6-vicuna-7b caption inference 1
bash scripts/run_single_inference.sh pascal-50s liuhaotian/llava-v1.6-vicuna-7b visual_context inference 4
bash scripts/run_single_inference.sh pascal-50s liuhaotian/llava-v1.6-vicuna-7b caption inference 4
# 未着手=================================================================================================




### extract -- thumb -- yi-34b
# bash scripts/run_single_inference.sh thumb Yi-VL-34B visual_context inference 5
# bash scripts/run_single_inference.sh thumb Yi-VL-34B visual_context inference 6

### extract -- f8kcf -- llava
### extract -- f8kcf -- yi
conda activate llava-Next
huggingface-cli login
huggingface-cli whoami
clear
