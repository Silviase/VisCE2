# DATASET_ID=$1
# MODEL_ID=$2
# PROMPT_ID=$3
# RESULT_DIR=$4
# DEVICES=$5

# 5/28 
### extract -- (thumb / f8kexp / composite) -- llava -- (context / caption)
bash scripts/run_single_inference.sh thumb liuhaotian/llava-v1.5-7b visual_context inference 5
bash scripts/run_single_inference.sh thumb liuhaotian/llava-v1.5-13b visual_context inference 5
bash scripts/run_single_inference.sh thumb liuhaotian/llava-v1.5-7b caption inference 6
bash scripts/run_single_inference.sh thumb liuhaotian/llava-v1.5-13b caption inference 6
bash scripts/run_single_inference.sh flickr8k-expert liuhaotian/llava-v1.5-7b visual_context inference 1
bash scripts/run_single_inference.sh flickr8k-expert liuhaotian/llava-v1.5-13b visual_context inference 1
bash scripts/run_single_inference.sh flickr8k-expert liuhaotian/llava-v1.5-7b caption inference 4
bash scripts/run_single_inference.sh flickr8k-expert liuhaotian/llava-v1.5-13b caption inference 4
bash scripts/run_single_inference.sh composite liuhaotian/llava-v1.5-7b visual_context inference 1
bash scripts/run_single_inference.sh composite liuhaotian/llava-v1.5-13b visual_context inference 4
bash scripts/run_single_inference.sh composite liuhaotian/llava-v1.5-7b caption inference 5
bash scripts/run_single_inference.sh composite liuhaotian/llava-v1.5-13b caption inference 6
### extract -- (thumb / f8kexp / composite) -- yi-6b -- (context / caption)
bash scripts/run_single_inference.sh thumb Yi-VL-6B visual_context inference 7
bash scripts/run_single_inference.sh thumb Yi-VL-6B caption inference 7
bash scripts/run_single_inference.sh flickr8k-expert Yi-VL-6B visual_context inference 5
bash scripts/run_single_inference.sh flickr8k-expert Yi-VL-6B caption inference 5
bash scripts/run_single_inference.sh composite Yi-VL-6B visual_context inference 6
bash scripts/run_single_inference.sh composite Yi-VL-6B caption inference 6

# 5/29
### eval -- thumb -- llava -- (visce / caps / refs)
bash scripts/run_single_eval_simple.sh thumb liuhaotian/llava-v1.5-7b visce 4 --debug --use_cand --use_context 
bash scripts/run_single_eval_simple.sh thumb liuhaotian/llava-v1.5-13b visce 4 --debug --use_cand --use_context 
bash scripts/run_single_eval_simple.sh thumb liuhaotian/llava-v1.5-7b captions 5 --debug --use_cand --use_caption
bash scripts/run_single_eval_simple.sh thumb liuhaotian/llava-v1.5-13b captions 5 --debug --use_cand --use_caption 
bash scripts/run_single_eval_simple.sh thumb liuhaotian/llava-v1.5-7b references 6 --debug --use_cand --use_refs 
bash scripts/run_single_eval_simple.sh thumb liuhaotian/llava-v1.5-13b references 6 --debug --use_cand --use_refs 
### eval -- flickr8k-expert -- llava -- (visce / caps / refs)
bash scripts/run_single_eval_simple.sh flickr8k-expert liuhaotian/llava-v1.5-7b visce 4 --debug --use_cand --use_context 
bash scripts/run_single_eval_simple.sh flickr8k-expert liuhaotian/llava-v1.5-13b visce 5 --debug --use_cand --use_context 
bash scripts/run_single_eval_simple.sh flickr8k-expert liuhaotian/llava-v1.5-7b captions 6 --debug --use_cand --use_caption
bash scripts/run_single_eval_simple.sh flickr8k-expert liuhaotian/llava-v1.5-13b captions 4 --debug --use_cand --use_caption 
bash scripts/run_single_eval_simple.sh flickr8k-expert liuhaotian/llava-v1.5-7b references 5 --debug --use_cand --use_refs 
bash scripts/run_single_eval_simple.sh flickr8k-expert liuhaotian/llava-v1.5-13b references 6 --debug --use_cand --use_refs 

# 5/30 1,4,5,6が空いてる 
### extract -- pascal-50s -- (llava 1.5 7b / 13b / yi) -- (visce / caps)
bash scripts/run_single_inference.sh pascal-50s liuhaotian/llava-v1.5-7b visual_context inference 1
bash scripts/run_single_inference.sh pascal-50s liuhaotian/llava-v1.5-13b visual_context inference 1
bash scripts/run_single_inference.sh pascal-50s liuhaotian/llava-v1.5-7b caption inference 4
bash scripts/run_single_inference.sh pascal-50s liuhaotian/llava-v1.5-13b caption inference 4
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

# 5/31
### eval -- (thumb / flickr / composite) -- (llava15-7b) -- base
bash scripts/run_single_eval_simple.sh thumb liuhaotian/llava-v1.5-7b base 1 --debug --use_cand 
bash scripts/run_single_eval_simple.sh flickr8k-expert liuhaotian/llava-v1.5-7b base 1 --debug --use_cand 
bash scripts/run_single_eval_simple.sh composite liuhaotian/llava-v1.5-7b base 1 --debug --use_cand 

### eval -- (thumb / flickr / composite) -- (llava16-7b) -- (base / visce / caps / refs)
bash scripts/run_single_eval_simple.sh thumb liuhaotian/llava-v1.6-vicuna-7b base 4 --debug --use_cand 
bash scripts/run_single_eval_simple.sh thumb liuhaotian/llava-v1.6-vicuna-7b visce 4 --debug --use_cand --use_context
bash scripts/run_single_eval_simple.sh thumb liuhaotian/llava-v1.6-vicuna-7b captions 4 --debug --use_cand --use_caption
bash scripts/run_single_eval_simple.sh thumb liuhaotian/llava-v1.6-vicuna-7b references 4 --debug --use_cand --use_refs
bash scripts/run_single_eval_simple.sh flickr8k-expert liuhaotian/llava-v1.6-vicuna-7b base 5 --debug --use_cand
bash scripts/run_single_eval_simple.sh flickr8k-expert liuhaotian/llava-v1.6-vicuna-7b visce 5 --debug --use_cand --use_context
bash scripts/run_single_eval_simple.sh flickr8k-expert liuhaotian/llava-v1.6-vicuna-7b captions 5 --debug --use_cand --use_caption
bash scripts/run_single_eval_simple.sh flickr8k-expert liuhaotian/llava-v1.6-vicuna-7b references 5 --debug --use_cand --use_refs
bash scripts/run_single_eval_simple.sh composite liuhaotian/llava-v1.6-vicuna-7b base 6 --debug --use_cand
bash scripts/run_single_eval_simple.sh composite liuhaotian/llava-v1.6-vicuna-7b visce 6 --debug --use_cand --use_context
bash scripts/run_single_eval_simple.sh composite liuhaotian/llava-v1.6-vicuna-7b captions 6 --debug --use_cand --use_caption
bash scripts/run_single_eval_simple.sh composite liuhaotian/llava-v1.6-vicuna-7b references 6 --debug --use_cand --use_refs

### eval -- composite -- llava15-7b -- (visce / caps / refs)
bash scripts/run_single_eval_simple.sh composite liuhaotian/llava-v1.5-7b visce 1 --debug --use_cand --use_context
bash scripts/run_single_eval_simple.sh composite liuhaotian/llava-v1.5-7b captions 1 --debug --use_cand --use_caption
bash scripts/run_single_eval_simple.sh composite liuhaotian/llava-v1.5-7b references 1 --debug --use_cand --use_refs

### meta-eval -- composite -- llava15-7b -- base
bash scripts/run_metaeval.sh composite liuhaotian/llava-v1.5-7b base

### eval -- (thumb / flickr / composite) -- (llava15-7b llava15-13b llava16-7b) -- (visce2)
bash scripts/run_single_eval_simple.sh thumb liuhaotian/llava-v1.5-7b visce_2 1 --debug --use_cand --use_context
bash scripts/run_single_eval_simple.sh thumb liuhaotian/llava-v1.5-13b visce_2 1 --debug --use_cand --use_context
bash scripts/run_single_eval_simple.sh thumb liuhaotian/llava-v1.6-vicuna-7b visce_2 1 --debug --use_cand --use_context
bash scripts/run_single_eval_simple.sh flickr8k-expert liuhaotian/llava-v1.5-7b visce_2 4 --debug --use_cand --use_context
bash scripts/run_single_eval_simple.sh flickr8k-expert liuhaotian/llava-v1.5-13b visce_2 4 --debug --use_cand --use_context
bash scripts/run_single_eval_simple.sh flickr8k-expert liuhaotian/llava-v1.6-vicuna-7b visce_2 4 --debug --use_cand --use_context
# 未着手=================================================================================================
bash scripts/run_single_eval_simple.sh composite liuhaotian/llava-v1.5-7b visce_2 1 --debug --use_cand --use_context
bash scripts/run_single_eval_simple.sh composite liuhaotian/llava-v1.5-13b visce_2 1 --debug --use_cand --use_context
bash scripts/run_single_eval_simple.sh composite liuhaotian/llava-v1.6-vicuna-7b visce_2 1 --debug --use_cand --use_context

# When finished some evaluations, you need to aggregate the results
bash scripts/run_merge_eval.sh

# When meta-evaluation is needed, you run the following command
bash scripts/run_metaeval.sh thumb liuhaotian/llava-v1.6-vicuna-7b visce_2
bash scripts/run_metaeval.sh flickr8k-expert liuhaotian/llava-v1.5-7b visce_2
bash scripts/run_metaeval.sh flickr8k-expert liuhaotian/llava-v1.5-13b visce_2
bash scripts/run_metaeval.sh flickr8k-expert liuhaotian/llava-v1.6-vicuna-7b visce_2


conda activate llava-Next
huggingface-cli login
huggingface-cli whoami
clear
cd ../../2024_EMNLP_VisCE2p