# 06/01

### extract -- (thumb / f8kexp / composite / pascal) -- (llava-15 7b/13b llava-16-vicuna-7b/13b) -- (context / caption)
### can use 0,1,2,3,4,5,6
bash scripts/run_single_inference.sh thumb liuhaotian/llava-v1.5-7b visual_context inference 0
bash scripts/run_single_inference.sh thumb liuhaotian/llava-v1.5-13b visual_context inference 0
bash scripts/run_single_inference.sh thumb liuhaotian/llava-v1.6-vicuna-7b visual_context inference 1
bash scripts/run_single_inference.sh thumb liuhaotian/llava-v1.6-vicuna-13b visual_context inference 1
bash scripts/run_single_inference.sh thumb liuhaotian/llava-v1.5-7b caption inference 2
bash scripts/run_single_inference.sh thumb liuhaotian/llava-v1.5-13b caption inference 2
bash scripts/run_single_inference.sh thumb liuhaotian/llava-v1.6-vicuna-7b caption inference 3
bash scripts/run_single_inference.sh thumb liuhaotian/llava-v1.6-vicuna-13b caption inference 3
bash scripts/run_single_inference.sh flickr8k-expert liuhaotian/llava-v1.5-7b visual_context inference 4
bash scripts/run_single_inference.sh flickr8k-expert liuhaotian/llava-v1.5-13b visual_context inference 4
bash scripts/run_single_inference.sh flickr8k-expert liuhaotian/llava-v1.6-vicuna-7b visual_context inference 5
bash scripts/run_single_inference.sh flickr8k-expert liuhaotian/llava-v1.6-vicuna-13b visual_context inference 5
bash scripts/run_single_inference.sh flickr8k-expert liuhaotian/llava-v1.5-7b caption inference 6
bash scripts/run_single_inference.sh flickr8k-expert liuhaotian/llava-v1.5-13b caption inference 6
# ----　2巡目 ----
bash scripts/run_single_inference.sh flickr8k-expert liuhaotian/llava-v1.6-vicuna-7b caption inference 0
bash scripts/run_single_inference.sh flickr8k-expert liuhaotian/llava-v1.6-vicuna-13b caption inference 0
bash scripts/run_single_inference.sh composite liuhaotian/llava-v1.5-7b visual_context inference 1
bash scripts/run_single_inference.sh composite liuhaotian/llava-v1.5-13b visual_context inference 1
bash scripts/run_single_inference.sh composite liuhaotian/llava-v1.6-vicuna-7b visual_context inference 2
bash scripts/run_single_inference.sh composite liuhaotian/llava-v1.6-vicuna-13b visual_context inference 2
bash scripts/run_single_inference.sh composite liuhaotian/llava-v1.5-7b caption inference 3
bash scripts/run_single_inference.sh composite liuhaotian/llava-v1.5-13b caption inference 3
bash scripts/run_single_inference.sh composite liuhaotian/llava-v1.6-vicuna-7b caption inference 4
bash scripts/run_single_inference.sh composite liuhaotian/llava-v1.6-vicuna-13b caption inference 4
bash scripts/run_single_inference.sh pascal-50s liuhaotian/llava-v1.5-7b visual_context inference 5
bash scripts/run_single_inference.sh pascal-50s liuhaotian/llava-v1.5-13b visual_context inference 5
bash scripts/run_single_inference.sh pascal-50s liuhaotian/llava-v1.6-vicuna-7b visual_context inference 6
bash scripts/run_single_inference.sh pascal-50s liuhaotian/llava-v1.6-vicuna-13b visual_context inference 6
# ----　3巡目 ----
bash scripts/run_single_inference.sh pascal-50s liuhaotian/llava-v1.5-7b caption inference 0
bash scripts/run_single_inference.sh pascal-50s liuhaotian/llava-v1.5-13b caption inference 0
bash scripts/run_single_inference.sh pascal-50s liuhaotian/llava-v1.6-vicuna-7b caption inference 1
bash scripts/run_single_inference.sh pascal-50s liuhaotian/llava-v1.6-vicuna-13b caption inference 1
