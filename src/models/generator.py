import torch
import dotenv
import os

class Generator:
    def __init__(self, model_id):
        self._load(model_id)
        self.model_id = model_id
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
    def _load(self, model_id):
        raise NotImplementedError("Subclass must implement _load method")

    def generate(self, image_paths, prompt):
        raise NotImplementedError("Subclass must implement func method")
    
    
def get_generator(model_id):
    if 'liuhaotian/llava-v1.5' in model_id:
        from .llava_v15 import LlavaV15
        return LlavaV15(model_id)
    elif 'liuhaotian/llava-v1.6' in model_id:
        from .llava_v16 import LlavaV16
        return LlavaV16(model_id)
    elif 'Yi-VL' in model_id:
        from .yi_vl import YiVL
        # cannot load from 01-ai/Yi-VL-6B directly
        # Execute `git clone` and save models to $HF_HOME. You can set anywhere you like.
        model_dir = os.path.join(os.getenv("HF_HOME"), model_id)
        return YiVL(model_dir)
    else:
        raise ValueError(f"Model {model_id} not found")

