import torch

class Generator:
    def __init__(self, model_id):
        self._load(model_id)
        self.model_id = model_id
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
    def _load(self):
        raise NotImplementedError("Subclass must implement _load method")

    def generate(self, image_paths, prompt):
        raise NotImplementedError("Subclass must implement func method")
    
    
def get_generator(model_id):
    if 'liuhaotian/llava-v1.5' in model_id:
        from .llava_v15 import LlavaV15
        return LlavaV15(model_id)
    else:
        raise ValueError(f"Model {model_id} not found")

