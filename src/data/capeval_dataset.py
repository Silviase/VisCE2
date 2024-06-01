from datasets import load_dataset
import re


class CapEvalDataset:
    def __init__(self, dataset_id, split=-1):
        self.dataset_id = dataset_id
        self.split = split
        self._load()
        
    def _load(self):
        self.dataset = load_dataset('Silviase/CapEval', self.dataset_id, split='train')

    def metaeval(self, model_id, prompt, eval_results_dir='results_eval', metaeval_results_dir='results_metaeval') -> None:
        """
        Meta-evaluate the {eval_result_path}/{model_id}_{prompt}.json and aggregate the meta-evaluation results to {metaeval_output_dir}/metaeval_result.json
        The meta-evaluation key is [{dataset_id}][{model_id}_{prompt}]
        
        Args:
            model_id (str): The model id
            prompt (str): The prompt
            eval_result_dir (str): The path to the evaluation results
            metaeval_result_dir (str): The path to the meta-evaluation results
        
        Returns:
            None
        """
        raise NotImplementedError("Subclass must implement eval method")
    
    
def get_dataset(dataset_id, split=-1):
    if dataset_id == 'flickr8k-expert':
        from .flickr8k_expert import Flickr8kExpert
        return Flickr8kExpert(dataset_id, split=split).dataset
    elif dataset_id == 'flickr8k-cf':
        from .flickr8k_cf import Flickr8kCF
        return Flickr8kCF(dataset_id, split=split).dataset
    elif dataset_id == 'composite':
        from .composite import Composite
        return Composite(dataset_id, split=split).dataset
    elif dataset_id == 'thumb':
        from .thumb import THumB
        return THumB(dataset_id, split=split).dataset
    elif dataset_id == 'pascal-50s':
        from .pascal50s import Pascal50s
        return Pascal50s(dataset_id, split=split).dataset
    else:
        raise NotImplementedError("The dataset is not implemented yet.")

def get_data_sys(dataset_id):
    if dataset_id == 'flickr8k-expert':
        from .flickr8k_expert import Flickr8kExpert
        return Flickr8kExpert(dataset_id)
    elif dataset_id == 'flickr8k-cf':
        from .flickr8k_cf import Flickr8kCF
        return Flickr8kCF(dataset_id)
    elif dataset_id == 'composite':
        from .composite import Composite
        return Composite(dataset_id)
    elif dataset_id == 'thumb':
        from .thumb import THumB
        return THumB(dataset_id)
    elif dataset_id == 'pascal-50s':
        from .pascal50s import Pascal50s
        return Pascal50s(dataset_id)
    else:
        raise NotImplementedError("The dataset is not implemented yet.")
    
    
def get_hierarchical_from_dict(dict, key_hierarchy):
    """
    Get the value from the dictionary using the key hierarchy
    
    Args:
        dict (dict): The dictionary
        key_hierarchy (list): The list of keys
    
    Returns:
        The value from the dictionary
    """
    for key in key_hierarchy:
        dict = dict[key]
    return dict

def get_digit_scores(s):
    """
    Get the first [0-100] digit scores from the string
    
    Args:
        string (str): The string
    
    Returns:
        The digit scores
    """
    s = str(s)
    score_str = re.findall(r'[0-9]+', s)
    score_str = score_str[0] if score_str else ''
    # if score_str is not detected, return 0
    if score_str == '':
        print(f"score_str is not detected: {s}")
        return 0
    score_float = float(score_str)
    return score_float