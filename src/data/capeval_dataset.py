from datasets import load_dataset

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
