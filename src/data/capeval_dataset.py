import os
import pandas as pd

class CapEvalDataset:
    def __init__(self, dataset_id, split=-1):
        self.dataset_id = dataset_id
        self.split = split
        self._load(split=split)
        
    # TODO: Implement by jsonl
    def _load(self, split=-1):
        if split == -1:
            self.data = pd.read_csv(os.path.join(os.getcwd(), f'preprocessed_depr/{self.dataset_id}.tsv'), sep='\t')
        else:
            self.data = pd.read_csv(os.path.join(os.getcwd(), f'preprocessed_depr/{self.dataset_id}/split/{split}.tsv'), sep='\t')

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
        return Flickr8kExpert(dataset_id, split=split)
