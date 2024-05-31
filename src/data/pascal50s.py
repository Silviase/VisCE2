from .capeval_dataset import CapEvalDataset, get_dataset
from scipy.stats import pearsonr
import json
import os
import pandas as pd


class Pascal50s(CapEvalDataset):
    def __init__(self, dataset_id, split=-1):
        super().__init__(dataset_id, split=split)

    def metaeval(self, model_id, prompt, eval_results_dir='results/eval', metaeval_results_dir='results/metaeval') -> None:
        """
        Meta-evaluate the {eval_result_path}/{model_id}_{prompt}.json and aggregate the meta-evaluation results to {metaeval_output_dir}/metaeval_result.json
        The meta-evaluation key is [{dataset_id}][{model_id}_{prompt}]
        In Flickr8k-expert, the evaluation metric -> Kendall's tau (b).
        
        Args:
            model_id (str): The model id
            prompt (str): The prompt
            eval_result_dir (str): The path to the evaluation results
            metaeval_result_dir (str): The path to the meta-evaluation results
        
        Returns:
            None
        """
        raise NotImplementedError("Subclass must implement eval method")
    

if __name__ == '__main__':
    pascal = get_dataset('pascal-50s', split=1)
    print(pascal[0])