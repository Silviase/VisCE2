from .capeval_dataset import CapEvalDataset, get_dataset, get_hierarchical_from_dict, get_digit_scores
from scipy.stats import kendalltau
import json
import os
import pandas as pd


class Flickr8kExpert(CapEvalDataset):
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
        
        # load result .parquet file (without image version)
        eval_result_path = f'{eval_results_dir}/flickr8k-expert.parquet'
        eval_results = pd.read_parquet(eval_result_path)

        # calculate Kendall's tau (b) using the evaluation results
        # key is 'score_human' and 'score_model'
        score_human = eval_results['scores'].apply(
            lambda x: get_hierarchical_from_dict(x, ['human'])
        )
        score_model = eval_results['scores'].apply(
            lambda x: get_hierarchical_from_dict(x, [model_id, prompt])
        )
        
        # postprocess
        score_human = score_human.apply(get_digit_scores)
        score_model = score_model.apply(get_digit_scores)
        
        kendall_tau_b, std = kendalltau(score_human, score_model, variant='b')
        
        # load the meta-evaluation results
        metaeval_results_path = f'{metaeval_results_dir}/metaeval_result.json'
        if os.path.exists(metaeval_results_path):
            with open(metaeval_results_path, 'r') as f:
                metaeval_results = json.load(f)
        else:
            metaeval_results = {}
        
        # update the meta-evaluation results
        if self.dataset_id not in metaeval_results:
            metaeval_results[self.dataset_id] = {}
        
        if f'{model_id}_{prompt}' not in metaeval_results[self.dataset_id]:
            metaeval_results[self.dataset_id][f'{model_id}_{prompt}'] = {}
        
        metaeval_results[self.dataset_id][f'{model_id}_{prompt}']['kendall_tau_b'] = kendall_tau_b
        metaeval_results[self.dataset_id][f'{model_id}_{prompt}']['std'] = std
        
        # save the meta-evaluation results
        with open(metaeval_results_path, 'w') as f:
            json.dump(metaeval_results, f, indent=4)
        
        return
    

if __name__ == '__main__':
    f8kex = get_dataset('flickr8k-expert', split=1)
    print(f8kex[0])