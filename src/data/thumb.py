from .capeval_dataset import CapEvalDataset, get_dataset, get_hierarchical_from_dict, get_digit_scores
from scipy.stats import pearsonr
import json
import os
import pandas as pd


class THumB(CapEvalDataset):
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
        eval_result_path = f'{eval_results_dir}/thumb.parquet'
        eval_results = pd.read_parquet(eval_result_path)

        # calculate Pearson's correlation coefficient using the evaluation results
        # key is 'score_human' and 'score_model'
        score_human_p = eval_results['scores'].apply(
            lambda x: x['human_p']
        )
        score_human_r = eval_results['scores'].apply(
            lambda x: x['human_r']
        )
        score_human_total = eval_results['scores'].apply(
            lambda x: x['human_total']
        )
        score_model = eval_results['scores'].apply(
            lambda x: x[model_id][prompt]
        )
        
        # postprocess
        score_human_p = score_human_p.apply(get_digit_scores)
        score_human_r = score_human_r.apply(get_digit_scores)
        score_human_total = score_human_total.apply(get_digit_scores)
        score_model = score_model.apply(get_digit_scores)
        
        pearson_p, std_p = pearsonr(score_human_p, score_model)
        pearson_r, std_r = pearsonr(score_human_r, score_model)
        pearson_total, std_total = pearsonr(score_human_total, score_model)
        
        
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
        metaeval_results[self.dataset_id][f'{model_id}_{prompt}']['pearson_p'] = pearson_p
        metaeval_results[self.dataset_id][f'{model_id}_{prompt}']['std_p'] = std_p
        metaeval_results[self.dataset_id][f'{model_id}_{prompt}']['pearson_r'] = pearson_r
        metaeval_results[self.dataset_id][f'{model_id}_{prompt}']['std_r'] = std_r
        metaeval_results[self.dataset_id][f'{model_id}_{prompt}']['pearson_total'] = pearson_total
        metaeval_results[self.dataset_id][f'{model_id}_{prompt}']['std_total'] = std_total
        
        # save the meta-evaluation results
        with open(metaeval_results_path, 'w') as f:
            json.dump(metaeval_results, f, indent=4)
        
        return
    

if __name__ == '__main__':
    thumb = get_dataset('thumb', split=1)
    print(thumb[0])