from .capeval_dataset import CapEvalDataset
from scipy.stats import kendalltau
import json
import os


class Flickr8kExpert(CapEvalDataset):
    def __init__(self, dataset_name, split=-1):
        super().__init__(dataset_name, split=split)

    def metaeval(self, model_id, prompt, eval_results_dir='results/eval', metaeval_results_dir='results/metaeval') -> None:
        """
        Meta-evaluate the {eval_result_path}/{model_id}_{prompt}.json and aggregate the meta-evaluation results to {metaeval_output_dir}/metaeval_result.json
        The meta-evaluation key is [{dataset_name}][{model_id}_{prompt}]
        In Flickr8k-expert, the evaluation metric -> Kendall's tau (b).
        
        Args:
            model_id (str): The model id
            prompt (str): The prompt
            eval_result_dir (str): The path to the evaluation results
            metaeval_result_dir (str): The path to the meta-evaluation results
        
        Returns:
            None
        """
        
        # load result jsonl file
        eval_result_path = f'{eval_results_dir}/{model_id}_{prompt}.jsonl'
        with open(eval_result_path, 'r') as f:
            eval_results = [json.loads(line) for line in f]

        # calculate Kendall's tau (b) using the evaluation results
        # key is 'score_human' and 'score_model'
        score_human = [result['score_human'] for result in eval_results]
        score_model = [result['score_model'] for result in eval_results]
        kendall_tau_b, kendall_tau_b_std = kendalltau(score_human, score_model, variant='b')
        
        # load the meta-evaluation results
        metaeval_results_path = f'{metaeval_results_dir}/metaeval_result.json'
        if os.path.exists(metaeval_results_path):
            with open(metaeval_results_path, 'r') as f:
                metaeval_results = json.load(f)
        else:
            metaeval_results = {}
        
        # update the meta-evaluation results
        if self.dataset_name not in metaeval_results:
            metaeval_results[self.dataset_name] = {}
        metaeval_results[self.dataset_name][f'{model_id}_{prompt}'] = {
            'kendall_tau_b': kendall_tau_b,
            'kendall_tau_b_std': kendall_tau_b_std
        }
        
        return
    

if __name__ == '__main__':
    dataset = Flickr8kExpert()
    print(dataset.data.iloc[0])