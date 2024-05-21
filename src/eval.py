from data.capeval_dataset import get_dataset
from models.generator import get_generator
from utils.prompter import Prompter
import os
import json
import argparse
from tqdm import tqdm

class Evaluator:
    def __init__(self, 
                 dataset_id: str, 
                 model_id:str, 
                 prompt_path:str, 
                 prompt_cfg:dict, 
                 eval_results_dir:str='results/eval',
                 eval_results_file_name:str='eval_result',
                 split=-1) -> None:
        self.dataset_id = dataset_id
        self.model_id = model_id
        self.split = split
        self.prompt_path = prompt_path
        self.eval_results_dir = os.path.join(eval_results_dir, self.dataset_id, self.model_id)
        self.eval_results_file_name = eval_results_file_name
        if not os.path.exists(self.eval_results_dir):
            os.makedirs(self.eval_results_dir)
        
        self.dataset = get_dataset(dataset_id, split=split)
        self.generator = get_generator(model_id)
        self.prompter = Prompter(prompt_path, **prompt_cfg)


    def inference(self, result_key='score_model') -> None:
        """
        Inference the model using the prompt and save the results to the result_eval_dir.
        
        Args:
            prompt (str): The prompt
            result_key (str): The key of the result
        
        Returns:
            None
        """
        # inference the model
        for i, row in tqdm(self.dataset.data.iterrows(), total=len(self.dataset.data)):
            image_path = row['image_path']
            prompt = self.prompter.format(row)
            
            escaped_prompt = prompt.replace("\n", "\\n")
            
            # TODO: rowから取り出すところの操作を統一する
            results = {
                'Split': self.split,
                'ID': i,
                'image_path': image_path,
                'prompt': escaped_prompt,
                result_key: self.generator.generate(image_path, prompt),
                'cand': row['caption'],
                'refs': row['references'].split('+++'),
            }
            
            json_line = json.dumps(results)
            
            with open(os.path.join(self.eval_results_dir, f"{self.eval_results_file_name}.jsonl"), 'a') as f:
                f.write(json_line + '\n')

def parse_args(args):
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_id', type=str, required=True)
    parser.add_argument('--model_id', type=str, required=True)
    parser.add_argument('--prompt_path', type=str, required=True)
    parser.add_argument('--split', type=int, default=-1)
    parser.add_argument('--result_key', type=str, default='score_model')
    parser.add_argument('--eval_results_dir', type=str, default='results/eval')
    parser.add_argument('--eval_results_file_name', type=str, default='eval_result')
    
    # Prompter Configurations
    parser.add_argument('--use_cand', action='store_true')
    parser.add_argument('--use_refs', action='store_true')
    parser.add_argument('--use_cand_a', action='store_true')
    parser.add_argument('--use_cand_b', action='store_true')
    parser.add_argument('--use_context', action='store_true')
    parser.add_argument('--use_object', action='store_true')
    parser.add_argument('--use_attribute', action='store_true')
    parser.add_argument('--use_relation', action='store_true')
    
    # Debug
    parser.add_argument('--debug', action='store_true')
    
    return parser.parse_args(args)


def main(args):
    args = parse_args(args)
    
    if args.debug:
        print(args)
    
    evaluator = Evaluator(
        args.dataset_id, 
        args.model_id, 
        args.prompt_path, 
        prompt_cfg=dict(
            use_cand=args.use_cand,
            use_refs=args.use_refs,
            use_cand_a=args.use_cand_a,
            use_cand_b=args.use_cand_b,
            use_context=args.use_context,
            use_object=args.use_object,
            use_attribute=args.use_attribute,
            use_relation=args.use_relation,
        ),
        split=args.split,
    )
    evaluator.inference(result_key=args.result_key)
    

if __name__ == '__main__':
    import sys
    main(sys.argv[1:])