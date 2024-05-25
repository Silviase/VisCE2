from data.capeval_dataset import get_dataset
from models.generator import get_generator
from utils.prompter import Prompter
import os
import argparse

class Evaluator:
    def __init__(self, 
                 dataset_id: str, 
                 model_id:str, 
                 prompt_cfg:dict, 
                 eval_results_dir:str='results/eval',
                 split=-1,
                 mode='eval',
                 result_key='model') -> None:
        
        self.dataset_id = dataset_id
        self.model_id = model_id
        self.prompt_id = prompt_cfg['prompt_path'].split('/')[-1].split('.')[0]
        self.split = split
        self.eval_results_dir = eval_results_dir
        self.mode=mode
        self.result_key=result_key
        
        # load hf dataset
        self.dataset = get_dataset(self.dataset_id, self.split)
        self.generator = get_generator(self.model_id)
        self.prompter = Prompter(**prompt_cfg)


    def inference(self) -> None:
        """
        Inference the model using the prompt and save the results to the result_eval_dir.
        
        Args:
            prompt (str): The prompt
            result_key (str): The key of the result
        
        Returns:
            None
        """
        
        def query(example):
            example['prompt'] = self.prompter.format(example)
            if self.mode == 'extract':
                example[self.result_key] = self.generator.generate(example['image'], example['prompt'])
            elif self.mode == 'eval':
                example['scores'][self.result_key] = self.generator.generate(example['image'], example['prompt'])
            else:
                raise ValueError(f"Invalid mode: {self.mode}")
            return example
        
        # Map the query function to the dataset; not batched (llava15 is not supported for batched inference)
        eval_results = self.dataset.map(query, batched=False)
            
        # save the result
        result_file_path = os.path.join(
            self.eval_results_dir, 
            self.dataset_id, 
            self.model_id, 
            self.prompt_id,
            'result.parquet' if self.split == -1 else f'result_{self.split}.parquet'
        )
        
        if not os.path.exists(os.path.dirname(result_file_path)):
            os.makedirs(os.path.dirname(result_file_path))
        
        # convert hf dataset to parquet file
        eval_results.to_parquet(result_file_path)
        
        return 

def parse_args(args):
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_id', type=str, required=True)
    parser.add_argument('--model_id', type=str, required=True)
    parser.add_argument('--eval_results_dir', type=str, default='results/eval')
    parser.add_argument('--split', type=int, default=-1)
    parser.add_argument('--result_key', type=str, default='score_model')
    parser.add_argument('--mode', type=str, default='eval', choices=['eval', 'extract'])
    
    # Prompter Configurations
    parser.add_argument('--prompt_path', type=str, required=True)
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
        prompt_cfg=dict(
            prompt_path=args.prompt_path,
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
        mode=args.mode,
        result_key=args.result_key,
    )
    evaluator.inference()
    

if __name__ == '__main__':
    import sys
    main(sys.argv[1:])