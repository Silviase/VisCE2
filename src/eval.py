from data.capeval_dataset import get_dataset
from models.generator import get_generator
from utils.prompter import Prompter
import os
import argparse
import pandas as pd
from tqdm import tqdm
tqdm.pandas()

class Evaluator:
    def __init__(self, 
                 dataset_id: str, 
                 model_id:str, 
                 prompt_cfg:dict,) -> None:
        
        self.dataset_id = dataset_id
        self.model_id = model_id
        self.prompt_id = prompt_cfg['prompt_path'].split('/')[-1].split('.')[0]
        
        # load hf dataset
        self.dataset = get_dataset(self.dataset_id, -1)
        self.generator = get_generator(self.model_id)
        self.prompter = Prompter(**prompt_cfg)

    def query(self, example):
        example['prompt'] = self.prompter.format(example)
        example['scores'][self.model_id] = {}
        example['scores'][self.model_id][self.prompt_id] = self.generator.generate(example['image'], example['prompt'])
        return example

    def eval(self) -> None:
        # Conduct the evaluation
        eval_result = self.dataset.map(self.query, batched=False)
        result_file_path = os.path.join(
            'results/eval',
            self.dataset_id,
            self.model_id,
            self.prompt_id,
            'result.parquet'
        )
        
        if not os.path.exists(os.path.dirname(result_file_path)):
            os.makedirs(os.path.dirname(result_file_path))
        
        eval_result_df = pd.DataFrame(eval_result)
        eval_result_df = eval_result_df.drop(columns=['image'])
        eval_result_df.to_parquet(result_file_path)
    


def parse_args(args):
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_id', type=str, required=True)
    parser.add_argument('--model_id', type=str, required=True)
    
    # Debug
    parser.add_argument('--debug', action='store_true')
    
    # Prompt Config
    parser.add_argument('--prompt_path', type=str, required=True)
    parser.add_argument('--source_model_id', type=str)
    parser.add_argument('--use_cand', action='store_true')
    parser.add_argument('--use_refs', action='store_true')
    parser.add_argument('--use_cand_a', action='store_true')
    parser.add_argument('--use_cand_b', action='store_true')
    parser.add_argument('--use_caption', action='store_true')
    parser.add_argument('--use_context', action='store_true')
    parser.add_argument('--use_object', action='store_true')
    parser.add_argument('--use_attribute', action='store_true')
    parser.add_argument('--use_relation', action='store_true')
    
    
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
            source_model_id=args.source_model_id,
            use_cand=args.use_cand,
            use_refs=args.use_refs,
            use_cand_a=args.use_cand_a,
            use_cand_b=args.use_cand_b,
            use_context=args.use_context,
            use_object=args.use_object,
            use_attribute=args.use_attribute,
            use_relation=args.use_relation,
        ),
    )
    evaluator.eval()
    

if __name__ == '__main__':
    import sys
    main(sys.argv[1:])
