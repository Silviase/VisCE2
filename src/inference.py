from data.capeval_dataset import get_dataset
from models.generator import get_generator
from utils.prompter import Prompter
import os
import argparse
import pandas as pd
from tqdm import tqdm
tqdm.pandas()

class Inferer:
    def __init__(self, 
                 dataset_id: str, 
                 model_id:str, 
                 prompt_cfg:dict, 
                 result_key='model') -> None:
        
        self.dataset_id = dataset_id
        self.model_id = model_id
        self.prompt_id = prompt_cfg['prompt_path'].split('/')[-1].split('.')[0]
        self.result_key = result_key
        
        # load hf dataset
        self.dataset = get_dataset(self.dataset_id, -1)
        self.generator = get_generator(self.model_id)
        self.prompter = Prompter(**prompt_cfg)


    def inference(self) -> None:
        # Extract unique image_path rows
        original_df = pd.DataFrame(self.dataset)
        unique_paths = original_df.drop_duplicates(subset=['image_path'])[['image_path', 'image']]

        # Define the query function
        def _query(example):
            example['prompt'] = self.prompter.format(example)
            example['result'] = self.generator.generate(example['image'], example['prompt'])
            return example
        
        # Map to the dataset
        extracted_df = unique_paths.progress_apply(_query, axis=1)
        
        # Create row if original_df does not have 
        if self.prompt_id not in original_df.columns:
            original_df[self.prompt_id] = [{} for _ in range(len(original_df))]

        # Add information to original_df
        for _, row in extracted_df.iterrows():
            matching_rows = original_df[original_df['image_path'] == row['image_path']]
            if not matching_rows.empty:
                for orig_index in matching_rows.index:
                    original_df.at[orig_index, self.prompt_id][self.model_id] = row['result']
        
        # drop the image column
        original_df.drop(columns=['image'], inplace=True)
        
        # save the result
        result_file_path = os.path.join(
            'results/inference', 
            self.dataset_id, 
            self.model_id, 
            self.prompt_id,
            'result.parquet'
        )
        
        if not os.path.exists(os.path.dirname(result_file_path)):
            os.makedirs(os.path.dirname(result_file_path))
        
        original_df.to_parquet(result_file_path)
        return 


def parse_args(args):
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_id', type=str, required=True)
    parser.add_argument('--model_id', type=str, required=True)
    parser.add_argument('--result_key', type=str, default='score_model')
    parser.add_argument('--prompt_path', type=str, required=True)
    # Debug
    parser.add_argument('--debug', action='store_true')
    
    return parser.parse_args(args)


def main(args):
    args = parse_args(args)
    
    if args.debug:
        print(args)
    
    inferer = Inferer(
        args.dataset_id, 
        args.model_id, 
        prompt_cfg=dict(
            prompt_path=args.prompt_path,
        ),
        result_key=args.result_key,
    )
    inferer.inference()
    

if __name__ == '__main__':
    import sys
    main(sys.argv[1:])