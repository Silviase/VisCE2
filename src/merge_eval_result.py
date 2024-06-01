import os
import argparse
import pandas as pd
from tqdm import tqdm
from datasets import Dataset

def merge_dicts(dict1, dict2):
    """
    Recursively merge two dictionaries.
    """
    for key in dict2:
        if key in dict1:
            if isinstance(dict1[key], dict) and isinstance(dict2[key], dict):
                merge_dicts(dict1[key], dict2[key])
            else:
                # Override the value in dict1 with the value in dict2
                dict1[key] = dict2[key]
        else:
            dict1[key] = dict2[key]
    return dict1


def list_parquet_files(directory):
    score_parquet_files = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            file_path = os.path.join(root, file)
            if file_path.endswith(".parquet"):
                score_parquet_files.append(file_path)
    return score_parquet_files

def parse_args(args):
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset-id', type=str, required=True)
    return parser.parse_args(args)


def main(args):
    args = parse_args(args)
    
    score_dicts = None
    image_paths = None
    
    root_dir = os.path.join('results', 'eval', args.dataset_id)
    score_parquet_files = list_parquet_files(root_dir)
    
    for file in score_parquet_files:
        df = pd.read_parquet(file)
        if score_dicts is None:
            score_dicts = [{} for _ in range(len(df))]
        if image_paths is None:
            image_paths = df['image_path']
        # apply merge_dicts function to merge the dictionaries
        for i, row in tqdm(df.iterrows(), total=len(df)):
            score_dicts[i] = merge_dicts(score_dicts[i], row['scores'])
    
    
    source_df = pd.DataFrame(columns=['image_path', 'scores'])
    source_df['image_path'] = image_paths
    source_df['scores'] = score_dicts
    
    # Save the merged dataframe
    source_df.to_parquet(os.path.join('results', 'eval', f'{args.dataset_id}.parquet'))
    
    print('Merged and saved {} eval files to {}'.format(
        len(score_parquet_files), 
        os.path.join(root_dir, 'result.parquet')))
    
    dataset = Dataset.from_pandas(source_df)
    dataset.push_to_hub('Silviase/CapMetaEval', args.dataset_id)
    
    return 


if __name__ == '__main__':
    import sys
    main(sys.argv[1:])