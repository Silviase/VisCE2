import os
import argparse
import pandas as pd
from tqdm import tqdm


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
    caption_parquet_files = []
    context_parquet_files = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            file_path = os.path.join(root, file)
            if file_path.endswith(".parquet"):
                if 'caption' in file_path:
                    caption_parquet_files.append(file_path)
                if 'context' in file_path:
                    context_parquet_files.append(file_path)
    return caption_parquet_files, context_parquet_files

def parse_args(args):
    parser = argparse.ArgumentParser()
    parser.add_argument('--result-dir', type=str, required=True)
    return parser.parse_args(args)


def main(args):
    args = parse_args(args)
    
    caption_dicts, context_dicts = None, None
    image_paths = None
    
    print(args.result_dir)
    caption_parquet_files, context_parquet_files = list_parquet_files(args.result_dir)
    
    
    for file in caption_parquet_files:
        df = pd.read_parquet(file)
        if caption_dicts is None:
            caption_dicts = [{} for _ in range(len(df))]
        if image_paths is None:
            image_paths = df['image_path']
        # apply merge_dicts function to merge the dictionaries
        for i, row in tqdm(df.iterrows(), total=len(df)):
            caption_dicts[i] = merge_dicts(caption_dicts[i], row['caption'])
    
    for file in context_parquet_files:
        df = pd.read_parquet(file)
        if context_dicts is None:
            context_dicts = [{} for _ in range(len(df))]
        # apply merge_dicts function to merge the dictionaries
        for i, row in tqdm(df.iterrows(), total=len(df)):
            context_dicts[i] = merge_dicts(context_dicts[i], row['visual_context'])
    
    
    source_file = pd.DataFrame(columns=['image_path', 'caption', 'visual_context'])
    source_file['image_path'] = image_paths
    source_file['caption'] = caption_dicts
    source_file['visual_context'] = context_dicts
    
    # Save the merged dataframe
    source_file.to_parquet(os.path.join(args.result_dir, 'result.parquet'))
    
    print('Merged and saved {} caption files and {} context files to {}'.format(
        len(caption_parquet_files), 
        len(context_parquet_files), 
        os.path.join(args.result_dir, 'result.parquet')))

if __name__ == '__main__':
    import sys
    main(sys.argv[1:])