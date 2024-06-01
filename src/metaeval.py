from data.capeval_dataset import get_data_sys
import argparse

class MetaEvaluator:
    def __init__(self, dataset_id, model_id, prompt_id):
        self.dataset_id = dataset_id
        self.model_id = model_id
        self.prompt_id = prompt_id
        print (f"dataset_id: {dataset_id}, model_id: {model_id}, prompt_id: {prompt_id}")
        
    def metaeval(self):
        self.dataset = get_data_sys(self.dataset_id)
        self.dataset.metaeval(self.model_id, self.prompt_id)
        
def parse_args(args):
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset-id', type=str, required=True)
    parser.add_argument('--model-id', type=str, required=True)
    parser.add_argument('--prompt-id', type=str, required=True)
    return parser.parse_args(args)

if __name__ == '__main__':
    import sys
    args = parse_args(sys.argv[1:])
    me = MetaEvaluator(args.dataset_id, args.model_id, args.prompt_id)
    me.metaeval()