import torch
import os
from PIL import Image
import json
import numpy as np
import torch

class Flickr8k(torch.utils.data.Dataset):
    def __init__(self, json_file, root='datasets/flickr8k/'):
        self.im_folder = os.path.join(root, 'images')

        with open(os.path.join(root, json_file)) as fp:
            data = json.load(fp)

        self.data = list()
        for i in data:
            for human_judgement in data[i]['human_judgement']:
                if np.isnan(human_judgement['rating']):
                    print('NaN')
                    continue
                d = {
                    'ID': i,
                    'image_path': data[i]['image_path'].split('/')[-1],
                    'references': [' '.join(gt.split()) for gt in data[i]['ground_truth']],
                    'candidate': ' '.join(human_judgement['caption'].split()),
                    'scores': {
                            'human': human_judgement['rating']
                        }
                }
                self.data.append(d)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        id = self.data[idx]['ID']
        image_path = self.data[idx]['image_path']
        candidate = self.data[idx]['candidate']
        references = self.data[idx]['references']
        scores = self.data[idx]['scores']

        return self.get_image(image_path), candidate, references, scores

if __name__ == '__main__':
    import torchvision.transforms as transforms
    from torch.utils.data import DataLoader
    from tqdm import tqdm
    
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])

    dataset = Flickr8k('flickr8k.json')
    
    # Save dataset to a single json file
    with open('preprocess/flickr8k-expert.json', 'w') as fp:
        json.dump(dataset.data, fp, indent=2)
        
    dataset = Flickr8k('crowdflower_flickr8k.json')
    with open('preprocess/flickr8k-cf.json', 'w') as fp:
        json.dump(dataset.data, fp, indent=2)