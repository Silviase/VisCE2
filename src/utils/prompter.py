from .constants import desc_precision, desc_coverage, desc_task
from copy import deepcopy

class Prompter:
    def __init__(self, **kwargs):
        self.prompt_base = self._load_prompt_base(kwargs.get('prompt_path'))
        self.source_model_id = kwargs.get('source_model_id', 'model')
        
        # Use or not (default: False)
        self.use_cand = kwargs.get('use_cand', False)
        self.use_refs = kwargs.get('use_refs', False)
        self.use_cand_a = kwargs.get('use_cand_a', False)
        self.use_cand_b = kwargs.get('use_cand_b', False)
        self.use_caption = kwargs.get('use_caption', False)
        self.use_context = kwargs.get('use_context', False)
        self.use_object = kwargs.get('use_object', False)
        self.use_attribute = kwargs.get('use_attribute', False)
        self.use_relation = kwargs.get('use_relation', False)

    def _load_prompt_base(self, prompt_path):
        with open(prompt_path, 'r') as f:
            prompt_base = f.read()
        return prompt_base
    
    def format(self, datum):
        prompt = deepcopy(self.prompt_base)
        
        # Task description
        prompt = prompt.replace('[task]', desc_task)
        
        # Use or Not
        if self.use_cand:
            prompt = prompt.replace('[cand]', f"Candidate:\n{datum['candidate']}")
        if self.use_refs:
            refs = '\n'.join(datum['references'])
            prompt = prompt.replace('[refs]', f"References:\n{refs}")
        if self.use_cand_a:
            prompt = prompt.replace('[cand_a]', f"A: {datum['cand_a']}")
        if self.use_cand_b:
            prompt = prompt.replace('[cand_b]', f"B: {datum['cand_b']}")
        if self.use_caption:
            prompt = prompt.replace('[caption]', f"Context:\n{datum['caption'][self.source_model_id]}")
        if self.use_context:
            prompt = prompt.replace('[context]', f"Context:\n{datum['visual_context'][self.source_model_id]}")
        if self.use_object:
            prompt = prompt.replace('[object]', f"Object:\n{datum['object']}")
        if self.use_attribute:
            prompt = prompt.replace('[attribute]', f"Attribute:\n{datum['attribute']}")
        if self.use_relation:
            prompt = prompt.replace('[relation]', f"Relation:\n{datum['relation']}")

        # Aspect description
        prompt = prompt.replace('[desc_precision]', desc_precision)
        prompt = prompt.replace('[desc_coverage]', desc_coverage)

        # Answer format
        prompt = prompt.replace('[0-5]', 'Answer on a scale of 0 to 5.')
        prompt = prompt.replace('[0-10]', 'Answer on a scale of 0 to 10.')
        prompt = prompt.replace('[0-100]', 'Answer on a scale of 0 to 100.')
        prompt = prompt.replace('[ab]', 'Answer with A or B.')
        
        # Suffix
        prompt = prompt.replace('[suffix]', 'Answer:')
        
        return prompt


if __name__ == '__main__':
    prompter = Prompter('prompts/base.txt', use_cand=True, use_refs=True)
    from src.data.capeval_dataset import get_dataset
    data = get_dataset('flickr8k-expert', 0).data
    datum = data.iloc[0]
    print('================================')
    print(prompter.format(datum))
    print('================================')
    datum = data.iloc[1]
    print(prompter.format(datum))