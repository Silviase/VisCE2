from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from llava.conversation import conv_templates, SeparatorStyle
from llava.model.builder import load_pretrained_model
from llava.mm_utils import process_images, tokenizer_image_token, get_model_name_from_path, KeywordsStoppingCriteria
from llava.utils import disable_torch_init
from PIL import Image
import torch
from .generator import Generator, get_generator

class LlavaV16(Generator):

    def _load(self, model_id):
        disable_torch_init()
        
        # e.g.) model_id = "liuhaotian/llava-v1.6-7b"
        self.load_4bit = True # if '13b' in model_id else False
        self.conv_mode = "llava_v1"
        print("Conversation mode: {}".format(self.conv_mode))
        self.tokenizer, self.model, self.image_processor, self.context_len = load_pretrained_model(
            model_path=model_id,
            model_base=None,
            model_name=get_model_name_from_path(model_id),
            load_4bit=self.load_4bit,
        )


    def generate(self, image, prompt):
        """
        generate with the prompt and a single image.

        Args:
            image_path (str): The path to the image
            prompt (str): The prompt
        """
        # text
        prompt = DEFAULT_IMAGE_TOKEN + '\n' + prompt
        conv = conv_templates[self.conv_mode].copy()
        conv.append_message(conv.roles[0], prompt)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()
        
        input_ids = tokenizer_image_token(prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0)
        
        # image
        image = [image]
        image_sizes = [x.size for x in image]
        image_tensor = process_images(image, self.image_processor, self.model.config).to(self.model.device, dtype=torch.float16)
        
        # stopping criteria
        stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
        keywords = [stop_str]
        stopping_criteria = KeywordsStoppingCriteria(keywords, self.tokenizer, input_ids)
        
        # inference
        with torch.inference_mode():
            output_ids = self.model.generate(
                input_ids,
                images=image_tensor,
                image_sizes=image_sizes,
                do_sample=False,
                max_new_tokens=256,
                use_cache=True,
                stopping_criteria=[stopping_criteria],
            )

        outputs = self.tokenizer.decode(output_ids[0]).strip()
        conv.messages[-1][-1] = outputs
        
        # delete <s> and </s>
        outputs = outputs.replace('<s>', '').replace('</s>', '')
        
        return outputs


if __name__ == "__main__":
    img = Image.open("/nas64/silviase/Project/prj-blink-ja/BLINK_Benchmark/assets/sample_neko.jpg")
    prompts = [
        "Describe the image in detail.",
        "Evaluate the caption is appropriate from 0 to 10 point scale. caption: A cat is sitting on the floor.",
    ]
    
    model = get_generator('liuhaotian/llava-v1.6-vicuna-7b')
    
    for prompt in prompts:
        print(model.generate(img, prompt))
        
    model = get_generator("liuhaotian/llava-v1.6-vicuna-13b")
    
    for prompt in prompts:
        print(model.generate(img, prompt))