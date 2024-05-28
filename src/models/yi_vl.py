from .yi_llava.conversation import conv_templates
from .yi_llava.mm_utils import (
    KeywordsStoppingCriteria,
    expand2square,
    get_model_name_from_path,
    load_pretrained_model,
    tokenizer_image_token,
)
from .yi_llava.model.constants import DEFAULT_IMAGE_TOKEN, IMAGE_TOKEN_INDEX, key_info
from PIL import Image
from .generator import Generator, get_generator
import os
import torch



class YiVL(Generator):

    def _load(self, model_id):
        # Disable the redundant torch default initialization to accelerate model creation.
        setattr(torch.nn.Linear, "reset_parameters", lambda self: None)
        setattr(torch.nn.LayerNorm, "reset_parameters", lambda self: None)
        # e.g.) model_id = "01-ai/Yi-VL-6B"
        self.load_4bit = True # if '13b' in model_id else False
        self.conv_mode = "mm_default"
        # model_path = os.path.expanduser(model_id)
        # key_info["model_path"] = model_path
        model_path = os.path.expanduser(model_id)
        key_info["model_path"] = model_path
        get_model_name_from_path(model_path)
        self.tokenizer, self.model, self.image_processor, self.context_len = load_pretrained_model(
            model_path=model_id,
            load_4bit=self.load_4bit,
        )
        
        print('Completed loading model')


    def generate(self, image, prompt):
        """
        generate with the prompt and a single image.

        Args:
            image_path (str): The path to the image
            prompt (str): The prompt
        """
        
        # text
        qs = prompt
        qs = DEFAULT_IMAGE_TOKEN + "\n" + qs
        conv = conv_templates[self.conv_mode].copy()
        conv.append_message(conv.roles[0], qs)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()

        input_ids = (
            tokenizer_image_token(prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt")
            .unsqueeze(0)
            .cuda()
        )

        if getattr(self.model.config, "image_aspect_ratio", None) == "pad":
            image = expand2square(
                image, tuple(int(x * 255) for x in self.image_processor.image_mean)
            )
        image_tensor = self.image_processor.preprocess(image, return_tensors="pt")[
            "pixel_values"
        ][0]

        stop_str = conv.sep
        keywords = [stop_str]
        stopping_criteria = KeywordsStoppingCriteria(keywords, self.tokenizer, input_ids)
        
        # inference
        with torch.inference_mode():
            output_ids = self.model.generate(
                input_ids,
                images=image_tensor.unsqueeze(0).to(dtype=torch.bfloat16).cuda(),
                do_sample=True,
                max_new_tokens=256,
                stopping_criteria=[stopping_criteria],
            )

        input_token_len = input_ids.shape[1]
        n_diff_input_output = (input_ids != output_ids[:, :input_token_len]).sum().item()
        if n_diff_input_output > 0:
            print(
                f"[Warning] {n_diff_input_output} output_ids are not the same as the input_ids"
            )
        outputs = self.tokenizer.batch_decode(
            output_ids[:, input_token_len:], skip_special_tokens=True
        )[0]
        outputs = outputs.strip()

        if outputs.endswith(stop_str):
            outputs = outputs[: -len(stop_str)]
        outputs = outputs.strip()

        return outputs


if __name__ == "__main__":
    img = Image.open("/nas64/silviase/Project/prj-blink-ja/BLINK_Benchmark/assets/sample_neko.jpg")
    prompts = [
        "Describe the image in detail.",
        "Evaluate the caption is appropriate from 0 to 10 point scale. caption: A cat is sitting on the floor.",
    ]
    
    model = get_generator("Yi-VL-6B")
    
    for prompt in prompts:
        print("=====================================")
        print(prompt)
        print(model.generate(img, prompt))
        print()
        
        