# from workers.baseworker import *
# import sys

######################## Multi-image application ########################

# class LLaVA(BaseWorker):

#     def init_components(self, config):
#         sys.path.insert(0, '/path/to/LLaVA/packages/')
#         from llava.model.builder import load_pretrained_model
#         from llava.conversation import conv_templates, SeparatorStyle
#         from llava.constants import DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN

#         self.tokenizer, self.model, self.processor, context_len = load_pretrained_model(
#             model_path=config.model_dir,
#             model_base=None,
#             model_name=config.model_dir,
#             device_map='cuda',
#         )
        
#         if getattr(self.model.config, 'mm_use_im_start_end', False):
#             self.single_img_tokens = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN
#         else:
#             self.single_img_tokens = DEFAULT_IMAGE_TOKEN

#         self.conv_temp = conv_templates["llava_llama_2"]
#         stop_str = self.conv_temp.sep if self.conv_temp.sep_style != SeparatorStyle.TWO else self.conv_temp.sep2
#         self.keywords = [stop_str]

#         self.model.eval()

#     def forward(self, questions, image_paths, device, gen_kwargs):
#         from llava.constants import IMAGE_TOKEN_INDEX
#         from llava.mm_utils import process_images, tokenizer_image_token, KeywordsStoppingCriteria

#         answers = []
#         for question,images_path in zip(questions, image_paths):
#             conv = self.conv_temp.copy()

#             # Multi-image
#             image_tensor = process_images(
#                 [Image.open(image_path).convert('RGB') for image_path in images_path],
#                 self.processor, self.model.config
#             ).to(device)

#             question = question.replace('<ImageHere><ImageHere>', '<ImageHere>\n<ImageHere>\n') # NOTE: handle the special cases in CLEVR-Change dataset
#             input_prompt = question.replace('<ImageHere>', self.single_img_tokens)

#             conv.append_message(conv.roles[0], input_prompt)
#             conv.append_message(conv.roles[1], None)
#             prompt = conv.get_prompt()
#             input_ids = tokenizer_image_token(
#                 prompt=prompt, 
#                 tokenizer=self.tokenizer, 
#                 image_token_index=IMAGE_TOKEN_INDEX, 
#                 return_tensors='pt'
#             ).unsqueeze(0).to(device)

#             with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
#                 output_ids = self.model.generate(
#                     input_ids,
#                     images=image_tensor,
#                     use_cache=True,
#                     stopping_criteria=[KeywordsStoppingCriteria(self.keywords, self.tokenizer, input_ids)],
#                     **gen_kwargs
#                 )
#             answer = self.tokenizer.decode(output_ids[0], skip_special_tokens=True).strip()
#             answers.append(answer)

#         return answers

from workers.baseworker import BaseWorker
from transformers import AutoProcessor, LlavaForConditionalGeneration
from PIL import Image
import torch

class LLaVA(BaseWorker):
    def init_components(self, config):
        model_id = config.model_dir  # e.g., 'llava-hf/llava-1.5-7b-hf'

        # Load model and processor
        self.processor = AutoProcessor.from_pretrained(model_id)
        self.model = LlavaForConditionalGeneration.from_pretrained(
            model_id,
            torch_dtype=torch.float16,
        ).to("cuda" if torch.cuda.is_available() else "cpu")

        self.tokenizer = self.processor.tokenizer  # for compatibility
        self.device = self.model.device
        self.model.eval()

    def forward(self, questions, image_paths, device, gen_kwargs):
        """
        :param questions: list of strings
        :param image_paths: list of list of image paths (multi-image per question)
        :param device: 'cuda' or 'cpu'
        :param gen_kwargs: generation settings
        """
        answers = []

        for question, paths in zip(questions, image_paths):
            # Prepare chat-like multimodal prompt
            question = question.replace('<ImageHere>', '')
            conversation = [{
                "role": "user",
                "content": [{"type": "text", "text": question}] +
                           [{"type": "image"} for _ in paths]
            }]
            prompt = self.processor.apply_chat_template(conversation, add_generation_prompt=True)
            # Load and process all images
            images = [Image.open(p).convert("RGB") for p in paths]
            inputs = self.processor(images=images, text=prompt, return_tensors="pt").to(self.device, torch.float16)
            # image_tensor = process_images(
            #     [Image.open(image_path).convert('RGB') for image_path in images_path],
            #     self.processor, self.model.config
            # ).to(device)
            print(question)
            # Generate answer
            with torch.no_grad():
                outputs = self.model.generate(**inputs, max_new_tokens=200, do_sample=False)

            # Decode response (skip special/image tokens)
            decoded = self.processor.tokenizer.decode(outputs[0][2:], skip_special_tokens=True).strip()

            answers.append(decoded)

        return answers