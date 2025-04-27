import torch
from PIL import Image
from transformers import AutoModel, AutoTokenizer, AutoConfig
from transformers import AutoProcessor
from torch import nn
from transformers import AutoModel, AutoProcessor, AutoTokenizer, AutoModelForCausalLM
import torchvision.transforms.functional as TVF
from peft import PeftModel


CLIP_PATH = "google/siglip-so400m-patch14-384"
CHECKPOINT_PATH = "joy-caption-alpha-two/cgrkzexw-599808"
CAPTION_TYPE_MAP = {
	"Descriptive": [
		"Write a descriptive caption for this image in a formal tone.",
		"Write a descriptive caption for this image in a formal tone within {word_count} words.",
		"Write a {length} descriptive caption for this image in a formal tone.",
	],
	"Descriptive (Informal)": [
		"Write a descriptive caption for this image in a casual tone.",
		"Write a descriptive caption for this image in a casual tone within {word_count} words.",
		"Write a {length} descriptive caption for this image in a casual tone.",
	],
	"Training Prompt": [
		"Write a stable diffusion prompt for this image.",
		"Write a stable diffusion prompt for this image within {word_count} words.",
		"Write a {length} stable diffusion prompt for this image.",
	],
	"MidJourney": [
		"Write a MidJourney prompt for this image.",
		"Write a MidJourney prompt for this image within {word_count} words.",
		"Write a {length} MidJourney prompt for this image.",
	],
	"Booru tag list": [
		"Write a list of Booru tags for this image.",
		"Write a list of Booru tags for this image within {word_count} words.",
		"Write a {length} list of Booru tags for this image.",
	],
	"Booru-like tag list": [
		"Write a list of Booru-like tags for this image.",
		"Write a list of Booru-like tags for this image within {word_count} words.",
		"Write a {length} list of Booru-like tags for this image.",
	],
	"Art Critic": [
		"Analyze this image like an art critic would with information about its composition, style, symbolism, the use of color, light, any artistic movement it might belong to, etc.",
		"Analyze this image like an art critic would with information about its composition, style, symbolism, the use of color, light, any artistic movement it might belong to, etc. Keep it within {word_count} words.",
		"Analyze this image like an art critic would with information about its composition, style, symbolism, the use of color, light, any artistic movement it might belong to, etc. Keep it {length}.",
	],
	"Product Listing": [
		"Write a caption for this image as though it were a product listing.",
		"Write a caption for this image as though it were a product listing. Keep it under {word_count} words.",
		"Write a {length} caption for this image as though it were a product listing.",
	],
	"Social Media Post": [
		"Write a caption for this image as if it were being used for a social media post.",
		"Write a caption for this image as if it were being used for a social media post. Limit the caption to {word_count} words.",
		"Write a {length} caption for this image as if it were being used for a social media post.",
	],
}



class ImageAdapter(nn.Module):
    def __init__(self, input_features: int, output_features: int, ln1: bool, pos_emb: bool, num_image_tokens: int, deep_extract: bool):
        super().__init__()
        self.deep_extract = deep_extract

        if self.deep_extract:
            input_features = input_features * 5

        self.linear1 = nn.Linear(input_features, output_features)
        self.activation = nn.GELU()
        self.linear2 = nn.Linear(output_features, output_features)
        self.ln1 = nn.Identity() if not ln1 else nn.LayerNorm(input_features)
        self.pos_emb = None if not pos_emb else nn.Parameter(torch.zeros(num_image_tokens, input_features))

        self.other_tokens = nn.Embedding(3, output_features)
        self.other_tokens.weight.data.normal_(mean=0.0, std=0.02)

    def forward(self, vision_outputs: torch.Tensor):
        if self.deep_extract:
            x = torch.concat((
                vision_outputs[-2],
                vision_outputs[3],
                vision_outputs[7],
                vision_outputs[13],
                vision_outputs[20],
            ), dim=-1)
        else:
            x = vision_outputs[-2]

        x = self.ln1(x)

        if self.pos_emb is not None:
            x = x + self.pos_emb

        x = self.linear1(x)
        x = self.activation(x)
        x = self.linear2(x)

        other_tokens = self.other_tokens(torch.tensor([0, 1], device=self.other_tokens.weight.device).expand(x.shape[0], -1))
        x = torch.cat((other_tokens[:, 0:1], x, other_tokens[:, 1:2]), dim=1)

        return x

    def get_eot_embedding(self):
        return self.other_tokens(torch.tensor([2], device=self.other_tokens.weight.device)).squeeze(0)

def filter_background_sentences(prompt: str) -> str:
    """
    删除包含独立单词 background 的片段：
    从上一个标点符号（.,;:!?）之后到下一个标点符号之前的所有文字都去掉。
    """
    import re
    text = prompt

    # 循环处理所有出现的 background
    while True:
        m = re.search(r'\bbackground\b', text, re.IGNORECASE)
        if not m:
            break
        start, end = m.start(), m.end()

        # 找上一个标点
        left = max(text.rfind(p, 0, start) for p in '.,;:!?')
        # 找下一个标点
        rights = [text.find(p, end) for p in '.,;:!?']
        rights = [i for i in rights if i >= 0]
        right = min(rights) if rights else len(text)

        # 保留标点，只删除它们之间的文字
        text = text[:left+1] + text[right:]

    # 一次性清理多余空格和标点
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'\s+([.,;:!?])', r'\1', text)
    text = re.sub(r',\.', '.', text)
    return text.strip()



class ImageAnalyzer_JoyCaption:
    def __init__(self, device='cuda'):
        self.device = device
        self.setup_model()

    def setup_model(self):
        print("Loading CLIP")
        self.clip_processor = AutoProcessor.from_pretrained(CLIP_PATH)
        self.clip_model = AutoModel.from_pretrained(CLIP_PATH).vision_model
        self.clip_model.eval().to(self.device)

        print("Loading tokenizer")
        self.tokenizer = AutoTokenizer.from_pretrained(
            CHECKPOINT_PATH + "/text_model", use_fast=True
        )

        print("Loading LLM")
        # 1. 从 base_model_name_or_path 读取 config
        base = "unsloth/Meta-Llama-3.1-8B-Instruct"
        # 3.1 加载原始模型
        base_model = AutoModelForCausalLM.from_pretrained(
            base,
            torch_dtype=torch.bfloat16,
        )
        base_model.eval().to(self.device)

        # 3.2 挂载 LoRA adapter
        self.text_model = PeftModel.from_pretrained(
            base_model,
            CHECKPOINT_PATH + "/text_model",
            torch_dtype=torch.bfloat16,
        )
        self.text_model.eval().to(self.device)

        print("Loading image adapter")
        self.image_adapter = ImageAdapter(
            self.clip_model.config.hidden_size,
            self.text_model.config.hidden_size,
            False, False, num_image_tokens=38, deep_extract=False
        )
        self.image_adapter.load_state_dict(
            torch.load(CHECKPOINT_PATH + "/image_adapter.pt", map_location="cpu")
        )
        self.image_adapter.eval().to(self.device)

    @torch.no_grad()
    def analyze_image(self, input_image: Image.Image, caption_type='Descriptive', caption_length="any", extra_options=None, name_input=None, custom_prompt=None) -> str:
        torch.cuda.empty_cache()

        length = None if caption_length == "any" else caption_length
        if isinstance(length, str):
            try:
                length = int(length)
            except ValueError:
                pass

        if length is None:
            map_idx = 0
        elif isinstance(length, int):
            map_idx = 1
        elif isinstance(length, str):
            map_idx = 2
        else:
            raise ValueError(f"Invalid caption length: {length}")

        prompt_str = CAPTION_TYPE_MAP[caption_type][map_idx]

        if extra_options is not None and len(extra_options) > 0:
            prompt_str += " " + " ".join(extra_options)

        prompt_str = prompt_str.format(name=name_input, length=caption_length, word_count=caption_length)

        if custom_prompt is not None and custom_prompt.strip() != "":
            prompt_str = custom_prompt.strip()

        # logger.info(f"Using prompt: {prompt_str}")

        image = input_image.resize((384, 384), Image.LANCZOS)
        pixel_values = TVF.pil_to_tensor(image).unsqueeze(0) / 255.0
        pixel_values = TVF.normalize(pixel_values, [0.5], [0.5]).to(self.device)

        with torch.amp.autocast_mode.autocast(self.device):
            vision_outputs = self.clip_model(pixel_values=pixel_values, output_hidden_states=True)
            embedded_images = self.image_adapter(vision_outputs.hidden_states)

        convo = [
            {"role": "system", "content": "You are a helpful image captioner."},
            {"role": "user", "content": prompt_str},
        ]

        convo_string = self.tokenizer.apply_chat_template(convo, tokenize=False, add_generation_prompt=True)
        convo_tokens = self.tokenizer.encode(convo_string, return_tensors="pt", add_special_tokens=False, truncation=False)
        prompt_tokens = self.tokenizer.encode(prompt_str, return_tensors="pt", add_special_tokens=False, truncation=False)
        convo_tokens = convo_tokens.squeeze(0)
        prompt_tokens = prompt_tokens.squeeze(0)

        eot_id_indices = (convo_tokens == self.tokenizer.convert_tokens_to_ids("<|eot_id|>")).nonzero(as_tuple=True)[0].tolist()
        preamble_len = eot_id_indices[1] - prompt_tokens.shape[0]

        embed_layer = self.text_model.get_input_embeddings()
        convo_embeds = embed_layer(convo_tokens.unsqueeze(0).to(self.device))

        input_embeds = torch.cat([
            convo_embeds[:, :preamble_len],
            embedded_images.to(dtype=convo_embeds.dtype),
            convo_embeds[:, preamble_len:],
        ], dim=1).to(self.device)

        input_ids = torch.cat([
            convo_tokens[:preamble_len].unsqueeze(0),
            torch.zeros((1, embedded_images.shape[1]), dtype=torch.long),
            convo_tokens[preamble_len:].unsqueeze(0),
        ], dim=1).to(self.device)
        attention_mask = torch.ones_like(input_ids)

        generate_ids = self.text_model.generate(input_ids, inputs_embeds=input_embeds, attention_mask=attention_mask, max_new_tokens=300, do_sample=True)

        generate_ids = generate_ids[:, input_ids.shape[1]:]
        if generate_ids[0][-1] == self.tokenizer.eos_token_id or generate_ids[0][-1] == self.tokenizer.convert_tokens_to_ids("<|eot_id|>"):
            generate_ids = generate_ids[:, :-1]

        caption = self.tokenizer.batch_decode(generate_ids, skip_special_tokens=False, clean_up_tokenization_spaces=False)[0]
        caption.strip()
        return filter_background_sentences(caption)