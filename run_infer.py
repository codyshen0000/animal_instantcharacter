
import torch
from PIL import Image
import os
from datetime import datetime
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
import numpy as np
from pipeline import InstantCharacterFluxPipeline
from joycaption import ImageAnalyzer_JoyCaption
from transformers import AutoModelForImageSegmentation
from torchvision import transforms
seed = 1234567890
import random
os.environ['PYTHONHASHSEED'] = str(seed)
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


# Step 1 Load base model and adapter
ip_adapter_path = 'ckpt/instantcharacter_ip-adapter.bin'
base_model = 'black-forest-labs/FLUX.1-dev'
image_encoder_path = 'google/siglip-so400m-patch14-384'
image_encoder_2_path = 'facebook/dinov2-giant'
# load mllm captopn model
image_analyzer = ImageAnalyzer_JoyCaption()

pipe = InstantCharacterFluxPipeline.from_pretrained(base_model, torch_dtype=torch.bfloat16)
pipe.to("cuda")
pipe.init_adapter(
    image_encoder_path=image_encoder_path, 
    image_encoder_2_path=image_encoder_2_path, 
    subject_ipadapter_cfg=dict(subject_ip_adapter_path=ip_adapter_path, nb_token=1024), 
)

birefnet_path = 'ZhengPeng7/BiRefNet'
# load matting model
birefnet = AutoModelForImageSegmentation.from_pretrained(birefnet_path, trust_remote_code=True)
birefnet.to('cuda')
birefnet.eval()
birefnet_transform_image = transforms.Compose([
    transforms.Resize((1024, 1024)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

def remove_bkg(subject_image):

    def infer_matting(img_pil):
        input_images = birefnet_transform_image(img_pil).unsqueeze(0).to('cuda')

        with torch.no_grad():
            preds = birefnet(input_images)[-1].sigmoid().cpu()
        pred = preds[0].squeeze()
        pred_pil = transforms.ToPILImage()(pred)
        mask = pred_pil.resize(img_pil.size)
        mask = np.array(mask)
        mask = mask[..., None]
        return mask

    def get_bbox_from_mask(mask, th=128):
        height, width = mask.shape[:2]
        x1, y1, x2, y2 = 0, 0, width - 1, height - 1

        sample = np.max(mask, axis=0)
        for idx in range(width):
            if sample[idx] >= th:
                x1 = idx
                break
        
        sample = np.max(mask[:, ::-1], axis=0)
        for idx in range(width):
            if sample[idx] >= th:
                x2 = width - 1 - idx
                break

        sample = np.max(mask, axis=1)
        for idx in range(height):
            if sample[idx] >= th:
                y1 = idx
                break

        sample = np.max(mask[::-1], axis=1)
        for idx in range(height):
            if sample[idx] >= th:
                y2 = height - 1 - idx
                break

        x1 = np.clip(x1, 0, width-1).round().astype(np.int32)
        y1 = np.clip(y1, 0, height-1).round().astype(np.int32)
        x2 = np.clip(x2, 0, width-1).round().astype(np.int32)
        y2 = np.clip(y2, 0, height-1).round().astype(np.int32)

        return [x1, y1, x2, y2]

    def pad_to_square(image, pad_value = 255, random = False):
        '''
            image: np.array [h, w, 3]
        '''
        H,W = image.shape[0], image.shape[1]
        if H == W:
            return image

        padd = abs(H - W)
        if random:
            padd_1 = int(np.random.randint(0,padd))
        else:
            padd_1 = int(padd / 2)
        padd_2 = padd - padd_1

        if H > W:
            pad_param = ((0,0),(padd_1,padd_2),(0,0))
        else:
            pad_param = ((padd_1,padd_2),(0,0),(0,0))

        image = np.pad(image, pad_param, 'constant', constant_values=pad_value)
        return image

    salient_object_mask = infer_matting(subject_image)[..., 0]
    x1, y1, x2, y2 = get_bbox_from_mask(salient_object_mask)
    subject_image = np.array(subject_image)
    salient_object_mask[salient_object_mask > 128] = 255
    salient_object_mask[salient_object_mask < 128] = 0
    sample_mask = np.concatenate([salient_object_mask[..., None]]*3, axis=2)
    obj_image = sample_mask / 255 * subject_image + (1 - sample_mask / 255) * 255
    crop_obj_image = obj_image[y1:y2, x1:x2]
    crop_pad_obj_image = pad_to_square(crop_obj_image, 255)
    subject_image = Image.fromarray(crop_pad_obj_image.astype(np.uint8))
    return subject_image


# Step 2 Load reference image
ref_image_path = 'assets/cat.png'  # white background
ref_image = Image.open(ref_image_path).convert('RGB')
ref_image = remove_bkg(ref_image)

prompt_tag = image_analyzer.analyze_image(ref_image, custom_prompt='describe the main subject in detail, especially the face, do not include background')
# prompt = filter_background_sentences(prompt)
print('*********prompt:', prompt_tag)


# Step 3 Inference with style
lora_file_path = 'ckpt/Makoto_Shinkai_style.safetensors'
trigger = 'Makoto Shinkai style'
# prompt = "Ghibli-style, a bench, a striking cat with a unique and captivating appearance. It has a dramatic split-colored face: the left side is gray with a green eye, and the right side is cream-colored with a blue or light red-toned eye (appearing almost red due to lighting). The cat has very long, fluffy fur around its ears and face, giving it a wild, lion-like look. It is wearing a light gray sweater and has a gold tag hanging from its collar. "
prompt = prompt_tag + ',the cat is playing the guitar in the street'
print('*********finalprompt:', prompt)
image = pipe.with_style_lora(
    lora_file_path=lora_file_path,
    trigger=trigger,
    prompt=prompt, 
    num_inference_steps=28,
    guidance_scale=3.5,
    subject_image=ref_image,
    subject_scale=0.9,
    height=1152,
    width=768,
    generator=torch.manual_seed(seed),
).images[0]
image.save(f"flux_instantcharacter_style_Makoto_{timestamp}.webp")
