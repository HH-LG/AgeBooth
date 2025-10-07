from typing import Dict, List, Optional, Union, Any, Tuple, Callable


import numpy as np
import torch


from .pulid import attention_processor as attention
from .pulid.utils import resize_numpy_image_long, img2tensor 

from PIL import Image
# other params
DEFAULT_NEGATIVE_PROMPT = (
    # 'flaws in the eyes, flaws in the face, flaws, lowres, non-HDRi, low quality, worst quality,'
    # 'artifacts noise, text, watermark, glitch, deformed, mutated, ugly, disfigured, hands, '
    # 'low resolution, partially rendered objects,  deformed or partially rendered eyes, '
    # 'deformed, deformed eyeballs, cross-eyed,blurry, '
    # 'overexposure, '
    # 'color bleed, '
    # 'color artifacts, '
    # 'too bright, '
    # 'clipping, '
    # 'over-saturation, '
    # 'pure color, '
)

@torch.inference_mode()
def mix_run(*args):
    id_image = args[0]
    supp_images = args[1:4]
    prompt, prompt_old, neg_prompt, scale, seed, steps, H, W, id_scale, num_zero, ortho, alpha, pipeline = args[4:]

    seed = int(seed)
    if seed == -1:
        seed = torch.Generator(device="cuda").seed()
    gen = torch.cuda.manual_seed(seed)

    attention.NUM_ZERO = num_zero
    if ortho == 'v2':
        attention.ORTHO = False
        attention.ORTHO_v2 = True
    elif ortho == 'v1':
        attention.ORTHO = True
        attention.ORTHO_v2 = False
    else:
        attention.ORTHO = False
        attention.ORTHO_v2 = False

    id_images = [id_image] + [photo for photo in supp_images if photo is not None]

    img = pipeline(
        prompt, 
        prompt_old=prompt_old,
        alpha=alpha,
        height=H,
        width=W,
        negative_prompt=neg_prompt, 
        id_scale=id_scale, 
        guidance_scale=scale,
        num_inference_steps=steps, 
        generator=gen,
        id_photos=id_images,
    ).images[0]

    return np.array(img), str(seed)

def test_mix_examples(pipeline, examples=None, alpha=1.0, return_scale=False):
    # 参数初始化（与原始代码保持一致）
    # 设置默认参数
    H = 1024
    W = 1024
    neg_prompt = DEFAULT_NEGATIVE_PROMPT
    scale = 7.0  # 非加速模型默认值
    steps = 25    # 非加速模型默认步数
    ortho = 'v2'  # 使用v2正交模式

    # 遍历所有示例
    output_img_list, used_seed_list, used_scale_list = [], [], []
    for i, example in enumerate(examples):
        # 加载输入图像
        main_img = Image.open(example[1]).convert("RGB")
        main_img = resize_numpy_image_long(np.array(main_img), 1024)
        
        # 构造输入参数（补充默认值）
        inputs = [
            main_img,          # face_image
            None,              # supp_image1
            None,              # supp_image2
            None,              # supp_image3
            example[0],        # prompt
            example[5],        # prompt_old
            neg_prompt,
            scale,
            example[2],        # seed
            steps,
            H,
            W,
            example[3],        # id_scale
            example[4],        # num_zero
            ortho,
            alpha,
            pipeline
        ]

        # 执行生成
        output_img, used_seed  = mix_run(*inputs)
        output_img_list.append(output_img)
        used_seed_list.append(used_seed)
        used_scale_list.append(pipeline.id_scale)
        # 保存结果（可修改保存路径）
        
    if return_scale:
        return output_img_list, used_seed_list, used_scale_list
    return output_img_list, used_seed_list

@torch.inference_mode()
def run(*args):
    id_image = args[0]
    supp_images = args[1:4]
    prompt, neg_prompt, scale, seed, steps, H, W, id_scale, num_zero, ortho, pipeline = args[4:]

    seed = int(seed)
    if seed == -1:
        seed = torch.Generator(device="cpu").seed()
    gen = torch.manual_seed(seed)

    attention.NUM_ZERO = num_zero
    if ortho == 'v2':
        attention.ORTHO = False
        attention.ORTHO_v2 = True
    elif ortho == 'v1':
        attention.ORTHO = True
        attention.ORTHO_v2 = False
    else:
        attention.ORTHO = False
        attention.ORTHO_v2 = False

    id_images = [id_image] + [photo for photo in supp_images if photo is not None]

    img = pipeline(
        prompt, 
        height=H,
        width=W,
        negative_prompt=neg_prompt, 
        id_scale=id_scale, 
        guidance_scale=scale,
        num_inference_steps=steps, 
        generator=gen,
        id_photos=id_images,
    ).images[0]

    return np.array(img), str(seed)

def test_jugger_examples():
    torch.set_grad_enabled(False)

    # default_cfg = 7.0
    # default_steps = 25

    pipeline = PuLIDSDXLPipeline1_1.from_pretrained(
        "models/Juggrnaut-XL-v9",
        torch_dtype=torch.float16, 
        variant="fp16"
    ).to('cuda')
    pipeline.load_adapters()
    # scheduler
    pulid_config = {
        "_class_name": "DPMSolverMultistepScheduler",
        "_diffusers_version": "0.24.0.dev0",
        "beta_start": 0.00085,
        "beta_end": 0.012, 
        "beta_schedule": "scaled_linear",
        "use_karras_sigmas": True,
        "final_sigmas_type": "zero",
    }

    pipeline.scheduler = DPMSolverMultistepScheduler.from_config(pulid_config)
    # other params
    DEFAULT_NEGATIVE_PROMPT = (
        'flaws in the eyes, flaws in the face, flaws, lowres, non-HDRi, low quality, worst quality,'
        'artifacts noise, text, watermark, glitch, deformed, mutated, ugly, disfigured, hands, '
        'low resolution, partially rendered objects,  deformed or partially rendered eyes, '
        'deformed, deformed eyeballs, cross-eyed,blurry'
    )

    jugger_example_inps = [
        [
            'robot,simple robot,robot with glass face,ellipse head robot,(made partially out of glass),hexagonal shapes,ferns growing inside head,butterflies on head,butterflies flying around',
            'example_inputs/hinton.jpeg',
            15022214902832471291,
            0.8,
            20,
        ],
        ['sticker art, 1girl', 'example_inputs/liuyifei.jpg', 42, 0.8, 20],
        [
            '1girl, cute model, Long thick Maxi Skirt, Knit sweater, swept back hair, alluring smile, working at a clothing store, perfect eyes, highly detailed beautiful expressive eyes, detailed eyes, 35mm photograph, film, bokeh, professional, 4k, highly detailed dynamic lighting, photorealistic, 8k, raw, rich, intricate details,',
            'example_inputs/liuyifei.png',
            42,
            0.8,
            20,
        ],
        ['Chinese paper-cut, 1girl', 'example_inputs/liuyifei.png', 42, 0.8, 20],
        ['Studio Ghibli, 1boy', 'example_inputs/hinton.jpeg', 42, 0.8, 20],
        ['1man made of ice sculpture', 'example_inputs/lecun.jpg', 42, 0.8, 20],
        ['portrait of green-skinned shrek, wearing lacoste purple sweater', 'example_inputs/lecun.jpg', 42, 0.8, 20],
        ['1990s Japanese anime, 1girl', 'example_inputs/liuyifei.png', 42, 0.8, 20],
        ['made of little stones, portrait', 'example_inputs/hinton.jpeg', 42, 0.8, 20],
    ]

    
    # 设置默认参数
    H = 1152
    W = 896
    neg_prompt = DEFAULT_NEGATIVE_PROMPT
    scale = 7.0  # 非加速模型默认值
    steps = 25    # 非加速模型默认步数
    ortho = 'v2'  # 使用v2正交模式

    # 遍历所有示例
    for i, example in enumerate(jugger_example_inps):
        # 加载输入图像
        main_img = Image.open(example[1]).convert("RGB")
        main_img = resize_numpy_image_long(np.array(main_img), 1024)
        
        # 构造输入参数（补充默认值）
        inputs = [
            main_img,          # face_image
            None,              # supp_image1
            None,              # supp_image2
            None,              # supp_image3
            example[0],        # prompt
            neg_prompt,
            scale,
            example[2],        # seed
            steps,
            H,
            W,
            example[3],        # id_scale
            example[4],        # num_zero
            ortho,
            pipeline
        ]

        # 执行生成
        output_img, used_seed = run(*inputs)
            
        # 保存结果（可修改保存路径）
        output_path = f"{i}_output_{used_seed}.png"
        Image.fromarray(output_img).save(output_path)
        print(f"Successfully generated: {output_path}")