import numpy as np
import torch
from PIL import Image
import os
from .pipeline_infu_flux import InfUFluxPipeline
# other params
DEFAULT_NEGATIVE_PROMPT = None

@torch.inference_mode()
def mix_run(*args):
    (
        id_image,          # face_image
        prompt,
        prompt_old,
        neg_prompt,
        scale,
        seed, 
        steps,
        H,
        W,
        id_scale,
        alpha,
        pipeline
    ) = args

    seed = int(seed)
    if seed == -1:
        seed = torch.Generator(device="cuda").seed()
    gen = torch.cuda.manual_seed(seed)

    images = pipeline(
        id_image=id_image,
        prompt=prompt,
        prompt_old=prompt_old,
        negative_prompt=neg_prompt,
        num_steps=steps,
        height=H,
        width=W,
        guidance_scale=scale,
        infusenet_conditioning_scale=id_scale,
        alpha=alpha,
        seed=seed,
    )[0]
    pipeline.id_scale = id_scale

    return np.array(images), str(seed)

def test_mix_examples(pipeline, examples=None, alpha=1.0, return_scale=False):
    # 参数初始化（与原始代码保持一致）
    # 设置默认参数
    # H = 1152
    # W = 864
    H = 1024
    W = 1024
    guidance_scale = 3.5
    steps = 30    # 非加速模型默认步数
    # size = (224, 224)

    neg_prompt = DEFAULT_NEGATIVE_PROMPT

    # 遍历所有示例
    output_img_list, used_seed_list, used_scale_list = [], [], []
    for i, example in enumerate(examples):
        # 加载输入图像
        id_image = Image.open(example[1]).convert("RGB")
        
        # 构造输入参数（补充默认值）
        inputs = [
            id_image,          # face_image
            example[0],        # prompt
            example[5],        # prompt_old
            neg_prompt,
            guidance_scale,
            example[2],        # seed
            steps,
            H,
            W,
            example[3],        # id_scale
            alpha,
            pipeline
        ]

        # 执行生成
        output_img, used_seed  = mix_run(*inputs)
        output_img_list.append(output_img)
        used_seed_list.append(used_seed)
        if return_scale:
            used_scale_list.append(pipeline.id_scale)
        
    if return_scale:
        return output_img_list, used_seed_list, used_scale_list
    return output_img_list, used_seed_list

def get_infU(args):
    infu_model_path = os.path.join(args.model_dir, f'infu_flux_{args.infu_flux_version}', args.model_version)
    insightface_root_path = './'
    pipe = InfUFluxPipeline(
        base_model_path=args.base_model_path,
        infu_model_path=infu_model_path,
        insightface_root_path=insightface_root_path,
        infu_flux_version=args.infu_flux_version,
        model_version=args.model_version,
        quantize_8bit=args.quantize_8bit,
        cpu_offload=args.cpu_offload,
    )
    # Load LoRAs (optional)
    lora_dir = os.path.join(args.model_dir, 'supports', 'optional_loras')
    if not os.path.exists(lora_dir): lora_dir = './models/InfiniteYou/supports/optional_loras'
    loras = []
    if args.enable_realism_lora:
        loras.append([os.path.join(lora_dir, 'flux_realism_lora.safetensors'), 'realism', 1.0])
    if args.enable_anti_blur_lora:
        loras.append([os.path.join(lora_dir, 'flux_anti_blur_lora.safetensors'), 'anti_blur', 1.0])
    pipe.load_loras(loras)

    return pipe

def add_args(parser):
    #parser.add_argument('--id_image', default='./assets/examples/man.jpg', help="""input ID image""")
    parser.add_argument('--control_image', default=None, help="""control image [optional]""")
    #parser.add_argument('--out_results_dir', default='./results', help="""output folder""")
    #parser.add_argument('--prompt', default='A man, portrait, cinematic')
    parser.add_argument('--base_model_path', default='black-forest-labs/FLUX.1-dev')
    parser.add_argument('--model_dir', default='ByteDance/InfiniteYou')
    parser.add_argument('--infu_flux_version', default='v1.0', help="""InfiniteYou-FLUX version: currently only v1.0""")
    parser.add_argument('--model_version', default='aes_stage2', help="""model version: aes_stage2 | sim_stage1""")
    #parser.add_argument('--cuda_device', default=0, type=int)
    #parser.add_argument('--seed', default=0, type=int, help="""seed (0 for random)""")
    parser.add_argument('--guidance_scale', default=3.5, type=float)
    # parser.add_argument('--num_steps', default=30, type=int)
    # parser.add_argument('--infusenet_conditioning_scale', default=1.0, type=float)
    # parser.add_argument('--infusenet_guidance_start', default=0.0, type=float)
    # parser.add_argument('--infusenet_guidance_end', default=1.0, type=float)
    # The LoRA options below are entirely optional. Here we provide two examples to facilitate users to try, but they are NOT used in our paper.
    parser.add_argument('--enable_realism_lora', action='store_true')
    parser.add_argument('--enable_anti_blur_lora', action='store_true')
    # Memory reduction options
    parser.add_argument('--quantize_8bit', action='store_true')
    parser.add_argument('--cpu_offload', action='store_true')
    return parser

def mix(lora1_a, lora1_b, lora2_a, lora2_b, alpha, method):
    if method == 'direct_linear':
        res = None
    elif method == 'a_svd_linear':
        pass
    elif method == 'b_svd_linear':
        pass
    elif method == 'svd_linear':
        pass
    elif method == 'full_svd_linear':
        pass

    return res

def lora_decouple(lora1, lora2, suffix=["loraA.weight, loraB.weight"]):
    lora1_a, lora1_b = {}, {}
    lora2_a, lora2_b = {}, {}
    for k in lora1:
        if k.endswith(suffix[0]):
            lora1_a[k] = lora1[k]
            lora2_a[k] = lora2[k]
        elif k.endswith(suffix[1]):
            lora1_b[k] = lora1[k]
            lora2_b[k] = lora2[k]
    return lora1_a, lora1_b, lora2_a, lora2_b

def lora_mix(lora1, lora2, alpha, method):
    lora_final = {}
    lora1_a, lora1_b, lora2_a, lora2_b = lora_decouple(lora1, lora2)
    for k in lora1:
        lora_final[k] = mix(lora1[k], lora2[k], alpha, method)
    return lora_final

def load_lora_into_transformer(lora_state_dict, pipe):
    pipe.load_lora_weights(lora_state_dict, adapter_name = "default")
    pipe.set_adapters("default", adapter_weights=1.0)

