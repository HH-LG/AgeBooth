import random
import json
import sys
import os
from PIL import Image
import sys
from PuLID.pipe_diffusers import PuLIDSDXLPipeline1_2
from PuLID.pipeline_tools import Add_On
from PuLID.test_tools import (
    test_mix_examples,
)

from diffusers import DiffusionPipeline, DPMSolverMultistepScheduler
import torch
from svdlora_pytorch.utils import (
    get_lora_weights,
    split_lora_weights,
    initialize_svdlora_layer,
)
import svdlora_pytorch.svdlora as svdlora
from config_utils import parse_args
from diffusers.models.lora import LoRACompatibleLinear

args = parse_args()
device = "cuda" if torch.cuda.is_available() else "cpu"
pipe = PuLIDSDXLPipeline1_2.from_pretrained(
    args.pretrained_model_name_or_path,
    torch_dtype=torch.float16, 
    variant="fp16"
).to(device)
add_on = Add_On(device, pipe.unet)
# add_on.load(os.path.join(args.lora_name_or_path, 'add_on'), rank=16)
add_on.add_to_pipeline(pipe)

lora_weights = get_lora_weights(args.young_lora_path)
lora_weights_2 = get_lora_weights(args.old_lora_path)

lora_weights_dict_1_a, lora_weights_dict_1_b = split_lora_weights(
    lora_weights, suffix=["lora.down.weight", "lora.up.weight"]
)
lora_weights_dict_2_a, lora_weights_dict_2_b =  split_lora_weights(
    lora_weights_2, suffix=["lora.down.weight", "lora.up.weight"]
)

def change_weights(unet, method="direct_linear"):
    for attn_processor_name, attn_processor in unet.attn_processors.items():
        # Parse the attention module.
        attn_module = unet
        for n in attn_processor_name.split(".")[:-1]:
            attn_module = getattr(attn_module, n)
        # Get prepared for svdlora
        attn_name = "unet." + ".".join(attn_processor_name.split(".")[:-1])
        kwargs = {
            "state_dict_1_a": lora_weights_dict_1_a,
            "state_dict_1_b": lora_weights_dict_1_b,
            "state_dict_2_a": lora_weights_dict_2_a,
            "state_dict_2_b": lora_weights_dict_2_b,
            "method": method,
        }
        # Set the `lora_layer` attribute of the attention-related matrices.
        def get_compatible(layer):
            new_layer = LoRACompatibleLinear(
                in_features=layer.in_features,
                out_features=layer.out_features,
                bias=layer.bias is not None,
                device=layer.weight.device,
                dtype=layer.weight.dtype,
            )
            if layer.bias is not None:
                new_layer.bias.data = layer.bias.data.clone().detach()
            new_layer.weight.data = layer.weight.data.clone().detach()
            return new_layer

        attn_module.to_q = get_compatible(attn_module.to_q)
        attn_module.to_k = get_compatible(attn_module.to_k)
        attn_module.to_v = get_compatible(attn_module.to_v)
        attn_module.to_out[0] = get_compatible(attn_module.to_out[0])

        attn_module.to_q.set_lora_layer(
            initialize_svdlora_layer(
                **kwargs,
                part=attn_name + ".to_q",
                interp_kv=False,
                in_features=attn_module.to_q.in_features,
                out_features=attn_module.to_q.out_features,
            )
        )
        attn_module.to_k.set_lora_layer(
            initialize_svdlora_layer(
                **kwargs,
                part=attn_name + ".to_k",
                interp_kv=False,
                in_features=attn_module.to_k.in_features,
                out_features=attn_module.to_k.out_features,
            )
        )
        attn_module.to_v.set_lora_layer(
            initialize_svdlora_layer(
                **kwargs,
                part=attn_name + ".to_v",
                interp_kv=False,
                in_features=attn_module.to_v.in_features,
                out_features=attn_module.to_v.out_features,
            )
        )
        attn_module.to_out[0].set_lora_layer(
            initialize_svdlora_layer(
                **kwargs,
                part=attn_name + ".to_out.0",
                interp_kv=False,
                in_features=attn_module.to_out[0].in_features,
                out_features=attn_module.to_out[0].out_features,
            )
        )
    return unet

pipe.unet = change_weights(pipe.unet, args.method)

# scheduler
pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)

def make_example(file_path: str, gender: str):
    templates = {
        'young_m': 'portrait, realistic, a boy at the age of sbu{}',
        'young_f': 'portrait, realistic, a girl at the age of sbu{}',
        'old_m':   'portrait, realistic, an old man at the age of sbu{}',
        'old_f':   'portrait, realistic, an old woman at the age of sbu{}',
    }

    young_key = f"young_{gender}"
    old_key = f"old_{gender}"

    if young_key not in templates or old_key not in templates:
        raise ValueError(f"无效的 gender: {gender}，只能是 'm' 或 'f'")

    return [
        templates[young_key],
        file_path,
        42, # seed
        0.3,# id scale
        20, # zero
        templates[old_key]
    ]
   
def run():
    example = make_example(args.image_path, args.sex)
    example[0].format(args.prompt)
    example[-1].format(args.prompt)
    example[2] = args.seed
    example[3] = args.id_scale

    def filter(prompt): # 去掉prompt中的逗号空格
        return prompt.replace(',', '').replace(' ', '_')
    name = 'output' if args.prompt == '' else filter(args.prompt)
    # 对每个 example，遍历 alpha
    alpha_range = [round(x * args.alpha_step, 5) for x in range(int(0 / args.alpha_step), int(1 / args.alpha_step) + 1)]

    print("Sampling example:", example)
    seed = example[-4]
    for sample_idx in range(args.num_samples):
        for alpha_step in alpha_range:
            if not 0 <= alpha_step <= 1:
                continue

            print(f"Generating with alpha = {alpha_step}")
            svdlora.alpha = alpha_step

            # 更新 seed
            example[-4] = seed + sample_idx
            # 在输出文件夹下创建一个子文件夹，文件夹名为 alpha 的值
            alpha_folder = os.path.join(args.output_folder, str(alpha_step))
            os.makedirs(alpha_folder, exist_ok=True)

            try:
                image_list, used_seed_list, used_scale_list = test_mix_examples(
                    pipe, examples=[example], alpha=alpha_step, return_scale=True
                )
            except Exception as e:
                print(f"Error generating image with alpha = {alpha_step}: {e}")
                continue

            image, used_seed, used_scale = image_list[0], used_seed_list[0], used_scale_list[0]
            output_filename = f"{name}_{used_scale:.2f}_{used_seed}.png"
            output_path = os.path.join(alpha_folder, output_filename)

            Image.fromarray(image).save(output_path)
            print(f"Successfully generated: {output_path}")


if __name__ == "__main__":
    run()
