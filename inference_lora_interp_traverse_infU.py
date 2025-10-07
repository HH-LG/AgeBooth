import json
import random
import os
from PIL import Image
import sys
from InfiniteYou.test_tools import (
    get_infU, 
    test_mix_examples,
    add_args,
)

from config_utils import (
    parse_args,
)
from svdlora_pytorch.utils import (
    get_lora_weights,
    split_lora_weights,
    initialize_svdlora_layer,
)
import svdlora_pytorch.svdlora as svdlora
from diffusers.models.lora import LoRACompatibleLinear


args = parse_args()
pipe = get_infU(args)

lora_weights = get_lora_weights(args.lora_name_or_path)
lora_weights_2 = get_lora_weights(args.lora_name_or_path_2)
lora_weights_dict_1_a, lora_weights_dict_1_b = split_lora_weights(
    lora_weights
)
lora_weights_dict_2_a, lora_weights_dict_2_b =  split_lora_weights(
    lora_weights_2
)
def change_weights(pipe, method="direct_linear"):
    for name in lora_weights.keys():
        parent_module = pipe
        name = '.'.join(name.split(".")[:-2])
        def get_next(current_module, n:str):
            if n.isdigit():
                return current_module[int(n)]
            else:
                return getattr(current_module, n)
        def set_next(current_module, n:str, value):
            if n.isdigit():
                current_module[int(n)] = value
            else:
                setattr(current_module, n, value)

        names = name.split('.')
        for n in names[:-1]:
            parent_module = get_next(parent_module, n)
        last_module = get_next(parent_module, names[-1])
        # Parse the attention module.
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

        set_next(parent_module, names[-1], get_compatible(last_module))
        last_module = get_next(parent_module, names[-1])
        
        last_module.set_lora_layer(
            initialize_svdlora_layer(
                **kwargs,
                part=name,
                interp_kv=False,
                in_features=last_module.in_features,
                out_features=last_module.out_features,
            )
        )
    return pipe

pipe.pipe = change_weights(pipe.pipe, args.method)

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
    alpha_range = [round(x * args.alpha_step, 3) for x in range(int(0 / args.alpha_step), int(1 / args.alpha_step) + 1)]
    print("Sampling example:", example)
    seed = example[-4]
    for sample_idx in range(args.num_samples):
        for alpha_step in alpha_range:
            if not 0 <= alpha_step <= 1:
                continue

            print(f"Generating with alpha = {alpha_step}")
            svdlora.alpha = alpha_step

            # 在输出文件夹下创建一个子文件夹，文件夹名为 alpha 的值
            example[-4] = seed + sample_idx
            alpha_folder = os.path.join(args.output_folder, str(alpha_step))
            os.makedirs(alpha_folder, exist_ok=True)

            image_list, used_seed_list, used_scale_list = test_mix_examples(
                pipe, examples=[example], alpha=alpha_step, return_scale=True
            )


            image, used_seed, used_scale = image_list[0], used_seed_list[0], used_scale_list[0]
            output_filename = f"{name}_{used_scale:.2f}_{used_seed}.png"
            output_path = os.path.join(alpha_folder, output_filename)
            Image.fromarray(image).save(output_path)
            print(f"Successfully generated: {output_path}")

if __name__ == "__main__":
    run()
