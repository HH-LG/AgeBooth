import argparse
# from InfiniteYou.test_tools import (
    # add_args
# )

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

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        default="models/Juggrnaut-XL-v9",
        help="Pretrained model path",
    )

    parser.add_argument(
        "--young_lora_path",
        type=str,
        help="LoRA path",
        default="/home/u9920210112/zsh/workplace/ziplora-pytorch-tochange/ageLoRA/age15/pytorch_lora_weights.safetensors"
    )
    parser.add_argument(
        "--old_lora_path",
        type=str,
        help="LoRA path",
        default="/home/u9920210112/zsh/workplace/ziplora-pytorch-tochange/ageLoRA/age75/pytorch_lora_weights.safetensors"
    )
    parser.add_argument(
        "--image_path",
        type=str,
        help="Path to the input image",
        default="./assets/example.png"
    )

    parser.add_argument(
        "--sex",
        type=str,
        choices=['f', 'm'],
        help="Specify the gender: 'f' for female, 'm' for male",
        default="f"
    )
    parser.add_argument(
        "--output_folder",
        type=str,
        help="Output folder path",
        default="/home/ubuntu/oyzh/my_project/test",
    )
    parser.add_argument(
        "--id_scale",
        type=float,
        help="scale for id embedding",
        default=0.3,
    )
    parser.add_argument(
        "--method",
        type=str,
        help="fusion method to use",
        default='direct_linear',
    )
    parser.add_argument(
        "--prompt",
        type=str,
        help="Prompt for generation",
        default='',
    )
    parser.add_argument(
        "--alpha_step",
        type=float,
        help="Whether to use LoRA or not",
        default=0.1,
    ) 
    parser.add_argument(
        "--num_samples",
        type=int,
        help="Number of samples to generate",
        default=1,
    ) 
    parser.add_argument(
        "--age_range",
        type=int,
        nargs=2,
        help="Age range for generation",
    )

    parser.add_argument('--seed', type=int, default=-1, help='Random seed for generation')
    parser = add_args(parser)

    return parser.parse_args()