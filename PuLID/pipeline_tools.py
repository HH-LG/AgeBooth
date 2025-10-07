import gc
import os
import safetensors
from typing import Dict, List, Optional, Union, Any, Tuple, Callable


import cv2
import insightface
import numpy as np
import torch
import torch.nn as nn
import onnxruntime
from facexlib.parsing import init_parsing_model
from facexlib.utils.face_restoration_helper import FaceRestoreHelper
from diffusers.models.lora import LoRALinearLayer, LoRACompatibleLinear

from insightface.app import FaceAnalysis
from safetensors.torch import load_file
from torchvision.transforms import InterpolationMode
from torchvision.transforms.functional import normalize, resize

from .eva_clip import create_model_and_transforms
from .eva_clip.constants import OPENAI_DATASET_MEAN, OPENAI_DATASET_STD
from .pulid.encoders_transformer import IDFormer
from .pulid.utils import is_torch2_available
from .pulid import attention_processor as attention
from .pulid.utils import resize_numpy_image_long, img2tensor 

import PIL
from PIL import Image
import torch

from facexlib.genderage import init_genderage_model
from facexlib.detection import init_detection_model



if is_torch2_available():
    from .pulid.attention_processor import AttnProcessor2_0 as AttnProcessor
    from .pulid.attention_processor import IDAttnProcessor2_0 as IDAttnProcessor
else:
    from .pulid.attention_processor import AttnProcessor, IDAttnProcessor

PipelineImageInput = Union[
    np.ndarray,
    torch.FloatTensor,
    List[np.ndarray],
    List[torch.FloatTensor],
]

# other params
DEFAULT_NEGATIVE_PROMPT = (
    'flaws in the eyes, flaws in the face, flaws, lowres, non-HDRi, low quality, worst quality,'
    'artifacts noise, text, watermark, glitch, deformed, mutated, ugly, disfigured, hands, '
    'low resolution, partially rendered objects,  deformed or partially rendered eyes, '
    'deformed, deformed eyeballs, cross-eyed,blurry'
)

jugger_example_inps = [
    [
        'robot in sbu age,simple robot,robot with glass face,ellipse head robot,(made partially out of glass),hexagonal shapes,ferns growing inside head,butterflies on head,butterflies flying around',
        'example_inputs/hinton.jpeg',
        15022214902832471291,
        0.8,
        20,
    ],
    ['sticker art, 1girl in sbu age', 'example_inputs/liuyifei.png', 42, 0.8, 20],
    [
        '1girl in sbu age, cute model, Long thick Maxi Skirt, Knit sweater, swept back hair, alluring smile, working at a clothing store, perfect eyes, highly detailed beautiful expressive eyes, detailed eyes, 35mm photograph, film, bokeh, professional, 4k, highly detailed dynamic lighting, photorealistic, 8k, raw, rich, intricate details,',
        'example_inputs/liuyifei.png',
        42,
        0.8,
        20,
    ],
    ['Chinese paper-cut, 1girl in sbu age', 'example_inputs/liuyifei.png', 42, 0.8, 20],
    ['Studio Ghibli, 1boy in sbu age', 'example_inputs/hinton.jpeg', 42, 0.8, 20],
    ['1man in sbu age made of ice sculpture', 'example_inputs/lecun.jpg', 42, 0.8, 20],
    ['portrait of green-skinned shrek in sbu age, wearing lacoste purple sweater', 'example_inputs/lecun.jpg', 42, 0.8, 20],
    ['1990s Japanese anime, 1girl in sbu age', 'example_inputs/liuyifei.png', 42, 0.8, 20],
    ['made of little stones, portrait in sbu age', 'example_inputs/hinton.jpeg', 42, 0.8, 20],
]

test_examples = [
    [
        'A person in sbu age.',
        'example_inputs/hinton.jpeg',
        15022214902832471291,
        0.8,
        20,
    ],
    ['A person in sbu age, sticker art.', 'example_inputs/liuyifei.png', 42, 0.8, 20],
    [
        'A person in sbu age.',
        'example_inputs/liuyifei.png',
        42,
        0.8,
        20,
    ],
    ['A person swinging in the park in sbu age.', 'example_inputs/liuyifei.png', 42, 0.8, 20],
    ['A person in sbu age, riding a bike.', 'example_inputs/hinton.jpeg', 42, 0.8, 20],
    ['A person in sbu age, portrait.', 'example_inputs/lecun.jpg', 42, 0.8, 20],
    ['A person in sbu age, readding a book.', 'example_inputs/lecun.jpg', 42, 0.8, 20],
    ['A person in sbu age, listening to music.', 'example_inputs/liuyifei.png', 42, 0.8, 20],
    ['A person in sbu age, fishing.', 'example_inputs/hinton.jpeg', 42, 0.8, 20],
]


@torch.inference_mode()
def run(*args):
    id_image = args[0]
    supp_images = args[1:4]
    prompt, neg_prompt, scale, seed, steps, H, W, id_scale, num_zero, ortho, pipeline = args[4:]

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

def test_jugger_examples(pipeline, examples=test_examples):
    # 参数初始化（与原始代码保持一致）
    # 设置默认参数
    H = 1152
    W = 896
    neg_prompt = DEFAULT_NEGATIVE_PROMPT
    scale = 7.0  # 非加速模型默认值
    steps = 25    # 非加速模型默认步数
    ortho = 'v2'  # 使用v2正交模式

    # 遍历所有示例
    output_img_list, used_seed_list = [], []
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
        output_img, used_seed  = run(*inputs)
        output_img_list.append(output_img)
        used_seed_list.append(used_seed)
        # 保存结果（可修改保存路径）
    return output_img_list, used_seed_list

def to_gray(img):
    x = 0.299 * img[:, 0:1] + 0.587 * img[:, 1:2] + 0.114 * img[:, 2:3]
    x = x.repeat(1, 3, 1, 1)
    return x

def get_compatible(layer: nn.Linear):
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

def set_lora_layer(layer: LoRACompatibleLinear, rank):
    layer.set_lora_layer(
        LoRALinearLayer(
            in_features=layer.in_features,
            out_features=layer.out_features,
            rank=rank,
        )
    )
        
def get_and_set_lora_layer(layer: nn.Linear, rank):
    new_layer = get_compatible(layer)
    set_lora_layer(new_layer, rank)
    return new_layer

def hack_unet_attn_layers(unet):
    id_adapter_attn_procs = {}
    for name, _ in unet.attn_processors.items():
        cross_attention_dim = None if name.endswith("attn1.processor") else unet.config.cross_attention_dim
        if name.startswith("mid_block"):
            hidden_size = unet.config.block_out_channels[-1]
        elif name.startswith("up_blocks"):
            block_id = int(name[len("up_blocks.")])
            hidden_size = list(reversed(unet.config.block_out_channels))[block_id]
        elif name.startswith("down_blocks"):
            block_id = int(name[len("down_blocks.")])
            hidden_size = unet.config.block_out_channels[block_id]
        if cross_attention_dim is not None:
            id_adapter_attn_procs[name] = IDAttnProcessor(
                hidden_size=hidden_size,
                cross_attention_dim=cross_attention_dim,
            ).to(unet.device)
        else:
            id_adapter_attn_procs[name] = AttnProcessor()
    unet.set_attn_processor(id_adapter_attn_procs)
    # id_adapter_attn_layers = nn.ModuleList(unet.attn_processors.values()) taiqiangl wudi maojindeliliang
    return nn.ModuleList(unet.attn_processors.values())

class LoRALinearWrapper(nn.Parameter):
    def __new__(cls, weight, rank=8):
        # 初始时 low-rank 更新为零，因此 effective_weight 等于 base_weight
        effective_weight = weight.clone().detach()
        # 创建一个 Parameter 对象
        obj = super().__new__(cls, effective_weight)
        # 保存原始权重，确保其不被更新
        obj.base_weight = weight.clone().detach()
        obj.rank = rank
        # 定义 low-rank 更新的两个可训练矩阵
        obj.lora_A = nn.Parameter(torch.zeros(weight.size(0), rank, device=weight.device))
        obj.lora_B = nn.Parameter(torch.zeros(rank, weight.size(1), device=weight.device))
        # 初始化 low-rank 参数
        nn.init.kaiming_uniform_(obj.lora_A, a=5 ** 0.5)
        nn.init.zeros_(obj.lora_B)
        return obj

    def __init__(self, weight, rank=8):
        # __new__ 中已完成初始化，这里可以不用做额外工作
        pass

    def effective_weight(self):
        # 动态计算有效权重
        return self.base_weight + self.lora_A @ self.lora_B

    def __rmatmul__(self, other):
        # 当执行 other @ self 时，使用有效权重计算结果
        return other @ self.effective_weight()

    def __repr__(self):
        return f"LoRALinearWrapper(effective_weight={self.effective_weight().__repr__()})"

def get_id_embedding(add_on, image_list):
    """
    Args:
        image in image_list: numpy rgb image, range [0, 255]
    """
    id_cond_list = []
    id_vit_hidden_list = []
    for ii, image in enumerate(image_list):
        add_on.face_helper.clean_all()
        image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        # get antelopev2 embedding
        face_info = add_on.app.get(image_bgr)
        if len(face_info) > 0:
            face_info = sorted(
                face_info, key=lambda x: (x['bbox'][2] - x['bbox'][0]) * (x['bbox'][3] - x['bbox'][1])
            )[
                -1
            ]  # only use the maximum face
            id_ante_embedding = face_info['embedding']
        else:
            id_ante_embedding = None

        # using facexlib to detect and align face
        add_on.face_helper.read_image(image_bgr)
        add_on.face_helper.get_face_landmarks_5(only_center_face=True)
        add_on.face_helper.align_warp_face()
        if len(add_on.face_helper.cropped_faces) == 0:
            raise RuntimeError('facexlib align face fail')
        align_face = add_on.face_helper.cropped_faces[0]
        # incase insightface didn't detect face
        if id_ante_embedding is None:
            # print('fail to detect face using insightface, extract embedding on align face')
            id_ante_embedding = add_on.handler_ante.get_feat(align_face)

        id_ante_embedding = torch.from_numpy(id_ante_embedding).to(add_on.device)
        if id_ante_embedding.ndim == 1:
            id_ante_embedding = id_ante_embedding.unsqueeze(0)

        # parsing
        input = img2tensor(align_face, bgr2rgb=True).unsqueeze(0) / 255.0
        input = input.to(add_on.device)
        parsing_out = add_on.face_helper.face_parse(normalize(input, [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]))[
            0
        ]
        parsing_out = parsing_out.argmax(dim=1, keepdim=True)
        bg_label = [0, 16, 18, 7, 8, 9, 14, 15]
        bg = sum(parsing_out == i for i in bg_label).bool()
        white_image = torch.ones_like(input)
        # only keep the face features
        face_features_image = torch.where(bg, white_image, to_gray(input))

        # transform img before sending to eva-clip-vit
        face_features_image = resize(
            face_features_image, add_on.clip_vision_model.image_size, InterpolationMode.BICUBIC
        )
        face_features_image = normalize(face_features_image, add_on.eva_transform_mean, add_on.eva_transform_std)
        id_cond_vit, id_vit_hidden = add_on.clip_vision_model(
            face_features_image, return_all_features=False, return_hidden=True, shuffle=False
        )
        id_cond_vit_norm = torch.norm(id_cond_vit, 2, 1, True)
        id_cond_vit = torch.div(id_cond_vit, id_cond_vit_norm)

        id_cond = torch.cat([id_ante_embedding, id_cond_vit], dim=-1)

        id_cond_list.append(id_cond)
        id_vit_hidden_list.append(id_vit_hidden)

    id_uncond = torch.zeros_like(id_cond_list[0])
    id_vit_hidden_uncond = []
    for layer_idx in range(0, len(id_vit_hidden_list[0])):
        id_vit_hidden_uncond.append(torch.zeros_like(id_vit_hidden_list[0][layer_idx]))

    id_cond = torch.stack(id_cond_list, dim=1)
    id_vit_hidden = id_vit_hidden_list[0]
    for i in range(1, len(image_list)):
        for j, x in enumerate(id_vit_hidden_list[i]):
            id_vit_hidden[j] = torch.cat([id_vit_hidden[j], x], dim=1)
    id_embedding = add_on.id_adapter(id_cond, id_vit_hidden)
    uncond_id_embedding = add_on.id_adapter(id_uncond, id_vit_hidden_uncond)

    # return id_embedding
    return uncond_id_embedding, id_embedding

class Add_On(nn.Module):
    def __init__(self, device, unet):
        super().__init__()
        # 初始化模型
        self.det_net = init_detection_model('yolov8x_person_face', half=False)
        self.genderage_net = init_genderage_model('mivolo_d1', half=False)

        self.device = device
        self.id_adapter_attn_layers = hack_unet_attn_layers(unet)
        self.load_adapters()
        self.is_finetuned = False

    def estimate_age(self, img):
        with torch.no_grad():
            detected_objects = self.det_net.detect_faces(img)
            self.genderage_net.predict(img, detected_objects)
            ages = detected_objects.ages

        if not ages:
            print(f"No faces detected!")
            return

        # 取平均年龄保留一位小数
        avg_age = round(sum(ages) / len(ages), 1)
        return avg_age

    def load_adapters(self):

        # self.hack_unet_attn_layers(self.unet)

        # self.scheduler = DPMSolverMultistepScheduler.from_config(self.scheduler.config)

        # ID adapters
        self.id_adapter = IDFormer().to(self.device)

        # preprocessors
        # face align and parsing
        self.face_helper = FaceRestoreHelper(
            upscale_factor=1,
            face_size=512,
            crop_ratio=(1, 1),
            det_model='retinaface_resnet50',
            save_ext='png',
            device=self.device,
        )
        self.face_helper.face_parse = None
        self.face_helper.face_parse = init_parsing_model(model_name='bisenet', device=self.device)
        # clip-vit backbone
        model, _, _ = create_model_and_transforms('EVA02-CLIP-L-14-336', 'eva_clip', force_custom_clip=True, cache_dir="models/EVA-CLIP")
        model = model.visual
        self.clip_vision_model = model.to(self.device)
        eva_transform_mean = getattr(self.clip_vision_model, 'image_mean', OPENAI_DATASET_MEAN)
        eva_transform_std = getattr(self.clip_vision_model, 'image_std', OPENAI_DATASET_STD)
        if not isinstance(eva_transform_mean, (list, tuple)):
            eva_transform_mean = (eva_transform_mean,) * 3
        if not isinstance(eva_transform_std, (list, tuple)):
            eva_transform_std = (eva_transform_std,) * 3
        self.eva_transform_mean = eva_transform_mean
        self.eva_transform_std = eva_transform_std
        # antelopev2
        # snapshot_download('DIAMONIK7777/antelopev2', local_dir='models/antelopev2')
        sess_options = onnxruntime.SessionOptions()
        sess_options.intra_op_num_threads = 10
        sess_options.inter_op_num_threads = 10
        self.app = FaceAnalysis(
            name='antelopev2', root='.', providers=['CUDAExecutionProvider', 'CPUExecutionProvider'], sess_options=sess_options,
        )
        self.app.prepare(ctx_id=0, det_size=(640, 640))
        self.handler_ante = insightface.model_zoo.get_model('models/antelopev2/glintr100.onnx', sess_options=sess_options,)
        self.handler_ante.prepare(ctx_id=0)

        gc.collect()
        torch.cuda.empty_cache()

        # self.load_pretrain()
    # def load_pretrain(self):
        ckpt_path = 'models/pulid_v1.1.safetensors'
        state_dict = load_file(ckpt_path)
        state_dict_dict = {}
        for k, v in state_dict.items():
            module = k.split('.')[0]
            state_dict_dict.setdefault(module, {})
            new_k = k[len(module) + 1 :]
            state_dict_dict[module][new_k] = v

        for module in state_dict_dict:
            print(f'loading from {module}')
            getattr(self, module).load_state_dict(state_dict_dict[module], strict=True)

    def forward(self, batch_image_list):
        """
        Args:
            image in image_list: numpy rgb image, range [0, 255]
        """
        batch_id_cond_list = None
        batch_vit_hidden_list = None
        for image_list in batch_image_list:
            id_cond_list = []
            id_vit_hidden_list = []
            for ii, image in enumerate(image_list):
                self.face_helper.clean_all()
                image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                # get antelopev2 embedding
                face_info = self.app.get(image_bgr)
                if len(face_info) > 0:
                    face_info = sorted(
                        face_info, key=lambda x: (x['bbox'][2] - x['bbox'][0]) * (x['bbox'][3] - x['bbox'][1])
                    )[
                        -1
                    ]  # only use the maximum face
                    id_ante_embedding = face_info['embedding']
                else:
                    id_ante_embedding = None

                # using facexlib to detect and align face
                self.face_helper.read_image(image_bgr)
                self.face_helper.get_face_landmarks_5(only_center_face=True)
                self.face_helper.align_warp_face()
                if len(self.face_helper.cropped_faces) == 0:
                    raise RuntimeError('facexlib align face fail')
                align_face = self.face_helper.cropped_faces[0]
                # incase insightface didn't detect face
                if id_ante_embedding is None:
                    # print('fail to detect face using insightface, extract embedding on align face')
                    id_ante_embedding = self.handler_ante.get_feat(align_face)

                id_ante_embedding = torch.from_numpy(id_ante_embedding).to(self.device)
                if id_ante_embedding.ndim == 1:
                    id_ante_embedding = id_ante_embedding.unsqueeze(0)

                # parsing
                input = img2tensor(align_face, bgr2rgb=True).unsqueeze(0) / 255.0
                input = input.to(self.device)
                parsing_out = self.face_helper.face_parse(normalize(input, [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]))[
                    0
                ]
                parsing_out = parsing_out.argmax(dim=1, keepdim=True)
                bg_label = [0, 16, 18, 7, 8, 9, 14, 15]
                bg = sum(parsing_out == i for i in bg_label).bool()
                white_image = torch.ones_like(input)
                # only keep the face features
                face_features_image = torch.where(bg, white_image, to_gray(input))

                # transform img before sending to eva-clip-vit
                face_features_image = resize(
                    face_features_image, self.clip_vision_model.image_size, InterpolationMode.BICUBIC
                )
                face_features_image = normalize(face_features_image, self.eva_transform_mean, self.eva_transform_std)
                id_cond_vit, id_vit_hidden = self.clip_vision_model(
                    face_features_image, return_all_features=False, return_hidden=True, shuffle=False
                )
                id_cond_vit_norm = torch.norm(id_cond_vit, 2, 1, True)
                id_cond_vit = torch.div(id_cond_vit, id_cond_vit_norm)

                id_cond = torch.cat([id_ante_embedding, id_cond_vit], dim=-1)

                id_cond_list.append(id_cond)
                id_vit_hidden_list.append(id_vit_hidden)
            
            id_cond = torch.stack(id_cond_list, dim=1)
            id_vit_hidden = id_vit_hidden_list[0]
            for i in range(1, len(image_list)):
                for j, x in enumerate(id_vit_hidden_list[i]):
                    id_vit_hidden[j] = torch.cat([id_vit_hidden[j], x], dim=1)
            if batch_id_cond_list is None:
                batch_id_cond_list = id_cond
                batch_vit_hidden_list = id_vit_hidden
            else:
                batch_id_cond_list = torch.cat([batch_id_cond_list, id_cond], dim=0)
                for i in range(len(batch_vit_hidden_list)):
                    batch_vit_hidden_list[i] = torch.cat([batch_vit_hidden_list[i], id_vit_hidden[i]], dim=0)

        id_embedding = self.id_adapter(batch_id_cond_list, batch_vit_hidden_list)
        return id_embedding
    
    def finetune(self, rank):
        self.is_finetuned = True
        add_on_lora_parameters = []
        for name, module in self.named_modules():
            cur_module = self
            if name.startswith('id_adapter.'):
                if name.endswith('to_q'):
                    for n in name.split('.')[:-1]:
                        if n.isdigit():
                            cur_module = cur_module[int(n)]
                        else:
                            cur_module = getattr(cur_module, n)
                    cur_module.to_q = get_and_set_lora_layer(cur_module.to_q, rank)
                    cur_module.to_kv = get_and_set_lora_layer(cur_module.to_kv, rank)
                    cur_module.to_out = get_and_set_lora_layer(cur_module.to_out, rank)

                    add_on_lora_parameters.extend(cur_module.to_q.lora_layer.parameters())
                    add_on_lora_parameters.extend(cur_module.to_kv.lora_layer.parameters())
                    add_on_lora_parameters.extend(cur_module.to_out.lora_layer.parameters())

            elif name.startswith('id_adapter_attn_layers.'):
                if name.endswith('id_to_k'):
                    for n in name.split('.')[:-1]:
                        if n.isdigit():
                            cur_module = cur_module[int(n)]
                        else:
                            cur_module = getattr(cur_module, n)
                    cur_module.id_to_k = get_and_set_lora_layer(cur_module.id_to_k, rank)
                    cur_module.id_to_v = get_and_set_lora_layer(cur_module.id_to_v, rank)

                    add_on_lora_parameters.extend(cur_module.id_to_k.lora_layer.parameters())
                    add_on_lora_parameters.extend(cur_module.id_to_v.lora_layer.parameters())

            # elif name.startswith('clip_vision_model.'):
                # if name.endswith('q_proj'):
                    # for n in name.split('.')[:-1]:
                        # if n.isdigit():
                            # cur_module = cur_module[int(n)]
                        # else:
                            # cur_module = getattr(cur_module, n)
                    # cur_module.q_proj = get_and_set_lora_layer(cur_module.q_proj, rank)
                    # cur_module.k_proj = get_and_set_lora_layer(cur_module.k_proj, rank)
                    # cur_module.v_proj = get_and_set_lora_layer(cur_module.v_proj, rank)

                    # add_on_lora_parameters.extend(cur_module.q_proj.lora_layer.parameters())
                    # add_on_lora_parameters.extend(cur_module.k_proj.lora_layer.parameters())
                    # add_on_lora_parameters.extend(cur_module.v_proj.lora_layer.parameters())

        self.id_adapter.proj_out = LoRALinearWrapper(self.id_adapter.proj_out, rank)
        add_on_lora_parameters.extend([self.id_adapter.proj_out.lora_A, self.id_adapter.proj_out.lora_B])
        return add_on_lora_parameters
    
    def save(self, output_dir):
        lora_state_dict = {}
        for name, module in self.named_modules():
            # 第一类：通过 set_lora_layer 获取 lora_layer
            if hasattr(module, "set_lora_layer"):
                lora_layer = getattr(module, "lora_layer", None)
                if lora_layer is not None:
                    current_lora_layer_sd = lora_layer.state_dict()
                    for lora_layer_matrix_name, lora_param in current_lora_layer_sd.items():
                        # 确保参数移动到 CPU
                        lora_state_dict[f"{name}.lora_layer.{lora_layer_matrix_name}"] = (
                            lora_param.cpu() if lora_param.device.type != "cpu" else lora_param
                        )
        lora_state_dict['id_adapter.proj_out.lora_A'] = (
            self.id_adapter.proj_out.lora_A.cpu() 
            if self.id_adapter.proj_out.lora_A.device.type != "cpu" 
            else self.id_adapter.proj_out.lora_A
        )
        lora_state_dict['id_adapter.proj_out.lora_B'] = (
            self.id_adapter.proj_out.lora_B.cpu() 
            if self.id_adapter.proj_out.lora_B.device.type != "cpu" 
            else self.id_adapter.proj_out.lora_B
        )

        output_dir = os.path.join(output_dir, 'add_on')
        os.makedirs(output_dir, exist_ok=True)
        file_path = os.path.join(output_dir, "pytorch_lora_weights.safetensors")
        # 保存 LoRA 状态字典
        safetensors.torch.save_file(lora_state_dict, file_path, metadata={"format": "pt"})
        print(f"Adapter LoRA state dict saved to {file_path}")

    def load(self, model_path, rank):
        """
        加载保存的 LoRA 参数，并更新模型中对应模块的参数。
        逻辑：
        - 如果 key 格式为 "模块名.lora_layer.矩阵名"，则更新对应模块的 lora_layer 参数；
        - 否则，key 格式为 "模块名.参数名"，直接更新该模块的参数（例如 LoRALinearWrapper）。
        """
        if not model_path.endswith('safetensors'):
            model_path = os.path.join(model_path, 'pytorch_lora_weights.safetensors')
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"File {model_path} not found!") 
    
        # modify modules
        if not self.is_finetuned:
            self.finetune(rank)

        lora_state_dict = load_file(model_path)
    
        for name, param in lora_state_dict.items():
            cur_item = self
            parts = name.split('.')
            for n in parts[:-1]:
                if n.isdigit():
                    cur_item = cur_item[int(n)]
                else:
                    cur_item = getattr(cur_item, n)
            setattr(cur_item, parts[-1], nn.Parameter(param.to(self.device)))
                
        print(f"Adapter LoRA parameters loaded from {model_path}")

    
    def add_to_pipeline(self, pipeline):
        pipeline.eva_transform_mean = self.eva_transform_mean
        pipeline.eva_transform_std = self.eva_transform_std
        pipeline.id_adapter = self.id_adapter
        pipeline.face_helper = self.face_helper
        pipeline.app = self.app
        pipeline.handler_ante = self.handler_ante
        pipeline.clip_vision_model = self.clip_vision_model # to finetune
        pipeline.id_adapter_attn_layers = self.id_adapter_attn_layers # to finetune
        pipeline.det_net = self.det_net
        pipeline.genderage_net = self.genderage_net
        return pipeline
    
