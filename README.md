
# AgeBooth: Controllable Facial Aging and Rejuvenation via Diffusion Models

Official implementation of [AgeBooth: Controllable Facial Aging and
 Rejuvenation via Diffusion Models](https://arxiv.org/pdf/2502.18461).

## 🎨 Examples

In the following examples, we demonstrate how AgeBooth leverages InstantID, PuLID, and InfiniteYou models to generate controllable facial aging and rejuvenation effects. These models are used within AgeBooth to create personalized age transformations, allowing adjustments to the aging process of a reference individual from 15 to 75 years old.

![more_visuals](pictures/more_visuals.svg)
![more_visuals1](pictures/more_visuals1.svg)

## 🔧 Installation

```
git clone https://github.com/HVision-NKU/AgeBooth.git
cd AgeBooth
pip install -r requirements.txt
```

## 🔥Usage

This repository provides scripts to **train LoRA adapters** for different age ranges and then perform **age interpolation** with different ID customization methods.

### Workflow Overview

1. Train a young LoRA(e.g., 10–20 years old)
2. Train an old LoRA(e.g., 70–80 years old)
3. Run inference with both LoRAs → generates interpolated images across ages

### 1. Train LoRA

Use the script **`train_dreambooth_lora_sdxl.sh` / `train_dreambooth_lora_flux.sh`** to train a LoRA for a specific age range.
You only need to set:

- `MODEL_NAME` → base model path (e.g., `RunDiffusion/Juggrnaut-XL-v9`)
- `OUTPUT_DIR` → path where the LoRA will be saved
- `INSTANCE_DIR` → dataset location for the target age group

👉 Train one LoRA for the **young age group** (e.g., 10–20) and another for the **old age group** (e.g., 70–80).

---

### 2. Run Inference

Once both LoRAs are trained, use **`inference_lora_interp_traverse_pulid.sh` / `inference_lora_interp_traverse_infU.sh`** for PuLID/InfiniteYou inference.
You need to specify:

- `METHOD` → interpolation method (`direct_linear` or `svdmix`)
- `YOUNG_LORA` → checkpoint path of the trained young LoRA
- `OLD_LORA` → checkpoint path of the trained old LoRA
- `IMAGE_PATH` → input image for identity preservation
- `SEX` → gender of the subject (`m` or `f`)

To control the strength of identity injection, you can adjust the `ID_SCALE` parameter. You can find some pre-downloaded identity images in the `example_inputs` folder.

The outputs will be saved in the `outputs/` directory, organized by method and parameters.


## 🚩TODO

- [x] training code for SDXL age LoRA
- [x] training code for FLUX age LoRA
- [x] AgeBooth inference for PuLID
- [x] AgeBooth inference for InfiniteYou
- [ ] AgeBooth inference for InstantID
- [ ] Pretrained age LoRAs

## 📄Citation
If you find this AgeBooth useful for your research or applications, please cite using this BibTeX:

```BibTeX
@inproceedings{,
  title={},
  author={},
  booktitle={},
  year={2025}
}
```