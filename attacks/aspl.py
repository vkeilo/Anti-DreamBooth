import argparse
import copy
import hashlib
import itertools
import logging
import os
from pathlib import Path

import datasets
import diffusers
import numpy as np
import torch
import torch.nn.functional as F
import torch.utils.checkpoint
import transformers
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import set_seed
from diffusers import AutoencoderKL, DDPMScheduler, DiffusionPipeline, UNet2DConditionModel
from diffusers.utils.import_utils import is_xformers_available
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from tqdm.auto import tqdm
from transformers import AutoTokenizer, PretrainedConfig


logger = get_logger(__name__)


class DreamBoothDatasetFromTensor(Dataset):
    """Just like DreamBoothDataset, but take instance_images_tensor instead of path"""

    def __init__(
        self,
        instance_images_tensor,
        instance_prompt,
        tokenizer,
        class_data_root=None,
        class_prompt=None,
        size=512,
        center_crop=False,
    ):
        self.size = size
        self.center_crop = center_crop
        self.tokenizer = tokenizer

        self.instance_images_tensor = instance_images_tensor
        self.num_instance_images = len(self.instance_images_tensor)
        self.instance_prompt = instance_prompt
        self._length = self.num_instance_images

        if class_data_root is not None:
            self.class_data_root = Path(class_data_root)
            self.class_data_root.mkdir(parents=True, exist_ok=True)
            self.class_images_path = list(self.class_data_root.iterdir())
            self.num_class_images = len(self.class_images_path)
            self._length = max(self.num_class_images, self.num_instance_images)
            self.class_prompt = class_prompt
        else:
            self.class_data_root = None

        self.image_transforms = transforms.Compose(
            [
                transforms.Resize(size, interpolation=transforms.InterpolationMode.BILINEAR),
                transforms.CenterCrop(size) if center_crop else transforms.RandomCrop(size),
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5]),
            ]
        )

    def __len__(self):
        return self._length

    def __getitem__(self, index):
        example = {}
        instance_image = self.instance_images_tensor[index % self.num_instance_images]
        example["instance_images"] = instance_image
        example["instance_prompt_ids"] = self.tokenizer(
            self.instance_prompt,
            truncation=True,
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            return_tensors="pt",
        ).input_ids

        if self.class_data_root:
            class_image = Image.open(self.class_images_path[index % self.num_class_images])
            if not class_image.mode == "RGB":
                class_image = class_image.convert("RGB")
            example["class_images"] = self.image_transforms(class_image)
            example["class_prompt_ids"] = self.tokenizer(
                self.class_prompt,
                truncation=True,
                padding="max_length",
                max_length=self.tokenizer.model_max_length,
                return_tensors="pt",
            ).input_ids

        return example


def import_model_class_from_model_name_or_path(pretrained_model_name_or_path: str, revision: str):
    text_encoder_config = PretrainedConfig.from_pretrained(
        pretrained_model_name_or_path,
        subfolder="text_encoder",
        revision=revision,
    )
    model_class = text_encoder_config.architectures[0]

    if model_class == "CLIPTextModel":
        from transformers import CLIPTextModel

        return CLIPTextModel
    elif model_class == "RobertaSeriesModelWithTransformation":
        from diffusers.pipelines.alt_diffusion.modeling_roberta_series import RobertaSeriesModelWithTransformation

        return RobertaSeriesModelWithTransformation
    else:
        raise ValueError(f"{model_class} is not supported.")


def parse_args(input_args=None):
    parser = argparse.ArgumentParser(description="Simple example of a training script.")
    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        default=None,
        required=True,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--revision",
        type=str,
        default=None,
        required=False,
        help=(
            "Revision of pretrained model identifier from huggingface.co/models. Trainable model components should be"
            " float32 precision."
        ),
    )
    parser.add_argument(
        "--tokenizer_name",
        type=str,
        default=None,
        help="Pretrained tokenizer name or path if not the same as model_name",
    )
    parser.add_argument(
        "--instance_data_dir_for_train",
        type=str,
        default=None,
        required=True,
        help="A folder containing the training data of instance images.",
    )
    parser.add_argument(
        "--instance_data_dir_for_adversarial",
        type=str,
        default=None,
        required=True,
        help="A folder containing the images to add adversarial noise",
    )
    parser.add_argument(
        "--class_data_dir",
        type=str,
        default=None,
        required=False,
        help="A folder containing the training data of class images.",
    )
    parser.add_argument(
        "--instance_prompt",
        type=str,
        default=None,
        required=True,
        help="The prompt with identifier specifying the instance",
    )
    parser.add_argument(
        "--class_prompt",
        type=str,
        default=None,
        help="The prompt to specify images in the same class as provided instance images.",
    )
    parser.add_argument(
        "--with_prior_preservation",
        default=False,
        action="store_true",
        help="Flag to add prior preservation loss.",
    )
    parser.add_argument(
        "--prior_loss_weight",
        type=float,
        default=1.0,
        help="The weight of prior preservation loss.",
    )
    parser.add_argument(
        "--num_class_images",
        type=int,
        default=100,
        help=(
            "Minimal class images for prior preservation loss. If there are not enough images already present in"
            " class_data_dir, additional images will be sampled with class_prompt."
        ),
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="text-inversion-model",
        help="The output directory where the model predictions and checkpoints will be written.",
    )
    parser.add_argument("--seed", type=int, default=None, help="A seed for reproducible training.")
    parser.add_argument(
        "--resolution",
        type=int,
        default=512,
        help=(
            "The resolution for input images, all the images in the train/validation dataset will be resized to this"
            " resolution"
        ),
    )
    parser.add_argument(
        "--center_crop",
        default=False,
        action="store_true",
        help=(
            "Whether to center crop the input images to the resolution. If not set, the images will be randomly"
            " cropped. The images will be resized to the resolution first before cropping."
        ),
    )
    parser.add_argument(
        "--train_text_encoder",
        action="store_true",
        help="Whether to train the text encoder. If set, the text encoder should be float32 precision.",
    )
    parser.add_argument(
        "--train_batch_size",
        type=int,
        default=4,
        help="Batch size (per device) for the training dataloader.",
    )
    parser.add_argument(
        "--sample_batch_size",
        type=int,
        default=8,
        help="Batch size (per device) for sampling images.",
    )
    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=20,
        help="Total number of training steps to perform.",
    )
    parser.add_argument(
        "--max_f_train_steps",
        type=int,
        default=10,
        help="Total number of sub-steps to train surogate model.",
    )
    parser.add_argument(
        "--max_adv_train_steps",
        type=int,
        default=10,
        help="Total number of sub-steps to train adversarial noise.",
    )
    parser.add_argument(
        "--checkpointing_iterations",
        type=int,
        default=5,
        help=("Save a checkpoint of the training state every X iterations."),
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=5e-6,
        help="Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument(
        "--logging_dir",
        type=str,
        default="logs",
        help=(
            "[TensorBoard](https://www.tensorflow.org/tensorboard) log directory. Will default to"
            " *output_dir/runs/**CURRENT_DATETIME_HOSTNAME***."
        ),
    )
    parser.add_argument(
        "--allow_tf32",
        action="store_true",
        help=(
            "Whether or not to allow TF32 on Ampere GPUs. Can be used to speed up training. For more information, see"
            " https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices"
        ),
    )
    parser.add_argument(
        "--report_to",
        type=str,
        default="tensorboard",
        help=(
            'The integration to report the results and logs to. Supported platforms are `"tensorboard"`'
            ' (default), `"wandb"` and `"comet_ml"`. Use `"all"` to report to all integrations.'
        ),
    )
    parser.add_argument(
        "--mixed_precision",
        type=str,
        default="fp16",
        choices=["no", "fp16", "bf16"],
        help=(
            "Whether to use mixed precision. Choose between fp16 and bf16 (bfloat16). Bf16 requires PyTorch >="
            " 1.10.and an Nvidia Ampere GPU.  Default to the value of accelerate config of the current system or the"
            " flag passed with the `accelerate.launch` command. Use this argument to override the accelerate config."
        ),
    )
    parser.add_argument(
        "--enable_xformers_memory_efficient_attention",
        action="store_true",
        help="Whether or not to use xformers.",
    )
    parser.add_argument(
        "--pgd_alpha",
        type=float,
        default=1.0 / 255,
        help="The step size for pgd.",
    )
    parser.add_argument(
        "--pgd_eps",
        type=float,
        default=0.05,
        help="The noise budget for pgd.",
    )
    parser.add_argument(
        "--target_image_path",
        default=None,
        help="target image for attacking",
    )
    parser.add_argument(  
        "--robust_start",  
        type=float,  
        default=0.3,  
        help="Start point for robust noise (0 < robust_start < 1)",  
    )  
    parser.add_argument(  
        "--robust_fixed",   
        type=float,  
        default=0.8,  
        help="Fixed noise point (robust_start < robust_fixed < 1)",  
    )  
    parser.add_argument(  
        "--sigma",  
        type=float,  
        default=0,  
        help="Maximum noise strength for robust training",  
    )
    parser.add_argument(  
        "--num_weight_samples_when_delta",  
        type=int,  
        default=1,  
        help="",  
    )
    parser.add_argument(  
        "--num_weight_samples_when_theta",  
        type=int,  
        default=1,  
        help="",  
    )


    if input_args is not None:
        args = parser.parse_args(input_args)
    else:
        args = parser.parse_args()

    return args


class PromptDataset(Dataset):
    "A simple dataset to prepare the prompts to generate class images on multiple GPUs."

    def __init__(self, prompt, num_samples):
        self.prompt = prompt
        self.num_samples = num_samples

    def __len__(self):
        return self.num_samples

    def __getitem__(self, index):
        example = {}
        example["prompt"] = self.prompt
        example["index"] = index
        return example

# vkeilo add it for noise add to model
def add_noise_to_model(model, noise_strength):  
    """向模型参数添加高斯噪声"""  
    with torch.no_grad():  
        for param in model.parameters():  
            if param.requires_grad:  
                noise = torch.randn_like(param) * noise_strength  
                param.add_(noise)

def _apply_weight_noise_(module: torch.nn.Module, sigma: float, sign: float = +1.0, seed: int = None):
    """
    在 module 的参数上加/撤噪声 (可逆版本)。
    - sigma: 噪声标准差
    - sign: +1 表示加噪, -1 表示撤噪
    - seed: 若给定, 可保证噪声可复现 (撤噪时必须用相同 seed)
    """
    if sigma == 0.0:
        return

    # 固定随机数生成器，保证可逆性
    device = next(module.parameters()).device
    g = torch.Generator(device=device)
    if seed is not None:
        g.manual_seed(int(seed))
    else:
        # 没有 seed 时随机生成一个
        seed = torch.seed()
        g.manual_seed(seed)

    for name, param in module.named_parameters():
        if not param.requires_grad:
            continue
        if not param.data.dtype.is_floating_point:
            continue
        eps = torch.randn(param.data.shape, device=param.data.device, dtype=param.data.dtype, generator=g)
        param.data.add_(sign * sigma * eps)
        
def load_data(data_dir, size=512, center_crop=True) -> torch.Tensor:
    image_transforms = transforms.Compose(
        [
            transforms.Resize(size, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.CenterCrop(size) if center_crop else transforms.RandomCrop(size),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ]
    )

    # vkeilo add it for order
    image_paths = sorted(list(Path(data_dir).glob("*.jpg")) + list(Path(data_dir).glob("*.png")))
    # print(image_paths)
    images = [image_transforms(Image.open(p).convert("RGB")) for p in image_paths]
    images = torch.stack(images)
    image_names = [p.stem for p in image_paths]
    return images,image_names

def train_one_epoch(  
    args,  
    models,  
    tokenizer,  
    noise_scheduler,  
    vae,  
    data_tensor: torch.Tensor,  
    num_steps=20,  
    current_step=0,  
    total_steps=1000,  
    num_weight_samples: int = 5,   # 新增：每个step的权重采样次数K
):  
    import copy, itertools, torch, torch.nn.functional as F

    # 创建干净的模型副本（用于参数更新）  
    unet, text_encoder = copy.deepcopy(models[0]), copy.deepcopy(models[1])  

    # 噪声强度调度  
    progress = current_step / max(total_steps, 1)    
    if progress < args.robust_start:    
        noise_strength = 0.0    
    elif progress < args.robust_fixed:    
        linear_progress = (progress - args.robust_start) / (args.robust_fixed - args.robust_start)    
        noise_strength = linear_progress * args.sigma    
    else:    
        noise_strength = args.sigma  

    # 备份“干净权重”用于每个step开头恢复  
    unet_clean_state = copy.deepcopy(unet.state_dict())  
    text_encoder_clean_state = copy.deepcopy(text_encoder.state_dict())  

    params_to_optimize = itertools.chain(unet.parameters(), text_encoder.parameters())  
    optimizer = torch.optim.AdamW(  
        params_to_optimize,  
        lr=args.learning_rate,  
        betas=(0.9, 0.999),  
        weight_decay=1e-2,  
        eps=1e-08,  
    )  

    train_dataset = DreamBoothDatasetFromTensor(  
        data_tensor,  
        args.instance_prompt,  
        tokenizer,  
        args.class_data_dir,  
        args.class_prompt,  
        args.resolution,  
        args.center_crop,  
    )  

    weight_dtype = torch.bfloat16  
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  

    vae.to(device, dtype=weight_dtype).eval()  # VAE不更新
    text_encoder.to(device, dtype=weight_dtype)  
    unet.to(device, dtype=weight_dtype)  

    K = max(1, int(num_weight_samples))
    if noise_strength == 0.0:
        K = 1

    for step in range(num_steps):  
        # 恢复到干净参数
        unet.load_state_dict(unet_clean_state)  
        text_encoder.load_state_dict(text_encoder_clean_state)  

        unet.train(); text_encoder.train()

        step_data = train_dataset[step % len(train_dataset)]
        pixel_values = torch.stack([step_data["instance_images"].to(device), 
                                    step_data["class_images"].to(device)]
                                ).to(device, dtype=weight_dtype)  
        input_ids = torch.cat([step_data["instance_prompt_ids"].to(device), step_data["class_prompt_ids"].to(device)], dim=0).to(device)  

        with torch.no_grad():
            latents = vae.encode(pixel_values).latent_dist.sample() * vae.config.scaling_factor

        optimizer.zero_grad(set_to_none=True)
        total_loss_val = 0.0

        # ---------- 这里是关键：同一step内做K次权重噪声采样并平均梯度 ----------
        # for k in range(K):
        #     # 临时加噪（forward前加，backward后撤销）
        #     seed_te   = torch.seed()
        #     seed_unet = torch.seed()
        #     if noise_strength > 0:
        #         _apply_weight_noise_(unet,         noise_strength, +1.0, seed_te)  # 你已有的函数：对每个param加 N(0,σ) 噪声
        #         _apply_weight_noise_(text_encoder, noise_strength, +1.0, seed_unet)

        #     # 前向
        #     noise = torch.randn_like(latents)  
        #     bsz = latents.shape[0]  
        #     timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (bsz,), device=latents.device).long()  
        #     noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)  
        #     encoder_hidden_states = text_encoder(input_ids)[0]  
        #     model_pred = unet(noisy_latents, timesteps, encoder_hidden_states).sample  

        #     if noise_scheduler.config.prediction_type == "epsilon":  
        #         target = noise  
        #     elif noise_scheduler.config.prediction_type == "v_prediction":  
        #         target = noise_scheduler.get_velocity(latents, noise, timesteps)  
        #     else:  
        #         raise ValueError(f"Unknown prediction type {noise_scheduler.config.prediction_type}")  

        #     if args.with_prior_preservation:  
        #         model_pred, model_pred_prior = torch.chunk(model_pred, 2, dim=0)  
        #         target,     target_prior     = torch.chunk(target,     2, dim=0)  
        #         instance_loss = F.mse_loss(model_pred.float(),       target.float(),       reduction="mean")  
        #         prior_loss    = F.mse_loss(model_pred_prior.float(), target_prior.float(), reduction="mean")  
        #         loss = instance_loss + args.prior_loss_weight * prior_loss  
        #     else:  
        #         loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean")  

        #     # 累计平均梯度：每次只反传 loss/K
        #     (loss / float(K)).backward()
        #     total_loss_val += float(loss.detach())

        #     # 立刻撤销刚才加的噪声（把 +noise 加到的参数再减回去）
        #     if noise_strength > 0:
        #         _apply_weight_noise_(unet,         noise_strength, -1.0, seed_te)
        #         _apply_weight_noise_(text_encoder, noise_strength, -1.0, seed_unet)
        # ---------- K次结束，做一次优化 ----------
        avg_grads = [torch.zeros_like(p, memory_format=torch.preserve_format) for p in params_to_optimize]

        # fix NaN in pth
        for k in range(K):
            # 恢复干净参数
            unet.load_state_dict(unet_clean_state)
            text_encoder.load_state_dict(text_encoder_clean_state)

            # 加噪
            seed_te   = torch.seed()
            seed_unet = torch.seed()
            if noise_strength > 0:
                _apply_weight_noise_(unet, noise_strength, +1.0, seed_unet)
                _apply_weight_noise_(text_encoder, noise_strength, +1.0, seed_te)
            noise = torch.randn_like(latents)  
            bsz = latents.shape[0]  
            timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (bsz,), device=latents.device).long()  
            noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)  
            encoder_hidden_states = text_encoder(input_ids)[0]  
            model_pred = unet(noisy_latents, timesteps, encoder_hidden_states).sample  

            if noise_scheduler.config.prediction_type == "epsilon":  
                target = noise  
            elif noise_scheduler.config.prediction_type == "v_prediction":  
                target = noise_scheduler.get_velocity(latents, noise, timesteps)  
            else:  
                raise ValueError(f"Unknown prediction type {noise_scheduler.config.prediction_type}")  
            # forward + backward
            if args.with_prior_preservation:  
                model_pred, model_pred_prior = torch.chunk(model_pred, 2, dim=0)  
                target,     target_prior     = torch.chunk(target,     2, dim=0)  
                instance_loss = F.mse_loss(model_pred.float(),       target.float(),       reduction="mean")  
                prior_loss    = F.mse_loss(model_pred_prior.float(), target_prior.float(), reduction="mean")  
                loss = instance_loss + args.prior_loss_weight * prior_loss  
            else:  
                loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean")  

            (loss / K).backward()

            # 把当前梯度累积到 avg_grads，然后立刻清空 .grad，防止累积爆炸
            for p, g_buf in zip(params_to_optimize, avg_grads):
                if p.grad is not None:
                    g_buf.add_(p.grad)
                    p.grad = None  # 清空显存，防止梯度堆积

            # 撤噪
            if noise_strength > 0:
                _apply_weight_noise_(unet, noise_strength, -1.0, seed_unet)
                _apply_weight_noise_(text_encoder, noise_strength, -1.0, seed_te)

        # 统一赋回平均梯度（除以 K）
        for p, g_buf in zip(params_to_optimize, avg_grads):
            p.grad = g_buf / K


        torch.nn.utils.clip_grad_norm_(params_to_optimize, 1.0, error_if_nonfinite=True)  
        optimizer.step()  
        optimizer.zero_grad(set_to_none=True)

        # 刷新“干净权重”备份（下一step用）
        unet_clean_state = copy.deepcopy(unet.state_dict())  
        text_encoder_clean_state = copy.deepcopy(text_encoder.state_dict())  

        print(f"Step #{step}, noise={noise_strength:.4e}, avg_loss={total_loss_val/float(K):.6f}")  

    return [unet, text_encoder]

def pgd_attack(
    args,
    models,
    tokenizer,
    noise_scheduler,
    vae,
    data_tensor: torch.Tensor,
    original_images: torch.Tensor,
    target_tensor: torch.Tensor,
    num_steps: int,
    current_step=0,         # 保留
    total_steps=1000,       # 保留
    num_weight_samples=5,   # ★ 新增：δ-更新时 K 次权重噪声采样
):
    """Return new perturbed data (串行 K 次权重采样，平均梯度更新 δ)"""

    unet, text_encoder = models
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    weight_dtype = torch.bfloat16

    # 统一 dtype + eval + 冻结参数（不训练模型）
    vae.to(device, dtype=weight_dtype).eval()
    text_encoder.to(device, dtype=weight_dtype).eval()
    unet.to(device, dtype=weight_dtype).eval()
    for m in (vae, text_encoder, unet):
        for p in m.parameters():
            p.requires_grad_(False)

    # 噪声强度调度（跟 theta-step 一致）
    progress = current_step / max(total_steps, 1)
    if progress < args.robust_start:
        noise_strength = 0.0
    elif progress < args.robust_fixed:
        lin = (progress - args.robust_start) / (args.robust_fixed - args.robust_start)
        noise_strength = lin * args.sigma
    else:
        noise_strength = args.sigma

    # PGD 变量
    perturbed_images = data_tensor.detach().clone().to(device)   # BCHW in [-1,1]
    perturbed_images.requires_grad_(True)

    # 文本 token（小）提前准备
    input_ids = tokenizer(
        args.instance_prompt,
        truncation=True,
        padding="max_length",
        max_length=tokenizer.model_max_length,
        return_tensors="pt",
    ).input_ids.to(device)
    if input_ids.shape[0] == 1:
        input_ids = input_ids.repeat(len(perturbed_images), 1)

    # 备份“干净”模型权重用于 K 次循环里的加/撤噪（避免误差积累）
    clean_unet_sd = copy.deepcopy(unet.state_dict())
    clean_te_sd   = copy.deepcopy(text_encoder.state_dict())

    # # 可逆加/撤噪：用固定种子产生同一 ε，sign=+1 加，sign=-1 撤
    # def _apply_weight_noise_with_seed_(module: torch.nn.Module, sigma: float, seed: int, sign: float):
    #     if sigma == 0.0:
    #         return
    #     g = torch.Generator(device=device)
    #     g.manual_seed(int(seed))
    #     for _, p in module.named_parameters():
    #         if not p.dtype.is_floating_point:  # 跳过 int/bool 等
    #             continue
    #         # eps = torch.randn_like(p, generator=g)
    #         p.data.add_(sign * sigma * eps)

    K = max(1, int(num_weight_samples))
    if noise_strength == 0.0:
        K = 1

    alpha, eps = args.pgd_alpha, args.pgd_eps

    for step in range(num_steps):
        # 每步开始：恢复干净权重
        unet.load_state_dict(clean_unet_sd, strict=True)
        text_encoder.load_state_dict(clean_te_sd, strict=True)

        if perturbed_images.grad is not None:
            perturbed_images.grad.zero_()

        avg_loss_val = 0.0

        # # ====== 同一 PGD 步内做 K 次权重噪声采样，平均梯度到 δ ======
        # for k in range(K):
        #     # 临时加噪（可逆）
        #     seed_te   = torch.seed()
        #     seed_unet = torch.seed()
        #     _apply_weight_noise_(text_encoder, noise_strength,   +1.0, seed_te)
        #     _apply_weight_noise_(unet,         noise_strength, +1.0, seed_unet)

        #     # 前向（注意 VAE 需要对图像反传，所以不加 no_grad；其它用 autocast 节省显存）
        #     # 1) 文本编码（可 no_grad）
        #     with torch.no_grad(), torch.autocast(device_type="cuda", dtype=weight_dtype, enabled=(device.type=="cuda")):
        #         encoder_hidden_states = text_encoder(input_ids)[0]  # [B, 77, 1024] for SD2.1

        #     # 2) VAE 编码图像 → latents（要对图像回传）
        #     with torch.autocast(device_type="cuda", dtype=weight_dtype, enabled=(device.type=="cuda")):
        #         latents = vae.encode(perturbed_images.to(dtype=weight_dtype)).latent_dist.sample()
        #         latents = latents * vae.config.scaling_factor

        #         noise = torch.randn_like(latents)
        #         bsz   = latents.shape[0]
        #         timesteps = torch.randint(
        #             0, noise_scheduler.config.num_train_timesteps, (bsz,),
        #             device=latents.device
        #         ).long()
        #         noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

        #         # UNet 预测
        #         model_pred = unet(noisy_latents, timesteps, encoder_hidden_states).sample

        #         # 组装目标
        #         if noise_scheduler.config.prediction_type == "epsilon":
        #             target = noise
        #         elif noise_scheduler.config.prediction_type == "v_prediction":
        #             target = noise_scheduler.get_velocity(latents, noise, timesteps)
        #         else:
        #             raise ValueError(f"Unknown prediction type {noise_scheduler.config.prediction_type}")

        #         loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean")

        #         # one-step target shift（若启用）
        #         if target_tensor is not None:
        #             step_out = noise_scheduler.step(model_pred, timesteps, noisy_latents)
        #             xtm1_pred   = step_out.prev_sample
        #             xtm1_target = noise_scheduler.add_noise(
        #                 target_tensor.to(device, dtype=weight_dtype),
        #                 noise,
        #                 timesteps - 1
        #             )
        #             loss = loss - F.mse_loss(xtm1_pred, xtm1_target, reduction="mean")

        #     # 把本次采样的梯度均摊到 δ：loss/K
        #     (loss / float(K)).backward()
        #     avg_loss_val += float(loss.detach()) / float(K)

        #     # 立刻撤回这次噪声，回到干净权重
        #     _apply_weight_noise_(unet,         noise_strength, -1.0, seed_unet)
        #     _apply_weight_noise_(text_encoder, noise_strength,   -1.0, seed_te)

        #     # 及时丢掉临时张量引用
        #     del encoder_hidden_states, latents, noisy_latents, model_pred, target, noise, timesteps, loss

        # ====== 同一 PGD 步内做 K 次权重噪声采样，平均梯度到 δ（稳定版） ======

        # fix NaN in pth
        optimizer = torch.optim.SGD([perturbed_images], lr=1.0)  # 临时优化器，用于统一处理梯度
        optimizer.zero_grad(set_to_none=True)
        avg_grads = torch.zeros_like(perturbed_images)

        for k in range(K):
            # 临时加噪（可逆）
            seed_te   = torch.seed()
            seed_unet = torch.seed()
            _apply_weight_noise_(text_encoder, noise_strength, +1.0, seed_te)
            _apply_weight_noise_(unet,         noise_strength, +1.0, seed_unet)

            # 1) 文本编码（no_grad, 节省显存）
            with torch.no_grad(), torch.autocast(device_type="cuda", dtype=weight_dtype, enabled=(device.type == "cuda")):
                encoder_hidden_states = text_encoder(input_ids)[0]  # [B, 77, 1024] for SD2.1

            # 2) VAE 编码图像 → latents
            with torch.autocast(device_type="cuda", dtype=weight_dtype, enabled=(device.type == "cuda")):
                latents = vae.encode(perturbed_images.to(dtype=weight_dtype)).latent_dist.sample()
                latents = latents * vae.config.scaling_factor

                noise = torch.randn_like(latents)
                bsz = latents.shape[0]
                timesteps = torch.randint(
                    0, noise_scheduler.config.num_train_timesteps, (bsz,),
                    device=latents.device
                ).long()
                noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

                # UNet 前向预测
                model_pred = unet(noisy_latents, timesteps, encoder_hidden_states).sample

                # 目标组装
                if noise_scheduler.config.prediction_type == "epsilon":
                    target = noise
                elif noise_scheduler.config.prediction_type == "v_prediction":
                    target = noise_scheduler.get_velocity(latents, noise, timesteps)
                else:
                    raise ValueError(f"Unknown prediction type {noise_scheduler.config.prediction_type}")

                loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean")

                # one-step target shift（若启用）
                if target_tensor is not None:
                    step_out = noise_scheduler.step(model_pred, timesteps, noisy_latents)
                    xtm1_pred = step_out.prev_sample
                    xtm1_target = noise_scheduler.add_noise(
                        target_tensor.to(device, dtype=weight_dtype),
                        noise,
                        timesteps - 1
                    )
                    loss = loss - F.mse_loss(xtm1_pred, xtm1_target, reduction="mean")

            # ========= 新逻辑：显式收集梯度，防止累积溢出 =========
            optimizer.zero_grad(set_to_none=True)
            loss.backward()

            if not torch.isfinite(loss):
                print(f"[WARN] Non-finite loss at PGD step {step}, sample {k}, skipped.")
                _apply_weight_noise_(unet, noise_strength, -1.0, seed_unet)
                _apply_weight_noise_(text_encoder, noise_strength, -1.0, seed_te)
                continue

            # 把本次梯度加进平均梯度缓冲区
            avg_grads.add_(perturbed_images.grad.detach() / float(K))

            # 清空 grad，释放显存
            perturbed_images.grad = None
            optimizer.zero_grad(set_to_none=True)

            avg_loss_val += float(loss.detach()) / float(K)

            # 撤回这次噪声，回到干净权重
            _apply_weight_noise_(unet, noise_strength, -1.0, seed_unet)
            _apply_weight_noise_(text_encoder, noise_strength, -1.0, seed_te)

            del encoder_hidden_states, latents, noisy_latents, model_pred, target, noise, timesteps, loss

        # ✅ 最终一次性赋值平均梯度到 perturbed_images
        perturbed_images.grad = avg_grads

        # ====== 一次 PGD 步（对 δ）======
        with torch.no_grad():
            g = perturbed_images.grad
            adv = perturbed_images + alpha * g.sign()
            eta = torch.clamp(adv - original_images.to(device), min=-eps, max=+eps)
            perturbed_images = torch.clamp(original_images.to(device) + eta, min=-1, max=+1).detach()
            perturbed_images.requires_grad_(True)

        print(f"[PGD step {step:02d}] noise={noise_strength:.4e} | avg_loss={avg_loss_val:.6f}")

        if device.type == "cuda":
            torch.cuda.empty_cache()

    return perturbed_images

# def train_one_epoch(  
#     args,  
#     models,  
#     tokenizer,  
#     noise_scheduler,  
#     vae,  
#     data_tensor: torch.Tensor,  
#     num_steps=20,  
#     current_step=0,  # 新增参数    
#     total_steps=1000,  # 新增参数    
# ):  
#     # 创建干净的模型副本（用于参数更新）  
#     unet, text_encoder = copy.deepcopy(models[0]), copy.deepcopy(models[1])  
      
#     # 计算噪声强度  
#     progress = current_step / total_steps    
#     if progress < args.robust_start:    
#         noise_strength = 0.0    
#     elif progress < args.robust_fixed:    
#         linear_progress = (progress - args.robust_start) / (args.robust_fixed - args.robust_start)    
#         noise_strength = linear_progress * args.sigma    
#     else:    
#         noise_strength = args.sigma  
      
#     # 备份干净模型的状态字典  
#     unet_clean_state = copy.deepcopy(unet.state_dict())  
#     text_encoder_clean_state = copy.deepcopy(text_encoder.state_dict())  
  
#     params_to_optimize = itertools.chain(unet.parameters(), text_encoder.parameters())  
#     optimizer = torch.optim.AdamW(  
#         params_to_optimize,  
#         lr=args.learning_rate,  
#         betas=(0.9, 0.999),  
#         weight_decay=1e-2,  
#         eps=1e-08,  
#     )  
  
#     train_dataset = DreamBoothDatasetFromTensor(  
#         data_tensor,  
#         args.instance_prompt,  
#         tokenizer,  
#         args.class_data_dir,  
#         args.class_prompt,  
#         args.resolution,  
#         args.center_crop,  
#     )  
  
#     weight_dtype = torch.bfloat16  
#     device = torch.device("cuda")  
  
#     vae.to(device, dtype=weight_dtype)  
#     text_encoder.to(device, dtype=weight_dtype)  
#     unet.to(device, dtype=weight_dtype)  
  
#     for step in range(num_steps):  
#         # 每次迭代开始时恢复到干净状态  
#         unet.load_state_dict(unet_clean_state)  
#         text_encoder.load_state_dict(text_encoder_clean_state)  
          
#         # 添加噪声到模型参数（用于前向传播）  
#         if noise_strength > 0:    
#             add_noise_to_model(unet, noise_strength)    
#             add_noise_to_model(text_encoder, noise_strength)  
          
#         unet.train()  
#         text_encoder.train()  
  
#         step_data = train_dataset[step % len(train_dataset)]  
#         pixel_values = torch.stack([step_data["instance_images"], step_data["class_images"]]).to(  
#             device, dtype=weight_dtype  
#         )  
#         input_ids = torch.cat([step_data["instance_prompt_ids"], step_data["class_prompt_ids"]], dim=0).to(device)  
  
#         latents = vae.encode(pixel_values).latent_dist.sample()  
#         latents = latents * vae.config.scaling_factor  
  
#         noise = torch.randn_like(latents)  
#         bsz = latents.shape[0]  
#         timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (bsz,), device=latents.device)  
#         timesteps = timesteps.long()  
  
#         noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)  
#         encoder_hidden_states = text_encoder(input_ids)[0]  
#         model_pred = unet(noisy_latents, timesteps, encoder_hidden_states).sample  
  
#         if noise_scheduler.config.prediction_type == "epsilon":  
#             target = noise  
#         elif noise_scheduler.config.prediction_type == "v_prediction":  
#             target = noise_scheduler.get_velocity(latents, noise, timesteps)  
#         else:  
#             raise ValueError(f"Unknown prediction type {noise_scheduler.config.prediction_type}")  
  
#         if args.with_prior_preservation:  
#             model_pred, model_pred_prior = torch.chunk(model_pred, 2, dim=0)  
#             target, target_prior = torch.chunk(target, 2, dim=0)  
#             instance_loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean")  
#             prior_loss = F.mse_loss(model_pred_prior.float(), target_prior.float(), reduction="mean")  
#             loss = instance_loss + args.prior_loss_weight * prior_loss  
#         else:  
#             loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean")  
  
#         # 计算梯度（基于带噪声的模型）  
#         loss.backward()  
          
#         # 将梯度应用到干净模型参数  
#         # 注意：由于我们在每次迭代开始时恢复了干净状态，  
#         # 梯度会自动应用到当前的模型参数（即干净参数）  
#         torch.nn.utils.clip_grad_norm_(params_to_optimize, 1.0, error_if_nonfinite=True)  
#         optimizer.step()  
#         optimizer.zero_grad()  
          
#         # 更新干净状态字典以保存参数更新  
#         unet_clean_state = copy.deepcopy(unet.state_dict())  
#         text_encoder_clean_state = copy.deepcopy(text_encoder.state_dict())  
          
#         print(f"Step #{step}, loss: {loss.detach().item()}")  
  
#     return [unet, text_encoder]

# def pgd_attack(
#     args,
#     models,
#     tokenizer,
#     noise_scheduler,
#     vae,
#     data_tensor: torch.Tensor,
#     original_images: torch.Tensor,
#     target_tensor: torch.Tensor,
#     num_steps: int,
#     current_step=0,  # 新增参数  
#     total_steps=1000,  # 新增参数  
# ):
#     """Return new perturbed data"""
    
#     unet, text_encoder = models
#     weight_dtype = torch.bfloat16
#     device = torch.device("cuda")

#     vae.to(device, dtype=weight_dtype)
#     text_encoder.to(device, dtype=weight_dtype)
#     unet.to(device, dtype=weight_dtype)

#     # vkeilo add it for noise add
#     progress = current_step / total_steps  
#     if progress < args.robust_start:  
#         noise_strength = 0.0  
#     elif progress < args.robust_fixed:  
#         # 线性增长  
#         linear_progress = (progress - args.robust_start) / (args.robust_fixed - args.robust_start)  
#         noise_strength = linear_progress * args.sigma  
#     else:  
#         noise_strength = args.sigma  
#     # if noise_strength > 0:  
#     #     add_noise_to_model(unet, noise_strength)  
#     #     add_noise_to_model(text_encoder, noise_strength)

#     perturbed_images = data_tensor.detach().clone()
#     perturbed_images.requires_grad_(True)

#     input_ids = tokenizer(
#         args.instance_prompt,
#         truncation=True,
#         padding="max_length",
#         max_length=tokenizer.model_max_length,
#         return_tensors="pt",
#     ).input_ids.repeat(len(data_tensor), 1)

#     # 在函数开始处备份模型状态  
#     unet_state_dict = copy.deepcopy(unet.state_dict())  
#     text_encoder_state_dict = copy.deepcopy(text_encoder.state_dict())  
#     for step in range(num_steps):  
#         # 恢复到原始状态  
#         unet.load_state_dict(unet_state_dict)  
#         text_encoder.load_state_dict(text_encoder_state_dict)  
#         # 添加噪声  
#         if noise_strength > 0:    
#             add_noise_to_model(unet, noise_strength)    
#             add_noise_to_model(text_encoder, noise_strength)  

#         perturbed_images.requires_grad = True
#         latents = vae.encode(perturbed_images.to(device, dtype=weight_dtype)).latent_dist.sample()
#         latents = latents * vae.config.scaling_factor

#         # Sample noise that we'll add to the latents
#         noise = torch.randn_like(latents)
#         bsz = latents.shape[0]
#         # Sample a random timestep for each image
#         timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (bsz,), device=latents.device)
#         timesteps = timesteps.long()
#         # Add noise to the latents according to the noise magnitude at each timestep
#         # (this is the forward diffusion process)
#         noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

#         # Get the text embedding for conditioning
#         encoder_hidden_states = text_encoder(input_ids.to(device))[0]

#         # Predict the noise residual
#         model_pred = unet(noisy_latents, timesteps, encoder_hidden_states).sample

#         # Get the target for loss depending on the prediction type
#         if noise_scheduler.config.prediction_type == "epsilon":
#             target = noise
#         elif noise_scheduler.config.prediction_type == "v_prediction":
#             target = noise_scheduler.get_velocity(latents, noise, timesteps)
#         else:
#             raise ValueError(f"Unknown prediction type {noise_scheduler.config.prediction_type}")

#         unet.zero_grad()
#         text_encoder.zero_grad()
#         loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean")

#         # target-shift loss
#         if target_tensor is not None:
#             xtm1_pred = torch.cat(
#                 [
#                     noise_scheduler.step(
#                         model_pred[idx : idx + 1],
#                         timesteps[idx : idx + 1],
#                         noisy_latents[idx : idx + 1],
#                     ).prev_sample
#                     for idx in range(len(model_pred))
#                 ]
#             )
#             xtm1_target = noise_scheduler.add_noise(target_tensor, noise, timesteps - 1)
#             loss = loss - F.mse_loss(xtm1_pred, xtm1_target)

#         loss.backward()

#         alpha = args.pgd_alpha
#         eps = args.pgd_eps
#         # print(f"max: {perturbed_images.max().item()},min: {perturbed_images.min().item()},eps: {eps}")
#         adv_images = perturbed_images + alpha * perturbed_images.grad.sign()
#         eta = torch.clamp(adv_images - original_images, min=-eps, max=+eps)
#         perturbed_images = torch.clamp(original_images + eta, min=-1, max=+1).detach_()
#         print(f"PGD loss - step {step}, loss: {loss.detach().item()}")
#     return perturbed_images


def main(args):
    logging_dir = Path(args.output_dir, args.logging_dir)

    accelerator = Accelerator(
        mixed_precision=args.mixed_precision,
        log_with=args.report_to,
        # logging_dir=logging_dir,
    )

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)
    if accelerator.is_local_main_process:
        datasets.utils.logging.set_verbosity_warning()
        transformers.utils.logging.set_verbosity_warning()
        diffusers.utils.logging.set_verbosity_info()
    else:
        datasets.utils.logging.set_verbosity_error()
        transformers.utils.logging.set_verbosity_error()
        diffusers.utils.logging.set_verbosity_error()

    if args.seed is not None:
        set_seed(args.seed)
    # Generate class images if prior preservation is enabled.
    if args.with_prior_preservation:
        class_images_dir = Path(args.class_data_dir)
        if not class_images_dir.exists():
            class_images_dir.mkdir(parents=True)
        cur_class_images = len(list(class_images_dir.iterdir()))

        if cur_class_images < args.num_class_images:
            torch_dtype = torch.float16 if accelerator.device.type == "cuda" else torch.float32
            if args.mixed_precision == "fp32":
                torch_dtype = torch.float32
            elif args.mixed_precision == "fp16":
                torch_dtype = torch.float16
            elif args.mixed_precision == "bf16":
                torch_dtype = torch.bfloat16
            pipeline = DiffusionPipeline.from_pretrained(
                args.pretrained_model_name_or_path,
                torch_dtype=torch_dtype,
                safety_checker=None,
                revision=args.revision,
            )
            pipeline.set_progress_bar_config(disable=True)

            num_new_images = args.num_class_images - cur_class_images
            logger.info(f"Number of class images to sample: {num_new_images}.")

            sample_dataset = PromptDataset(args.class_prompt, num_new_images)
            sample_dataloader = torch.utils.data.DataLoader(sample_dataset, batch_size=args.sample_batch_size)

            sample_dataloader = accelerator.prepare(sample_dataloader)
            pipeline.to(accelerator.device)

            for example in tqdm(
                sample_dataloader,
                desc="Generating class images",
                disable=not accelerator.is_local_main_process,
            ):
                images = pipeline(example["prompt"]).images

                for i, image in enumerate(images):
                    hash_image = hashlib.sha1(image.tobytes()).hexdigest()
                    image_filename = class_images_dir / f"{example['index'][i] + cur_class_images}-{hash_image}.jpg"
                    image.save(image_filename)

            del pipeline
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    # import correct text encoder class
    text_encoder_cls = import_model_class_from_model_name_or_path(args.pretrained_model_name_or_path, args.revision)

    # Load scheduler and models
    text_encoder = text_encoder_cls.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="text_encoder",
        revision=args.revision,
    )
    unet = UNet2DConditionModel.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="unet", revision=args.revision
    )

    tokenizer = AutoTokenizer.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="tokenizer",
        revision=args.revision,
        use_fast=False,
    )

    noise_scheduler = DDPMScheduler.from_pretrained(args.pretrained_model_name_or_path, subfolder="scheduler")

    vae = AutoencoderKL.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="vae", revision=args.revision
    ).cuda()

    vae.requires_grad_(False)

    if not args.train_text_encoder:
        text_encoder.requires_grad_(False)

    if args.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True

    clean_data, _ = load_data(
        args.instance_data_dir_for_train,
        size=args.resolution,
        center_crop=args.center_crop,
    )
    # perturbed_data = load_data(
    #     args.instance_data_dir_for_adversarial,
    #     size=args.resolution,
    #     center_crop=args.center_crop,
    # )
    # vkeilo add it for batch attack(>4)
    perturbed_data_all,img_names_all = load_data(
        args.instance_data_dir_for_adversarial,
        size=args.resolution,
        center_crop=args.center_crop,
    )
    print(img_names_all)
    batch_size = 4
    # max_batch = 2
    now_batch = 0
    num_batches = (len(perturbed_data_all) + batch_size - 1) // batch_size
    for batch_idx in range(num_batches):
        start_idx = batch_idx * batch_size
        end_idx = min((batch_idx + 1) * batch_size, len(perturbed_data_all))
        perturbed_data = perturbed_data_all[start_idx:end_idx].clone()

        original_data = perturbed_data.clone()
        original_data.requires_grad_(False)

        if args.enable_xformers_memory_efficient_attention:
            if is_xformers_available():
                unet.enable_xformers_memory_efficient_attention()
            else:
                raise ValueError("xformers is not available. Make sure it is installed correctly")

        target_latent_tensor = None
        if args.target_image_path is not None:
            target_image_path = Path(args.target_image_path)
            assert target_image_path.is_file(), f"Target image path {target_image_path} does not exist"

            target_image = Image.open(target_image_path).convert("RGB").resize((args.resolution, args.resolution))
            target_image = np.array(target_image)[None].transpose(0, 3, 1, 2)

            target_image_tensor = torch.from_numpy(target_image).to("cuda", dtype=torch.float32) / 127.5 - 1.0
            target_latent_tensor = (
                vae.encode(target_image_tensor).latent_dist.sample().to(dtype=torch.bfloat16) * vae.config.scaling_factor
            )
            target_latent_tensor = target_latent_tensor.repeat(len(perturbed_data), 1, 1, 1).cuda()

        f = [unet, text_encoder]
        for i in range(args.max_train_steps):
            # 1. f' = f.clone()
            f_sur = copy.deepcopy(f)
            f_sur = train_one_epoch(
                args,
                f_sur,
                tokenizer,
                noise_scheduler,
                vae,
                clean_data,
                args.max_f_train_steps,
                current_step=i,  # 新增  
                total_steps=args.max_train_steps,  # 新增 
                num_weight_samples=1,
            )
            perturbed_data = pgd_attack(
                args,
                f_sur,
                tokenizer,
                noise_scheduler,
                vae,
                perturbed_data,
                original_data,
                target_latent_tensor,
                args.max_adv_train_steps,
                current_step=i,  # 新增  
                total_steps=args.max_train_steps,  # 新增 
                num_weight_samples=args.num_weight_samples_when_delta
            )
            f = train_one_epoch(
                args,
                f,
                tokenizer,
                noise_scheduler,
                vae,
                perturbed_data,
                args.max_f_train_steps,
                current_step=i,  # 新增  
                total_steps=args.max_train_steps,  # 新增 
                num_weight_samples=args.num_weight_samples_when_theta
            )

            if (i + 1) % 100 == 0:
                # ====== 新增：分别统计并打印 NaN/Inf 占比 ======
                def print_nan_ratio(model, name, step):
                    total_params = 0
                    nan_params = 0
                    inf_params = 0
                    with torch.no_grad():
                        for p in model.parameters():
                            if not p.data.dtype.is_floating_point:
                                continue
                            t = p.data
                            total_params += t.numel()
                            nan_params += torch.isnan(t).sum().item()
                            inf_params += torch.isinf(t).sum().item()
                    nan_ratio = nan_params / max(total_params, 1) * 100
                    inf_ratio = inf_params / max(total_params, 1) * 100
                    print(f"[Step {step:04d}] {name}: "
                        f"NaN {nan_params:,}/{total_params:,} ({nan_ratio:.4f}%), "
                        f"Inf {inf_params:,}/{total_params:,} ({inf_ratio:.4f}%)")

                # 分别检查两个模型
                print_nan_ratio(f[0], "UNet", i + 1)
                print_nan_ratio(f[1], "TextEncoder", i + 1)
            if (i + 1) % args.checkpointing_iterations == 0:
                save_folder = f"{args.output_dir}/noise-ckpt/{i+1}"
                os.makedirs(save_folder, exist_ok=True)
                
                # vkeilo add it for model save
                # models_folder = f"{args.output_dir}/models/{i+1}"  
                # os.makedirs(models_folder, exist_ok=True)  
                # torch.save(f[0].state_dict(), os.path.join(models_folder, "unet.pth"))  
                # torch.save(f[1].state_dict(), os.path.join(models_folder, "text_encoder.pth")) 

                noised_imgs = perturbed_data.detach()
                
                # vkeilo add it for new sample save
                # 创建扩散管道用于图片生成  
                pipeline = DiffusionPipeline.from_pretrained(  
                    args.pretrained_model_name_or_path,  
                    unet=f[0],  # 使用当前训练的unet  
                    text_encoder=f[1],  # 使用当前训练的text_encoder  
                    torch_dtype=f[0].dtype,  
                    safety_checker=None,  
                    revision=args.revision,  
                )  
                pipeline.to("cuda")  
                
                # 生成4张图片  
                generated_images = pipeline(  
                    args.instance_prompt,  
                    num_images_per_prompt=4,  
                    num_inference_steps=100,  
                    guidance_scale=7.5  
                ).images  
                ag_gen_folder = f"{args.output_dir}/ag_gen_samples"
                os.makedirs(ag_gen_folder, exist_ok=True)
                # 保存生成的图片  
                for idx, image in enumerate(generated_images):  
                    save_path = os.path.join(ag_gen_folder, f"{i+1}_gen_{idx}.jpg")  
                    image.save(save_path)  
                
                print(f"Saved generated samples at step {i+1} to {ag_gen_folder}")  
                
                # 清理内存  
                del pipeline  


                # img_names = [
                #     str(instance_path).split("/")[-1]
                #     for instance_path in list(Path(args.instance_data_dir_for_adversarial).iterdir())
                # ]
                # vkeilo change it for batch attack(>4)
                img_names = img_names_all[start_idx:end_idx]
                for img_pixel, img_name in zip(noised_imgs, img_names):
                    save_path = os.path.join(save_folder, f"{i+1}_noise_{img_name}.png")
                    Image.fromarray(
                        (img_pixel * 127.5 + 128).clamp(0, 255).to(torch.uint8).permute(1, 2, 0).cpu().numpy()
                    ).save(save_path)
                print(f"Saved noise at step {i+1} to {save_folder}")


if __name__ == "__main__":
    args = parse_args()
    main(args)
    print("exp finished")
