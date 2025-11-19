# Modified from:
#   DiT:  https://github.com/facebookresearch/DiT/blob/main/sample.py
import torch
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.set_float32_matmul_precision('high')
setattr(torch.nn.Linear, 'reset_parameters', lambda self: None)
setattr(torch.nn.LayerNorm, 'reset_parameters', lambda self: None)
from torchvision.utils import save_image

import time
import argparse
from tokenizer.tokenizer_image.xqgan_model import VQ_models
from autoregressive_31_adaLN.models.gpt_opt import GPT_models
from autoregressive_31_adaLN.models.generate import generate
import os

def main(args):
    # Setup PyTorch:
    #torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.set_grad_enabled(False)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # create and load model
    vq_model = VQ_models[args.vq_model](
        codebook_size=args.codebook_size,
        codebook_embed_dim=args.codebook_embed_dim)
    vq_model.to(device)
    vq_model.eval()
    checkpoint = torch.load(args.vq_ckpt, map_location="cpu")
    vq_model.load_state_dict(checkpoint["model"])
    del checkpoint
    print(f"image tokenizer is loaded")

    # create and load gpt model
    precision = {'none': torch.float32, 'bf16': torch.bfloat16, 'fp16': torch.float16}[args.precision]
    latent_size = args.image_size // args.downsample_size
    gpt_model = GPT_models[args.gpt_model](
        vocab_size=args.codebook_size,
        block_size=latent_size ** 2,
        num_classes=args.num_classes,
        cls_token_num=args.cls_token_num,
        model_type=args.gpt_type,
    ).to(device=device, dtype=precision)
    
    checkpoint = torch.load(args.gpt_ckpt, map_location="cpu")
    if args.from_fsdp: # fspd
        model_weight = checkpoint
    elif "model" in checkpoint:  # ddp
        model_weight = checkpoint["model"]
    elif "module" in checkpoint: # deepspeed
        model_weight = checkpoint["module"]
    elif "state_dict" in checkpoint:
        model_weight = checkpoint["state_dict"]
    else:
        raise Exception("please check model weight, maybe add --from-fsdp to run command")
    # if 'freqs_cis' in model_weight:
    #     model_weight.pop('freqs_cis')
    gpt_model.load_state_dict(model_weight, strict=False)
    gpt_model.eval()
    del checkpoint
    print(f"gpt model is loaded")

    if args.compile:
        print(f"compiling the model...")
        gpt_model = torch.compile(
            gpt_model,
            mode="reduce-overhead",
            fullgraph=True
        ) # requires PyTorch 2.0 (optional)
    else:
        print(f"no need to compile model in demo") 

    # Labels to condition the model with (feel free to change):
    #class_labels = [207, 360, 387, 974, 88, 979, 417, 279]
    #class_labels = [250, 250, 250, 250, 250, 250, 250, 250]
    #class_labels = torch.randperm(1000)[:32].tolist()
    #class_labels = [0, 1, 2, 0, 1, 2, 0, 1]
    #class_labels = [0, 0, 0, 0, 0, 0, 0, 0]


    save_root = "./Supplementary-XL-1.45-100"
    os.makedirs(save_root, exist_ok=True)

    # 指定生成的 class_id 列表
    class_ids = [284,973,980,387,979,2,985,974,933,928,780,437,90,250,52,972,130,270,88,284,688,14,484]

    # === 生成指定类别，每类 20 次，每次 2x3 的图 ===
    for cls in range(238,1000,1):
        print(f"\n=== Generating class {cls} ===")
        
        save_dir = os.path.join(save_root, f"class_{cls}")
        os.makedirs(save_dir, exist_ok=True)

        # 每类生成 20 次
        for i in range(10):
            class_labels = [cls] * 6           # 每次生成 6 张图
            c_indices = torch.tensor(class_labels, device=device)

            # --- GPT 采样 ---
            t1 = time.time()
            index_sample = generate(
                gpt_model, c_indices, latent_size ** 2,
                cfg_scale=args.cfg_scale, cfg_interval=args.cfg_interval,
                temperature=args.temperature, top_k=args.top_k,
                top_p=args.top_p, sample_logits=True,
            )
            print(f"Sampling done in {time.time()-t1:.2f}s")

            # --- reshape ---
            code_b = index_sample.view(6, latent_size, latent_size).long()

            # --- decode ---
            t2 = time.time()
            z_q = vq_model.quantize.embedding.weight[code_b]   # [6, H, W, C]
            z_q = z_q.permute(0, 3, 1, 2).contiguous()         # [6, C, H, W]
            samples = vq_model.decode(z_q)                     # [6, 3, 256, 256]
            samples = samples.detach().cpu()
            print(f"Decoding done in {time.time()-t2:.2f}s")

            # === 拼成 2x3 图并保存 ===
            save_path = os.path.join(save_dir, f"batch_{i}.png")
            save_image(samples, save_path, nrow=3, normalize=True, value_range=(-1,1))
            print(f"Saved {save_path}")





if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpt-model", type=str, choices=list(GPT_models.keys()), default="GPT-B")
    parser.add_argument("--gpt-ckpt", type=str, default=None)
    parser.add_argument("--gpt-type", type=str, choices=['c2i', 't2i'], default="c2i", help="class-conditional or text-conditional")
    parser.add_argument("--from-fsdp", action='store_true')
    parser.add_argument("--cls-token-num", type=int, default=1, help="max token number of condition input")
    parser.add_argument("--precision", type=str, default='bf16', choices=["none", "fp16", "bf16"]) 
    parser.add_argument("--compile", action='store_true', default=False)
    parser.add_argument("--vq-model", type=str, choices=list(VQ_models.keys()), default="VQ-16")
    parser.add_argument("--vq-ckpt", type=str, default=None, help="ckpt path for vq model")
    parser.add_argument("--codebook-size", type=int, default=8192, help="codebook size for vector quantization")
    parser.add_argument("--codebook-embed-dim", type=int, default=32, help="codebook dimension for vector quantization")
    parser.add_argument("--image-size", type=int, choices=[256, 384, 512], default=384)
    parser.add_argument("--downsample-size", type=int, choices=[8, 16], default=16)
    parser.add_argument("--num-classes", type=int, default=1000)
    parser.add_argument("--cfg-scale", type=float, default=4.0)
    parser.add_argument("--cfg-interval", type=float, default=-1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--top-k", type=int, default=2000,help="top-k value to sample with")
    parser.add_argument("--temperature", type=float, default=1.0, help="temperature value to sample with")
    parser.add_argument("--top-p", type=float, default=1.0, help="top-p value to sample with")
    args = parser.parse_args()
    main(args)