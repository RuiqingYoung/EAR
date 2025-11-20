## Getting Started
### Requirements
- Linux with Python ≥ 3.7
- PyTorch ≥ 2.1
- A100 GPUs

### Download VQVAE models
Create a folder named `pretrained_models` and download the pre-trained weights into this folder:  
[XQGAN-8192.pt](https://huggingface.co/qiuk6/RobustTok/resolve/main/XQGAN-8192.pt?download=true)  
*(from https://github.com/lxa9867/ImageFolder)*

### Pre-extract Discrete Codes of Training Images
Use the following command to extract discrete codes from training images.  
`n` can be set according to the number of available GPUs:

```bash
torchrun --nproc_per_node=n ./autoregressive_31_adaLN/train/extract_codes_c2i.py \
    --vq-ckpt ./pretrained_models/XQGAN-8192.pt \
    --data-path ./imagenet-1k/data/train \
    --code-path ./imagenet_code_c2i_flip_ten_crop \
    --ten-crop \
    --crop-range 1.1 \
    --image-size 256
```

### Train EAR Models with DDP

Before running, make sure to modify the following variables in your `.sh` script according to your cluster setup:  

- `nnodes`: total number of nodes  
- `nproc_per_node`: number of GPUs per node  
- `node_rank`: rank of the current node  
- `master_addr`: address of the master node  
- `master_port`: port of the master node  

```bash
torchrun --nproc_per_node=8 ./autoregressive_31_adaLN/train/train_c2i.py \
    --lr 2e-4 \
    --ckpt-every 60000 \
    --global-batch-size 512 \
    --num-classes 1000 \
    --cloud-save-path ./cloud_disk_31_adaLN \
    --no-local-save \
    --code-path ./imagenet_code_c2i_flip_ten_crop \
    --results-dir ./results_adaLN \
    --image-size 256 \
    --gpt-model GPT-B \
    --epochs 300

torchrun --nproc_per_node=8 ./autoregressive_31_adaLN/train/train_c2i.py \
    --lr 2e-4 \
    --ckpt-every 60000 \
    --global-batch-size 512 \
    --num-classes 1000 \
    --cloud-save-path ./cloud_disk_31_adaLN \
    --no-local-save \
    --code-path ./imagenet_code_c2i_flip_ten_crop \
    --results-dir ./results_adaLN \
    --image-size 256 \
    --gpt-model GPT-L \
    --epochs 300

torchrun --nnodes=$WORLD_SIZE --node_rank=$RANK --master_addr=$MASTER_ADDR --master_port=$MASTER_PORT --nproc_per_node=8 ./autoregressive_31_adaLN/train/train_c2i.py \
    --lr 2e-4 \
    --ckpt-every 60000 \
    --global-batch-size 512 \
    --num-classes 1000 \
    --cloud-save-path ./cloud_disk_31_adaLN \
    --no-local-save \
    --code-path ./imagenet_code_c2i_flip_ten_crop \
    --results-dir ./results_adaLN \
    --image-size 256 \
    --gpt-model GPT-XL \
    --epochs 300
```

### Sampling
Use 'n' according to the number of GPUs you want to use.

```bash
torchrun --nproc_per_node=n autoregressive_31_adaLN/sample/sample_c2i_ddp.py \
    --sample-dir "sample31adaLN" \
    --vq-ckpt ./pretrained_models/XQGAN-8192.pt \
    --gpt-ckpt ./cloud_disk_31_adaLN/EAR-adaLN-B.pt \
    --gpt-model GPT-B \
    --image-size 256 \
    --image-size-eval 256 \
    --num-classes 1000 \
    --cfg-scale 1.65 \
    --top-k 9000

torchrun --nproc_per_node=n autoregressive_31_adaLN/sample/sample_c2i_ddp.py \
    --sample-dir "sample31adaLN" \
    --vq-ckpt ./pretrained_models/XQGAN-8192.pt \
    --gpt-ckpt ./cloud_disk_31_adaLN/EAR-adaLN-L.pt \
    --gpt-model GPT-L \
    --image-size 256 \
    --image-size-eval 256 \
    --num-classes 1000 \
    --cfg-scale 1.45 \
    --top-k 9000

torchrun --nproc_per_node=n autoregressive_31_adaLN/sample/sample_c2i_ddp.py \
    --sample-dir "sample31adaLN" \
    --vq-ckpt ./pretrained_models/XQGAN-8192.pt \
    --gpt-ckpt ./cloud_disk_31_adaLN/EAR-adaLN-XL.pt \
    --gpt-model GPT-XL \
    --image-size 256 \
    --image-size-eval 256 \
    --num-classes 1000 \
    --cfg-scale 1.475 \
    --top-k 9000
```



### Evaluation
Before evaluation, please refer [evaluation readme](evaluations/c2i/README.md) to install required packages. 
```bash
python3 evaluations/c2i/evaluator.py VIRTUAL_imagenet256_labeled.npz samples/GPT-XL-last_version-size-256-size-256-VQ-16-topk-9000-topp-1.0-temperature-1.0-cfg-1.475-seed-0.npz
```