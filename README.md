# NoisyQuant

An official implement of CVPR 2023 paper - NoisyQuant: Noisy Bias-Enhanced Post-Training Activation Quantization for Vision Transformers

## Todo list

* [X] simple implementation
* [X] full implementation

## Requirement

```bash
pip install timm
# We use torch==2.0.1+cu117 and timm==0.9.8
```

## How to run

Please have a look at [run.sh](https://github.com/kriskrisliu/NoisyQuant/blob/main/run.sh)

## Best practice

Let's check it out how NoisyQuant works.

1. Let's run a non-quantized vit

```bash
# Let's run a non-quantized vit
python validate.py /data/dataset/imagenet/val/ --model vit_base_patch16_224 --pretrained
```

which gives you results like:

```json
# ViT without quantization
{
    "model": "vit_base_patch16_224",
    "top1": 85.1,
    "top1_err": 14.9,
    "top5": 97.526,
    "top5_err": 2.474,
    "param_count": 86.57,
    "img_size": 224,
    "crop_pct": 0.9,
    "interpolation": "bicubic"
}
```

2. Let's see what happens (generally accuracy degradation) if we apply a vanillia quantization method:

```bash
python validate.py /data/dataset/imagenet/val/ --model vit_base_patch16_224 --pretrained --quant
```

which gives you 64.6, a big drop from 85.1:

```json
# vanillia quantization
{
    "model": "vit_base_patch16_224",
    "top1": 64.612,
    "top1_err": 35.388,
    "top5": 84.904,
    "top5_err": 15.096,
    "param_count": 86.57,
    "img_size": 224,
    "crop_pct": 0.9,
    "interpolation": "bicubic"
}
```

3. with NoisyQuant:

```bash
python validate.py /data/dataset/imagenet/val/ --model $MODEL --pretrained --quant --with_noisy_quant --calib_root /data/dataset/imagenet/train --calib_num 256 --percentile --search_mean --search_noisy
```

which gives you 83.28.

We just add some `random noise`.

```json
# NoisyQuant
{
    "model": "vit_base_patch16_224",
    "top1": 83.28,
    "top1_err": 16.72,
    "top5": 96.64,
    "top5_err": 3.36,
    "param_count": 86.57,
    "img_size": 224,
    "crop_pct": 0.9,
    "interpolation": "bicubic"
}
```

## Takeaway tips

Check `fast_quant.py` for more details. It is very easy to understand.
