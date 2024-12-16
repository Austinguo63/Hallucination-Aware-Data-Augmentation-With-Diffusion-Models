from semantic_aug.datasets.coco import COCODataset
from semantic_aug.datasets.spurge import SpurgeDataset
from semantic_aug.datasets.imagenet import ImageNetDataset
from semantic_aug.datasets.pascal import PASCALDataset
from semantic_aug.augmentations.compose import ComposeParallel
from semantic_aug.augmentations.compose import ComposeSequential
from semantic_aug.augmentations.real_guidance import RealGuidance
from semantic_aug.augmentations.textual_inversion import TextualInversion
from diffusers import StableDiffusionPipeline
from itertools import product
from torch import autocast
from PIL import Image
import shutil  # 用于复制文件

from Hallucination_Detector import HallucinationDetector

from tqdm import tqdm
import os
import torch
import argparse
import numpy as np
import random


DATASETS = {
    "spurge": SpurgeDataset, 
    "coco": COCODataset, 
    "pascal": PASCALDataset,
    "imagenet": ImageNetDataset
}

COMPOSE = {
    "parallel": ComposeParallel,
    "sequential": ComposeSequential
}

AUGMENT = {
    "real-guidance": RealGuidance,
    "textual-inversion": TextualInversion
}

def score(image):
    scores = 0
    return scores
def print_diagnostics(diagnostics, indent=""):
    """Pretty print nested dictionary of diagnostics"""
    for key, value in diagnostics.items():
        if isinstance(value, dict):
            print(f"{indent}{key}:")
            print_diagnostics(value, indent + "  ")
        else:
            if isinstance(value, float):
                print(f"{indent}{key}: {value:.6f}")
            else:
                print(f"{indent}{key}: {value}")
def generate_aug(args, indices, train_dataset, rnd, filter=True):
    
    failed_idx = []
    gt_output_dir = os.path.join(args.out, "source_images")
    os.makedirs(gt_output_dir, exist_ok=True)
    copied_source_indices = set()
    results = []

    for idx, num in tqdm(indices, desc="Generating Augmentations"):

        image = train_dataset.get_image_by_idx(idx)
        label = train_dataset.get_label_by_idx(idx)
        
    
        metadata = train_dataset.get_metadata_by_idx(idx)

        if args.class_name is not None: 
            if metadata["name"] != args.class_name: continue

        image_g, label = aug(
            image, label, metadata)
        
        if filter:
            org_tensor = torch.from_numpy(np.array(image)).permute(2,0,1).float() / 255.0
            g_tensor = torch.from_numpy(np.array(image_g)).permute(2,0,1).float() / 255.0
            score, is_hallucinated, diagnostics = detector.detect_hallucination_resize(
                    [org_tensor], g_tensor, size=(512, 512))
            #打印数据
            print("\nDetailed Diagnostics:")
            print_diagnostics(diagnostics)
                
            results.append({
                    'filename': f'{metadata["name"]}_{idx}_{num}',
                    'score': score,
                    'is_hallucinated': is_hallucinated,
                    'size': image.size,
                    'diagnostics': diagnostics
            })
                
            if is_hallucinated:
                failed_idx.append((idx, num))
                name = metadata['name'].replace(" ", "_")
                pil_image, image = image_g, os.path.join(
                    args.out, "failed-"+str(rnd), f"{name}-{idx}-{num}.png")
                os.makedirs(os.path.dirname(image), exist_ok=True)
                pil_image.save(image)
                continue

        name = metadata['name'].replace(" ", "_")
        print(name,label)
        output_source_name = f"{name}-{idx}-source.jpg"
        output_source_path = os.path.join(gt_output_dir, output_source_name)
        if idx not in copied_source_indices:
        # 获取源头图片路径
            source_image_path = train_dataset.all_images[idx]

        # 复制源头图片到输出目录，并使用新的命名
            #shutil.copy(source_image_path, output_source_path)

        # 将当前索引加入已处理集合
            copied_source_indices.add(idx)

        pil_image, image = image_g, os.path.join(
            args.out, f"{name}-{idx}-{num}.png")

        #pil_image.save(image)


    print("\nSummary Results (sorted by score):")
    print("=" * 80)
    print(f"{'Filename':<30} {'Score':>10} {'Hallucinated':>12} {'MSE':>10} {'Var':>10}")
    print("-" * 80)
        
    results.sort(key=lambda x: x['score'], reverse=True)
    for result in results:
        print(f"{result['filename']:<30} "
                  f"{result['score']:>10.6f} "
                  f"{'Yes' if result['is_hallucinated'] else 'No':>12} "
                  f"{result['diagnostics']['mse_score']:>10.6f} "
                  f"{result['diagnostics']['prediction_variance']:>10.6f}")
    return failed_idx

if __name__ == "__main__":

    parser = argparse.ArgumentParser("Inference script")
    
    parser.add_argument("--out", type=str, default="real-guidance/")

    parser.add_argument("--model-path", type=str, default="CompVis/stable-diffusion-v1-4")
    parser.add_argument("--embed-path", type=str, default="erasure-tokens/pascal-tokens/pascal-0-8.pt")
    
    parser.add_argument("--dataset", type=str, default="pascal")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--examples-per-class", type=int, default=1)
    parser.add_argument("--num-synthetic", type=int, default=10)

    parser.add_argument("--prompt", type=str, default="a photo of a {name}")
    
    parser.add_argument("--aug", nargs="+", type=str, default=["real-guidance"], 
                        choices=["real-guidance", "textual-inversion"])

    parser.add_argument("--guidance-scale", nargs="+", type=float, default=[7.5])
    parser.add_argument("--strength", nargs="+", type=float, default=[0.5])

    parser.add_argument("--mask", nargs="+", type=int, default=[0], choices=[0, 1])
    parser.add_argument("--inverted", nargs="+", type=int, default=[0], choices=[0, 1])
    
    parser.add_argument("--probs", nargs="+", type=float, default=None)
    
    parser.add_argument("--compose", type=str, default="parallel", 
                        choices=["parallel", "sequential"])

    parser.add_argument("--class-name", type=str, default=None)
    
    parser.add_argument("--erasure-ckpt-path", type=str, default=None)

    parser.add_argument("--threshold", type=float, default=0.04, help="Threshold for hallucination detection")


    parser.add_argument("--stop-ratio", nargs="+", type=float, default=0.3)

    parser.add_argument("--releasing-rate", nargs="+", type=float, default=0.1)


    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    aug = COMPOSE[args.compose]([
        
        AUGMENT[aug](
            embed_path=args.embed_path, 
            model_path=args.model_path, 
            prompt=args.prompt, 
            strength=strength, 
            guidance_scale=guidance_scale,
            mask=mask, 
            inverted=inverted,
            erasure_ckpt_path=args.erasure_ckpt_path
        )

        for (aug, guidance_scale, 
            strength, mask, inverted) in zip(
            args.aug, args.guidance_scale, 
            args.strength, args.mask, args.inverted
        )

    ], probs=args.probs)

    detector = HallucinationDetector(t1=0, t2=50, threshold=args.threshold)
    

    train_dataset = DATASETS[
        args.dataset](split="train", seed=args.seed, 
                    examples_per_class=args.examples_per_class)
    failed_idx = list(product(range(len(train_dataset)), range(args.num_synthetic)))
    rnd = 0
    #while len(failed_idx) > len(train_dataset) * args.num_synthetic * args.stop_ratio:
        #failed_idx = generate_aug(args, failed_idx, train_dataset, rnd)
        #args.threshold *= 1 + args.releasing_rate
        #args.threshold = round(args.threshold, 3)
        #detector.threshold = args.threshold
        #rnd += 1
    generate_aug(args, failed_idx, train_dataset, rnd, filter=False)