# Auhor: Zhenyuan Dong, Hongyu Guo, Yuanchen Wang, Kehan Chen
# Date: 2024-Nov
# Acknowledgement: 
# This work were tested on Da-Fusion Paper, contain ton of original code from Da-Fusion. Findout more: https://github.com/brandontrabucco/da-fusion/
# This work contain partial code that involving help from ChatGPT, as well as the debug phase involving the AI tool. Sould far within the limit of 25%. 
# Design of the pipeline are original based on discussion and research by team.

import torch
import torch.nn.functional as F
from typing import List, Tuple, Optional
from PIL import Image
import numpy as np
import torchvision.transforms as transforms

class HallucinationDetector:
    
    def __init__(self, threshold): 
        threshold = 0.03 
        self.threshold = threshold 

    def calculate_image(self, img):
        
        mean = img.mean().item()
        std = img.std().item()
        minval = img.min().item()
        maxval= img.max().item()
        
        return {
            'mean': mean,
            'std': std,
            'min': minval,
            'max': maxval
        }
    
        
    def resize(self, predictions, reference_image, size=(512, 512)): # resize to 512x512 as default

        transform = transforms.Compose([ 
            transforms.Resize(size)
        ])

        if isinstance(reference_image, Image.Image):  # check if reference is image
            transform_pil = transforms.Compose([ 
                transforms.Resize(size), 
                transforms.ToTensor(), 
                transforms.Lambda(lambda x: x.float())
            ])
            reference_image = transform_pil(reference_image)
        else:
            reference_image = transform(reference_image.float()) 
            
            
        single_perd= predictions.float()
        single_perd = transform(single_perd)

      
        # overall printout stats
        reference_stats = self.calculate_image(reference_image)
        pred_stats = self.calculate_image(single_perd)

        pixel_mean_diff = (single_perd - reference_image).abs().mean().item()
        pixel_max_diff = (single_perd - reference_image).abs().max().item()
        
        overall = {}
        overall['reference_stats'] = reference_stats
        overall['single_prediction_stats'] = pred_stats
        overall['pixel_mean_diff'] = pixel_mean_diff
        overall['pixel_max_diff']= pixel_max_diff
        
        # mse score, variance 
        mse_score = F.mse_loss(single_perd, reference_image)
        pred_var = single_perd.var(dim=0).mean()
        
        
        # perceptual difference at 3 scales 
        scales = [1, 0.5, 0.25]
        perceptual_diffs = []
        
        for scale in scales:
            
            scale_wide = int(size[0] * scale)
            scale_hight = int(size[1] * scale)
            
            scale_full = scale_wide, scale_hight
        
            
            ref_scaled = F.interpolate(reference_image.unsqueeze(0), size = scale_full, mode='bilinear')
            pred_scaled = F.interpolate(single_perd.unsqueeze(0), size = scale_full, mode='bilinear')
            
            diff = F.l1_loss(pred_scaled, ref_scaled).item()

            perceptual_diffs.append(diff)
        
        
        perceptual_diffs_all= {}
  
  
        for scale, diff in zip(scales, perceptual_diffs):
            perceptual_diffs_all[f'scale_{scale}'] = diff
            
        sum_per_diffs = sum(perceptual_diffs)
            
        overall.update({
            'mse_score': mse_score.item(),
            'prediction_variance': pred_var.item(),
            'perceptual_diffs': perceptual_diffs_all
        })
        
        # add weight to MSE and prediction variance
        mse_weight = 0.1  
        var_weight = 0.9
        #weighted_base = (mse_weight * mse_score + var_weight * pred_var)
        weighted_base = mse_score
        hal_score = (weighted_base + sum_per_diffs) / (2 + len(scales))
        
        if mse_score > 0.01:
            print("MSE score too high, possible hallucination")
            mse_weight = 0.1  
            var_weight = 0.9
            #weighted_base = (mse_weight * mse_score + var_weight * pred_var)
            weighted_base = mse_score
            hal_score = (weighted_base + sum_per_diffs) / (2 + len(scales))
            is_hallucinated = True
            return hal_score.item(), is_hallucinated, overall
       
            
        if pred_var > 0.001:
            if mse_score < 0.0001:
                print("edge case1")
                is_hallucinated = False
                return hal_score.item(), is_hallucinated, overall
            print("variance too high, possible hallucination")
            mse_weight = 0.1  
            var_weight = 0.9
            #weighted_base = (mse_weight * mse_score + var_weight * pred_var)
            weighted_base = mse_score
            hal_score = (weighted_base + sum_per_diffs) / (2 + len(scales))
            is_hallucinated = True
            return hal_score.item(), is_hallucinated, overall
   
        
        
        else: 
            if hal_score < 0.01 or mse_score < 0.0001:
                mse_weight = 0.1  
                var_weight = 0.9
                #weighted_base = (mse_weight * mse_score + var_weight * pred_var)
                weighted_base = mse_score
                
                hal_score = (weighted_base + sum_per_diffs) / (2 + len(scales))
                
                is_hallucinated = False
                return hal_score.item(), is_hallucinated, overall
            
            is_hallucinated = hal_score >= self.threshold
        
        return hal_score.item(), is_hallucinated, overall


    def __call__(self, predictions, reference_image):
        return self.detect_hallucination_resize(predictions, reference_image)
