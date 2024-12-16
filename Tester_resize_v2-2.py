import os
import torch
import numpy as np
from PIL import Image
from Hallucination_Detector import HallucinationDetector # change here for diff version
import torchvision.transforms as transforms
import json

def print_data(data, indent=""): # print data for diagnostics
    
    for key, value in data.items():
        
        if isinstance(value, dict):
            print(f"{indent}{key}: ")
            print_data(value, indent + "  ")
            
        else :
            print(f"{indent}{key}: {value}")
              

def test_hallucination_detector():
    
    detector = HallucinationDetector(threshold=0.03) 
    
    base_dir = "real-guidance-hd10" # file base directory
    test_images_dir = os.path.join(base_dir, "test_images") # test images directory


    gt_name = "real_bird.jpg" # ground truth image load in
    gt_path = os.path.join(test_images_dir, gt_name) 
    gt_img = Image.open(gt_path)
    gt_tensor = torch.from_numpy(np.array(gt_img)).permute(2,0,1).float() / 255.0


    image_files = [ x for x in os.listdir(test_images_dir) 
                   if x.lower().endswith(('.png', '.jpg', '.jpeg','.jpeg')) and x != gt_name]

    print("=" * 100)
    print("Detailed Analysis")
    print(f"Ground Truth image name: {gt_name}")
    print(f"Ground Truth image size: {gt_img.size}")
    print(f"** All images will be resized to the size of Ground Truth: {gt_img.size} **")
    print("=" * 100)

    results = []
    
    
        
    for img in image_files:
            
            
        print("=" * 50)
        print(f"Analyzing: {img}")
        print("-" * 50)
            
        img_path = os.path.join(test_images_dir, img)
            
        test_img = Image.open(img_path)
        test_tensor = torch.from_numpy(np.array(test_img)).permute(2,0,1).float() / 255.0
                
                
        score, is_hallucinated, detail_data = detector.resize(
            test_tensor, gt_tensor, size=(512, 512))
                
        print_data(detail_data)
                
        results.append({
            'filename': img,
            'score': score,
            'is_hallucinated': is_hallucinated,
            'size': test_img.size,
            'detail_data': detail_data
        })
                
        
        
        # Print summary table
    print("\nTable of Results (Lower score first):")
    print("=" * 100)
    print(f"{'Filename': <30} {'Score':<15} {'Hallucinated':<15} {'MSE':<15} {'Var':<10}")
    print("-" * 100)
        
    results.sort(key=lambda x: x['score'], reverse=False)
    for result in results:
                    
        print(f"{result['filename']:<30} "
              f"{result['score']:<15.8f} "
              
              f"{'Yes' if result['is_hallucinated'] else 'No':<15} "
              f"{result['detail_data']['mse_score']:<15.8f} "
              f"{result['detail_data']['prediction_variance']:<10.8f}")
            


if __name__ == "__main__":
    test_hallucination_detector()