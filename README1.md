# Running Guidance

## Overview
This project provides a tool for generating augmentations under hallucination detection. The commands are showing below. For now, our project support COCO and Pascal dataset.

---

## How to Run Generation Code

Run the following command in the terminal to generate augmentations

```bash
python generate_augmentations_hd.py \
    --out <output_path> \
    --model-path <model_path> \
    --embed-path <embed_path> \
    --dataset <dataset_name> \
    --seed <random_seed> \
    --examples-per-class <examples_per_class> \
    --num-synthetic <num_synthetic_samples> \
    --prompt <prompt_template> \
    --aug <augmentation_methods> \
    --guidance-scale <guidance_scale_values> \
    --strength <strength_values> \
    --mask <mask_flag> \
    --inverted <inverted_flag> \
    --probs <custom_class_probabilities> \
    --compose <composition_method> \
    --class-name <target_class_name> \
    --erasure-ckpt-path <erasure_checkpoint_path> \
    --threshold <hallucination_threshold> \
    --stop-ratio <stop_ratio_values> \
    --releasing-rate <releasing_rate_values>
```
## How to Run downstream test

Run the following command to execute the downstream task. 

```bash
python train_classifier.py \
    --logdir <log_directory> \
    --model-path <model_path> \
    --prompt <prompt_template> \
    --synthetic-probability <synthetic_probability> \
    --synthetic-dir <input_images_direction> \
    --image-size <image_size> \
    --classifier-backbone <classifier_backbone> \
    --iterations-per-epoch <iterations_per_epoch> \
    --num-epochs <number_of_epochs> \
    --batch-size <batch_size> \
    --num-synthetic <number_of_synthetic_samples> \
    --num-trials <number_of_trials> \
    --examples-per-class <examples_per_class_list> \
    --embed-path <embed_path> \
    --dataset <dataset_name> \
    --aug <augmentation_methods> \
    --strength <strength_values> \
    --guidance-scale <guidance_scale_values> \
    --mask <mask_flag_list> \
    --inverted <inverted_flag_list> \
    --probs <custom_class_probabilities> \
    --compose <composition_method> \
    --erasure-ckpt-path <erasure_checkpoint_path> \
    --tokens-per-class <tokens_per_class> \
    --use-randaugment \
    --use-cutmix
```

