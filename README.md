# Contrastive Language-Image Pre-training (CLIP)

This repository contains the implementation and exploration of CLIP, a model that learns visual concepts from natural language descriptions.

## Overview
CLIP leverages a contrastive training approach where an image encoder and a text encoder are jointly trained to match image-text pairs.

## Features
- Implementation of image and text encoders
- Transformer-based architectures
- Multi-modal embeddings and alignment

## Setup
To run this project, clone the repository and install the necessary dependencies:
```bash
git clone https://github.com/uddithmachiraju/Contrastive-Language-Image-Pre-training-CLIP.git
cd Contrastive-Language-Image-Pre-training-CLIP
python train.py

## Model Accuracy and Results

### Accuracy Plot
![Model Accuracy](![image](https://github.com/user-attachments/assets/b8ed2f8b-ba5e-4e7c-b3b0-984192a2fc17)
)

The plot above shows the training and validation accuracy over the epochs.

### Sample Predictions
![Sample Predictions](![Screenshot 2024-11-10 213844](https://github.com/user-attachments/assets/d4eeef45-1fdf-445d-86ee-971dd6d30cfb)
![Screenshot 2024-11-10 213908](https://github.com/user-attachments/assets/b92976af-4344-4b2a-93f6-6d6b7ae2cc62)
) 

The image above illustrates the model's predictions on test samples. The model successfully identifies and matches the images with their respective descriptions.


