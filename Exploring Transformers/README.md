# Specific Task 5: Exploring Transformers

### Dataset Description
A set of simulated strong gravitational lensing images with and without substructure. 

### Task 1 
Use a vision transformer method of our choice to build a robust and efficient model for binary classification or unsupervised anomaly detection on the provided dataset.

For this task , I trained following models
- vit_base_patch16_224
- vit_large_patch16_224
- swin_base_patch4_window7_224
- Ensamble of these models

### Results

<p align="center">
  <img src="https://user-images.githubusercontent.com/78193865/228515429-dfc1bb6d-1814-4266-9746-244a1eae2f52.png" width="350" title="hover text">
</p>

<p align="center">
  <img src="https://user-images.githubusercontent.com/78193865/228515480-ad83f6f5-bd78-48c8-a1e9-0a8b1e6a1649.png" width="350" title="hover text">
</p>

### Task 2

This task is the case of unsupervised anomaly detection where we have to train a vision transformer model to learn the distribution of the provided strong lensing image with no substructure.

For this unsupervised task, I trained `vit_base_patch16_224`, since the dataset is simple so the loss between the original image and the constructed image is very low.

### Results
<p align="center">
  <img src="https://user-images.githubusercontent.com/78193865/229313410-1786334d-76cd-4233-966e-d34721e795fc.png" width="350" title="hover text">
</p>
