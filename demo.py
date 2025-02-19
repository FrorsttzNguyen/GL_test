import os
import torch
import time
import numpy as np
import argparse
import matplotlib.pyplot as plt
from OT_M.otm import den2seq    
from memory_check import check_memory, check_cpu_memory
from datasets.single_crowd import Crowd
from models.vgg import vgg19
from scipy.ndimage import gaussian_filter

# Load the model
model = vgg19()
checkpoint = torch.load('/media/eyecode/data/hien/GL_checkpoint/ucf_vgg19_ot_84.pth', map_location='cuda:1')
model.load_state_dict(checkpoint, strict=False)

total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

print(f'Total param: {total_params}')
print(f'Trainable param : {trainable_params}')

model.to('cuda:1')
model.eval()

all_images = sorted(os.listdir('/home/eyecode-hien/GeneralizedLoss-Counting-Pytorch/ucf_processed_img'))

selected_images = [os.path.join('/home/eyecode-hien/GeneralizedLoss-Counting-Pytorch/ucf_processed_img', img) for idx, img in enumerate(all_images) if idx % 2 == 0]
print(f"Selected images: {selected_images[:10]} \n") 

selected_npy = [os.path.join('/home/eyecode-hien/GeneralizedLoss-Counting-Pytorch/ucf_processed_img', npy) for idx, npy in enumerate(all_images) if idx % 2 != 0]
print(f"Selected pointmap: {selected_npy[:10]}") 

image_path = selected_images[0]
dataset = Crowd(image_path, crop_size=512, downsample_ratio=8, method='val')
dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False)

for img, gt_points, name in dataloader:
    img = img.to('cuda:1')
    print('img shape: ',img.shape)
    print('img type: ', type(img))
    imh, imw = img.shape[2:4]
    with torch.no_grad():
        # check_memory(model, args.device) #before inference
        check_cpu_memory()      
        outputs = model(img)
        density_map = outputs.squeeze(0).squeeze(0).cpu().numpy()
        print(f"Inference completed for image: {name[0]}")
        print(f'\n densitymap : {density_map}')
    
    density_map_smooth = gaussian_filter(density_map, sigma= 1)
    density_map_smooth_tensor = torch.from_numpy(density_map_smooth).float()
    density_map_smooth_tensor_norm = (density_map_smooth_tensor - density_map_smooth_tensor.min()) / (density_map_smooth_tensor.max() - density_map_smooth_tensor.min())
    
    dh, dw = density_map_smooth.shape
    scale_factor = imw/dw
    pseudo_test = torch.load('/home/eyecode-hien/GeneralizedLoss-Counting-Pytorch/OT_M/samples/pseudo.pth')
    print(f'scale_factor : {scale_factor}')
    print(f'scale factor type : {type(scale_factor)}')
    
    print(f'pseudo type : {type(pseudo_test)}')
    print(f'pseudo shape : {pseudo_test.shape}')
    print(f'\n pseudo :{pseudo_test}')
    print(f'density map type {type(density_map_smooth_tensor)}')
    print(f"Density map smooth shape: {density_map_smooth_tensor.shape}")
    print(f"Min value: {density_map_smooth_tensor.min().item()}, Max value: {density_map_smooth_tensor.max().item()}")
    print(f"Sum of density map: {density_map_smooth_tensor.sum().item()}")
    
    dot = den2seq(pseudo_test, scale_factor= 2.0, max_itern= 10,ot_scaling= 0.75)
    
    print(f'sum of dot(prediction) : {len(dot)}')

    
    
    
        