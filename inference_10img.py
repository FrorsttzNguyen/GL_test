import os
import torch
import time
import numpy as np
import argparse
import matplotlib.pyplot as plt
from memory_check import check_memory, check_cpu_memory
from datasets.single_crowd import Crowd
from models.vgg import vgg19
from scipy.ndimage import gaussian_filter


def parse_args():
    parser = argparse.ArgumentParser(description='Inference for multiple preprocessed images')
    parser.add_argument('--data_dir', required=True, help='Path to the directory containing preprocessed images and npy files')
    parser.add_argument('--checkpoint_path', required=True, help='Path to the trained model checkpoint')
    parser.add_argument('--save_density_dir', help='Directory to save density map images')
    parser.add_argument('--save_pth_dir', help='Directory to save density map tensors')
    parser.add_argument('--device', default='cuda', help='Device to run inference (cuda or cpu)')
    args = parser.parse_args()
    return args


def save_density_map_as_image(density_map, point_map, save_path):
    """
    Save the predicted density map as an image and calculate the estimated number of people.
    - Normalize the density map for visualization.
    - Use matplotlib to save the image with a color map.
    - Sum the values of the density map to estimate the number of people.
    """
    estimated_people_count = density_map.sum().item()
    
    # Load ground truth points from point map
    if os.path.exists(point_map):
        gt_points = np.load(point_map)
        ground_truth_count = len(gt_points)
    else:
        gt_points = None
        ground_truth_count = 0
        print(f"Ground truth point map not found: {point_map}")

    # Visualize and save the density map as an image
    plt.figure(figsize=(10, 10))
    plt.imshow(density_map, cmap='jet')  # Use 'jet' colormap
    plt.colorbar()
    plt.title(f'Predicted Density Map\nEstimated People Count: {estimated_people_count:.2f} | Ground Truth: {ground_truth_count}')
    plt.axis('off')
    plt.savefig(save_path)
    plt.close()
    
    print(f'Density map (image) saved at: {save_path}')
    print(f'Estimated people count: {estimated_people_count}')
    print(f'Ground truth people count: {ground_truth_count}')


def save_density_map_as_tensor(density_map, save_path):
    """
    Save the predicted density map as a .pth file.
    """
    torch.save(torch.tensor(density_map), save_path)
    print(f'Density map (tensor) saved at: {save_path}')


if __name__ == '__main__':
    total_start_time = time.time()  
    args = parse_args()

    all_images = sorted(os.listdir(args.data_dir))

    selected_images = [os.path.join(args.data_dir, img) for idx, img in enumerate(all_images) if idx % 2 == 0]
    print(f"Selected images: {selected_images[:10]}") 

    selected_npy = [os.path.join(args.data_dir, npy) for idx, npy in enumerate(all_images) if idx % 2 != 0]
    print(f"Selected pointmap: {selected_npy[:10]}") 

    # Load the model
    model = vgg19()
    checkpoint = torch.load(args.checkpoint_path, map_location=args.device)
    model.load_state_dict(checkpoint, strict=False)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f'Total param: {total_params}')
    print(f'Trainable param : {trainable_params}')

    model.to(args.device)
    model.eval()
    
    # check_memory(model, args.device)
    check_cpu_memory()

    for idx, image_path in enumerate(selected_images[:10]):
        print(f"Processing image {idx + 1}/10: {image_path}")
        start_time = time.time()
        
        point_map_path = selected_npy[idx] 
        print(f"Using ground truth point map: {point_map_path}")
        
        dataset = Crowd(image_path, crop_size=512, downsample_ratio=8, method='val')
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False)

        for img, gt_points, name in dataloader:
            img = img.to(args.device)
            with torch.no_grad():
                # check_memory(model, args.device) #before inference
                check_cpu_memory()
                
                outputs = model(img)
                density_map = outputs.squeeze(0).squeeze(0).cpu().numpy()
                print(f"Inference completed for image: {name[0]}")
                
                # check_memory(model, args.device) # after inference
                check_cpu_memory()

            density_map_smooth = gaussian_filter(density_map, sigma= 1)
            print(f"Density map shape: {density_map_smooth.shape}")
            print(f"Min value: {density_map_smooth.min().item()}, Max value: {density_map_smooth.max().item()}")
            print(f"Sum of density map: {density_map_smooth.sum().item()}")


            image_name = os.path.basename(image_path).replace('.jpg', '')
            density_image_path = os.path.join(args.save_density_dir, f"{image_name}_density.png")
            density_pth_path = os.path.join(args.save_pth_dir, f"{image_name}_density.pth")

            save_density_map_as_image(density_map_smooth, point_map_path, density_image_path)
            save_density_map_as_tensor(density_map_smooth, density_pth_path)

            if gt_points is not None:
                print(f"Ground truth points: {len(gt_points)}")
                print(f"Ground truth file: {name[0]}")

            break 
        
        end_time = time.time() 
        elapsed_time = end_time - start_time
        print(f"Time taken for processing image {idx + 1}/10: {elapsed_time:.2f} seconds")
        
    total_end_time = time.time() 
    total_elapsed_time = total_end_time - total_start_time
    print(f"\nTotal time taken for the program: {total_elapsed_time:.2f} seconds")
    print("Processing for all 10 selected images completed!")

# folder_path = /home/eyecode-hien/GeneralizedLoss-Counting-Pytorch/real_process_img
# checkpoint_path = /media/eyecode/data/hien/GL_checkpoint/ucf_vgg19_ot_84.pth
# # save_dir = /home/eyecode-hien/GeneralizedLoss-Counting-Pytorch/generated_denmap_real
# /home/eyecode-hien/GeneralizedLoss-Counting-Pytorch/ucf_processed_img
# python3 inference_10img.py --data-dir /media/eyecode/data/hien/ucf_data/UCF-QNRF_ECCV18/processing --checkpoint-path /media/eyecode/data/hien/GL_checkpoint/ucf_vgg19_ot_84.pth