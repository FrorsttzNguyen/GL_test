import torch
import numpy as np
from datasets.single_crowd import Crowd
from models.vgg import vgg19
import argparse
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter



'''
    checkpoint_path: /media/eyecode/data/hien/GL_checkpoint/ucf_vgg19_ot_84.pth
'''

def parse_args():
    parser = argparse.ArgumentParser(description='Inference for a single preprocessed image')
    parser.add_argument('--data-dir', required=True, help='Path to the directory containing preprocessed image and npy')
    parser.add_argument('--checkpoint-path', required=True, help='Path to the trained model checkpoint')
    parser.add_argument('--save-density-path', default='./density_map.png', help='Path to save the density map image')
    parser.add_argument('--save-pth-path', default='./density_map.pth', help='Path to save the density map tensor')
    parser.add_argument('--device', default='cuda', help='Device to run inference (cuda or cpu)')
    args = parser.parse_args()
    return args

def save_density_map_as_image(density_map, save_path):
    """
    Save the predicted density map as an image.
    - Normalize the density map for visualization.
    - Use matplotlib to save the image with a color map.
    """
    plt.figure(figsize=(10, 10))
    plt.imshow(density_map, cmap='jet')  # Use 'jet' colormap
    plt.colorbar()
    plt.title('Predicted Density Map')
    plt.axis('off')
    plt.savefig(save_path)
    plt.close()
    print(f'Density map (image) saved at: {save_path}')

def save_density_map_as_tensor(density_map, save_path):
    """
    Save the predicted density map as a .pth file.
    """
    torch.save(torch.tensor(density_map), save_path)
    print(f'Density map (tensor) saved at: {save_path}')

if __name__ == '__main__':
    args = parse_args()

    # Load dataset for a single image using Crowd
    dataset = Crowd(args.data_dir, crop_size=512, downsample_ratio=8, method='val')
    print(f"Dataset contains: {dataset.im_list}")
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False)

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

    for img, gt_points, name in dataloader:
        img = img.to(args.device)
        with torch.no_grad():
            outputs = model(img)
            density_map = outputs.squeeze(0).squeeze(0).cpu().numpy()
            print(f"Inference completed for image: {name[0]}")
        
        density_map_smooth = gaussian_filter(density_map, sigma=2)
        print(f"Density map shape: {density_map_smooth.shape}")
        print(f"Min value: {density_map_smooth.min().item()}, Max value: {density_map_smooth.max().item()}")
        print(f"Sum of density map: {density_map_smooth.sum().item()}")
        print(f"Density map values:\n{density_map_smooth}")

        save_density_map_as_image(density_map_smooth, args.save_density_path)
        save_density_map_as_tensor(density_map_smooth, args.save_pth_path)

        if gt_points is not None:
            print(f"Ground truth points: {len(gt_points)}")
            print(f"Ground truth file: {name[0]}")

        break





