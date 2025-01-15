import os
import cv2
import numpy as np
import argparse
from PIL import Image

def cal_new_size(im_h, im_w, min_size, max_size):
    if im_h < im_w:
        if im_h < min_size:
            ratio = 1.0 * min_size / im_h
            im_h = min_size
            im_w = round(im_w * ratio)
        elif im_h > max_size:
            ratio = 1.0 * max_size / im_h
            im_h = max_size
            im_w = round(im_w * ratio)
        else:
            ratio = 1.0
    else:
        if im_w < min_size:
            ratio = 1.0 * min_size / im_w
            im_w = min_size
            im_h = round(im_h * ratio)
        elif im_w > max_size:
            ratio = 1.0 * max_size / im_w
            im_w = max_size
            im_h = round(im_h * ratio)
        else:
            ratio = 1.0
    return im_h, im_w, ratio


def generate_image(im_path, min_size, max_size):
    im = Image.open(im_path)
    im_w, im_h = im.size
    im_h, im_w, rr = cal_new_size(im_h, im_w, min_size, max_size)
    im = np.array(im)
    if rr != 1.0:
        im = cv2.resize(np.array(im), (im_w, im_h), cv2.INTER_CUBIC)
    return Image.fromarray(im)


def process_single_image(im_path, save_dir, min_size=512, max_size=2048):
    # Preprocess the image
    im = generate_image(im_path, min_size, max_size)
    
    # Save the processed image with _processed.jpg suffix
    im_name, ext = os.path.splitext(os.path.basename(im_path))
    if ext.lower() in ['.jpg', '.jpeg']:
        im_save_path = os.path.join(save_dir, f"{im_name}_processed.jpg")
    else:
        im_save_path = os.path.join(save_dir, f"{im_name}_processed{ext}")
    
    im.save(im_save_path)
    print(f"Processed image saved at: {im_save_path}")



if __name__ == '__main__':
    # Parse arguments
    parser = argparse.ArgumentParser(description="Process images in a folder and save results.")
    parser.add_argument('--folder_path', type=str, required=True, help='Path to the folder containing images.')
    parser.add_argument('--save_dir', type=str, required=True, help='Directory to save processed images.')
    parser.add_argument('--min_size', type=int, default=512, help='Minimum size for resizing (default: 512).')
    parser.add_argument('--max_size', type=int, default=2048, help='Maximum size for resizing (default: 2048).')
    args = parser.parse_args()

    folder_path = args.folder_path
    save_dir = args.save_dir
    min_size = args.min_size
    max_size = args.max_size
    
    if not os.path.exists(folder_path):
        print('Folder not found: {folder_path}')
    else:
        for img_name in os.listdir(folder_path):
            img_path = os.path.join(folder_path, img_name)
            if os.path.isfile(img_path) and img_path.lower().endswith(('.jpg', '.jpeg', '.png')):
                try:
                    process_single_image(im_path= img_path, save_dir= save_dir, min_size= min_size, max_size= max_size)
                except Exception as e:
                    print(f'Error processing image {img_path}: {e}')
        print('All images processed')
        
        
# folder_path = /media/eyecode/data/hien/real_img
# # save_dir = /home/eyecode-hien/GeneralizedLoss-Counting-Pytorch/generated_denmap_real
