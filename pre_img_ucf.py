import os
import matplotlib.pyplot as plt
import cv2
import argparse
from PIL import Image
from preprocess_creBayess import generate_data
from scipy.io import loadmat

import os
import numpy as np

def cal_new_size(im_h, im_w, min_size, max_size):
    if im_h < im_w:
        if im_h < min_size:
            ratio = 1.0 * min_size / im_h
            im_h = min_size
            im_w = round(im_w*ratio)
        elif im_h > max_size:
            ratio = 1.0 * max_size / im_h
            im_h = max_size
            im_w = round(im_w*ratio)
        else:
            ratio = 1.0
    else:
        if im_w < min_size:
            ratio = 1.0 * min_size / im_w
            im_w = min_size
            im_h = round(im_h*ratio)
        elif im_w > max_size:
            ratio = 1.0 * max_size / im_w
            im_w = max_size
            im_h = round(im_h*ratio)
        else:
            ratio = 1.0
    return im_h, im_w, ratio


def generate_data(im_path, min_size, max_size):
    im = Image.open(im_path)
    im_w, im_h = im.size
    mat_path = im_path.replace('.jpg', '_ann.mat')
    points = loadmat(mat_path)['annPoints'].astype(np.float32)
    idx_mask = (points[:, 0] >= 0) * (points[:, 0] <= im_w) * (points[:, 1] >= 0) * (points[:, 1] <= im_h)
    points = points[idx_mask]
    im_h, im_w, rr = cal_new_size(im_h, im_w, min_size, max_size)
    im = np.array(im)
    if rr != 1.0:
        im = cv2.resize(np.array(im), (im_w, im_h), cv2.INTER_CUBIC)
        points = points * rr
    return Image.fromarray(im), points


def process_single_image(im_path, save_dir, min_size=512, max_size=2048, is_train=False):
    # Preprocess the image and get ground truth points
    im, points = generate_data(im_path, min_size, max_size)  # preprocessing
    
    if is_train:
        dis = find_dis(points)  # calculate distance for training
        points = np.concatenate((points, dis), axis=1)  # append distance to points
        
    # save_dir = '/home/eyecode-hien/GeneralizedLoss-Counting-Pytorch/generated_denmap_ucf'
    
    # Ensure the save directory exists
    os.makedirs(save_dir, exist_ok=True)
    
    # Create paths for saving the processed image and ground truth
    im_save_path = os.path.join(save_dir, os.path.basename(im_path).replace('.jpg', '.jpg'))
    gd_save_path = os.path.join(save_dir, os.path.basename(im_path).replace('.jpg', '.npy'))
    
    # Save the processed image and ground truth data
    im.save(im_save_path)
    np.save(gd_save_path, points)  # ground truth
    
    print(f"Processed image saved at: {im_save_path}")
    print(f"Ground truth data saved at: {gd_save_path}")

# img_dir = '/media/eyecode/data/hien/shanghaitech_data/ShanghaiTech_Crowd_Counting_Dataset/part_A_final/test_data/images'
# img_list = list()
# for img_name in os.listdir(img_dir):
#     img_path = os.path.join(img_dir, img_name)
#     img_list.append(img_path)
# img_list.sort()
    
# mat_dir = '/media/eyecode/data/hien/shanghaitech_data/ShanghaiTech_Crowd_Counting_Dataset/part_A_final/test_data/ground_truth'
# mat_list = list()
# for mat_name in os.listdir(mat_dir):
#     mat_path = os.path.join(mat_dir, mat_name)
#     mat_list.append(mat_path)
# mat_list.sort()




if __name__ == '__main__':
    # Parse arguments
    parser = argparse.ArgumentParser(description="Process images in a folder and save results.")
    parser.add_argument('--test_dir', type=str, required=True, help='Path to the folder containing images.')
    parser.add_argument('--save_dir', type=str, required=True, help='Directory to save processed images.')
    parser.add_argument('--min_size', type=int, default=512, help='Minimum size for resizing (default: 512).')
    parser.add_argument('--max_size', type=int, default=2048, help='Maximum size for resizing (default: 2048).')
    args = parser.parse_args()

    test_dir = args.test_dir
    save_dir = args.save_dir
    min_size = args.min_size
    max_size = args.max_size
    
    
    # test_dir = '/media/eyecode/data/hien/ucf_data/UCF-QNRF_ECCV18/Test'
    test_list = list()
    for test_name in os.listdir(test_dir):
        test_path = os.path.join(test_dir, test_name)
        test_list.append(test_path)
    test_list.sort()
    print(len(test_list))
    
    selected_images = [img for idx, img in enumerate(test_list) if idx % 2 == 0]
    print(f"Selected images (first 10): {selected_images[:10]}") 
    
    for i, image_path in enumerate(selected_images[:10]):
        print(f'Processing image {i + 1}/10 : {image_path}')
        try:
            process_single_image(image_path, save_dir= save_dir, min_size=512, max_size=2048, is_train=False)
            print("Processing completed successfully!")
        except Exception as e:
            print(f"Error during processing: {e}")


    print('All processes completed')

        
    # /media/eyecode/data/hien/ucf_data
    # /home/eyecode-hien/GeneralizedLoss-Counting-Pytorch/UCF-QNRF_ECCV18/Test
    
# save_dir = /home/eyecode-hien/GeneralizedLoss-Counting-Pytorch/generated_denmap_ucf
# test_dir = /media/eyecode/data/hien/ucf_data/UCF-QNRF_ECCV18/Test