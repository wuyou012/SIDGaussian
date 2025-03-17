#!/usr/bin/env python3
# filepath: /home/hezongqi/create_comparison_video.py

import os
import argparse
import cv2
import numpy as np
from tqdm import tqdm

def create_comparison_video(folder1, folder2, output_path, fps=30, width=None, height=None, add_text=True):
    """
    read the same files for two folders and create a video to compare the images in the two folders.
    
    parameters:
        folder1 (str): file path for the first folder
        folder2 (str): file path for the second folder
        output_path (str): output video path
        fps (int): video frame rate
        width (int): adjusted width of a single image, None means keep the original size
        height (int): adjusted height of a single image, None means keep the original size
        add_text (bool): whether to add the folder name as a label
    """


    files1 = set(os.listdir(folder1))
    files2 = set(os.listdir(folder2))
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp'}
    files1 = {f for f in files1 if os.path.splitext(f.lower())[1] in image_extensions}
    files2 = {f for f in files2 if os.path.splitext(f.lower())[1] in image_extensions}
    
    # find the common files
    common_files = sorted(list(files1.intersection(files2)))
    
    if not common_files:
        print("error, no common files found")
        return
    
    print(f"find {len(common_files)} common files")
    

    first_img1 = cv2.imread(os.path.join(folder1, common_files[0]))
    first_img2 = cv2.imread(os.path.join(folder2, common_files[0]))
    
    if first_img1 is None or first_img2 is None:
        print(f"error: can't read: {common_files[0]}")
        return
    

    img1_h, img1_w = first_img1.shape[:2]
    img2_h, img2_w = first_img2.shape[:2]
    
    if width is not None and height is not None:
        target_size = (width, height)
    else:
        target_w = max(img1_w, img2_w)
        target_h = max(img1_h, img2_h)
        target_size = (target_w, target_h)
    
    combined_width = target_size[0] * 2
    combined_height = target_size[1]
    
    output_root = os.path.dirname(output_path)
    if not os.path.exists(output_root):          
        os.mkdir(output_root) 
        
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter(output_path, fourcc, fps, (combined_width, combined_height))
    
    folder1_name = os.path.basename(os.path.normpath(folder1))
    folder1_name = '3DGS_30000'
    folder2_name = os.path.basename(os.path.normpath(folder2))
    folder2_name = 'Ours_10000'
    

    for img_file in tqdm(common_files, desc="generate compare video"):
        img1_path = os.path.join(folder1, img_file)
        img2_path = os.path.join(folder2, img_file)
        
        img1 = cv2.imread(img1_path)
        img2 = cv2.imread(img2_path)
        
        if img1 is None or img2 is None:
            print(f"warning: can;t read {img_file}, skip")
            continue

        img1_resized = cv2.resize(img1, target_size)
        img2_resized = cv2.resize(img2, target_size)
        
        # add label
        if add_text:
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 1
            font_thickness = 2
            text_color = (255, 255, 255)  # white
            bg_color = (0, 0, 0)  # background color
            

            cv2.putText(
                img1_resized, 
                folder1_name, 
                (10, 30), 
                font,
                font_scale, 
                text_color, 
                font_thickness
            )
            cv2.putText(
                img2_resized, 
                folder2_name, 
                (10, 30), 
                font, 
                font_scale, 
                text_color, 
                font_thickness
            )
        
        combined_img = np.hstack((img1_resized, img2_resized))

        if add_text:
            cv2.putText(
                combined_img, 
                img_file, 
                (10, combined_height - 10), 
                font, 
                0.5, 
                text_color, 
                1
            )
        

        video_writer.write(combined_img)

    video_writer.release()
    print(f"comparsion video saved to: {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='create comparison video for two folders')
    parser.add_argument('--folder1', type=str, required=True, help='file path for the first folder')
    parser.add_argument('--folder2', type=str, required=True, help='file path for the second folder')
    parser.add_argument('--output', type=str, default='comparison.mp4', help='output video path')
    parser.add_argument('--fps', type=int, default=30, help='video frame rate')
    parser.add_argument('--width', type=int, default=None, help='weight for adjust')
    parser.add_argument('--height', type=int, default=None, help='height for adjust')
    parser.add_argument('--no-text', action='store_true', help='whether to add text')
    
    args = parser.parse_args()
    
    create_comparison_video(
        args.folder1, 
        args.folder2, 
        args.output, 
        args.fps, 
        args.width, 
        args.height, 
        not args.no_text
    )