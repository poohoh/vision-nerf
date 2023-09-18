import torch
import glob
import os
import imageio
import tqdm

def read_files(path):
    images = [file for file in os.listdir(path) if file.endswith('.png') or file.endswith('.jpg')]
    directories = [file for file in os.listdir(path) if os.path.isdir(os.path.join(path, file))]

    if images:
        generate_video(path)

    if directories:
        for directory in directories:
            read_files(os.path.join(path, directory))

def generate_video(path):
    input_images = glob.glob(os.path.join(path, '*.png'))
    input_images = [f for f in input_images]
    input_images = sorted(input_images)

    input_images = [imageio.v3.imread(image) for image in input_images]

    gif_config = {
        'loop': 0,
        'duration': 1000/6,
    }
    imageio.mimsave(os.path.join(path, 'output.gif'), input_images, format='gif', **gif_config)

    print(f'completed: {path}')

root_path = 'C:/Users/KimJunha/Desktop/work/synthetic data/vision nerf/real/128x128/CCTV/result/대각선 앞'

read_files(root_path)