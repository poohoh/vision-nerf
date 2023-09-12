import glob
import os.path

import cv2
import tqdm

input_images = glob.glob("./input/*.png")

mean_size = 3

gaussian_size = 3
gaussian_sigma = 1

d = 3
sigmaColor = 100
sigmaSpace = 100

mean_dir = f'./mean_{mean_size}'
gaussian_dir = f'./gaussian_{gaussian_size}_{gaussian_sigma}'
bilateral_dir = f'./bilateral_{d}_{sigmaColor}_{sigmaSpace}'

os.makedirs(mean_dir, exist_ok=True)
os.makedirs(gaussian_dir, exist_ok=True)
os.makedirs(bilateral_dir, exist_ok=True)

for image_path in tqdm.tqdm(input_images):
    print(image_path)
    image_name = os.path.split(os.path.splitext(image_path)[0])[1]

    im = cv2.imread(image_path)

    mean_blur = cv2.blur(im, (mean_size, mean_size))
    cv2.imwrite(f'./{mean_dir}/mean_{image_name}.png', mean_blur)

    gaussian_blur = cv2.GaussianBlur(im, (gaussian_size, gaussian_size), gaussian_sigma)
    cv2.imwrite(f'./{gaussian_dir}/gaussian_{image_name}.png', gaussian_blur)

    bilateral_blur = cv2.bilateralFilter(im, 10, sigmaColor, sigmaSpace)
    cv2.imwrite(f'./{bilateral_dir}/bilateral_{image_name}.png', bilateral_blur)



    # cv2.imshow('image', im)
    # cv2.imshow('mean filter', mean_blur)
    # cv2.imshow('gaussian filter', gaussian_blur)
    # cv2.imshow('bilateral filter', bilateral_blur)

    # cv2.waitKey()
    # cv2.destroyAllWindows()