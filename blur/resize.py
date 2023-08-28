import glob
import os.path
import cv2
import numpy as np

input_path = './input/*.jpg'
file_list = [file for file in glob.glob(input_path)]

img_resize_list = []

os.makedirs('./resize', exist_ok=True)

for f in file_list:
    img = cv2.imread(f)

    y, x = np.where(np.sum(img != [255, 255, 255], axis=-1))
    y1, y2, x1, x2 = max(y.min(), 0), min(y.max(), img.shape[0]), max(x.min(), 0), min(x.max(), img.shape[1])

    cropped = img[y1:y2, x1:x2]
    cv2.imshow('cropped', cropped)
    cv2.waitKey()
    cv2.destroyAllWindows()

    # img.shape -> (height, width, rgb)
    height = cropped.shape[0]
    width = cropped.shape[1]
    
    if height > width:
        new_height = 108
        new_width = int(width * 108 / height)
    else:
        new_width = 108
        new_height = int(height * 108 / width)

    resized = cv2.resize(cropped, (new_width, new_height), interpolation=cv2.INTER_AREA)

    cv2.imshow('resized', resized)
    cv2.waitKey()
    cv2.destroyAllWindows()

    # # 가로 크기가 1000보다 크면 1000으로
    # if width > 1000:
    #
    #     new_width = 1000
    #     # new_width = int(width/2)
    #     new_height = int(new_width * aspect_ratio)
    #
    #     img_resize = cv2.resize(img, dsize=(new_width, new_height), interpolation=cv2.INTER_AREA)
    # else:
    #     img_resize = img
    # elif width < 600:
    #     img_resize = cv2.resize(img, dsize=(new_width, new_height), interpolation=cv2.INTER_LANCZOS4)

    title, ext = os.path.splitext(f)
    title_without_path = os.path.split(title)[1]
    cv2.imwrite(os.path.join('./resize', title_without_path) + ext, resized)

    print('resize: ' + title)