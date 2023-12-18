import glob
import os.path
import tqdm

import cv2

BG_DIR = "C:/Users/KimJunha/Desktop/test/test/bg_origin"

bgs = glob.glob(os.path.join(BG_DIR, '*.png'))
for f in tqdm.tqdm(bgs):
    bg = cv2.imread(f)

    resized = cv2.resize(
        bg, (128, 128), interpolation=cv2.INTER_AREA
    )

    img_name = os.path.splitext(os.path.basename(f))[0]

    save_path = os.path.join(BG_DIR, '..', 'background')
    os.makedirs(save_path, exist_ok=True)

    cv2.imwrite(os.path.join(save_path, f'{img_name}_resize.png'), resized)