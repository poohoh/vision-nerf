"""
PointRend background removal + normalization for car images
(c) Alex Yu 2020
Usage: python [-S scale=4.37] [-s size=128]
outputs to *_mask.png, then *_mask_*.png (for other instances).
also writes _crop.txt
"""
import random
import sys
import argparse
import os
import os.path as osp
import json
from math import floor, ceil

ROOT_PATH = osp.dirname(os.path.abspath(__file__))  # preproc.py 파일 위치가 ROOT_PATH
POINTREND_ROOT_PATH = osp.join(ROOT_PATH, "detectron2", "projects", "PointRend")  # pointrend 루트 경로는 /detectron2/projects/PointRend


if not os.path.exists(POINTREND_ROOT_PATH):  # POINTREND 루트 경로가 없으면
    import urllib.request, zipfile

    print("Downloading minimal PointRend source package")
    zipfile_name = "pointrend_min.zip"

    # PointRend 패키지 다운로드
    urllib.request.urlretrieve(
        "https://alexyu.net/data/pointrend_min.zip", zipfile_name
    )

    # 압축 파일 해제
    with zipfile.ZipFile(zipfile_name) as zipfile:
        zipfile.extractall(ROOT_PATH)
    os.remove(zipfile_name)

sys.path.insert(0, POINTREND_ROOT_PATH)  # 해당 디렉토리 환경변수 지정

try:
    import detectron2
except:
    print(
        "Please install Detectron2 by selecting the right version",
        "from https://github.com/facebookresearch/detectron2/blob/master/INSTALL.md",
    )
# import PointRend project
import point_rend

from detectron2.utils.logger import setup_logger

setup_logger()

# import some common libraries
import numpy as np
import cv2
import torch
import tqdm
import glob

from matplotlib import pyplot as plt
import matplotlib.patches as patches

# import some common detectron2 utilities
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer, ColorMode
from detectron2.data import MetadataCatalog


def _crop_image(img, rect, const_border=False, value=0):
    """
    Image cropping helper
    """
    
    """
        잘라내려는 영역이 이미지 크기를 벗어나면 확장하는 함수
        rad > ccen 이면 그만큼 확장
    """
    
    x, y, w, h = rect

    # rect_main = [ccen - rad, rcen - rad, 2 * rad, 2 * rad]
    left = abs(x) if x < 0 else 0
    top = abs(y) if y < 0 else 0
    right = abs(img.shape[1] - (x + w)) if x + w >= img.shape[1] else 0
    bottom = abs(img.shape[0] - (y + h)) if y + h >= img.shape[0] else 0

    color = [value] * img.shape[2] if const_border else None
    new_img = cv2.copyMakeBorder(
        img,
        top,
        bottom,
        left,
        right,
        cv2.BORDER_CONSTANT if const_border else cv2.BORDER_REPLICATE,
        value=color,
    )  # 가장자리 추가
    if len(new_img.shape) == 2:
        new_img = new_img[..., None]  # 차원 추가

    x = x + left  # 음수(ccen < rad)이면 0, 양수(ccen > rad)이면 그대로
    y = y + top

    return new_img[y : (y + h), x : (x + w), :]


def _is_image_path(f):
    return (
        f.endswith(".jpg")
        or f.endswith(".jpeg")
        or f.endswith(".png")
        or f.endswith(".bmp")
        or f.endswith(".tiff")
        or f.endswith(".gif")
    )


class PointRendWrapper:
    def __init__(self, filter_class=-1):
        """
        :param filter_class output only intances of filter_class (-1 to disable). Note: class 0 is person.
        """
        self.filter_class = filter_class
        self.coco_metadata = MetadataCatalog.get("coco_2017_val")
        self.cfg = get_cfg()  # dectectron2의 default config 생성

        # Add PointRend-specific config

        point_rend.add_pointrend_config(self.cfg)

        # Load a config from file
        self.cfg.merge_from_file(
            os.path.join(
                POINTREND_ROOT_PATH,
                "configs/InstanceSegmentation/pointrend_rcnn_R_50_FPN_3x_coco.yaml",
            )
        )
        self.cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # set threshold for this model
        # Use a model from PointRend model zoo: https://github.com/facebookresearch/detectron2/tree/master/projects/PointRend#pretrained-models
        self.cfg.MODEL.WEIGHTS = "C:/Users/KimJunha/workspace/vision-nerf/pixel-nerf/checkpoints/model_final_edd263.pkl"
        self.predictor = DefaultPredictor(self.cfg)  # 간단한 end-to-end predictor 리턴

    def segment(self, im, out_name="", visualize=False):
        """
        Run PointRend
        :param out_name if set, writes segments B&W mask to this image file
        :param visualize if set, and out_name is set, outputs visualization rater than B&W mask
        """
        outputs = self.predictor(im)

        insts = outputs["instances"]  # 이미지에서 객체에 해당하는 것들
        # if self.filter_class != -1:
        #     insts = insts[insts.pred_classes == self.filter_class]  # 0 is person

        CLASSES = [2, 5, 7]  # 2,5,7에 해당하는 클래스들의 마스크 추출
        self.filter_classes = CLASSES
        mask = torch.zeros_like(insts.pred_classes, dtype=torch.bool)
        for cls in self.filter_classes:
            mask |= (insts.pred_classes == cls)
        insts = insts[mask]

        class_ids = insts.pred_classes.tolist()
        print("class IDs:", class_ids)
        
        if visualize:
            v = Visualizer(
                im[:, :, ::-1],
                self.coco_metadata,
                scale=1.2,
                instance_mode=ColorMode.IMAGE_BW,
            )

            point_rend_result = v.draw_instance_predictions(insts.to("cpu")).get_image()
            if out_name:
                cv2.imwrite(out_name + ".png", point_rend_result[:, :, ::-1])
            return point_rend_result[:, :, ::-1]
        else:
            im_names = []
            masks = []
            for i in range(len(insts)):
                mask = insts[i].pred_masks.to("cpu").permute(
                    1, 2, 0
                ).numpy() * np.uint8(255)
                if out_name:
                    im_name = out_name
                    if i:
                        im_name += "_" + str(i) + ".png"
                    else:
                        im_name += ".png"
                    im_names.append(im_name)
                    cv2.imwrite(im_name, mask)
                masks.append(mask)
            if out_name:
                with open(out_name + ".json", "w") as fp:
                    json.dump({"files": im_names}, fp)
            return masks


# 이미지에서 생성되는 여러 마스크 중 가장 큰 마스크 리턴
def get_largest_mask(im):
    masks = pointrend.segment(im)
    if len(masks) == 0:
        print("WARNING: PointRend detected no objects in image skipping")
        return

    max_area = 0
    largest_mask_index = 0
    for i, mask in enumerate(masks):
        _, contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            area = cv2.contourArea(max(contours, key=cv2.contourArea))
            if area > max_area:
                max_area = area
                largest_mask_index = i

    return masks[largest_mask_index]

# 차량과 배경 합성
def composite_bg(path):
    OUTPUT_DIR = os.path.join(path, '..', 'bg_synthesis')
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    images = glob.glob(os.path.join(path, '*.png'))
    backgrounds = glob.glob(os.path.join(BG_DIR, "*.png"))

    for image in images:
        object_im = cv2.imread(image)
        mask = get_largest_mask(object_im)
        if not mask:
            continue
        background = random.choice(backgrounds)  # 여러 배경 중에 랜덤으로 선택

        # 마스크 저장
        img_no_ext = osp.splitext(os.path.basename(image))[0]
        cv2.imwrite(os.path.join(OUTPUT_DIR, f'{img_no_ext}_mask.png', mask))

        # 마스크를 사용하여 객체 이미지에서 객체 추출
        object_only = cv2.bitwise_and(object_im, object_im, mask=mask)
        object_only_rgba = cv2.cvtColor(object_only, cv2.COLOR_BGR2BGRA)

        # 마스크를 이용하여 투명도 채널 설정
        object_only_rgba[:, :, 3] = mask
        
        # 배경 이미지에 객체 이미지 합성
        x, y = (object_only.shape[1] // 2, object_only.shape[0] // 2)  # 차량이 배경에서 위치할 포지션
        overlay_area = background[y:y+object_only.shape[0], x:x+object_only.shape[1]]

        background[y:y+object_only.shape[0], x:x+object_only.shape[1]] = cv2.addWeighted(overlay_area, 1, object_only_rgba, 1, 0)

        save_path = os.path.join(OUTPUT_DIR, os.path.basename(image))
        cv2.imwrite(save_path, background)


# 하위 디렉토리를 모두 읽어서 내부에 이미지가 있으면 합성
def read_files(path):
    dirs = [ f for f in os.listdir(path) if os.path.isdir(os.path.join(path, f))]

    if dirs:
        for dir in dirs:
            read_files(dir)
    else:
        composite_bg(os.path.join(path))



ROOT_DIR = "C:/Users/KimJunha/Desktop/test/car"
BG_DIR = "C:/Users/KimJunha/Desktop/test/background"



parser = argparse.ArgumentParser()
parser.add_argument(
    "--coco_class",
    type=int,
    default=2,
    help="COCO class wanted (0 = human, 2 = car, 5 = bus, 7 = truck)",
)
parser.add_argument(
    "--size",
    "-s",
    type=int,
    default=128,
    help="output image side length (will be square)",
)
parser.add_argument(
    "--scale",
    "-S",
    type=float,
    default=4.37,
    help="bbox scaling rel minor axis of fitted ellipse. "
    + "Will take max radius from this and major_scale.",
)
parser.add_argument(
    "--major_scale",
    "-M",
    type=float,
    default=0.8,
    help="bbox scaling rel major axis of fitted ellipse. "
    + "Will take max radius from this and major_scale.",
)
parser.add_argument(
    "--const_border",
    action="store_true",
    help="constant white border instead of replicate pad",
)
parser.add_argument(
    "--rate",
    type=float,
    default=None,
    help="calculate radius using this rate"
)
args = parser.parse_args()

pointrend = PointRendWrapper(args.coco_class)

read_files(ROOT_DIR)