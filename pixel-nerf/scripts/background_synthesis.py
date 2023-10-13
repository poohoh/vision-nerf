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
def get_largest_mask(img_path):
    img = cv2.imread(img_path)

    # # resize image
    # img = cv2.resize(img, (img.shape[1] * 3, img.shape[0] * 3), interpolation=cv2.INTER_AREA)
    # cv2.imshow('resized_image', img)
    # cv2.waitKey()
    # cv2.destroyAllWindows()

    masks = pointrend.segment(img)
    if len(masks) == 0:
        print("WARNING: PointRend detected no objects in image skipping")
        return None

    # show and save masks
    # for i, mask in enumerate(masks):
    #     # cv2.imshow('mask', mask)
    #     # cv2.waitKey()
    #     # cv2.destroyAllWindows()
    #
    #     cv2.imwrite(os.path.join(args.root_dir, f'mask_{i}.png'), mask)

    # get largest mask
    max_area = 0
    largest_mask_index = 0
    for i, mask in enumerate(masks):
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            area = cv2.contourArea(max(contours, key=cv2.contourArea))
            if area > max_area:
                max_area = area
                largest_mask_index = i

    return masks[largest_mask_index]

# 흰색 배경을 투명하게
def make_background_trasparent(img_path):
    img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
    
    # RGBA로 변환
    if img.shape[2] == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2BGRA)
    
    # 흰색 픽셀을 투명하게 설정
    mask = (img[:, :, :3] == [255,255,255]).all(axis=2)
    img[mask, 3] = 0

    return img

# load background
def load_background():
    backgrounds = glob.glob(os.path.join(args.bg_dir, "*.png"))
    background = random.choice(backgrounds)  # random choice
    background = cv2.imread(background)

    return background

# 합성 데이터의 배경을 투명하게 만든 후 임의의 배경 이미지와 합성
def composite_with_transparent(img):
    # make background transparent
    foreground = make_background_trasparent(img)

    # load background
    background = load_background()

    # find and draw contours of transparent image
    contours, _ = cv2.findContours(foreground[:, :, 3], cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contour_img = np.zeros((*foreground.shape[:2], 3), dtype=np.uint8)
    cv2.drawContours(contour_img, contours, -1, (255, 255, 255), 3)

    # draw longest contour
    longest_contour = max(contours, key=cv2.contourArea)
    long_cnt_img = np.zeros((*foreground.shape[:2], 3), dtype=np.uint8)
    cv2.drawContours(long_cnt_img, longest_contour, -1, (255, 255, 255), 3)

    # cv2.imshow('longest_contour', long_cnt_img)
    # cv2.imshow('contours', contour_img)
    # cv2.imshow('transparent_img', transparent_img)
    # cv2.imshow('alpha', transparent_img[:, :, 3])
    # cv2.imshow('background', background)
    # cv2.waitKey()
    # cv2.destroyAllWindows()

    # composite
    for i in range(foreground.shape[1]):
        for j in range(foreground.shape[0]):
            if foreground[i, j, 3] != 0:
                background[i, j] = foreground[i, j, :3]

    result = []
    result.append(background)
    result.append(foreground)
    result.append(long_cnt_img)
    result.append(contour_img)

    return result

def composite_with_mask(object_img_path, mask):
    # image load
    object_img = cv2.imread(object_img_path)
    background_img = load_background()

    # object
    object_only = cv2.bitwise_and(object_img, object_img, mask=mask)

    # background
    inverse_mask = cv2.bitwise_not(mask)
    background_only = cv2.bitwise_and(background_img, background_img, mask=inverse_mask)

    result = []
    result.append(cv2.add(object_only, background_only))
    result.append(object_only)
    result.append(background_only)

    return result


# 차량과 배경 합성
def composite_background(path):
    images = glob.glob(os.path.join(path, '*.png'))

    for image in tqdm.tqdm(images):
        img_name = os.path.splitext(os.path.basename(image))[0]
        out_dir = os.path.join(path, '..', 'bg_synthesis', img_name)
        os.makedirs(out_dir, exist_ok=True)

        # 1) get mask from segmentation
        mask = get_largest_mask(image)
        if mask is not None:
            mask_result, object_only, background_only  = composite_with_mask(image, mask)

            # save mask result
            cv2.imwrite(os.path.join(out_dir, f'{img_name}_mask_result.png'), mask_result)
            cv2.imwrite(os.path.join(out_dir, f'{img_name}_object_only.png'), object_only)
            cv2.imwrite(os.path.join(out_dir, f'{img_name}_background_only.png'), background_only)
            cv2.imwrite(os.path.join(out_dir, f'{img_name}_mask.png'), mask)

        # 2) composite transparent background
        transparent_result, foreground, long_cnt_img, contour_img = composite_with_transparent(image)

        # save transparent result
        cv2.imwrite(os.path.join(out_dir, f'{img_name}_transparent_result.png'), transparent_result)
        cv2.imwrite(os.path.join(out_dir, f'{img_name}_transparent.png'), foreground)
        cv2.imwrite(os.path.join(out_dir, f'{img_name}_longest_contour.png'), long_cnt_img)
        cv2.imwrite(os.path.join(out_dir, f'{img_name}_contours.png'), contour_img)
        cv2.imwrite(os.path.join(out_dir, f'{img_name}_alpha.png'), foreground[:, :, 3])



# 하위 디렉토리를 모두 읽어서 내부에 이미지가 있으면 배경 합성
def read_files(path):
    dirs = [ f for f in os.listdir(path) if os.path.isdir(os.path.join(path, f))]

    if dirs:
        for dir in dirs:
            read_files(dir)
    else:
        composite_background(os.path.join(path))

def get_parser():
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
    parser.add_argument('--root_dir', default="C:/Users/KimJunha/Desktop/test/car", type=str)
    parser.add_argument('--bg_dir', default="C:/Users/KimJunha/Desktop/test/background", type=str)

    return parser.parse_args()

args = get_parser()

pointrend = PointRendWrapper(args.coco_class)

read_files(args.root_dir)