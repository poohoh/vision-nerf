"""
PointRend background removal + normalization for car images
(c) Alex Yu 2020
Usage: python [-S scale=4.37] [-s size=128]
outputs to *_mask.png, then *_mask_*.png (for other instances).
also writes _crop.txt
"""
import sys
import argparse
import os
import os.path as osp
import json
from math import floor, ceil

ROOT_PATH = osp.dirname(os.path.abspath(__file__))  # preproc.py 파일 위치가 ROOT_PATH
POINTREND_ROOT_PATH = osp.join(ROOT_PATH, "detectron2", "projects", "PointRend")  # pointrend 루트 경로는 /detectron2/projects/PointRend
INPUT_DIR = osp.join(ROOT_PATH, "..", "input", "input")  # input에 입력 데이터
OUTPUT_DIR = os.path.join(INPUT_DIR, "..", 'preprocess')  # 전처리된 이미지 경로

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
    x, y, w, h = rect

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

    x = x + left
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
        self.cfg.MODEL.WEIGHTS = "checkpoints/model_final_edd263.pkl"
        self.predictor = DefaultPredictor(self.cfg)  # 간단한 end-to-end predictor 리턴

    def segment(self, im, out_name="", visualize=False):
        """
        Run PointRend
        :param out_name if set, writes segments B&W mask to this image file
        :param visualize if set, and out_name is set, outputs visualization rater than B&W mask
        """
        outputs = self.predictor(im)

        insts = outputs["instances"]
        if self.filter_class != -1:
            insts = insts[insts.pred_classes == self.filter_class]  # 0 is person
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


if __name__ == "__main__":
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
    args = parser.parse_args()

    pointrend = PointRendWrapper(args.coco_class)

    input_images = glob.glob(os.path.join(INPUT_DIR, "*"))
    input_images = [
        f
        for f in input_images
        if _is_image_path(f) and not f.endswith("_normalize.png")
    ]

    os.makedirs(INPUT_DIR, exist_ok=True)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    for image_path in tqdm.tqdm(input_images):
        print(image_path)
        im = cv2.imread(image_path)

        # 확장자를 제거한 이미지 파일 이름
        img_no_ext = os.path.split(os.path.splitext(image_path)[0])[1]

        # detectron2의 pointrend를 통해 mask 생성
        masks = pointrend.segment(im)
        if len(masks) == 0:
            print("WARNING: PointRend detected no objects in", image_path, "skipping")
            continue
        mask_main = masks[0]
        assert mask_main.shape[:2] == im.shape[:2]
        assert mask_main.shape[-1] == 1
        assert mask_main.dtype == "uint8"

        # 마스크, masked 저장
        for idx in range(len(masks)):
            mask = masks[idx]
            cv2.imwrite(os.path.join(OUTPUT_DIR, f"{img_no_ext}_mask_{idx}.jpg"), mask)

            mask_flt_temp = mask.astype(np.float32) / 255.0
            masked = im.astype(np.float32) * mask_flt_temp + 255 * (1.0 - mask_flt_temp)
            masked = masked.astype(np.uint8)
            cv2.imwrite(os.path.join(OUTPUT_DIR, f"{img_no_ext}_masked_{idx}.jpg"), masked)

        # mask is (H, W, 1) with values{0, 255}

        cnt, _ = cv2.findContours(mask_main, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)  # 첫 번째 마스크에서 contour를 찾음
        # 여러 개의 contour 중에서 가장 긴 것 추출
        cnt_length = [len(contour) for contour in cnt]
        longest_cnt = cnt[cnt_length.index(max(cnt_length))]

        # mask의 contour 그리기
        main_contour = np.zeros((*im.shape[:2], 3), dtype=np.uint8)
        cv2.drawContours(main_contour, cnt, -1, (0, 255, 0), 3)
        cv2.imwrite(os.path.join(OUTPUT_DIR, 'main_contour.png'), main_contour)
        try:

            ellipse = cv2.fitEllipse(longest_cnt)  # 찾아낸 contour에 타원을 적합시킴

            cen_pt = ellipse[0]  # 타원의 중심점 추출
            min_ax, max_ax = min(ellipse[1]), max(ellipse[1])  # 타원의 주축 길이 추출
            
            # contour와 타원 그리기
            imgvis = np.zeros((*im.shape[:2], 3), dtype=np.uint8)
            cv2.drawContours(imgvis, cnt, -1, (0, 255, 0), 3)
            cv2.circle(imgvis, (int(cen_pt[0]), int(cen_pt[1])), 7, (255, 255, 255), -1)  # 중심점 그리기
            imgvis = cv2.ellipse(imgvis, ellipse, (255, 0, 0), 2)
            cv2.imwrite(os.path.join(OUTPUT_DIR, 'vs.png'), imgvis)
            print(len(cnt), cnt[0].shape)
            print(cen_pt, min_ax)
        except Exception as ex:
            print('cv2.fitEllipse error: ', ex)
            continue


        # mask로부터 bounding box 찾기
        rows = np.any(mask_main, axis=1)
        cols = np.any(mask_main, axis=0)
        rnz = np.where(rows)[0]
        cnz = np.where(cols)[0]
        if len(rnz) == 0:
            cmin = rmin = 0
            cmax = mask_main.shape[-1]
            rmax = mask_main.shape[-2]
        else:
            rmin, rmax = rnz[[0, -1]]
            cmin, cmax = cnz[[0, -1]]
        rcen = int(round((rmin + rmax) * 0.5))
        ccen = int(round((cmin + cmax) * 0.5))
        rad = int(ceil(min(cmax - cmin, rmax - rmin) * args.scale * 0.5))

        # mask에 bounding box 그리기
        img_bbox = cv2.cvtColor(mask_main, cv2.COLOR_GRAY2BGR)
        cv2.rectangle(img_bbox, (cmin, rmin), (cmax, rmax), (0, 255, 0), 2)
        cv2.circle(img_bbox, (ccen, rcen), 7, (0, 0, 255), -1)  # 중심점 그리기
        cv2.imwrite(os.path.join(OUTPUT_DIR, 'mask_bbox.png'), img_bbox)


        ccen, rcen = map(int, map(round, cen_pt))  # 타원의 중심점의 좌표를 정수로 변환
        rad = max(min_ax * args.scale, max_ax * args.major_scale) * 0.5  # 타원의 반지름을 계산
        rad = int(ceil(rad))  # 반지름을 올림해서 정수로 변환
        rect_main = [ccen - rad, rcen - rad, 2 * rad, 2 * rad]  # 이미지를 잘라낼 영역의 좌표 계산

        im_crop = _crop_image(im, rect_main, args.const_border, value=255)
        mask_crop = _crop_image(mask_main, rect_main, True, value=0)

        mask_flt = mask_crop.astype(np.float32) / 255.0  # 마스크를 [0, 1] 범위로 정규화
        masked_crop = im_crop.astype(np.float32) * mask_flt + 255 * (1.0 - mask_flt)  # 잘라낸 이미지에 마스크를 적용
        masked_crop = masked_crop.astype(np.uint8)  # 결과 이미지 데이터 타입을 uint8로 변환



        # im_crop = cv2.resize(im_crop, (args.size, args.size), interpolation=cv2.INTER_LINEAR)
        # mask_crop = cv2.resize(
        #     mask_crop, (args.size, args.size), interpolation=cv2.INTER_LINEAR
        # )
        masked_crop = cv2.resize(
            masked_crop, (args.size, args.size), interpolation=cv2.INTER_AREA
        )

        if len(mask_crop.nonzero()[0]) == 0:
            print("WARNING: cropped mask is empty for", image_path, "skipping")
            continue

        # out_im_path = os.path.join(INPUT_DIR,
        #                             img_no_ext + ".jpg")
        # out_mask_path = os.path.join(INPUT_DIR,
        #                               img_no_ext + "_mask.png")
        out_masked_path = os.path.join(OUTPUT_DIR, img_no_ext + "_normalize.png")

        # cv2.imwrite(out_im_path, im_crop)
        # cv2.imwrite(out_mask_path, mask_crop)
        cv2.imwrite(out_masked_path, masked_crop)

        #  np.savetxt(os.path.join(INPUT_DIR,
        #                          img_no_ext + "_crop.txt"),
        #             rect_main,
        #             fmt='%.18f')
