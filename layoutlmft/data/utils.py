import torch
import math
from detectron2.data.detection_utils import read_image
from detectron2.data.transforms import ResizeTransform, TransformList
from PIL import Image
import os

def normalize_bbox(bbox, size):
    return [
        int(1000 * bbox[0] / size[0]),
        int(1000 * bbox[1] / size[1]),
        int(1000 * bbox[2] / size[0]),
        int(1000 * bbox[3] / size[1]),
    ]


def simplify_bbox(bbox):
    return [
        min(bbox[0::2]),
        min(bbox[1::2]),
        max(bbox[2::2]),
        max(bbox[3::2]),
    ]


def merge_bbox(bbox_list):
    x0, y0, x1, y1 = list(zip(*bbox_list))
    return [min(x0), min(y0), max(x1), max(y1)]


def load_image(image_path):
    image = read_image(image_path, format="BGR")
    h = image.shape[0]
    w = image.shape[1]
    img_trans = TransformList([ResizeTransform(h=h, w=w, new_h=224, new_w=224)])
    image = torch.tensor(img_trans.apply_image(image).copy()).permute(2, 0, 1)  # copy to make it writeable
    return image, (w, h)


def normalizebbox(bbox,ocr_width, ocr_height,width, height):
    result = [bbox[0],bbox[1],bbox[4],bbox[5]]
    for i, data in enumerate(result):
        if i < 2 :
            result[i] = int(data/ocr_width*width)
        else:
            result[i] = math.ceil(data/ocr_height*height)
    return result

def point_in_rect(x1,y1,R):
    if R[0] <= x1 and R[1] <= y1 and R[2] >= x1 and R[3] >= y1:
        return True
    return False
def bbox_overlap(R1,R2):
    if point_in_rect(R2[0],R2[1],R1) or point_in_rect(R2[0],R2[3],R1) \
        or point_in_rect(R2[2],R2[1],R1) or point_in_rect(R2[2],R2[3],R1):
        return True
    else:
        return False

def compute_y(table_bbox,bbox):
    if table_bbox[3] <= bbox[1] or table_bbox[1] >= bbox[3]:
        return 0
    elif table_bbox[1] < bbox[1]:
        return (table_bbox[3]-bbox[1])/(table_bbox[3]-table_bbox[1])*100
    elif table_bbox[1] >= bbox[1]:
        return (bbox[3]-table_bbox[1])/(table_bbox[3]-table_bbox[1])*100


def bboxinRect(textbbox,bbox):
    x1, y1 = (textbbox[0]+ textbbox[2])/2,(textbbox[1]+ textbbox[3])/2
    return point_in_rect(x1,y1,bbox)


def get_image(MODE,id):
    if MODE == 'val':
        ocr_width, ocr_height = (3100, 4385)
    else:
        ocr_width, ocr_height = (4133, 5847)
    # 3508， 2489
    image_path = os.path.join(f"/home/zhanghang-s21/data/bishe/MYXFUND/my{MODE}",id)
    with Image.open(image_path) as image:
        image_width, image_height = image.size[0] , image.size[1]
    if image_width > image_height:
        ocr_width, ocr_height = ocr_height, ocr_width
    if id.find('mytrain') != -1:
        ocr_width, ocr_height = image_width, image_height
    return ocr_width, ocr_height, image_width, image_height