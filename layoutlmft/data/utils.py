import torch
import math
from detectron2.data.detection_utils import read_image
from detectron2.data.transforms import ResizeTransform, TransformList
from PIL import Image
import os

def get_overlap_byrelative(group_box,bbox,index):
    l_begin,l_end = index[group_box[1]], index[group_box[3]]
    r_begin,r_end = index[bbox[1]], index[bbox[3]]
    if l_begin >= r_end or l_end <= r_begin:
        return False
    else:
        return True

def group_by_threshold(lst, threshold):
    lst.sort()
    result = []
    current_group = [lst[0]]
    for i in range(1, len(lst)):
        if lst[i] - current_group[-1] <= threshold:
            current_group.append(lst[i])
        else:
            result.append(current_group)
            current_group = [lst[i]]
    result.append(current_group)
    i = 0
    index_dict = {}
    for group in result:
        for id in group:
            index_dict[id] = i
        i+=1
    return index_dict

def get_overlap_byrelative(group_box,bbox,index):
    l_begin,l_end = index[group_box[1]], index[group_box[3]]
    r_begin,r_end = index[bbox[1]], index[bbox[3]]
    if l_begin >= r_end or l_end <= r_begin:
        return False
    else:
        return True
    
    
def overlapping_rectangles(rect1, rect2):
    """
    判断两个矩形是否有重叠，如果有，返回重叠的面积。
    :param rect1: 第一个矩形的四个顶点坐标，格式为[x1, y1, x2, y2]，其中(x1, y1)为左下角坐标，(x2, y2)为右上角坐标。
    :param rect2: 第二个矩形的四个顶点坐标，格式为[x1, y1, x2, y2]，其中(x1, y1)为左下角坐标，(x2, y2)为右上角坐标。
    :return: 如果两个矩形有重叠，返回重叠的面积，否则返回0。
    """
    # 检查两个矩形是否有重叠
    if (rect1[0] > rect2[2]) or (rect1[2] < rect2[0]) or (rect1[1] > rect2[3]) or (rect1[3] < rect2[1]):
        return 0
    
    # 计算重叠的面积
    x_overlap = min(rect1[2], rect2[2]) - max(rect1[0], rect2[0])
    y_overlap = min(rect1[3], rect2[3]) - max(rect1[1], rect2[1])
    overlap_area = x_overlap * y_overlap
    
    return overlap_area/((rect2[3]-rect2[1])*(rect2[2]-rect2[0]))*100

def normalize_bbox(bbox, size):
    return [
        int(1000 * bbox[0] / size[0]),
        int(1000 * bbox[1] / size[1]),
        int(1000 * bbox[2] / size[0]),
        int(1000 * bbox[3] / size[1]),
    ]

def map_range(x, in_min, in_max, out_min, out_max):
    return (x - in_min) * (out_max - out_min) // (in_max - in_min) + out_min

def map_interval(a, b, c, d, x):
    return map_range(x, a, b, c, d)


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

def overlap(x1, x2, y1, y2):
    """
    判断两条线段重叠度的函数
    x1, x2: 第一条线段的端点
    y1, y2: 第二条线段的端点
    返回值: 如果两条线段重叠，则返回重叠长度；否则返回0
    """
    if x1 > x2:
        x1, x2 = x2, x1
    if y1 > y2:
        y1, y2 = y2, y1
    if x1 <= y1 <= x2:
        if y2 <= x2:
            return y2 - y1
        else:
            return x2 - y1
    elif y1 <= x1 <= y2:
        if x2 <= y2:
            return x2 - x1
        else:
            return y2 - x1
    else:
        return 0

def compute_y(table_bbox,bbox):
    y1,y2,y3,y4=table_bbox[3],table_bbox[1],bbox[3],bbox[1]
    return overlap(y1,y2,y3,y4)/(table_bbox[3]-table_bbox[1])*100


def compute_x(table_bbox,bbox):
    x1,x2,x3,x4=table_bbox[2],table_bbox[0],bbox[2],bbox[0]
    return overlap(x1,x2,x3,x4)/(table_bbox[2]-table_bbox[0])*100


def compute_y_overlap(table_bbox,bbox):
    # 左上角坐标为基准
    y1,y2,y3,y4=table_bbox[1],table_bbox[3],bbox[1],bbox[3]
    return abs(y1-y3)

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