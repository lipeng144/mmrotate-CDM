import sys
import os
import re
import shutil
import numpy as np
import json
import xml.etree.ElementTree as ET
from mmrotate.datasets.transforms.rotatedtrans import  obb2poly_np
# 检测框的ID起始值
START_BOUNDING_BOX_ID = 1
# 类别列表无必要预先创建，程序中会根据所有图像中包含的ID来创建并更新
PRE_DEFINE_CATEGORIES = {"ship": 0}
# If necessary, pre-define category and its id
#  PRE_DEFINE_CATEGORIES = {"aeroplane": 1, "bicycle": 2, "bird": 3, "boat": 4,
                         #  "bottle":5, "bus": 6, "car": 7, "cat": 8, "chair": 9,
                         #  "cow": 10, "diningtable": 11, "dog": 12, "horse": 13,
                         #  "motorbike": 14, "person": 15, "pottedplant": 16,
                         #  "sheep": 17, "sofa": 18, "train": 19, "tvmonitor": 20}


def get(root, name):
    vars = root.findall(name)
    return vars


def get_and_check(root, name, length):
    vars = root.findall(name)
    if len(vars) == 0:
        raise NotImplementedError('Can not find %s in %s.'%(name, root.tag))
    if length > 0 and len(vars) != length:
        raise NotImplementedError('The size of %s is supposed to be %d, but is %d.'%(name, length, len(vars)))
    if length == 1:
        vars = vars[0]
    return vars


def extract_file_id(filename):
    # 使用正则表达式提取文件名中的所有数字
    match = re.findall(r'\d+', filename)

    # 将提取到的所有数字合并成一个字符串
    if match:
        file_id_str = ''.join(match)  # 将所有数字连接成一个字符串
        file_id = int(file_id_str)  # 将字符串转换为整数
        return file_id
    else:
        return None

    # 得到图片唯一标识号
def get_filename_as_int(filename):
    try:
        filename = os.path.splitext(filename)[0]
        return int(filename)
    except:
        raise NotImplementedError('Filename %s is supposed to be an integer.'%(filename))


def convert(xml_list, xml_dir, json_file):
    '''
    :param xml_list: 需要转换的XML文件列表
    :param xml_dir: XML的存储文件夹
    :param json_file: 导出json文件的路径
    :return: None
    '''
    list_fp = xml_list
    # 标注基本结构
    json_dict = {"images":[],
                 "type": "instances",
                 "annotations": [],
                 "categories": []}
    categories = PRE_DEFINE_CATEGORIES
    bnd_id = START_BOUNDING_BOX_ID
    id=0
    for line in list_fp:
        line = line.strip()
        print("buddy~ Processing {}".format(line))
        # 解析XML
        xml_f = os.path.join(xml_dir, line+'.xml')
        tree = ET.parse(xml_f)
        root = tree.getroot()
        path = get(root, 'path')
        # 取出图片名字
        if len(path) == 1:
            filename = os.path.basename(path[0].text)
        elif len(path) == 0:
            filename = get_and_check(root, 'filename', 1).text
        else:
            raise NotImplementedError('%d paths found in %s'%(len(path), line))
        ## The filename must be a number
        image_id = extract_file_id(filename)  # 图片ID
        size = get_and_check(root, 'size', 1)
        # 图片的基本信息
        width = int(get_and_check(size, 'width', 1).text)
        height = int(get_and_check(size, 'height', 1).text)
        image = {'file_name': filename[:-4]+'.jpg',
                 'height': height,
                 'width': width,
                 'id':image_id}
        json_dict['images'].append(image)
        ## Cruuently we do not support segmentation
        #  segmented = get_and_check(root, 'segmented', 1).text
        #  assert segmented == '0'
        # 处理每个标注的检测框
        for obj in get(root, 'object'):
            # 取出检测框类别名称
            category = get_and_check(obj, 'name', 1).text
            # 更新类别ID字典
            if category not in categories:
                new_id = len(categories)
                categories[category] = new_id
            category_id = categories[category]
            robndbox = get_and_check(obj, 'robndbox', 1)

            cx = get_and_check(robndbox, 'cx', 1).text
            cy = get_and_check(robndbox, 'cy', 1).text
            h = get_and_check(robndbox, 'h', 1).text
            w = get_and_check(robndbox, 'w', 1).text
            angel=get_and_check(robndbox, 'angle', 1).text
            rbox=np.array([[
                    float(cx),
                    float(cy),
                    h,
                    w,
                    float(angel), 0 ]],   dtype=np.float32)
            polygon=list(obb2poly_np(rbox, 'le90')[0, :-1])
            #int_polygon = [int(num) for num in polygon]
            xmin = min(polygon[i] for i in range(0, len(polygon), 2))
            xmax = max(polygon[i] for i in range(0, len(polygon), 2))
            ymin = min(polygon[i] for i in range(1, len(polygon), 2))
            ymax = max(polygon[i] for i in range(1, len(polygon), 2))
            assert(xmax > xmin)
            assert(ymax > ymin)
            o_width = abs(xmax - xmin)
            o_height = abs(ymax - ymin)
            annotation = dict()
            annotation['area'] = o_width*o_height
            annotation['iscrowd'] = 0
            annotation['image_id'] = image_id
            annotation['bbox'] = [xmin, ymin, o_width, o_height]
            annotation['category_id'] = category_id
            annotation['id'] = bnd_id
            annotation['ignore'] = 0
            # 设置分割数据，点的顺序为逆时针方向
            annotation['segmentation'] = [polygon]

            json_dict['annotations'].append(annotation)
            bnd_id = bnd_id + 1

    # 写入类别ID字典
    for cate, cid in categories.items():
        cat = {'supercategory': 'none', 'id': cid, 'name': cate}
        json_dict['categories'].append(cat)
    # 导出到json
    json_fp = open(json_file, 'w')
    json_str = json.dumps(json_dict)
    json_fp.write(json_str)
    json_fp.close()


if __name__ == '__main__':
    root_path = 'D:/pythondata/mmrotate-1.x/tools/data/rsdd/'
    xml_dir = os.path.join(root_path, 'Annotations')

    xml_labels = os.listdir(os.path.join(root_path, 'Annotations'))
    #np.random.shuffle(xml_labels)
    #split_point = int(len(xml_labels)/10)
    txtname='D:/pythondata/mmrotate-1.x/tools/data/rsdd/ImageSets/test_offshore.txt'
    with open(txtname, 'r', encoding='utf-8') as file:
         filename = file.readlines()
    # validation data
    #xml_list = xml_labels[0:split_point]
    json_file = 'D:/pythondata/mmrotate-1.x/tools/data/rsdd/ImageSets/test_offshore.json'
    convert(filename, xml_dir, json_file)

