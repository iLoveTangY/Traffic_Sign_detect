#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@author: Tang Yang
@time: 2018/6/10 13:05
@file: run_all.py
@desc: 
"""
import os
import cv2

from core.dense_net import DenseNet
from core.detector import Detector
from core.fcn_detector import FcnDetector
from core.model import P_Net, R_Net, O_Net
from core.mtcnn_detector import MtcnnDetector
from utils import *

ALL_CLASSES = "i2 i4 i5 il100 il60 il80 io ip p10 p11 p12 p19 p23 p26 p27 p3 p5 p6 pg ph4 ph4.5 ph5 " \
              "pl100 pl120 pl20 pl30 pl40 pl5 pl50 pl60 pl70 pl80 pm20 pm30 pm55 pn pne po pr40 w13 w32 w55 w57 w59 wo"
COLLECT = ALL_CLASSES.split(' ')


def visualize_result(img, detect_ret, pred, thresh=0.998, show=True, save_image=False, save_path=''):
    # 交换x，y坐标顺序
    bbox = np.array(detect_ret[:, :4])
    bbox[:, [0, 1]] = bbox[:, [1, 0]]
    bbox[:, [2, 3]] = bbox[:, [3, 2]]
    scores = np.array(detect_ret[:, 4])
    classes = np.array(pred)

    label_dict = get_label_dict_from_string(COLLECT)
    # 绘制结果
    visualize_boxes_and_labels_on_image_array(img, bbox, classes, scores, label_dict, line_thickness=2,
                                              use_normalized_coordinates=False, min_score_thresh=thresh)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    im = Image.fromarray(img)
    if show:
        im.show()

    if save_image and save_path != '':
        im.save(save_path)
    elif save_image and save_path == '':
        print("Must specific save path!")


def detect_images(image_paths, mtcnn_detector, model):
    for file in image_paths:
        # 一张图片中所有检测到的交通标志
        images_res = []
        img = cv2.imread(file)
        # MTCNN检测
        _, boxes_c = mtcnn_detector.detect(img)
        # print(boxes_c.shape)
        # visssss(img, boxes_c, 'plain13134.jpg', thresh=0.998)
        for i in range(boxes_c.shape[0]):
            bbox = boxes_c[i, :4].astype('int32')
            # 处理box超出边界的情况
            if bbox[1] < 0:
                bbox[1] = 0
            if bbox[0] < 0:
                bbox[0] = 0
            if bbox[2] > 2048:
                bbox[2] = 2048
            if bbox[3] > 2048:
                bbox[3] = 2048
            # 截出检测到的box里面的图片用于后面分类
            crop = img[bbox[1]:bbox[3], bbox[0]:bbox[2], :]
            # 截出来的图片resize到(48, 48)用于后面分类
            crop = cv2.resize(crop, (48, 48))
            images_res.append(crop)
        # 图片像素必须得是np.float32类型
        images_res = np.array(images_res).astype(np.float32)
        images_res = normalize_images(images_res)
        # DenseNet分类
        pred = model.test(images_res)
        # 如果是第45类，那么图片是背景而不是交通标志，我们会在下一步中忽略掉
        bg_box = np.where(pred == 45)
        # 忽略掉背景图片(将分数和坐标全部设置为0)
        for ii in bg_box[0]:
            boxes_c[ii, :] = 0
        visualize_result(img, boxes_c, pred, thresh=0.998, show=True, save_image=True, save_path=str(file)+"_detectd.jpg")


def load_model(model_path):
    """
    根据指定的模型路径加载mtcnn和DenseNet的模型
    :param model_path:
    :return:
    """
    batch_size = [1, 1, 1]
    model = DenseNet(24, 40, 3, 0.8, 'DenseNet-BC', reduction=0.5, bc_mode=True)
    detectors = [None, None, None]
    # 加载用于分类的DenseNet模型
    model.load_model()

    # 计算出PNet、RNet、ONet模型的路径
    prefix = ['pnet/pnet', 'rnet/rnet', 'onet/onet']
    prefix = [os.path.join(model_path, x) for x in prefix]
    epoch = [7, 7, 7]
    model_path = ['%s-%s' % (x, y) for x, y in zip(prefix, epoch)]

    # 加载PNet, 这里只能用FcnDetector
    p_net = FcnDetector(P_Net, model_path[0])
    detectors[0] = p_net

    # 加载RNet
    r_net = Detector(R_Net, 24, batch_size[1], model_path[1])
    detectors[1] = r_net

    # 加载ONet
    o_net = Detector(O_Net, 48, batch_size[2], model_path[2])
    detectors[2] = o_net

    # 建立MTCNN检测器
    mtcnn_detector = MtcnnDetector(detectors=detectors, min_face_size=24,
                                   stride=2, threshold=(0.6, 0.6, 0.7), slide_window=False)

    return model, mtcnn_detector


def eval_model():
    dense_net_model, mtcnn_detector = load_model('./model')
    test_floder = './test'
    files = os.listdir(test_floder)
    image_paths = []
    for file in files:
        post_fix = file.split('.')[-1]
        if post_fix in ['jpg', 'JPG']:
            image_paths.append(os.path.join(test_floder, file))
    detect_images(image_paths, mtcnn_detector, dense_net_model)


if __name__ == '__main__':
    eval_model()
