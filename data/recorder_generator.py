#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@author: Tang Yang
@time: 2018/6/8 10:03
@file: recorder_generator.py
@desc: 
"""
import tensorflow as tf
import os
import json
from skimage import io

from object_detection.utils import dataset_util
import hashlib

aa = "i2 i4 i5 il100 il60 il80 io ip p10 p11 p12 p19 p23 p26 p27 p3 p5 p6 pg ph4 ph4.5 ph5 " \
     "pl100 pl120 pl20 pl30 pl40 pl5 pl50 pl60 pl70 pl80 pm20 pm30 pm55 pn pne po pr40 w13 w32 w55 w57 w59 wo"
categories = aa.split(' ')


def create_tf_example(image_path, img, xmin, ymin, xmax, ymax, classes, classes_text=(),
                      truncated=(), poses=(), difficult_obj=(), source_id='', coordinate_normalize=False):
    """
    创建 tf example 实例
    :param img:
    :param image_path: 图片路径
    :param xmin: 多边形的boundingbox
    :param ymin:
    :param xmax:
    :param ymax:
    :param classes: 类别
    :param classes_text:
    :param truncated:
    :param poses:
    :param difficult_obj:
    :param source_id:
    :return: 创建好的tf example
    """
    with tf.gfile.GFile(image_path, 'rb') as fid:
        encoded_jpg = fid.read()
    key = hashlib.sha256(encoded_jpg).hexdigest()
    height = img.shape[0]
    width = img.shape[1]
    if coordinate_normalize:
        xmin = [float(x) / width for x in xmin]
        xmax = [float(x) / width for x in xmax]
        ymin = [float(y) / height for y in ymin]
        ymax = [float(y) / height for y in ymax]
    example = tf.train.Example(
        features=tf.train.Features(feature={
            'image/height':
                dataset_util.int64_feature(height),
            'image/width':
                dataset_util.int64_feature(width),
            'image/filename':
                dataset_util.bytes_feature(
                    os.path.basename(image_path).encode('utf8')),
            'image/source_id':
                dataset_util.bytes_feature(
                    os.path.basename(source_id).encode('utf8')),
            'image/key/sha256':
                dataset_util.bytes_feature(key.encode('utf8')),
            'image/encoded':
                dataset_util.bytes_feature(encoded_jpg),
            'image/format':
                dataset_util.bytes_feature('jpeg'.encode('utf8')),
            'image/object/bbox/xmin':
                dataset_util.float_list_feature(xmin),
            'image/object/bbox/xmax':
                dataset_util.float_list_feature(xmax),
            'image/object/bbox/ymin':
                dataset_util.float_list_feature(ymin),
            'image/object/bbox/ymax':
                dataset_util.float_list_feature(ymax),
            'image/object/class/text':
                dataset_util.bytes_list_feature(classes_text),
            'image/object/class/label':
                dataset_util.int64_list_feature(classes),
            'image/object/difficult':
                dataset_util.int64_list_feature(difficult_obj),
            'image/object/truncated':
                dataset_util.int64_list_feature(truncated),
            'image/object/view':
                dataset_util.bytes_list_feature(poses),
        }))
    return example


def _read_label(label_file):
    with open(label_file, 'r') as f:
        ret = json.load(f)
    return ret


class RecordGenerator:
    def __init__(self, image_dir, label_file):
        self._image_dir = image_dir
        self._label = _read_label(label_file)

    def _get_tf_example(self, img, image_full_name):
        image_name = image_full_name.split('.')[0]
        label_field = self._label['imgs'][image_name]['objects']
        xmin = []
        xmax = []
        ymin = []
        ymax = []
        classes = []
        for item in label_field:
            category = item['category']
            if category not in categories:
                continue

            classes.append(categories.index(category))
            bbox = item['bbox']
            xmin.append(int(bbox['xmin']))
            ymin.append(int(bbox['ymin']))
            xmax.append(int(bbox['xmax']))
            ymax.append(int(bbox['ymax']))

        return create_tf_example(os.path.join(self._image_dir, image_full_name),
                                 img, xmin, ymin, xmax, ymax, classes, coordinate_normalize=True)

    def write(self, output_path):
        if not os.path.exists(output_path):
            os.mkdir(output_path)
        files = os.listdir(self._image_dir)
        all_len = len(files)
        with tf.python_io.TFRecordWriter(output_path) as writer:
            for idx, file in enumerate(files):
                print('%d/%d' % (idx+1, all_len))
                if not os.path.isdir(file):
                    postfix = file.split('.')[-1]
                    if postfix in ['jpg', 'JPG']:
                        img = io.imread(os.path.join(self._image_dir, file))
                        example = self._get_tf_example(img, file)
                        writer.write(example.SerializeToString())


def write_to_pbtxt(pbtxt_path):
    with open(pbtxt_path, 'w') as f:
        for idx, item in enumerate(categories):
            f.write('item {\n')
            f.write('  id: %d\n' % (idx+1))
            f.write('  name: %s\n' % item)
            f.write('}\n\n')


if __name__ == '__main__':
    # r = RecordGenerator(r'E:\data\data\train',
    #                     r'E:\data\data\annotations.json')
    # r.write(r'C:\Users\DELL\Desktop\train.tfrecords')
    write_to_pbtxt(r'C:\Users\DELL\Desktop\traffic_sign.pbtxt')
