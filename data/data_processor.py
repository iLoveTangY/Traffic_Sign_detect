#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@author: Tang Yang
@time: 2018/6/14 14:15
@file: data_processor.py
@desc: 
"""
import json
import matplotlib.pyplot as plt
import numpy as np


def show_data_labe(label_file):
    show_items = {}
    with open(label_file, 'r') as f:
        ret = json.load(f)
    ret = dict(ret)
    for _, value in ret['imgs'].items():
        if len(value['objects']) != 0:
            for item in value['objects']:
                category = item['category']
                if category in show_items:
                    show_items[category] += 1
                else:
                    show_items[category] = 1

    lst = sorted(show_items.items(), key=lambda tmp: tmp[1], reverse=True)
    x = []
    y = []

    for value in lst:
        x.append(value[0])
        y.append(value[1])

    width = 0.35
    y = y[:45]
    x = x[:45]
    ind = np.arange(len(y))
    print(len(y))
    plt.bar(ind, np.array(list(y)), width=0.35)
    plt.xticks(ind + width / 2, tuple(x))
    plt.show()


if __name__ == '__main__':
    show_data_labe(r'E:\data\data\annotations.json')
