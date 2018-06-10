#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@author: Tang Yang
@time: 2018/6/10 12:28
@file: config.py
@desc: 
"""

from easydict import EasyDict as edict

config = edict()

config.BATCH_SIZE = 128
config.CLS_OHEM = True
config.CLS_OHEM_RATIO = 0.7
config.BBOX_OHEM = False
config.BBOX_OHEM_RATIO = 0.7
config.EPS = 1e-14
config.LR_EPOCH = [8, 14]
