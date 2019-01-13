#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sat Jan 12 10:45:08 2019

Naive tests that will be formalised later

@author: miika
"""
from __future__ import print_function, division
import numpy as np
from centroids import centroids
from sda import sda
from sda_ranking import sda_ranking
from predict_sda import predict_sda

# test 1
x = np.array([range(2,10),np.power(range(2,10),2),np.power(range(2,10),3)])
L = ["a","b","b","a","a","b","a","b"]
centroids(x.T, L, var_groups=True, centered_data=True, verbose=True)
#continue test 1 development

# test 2
khan_x = np.genfromtxt('/home/miika/khan_x.csv', delimiter=',')[1:,1:]
import csv
with open('/home/miika/khan_y.csv', 'rb') as f:
    reader = csv.reader(f)
    temp_y = list(reader)
khan_y = [tx[1] for tx in temp_y[1:]]

sda_out = sda(Xtrain=khan_x, L=khan_y)
sda_ranks = sda_ranking(khan_x,khan_y)
sda_out_pred = predict_sda(sda_out, khan_x)
assert sda_out_pred["predicted_class"] == khan_y # predicted class equals training class label
