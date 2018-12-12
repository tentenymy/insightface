"""Helper for evaluation on the Labeled Faces in the Wild dataset
"""

# MIT License
#
# Copyright (c) 2016 David Sandberg
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import argparse
import numpy as np
from sklearn.model_selection import KFold
from scipy import interpolate
import sklearn
import cv2
import math
import datetime
import pickle
from sklearn.decomposition import PCA
import mxnet as mx
from mxnet import ndarray as nd
from easydict import EasyDict as edict
import heapq
from sklearn.metrics import roc_curve
import time


def parse_argument():
    parser = argparse.ArgumentParser(description='do verification')
    parser.add_argument('--data_dir', default='', type=str, help='')
    parser.add_argument('--model', default='../model/softmax,50', type=str,
                        help='path to load model.')
    parser.add_argument('--target',
                        default='suningshop_v7,renren,classify_renren_1000,classify_renren_5000,classify_renren_10000,lfw,cfp_fp',
                        type=str, help='test targets.')
    parser.add_argument('--gpu', default="", type=str, help='gpu id')
    parser.add_argument('--batch_size', default=100, type=int, help='')
    parser.add_argument('--model_epoch_range', default='0,1000', type=str, help='e.g.[1,50]')
    parser.add_argument('--mode', default=0, type=int, help='')
    parser.add_argument('--nfolds', default=10, type=int, help='')
    parser.add_argument('--testing_model_list', default='', type=str, help='')
    parser.add_argument('--output_path', default="log.txt", type=str, help='')
    parser.add_argument('--badcase_tag', default="", type=str, help='')
    parser.add_argument('--issame_bin', default="", type=str,
                        help='for facescrub/megaface, issame matrix')
    return parser.parse_args()


class LFold:
    def __init__(self, n_splits=2, shuffle=False):
        self.n_splits = n_splits
        if self.n_splits > 1:
            self.k_fold = KFold(n_splits=n_splits, shuffle=shuffle)

    def split(self, indices):
        if self.n_splits > 1:
            return self.k_fold.split(indices)
        else:
            return [(indices, indices)]


def calculate_roc(thresholds, embeddings1, embeddings2, actual_issame, nrof_folds=10, pca=0):
    assert (embeddings1.shape[0] == embeddings2.shape[0])
    assert (embeddings1.shape[1] == embeddings2.shape[1])
    nrof_pairs = min(len(actual_issame), embeddings1.shape[0])
    nrof_thresholds = len(thresholds)
    k_fold = LFold(n_splits=nrof_folds, shuffle=False)

    tprs = np.zeros((nrof_folds, nrof_thresholds))
    fprs = np.zeros((nrof_folds, nrof_thresholds))
    accuracy = np.zeros(nrof_folds)
    indices = np.arange(nrof_pairs)

    dist = np.empty(embeddings1.shape[0])
    if pca == 0:
        diff = np.subtract(embeddings1, embeddings2)
        dist = np.sum(np.square(diff), 1)

    for fold_idx, (train_set, test_set) in enumerate(k_fold.split(indices)):

        if pca > 0:
            print('doing pca on', fold_idx)
            embed1_train = embeddings1[train_set]
            embed2_train = embeddings2[train_set]
            _embed_train = np.concatenate((embed1_train, embed2_train), axis=0)
            pca_model = PCA(n_components=pca)
            pca_model.fit(_embed_train)
            embed1 = pca_model.transform(embeddings1)
            embed2 = pca_model.transform(embeddings2)
            embed1 = sklearn.preprocessing.normalize(embed1)
            embed2 = sklearn.preprocessing.normalize(embed2)
            diff = np.subtract(embed1, embed2)
            dist = np.sum(np.square(diff), 1)

        # Find the best threshold for the fold
        acc_train = np.zeros(nrof_thresholds)
        for threshold_idx, threshold in enumerate(thresholds):
            _, _, acc_train[threshold_idx] = calculate_accuracy(threshold, dist[train_set],
                                                                actual_issame[train_set])
        best_threshold_index = np.argmax(acc_train)

        for threshold_idx, threshold in enumerate(thresholds):
            tprs[fold_idx, threshold_idx], fprs[fold_idx, threshold_idx], _ = calculate_accuracy(
                threshold, dist[test_set], actual_issame[test_set])
        _, _, accuracy[fold_idx] = calculate_accuracy(
            thresholds[best_threshold_index], dist[test_set], actual_issame[test_set])

    tpr = np.mean(tprs, 0)
    fpr = np.mean(fprs, 0)
    return tpr, fpr, accuracy


def calculate_accuracy(threshold, dist, actual_issame):
    predict_issame = np.less(dist, threshold)
    tp = np.sum(np.logical_and(predict_issame, actual_issame))
    fp = np.sum(np.logical_and(predict_issame, np.logical_not(actual_issame)))
    tn = np.sum(np.logical_and(np.logical_not(predict_issame), np.logical_not(actual_issame)))
    fn = np.sum(np.logical_and(np.logical_not(predict_issame), actual_issame))

    tpr = 0 if (tp + fn == 0) else float(tp) / float(tp + fn)
    fpr = 0 if (fp + tn == 0) else float(fp) / float(fp + tn)
    acc = float(tp + tn) / dist.size
    return tpr, fpr, acc


def calculate_val(thresholds, embeddings1, embeddings2, actual_issame, far_target, nrof_folds=10):
    assert (embeddings1.shape[0] == embeddings2.shape[0])
    assert (embeddings1.shape[1] == embeddings2.shape[1])
    nrof_pairs = min(len(actual_issame), embeddings1.shape[0])
    nrof_thresholds = len(thresholds)
    k_fold = LFold(n_splits=nrof_folds, shuffle=False)

    val = np.zeros(nrof_folds)
    far = np.zeros(nrof_folds)

    diff = np.subtract(embeddings1, embeddings2)
    dist = np.sum(np.square(diff), 1)
    indices = np.arange(nrof_pairs)

    for fold_idx, (train_set, test_set) in enumerate(k_fold.split(indices)):
        far_train = np.zeros(nrof_thresholds)
        for threshold_idx, threshold in enumerate(thresholds):
            _, far_train[threshold_idx] = calculate_val_far(threshold, dist[train_set],
                                                            actual_issame[train_set])
        if np.max(far_train) >= far_target:
            f = interpolate.interp1d(far_train, thresholds, kind='slinear')
            threshold = f(far_target)
        else:
            threshold = 0.0

        val[fold_idx], far[fold_idx] = calculate_val_far(threshold, dist[test_set],
                                                         actual_issame[test_set])

    val_mean = np.mean(val)
    far_mean = np.mean(far)
    val_std = np.std(val)
    return val_mean, val_std, far_mean


def calculate_val_far(threshold, dist, actual_issame, reverse=False):
    if reverse:
        predict_issame = np.greater(dist, threshold)
    else:
        predict_issame = np.less(dist, threshold)
    true_accept = np.sum(np.logical_and(predict_issame, actual_issame))
    false_accept = np.sum(np.logical_and(predict_issame, np.logical_not(actual_issame)))
    n_same = np.sum(actual_issame)
    n_diff = np.sum(np.logical_not(actual_issame))
    val = float(true_accept)
    far = float(false_accept)
    if n_same != 0:
        val = float(true_accept) / float(n_same)
    if n_diff != 0:
        far = float(false_accept) / float(n_diff)
    return val, far


def evaluate(embeddings, actual_issame, nrof_folds=10, pca=0):
    thresholds = np.arange(0, 4, 0.01)
    embeddings1 = embeddings[0::2]
    embeddings2 = embeddings[1::2]
    tpr, fpr, accuracy = calculate_roc(
        thresholds, embeddings1, embeddings2,
        np.asarray(actual_issame), nrof_folds=nrof_folds, pca=pca)
    index = np.argmax(accuracy)

    thresholds = np.arange(0, 4, 0.001)
    val, val_std, far = calculate_val(
        thresholds, embeddings1, embeddings2,
        np.asarray(actual_issame), 1e-3, nrof_folds=nrof_folds)

    return tpr, fpr, accuracy, val, val_std, far


def evaluate_suning(embeddings):
    # Calculate evaluation metrics
    embeddings1 = embeddings[0::2]
    embeddings2 = embeddings[1::2]
    tpr, far, thre, accuracy = cal_validation(embeddings1, embeddings2)
    return tpr, far, thre, accuracy


def cal_validation(query, database, far_target=0.001):
    score_matrix = cal_matrix(query, database)
    far_diff = 1
    for thre in range(300, 600):
        thre = float(thre) / 1000.0
        far, tpr, accuracy = cal_far_and_tpr(score_matrix, thre)
        if abs(far - 0.001) < far_diff:
            far_diff = abs(far - 0.001)
            far_nearest = far
            tpr_nearest = tpr
            thre_nearest = thre
            accuracy_nearest = accuracy
    return tpr_nearest, far_nearest, thre_nearest, accuracy_nearest


def cal_matrix(query, database):
    n = len(query)
    matrix = [[0 for i in xrange(n)] for j in xrange(n)]
    for i in xrange(n):
        for j in xrange(i, n):
            embeddings1, embeddings2 = query[i], database[j]
            num = float(np.sum(embeddings1 * embeddings2))
            denorm = np.linalg.norm(embeddings1) * np.linalg.norm(embeddings2)
            matrix[i][j] = (num / denorm)
    return matrix


def cal_far_and_tpr(score_matrix, threshold):
    tp, tn, fp, fn = 0, 0, 0, 0
    pos, neg = 0, 0
    n = len(score_matrix)
    for i in xrange(n):
        for j in xrange(i, n):
            score = 1 if score_matrix[i][j] > threshold else 0
            if i == j:
                pos += 1
                if score == 1:
                    tp += 1
                else:
                    fn += 1
            else:
                neg += 1
                if score == 1:
                    fp += 1
                else:
                    tn += 1
    far = fp / neg
    tpr = tp / pos
    accuracy = (tp + tn) / (pos + neg)
    return far, tpr, accuracy


# load lfw/renren/suningshop/douyin
def load_bin(path, image_size):
    bins, issame_list = pickle.load(open(path, 'rb'))
    data_list = []
    for flip in [0, 1]:
        data = nd.empty((len(issame_list) * 2, 3, image_size[0], image_size[1]))
        data_list.append(data)
    for i in xrange(len(issame_list) * 2):
        _bin = bins[i]
        img = mx.image.imdecode(_bin)
        img = nd.transpose(img, axes=(2, 0, 1))
        for flip in [0, 1]:
            if flip == 1:
                img = mx.ndarray.flip(data=img, axis=2)
            data_list[flip][i][:] = img
    return (data_list, issame_list)


# load megaface
def load_megaface_bin(path, image_size):
    bins, issame_list = pickle.load(open(path, 'rb'))
    data_list = []
    for flip in [0, 1]:
        data = nd.empty((len(issame_list), 3, image_size[0], image_size[1]))
        data_list.append(data)
    for i in xrange(len(issame_list)):
        _bin = bins[i]
        if len(_bin) != 0:
            img = mx.image.imdecode(_bin)
            img = nd.transpose(img, axes=(2, 0, 1))
            for flip in [0, 1]:
                if flip == 1:
                    img = mx.ndarray.flip(data=img, axis=2)
                data_list[flip][i][:] = img
    return (data_list, issame_list)


# test
def test(data_set, mx_model, batch_size, nfolds=10, data_extra=None, label_shape=None,
         mode_classify=0):
    # print('testing verification..')
    data_list = data_set[0]
    issame_list = data_set[1]
    model = mx_model
    embeddings_list = []
    if data_extra is not None:
        _data_extra = nd.array(data_extra)
    time_consumed = 0.0
    if label_shape is None:
        _label = nd.ones((batch_size,))
    else:
        _label = nd.ones(label_shape)


    time_current = time.time()
    for i in xrange(len(data_list)):
        data = data_list[i]
        embeddings = None
        ba = 0
        while ba < data.shape[0]:

            bb = min(ba + batch_size, data.shape[0])
            count = bb - ba

            _data = nd.slice_axis(data, axis=0, begin=bb - batch_size, end=bb)

            time0 = datetime.datetime.now()
            if data_extra is None:
                db = mx.io.DataBatch(data=(_data,), label=(_label,))
            else:
                db = mx.io.DataBatch(data=(_data, _data_extra), label=(_label,))

            model.forward(db, is_train=False)
            net_out = model.get_outputs()

            _embeddings = net_out[0][batch_size - count:].asnumpy()

            time_now = datetime.datetime.now()
            diff = time_now - time0
            time_consumed += diff.total_seconds()

            if embeddings is None:
                embeddings = np.zeros((data.shape[0], _embeddings.shape[1]))
            embeddings[ba:bb, :] = _embeddings[:, :]
            ba = bb
        embeddings_list.append(embeddings)

    is_flip = True if len(data_list) > 1 else False
    if is_flip:
        embeddings = embeddings_list[0] + embeddings_list[1]
    else:
        embeddings = embeddings_list[0]

    try:
        embeddings = sklearn.preprocessing.normalize(embeddings)
    except ValueError:
        print("embeddings explode")

    time_spend = 1000 * (time.time() - time_current) / len(embeddings)
    print("time: %.2f ms image: %d" % (time_spend, len(embeddings)))

    if mode_classify == 0:
        # lfw, suningshop, renren, douyin, cfp_ff, cfp_fp, agedb_30
        tpr, fpr, accuracy, val, val_std, far = evaluate(embeddings, issame_list, nrof_folds=nfolds)
        acc, acc_std = np.mean(accuracy), np.std(accuracy)
        return val, val_std, acc, acc_std, far, embeddings_list, time_spend
    elif mode_classify == 1:
        # classify_renren_*, classify_douyin_*, classify_suning_test, classify_suning_1000
        acc, val, far = classify_images_half(embeddings)
        return val, 0.0, acc, 0.0, far, embeddings_list, time_spend

    elif mode_classify == 2:
        # classify_megaface
        issame_matrix = np.array(issame_list)  # (13530 * 3530)
        accuracy = classify_megaface(embeddings, issame_matrix)
        return 0.0, 0.0, accuracy, 0.0, 0.0, [], time_spend
    else:
        # classify_suning_test
        tpr, far, thre, accuracy = evaluate_suning(embeddings)
        return tpr, 0.0, accuracy, 0.0, far, [], time_spend


# classify megaface
def classify_megaface(embeddings_list, issame_matrix):
    # issame_matrix[13530, 3530]: bool
    # distance_matrix_facescrub_facescrub[i,j] = distance for facescrub[i] and facescrub[j] (3530*3530)
    # distance_matrix_megaface_facescrub[i,j] = distance for megaface[i] and facescrub[j] (10000*3530)
    facescrub_size = issame_matrix.shape[1]
    megaface_size = issame_matrix.shape[0] - issame_matrix.shape[1]
    distance_matrix = np.zeros((facescrub_size + megaface_size, facescrub_size))
    distance_max_megeface = np.zeros(facescrub_size)
    for i in range(facescrub_size):
        distance_matrix[:, i] = np.matmul(embeddings_list, embeddings_list[i])
        distance_max_megeface[i] = np.max(distance_matrix[facescrub_size:, i])

    # find facescrub in megaface
    correct_number = 0.0
    total_number = 0.0
    for i in range(facescrub_size):
        for j in range(facescrub_size):
            if i == j or not issame_matrix[i, j]:
                continue
            total_number += 1
            if distance_max_megeface[i] <= distance_matrix[i, j]:
                correct_number += 1
    accuracy = correct_number / total_number

    # val
    # thresholds = np.arange(0.2, 0.5, 0.001)[::-1]
    # far_target = 0.0001
    # nrof_thresholds = len(thresholds)
    # far_train = np.zeros(nrof_thresholds)
    # for threshold_idx, threshold in enumerate(thresholds):
    # 	_, far_train[threshold_idx] = calculate_val_far(threshold, distance_matrix, issame_matrix, reverse=True)
    # 	print(threshold_idx, threshold, far_train[threshold_idx])
    # if np.max(far_train) >= far_target:
    # 	f = interpolate.interp1d(far_train, thresholds, kind='slinear')
    # 	threshold = f(far_target)
    # else:
    # 	threshold = 0.0
    #
    # val, far = calculate_val_far(threshold, distance_matrix, issame_matrix, reverse=True)
    # print("val: ", val, "far: ", far)
    return accuracy


def classify_images(embeddings_list):
    # matrix[0][i][j] = distance between cls[i]image[0](embeddings[i * 2] and cls[j]image[1] (embeddings[j * 2 + 1]
    # matrix[1][i][j] = distance between cls[i]image[1](embeddings[i * 2 + 1] and cls[j]image[0] (embeddings[j * 2]
    pairs_number = int(embeddings_list.shape[0] / 2)
    distance_matrix = np.zeros((2, pairs_number, pairs_number))

    distance_list = []
    issame_list = []
    for i in range(pairs_number):
        for j in range(i, pairs_number):
            distance_matrix[0][i][j] = np.matmul(embeddings_list[i * 2], embeddings_list[j * 2 + 1])
            distance_matrix[1][j][i] = distance_matrix[0][i][j]
            distance_matrix[1][i][j] = np.matmul(embeddings_list[i * 2 + 1], embeddings_list[j * 2])
            distance_matrix[0][i][j] = distance_matrix[1][i][j]
            if i == j:
                issame_list.append(True)
                distance_list.append(distance_matrix[0][i][j])
            else:
                issame_list.append(False)
                distance_list.append(distance_matrix[0][i][j])
                issame_list.append(False)
                distance_list.append(distance_matrix[1][i][j])

    # get min
    correct_number = 0.0
    correct_number_top_five = 0.0
    for i in range(pairs_number):
        answer = np.argmax(distance_matrix[0][i][:])
        top_five_answer = np.argpartition(distance_matrix[0][i][:], -5)[-5:]
        if i == answer:
            correct_number += 1.0
        if i in top_five_answer:
            correct_number_top_five += 1.0

        answer = np.argmax(distance_matrix[1][i][:])
        top_five_answer = np.argpartition(distance_matrix[1][i][:], -5)[-5:]
        if i == answer:
            correct_number += 1.0
        if i in top_five_answer:
            correct_number_top_five += 1.0

    acc = correct_number / (pairs_number * 2)
    acc_top_five = correct_number_top_five / (pairs_number * 2)
    # print("total: ", acc, "top5: ", acc_top_five)

    # val
    thresholds = np.arange(0.2, 0.8, 0.01)[::-1]
    far_target = 0.001
    nrof_thresholds = len(thresholds)
    far_train = np.zeros(nrof_thresholds)
    for threshold_idx, threshold in enumerate(thresholds):
        _, far_train[threshold_idx] = calculate_val_far(threshold, distance_list, issame_list,
                                                        reverse=True)
    # print(threshold_idx, threshold, far_train[threshold_idx])
    if np.max(far_train) >= far_target:
        try:
            f = interpolate.interp1d(far_train, thresholds, kind='slinear')
        except (ValueError) as e:
            print("cannot interpolate")
            return 0.0
        else:
            threshold = f(far_target)
    else:
        threshold = 0.0

    val, far = calculate_val_far(threshold, distance_list, issame_list, reverse=True)
    # print("val: ", val, "far: ", far, "thres: ", threshold)
    return acc, val, far


def classify_images_half(embeddings_list):
    # matrix[0][i][j] = distance between cls[i]image[0](embeddings[i * 2] and cls[j]image[1] (embeddings[j * 2 + 1]
    # matrix[1][i][j] = distance between cls[i]image[1](embeddings[i * 2 + 1] and cls[j]image[0] (embeddings[j * 2]
    pairs_number = int(embeddings_list.shape[0] / 2)
    distance_matrix = np.zeros((2, pairs_number, pairs_number))

    distance_list = []
    issame_list = []
    for i in range(pairs_number):
        for j in range(i, pairs_number):
            distance_matrix[0][i][j] = np.matmul(embeddings_list[i * 2], embeddings_list[j * 2 + 1])
            distance_matrix[1][j][i] = distance_matrix[0][i][j]
            distance_matrix[1][i][j] = np.matmul(embeddings_list[i * 2 + 1], embeddings_list[j * 2])
            distance_matrix[0][i][j] = distance_matrix[1][i][j]
            if i == j:
                issame_list.append(True)
                distance_list.append(distance_matrix[0][i][j])
            else:
                issame_list.append(False)
                distance_list.append(distance_matrix[0][i][j])
                issame_list.append(False)
                distance_list.append(distance_matrix[1][i][j])

    # get min
    correct_number = 0.0
    for i in range(pairs_number):
        answer = np.argmax(distance_matrix[0][i][:])
        if i == answer:
            correct_number += 1.0

    acc = correct_number / (pairs_number)

    # val
    thresholds = np.arange(0.2, 0.8, 0.01)[::-1]

    nrof_thresholds = len(thresholds)
    far_train = np.zeros(nrof_thresholds)
    for threshold_idx, threshold in enumerate(thresholds):
        _, far_train[threshold_idx] = calculate_val_far(threshold, distance_list, issame_list,
                                                        reverse=True)

    fpr_temp, tpr_temp, threshold_temp = roc_curve(issame_list, distance_list)
    tpr_list = []
    for far_target in [0.001, 0.0001, 0.00001, 0.000001]:
        tmp = np.where(fpr_temp < far_target)
        index = len(tmp[0].tolist())
        far = fpr_temp[index]
        tpr = tpr_temp[index]
        tpr_list.append(tpr)
        threshold = threshold_temp[index]
        print("Val: %.4f, Far: %.7f, Thres: %.4f" % (tpr, far_target, threshold))

    return tpr_list[-1], tpr_list[-2], 0.00001


def test_badcase(data_set, model, batch_size, name='', badcase_tag='', data_extra=None,
                 label_shape=None, classify=False):
    print('testing verification badcase..')
    data_list = data_set[0]
    issame_list = data_set[1]

    if data_extra is not None:
        _data_extra = nd.array(data_extra)
    time_consumed = 0.0
    if label_shape is None:
        _label = nd.ones((batch_size,))
    else:
        _label = nd.ones(label_shape)

    # get embedding list
    embeddings_list = []
    for i in xrange(len(data_list)):
        data = data_list[i]
        embeddings = None
        ba = 0
        while ba < data.shape[0]:
            bb = min(ba + batch_size, data.shape[0])
            count = bb - ba
            _data = nd.slice_axis(data, axis=0, begin=bb - batch_size, end=bb)
            time0 = datetime.datetime.now()
            if data_extra is None:
                db = mx.io.DataBatch(data=(_data,), label=(_label,))
            else:
                db = mx.io.DataBatch(data=(_data, _data_extra), label=(_label,))
            model.forward(db, is_train=False)
            net_out = model.get_outputs()
            _embeddings = net_out[0].asnumpy()

            time_now = datetime.datetime.now()
            diff = time_now - time0
            time_consumed += diff.total_seconds()
            if embeddings is None:
                embeddings = np.zeros((data.shape[0], _embeddings.shape[1]))
            embeddings[ba:bb, :] = _embeddings[(batch_size - count):, :]
            ba = bb
        embeddings_list.append(embeddings)

    embeddings = embeddings_list[0] + embeddings_list[1]
    embeddings = sklearn.preprocessing.normalize(embeddings)

    # calculate accuarcy
    data = data_list[0]
    embeddings1 = embeddings[0::2]
    embeddings2 = embeddings[1::2]
    assert (embeddings1.shape[0] == embeddings2.shape[0])
    assert (embeddings1.shape[1] == embeddings2.shape[1])

    if not classify:
        thresholds = np.arange(0, 4, 0.01)
        actual_issame = np.asarray(issame_list)
        nrof_folds = 10

        nrof_pairs = min(len(actual_issame), embeddings1.shape[0])
        nrof_thresholds = len(thresholds)
        k_fold = LFold(n_splits=nrof_folds, shuffle=False)

        tprs = np.zeros((nrof_folds, nrof_thresholds))
        fprs = np.zeros((nrof_folds, nrof_thresholds))
        accuracy = np.zeros((nrof_folds))
        indices = np.arange(nrof_pairs)

        diff = np.subtract(embeddings1, embeddings2)
        dist = np.sum(np.square(diff), 1)

        pouts = []
        nouts = []

        for fold_idx, (train_set, test_set) in enumerate(k_fold.split(indices)):

            # Find the best threshold for the fold
            acc_train = np.zeros((nrof_thresholds))
            for threshold_idx, threshold in enumerate(thresholds):
                p2 = dist[train_set]
                p3 = actual_issame[train_set]
                _, _, acc_train[threshold_idx] = calculate_accuracy(threshold, p2, p3)
            best_threshold_index = np.argmax(acc_train)
            for threshold_idx, threshold in enumerate(thresholds):
                tprs[fold_idx, threshold_idx], fprs[
                    fold_idx, threshold_idx], _ = calculate_accuracy(
                    threshold, dist[test_set], actual_issame[test_set])
            _, _, accuracy[fold_idx] = calculate_accuracy(thresholds[best_threshold_index],
                                                          dist[test_set],
                                                          actual_issame[test_set])
            best_threshold = thresholds[best_threshold_index]

            for iid in test_set:
                ida = iid * 2
                idb = ida + 1
                asame = actual_issame[iid]
                _dist = dist[iid]
                violate = _dist - best_threshold
                if not asame:
                    violate *= -1.0
                if violate > 0.0:
                    imga = data[ida].asnumpy().transpose((1, 2, 0))[..., ::-1]  # to bgr
                    imgb = data[idb].asnumpy().transpose((1, 2, 0))[..., ::-1]
                    if asame:
                        pouts.append((imga, imgb, _dist, best_threshold, ida, idb))
                    else:
                        nouts.append((imga, imgb, _dist, best_threshold, ida, idb))
                    with open('./freq.txt', 'a+') as file_freq:
                        file_freq.write('%d\n' % iid)
            with open('.freq.txt', 'a+') as file_freq:
                file_freq.write('\n')

        tpr = np.mean(tprs, 0)
        fpr = np.mean(fprs, 0)
        acc = np.mean(accuracy)
        pouts = sorted(pouts, key=lambda x: x[2], reverse=True)
        nouts = sorted(nouts, key=lambda x: x[2], reverse=False)

        draw(pouts, nouts, name, badcase_tag)
        print("postive: ", len(pouts), "negative: ", len(nouts))
        print("acc:", acc)
    else:
        # matrix[1][i][j] = distance between cls[i]image[1](embeddings[i * 2 + 1] and cls[j]image[0] (embeddings[j * 2]
        pouts = []
        nouts = []
        pairs_number = int(embeddings.shape[0] / 2)
        distance_matrix = np.zeros((2, pairs_number, pairs_number))
        for i in range(pairs_number):
            for j in range(i, pairs_number):
                distance_matrix[0][i][j] = np.dot(embeddings[i * 2], embeddings[j * 2 + 1])
                distance_matrix[1][j][i] = distance_matrix[0][i][j]
                distance_matrix[1][i][j] = np.dot(embeddings[i * 2 + 1], embeddings[j * 2])
                distance_matrix[0][i][j] = distance_matrix[1][i][j]

        # get min
        correct_number = 0.0
        divided_pairs_number = int(pairs_number / 2)
        for i in range(divided_pairs_number):
            answer_index = np.argmax(distance_matrix[0][i][:])
            if i == answer_index:
                correct_number += 1.0
            else:
                img_pivot = data[i * 2].asnumpy().transpose((1, 2, 0))[..., ::-1]
                img_true = data[i * 2 + 1].asnumpy().transpose((1, 2, 0))[..., ::-1]
                img_false = data[answer_index * 2 + 1].asnumpy().transpose((1, 2, 0))[..., ::-1]
                pouts.append((img_pivot, img_false, distance_matrix[0][i][answer_index], 1.0, i * 2,
                              answer_index * 2 + 1))
                nouts.append((img_pivot, img_true, distance_matrix[0][i][i], 1.0, i * 2, i * 2 + 1))

            answer_index = np.argmax(distance_matrix[1][i][:])
            if i == answer_index:
                correct_number += 1.0
            else:
                img_pivot = data[i * 2 + 1].asnumpy().transpose((1, 2, 0))[..., ::-1]
                img_true = data[i * 2].asnumpy().transpose((1, 2, 0))[..., ::-1]
                img_false = data[answer_index * 2].asnumpy().transpose((1, 2, 0))[..., ::-1]
                pouts.append((img_pivot, img_false, distance_matrix[1][i][answer_index], 1.0,
                              i * 2 + 1, answer_index * 2))
                nouts.append((img_pivot, img_true, distance_matrix[1][i][i], 1.0, i * 2 + 1, i * 2))
        draw(pouts, nouts, name + "_v1", badcase_tag)

        pouts = []
        nouts = []
        for i in range(divided_pairs_number, 2 * divided_pairs_number):
            answer_index = np.argmax(distance_matrix[0][i][:])
            if i == answer_index:
                correct_number += 1.0
            else:
                img_pivot = data[i * 2].asnumpy().transpose((1, 2, 0))[..., ::-1]
                img_true = data[i * 2 + 1].asnumpy().transpose((1, 2, 0))[..., ::-1]
                img_false = data[answer_index * 2 + 1].asnumpy().transpose((1, 2, 0))[..., ::-1]
                pouts.append((img_pivot, img_false, distance_matrix[0][i][answer_index], 1.0, i * 2,
                              answer_index * 2 + 1))
                nouts.append((img_pivot, img_true, distance_matrix[0][i][i], 1.0, i * 2, i * 2 + 1))
            # print(2 * i, answer_index * 2 + 1)

            answer_index = np.argmax(distance_matrix[1][i][:])
            if i == answer_index:
                correct_number += 1.0
            else:
                img_pivot = data[i * 2 + 1].asnumpy().transpose((1, 2, 0))[..., ::-1]
                img_true = data[i * 2].asnumpy().transpose((1, 2, 0))[..., ::-1]
                img_false = data[answer_index * 2].asnumpy().transpose((1, 2, 0))[..., ::-1]
                pouts.append((img_pivot, img_false, distance_matrix[1][i][answer_index], 1.0,
                              i * 2 + 1, answer_index * 2))
                nouts.append((img_pivot, img_true, distance_matrix[1][i][i], 1.0, i * 2 + 1, i * 2))
        draw(pouts, nouts, name + "_v2", badcase_tag)
        print("acc: %0.5f" % (correct_number / (pairs_number * 2.0)))


def draw(pouts, nouts, name, badcase_tag):
    # draw images
    gap = 10
    image_shape = (112, 224, 3)
    out_dir = "./badcases_" + badcase_tag
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    if len(nouts) > 0:
        threshold = nouts[0][3]
    else:
        threshold = pouts[-1][3]

    for item in [(pouts, 'positive(false_negative).png'), (nouts, 'negative(false_positive).png')]:
        cols = 4
        rows = 8000
        outs = item[0]
        if len(outs) == 0:
            continue

        _rows = int(math.ceil(len(outs) / cols))
        rows = min(rows, _rows)
        hack = {}

        if name.startswith('cfp') and item[1].startswith('pos'):
            hack = {0: 'manual/238_13.jpg.jpg', 6: 'manual/088_14.jpg.jpg',
                    10: 'manual/470_14.jpg.jpg',
                    25: 'manual/238_13.jpg.jpg', 28: 'manual/143_11.jpg.jpg'}

        filename = item[1]
        if len(name) > 0:
            filename = name + "_" + filename
        filename = os.path.join(out_dir, filename)
        img = np.zeros((image_shape[0] * rows + 20, image_shape[1] * cols + (cols - 1) * gap, 3),
                       dtype=np.uint8)
        img[:, :, :] = 255
        text_color = (153, 255, 51)
        for outi, out in enumerate(outs):
            row = outi // cols
            col = outi % cols
            if row == rows:
                break
            imga = out[0].copy()
            imgb = out[1].copy()
            if outi in hack:
                idx = out[4]
                aa = hack[outi]
                imgb = cv2.imread(aa)

            dist = out[2]
            _img = np.concatenate((imga, imgb), axis=1)

            k = ("%1.1f,%d,%d" % (dist, out[4], out[5]))

            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(_img, k, (40, image_shape[0] // 2 + 7), font, 0.6, text_color, 2)
            img[row * image_shape[0]:(row + 1) * image_shape[0],
            (col * image_shape[1] + gap * col):((col + 1) * image_shape[1] + gap * col), :] = _img

        font = cv2.FONT_HERSHEY_SIMPLEX
        k = "threshold: %.3f" % threshold
        cv2.putText(img, k, (img.shape[1] // 2 - 70, img.shape[0] - 5), font, 0.6, text_color, 2)
        cv2.imwrite(filename, img)


# load property: num_classes, image_height, image_width
def load_property(data_dir):
    prop = edict()
    for line in open(os.path.join(data_dir, 'property')):
        vec = line.strip().split(',')
        assert len(vec) == 3
        prop.num_classes = int(vec[0])
        prop.image_size = [int(vec[1]), int(vec[2])]
    return prop


# get model epoch number
def get_model_epochs(model_str, model_epoch_range_str):
    model_fields = model_str.split(',')
    model_prefix = model_str.split(',')[0]
    model_epochs = []
    model_prefix_list = []
    if len(model_fields) == 1:
        # read directory
        model_epoch_range = [int(x) for x in model_epoch_range_str.split(',')]
        assert (len(model_epoch_range) == 2)

        dir_path = os.path.dirname(model_prefix)
        for file_name in os.listdir(dir_path):
            if not file_name.endswith('.params'):
                continue
            file_path = os.path.join(dir_path, file_name)
            if file_path.startswith(model_prefix):
                epoch = int(file_name.split('.')[0].split('-')[1])
                if model_epoch_range[0] <= epoch <= model_epoch_range[1]:
                    model_epochs.append(epoch)
                    model_prefix_list.append(model_prefix)
        model_epochs = sorted(model_epochs)
    else:
        model_epochs = [int(x) for x in model_fields[1].split('|')]
        model_prefix_list = [model_prefix]
    return model_prefix_list, model_epochs


def main():
    # parse argument
    args = parse_argument()
    prop = load_property(args.data_dir)
    image_size = prop.image_size

    model_dir = os.path.dirname(args.model.split(',')[0])
    args.output_path = os.path.join(model_dir, args.output_path)

    # setup GPU and net
    ctx = []
    if len(args.gpu) > 0:
        ctx = [mx.gpu(int(i)) for i in args.gpu.split(',')]
    if len(ctx) == 0:
        ctx = [mx.cpu()]
        print('cpu number: 1')
    else:
        print('gpu number:', len(ctx))

    # setup models
    model_prefix_list = []
    model_epochs = []
    if os.path.exists(args.testing_model_list):
        with open(args.testing_model_list) as file_testing_model_list:
            lines = file_testing_model_list.read().split('\n')
            for line in lines:
                model_fields = line.split(',')
                if len(model_fields) == 2:
                    model_prefix_list.append(model_fields[0])
                    model_epochs.append(int(model_fields[1]))
                    print(model_fields)
        args.output_path = "./log.txt"
    else:
        model_prefix_list, model_epochs = get_model_epochs(args.model, args.model_epoch_range)

    # load dataset and model
    for dataset_name in args.target.split(','):
        # set evalation mode
        mode_classify = 0  # 0 normal(lfw), 1 classify, 2 classify megaface
        if "megaface" in dataset_name:
            mode_classify = 2
        elif "classify_suning_test" in dataset_name:
            mode_classify = 3
        elif "classify" in dataset_name:
            mode_classify = 1

        # load dataset
        dataset_path = os.path.join(args.data_dir, dataset_name + ".bin")
        if not os.path.exists(dataset_path):
            continue
        if mode_classify != 2:
            dataset = load_bin(dataset_path, image_size)
        else:
            dataset = load_megaface_bin(dataset_path, image_size)

        # evalation
        acc_list = []
        val_list = []
        time_total = 0.0
        for i in range(len(model_epochs)):
            model_prefix = model_prefix_list[i]
            model_epoch = model_epochs[i]
            sym, arg_params, aux_params = mx.model.load_checkpoint(model_prefix, model_epoch)
            all_layers = sym.get_internals()
            sym = all_layers['fc1_output']
            model = mx.mod.Module(symbol=sym, context=ctx, label_names=None)
            model.bind(data_shapes=[('data', (args.batch_size, 3, image_size[0], image_size[1]))])
            model.set_params(arg_params, aux_params)
            with open(args.output_path, 'a+') as output_file:
                output_file.write('%s\t%d\n' % (model_prefix, model_epoch))
            print("model: %s, %d" % (model_prefix, model_epoch))

            if args.mode == 0:
                val, val_std, acc, acc_std, far, emb, time_spend = test(dataset, model,
                                                                     args.batch_size,
                                                          args.nfolds, None, None, mode_classify)
                print('[%s]Acc: %1.5f+-%1.5f, Val: %1.5f+-%1.5f, Far: %.9f' % (dataset_name,
                                                                               acc, acc_std, val,
                                                                               val_std, far))
                time_total += time_spend

                with open(args.output_path, 'a+') as output_file:
                    output_file.write('%s\t%1.5f\t%1.5f\t%1.5f\t%1.5f\t%.9f\n' % (dataset_name,
                                                                                  acc, acc_std, val,
                                                                                  val_std, far))
                acc_list.append(acc)
                val_list.append(val)
            else:
                test_badcase(dataset, model, args.batch_size, args.target,
                             model_prefix.split('/')[-1] + "_" + str(model_epoch), None, None,
                             mode_classify)

        if len(acc_list) > 1 and args.mode == 0:
            print('[%s]Max Acc: %1.5f, Max Val: %1.5f' % (
            dataset_name, np.max(acc_list), np.max(val_list)))

        print("average time: ", time_total / len(model_epochs), " ms, models: ", len(model_epochs))


if __name__ == '__main__':
    main()
