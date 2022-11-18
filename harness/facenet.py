from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from re import T
from unittest import result

import numpy
import numpy as np
import argparse
import time
import os
import sys
import math
from sklearn import metrics
from scipy.optimize import brentq
from scipy import interpolate
from sklearn.model_selection import KFold
from tpu_perf.infer import SGInfer
from tpu_perf.harness import harness
import cv2


def lfw_read_pairs(pairs_filename):
    pairs = []
    with open(pairs_filename, 'r') as f:
        for line in f.readlines()[1:]:
            pair = line.strip().split()
            pairs.append(pair)
    return np.array(pairs)

def add_extension(path):
    if os.path.exists(path+'.jpg'):
        return path+'.jpg'
    elif os.path.exists(path+'.png'):
        return path+'.png'
    else:
        raise RuntimeError('No file "%s" with extension png or jpg.' % path)

def lfw_get_paths(lfw_dir, pairs):
    nrof_skipped_pairs = 0
    path_list = []
    issame_list = []
    for pair in pairs:
        if len(pair) == 3:
            path0 = add_extension(os.path.join(lfw_dir, pair[0], pair[0] + '_' + '%04d' % int(pair[1])))
            path1 = add_extension(os.path.join(lfw_dir, pair[0], pair[0] + '_' + '%04d' % int(pair[2])))
            issame = True
        elif len(pair) == 4:
            path0 = add_extension(os.path.join(lfw_dir, pair[0], pair[0] + '_' + '%04d' % int(pair[1])))
            path1 = add_extension(os.path.join(lfw_dir, pair[2], pair[2] + '_' + '%04d' % int(pair[3])))
            issame = False
        if os.path.exists(path0) and os.path.exists(path1):    # Only add the pair if both paths exist
            path_list += (path0,path1)
            issame_list.append(issame)
        else:
            nrof_skipped_pairs += 1
    if nrof_skipped_pairs>0:
        print('Skipped %d image pairs' % nrof_skipped_pairs)
    
    return path_list, issame_list


def distance(embeddings1, embeddings2, distance_metric=0):
    if distance_metric==0:
        # Euclidian distance
        diff = np.subtract(embeddings1, embeddings2)
        dist = np.sum(np.square(diff),1)
    elif distance_metric==1:
        # Distance based on cosine similarity
        dot = np.sum(np.multiply(embeddings1, embeddings2), axis=1)
        norm = np.linalg.norm(embeddings1, axis=1) * np.linalg.norm(embeddings2, axis=1)
        similarity = dot / norm
        dist = np.arccos(similarity) / math.pi
    else:
        raise 'Undefined distance metric %d' % distance_metric 
        
    return dist

def calculate_accuracy(threshold, dist, actual_issame, save=False):
    predict_issame = np.less(dist, threshold)
    tp = np.sum(np.logical_and(predict_issame, actual_issame))
    fp = np.sum(np.logical_and(predict_issame, np.logical_not(actual_issame)))
    tn = np.sum(np.logical_and(np.logical_not(predict_issame), np.logical_not(actual_issame)))
    fn = np.sum(np.logical_and(np.logical_not(predict_issame), actual_issame))
    # if save==True:
    #     with open('./output/predict.txt', 'a') as f:
    #         for i in predict_issame:
    #             f.write(str(i))
    #             f.write('\n')
  
    tpr = 0 if (tp+fn==0) else float(tp) / float(tp+fn)
    fpr = 0 if (fp+tn==0) else float(fp) / float(fp+tn)
    acc = float(tp+tn)/dist.size
    return tpr, fpr, acc

def calculate_val_far(threshold, dist, actual_issame):
    predict_issame = np.less(dist, threshold)
    true_accept = np.sum(np.logical_and(predict_issame, actual_issame))
    false_accept = np.sum(np.logical_and(predict_issame, np.logical_not(actual_issame)))
    n_same = np.sum(actual_issame)
    n_diff = np.sum(np.logical_not(actual_issame))
    val = float(true_accept) / float(n_same)
    far = float(false_accept) / float(n_diff)
    return val, far

def facenet_calculate_roc(thresholds, embeddings1, embeddings2, actual_issame, nrof_folds=10, distance_metric=0, subtract_mean=False):
    assert(embeddings1.shape[0] == embeddings2.shape[0])
    assert(embeddings1.shape[1] == embeddings2.shape[1])
    nrof_pairs = min(len(actual_issame), embeddings1.shape[0])
    nrof_thresholds = len(thresholds)
    k_fold = KFold(n_splits=nrof_folds, shuffle=False)
    
    tprs = np.zeros((nrof_folds,nrof_thresholds))
    fprs = np.zeros((nrof_folds,nrof_thresholds))
    accuracy = np.zeros((nrof_folds))
    
    indices = np.arange(nrof_pairs)
    
    for fold_idx, (train_set, test_set) in enumerate(k_fold.split(indices)):
        if subtract_mean:
            mean = np.mean(np.concatenate([embeddings1[train_set], embeddings2[train_set]]), axis=0)
        else:
          mean = 0.0
        dist = distance(embeddings1-mean, embeddings2-mean, distance_metric)
        
        # Find the best threshold for the fold
        acc_train = np.zeros((nrof_thresholds))
        for threshold_idx, threshold in enumerate(thresholds):
            _, _, acc_train[threshold_idx] = calculate_accuracy(threshold, dist[train_set], actual_issame[train_set])
        best_threshold_index = np.argmax(acc_train)
        for threshold_idx, threshold in enumerate(thresholds):
            tprs[fold_idx,threshold_idx], fprs[fold_idx,threshold_idx], _ = calculate_accuracy(threshold, dist[test_set], actual_issame[test_set])
        _, _, accuracy[fold_idx] = calculate_accuracy(thresholds[best_threshold_index], dist[test_set], actual_issame[test_set], save=True)
          
        tpr = np.mean(tprs,0)
        fpr = np.mean(fprs,0)
    return tpr, fpr, accuracy

def facenet_calculate_val(thresholds, embeddings1, embeddings2, actual_issame, far_target, nrof_folds=10, distance_metric=0, subtract_mean=False):
    assert(embeddings1.shape[0] == embeddings2.shape[0])
    assert(embeddings1.shape[1] == embeddings2.shape[1])
    nrof_pairs = min(len(actual_issame), embeddings1.shape[0])
    nrof_thresholds = len(thresholds)
    k_fold = KFold(n_splits=nrof_folds, shuffle=False)
    
    val = np.zeros(nrof_folds)
    far = np.zeros(nrof_folds)
    
    indices = np.arange(nrof_pairs)
    
    for fold_idx, (train_set, test_set) in enumerate(k_fold.split(indices)):
        if subtract_mean:
            mean = np.mean(np.concatenate([embeddings1[train_set], embeddings2[train_set]]), axis=0)
        else:
          mean = 0.0
        dist = distance(embeddings1-mean, embeddings2-mean, distance_metric)
      
        # Find the threshold that gives FAR = far_target
        far_train = np.zeros(nrof_thresholds)
        for threshold_idx, threshold in enumerate(thresholds):
            _, far_train[threshold_idx] = calculate_val_far(threshold, dist[train_set], actual_issame[train_set])
        if np.max(far_train)>=far_target:
            f = interpolate.interp1d(far_train, thresholds, kind='slinear')
            threshold = f(far_target)
        else:
            threshold = 0.0
    
        val[fold_idx], far[fold_idx] = calculate_val_far(threshold, dist[test_set], actual_issame[test_set])
  
    val_mean = np.mean(val)
    far_mean = np.mean(far)
    val_std = np.std(val)
    return val_mean, val_std, far_mean

def lfw_evaluate(embeddings, actual_issame, nrof_folds=10, distance_metric=0, subtract_mean=False):
    # Calculate evaluation metrics
    thresholds = np.arange(0, 4, 0.01)
    embeddings1 = embeddings[0::2]
    embeddings2 = embeddings[1::2]
    tpr, fpr, accuracy = facenet_calculate_roc(thresholds, embeddings1, embeddings2,
        np.asarray(actual_issame), nrof_folds=nrof_folds, distance_metric=distance_metric, subtract_mean=subtract_mean)
    thresholds = np.arange(0, 4, 0.001)
    val, val_std, far = facenet_calculate_val(thresholds, embeddings1, embeddings2,
        np.asarray(actual_issame), 1e-3, nrof_folds=nrof_folds, distance_metric=distance_metric, subtract_mean=subtract_mean)
    return tpr, fpr, accuracy, val, val_std, far


from multiprocessing import Process, Queue
import threading
import logging

class Runner:
    def __init__(self, bmodel, devices, image_paths_array, threads):
        self.image_paths_array = image_paths_array

        self.model = SGInfer(bmodel)
        self.input_info = self.model.get_input_info()
        self.embeddings = np.zeros((len(image_paths_array), 512))
        self.batch_size = 1
        self.batch_num = len(image_paths_array)
        self.ids = dict()
        self.result = []
        self.q = Queue(maxsize=len(image_paths_array))
        print('Preprocess on CPU...')
        self.preprocess()
        print('Preprocess finished. Inference begin...')
        self.time_start = time.time()
        self.time_sum = 0
        self.relay = threading.Thread(target=self.relay)
        self.relay.start()
        self.post = threading.Thread(target=self.postprocess)
        self.post.start()
        self.join()


    def preprocess(self):
        input_info = next(iter(self.input_info.values()))
        batch_size = input_info['shape'][0]
        self.batch_size = batch_size
        input_scale = input_info['scale']
        is_fp32 = input_scale == 1
        dtype=np.float32
        num_img = len(self.image_paths_array)
        epoc = int(num_img/batch_size)
        self.batch_num = epoc
        img_array = np.zeros((input_info['shape'][0],input_info['shape'][1],input_info['shape'][2],input_info['shape'][3]))

        for i in range(epoc):
            for j in range(batch_size):
                img = cv2.imread(self.image_paths_array[i*batch_size+j][0])
                img = (img - 127.5)/128.0
                if (i*batch_size+j) % 2 == 1:
                    img = cv2.flip(img, 1)
                img_array[j]=img
            img_array = np.array(img_array, dtype=np.float32)
            if not is_fp32:
                img_array *= input_scale
                dtype = np.int8
            self.q.put((np.ascontiguousarray(img_array.astype(dtype)), i))


    def relay(self):
        try:
            while True:
                task = self.q.get()
                if task is None:
                    break
                self._relay(task)
        except Exception as err:
            logging.error(f'Relay task failed, {err}')
            raise

    def _relay(self, task):
        data, ids = task
        task_id = self.model.put(data)
        self.ids[task_id] = ids

    def _postprocess(self):
        arg_results = dict()
        while True:
            task_id, results, valid = self.model.get()
            if task_id == 0:
                break
            self.result.append((task_id, results))

    def postprocess(self):
        try:
            self._postprocess()
        except Exception as err:
            logging.error(f'Task postprocess failed, {err}')
            raise

    def join(self):
        self.q.put(None)
        self.relay.join()
        self.model.put()
        self.post.join()
        self.time_end = time.time()
        self.time_sum = self.time_end-self.time_start
        print('Inference time: ',self.time_sum)

    def get_output(self):
        epoc = int(len(self.image_paths_array)/self.batch_size)

        for task_id, results in self.result:
            output = results[0]
            i = self.ids[task_id]
            self.embeddings[i*self.batch_size:i*self.batch_size+self.batch_size] = output
        return self.embeddings.copy()


@harness('facenet')
def harness_facenet(tree, config, args):
    # print(args)
    lfw_dir = tree.expand_variables(config, args['lfw_dir'])
    lfw_pairs = tree.expand_variables(config, args['lfw_pairs'])
    bmodel = tree.expand_variables(config, args['bmodel'])
    # batch_size = tree.expand_variables(config, args['batch_size'])
    lfw_nrof_folds = tree.expand_variables(config, args['lfw_nrof_folds'])
    name = tree.expand_variables(config, args['name'])
    accuracy, infer_time = main(lfw_dir, lfw_pairs, bmodel, lfw_nrof_folds= lfw_nrof_folds, name=name)
    # return {'accuracy':f'{accuracy:.2%}', 'time': infer_time}
    return {'accuracy':f'{accuracy:.2%}'}


def main(lfw_dir, lfw_pairs, bmodel_path, devices=0, threads=8, batch_size=4, lfw_nrof_folds=2, name='', distance_metric=1, use_flipped_images=True, subtract_mean=True, use_fixed_image_standardization=True):
    # Read the file containing the pairs used for testing
    pairs = lfw_read_pairs(os.path.expanduser(lfw_pairs))
    # Get the paths for the corresponding images
    paths, actual_issame = lfw_get_paths(os.path.expanduser(lfw_dir), pairs)
    print('Runnning forward pass on LFW images')
    nrof_embeddings = len(actual_issame) * 2  
    nrof_flips = 2 if use_flipped_images else 1  
    nrof_images = nrof_embeddings * nrof_flips
    image_paths_array = np.expand_dims(np.repeat(np.array(paths), nrof_flips), 1)
    embedding_size = 512  # 512
    emb_array = np.zeros((nrof_images, embedding_size))  # 24000*512

    runner = Runner(
        bmodel_path, devices, image_paths_array, threads = threads)
    emb_array = runner.get_output()
    
    # np.save('./output/' + name + '_emb_array.npy', emb_array)

    print('Validation...')
    embeddings = np.zeros((nrof_embeddings, embedding_size * nrof_flips))  

    if use_flipped_images:  # True
        # Concatenate embeddings for flipped and non flipped version of the images
        embeddings[:, :embedding_size] = emb_array[0::2, :]
        embeddings[:, embedding_size:] = emb_array[1::2, :]
    else:
        embeddings = emb_array
    tpr, fpr, accuracy, val, val_std, far = lfw_evaluate(embeddings, actual_issame, nrof_folds=lfw_nrof_folds,
                                                         distance_metric=distance_metric,
                                                         subtract_mean=subtract_mean)
    print('Accuracy: %2.5f+-%2.5f' % (np.mean(accuracy), np.std(accuracy)))
    print('Validation rate: %2.5f+-%2.5f @ FAR=%2.5f' % (val, val_std, far))

    # predict_labels = []
    # with open('./output/predict.txt', 'r') as f:
    #     lines = f.readlines()
    #     for line in lines:
    #         line = line.strip('\n')
    #         predict_labels.append(line)
    
    # with open('./output/' + name + '_predict.csv', 'w') as f:
    #     f.write('index,predict,label')
    #     f.write('\n')
    #     for i in range(len(actual_issame)):
    #         f.write(str(i)+','+predict_labels[i]+','+str(actual_issame[i]))
    #         f.write('\n')
    
    # with open('./output/predict.txt', 'w') as f:
    #     print('finished.')
    # auc = metrics.auc(fpr, tpr)
    # print('Area Under Curve (AUC): %1.3f' % auc)
    # eer = brentq(lambda x: 1. - x - interpolate.interp1d(fpr, tpr)(x), 0., 1.)
    # print('Equal Error Rate (EER): %1.3f' % eer)
    return np.mean(accuracy), runner.time_sum


def parse_arguments(argv):
    parser = argparse.ArgumentParser()

    parser.add_argument('--lfw_dir', type=str, 
                        help='Path to the data directory containing aligned LFW face patches.')
    parser.add_argument('--lfw_pairs', type=str,
                        help='The file containing the pairs to use for validation.', default='data/pairs.txt')
    parser.add_argument('--bmodel', type=str,
                        help='Path to bmodel.', default='./compilation.bmodel')
    parser.add_argument('--devices', type=int,
                        help='device id.', default=0)
    parser.add_argument('--batch_size', type=int,
                        help='Batch size.', default=4)
    parser.add_argument('--threads', type=int,
                        help='Num of threads.', default=8)
    parser.add_argument('--lfw_nrof_folds', type=int,
                        help='Number of folds to use for cross validation. Mainly used for testing.', default=10)
    parser.add_argument('--distance_metric', type=int,
                        help='Distance metric  0:euclidian, 1:cosine similarity.', default=1)
    parser.add_argument('--use_flipped_images',
                        help='Concatenates embeddings for the image and its horizontally flipped counterpart.',
                        action='store_true')
    parser.add_argument('--subtract_mean',
                        help='Subtract feature mean before calculating distance.', action='store_true')
    parser.add_argument('--use_fixed_image_standardization',
                        help='Performs fixed standardization of images.', action='store_true')
    return parser.parse_args(argv)


if __name__ == '__main__':
    args = parse_arguments(sys.argv[1:])
    main(args.lfw_dir, args.lfw_pairs, args.bmodel, args.devices , args.threads, args.batch_size, args.lfw_nrof_folds, args.distance_metric, args.use_flipped_images, args.subtract_mean, args.use_fixed_image_standardization)