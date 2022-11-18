import numpy as np
import argparse
import time
import os
import sys
import math
from scipy import interpolate
from sklearn.model_selection import KFold
from tpu_perf.infer import SGInfer
from tpu_perf.harness import harness
import cv2
import tarfile
from PIL import Image, TarIO

def lfw_read_pairs(pairs_filename):
    pairs = []
    with open(pairs_filename, 'r') as f:
        for line in f.readlines()[1:]:
            pair = line.strip().split()
            pairs.append(pair)
    return np.array(pairs)

def add_extension(path, name_list):
    if path+'.jpg' in name_list:
        return path+'.jpg'
    elif path+'.png' in name_list:
        return path+'.png'
    else:
        raise RuntimeError('No file "%s" with extension png or jpg.' % path)

def lfw_get_paths(name_list, lfw_dir, pairs):
    nrof_skipped_pairs = 0
    path_list = []
    issame_list = []
    for pair in pairs:
        if len(pair) == 3:
            path0 = add_extension(os.path.join(lfw_dir, pair[0], pair[0] + '_' + '%04d' % int(pair[1])), name_list)
            path1 = add_extension(os.path.join(lfw_dir, pair[0], pair[0] + '_' + '%04d' % int(pair[2])), name_list)
            issame = True
        elif len(pair) == 4:
            path0 = add_extension(os.path.join(lfw_dir, pair[0], pair[0] + '_' + '%04d' % int(pair[1])), name_list)
            path1 = add_extension(os.path.join(lfw_dir, pair[2], pair[2] + '_' + '%04d' % int(pair[3])), name_list)
            issame = False
        if path0 in name_list and path1 in name_list:    # Only add the pair if both paths exist
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

from multiprocessing import Queue
import threading

class Runner:
    def __init__(self, bmodel, devices, lfw_dir, image_paths_array):
        self.lfw_dir = lfw_dir
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
                fp = TarIO.TarIO(self.lfw_dir, self.image_paths_array[i*batch_size+j][0])
                img = Image.open(fp)
                img = np.asarray(img)
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
    batch_size = tree.expand_variables(config, args['batch_size'])
    lfw_nrof_folds = tree.expand_variables(config, args['lfw_nrof_folds'])
    name = tree.expand_variables(config, args['name'])
    accuracy, infer_time = main(lfw_dir, lfw_pairs, bmodel, lfw_nrof_folds= lfw_nrof_folds, name=name)
    return {'accuracy':f'{accuracy:.2%}'}

def main(lfw_dir, lfw_pairs, bmodel_path, devices=0, batch_size=4, lfw_nrof_folds=2, name='', distance_metric=1, use_flipped_images=True, subtract_mean=True, use_fixed_image_standardization=True):

    name_list = []
    with tarfile.open(lfw_dir) as ft:
        for i in ft.getmembers():
            name_list.append(i.name)
    # Read the file containing the pairs used for testing
    pairs = lfw_read_pairs(os.path.expanduser(lfw_pairs))
    # Get the paths for the corresponding images
    paths, actual_issame = lfw_get_paths(name_list, os.path.expanduser('lfw_mtcnnpy_160/'), pairs)
    print('Runnning forward pass on LFW images')
    nrof_embeddings = len(actual_issame) * 2  
    nrof_flips = 2 if use_flipped_images else 1  
    nrof_images = nrof_embeddings * nrof_flips
    image_paths_array = np.expand_dims(np.repeat(np.array(paths), nrof_flips), 1)
    embedding_size = 512  
    emb_array = np.zeros((nrof_images, embedding_size))  # 24000*512
    runner = Runner(
        bmodel_path, devices, lfw_dir, image_paths_array)
    emb_array = runner.get_output()
    print('Validation...')
    embeddings = np.zeros((nrof_embeddings, embedding_size * nrof_flips))  

    if use_flipped_images: 
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
    main(args.lfw_dir, args.lfw_pairs, args.bmodel, args.devices , args.batch_size, args.lfw_nrof_folds, args.distance_metric, args.use_flipped_images, args.subtract_mean, args.use_fixed_image_standardization)