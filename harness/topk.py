import logging
import sys
import os
import cv2
import numbers
import numpy as np

from tpu_perf.infer import SGInfer

# resize
def resize(image, size, interpolation=cv2.INTER_LINEAR):
    assert isinstance(image, np.ndarray)

    if isinstance(size, int):
        h, w = image.shape[:2]
        if (w <= h and w == size) or (h <= w and h == size):
            return image
        if w < h:
            ow = size
            oh = int(size * h / w)
            return cv2.resize(image, (ow, oh), interpolation=interpolation)
        else:
            oh = size
            ow = int(size * w / h)
            return cv2.resize(image, (ow, oh), interpolation=interpolation)
    else:
        return cv2.resize(image, tuple(size[::-1]), interpolation=interpolation)

# crop
def crop(image, y, x, h, w):
    assert isinstance(image, np.ndarray)

    return image[y:y+h, x:x+w]

# center crop
def center_crop(image, size):
    assert isinstance(image, np.ndarray)

    if isinstance(size, numbers.Number):
        size = (int(size), int(size))

    h, w = image.shape[:2]
    oh, ow = size
    x = int(round((w - ow) / 2.))
    y = int(round((h - oh) / 2.))

    return crop(image, y, x, oh, ow)

def read_image_list(fn):
    with open(fn) as f:
        for line in f:
            line = line.strip(' \n')
            if not line:
                break
            image_fn, label = line.split()
            yield image_fn, int(label)

import math
def sever(l, n):
    step = math.ceil(len(l) / n)
    return [l[i * step : (i + 1) * step] for i in range(0, n)]

from multiprocessing import Process, Queue
import threading

class Runner:
    def __init__(self, bmodel, val_path, list_file, config, threads):
        self.val_path = val_path
        self.config = config
        self.model = SGInfer(bmodel)
        self.input_info = self.model.get_input_info()
        self.size = config.get('size', 224)

        self.labels = dict()
        self.stats = dict(count = 0, top1 = 0, top5 = 0)

        pairs = [pair for pair in read_image_list(list_file)]
        parts = sever(pairs, threads)
        self.pre_procs = []
        self.q = Queue(maxsize=threads * 2)
        for part in parts:
            p = Process(target=self.preprocess, args=(part,))
            self.pre_procs.append(p)
            p.start()

        self.relay = threading.Thread(target=self.relay)
        self.relay.start()

        self.post = threading.Thread(target=self.postprocess)
        self.post.start()

    def preprocess(self, part):
        try:
            self._preprocess(part)
        except Exception as err:
            logging.error(f'Preprocess failed, {err}')
            raise

    def _preprocess(self, part):
        input_info = next(iter(self.input_info.values()))
        batch_size = input_info['shape'][0]
        input_scale = input_info['scale']
        is_fp32 = input_scale == 1

        bulk_label = []
        bulk = []
        def enqueue():
            nonlocal bulk_label, bulk
            if not bulk:
                return
            self.q.put((np.stack(bulk), bulk_label))
            bulk = []
            bulk_label = []
        for fn, label in part:
            path = os.path.join(self.val_path, fn)
            if not os.path.exists(path):
                logging.error(f'File {path} does no exist')
                raise RuntimeError('File not exist')
            img = cv2.imread(path)

            resized = resize(img, 256)
            cropped = center_crop(resized, self.size)
            data = cropped.astype(np.float32)
            if 'mean' in self.config:
                data -= self.config['mean']
            if 'scale' in self.config:
                data *= self.config['scale']
            data = data.transpose([2, 0, 1])

            dtype = np.float32
            if not is_fp32:
                data *= input_scale
                dtype = np.int8

            bulk.append(data.astype(dtype))
            bulk_label.append(label)
            if len(bulk) < batch_size:
                continue
            enqueue()
        enqueue()

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
        data, labels = task
        task_id = self.model.put(data)
        self.labels[task_id] = labels

    def _postprocess(self):
        arg_results = dict()
        while True:
            task_id, results, valid = self.model.get()
            if task_id == 0:
                break
            output = results[0]
            output = output.reshape(output.shape[0], -1)
            argmaxs = np.argmax(output, axis=-1)
            topks = np.argpartition(output, -5)[:, -5:]
            arg_results[task_id] = (argmaxs, topks)
        for task_id, (argmaxs, topks) in arg_results.items():
            labels = self.labels.pop(task_id)
            for label, argmax, topk in zip(labels, argmaxs, topks):
                self.stats['count'] += 1
                if label == argmax:
                    self.stats['top1'] += 1
                if label in topk:
                    self.stats['top5'] += 1

    def postprocess(self):
        try:
            self._postprocess()
        except Exception as err:
            logging.error(f'Task postprocess failed, {err}')
            raise

    def join(self):
        for p in self.pre_procs:
            p.join()
        self.q.put(None)
        self.relay.join()
        self.model.put()
        self.post.join()

    def get_stats(self):
        stats = self.stats.copy()
        count = stats.pop('count')
        for k in stats:
            stats[k] /= count
        return stats

from tpu_perf.harness import harness

@harness('topk')
def harness_main(tree, config, args):
    input_config = config['dataset']
    scale = input_config['scale']
    mean = input_config['mean']
    size = input_config['size']
    pre_config = dict(mean=mean, scale=scale, size=size)
    val_path = tree.expand_variables(config, input_config['image_path'])
    list_file = tree.expand_variables(config, input_config['image_label'])
    bmodel = tree.expand_variables(config, args['bmodel'])
    runner = Runner(
        bmodel, val_path, list_file, pre_config,
        args.get('threads', 8))
    runner.join()
    return runner.get_stats()

def main():
    import argparse
    parser = argparse.ArgumentParser(description='tpu_perf topk harness')
    parser.add_argument(
        '--bmodel', type=str, help='Bmodel path')
    parser.add_argument(
        '--image_path', type=str, help='image set path')
    parser.add_argument(
        '--list_file', type=str, help='List file (with labels) path')
    parser.add_argument(
        '--mean', required=True, type=str, help='Mean value like 128,128,128')
    parser.add_argument(
        '--scale', required=True, type=float, help='Float scale value')
    parser.add_argument(
        '--size', required=True, type=int, help='Crop size. (Resized to 256 then crop)')
    parser.add_argument('--threads', type=int, default=4)
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format='[%(levelname)s %(filename)s:%(lineno)d] %(message)s')
    mean = [float(v) for v in args.mean.split(',')]
    if len(mean) != 3:
        mean = mean[:1] * 3
    config = dict(mean=mean, scale=args.scale, size=args.size)
    print(config)
    runner = Runner(args.bmodel, args.image_path, args.list_file, config, args.threads)
    runner.join()
    print(runner.get_stats())

if __name__ == '__main__':
    main()
