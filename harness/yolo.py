import numpy as np
import sys
import os
import argparse
import json
import cv2
import numpy as np
import time
import datetime
import random
from tpu_perf.infer import SGInfer
from tpu_perf.harness import harness
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from collections import OrderedDict
import argparse
from pathlib import Path

class COCOResults(object):
    METRICS = {
        "bbox": ["AP", "AP50", "AP75", "APs", "APm", "APl"],
        "segm": ["AP", "AP50", "AP75", "APs", "APm", "APl"],
        "box_proposal": [
            "AR@100",
            "ARs@100",
            "ARm@100",
            "ARl@100",
            "AR@1000",
            "ARs@1000",
            "ARm@1000",
            "ARl@1000",
        ],
        "keypoints": ["AP", "AP50", "AP75", "APm", "APl"],
    }

    def __init__(self, *iou_types):
        allowed_types = ("box_proposal", "bbox", "segm", "keypoints")
        assert all(iou_type in allowed_types for iou_type in iou_types)
        results = OrderedDict()
        for iou_type in iou_types:
            results[iou_type] = OrderedDict(
                [(metric, -1) for metric in COCOResults.METRICS[iou_type]]
            )
        self.results = results

    def update(self, coco_eval):
        if coco_eval is None:
            return
        from pycocotools.cocoeval import COCOeval

        assert isinstance(coco_eval, COCOeval)
        s = coco_eval.stats
        iou_type = coco_eval.params.iouType
        res = self.results[iou_type]
        metrics = COCOResults.METRICS[iou_type]
        for idx, metric in enumerate(metrics):
            res[metric] = s[idx]

    def __repr__(self):
        results = '\n'
        for task, metrics in self.results.items():
            results += 'Task: {}\n'.format(task)
            metric_names = metrics.keys()
            metric_vals = ['{:.4f}'.format(v) for v in metrics.values()]
            results += (', '.join(metric_names) + '\n')
            results += (', '.join(metric_vals) + '\n')
        return results

def preprocess(img, new_shape):
  """ Preprocessing of a frame.

  Args:
    image : np.array, input image
    detected_size : list, yolov3 detection input size

  Returns:
    Preprocessed data.
  """
  shape = img.shape[:2]
  if isinstance(new_shape, int):
      new_shape = (new_shape, new_shape)

  ######

  r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
  ratio = r, r
  new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
  dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]
  dw, dh = dw / 2, dh / 2
  if shape[::-1] != new_unpad:
      img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
  top, bottom = int(round(dh-0.1)), int(round(dh+0.1))
  left, right = int(round(dw-0.1)), int(round(dw+0.1))
  img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(114, 114, 114))
  return img[:, :, ::-1].transpose([2, 0, 1]).astype(np.float32) / 255 , ratio[0], (top, left)

CONF_THRESH = 0.0

ANCHORS = np.array([
    [10,  13, 16,  30,  33,  23 ],
    [30,  61, 62,  45,  59,  119],
    [116, 90, 156, 198, 373, 326]
])
ANCHOR_GRID = ANCHORS.reshape(3, -1, 2).reshape(3, 1, -1, 1, 1, 2)
STRIDES = [8, 16, 32]
CONF_THR = 0.6
IOU_THR = 0.5


def xywh2xyxy(x):
    # Convert nx4 boxes from [x, y, w, h] to [x1, y1, x2, y2] where xy1=top-left, xy2=bottom-right
    y = np.zeros_like(x)
    y[:, 0] = x[:, 0] - x[:, 2] / 2  # top left x
    y[:, 1] = x[:, 1] - x[:, 3] / 2  # top left y
    y[:, 2] = x[:, 0] + x[:, 2] / 2  # bottom right x
    y[:, 3] = x[:, 1] + x[:, 3] / 2  # bottom right y
    return y

def coco80_to_coco91_class():  # converts 80-index (val2014) to 91-index (paper)
    # https://tech.amikelive.com/node-718/what-object-categories-labels-are-in-coco-dataset/
    # a = np.loadtxt('data/coco.names', dtype='str', delimiter='\n')
    # b = np.loadtxt('data/coco_paper.names', dtype='str', delimiter='\n')
    # x1 = [list(a[i] == b).index(True) + 1 for i in range(80)]  # darknet to coco
    # x2 = [list(b[i] == a).index(True) if any(b[i] == a) else None for i in range(91)]  # coco to darknet
    x = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 27, 28, 31, 32, 33, 34,
         35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63,
         64, 65, 67, 70, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 84, 85, 86, 87, 88, 89, 90]
    return x

def sigmoid(x):
    s = 1 / (1 + np.exp(-x))
    return s


def make_grid(nx, ny):
    z = np.stack(np.meshgrid(np.arange(nx), np.arange(ny)), 2)
    return z.reshape(1, 1, ny, nx, 2).astype(np.float32)


def predict_preprocess(x):
    for i in range(len(x)):
        bs, na, ny, nx, no = x[i].shape
        grid = make_grid(nx, ny)
        x[i] = sigmoid(x[i])
        x[i][..., 0:2] = (x[i][..., 0:2] * 2. - 0.5 + grid) * STRIDES[i]
        x[i][..., 2:4] = (x[i][..., 2:4] * 2) ** 2 * ANCHOR_GRID[i]
        x[i] = x[i].reshape(bs, -1, no)
    return np.concatenate(x, 1)

def _nms(dets, scores, prob_threshold):
    #import pdb; pdb.set_trace()
    x1 = dets[:, 0]
    y1 = dets[:, 1]
    x2 = dets[:, 2]
    y2 = dets[:, 3]

    areas = (x2 - x1 + 1) * (y2 - y1 + 1)

    score_index = np.argsort(scores)[::-1]

    keep = []

    while score_index.size > 0:
        max_index = score_index[0]
        # 最大的肯定是需要的框
        keep.append(max_index)
        xx1 = np.maximum(x1[max_index], x1[score_index[1:]])
        yy1 = np.maximum(y1[max_index], y1[score_index[1:]])
        xx2 = np.minimum(x2[max_index], x2[score_index[1:]])
        yy2 = np.minimum(y2[max_index], y2[score_index[1:]])

        width = np.maximum(0.0, xx2 - xx1 + 1)
        height = np.maximum(0.0, yy2 - yy1 + 1)

        union = width * height

        iou = union / (areas[max_index] + areas[score_index[1:]] - union)
        ids = np.where(iou < prob_threshold)[0]
        # 以为算iou的时候没把第一个参考框索引考虑进来，所以这里都要+1
        score_index = score_index[ids+1]
    return keep

def non_max_suppression(prediction, conf_thres=0.001, iou_thres=0.3, classes=None):
    """Performs Non-Maximum Suppression (NMS) on inference results

    Returns:
         detections with shape: nx6 (x1, y1, x2, y2, conf, cls)
    """
    if(len(prediction)>1):
        prediction = [prediction['1'], prediction['2'], prediction['3']]
        prediction = predict_preprocess(prediction)
    else:
        prediction = [prediction['1']]
        prediction = np.concatenate(prediction,1)
    nc = prediction[0].shape[1] - 5  # number of classes
    xc = prediction[..., 4] > conf_thres  # candidates

    # Settings
    min_wh, max_wh = 2, 4096  # (pixels) minimum and maximum box width and height
    max_det = 300  # maximum number of detections per image
    time_limit = 10.0  # seconds to quit after
    redundant = True  # require redundant detections
    multi_label = nc > 1  # multiple labels per box (adds 0.5ms/img)

    t = time.time()
    output = [None] * prediction.shape[0]
    for xi, x in enumerate(prediction):  # image index, image inference
        # Apply constraints
        x = x[xc[xi]]  # confidence

        # If none remain process next image
        if not x.shape[0]:
            continue

        # Compute conf
        x[:, 5:] *= x[:, 4:5]  # conf = obj_conf * cls_conf

        # Box (center x, center y, width, height) to (x1, y1, x2, y2)
        box = xywh2xyxy(x[:, :4])

        # Detections matrix nx6 (xyxy, conf, cls)
        if multi_label:
            i, j = np.stack((x[:, 5:] > conf_thres).nonzero())
            x = np.concatenate((box[i], x[i, j + 5, None], j[:, None].astype(np.float32)), 1)
        else:  # best class only
            conf = x[:, 5:].max(1, keepdims=True)
            j = x[:, 5:].argmax(1).reshape(-1, 1)
            x = np.concatenate((box, conf, j.astype(np.float32)), 1)[conf.reshape(-1) > conf_thres]

        # Filter by class
        if classes:
            x = x[(x[:, 5:6] == np.array(classes)).any(1)]

        # If none remain process next image
        n = x.shape[0]  # number of boxes
        if not n:
            continue

        # Batched NMS
        c = x[:, 5:6] * max_wh # classes
        boxes, scores = x[:, :4] + c, x[:, 4]  # boxes (offset by class), scores
        i = _nms(boxes, scores, iou_thres)
        if len(i) > max_det:  # limit detections
            i = i[:max_det]

        output[xi] = x[i]
        if (time.time() - t) > time_limit:
            break  # time limit exceeded

    return output



def inference(net, input_path, loops, tpu_id):
  """ Load a bmodel and do inference.
  Args:
   bmodel_path: Path to bmodel
   input_path: Path to input file
   loops: Number of loops to run
   tpu_id: ID of TPU to use

  Returns:
    True for success and False for failure
  """
  # set configurations
  load_from_file = True
  threshold = 0.001
  nms_threshold = 0.6
  num_classes = 2

  cap = cv2.VideoCapture(input_path)
  # init Engine and load bmodel
  # get model info

  input_info = net.get_input_info()
  input_info = next(iter(input_info.values()))
  input_scale = input_info['scale']
  is_fp32 = input_scale == 1
  input_h = input_info['shape'][2]
  input_w = input_info['shape'][3]

  status = True
  # pipeline of inference
  for i in range(loops):
    # read an image
    img = cv2.imread(input_path)
    data, ratio, (top, left) = preprocess(img, (input_h,input_w))

    input_data = np.array([data], dtype=np.float32)
    dtype = np.float32
    if not is_fp32:
      input_data *= input_scale
      dtype = np.int8

    task_id = net.put(input_data.astype(dtype))
    task_id, results,valid = net.get()
    output={}
    if (len(results) > 1):
        output['1'] = results[0]
        output['2'] = results[1]
        output['3'] = results[2]
    else:
        output['1'] = results[0]

    prediction = non_max_suppression(output, conf_thres=threshold, iou_thres=nms_threshold, classes=None)
    for i in prediction:
      if i is None:
        continue
      for j in i:
        bbox = j[:4]
        bbox[0::2] -= left
        bbox[1::2] -= top
        bbox /= ratio

  cap.release()
  return prediction

def process(bmodel, devices, imagedir, anno):
  with open(anno) as g:
    js = json.load(g)
  preds = []
  if os.path.isfile(bmodel):
    net = SGInfer(bmodel, devices=devices)
  else:
    print("no bmodel!")
    sys.exit(1)
  tested_file = []
  totalimages=len(js['images'])
  idx=np.arange(totalimages)
  samples = 5000
  if samples>1:
    random.seed(1686)
    random.shuffle(idx)
    idx=idx<=samples+128
  proc=0
  processed=0
  for img in js['images']:
    print(".",end='',flush=True)
    img_p = img['file_name']
    if samples < 0:
      if not os.path.isfile('/'.join((imagedir,img_p))):
        continue
    else:
      if not idx[proc]:
        proc=proc+1
        continue
    tested_file.append(img_p)
    processed=processed+1
    proc=proc+1
    pred = inference(net, os.path.join(imagedir, img_p), 1, 0)
    coco91class = coco80_to_coco91_class()
    for pp in pred:
      if pp is None:
        continue
      for p in pp:
        bbox = p[:4]
        prob = float(p[4])
        clse = int(p[5])
        preds.append(dict(
          image_id=img['id'],
          category_id=coco91class[clse],
          bbox=[float(bbox[0]), float(bbox[1]), float(bbox[2]-bbox[0]), float(bbox[3]-bbox[1])],
          score=prob))
    if processed>=samples:
        break

  print("inference finished")

  with open('output/yolo.json','w') as f:
      json.dump(preds, f, indent=4)

  print("start MAP calculate...")
  #map test
  coco = COCO(anno)
  results = COCOResults('bbox')
  coco_dt = coco.loadRes('output/yolo.json')
  coco_eval = COCOeval(coco, coco_dt, 'bbox')
  coco_eval.params.imgIds = [int(Path(x).stem) for x in tested_file if x.endswith('jpg')]
  coco_eval.evaluate()
  coco_eval.accumulate()
  coco_eval.summarize()
  results.update(coco_eval)
  return results

@harness('yolo')
def harness_yolo(tree, config, args):
  bmodel = tree.expand_variables(config, args['bmodel'])
  dataset_info = config['dataset']
  imagedir = tree.expand_variables(config, dataset_info['imagedir'])
  anno = tree.expand_variables(config, dataset_info['anno'])
  devices = tree.global_config['devices']
  result = process(bmodel, devices, imagedir, anno)
  output = result.results['bbox']
  return {k: f'{v:.2%}' for k, v in output.items()}

#for test
def main():
  import argparse
  parser = argparse.ArgumentParser(description='tpu_perf topk harness')
  parser.add_argument(
    '--bmodel', type=str, help='Bmodel path')
  parser.add_argument(
    '--imagedir', type=str, help='Val image path')
  parser.add_argument(
    '--anno', type=str, help='Annotation path')
  parser.add_argument('--devices', '-d',
    type=int, nargs='*', help='Devices',
    default=[0])
  args = parser.parse_args()
  result = process(args.bmodel, args.devices, args.imagedir, args.anno)
  print(result)

if __name__ == '__main__':
   main()

