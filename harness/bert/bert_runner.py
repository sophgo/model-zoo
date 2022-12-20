import threading
import time
import logging

import numpy as np
from tqdm import trange

try:
    from .dataloder import RawResult
except ImportError:
    from dataloder import RawResult

# import by other means
# from python.tpu_perf.infer import SGInfer

from tpu_perf.infer import SGInfer

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("main")


class SGRunner(object):
    '''
        Be careful when model source alter.
        input shapes:
            tensorflow: indexes, segment, mask --> [n, 384], [n, 384], [n, 384]
            pytorch: mask, indexes, segment --> [n, 384], [n, 384], [n, 384]
        output shapes:
            tensorflow: [n, 384, 2],
            pytorch: [n, 384], [n, 384],
    '''
    def __init__(self, args):
        print("Loading bmodel...")
        self.runner = SGInfer(args.model, devices=args.devices)
        self.info = self.runner.get_input_info()
        self.out_info = self.runner.get_output_info()

        info = iter(self.info.values())
        self.batch_size = next(info)['shape'][0]
        print('Batch size: {}'.format(self.batch_size))

        print("Constructing SUT...")
        print("Finished constructing SUT.")
        self.query_count = 0
        # task id sample id pair info
        self.task_map = {}
        self.tf_or_pt = 'pt'
        self.accuracy = args.accuracy
        self.runner_result = []



    def run_one_item(self, qitem):
        # run the prediction
        task_id = self.runner.put(qitem.img)
        res = self.runner.get()
        if res[0] != task_id:
            log.error('task id dismatch {} vs {}'.format(task_id, res[0]))
        self.process_result(qitem, *res)

    def wait_result(self):
        # print('Waiting results')
        while not (self.query_count <= 0):
            task_id, values, valid = self.runner.try_get()
            if task_id == 0:
                time.sleep(0.00001)
                continue

            while task_id not in self.task_map:
                time.sleep(0.00001)

            ids = self.task_map[task_id]
            del self.task_map[task_id]

            # tf or pt
            if self.tf_or_pt == 'tf':
                outputs = values[-1]
                start_scores = outputs[:, :, 0]
                end_scores = outputs[:, :, 1]
            else:
                start_scores, end_scores = values
                start_scores = start_scores.astype(np.float32) * self.out_info['22']['scale']
                end_scores = end_scores.astype(np.float32) * self.out_info['24']['scale']
            res = []
            for i, unique_id in enumerate(ids):
                res.append(RawResult(
                    unique_id=unique_id,
                    start_logits=start_scores[i].tolist(),
                    end_logits=end_scores[i].tolist()
                ))
            self.runner_result += res
            self.query_count -= 1

    def run_one_query(self, query_samples):
        # make up batch
        if len(query_samples) % self.batch_size != 0:
            padding_sample_num = self.batch_size - (len(query_samples) % self.batch_size)
            query_samples += query_samples[-1 * padding_sample_num:]

        # process samples
        self.query_count += len(query_samples[0]) // self.batch_size
        # receiver thread focusing on get result
        receiver = threading.Thread(target=self.wait_result)
        receiver.start()

        bar = trange(0, len(query_samples[0]), self.batch_size)
        for idx in bar:
            task_id = 0
            input_ids_data = query_samples[0][idx: idx + self.batch_size]
            segment_ids_data = query_samples[1][idx: idx + self.batch_size]
            input_mask_data = query_samples[2][idx: idx + self.batch_size]
            samples_id = query_samples[3][idx: idx + self.batch_size]
            if self.tf_or_pt == 'tf':
                task_id = self.runner.put(input_ids_data,
                                          segment_ids_data,
                                          input_mask_data)
            elif self.tf_or_pt == 'pt':
                task_id = self.runner.put(input_mask_data,
                                          input_ids_data,
                                          segment_ids_data)

            self.task_map[task_id] = samples_id
            bar.set_description("Put task_id {:d} with shape = {:}".format(task_id, input_ids_data.shape), True)


def get_runner(args):
    return SGRunner(args)

