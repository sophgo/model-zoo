import numpy as np
import collections
import os
import pickle
from transformers import BertTokenizer
import random

try:
    from .create_squad_data import read_squad_examples, convert_examples_to_features
except ImportError:
    from create_squad_data import read_squad_examples, convert_examples_to_features

max_seq_length = 384
max_query_length = 64
doc_stride = 128

RawResult = collections.namedtuple("RawResult", ["unique_id", "start_logits", "end_logits"])

class SQuAD_v1_loader():
    '''
        Args:
        load_fn :
            Called by dataloader in start_test()
    '''
    def __init__(self, count_override=None,
                 cache_path='eval_features.pickle',
                 input_file='',
                 load_fn=None):
        print("Constructing SQuAD v1 loader...")
        eval_features = []
        # Load features if cached, convert from examples otherwise.
        if os.path.exists(cache_path):
            print("Loading cached features from '%s'..." % cache_path)
            with open(cache_path, 'rb') as cache_file:
                eval_features = pickle.load(cache_file)
        else:
            print("No cached features at '%s'... converting from examples..." % cache_path)

            print("Creating tokenizer...")
            vocab_file = os.path.join(os.path.dirname(__file__), "./vocab.txt")
            tokenizer = BertTokenizer(vocab_file)

            print("Reading examples...")
            eval_examples = read_squad_examples(input_file=input_file,
                                                is_training=False,
                                                version_2_with_negative=False)

            print("Converting examples to features, will take a long time... ")

            def append_feature(feature):
                eval_features.append(feature)

            convert_examples_to_features(
                examples=eval_examples,
                tokenizer=tokenizer,
                max_seq_length=max_seq_length,
                doc_stride=doc_stride,
                max_query_length=max_query_length,
                is_training=False,
                output_fn=append_feature,
                verbose_logging=False)

            print("Caching features at '%s'..." % cache_path)
            with open(cache_path, 'wb') as cache_file:
                pickle.dump(eval_features, cache_file)

        self.eval_features = eval_features
        self.count = count_override or len(self.eval_features)
        self.idx = [i for i in range(self.count)]
        self.load_fn = load_fn
        print("Finished constructing SQuAD loader.")

    def __len__(self):
        return self.count

    def __getitem__(self, item):
        # no need to scale
        if isinstance(item, slice):
            input_ids = np.array([d.input_ids for d in self.eval_features[item]],
                                      dtype=np.int32)
            segment_ids = np.array([d.segment_ids for d in self.eval_features[item]],
                                        dtype=np.int32)
            input_mask = np.array([d.input_mask for d in self.eval_features[item]],
                                       dtype=np.int32)
            idx = [d.unique_id for d in self.eval_features[item]]
        else:
            input_ids = np.array(self.eval_features[item].input_ids,
                                      dtype=np.int32)
            segment_ids = np.array(self.eval_features[item].segment_ids,
                                        dtype=np.int32)
            input_mask = np.array(self.eval_features[item].input_mask,
                                       dtype=np.int32)
            idx = [self.eval_features[item].unique_id]

        return np.ascontiguousarray(input_ids), \
               np.ascontiguousarray(segment_ids),\
               np.ascontiguousarray(input_mask), \
               idx



    def __iter__(self):
        pass

    def __next__(self):
        pass

    def load_gen(self, config):
        '''
        load generator in terms of accuracy or throughput benchmarks
        '''
        if config['accuracy']:
            count = min(self.count, len(self.eval_features))
            queries = self[0:count]
        else:
            count = self.count
            list_q = random.choices(self, k=count)
            queries = (np.concatenate([q[0].reshape(1, -1) for q in list_q]),
                       np.concatenate([q[1].reshape(1, -1) for q in list_q]),
                       np.concatenate([q[2].reshape(1, -1) for q in list_q]),
                       [q[3][0] for q in list_q])

        return queries

    def start_test(self, config):
        queries = self.load_gen(config)
        self.load_fn(queries)




def get_dataloader(count_override, input_file, cache_path, load_fn):
    return SQuAD_v1_loader(count_override,
                           cache_path,
                           input_file,
                           load_fn)
