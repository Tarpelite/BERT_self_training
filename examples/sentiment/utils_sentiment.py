import logging
import os
import pickle
import numpy as np
from tqdm import *
import math
import numbers
import numpy as np
import torch


logger = logging.getLogger(__name__)


class Meter(object):
    def reset(self):
        pass
    def add(self):
        pass
    def value(self):
        pass

class AUCMeter(Meter):
    def __init__(self):
        super(AUCMeter, self).__init__()
        self.reset()
    
    def reset(self):
        self.scores = torch.DoubleTensor(torch.DoubleStorage()).numpy()
        self.targets = torch.LongTensor(torch.LongStorage()).numpy()

    def add(self, output, target):
        if torch.is_tensor(output):
            output = output.cpu().squeeze().numpy()
        if torch.is_tensor(target):
            target = target.cpu().squeeze().numpy()
        elif isinstance(target, numbers.Number):
            target = np.asarray([target])
        assert np.ndim(output) == 1, \
            'wrong output size (1D expected)'
        assert np.ndim(target) == 1, \
            'wrong target size (1D expected)'
        assert output.shape[0] == target.shape[0], \
            'number of outputs and targets does not match'
        assert np.all(np.add(np.equal(target, 1), np.equal(target, 0))), \
            'targets should be binary (0, 1)'

        self.scores = np.append(self.scores, output)
        self.targets = np.append(self.targets, target)
        self.sortind = None


    def value(self, max_fpr=1.0):
        assert max_fpr > 0

        # case when number of elements added are 0
        if self.scores.shape[0] == 0:
            return 0.5

        # sorting the arrays
        if self.sortind is None:
            scores, sortind = torch.sort(torch.from_numpy(self.scores), dim=0, descending=True)
            scores = scores.numpy()
            self.sortind = sortind.numpy()
        else:
            scores, sortind = self.scores, self.sortind

        # creating the roc curve
        tpr = np.zeros(shape=(scores.size + 1), dtype=np.float64)
        fpr = np.zeros(shape=(scores.size + 1), dtype=np.float64)

        for i in range(1, scores.size + 1):
            if self.targets[sortind[i - 1]] == 1:
                tpr[i] = tpr[i - 1] + 1
                fpr[i] = fpr[i - 1]
            else:
                tpr[i] = tpr[i - 1]
                fpr[i] = fpr[i - 1] + 1

        tpr /= (self.targets.sum() * 1.0)
        fpr /= ((self.targets - 1.0).sum() * -1.0)

        for n in range(1, scores.size + 1):
            if fpr[n] >= max_fpr:
                break

        # calculating area under curve using trapezoidal rule
        #n = tpr.shape[0]
        h = fpr[1:n] - fpr[0:n - 1]
        sum_h = np.zeros(fpr.shape)
        sum_h[0:n - 1] = h
        sum_h[1:n] += h
        area = (sum_h * tpr).sum() / 2.0

        return area / max_fpr

class InputExample(object):
    def __init__(self, guid, text, label=None):
        self.guid = guid
        self.text = text
        self.label = label

class InputFeatures(object):
    def __init__(self, input_ids, input_mask, segment_ids, label_ids):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.label_ids = label_ids 
    

class DataProcessor(object):
    '''Processor for the DQD dataset '''

    def get_train_examples(self, data_dir, task):
        file = os.path.join(data_dir, task, "train.pkl")
        examples = self._create_examples(
        self._read_pkl(file), "{}-train".format(task))
        return examples
    
    def get_dev_examples(self, data_dir, task):
        file = os.path.join(data_dir, task, "dev.pkl")
        examples = self._create_examples(
        self._read_pkl(file), "{}-dev".format(task))
        return examples

    def get_labels(self, data_dir):

        return ["1", "2", "3", "4", "5"]
    
    def _create_examples(self, data, set_type):
        examples = []
        for (i, elem) in enumerate(data):
            guid = "%s-%s" % (set_type, i)
            text = elem[0]
            label = str(elem[1])
            examples.append(
                InputExample(
                    guid = guid,
                    text = text, 
                    label= label
                )
            )
        return examples
    
    def _read_pkl(self, input_file):
        data = pickle.load(open(input_file, "rb"))
        return data

def convert_examples_to_features(
    examples,
    label_list,
    max_seq_length,
    tokenizer,
    cls_token_at_end=False,
    cls_token="[CLS]",
    cls_token_segment_id=1,
    sep_token="[SEP]",
    sep_token_extra=False,
    pad_on_left=False,
    pad_token=0,
    pad_token_segment_id=0,
    pad_token_label_id=-100,
    sequence_a_segment_id=0,
    mask_padding_with_zero=True,
):
    label_map = {label:i for i,label in enumerate(label_list)}

    features = []
    all_lens = []
    for (ex_index, example) in enumerate(tqdm(examples)):
        if ex_index % 10000 == 0:
            logger.info("Writing example %d of %d", ex_index, len(examples))
        
        tokens = tokenizer.tokenize(example.text)
        all_lens.append(len(tokens))
        label_id = label_map[example.label]


        if len(tokens) > max_seq_length  - 2:
            tokens = tokens[:(max_seq_length -2)]
        tokens = [cls_token] + tokens + [sep_token]

        
        segment_ids = [cls_token_segment_id] + [sequence_a_segment_id] * (len(tokens) - 1)

        input_ids = tokenizer.convert_tokens_to_ids(tokens)

        input_mask = [1 if mask_padding_with_zero else 0] * len(input_ids)

        padding_length = max_seq_length - len(input_ids)
        input_ids  += [pad_token] * padding_length
        input_mask += [0 if mask_padding_with_zero else 1] * padding_length
        segment_ids += [pad_token_segment_id] * padding_length


        assert len(input_ids) == max_seq_length
        assert len(input_mask) == max_seq_length
        assert len(segment_ids) == max_seq_length
       

        if ex_index < 5 :
            logger.info("*** Example ***")
            logger.info("guid: %s", example.guid)
            logger.info("tokens: %s", " ".join([str(x) for x in tokens]))
            logger.info("input_ids: %s", " ".join([str(x) for x in input_ids]))
            logger.info("input_mask: %s", " ".join([str(x) for x in input_mask]))
            logger.info("segment_ids: %s", " ".join([str(x) for x in segment_ids]))

            logger.info("label_ids: %s", label_id)
        
        features.append(
            InputFeatures(
                input_ids = input_ids,
                input_mask = input_mask,
                segment_ids = segment_ids,
                label_ids=label_id)
        )
    logger.info("max_len:{}  min_len:{} avg_len:{}".format(max(all_lens), min(all_lens), sum(all_lens)/len(all_lens)))
    return features
        

        

        




        
        
        


    

        
