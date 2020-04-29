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
    def __init__(self, guid, title_a, text_a, title_b, text_b, label=None):
        self.guid = guid
        self.title_a = title_a
        self.text_a = text_a
        self.title_b = title_b
        self.text_b = text_b
        self.label = label

class InputFeatures(object):
    def __init__(self, input_ids_a, input_ids_b, input_mask_a, input_mask_b, segment_ids_a, segment_ids_b, label_ids):
        self.input_ids_a = input_ids_a
        self.input_ids_b = input_ids_b
        self.input_mask_a = input_mask_a
        self.input_mask_b = input_mask_b
        self.segment_ids_a = segment_ids_a
        self.segment_ids_b = segment_ids_b
        self.label_ids = label_ids
    

class DataProcessor(object):
    '''Processor for the DQD dataset '''

    def get_askubuntu_train_examples(self, data_dir):
        cached_path = os.path.join(data_dir, "askubuntu_train.cache")
        if os.path.exists(cached_path):
            logger.info("read examples from cache")
            return self._read_pkl(cached_path)
        else:
            examples = self._create_examples(
        self._read_pkl(os.path.join(data_dir, "askubuntu_train.pkl")), "askubuntu_train")
            with open(cached_path, "wb") as f:
                pickle.dump(examples, f, protocol=4)
        return examples
    
    def get_askubuntu_dev_examples(self, data_dir):
        cached_path = os.path.join(data_dir, "askubuntu_dev.cache")
        if os.path.exists(cached_path):
            return self._read_pkl(cached_path)
        else:
            examples = self._create_examples(
        self._read_pkl(os.path.join(data_dir, "askubuntu_dev.pkl")), "askubuntu_dev")
            with open(cached_path, "wb") as f:
                pickle.dump(examples, f, protocol=4)
        return examples

    def get_askubuntu_test_examples(self, data_dir):
        cached_path = os.path.join(data_dir, "askubuntu_test.cache")
        if os.path.exists(cached_path):
            return self._read_pkl(cached_path)
        else:
            examples = self._create_examples(
        self._read_pkl(os.path.join(data_dir, "askubuntu_test.pkl")), "askubuntu_test")
            with open(cached_path, "wb") as f:
                pickle.dump(examples, f, protocol=4)
        return examples

    def get_superuser_train_examples(self, data_dir):
        cached_path = os.path.join(data_dir, "superuser_train.cache")
        if os.path.exists(cached_path):
            return self._read_pkl(cached_path)
        else:
            examples = self._create_examples(
        self._read_pkl(os.path.join(data_dir, "superuser_train.pkl")), "superuser_train")
            with open(cached_path, "wb") as f:
                pickle.dump(examples, f, protocol=4)
        return examples

    def get_superuser_dev_examples(self, data_dir):
        cached_path = os.path.join(data_dir, "superuser_dev.cache")
        if os.path.exists(cached_path):
            return self._read_pkl(cached_path)
        else:
            examples = self._create_examples(
        self._read_pkl(os.path.join(data_dir, "superuser_dev.pkl")), "superuser_dev")
            with open(cached_path, "wb") as f:
                pickle.dump(examples, f, protocol=4)
        return examples
    
    def get_superuser_test_examples(self, data_dir):
        cached_path = os.path.join(data_dir, "superuser_test.cache")
        if os.path.exists(cached_path):
            return self._read_pkl(cached_path)
        else:
            examples = self._create_examples(
        self._read_pkl(os.path.join(data_dir, "superuser_test.pkl")), "superuser_test")
            with open(cached_path, "wb") as f:
                pickle.dump(examples, f, protocol=4)
        return examples

    def get_labels(self, data_dir):

        return ["0", "1"]
    
    def _create_examples(self, data, set_type):
        examples = []
        for (i, elem) in enumerate(data):
            guid = "%s-%s" % (set_type, i)
            title_a = elem[0]
            text_a = elem[1]
            title_b = elem[2]
            text_b = elem[3]
            label = elem[4]
            examples.append(
                InputExample(
                    guid = guid,
                    title_a = title_a,
                    text_a = text_a,
                    title_b = title_b,
                    text_b = text_b,
                    label = label
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
    for (ex_index, example) in enumerate(tqdm(examples)):
        if ex_index % 10000 == 0:
            logger.info("Writing example %d of %d", ex_index, len(examples))
        
        title_tokens_a = tokenizer.tokenize(example.title_a)
        text_tokens_a = tokenizer.tokenize(example.text_a)

        title_tokens_b = tokenizer.tokenize(example.title_b)
        text_tokens_b = tokenizer.tokenize(example.text_b)

        label_id = label_map[example.label]


        tokens_a = [cls_token] + title_tokens_a + [sep_token]
        if len(text_tokens_a) > max_seq_length - len(tokens_a) - 1:
            tokens_a += text_tokens_a[:(max_seq_length - len(tokens_a) -1)] + [sep_token]
        else:
            tokens_a += text_tokens_a + [sep_token]
        
        tokens_b = [cls_token] + title_tokens_b + [sep_token]
        if len(text_tokens_b) > max_seq_length - len(tokens_b) - 1:
            tokens_b += text_tokens_b[:(max_seq_length - len(tokens_b) -1)] + [sep_token]
        else:
            tokens_b += text_tokens_b + [sep_token]
        
        segment_ids_a = [cls_token_segment_id] + [sequence_a_segment_id] * (len(tokens_a) - 1)
        segment_ids_b = [cls_token_segment_id] + [sequence_a_segment_id] * (len(tokens_b) - 1)

        input_ids_a = tokenizer.convert_tokens_to_ids(tokens_a)
        input_ids_b = tokenizer.convert_tokens_to_ids(tokens_b)

        input_mask_a = [1 if mask_padding_with_zero else 0] * len(input_ids_a)
        input_mask_b = [1 if mask_padding_with_zero else 0] * len(input_ids_b)

        padding_length = max_seq_length - len(input_ids_a)
        input_ids_a  += [pad_token] * padding_length
        input_mask_a += [0 if mask_padding_with_zero else 1] * padding_length
        segment_ids_a += [pad_token_segment_id] * padding_length

        padding_length = max_seq_length - len(input_ids_b)
        input_ids_b += [pad_token] * padding_length
        input_mask_b  += [0 if mask_padding_with_zero else 1]* padding_length
        segment_ids_b += [pad_token_segment_id] * padding_length

        assert len(input_ids_a) == max_seq_length
        assert len(input_mask_a) == max_seq_length
        assert len(segment_ids_a) == max_seq_length
        assert len(input_ids_b) == max_seq_length
        assert len(input_mask_b) == max_seq_length
        assert len(segment_ids_b) == max_seq_length

        if ex_index < 5 :
            logger.info("*** Example ***")
            logger.info("guid: %s", example.guid)
            logger.info("tokens_a: %s", " ".join([str(x) for x in tokens_a]))
            logger.info("input_ids_a: %s", " ".join([str(x) for x in input_ids_a]))
            logger.info("input_mask_a: %s", " ".join([str(x) for x in input_mask_a]))
            logger.info("segment_ids_a: %s", " ".join([str(x) for x in segment_ids_a]))

            logger.info("tokens_b: %s", " ".join([str(x) for x in tokens_b]))
            logger.info("input_ids_b: %s", " ".join([str(x) for x in input_ids_b]))
            logger.info("input_mask_b: %s", " ".join([str(x) for x in input_mask_b]))
            logger.info("segment_ids_b: %s", " ".join([str(x) for x in segment_ids_b]))


            logger.info("label_ids: %s", label_id)
        
        features.append(
            InputFeatures(
                input_ids_a = input_ids_a,
                input_ids_b = input_ids_b,
                input_mask_a = input_mask_a,
                input_mask_b = input_mask_b,
                segment_ids_a = segment_ids_a,
                segment_ids_b = segment_ids_b,
                label_ids=label_id)
        )
    return features
        

        

        




        
        
        


    

        
