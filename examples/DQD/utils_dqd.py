import logging
import os
import pickle
import numpy as np
from tqdm import *

logger = logging.getLogger(__name__)

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
        return self._create_examples(
        self._read_pkl(os.path.join(data_dir, "askubuntu_train.pkl")), "askubuntu_train")
    
    def get_askubuntu_dev_examples(self, data_dir):
        return self._create_examples( 
        self._read_pkl(os.path.join(data_dir, "askubuntu_dev.pkl")),
        "ask_ubuntu_dev")
    
    def get_askubuntu_test_examples(self, data_dir):
        return self._create_examples(
        self._read_pkl(os.path.join(data_dir, "askubuntu_test.pkl")),
        "ask_ubuntu_test"
        )
    
    def get_superuser_train_examples(self, data_dir):
        return self._create_examples(
            self._read_pkl(os.path.join(data_dir, "superuser_train.pkl")),
            "apple_train"
        )
    
    def get_superuser_dev_examples(self, data_dir):
        return self._create_examples(
            self._read_pkl(os.path.join(data_dir, "superuser_dev.pkl")),
            "apple_dev"
        )
    
    def get_superuser_test_examples(self, data_dir):
        return self._create_examples(
            self._read_pkl(os.path.join(data_dir, "superuser_test.pkl")),
            "apple_test"
        )
    
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
        
        tokens_b = [cls_token] + title_tokens_b + [sep_token]
        if len(text_tokens_b) > max_seq_length - len(tokens_b) - 1:
            tokens_b += text_tokens_b[:(max_seq_length - len(tokens_b) -1)] + [sep_token]
        
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
        

        

        




        
        
        


    

        
