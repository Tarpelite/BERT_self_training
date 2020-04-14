# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
""" Named entity recognition fine-tuning: utilities to work with CoNLL-2003 task. """


import logging
import os
import pickle
import numpy as np

logger = logging.getLogger(__name__)


class InputExample(object):
    """A single training/test example for token classification."""

    def __init__(self, guid, text, label = None):
        """Constructs a InputExample.

        Args:
            guid: Unique id for the example.
            words: list. The words of the sequence.
            labels: (Optional) list. The labels for each word of the sequence. This should be
            specified for train and dev examples, but not for test examples.
        """
        self.guid = guid
        self.text = text
        self.label = label


class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self, input_ids, input_mask, segment_ids, label_ids, label_mask):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.label_ids = label_ids
        self.label_mask = label_mask

class DataProcessor(object):
    def get_conll_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_pkl(os.path.join(data_dir, "conll_train.pkl")), "conll_train")

    def get_conll_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_pkl(os.path.join(data_dir, "conll_test.pkl")), "conll_dev")

    def get_sep_twitter_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_pkl(os.path.join(data_dir, "sep_twitter_train.pkl")), "twitter_train")

    def get_sep_twitter_test_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_pkl(os.path.join(data_dir, "sep_twitter_test.pkl")), "twitter_test")

    def get_labels(self, data_dir):
        """See base class."""
        return ['B', 'I', 'O']

    def _create_examples(self, data, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, elem) in enumerate(data):
            guid = "%s-%s" % (set_type, i)
            text = elem[0]
            label = elem[1]
            examples.append(
                InputExample(guid=guid, text=text, label=label))
        return examples

    def _read_pkl(self, input_file):
        """Reads a tab separated value file."""
        data = pickle.load(open(input_file, 'rb'))
        return data

def convert_examples_to_features(examples, label_list, max_seq_length, tokenizer):
    """Loads a data file into a list of `InputBatch`s."""

    label_map = {label : i for i, label in enumerate(label_list)}

    features = []
    for (ex_index, example) in enumerate(examples):
        tokens = example.text

#         # Account for [CLS] and [SEP] with "- 2"
#         if len(tokens) > max_seq_length - 2:
#             tokens = tokens[:(max_seq_length - 2)]

        bert_tokens = []
        orig_to_tok_map = []

        bert_tokens.append("[CLS]")
        for token in tokens:
            new_tokens = tokenizer.tokenize(token)
            if len(bert_tokens) + len(new_tokens) > max_seq_length - 1:
                # print("You shouldn't see this since the test set is already pre-separated.")
                break
            else:
                orig_to_tok_map.append(len(bert_tokens))
                bert_tokens.extend(new_tokens)
        bert_tokens.append("[SEP]")

        if len(bert_tokens) == 2: # edge case
            continue

        # The convention in BERT is:
        # (a) For sequence pairs:
        #  tokens:   [CLS] is this jack ##son ##ville ? [SEP] no it is not . [SEP]
        #  type_ids: 0   0  0    0    0     0       0 0    1  1  1  1   1 1
        # (b) For single sequences:
        #  tokens:   [CLS] the dog is hairy . [SEP]
        #  type_ids: 0   0   0   0  0     0 0
        #
        # Where "type_ids" are used to indicate whether this is the first
        # sequence or the second sequence. The embedding vectors for `type=0` and
        # `type=1` were learned during pre-training and are added to the wordpiece
        # embedding vector (and position vector). This is not *strictly* necessary
        # since the [SEP] token unambigiously separates the sequences, but it makes
        # it easier for the model to learn the concept of sequences.
        #
        # For classification tasks, the first vector (corresponding to [CLS]) is
        # used as as the "sentence vector". Note that this only makes sense because
        # the entire model is fine-tuned.

        input_ids = tokenizer.convert_tokens_to_ids(bert_tokens)

        # The mask has 1 for real tokens and 0 for padding tokens. Only real
        # tokens are attended to.
        input_mask = [1] * len(input_ids)

        # Zero-pad up to the sequence length.
        padding = [0] * (max_seq_length - len(input_ids))
        input_ids += padding
        input_mask += padding

        assert len(input_ids) == max_seq_length
        assert len(input_mask) == max_seq_length

        segment_ids = [0] * max_seq_length # no use for our problem

        labels = example.label
        label_ids = [0] * max_seq_length
        label_mask = [0] * max_seq_length

        for label, target_index in zip(labels, orig_to_tok_map):
            label_ids[target_index] = label_map[label]
            label_mask[target_index] = 1

        assert len(segment_ids) == max_seq_length
        assert len(label_ids) == max_seq_length
        assert len(label_mask) == max_seq_length

        features.append(
                InputFeatures(input_ids=input_ids,
                              input_mask=input_mask,
                              segment_ids=segment_ids,
                              label_ids=label_ids,
                              label_mask=label_mask))
    return features

def accuracy(out, label_ids, label_mask):
    # axis-0: seqs in batch; axis-1: toks in seq; axis-2: potential labels of tok
    outputs = np.argmax(out, axis=2)
    matched = outputs == label_ids
    num_correct = np.sum(matched * label_mask)
    num_total = np.sum(label_mask)
    return num_correct, num_total


def true_and_pred(out, label_ids, label_mask):
    # axis-0: seqs in batch; axis-1: toks in seq; axis-2: potential labels of tok
    tplist = []
    outputs = np.argmax(out, axis=2)
    for i in range(len(label_ids)):
        trues = []
        preds = []
        for true, pred, mask in zip(label_ids[i], outputs[i], label_mask[i]):
            if mask:
                trues.append(true)
                preds.append(pred)
        tplist.append((trues, preds))
    return tplist

def compute_tfpn(_trues, _preds, label_map):
    trues = _trues + [label_map['O']]
    preds = _preds + [label_map['O']]

    true_ent_list = []
    pred_ent_list = []

    ent_start = -1
    for i, label in enumerate(trues):
        if ent_start == -1:
            if label == label_map['B']:
                ent_start = i
            elif label == label_map['I']:
                assert 0 == 1 # should not occur in ground truth
        else:
            if label == label_map['B']:
                true_ent_list.append((ent_start, i))
                ent_start = i
            elif label == label_map['O']:
                true_ent_list.append((ent_start, i))
                ent_start = -1

    ent_start = -1
    for i, label in enumerate(preds):
        if ent_start == -1:
            if label == label_map['B']:
                ent_start = i
            elif label == label_map['I']:
                ent_start = i # entities in prediction might start with "I"
        else:
            if label == label_map['B']:
                pred_ent_list.append((ent_start, i))
                ent_start = i
            elif label == label_map['O']:
                pred_ent_list.append((ent_start, i))
                ent_start = -1

    TP, FP, FN, _TP = 0, 0, 0, 0
    for ent in true_ent_list:
        if ent in pred_ent_list:
            TP += 1
        else:
            FN += 1
    for ent in pred_ent_list:
        if ent in true_ent_list:
            _TP += 1
        else:
            FP += 1
    assert TP == _TP
    return TP, FP, FN

def compute_f1(TP, FP, FN):
    P = TP / (TP + FP)
    R = TP / (TP + FN)
    F1 = 2 * P * R / (P + R)
    return "Precision: " + str(P) + ", Recall: " + str(R) + ", F1: " + str(F1)


