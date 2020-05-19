from __future__ import absolute_import, division, print_function

import argparse
import csv
import logging
import os
import random
import sys
import pickle

os.environ['CUDA_VISIBLE_DEVICES'] = "0"

import numpy as np
import torch
from torch import nn
from torch.nn import CrossEntropyLoss
from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler,
                              TensorDataset)
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm, trange

from pytorch_pretrained_bert.file_utils import PYTORCH_PRETRAINED_BERT_CACHE
from pytorch_pretrained_bert.modeling import BertPreTrainedModel, BertModel, BertConfig, WEIGHTS_NAME, CONFIG_NAME
from pytorch_pretrained_bert.tokenization import BertTokenizer
from pytorch_pretrained_bert.optimization import BertAdam, warmup_linear

from transformers import AdamW, get_linear_schedule_with_warmup
from torch.nn.functional import softmax

try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:
    from tensorboardX import SummaryWriter

logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt = '%m/%d/%Y %H:%M:%S',
                    level = logging.INFO)
logger = logging.getLogger(__name__)

def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)


class MyBertForTokenClassification(BertPreTrainedModel):
    def __init__(self, config, num_labels):
        super(MyBertForTokenClassification, self).__init__(config)
        self.num_labels = num_labels
        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, num_labels)
        self.apply(self.init_bert_weights)

    def forward(self, input_ids, token_type_ids, attention_mask, labels=None, label_mask=None):
        sequence_output, _ = self.bert(input_ids, token_type_ids, attention_mask, output_all_encoded_layers=False)
        sequence_output = self.dropout(sequence_output)
        logits = self.classifier(sequence_output)

        if labels is not None:
            loss_fct = CrossEntropyLoss()
            # Only keep active parts of the loss
            active_loss = label_mask.view(-1) == 1
            active_logits = logits.view(-1, self.num_labels)[active_loss]
            active_labels = labels.view(-1)[active_loss]
            loss = loss_fct(active_logits, active_labels)
            return loss
        else:
            return logits


class InputExample(object):
    """A single training/test example for simple sequence classification."""

    def __init__(self, guid, text, label=None):
        """Constructs a InputExample.
        Args:
            guid: Unique id for the example.
            text_a: string. The untokenized text of the first sequence. For single
            sequence tasks, only this sequence must be specified.
            text_b: (Optional) string. The untokenized text of the second sequence.
            Only must be specified for sequence pair tasks.
            label: (Optional) string. The label of the example. This should be
            specified for train and dev examples, but not for test examples.
        """
        self.guid = guid
        self.text = text # list of tokens
        self.label = label # list of labels


class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self, input_ids, input_mask, segment_ids, label_ids, label_mask):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.label_ids = label_ids
        self.label_mask = label_mask # necessary since the label mismatch for wordpieces


class DataProcessor(object):
    """Processor for the MRPC data set (GLUE version)."""

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

def joint_tri_train(args, model_f1, model_f2, model_ft, source_features, target_features):

    Nt = args.N_init
    features_L = source_features
    all_input_ids = torch.tensor([f.input_ids for f in features_L], dtype=torch.long)
    all_input_mask = torch.tensor([f.input_mask for f in features_L], dtype=torch.long)
    all_segment_ids = torch.tensor([f.segment_ids for f in features_L], dtype=torch.long)
    all_label_ids = torch.tensor([f.label_ids for f in features_L], dtype=torch.long)
    all_label_mask = torch.tensor([f.label_mask for f in features_L], dtype=torch.long)

    dataset_L = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids, all_label_mask)

    model_f1, model_f2 = train_f1_f2(args, model_f1, model_f2, dataset_L)

    labeled_features = labelling(args, target_features, model_f1, model_f2, Nt)

    dataset_L, dataset_TL = prepare_dataset(source_features, labeled_features)

    model_ft = train_ft(args, model_ft, dataset_TL)

    return model_f1, model_f2, model_ft

def train_f1_f2(args, model_f1, model_f2, train_dataset):
    if args.local_rank in [-1, 0]:
        tb_writer = SummaryWriter()
    
    args.train_batch_size = args.mini_batch_size * max(1, args.n_gpu)
    train_sampler = RandomSampler(train_dataset) if args.local_rank == -1 else DistributedSampler(train_dataset)
    train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=args.train_batch_size)

    args.num_train_epochs = 1
    t_total = len(train_dataloader) // args.gradient_accumulation_steps * args.num_train_epochs

    if args.warmup_proportion > 0:
        args.warmup_steps = int(t_total * args.warmup_proportion)

    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in list(model_f1.named_parameters()) + list(model_f2.named_parameters()) if not any(nd in n for nd in no_decay)],
            "weight_decay": args.weight_decay,
        },
        {"params": [p for n, p in list(model_f1.named_parameters()) + list(model_f2.named_parameters()) if any(nd in n for nd in no_decay)], "weight_decay": 0.0},
    ] 

    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=t_total
    )


    if args.fp16:
        try:
            from apex import amp
        except ImportError:
            raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")
        [model_f1, model_f2], optimizer = amp.initialize([model_f1, model_f2], optimizer, opt_level=args.fp16_opt_level)
    
    # multi-gpu training (should be after apex fp16 initialization)
    if args.n_gpu > 1:
        model_f1 = torch.nn.DataParallel(model_f1)
        model_f2 = torch.nn.DataParallel(model_f2)

    
    # Distributed training (should be after apex fp16 initialization)
    if args.local_rank != -1:
        model_f1 = torch.nn.parallel.DistributedDataParallel(
            model_f1, device_ids=[args.local_rank], output_device=args.local_rank, find_unused_parameters=True
        )

        model_f2 = torch.nn.parallel.DistributedDataParallel(
            model_f2, device_ids=[args.local_rank], output_device=args.local_rank, find_unused_parameters=True
        )
    
 

    global_step = 0
    epochs_trained = 0
    steps_trained_in_current_epoch = 0
   
    tr_loss, logging_loss = 0.0, 0.0
    model_f1.zero_grad()
    model_f2.zero_grad()


    set_seed(args)
    logger.info("***** train f1 f2 ******")
    logger.info("***** Num examples: {} ********".format(len(train_dataset)))

    for _ in range(1):
        epoch_iterator = tqdm(train_dataloader, desc="Iter(loss=X.XXX, lr=X.XXXXXXXX)", disable=args.local_rank not in [-1, 0])

        for step, batch in enumerate(epoch_iterator):
            if steps_trained_in_current_epoch > 0:
                steps_trained_in_current_epoch -= 1
                continue
                
            model_f1.train()
            model_f2.train()
            batch = tuple(t.to(args.device) for t in batch)

            inputs = {
                "input_ids":batch[0],
                "attention_mask":batch[1],
                "token_type_ids":batch[2],
                "labels":batch[3],
                "label_mask":batch[4]
            }

            outputs1 = model_f1(**inputs)
            loss1 = outputs1[0]

            
            outputs2 = model_f2(**inputs)
            loss2 = outputs2[0]

            w1 = model_f1.classifier.weight #[hidden_size, num_labels]
            w2 = model_f2.classifier.weight.transpose(-1, -2) #[num_labels, hidden_size]

            norm_term = torch.norm(torch.matmul(w1, w2))

            loss = loss1 + loss2 + args.alpha * norm_term

            if args.n_gpu > 1:
                loss = loss.mean()
            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps

            if args.fp16:
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                loss.backward()

            tr_loss += loss.item()
            if (step + 1) % args.gradient_accumulation_steps == 0:
                epoch_iterator.set_description('Iter (loss=%5.3f) lr=%9.7f' % (loss.item(), scheduler.get_lr()[0]))
                if args.fp16:
                    torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), args.max_grad_norm)
                else:
                    torch.nn.utils.clip_grad_norm_(model_f1.parameters(), args.max_grad_norm)
                    torch.nn.utils.clip_grad_norm_(model_f2.parameters(), args.max_grad_norm)
                    

                optimizer.step()
                scheduler.step()  # Update learning rate schedule
                model_f1.zero_grad()
                model_f2.zero_grad()
                global_step += 1

                if args.local_rank in [-1, 0] and args.logging_steps > 0 and global_step % args.logging_steps == 0:
                    # Log metrics
                  
                    tb_writer.add_scalar("f1_f2_lr", scheduler.get_lr()[0], global_step)
                    tb_writer.add_scalar("f1_f2_loss", (tr_loss - logging_loss) / args.logging_steps, global_step)
                    logging_loss = tr_loss

                    

    if args.local_rank in [-1, 0]:
        tb_writer.close()

    return model_f1, model_f2

def train_ft(args, model_ft, train_dataset):
    if args.local_rank in [-1, 0]:
        tb_writer = SummaryWriter()
    
    args.train_batch_size = args.per_gpu_train_batch_size * max(1, args.n_gpu)

    train_sampler = RandomSampler(train_dataset) if args.local_rank == -1 else DistributedSampler(train_dataset)
    train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=args.train_batch_size)

    args.num_train_epochs = 1
    t_total = len(train_dataloader) // args.gradient_accumulation_steps * args.num_train_epochs
    
    if args.warmup_ratio > 0:
        args.warmup_steps = int(t_total * args.warmup_ratio)
    
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in list(model_ft.named_parameters()) if not any(nd in n for nd in no_decay)],
            "weight_decay": args.weight_decay,
        },
        {"params": [p for n, p in list(model_ft.named_parameters())  if any(nd in n for nd in no_decay)], "weight_decay": 0.0},
    ] 

    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=t_total
    )

    
    
    if args.fp16:
        try:
            from apex import amp
        except ImportError:
            raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")
        model_ft, optimizer = amp.initialize(model_ft, optimizer, opt_level=args.fp16_opt_level)
    
    # multi-gpu training (should be after apex fp16 initialization)
    if args.n_gpu > 1:
        model_ft = torch.nn.DataParallel(model_ft)
    
    # Distributed training (should be after apex fp16 initialization)
    if args.local_rank != -1:
        model_ft = torch.nn.parallel.DistributedDataParallel(
            model_ft, device_ids=[args.local_rank], output_device=args.local_rank, find_unused_parameters=True
        )
    
    global_step = 0
    epochs_trained = 0
    steps_trained_in_current_epoch = 0
    tr_loss, logging_loss = 0.0, 0.0

    model_ft.zero_grad()
    train_iterator = trange(
        epochs_trained, int(args.num_train_epochs), desc="Epoch", disable=args.local_rank not in [-1, 0]
    )

    set_seed(args)
    logger.info("******* train ft *************")
    for _ in range(1):
        epoch_iterator = tqdm(train_dataloader, desc="Iter(loss=X.XXX, lr=X.XXXXXXXX)", disable=args.local_rank not in [-1, 0])

        for step, batch in enumerate(epoch_iterator):
            model_ft.train()
            batch = tuple(t.to(args.device) for t in batch)

            inputs = {
                "input_ids":batch[0],
                "attention_mask":batch[1],
                "token_type_ids":batch[2],
                "labels":batch[3],
                "label_mask":batch[4],
            }

            outputs = model_ft(**inputs)
            loss = outputs[0]

            if args.n_gpu > 1:
                loss = loss.mean()  # mean() to average on multi-gpu 
            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps

            if args.fp16:
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                loss.backward()
            
            tr_loss += loss.item()
            if (step + 1) % args.gradient_accumulation_steps == 0:
                epoch_iterator.set_description('Iter (loss=%5.3f) lr=%9.7f' % (loss.item(), scheduler.get_lr()[0]))
                if args.fp16:
                    torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), args.max_grad_norm)
                else:
                    torch.nn.utils.clip_grad_norm_(model_ft.parameters(), args.max_grad_norm)

                optimizer.step()
                scheduler.step()  # Update learning rate schedule
                model_ft.zero_grad()
                global_step += 1
   
                
    if args.local_rank in [-1, 0]:
        tb_writer.close()

    return model_ft

def labelling(args, all_target_data, model_f1, model_f2, N_init):

    # make target_data_loader
    np.random.shuffle(all_target_data)
    cand_data = all_target_data[:N_init]
    all_input_ids = torch.tensor([x.input_ids for x in cand_data], dtype=torch.long)
    all_input_mask = torch.tensor([x.input_mask for x in cand_data], dtype=torch.long)
    all_segment_ids = torch.tensor([x.segment_ids for x in cand_data], dtype=torch.long)
    all_label_ids = torch.tensor([x.label_ids for x in cand_data], dtype=torch.long)
    all_label_mask = torch.tensor([x.label_mask for x in cand_data], dtype=torch.long)


    dataset = TensorDataset(all_input_ids,
    all_input_mask, all_segment_ids, all_label_ids, all_label_mask)


    eval_sampler = SequentialSampler(dataset) if args.local_rank == -1 else DistributedSampler(dataset)
    eval_dataloader = DataLoader(dataset, sampler=eval_sampler, batch_size=args.mini_batch_size)

    if args.n_gpu > 1:
        model = torch.nn.DataParallel(model)
    
    labeled_data = []
    logger.info("***** Running Labelling*****")

    logger.info("  Num examples = %d", len(dataset))
    logger.info("  Batch size = %d", args.mini_batch_size)
    eval_loss = 0.0
    nb_eval_steps = 0
    preds = None
    out_label_ids = None
    model_f1.eval()
    model_f2.eval()
    all_logits1 = []
    all_logits2 = []
    choose_index = []
    for batch in tqdm(eval_dataloader, desc="Evaluating"):
        batch = tuple(t.to(args.device) for t in batch)

        with torch.no_grad():
            inputs = {
                "input_ids": batch[0], 
                "attention_mask": batch[1],
                "token_type_ids":batch[2], 
                "labels": batch[3],
                "label_mask":batch[4]
            }

            outputs1 = model_f1(**inputs)
            outputs2 = model_f2(**inputs)

            logits1 = outputs1[1] # [batch_size, seq_len, num_labels]
            logits2 = outputs2[1]

            logits1 = softmax(logits1, dim=2)
            logits2 = softmax(logits2, dim=2)
        if len(all_logits1) == 0:
            all_logits1 = logits1.detach().cpu().numpy()
            all_logits2 = logits2.detach().cpu().numpy()
        else:
            all_logits1 = np.append(all_logits1, logits1.detach().cpu().numpy(), axis=0)
            all_logits2 = np.append(all_logits2, logits2.detach().cpu().numpy(), axis=0)

    
    # do collect
    all_preds_max_1 = np.max(all_logits1, axis=2)
    # [batch_size, seq_len, 1]
    all_preds_max_2 = np.max(all_logits2, axis=2)
    # [batch_size, seq_len, 1]

    all_labels_1 = np.argmax(all_logits1, axis=2)
    all_labels_2 = np.argmax(all_logits2, axis=2)
    

    assert len(dataset) == len(all_preds_max_1) == len(all_preds_max_2) == len(all_labels_1) == len(all_labels_2)

    labeled_data = []
    for i in range(len(dataset)):
        record = cand_data[i]

        max_1 = all_preds_max_1[i]
        max_2 = all_preds_max_2[i]

        labels_1 = all_labels_1[i]
        labels_2 = all_labels_2[i]
        
        flag = True 
        for j in range(len(max_1)):
            if labels_1[j] != labels_2[j]:
                flag = False
                break
            elif max(max_1[j], max_2[j]) < args.threshold:
                flag = False
                break
        if not flag:
            continue
        
        assert len(labels_1) == len(cand_data[i].input_ids)
        cand_data[i].label_ids = labels_1
        labeled_data.append(cand_data[i])
    logger.info("**** collect labeled data size %s", len(labeled_data))

    return labeled_data
def prepare_dataset(source_features, labeled_features):

    features_L = source_features + labeled_features
    
    all_input_ids = torch.tensor([f.input_ids for f in features_L], dtype=torch.long)
    all_input_mask = torch.tensor([f.input_mask for f in features_L], dtype=torch.long)
    all_segment_ids = torch.tensor([f.segment_ids for f in features_L], dtype=torch.long)
    all_label_ids = torch.tensor([f.label_ids for f in features_L], dtype=torch.long)
    all_label_mask = torch.tensor([f.label_mask for f in features_L], dtype=torch.long)

    dataset_L = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids, all_label_mask)


    all_input_ids = torch.tensor([f.input_ids for f in labeled_features], dtype=torch.long)
    all_input_mask = torch.tensor([f.input_mask for f in labeled_features], dtype=torch.long)
    all_segment_ids = torch.tensor([f.segment_ids for f in labeled_features], dtype=torch.long)
    all_label_ids = torch.tensor([f.label_ids for f in labeled_features], dtype=torch.long)
    all_label_mask = torch.tensor([f.label_mask for f in labeled_features], dtype=torch.long)


    dataset_TL =  TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids, all_label_mask)

    return dataset_L, dataset_TL

def tri_train(args, model_f1, model_f2, model_ft, source_features, target_features):

    Nt = args.N_init
    labeled_features = labelling(args, target_features, model_f1, model_f2, Nt)

    dataset_L, dataset_TL = prepare_dataset(source_features, labeled_features)

    model_ft = train_ft(args, model_ft, dataset_TL)

    return model_f1, model_f2, model_ft



def main():
    parser = argparse.ArgumentParser()

    ## Required parameters
    parser.add_argument("--data_dir",
                        default=None,
                        type=str,
                        required=True,
                        help="The input data dir. Should contain the .tsv files (or other data files) for the task.")
    parser.add_argument("--bert_model", default=None, type=str, required=True,
                        help="Bert pre-trained model selected in the list: bert-base-uncased, "
                        "bert-large-uncased, bert-base-cased, bert-large-cased, bert-base-multilingual-uncased, "
                        "bert-base-multilingual-cased, bert-base-chinese.")
    parser.add_argument("--output_dir",
                        default=None,
                        type=str,
                        required=True,
                        help="The output directory where the model predictions and checkpoints will be written.")

    ## Other parameters
    parser.add_argument("--cache_dir",
                        default="",
                        type=str,
                        help="Where do you want to store the pre-trained models downloaded from s3")
    
    parser.add_argument("--max_seq_length",
                        default=128,
                        type=int,
                        help="The maximum total input sequence length after WordPiece tokenization. \n"
                             "Sequences longer than this will be truncated, and sequences shorter \n"
                             "than this will be padded.")
    parser.add_argument("--do_train",
                        action='store_true',
                        help="Whether to run training.")
    parser.add_argument("--do_eval",
                        action='store_true',
                        help="Whether to run eval on the dev set.")
    parser.add_argument("--do_test",
                        action='store_true',
                        help="Whether to run eval on the test set.")
    parser.add_argument("--do_lower_case",
                        action='store_true',
                        help="Set this flag if you are using an uncased model.")
    parser.add_argument("--train_batch_size",
                        default=32,
                        type=int,
                        help="Total batch size for training.")
    parser.add_argument("--eval_batch_size",
                        default=8,
                        type=int,
                        help="Total batch size for eval.")
    parser.add_argument("--learning_rate",
                        default=5e-5,
                        type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--num_train_epochs",
                        default=3.0,
                        type=float,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--warmup_proportion",
                        default=0.1,
                        type=float,
                        help="Proportion of training to perform linear learning rate warmup for. "
                             "E.g., 0.1 = 10%% of training.")
    parser.add_argument("--no_cuda",
                        action='store_true',
                        help="Whether not to use CUDA when available")
    parser.add_argument("--local_rank",
                        type=int,
                        default=-1,
                        help="local_rank for distributed training on gpus")
    parser.add_argument('--seed',
                        type=int,
                        default=42,
                        help="random seed for initialization")
    parser.add_argument('--gradient_accumulation_steps',
                        type=int,
                        default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument('--fp16',
                        action='store_true',
                        help="Whether to use 16-bit float precision instead of 32-bit")
    parser.add_argument('--loss_scale',
                        type=float, default=0,
                        help="Loss scaling to improve fp16 numeric stability. Only used when fp16 set to True.\n"
                             "0 (default value): dynamic loss scaling.\n"
                             "Positive power of 2: static loss scaling value.\n")
    parser.add_argument('--freeze_bert',
                        action='store_true',
                        help="Whether to freeze BERT")
    parser.add_argument('--save_all_epochs',
                        action='store_true',
                        help="Whether to save model in each epoch")
    parser.add_argument('--coarse_tagset',
                        action='store_true',
                        help="Whether to save model in each epoch")
    parser.add_argument('--supervised_training',
                        action='store_true',
                        help="Only use this for supervised top-line model")
    parser.add_argument("--model_f1_path", default="", type=str)
    parser.add_argument("--model_f2_path", default="", type=str)
    parser.add_argument("--threshold", default=0.95, type=float)
    parser.add_argument("--N_init", default=100000, type=int)
    parser.add_argument("--joint_loss", action="store_true")
    parser.add_argument("--mini_batch_size", type=int, default=32)
    

    args = parser.parse_args()

    if args.local_rank == -1 or args.no_cuda:
        device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        n_gpu = torch.cuda.device_count()
    else:
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        n_gpu = 1
        # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.distributed.init_process_group(backend='nccl')
    logger.info("device: {} n_gpu: {}, distributed training: {}, 16-bits training: {}".format(
        device, n_gpu, bool(args.local_rank != -1), args.fp16))

    if args.gradient_accumulation_steps < 1:
        raise ValueError("Invalid gradient_accumulation_steps parameter: {}, should be >= 1".format(
                            args.gradient_accumulation_steps))

    args.train_batch_size = args.train_batch_size // args.gradient_accumulation_steps
    args.n_gpu = n_gpu
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)

    if not args.do_train and not args.do_eval and not args.do_test:
        raise ValueError("At least one of `do_train` or `do_eval` or `do_test` must be True.")

    if os.path.exists(args.output_dir) and os.listdir(args.output_dir) and args.do_train:
        #raise ValueError("Output directory ({}) already exists and is not empty.".format(args.output_dir))
        print("WARNING: Output directory already exists and is not empty.")
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    processor = DataProcessor()
    label_list = processor.get_labels(args.data_dir)
    num_labels = len(label_list)
    label_map = {label : i for i, label in enumerate(label_list)}
    tokenizer = BertTokenizer.from_pretrained(args.bert_model, do_lower_case=args.do_lower_case)

    train_examples = None
    num_train_optimization_steps = None

    args.device = device

    if args.do_train:
        f1_state_dict = torch.load(os.path.join(args.model_f1_path, WEIGHTS_NAME))
        f2_state_dict = torch.load(os.path.join(args.model_f2_path, WEIGHTS_NAME))

        model_f1 = MyBertForTokenClassification.from_pretrained(args.model_f1_path, state_dict = f1_state_dict, num_labels=num_labels)
        model_f2 = MyBertForTokenClassification.from_pretrained(args.model_f2_path, state_dict = f2_state_dict, num_labels=num_labels)
        model_ft = MyBertForTokenClassification.from_pretrained(args.model_f2_path, state_dict = f2_state_dict, num_labels=num_labels)

        source_examples = processor.get_conll_train_examples(args.data_dir)
        target_examples = processor.get_sep_twitter_train_examples(args.data_dir)
        target_examples.extend(processor.get_sep_twitter_test_examples(args.data_dir))

        source_features = convert_examples_to_features(source_examples,
        label_list, args.max_seq_length, tokenizer)

        target_features = convert_examples_to_features(target_examples,
        label_list, args.max_seq_length, tokenizer)

        args.source_features = source_features
        args.target_features = target_features
        logger.info("**** source examples: {} *******".format(len(source_features)))
        logger.info("****  target examples: {} *****".format(len(target_features)))

        model_f1.to(args.device)
        model_f2.to(args.device)
        model_ft.to(args.device)
        if args.joint_loss:
            tri_train_func = joint_tri_train
        else:
            tri_train_func = tri_train
        
        model_f1, model_f2, model_ft = tri_train_func(args, model_f1, model_f2,
        model_ft, source_features, target_features)

        model = model_ft

    if args.do_train and (args.local_rank == -1 or torch.distributed.get_rank() == 0):
        # Save a trained model and the associated configuration
        model_to_save = model.module if hasattr(model, 'module') else model  # Only save the model it-self
        output_model_file = os.path.join(args.output_dir, WEIGHTS_NAME)
        torch.save(model_to_save.state_dict(), output_model_file)
        output_config_file = os.path.join(args.output_dir, CONFIG_NAME)
        with open(output_config_file, 'w') as f:
            f.write(model_to_save.config.to_json_string())

    if args.do_eval and (args.local_rank == -1 or torch.distributed.get_rank() == 0):
        eval_examples = processor.get_conll_dev_examples(args.data_dir)
        eval_features = convert_examples_to_features(
            eval_examples, label_list, args.max_seq_length, tokenizer)
        logger.info("***** Running evaluation *****")
        logger.info("  Num examples = %d", len(eval_examples))
        logger.info("  Batch size = %d", args.eval_batch_size)
        all_input_ids = torch.tensor([f.input_ids for f in eval_features], dtype=torch.long)
        all_input_mask = torch.tensor([f.input_mask for f in eval_features], dtype=torch.long)
        all_segment_ids = torch.tensor([f.segment_ids for f in eval_features], dtype=torch.long)
        all_label_ids = torch.tensor([f.label_ids for f in eval_features], dtype=torch.long)
        all_label_mask = torch.tensor([f.label_mask for f in eval_features], dtype=torch.long)
        eval_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids, all_label_mask)
        # Run prediction for full data
        eval_sampler = SequentialSampler(eval_data)
        eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=args.eval_batch_size)

        model.eval()
        eval_loss, eval_accuracy = 0, 0
        nb_eval_steps, nb_eval_examples = 0, 0
        eval_TP, eval_FP, eval_FN = 0, 0, 0

        for input_ids, input_mask, segment_ids, label_ids, label_mask in tqdm(eval_dataloader, desc="Evaluating"):
            input_ids = input_ids.to(device)
            input_mask = input_mask.to(device)
            segment_ids = segment_ids.to(device)
            label_ids = label_ids.to(device)
            label_mask = label_mask.to(device)

            with torch.no_grad():
                tmp_eval_loss = model(input_ids, segment_ids, input_mask, label_ids, label_mask)
                logits = model(input_ids, segment_ids, input_mask)

            logits = logits.detach().cpu().numpy()
            label_ids = label_ids.to('cpu').numpy()
            label_mask = label_mask.to('cpu').numpy()

            tmp_eval_correct, tmp_eval_total = accuracy(logits, label_ids, label_mask)
            tplist = true_and_pred(logits, label_ids, label_mask)
            for trues, preds in tplist:
                TP, FP, FN = compute_tfpn(trues, preds, label_map)
                eval_TP += TP
                eval_FP += FP
                eval_FN += FN

            eval_loss += tmp_eval_loss.mean().item()
            eval_accuracy += tmp_eval_correct

            nb_eval_examples += tmp_eval_total
            nb_eval_steps += 1

        eval_loss = eval_loss / nb_eval_steps
        eval_accuracy = eval_accuracy / nb_eval_examples # micro average
        result = {'eval_loss': eval_loss,
                  'eval_accuracy': eval_accuracy,
                  'eval_f1': compute_f1(eval_TP, eval_FP, eval_FN)}

        output_eval_file = os.path.join(args.output_dir, "eval_results.txt")
        with open(output_eval_file, "w") as writer:
            logger.info("***** Eval results *****")
            for key in sorted(result.keys()):
                logger.info("  %s = %s", key, str(result[key]))
                writer.write("%s = %s\n" % (key, str(result[key])))

    if args.do_test and (args.local_rank == -1 or torch.distributed.get_rank() == 0):
        test_examples = processor.get_sep_twitter_test_examples(args.data_dir)
        test_features = convert_examples_to_features(
            test_examples, label_list, args.max_seq_length, tokenizer)
        logger.info("***** Running final test *****")
        logger.info("  Num examples = %d", len(test_examples))
        logger.info("  Batch size = %d", args.eval_batch_size)
        all_input_ids = torch.tensor([f.input_ids for f in test_features], dtype=torch.long)
        all_input_mask = torch.tensor([f.input_mask for f in test_features], dtype=torch.long)
        all_segment_ids = torch.tensor([f.segment_ids for f in test_features], dtype=torch.long)
        all_label_ids = torch.tensor([f.label_ids for f in test_features], dtype=torch.long)
        all_label_mask = torch.tensor([f.label_mask for f in test_features], dtype=torch.long)
        test_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids, all_label_mask)
        # Run prediction for full data
        test_sampler = SequentialSampler(test_data)
        test_dataloader = DataLoader(test_data, sampler=test_sampler, batch_size=args.eval_batch_size)

        model.eval()
        test_loss, test_accuracy = 0, 0
        nb_test_steps, nb_test_examples = 0, 0
        test_TP, test_FP, test_FN = 0, 0, 0

        for input_ids, input_mask, segment_ids, label_ids, label_mask in tqdm(test_dataloader, desc="Testing"):
            input_ids = input_ids.to(device)
            input_mask = input_mask.to(device)
            segment_ids = segment_ids.to(device)
            label_ids = label_ids.to(device)
            label_mask = label_mask.to(device)

            with torch.no_grad():
                tmp_test_loss = model(input_ids, segment_ids, input_mask, label_ids, label_mask)
                logits = model(input_ids, segment_ids, input_mask)

            logits = logits.detach().cpu().numpy()
            label_ids = label_ids.to('cpu').numpy()
            label_mask = label_mask.to('cpu').numpy()

            tmp_test_correct, tmp_test_total = accuracy(logits, label_ids, label_mask)
            tplist = true_and_pred(logits, label_ids, label_mask)
            for trues, preds in tplist:
                TP, FP, FN = compute_tfpn(trues, preds, label_map)
                test_TP += TP
                test_FP += FP
                test_FN += FN

            test_loss += tmp_test_loss.mean().item()
            test_accuracy += tmp_test_correct

            nb_test_examples += tmp_test_total
            nb_test_steps += 1

        test_loss = test_loss / nb_test_steps
        test_accuracy = test_accuracy / nb_test_examples # micro average
        result = {'test_loss': test_loss,
                  'test_accuracy': test_accuracy,
                  'test_f1': compute_f1(test_TP, test_FP, test_FN)}

        output_test_file = os.path.join(args.output_dir, "test_results.txt")
        with open(output_test_file, "w") as writer:
            logger.info("***** Test results *****")
            for key in sorted(result.keys()):
                logger.info("  %s = %s", key, str(result[key]))
                writer.write("%s = %s\n" % (key, str(result[key])))

if __name__ == "__main__":
    main()
